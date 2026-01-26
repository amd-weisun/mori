import torch
import triton
import triton.language as tl

@triton.jit
def _transform_kernel(
    src_ptr, dst_ptr,
    indices_ptr, expert_ids_ptr, slot_ids_ptr,
    stride_src_n, stride_src_h,
    stride_dst_e, stride_dst_c, stride_dst_h,
    num_tokens, 
    H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Load Metadata (Once per block of N)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < num_tokens
    
    src_idx = tl.load(indices_ptr + offs_n, mask=mask_n, other=0)
    expert_id = tl.load(expert_ids_ptr + offs_n, mask=mask_n, other=0)
    slot_id = tl.load(slot_ids_ptr + offs_n, mask=mask_n, other=0)
    
    # Pre-calculate base pointers for rows
    src_row_base = src_ptr + src_idx * stride_src_n
    dst_row_base = dst_ptr + expert_id * stride_dst_e + slot_id * stride_dst_c
    
    # 2. Loop over H
    for off_h in range(0, H, BLOCK_SIZE_H):
        offs_h = off_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = offs_h < H
        
        # Pointers for this chunk
        src_ptrs = src_row_base[:, None] + offs_h[None, :] * stride_src_h
        dst_ptrs = dst_row_base[:, None] + offs_h[None, :] * stride_dst_h
        
        # Load and Store
        vals = tl.load(src_ptrs, mask=mask_n[:, None] & mask_h[None, :], other=0.0)
        tl.store(dst_ptrs, vals, mask=mask_n[:, None] & mask_h[None, :])


@triton.jit
def _inverse_transform_kernel(
    src_ptr, dst_ptr,
    indices_ptr, expert_ids_ptr, slot_ids_ptr,
    stride_src_e, stride_src_c, stride_src_h,
    stride_dst_n, stride_dst_h,
    num_tokens, 
    H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Load Metadata (Once per block of N)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < num_tokens
    
    dst_idx = tl.load(indices_ptr + offs_n, mask=mask_n, other=0)
    expert_id = tl.load(expert_ids_ptr + offs_n, mask=mask_n, other=0)
    slot_id = tl.load(slot_ids_ptr + offs_n, mask=mask_n, other=0)
    
    # Pre-calculate base pointers for rows
    src_row_base = src_ptr + expert_id * stride_src_e + slot_id * stride_src_c
    dst_row_base = dst_ptr + dst_idx * stride_dst_n
    
    # 2. Loop over H
    for off_h in range(0, H, BLOCK_SIZE_H):
        offs_h = off_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = offs_h < H
        
        # Pointers for this chunk
        src_ptrs = src_row_base[:, None] + offs_h[None, :] * stride_src_h
        dst_ptrs = dst_row_base[:, None] + offs_h[None, :] * stride_dst_h
        
        # Load
        vals = tl.load(src_ptrs, mask=mask_n[:, None] & mask_h[None, :], other=0.0)
        
        # Atomic Add to Dst
        tl.atomic_add(dst_ptrs, vals, mask=mask_n[:, None] & mask_h[None, :])


def triton_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count):
    """
    Triton-accelerated version of transform_dispatch_output.
    """
    # 1. Metadata Preparation (PyTorch - Optimized)
    valid_tokens = dispatch_output[:recv_count]
    valid_indices = dispatch_indices[:recv_count]
    
    N_capacity = dispatch_output.size(0)
    _, H = valid_tokens.shape
    _, K = valid_indices.shape
    E = config.num_experts_per_rank
    
    flat_indices = valid_indices.view(-1)
    is_local = (flat_indices // E) == config.rank
    active_flat_indices = torch.nonzero(is_local).squeeze(-1)
    
    if active_flat_indices.numel() == 0:
            return (
                torch.zeros((E, N_capacity, H), device=dispatch_output.device, dtype=dispatch_output.dtype),
                torch.empty((0,), device=dispatch_output.device, dtype=torch.long),
                torch.zeros((E,), device=dispatch_output.device, dtype=torch.long)
            )

    token_indices = active_flat_indices.div(K, rounding_mode='floor')
    local_expert_ids = flat_indices[active_flat_indices] % E
    
    # Sort by expert ID
    sort_order = torch.argsort(local_expert_ids)
    sorted_token_indices = token_indices[sort_order]
    sorted_expert_ids = local_expert_ids[sort_order]
    
    # Calculate counts
    expert_counts = torch.bincount(sorted_expert_ids, minlength=E)
    
    # Optimized slot_indices generation (Vectorized)
    offsets = torch.zeros_like(expert_counts)
    offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)
    base_offsets = torch.repeat_interleave(offsets, expert_counts)
    slot_indices = torch.arange(sorted_expert_ids.size(0), device=dispatch_output.device) - base_offsets
    
    # 2. Data Movement (Triton)
    packed_output = torch.zeros((E, N_capacity, H), dtype=dispatch_output.dtype, device=dispatch_output.device)
    
    num_active = sorted_token_indices.size(0)
    
    grid = lambda META: (triton.cdiv(num_active, META['BLOCK_SIZE_N']),)
    
    _transform_kernel[grid](
        valid_tokens, packed_output,
        sorted_token_indices, sorted_expert_ids, slot_indices,
        valid_tokens.stride(0), valid_tokens.stride(1),
        packed_output.stride(0), packed_output.stride(1), packed_output.stride(2),
        num_active, H,
        BLOCK_SIZE_N=64, BLOCK_SIZE_H=512,
        num_stages=4, num_warps=4
    )
    
    return packed_output, sorted_token_indices, expert_counts

def triton_inverse_transform_dispatch_output(packed_output, original_indices, expert_counts, original_N):
    """
    Triton-accelerated version of inverse_transform_dispatch_output.
    """
    E, _, H = packed_output.shape
    device = packed_output.device
    
    # Reconstruct metadata
    offsets = torch.zeros_like(expert_counts)
    offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)
    base_offsets = torch.repeat_interleave(offsets, expert_counts)
    
    total_active = expert_counts.sum().item()
    slot_indices = torch.arange(total_active, device=device) - base_offsets
    expert_ids = torch.repeat_interleave(torch.arange(E, device=device), expert_counts)
    
    rec_output = torch.zeros((original_N, H), dtype=packed_output.dtype, device=device)
    
    grid = lambda META: (triton.cdiv(total_active, META['BLOCK_SIZE_N']),)
    
    _inverse_transform_kernel[grid](
        packed_output, rec_output,
        original_indices, expert_ids, slot_indices,
        packed_output.stride(0), packed_output.stride(1), packed_output.stride(2),
        rec_output.stride(0), rec_output.stride(1),
        total_active, H,
        BLOCK_SIZE_N=64, BLOCK_SIZE_H=512,
        num_stages=4, num_warps=4
    )
    
    return rec_output
