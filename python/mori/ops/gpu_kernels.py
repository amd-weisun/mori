import torch
import mori

@torch.compile
def _prepare_inverse_metadata(expert_counts, device):
    E = expert_counts.size(0)
    offsets = torch.zeros_like(expert_counts)
    offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)
    base_offsets = torch.repeat_interleave(offsets, expert_counts)
    
    # Use base_offsets size to determine total active elements, avoiding .item() sync
    total_active = base_offsets.size(0)
    slot_indices = torch.arange(total_active, device=device) - base_offsets
    expert_ids = torch.repeat_interleave(torch.arange(E, device=device), expert_counts)
    
    return slot_indices, expert_ids

def transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count, dispatch_scales=None):
    """
    GPU-accelerated version of transform_dispatch_output using C++ kernels.
    """
    # 1. Metadata Preparation (C++ Kernel)
    # This is now fully fused and asynchronous
    (
        sorted_token_indices, 
        sorted_expert_ids, 
        slot_indices, 
        expert_counts, 
        total_valid_count
    ) = mori.cpp.prepare_transform_metadata_gpu(
        dispatch_indices[:recv_count],
        config.num_experts_per_rank,
        config.rank
    )
    
    N_capacity = dispatch_output.size(0)
    H = dispatch_output.size(1)
    E = config.num_experts_per_rank
    
    # 2. Data Movement (C++ GPU Kernel)
    packed_output = torch.zeros((E, N_capacity, H), dtype=dispatch_output.dtype, device=dispatch_output.device)
    valid_tokens = dispatch_output[:recv_count]
    
    packed_scales = None
    if dispatch_scales is not None:
        scale_dim = dispatch_scales.size(1)
        packed_scales = torch.zeros((E, N_capacity, scale_dim), dtype=dispatch_scales.dtype, device=dispatch_scales.device)
        dispatch_scales = dispatch_scales[:recv_count]

    # Call the C++ binding
    # We pass total_valid_count to allow the kernel to exit early if needed,
    # although the kernel launch grid is based on max possible tokens.
    # The kernel will check *total_valid_count if provided.
    mori.cpp.transform_dispatch_output_gpu(
        valid_tokens, 
        packed_output,
        sorted_token_indices, 
        sorted_expert_ids, 
        slot_indices,
        total_valid_count,
        dispatch_scales,
        packed_scales
    )
    
    valid_count = total_valid_count.item()
    return packed_output, sorted_token_indices[:valid_count], expert_counts, packed_scales

def inverse_transform_dispatch_output_gpu(packed_output, original_indices, expert_counts, original_N):
    """
    GPU-accelerated version of inverse_transform_dispatch_output using C++ kernels.
    """
    E, _, H = packed_output.shape
    device = packed_output.device
    
    # Reconstruct metadata (Compiled)
    slot_indices, expert_ids = _prepare_inverse_metadata(expert_counts, device)
    
    rec_output = torch.zeros((original_N, H), dtype=packed_output.dtype, device=device)
    
    # Call the C++ binding
    mori.cpp.inverse_transform_dispatch_output_gpu(
        packed_output,
        rec_output,
        original_indices.to(torch.int32),
        expert_ids.to(torch.int32),
        slot_indices.to(torch.int32)
    )
    
    return rec_output
