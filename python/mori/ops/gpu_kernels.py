import torch
import mori

# Helper functions for torch.compile to fuse metadata operations
@torch.compile
def _prepare_transform_metadata(dispatch_indices, recv_count, num_experts_per_rank, rank, device):
    valid_indices = dispatch_indices[:recv_count]
    _, K = valid_indices.shape
    
    flat_indices = valid_indices.view(-1)
    is_local = (flat_indices // num_experts_per_rank) == rank
    active_flat_indices = torch.nonzero(is_local).squeeze(-1)
    
    token_indices = active_flat_indices.div(K, rounding_mode='floor')
    local_expert_ids = flat_indices[active_flat_indices] % num_experts_per_rank
    
    # Sort by expert ID
    sort_order = torch.argsort(local_expert_ids)
    sorted_token_indices = token_indices[sort_order]
    sorted_expert_ids = local_expert_ids[sort_order]
    
    # Calculate counts
    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts_per_rank)
    
    # Optimized slot_indices generation
    offsets = torch.zeros_like(expert_counts)
    offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)
    base_offsets = torch.repeat_interleave(offsets, expert_counts)
    slot_indices = torch.arange(sorted_expert_ids.size(0), device=device) - base_offsets
    
    return sorted_token_indices, sorted_expert_ids, slot_indices, expert_counts

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

def transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count):
    """
    GPU-accelerated version of transform_dispatch_output using C++ kernels.
    """
    # 1. Metadata Preparation (Compiled)
    sorted_token_indices, sorted_expert_ids, slot_indices, expert_counts = _prepare_transform_metadata(
        dispatch_indices, 
        recv_count, 
        config.num_experts_per_rank, 
        config.rank, 
        dispatch_output.device
    )
    
    N_capacity = dispatch_output.size(0)
    H = dispatch_output.size(1)
    E = config.num_experts_per_rank
    
    # 2. Data Movement (C++ GPU Kernel)
    packed_output = torch.zeros((E, N_capacity, H), dtype=dispatch_output.dtype, device=dispatch_output.device)
    valid_tokens = dispatch_output[:recv_count]
    
    # Call the C++ binding
    # Note: We pass the full tensors. The kernel handles the indexing.
    # Ensure indices are int32 as expected by C++ (index_t is int32_t)
    mori.cpp.transform_dispatch_output_gpu(
        valid_tokens, 
        packed_output,
        sorted_token_indices.to(torch.int32), 
        sorted_expert_ids.to(torch.int32), 
        slot_indices.to(torch.int32)
    )
    
    return packed_output, sorted_token_indices, expert_counts

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
