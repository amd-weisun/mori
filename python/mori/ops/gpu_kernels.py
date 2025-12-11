import torch
import mori

def transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count):
    """
    GPU-accelerated version of transform_dispatch_output using C++ kernels.
    """
    # 1. Metadata Preparation (PyTorch)
    # This part remains in Python/PyTorch as it involves complex indexing logic
    # that is efficient enough in PyTorch or hard to port to a single kernel.
    
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
    
    # 2. Data Movement (C++ GPU Kernel)
    packed_output = torch.zeros((E, N_capacity, H), dtype=dispatch_output.dtype, device=dispatch_output.device)
    
    # Call the C++ binding
    # Note: We pass the full tensors. The kernel handles the indexing.
    # Ensure indices are int32 or int64 as expected by C++ (usually int64/long in PyTorch maps to index_t)
    mori.transform_dispatch_output_gpu(
        valid_tokens, 
        packed_output,
        sorted_token_indices.to(torch.int64), 
        sorted_expert_ids.to(torch.int64), 
        slot_indices.to(torch.int64)
    )
    
    return packed_output, sorted_token_indices, expert_counts

def inverse_transform_dispatch_output_gpu(packed_output, original_indices, expert_counts, original_N):
    """
    GPU-accelerated version of inverse_transform_dispatch_output using C++ kernels.
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
    
    # Call the C++ binding
    mori.inverse_transform_dispatch_output_gpu(
        packed_output,
        rec_output,
        original_indices.to(torch.int64),
        expert_ids.to(torch.int64),
        slot_indices.to(torch.int64)
    )
    
    return rec_output
