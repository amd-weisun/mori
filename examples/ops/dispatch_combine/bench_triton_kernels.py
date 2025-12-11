import torch
import triton
import time
import mori
from triton_kernels import triton_transform_dispatch_output, triton_inverse_transform_dispatch_output

class MockConfig:
    def __init__(self, num_experts_per_rank, rank):
        self.num_experts_per_rank = num_experts_per_rank
        self.rank = rank

def baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count):
    """
    Baseline PyTorch implementation.
    """
    # 1. Slice valid data
    valid_tokens = dispatch_output[:recv_count]   # [M, H]
    valid_indices = dispatch_indices[:recv_count] # [M, K]
    
    N_capacity = dispatch_output.size(0)
    _, H = valid_tokens.shape
    _, K = valid_indices.shape
    E = config.num_experts_per_rank
    
    # 2. Find which tokens go to which local expert
    flat_indices = valid_indices.view(-1) # [M*K]
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
    
    # 3. Sort by expert ID
    sort_order = torch.argsort(local_expert_ids)
    sorted_token_indices = token_indices[sort_order]
    sorted_expert_ids = local_expert_ids[sort_order]
    
    # 4. Calculate counts and pack
    expert_counts = torch.bincount(sorted_expert_ids, minlength=E)
    
    # Generate slot indices: [0, 1, ... c0-1, 0, 1, ... c1-1, ...]
    # This loop is the main CPU bottleneck
    slot_indices_list = [torch.arange(c, device=dispatch_output.device) for c in expert_counts]
    slot_indices = torch.cat(slot_indices_list)
    
    packed_output = torch.zeros((E, N_capacity, H), dtype=dispatch_output.dtype, device=dispatch_output.device)
    packed_output[sorted_expert_ids, slot_indices] = valid_tokens[sorted_token_indices]
    
    return packed_output, sorted_token_indices, expert_counts

def baseline_inverse_transform_dispatch_output(packed_output, original_indices, expert_counts, original_N):
    """
    Baseline PyTorch implementation.
    """
    E, _, H = packed_output.shape
    device = packed_output.device
    
    # Generate read indices matching the write order
    slot_indices_list = [torch.arange(c, device=device) for c in expert_counts]
    slot_indices = torch.cat(slot_indices_list)
    
    expert_ids = torch.repeat_interleave(torch.arange(E, device=device), expert_counts)
    
    # Extract valid tokens
    flat_values = packed_output[expert_ids, slot_indices]
    
    # Scatter add back
    rec_output = torch.zeros((original_N, H), dtype=packed_output.dtype, device=device)
    rec_output.index_add_(0, original_indices, flat_values)
    
    return rec_output

def run_benchmark():
    device = torch.device("cuda")
    
    # Benchmark Parameters
    configs = [
        (128*16, 7168, 288, 8),
    ]
    
    print(f"{'N':<10} {'H':<10} {'Operation':<20} {'Baseline (ms)':<15} {'Triton (ms)':<15} {'C++ GPU (ms)':<15} {'Speedup (Tri)':<15} {'Speedup (Cpp)':<15}")
    print("-" * 115)

    for N, H, E, K in configs:
        config = MockConfig(E, 0)
        
        # Generate Data
        dispatch_output = torch.randn(N, H, device=device, dtype=torch.bfloat16)
        # Random indices simulating routing
        dispatch_indices = torch.randint(0, E, (N, K), device=device, dtype=torch.int32)
        recv_count = N # Assume full buffer for stress test
        
        # Warmup & Correctness Check
        base_packed, base_idx, base_counts = baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
        tri_packed, tri_idx, tri_counts = triton_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
        cpp_packed, cpp_idx, cpp_counts = mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count)

        assert torch.allclose(base_packed, tri_packed), "Triton Transform Output Mismatch"
        assert torch.allclose(base_packed, cpp_packed), "C++ Transform Output Mismatch"
        
        # Benchmark Transform
        ms_base_trans = triton.testing.do_bench(lambda: baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count))
        ms_tri_trans = triton.testing.do_bench(lambda: triton_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count))
        ms_cpp_trans = triton.testing.do_bench(lambda: mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count))
        
        print(f"{N:<10} {H:<10} {'Transform':<20} {ms_base_trans:<15.4f} {ms_tri_trans:<15.4f} {ms_cpp_trans:<15.4f} {ms_base_trans/ms_tri_trans:<15.2f} {ms_base_trans/ms_cpp_trans:<15.2f}")
        
        # Benchmark Inverse
        # Use the packed output from transform step
        ms_base_inv = triton.testing.do_bench(lambda: baseline_inverse_transform_dispatch_output(base_packed, base_idx, base_counts, N))
        ms_tri_inv = triton.testing.do_bench(lambda: triton_inverse_transform_dispatch_output(tri_packed, tri_idx, tri_counts, N))
        ms_cpp_inv = triton.testing.do_bench(lambda: mori.inverse_transform_dispatch_output_gpu(cpp_packed, cpp_idx, cpp_counts, N))
        
        print(f"{N:<10} {H:<10} {'Inverse':<20} {ms_base_inv:<15.4f} {ms_tri_inv:<15.4f} {ms_cpp_inv:<15.4f} {ms_base_inv/ms_tri_inv:<15.2f} {ms_base_inv/ms_cpp_inv:<15.2f}")

if __name__ == "__main__":
    run_benchmark()
