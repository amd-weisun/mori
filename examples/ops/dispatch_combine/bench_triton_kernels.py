import torch
import triton
import time
import argparse
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
        (128*8, 7168, 288, 8),
        (128*4, 7168, 288, 8),
        (128*2, 7168, 288, 8),
        (128*1, 7168, 288, 8),
    ]
    
    print(f"{'N':<10} {'H':<10} {'Operation':<20} {'Baseline (ms)':<15} {'Triton (ms)':<15} {'HIP GPU (ms)':<15} {'Speedup (Tri)':<15} {'Speedup (Cpp)':<15}")
    print("-" * 115)

    for N, H, E, K in configs:
        config = MockConfig(E, 0)
        
        # Generate Data
        dispatch_output = torch.randn(N, H, device=device, dtype=torch.bfloat16)
        # Random indices simulating routing
        dispatch_indices = torch.randint(0, E, (N, K), device=device, dtype=torch.int32)
        recv_count = N # Assume full buffer for stress test
        
        # Warmup & Correctness Check (compute one variant at a time to reduce peak memory)
        base_packed, base_idx, base_counts = baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
        cpp_packed, cpp_idx, cpp_counts = mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count)

        # Check counts match
        assert torch.allclose(base_counts, cpp_counts.to(base_counts.dtype)), "HIP Expert Counts Mismatch"

        # Check Inverse Reconstruction (End-to-End Correctness)
        # This is robust to reordering within experts
        base_rec = baseline_inverse_transform_dispatch_output(base_packed, base_idx, base_counts, N)
        cpp_rec = mori.inverse_transform_dispatch_output_gpu(cpp_packed, cpp_idx, cpp_counts, N)
        
        # Note: We only check reconstruction on the valid tokens if N != recv_count, 
        # but here recv_count=N so we check everything.
        # However, inverse_transform accumulates. If dispatch_indices has duplicates, it sums them.
        # The baseline and CPP should produce identical sums.
        
        # For strict packed output comparison, we would need to sort.
        # But since the C++ kernel uses atomic adds, the order within an expert is non-deterministic.
        # So we skip strict packed comparison and rely on inverse reconstruction or sorted comparison.
        
        def sort_packed(packed, counts):
            # Helper to sort packed output for comparison
            E, C, H = packed.shape
            sorted_packed = torch.zeros_like(packed)
            for e in range(E):
                c = counts[e].item()
                if c > 0:
                    # Sort by the first element of the hidden dim as a proxy, or full sort
                    # Full sort of (C, H) rows is hard in torch.
                    # Let's just sort by the first column for basic check
                    rows = packed[e, :c, :]
                    sort_idx = torch.argsort(rows[:, 0])
                    sorted_packed[e, :c, :] = rows[sort_idx]
            return sorted_packed

        # assert torch.allclose(base_packed, tri_packed), "Triton Transform Output Mismatch"
        # assert torch.allclose(sort_packed(base_packed, base_counts), sort_packed(cpp_packed, cpp_counts)), "HIP Transform Output Mismatch (Sorted)"
        
        # Stronger check: Inverse should match baseline inverse
        # Note: baseline_inverse_transform_dispatch_output accumulates.
        # If we feed the same inputs (logically), we should get same outputs.
        assert torch.allclose(base_rec, cpp_rec, atol=1e-2, rtol=1e-2), "HIP Inverse/Reconstruction Mismatch"
        
        # Free large intermediates before benchmarking to keep peak memory low
        del base_packed, base_idx, base_counts
        del cpp_packed, cpp_idx, cpp_counts
        torch.cuda.empty_cache()
        
        # Benchmark Transform
        ms_base_trans = triton.testing.do_bench(lambda: baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count))
        # Disable Triton transform in benchmark due to memory fault at large N
        ms_tri_trans = float('nan')
        ms_cpp_trans = triton.testing.do_bench(lambda: mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count))
        
        tri_speedup = 'NA'
        try:
            tri_speedup = f"{ms_base_trans/ms_tri_trans:<15.2f}"
        except Exception:
            tri_speedup = f"{'NA':<15}"
        print(f"{N:<10} {H:<10} {'Transform':<20} {ms_base_trans:<15.4f} {ms_tri_trans:<15} {ms_cpp_trans:<15.4f} {tri_speedup} {ms_base_trans/ms_cpp_trans:<15.2f}")
        
        # Benchmark Inverse
        # Recompute packed outputs per implementation to avoid holding all variants concurrently
        # Baseline inverse
        _base_packed, _base_idx, _base_counts = baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
        ms_base_inv = triton.testing.do_bench(lambda: baseline_inverse_transform_dispatch_output(_base_packed, _base_idx, _base_counts, N))
        del _base_packed, _base_idx, _base_counts
        torch.cuda.empty_cache()

        # Triton inverse
        # Disable Triton inverse in benchmark due to memory fault at large N
        ms_tri_inv = float('nan')

        # C++ inverse
        _cpp_packed, _cpp_idx, _cpp_counts = mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count)
        ms_cpp_inv = triton.testing.do_bench(lambda: mori.inverse_transform_dispatch_output_gpu(_cpp_packed, _cpp_idx, _cpp_counts, N))
        del _cpp_packed, _cpp_idx, _cpp_counts
        torch.cuda.empty_cache()
        
        tri_inv_speedup = 'NA'
        try:
            tri_inv_speedup = f"{ms_base_inv/ms_tri_inv:<15.2f}"
        except Exception:
            tri_inv_speedup = f"{'NA':<15}"
        print(f"{N:<10} {H:<10} {'Inverse':<20} {ms_base_inv:<15.4f} {ms_tri_inv:<15} {ms_cpp_inv:<15.4f} {tri_inv_speedup} {ms_base_inv/ms_cpp_inv:<15.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark or single-run dispatch combine kernels")
    parser.add_argument("--once", action="store_true", help="Run a single kernel call for debugging")
    parser.add_argument("--op", choices=["transform", "inverse"], default="transform", help="Operation to run once when using --once")
    parser.add_argument("--impl", choices=["baseline", "triton", "cpp"], default="cpp", help="Implementation to run once when using --once")
    parser.add_argument("--N", type=int, default=128*8, help="N (capacity) size")
    parser.add_argument("--H", type=int, default=7168, help="Hidden size H")
    parser.add_argument("--E", type=int, default=288, help="Experts per rank E")
    parser.add_argument("--K", type=int, default=8, help="Top-K routing choices")
    parser.add_argument("--mem", action="store_true", help="Print GPU memory stats before/after the single run")
    args = parser.parse_args()

    if not args.once:
        run_benchmark()
    else:
        device = torch.device("cuda")
        N, H, E, K = args.N, args.H, args.E, args.K
        config = MockConfig(E, 0)
        dispatch_output = torch.randn(N, H, device=device, dtype=torch.bfloat16)
        dispatch_indices = torch.randint(0, E, (N, K), device=device, dtype=torch.int32)
        recv_count = N

        def print_mem(tag):
            if not args.mem:
                return
            stats = {
                'allocated': torch.cuda.memory_allocated(device),
                'reserved': torch.cuda.memory_reserved(device),
                'max_allocated': torch.cuda.max_memory_allocated(device),
                'max_reserved': torch.cuda.max_memory_reserved(device),
            }
            print(f"[MEM {tag}] alloc={stats['allocated']}, reserv={stats['reserved']}, max_alloc={stats['max_allocated']}, max_reserv={stats['max_reserved']}")

        print_mem("before")

        if args.op == "transform":
            if args.impl == "baseline":
                _ = baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
            elif args.impl == "triton":
                _ = triton_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
            else:  # cpp
                _ = mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count)
            torch.cuda.synchronize()
            print_mem("after")
            print(f"Ran single transform: impl={args.impl}, shape=({N},{H}), E={E}, K={K}")
        else:  # inverse
            # Need packed outputs from chosen implementation first
            if args.impl == "baseline":
                packed, idx, counts = baseline_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
                _ = baseline_inverse_transform_dispatch_output(packed, idx, counts, N)
            elif args.impl == "triton":
                packed, idx, counts = triton_transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count)
                _ = triton_inverse_transform_dispatch_output(packed, idx, counts, N)
            else:  # cpp
                packed, idx, counts = mori.transform_dispatch_output_gpu(dispatch_output, dispatch_indices, config, recv_count)
                _ = mori.inverse_transform_dispatch_output_gpu(packed, idx, counts, N)
            torch.cuda.synchronize()
            print_mem("after")
            print(f"Ran single inverse: impl={args.impl}, shape=({N},{H}), E={E}, K={K}")
