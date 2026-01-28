# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""DeepEP low-latency dispatch/combine test.

Usage (single node with mp.spawn):
  # Run with default production settings (8 GPUs, intra-node LL):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py

  # Run debug setting (1 GPU, minimal config):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --setting debug

  # Simulate 2-node topology (8 GPUs as 2 nodes × 4 GPUs):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --setting internode_2node

  # Override gpu_per_node on any setting:
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --gpu-per-node 4

  # Run all preset settings:
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --all

Usage (multi-node with torchrun):
  # Node 0 (2 nodes, 8 GPUs per node):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
      --master_addr=${MASTER_ADDR} --master_port=29500 \
      tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode

  # Node 1:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
      --master_addr=${MASTER_ADDR} --master_port=29500 \
      tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode

  # Override gpu_per_node (defaults to nproc_per_node):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
      --master_addr=${MASTER_ADDR} --master_port=29500 \
      tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode --gpu-per-node 4

Usage (SLURM):
  srun -N 2 --ntasks-per-node=8 --gpus-per-node=8 \
      python tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

import mori
from utils import TorchDistContext, get_free_port

# Default to isolation mode for shmem if not explicitly set
if "MORI_SHMEM_MODE" not in os.environ:
    os.environ["MORI_SHMEM_MODE"] = "isolation"

SKIP_CHECKS = os.getenv("MORI_DEEPEP_SKIP_CHECKS", "0") == "1"
SKIP_DISPATCH_CHECKS = SKIP_CHECKS or os.getenv("MORI_DEEPEP_SKIP_DISPATCH_CHECKS", "0") == "1"
SKIP_COMBINE_CHECKS = SKIP_CHECKS or os.getenv("MORI_DEEPEP_SKIP_COMBINE_CHECKS", "0") == "1"
DEBUG_LOG = os.getenv("MORI_DEEPEP_DEBUG_LOG", "0") == "1"

# Global flags set by command-line args
DISPATCH_ONLY = False


PRESET_SETTINGS = {
    "debug": {
        "name": "debug",
        "num_processes": 1,
        "hidden_dim": 256,
        "max_num_inp_token_per_rank": 4,
        "total_experts": 4,
        "num_experts_per_token": 1,
        "gpu_per_node": 1,
        "use_fp8": True,
    },
    "small": {
        "name": "small",
        "num_processes": 8,
        "hidden_dim": 1024,
        "max_num_inp_token_per_rank": 32,
        "total_experts": 64,
        "num_experts_per_token": 4,
        "gpu_per_node": 8,
        "use_fp8": True,
    },
    "production": {
        "name": "production",
        "num_processes": 8,
        "hidden_dim": 7168,
        "max_num_inp_token_per_rank": 128,
        "total_experts": 288,
        "num_experts_per_token": 8,
        "gpu_per_node": 8,
        "use_fp8": True,
    },
    "internode_2node": {
        "name": "internode_2node",
        "num_processes": 8,
        "hidden_dim": 7168,
        "max_num_inp_token_per_rank": 128,
        "total_experts": 288,
        "num_experts_per_token": 8,
        "gpu_per_node": 4,  # Simulate 2 nodes with 4 GPUs each
        "use_fp8": True,
    },
    "internode_4node": {
        "name": "internode_4node",
        "num_processes": 8,
        "hidden_dim": 7168,
        "max_num_inp_token_per_rank": 128,
        "total_experts": 288,
        "num_experts_per_token": 8,
        "gpu_per_node": 2,  # Simulate 4 nodes with 2 GPUs each
        "use_fp8": True,
    },
}


def _log(msg: str, force: bool = False):
    if force or DEBUG_LOG:
        print(msg, flush=True)


def dequant_dispatch_output(dispatch_output, dispatch_scales, hidden_dim):
    assert dispatch_scales is not None
    num_scales = hidden_dim // 128
    e, c, _ = dispatch_output.shape
    output_fp32 = dispatch_output.float().view(e, c, num_scales, 128)
    scales_fp32 = dispatch_scales.float().view(e, c, num_scales, 1)
    return (output_fp32 * scales_fp32).view(e, c, hidden_dim).to(torch.bfloat16)


def dequant_input_like_fp8(inputs: torch.Tensor, data_type: torch.dtype) -> torch.Tensor:
    assert inputs.dim() == 2 and inputs.size(1) % 128 == 0
    num_scales = inputs.size(1) // 128
    x_view = inputs.float().view(inputs.size(0), num_scales, 128)
    amax = x_view.abs().amax(dim=2).clamp(1e-4)
    k_fp8_amax = 240.0 if torch.version.hip is not None else 448.0
    scale = k_fp8_amax / amax
    scale_inv = amax / k_fp8_amax
    try:
        q = (x_view * scale.unsqueeze(-1)).to(torch.float8_e4m3fnuz)
    except AttributeError:
        q = (x_view * scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    dequant = q.float() * scale_inv.unsqueeze(-1)
    return dequant.view(inputs.size(0), inputs.size(1)).to(data_type)


def run_test_worker(
    local_rank: int,
    num_local_ranks: int,
    setting: dict,
    port: int,
    gpu_per_node_override: int | None = None,
):
    """Worker function for mp.spawn mode (single node)."""
    rank = local_rank
    world_size = num_local_ranks

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        _log(f"[Rank {rank}] Initializing shmem...", force=True)
        mori.shmem.shmem_torch_process_group_init("default")

        try:
            run_test_impl(rank, world_size, setting, gpu_per_node_override)
        finally:
            mori.shmem.shmem_finalize()


def run_test_multinode(setting: dict, iterations: int = 1):
    """Entry point for torchrun/srun mode (multi-node)."""
    # Get local rank from environment (set by torchrun/srun) before init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_per_node = setting["gpu_per_node"]

    # Set CUDA device before init_process_group
    torch.cuda.set_device(local_rank)

    # Initialize process group with gloo backend (required for shmem CPU operations)
    # NCCL only supports CUDA devices, but shmem needs CPU backend support
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"[DeepEP MultiNode] world_size={world_size}, gpu_per_node={gpu_per_node}", flush=True)
        print(f"[DeepEP MultiNode] num_nodes={world_size // gpu_per_node}", flush=True)

    # Register process group with name "default" for MORI shmem
    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)

    mori.shmem.shmem_torch_process_group_init("default")

    try:
        for iteration in range(iterations):
            if iterations > 1 and rank == 0:
                print(f"\n[DeepEP] Iteration {iteration + 1}/{iterations}", flush=True)
            run_test_impl(rank, world_size, setting, gpu_per_node)
            dist.barrier()  # Sync between iterations
    finally:
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[DeepEP MultiNode] All {iterations} iteration(s) passed!", flush=True)


def run_test_impl(
    rank: int,
    world_size: int,
    setting: dict,
    gpu_per_node_override: int | None = None,
):
    """Core test implementation shared by both single-node and multi-node modes."""
    hidden_dim = setting["hidden_dim"]
    max_num_inp_token_per_rank = setting["max_num_inp_token_per_rank"]
    total_experts = setting["total_experts"]
    num_experts_per_token = setting["num_experts_per_token"]
    gpu_per_node = gpu_per_node_override or setting.get("gpu_per_node", world_size)
    use_fp8 = setting.get("use_fp8", hidden_dim >= 128)
    data_type = torch.bfloat16

    assert total_experts % world_size == 0
    num_experts_per_rank = total_experts // world_size

    # Determine kernel type based on topology
    is_internode = world_size > gpu_per_node
    kernel_type = (
        mori.ops.EpDispatchCombineDeepepKernelType.InterNodeLL
        if is_internode
        else mori.ops.EpDispatchCombineDeepepKernelType.IntraNode
    )

    if rank == 0:
        num_nodes = (world_size + gpu_per_node - 1) // gpu_per_node
        print(
            f"[DeepEP] Running setting '{setting['name']}': "
            f"world_size={world_size}, gpu_per_node={gpu_per_node}, num_nodes={num_nodes}, "
            f"hidden_dim={hidden_dim}, experts={total_experts}, topk={num_experts_per_token}, "
            f"use_fp8={use_fp8}, kernel_type={kernel_type}",
            flush=True,
        )

    device = torch.device("cuda", rank % gpu_per_node)
    rng = torch.Generator(device=device)
    rng.manual_seed(123)

    # Create config
    config = mori.ops.EpDispatchCombineDeepepConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=hidden_dim // 128,
        scale_type_size=4,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=4,
        block_num=40,
        warp_num_per_block=8,
        use_external_inp_buf=True,
        use_fp8=use_fp8,
        use_deepep_layout=True,
        use_weighted_combine=True,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
    )
    op = mori.ops.EpDispatchCombineDeepepOp(config)

    # Generate test data (same RNG seed across all ranks for reproducibility)
    num_token = torch.tensor(
        [max_num_inp_token_per_rank for _ in range(world_size)]
    ).to(device)

    all_rank_indices = []
    for r in range(world_size):
        indices = torch.empty(num_token[r], num_experts_per_token, dtype=torch.int64)
        for i in range(num_token[r]):
            perm = torch.randperm(total_experts, generator=rng, device=device)
            indices[i] = perm[:num_experts_per_token]
        all_rank_indices.append(indices.to(torch.int32).to(device))

    all_rank_weights = [
        torch.ones(num_token[r], num_experts_per_token, dtype=torch.float32, device=device)
        for r in range(world_size)
    ]

    all_rank_scales = [
        torch.rand(num_token[r], hidden_dim // 128, dtype=torch.float32, generator=rng, device=device)
        for r in range(world_size)
    ]

    all_rank_input = [
        (torch.rand(num_token[r], hidden_dim, dtype=torch.float32, generator=rng, device=device) * 2 - 1).to(data_type)
        for r in range(world_size)
    ]

    # Ensure all ranks finish setup before dispatch
    dist.barrier()
    _log(f"[Rank {rank}] Starting dispatch...", force=True)

    # Select dispatch/combine functions based on kernel type
    dispatch_func = op.dispatch_internode_deepep_ll if is_internode else op.dispatch_deepep_ll
    combine_func = op.combine_internode_deepep_ll if is_internode else op.combine_deepep_ll

    # Dispatch
    recv_x, recv_count, handle, _, _ = dispatch_func(
        all_rank_input[rank],
        all_rank_indices[rank],
        num_max_dispatch_tokens_per_rank=max_num_inp_token_per_rank,
        num_experts=total_experts,
        use_fp8=use_fp8,
        weights=all_rank_weights[rank],
        scales=all_rank_scales[rank],
    )

    if use_fp8:
        dispatch_output, dispatch_scales = recv_x
    else:
        dispatch_output, dispatch_scales = recv_x, None

    dispatch_weights, dispatch_indices = handle
    dispatch_recv_num_token = recv_count.sum().to(torch.int32)

    torch.cuda.synchronize()
    dist.barrier()
    _log(f"[Rank {rank}] Dispatch complete. Total tokens received: {dispatch_recv_num_token.item()}", force=True)

    # Validate dispatch result
    if not SKIP_DISPATCH_CHECKS:
        validate_dispatch(
            rank, config, recv_count, all_rank_indices
        )

    # Skip combine phase if dispatch-only mode
    if DISPATCH_ONLY:
        op.reset()
        if rank == 0:
            print(f"[DeepEP] Dispatch-only test '{setting['name']}' PASSED", flush=True)
        return

    # Combine
    _log(f"[Rank {rank}] Starting combine...", force=True)
    combine_input = dispatch_output
    if use_fp8:
        combine_input = dequant_dispatch_output(dispatch_output, dispatch_scales, hidden_dim)

    combine_output, _, _ = combine_func(
        combine_input, dispatch_indices, dispatch_weights, handle=handle
    )

    torch.cuda.synchronize()
    dist.barrier()
    _log(f"[Rank {rank}] Combine complete.", force=True)

    # Validate combine result
    if not SKIP_COMBINE_CHECKS and rank == 0:
        validate_combine(
            config, combine_output, all_rank_input, all_rank_weights, use_fp8, data_type
        )

    op.reset()

    if rank == 0:
        print(f"[DeepEP] Test '{setting['name']}' PASSED", flush=True)


def validate_dispatch(rank, config, recv_count, all_rank_indices):
    """Validate dispatch results by checking token counts."""
    expected_from_indices = 0
    rank_begin = rank * config.num_experts_per_rank
    rank_end = rank_begin + config.num_experts_per_rank
    for r in range(config.world_size):
        idx = all_rank_indices[r]
        expected_from_indices += int(((idx >= rank_begin) & (idx < rank_end)).sum().item())

    actual_from_counts = int(recv_count.sum().item())

    if actual_from_counts != expected_from_indices:
        raise AssertionError(
            f"Rank {rank}: dispatch count mismatch. "
            f"Expected {expected_from_indices}, got {actual_from_counts}"
        )
    _log(f"[Rank {rank}] Dispatch validation passed: {actual_from_counts} tokens", force=True)


def validate_combine(config, combine_output, all_rank_input, all_rank_weights, use_fp8, data_type):
    """Validate combine results (rank 0 only, samples first few tokens)."""
    num_tokens_to_check = min(100, config.max_num_inp_token_per_rank)
    base_input = all_rank_input[0]
    if use_fp8:
        base_input = dequant_input_like_fp8(base_input, data_type)

    for i in range(num_tokens_to_check):
        got = combine_output[i]
        weights = all_rank_weights[0][i].to(torch.float32)
        expected = torch.zeros_like(got)
        for k in range(weights.numel()):
            expected = (expected + (base_input[i].to(torch.float32) * weights[k])).to(data_type)

        atol = 0.25 if use_fp8 else 1e-2
        rtol = 0.25 if use_fp8 else 1e-2
        if not torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol):
            raise AssertionError(
                f"Combine mismatch at token {i}: "
                f"got={got[:8].tolist()}, expected={expected[:8].tolist()}"
            )

    _log("[Rank 0] Combine validation passed", force=True)


def main():
    parser = argparse.ArgumentParser(description="DeepEP low-latency dispatch/combine test")
    parser.add_argument(
        "--setting",
        choices=list(PRESET_SETTINGS.keys()),
        default="production",
        help="Test setting to run (default: production)",
    )
    parser.add_argument(
        "--gpu-per-node",
        type=int,
        default=None,
        help="Override gpu_per_node to simulate multi-node topology",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Override number of processes (GPUs) to use",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all preset settings",
    )
    parser.add_argument(
        "--multinode",
        action="store_true",
        help="Run in multi-node mode (use with torchrun/srun)",
    )
    parser.add_argument(
        "--dispatch-only",
        action="store_true",
        help="Run dispatch only (skip combine phase) for debugging",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (for stress testing)",
    )
    args = parser.parse_args()

    # Set global flags
    global DISPATCH_ONLY
    DISPATCH_ONLY = args.dispatch_only

    # Multi-node mode: use torchrun/srun for process management
    if args.multinode:
        setting = PRESET_SETTINGS[args.setting].copy()
        if args.gpu_per_node:
            # Explicit override via --gpu-per-node
            setting["gpu_per_node"] = args.gpu_per_node
        else:
            # Default to LOCAL_WORLD_SIZE (nproc_per_node from torchrun)
            setting["gpu_per_node"] = int(os.environ.get("LOCAL_WORLD_SIZE", setting["gpu_per_node"]))
        run_test_multinode(setting, iterations=args.iterations)
        return

    # Single-node mode: use mp.spawn for process management
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    settings_to_run = list(PRESET_SETTINGS.values()) if args.all else [PRESET_SETTINGS[args.setting]]

    for setting in settings_to_run:
        num_processes = args.num_processes or setting["num_processes"]
        gpu_per_node = args.gpu_per_node

        print("=" * 80, flush=True)
        print(f"[DeepEP] Running setting '{setting['name']}' with {num_processes} processes", flush=True)
        print("=" * 80, flush=True)

        port = get_free_port()
        mp.spawn(
            run_test_worker,
            args=(num_processes, setting, port, gpu_per_node),
            nprocs=num_processes,
        )

        print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
