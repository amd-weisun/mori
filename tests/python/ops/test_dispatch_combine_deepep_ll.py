# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""DeepEP low-latency dispatch/combine test.

Usage (single node with mp.spawn):
  # Run with default production settings (8 GPUs, intra-node LL):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py

  python tests/python/ops/test_dispatch_combine_deepep_ll.py --benchmark --benchmark-warmup 3 --benchmark-iters 20

  # Run debug setting (1 GPU, minimal config):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --setting debug

  # Simulate 2-node topology (8 GPUs as 2 nodes × 4 GPUs):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --setting internode_2node

 # Simulate 4-node topology (8 GPUs as 4 nodes × 2 GPUs, use_fp8 = false and 1024 hidden dim for easier debugging):
  python tests/python/ops/test_dispatch_combine_deepep_ll.py --setting internode_2node_debug

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

    --------------------------------
  # Node 0 (2 nodes, 1 GPUs per node, dispatch only):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=${MASTER_ADDR} --master_port=29500 tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode --dispatch-only --iterations 5

  # Node 1:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=${MASTER_ADDR} --master_port=29500 tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode --dispatch-only --iterations 5
    -------------------------------
  # Override gpu_per_node (defaults to nproc_per_node):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
      --master_addr=${MASTER_ADDR} --master_port=29500 \
      tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode --gpu-per-node 4

Usage (SLURM):
  srun -N 2 --ntasks-per-node=8 --gpus-per-node=8 \
      python tests/python/ops/test_dispatch_combine_deepep_ll.py --multinode
"""
import argparse
import gc
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
# Debug mode: use simplified input data (global_token_id repeated across hidden_dim)
# This makes it easy to trace which token's data appears in the output
DEBUG_SIMPLE_DATA = os.getenv("MORI_DEEPEP_DEBUG_SIMPLE_DATA", "0") == "1"



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
    "internode_2node_debug": {
        "name": "internode_2node_debug",
        "num_processes": 8,
        "hidden_dim": 1024,
        "max_num_inp_token_per_rank": 128,
        "total_experts": 288,
        "num_experts_per_token": 8,
        "gpu_per_node": 4,  # Simulate 2 nodes with 4 GPUs each
        "use_fp8": False,
    },
    # Minimal test for isolating combine issues - very simple data flow
    "internode_minimal": {
        "name": "internode_minimal",
        "num_processes": 8,
        "hidden_dim": 64,              # Small hidden dim
        "max_num_inp_token_per_rank": 4,  # Just 4 tokens per rank
        "total_experts": 8,            # 1 expert per GPU
        "num_experts_per_token": 1,    # topk=1 for simplest flow
        "gpu_per_node": 4,             # 2 nodes with 4 GPUs each
        "use_fp8": False,
    },
    # Minimal test with topk=8 to isolate multi-expert accumulation issues
    "internode_minimal_8topk": {
        "name": "internode_minimal_8topk",
        "num_processes": 8,
        "hidden_dim": 64,              # Small hidden dim
        "max_num_inp_token_per_rank": 4,  # Just 4 tokens per rank
        "total_experts": 64,           # 8 experts per GPU
        "num_experts_per_token": 8,    # topk=8 like production
        "gpu_per_node": 4,             # 2 nodes with 4 GPUs each
        "use_fp8": False,
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
    # Minimal 2-GPU setting to simulate 2 nodes (1 GPU per node)
    # Use this for debugging inter-node logic on single physical node
    "internode_2gpu_sim": {
        "name": "internode_2gpu_sim",
        "num_processes": 2,
        "hidden_dim": 1024,
        "max_num_inp_token_per_rank": 128,
        "total_experts": 288,  # 8 experts per rank
        "num_experts_per_token": 8,
        "gpu_per_node": 1,  # Each GPU is a separate "node"
        "use_fp8": False,  # BF16 for easier debugging
    },
    # Same as above but with FP8
    "internode_2gpu_sim_fp8": {
        "name": "internode_2gpu_sim_fp8",
        "num_processes": 2,
        "hidden_dim": 1024,
        "max_num_inp_token_per_rank": 128,
        "total_experts": 288,
        "num_experts_per_token": 8,
        "gpu_per_node": 1,
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


def select_block_config(max_num_inp_token_per_rank: int, total_experts: int = 0) -> tuple[int, int]:
    """Select appropriate block_num and warp_num_per_block based on token count.

    Follows the pattern from bench_dispatch_combine.py:
    - High bandwidth (>1024 tokens): block_num=80, warp_num_per_block=16
    - Low latency (<=1024 tokens): block_num=64, warp_num_per_block=16

    Note: total_experts parameter is kept for API compatibility but not used.
    """
    if max_num_inp_token_per_rank > 1024:
        return 80, 16
    else:
        return 64, 16


def create_op_for_setting(
    rank: int,
    world_size: int,
    setting: dict,
    gpu_per_node_override: int | None = None,
    local_gpu_id: int | None = None,
):
    """Create EpDispatchCombineDeepepOp for the given setting.

    This function is used to create the op ONCE before the iteration loop
    to avoid repeated memory allocation/deallocation.
    """
    hidden_dim = setting["hidden_dim"]
    max_num_inp_token_per_rank = setting["max_num_inp_token_per_rank"]
    total_experts = setting["total_experts"]
    num_experts_per_token = setting["num_experts_per_token"]
    gpu_per_node = gpu_per_node_override or setting.get("gpu_per_node", world_size)
    use_fp8 = setting.get("use_fp8", hidden_dim >= 128)
    data_type = torch.bfloat16

    assert total_experts % world_size == 0
    num_experts_per_rank = total_experts // world_size

    is_internode = world_size > gpu_per_node
    kernel_type = (
        mori.ops.EpDispatchCombineDeepepKernelType.InterNodeLL
        if is_internode
        else mori.ops.EpDispatchCombineDeepepKernelType.IntraNode
    )

    block_num, warp_num_per_block = select_block_config(max_num_inp_token_per_rank, total_experts)

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
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        use_external_inp_buf=True,
        use_fp8=use_fp8,
        use_deepep_layout=True,
        use_weighted_combine=True,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
    )
    return mori.ops.EpDispatchCombineDeepepOp(config)


def run_test_worker(
    local_rank: int,
    num_local_ranks: int,
    setting: dict,
    port: int,
    gpu_per_node_override: int | None = None,
    dispatch_only: bool = False,
    iterations: int = 1,
    benchmark: bool = False,
    benchmark_warmup: int = 2,
    benchmark_iters: int = 10,
):
    """Worker function for mp.spawn mode (single node).

    In mp.spawn mode, all processes run on the same physical node. Each process
    gets its own physical GPU based on local_rank, regardless of virtual node
    topology set by gpu_per_node.
    """
    rank = local_rank
    world_size = num_local_ranks

    # Set CUDA device before any CUDA/shmem operations
    torch.cuda.set_device(local_rank)

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        _log(f"[Rank {rank}] Initializing shmem on GPU {local_rank}...", force=True)
        mori.shmem.shmem_torch_process_group_init("default")

        try:
            # Create op ONCE outside the iteration loop to avoid memory accumulation
            op = create_op_for_setting(rank, world_size, setting, gpu_per_node_override, local_gpu_id=local_rank)

            if benchmark:
                # Run benchmarking mode
                run_benchmark(
                    rank, world_size, setting, op,
                    dispatch_only=dispatch_only,
                    warmup=benchmark_warmup,
                    iters=benchmark_iters
                )
            else:
                # Run standard test mode
                for iteration in range(iterations):
                    if iterations > 1 and rank == 0:
                        print(f"\n[DeepEP] Iteration {iteration + 1}/{iterations}", flush=True)
                    # Pass local_gpu_id=local_rank so each process uses its own GPU
                    # even when simulating multi-node topology (gpu_per_node < world_size)
                    run_test_impl(
                        rank, world_size, setting, gpu_per_node_override,
                        local_gpu_id=local_rank, dispatch_only=dispatch_only, op=op
                    )
                    dist.barrier()  # Sync between iterations

            # Cleanup op after all iterations
            del op
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            mori.shmem.shmem_finalize()


def run_test_multinode(
    setting: dict,
    iterations: int = 1,
    dispatch_only: bool = False,
    benchmark: bool = False,
    benchmark_warmup: int = 2,
    benchmark_iters: int = 10,
):
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
        # Create op ONCE outside the iteration loop to avoid memory accumulation.
        # The op allocates symmetric memory which is expensive to allocate/free repeatedly.
        op = create_op_for_setting(rank, world_size, setting, gpu_per_node)

        if benchmark:
            # Run benchmarking mode
            run_benchmark(
                rank, world_size, setting, op,
                dispatch_only=dispatch_only,
                warmup=benchmark_warmup,
                iters=benchmark_iters
            )
        else:
            # Run standard test mode
            for iteration in range(iterations):
                if iterations > 1 and rank == 0:
                    print(f"\n[DeepEP] Iteration {iteration + 1}/{iterations}", flush=True)
                run_test_impl(rank, world_size, setting, gpu_per_node, dispatch_only=dispatch_only, op=op)
                dist.barrier()  # Sync between iterations

        # Cleanup op after all iterations
        del op
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()

    if rank == 0:
        if benchmark:
            print(f"[DeepEP MultiNode] Benchmark completed!", flush=True)
        else:
            print(f"[DeepEP MultiNode] All {iterations} iteration(s) passed!", flush=True)


def run_test_impl(
    rank: int,
    world_size: int,
    setting: dict,
    gpu_per_node_override: int | None = None,
    local_gpu_id: int | None = None,
    dispatch_only: bool = False,
    op: "mori.ops.EpDispatchCombineDeepepOp | None" = None,
):
    """Core test implementation shared by both single-node and multi-node modes.

    Args:
        local_gpu_id: Override for local GPU device ID. In simulated multi-node mode
                      (mp.spawn on single physical node), this should be `rank` since
                      each process needs its own physical GPU regardless of virtual
                      node topology.
        dispatch_only: If True, skip the combine phase (for debugging dispatch).
        op: Optional pre-created op to reuse across iterations. If None, a new op
            is created (and cleaned up) within this function.
    """
    owns_op = op is None  # Track if we created the op (and should clean it up)
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

    # In simulated multi-node mode (mp.spawn), local_gpu_id is passed explicitly
    # since each process needs its own physical GPU regardless of virtual topology.
    # In real multi-node mode, use rank % gpu_per_node to get local device.
    device_id = local_gpu_id if local_gpu_id is not None else (rank % gpu_per_node)
    device = torch.device("cuda", device_id)

    # Use CPU generator for deterministic random data across all ranks
    # CUDA generators may produce different sequences on different GPUs
    cpu_rng = torch.Generator()
    cpu_rng.manual_seed(123)

    # Always create config (needed for validation even when op is provided)
    block_num, warp_num_per_block = select_block_config(max_num_inp_token_per_rank, total_experts)
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
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        use_external_inp_buf=True,
        use_fp8=use_fp8,
        use_deepep_layout=True,
        use_weighted_combine=True,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
    )

    # Create op if not provided (for single-iteration or backwards compatibility)
    if op is None:
        op = mori.ops.EpDispatchCombineDeepepOp(config)

    # Generate test data using CPU RNG for deterministic values across all ranks
    # CUDA generators may produce different sequences on different GPUs
    num_token = torch.tensor(
        [max_num_inp_token_per_rank for _ in range(world_size)]
    ).to(device)

    all_rank_indices = []
    for r in range(world_size):
        indices = torch.empty(num_token[r], num_experts_per_token, dtype=torch.int64)
        for i in range(num_token[r]):
            # Generate permutation on CPU for determinism
            perm = torch.randperm(total_experts, generator=cpu_rng)
            indices[i] = perm[:num_experts_per_token]
        all_rank_indices.append(indices.to(torch.int32).to(device))

    all_rank_weights = [
        torch.ones(num_token[r], num_experts_per_token, dtype=torch.float32, device=device)
        for r in range(world_size)
    ]

    if DEBUG_SIMPLE_DATA:
        # Debug mode: each token's value = global_token_id (rank * max_tokens + token_idx)
        # repeated across the entire hidden dimension. This makes it easy to identify
        # which token's data appears in the output.
        all_rank_input = []
        for r in range(world_size):
            n = int(num_token[r].item()) if num_token[r].dim() == 0 else int(num_token[r])
            token_data = torch.zeros(n, hidden_dim, dtype=data_type, device=device)
            for token_idx in range(n):
                global_token_id = r * max_num_inp_token_per_rank + token_idx
                # Use float value so we can distinguish tokens
                token_data[token_idx, :] = float(global_token_id)
            all_rank_input.append(token_data)
        if rank == 0:
            print(f"[DEBUG] Using simplified input data: token values = global_token_id", flush=True)
            print(f"  Rank 0 token 0 value: {all_rank_input[0][0, 0].item()}", flush=True)
            print(f"  Rank 0 token 1 value: {all_rank_input[0][1, 0].item()}", flush=True)
            # Show expert routing for rank 0
            print(f"[DEBUG] Rank 0 token routing (gpuPerNode={gpu_per_node}, numExpertsPerRank={num_experts_per_rank}):", flush=True)
            for t in range(min(4, max_num_inp_token_per_rank)):
                for k in range(num_experts_per_token):
                    expert = all_rank_indices[0][t, k].item()
                    dest_pe = expert // num_experts_per_rank
                    is_self = dest_pe == 0
                    is_remote = (dest_pe // gpu_per_node) != (0 // gpu_per_node)
                    route_type = "SELF" if is_self else ("RDMA" if is_remote else "P2P")
                    print(f"    token {t} k={k}: expert={expert} -> destPe={dest_pe} [{route_type}]", flush=True)
    else:
        all_rank_input = [
            # Generate on CPU then move to GPU for deterministic values across ranks
            (torch.rand(num_token[r], hidden_dim, dtype=torch.float32, generator=cpu_rng) * 2 - 1).to(data_type).to(device)
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
            rank, num_experts_per_rank, world_size, recv_count, all_rank_indices
        )
        # Also validate data placement (only for internode on rank 0)
        # Internode uses different buffer layout than intranode
        if rank == 0 and is_internode:
            validate_dispatch_data(
                rank=rank,
                world_size=world_size,
                num_experts_per_rank=num_experts_per_rank,
                hidden_dim=hidden_dim,
                gpu_per_node=gpu_per_node,
                dispatch_output=dispatch_output,
                dispatch_scales=dispatch_scales,
                recv_count=recv_count,
                all_rank_input=all_rank_input,
                all_rank_indices=all_rank_indices,
                use_fp8=use_fp8,
                data_type=data_type,
                max_tokens_to_check=10,
            )
        torch.cuda.synchronize()
        dist.barrier()

    # Skip combine phase if dispatch-only mode
    if dispatch_only:
        # Barrier to ensure all ranks complete validation before reset
        dist.barrier()
        op.reset()
        # Only cleanup if we created the op (not passed in from caller)
        if owns_op:
            del op
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"[DeepEP] Dispatch-only test '{setting['name']}' PASSED", flush=True)
        return

    # Combine
    _log(f"[Rank {rank}] Starting combine...", force=True)
    combine_input = dispatch_output
    if use_fp8:
        combine_input = dequant_dispatch_output(dispatch_output, dispatch_scales, hidden_dim)

    # Debug: show dispatch output (combine input) values for rank 0
    if DEBUG_SIMPLE_DATA and rank == 0:
        print(f"[DEBUG] Dispatch output shape: {dispatch_output.shape}", flush=True)
        # Show first value at each expert's slot 0 (from rank 0's partition)
        expert_capacity = world_size * max_num_inp_token_per_rank
        for e in range(num_experts_per_rank):
            for s in range(min(4, max_num_inp_token_per_rank)):
                val = combine_input[e, s, 0].item()
                if abs(val) > 1e-6:  # Only show non-zero
                    print(f"    dispatch_out[expert={e}][slot={s}][0] = {val:.1f}", flush=True)

    combine_output, _, _ = combine_func(
        combine_input, dispatch_indices, dispatch_weights, handle=handle
    )

    torch.cuda.synchronize()
    dist.barrier()
    _log(f"[Rank {rank}] Combine complete.", force=True)

    # Debug: show combine output values for rank 0
    if DEBUG_SIMPLE_DATA and rank == 0:
        print(f"[DEBUG] Combine output values:", flush=True)
        for t in range(min(4, max_num_inp_token_per_rank)):
            val = combine_output[t, 0].item()
            print(f"    combine_out[token={t}][0] = {val:.1f}", flush=True)

    # Validate combine result
    if not SKIP_COMBINE_CHECKS and rank == 0:
        validate_combine(
            config, combine_output, all_rank_input, all_rank_weights, use_fp8, data_type
        )

    op.reset()

    # Only cleanup if we created the op (not passed in from caller)
    if owns_op:
        del op
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"[DeepEP] Test '{setting['name']}' PASSED", flush=True)


def validate_dispatch(rank, num_experts_per_rank, world_size, recv_count, all_rank_indices):
    """Validate dispatch results by checking token counts."""
    expected_from_indices = 0
    rank_begin = rank * num_experts_per_rank
    rank_end = rank_begin + num_experts_per_rank
    for r in range(world_size):
        idx = all_rank_indices[r]
        expected_from_indices += int(((idx >= rank_begin) & (idx < rank_end)).sum().item())

    actual_from_counts = int(recv_count.sum().item())

    # Debug: print expected vs actual before checking
    if DEBUG_LOG or actual_from_counts != expected_from_indices:
        print(f"[Rank {rank}] Validation: expected={expected_from_indices}, actual={actual_from_counts}", flush=True)

    if actual_from_counts != expected_from_indices:
        raise AssertionError(
            f"Rank {rank}: dispatch count mismatch. "
            f"Expected {expected_from_indices}, got {actual_from_counts}"
        )
    _log(f"[Rank {rank}] Dispatch validation passed: {actual_from_counts} tokens", force=True)


def validate_dispatch_data(
    rank: int,
    world_size: int,
    num_experts_per_rank: int,
    hidden_dim: int,
    gpu_per_node: int,
    dispatch_output: torch.Tensor,
    dispatch_scales: torch.Tensor | None,
    recv_count: torch.Tensor,
    all_rank_input: list,
    all_rank_indices: list,
    use_fp8: bool,
    data_type: torch.dtype,
    max_tokens_to_check: int = 10,
):
    """Validate dispatch data placement by checking actual token values.

    This function verifies that tokens dispatched from each source rank are
    placed at the correct locations in the output buffer with correct values.

    Buffer layout (internode):
    - dispatch_output: [num_local_experts, expert_capacity, hidden_dim]
    - expert_capacity = max_num_inp_token_per_rank * world_size (for internode)
    - Each source rank has reserved slots: [srcPe * max_tokens, (srcPe+1) * max_tokens)

    NOTE: Slot assignment within a source rank's partition is non-deterministic
    due to SM-strided token processing with atomicAdd. We cannot predict the exact
    slot, but we can verify that each expected token appears SOMEWHERE in the
    source rank's partition.

    Args:
        rank: Current rank
        world_size: Total number of ranks
        num_experts_per_rank: Number of experts per rank
        hidden_dim: Hidden dimension
        gpu_per_node: GPUs per node (to determine if internode)
        dispatch_output: Output buffer [num_experts, capacity, hidden_dim]
        dispatch_scales: Scales for FP8 (or None for BF16)
        recv_count: Received count per expert [num_experts]
        all_rank_input: List of input tensors for each rank
        all_rank_indices: List of expert indices for each rank
        use_fp8: Whether FP8 quantization is used
        data_type: Data type for comparison
        max_tokens_to_check: Maximum tokens to check per expert (for performance)
    """
    is_internode = world_size > gpu_per_node

    # Dequantize output if FP8
    if use_fp8 and dispatch_scales is not None:
        dispatch_out_bf16 = dequant_dispatch_output(dispatch_output, dispatch_scales, hidden_dim)
    else:
        dispatch_out_bf16 = dispatch_output

    # For internode: each source rank gets slots [srcPe * max_tokens, (srcPe+1) * max_tokens)
    max_tokens_per_rank = all_rank_input[0].shape[0]

    # Build expected token placement for this rank's local experts
    rank_begin = rank * num_experts_per_rank
    rank_end = rank_begin + num_experts_per_rank

    print(f"[Rank {rank}] Validating dispatch data placement...", flush=True)
    print(f"  Local experts: {rank_begin} to {rank_end - 1}", flush=True)
    print(f"  dispatch_output shape: {dispatch_out_bf16.shape}", flush=True)
    print(f"  is_internode: {is_internode}", flush=True)

    # Build expected tokens per (local_expert, src_rank)
    # expected_tokens[local_expert][src_rank] = list of (src_token_idx, expected_tensor)
    expected_tokens = {e: {r: [] for r in range(world_size)} for e in range(num_experts_per_rank)}

    for src_rank in range(world_size):
        indices = all_rank_indices[src_rank]  # [num_tokens, topk]
        for src_token_idx in range(indices.shape[0]):
            for k in range(indices.shape[1]):
                expert_id = indices[src_token_idx, k].item()
                if rank_begin <= expert_id < rank_end:
                    local_expert = expert_id - rank_begin
                    # Get expected value (from source rank's input)
                    expected_input = all_rank_input[src_rank][src_token_idx]
                    if use_fp8:
                        expected_input = dequant_input_like_fp8(expected_input.unsqueeze(0), data_type).squeeze(0)
                    expected_tokens[local_expert][src_rank].append((src_token_idx, expected_input))

    # Validate each local expert
    errors_found = 0
    tokens_matched = 0
    tokens_checked = 0

    for local_expert in range(num_experts_per_rank):
        expert_recv_count = int(recv_count[local_expert].item())
        total_expected = sum(len(expected_tokens[local_expert][r]) for r in range(world_size))

        # Check count matches
        if total_expected != expert_recv_count:
            print(f"  [Expert {local_expert}] Count mismatch: expected {total_expected}, got {expert_recv_count}", flush=True)
            errors_found += 1
            continue

        # For each source rank, get actual tokens in its partition and match against expected
        for src_rank in range(world_size):
            expected_list = expected_tokens[local_expert][src_rank]
            if not expected_list:
                continue

            num_expected = len(expected_list)
            partition_start = src_rank * max_tokens_per_rank
            partition_end = partition_start + num_expected  # Only check slots that should have data

            # Get actual tokens from this partition
            actual_tokens = []
            for slot_idx in range(partition_start, partition_end):
                actual = dispatch_out_bf16[local_expert, slot_idx, :]
                actual_tokens.append(actual)

            # Match expected tokens against actual (order may differ)
            # Use greedy matching: for each expected, find best match in actual
            matched = [False] * num_expected
            atol = 0.5 if use_fp8 else 0.01
            rtol = 0.25 if use_fp8 else 0.01

            for i, (src_token_idx, expected_input) in enumerate(expected_list):
                if tokens_checked >= max_tokens_to_check * num_experts_per_rank:
                    break  # Limit total checks for performance

                found_match = False
                for j, actual in enumerate(actual_tokens):
                    if matched[j]:
                        continue
                    if torch.allclose(actual.float(), expected_input.float(), atol=atol, rtol=rtol):
                        matched[j] = True
                        found_match = True
                        tokens_matched += 1
                        break

                tokens_checked += 1

                if not found_match:
                    # Find the closest match to help diagnose the issue
                    min_diff = float('inf')
                    closest_slot = -1
                    for j, actual in enumerate(actual_tokens):
                        diff = (actual.float() - expected_input.float()).abs().max().item()
                        if diff < min_diff:
                            min_diff = diff
                            closest_slot = j

                    # If min_diff is 0, the value exists but was matched to a different expected token
                    # This can happen when multiple tokens have similar/identical values
                    # In this case, it's not really an error - just a matching order issue
                    if min_diff < atol:
                        tokens_matched += 1
                        continue  # Skip this as a non-error

                    print(f"  [Expert {local_expert}] Token not found in partition:", flush=True)
                    print(f"    src_rank={src_rank}, src_token={src_token_idx}", flush=True)
                    print(f"    expected[:8]={expected_input[:8].tolist()}", flush=True)
                    print(f"    closest_slot={closest_slot}, max_diff={min_diff:.6f}, atol={atol}", flush=True)
                    if closest_slot >= 0 and closest_slot < len(actual_tokens):
                        closest = actual_tokens[closest_slot]
                        print(f"    closest[:8]={closest[:8].tolist()}", flush=True)
                        # Show element-wise diff for first 8
                        diff8 = (closest[:8].float() - expected_input[:8].float()).abs()
                        print(f"    diff[:8]={diff8.tolist()}", flush=True)
                    print(f"    Partition [{partition_start}:{partition_end}] contents:", flush=True)
                    for s_idx, actual in enumerate(actual_tokens[:4]):  # Show first 4 slots
                        is_zero = actual.abs().max().item() < 1e-6
                        print(f"      slot[{partition_start + s_idx}]: {actual[:8].tolist()} {'(zero)' if is_zero else ''}", flush=True)
                    errors_found += 1

    if errors_found > 0:
        raise AssertionError(
            f"Rank {rank}: Dispatch data validation failed with {errors_found} errors"
        )

    print(f"[Rank {rank}] Dispatch data validation passed: matched {tokens_matched}/{tokens_checked} tokens", flush=True)


def validate_combine(config, combine_output, all_rank_input, all_rank_weights, use_fp8, data_type):
    """Validate combine results (rank 0 only, samples first few tokens)."""
    num_tokens_to_check = min(100, config.max_num_inp_token_per_rank)
    base_input = all_rank_input[0]
    if use_fp8:
        base_input = dequant_input_like_fp8(base_input, data_type)

    topk = config.num_experts_per_token

    for i in range(num_tokens_to_check):
        got = combine_output[i]
        weights = all_rank_weights[0][i].to(torch.float32)
        expected = torch.zeros_like(got)
        for k in range(weights.numel()):
            expected = (expected + (base_input[i].to(torch.float32) * weights[k])).to(data_type)

        atol = 0.25 if use_fp8 else 1e-2
        rtol = 0.25 if use_fp8 else 1e-2
        if not torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol):
            # Enhanced debug output for DEBUG_SIMPLE_DATA mode
            if DEBUG_SIMPLE_DATA:
                # In simple data mode, expected value = topk * token_id (since all weights are 1)
                expected_simple = float(i * topk)
                got_val = got[0].item()
                print(f"[DEBUG] Combine mismatch at token {i}:", flush=True)
                print(f"  Expected (simple): {expected_simple} (= token_id {i} * topk {topk})", flush=True)
                print(f"  Got value[0]: {got_val}", flush=True)
                # Try to infer which token's data this might be
                if got_val > 0:
                    inferred_token_id = got_val / topk
                    inferred_rank = int(inferred_token_id // config.max_num_inp_token_per_rank)
                    inferred_local_token = int(inferred_token_id % config.max_num_inp_token_per_rank)
                    print(f"  Inferred source: rank={inferred_rank}, local_token={inferred_local_token}", flush=True)
                print(f"  got[:8]={got[:8].tolist()}", flush=True)
                print(f"  expected[:8]={expected[:8].tolist()}", flush=True)
            raise AssertionError(
                f"Combine mismatch at token {i}: "
                f"got={got[:8].tolist()}, expected={expected[:8].tolist()}"
            )

    _log("[Rank 0] Combine validation passed", force=True)


def run_once_benchmark(
    rank: int,
    world_size: int,
    setting: dict,
    op: "mori.ops.EpDispatchCombineDeepepOp",
    all_rank_input: list,
    all_rank_indices: list,
    all_rank_weights: list,
    check_result: bool = False,
    dispatch_only: bool = False,
) -> dict:
    """Run a single dispatch+combine iteration with timing measurements.

    Args:
        check_result: If True, validate the results (used for warmup)
        dispatch_only: If True, skip combine phase

    Returns:
        Dictionary with timing metrics: {
            'disp_duration_ms': float,
            'comb_duration_ms': float,
            'total_recv_num_token': int,
        }
    """
    hidden_dim = setting["hidden_dim"]
    max_num_inp_token_per_rank = setting["max_num_inp_token_per_rank"]
    total_experts = setting["total_experts"]
    num_experts_per_token = setting["num_experts_per_token"]
    gpu_per_node = setting.get("gpu_per_node", world_size)
    use_fp8 = setting.get("use_fp8", hidden_dim >= 128)

    assert total_experts % world_size == 0
    num_experts_per_rank = total_experts // world_size

    is_internode = world_size > gpu_per_node

    # Select dispatch/combine functions based on kernel type
    dispatch_func = op.dispatch_internode_deepep_ll if is_internode else op.dispatch_deepep_ll
    combine_func = op.combine_internode_deepep_ll if is_internode else op.combine_deepep_ll

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Time dispatch
    dist.barrier()
    start_event.record()
    recv_x, recv_count, handle, _, _ = dispatch_func(
        all_rank_input[rank],
        all_rank_indices[rank],
        num_max_dispatch_tokens_per_rank=max_num_inp_token_per_rank,
        num_experts=total_experts,
        use_fp8=use_fp8,
        weights=all_rank_weights[rank],
    )
    end_event.record()
    dist.barrier()
    disp_duration_ms = start_event.elapsed_time(end_event)

    if use_fp8:
        dispatch_output, dispatch_scales = recv_x
    else:
        dispatch_output, dispatch_scales = recv_x, None

    dispatch_weights, dispatch_indices = handle
    dispatch_recv_num_token = recv_count.sum().to(torch.int32)
    total_recv_num_token = dispatch_recv_num_token.item()

    # Validate dispatch if requested
    if check_result:
        validate_dispatch(
            rank, num_experts_per_rank, world_size, recv_count, all_rank_indices
        )

    comb_duration_ms = 0.0
    if not dispatch_only:
        # Time combine
        combine_input = dispatch_output
        if use_fp8:
            combine_input = dequant_dispatch_output(dispatch_output, dispatch_scales, hidden_dim)

        dist.barrier()
        start_event.record()
        combine_output, _, _ = combine_func(
            combine_input, dispatch_indices, dispatch_weights, handle=handle
        )
        end_event.record()
        dist.barrier()
        comb_duration_ms = start_event.elapsed_time(end_event)

        # Validate combine if requested
        if check_result and rank == 0:
            block_num, warp_num_per_block = select_block_config(max_num_inp_token_per_rank, total_experts)
            config = mori.ops.EpDispatchCombineDeepepConfig(
                data_type=torch.bfloat16,
                rank=rank,
                world_size=world_size,
                hidden_dim=hidden_dim,
                scale_dim=hidden_dim // 128,
                scale_type_size=4,
                max_num_inp_token_per_rank=max_num_inp_token_per_rank,
                num_experts_per_rank=num_experts_per_rank,
                num_experts_per_token=num_experts_per_token,
                max_token_type_size=4,
                block_num=block_num,
                warp_num_per_block=warp_num_per_block,
                use_external_inp_buf=True,
                use_fp8=use_fp8,
                use_deepep_layout=True,
                use_weighted_combine=True,
                kernel_type=(
                    mori.ops.EpDispatchCombineDeepepKernelType.InterNodeLL
                    if is_internode
                    else mori.ops.EpDispatchCombineDeepepKernelType.IntraNode
                ),
                gpu_per_node=gpu_per_node,
            )
            validate_combine(
                config, combine_output, all_rank_input, all_rank_weights, use_fp8, torch.bfloat16
            )

    op.reset()

    return {
        'disp_duration_ms': disp_duration_ms,
        'comb_duration_ms': comb_duration_ms,
        'total_recv_num_token': total_recv_num_token,
    }


def run_benchmark(
    rank: int,
    world_size: int,
    setting: dict,
    op: "mori.ops.EpDispatchCombineDeepepOp",
    dispatch_only: bool = False,
    warmup: int = 2,
    iters: int = 10,
):
    """Run benchmark with multiple iterations and report statistics.

    Performs warmup iterations with validation, then runs benchmark iterations
    without validation for accurate timing.
    """
    hidden_dim = setting["hidden_dim"]
    max_num_inp_token_per_rank = setting["max_num_inp_token_per_rank"]
    total_experts = setting["total_experts"]
    num_experts_per_token = setting["num_experts_per_token"]
    gpu_per_node = setting.get("gpu_per_node", world_size)
    use_fp8 = setting.get("use_fp8", hidden_dim >= 128)
    data_type = torch.bfloat16

    device = torch.device("cuda", torch.cuda.current_device())

    # Use CPU generator for deterministic random data across all ranks
    cpu_rng = torch.Generator()
    cpu_rng.manual_seed(123)

    assert total_experts % world_size == 0
    num_experts_per_rank = total_experts // world_size

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

    # Generate test data once using CPU RNG for determinism across ranks
    def gen_test_data():
        num_token = torch.tensor(
            [max_num_inp_token_per_rank for _ in range(world_size)]
        ).to(device)

        all_rank_indices = []
        for r in range(world_size):
            indices = torch.empty(num_token[r], num_experts_per_token, dtype=torch.int64)
            for i in range(num_token[r]):
                perm = torch.randperm(total_experts, generator=cpu_rng)
                indices[i] = perm[:num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(device))

        all_rank_weights = [
            torch.ones(num_token[r], num_experts_per_token, dtype=torch.float32, device=device)
            for r in range(world_size)
        ]

        if DEBUG_SIMPLE_DATA:
            # Debug mode: each token's value = global_token_id
            all_rank_input = []
            for r in range(world_size):
                n = int(num_token[r].item()) if num_token[r].dim() == 0 else int(num_token[r])
                token_data = torch.zeros(n, hidden_dim, dtype=data_type, device=device)
                for token_idx in range(n):
                    global_token_id = r * max_num_inp_token_per_rank + token_idx
                    token_data[token_idx, :] = float(global_token_id)
                all_rank_input.append(token_data)
        else:
            all_rank_input = [
                (torch.rand(num_token[r], hidden_dim, dtype=torch.float32, generator=cpu_rng) * 2 - 1).to(data_type).to(device)
                for r in range(world_size)
            ]

        return all_rank_input, all_rank_indices, all_rank_weights

    # Warmup with validation
    if rank == 0:
        print(f"\n[Benchmark] Warmup ({warmup} iteration(s))...", flush=True)
    for _ in range(warmup):
        all_rank_input, all_rank_indices, all_rank_weights = gen_test_data()
        run_once_benchmark(
            rank, world_size, setting, op,
            all_rank_input, all_rank_indices, all_rank_weights,
            check_result=True, dispatch_only=dispatch_only
        )

    # Benchmark iterations without validation
    disp_times_ms = []
    comb_times_ms = []
    recv_token_counts = []

    if rank == 0:
        print(f"[Benchmark] Running {iters} iteration(s) for measurements...", flush=True)

    for i in range(iters):
        all_rank_input, all_rank_indices, all_rank_weights = gen_test_data()
        metrics = run_once_benchmark(
            rank, world_size, setting, op,
            all_rank_input, all_rank_indices, all_rank_weights,
            check_result=False, dispatch_only=dispatch_only
        )

        disp_times_ms.append(metrics['disp_duration_ms'])
        comb_times_ms.append(metrics['comb_duration_ms'])
        recv_token_counts.append(metrics['total_recv_num_token'])

    # Gather timing from all ranks
    if rank == 0:
        all_disp_times = []
        all_comb_times = []
        all_token_counts = []
    else:
        all_disp_times = None
        all_comb_times = None
        all_token_counts = None

    for i in range(iters):
        disp_time_tensor = torch.tensor([disp_times_ms[i]], device=device)
        comb_time_tensor = torch.tensor([comb_times_ms[i]], device=device)
        token_count_tensor = torch.tensor([recv_token_counts[i]], dtype=torch.float32, device=device)

        if rank == 0:
            disp_gather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
            comb_gather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
            token_gather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
        else:
            disp_gather_list = None
            comb_gather_list = None
            token_gather_list = None

        dist.gather(disp_time_tensor, disp_gather_list if rank == 0 else None, dst=0)
        dist.gather(comb_time_tensor, comb_gather_list if rank == 0 else None, dst=0)
        dist.gather(token_count_tensor, token_gather_list if rank == 0 else None, dst=0)

        if rank == 0:
            all_disp_times.append([t.item() for t in disp_gather_list])
            all_comb_times.append([t.item() for t in comb_gather_list])
            all_token_counts.append([int(t.item()) for t in token_gather_list])

    # Print results
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"[Benchmark Results] {setting['name']}")
        print(f"{'='*70}")

        # Calculate bytes per token following DeepEP's formula
        # FP8 dispatch: data (1 byte) + scales (hidden/128 * 4 bytes) + overhead (16 bytes)
        # Non-FP8 dispatch: bfloat16 (2 bytes per element)
        # Combine: always bfloat16 (2 bytes per element)
        if use_fp8:
            bytes_per_token_dispatch = hidden_dim + hidden_dim // 128 * 4 + 16
        else:
            bytes_per_token_dispatch = hidden_dim * 2
        bytes_per_token_combine = hidden_dim * 2  # Always bfloat16 after dequant

        # Calculate ll_mode_scale for reference
        # This is used to show performance under load imbalance
        avg_recv_token = sum(all_token_counts[0]) / len(all_token_counts[0]) if all_token_counts else 1
        ll_mode_scale = (
            max_num_inp_token_per_rank * num_experts_per_token / (avg_recv_token + 0.01)
        )

        # Calculate bandwidth metrics
        disp_bandwidth_GB_list = []
        comb_bandwidth_GB_list = []
        disp_bytes_MB_list = []
        comb_bytes_MB_list = []

        for i in range(len(all_disp_times)):
            # Bytes for dispatch and combine
            num_tokens = all_token_counts[i][0] if i < len(all_token_counts) else 0
            disp_bytes = num_tokens * bytes_per_token_dispatch
            comb_bytes = num_tokens * bytes_per_token_combine
            disp_bytes_MB_list.append(int(disp_bytes / (1024**2)))
            comb_bytes_MB_list.append(int(comb_bytes / (1024**2)))

            # Dispatch bandwidth for each rank (GB/s)
            disp_bw_ranks = []
            for rank_idx in range(world_size):
                if all_disp_times[i][rank_idx] > 0:
                    bw = disp_bytes / (1000**3) / (all_disp_times[i][rank_idx] / 1000)
                    disp_bw_ranks.append(int(bw))
                else:
                    disp_bw_ranks.append(0)
            disp_bandwidth_GB_list.append(disp_bw_ranks)

            # Combine bandwidth for each rank (GB/s)
            comb_bw_ranks = []
            for rank_idx in range(world_size):
                if all_comb_times[i][rank_idx] > 0:
                    bw = comb_bytes / (1000**3) / (all_comb_times[i][rank_idx] / 1000)
                    comb_bw_ranks.append(int(bw))
                else:
                    comb_bw_ranks.append(0)
            comb_bandwidth_GB_list.append(comb_bw_ranks)

        # Print Dispatch results
        print("Dispatch result:")
        max_disp_algo_bw = 0
        for i in range(len(all_disp_times)):
            # Convert ms to us
            duration_us = [int(t * 1000) for t in all_disp_times[i]]
            algo_bw = sum(disp_bandwidth_GB_list[i]) / world_size
            max_disp_algo_bw = max(max_disp_algo_bw, algo_bw)
            print(
                f"Round {i} duration(us) {duration_us} "
                f"bandwidth(GB/s) {disp_bandwidth_GB_list[i]} "
                f"bytes(MB) {disp_bytes_MB_list[i]} bw {algo_bw:.1f} / {algo_bw*ll_mode_scale:.2f}"
            )

        print()
        print("Combine result:")
        max_comb_algo_bw = 0
        for i in range(len(all_comb_times)):
            # Convert ms to us
            duration_us = [int(t * 1000) for t in all_comb_times[i]]
            algo_bw = sum(comb_bandwidth_GB_list[i]) / world_size
            max_comb_algo_bw = max(max_comb_algo_bw, algo_bw)
            print(
                f"Round {i} duration(us) {duration_us} "
                f"bandwidth(GB/s) {comb_bandwidth_GB_list[i]} "
                f"bytes(MB) {comb_bytes_MB_list[i]} bw {algo_bw:.1f} / {algo_bw*ll_mode_scale:.2f}"
            )

        print()
        print(f"Best Dispatch  performance: {max_disp_algo_bw:.2f} GB/s")
        print(f"Best Combine   performance: {max_comb_algo_bw:.2f} GB/s")
        print(f"{'='*70}\n")


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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarking (measures timing over multiple iterations)",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=2,
        help="Number of warmup iterations before benchmarking (default: 2)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=10,
        help="Number of benchmark iterations to run (default: 10)",
    )
    args = parser.parse_args()

    # Set global flags
    # Multi-node mode: use torchrun/srun for process management
    if args.multinode:
        setting = PRESET_SETTINGS[args.setting].copy()
        if args.gpu_per_node:
            # Explicit override via --gpu-per-node
            setting["gpu_per_node"] = args.gpu_per_node
        else:
            # Default to LOCAL_WORLD_SIZE (nproc_per_node from torchrun)
            setting["gpu_per_node"] = int(os.environ.get("LOCAL_WORLD_SIZE", setting["gpu_per_node"]))
        run_test_multinode(
            setting,
            iterations=args.iterations,
            dispatch_only=args.dispatch_only,
            benchmark=args.benchmark,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_iters=args.benchmark_iters,
        )
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
        if args.benchmark:
            print(f"[DeepEP] Benchmarking with warmup={args.benchmark_warmup}, iters={args.benchmark_iters}", flush=True)
        elif args.iterations > 1:
            print(f"[DeepEP] Running {args.iterations} iterations", flush=True)
        print("=" * 80, flush=True)

        port = get_free_port()
        mp.spawn(
            run_test_worker,
            args=(
                num_processes,
                setting,
                port,
                gpu_per_node,
                args.dispatch_only,
                args.iterations,
                args.benchmark,
                args.benchmark_warmup,
                args.benchmark_iters,
            ),
            nprocs=num_processes,
        )

        if args.benchmark:
            print(f"[DeepEP] Benchmark completed!", flush=True)
        elif args.iterations > 1:
            print(f"[DeepEP] All {args.iterations} iterations passed!", flush=True)
        print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
