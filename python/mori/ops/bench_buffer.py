"""Micro-benchmarks for MORI Buffer operations.

Run with torchrun, for example:
  torchrun --nproc_per_node=8 python python/mori/ops/bench_buffer.py --num-tokens 8192 --hidden-dim 4096
"""

import argparse
import os
import time
from typing import Callable, Dict, List, Optional, Tuple
import torch.multiprocessing as mp
import torch
import torch.distributed as dist

import mori
from mori.ops.Buffer import Buffer

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Buffer dispatch/combine variants and helpers")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="bf16")
    parser.add_argument("--total-experts", type=int, default=256)
    parser.add_argument("--disable-reorder", action="store_true")
    parser.add_argument("--disable-gpu-ll-layout-transform", action="store_true")
    parser.add_argument("--low-latency", default=False, action="store_true")
    return parser.parse_args()


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")


def normalize_recv(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.view(-1)[0].item())
    if isinstance(value, (list, tuple)) and value:
        return int(value[0])
    return int(value)


def benchmark_op(name: str, fn: Callable[[], None], iters: int, warmup: int) -> Tuple[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
    return name, avg_ms


def measure_breakdown(iter_fn: Callable[[], List[Tuple[str, float]]], iters: int, warmup: int) -> List[Tuple[str, float]]:
    order: List[str] = []
    totals: Dict[str, float] = {}
    total_iters = max(1, iters)
    for step in range(warmup + iters):
        stage_times = iter_fn()
        if step < warmup:
            continue
        if not order:
            order = [name for name, _ in stage_times]
            totals = {name: 0.0 for name in order}
        for name, duration in stage_times:
            totals[name] += duration
    return [(name, totals[name] * 1000.0 / total_iters) for name in order]


def aggregate_stage_totals(stage_results: List[Tuple[str, float]], groups: Dict[str, List[str]]) -> List[Tuple[str, float]]:
    aggregated: List[Tuple[str, float]] = []
    stage_dict = dict(stage_results)
    for group, names in groups.items():
        aggregated.append((group, sum(stage_dict.get(name, 0.0) for name in names)))
    return aggregated


def gather_and_print(results: List[Tuple[str, float]]) -> None:
    world = dist.get_world_size()
    collected: List[Optional[List[Tuple[str, float]]]] = [None for _ in range(world)]
    dist.all_gather_object(collected, results)
    if dist.get_rank() != 0:
        return

    summary: Dict[str, List[float]] = {}
    for rank_results in collected:
        if not rank_results:
            continue
        for name, latency in rank_results:
            summary.setdefault(name, []).append(latency)

    def get_stats(name: str) -> Optional[Dict[str, float]]:
        values = summary.get(name)
        if not values:
            return None
        avg = sum(values) / len(values)
        return {
            "avg": avg,
            "min": min(values),
            "max": max(values),
        }

    def print_metric(label: str, stats: Dict[str, float]) -> None:
        print(
            f"{label:45s} avg={stats['avg']:8.3f} min={stats['min']:8.3f} max={stats['max']:8.3f}"
        )

    def print_stage_group(title: str, keys: Dict[str, str]) -> None:
        total_stats = get_stats(keys["total"])
        if not total_stats:
            return
        total_avg = total_stats["avg"]
        print(f"\n--- {title} ---")
        for phase_key, phase_label in (
            ("pre", "Preprocess"),
            ("ops", "Ops"),
            ("post", "Postprocess"),
        ):
            phase_stats = get_stats(keys[phase_key])
            if not phase_stats:
                continue
            pct = (phase_stats["avg"] / total_avg * 100.0) if total_avg else 0.0
            print(
                f"  {phase_label:12s}: {phase_stats['avg']:6.3f} ms ({pct:5.1f}%)"
            )
        print(
            f"  Total        : {total_stats['avg']:6.3f} ms (min={total_stats['min']:6.3f} max={total_stats['max']:6.3f})"
        )

    helper_metrics = [
        ("transform_dispatch_output_gpu", "transform_dispatch_output_gpu"),
        ("_reorder_mori_dispatch_outputs", "_reorder_mori_dispatch_outputs"),
        ("_revert_mori_dispatch_outputs", "_revert_mori_dispatch_outputs"),
        ("inverse_transform_dispatch_output_gpu", "inverse_transform_dispatch_output_gpu"),
    ]

    printed_header = False
    for key, label in helper_metrics:
        stats = get_stats(key)
        if not stats:
            continue
        if not printed_header:
            print("\n=== Helper Ops (ms) ===")
            printed_header = True
        print_metric(label, stats)
        
    if get_stats("dispatch.total") or get_stats("combine.total"):
        print("\n=== Standard Dispatch/Combine ===")
        print_stage_group(
            "Dispatch",
            {
                "total": "dispatch.total",
                "pre": "dispatch.preprocess",
                "ops": "dispatch.ops",
                "post": "dispatch.postprocess",
            },
        )
        print_stage_group(
            "Combine",
            {
                "total": "combine.total",
                "pre": "combine.preprocess",
                "ops": "combine.ops",
                "post": "combine.postprocess",
            },
        )

    if get_stats("low_latency_dispatch.total") or get_stats("low_latency_combine.total"):
        print("\n=== Low-Latency Dispatch/Combine ===")
        print_stage_group(
            "LL Dispatch",
            {
                "total": "low_latency_dispatch.total",
                "pre": "low_latency_dispatch.preprocess",
                "ops": "low_latency_dispatch.ops",
                "post": "low_latency_dispatch.postprocess",
            },
        )
        print_stage_group(
            "LL Combine",
            {
                "total": "low_latency_combine.total",
                "pre": "low_latency_combine.preprocess",
                "ops": "low_latency_combine.ops",
                "post": "low_latency_combine.postprocess",
            },
        )


def make_topk_indices(num_tokens: int, topk: int, experts_per_rank: int, total_experts: int, rank: int, device: torch.device) -> torch.Tensor:
    local = torch.randint(0, experts_per_rank, (num_tokens, 1), device=device, dtype=torch.int64)
    local += rank * experts_per_rank
    if topk == 1:
        return local
    extra = torch.randint(0, total_experts, (num_tokens, topk - 1), device=device, dtype=torch.int64)
    return torch.cat([local, extra], dim=1)


def capture_op_dispatch(buffer: Buffer, inp: torch.Tensor, topk_weights: torch.Tensor, dispatch_indices_arg: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
    op = buffer._get_op(inp.dtype, inp.size(1), 0)
    out, weights, scales, indices, recv = op.dispatch(inp, topk_weights, None, dispatch_indices_arg)
    recv_count = normalize_recv(recv)
    src_pos = op.get_dispatch_src_token_pos()[:recv_count]
    truncated = out[:recv_count]
    trunc_indices = indices[:recv_count] if indices is not None else None
    trunc_weights = weights[:recv_count] if weights is not None else None
    trunc_scales = scales[:recv_count] if scales is not None else None
    return {
        "full_output": out,
        "full_indices": indices,
        "full_scales": scales,
        "recv_count": recv_count,
        "truncated_output": truncated,
        "truncated_indices": trunc_indices,
        "truncated_weights": trunc_weights,
        "truncated_scales": trunc_scales,
        "src_pos": src_pos,
    }


def run(rank: int, args: argparse.Namespace) -> None:
    ensure_cuda()
    world_size = args.num_processes
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    dist.init_process_group(
        backend=args.backend,
        rank=rank,
        world_size=world_size,
    )
    world = dist.get_world_size()
    local_rank = rank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if args.low_latency:
        args.num_tokens = 128
        args.total_experts = 288


    if args.total_experts < world:
        raise ValueError("total_experts must be at least world size so each rank hosts >= 1 expert")
    if args.total_experts % world != 0:
        raise ValueError("total_experts must be divisible by world size to derive experts per rank")
    experts_per_rank = args.total_experts // world

    dtype = DTYPE_MAP[args.dtype]
    max_tokens = args.num_tokens
    buffer = Buffer(
        group=dist.group.WORLD,
        num_qps_per_rank=experts_per_rank,
        max_num_inp_token_per_rank=max_tokens,
        num_experts_per_token=args.topk,
        reorder=not args.disable_reorder,
        use_gpu_ll_layout_transform=not args.disable_gpu_ll_layout_transform,
    )
    device = buffer.device
    total_experts = args.total_experts

    tokens = torch.randn(args.num_tokens, args.hidden_dim, device=device, dtype=dtype)
    topk_idx = make_topk_indices(args.num_tokens, args.topk, experts_per_rank, total_experts, rank, device)
    topk_weights = torch.rand(args.num_tokens, args.topk, device=device, dtype=torch.float32)
    dispatch_indices_arg = topk_idx.to(dtype=torch.int32)

    dist.barrier()

    manual = capture_op_dispatch(buffer, tokens, topk_weights, dispatch_indices_arg)
    if manual["recv_count"] == 0:
        raise RuntimeError("Dispatch produced zero tokens on this rank; adjust token/expert parameters")
    transform_cache = mori.transform_dispatch_output_gpu(
        manual["full_output"],
        manual["full_indices"],
        buffer.config,
        manual["recv_count"],
        manual["full_scales"],
    )
    reorder_out = buffer._reorder_mori_dispatch_outputs(
        manual["truncated_output"],
        manual["truncated_indices"],
        manual["truncated_weights"],
        manual["src_pos"],
        manual["truncated_scales"],
    )
    if reorder_out is None:
        raise RuntimeError("Reorder guard triggered; adjust token/expert settings")
    reordered_x, _, reordered_idx, reordered_weights = reorder_out

    torch.cuda.synchronize()
    dist.barrier()

    benchmarks: List[Tuple[str, Callable[[], None]]] = []

    benchmarks.append(
        (
            "transform_dispatch_output_gpu",
            lambda: mori.transform_dispatch_output_gpu(
                manual["full_output"],
                manual["full_indices"],
                buffer.config,
                manual["recv_count"],
                manual["full_scales"],
            ),
        )
    )
    benchmarks.append(
        (
            "_reorder_mori_dispatch_outputs",
            lambda: buffer._reorder_mori_dispatch_outputs(
                manual["truncated_output"],
                manual["truncated_indices"],
                manual["truncated_weights"],
                manual["src_pos"],
                manual["truncated_scales"],
            ),
        )
    )
    benchmarks.append(
        (
            "_revert_mori_dispatch_outputs",
            lambda: buffer._revert_mori_dispatch_outputs(
                reordered_x,
                reordered_idx,
                reordered_weights,
                manual["src_pos"],
            ),
        )
    )
    benchmarks.append(
        (
            "inverse_transform_dispatch_output_gpu",
            lambda: mori.inverse_transform_dispatch_output_gpu(
                transform_cache[0],
                transform_cache[1],
                transform_cache[2],
                manual["recv_count"],
            ),
        )
    )
    if not args.low_latency:

        def standard_pipeline_iteration() -> List[Tuple[str, float]]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            op, inp, inp_scales, dispatch_indices_arg = buffer._preprocess_dispatch(
                tokens,
                topk_idx,
                topk_weights,
                None,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dispatch_output, dispatch_weights_local, dispatch_scales, dispatch_indices, dispatch_recv_num_token = \
                op.dispatch(inp, topk_weights, inp_scales, dispatch_indices_arg)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            recv_x, _, recv_topk_weights, _, dispatch_handle = buffer._postprocess_dispatch(
                op,
                dispatch_output,
                dispatch_weights_local,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
                inp_scales,
                dispatch_indices_arg,
            )
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            combine_src = recv_x[0] if isinstance(recv_x, tuple) else recv_x
            if combine_src.numel() == 0:
                raise RuntimeError("Dispatch produced zero tokens on this rank; adjust token/expert parameters")
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            combine_in, combine_dispatch_indices_arg, combine_weights_local = buffer._preprocess_combine(
                combine_src,
                dispatch_handle,
                recv_topk_weights,
            )
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            combined_x_stage = buffer.ops.combine(
                combine_in,
                combine_weights_local,
                combine_dispatch_indices_arg,
            )
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            buffer._postprocess_combine(combined_x_stage)
            torch.cuda.synchronize()
            t7 = time.perf_counter()

            return [
                ("dispatch.preprocess", t1 - t0),
                ("dispatch.ops", t2 - t1),
                ("dispatch.postprocess", t3 - t2),
                ("combine.preprocess", t5 - t4),
                ("combine.ops", t6 - t5),
                ("combine.postprocess", t7 - t6),
            ]

        breakdown_results: List[Tuple[str, float]] = []
        standard_breakdown = measure_breakdown(standard_pipeline_iteration, args.iters, args.warmup_iters)
        breakdown_results.extend(standard_breakdown)
        standard_totals = aggregate_stage_totals(
            standard_breakdown,
            {
                "dispatch.total": [
                    "dispatch.preprocess",
                    "dispatch.ops",
                    "dispatch.postprocess",
                ],
                "combine.total": [
                    "combine.preprocess",
                    "combine.ops",
                    "combine.postprocess",
                ],
            },
        )
    else:
        breakdown_results = []
        standard_totals = []

    if args.low_latency:
        ll_use_fp8 = False

        def low_latency_pipeline_iteration() -> List[Tuple[str, float]]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dispatch_arg = Buffer._per_token_cast_to_fp8(tokens) if ll_use_fp8 else tokens
            op, inp, inp_scales, dispatch_indices_arg = buffer._preprocess_dispatch(
                dispatch_arg,
                topk_idx,
                topk_weights,
                None,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dispatch_output, dispatch_weights_local, dispatch_scales, dispatch_indices, dispatch_recv_num_token = \
                op.dispatch(inp, topk_weights, inp_scales, dispatch_indices_arg)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            ll_recv_x, _, ll_handle = buffer._postprocess_low_latency_dispatch(
                dispatch_output,
                dispatch_indices,
                dispatch_scales,
                dispatch_recv_num_token,
                dispatch_weights_local,
                dispatch_indices_arg,
                ll_use_fp8,
            )
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            ll_tensor = ll_recv_x if isinstance(ll_recv_x, torch.Tensor) else ll_recv_x[0]
            if ll_tensor.numel() == 0:
                raise RuntimeError("Low-latency dispatch produced zero tokens; adjust gating inputs")

            torch.cuda.synchronize()
            t4 = time.perf_counter()
            rec_output, dispatch_weights_local, dispatch_indices_arg = buffer._preprocess_low_latency_combine(
                ll_tensor,
                ll_handle,
            )
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            buffer.ops.combine(
                rec_output,
                dispatch_weights_local,
                dispatch_indices_arg,
                block_num=buffer.config.block_num,
                warp_per_block=16,
            )
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            t7 = t6
            return [
                ("low_latency_dispatch.preprocess", t1 - t0),
                ("low_latency_dispatch.ops", t2 - t1),
                ("low_latency_dispatch.postprocess", t3 - t2),
                ("low_latency_combine.preprocess", t5 - t4),
                ("low_latency_combine.ops", t6 - t5),
                ("low_latency_combine.postprocess", t7 - t6),
            ]

        ll_breakdown = measure_breakdown(low_latency_pipeline_iteration, args.iters, args.warmup_iters)
        breakdown_results.extend(ll_breakdown)
        ll_totals = aggregate_stage_totals(
            ll_breakdown,
            {
                "low_latency_dispatch.total": [
                    "low_latency_dispatch.preprocess",
                    "low_latency_dispatch.ops",
                    "low_latency_dispatch.postprocess",
                ],
                "low_latency_combine.total": [
                    "low_latency_combine.preprocess",
                    "low_latency_combine.ops",
                    "low_latency_combine.postprocess",
                ],
            },
        )
    else:
        ll_totals = []

    local_results: List[Tuple[str, float]] = []
    for name, fn in benchmarks:
        local_results.append(benchmark_op(name, fn, args.iters, args.warmup_iters))
    local_results.extend(standard_totals)
    local_results.extend(ll_totals)
    local_results.extend(breakdown_results)

    gather_and_print(local_results)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    num_processes = args.num_processes
    print('-------------------------------------------------------------------------', flush=True)
    mp.spawn(run, args=(args,), nprocs=num_processes, join=True)
    print('*************************************************************************', flush=True)


if __name__ == "__main__":
    main()
