"""Micro-benchmarks for MORI Buffer operations.

Run with torchrun, for example:
  torchrun --nproc_per_node=8 python python/mori/ops/bench_buffer.py --num-tokens 8192 --hidden-dim 4096
"""

import argparse
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

import mori
from mori.ops.Buffer import Buffer

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Buffer dispatch/combine variants and helpers")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="bf16")
    parser.add_argument("--total-experts", type=int, default=32)
    parser.add_argument("--disable-reorder", action="store_true")
    parser.add_argument("--disable-gpu-ll-layout-transform", action="store_true")
    parser.add_argument("--skip-low-latency", action="store_true")
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
    print("\n=== Buffer Benchmark (ms) ===")
    for name, values in summary.items():
        avg = sum(values) / len(values)
        print(f"{name:45s} avg={avg:8.3f} min={min(values):8.3f} max={max(values):8.3f}")


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


def run() -> None:
    args = parse_args()
    ensure_cuda()
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

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

    recv_x, recv_idx, recv_weights, _, dispatch_handle, _ = buffer.dispatch(
        tokens,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )
    combine_input = torch.randn_like(recv_x)
    combine_weights = recv_weights.clone() if isinstance(recv_weights, torch.Tensor) else None

    if args.skip_low_latency:
        ll_recv_tensor = torch.empty(0, device=device, dtype=dtype)
        ll_handle = None
    else:
        ll_recv_x, _, ll_handle, _, _ = buffer.low_latency_dispatch(
            tokens,
            topk_idx,
            max_tokens,
            total_experts,
            use_fp8=False,
            topk_weights=topk_weights,
        )
        ll_recv_tensor = ll_recv_x if isinstance(ll_recv_x, torch.Tensor) else ll_recv_x[0]
        if ll_recv_tensor.numel() == 0:
            raise RuntimeError("Low latency dispatch produced zero tokens; adjust gating inputs")
    ll_compute_input = torch.randn_like(ll_recv_tensor) if ll_recv_tensor.numel() else None

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
            "dispatch",
            lambda: buffer.dispatch(tokens, topk_idx=topk_idx, topk_weights=topk_weights),
        )
    )
    benchmarks.append(
        (
            "combine",
            lambda: buffer.combine(combine_input, dispatch_handle, topk_weights=combine_weights),
        )
    )
    if not args.skip_low_latency:
        benchmarks.append(
            (
                "low_latency_dispatch",
                lambda: buffer.low_latency_dispatch(
                    tokens,
                    topk_idx,
                    max_tokens,
                    total_experts,
                    use_fp8=False,
                    topk_weights=topk_weights,
                ),
            )
        )
        benchmarks.append(
            (
                "low_latency_combine",
                lambda: buffer.low_latency_combine(
                    ll_compute_input,
                    topk_idx,
                    topk_weights,
                    ll_handle,
                ),
            )
        )

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

    def dispatch_iteration() -> List[Tuple[str, float]]:
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
        buffer._postprocess_dispatch(
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
        return [
            ("dispatch.preprocess", t1 - t0),
            ("dispatch.ops", t2 - t1),
            ("dispatch.postprocess", t3 - t2),
        ]

    def combine_iteration() -> List[Tuple[str, float]]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        combine_in, dispatch_indices_arg, combine_weights_local = buffer._preprocess_combine(
            combine_input,
            dispatch_handle,
            combine_weights,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        combined_x_stage = buffer.ops.combine(combine_in, combine_weights_local, dispatch_indices_arg)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        buffer._postprocess_combine(combined_x_stage)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return [
            ("combine.preprocess", t1 - t0),
            ("combine.ops", t2 - t1),
            ("combine.postprocess", t3 - t2),
        ]

    breakdown_results: List[Tuple[str, float]] = []
    breakdown_results.extend(measure_breakdown(dispatch_iteration, args.iters, args.warmup_iters))
    breakdown_results.extend(measure_breakdown(combine_iteration, args.iters, args.warmup_iters))

    if not args.skip_low_latency:
        ll_use_fp8 = False

        def low_latency_dispatch_iteration() -> List[Tuple[str, float]]:
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
            buffer._postprocess_low_latency_dispatch(
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
            return [
                ("low_latency_dispatch.preprocess", t1 - t0),
                ("low_latency_dispatch.ops", t2 - t1),
                ("low_latency_dispatch.postprocess", t3 - t2),
            ]

        def low_latency_combine_iteration() -> List[Tuple[str, float]]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rec_output, dispatch_weights_local, dispatch_indices_arg = buffer._preprocess_low_latency_combine(
                ll_compute_input,
                ll_handle,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            buffer.ops.combine(
                rec_output,
                dispatch_weights_local,
                dispatch_indices_arg,
                block_num=buffer.config.block_num,
                warp_per_block=16,
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t3 = t2
            return [
                ("low_latency_combine.preprocess", t1 - t0),
                ("low_latency_combine.ops", t2 - t1),
                ("low_latency_combine.postprocess", t3 - t2),
            ]

        breakdown_results.extend(measure_breakdown(low_latency_dispatch_iteration, args.iters, args.warmup_iters))
        if ll_compute_input is not None:
            breakdown_results.extend(measure_breakdown(low_latency_combine_iteration, args.iters, args.warmup_iters))

    local_results: List[Tuple[str, float]] = []
    for name, fn in benchmarks:
        local_results.append(benchmark_op(name, fn, args.iters, args.warmup_iters))
    local_results.extend(breakdown_results)

    gather_and_print(local_results)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    with torch.inference_mode():
        run()


if __name__ == "__main__":
    main()
