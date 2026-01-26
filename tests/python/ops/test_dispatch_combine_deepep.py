# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pytest
import os
import sys

import mori
import traceback
from multiprocessing import Queue

_tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
from utils import TorchDistContext, get_free_port
import torch
import torch.distributed as dist


# Default to isolation mode for shmem if not explicitly set
if "MORI_SHMEM_MODE" not in os.environ:
    os.environ["MORI_SHMEM_MODE"] = "isolation"

SKIP_CHECKS = os.getenv("MORI_DEEPEP_SKIP_CHECKS", "0") == "1"
SKIP_DISPATCH_CHECKS = SKIP_CHECKS or os.getenv("MORI_DEEPEP_SKIP_DISPATCH_CHECKS", "0") == "1"
SKIP_COMBINE_CHECKS = SKIP_CHECKS or os.getenv("MORI_DEEPEP_SKIP_COMBINE_CHECKS", "0") == "1"
DISPATCH_CHECK_MODE = os.getenv("MORI_DEEPEP_DISPATCH_CHECK_MODE", "full")
MAX_DISPATCH_EXPERTS_CHECK = int(os.getenv("MORI_DEEPEP_MAX_DISPATCH_EXPERTS_CHECK", "-1"))
MAX_DISPATCH_SLOTS_CHECK = int(os.getenv("MORI_DEEPEP_MAX_DISPATCH_SLOTS_CHECK", "-1"))
MAX_COMBINE_TOKENS_CHECK = int(os.getenv("MORI_DEEPEP_MAX_COMBINE_TOKENS_CHECK", "-1"))
DEBUG_LOG = os.getenv("MORI_DEEPEP_DEBUG_LOG", "0") == "1"


def _log(msg: str, force: bool = False):
    if force or DEBUG_LOG:
        print(msg, flush=True)


class PerRankTorchDistProcessManager:
    def __init__(self, init_mori_shmem=True):
        self.task_queues = []
        self.result_queue = Queue()
        self.processes = []
        self.init_mori_shmem = init_mori_shmem

    @staticmethod
    def _worker(rank, world_size, port, init_shmem, task_queue, result_queue):
        print(f"[TorchDist] rank {rank} starting worker", flush=True)
        with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
            print(f"[TorchDist] rank {rank} process group ready", flush=True)
            if init_shmem:
                print(f"[TorchDist] rank {rank} shmem init", flush=True)
                mori.shmem.shmem_torch_process_group_init("default")
                print(f"[TorchDist] rank {rank} shmem ready", flush=True)
            while True:
                task = task_queue.get()
                if task == "STOP":
                    if init_shmem:
                        mori.shmem.shmem_finalize()
                    break
                func, args = task
                print(f"[TorchDist] rank {rank} running task", flush=True)
                try:
                    result = func(rank, *args)
                    result_queue.put((rank, result))
                except Exception:
                    result_queue.put((rank, traceback.format_exc()))

    def start_workers(self, world_size):
        port = get_free_port()
        self.task_queues = [Queue() for _ in range(world_size)]
        self.processes = [
            torch.multiprocessing.Process(
                target=self._worker,
                args=(
                    rank,
                    world_size,
                    port,
                    self.init_mori_shmem,
                    self.task_queues[rank],
                    self.result_queue,
                ),
            )
            for rank in range(world_size)
        ]
        for p in self.processes:
            p.start()

    def shutdown(self):
        for queue in self.task_queues:
            queue.put("STOP")
        for p in self.processes:
            p.join()


class EpDispatchCombineDeepepTestCase:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda", self.config.rank)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(123)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier()

    def dequant_dispatch_output(self, dispatch_output, dispatch_scales):
        assert dispatch_scales is not None
        num_scales = self.config.hidden_dim // 128
        e, c, _ = dispatch_output.shape
        output_fp32 = dispatch_output.float().view(e, c, num_scales, 128)
        scales_fp32 = dispatch_scales.float().view(e, c, num_scales, 1)
        return (output_fp32 * scales_fp32).view(e, c, self.config.hidden_dim).to(torch.bfloat16)

    def dequant_input_like_fp8(self, inputs: torch.Tensor) -> torch.Tensor:
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
        return dequant.view(inputs.size(0), inputs.size(1)).to(self.config.data_type)

    def gen_test_data(self, use_max_token_num=False):
        if use_max_token_num:
            num_token = torch.tensor(
                [
                    self.config.max_num_inp_token_per_rank
                    for _ in range(self.config.world_size)
                ]
            ).to(self.device)
        else:
            num_token = torch.randint(
                0,
                self.config.max_num_inp_token_per_rank + 1,
                [self.config.world_size],
                generator=self.rng,
                device=self.device,
            )

        all_rank_indices = []
        for r in range(self.config.world_size):
            indices = torch.empty(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.int64,
            )
            for i in range(num_token[r]):
                perm = torch.randperm(
                    self.config.num_experts_per_rank * self.config.world_size,
                    generator=self.rng,
                    device=self.device,
                )
                indices[i] = perm[: self.config.num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(self.device))

        all_rank_weights = [
            torch.ones(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.float32,
                device=self.device,
            )
            for r in range(self.config.world_size)
        ]

        all_rank_scales = [
            torch.rand(
                num_token[r],
                self.config.scale_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.config.world_size)
        ]
        if self.config.scale_type_size == 1:
            all_rank_scales = [t.to(torch.float8_e4m3fnuz) for t in all_rank_scales]

        all_rank_input = []
        for r in range(self.config.world_size):
            # token_ids = torch.arange(num_token[r], device=self.device, dtype=torch.float32)
            # base = r * self.config.max_num_inp_token_per_rank
            # values = (base + token_ids).view(-1, 1)
            # all_rank_input.append(values.repeat(1, self.config.hidden_dim).to(self.config.data_type))
            # Use uniform distribution in [-1, 1] to avoid FP8 quantization issues with outliers
            all_rank_input.append(
                (torch.rand(
                    num_token[r],
                    self.config.hidden_dim,
                    dtype=torch.float32,
                    generator=self.rng,
                    device=self.device,
                ) * 2 - 1).to(self.config.data_type)
            )

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def check_dispatch_result(
        self,
        op,
        test_data,
        dispatch_output,
        dispatch_weights,
        dispatch_scales,
        dispatch_indices,
        dispatch_recv_num_token,
        use_ll=False,
    ):
        _log(f"[DeepEP] rank {self.config.rank} check_dispatch_result start")
        # if self.config.rank != 0:
        #     return
        if use_ll and DISPATCH_CHECK_MODE == "counts":
            if dispatch_recv_num_token.ndim == 0:
                expected = int(dispatch_recv_num_token.item())
            else:
                expected = int(dispatch_recv_num_token[0].item())
            recv_count = op.get_dispatch_recv_token_count_per_expert()
            expected_from_counts = int(recv_count.sum().item())
            expected_from_indices = 0
            rank_begin = self.config.rank * self.config.num_experts_per_rank
            rank_end = rank_begin + self.config.num_experts_per_rank
            for r in range(self.config.world_size):
                idx = test_data[1][r]
                expected_from_indices += int(((idx >= rank_begin) & (idx < rank_end)).sum().item())
            assert expected_from_counts == expected, (
                f"dispatch_recv_num_token={expected} expected_from_counts={expected_from_counts} "
                f"expected_from_indices={expected_from_indices} rank={self.config.rank}"
            )
            _log(f"[DeepEP] rank {self.config.rank} check_dispatch_result end")
            return
        torch.cuda.synchronize()
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        src_token_pos = op.get_dispatch_src_token_pos().cpu()
        if self.config.use_fp8:
            dispatch_weights = dispatch_weights.cpu() if dispatch_weights is not None else None
            dispatch_scales_device = dispatch_scales
            dispatch_scales = dispatch_scales.cpu() if dispatch_scales is not None else None
            dispatch_indices = dispatch_indices.cpu()
            dequant_output = self.dequant_dispatch_output(dispatch_output, dispatch_scales_device)
            dispatch_output_cpu = dequant_output.cpu()
        else:
            dispatch_output = dispatch_output.cpu()
            dispatch_weights = dispatch_weights.cpu() if dispatch_weights is not None else None
            dispatch_scales = dispatch_scales.cpu() if dispatch_scales is not None else None
            dispatch_indices = dispatch_indices.cpu()
            dispatch_output_cpu = dispatch_output
        _log(f"[DeepEP] rank {self.config.rank} check_dispatch_result after cpu")

        valid_mask = src_token_pos >= 0
        valid_positions = torch.nonzero(valid_mask, as_tuple=False)
        if dispatch_recv_num_token.ndim == 0:
            expected = int(dispatch_recv_num_token.item())
        else:
            expected = int(dispatch_recv_num_token[0].item())
        actual = int(valid_positions.size(0))
        recv_count = op.get_dispatch_recv_token_count_per_expert().cpu()
        expected_from_counts = int(recv_count.sum().item())
        expected_from_indices = 0
        rank_begin = self.config.rank * self.config.num_experts_per_rank
        rank_end = rank_begin + self.config.num_experts_per_rank
        for r in range(self.config.world_size):
            idx = all_rank_indices[r]
            expected_from_indices += int(((idx >= rank_begin) & (idx < rank_end)).sum().item())
        if use_ll:
            assert expected_from_counts == expected, (
                f"dispatch_recv_num_token={expected} expected_from_counts={expected_from_counts} "
                f"expected_from_indices={expected_from_indices} rank={self.config.rank} "
                f"use_fp8={self.config.use_fp8} num_experts_per_rank={self.config.num_experts_per_rank} "
                f"max_num_inp_token_per_rank={self.config.max_num_inp_token_per_rank}"
            )
        else:
            assert actual == expected, (
                f"valid_positions={actual} dispatch_recv_num_token={expected} "
                f"expected_from_indices={expected_from_indices} "
                f"expected_from_counts={expected_from_counts} "
                f"rank={self.config.rank} use_fp8={self.config.use_fp8} "
                f"num_experts_per_rank={self.config.num_experts_per_rank} "
                f"max_num_inp_token_per_rank={self.config.max_num_inp_token_per_rank}"
            )

        if not use_ll:
            for expert_id, slot_id in valid_positions.tolist():
                pos = int(src_token_pos[expert_id, slot_id])
                src_rank = pos // self.config.max_num_inp_token_per_rank
                src_id = pos % self.config.max_num_inp_token_per_rank
                if self.config.use_fp8:
                    assert torch.allclose(
                        all_rank_input[src_rank][src_id],
                        dequant_output[expert_id, slot_id],
                        atol=1e-2,
                        rtol=1e-2,
                    )
                else:
                    assert torch.equal(
                        all_rank_input[src_rank][src_id].cpu(),
                        dispatch_output[expert_id, slot_id],
                    )
                if dispatch_weights is not None:
                    assert torch.equal(
                        all_rank_weights[src_rank][src_id].cpu(),
                        dispatch_weights[expert_id, slot_id],
                    )
                if dispatch_scales is not None and not self.config.use_fp8:
                    assert torch.equal(
                        all_rank_scales[src_rank][src_id].cpu(),
                        dispatch_scales[expert_id, slot_id],
                    )
                assert torch.equal(
                    all_rank_indices[src_rank][src_id].cpu(),
                    dispatch_indices[expert_id, slot_id],
                )
        else:
            # LL path: use src_token_pos map to validate indices/weights deterministically.
            max_experts_to_check = (
                self.config.num_experts_per_rank
                if MAX_DISPATCH_EXPERTS_CHECK < 0
                else max(1, MAX_DISPATCH_EXPERTS_CHECK)
            )
            max_slots_to_check = (
                self.config.world_size * self.config.max_num_inp_token_per_rank
                if MAX_DISPATCH_SLOTS_CHECK < 0
                else max(1, MAX_DISPATCH_SLOTS_CHECK)
            )
            for local_expert in range(min(self.config.num_experts_per_rank, max_experts_to_check)):
                count = int(recv_count[local_expert].item())
                if count == 0:
                    continue
                count = min(count, max_slots_to_check)
                global_expert = self.config.rank * self.config.num_experts_per_rank + local_expert
                for s in range(count):
                    pos = int(src_token_pos[local_expert, s].item())
                    assert pos >= 0, f"Invalid src token pos for expert={global_expert} slot={s}"
                    src_rank = pos // self.config.max_num_inp_token_per_rank
                    src_id = pos % self.config.max_num_inp_token_per_rank
                    expected_idx = all_rank_indices[src_rank][src_id].cpu()
                    got_idx = dispatch_indices[local_expert, s]
                    if not torch.equal(expected_idx, got_idx):
                        _log(
                            f"[DeepEP] dispatch idx mismatch expert={global_expert} slot={s} "
                            f"src_rank={src_rank} src_id={src_id} expected={expected_idx.tolist()} "
                            f"got={got_idx.tolist()}",
                            force=True,
                        )
                    assert torch.equal(expected_idx, got_idx)
                    if dispatch_weights is not None:
                        assert torch.allclose(
                            all_rank_weights[src_rank][src_id].cpu(),
                            dispatch_weights[local_expert, s],
                            atol=1e-5,
                            rtol=1e-5,
                        )
        _log(f"[DeepEP] rank {self.config.rank} check_dispatch_result end")

    def check_combine_result(
        self,
        op,
        test_data,
        combine_output,
        combine_output_weight=None,
        dispatch_output=None,
        dispatch_scales=None,
    ):
        if self.config.rank != 0:
            return
        _log(f"[DeepEP] rank {self.config.rank} check_combine_result start")
        
        all_rank_num_token = test_data[0]
        all_rank_indices = test_data[1]
        all_rank_input = test_data[2]
        all_rank_weights = test_data[3]

        max_tokens_to_check = (
            all_rank_num_token[self.config.rank].item()
            if MAX_COMBINE_TOKENS_CHECK < 0
            else min(
                all_rank_num_token[self.config.rank].item(),
                max(1, MAX_COMBINE_TOKENS_CHECK),
            )
        )
        for i in range(max_tokens_to_check):
            base_input = all_rank_input[self.config.rank]
            if self.config.use_fp8:
                base_input = self.dequant_input_like_fp8(base_input)
            got = combine_output[i]
            if self.config.use_weighted_combine:
                weights = all_rank_weights[self.config.rank][i].to(torch.float32)
                expected = torch.zeros_like(got)
                for k in range(weights.numel()):
                    expected = (expected + (base_input[i].to(torch.float32) * weights[k])).to(
                        self.config.data_type
                    )
            else:
                expected = torch.zeros_like(got)
                for _ in range(self.config.num_experts_per_token):
                    expected = (expected + base_input[i].to(torch.float32)).to(self.config.data_type)
            atol = 1e-2
            rtol = 1e-2
            if self.config.use_fp8:
                # FP8 quantization introduces ~1-2 LSB error in BF16 representation
                # which accumulates when summing topk experts
                atol = 0.25
                rtol = 0.25
            if not torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol):
                tok = i
                got_row = got.float()[:8].cpu().tolist()
                exp_row = expected.float()[:8].cpu().tolist()
                expert_capacity = self.config.world_size * self.config.max_num_inp_token_per_rank
                dest_tok_map = op.get_dispatch_dest_tok_id_map().cpu()
                dest_tok_ids = dest_tok_map[tok].tolist()
                decoded = []
                for k, val in enumerate(dest_tok_ids):
                    dest_expert = val // expert_capacity
                    dest_local_tok = val - dest_expert * expert_capacity
                    decoded.append((k, int(dest_expert), int(dest_local_tok)))
                topk = all_rank_indices[self.config.rank][tok].cpu().tolist()
                input_row = all_rank_input[self.config.rank][tok].cpu().tolist()
                weights_row = all_rank_weights[self.config.rank][tok].cpu().tolist()
                local_debug = []
                if dispatch_output is not None:
                    if self.config.use_fp8 and dispatch_scales is not None:
                        dispatch_out_cpu = self.dequant_dispatch_output(
                            dispatch_output, dispatch_scales
                        ).cpu()
                    else:
                        dispatch_out_cpu = dispatch_output.cpu()
                    for k, dest_expert, dest_local_tok in decoded:
                        dest_pe = dest_expert // self.config.num_experts_per_rank
                        local_expert = dest_expert % self.config.num_experts_per_rank
                        if dest_pe == self.config.rank:
                            val = dispatch_out_cpu[local_expert, dest_local_tok, 0].item()
                            local_debug.append((k, dest_expert, dest_local_tok, val))
                _log(
                    f"[DeepEP] combine mismatch token={tok} got={got_row} expected={exp_row}",
                    force=True,
                )
                _log(
                    f"[DeepEP] token {tok} topk_experts={topk} dest_tok_ids={decoded}",
                    force=True,
                )
                _log(
                    f"[DeepEP] input_row={input_row}",
                    force=True,
                )
                _log(
                    f"[DeepEP] weights_row={weights_row}",
                    force=True,
                )
                if local_debug:
                    _log(f"[DeepEP] local dispatch slots={local_debug}", force=True)
                assert False

            if combine_output_weight is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.config.rank][i] * multiplier,
                )
                assert torch.allclose(got_weight, expected_weight, atol=1e-1, rtol=1e-1)
        _log(f"[DeepEP] rank {self.config.rank} check_combine_result end")

    def run_test_once(self, op, test_data, use_ll=False, debug=False):
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        if use_ll:
            if self.config.rank == 0:
                _log("[DeepEP] start dispatch", force=True)
            total_experts = self.config.world_size * self.config.num_experts_per_rank
            recv_x, recv_count, handle, _, _ = op.dispatch_deepep_ll(
                all_rank_input[self.config.rank],
                all_rank_indices[self.config.rank],
                num_max_dispatch_tokens_per_rank=self.config.max_num_inp_token_per_rank,
                num_experts=total_experts,
                use_fp8=self.config.use_fp8,
                weights=all_rank_weights[self.config.rank],
                scales=all_rank_scales[self.config.rank],
                block_num=self.config.block_num,
                warp_per_block=self.config.warp_num_per_block,
            )
            if self.config.rank == 0:
                _log("[DeepEP] dispatch complete", force=True)
            if self.config.use_fp8:
                dispatch_output, dispatch_scales = recv_x
            else:
                dispatch_output, dispatch_scales = recv_x, None
            dispatch_weights, dispatch_indices = handle
            dispatch_recv_num_token = recv_count.sum().to(torch.int32)
        else:
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.config.rank],
                all_rank_weights[self.config.rank],
                all_rank_scales[self.config.rank],
                all_rank_indices[self.config.rank],
            )
        _log(f"[DeepEP] rank {self.config.rank} enter dispatch barrier")
        self.sync()
        _log(f"[DeepEP] rank {self.config.rank} exit dispatch barrier")

        if use_ll:
            if self.config.rank == 0:
                _log("[DeepEP] start combine", force=True)
            _log(f"[DeepEP] rank {self.config.rank} ready for combine")
            combine_input = dispatch_output
            if self.config.use_fp8:
                _log(f"[DeepEP] rank {self.config.rank} dequant combine input")
                combine_input = self.dequant_dispatch_output(dispatch_output, dispatch_scales)
            _log(f"[DeepEP] rank {self.config.rank} enter combine call")
            combine_output, _, _ = op.combine_deepep_ll(
                combine_input, dispatch_indices, dispatch_weights, handle=handle
            )
            _log(f"[DeepEP] rank {self.config.rank} exit combine call")
            if self.config.rank == 0:
                _log("[DeepEP] combine complete", force=True)
            combine_output_weight = None
        else:
            combine_output, combine_output_weight = op.combine(
                dispatch_output, dispatch_weights, dispatch_indices, call_reset=False
            )
        _log(f"[DeepEP] rank {self.config.rank} enter combine barrier")
        self.sync()
        _log(f"[DeepEP] rank {self.config.rank} exit combine barrier")

        # Postpone heavy CPU validation until after dispatch+combine workflow completes.
        if not SKIP_DISPATCH_CHECKS:
            _log(f"[DeepEP] rank {self.config.rank} enter dispatch check")
            self.check_dispatch_result(
                op,
                test_data,
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
                use_ll=use_ll,
            )
            _log(f"[DeepEP] rank {self.config.rank} exit dispatch check")
            if self.config.rank == 0:
                _log("[DeepEP] dispatch check complete", force=True)

        if not SKIP_COMBINE_CHECKS:
            if self.config.rank == 0:
                _log("[DeepEP] start combine check", force=True)
            self.check_combine_result(
                op,
                test_data,
                combine_output,
                combine_output_weight,
                dispatch_output=dispatch_output,
                dispatch_scales=dispatch_scales,
            )
            if self.config.rank == 0:
                _log("[DeepEP] combine check complete", force=True)
        op.reset()


@pytest.fixture(scope="session")
def torch_dist_process_manager():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to spawn")
    except RuntimeError:
        pass
    manager = PerRankTorchDistProcessManager()
    manager.start_workers(world_size=8)
    yield manager
    manager.shutdown()


def _test_dispatch_combine_deepep(
    rank,
    world_size,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    total_experts,
    num_experts_per_token,
    use_ll,
    debug=False,
):
    assert total_experts % world_size == 0
    if rank == 0:
        print("[DeepEP] test worker start", flush=True)
    num_experts_per_rank = total_experts // world_size
    gpu_per_node = 1 if world_size == 1 else 8
    use_fp8 = hidden_dim >= 128
    config = mori.ops.EpDispatchCombineDeepepConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=hidden_dim // 128,
        scale_type_size=scale_type_size,
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
        kernel_type=mori.ops.EpDispatchCombineDeepepKernelType.IntraNode,
        gpu_per_node=gpu_per_node,
    )
    op = mori.ops.EpDispatchCombineDeepepOp(config)
    test_case = EpDispatchCombineDeepepTestCase(config)
    test_data = test_case.gen_test_data(use_max_token_num=True)
    # Ensure all ranks finish setup before dispatch to avoid uninitialized symmetric buffers.
    dist.barrier()
    if rank == 0:
        print("[DeepEP] test data generated", flush=True)
    if not use_ll:
        pytest.skip("DeepEP non-LL dispatch/combine not implemented yet")
    test_case.run_test_once(op, test_data, use_ll=use_ll, debug=debug)
    if rank == 0:
        print("[DeepEP] test worker done", flush=True)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (7168,))
@pytest.mark.parametrize("scale_dim", (56,))
@pytest.mark.parametrize("scale_type_size", (4,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (128,))
@pytest.mark.parametrize("total_experts", (288,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("use_ll", (True,))
def test_dispatch_combine_deepep(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    total_experts,
    num_experts_per_token,
    use_ll,
):
    for rank in range(world_size):
        torch_dist_process_manager.task_queues[rank].put(
            (
                _test_dispatch_combine_deepep,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    scale_dim,
                    scale_type_size,
                    max_num_inp_token_per_rank,
                    total_experts,
                    num_experts_per_token,
                    use_ll,
                    False,
                ],
            )
        )

    results = []
    for _ in range(world_size):
        rank, result = torch_dist_process_manager.result_queue.get()
        results.append(result)

    for result in results:
        if result is not None:
            pytest.assume(False, result)

    print(
        "[DeepEP] test_dispatch_combine_deepep passed "
        f"(world_size={world_size}, total_experts={total_experts}, "
        f"tokens_per_rank={max_num_inp_token_per_rank}, "
        f"num_experts_per_token={num_experts_per_token})",
        flush=True,
    )


def test_dispatch_combine_deepep_debug_minimal(torch_dist_process_manager):
    # Use a dedicated 1-rank process manager to avoid world_size mismatch with the session manager.
    world_size = 1
    data_type = torch.bfloat16
    hidden_dim = 256
    scale_dim = 2
    scale_type_size = 4
    max_num_inp_token_per_rank = 4
    total_experts = 4
    num_experts_per_token = 1
    use_ll = True
    manager = PerRankTorchDistProcessManager()
    manager.start_workers(world_size=world_size)
    try:
        for rank in range(world_size):
            manager.task_queues[rank].put(
                (
                    _test_dispatch_combine_deepep,
                    [
                        world_size,
                        data_type,
                        hidden_dim,
                        scale_dim,
                        scale_type_size,
                        max_num_inp_token_per_rank,
                        total_experts,
                        num_experts_per_token,
                        use_ll,
                        True,
                    ],
                )
            )

        results = []
        for _ in range(world_size):
            rank, result = manager.result_queue.get()
            results.append(result)

        for result in results:
            if result is not None:
                pytest.assume(False, result)
        print(
            "[DeepEP] test_dispatch_combine_deepep_debug_minimal passed "
            f"(world_size={world_size}, total_experts={total_experts}, "
            f"tokens_per_rank={max_num_inp_token_per_rank}, "
            f"num_experts_per_token={num_experts_per_token})",
            flush=True,
        )
    finally:
        manager.shutdown()
