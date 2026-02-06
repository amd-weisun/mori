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
from mori import cpp as mori_cpp

from dataclasses import dataclass
import torch
import torch.distributed as dist


class EpDispatchCombineDeepepKernelType(mori_cpp.EpDispatchCombineDeepepKernelType):
    def __str__(self):
        return self.name


@dataclass
class EpDispatchCombineDeepepConfig:
    data_type: torch.dtype
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    use_external_inp_buf: bool = True
    use_fp8: bool = True
    use_deepep_layout: bool = True
    use_weighted_combine: bool = True
    bypass_start_barrier: bool = True  # Set to False for multi-iteration tests
    kernel_type: EpDispatchCombineDeepepKernelType = EpDispatchCombineDeepepKernelType.IntraNode
    gpu_per_node: int = 8
    rdma_block_num: int = 1


def _cpp_dispatch_combine_deepep_factory(entity_name):
    return getattr(mori_cpp, entity_name)


class EpDispatchCombineDeepepOp:
    def __init__(self, config: EpDispatchCombineDeepepConfig):
        self.config = config

        handle_class = _cpp_dispatch_combine_deepep_factory("EpDispatchCombineDeepepHandle")
        self._handle = handle_class(
            mori_cpp.EpDispatchCombineDeepepConfig(
                rank=config.rank,
                world_size=config.world_size,
                hidden_dim=config.hidden_dim,
                scale_dim=config.scale_dim,
                scale_type_size=config.scale_type_size,
                max_token_type_size=config.max_token_type_size,
                max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
                num_experts_per_rank=config.num_experts_per_rank,
                num_experts_per_token=config.num_experts_per_token,
                warp_num_per_block=config.warp_num_per_block,
                block_num=config.block_num,
                use_external_inp_buf=config.use_external_inp_buf,
                use_fp8=config.use_fp8,
                use_deepep_layout=config.use_deepep_layout,
                use_weighted_combine=config.use_weighted_combine,
                bypass_start_barrier=config.bypass_start_barrier,
                gpu_per_node=config.gpu_per_node,
                rdma_block_num=config.rdma_block_num,
            )
        )

        self._dispatch_func = _cpp_dispatch_combine_deepep_factory("launch_dispatch_deepep")
        self._combine_func = _cpp_dispatch_combine_deepep_factory("launch_combine_deepep")
        self._reset_func = _cpp_dispatch_combine_deepep_factory("launch_reset_deepep")
        self._dispatch_ll_func = _cpp_dispatch_combine_deepep_factory(
            "launch_intra_node_dispatch_deepep_ll"
        )
        self._combine_ll_func = _cpp_dispatch_combine_deepep_factory(
            "launch_intra_node_combine_deepep_ll"
        )
        self._dispatch_internode_ll_func = _cpp_dispatch_combine_deepep_factory(
            "launch_inter_node_dispatch_deepep_ll"
        )
        self._combine_internode_ll_func = _cpp_dispatch_combine_deepep_factory(
            "launch_inter_node_combine_deepep_ll"
        )
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_deepep_factory(
            "get_dispatch_src_token_pos_deepep"
        )
        self._get_cur_rank_num_token = _cpp_dispatch_combine_deepep_factory(
            "get_cur_rank_num_token_deepep"
        )
        self._get_dispatch_sender_token_idx_map_func = _cpp_dispatch_combine_deepep_factory(
            "get_dispatch_sender_token_idx_map_deepep"
        )
        self._get_dispatch_receiver_token_idx_map_func = _cpp_dispatch_combine_deepep_factory(
            "get_dispatch_receiver_token_idx_map_deepep"
        )
        self._get_dispatch_dest_tok_id_map_func = _cpp_dispatch_combine_deepep_factory(
            "get_dispatch_dest_tok_id_map_deepep"
        )
        self._get_dispatch_recv_token_count_per_expert = _cpp_dispatch_combine_deepep_factory(
            "get_dispatch_recv_token_count_per_expert_deepep"
        )
        self._get_registered_combine_input_buffer = _cpp_dispatch_combine_deepep_factory(
            "get_registered_combine_input_buffer_deepep"
        )

    def get_registered_combine_input_buffer(self, dtype: torch.dtype):
        return self._get_registered_combine_input_buffer(self._handle, dtype)

    def get_cur_rank_num_token(self):
        return self._get_cur_rank_num_token(self._handle)


    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        raise RuntimeError(
            "DeepEP non-LL dispatch is not implemented yet; use dispatch_deepep_ll()."
        )
        return self._dispatch_func(
            self._handle,
            self.config.kernel_type.value,
            input,
            weights,
            scales,
            indices,
            block_num,
            warp_per_block,
        )

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ):
        raise RuntimeError(
            "DeepEP non-LL combine is not implemented yet; use combine_deepep_ll()."
        )
        output = self._combine_func(
            self._handle,
            self.config.kernel_type.value,
            input,
            weights,
            indices,
            block_num,
            warp_per_block,
        )
        if call_reset:
            self._reset_func(self._handle)
        return output

    def dispatch_deepep_ll(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int | None = None,
        num_experts: int | None = None,
        use_fp8: bool | None = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        *,
        weights: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Low-latency DeepEP dispatch (expert-major layout).

        Inputs:
        - x: [num_tokens, hidden_dim] token embeddings for the current rank.
        - topk_idx: [num_tokens, num_experts_per_token] global expert ids for each token.
        - num_max_dispatch_tokens_per_rank: optional override (defaults to config).
        - num_experts: optional override (defaults to config.world_size * config.num_experts_per_rank).
        - use_fp8: optional override (defaults to config.use_fp8).
        - async_finish/return_recv_hook: reserved (currently ignored, returns None).
        - weights: optional [num_tokens, num_experts_per_token] routing weights.
        - scales: optional [num_tokens, scale_dim] per-token scale data.
          For fp8 mode, scale_dim should be hidden_dim / 128.

        Returns:
        - recv_x: dispatch output; fp8 mode returns (dispatch_output, dispatch_scales), otherwise dispatch_output.
        - recv_count: [num_experts_per_rank] per-expert token counts.
        - handle: internal handle for combine (opaque to users).
        - event: None (reserved for future async support).
        - hook: None (reserved for future async support).
        """
        if num_max_dispatch_tokens_per_rank is None:
            num_max_dispatch_tokens_per_rank = self.config.max_num_inp_token_per_rank
        if num_experts is None:
            num_experts = self.config.world_size * self.config.num_experts_per_rank
        if use_fp8 is None:
            use_fp8 = self.config.use_fp8

        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            _,
        ) = self._dispatch_ll_func(
            self._handle,
            x,
            weights,
            scales,
            topk_idx,
            block_num,
            warp_per_block,
        )
        recv_count = self.get_dispatch_recv_token_count_per_expert()
        recv_x = (dispatch_output, dispatch_scales) if use_fp8 else dispatch_output
        handle = (dispatch_weights, dispatch_indices)
        return recv_x, recv_count, handle, None, None

    def combine_deepep_ll(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple | None = None,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: torch.Tensor | None = None,
        *,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Low-latency DeepEP combine (expert-major dispatch output).

        Inputs:
        - x: expert-major dispatch output buffer (from dispatch_deepep_ll).
        - topk_idx: expert-major dispatch indices buffer (from dispatch_deepep_ll).
        - topk_weights: expert-major dispatch weights buffer (from dispatch_deepep_ll).
        - handle/zero_copy/async_finish/return_recv_hook/out: reserved (currently ignored).

        Returns:
        - combine_output: [max_num_inp_token_per_rank, hidden_dim] combined tokens for the rank.
          If config.use_weighted_combine is true, combines with routing weights; otherwise weights are ignored.
          Input must be bf16 (caller should dequantize fp8 outputs before calling).
        - event: None (reserved for future async support).
        - hook: None (reserved for future async support).
        """
        combine_output, _ = self._combine_ll_func(
            self._handle,
            x,
            topk_weights,
            topk_idx,
            block_num,
            warp_per_block,
        )
        return combine_output, None, None

    def dispatch_internode_deepep_ll(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int | None = None,
        num_experts: int | None = None,
        use_fp8: bool | None = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        *,
        weights: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Inter-node low-latency DeepEP dispatch (expert-major layout).

        Same interface as dispatch_deepep_ll but uses RDMA for cross-node communication.
        """
        if num_max_dispatch_tokens_per_rank is None:
            num_max_dispatch_tokens_per_rank = self.config.max_num_inp_token_per_rank
        if num_experts is None:
            num_experts = self.config.world_size * self.config.num_experts_per_rank
        if use_fp8 is None:
            use_fp8 = self.config.use_fp8

        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            _,
        ) = self._dispatch_internode_ll_func(
            self._handle,
            x,
            weights,
            scales,
            topk_idx,
            block_num,
            warp_per_block,
        )
        recv_count = self.get_dispatch_recv_token_count_per_expert()
        recv_x = (dispatch_output, dispatch_scales) if use_fp8 else dispatch_output
        handle = (dispatch_weights, dispatch_indices)
        return recv_x, recv_count, handle, None, None

    def combine_internode_deepep_ll(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple | None = None,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: torch.Tensor | None = None,
        *,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Inter-node low-latency DeepEP combine (expert-major dispatch output).

        Same interface as combine_deepep_ll but uses RDMA for cross-node communication.
        """
        combine_output, _ = self._combine_internode_ll_func(
            self._handle,
            x,
            topk_weights,
            topk_idx,
            block_num,
            warp_per_block,
        )
        return combine_output, None, None

    def reset(self, sync_barrier: bool = True):
        """Reset buffers between iterations.

        Args:
            sync_barrier: If True (default), launches a cross-device barrier kernel to
                synchronize all ranks before returning. This prevents race conditions
                between buffer resets and RDMA writes from other ranks. Set to False
                only if external synchronization (e.g., dist.barrier()) is provided.
        """
        self._reset_func(self._handle, sync_barrier)

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.config.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()

        if self.config.kernel_type.value in (
            EpDispatchCombineDeepepKernelType.IntraNode.value,
            EpDispatchCombineDeepepKernelType.InterNode.value,
            EpDispatchCombineDeepepKernelType.IntraNodeLL.value,
            EpDispatchCombineDeepepKernelType.InterNodeLL.value,
        ):
            return self._get_dispatch_src_token_pos_func(self._handle)

        dispatch_sender_token_id_map = self._get_dispatch_sender_token_idx_map_func(self._handle)
        dispatch_receiver_token_id_map = self._get_dispatch_receiver_token_idx_map_func(self._handle)

        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        cur_rank_num_token = self._get_cur_rank_num_token(self._handle)
        all_rank_num_token = [torch.empty(1) for _ in range(self.config.world_size)]
        dist.all_gather(all_rank_num_token, torch.Tensor([cur_rank_num_token]))

        reverse_sender_token_id_map = {}
        for r in range(self.config.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : int(all_rank_num_token[r][0].item())
                    * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.config.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )
        src_token_pos = []
        for recv_mapped_id in dispatch_receiver_token_id_map.tolist():
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            if recv_mapped_id not in reverse_sender_token_id_map:
                print(
                    f"Warning: rank {self.config.rank} src_pe {src_pe} max_num_token_to_send_per_rank {max_num_token_to_send_per_rank} recv_mapped_id {recv_mapped_id} not in reverse_sender_token_id_map"
                )
                raise
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            src_token_pos.append(src_pe * max_num_token_to_send_per_rank + src_tok_id)

        return torch.tensor(src_token_pos, dtype=torch.int)

    def get_dispatch_recv_token_count_per_expert(self):
        return self._get_dispatch_recv_token_count_per_expert(self._handle)

    def get_dispatch_dest_tok_id_map(self):
        return self._get_dispatch_dest_tok_id_map_func(self._handle)
