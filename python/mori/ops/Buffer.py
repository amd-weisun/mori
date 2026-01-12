import logging
import mori
import os
import socket
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

from mori import shmem
from mori.ops.dispatch_combine import EpDispatchCombineOp, EpDispatchCombineConfig, EpDispatchCombineKernelType

# Mock classes to maintain API compatibility
class Config:
    def __init__(self, num_sms: int, *args):
        self.num_sms = num_sms

class EventHandle:
    def __init__(self, event=None):
        self.event = event

class EventOverlap:
    def __init__(self, event=None, tensors=None):
        self.event = event
        self.tensors = tensors

logger = logging.getLogger(__name__)


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model.
    Re-implemented using MORI backend.
    """

    num_sms: int = 20
    _printed_warnings = set()

    @classmethod
    def _log_warning_once(cls, msg: str):
        if msg not in cls._printed_warnings:
            print(msg, flush=True)
            cls._printed_warnings.add(msg)

    def __init__(self, group: dist.ProcessGroup,
                 num_nvl_bytes: int = 0, num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False, num_qps_per_rank: int = 1, max_num_inp_token_per_rank : int = 128, gpu_per_node: Optional[int] = None,
                 num_experts_per_token : int = 1,
                 group_name: str = "default",
                 use_gpu_ll_layout_transform : bool = True,
                 reorder: bool = True,
                 block_num: int = 32,
                 warp_num_per_block: int = 16,
                 rdma_block_num: int = 16) -> None:
        """
        Initialize the communication buffer.

        Note: MORI relies on `mori.shmem` shared memory state. Call
        before constructing the buffer so that `shmem_torch_process_group_init`
        has been invoked. The constructor will trigger the initialization if needed.
        """
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank
        self.gpu_per_node = gpu_per_node  # Will infer from `group` if not provided
        self.world_size = dist.get_world_size(group=group)
        self.max_num_inp_token_per_rank = max_num_inp_token_per_rank
        self.num_experts_per_token = num_experts_per_token
        # Cache for MORI ops
        self.group_name = group_name
        self.block_num = block_num
        self.rdma_block_num = rdma_block_num
        self.warp_num_per_block = warp_num_per_block
        self.ops = None
        self.use_gpu_ll_layout_transform = use_gpu_ll_layout_transform
        self.config = None
        self.reorder = reorder # Whether to reorder outputs to match DeepEp API
        self.setup()

    def _get_op(self, dtype: torch.dtype, hidden_dim: int, scale_dim: int = 0) -> EpDispatchCombineOp:
        if self.ops is None:
            kernel_type = EpDispatchCombineKernelType.IntraNode
            # Simple heuristic for kernel type
            if self.group_size > 8: 
                 kernel_type = EpDispatchCombineKernelType.InterNodeV1
            
            if self.low_latency_mode:
                 kernel_type = EpDispatchCombineKernelType.InterNodeV1LL

            # TODO: These parameters might need to be tuned or exposed
            self.config = EpDispatchCombineConfig(
                data_type=dtype,
                rank=self.rank,
                world_size=self.group_size,
                hidden_dim=hidden_dim,
                scale_dim=scale_dim,
                scale_type_size=4 if scale_dim > 0 else 1,
                max_token_type_size=4, 
                max_num_inp_token_per_rank=self.max_num_inp_token_per_rank, # Increased limit
                num_experts_per_rank=self.num_qps_per_rank, # Default assumption
                num_experts_per_token=self.num_experts_per_token, # topK
                warp_num_per_block=self.warp_num_per_block,
                block_num=self.block_num,
                kernel_type=kernel_type,
                gpu_per_node=self.gpu_per_node,
                rdma_block_num=self.rdma_block_num,
            )
            self.ops = EpDispatchCombineOp(self.config)
        
        return self.ops

    def _infer_gpu_per_node(self) -> int:
        if not dist.is_initialized():
            return 1
        try:
            if self.group is None or self.world_size <= 0:
                return 1
            local_host = socket.gethostname()
            hostnames = [None] * self.world_size
            dist.all_gather_object(hostnames, local_host, group=self.group)
            same_host_count = sum(1 for hostname in hostnames if hostname == local_host)
            return max(1, same_host_count)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to infer GPUs per node from hostname: %s", exc)
            return 1

    def setup(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
            )

        if self.gpu_per_node is None:
            self.gpu_per_node = self._infer_gpu_per_node()



        local_rank = self.rank % max(1, self.gpu_per_node)
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        world_group = torch.distributed.group.WORLD
        assert world_group is not None


        # Explicitly set the name if possible, or just register
        try:
            torch._C._distributed_c10d._register_process_group(self.group_name, world_group)
        except RuntimeError:
            logger.info(
                "Process group '%s' already registered, reusing existing group",
                self.group_name,
            )
        try:
            mori.shmem.shmem_torch_process_group_init(self.group_name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to initialize MORI shmem for group %s", self.group_name)
            raise RuntimeError(
                f"Unable to initialize MORI shmem for group '{self.group_name}'"
            ) from exc
 

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(999)


    def reset(self):
        for op in self.ops.values():
            op.reset()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        # MORI handles buffer sizing internally
        return 0

    def get_local_buffer_tensor(self, dtype: torch.dtype, size: Optional[torch.Size] = None,
                                offset: int = 0, use_rdma_buffer: bool = False) -> torch.Tensor:
        # Not directly supported by MORI in the same way
        # We could potentially use op.get_registered_combine_input_buffer if we knew the config
        raise NotImplementedError("get_local_buffer_tensor is not supported in MORI backend yet.")

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        return Config(Buffer.num_sms)

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        return Config(Buffer.num_sms)

    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.
        In MORI, layout calculation is integrated into dispatch.
        Returning dummy values to satisfy API.
        """
        num_tokens = topk_idx.size(0)
        num_ranks = self.group_size
        
        return (torch.zeros(num_ranks, dtype=torch.int, device=topk_idx.device),
                None,
                torch.zeros(num_experts, dtype=torch.int, device=topk_idx.device),
                torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device=topk_idx.device),
                EventOverlap())

    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
                  Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        if topk_idx is None or topk_weights is None:
             raise NotImplementedError("dispatch with handle (cached layout) is not fully supported yet. Please provide topk_idx and topk_weights.")
        
        if isinstance(x, tuple):
            inp, inp_scales = x
            dtype = inp.dtype
            # tmp solution: convert float8 to float8uz

            hidden_dim = inp.size(1)
            scale_dim = inp_scales.size(1) if inp_scales is not None else 0
        else:
            inp = x
            inp_scales = None
            dtype = inp.dtype
            hidden_dim = inp.size(1)
            scale_dim = 0

        self.num_experts_per_token = topk_idx.size(1)
        max_num_tokens_per_rank = max(num_tokens_per_rank) if num_tokens_per_rank is not None else inp.size(0)
        self.max_num_inp_token_per_rank = max(max_num_tokens_per_rank, inp.size(0)) 


        if dtype == torch.float8_e4m3fn:
            if self.rank == 0:
                Buffer._log_warning_once("[warning] Converting float8_e4m3fn input to float8_e4m3fnuz for MORI dispatch, workaround for debugging on MI300X.")
            inp = inp.to(torch.float8_e4m3fnuz)
            dtype = torch.float8_e4m3fnuz
            # if self.rank == 0:    
            #     print(f"[warning] converted inp = {inp}.", flush=True)
        op = self._get_op(dtype, hidden_dim, scale_dim)
        
        

        # MORI dispatch expects int32 indices.
        dispatch_indices_arg = topk_idx.to(dtype=torch.int32)
        
        dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, dispatch_recv_num_token = \
            op.dispatch(inp, topk_weights, inp_scales, dispatch_indices_arg)

        dispatch_indices_clone = dispatch_indices

        def _normalize_recv_num_token(value):
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return 0
                return int(value.view(-1)[0].item())
            if isinstance(value, (list, tuple)) and value:
                return int(value[0])
            return int(value)

        num_valid_tokens = max(0, _normalize_recv_num_token(dispatch_recv_num_token))
        #num_valid_tokens = op.get_cur_rank_num_token()
        def _truncate(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            return tensor[:num_valid_tokens]

        dispatch_output = _truncate(dispatch_output)
        dispatch_weights = _truncate(dispatch_weights)
        dispatch_scales = _truncate(dispatch_scales)
        dispatch_indices = _truncate(dispatch_indices)
        src_token_pos = op.get_dispatch_src_token_pos()[:num_valid_tokens]
        src_token_pos = _truncate(src_token_pos)
        # reorder to match DeepEp order
        if self.reorder:
            dispatch_output, dispatch_scales, dispatch_indices, dispatch_weights = \
                self._reorder_mori_dispatch_outputs(dispatch_output, dispatch_indices, dispatch_weights, src_token_pos, dispatch_scales)
        
        # Construct return values
        recv_x = (dispatch_output, dispatch_scales) if inp_scales is not None else dispatch_output
        recv_topk_idx = dispatch_indices
        recv_topk_weights = dispatch_weights

        

        # Count how many tokens each local expert actually received using the truncated indices.
        num_local_experts = self.num_qps_per_rank
        num_recv_tokens_per_expert_list = [0] * num_local_experts
        if dispatch_indices is not None and dispatch_indices.numel() > 0:
            flat_indices = dispatch_indices.reshape(-1)
            local_offset = self.rank * self.num_qps_per_rank
            local_indices = flat_indices - local_offset
            mask = (local_indices >= 0) & (local_indices < num_local_experts)
            if mask.any():
                local_indices = local_indices[mask]
                counts = torch.bincount(local_indices, minlength=num_local_experts)
                if counts.numel() > 0:
                    num_recv_tokens_per_expert_list = counts.to(torch.int).tolist()
        
        # Store dispatch_indices in handle for combine
        new_handle = (dispatch_indices,src_token_pos, num_valid_tokens, dispatch_indices_arg)

        
        
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, new_handle, EventOverlap()

    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        dtype = x.dtype
        hidden_dim = x.size(1)
        op = self._get_op(dtype, hidden_dim)
        
        # Retrieve indices from handle
        if not handle or len(handle) < 1:
             raise ValueError("Invalid handle passed to combine. Expected handle from dispatch containing indices.")
        
        dispatch_indices = handle[0]
        dispatch_indices_arg = handle[3]
        if self.reorder:
            x , dispatch_indices, topk_weights = \
                self._revert_mori_dispatch_outputs(x, dispatch_indices, topk_weights, handle[1])


        combined_x = op.combine(x, topk_weights, dispatch_indices_arg)
        
        return combined_x[0] if isinstance(combined_x, tuple) else combined_x, combined_x[1] if isinstance(combined_x, tuple) else None, EventOverlap()

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Internode dispatch implementation.
        Mapped to dispatch in MORI.
        """
        return self.dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank,
                             is_token_in_rank, num_tokens_per_expert,
                             topk_idx, topk_weights, expert_alignment,
                             config, previous_event, async_finish,
                             allocate_on_comm_stream)

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation.
        Mapped to combine in MORI.
        """
        return self.combine(x, handle, topk_weights, config, previous_event, async_finish, allocate_on_comm_stream)

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        # Not needed or supported in MORI
        pass

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             use_fp8: bool = False, async_finish: bool = False, return_recv_hook: bool = False, topk_weights : torch.Tensor = None) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        Low latency dispatch.
        Not fully supported with the same API.
        """

        if(async_finish or return_recv_hook):
            raise NotImplementedError("MORI  async_finish/return_recv_hook is not supported yet.")

        
        # if use_fp8 then we need to call quantization
        # now let's just use CPU version for testing
        if use_fp8:
            inp, inp_scales = Buffer._per_token_cast_to_fp8(inp)
            dtype = inp.dtype
            scale_dim = inp_scales.size(1) 
        else:
            inp = x
            inp_scales = None
            dtype = inp.dtype
            hidden_dim = inp.size(1)
            scale_dim = 0
        

        self.num_experts_per_token = topk_idx.size(1)
        self.max_num_inp_token_per_rank = inp.size(0)
        
        op = self._get_op(dtype, hidden_dim, scale_dim)

        # MORI dispatch expects int32 indices.
        dispatch_indices_arg = topk_idx.to(dtype=torch.int32)
  
        dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, dispatch_recv_num_token = \
            op.dispatch(inp, topk_weights, inp_scales, dispatch_indices_arg)
        
        recv_count = dispatch_recv_num_token[0].item()
        
        if self.use_gpu_ll_layout_transform:
            packed_input, sorted_indices, expert_counts, packed_scales = mori.transform_dispatch_output_gpu(
                dispatch_output,
                dispatch_indices,
                self.config,
                recv_count,
                dispatch_scales
            )
        else:
            packed_input, sorted_indices, expert_counts, packed_scales = Buffer._transform_dispatch_output(
                dispatch_output,
                dispatch_indices,
                self.config,
                recv_count,
                dispatch_scales
            )


        recv_x = (packed_input, packed_scales) if use_fp8 else packed_input
        
        new_handle = (sorted_indices, expert_counts, recv_count, dispatch_weights, packed_scales)

        return recv_x, expert_counts, new_handle, EventOverlap(), None
        

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        Low latency combine.
        Not fully supported with the same API.
        """
        sorted_indices = handle[0]
        expert_counts = handle[1]
        recv_count = handle[2]
        dispatch_weights = handle[3]
        dispatch_scales = handle[4]
        # recv_topk_weights = handle[3]
        if self.use_gpu_ll_layout_transform:
            rec_output = mori.inverse_transform_dispatch_output_gpu(
                    x, sorted_indices, expert_counts, recv_count
            )
        else:
            rec_output = Buffer._inverse_transform_dispatch_output_gpu(
                x, sorted_indices, expert_counts, recv_count
            )

        dtype = rec_output.dtype
        hidden_dim = rec_output.size(1)
        op = self._get_op(dtype, hidden_dim)


        topk_idx = topk_idx.to(dtype=torch.int32)


        combine_output,combine_output_weight = op.combine(
            rec_output,
            dispatch_weights,
            # None,
            topk_idx,
            block_num=self.config.block_num,
            warp_per_block=16,
            call_reset = True,
        )

        return combine_output, EventOverlap(), None
        

    def get_next_low_latency_combine_buffer(self, handle: object):
        raise NotImplementedError("get_next_low_latency_combine_buffer is not supported.")


    # helper functions for matching DeepEp API
    def _reorder_mori_dispatch_outputs(self, recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                         token_order: torch.Tensor, dispatch_scales: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        reoder the outputs from mori dispatch to match DeepEp order.
        """    
        if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
            if dist.get_rank() == 0:
                Buffer._log_warning_once('[warning] reorder_mori_outputs guard: shape mismatch or empty order.')
            return recv_x,dispatch_scales, recv_topk_idx, recv_topk_weights
        if token_order.min() < 0 :
            if dist.get_rank() == 0:
                Buffer._log_warning_once('[warning] reorder_mori_outputs guard: order contains invalid indices.')
            return recv_x,dispatch_scales, recv_topk_idx, recv_topk_weights
        unique_tokens = torch.unique(token_order)
        if unique_tokens.numel() != token_order.numel():
            if dist.get_rank() == 0:
                Buffer._log_warning_once('[warning] reorder_mori_outputs guard: order contains repeated tokens.')
            return recv_x, dispatch_scales, recv_topk_idx, recv_topk_weights
        
        perm = torch.argsort(token_order)
        return recv_x[perm],dispatch_scales[perm] if dispatch_scales is not None else None, recv_topk_idx[perm], recv_topk_weights[perm]


    def _revert_mori_dispatch_outputs(self, recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                            token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
            if dist.get_rank() == 0:
                Buffer._log_warning_once('[warning] revert_mori_outputs guard: shape mismatch or empty order.')
            return recv_x, recv_topk_idx, recv_topk_weights
        if token_order.min() < 0 :
            if dist.get_rank() == 0:
                Buffer._log_warning_once('[warning] revert_mori_outputs guard: order contains invalid indices.')
            return recv_x, recv_topk_idx, recv_topk_weights
        unique_tokens = torch.unique(token_order)
        if unique_tokens.numel() != token_order.numel():
            if dist.get_rank() == 0:
                Buffer._log_warning_once('[warning] revert_mori_outputs guard: order contains repeated tokens.')
            return recv_x, recv_topk_idx, recv_topk_weights
        perm = torch.argsort(token_order)
        inverted = torch.empty_like(perm)
        inverted[perm] = torch.arange(perm.numel(), device=perm.device)
        return recv_x[inverted], recv_topk_idx[inverted], recv_topk_weights[inverted]

    
    @staticmethod
    def _per_token_cast_to_fp8(x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        max_range = min(torch.finfo(torch.float8_e4m3fn).max, torch.finfo(torch.float8_e4m3fnuz).max) 
        return (x_view * (max_range / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / max_range).view(m, -1)

    @staticmethod
    def _per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
        x_scales = x_scales.view(x_fp8.size(0), -1, 1)
        return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


    
    @staticmethod
    def _transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count, dispatch_scales=None):
        """
        Transforms dispatch output to a packed layout [Experts, N, hidden_dim]
        where tokens are packed contiguously for each expert.
        
        Args:
            dispatch_output: [N, H] tensor of received tokens
            dispatch_indices: [N, K] tensor of expert indices for each token
            config: EpDispatchCombineConfig object
            recv_count: Scalar, number of valid tokens received
            dispatch_scales: Optional [N, scale_dim] tensor of per-token scales
        """
        # 1. Slice valid data
        valid_tokens = dispatch_output[:recv_count]   # [M, H]
        valid_indices = dispatch_indices[:recv_count] # [M, K]
        N_capacity = dispatch_output.size(0)
        _, H = valid_tokens.shape
        _, K = valid_indices.shape
        if dispatch_scales is not None:
            valid_scales = dispatch_scales[:recv_count]
            scale_dim = valid_scales.size(1)
        else:
            valid_scales = None
            scale_dim = 0
        E = config.num_experts_per_rank
        
        # 2. Find which tokens go to which local expert
        flat_indices = valid_indices.reshape(-1) # [M*K]
        is_local = (flat_indices // E) == config.rank
        active_flat_indices = torch.nonzero(is_local, as_tuple=False).squeeze(-1)
        
        if active_flat_indices.numel() == 0:
            packed_output = dispatch_output.new_zeros((E, N_capacity, H))
            sorted_token_indices = torch.empty((0,), device=dispatch_output.device, dtype=torch.int32)
            expert_counts = torch.zeros((E,), device=dispatch_output.device, dtype=torch.int32)
            packed_scales = (
                dispatch_scales.new_zeros((E, N_capacity, scale_dim))
                if scale_dim > 0 and dispatch_scales is not None
                else None
            )
            return packed_output, sorted_token_indices, expert_counts, packed_scales

        token_indices = (active_flat_indices // K).to(torch.long)
        local_expert_ids = (flat_indices.index_select(0, active_flat_indices).remainder(E)).to(torch.long)
        
        # 3. Sort by expert ID
        sort_order = torch.argsort(local_expert_ids, stable=True)
        sorted_token_indices_long = token_indices.index_select(0, sort_order)
        sorted_expert_ids = local_expert_ids.index_select(0, sort_order)
        
        # 4. Calculate counts and pack
        expert_counts_long = torch.bincount(sorted_expert_ids, minlength=E)
        
        # Generate slot indices: [0, 1, ... c0-1, 0, 1, ... c1-1, ...]
        slot_indices_list = [
            torch.arange(int(count.item()), device=dispatch_output.device, dtype=torch.long)
            for count in expert_counts_long
        ]
        slot_indices = torch.cat(slot_indices_list) if slot_indices_list else torch.empty((0,), device=dispatch_output.device, dtype=torch.long)
        
        packed_output = dispatch_output.new_zeros((E, N_capacity, H))
        packed_output[sorted_expert_ids, slot_indices] = valid_tokens.index_select(0, sorted_token_indices_long)

        packed_scales = None
        if scale_dim > 0 and valid_scales is not None:
            packed_scales = valid_scales.new_zeros((E, N_capacity, scale_dim))
            packed_scales[sorted_expert_ids, slot_indices] = valid_scales.index_select(0, sorted_token_indices_long)
        
        return (
            packed_output,
            sorted_token_indices_long.to(torch.int32),
            expert_counts_long.to(torch.int32),
            packed_scales,
        )

    @staticmethod
    def _inverse_transform_dispatch_output(packed_output, original_indices, expert_counts, original_N):
        """
        Reconstructs dispatch_output from packed_output [E, N, H].
        
        Args:
            packed_output: [E, N, H] tensor (result of GEMM)
            original_indices: [M] tensor mapping rows back to dispatch_output indices
            expert_counts: [E] tensor of token counts per expert
            original_N: Original number of tokens (N)
            
        Returns:
            rec_output: [N, H]
        """
        E, _, H = packed_output.shape
        device = packed_output.device
        counts_long = expert_counts.to(torch.long)
        
        # Generate read indices matching the write order
        slot_indices_list = [
            torch.arange(int(count.item()), device=device, dtype=torch.long)
            for count in counts_long
        ]
        slot_indices = torch.cat(slot_indices_list) if slot_indices_list else torch.empty((0,), device=device, dtype=torch.long)
        
        expert_ids = torch.repeat_interleave(torch.arange(E, device=device, dtype=torch.long), counts_long)
        
        # Extract valid tokens
        flat_values = packed_output[expert_ids, slot_indices]
        
        # Scatter add back
        rec_output = torch.zeros((original_N, H), dtype=packed_output.dtype, device=device)
        rec_output.index_add_(0, original_indices.to(torch.long), flat_values)
        
        return rec_output