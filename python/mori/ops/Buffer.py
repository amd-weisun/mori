import logging
import mori
import os
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

    def __init__(self, group: dist.ProcessGroup,
                 num_nvl_bytes: int = 0, num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False, num_qps_per_rank: int = 1, max_num_inp_token_per_rank : int = 128, gpu_per_node: int = 1,
                 num_experts_per_token : int = 8,
                 group_name: str = "default") -> None:
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
        self.gpu_per_node = gpu_per_node  # Assuming 8 GPUs per node
        self.world_size = dist.get_world_size(group=group)
        self.max_num_inp_token_per_rank = max_num_inp_token_per_rank
        self.num_experts_per_token = num_experts_per_token
        # Cache for MORI ops
        self.group_name = group_name
        self.ops = {}
        self._cleanup_done = False
        self.setup()

    def _get_op(self, dtype: torch.dtype, hidden_dim: int, scale_dim: int = 0) -> EpDispatchCombineOp:
        key = (dtype, hidden_dim, scale_dim, self.low_latency_mode)
        if key not in self.ops:
            # Determine kernel type
            kernel_type = EpDispatchCombineKernelType.IntraNode
            # Simple heuristic for kernel type
            if self.group_size > 8: 
                 kernel_type = EpDispatchCombineKernelType.InterNodeV1
            
            if self.low_latency_mode:
                 kernel_type = EpDispatchCombineKernelType.InterNodeV1LL

            # TODO: These parameters might need to be tuned or exposed
            config = EpDispatchCombineConfig(
                data_type=dtype,
                rank=self.rank,
                world_size=self.group_size,
                hidden_dim=hidden_dim,
                scale_dim=scale_dim,
                scale_type_size=1 if scale_dim > 0 else 0,
                max_token_type_size=4, 
                max_num_inp_token_per_rank=self.max_num_inp_token_per_rank, # Increased limit
                num_experts_per_rank=self.num_qps_per_rank, # Default assumption
                num_experts_per_token=self.num_experts_per_token, # topK
                warp_num_per_block=16,
                block_num=32,
                kernel_type=kernel_type,
                gpu_per_node=self.gpu_per_node,
                rdma_block_num=16,
            )
            self.ops[key] = EpDispatchCombineOp(config)
        return self.ops[key]

    def setup(self):
        local_rank = self.rank % self.gpu_per_node
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                rank=self.rank,
                world_size=self.world_size,
            )

        print(f"init process group done. world_group type: {type(torch.distributed.group.WORLD)}")
        world_group = torch.distributed.group.WORLD
        assert world_group is not None

        print("process group ok")
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

        print(f"I'm pe {mori.shmem.shmem_mype()} in {mori.shmem.shmem_npes()} pes")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(999)

    def cleanup(self):
        if self._cleanup_done:
            return
        try:
            mori.shmem.shmem_finalize()
        except Exception:  # pylint: disable=broad-except
            logger.warning("mori.shmem.shmem_finalize failed")
        # try:
        #     if dist.is_initialized():
        #         dist.destroy_process_group()
        # except Exception:  # pylint: disable=broad-except
        #     logger.warning("dist.destroy_process_group failed")
        self._cleanup_done = True


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
        
        if isinstance(x, tuple):
            inp, inp_scales = x
            dtype = inp.dtype
            hidden_dim = inp.size(1)
            scale_dim = inp_scales.size(1) if inp_scales is not None else 0
        else:
            inp = x
            inp_scales = None
            dtype = inp.dtype
            hidden_dim = inp.size(1)
            scale_dim = 0

        op = self._get_op(dtype, hidden_dim, scale_dim)
        
        if topk_idx is None or topk_weights is None:
             raise NotImplementedError("dispatch with handle (cached layout) is not fully supported yet. Please provide topk_idx and topk_weights.")

        # MORI dispatch expects int32 indices.
        dispatch_indices_arg = topk_idx.to(dtype=torch.int32)
        
        #DEBUG ONLY
        # print(f"[Rank {self.rank}] Dispatching with dtype={dtype}, hidden_dim={hidden_dim}, scale_dim={scale_dim}, num_tokens={inp.size(0)}")
        # print(f"[inp shape {inp.shape if not isinstance(x, tuple)  else inp[0].shape}] , topk_weights shape {topk_weights.shape}, dtype = {topk_weights.dtype},  topk_idx shape={dispatch_indices_arg.shape}, dtype = {dispatch_indices_arg.dtype}")
        dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, dispatch_recv_num_token = \
            op.dispatch(inp, topk_weights, inp_scales, dispatch_indices_arg)

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
        src_token_pos =  _truncate(src_token_pos)
        if num_valid_tokens > 0:
        dispatch_output, dispatch_indices, dispatch_weights = \
                self._reorder_mori_dispatch_outputs(dispatch_output, dispatch_indices, dispatch_weights, src_token_pos)
        
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
        new_handle = (dispatch_indices,src_token_pos)

        
        
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

        x , dispatch_indices, topk_weights = \
             self._revert_mori_dispatch_outputs(x, dispatch_indices, topk_weights, handle[1])

        # print(f"[Rank {self.rank}] Combining with dtype={dtype}, hidden_dim={hidden_dim}, num_tokens={x.size(0)}")
        # print(f"[inp shape {x.shape}] , topk_weights shape {topk_weights.shape if topk_weights is not None else None}, dtype = {topk_weights.dtype if topk_weights is not None else None},  dispatch_indices shape={dispatch_indices.shape}, dtype = {dispatch_indices.dtype}")
        
        # MORI combine
        combined_x = op.combine(x, topk_weights, dispatch_indices)
        
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
                             use_fp8: bool = True, async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        Low latency dispatch.
        Not fully supported with the same API.
        """
        raise NotImplementedError("low_latency_dispatch is not supported. Use dispatch with low_latency_mode=True.")

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        Low latency combine.
        Not fully supported with the same API.
        """
        raise NotImplementedError("low_latency_combine is not supported. Use combine.")

    def get_next_low_latency_combine_buffer(self, handle: object):
        raise NotImplementedError("get_next_low_latency_combine_buffer is not supported.")


    # helper functions for matching DeepEp API
    def _reorder_mori_dispatch_outputs(recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                         token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        reoder the outputs from mori dispatch to match DeepEp order.
        """    
        if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
            if dist.get_rank() == 0:
                print('[warning] reorder_mori_outputs guard: shape mismatch or empty order.', flush=True)
            return recv_x, recv_topk_idx, recv_topk_weights
        if token_order.min() < 0 :
            if dist.get_rank() == 0:
                print('[warning] reorder_mori_outputs guard: order contains invalid indices.', flush=True)
            return recv_x, recv_topk_idx, recv_topk_weights
        unique_tokens = torch.unique(token_order)
        if unique_tokens.numel() != token_order.numel():
            if dist.get_rank() == 0:
                print('[warning] reorder_mori_outputs guard: order contains repeated tokens.', flush=True)
            return recv_x, recv_topk_idx, recv_topk_weights
        perm = torch.argsort(token_order)
        return recv_x[perm], recv_topk_idx[perm], recv_topk_weights[perm]


    def _revert_mori_dispatch_outputs(recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor,
                            token_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_order.numel() == 0 or recv_x.size(0) != token_order.numel():
            if dist.get_rank() == 0:
                print('[warning] revert_mori_outputs guard: shape mismatch or empty order.', flush=True)
            return recv_x, recv_topk_idx, recv_topk_weights
        if token_order.min() < 0 :
            if dist.get_rank() == 0:
                print('[warning] revert_mori_outputs guard: order contains invalid indices.', flush=True)
            return recv_x, recv_topk_idx, recv_topk_weights
        unique_tokens = torch.unique(token_order)
        if unique_tokens.numel() != token_order.numel():
            if dist.get_rank() == 0:
                print('[warning] revert_mori_outputs guard: order contains repeated tokens.', flush=True)
            return recv_x, recv_topk_idx, recv_topk_weights
        perm = torch.argsort(token_order)
        inverted = torch.empty_like(perm)
        inverted[perm] = torch.arange(perm.numel(), device=perm.device)
        return recv_x[inverted], recv_topk_idx[inverted], recv_topk_weights[inverted]