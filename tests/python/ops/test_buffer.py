
import pytest
import torch
import torch.distributed as dist
from mori.ops.Buffer import Buffer
from tests.python.utils import TorchDistProcessManager

def run_buffer_test(rank, world_size):
    # Buffer expects the process group to be initialized if we want to use an existing one,
    # or it initializes it itself.
    # TorchDistProcessManager initializes the process group and registers "default".
    
    # We need to handle the fact that Buffer might try to register "default" again.
    # But let's see if it works.
    
    # Create Buffer
    # We use the default group which is WORLD
    # Set num_qps_per_rank such that total experts covers our random indices
    # We want num_experts (8) to be valid.
    # total_experts = num_qps_per_rank * world_size
    # So num_qps_per_rank = num_experts // world_size
    num_experts = 8
    num_qps_per_rank = num_experts // world_size
    
    group = dist.group.WORLD
    buffer = Buffer(group, num_qps_per_rank=num_qps_per_rank, group_name="default")
    
    # Create dummy data
    num_tokens = 128
    hidden_dim = 64
    topk = 2
    
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=buffer.device)
    
    # Random topk indices
    # Ensure indices are within range [0, num_experts * world_size)
    # But Buffer assumes num_experts_per_rank.
    # In Buffer.py: num_experts_per_rank=self.num_qps_per_rank (which defaults to 1 in init arg, but we passed 1)
    # Wait, Buffer init says: num_qps_per_rank: int = 1
    # And in _get_op: num_experts_per_rank=self.num_qps_per_rank
    
    # So total experts = num_qps_per_rank * world_size?
    # The dispatch/combine op config has num_experts_per_rank.
    
    topk_idx = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64, device=buffer.device)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device=buffer.device)
    # Normalize weights
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    # Dispatch
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights)
        
    assert recv_x.shape[1] == hidden_dim
    assert recv_x.dtype == torch.bfloat16
    
    # Combine
    # For combine, we need to send back something of the same shape as recv_x
    # recv_x is [num_recv_tokens, hidden]
    
    combined_x, _, event = buffer.combine(recv_x, handle, topk_weights=None) # topk_weights not used in combine for now?
    
    assert combined_x.shape == x.shape
    assert combined_x.dtype == x.dtype
    
    return True

@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Need at least 8 GPUs")
def test_buffer_dispatch_combine():
    world_size = 8
    # We set init_mori_shmem=False because Buffer.setup() calls shmem_torch_process_group_init
    # and we want to test that flow, or at least avoid double init if Buffer does it.
    # However, Buffer.setup() catches the exception if it fails.
    # But TorchDistContext registers "default". Buffer.setup() also registers "default".
    # If Buffer.setup() crashes on register, we have a problem.
    
    # Let's try with init_mori_shmem=False.
    # TorchDistContext will init dist and register "default".
    # Buffer.setup will try to register "default" and might crash.
    
    # To avoid crash, we might need to patch Buffer or use a different group name?
    # But Buffer uses "default" by default.
    
    manager = TorchDistProcessManager(init_mori_shmem=False)
    manager.start_workers(world_size)
    
    for rank in range(world_size):
        manager.task_queue.put((run_buffer_test, []))
        
    results = []
    for _ in range(world_size):
        rank, res = manager.result_queue.get()
        results.append(res)
        
    manager.shutdown()
    
    for res in results:
        if isinstance(res, str): # Exception traceback
            pytest.fail(res)
        assert res is True

if __name__ == "__main__":
    test_buffer_dispatch_combine()
