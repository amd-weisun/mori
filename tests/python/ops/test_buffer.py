
import pytest
import torch
import torch.distributed as dist
from mori.ops.Buffer import Buffer
from tests.python.utils import TorchDistProcessManager

def run_buffer_test(rank, world_size, group_name="default"):
    # Buffer expects the process group to be initialized if we want to use an existing one,
    # or it initializes it itself.
    # TorchDistProcessManager initializes the process group and registers "default".
    
    # We need to handle the fact that Buffer might try to register "default" again.
    # But let's see if it works.
    
    # Create Buffer
    # We use the default group which is WORLD
    # Configure experts per rank to match the data the kernel sees.
    num_experts_per_rank = 8
    num_experts = num_experts_per_rank * world_size
    num_qps_per_rank = num_experts_per_rank
    
    group = dist.group.WORLD
    num_tokens = 8
    hidden_dim = 8 # 4096 
    topk = 4
    print(f"[Rank {rank}] Creating Buffer (group={group_name})...")
    buffer = Buffer(group, num_qps_per_rank=num_qps_per_rank, max_num_inp_token_per_rank = num_tokens, num_experts_per_token =topk, gpu_per_node = world_size, group_name=group_name)
    print(f"[Rank {rank}] Buffer created.")
    
    # Create dummy data

    
    # Give each token row a deterministic pattern: (token_index * rank) repeated across the hidden dimension.
    row_values = torch.arange(num_tokens, dtype=torch.float32, device=buffer.device) * rank
    x = row_values.unsqueeze(1).expand(num_tokens, hidden_dim).to(torch.bfloat16)
    # if(rank == 0):
    print(f"[Rank {rank}] Input tensor x shape: {x.shape}, dtype: {x.dtype}")   
    print(f"[Rank {rank}] input tensor x value {x}")
    
    # Random topk indices
    # Ensure indices are within range [0, num_experts * world_size)
    # But Buffer assumes num_experts_per_rank.
    # In Buffer.py: num_experts_per_rank=self.num_qps_per_rank (which defaults to 1 in init arg, but we passed 1)
    # Wait, Buffer init says: num_qps_per_rank: int = 1
    # And in _get_op: num_experts_per_rank=self.num_qps_per_rank
    
    # So total experts = num_qps_per_rank * world_size?
    # The dispatch/combine op config has num_experts_per_rank.
    
    topk_idx = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64, device=buffer.device)
    topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32, device=buffer.device)
    
    # Dispatch
    print(f"[Rank {rank}] Dispatching...")
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights)
    print(f"[Rank {rank}] Dispatch done.")
        
    assert recv_x.shape[1] == hidden_dim
    assert recv_x.dtype == torch.bfloat16

    print(f"[Rank {rank}] recv_x  shape: {recv_x.shape}, dtype: {recv_x.dtype}")   
    print(f"[Rank {rank}] recv_x  value {recv_x}")
    print(f"[Rank {rank}] recv_topk_idx  shape: {recv_topk_idx.shape}, dtype: {recv_topk_idx.dtype}")   
    print(f"[Rank {rank}] recv_topk_idx  value {recv_topk_idx}")
    print(f"[Rank {rank}] handle  shape: {handle[0].shape}, dtype: {handle[0].dtype}")   
    print(f"[Rank {rank}] handle  value {handle[0]}")
    
    # Combine
    # For combine, we need to send back something of the same shape as recv_x
    # recv_x is [num_recv_tokens, hidden]
    
    print(f"[Rank {rank}] Combining...")
    combined_out, combine_output_weight, event = buffer.combine(recv_x, handle, topk_weights=None) # topk_weights not used in combine for now?
    print(f"[Rank {rank}] Combine done.")
    combined_tensor = combined_out[0] if isinstance(combined_out, tuple) else combined_out


    # if(rank == 0):
    print(f"[Rank {rank}] combine tensor  shape: {combined_tensor.shape}, dtype: {combined_tensor.dtype}")   
    print(f"[Rank {rank}] combined_tensor  value {combined_tensor}")

    print(f"[Rank {rank}] combine_output_weight  shape: {combine_output_weight.shape}, dtype: {combine_output_weight.dtype}")   
    print(f"[Rank {rank}] combine_output_weight  value {combine_output_weight}")

    assert combined_tensor.shape == x.shape
    assert combined_tensor.dtype == x.dtype
    dist.barrier()
    # buffer.cleanup()
    
    return True

@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Need at least 8 GPUs")
def test_buffer_dispatch_combine():
    world_size = 2
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
    
    print(f"Starting workers with world_size={world_size}...")
    manager = TorchDistProcessManager(init_mori_shmem=False)
    manager.start_workers(world_size)
    
    print("Queuing tasks...")
    for rank in range(world_size):
        group_name = f"buffer_group_{rank}"
        manager.task_queue.put((run_buffer_test, [world_size, group_name]))
        
    results = []
    print("Waiting for results...")
    for _ in range(world_size):
        rank, res = manager.result_queue.get()
        print(f"Received result from rank {rank}")
        results.append(res)
        
    manager.shutdown()
    print("Workers shutdown.")
    
    for res in results:
        if isinstance(res, str): # Exception traceback
            pytest.fail(res)
        assert res is True

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    test_buffer_dispatch_combine()
