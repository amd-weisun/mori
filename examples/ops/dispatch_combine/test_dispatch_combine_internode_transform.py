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
import mori
import os

import torch
import torch.distributed as dist
import argparse
import time
from tqdm import tqdm


kernel_type_map = {
    "v0": mori.ops.EpDispatchCombineKernelType.InterNode,
    "v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
    "v1_ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
}


class EpDispatchCombineTestCase:
    def __init__(
        self,
        rank,
        gpu_per_node,
        world_size,
        max_tokens,
        total_experts,
        kernel_type,
        dtype=torch.bfloat16,
    ):
        self.rank = rank
        self.gpu_per_node = gpu_per_node
        self.world_size = world_size
        self.total_experts = total_experts
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=7168,
            scale_dim=32,
            scale_type_size=4,
            max_num_inp_token_per_rank=max_tokens,
            num_experts_per_rank= self.total_experts//self.world_size,
            num_experts_per_token=8,
            warp_num_per_block=16,
            block_num=32,
            max_token_type_size=2,
            kernel_type=kernel_type_map[kernel_type],
            gpu_per_node=self.gpu_per_node,
            rdma_block_num=16,
        )

    def setup(self):
        local_rank = self.rank % self.gpu_per_node
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

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
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

        print(f"I'm pe {mori.shmem.shmem_mype()} in {mori.shmem.shmem_npes()} pes")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(999)

    def cleanup(self):
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()

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
            for _ in range(self.world_size)
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

    def gen_test_data(self, use_max_token_num=False):
        # gen num_tokens
        if use_max_token_num:
            num_token = torch.tensor(
                [self.config.max_num_inp_token_per_rank for i in range(self.world_size)]
            ).to(self.device)
        else:
            num_token = torch.randint(
                1,
                self.config.max_num_inp_token_per_rank + 1,
                [self.world_size],
                generator=self.rng,
                device=self.device,
            )

        # gen indices
        all_rank_indices = []
        for r in range(self.world_size):
            indices = torch.empty(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.int64,
                # device=self.device,
            )
            for i in range(num_token[r]):
                perm = torch.randperm(
                    self.config.num_experts_per_rank * self.config.world_size,
                    generator=self.rng,
                    device=self.device,
                )
                indices[i] = perm[: self.config.num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(self.device))

        # num_total_experts = self.config.num_experts_per_rank * self.config.world_size
        # num_nodes = self.config.world_size // self.config.gpu_per_node

        # even_indices = (
        #     torch.arange(
        #         self.config.max_num_inp_token_per_rank
        #         * self.config.num_experts_per_token,
        #         device="cuda",
        #     ).view(
        #         self.config.max_num_inp_token_per_rank,
        #         self.config.num_experts_per_token,
        #     )
        #     % 256
        # )
        # even_indices = even_indices.to(torch.int32)
        # all_rank_indices = [even_indices for _ in range(self.world_size)]

        # gen weights
        all_rank_weights = [
            torch.rand(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.world_size)
        ]

        # gen scales
        all_rank_scales = [
            torch.rand(
                num_token[r],
                self.config.scale_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.world_size)
        ]

        # gen input & output
        # some functions such as randn and cat are not implemented for fp8
        all_rank_input = []
        for r in range(self.world_size):
            all_rank_input.append(
                torch.randn(
                    num_token[r],
                    self.config.hidden_dim,
                    dtype=torch.float32,
                    generator=self.rng,
                    device=self.device,
                ).to(self.config.data_type)
            )

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def count_token_num(self, all_rank_indices):
        # Per-rank counts
        rank_counts = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )
        rank_counts_remote_recv = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )
        rank_counts_remote_send = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )

        for src_rank, indices in enumerate(all_rank_indices):
            src_node = src_rank // self.config.gpu_per_node

            # Map expert IDs to rank IDs
            token_ranks = (
                indices // self.config.num_experts_per_rank
            )  # [num_tokens, num_experts_per_token]

            # Deduplicate rank IDs per token
            unique_ranks_per_token = [torch.unique(row) for row in token_ranks]

            # For each token, update counts
            for ur in unique_ranks_per_token:
                rank_counts[ur] += 1  # All ranks that receive this token

                dst_nodes = {
                    dst_rank // self.config.gpu_per_node for dst_rank in ur.tolist()
                }

                for dst_rank in ur.tolist():
                    dst_node = dst_rank // self.config.gpu_per_node
                    if dst_node != src_node:
                        # Receiving side
                        rank_counts_remote_recv[dst_rank] += 1

                # Sending side (dedup by node: count once if token goes to a remote node)
                for dst_node in dst_nodes:
                    if dst_node != src_node:
                        rank_counts_remote_send[src_rank] += 1

        if self.config.rank == 0:
            print("Rank counts (deduplicated):", rank_counts)
            # print("Rank counts local nodes:", rank_counts - rank_counts_remote_recv)
            # print("Rank counts from other nodes:", rank_counts_remote_recv)
            # print("Rank counts to other nodes:", rank_counts_remote_send)
        return rank_counts, rank_counts_remote_recv, rank_counts_remote_send

    def transform_dispatch_output(dispatch_output, dispatch_indices, config, recv_count):
        """
        Transforms dispatch output to a packed layout [Experts, N, hidden_dim]
        where tokens are packed contiguously for each expert.
        
        Args:
            dispatch_output: [N, H] tensor of received tokens
            dispatch_indices: [N, K] tensor of expert indices for each token
            config: EpDispatchCombineConfig object
            recv_count: Scalar, number of valid tokens received
        """
        # 1. Slice valid data
        valid_tokens = dispatch_output[:recv_count]   # [M, H]
        valid_indices = dispatch_indices[:recv_count] # [M, K]
        
        N_capacity = dispatch_output.size(0)
        _, H = valid_tokens.shape
        _, K = valid_indices.shape
        E = config.num_experts_per_rank
        
        # 2. Find which tokens go to which local expert
        flat_indices = valid_indices.view(-1) # [M*K]
        is_local = (flat_indices // E) == config.rank
        active_flat_indices = torch.nonzero(is_local).squeeze(-1)
        
        if active_flat_indices.numel() == 0:
             return (
                 torch.zeros((E, N_capacity, H), device=dispatch_output.device, dtype=dispatch_output.dtype),
                 torch.empty((0,), device=dispatch_output.device, dtype=torch.long),
                 torch.zeros((E,), device=dispatch_output.device, dtype=torch.long)
             )

        token_indices = active_flat_indices.div(K, rounding_mode='floor')
        local_expert_ids = flat_indices[active_flat_indices] % E
        
        # 3. Sort by expert ID
        sort_order = torch.argsort(local_expert_ids)
        sorted_token_indices = token_indices[sort_order]
        sorted_expert_ids = local_expert_ids[sort_order]
        
        # 4. Calculate counts and pack
        expert_counts = torch.bincount(sorted_expert_ids, minlength=E)
        
        # Generate slot indices: [0, 1, ... c0-1, 0, 1, ... c1-1, ...]
        slot_indices_list = [torch.arange(c, device=dispatch_output.device) for c in expert_counts]
        slot_indices = torch.cat(slot_indices_list)
        
        packed_output = torch.zeros((E, N_capacity, H), dtype=dispatch_output.dtype, device=dispatch_output.device)
        packed_output[sorted_expert_ids, slot_indices] = valid_tokens[sorted_token_indices]
        
        return packed_output, sorted_token_indices, expert_counts

    def inverse_transform_dispatch_output(packed_output, original_indices, expert_counts, original_N):
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
        
        # Generate read indices matching the write order
        slot_indices_list = [torch.arange(c, device=device) for c in expert_counts]
        slot_indices = torch.cat(slot_indices_list)
        
        expert_ids = torch.repeat_interleave(torch.arange(E, device=device), expert_counts)
        
        # Extract valid tokens
        flat_values = packed_output[expert_ids, slot_indices]
        
        # Scatter add back
        rec_output = torch.zeros((original_N, H), dtype=packed_output.dtype, device=device)
        rec_output.index_add_(0, original_indices, flat_values)
        
        return rec_output

    def run_test_once(self, op, test_data, error_round, round):
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_scales[self.rank],
            all_rank_indices[self.rank],
            block_num=self.config.block_num,
            warp_per_block=16,
        )
        torch.cuda.synchronize()

        # --- Simulated GEMM Start ---
        recv_count = dispatch_recv_num_token.item()
        
        # 1. Transform Layout
        # Validate counts before transform
        assert recv_count <= dispatch_output.size(0), "recv_count exceeds capacity"
        assert dispatch_indices.size(0) >= recv_count, "indices shorter than recv_count"
        if args_cli.transform_impl == "gpu":
            packed_input, sorted_indices, expert_counts = mori.transform_dispatch_output_gpu(
                dispatch_output,
                dispatch_indices,
                self.config,
                recv_count,
            )
        else:
            packed_input, sorted_indices, expert_counts = EpDispatchCombineTestCase.transform_dispatch_output(
                dispatch_output,
                dispatch_indices,
                self.config,
                recv_count,
            )

        if self.rank == 0 and round == 0:
            print("\n--- Packed Output Visualization (Rank 0) ---")
            print(f"Original dispatch_output Shape: {dispatch_output.shape}")
            print(f"Packed Shape: {packed_input.shape}")
            print(f"Expert Counts (Tokens per Expert): {expert_counts.tolist()}")
            if recv_count > 0:
                unique_tokens = torch.unique(sorted_indices)
                print(f"Unique Tokens Received: {len(unique_tokens)}")
                print(f"Total Token-Expert Pairs: {len(sorted_indices)}")
                if len(sorted_indices) > len(unique_tokens):
                    print(">> Duplication Confirmed: Some tokens are present in multiple expert rows.")
                else:
                    print(">> No Duplication: Each token is assigned to at most one local expert.")
            print("--------------------------------------------\n")
        
        # 2. Simulated GEMM (Multiply by 1.0)
        # Keep packed_input unchanged; adjust after inverse using token_counts.
        gemm_output = packed_input * 1.0
        
        # 3. Inverse Transform
        if args_cli.transform_impl == "gpu":
            rec_output = mori.inverse_transform_dispatch_output_gpu(
                gemm_output, sorted_indices, expert_counts, dispatch_output.size(0)
            )
        else:
            rec_output = EpDispatchCombineTestCase.inverse_transform_dispatch_output(
                gemm_output, sorted_indices, expert_counts, dispatch_output.size(0)
            )

        # Normalize accumulated contributions back to the original per-token value
        if recv_count > 0:
            token_counts = torch.bincount(sorted_indices, minlength=recv_count)
            valid = token_counts > 0
            rec_output[:recv_count][valid] = rec_output[:recv_count][valid] / token_counts[valid].unsqueeze(-1)

        if self.rank == 0 and round == 0:
            # print("\n--- rec_output Visualization (Rank 0) ---")
            # print(f"rec_output Shape: {rec_output.shape}")
            # print(f"Original dispatch_output Shape: {dispatch_output.shape}")
            # print(f"simulated gemm_output Shape: {gemm_output.shape}")
            # if recv_count > 0:
            #     expected_rec = dispatch_output[:recv_count]
            #     valid_rec = rec_output[:recv_count]
                
            #     diff = (valid_rec - expected_rec).abs()
            #     max_diff = diff.max().item()
                
            #     print(f"Reconstruction Check (Valid Tokens: {recv_count}):")
            #     print(f"  Max Diff (Strict Order): {max_diff:.6f}")
            #     if max_diff < 1e-2:
            #         print(">> SUCCESS: Reconstruction matches original input.")
            #     else:
            #         print(">> FAILURE: Reconstruction mismatch.")
                    
            #         # Debug: Check if it's just a permutation issue
            #         # Flatten to compare set of values
            #         rec_sorted, _ = torch.sort(valid_rec.flatten())
            #         exp_sorted, _ = torch.sort(expected_rec.flatten())
            #         sort_diff = (rec_sorted - exp_sorted).abs().max().item()
            #         print(f"  Max Diff (Sorted Values): {sort_diff:.6f}")
                    
            #         if sort_diff < 1e-2:
            #             print(">> DIAGNOSTIC: Values are correct but order is wrong (Permutation Issue).")
            #         else:
            #             print(">> DIAGNOSTIC: Values are incorrect.")
            # else:
            #     print("No tokens received.")
            # print("--------------------------------------------\n")

            print("\n--- Sorted Indices Verification (Rank 0) ---")
            print(f"sorted_indices Shape: {sorted_indices.shape}")
            print(f"dispatch_indices Shape: {dispatch_indices.shape}")
            if recv_count > 0:
                # Verify that sorted_indices matches what we expect from dispatch_indices
                valid_indices = dispatch_indices[:recv_count]
                E = self.config.num_experts_per_rank
                K = self.config.num_experts_per_token
                
                flat_indices = valid_indices.view(-1)
                is_local = (flat_indices // E) == self.config.rank
                active_flat_indices = torch.nonzero(is_local).squeeze(-1)
                
                expected_token_indices = active_flat_indices.div(K, rounding_mode='floor')
                local_expert_ids = flat_indices[active_flat_indices] % E
                
                # Sort by expert ID (matching gpu_kernels.py logic)
                sort_order = torch.argsort(local_expert_ids, stable=True)
                expected_sorted_indices = expected_token_indices[sort_order]
                
                matches = torch.equal(sorted_indices, expected_sorted_indices)
                
                if matches:
                    print(">> SUCCESS: sorted_indices matches expected derivation from dispatch_indices.")
                else:
                    print(">> FAILURE: sorted_indices mismatch against expected derivation.")
                    print(f"  sorted_indices shape: {sorted_indices.shape}")
                    print(f"  expected shape: {expected_sorted_indices.shape}")
                    if sorted_indices.shape == expected_sorted_indices.shape:
                        diff = (sorted_indices != expected_sorted_indices).sum().item()
                        print(f"  Mismatched elements: {diff}")
            else:
                print("No tokens received.")
            print("--------------------------------------------\n")

        if recv_count > 0:
            expected_rec = dispatch_output[:recv_count]
            print(f"\n--- rec_output check (Rank {self.rank}) ---")
            print(f"Reconstruction Check (Valid Tokens: {recv_count}):")
            print(f"rec_output Shape: {rec_output.shape}")
            print(f"Original dispatch_output Shape: {dispatch_output.shape}")
            print(f"simulated gemm_output Shape: {gemm_output.shape}")
            if not torch.allclose(rec_output[:recv_count], expected_rec, atol=1e-2, rtol=1e-2):
                print(f"Rank {self.rank}: Reconstruction mismatch!")
                diff = (rec_output[:recv_count] - expected_rec).abs()
                print(f"Max diff: {diff.max().item()}")
                assert False
            else:
                print(f"Rank {self.rank}: Reconstruction matches original input.")

        # --- Simulated GEMM End ---
        rank_counts, _, _ = self.count_token_num(all_rank_indices)

        src_token_pos = op.get_dispatch_src_token_pos().tolist()
        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        recv_token_num = len(src_token_pos)

        # Check recv token num
        print(f"rank {self.rank} recv {recv_token_num} tokens")
        token_num_pass = rank_counts[self.rank] == recv_token_num
        if not token_num_pass:
            print(
                f"rank {self.rank} expected token num {rank_counts[self.rank]} got {recv_token_num}"
            )
            assert False

        # Check token equality
        for i, src_token_id in enumerate(src_token_pos):
            src_pe = src_token_id // max_num_token_to_send_per_rank
            src_tok_id = src_token_id % max_num_token_to_send_per_rank
            is_pass = torch.equal(
                dispatch_output[i], all_rank_input[src_pe][src_tok_id]
            )
            if not is_pass:
                print(
                    f"rank {self.rank} token {i} assert {is_pass} expected { all_rank_input[src_pe][src_tok_id]} got {dispatch_output[i]}"
                )
                assert False
                # error_round.add(round)
            if dispatch_weights is not None:
                assert torch.equal(
                    dispatch_weights[i], all_rank_weights[src_pe][src_tok_id]
                )
            assert torch.equal(
                dispatch_indices[i], all_rank_indices[src_pe][src_tok_id]
            )
            assert torch.equal(dispatch_scales[i], all_rank_scales[src_pe][src_tok_id])

        if self.rank % self.gpu_per_node == 0:
            print(f"Node {self.rank // self.gpu_per_node} Dispatch Pass")

        combine_output, combine_output_weight = op.combine(
            rec_output,
            dispatch_weights,
            all_rank_indices[self.rank],
            block_num=self.config.block_num,
            warp_per_block=16,
        )
        torch.cuda.synchronize()
        for i in range(all_rank_num_token[self.rank]):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))
            unique_innode_pes = len(
                [
                    pe
                    for pe in set(pes)
                    if (pe // self.gpu_per_node == self.rank // self.gpu_per_node)
                ]
            )
            final_unique_pes = unique_pes
            if final_unique_pes == 0:
                continue

            got, expected = combine_output[i], (
                all_rank_input[self.rank][i].to(torch.float32) * final_unique_pes
            ).to(self.config.data_type)

            ok = torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)
            if not ok:
                print(
                    self.rank,
                    f"token {i} pes {pes} unique pes {unique_pes} unique innode pes {unique_innode_pes}",
                )
                print(
                    f"{self.rank} got: ",
                    got,
                    f"{self.rank} expected: ",
                    expected,
                    all_rank_input[self.rank][i],
                )
                # delta = got.float() - expected.float()
                # print(self.rank, "delta:", delta)
                # error_round.add(round)
                assert False
                # pass
            # else:
            #     print(f"{self.rank} token {i} pass")

            if dispatch_weights is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.rank][i] * final_unique_pes,
                )
                weight_match = torch.allclose(
                    got_weight, expected_weight, atol=1e-5, rtol=1e-5
                )
                if not weight_match and self.config.rank == 0:
                    print(f"Weight mismatch for token {i}:")
                    print(
                        f"  indices[{i}]: {all_rank_indices[self.rank][i].cpu().tolist()}"
                    )
                    print(f"  pes: {pes}")
                    print(f"  unique_pes: {unique_pes}")
                    print(f"  got_weight: {got_weight}")
                    print(
                        f"  expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
                    )
                    print(f"  original weights[{i}]: {all_rank_weights[self.rank][i]}")
                    print(f"  diff: {torch.abs(got_weight - expected_weight)}")
                    print(
                        f"  max_diff: {torch.abs(got_weight - expected_weight).max()}"
                    )
                assert weight_match, f"Weight assertion failed for token {i}"
        if self.rank % self.gpu_per_node == 0:
            print(f"Node {self.rank // self.gpu_per_node} Combine Pass")

    def test_dispatch_combine(self):
        error_round = set()
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(5000):
            if self.rank == 0:
                print(f"Round {i} begin")
            test_data = self.gen_test_data(use_max_token_num=False)
            if self.rank == 0:
                print(f"Round {i} gen test_data done")
            self.run_test_once(op, test_data, error_round, i)
        print(
            "rank: ",
            self.rank,
            "error times: ",
            len(error_round),
            "appear round: ",
            error_round,
        )

        del op

    def stress_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)

        if self.rank == 0:
            print("Stress Test")
        test_data_list = [self.gen_test_data(use_max_token_num=False) for i in range(5)]
        for i in tqdm(range(5000)):
            (
                all_rank_num_token,
                all_rank_indices,
                all_rank_input,
                all_rank_weights,
                all_rank_scales,
            ) = test_data_list[i % 5]
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            _, _ = op.combine(
                dispatch_output,
                dispatch_weights,
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            torch.cuda.synchronize()
            time.sleep(0.0001)

        if self.rank == 0:
            print("Stress Test with CUDA Graph")
        test_data = self.gen_test_data(use_max_token_num=False)
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            _, _ = op.combine(
                dispatch_output,
                dispatch_weights,
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
        torch.cuda.synchronize()

        for i in tqdm(range(5000)):
            g.replay()
            torch.cuda.synchronize()
            time.sleep(0.0001)

        del op

    def run_bench_once(self, op, test_data, repeat=10):
        if args_cli.detailed_profiling:
            num_events = 4 * repeat + 1
        else:
            num_events = 2 * repeat + 1
        events = [torch.cuda.Event(enable_timing=True) for i in range(num_events)]

        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        for i in range(3):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            
            total_recv_num_token = dispatch_recv_num_token[0].item()
            
            # --- Simulated GEMM Start ---
            # EpDispatchCombineTestCase.transform_dispatch_output
            # mori.triton_transform_dispatch_output
            # mori.transform_dispatch_output_gpu
            if args_cli.transform_impl == "gpu":
                packed_input, sorted_indices, expert_counts = mori.transform_dispatch_output_gpu(
                    dispatch_output,
                    dispatch_indices,
                    self.config,
                    total_recv_num_token
                )
            else:
                packed_input, sorted_indices, expert_counts = EpDispatchCombineTestCase.transform_dispatch_output(
                    dispatch_output,
                    dispatch_indices,
                    self.config,
                    total_recv_num_token
                )
            torch.cuda.synchronize()
            # 2. Simulated GEMM (Multiply by 1.0)
            gemm_output = packed_input * 1.0
            # EpDispatchCombineTestCase.inverse_transform_dispatch_output
            # mori.triton_inverse_transform_dispatch_output
            # mori.inverse_transform_dispatch_output_gpu
            if args_cli.transform_impl == "gpu":
                rec_output = mori.inverse_transform_dispatch_output_gpu(
                    gemm_output, sorted_indices, expert_counts, dispatch_output.size(0)
                )
            else:
                rec_output = EpDispatchCombineTestCase.inverse_transform_dispatch_output(
                    gemm_output, sorted_indices, expert_counts, dispatch_output.size(0)
                )

            if total_recv_num_token > 0:
                token_counts = torch.bincount(sorted_indices, minlength=total_recv_num_token)
                valid = token_counts > 0
                rec_output[:total_recv_num_token][valid] = rec_output[:total_recv_num_token][valid] / token_counts[valid].unsqueeze(-1)

            if total_recv_num_token > 0:
                expected_rec = dispatch_output[:total_recv_num_token]
                
                if not torch.allclose(rec_output[:total_recv_num_token], expected_rec, atol=1e-2, rtol=1e-2):
                    print(f"Rank {self.rank}: Reconstruction mismatch!")
                    diff = (rec_output[:total_recv_num_token] - expected_rec).abs()
                    print(f"Max diff: {diff.max().item()}")
                    assert False
            # --- Simulated GEMM End ---


            combine_output, _ = op.combine(
                rec_output,
                dispatch_weights,
                # None,
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            torch.cuda.synchronize()

        total_rdma_recv_num_token = (
            self.config.max_num_inp_token_per_rank * self.config.world_size // 8
        )
        print(
            f"rank {self.rank} recv {total_recv_num_token} tokens {total_rdma_recv_num_token} rdma tokens"
        )

        torch.cuda.synchronize()
        dist.barrier()
        events[0].record()
        for i in range(repeat):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            if args_cli.detailed_profiling:
                events[4 * i + 1].record()

            # EpDispatchCombineTestCase.transform_dispatch_output
            # mori.triton_transform_dispatch_output
            # mori.transform_dispatch_output_gpu
            if args_cli.transform_impl == "gpu":
                packed_input, sorted_indices, expert_counts = mori.transform_dispatch_output_gpu(
                    dispatch_output,
                    dispatch_indices,
                    self.config,
                    total_recv_num_token
                )
            else:
                packed_input, sorted_indices, expert_counts = EpDispatchCombineTestCase.transform_dispatch_output(
                    dispatch_output,
                    dispatch_indices,
                    self.config,
                    total_recv_num_token
                )
            if args_cli.detailed_profiling:
                events[4 * i + 2].record()
            else:
                events[2 * i + 1].record()

            gemm_output = packed_input 
            
            # EpDispatchCombineTestCase.inverse_transform_dispatch_output
            # mori.triton_inverse_transform_dispatch_output
            # mori.inverse_transform_dispatch_output_gpu
            if args_cli.transform_impl == "gpu":
                rec_output = mori.inverse_transform_dispatch_output_gpu(
                    gemm_output, sorted_indices, expert_counts, dispatch_output.size(0)
                )
            else:
                rec_output = EpDispatchCombineTestCase.inverse_transform_dispatch_output(
                    gemm_output, sorted_indices, expert_counts, dispatch_output.size(0)
                )
            if args_cli.detailed_profiling:
                events[4 * i + 3].record()
        
            combine_output, _ = op.combine(
                rec_output,
                dispatch_weights,
                all_rank_indices[self.rank],
                block_num=self.config.block_num,
                warp_per_block=16,
            )
            if args_cli.detailed_profiling:
                events[4 * i + 4].record()
            else:
                events[2 * i + 2].record()
        torch.cuda.synchronize()

        element_size = all_rank_input[self.rank].element_size()
        total_bytes = total_recv_num_token * self.config.hidden_dim * element_size
        ll_mode_scale = (
            self.config.max_num_inp_token_per_rank
            * self.config.num_experts_per_token
            / (total_recv_num_token + 1)  # avoid division by zero
        )
        total_rdma_bytes = (
            total_rdma_recv_num_token * self.config.hidden_dim * element_size
        )

        disp_duration_list = []
        trans_duration_list = []
        inv_duration_list = []
        comb_duration_list = []
        
        if args_cli.detailed_profiling:
            for i in range(repeat):
                base = 4 * i
                disp_duration_list.append(events[base].elapsed_time(events[base + 1]))
                trans_duration_list.append(events[base + 1].elapsed_time(events[base + 2]))
                inv_duration_list.append(events[base + 2].elapsed_time(events[base + 3]))
                comb_duration_list.append(events[base + 3].elapsed_time(events[base + 4]))
        else:
            for i in range(repeat):
                base = 2 * i
                # In non-detailed mode, disp_duration includes transform
                disp_duration_list.append(events[base].elapsed_time(events[base + 1]))
                # In non-detailed mode, comb_duration includes inverse
                comb_duration_list.append(events[base + 1].elapsed_time(events[base + 2]))
                trans_duration_list.append(0.0)
                inv_duration_list.append(0.0)

        disp_rdma_bandwidth_list = [
            total_rdma_bytes / (1000**3) / (t / (10**3)) for t in disp_duration_list
        ]
        disp_bandwidth_list = [
            total_bytes / (1000**3) / (t / (10**3)) for t in disp_duration_list
        ]

        comb_rdma_bandwidth_list = [
            total_rdma_bytes / (1000**3) / (t / (10**3)) for t in comb_duration_list
        ]
        comb_bandwidth_list = [
            total_bytes / (1000**3) / (t / (10**3)) for t in comb_duration_list
        ]

        if args_cli.detailed_profiling:
            disp_trans_duration_list = [d + t for d, t in zip(disp_duration_list, trans_duration_list)]
            inv_comb_duration_list = [i + c for i, c in zip(inv_duration_list, comb_duration_list)]
        else:
            # In non-detailed mode, disp_duration ALREADY includes transform
            disp_trans_duration_list = disp_duration_list
            inv_comb_duration_list = comb_duration_list

        disp_trans_rdma_bandwidth_list = [
            total_rdma_bytes / (1000**3) / (t / (10**3)) for t in disp_trans_duration_list
        ]
        disp_trans_bandwidth_list = [
            total_bytes / (1000**3) / (t / (10**3)) for t in disp_trans_duration_list
        ]
        
        inv_comb_rdma_bandwidth_list = [
            total_rdma_bytes / (1000**3) / (t / (10**3)) for t in inv_comb_duration_list
        ]
        inv_comb_bandwidth_list = [
            total_bytes / (1000**3) / (t / (10**3)) for t in inv_comb_duration_list
        ]

        return (
            disp_duration_list,
            disp_rdma_bandwidth_list,
            disp_bandwidth_list,
            comb_duration_list,
            comb_rdma_bandwidth_list,
            comb_bandwidth_list,
            ll_mode_scale,
            trans_duration_list,
            inv_duration_list,
            disp_trans_duration_list,
            disp_trans_rdma_bandwidth_list,
            disp_trans_bandwidth_list,
            inv_comb_duration_list,
            inv_comb_rdma_bandwidth_list,
            inv_comb_bandwidth_list,
        )

    def bench_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(use_max_token_num=True)

        repeat = 50
        disp_duration_us_list = []
        disp_rdma_bandwidth_GB_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_rdma_bandwidth_GB_list = []
        comb_bandwidth_GB_list = []
        trans_duration_us_list = []
        inv_duration_us_list = []
        disp_trans_duration_us_list = []
        disp_trans_rdma_bandwidth_GB_list = []
        disp_trans_bandwidth_GB_list = []
        inv_comb_duration_us_list = []
        inv_comb_rdma_bandwidth_GB_list = []
        inv_comb_bandwidth_GB_list = []

        error_round = set()
        for i in range(1):
            if self.rank == 0:
                print(f"WarmUp Round {i} begin")
            self.run_test_once(op, test_data, error_round, i)
        assert (
            len(error_round) == 0
        ), f"Warmup failed with errors in rounds: {error_round}"

        (
            disp_duration,
            disp_rdma_bandwidth,
            disp_bandwidth,
            comb_duration,
            comb_rdma_bandwidth,
            comb_bandwidth,
            ll_mode_scale,
            trans_duration,
            inv_duration,
            disp_trans_duration,
            disp_trans_rdma_bandwidth,
            disp_trans_bandwidth,
            inv_comb_duration,
            inv_comb_rdma_bandwidth,
            inv_comb_bandwidth,
        ) = self.run_bench_once(op, test_data, repeat)

        for i in range(repeat):
            disp_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            disp_rdma_bandwidth_output = [
                torch.zeros(1) for _ in range(self.world_size)
            ]
            disp_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            comb_rdma_bandwidth_output = [
                torch.zeros(1) for _ in range(self.world_size)
            ]
            comb_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            trans_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            inv_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            
            disp_trans_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            disp_trans_rdma_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            disp_trans_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            inv_comb_duration_output = [torch.zeros(1) for _ in range(self.world_size)]
            inv_comb_rdma_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]
            inv_comb_bandwidth_output = [torch.zeros(1) for _ in range(self.world_size)]

            dist.all_gather(
                disp_duration_output, torch.tensor([disp_duration[i] * 1000])
            )
            dist.all_gather(
                disp_rdma_bandwidth_output, torch.tensor([disp_rdma_bandwidth[i]])
            )
            dist.all_gather(disp_bandwidth_output, torch.tensor([disp_bandwidth[i]]))
            dist.all_gather(
                comb_duration_output, torch.tensor([comb_duration[i] * 1000])
            )
            dist.all_gather(
                comb_rdma_bandwidth_output, torch.tensor([comb_rdma_bandwidth[i]])
            )
            dist.all_gather(comb_bandwidth_output, torch.tensor([comb_bandwidth[i]]))
            dist.all_gather(trans_duration_output, torch.tensor([trans_duration[i] * 1000]))
            dist.all_gather(inv_duration_output, torch.tensor([inv_duration[i] * 1000]))

            dist.all_gather(disp_trans_duration_output, torch.tensor([disp_trans_duration[i] * 1000]))
            dist.all_gather(disp_trans_rdma_bandwidth_output, torch.tensor([disp_trans_rdma_bandwidth[i]]))
            dist.all_gather(disp_trans_bandwidth_output, torch.tensor([disp_trans_bandwidth[i]]))
            dist.all_gather(inv_comb_duration_output, torch.tensor([inv_comb_duration[i] * 1000]))
            dist.all_gather(inv_comb_rdma_bandwidth_output, torch.tensor([inv_comb_rdma_bandwidth[i]]))
            dist.all_gather(inv_comb_bandwidth_output, torch.tensor([inv_comb_bandwidth[i]]))

            disp_duration_us_list.append([int(t.item()) for t in disp_duration_output])
            disp_rdma_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_rdma_bandwidth_output]
            )
            disp_bandwidth_GB_list.append(
                [int(t.item()) for t in disp_bandwidth_output]
            )
            comb_duration_us_list.append([int(t.item()) for t in comb_duration_output])
            comb_rdma_bandwidth_GB_list.append(
                [int(t.item()) for t in comb_rdma_bandwidth_output]
            )
            comb_bandwidth_GB_list.append(
                [int(t.item()) for t in comb_bandwidth_output]
            )
            trans_duration_us_list.append([int(t.item()) for t in trans_duration_output])
            inv_duration_us_list.append([int(t.item()) for t in inv_duration_output])

            disp_trans_duration_us_list.append([int(t.item()) for t in disp_trans_duration_output])
            disp_trans_rdma_bandwidth_GB_list.append([int(t.item()) for t in disp_trans_rdma_bandwidth_output])
            disp_trans_bandwidth_GB_list.append([int(t.item()) for t in disp_trans_bandwidth_output])
            inv_comb_duration_us_list.append([int(t.item()) for t in inv_comb_duration_output])
            inv_comb_rdma_bandwidth_GB_list.append([int(t.item()) for t in inv_comb_rdma_bandwidth_output])
            inv_comb_bandwidth_GB_list.append([int(t.item()) for t in inv_comb_bandwidth_output])

        if self.rank == 0:
            for i in range(len(disp_duration_us_list)):
                print(f"Round {i}")
                print(
                    f"  dispatch duration {disp_duration_us_list[i]} avg {sum(disp_duration_us_list[i]) / self.config.world_size:.2f} µs"
                )
                print(
                    f"  rdma bandwidth {disp_rdma_bandwidth_GB_list[i]} avg {sum(disp_rdma_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )
                print(
                    f"  bandwidth {disp_bandwidth_GB_list[i]} avg {sum(disp_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )

            for i in range(len(comb_duration_us_list)):
                print(f"Round {i}")
                print(
                    f"  combine duration {comb_duration_us_list[i]} avg {sum(comb_duration_us_list[i]) / self.config.world_size:.2f} µs"
                )
                print(
                    f"  rdma bandwidth {comb_rdma_bandwidth_GB_list[i]} avg {sum(comb_rdma_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )
                print(
                    f"  bandwidth {comb_bandwidth_GB_list[i]} avg {sum(comb_bandwidth_GB_list[i]) / self.config.world_size:.2f} GB/s"
                )

        def collect_metrics(per_round_data):
            minv = min([min(data) for data in per_round_data])
            maxv = max([max(data) for data in per_round_data])
            avgl = [(sum(data) / len(data)) for data in per_round_data]
            avgv = sum(avgl) / len(avgl)
            return int(minv), int(maxv), int(avgv)

        disp_bw = collect_metrics(disp_bandwidth_GB_list[1:])
        disp_rdma_bw = collect_metrics(disp_rdma_bandwidth_GB_list[1:])
        disp_ll_bw = [int(e * ll_mode_scale) for e in disp_bw]
        disp_lat = collect_metrics(disp_duration_us_list[1:])

        comb_bw = collect_metrics(comb_bandwidth_GB_list[1:])
        comb_rdma_bw = collect_metrics(comb_rdma_bandwidth_GB_list[1:])
        comb_ll_bw = [int(e * ll_mode_scale) for e in comb_bw]
        comb_lat = collect_metrics(comb_duration_us_list[1:])

        trans_lat = collect_metrics(trans_duration_us_list[1:])
        inv_lat = collect_metrics(inv_duration_us_list[1:])

        disp_trans_bw = collect_metrics(disp_trans_bandwidth_GB_list[1:])
        disp_trans_rdma_bw = collect_metrics(disp_trans_rdma_bandwidth_GB_list[1:])
        disp_trans_ll_bw = [int(e * ll_mode_scale) for e in disp_trans_bw]
        disp_trans_lat = collect_metrics(disp_trans_duration_us_list[1:])

        inv_comb_bw = collect_metrics(inv_comb_bandwidth_GB_list[1:])
        inv_comb_rdma_bw = collect_metrics(inv_comb_rdma_bandwidth_GB_list[1:])
        inv_comb_ll_bw = [int(e * ll_mode_scale) for e in inv_comb_bw]
        inv_comb_lat = collect_metrics(inv_comb_duration_us_list[1:])

        from prettytable import PrettyTable

        disp_table = PrettyTable()
        comb_table = PrettyTable()
        trans_table = PrettyTable()
        disp_trans_table = PrettyTable()
        inv_comb_table = PrettyTable()

        field_names = [
            "Metrics",
            "RDMA Bandwidth (GB/s)",
            "XGMI Bandwidth (GB/s)",
            "LL Bandwidth (GB/s)",
            "Latency (us)",
        ]
        disp_table.title = "Dispatch Performance"
        disp_table.field_names = field_names
        disp_table.add_rows(
            [
                [
                    "Best",
                    disp_rdma_bw[1],
                    disp_bw[1],
                    disp_ll_bw[1],
                    disp_lat[0],
                ],
                [
                    "Worst",
                    disp_rdma_bw[0],
                    disp_bw[0],
                    disp_ll_bw[0],
                    disp_lat[1],
                ],
                [
                    "Average",
                    disp_rdma_bw[2],
                    disp_bw[2],
                    disp_ll_bw[2],
                    disp_lat[2],
                ],
            ]
        )
        comb_table.field_names = field_names
        comb_table.title = "Combine Performance"
        comb_table.add_rows(
            [
                [
                    "Best",
                    comb_rdma_bw[1],
                    comb_bw[1],
                    comb_ll_bw[1],
                    comb_lat[0],
                ],
                [
                    "Worst",
                    comb_rdma_bw[0],
                    comb_bw[0],
                    comb_ll_bw[0],
                    comb_lat[1],
                ],
                [
                    "Average",
                    comb_rdma_bw[2],
                    comb_bw[2],
                    comb_ll_bw[2],
                    comb_lat[2],
                ],
            ]
        )
        
        if args_cli.detailed_profiling:
            trans_table.title = "Transform Overhead (us)"
            trans_table.field_names = ["Metrics", "Transform", "Inverse"]
            trans_table.add_rows([
                ["Best", trans_lat[0], inv_lat[0]],
                ["Worst", trans_lat[1], inv_lat[1]],
                ["Average", trans_lat[2], inv_lat[2]],
            ])

            disp_trans_table.title = "Dispatch + Transform Performance"
            disp_trans_table.field_names = field_names
            disp_trans_table.add_rows([
                ["Best", disp_trans_rdma_bw[1], disp_trans_bw[1], disp_trans_ll_bw[1], disp_trans_lat[0]],
                ["Worst", disp_trans_rdma_bw[0], disp_trans_bw[0], disp_trans_ll_bw[0], disp_trans_lat[1]],
                ["Average", disp_trans_rdma_bw[2], disp_trans_bw[2], disp_trans_ll_bw[2], disp_trans_lat[2]],
            ])

            inv_comb_table.title = "Inverse + Combine Performance"
            inv_comb_table.field_names = field_names
            inv_comb_table.add_rows([
                ["Best", inv_comb_rdma_bw[1], inv_comb_bw[1], inv_comb_ll_bw[1], inv_comb_lat[0]],
                ["Worst", inv_comb_rdma_bw[0], inv_comb_bw[0], inv_comb_ll_bw[0], inv_comb_lat[1]],
                ["Average", inv_comb_rdma_bw[2], inv_comb_bw[2], inv_comb_ll_bw[2], inv_comb_lat[2]],
            ])

        if self.rank == 0:
            print(disp_table)
            print(comb_table)
            if args_cli.detailed_profiling:
                print(trans_table)
                print(disp_trans_table)
                print(inv_comb_table)

        del op


def test_dispatch_combine(
    local_rank, num_node, gpu_per_node, max_tokens, total_experts, kernel_type, cmd="test"
):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank

    test_case = EpDispatchCombineTestCase(
        global_rank,
        gpu_per_node,
        world_size,
        max_tokens,
        total_experts,
        kernel_type,
        torch.bfloat16,
        # torch.float8_e4m3fnuz,
    )
    test_case.setup()
    if cmd == "test":
        test_case.test_dispatch_combine()
    elif cmd == "bench":
        test_case.bench_dispatch_combine()
    elif cmd == "stress":
        test_case.stress_dispatch_combine()
    else:
        raise ValueError(f"unsupported command: {cmd}")

    test_case.cleanup()


parser = argparse.ArgumentParser(description="dispatch/combine internode test")
parser.add_argument(
    "--cmd",
    type=str,
    default="test",
    choices=["test", "bench", "stress"],
    help="Available subcommands: test, bench, stress",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum number of input tokens per rank (default: 4096)",
)

parser.add_argument(
    "--total-experts",
    type=int,
    default=288, # 
    help="Maximum number of input tokens per rank (default: 288)",
)
parser.add_argument(
    "--kernel-type",
    type=str,
    default="v1",
    help="Type of kernel to test",
    choices=["v0", "v1", "v1_ll"],
)
parser.add_argument(
    "--transform-impl",
    type=str,
    default="gpu",
    help="Transform backend: gpu|python",
    choices=["gpu", "python"],
)
parser.add_argument(
    "--detailed-profiling",
    action="store_true",
    help="Enable detailed profiling of transform kernels (may add overhead)",
)
args_cli = parser.parse_args()

# Utility: print available CUDA devices
def print_available_cuda_devices():
    try:
        count = torch.cuda.device_count()
        print(f"CUDA available: {torch.cuda.is_available()} | device_count: {count}")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            print(f"  [{i}] {name}")
    except Exception as e:
        print(f"Failed to query CUDA devices: {e}")

if __name__ == "__main__":
    # Print GPU inventory before spawning workers
    print_available_cuda_devices()

    gpu_per_node = os.environ.get("GPU_PER_NODE", None)
    gpu_per_node = int(gpu_per_node) if gpu_per_node is not None else 8
    num_node = int(os.environ["WORLD_SIZE"])

    world_size = num_node * gpu_per_node
    
    assert args_cli.num_total_experts % num_ranks == 0, "num_experts must be divisible by world_size"
    num_experts_per_rank = args_cli.num_total_experts // world_size
    print(f"num_experts_per_rank: {num_experts_per_rank} | world_size: {world_size} | gpu_per_node: {gpu_per_node} | num_node: {num_node}")
    torch.multiprocessing.spawn(
        test_dispatch_combine,
        args=(
            num_node,
            gpu_per_node,
            args_cli.max_tokens,
            args_cli.num_total_experts,
            args_cli.kernel_type,
            args_cli.cmd,
        ),
        nprocs=gpu_per_node,
        join=True,
    )
