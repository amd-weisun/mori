#include "mori/ops/dispatch_combine/layout_transform_kernels.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"
#include <algorithm>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

namespace mori {
namespace moe {

// Helper for AtomicAdd
template <typename T>
__device__ inline void AtomicAdd(T* address, T val) {
    atomicAdd(address, val);
}

// Specialization for __half
template <>
__device__ inline void AtomicAdd<__half>(__half* address, __half val) {
#if defined(__HIP_PLATFORM_AMD__)
    unsafeAtomicAdd(address, val);
#else
    atomicAdd(address, val);
#endif
}

// Specialization for __hip_bfloat16
template <>
__device__ inline void AtomicAdd<__hip_bfloat16>(__hip_bfloat16* address, __hip_bfloat16 val) {
#if defined(__HIP_PLATFORM_AMD__)
    unsafeAtomicAdd(address, val);
#else
    atomicAdd(address, val);
#endif
}

template <typename T>
__global__ void TransformDispatchOutputKernel(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, const int* num_tokens_ptr) {
    
    constexpr int kTileH = 256;
    int tiles_per_token = std::max(1, (H + kTileH - 1) / kTileH);

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int laneId = threadIdx.x % warpSize;

    int valid_tokens = num_tokens;
    if (num_tokens_ptr) {
        valid_tokens = *num_tokens_ptr;
    }
    if (valid_tokens == 0) return;

    int total_valid_warps = valid_tokens * tiles_per_token;
    if (warpId >= total_valid_warps) return;

    int tile_id = warpId % tiles_per_token;
    int token_id = warpId / tiles_per_token;
    if (token_id >= valid_tokens) return;

    // Metadata load (broadcast from lane 0)
    index_t src_idx, expert_id, slot_id;
    if (laneId == 0) {
        src_idx = indices[token_id];
        expert_id = expert_ids[token_id];
        slot_id = slot_ids[token_id];
    }
    src_idx = __shfl(src_idx, 0);
    expert_id = __shfl(expert_id, 0);
    slot_id = __shfl(slot_id, 0);

    const T* src_row = src + src_idx * stride_src_n;
    T* dst_row = dst + expert_id * stride_dst_e + slot_id * stride_dst_c;

    int tile_start = tile_id * kTileH;
    int tile_end = std::min(tile_start + kTileH, H);
    int tile_len = tile_end - tile_start;
    if (tile_len <= 0) return;

    const T* src_tile = src_row + tile_start;
    T* dst_tile = dst_row + tile_start;

    if (stride_src_h == 1 && stride_dst_h == 1) {
         core::WarpCopy(dst_tile, src_tile, tile_len);
    } else {
        for (int i = laneId; i < tile_len; i += warpSize) {
            int global_idx = tile_start + i;
            dst_row[global_idx * stride_dst_h] = src_row[global_idx * stride_src_h];
        }
    }
}

template <typename T>
__device__ inline void VectorizedAtomicAdd(T* dst, T val) {
    AtomicAdd(dst, val);
}

// Specialization for __half to use __half2 optimization if possible
// Note: This assumes 4-byte alignment for __half2 access
template <>
__device__ inline void VectorizedAtomicAdd<__half>(__half* dst, __half val) {
    AtomicAdd(dst, val);
}

template <typename T>
__global__ void InverseTransformDispatchOutputKernel(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H) {
    
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    if (warpId >= num_tokens) return;

    index_t dst_idx, expert_id, slot_id;
    if (laneId == 0) {
        dst_idx = indices[warpId];
        expert_id = expert_ids[warpId];
        slot_id = slot_ids[warpId];
    }
    dst_idx = __shfl(dst_idx, 0);
    expert_id = __shfl(expert_id, 0);
    slot_id = __shfl(slot_id, 0);

    const T* src_row = src + expert_id * stride_src_e + slot_id * stride_src_c;
    T* dst_row = dst + dst_idx * stride_dst_n;

    // Fast path: Contiguous memory (stride_h == 1)
    if (stride_src_h == 1 && stride_dst_h == 1) {
        // Vectorized Load Optimization
        // We load 16 bytes (128 bits) at a time
        constexpr int VecBytes = 16;
        constexpr int ElemsPerVec = VecBytes / sizeof(T);
        using VecType = typename core::VecTypeSelector<VecBytes>::dataType; // usually uint4 or float4

        int vec_limit = (H / ElemsPerVec) * ElemsPerVec;

        // Main vectorized loop
        for (int i = laneId * ElemsPerVec; i < vec_limit; i += warpSize * ElemsPerVec) {
            // Load 16 bytes
            VecType vec_val = *reinterpret_cast<const VecType*>(src_row + i);
            T* vals = reinterpret_cast<T*>(&vec_val);

            // Unpack and Atomic Add
            // Compiler should unroll this loop
            #pragma unroll
            for (int k = 0; k < ElemsPerVec; ++k) {
                AtomicAdd(dst_row + i + k, vals[k]);
            }
        }

        // Cleanup loop for remaining elements
        for (int i = vec_limit + laneId; i < H; i += warpSize) {
            AtomicAdd(dst_row + i, src_row[i]);
        }
    } else {
        // Slow path: Strided memory
        for (int i = laneId; i < H; i += warpSize) {
            AtomicAdd(dst_row + i * stride_dst_h, src_row[i * stride_src_h]);
        }
    }
}

template <typename T>
void LaunchTransformDispatchOutput(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream,
    const int* num_tokens_ptr) {
    // num_tokens: 
    
    constexpr int kTileH = 256;
    int tiles_per_token = std::max(1, (H + kTileH - 1) / kTileH);
    int num_warps = num_tokens * tiles_per_token;
    int block_size = 256;
    int warps_per_block = block_size / 64; // Assuming warpSize 64 for AMD
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    
    TransformDispatchOutputKernel<T><<<num_blocks, block_size, 0, stream>>>(
        src, dst, indices, expert_ids, slot_ids,
        stride_src_n, stride_src_h,
        stride_dst_e, stride_dst_c, stride_dst_h,
        num_tokens, H, num_tokens_ptr
    );
}

template <typename T>
void LaunchInverseTransformDispatchOutput(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream) {
    
    int block_size = 256;
    int warps_per_block = block_size / 64; 
    int num_blocks = (num_tokens + warps_per_block - 1) / warps_per_block;
    
    InverseTransformDispatchOutputKernel<T><<<num_blocks, block_size, 0, stream>>>(
        src, dst, indices, expert_ids, slot_ids,
        stride_src_e, stride_src_c, stride_src_h,
        stride_dst_n, stride_dst_h,
        num_tokens, H
    );
}

// Explicit Instantiation
template void LaunchTransformDispatchOutput<__half>(
    const __half* src, __half* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream, const int* num_tokens_ptr);

template void LaunchTransformDispatchOutput<__hip_bfloat16>(
    const __hip_bfloat16* src, __hip_bfloat16* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream, const int* num_tokens_ptr);

template void LaunchTransformDispatchOutput<float>(
    const float* src, float* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream, const int* num_tokens_ptr);

template void LaunchInverseTransformDispatchOutput<float>(
    const float* src, float* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

template void LaunchInverseTransformDispatchOutput<__half>(
    const __half* src, __half* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

template void LaunchInverseTransformDispatchOutput<__hip_bfloat16>(
    const __hip_bfloat16* src, __hip_bfloat16* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

// -----------------------------------------------------------------------------
// Metadata Preparation Kernels
// -----------------------------------------------------------------------------

__global__ void CountExpertsKernel(
    const index_t* dispatch_indices,
    index_t* expert_counts,
    int64_t num_tokens,
    int64_t K,
    int64_t num_experts_per_rank,
    int64_t rank) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * K) return;

    index_t expert_idx = dispatch_indices[idx];
    if ((expert_idx / num_experts_per_rank) == rank) {
        index_t local_expert = expert_idx % num_experts_per_rank;
        atomicAdd(&expert_counts[local_expert], 1);
    }
}

// Single block kernel to compute prefix sum of expert counts
// E is typically small (e.g., 8, 16, 64, 256). Max 1024 threads covers most cases.
__global__ void PrefixSumKernel(
    index_t* expert_counts,
    index_t* offsets,
    int* total_valid_count,
    int64_t num_experts) {
    
    // Shared memory for scan
    // Assuming num_experts <= 1024 for single block scan
    // If > 1024, we need a more complex scan or multiple blocks.
    // For now, assume E <= 1024.
    
    extern __shared__ index_t temp[];
    int tid = threadIdx.x;
    
    if (tid < num_experts) {
        temp[tid] = expert_counts[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Hillis-Steele Scan (Inclusive)
    for (int stride = 1; stride < num_experts; stride *= 2) {
        index_t val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // Write to offsets (Exclusive Scan needed for offsets)
    // temp[tid] is inclusive sum.
    // offsets[tid] = temp[tid] - expert_counts[tid]
    if (tid < num_experts) {
        offsets[tid] = temp[tid] - expert_counts[tid];
    }
    
    if (tid == num_experts - 1) {
        *total_valid_count = temp[tid];
    }
}

__global__ void ScatterIndicesKernel(
    const index_t* dispatch_indices,
    index_t* sorted_token_indices,
    index_t* sorted_expert_ids,
    index_t* slot_indices,
    index_t* current_offsets, // Initialized with offsets
    const index_t* base_offsets, // Original offsets for slot calculation
    int64_t num_tokens,
    int64_t K,
    int64_t num_experts_per_rank,
    int64_t rank) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * K) return;

    index_t expert_idx = dispatch_indices[idx];
    if ((expert_idx / num_experts_per_rank) == rank) {
        index_t local_expert = expert_idx % num_experts_per_rank;
        
        // Atomic increment to get the slot
        index_t write_pos = atomicAdd(&current_offsets[local_expert], 1);
        
        index_t token_idx = idx / K;
        
        sorted_token_indices[write_pos] = token_idx;
        sorted_expert_ids[write_pos] = local_expert;
        slot_indices[write_pos] = write_pos - base_offsets[local_expert];
    }
}

void LaunchPrepareTransformMetadata(
    const index_t* dispatch_indices,
    index_t* sorted_token_indices,
    index_t* sorted_expert_ids,
    index_t* slot_indices,
    index_t* expert_counts,
    int* total_valid_count,
    int64_t num_tokens,
    int64_t K,
    int64_t num_experts_per_rank,
    int64_t rank,
    hipStream_t stream) {
    
    // 1. Zero out expert counts
    (void)hipMemsetAsync(expert_counts, 0, num_experts_per_rank * sizeof(index_t), stream);
    
    // 2. Count Experts
    int block_size = 256;
    int64_t total_items = num_tokens * K;
    int num_blocks = (total_items + block_size - 1) / block_size;
    
    CountExpertsKernel<<<num_blocks, block_size, 0, stream>>>(
        dispatch_indices, expert_counts, num_tokens, K, num_experts_per_rank, rank
    );
    
    // 3. Prefix Sum (Scan)
    // We need temporary storage for offsets.
    // Since we don't want to allocate inside the function, we can reuse 'slot_indices' or similar if safe?
    // No, we need a dedicated buffer for offsets to track atomic adds.
    // We can allocate a small buffer on the stream ordered allocator if available, or just use a raw hipMallocAsync.
    // Since E is small, hipMallocAsync is fine.
    
    index_t* d_offsets;
    index_t* d_base_offsets;
    (void)hipMallocAsync(&d_offsets, num_experts_per_rank * sizeof(index_t), stream);
    (void)hipMallocAsync(&d_base_offsets, num_experts_per_rank * sizeof(index_t), stream);
    
    // Launch Scan
    // Shared memory size: num_experts * sizeof(index_t)
    int scan_threads = 1024; 
    if (num_experts_per_rank > 1024) {
        // Fallback or error. For now assume E <= 1024.
        // In production, use rocPrim::inclusive_scan.
    }
    
    PrefixSumKernel<<<1, scan_threads, num_experts_per_rank * sizeof(index_t), stream>>>(
        expert_counts, d_base_offsets, total_valid_count, num_experts_per_rank
    );
    
    // Copy base offsets to current offsets for atomic incrementing
    (void)hipMemcpyAsync(d_offsets, d_base_offsets, num_experts_per_rank * sizeof(index_t), hipMemcpyDeviceToDevice, stream);
    
    // 4. Scatter
    ScatterIndicesKernel<<<num_blocks, block_size, 0, stream>>>(
        dispatch_indices,
        sorted_token_indices,
        sorted_expert_ids,
        slot_indices,
        d_offsets,
        d_base_offsets,
        num_tokens, K, num_experts_per_rank, rank
    );
    
    (void)hipFreeAsync(d_offsets, stream);
    (void)hipFreeAsync(d_base_offsets, stream);
}

}  // namespace moe
}  // namespace mori
