#include "mori/ops/dispatch_combine/layout_transform_kernels.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

namespace mori {
namespace moe {

// Helper for AtomicAdd
template <typename T>
__device__ inline void AtomicAdd(T* address, T val) {
    atomicAdd(address, val);
}

// Specialization for bfloat16 if needed, or rely on HIP
// HIP 5.x+ supports atomicAdd for __hip_bfloat16

template <typename T>
__global__ void TransformDispatchOutputKernel(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H) {
    
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    if (warpId >= num_tokens) return;

    // Metadata load (broadcast from lane 0)
    index_t src_idx, expert_id, slot_id;
    if (laneId == 0) {
        src_idx = indices[warpId];
        expert_id = expert_ids[warpId];
        slot_id = slot_ids[warpId];
    }
    src_idx = __shfl(src_idx, 0);
    expert_id = __shfl(expert_id, 0);
    slot_id = __shfl(slot_id, 0);

    const T* src_row = src + src_idx * stride_src_n;
    T* dst_row = dst + expert_id * stride_dst_e + slot_id * stride_dst_c;

    if (stride_src_h == 1 && stride_dst_h == 1) {
         core::WarpCopy(dst_row, src_row, H);
    } else {
        for (int i = laneId; i < H; i += warpSize) {
            dst_row[i * stride_dst_h] = src_row[i * stride_src_h];
        }
    }
}

template <typename T>
__device__ inline void VectorizedAtomicAdd(T* dst, T val) {
    atomicAdd(dst, val);
}

// Specialization for __half to use __half2 optimization if possible
// Note: This assumes 4-byte alignment for __half2 access
template <>
__device__ inline void VectorizedAtomicAdd<__half>(__half* dst, __half val) {
    atomicAdd(dst, val);
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
    int num_tokens, int H, hipStream_t stream) {
    
    int block_size = 256;
    int warps_per_block = block_size / 64; // Assuming warpSize 64 for AMD
    // Better to use device property or macro, but 64 is safe for AMD
    // core::DeviceWarpSize() is available
    
    int num_blocks = (num_tokens + warps_per_block - 1) / warps_per_block;
    
    TransformDispatchOutputKernel<T><<<num_blocks, block_size, 0, stream>>>(
        src, dst, indices, expert_ids, slot_ids,
        stride_src_n, stride_src_h,
        stride_dst_e, stride_dst_c, stride_dst_h,
        num_tokens, H
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
template void LaunchTransformDispatchOutput<float>(
    const float* src, float* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

template void LaunchTransformDispatchOutput<__half>(
    const __half* src, __half* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

template void LaunchTransformDispatchOutput<__hip_bfloat16>(
    const __hip_bfloat16* src, __hip_bfloat16* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

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

}  // namespace moe
}  // namespace mori
