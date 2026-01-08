#pragma once

#include <hip/hip_runtime.h>
#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include <hip/hip_fp8.h>
#include <hip/library_types.h>

namespace mori {
namespace moe {

template <typename T>
void LaunchTransformDispatchOutput(
    const T* src, T* dst,
    const float* scales_src, float* scales_dst,
    const index_t * indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, int scale_dim, hipStream_t stream,
    const int* num_tokens_ptr = nullptr);

template <typename T>
void LaunchInverseTransformDispatchOutput(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

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
    hipStream_t stream);

}  // namespace moe
}  // namespace mori
