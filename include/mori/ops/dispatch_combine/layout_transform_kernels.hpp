#pragma once

#include <hip/hip_runtime.h>
#include "mori/core/core.hpp"

namespace mori {
namespace moe {

template <typename T>
void LaunchTransformDispatchOutput(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_n, int64_t stride_src_h,
    int64_t stride_dst_e, int64_t stride_dst_c, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

template <typename T>
void LaunchInverseTransformDispatchOutput(
    const T* src, T* dst,
    const index_t* indices, const index_t* expert_ids, const index_t* slot_ids,
    int64_t stride_src_e, int64_t stride_src_c, int64_t stride_src_h,
    int64_t stride_dst_n, int64_t stride_dst_h,
    int num_tokens, int H, hipStream_t stream);

}  // namespace moe
}  // namespace mori
