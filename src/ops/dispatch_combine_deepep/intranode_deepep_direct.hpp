// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine_deepep/dispatch_combine_deepep.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {
namespace deepep {

/*
 * Skeleton for DeepEP-format intra-node dispatch/combine kernels.
 * These are intentionally separate from the existing MORI kernels to avoid
 * behavioral changes while we prototype direct-format support.
 */

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint32_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
    args.combineGridBarrier[0] = 0;
    // Use system-scope release for cross-GPU visibility with proper ordering
    detail::AtomicStoreReleaseSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    // Use system-scope acquire for cross-GPU visibility with proper ordering
    while (detail::AtomicLoadAcquireSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}



// Low-latency variants with optional fp8 quant/dequant similar to DeepEP.
namespace detail {
constexpr int kNumPerChannels = 128;
constexpr int kElemsPerLoad = 8;  // uint4 = 16 bytes = 8 bf16 elements
constexpr float kFP8Margin = 1e-4f;

// Finish counter pattern: when counter reaches this value, all tokens for that destPe are dispatched
// The pattern works as: finish_counter = dispatched_count + (FINISHED_SUM_TAG - expected_count)
// When dispatched_count == expected_count, finish_counter == FINISHED_SUM_TAG
constexpr uint32_t kFinishedSumTag = 0x80000000u;  // High bit set to avoid collision with counts
#ifdef __HIP_PLATFORM_AMD__
constexpr float kFP8Amax = 240.0f;
constexpr float kFP8AmaxInv = 1.0f / 240.0f;
#else
constexpr float kFP8Amax = 448.0f;
constexpr float kFP8AmaxInv = 1.0f / 448.0f;
#endif

__device__ inline float DeepepFp8Scale(float amax) {
  float safeAmax = fmaxf(amax, kFP8Margin);
  return kFP8Amax / safeAmax;
}

__device__ inline float DeepepFp8ScaleInv(float amax) {
  float safeAmax = fmaxf(amax, kFP8Margin);
  return safeAmax * kFP8AmaxInv;
}

__device__ inline __hip_fp8_storage_t CastFloatToFp8(float v, float scale) {
#ifdef __HIP_PLATFORM_AMD__
  return __hip_cvt_float_to_fp8(v * scale, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#else
  return __hip_cvt_float_to_fp8(v * scale, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#endif
}

// Quarter-warp reduction for 16 threads (used for 128-element channel with 8 elements per thread)
__device__ inline float QuarterWarpReduceMax(float value) {
  value = fmaxf(value, __shfl_xor(value, 8));
  value = fmaxf(value, __shfl_xor(value, 4));
  value = fmaxf(value, __shfl_xor(value, 2));
  value = fmaxf(value, __shfl_xor(value, 1));
  return value;
}

// CastFp8ToFloat unused in current LL combine path (caller dequantizes fp8 outputs).
}  // namespace detail

template <typename T, bool kUseFP8>
__global__ void EpDispatchIntraNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int myPe = config.rank;
  int npes = config.worldSize;
  const index_t expertCapacity = config.worldSize * config.maxNumInpTokenPerRank;
  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];

  // Reset local counters and synchronize all ranks before using remote counters.
  if (globalWarpId == 0 && laneId == 0) {
    index_t* localExpertCounter =
        args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      localExpertCounter[e] = 0;
    }
    for (int pe = 0; pe < npes; ++pe) {
      args.destPeTokenCounter[pe] = 0;
    }
  }
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);

  if (args.tokenIndices && args.inpTokenBuf) {
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken; i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;
      index_t destTokId = 0;

      if (laneId == 0) {
        index_t* expertCounter =
            args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe);
        destTokId = atomicAdd(expertCounter + localExpert, 1);
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        args.dispDestTokIdMap[i] = destExpert * expertCapacity + destTokId;
      }
      destTokId = __shfl(destTokId, 0);
      index_t destLinearTok = localExpert * expertCapacity + destTokId;

      if (laneId == 0) {
        // Use system-scope release for cross-GPU visibility with proper ordering
        detail::AtomicStoreReleaseSystem(
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destLinearTok,
            static_cast<index_t>(myPe * config.maxNumInpTokenPerRank + srcTokId));
      }

      if (laneId < config.numExpertPerToken) {
        if (args.weightsBuf) {
          args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
          destPe)[destLinearTok * config.numExpertPerToken + laneId] =
          args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
        }
        args.shmemOutIndicesMemObj->template GetAs<index_t*>(
        destPe)[destLinearTok * config.numExpertPerToken + laneId] =
        args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
      }

      size_t baseOffset = destLinearTok * config.hiddenDim;

      // Copy pre-computed scales for non-FP8 path (FP8 scales are computed inline below)
      if constexpr (!kUseFP8) {
        if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
          index_t destScaleOffset =
              destLinearTok * config.scaleDim * config.scaleTypeSize;
          index_t srcScaleOffset = srcTokId * config.scaleDim * config.scaleTypeSize;
          core::WarpCopy(
              args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
              args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
        }
      }

      index_t srcTokOffset = srcTokId * config.hiddenDim;
      if constexpr (kUseFP8) {
        auto* destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
          args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe)) + baseOffset;
        int numScales = config.hiddenDim / detail::kNumPerChannels;
        auto* srcPtr = reinterpret_cast<const uint4*>(args.inpTokenBuf + srcTokOffset);
        auto* destPtr = reinterpret_cast<uint64_t*>(destFp8);
        float* scalePtr = args.shmemOutScalesMemObj->template GetAs<float*>(destPe) +
                          destLinearTok * numScales;

        // Process 4 channels (512 elements) per iteration using vectorized loads
        // 64 threads * 8 elements/load = 512 elements = 4 * 128-element channels
        // Threads 0-15 -> channel 0, threads 16-31 -> channel 1, etc.
        constexpr int kChannelsPerIter = warpSize * detail::kElemsPerLoad / detail::kNumPerChannels;
        static_assert(kChannelsPerIter == 4, "Expected 4 channels per iteration");

        for (int scaleBase = 0; scaleBase < numScales; scaleBase += kChannelsPerIter) {
          // Vectorized load: each thread reads 8 bf16 elements (16 bytes)
          uint4 packed = srcPtr[scaleBase * detail::kNumPerChannels / detail::kElemsPerLoad + laneId];
          auto* bf16Values = reinterpret_cast<const T*>(&packed);

          // Convert to float and compute local amax (8 elements per thread)
          float fp32Values[detail::kElemsPerLoad];
          float amax = detail::kFP8Margin;
          #pragma unroll
          for (int j = 0; j < detail::kElemsPerLoad; ++j) {
            fp32Values[j] = static_cast<float>(bf16Values[j]);
            amax = fmaxf(amax, fabsf(fp32Values[j]));
          }

          // Quarter-warp reduction: 16 threads per channel (128 elements / 8 per thread)
          amax = detail::QuarterWarpReduceMax(amax);
          float scale = detail::kFP8Amax / amax;
          float scaleInv = amax * detail::kFP8AmaxInv;

          // Store scale (only lane 0, 16, 32, 48 write their respective channel's scale)
          if ((laneId & 15) == 0) {
            scalePtr[scaleBase + (laneId >> 4)] = scaleInv;
          }

          // Quantize using vectorized FP8 conversion
          uint64_t fp8Packed;
          auto* fp8x2Values = reinterpret_cast<__hip_fp8x2_storage_t*>(&fp8Packed);
          #pragma unroll
          for (int j = 0; j < detail::kElemsPerLoad; j += 2) {
            float2 fp32x2 = {fp32Values[j] * scale, fp32Values[j + 1] * scale};
#if defined(__gfx942__)
            fp8x2Values[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#elif defined(__gfx950__)
            fp8x2Values[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3);
#else
            // Fallback for other architectures
            fp8x2Values[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#endif
          }

          // Vectorized store: 8 FP8 values (8 bytes) per thread
          destPtr[scaleBase * detail::kNumPerChannels / detail::kElemsPerLoad + laneId] = fp8Packed;
        }
      } else {
        core::WarpCopy(args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + baseOffset,
                       args.inpTokenBuf + srcTokOffset, config.hiddenDim);
      }

      // Phase 3 optimization: Increment finish counter for this destPe after copying token data.
      // Uses release semantics to ensure all prior writes are visible.
      if (laneId == 0) {
        detail::AtomicAddReleaseSystem(args.finishCounterPerDestPe + destPe, 1u);
      }
    }
  }

  // Phase 3: Count warp computes expected tokens per destPe and adds offset to finish counters.
  // This runs after the dispatch loop for the count warp. Uses the last warp.
  // The finish counter pattern: when finish_counter[destPe] == kFinishedSumTag, all tokens are done.
  // Math: finish_counter = dispatched_count + (kFinishedSumTag - expected_count) = kFinishedSumTag
  constexpr int kMaxGpuPerNode = 8;  // MI300X has 8 GPUs per node
  assert(npes <= kMaxGpuPerNode && "Intranode dispatch assumes worldSize <= 8 GPUs per node");
  if (globalWarpId == globalWarpNum - 1) {
    // Count expected tokens per destPe by reading tokenIndices
    // Use registers to accumulate counts per destPe (worldSize is typically small, 8 for intranode)
    int countPerDestPe[kMaxGpuPerNode] = {0};
    for (int i = laneId; i < args.curRankNumToken * config.numExpertPerToken; i += warpSize) {
      if (args.tokenIndices) {
        index_t destExpert = args.tokenIndices[i];
        index_t destPe = destExpert / config.numExpertPerRank;
        if (destPe < npes && destPe < kMaxGpuPerNode) {
          countPerDestPe[destPe]++;
        }
      }
    }
    // Warp-reduce counts per destPe and add offset to finish counters
    for (int pe = 0; pe < npes && pe < kMaxGpuPerNode; ++pe) {
      int count = countPerDestPe[pe];
      // Warp reduction using shuffle
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        count += __shfl_xor(count, offset);
      }
      // Lane 0 adds the offset to finish counter
      if (laneId == 0) {
        uint32_t finishOffset = detail::kFinishedSumTag - static_cast<uint32_t>(count);
        detail::AtomicAddReleaseSystem(args.finishCounterPerDestPe + pe, finishOffset);
      }
    }
  }

  // Signal phase: Wait for finish counter per destPe instead of global barrier.
  // The finish counter pattern allows signaling per-destPe as soon as all tokens for that
  // destPe are dispatched, without waiting for all warps globally. This is the key optimization.
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait for finish counter to reach kFinishedSumTag (all tokens for this destPe dispatched)
      while (detail::AtomicLoadAcquireSystem(args.finishCounterPerDestPe + destPe) != detail::kFinishedSumTag) {
      }

      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      // Use system-scope release for cross-GPU visibility with proper ordering
      detail::AtomicStoreReleaseSystem(signal, numTokenSignal);
    }
  }

  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      // Use system-scope release to reset signal
      detail::AtomicStoreReleaseSystem(signal, static_cast<index_t>(0));
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[destPe] = 0;
    }
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        args.recvTokenCountPerExpert[e] = localExpertCounter[e];
      }
      // reset per-expert counters for this rank
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        localExpertCounter[e] = 0;
      }
    }
  }
}

template <typename T, bool kUseFP8, bool kUseWeights>
__global__ void EpCombineIntraNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;


  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  const index_t expertCapacity = config.worldSize * config.maxNumInpTokenPerRank;
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  // Step 1: copy only valid expert slots into symmetric combine buffer (visible to all ranks).
  if (args.config.useExternalInpBuffer) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      index_t count = args.recvTokenCountPerExpert ? args.recvTokenCountPerExpert[e] : 0;
      for (int slot = globalWarpId; slot < count; slot += globalWarpNum) {
        index_t linear = e * expertCapacity + slot;
        core::WarpCopy(
            args.shmemCombineInpTokMemObj->template GetAs<T*>() + linear * config.hiddenDim,
            args.inpTokenBuf + linear * config.hiddenDim,
            config.hiddenDim);
      }
    }
  }

  if (args.weightsBuf) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      index_t count = args.recvTokenCountPerExpert ? args.recvTokenCountPerExpert[e] : 0;
      for (int slot = globalWarpId; slot < count; slot += globalWarpNum) {
        index_t linear = e * expertCapacity + slot;
        core::WarpCopy(
            args.shmemInpWeightsMemObj->template GetAs<float*>() + linear * config.numExpertPerToken,
            args.weightsBuf + linear * config.numExpertPerToken,
            config.numExpertPerToken);
      }
    }
  }

  // Step 2: cross-rank barrier so all writes are visible before any rank reads.
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
  *args.totalRecvTokenNum = 0;
  if (args.curRankNumToken == 0) {
    return;
  }

    extern __shared__ char sharedMem[];
    auto* smem = reinterpret_cast<uint8_t*>(sharedMem);
    auto* srcPtrsBase = reinterpret_cast<T**>(smem);
    size_t smemOffset = warpNum * config.numExpertPerToken * sizeof(T*);
    auto* srcWeightsPtrBase = reinterpret_cast<float**>(smem + smemOffset);
    smemOffset += warpNum * config.numExpertPerToken * sizeof(float*);
    float* srcWeightScalesBase = nullptr;
    if constexpr (kUseWeights) {
      srcWeightScalesBase = reinterpret_cast<float*>(smem + smemOffset);
    }

    T** srcPtrs = srcPtrsBase + warpId * config.numExpertPerToken;
    float** srcWeightsPtr = srcWeightsPtrBase + warpId * config.numExpertPerToken;
    float* srcWeightScales = kUseWeights ? (srcWeightScalesBase + warpId * config.numExpertPerToken) : nullptr;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  assert(config.numExpertPerToken < warpSize);
  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        max(0, min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      // Step 3: map each top-k expert to (dest_pe, local_expert, slot).
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      if (destPe < config.worldSize) {
        // Step 4: read remote expert-major slot from symmetric buffers.
        size_t baseOffset = (localExpert * expertCapacity + destLocalTokId) * config.hiddenDim;
        srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                     baseOffset + hiddenDimOffset;
        srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                   (localExpert * expertCapacity + destLocalTokId) *
                               config.numExpertPerToken;
      } else {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
      }

      if constexpr (kUseWeights) {
        float w = 1.0f;
        if (args.weightsBuf && srcWeightsPtr[j] != nullptr) {
          w = srcWeightsPtr[j][j];
        }
        srcWeightScales[j] = w;
      }
    }

    // Step 5: accumulate into local output buffer.
    core::WarpAccum<T, 4>(args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                  tokenId * config.hiddenDim + hiddenDimOffset,
                srcPtrs, kUseWeights ? srcWeightScales : nullptr,
                config.numExpertPerToken, hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtr, nullptr, config.numExpertPerToken,
                                config.numExpertPerToken);
    }
  }
}

}  // namespace deepep
}  // namespace moe
}  // namespace mori
