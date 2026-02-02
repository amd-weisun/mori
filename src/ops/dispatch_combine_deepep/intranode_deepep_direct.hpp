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
// Phase 4: Optimized CrossDeviceBarrier with device-scope atomics
// Aligned with DeepEP's grid_barrier pattern but extended for cross-device synchronization.
// Uses device-scope relaxed atomics for counters (xGMI is cache-coherent).
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint32_t crossDeviceBarrierFlag) {
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpNum = gridDim.x * warpNum;

  // Step 1: Intra-grid barrier (all warps arrive)
  // Use device-scope relaxed atomic for counter increment
  if (laneId == 0) {
    detail::AtomicAddRelaxed(args.combineGridBarrier, 1u);
  }

  // Step 2: First few threads signal to remote ranks after local grid barrier completes
  if (globalThdId < args.config.worldSize) {
    // Wait for all warps to arrive (device-scope poll)
    while (detail::AtomicLoadRelaxed(args.combineGridBarrier) != static_cast<uint32_t>(globalWarpNum)) {
    }
    // Reset barrier for next use
    args.combineGridBarrier[0] = 0;
    // Signal to remote rank using device-scope release (xGMI is cache-coherent)
    detail::AtomicStoreRelease(
        args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  // Step 3: Increment flag counter (device-scope relaxed)
  if (globalThdId == 0) {
    detail::AtomicAddRelaxed(args.crossDeviceBarrierFlag, 1u);
  }

  // Step 4: Wait for signals from all remote ranks (device-scope acquire)
  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (detail::AtomicLoadAcquire(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}



// Low-latency variants with optional fp8 quant/dequant similar to DeepEP.
namespace detail {
constexpr int kNumPerChannels = 128;
constexpr int kElemsPerLoad = 8;  // uint4 = 16 bytes = 8 bf16 elements
constexpr float kFP8Margin = 1e-4f;

// Expert-centric constants (aligned with DeepEP)
constexpr int kNumWarpGroups = 2;       // Each SM handles 2 experts (for count/signal phase)
constexpr int kNumWarpsPerGroup = 8;    // 8 warps per warp group
constexpr int kNumWarpsPerBlock = kNumWarpGroups * kNumWarpsPerGroup;  // 16 warps

// Finish counter pattern: when counter reaches this value, all tokens for that expert are dispatched
// The pattern works as: finish_counter = dispatched_count + (FINISHED_SUM_TAG - expected_count)
// When dispatched_count == expected_count, finish_counter == FINISHED_SUM_TAG
// Note: kFinishedSumTag is defined in dispatch_combine_deepep.hpp
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

// Warp-level sum reduction
__device__ inline int WarpReduceSum(int value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

// CastFp8ToFloat unused in current LL combine path (caller dequantizes fp8 outputs).
}  // namespace detail

// DeepEP-aligned dispatch kernel with expert-centric count/signal phases
// Key design:
// 1. ALL SMs process ALL tokens (SM-based token striding)
// 2. ALL warps with valid top-k destinations do the data copy (no filtering by responsible expert)
// 3. Expert-centric count phase: each SM counts tokens for its 2 responsible experts
// 4. Expert-centric signal phase: each SM's warp groups wait for their experts' finish counters
// 5. LOCAL atomics for slot assignment (~10 cycles vs ~50+ for remote xGMI atomics)
// 6. Source-rank-partitioned buffer layout: each rank writes to its own slot range
// 7. Per-expert finish counters for fine-grained completion detection
template <typename T, bool kUseFP8>
__global__ void EpDispatchIntraNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int smId = static_cast<int>(blockIdx.x);
  const int numSms = static_cast<int>(gridDim.x);
  const int globalWarpId = smId * warpNum + warpId;
  const int globalWarpNum = numSms * warpNum;
  const int myPe = config.rank;
  const int npes = config.worldSize;
  const index_t expertCapacity = config.worldSize * config.maxNumInpTokenPerRank;
  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];

  // Expert-centric warp group assignment (for count/signal phases)
  constexpr int kNumWarpGroups = detail::kNumWarpGroups;
  constexpr int kNumWarpsPerGroup = detail::kNumWarpsPerGroup;
  const int warpGroupId = warpId / kNumWarpsPerGroup;
  const int subWarpId = warpId % kNumWarpsPerGroup;
  const int numLocalExperts = config.numExpertPerRank;
  const int numExperts = npes * numLocalExperts;

  // Each SM is responsible for 2 experts (for count/signal phases)
  const int responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;
  const bool hasResponsibleExpert = (responsibleExpertIdx < numExperts);

  // Shared memory for coordination (DeepEP-style)
  __shared__ index_t sharedNumTokensSentPerExpert[kNumWarpGroups];
  __shared__ index_t sharedDestTokId[16];      // Max 16 warps per block
  __shared__ index_t sharedDestLinearTok[16];

  // Synchronize all ranks before accessing remote symmetric memory.
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);

  // ==================== PHASE 1: Token Dispatch ====================
  // ALL SMs process ALL tokens. Each warp with valid destExpert does the copy.
  if (args.tokenIndices && args.inpTokenBuf) {
    // SM-based token striding: each SM handles tokens[smId], tokens[smId + numSms], ...
    for (int tokenIdx = smId; tokenIdx < args.curRankNumToken; tokenIdx += numSms) {
      // Each warp reads its own top-k index (warpId < numTopk)
      index_t destExpert = -1;
      index_t destPe = -1;
      index_t localExpert = -1;
      index_t destTokId = 0;
      index_t destLinearTok = 0;

      if (warpId < config.numExpertPerToken) {
        destExpert = args.tokenIndices[tokenIdx * config.numExpertPerToken + warpId];
        if (destExpert >= 0) {
          destPe = destExpert / numLocalExperts;
          localExpert = destExpert % numLocalExperts;

          // Allocate slot using LOCAL atomic
          if (laneId == 0) {
            destTokId = atomicAdd(args.atomicCounterPerExpert + destExpert, 1);
            atomicAdd(args.destPeTokenCounter + destPe, 1);

            // Source-rank-partitioned buffer layout
            index_t srcRankOffset = myPe * config.maxNumInpTokenPerRank;
            sharedDestTokId[warpId] = destTokId;
            sharedDestLinearTok[warpId] = localExpert * expertCapacity + srcRankOffset + destTokId;

            // Store mapping for combine phase
            args.dispDestTokIdMap[tokenIdx * config.numExpertPerToken + warpId] =
                destExpert * expertCapacity + srcRankOffset + destTokId;
          }
        }
      }
      __syncthreads();

      // Read back from shared memory
      if (warpId < config.numExpertPerToken && destExpert >= 0) {
        destTokId = sharedDestTokId[warpId];
        destLinearTok = sharedDestLinearTok[warpId];
      }

      // FP8 quantization path
      if constexpr (kUseFP8) {
        constexpr int kNumElemsPerRead = sizeof(uint4) / sizeof(T);
        const int numScales = config.hiddenDim / detail::kNumPerChannels;
        const auto* srcPtr = reinterpret_cast<const uint4*>(args.inpTokenBuf + tokenIdx * config.hiddenDim);

        if (warpId < config.numExpertPerToken && destExpert >= 0) {
          auto* destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
              args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe)) +
              destLinearTok * config.hiddenDim;
          auto* destPtr = reinterpret_cast<uint64_t*>(destFp8);
          float* scalePtr = args.shmemOutScalesMemObj->template GetAs<float*>(destPe) +
                            destLinearTok * numScales;

          // Store source token mapping
          detail::AtomicStoreRelease(
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destLinearTok,
              static_cast<index_t>(myPe * config.maxNumInpTokenPerRank + tokenIdx));

          // Copy weights and indices
          if (laneId < config.numExpertPerToken) {
            if (args.weightsBuf) {
              args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                  destPe)[destLinearTok * config.numExpertPerToken + laneId] =
                  args.weightsBuf[tokenIdx * config.numExpertPerToken + laneId];
            }
            args.shmemOutIndicesMemObj->template GetAs<index_t*>(
                destPe)[destLinearTok * config.numExpertPerToken + laneId] =
                args.tokenIndices[tokenIdx * config.numExpertPerToken + laneId];
          }

          // Process channels with vectorized FP8 quantization
          constexpr int kChannelsPerIter = warpSize * detail::kElemsPerLoad / detail::kNumPerChannels;

          for (int scaleBase = 0; scaleBase < numScales; scaleBase += kChannelsPerIter) {
            uint4 packed = srcPtr[scaleBase * detail::kNumPerChannels / detail::kElemsPerLoad + laneId];
            auto* bf16Values = reinterpret_cast<const T*>(&packed);

            float fp32Values[detail::kElemsPerLoad];
            float amax = detail::kFP8Margin;
            #pragma unroll
            for (int j = 0; j < detail::kElemsPerLoad; ++j) {
              fp32Values[j] = static_cast<float>(bf16Values[j]);
              amax = fmaxf(amax, fabsf(fp32Values[j]));
            }

            amax = detail::QuarterWarpReduceMax(amax);
            float scale = detail::kFP8Amax / amax;
            float scaleInv = amax * detail::kFP8AmaxInv;

            if ((laneId & 15) == 0) {
              scalePtr[scaleBase + (laneId >> 4)] = scaleInv;
            }

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
              fp8x2Values[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#endif
            }

            destPtr[scaleBase * detail::kNumPerChannels / detail::kElemsPerLoad + laneId] = fp8Packed;
          }

          // Increment per-expert finish counter after copy (release semantics)
          if (laneId == 0) {
            detail::AtomicAddRelease(args.finishCounterPerExpert + destExpert, 1u);
          }
        }
      } else {
        // Non-FP8 path
        if (warpId < config.numExpertPerToken && destExpert >= 0) {
          // Store source token mapping
          if (laneId == 0) {
            detail::AtomicStoreRelease(
                args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destLinearTok,
                static_cast<index_t>(myPe * config.maxNumInpTokenPerRank + tokenIdx));
          }

          // Copy weights and indices
          if (laneId < config.numExpertPerToken) {
            if (args.weightsBuf) {
              args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                  destPe)[destLinearTok * config.numExpertPerToken + laneId] =
                  args.weightsBuf[tokenIdx * config.numExpertPerToken + laneId];
            }
            args.shmemOutIndicesMemObj->template GetAs<index_t*>(
                destPe)[destLinearTok * config.numExpertPerToken + laneId] =
                args.tokenIndices[tokenIdx * config.numExpertPerToken + laneId];
          }

          // Copy pre-computed scales
          if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
            index_t destScaleOffset = destLinearTok * config.scaleDim * config.scaleTypeSize;
            index_t srcScaleOffset = tokenIdx * config.scaleDim * config.scaleTypeSize;
            core::WarpCopy(
                args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
                args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
          }

          // Copy token data
          size_t baseOffset = destLinearTok * config.hiddenDim;
          core::WarpCopy(args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + baseOffset,
                         args.inpTokenBuf + tokenIdx * config.hiddenDim, config.hiddenDim);

          // Increment per-expert finish counter after copy (release semantics)
          if (laneId == 0) {
            detail::AtomicAddRelease(args.finishCounterPerExpert + destExpert, 1u);
          }
        }
      }
      __syncthreads();  // Sync before next token iteration
    }
  }

  // ==================== PHASE 2: Expert-centric Count Phase ====================
  // Last warp per SM counts tokens for this SM's 2 responsible experts
  constexpr int kLastWarpId = kNumWarpGroups * kNumWarpsPerGroup - 1;
  if (warpId == kLastWarpId) {
    const int expertBeginIdx = smId * kNumWarpGroups;
    const int expertEndIdx = min(expertBeginIdx + kNumWarpGroups, numExperts);

    // Per-lane count for each responsible expert
    int expertCount[kNumWarpGroups] = {0};

    #pragma unroll 2
    for (int i = laneId; i < args.curRankNumToken * config.numExpertPerToken; i += warpSize) {
      index_t idx = args.tokenIndices ? args.tokenIndices[i] : -1;
      if (idx >= expertBeginIdx && idx < expertEndIdx) {
        expertCount[idx - expertBeginIdx]++;
      }
    }

    // Warp-reduce and update finish counter
    #pragma unroll 2
    for (int i = expertBeginIdx; i < expertEndIdx; ++i) {
      int sum = detail::WarpReduceSum(expertCount[i - expertBeginIdx]);
      if (laneId == 0) {
        sharedNumTokensSentPerExpert[i - expertBeginIdx] = sum;
        // Add offset to reach FINISHED_SUM_TAG when all dispatches complete
        detail::AtomicAddRelaxed(args.finishCounterPerExpert + i,
                                 detail::kFinishedSumTag - static_cast<uint32_t>(sum));
      }
    }
  }
  __syncthreads();

  // ==================== PHASE 3: Expert-centric Wait Phase ====================
  // Each warp group's first warp waits for its responsible expert's finish counter
  // and updates the remote expert token counter
  if (hasResponsibleExpert && subWarpId == 0 && laneId == 0) {
    const int destPe = responsibleExpertIdx / numLocalExperts;
    const int localExpertIdx = responsibleExpertIdx % numLocalExperts;
    const int numTokensSent = sharedNumTokensSentPerExpert[warpGroupId];

    // Wait for per-expert finish counter to reach FINISHED_SUM_TAG
    while (detail::AtomicLoadAcquire(args.finishCounterPerExpert + responsibleExpertIdx)
           != detail::kFinishedSumTag) {
    }

    // Update remote expert token counter (batched: one update per expert)
    atomicAdd(args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe) + localExpertIdx,
              numTokensSent);

    // Clean workspace for next use
    args.atomicCounterPerExpert[responsibleExpertIdx] = 0;
    args.finishCounterPerExpert[responsibleExpertIdx] = 0;
  }

  // Grid barrier to ensure all expert counters are updated before signal phase
  detail::GridBarrier(args.dispatchGridBarrier, numSms);

  // ==================== PHASE 4: Signal Phase (per-destPe) ====================
  // Warp 0 sends signals to all destination PEs with total token count
  if (warpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Sum up tokens sent to this destPe from destPeTokenCounter
      index_t numTokensSent = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe);
      index_t numTokenSignal = numTokensSent + 1;

      // Send signal to destination PE
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      detail::AtomicStoreRelease(signal, numTokenSignal);
    }
  }

  // ==================== PHASE 5: Receive Phase ====================
  // Wait for signals from all source ranks
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (warpId == 0) {
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      index_t* signal = recvTokenNums + srcPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      detail::AtomicStoreRelease(signal, static_cast<index_t>(0));  // Reset for next use
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
    }
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        args.recvTokenCountPerExpert[e] = localExpertCounter[e];
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
        srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) + baseOffset;
        if constexpr (kUseWeights) {
          srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
              (localExpert * expertCapacity + destLocalTokId) * config.numExpertPerToken;
          srcWeightScales[j] =
              srcWeightsPtr[j][j];  // weight at column j for this expert
        }
      } else {
        srcPtrs[j] = nullptr;
        if constexpr (kUseWeights) {
          srcWeightsPtr[j] = nullptr;
          srcWeightScales[j] = 0.0f;
        }
      }
    }
    __syncwarp();

    // Step 5: weighted accumulation of all top-k contributions.
    if constexpr (kUseWeights) {
      core::WarpAccum(args.outTokenBuf + tokenId * config.hiddenDim + hiddenDimOffset, srcPtrs,
                      srcWeightScales, config.numExpertPerToken, hiddenDimSize,
                      hiddenDimOffset);
    } else {
      core::WarpAccumNoWeights(args.outTokenBuf + tokenId * config.hiddenDim + hiddenDimOffset,
                                srcPtrs, nullptr, config.numExpertPerToken,
                                hiddenDimSize, hiddenDimOffset);
    }
  }

  // Copy back new weights / indices if caller provided combined-output buffers.
  if (args.outWeights) {
    for (int i = globalWarpId; i < args.curRankNumToken; i += globalWarpNum) {
      index_t destTokId = args.dispDestTokIdMap[i * config.numExpertPerToken];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      size_t baseWeightOff = (localExpert * expertCapacity + destLocalTokId) * config.numExpertPerToken;
      core::WarpCopy(args.outWeights + i * config.numExpertPerToken,
                     args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) + baseWeightOff,
                     config.numExpertPerToken);
    }
  }

  if (args.outIndices) {
    for (int i = globalWarpId; i < args.curRankNumToken; i += globalWarpNum) {
      index_t destTokId = args.dispDestTokIdMap[i * config.numExpertPerToken];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      size_t baseIdxOff = (localExpert * expertCapacity + destLocalTokId) * config.numExpertPerToken;
      core::WarpCopy(args.outIndices + i * config.numExpertPerToken,
                     args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe) + baseIdxOff,
                     config.numExpertPerToken);
    }
  }

  // Copy back scales if fp8
  if (args.outScales) {
    const int numScales = config.hiddenDim / detail::kNumPerChannels;
    for (int i = globalWarpId; i < args.curRankNumToken; i += globalWarpNum) {
      index_t destTokId = args.dispDestTokIdMap[i * config.numExpertPerToken];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      size_t baseScaleOff = (localExpert * expertCapacity + destLocalTokId) * numScales;
      core::WarpCopy(args.outScales + i * numScales,
                     args.shmemOutScalesMemObj->template GetAs<float*>(destPe) + baseScaleOff,
                     numScales);
    }
  }

  // Copy out token data from fp8 symmetric buffer
  if constexpr (kUseFP8) {
    const int numScales = config.hiddenDim / detail::kNumPerChannels;
    using FP8 = __hip_fp8_storage_t;
    for (int i = globalWarpId; i < args.curRankNumToken; i += globalWarpNum) {
      index_t destTokId = args.dispDestTokIdMap[i * config.numExpertPerToken];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      size_t baseTokOff = (localExpert * expertCapacity + destLocalTokId) * config.hiddenDim;
      core::WarpCopy(args.fp8OutTokenBuf + i * config.hiddenDim,
                     args.shmemDispatchOutTokMemObj->template GetAs<FP8*>(destPe) + baseTokOff,
                     config.hiddenDim);
    }
  } else {
    // Standard path: copy non-fp8 to outTokenBuf via accumulation (already done above)
    // Additional path for callers that just want raw dispatch output
    if (args.fp8OutTokenBuf) {
      for (int i = globalWarpId; i < args.curRankNumToken; i += globalWarpNum) {
        core::WarpCopyAccumTopK(
                                args.outTokenBuf + i * config.hiddenDim, srcPtrs,
                                srcWeightsPtr, nullptr, config.numExpertPerToken,
                                config.numExpertPerToken);
      }
    }
  }
}

}  // namespace deepep
}  // namespace moe
}  // namespace mori
