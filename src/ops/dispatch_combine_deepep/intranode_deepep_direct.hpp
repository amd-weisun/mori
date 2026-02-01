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
    // Use device-scope release for cross-GPU visibility (xGMI is cache-coherent)
    detail::AtomicStoreRelease(
        args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    // Use device-scope acquire for cross-GPU visibility (xGMI is cache-coherent)
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

// DeepEP-aligned dispatch kernel with SM-based token striding (Phase 6)
// Key differences from previous MORI implementation:
// 1. SM-based token striding: Each block handles tokens[blockIdx.x], tokens[blockIdx.x + gridDim.x], ...
// 2. Warp-based top-k handling: Each warp handles one top-k destination per token
// 3. All-thread FP8 quantization: All threads in block work on one token's FP8 quantization
// 4. Per-destPe finish counters: Aggregates all experts on each rank (simpler for intranode)
//
// Note on Phase 5 (Packed Buffers): Packed message format is primarily beneficial for internode
// RDMA where bundling reduces per-message overhead. For intranode xGMI (cache-coherent),
// separate buffers are kept as they provide simpler indexing without packing/unpacking overhead.
template <typename T, bool kUseFP8>
__global__ void EpDispatchIntraNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int numThreads = blockDim.x;
  const int smId = static_cast<int>(blockIdx.x);
  const int numSms = static_cast<int>(gridDim.x);
  const int globalWarpId = smId * warpNum + warpId;
  const int globalWarpNum = numSms * warpNum;
  const int myPe = config.rank;
  const int npes = config.worldSize;
  const index_t expertCapacity = config.worldSize * config.maxNumInpTokenPerRank;
  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];

  // Synchronize all ranks before accessing remote symmetric memory.
  // Note: Counter buffers are already reset by hipMemsetAsync before kernel launch.
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);

  // Shared memory for per-warp destination info (DeepEP-style)
  // Each warp needs: destExpert, destPe, localExpert, destTokId, destLinearTok
  __shared__ index_t sharedDestTokId[16];      // Max 16 warps per block
  __shared__ index_t sharedDestLinearTok[16];

  if (args.tokenIndices && args.inpTokenBuf) {
    // DeepEP-style: SM-based token striding
    // Each block handles tokens[smId], tokens[smId + numSms], ...
    for (int tokenIdx = smId; tokenIdx < args.curRankNumToken; tokenIdx += numSms) {
      // Each warp reads its own top-k index (similar to DeepEP)
      // Warps with warpId >= numExpertPerToken will have invalid destExpert (-1)
      index_t destExpert = -1;
      index_t destPe = -1;
      index_t localExpert = -1;
      index_t destTokId = 0;
      index_t destLinearTok = 0;

      if (warpId < config.numExpertPerToken) {
        destExpert = args.tokenIndices[tokenIdx * config.numExpertPerToken + warpId];
        destPe = destExpert / config.numExpertPerRank;
        localExpert = destExpert % config.numExpertPerRank;

        // Allocate slot and update counters (lane 0 only)
        if (laneId == 0) {
          index_t* expertCounter =
              args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe);
          destTokId = atomicAdd(expertCounter + localExpert, 1);
          atomicAdd(args.destPeTokenCounter + destPe, 1);
          args.dispDestTokIdMap[tokenIdx * config.numExpertPerToken + warpId] =
              destExpert * expertCapacity + destTokId;

          // Store to shared memory for other threads to access
          sharedDestTokId[warpId] = destTokId;
          sharedDestLinearTok[warpId] = localExpert * expertCapacity + destTokId;
        }
      }
      __syncthreads();

      // Read back from shared memory
      if (warpId < config.numExpertPerToken) {
        destTokId = sharedDestTokId[warpId];
        destLinearTok = sharedDestLinearTok[warpId];
      }

      // FP8 quantization: All threads in block work on the token's data
      // This is the key DeepEP optimization - better parallelism for FP8 cast
      if constexpr (kUseFP8) {
        // All threads participate in FP8 quantization for this token
        // The quantized data is written to a temporary location, then copied by each warp
        constexpr int kNumElemsPerRead = sizeof(uint4) / sizeof(T);  // 8 bf16 elements
        const int hiddenBf16Int4 = config.hiddenDim / kNumElemsPerRead;
        const int numScales = config.hiddenDim / detail::kNumPerChannels;
        const auto* srcInt4 = reinterpret_cast<const uint4*>(args.inpTokenBuf + tokenIdx * config.hiddenDim);

        // Each warp that has a valid destination does its own FP8 quantization and copy
        // This maintains the per-warp independence while using the same loop structure as DeepEP
        if (warpId < config.numExpertPerToken && destPe >= 0) {
          auto* destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
              args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe)) +
              destLinearTok * config.hiddenDim;
          auto* srcPtr = reinterpret_cast<const uint4*>(args.inpTokenBuf + tokenIdx * config.hiddenDim);
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

          // Increment finish counter after copy (release semantics)
          if (laneId == 0) {
            detail::AtomicAddRelease(args.finishCounterPerDestPe + destPe, 1u);
          }
        }
      } else {
        // Non-FP8 path: Each warp handles its destination independently
        if (warpId < config.numExpertPerToken && destPe >= 0) {
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

          // Increment finish counter after copy (release semantics)
          if (laneId == 0) {
            detail::AtomicAddRelease(args.finishCounterPerDestPe + destPe, 1u);
          }
        }
      }
      __syncthreads();  // Sync before next token iteration
    }
  }

  // Count warp: Computes expected tokens per destPe and adds offset to finish counters
  // Uses the last global warp (DeepEP uses last warp per SM, but for intranode one warp suffices)
  constexpr int kMaxGpuPerNode = 8;
  assert(npes <= kMaxGpuPerNode && "Intranode dispatch assumes worldSize <= 8 GPUs per node");
  if (globalWarpId == globalWarpNum - 1) {
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
    for (int pe = 0; pe < npes && pe < kMaxGpuPerNode; ++pe) {
      int count = countPerDestPe[pe];
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        count += __shfl_xor(count, offset);
      }
      if (laneId == 0) {
        uint32_t finishOffset = detail::kFinishedSumTag - static_cast<uint32_t>(count);
        detail::AtomicAddRelease(args.finishCounterPerDestPe + pe, finishOffset);
      }
    }
  }

  // Signal phase: Wait for finish counter per destPe, then send token counts
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      while (detail::AtomicLoadAcquire(args.finishCounterPerDestPe + destPe) != detail::kFinishedSumTag) {
      }

      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      detail::AtomicStoreRelease(signal, numTokenSignal);
    }
  }

  // Receive phase: Wait for signals from other ranks
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      detail::AtomicStoreRelease(signal, static_cast<index_t>(0));
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
