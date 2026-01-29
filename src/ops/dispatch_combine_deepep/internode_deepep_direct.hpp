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

namespace internode_ll {

inline __host__ __device__ bool SymmMemIsValid(const mori::application::SymmMemObjPtr& mem) {
  return (mem.cpu != nullptr) && (mem.gpu != nullptr);
}

inline __device__ bool IsRemoteRank(int myPe, int destPe, int gpuPerNode) {
  return (myPe / gpuPerNode) != (destPe / gpuPerNode);
}

template <typename T>
inline __device__ void RdmaWriteScalar(const mori::application::SymmMemObjPtr& dest, size_t byteOffset,
                                       T value, int pe) {
  shmem::ShmemPutTypeImmNbiThread<T>(dest, byteOffset, value, pe);
  shmem::ShmemQuietThread(pe);
}

inline __device__ void RdmaWriteBlock(const mori::application::SymmMemObjPtr& dest,
                                      size_t destOffsetBytes,
                                      const mori::application::SymmMemObjPtr& source,
                                      size_t srcOffsetBytes,
                                      size_t bytes,
                                      int pe) {
  if (bytes == 0) return;
  shmem::ShmemPutMemNbiThread(dest, destOffsetBytes, source, srcOffsetBytes, bytes, pe);
  shmem::ShmemQuietThread(pe);
}

inline __device__ int64_t GlobalWarpId() {
  const int warpsPerBlock = blockDim.x / warpSize;
  return static_cast<int64_t>(blockIdx.x) * warpsPerBlock + (threadIdx.x / warpSize);
}

inline __device__ bool IsControlWarp() {
  return GlobalWarpId() == 0;
}

inline __device__ int LaneId() {
  return threadIdx.x & (warpSize - 1);
}

}  // namespace internode_ll

namespace internode_detail {

constexpr int kNumPerChannels = 128;

__device__ inline float DeepepFp8Scale(float amax) {
  constexpr float kFP8Margin = 1e-4f;
#ifdef __HIP_PLATFORM_AMD__
  constexpr float kFP8Amax = 240.0f;
#else
  constexpr float kFP8Amax = 448.0f;
#endif
  float safeAmax = fmaxf(amax, kFP8Margin);
  return kFP8Amax / safeAmax;
}

__device__ inline float DeepepFp8ScaleInv(float amax) {
  constexpr float kFP8Margin = 1e-4f;
#ifdef __HIP_PLATFORM_AMD__
  constexpr float kFP8AmaxInv = 1.0f / 240.0f;
#else
  constexpr float kFP8AmaxInv = 1.0f / 448.0f;
#endif
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

}  // namespace internode_detail

// Dispatch kernel implementation - parallel warp-based approach following V1 pattern
template <typename T, bool kUseFP8>
__global__ void EpDispatchInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  
  int myPe = config.rank;
  int npes = config.worldSize;
  int myNode = myPe / config.gpuPerNode;
  int nNodes = npes / config.gpuPerNode;
  int numExpertPerToken = config.numExpertPerToken;
  const index_t expertCapacity = static_cast<index_t>(config.worldSize) * config.maxNumInpTokenPerRank;
  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];

  // Reset local counters - only control warp does this
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

  // Barrier to sync all ranks before dispatch
  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);
  if (globalWarpId == 0) {
    if (laneId < args.config.worldSize) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      args.combineGridBarrier[0] = 0;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(laneId) + args.config.rank,
          crossDeviceBarrierFlag);
    }
  }
  
  if (globalWarpId == 0 && laneId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();

  if (args.curRankNumToken == 0) {
    return;
  }

  // Phase 1: Intra-node dispatch (XGMI)
  for (int i = globalWarpId; i < args.curRankNumToken * numExpertPerToken; i += globalWarpNum) {
    index_t srcTokId = i / numExpertPerToken;
    index_t expertIdx = i % numExpertPerToken;
    index_t destExpert = args.tokenIndices[i];
    index_t destPe = destExpert / config.numExpertPerRank;
    index_t localExpert = destExpert % config.numExpertPerRank;
    int destNode = destPe / config.gpuPerNode;

    // Check for duplicates within this token's experts
    bool isDup = false;
    if (laneId < numExpertPerToken) {
      index_t laneExpert = args.tokenIndices[srcTokId * numExpertPerToken + laneId];
      index_t lanePe = laneExpert / config.numExpertPerRank;
      int laneNode = lanePe / config.gpuPerNode;
      if (destNode == myNode && laneNode == myNode && laneId < expertIdx && lanePe == destPe) {
        isDup = true;
      }
    }
    isDup = __any_sync(0xffffffffffffffffull, isDup);

    if (isDup || destNode != myNode) {
      continue;  // Skip duplicates and inter-node for this phase
    }

    // Allocate slot
    index_t destTokId = 0;
    if (laneId == 0) {
      destTokId = atomicAdd(
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe) + localExpert, 1);
      atomicAdd(args.destPeTokenCounter + destPe, 1);
      args.dispDestTokIdMap[i] = destExpert * expertCapacity + destTokId;
    }
    destTokId = __shfl(destTokId, 0);
    index_t destLinearTok = localExpert * expertCapacity + destTokId;

    // Write src token ID mapping
    if (laneId == 0) {
      core::AtomicStoreRelaxedSystem(
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destLinearTok,
          static_cast<index_t>(myPe * config.maxNumInpTokenPerRank + srcTokId));
    }

    // Write weights and indices
    if (laneId < numExpertPerToken) {
      if (args.weightsBuf) {
        args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe)
            [destLinearTok * numExpertPerToken + laneId] =
            args.weightsBuf[srcTokId * numExpertPerToken + laneId];
      }
      args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe)
          [destLinearTok * numExpertPerToken + laneId] =
          args.tokenIndices[srcTokId * numExpertPerToken + laneId];
    }

    size_t baseOffset = destLinearTok * config.hiddenDim;
    index_t srcTokOffset = srcTokId * config.hiddenDim;

    // Write hidden states (FP8 or BF16)
    if constexpr (kUseFP8) {
      auto* destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
          args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe)) + baseOffset;
      int numScales = config.hiddenDim / internode_detail::kNumPerChannels;
      
      for (int scaleIdx = 0; scaleIdx < numScales; ++scaleIdx) {
        int channelBase = scaleIdx * internode_detail::kNumPerChannels;
        float amax = 0.0f;
        for (int j = laneId; j < internode_detail::kNumPerChannels; j += warpSize) {
          int offset = channelBase + j;
          float v = static_cast<float>(args.inpTokenBuf[srcTokOffset + offset]);
          amax = fmaxf(amax, fabsf(v));
        }
        for (int mask = warpSize / 2; mask > 0; mask >>= 1) 
          amax = fmaxf(amax, __shfl_xor(amax, mask));
        float scale = internode_detail::DeepepFp8Scale(amax);
        float scaleInv = internode_detail::DeepepFp8ScaleInv(amax);
        if (laneId == 0) {
          args.shmemOutScalesMemObj->template GetAs<float*>(destPe)
              [destLinearTok * numScales + scaleIdx] = scaleInv;
        }
        for (int j = laneId; j < internode_detail::kNumPerChannels; j += warpSize) {
          int offset = channelBase + j;
          float v = static_cast<float>(args.inpTokenBuf[srcTokOffset + offset]);
          destFp8[offset] = internode_detail::CastFloatToFp8(v, scale);
        }
      }
    } else {
      core::WarpCopy(args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + baseOffset,
                     args.inpTokenBuf + srcTokOffset, config.hiddenDim);
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        index_t destScaleOffset = destLinearTok * config.scaleDim * config.scaleTypeSize;
        index_t srcScaleOffset = srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
            args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
      }
    }
  }

  __threadfence_system();
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Phase 2: Send signals to intra-node peers
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      int destNode = destPe / config.gpuPerNode;
      if (destNode != myNode) continue;  // Only intra-node in this phase
      
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      args.dispatchGridBarrier[0] = 0;

      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 3: Wait for intra-node signals and accumulate
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      int srcNode = srcPe / config.gpuPerNode;
      if (srcNode != myNode) continue;  // Only intra-node
      
      index_t* signal = recvTokenNums + srcPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[srcPe] = 0;
    }
    
    if (laneId == 0) {
      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        args.recvTokenCountPerExpert[e] = localExpertCounter[e];
        localExpertCounter[e] = 0;
      }
    }
  }
}

// Combine kernel implementation - follows intranode pattern but with inter-node awareness
template <typename T, bool kUseFP8, bool kUseWeights>
__global__ void EpCombineInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
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

  // Step 1: Copy valid expert slots into symmetric combine buffer
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

  // Step 2: Cross-rank barrier
  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);
  if (globalWarpId == 0) {
    if (laneId < args.config.worldSize) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      args.combineGridBarrier[0] = 0;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(laneId) + args.config.rank,
          crossDeviceBarrierFlag);
    }
  }
  
  if (globalWarpId == 0 && laneId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
  
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
    index_t hiddenDimSize = max(0, min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      if (destPe < config.worldSize) {
        size_t baseOffset = (localExpert * expertCapacity + destLocalTokId) * config.hiddenDim;
        srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                     baseOffset + hiddenDimOffset;
        srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                           (localExpert * expertCapacity + destLocalTokId) * config.numExpertPerToken;
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
