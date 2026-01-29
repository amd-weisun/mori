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

inline __device__ bool IsRemoteRank(int myPe, int destPe, int gpuPerNode) {
  return (myPe / gpuPerNode) != (destPe / gpuPerNode);
}

template <typename T>
inline __device__ T* GetSymmPtr(const mori::application::SymmMemObjPtr& mem, int pe) {
  return mem.IsValid() ? mem->template GetAs<T*>(pe) : nullptr;
}

template <typename T>
inline __device__ const T* GetSymmPtrConst(const mori::application::SymmMemObjPtr& mem, int pe) {
  return mem.IsValid() ? mem->template GetAs<T*>(pe) : nullptr;
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

namespace detail {

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

}  // namespace detail

template <typename T>
__device__ void ResetDispatchCounters(EpDispatchCombineArgs<T>& args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  int laneId = internode_ll::LaneId();
  int myPe = config.rank;

  if (laneId == 0 && args.totalRecvTokenNum) {
    *args.totalRecvTokenNum = 0;
  }

  __syncwarp();

  if (laneId == 0) {
    if (args.destExpertTokenCounterMemObj.IsValid()) {
      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(myPe);
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        localExpertCounter[e] = 0;
      }
    }
    if (args.destPeTokenCounter) {
      for (int pe = 0; pe < config.worldSize; ++pe) {
        args.destPeTokenCounter[pe] = 0;
      }
    }
    if (args.destNodeTokenCounter) {
      for (int node = 0; node < config.worldSize / config.gpuPerNode; ++node) {
        args.destNodeTokenCounter[node] = 0;
      }
    }
    if (args.localPeTokenCounter) {
      size_t total = static_cast<size_t>(config.worldSize) * config.numExpertPerRank;
      for (size_t i = 0; i < total; ++i) {
        args.localPeTokenCounter[i] = 0;
      }
    }
    if (args.recvTokenCountPerExpert) {
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        args.recvTokenCountPerExpert[e] = 0;
      }
    }
  }

  __syncwarp();

  if (laneId == 0 && args.dispTokIdToSrcTokIdMemObj.IsValid()) {
    index_t* localMap = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe);
    size_t totalTokenSlots =
        static_cast<size_t>(config.numExpertPerRank) * config.worldSize * config.maxNumInpTokenPerRank;
    for (size_t i = 0; i < totalTokenSlots; ++i) {
      localMap[i] = static_cast<index_t>(-1);
    }
  }
}

template <typename T, bool kUseFP8>
__device__ void CopyTokenToDest(EpDispatchCombineArgs<T>& args,
                                int destPe,
                                index_t destLinearTok,
                                const T* tokenPtr,
                                const float* fp8ScalePtr,
                                const uint8_t* genericScalePtr,
                                const float* weightPtr,
                                const index_t* indexPtr,
                                int laneId) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  bool isRemote = internode_ll::IsRemoteRank(config.rank, destPe, config.gpuPerNode);
  size_t hiddenBytes = static_cast<size_t>(config.hiddenDim) * sizeof(T);
  size_t indicesBytes =
      static_cast<size_t>(config.numExpertPerToken) * sizeof(index_t);
  size_t weightsBytes =
      static_cast<size_t>(config.numExpertPerToken) * sizeof(float);
  size_t scalesBytes =
      static_cast<size_t>(config.scaleDim) * config.scaleTypeSize;

  if (!isRemote) {
    T* destTok =
        args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) +
        destLinearTok * config.hiddenDim;
    for (int i = laneId; i < config.hiddenDim; i += warpSize) {
      destTok[i] = tokenPtr[i];
    }
    if (args.shmemOutIndicesMemObj.IsValid()) {
      index_t* destIdx =
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe) +
          destLinearTok * config.numExpertPerToken;
      if (indexPtr) {
        for (int i = laneId; i < config.numExpertPerToken; i += warpSize) {
          destIdx[i] = indexPtr[i];
        }
      }
    }
    if (args.shmemDispatchOutWeightsMemObj.IsValid() && weightPtr) {
      float* destWeight =
          args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe) +
          destLinearTok * config.numExpertPerToken;
      for (int i = laneId; i < config.numExpertPerToken; i += warpSize) {
        destWeight[i] = weightPtr[i];
      }
    }
    if constexpr (kUseFP8) {
      int numScales = config.hiddenDim / detail::kNumPerChannels;
      float* destScales =
          args.shmemOutScalesMemObj->template GetAs<float*>(destPe) +
          destLinearTok * numScales;
      if (fp8ScalePtr) {
        for (int i = laneId; i < numScales; i += warpSize) {
          destScales[i] = fp8ScalePtr[i];
        }
      }
    } else if (args.shmemOutScalesMemObj.IsValid() && scalesBytes > 0 && genericScalePtr) {
      uint8_t* destScales =
          args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) +
          destLinearTok * scalesBytes;
      for (size_t offset = laneId; offset < scalesBytes; offset += warpSize) {
        destScales[offset] = genericScalePtr[offset];
      }
    }
  } else {
    size_t destOffsetBytes = static_cast<size_t>(destLinearTok) * hiddenBytes;
    size_t srcOffsetBytes = destOffsetBytes;
    T* staging =
        args.shmemStagingTokMemObj->template GetAs<T*>() +
        destLinearTok * config.hiddenDim;
    for (int i = laneId; i < config.hiddenDim; i += warpSize) {
      staging[i] = tokenPtr[i];
    }
    __syncwarp();
    if (laneId == 0) {
      internode_ll::RdmaWriteBlock(args.shmemDispatchOutTokMemObj,
                                   destOffsetBytes,
                                   args.shmemStagingTokMemObj,
                                   srcOffsetBytes,
                                   hiddenBytes,
                                   destPe);
    }
    if (args.shmemOutIndicesMemObj.IsValid()) {
      size_t elemOffset =
          static_cast<size_t>(destLinearTok) * config.numExpertPerToken;
      index_t* stagingIdx =
          args.shmemInpIndicesMemObj->template GetAs<index_t*>() +
          elemOffset;
      if (indexPtr) {
        for (int i = laneId; i < config.numExpertPerToken; i += warpSize) {
          stagingIdx[i] = indexPtr[i];
        }
      }
      __syncwarp();
      if (laneId == 0) {
        internode_ll::RdmaWriteBlock(args.shmemOutIndicesMemObj,
                                     elemOffset * sizeof(index_t),
                                     args.shmemInpIndicesMemObj,
                                     elemOffset * sizeof(index_t),
                                     indicesBytes,
                                     destPe);
      }
    }
    if (args.shmemDispatchOutWeightsMemObj.IsValid() && weightPtr) {
      size_t elemOffset =
          static_cast<size_t>(destLinearTok) * config.numExpertPerToken;
      float* stagingWeight =
          args.shmemInpWeightsMemObj->template GetAs<float*>() +
          elemOffset;
      for (int i = laneId; i < config.numExpertPerToken; i += warpSize) {
        stagingWeight[i] = weightPtr[i];
      }
      __syncwarp();
      if (laneId == 0) {
        internode_ll::RdmaWriteBlock(args.shmemDispatchOutWeightsMemObj,
                                     elemOffset * sizeof(float),
                                     args.shmemInpWeightsMemObj,
                                     elemOffset * sizeof(float),
                                     weightsBytes,
                                     destPe);
      }
    }
    if constexpr (kUseFP8) {
      int numScales = config.hiddenDim / detail::kNumPerChannels;
      size_t elemOffset =
          static_cast<size_t>(destLinearTok) * numScales;
      float* stagingScale =
          args.shmemInpScalesMemObj->template GetAs<float*>() +
          elemOffset;
      if (fp8ScalePtr) {
        for (int i = laneId; i < numScales; i += warpSize) {
          stagingScale[i] = fp8ScalePtr[i];
        }
      }
      __syncwarp();
      if (laneId == 0) {
        internode_ll::RdmaWriteBlock(args.shmemOutScalesMemObj,
                                     elemOffset * sizeof(float),
                                     args.shmemInpScalesMemObj,
                                     elemOffset * sizeof(float),
                                     numScales * sizeof(float),
                                     destPe);
      }
    } else if (args.shmemOutScalesMemObj.IsValid() && scalesBytes > 0 && genericScalePtr) {
      size_t elemOffset =
          static_cast<size_t>(destLinearTok) * scalesBytes;
      uint8_t* stagingScale =
          args.shmemInpScalesMemObj->template GetAs<uint8_t*>() +
          elemOffset;
      for (size_t offset = laneId; offset < scalesBytes; offset += warpSize) {
        stagingScale[offset] = genericScalePtr[offset];
      }
      __syncwarp();
      if (laneId == 0) {
        internode_ll::RdmaWriteBlock(args.shmemOutScalesMemObj,
                                     elemOffset,
                                     args.shmemInpScalesMemObj,
                                     elemOffset,
                                     scalesBytes,
                                     destPe);
      }
    }
  }
}

template <typename T, bool kUseFP8>
__global__ void EpDispatchInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  if (!internode_ll::IsControlWarp()) {
    return;
  }

  const EpDispatchCombineDeepepConfig& config = args.config;
  const int laneId = internode_ll::LaneId();
  const int myPe = config.rank;
  const int gpuPerNode = config.gpuPerNode;
  const int worldSize = config.worldSize;
  const index_t expertCapacity =
      static_cast<index_t>(config.worldSize) * config.maxNumInpTokenPerRank;

  ResetDispatchCounters(args);

  if (args.curRankNumToken == 0) {
    return;
  }

  for (index_t tok = 0; tok < args.curRankNumToken; ++tok) {
    const index_t tokenOffset = tok * config.hiddenDim;
    const T* tokenPtr = args.inpTokenBuf ? (args.inpTokenBuf + tokenOffset) : nullptr;
    const float* fp8ScalePtr = nullptr;
    const uint8_t* genericScalePtr = nullptr;
    if constexpr (kUseFP8) {
      if (args.scalesBuf) {
        int numScales = config.hiddenDim / detail::kNumPerChannels;
        fp8ScalePtr =
            reinterpret_cast<const float*>(args.scalesBuf + tok * numScales * sizeof(float));
      }
    } else if (args.scalesBuf && config.scaleDim > 0 && config.scaleTypeSize > 0) {
      genericScalePtr =
          args.scalesBuf + tok * config.scaleDim * config.scaleTypeSize;
    }
    const float* weightPtr =
        args.weightsBuf ? (args.weightsBuf + tok * config.numExpertPerToken) : nullptr;
    const index_t* indexPtr =
        args.tokenIndices ? (args.tokenIndices + tok * config.numExpertPerToken) : nullptr;

    for (int expertIdx = 0; expertIdx < config.numExpertPerToken; ++expertIdx) {
      if (!args.tokenIndices) {
        continue;
      }
      const index_t destExpert = args.tokenIndices[tok * config.numExpertPerToken + expertIdx];
      const index_t destPe = destExpert / config.numExpertPerRank;
      const index_t localExpert = destExpert % config.numExpertPerRank;
      const bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      const bool isSameNode = (myPe / gpuPerNode) == (destPe / gpuPerNode);
      const index_t localCounterIdx = destPe * config.numExpertPerRank + localExpert;

      index_t destLocalSlot = args.localPeTokenCounter
                                  ? args.localPeTokenCounter[localCounterIdx]
                                  : 0;
      if (laneId == 0 && args.localPeTokenCounter) {
        args.localPeTokenCounter[localCounterIdx] = destLocalSlot + 1;
      }
      destLocalSlot = __shfl_sync(0xffffffff, destLocalSlot, 0);
      const index_t destTokId = myPe * config.maxNumInpTokenPerRank + destLocalSlot;
      const index_t destLinearTok = localExpert * expertCapacity + destTokId;

      if (laneId == 0 && args.destPeTokenCounter) {
        args.destPeTokenCounter[destPe] += 1;
      }
      if (!isRemote && args.destExpertTokenCounterMemObj.IsValid()) {
        index_t* localExpertCounter =
            args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe);
        if (laneId == 0) {
          localExpertCounter[localExpert] += 1;
        }
      }

      if (args.dispDestTokIdMap) {
        args.dispDestTokIdMap[tok * config.numExpertPerToken + expertIdx] =
            destExpert * expertCapacity + destTokId;
      }

      const index_t srcGlobalTok =
          myPe * config.maxNumInpTokenPerRank + tok;
      if (laneId == 0 && args.dispTokIdToSrcTokIdMemObj.IsValid()) {
        if (isRemote && !isSameNode) {
          internode_ll::RdmaWriteScalar(args.dispTokIdToSrcTokIdMemObj,
                                        destLinearTok * sizeof(index_t),
                                        srcGlobalTok,
                                        destPe);
        } else {
          index_t* destPtr =
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe);
          destPtr[destLinearTok] = srcGlobalTok;
        }
      }

      if (tokenPtr) {
        CopyTokenToDest<T, kUseFP8>(args,
                                    destPe,
                                    destLinearTok,
                                    tokenPtr,
                                    fp8ScalePtr,
                                    genericScalePtr,
                                    weightPtr,
                                    indexPtr,
                                    laneId);
      }
    }
  }

  __syncwarp();

  if (laneId == 0) {
    index_t* counterBase =
        args.srcExpertTokenCounterMemObj.IsValid()
            ? args.srcExpertTokenCounterMemObj->template GetAs<index_t*>()
            : nullptr;
    index_t* stagingRow = counterBase ? (counterBase + myPe * config.numExpertPerRank) : nullptr;
    for (int destPe = 0; destPe < worldSize; ++destPe) {
      const bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      const bool isSameNode = (myPe / gpuPerNode) == (destPe / gpuPerNode);
      index_t totalForDest = args.destPeTokenCounter ? args.destPeTokenCounter[destPe] : 0;

      if (stagingRow) {
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          const index_t idx = destPe * config.numExpertPerRank + e;
          stagingRow[e] = args.localPeTokenCounter ? args.localPeTokenCounter[idx] : 0;
        }
      }

      if (isRemote && stagingRow) {
        const size_t bytes = static_cast<size_t>(config.numExpertPerRank) * sizeof(index_t);
        internode_ll::RdmaWriteBlock(args.srcExpertTokenCounterMemObj,
                                     myPe * bytes,
                                     args.srcExpertTokenCounterMemObj,
                                     myPe * bytes,
                                     bytes,
                                     destPe);
      } else if (stagingRow && args.destExpertTokenCounterMemObj.IsValid()) {
        index_t* destCounter =
            args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe);
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          destCounter[e] = stagingRow[e];
        }
      }

      if (args.destPeTokenCounter) {
        args.destPeTokenCounter[destPe] = 0;
      }

      if (args.localPeTokenCounter) {
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          args.localPeTokenCounter[destPe * config.numExpertPerRank + e] = 0;
        }
      }

      if (args.recvTokenNumMemObj.IsValid()) {
        index_t signalValue = totalForDest + 1;
        if (isRemote && !isSameNode) {
          internode_ll::RdmaWriteScalar(args.recvTokenNumMemObj,
                                        myPe * sizeof(index_t),
                                        signalValue,
                                        destPe);
        } else {
          index_t* signal =
              args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
          *signal = signalValue;
        }
      }
    }
  }
}

template <typename T>
__device__ void AwaitIncomingSignals(EpDispatchCombineArgs<T>& args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  const int laneId = internode_ll::LaneId();
  const int worldSize = config.worldSize;

  if (!args.recvTokenNumMemObj.IsValid()) {
    if (laneId == 0 && args.totalRecvTokenNum) {
      *args.totalRecvTokenNum = 0;
    }
    __syncwarp();
    return;
  }

  if (laneId == 0) {
    index_t totalTokens = 0;
    index_t* signals = args.recvTokenNumMemObj->template GetAs<index_t*>();
    index_t* counterBase =
        args.srcExpertTokenCounterMemObj.IsValid()
            ? args.srcExpertTokenCounterMemObj->template GetAs<index_t*>()
            : nullptr;
    if (args.recvTokenCountPerExpert) {
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        args.recvTokenCountPerExpert[e] = 0;
      }
    }
    for (int srcPe = 0; srcPe < worldSize; ++srcPe) {
      index_t value = signals[srcPe];
      while (value == 0) {
        value = core::AtomicLoadRelaxedSystem(signals + srcPe);
      }
      signals[srcPe] = 0;
      totalTokens += (value - 1);
      if (args.recvTokenCountPerExpert && counterBase) {
        const index_t* srcCounters = counterBase + srcPe * config.numExpertPerRank;
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          args.recvTokenCountPerExpert[e] += srcCounters[e];
        }
      }
    }
    if (args.totalRecvTokenNum) {
      *args.totalRecvTokenNum = totalTokens;
    }
  }
  __syncwarp();
}

template <typename T, bool kUseFP8, bool kUseWeights>
__global__ void EpCombineInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  AwaitIncomingSignals(args);

  const EpDispatchCombineDeepepConfig& config = args.config;
  const int laneId = internode_ll::LaneId();
  const int warpId = threadIdx.x / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;
  const index_t expertCapacity =
      static_cast<index_t>(config.worldSize) * config.maxNumInpTokenPerRank;

  if (args.config.useExternalInpBuffer && args.inpTokenBuf && args.shmemCombineInpTokMemObj.IsValid()) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      index_t recvCount = args.recvTokenCountPerExpert ? args.recvTokenCountPerExpert[e] : 0;
      for (index_t slot = globalWarpId; slot < recvCount; slot += globalWarpNum) {
        index_t linear = e * expertCapacity + slot;
        core::WarpCopy(
            args.shmemCombineInpTokMemObj->template GetAs<T*>() + linear * config.hiddenDim,
            args.inpTokenBuf + linear * config.hiddenDim,
            config.hiddenDim);
      }
    }
  }

  if (args.weightsBuf && args.shmemInpWeightsMemObj.IsValid()) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      index_t recvCount = args.recvTokenCountPerExpert ? args.recvTokenCountPerExpert[e] : 0;
      for (index_t slot = globalWarpId; slot < recvCount; slot += globalWarpNum) {
        index_t linear = e * expertCapacity + slot;
        core::WarpCopy(
            args.shmemInpWeightsMemObj->template GetAs<float*>() + linear * config.numExpertPerToken,
            args.weightsBuf + linear * config.numExpertPerToken,
            config.numExpertPerToken);
      }
    }
  }

  __syncthreads();

  if (args.curRankNumToken == 0) {
    if (laneId == 0 && args.totalRecvTokenNum) {
      *args.totalRecvTokenNum = 0;
    }
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
  float* srcWeightScales = kUseWeights ? (srcWeightScalesBase + warpId * config.numExpertPerToken)
                                       : nullptr;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        max<index_t>(0, min(static_cast<index_t>(config.hiddenDim) - hiddenDimOffset,
                            static_cast<index_t>(hiddenDimPerWarp)));

    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId =
          args.dispDestTokIdMap ? args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j]
                                : static_cast<index_t>(-1);
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = (config.numExpertPerRank == 0)
                           ? 0
                           : destExpert / config.numExpertPerRank;
      index_t localExpert = (config.numExpertPerRank == 0)
                                ? 0
                                : destExpert % config.numExpertPerRank;

      if (destTokId == static_cast<index_t>(-1)) {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
        if constexpr (kUseWeights) srcWeightScales[j] = 0.0f;
        continue;
      }

      size_t baseOffset =
          (localExpert * expertCapacity + destLocalTokId) * config.hiddenDim + hiddenDimOffset;
      srcPtrs[j] =
          args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) + baseOffset;
      srcWeightsPtr[j] =
          args.shmemInpWeightsMemObj.IsValid()
              ? args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                    (localExpert * expertCapacity + destLocalTokId) * config.numExpertPerToken
              : nullptr;

      if constexpr (kUseWeights) {
        float w = 1.0f;
        if (srcWeightsPtr[j] != nullptr) {
          w = srcWeightsPtr[j][j];
        }
        srcWeightScales[j] = w;
      }
    }

    core::WarpAccum<T, 4>(
        args.shmemCombineOutTokMemObj->template GetAs<T*>() +
            tokenId * config.hiddenDim + hiddenDimOffset,
        srcPtrs,
        kUseWeights ? srcWeightScales : nullptr,
        config.numExpertPerToken,
        hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(
          args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
              tokenId * config.numExpertPerToken,
          srcWeightsPtr,
          nullptr,
          config.numExpertPerToken,
          config.numExpertPerToken);
    }
  }
}

}  // namespace deepep
}  // namespace moe
}  // namespace mori
