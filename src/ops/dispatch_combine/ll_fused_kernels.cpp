// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/core/core.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"
#include "src/ops/dispatch_combine/intranode.hpp"
#include "src/ops/dispatch_combine/internode_v1.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

namespace mori {
namespace moe {

using namespace mori::core;

// Lightweight AtomicAdd helpers for fused combine
template <typename T>
__device__ inline void AtomicAdd(T* address, T val) {
  atomicAdd(address, val);
}

template <>
__device__ inline void AtomicAdd<__hip_bfloat16>(__hip_bfloat16* address, __hip_bfloat16 val) {
#if defined(__HIP_PLATFORM_AMD__)
  unsafeAtomicAdd(address, val);
#else
  atomicAdd(address, val);
#endif
}

// Ensure hip_bfloat16 alias resolves to a specialization
template <>
__device__ inline void AtomicAdd<hip_bfloat16>(hip_bfloat16* address, hip_bfloat16 val) {
#if defined(__HIP_PLATFORM_AMD__)
  unsafeAtomicAdd(reinterpret_cast<__hip_bfloat16*>(address), static_cast<__hip_bfloat16>(val));
#else
  atomicAdd(reinterpret_cast<__hip_bfloat16*>(address), static_cast<__hip_bfloat16>(val));
#endif
}

// Zero expert counts and pair counter
template <typename T>
__device__ inline void ZeroFusedMetadata(EpDispatchCombineArgs<T>& args) {
  int laneId = threadIdx.x & (warpSize - 1);
  for (int e = laneId; e < args.config.numExpertPerRank; e += warpSize) {
    args.lowLatencyExpertCountMemObj->template GetAs<index_t*>()[e] = 0;
  }
  if (laneId == 0) {
    args.lowLatencyPairCountMemObj->template GetAs<index_t*>()[0] = 0;
  }
  int capacity = args.config.maxNumInpTokenPerRank;
  int total = args.config.numExpertPerRank * capacity;
  for (int i = laneId; i < total; i += warpSize) {
    args.lowLatencySortedTokenIdxMemObj->template GetAs<index_t*>()[i] = -1;
  }
}

template <typename T>
__device__ inline void PackLocalToken(EpDispatchCombineArgs<T>& args, int tokenId, int expertPos,
                                      int slotId, int pairIdx, int capacity) {
  const EpDispatchCombineConfig& config = args.config;
  int expertId = args.shmemOutIndicesMemObj->template GetAs<index_t*>()[tokenId * config.numExpertPerToken + expertPos] %
                 config.numExpertPerRank;

  T* packedTok = args.lowLatencyPackedTokMemObj->template GetAs<T*>();
  const T* src = args.shmemDispatchOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim;
  T* dst = packedTok + (expertId * capacity + slotId) * config.hiddenDim;
  WarpCopy(dst, src, config.hiddenDim);

  float* packedWeight = args.lowLatencyPackedWeightMemObj->template GetAs<float*>();
  if (args.weightsBuf) {
    packedWeight[expertId * capacity + slotId] =
        args.shmemDispatchOutWeightsMemObj->template GetAs<float*>()[tokenId * config.numExpertPerToken + expertPos];
  } else {
    packedWeight[expertId * capacity + slotId] = 1.0f;
  }

  if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
    uint8_t* dstScale = args.lowLatencyPackedScaleMemObj->template GetAs<uint8_t*>() +
                        (expertId * capacity + slotId) * config.scaleDim * config.scaleTypeSize;
    const uint8_t* srcScale = args.shmemOutScalesMemObj->template GetAs<uint8_t*>() +
                              tokenId * config.scaleDim * config.scaleTypeSize;
    WarpCopy<uint8_t, 4>(dstScale, srcScale, config.scaleDim * config.scaleTypeSize);
  }

  args.lowLatencySortedTokenIdxMemObj->template GetAs<index_t*>()[expertId * capacity + slotId] =
      tokenId;
  args.lowLatencyPairCountMemObj->template GetAs<index_t*>()[0] = pairIdx + 1;
}

// Pack local pairs into fused layout (expert-major order)
template <typename T>
__device__ inline void PackLocalPairs(EpDispatchCombineArgs<T>& args, int totalRecvTokenNum) {
  const EpDispatchCombineConfig& config = args.config;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpNum = (blockDim.x * gridDim.x) / warpSize;
  int capacity = config.maxNumInpTokenPerRank;
  int maxPairs = config.numExpertPerRank * capacity;

  for (int tokenId = warpId; tokenId < totalRecvTokenNum; tokenId += warpNum) {
    for (int e = laneId; e < config.numExpertPerToken; e += warpSize) {
      int64_t globalExpert = args.shmemOutIndicesMemObj->template GetAs<index_t*>()[tokenId * config.numExpertPerToken + e];
      int destPe = static_cast<int>(globalExpert / config.numExpertPerRank);
      if (destPe != config.rank) continue;
      int expertId = static_cast<int>(globalExpert % config.numExpertPerRank);
      int slotId = atomicAdd(
          args.lowLatencyExpertCountMemObj->template GetAs<index_t*>() + expertId, 1);
      if (slotId >= capacity) continue;
      int pairIdx = atomicAdd(args.lowLatencyPairCountMemObj->template GetAs<index_t*>(), 1);
      if (pairIdx >= maxPairs) {
        // Cap the counter to avoid out-of-bounds consumers
        atomicExch(args.lowLatencyPairCountMemObj->template GetAs<index_t*>(), maxPairs);
        continue;
      }
      PackLocalToken(args, tokenId, e, slotId, pairIdx, capacity);
    }
  }
}

// ----------------------------- Intra-node fused kernels ---------------------------------------

template <typename T>
__global__ void EpDispatchIntraNodeKernelLLFused(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeKernelBody(args);
  __syncthreads();
  if (blockIdx.x == 0) ZeroFusedMetadata(args);
  __syncthreads();
  int totalRecvTokenNum = *args.totalRecvTokenNum;
  PackLocalPairs(args, totalRecvTokenNum);
}

template <typename T>
__global__ void EpCombineIntraNodeKernelLLFused(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int pairCount = args.lowLatencyPairCountMemObj->template GetAs<index_t*>()[0];
  int capacity = config.maxNumInpTokenPerRank;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpNum = (blockDim.x * gridDim.x) / warpSize;

  const T* packed = args.lowLatencyPackedTokMemObj->template GetAs<T*>();
  const float* packedW = args.lowLatencyPackedWeightMemObj->template GetAs<float*>();
  const index_t* sortedIdx = args.lowLatencySortedTokenIdxMemObj->template GetAs<index_t*>();

  // Zero output buffer
  T* dstBase = args.shmemCombineOutTokMemObj->template GetAs<T*>();
  int totalTokens = args.config.maxNumInpTokenPerRank;
  for (int t = warpId; t < totalTokens; t += warpNum) {
    for (int h = laneId; h < config.hiddenDim; h += warpSize) {
      dstBase[t * config.hiddenDim + h] = static_cast<T>(0);
    }
  }
  __syncthreads();

  for (int pair = warpId; pair < pairCount; pair += warpNum) {
    int expertId = pair / capacity;
    int slotId = pair - expertId * capacity;
    int tokenId = sortedIdx[pair];
    if (tokenId < 0) continue;
    const T* src = packed + (expertId * capacity + slotId) * config.hiddenDim;
    T* dst = args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim;
    float w = packedW[expertId * capacity + slotId];
    for (int h = laneId; h < config.hiddenDim; h += warpSize) {
      AtomicAdd(dst + h, static_cast<T>(w) * src[h]);
    }
  }
}

// ----------------------------- Inter-node V1 fused kernels ------------------------------------

template <typename T>
__global__ void EpDispatchInterNodeV1KernelLLFused(EpDispatchCombineArgs<T> args) {
  EpDispatchInterNodeV1KernelLowLatencyBody(args);
  __syncthreads();
  if (blockIdx.x == 0) ZeroFusedMetadata(args);
  __syncthreads();
  int totalRecvTokenNum = *args.totalRecvTokenNum;
  PackLocalPairs(args, totalRecvTokenNum);
}

template <typename T>
__global__ void EpCombineInterNodeV1KernelLLFused(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int pairCount = args.lowLatencyPairCountMemObj->template GetAs<index_t*>()[0];
  int capacity = config.maxNumInpTokenPerRank;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpNum = (blockDim.x * gridDim.x) / warpSize;

  const T* packed = args.lowLatencyPackedTokMemObj->template GetAs<T*>();
  const float* packedW = args.lowLatencyPackedWeightMemObj->template GetAs<float*>();
  const index_t* sortedIdx = args.lowLatencySortedTokenIdxMemObj->template GetAs<index_t*>();

  // Zero output buffer
  T* dstBase = args.shmemCombineOutTokMemObj->template GetAs<T*>();
  int totalTokens = args.config.maxNumInpTokenPerRank;
  for (int t = warpId; t < totalTokens; t += warpNum) {
    for (int h = laneId; h < config.hiddenDim; h += warpSize) {
      dstBase[t * config.hiddenDim + h] = static_cast<T>(0);
    }
  }
  __syncthreads();

  for (int pair = warpId; pair < pairCount; pair += warpNum) {
    int expertId = pair / capacity;
    int slotId = pair - expertId * capacity;
    int tokenId = sortedIdx[pair];
    if (tokenId < 0) continue;
    const T* src = packed + (expertId * capacity + slotId) * config.hiddenDim;
    T* dst = args.shmemCombineOutTokMemObj->template GetAs<T*>() + tokenId * config.hiddenDim;
    float w = packedW[expertId * capacity + slotId];
    for (int h = laneId; h < config.hiddenDim; h += warpSize) {
      AtomicAdd(dst + h, static_cast<T>(w) * src[h]);
    }
  }
}

// -------------------------------- Explicit instantiation --------------------------------------
#define INSTANTIATE(T)                                                                                      \
  template __global__ void EpDispatchIntraNodeKernelLLFused<T>(EpDispatchCombineArgs<T> args);             \
  template __global__ void EpCombineIntraNodeKernelLLFused<T>(EpDispatchCombineArgs<T> args);              \
  template __global__ void EpDispatchInterNodeV1KernelLLFused<T>(EpDispatchCombineArgs<T> args);          \
  template __global__ void EpCombineInterNodeV1KernelLLFused<T>(EpDispatchCombineArgs<T> args);

INSTANTIATE(float)
INSTANTIATE(hip_bfloat16)

#undef INSTANTIATE

}  // namespace moe
}  // namespace mori
