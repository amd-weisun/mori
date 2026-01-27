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
#include "mori/shmem/shmem.hpp"
#include "mori/shmem/shmem_device_api_wrapper.hpp"

namespace mori {
namespace moe {
namespace deepep {

/*
 * Multi-node (inter-node) low-latency dispatch/combine kernels for DeepEP format.
 *
 * Design follows MORI intranode style:
 * - Uses EpDispatchCombineArgs<T> for kernel arguments
 * - Uses CrossDeviceBarrier for synchronization
 * - Uses core::WarpCopy for data movement
 * - Uses expert-major layout: [local_expert, worldSize * maxNumInpTokenPerRank, hidden]
 *
 * Inter-node specific:
 * - Uses RDMA for cross-node data transfer
 * - Negative count encoding for signaling
 * - Packed message format for FP8: [src_idx | hidden_data | scales]
 */

/* ---------------------------------------------------------------------------------------------- */
/*                                    Inter-Node Helper Functions                                  */
/* ---------------------------------------------------------------------------------------------- */

namespace internode_ll {

// Pack two int32 values into int64
__device__ __forceinline__ int64_t Pack2(int a, int b) {
  return (static_cast<int64_t>(a) << 32) | (static_cast<uint32_t>(b));
}

// Unpack int64 into two int32 values
__device__ __forceinline__ void Unpack2(int64_t packed, int& a, int& b) {
  a = static_cast<int>(packed >> 32);
  b = static_cast<int>(packed & 0xFFFFFFFF);
}

// Check if a rank is on a remote node
__device__ __forceinline__ bool IsRemoteRank(int srcRank, int dstRank, int gpuPerNode) {
  return (srcRank / gpuPerNode) != (dstRank / gpuPerNode);
}

// Get node ID from global rank
__device__ __forceinline__ int GetNodeId(int rank, int gpuPerNode) {
  return rank / gpuPerNode;
}

}  // namespace internode_ll

/* ---------------------------------------------------------------------------------------------- */
/*                              Multi-Node Dispatch Kernel                                         */
/* ---------------------------------------------------------------------------------------------- */

/*
 * Inter-node low-latency dispatch kernel.
 *
 * Similar to EpDispatchIntraNodeDeepepLLKernel but:
 * - Uses RDMA for remote ranks on different nodes
 * - Uses SHMEM for same-node ranks
 * - Packs messages with source token index for combine phase routing
 */

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
  const index_t expertCapacity = config.worldSize * config.maxNumInpTokenPerRank;
  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  const int gpuPerNode = config.gpuPerNode;
  const int myNode = myPe / gpuPerNode;

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
    for (int node = 0; node < npes / gpuPerNode; ++node) {
      args.destNodeTokenCounter[node] = 0;
    }
  }
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);

  if (args.tokenIndices && args.inpTokenBuf) {
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken; i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;
      index_t destNode = destPe / gpuPerNode;
      index_t destTokId = 0;
      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);

      if (laneId == 0) {
        index_t* expertCounter =
            args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe);
        destTokId = atomicAdd(expertCounter + localExpert, 1);
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        if (isRemote) {
          atomicAdd(args.destNodeTokenCounter + destNode, 1);
        }
        args.dispDestTokIdMap[i] = destExpert * expertCapacity + destTokId;
      }
      destTokId = __shfl(destTokId, 0);
      index_t destLinearTok = localExpert * expertCapacity + destTokId;

      if (laneId == 0) {
        core::AtomicStoreRelaxedSystem(
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destLinearTok,
            static_cast<index_t>(myPe * config.maxNumInpTokenPerRank + srcTokId));
      }

      // Copy weights and indices for same-node ranks
      if (!isRemote) {
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
      }

      size_t baseOffset = destLinearTok * config.hiddenDim;
      index_t srcTokOffset = srcTokId * config.hiddenDim;

      if (isRemote) {
        // For remote ranks, use RDMA
        // First, pack and quantize the data locally, then issue RDMA put
        if constexpr (kUseFP8) {
          // FP8 quantization and RDMA
          auto* localFp8Staging = reinterpret_cast<__hip_fp8_storage_t*>(
              args.shmemStagingTokMemObj->template GetAs<uint8_t*>()) +
              srcTokId * config.hiddenDim;
          float* localScalesStaging = reinterpret_cast<float*>(
              args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
              config.maxNumInpTokenPerRank * config.hiddenDim * sizeof(__hip_fp8_storage_t)) +
              srcTokId * (config.hiddenDim / detail::kNumPerChannels);

          int numScales = config.hiddenDim / detail::kNumPerChannels;
          for (int scaleIdx = 0; scaleIdx < numScales; ++scaleIdx) {
            int channelBase = scaleIdx * detail::kNumPerChannels;
            float amax = 0.0f;
            for (int j = laneId; j < detail::kNumPerChannels; j += warpSize) {
              int offset = channelBase + j;
              float v = static_cast<float>(args.inpTokenBuf[srcTokOffset + offset]);
              amax = fmaxf(amax, fabsf(v));
            }
            for (int mask = warpSize / 2; mask > 0; mask >>= 1)
              amax = fmaxf(amax, __shfl_xor(amax, mask));
            float scale = detail::DeepepFp8Scale(amax);
            float scaleInv = detail::DeepepFp8ScaleInv(amax);
            if (laneId == 0) {
              localScalesStaging[scaleIdx] = scaleInv;
            }
            for (int j = laneId; j < detail::kNumPerChannels; j += warpSize) {
              int offset = channelBase + j;
              float v = static_cast<float>(args.inpTokenBuf[srcTokOffset + offset]);
              localFp8Staging[offset] = detail::CastFloatToFp8(v, scale);
            }
          }
          __syncwarp();

          // Issue RDMA put for FP8 data
          if (laneId == 0) {
            void* remoteFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
                args.shmemDispatchOutTokMemObj->peerPtrs[destPe]) + baseOffset;
            mori_shmem_putmem_nbi_thread(remoteFp8, localFp8Staging,
                                          config.hiddenDim * sizeof(__hip_fp8_storage_t), destPe, 0);
            void* remoteScales = reinterpret_cast<void*>(args.shmemOutScalesMemObj->peerPtrs[destPe]);
            remoteScales = reinterpret_cast<float*>(remoteScales) + destLinearTok * numScales;
            mori_shmem_putmem_nbi_thread(remoteScales, localScalesStaging,
                                          numScales * sizeof(float), destPe, 0);
          }
        } else {
          // BF16: direct RDMA
          if (laneId == 0) {
            void* remoteBuf = args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + baseOffset;
            mori_shmem_putmem_nbi_thread(remoteBuf, args.inpTokenBuf + srcTokOffset,
                                          config.hiddenDim * sizeof(T), destPe, 0);
          }
        }

        // Also send weights and indices via RDMA for remote ranks
        if (laneId == 0) {
          if (args.weightsBuf) {
            void* remoteWeights = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe) +
                                  destLinearTok * config.numExpertPerToken;
            mori_shmem_putmem_nbi_thread(remoteWeights,
                                          args.weightsBuf + srcTokId * config.numExpertPerToken,
                                          config.numExpertPerToken * sizeof(float), destPe, 0);
          }
          void* remoteIndices = args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe) +
                                destLinearTok * config.numExpertPerToken;
          mori_shmem_putmem_nbi_thread(remoteIndices,
                                        args.tokenIndices + srcTokId * config.numExpertPerToken,
                                        config.numExpertPerToken * sizeof(index_t), destPe, 0);
        }
      } else {
        // For same-node ranks, use SHMEM (same as intranode)
        if constexpr (kUseFP8) {
          auto* destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
              args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe)) + baseOffset;
          int numScales = config.hiddenDim / detail::kNumPerChannels;
          for (int scaleIdx = 0; scaleIdx < numScales; ++scaleIdx) {
            int channelBase = scaleIdx * detail::kNumPerChannels;
            float amax = 0.0f;
            for (int j = laneId; j < detail::kNumPerChannels; j += warpSize) {
              int offset = channelBase + j;
              float v = static_cast<float>(args.inpTokenBuf[srcTokOffset + offset]);
              amax = fmaxf(amax, fabsf(v));
            }
            for (int mask = warpSize / 2; mask > 0; mask >>= 1)
              amax = fmaxf(amax, __shfl_xor(amax, mask));
            float scale = detail::DeepepFp8Scale(amax);
            float scaleInv = detail::DeepepFp8ScaleInv(amax);
            if (laneId == 0) {
              args.shmemOutScalesMemObj->template GetAs<float*>(destPe)
                  [destLinearTok * numScales + scaleIdx] = scaleInv;
            }
            for (int j = laneId; j < detail::kNumPerChannels; j += warpSize) {
              int offset = channelBase + j;
              float v = static_cast<float>(args.inpTokenBuf[srcTokOffset + offset]);
              destFp8[offset] = detail::CastFloatToFp8(v, scale);
            }
          }
        } else {
          core::WarpCopy(args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + baseOffset,
                         args.inpTokenBuf + srcTokOffset, config.hiddenDim);
        }
      }
    }
  }

  __threadfence_system();
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Signal token counts to destination ranks
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      args.dispatchGridBarrier[0] = 0;

      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;

      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      if (isRemote) {
        // For remote ranks, use RDMA atomic
        shmem::ShmemInt32WaitUntilEquals(signal, 0);
        mori_shmem_int32_p(signal, numTokenSignal, destPe, 0);
      } else {
        shmem::ShmemInt32WaitUntilEquals(signal, 0);
        core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
      }
    }
  }

  // Wait for token counts from all source ranks
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      index_t* signal = recvTokenNums + srcPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[srcPe] = 0;
    }
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        args.recvTokenCountPerExpert[e] = localExpertCounter[e];
      }
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        localExpertCounter[e] = 0;
      }
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                              Multi-Node Combine Kernel                                          */
/* ---------------------------------------------------------------------------------------------- */

/*
 * Inter-node low-latency combine kernel.
 *
 * Similar to EpCombineIntraNodeDeepepLLKernel but:
 * - Uses RDMA to send expert outputs back to remote source ranks
 * - Uses SHMEM for same-node ranks
 */

template <typename T, bool kUseFP8, bool kUseWeights>
__global__ void EpCombineInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int myPe = config.rank;
  const int gpuPerNode = config.gpuPerNode;

  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  const index_t expertCapacity = config.worldSize * config.maxNumInpTokenPerRank;
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];

  // Step 1: Copy expert outputs to symmetric combine buffer for same-node access
  // and issue RDMA puts for remote ranks
  if (args.config.useExternalInpBuffer) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      index_t count = args.recvTokenCountPerExpert ? args.recvTokenCountPerExpert[e] : 0;
      for (int slot = globalWarpId; slot < count; slot += globalWarpNum) {
        index_t linear = e * expertCapacity + slot;

        // Get the source rank and token for this slot
        index_t srcInfo = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[linear];
        index_t srcPe = srcInfo / config.maxNumInpTokenPerRank;
        index_t srcTokId = srcInfo % config.maxNumInpTokenPerRank;
        bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);

        if (isRemote) {
          // For remote ranks, issue RDMA put
          if (laneId == 0) {
            void* remoteBuf = args.shmemCombineOutTokMemObj->template GetAs<T*>(srcPe) +
                              srcTokId * config.hiddenDim;
            mori_shmem_putmem_nbi_thread(remoteBuf, args.inpTokenBuf + linear * config.hiddenDim,
                                          config.hiddenDim * sizeof(T), srcPe, 0);
          }
        } else {
          // For same-node ranks, use WarpCopy
          core::WarpCopy(
              args.shmemCombineInpTokMemObj->template GetAs<T*>() + linear * config.hiddenDim,
              args.inpTokenBuf + linear * config.hiddenDim,
              config.hiddenDim);
        }
      }
    }
  }

  if (args.weightsBuf) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      index_t count = args.recvTokenCountPerExpert ? args.recvTokenCountPerExpert[e] : 0;
      for (int slot = globalWarpId; slot < count; slot += globalWarpNum) {
        index_t linear = e * expertCapacity + slot;

        index_t srcInfo = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[linear];
        index_t srcPe = srcInfo / config.maxNumInpTokenPerRank;
        bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);

        if (!isRemote) {
          core::WarpCopy(
              args.shmemInpWeightsMemObj->template GetAs<float*>() + linear * config.numExpertPerToken,
              args.weightsBuf + linear * config.numExpertPerToken,
              config.numExpertPerToken);
        }
      }
    }
  }

  // Step 2: Cross-rank barrier so all writes are visible
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
  *args.totalRecvTokenNum = 0;

  if (args.curRankNumToken == 0) {
    return;
  }

  // Shared memory setup (same as intranode)
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
      // Step 3: Map each top-k expert to (dest_pe, local_expert, slot)
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId - destExpert * expertCapacity;
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t localExpert = destExpert % config.numExpertPerRank;

      if (destPe < config.worldSize) {
        bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);

        if (isRemote) {
          // For remote experts, the data should already be in our combine out buffer
          // via the RDMA put in the send phase
          size_t baseOffset = tokenId * config.hiddenDim;
          srcPtrs[j] = args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                       baseOffset + hiddenDimOffset;
          srcWeightsPtr[j] = nullptr;  // Weights handled separately for remote
        } else {
          // For same-node experts, read from their symmetric buffer
          size_t baseOffset = (localExpert * expertCapacity + destLocalTokId) * config.hiddenDim;
          srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                       baseOffset + hiddenDimOffset;
          srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                             (localExpert * expertCapacity + destLocalTokId) *
                                 config.numExpertPerToken;
        }
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

    // Step 4: Accumulate into local output buffer
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
