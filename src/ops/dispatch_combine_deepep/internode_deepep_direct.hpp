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
/*                          Inter-Node Barrier (uses RDMA for remote ranks)                       */
/* ---------------------------------------------------------------------------------------------- */
/*
 * 1. Intra-node barrier: direct atomic stores/loads for same-node ranks
 * 2. Inter-node barrier: RDMA atomics via proxy PEs for cross-node synchronization
 *
 * Proxy PE concept: Each rank communicates with its corresponding rank on other nodes.
 * E.g., rank 0 on node 0 <-> rank 8 on node 1 (if gpuPerNode=8)
 */
template <typename T>
inline __device__ void CrossDeviceBarrierInterNodeKernel(EpDispatchCombineArgs<T> args,
                                                          const uint32_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;
  int myPe = args.config.rank;
  int gpuPerNode = args.config.gpuPerNode;
  int myNode = myPe / gpuPerNode;
  int nNodes = args.config.worldSize / gpuPerNode;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  // Step 1: Intra-node barrier - signal to all same-node ranks
  if (globalThdId < gpuPerNode) {
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
    args.combineGridBarrier[0] = 0;

    int destPe = myNode * gpuPerNode + globalThdId;
    // Direct atomic store for same-node ranks
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(destPe) + myPe,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  // Wait for all same-node ranks to signal
  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < gpuPerNode) {
    int srcPe = myNode * gpuPerNode + thdId;
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + srcPe) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();

  // Step 2: Inter-node barrier - use RDMA put instead of atomic (to avoid RDMA atomic issues)
  // Each rank writes its barrier flag to all proxy PEs on other nodes via RDMA put.
  // Since each proxyPe slot is written by only one source rank (myPe), we can use put.
  if (globalThdId < nNodes && globalThdId != myNode) {
    int proxyPe = globalThdId * gpuPerNode + (myPe % gpuPerNode);
    // Use RDMA put to write the barrier flag value directly
    shmem::ShmemPutTypeImmNbiThread<uint32_t>(
        args.crossDeviceBarrierMemObj,
        myPe * sizeof(uint32_t),
        crossDeviceBarrierFlag,
        proxyPe);
  }

  // Wait for all remote nodes to signal (via their proxy PEs)
  // Each proxy PE slot is only written via RDMA add from the corresponding remote rank.
  // After N barrier invocations, the slot has value N. So we wait for crossDeviceBarrierFlag.
  if (thdId < nNodes && thdId != myNode) {
    int proxyPe = thdId * gpuPerNode + (myPe % gpuPerNode);
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) < crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

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
    // Reset local per-(destPe, localExpert) counters for inter-node dispatch
    int localPeCounterSize = npes * config.numExpertPerRank;
    for (int idx = 0; idx < localPeCounterSize; ++idx) {
      args.localPeTokenCounter[idx] = 0;
    }
  }
  CrossDeviceBarrierInterNodeKernel(args, crossDeviceBarrierFlag);

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
        // Use source-rank-partitioned layout for ALL tokens (same-node and remote).
        // Each source rank has a reserved section in each expert's buffer:
        // [srcPe * maxNumInpTokenPerRank, (srcPe+1) * maxNumInpTokenPerRank)
        // This avoids RDMA fetch atomics for remote ranks.

        // Compute our slot within our reserved section.
        // Use local counter partitioned by (destPe, localExpert).
        // Index: destPe * numExpertPerRank + localExpert
        index_t localCounterIdx = destPe * config.numExpertPerRank + localExpert;
        index_t localSlot = atomicAdd(args.localPeTokenCounter + localCounterIdx, 1);

        // Our destTokId within the expert buffer uses source-rank partitioning
        destTokId = myPe * config.maxNumInpTokenPerRank + localSlot;

        if (!isRemote) {
          // For same-node ranks, use local atomicAdd on the expert counter
          index_t* expertCounter =
              args.destExpertTokenCounterMemObj->template GetAs<index_t*>(destPe);
          atomicAdd(expertCounter + localExpert, 1);
        }
        // For remote ranks, we don't use RDMA atomic here.
        // The counts are written via RDMA put after the main loop ends.
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        if (isRemote) {
          atomicAdd(args.destNodeTokenCounter + destNode, 1);
        }
        args.dispDestTokIdMap[i] = destExpert * expertCapacity + destTokId;
      }
      destTokId = __shfl(destTokId, 0);
      index_t destLinearTok = localExpert * expertCapacity + destTokId;

      if (laneId == 0) {
        index_t srcTokMappingValue = static_cast<index_t>(myPe * config.maxNumInpTokenPerRank + srcTokId);
        if (isRemote) {
          // For remote ranks, use RDMA put for the srcTokId mapping
          shmem::ShmemPutTypeImmNbiThread<index_t>(
              args.dispTokIdToSrcTokIdMemObj,
              destLinearTok * sizeof(index_t),
              srcTokMappingValue,
              destPe);
        } else {
          // For same-node ranks, use local atomic store
          core::AtomicStoreRelaxedSystem(
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe) + destLinearTok,
              srcTokMappingValue);
        }
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

          // Issue RDMA put for FP8 data using SymmMemObjPtr-based API
          if (laneId == 0) {
            // Source offset within shmemStagingTokMemObj
            size_t srcFp8Offset = srcTokId * config.hiddenDim * sizeof(__hip_fp8_storage_t);
            // Destination offset within shmemDispatchOutTokMemObj
            size_t destFp8Offset = baseOffset * sizeof(__hip_fp8_storage_t);
            shmem::ShmemPutMemNbiThread(
                args.shmemDispatchOutTokMemObj, destFp8Offset,
                args.shmemStagingTokMemObj, srcFp8Offset,
                config.hiddenDim * sizeof(__hip_fp8_storage_t), destPe, 0);

            // Scales offset
            size_t stagingScalesBase = config.maxNumInpTokenPerRank * config.hiddenDim * sizeof(__hip_fp8_storage_t);
            size_t srcScalesOffset = stagingScalesBase + srcTokId * numScales * sizeof(float);
            size_t destScalesOffset = destLinearTok * numScales * sizeof(float);
            shmem::ShmemPutMemNbiThread(
                args.shmemOutScalesMemObj, destScalesOffset,
                args.shmemStagingTokMemObj, srcScalesOffset,
                numScales * sizeof(float), destPe, 0);
          }
        } else {
          // BF16: stage to symmetric buffer then RDMA put
          // First copy to staging buffer
          T* localStaging = args.shmemStagingTokMemObj->template GetAs<T*>() +
                            srcTokId * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            localStaging[j] = args.inpTokenBuf[srcTokOffset + j];
          }
          __syncwarp();

          // Issue RDMA put using SymmMemObjPtr-based API
          if (laneId == 0) {
            size_t srcOffset = srcTokId * config.hiddenDim * sizeof(T);
            size_t destOffset = baseOffset * sizeof(T);
            shmem::ShmemPutMemNbiThread(
                args.shmemDispatchOutTokMemObj, destOffset,
                args.shmemStagingTokMemObj, srcOffset,
                config.hiddenDim * sizeof(T), destPe, 0);
          }
        }

        // Also send weights and indices via RDMA for remote ranks
        // First stage to local symmetric buffers, then issue RDMA puts
        if (args.weightsBuf) {
          float* localWeightsStaging = args.shmemInpWeightsMemObj->template GetAs<float*>() +
                                       srcTokId * config.numExpertPerToken;
          if (laneId < config.numExpertPerToken) {
            localWeightsStaging[laneId] = args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
          }
        }
        index_t* localIndicesStaging = args.shmemInpIndicesMemObj->template GetAs<index_t*>() +
                                       srcTokId * config.numExpertPerToken;
        if (laneId < config.numExpertPerToken) {
          localIndicesStaging[laneId] = args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
        }
        __syncwarp();

        if (laneId == 0) {
          if (args.weightsBuf) {
            size_t srcWeightsOffset = srcTokId * config.numExpertPerToken * sizeof(float);
            size_t destWeightsOffset = destLinearTok * config.numExpertPerToken * sizeof(float);
            shmem::ShmemPutMemNbiThread(
                args.shmemDispatchOutWeightsMemObj, destWeightsOffset,
                args.shmemInpWeightsMemObj, srcWeightsOffset,
                config.numExpertPerToken * sizeof(float), destPe, 0);
          }
          size_t srcIndicesOffset = srcTokId * config.numExpertPerToken * sizeof(index_t);
          size_t destIndicesOffset = destLinearTok * config.numExpertPerToken * sizeof(index_t);
          shmem::ShmemPutMemNbiThread(
              args.shmemOutIndicesMemObj, destIndicesOffset,
              args.shmemInpIndicesMemObj, srcIndicesOffset,
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

  // Ensure all token data RDMA puts from the main loop are complete.
  // Each warp may have issued puts to various destPes. We need to quiet ALL outstanding
  // puts before proceeding to expert count signaling, since ShmemQuietThread(pe) only
  // affects puts issued by the SAME warp, not puts from other warps.
  {
    __syncthreads();  // Ensure all warps finished the main loop
    // Each lane calls quiet for all outstanding puts from this thread
    if (laneId == 0) {
      shmem::ShmemQuietThread();  // Quiet all pending RDMA puts from this warp
    }
    __syncthreads();  // Ensure all warps have completed their quiet
  }

  // After processing all tokens, use RDMA put to write per-(srcPe, localExpert) counts
  // to each remote destination rank. This avoids RDMA atomics.
  // localPeTokenCounter[destPe * numExpertPerRank + localExpert] contains the count
  // of tokens sent from this rank (myPe) to (destPe, localExpert).
  // We write this to srcExpertTokenCounterMemObj on destPe at slot [myPe * numExpertPerRank + localExpert].
  {
    // Synchronize so all token writes are done before sending counts
    __syncthreads();
    // Each warp handles one destPe
    for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      if (isRemote) {
        // Write counts for all local experts to this remote destPe
        for (int e = laneId; e < config.numExpertPerRank; e += warpSize) {
          index_t localCounterIdx = destPe * config.numExpertPerRank + e;
          index_t count = args.localPeTokenCounter[localCounterIdx];
          // Write to slot [myPe * numExpertPerRank + e] on destPe
          index_t destSlot = myPe * config.numExpertPerRank + e;
          shmem::ShmemPutTypeImmNbiThread<index_t>(
              args.srcExpertTokenCounterMemObj,
              destSlot * sizeof(index_t),
              count,
              destPe);
        }
        // Ensure RDMA puts to this destPe complete before signaling
        if (laneId == 0) {
          shmem::ShmemQuietThread(destPe);
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
        // For remote ranks, use RDMA put to write the signal.
        // We can't directly poll remote memory, so we skip the wait and rely on
        // the barrier synchronization to ensure ordering. The signal value is
        // only written after all data has been sent.
        size_t signalOffset = myPe * sizeof(index_t);
        shmem::ShmemPutTypeImmNbiThread<index_t>(
            args.recvTokenNumMemObj,
            signalOffset,
            numTokenSignal,
            destPe);
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

      // Sum expert counts from all source ranks:
      // - Same-node ranks: already accumulated in destExpertTokenCounterMemObj
      // - Remote ranks: stored in srcExpertTokenCounterMemObj[srcPe * numExpertPerRank + e]
      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
      index_t* srcExpertCounter =
          args.srcExpertTokenCounterMemObj->template GetAs<index_t*>();

      for (int e = 0; e < config.numExpertPerRank; ++e) {
        // Start with same-node counts
        index_t totalCount = localExpertCounter[e];
        // Add remote counts from srcExpertTokenCounterMemObj
        for (int srcPe = 0; srcPe < npes; ++srcPe) {
          bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);
          if (isRemote) {
            totalCount += srcExpertCounter[srcPe * config.numExpertPerRank + e];
          }
        }
        args.recvTokenCountPerExpert[e] = totalCount;
      }

      // Reset counters for next dispatch
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        localExpertCounter[e] = 0;
      }
      for (int srcPe = 0; srcPe < npes; ++srcPe) {
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          srcExpertCounter[srcPe * config.numExpertPerRank + e] = 0;
        }
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
  // and issue RDMA puts for remote ranks.
  // With source-rank-partitioned slots, each source PE has slots at:
  // [srcPe * maxNumInpTokenPerRank, (srcPe+1) * maxNumInpTokenPerRank)
  // We iterate through all source PEs and check for valid entries.
  if (args.config.useExternalInpBuffer) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      // Iterate through all source PEs
      for (int srcPe = 0; srcPe < config.worldSize; ++srcPe) {
        index_t slotBase = srcPe * config.maxNumInpTokenPerRank;
        bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);

        // Each warp handles a portion of this srcPe's slots
        for (int localSlot = globalWarpId; localSlot < config.maxNumInpTokenPerRank;
             localSlot += globalWarpNum) {
          index_t slot = slotBase + localSlot;
          index_t linear = e * expertCapacity + slot;

          // Check if this slot has valid data (srcInfo != -1)
          // Read at lane 0 and broadcast to all lanes
          index_t srcInfo;
          if (laneId == 0) {
            srcInfo = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[linear];
          }
          srcInfo = __shfl(srcInfo, 0);

          if (srcInfo == static_cast<index_t>(-1)) {
            continue;  // Slot not used, all lanes skip together
          }

          index_t srcTokId = srcInfo % config.maxNumInpTokenPerRank;

          if (isRemote) {
            // For remote ranks, first stage to symmetric buffer, then RDMA put
            T* localStaging = args.shmemCombineInpTokMemObj->template GetAs<T*>() +
                              linear * config.hiddenDim;
            for (int j = laneId; j < config.hiddenDim; j += warpSize) {
              localStaging[j] = args.inpTokenBuf[linear * config.hiddenDim + j];
            }
            __syncwarp();

            if (laneId == 0) {
              size_t srcOffset = linear * config.hiddenDim * sizeof(T);
              size_t destOffset = srcTokId * config.hiddenDim * sizeof(T);
              shmem::ShmemPutMemNbiThread(
                  args.shmemCombineOutTokMemObj, destOffset,
                  args.shmemCombineInpTokMemObj, srcOffset,
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
  }

  if (args.weightsBuf) {
    for (int e = 0; e < config.numExpertPerRank; ++e) {
      // Iterate through all source PEs
      for (int srcPe = 0; srcPe < config.worldSize; ++srcPe) {
        index_t slotBase = srcPe * config.maxNumInpTokenPerRank;
        bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);

        if (!isRemote) {
          for (int localSlot = globalWarpId; localSlot < config.maxNumInpTokenPerRank;
               localSlot += globalWarpNum) {
            index_t slot = slotBase + localSlot;
            index_t linear = e * expertCapacity + slot;

            // Check if this slot has valid data (srcInfo != -1)
            index_t srcInfo;
            if (laneId == 0) {
              srcInfo = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>()[linear];
            }
            srcInfo = __shfl(srcInfo, 0);

            if (srcInfo == static_cast<index_t>(-1)) {
              continue;  // Slot not used, all lanes skip together
            }

            core::WarpCopy(
                args.shmemInpWeightsMemObj->template GetAs<float*>() + linear * config.numExpertPerToken,
                args.weightsBuf + linear * config.numExpertPerToken,
                config.numExpertPerToken);
          }
        }
      }
    }
  }

  // Ensure all RDMA puts from the above loops are complete before barrier.
  // Each warp may have issued puts to various srcPes (for combine, sending back to source).
  {
    __syncthreads();  // Ensure all warps finished the RDMA puts
    if (laneId == 0) {
      shmem::ShmemQuietThread();  // Quiet all pending RDMA puts from this warp
    }
    __syncthreads();  // Ensure all warps have completed their quiet
  }

  // Step 2: Cross-rank barrier so all writes are visible
  CrossDeviceBarrierInterNodeKernel(args, crossDeviceBarrierFlag);
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
