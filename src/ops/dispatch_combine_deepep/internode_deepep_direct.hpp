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

// Debug flag for inter-node dispatch/combine - set to 1 to enable debug prints
#define INTERNODE_DEEPEP_DEBUG 0

// Timeout for RDMA polling loops (200G cycles ~= 100s at 2GHz)
#define INTERNODE_TIMEOUT_CYCLES 200000000000ll

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

// RDMA-aware polling: wait until value > threshold with memory fence.
// The __threadfence_system() inside the loop forces cache invalidation to
// see RDMA-written data. Without this, the GPU cache may hold stale values
// indefinitely, causing deadlocks when waiting for remote RDMA updates.
template <typename T>
__device__ __forceinline__ T RdmaWaitUntilGreaterThan(T* addr, T val) {
  T got;
  do {
    got = core::AtomicLoadRelaxedSystem(addr);
    __threadfence_system();  // Force cache invalidation to see RDMA writes
  } while (got <= val);
  return got;
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
  // Wait for all warps to arrive, then reset barrier (only thread 0 resets)
  if (globalThdId < gpuPerNode) {
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
  }
  __syncthreads();
  if (globalThdId == 0) {
    args.combineGridBarrier[0] = 0;
  }
  __syncthreads();

  if (globalThdId < gpuPerNode) {
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
    // Must quiet to ensure the RDMA put is visible on remote rank before we poll
    shmem::ShmemQuietThread(proxyPe);
  }

  // Wait for all remote nodes to signal (via their proxy PEs)
  // Each proxy PE slot is only written via RDMA add from the corresponding remote rank.
  // After N barrier invocations, the slot has value N. So we wait for crossDeviceBarrierFlag.
  // CRITICAL: Add __threadfence_system() inside the loop to force cache invalidation.
  // Without this, RDMA-written data may not be visible due to GPU cache coherence issues.
  if (thdId < nNodes && thdId != myNode) {
    int proxyPe = thdId * gpuPerNode + (myPe % gpuPerNode);
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + proxyPe) < crossDeviceBarrierFlag) {
      __threadfence_system();  // Force cache invalidation to see RDMA writes
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

Phase 1: RESET (thread 0 only)
├── Reset local counters: destPeTokenCounter, localPeTokenCounter, etc.
├── Reset srcExpertTokenCounterMemObj (receives remote counts)
├── Reset recvTokenNumMemObj (signal slots)
└── CrossDeviceBarrier #1 (crossDeviceBarrierFlag)

Phase 2: TOKEN DISPATCH (all warps)
├── For each token:
│   ├── Compute destPe, localExpert, destTokId
│   ├── Update local counters (atomicAdd)
│   ├── If REMOTE:
│   │   ├── RDMA PUT: dispTokIdToSrcTokIdMemObj[destLinearTok] → destPe
│   │   ├── RDMA PUT: shmemDispatchOutTokMemObj[baseOffset] → destPe
│   │   ├── RDMA PUT: shmemOutScalesMemObj (if FP8) → destPe
│   │   ├── RDMA PUT: shmemDispatchOutWeightsMemObj → destPe
│   │   └── RDMA PUT: shmemOutIndicesMemObj → destPe
│   └── If SAME-NODE:
│       └── Direct memory writes to destPe's buffers
└── Quiet all remote PEs after token dispatch loop

Phase 3: COUNT EXCHANGE (per remote destPe)
├── RDMA PUT: localPeTokenCounter → srcExpertTokenCounterMemObj on destPe
├── Quiet each destPe after count puts
└── Local grid barrier (dispatchGridBarrier)

Phase 4: CROSS-DEVICE BARRIER #2 (crossDeviceBarrierFlag + 1)
└── Ensures all count PUTs are globally visible

Phase 5: SIGNAL SENDING (warp 0 only)
├── For each destPe:
│   ├── If REMOTE: RDMA PUT signal → recvTokenNumMemObj[myPe] on destPe
│   └── If SAME-NODE: Direct store to signal slot
└── Quiet after each remote signal

Phase 6: SIGNAL RECEIVING & COUNT ACCUMULATION (warp 0 only)
├── Poll recvTokenNumMemObj for signals from all srcPe
├── __syncwarp() + __threadfence_system()
├── Read destExpertTokenCounterMemObj (same-node counts)
├── Read srcExpertTokenCounterMemObj (remote counts) ← USES AtomicLoadRelaxedSystem
└── Sum counts → recvTokenCountPerExpert

Phase 7: CROSS-DEVICE BARRIER #3 (crossDeviceBarrierFlag + 2)
└── Final sync before kernel exit

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

#if INTERNODE_DEEPEP_DEBUG
  // Debug: Print kernel config and pointers at start
  if (globalWarpId == 0 && laneId == 0) {
    printf("[DEBUG][Rank %d] Kernel start: gridDim.x=%d, blockDim.x=%d, globalWarpNum=%d\n",
           myPe, gridDim.x, blockDim.x, globalWarpNum);
    printf("[DEBUG][Rank %d] curRankNumToken=%d, numExpertPerToken=%d, totalIterations=%d\n",
           myPe, args.curRankNumToken, config.numExpertPerToken,
           args.curRankNumToken * config.numExpertPerToken);
    printf("[DEBUG][Rank %d] tokenIndices=%p, inpTokenBuf=%p\n",
           myPe, (void*)args.tokenIndices, (void*)args.inpTokenBuf);
  }
#endif

  // Reset local counters and synchronize all ranks before using remote counters.
  if (globalWarpId == 0 && laneId == 0) {
    // Reset totalRecvTokenNum (normally reset by combine kernel, but needed for dispatch-only)
    *args.totalRecvTokenNum = 0;

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
    // Reset srcExpertTokenCounterMemObj (receives remote counts via RDMA)
    index_t* srcExpertCounter =
        args.srcExpertTokenCounterMemObj->template GetAs<index_t*>();
    for (int srcPe = 0; srcPe < npes; ++srcPe) {
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        srcExpertCounter[srcPe * config.numExpertPerRank + e] = 0;
      }
    }
    // Reset dispatch grid barrier
    args.dispatchGridBarrier[0] = 0;
    // Reset dispTokOffset
    args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    // Reset recvTokenNum signals
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int pe = 0; pe < npes; ++pe) {
      recvTokenNums[pe] = 0;
    }
#if INTERNODE_DEEPEP_DEBUG
    printf("[DEBUG][Rank %d] Dispatch start: reset complete, barrierFlag=%u, curRankNumToken=%d, numExpertPerToken=%d, totalTokensToDispatch=%d\n",
           myPe, crossDeviceBarrierFlag, args.curRankNumToken, config.numExpertPerToken,
           args.curRankNumToken * config.numExpertPerToken);
#endif
  }
  CrossDeviceBarrierInterNodeKernel(args, crossDeviceBarrierFlag);

  if (args.tokenIndices && args.inpTokenBuf) {
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken; i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
#if INTERNODE_DEEPEP_DEBUG
      // Debug: Print first few accesses to verify bounds
      if (i < 5 && laneId == 0) {
        printf("[DEBUG][Rank %d] Warp %d accessing tokenIndices[%d], ptr=%p\n",
               myPe, globalWarpId, i, (void*)(args.tokenIndices + i));
      }
#endif
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
  // Each warp may have issued puts to various destPes. ShmemQuietThread(pe) only
  // affects puts issued by the SAME warp, so each warp must quiet ALL remote PEs
  // it may have sent to (not just one PE per warp).
  // NOTE: We only quiet REMOTE PEs (inter-node) since same-node transfers don't use RDMA.
  {
    __syncthreads();  // Ensure all warps finished the main loop
    // Each warp quiets ALL remote PEs it may have sent to
    if (laneId == 0) {
      for (int destPe = 0; destPe < npes; ++destPe) {
        bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
        if (isRemote) {
          shmem::ShmemQuietThread(destPe);
        }
      }
    }
    __syncthreads();  // Ensure all warps have completed their quiet
#if INTERNODE_DEEPEP_DEBUG
    if (globalWarpId == 0 && laneId == 0) {
      printf("[DEBUG][Rank %d] After token dispatch: destPeTokenCounter = [", myPe);
      for (int pe = 0; pe < npes; ++pe) {
        printf("%d%s", args.destPeTokenCounter[pe], pe < npes-1 ? ", " : "");
      }
      printf("]\n");
      // Also print localPeTokenCounter totals per destPe
      printf("[DEBUG][Rank %d] localPeTokenCounter totals per destPe = [", myPe);
      for (int pe = 0; pe < npes; ++pe) {
        index_t total = 0;
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          total += args.localPeTokenCounter[pe * config.numExpertPerRank + e];
        }
        printf("%d%s", total, pe < npes-1 ? ", " : "");
      }
      printf("]\n");
      // Print destNodeTokenCounter (remote tokens per node)
      int nNodes = npes / gpuPerNode;
      printf("[DEBUG][Rank %d] destNodeTokenCounter (remote only) = [", myPe);
      for (int node = 0; node < nNodes; ++node) {
        printf("%d%s", args.destNodeTokenCounter[node], node < nNodes-1 ? ", " : "");
      }
      printf("]\n");
      // Compute total tokens sent
      index_t totalSent = 0;
      for (int pe = 0; pe < npes; ++pe) {
        totalSent += args.destPeTokenCounter[pe];
      }
      printf("[DEBUG][Rank %d] Total tokens sent = %d\n", myPe, totalSent);
    }
#endif
  }

  // After processing all tokens, use RDMA put with signal to write per-(srcPe, localExpert)
  // counts to each remote destination rank. Using ShmemPutMemNbiSignalThread ensures
  // hardware-enforced ordering: when the signal arrives, the data is guaranteed visible.
  // This avoids the race condition where separate count puts and signals can arrive
  // out of order on network RDMA.
  //
  // localPeTokenCounter[destPe * numExpertPerRank + localExpert] contains the count
  // of tokens sent from this rank (myPe) to (destPe, localExpert).
  // We stage this to our local portion of srcExpertTokenCounterMemObj, then RDMA put
  // with signal to destPe.
  {
    // Synchronize so all token writes are done before sending counts
    __syncthreads();

    // Get pointers to local staging area (our portion of srcExpertTokenCounterMemObj)
    index_t* localSrcExpertCounter =
        args.srcExpertTokenCounterMemObj->template GetAs<index_t*>();

    // Each warp handles one destPe
    for (int destPe = globalWarpId; destPe < npes; destPe += globalWarpNum) {
      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      if (isRemote) {
        // Stage counts to our local portion of srcExpertTokenCounterMemObj
        // Use slots [myPe * numExpertPerRank, myPe * numExpertPerRank + numExpertPerRank)
        // which are our "outgoing" slots (not used for receiving from ourselves).
        if (laneId == 0) {
          for (int e = 0; e < config.numExpertPerRank; ++e) {
            index_t localCounterIdx = destPe * config.numExpertPerRank + e;
            index_t count = args.localPeTokenCounter[localCounterIdx];
            // Stage to our local portion
            localSrcExpertCounter[myPe * config.numExpertPerRank + e] = count;
          }
        }
        __syncwarp();

        // Use ShmemPutMemNbiSignalThread to send counts AND signal atomically.
        // The RDMA hardware guarantees that when the signal arrives at destPe,
        // the count data is also visible. This avoids the out-of-order issue.
        if (laneId == 0) {
          // Signal value: numTokens + 1 (same encoding as before)
          index_t numTokenSignal = args.destPeTokenCounter[destPe] + 1;

          // Source: our local srcExpertTokenCounterMemObj at [myPe * numExpertPerRank]
          size_t srcOffset = myPe * config.numExpertPerRank * sizeof(index_t);
          // Dest: remote srcExpertTokenCounterMemObj at [myPe * numExpertPerRank]
          size_t destOffset = myPe * config.numExpertPerRank * sizeof(index_t);
          size_t countBytes = config.numExpertPerRank * sizeof(index_t);
          // Signal dest: recvTokenNumMemObj[myPe] on destPe
          size_t signalOffset = myPe * sizeof(index_t);

          shmem::ShmemPutMemNbiSignalThread(
              args.srcExpertTokenCounterMemObj, destOffset,
              args.srcExpertTokenCounterMemObj, srcOffset,
              countBytes,
              args.recvTokenNumMemObj, signalOffset,
              static_cast<uint64_t>(numTokenSignal),
              core::atomicType::ATOMIC_SWAP,  // Use swap to set the signal value
              destPe);

          // Quiet to ensure the put+signal is issued
          shmem::ShmemQuietThread(destPe);

#if INTERNODE_DEEPEP_DEBUG
          index_t sentTotal = 0;
          for (int e = 0; e < config.numExpertPerRank; ++e) {
            sentTotal += args.localPeTokenCounter[destPe * config.numExpertPerRank + e];
          }
          printf("[DEBUG][Rank %d] Sent count+signal RDMA to destPe=%d: total=%d tokens, signal=%d\n",
                 myPe, destPe, sentTotal, numTokenSignal);
#endif
        }
      }
    }
  }

  __threadfence_system();
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Wait for all local warps to complete their count+signal puts.
  if (globalWarpId == 0 && laneId == 0) {
    shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
    args.dispatchGridBarrier[0] = 0;
  }
  __syncthreads();

  // Cross-device barrier to ensure all ranks have sent their count+signal RDMA puts.
  CrossDeviceBarrierInterNodeKernel(args, crossDeviceBarrierFlag + 1);

  // Signal token counts to same-node destination ranks (direct store, not RDMA)
  // Remote ranks already received their signals via the bundled put above.
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      if (!isRemote) {
        // For same-node ranks, use direct store for the signal
        index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
        index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
        shmem::ShmemInt32WaitUntilEquals(signal, 0);
        core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
      }
    }
  }

  // Wait for token counts from all source ranks
  // With ShmemPutMemNbiSignalThread, when the signal arrives the count data is
  // guaranteed to be visible (hardware-enforced ordering).
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      index_t* signal = recvTokenNums + srcPe;
      // Use RDMA-aware polling with memory fence to see remote RDMA writes
      index_t recvTokenNum = internode_ll::RdmaWaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
#if INTERNODE_DEEPEP_DEBUG
      printf("[DEBUG][Rank %d] Received signal from srcPe=%d: recvTokenNum=%d\n",
             myPe, srcPe, recvTokenNum);
#endif
    }
    // Ensure all lanes have received their signals before lane 0 reads counts.
    __syncwarp();
    // Memory fence to ensure count data (bundled with signal) is visible
    __threadfence_system();

    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;

      index_t* localExpertCounter =
          args.destExpertTokenCounterMemObj->template GetAs<index_t*>(config.rank);
      index_t* srcExpertCounter =
          args.srcExpertTokenCounterMemObj->template GetAs<index_t*>();

      // Sum expert counts from all source ranks
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        // Start with same-node counts
        index_t totalCount = localExpertCounter[e];
#if INTERNODE_DEEPEP_DEBUG
        index_t sameNodeCount = localExpertCounter[e];
        index_t remoteCount = 0;
#endif
        // Add remote counts from srcExpertTokenCounterMemObj
        for (int srcPe = 0; srcPe < npes; ++srcPe) {
          bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);
          if (isRemote) {
            index_t srcCount = core::AtomicLoadRelaxedSystem(
                srcExpertCounter + srcPe * config.numExpertPerRank + e);
            totalCount += srcCount;
#if INTERNODE_DEEPEP_DEBUG
            remoteCount += srcCount;
            if (srcCount > 0) {
              printf("[DEBUG][Rank %d] Expert %d: from srcPe=%d got count=%d\n",
                     myPe, e, srcPe, srcCount);
            }
#endif
          }
        }
        args.recvTokenCountPerExpert[e] = totalCount;
#if INTERNODE_DEEPEP_DEBUG
        printf("[DEBUG][Rank %d] Expert %d: sameNode=%d, remote=%d, total=%d\n",
               myPe, e, sameNodeCount, remoteCount, totalCount);
#endif
      }

      // Reset counters for next dispatch
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        localExpertCounter[e] = 0;
      }
      for (int srcPe = 0; srcPe < npes; ++srcPe) {
        args.destPeTokenCounter[srcPe] = 0;
        for (int e = 0; e < config.numExpertPerRank; ++e) {
          srcExpertCounter[srcPe * config.numExpertPerRank + e] = 0;
        }
      }
#if INTERNODE_DEEPEP_DEBUG
      // Print final summary
      index_t totalRecv = *args.totalRecvTokenNum;
      printf("[DEBUG][Rank %d] Final dispatch summary: totalRecvTokenNum=%d, recvTokenCountPerExpert=[",
             myPe, totalRecv);
      for (int e = 0; e < config.numExpertPerRank; ++e) {
        printf("%d%s", args.recvTokenCountPerExpert[e], e < config.numExpertPerRank-1 ? ", " : "");
      }
      printf("]\n");
#endif
    }
  }

  // Final barrier to ensure all ranks have completed dispatch before kernel exits.
  // This prevents race conditions where one rank starts validation while another
  // is still processing RDMA operations.
  // Note: We use +2 because +1 was used for the mid-dispatch count sync barrier.
  CrossDeviceBarrierInterNodeKernel(args, crossDeviceBarrierFlag + 2);
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
            // Use shmemStagingTokMemObj (known to work from dispatch) as source.
            // Use linear offset for staging to avoid race conditions between warps.
            T* localStaging = args.shmemStagingTokMemObj->template GetAs<T*>() +
                              linear * config.hiddenDim;
            for (int j = laneId; j < config.hiddenDim; j += warpSize) {
              localStaging[j] = args.inpTokenBuf[linear * config.hiddenDim + j];
            }
            __syncwarp();

            // Use warp-level RDMA put like the legacy internode kernel does.
            // Element offsets (not byte offsets) and element counts are used.
            size_t srcElmOffset = linear * config.hiddenDim;
            size_t destElmOffset = srcTokId * config.hiddenDim;
            shmem::ShmemPutTypeNbiWarp<T>(
                args.shmemCombineOutTokMemObj, destElmOffset,
                args.shmemStagingTokMemObj, srcElmOffset,
                config.hiddenDim, srcPe, 0);
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

  // Synchronize all threads before barrier.
  // Note: We don't call ShmemQuietThread here like the legacy kernel doesn't.
  // The warp-level RDMA puts and the cross-device barrier handle synchronization.
  // Calling quiet after warp-level puts can cause issues.
  __threadfence_system();
  __syncthreads();

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
