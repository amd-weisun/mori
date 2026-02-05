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

// Debug prints for dispatch phase - enable via cmake -DENABLE_DISPATCH_DEBUG_PRINTF=ON
// #define ENABLE_DISPATCH_DEBUG_PRINTF 1

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine_deepep/dispatch_combine_deepep.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {
namespace deepep {

/*
 * Multi-node (inter-node) low-latency dispatch/combine kernels for DeepEP format.
 *
 * Algorithm follows DeepEP's proven 6-phase dispatch / 3-phase combine structure:
 *
 * DISPATCH (6 phases):
 *   1. Token Dispatch - SM-strided, local slot assignment, RDMA put for remote
 *   2. Expert Count Aggregation - Last warp adds finish tag offset
 *   3. RDMA Quiet - Thread 0 drains pending RDMA puts
 *   4. Count Signal Sending - Send negative-encoded count via RDMA atomic
 *   5. Grid Barrier - Device-scope barrier between send and receive
 *   6. Receive + Unpack - Poll for signals, copy to packed buffer
 *
 * COMBINE (3 phases):
 *   1. Send Expert Outputs - RDMA put to source ranks
 *   2. Signal Completion - Send completion flag
 *   3. Receive + Accumulate - Wait for flags, weighted accumulation
 *
 * Key design patterns:
 * - Local slot assignment: atomicAdd on local counter, no RDMA fetch atomics
 * - Negative-encoded signals: -count - 1 combines count + signal in one RDMA atomic
 * - Finish counter: FINISHED_SUM_TAG pattern for completion detection without barriers
 * - Source-rank partitioning: each rank gets slots [srcPe * maxTokens, (srcPe+1) * maxTokens)
 */

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

// Warp synchronization with memory fence for AMD GPUs.
// On AMD GPUs, HIP's __syncwarp() only synchronizes thread execution, NOT memory.
// For RDMA operations where the NIC reads from GPU memory, we need system-scope
// memory visibility before posting WQEs.
//
// This matches DeepEP's syncwarp() pattern which uses AMD GCN fence intrinsics,
// but we use the stronger __threadfence_system() for NIC visibility.
__device__ __forceinline__ void syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Stronger sync for RDMA: ensures writes are visible to NIC (external PCIe device)
__device__ __forceinline__ void syncwarp_system() {
  __builtin_amdgcn_wave_barrier();
  __threadfence_system();
}

// Cross-device barrier for internode using RDMA signaling.
// Similar to DeepEP's shmem_device_barrier_all() but using MORI primitives.
// All ranks must call this. Uses crossDeviceBarrierFlag to track barrier count.
// The barrier waits for all ranks to signal with the same flag value.
template <typename T>
__device__ inline void CrossDeviceBarrierInterNode(
    mori::moe::deepep::EpDispatchCombineArgs<T>& args,
    int numSms,
    int barrierIdx = 0) {
  const int myPe = args.config.rank;
  const int npes = args.config.worldSize;
  const int gpuPerNode = args.config.gpuPerNode;
  const int threadId = threadIdx.x;
  const int smId = blockIdx.x;

  // Grid barrier first to ensure all blocks on this rank are done
  // Use barrier index 1 since combineGridBarrier index 0 is used in dispatch Phase 4
  // CrossDeviceBarrier is called with barrierIdx=0 by default, so we use 1+barrierIdx
  detail::GridBarrier(args.combineGridBarrier, numSms, 1 + barrierIdx);

  // System-wide fence to ensure all writes are visible
  __threadfence_system();

  // Drain all pending RDMA operations
  if (threadId == 0 && smId == 0) {
    shmem::ShmemQuietThread();
  }
  __syncthreads();

  // Read current barrier flag and increment
  uint32_t barrierFlag = args.crossDeviceBarrierFlag[0];
  if (threadId == 0 && smId == 0) {
    detail::AtomicAddRelaxed(args.crossDeviceBarrierFlag, 1u);
  }

  // Signal all other ranks that we're at the barrier
  if (threadId == 0 && smId == 0) {
    for (int destPe = 0; destPe < npes; ++destPe) {
      if (destPe == myPe) continue;
      bool isRemote = IsRemoteRank(myPe, destPe, gpuPerNode);
      if (isRemote) {
        // RDMA PUT the barrier flag value
        shmem::ShmemPutTypeImmNbiThread<uint32_t>(
            args.crossDeviceBarrierMemObj,
            myPe * sizeof(uint32_t),
            barrierFlag,
            destPe, 0);
      } else {
        // P2P direct write
        uint32_t* remotePtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(destPe) + myPe;
        detail::AtomicStoreReleaseSystem(remotePtr, barrierFlag);
      }
    }
    // Also set our own flag
    uint32_t* localPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>() + myPe;
    detail::AtomicStoreReleaseSystem(localPtr, barrierFlag);

    // Drain the RDMA signals
    shmem::ShmemQuietThread();
  }
  __syncthreads();

  // Wait for all ranks to arrive (check that all ranks reached this barrier value)
  if (threadId < npes) {
    uint32_t* localPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>() + threadId;
    while (detail::AtomicLoadAcquireSystem(localPtr) < barrierFlag) {
      // spin
    }
  }
  __syncthreads();
}

}  // namespace internode_ll

/* ---------------------------------------------------------------------------------------------- */
/*                              Multi-Node Dispatch Kernel                                         */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool kUseFP8>
__global__ void EpDispatchInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  const int smId = blockIdx.x;
  const int numSms = gridDim.x;
  const int threadId = threadIdx.x;
  const int warpId = threadId / warpSize;
  const int laneId = threadId % warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = smId * warpNum + warpId;
  const int globalWarpNum = numSms * warpNum;

  const int myPe = config.rank;
  const int npes = config.worldSize;
  const int gpuPerNode = config.gpuPerNode;
  const int numLocalExperts = config.numExpertPerRank;
  const int numTopK = config.numExpertPerToken;
  const int numTokens = args.curRankNumToken;
  const int maxTokensPerRank = config.maxNumInpTokenPerRank;
  const index_t expertCapacity = npes * maxTokensPerRank;

  // ========== PHASE 1: TOKEN DISPATCH ==========
  // SM-strided token processing: each SM handles tokens[smId, smId+numSms, smId+2*numSms, ...]

  for (int tokenIdx = smId; tokenIdx < numTokens; tokenIdx += numSms) {
    // Each warp handles one top-k expert for this token
    int destExpert = -1;
    if (warpId < numTopK) {
      destExpert = args.tokenIndices[tokenIdx * numTopK + warpId];
    }

    if (destExpert >= 0) {
      int destPe = destExpert / numLocalExperts;
      int localExpert = destExpert % numLocalExperts;
      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);

      // Local slot assignment - NO remote atomics
      // Each source rank gets reserved slots: [srcPe * maxTokensPerRank, (srcPe+1) * maxTokensPerRank)
      index_t slotIdx = 0;
      if (laneId == 0) {
        slotIdx = atomicAdd(args.atomicCounterPerExpert + destExpert, 1);
#ifdef ENABLE_DISPATCH_DEBUG_PRINTF
        // Track slot allocation for problematic tokens (ranks 4-5 to experts 14-17)
        if ((myPe == 4 || myPe == 5) && destExpert >= 14 && destExpert <= 17 && slotIdx < 3) {
          float srcVal = static_cast<float>(args.inpTokenBuf[tokenIdx * config.hiddenDim]);
          printf("[SLOT] myPe=%d tok=%d exp=%d slot=%d srcVal=%.1f\n",
                 myPe, (int)tokenIdx, destExpert, (int)slotIdx, srcVal);
        }
#endif
      }
      slotIdx = __shfl(slotIdx, 0);

      // Compute destination token ID using source-rank partitioning
      index_t destTokId = myPe * maxTokensPerRank + slotIdx;
      index_t destLinearTok = localExpert * expertCapacity + destTokId;

      // Store mapping for combine phase
      if (laneId == 0) {
        args.dispDestTokIdMap[tokenIdx * numTopK + warpId] = destExpert * expertCapacity + destTokId;
      }

      // Store source token ID mapping at destination
      index_t srcTokMapping = myPe * maxTokensPerRank + tokenIdx;
      if (laneId == 0) {
        if (isRemote) {
          shmem::ShmemPutTypeImmNbiThread<index_t>(
              args.dispTokIdToSrcTokIdMemObj,
              destLinearTok,
              srcTokMapping,
              destPe);
        } else {
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destLinearTok] = srcTokMapping;
        }
      }

      // Copy token data
      size_t baseOffset = destLinearTok * config.hiddenDim;
      size_t srcOffset = tokenIdx * config.hiddenDim;

      if constexpr (kUseFP8) {
        // FP8 quantization and copy
        // Use (tokenIdx * numTopK + warpId) as staging index to avoid race condition:
        // Multiple warps in the same block may process the same tokenIdx for different top-K experts.
        // For remote, we stage to shmemStagingTokMemObj; for local, we write directly to destination.
        index_t stagingIdx = tokenIdx * numTopK + warpId;
        __hip_fp8_storage_t* destFp8;
        if (isRemote) {
          // Stage to local staging buffer with unique stagingIdx
          destFp8 = args.shmemStagingTokMemObj->template GetAs<__hip_fp8_storage_t*>() + stagingIdx * config.hiddenDim;
        } else {
          // Write directly to destination at baseOffset
          destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
              args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(destPe)) + baseOffset;
        }
        int numScales = config.hiddenDim / detail::kNumPerChannels;

        for (int scaleIdx = 0; scaleIdx < numScales; ++scaleIdx) {
          int channelBase = scaleIdx * detail::kNumPerChannels;
          float amax = detail::kFP8Margin;

          // Find max absolute value in channel
          for (int j = laneId; j < detail::kNumPerChannels; j += warpSize) {
            float v = static_cast<float>(args.inpTokenBuf[srcOffset + channelBase + j]);
            amax = fmaxf(amax, fabsf(v));
          }
          amax = detail::QuarterWarpReduceMax(amax);

          float scale = detail::DeepepFp8Scale(amax);
          float scaleInv = detail::DeepepFp8ScaleInv(amax);

          // Store inverse scale (one per channel group)
          if (laneId == 0) {
            if (isRemote) {
              // Stage locally first, will RDMA put later
              // Scales are stored after token data: [token data][scales]
              reinterpret_cast<float*>(
                  args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                  maxTokensPerRank * numTopK * config.hiddenDim)[stagingIdx * numScales + scaleIdx] = scaleInv;
            } else {
              args.shmemOutScalesMemObj->template GetAs<float*>(destPe)[destLinearTok * numScales + scaleIdx] = scaleInv;
            }
          }

          // Quantize and store
          for (int j = laneId; j < detail::kNumPerChannels; j += warpSize) {
            float v = static_cast<float>(args.inpTokenBuf[srcOffset + channelBase + j]);
            destFp8[channelBase + j] = detail::CastFloatToFp8(v, scale);
          }
        }

        // Warp sync with system-scope memory fence for NIC visibility before RDMA PUT
        internode_ll::syncwarp_system();

        // RDMA put for remote ranks
        if (isRemote && laneId == 0) {
          // Put FP8 data from staging buffer to destination
          size_t stagingOffset = stagingIdx * config.hiddenDim * sizeof(__hip_fp8_storage_t);

#ifdef ENABLE_DISPATCH_DEBUG_PRINTF
          // Targeted debug: track tokens from ranks 4-5 to local experts that map to global 14-17
          int globalExp = destPe * numLocalExperts + localExpert;
          if ((myPe == 4 || myPe == 5) && globalExp >= 14 && globalExp <= 17 && slotIdx < 3) {
            float stagedVal = static_cast<float>(destFp8[0]);
            printf("[FP8-PUT] myPe=%d tok=%d exp=%d slot=%d stagingIdx=%d baseOff=%lu stagingOff=%lu destPe=%d val=%.1f\n",
                   myPe, (int)tokenIdx, globalExp, (int)slotIdx, (int)stagingIdx,
                   (unsigned long)baseOffset, (unsigned long)stagingOffset, destPe, stagedVal);
          }
#endif

          shmem::ShmemPutMemNbiThread(
              args.shmemDispatchOutTokMemObj,
              baseOffset * sizeof(__hip_fp8_storage_t),
              args.shmemStagingTokMemObj,
              stagingOffset,
              config.hiddenDim * sizeof(__hip_fp8_storage_t),
              destPe, 0);

          // Put scales from staging buffer to destination
          size_t stagingScalesOffset = maxTokensPerRank * numTopK * config.hiddenDim + stagingIdx * numScales * sizeof(float);
          shmem::ShmemPutMemNbiThread(
              args.shmemOutScalesMemObj,
              destLinearTok * numScales * sizeof(float),
              args.shmemStagingTokMemObj,
              stagingScalesOffset,
              numScales * sizeof(float),
              destPe, 0);
        }
      } else {
        // BF16/FP32: direct copy
        if (isRemote) {
          // Stage locally first
          // Use (tokenIdx * numTopK + warpId) as staging index to avoid race condition:
          // Multiple warps in the same block may process the same tokenIdx for different top-K experts
          index_t stagingIdx = tokenIdx * numTopK + warpId;
          T* localStaging = args.shmemStagingTokMemObj->template GetAs<T*>() + stagingIdx * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            localStaging[j] = args.inpTokenBuf[srcOffset + j];
          }
          // Warp sync with system-scope memory fence for NIC visibility before RDMA PUT
          internode_ll::syncwarp_system();

          // RDMA put
          if (laneId == 0) {
#ifdef ENABLE_DISPATCH_DEBUG_PRINTF
          // Targeted debug: track tokens from ranks 4-5 to local experts that map to global 14-17
          int globalExp = destPe * numLocalExperts + localExpert;
          if ((myPe == 4 || myPe == 5) && globalExp >= 14 && globalExp <= 17 && slotIdx < 3) {
            float stagedVal = static_cast<float>(localStaging[0]);
            printf("[BF16-PUT] myPe=%d tok=%d exp=%d slot=%d stagingIdx=%d baseOff=%lu destPe=%d val=%.1f\n",
                   myPe, (int)tokenIdx, globalExp, (int)slotIdx, (int)stagingIdx,
                   (unsigned long)baseOffset, destPe, stagedVal);
          }
#endif
            shmem::ShmemPutMemNbiThread(
                args.shmemDispatchOutTokMemObj,
                baseOffset * sizeof(T),
                args.shmemStagingTokMemObj,
                stagingIdx * config.hiddenDim * sizeof(T),
                config.hiddenDim * sizeof(T),
                destPe, 0);
          }
        } else {
          // Same-node: direct write
          T* destPtr = args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + baseOffset;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            destPtr[j] = args.inpTokenBuf[srcOffset + j];
          }
        }
      }

      // Copy weights and indices
      if (!isRemote) {
        if (laneId < numTopK) {
          if (args.weightsBuf) {
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe)
                [destLinearTok * numTopK + laneId] = args.weightsBuf[tokenIdx * numTopK + laneId];
          }
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe)
              [destLinearTok * numTopK + laneId] = args.tokenIndices[tokenIdx * numTopK + laneId];
        }
      } else {
        // Stage and RDMA put weights/indices for remote
        if (args.weightsBuf && laneId < numTopK) {
          args.shmemInpWeightsMemObj->template GetAs<float*>()[tokenIdx * numTopK + laneId] =
              args.weightsBuf[tokenIdx * numTopK + laneId];
        }
        if (laneId < numTopK) {
          args.shmemInpIndicesMemObj->template GetAs<index_t*>()[tokenIdx * numTopK + laneId] =
              args.tokenIndices[tokenIdx * numTopK + laneId];
        }
        // Warp sync with system-scope memory fence for NIC visibility before RDMA PUT
        internode_ll::syncwarp_system();

        if (laneId == 0) {
          if (args.weightsBuf) {
            shmem::ShmemPutMemNbiThread(
                args.shmemDispatchOutWeightsMemObj,
                destLinearTok * numTopK * sizeof(float),
                args.shmemInpWeightsMemObj,
                tokenIdx * numTopK * sizeof(float),
                numTopK * sizeof(float),
                destPe, 0);
          }
          shmem::ShmemPutMemNbiThread(
              args.shmemOutIndicesMemObj,
              destLinearTok * numTopK * sizeof(index_t),
              args.shmemInpIndicesMemObj,
              tokenIdx * numTopK * sizeof(index_t),
              numTopK * sizeof(index_t),
              destPe, 0);
        }
      }

      // Signal dispatch completion for this token (local atomic, wavefront sync is sufficient)
      internode_ll::syncwarp();
      if (laneId == 0) {
        detail::AtomicAddRelease(args.finishCounterPerExpert + destExpert, 1u);
      }
    }

    __syncthreads();
  }

  // ========== PHASE 2: RDMA QUIET + GRID BARRIER ==========
  // Ensure all RDMA puts are drained BEFORE signaling (following DeepEP pattern)
  // Grid barrier alone doesn't guarantee RDMA completion
  //
  // CRITICAL: Must grid-barrier FIRST to ensure all blocks have finished posting RDMA,
  // THEN quiet to drain all posted operations. Otherwise, faster blocks might quiet
  // before slower blocks have posted their RDMA operations.

  // System-wide fence for P2P write visibility
  __threadfence_system();
  __syncthreads();

  // Grid barrier to ensure all blocks have finished Phase 1 dispatch (all RDMA posted)
  detail::GridBarrier(args.dispatchGridBarrier, numSms);

  // Now drain all RDMA puts - safe because all blocks have finished posting
  if (threadId == 0) {
    shmem::ShmemQuietThread();
  }
  __syncthreads();

  // ========== PHASE 3: COUNT SIGNAL SENDING ==========
  // Send negative-encoded count to each destination rank for each expert
  // MUST send signal even when count == 0, otherwise receiver will hang
  // Skip self (destPe == myPe) since we handle self-tokens directly in Phase 5

  // Each warp handles one expert
  for (int expert = globalWarpId; expert < npes * numLocalExperts; expert += globalWarpNum) {
    int destPe = expert / numLocalExperts;
    if (destPe == myPe) continue;  // Skip self - handled directly via atomicCounterPerExpert

    if (laneId == 0) {
      int localExpert = expert % numLocalExperts;

      // Get count of tokens this rank sent to this expert
      index_t count = args.atomicCounterPerExpert[expert];

      // Send negative-encoded count: -count - 1
      // Even 0 tokens becomes -1, which is distinguishable from initial value 0
      int64_t encodedCount = -static_cast<int64_t>(count) - 1;

      bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
      if (isRemote) {
        // Use PUT (not AMO_ADD) to write the signal value directly
        // DeepEP uses rocshmem_long_p which is a PUT, not atomic add
        shmem::ShmemPutTypeImmNbiThread<int64_t>(
            args.rdmaRecvCountMemObj,
            (localExpert * npes + myPe) * sizeof(int64_t),  // byte offset
            encodedCount,
            destPe, 0);
      } else {
        // Same-node: direct atomic store with system-scope release
        int64_t* signalSlot = args.rdmaRecvCountMemObj->template GetAs<int64_t*>(destPe) +
                              localExpert * npes + myPe;
        detail::AtomicStoreReleaseSystem(signalSlot, encodedCount);
      }
    }
  }

  // ========== PHASE 4: GRID BARRIER ==========
  // Synchronize all blocks before draining RDMA
  // CRITICAL: Grid barrier FIRST to ensure all blocks have finished posting signals,
  // THEN quiet to drain all posted operations.
  __threadfence_system();
  __syncthreads();

  // Need a second barrier counter since dispatchGridBarrier was used in Phase 2
  // We reuse combineGridBarrier here (will be reset before combine kernel)
  detail::GridBarrier(args.combineGridBarrier, numSms, 0);  // Barrier 0 for dispatch kernel

  // Now drain all RDMA puts (count signals) - safe because all blocks have finished posting
  if (threadId == 0) {
    shmem::ShmemQuietThread();
  }
  __syncthreads();

  // ========== PHASE 5: RECEIVE + UNPACK ==========
  // Poll for negative-encoded counts, allocate packed buffer space
  // All signals should be ready after the grid barrier, just need to read them

  // Process signals from other ranks
  for (int localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
    for (int srcPe = globalWarpId; srcPe < npes; srcPe += globalWarpNum) {
      if (srcPe == myPe) continue;  // Skip self - handled separately below

      if (laneId == 0) {
        // Read signal (should already be set after grid barrier)
        int64_t* signalSlot = args.rdmaRecvCountMemObj->template GetAs<int64_t*>() +
                              localExpert * npes + srcPe;
        int64_t rawCount = 0;

        // Poll until signal arrives (with grid barrier, should be immediate)
        while ((rawCount = detail::AtomicLoadAcquireSystem(signalSlot)) == 0) {
          // spin - should not spin long since all sends completed before barrier
        }

        // Decode count: -1 means 0 tokens, -2 means 1 token, etc.
        int numRecvTokens = static_cast<int>(-rawCount - 1);

        if (numRecvTokens > 0) {
          // Allocate space in packed output buffer
          index_t recvTokenBeginIdx = atomicAdd(args.packedRecvCount + localExpert, numRecvTokens);

          // Store layout info for combine kernel
          args.layoutRange[localExpert * npes + srcPe] =
              internode_ll::Pack2(numRecvTokens, recvTokenBeginIdx);

          // Add to total recv count
          atomicAdd(args.recvTokenCountPerExpert + localExpert, numRecvTokens);
          atomicAdd(args.totalRecvTokenNum, numRecvTokens);
        }

        // Reset signal for next iteration
        detail::AtomicStoreReleaseSystem(signalSlot, int64_t{0});
      }
    }
  }

  // Process self-tokens: tokens this rank sent to its own local experts
  // These don't need RDMA signals, just read the counter directly
  if (globalWarpId == 0 && laneId == 0) {
    for (int localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
      int globalExpert = myPe * numLocalExperts + localExpert;
      index_t selfCount = args.atomicCounterPerExpert[globalExpert];

      if (selfCount > 0) {
        // Allocate space in packed output buffer for self-tokens
        index_t recvTokenBeginIdx = atomicAdd(args.packedRecvCount + localExpert, selfCount);

        // Store layout info for combine kernel (self as srcPe)
        args.layoutRange[localExpert * npes + myPe] =
            internode_ll::Pack2(static_cast<int>(selfCount), static_cast<int>(recvTokenBeginIdx));

        // Add to total recv count
        atomicAdd(args.recvTokenCountPerExpert + localExpert, selfCount);
        atomicAdd(args.totalRecvTokenNum, selfCount);
      }
    }
  }

  __syncthreads();

  // ========== PHASE 6: CROSS-DEVICE BARRIER ==========
  // Ensure all ranks have completed their writes before any rank proceeds to next iteration.
  // This prevents races where one rank's buffer reset overlaps with another rank's writes.
  internode_ll::CrossDeviceBarrierInterNode(args, numSms);
}

/* ---------------------------------------------------------------------------------------------- */
/*                              Multi-Node Combine Kernel                                          */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool kUseFP8, bool kUseWeights, int kNumWarpGroups, int kNumWarpsPerGroup>
__global__ void EpCombineInterNodeDeepepLLKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineDeepepConfig& config = args.config;
  const int smId = blockIdx.x;
  const int numSms = gridDim.x;
  const int threadId = threadIdx.x;
  const int warpId = threadId / warpSize;
  const int laneId = threadId % warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = smId * warpNum + warpId;
  const int globalWarpNum = numSms * warpNum;

  const int myPe = config.rank;
  const int npes = config.worldSize;
  const int gpuPerNode = config.gpuPerNode;
  const int numLocalExperts = config.numExpertPerRank;
  const int numTopK = config.numExpertPerToken;
  const int numTokens = args.curRankNumToken;
  const index_t expertCapacity = npes * config.maxNumInpTokenPerRank;
  const int numExpertsTotal = npes * numLocalExperts;

  // DeepEP-style work distribution: each warp group is responsible for specific experts
  // responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id
  static_assert(kNumWarpGroups > 0, "kNumWarpGroups must be positive");
  static_assert(kNumWarpsPerGroup > 0, "kNumWarpsPerGroup must be positive");
  const int warpGroupId = warpId / kNumWarpsPerGroup;
  const int subWarpId = warpId % kNumWarpsPerGroup;
  const int responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;

  // Compute destination rank and local expert for this warp group's responsible expert
  const int dstRank = responsibleExpertIdx / numLocalExperts;
  const int localExpertIdx = responsibleExpertIdx % numLocalExperts;
  const int globalExpertIdx = myPe * numLocalExperts + localExpertIdx;

  // ========== PHASE 1: SEND EXPERT OUTPUTS ==========
  // DeepEP style: each warp group handles one expert, sub-warps handle different tokens
  // Buffer layout: [global_expert_idx * max_tokens + src_idx] (matches DeepEP)

  if (responsibleExpertIdx < numExpertsTotal) {
    // Get layout info for this expert -> destination rank
    int64_t layout = args.layoutRange[localExpertIdx * npes + dstRank];
    int numTokensToSend, offset;
    internode_ll::Unpack2(layout, numTokensToSend, offset);

    if (numTokensToSend > 0 && dstRank != myPe) {
      bool isRemote = internode_ll::IsRemoteRank(myPe, dstRank, gpuPerNode);

      // Sub-warps handle different tokens (strided by kNumWarpsPerGroup)
      for (int tokenIdx = subWarpId; tokenIdx < numTokensToSend; tokenIdx += kNumWarpsPerGroup) {
        // Source token index from src_info (slot in source PE's token stream)
        // srcIdx is the token's position in dstRank's token stream that was dispatched to this expert
        // For combine, we need to map back: srcLinear is where we stored the output
        index_t srcLinear = localExpertIdx * expertCapacity + dstRank * config.maxNumInpTokenPerRank + tokenIdx;

        // Destination layout: [global_expert_idx * max_tokens + src_idx]
        // This matches DeepEP's buffer layout for rdma_recv_x
        index_t destLinear = globalExpertIdx * config.maxNumInpTokenPerRank + tokenIdx;

        if (isRemote) {
          // Stage to local buffer, then RDMA PUT
          T* localStaging = args.shmemStagingTokMemObj->template GetAs<T*>() + srcLinear * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            localStaging[j] = args.inpTokenBuf[srcLinear * config.hiddenDim + j];
          }
          // Warp sync with system-scope memory fence for NIC visibility before RDMA PUT
          internode_ll::syncwarp_system();

          if (laneId == 0) {
            shmem::ShmemPutMemNbiThread(
                args.shmemCombineOutTokMemObj,
                destLinear * config.hiddenDim * sizeof(T),
                args.shmemStagingTokMemObj,
                srcLinear * config.hiddenDim * sizeof(T),
                config.hiddenDim * sizeof(T),
                dstRank, 0);

            // Also send weights back for accumulation
            if constexpr (kUseWeights) {
              shmem::ShmemPutMemNbiThread(
                  args.shmemCombineOutWeightsMemObj,
                  destLinear * numTopK * sizeof(float),
                  args.shmemDispatchOutWeightsMemObj,
                  srcLinear * numTopK * sizeof(float),
                  numTopK * sizeof(float),
                  dstRank, 0);
            }
          }

          // Fence after each PUT (like DeepEP with ROCM_DISABLE_CTX)
          if (laneId == 0) {
            shmem::ShmemFenceThread();
          }
        } else {
          // Same-node: direct P2P copy using the same destination layout
          T* destPtr = args.shmemCombineInpTokMemObj->template GetAs<T*>(dstRank) + destLinear * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            destPtr[j] = args.inpTokenBuf[srcLinear * config.hiddenDim + j];
          }
        }
      }
    }
  }

  // Synchronize after Phase 1 sends
  // CRITICAL: Grid barrier FIRST to ensure all blocks have finished posting RDMA,
  // THEN quiet to drain all posted operations.
  __threadfence_system();
  __syncthreads();
  detail::GridBarrier(args.combineGridBarrier, numSms, 0);  // Barrier 0

  // Now drain all RDMA puts - safe because all blocks have finished posting
  if (threadId == 0) {
    shmem::ShmemQuietThread();
  }
  __syncthreads();

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  if (subWarpId == 0 && laneId == 0) {
    printf("[COMBINE-PHASE1-DONE] myPe=%d smId=%d warpGroupId=%d responsibleExpert=%d dstRank=%d\n",
           myPe, smId, warpGroupId, responsibleExpertIdx, dstRank);
  }
  __syncthreads();
#endif

  // ========== PHASE 2: SIGNAL COMPLETION ==========
  // DeepEP style: each warp group signals for its responsible expert

  if (responsibleExpertIdx < numExpertsTotal && dstRank != myPe) {
    int64_t layout = args.layoutRange[localExpertIdx * npes + dstRank];
    int numTokensToSend, offset;
    internode_ll::Unpack2(layout, numTokensToSend, offset);

    if (numTokensToSend > 0 && subWarpId == 0 && laneId == 0) {
      bool isRemote = internode_ll::IsRemoteRank(myPe, dstRank, gpuPerNode);

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
      printf("[COMBINE-SIGNAL] myPe=%d -> dstRank=%d globalExpert=%d numTok=%d isRemote=%d\n",
             myPe, dstRank, globalExpertIdx, numTokensToSend, isRemote);
#endif

      if (isRemote) {
        // RDMA atomic add to signal completion
        shmem::ShmemAtomicTypeNonFetchThread<int64_t>(
            args.rdmaRecvFlagMemObj,
            globalExpertIdx * sizeof(int64_t),
            int64_t{1},
            core::atomicType::AMO_ADD,
            dstRank, 0);
        shmem::ShmemFenceThread();  // Fence after signal (like DeepEP)
      } else {
        // P2P: write directly to dstRank's flag buffer
        int64_t* flagSlot = args.rdmaRecvFlagMemObj->template GetAs<int64_t*>(dstRank) + globalExpertIdx;
        detail::AtomicStoreReleaseSystem(flagSlot, int64_t{1});
      }
    }
  }

  // Sync after Phase 2 signaling
  // CRITICAL: Grid barrier FIRST to ensure all blocks have finished posting signals,
  // THEN quiet to drain all posted operations.
  __threadfence_system();
  __syncthreads();
  detail::GridBarrier(args.combineGridBarrier, numSms, 1);  // Barrier 1

  // Now drain all RDMA signals - safe because all blocks have finished posting
  if (threadId == 0) {
    shmem::ShmemQuietThread();
  }
  __syncthreads();

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  if (subWarpId == 0 && laneId == 0) {
    printf("[COMBINE-PHASE2-DONE] myPe=%d smId=%d warpGroupId=%d responsibleExpert=%d signaled to dstRank=%d\n",
           myPe, smId, warpGroupId, responsibleExpertIdx, dstRank);
  }
  __syncthreads();
#endif

  // ========== PHASE 3: RECEIVE + ACCUMULATE ==========
  // Wait for signals from ranks that sent us data

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  if (subWarpId == 0 && laneId == 0) {
    printf("[COMBINE-PHASE3-START] myPe=%d smId=%d warpGroupId=%d responsibleExpert=%d about to wait\n",
           myPe, smId, warpGroupId, responsibleExpertIdx);
  }
  __syncthreads();
#endif

  // Each warp group waits for its responsible expert (like DeepEP)
  if (responsibleExpertIdx < numExpertsTotal && dstRank != myPe) {
    // We need to wait for signal from dstRank telling us data for globalExpertIdx has arrived
    // Actually, we need to wait for OTHER ranks to signal US
    // responsibleExpertIdx tells us which (destPe, localExpert) pair we're responsible for waiting on
    int srcPe = dstRank;  // The PE that would send to us
    int srcGlobalExpert = srcPe * numLocalExperts + localExpertIdx;

    // Check if srcPe actually sent us tokens for this expert
    // During dispatch, tokens from myPe to srcPe's expert were counted
    index_t count = args.atomicCounterPerExpert[srcGlobalExpert];

    if (count > 0 && subWarpId == 0 && laneId == 0) {
      // Wait for completion signal from srcPe
      int64_t* flagSlot = args.rdmaRecvFlagMemObj->template GetAs<int64_t*>() + srcGlobalExpert;

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
      printf("[COMBINE-WAIT] myPe=%d waiting for srcPe=%d expert=%d count=%d\n",
             myPe, srcPe, srcGlobalExpert, (int)count);
#endif

      // Poll until signal arrives with acquire semantics
      int waitIter = 0;
      while (detail::AtomicLoadAcquireSystem(flagSlot) == 0) {
        waitIter++;
#ifdef ENABLE_COMBINE_DEBUG_PRINTF
        if (waitIter % 100000000 == 0) {
          printf("[COMBINE-WAIT-TIMEOUT] myPe=%d srcPe=%d expert=%d iter=%d\n",
                 myPe, srcPe, srcGlobalExpert, waitIter);
        }
#endif
      }

      // Reset flag for next iteration (critical for multi-iteration correctness)
      detail::AtomicStoreReleaseSystem(flagSlot, int64_t{0});

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
      printf("[COMBINE-WAIT-DONE] myPe=%d srcPe=%d expert=%d\n", myPe, srcPe, srcGlobalExpert);
#endif
    }
  }

  // Sync after Phase 3 wait - barrier to ensure all blocks have received their signals
  __threadfence_system();
  __syncthreads();
  detail::GridBarrier(args.combineGridBarrier, numSms, 2);  // Barrier 2

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  if (subWarpId == 0 && laneId == 0) {
    printf("[COMBINE-PHASE3-WAIT-DONE] myPe=%d smId=%d warpGroupId=%d responsibleExpert=%d\n",
           myPe, smId, warpGroupId, responsibleExpertIdx);
  }
  __syncthreads();
#endif

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  __threadfence_system();
  __syncthreads();
  if (threadId == 0) {
    printf("[COMBINE-GRID-BARRIER-DONE] myPe=%d smId=%d passed grid barrier\n", myPe, smId);
  }
  __syncthreads();
#endif

  // Shared memory for source pointers
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * numTopK;
  float* srcWeightScales = nullptr;
  if constexpr (kUseWeights) {
    srcWeightScales = reinterpret_cast<float*>(sharedMem + warpNum * numTopK * sizeof(T*)) + warpId * numTopK;
  }

  // Process each output token (SM-strided, like DeepEP)
  for (int tokenIdx = smId; tokenIdx < numTokens; tokenIdx += numSms) {
    // Only first N threads participate (where N covers hidden dimension)
    if (threadId < config.hiddenDim) {
      // Gather source pointers for each top-k expert
      for (int j = 0; j < numTopK; ++j) {
        if (laneId == 0) {
          index_t destTokId = args.dispDestTokIdMap[tokenIdx * numTopK + j];
          index_t destExpert = destTokId / expertCapacity;
          index_t destLocalTokId = destTokId % expertCapacity;
          index_t destPe = destExpert / numLocalExperts;
          index_t localExpert = destExpert % numLocalExperts;
          index_t destGlobalExpert = destPe * numLocalExperts + localExpert;
          // srcIdx is the slot within the destination buffer
          index_t srcIdx = destLocalTokId % config.maxNumInpTokenPerRank;

          if (destPe < npes) {
            // Buffer layout: [global_expert_idx * max_tokens + src_idx]
            size_t bufferOffset = destGlobalExpert * config.maxNumInpTokenPerRank + srcIdx;

            if (destPe == myPe) {
              // Self-token: use original dispatch output layout
              size_t selfOffset = localExpert * expertCapacity + destLocalTokId;
              srcPtrs[j] = args.inpTokenBuf + selfOffset * config.hiddenDim;
            } else {
              bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
              if (isRemote) {
                srcPtrs[j] = args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                             bufferOffset * config.hiddenDim;
              } else {
                srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>() +
                             bufferOffset * config.hiddenDim;
              }
            }
          } else {
            srcPtrs[j] = nullptr;
          }

          if constexpr (kUseWeights) {
            float w = 1.0f;
            if (destPe < npes) {
              size_t bufferOffset = destGlobalExpert * config.maxNumInpTokenPerRank + srcIdx;
              if (destPe == myPe) {
                size_t selfOffset = localExpert * expertCapacity + destLocalTokId;
                w = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>()[selfOffset * numTopK + j];
              } else {
                bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
                if (isRemote) {
                  w = args.shmemCombineOutWeightsMemObj->template GetAs<float*>()[bufferOffset * numTopK + j];
                } else {
                  size_t selfOffset = localExpert * expertCapacity + destLocalTokId;
                  w = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe)[selfOffset * numTopK + j];
                }
              }
            }
            srcWeightScales[j] = w;
          }
        }
      }
      // Local memory access follows, wavefront sync is sufficient
      internode_ll::syncwarp();

      // Accumulate with weights
      float combinedValue = 0.0f;
      for (int j = 0; j < numTopK; ++j) {
        if (srcPtrs[j] != nullptr) {
          float val = static_cast<float>(srcPtrs[j][threadId]);
          float weight = kUseWeights ? srcWeightScales[j] : 1.0f;
          combinedValue += val * weight;
        }
      }

      // Write result
      T* outPtr = args.shmemStagingTokMemObj->template GetAs<T*>() + tokenIdx * config.hiddenDim;
      outPtr[threadId] = static_cast<T>(combinedValue);
    }
    __syncthreads();
  }

  // Sync after accumulation
  shmem::ShmemQuietThread();
  __threadfence_system();
  __syncthreads();

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  if (threadId == 0) {
    printf("[COMBINE-ACCUMULATE-DONE] myPe=%d smId=%d numTokens=%d\n", myPe, smId, numTokens);
  }
  __syncthreads();
#endif

  // Reset total recv token for next iteration
  if (threadId == 0) {
    *args.totalRecvTokenNum = 0;
  }

#ifdef ENABLE_COMBINE_DEBUG_PRINTF
  if (threadId == 0) {
    printf("[COMBINE-KERNEL-DONE] myPe=%d smId=%d\n", myPe, smId);
  }
#endif
}

}  // namespace deepep
}  // namespace moe
}  // namespace mori
