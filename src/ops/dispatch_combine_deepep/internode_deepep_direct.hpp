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
        auto* destFp8 = reinterpret_cast<__hip_fp8_storage_t*>(
            args.shmemDispatchOutTokMemObj->template GetAs<uint8_t*>(isRemote ? myPe : destPe)) + baseOffset;
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
              reinterpret_cast<float*>(
                  args.shmemStagingTokMemObj->template GetAs<uint8_t*>() +
                  maxTokensPerRank * config.hiddenDim)[tokenIdx * numScales + scaleIdx] = scaleInv;
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

        __syncwarp();

        // RDMA put for remote ranks
        if (isRemote && laneId == 0) {
          // Put FP8 data
          shmem::ShmemPutMemNbiThread(
              args.shmemDispatchOutTokMemObj,
              baseOffset * sizeof(__hip_fp8_storage_t),
              args.shmemDispatchOutTokMemObj,
              baseOffset * sizeof(__hip_fp8_storage_t),
              config.hiddenDim * sizeof(__hip_fp8_storage_t),
              destPe, 0);

          // Put scales
          size_t stagingScalesOffset = maxTokensPerRank * config.hiddenDim + tokenIdx * numScales * sizeof(float);
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
          T* localStaging = args.shmemStagingTokMemObj->template GetAs<T*>() + tokenIdx * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            localStaging[j] = args.inpTokenBuf[srcOffset + j];
          }
          __syncwarp();

          // RDMA put
          if (laneId == 0) {
            shmem::ShmemPutMemNbiThread(
                args.shmemDispatchOutTokMemObj,
                baseOffset * sizeof(T),
                args.shmemStagingTokMemObj,
                tokenIdx * config.hiddenDim * sizeof(T),
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
        __syncwarp();

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

      // Signal dispatch completion for this token
      __syncwarp();
      if (laneId == 0) {
        detail::AtomicAddRelease(args.finishCounterPerExpert + destExpert, 1u);
      }
    }

    __syncthreads();
  }

  // ========== PHASE 2: RDMA QUIET + GRID BARRIER ==========
  // Ensure all RDMA puts are drained BEFORE grid barrier (following DeepEP pattern)
  // Grid barrier alone doesn't guarantee RDMA completion
  //
  // For same-node P2P writes: __threadfence_system() ensures visibility
  // For remote RDMA puts: ShmemQuietThread() drains pending operations

  // System-wide fence for P2P write visibility + RDMA quiet for remote puts
  if (threadId == 0) {
    shmem::ShmemQuietThread();
  }
  __threadfence_system();  // System-wide visibility for all P2P writes
  __syncthreads();

  // Then grid barrier to ensure all blocks have finished Phase 1 dispatch
  detail::GridBarrier(args.dispatchGridBarrier, numSms);

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

  // Drain all RDMA puts (count signals)
  if (threadId == 0) {
    shmem::ShmemQuietThread();
    __threadfence_system();  // System-wide visibility
  }
  __syncthreads();

  // ========== PHASE 4: GRID BARRIER ==========
  // Synchronize all blocks before receiving

  // Need a second barrier counter since dispatchGridBarrier was used in Phase 2
  // We reuse combineGridBarrier here (will be reset before combine kernel)
  detail::GridBarrier(args.combineGridBarrier, numSms);

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
}

/* ---------------------------------------------------------------------------------------------- */
/*                              Multi-Node Combine Kernel                                          */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool kUseFP8, bool kUseWeights>
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

  // ========== PHASE 1: SEND EXPERT OUTPUTS ==========
  // Copy expert outputs to staging buffer and RDMA put to source ranks

  for (int localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
    for (int srcPe = 0; srcPe < npes; ++srcPe) {
      if (srcPe == myPe) continue;

      // Get layout info from dispatch phase
      int64_t layout = args.layoutRange[localExpert * npes + srcPe];
      int numTokensToSend, offset;
      internode_ll::Unpack2(layout, numTokensToSend, offset);

      if (numTokensToSend == 0) continue;

      bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);

      // Each warp handles a subset of tokens
      // slotIdx iterates from 0 to numTokensToSend-1, representing the slot within srcPe's partition
      // offset is the packed buffer offset (not used for reading input)
      for (int slotIdx = globalWarpId; slotIdx < numTokensToSend; slotIdx += globalWarpNum) {
        // Linear index in expert-major layout: localExpert * expertCapacity + srcPe * maxTokens + slot
        index_t linear = localExpert * expertCapacity + srcPe * config.maxNumInpTokenPerRank + slotIdx;

        if (isRemote) {
          // Stage and RDMA put
          // Use expert-major layout: linear = localExpert * expertCapacity + srcPe * maxTokens + slot
          // This matches what Phase 3 expects when reading via dispDestTokIdMap
          T* localStaging = args.shmemStagingTokMemObj->template GetAs<T*>() + linear * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            localStaging[j] = args.inpTokenBuf[linear * config.hiddenDim + j];
          }
          __syncwarp();

          if (laneId == 0) {
            // Put at expert-major offset, not flat srcTokId offset
            shmem::ShmemPutMemNbiThread(
                args.shmemCombineOutTokMemObj,
                linear * config.hiddenDim * sizeof(T),  // Use linear (expert-major), not srcTokId (flat)
                args.shmemStagingTokMemObj,
                linear * config.hiddenDim * sizeof(T),
                config.hiddenDim * sizeof(T),
                srcPe, 0);
          }
        } else {
          // Same-node: direct P2P copy to srcPe's combine input buffer
          // srcPe will read from their shmemCombineInpTokMemObj at this offset
          T* destPtr = args.shmemCombineInpTokMemObj->template GetAs<T*>(srcPe) + linear * config.hiddenDim;
          for (int j = laneId; j < config.hiddenDim; j += warpSize) {
            destPtr[j] = args.inpTokenBuf[linear * config.hiddenDim + j];
          }
        }
      }
    }
  }

  // Drain RDMA puts and ensure P2P writes are visible
  if (threadId == 0) {
    shmem::ShmemQuietThread();
  }
  __threadfence_system();  // System-wide visibility for all P2P writes
  __syncthreads();

  // ========== PHASE 2: SIGNAL COMPLETION ==========
  // Send completion flag to each source rank

  for (int localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
    for (int srcPe = globalWarpId; srcPe < npes; srcPe += globalWarpNum) {
      if (srcPe == myPe) continue;

      int64_t layout = args.layoutRange[localExpert * npes + srcPe];
      int numTokensToSend, offset;
      internode_ll::Unpack2(layout, numTokensToSend, offset);

      if (numTokensToSend == 0) continue;

      bool isRemote = internode_ll::IsRemoteRank(myPe, srcPe, gpuPerNode);
      int globalExpertIdx = myPe * numLocalExperts + localExpert;

      if (laneId == 0) {
        if (isRemote) {
          shmem::ShmemAtomicTypeNonFetchThread<int64_t>(
              args.rdmaRecvFlagMemObj,
              globalExpertIdx * sizeof(int64_t),
              int64_t{1},
              core::atomicType::AMO_ADD,
              srcPe, 0);
          shmem::ShmemQuietThread(srcPe);
        } else {
          // Same-node: write to srcPe's buffer via P2P (not local buffer!)
          int64_t* flagSlot = args.rdmaRecvFlagMemObj->template GetAs<int64_t*>(srcPe) + globalExpertIdx;
          detail::AtomicAddReleaseSystem(flagSlot, int64_t{1});
        }
      }
    }
  }

  // Ensure all signals are sent and visible
  __threadfence_system();
  __syncthreads();

  // ========== PHASE 3: RECEIVE + ACCUMULATE ==========
  // Wait for all expert outputs, then accumulate with weights

  // Wait for completion signals using per-block expert assignment (like DeepEP)
  // Each block is responsible for specific experts based on blockIdx.x
  // This avoids deadlock by ensuring we only wait for our responsible experts
  constexpr int kNumWarpGroups = 2;
  int warpGroupId = warpId / (warpNum / kNumWarpGroups);
  int responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;

  if (responsibleExpertIdx < npes * numLocalExperts) {
    int destPe = responsibleExpertIdx / numLocalExperts;
    int localExpert = responsibleExpertIdx % numLocalExperts;

    // Only wait if this is a remote expert we sent tokens to
    if (destPe != myPe) {
      int globalExpert = destPe * numLocalExperts + localExpert;
      index_t count = args.atomicCounterPerExpert[globalExpert];

      if (count > 0 && warpId % (warpNum / kNumWarpGroups) == 0 && laneId == 0) {
        // Wait for completion signal from destPe for this expert
        int globalExpertIdx = destPe * numLocalExperts + localExpert;
        int64_t* flagSlot = args.rdmaRecvFlagMemObj->template GetAs<int64_t*>() + globalExpertIdx;

        // Poll until signal arrives (acquire for visibility of data)
        while (detail::AtomicLoadAcquireSystem(flagSlot) == 0) {
          // spin
        }
      }
    }
  }
  __syncthreads();

  // Grid barrier to ensure all blocks have received their signals
  detail::GridBarrier(args.combineGridBarrier, numSms);

  // Shared memory for source pointers
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * numTopK;
  float* srcWeightScales = nullptr;
  if constexpr (kUseWeights) {
    srcWeightScales = reinterpret_cast<float*>(sharedMem + warpNum * numTopK * sizeof(T*)) + warpId * numTopK;
  }

  // Process each output token
  index_t warpsPerToken = max(1, (globalWarpNum + numTokens - 1) / numTokens);
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < numTokens * warpsPerToken; i += globalWarpNum) {
    index_t tokenIdx = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize = min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp);
    if (hiddenDimSize <= 0) continue;

    // Gather source pointers for each top-k expert
    for (int j = laneId; j < numTopK; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenIdx * numTopK + j];
      index_t destExpert = destTokId / expertCapacity;
      index_t destLocalTokId = destTokId % expertCapacity;
      index_t destPe = destExpert / numLocalExperts;
      index_t localExpert = destExpert % numLocalExperts;

      if (destPe < npes) {
        // baseOffset uses expert-major layout matching dispatch and Phase 1 send
        size_t baseOffset = (localExpert * expertCapacity + destLocalTokId) * config.hiddenDim;

        if (destPe == myPe) {
          // Self-token: expert output is in our own dispatch output buffer (inpTokenBuf)
          // No inter-rank transfer needed
          srcPtrs[j] = args.inpTokenBuf + baseOffset + hiddenDimOffset;
        } else {
          bool isRemote = internode_ll::IsRemoteRank(myPe, destPe, gpuPerNode);
          if (isRemote) {
            // Data was sent to our combine out buffer via RDMA
            srcPtrs[j] = args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                         baseOffset + hiddenDimOffset;
          } else {
            // Data was written to our combine input buffer via P2P by destPe
            // (Phase 1 on destPe wrote to args.shmemCombineInpTokMemObj->GetAs<T*>(myPe))
            srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>() +
                         baseOffset + hiddenDimOffset;
          }
        }
      } else {
        srcPtrs[j] = nullptr;
      }

      if constexpr (kUseWeights) {
        float w = 1.0f;
        if (args.weightsBuf && j < numTopK) {
          w = args.weightsBuf[tokenIdx * numTopK + j];
        }
        srcWeightScales[j] = w;
      }
    }

    __syncwarp();

    // Accumulate into the staging buffer (shmemStagingTokMemObj)
    // We can't use:
    // - shmemCombineOutTokMemObj: contains RDMA-received expert outputs
    // - shmemCombineInpTokMemObj: contains P2P-received expert outputs
    // - shmemDispatchOutTokMemObj: contains self-token expert outputs (inpTokenBuf points here)
    // shmemStagingTokMemObj is safe because it's only used transiently in combine Phase 1
    T* outPtr = args.shmemStagingTokMemObj->template GetAs<T*>() +
                tokenIdx * config.hiddenDim + hiddenDimOffset;
    core::WarpAccum<T, 4>(outPtr, srcPtrs, kUseWeights ? srcWeightScales : nullptr,
                          numTopK, hiddenDimSize);
  }

  // Reset total recv token for next iteration
  if (threadId == 0) {
    *args.totalRecvTokenNum = 0;
  }
}

}  // namespace deepep
}  // namespace moe
}  // namespace mori
