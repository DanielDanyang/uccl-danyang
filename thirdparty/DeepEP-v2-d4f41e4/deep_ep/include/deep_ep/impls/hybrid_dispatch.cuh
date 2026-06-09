#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

#ifdef DEEPEP_USE_UCCL_GIN
#include <uccl_gin/uccl_gin_handle.cuh>
#endif

namespace deep_ep::elastic {

template <bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumScaleoutWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumChannelsPerSM = kNumScaleoutWarps,
          int kNumChannels = kNumScaleoutWarps * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kScaleoutUpdateInterval = 3,
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,
          int kNumForwardThreads = kNumForwardWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    int* token_metadata_at_forward,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    // TODO(NCCL): so many params, plans to optimize?
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
#ifdef DEEPEP_USE_UCCL_GIN
    const uccl_gin::UCCLGinResources uccl_gin_resources,
#endif
    void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size");
#ifdef DEEPEP_USE_UCCL_GIN
    EP_STATIC_ASSERT(kNumMaxTokensPerChannel < handle::kUCCLGinTailFinishDelta,
                     "UCCL-GIN packed tail finish bit requires a larger finish delta");
#endif

    // Utils
    // NOTES: a warp is a channel (different channels may share QPs)
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, (kNumNotifyWarps > 0)>(
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, warp_idx < kNumNotifyWarps);
#ifdef DEEPEP_USE_UCCL_GIN
    EP_STATIC_ASSERT(kScaleoutUpdateInterval + handle::kUCCLGinTailFinishDelta <= uccl_gin::kAtomicValueMax,
                     "UCCL-GIN tail delta must fit the ordered atomic immediate");
    EP_STATIC_ASSERT(kNumChannels * kNumScaleoutRanks * sizeof(int64_t) <= uccl_gin::kAtomicOffMask + 1,
                     "UCCL-GIN compact tail buffer must fit the ordered atomic offset field");
    const auto gin = handle::UCCLGin(nccl_dev_comm, nccl_window, uccl_gin_resources, qp_idx, sharing_mode);
    constexpr int kUCCLGinAtomicTailWords = kNumChannels * kNumScaleoutRanks;

#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
    auto* dispatch_profile_counters =
        reinterpret_cast<uint64_t*>(uccl_gin_resources.atomic_tail_base) +
        kUCCLGinAtomicTailWords;
#endif
#if defined(DEEPEP_UCCL_GIN_CHUNK_PROFILE)
    constexpr int kUCCLGinChunkProfileOffsetWords =
        kUCCLGinAtomicTailWords
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
        + uccl_gin::kDispatchClockNumCounters
#endif
        ;
    auto* dispatch_chunk_counters =
        reinterpret_cast<uint64_t*>(uccl_gin_resources.atomic_tail_base) +
        kUCCLGinChunkProfileOffsetWords;
#endif

    // The receiver forward warp reads compact software-atomic tails from the
    // host-mapped atomic buffer. Clear the compact slots at kernel start because
    // they are not part of DeepEP's normal workspace cleanup.
    for (int i = sm_idx * kNumThreads + thread_idx; i < kNumChannels * kNumScaleoutRanks; i += kNumSMs * kNumThreads)
        reinterpret_cast<int64_t*>(uccl_gin_resources.atomic_tail_base)[i] = 0;
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
    for (int i = sm_idx * kNumThreads + thread_idx; i < uccl_gin::kDispatchClockNumCounters; i += kNumSMs * kNumThreads)
        dispatch_profile_counters[i] = 0;
#endif
#if defined(DEEPEP_UCCL_GIN_CHUNK_PROFILE)
    for (int i = sm_idx * kNumThreads + thread_idx; i < uccl_gin::kDispatchChunkNumCounters; i += kNumSMs * kNumThreads)
        dispatch_chunk_counters[i] = 0;
#endif
    __threadfence_system();
    cooperative_groups::this_grid().sync();
#else
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
#endif

    // Global parallel barriers for scale-out subteam and scale-up subteam
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, false, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // The golden layout during the whole process for both scale-out and forward warps
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
    uint64_t profile_local[uccl_gin::kDispatchClockNumCounters] = {};
    uint64_t profile_scaleout_d2h_max = 0;
    uint64_t profile_forward_tail_wait_max = 0;
    uint64_t profile_forward_load_max = 0;
    const auto profile_add = [&](const uccl_gin::DispatchClockCounter& counter, const uint64_t& value) {
        profile_local[static_cast<uint32_t>(counter)] += value;
    };
    const auto profile_inc = [&](const uccl_gin::DispatchClockCounter& counter) {
        profile_local[static_cast<uint32_t>(counter)] += 1;
    };
    const auto profile_max = [&](const uccl_gin::DispatchClockCounter& counter,
                                 const uint64_t& cycles,
                                 const uint64_t& detail) {
        const auto packed = uccl_gin::dispatch_clock_pack_max(cycles, detail);
        if (counter == uccl_gin::kDispatchClockScaleoutD2HMaxPacked)
            profile_scaleout_d2h_max = packed > profile_scaleout_d2h_max ? packed : profile_scaleout_d2h_max;
        else if (counter == uccl_gin::kDispatchClockForwardTailWaitMaxPacked)
            profile_forward_tail_wait_max = packed > profile_forward_tail_wait_max ? packed : profile_forward_tail_wait_max;
        else if (counter == uccl_gin::kDispatchClockForwardLoadMaxPacked)
            profile_forward_load_max = packed > profile_forward_load_max ? packed : profile_forward_load_max;
    };
#endif
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE)
    uint64_t sample_push_cycles = 0;
    uint64_t sample_push_events = 0;
    uint64_t sample_initial_inflight_sum = 0;
    uint64_t sample_initial_inflight_max = 0;
    uint64_t sample_initial_at_cap = 0;
    uint64_t sample_forward_tail_wait_cycles = 0;
    uint64_t sample_forward_tail_wait_events = 0;
#endif

    // All the buffers
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
#ifdef DEEPEP_USE_UCCL_GIN
    constexpr int kNumCompactSendTokens = kNumChannels * kNumMaxTokensPerChannel;
    EP_STATIC_ASSERT(kNumScaleoutRanks == 2, "UCCL-GIN compact dispatch is currently specialized for EP8x2");
    EP_STATIC_ASSERT(kNumCompactSendTokens + kNumScaleoutRanks * kNumChannels * kNumMaxTokensPerChannel <=
                     kNumMaxTokensPerRank + kNumScaleoutRanks * (kNumMaxTokensPerRank + kNumMaxChannels),
                     "UCCL-GIN compact send padding must fit the original DeepEP V2 dispatch buffer size");
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, kNumCompactSendTokens, scaleup_buffer.get_buffer_end_ptr());
#else
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());
#endif
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks, kNumChannels * kNumMaxTokensPerChannel, scaleout_send_buffer.get_buffer_end_ptr());

    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Do full-grid reduction
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // Encode and write into the send buffer
                        workspace_layout.get_scaleout_rank_expert_count_ptr<true>()[i] =
                            math::encode_decode_positive<int>(status & 0xffffffffll);

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
#ifdef DEEPEP_USE_UCCL_GIN
            __threadfence_system();
#endif
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Issue scaleout writes to peers
            EP_STATIC_ASSERT(kReuseSlotIndices or kNumScaleoutRanks <= kNumNotifyThreads,
                             "kNumScaleoutRanks must be less than kNumNotifyThreads");
            if (thread_idx < kNumScaleoutRanks) {
                const auto dst_scaleout_rank_idx = thread_idx;
#ifdef DEEPEP_USE_UCCL_GIN
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
                    ncclGinOptFlagsAggregateRequests,
                    ncclGin_None(),
                    thread_idx);
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx,
                    0,
                    ncclGin_None(),
                    thread_idx);
#else
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
                    ncclGinOptFlagsAggregateRequests);
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);
#endif
            }
            __syncwarp();

            // Util functions to get metadata from scale-out peers
            // NOTES: this is correct as RDMA operations has a minimum write granularity of 1024 bytes (a whole integer write is atomic)
            const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {
                int count = 0;
                #pragma unroll
                for (int j = 0; j < kNumScaleoutRanks; ++ j) {
                    const auto ptr = get_ptr_func(j);
                    int decoded;
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                        decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                        if (math::is_decoded_positive_ready(decoded))
                            return true;

                        if (is_last_check) {
                            printf("DeepEP hybrid notify (scale-out %s reduction) timeout, "
                                   "scale-out: %d, scale-up: %d, "
                                   "thread: %d, wait scale-out: %d, decoded: %d\n",
                                   is_expert_reduction ? "expert" : "rank",
                                   scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,
                                   decoded);
                        }
                        return false;
                    });

                    // Add and clean for next usages
                    count += decoded, *ptr = 0;
                }
                return count;
            };

            // Write into all scale-up peers' rank-level counters
            #pragma unroll
            for (int i = thread_idx; i < kNumScaleupRanks; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                });

                // Write into the remote scale-up peer
                const int64_t counter = (static_cast<int64_t>(kNumScaleupRanks) << 32ll) | count;
                gin.put_value<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                    counter, i);
            }
            __syncwarp();

            // Atomic add into all scale-up peers' expert-level counters
            #pragma unroll
            for (int i = thread_idx; i < kNumExpertsPerScaleout; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                }, true);

                // Write into the remote scale-up peer
                const int64_t counter = (1ll << 32ll) | count;
                const auto dst_scaleup_rank_idx = i / kNumExpertsPerRank;
                const auto expert_idx_in_dst_rank = i % kNumExpertsPerRank;
                gin.red_add_rel<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                    counter, dst_scaleup_rank_idx);
            }
            // There are shared memory reads above, a barrier is necessary
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // NOTES: from now on, the `rank` and `expert`s size change into the local size
            expert_count = rank_expert_count + kNumScaleupRanks;

            // Wait local counters to be ready
            // NOTES: here we only care the prefix sum by scale-up peers (used for later epilogue), not all ranks
            EP_STATIC_ASSERT(kNumNotifyWarps == 0 or kNumScaleupRanks + kNumExpertsPerRank <= kNumNotifyWarps * 32,
                             "Insufficient notify threads");
            comm::timeout_while<kNumTimeoutCycles>(thread_idx < kNumScaleupRanks + kNumExpertsPerRank,
                [&](const bool& is_last_check) {
                const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + thread_idx);
                if ((status >> 32ull) == kNumScaleupRanks) {
                    // Clean GPU workspace and write into host workspace
                    const auto count = static_cast<int>(status & 0xffffffffll);
                    const auto aligned_count = math::align<int>(
                        count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);

                    workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] = 0;
                    if constexpr (kDoCPUSync) {
                        host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] =
                            math::encode_decode_positive(aligned_count);
                    }

                    // Update statistics counters
                    if (cumulative_local_expert_recv_stats != nullptr and thread_idx >= kNumScaleupRanks)
                        atomicAdd(cumulative_local_expert_recv_stats + (thread_idx - kNumScaleupRanks), count);

                    // Save for later prefix sum calculation
                    rank_expert_count[thread_idx] = aligned_count;
                    return true;
                }

                if (is_last_check) {
                    printf("DeepEP hybrid notify (scale-up reduction) timeout,"
                           "scale-out: %d/%d, scale-up: %d/%d, "
                           "thread: %d, status: %d | %d, expected: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                           static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumScaleupRanks);
                }
                return false;
            });
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Do prefix sum by the warps of the first SM
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumScaleupRanks, 0);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
#ifdef DEEPEP_USE_UCCL_GIN
        const auto scaleout_send_channel_buffer = scaleout_send_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
        constexpr int kUCCLGinCompactChunkTokens = 4;
        EP_STATIC_ASSERT(kUCCLGinCompactChunkTokens <= 0xFF,
                         "UCCL-GIN piggyback count delta must fit TransferCmd::atomic_val");
        EP_STATIC_ASSERT(handle::kUCCLGinTailFinishDelta <= uccl_gin::kAtomicValueMax,
                         "UCCL-GIN finish delta must fit the ordered atomic immediate");
        const int remote_scaleout_rank_idx = scaleout_rank_idx ^ 1;
        int compact_batch_first_slot = 0;
        int compact_batch_count = 0;

        const auto flush_compact_remote_batch = [&](
                const bool& finish_flag = false,
                const uint32_t& flush_reason = uccl_gin::kDispatchChunkFlushReasonFinish) {
            if (compact_batch_count > 0 and lane_idx == remote_scaleout_rank_idx) {
#if defined(DEEPEP_UCCL_GIN_CHUNK_PROFILE)
                const auto chunk_count = static_cast<uint32_t>(compact_batch_count);
                uccl_gin::dispatch_chunk_add(dispatch_chunk_counters, uccl_gin::kDispatchChunkChunks, 1);
                uccl_gin::dispatch_chunk_add(dispatch_chunk_counters, uccl_gin::kDispatchChunkTokens, chunk_count);
                uccl_gin::dispatch_chunk_add(dispatch_chunk_counters,
                                             uccl_gin::dispatch_chunk_size_bin(chunk_count), 1);
                uccl_gin::dispatch_chunk_add(dispatch_chunk_counters,
                                             uccl_gin::dispatch_chunk_reason_counter(flush_reason), 1);
#endif
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_start = clock64();
#endif
#if defined(DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE)
                uint64_t sample_start = 0;
                if (channel_idx % 16 == 0) {
                    auto* sample_queue = gin.lane(channel_idx);
#ifdef USE_MSCCLPP_FIFO_BACKEND
                    const uint64_t initial_head = mscclpp::atomicLoad<uint64_t, mscclpp::scopeDevice>(
                        sample_queue->fifo.head, mscclpp::memoryOrderRelaxed);
                    const uint64_t initial_inflight = initial_head - *sample_queue->fifo.tailCache;
                    const uint64_t inflight_limit = static_cast<uint64_t>(sample_queue->fifo.size);
#else
                    const uint64_t initial_inflight = sample_queue->head() - sample_queue->tail();
                    const uint64_t inflight_limit = static_cast<uint64_t>(kUCCLGinMaxInflightNormal);
#endif
                    sample_initial_inflight_sum += initial_inflight;
                    sample_initial_inflight_max = initial_inflight > sample_initial_inflight_max ?
                        initial_inflight : sample_initial_inflight_max;
                    sample_initial_at_cap += initial_inflight >= inflight_limit;
                    sample_start = clock64();
                }
#endif
                gin.rail_put_tail_add(
                    scaleout_recv_buffer.get_token_buffer(compact_batch_first_slot).get_base_ptr(),
                    scaleout_send_channel_buffer.get_token_buffer(compact_batch_first_slot).get_base_ptr(),
                    compact_batch_count * tma_buffer.get_num_bytes<false>(),
                    remote_scaleout_rank_idx,
                    channel_idx,
                    scaleout_rank_idx,
                    compact_batch_count,
                    channel_idx);
#if defined(DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE)
                if (channel_idx % 16 == 0) {
                    sample_push_cycles += clock64() - sample_start;
                    sample_push_events += 1;
                }
#endif
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_cycles = clock64() - profile_start;
                profile_add(uccl_gin::kDispatchClockScaleoutD2HCycles, profile_cycles);
                profile_inc(uccl_gin::kDispatchClockScaleoutD2HEvents);
                profile_max(
                    uccl_gin::kDispatchClockScaleoutD2HMaxPacked,
                    profile_cycles,
                    uccl_gin::dispatch_clock_detail(
                        static_cast<uint32_t>(channel_idx),
                        uccl_gin_resources.num_queues == 0 ? 0 :
                            gin.lane_index(channel_idx)));
#endif
            }
            if (finish_flag and lane_idx == remote_scaleout_rank_idx) {
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_start = clock64();
#endif
                gin.rail_tail_add(channel_idx, scaleout_rank_idx, remote_scaleout_rank_idx,
                                  0, true, channel_idx);
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                profile_add(uccl_gin::kDispatchClockScaleoutTailCycles, clock64() - profile_start);
                profile_inc(uccl_gin::kDispatchClockScaleoutTailEvents);
#endif
            }
            compact_batch_count = 0;
            __syncwarp();
        };
#endif

        // Channel metadata maintenance
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid number of scale-out ranks");
        int stored_scaleout_tail = 0, stored_old_scaleout_tail = 0;
        const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
            if (lane_idx < kNumScaleoutRanks and
                (stored_scaleout_tail >= stored_old_scaleout_tail + kScaleoutUpdateInterval or finish_flag)) {
                const auto tail_delta = stored_scaleout_tail - stored_old_scaleout_tail;
#ifdef DEEPEP_USE_UCCL_GIN
                if (lane_idx == scaleout_rank_idx) {
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                    const auto profile_start = clock64();
#endif
                    gin.rail_tail_add(channel_idx, scaleout_rank_idx, lane_idx, tail_delta, finish_flag, channel_idx);
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                    profile_add(uccl_gin::kDispatchClockScaleoutTailCycles, clock64() - profile_start);
                    profile_inc(uccl_gin::kDispatchClockScaleoutTailEvents);
#endif
                    stored_old_scaleout_tail = stored_scaleout_tail;
                }
#else
                const auto signaled_tail = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
                const auto ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx);
                const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);

                // NOTES: the "release" scope will be `sys` for the local rank (we may involve NVLink so not `gpu`)
                // For RDMA requests, "release" is ensured by "atomic"
                gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
                stored_old_scaleout_tail = stored_scaleout_tail;
#endif
            }
            __syncwarp();
        };

        // Preload next token
        const auto preload_next_token = [&](const int& token_idx) {
            if (token_idx >= num_tokens)
                return;

#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
            const auto profile_start = lane_idx == 0 ? clock64() : 0;
#endif
            // Issue TMA load
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF `cp.async`
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
            if (lane_idx == 0) {
                profile_add(uccl_gin::kDispatchClockScaleoutPreloadCycles, clock64() - profile_start);
                profile_inc(uccl_gin::kDispatchClockScaleoutPreloadEvents);
            }
#endif
        };

        // Iterate all tokens
        preload_next_token(channel_idx);
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_scaleout_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerScaleout : -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
#ifdef DEEPEP_USE_UCCL_GIN
            const int compact_remote_slot_idx = ptx::exchange(stored_scaleout_tail, remote_scaleout_rank_idx);
#endif
            const auto stored_old_slot_idx = ptx::exchange(
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)
                stored_dst_slot_idx = stored_old_slot_idx;

            // Update scale-out tail
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // Wait TMA arrival and issue the TMA store into send buffer
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

#ifndef DEEPEP_USE_UCCL_GIN
                // So if no ranks will go by RDMA, we skip the send buffer stores
                if (scaleout_rank_mask ^ (1 << scaleout_rank_idx)) {
                    ptx::tma_store_1d(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                      tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                }
#endif
            }
            __syncwarp();

#ifdef DEEPEP_USE_UCCL_GIN
            const bool has_remote_scaleout_token = ((scaleout_rank_mask >> remote_scaleout_rank_idx) & 1) != 0;
            if (lane_idx == remote_scaleout_rank_idx and has_remote_scaleout_token) {
                EP_DEVICE_ASSERT(compact_remote_slot_idx >= 0 and compact_remote_slot_idx < kNumMaxTokensPerChannel);
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_start = clock64();
#endif
                ptx::tma_store_1d(scaleout_send_channel_buffer.get_token_buffer(compact_remote_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                profile_add(uccl_gin::kDispatchClockScaleoutCompactStoreCycles, clock64() - profile_start);
                profile_inc(uccl_gin::kDispatchClockScaleoutCompactStoreEvents);
                profile_inc(uccl_gin::kDispatchClockScaleoutRemoteTokens);
#endif
            }
            __syncwarp();
#endif

            // Local rank can be bypassed
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_start = clock64();
#endif
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                profile_add(uccl_gin::kDispatchClockScaleoutLocalStoreCycles, clock64() - profile_start);
                profile_inc(uccl_gin::kDispatchClockScaleoutLocalStoreEvents);
                profile_inc(uccl_gin::kDispatchClockScaleoutLocalTokens);
#endif
            }
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
            const auto profile_store_wait_start = lane_idx == 0 ? clock64() : 0;
#endif
            ptx::tma_store_commit();
            ptx::tma_store_wait();
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
            if (lane_idx == 0) {
                profile_add(uccl_gin::kDispatchClockScaleoutStoreWaitCycles, clock64() - profile_store_wait_start);
                profile_inc(uccl_gin::kDispatchClockScaleoutStoreWaitEvents);
            }
#endif
            __syncwarp();

            // Preload the next token (overlapping with the IBGDA issues)
            preload_next_token(token_idx + kNumChannels);

            // Issue IBGDA requests
#ifdef DEEPEP_USE_UCCL_GIN
            if (has_remote_scaleout_token) {
                if (compact_batch_count == 0) {
                    compact_batch_first_slot = compact_remote_slot_idx;
                } else if (compact_batch_first_slot + compact_batch_count != compact_remote_slot_idx) {
                    flush_compact_remote_batch(false, uccl_gin::kDispatchChunkFlushReasonNonContig);
                    compact_batch_first_slot = compact_remote_slot_idx;
                }
                ++compact_batch_count;
                if (compact_batch_count >= kUCCLGinCompactChunkTokens)
                    flush_compact_remote_batch(false, uccl_gin::kDispatchChunkFlushReasonFull);
            }
#else
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
                gin.put<ncclTeamTagRail>(
                        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
                        tma_buffer.get_num_bytes<false>(),
                        stored_dst_scaleout_rank_idx,
                        ncclGinOptFlagsAggregateRequests);
            }
#endif
            __syncwarp();

            // Issue scale-out tail update
            update_scaleout_tail();
        }

        // Flush unflushed tails
#ifdef DEEPEP_USE_UCCL_GIN
        flush_compact_remote_batch(true, uccl_gin::kDispatchChunkFlushReasonFinish);
#endif
        update_scaleout_tail(true);
    } else {
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Shape of `dst_buffer_slot_idx`: `[kNumChannels, kNumScaleoutRanks, kNumMaxTokensPerChannel, kNumTopk]`
        dst_buffer_slot_idx += channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel * kNumTopk);

        // Transform linked list index
        const auto transform_linked_list_idx = [=](const int& idx) {
            constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;
            return channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                idx * kNumScaleupRanks + scaleup_rank_idx;
        };

        // Forward tokens from scale-out ranks
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Too many scale-out ranks");
        int num_tokens_processed = 0;
        int stored_scaleout_old_tail_idx = 0;
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};
        int stored_finish_flag = lane_idx >= kNumScaleoutRanks;
        int stored_scaleout_tail_idx = 0;
        int last_forward_src_token_global_idx = -1;
        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;
        uint32_t wip_mask;
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
            // Pick next rank in round-robin
            const auto offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks;
            const auto hi_mask = (wip_mask >> offset) << offset;
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);

            // Wait for this rank to have data (or finish)
#if defined(DEEPEP_USE_UCCL_GIN) && \
    (defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE) || \
     defined(DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE))
            const auto profile_tail_wait_start = lane_idx == 0 ? clock64() : 0;
#endif
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
            bool profile_tail_fresh_recorded = false;
            // Discriminator: replicate the timeout_while first check WITHOUT
            // re-reading the tail. If the chosen src rank already has data/finish
            // visible from the previous round, the forward warp is NOT starved
            // (delivery kept up); otherwise it must stall waiting for the tail.
            uint32_t profile_arrived_or_finished =
                stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
            const bool profile_tail_ready =
                ptx::exchange(profile_arrived_or_finished, recv_scaleout_rank_idx);
#endif
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                const uint32_t arrived_or_finished =
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
                    return true;

                // Timeout
                if (is_last_check) {
                    if (lane_idx < kNumScaleoutRanks) {
                        printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                               "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d)\n",
                               scaleout_rank_idx, scaleup_rank_idx,
                               channel_idx, lane_idx, stored_scaleout_old_tail_idx,
                               stored_finish_flag, stored_scaleout_tail_idx);
                    }
                    return false;
                }

                // Read new signaled tails
                if (lane_idx < kNumScaleoutRanks) {
#ifdef DEEPEP_USE_UCCL_GIN
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        gin.rail_tail_ptr(channel_idx, lane_idx));
                    gin.decode_rail_tail(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
#else
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
                    math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
#endif
                }
                __syncwarp();
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                if (not profile_tail_ready and not profile_tail_fresh_recorded) {
                    const uint32_t fresh_arrived_or_finished =
                        stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                    const uint32_t fresh_ready_mask = ptx::gather(fresh_arrived_or_finished);
                    if (lane_idx == 0) {
                        constexpr uint32_t valid_scaleout_mask =
                            kNumScaleoutRanks == 32 ? 0xffffffffu : ((1u << kNumScaleoutRanks) - 1u);
                        const uint32_t valid_fresh_ready_mask = fresh_ready_mask & valid_scaleout_mask;
                        const uint32_t selected_mask = 1u << recv_scaleout_rank_idx;
                        if (valid_fresh_ready_mask & selected_mask) {
                            profile_inc(uccl_gin::kDispatchClockForwardTailFreshSelectedReadyEvents);
                        } else if (valid_fresh_ready_mask & ~selected_mask) {
                            profile_inc(uccl_gin::kDispatchClockForwardTailFreshOtherReadyEvents);
                        } else {
                            profile_inc(uccl_gin::kDispatchClockForwardTailFreshNoReadyEvents);
                        }
                    }
                    profile_tail_fresh_recorded = true;
                }
#endif
                return false;
            });
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE)
            if (lane_idx == 0 and channel_idx % 16 == 0) {
                sample_forward_tail_wait_cycles += clock64() - profile_tail_wait_start;
                sample_forward_tail_wait_events += 1;
            }
#endif
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
            if (lane_idx == 0) {
                const auto profile_cycles = clock64() - profile_tail_wait_start;
                profile_add(uccl_gin::kDispatchClockForwardTailWaitCycles, profile_cycles);
                profile_inc(uccl_gin::kDispatchClockForwardTailWaitEvents);
                if (profile_tail_ready) {
                    profile_inc(uccl_gin::kDispatchClockForwardTailReadyEvents);
                } else {
                    profile_inc(uccl_gin::kDispatchClockForwardTailStallEvents);
                    profile_add(uccl_gin::kDispatchClockForwardTailStallCycles, profile_cycles);
                }
                profile_max(
                    uccl_gin::kDispatchClockForwardTailWaitMaxPacked,
                    profile_cycles,
                    uccl_gin::dispatch_clock_detail(
                        static_cast<uint32_t>(channel_idx),
                        static_cast<uint32_t>(recv_scaleout_rank_idx)));
            }
#endif

            // Process one chunk from the current rank
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);
            const auto end_slot_idx = std::min(
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + kNumSlotsPerForwardChunk
            );
            if (lane_idx == recv_scaleout_rank_idx)
                stored_scaleout_old_tail_idx = end_slot_idx;

            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);
            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);

#ifdef DEEPEP_USE_UCCL_GIN
                // EFA/SRD completion ordering alone is not a strong receiver-side
                // payload-ready proof.  Keep the original UCCL/EP philosophy:
                // the tail publishes the available slot range, while each slot's
                // native V2 metadata proves that the payload for this
                // (channel, source-rank) stream is visible before the forwarder
                // consumes it.
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_meta_wait_start = lane_idx == 0 ? clock64() : 0;
#endif
                comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                    const auto expected_rank_idx = recv_scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;
                    const auto old_src_token_idx = ptx::exchange(last_forward_src_token_global_idx, recv_scaleout_rank_idx);
                    const auto observed_src_token_idx = ptx::ld_acquire_sys<int>(
                        token_buffer.get_src_token_global_idx_ptr());
                    const bool ready =
                        observed_src_token_idx / kNumMaxTokensPerRank == expected_rank_idx and
                        observed_src_token_idx > old_src_token_idx;
                    if (ready) {
                        if (lane_idx == recv_scaleout_rank_idx)
                            last_forward_src_token_global_idx = observed_src_token_idx;
                        return true;
                    }
                    if (is_last_check and lane_idx == recv_scaleout_rank_idx) {
                        printf("DeepEP UCCL-GIN ready metadata timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, src scale-out: %d, slot: %d, observed: %d, prev: %d, expected rank: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, recv_scaleout_rank_idx, slot_idx, observed_src_token_idx,
                               old_src_token_idx, expected_rank_idx);
                    }
                    return false;
                });
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                if (lane_idx == 0) {
                    profile_add(uccl_gin::kDispatchClockForwardMetaWaitCycles, clock64() - profile_meta_wait_start);
                    profile_inc(uccl_gin::kDispatchClockForwardMetaWaitEvents);
                }
#endif
                __syncwarp();
#endif

                // Wait TMA arrival
                ptx::tma_store_wait();
                __syncwarp();

                // TMA load into shared memory
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                const auto profile_forward_load_start = lane_idx == 0 ? clock64() : 0;
#endif
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                if (lane_idx == 0) {
                    const auto profile_cycles = clock64() - profile_forward_load_start;
                    profile_add(uccl_gin::kDispatchClockForwardLoadCycles, profile_cycles);
                    profile_inc(uccl_gin::kDispatchClockForwardLoadEvents);
                    profile_max(
                        uccl_gin::kDispatchClockForwardLoadMaxPacked,
                        profile_cycles,
                        uccl_gin::dispatch_clock_detail(
                            static_cast<uint32_t>(channel_idx),
                            static_cast<uint32_t>(slot_idx)));
                }
#endif
                __syncwarp();

                // Read top-k indices
                EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                int stored_dst_scaleup_rank_idx = -1;
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;
                dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout;
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < kNumExpertsPerScaleout ?
                    dst_expert_idx / kNumExpertsPerRank : -1;

                // Write the per-scaleup channel index for this token
                int linked_list_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                    const auto exchanged = ptx::exchange(
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                if (not kReuseSlotIndices and lane_idx < kNumTopk) {
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Deduplicate for scale-up ranks
                int stored_dst_slot_idx = -1;
                const auto dst_slot_idx_ptr = dst_buffer_slot_idx +
                    recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + slot_idx * kNumTopk;
                if constexpr (kReuseSlotIndices) {
                    if (lane_idx < kNumTopk)
                        stored_dst_slot_idx = __ldg(dst_slot_idx_ptr + lane_idx);
                } else {
                    // Deduplicate for NVLink ranks
                    if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                }
                __syncwarp();

                // Issue TMAs
                if (stored_dst_slot_idx >= 0) {
                    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        stored_dst_scaleup_rank_idx);
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                    const auto profile_scaleup_store_start = clock64();
#endif
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    ptx::tma_store_commit();
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                    profile_add(uccl_gin::kDispatchClockForwardScaleupStoreCycles, clock64() - profile_scaleup_store_start);
                    profile_inc(uccl_gin::kDispatchClockForwardScaleupStoreEvents);
#endif
                }
                __syncwarp();

                // Add per-scale-up counter
                EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
                const auto scaleup_send_mask = ptx::reduce_or(
                    stored_dst_scaleup_rank_idx >= 0 ?
                    (mask_t(1) << stored_dst_scaleup_rank_idx) : mask_t(0));
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_scaleup_send_counters[j] += (scaleup_send_mask >> (j * 32 + lane_idx)) & 1;

                // Record metadata at forward
                if constexpr (not kReuseSlotIndices) {
                    EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                    const auto metadata_ptr = token_metadata_at_forward +
                        num_tokens_processed * kNumForwardMetadataDims;

                    // Source token index and last token index flag
                    if (ptx::elect_one_sync()) {
                        metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                        metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                    }

                    // Second, original top-k indices and destination slots
                    if (lane_idx < kNumTopk) {
                        metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                        metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                        dst_slot_idx_ptr[lane_idx] = stored_dst_slot_idx;
                    }
                }
                num_tokens_processed += 1;
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
                if (lane_idx == 0)
                    profile_inc(uccl_gin::kDispatchClockForwardTokens);
#endif
                __syncwarp();
            }
        }

        // Assign the source token index part of the metadata into `-1` as an ending mark
        if (not kReuseSlotIndices and ptx::elect_one_sync())
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
        __syncwarp();

        // Update linked list's ending position
        if constexpr (not kReuseSlotIndices) {
            const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                    ptx::st_relaxed_sys(
                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),
                        transform_linked_list_idx(stored_scaleup_send_counters[i]));
                }
            }
        }
        __syncwarp();

        // Clean tails for next usages
        if (lane_idx < kNumScaleoutRanks) {
#ifdef DEEPEP_USE_UCCL_GIN
            *gin.rail_tail_ptr(channel_idx, lane_idx) = 0;
#else
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
#endif
        }
#ifdef DEEPEP_USE_UCCL_GIN
        __threadfence_system();
#endif
        __syncwarp();
    }

#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE)
    if (warp_idx >= kNumNotifyWarps and
        warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        const int sample_channel =
            sm_idx * kNumChannelsPerSM + (warp_idx - kNumNotifyWarps);
        if (sample_channel % 16 == 0 and lane_idx == (scaleout_rank_idx ^ 1)) {
            printf("UCCL_GIN_DISPATCH_SAMPLE_PUSH rank=%d channel=%d queue=%u "
                   "events=%llu cycles=%llu initial_inflight_sum=%llu "
                   "initial_inflight_max=%llu initial_at_cap=%llu\n",
                   rank_idx, sample_channel, gin.lane_index(sample_channel),
                   static_cast<unsigned long long>(sample_push_events),
                   static_cast<unsigned long long>(sample_push_cycles),
                   static_cast<unsigned long long>(sample_initial_inflight_sum),
                   static_cast<unsigned long long>(sample_initial_inflight_max),
                   static_cast<unsigned long long>(sample_initial_at_cap));
        }
    } else if (warp_idx >= kNumNotifyWarps + kNumScaleoutWarps) {
        const int sample_channel =
            sm_idx * kNumChannelsPerSM +
            (warp_idx - kNumNotifyWarps - kNumScaleoutWarps);
        if (sample_channel % 16 == 0 and lane_idx == 0) {
            printf("UCCL_GIN_DISPATCH_SAMPLE_FORWARD rank=%d channel=%d "
                   "events=%llu tail_wait_cycles=%llu\n",
                   rank_idx, sample_channel,
                   static_cast<unsigned long long>(sample_forward_tail_wait_events),
                   static_cast<unsigned long long>(sample_forward_tail_wait_cycles));
        }
    }
#endif

#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
    for (int i = 0; i < uccl_gin::kDispatchClockNumCounters; ++i) {
        if (profile_local[i] != 0)
            uccl_gin::dispatch_clock_add(dispatch_profile_counters, i, profile_local[i]);
    }
    if (profile_scaleout_d2h_max != 0)
        uccl_gin::dispatch_clock_max(dispatch_profile_counters, uccl_gin::kDispatchClockScaleoutD2HMaxPacked,
                                     profile_scaleout_d2h_max >> 24,
                                     profile_scaleout_d2h_max & ((1ull << 24) - 1));
    if (profile_forward_tail_wait_max != 0)
        uccl_gin::dispatch_clock_max(dispatch_profile_counters, uccl_gin::kDispatchClockForwardTailWaitMaxPacked,
                                     profile_forward_tail_wait_max >> 24,
                                     profile_forward_tail_wait_max & ((1ull << 24) - 1));
    if (profile_forward_load_max != 0)
        uccl_gin::dispatch_clock_max(dispatch_profile_counters, uccl_gin::kDispatchClockForwardLoadMaxPacked,
                                     profile_forward_load_max >> 24,
                                     profile_forward_load_max & ((1ull << 24) - 1));
#endif

    // Scale-up barrier to ensure data arrival
    // As scale-out tokens have already been consumed by forwarders, no need to do scale-out barrier again
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag1, true, true, false>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, /* do not scale-out */ false, true);

#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
    if (sm_idx == 0 and thread_idx == 0) {
        const auto* c = dispatch_profile_counters;
        printf("UCCL_GIN_DISPATCH_CLOCK rank=%d "
               "scaleout_preload_cycles=%llu scaleout_preload_events=%llu "
               "scaleout_compact_store_cycles=%llu scaleout_compact_store_events=%llu "
               "scaleout_local_store_cycles=%llu scaleout_local_store_events=%llu "
               "scaleout_store_wait_cycles=%llu scaleout_store_wait_events=%llu "
               "scaleout_d2h_cycles=%llu scaleout_d2h_events=%llu "
               "scaleout_tail_cycles=%llu scaleout_tail_events=%llu "
               "forward_tail_wait_cycles=%llu forward_tail_wait_events=%llu "
               "forward_meta_wait_cycles=%llu forward_meta_wait_events=%llu "
               "forward_load_cycles=%llu forward_load_events=%llu "
               "forward_scaleup_store_cycles=%llu forward_scaleup_store_events=%llu "
               "forward_tokens=%llu scaleout_remote_tokens=%llu scaleout_local_tokens=%llu "
               "scaleout_d2h_max_packed=%llu forward_tail_wait_max_packed=%llu "
               "forward_load_max_packed=%llu "
               "forward_tail_ready_events=%llu forward_tail_stall_events=%llu "
               "forward_tail_stall_cycles=%llu\n",
               rank_idx,
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutPreloadCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutPreloadEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutCompactStoreCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutCompactStoreEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutLocalStoreCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutLocalStoreEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutStoreWaitCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutStoreWaitEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutD2HCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutD2HEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutTailCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutTailEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailWaitCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailWaitEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardMetaWaitCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardMetaWaitEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardLoadCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardLoadEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardScaleupStoreCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardScaleupStoreEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTokens]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutRemoteTokens]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutLocalTokens]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockScaleoutD2HMaxPacked]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailWaitMaxPacked]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardLoadMaxPacked]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailReadyEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailStallEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailStallCycles]));
        printf("UCCL_GIN_DISPATCH_CLOCK_FRESH rank=%d "
               "forward_tail_fresh_selected_ready_events=%llu "
               "forward_tail_fresh_other_ready_events=%llu "
               "forward_tail_fresh_no_ready_events=%llu\n",
               rank_idx,
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailFreshSelectedReadyEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailFreshOtherReadyEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchClockForwardTailFreshNoReadyEvents]));
    }
#endif

#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_CHUNK_PROFILE)
    if (sm_idx == 0 and thread_idx == 0) {
        const auto* c = dispatch_chunk_counters;
        printf("UCCL_GIN_CHUNK_PROFILE rank=%d chunks=%llu tokens=%llu "
               "bin_1=%llu bin_2=%llu bin_3_4=%llu bin_5_8=%llu "
               "bin_9_16=%llu bin_17_24=%llu bin_25_31=%llu bin_32=%llu "
               "bin_gt32=%llu flush_noncontig=%llu flush_full=%llu "
               "flush_finish=%llu\n",
               rank_idx,
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkChunks]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkTokens]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin1]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin2]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin3To4]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin5To8]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin9To16]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin17To24]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin25To31]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBin32]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkBinGt32]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkFlushNonContig]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkFlushFull]),
               static_cast<unsigned long long>(c[uccl_gin::kDispatchChunkFlushFinish]));
    }
#endif

    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Clean scale-up counters
    // All scale-out counters should be cleaned before
    EP_STATIC_ASSERT(kNumScaleupRanks <= kNumThreads, "Insufficient threads");
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumScaleupRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic
