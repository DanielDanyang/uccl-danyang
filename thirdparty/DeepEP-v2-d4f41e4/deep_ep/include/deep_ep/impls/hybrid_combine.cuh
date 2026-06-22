#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>

#ifdef DEEPEP_USE_UCCL_GIN
#include <cooperative_groups.h>
#include <uccl_gin/uccl_gin_handle.cuh>
#endif

namespace deep_ep::elastic {

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs,
          int kNumScaleupWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumScaleupUpdateInterval = 3,
          int kNumChannelsPerSM = kNumForwardWarps,
          int kNumChannels = kNumChannelsPerSM * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumWarps = kNumScaleupWarps + kNumForwardWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_combine_impl(nv_bfloat16* x,
                    float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward,
                    int* channel_linked_list,
                    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
#ifdef DEEPEP_USE_UCCL_GIN
                    const uccl_gin::UCCLGinResources uccl_gin_resources,
#endif
                    void* buffer, void* workspace,
                    const int scaleout_rank_idx, const int scaleup_rank_idx,
                    int num_reduced_tokens) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // Combine vector type selection
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token layouts
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // TMA buffers
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // All the buffer layouts
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank,
        scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx % kNumChannelsPerSM);
#ifdef DEEPEP_USE_UCCL_GIN
    const auto gin = handle::UCCLGin(
        nccl_dev_comm, nccl_window, uccl_gin_resources, qp_idx, sharing_mode);
    EP_STATIC_ASSERT(kNumChannels * kNumScaleoutRanks * sizeof(int64_t) <= uccl_gin::kAtomicOffMask + 1,
                     "UCCL-GIN combine tail buffer must fit the ordered atomic offset field");
    constexpr int kUCCLGinAtomicTailWords = kNumChannels * kNumScaleoutRanks;
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
    auto* combine_profile_counters =
        reinterpret_cast<uint64_t*>(uccl_gin_resources.atomic_tail_base) +
        kUCCLGinAtomicTailWords;
#endif
    // Combine reuses the dispatch compact (channel, source-rank) tail buffer.
    // The previous receiver kernel waits until every incoming finish has been
    // applied, then clears its slots before returning. Therefore no old remote
    // finish can race this reset; a host-side quiet here would only serialize
    // the transport. Keep the start reset as defensive initialization.
    for (int i = sm_idx * kNumThreads + thread_idx; i < kNumChannels * kNumScaleoutRanks; i += kNumSMs * kNumThreads)
        reinterpret_cast<int64_t*>(uccl_gin_resources.atomic_tail_base)[i] = 0;
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
    for (int i = sm_idx * kNumThreads + thread_idx; i < uccl_gin::kCombineProfileNumCounters; i += kNumSMs * kNumThreads)
        combine_profile_counters[i] = 0;
#endif
    __threadfence_system();
    cooperative_groups::this_grid().sync();
#else
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
#endif

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: this barrier needs a grid sync, as there are channel scale-up tail cleaning before
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // Different warp roles
    if (warp_idx < kNumScaleupWarps) {
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        // Shift into the right buffer if using rank layout
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Expanding mode must not be backward
        if constexpr (kUseExpandedLayout)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        int update_counter = 0;
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        const auto update_tails = [&](const bool& finish = false) {
            ++ update_counter;
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                ptx::tma_store_wait();
                __syncwarp();

                // Issue
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                update_counter = 0;
            }
            __syncwarp();
        };

        // Shape of `channel_linked_list`: `[kNumChannels, kNumMaxTokensPerChannel + 1, kNumScaleupRanks]`
        // Iterate until all scale-up peers finish
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1;
        while (true) {
            // Load token indices in the list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                stored_token_idx[i] = i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks ?
                    __ldg(channel_linked_list +
                          channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                          stored_ll_idx[i] * kNumScaleupRanks + j) : -1;
            }
            __syncwarp();

            // Check whether all ranks are finished
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                break;

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
            using mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;
            mask_t wip_mask = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                wip_mask |= static_cast<mask_t>(ptx::gather(stored_token_idx[j] >= 0)) << (j * 32);
            while (wip_mask) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;
                const auto hi_mask = (wip_mask >> start) << start;
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;

                // Exchange token index from the owning lane using static partition iteration
                int token_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                }
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);

                // Get source metadata and decide the destination buffer
                constexpr int kMetadataStride = 2 + kNumTopk;
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);
                auto token_buffer = [&]() {
                    if constexpr (kUseScaleupRankLayout) {
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));

                // Some checks
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                //  - expand + reduce
                //  - expand + send all
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    if (ptx::elect_one_sync()) {
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {
                    // Do local reduction
                    // Sort valid top-k indices to front
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                    const auto reduce_profile_start = clock64();
#endif
                    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                    if (ptx::elect_one_sync()) {
                        uccl_gin::combine_profile_add(
                            combine_profile_counters, uccl_gin::kCombineProfileReduceCycles,
                            clock64() - reduce_profile_start);
                        uccl_gin::combine_profile_add(
                            combine_profile_counters, uccl_gin::kCombineProfileReduceEvents, 1);
                    }
#endif
                    ptx::tma_store_fence();
                    __syncwarp();
                } else {
                    // No local reduction, send all data (expanded send)
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            continue;

                        if (ptx::elect_one_sync()) {
                            // Load
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            const auto dst_token_buffer = scaleup_buffer
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            ptx::tma_store_commit();
                        }
                        __syncwarp();
                    }
                }

                // Write top-k weights
                if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {
                    const float value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    if (no_local_reduce) {
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            update_tails();

            // Move linked list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
        }

        // Update for the unissued ones
        update_tails(true);
    } else {
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

        // Shift into the right buffer
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumScaleoutRanks * kNumMaxTokensPerChannel>(channel_idx);

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Overlap TMA stores and reduction
        int last_src_scaleout_rank_idx = -1;
        int last_is_token_last_in_chunk = 0;
        void* last_recv_token_buffer_ptr = nullptr;
        void* last_send_token_buffer_ptr = nullptr;
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE) && \
    !defined(DEEPEP_UCCL_GIN_COMBINE_CLOCK_ONLY)
        uint64_t profile_last_local_ptr = 0;
        uint64_t profile_last_remote_ptr = 0;
        int profile_last_dst = -1;
        uint32_t profile_run_length = 0;
        const auto profile_flush_run = [&]() {
            if (profile_run_length == 0)
                return;
            uccl_gin::combine_profile_add(
                combine_profile_counters, uccl_gin::kCombineProfileRuns, 1);
            uccl_gin::combine_profile_add(
                combine_profile_counters,
                uccl_gin::combine_profile_run_bin(profile_run_length), 1);
            profile_run_length = 0;
        };
        const auto profile_remote_put = [&](void* remote_ptr, void* local_ptr, const int& dst) {
            const auto remote = reinterpret_cast<uint64_t>(remote_ptr);
            const auto local = reinterpret_cast<uint64_t>(local_ptr);
            const auto bytes = static_cast<uint64_t>(token_layout.get_num_bytes<false>());
            uccl_gin::combine_profile_add(
                combine_profile_counters, uccl_gin::kCombineProfileRemotePuts, 1);
            if (profile_last_dst < 0) {
                profile_run_length = 1;
            } else {
                const bool same_dst = profile_last_dst == dst;
                const bool local_contiguous = profile_last_local_ptr + bytes == local;
                const bool remote_contiguous = profile_last_remote_ptr + bytes == remote;
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileTransitions, 1);
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileSameDstTransitions,
                    same_dst);
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileLocalContiguousTransitions,
                    local_contiguous);
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileRemoteContiguousTransitions,
                    remote_contiguous);
                const bool both_contiguous = same_dst and local_contiguous and remote_contiguous;
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileBothContiguousTransitions,
                    both_contiguous);
                if (both_contiguous) {
                    ++profile_run_length;
                } else {
                    profile_flush_run();
                    profile_run_length = 1;
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileBreakDst,
                        not same_dst);
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileBreakLocalGap,
                        not local_contiguous);
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileBreakRemoteGap,
                        not remote_contiguous);
                }
            }
            profile_last_dst = dst;
            profile_last_local_ptr = local;
            profile_last_remote_ptr = remote;
        };
#endif
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_REORDER_PROFILE)
        uint32_t reorder_dump_count = 0;
        const auto profile_reorder_put = [&](void* remote_ptr, void* local_ptr, const int& dst) {
            if (channel_idx != 0 or reorder_dump_count >= 256)
                return;
            const auto remote = reinterpret_cast<uint64_t>(remote_ptr);
            const auto local = reinterpret_cast<uint64_t>(local_ptr);
            printf("UCCL_GIN_COMBINE_REORDER_CAND rank=%d expanded=%d multiple_reduction=%d "
                   "channel=%d seq=%u dst=%d local_off=%llu remote_off=%llu bytes=%u\n",
                   scaleout_rank_idx, static_cast<int>(kUseExpandedLayout),
                   static_cast<int>(kAllowMultipleReduction), channel_idx,
                   reorder_dump_count, dst,
                   static_cast<unsigned long long>(local - uccl_gin_resources.window_base),
                   static_cast<unsigned long long>(remote - uccl_gin_resources.window_base),
                   static_cast<uint32_t>(token_layout.get_num_bytes<false>()));
            ++reorder_dump_count;
        };
#endif
        const auto flush_last_tma_and_issue_rdma = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                ptx::tma_store_wait();

                // Issue only if not local rank
                if (last_src_scaleout_rank_idx != scaleout_rank_idx) {
#ifdef DEEPEP_USE_UCCL_GIN
#if defined(DEEPEP_UCCL_GIN_COMBINE_REORDER_PROFILE)
                    profile_reorder_put(last_recv_token_buffer_ptr, last_send_token_buffer_ptr,
                                        last_src_scaleout_rank_idx);
#endif
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE) && !defined(DEEPEP_UCCL_GIN_COMBINE_CLOCK_ONLY)
                    profile_remote_put(last_recv_token_buffer_ptr, last_send_token_buffer_ptr,
                                       last_src_scaleout_rank_idx);
#endif
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                    const auto d2h_profile_start = clock64();
#endif
                    gin.put<ncclTeamTagRail>(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        token_layout.get_num_bytes<false>(),
                        last_src_scaleout_rank_idx,
                        last_is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests,
                        ncclGin_None(),
                        channel_idx
                    );
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                    const auto d2h_profile_cycles = clock64() - d2h_profile_start;
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileD2HCycles,
                        d2h_profile_cycles);
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileD2HEvents, 1);
                    uccl_gin::combine_profile_max(
                        combine_profile_counters, uccl_gin::kCombineProfileD2HMaxPacked,
                        d2h_profile_cycles,
                        uccl_gin::dispatch_clock_detail(channel_idx, channel_idx % uccl_gin_resources.num_queues));
#endif
#else
                    gin.put<ncclTeamTagRail>(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        token_layout.get_num_bytes<false>(),
                        last_src_scaleout_rank_idx,
                        last_is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                    );
#endif
                }
            }
            __syncwarp();
        };

        // Replay the dispatch
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        for (int i = 0; ; ++ i) {
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;

            // Scaleup rank mask
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
            const auto scaleup_mask = ptx::reduce_or(
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // Wait all tails to arrive
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
            const auto scaleup_wait_profile_start = clock64();
#endif
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                if (ptx::all(arrived))
                    return true;

                // Reload cached
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                return false;
            });
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
            if (ptx::elect_one_sync()) {
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileScaleupWaitCycles,
                    clock64() - scaleup_wait_profile_start);
                uccl_gin::combine_profile_add(
                    combine_profile_counters, uccl_gin::kCombineProfileScaleupWaitEvents, 1);
            }
#endif

            // Increase received count
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);
            
            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            continue;

                        // Issue TMA load, and wait
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        // Issue TMA store, and wait
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        // Issue IBGDA
                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {
#ifdef DEEPEP_USE_UCCL_GIN
#if defined(DEEPEP_UCCL_GIN_COMBINE_REORDER_PROFILE)
                            profile_reorder_put(recv_buffer_ptr, send_buffer_ptr,
                                                src_scaleout_rank_idx);
#endif
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE) && !defined(DEEPEP_UCCL_GIN_COMBINE_CLOCK_ONLY)
                            profile_remote_put(recv_buffer_ptr, send_buffer_ptr,
                                               src_scaleout_rank_idx);
#endif
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                            const auto d2h_profile_start = clock64();
#endif
                            gin.put<ncclTeamTagRail>(
                                recv_buffer_ptr,
                                send_buffer_ptr,
                                token_layout.get_num_bytes<false>(),
                                src_scaleout_rank_idx,
                                topk_valid_mask == 0 and is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests,
                                ncclGin_None(),
                                channel_idx
                            );
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                            const auto d2h_profile_cycles = clock64() - d2h_profile_start;
                            uccl_gin::combine_profile_add(
                                combine_profile_counters, uccl_gin::kCombineProfileD2HCycles,
                                d2h_profile_cycles);
                            uccl_gin::combine_profile_add(
                                combine_profile_counters, uccl_gin::kCombineProfileD2HEvents, 1);
                            uccl_gin::combine_profile_max(
                                combine_profile_counters, uccl_gin::kCombineProfileD2HMaxPacked,
                                d2h_profile_cycles,
                                uccl_gin::dispatch_clock_detail(channel_idx, channel_idx % uccl_gin_resources.num_queues));
#endif
#else
                            gin.put<ncclTeamTagRail>(
                                recv_buffer_ptr,
                                send_buffer_ptr,
                                token_layout.get_num_bytes<false>(),
                                src_scaleout_rank_idx,
                                topk_valid_mask == 0 and is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                            );
#endif
                        }
                    }
                }
                __syncwarp();
            } else {
                // NOTES: we must do deduplicate and only add once from one rank
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index
                int stored_src_buffer_idx = 0;
                if constexpr (kUseScaleupRankLayout) {
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }
                
                // Preprocess top-k indices
                int topk_slot_idx[kNumTokensInScaleupLayout];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                const auto reduce_profile_start = clock64();
#endif
                combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks)>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ [=](const int& slot_idx) {
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        flush_last_tma_and_issue_rdma();
                    }
                );
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
                if (ptx::elect_one_sync()) {
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileReduceCycles,
                        clock64() - reduce_profile_start);
                    uccl_gin::combine_profile_add(
                        combine_profile_counters, uccl_gin::kCombineProfileReduceEvents, 1);
                }
#endif

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                stored_src_buffer_idx = ptx::exchange(
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                if (not kUseExpandedLayout and stored_src_scaleup_rank_idx >= 0) {
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    .get_topk_weights_ptr()[lane_idx];
                }
                ptx::tma_store_fence();
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                int scaleout_recv_buffer_rank_idx;
                if constexpr (kUseScaleoutRankLayout) {
                    scaleout_recv_buffer_rank_idx = scaleout_rank_idx;
                } else {
                    const int src_topk_idx = ptx::get_master_lane_idx(ptx::gather(stored_src_scaleup_rank_idx >= 0));
                    scaleout_recv_buffer_rank_idx = src_topk_idx;
                }
                const auto recv_token_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_recv_buffer_rank_idx).get_token_buffer(src_token_idx);
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer.get_token_buffer(i);

                // Write into scale-out send buffer or local rank recv buffer bypass
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Record RDMA info to issue later
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_is_token_last_in_chunk = is_token_last_in_chunk;
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();
            }
        }

        // Issue the last RDMA
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_issue_rdma();
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE) && \
    !defined(DEEPEP_UCCL_GIN_COMBINE_CLOCK_ONLY)
        if (ptx::elect_one_sync())
            profile_flush_run();
#endif
#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_REORDER_PROFILE)
        (void)reorder_dump_count;
#endif

        // Clean scaleup tails
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();

        // Update, wait and clean
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
#ifdef DEEPEP_USE_UCCL_GIN
        // One-shot finish signal per (channel, scale-out rank). The payload puts
        // above are plain WRITEs tracked by the proxy on lane=channel_idx; the
        // finish ATOMIC posts on the same lane only after those WRITEs complete,
        // so no device-side flush is needed (proxy enforces payload-before-tail).
        if (lane_idx < kNumScaleoutRanks) {
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
            const auto finish_d2h_profile_start = clock64();
#endif
            gin.rail_tail_add(channel_idx, scaleout_rank_idx, lane_idx,
                              /*count_delta=*/0, /*finish=*/true, channel_idx);
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
            uccl_gin::combine_profile_add(
                combine_profile_counters, uccl_gin::kCombineProfileFinishD2HCycles,
                clock64() - finish_d2h_profile_start);
            uccl_gin::combine_profile_add(
                combine_profile_counters, uccl_gin::kCombineProfileFinishD2HEvents, 1);
#endif
        }
        __syncwarp();

        // Wait finish signal arrival from the corresponding source rank.
        if (lane_idx < kNumScaleoutRanks) {
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
            const auto finish_wait_profile_start = clock64();
#endif
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                int finish = 0, count = 0;
                gin.decode_rail_tail(
                    ptx::ld_acquire_sys<int64_t>(gin.rail_tail_ptr(channel_idx, lane_idx)),
                    finish, count);
                if (finish) {
                    // Clean for next usages
                    *gin.rail_tail_ptr(channel_idx, lane_idx) = 0;
                    __threadfence_system();
                    return true;
                }
                if (is_last_check) {
                    printf("DeepEP combine (scale-out wait all, UCCL-GIN) timeout, "
                           "scale-out: %d/%d, scale-up: %d/%d, channel: %d, lane: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, lane_idx);
                }
                return false;
            });
#if defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
            const auto finish_wait_profile_cycles = clock64() - finish_wait_profile_start;
            uccl_gin::combine_profile_add(
                combine_profile_counters, uccl_gin::kCombineProfileFinishWaitCycles,
                finish_wait_profile_cycles);
            uccl_gin::combine_profile_add(
                combine_profile_counters, uccl_gin::kCombineProfileFinishWaitEvents, 1);
            uccl_gin::combine_profile_max(
                combine_profile_counters, uccl_gin::kCombineProfileFinishWaitMaxPacked,
                finish_wait_profile_cycles,
                uccl_gin::dispatch_clock_detail(channel_idx, lane_idx));
#endif
        }
        __syncwarp();
#else
        const auto expected_signal = math::pack2<int, int64_t>(1, 0);
        gin.flush<ncclCoopWarp>();
        if (lane_idx < kNumScaleoutRanks) {
            // Update remote tails
            gin.red_add_rel<ncclTeamTagRail>(
                workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx),
                expected_signal, lane_idx);
        }
        __syncwarp();

        // Wait tail arrival
        if (lane_idx < kNumScaleoutRanks) {
            const auto wait_ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx);
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = ptx::ld_acquire_sys<int64_t>(wait_ptr);
                if (signal == expected_signal) {
                    // Clean for next usages
                    *wait_ptr = 0;
                    return true;
                }

                if (is_last_check) {
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "channel: %d, lane: %d, signal: %lld, expected: %lld\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, lane_idx,
                           signal, expected_signal);
                }
                return false;
            });
        }
        __syncwarp();
#endif
    }

#if defined(DEEPEP_USE_UCCL_GIN) && defined(DEEPEP_UCCL_GIN_COMBINE_PROFILE)
    cooperative_groups::this_grid().sync();
    if (sm_idx == 0 and thread_idx == 0) {
        const auto* c = combine_profile_counters;
        printf("UCCL_GIN_COMBINE_PROFILE rank=%d expanded=%d multiple_reduction=%d "
               "scaleup_wait_cycles=%llu scaleup_wait_events=%llu "
               "reduce_cycles=%llu reduce_events=%llu "
               "d2h_cycles=%llu d2h_events=%llu "
               "finish_d2h_cycles=%llu finish_d2h_events=%llu "
               "finish_wait_cycles=%llu finish_wait_events=%llu "
               "remote_puts=%llu transitions=%llu same_dst=%llu "
               "local_contig=%llu remote_contig=%llu both_contig=%llu "
               "runs=%llu run_1=%llu run_2=%llu run_3_4=%llu run_5_8=%llu "
               "run_9_16=%llu run_17_32=%llu run_gt32=%llu "
               "break_dst=%llu break_local_gap=%llu break_remote_gap=%llu "
               "d2h_max_packed=%llu finish_wait_max_packed=%llu\n",
               scaleout_rank_idx, static_cast<int>(kUseExpandedLayout),
               static_cast<int>(kAllowMultipleReduction),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileScaleupWaitCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileScaleupWaitEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileReduceCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileReduceEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileD2HCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileD2HEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileFinishD2HCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileFinishD2HEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileFinishWaitCycles]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileFinishWaitEvents]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRemotePuts]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileTransitions]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileSameDstTransitions]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileLocalContiguousTransitions]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRemoteContiguousTransitions]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileBothContiguousTransitions]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRuns]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBin1]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBin2]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBin3To4]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBin5To8]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBin9To16]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBin17To32]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileRunBinGt32]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileBreakDst]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileBreakLocalGap]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileBreakRemoteGap]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileD2HMaxPacked]),
               static_cast<unsigned long long>(c[uccl_gin::kCombineProfileFinishWaitMaxPacked]));
    }
#endif

    // No barrier at epilogue
}

}  // namespace deep_ep::elastic
