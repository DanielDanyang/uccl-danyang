#pragma once
//
// deep_ep::elastic::handle::UCCLGin — faithful drop-in for handle::NCCLGin.
//
// Same method surface as NCCLGin (handle.cuh), so DeepEP V2 kernel call sites
// (gin.put<Team>, gin.red_add_rel<Team>, gin.get_sym_ptr<Team>, ...) compile
// unchanged when the kernel's gin type is swapped to UCCLGin (DEEPEP_GIN_T, P2).
//
// Routing:
//   Team == ncclTeamTagRail  -> UCCL D2H + proxy + EFA  (the EFA hot path we own)
//   Team == ncclTeamTagLsa / ncclTeamTagWorld -> delegate to a composed NCCLGin
//     (NVLink ptx / NCCL GIN: barriers, scaleup, world — unchanged, low-frequency)
//
// NOT a transparent drop-in for ONE thing — red_add_rel<Rail> (the tail):
//   NCCL applies the add on the receiver GPU into the *window*; UCCL applies it on
//   the receiver *CPU proxy* into a host-mapped *atomic buffer* (CPU can't fetch_add
//   device HBM). So the tail STORAGE moves to res.atomic_tail_base, and the kernel's
//   forward-warp READ of the tail must be redirected there too (P2 kernel patch,
//   mirrors the deleted fork's v2_atomic_tail_ptr). The write side below keeps the
//   same call site; the read side is the P2 todo.
//
// Lives in deep_ep::elastic::handle so it is a name-compatible sibling of NCCLGin.
// This header is for the DeepEP integration; the lean standalone microbench uses
// uccl_gin.cuh instead.

#include <nccl.h>
#include <nccl_device.h>
#include <deep_ep/common/handle.cuh>   // handle::NCCLGin (composed for Lsa/World)
#include <type_traits>

#include "resources.cuh"
#include "uccl_gin_rail.cuh"

namespace deep_ep::elastic::handle {

static constexpr int kUCCLGinTailFinishDelta = 1 << 13;
static constexpr int kUCCLGinTailCountMask = kUCCLGinTailFinishDelta - 1;
static constexpr unsigned long long kUCCLGinQuietPrintCycles = 20000000000ull;

struct UCCLGin {
  const ncclDevComm_t& nccl_dev_comm;      // comm.cuh compatibility
  const ncclWindow_t& nccl_window;         // comm.cuh / offset compatibility
  NCCLGin nccl;                          // Lsa / World / barrier / members (reuse upstream)
  uccl_gin::UCCLGinResources res;        // Rail backend resources

  __device__ __forceinline__
  UCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
          const uccl_gin::UCCLGinResources& res, const int& qp_idx = 0,
          const ncclGinResourceSharingMode& sharing = NCCL_GIN_RESOURCE_SHARING_GPU)
      : nccl_dev_comm(nccl_dev_comm), nccl_window(nccl_window),
        nccl(nccl_dev_comm, nccl_window, qp_idx, sharing), res(res) {}

  // Global proxy peer rank for a Rail dst (Rail rank == scaleout rank).
  __device__ __forceinline__ int rail_global_rank(int dst_scaleout) const {
    return dst_scaleout * res.num_scaleup_ranks + res.scaleup_rank;
  }
  __device__ __forceinline__ d2hq::D2HHandle* lane(int hint) const {
    if (res.d2h_queues == nullptr || res.num_queues == 0) {
      __trap();
    }
    return res.d2h_queues[static_cast<uint32_t>(hint) % res.num_queues];
  }

  // ---- pointer translation / membership: identical to NCCLGin -----------
  template <typename team_t, typename dtype_t = void*>
  __device__ __forceinline__ dtype_t* get_sym_ptr(dtype_t* ptr, const int& dst) const {
    return nccl.get_sym_ptr<team_t>(ptr, dst);   // Rail self/null + Lsa NVLink, same semantics
  }
  template <typename dtype_t = void*>
  __device__ __forceinline__ uint64_t get_sym_offset(dtype_t* ptr) const {
    return nccl.get_sym_offset(ptr);
  }
  template <typename team_t>
  __device__ __forceinline__ bool is_nvlink_accessible(const int& dst) const {
    return nccl.is_nvlink_accessible<team_t>(dst);
  }

  // ---- put --------------------------------------------------------------
  template <typename team_t, typename remote_action_t = ncclGin_None>
  __device__ __forceinline__
  void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes,
           const int& dst_rank_idx, const int& extra_options = 0,
           const remote_action_t& remote_action = remote_action_t(),
           int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      // remote_action piggybacks a signal in one GIN op; the UCCL split (WRITE +
      // separate ordered ATOMIC) cannot fuse it. DeepEP's scaleout payload put
      // uses remote_action=None (tail is a separate red_add), so require None.
      static_assert(std::is_same_v<remote_action_t, ncclGin_None>,
                    "UCCLGin: Rail put with remote_action not supported (use red_add_rel)");
      (void)extra_options;  // AggregateRequests handled by our own coalescing (P3)
      if (dst_rank_idx == res.scaleout_rank) {
        auto* dst32 = reinterpret_cast<uint32_t*>(recv_sym_ptr);
        const auto* src32 = reinterpret_cast<const uint32_t*>(send_sym_ptr);
        const int words = num_bytes / static_cast<int>(sizeof(uint32_t));
        for (int i = 0; i < words; ++i) dst32[i] = src32[i];
        auto* dst8 = reinterpret_cast<uint8_t*>(recv_sym_ptr) + words * sizeof(uint32_t);
        const auto* src8 = reinterpret_cast<const uint8_t*>(send_sym_ptr) + words * sizeof(uint32_t);
        for (int i = words * static_cast<int>(sizeof(uint32_t)); i < num_bytes; ++i)
          dst8[i - words * static_cast<int>(sizeof(uint32_t))] =
              src8[i - words * static_cast<int>(sizeof(uint32_t))];
        __threadfence_system();
        return;
      }
      const uint32_t loff = uccl_gin::window_off(reinterpret_cast<uint64_t>(send_sym_ptr), res.window_base);
      const uint32_t roff = uccl_gin::window_off(reinterpret_cast<uint64_t>(recv_sym_ptr), res.window_base);
      uccl_gin::rail_put(lane(lane_hint), rail_global_rank(dst_rank_idx),
                         static_cast<uint32_t>(num_bytes), loff, roff);
    } else {
      nccl.put<team_t>(recv_sym_ptr, send_sym_ptr, num_bytes, dst_rank_idx,
                       extra_options, remote_action);
    }
  }

  // ---- put_value --------------------------------------------------------
  template <typename team_t, typename dtype_t>
  __device__ __forceinline__
  void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                 const int& extra_options = 0, int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      // Single-word WRITE to the peer window slot (e.g. notify counts).
      (void)extra_options;
      const uint32_t roff = uccl_gin::window_off(reinterpret_cast<uint64_t>(sym_ptr), res.window_base);
      // value staged into the window region pointed by sym_ptr is the caller's job
      // for a real put_value we need a local source; v1 routes through red_add or
      // a small staged WRITE. Keep as a TODO-trap until the count path is wired (P2).
      (void)roff; __trap();
    } else {
      nccl.put_value<team_t>(sym_ptr, value, dst_rank_idx, extra_options);
    }
  }

  // ---- red_add_rel (tail) ----------------------------------------------
  // NOT a transparent drop-in for Rail. NCCL applies the add on the receiver GPU
  // into the window at (sym_ptr - lsa_base); UCCL applies it on the receiver CPU
  // proxy into a host-mapped atomic buffer, AND the ordered-atomic immediate only
  // carries a ~13-bit offset (<=8191) — too small for raw window tail offsets. So
  // the tail needs a COMPACT (channel, source-rank) index (cf. the deleted fork's
  // v2_atomic_tail_ptr), which cannot be recovered from `sym_ptr` alone, and the
  // forward-warp READ must point at the same compact slot in atomic_tail_base.
  //
  // => P2 handles the tail explicitly: the kernel's tail write + forward-warp read
  // are patched to a UCCL compact-index op (see uccl_gin::rail_red_add). The Lsa
  // branch stays a clean delegate.
  template <typename team_t, typename dtype_t>
  __device__ __forceinline__
  void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                   const int& extra_options = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      (void)sym_ptr; (void)value; (void)dst_rank_idx; (void)extra_options;
      __trap();  // P2: replace tail call sites with the compact-index UCCL op
    } else {
      nccl.red_add_rel<team_t>(sym_ptr, value, dst_rank_idx, extra_options);
    }
  }

  // Explicit compact-index tail add for the P2 kernel patch (Rail only): the
  // caller passes the (channel, source-rank) compact slot directly, matching the
  // forward-warp read in atomic_tail_base.
  __device__ __forceinline__ uint32_t rail_tail_offset(int channel_idx, int src_rank_idx) const {
    if (channel_idx < 0 || src_rank_idx < 0 ||
        src_rank_idx >= res.num_scaleout_ranks) {
      __trap();
    }
    return static_cast<uint32_t>(
        (channel_idx * res.num_scaleout_ranks + src_rank_idx) * (int)sizeof(int64_t));
  }

  __device__ __forceinline__ int64_t* rail_tail_ptr(int channel_idx, int src_rank_idx) const {
    return reinterpret_cast<int64_t*>(res.atomic_tail_base + rail_tail_offset(channel_idx, src_rank_idx));
  }

  __device__ __forceinline__ void rail_put_tail_add(
      void* recv_sym_ptr, void* send_sym_ptr, int num_bytes, int dst_scaleout,
      int channel_idx, int src_rank_idx, int count_delta, int lane_hint = 0) const {
    if (dst_scaleout < 0 || dst_scaleout >= res.num_scaleout_ranks ||
        count_delta <= 0 || count_delta > 0xFF) {
      __trap();
    }
    if (dst_scaleout == res.scaleout_rank) {
      put<ncclTeamTagRail>(recv_sym_ptr, send_sym_ptr, num_bytes, dst_scaleout,
                           0, ncclGin_None(), lane_hint);
      atomicAdd_system(reinterpret_cast<unsigned long long*>(
                           res.atomic_tail_base + rail_tail_offset(channel_idx, src_rank_idx)),
                       static_cast<unsigned long long>(count_delta));
      return;
    }
    const uint32_t loff = uccl_gin::window_off(reinterpret_cast<uint64_t>(send_sym_ptr), res.window_base);
    const uint32_t roff = uccl_gin::window_off(reinterpret_cast<uint64_t>(recv_sym_ptr), res.window_base);
    uccl_gin::rail_put_tail_add(
        lane(lane_hint), rail_global_rank(dst_scaleout), static_cast<uint32_t>(num_bytes),
        loff, roff, static_cast<uint32_t>(count_delta),
        rail_tail_offset(channel_idx, src_rank_idx));
  }

  __device__ __forceinline__ void decode_rail_tail(int64_t raw, int& finish, int& count) const {
    finish = raw >= kUCCLGinTailFinishDelta;
    count = static_cast<int>(raw - (finish ? kUCCLGinTailFinishDelta : 0));
  }

  __device__ __forceinline__ void rail_tail_add(int channel_idx, int src_rank_idx, int dst_scaleout,
                                                int count_delta, bool finish, int lane_hint = 0) const {
    if (dst_scaleout < 0 || dst_scaleout >= res.num_scaleout_ranks ||
        count_delta < 0 || count_delta > kUCCLGinTailCountMask ||
        count_delta + (finish ? kUCCLGinTailFinishDelta : 0) > uccl_gin::kAtomicValueMax) {
      __trap();
    }
    const uint32_t off = rail_tail_offset(channel_idx, src_rank_idx);
    const int delta = count_delta + (finish ? kUCCLGinTailFinishDelta : 0);
    if (dst_scaleout == res.scaleout_rank) {
      atomicAdd_system(reinterpret_cast<unsigned long long*>(res.atomic_tail_base + off),
                       static_cast<unsigned long long>(delta));
      return;
    }
    uccl_gin::rail_red_add(lane(lane_hint), rail_global_rank(dst_scaleout), delta, off);
  }

  __device__ __forceinline__ void quiet(int lane_hint = 0) const {
    auto* q = lane(lane_hint);
    uint64_t slot = 0;
    TransferCmd cmd{};
    cmd.cmd_type = CmdType::QUIET;
    q->atomic_set_and_commit(cmd, &slot);

    auto last_print = clock64();
    while (true) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
      if (q->fifo.poll(slot)) break;
#else
      const uint64_t tail = q->ring->volatile_tail();
      if (tail > slot) break;
#endif
      if (clock64() - last_print > kUCCLGinQuietPrintCycles) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
        printf("[UCCL-GIN quiet] waiting lane=%d slot=%llu\n",
               lane_hint,
               static_cast<unsigned long long>(slot));
#else
        printf("[UCCL-GIN quiet] waiting lane=%d slot=%llu head=%llu tail=%llu\n",
               lane_hint,
               static_cast<unsigned long long>(slot),
               static_cast<unsigned long long>(q->ring->head),
               static_cast<unsigned long long>(tail));
#endif
        last_print = clock64();
      }
      __nanosleep(64);
    }
  }

  // ---- the rest: delegate to NCCLGin (Lsa/World/barrier; Rail get/signal are
  //      not on the dispatch hot path in v1, handled in later phases) ---------
  template <typename team_t, typename remote_action_t>
  __device__ __forceinline__ void signal(const int& dst, const remote_action_t& a) const {
    nccl.signal<team_t>(dst, a);
  }
  template <typename team_t, typename coop_t = ncclCoopThread, typename segment_t = ncclGin_SegmentDevice>
  __device__ __forceinline__ void get(void* s, void* d, const int& nb, const int& src, const int& xo = 0) const {
    nccl.get<team_t, coop_t, segment_t>(s, d, nb, src, xo);
  }
  __device__ __forceinline__ void wait(ncclGinRequest_t& r) const { nccl.wait(r); }
  template <typename coop_t = ncclCoopThread>
  __device__ __forceinline__ void flush() const { nccl.flush<coop_t>(); }
  template <typename team_t, typename coop_t = ncclCoopThread>
  __device__ __forceinline__ void flush_async(const int& src, ncclGinRequest_t* r, const int& xo = 0) const {
    nccl.flush_async<team_t, coop_t>(src, r, xo);
  }
};

}  // namespace deep_ep::elastic::handle
