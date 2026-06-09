#pragma once
//
// handle::UCCLGin — the UCCL-GIN abstraction. Mirrors the method surface of
// DeepEP's `deep_ep::elastic::handle::NCCLGin` (`deep_ep/common/handle.cuh`) so
// the SAME kernel call sites (`gin.put<Team>(...)`, `gin.red_add_rel<Team>(...)`)
// work by just swapping the gin type:
//
//   Team == ncclTeamTagRail (scale-out / inter-node) -> UCCL D2H + proxy + EFA
//   Team == ncclTeamTagLsa  (scale-up / NVLink)       -> forward to NCCL/NVLink
//
// This is the thing the standalone microbench tests (vs native NCCLGin), and the
// thing DeepEP V2 kernels will later be retargeted onto (see uccl_gin_plan.md).
//
// SCOPE (v1): the Rail branch of `put` + `red_add_rel` is implemented (the two
// ops the microbench compares). put_value / signal / wait / flush / get_sym_ptr
// and the Lsa branch are declared as the surface but left for P1; calling an
// unimplemented path traps so gaps are loud, not silent.
//
// Team tags come from <nccl_device.h> (the exact tags DeepEP uses), so this
// header is faithful both standalone and when later folded into DeepEP.

#include "resources.cuh"
#include "uccl_gin_rail.cuh"
#include <nccl_device.h>   // ncclTeamTagRail, ncclTeamTagLsa
#include <type_traits>

namespace uccl_gin {

struct UCCLGin {
  UCCLGinResources res;

  __device__ __forceinline__ explicit UCCLGin(const UCCLGinResources& r) : res(r) {}

  // Choose a D2H lane. NCCLGin hides lane behind qp/context; here the caller may
  // pass a hint (e.g. channel idx); default round-robins on the hint.
  __device__ __forceinline__ d2hq::D2HHandle* lane(int hint) const {
    if (res.d2h_queues == nullptr || res.num_queues == 0) {
      __trap();
    }
    return res.d2h_queues[uccl_gin::queue_index_from_hint(res, hint)];
  }

  // ---- put -------------------------------------------------------------
  // Signature mirrors handle::NCCLGin::put: symmetric pointers in, internally
  // converted to window offsets. `dst_rank` is the global proxy peer rank.
  template <typename team_t>
  __device__ __forceinline__ void put(void* recv_sym_ptr, void* send_sym_ptr,
                                      int num_bytes, int dst_rank,
                                      int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      const uint32_t loff = window_off(reinterpret_cast<uint64_t>(send_sym_ptr), res.window_base);
      const uint32_t roff = window_off(reinterpret_cast<uint64_t>(recv_sym_ptr), res.window_base);
      rail_put(lane(lane_hint), dst_rank, static_cast<uint32_t>(num_bytes), loff, roff);
    } else {
      // Lsa (NVLink) — P1: forward to NCCLGin / NVLink ptx. Not in standalone.
      __trap();
    }
  }

  // ---- red_add_rel -----------------------------------------------------
  // Mirrors handle::NCCLGin::red_add_rel. `sym_ptr` is a counter inside the
  // atomic buffer; offset is taken relative to atomic_tail_base. Ordered
  // (PackAtomicWithSeq) so a stream of adds to one counter is not reordered.
  template <typename team_t>
  __device__ __forceinline__ void red_add_rel(void* sym_ptr, int value,
                                              int dst_rank, int lane_hint = 0) const {
    if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) {
      const uint32_t off = static_cast<uint32_t>(
          reinterpret_cast<uint64_t>(sym_ptr) - res.atomic_tail_base);
      rail_red_add(lane(lane_hint), dst_rank, value, off);
    } else {
      __trap();  // Lsa: ptx::red_add_rel_sys in P1
    }
  }

  // ---- surface declared for parity, not yet implemented (P1) -----------
  template <typename team_t>
  __device__ __forceinline__ void put_value(void* /*sym_ptr*/, int /*value*/,
                                            int /*dst_rank*/, int /*lane_hint*/ = 0) const {
    __trap();
  }
  __device__ __forceinline__ void flush() const { /* P1: quiet via proxy completion */ }
};

}  // namespace uccl_gin
