#pragma once
//
// UCCLGinResources — the stable resource bundle injected into `UCCLGin` once at
// construction, so the kernel/JIT signature does not churn as the backend grows.
//
// NCCLGin gets everything from nccl_dev_comm/window/qp_idx/sharing_mode; the
// UCCL Rail backend additionally needs the D2H rings, the registered-window base
// (offset origin for `put`), and the atomic buffer base (offset origin for the
// ordered `red_add_rel`). See ep/docs/uccl_gin_plan.md (§7 UCCLGinResources).

#include "../d2h_queue_device.cuh"  // d2hq::D2HHandle
#include <cstdint>

namespace uccl_gin {

struct UCCLGinResources {
  // Rail (EFA) transport.
  d2hq::D2HHandle** d2h_queues = nullptr;  // device array of D2H handle pointers
  uint32_t num_queues = 0;

  // Offset origins (single registered symmetric window + atomic buffer).
  uint64_t window_base = 0;        // `put` payload offsets are relative to this
  uint64_t atomic_tail_base = 0;   // `red_add_rel` counter offsets are relative to this

  // Topology / lane mapping (mirrors NCCLGin's rank info).
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int scaleout_rank = 0;
  int scaleup_rank = 0;
  uint32_t num_lanes = 1;
};

}  // namespace uccl_gin
