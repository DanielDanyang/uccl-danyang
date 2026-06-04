#pragma once

#include <cstddef>
#include <cstdint>

// Signal-scratch geometry for the native V2 EFA dispatch/combine path.
//
// The tail word the scaleout warp produces is computed in registers and must be
// staged into registered memory before the proxy RDMA-writes it to the remote
// workspace.  Each D2H queue owns a contiguous block of int64 scratch slots,
// one per ring slot, so a scratch slot is never reused before its command is
// drained (slot lifetime mirrors the ring slot it is bound to).  The scratch
// block lives inside the registered NCCL window (carved from the flexible
// cpu_buffer/engram region), so a single MR covers it.

namespace uccl::v2_efa {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define V2_EFA_DEV __device__ __forceinline__
#else
#define V2_EFA_DEV inline
#endif

constexpr uint32_t kSignalScratchSlotBytes = sizeof(int64_t);

// Total scratch bytes for `num_queues` queues, each owning `slots_per_queue`
// int64 slots (slots_per_queue should equal the D2H ring capacity).
inline size_t signal_scratch_bytes(uint32_t num_queues,
                                   uint32_t slots_per_queue) {
  return static_cast<size_t>(num_queues) * static_cast<size_t>(slots_per_queue) *
         kSignalScratchSlotBytes;
}

// Pointer to the scratch slot bound to (queue_idx, slot) where `slot` is the
// D2H ring slot modulo `slots_per_queue`.
V2_EFA_DEV int64_t* signal_scratch_slot_for(uint64_t scratch_base,
                                            uint32_t queue_idx, uint32_t slot,
                                            uint32_t slots_per_queue) {
  return reinterpret_cast<int64_t*>(scratch_base) +
         static_cast<uint64_t>(queue_idx) * slots_per_queue + slot;
}

}  // namespace uccl::v2_efa
