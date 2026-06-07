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

enum DispatchClockCounter : uint32_t {
  kDispatchClockScaleoutPreloadCycles = 0,
  kDispatchClockScaleoutPreloadEvents,
  kDispatchClockScaleoutCompactStoreCycles,
  kDispatchClockScaleoutCompactStoreEvents,
  kDispatchClockScaleoutLocalStoreCycles,
  kDispatchClockScaleoutLocalStoreEvents,
  kDispatchClockScaleoutStoreWaitCycles,
  kDispatchClockScaleoutStoreWaitEvents,
  kDispatchClockScaleoutD2HCycles,
  kDispatchClockScaleoutD2HEvents,
  kDispatchClockScaleoutTailCycles,
  kDispatchClockScaleoutTailEvents,
  kDispatchClockForwardTailWaitCycles,
  kDispatchClockForwardTailWaitEvents,
  kDispatchClockForwardMetaWaitCycles,
  kDispatchClockForwardMetaWaitEvents,
  kDispatchClockForwardLoadCycles,
  kDispatchClockForwardLoadEvents,
  kDispatchClockForwardScaleupStoreCycles,
  kDispatchClockForwardScaleupStoreEvents,
  kDispatchClockForwardTokens,
  kDispatchClockScaleoutRemoteTokens,
  kDispatchClockScaleoutLocalTokens,
  kDispatchClockScaleoutD2HMaxPacked,
  kDispatchClockForwardTailWaitMaxPacked,
  kDispatchClockForwardLoadMaxPacked,
  // Tail-delivery discriminator: was the next tail already visible when the
  // forward warp first looked (ready) or did it have to spin (stall)?
  // ready-dominant  => forward not starved => not delivery/first-visibility bound
  // stall-dominant  => forward starved waiting for the count tail to appear
  kDispatchClockForwardTailReadyEvents,
  kDispatchClockForwardTailStallEvents,
  kDispatchClockForwardTailStallCycles,
  kDispatchClockNumCounters
};

enum DispatchChunkCounter : uint32_t {
  kDispatchChunkChunks = 0,
  kDispatchChunkTokens,
  kDispatchChunkBin1,
  kDispatchChunkBin2,
  kDispatchChunkBin3To4,
  kDispatchChunkBin5To8,
  kDispatchChunkBin9To16,
  kDispatchChunkBin17To24,
  kDispatchChunkBin25To31,
  kDispatchChunkBin32,
  kDispatchChunkBinGt32,
  kDispatchChunkFlushNonContig,
  kDispatchChunkFlushFull,
  kDispatchChunkFlushFinish,
  kDispatchChunkNumCounters
};

enum DispatchChunkFlushReason : uint32_t {
  kDispatchChunkFlushReasonNonContig = 0,
  kDispatchChunkFlushReasonFull,
  kDispatchChunkFlushReasonFinish
};

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

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ void dispatch_clock_add(
    uint64_t* counters, uint32_t counter, uint64_t cycles) {
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
  if (counters != nullptr && counter < kDispatchClockNumCounters && cycles != 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  counters + counter),
              static_cast<unsigned long long>(cycles));
  }
#else
  (void)counters;
  (void)counter;
  (void)cycles;
#endif
}

__device__ __forceinline__ void dispatch_clock_inc(
    uint64_t* counters, uint32_t counter, uint64_t value = 1) {
  dispatch_clock_add(counters, counter, value);
}

__device__ __forceinline__ uint64_t dispatch_clock_detail(
    uint32_t channel, uint32_t aux) {
  return ((static_cast<uint64_t>(channel) & 0xfffull) << 12) |
         (static_cast<uint64_t>(aux) & 0xfffull);
}

__device__ __forceinline__ uint64_t dispatch_clock_pack_max(
    uint64_t cycles, uint64_t detail) {
  constexpr uint64_t kDetailBits = 24;
  constexpr uint64_t kDetailMask = (1ull << kDetailBits) - 1;
  return (cycles << kDetailBits) | (detail & kDetailMask);
}

__device__ __forceinline__ void dispatch_clock_max(
    uint64_t* counters, uint32_t counter, uint64_t cycles, uint64_t detail) {
#if defined(DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE)
  if (counters != nullptr && counter < kDispatchClockNumCounters && cycles != 0) {
    const auto packed = dispatch_clock_pack_max(cycles, detail);
    atomicMax(reinterpret_cast<unsigned long long*>(counters + counter),
              static_cast<unsigned long long>(packed));
  }
#else
  (void)counters;
  (void)counter;
  (void)cycles;
  (void)detail;
#endif
}

__device__ __forceinline__ void dispatch_chunk_add(
    uint64_t* counters, uint32_t counter, uint64_t value) {
#if defined(DEEPEP_UCCL_GIN_CHUNK_PROFILE)
  if (counters != nullptr && counter < kDispatchChunkNumCounters && value != 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(counters + counter),
              static_cast<unsigned long long>(value));
  }
#else
  (void)counters;
  (void)counter;
  (void)value;
#endif
}

__device__ __forceinline__ uint32_t dispatch_chunk_size_bin(uint32_t count) {
  if (count <= 1) return kDispatchChunkBin1;
  if (count == 2) return kDispatchChunkBin2;
  if (count <= 4) return kDispatchChunkBin3To4;
  if (count <= 8) return kDispatchChunkBin5To8;
  if (count <= 16) return kDispatchChunkBin9To16;
  if (count <= 24) return kDispatchChunkBin17To24;
  if (count <= 31) return kDispatchChunkBin25To31;
  if (count == 32) return kDispatchChunkBin32;
  return kDispatchChunkBinGt32;
}

__device__ __forceinline__ uint32_t dispatch_chunk_reason_counter(uint32_t reason) {
  if (reason == kDispatchChunkFlushReasonFull) return kDispatchChunkFlushFull;
  if (reason == kDispatchChunkFlushReasonFinish) return kDispatchChunkFlushFinish;
  return kDispatchChunkFlushNonContig;
}
#endif

}  // namespace uccl_gin
