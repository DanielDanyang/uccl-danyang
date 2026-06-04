#pragma once

#include <cstdint>

enum class CmdType : uint8_t {
  EMPTY = 0,
  WRITE = 1,
  ATOMIC = 2,
  QUIET = 3,
  BARRIER = 4,
};

__host__ __device__ inline CmdType make_cmd_type(CmdType base, bool is_combine,
                                                 bool low_latency) {
  uint8_t v = static_cast<uint8_t>(base);
  if (is_combine) v |= (1u << 6);
  if (low_latency) v |= (1u << 7);
  return static_cast<CmdType>(v);
}

__host__ __device__ inline CmdType get_base_cmd(CmdType c) {
  return static_cast<CmdType>(static_cast<uint8_t>(c) & 0x3Fu);
}

static constexpr int kWriteAddrShiftNormal = 2;

#pragma pack(push, 1)
struct TransferCmd {
  CmdType cmd_type;
  uint8_t dst_rank;
  union {
    struct {
      uint32_t atomic_val : 8;
      uint32_t bytes : 24;
    };
    uint32_t bytes_and_val;
  };
  uint32_t req_rptr;
  union {
    uint32_t req_lptr;
    int value;
  };
  union {
    uint16_t expert_idx;
    uint16_t atomic_offset;
  };
};
#pragma pack(pop)

static_assert(sizeof(TransferCmd) == 16, "TransferCmd must stay 128 bits");

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
}

// Device-only prefix-compatible view of RingBuffer<TransferCmd,
// DeviceToHost, 2048>.  The host proxy allocates the full RingBuffer from
// ring_buffer.cuh; the JIT kernel only needs the stable prefix below.
struct alignas(128) DeviceToHostCmdBuffer {
  static constexpr uint32_t kCapacity = 2048;

  uint64_t head;
  uint64_t tail;
  TransferCmd buf[kCapacity];

  __host__ __device__ static constexpr uint32_t mask() {
    return kCapacity - 1;
  }

  __device__ __forceinline__ bool atomic_set_and_commit(
      TransferCmd const& item, uint64_t* out_slot = nullptr) {
    uint64_t slot;
    while (true) {
      uint64_t h = ld_volatile(&head);
      uint64_t t = ld_volatile(&tail);
      if (h - t == kCapacity) {
        __nanosleep(64);
        continue;
      }
      auto prev = atomicCAS(reinterpret_cast<unsigned long long*>(&head),
                            static_cast<unsigned long long>(h),
                            static_cast<unsigned long long>(h + 1));
      if (prev == h) {
        slot = h;
        break;
      }
    }

    uint32_t idx = static_cast<uint32_t>(slot) & mask();
    TransferCmd tmp = item;
    auto saved_cmd_type = tmp.cmd_type;
    tmp.cmd_type = CmdType::EMPTY;
    buf[idx] = tmp;
    __threadfence_system();
    buf[idx].cmd_type = saved_cmd_type;
    if (out_slot) *out_slot = slot;
    return true;
  }
};
