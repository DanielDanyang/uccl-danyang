# UCCL-GIN Architecture Document

## 从 `495b7221` (UCCL/EP V1) 到当前 `uccl-gin` (compact chunking + piggyback tail)

本文档讲解三个独立子系统(NCCL-GIN, DeepEP V2, UCCL/EP V1),然后解释如何将它们
缝合为 UCCL-GIN,以及 compact channel staging + piggyback tail 如何把 dispatch 从
~5 GB/s 推到 ~38 GB/s。

---

## 目录

1. [背景: 三个子系统概览](#1-背景-三个子系统概览)
2. [NCCL-GIN: GPU 如何直接发起网络 IO](#2-nccl-gin-gpu-如何直接发起网络-io)
3. [DeepEP V2: MoE dispatch 的分层 warp 架构](#3-deepep-v2-moe-dispatch-的分层-warp-架构)
4. [UCCL/EP V1 (`495b7221`): CPU proxy + D2H ring 传输底座](#4-ucclep-v1-495b7221-cpu-proxy--d2h-ring-传输底座)
5. [UCCL-GIN 合成: 把 V2 的 scale-out 映射到 V1 的传输底座](#5-uccl-gin-合成-把-v2-的-scale-out-映射到-v1-的传输底座)
6. [修改清单: 当前 vs `495b7221`](#6-修改清单-当前-vs-495b7221)
7. [关键数据流详解](#7-关键数据流详解)
8. [Compact Channel Staging: 从 per-token WRITE 到 multi-token chunk](#8-compact-channel-staging-从-per-token-write-到-multi-token-chunk)
9. [Piggyback tail: 消除独立 tail WRITE_WITH_IMM](#9-piggyback-tail-消除独立-tail-write_with_imm)
10. [Sender-side completion dependency: 在 EFA 上实现 payload-before-tail](#10-sender-side-completion-dependency-在-efa-上实现-payload-before-tail)
11. [Combine 迁移](#11-combine-迁移)
12. [当前性能状态](#12-当前性能状态)

---

## 1. 背景: 三个子系统概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DeepEP V2 dispatch kernel                        │
│  (hybrid_dispatch.cuh — MoE token routing, warp-level parallelism)      │
│                                                                         │
│   3 warp roles: [NOTIFY] [SCALEOUT_SEND] [FORWARD]                      │
│                       │                              │                   │
│                       ▼                              ▼                   │
│              Scale-out (EFA)                   Scale-up (NVLink)         │
│                       │                              │                   │
│          ┌────────────┴────────────┐                  │                   │
│          ▼                         ▼                  ▼                   │
│   ┌──────────────┐          ┌──────────────┐   ┌──────────────┐          │
│   │  NCCL-GIN     │          │  UCCL-GIN    │   │  NCCL-GIN    │          │
│   │  (原生路径)    │    vs    │  (我们的路径) │   │  (Lsa/NVLink) │          │
│   │  ~5 GB/s EFA  │          │  ~38 GB/s EFA│   │  ~120 GB/s    │          │
│   └──────┬───────┘          └──────┬───────┘   └──────────────┘          │
│          │                         │                                      │
│          ▼                         ▼                                      │
│   aws-ofi-nccl               UCCL CPU proxy                               │
│   libfabric/EFA              D2H ring + raw ibverbs/EFA                   │
│   GIN proxy                  compact channel staging + piggyback tail     │
│   (FORCE_SO signals MR)      (sender-side dep + per-slot metadata check)  │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心差异**: NCCL-GIN 靠 NIC 强序(FORCE_SO MR)保证 payload-before-signal;UCCL-GIN
靠 GPU 侧 compact channel staging + piggyback tail + sender-side dependency +
per-slot metadata readiness 四重机制重建等价保证。

---

## 2. NCCL-GIN: GPU 如何直接发起网络 IO

### 2.1 核心概念

NCCL 2.30+ 的 GIN (GPU-Initiated Networking):GPU SM 通过 `ncclGin` 对象直接发起
RDMA put/get/signal。需要:
1. **GIN context**: 每个 warp/channel 一个,预分配 device-side queue、signals buffer
2. **Window registration**: `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`
3. **Plugin bridge**: `ncclGin.put(...)` → `ncclDevComm_t` → host plugin 的
   `ncclGinPlugin_v13`

### 2.2 handle::NCCLGin

`DeepEP/deep_ep/include/deep_ep/common/handle.cuh` 是 DeepEP V2 的统一 GIN 接口:

```
handle::NCCLGin {
    ncclDevComm_t&  nccl_dev_comm;
    ncclWindow_t&   nccl_window;
    ncclGin         gin;
    ncclTeam        team_world, team_lsa, team_rail;
    uint64_t        lsa_base_ptr;
}
```

Team 路由:

| Team tag          | 含义            | 通信路径            |
|-------------------|-----------------|---------------------|
| `ncclTeamTagLsa`  | Local/Scale-Up  | NVLink/NVSwitch ptx |
| `ncclTeamTagRail` | Remote/Scale-Out| RDMA/NIC            |
| `ncclTeamTagWorld`| 全量            | Lsa + Rail          |

### 2.3 NCCL FORCE_SO: 信令强序

NCCL GIN proxy 通过 **strong-ordered signals MR** 保证 payload-before-signal:

```
nccl/src/gin/gin_host_proxy.cc:475:
// Enforcing strong ordering on the signals mr is vital to ensure
// ordering between puts and signals.
NCCLCHECK(ncclGinProxyRegMrSym(collComm, proxyCtx->signalsDev, signalsBufSize,
                               NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO, ...));
```

```
GPU warp:  gin.put(...)        gin.signal(..., ncclGin_VASignalAdd(...))
               │                        │
               ▼                        ▼
GIN queue:  [PUT desc]  [PUT desc]  [SIGNAL desc]
               │            │            │
               ▼            ▼            ▼
NIC (SO MR):  WRITE 数据到达 remote memory 后,SIGNAL (atomic add) 才可见
               → receiver GPU 读 signaled tail 时,payload 一定已落地
```

这是 **NIC 级强序**: FORCE_SO MR 上的操作不会互相超越。UCCL-GIN 不能依赖这个
(EFA SRD 不保证顺序),必须自己重建等价保证(见 §10)。

### 2.4 `ncclGinOptFlagsAggregateRequests` 的真实含义

```
gin_device_common.h:  ncclGinOptFlagsAggregateRequests = (1 << 1)
gin_gdaki.h:          DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING
gin/proxy/:           未引用 — EFA proxy 下标志被忽略
```

这不是合并多个 put 为一条 RDMA WRITE。它只是延迟 NIC doorbell:
连续 `gin.put()` 写 WR 到 NIC submission queue,不敲中间门铃;最后一条不带 flag
的 put 或 flush 才敲一次。减少的是 GPU→NIC doorbell 写的开销,不是 RDMA 操作数。
每个 `gin.put()` 仍是独立 WR。

---

## 3. DeepEP V2: MoE dispatch 的分层 warp 架构

### 3.1 Kernel 整体结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    SM (blockIdx.x)                                │
│                                                                   │
│  ┌──────────────────────┐                                        │
│  │  NOTIFY warps        │  统计 recv tokens, barrier 同步        │
│  └──────────────────────┘                                        │
│                                                                   │
│  ┌──────────────────────┐                                        │
│  │  SCALEOUT warps       │  每个 warp = 一个 channel              │
│  │  = kNumScaleoutWarps  │                                        │
│  │                       │  route token → dst                    │
│  │                       │  compact batch TMA store              │
│  │                       │  rail_put_tail_add(piggyback tail)    │
│  └──────────────────────┘                                        │
│                                                                   │
│  ┌──────────────────────┐                                        │
│  │  FORWARD warps        │  每个 warp = 一个 channel              │
│  │                       │                                        │
│  │                       │  读 tail counter → 消费 recv slot     │
│  │                       │  per-slot metadata readiness check    │
│  │                       │  route → scaleup (NVLink/本地)        │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 原始 DeepEP V2 的 per-token put

原始代码 (`DeepEP/.../hybrid_dispatch.cuh:444`):

```cpp
// 每个 token 一条 gin.put — 一条 RDMA WRITE
gin.put<ncclTeamTagRail>(
    scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
    scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
    tma_buffer.get_num_bytes<false>(),
    stored_dst_scaleout_rank_idx,
    ncclGinOptFlagsAggregateRequests);
```

`send_buffer` 按 `token_idx` 索引 — 同 dst 的 token 在内存里不连续。
FP8 hidden=7168 时每条 WRITE ~14KB,在 EFA 上只能打到 ~5 GB/s。

### 3.3 Buffer 布局改造

```
buffer (一段连续 GPU HBM):
  ├─ scaleup_buffer          = kNumScaleupRanks × kNumScaleoutRanks × kNumMaxTokensPerRank
  ├─ scaleout_send_buffer    = kNumChannels × kNumMaxTokensPerChannel     ← compact per-channel
  └─ scaleout_recv_buffer    = kNumScaleoutRanks × kNumChannels × kNumMaxTokensPerChannel

原来: scaleout_send_buffer = BufferLayout(token_layout, 1, kNumMaxTokensPerRank, ...)
       send_buffer[token_idx]  ← 稀疏,dst 穿插

现在: scaleout_send_buffer = BufferLayout(token_layout, 1, kNumCompactSendTokens, ...)
       其中 kNumCompactSendTokens = kNumChannels × kNumMaxTokensPerChannel
       send_buffer[channel][compact_slot]  ← 同 channel 内 compact 连续
```

---

## 4. UCCL/EP V1 (`495b7221`): CPU proxy + D2H ring 传输底座

### 4.1 整体架构

```
 GPU (device)                    CPU (host)                    远端 GPU
 ────────────                    ─────────                     ────────

 ┌──────────┐     D2H ring      ┌──────────┐    EFA verbs     ┌──────────┐
 │ dispatch │ ───────────────→  │  Proxy   │ ───────────────→ │ 远端      │
 │ kernel   │    (TransferCmd)  │  thread  │   RDMA WRITE     │ recv buf  │
 │ (SM)     │                   │          │                  │           │
 │          │ ←───────────────  │          │ ←─────────────── │           │
 │          │   completion ack  │          │   CQE / atomic   │           │
 └──────────┘                   └──────────┘                  └──────────┘
```

### 4.2 TransferCmd 格式 (16 bytes)

```
Byte:  0        1        2      4        6        8       12
     ┌────────┬────────┬────────┬────────┬────────┬────────┐
     │cmd_type│dst_rank│bytes   │req_rptr│req_lptr│atomic  │
     │(1B)    │(1B)    │(2B)    │(4B)    │/value  │_offset │
     │        │        │+atomic │        │(4B)    │(2B)    │
     │        │        │_val(1B)│        │        │        │
     └────────┴────────┴────────┴────────┴────────┴────────┘

CmdType: EMPTY=0, WRITE=1, ATOMIC=2, QUIET=3, BARRIER=4
```

Piggyback tail 复用 `atomic_val`(8-bit) 字段携带 count delta (1..255),
`atomic_offset` 携带 tail buffer byte offset — 不改变 16B ABI。

### 4.3 WRITE cmd vs ATOMIC cmd 的差异

```
WRITE cmd (payload transfer):
  cmd_type = WRITE
  req_lptr = local window offset (4-byte shifted)
  req_rptr = remote window offset (4-byte shifted)
  bytes    = payload bytes

WRITE cmd + piggyback tail:
  同上, 但 atomic_val   = count_delta (1..255)
          atomic_offset = tail buffer byte offset

ATOMIC cmd (standalone tail/finish):
  cmd_type  = ATOMIC
  value     = signed delta (±16383)
  req_rptr  = tail buffer byte offset
  atomic_offset = 1 (non-zero → ordered PackAtomicWithSeq path)
```

### 4.4 V1 的 chunked RDMA

V1 的 send buffer 按 `send_buffer[dst][slot]` 布局 — 同 dst 天然连续。
coordinator warp 可攒 16 token 发一条大 RDMA WRITE:

```cpp
// internode.cu:1080-1106
size_t const num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
uccl::nvshmemi_ibgda_put_nbi_warp(..., num_bytes_per_msg, ...);
// EFA 路径: tail offset + count 嵌入同一 WR (piggyback)
```

V2 缺这个"同 dst 连续"性质,compact channel staging 把它补回来。

---

## 5. UCCL-GIN 合成: 把 V2 的 scale-out 映射到 V1 的传输底座

### 5.1 handle::UCCLGin 的 team 路由

```
handle::UCCLGin {
    nccl_dev_comm, nccl_window;    // 兼容 NCCLGin 的构造签名
    NCCLGin nccl;                  // 组合: Lsa/World 仍走原生 NCCL-GIN
    UCCLGinResources res;         // Rail 后端资源
}

put<Rail>()         → rail_put / rail_put_tail_add  → D2H → proxy → EFA
put<Lsa>()          → nccl.put<Lsa>()               → 原生 NCCL NVLink
put<World>()        → nccl.put<World>()              → 原生 NCCL
put_value<Rail>()   → __trap() (P2 TODO)
red_add_rel<Rail>() → __trap() (已用 compact-index API 替代)
red_add_rel<Lsa>()  → nccl.red_add_rel<Lsa>()        → 原生 NCCL

Explicit compact-index tail API:
rail_put_tail_add() → 一条 WRITE cmd + piggyback tail (atomic_val + atomic_offset)
rail_tail_add()     → 独立 tail ATOMIC cmd (finish flag 用)
rail_red_add()      → 独立 ordered ATOMIC cmd (single delta)
```

### 5.2 为什么 tail 需要 compact-index API

NCCL-GIN 的 `red_add_rel` 传 window 内的 `sym_ptr`。UCCL 做不到:
1. UCCL 的 atomic 走 `WRITE_WITH_IMM`(CPU 代理软件 atomic,EFA 不支持硬件 atomics)
2. `PackAtomicWithSeq` 的 offset 限制在 `kAtomicOffMask = 0x1FFF`(13 bits, ≤8191 bytes)
3. DeepEP V2 的 window offset 远超 8KB

所以 tail 存储从"window 内部署"迁移到 compact **atomic_tail_base**:
```
atomic_tail_base[channel_idx * num_scaleout_ranks + src_rank_idx]
```
每个条目 8 字节(int64_t)。编码:finish bit = `kUCCLGinTailFinishDelta = 8192`,
count 部分 = raw % 8192。forward warp 解码:
```
finish = raw >= 8192
count  = raw - (finish ? 8192 : 0)
```

### 5.3 UCCLGinResources

```cpp
struct UCCLGinResources {
    d2hq::D2HHandle** d2h_queues;    // device array of D2H handle pointers
    uint32_t num_queues;
    uint64_t window_base;            // registered window offset origin
    uint64_t atomic_tail_base;       // tail counter buffer base
    int num_scaleout_ranks;
    int num_scaleup_ranks;
    int scaleout_rank;
    int scaleup_rank;
    uint32_t num_lanes;
};
```

### 5.4 文件结构

```
ep/
├── include/
│   ├── uccl_gin/
│   │   ├── uccl_gin_handle.cuh      ← handle::UCCLGin (DeepEP 集成)
│   │   ├── uccl_gin_rail.cuh        ← rail_put / rail_put_tail_add / rail_red_add
│   │   └── resources.cuh            ← UCCLGinResources POD + profile counters
│   ├── proxy.hpp                    ← proxy context, atomic batch, profile
│   ├── ring_buffer.cuh              ← TransferCmd, CmdType (V1 继承)
│   └── d2h_queue_device.cuh         ← d2hq::D2HHandle (V1 继承)
├── src/
│   ├── proxy.cpp                    ← async per-tail dep, piggyback decode, profiling
│   ├── rdma.cpp                     ← WRITE_WITH_IMM + PackAtomicWithSeq + piggyback
│   └── uccl_ep.cc                   ← 资源初始化 + JIT 编译入口
├── tests/
│   └── uccl_gin_microbench/         ← 独立 microbench (vs NCCL GIN)
└── docs/
    ├── uccl_gin_plan.md
    ├── uccl_gin_architecture.md     ← 本文档
    ├── uccl_gin_perf_plan.md        ← 性能优化计划与数据
    └── uccl_gin_perf_cx7_vs_efa.md  ← CX7 vs EFA benchmark

thirdparty/DeepEP-v2-d4f41e4/        ← vendored DeepEP V2 (in-tree)
├── deep_ep/
│   ├── include/deep_ep/
│   │   ├── common/handle.cuh        ← NCCLGin (纯净)
│   │   └── impls/
│   │       └── hybrid_dispatch.cuh  ← 主 dispatch kernel (UCCL-GIN patch)
│   └── buffers/elastic.py           ← Python buffer (atomic_tail_base 分配)
└── csrc/
    ├── elastic/buffer.hpp           ← get_native_v2_resources() 接口注入
    └── jit/                         ← JIT compiler include path 注入
```

---

## 6. 修改清单: 当前 vs `495b7221`

### 6.1 新增文件

| 文件 | 用途 |
|------|------|
| `ep/include/uccl_gin/uccl_gin_handle.cuh` | `handle::UCCLGin`: Rail→UCCL D2H, Lsa/World→NCCL delegate |
| `ep/include/uccl_gin/uccl_gin_rail.cuh` | `rail_put` / `rail_put_tail_add` / `rail_red_add` |
| `ep/include/uccl_gin/resources.cuh` | `UCCLGinResources` POD + profiling counters |
| `ep/tests/uccl_gin_microbench/` | 独立 microbench: put + red_add vs 原生 NCCL GIN |
| `thirdparty/DeepEP-v2-d4f41e4/` | Vendored DeepEP V2 (upstream `d4f41e4`) |
| `ep/docs/uccl_gin_*.md` | 设计/性能/架构文档 |

### 6.2 proxy.cpp 修改

**异步 per-tail 依赖**:
- `PendingAtomicBatch` deque: tail atomic 入队,记录依赖的 inflight WRITE WR id
- `retire_inflight_write()`: WRITE CQE 完成时递减依赖 batch 的 `pending_writes`
- `progress_pending_atomics()`: 队首 batch 依赖满足时 post atomics
- `enqueue_atomics_ordered()`: 替代旧的 `flush_atomics()`,不再同步 drain

**Piggyback tail decode**:
- `rdma.cpp`: WRITE cmd 的 `atomic_val > 0` 时 piggyback tail delta + offset,
  复用已有 `WRITE_WITH_IMM` + `PackAtomicWithSeq` receiver reorder/apply 逻辑

**Finish dependency 收窄**:
- 只有 plain WRITE (`atomic_val == 0`) 进入 `atomic_dependency_wrs_`
- WRITE_WITH_IMM 的 count delta 和 finish ATOMIC 共享 per-tail sequence,
  receiver 按 seq 顺序 apply,所以 sender 端不需要额外依赖

**Profiling**:
- `profile_commands_`: 拆解 poll/progress/post 耗时
- `piggyback_atomic_write_cmds`: 统计 payload WR 中携带的 tail count

### 6.3 hybrid_dispatch.cuh 修改 (compact channel staging)

**Buffer 重索引**:
```
原来: scaleout_send_buffer = BufferLayout(token_layout, 1, kNumMaxTokensPerRank, ...)
      send_buffer[token_idx]  ← 稀疏,dst 穿插

现在: constexpr int kNumCompactSendTokens = kNumChannels * kNumMaxTokensPerChannel;
      scaleout_send_buffer = BufferLayout(token_layout, 1, kNumCompactSendTokens, ...)
      send_buffer[channel][compact_slot]  ← compact,同 dst 连续
```

**Scaleout warp compact batch**:
```cpp
constexpr int kUCCLGinCompactChunkTokens = 4;           // chunk 目标
const int remote_scaleout_rank_idx = scaleout_rank_idx ^ 1;  // EP8x2: 唯一 remote dst

// compact slot = exchange(tail, remote_dst) → V2 recv slot, monotonic 递增
const int compact_remote_slot_idx = ptx::exchange(stored_scaleout_tail, remote_scaleout_rank_idx);

// TMA store 到 compact slot (同 dst 连续)
tma_store_1d(send_channel_buffer[compact_remote_slot_idx], ...);

// 攒 batch
if (compact_batch_count >= kUCCLGinCompactChunkTokens)
    flush_compact_remote_batch();
```

**Flush: 一条大 WRITE + piggyback tail**:
```cpp
flush_compact_remote_batch():
    gin.rail_put_tail_add(
        recv[first_slot],              // 目标连续 (V2 expanded layout)
        send_channel[first_slot],      // 源连续 (compact staging)
        count * token_bytes,           // 4 × ~14KB ≈ 56KB for FP8 hidden=7168
        remote_scaleout_rank_idx,
        channel_idx, scaleout_rank_idx,
        count);                        // tail delta = chunk token count (1..255)
    // finish flag 单独走 rail_tail_add(count=0, finish=true)
```

**Forward warp metadata readiness**:
```cpp
// 每个 slot 在消费前检查 src_token_global_idx
comm::timeout_while([&]() {
    observed = ld_acquire_sys(token_buffer.get_src_token_global_idx_ptr());
    ready = (observed / kNumMaxTokensPerRank == expected_rank) && (observed > old);
    ...
});
```

### 6.4 vendored DeepEP V2 注入 (`get_native_v2_resources`)

在 `csrc/elastic/buffer.hpp::ElasticBuffer` 新增方法:
```
get_native_v2_resources() → pybind11::dict {
    workspace_bytes, buffer_bytes, cpu_buffer_bytes,
    workspace_ptr, buffer_ptr, rdma_workspace_ptr,
    mapped_host_workspace_ptr, host_workspace_ptr,
    nccl_dev_comm_ptr, nccl_window_ptr
}
```
暴露 symmetric window 基地址、MR 原地址和 NCCL handle 给 UCCL-GIN proxy 层,
用于 `ibv_reg_mr` 和 window offset 计算。

---

## 7. 关键数据流详解

### 7.1 Dispatch 全链路

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    DeepEP V2 Dispatch — compact + piggyback tail                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  ┌─ GPU SM [scaleout warp, channel c] ──────────────────────────────────────────┐   ║
║  │                                                                              │    ║
║  │  for each token:                                                             │    ║
║  │    TMA load token → smem                                                     │    ║
║  │    dedup → dst_rank, slot                                                    │    ║
║  │    if dst is remote:                                                         │    ║
║  │      compact_slot = exchange(tail, remote_dst)  ← V2 已分配的 recv slot      │    ║
║  │      TMA store → send[channel][compact_slot]    ← compact,同 dst 连续!       │    ║
║  │      batch.count++                                                           │    ║
║  │      if batch.count == 4:                                                    │    ║
║  │        flush:                                                                │    ║
║  │          rail_put_tail_add(send[channel][first], recv[first],                │    ║
║  │                            4×14KB, dst, ch, src, count=4)                    │    ║
║  │            │                                                                 │    ║
║  │            ▼                                                                 │    ║
║  │        TransferCmd{WRITE, bytes=4×14KB, atomic_val=4,                        │    ║
║  │                    atomic_offset=tail_byte_off}                               │    ║
║  │        → D2H ring: 一条 WRITE cmd 同时带 payload + tail delta                │    ║
║  │                                                                              │    ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                          │                                           ║
║                          D2H ring buffer (一条 cmd ≈ 4 token payload)                 ║
║                                          │                                           ║
║  ┌─ CPU Proxy thread ───────────────────────────────────────────────────────────┐   ║
║  │                                                                              │    ║
║  │  cmd.atomic_val > 0:                                                         │    ║
║  │    → ibv_post_send(WRITE_WITH_IMM,                                           │    ║
║  │        imm=PackAtomicWithSeq(count=4, offset=tail_byte_off, seq))            │    ║
║  │    一条 EFA WRITE_WITH_IMM: payload 4 token + imm 带 tail delta              │    ║
║  │    → inflight_write_wrs_ (retirement tracking)                               │    ║
║  │    → 因为 atomic_val != 0, 不进入 finish dependency                          │    ║
║  │                                                                              │    ║
║  │  cmd.atomic_val == 0:                                                        │    ║
║  │    → ibv_post_send(RDMA_WRITE)  ← 纯 payload,无 tail                         │    ║
║  │    → atomic_dependency_wrs_  ← 需要等 CQE 才能发 finish                      │    ║
║  │                                                                              │    ║
║  │  cmd.cmd_type == ATOMIC:                                                     │    ║
║  │    → enqueue_atomics_ordered()  ← 依赖 plain WRITE CQE 全部完成后才 post     │    ║
║  │    → ibv_post_send(WRITE_WITH_IMM,                                           │    ║
║  │        imm=PackAtomicWithSeq(delta, offset, seq))  ← finish flag,独立 tail   │    ║
║  │                                                                              │    ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                          │                                           ║
║                              EFA verbs (WRITE_WITH_IMM)                               ║
║                                          │                                           ║
║  ┌─ Node 1 (receiver) ──────────────────────────────────────────────────────────┐   ║
║  │                                                                              │    ║
║  │  WRITE_WITH_IMM → NIC DMA 写入 payload + receiver CPU 收到 imm               │    ║
║  │    → CPU proxy 解码 PackAtomicWithSeq(seq, offset, delta)                    │    ║
║  │    → 如果 seq 乱序,暂存 reorder buffer (深度浅,绝大多数按序)                 │    ║
║  │    → seq 就绪时: atomicAdd(atomic_tail_base[offset], delta)                  │    ║
║  │                                                                              │    ║
║  │  GPU SM [forward warp, same channel c]:                                       │    ║
║  │    ld_acquire_sys(rail_tail_ptr) → signaled tail (count + finish bit)        │    ║
║  │    for slot in [old_tail, new_tail):                                         │    ║
║  │      ① metadata readiness: spin on src_token_global_idx                      │    ║
║  │      ② TMA store wait (NIC DMA 长尾)                                         │    ║
║  │      ③ TMA load token → smem                                                 │    ║
║  │      ④ route → scaleup rank (NVLink/本地)                                    │    ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 8. Compact Channel Staging: 从 per-token WRITE 到 multi-token chunk

### 8.1 问题

V2 的 `scaleout_send_buffer` 原先按 `token_idx` 索引 — 同 dst 的 token 在本地内存里
不连续。每条 `gin.put` 发 1 token(~14KB FP8),在 EFA 上只能打到 ~5 GB/s。

V1 每条 RDMA WRITE 发 16 token(~225KB),因为 `send_buffer[dst][slot]` 天然连续。

EFA 小包性能 microbench 数据 (gin_proxy_bench, 2 节点 × 1 GPU, 2 rails):

```
 4 KiB →  2.8 GB/s
 8 KiB →  5.1 GB/s
16 KiB →  8.8 GB/s
32 KiB → 12.5 GB/s
 1 GiB → 44.8 GB/s
```

### 8.2 设计

**同一个 `scaleout_send_buffer`,换索引方式。不新增 buffer,不新增 TMA store。**

```
原始 (sparse):
  TMA store → send_buffer[token_idx]           // dst 穿插
  gin.put(send_buffer[token_idx], recv[slot])  // 每 token 一条小 WRITE

Compact:
  compact_slot = exchange(tail, remote_dst)  // V2 recv slot, monotonic
  TMA store → send_buffer[channel][compact_slot]  // 同 dst 连续!
  攒够 4 token:
    rail_put_tail_add(send[channel][first], recv[first],
                      4 × token_bytes, dst, ch, src, count=4)
    // 一条大 WRITE = 一条 D2H cmd
```

**EP8x2 关键简化**: 每个 rank 只有一个 remote scaleout dst
(`remote_scaleout_rank_idx = scaleout_rank_idx ^ 1`)。local dst 走 bypass
(TMA store 直接到 recv buffer),不占 send buffer。所以 `send[channel][slot]` 里
所有 token 都去同一个 remote dst,slot index monotonic 递增,天然连续。

**Buffer 大小不变**: `kNumChannels × kNumMaxTokensPerChannel ≈ kNumMaxTokensPerRank`,
和原来 `1 × kNumMaxTokensPerRank` 相近 (带 compact padding)。

### 8.3 为什么 chunk size 是 4 而不是 32

Chunk sweep 实测结果 (EP8x2, tokens=8192, hidden=7168, FP8):

```
tokens/chunk   cache dispatch SO BW   dispatch_impl latency
64             ~27 GB/s               2.23-2.31 ms
32             ~31-32 GB/s            1.93-2.00 ms
16             ~33-34 GB/s            1.79-1.83 ms
 8             ~35-36 GB/s            1.70-1.73 ms
 4             ~37-38 GB/s            1.59-1.64 ms  ← 最优
 2             ~32 GB/s               1.90-1.93 ms
```

4-token 是当前最优平衡:chunk 再大,count/tail update 推迟,forward warp 更晚知道
payload 可消费,overlap 损失超过 WR 减少的收益。

### 8.4 Batch flush 逻辑

```cpp
// 每 token:
if batch_count == 0:
    first_slot = compact_slot
elif first_slot + batch_count != compact_slot:
    flush()  // 不连续 → 前一批结束
    first_slot = compact_slot
batch_count++
if batch_count >= kUCCLGinCompactChunkTokens:  // = 4
    flush()

// Channel 结束时:
flush(finish=true)  // 残余 batch + finish flag

// 注意: EP8x2 下 local bypass 不占 send slot,slot 天然连续,
// NonContig flush 极少发生 (profile 显示 ~0)
```

---

## 9. Piggyback tail: 消除独立 tail WRITE_WITH_IMM

### 9.1 问题

如果每个 chunk 都要一条独立 tail ATOMIC,proxy 每轮多跑一条
`ibv_post_send` + receiver CQE + software atomic apply。

### 9.2 设计: 复用 16B TransferCmd 的 fields

`rail_put_tail_add` (device side):

```cpp
// uccl_gin_rail.cuh
TransferCmd cmd{};
cmd.cmd_type = make_cmd_type(WRITE, ...);
cmd.bytes = bytes;
cmd.atomic_val = static_cast<uint8_t>(count_delta);  // 1..255
cmd.req_lptr = local_off_shifted;
cmd.req_rptr = remote_off_shifted;
cmd.atomic_offset = static_cast<uint16_t>(atomic_byte_off);
```

Proxy side (rdma.cpp):

```cpp
// cmd.atomic_val > 0 → 发 WRITE_WITH_IMM + PackAtomicWithSeq
if (cmd.atomic_val > 0) {
    imm = AtomicsImm::PackAtomicWithSeq(
        cmd.atomic_val, cmd.atomic_offset, seq, true);
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
}
```

一条 EFA WRITE_WITH_IMM = payload(4 token) + imm(tail delta + byte offset + sequence)。
Receiver proxy 解码 imm,reorder buffer 按 seq 排序后 apply atomicAdd。

### 9.3 与 finish 的 sequence 共享

Piggyback WRITE_WITH_IMM 的 count delta 和后续的 finish ATOMIC 打到
**同一个 per-(channel, src_rank) tail counter**。两者共享 receiver 端 sequence
计数器。因为 receiver 按 seq apply:
1. count delta 先 apply (在 payload WRITE 的 imm 中)
2. finish delta 后 apply (独立的 ATOMIC,更大 seq)
3. 中间如有乱序,reorder buffer 缓冲

所以 sender 端:只有 `atomic_val == 0` 的 plain WRITE 才需要追踪 completion
dependency。Piggyback WRITE_WITH_IMM 不需要。这使得 dependency_max 从 72 降到 2。

---

## 10. Sender-side completion dependency: 在 EFA 上实现 payload-before-tail

### 10.1 问题

EFA SRD 不保证多个 RDMA WRITE 按发出顺序到达。如果 finish ATOMIC 比某条 payload
WRITE 先到,receiver 可能读到 finish flag 后立即消费 slot,但 slot 里的 payload 还未
落盘 → 读到垃圾。

NCCL 用 FORCE_SO signals MR (NIC 级强序) 解决。EFA 不支持。

### 10.2 双层保证

**Layer 1: Sender-side async per-tail dependency** (proxy.cpp)

```
WRITE W0..W31 posted → inflight_write_wrs_

Piggyback WRITE_WITH_IMM (atomic_val > 0):
  → 不进入 atomic_dependency_wrs_  ← 与 finish 共享 sequence

Plain WRITE (atomic_val == 0):
  → atomic_dependency_wrs_.push_back(wr_id)

Finish ATOMIC:
  → enqueue_atomics_ordered():
      batch = {pending_writes = |dependency_wrs_|}
      for each wr_id: atomic_dep_by_wr_[wr_id] = &batch
      pending_atomics_.push_back(batch)

CQ poll:
  retire_inflight_write(wr_id):
    batch = atomic_dep_by_wr_[wr_id]
    if (--batch->pending_writes == 0)
      → post finish ATOMIC   ← 所有 payload CQE 已回
```

关键:后续 WRITE 继续 post,不受 finish dependency 阻塞。旧的同步 drain 会等所有
WRITE CQE 才能发下一个 batch,新路径去掉了这个瓶颈。

**Layer 2: Receiver-side per-slot metadata readiness** (hybrid_dispatch.cuh)

```cpp
// tail 公布 slot range; 每个 slot 独立验证 payload 落地
observed = ld_acquire_sys(token_buffer.get_src_token_global_idx_ptr());
ready = (observed / kNumMaxTokensPerRank == expected_rank) && (observed > old);
```

UCCL-EP 论文 §3.3 Figure 7 确认 receiver-side ordering 优于 sender-side
CQE drain(sender 端等 CQE 多一个 RTT)。

### 10.3 与 NCCL FORCE_SO 的等价性

```
NCCL FORCE_SO:
  NIC 保证 SIGNAL 在 payload WRITE 到达后可见
  → receiver GPU 信任 signaled tail

UCCL-GIN 等价:
  Sender: tail 在 payload WR CQE 完成后才 post (Layer 1)
  Receiver: tail 公布 slot range,per-slot metadata 证明 payload 已落地 (Layer 2)
  → receiver GPU 信任 signaled tail + per-slot check
```

### 10.4 D2H inflight cap

`kUCCLGinMaxInflightNormal` 限制每个 D2H ring 内同时 inflight (已 commit 但未 ack)
的命令数量。防止 GPU 侧过快灌满 ring + proxy 来不及 drain。当前值:
```
UCCL_GIN_MAX_INFLIGHT_NORMAL = 8  (编译时宏,可通过 Makefile 调整)
```

Sweep 已证明 cap=0 (无限) 无改善,cap=8 仅用于 sequence 安全 (4-bit seq 最多 16 条
inflight 不 wrap) 和可控背压。

---

## 11. Combine 迁移

Combine 与 dispatch 共享同一个 UCCL-GIN transport 底座:

```
GPU combine forward warp
  → replay token_metadata_at_forward
  → reduce / TMA store 到 V2 scaleout send buffer
  → 每个 remote token gin.put<Rail> → rail_put → D2H WRITE cmd
  → 每 channel/source 结束发 standalone finish ATOMIC

CPU proxy
  → plain payload WRITE 进入 finish dependency tracking
  → finish 等这些 WRITE CQE 后发送
  → receiver 等 finish apply 后清 tail
```

Combine 与 dispatch 的关键差异:
- **没有 compact staging**: combine 的 emission order 导致 remote_contig=0%
  (P1.2 merge-opportunity profile 证实)
- **没有 piggyback tail**: 每条 token 独立 `gin.put<Rail>`,count 由独立的
  finish ATOMIC 传达
- **per-token WRITE 数量**: ~8155 条 vs dispatch compact 后的 ~2000 条

这解释了 combine 的 `28-30 GB/s SO` 低于 dispatch `37-38 GB/s SO`。combine 的
receiver-facing staging/layout-aware compact 是主要候选优化,但需要先量化
local copy 成本 vs 减少 WR 的收益 (§PT.2 of perf plan)。

---

## 12. 当前性能状态

### 12.1 已验证配置 (EP16, CUDA 13.0, NCCL 2.30.4, aws-ofi-nccl master)

```
dispatch:          37-38 GB/s SO, 1.60-1.64 ms
expanded dispatch: 37-38 GB/s SO, 1.61-1.64 ms
cached dispatch:   37-38 GB/s SO, 1.60-1.63 ms
combine:           28-30 GB/s SO, 3.96-4.16 ms
reduced combine:   ~31 GB/s SO, 3.75-3.82 ms
```

### 12.2 演进路线

| 版本 | dispatch SO BW | 关键改动 |
|------|---------------|---------|
| per-token gin.put (原始) | ~5 GB/s | 每 token 一条 WRITE |
| + async tail | ~8 GB/s | 删除同步 drain |
| + compact channel staging | 逐步提升 | 4-token chunk, per-channel buffer |
| + piggyback tail | ~38 GB/s | tail 嵌入 payload WRITE_WITH_IMM |
| + dependency narrowing | ~38 GB/s | dependency_max 72→2, ~2-4% |
| + inflight cap + cleanups | ~38 GB/s | 背压, profiling |

### 12.3 已知差距与方向

| 项目 | 当前 | V1 baseline | 方向 |
|------|------|-----------|------|
| dispatch SO BW | 37-38 GB/s | 59 GB/s | ready/tag/landing 解耦 payload 大小与 tail 可见性 |
| combine SO BW | 28-30 GB/s | — | receiver-facing staging,减少 per-token WR |
| EP 配置 | EP8x2 | 通用 | 泛化 compact state 到 >2 scaleout ranks |
| `put_value<Rail>` | `__trap()` | functional | P2 实现 |
| 跨迭代 tail race | 未解决 | — | gin.quiet() 或 epoch/double buffer |

### 12.4 UCCL-EP 论文对照

| 论文结论 | UCCL-GIN 现状 |
|----------|--------------|
| chunk RDMA + piggyback atomic | ✅ rail_put_tail_add |
| Receiver-side ordering | ✅ per-slot metadata readiness + proxy reorder buffer |
| Per-channel FIFO ordering | ✅ lane(channel_idx) |
| HT mode chunk = 16 tokens (default) | △ 4 tokens 最优 (EFA 约束) |
| LL mode token packing = future work | △ 方向一致,当前 compact 即为类似思路 |

---

## 附录: 关键常量

| 常量 | 值 | 来源 | 含义 |
|------|-----|------|------|
| `kUCCLGinCompactChunkTokens` | 4 | hybrid_dispatch.cuh | Compact batch 目标 |
| `kUCCLGinMaxInflightNormal` | 8 | common.hpp | D2H ring inflight cap |
| `kUCCLGinTailFinishDelta` | 8192 | uccl_gin_handle.cuh | Tail finish bit (1<<13) |
| `kUCCLGinTailCountMask` | 8191 | uccl_gin_handle.cuh | Tail count 掩码 |
| `kAtomicOffMask` | 0x1FFF (8191) | uccl_gin_rail.cuh | Ordered atomic offset 上限 |
| `kAtomicValueMax` | 16383 | uccl_gin_rail.cuh | Atomic delta 上限 (15-bit signed) |
| `kAtomicValueMin` | -16384 | uccl_gin_rail.cuh | Atomic delta 下限 |
| `kWriteAddrShiftNormal` | 2 | ring_buffer.cuh | WRITE offset 4-byte 移位 |
| `TransferCmd` 大小 | 16 bytes | ring_buffer.cuh | D2H 命令大小 |
| `kMaxSendAtomicValue` | 16383 | common.hpp | 最大 atomic 值 |
| `kAtomicBufferSize` | 81960 | common.hpp | Host atomic buffer 大小 |

---

*文档更新于 2026-06-08,基于当前 uccl-gin 分支代码状态。*
*参考: UCCL-EP 论文 (arXiv:2512.19849v2), 参考 commit `495b7221`, NCCL 源码 (`nccl/src/gin/`).*
