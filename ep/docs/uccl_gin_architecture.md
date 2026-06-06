# UCCL-GIN Architecture Document

## 从 `495b7221` (UCCL/EP V1) 到当前 `uccl-gin` (compact32 + piggyback tail)

本文档讲解三个独立子系统(NCCL-GIN, DeepEP V2, UCCL/EP V1),然后解释如何将它们
缝合为 UCCL-GIN,以及 compact32 + piggyback tail 如何把 dispatch 从 ~5 GB/s 推到
~30 GB/s。

---

## 目录

1. [背景: 三个子系统概览](#1-背景-三个子系统概览)
2. [NCCL-GIN: GPU 如何直接发起网络 IO](#2-nccl-gin-gpu-如何直接发起网络-io)
3. [DeepEP V2: MoE dispatch 的分层 warp 架构](#3-deepep-v2-moe-dispatch-的分层-warp-架构)
4. [UCCL/EP V1 (`495b7221`): CPU proxy + D2H ring 传输底座](#4-ucclep-v1-495b7221-cpu-proxy--d2h-ring-传输底座)
5. [UCCL-GIN 合成: 把 V2 的 scale-out 映射到 V1 的传输底座](#5-uccl-gin-合成-把-v2-的-scale-out-映射到-v1-的传输底座)
6. [修改清单: 当前 vs `495b7221`](#6-修改清单-当前-vs-495b7221)
7. [关键数据流详解](#7-关键数据流详解)
8. [Compact32: 从 1-token WRITE 到 32-token chunk](#8-compact32-从-1-token-write-到-32-token-chunk)
9. [Piggyback tail: 消除独立 tail WRITE_WITH_IMM](#9-piggyback-tail-消除独立-tail-write_with_imm)
10. [Ordering 保证](#10-ordering-保证)
11. [当前性能状态](#11-当前性能状态)

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
│   │  ~5 GB/s EFA  │          │  ~30 GB/s EFA│   │  ~120 GB/s    │          │
│   └──────┬───────┘          └──────┬───────┘   └──────────────┘          │
│          │                         │                                      │
│          ▼                         ▼                                      │
│   aws-ofi-nccl               UCCL CPU proxy                               │
│   libfabric/EFA              D2H ring + raw ibverbs/EFA                   │
│   GIN proxy                  compact32 chunk + piggyback tail             │
│   (FORCE_SO signals MR)      (async per-tail dep + receiver metadata)     │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心差异**: NCCL-GIN 靠 NIC 强序(FORCE_SO MR)发小消息;UCCL-GIN 靠 GPU 侧 compact
staging 发 32-token chunk + piggyback tail,把消息大小从 14KB 拉到 ~450KB。

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

### 2.4 NCCLGin::put\<Rail\> 和 red_add_rel\<Rail\>

```cpp
// put: 一次 RDMA WRITE
gin.put(TEAM_WORLD_RAIL(), dst_rank_idx,
        nccl_window, recv_offset,
        nccl_window, send_offset,
        num_bytes, remote_action, ...);

// red_add_rel: 对 window 内的 counter 做 atomic add
// 非 NVLink 可达 → gin.signal(VASignalAdd(window, offset, value))
// NVLink 可达     → ptx::red_add_rel_sys(dst_ptr, value)
```

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
│  │                       │  compact32 batch TMA store            │
│  │                       │  rail_put_tail_add(piggyback tail)    │
│  └──────────────────────┘                                        │
│                                                                   │
│  ┌──────────────────────┐                                        │
│  │  FORWARD warps        │  每个 warp = 一个 channel              │
│  │                       │                                        │
│  │                       │  读 signaled tail → 消费 recv slot    │
│  │                       │  per-slot metadata readiness check    │
│  │                       │  route → scaleup (NVLink/本地)        │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Buffer 布局

```
buffer (一段连续 GPU HBM):
  ├─ scaleup_buffer          = kNumScaleupRanks × kNumScaleoutRanks × kNumMaxTokensPerRank
  ├─ scaleout_send_buffer    = kNumChannels × kNumMaxTokensPerChannel     ← compact per-channel
  └─ scaleout_recv_buffer    = kNumScaleoutRanks × kNumChannels × kNumMaxTokensPerChannel
```

`scaleout_send_buffer` 从原来的 `1 × kNumMaxTokensPerRank`(token_idx 索引,稀疏)
改为 `kNumChannels × kNumMaxTokensPerChannel`(channel 索引,同 channel 内 compact 连续)。

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
     │(1B)    │(1B)    │(3B)    │(4B)    │/value  │_offset │
     │        │        │+atomic │        │(4B)    │/expert │
     │        │        │_val(1B)│        │        │_idx(2B)│
     └────────┴────────┴────────┴────────┴────────┴────────┘

CmdType: EMPTY=0, WRITE=1, ATOMIC=2, QUIET=3, BARRIER=4
```

Piggyback tail 复用 `atomic_val`(8-bit) 字段携带 count delta,`atomic_offset` 携带
tail buffer byte offset——不改变 16B ABI。

### 4.3 V1 的 chunked RDMA

V1 的 coordinator warp 等 sender warp 攒够 6-32 token 后,发一条大 RDMA WRITE:

```cpp
// internode.cu:1080-1106
size_t const num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
uccl::nvshmemi_ibgda_put_nbi_warp(..., num_bytes_per_msg, ...);
// EFA 路径: tail offset + count 嵌入同一 WR (piggyback)
```

V1 之所以能做到,是因为 `send_buffer[dst][slot]` 天然连续——sender warp 按 dst 写,
coordinator 按 dst 发。这是 V2 缺的东西,compact32 补回来了。

### 4.4 V1 的 epoch tag

V1 不依赖 NIC 排序。每个 token 的 metadata 里嵌入递增 epoch number,receiver
逐 token 自旋验证:

```cpp
// internode.cu:1268-1297
int expected_tag = ((i + 1) & 0xFFFFFF) << 8;
int raw = ld_acquire_sys_global(ptr);
if ((raw & 0xFFFFFF00) == expected_tag) {
    seen_bits = raw & 0xFF;  // 新鲜
} else {
    while (true) {           // 自旋直到 epoch 匹配
        raw = ld_volatile_global(ptr);
        if ((raw & 0xFFFFFF00) == expected_tag) break;
    }
}
```

---

## 5. UCCL-GIN 合成: 把 V2 的 scale-out 映射到 V1 的传输底座

### 5.1 handle::UCCLGin 的 team 路由

```
handle::UCCLGin {
    nccl_dev_comm, nccl_window;    // 兼容 NCCLGin 的构造参数
    NCCLGin nccl;                  // 组合: Lsa/World 仍走原生 NCCL-GIN
    UCCLGinResources res;         // Rail 后端资源
}

put<Rail>()         → rail_put / rail_put_tail_add  → D2H → proxy → EFA
put<Lsa>()          → nccl.put<Lsa>()               → 原生 NCCL NVLink
put<World>()        → nccl.put<World>()              → 原生 NCCL
rail_put_tail_add() → 一条 WRITE cmd + piggyback tail (atomic_val + atomic_offset)
rail_tail_add()     → 独立 tail ATOMIC cmd (finish flag 用)
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
每个条目 8 字节(int64_t),finish bit = `kUCCLGinTailFinishDelta = 8192`,count mask = 8191。

### 5.3 文件结构

```
ep/
├── include/
│   ├── uccl_gin/
│   │   ├── uccl_gin_handle.cuh       ← handle::UCCLGin (DeepEP 集成)
│   │   ├── uccl_gin_rail.cuh         ← rail_put / rail_red_add / rail_put_tail_add
│   │   └── resources.cuh             ← UCCLGinResources POD
│   ├── proxy.hpp                     ← PendingAtomicBatch, profile counters
│   ├── ring_buffer.cuh               ← TransferCmd, CmdType (V1 继承)
│   └── d2h_queue_device.cuh          ← d2hq::D2HHandle (V1 继承)
├── src/
│   ├── proxy.cpp                     ← async per-tail dep, piggyback decode, compact profile
│   ├── rdma.cpp                      ← WRITE_WITH_IMM + PackAtomicWithSeq + piggyback
│   └── uccl_ep.cc                    ← 资源初始化 + JIT 编译入口
└── docs/
    ├── uccl_gin_plan.md
    ├── uccl_gin_architecture.md      ← 本文档
    └── uccl_gin_compact_staging.md   ← compact32 设计讨论

thirdparty/DeepEP-v2-d4f41e4/         ← vendored DeepEP V2 (in-tree)
└── deep_ep/
    ├── include/deep_ep/
    │   ├── common/handle.cuh         ← NCCLGin (纯净)
    │   └── impls/
    │       └── hybrid_dispatch.cuh   ← 主 kernel (compact32 + piggyback patch)
    └── buffers/elastic.py            ← Python buffer (atomic_tail_base 分配)
```

---

## 6. 修改清单: 当前 vs `495b7221`

### 6.1 新增文件

| 文件 | 用途 |
|------|------|
| `ep/include/uccl_gin/uccl_gin_handle.cuh` | `handle::UCCLGin`: Rail→UCCL D2H, Lsa/World→NCCL delegate |
| `ep/include/uccl_gin/uccl_gin_rail.cuh` | `rail_put` / `rail_red_add` / `rail_put_tail_add` |
| `ep/include/uccl_gin/resources.cuh` | `UCCLGinResources` POD |
| `ep/docs/uccl_gin_architecture.md` | 本文档 |
| `ep/docs/uccl_gin_compact_staging.md` | compact32 设计讨论 |
| `thirdparty/DeepEP-v2-d4f41e4/` | Vendored DeepEP V2 (upstream `d4f41e4`) |

### 6.2 proxy.cpp 修改

**异步 per-tail 依赖**:
- `PendingAtomicBatch` deque: tail atomic 入队,记录依赖的 inflight WRITE WR id
- `retire_inflight_write()`: WRITE CQE 完成时递减依赖 batch 的 `pending_writes`
- `progress_pending_atomics()`: 队首 batch 依赖满足时 post atomics
- `enqueue_atomics_ordered()`: 替代旧的 `flush_atomics()`,不再同步 drain
- `drain_pending_atomics()`: QUIET/BARRIER 时阻塞排空所有 pending batch

**Piggyback tail decode**:
- `rdma.cpp`: WRITE cmd 的 `atomic_val > 0` 时 piggyback tail delta + offset,
  复用已有 `WRITE_WITH_IMM` + `PackAtomicWithSeq` receiver reorder/apply 逻辑

**Profiling**:
- `piggyback_atomic_write_cmds`: 统计 payload WR 中携带的 tail count update
- `semantic_remote_*`: 统计 per (ring,dst) 的 token 连续性

### 6.3 hybrid_dispatch.cuh 修改 (compact32 + piggyback)

**Buffer 重索引**:
```
原来: scaleout_send_buffer = BufferLayout(token_layout, 1, kNumMaxTokensPerRank, ...)
      send_buffer[token_idx]  ← 稀疏,dst 穿插
现在: scaleout_send_buffer = BufferLayout(token_layout, kNumChannels, kNumMaxTokensPerChannel, ...)
      send_buffer[channel][compact_slot]  ← compact,同 dst 连续
```

**Scaleout warp compact batch**:
```cpp
constexpr int kUCCLGinCompactChunkTokens = 32;       // chunk 目标
const int remote_scaleout_rank_idx = scaleout_rank_idx ^ 1;  // EP8x2: 唯一 remote dst

// compact slot = V2 已分配的 expanded recv slot
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
        count * token_bytes,           // ~450KB for 32 tokens
        remote_scaleout_rank_idx,
        channel_idx, scaleout_rank_idx,
        count);                        // tail delta = chunk token count
    // finish flag 单独走 rail_tail_add(0, finish=true)
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

---

## 7. 关键数据流详解

### 7.1 Dispatch 全链路 (compact32 + piggyback)

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    DeepEP V2 Dispatch — compact32 + piggyback tail                    ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  ┌─ GPU SM [scaleout warp, channel c] ──────────────────────────────────────────┐   ║
║  │                                                                              │    ║
║  │  for each token:                                                             │    ║
║  │    TMA load token → smem                                                     │    ║
║  │    dedup → dst_rank, slot                                                    │    ║
║  │    if dst is remote:                                                         │    ║
║  │      compact_slot = exchange(tail, remote_dst)  ← V2 已分配的 recv slot      │    ║
║  │      TMA store → send[channel][compact_slot]    ← compact 连续!              │    ║
║  │      batch.count++                                                           │    ║
║  │      if batch.count == 32:                                                   │    ║
║  │        flush:                                                                │    ║
║  │          rail_put_tail_add(send[channel][first], recv[first],                │    ║
║  │                            32×14KB, dst, ch, src, count=32)                  │    ║
║  │            │                                                                 │    ║
║  │            ▼                                                                 │    ║
║  │        TransferCmd{WRITE, bytes=32×14KB, atomic_val=32,                      │    ║
║  │                    atomic_offset=tail_byte_off}                               │    ║
║  │        → D2H ring: 一条 WRITE cmd 同时带 payload + tail delta                │    ║
║  │                                                                              │    ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                          │                                           ║
║                          D2H ring buffer (一条 cmd = 32 token)                       ║
║                                          │                                           ║
║  ┌─ CPU Proxy thread ───────────────────────────────────────────────────────────┐   ║
║  │                                                                              │    ║
║  │  cmd.atomic_val > 0:                                                         │    ║
║  │    → ibv_post_send(WRITE_WITH_IMM,                                           │    ║
║  │        imm=PackAtomicWithSeq(count=32, offset=tail_byte_off, seq))           │    ║
║  │    一条 EFA WRITE_WITH_IMM: payload 32 token + imm 带 tail delta             │    ║
║  │                                                                              │    ║
║  │  cmd.atomic_val == 0:                                                        │    ║
║  │    → ibv_post_send(RDMA_WRITE)  ← 纯 payload,无 tail                         │    ║
║  │                                                                              │    ║
║  │  cmd.cmd_type == ATOMIC:                                                     │    ║
║  │    → ibv_post_send(WRITE_WITH_IMM,                                           │    ║
║  │        imm=PackAtomicWithSeq(value, offset, seq))  ← finish flag,独立 tail   │    ║
║  │                                                                              │    ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                          │                                           ║
║                              EFA verbs (WRITE_WITH_IMM)                               ║
║                                          │                                           ║
║  ┌─ Node 1 (receiver) ──────────────────────────────────────────────────────────┐   ║
║  │                                                                              │    ║
║  │  WRITE_WITH_IMM → NIC DMA 写入 payload + receiver CPU 收到 imm               │    ║
║  │    → CPU proxy 解码 PackAtomicWithSeq(seq, offset, count)                    │    ║
║  │    → 如果 seq 乱序,暂存 reorder buffer                                       │    ║
║  │    → seq 就绪时: atomicAdd(atomic_tail_base[offset], count)                  │    ║
║  │                                                                              │    ║
║  │  GPU SM [forward warp, same channel c]:                                       │    ║
║  │    ld_acquire_sys(rail_tail_ptr) → signaled tail                             │    ║
║  │    for slot in [old_tail, new_tail):                                         │    ║
║  │      ① metadata readiness: spin on src_token_global_idx                      │    ║
║  │      ② TMA store wait (NIC DMA 长尾)                                         │    ║
║  │      ③ TMA load token → smem                                                 │    ║
║  │      ④ route → scaleup rank (NVLink/本地)                                    │    ║
║  └──────────────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 8. Compact32: 从 1-token WRITE 到 32-token chunk

### 8.1 问题

V2 的 `scaleout_send_buffer` 原先按 `token_idx` 索引——token 0(dst=1) → slot 0,
token 1(dst=0) → slot 1,同 dst 的 token 在本地内存里**从来不连续**。每条
`gin.put` 发 1 token(~14KB),在 EFA 上只能打到 ~5 GB/s。

V1 每条 RDMA WRITE 发 6-32 token(~84-450KB),因为 `send_buffer[dst][slot]` 天然
连续。UCCL-EP 论文 §3.3 明确说 HT 模式 chunk 典型值是 **32 tokens**。

EFA 小包性能 microbench 数据:

```
 4 KiB →  2.8 GB/s
 8 KiB →  5.1 GB/s
16 KiB →  8.8 GB/s
32 KiB → 12.5 GB/s
 1 GiB → 44.8 GB/s
```

当前 14KB/token 在曲线底部;32 token × 14KB ≈ 450KB 进入大消息区间。

### 8.2 设计

**同一个 `scaleout_send_buffer`,换索引方式。不新增 buffer,不新增 TMA store。**

```
当前 (sparse):
  TMA store → send_buffer[token_idx]           // dst 穿插
  gin.put(send_buffer[token_idx], recv[slot])  // 每 token 一条小 WRITE

Compact32:
  compact_slot = exchange(tail, remote_dst)  // V2 已分配的 recv slot,monotonic
  TMA store → send_buffer[channel][compact_slot]  // 同 dst 连续!
  攒够 32 token:
    rail_put_tail_add(send[channel][first], recv[first],
                      32 × token_bytes, dst, ch, src, count=32)
    // 一条大 WRITE = 一条 D2H cmd
```

```
scaleout_send_buffer 分区:

  ┌────────────────────────────────────────────────────────────┐
  │              scaleout_send_buffer                           │
  │        (kNumChannels × kNumMaxTokensPerChannel tokens)      │
  │                                                             │
  │  channel 0:  [slot 0][slot 1][slot 2]...[slot N-1]         │
  │              └── compact, 同 dst 连续 ──┘                   │
  │  channel 1:  [slot 0][slot 1][slot 2]...[slot N-1]         │
  │  ...                                                        │
  └────────────────────────────────────────────────────────────┘
```

**EP8x2 关键简化**: 每个 rank 只有一个 remote scaleout dst
(`remote_scaleout_rank_idx = scaleout_rank_idx ^ 1`)。local dst 走 bypass
(TMA store 直接到 recv buffer),不占 send buffer。所以 `send[channel][slot]` 里
所有 token 都去同一个 remote dst,`stored_dst_slot_idx` 本身 monotonic 递增,
就是天然的 compact slot index。

**Buffer 大小不变**: `kNumChannels × kNumMaxTokensPerChannel ≈ kNumMaxTokensPerRank`,
和原来 `1 × kNumMaxTokensPerRank` 相近。

### 8.3 Batch flush 逻辑

```cpp
// 每 token:
if batch_count == 0:
    first_slot = compact_slot
elif first_slot + batch_count != compact_slot:
    flush()  // 不连续 → 前一批结束
    first_slot = compact_slot
batch_count++
if batch_count >= 32:
    flush()

// Channel 结束时:
flush(finish=true)  // 残余 batch + finish flag
```

`stored_dst_slot_idx` 在 EP8x2 下 monotonic 递增,所以 `first_slot + count == next_slot`
基本总是成立。不连续的情况只出现在中间有 local-dst token 跳过 send buffer——但
EP8x2 下 local 和 remote 是不同 dst,local bypass 不消耗 send slot,所以 slot 序列
天然连续。

### 8.4 为什么不用 smem batch

H200 单 SM 228KB shared memory,16 warps/SM 同时工作。每个 token ~14KB,32 token
= 448KB,远超 SM 容量。compact staging 在 HBM 里做,利用已有 TMA store,零额外
GPU copy。

---

## 9. Piggyback tail: 消除独立 tail WRITE_WITH_IMM

### 9.1 问题

Compact32 之前,每 3 token 一条独立 tail ATOMIC(WRITE_WITH_IMM)。profile 显示
`atomic_cmds ≈ 155k` per proxy thread——每条都需要一次 `ibv_post_send` + receiver
CQE + CPU software atomic apply。

Compact32 减少了 WRITE cmd 数量(455k → ~3k),但如果不 piggyback,每个 chunk 仍需
一条独立 tail ATOMIC: ~3k 条。

### 9.2 设计: 复用 16B TransferCmd 的 atomic_val 字段

```
TransferCmd (16 bytes):
  WRITE cmd + piggyback tail:
    cmd_type  = WRITE
    bytes     = 32 × 14KB
    atomic_val = 32 (count delta, 1..255)
    atomic_offset = tail_byte_off (13-bit)
    req_lptr  = local window offset
    req_rptr  = remote window offset

  vs 独立 ATOMIC cmd (finish flag):
    cmd_type  = ATOMIC
    value     = finish_delta (8192 + 0)
    req_rptr  = tail_byte_off
    atomic_offset = 1 (ordered path)
```

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

一条 EFA WRITE_WITH_IMM = payload(32 token) + imm(tail delta + offset + seq)。
Receiver proxy 解码 imm,reorder buffer 按 seq 排序后 apply atomicAdd。

### 9.3 Profile 验证

```
                   compact32 only    compact32 + piggyback
                   ─────────────     ────────────────────
write_cmds:        ~956k              ~956k  (不变)
piggyback:         0                  ~952k  (count tail 并入 payload)
atomic_cmds:       ~952k              ~238k  (只剩 finish flag)
dispatch SO BW:    ~18 GB/s           ~30 GB/s
```

独立的 count tail WRITE_WITH_IMM 基本消除。剩余 `atomic_cmds` 主要是每个 channel
的 finish flag 更新。

---

## 10. Ordering 保证

### 10.1 双层保证

UCCL-GIN 没有 NCCL 的 FORCE_SO signals MR(NIC 强序),用双层保证重建等价的
payload-before-tail 语义:

**Layer 1: Sender-side async per-tail dependency** (proxy.cpp)

```
WRITE W0..W31 posted → inflight_write_wrs_
                         atomic_dependency_wrs_ = [W0..W31]

Piggyback tail ATOMIC:
  batch.pending_writes = count of still-inflight WRITEs
  atomic_dep_by_wr_[W0] = &batch
  ...

  后续 WRITE 继续 post,不受阻碍  ← 与早期同步 drain 不同

CQ poll:
  W0 acked → retire_inflight_write(W0) → batch.pending_writes--
  ...
  pending_writes == 0 → post atomic batch
```

**Layer 2: Receiver-side metadata readiness** (hybrid_dispatch.cuh)

```cpp
// tail 公布 slot range; 每个 slot 独立验证 payload 可见性
observed = ld_acquire_sys(token_buffer.get_src_token_global_idx_ptr());
ready = (observed / kNumMaxTokensPerRank == expected_rank) && (observed > old);
```

UCCL-EP 论文 §3.3 Figure 7 确认 receiver-side ordering 优于 sender-side
CQE drain(sender 端等 CQE 多一个 RTT)。

### 10.2 与 NCCL FORCE_SO 的等价性

```
NCCL FORCE_SO:
  NIC 保证 SIGNAL 在 payload WRITE 到达后可见
  → receiver GPU 信任 signaled tail

UCCL-GIN 等价:
  Sender: tail 在 payload WR CQE 完成后才 post (Layer 1)
  Receiver: tail 公布 slot range,per-slot metadata 证明 payload 已落地 (Layer 2)
  → receiver GPU 信任 signaled tail + per-slot check
```

---

## 11. 当前性能状态

### 11.1 已验证配置 (EP8x2, CUDA 13.0, NCCL 2.30.4, aws-ofi-nccl master)

| 版本 | dispatch SO BW | write_cmds | atomic_cmds | 说明 |
|------|---------------|------------|-------------|------|
| per-token gin.put | ~4-8 GB/s | ~455k | ~155k | 每条 token 独立小 WRITE |
| per-token + async tail | ~8 GB/s | ~455k | ~155k | 删除了同步 drain |
| compact32 | ~18 GB/s | ~956k | ~952k | 32-token chunk,独立 tail |
| compact32 + piggyback | **~30 GB/s** | ~956k | ~238k | tail 嵌入 payload WRITE |

### 11.2 已知差距

| 项目 | 当前 | 目标 |
|------|------|------|
| dispatch SO BW | ~30 GB/s | ~44 GB/s (EFA 理论上限) |
| combine SO BW | ~7-11 GB/s | 30+ GB/s (待 compact/piggyback) |
| `AggregateRequests` | 被丢弃 | proxy payload coalescing |
| EP 配置 | EP8x2 only | EP16, EP24+ |
| `put_value` | `__trap()` | 实现(notify count path) |
| `red_add_rel<Rail>` | `__trap()` | compact-index API 替代 |
| `kAtomicOffMask` channel 上限 | ~511 channels | 全 H200 132SM |
| 跨迭代 tail race | `atomic_tail_base` clear 不保证 proxy 已 drain | gin.quiet() 或 host-side drain |

### 11.3 UCCL-EP 论文对照

| 论文结论 | UCCL-GIN 现状 |
|----------|--------------|
| HT mode chunk = 32 tokens | ✅ compact32 = 32 tokens |
| Write + piggyback atomic | ✅ `rail_put_tail_add` |
| Receiver-side ordering | ✅ metadata readiness + proxy reorder buffer |
| Per-channel FIFO ordering | ✅ `lane(channel_idx)` |
| LL mode token packing = future work | 本设计就是 LL 粒度的 HT-style chunking |

---

## 附录: 关键常量

| 常量 | 值 | 来源 | 含义 |
|------|-----|------|------|
| `kUCCLGinCompactChunkTokens` | 32 | hybrid_dispatch.cuh | compact batch 目标 |
| `kUCCLGinTailFinishDelta` | 8192 | uccl_gin_handle.cuh | Tail finish bit |
| `kAtomicOffMask` | 0x1FFF (8191) | uccl_gin_rail.cuh | Ordered atomic offset 上限 |
| `kAtomicValueMax` | 16383 | uccl_gin_rail.cuh | Atomic delta 上限 (15-bit) |
| `kWriteAddrShiftNormal` | 2 | ring_buffer.cuh | WRITE offset 4-byte 移位 |
| `TransferCmd` 大小 | 16 bytes | ring_buffer.cuh | D2H 命令大小 |
| `kMaxSendAtomicValue` | 16383 | common.hpp | Piggyback delta upper bound |
| piggyback `atomic_val` | 1..255 | uccl_gin_rail.cuh | 单次 piggyback count delta 上限 |

---

*文档更新于 2026-06-06,基于 compact32 + piggyback tail 代码状态。*
*参考: UCCL-EP 论文 (arXiv:2512.19849v2), 参考 commit `495b7221`, NCCL 源码 (`nccl/src/gin/`).*
