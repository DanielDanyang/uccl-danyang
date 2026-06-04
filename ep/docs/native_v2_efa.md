# Native DeepEP V2 on AWS EFA —— 集成 scaffold 详解

> 对应 commit `ecba87ae` *"Add native DeepEP V2 EFA integration scaffold"*
> 阅读前置：`AGENTS.md`（开发准则）、`plan.md`（完整计划）。
> 本文只解释 commit 里**实际落地的代码**，并配 ASCII 图说明数据流与内存模型。

---

## 0. 一句话概述

把 DeepEP V2 的 hybrid dispatch kernel **fork** 出来，仅替换它的
**scaleout（跨节点 / `ncclTeamTagRail`）GIN 调用**：原来用 NCCL GIN proxy 做的
RDMA，现在改成 **GPU 写旧 16B `TransferCmd` → D2H FIFO → UCCL CPU proxy → EFA verbs**。
**scaleup（节点内 / `ncclTeamTagLsa`）NVLink GIN 路径完全不动。**

```
            ┌─────────────────────── 不变 ───────────────────────┐
            │  scaleup (NVLink) 仍然走 NCCL GIN (ncclTeamTagLsa)   │
            └────────────────────────────────────────────────────┘
  V2 JIT kernel  ──►  替换 scaleout (EFA) GIN  ──►  旧 TransferCmd
  (hybrid_dispatch)        (ncclTeamTagRail)            │ D2H FIFO
                                                        ▼
                                              UCCL Proxy + EFA verbs
```

---

## 1. 文件改动清单

| 文件 | 行数 | 作用 |
|------|------|------|
| `.gitmodules` | +3 | 新增 submodule `thirdparty/DeepEP-v2-d4f41e4`（固定上游 commit） |
| `thirdparty/DeepEP-v2-d4f41e4` | +1 | submodule gitlink |
| `ep/Makefile` | +23 | 加 DeepEP V2 / fmt / NCCL include；把两个 V2 `.cc` 链进 `.so` |
| **C++ / CUDA（新增）** | | |
| `ep/include/v2_efa/hybrid_dispatch_native.cuh` | 799 | **核心**：fork 自上游 `hybrid_dispatch.cuh`，替换 scaleout GIN |
| `ep/include/v2_efa/hybrid_combine_native.cuh` | 620 | combine 的 fork，**当前与上游逐字节相同**（阶段 4 再改） |
| `ep/include/v2_efa/jit_plan.hpp` | 232 | 生成 JIT 源码 + launch plan（grid/warp/smem） |
| `ep/include/v2_efa/runtime.hpp` | 121 | `RuntimeConfig` / `V2EfaRuntime` / launch 声明 |
| `ep/include/v2_efa/topology.hpp` | 50 | `route_expert()`：expert → (owner_rank, scaleout, scaleup) |
| `ep/include/v2_efa/workspace.hpp` | 43 | signal-scratch 几何（`signal_scratch_slot_for`） |
| `ep/src/v2_efa_runtime.cc` | 114 | config 校验 + JIT plan 构建 |
| `ep/src/v2_efa_deep_ep_jit.cc` | 261 | JIT bridge：编译 + `launch_kernel(...)` |
| `ep/src/uccl_ep.cc` | +138 | nanobind 绑定：`d2h_queue_capacity` + `V2EfaRuntime` |
| **Python wrapper（新增）** | | |
| `ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py` | 893 | `ElasticBuffer`：生命周期 + `dispatch()` + 原生路径 |
| `ep/deep_ep_v2_wrapper/deep_ep/__init__.py` | 61 | 包入口、JIT init、把上游 `deep_ep` 挂到同名 namespace |
| `ep/deep_ep_v2_wrapper/deep_ep/utils/*` | 38 | `EventOverlap` 等 |
| `ep/tests/v2_efa_native_dispatch_smoke.py` | 146 | EP16 单 dispatch 正确性 + 粗吞吐 smoke test |
| `AGENTS.md` / `plan.md` | 189 / 959 | 记录与计划文档 |

> 注意：原 UCCL-EP **V1 路径完全保留**（`internode.cu` / `intranode.cu` /
> `deep_ep_wrapper/` / `thirdparty/DeepEP/` 都没动）。V2 是**并排新增**。

---

## 2. 整体架构（一次 dispatch 的端到端）

```
 ┌──────────────────────────── 发送端 GPU (sender) ─────────────────────────────┐
 │                                                                              │
 │  hybrid_dispatch_impl<...>  (1 个 kernel，每 SM 一个 block)                    │
 │  ┌────────────┐   ┌──────────────┐   ┌──────────────┐                        │
 │  │ notify warp│   │ scaleout warp│   │ forward warp │   (同一个 block 内)      │
 │  │  (SM0)     │   │ (per channel)│   │ (per channel)│                        │
 │  └─────┬──────┘   └──────┬───────┘   └──────┬───────┘                        │
 │        │ rank/expert     │ payload + tail   │ 消费本地 recv,                  │
 │        │ count           │                  │ 走 NVLink GIN 到 scaleup        │
 │        ▼                 ▼                  ▼  (← 不变)                        │
 │   ╔═══════════════════════════════════════════════╗                          │
 │   ║  本地分支 (dst==自己) : 直接 st_release_sys      ║   ← proxy 拒绝 self      │
 │   ║  远端分支 (dst!=自己) : 写旧 TransferCmd 到 D2H  ║                          │
 │   ╚════════════════════════╤══════════════════════╝                          │
 │                            │ d2h_queues[idx]->atomic_set_and_commit / reserve │
 └────────────────────────────┼─────────────────────────────────────────────────┘
                              │ host-pinned ring buffer (TransferCmd)
                              ▼
 ┌────────────────────────── CPU 侧 UcclProxy 线程 ─────────────────────────────┐
 │  poll_d2h() → 按 ring FIFO 顺序读，遇到 EMPTY 即 break（保序）                  │
 │  每个 ring rb_idx → 固定一个 data QP: data_qps_by_channel[rb_idx % n]          │
 │  post_rdma_async_batched()  → EFA SRD verbs WRITE (GPUDirect)                 │
 └────────────────────────────┬─────────────────────────────────────────────────┘
                              │ EFA RDMA WRITE (单一注册 MR)
                              ▼
 ┌──────────────────────────── 接收端 GPU (receiver) ───────────────────────────┐
 │  payload  → scaleout_recv_buffer[sender][channel][slot]                       │
 │  tail     → workspace.scaleout_channel_signaled_tail[channel][sender]         │
 │  count    → workspace.scaleout_{rank,expert}_count[sender]                    │
 │                            │                                                  │
 │  forward warp spin-wait: tail > old_tail (ld_acquire_sys)                     │
 │     → copy recv_buffer → scaleup_buffer (NVLink GIN, 不变)                     │
 │  dispatch_copy_epilogue kernel: scaleup_buffer → recv_x / recv_topk / ...     │
 └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心设计：scaleout GIN → 旧 `TransferCmd`

fork 只动 `hybrid_dispatch_native.cuh` 里 4 个 `ncclTeamTagRail` 调用点，
每个都拆成 **local（本地 scaleout rank）** 与 **remote（跨节点）** 两条分支。
原因：UCCL proxy 会**拒绝 self / intra-node 命令**（`proxy.cpp` 直接 `abort`），
所以本地那一份必须在 kernel 里用普通 release store 直接写本地 workspace。

| 原 GIN 调用 | 位置 | local 分支 | remote 分支 |
|-------------|------|-----------|-------------|
| `gin.put` rank_count | notify warp | `st_release_sys` 拷到本地 recv 槽 | `v2_d2h_write` TransferCmd |
| `gin.put` expert_count | notify warp | 同上 | 同上 |
| `gin.put` payload | scaleout warp | 早已有 `tma_store` 落本地 recv | `v2_d2h_write` TransferCmd |
| `gin.red_add_rel` tail | scaleout warp | `st_release_sys` 写本地 tail | scratch + `reserve/commit` |

### 3.1 三个新增 kernel 参数

```cpp
// hybrid_dispatch_impl(... 原有参数 ...,
    DeviceToHostCmdBuffer** d2h_queues,      // GPU 上的指针数组，元素是 host-pinned ring
    const uint32_t          num_d2h_queues,  // = 所有 proxy 线程的 channel 总数
    const uint64_t          signal_scratch_base);  // mapped scratch 基址（窗口尾部）
```

### 3.2 offset 编码：`v2_window_off`

旧 `TransferCmd` 只有 `req_lptr` / `req_rptr` 两个 **32-bit** 字段，没有 region
概念。V2 把 workspace+buffer+scratch 放进**同一个连续 NCCL symmetric window**，
所有地址都换算成「相对 window base 的偏移（右移 2 位，4 字节粒度）」：

```cpp
__device__ uint32_t v2_window_off(const void* ptr, uint64_t window_base) {
    return (uint64_t(ptr) - window_base) >> kWriteAddrShiftNormal;  // shift = 2
}
// kernel 里 window_base = (uint64_t)workspace  (mapped workspace 指针 = 窗口基址)
```

因为是 symmetric window，**同一个 offset 在本地（源）和远端（目的）都成立**，
所以 `req_lptr` 和 `req_rptr` 都用 `v2_window_off` 算即可。

> 16 GiB 约束：`offset = byteoff >> 2`，uint32 上限 `2^32-1` ⇒ 窗口 ≤ `2^34` B。
> Python 侧 `init_native_v2_efa_transport` 断言窗口 ≤ 16 GiB 且 4 字节对齐。

---

## 4. 内存模型：只注册设备段，scratch 放 GPU buffer 尾

DeepEP V2 的整个 symmetric window 是一段连续 VA `[Workspace | GPU buffer | CPU
buffer]`，但 **CPU/engram 段物理上是 NUMA/主机内存**（`symmetric.hpp` 用 CPU prop
`cuMemCreate`，只是 device-accessible，不是 HBM）。单个 GPUDirect MR **不能跨设备
显存 + 主机内存**，所以我们：

- **EFA MR 只注册设备段** `[raw_window_ptr, ws_bytes + gpu_bytes)`（纯 HBM）。
- **signal_scratch 从 GPU buffer 的设备尾部切**，留在 HBM 内、在同一个 MR 里。
- **CPU/engram 段完全不参与** native EFA path。

```
  raw_window_base (= rdma_workspace_ptr，EFA MR 注册基址)
  │   ┌──────────────── EFA MR = 设备段 (ws + gpu) ────────────────┐
  ▼   ▼                                                            ▼
  ┌───────────────────────┬──────────────────────────────────────────┬─────────────┐
  │      Workspace        │              GPU buffer                   │  CPU/engram │
  │   (ws_bytes, HBM)     │           (gpu_bytes, HBM)               │ (NUMA, 不注册)│
  │ ┌───────────────────┐ │ ┌──────────────────────┬───────────────┐ │ ┌─────────┐ │
  │ │ rank/expert count │ │ │ scaleup_buffer       │ signal_scratch│ │ │ engram  │ │
  │ │ channel tail[][]  │ │ │ scaleout_send_buffer │ (每 queue×ring│ │ │ (unused)│ │
  │ │ notify reduction  │ │ │ scaleout_recv_buffer │  槽一个 int64)│ │ │         │ │
  │ └───────────────────┘ │ │  ← DeepEP layout ←   │ ← 尾部预留 ←  │ │ └─────────┘ │
  │                       │ └──────────────────────┴───────────────┘ │             │
  └───────────────────────┴──────────────────────────────────────────┴─────────────┘
  ▲                       ▲                          ▲                ▲
  workspace_ptr      buffer_ptr            scratch_base         buffer_ptr+gpu_bytes
  (= window_base)    (=ws+workspace)   (=buf+gpu-scratch_bytes)
```

`elastic.py::init_from_deep_ep_v2` 强断言 `buffer_ptr == workspace_ptr + ws_bytes`
且 workspace/buffer 都落在设备窗口 `[workspace_ptr, workspace_ptr + ws + gpu)` 内；
`init_native_v2_efa_transport` 把 scratch 落在 `buffer_ptr + gpu_bytes -
scratch_bytes`（128 对齐）。

- **kernel `window_base` 用 mapped `workspace_ptr`**；scratch 也用 mapped 尾地址。
- **proxy 注册 MR 用 raw `rdma_workspace_ptr`**，长度 = ws + gpu。
- symmetric window 保证 mapped 与 raw 的内部布局 offset 一致，所以两边换算等价。
- **调用方责任**：DeepEP buffer 必须留出尾部 headroom（`num_bytes >=
  deepep_layout_bytes + scratch_bytes`），否则 DeepEP 的 layout 会顶进 scratch 尾。
  当前 smoke test 用 512MB GPU buffer，远大于小配置所需，天然有余量。

---

## 5. D2H 队列、signal scratch 与 ring slot 绑定

### 5.1 队列选择（lane 不进 command）

```
notify count :  q = dst_scaleout_rank_idx % num_d2h_queues
payload      :  q = channel_idx          % num_d2h_queues
tail         :  q = channel_idx          % num_d2h_queues   ← 与 payload 同队列
```

同一个 channel 的 payload 和 tail **永远进同一个 ring**，这是后面 ordering 的基础。

### 5.2 为什么 tail 需要 scratch + reserve/commit

旧 `TransferCmd` 是纯 WRITE，没有「立即数」字段。tail 是一个 GPU 算出来的
`int64` 值（`pack2(finish_flag, tail_count)`），必须先**落进已注册内存**才能被
RDMA 读走。于是给每个 ring slot 配一个 int64 scratch 槽，**生命周期与 ring slot
绑定**（slot 没被 proxy drain 之前不会复用 → scratch 也不会被覆盖）：

```
signal_scratch (在 GPU buffer 设备尾部, HBM, 同一 EFA MR 内)
┌──────────────── queue 0 ────────────────┬──────────── queue 1 ───────────┬ ...
│ slot0 │ slot1 │ slot2 │ ... │ slot 2047 │ slot0 │ slot1 │ ... │ slot2047 │
└───┬───┴───────┴───────┴─────┴───────────┴───────┴───────┴─────┴──────────┘
    │  signal_scratch_slot_for(base, q_idx, ring_slot & (cap-1), cap)
    │     = base + q_idx*cap + (ring_slot & (cap-1))      [cap = kQueueSize = 2048]
    ▼
  写 tail_word，再发 TransferCmd(req_lptr=该 scratch, req_rptr=远端 tail)
```

普通 payload 不需要 scratch（源数据本来就在已注册的 `scaleout_send_buffer`），
所以 payload 用现成的 `atomic_set_and_commit`；tail 用自定义的两段式
`v2_d2h_reserve_slot` + `v2_d2h_commit_slot`，只为了拿到「具体 ring slot 号」来
绑定 scratch。两条路径都只对 `head` 做 `atomicCAS`，互相安全。

```
payload:  atomic_set_and_commit(cmd)                         一步：抢 slot + 写 + 提交
tail   :  slot = reserve_slot()   ── 抢 slot（head++，槽仍 EMPTY）
          *scratch[slot] = tail_word
          commit_slot(slot, cmd)  ── 写 cmd（先 EMPTY → threadfence → 设真 cmd_type）
```

---

## 6. Dispatch warp 级数据流（scaleout / forward warp）

```
scaleout warp (channel = sm*kNumChannelsPerSM + warp)        每 token 循环:
────────────────────────────────────────────────────────
  load topk_idx[lane] → dst_scaleout_rank (= expert / experts_per_scaleout)
  TMA load x[token] → smem (tma_buffer)
  deduplicate ranks，分配 dst_slot
  TMA store smem → scaleout_send_buffer[token]        (本地暂存，EFA 读这里)
  ┌ 本地 rank: TMA store → scaleout_recv_buffer[slot] (直接落，不过网)
  └ 远端 rank: ── PAYLOAD ──────────────────────────┐
                v2_d2h_write(q=chan%nq,             │  ← 先入队（slot 较小）
                  src = send_buffer[token],         │
                  dst = recv_buffer[dst_slot])      │
  每 3 token 或结束: ── TAIL ──────────────────────┐│
    本地: st_release_sys(local_tail, word)         ││  ← 后入队（slot 较大）
    远端: scratch+reserve/commit(q=chan%nq, ...)   ▼▼
                                            同一 ring → 同一 QP → 保序

forward warp (同 channel)
────────────────────────────────────────────────────────
  spin-wait: channel_tail[channel][sender] > old_tail   (ld_acquire_sys)
  copy: scaleout_recv_buffer[sender][channel][slot] → scaleup_buffer
        (走 NVLink GIN ncclTeamTagLsa — 不变)
  build: token_metadata_at_forward, dst_buffer_slot_idx
  末尾: *channel_tail = 0   (Phase 1 保留清零；阶段 3 才去掉)
```

---

## 7. Ordering 保证（为什么 tail 看见时 payload 一定已到）

这是整个方案正确性的关键，由**三层**共同保证：

```
① kernel 程序顺序   : 同一迭代里 payload 在 581 行 tail 之前 enqueue
                      → payload 的 ring slot 号 < tail 的 ring slot 号
② proxy 严格保序    : poll_d2h() 按 ring FIFO 扫描，遇到第一个 EMPTY 槽就 break
   (proxy.cpp:766)    → 绝不会越过未提交的 slot 去处理后面的（即使后面已提交）
③ 单 ring 单 QP     : ring rb_idx → data_qps_by_channel[rb_idx % n] 固定一个 QP
   (rdma.cpp:1441)    → payload 与 tail 同 ring ⇒ 同 QP ⇒ EFA SRD 同 QP 保序
```

```
   ring (channel % nq)         一个 data QP (EFA SRD)
   ┌─────────────────┐          ─────────────────────►  时间
   │ slot k  payload │  ──post──►  WRITE payload
   │ slot k+1 tail   │  ──post──►  WRITE tail
   └─────────────────┘          同 QP 内严格有序：tail 落地时 payload 必已落地
```

> 阶段 5 的 TODO（`plan.md`）里提到的「同 channel 必须同 QP」在现有
> `data_qps_by_channel` 基础设施下**已经天然满足**，无需额外改动。

---

## 8. Code review 修复点（本 commit 已包含）

上一轮 `/code-review` 提出的 6 条，全部已修：

| # | 问题 | 修复 |
|---|------|------|
| 1 | scratch 大小用硬编码 `_V2_KQUEUE_SIZE=2048`，与 C++ `kQueueSize` 解耦，改了就静默越界 | 新增绑定 `ep.d2h_queue_capacity()`（返回 C++ `kQueueSize`），Python 经 `_d2h_queue_capacity()` 查询，单一真相源 |
| 2 | 没校验 buffer/scratch 在 window 内、buffer 紧跟 workspace；布局变了会静默写错 offset | `init_from_deep_ep_v2` 强断言：`buffer_ptr==workspace_ptr+ws_bytes`、`cpu_ptr==buffer_ptr+gpu_bytes`、三段都落在 `[base, base+window_bytes)` |
| 3 | smoke test `--perf` 循环缺每轮 `dist.barrier()`，Phase-1 去掉了开头 GPU barrier 后会出现跨轮 stale-tail | perf 循环内补 `dist.barrier()`（test:130） |
| 4 | kernel 模板参 `kNumQPs` 被错喂成 `num_lanes`，而 caller 的 `num_qps` 被丢弃 | `num_qps` 改由 `UCCL_V2_NUM_QPS`（默认 24，clamp 到 `num_allocated_qps`）决定；与 D2H lane 数解耦 |
| 5 | `transfer_layout.hpp` 里 `DispatchTransferLayout`/`CombineTransferLayout` 是死代码 | 整文件删除，`runtime.hpp` 去掉 include |
| 6 | Makefile `NCCL_INC` 用 `2>/dev/null` 静默吞错 | 改为 `$(error ...)`：解析不到 nccl 或找不到 `nccl.h` 直接报错 |

---

## 9. JIT 编译路径

```
Python init_deep_ep_jit(root, cuda_home, nccl_root)
   └─► ep.init_deep_ep_jit  (uccl_ep.cc)
         └─► v2::init_deep_ep_jit_bridge   (call_once)
               Compiler / KernelRuntime / IncludeParser prepare_init

dispatch() 第一次:
   build_native_hybrid_dispatch_jit_plan()  (runtime.cc → jit_plan.hpp)
     生成一段 .cu 源码:
        #include <deep_ep/common/*.cuh>
        #include "v2_efa/hybrid_dispatch_native.cuh"   ← 自身又 include ring_buffer.cuh
        static void __instantiate_kernel(){ &hybrid_dispatch_impl<模板实参...>; }
   compiler->build(name, source) → KernelRuntime（含 kernel handle）
   launch_kernel(kernel, config, 全部实参)   (v2_efa_deep_ep_jit.cc)
```

`jit_plan.hpp` 还负责算 grid / warp 划分 / smem / cluster：

```
grid_dim_x   = num_sms
num_threads  = (notify + scaleout + forward warps) * 32
  notify warps   = 4 (非 cached) / 0 (cached)
  scaleout warps = num_channels_per_sm
  forward warps  = num_channels_per_sm
cluster_dim  = 2 - (num_sms % 2)     cooperative = true
```

---

## 10. Python 层生命周期（`ElasticBuffer`）

```
wrap = ElasticBuffer(group, num_max_tokens_per_rank, hidden, num_topk)
wrap.init_from_deep_ep_v2(real_buf, num_lanes=4)
  │
  ├─ get_native_v2_resources()            ← 从真实 DeepEP V2 buffer 取指针/大小
  ├─ 断言 buffer 紧跟 workspace & 都在设备段内
  ├─ init_native_v2_efa_transport(...)
  │    ├─ 起 num_proxy_threads 个 ep.Proxy，每个注册设备段 [ws+gpu] 为 MR
  │    ├─ all_gather_object 交换 {rank, window base, listen_ports}
  │    ├─ set_peers_meta + register_proxies + start_dual   (建 QP)
  │    ├─ 收集所有 proxy 的 D2H channel 指针 → GPU int64 数组 (d2h_queues**)
  │    └─ 从 GPU buffer 设备尾部切 signal_scratch (大小 = num_q * cap * 8，cap 查 C++)
  └─ init_native_v2_deep_ep_resources(...) ← 存 nccl_dev_comm / window / buffer 指针

wrap.dispatch(x, topk_idx, topk_weights, num_experts, num_sms, do_expand)
  └─ _dispatch_native_hybrid()
       ├─ runtime.launch_native_hybrid_dispatch(...)        ← 主 kernel
       └─ runtime.launch_dispatch_copy_epilogue(...)        ← 写出 recv_x / recv_topk / metadata
```

`dispatch()` 仍然兼容 DeepEP 的签名（多余参数 `del` 掉），返回
`(recv_x, recv_topk_idx, recv_topk_weights, handle, event)`。

---

## 11. 当前状态 / 尚未完成

| 已完成（本 commit） | 未完成（后续阶段） |
|----------------------|---------------------|
| dispatch 主路径：scaleout GIN → TransferCmd | combine 原生化（`hybrid_combine_native.cuh` 仍是上游原版） |
| 设备段单 MR + GPU-tail scratch + offset 编码 | 阶段 2：多线程持久 proxy 调优 |
| signal scratch + ring-slot 绑定 | 阶段 3：去掉 pre-dispatch `dist.barrier()`（epoch tail） |
| ordering 三层保证 | 阶段 5：QP 数 / inflight / coalescing 性能调优 |
| EP16 单 dispatch smoke test | README-style ≥ 80 GB/s 性能达标 |
| 6 条 review 修复 | |

> ⚠️ **必须后续补：scratch tail 防撞精确 guard。** 当前只断言 scratch 能放进 GPU
> buffer（`elastic.py::init_native_v2_efa_transport`），并记录 `_v2_buffer_usable_bytes`，
> 但**没有检查 DeepEP dispatch/combine 的 BufferLayout 是否会用到这段 tail**。
> smoke test 用 512MB buffer 余量极大、可先这样；**大配置（大 hidden / 多 token /
> 多 channel）下若 buffer 偏紧，layout 会顶进 scratch tail → silent corruption。**
> 精确 guard 需要 DeepEP 的 layout 字节（`_C.calculate_elastic_buffer_size`，需要
> `nccl_comm` handle）——上线大配置前必须补。

> **判定标准**（见 `plan.md` 第八节）：dispatch ≥ 80 GB/s 接近完成，= 90 GB/s 目标完成。
> 本 commit 是把主路径骨架立起来，性能数尚未采。

---

## 12. 关键文件速查

```
ep/include/v2_efa/
  hybrid_dispatch_native.cuh   ← 改这里：4 个 scaleout call site + 3 个新参数
  workspace.hpp                ← signal_scratch_slot_for / signal_scratch_bytes
  jit_plan.hpp                 ← 生成 JIT 源码 & launch plan
  runtime.hpp / topology.hpp   ← 配置、路由、launch 声明
ep/src/
  v2_efa_deep_ep_jit.cc        ← JIT 编译 + launch_kernel 实参拼装
  v2_efa_runtime.cc            ← config 校验 + plan 构建
  uccl_ep.cc (+138)            ← nanobind: d2h_queue_capacity + V2EfaRuntime
ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py
                               ← 生命周期 + dispatch() + _dispatch_native_hybrid()
ep/include/ring_buffer.cuh     ← (复用) TransferCmd / DeviceToHostCmdBuffer
ep/src/proxy.cpp, rdma.cpp     ← (复用) D2H drain / EFA verbs / ring→QP
```
