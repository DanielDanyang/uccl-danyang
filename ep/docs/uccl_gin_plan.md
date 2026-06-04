# UCCL-GIN 计划：用一个 GIN 形状的 API 后端承载 DeepEP V2

> 目标:不再逐个 fork V2 kernel、逐个 call site 手抄 GIN 语义,而是提供一个
> **和 `handle::NCCLGin` 接口/语义一致的 `handle::UCCLGin`**,其 scale-out (`Rail`)
> 后端走 UCCL D2H + CPU proxy + EFA;scale-up (`Lsa`) 原样转发 NCCL/NVLink。
> 这样 DeepEP V2 (dispatch / combine / low-latency / 未来版本) 以最小改动跑在 AWS EFA 上。
>
> 前置阅读:`AGENTS.md`、`plan.md`、`ep/docs/native_v2_efa.md`、`worklog.md`。

---

## 0. 核心洞察:抽象边界已经存在

DeepEP V2 的每个 kernel **不直接调 `ncclGin`**,而是调一个薄封装
`deep_ep::elastic::handle::NCCLGin`(`deep_ep/common/handle.cuh`)。它已经按
`team_t`(`ncclTeamTagLsa` / `ncclTeamTagRail` / `ncclTeamTagWorld`)把语义分好了:

```
DeepEP V2 kernel (hybrid_dispatch.cuh / hybrid_combine.cuh / ...)
        │   只调这几个模板方法:
        │     gin.put<team_t>(recv, send, bytes, dst, opt, remote_action)
        │     gin.put_value<team_t>(ptr, value, dst)
        │     gin.red_add_rel<team_t>(ptr, value, dst)
        │     gin.get_sym_ptr<team_t>(ptr, dst)
        │     gin.signal/wait/flush/flush_async(...)
        ▼
   handle::NCCLGin            ← 抽象边界(已存在!)
        │  Lsa  → 直接 NVLink ptx (st/red_add_rel_sys) 或 ncclGin
        │  Rail → ncclGin.put / putValue / signal(VASignalAdd) → NCCL GIN proxy → IB
        ▼
   ncclGin (NCCL device runtime)
```

**所以我们要做的不是"在 kernel 里替换 call site"(过去几轮的痛苦来源),而是再写一个
同形状的 `handle::UCCLGin`,只替换 `Rail` 分支的后端。** kernel 调用面一字不改。

---

## 1. 现状 vs 目标

```
=== 现状(逐 call site fork,痛苦)========================================

  hybrid_dispatch.cuh  ──fork──►  hybrid_dispatch_native.cuh
                                     │  手动把每个 gin.put<Rail> 改成
                                     │  v2_d2h_write(...) / red_add 改成 PackAtomicWithSeq
                                     ▼
  问题:每个 call site 重抄一遍语义 → count/finish 拆词 bug、SRD 乱序、
        丢掉 AggregateRequests → 30 GB/s;combine/LL 还得再 fork 再抄一遍。

=== 目标(一个 GIN 后端,kernel 不动)====================================

  hybrid_dispatch.cuh (上游原版, 几乎不 fork)
        │   gin.put<Rail>(...) / red_add_rel<Rail>(...) ← 调用面不变
        ▼
  handle::UCCLGin   ← 新增,镜像 handle::NCCLGin 的方法签名
        │  Lsa  → 原样转发 (NVLink ptx / ncclGin)  ← 不碰
        │  Rail → UCCL 后端:
        │           put        → coalesced D2H TransferCmd WRITE
        │           red_add_rel→ PackAtomicWithSeq + receiver reorder
        │           put_value  → D2H WRITE (single word)
        │           signal/wait/flush → D2H + proxy completion/ack
        ▼
  D2H ring (TransferCmd ABI) → UcclProxy → EFA verbs
```

一次把 EFA 的硬骨头(无 atomic / 无序 / 小消息合并)在 `Rail` 后端**解决一次**,
dispatch / combine / LL 全部受益。

---

## 2. 要镜像的 API 表面(`handle::NCCLGin` → `handle::UCCLGin`)

| 方法 (DeepEP 调用面) | `Lsa` (NVLink) | `Rail` (EFA) 后端 | 现有零件 |
|---|---|---|---|
| `put<team>(recv,send,bytes,dst,opt,ra)` | 直发/NVLink | **coalesced** D2H WRITE(连续 run 合一条) | 单 token 版已有;**合并待做** |
| `put_value<team>(ptr,val,dst)` | `st_relaxed_sys` | D2H WRITE 单 word | 已有(count 写) |
| `red_add_rel<team>(ptr,val,dst)` | `red_add_rel_sys` | PackAtomicWithSeq + reorder apply | **已有**(tail) |
| `get_sym_ptr<team>(ptr,dst)` | NVLink 对端指针 | self/nullptr(EFA 不可直访) | offset 编码已有 |
| `signal<team>(dst,action)` | — | D2H + immediate(VASignalAdd 等价) | 部分(atomic imm) |
| `wait(req)` / `flush()` / `flush_async()` | — | proxy completion / `acked_wrs_` / quiet | 复用 proxy ack |
| barrier (`comm.cuh::gpu_barrier<Rail>`) | NVLink barrier | host `dist.barrier()`(Phase1) / epoch(Phase3) | 已有 |
| `remote_action`(piggyback,如 put 带 signal) | — | D2H WRITE + 紧随 tail atomic(同 ring 保序) | 部分 |

> 关键:表里**带「待做/部分」的就是性能与正确性的真正工程量**,而且都集中在
> `Rail` 后端这一处,而不是散在各 kernel。

---

## 3. 分层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│ L3  DeepEP V2 kernels (上游, 不 fork 或极小 fork)                       │
│     hybrid_dispatch / hybrid_combine / dispatch / low_latency ...      │
│     调用面: gin.put<team_t>(...) 等                                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ 编译期选择 gin 类型 (见 §6)
┌───────────────────────────────▼──────────────────────────────────────┐
│ L2  handle::UCCLGin  (新增, device __forceinline__, 镜像 NCCLGin)       │
│     template<team_t> put / put_value / red_add_rel / get_sym_ptr /     │
│                       signal / wait / flush                            │
│       if Lsa  → 转发 ncclGin / NVLink ptx     (零改动, 直接复用)         │
│       if Rail → 调 L1 device 后端                                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│ L1  UCCL Rail device 后端 (device 端, 纯 <cstdint>, JIT 可编)           │
│     transfer_cmd_device.cuh (D2H ring ABI) + uccl_gin_rail.cuh:        │
│       uccl_gin_put()          连续 run 合并 → 1 条 WRITE                │
│       uccl_gin_red_add_seq()  PackAtomicWithSeq                        │
│       uccl_gin_put_value()    单 word WRITE                            │
│       window offset 编码 (req_lptr/req_rptr = (addr-base)>>2)          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ host-pinned D2H ring (TransferCmd)
┌───────────────────────────────▼──────────────────────────────────────┐
│ L0  UcclProxy (host) + rdma.cpp + EFA verbs                            │
│     drain D2H → [coalesce 连续 WRITE] → post → CQ/ack → tail advance   │
│     ordered atomic apply (PackAtomicWithSeq reorder buffer)            │
│     completion → notify_gpu_completion (wait/flush 语义)               │
└────────────────────────────────────────────────────────────────────────┘
```

L0 大多已存在(V1 transport);L1 大多已存在(本项目这几轮写的);**新增主要是 L2
把它们包成 GIN 形状,以及 L1/L0 的 coalescing(AggregateRequests 等价物)。**

---

## 3.5 文件结构 + DeepEP vendoring

**DeepEP V2 已从 git submodule 改成 vendored 源码副本**(直接拷进仓库、可仓内修改;
见 `thirdparty/DeepEP-v2-d4f41e4/VENDORED.md`)。这样"在 DeepEP 里做极小 patch"就是
改我们自己的文件,不再有 submodule gitlink / `git submodule update` 的脆弱性。

核心原则:**thirdparty 里只留极小、可 re-vendor 的 patch;UCCL 全部后端留在 `ep/`。**
不再把整份 kernel fork 进 `ep/`(那是上一轮 800 行 `hybrid_dispatch_native.cuh` 的痛点)。

```
thirdparty/DeepEP-v2-d4f41e4/        ← vendored 副本 (不是 submodule), 只留极小 patch
  VENDORED.md                        [新] 上游 commit + 改动清单 + re-vendor 说明
  csrc/elastic/buffer.hpp            [已改] get_native_v2_resources()
  csrc/kernels/backend/api.cuh       [已改] get_raw_window_ptr()
  csrc/jit/{compiler,kernel_runtime}.hpp  [已改] JIT bridge
  deep_ep/include/deep_ep/
    common/handle.cuh                [小 patch] 让 gin 类型可替换 (DEEPEP_GIN_T)
    impls/hybrid_dispatch.cuh        [小 patch] 用 DEEPEP_GIN_T 而非硬写 NCCLGin
    impls/hybrid_combine.cuh         [小 patch] 同上
  third-party/fmt/                   [vendored] header-only, 一并拷入
thirdparty/DeepEP-v2-d4f41e4.local-changes.patch   [记录] 相对 pristine d4f41e4 的 diff

ep/                                  ← UCCL 拥有的全部后端 (clean git)
  include/v2_efa/
    uccl_gin.cuh           [新] namespace deep_ep::elastic::handle { struct UCCLGin }
                                镜像 NCCLGin 方法签名; Lsa 转发, Rail 调下面
    uccl_gin_rail.cuh      [新] L1 device 后端: put(coalesce)/red_add_seq/put_value/offset
                                (把现散在 hybrid_dispatch_native.cuh 的 v2_d2h_* 收进来)
    transfer_cmd_device.cuh[已有] D2H ring ABI (lean, JIT 可编)
    workspace.hpp          [已有] signal scratch / atomic tail 几何
    jit_plan.hpp           [改] 生成 source: include 上游 hybrid_dispatch.cuh + uccl_gin.cuh,
                                #define DEEPEP_GIN_T handle::UCCLGin 后实例化
    runtime.hpp / topology.hpp ...   [已有]
    hybrid_dispatch_native.cuh       [删] 800 行 fork → 由"上游 kernel + UCCLGin"取代
  src/
    proxy.cpp              [改] L0: 连续 WRITE coalescing (AggregateRequests 等价)
    rdma.cpp               [已有] EFA verbs + PackAtomicWithSeq apply
    v2_efa_deep_ep_jit.cc  [改] launch 资源喂给 UCCLGin 构造, 不再喂 fork kernel
    uccl_ep.cc / uccl_proxy.cpp  [已有] binding
  deep_ep_v2_wrapper/...   [已有] 生命周期 / dispatch 接线
  docs/uccl_gin_plan.md    [本计划] / native_v2_efa.md [fork 版历史参考]
```

thirdparty 那个"极小 patch"(让 kernel 用 UCCLGin)长这样:

```cpp
// deep_ep/impls/hybrid_dispatch.cuh 顶部
#ifndef DEEPEP_GIN_T
#define DEEPEP_GIN_T ::deep_ep::elastic::handle::NCCLGin   // 默认上游行为
#endif
...
const auto gin = DEEPEP_GIN_T(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
```
JIT 生成 source 时 `#define DEEPEP_GIN_T handle::UCCLGin` + `#include "v2_efa/uccl_gin.cuh"`。
re-vendor 时这种 ~3 行 patch 几乎不冲突,远好过维护 800 行 fork。

> 路径 `thirdparty/DeepEP-v2-d4f41e4` 保持不变 → `ep/Makefile`(`DEEPEP_V2_ROOT`)和
> Python `__init__` 零改动。`figures/`(README 图)未 vendored,无关构建。

---

## 4. 逐方法后端设计

### 4.1 `put<Rail>` —— 必须做 coalescing(这是 30→? 的关键)

上游 `gin.put<Rail>(..., ncclGinOptFlagsAggregateRequests)` 把"聚合"下放给 GIN 层。
我们的 GIN 层 = UCCL proxy,所以 **coalescing 落在 L1/L0,kernel 仍是 per-token put**。

```
kernel: 每 token 调 uccl_gin_put<Rail>(recv_slot, send_slot, token_bytes, dst)
                │  (调用面和上游一样, per token)
                ▼
两种合并位置(二选一或都做):

  (A) L0 proxy 侧合并 (推荐, 最贴 V2 的 "GIN 聚合在下层"):
      proxy drain 时, 把相邻命令里
        dst_rank 相同 && req_lptr 连续 && req_rptr 连续 && 同 flags
      的 run 合成 1 条 ibv WR (bytes 累加)。
      → kernel 不动, 改动集中在 post_rdma_async_batched 收集循环。

  (B) L1 kernel 侧合并 (更早, 参考 legacy/internode.cu 的滑动窗口):
      scaleout warp 攒连续 run, 一次 v2_d2h_write(bytes=run*token_bytes)。
      → 省 GPU 侧 per-token __threadfence_system, 但要改 scaleout warp。
```

推荐先做 (A):faithful、kernel 不动、改动一处。(B) 作为后续 GPU 侧优化。

### 4.2 `red_add_rel<Rail>` —— 有序软原子(已做,收进 API)

```
Lsa : ptx::red_add_rel_sys(dst, value)                 ← 不变
Rail: 把 value 走 PackAtomicWithSeq immediate:
        seq = next_seq_per(dst_rank, tail_word_index)++  (mod kReorderingBufferSize)
        D2H ATOMIC cmd (atomic_offset!=0) → proxy
      receiver proxy reorder buffer 按 seq 有序 apply 到本地 atomic buffer
```
约束(已加 static_assert):`ceil(M / interval) + 1 <= kReorderingBufferSize`。
这是 EFA 无硬件 atomic / 无序的"语义补齐",**收进 `uccl_gin.red_add_rel<Rail>`**。

### 4.3 `get_sym_ptr<team>` / `put_value<team>`

```
get_sym_ptr<Lsa> : 返回 NVLink 对端指针 (原样)
get_sym_ptr<Rail>: self→ptr, 否则 nullptr (EFA 不能直访远端 VA)
put_value<Lsa>   : st_relaxed_sys (原样)
put_value<Rail>  : D2H WRITE 单 word (notify count 已在用)
window offset    : req = ((addr - window_base) >> 2), 单一 symmetric window MR
```

### 4.4 `signal / wait / flush` —— 映射到 proxy completion

```
signal<Rail>(dst, action)   → D2H WRITE/ATOMIC (+piggyback remote_action)
wait(request)               → 自旋等 proxy 把对应 WR completion 标记 (acked_wrs_)
flush()/flush_async()       → quiet 语义: 等本 lane 所有 in-flight WR CQE 回来
                              (复用 ctx_.quiet_* / notify_gpu_completion())
```

### 4.5 barrier (`comm.cuh::gpu_barrier<Rail>`)

```
Phase 1: 删 Rail 开头 barrier, 用 host dist.barrier() (已做)
Phase 3: epoch tail, 去掉 host barrier
Lsa barrier: 保留 NVLink barrier 不动
```

---

## 5. "最小 kernel 改动"机制

DeepEP kernel 现在硬写 `const handle::NCCLGin& gin`。要做到 minimal,有三档:

```
档位 1 (最小, 理想): kernel 模板化 gin 类型
   template<class Gin> __global__ void hybrid_dispatch_impl(..., const Gin& gin)
   → 实例化时传 UCCLGin(Rail) / NCCLGin(Lsa-only)。调用面零改。
   依赖: 上游 kernel 把 gin 类型写成模板参数(可能要小 patch 构造处)。

档位 2 (折中, 当前可行): 极小 fork
   只 fork kernel 顶部 "构造 gin 的那几行" + 签名类型, 把 handle::NCCLGin 换成
   handle::UCCLGin; 所有 gin.put<...>/red_add_rel<...> call site 一字不改。
   → 比现在"逐 call site 改"少 ~95% 改动, combine/LL 同样套路。

档位 3 (最后手段): 保持现在的 native fork, 但把散落逻辑收进 UCCLGin 方法,
   call site 调 uccl_gin.put<Rail>(...) 而非 inline。
```

目标走 **档位 2**(现实、改动集中);若上游 kernel 能模板化则升到档位 1。

```
   handle::NCCLGin  ────────────►  handle::UCCLGin
   (Lsa+Rail 都走 NCCL)            (Lsa 转发 NCCL, Rail 走 UCCL)
                                   ▲
   kernel 只把 gin 的“类型/构造”换成 UCCLGin, call site 不变
```

---

## 6. `put<Rail>` 端到端数据流(目标)

```
GPU scaleout warp                  CPU UcclProxy (per ring/lane)        remote GPU
─────────────────                  ───────────────────────────         ──────────
for token in channel:
  TMA store x[token] → send_buf
  uccl_gin.put<Rail>(                D2H ring (TransferCmd WRITE)
    recv_buf[dst_slot],     ──push──►  drain in-order
    send_buf[token],                   ┌── coalesce 连续 run ──┐
    token_bytes, dst)                  │ dst 同 & off 连续 →   │
                                       │ 合成 1 条大 WR        │
  每 interval token:                   └──────────┬───────────┘
  uccl_gin.red_add_rel<Rail>(                     │ ibv post (signaled)
    tail_word, delta, dst)  ──push──►  ordered ATOMIC (seq) │
                                       reorder apply ────────┼──► EFA RDMA WRITE
                                                             ▼   payload → recv_buf
                                       CQ poll → acked_wrs_      tail   → atomic_buf
                                       → advance ring tail            ▼
                                       (slot/scratch 可复用)    forward warp:
                                                                spin tail, copy→scaleup
```

ordering 保证(已验证可用):同 channel 的 payload 与 tail 走**同 ring → 同 QP**;
tail 用 PackAtomicWithSeq 在 receiver 有序 apply,绝对值/乱序问题消除。

---

## 7. 实施阶段与依赖

```
P0  抽出 UCCL Rail device 后端 (L1)                       [基础]
    transfer_cmd_device.cuh (已有) + 新建 uccl_gin_rail.cuh:
      uccl_gin_put / red_add_seq / put_value / offset 编码
    把现在散在 hybrid_dispatch_native.cuh 的 v2_d2h_* 收进来
        │
        ▼
P1  handle::UCCLGin (L2)  ── 镜像 NCCLGin 方法签名         [核心抽象]
      Lsa 分支: 直接 #include 复用 NCCLGin 的 Lsa 实现 (ptx/ncclGin)
      Rail 分支: 调 P0 的后端
        │
        ▼
P2  kernel 接线 (档位 2):极小 fork hybrid_dispatch        [打通 dispatch]
      只换 gin 类型/构造; call site 不动
      回归: EP16 dispatch correctness (已有 smoke test)
        │
        ├────────────────────────────┐
        ▼                             ▼
P3  put<Rail> coalescing (L0 proxy)   P4  combine 套同一个 UCCLGin
    AggregateRequests 等价物              fork hybrid_combine 仅换 gin 类型
    目标: 30 → 接近 EFA 上限    → combine correctness 免费跟上
        │                             │
        └──────────────┬──────────────┘
                       ▼
P5  perf 调优 + Phase3 去 barrier (epoch tail)
    interval / coalesce 上限 / lane 数 sweep; README 风格 sweep
                       ▼
P6  (可选) low-latency 路径也套 UCCLGin → 全 DeepEP V2 覆盖
```

依赖关系:P0→P1→P2 是主干;P3(性能)和 P4(combine)在 P2 后并行;P5 收尾。

---

## 8. 现有工作如何归位(不是重写,是收敛)

```
现在散落的东西                        归位到 UCCL-GIN 哪一层
────────────────                      ──────────────────────
v2_d2h_write / atomic_set_and_commit  → L1 uccl_gin_put / put_value
PackAtomicWithSeq + reorder (tail)    → L1 uccl_gin_red_add_seq + L0 apply
单 window MR + offset 编码             → L1 offset helper
host dist.barrier (Phase1)            → L2 barrier<Rail>
proxy EFA normal WRITE / 软原子路径    → L0 后端 (已有)
transfer_cmd_device.cuh (lean ABI)    → L1 基础 (已有)
```

也就是说:**这几轮 debug 写的代码就是 UCCL-GIN 的 Rail 后端零件**,P0/P1 主要是把它们
从 "inline 在 fork 的 kernel 里" 重构成 "GIN 形状的可复用方法"。

---

## 9. 必须对齐的"性能语义"(否则正确但 ~30 GB/s)

```
[必做] put<Rail> 的 AggregateRequests 等价 → coalescing (P3)
[必做] red_add_rel<Rail> 有序软原子 → PackAtomicWithSeq (已有)
[必做] tail batching interval (默认 32) → 减少 signal 数 (已有)
[关注] 不要把 NCCL-GIN "每 op 一次 proxy 往返" 的慢路径也对齐回来
       (那是 EFA 上 5 GB/s 的根源); 实现里要批/合并, 不是逐 op 仿真
[关注] flush/quiet 频率: wait/flush 太频会串行化, 只在必要边界做
```

---

## 10. 验证策略

```
单元级:   uccl_gin_rail.cuh 的 put/red_add_seq 用一个独立 microbench kernel 单测
          (1 channel, 已知 pattern, 校验 receiver buffer/atomic 值)
正确性:   EP16 v2_efa_native_dispatch_smoke.py (paired + spread), 已有
对照:     同一 kernel 用 NCCLGin (单机或 GIN 可用时) vs UCCLGin, 结果一致
性能:     dispatch-only vs epilogue split (已有 UCCL_V2_PROFILE_TIMINGS)
          coalescing 前后 WR 数对比 (UCCL_PROXY_PROFILE_COMMANDS, 已有)
回归:     V1 normal/LL 路径不受影响 (UCCLGin 只接管 Rail; Lsa/NCCL 不动)
```

---

## 11. 风险 / 未决问题

```
R1  上游 kernel 是否容易模板化 gin 类型(档位1)? 若否, 走档位2 极小 fork。
R2  ncclGin 的某些 remote_action / signal 变体, Rail 后端能否全部表达?
    DeepEP 实际只用 put / put_value / red_add_rel(VASignalAdd)/ get_sym_ptr /
    barrier 这几样, 先覆盖这些; 罕用的 signal 变体按需补。
R3  coalescing 位置 (proxy vs kernel): 先 proxy(A), 量收益再决定要不要 kernel(B)。
R4  Lsa 分支复用 NCCLGin 实现: 要确保 include 不把 host-heavy 头拖进 JIT TU
    (和 transfer_cmd_device.cuh 同样的 lean 原则)。
```

---

## 12. 一句话总结

> 过去几轮的 bug 全来自"在 fork 的 kernel 里逐 call site 手抄 GIN 语义"。
> 正确的抽象边界其实就是 DeepEP 已有的 `handle::NCCLGin`。
> 做一个同形状的 `handle::UCCLGin`(Rail 走 UCCL D2H+proxy,Lsa 转发 NCCL),
> 把 EFA 的硬骨头(无 atomic→有序软原子、小消息→coalescing、barrier→host/epoch)
> **在后端解决一次**,DeepEP V2 的 dispatch/combine/LL 就能以最小改动跑在 EFA 上。
```
