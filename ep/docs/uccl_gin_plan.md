# UCCL-GIN 计划：用一个 GIN 形状的 API 后端承载 DeepEP V2

> 目标:不再逐个 fork V2 kernel、逐个 call site 手抄 GIN 语义,而是提供一个
> **和 `handle::NCCLGin` 接口/语义一致的 `handle::UCCLGin`**,其 scale-out (`Rail`)
> 后端走 UCCL D2H + CPU proxy + EFA;scale-up (`Lsa`) 原样转发 NCCL/NVLink。
> 这样 DeepEP V2 (dispatch / combine / low-latency / 未来版本) 以最小改动跑在 AWS EFA 上。
>
> 前置阅读:`AGENTS.md` 和 `worklog.md`。根目录旧 `plan.md` 以及
> `ep/docs/native_v2_efa.md` 属于上一轮 fork-based native V2 路线，已删除。

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
  include/uccl_gin/                  [待建] UCCL-GIN device backend；不要复活 v2_efa fork
    uccl_gin.cuh           [新] namespace deep_ep::elastic::handle { struct UCCLGin }
                                镜像 NCCLGin 方法签名; Lsa 转发, Rail 调下面
    uccl_gin_rail.cuh      [新] L1 device 后端: put(coalesce)/red_add_seq/put_value/offset
                                如需参考旧实验，只看 git 历史/日志，不整份恢复 fork
    transfer_cmd_device.cuh[新] D2H ring ABI 的 device 端最小头文件 (lean, JIT 可编)
    resources.cuh          [新] UCCLGinResources POD / window offset / signal scratch 几何
  src/
    proxy.cpp              [改] L0: 连续 WRITE coalescing (AggregateRequests 等价)
    rdma.cpp               [已有] EFA verbs + PackAtomicWithSeq apply
    uccl_ep.cc / uccl_proxy.cpp  [已有] binding
  docs/uccl_gin_plan.md    [本计划]

已删除并禁止复活的上一轮 fork 路径：
  include/v2_efa/*, src/v2_efa_*.cc, deep_ep_v2_wrapper/*,
  tests/v2_efa_native_dispatch_smoke.py, docs/native_v2_efa.md
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
JIT 生成 source 时 `#define DEEPEP_GIN_T handle::UCCLGin` + `#include "uccl_gin/uccl_gin.cuh"`。
re-vendor 时这种 ~3 行 patch 几乎不冲突,远好过维护 800 行 fork。

> 路径 `thirdparty/DeepEP-v2-d4f41e4` 保持不变，后续 UCCL-GIN build/JIT glue
> 从这里取 DeepEP V2 headers 和 Python 资源。`figures/`(README 图)未 vendored,无关构建。

---

## 3.6 依赖边界:不改 NCCL、不重装 NCCL

我们替换的是 **DeepEP 的 handle 层(`handle::UCCLGin`)**,不是 **NCCL 的 `ncclGin`**。
`handle::NCCLGin` 是 DeepEP 对 `ncclGin` 的薄封装,在我们 vendored 的树里,归我们改;
`ncclGin` 是 NCCL 设备 API,留在已装的 wheel 里不动。

```
kernel: gin.put<Rail>(...)            ← call site 不变
          │
   handle::UCCLGin   ← 我们的 seam(编译期换,在 vendored DeepEP 里)
          ├─ Rail → UCCL D2H + proxy + EFA          ← 根本不调 ncclGin
          └─ Lsa  → NVLink ptx (get_sym_ptr/red_add_rel_sys/st_relaxed_sys)
                    + barrier 用 raw ncclGin signal  ← 调 stock NCCL,未改
          ▼
   NCCL (nvidia-nccl-cuXX wheel,原样安装,不改不重编)
```

- **Rail(scaleout/inter-node)路径根本不进 `ncclGin`** → 不需要改 NCCL 源码。
- **Lsa(scaleup/NVLink)+ barrier 仍用 stock NCCL**(dev_comm / window / symmetric
  mapping / `ncclGin signal`),用已装 wheel,**不改不重装**。
- 仓库里的 `nccl/` 是**参考源码**(读 GIN API 形状用),**不构建、不 fork**。

**要重编的只有(都不是 NCCL)**:
```
1. vendored DeepEP:改了 handle.cuh/comm.cuh/host getter
   → 重编 DeepEP `_C`(host pybind) + 清 ~/.deep_ep JIT cache(kernel 运行时重新 JIT)
2. UCCL ep:make -C ep install (proxy coalescing / binding)
NCCL:不动。
```

**不要走符号拦截的歪路**:用 `LD_PRELOAD` / 拦 `ncclGin` 符号把 EFA put 重定向 ——
device 端符号拦不住、跨 NCCL 版本极脆。`handle::UCCLGin` 是编译期的干净 seam:
同一个 `gin.put<Rail>` 调用,编译期选不同后端,**完全不碰 NCCL**。

---

## 4. 逐方法后端设计

### 4.1 `put<Rail>` —— 必须做 coalescing(这是 30→? 的关键)

上游 `gin.put<Rail>(..., ncclGinOptFlagsAggregateRequests)` 把"聚合"下放给 GIN 层。
我们的 GIN 层 = UCCL proxy,所以 **coalescing 落在 L1/L0,kernel 仍是 per-token put**。

```
kernel: 每 token 调 uccl_gin_put<Rail>(recv_slot, send_slot, token_bytes, dst)
                │  (调用面和上游一样, per token)
                ▼
两种合并位置:

  (A) L0 proxy 侧合并:
      proxy drain 时, 把命令里
        dst_rank 相同 && req_lptr 连续 && req_rptr 连续 && 同 flags
      的 run 合成 1 条 ibv WR (bytes 累加)。kernel 不动, 改 post_rdma_async_batched。

  (B) L1 kernel 侧合并 (参考 legacy/internode.cu 的滑动窗口):
      scaleout warp 攒连续 run, 一次 v2_d2h_write(bytes=run*token_bytes)。
      省 GPU 侧 per-token __threadfence_system, 但要改 scaleout warp。
```

**优先级修正(原来默认推荐 A,改为先测再决定,预期 B 为主)**:

```
为什么 A 大概率不够:num_channels(sms20×4≈80) ≫ num_rings(lanes4)。
一个 ring 被 ~20 个 channel 的 warp 并发 atomicCAS 抢写 → ring 里相邻槽来自
不同 channel/dst,offset 不连续 → proxy "相邻+连续" 合并 run-length≈1,基本无效。
ring 不保留 per-channel 连续性;要连续 run 得在 kernel 侧(B)入队前就攒好。
```

所以 P3 **先加一个 run-length profiling gate**(扩展 `UCCL_PROXY_PROFILE_COMMANDS`,
统计 proxy 端可合并 run 的长度分布),再决定:
- 若 A 的 run-length 实测就是 ≈1 → 直接上 **B(kernel 侧 per-channel coalescing)**;
- 若某些 pattern 下 A 仍有可观 run(如 paired-remote 单 dst)→ A 作为补充。

**coalescer 的停止边界(A/B 都适用,必须显式)**:

```
遇到任一条就结束当前 run、单独成一条:
  remote_action != None     (piggyback signal, 不能并入普通 put)
  tail / signal ATOMIC      (有序软原子, 边界)
  flush / quiet / barrier    (fence 语义)
  不同 dst_rank
  不同 lane / ring
  req_lptr 或 req_rptr 不连续
```
原因见 §4.4':NCCL-GIN 的 `put(..., remote_action)` 把 put 与 action 绑定在一个
GIN op 里发布;拆成 "WRITE + 后续 ATOMIC" 后,合并不能跨这个边界,且 action 的
payload-before-action 顺序同样要靠 same-ring + PackAtomicWithSeq 保证(EFA SRD 无序)。

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

### 4.5 barrier (`comm.cuh` barrier 系列)—— 不只在 handle 上,必须 patch comm.cuh

**关键:barrier 不是只通过 `handle::NCCLGin` 方法走的**,只换 kernel 的 gin 类型不够:

```
comm.cuh:
  gpu_barrier(:213) / scaleup_barrier_wo_local_sync(:185) /
  scaleout_barrier_wo_local_sync(:200) / nvlink_barrier_wo_local_sync(:89)
    → 都硬写 const handle::NCCLGin&  ⇒ kernel 传 UCCLGin 直接编不过

  gin_barrier_wo_local_sync(:135)
    → 内部自己 const ncclGin gin(nccl_dev_comm,0,...)(:156) + signal shadow
    → scaleout_barrier 用 team=ncclTeamTagRail 调它(:204)
    ⇒ scaleout barrier 不管传什么 handle,都直接走 NCCL-GIN
```

处理:
```
1. patch comm.cuh:把 barrier 系列模板化 gin 类型 (template<class Gin>,default NCCLGin)
   → kernel 用 UCCLGin 时 scaleup barrier 仍能编(它要 gin.get_sym_ptr<Lsa>,UCCLGin 转发)
2. Rail(scaleout) barrier:Phase 1 直接删(已做),用 host dist.barrier() 替代
   → 因此不需要给 UCCLGin 实现 Rail barrier;gin_barrier_wo_local_sync 的 raw-ncclGin
     Rail 路径在我们 kernel 里是 dead path(do_scaleout=false)
3. scaleup-only 结尾 barrier:走 NVLink/Lsa(raw ncclGin from nccl_dev_comm)→ 保留,可接受
Phase 3: epoch tail,去掉 host barrier
```

> 即:comm.cuh 也是 thirdparty 极小 patch 的一部分(模板化,非重写);default=NCCLGin
> 保持上游其它 caller 不变。

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
   kernel 把 gin 的“类型/构造”换成 UCCLGin
```

**修正:patch 面不止 kernel 顶部,还包括 `comm.cuh`。** `gin.put/red_add_rel/...`
这些 call site 确实一字不改,但 **barrier 不全走 handle**:`comm.cuh` 的
`gpu_barrier` 等硬写 `const handle::NCCLGin&`(见 §4.5),kernel 传 UCCLGin 会编不过。
所以"档位 2"的最小 patch = **kernel 顶部 gin 类型/构造 + `comm.cuh` barrier 系列模板化
(default NCCLGin)**;两者都是上游 vendored 树里的小改,不是逐 call site 重写。

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
    参考旧实验时只读 git 历史/日志里的 v2_d2h_* 思路，不恢复整份 fork
    ★ 同时定 UCCLGinResources POD struct(稳定 ABI, 见下)
        │
        ▼
P1  handle::UCCLGin (L2)  ── 镜像 NCCLGin 方法签名         [核心抽象]
      Lsa 分支: 直接 #include 复用 NCCLGin 的 Lsa 实现 (ptx/ncclGin)
      Rail 分支: 调 P0 的后端
      ★ 构造接受 UCCLGinResources(一次注入, 不靠改 kernel 签名抖 ABI)
        │
        ▼
P2  kernel + comm.cuh 接线 (档位 2)                        [打通 dispatch]
      a. thirdparty 极小 patch:hybrid_dispatch.cuh 用 DEEPEP_GIN_T 可替换 gin 类型
      b. ★ patch comm.cuh:barrier 系列模板化 gin 类型 (default NCCLGin)
         —— 否则 kernel 用 UCCLGin 时连 scaleup barrier 都编不过(见 §4.5)
      c. Rail barrier 依赖 Phase-1 已删,不需 UCCLGin Rail barrier
      回归: EP16 dispatch correctness (已有 smoke test)
        │
        ├────────────────────────────┐
        ▼                             ▼
P3  put<Rail> coalescing             P4  combine 套同一个 UCCLGin
    ★ 先 run-length profiling gate:     patch hybrid_combine 仅换 gin 类型
       扩展 UCCL_PROXY_PROFILE_COMMANDS  + comm.cuh barrier 已在 P2 解决
       统计可合并 run 分布              → combine correctness 免费跟上
    然后按实测选:
      run≈1 → B(kernel 侧 per-channel 合并, 预期主力)
      仍有可观 run → A(proxy 侧) 作补充
    coalescer stop 边界见 §4.1(remote_action/atomic/fence/dst/lane/offset)
    目标: 30 → 接近 EFA 上限 (~40-44)
        │                             │
        └──────────────┬──────────────┘
                       ▼
P5  perf 调优 + Phase3 去 barrier (epoch tail)
    interval / coalesce 上限 / lane 数 sweep; README 风格 sweep
                       ▼
P6  (可选) low-latency 路径也套 UCCLGin → 全 DeepEP V2 覆盖
```

依赖关系:P0→P1→P2 是主干;P3(性能)和 P4(combine)在 P2 后并行;P5 收尾。

### UCCLGinResources(P0 定义,P1 注入,稳定 kernel/JIT ABI)

NCCLGin 只靠 `nccl_dev_comm/window/qp_idx/sharing_mode`;UCCLGin<Rail> 还需要一组
UCCL 资源。定成一个 POD,**一次性注入构造**,避免后面 JIT kernel 签名反复变:

```cpp
struct UCCLGinResources {            // device 可读 (纯 POD)
    DeviceToHostCmdBuffer** d2h_queues;   // [num_queues] host-pinned ring
    uint32_t  num_queues;                 // = 所有 proxy thread 的 channel 总数
    uint64_t  window_base;                // mapped workspace = offset 原点
    uint64_t  atomic_tail_base;           // 软原子 tail shadow 基址
    uint64_t  signal_scratch_base;        // (若仍需) tail scratch 基址
    int       num_scaleout_ranks;         // rank/lane 映射
    int       num_scaleup_ranks;
    int       scaleout_rank;
    int       scaleup_rank;
    uint32_t  num_lanes;                  // per-proxy lane 信息
};
// 传法:作为 kernel 参数传入(显式、稳定),kernel 内 UCCLGin gin(dev_comm, window, res);
// Lsa 用 dev_comm/window(转发 NCCL);Rail 用 res。
```

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
正确性:   新建 UCCL-GIN dispatch smoke（paired + spread），不要复用已删除的 v2_efa smoke
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
