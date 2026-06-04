# DeepEP V2 on AWS EFA — 完整计划

---

## 一、AWS 环境基线

### 机器

| 节点 | IP | 规格 |
|------|----|------|
| `p5en_0` | `172.31.78.36` | 8× H200, 16× EFA, driver 580 |
| `p5en_1` | `172.31.72.96` | 8× H200, 16× EFA, driver 580 |

CUDA: `/usr/local/cuda-13.0`（`/usr/local/cuda` 软链可能指向旧版，构建和运行统一用 13.0）。

本地仓库路径：`/Users/daniel/Documents/code/uccl-danyang/`

远端仓库路径：`/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/`

当前开发主线是 `ep/` 的 native DeepEP V2 on AWS EFA 路径。`uccl-ep/` 已废弃；旧
`DeepEP-danyang/` 只保留历史 benchmark/profiling 结论，不再作为当前代码开发目标。

### 虚拟环境

```bash
python3 -m venv /home/ubuntu/.venvs/deepep-danyang-cu13
source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate
```

依赖：`torch==2.12.0+cu130`，`nvidia-nccl-cu13==2.30.4`，`nvidia-nvshmem-cu13==3.6.5`，`numpy pytest pybind11 ninja packaging`。

### 构建 UCCL EP

```bash
source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
make -C ep install PYTHON="$VIRTUAL_ENV/bin/python" CUDA_PATH="$CUDA_HOME" SM=90 -j
```

DeepEP V2 依赖策略：
- 不更新、不替换旧 `thirdparty/DeepEP/`，它仍属于原 UCCL-EP V1 兼容路径。
- 新增独立 git submodule：`thirdparty/DeepEP-v2-d4f41e4/`。
- 固定上游 commit：`d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb`。
- 递归初始化该 submodule 自己的 `third-party/fmt`。
- V2 C++ JIT bridge include `thirdparty/DeepEP-v2-d4f41e4/csrc/jit/*`。
- V2 Python wrapper 默认从 `thirdparty/DeepEP-v2-d4f41e4/deep_ep` 取 DeepEP V2
  Python/JIT 资源。
- 旧 V1 wrapper、V1 bench、V1 static kernel 和旧 `thirdparty/DeepEP/` 不能被 V2 改动污染。

### 多机通过路径：aws-ofi-nccl master + GIN proxy

GIN proxy 需要 aws-ofi-nccl master（`git-c8a3df2`，位于
`/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master`），
导出 `ncclGinPlugin_v13` 和 `ncclGinPlugin_v11`。

公共环境变量：

```bash
source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:/home/ubuntu/.venvs/deepep-danyang-cu13/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
unset EP_DISABLE_GIN
unset OFI_NCCL_GIN_GDAKI
export MASTER_ADDR=172.31.78.36
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=4
export NCCL_NET_PLUGIN=ofi
export NCCL_SOCKET_IFNAME=enp71s0
```

EP16（2 节点 × 8 卡）correctness 通过命令：

```bash
# p5en_0
export MASTER_PORT=8375; export RANK=0
python tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-sms 20

# p5en_1
export MASTER_PORT=8375; export RANK=1
python tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-sms 20
```

---

## 二、性能基线

### DeepEP 性能测试命令

双机 EP16（使用公共环境变量）：

```bash
export MASTER_PORT=8382
# p5en_0: RANK=0; p5en_1: RANK=1
python tests/elastic/test_ep.py --num-processes 8 --test-first-only --num-sms 20
```

### 结果

| Arch | NIC | Topo | Dispatch BW | Combine BW | #SMs |
|------|-----|------|-------------|------------|------|
| SM90 | AWS EFA (proxy GIN) | EP8×2 | 5 GB/s (RDMA/SO) | 15 GB/s (RDMA/SO) | 20 |

DeepEP `num_sms` sweep（EP16，rails=4）：
| #SM | Dispatch SO | Combine SO |
|-----|-------------|------------|
| 4 | 3 GB/s | 8 GB/s |
| 8 | 4 GB/s | 12 GB/s |
| 20 | 4-6 GB/s | 15 GB/s |

### NCCL EFA 基线

普通 NCCL alltoall（EP16，rails=4，`alltoall_perf`）：`~91 GB/s algbw`。
NCCL EFA net path 本身不是瓶颈，DeepEP 的 `5 GB/s` 是 GIN kernel 协议瓶颈。

### GIN Proxy Microbenchmark

源码：`tools/gin_proxy_bench.cu`；口径：device-side GIN all-to-all，EP16，rails=4。

| Bytes/peer | Per-rank BW |
|------------|-------------|
| 1 MiB | 5.21 GB/s |
| 4 MiB | 17.50 GB/s |
| 16 MiB | 39.49 GB/s |
| 64 MiB | 44.25 GB/s |
| 256 MiB | 44.39 GB/s |

Rails sweep（64 MiB/peer）：1 rail → 22 GB/s；2/4 rails → plateau ~44 GB/s；≥8 rails segfault。

**结论**：纯 GIN 大包能到 `44 GB/s` per rank remote；DeepEP `5 GB/s` 是小消息 pattern 和 barrier/flush 频率放大了 proxy 弱点。Native UCCL-EP 目标绕过这个瓶颈，直接到 80–90 GB/s。

---

## 三、为什么重写 / 设计方向

### 3.1 原始 UCCL-EP 的工作方式

原 `uccl/ep` 跨节点数据面：

```
GPU kernel (V1 static .cu)
  SourceMeta / prefix / packed token staging
  write TransferCmd
         |  16B TransferCmd
         v
  D2H queue / FIFO / ring
         |  CPU poll
         v
  UCCL Proxy
  post_gpu_command / quiet / barrier / notify_gpu_completion
         |  ibverbs / EFA
         v
  remote staging buffer → V1 unpack/combine
```

旧 TransferCmd（128-bit）：`cmd_type | dst_rank | bytes | req_rptr | req_lptr | expert_idx`。

transport substrate（proxy、D2H queue、EFA QP/MR/CQ、WR batching、CQ polling、ack）已成熟，**应复用**。
V1 语义层（SourceMeta、prefix matrix、packed staging、静态 internode/intranode kernel）**不应延续**。

### 3.2 V2 的不同之处

DeepEP V2 的核心是 JIT kernel 的 `BufferLayout / TokenLayout / expanded dispatch / reduced combine`：

```
ElasticBuffer.buffer = [
  scaleup_buffer:       [scaleup_rank][scaleout_rank × max_tokens][TokenLayout]
  scaleout_send_buffer: [1][max_tokens][TokenLayout]   ← scaleout warp TMA store，EFA 读取 source
  scaleout_recv_buffer: [scaleout_ranks][channels × max_tokens_per_ch][TokenLayout]
                                                       ← EFA RDMA write 目标
]
TokenLayout = hidden | sf | topk_idx | topk_weights | src_global_idx | linked_list_idx | mbarrier
```

V2 dispatch 目标是把 token 直接落到 `remote.scaleout_recv_buffer[src_rank][channel][slot]`，
然后 forward warp 流式消费，写入 `scaleup_buffer`。这就是为什么不能改 `internode.cu`。

### 3.3 主方向：V2 JIT → 旧 TransferCmd → UCCL proxy

```
不要：V2 JIT → V2TransferCmd → adapter → old TransferCmd → UCCL proxy
要：  V2 JIT → old TransferCmd → UCCL proxy
```

旧 TransferCmd 已经足够表达 WRITE；V2 和 V1 的差异只在 JIT 如何计算 offset 和选择 queue/channel，
CPU proxy/RDMA 层不需要知道 expert、expanded slot、reduced combine。

V2 的三个特殊需求都可以用旧 TransferCmd 表达：

```
1. region (buffer / workspace / signal_scratch):
   → JIT 把 V2 指针算成 transport window offset → TransferCmd.req_lptr / req_rptr

2. signal (tail / count / done):
   → GPU 把 tail_word 写入 signal scratch[slot]
   → 再发旧 TransferCmd WRITE: scratch[slot] → remote workspace tail ptr

3. lane:
   → channel_idx % num_d2h_queues 选择 queue/proxy/lane
   → TransferCmd 本身不携带 lane 字段
```

### 3.4 复用 vs 新增 vs 删除

| 分类 | 内容 |
|------|------|
| **复用** | `src/proxy.cpp`, `src/rdma.cpp`, `src/common.cpp`, `src/uccl_proxy.cpp`, `src/fifo.cpp`, `src/adaptive_sleeper.cc`, D2H queue FIFO/ring, `acked_wrs_` / `notify_gpu_completion()` |
| **必须修改** | V2 JIT kernel（fork `hybrid_dispatch.cuh`），offset 计算，signal scratch 写法，lane/channel 映射，transport window 注册，receiver 落点，Python binding |
| **应删除** | `V2TransferCmd`，`V2EfaVerbsPostSink`，`V2EfaConnectionHandle` 自建 QP/CQ 路径，`EfaPostSink` prototype，semantic materialize/overlay fallback |

### 3.5 V1/V2 差异原则对照表

| 差异点 | V1 做法 | V2 目标做法 | 必须不同的理由 |
|--------|--------|------------|--------------|
| kernel 形态 | 静态 `.cu` | fork V2 JIT `.cuh` | V2 kernel 依赖运行时参数，静态 kernel 只会 V1 packed staging |
| 数据布局 | `SourceMeta`、prefix matrix | `BufferLayout / TokenLayout`、expanded/reduced | 字段语义不同，不是改名 |
| command 格式 | 旧 16B `TransferCmd` | **继续使用旧 16B `TransferCmd`** | old command 足够表达 WRITE |
| lane 语义 | queue/proxy/channel 隐含 | 继续 `channel_idx % num_fifo_queues` 选 queue | 和 V1 一致，不新增 lane 字段 |
| Tail 通知 | V1 counter/barrier | tail_word 写 scratch + 旧 TransferCmd WRITE | EFA 无硬件 remote atomic |
| MR/offset | 固定 rdma buffer | V2 workspace/buffer/scratch 使用一个连续 symmetric window，旧 `TransferCmd` 保存 `offset >> 2` | 旧 command 没 region 字段；单 window 正好匹配 DeepEP V2 布局 |
| receiver 落点 | V1 staging buffer | 直接写 `scaleout_recv_buffer` | 避免额外 GPU memcpy |
| proxy 框架 | CPU proxy/FIFO/EFA post | 原样复用 | transport substrate 与 V1 语义耦合弱 |
| QP 默认值 | V1 默认 `num_qps_per_rank=24`，部分 bench 用 `max(num_sms, expert/rank)` | native V2 默认固定 `24`，可用 `UCCL_V2_NUM_QPS` 或 dispatch `num_qps` 覆盖 | EFA scaleout 已由 UCCL proxy queues/lanes 决定，DeepEP V2 `num_sms*16+1` 是给 NCCL GIN scaleout 的，不应默认套用到 AWS EFA |

**关于 MR 注册（2026-06-03 修正，推翻早期"三独立 MR"说法）**：
`rdma.cpp` 的 WRITE 路径是**单 window**：remote = `ctx->remote_addr + (req_rptr<<2)`（单 rkey），
local = `ctx->mr->addr + (req_lptr<<2)`（单 lkey）。`req_lptr/req_rptr` 是相对**单一 base** 的
32-bit shifted offset（shift=2，≤16 GiB），TransferCmd **没有 region 字段**。
`lkey_for/rkey_for/gpu_mr_chunks` 是 `#ifdef USE_DMABUF` 下对**一个连续 window** 的 chunk 拆分，
不是多区域寻址。所以"三独立 MR"不可行。

**解决：单一统一 window（正好是 DeepEP 现有布局，零 transport 改动）**。
`csrc/elastic/buffer.hpp::get_native_v2_resources()` 已把 workspace+buffer 连续放在**一个 NCCL
symmetric window**：`raw_workspace`=window base，`raw_buffer`=`raw_workspace+ws_bytes`，尾部还有
`num_cpu_buffer_bytes`（engram/agrs，按 API 灵活使用）。mapped 侧 `buffer = workspace + ws_bytes`
（buffer.hpp:132），mapped 与 raw 布局一致。
- 把**整个 window**注册成**一个 MR**（base = raw_workspace，len = ws+gpu_buffer+cpu_buffer）。
- kernel offset = `target_mapped_addr - workspace_base`（workspace/buffer/scratch 通用），
  `req_lptr=req_rptr=offset>>2`（symmetric window 两侧 offset 相同）。
- 约束：window 总大小 < 16 GiB（shift=2 寻址上限），需断言。
- **signal_scratch** 占用尾部 cpu_buffer/engram 区域（仅作 local source，单 MR 覆盖）。

**关于 GPUDirect 内存序**：`ring_buffer.cuh::commit_with_head` 在 `DeviceToHost` 方向调用
`__threadfence_system()`，保证 GPU 对 RDMA buffer 的写在 CPU 读到 D2H slot 之前对 NIC 可见。
V2 signal scratch 放 GPU memory 使用同样机制，无额外问题。

---

## 四、V2 关键架构（代码精读）

### Buffer 结构（`layout.cuh`，GPU 内存）

```
ElasticBuffer.buffer（GPU memory，NCCL symmetric window）= [
  scaleup_buffer:        [kNumScaleupRanks][kNumScaleoutRanks * kNumMaxTokensPerRank][TokenLayout]
  scaleout_send_buffer:  [1][kNumMaxTokensPerRank][TokenLayout]
                         ← scaleout warp TMA store 目标；EFA 从这里读取 payload
  scaleout_recv_buffer:  [kNumScaleoutRanks][kNumChannels * kNumMaxTokensPerChannel][TokenLayout]
                         ← EFA RDMA write 应直接写这里
]
TokenLayout = [hidden | sf | topk_idx(int×topk) | topk_weights(float×topk) |
               src_global_idx(int) | linked_list_idx(int) | mbarrier]
```

### WorkspaceLayout（GPU workspace，关键元数据）

```
scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx) → int64_t
  = math::pack2<int, int64_t>(finish_flag, tail_count)
  内存布局：lo32 = finish_flag, hi32 = tail_count
  由 scaleout warp 的 gin.red_add_rel 增量写入（每 kScaleoutUpdateInterval=3 token）
  由 forward warp spin-wait 消费（streaming，不等全量）
  EFA fork: CPU proxy 写绝对值（不是 delta），forward warp 读法不变
```

### V2 dispatch 数据流（`hybrid_dispatch.cuh`）

```
notify warp（SM0，per-SM 汇总 rank/expert count，再广播给所有 scaleout ranks）：
  gin.put<ncclTeamTagRail>(dst.workspace.rank_count, ...)    ← EFA，替换为 D2H FIFO
  gin.put<ncclTeamTagRail>(dst.workspace.expert_count, ...)  ← EFA，替换为 D2H FIFO
  gin.put_value<ncclTeamTagLsa>(...)                         ← NVLink，保留不动
  gin.red_add_rel<ncclTeamTagLsa>(...)                       ← NVLink，保留不动

scaleout warp（per channel = sm × kNumChannelsPerSM）：
  TMA store x[token] → scaleout_send_buffer[token_idx]
  gin.put<ncclTeamTagRail>(
      remote=receiver.scaleout_recv_buffer[src_rank][channel][dst_slot],
      local=scaleout_send_buffer[token_idx], bytes, dst_rank)    ← EFA，替换
  if every 3 tokens or finish:
    gin.red_add_rel<ncclTeamTagRail>(
        receiver.workspace.channel_tail[channel][src_rank] += delta)   ← EFA，替换

forward warp（per channel，同一 kernel）：
  spin-wait: channel_tail[channel][src_rank] > old_tail       ← ld_acquire_sys
  copy: scaleout_recv_buffer[src][channel][slot] → scaleup_buffer   ← NVLink GIN，保留
  build: token_metadata_at_forward, dst_buffer_slot_idx
  末尾重置：*channel_signaled_tail_ptr = 0                    ← Phase 1 保留，Phase 3 去掉
```

### Tail word 格式

```
Phase 1（与 V2 原版相同）：
  int64_t = math::pack2<int, int64_t>(finish_flag, tail_count)
  内存：lo32 = finish_flag, hi32 = tail_count

Phase 3 扩展（去 barrier 时）：
  lo32 = (epoch << 1) | finish_flag   ← epoch 31 bits + finish 1 bit
  hi32 = tail_count                   ← 不变

注意：EFA fork 写绝对值（非 delta）。每个 tail entry 对应唯一 sender，不存在竞争。
```

### Channel → QP/lane 映射

```
efa_lane(channel_idx) = channel_idx % num_efa_lanes

规则：一个 channel 的所有操作（payload write、tail write）必须使用同一个 efa_lane，
      不能用 expert_id 分 lane。QP ordering 是 tail 可见时 payload 已到达的保证。
```

---

## 五、基于原 `uccl/ep` 还需要改什么

### 应原样复用的部分

| 原 `uccl/ep` 能力 | 复用方式 |
|-------------------|----------|
| `include/ring_buffer.cuh::TransferCmd` | 主路径继续使用 |
| D2H queue / FIFO / ring | GPU→CPU command queue，ack/tail 生命周期已成熟 |
| `src/proxy.cpp::post_gpu_command()` | 多 queue poll、slot ack、backpressure |
| `src/proxy.cpp::post_gpu_commands_mixed()` | WRITE/QUIET/BARRIER 分流 |
| `src/rdma.cpp::post_rdma_async_batched()` | EFA/IB verbs post、CQ、chunk/MR 逻辑 |
| `src/uccl_proxy.cpp` lifecycle | proxy thread、peer meta、listen port |
| CQ poll / `acked_wrs_` / `notify_gpu_completion()` | D2H queue deadlock 防止的关键 |
| 单 window MR / chunk 查表 | V2 workspace + buffer + signal scratch 作为同一连续 symmetric window 注册；如启用 DMA-BUF，只做同一 window 的 chunk 拆分 |

### 必须修改的部分

| 需要改的点 | 改动内容 | 为什么 |
|------------|----------|--------|
| V2 JIT kernel | fork `hybrid_dispatch.cuh`，把 scaleout GIN call site 改成写旧 TransferCmd | V1 static kernel 只会 packed staging |
| offset 计算 | JIT 把 V2 buffer/workspace/scratch 指针算成 transport offset | 旧 TransferCmd 只有 offset，没有 region 字段 |
| signal 写法 | GPU 先写 tail_word 到 signal scratch，再发 TransferCmd WRITE | 旧 TransferCmd 没 immediate value 字段 |
| lane/channel | 不把 lane 放进 command；`channel_idx % num_queues` 选 D2H queue | V1 也是 queue 隐含 lane |
| transport window | V2 workspace + buffer + scratch 使用同一连续 symmetric window，旧 `TransferCmd` 保存 `offset >> 2` | 旧 command 没 region 字段，单 window 与 V2 buffer 分配天然匹配 |
| receiver 落点 | remote offset 直接指向 `scaleout_recv_buffer` 或 workspace tail ptr | 避免额外 memcpy |
| Python binding | 用 V2 buffer resource 初始化 `UcclProxy` | 回到原 UCCL proxy lifecycle |
| 删除原型路径 | 不带入 `V2TransferCmd`、`EfaPostSink`、`V2EfaVerbsPostSink`、materialize fallback | 防止 benchmark 跑到非主路径 |

---

## 六、实现计划

### 阶段 0：立即修复

**0a. CQE correctness 验证**
- EFA SRD QP 保持 `sq_sig_all=1`（`sq_sig_all=0 + signal-only CQE` 在 p5en 上 `ibv_wr_complete ret=22`）
- completion accounting 回到 `acked_wrs_` / `notify_gpu_completion()` / ring ack mask
- signal scratch slot 生命周期绑定 D2H command slot（slot ack 才可复用）
- signal scratch capacity ≥ max inflight D2H commands（加初始化断言）
- 跑 EP16 remote-pair correctness，确认不 timeout、不 silent corruption

**0b. README-size dispatch bench**
- 建立与 GIN path（5 GB/s）可对比的 scaffold baseline

---

### 阶段 1：fork `hybrid_dispatch.cuh`（决定能否到 90 GB/s）

**fork 范围：只替换 `ncclTeamTagRail`（EFA scaleout）；保留 `ncclTeamTagLsa`（NVLink scaleup）。**

| GIN 调用 | 位置 | 处理 |
|---------|-----|------|
| `gin.put<ncclTeamTagRail>` rank_count | notify warp, SM0 | D2H FIFO（notify cmd） |
| `gin.put<ncclTeamTagRail>` expert_count | notify warp, SM0 | D2H FIFO（notify cmd） |
| `gin.put<ncclTeamTagRail>` payload | scaleout warp | D2H FIFO（payload cmd） |
| `gin.red_add_rel<ncclTeamTagRail>` tail | scaleout warp | D2H FIFO（tail cmd） |
| `gin.put_value<ncclTeamTagLsa>` | notify warp | **保留不动** |
| `gin.red_add_rel<ncclTeamTagLsa>` | notify warp | **保留不动** |
| `gin.get_sym_ptr<ncclTeamTagLsa>` | forward warp | **保留不动** |

**GPU barrier 处理**：
- 开头 `gpu_barrier<kHybridDispatchTag0>` 用 ncclTeamTagRail → Phase 1 直接移除（Python `dist.barrier()` 替代）
- 结尾 `gpu_barrier<kHybridDispatchTag1>(do_scaleout=false)` 是 scaleup-only NVLink barrier → 保留

**新增 kernel 参数（EFA fork 专属）**：

```cpp
DeviceToHostCmdBuffer** d2h_queues,
uint32_t                num_d2h_queues,
uint64_t                buffer_base,
uint64_t                workspace_base,
uint64_t                signal_scratch_base,
uint64_t                buffer_transport_offset,
uint64_t                workspace_transport_offset,
uint64_t                signal_scratch_transport_offset,
```

#### 1a. GPUDirect RDMA smoke test（hard gate）

用原 UCCL Proxy / rdma.cpp 验证：
1. 注册 cuda transport window 为 MR
2. GPU 写一个旧 TransferCmd WRITE
3. proxy drain 后对端 GPU tensor 内容正确

#### 1b. 注册整个 NCCL symmetric window 为单一 EFA MR（修正）

```
ibv_reg_mr(raw_workspace, ws_bytes + gpu_buffer_bytes + cpu_buffer_bytes) → 单 mr / rkey
  raw_workspace = nccl_context->get_raw_window_ptr()（get_native_v2_resources 的 rdma_workspace_ptr）
bootstrap: allgather 这一个 remote base + rkey
ctx->mr = 该 MR；ctx->remote_addr/remote_rkey/remote_len = 对端同一 window
signal_scratch = window 尾部 cpu_buffer 区域的一个切片（同一 MR 覆盖）
```

kernel 把 workspace/buffer/scratch 目标地址都换算成相对 `workspace_base`(=mapped window base)
的 offset，`req = offset >> 2`。symmetric window 保证两侧 offset 相同。
约束：window 总大小 < 16 GiB（shift=2 上限），`init` 时断言；超限再走 UCCL DMABUF chunk 方案，
**不能悄悄截断 offset**。

#### 1c. 替换 notify warp 的 `gin.put<ncclTeamTagRail>`

```cpp
// 替换 gin.put<ncclTeamTagRail>(dst.workspace.rank_count, src.workspace.rank_count, bytes, dst_rank)
TransferCmd cmd{};
cmd.cmd_type = make_cmd_type(CmdType::WRITE, false, false);
cmd.dst_rank = dst_scaleout_rank_idx;
cmd.bytes    = kNumScaleupRanks * sizeof(int);
cmd.req_lptr = encode_write_offset(unified(workspace.get_scaleout_rank_count_ptr<true>(dst_rank)));
cmd.req_rptr = encode_write_offset(unified(workspace.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx)));
// queue 选择表达 lane，不进 command
d2h_push(dst_scaleout_rank_idx % num_d2h_queues, cmd);
```

#### 1c'. 替换 `gin.put()` payload（scaleout warp）

```cpp
const uint32_t queue_idx = channel_idx % num_d2h_queues;
const uint64_t local_off  = unified(scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr());
const uint64_t remote_off = unified(
    scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx)
        .get_channel_buffer(channel_idx).get_token_buffer(dst_slot).get_base_ptr());

TransferCmd cmd{};
cmd.cmd_type = make_cmd_type(CmdType::WRITE, false, false);
cmd.dst_rank = stored_dst_scaleout_rank_idx;
cmd.bytes    = token_bytes;
cmd.req_lptr = encode_write_offset(local_off);
cmd.req_rptr = encode_write_offset(remote_off);
d2h_push(queue_idx, cmd);
```

#### 1d. 替换 `gin.red_add_rel()` → tail write

EFA native 写**绝对值**（非 delta）。每个 `(channel_idx, scaleout_rank_idx)` 对应唯一 sender，
不存在并发竞争，绝对值等价 atomic add。tail_word 由 GPU scaleout warp 在已知精确 tail_count 时生成：

```cpp
// 在 scaleout warp 的 update_scaleout_tail() 位置（每 3 token 或 finish 时）：
if (should_update && lane_idx < kNumScaleoutRanks) {
    const uint32_t queue_idx = channel_idx % num_d2h_queues;
    const uint64_t scratch_ptr = signal_scratch_slot_for(queue_idx, d2h_slot);

    // Phase 1 格式与 V2 原版相同
    const int64_t tail_word = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
    *reinterpret_cast<int64_t*>(scratch_ptr) = tail_word;
    // __threadfence_system() 通过后续 d2h_push 的 commit_with_head 隐含调用

    TransferCmd cmd{};
    cmd.cmd_type = make_cmd_type(CmdType::WRITE, false, false);
    cmd.dst_rank = lane_idx;
    cmd.bytes    = sizeof(int64_t);
    cmd.req_lptr = encode_write_offset(unified(scratch_ptr));
    cmd.req_rptr = encode_write_offset(unified(
        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx)));
    d2h_push(queue_idx, cmd);  // 同一 queue，payload 已先入，tail 后入，QP ordering 保证到达顺序
}
```

**Phase 1 注意**：forward warp 末尾的 `*channel_signaled_tail_ptr = 0` 清零保留。
多轮 dispatch 前需要 `torch.cuda.synchronize()` + `dist.barrier()` 确保 forward warp 完成清零，
才可以开始写新 tail（防止 stale tail 触发错误的 forward skip）。

#### 1e. forward warp 与 epilogue

- forward warp 结构保留 V2 原始语义（chunk streaming，build metadata）
- forward warp 内 scaleup NVLink TMA store（`gin.get_sym_ptr<ncclTeamTagLsa>`）不改动
- `dispatch_copy_epilogue` 保留 V2 原始语义

---

### 阶段 2：多线程持久化 CPU 代理

阶段 1 后 proxy post rate 大幅增加，Python 单线程 drain 成为瓶颈：

- 复用原 `UcclProxy` / `Proxy` lifecycle、peer meta、listen port、thread pinning
- 多个持久 C++ proxy 线程（参考 UCCL-EP 论文 Figure 17）各持有自己的 D2H queue/proxy/lane
- 每线程负责 `channel_idx % num_fifo_queues == thread_idx` 的 channel
- completion counter、slot ack、tail advance 继续使用 `acked_wrs_` / `notify_gpu_completion()`

---

### 阶段 3：去掉 pre-dispatch `dist.barrier()`

阶段 1 的 tail word 保持 V2 原格式；阶段 3 将 lo32 扩展为 `(epoch << 1) | finish_flag`：

1. receiver 不清零 tail（epoch 区分轮次）
2. sender tail write 带当前 epoch
3. forward warp spin-wait 额外检查 `(lo32 >> 1) == current_epoch`
4. Python dispatch 前 `epoch += 1`，去掉 `dist.barrier()` 调用

预期：`stage_and_pre_barrier_ms` 0.9ms → 0.05ms。

---

### 阶段 4：combine native path（fork `hybrid_combine.cuh`）

与 dispatch 对称，替换 GIN 为 D2H FIFO + EFA write：
- 保留 combine forward warp 和 reduce epilogue 结构
- 支持 topk>1 多 contributor reduce
- 去掉 `_semantic_combine_data` fallback

---

### 阶段 5：性能调优

- **QP 数量**：同一 channel 的所有 payload write 和 tail write 只能用同一 QP（否则 tail ordering 失效）
- **`kMaxInflight`**：控制 D2H queue 深度，防止 EFA CQ 溢出
- **intra-node token 去重**：`skip_scaleout_rank` 完整跳过本地 rank
- **Token coalescing**：local 和 remote 都连续时合并为单次 RDMA write；否则不合并

---

## 七、性能路线图

| 阶段 | 关键改动 | 预期 dispatch GB/s |
|------|---------|-------------------|
| 现状（scaffold） | — | 2.91 |
| 阶段 0a | CQE correctness 修复 | ~3.2 |
| scaffold + 多线程 proxy | 阶段 2 alone | 15–25（GPU memcpy 瓶颈仍在） |
| **阶段 1（fork）** | GPUDirect + scaleout_recv_buffer | **基线大幅提升** |
| 阶段 1+2+3 | fork + 多线程 + 无 barrier | **~60–70** |
| 阶段 1+2+3+4+5 | + combine + 调优 | **~90** |

**完成判定：dispatch >= 80 GB/s 为接近完成，= 90 GB/s 为目标完成。**

EFA vs CX7 带宽相同（均 400 Gbps/GPU），V1 uccl-ep on CX7 EP16 = 61 GB/s，
V2 DeepSeek README EP16 = 90 GB/s，native V2 on EFA 目标同样 90 GB/s。

---

## 八、完成判定标准

- [ ] GPUDirect smoke test：`ibv_reg_mr(cuda_ptr)` + RDMA write 到对端 GPU tensor 成功
- [ ] EP16 dispatch + combine correctness（topk=8，do_expand=True，multi-contributor reduce）
- [ ] forward warp 直接消费 `scaleout_recv_buffer`（EFA RDMA 落点），写入 `scaleup_buffer`；无独立 EFA window 中转
- [ ] `_semantic_dispatch_data` / `_semantic_combine_data` 从生产路径消失
- [ ] dispatch/combine 主路径不再使用 `V2TransferCmd`、`V2EfaConnectionHandle`、`V2EfaVerbsPostSink`
- [ ] V2 JIT 直接生成旧 16B `TransferCmd`，由原 UCCL proxy drain/post/ack
- [ ] `dist.barrier()` 在 dispatch/combine 热路径中不再出现（阶段 3 后）
- [ ] forward warp streaming tail（每 3 token 推进）正确工作
- [ ] README-style EP16 dispatch bench **>= 80 GB/s**
- [ ] Timing breakdown 能用 RDMA post 数、CQ rate、lane utilization 解释性能

---

## 九、目标 Native V2 数据流图

```
GPU notify warp (SM0)               CPU proxy (notify lane)           GPU receiver workspace
-------------------                 -----------------------           ----------------------
count rank/expert per SM
  → GPU workspace (local atomic)
wait kNumSMs arrive
→ D2H FIFO push old TransferCmd ──→ post RDMA write
  dst=receiver.workspace.           remote = receiver.workspace.
  scaleout_{rank,expert}_count      scaleout_{rank,expert}_count
                                    receiver spin-waits on count > 0
                                    then NVLink (GIN) → scaleup ranks

GPU scaleout warp                   CPU proxy (per channel)           GPU receiver buffer
-----------------                   -----------------------           -------------------
TMA store x[token]
→ scaleout_send_buffer[token_idx]
        |
D2H FIFO push old TransferCmd ────→ poll FIFO
  cmd_type=WRITE                     decode old TransferCmd
  queue = chan % queues               post RDMA write (GPUDirect)
  req_lptr = unified(send_slot)        local = sender scaleout_send_buffer
  req_rptr = unified(recv_slot)        remote = receiver scaleout_recv_buffer
                                              ↓
  (every 3 tokens or finish:)
  GPU writes tail_word → scratch[slot]
D2H FIFO push old TransferCmd ────→ post RDMA write (same QP/queue)
  req_lptr = unified(scratch[slot])    local = signal_scratch
  req_rptr = unified(workspace.tail)   remote = workspace.channel_tail[chan][src_rank]
                                              ↓
                                    poll CQ / ack via UCCL proxy
                                              ↓ EFA RDMA (GPUDirect)
                                    payload → scaleout_recv_buffer[src][chan][slot]
                                    tail    → workspace.channel_tail[chan][src_rank]
                                              ↓
                                    GPU forward warp
                                    spin-wait: channel_tail > old_tail (ld_acquire_sys)
                                    copy: scaleout_recv_buffer → scaleup_buffer (NVLink GIN)
                                    build: token_metadata_at_forward, dst_buffer_slot_idx
                                              ↓ dispatch_copy_epilogue (V2 JIT)
                                    scaleup_buffer → recv_x / recv_topk_idx / recv_src_metadata
```

**核心设计原则**：
- transport 方法就是 UCCL EP V1：GPU 写旧 TransferCmd 到 D2H FIFO，CPU proxy drain，EFA verbs post，CQ/ack 复用原实现
- 数据语义像 DeepEP V2：payload 从 `scaleout_send_buffer` 读，写到 `scaleout_recv_buffer`，forward warp 和 epilogue 保持 V2 原始语义
- ordering 靠 EFA SRD 同 QP：payload unsignaled + tail signaled，同 `efa_lane` 的 QP 保证 tail 可见时 payload 已可见
- scaleup NVLink 不动：forward warp 里 `gin.get_sym_ptr<ncclTeamTagLsa>` TMA store 仍走 GIN

---

## 十、文件修改清单（阶段 1 实施）

工作目录：`ep/`（全新 fork，基于 `uccl/ep` + DeepEP V2 kernel）。

#### 已完成的 fork 操作

- `ep/` 整体复制自 `uccl/ep/`，V1 完整保留（`internode.cu`, `intranode.cu`, `layout.cu`, `ep_runtime.cu` 等全在）
- `include/v2_efa/hybrid_dispatch_native.cuh`（从 `deep_ep/impls/hybrid_dispatch.cuh` 复制，672 行，全量原版，待修改）
- `include/v2_efa/hybrid_combine_native.cuh`（从 `deep_ep/impls/hybrid_combine.cuh` 复制，620 行，全量原版，待修改）
- `include/v2_efa/workspace.hpp`(61), `jit_plan.hpp`(624), `topology.hpp`(50)（从 `uccl-ep` 带入）
- `src/v2_efa_deep_ep_jit.cc`（273 行，已清理：删除全部旧 scaffold / V2TransferCmd 函数，只保留 JIT 基础设施 + native hybrid dispatch + copy epilogue，include 改为 `ring_buffer.cuh`，queue 参数改为 `DeviceToHostCmdBuffer**`，已加 `signal_scratch_base`）
- `deep_ep_v2_wrapper/`（从 `uccl-ep` 带入，Python V2 层）

#### 当前头文件归整状态

实际状态与早期假设不符，盘点如下：

| 缺口 | 现象 | 处理 |
|------|------|------|
| `include/v2_efa/runtime.hpp` **未 fork** | `v2_efa_deep_ep_jit.cc:1` 和 `v2_efa_runtime.cc:1` 都 `#include "v2_efa/runtime.hpp"`，文件不存在 | 从 `uccl-ep` fork 并清理（见下「头文件归整」） |
| `descriptor.hpp` **未 fork**，但 `workspace.hpp:6` 仍 `#include "v2_efa/descriptor.hpp"` | 编译即断 | `workspace.hpp` 去掉该 include；`DescriptorPlanStats` 等旧统计结构整体不带入 |
| `transfer_cmd.hpp` **未 fork** | `DispatchTransferLayout`/`CombineTransferLayout` 已确认是废弃 descriptor 设计的死代码 | 不抢救、不带入；native V2 主路径直接写旧 16B `TransferCmd` |
| `src/v2_efa_runtime.cc`（288 行）仍是**旧 scaffold** | 用 `DescriptorPlanStats`/`worst_case_stats`/`max_dispatch_segments` 等已废弃符号 | 与 `v2_efa_deep_ep_jit.cc` 同样清理 + 重写为 MR/proxy 初始化 |

注：`uccl-ep/include/v2_efa/` 下的 `transfer_cmd.hpp`、`verbs_sink.hpp`、`efa_adapter.hpp`、`uccl_transfer_adapter.hpp`、`descriptor.hpp`、`dispatch_jit.cuh`、`combine_jit.cuh`、`transfer_d2h_queue.cuh`、`proxy.hpp` **都不应进入 `ep/`**。早期抢救出来的 `transfer_layout.hpp` 已确认无引用并删除。

---

### 修改文件

#### 0. 头文件归整（header reconciliation，编译前置，无依赖，先做）

让 `ep/` 重新可编译，不引入任何废弃头：

```
0a. include/v2_efa/workspace.hpp
    删除 #include "v2_efa/descriptor.hpp"（line 6）
    若 WorkspacePlan 依赖 DescriptorPlanStats，就地内联所需字段或删除该依赖。

0b. fork + 清理 include/v2_efa/runtime.hpp（303 → ~150 行）
    从 uccl-ep fork，然后：
      - include 改为 jit_plan.hpp + topology.hpp + workspace.hpp
        （删 descriptor.hpp / transfer_cmd.hpp / transfer_layout.hpp）
      - 删除全部旧 scaffold 自由函数声明（descriptor_enqueue / forward_metadata /
        receiver_metadata / materialize_records / signal_offsets / expand_records /
        combine_descriptor_enqueue / combine_forward_metadata）
      - 删除 V2EfaRuntime 里对应的旧 build_* / launch_* 方法声明
      - 只保留：init/compile JIT、launch_v2_efa_native_hybrid_dispatch_plan、
        launch_v2_efa_dispatch_copy_epilogue_plan，及 V2EfaRuntime 的
        launch_native_hybrid_dispatch / launch_dispatch_copy_epilogue
      - 这两个保留声明的签名要与已清理的 v2_efa_deep_ep_jit.cc 对齐
        （DeviceToHostCmdBuffer** queues、signal_scratch_base 参数）

0c. src/v2_efa_runtime.cc（288 → 重写）
    删掉 worst_case_stats / DescriptorPlanStats / max_dispatch_segments 等旧 scaffold；
    保留 RuntimeConfig 校验 + status，其余并入步骤 5（MR/proxy 初始化）。
```

完成 0 后，`ep/` 应能在不接通新功能的前提下编译通过（hybrid kernel 仍是原版 fork）。

---

#### 1. `include/v2_efa/workspace.hpp`　61 → ~130 行　+69

补充 signal scratch 基础设施：

```
新增常量：
  kSignalScratchSlotsPerQueue = kQueueSize  // 与 D2H queue 深度 1:1
  kSignalScratchSlotBytes = sizeof(int64_t) // 一个 tail_word

新增函数：
  // GPU device-side：根据 queue 下标和 D2H slot 下标得到 scratch 指针
  __device__ inline int64_t* signal_scratch_slot_for(
      void* scratch_base, uint32_t queue_idx, uint32_t slot_idx,
      uint32_t slots_per_queue);

新增 scratch 总大小计算：
  inline size_t signal_scratch_bytes(uint32_t num_queues);
```

---

#### 2. `include/v2_efa/hybrid_dispatch_native.cuh`　672 → ~820 行　+148

全量原版 `hybrid_dispatch.cuh` fork，尚未修改。需要：

**新增内核参数**（在 `__global__ void hybrid_dispatch_kernel` 签名末尾）：

```cpp
DeviceToHostCmdBuffer** d2h_queues,       // [num_d2h_queues]，pinned host memory 中的 ring buffer
uint32_t                num_d2h_queues,
uint64_t                buffer_base,      // V2 ElasticBuffer.buffer GPU 地址
uint64_t                workspace_base,   // V2 workspace GPU 地址
uint64_t                signal_scratch_base, // signal_scratch GPU 地址（GPU memory，需 __threadfence_system）
```

**替换 notify warp 的两个 `gin.put<ncclTeamTagRail>`**（原始约 178、183 行）：

```cpp
// 每个 dst_scaleout_rank 一条 TransferCmd WRITE
const uint32_t q = dst_scaleout_rank_idx % num_d2h_queues;
TransferCmd cmd{};
cmd.cmd_type = make_cmd_type(CmdType::WRITE, false, false);
cmd.dst_rank = static_cast<uint8_t>(dst_scaleout_rank_idx);
cmd.bytes    = kNumScaleupRanks * sizeof(int);
cmd.req_lptr = encode_lptr(scratch_base + ..., scratch_transport_offset, false);
cmd.req_rptr = encode_rptr(workspace_base + offset_of_rank_count(...), workspace_transport_offset, false);
d2h_queues[q]->atomic_set_and_commit(cmd);
// expert_count 同上，另一条 TransferCmd
```

**替换 scaleout warp 的 `gin.put<ncclTeamTagRail>` payload**（原始约 444 行）：

```cpp
const uint32_t q = channel_idx % num_d2h_queues;
TransferCmd cmd{};
cmd.cmd_type = make_cmd_type(CmdType::WRITE, false, false);
cmd.dst_rank = static_cast<uint8_t>(stored_dst_scaleout_rank_idx);
cmd.bytes    = token_bytes;
cmd.req_lptr = encode_lptr(buffer_base + send_slot_offset, buffer_transport_offset, false);
cmd.req_rptr = encode_rptr(buffer_base + recv_slot_offset, buffer_transport_offset, false);
d2h_queues[q]->atomic_set_and_commit(cmd);
```

**替换 scaleout warp 的 `gin.red_add_rel<ncclTeamTagRail>` tail**（原始约 342 行）：

```cpp
const uint32_t q = channel_idx % num_d2h_queues;
uint64_t d2h_slot; // 从上一条 payload 的 atomic_set_and_commit out_slot 取
int64_t* scratch_ptr = signal_scratch_slot_for(signal_scratch, q, d2h_slot % kSignalScratchSlotsPerQueue, kSignalScratchSlotsPerQueue);
*scratch_ptr = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
// __threadfence_system() 通过下面 atomic_set_and_commit 的 commit_with_head 隐含
TransferCmd sig{};
sig.cmd_type = make_cmd_type(CmdType::WRITE, false, false);
sig.dst_rank = static_cast<uint8_t>(lane_idx);
sig.bytes    = sizeof(int64_t);
sig.req_lptr = encode_lptr((uint64_t)scratch_ptr, scratch_transport_offset, false);
sig.req_rptr = encode_rptr(workspace_base + tail_ptr_offset, workspace_transport_offset, false);
d2h_queues[q]->atomic_set_and_commit(sig);
```

**删除开头 gpu_barrier<ncclTeamTagRail>**（约 81 行，5 行），Phase 1 用 Python `dist.barrier()` 替代。

---

#### 3. `include/v2_efa/hybrid_combine_native.cuh`　620 → ~720 行　+100

从 `hybrid_combine.cuh` fork 来，同样替换 3 个 `ncclTeamTagRail` 调用（95、371、478、587 行）：

- 95 行 `gpu_barrier<ncclTeamTagRail>`：删除（Phase 1 用 `dist.barrier()` 替代）
- 371 行 `gin.put<ncclTeamTagRail>` reduce_recv_buffer put：替换为 D2H TransferCmd WRITE
- 478 行 `gin.put<ncclTeamTagRail>` combine payload put：替换为 D2H TransferCmd WRITE
- 587 行 `gin.red_add_rel<ncclTeamTagRail>` combine tail：替换为 scratch write + D2H TransferCmd WRITE

新增内核参数与 dispatch 完全对称（同样 10 个 EFA 参数）。

---

#### 4. `include/v2_efa/runtime.hpp`（在步骤 0c 已 fork+清理的基础上加字段）　~150 → ~230 行

> 前提：步骤 0c 已把 runtime.hpp fork 进 `ep/` 并删干净旧 scaffold 声明。本步只在
> `V2EfaRuntime` 类上加 MR/proxy 字段与方法（V2EfaConnectionHandle 在旧版根本没保留下来，
> 无需「删除」）。

```
新增字段：
  void*     signal_scratch_ = nullptr;    // GPU memory，cudaMalloc
  size_t    signal_scratch_bytes_ = 0;
  ibv_mr*   signal_scratch_mr_ = nullptr;
  ibv_mr*   buffer_mr_ = nullptr;        // V2 buffer MR（注册 ElasticBuffer.buffer）
  ibv_mr*   workspace_mr_ = nullptr;     // V2 workspace MR
  uint64_t  buffer_transport_offset_ = 0;
  uint64_t  workspace_transport_offset_ = 0;
  uint64_t  scratch_transport_offset_ = 0;
  UcclProxy* proxy_ = nullptr;           // 替代 V2EfaConnectionHandle 的 transport

新增方法声明：
  void register_v2_mrs(void* buf_ptr, size_t buf_bytes,
                       void* ws_ptr, size_t ws_bytes);
  void allgather_remote_mr_info(/* distributed handle */);
```

---

#### 5. `src/v2_efa_runtime.cc`（步骤 0d 清理后再实现）　288 → ~360 行

> 前提：步骤 0d 已删掉旧 scaffold 方法（worst_case_stats / DescriptorPlanStats 等），
> 只剩 RuntimeConfig 校验 + status。本步实现 `init_native_v2_efa_transport()`：

```
新增（单 window 模型）：
  1. signal_scratch = window 尾部 cpu_buffer 区域切片（不单独 cudaMalloc，复用 engram 空间）
  2. ibv_reg_mr(raw_workspace, ws+gpu_buffer+cpu_buffer bytes) → 单 mr（覆盖 workspace+buffer+scratch）
  3. 断言 window 总大小 < 16 GiB
  4. allgather 单个 remote base + rkey（约 20 行）
  5. 创建 UcclProxy 实例（per lane），proxy->connect(remote_info) 建 QP
  6. ctx->mr / ctx->remote_addr / ctx->remote_rkey / ctx->remote_len 指向该 window
注：现有 init_native_v2_efa_transport 已经在注册一个 window（_v2_efa_window），
   改为直接注册 DeepEP 的 raw_workspace window 即可，proxy.cpp 无需新增 chunk（步骤 10 取消）。
```

---

#### 6. `src/v2_efa_deep_ep_jit.cc`（已清理至 273 行）　273 → ~290 行

> dispatch 路径已在清理时接好 `DeviceToHostCmdBuffer**` + `signal_scratch_base`。本步只补 combine：

```
launch_native_hybrid_dispatch()：基本就绪
  - 已有：DeviceToHostCmdBuffer** d2h_queues, num_queues,
          buffer_base, workspace_base, signal_scratch_base, layout
  - 待补：buffer/workspace/scratch_transport_offset（如改用 offset 编码而非裸地址）

新增 launch_native_hybrid_combine()：
  - 与 dispatch 对称，同样传入 D2H queues + 三个 base + transport offset
```

---

#### 7. `src/uccl_ep.cc`（原 `uccl/ep` V1 binding）　2574 → ~2600 行

> `ep/uccl_ep.cc` 是原 UCCL V1 的 Python binding，**不含** V2EfaConnectionHandle
> （那只存在于 `uccl-ep`）。本步是**新增** V2 binding，不是删类。

```
保留：
  全部 V1 binding + UcclProxy binding（V2 proxy lifecycle 直接复用）

新增（约 +30 行）：
  init_native_v2_efa_transport() 暴露（接受 buffer/workspace Python tensor 指针）
  signal_scratch_ptr 属性 getter（GPU memory uintptr_t，供 JIT kernel 使用）
  buffer_transport_offset / workspace_transport_offset / scratch_transport_offset 属性
```

---

#### 8. `deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`　1967 → ~1820 行　-147

```
init_native_v2_efa_transport()（约 +40 行）：
  - 新增：注册 V2 buffer + workspace 的 MR（调用 ep.register_v2_mrs()）
  - 新增：一次性分配 num_lanes 个 D2H queues（persistent，不再 per-dispatch）
  - 删除：per-dispatch allocate_d2h_queue() 调用

_dispatch_native_hybrid()（约 -60 行，+30 行）：
  - 删除：每次 dispatch 分配新 D2H queues 的逻辑
  - 修改：kernel launch 参数增加 buffer_base, workspace_base, scratch_base,
          buffer_transport_offset, workspace_transport_offset, scratch_transport_offset
  - 删除：dispatch 完成后的 D2H queue 持有逻辑（queues 现在是 persistent 的）

_combine_native_hybrid()（新增，约 +80 行）：
  - 与 dispatch 对称，调用 launch_native_hybrid_combine()
  - 传入同样的 10 个 EFA 参数
```

---

#### 9. `Makefile`　161 → ~170 行　+9

```makefile
# 第 96 行，SRC_CU 由空改为：
SRC_CU := src/ep_runtime.cu src/internode.cu src/internode_ll.cu \
          src/intranode.cu src/layout.cu
```

---

#### 10. `src/proxy.cpp`（已在 uccl-ep，与 uccl/ep 同步）　~30 行改动

```
在 per_thread_rdma_init() 或 connect() 中：
  新增：把 V2 buffer_mr / workspace_mr / signal_scratch_mr 的 base/len/mr
        插入 ctx_.gpu_mr_chunks（已有的 chunk 向量），使 lkey_for(addr) / rkey_for(addr) 可查
  共约 20 行，集中在一个函数

无新增函数，无 API 变化，不影响 V1 路径
```

---

### 确认不带入 `ep/` 的废弃头（`uccl-ep` 里存在，但本 fork 不复制）

无「删除」动作——这些文件从未进 `ep/`。只需保证 `ep/` 内不再有任何 include 指向它们。

| `uccl-ep` 文件 | 处理 |
|------|------|
| `transfer_cmd.hpp` | 不带入；不抢救 `DispatchTransferLayout` / `CombineTransferLayout`，它们不属于 native V2 主路径 |
| `verbs_sink.hpp` | 不带入（V2EfaVerbsPostSink 废弃） |
| `efa_adapter.hpp` | 不带入（EfaPostSink / CoalescingEfaPostSink 废弃） |
| `uccl_transfer_adapter.hpp` | 不带入（adapter 层废弃） |
| `descriptor.hpp` | 不带入；`workspace.hpp` 去掉对它的 include（步骤 0b） |
| `dispatch_jit.cuh` / `combine_jit.cuh` | 不带入（device-side V2TransferCmd shim 废弃，已确认 `ep/` 中无） |
| `transfer_d2h_queue.cuh` / `proxy.hpp` | 不带入（已被 `ring_buffer.cuh` + 原 UCCL proxy 取代） |

---

### 修改量汇总

| 文件 | 当前行数 | 预计行数 | 净变化 |
|------|---------|---------|--------|
| `runtime.hpp`（fork+清理，步骤 0c→0c 后再加字段） | 0（未 fork） | ~230 | +230 |
| `workspace.hpp` | 61 | ~130 | +69 |
| `hybrid_dispatch_native.cuh` | 672 (原版 fork) | ~820 | +148 |
| `hybrid_combine_native.cuh` | 620 (原版 fork) | ~720 | +100 |
| `v2_efa_runtime.cc`（清理旧 scaffold + 重写） | 288 | ~360 | +72 |
| `v2_efa_deep_ep_jit.cc` | 273 (已清理) | ~290 | +17 |
| `uccl_ep.cc` | 2574 | ~2600 | +V2 binding/−死代码 |
| `elastic.py` | 1967 | ~1820 | -147 |
| `proxy.cpp` | 1557 | ~1587 | +30 |
| `Makefile` | 161 | ~170 | +9 |

> 注：`ep/` 是干净 fork，`uccl-ep` 里的废弃头从未带入，所以没有「删除 ~1282 行」这一项。
> `uccl_ep.cc` 是原 `uccl/ep` 的 V1 binding（2574 行），改动是**新增** V2 binding，
> 不是删 V2EfaConnectionHandle（后者只存在于 `uccl-ep`）。

---

### 实施顺序建议

```
0. 头文件归整         ← 编译前置：workspace.hpp 去 descriptor +
                        fork/清理 runtime.hpp + 清理 v2_efa_runtime.cc 旧 scaffold；
                        不保留 transfer_layout.hpp 死 descriptor
                        目标：ep/ 先恢复可编译（kernel 仍原版）
1. workspace.hpp      ← 加 signal scratch 常量和 helper
2. hybrid_dispatch_native.cuh ← 依赖 workspace.hpp，加参数 + 4 call site
3. hybrid_combine_native.cuh  ← 同上，3 call site
4. runtime.hpp        ← 在 0c 基础上加 MR/proxy 字段与方法声明
5. v2_efa_runtime.cc  ← 实现 MR 注册和 proxy 初始化
6. proxy.cpp          ← 加 chunk 注册（依赖 v2_efa_runtime.cc 设计确定后）
7. v2_efa_deep_ep_jit.cc  ← 接通新 kernel launch 参数
8. uccl_ep.cc         ← 新增 V2 binding，暴露 signal_scratch / transport offset 属性
9. elastic.py         ← Python 层接通（端到端 smoke test 之前最后改）
10. Makefile          ← 加 V1 .cu，最后确认编译
```
