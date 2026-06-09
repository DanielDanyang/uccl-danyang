# Worklog

## 2026-06-07: CX7 (GDAKI) vs EFA (proxy GIN) — GIN put microbenchmark

### 背景

拿到两台 GH200 + CX7 400G NDR InfiniBand 机器 (`gh200_0`/`gh200_1`，
hostname `mi-sky-gh200-1`/`mi-sky-gh200-2`)，想和现有 P5en EFA 做一个
apple-to-apple GIN put benchmark 对比。

### 环境搭建

- **GH200**: Grace-Hopper aarch64，CUDA 12.0 太旧不能直接 build NCCL 2.30.4。
  最终用 Docker `nvidia/cuda:13.2.0-devel-ubuntu24.04` 编译，提取 CUDA 13.2
  runtime libs 到 host 运行。NCCL 2.30.4 来自 pip `nvidia-nccl-cu13` (同样 aarch64)。
- **P5en**: x86_64，已有 CUDA 13.0 / NCCL 2.30.4 / aws-ofi-nccl master / MPI，
  直接 build。

关键发现：aarch64 glibc 的 `<bits/math-vector.h>` (SVE types) 在 nvcc 12.0
下无法编译，只能靠 Docker + CUDA 13.2 nvcc 绕过。

### 结果摘要

CX7 GDAKI (native IB) 对 EFA proxy GIN (2 rails, 400 Gbps 总带宽) 的对比：

| 指标 | CX7 GDAKI | EFA proxy | CX7/EFA |
|------|-----------|-----------|---------|
| 大包峰值 (1 GiB, remote) | 48.43 GB/s | 41.67 GB/s | 1.16× |
| 小消息 1 KiB | 25.53 GB/s | 0.94 GB/s | **27.2×** |
| 小消息 4 KiB | 48.95 GB/s | 3.81 GB/s | **12.8×** |
| 小消息 8 KiB | 49.56 GB/s | 7.61 GB/s | **6.5×** |
| 小消息 16 KiB | 49.22 GB/s | 14.92 GB/s | **3.3×** |
| 小消息 32 KiB | 49.55 GB/s | 26.20 GB/s | 1.89× |
| 饱和消息大小 | 2 KiB | 128 KiB | 64× |

详细报告: `ep/docs/uccl_gin_perf_cx7_vs_efa.md`

### 关键结论

1. **DeepEP 典型 7-8 KiB 消息粒度下 CX7 是 EFA 的 6-8×**，根因是 EFA proxy 路径
   每消息有 ~3-5 µs CPU 开销 (CQ poll/proxy progress/SRD reorder)。
2. CX7 GDAKI 可以不经 coalescing 在 2 KiB 直接到达线速；EFA proxy 需要
   aggressive coalescing 才能接近可用性能。
3. EFA proxy 大包性能可接受 (83% of 400G)，但小消息差距无法靠增加 rails 弥补。
4. 这验证了之前 EFA dispatch profiling 的瓶颈诊断 (3-6 GB/s SO)：proxy 的
   per-message 开销是根因，不是网卡带宽不够。

### 原始数据位置

- CX7: `/tmp/gin_bench_*` 在 `gh200_0`，EFA: `/tmp/gin_bench_efa_*` 在 `p5en_0`
- 报告: `ep/docs/uccl_gin_perf_cx7_vs_efa.md`

## 2026-06-05: UCCL-GIN dispatch ordering cleanup and V2 receiver readiness guard

### 背景

这轮目标是按 review 建议把 dispatch 重新拉回“尽量复用原 UCCL/EP transport substrate”的方向：

- 不再让 proxy 在看到 tail/ATOMIC 时自发做一层粗粒度 CQE 等待。
- 保持旧 `TransferCmd` + FIFO/proxy/CQ/ack 路径。
- payload 是否 ready 尽量在 receiver 侧判断，贴近原 `uccl/ep` 的 epoch-tag 思路。

### 修改 1: proxy 保持 D2H 命令顺序

旧实现把一个 batch 里的命令按类型分桶，然后固定按 `WRITE -> ATOMIC -> QUIET` 发。这个会破坏 device 侧原本的顺序，尤其是 `WRITE ... QUIET ... ATOMIC` 会被重排成 `WRITE ... ATOMIC ... QUIET`。

本轮把 `ep/src/proxy.cpp::post_gpu_commands_mixed()` 改成顺序处理：

- 连续 `WRITE` 仍批量 post。
- 连续 `ATOMIC` 仍批量 post。
- 遇到 `WRITE -> ATOMIC` 或 `ATOMIC -> WRITE` 时先 flush 前一类。
- 遇到 `QUIET/BARRIER` 时先 flush 前面的 `WRITE/ATOMIC`，再处理 control command。

这保留 batching，但不跨越 control command 重排。

### 修改 2: 删除 hot path 的 per-tail QUIET

我最初尝试在每次 UCCL-GIN tail add 前发 device-side `QUIET`，但小配置直接失败：

- 日志：`/tmp/uccl_gin_quiet_small_rank0.log`
- 现象：大量 `[UCCL-GIN quiet] waiting lane=... slot=...`
- 结论：per-channel/per-tail QUIET 太重，而且不是原 `uccl/ep` 的用法。原 V1 更像是在阶段边界用少量 quiet，真正的数据 ready 由 receiver-side epoch tag 判定。

因此 dispatch hot path 撤掉了 per-tail `gin.quiet()` 调用；`UCCLGin::quiet()` API 先保留，后续用于更粗粒度的阶段边界或其他 API 覆盖。

### 修改 3: receiver-side V2 readiness guard

无 QUIET 后，小配置通过，但 README-like 大配置出现 correctness failure：

- 日志：`/tmp/uccl_gin_noquiet_readme_rank1.log`
- 失败：
  `sorted_src_token_global_idx` 里出现重复 token，比如 `[0, 0, 1, ...]`

这说明 tail/count 已经让 forwarder 开始消费 slot，但对应 payload/metadata 还没有稳定可见。原 `uccl/ep` 在 `internode.cu` 中用 payload 里的 epoch tag 做 receiver-side spin-wait；V2 token layout 没有同样的 per-destination tag 字段，而且一个 source token 的 send buffer 可能发给多个 dst slot，不能直接把 dst-slot tag 塞进同一个 send token padding。

本轮先采用 V2 原生 metadata 的轻量替代：

- 在 forwarder TMA load 前读取 `token_buffer.get_src_token_global_idx_ptr()`。
- 对每个 `(channel, source scaleout rank)` 维护上一条已经消费的 `src_token_global_idx`。
- 等到 observed metadata 同时满足：
  - 来自预期 global rank：`recv_scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx`
  - 严格大于该 source stream 的上一条 token id
- 满足后才 TMA load 和 forward。

这不是完整 V1 epoch tag，但语义上回到了“receiver 不信 tail，先等 payload 自己的 ready/monotonic metadata”。

### 验证

编译：

- `p5en_0`: `make -C ep install ...` 通过
- `p5en_1`: `make -C ep install ...` 通过
- JIT header smoke 通过：
  `/tmp/uccl_gin_hybrid_dispatch_compile.cu`

小配置 correctness：

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-perf-test \
  --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256
```

- rank0 log: `/tmp/uccl_gin_ready_small_rank0.log`
- rank1 log: `/tmp/uccl_gin_ready_small_rank1.log`
- 结果：两边 exit code 0。

README-like EP8 x 2 / EP16：

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only \
  --num-sms 20 --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256
```

- rank0 log: `/tmp/uccl_gin_ready_readme_rank0.log`
- rank1 log: `/tmp/uccl_gin_ready_readme_rank1.log`
- 结果：两边 exit code 0。

关键性能：

- rank0 dispatch: 约 `36 GB/s (SO)`，`115-119 GB/s (SU)`，约 `3.4 ms`
- rank1 dispatch:
  - EP12: `36 GB/s (SO)`
  - 其他多数 EP: 约 `18 GB/s (SO)`，约 `6.7 ms`
- combine/reduced combine: 多数约 `17 GB/s (SO)`，个别 EP 约 `22 GB/s (SO)`

### 当前判断

- correctness 现在依赖 receiver-side metadata monotonic guard，而不是 proxy-side coarse CQE fence。
- 这比 per-tail QUIET 更接近原 UCCL/EP 的“receiver 侧 ready 判定”哲学。
- 性能仍有明显 rank/EP imbalance，下一步应继续看：
  - FIFO/proxy lane 映射是否把部分 channel 压到少数 proxy thread/NIC。
  - `channel_idx % num_queues` 是否应该改成更接近原 UCCL/EP 的 channel/proxy 映射。
  - 是否需要完整 V2 per-slot ready tag buffer，而不是复用 `src_token_global_idx` 单调性。

## 2026-06-04: native V2 dispatch 远端验证与 bug 记录

### 环境

- 本地仓库：`/Users/daniel/Documents/code/uccl-danyang`
- 远端仓库：`/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang`
- 机器：`p5en_0` + `p5en_1`
- 运行方式：每台 8 个 local process，总 world size 16，也就是 EP8 x 2 / EP16。
- 关键环境：
  - `source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate`
  - `CUDA_HOME=/usr/local/cuda-13.0`
  - `LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:$NCCL_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH`
  - `EP_SUPPRESS_NCCL_CHECK=1`
  - `EP_REUSE_NCCL_COMM=0`
  - `NCCL_NET_PLUGIN=ofi`
  - `FI_PROVIDER=efa`
  - `FI_EFA_USE_DEVICE_RDMA=1`
  - `OFI_NCCL_FORCE_NUM_RAILS=4`
  - `NCCL_SOCKET_IFNAME=enp71s0`
  - `DEEPEP_REPO_ROOT=/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang`
- 每次上服务器跑前都检查过 `nvidia-smi`，当时两台没有其他用户 GPU 进程。

### 代码同步

这次主要同步了两个 Python 文件到两台机器：

- `ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
- `ep/tests/v2_efa_native_dispatch_smoke.py`

用 `sha256sum` 确认过本地、`p5en_0`、`p5en_1` 三处一致：

- `elastic.py`: `40aab084909c32af9376636c39b48163d20b814e6ce7c52f940c8dc46129dc24`
- `v2_efa_native_dispatch_smoke.py`: `42c33719430ba3247fd8ef158898cf2c850983262d6c8f21e12c81f89e743dae`

### Bug 1: correctness PASS 后 worker 退出时 SIGSEGV

现象：

- EP16 小配置已经打印 `native_dispatch_correctness PASS`。
- 所有 worker 都能走到 trace 里的 `done`。
- 但 `torch.multiprocessing.spawn` 最后仍报某个 child `SIGSEGV`。

定位：

- 之前 `cudaFree failed: invalid argument` 已经通过 `owns_gpu_buffer=False` 修过，因为 native V2 proxy 注册的是 DeepEP 自己持有的 symmetric window，UCCL proxy 不应该 `cudaFree` 这个 window。
- 剩余 SIGSEGV 出现在 Python/C++ 对象 finalization 阶段。
- `ep.register_proxies(local_rank, proxies)` 会把 `nb::object` 存在 C++ 全局 `g_proxies_by_dev` 里。
- `wrap.destroy()` 之前只清了 Python wrapper 的 `_v2_proxies`，没有清 C++ registry，所以 proxy 对象可能被拖到解释器退出时才析构。

修复：

- 在 `ElasticBuffer.destroy()` 中：
  - `torch.cuda.synchronize()`
  - 遍历 `_v2_proxies` 调 `proxy.stop()`
  - 调 `ep.unregister_proxy(device_index)` 清 C++ 全局 registry
  - 清 `_v2_d2h_queue_ptrs`
- `init_native_v2_efa_transport()` 中记录 `_v2_proxy_device_index = local_rank`，供 destroy 使用。

验证：

- 命令：
  ```bash
  python ep/tests/v2_efa_native_dispatch_smoke.py \
    --num-processes 8 --tokens 64 --hidden 2048 --sms 8 --lanes 4
  ```
- 结果：
  - rank0 日志打印：
    `native_dispatch_correctness PASS world=16 tokens=64 hidden=2048 sms=8 lanes=4`
  - `p5en_0` 和 `p5en_1` 两端 exit code 都是 0。
- 可看日志：
  - `/tmp/uccl_v2_dispatch_smoke_ep16_rank0.log`
  - `/tmp/uccl_v2_dispatch_smoke_ep16_rank1.log`
  - 后续 sizing 修复后的日志：
    - `/tmp/uccl_v2_dispatch_smoke_ep16_rank0_after_size.log`
    - `/tmp/uccl_v2_dispatch_smoke_ep16_rank1_after_size.log`

### Bug 2: 大配置 1GB fixed window 触发 CUDA illegal address

现象：

- 大配置：
  ```bash
  python ep/tests/v2_efa_native_dispatch_smoke.py \
    --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 --lanes 4 --iters 10 --perf
  ```
- 使用旧 smoke 默认 `--window-mb 1024` 时，报：
  - `CUDA_ERROR_ILLEGAL_ADDRESS`
  - stack 落在 `launch_dispatch_copy_epilogue`
- 即使设置 `UCCL_V2_DISPATCH_LAUNCH_ONLY=1`，也会在主 dispatch kernel 同步时报 illegal address。

定位：

- 这个不是单纯 epilogue 的问题，因为 launch-only 也失败。
- 原因是 native wrapper 绕过了上游 DeepEP C++ 里的 buffer size assert：
  - 上游在 `thirdparty/DeepEP-v2-d4f41e4/csrc/elastic/buffer.hpp` 中用
    `get_dispatch_buffer_size(...) <= num_buffer_bytes` 检查。
  - 我们走 Python wrapper + native JIT path，没有经过这个 host assert。
- 对 `tokens=8192 hidden=7168 EP16`，1GB GPU buffer 小于 V2 hybrid dispatch layout 需求；kernel 会写出 DeepEP buffer 范围，导致 illegal address。

修复：

- 在 wrapper 加 `_v2_dispatch_buffer_bytes(...)`，按 DeepEP V2 hybrid dispatch 的
  `BufferLayout<false>` 公式计算所需字节数。
- 在 `_dispatch_native_hybrid()` 里根据当前 JIT 参数：
  - `token_layout_bytes`
  - `num_max_tokens_per_rank`
  - `num_scaleout_ranks`
  - `num_scaleup_ranks`
  - `num_channels`
  检查 `required_buffer_bytes <= _v2_buffer_usable_bytes`。
- 如果不够，直接抛明确错误，不再让 CUDA kernel 走 undefined behavior。
- smoke 默认改为：
  - `--window-mb 0`
  - 使用上游 `deep_ep.ElasticBuffer.get_buffer_size_hint(...)`
  - 再额外加 `--extra-window-mb 16` 给 native EFA signal scratch tail。

验证：

- 小配置仍然 PASS：
  - `/tmp/uccl_v2_dispatch_smoke_ep16_rank0_after_size.log`
  - `/tmp/uccl_v2_dispatch_smoke_ep16_rank1_after_size.log`
- 大配置不再报 `CUDA_ERROR_ILLEGAL_ADDRESS`，说明 buffer 越界问题被修掉了。

### Bug 3 / 当前阻塞: 大配置 tail/forwarding timeout

现象：

- 大配置使用上游 size hint 后继续跑：
  ```bash
  python ep/tests/v2_efa_native_dispatch_smoke.py \
    --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 --lanes 4 --iters 10 --perf
  ```
- 不再出现 illegal address。
- 但出现协议层 timeout：
  - `DeepEP hybrid dispatch (forwarding) timeout`
  - `DeepEP NVLink barrier timeout, tag: 7`
  - 最终 `CUDA_ERROR_LAUNCH_FAILED`
- 典型日志片段：
  - rank0 侧：
    `DeepEP hybrid dispatch (forwarding) timeout, scale-out: 0, scale-up: 1, channel: 66, lane: 0, old scale-out tail: 0, scale-out tail: (1, 0)`
    `DeepEP hybrid dispatch (forwarding) timeout, scale-out: 0, scale-up: 1, channel: 66, lane: 1, old scale-out tail: 102, scale-out tail: (0, 102)`
  - rank1 侧也有对称现象：
    `old scale-out tail: 102, scale-out tail: (0, 102)` 和另一个 lane 的 finish flag。

日志：

- `/tmp/uccl_v2_dispatch_perf_ep16_rank0_after_size.log`
- `/tmp/uccl_v2_dispatch_perf_ep16_rank1_after_size.log`

当前判断：

- 这次失败说明大配置主 dispatch 已经进入真实通信协议问题，而不是简单 buffer size。
- 之前怀疑过 `signal_scratch` slot 在 RDMA completion 前被复用，但重新对照代码后，这个假设不成立：
  - `ep/include/ring_buffer.cuh` 的 `advance_tail_from_mask()` 只跨过连续 acked slot，并用 release store 把 tail 发布给 GPU。
  - `ep/src/proxy.cpp` 的非 FIFO backend 只在 CQE 回来后通过 `mark_acked()` 标记 slot，再推进 ring tail。
  - native tail scratch slot 绑定 D2H ring slot；只要走当前 completion-gated ring backend，scratch slot 生命周期跟 WR completion 对齐，不会在对应 WR 完成前被 GPU 复用。
- 更合理的根因是 AWS EFA/SRD 的乱序语义：
  - `2512.19849v2.pdf` 的 AWS/EFA 部分明确指出 EFA SRD 是 reliable but unordered，原 UCCL-EP 需要用 FIFO/channel ordering、sequence number、RDMA immediate control buffering 等机制在软件层补 ordering。
  - 原 DeepEP V2 的 scaleout tail 更新使用 `gin.red_add_rel(...)` 写 delta，语义上依赖 release/order 或可交换的累加更新。
  - 当前 native V2 fork 把它替换成对同一个远端 tail word 的多次绝对值 `WRITE`：`(finish_flag, stored_scaleout_tail)`。
  - 在 EFA SRD 上，同一个 tail word 的多个绝对值 WRITE 可能乱序到达；最终落地的可能是陈旧中间值，例如 `(0, 102)` 覆盖后到达的 `(1, N)`，forward warp 就永远等不到 finish。
- 典型症状也吻合：
  - `lane 0: (1, 0)` 可以是无 token lane 的正常 finish。
  - `lane 1: old tail 102, tail (0,102)` 是目标 lane 卡在中间 tail，最后 finish 没有稳定落地。
  - 小配置 token 少，每个 `(channel,lane)` 的 tail update 次数少，乱序覆盖窗口小，所以能过；大配置把很多 token 压到同一个 lane，tail write 次数多，问题暴露。

下一步建议：

- 不要调 timeout；这不是超时参数问题。
- 先做一个确认实验：
  - 把测试路由从“所有 token 压到同一个 remote dst/lane”临时改成均匀分散到多个 scaleout dst；如果大配置变绿，基本坐实同一 tail word 多次绝对值写的乱序覆盖。
  - 或者临时让 proxy 对 tail WRITE 做强 completion fence：前一个 tail WRITE CQE 回来后再 post 下一个 tail WRITE；如果变绿，也能确认 ordering 是根因。
- 正式修复方向：
  - correctness-first：复用原 UCCL proxy 的 quiet/barrier/CQ/ack 思路，让同一 `(channel, dst tail word)` 上的 tail WRITE 有序，且 payload WRITE 在对应 tail WRITE 前完成或被 fence。
  - 这会牺牲一部分 streaming 性能，但能先恢复正确性；后续再减少 tail update 频率、按 channel batching、或用 UCCL-EP 论文里的 sequence/immediate control buffering 做更高性能的软件 ordering。
  - 不要只在 receiver 侧取 `max(count)`；如果最终落地值被旧 WRITE 覆盖，receiver 可能根本没 poll 到 finish，单纯 latch/max 不够鲁棒。

### 已知噪音

- 日志里有很多：
  `Remote atomic buffer not registered`
- 当前 native V2 dispatch 主路径没有依赖 remote atomic buffer；这些 warning 来自 UCCL proxy 旧路径的元数据/atomic 支持，暂时不是本次 correctness 小配置失败或大配置 timeout 的直接根因。

## 2026-06-04: tail ordering 实验和 software-atomic tail shadow

### AGENTS 临时策略

- 用户明确说明最近服务器无人使用，暂时不用反复执行 GPU 空闲检查。
- 已更新 `agents.md`：保留“不打断别人任务”的长期原则，但当前阶段按用户临时指令不反复执行空闲检查。

### 论文和代码复核

- 阅读 `2512.19849v2.pdf` 的 AWS/EFA 部分后，确认 EFA SRD 是 reliable but unordered。
- 论文中 UCCL-EP 的关键做法不是假设 NIC 有序，而是通过 FIFO/channel mapping、sequence number、RDMA immediate control buffering 等软件机制补 ordering。
- 重新复核当前 `ep` 代码后，之前的 scratch 过早复用假设不成立：
  - `ring_buffer.cuh::advance_tail_from_mask()` 只推进 contiguous acked slot。
  - `proxy.cpp::notify_gpu_completion()` 只在 CQE 后 `mark_acked()`，再推进 ring tail。
  - 因此当前 ring backend 下，D2H slot/scratch slot 生命周期是 completion-gated。

### 诊断改动

- `ep/tests/v2_efa_native_dispatch_smoke.py`
  - 新增 `--route-mode spread-remote`，用于把 token 分散到远端 8 个 local rank，减少单个 tail word 的更新压力。
  - 加了 import hygiene：加载 wrapper 前从 `sys.path` 移除 repo root，避免源码目录 `uccl/` shadow venv 中安装的 `uccl.ep`。
- `ep/src/proxy.cpp`
  - 新增诊断环境变量 `UCCL_V2_SERIALIZE_RDMA_WRITES=1`。
  - 打开后普通 RDMA WRITE 一条一条 post，并等 CQE 后再发下一条；这只用于确认 ordering，不是最终性能路径。
- `ep/include/v2_efa/hybrid_dispatch_native.cuh`
  - scaleout tail 从单个 packed absolute tail word 改为两个 additive shadow slot：
    - `count`
    - `finish`
  - 远端 tail 更新改为旧 `TransferCmd::ATOMIC`，复用 UCCL EFA software atomic immediate 路径，由 receiver CPU proxy 对 mapped atomic buffer 做 `fetch_add`。
  - forward warp 从 mapped atomic tail buffer 读 `count/finish`。
- `ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`
  - 所有 proxy thread 共享 thread 0 分配的 atomic buffer pointer。
  - 该 pointer 作为 `atomic_tail_base` 传入 V2 native JIT。

### 远端验证

构建：

```bash
make -C ep install PYTHON="$VIRTUAL_ENV/bin/python" CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j
```

结果：通过。

环境修正：

- 上游 DeepEP V2 的 `check_nccl_so()` 会把 `libnccl-net-ofi.so` 和 `libnccl-tuner-ofi.so` 误判成重复 NCCL runtime；使用上游已有开关：
  - `EP_SUPPRESS_NCCL_CHECK=1`
- 运行测试时不能把 repo root 放进 `PYTHONPATH`，否则会 shadow venv 里的 `uccl.ep`；只放：
  - `$PWD/thirdparty/DeepEP-v2-d4f41e4`

测试 1：大配置 spread-remote 诊断

```bash
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 \
  --lanes 4 --iters 2 --perf --route-mode spread-remote
```

结果：失败，仍是 forwarding timeout。

日志：

- `/tmp/uccl_v2_dispatch_spread_ep16_rank0.log`
- `/tmp/uccl_v2_dispatch_spread_ep16_rank1.log`

测试 2：全 RDMA WRITE serialization 诊断

```bash
UCCL_V2_SERIALIZE_RDMA_WRITES=1 \
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 \
  --lanes 4 --iters 1 --perf --route-mode paired-remote
```

结果：失败，仍是 forwarding timeout。

日志：

- `/tmp/uccl_v2_dispatch_serial_ep16_rank0.log`
- `/tmp/uccl_v2_dispatch_serial_ep16_rank1.log`

测试 3：software-atomic tail shadow，小配置冷 JIT cache

先清 cache：

```bash
rm -rf ~/.deep_ep
```

再跑：

```bash
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 64 --hidden 2048 --sms 8 \
  --lanes 4 --route-mode paired-remote
```

结果：通过。

日志：

- `/tmp/uccl_v2_dispatch_atomic_tail_small_cold_rank0.log`
- `/tmp/uccl_v2_dispatch_atomic_tail_small_cold_rank1.log`
- rank0 关键行：
  - `native_dispatch_correctness PASS world=16 tokens=64 hidden=2048 sms=8 lanes=4 route_mode=paired-remote`

测试 4：software-atomic tail shadow，大配置冷 JIT cache

```bash
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 \
  --lanes 4 --iters 2 --perf --route-mode paired-remote
```

结果：失败，仍是 forwarding timeout。

日志：

- `/tmp/uccl_v2_dispatch_atomic_tail_big_cold_rank0.log`
- `/tmp/uccl_v2_dispatch_atomic_tail_big_cold_rank1.log`

典型现象：

- `old scale-out tail: 102, scale-out tail: (0, 102)`
- 另一 lane 常见 `scale-out tail: (1, 0)`

### 当前判断更新

- 简单“EFA SRD 乱序覆盖 absolute tail write”不是完整解释：
  - spread-remote 没有修好。
  - 全 WRITE serialization 没有修好。
  - software-atomic additive tail shadow 也没有修好大配置。
- 但小配置仍通过，说明新的 software-atomic tail shadow 没有破坏基本路径。
- 大配置稳定卡在 `tail=102`，更像是以下之一：
  - receiver forward warp 的 channel slot 语义和 native payload 写入 slot 不匹配；
  - 大配置下某些 payload WR 没有按 forward 期待的 slot/layout 落地；
  - receiver CPU proxy 没有及时消费/应用某些 software atomic immediate，导致 count/finish 只到中间值；
  - tail/count 索引缺少某个 V2 维度，导致多个 writer/reader 混到同一个 shadow slot。

### 下一步

- 给 EFA software atomic path 加最小 instrumentation：
  - sender 侧统计 `TransferCmd::ATOMIC` post 数量和值；
  - receiver 侧统计 `IBV_WC_RECV_RDMA_WITH_IMM` atomic CQE 数量、offset、value；
  - 对 timeout 中的 `(channel,lane)` 反推 shadow offset，确认 receiver atomic buffer 里实际 count/finish 值。
- 同时复核 V2 `scaleout_recv_buffer.get_rank_buffer(...).get_channel_buffer(...)` 的 writer/reader rank 维度，确认 native payload 和 forward warp 是否使用同一套 channel/lane/slot 解释。

## 2026-06-04：dispatch tail signal 改成 ordered packed atomic

### 背景

上一轮怀疑过 EFA SRD 对 absolute tail WRITE 的乱序覆盖，但实验已经否定：

- spread remote 不能修 timeout；
- serialize RDMA WRITE 不能修 timeout；
- software atomic count/finish shadow 能过小配置，但大配置仍 timeout 或 correctness 失败。

今天继续排查后，关键现象变成：

- sender scaleout warp 能走到 finish，`UCCL_V2_FINISH_EXIT` 有输出；
- proxy 没有 `EMPTY` gap；
- `psum_num_recv_tokens_per_scaleup_rank[-1] == 8192`，总 token 数正确；
- 但 `recv_src_metadata` 有重复/缺失，说明 forward/copy 阶段看到的 token stream 不完整。

### 真正问题

旧的 native tail shadow 把原 DeepEP V2 的单个 packed `red_add_rel` 拆成了两个
software atomic word：

```text
count_word  += tail_delta
finish_word += 1
```

这和原 GIN 语义不等价。receiver forward warp 可能先看到 `finish_word == 1`，
然后在最后几个 `count_word` update 到达或被 CPU proxy apply 之前退出 channel。
这样会出现：

```text
notify/psum 总数正确
tail 最终也可能正确
但 forward 已经提前停止，metadata/token 少量缺失或重复
```

所以问题不是“sender 没发 finish”，也不是“payload slot 生命周期”，而是 V2 tail
signal 的可见顺序被拆 word 破坏了。

### 修改

文件：

- `ep/include/v2_efa/hybrid_dispatch_native.cuh`
- `ep/src/rdma.cpp`

设计：

```text
原来:

  sender GPU
    count delta  ---- ATOMIC imm ----> receiver CPU proxy ----> count_word
    finish       ---- ATOMIC imm ----> receiver CPU proxy ----> finish_word

现在:

  sender GPU
    packed_delta = count_delta + finish * (max_tokens_per_channel + 1)
       |
       v
    ordered ATOMIC imm stream, same offset, with UCCL PackAtomicWithSeq
       |
       v
  receiver CPU proxy reorder buffer
       |
       v
    one tail_word += packed_delta

  receiver GPU forward:
    finish = tail_word / finish_delta
    count  = tail_word % finish_delta
```

关键点：

- 一个 `(channel, source_scaleout_rank)` 只有一个 tail word，和原 DeepEP V2 的
  packed signal 语义一致。
- 远端 EFA immediate atomic 使用 UCCL 已有 `PackAtomicWithSeq` 和 receiver reorder
  buffer，保证同一个 tail word 的 count delta 在 finish delta 前 apply。
- local scaleout rank 直接对同一个 tail word 做 `red_add_rel_sys`。
- 保留旧 16B `TransferCmd`；只用 `TransferCmd::atomic_offset != 0` 作为 ordered
  atomic 标志，没有引入新 wire command。

### 服务器验证

构建：

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang
source /home/ubuntu/.venvs/deepep-danyang-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
make -C ep install PYTHON="$VIRTUAL_ENV/bin/python" CUDA_PATH="$CUDA_HOME" SM=90 -j8
```

注意：`p5en_0` 和 `p5en_1` 共享 EFS，扩展链接建议串行做，避免 `Stale file handle`。

测试命令：

```bash
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 \
  --lanes 4 --iters 1 --perf --route-mode paired-remote
```

环境：

```bash
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:/home/ubuntu/.venvs/deepep-danyang-cu13/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/thirdparty/DeepEP-v2-d4f41e4:$PYTHONPATH
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=4
export NCCL_NET_PLUGIN=ofi
export NCCL_SOCKET_IFNAME=enp71s0
unset EP_DISABLE_GIN
unset OFI_NCCL_GIN_GDAKI
```

结果：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=paired-remote
native_dispatch_perf per_iter=4.630ms approx_per_rank_BW=25.36GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_ordered_tail_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_ordered_tail_rank1.log`

缓存后 10-iteration perf：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=paired-remote
native_dispatch_perf per_iter=4.330ms approx_per_rank_BW=27.12GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_ordered_tail_iter10_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_ordered_tail_iter10_rank1.log`

### Tail batching sweep

新增 `UCCL_V2_SCALEOUT_UPDATE_INTERVAL` JIT 参数，默认后来调成 32。它只影响
native EFA dispatch 的 tail signal batch 粒度，不改变 V2 buffer/layout 语义。

| interval | correctness | per_iter | approx per-rank BW | 日志 |
| --- | --- | ---: | ---: | --- |
| 3 | PASS | 4.330 ms | 27.12 GB/s | `/tmp/uccl_v2_dispatch_ordered_tail_iter10_rank0.log` |
| 8 | PASS | 3.905 ms | 30.07 GB/s | `/tmp/uccl_v2_dispatch_interval_8_rank0.log` |
| 16 | PASS | 4.104 ms | 28.62 GB/s | `/tmp/uccl_v2_dispatch_interval_16_rank0.log` |
| 32 | PASS | 3.867 ms | 30.37 GB/s | `/tmp/uccl_v2_dispatch_interval_32_rank0.log` |
| 64 | PASS | 3.935 ms | 29.84 GB/s | `/tmp/uccl_v2_dispatch_interval_64_rank0.log` |
| 128 | PASS | 3.990 ms | 29.43 GB/s | `/tmp/uccl_v2_dispatch_interval_128_rank0.log` |

结论：EFA 上 tail software atomic 的开销确实可见，coarser batching 能把 BW 从
约 27 GB/s 提到约 30 GB/s；但继续增大 interval 会损失 forward overlap，32 在这组
EP8 x 2 paired-remote case 中最好。

默认值已从上游 GIN 的 3 调成 EFA native 的 32，并保留
`UCCL_V2_SCALEOUT_UPDATE_INTERVAL` 覆盖。默认路径验证：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=paired-remote
native_dispatch_perf per_iter=3.954ms approx_per_rank_BW=29.70GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_default_interval32_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_default_interval32_rank1.log`

### 当前状态

- EP8 x 2 native V2 dispatch correctness 已通过。
- 当前 paired-remote dispatch 约 `30.37 GB/s` per-rank approximate BW。
- 还没有进入 combine。
- 距离 README SM90 EP16 `90 GB/s` 仍有明显差距，下一步应 profile ordered tail
  的 proxy apply / CQ poll 开销，以及把 perf test 从单次 smoke 扩展到更稳定的
  README 风格 sweep。

## 2026-06-04 dispatch profiling / proxy layout sweep

### 先处理 review 中确定合理的代码问题

本轮继续针对 native V2 dispatch 的性能 review 做收敛。按 `agents.md` 要求，在所有
远端 build / benchmark 前都检查两台机器 GPU compute app 是否为空：

```bash
ssh p5en_0 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
ssh p5en_1 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

本轮所有远端实验前检查结果均为空。

落地的代码修正：

- `agents.md`：恢复/强化 GPU 空闲检查要求。之后远端 build、test、profiling、
  benchmark 前必须查两台机器；如果发现其他用户或未知 GPU 进程，立即停止远端操作。
- `ep/include/v2_efa/hybrid_dispatch_native.cuh`：
  - 删除 `UCCL_V2_DISPATCH_DEBUG_STAGE` 早退 scaffold，避免 hot kernel 里保留
    调试分阶段返回路径。
  - 增加 tail software-atomic reorder window 静态约束：
    `ceil(kNumMaxTokensPerChannel / kScaleoutUpdateInterval) + 1 <= kReorderingBufferSize`。
    这是 load-bearing 的 correctness 条件；否则 4-bit seq wrap 可能让同一
    `(channel, source)` tail word 的 update 乱序重放时发生 silent corruption。
- `ep/include/v2_efa/jit_plan.hpp`：删除 `UCCL_V2_DISPATCH_DEBUG_STAGE` 的 JIT 注入。
- `ep/include/uccl_proxy.hpp` / `ep/src/uccl_ep.cc`：导出
  `UcclProxy.get_atomic_buffer_bytes()`。
- `ep/deep_ep_v2_wrapper/deep_ep/buffers/elastic.py`：
  - 初始化 native V2 EFA 后记录 `_v2_atomic_tail_bytes`。
  - dispatch 前检查 `num_channels * num_scaleout_ranks * 8` 是否被 UCCL atomic
    tail shadow buffer 覆盖，避免更大 SM/channel 配置在 kernel 中途 abort。
  - 新增 `UCCL_V2_PROFILE_TIMINGS=1` CUDA event profiling，记录
    `dispatch_ms`、`copy_epilogue_ms`、`gpu_total_ms`。
- `ep/tests/v2_efa_native_dispatch_smoke.py`：perf 模式下打印
  `native_dispatch_split ...`。
- `ep/src/proxy.cpp`：EFA normal mode 使用 WRITE-with-imm software atomic，由
  receiver proxy apply 到本地 atomic buffer，不需要远端 hardware atomic MR。
  因此默认静默 `Remote atomic buffer not registered` setup warning，仅在
  `UCCL_PROXY_DEBUG_ATOMIC_BUFFER=1` 或非 normal/LL 路径打印。
- `ep/Makefile` / `ep/include/common.hpp`：新增编译期 proxy layout knob：
  `NUM_PROXY_THREADS` 和 `CHANNEL_PER_PROXY`。默认保持原 UCCL/V1 的 `4 x 8`，
  但可以重编译测试 `8 x 4`、`8 x 8` 等布局。

默认 `4 x 8` clean rebuild 已通过，两台机器构建日志中均可见：

```text
UCCL proxy layout: NUM_PROXY_THREADS=4, CHANNEL_PER_PROXY=8
```

构建 warning 仍是旧 V1 binding 的 `std::optional<std::function<void()>> recv_hook`
maybe-uninitialized，以及 nanobind `NB_STABLE_ABI` redefine；不是本轮 native V2
改动引入的 fatal error。

### H1: dispatch vs copy epilogue

review 猜测 timed window 可能主要被 `dispatch_copy_epilogue` 吃掉。用
`UCCL_V2_PROFILE_TIMINGS=1` 在默认 `4 x 8` 上拆分。

命令核心参数：

```bash
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 \
  --lanes 4 --iters 10 --perf --route-mode spread-remote
```

结果：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=spread-remote
native_dispatch_perf per_iter=4.010ms approx_per_rank_BW=29.29GB/s
native_dispatch_split dispatch=3.631ms copy_epilogue=0.170ms gpu_total=3.822ms
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_default4x8_split_spread_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_default4x8_split_spread_rank1.log`

结论：H1 在当前 native V2 EFA 路径上不成立。copy epilogue 只有约 `0.17 ms`，
主耗时是 dispatch/transport 本体约 `3.63 ms`。host timed window 比 GPU event
总时间多约 `0.19 ms`，主要是 CPU launch/sync/allocation 等开销。

### H4: paired-remote vs spread-remote

默认 `4 x 8`，不启用 profiling，避免 CUDA event 影响 perf。

paired-remote：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=paired-remote
native_dispatch_perf per_iter=3.879ms approx_per_rank_BW=30.28GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_default4x8_paired_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_default4x8_paired_rank1.log`

spread-remote：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=spread-remote
native_dispatch_perf per_iter=3.968ms approx_per_rank_BW=29.59GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_default4x8_spread_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_default4x8_spread_rank1.log`

结论：H4 不成立或很弱。把 token 从单 peer 集中改成 spread 并没有显著提升，甚至在
这次 run 中略慢。当前瓶颈不是简单的“所有流量压到一个远端 peer”。

### NIC 使用情况

默认 `4 x 8` 和后续 `8 x 4` / `8 x 8` 日志都显示每节点 16 个 EFA NIC 都被选到。
例如默认日志中可见：

```text
rdmap85s0 rdmap86s0 rdmap87s0 rdmap88s0
rdmap110s0 rdmap111s0 rdmap112s0 rdmap113s0
rdmap135s0 rdmap136s0 rdmap137s0 rdmap138s0
rdmap160s0 rdmap161s0 rdmap162s0 rdmap163s0
```

结论：H3 中“整机只用少数 NIC”的说法不成立。更精确地说，当前每个 GPU 主要使用
与其 NUMA/GPU 邻近的 2 个 EFA NIC；整机 EP16 聚合时覆盖了全部 16 个 EFA NIC。

### H2/H3: proxy threads / ring 数 sweep

先确认了一个脚本层问题：`--lanes` 当前不是 proxy thread 数。native wrapper 实际用
编译出的 `ep.get_num_proxy_threads()` 创建 proxy，每个 `UcclProxy` 再创建
`CHANNEL_PER_PROXY` 条 D2H ring。因此如果不重编译，`--lanes 4/8/16` sweep 会误导。

为此新增 build knob 后做了两组 sweep。

#### 8 x 4: 更多 proxy threads，总 ring 数仍 32

构建：

```bash
make -C ep install ... NUM_PROXY_THREADS=8 CHANNEL_PER_PROXY=4
```

结果：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=spread-remote
native_dispatch_perf per_iter=3.942ms approx_per_rank_BW=29.80GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_8x4_spread_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_8x4_spread_rank1.log`

#### 8 x 8: 更多 proxy threads，同时总 ring 数增到 64

构建：

```bash
make -C ep install ... NUM_PROXY_THREADS=8 CHANNEL_PER_PROXY=8
```

结果：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=spread-remote
native_dispatch_perf per_iter=4.069ms approx_per_rank_BW=28.86GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_8x8_spread_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_8x8_spread_rank1.log`

结论：

- `4 x 8` spread: `29.59 GB/s`
- `8 x 4` spread: `29.80 GB/s`
- `8 x 8` spread: `28.86 GB/s`

增加 proxy thread 或 D2H ring/QP 没有带来明显收益，`8 x 8` 还略差。当前 30 GB/s
附近的瓶颈不像是简单的 proxy thread 数不足或 ring head contention。更可能的问题是：

- 每 token 仍有大量 7-8 KiB payload WRITE，CPU proxy per-WR post/poll/apply 成本高；
- tail batching 已经把 signal 成本压了一部分，但 payload 仍是细粒度 token write；
- GPU 侧 scaleout warp 按 token enqueue，整体协议还没有形成 UCCL/EP 风格的
  per-expert semantic batching；
- EFA normal path 的 WRITE-with-imm software atomic/reorder apply 不是主瓶颈，但仍有
  固定开销。

本轮结束时已把服务器 extension 重新构建回默认 `4 x 8`，避免留下实验 layout。

### Proxy command profile

为直接验证 per-WR 压力，新增 `UCCL_PROXY_PROFILE_COMMANDS=1`。它不改变主路径，只在
proxy 线程退出时打印该线程处理的 command 构成。实现上用单次 `write(2)` 输出整行，
避免多 Python rank 共享 stderr 时日志交错。

命令核心参数：

```bash
export UCCL_PROXY_PROFILE_COMMANDS=1
python ep/tests/v2_efa_native_dispatch_smoke.py \
  --num-processes 8 --tokens 8192 --hidden 7168 --sms 20 \
  --lanes 4 --iters 10 --perf --route-mode spread-remote
```

结果：

```text
native_dispatch_correctness PASS world=16 tokens=8192 hidden=7168 sms=20 lanes=4 route_mode=spread-remote
native_dispatch_perf per_iter=3.991ms approx_per_rank_BW=29.43GB/s
```

日志：

- 本机 `/tmp/uccl_v2_dispatch_proxy_profile3_spread_rank0.log`
- 本机 `/tmp/uccl_v2_dispatch_proxy_profile3_spread_rank1.log`

两节点各解析到 `32` 条 `UCCL_PROXY_PROFILE` 行，即 `8 local ranks x 4 proxy threads`。
每个节点在整个 smoke run 中的聚合 command 量完全一致：

```text
total_cmds   817,344
write_cmds   786,624
atomic_cmds   30,720
write_bytes 11,299,461,120
```

按 proxy thread 聚合：

| thread | write_cmds | atomic_cmds | write_bytes |
| --- | ---: | ---: | ---: |
| 0 | 235,968 | 9,216 | 3,387,635,712 |
| 1 | 235,776 | 9,216 | 3,387,629,568 |
| 2 | 157,440 | 6,144 | 2,262,097,920 |
| 3 | 157,440 | 6,144 | 2,262,097,920 |

结论：

- tail batching 后，atomic/signal command 量约为 WRITE command 的 `3.9%`
  (`30,720 / 786,624`)；tail 已经不是数量上的主导项。
- 真正的压力来自大量 per-token payload WRITE：单次 smoke run 每节点约 `78.7 万`
  个 WRITE command、`11.3 GB` payload 被 CPU proxy post/poll。
- `8 x 4` 和 `8 x 8` sweep 没有明显提升，说明单纯增加 proxy threads/rings 不解决
  这个 per-token WR 形态。下一步更像是协议层优化：把 dispatch 里按 token 的 payload
  WRITE 变成 V2-native per-expert/per-peer semantic batching，减少 WR 数；这也更接近
  原 `uccl/ep` 在 EFA 上取得效果的方向。

## 2026-06-04：决定转向 UCCL-GIN API + 冻结快照 + vendoring

### 根因再定位(对比三处实现)

- `ep/src/internode.cu`(V1 normal)和上游 `legacy/internode.cu` 都是
  **staging buffer + 滑动窗口 + 一次 put 发连续 run**(`num_max_rdma_chunked_send_tokens`)。
- 上游 V2 `hybrid_dispatch.cuh` **不在 kernel 里合并**:每 token 一次
  `gin.put<ncclTeamTagRail>(..., ncclGinOptFlagsAggregateRequests)`,把聚合下放给
  NCCL GIN 层。
- 我们的 `hybrid_dispatch_native.cuh` fork 时把 `gin.put` 换成 per-token
  `v2_d2h_write`,**连 `AggregateRequests` 的聚合也一起丢了,且没有替代** → 786K 条
  per-token 14KB WRITE 砸到 CPU proxy → 卡在 ~30 GB/s 小消息区。

### 设计决定:UCCL-GIN

抽象边界其实就是 DeepEP 已有的 `handle::NCCLGin`。决定做一个同形状的
`handle::UCCLGin`:`Lsa` 转发 NCCL/NVLink,`Rail` 走 UCCL D2H+proxy+EFA,把
EFA 的硬骨头(无 atomic→有序软原子、小消息→coalescing、barrier→host/epoch)在后端
解决一次,DeepEP V2 dispatch/combine/LL 以最小改动跟上。计划见
`ep/docs/uccl_gin_plan.md`(含文件结构、thirdparty 极小 patch 策略、性能语义清单)。

### 冻结当前 fork-based 代码(防丢)

- 发现 submodule 里 4 个改动(buffer.hpp / api.cuh / compiler.hpp / kernel_runtime.hpp,
  167 行)**只在工作区、不在父仓历史**,最脆弱。
- 在 `deepepv2` 上 commit `e9f64b19` 冻结全部工作区改动 + worklog + docs;
  submodule 改动另存为 `thirdparty/DeepEP-v2-d4f41e4.local-changes.patch`(gitlink 仍指
  可 fetch 的 `d4f41e4`)。
- `nccl/`(21M 参考克隆)+ `2512.19849v2.pdf` 加进 `.gitignore`,不进历史。
- push 到 `origin/deepepv2` + tag `deepepv2-snapshot-20260604`(远端持久备份)。
- 新开分支 `uccl-gin`,在其上重写。

### DeepEP V2:submodule → vendored(in-tree 副本)

- 在 `uccl-gin` 上把 `thirdparty/DeepEP-v2-d4f41e4` 从 git submodule 转成 vendored
  普通目录(`git rm --cached` gitlink、删内层 `.git`/`.gitmodules`、`git add` 真实文件),
  `third-party/fmt` 一并 vendored(option A,header-only,128 文件)。232 个文件纳入,
  无 160000 gitlink、无构建产物;`figures/` README 图未纳入。
- 新增 `thirdparty/DeepEP-v2-d4f41e4/VENDORED.md`(上游 commit + 改动清单 + re-vendor 说明)。
- 更新 `AGENTS.md`(submodule → vendored)、`ep/docs/uccl_gin_plan.md`(文件结构 + vendoring)。
- 新增根 `CLAUDE.md`,`@AGENTS.md` 引入项目规则。
- 路径 `thirdparty/DeepEP-v2-d4f41e4` 不变 → Makefile/Python 零改动。

### 下一步(uccl-gin)

按 `uccl_gin_plan.md`:P0 抽 `uccl_gin_rail.cuh`(收纳现有 v2_d2h_* / PackAtomicWithSeq)→
P1 `handle::UCCLGin` 骨架(Lsa 复用 NCCLGin、Rail 调 P0)→ P2 thirdparty 极小 patch
(`DEEPEP_GIN_T` 可替换)接通 dispatch、删 800 行 fork → P3 proxy 侧 coalescing → P4 combine。

## 2026-06-04:UCCL-GIN Rail API 独立微基准(不接 DeepEP,对比 NCCL-GIN)

目标:先把 UCCL-GIN 的几个 Rail op 独立出来测,和直接 NCCL-GIN 对比,确认 API 本身,
再谈接进 DeepEP。决策:**形态 A(纯 C++/CUDA standalone,两条路一个 MPI 程序),
第一版只覆盖 put + red_add_rel**。

写好的文件(本地写,无法本地编译,需上服务器 compile-iterate):

- `ep/include/uccl_gin/uccl_gin_rail.cuh`(L1 backend):device `rail_put`(WRITE,
  window offset >>2)、`rail_red_add`(ordered ATOMIC:`value=delta`、
  `req_rptr=atomic-buf 字节 offset`、`atomic_offset=1` 触发 PackAtomicWithSeq)。
  字段语义对照 `ep/src/rdma.cpp:2907-2936` 的 ATOMIC 编码确认过。
- `ep/include/uccl_gin/resources.cuh`:`UCCLGinResources` POD(d2h_queues/num_queues/
  window_base/atomic_tail_base/topology),一次注入,稳定 ABI。
- `ep/include/uccl_gin/uccl_gin.cuh`:**`handle::UCCLGin` 抽象**——镜像 DeepEP
  `handle::NCCLGin` 的方法签名(`gin.put<Team>` / `gin.red_add_rel<Team>`,team tag 用
  `ncclTeamTagRail/Lsa`)。v1 实现 Rail 的 put+red_add(调 L1 backend),Lsa 与
  put_value/signal/wait/flush 先 `__trap()` 占位(P1 补)。**这才是被测的抽象**;微基准
  通过它调用,和 NCCL 路的 `gin.put(...)` 调用面对称。
- `ep/tests/uccl_gin_microbench/microbench.cu`:MPI rendezvous + paired-remote
  workload;NCCL-GIN 路(`ncclMemAlloc`+window+`ncclDevComm`(GIN)+`ncclGin.put(SignalInc)`,
  仿 `nccl/docs/examples/06_device_api/02_alltoall_gin/main.cu`)与 UCCL-GIN 路
  (`UcclProxy`+D2H ring+`rail_put/rail_red_add`)同程序对比;size sweep 打印 per-rank GB/s。
- `Makefile`(复用 `ep/src/*.o`,链 NCCL/EFA/MPI)、`README.md`(build/run/已知缺口)。

API grounding(读过并对齐):`uccl_proxy.hpp`(UcclProxy ctor/set_peers_meta/
start_dual/get_d2h_channel_device_addrs)、`d2h_queue_device.cuh`(D2HHandle.
atomic_set_and_commit)、`ring_buffer.cuh`(TransferCmd/make_cmd_type/shift)、
`proxy.hpp`(PeerMeta)、NCCL `nccl_device.h` 示例。

**已知缺口(README 里列了,上服务器要验/补)**:
1. NCCL device 链接可能要 device runtime lib(不止 `-lnccl`),GIN 需 aws-ofi-nccl master。
2. UCCL proxy bootstrap 的 PeerMeta OOB 用 MPI_Allgather + enp71s0 IP(替代 Python
   all_gather_object);start_dual 后可能要 settle。
3. **timing 语义**:cudaEvent 只计 kernel enqueue,UCCL put 是异步(proxy drain);
   真 end-to-end BW 要等 receiver 收齐(counter),不能只 streamSync。v1 先 sync+barrier,
   服务器上要改成等 counter 再信 GB/s。
4. correctness 校验在 main 里是 TODO(拷回 recv + counter 对比 rank-tagged 数据)。
5. 两路都单 stream(1 CTA / 1 D2H lane)做 apples-to-apples;多 lane/CTA fan-out +
   coalescing 是后续 sweep。

参考点(AGENTS.md):NCCL-GIN 大包 ~44 GB/s/rank,16KiB ~8.8、32KiB ~12.5(单流);
UCCL per-token 未合并 ~30。微基准把同一组 op 隔离出来,好把差距归因(op 开销 vs
coalescing vs proxy rate)。

## 2026-06-05:uccl-gin branch 服务器编译与 smoke

操作:

- 远端 `/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang` 切到 `uccl-gin`
  branch,HEAD=`5abc0810`。同步本地新增的 `ep/include/uccl_gin/`、
  `ep/tests/uccl_gin_microbench/` 和 `worklog.md`。
- 编译主 `ep`:  
  `make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16`
  通过。只有既有 warning(nanobind 宏重定义、`proxy.cpp` write return、optional
  maybe-uninitialized)。
- 编译 microbench:  
  `make -C ep/tests/uccl_gin_microbench PYTHON=$VIRTUAL_ENV/bin/python CUDA_HOME=/usr/local/cuda-13.0`
  通过。

编译/代码 review 中修掉的问题:

- Makefile 不能假设 `mpicxx` 在 PATH。改为优先用 `/opt/amazon/openmpi5/bin/mpicxx`
  并显式加 `MPI_INC/MPI_LIB`。
- `nvcc` 不接受 OpenMPI wrapper 给出的 `-Wl,...`;改成 `-Xlinker`。
- NCCL wheel 只有 `libnccl.so.2`,没有 `libnccl.so`;改为链接 `-l:libnccl.so.2`。
- `rdma.o` 用到 `efadv_create_qp_ex`,需要链接 `-lefa`。
- NCCL header 必须让 wheel include path 排在 repo include 前面,并加
  `-DNCCL_CHECK_CUDACC=1`,否则会看不到 `ncclGin/ncclDevCommRequirements`。
- 默认 build 开了 `USE_MSCCLPP_FIFO_BACKEND`,D2H 资源不能用
  `get_d2h_channel_device_addrs()+init_from_dev_ptr`;改为
  `get_d2h_channel_handle_addrs()` 并把 device `D2HHandle*` 指针数组传进
  `UCCLGinResources`。
- NCCL-GIN reference kernel 里 `waitSignal(ncclCoopCta())/flush(ncclCoopCta())`
  原先只在 `threadIdx.x==0` 调用,会卡死;改为 CTA 全线程参与 cooperative op。
- UCCL 原 timing 只计 kernel enqueue,会出现 1MiB `481 GB/s` 这种假数;改成
  CPU wall-clock 并等待 receiver-side atomic counter 到 1 后停表。
- `uccl_gin_rail.cuh` 补了 device-side guard:window offset 不能低于 base/必须 4B
  对齐/shift 后要进 32bit;ordered atomic 的 offset 必须 <=8191 且 8B 对齐,delta
  必须进 15bit signed。
- `UCCLGin::lane()` 补 `num_queues>0 && d2h_queues!=nullptr` guard。

运行日志:

- UCCL 2 节点 x 1 GPU bootstrap smoke:
  `/tmp/uccl_gin_microbench_smoke_2node_uccl.log`。能启动并走到 proxy/RDMA,但这个
  rank layout 不符合 UCCL normal-mode 的 `rank±MAX_NUM_GPUS` 假设,不能作为完整
  drain 测试。
- NCCL 2 节点 x 1 GPU smoke:
  `/tmp/uccl_gin_microbench_smoke_2node_nccl.log`。修 cooperative wait 后能完成。
- UCCL 2 节点 x 1 GPU counter-wait 尝试:
  `/tmp/uccl_gin_microbench_sweep_2node_1gpu_counter_wait.log`。暴露
  `Posting rdma to a different rank` abort,根因是 1-rank/node 的 peer rank diff=1,
  而原 transport normal mode 要求跨节点同 local rank diff=`MAX_NUM_GPUS`(8)。
- EP16 UCCL-only counter-wait:
  `/tmp/uccl_gin_microbench_ep16_uccl_counter_wait.log`
  - 4KiB: `0.58 GB/s/rank`
  - 16KiB: `4.00 GB/s/rank`
  - 64KiB: `11.79 GB/s/rank`
- EP16 NCCL-only reference:
  `/tmp/uccl_gin_microbench_ep16_nccl.log`
  - 4KiB: `0.14 GB/s/rank`
  - 16KiB: `0.57 GB/s/rank`
  - 64KiB: `2.31 GB/s/rank`

结论:

- `UCCLGinResources -> UCCLGin -> old TransferCmd -> UcclProxy/EFA` 的 standalone
  Rail put+red_add 路径已经能在 EP16 跑通,并且使用了每 GPU 对应的 EFA NIC。
- 当前 microbench 还没有 payload correctness check;只通过 receiver counter 表明
  proxy drain + receiver atomic 已完成。下一步应补 recv buffer 校验,再做多 CTA/lane、
  coalescing 和 DeepEP V2 kernel call-site 替换。

## 2026-06-05:回主线 P1 —— 忠实 drop-in `handle::UCCLGin` + drop-in 局限性发现

微基准初步验证够了(EP16 隔离 put+red_add,UCCL 比 NCCL-GIN 快 4–7×,方向成立)。回主线做 P1。

- 完整读了 `deep_ep/common/handle.cuh` 的 `handle::NCCLGin` 接口。
- 新增 `ep/include/uccl_gin/uccl_gin_handle.cuh`:`deep_ep::elastic::handle::UCCLGin`,
  **组合一个 NCCLGin**(Lsa/World/barrier/get/signal/wait/flush 全部委托,语义不变),
  **只重写 Rail 分支**;签名与 NCCLGin 对齐,目标是 kernel call site 不改、只换 gin 类型。
- 标准版微基准仍用 lean `uccl_gin.cuh`(Rail-only);drop-in 用这个新 handle。

**关键发现:drop-in 命题比 plan 原来说的弱——只有 `put<Rail>` 是干净的换类型;
`red_add_rel<Rail>`(tail)和 `put_value<Rail>` 不是透明 drop-in:**
- NCCL 的 `red_add_rel<Rail>` 在 receiver GPU 上对 *window* 做 add(`gin.signal(VASignalAdd)`);
  UCCL 的有序原子是 receiver *CPU proxy* 对 *host-mapped atomic buffer* 做 fetch_add
  (CPU 不能 fetch_add 设备 HBM 的 window)。
- 而且 ordered-atomic immediate 的 offset 字段只有 ~13 bit(≤8191),装不下 window tail
  的大 offset;必须用**紧凑 (channel, src-rank) 索引**(就是删掉的 fork 里 `v2_atomic_tail_ptr`)。
  这个索引无法只从 `sym_ptr` 还原。
- 还有 receiver 侧:forward warp 读 tail 的位置也得从 window 改成 atomic_tail_base 的同一
  紧凑槽。
- 结论:`red_add_rel<Rail>` 在 handle 里先 `__trap()` + 文档;另给一个显式
  `rail_tail_add(channel, src_rank, dst_scaleout, delta)` 供 P2 kernel patch 调用。
  `put_value<Rail>` 同样先 trap(需要 local source 暂存,P2 接 notify count 路径时补)。

所以 **P2 不是"只换 gin 类型",而是 = 换 gin 类型 + patch tail 写/读两侧成紧凑索引 op +
comm.cuh barrier 模板化**。put(payload,占绝大多数字节)是干净换;tail/count(signaling)
需要 kernel 配合。这不影响方向(put 是 BW 主体),但要在 plan 里写清。

## 2026-06-05:回主线 P2 device-side dispatch patch(不继续 microbench)

用户明确说 microbench 差不多即可,回到主线。停止继续扩 microbench/correctness,开始接
DeepEP V2 dispatch。

代码推进:

- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/common/comm.cuh`
  - `nvlink_barrier_wo_local_sync / scaleup_barrier_wo_local_sync /
    scaleout_barrier_wo_local_sync / gpu_barrier` 的 gin 参数从硬写
    `const handle::NCCLGin&` 改成模板 `const Gin&`(default 仍是 NCCLGin)。
  - 默认上游调用不需要变;UCCLGin 只要暴露 `nccl_dev_comm` 和 Lsa `get_sym_ptr`,
    scaleup/NVLink barrier 就可继续复用原逻辑。

- `ep/include/uccl_gin/uccl_gin_handle.cuh`
  - `UCCLGin` 增加 `nccl_dev_comm/nccl_window` 引用,兼容 `comm.cuh`。
  - 增加 compact tail helpers:
    - `rail_tail_offset(channel, src_scaleout)`
    - `rail_tail_ptr(channel, src_scaleout)`
    - `decode_rail_tail(raw, finish, count)`
    - `rail_tail_add(channel, src, dst, count_delta, finish)`
  - tail 编码决定:
    - ordinary update: `+ count_delta`
    - final update: `+ 8192 + count_delta`
    - receiver: `finish = raw >= 8192`, `count = raw - finish*8192`
  - 这样 finish/count 走同一个 compact slot,PackAtomicWithSeq 对同一 index 保序,
    避免两 slot 的 finish-vs-count 乱序问题。

- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - 在 `DEEPEP_USE_UCCL_GIN` 宏打开时 include `uccl_gin_handle.cuh`。
  - kernel signature 在宏打开时多一个 `uccl_gin::UCCLGinResources` 参数。
  - 构造 `handle::UCCLGin(nccl_dev_comm, nccl_window, resources, qp_idx, sharing_mode)`。
  - kernel 开始时清零 compact tail slots(`kNumChannels * kNumScaleoutRanks`)。
  - scaleout warp tail write 从 `gin.red_add_rel<Rail>(packed_tail_delta)` 改为
    `gin.rail_tail_add(...)`。
  - forward warp tail read 从 workspace packed tail 改为读 `gin.rail_tail_ptr(...)`
    并 `decode_rail_tail(...)`。
  - 默认宏不开时原 NCCL-GIN 路径完全保持。

- `ep/docs/uccl_gin_plan.md`
  - 修正旧说法:`red_add_rel<Rail>`/`put_value<Rail>` 不是透明 drop-in。
  - 记录 compact tail 编码和 P2 真实 patch 面。

服务器验证:

- 同步到 `p5en_0:/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang`。
- 默认 `ep` 构建通过:
  `make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16`
- 用临时 TU 实例化 `DEEPEP_USE_UCCL_GIN` 的 `hybrid_dispatch_impl`:
  `/tmp/uccl_gin_hybrid_dispatch_compile.cu`
  编译命令核心:
  `nvcc -std=c++20 --expt-relaxed-constexpr -arch=sm_90 -DNCCL_CHECK_CUDACC=1 ... -c`
  结果通过,产物 `/tmp/uccl_gin_hybrid_dispatch_compile.o`。

当前剩余主线缺口:

- host/JIT 侧还没有把真实 `UCCLGinResources` 传进 `launch_dispatch`。
- Python/ElasticBuffer 侧还没有启动 UcclProxy 并构造 device-side
  `UCCLGinResources` 给 DeepEP V2 JIT kernel。
- `put_value<Rail>` 仍未接 notify/count path;hybrid dispatch 当前 notify scaleout
  用的是 `put<Rail>` 两条 count buffer WRITE,所以 dispatch payload/tail 主路径可先继续。

## 2026-06-05:接入 host/JIT UCCL-GIN resources(编译验证通过)

本轮先 commit 了上一轮 device-side 起点:

- commit: `8664de6a Add UCCL-GIN dispatch device path`
- 没有添加 co-author。

继续推进内容:

- `thirdparty/DeepEP-v2-d4f41e4/csrc/kernels/elastic/dispatch.hpp`
  - 新增 host-side ABI mirror `NativeUCCLGinResources`。
  - 这个 struct 和 `ep/include/uccl_gin/resources.cuh::UCCLGinResources`
    保持字段顺序/大小一致,但 DeepEP `_C` 默认构建不需要 include UCCL header。
  - `DispatchRuntime::Args` 增加:
    - `bool use_uccl_gin_resources`
    - `NativeUCCLGinResources uccl_gin_resources`
  - `launch_impl` 在 hybrid dispatch 且 resources 已启用时,多传一个 POD 参数给
    JIT kernel;默认 NCCL-GIN 路径仍走原签名。

- `thirdparty/DeepEP-v2-d4f41e4/csrc/elastic/buffer.hpp`
  - `ElasticBuffer` 增加 `set_uccl_gin_resources(pybind11::dict)` binding。
  - dict 字段:
    - `d2h_queues_ptr`
    - `num_queues`
    - `window_base`
    - `atomic_tail_base`
    - `num_scaleout_ranks`
    - `num_scaleup_ranks`
    - `scaleout_rank`
    - `scaleup_rank`
    - `num_lanes`
  - setter 会校验 topology 必须和 DeepEP/NCCL context 一致。
  - dispatch 时如果 resources 已设置,把它传入 `launch_dispatch`。

- `ep/src/uccl_ep.cc`
  - 新增 Python-visible `UCCLGinResourceHandle`。
  - 它负责:
    - 从一组 `UcclProxy` 收集每个 proxy 的 device-side D2H handle 地址。
    - `cudaMalloc` 一个 device array(`d2hq::D2HHandle**`)并持有其生命周期。
    - 复用 proxy0 的 atomic buffer,并把同一个 atomic buffer 设置给所有 proxy。
    - `as_dict()` 导出 DeepEP `_C.set_uccl_gin_resources()` 需要的稳定 dict。
  - 新增 helper `uccl.ep.build_uccl_gin_resources(...)`。

- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/buffers/elastic.py`
  - 新增 `ElasticBuffer.init_uccl_gin(...)`。
  - 它从 `runtime.get_native_v2_resources()` 获取 DeepEP V2 的 workspace/window
    指针:
    - proxy/MR 注册使用 `rdma_workspace_ptr`
    - kernel offset origin 使用 mapped `workspace_ptr`
    - 注册长度使用 `workspace_bytes + buffer_bytes`(GPU/HBM segment only)
  - 创建/启动 `uccl.ep.Proxy` 组,all_gather peer metadata,构造
    `UCCLGinResourceHandle`,然后调用 `_C.ElasticBuffer.set_uccl_gin_resources()`。
  - 自动补:
    - `DEEPEP_REPO_ROOT=<repo root>`
    - `EP_JIT_EXTRA_FLAGS+=-DDEEPEP_USE_UCCL_GIN`
  - 注意:如果进程已经触发过 DeepEP JIT 编译,这些 env 可能太晚;真正跑测时建议清
    `EP_JIT_CACHE_DIR`/`~/.deep_ep` 并在第一次 dispatch 前调用 `init_uccl_gin()`。

- `uccl/__init__.py`
  - 加 `pkgutil.extend_path`。
  - 原因:从仓库根目录运行测试时,源码树里的 `uccl/` 会遮蔽 site-packages 里
    已安装的 `uccl.ep.abi3.so`;扩展 `__path__` 后 `import uccl.ep` 能继续找到
    已安装 extension。

远端验证:

- 同步到 `p5en_0:/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang`。
- `uccl.ep` 编译通过:
  `make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16`
- DeepEP `_C` 编译通过:
  `cd thirdparty/DeepEP-v2-d4f41e4 && python setup.py build_ext --inplace`
  - 第一次在最后链接失败:`/usr/bin/ld: cannot find -l:libnccl.so.2`。
  - 根因是 vendored DeepEP `setup.py` 把 NCCL rpath 写进 `extra_link_args`,
    但没有把 NCCL lib dir 放入 `library_dirs`;补
    `LIBRARY_PATH=/home/ubuntu/.venvs/deepep-danyang-cu13/lib/python3.12/site-packages/nvidia/nccl/lib`
    后增量 build/link 通过。
- Python API 可见性检查通过(从 `/tmp` 运行,避免仓库根 `uccl/` 源包遮蔽已安装
  `uccl.ep` extension):
  - `uccl.ep.build_uccl_gin_resources == True`
  - `uccl.ep.UCCLGinResourceHandle == True`
  - `deep_ep.ElasticBuffer.init_uccl_gin == True`
  - `deep_ep._C.ElasticBuffer.set_uccl_gin_resources == True`
- `uccl/__init__.py` namespace path 修复后,从仓库根目录也能
  `import uccl.ep`,解析到
  `/home/ubuntu/.venvs/deepep-danyang-cu13/lib/python3.12/site-packages/uccl/ep.abi3.so`。
- 宏打开的 device kernel 实例化编译通过:
  `/tmp/uccl_gin_hybrid_dispatch_compile.cu`
  `nvcc -std=c++20 --expt-relaxed-constexpr -arch=sm_90 -DNCCL_CHECK_CUDACC=1 ... -c`
  产物 `/tmp/uccl_gin_hybrid_dispatch_compile.o`。

当前状态:

- host/JIT resources 注入链路的第一版已经写完并编译通过。
- 还没有跑真实 EP8x2 dispatch correctness/benchmark。
- 下一步应该跑最小双机 dispatch correctness;若卡住,优先看:
  - `init_uccl_gin()` 是否必须在任何 DeepEP JIT 编译前调用。
  - `put_value<Rail>`/notify count 路径是否仍有 NCCL-GIN Rail call 没替换。
  - compact tail buffer 目前受 `kAtomicOffMask` 限制,适合 EP8x2;更大 scaleout
    需要扩大/分片 atomic tail layout。

## 2026-06-05:UCCL-GIN DeepEP V2 EP8x2 first-case correctness 跑通

本轮先在本地 commit 了 host/JIT resources 注入链路:

- commit: `8b25aa42 Wire UCCL-GIN resources into DeepEP dispatch`
- 没有添加 co-author。

随后把 `test_ep.py` 的 UCCL-GIN 初始化接进 first-case 测试:

- `construct_elastic_buffer()` 之后,如果 `DEEPEP_USE_UCCL_GIN=1`,调用
  `buffer.init_uccl_gin()`。
- 这样 DeepEP V2 的真实 `ElasticBuffer.dispatch/combine` 会在第一次 JIT dispatch
  前收到 UCCL-GIN resources,而不是只测 standalone microbench。

服务器验证过程和修复:

1. 第一次 EP8x2 smoke 进入 dispatch 后 abort:

   - 日志:`/tmp/uccl_gin_deepep_ep8x2_smoke_rank*.log`
   - 关键错误:`Posting atomic to itself`
   - 根因:原 NCCL-GIN Rail team 的本 scaleout rank 路径不走 EFA proxy;UCCL-GIN
     `rail_tail_add()` 直接把 `dst_scaleout == self` 也编码成 proxy ATOMIC,触发
     UCCL proxy 的 self-command abort。
   - 修复:`UCCLGin::rail_tail_add()` 对本 scaleout rank 直接
     `atomicAdd_system()` 到 compact tail slot。

2. 下一轮进入 `dispatch_copy_epilogue` 后 device assert:

   - 错误:
     `dispatch_copy_epilogue.cuh:106, condition: ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1`
   - 第一处根因:kernel 读 compact tail 使用的是 host VA。`cudaHostAllocMapped`
     的 host pointer 不能直接当 device pointer 传给 kernel。
   - 修复:`UCCLGinResourceHandle` 在 host 侧对 proxy0 的 atomic buffer 调
     `cudaHostGetDevicePointer()`,把 device-mapped VA 作为 `atomic_tail_base`
     传入 `UCCLGinResources`。

3. 修完 mapped VA 后仍然有 epilogue metadata assert:

   - 根因:payload `put<Rail>` 默认 lane 0,tail `rail_tail_add()` 用
     `channel_idx % num_queues`。同一 channel 的 payload 和 tail 落在不同 D2H
     queue/proxy 上,tail 可能先于 payload 到达,forward/epilogue 读取到坏 metadata。
   - 修复:在 `DEEPEP_USE_UCCL_GIN` 下,`hybrid_dispatch.cuh` 的 scaleout payload
     `gin.put<Rail>()` 额外传 `channel_idx` lane hint,保证 payload 和该 channel
     的 tail 进入同一 D2H queue。

4. 再跑后失败点变成 host CPU wait:

   - 错误:
     `Dispatch CPU wait exception ... CPU side received count ... 0 0 ...`
   - rank1 日志出现大量:
     `DeepEP hybrid notify (scale-out expert/rank reduction) timeout ... wait scale-out: 0, decoded: -1`
   - 说明 payload/tail 已能前进,但 notify 阶段的 scaleout count WRITE 没有被对端
     notify warp 看到。
   - 根因:UCCL-GIN 的 Rail `put` 是“GPU 写 send buffer -> GPU 提交 D2H command ->
     CPU proxy/NIC 读 send buffer”。原 NCCL-GIN 内部提供了 device put 前的发布顺序,
     UCCL-GIN 没有;proxy/NIC 可能读到 count send buffer 里的旧 0。
   - 修复:
     - `UCCLGin::put<Rail>()` 在 remote D2H command commit 前做
       `__threadfence_system()`。
     - notify 阶段 SM0 写完 encoded rank/expert count send buffer 后,在
       `named_barrier` 前加 `__threadfence_system()`;这样所有参与写 count 的
       notify 线程都把全局写发布到 system scope。

5. 服务器端编译检查:

   - `p5en_0` GPU 空闲:8 张卡均 `0 MiB/0%`。
   - `p5en_1` GPU 空闲:8 张卡均 `0 MiB/0%`。
   - 同步文件到两台机器的
     `/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/`。
   - 在 `p5en_0` 编译临时 TU:
     `/tmp/uccl_gin_hybrid_dispatch_compile.cu`
   - 命令核心:
     `nvcc -std=c++20 --expt-relaxed-constexpr -arch=sm_90 -DNCCL_CHECK_CUDACC=1 -DDEEPEP_USE_UCCL_GIN ... -c`
   - 结果:通过。

6. EP8x2 smoke correctness:

   - 命令核心:
     `DEEPEP_USE_UCCL_GIN=1 EP_JIT_EXTRA_FLAGS=-DDEEPEP_USE_UCCL_GIN`
     `python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py`
     `--num-processes 8 --test-first-only --skip-perf-test --num-sms 8`
     `--num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256`
   - 多机环境:
     - `WORLD_SIZE=2`
     - `RANK=0/1`
     - `MASTER_ADDR=172.31.78.36`
     - `OFI_NCCL_FORCE_NUM_RAILS=4`
     - `FI_PROVIDER=efa`
     - `NCCL_NET_PLUGIN=ofi`
   - JIT cache:
     - rank0:`/tmp/deepep_uccl_gin_jit_rank0`
     - rank1:`/tmp/deepep_uccl_gin_jit_rank1`
   - 日志:
     - rank0:`/tmp/uccl_gin_deepep_ep8x2_smoke_rank0.log`
     - rank1:`/tmp/uccl_gin_deepep_ep8x2_smoke_rank1.log`
   - 结果:
     - rank0 exit code `0`
     - rank1 exit code `0`

当前结论:

- UCCL-GIN 已经作为 DeepEP V2 `hybrid_dispatch.cuh` 的 Rail backend 跑通真实
  `ElasticBuffer.dispatch/combine` first-case EP8x2 correctness。
- 当前通过的是小规模 correctness smoke,不是 README 风格性能数据。
- 下一步回到主线应做:
  - 用更接近 README 的 `hidden=7168,num_tokens=8192,num_sms=20` 做 dispatch
    correctness/perf。
  - 分离 dispatch-only 和 epilogue/host wait 时间,确认 UCCL-GIN Rail backend
    真实瓶颈。
  - 清理临时 debug/env-gated test hook 是否应变成正式入口或单独测试脚本。

## 2026-06-05:补 UCCL-GIN Rail release ordering,README-like EP8x2 跑通

目标:

- 在已通过小配置 EP8x2 smoke 后,跑更接近 README 的 first-case:
  - `--num-processes 8`
  - `--test-first-only`
  - `--num-sms 20`
  - `--num-tokens 8192`
  - `--hidden 7168`
  - `--num-topk 8`
  - `--num-experts 256`

第一次大配置结果:

- 日志:
  - rank0:`/tmp/uccl_gin_deepep_ep8x2_readme_rank0.log`
  - rank1:`/tmp/uccl_gin_deepep_ep8x2_readme_rank1.log`
- 失败:
  `dispatch_copy_epilogue.cuh:106, condition: ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1`
- 这个错误和之前 payload/tail 不同 queue 的症状一样,但当时 payload 和 tail 已经固定
  到同一 `channel_idx` queue。

代码检查结论:

- `post_gpu_commands_mixed()` 会按 command 类型分桶:
  - 先收集所有 `WRITE`
  - 再收集所有 `ATOMIC`
  - post 时先 post RDMA WRITE batch,再 post ATOMIC batch
- 但是没有等待 WRITE CQE 就 post ATOMIC。
- UCCL-GIN 的 tail 用 `rail_red_add()` 表达 NCCL-GIN 的 `red_add_rel`。
  `rel` 语义要求 tail 可见时,前面的 payload WRITE 已经对 receiver 可见。
- EFA/SRD 下“先 post WRITE,后 post ATOMIC”不等于“receiver 先看到 payload,再看到
  tail”;大配置中 forward warp 会按 tail 读取到还没完成的 payload,最终 epilogue
  看到坏的 topk metadata。

修复:

- 在 `ep/src/proxy.cpp::post_gpu_commands_mixed()` 中,如果同一批 command 里存在
  ATOMIC tail:
  - RDMA WRITE batch post 后,把本批 `rdma_wrs` 放入 `pending_release_wrs`。
  - 持续 `poll_cq_dual()`。
  - 每轮从 `acked_wrs_` 中移除已完成的本批 WRITE。
  - 调 `notify_gpu_completion()` 正常推进 GPU ring tail。
  - 等本批 WRITE 全部 CQE 到达后,再 post ATOMIC batch。
- 这个是 UCCL-GIN Rail `red_add_rel` 的正式 release 语义,不是调试 fallback。
- 代价:当前实现是 per proxy batch 的 coarse release fence,性能会受影响;后续可以做
  per-ring/per-destination 的更细 fence 或 coalesced tail,但 correctness 语义必须保留。

构建注意:

- 两台机器并行 `make -C ep install` 会在共享 EFS 上同时链接同一个 `ep.abi3.so`,
  触发:
  `final link failed: Stale file handle`
- 解决:顺序构建/安装。
  - `p5en_0`:先 `rm -f ep/ep.abi3.so`,再 `make -C ep install ...` 通过。
  - `p5en_1`:随后 `make -C ep install ...` 通过。

第二次大配置结果:

- GPU 空闲检查:
  - `p5en_0`:8 张 GPU 均 `0 MiB/0%`
  - `p5en_1`:8 张 GPU 均 `0 MiB/0%`
- 命令核心:
  `DEEPEP_USE_UCCL_GIN=1 EP_JIT_EXTRA_FLAGS=-DDEEPEP_USE_UCCL_GIN`
  `python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py`
  `--num-processes 8 --test-first-only --num-sms 20 --num-tokens 8192`
  `--hidden 7168 --num-topk 8 --num-experts 256`
- 结果:
  - rank0 exit code `0`
  - rank1 exit code `0`
- 日志:
  - rank0:`/tmp/uccl_gin_deepep_ep8x2_readme_rank0.log`
  - rank1:`/tmp/uccl_gin_deepep_ep8x2_readme_rank1.log`

README-like 性能摘录:

- dispatch:
  - rank0 local ranks:约 `22 GB/s (SO)`, `70-73 GB/s (SU)`,
    `~5.5-5.6 ms`,每 rank 约 `396-402 MB`
  - rank1 local ranks:多数约 `14 GB/s (SO)`, `45-46 GB/s (SU)`,
    `~8.8 ms`;local rank 4/EP12 约 `22 GB/s (SO)`
- expanded dispatch:
  - rank0:约 `22 GB/s (SO)`, `71-73 GB/s (SU)`, `~5.5 ms`
  - rank1:多数约 `14 GB/s (SO)`,EP12 约 `22 GB/s (SO)`
- cached dispatch:
  - rank0:约 `22 GB/s (SO)`, `71-73 GB/s (SU)`
  - rank1:多数约 `14 GB/s (SO)`,EP12 约 `22 GB/s (SO)`
- combine/reduced combine:
  - rank0:约 `13-22 GB/s (SO)`
  - rank1:约 `11-17 GB/s (SO)`

当前结论:

- UCCL-GIN Rail backend 已能通过真实 DeepEP V2 EP8x2 README-like first-case
  correctness 和 perf 输出。
- 当前 dispatch 已明显高于之前 aws-ofi-nccl proxy GIN 的 DeepEP V2 `~5 GB/s`
  级别,但仍未接近 README CX7/IB 的 `~90 GB/s`。
- 下一步优化应聚焦:
  - release fence 粒度太粗,导致 tail 发布等待整批 WRITE CQE。
  - 仅 4 proxy threads/queues 时,80 channels 会在 4 条 CPU proxy/CQ 路径上排队。
  - rank1 大多数 local rank 只有 `14 GB/s`,存在明显 per-rank/NIC/proxy 不均衡。
  - 需要加 dispatch-only 分段计时和 proxy profile,区分 transport、notify、forward、
    epilogue 和 release-fence wait 的耗时。

## 2026-06-05:打开 UCCL proxy command profile

目的:

- 复用已有 `UCCL_PROXY_PROFILE_COMMANDS=1`,先不加新 profiling 代码。
- 同配置重跑 README-like EP8x2,观察 proxy thread 的 WR/atomic 分布。

命令差异:

- 在上一节 README-like EP8x2 环境上额外设置:
  `UCCL_PROXY_PROFILE_COMMANDS=1`
- 日志:
  - rank0:`/tmp/uccl_gin_deepep_ep8x2_profile_rank0.log`
  - rank1:`/tmp/uccl_gin_deepep_ep8x2_profile_rank1.log`
- 结果:
  - rank0 exit code `0`
  - rank1 exit code `0`

性能稳定性:

- rank0 dispatch 仍约 `22 GB/s (SO)`,`~5.5 ms`。
- rank1 除 EP12 外,多数 dispatch 仍约 `14 GB/s (SO)`,`~8.8 ms`。
- 打开 command profile 没有改变 correctness。

proxy profile 观察:

- 每个 global rank 有 4 个 proxy thread,每个 thread 显示 `rings=8`。
- thread 0/1 明显比 thread 2/3 更忙:
  - thread 0/1 典型:
    - `write_cmds ~= 457k-458k`
    - `atomic_cmds ~= 155k-156k`
    - `write_bytes ~= 3.44 GB`(整个测试进程累计)
  - thread 2/3 典型:
    - `write_cmds ~= 303k-306k`
    - `atomic_cmds ~= 103k-104k`
    - `write_bytes ~= 2.28-2.30 GB`
- 这不是单个 rank 的偶然现象,rank0 节点和 rank1 节点都类似。

结论:

- 当前 channel/proxy queue 映射存在约 `1.5x` 的 proxy thread 负载不均。
- 由于 `UCCL_PROXY_PROFILE` 的 `seconds` 覆盖整个测试进程生命周期,其中包含 Python
  correctness/perf 多阶段时间,`write_GBps` 绝对值不适合作为 transport 带宽;更有用的是
  command 数量和 per-thread 分布。
- 下一步代码优化建议:
  - 检查 `init_uccl_gin()` 构造的 `num_queues` 与每个 proxy 的 `rings=8` 对应关系。
  - 检查 `lane_hint=channel_idx` 经过 `num_queues` 取模后是否均匀落到 4 个 proxy
    thread;如果 device array 是 proxy-major/ring-major 排列,需要确认映射没有把 80
    channels 偏到前两个 proxy。
  - notify 的 scaleout count 仍默认 lane 0,会额外压 thread 0;可以把 notify count
    按 `dst_scaleout_rank_idx` 或 count index 显式分散到 queues,但必须保持 count
    WRITE 和对应 wait 语义正确。
  - release fence 应从“本批有 ATOMIC 就等所有 WRITE”改成 per-ring/per-tail-word
    fence,避免一个 busy ring 拖住同 proxy batch 里的其他 ring。

## 2026-06-05:在 UCCL-GIN 层补回 payload-before-tail 保证(对照 NCCL-GIN 源码)

### NCCL-GIN 怎么做到的(查 nccl 源码确认)

原版 DeepEP forward warp 直接信 tail、不做 readiness,是因为 NCCL-GIN **保证 signal
(tail)在 payload 之后生效**。机制不是“一个 GIN op 原子完成”,而是 **对 signals MR
强制 strong ordering(SO)**:
- `nccl/src/gin/gin_host_proxy.cc:475`:“Enforcing strong ordering on the signals mr
  is vital to ensure ordering between puts and signals.”,signals MR 用
  `NCCL_NET_MR_FLAG_FORCE_SO` 注册。
- `nccl/src/gin/gin_host.cc:374`:`NCCL_WIN_STRICT_ORDERING → NCCL_NET_MR_FLAG_FORCE_SO`。
- 即:payload 走 relaxed MR 拿带宽,signal 走 SO MR;NIC/libfabric 保证对 SO MR 的写
  排在之前 payload 写之后。无 CPU 等待、无序列化。
（更正:之前把 profiling 的 “RDMA/SO” 当 strong-ordering 是错的——那是 scale-out。
故无证据 EFA 经 raw ibverbs 支持 SO MR。）

### 为什么 UCCL-GIN 不能照搬 SO MR

- UCCL 走 raw ibverbs/efadv,EFA MR 用 `IBV_ACCESS_RELAXED_ORDERING` 注册
  (`ep/src/rdma.cpp:616/782`),没有 NCCL/libfabric 的 `FORCE_SO` 通道。
- 关键:UCCL 的 tail 不是“NIC 写 MR”,而是 **receiver CPU proxy 收 write-with-imm 后
  fetch_add**(EFA 无硬件 remote atomic)。没有可被 strong-order 的目标 MR。

### UCCL-GIN 等价保证:proxy completion fence

UCCL 有 NCCL 没有的 CPU proxy completion 信息。用它定序:**先 post payload WRITE,等
其 CQE(EFA 可靠,CQE == 已送达 receiver 内存),再 post tail ATOMIC**。tail 的 imm
在 payload 落地后才到 → CPU proxy apply count 时 payload 已在 recv buffer → forward
warp 信 tail 即可,与原版一致。这正是之前删掉的 per-batch fence 思路;当时换成
receiver readiness 反而引入 epoch bug(同 token 每轮同 `src_token_global_idx`、slot 不清、
无轮号 → 多轮读陈旧)。现改回 fence + **删 readiness**。

### 本次改动

- `ep/src/proxy.cpp`:
  - `post_gpu_commands_mixed` 末尾:`flush_writes(); 若有 atomic 则 quiet_cq({}) 等
    payload WRITE completion; 再 flush_atomics()`,保证 tail 在 payload 之后 post。
  - `quiet_cq`:completion 后从 `inflight_write_wrs_` erase(原来只清本地 pending、不清
    `inflight_write_wrs_` → 无界增长 + 跨批可能死等已被清出 `acked_wrs_` 的旧 WR)。
    顺带修的 lifecycle bug。
- `thirdparty/.../impls/hybrid_dispatch.cuh`:删 `#ifdef DEEPEP_USE_UCCL_GIN` 的
  readiness spin 段 + `last_forward_src_token_global_idx`;forward warp 回到原版“信 tail”。

### 取舍 / 待办

- fence 当前是 per-proxy-thread(等该线程全部 inflight write),非 per-ring,会让 busy
  ring 拖住同线程其他 ring;correctness 优先,后续改 per-ring。
- tail 频率:每 `kScaleoutUpdateInterval` token 一次,应从上游默认 3 调到 ~32(fork
  sweep 最优),否则 fence 触发过频——首要 perf 旋钮。
- “NCCL 在 GIN handle 的就在 UCCL-GIN handle”:NCCL `ncclGinOptFlagsAggregateRequests`
  ↔ UCCL proxy 侧 coalescing(合并连续 WRITE),是 P3,不在本次。
- ⚠️ 本次本地盲改,需上服务器 compile + 多轮 + payload correctness 验证(确认删 readiness
  后多轮不读陈旧、fence 不死锁)。

## 2026-06-05:改成 per-ring 小 batch payload-before-tail fence

上一次实现的问题:

- fence 只放在 `post_gpu_commands_mixed()` 末尾,如果一个 proxy batch 中出现
  `WRITE -> ATOMIC -> WRITE`,中间的 ATOMIC 会被后面的 WRITE 触发提前
  `flush_atomics()`,没有等前面的 payload WRITE CQE。
- `quiet_cq({})` 会把整个 proxy thread 的 `inflight_write_wrs_` 全等完,这比原
  UCCL/EP 的 channel/ring 语义更粗,会让无关 channel 被最慢 WR 拖住。
- 只记录当前函数里的 WRITE 也不够:payload WRITE 可能在上一次 proxy poll 已经 post,
  对应 tail ATOMIC 在下一次 poll 才出现。

本次改动:

- `ep/src/proxy.cpp`
  - 新增 `pending_signal_write_wrs_[ring]`:每个 D2H ring 记录“上一次 tail 之后、已经
    post 但还没有被对应 tail signal 消费”的 payload WRITE WR id。
  - WRITE batch post 后,WR id 进入对应 ring 的 pending 列表和 `inflight_write_wrs_`。
  - ATOMIC/tail 到来时,先 `flush_writes()`,再消费该 ATOMIC 所属 ring 的 pending
    payload WR,形成一个小 release batch。
  - `flush_atomics_ordered()` 只等待这个小 release batch 的 CQE,再 post ATOMIC batch。
    它不会全局 drain 同 proxy thread 上其他 ring 的 WRITE。
  - `wait_for_write_cq()` 会用 `inflight_write_wrs_` 过滤 release WR:已经被普通
    completion/notify 退役的 WR 视为已满足,避免跨 poll 场景死等旧 ack。
  - `notify_gpu_completion()` 退役 WR 时,同时从 per-ring pending 列表删除,保持列表有界。
- `thirdparty/DeepEP-v2-d4f41e4/.../hybrid_dispatch.cuh`
  - `DEEPEP_USE_UCCL_GIN` 下把默认 `kScaleoutUpdateInterval` 从 3 调到 32,让 tail
    signal 变成小 batch,类似 NCCL-GIN aggregate/signal batching 和原 UCCL/EP
    的 batching 思路。
  - forwarder 注释改为依赖 Rail transport 的 payload-before-tail ordering,不再保留
    V2-only metadata readiness spin。

设计状态:

- correctness 语义现在是:同一 D2H ring/channel 上,tail ATOMIC 发布前只等待该 ring
  自上次 tail 以来的 payload WRITE 完成;这比全 proxy quiet 更接近原 UCCL/EP 的
  channel-level substrate。
- performance 语义现在是:device 侧每约 32 个 token 发一次 tail,proxy 侧每个 tail
  batch 做一次 exact release fence。下一步需要上服务器重新 build/JIT,验证 EP8x2
  correctness 和 README-like dispatch bandwidth。

## 2026-06-05:服务器验证 exact fence,退回 coarse proxy quiet

用户更新约束:

- 最近服务器一般没人用,后续默认不必每次执行 GPU 空闲检查;只有异常/长实验/不确定时再查。
- 已同步更新 `AGENTS.md`。

构建:

- 同步本轮改动到 `p5en_0:/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang`。
- `make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16`
  通过。
- Python import 检查通过:
  - `uccl.ep` 来自 venv site-packages 的 `ep.abi3.so`
  - `deep_ep` 来自 vendored `thirdparty/DeepEP-v2-d4f41e4`
  - `deep_ep.ElasticBuffer.init_uccl_gin == True`

实验 1: exact per-ring fence + `kScaleoutUpdateInterval=32`

- 命令:EP8x2 smoke
  `--num-processes 8 --test-first-only --skip-perf-test --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256`
- 日志:
  - rank0:`/tmp/uccl_gin_exact_smoke_rank0.log`
  - rank1:`/tmp/uccl_gin_exact_smoke_rank1.log`
- 结果:失败。
- 现象:
  - 多个 rank 触发 `dispatch_copy_epilogue.cuh:106`
    `ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1`
  - expanded dispatch 最终报 CPU wait count 全 0。
- 结论:硬把 UCCL-GIN 默认 tail interval 从 3 改成 32 不安全;小 batch 不能直接改
  template 默认值,需要有更完整的 coalescing/receiver 语义验证。

实验 2: exact per-ring fence + 上游默认 `kScaleoutUpdateInterval=3`

- 同步退回 interval=3 后重跑同样 EP8x2 smoke。
- 日志:
  - rank0:`/tmp/uccl_gin_exact_smoke_i3_rank0.log`
  - rank1:`/tmp/uccl_gin_exact_smoke_i3_rank1.log`
- 结果:rank0/rank1 exit code 均为 0。
- 结论:exact per-ring fence 在小配置 correctness 上可行。

实验 3: exact per-ring fence + interval=3 的 README-like EP8x2

- 命令:
  `--num-processes 8 --test-first-only --num-sms 20 --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
- 日志:
  - rank0:`/tmp/uccl_gin_exact_readme_i3_rank0.log`
  - rank1:`/tmp/uccl_gin_exact_readme_i3_rank1.log`
- 结果:失败。
- 现象:大配置很快触发大量 `dispatch_copy_epilogue.cuh:106` metadata assert。
- 结论:per-ring exact release fence 的依赖范围太窄,大配置下会漏掉跨 queue/ring 的
  payload-before-tail 依赖。小配置通过不代表语义正确。

代码状态修正:

- `ep/src/proxy.cpp` 已退回 conservative coarse proxy quiet:
  - 遇到 ATOMIC/tail 前先 post pending WRITE。
  - `quiet_cq({})` drain 当前 proxy thread 的全部 inflight WRITE。
  - 再 post ATOMIC batch。
- 删除了 per-ring pending WR 状态和 unused `wait_for_write_cq`。
- `hybrid_dispatch.cuh` 保持上游默认 `kScaleoutUpdateInterval=3`,只保留删除
  receiver metadata readiness spin 和注释更新。

下一步:

- 重新同步 coarse quiet 版本并跑 README-like EP8x2,应恢复上一版 correctness。
- 小 batch/性能优化需要另开一个安全设计:不能只靠 ring id 推断 dependency,更像
  UCCL/EP 原 internode 的 batching,需要显式 batch 边界或 proxy coalescing。

实验 4: coarse proxy quiet + 上游默认 `kScaleoutUpdateInterval=3` 的 README-like EP8x2

- 已重新同步 coarse quiet 版本到服务器并 rebuild `ep`。
- 命令:
  `--num-processes 8 --test-first-only --num-sms 20 --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
- 日志:
  - rank0:`/tmp/uccl_gin_coarse_readme_rank0.log`
  - rank1:`/tmp/uccl_gin_coarse_readme_rank1.log`
- 结果:rank0/rank1 exit code 均为 0。
- rank0/node0 local rank 0-7:
  - expanded dispatch 约 `36-37 GB/s (RDMA/SO)`,约 `118-120 GB/s (NVLink/SU)`,
    latency 约 `3348-3362 us`。
  - cached dispatch 约 `36-37 GB/s (RDMA/SO)`,约 `118-120 GB/s (NVLink/SU)`,
    latency 约 `3340-3364 us`。
  - combine RDMA/SO 约 `13-22 GB/s`。
- rank1/node1 local rank 8-15:
  - expanded dispatch 约 `6 GB/s (RDMA/SO)`,约 `19-20 GB/s (NVLink/SU)`,
    latency 约 `20642-20653 us`。
  - cached dispatch 约 `6 GB/s (RDMA/SO)`,约 `19-20 GB/s (NVLink/SU)`,
    latency 约 `20457-20474 us`。
  - combine RDMA/SO 约 `13-22 GB/s`。

结论:

- coarse proxy quiet 恢复了 README-like EP8x2 correctness,说明当前主路径仍应保持
  substrate-level ordering,不能采用刚验证失败的 per-ring exact fence。
- 但是 coarse quiet 版本出现明显 rank 不对称:node0 dispatch 已到 `36-37 GB/s`,
  node1 dispatch 只有 `6 GB/s`。这比单纯“proxy quiet 串行化”更像是单侧 proxy/CQ/NIC
  lane 选择、CPU 线程调度、remote direction 或 JIT/cache 状态导致的失衡。
- 下一轮性能排查应先复现并拆分 node0/node1:
  - 开启 proxy command/profile 计数,比较两侧 WRITE/ATOMIC 数量、CQ wait 时间、quiet 次数。
  - 检查两侧 selected NIC/lane 分布是否一致。
  - 做一次 warm-cache 重跑确认是否稳定复现。
  - 再设计真正的小 batch/coalescing,不能用硬改 `kScaleoutUpdateInterval=32`。

## 2026-06-05:异步 per-tail dependency + receiver metadata readiness

review 结论采纳:

- `flush_atomics_ordered()` 里每个 tail batch 前 `quiet_cq({})` 会把整个 proxy
  thread 的 inflight WRITE 全部同步 drain。
- 由于 DeepEP V2 dispatch 默认约每 `kScaleoutUpdateInterval=3` 个 token 发一次
  scaleout tail,这会把网络上在飞 payload 限制在约 3 个 WR,严重破坏 EFA pipeline。
- 这不是最终性能方案,应该改成异步 per-tail dependency:tail batch 记录自己依赖的
  payload WR id,proxy 继续 post 后续 WRITE;在普通 CQ polling 中看到依赖 WR 完成后,
  再 post 对应 ATOMIC/tail。

代码改动:

- `ep/src/proxy.cpp`
  - 新增 `PendingAtomicBatch` 队列。
  - WRITE post 后把 WR id 加入 `atomic_dependency_wrs_`。
  - 遇到 ATOMIC/tail 时不再调用 `quiet_cq({})`,而是 `enqueue_pending_atomics(...)`。
  - `notify_gpu_completion()` 通过 `retire_inflight_write()` 递减 pending batch 的
    `pending_writes`。
  - `progress_pending_atomics()` 只 post 队首且依赖已满足的 ATOMIC batch,保持 tail
    batch FIFO 顺序。
  - `QUIET/BARRIER` 保守调用 `drain_pending_atomics()`,不影响 dispatch hot path。
- `hybrid_dispatch.cuh`
  - 恢复 UCCL-GIN receiver-side metadata readiness check。
  - tail 只表示 slot range 已发布;每个 slot 还要用 V2 `src_token_global_idx`
    证明 payload metadata 已经对 receiver GPU 可见。

中间失败:

- 只做异步 per-tail dependency,不加 receiver metadata readiness,EP8x2 smoke 会触发
  `dispatch_copy_epilogue.cuh:106` metadata assert。
- 日志:
  - rank0:`/tmp/uccl_gin_async_smoke_rank0.log`
  - rank1:`/tmp/uccl_gin_async_smoke_rank1.log`
- 结论:sender CQE + tail 后发并不足以当作 receiver GPU payload-ready 证明;这和
  UCCL/EP 原来依赖 per-token epoch/tag readiness 的思路一致。

验证 1:async tail + receiver metadata readiness 的 EP8x2 smoke

- 命令:
  `--num-processes 8 --test-first-only --skip-perf-test --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256`
- 日志:
  - rank0:`/tmp/uccl_gin_async_meta_smoke_rc_rank0.log`
  - rank1:`/tmp/uccl_gin_async_meta_smoke_rc_rank1.log`
- 结果:rank0/rank1 rc 均为 0。
- 注:一次重跑遇到 `MASTER_PORT=29653` 的 `EADDRINUSE`,换到 `29711` 后通过。

验证 2:async tail + receiver metadata readiness 的 README-like EP8x2

- 命令:
  `--num-processes 8 --test-first-only --num-sms 20 --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
- 日志:
  - rank0:`/tmp/uccl_gin_async_meta_readme_rank0.log`
  - rank1:`/tmp/uccl_gin_async_meta_readme_rank1.log`
- 结果:rank0/rank1 rc 均为 0。
- rank0/node0 local rank 0-7:
  - dispatch/expanded dispatch 约 `36-37 GB/s (RDMA/SO)`,latency 约
    `3332-3359 us`。
  - cached dispatch 约 `36 GB/s (RDMA/SO)`,latency 约 `3373-3393 us`。
  - combine/reduced combine 约 `16-22 GB/s (RDMA/SO)`。
- rank1/node1 local rank 8-15:
  - dispatch/expanded dispatch 约 `8 GB/s (RDMA/SO)`,latency 约
    `16223-16267 us`。
  - cached dispatch 约 `7 GB/s (RDMA/SO)`,latency 约 `16324-16339 us`。
  - combine/reduced combine 约 `13-17 GB/s (RDMA/SO)`。

结论:

- 异步 tail dependency + receiver readiness correctness 可行,并删除了 hot path
  中 per-tail synchronous global CQ drain。
- 但是 dispatch BW 还没有达到预期:node0 仍约 `36-37 GB/s`,node1 从 coarse quiet
  的约 `6 GB/s` 只提高到 `7-8 GB/s`。
- 因此“每 3 token 全局 drain”确实是坏设计,但不是当前唯一主瓶颈。下一步应重点排查:
  - node0/node1 方向不对称:proxy/CQ/NIC lane/CPU pinning/路由是否不同。
  - receiver metadata readiness spin 是否成为 rank1 的等待主因。
  - `AggregateRequests` 仍被忽略,每 token 一条约 14KB WRITE,还没有真正 small-batch/coalescing。
  - tail frequency 仍是 interval=3;硬改 32 已证伪,需要按 UCCL/EP 的语义 batching
    方式做,不能只改常量。

## 2026-06-05:本地实现 proxy-side tail atomic coalescing,未上服务器

用户要求:

- 先改代码,不要上 server。
- 优先处理 tail 数量过多的问题:不改 device 端 `kScaleoutUpdateInterval`,而是在
  proxy 侧把同一目标的 tail add 合并,减少 receiver `WRITE_WITH_IMM` CQE/apply 次数。
- 下一次上机需要 profile 两侧不对称:原始 atomic 命令数、实际 post 的 atomic WR 数、
  coalesce 数量,以及 proxy poll/progress/post 时间占比。

代码改动:

- `ep/src/proxy.cpp`
  - `PendingAtomicBatch` 现在在 post 前做 conservative coalescing。
  - 只合并同一个 pending batch 中相邻且同目标的 ordered atomic:
    `D2H ring/channel + dst_rank + cmd_type + req_rptr + atomic_offset` 必须一致。
  - 合并后使用最后一个 D2H wr id 作为实际 verbs WR;前面被合并掉的 atomic ring
    slots 记录到 `atomic_completion_aliases_`。
  - 当实际 WR 的 CQE 到来时,`expand_atomic_completion_aliases()` 把 alias wr ids
    一起加入 `acked_wrs_`,避免 D2H ring 因被合并的 atomic slot 未 ack 而卡死。
  - 合并只在 `value` 累加仍落在 `[-kMaxSendAtomicValue, kMaxSendAtomicValue]`
    范围时发生。
  - profile 输出新增:
    - `posted_atomic_wrs`
    - `coalesced_atomic_wrs`
    - `poll_us`
    - `progress_atomic_us`
    - `post_gpu_us`
  - profile timing 仅在 `UCCL_PROXY_PROFILE_COMMANDS` 打开时取时钟,不污染默认 hot path。

当前状态:

- 仅本地修改,未同步/编译/运行服务器。
- `git diff --check` 通过。

## 2026-06-06:清理服务器环境,用 CUDA 13/NCCL 2.30.4 验证 UCCL-GIN coalescing

用户要求:

- 上服务器操作,先清理之前环境。
- README 要求 CUDA 13+,不能继续使用临时装错的 CUDA 12.8 / PyTorch cu128 环境。

环境清理和重新搭建:

- 错误环境:
  - `/home/ubuntu/uccl-gin-cu13-venv` 是一次误建环境,已不再使用。
  - EFS 上的半成品 `/home/ubuntu/efs/yzhou/playground/daniel/.venvs/uccl-gin-cu13`
    删除时遇到 stale file handle,不要依赖它。
- 正确环境:
  - 两台机器新建 `/home/ubuntu/.venvs/uccl-gin-cu13`。
  - 安装 `torch==2.12.0+cu130`,`cuda-toolkit==13.0.2`。
  - PyTorch cu130 默认安装 `nvidia-nccl-cu13==2.29.7`,但 vendored DeepEP V2
    `_C` 按 NCCL `2.30.4` 编译。第一次 smoke 失败:
    `NCCL library version is too old. This application was compiled with NCCL version 23004, but is running with NCCL library version 22907.`
  - 随后升级两台机器 venv 内 `nvidia-nccl-cu13==2.30.4`。虽然 pip 提示与 torch
    declared dependency 不一致,但这是当前 DeepEP V2 `_C` 必需的运行时版本。
- 当前实例 IP 已变化:
  - `p5en_0`: `ip-172-31-70-225`,内网 IP `172.31.70.225`
  - `p5en_1`: `ip-172-31-71-140`,内网 IP `172.31.71.140`
  - 旧 `MASTER_ADDR=172.31.78.36` 已失效,会导致 c10d `No route to host`。

构建:

- 同步本地 tracked 工作树到
  `/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/`。
- `p5en_0`/`p5en_1` 都执行:
  - `source /home/ubuntu/.venvs/uccl-gin-cu13/bin/activate`
  - `CUDA_HOME=/usr/local/cuda-13.0`
  - `LD_LIBRARY_PATH` 优先包含:
    `/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/nvidia/nccl/lib`
  - `make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16`
- 编译通过。`ep.abi3.so` 安装到两台机器各自的
  `/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/uccl/`。
- import check:
  - `torch 2.12.0+cu130`,CUDA `13.0`,GPU 可用。
  - `uccl.ep` 从当前 venv import。
  - `deep_ep` 从 vendored
    `thirdparty/DeepEP-v2-d4f41e4/deep_ep` import。

验证 1:EP8x2 smoke

- 命令:
  `--num-processes 8 --test-first-only --skip-perf-test --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256`
- 环境:
  - `MASTER_ADDR=172.31.70.225`
  - `DEEPEP_USE_UCCL_GIN=1`
  - `EP_JIT_EXTRA_FLAGS=-DDEEPEP_USE_UCCL_GIN`
  - `NCCL_NET_PLUGIN=ofi`
  - `FI_PROVIDER=efa`
  - `FI_EFA_USE_DEVICE_RDMA=1`
  - `OFI_NCCL_FORCE_NUM_RAILS=4`
  - `UCCL_PROXY_PROFILE_COMMANDS=1`
- 日志:
  - rank0:`/tmp/uccl_gin_cu13_smoke_rank0.log`
  - rank1:`/tmp/uccl_gin_cu13_smoke_rank1.log`
- 结果:rank0/rank1 rc 均为 0。

验证 2:README-like EP8x2 first-case

- 命令:
  `--num-processes 8 --test-first-only --num-sms 20 --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256`
- 日志:
  - rank0:`/tmp/uccl_gin_cu13_readme_rank0.log`
  - rank1:`/tmp/uccl_gin_cu13_readme_rank1.log`
- 结果:rank0/rank1 rc 均为 0。
- dispatch / expanded dispatch / cached dispatch:
  - 两台机器所有 rank 基本都在 `8 GB/s (RDMA/SO)`,latency 约 `15.9-16.1 ms`。
  - 这比上一轮 node0 的 `36-37 GB/s` 更差,说明当前 coalescing 改动没有带来预期
    性能收益,且 CUDA13/NCCL2304 这轮的整体行为需要和上一轮环境做 A/B。
- combine / reduced combine:
  - 大多 `14-22 GB/s (RDMA/SO)`。
  - combine 仍主要是上游 NCCL-GIN/DeepEP 路径,不是完整 UCCL-GIN combine 替换。

proxy profile 结论:

- 每个 proxy thread 仍看到大量 atomic:
  - `atomic_cmds` 约 `103k-156k`。
  - `posted_atomic_wrs` 也约 `102k-156k`。
  - `coalesced_atomic_wrs` 只有几百到一千左右。
- 因此当前 conservative “同 pending batch 内相邻同目标 atomic 合并”命中率太低,
  基本没有真正减少 receiver `WRITE_WITH_IMM` apply 数量。
- 后续如果继续走 coalescing,需要改变合并窗口或按照原 UCCL/EP 的 semantic batching
  思路在 device/proxy 协议层制造更大的同目标连续批次;仅在现有 pending batch 内合并
  相邻 atomic 远远不够。

## 2026-06-06:继续分析 batching,阅读原 UCCL-EP 和 NCCL GIN proxy 实现

用户要求:

- 参考原 `uccl/ep` 的 batching 写法,不要只靠自己写的新策略。
- 进一步看 NCCL GIN 是怎么处理 `AggregateRequests` / batching / proxy 的。
- 每次重要发现和更改都必须写进 `worklog.md`。

本地代码改动记录:

- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - notify count 的 Rail put lane hint 从默认 queue 改成 `thread_idx`。
  - payload Rail put 仍按 `channel_idx` 选择 lane。
  - 目的:避免所有 notify 小写集中到 queue 0,让它和 scaleout lane 分散方式更接近。
- `ep/src/proxy.cpp` / `ep/include/proxy.hpp`
  - atomic coalescing 从“相邻同目标”扩展成“同一个 pending batch 内按目标聚合”。
  - 合并条件仍保守:同 D2H ring、同 `dst_rank`、同 `cmd_type`、同 `req_rptr`、
    同 `atomic_offset`,且不是 low-latency atomic,累加后的 `value` 仍能放进
    `[-kMaxSendAtomicValue, kMaxSendAtomicValue]`。
  - 新增 `ready_atomic_batch_`:已经满足依赖的 pending atomic 不立刻 post,先攒到
    `512` 个 wr 或 force/drain 时再 coalesce + post。
  - 注意:这部分是本地实验性修改,目前尚未重新在服务器编译/跑 benchmark。
    上一轮服务器验证的是更早版本的 coalescing,结果仍约 `8 GB/s (RDMA/SO)`。

阅读原 UCCL-EP V1 batching:

- 参考文件:
  `/Users/daniel/Documents/code/uccl-danyang/uccl-danyang/ep/src/internode.cu`
  以及当前仓库原 V1 路径 `ep/src/internode.cu`。
- 关键观察:
  - 原 V1 并不是在 CPU proxy 里简单把很多零散 WR 合成一个 batch。
  - 它先在 GPU 语义层按 RDMA rank/channel 维护 send window/tail/next_tail,把 token
    copy 到 per-dst channel send buffer。
  - coordinator 再按连续 token chunk 发起大块 RDMA:
    `num_bytes_per_token * num_tokens_to_issue`。
  - EFA 分支把 tail add 作为 RDMA send 参数的一部分传下去,即 payload chunk 和 tail
    增量在语义上是一组 batch。
  - V1 receiver 还有 per-token/source epoch tag 作为 ready 判断,不是只依赖
    sender-side CQ 顺序。
- 对 V2 的影响:
  - 当前 V2 `hybrid_dispatch.cuh` 的 local staging 是按原始 `token_idx` sparse 存在
    `scaleout_send_buffer.get_token_buffer(token_idx)`。
  - 远端 expanded slot 对同一个 channel/dst 是连续的,但本地源地址不是连续 per-dst
    compact layout。
  - 因此要完全复刻 V1 的“一个 RDMA WRITE 发送多个 token chunk”,需要两条路之一:
    - 改 V2 staging layout,把发往同一 dst/channel 的 token 先 compact 到连续 send buffer。
    - 或者在 proxy/verbs 层支持 multi-SGE gather:多个 sparse local token buffer SGE
      写到一段连续 remote expanded slots。
  - 当前 EFA QP 创建时 `max_send_sge = 1`,所以 multi-SGE 不是小改;需要改 QP cap、
    post path 和 completion alias。

阅读 NCCL GIN 源码:

- 本地 NCCL 源码路径:
  - `nccl/src/include/nccl_device/gin.h`
  - `nccl/src/include/nccl_device/gin/proxy/gin_proxy.h`
  - `nccl/src/gin/gin_host_proxy.cc`
  - `nccl/src/transport/net_ib/gin.cc`
  - `nccl/src/include/nccl_device/gin/gdaki/gin_gdaki.h`
- `AggregateRequests` 的真实含义:
  - `ncclGinOptFlagsAggregateRequests` 在 GDAKI/device verbs 后端被映射成
    `DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING`。
  - 也就是说,在真 device verbs 路径里它主要是“跳过 doorbell、让后续 flush/doorbell
    合并提交”的语义。
  - 在 CPU proxy 后端,`ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY>` 接收 `optFlags`,
    但实际调用内部 `nccl::gin::proxy::put(...)` 时没有继续使用该 flag。
  - 结论:对 NCCL proxy GIN 来说,`AggregateRequests` 本身不会自动把多个 payload
    GFD/WR 合成一个大 WR。
- NCCL proxy device queue:
  - GPU 端写的是 128B `ncclGinProxyGfd_t` descriptor 到 per-peer queue。
  - `postGfd()` 只做 PI/CI credit、descriptor 发布和 queue slot 管理。
  - 每个 GFD 仍代表一个 put/get/signal/flush 请求。
- NCCL host proxy:
  - `ncclGinProxyProgress()` 每次 poll completions,然后对每个 target rank 最多 poll
    一个 GFD 并直接调用 backend `iput` / `iputSignal` / `iget` / `iflush`。
  - 没看到 host proxy 把多个 GFD 合并成一个 verbs WR 的逻辑。
- NCCL IB proxy 的 payload-before-signal:
  - `ncclGinIbProxyIPutSignal()` 生成两个 WR:
    - payload `IBV_WR_RDMA_WRITE`,不 signaled。
    - signal `IBV_WR_ATOMIC_FETCH_AND_ADD`,signaled。
  - 两个 WR 链在一起,一次 `ibv_post_send` 提交。
  - 它不是“先等 payload CQE,再 post signal”;因此不会像我们早期同步 drain 那样
    把流水线深度限制在几个 token。

本轮设计结论:

- NCCL GIN proxy 给我们的主要启发不是“proxy 自动 payload coalescing”,而是:
  - device queue 要薄,proxy 要持续 drain。
  - payload 和 signal/tail 应尽量同批提交,不要在每个 tail 前同步 CQ drain。
  - `AggregateRequests` 对 EFA CPU proxy 不能直接带来 payload batch;如果 AWS EFA
    路径要减少小 WR,必须由 UCCL-GIN 自己在 V2 semantic 层或 verbs gather 层做 batch。
- 更接近原 UCCL-EP 哲学的下一步候选:
  - 短期:保持异步 tail dependency,移除同步全局 drain;继续降低 receiver atomic apply
    次数,但不要指望 NCCL 的 `AggregateRequests` 自动生效。
  - 中期:实现 V2 semantic batching,让同一 channel/dst 的多个 token 成为一个
    payload batch。由于 V2 local source sparse,需要评估 compact staging vs
    multi-SGE gather。
  - 如果走 multi-SGE gather,必须同步修改 EFA QP `max_send_sge`、EFA post path、
    completion alias 和 imm/ack 语义;这比单纯 proxy-side atomic coalescing 更接近
    真正 payload batching。

当前状态:

- 本轮只做本地源码阅读和文档更新。
- 最新本地实验性代码尚未同步服务器、未编译、未 benchmark。

## 2026-06-06:profiling V2 semantic batching 是否值得做

背景:

- 用户提出:在下结论前先 profiling 关键路径,再判断是否可以实现 V2 semantic
  batching,让同一 channel/dst 的多个 token 成为一个 payload batch。
- 本轮目标不是先改 batching 语义,而是量化现有 V2/UCCL-GIN 命令流里到底有多少
  可 batch 的连续 token。

本地 instrumentation:

- 修改 `ep/include/proxy.hpp` 和 `ep/src/proxy.cpp`,在
  `UCCL_PROXY_PROFILE_COMMANDS=1` 时追加只读 profile,不改变 transfer 语义。
- 在 `post_gpu_commands_mixed()` 开始处统计:
  - `stream_remote_*`: 当前命令流中连续 WRITE,同 ring/dst/bytes 且 remote offset
    连续。
  - `stream_local_*`: 同时要求 local offset 也连续。
  - `semantic_remote_*`: 按 `(ring,dst)` 忽略中间 tail/atomic 后,remote offset 是否可
    继续连续。
  - `semantic_local_*`: semantic run 中 local offset 是否也连续。
  - `semantic_gather_*`: remote 连续但 local 不连续的 semantic run,即只能通过
    multi-SGE gather 或额外 compact staging 才可能合成一个 payload batch。
- 修正了前一轮实验性 `ready_atomic_batch_` 路径:该路径在 smoke test 中触发
  DeepEP forward timeout,不作为正确 profiling 基线;已删除,恢复成原本按
  pending batch 顺序 progress 的路径。

服务器构建:

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang
source /home/ubuntu/.venvs/uccl-gin-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export LIBRARY_PATH=/home/ubuntu/local-lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ubuntu/local-lib:/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16
```

构建结果:

- p5en_0 构建通过,只有 warning。
- 同步 `ep/ep.abi3.so` 到 p5en_1 的 venv `uccl/` 包目录。

smoke test:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-perf-test \
  --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256
```

环境:

```bash
export DEEPEP_USE_UCCL_GIN=1
export EP_JIT_EXTRA_FLAGS=-DDEEPEP_USE_UCCL_GIN
export UCCL_PROXY_PROFILE_COMMANDS=1
export NCCL_NET_PLUGIN=ofi
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=4
unset EP_DISABLE_GIN
unset OFI_NCCL_GIN_GDAKI
```

smoke 结果:

- 两节点均 `rc=0`。
- 日志:
  - p5en_0: `/tmp/uccl_gin_semantic_profile_smoke2_rank0.log`
  - p5en_1: `/tmp/uccl_gin_semantic_profile_smoke2_rank1.log`
- 小配置已经显示 local contiguous run 为 0,remote contiguous run 很短。

README-like profile:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-check \
  --num-sms 20 --num-tokens 8192 --hidden 7168 \
  --num-topk 8 --num-experts 256 --ignore-local-traffic
```

结果:

- 两节点均 `rc=0`。
- 日志:
  - p5en_0: `/tmp/uccl_gin_semantic_profile_readme_rank0.log`
  - p5en_1: `/tmp/uccl_gin_semantic_profile_readme_rank1.log`
- 打开 profiling 后 bandwidth 不代表最终性能,但可用于判断命令结构:
  - dispatch/expanded/cached 大约 `4 GB/s (RDMA/SO)`,约 `15.8-16.2 ms`。
  - combine/reduced combine 大约 `7-11 GB/s (RDMA/SO)`。

两节点 profile 汇总:

```text
profiles                         64
write_cmds sum                   24,282,920
write_bytes sum                  182,578,798,336
atomic_cmds sum                  8,284,068
posted_atomic_wrs sum            8,236,036
coalesced_atomic_wrs sum         48,032

stream_remote_runs sum           1,562,611
stream_remote_tokens sum         3,365,843
stream_remote_max                3
stream_local_runs sum            0
stream_local_tokens sum          0
stream_local_max                 0

semantic_remote_runs sum         2,124,482
semantic_remote_tokens sum       4,626,098
semantic_remote_max              6
semantic_local_runs sum          0
semantic_local_tokens sum        0
semantic_local_max               0

semantic_gather_runs sum         2,124,482
semantic_gather_tokens sum       4,626,098
semantic_gather_max              6
semantic_local_token_fraction    0/4626098
semantic_gather_token_fraction   4626098/4626098
```

解释:

- 现有 command stream 里真正连续出现的 remote payload 最多只有 3 个 token。
- 即使按 `(ring,dst)` 忽略 tail/atomic,把同一 channel/dst 的 token 当作 semantic
  run,remote contiguous run 也只有 5-6 个 token。
- `semantic_local_runs=0` 表明这些候选 batch 的本地源地址从未连续;全部
  `semantic_remote_tokens` 都落入 `semantic_gather_tokens`。
- 因此“在 proxy 里把已有 token WRITE 简单拼成一个大 contiguous WRITE”不成立:
  remote expanded slot 偶尔连续,但 local staging 是按原始 token index sparse 存放。

当前判断:

- V2 semantic batching 这个方向不是完全走不通,但不能是 proxy-only 的简单 WR
  coalescing。
- 如果只在 proxy 层做,最多只能做 5-6 token 的 multi-SGE gather batch;这要求改
  EFA QP `max_send_sge`、post path、completion alias,而且 batch 太小,未必能抵消
  gather/SGE 的 CPU 和 NIC 成本。
- 更像原 UCCL-EP/V1 的高性能方案,应该是在 V2 JIT/semantic 层引入 compact
  per-channel/per-dst send staging,让本地源地址也连续,再发一个真正的大 payload
  WRITE batch。
- 下一步如果继续推进 semantic batching,建议先实现/评估 compact staging 的额外
  GPU copy 成本,而不是直接改 proxy 做 multi-SGE。

## 2026-06-06:对比 V1 chunked RDMA 与 V2 sparse send buffer 后的 batching 设计判断

对照源码:

- V1: `ep/src/internode.cu:1066-1106`
  - coordinator warp 等到某个 dst/channel 有足够 token ready。
  - `num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens)`。
  - 单次 `nvshmemi_ibgda_put_nbi_warp()` 发送
    `num_bytes_per_token * num_tokens_to_issue`。
  - EFA 分支把 `rdma_channel_tail` offset 和 `num_tokens_to_issue` 作为参数传下去,
    payload WRITE 与 tail delta 在语义上绑定,不是额外每 token 一个独立 tail WR。
- V2: `thirdparty/DeepEP-v2-d4f41e4/.../hybrid_dispatch.cuh:421-508`
  - 每个 scaleout warp 按 `token_idx = channel_idx; token_idx += kNumChannels`
    遍历原始 token。
  - 先把当前 token TMA store 到
    `scaleout_send_buffer.get_token_buffer(token_idx)`。
  - 远端目的地是
    `scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx)`。
  - 随后对单个 token 调 `gin.put(..., tma_buffer.get_num_bytes<false>(), ...)`。

关键差异:

```text
V1 send layout:

  send_buffer[dst_rank][channel][slot 0][slot 1][slot 2]...
                              └──── contiguous per dst/channel ────┘

  coordinator sees ready tail/head:
      slot 0..5 ready  ->  one RDMA WRITE for 6 tokens


V2 current send layout:

  scaleout_send_buffer[token_idx]

  token_idx stream in one channel:
      token 0 -> dst 1 -> local slot token[0]
      token 80 -> dst 3 -> local slot token[80]
      token 160 -> dst 1 -> local slot token[160]

  remote expanded slots may be contiguous per dst/channel,
  but local source slots are sparse in original token_idx space.
```

设计判断:

- 用户提出的 “V1 每 WRITE 大约 6 tokens,当前 V2 每 WRITE 1 token,WR 数量差巨大”
  是结构性问题,不是简单 tuning。
- 但直接在 smem 里攒 6-8 个 token 再发也不现实:
  - 当前 `tma_buffer` 是按 warp/单 token 设计。
  - hidden=7168 时单 token staging 已经是十几 KB 量级。
  - 多个 scaleout/forward warp 同时持有多 token batch 会超过 H200 单 SM shared memory
    可用容量。
- 因此真正可行的方向不是 “smem batch”,而是 “global compact staging + coordinator”:

```text
Scaleout producer warps:

  input token stream
        |
        | classify dst/channel, reserve compact slot
        v
  compact_send[dst][channel][slot]
        |
        | publish per-dst/channel ready tail
        v

Coordinator / issuer warp:

  read ready tail/head
        |
        | issue chunks of N tokens
        v
  RDMA WRITE compact_send[dst][channel][head:head+N]
        |
        | piggyback / coalesced tail delta N
        v
  remote V2 expanded recv[dst][channel][slot:slot+N]
```

需要注意的 V2 语义:

- compact slot 必须仍然对应 V2 expanded slot,不能回到 V1 packed token staging。
- `dst_buffer_slot_idx`、`token_metadata_at_forward`、forward linked-list/reorder 所需
  metadata 需要能从 compact slot 映射回 V2 的 expanded slot 和原始 token metadata。
- dispatch 可以先做;combine 后续要按 V2 reduced-combine 的反向 metadata 设计同类
  compact/gather 机制。

短期不推荐:

- 不推荐先做 proxy-only multi-SGE gather:
  - profile 显示 semantic remote run 最大只有 5-6。
  - local contiguous run 为 0,所有候选都需要 gather。
  - EFA QP 当前 `max_send_sge=1`;改 multi-SGE 要动 QP cap、post path、completion
    alias,收益不一定覆盖 SGE 开销。
- 不推荐在 smem 里直接攒 token batch:
  - shared memory 容量和 V2 多 warp 并发模式不匹配。

更合理的阶段方案:

1. 在 V2 dispatch JIT 中新增 global compact send buffer layout:
   `compact_send[scaleout_rank][channel][slot]`。
2. producer warp 对每个 remote token:
   - 用 per-dst/channel tail reserve compact slot。
   - TMA store 单 token 到 compact slot。
   - 写必要 metadata: original token idx、topk、weight、expanded slot。
3. coordinator warp 复用 V1 chunked send 思路:
   - 每 dst/channel 读 ready tail 和 last issued head。
   - 满 `chunk_tokens` 或 finish 时发一个大 payload WRITE。
   - tail delta 按 chunk 发,避免每 3 token 独立 tail。
4. receiver 仍直接落 V2 expanded layout,不要引入 V1 packed receive staging。
5. 先用 instrumentation 估算:
   - compact copy cost。
   - chunk size 分布。
   - WR 数减少比例。
   - tail WR/apply 减少比例。

## 2026-06-06:review `ep/docs/uccl_gin_compact_staging.md`

用户指出:

- 如果方案等于重写半个 scaleout kernel,并且引入新的 V1-like buffer,那升级到 V2 的
  意义会被削弱。
- 需要 review `ep/docs/uccl_gin_compact_staging.md` 中“不新增 buffer,只改
  `scaleout_send_buffer` 索引方式”的方案。

代码对照:

- `hybrid_dispatch.cuh` 当前 dispatch buffer:
  - `scaleup_buffer`
  - `scaleout_send_buffer = BufferLayout(token_layout, 1, kNumMaxTokensPerRank, ...)`
  - `scaleout_recv_buffer = BufferLayout(token_layout, kNumScaleoutRanks,
    kNumChannels * kNumMaxTokensPerChannel, scaleout_send_buffer.get_buffer_end_ptr())`
- `buffer.hpp::get_dispatch_buffer_size()` 也只给 hybrid dispatch 的
  `scaleout_send_buffer` 预留 `kNumMaxTokensPerRank` 个 token slot。
- 因此文档里 “同一个 `scaleout_send_buffer`,换一种索引方式,不新增 buffer” 这个
  目标是好目标,但示例代码
  `BufferLayout(token_layout, kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel, ...)`
  实际会把 send buffer token slot 数扩成约
  `kNumScaleoutRanks * kNumMaxTokensPerRank`,不是同一块大小。

我认为文档方向正确的部分:

- 关键洞察是对的:`scaleout_send_buffer` 只有 scaleout warp 写和同一 scaleout warp
  读,receiver/forward 不读它。因此改它的临时发送索引,不等于回到 V1 的接收语义。
- “不在 smem 攒 batch,直接把 TMA store 目标换成 compact send slot” 是正确方向,
  避免了 shared memory 容量问题。
- 这不会破坏 V2 最大价值:
  - receiver 仍是 V2 expanded layout。
  - forward metadata、linked list、`dst_buffer_slot_idx` 仍按 V2 走。
  - 没有引入 V1 `SourceMeta` / prefix matrix / packed receive staging。
  - 只是把 scaleout sender 的临时 send scratch 从 `token_idx` 视图换成
    channel/dst compact 视图。

文档需要修正的 load-bearing 问题:

1. buffer 大小假设不成立。
   - 如果要支持任意 `kNumScaleoutRanks`,每个 `(channel,dst)` 都分
     `kNumMaxTokensPerChannel` slot,就需要约
     `kNumScaleoutRanks * kNumMaxTokensPerRank` 个 send slot。
   - 当前 DeepEP V2 只预留 `kNumMaxTokensPerRank` 个 send slot。
   - 若坚持不新增 buffer,只能做更紧的 layout:
     - 对 EP8x2/两节点优化:每个 rank 只有一个 remote scaleout dst,local bypass
       不用 send buffer,因此可把同一块 send buffer 解释成
       `send[channel][slot]`,slot 可直接用 `stored_dst_slot_idx`。
     - 对多 scaleout rank:需要 overflow/ring credit 或增加 buffer size;不能只靠
       `slots_per_channel_dst = tokens / channels / ranks`。极端路由下一个 channel
       的所有 token 都可能去同一个 remote dst。
2. `cur_batch` 单 batch 状态会在 dst 交错时频繁 flush。
   - 如果 token 流是 `dst0,dst1,dst0,dst1`,单个 `cur_batch` 基本退化成 1-token
     batch。
   - 应该维护 per-dst batch 状态数组,至少对 scaleout dst 数量很小的 EP8x2 可以
     用一个 remote batch;多节点则每 dst 一个小状态。
3. compact slot 不需要新的 `atomicAdd`。
   - `stored_dst_slot_idx` 已经是当前 channel/dst 的 V2 recv slot,由
     `ptx::exchange(stored_scaleout_tail, dst)` 得到,天然唯一且单调。
   - 可以直接用 `compact_slot = stored_dst_slot_idx` 作为 send compact slot,这样
     local compact slot 和 remote expanded slot 一一对应,不需要额外
     `compact_tail[channel][dst]`。
4. `kNumChannels` 不应手写估算。
   - kernel 中 `kNumChannels = kNumScaleoutWarps * kNumSMs`。
   - 实际 README-like profile 更像 80 个 channel,文档里的 `8 x 20 = 160` 需要
     用 JIT 参数确认,不能作为容量证明。

更推荐的收敛版本:

```text
EP8x2 first:

  scaleout_send_buffer still has kNumMaxTokensPerRank slots

  reinterpret as:
    send[channel][slot], slot in [0, kNumMaxTokensPerChannel)

  for remote token:
    compact_slot = stored_dst_slot_idx
    TMA store -> send[channel][compact_slot]
    append to this channel's current remote batch

  coordinator/flush:
    when contiguous compact slots accumulated to N or finish:
      rail_put(send[channel][first_slot],
               recv[remote_rank][channel][first_slot],
               N * token_bytes)
      rail_tail_add(N, finish?)
```

这个版本的优点:

- 不新增 send buffer,不改 `calculate_buffer_size()`。
- 不新增 compact tail allocator。
- 对 EP8x2/两节点最强:每个 local rank 只有一个 remote scaleout peer,local bypass 不占
  send buffer。
- 仍然保留 V2 expanded receiver layout。

限制:

- 这首先是 EP8x2/AWS 两节点优化。若要推广到 3+ scaleout ranks,必须重新处理
  send buffer 容量:
  - 要么增加 send buffer 至 `(kNumScaleoutRanks - 1) * kNumMaxTokensPerRank` 级别。
  - 要么做 per-dst ring/credit/reuse,但这会明显接近 V1 coordinator 复杂度。
- batch 大小受每个 channel 的 token 数限制,不会达到论文 HT 的 32 token 常态;
  但从 1 token WRITE 变成 4-8 token WRITE 仍然可能显著改善 EFA 小包瓶颈。

结论:

- `uccl_gin_compact_staging.md` 的大方向值得做,但应改成 “EP8x2 优先、不新增
  buffer、用 `stored_dst_slot_idx` 作为 compact send slot” 的版本。
- 不建议按文档当前示例直接实现 `[channel][dst][slot]` 全量分区,因为那会扩大
  send buffer 或在 skew route 下溢出,与“不新增 buffer”的目标冲突。

## 2026-06-06:按 UCCL-EP 论文修正 compact staging 的 chunk 目标

用户指出:

- 4-8 token batch 不够。
- `2512.19849v2.pdf` 的 EFA/HT 优化中 chunk 默认/典型是 32 token。

重新阅读论文相关段落:

- §2.2:GPU-initiated token-level communication 的 transfer unit 可以是 per-token 到
  32 tokens,用于在细粒度 overlap 和网络利用率之间折中。
- §3.3:HT kernel 使用多个 communication channel/ring buffer 暂存待发送 token,以
  configurable chunk 发送,典型值 32 tokens。
- §3.3/§4.1:在 EFA/SRD 上不能靠 sender 等 CQE 来保证 write-then-atomic;更好的方式是
  receiver-side immediate/control-buffer/reorder,只对局部 channel enforce ordering。

对 V2 compact staging 的修正:

- 之前把 “4-8 token” 作为目标是不够的,只能作为 smoke/debug 阶段参数。
- 最终目标应是 32-token chunk:
  - FP8 hidden=7168 时单 token 约 11-14KB。
  - 32 token WRITE 约 350-450KB,进入 EFA 大消息区间。
- 32-token chunk 不是靠 smem batch,也不是靠 proxy gather,而是靠 sender-side
  per-channel ring/window。

重要纠正:

- 不应把同一个 send buffer 平均切成 `[channel][dst][slot]`:
  - 这会导致每 dst slot 只有平均份额,无法保证 skew route。
  - 如果给每 dst 都保留 `kNumMaxTokensPerChannel`,则实际扩大 send buffer。
- 对 EP8x2 first,可以不新增 payload buffer:
  - local dst 走 bypass,不占 `scaleout_send_buffer`。
  - 每个 rank 只有一个 remote scaleout dst。
  - 因此可把现有 `scaleout_send_buffer` 解释成 `send[channel][slot]`。
  - `slot = stored_dst_slot_idx`,即 V2 已经分配的 remote expanded slot。
  - README-like `num_sms=20`,若 `num_channels_per_sm=4`,则
    `kNumChannels=80`,`kNumMaxTokensPerChannel=ceil(8192/80)=103`;
    每 channel 可发 3 个 32-token chunk + 1 个残余 chunk。

更新了 `ep/docs/uccl_gin_compact_staging.md`:

- 把目标从 6-8 token 改成 32 token。
- 明确 EP8x2 first 的 no-new-buffer 条件。
- 移除新增 `compact_tail/atomicAdd` 的设计,改用 `stored_dst_slot_idx`。
- 明确 3+ scaleout ranks 需要额外 send capacity 或 per-dst ring credit,不能套用
  EP8x2 简化版。

## 2026-06-06:实现 EP8x2 UCCL-GIN dispatch compact staging 初版

用户指出的 buffer size review:

- 文档/旧方案中的
  `BufferLayout(token_layout, kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel, ...)`
  会让 send buffer 变成 `kNumScaleoutRanks` 倍,不符合“不新增 buffer”。
- EP8x2 下每个 rank 只有一个 remote scaleout dst,local dst bypass,因此 send buffer
  应该按 `send[channel][slot]` 解释,不需要乘 `kNumScaleoutRanks`。

本地代码改动:

- 文件:
  `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
- 仅在 `DEEPEP_USE_UCCL_GIN` 下启用 compact dispatch;非 UCCL/NCCL 原路径不动。
- 新增 UCCL-GIN EP8x2 static assertions:
  - `kNumScaleoutRanks == 2`。
  - padded compact send + recv layout 仍落在 DeepEP V2 原 dispatch buffer size 内。
- send buffer:
  - 从 `kNumMaxTokensPerRank` 变为 `kNumChannels * kNumMaxTokensPerChannel` 的 padded
    per-channel view。
  - 不改 host `calculate_buffer_size()`;利用上游 dispatch recv buffer 中为
    `kNumChannels * kNumMaxTokensPerChannel` 预留的 padding slack。
- remote payload:
  - remote lane 用 `compact_remote_slot_idx =
    ptx::exchange(stored_scaleout_tail, remote_scaleout_rank_idx)` 获取当前 V2 remote
    expanded slot。
  - TMA store 到 `scaleout_send_buffer[channel][compact_remote_slot_idx]`。
  - warp-uniform `compact_batch_count` 攒到 `32` token 或遇到非连续 slot/finish 后 flush。
  - flush 时只由 remote lane 发一个大 `gin.put<Rail>` 和一个
    `rail_tail_add(count_delta)`。
- local payload/tail:
  - local rank 仍 bypass 到 `scaleout_recv_buffer`。
  - `update_scaleout_tail()` 在 UCCL-GIN 下只让 local lane 发 local tail;remote tail 由
    compact batch flush 负责,避免 tail 早于 batched payload 到达。

关键注意:

- flush lambda 必须 whole-warp 调用,不能只在 remote lane 调用,否则内部 `__syncwarp()`
  会死锁。当前 batch state 设计成 warp-uniform,只有实际 WR/tail post 由 remote lane 执行。
- 这是 EP8x2 first 实现,不是 3+ scaleout ranks 通用方案。
- 目前尚未服务器编译/验证。

服务器验证:

构建:

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang
source /home/ubuntu/.venvs/uccl-gin-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export LIBRARY_PATH=/home/ubuntu/local-lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ubuntu/local-lib:/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16
```

- p5en_0 构建通过。
- 将 `ep/ep.abi3.so` 复制到 p5en_1 的
  `/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/uccl/`。

第一次 smoke 失败原因:

- 日志:
  - p5en_0: `/tmp/uccl_gin_compact32_smoke_rank0.log`
  - p5en_1: `/tmp/uccl_gin_compact32_smoke_rank1.log`
- 失败发生在 NCCL init,不是 compact kernel:
  `Failed to bind NVLink SHARP (NVLS) Multicast memory ... Disable NVLS`.
- 后续验证都加 `NCCL_NVLS_ENABLE=0`。

smoke correctness:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-perf-test \
  --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256
```

环境关键项:

```bash
export DEEPEP_USE_UCCL_GIN=1
export EP_JIT_EXTRA_FLAGS=-DDEEPEP_USE_UCCL_GIN
export EP_JIT_CACHE_DIR=/tmp/deepep_jit_uccl_compact_32
export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=ofi
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=4
```

- 日志:
  - p5en_0: `/tmp/uccl_gin_compact32_smoke_nvls0_rank0.log`
  - p5en_1: `/tmp/uccl_gin_compact32_smoke_nvls0_rank1.log`
- 结果:无 traceback/assert/timeout,进程退出后无残留 GPU 进程。

README-like EP8x2 first-case:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-check \
  --num-sms 20 --num-tokens 8192 --hidden 7168 \
  --num-topk 8 --num-experts 256 --ignore-local-traffic
```

- 日志:
  - p5en_0: `/tmp/uccl_gin_compact32_readme_rank0.log`
  - p5en_1: `/tmp/uccl_gin_compact32_readme_rank1.log`
- 两节点 `EXIT:0`。
- dispatch:
  - rank0 local ranks: `18 GB/s (RDMA/SO)`,约 `3389-3404 us`。
  - rank1 local ranks: `18 GB/s (RDMA/SO)`,约 `3363-3388 us`。
- expanded dispatch:
  - rank0: `18 GB/s (RDMA/SO)`,约 `3381-3409 us`。
  - rank1: `18 GB/s (RDMA/SO)`,约 `3352-3374 us`。
- cached dispatch:
  - rank0: `18 GB/s (RDMA/SO)`,约 `3389-3409 us`。
  - rank1: `18 GB/s (RDMA/SO)`,约 `3307-3359 us`。
- combine/reduced combine 当前未改,仍约 `7-11 GB/s (RDMA/SO)`。

profile 验证 command 数:

- 日志:
  - p5en_0: `/tmp/uccl_gin_compact32_readme_profile_rank0.log`
  - p5en_1: `/tmp/uccl_gin_compact32_readme_profile_rank1.log`
- 两节点 `EXIT:0`。
- 汇总 `64` 条 `UCCL_PROXY_PROFILE`:

```text
write_cmds sum             956,288
write_bytes sum            182,578,798,336
atomic_cmds sum            952,320
posted_atomic_wrs sum      952,320
coalesced_atomic_wrs sum   0
```

对比 compact 前 profile:

```text
write_cmds sum    24,282,920  -> 956,288   (约 25.4x 减少)
atomic_cmds sum    8,284,068  -> 952,320   (约 8.7x 减少)
write_bytes sum   保持 182,578,798,336
```

结论:

- EP8x2 UCCL-GIN compact32 dispatch 初版已能跑通 README-like first-case。
- 32-token batching 确实显著降低 D2H/WR command 数。
- dispatch 从之前常见 `~8 GB/s` 提升到 `~18 GB/s`,但距离目标仍远;下一步应 profile
  剩余瓶颈,尤其是:
  - batched WRITE 是否仍被 tail/atomic dependency 或 proxy poll 限制。
  - 每 rank/proxy 只有多少并发 WR/QP 深度。
  - 32-token chunk 是否真的达到预期 WR size 分布,是否有大量 partial chunk。
  - combine 仍未 compact/native 优化。

## 2026-06-06: 参考原 UCCL/EP EFA write+piggyback atomic

背景:

- 原 `/Users/daniel/Documents/code/uccl-danyang/uccl-danyang/ep/src/internode.cu`
  的 EFA 路径在 `nvshmemi_ibgda_put_nbi_warp` 里把 payload RDMA WRITE 和 tail
  atomic delta 放在同一个 WR 上:

```cpp
uccl::nvshmemi_ibgda_put_nbi_warp<true>(
    dst_off, src_off, num_bytes_per_msg, dst_rank, channel_id, lane_id, ...,
#ifndef EFA
    0, 0
#else
    tail_offset_in_atomic_buffer,
    num_tokens_to_issue
#endif
);
```

- 当前 UCCL-GIN compact32 虽然已经把 payload WRITE 降到 chunk 级别,但仍然是:

```text
payload WRITE  +  独立 tail WRITE_WITH_IMM
```

  profile 里独立 `atomic_cmds` 仍有约 `952,320` 条。

本地代码改动:

- `ep/include/uccl_gin/uccl_gin_rail.cuh`
  - 新增 `rail_put_tail_add(...)`。
  - 保持旧 16B `TransferCmd` ABI: `CmdType::WRITE` + `bytes` + `req_lptr` +
    `req_rptr`,并把 count delta 放到 `TransferCmd::atomic_val` 8-bit 字段,
    tail offset 放到 `atomic_offset`。
  - 限制: piggyback count delta 必须在 `1..255`; finish delta 仍走单独
    `rail_red_add`。
- `ep/include/uccl_gin/uccl_gin_handle.cuh`
  - 新增 `UCCLGin::rail_put_tail_add(...)`,kernel 只表达 V2 channel/source tail
    语义,command 编码留在 UCCL-GIN backend。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - compact remote batch flush 从 `gin.put + gin.rail_tail_add(count)` 改为
    `gin.rail_put_tail_add(count)`。
  - 每个 channel 末尾只保留独立 finish `rail_tail_add(0, finish=true)`。
- `ep/src/rdma.cpp`
  - normal-mode WRITE piggyback 判定改为 `cmd.atomic_val > 0`,支持 tail offset 0
    这个合法槽位。
  - 复用已有的 `RDMA_WRITE_WITH_IMM` + `AtomicsImm::PackAtomicWithSeq` receiver
    reorder/apply 逻辑,没有新增 verbs 路径。
- `ep/include/proxy.hpp`, `ep/src/proxy.cpp`
  - profile 增加 `piggyback_atomic_write_cmds` 字段,用于确认 payload WR 中携带了
    多少 tail count update。

预期 profiling:

- `write_cmds` 应接近 compact32 原值。
- `piggyback_atomic_write_cmds` 应接近 remote payload chunk 数。
- `atomic_cmds` 应从约 `952,320` 显著下降,剩余主要是每个 channel 的 finish 控制更新。
- 服务器验证日志计划写到:
  - build: `/tmp/uccl_gin_piggy_build.log`
  - smoke: `/tmp/uccl_gin_piggy_smoke_rank0.log`,
    `/tmp/uccl_gin_piggy_smoke_rank1.log`
  - README-like: `/tmp/uccl_gin_piggy_readme_rank0.log`,
    `/tmp/uccl_gin_piggy_readme_rank1.log`
  - profile: `/tmp/uccl_gin_piggy_profile_rank0.log`,
    `/tmp/uccl_gin_piggy_profile_rank1.log`

服务器验证:

- 同步文件:

```bash
rsync -av ep/include/uccl_gin/uccl_gin_rail.cuh \
  ep/include/uccl_gin/uccl_gin_handle.cuh \
  ep/include/proxy.hpp ep/src/proxy.cpp ep/src/rdma.cpp \
  thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh \
  worklog.md \
  p5en_0:/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/ --relative
```

- 构建:

```bash
cd /home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang
source /home/ubuntu/.venvs/uccl-gin-cu13/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export LIBRARY_PATH=/home/ubuntu/local-lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ubuntu/local-lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16
```

  - 日志: `/tmp/uccl_gin_piggy_build.log`
  - 结果: `BUILD_RC:0`。只有既有 warning:
    - `write(...)` return value warning。
    - `std::optional<std::function<void()>> recv_hook` maybe-uninitialized warning
      来自 `src/uccl_ep.cc`。
  - 安装产物已从 EFS 复制到 `p5en_1`:
    `/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/uccl/ep.abi3.so`

中间遇到的问题:

- 第一次 smoke 少了 `PYTHONPATH`,rank0 报:
  `ModuleNotFoundError: No module named 'deep_ep'`。
- 第二次 smoke 少了 aws-ofi-nccl master 的 lib path,两端报:
  `NCCL GIN is unavailable`。修正为:

```bash
export PYTHONPATH=/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/thirdparty/DeepEP-v2-d4f41e4:/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang:$PYTHONPATH
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/home/ubuntu/local-lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
```

- 远端 `p5en_0` 没有 `p5en_1` SSH alias,所以不要在 `p5en_0` 上直接 `scp p5en_1:...`。
  这次改为本地分别 SSH 两台,或者让 `p5en_1` 从共享 EFS 复制 build 产物。
- `nohup ... &` launcher 会让外层 SSH 不可靠地滞留,导致 rank1 已启动但 rank0
  没启动。后续两机测试改用两个前台 SSH session,rank1 先开 session,rank0 后开 session。

smoke:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-perf-test \
  --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256
```

- 关键环境:
  `DEEPEP_USE_UCCL_GIN=1`,
  `EP_JIT_EXTRA_FLAGS=-DDEEPEP_USE_UCCL_GIN`,
  `EP_JIT_CACHE_DIR=/tmp/deepep_jit_uccl_piggy`,
  `NCCL_NVLS_ENABLE=0`,
  `LD_LIBRARY_PATH` 包含
  `/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib`。
- 日志:
  - `/tmp/uccl_gin_piggy_smoke_rank0.log`
  - `/tmp/uccl_gin_piggy_smoke_rank1.log`
- 结果:两端 `EXIT:0`。

README-like first-case:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --skip-check \
  --num-sms 20 --num-tokens 8192 --hidden 7168 \
  --num-topk 8 --num-experts 256 --ignore-local-traffic
```

- 日志:
  - `/tmp/uccl_gin_piggy_readme_rank0.log`
  - `/tmp/uccl_gin_piggy_readme_rank1.log`
- 两端 `EXIT:0`。
- dispatch:
  - rank0 EP0-7: `30 GB/s (SO)`,约 `2035-2051 us`。
  - rank1 EP8-15: `30 GB/s (SO)`,约 `2030-2057 us`。
- expanded dispatch:
  - 两端均约 `30 GB/s (SO)`,约 `2033-2064 us`。
- cached dispatch:
  - rank0: `30 GB/s (SO)`,约 `2042-2056 us`。
  - rank1: `30-31 GB/s (SO)`,约 `1994-2021 us`。
- combine/reduced combine 未做 piggyback/compact 优化,仍约 `7-11 GB/s (SO)`。

profile:

- 命令同 README-like first-case,额外设置
  `UCCL_PROXY_PROFILE_COMMANDS=1`。
- 日志:
  - `/tmp/uccl_gin_piggy_profile_rank0.log`
  - `/tmp/uccl_gin_piggy_profile_rank1.log`
- 两端 `EXIT:0`。
- dispatch 在 profile 开启时仍约 `30 GB/s (SO)`,约 `2002-2045 us`。
- 汇总 `64` 条 `UCCL_PROXY_PROFILE`:

```text
post_cmds_sum                    1,194,368
write_cmds_sum                     956,288
write_bytes_sum                182,578,798,336
piggyback_atomic_write_cmds_sum    952,320
atomic_cmds_sum                    238,080
completed_wrs_sum              110,429,355
posted_atomic_wrs_sum              238,080
coalesced_atomic_wrs_sum                 0
poll_us_sum                      5,743,428
progress_atomic_us_sum           1,395,330
post_gpu_us_sum                  5,248,920
post_batches_sum                   904,662
seconds_avg                          5.0186
cmds_per_sec_sum                238,053.272
```

按 proxy thread 汇总:

```text
thread 0: post_cmds=361,088, write_cmds=289,664, piggyback=285,696,
          atomic_cmds=71,424,  write_bytes=54,786,144,256
thread 1: post_cmds=357,120, write_cmds=285,696, piggyback=285,696,
          atomic_cmds=71,424,  write_bytes=54,801,849,600
thread 2: post_cmds=238,080, write_cmds=190,464, piggyback=190,464,
          atomic_cmds=47,616,  write_bytes=36,538,762,560
thread 3: post_cmds=238,080, write_cmds=190,464, piggyback=190,464,
          atomic_cmds=47,616,  write_bytes=36,452,041,920
```

对比上一轮 compact32 profile:

```text
write_cmds:    956,288 -> 956,288   (payload chunk 数不变)
atomic_cmds:   952,320 -> 238,080   (约 4x 减少)
piggyback:           0 -> 952,320   (count tail update 已并入 payload WRITE)
dispatch:        ~18 GB/s -> ~30 GB/s
```

结论:

- V1 风格的 EFA `WRITE + piggyback atomic` 在 V2 compact32 dispatch 上走通了。
- 独立 tail count `WRITE_WITH_IMM` 基本被消除;剩余 `atomic_cmds` 主要是每个
  channel/source 的 finish 控制更新。
- 性能从 compact32 的约 `18 GB/s (SO)` 提升到约 `30 GB/s (SO)`。
- 下一步真正的 dispatch 性能瓶颈不再是“每个 chunk 一个独立 count tail WR”,而更可能是:
  - 仍有 238k finish/control atomic WR。
  - proxy post/poll 和 D2H command drain 的 CPU 开销。
  - thread 0/1 比 thread 2/3 承担更多 command/bytes,需要检查 channel-to-queue 映射
    是否可更均匀。
  - combine 还没走 compact/piggyback/native V2 优化。

## 2026-06-06: review 后修补 proxy drain/dependency 和非 UCCL 编译参数

输入 review 覆盖的问题:

- `progress_pending_atomics(true)` force 路径 pop `PendingAtomicBatch` 前没有清理
  `atomic_dep_by_wr_`,理论上会留下指向 deque front 元素的悬空指针。
- `drain_pending_atomics` 里 `pending_atomic_updates` 是局部变量,传给
  `remote_process_completions` 后没有 `apply_pending_updates`。
- notify warp 的 `gin.put` 多传 UCCL-GIN 专用 `remote_action/lane_hint` 参数,非
  `DEEPEP_USE_UCCL_GIN` 编译会不匹配上游 `NCCLGin::put`。
- `wait_for_cq` outstanding 清零后还等 3 次空 poll,这是额外延迟,不是 correctness
  必需条件。

本地修复:

- `ep/include/proxy.hpp`
  - `PendingAtomicBatch` 新增 `dep_wrs`,记录这个 batch 注册到
    `atomic_dep_by_wr_` 的依赖 WR。
  - 新增 `clear_atomic_batch_deps(PendingAtomicBatch&)`。
- `ep/src/proxy.cpp`
  - `enqueue_pending_atomics` 记录 `dep_wrs`。
  - `progress_pending_atomics(force)` 在 pop batch 前调用
    `clear_atomic_batch_deps`;force 路径也会先清理反向映射,避免悬空指针。
  - `drain_pending_atomics` 不再用 `progress_pending_atomics(true)` 作为常规路径;
    它会继续 poll CQ,等 dependency WRITE 完成后再正常 post atomic,保留
    payload-before-tail 顺序。
  - `drain_pending_atomics` 在 `USE_RECEIVER_BARRIER && !use_normal_mode` 下调用
    `apply_pending_updates`,和主 `run_dual` 路径一致。
  - `wait_for_cq` 在 `pending_release_wrs` 清零后立即退出,去掉 3 次额外空 poll。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - notify warp 的两个 `gin.put<ncclTeamTagRail>` 分成 UCCL-GIN 和上游 NCCL-GIN
    两套参数,修复非 UCCL 编译形态。

复核后暂未改的 review 点:

- `ptx::exchange(last_forward_src_token_global_idx, recv_scaleout_rank_idx)`:
  `ptx::exchange` 是 warp shuffle,不是写入变量的 `std::exchange`;这行读取
  `recv_scaleout_rank_idx` lane 的单调 baseline,不会在 retry 时把 baseline 覆盖成
  rank id。因此本轮不改。
- self-path `rail_tail_add` 缺 fence:当前 self `put` 路径在返回前已有
  `__threadfence_system()`,且 review 后也确认这条原判断不成立。
- `coalesce_atomic_batch` 读 inactive union:当前 lambda 先排除 low-latency
  `cmd_type`,再读 normal-mode `atomic_offset`;当前 UCCL-GIN only non-LL ATOMIC。
  本轮不做结构性改动。

待服务器验证日志:

- build: `/tmp/uccl_gin_reviewfix_build.log`
- smoke:
  - `/tmp/uccl_gin_reviewfix_smoke_rank0.log`
  - `/tmp/uccl_gin_reviewfix_smoke_rank1.log`
- README-like:
  - `/tmp/uccl_gin_reviewfix_readme_rank0.log`
  - `/tmp/uccl_gin_reviewfix_readme_rank1.log`
- profile:
  - `/tmp/uccl_gin_reviewfix_profile_rank0.log`
  - `/tmp/uccl_gin_reviewfix_profile_rank1.log`

服务器验证结果:

- 同步到服务器分支 `uccl-gin`,目标路径
  `/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang`。
- build:
  - 命令: `make -C ep install PYTHON=$VIRTUAL_ENV/bin/python CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j 16`
  - 日志: `/tmp/uccl_gin_reviewfix_build.log`
  - 结果: `BUILD_RC:0`。只有既有 warning,没有编译错误。
- smoke:
  - 命令: `tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-sms 8 --num-tokens 64 --hidden 2048 --num-topk 6 --num-experts 256`
  - 日志:
    - `/tmp/uccl_gin_reviewfix_smoke_rank0.log`
    - `/tmp/uccl_gin_reviewfix_smoke_rank1.log`
  - 结果: 两端 `EXIT:0`。
- README-like first-case:
  - 命令: `tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-check --num-sms 20 --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 --ignore-local-traffic`
  - 日志:
    - `/tmp/uccl_gin_reviewfix_readme_rank0.log`
    - `/tmp/uccl_gin_reviewfix_readme_rank1.log`
  - 结果: 两端 `EXIT:0`。
  - rank0 dispatch 摘要: uncached / expanded / cached dispatch 均约
    `29-30 GB/s (SO)`,约 `2060-2080 us`。
  - combine 仍约 `8-11 GB/s (SO)`,本轮没有改 combine 路径。
- profile:
  - 命令同 README-like first-case,额外设置
    `UCCL_PROXY_PROFILE_COMMANDS=1`。
  - 日志:
    - `/tmp/uccl_gin_reviewfix_profile_rank0.log`
    - `/tmp/uccl_gin_reviewfix_profile_rank1.log`
  - 结果: 两端 `EXIT:0`,没有残留 `test_ep.py` 进程。
  - 汇总 `64` 条 `UCCL_PROXY_PROFILE`:

```text
post_batches_sum                 914,649
post_cmds_sum                  1,194,368
write_cmds_sum                   956,288
write_bytes_sum              182,578,798,336
piggyback_atomic_write_cmds_sum  952,320
atomic_cmds_sum                  238,080
quiet_cmds_sum                         0
barrier_cmds_sum                       0
completed_wrs_sum            125,384,721
posted_atomic_wrs_sum            238,080
coalesced_atomic_wrs_sum               0
poll_us_sum                    5,866,326
progress_atomic_us_sum         1,416,783
post_gpu_us_sum                5,243,800
seconds_avg                        4.7870
write_GBps_sum                    38.139
cmds_per_sec_sum              249,498.120
```

复核结论:

- 本轮修复没有改变 README-like dispatch 的主性能表现,仍保持 piggyback 版本的
  `~30 GB/s (SO)`。
- `quiet_cmds=0`、`barrier_cmds=0`,说明 hot dispatch path 没有退回同步 quiet/barrier
  drain。
- `coalesced_atomic_wrs=0` 是当前结构的预期结果:count tail 已经被 payload
  `WRITE_WITH_IMM` piggyback 消除,剩余独立 `atomic_cmds` 主要是 finish/control 类
  更新,同 target 可合并机会很少。
- 本轮真正修的是 correctness/race/维护性风险:
  - pending atomic batch pop 前清理 dependency backpointer。
  - drain 路径 apply receiver atomic updates。
  - 非 UCCL-GIN 编译不再给上游 NCCL-GIN `put` 多传参数。
  - wait-for-CQ 去掉完成后的额外 3 次空 poll。

## 2026-06-06: P4 per-batch WRITE merge opportunity profile

目标:

- 用户要求先不要直接实现 proxy WRITE 合并,而是先 profile 当前
  `post_gpu_commands_mixed` batch 里是否真的存在可安全合并的相邻 WRITE。
- 这个 profile 不改变传输行为,只回答 P4 的核心问题:如果按保守规则合并,理论上能省
  多少 WR/CQE。

本地修改:

- `ep/include/proxy.hpp`
  - 新增 merge opportunity counters:
    `merge_adjacent_pairs`, `mergeable_pairs`, `merge_runs`,
    `merge_run_cmds`, `merge_run_max`, `merge_saved_wrs`,
    `merge_fail_ring`, `merge_fail_target`, `merge_fail_local_gap`,
    `merge_fail_remote_gap`, `merge_fail_atomic`。
- `ep/src/proxy.cpp`
  - 扩展 `profile_write_batching_opportunity`。
  - 保守可合并定义:
    - 两条 WRITE 在同一个 D2H ring。
    - `dst_rank` 和 `cmd_type` 相同。
    - local offset 连续。
    - remote offset 连续。
    - piggyback atomic offset 相同,且合并后的 `atomic_val <= 255`。
  - `UCCL_PROXY_PROFILE` 输出新增上述 counters。

待服务器验证日志:

- build: `/tmp/uccl_gin_mergeprof_build.log`
- README-like:
  - `/tmp/uccl_gin_mergeprof_readme_rank0.log`
  - `/tmp/uccl_gin_mergeprof_readme_rank1.log`
- profile:
  - `/tmp/uccl_gin_mergeprof_profile_rank0.log`
  - `/tmp/uccl_gin_mergeprof_profile_rank1.log`

服务器验证结果:

- build:
  - 第一次链接失败: `cannot find -lnuma`。服务器只有 runtime `libnuma.so.1`,
    但用户目录已有 `/home/ubuntu/local-lib/libnuma.so -> /usr/lib/x86_64-linux-gnu/libnuma.so.1`。
  - 重新设置 `LIBRARY_PATH=/home/ubuntu/local-lib:$LIBRARY_PATH` 后 build 通过。
  - 日志: `/tmp/uccl_gin_mergeprof_build.log`,最终 `BUILD_RC:0`。
- profile:
  - 命令: README-like EP8x2 first-case,额外设置
    `UCCL_PROXY_PROFILE_COMMANDS=1`。
  - 日志:
    - `/tmp/uccl_gin_mergeprof_profile_rank0.log`
    - `/tmp/uccl_gin_mergeprof_profile_rank1.log`
  - 结果: 两端 `EXIT:0`。
  - dispatch 仍约 `29-30 GB/s (SO)`,说明新增 profile 没有改变 benchmark 结果。
  - 注意:新增 per-batch profile 使用 unordered map/run 扫描,proxy profile 的
    `seconds_avg` 从约 `4.8s` 增到 `22.3s`;因此这次 profile 只能看
    `merge_*` 机会比例,不能用来判断 proxy 真实吞吐。

汇总 `64` 条 `UCCL_PROXY_PROFILE`:

```text
write_cmds_sum                    956,288
piggyback_atomic_write_cmds_sum   952,320
atomic_cmds_sum                   238,080
merge_adjacent_pairs_sum          217,333
mergeable_pairs_sum                   361
merge_runs_sum                        361
merge_run_cmds_sum                    722
merge_run_max                           2
merge_saved_wrs_sum                   361
merge_fail_ring_sum               168,585
merge_fail_local_gap_sum           48,387
merge_fail_target_sum                   0
merge_fail_remote_gap_sum               0
merge_fail_atomic_sum                   0
stream_remote_runs_sum                  8
semantic_remote_runs_sum                8
```

结论:

- 保守 proxy-side adjacent WRITE merge 基本没有收益:
  `merge_saved_wrs_sum / write_cmds_sum = 361 / 956288 ≈ 0.038%`。
- 可合并 run 最大只有 `2`,没有出现能把多个 compact32 chunk 串成大 WR 的长 run。
- 失败原因主要是:
  - `merge_fail_ring`:相邻 WRITE 来自不同 D2H ring,占多数。
  - `merge_fail_local_gap`:同 ring 内也常常 local buffer 不连续。
- `remote_gap=0`、`atomic=0` 说明 remote slot 和 piggyback encoding 不是主要阻塞;
  真正阻塞是 D2H/ring interleaving 和 sender local compact buffer 顺序。
- 因此 `uccl_gin_perf_plan.md` 里的 P4 不应该作为实现项继续推进。除非先重排
  device-side command emission 或做更大的 sender-side semantic batching,否则在
  proxy 侧扫相邻命令合并不会带来可见性能收益。

## 2026-06-06: true-cost profile 拆分 dispatch/epilogue/proxy

目标:

- 用户要求先拆真实耗时,再更新 `ep/docs/uccl_gin_perf_plan.md` 指明下一步方向。
- 本轮要回答两个问题:
  - README-like dispatch 的 `~2 ms` 主要是不是 copy epilogue?
  - proxy 侧 post/poll/software atomic 在真实 run 中占多少?

本地修改:

- `ep/include/proxy.hpp`
  - 新增 `profile_merge_opportunity_` 开关。
- `ep/src/proxy.cpp`
  - `UCCL_PROXY_PROFILE_COMMANDS=1` 只输出轻量 proxy counters。
  - 只有额外设置 `UCCL_PROXY_PROFILE_MERGE_OPPORTUNITY=1` 时才跑昂贵的
    per-batch merge-opportunity 扫描。
- 原因:
  - 上一轮 merge-opportunity profile 会把 proxy `seconds_avg` 从约 `4.8s` 增到
    `22.3s`,不能作为真实 proxy 成本。

服务器构建:

- 同步文件:
  - `ep/include/proxy.hpp`
  - `ep/src/proxy.cpp`
- build 命令里继续设置:
  - `CUDA_HOME=/usr/local/cuda-13.0`
  - `LIBRARY_PATH=/home/ubuntu/local-lib:$LIBRARY_PATH`
- build 日志:
  - `/tmp/uccl_gin_truecost_build.log`
- 结果:
  - `BUILD_RC:0`

Trace profile:

- 命令: README-like EP8x2 first-case,开启 PyTorch profiler trace 和
  `UCCL_PROXY_PROFILE_COMMANDS=1`,不设置
  `UCCL_PROXY_PROFILE_MERGE_OPPORTUNITY`。
- 日志:
  - `/tmp/uccl_gin_truecost_profile_rank0.log`
  - `/tmp/uccl_gin_truecost_profile_rank1.log`
- trace 目录:
  - `/tmp/uccl_gin_truecost_rank0_traces`
  - `/tmp/uccl_gin_truecost_rank1_traces`
- 本地临时解析目录:
  - `/tmp/uccl_truecost_traces_local/rank0`
  - `/tmp/uccl_truecost_traces_local/rank1`
- 结果:
  - 两端 `EXIT:0`。

Trace 汇总:

```text
dispatch:
  dispatch_impl                  avg 2049.748 us, p50 2050.634 us
  dispatch_copy_epilogue_impl     avg  306.963 us, p50  306.623 us

expanded_dispatch:
  dispatch_impl                  avg 2050.016 us, p50 2048.736 us
  dispatch_copy_epilogue_impl     avg  382.375 us, p50  385.119 us

cached_dispatch:
  dispatch_impl                  avg 2033.177 us, p50 2028.225 us
  dispatch_copy_epilogue_impl     avg  299.743 us, p50  295.599 us

combine:
  combine_impl                   avg 12798.610 us, p50 13772.467 us
  combine_reduce_epilogue_impl    avg    97.668 us, p50    97.728 us

reduced_combine:
  combine_impl                   avg 13140.050 us, p50 13913.870 us
  combine_reduce_epilogue_impl    avg    96.528 us, p50    96.544 us
```

解读:

- copy epilogue 不是 dispatch 主瓶颈:
  - first dispatch:copy epilogue 约 `0.31 ms`,dispatch kernel 约 `2.05 ms`。
  - expanded dispatch:copy epilogue 约 `0.38 ms`,dispatch kernel 约 `2.05 ms`。
  - cached dispatch:copy epilogue 约 `0.30 ms`,dispatch kernel 约 `2.03 ms`。
- 因此继续优化 dispatch 要优先看 `hybrid_dispatch` scaleout/control path,
  而不是先重写 copy epilogue。
- combine 更糟糕:
  - `combine_impl` 约 `12.8-13.1 ms`。
  - reduce epilogue 只有 `~0.1 ms`。
  - 如果目标是完整 EP 性能,combine 需要独立成为高优先级主线。
- `spin_kernel` 约 `10.1 ms`,是 profiler barrier/sleep,不是 dispatch 本体。
- 注意:trace run 会放大 proxy 生命周期时间,所以 proxy `seconds` 不能用这组
  trace 日志当真实吞吐判断。

No-trace 轻量 proxy profile:

- 命令: README-like EP8x2 first-case,只设置
  `UCCL_PROXY_PROFILE_COMMANDS=1`,不 dump trace,不设置
  `UCCL_PROXY_PROFILE_MERGE_OPPORTUNITY`。
- 日志:
  - `/tmp/uccl_gin_truecost_light_rank0.log`
  - `/tmp/uccl_gin_truecost_light_rank1.log`
- 结果:
  - 两端 `EXIT:0`。

README-like line 汇总:

```text
dispatch          n=48 avg 2042.958 us, min 2023 us, max 2065 us
expanded dispatch n=16 avg 2045.875 us, min 2027 us, max 2065 us
cached dispatch   n=16 avg 2039.750 us, min 2023 us, max 2056 us
combine           n=32 avg 13113.563 us, min 10603 us, max 16953 us
reduced combine   n=16 avg 13095.500 us, min 10603 us, max 16907 us
```

轻量 proxy counters 汇总 `64` 条 `UCCL_PROXY_PROFILE`:

```text
post_batches_sum                  909,803
post_cmds_sum                   1,194,368
write_cmds_sum                    956,288
write_bytes_sum           182,578,798,336
piggyback_atomic_write_cmds_sum   952,320
atomic_cmds_sum                   238,080
quiet_cmds_sum                          0
barrier_cmds_sum                        0
completed_wrs_sum            120,118,206
posted_atomic_wrs_sum             238,080
coalesced_atomic_wrs_sum                0
poll_us_sum                     5,836,513
progress_atomic_us_sum          1,340,001
post_gpu_us_sum                 4,872,288
seconds_avg                         4.998
write_GBps_sum                     36.529
cmds_per_sec_sum              238,973.256
```

按 proxy thread 汇总:

```text
thread 0: post_cmds 361,088, write_bytes 54.786 GB, atomic_cmds 71,424
thread 1: post_cmds 357,120, write_bytes 54.802 GB, atomic_cmds 71,424
thread 2: post_cmds 238,080, write_bytes 36.539 GB, atomic_cmds 47,616
thread 3: post_cmds 238,080, write_bytes 36.452 GB, atomic_cmds 47,616
```

结论:

- 轻量 profile 下 proxy 全生命周期约 `5.0s`,回到上一轮真实量级;昂贵的
  merge-opportunity 扫描已经被隔离到单独 env。
- proxy 线程存在明显负载不均:
  - thread 0/1 比 thread 2/3 多约 `50%` 命令和字节。
  - 这值得继续用 per-ring counter 拆,但不是下一步唯一瓶颈。
- proxy hot path 总 CPU 计时不是 `dispatch_impl ~2.05 ms/iter` 的全部来源:
  - `post_gpu_us + poll_us + progress_atomic_us` 是 64 个 proxy 线程整个 run 的累计值。
  - dispatch kernel 本身还在做 compact staging、scaleout emission、tail wait/forward 等
    GPU 侧工作。
- P4 adjacent WRITE merge 仍应下调/删除:
  - 真实轻量 profile 默认不跑 merge 扫描。
  - 上一轮 merge 机会只有 `0.038%` WR 可省,不值得进入主实现。
- 下一步方向:
  - dispatch:优先降低剩余 control/finish atomic 和 proxy thread imbalance,同时用
    per-ring counters 找出 thread 0/1 偏重来源。
  - combine:如果目标是完整 EP BW,必须单独 profile/优化 combine path;现在 combine
    kernel 是 `~13 ms` 级别,远大于 dispatch。
  - 不要先投大量工程到 copy epilogue 或 proxy adjacent WRITE merge。

## 2026-06-06: P0.5 dispatch_impl kernel clock profile instrumentation

背景:

- 用户指出当前只知道 `dispatch_impl ~2.05 ms`,但没有 scaleout/forward/notify
  的 kernel 内部分解。
- 这个判断是对的。没有 kernel 内部分解就直接推进 P1/P2,容易只在 proxy 侧做微调,
  但真实瓶颈可能在 GPU-side forward wait、D2H ring backpressure 或 compact/TMA
  staging。

最终本地修改:

- `ep/include/uccl_gin/resources.cuh`
  - 新增 `DispatchClockCounter` 枚举和 device helper:
    `dispatch_clock_add`, `dispatch_clock_inc`。
  - 没有改 `UCCLGinResources` ABI;profile counter 不再作为 `_C` resource 字段传入。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/buffers/elastic.py`
  - 新增 env 开关 `UCCL_GIN_DISPATCH_CLOCK_PROFILE=1`。
  - 开启时给 JIT flags 添加
    `-DDEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE`。
  - 不分配新的 Python tensor,避免需要 DeepEP `_C` ABI 同步。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - 仅在 `DEEPEP_USE_UCCL_GIN && DEEPEP_UCCL_GIN_DISPATCH_CLOCK_PROFILE` 下埋点。
  - counter buffer 从已有 UCCL atomic scratch 后半段切出来:
    `atomic_tail_base + kNumChannels * kNumScaleoutRanks`。
  - kernel start 清 counter。
  - 每个 thread 用本地 `profile_local[]` 累计,最后一次性 `atomicAdd`,避免在 hot loop
    里直接全局 atomic 导致 profile 自己把 kernel 拖慢。
  - kernel end 由 `sm_idx==0 && thread_idx==0` 打一行:
    `UCCL_GIN_DISPATCH_CLOCK rank=...`。

被撤回的第一版设计:

- 第一版曾把 `dispatch_profile_counters` 加进 `UCCLGinResources` /
  `NativeUCCLGinResources`,并由 Python 分配 tensor 后传给 `_C`。
- 这会要求 DeepEP `_C` 重新 build;如果只重编 `uccl.ep` 而不重编 `_C`,host/device ABI
  会错位,profile counter 会读到垃圾地址。
- 实际验证时出现过巨大 counter 和 correctness failure,因此改成“不改 `_C` ABI,
  复用 atomic scratch”的最终设计。

当前 counter:

```text
scaleout_preload_cycles/events
scaleout_compact_store_cycles/events
scaleout_local_store_cycles/events
scaleout_store_wait_cycles/events
scaleout_d2h_cycles/events
scaleout_tail_cycles/events
forward_tail_wait_cycles/events
forward_meta_wait_cycles/events
forward_load_cycles/events
forward_scaleup_store_cycles/events
forward_tokens
scaleout_remote_tokens
scaleout_local_tokens
```

使用方式:

```bash
export UCCL_GIN_DISPATCH_CLOCK_PROFILE=1
export UCCL_PROXY_PROFILE_COMMANDS=1
unset UCCL_PROXY_PROFILE_MERGE_OPPORTUNITY
```

预期分析:

- `forward_tail_wait_cycles` 大: receiver 等 tail/软件 atomic apply/control path。
- `forward_meta_wait_cycles` 大: tail 已到但 payload metadata 不 ready,需要查
  payload WR visibility/receiver ready protocol。
- `scaleout_d2h_cycles` 大: scaleout warp 在 D2H ring reserve/commit 或 proxy
  backpressure 上等。
- `scaleout_compact_store_cycles + scaleout_store_wait_cycles` 大: compact staging/TMA
  store 本身昂贵。
- 如果这些都不大,再继续加 notify/barrier/linked-list metadata 细分。

本地检查:

- `git diff --check` 通过。

服务器构建和恢复记录:

- `uccl.ep` 构建通过:
  - `/tmp/uccl_gin_clockprof_build2.log`
- 第一版 ABI 方案尝试重建 DeepEP `_C` 时遇到过:
  - `/tmp/deepep_c_clockprof_build.log`: 缺 `ninja`。
  - `/tmp/deepep_c_clockprof_build2.log`: include path 不完整,缺 `util/gpu_rt.h`。
  - `/tmp/deepep_c_clockprof_build3.log`: link 时找不到 `libnccl.so.2`。
  - `/tmp/deepep_c_clockprof_build4.log`: 构建成功。
- 注意事故:
  - 曾错误地用跨 host pipe 写同一个 EFS 路径上的 `_C.so`,导致共享 EFS 上的
    `deep_ep/_C.cpython-312-x86_64-linux-gnu.so` 被截断成 0 byte。
  - 已用 build artifact 恢复:
    `thirdparty/DeepEP-v2-d4f41e4/build/lib.linux-x86_64-cpython-312/deep_ep/_C.cpython-312-x86_64-linux-gnu.so`
    -> `thirdparty/DeepEP-v2-d4f41e4/deep_ep/_C.cpython-312-x86_64-linux-gnu.so`。
  - 之后两端 import smoke 均通过。
- 最终 clock profile 设计不再需要重建 DeepEP `_C`。

服务器验证:

- 命令要点:
  - `UCCL_GIN_DISPATCH_CLOCK_PROFILE=1`
  - `UCCL_PROXY_PROFILE_COMMANDS=1`
  - `unset UCCL_PROXY_PROFILE_MERGE_OPPORTUNITY`
  - 新 JIT cache:
    `/tmp/deepep_jit_uccl_clockprof4_rank0`,
    `/tmp/deepep_jit_uccl_clockprof4_rank1`
- 日志:
  - `/tmp/uccl_gin_clockprof4_rank0.log`
  - `/tmp/uccl_gin_clockprof4_rank1.log`
- 两端 `EXIT:0`。

clock profile 汇总:

```text
valid_clock_rows 2972

RANGE forward_tokens             8039..8079 avg 8058.7
RANGE scaleout_remote_tokens     4012..4044 avg 4029.9
RANGE scaleout_local_tokens      4012..4042 avg 4028.9
RANGE scaleout_d2h_events        160..160   avg 160.0
RANGE scaleout_tail_events       0..1488    avg 1477.6
RANGE forward_tail_wait_events   2728..2855 avg 2785.3
RANGE forward_meta_wait_events   8039..8079 avg 8058.7

CLOCK_SUM scaleout_preload        cycles   2,249,944,367 events 12,151,037 avg      185
CLOCK_SUM scaleout_compact_store  cycles     556,810,535 events 11,976,726 avg       46
CLOCK_SUM scaleout_local_store    cycles   4,132,732,663 events 11,973,830 avg      345
CLOCK_SUM scaleout_store_wait     cycles  24,233,431,680 events 12,151,037 avg    1,994
CLOCK_SUM scaleout_d2h            cycles  95,445,160,621 events    475,520 avg  200,717
CLOCK_SUM scaleout_tail           cycles 125,841,059,500 events  4,391,353 avg   28,657
CLOCK_SUM forward_tail_wait       cycles 415,739,220,474 events  8,278,026 avg   50,222
CLOCK_SUM forward_meta_wait       cycles  74,981,641,221 events 23,950,597 avg    3,131
CLOCK_SUM forward_load            cycles 603,343,600,978 events 23,950,597 avg   25,191
CLOCK_SUM forward_scaleup_store   cycles  36,921,791,263 events 62,917,507 avg      587
```

同一轮 README-like timing:

```text
dispatch          n=48 avg 3868.23 us, min 3465 us, max 4253 us
expanded dispatch n=16 avg 3866.75 us, min 3465 us, max 4253 us
cached dispatch   n=16 avg 3861.25 us, min 3473 us, max 4235 us
combine           n=32 avg 6976.25 us, min 5608 us, max 7457 us
reduced combine   n=16 avg 6971.00 us, min 5613 us, max 7450 us
```

同一轮 proxy 汇总:

```text
post_cmds             722,080
write_cmds            482,720
write_bytes    90,286,794,368
piggyback_atomic_write_cmds 478,720
atomic_cmds           239,360
quiet_cmds                  0
barrier_cmds                0
completed_wrs     19,671,190
posted_atomic_wrs    239,360
poll_us           12,265,479
progress_atomic_us 2,779,800
post_gpu_us        9,727,650
```

解读:

- profile instrumentation 会明显拖慢 dispatch:
  - no-profile `dispatch_impl ~2.04 ms`。
  - clock profile run `dispatch ~3.86 ms`。
  - 因此这组 counter 只能看结构性分布,不能直接当真实性能。
- `forward_meta_wait` 平均只有约 `3.1k cycles/event`,不像是 payload metadata ready
  等待主瓶颈。
- `scaleout_compact_store` 平均只有约 `46 cycles/event`,compact staging 写本身不是
  当前最大问题。
- `scaleout_d2h` 单次 command issue 约 `200k cycles/event`,说明 D2H ring/proxy
  backpressure 仍值得继续拆。
- aggregate 最大的是 `forward_load` 和 `forward_tail_wait`,但这些是许多 forward
  warp 并行累加的总和,不等同于 wall time critical path。
- 下一步 profile 应加 per-rank/per-channel 或 per-warp max counter,把 sum counter
  变成 critical-path counter;然后再决定 P1/P2 的优先级。

## 2026-06-06: P0.6 dispatch critical-path max counters

背景:

- 用户给了两个“不一定靠谱但值得验证”的判断:
  - `forward_load avg ~25k cycles` 对 14KB TMA load 偏大,可能不是纯 TMA 慢,
    而是在等 EFA/NIC DMA 写入 recv buffer 后对 GPU TMA load 可见。
  - `scaleout_d2h avg ~200k cycles` 需要知道是均匀慢,还是少数 ring/channel
    outlier;如果是后者,会直接指向 proxy thread/ring imbalance。
- 这个判断和 P0.5 的结论一致:sum counter 不能代表 wall-time critical path,
  必须补 max/per-channel 视角后再决定 P1/P2。

本地修改:

- `ep/include/uccl_gin/resources.cuh`
  - 新增 packed max counters:
    - `kDispatchClockScaleoutD2HMaxPacked`
    - `kDispatchClockForwardTailWaitMaxPacked`
    - `kDispatchClockForwardLoadMaxPacked`
  - 新增 helper:
    - `dispatch_clock_detail(channel, aux)`
    - `dispatch_clock_pack_max(cycles, detail)`
    - `dispatch_clock_max(...)`
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - 每个 thread 本地维护 `profile_max_local[]`,最后再全局 `atomicMax`,避免 hot loop
    里每个事件都打全局 atomic。
  - `scaleout_d2h_max_packed`:
    - `cycles`: 单次 `rail_put_tail_add` issue 的最长耗时。
    - `detail.channel`: `channel_idx`。
    - `detail.aux`: `queue = channel_idx % num_queues`。当前 4 proxy threads x
      8 rings/thread 时,`proxy_thread = queue / 8`。
  - `forward_tail_wait_max_packed`:
    - `cycles`: 单次 forward tail wait 最长耗时。
    - `detail.channel`: `channel_idx`。
    - `detail.aux`: `src_scaleout_rank`。
  - `forward_load_max_packed`:
    - `cycles`: 单次 forward TMA load 最长耗时。
    - `detail.channel`: `channel_idx`。
    - `detail.aux`: `slot_idx`。

packed decode:

```text
cycles  = packed >> 24
channel = (packed >> 12) & 0xfff
aux     = packed & 0xfff
```

预期分析:

- 如果 `scaleout_d2h_max` 固定集中在某几个 `aux/queue`,优先排查 P2 proxy/ring
  负载不均。
- 如果 `forward_load_max` 只是少数 channel/slot outlier,要查对应 tail/payload
  到达和 receiver scheduling;如果所有 channel 都稳定偏高,更像 EFA NIC->HBM
  visibility/latency 的结构性成本。
- 如果 `forward_tail_wait_max` 集中在某个 source rank,要查 receiver-side atomic apply
  或跨 node lane 分布。

待服务器验证:

- 同步代码后 rebuild `uccl.ep`。
- 使用新的 JIT cache 跑:
  - `UCCL_GIN_DISPATCH_CLOCK_PROFILE=1`
  - `UCCL_PROXY_PROFILE_COMMANDS=1`
  - `unset UCCL_PROXY_PROFILE_MERGE_OPPORTUNITY`
- 保存 rank 日志并 grep `UCCL_GIN_DISPATCH_CLOCK`,重点解析三个 `*_max_packed` 字段。

第一轮服务器验证:

- build: `/tmp/uccl_gin_p06_build.log`, `BUILD_RC:0`。
- 日志:
  - `/tmp/uccl_gin_p06_rank0.log`
  - `/tmp/uccl_gin_p06_rank1.log`
- 两端 `EXIT:0`。
- 但本轮 profile overhead 过高:
  - dispatch 从 P0.5 的 `~3.86 ms` 增到 `~5.64 ms`。
  - 原因很可能是每个 thread 维护了
    `profile_max_local[kDispatchClockNumCounters]`,额外 20+ 个 `uint64_t` 局部变量带来
    寄存器压力。
- 已修复:
  - 把 max 本地状态收缩成 3 个变量:
    `profile_scaleout_d2h_max`, `profile_forward_tail_wait_max`,
    `profile_forward_load_max`。
  - 已用新 JIT cache 重跑。

第二轮服务器验证:

- 日志:
  - `/tmp/uccl_gin_p06b_rank0.log`
  - `/tmp/uccl_gin_p06b_rank1.log`
- 两端 `EXIT:0`。
- 注意:
  - dispatch 仍约 `5.66 ms`,所以 P0.6 max/clock instrumentation 本身会显著扰动
    kernel;这组结果只用于判断结构和 outlier,不能当真实性能。
  - 相比 P0.5 `~3.86 ms`,P0.6 多出来的 max/clock 采样本身就是主要 overhead,不是
    单纯 `profile_max_local[]` 数组造成。

P0.6 聚合:

```text
clock_rows 2964

AVG_ROW scaleout_d2h_cycles         avg 153,879 cycles/event, min 65,706, max 313,485
AVG_ROW forward_tail_wait_cycles    avg  51,598 cycles/event, min  2,143, max 10,981,593
AVG_ROW forward_load_cycles         avg  15,434 cycles/event, min  2,355, max 24,083
AVG_ROW forward_meta_wait_cycles    avg   2,380 cycles/event, min  1,158, max  3,995

scaleout_d2h_max_packed:
  max 3,723,190 cycles @ channel 36, queue 4, proxy_thread 0, rank 5
  per-row max p50 1,080,503 cycles, avg 1,121,408 cycles
  queue top counts: q14=159, q2=133, q1=129, q4=126, q3=120
  proxy_thread counts: t0=917, t1=830, t2=609, t3=608

forward_tail_wait_max_packed:
  max 768,095,364 cycles @ channel 13, src_scaleout_rank 0, rank 9
  per-row max p50 1,808,368 cycles, avg 3,966,857 cycles
  src counts: src0=1579, src1=1385

forward_load_max_packed:
  max 2,567,053 cycles @ channel 19, slot 60, rank 6
  per-row max p50 813,073 cycles, avg 850,842 cycles
```

proxy 汇总:

```text
post_cmds                  1,194,368
write_cmds                   956,288
write_bytes          182,578,798,336
piggyback_atomic_write_cmds  952,320
atomic_cmds                  238,080
completed_wrs             56,908,368
poll_us                  11,352,919
progress_atomic_us        2,618,879
post_gpu_us               9,455,853

thread0: post_cmds 361,088, write_bytes 54.786 GB, atomic_cmds 71,424
thread1: post_cmds 357,120, write_bytes 54.802 GB, atomic_cmds 71,424
thread2: post_cmds 238,080, write_bytes 36.539 GB, atomic_cmds 47,616
thread3: post_cmds 238,080, write_bytes 36.452 GB, atomic_cmds 47,616
```

解读:

- `scaleout_d2h`:
  - max 不集中在单个 queue,但 queue/proxy thread 分布仍和 proxy command 负载一致:
    thread0/1 明显重于 thread2/3。
  - 这继续支持 P2:先修 channel->queue/proxy 映射负载不均,但不是“某一个坏 ring”。
- `forward_load`:
  - max 分布在多个 channel/slot,slot top counts 也分散。
  - 这更像 NIC->HBM visibility / receiver-side data availability 的普遍成本或
    profile扰动下的 TMA load 等待,不是单个 channel/slot 的 bug。
- `forward_tail_wait`:
  - 有少数极大 outlier,主要在 rank1 等 src0,可能包含启动 skew、跨节点第一次到达
    或非 timed warmup 的等待。
  - 不能直接把这些极大值当成 steady-state 瓶颈;下一步如果继续 profile,应过滤
    warmup/first iteration,或只在 timed iterations 开关 counter。

下一步建议:

- 性能工程上先做 P2 queue/proxy balance:
  - 当前 thread0/1 仍比 thread2/3 多约 `50%` 命令/字节。
  - `scaleout_d2h` max 的 proxy_thread counts 也偏向 thread0/1。
- 同时改 profile 方法:
  - P0.6 当前会把 dispatch 从真实 `~2.0 ms` 扰到 `~5.6 ms`。
  - 后续只保留 sampled/max 版本,或让 counter 只在少量 timed iteration 打开。

## 2026-06-06 P0.7 D2H ring 背压假设验证

问题:

- P0.6 看到 `scaleout_d2h` 平均约 `154k cycles/event`,max p50 约 `1.08M`
  cycles,怀疑 GPU scaleout warp 在 D2H ring reserve/commit 等 proxy 腾 slot。
- 尝试两个便宜实验:
  1. `NUM_PROXY_THREADS=4 -> 8`
  2. `kQueueSize=2048 -> 4096`

### 实验 A: 8 proxy threads

构建:

```bash
make -C ep clean
make -C ep install NUM_PROXY_THREADS=8 CHANNEL_PER_PROXY=8 CUDA_PATH=/usr/local/cuda-13.0
```

验证:

```text
p5en_0: import uccl.ep; get_num_proxy_threads()=8, d2h_queue_capacity()=2048
p5en_1: import uccl.ep; get_num_proxy_threads()=8, d2h_queue_capacity()=2048
```

日志:

- no-clock:
  - `/tmp/uccl_gin_np8_rank0.log`
  - `/tmp/uccl_gin_np8_rank1.log`
- clock:
  - `/tmp/uccl_gin_np8_p06_rank0.log`
  - `/tmp/uccl_gin_np8_p06_rank1.log`

no-clock 结果:

```text
dispatch: 30-31 GB/s (SO), 1997-2018 us
expanded dispatch: 30-31 GB/s (SO), 2001-2033 us
```

proxy 聚合:

```text
post_cmds                  1,194,368
write_cmds                   956,288
write_bytes          182,578,798,336
atomic_cmds                  238,080

thread0: 36.587 GB, post_cmds 242,048
thread1: 36.603 GB, post_cmds 238,080
thread2: 18.340 GB, post_cmds 119,040
thread3: 18.245 GB, post_cmds 119,040
thread4: 18.199 GB, post_cmds 119,040
thread5: 18.199 GB, post_cmds 119,040
thread6: 18.199 GB, post_cmds 119,040
thread7: 18.207 GB, post_cmds 119,040
```

clock 结果:

```text
clock_rows 2947
scaleout_d2h_cycles      avg 178,097 cycles/event, p50 170,664
forward_tail_wait_cycles avg  74,298 cycles/event
forward_load_cycles      avg  14,921 cycles/event

scaleout_d2h_max_packed:
  max 3,951,780 cycles @ channel 21, queue 21, proxy_thread 2
  per-row max p50 1,154,197 cycles, avg 1,309,087 cycles
  proxy_thread counts: t0=511, t1=574, t2=353, t3=257, t4=431, t5=240, t6=242, t7=339
```

解读:

- 8 proxy threads 没改善 wall time;dispatch 仍约 `2.0 ms`。
- clock profile 下 `scaleout_d2h avg` 反而高于 4-thread baseline。
- 单纯增加 proxy thread 不是当前主线;thread0/1 仍更重,总 poll/post overhead 也增加。

### 实验 B: queue size 4096

代码:

- `ep/include/common.hpp` 新增 `UCCL_QUEUE_SIZE`,默认 `2048`。
- `ep/Makefile` 新增 `QUEUE_SIZE ?= 2048`,并传入 `-DUCCL_QUEUE_SIZE=$(QUEUE_SIZE)`。

构建:

```bash
make -C ep clean
make -C ep install NUM_PROXY_THREADS=4 CHANNEL_PER_PROXY=8 QUEUE_SIZE=4096 CUDA_PATH=/usr/local/cuda-13.0
```

验证:

```text
p5en_0: import uccl.ep; get_num_proxy_threads()=4, d2h_queue_capacity()=4096
p5en_1: import uccl.ep; get_num_proxy_threads()=4, d2h_queue_capacity()=4096
```

日志:

- build: `/tmp/uccl_gin_q4096_build.log`
- no-clock:
  - `/tmp/uccl_gin_q4096_rank0.log`
  - `/tmp/uccl_gin_q4096_rank1.log`
- clock:
  - `/tmp/uccl_gin_q4096_p06_rank0.log`
  - `/tmp/uccl_gin_q4096_p06_rank1.log`

no-clock 结果:

```text
dispatch: 29-30 GB/s (SO), 2069-2079 us
expanded dispatch: 30 GB/s (SO), 2043-2055 us
```

clock 结果:

```text
clock_rows 2964
scaleout_d2h_cycles      avg 153,038 cycles/event, p50 151,578
forward_tail_wait_cycles avg  36,070 cycles/event
forward_load_cycles      avg  14,263 cycles/event

scaleout_d2h_max_packed:
  max 3,934,454 cycles @ channel 40, queue 8, proxy_thread 1
  per-row max p50 1,043,145 cycles, avg 1,119,447 cycles
  proxy_thread counts: t0=968, t1=864, t2=626, t3=506
```

解读:

- 4096-slot ring 没有改善真实 dispatch,反而略慢。
- `scaleout_d2h avg` 和 2048 baseline 基本相同,说明 D2H ring capacity 不是当前
  主要瓶颈。
- `scaleout_d2h` 的高等待更可能来自 proxy 服务速率、WR/CQE 处理、receiver data
  visibility 或 emission/control pattern 共同形成的 backpressure。

结论:

- 暂时不继续做 8192 ring sweep。
- P2 仍应做,但重点从“更大 ring / 更多 thread”转为:
  - 改 channel -> queue/proxy 映射均衡;
  - 减少每个 token/batch 的 D2H command 数;
  - 降低 proxy WR/CQE 处理量;
  - 继续拆 receiver-side wait 和 proxy-side service rate。

远端状态恢复:

- 恢复构建日志: `/tmp/uccl_gin_restore_q2048_build.log`
- 已把 `ep.abi3.so` 从 p5en_0 复制到 p5en_1。
- 验证:

```text
p5en_0: get_num_proxy_threads()=4, d2h_queue_capacity()=2048
p5en_1: get_num_proxy_threads()=4, d2h_queue_capacity()=2048
```

## 2026-06-06: 根据 V1 baseline 重写 UCCL-GIN perf plan

目标:

- 用户指出当前 V2 UCCL-GIN 的 `ring_buffer.cuh::atomic_set_and_commit` 只在
  `head - tail == Capacity` 时背压,而 V1 `uccl_ibgda.cuh` 在 normal path 里先用
  `kMaxInflightNormal=8` 做 completion-credit 节流。
- 结合远端 V1 baseline 数据,重写 `ep/docs/uccl_gin_perf_plan.md`,把优化重点从
  “更大 ring / 更多 proxy thread / 零散 profile” 转到 “恢复 V1 式 D2H inflight cap
  和继续减少 command/WR/CQE 数量”。

已确认事实:

```text
V1 UCCL-EP @ 495b722:
  FP8 dispatch: ~50-52 GB/s RDMA
  D2H put/atomic push avg: ~38k cycles/event

当前 V2 UCCL-GIN:
  dispatch: ~30 GB/s RDMA
  scaleout_d2h avg: ~153k cycles/event
```

代码语义核对:

- 当前非 FIFO EFA path 的 D2H `tail` 是 CQ completion/ack 语义:
  - `poll_cq_* -> acked_wrs_`
  - `notify_gpu_completion -> mark_acked -> advance_tail_from_mask`
  - `advance_tail_from_mask` 通过 release store 发布 `tail`
- 因此在当前 path 中做 V1 式 `head - tail < max_inflight` cap 是有意义的;
  它限制的是未完成 command/WR,不是单纯限制 proxy 是否读过命令。

文档改动:

- 重写 `ep/docs/uccl_gin_perf_plan.md`。
- 新优先级:
  1. P1: 实现 bounded D2H push,对 `4/8/16/32/64/128/2048` 做 cap sweep。
  2. P2: 保留 V1-style quiet/ack,但禁止 hot path per-tail 同步 drain。
  3. P3: 继续减少 command/WR/CQE,尤其 finish/control ATOMIC 和 compact32 实际粒度。
  4. P4: 做 per-ring/profile,修 channel -> queue/proxy 映射偏斜。
  5. P5: P1/P3 后再复测 receiver wait/data visibility。
  6. P6: combine 单独立项。

当前判断:

- `QUEUE_SIZE=4096` 和 `NUM_PROXY_THREADS=8` 已证实不是主线。
- 深 ring 只是 host queue 排队深度,不是 NIC in-flight 深度。
- 短期最值得做的是把 V2 UCCL-GIN Rail helper 的 push policy 改回 V1 的
  completion-credit 模型。

## 2026-06-06: 修正 perf plan 优先级,前置 chunk/proxy 诊断

用户 review 指出上一版 plan 仍然过早押注 inflight cap:

- 当前 plan 中已有 `write_bytes` 和 `write_cmds`:

```text
182,578,798,336 / 956,288 ~= 190,923 bytes ~= 186 KiB per WRITE
```

按每 token payload `7.5-14 KiB` 粗估,实际平均 chunk 约 `13-25 tokens`,不是
compact32 的 32 tokens。由于 `piggyback/write ~= 0.996`,大多数 WRITE 都是 compact
chunk,所以 chunk 粒度不足不是“待确认小项”,而是高优先级根因候选。

- V2 `scaleout_d2h ~153k cycles/event` 可能只是 proxy 跟不上导致 GPU 等 tail;
  inflight cap 可能只把等待点从 ring full 挪到 cap wait,不一定提高 wall time。
  因此需要先比较 V1 vs V2 proxy per-command cost。

- V1 baseline 是 4096 tokens,V2 profile 是 8192 tokens;不能直接用 `50 vs 30 GB/s`
  当 apples-to-apples 差距。

文档更新:

- 再次重写 `ep/docs/uccl_gin_perf_plan.md`。
- 新优先级:
  0. 跑 V1 8192-token apples-to-apples baseline。
  1. P1A: `rail_put_tail_add count_delta` / chunk / flush reason profile。
  2. P1B: V1 vs V2 proxy `ns/command`, `ns/WR`, `ns/CQE`, `ns/update` profile。
  3. 根据 P1A/P1B 决定:
     - chunk 太小 -> 修 compact batching。
     - proxy per-command 太慢 -> 修 proxy fast path / 减 hot-path bookkeeping。
     - 两者都不是 -> 再做 inflight cap sweep 或 receiver wait。
  4. inflight cap 降级为 P2,作为排队形态实验,不再作为先验最高优先级。

当前判断:

- compact32 是否真正达到 32-token chunk 必须马上 profile。
- V2 proxy per-command 成本是否显著高于 V1 必须马上 profile。
- cap 仍值得做,但应在上述诊断之后做,并用 wall time/BW 判断,不能只看 per-push
  cycles 是否接近 V1。

## 2026-06-06: 完成 P0/P1A/P1B 诊断

本轮目标:

0. 跑 V1 `8192` token apples-to-apples baseline。
1. 给 V2 UCCL-GIN compact dispatch 增加 `count_delta/chunk/flush reason`
   profile。
2. 对比 V1 vs V2 proxy `ns/command`。

### 0. V1 8192-token baseline

远端 worktree:

```text
/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang-v1-baseline-495b722
commit 495b7221d084cce92553d6a038376358bd218a5a
```

普通 build 日志:

```text
/tmp/uccl_v1_495_8192_build.log
```

运行日志:

```text
/tmp/uccl_v1_495_ep16_8192_rank0.log
/tmp/uccl_v1_495_ep16_8192_rank1.log
```

命令形状:

```bash
torchrun --nnodes=2 --nproc_per_node=8 \
  bench/test_internode.py --num-processes=8 \
  --num-tokens=8192 --hidden=7168 --num-topk=8 --num-experts=256
```

结果:

```text
rank0 FP8 dispatch: 2035 us, 59.30 GB/s RDMA
rank1 FP8 dispatch: 2054 us, 58.81 GB/s RDMA

rank0 BF16 dispatch: 3419 us, 68.45 GB/s RDMA
rank1 BF16 dispatch: 3482 us, 67.28 GB/s RDMA

rank0 combine: 11557 us, 20.25 GB/s RDMA
rank1 combine: 11531 us, 20.32 GB/s RDMA
```

结论:

- V1 8192-token apples-to-apples dispatch baseline 是 `~59 GB/s FP8` 和
  `~67-68 GB/s BF16`,不是之前 4096-token 下的 `~50 GB/s`。
- 当前 V2 UCCL-GIN dispatch `~24-30 GB/s` 仍然明显低于 V1,但目标差距应按
  `59 GB/s` 量级来理解。

### 1. V2 count_delta / chunk / flush reason profile

代码改动:

- `ep/include/uccl_gin/resources.cuh`
  - 新增 `DispatchChunkCounter` 和 `DispatchChunkFlushReason`。
  - 新增 device helper: `dispatch_chunk_add`,
    `dispatch_chunk_size_bin`, `dispatch_chunk_reason_counter`。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/buffers/elastic.py`
  - 新增环境变量 `UCCL_GIN_CHUNK_PROFILE=1` 时自动追加
    `-DDEEPEP_UCCL_GIN_CHUNK_PROFILE`。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - 在 `flush_compact_remote_batch` 里统计实际 `compact_batch_count`。
  - flush reason 分三类:
    - `noncontig`: compact slot 不连续导致 flush。
    - `full`: 达到 `kUCCLGinCompactChunkTokens=32`。
    - `finish`: channel 结束时 flush partial chunk。

build 日志:

```text
/tmp/uccl_gin_chunkprof_build.log
```

运行日志:

```text
/tmp/uccl_gin_chunk_proxy_rank0.log
/tmp/uccl_gin_chunk_proxy_rank1.log
```

运行环境新增:

```bash
export UCCL_GIN_CHUNK_PROFILE=1
export UCCL_PROXY_PROFILE_COMMANDS=1
export EP_JIT_CACHE_DIR=/tmp/deepep_jit_uccl_chunkprof
```

聚合结果:

```text
chunk profile lines: 2976
avg chunk size: mean 25.49 tokens, median 25.49 tokens
min/max avg per line: 25.46 / 25.53 tokens
full chunk fraction: 75.0%
noncontig flush total: 0
```

典型单行:

```text
UCCL_GIN_CHUNK_PROFILE rank=7 chunks=320 tokens=8161
  bin_3_4=3 bin_5_8=77 bin_32=240
  flush_noncontig=0 flush_full=240 flush_finish=80
```

解释:

- 每 rank 每次 dispatch 基本是 `320` 个 remote payload chunk。
- 其中 `240` 个是满 `32-token` chunk,`80` 个是 channel 末尾的 `5-8 token`
  partial chunk。
- `flush_noncontig=0`,说明当前 compact staging 没有被 slot 不连续打断。
- `write_bytes/write_cmds ~= 191 KB/WRITE` 与 chunk profile 一致:
  `25.5 tokens * ~7.5 KB/token ~= 191 KB`。

结论:

- compact32 已经真正工作。之前“平均 chunk 只有 13-25 token,可能没凑满 32”
  的担心需要修正:平均 25.5 是 `3 个满 32 + 1 个尾部 partial` 的 channel
  粒度结果,不是中途 flush。
- 如果继续做 payload batching,方向不是修复 non-contiguous flush,而是:
  - 增大 per-channel chunk 上限并保证 receiver tail 语义仍成立;
  - 或者做跨 channel / 跨 proxy queue 的更高层聚合。但这会明显改变 V2 kernel
    结构,需要单独设计。

### 2. V1 vs V2 proxy ns/command profile

V2 使用现有 `UCCL_PROXY_PROFILE_COMMANDS=1`。V1 在远端临时加入同格式的
`UCCL_V1_PROXY_PROFILE` 输出,只记录:

```text
post_batches, post_cmds, write_cmds, write_bytes, atomic_cmds,
completed_wrs, poll_us, post_gpu_us
```

V1 profile 运行日志:

```text
/tmp/uccl_v1_495_proxyprof_build.log
/tmp/uccl_v1_495_proxyprof_8192_rank0.log
/tmp/uccl_v1_495_proxyprof_8192_rank1.log
```

注意:

- V1 profile build 会明显拖慢 tuning,因此 V1 bandwidth baseline 仍使用上面的普通
  build 结果。
- V1 proxy profile 源码改动已经从远端 worktree 恢复;profile `.so` 保存在
  `/tmp/uccl_ep_v1_495_proxyprof.abi3.so`。
- 当前两台 venv 里的 `uccl/ep.abi3.so` 已恢复为 V2 build:

```text
p5en_0: get_num_proxy_threads=4, d2h_queue_capacity=2048
p5en_1: get_num_proxy_threads=4, d2h_queue_capacity=2048
```

聚合结果:

```text
V1 proxy profile lines: 62
V1 post_gpu_ns_per_cmd:
  mean 166 ns, median 167 ns, min 144 ns, max 211 ns
V1 poll_ns_per_completed:
  mean 127 ns, median 127 ns, min 116 ns, max 146 ns
V1 write_bytes_per_cmd:
  mean 144,026 bytes, median 144,000 bytes
V1 post_cmds per proxy:
  median 3,581,701 commands

V2 proxy profile lines: 64
V2 post_gpu_ns_per_cmd:
  mean 7,949 ns, median 8,351 ns, min 5,467 ns, max 15,054 ns
V2 poll_ns_per_completed:
  mean 264 ns, median 191 ns, min 59 ns, max 1,362 ns
V2 progress_atomic_ns_per_atomic:
  mean 11,181 ns, median 11,192 ns, min 7,864 ns, max 17,689 ns
V2 write_bytes_per_cmd:
  mean 191,045 bytes, median 191,564 bytes
```

直接结论:

- V2 的 payload WRITE size 甚至比 V1 更大:
  - V1 `~144 KB/WRITE`
  - V2 `~191 KB/WRITE`
- 因此当前主瓶颈不再是 payload chunk 太小。
- V2 proxy `post_gpu` hot path 单 command 成本约 `8 us`,比 V1 `~0.17 us`
  慢约 `48x`。
- V2 `progress_atomic` 仍有约 `11 us/atomic` 的成本。即使 piggyback 已经减少独立
  tail WR,软件 atomic/apply bookkeeping 仍很重。

下一步方向:

1. 先不要优先做 D2H inflight cap。cap 可能改变 GPU 等待形态,但不会让 proxy
   每 command 从 `8 us` 变回 V1 的 `~0.17 us`。
2. 优先查 V2 `post_gpu_commands_mixed` 相比 V1 的 per-command 开销:
   - `PendingAtomicBatch` / `atomic_dep_by_wr_` / `retire_inflight_write`
   - piggyback atomic decode/apply 依赖追踪
   - profile/merge/coalesce 检查是否仍在 hot path 造成分支和容器操作
   - `std::vector`/`unordered_map`/`set` 是否进入 per-command 关键路径
3. 保留 chunk profile 代码作为 gated 诊断工具,默认不开。

## 2026-06-06: 批判性核对 dependency-tracking review

外部 review 提出:

- piggyback WRITE 已经把 payload + count 放入同一个 WRITE_WITH_IMM,所以不应继续为
  每条 piggyback WRITE 支付 standalone ATOMIC dependency-tracking 成本。
- `atomic_dependency_wrs_` 可能无界增长,导致每个 finish ATOMIC 做 O(N) 扫描。
- 可以进一步把 finish piggyback 到最后 payload chunk,删除独立 finish ATOMIC。

代码核对结果:

- 第一条部分成立:
  - `flush_writes()` 当前会把所有 WRITE 加入 `inflight_write_wrs_` 和
    `atomic_dependency_wrs_`。
  - 后续 standalone finish ATOMIC 会扫描这些依赖并写入 `atomic_dep_by_wr_`。
  - 因此 piggyback WRITE 确实仍承担后续 finish ordering 的 per-WR bookkeeping。
- 第二条不成立:
  - `enqueue_pending_atomics()` 末尾会调用 `deps.clear()`。
  - dependency vector 是“自上一个 standalone ATOMIC 以来的 WRITE”,不是全历史无界
    vector。
- 第三条方向正确但不能直接删除:
  - 当前 piggyback count 只表示该 payload chunk 的 token 数。
  - 发送 chunk 时尚未表达“这是 channel 最后一个 chunk”。
  - standalone finish 仍负责终止语义;只有设计 last-payload finish piggyback 后才能
    删除相应 dependency。

计划调整:

- 更新 `ep/docs/uccl_gin_perf_plan.md`。
- 下一步先细分 dependency hot path:
  - 每 finish 的 dependency fan-in。
  - dependency scan/map insert/map erase/retire/coalesce 各自耗时。
  - finish ATOMIC 与其他 ATOMIC 数量。
- 根据 profile 决定:
  - 用 per-ring/per-channel completion watermark 替代 per-WR unordered_map;
  - 或实现 last-payload finish piggyback;
  - 或先复用 V1 精简 WRITE post path。

## 2026-06-07: dependency 成本核对与 finish-piggyback 实验

### 批判性核对外部 review

外部 review 正确指出:

- 当前 piggyback payload WRITE 仍进入 `atomic_dependency_wrs_`,供后续 standalone
  finish ATOMIC 建 completion dependency。
- 旧 profile 的 `post_gpu_us` 不能继续泛泛地当作一个整体,需要拆出真正处理非空
  command batch 的 active cost。

但以下推断不成立或尚未被数据支持:

- `atomic_dependency_wrs_` 不会增长到全 dispatch 历史。每次
  `enqueue_pending_atomics()` 都会 `deps.clear()`。
- `post_gpu_us/post_cmds ~= 8 us` 不是 active per-command post cost。`post_gpu_us`
  包含大量没有 command 的 proxy loop 时间,与 V1 的计时口径不对称。
- dependency bookkeeping 是主瓶颈只是一个假设,不能直接据此删除 ordering。

### 提交当前已验证 profiling 基线

本地提交:

```text
25fc179b Profile UCCL-GIN dispatch bottlenecks
```

### 实验:最后一个 payload 同时 piggyback finish

实验性实现:

- 最后一个非空 compact payload WRITE 的 immediate value 同时携带 `count + finish`。
- 空 channel 保留 standalone finish ATOMIC。
- piggyback WRITE 不再进入 standalone atomic dependency list。

实验结果:

- 构建通过:
  - `/tmp/uccl_gin_finish_piggyback_build.log`
- 运行通过:
  - `/tmp/uccl_gin_finish_piggyback_rank0.log`
  - `/tmp/uccl_gin_finish_piggyback_rank1.log`
- `atomic_cmds=0`,dependency 计数归零,证明行为确实生效。
- 该实验没有证明 headline BW 收益,并把 finish 可见性绑定到最后一个大 payload
  WRITE 的完成/receiver apply。已撤销行为改动,不提交该路径。

### 新增 active/dependency profile

保留的 profiling 字段:

```text
mixed_ns
dependency_scan_ns
dependency_candidates
dependency_active
dependency_max
merge_profile_enabled
```

构建日志:

```text
/tmp/uccl_gin_dependency_profile_build.log
```

clock + proxy profile 日志:

```text
/tmp/uccl_gin_dependency_profile_rank0.log
/tmp/uccl_gin_dependency_profile_rank1.log
```

代表性 `rank0/thread0`:

```text
post_cmds=22568
mixed_ns=14027030                 # 约 622 ns/command
dependency_scan_ns=795170
dependency_candidates=18104       # 约 43.9 ns/candidate
dependency_active=11399
dependency_max=72
progress_atomic_us=45328
```

跨线程观察:

- active mixed path 约 `0.55-0.75 us/command`。
- dependency max fan-in 约 `48-93`。
- dependency scan 约 `35-50 ns/candidate`,每线程整个测试累计约 `0.4-0.8 ms`。
- `progress_pending_atomics` 累计约 `35-50 ms / 23 s`;虽然单 atomic 平均值看起来高,
  总量不足以解释 dispatch kernel 的毫秒级差距。

结论:

- dependency vector/container CPU 扫描不是当前主瓶颈。
- 旧的“V2 active post 比 V1 慢 48x”结论来自不对称计时,需要撤销。
- 下一步应核对同一 tail-word 的 receiver sequence/reorder 是否足以保证 finish 顺序,
  并重点看 receiver tail wait/load critical path。

### profiling 开销核对

关闭所有 profiling 的恢复基线:

```text
/tmp/uccl_gin_restored_noprofile_rank0.log
/tmp/uccl_gin_restored_noprofile_rank1.log

cached dispatch: 30-31 GB/s (SO), 2.00-2.02 ms
```

打开 kernel clock profile:

```text
cached dispatch: 11-14 GB/s (SO), 4.4-5.6 ms
```

因此 kernel clock profile 是侵入式诊断工具,只能比较 counter 内部相对分布,不能拿其
headline BW 评价优化是否回归。

## 2026-06-07:用 receiver sequence 收窄 finish dependency

### 设计依据

继续核对 transport 后确认:

- compact payload `WRITE_WITH_IMM` 在 `post_rdma_async_batched_normal_mode()` 中按
  `(dst_rank, tail_index)` 从 `next_seq_per_index` 分配 sequence。
- standalone ordered finish ATOMIC 在 `post_atomic_operations()` 中使用同一个
  `(dst_rank, tail_index)` sequence 空间。
- receiver `remote_process_completions_normal_mode()` 的 `SeqBuf` 会缓存乱序到达的
  delta,只按 sequence 连续 apply。
- payload count 的 CQE 与 payload WRITE 到达绑定。因此 receiver apply finish 前,
  同 tail-word 的 payload count 及其 payload 已经到达。

所以 sender 不必等待带 ordered piggyback count 的 payload WRITE CQE。仍需要保留
plain WRITE dependency,因为 plain WRITE 没有进入该 tail-word sequence。

### 代码改动

`ep/src/proxy.cpp::flush_writes()`:

- 所有 WRITE 仍进入 `inflight_write_wrs_`,保留 quiet/completion accounting。
- 只有 `atomic_val == 0` 的 plain WRITE 进入 `atomic_dependency_wrs_`。
- standalone finish 语义和 receiver reorder 协议不变。

### 构建与验证

构建日志:

```text
/tmp/uccl_gin_ordered_finish_build.log
```

带完整 correctness check:

```text
/tmp/uccl_gin_ordered_finish_rank0.log
/tmp/uccl_gin_ordered_finish_rank1.log
```

结果:

```text
EXIT:0 on both nodes
cached dispatch: 31-32 GB/s (SO), 1.93-2.00 ms
```

对照恢复基线:

```text
/tmp/uccl_gin_restored_noprofile_rank0.log
/tmp/uccl_gin_restored_noprofile_rank1.log
cached dispatch: 30-31 GB/s (SO), 2.00-2.02 ms
```

轻量 proxy profile:

```text
/tmp/uccl_gin_ordered_finish_profile_rank0.log
/tmp/uccl_gin_ordered_finish_profile_rank1.log
```

代表性 `rank0/thread0` 前后:

```text
                         before       after
dependency_candidates    18104        248
dependency_active        11399        63
dependency_max           72           2
progress_atomic_us       45328        19360
```

结论:

- 收窄 dependency 正确且有约 `2-4%` 性能收益。
- finish dependency 不是剩余 V1/V2 gap 的主因。
- 下一步应集中在 receiver WRITE_WITH_IMM CQE/reorder/apply 和 forward
  tail/load critical path,而不是继续微调 dependency container。

## 2026-06-07: receiver sequence 状态优化与 compact chunk sweep

### receiver ordered atomic profile

新增轻量 profile 字段:

```text
receiver_atomic_cqes
receiver_atomic_in_order
receiver_atomic_buffered
receiver_atomic_drained
receiver_atomic_max_buffered
```

仅在 `UCCL_PROXY_PROFILE_COMMANDS=1` 时计数。

32-token profile 日志:

```text
/tmp/uccl_gin_receiver_array_profile_rank0.log
/tmp/uccl_gin_receiver_array_profile_rank1.log
```

代表性数据:

```text
rank0/thread0:
receiver_atomic_cqes=22320
receiver_atomic_in_order=20438
receiver_atomic_buffered=1882
receiver_atomic_drained=1882
receiver_atomic_max_buffered=2
```

约 `92%` ordered atomic CQE 直接按序到达,乱序深度很浅。因此 reorder lookup 不是
当前主瓶颈。

### sequence 状态固定数组

代码改动:

- receiver `thread_local unordered_map<size_t, SeqBuf>` 改为
  `ProxyCtx::ordered_atomic_seqbufs[1024]`。
- sender `next_seq_per_index unordered_map` 改为 `ProxyCtx` 内 1024 项数组。
- 1024 来自 ordered atomic immediate 的 13-bit byte offset 和 8-byte tail word:
  `(1 << 13) / 8`。
- sequence/reorder/apply 语义不变。

构建日志:

```text
/tmp/uccl_gin_receiver_array_build.log
/tmp/uccl_gin_seq_array_build.log
```

### compact chunk sweep

所有配置均运行完整 correctness check,两节点均 `EXIT:0`:

```text
chunk  logs                                  cached dispatch
64     /tmp/uccl_gin_chunk64_rank{0,1}.log  ~27 GB/s, 2.23-2.31 ms
16     /tmp/uccl_gin_chunk16_rank{0,1}.log  ~33-34 GB/s, 1.79-1.83 ms
8      /tmp/uccl_gin_chunk8_rank{0,1}.log   ~35-36 GB/s, 1.70-1.73 ms
4      /tmp/uccl_gin_chunk4_rank{0,1}.log   ~37-38 GB/s, 1.59-1.64 ms
2      /tmp/uccl_gin_chunk2_rank{0,1}.log   ~32 GB/s, 1.90-1.93 ms
```

保留配置:

```text
kUCCLGinCompactChunkTokens = 4
```

4-token + sender/receiver sequence array 最终验证:

```text
/tmp/uccl_gin_seq_array_rank0.log
/tmp/uccl_gin_seq_array_rank1.log

cached dispatch: ~38 GB/s (SO), 1.60-1.63 ms
```

4-token profile:

```text
/tmp/uccl_gin_chunk4_profile_rank0.log
/tmp/uccl_gin_chunk4_profile_rank1.log

rank0/thread0:
receiver_atomic_cqes=120528
receiver_atomic_in_order=115181
receiver_atomic_buffered=5347
receiver_atomic_max_buffered=5
```

结论:

- 虽然 4-token 让 receiver atomic CQE 数增加约 `5.4x`,dispatch 仍比 32-token 快约
  `20-25%`。
- 当前关键限制是 payload/count 首次可见和 forward 流式消费延迟,不是 CQE 总吞吐。
- 2-token 开始受到 EFA 小消息效率/命令压力限制;4-token 是当前测试配置的最佳点。
- 相比本轮开始时约 `30-31 GB/s`,当前达到约 `38 GB/s`。

## 2026-06-07: tail wait discriminator 与 V1-style inflight cap

### profiling code review

检查了新增 profiling:

- `ep/include/uccl_gin/resources.cuh` 新增
  `forward_tail_ready_events`, `forward_tail_stall_events`,
  `forward_tail_stall_cycles`。
- `hybrid_dispatch.cuh` 在 forward warp 进入 `timeout_while` 前,复用已有
  `stored_scaleout_tail_idx / stored_scaleout_old_tail_idx / stored_finish_flag`
  做一次 ready/stall 判别。
- 该判别不额外重读 tail word,不改变 `timeout_while` 控制流,适合作为 tail 首次可见
  的 discriminator。

### tail discriminator run

同步文件:

```text
ep/include/uccl_gin/resources.cuh
thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh
```

重新安装:

```text
/tmp/uccl_gin_taildisc_build.log
```

运行命令要点:

```text
EP_JIT_CACHE_DIR=/tmp/deepep_jit_uccl_taildisc
UCCL_GIN_DISPATCH_CLOCK_PROFILE=1
UCCL_PROXY_PROFILE_COMMANDS=1
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic --skip-check
```

日志:

```text
/tmp/uccl_gin_taildisc_rank0.log
/tmp/uccl_gin_taildisc_rank1.log
```

汇总:

```text
rank0:
  avg scaleout_d2h cycles/event        85.5k
  avg forward_tail_wait cycles/event   24.4k
  tail stall fraction                  23.3%
  avg stalled tail wait cycles/event  104.1k
  avg forward_load cycles/event        18.0k

rank1:
  avg scaleout_d2h cycles/event        68.0k
  avg forward_tail_wait cycles/event   53.3k
  tail stall fraction                  29.4%
  avg stalled tail wait cycles/event  180.7k
  avg forward_load cycles/event         8.1k
```

结论:

- forward warp 约 `70-77%` tail check 第一次看时已经 ready。
- stall 事件占比不是多数,但单次 stall 很重,仍然是 critical path 的一部分。
- clock profile 很侵入,本 run cached dispatch 只有 `10-12 GB/s`;headline BW 不作为优化
  对比,只看 counters。

### P1 inflight cap implementation

实现 V1-style cap:

- `RingBuffer::atomic_set_and_commit` 增加可选 `max_inflight` 参数。
- `d2hq::D2HHandle::atomic_set_and_commit` 透传该参数。
- UCCL-GIN Rail 的 `rail_put`, `rail_put_tail_add`, `rail_red_add` 传
  `kUCCLGinMaxInflightNormal`。
- `common.hpp` 新增 `UCCL_GIN_MAX_INFLIGHT_NORMAL`,默认 `8`,和 V1 normal path 的
  `kMaxInflightNormal` 对齐。
- `ep/Makefile` 新增 `GIN_MAX_INFLIGHT_NORMAL` 变量。注意 DeepEP JIT 也包含 header,
  因此改变该值需要同步 header/JIT flags 或清 JIT cache。

第一次远端运行失败:

```text
/tmp/uccl_gin_cap8_rank0.log
/tmp/uccl_gin_cap8_rank1.log
```

原因:

- JIT 编译 `uccl_gin_rail.cuh` 时调用的是 `d2hq::D2HHandle::atomic_set_and_commit`,
  不是底层 `RingBuffer::atomic_set_and_commit`。
- 只改底层签名会导致 `too many arguments in function call`。

修复:

- 给 `D2HHandle::atomic_set_and_commit` 同步增加可选 `max_inflight` 参数并透传到底层
  ring。

构建日志:

```text
/tmp/uccl_gin_cap8_build.log
/tmp/uccl_gin_cap8_build2.log
```

最终验证:

```text
/tmp/uccl_gin_cap8_rank0.log
/tmp/uccl_gin_cap8_rank1.log
```

结果:

```text
correctness: pass
cached dispatch: 37-38 GB/s (SO), 1.62-1.64 ms
```

结论:

- cap=8 正确,但没有可见性能提升;和当前 best baseline `~38 GB/s` 基本持平。
- 当前 `38 -> 59 GB/s` gap 不能主要归因于 D2H ring 允许 inflight 到 2048。
- P1 inflight cap 保留为可调机制,但优化优先级降低。下一步应聚焦 tail stall 长尾、
  payload 首次可见、forward source/channel 调度与 V1 receiver ready/tag 消费方式。

## 2026-06-07: 更新 UCCL-GIN overview

Review 并修正 `ep/docs/uccl_gin_overview.md`:

- 明确文档描述的是当前 EP8x2 dispatch 主路径;compact path 仍使用
  `remote_scaleout_rank_idx = scaleout_rank_idx ^ 1`,尚未泛化到两个以上 scaleout rank。
- 将 FP8 hidden=7168 的 token payload 从错误的 `~14KB` 修正为 `~7-8KiB`;
  chunk=4 payload 约 `30KiB`,不是 `56KB`。
- 将 CPU receiver apply 从容易误解为 CUDA atomic 的 `atomicAdd` 修正为
  `std::atomic<int64_t>::fetch_add`。
- 将“payload 和 count 原子到达”改为准确语义:payload WRITE 和 immediate 属于同一
  WR,receiver 收到 CQE 后才 apply count。
- 分离当前 chunk=4 基线与历史 chunk=32 profiling 数据,删除将
  `post_gpu_us/post_cmds` 误当 active cost 的 `8 us/cmd / 48x` 结论。
- 补入 tail discriminator 结果、cap=8 验证结果,并删除“V2 无 cap”的陈旧描述。
- 记录 forward 每轮最多消费 `kNumSlotsPerForwardChunk=3`,与 payload compact chunk=4
  是不同粒度。
- 记录 ordered immediate sequence 只有 16 个值;默认 cap=8 能限制 sender/network
  侧单 ring inflight 并降低 sequence alias 风险,但不能单独证明 receiver CPU 已及时
  apply CQE。`GIN_MAX_INFLIGHT_NORMAL=0` 只能用于受控诊断,不应视为安全生产配置。

## 2026-06-07: combine → UCCL-GIN 转换(代码完成,运行时 fault 未定位)

### 改动(已 compile 验证,未跑通)

镜像 dispatch 的接法把 combine scale-out 接到 UCCL-GIN:
- `hybrid_combine.cuh`: `#ifdef DEEPEP_USE_UCCL_GIN` 下 gin=`handle::UCCLGin` + 清
  atomic_tail_base + grid sync;两处 `gin.put<Rail>` 经 `issue_rail_put` 走 UCCL D2H
  (lane_hint=channel_idx);finish 信号从 `flush+red_add_rel` 改为 `rail_tail_add(finish)`;
  receiver 等待从 workspace signaled_tail 改为 `rail_tail_ptr`/`decode_rail_tail`。
- `combine.hpp`: Args 加 `use_uccl_gin_resources`+`uccl_gin_resources`,launch_impl 加
  UCCL 分支,launch_combine 签名加 resources 指针(复用 dispatch.hpp 的 NativeUCCLGinResources)。
- `buffer.hpp`: launch_combine 调用传 `uccl_gin_resources_enabled ? &... : nullptr`。

### 编译验证(通过)

- host `make install`: BUILD_RC:0(combine.hpp/buffer.hpp host 侧 OK)。
- 独立 nvcc -cubin 编 `hybrid_combine_impl<...>`,带和不带 `-DDEEPEP_USE_UCCL_GIN`
  都成功(device codegen OK,NCCL 路径未破坏)。

### 运行时 fault(未解决)

EP8x2 correctness(README config 和 small smoke 64-token 都一样)在 combine 调用处
crash:`Cuda failure 700/716`,报在 `symmetric.hpp:136`(`ncclMemFree` 析构,属
sticky error 滞后上报,真正 fault 在 combine kernel 内)。

基线对照:
- pristine(UCCL dispatch + NCCL combine)同 config **通过**(RANK0_EXIT:0)。
- 我的 UCCL combine **crash**。

二分(每步重编 JIT header,fresh cache;注意 NOFINISH 会移除必需的 scaleout 同步,
是污染因子,所以 NOPUT-only 那次才是干净信号):
- NOPUT(保留真实 clear/finish/wait):仍 crash → **不是 put 路径**。
- 小 config 也 crash → **非 scale 相关,是 fundamental**。
- NOREGADJUST(关掉 combine 的 warpgroup_reg_alloc):仍 crash → **不是寄存器重分配**。
- dispatch/combine 的 `kNumChannels` 都是 32(JIT cache 实测),atomic tail 64 words →
  **无 channel 数不匹配、clear 无越界、tail offset 无溢出**。
- clear / rail_tail_add / rail_tail_ptr 三段和 dispatch **逐字节相同**,dispatch 跑得通 →
  机制本身没问题(含 by-value resources 参数的 ABI marshalling,dispatch 已证明)。

结论:combine kernel 的 UCCL 路径有一个运行时非法访问,但所有"和 dispatch 不同"的
单点都被排除,且 dispatch 用相同的 op 跑得通。需要 cuda-gdb 级别定位(compute-sanitizer
在本 EFA/GDR + symmetric memory 设置下会因 cuMemCreate/cuMemGetHandleForAddressRange
800/801 破坏 setup,无法干净复现)。

server 已恢复 pristine 工作态 .so;本地保留 combine 干净 WIP(去掉所有 bisect/printf
scaffolding)。下一步候选:cuda-gdb 单 pair 跑;或逐段在 combine 里二分 reduce/TMA/
get_sym_ptr<Lsa> 路径(这些是 combine 独有、用 UCCLGin 跑但我没改的代码)。

## 2026-06-07: UCCL-GIN combine WIP code review 与本地修复

### Review 结论

重新对照了:

- 当前 UCCL-GIN combine WIP；
- 已跑通的 UCCL-GIN dispatch；
- 上游 `hybrid_combine.cuh`；
- `UCCLGin` / `uccl_gin_rail.cuh`；
- proxy normal-mode WRITE / ordered atomic / receiver sequence 路径。

确认前一轮二分尚未排除两个代码层问题:

1. **初步判断，后续 normal-mode 路径核对后已推翻: combine phase bit**

   `uccl_gin_rail.cuh` 的 `rail_put`、`rail_put_tail_add`、`rail_red_add` 原来都硬编码:

   ```text
   make_cmd_type(..., is_combine=false, ...)
   ```

   后续逐个核对 consumer 后确认，这个判断不适用于当前 UCCL-GIN normal-mode:
   ordered atomic 的 bit 30 被 `PackAtomicWithSeq` 复用为 `seq[3]`，normal-mode
   receiver 只调用 `GetSeq()`；WRITE-with-imm receiver 也不读取 phase。当前路径传递
   combine phase 完全不可观察，反而会误导后续设计。

2. **高概率 crash 候选: combine 独有的 device-side 状态膨胀**

   当前 `UCCLGin` 每个 GPU 线程都复制一整份 `UCCLGinResources`，同时 combine WIP
   又在上游本来已经很重的 reduce/TMA lambda 链中增加了一层 `issue_rail_put` lambda。
   NOPUT 只让 lambda 内不真正发 WR，并不会消除 handle/resource copy、lambda capture
   和可能的 local-memory spill；NOREGADJUST 也只关闭 warpgroup register reallocation，
   不能排除普通 register spill/local stack fault。

   dispatch 使用相同 handle 能跑通，只证明较轻的 dispatch kernel 能承受，不能证明
   combine 的 reduce/TMA kernel 也能承受。因此这是当前最值得先消除的 combine 独有
   差异，但尚不能在没有服务器运行的情况下宣称它就是根因。

额外核对:

- combine launch 本来就是 cooperative launch；新增 `this_grid().sync()` 不是因为
  非 cooperative launch 直接非法。
- UCCL combine 的 tail offset 当前 EP8x2/32-channel 配置没有越界，但此前缺少和
  dispatch 对齐的编译期 ordered-atomic offset 上限检查。
- finish wait 清零 host-mapped tail 后此前没有 system fence；下一轮复用前存在可见性
  风险，虽然它不像当前首次 combine crash 的直接根因。

### 本地代码修复

- `ep/include/uccl_gin/uccl_gin_handle.cuh`
  - `UCCLGin::res` 从逐线程 value copy 改为引用 kernel 参数中已有的 resources；
- `ep/include/uccl_gin/uccl_gin_rail.cuh`
  - 后续 review 已删除无效的 `rail_is_combine` 传递；ordered Rail atomic 明确只支持
    normal-mode sequence receiver。
- `hybrid_combine.cuh`
  - 删除新增的嵌套 `issue_rail_put` lambda，两处 put 恢复为贴近上游的直接 call
    site，只在 UCCL 分支追加 `ncclGin_None()` 和 `channel_idx`；
  - 增加 compact tail 区域必须装入 ordered-atomic offset field 的静态检查；
  - wait 成功后清 tail，并补 `__threadfence_system()`。

### 本地验证

```text
git diff --check: pass
调用点 grep: pass，rail helper 新参数均有默认值，standalone microbench 调用未破坏
本机 nvcc: unavailable
```

因此本轮只完成本地静态 review/修复，尚未做 CUDA compile 或服务器 runtime 验证。

下一次服务器验证必须使用 fresh `EP_JIT_CACHE_DIR`，并先看:

1. UCCL combine 是否不再出现 700/716；
2. 若仍失败，fault 前是否出现
   `DeepEP combine (scale-out wait all, UCCL-GIN) timeout`，以区分主动
   `ptx::trap()` 和真正非法地址；
3. 保存 JIT compile 输出中的 registers / stack / spill 数据，对比修改前后的
   combine cubin；若 fault 消失且 spill 明显下降，才能坐实 device-state 膨胀根因。

## 2026-06-07: UCCL-GIN combine 修复服务器验证

### 同步与构建

只同步本轮涉及的五个代码文件到共享 EFS 工作区:

```text
ep/include/uccl_gin/uccl_gin_handle.cuh
ep/include/uccl_gin/uccl_gin_rail.cuh
thirdparty/DeepEP-v2-d4f41e4/csrc/elastic/buffer.hpp
thirdparty/DeepEP-v2-d4f41e4/csrc/kernels/elastic/combine.hpp
thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_combine.cuh
```

UCCL extension 在两台机器上安装成功:

```bash
make -C ep install PYTHON=$VIRTUAL_ENV/bin/python \
  CUDA_PATH=/usr/local/cuda-13.0 SM=90 -j16
```

日志:

```text
/tmp/uccl_gin_combine_review_build.log
/tmp/uccl_gin_combine_review_install.log
```

重新 build vendored DeepEP `_C` 时遇到并处理了两个环境问题:

1. ABI static assertion 引入 `ep/include` 后缺少 `util/gpu_rt.h`，补入仓库
   `include` 路径；
2. link 阶段找不到 `libnccl.so.2`，补入 venv 的 NCCL `LIBRARY_PATH` 和
   `LD_LIBRARY_PATH`。

最终 `_C` build 成功，说明新增 host/device resources ABI 检查可编译。

```text
失败日志:
  /tmp/uccl_gin_combine_review_deepep_build.log
  /tmp/uccl_gin_combine_review_deepep_build2.log
成功日志:
  /tmp/uccl_gin_combine_review_deepep_build3.log
```

### Fresh-cache small correctness

先跑之前稳定触发 combine `Cuda failure 700/716` 的 EP8x2 小配置:

```text
num_processes=8, num_sms=8, num_tokens=64, hidden=2048,
num_topk=6, num_experts=256
```

两端均退出 `0`，完整 dispatch/combine correctness 通过，原 fault 未复现。

```text
/tmp/uccl_gin_combine_review_smoke_rank0.log
/tmp/uccl_gin_combine_review_smoke_rank1.log
/tmp/deepep_jit_combine_review_rank0
/tmp/deepep_jit_combine_review_rank1
```

### Fresh-cache README 对齐 EP8x2

使用完整 correctness 检查运行:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic
```

配置使用 CUDA 13.0、aws-ofi-nccl master proxy、`DEEPEP_USE_UCCL_GIN=1`，
并为两个节点分别使用 fresh JIT cache。两端均退出 `0`，没有 timeout、
`Cuda failure 700/716`、Traceback 或 assertion:

```text
dispatch:          37-38 GB/s (SO), 1.60-1.64 ms
expanded dispatch: 37-38 GB/s (SO), 1.61-1.64 ms
cached dispatch:   37-38 GB/s (SO), 1.60-1.63 ms
combine:           28-29 GB/s (SO), 4.00-4.18 ms
reduced combine:   31-32 GB/s (SO), 3.72-3.80 ms
```

日志和 JIT cache:

```text
/tmp/uccl_gin_combine_review_readme_rank0.log
/tmp/uccl_gin_combine_review_readme_rank1.log
/tmp/deepep_jit_combine_review_readme_rank0
/tmp/deepep_jit_combine_review_readme_rank1
```

### 结论

- UCCL-GIN combine 现在不仅能编译，而且 small 与 README 对齐 EP8x2 的完整
  correctness、普通 combine、reduced combine 都已跑通；此前稳定出现的
  `700/716` fault 已消失。
- 当前验证同时包含 resources 改引用、移除额外 lambda、compact tail bound 和清 tail
  system fence，因此不能仅凭这次整体通过把 crash
  唯一归因到其中某一项。
- `rail_is_combine` 后续确认在当前 normal-mode 下无效并已删除。resources 引用和
  移除 lambda 同时降低了 combine kernel 的 device-side 状态压力；其中 resources
  value-copy 是旧 crash 的最强嫌疑，但仍未做单变量重现。若需要精确定位旧 crash 根因，应分别
  revert 单项并用 fresh JIT cache 重跑，而不是根据整体通过过度推断。
- combine 已进入可工作的主路径，但 `28-29 GB/s SO` 仍明显低于当前 dispatch
  `37-38 GB/s SO`，后续性能 profiling 应单独分解 combine scaleout send、receiver
  wait、reduce/TMA 和 D2H/proxy 背压。

## 2026-06-07: combine follow-up review: phase bit、跨轮清零与 uncompacted puts

### Review 结论

1. `rail_is_combine` 在当前 UCCL-GIN EFA normal-mode 确实无效:

   - ordered ATOMIC 经 `PackAtomicWithSeq`，legacy `is_combine` bit 被改写为
     `seq[3]`；
   - normal-mode receiver 对 ATOMIC 只读取 `GetSeq()`；
   - normal-mode WRITE-with-imm receiver 不读取 phase；
   - sender barrier 和 fast-mode 的 `get_is_combine()` consumer 不在当前路径执行。

   因此删除 `rail_is_combine` 和 Rail helper 的 phase 参数，并在 `AtomicsImm` 与
   `rail_red_add` 处注明:ordered UCCL-GIN atomic 必须留在 normal-mode sequence
   receiver；若误入 fast-mode，bit 30 会被错误解释成 combine phase。

2. “跨 iteration 的 late finish 会在下一轮 clear 后落地，因此 launch 前必须 host
   quiet”不适用于当前协议:

   - 每个 receiver warp 必须观察到每个来源的 finish 已由 CPU receiver apply，才会
     清对应槽并退出 kernel；
   - finish ATOMIC 又依赖该 lane 之前的 plain payload WRITE CQE；
   - 所以 receiver kernel 返回时，本地不存在尚未 apply 的旧 finish，也不存在旧
     payload 仍会在 finish 后落地。

   kernel-start reset 保留为防御性初始化，并补充上述生命周期注释；不加入会串行化
   transport 的 host quiet。这个结论依赖当前“每个来源都 wait + clear”协议，未来若
   改成异步跨 iteration pipeline，必须重新设计 epoch/double-buffer，而不是补一个
   粗粒度 quiet。

3. combine payload 目前确实未 compact:

   - 两处 scale-out put 都是每 token 一个 plain WRITE；
   - finish ATOMIC 通过 proxy dependency tracking 等待这些 WRITE CQE；
   - 这与当前 correctness 一致，也能解释 combine `28-29 GB/s SO` 低于 dispatch
     `37-38 GB/s SO`。

   本轮不改变 combine batching；后续性能阶段应实现保留 V2 combine layout 的 compact
   payload/piggyback 路径，并单独验证 WR 数量、message size 和 finish dependency。

### Follow-up 服务器验证

同步上述精简后，首次在两台节点上并行执行 `make -C ep install` 失败:

```text
p5en_0: ld: final link failed: Stale file handle
p5en_1: ld: cannot find -lnuma
```

这不是代码编译错误，而是两台机器同时链接共享 EFS 上同一个 `ep/ep.abi3.so`，以及
非 login shell 未带 `/home/ubuntu/local-lib`。改为顺序构建，并显式设置:

```bash
export LIBRARY_PATH=/home/ubuntu/local-lib:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ubuntu/local-lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

随后 p5en_0 build/install 和 p5en_1 install 均通过。后续共享 EFS 产物不得在两节点
并行链接。

构建日志:

```text
/tmp/uccl_gin_combine_followup_build.log       # 首次并行失败
/tmp/uccl_gin_combine_followup_build2.log      # p5en_0 顺序构建成功
/tmp/uccl_gin_combine_followup_install.log     # p5en_1 安装成功
```

使用 fresh JIT cache 再跑 README 对齐 EP8x2 完整 correctness:

```text
rank0 exit: 0
rank1 exit: 0
dispatch/cached dispatch: 37-38 GB/s (SO)
combine:                  28-30 GB/s (SO)
reduced combine:          31 GB/s (SO)
```

没有 timeout、CUDA 700/716、Traceback 或 assertion。测试本身包含多次 dispatch /
combine 调用，因此也覆盖了当前同步执行模型下的跨 iteration tail 复用。删除
`rail_is_combine` 后 correctness 和性能均未变化，进一步确认它在当前 normal-mode
路径中是 inert 的。

验证日志和 fresh JIT cache:

```text
/tmp/uccl_gin_combine_followup_rank0.log
/tmp/uccl_gin_combine_followup_rank1.log
/tmp/deepep_jit_combine_followup_rank0
/tmp/deepep_jit_combine_followup_rank1
```

## 2026-06-07: 重写 UCCL-GIN 性能计划

根据当前代码与最新验证结果，重写 `ep/docs/uccl_gin_perf_plan.md`:

- 从仅讨论 dispatch 改为同时覆盖 dispatch 与已经跑通的 combine。
- 固定当前 README-like EP8x2 基线:
  - dispatch/cached dispatch `37-38 GB/s SO`；
  - combine `28-30 GB/s SO`；
  - reduced combine `~31 GB/s SO`。
- 将已经完成或证伪的方向移出首要路径:
  - dispatch compact chunk sweep 已确认 4-token 最优；
  - compact32 平均不足 32 不是 non-contiguous flush bug；
  - sender dependency 已收窄；
  - proxy thread、ring size、inflight cap、receiver reorder lookup 均不是当前主瓶颈；
  - `rail_is_combine` 在 normal-mode inert；
  - 当前同步协议不需要 per-iteration host quiet。
- 将下一阶段最高优先级改为 combine critical-path profile:
  - kernel 阶段分解；
  - local/remote pointer 连续 run 和 merge opportunity；
  - plain WRITE / finish dependency latency。
- combine batching 只有在 profile 证明连续性机会足够后才实施，优先 direct contiguous
  multi-token WRITE，不预先引入新 staging buffer，也不改变 V2 receiver layout。
- dispatch 下一阶段转向 forward critical path:
  - 轻量 tail/payload latency discriminator；
  - `kNumSlotsPerForwardChunk=3` 对 compact publication=4 的消费粒度 sweep；
  - source/channel head-of-line blocking profile。
- 明确 CPU steady clock 与 GPU `clock64()` 不可直接相减；跨域端到端归因必须先做
  tagged sequence 和时钟校准，否则只分别分析 CPU/GPU 区间。
- 增加统一实验记录格式和量化验收线，避免低于噪声的微优化被当作性能进展。

## 2026-06-07: 开始执行新性能计划 P1 combine profiling

### 假设

当前 combine `28-30 GB/s SO / ~4 ms` 明显慢于 dispatch `37-38 GB/s SO`，但尚未
分清主要时间来自 reduce/TMA、每 token D2H put、finish dependency 还是 receiver
finish wait。Combine 两个真实 put call site 是否存在足够的 local/remote 连续 run
也未知，不能在没有数据时直接实现 compact staging。

### 本地改动

新增默认零开销、仅由 `UCCL_GIN_COMBINE_PROFILE=1` 开启的 JIT profiling:

- `ep/include/uccl_gin/resources.cuh`
  - 新增 combine clock / merge-opportunity counters 与 profile helper。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/buffers/elastic.py`
  - 环境变量自动追加 `-DDEEPEP_UCCL_GIN_COMBINE_PROFILE`。
- `hybrid_combine.cuh`
  - 统计 scale-up wait、reduce、D2H put、finish D2H、finish wait；
  - 统计真实 remote put 序列的 same-dst、local contiguous、remote contiguous、
    both-contiguous transition；
  - 统计 both-contiguous run-length histogram 与 break reason；
  - 输出中带 `expanded` / `multiple_reduction`，区分普通与 reduced combine JIT 实例。

Counters 复用 `atomic_tail_base` 中 compact tail 区域之后的 scratch，不新增 buffer、
command、wire 字段或 host API。Profiling 开启时 kernel 末尾增加 grid sync 以获得完整
计数，因此其 headline BW 只用于观察侵入性，不能作为优化前后性能结论。

### 当前状态

```text
git diff --check: pass
服务器 build/JIT/runtime: 尚未验证
```

## 2026-06-07: P1.2 combine merge opportunity 结果与 P1.3 开始

### 服务器验证

顺序构建和两节点 README-like EP8x2 profiling 均退出 `0`。

```text
build:
  /tmp/uccl_gin_combine_p1_build.log
  /tmp/uccl_gin_combine_p1_install.log

runtime:
  /tmp/uccl_gin_combine_p1_rank0.log
  /tmp/uccl_gin_combine_p1_rank1.log

extracted profile:
  /tmp/uccl_gin_combine_p1_profile_rank0.txt
  /tmp/uccl_gin_combine_p1_profile_rank1.txt

fresh JIT cache:
  /tmp/deepep_jit_combine_p1_rank0
  /tmp/deepep_jit_combine_p1_rank1
```

Profiling 开启时 combine headline 膨胀到约 `132-168 ms / ~1 GB/s`，原因是首版
profile 对每个 token/transition 都执行 host-mapped atomic。因此 clock 周期只能观察
相对长尾，不能作为正常路径的耗时占比。

### P1.2 决定性结果

rank0 聚合的两个实际 JIT 实例:

```text
mode expanded=0, multiple_reduction=1:
  samples=496, remote_puts/kernel=8155.1
  same_dst=100.00%, local_contig=66.54%, remote_contig=0.00%
  both_contig=0.00%, run_1=100.00%

mode expanded=1, multiple_reduction=1:
  samples=496, remote_puts/kernel=8155.1
  same_dst=100.00%, local_contig=66.54%, remote_contig=0.00%
  both_contig=0.00%, run_1=100.00%
```

根因来自当前 V2 combine replay layout:

```text
local ptr 约 2/3 随 replay slot i 连续；
remote ptr 使用 src_token_idx，当前 emission order 中每次都跳跃。
```

所以普通 RDMA WRITE 无法直接合并多个 token。Proxy coalescing 和
`ncclGinOptFlagsAggregateRequests` 都不能把连续 local bytes scatter 到不连续 remote
slot。原计划中的 direct contiguous batch 被数据否定，不能继续直接实现。

### P1.3 改动

为 pending finish atomic batch 增加仅在 `UCCL_PROXY_PROFILE_COMMANDS=1` 时记录的
低开销延迟:

```text
dependency_batches
dependency_enqueue_to_post_ns / max
dependency_ready_to_post_ns / max
```

当前协议中，带 plain WRITE dependencies 的 atomic batch 对应 combine finish:

- `enqueue -> post`: finish 等 payload CQE 的总时间；
- `ready -> post`: 最后一条 dependency 完成后，proxy progress 到真正 post 的调度延迟。

默认路径不读取时钟，也不改变 command、wire、ordering 或 layout。下一步在服务器仅开
proxy profile 跑 README-like EP8x2，以正常 combine kernel 测量 finish dependency。

### P1.3 服务器结果

同步后顺序 build/install 通过:

```text
/tmp/uccl_gin_combine_p13_build.log
/tmp/uccl_gin_combine_p13_install.log
```

首次运行未进入 benchmark。两台当前实例的私网地址已经变化:

```text
旧记录:
  p5en_0 = 172.31.78.36
  p5en_1 = 172.31.72.96

当前:
  p5en_0 = 172.31.70.225
  p5en_1 = 172.31.71.140
```

使用旧 `MASTER_ADDR=172.31.78.36` 时，torch TCPStore 报 `No route to host`。停止本轮
自己的测试进程后，改用 `172.31.70.225:29678` 重跑成功。

成功日志与 fresh JIT cache:

```text
/tmp/uccl_gin_combine_p13b_rank0.log
/tmp/uccl_gin_combine_p13b_rank1.log
/tmp/deepep_jit_combine_p13b_rank0
/tmp/deepep_jit_combine_p13b_rank1
```

两端退出 `0`。只开启 `UCCL_PROXY_PROFILE_COMMANDS=1`，未开启侵入式 GPU combine
profile。普通 combine 保持约 `27-29 GB/s SO / 4.1-4.3 ms`；reduced combine 在
proxy profile 开启时下降到 `16-19 GB/s / 6.3-7.3 ms`，说明该累计 profile 对 reduced
路径仍有明显扰动，不能用其 headline 作为正常性能。

聚合 finish dependency:

```text
node0:
  batches=72,204
  enqueue -> post avg/max = 126.485 / 518.638 us
  ready -> post avg/max   =  14.639 / 461.211 us

node1:
  batches=74,087
  enqueue -> post avg/max = 123.502 / 462.266 us
  ready -> post avg/max   =  13.863 / 268.134 us
```

结论:

- finish 等 payload CQE 的平均总延迟约 `0.12 ms`，只占普通 combine `4.1-4.3 ms`
  的约 `3%`；
- dependencies 已 ready 后 proxy 再 post finish 的平均调度延迟约 `14 us`；
- finish dependency / pending atomic container 不是当前 combine 主瓶颈；
- combine 后续应集中于每 token reduce/TMA、D2H emission 和约 `8155` 条不连续
  payload WRITE；
- P1.3 completed，下一步执行低侵入阶段分解和 dispatch P3.2 sweep。

## 2026-06-07: P3.2 dispatch forward consume 粒度 sweep

### 改动

为 `hybrid_dispatch.cuh` 增加编译期开关:

```text
DEEPEP_UCCL_GIN_FORWARD_CHUNK_SLOTS
```

它只改变 forward warp 在 tail 已发布后每轮最多消费的 slot 数，不改变 sender
publication、wire command、tail/count、receiver layout 或 buffer size。NCCL-GIN
保持上游默认 `3`。

首次 sweep JIT 编译失败，因为模板默认参数中的宏表达式未加括号，展开后的 `>` 被模板
解析器解释成结束符。修正为括号表达式后，JIT 与运行通过。

### 结果

README-like EP8x2，fresh JIT cache，`--skip-check` performance discriminator:

```text
slots=4:
  node0 cached dispatch: 36 GB/s, 1.688-1.697 ms
  node1 cached dispatch: 36-37 GB/s, 1.668-1.682 ms

slots=6:
  node0 cached dispatch: 38 GB/s, 1.590-1.609 ms
  node1 cached dispatch: 38-39 GB/s, 1.574-1.590 ms

slots=8:
  node0 cached dispatch: 37-38 GB/s, 1.619-1.644 ms
  node1 cached dispatch: 38-39 GB/s, 1.582-1.621 ms
```

日志与 cache:

```text
/tmp/uccl_gin_fwd4_rank{0,1}.log
/tmp/uccl_gin_fwd6_rank{0,1}.log
/tmp/uccl_gin_fwd8_rank{0,1}.log
/tmp/deepep_jit_fwd{4,6,8}_rank{0,1}
```

### Apples-to-apples 补测与最终结论

首次 `slots=3` 因模板表达式解析失败，没有形成有效 baseline。补跑修正后的显式
`slots=3`:

```text
slots=3:
  node0 avg 1.632 ms
  node1 avg 1.624 ms

slots=6:
  node0 avg 1.602 ms, improvement 1.9%
  node1 avg 1.583 ms, improvement 2.5%
```

补测日志:

```text
/tmp/uccl_gin_fwd3b_rank0.log
/tmp/uccl_gin_fwd3b_rank1.log
/tmp/deepep_jit_fwd3b_rank0
/tmp/deepep_jit_fwd3b_rank1
```

`slots=6` 的真实收益低于计划规定的 `3%` 保留门槛。删除实验宏并恢复上游默认
`kNumSlotsPerForwardChunk=kScaleoutUpdateInterval=3`，不给主路径留下低收益调参代码。
P3.2 completed and downgraded；forward consume 粒度不是当前主要 gap。

在得出最终结论前，曾用默认 `slots=6` 跑完整 correctness，两端退出 `0`:

```text
/tmp/uccl_gin_fwd6_correct_rank0.log
/tmp/uccl_gin_fwd6_correct_rank1.log
/tmp/deepep_jit_fwd6_correct_rank0
/tmp/deepep_jit_fwd6_correct_rank1
```

该 run 中 cached dispatch 为 `37-38 GB/s / 1.59-1.64 ms`，normal combine
`28-30 GB/s`，reduced combine `31-32 GB/s`，无 timeout、CUDA fault 或 assertion。
它证明更大 consume 粒度 correctness 可行，但性能不足以保留。

## 2026-06-08: Combine AggregateRequests cap 与 clock-only profile

### 假设与边界

Combine merge-opportunity profile 已证明当前 emission order 中:

```text
same_dst=100%
remote_contiguous=0%
both_contiguous=0%
run length=1
```

因此 `ncclGinOptFlagsAggregateRequests` 不能直接把多个普通 RDMA WRITE 合成一个连续
WRITE。仍值得验证的较小假设是:带 aggregate hint 的 payload put 是否被
`kUCCLGinMaxInflightNormal=8` 过度节流。

没有实现“把 finish piggyback 到最后 payload”。原因是 EFA SRD 不保证多个离散 WR
按到达顺序落地；最后发出的 payload 到达不能证明此前 payload 已全部到达。当前
standalone finish 的 sender-side completion dependency 提供了这项保证，而且此前
profile 显示其平均只占 combine wall time 约 `3%`。

### AggregateRequests 专用 cap sweep

临时改动:

- `rail_put` 接受可选 `max_inflight`；
- UCCL-GIN handle 对带 `ncclGinOptFlagsAggregateRequests` 的 put 使用独立 JIT cap；
- 普通 put 与 dispatch compact/piggyback 路径保持不变。

首次 cap=8 JIT 失败:

```text
uccl_gin_handle.cuh:113: expected an identifier
uccl_gin::8
```

根因是 `kUCCLGinMaxInflightNormal` 本身是宏，限定名被预处理成非法
`uccl_gin::8`。修正实验配置表达后，四组均成功运行。

README-like EP8x2，fresh JIT cache，`--skip-check`:

```text
aggregate cap   跨节点最慢普通 combine
8               4.166 ms
16              4.182 ms
32              4.171 ms
64              4.166 ms
```

日志:

```text
/tmp/uccl_gin_aggcap{8,16,32,64}_rank0.log
/tmp/uccl_gin_aggcap{8,16,32,64}_rank1.log
/tmp/deepep_jit_aggcap{8,16,32,64}_rank{0,1}
```

结论:没有提升或稳定趋势。按“无收益即回退”原则，完整删除 aggregate 专用 cap
实验代码。

### Normal put unlimited cap discriminator

不改代码，仅用 JIT flag:

```text
EP_JIT_EXTRA_FLAGS=-DUCCL_GIN_MAX_INFLIGHT_NORMAL=0
```

这使所有 normal put 只受 2048-slot D2H ring 容量限制。两节点均退出 `0`:

```text
cached dispatch: 约 38 GB/s, 最慢约 1.626 ms
normal combine:  约 28-29 GB/s, 最慢约 4.171 ms
reduced combine: 约 31 GB/s, 最慢约 3.783 ms
```

日志与 cache:

```text
/tmp/uccl_gin_normalcap0_rank0.log
/tmp/uccl_gin_normalcap0_rank1.log
/tmp/deepep_jit_normalcap0_rank0
/tmp/deepep_jit_normalcap0_rank1
```

结论:cap=8 不是 combine 或 dispatch 当前 wall-time 根因；不保留 cap=0，也不再优先
调 ring depth / inflight cap。

### P1.1 clock-only 采样 profile

原 `UCCL_GIN_COMBINE_PROFILE=1` 同时做逐 token merge 统计和阶段 clock atomic，使
普通 combine 膨胀到 `132-168 ms`。新增默认关闭的
`UCCL_GIN_COMBINE_CLOCK_PROFILE=1`:

- 复用现有阶段 counters；
- 禁用 merge-opportunity 逐 token 统计；
- 仅采样 `blockIdx.x % 8 == 0` 的 SM；
- profiling-off 路径不变。

运行成功，但普通 combine 仍被放大到约 `14.6-15.1 ms`，reduced combine 约
`18.4-19.1 ms`。因此不能用该 run 的 headline BW；事件均值只作为方向判别。

代表性稳定样本:

```text
node0:
  scaleup wait 约 2.6k cycles/event
  reduce       约 78k cycles/event
  D2H put      约 62k cycles/event
  finish D2H   约 52k cycles/event

node1:
  reduce       约 60-93k cycles/event
  D2H put      约 42-60k cycles/event
  finish wait 受慢节点/采样扰动影响，长尾明显
```

日志与 cache:

```text
/tmp/uccl_gin_combine_clock_rank0.log
/tmp/uccl_gin_combine_clock_rank1.log
/tmp/deepep_jit_combine_clock_rank0
/tmp/deepep_jit_combine_clock_rank1
```

结论:

- scale-up wait 通常不是主成本；
- 这批 clock counters 后续被复核为嵌套测量:forward reduce 的 callback 会触发
  D2H emission，所以 `reduce_cycles` 覆盖其中的 `d2h_cycles`，不能把二者相加；
- 放宽 D2H cap 已被独立实验否定，不能把 D2H cycles 简单解释成 cap 背压；
- 下一步执行 P2.1:只做 profile 模拟，在窗口 `4/8/16/32` 内判断按 remote
  `src_token_idx` 有界重排后能否创造足够的 contiguous run，再决定是否值得改 kernel。

### 回退后最终完整 correctness

将 aggregate 专用 cap 实验代码完整回退并同步服务器后，关闭全部 profiling，使用
fresh JIT cache 跑 README-like EP8x2 完整 correctness。两节点均退出 `0`，未出现
timeout、CUDA fault 或 assertion:

```text
cached dispatch:
  rank0 最慢 1.635 ms, 37-38 GB/s SO
  rank1 最慢 1.621 ms, 38 GB/s SO

normal combine:
  rank0 最慢 4.165 ms, 28-29 GB/s SO
  rank1 最慢 4.182 ms, 28-29 GB/s SO

reduced combine:
  rank0 最慢 3.800 ms, 31 GB/s SO
  rank1 最慢 3.784 ms, 31 GB/s SO
```

日志与 cache:

```text
/tmp/uccl_gin_combine_final_rank0.log
/tmp/uccl_gin_combine_final_rank1.log
/tmp/deepep_jit_combine_final_rank0
/tmp/deepep_jit_combine_final_rank1
```

最终保留的代码只有默认关闭的 clock-only 诊断模式；没有保留任何无性能收益的 cap
实验路径。

## 2026-06-07 18:19 PDT - Combine reduce/D2H 归因修正与 P2.1 重排 profile

### Review 结论吸收

确认一个重要测量问题:combine forward 路径第二个 `combine_reduce` 的
`Wait buffer release` callback 会调用 `flush_last_tma_and_issue_rdma()`，因此
`reduce_cycles` 覆盖了其中触发的 `d2h_cycles`。这意味着之前“per-token reduce 与
D2H emission 都不可忽略”的写法容易被误读成二者可相加。更准确的解释是:

```text
forward span measured by reduce_cycles
  includes callback-triggered D2H emission

pure reduce compute ~= reduce_cycles - nested d2h_cycles
```

所以 clock profile 只能用于阶段排序，不能作为 wall-time budget 求和。`clock64`
instrumentation 也会破坏原 kernel 设计中的 reduce/RDMA overlap，因此 headline BW 和
cycle sum 都不能直接作为真实性能。

另一个修正:`same_dst=100%` 是当前 2-node EP8x2/EP16 的拓扑 artifact。非本地目标只有
一个 remote node，所以它不能证明 emission order 天然按 dst 聚集；未来如果支持超过
2 个 scaleout node，任何 batching/reindex 都仍然必须先按 `dst_rank` bucket。

### P2.1 有界重排 profile 实现过程

第一版尝试在 GPU kernel 内直接统计窗口 `4/8/16/32` 的排序/contiguity counters。
实测这个方案太重:

- 全量 README-like 8192 token 会明显拖慢甚至 timeout；
- 缩到采样 3 个 SM、只看 channel 0 仍然过重；
- 进一步缩到 1k token 也不适合作为常规 profile。

这些中间尝试的日志保留在:

```text
/tmp/uccl_gin_combine_reorder_rank0.log
/tmp/uccl_gin_combine_reorder_rank1.log
/tmp/uccl_gin_combine_reorder_sample_rank0.log
/tmp/uccl_gin_combine_reorder_sample_rank1.log
/tmp/uccl_gin_combine_reorder_ch0_rank0.log
/tmp/uccl_gin_combine_reorder_ch0_rank1.log
/tmp/uccl_gin_combine_reorder_1k_rank0.log
/tmp/uccl_gin_combine_reorder_1k_rank1.log
```

随后把实现改成轻量 dump:kernel 只在 `UCCL_GIN_COMBINE_REORDER_PROFILE=1` 时打印
`channel_idx == 0` 的前 256 个 remote put candidates:

```text
UCCL_GIN_COMBINE_REORDER_CAND rank=... expanded=... multiple_reduction=...
  channel=0 seq=... dst=... local_off=... remote_off=... bytes=...
```

再用 CPU offline parser 在窗口 `4/8/16/32` 内按 `(dst, remote_off, local_off)` 排序，
统计 local/remote/both contiguity。这个方式不再污染 hot path，也避免在 GPU 内做
profile-only 排序。

成功运行日志与 JIT cache:

```text
/tmp/uccl_gin_combine_reorder_dump_rank0.log
/tmp/uccl_gin_combine_reorder_dump_rank1.log
/tmp/deepep_jit_combine_reorder_dump_rank0
/tmp/deepep_jit_combine_reorder_dump_rank1
```

两节点均退出 `0`。

### Offline parse 结果

rank0:

```text
groups 992, total_candidates 100949

KEY (rank=0, expanded=0, multiple_reduction=1):
  W4:  remote 0.0%, local 73.67%, both 0.0%
  W8:  remote 0.0%, local 68.66%, both 0.0%
  W16: remote 0.0%, local 66.43%, both 0.0%
  W32: remote 0.0%, local 64.92%, both 0.0%

KEY (rank=0, expanded=1, multiple_reduction=1):
  W4:  remote 0.0%, local 73.40%, both 0.0%
  W8:  remote 0.0%, local 68.31%, both 0.0%
  W16: remote 0.0%, local 66.30%, both 0.0%
  W32: remote 0.0%, local 64.89%, both 0.0%
```

rank1:

```text
groups 987, total_candidates 101367

KEY (rank=1, expanded=0, multiple_reduction=1):
  W4:  remote 0.0%, local 68.25%, both 0.0%
  W8:  remote 0.0%, local 67.13%, both 0.0%
  W16: remote 0.0%, local 66.53%, both 0.0%
  W32: remote 0.0%, local 65.44%, both 0.0%

KEY (rank=1, expanded=1, multiple_reduction=1):
  W4:  remote 0.0%, local 67.31%, both 0.0%
  W8:  remote 0.0%, local 66.92%, both 0.0%
  W16: remote 0.0%, local 66.68%, both 0.0%
  W32: remote 0.0%, local 66.06%, both 0.0%
```

结论:

- 小窗口有界重排不能创造 `remote_contiguous` run；`both_contiguous` 仍然是 `0%`。
- local 指针有 65-74% 连续性，但 remote V2 destination slot 完全打散，所以仅改变
  emission order 无法合并成 multi-token RDMA WRITE。
- P2.1 direct reorder batching 降级，不应实现。
- 后续若继续追求 combine payload batching，需要评估 receiver-facing staging /
  layout-aware compact 或 scatter/gather descriptor；这会触及 V2 layout 成本模型，
  不能再按“窗口内排序”这样的小改来估算收益。

代码清理:

- 删除了 GPU 内排序 profile 遗留的 reorder counter ABI 和尾部空 counter 打印。
- 保留默认关闭的轻量 `UCCL_GIN_COMBINE_REORDER_PROFILE=1` candidate dump，便于后续
  做更精细的 offline layout 分析。

### 清理后默认路径验证

同步到服务器:

```text
/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang
```

关闭所有 profile flag，使用 fresh JIT cache 跑 README-like EP8x2 full correctness:

```text
EP_JIT_CACHE_DIR=/tmp/deepep_jit_reorder_cleanup_rank0
EP_JIT_CACHE_DIR=/tmp/deepep_jit_reorder_cleanup_rank1

python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic
```

日志:

```text
/tmp/uccl_gin_reorder_cleanup_rank0.log
/tmp/uccl_gin_reorder_cleanup_rank1.log
```

结果:

```text
rank0 exit 0
rank1 exit 0
error keyword scan: no Traceback / CUDA error / assert / timeout / failed

cached dispatch:
  rank0: 38 GB/s SO, slowest 1625 us
  rank1: 38 GB/s SO, slowest 1622 us

normal combine:
  rank0: 28 GB/s SO, slowest 4197 us
  rank1: 28-30 GB/s SO, slowest 4130 us

reduced combine:
  rank0: 31 GB/s SO, slowest 3794 us
  rank1: 31 GB/s SO, slowest 3817 us
```

结论:清理 heavy reorder-profile 残留没有改变默认主路径；当前性能与此前 final
baseline 一致。测试结束后两台没有残留真实 `test_ep.py`/`torchrun`/GPU compute
进程，`pgrep` 只匹配到检查命令本身。

## 2026-06-07: PT.0 dispatch 分段 profiling 与 V1/V2 RDMA 过程对照

目标:

- 回答当前 V2 UCCL-GIN dispatch 和原 V1 UCCL-EP RDMA 过程的主要差距。
- 澄清“V1 是不是 32-token 一次”和“为什么当前 V2 4-token 最优”。
- 更新 `ep/docs/uccl_gin_perf_plan.md`，避免继续引用旧的 chunk=32 / `post_gpu_us`
  口径。

### 代码改动

增加低开销 proxy 分段 profile（仅 `UCCL_PROXY_PROFILE_COMMANDS=1` 开启）:

- `ep/include/proxy.hpp`
  - 新增 `profile_record_completion(uint64_t wr_id)`。
  - 新增 WRITE/ATOMIC post 计数、post ns、post->CQE ns/max 计数。
  - 新增 `wr_id -> post_time/type` map，只在 profiling 开启时使用。
- `ep/src/proxy.cpp`
  - `flush_writes()` 在 `post_rdma_async_batched` 前后计时，记录 WRITE post 热区。
  - `progress_pending_atomics()` 在 `post_atomic_operations` 前后计时，记录 ATOMIC
    post 热区。
  - `notify_gpu_completion()` 对 send CQE 调 `profile_record_completion`，按 wr_id
    计算 post 返回到 CQE 的时间。
  - `dump_command_profile()` 输出:
    `write_post_*`, `write_cqe_*`, `atomic_post_*`, `atomic_cqe_*`。

注意:

- 这组计时是为了拆分 `(b) proxy CPU/cmd` 与 `(c) post->CQE`，不用于 headline BW。
- p5en_1 缺 `numaif.h`，本轮只在 p5en_0 build，然后把 `.so` 复制到 p5en_1。

构建/同步:

```text
p5en_0 build log:
  /tmp/uccl_gin_pt0_build.log

p5en_1 build attempt log:
  /tmp/uccl_gin_pt0_build_p5en1.log
  失败原因: src/fifo.cpp include <numaif.h> 时找不到 numaif.h

同步二进制:
  p5en_0:/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/uccl/ep.abi3.so
  -> /tmp/uccl_ep_pt0.abi3.so
  -> p5en_1:/home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/uccl/ep.abi3.so
```

有一轮混合二进制 profile 日志不作为结论使用:

```text
/tmp/uccl_gin_pt0_proxy_rank0.log
/tmp/uccl_gin_pt0_proxy_rank1.log
```

### 1. 当前 V2 dispatch proxy 分段 profile

命令形状:

```bash
export UCCL_PROXY_PROFILE_COMMANDS=1
export UCCL_GIN_CHUNK_PROFILE=1
export EP_JIT_CACHE_DIR=/tmp/deepep_jit_pt0_proxy2_rank{0,1}
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic
```

日志:

```text
/tmp/uccl_gin_pt0_proxy2_rank0.log
/tmp/uccl_gin_pt0_proxy2_rank1.log
```

README-like 行（profile on，会扰动性能，不作为最终 headline）:

```text
rank0 node: cached dispatch 约 6 GB/s SO, 9.93-9.95 ms
rank1 node: cached dispatch 约 6-7 GB/s SO, 9.28-9.31 ms
```

chunk profile 聚合（每 node 取每个 rank 最后一条）:

```text
rank0 node:
  chunks=16616
  tokens=65291
  tokens/chunk=3.929
  bin_1=98, bin_2=359, bin_3_4=16159
  bin_5_8/bin_9_16/bin_32=0
  flush_full=15998
  flush_finish=618
  flush_noncontig=0

rank1 node:
  chunks=16623
  tokens=65241
  tokens/chunk=3.925
  bin_1=123, bin_2=379, bin_3_4=16121
  bin_5_8/bin_9_16/bin_32=0
  flush_full=15997
  flush_finish=626
  flush_noncontig=0
```

结论:

- 当前主路径确实是 `kUCCLGinCompactChunkTokens=4`。
- 不是“没有凑满 4”或“中途 non-contig flush”导致小 chunk；绝大多数是 full 4-token。
- 当前 4-token 最优必须从 receiver 可见性/forward pipeline 解释，不能简单归咎为
  batching bug。

proxy 分段聚合:

```text
rank0 node:
  WRITE post calls:      6,433,717
  WRITE WRs:            11,199,076
  WRITE post ns:         4,683,381,481
  WRITE post ns/WR:      418 ns
  WRITE CQE ns/WR:       67.1 us

  ATOMIC WRs:            199,040
  ATOMIC post ns/WR:     531 ns
  ATOMIC CQE ns/WR:      30.7 us

rank1 node:
  WRITE post calls:      6,618,793
  WRITE WRs:            11,206,585
  WRITE post ns/WR:      422 ns
  WRITE CQE ns/WR:       63.8 us

  ATOMIC WRs:            199,040
  ATOMIC post ns/WR:     539 ns
  ATOMIC CQE ns/WR:      31.3 us
```

结论:

- 旧结论“V2 proxy `post_gpu_us/post_cmds` 约 8us/cmd 是主因”不成立；那个指标包含
  空轮询、整轮 mixed loop、profile 扰动和不对称口径。
- 真正纯 CPU `ibv_post_send` 热区只有约 `0.42us/WRITE`，不是 `38 -> 59 GB/s`
  dispatch gap 的主因。
- 当前最重的直接可见段是 `post_send 返回 -> send CQE`，每个 4-token WRITE 约
  `64-67us`。这可能包含 EFA 小包 delivery、NIC/rail queueing、receiver 端进展和
  profiling 扰动；下一步要继续拆 receiver apply 与 forward 观察 tail。

### 2. 当前 V2 dispatch GPU clock profile

命令形状:

```bash
export UCCL_GIN_DISPATCH_CLOCK_PROFILE=1
export UCCL_GIN_CHUNK_PROFILE=1
unset UCCL_PROXY_PROFILE_COMMANDS
export EP_JIT_CACHE_DIR=/tmp/deepep_jit_pt0_clock_rank{0,1}
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic
```

日志:

```text
/tmp/uccl_gin_pt0_clock_rank0.log
/tmp/uccl_gin_pt0_clock_rank1.log
```

聚合方式:

- 每个 rank 取最后一条 `UCCL_GIN_DISPATCH_CLOCK`。
- clock profile 侵入性很强，只看阶段相对量级，不看 headline BW。

结果:

```text
rank0 node:
  scaleout_preload:        124 cycles/event
  scaleout_compact_store:   34 cycles/event
  scaleout_store_wait:    1173 cycles/event
  scaleout_d2h:         308788 cycles/event
  scaleout_tail:         15346 cycles/event
  forward_tail_wait:     82780 cycles/event
  forward_meta_wait:      3174 cycles/event
  forward_load:          21602 cycles/event
  forward_scaleup_store:   287 cycles/event

rank1 node:
  scaleout_preload:        124 cycles/event
  scaleout_compact_store:   34 cycles/event
  scaleout_store_wait:    1073 cycles/event
  scaleout_d2h:         291005 cycles/event
  scaleout_tail:          9594 cycles/event
  forward_tail_wait:     89789 cycles/event
  forward_meta_wait:      2579 cycles/event
  forward_load:          17836 cycles/event
  forward_scaleup_store:   395 cycles/event
```

结论:

- compact staging 自身很便宜；不是 4-token 最优的原因。
- `scaleout_d2h` 很高，但结合 proxy profile 看，不是 CPU post 慢，而是 GPU 侧等待
  command 被 proxy/NIC/receiver 进展吸收的下游症状。
- `forward_tail_wait` 和 `forward_load` 仍显著，说明 receiver tail/count 可见与
  payload 到 HBM/TMA load 是 dispatch critical path 的一部分。

### 3. V1 RDMA dispatch 过程与 chunk 事实

源码对照:

```text
V1 commit:
  495b7221d084cce92553d6a038376358bd218a5a

V1 benchmark config:
  ep/bench/test_internode.py
  Config(num_sms, nvl_chunk_send=8, nvl_chunk_recv=512,
         rdma_chunk_send=16, rdma_chunk_recv=512)

V1 sweep:
  rdma_chunk_size in range(4, 33, 4)
```

所以:

- V1 apples-to-apples baseline 不是固定 32-token 一次。
- 当前默认 HT/dispatch config 是 `rdma_chunk_send=16`。
- 32-token 是论文/某些 HT 配置或 tuning sweep 上界里的候选值，不应拿来直接说
  “V1 原来就是 32, V2 为何 4”。

V1 dispatch sender 关键路径:

```text
producer:
  token 按 dst RDMA rank/channel 写入连续 send_buffer[dst][slot]
  更新 rdma_send_channel_tail[dst]

coordinator:
  processed = rdma_send_channel_tail[dst] - last_issued_tail
  if processed >= rdma_chunk_send or all done:
      num_tokens_to_issue = min(processed, rdma_chunk_send)
      bytes = num_tokens_to_issue * num_bytes_per_token
      nvshmemi_ibgda_put_nbi_warp(..., bytes,
                                  atomic_offset=rdma_channel_tail,
                                  atomic_val=num_tokens_to_issue)

receiver:
  tail 表示 channel 进展，但 payload readiness 还由 per-token/source tag 检查。
```

当前 V2 UCCL-GIN dispatch:

```text
scaleout warp:
  token 按 V2 expanded/metadata 语义遍历
  TMA store 到 per-channel compact send slot
  每 4 token 发 rail_put_tail_add:
      payload WRITE_WITH_IMM + count delta

receiver/forward:
  count/tail apply 后才知道 compact slot 可消费
  forward 每轮最多消费 kNumSlotsPerForwardChunk=3
  再 TMA load payload -> scale-up layout
```

为什么 V2 现在 4-token 最优:

- 4-token WRITE 约 30KB，已经进入 EFA 小消息上沿，但仍能较早发布 count/tail。
- 16/32-token 会减少 WR 数，但 count/tail 要等整条大 payload WR 到达后才 apply；
  forward 更晚看到前几个 token ready，流水线被变粗。
- V1 可以用 16-token chunk 而不付出同样代价，是因为 receiver-facing staging 与
  per-token ready/tag 允许“大 payload chunk”和“细粒度消费”共存。

### 4. 更新 plan

更新 `ep/docs/uccl_gin_perf_plan.md`:

- 将贯穿性假设改成当前 dispatch 数据驱动版本。
- 增加 V1 RDMA dispatch 过程图和 chunk 事实。
- 在 5.1 记录本轮 proxy/chunk/clock profile 数字。
- 撤销旧的 V2 `post_gpu_us/post_cmds ~= 8us/cmd` 主因判断。
- 将 PT.1 CPU proxy hot-path 对 dispatch 降级。
- 将下一步改成:
  1. low-overhead receiver apply timing；
  2. source/channel HOL profile；
  3. 设计 V2-native ready/tag 或 receiver-facing landing，使更大 payload WRITE 不推迟
     细粒度 token ready。

## 2026-06-08 PT.0 补充: receiver apply 与 forward HOL profile

### 1. 目的

继续拆 dispatch 的剩余 gap，重点回答两个问题:

```text
1. receiver 端 WRITE_WITH_IMM / ordered atomic apply 本身是不是慢？
2. forward_tail_wait 是所有 source 都没 ready，还是 selected source 造成 HOL？
```

### 2. 代码改动

- `ep/include/proxy_ctx.hpp`
  - 新增 receiver atomic process / commit 的 profile counters。
- `ep/src/rdma.cpp`
  - 在 normal-mode receiver atomics path 中计时:
    - CQE decode + reorder + apply 的总处理时间；
    - 真正 `fetch_add` commit 的时间。
- `ep/src/proxy.cpp`
  - 在 `UCCL_PROXY_PROFILE` 输出 receiver apply counters。
- `ep/include/uccl_gin/resources.cuh`
  - 新增 dispatch clock fresh 分桶 counters:
    - selected source fresh read 后 ready；
    - other source ready；
    - no source ready。
- `thirdparty/DeepEP-v2-d4f41e4/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - 在 forward stall 后的一次 fresh tail read 中记录上述三类分桶。

### 3. 服务器操作

构建:

```text
p5en_0:
  make -C ep clean install CUDA_PATH=/usr/local/cuda-13.0 PYTHON=python
  build log: /tmp/uccl_gin_hol_build.log

p5en_1:
  复用 p5en_0 构建出的 ep.abi3.so:
  /home/ubuntu/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/uccl/ep.abi3.so
```

profile run:

```text
proxy / receiver timing:
  /tmp/uccl_gin_hol_proxy_rank0.log
  /tmp/uccl_gin_hol_proxy_rank1.log
  env:
    UCCL_PROXY_PROFILE_COMMANDS=1
    UCCL_GIN_CHUNK_PROFILE=1

dispatch clock + HOL:
  第一版日志:
    /tmp/uccl_gin_hol_clock_rank0.log
    /tmp/uccl_gin_hol_clock_rank1.log
  修复 CUDA device printf 参数上限后重跑:
    /tmp/uccl_gin_hol_clock2_rank0.log
    /tmp/uccl_gin_hol_clock2_rank1.log
  env:
    UCCL_GIN_DISPATCH_CLOCK_PROFILE=1
    UCCL_GIN_CHUNK_PROFILE=1
```

两组测试均为 README-like EP8x2:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic
```

### 4. 遇到的 bug

`UCCL_GIN_DISPATCH_CLOCK` 原本一条 device `printf` 中有 `rank + 32` 个参数。
CUDA device printf 有参数数量限制，最后一个 `forward_tail_fresh_no_ready_events`
被错读成 `0x4e49475f4c434355`，即 ASCII `"UCCL_GIN"`。

修复:

```text
把 fresh 分桶拆到第二条短 printf:
  UCCL_GIN_DISPATCH_CLOCK_FRESH
```

这个 bug 只影响 profiling 输出，不影响默认 dispatch/combine 路径。

### 5. 结果

proxy / receiver timing 聚合:

```text
rank0 node:
  WRITE post:                  427 ns/WR
  WRITE post->CQE:            63.4 us/WR
  ATOMIC post:                 539 ns/WR
  ATOMIC post->CQE:           30.7 us/WR
  receiver atomic process:     134 ns/CQE
  receiver atomic commit:       65 ns/fetch_add

rank1 node:
  WRITE post:                  422 ns/WR
  WRITE post->CQE:            67.0 us/WR
  ATOMIC post:                 535 ns/WR
  ATOMIC post->CQE:           30.8 us/WR
  receiver atomic process:     133 ns/CQE
  receiver atomic commit:       64 ns/fetch_add
```

dispatch clock + HOL 聚合:

```text
rank0 node:
  scaleout_d2h:       525.1k cycles / 4-token WRITE
  forward_tail_wait:  168.6k cycles / event
  fresh selected ready: 32.0%
  fresh other ready:    53.0%
  fresh no ready:       15.0%

rank1 node:
  scaleout_d2h:       517.6k cycles / 4-token WRITE
  forward_tail_wait:  180.5k cycles / event
  fresh selected ready: 27.9%
  fresh other ready:    55.3%
  fresh no ready:       16.9%
```

### 6. 结论

- receiver CQE decode / ordered apply / `fetch_add` 是百纳秒量级，不是 dispatch gap 主因。
- 超过一半 forward stall 在 fresh read 后发现“其他 source 已 ready”，说明
  source-selection HOL 是真实优化机会。
- 只有约 15-17% stall 是 fresh read 后所有 source 都没 ready，不能把 forward wait
  简单归因于 receiver apply 或 EFA delivery。
- plan 已更新:
  - `(b) proxy CPU/cmd` 与 `(d) receiver apply` 对 dispatch 降级；
  - 当前 dispatch 主嫌疑改成 `(c) 4-token WRITE post->CQE` 与
    `forward source-selection HOL`；
  - 下一步先验证 ready-source-first 是否能转化成 wall-time 收益；若低于保留门槛则回退。

## 2026-06-08 ready-source-first forward selection 尝试与回退

### 1. 假设

HOL profile 显示 fresh read 后超过一半 stall 是:

```text
selected source not ready, but another source ready
```

因此尝试最小行为改动:

```text
在 forward timeout loop fresh-read 之后:
  如果 selected source 已 ready -> 保持原 selected；
  如果 selected source 未 ready、但其他 source ready -> 切换到 ready source 并退出等待；
  否则继续等待。
```

该改动不碰 wire/layout/metadata，只改变 forward source selection。

### 2. 验证

profiling-off README-like EP8x2，fresh JIT cache:

```text
logs:
  /tmp/uccl_gin_ready_source_rank0.log
  /tmp/uccl_gin_ready_source_rank1.log
```

结果:

```text
rank0 node:
  dispatch:          38 GB/s, avg 1619 us
  expanded dispatch: 38 GB/s, avg 1622 us
  cached dispatch:   38 GB/s, avg 1604 us
  combine:           avg 27.9 GB/s, max 4949 us
  reduced combine:   avg 30.0 GB/s, max 4509 us

rank1 node:
  dispatch:          38 GB/s, avg 1617 us
  expanded dispatch: 38.1 GB/s, avg 1600 us
  cached dispatch:   38 GB/s, avg 1608 us
  combine:           avg 23.9 GB/s, max 5076 us
  reduced combine:   avg 24.9 GB/s, max 4817 us
```

### 3. 结论

- dispatch headline 没有超过保留门槛；仍是约 `38 GB/s`。
- combine path 理论上未改，但这次 rank1 combine 出现明显低值；无论是否噪声，
  该 patch 都没有收益。
- 按规则回退行为改动，只保留 HOL profiling counters。
- 新判断:HOL 现象真实存在，但简单切换 source 可能只是把等待转移到 metadata/payload
  load 或其他 pipeline 阶段；下一步不应继续堆 source switch 小改，而应先做
  post->CQE/NIC-rail profile 和更深 HOL phase profile。

## 2026-06-09 大 chunk + producer/coordinator 实验、ready 协议定位与回退

### 目标与 Step 0

验证“V2 不能使用 V1 的大 WRITE，只是因为 scaleout warp 被 D2H push 回压并阻塞”：

```text
cap=0 + chunk=16 + forward slots=16 -> 33-34 GB/s
cap=0 + chunk=32 + forward slots=16 -> 31-32 GB/s
```

日志：

```text
/tmp/uccl_step0_c16_s16_rank{0,1}.log
/tmp/uccl_step0_c32_s16_rank{0,1}.log
```

直接组合现有旋钮仍低于 `chunk=4 / slots=3` 的 `37-38 GB/s` 基线。

### Producer/coordinator 实现与 correctness 定位

实验把 scaleout warp 拆为只做 compact TMA store 的 producer，以及独立观察 ready
tail、发送 `rail_put_tail_add` 的 coordinator；forward burst 扩到 16 slots。

期间定位并修正了 shared-memory allocation、`BufferLayout` padding、多个 ring
`QUIET` 覆盖单一 proxy-wide `quiet_wr` 等问题。相关失败日志：

```text
/tmp/uccl_coordinator_quiet_c16_s16_rank0.log
/tmp/uccl_coordinator_quiet2_c16_s16_rank0.log
/tmp/uccl_coordinator_lane0_rank0.log
/tmp/uccl_coordinator_seqquiet_rank0.log
```

真正的 correctness 根因不是 quiet，而是 shared-memory `prepared_tail` 不能单独作为
TMA payload 已可被 CPU proxy/NIC 安全读取的正式 ready 协议：

```text
chunk=4 / slots=3 仍在第一次 dispatch_copy_epilogue 出现 metadata assertion。
coordinator 等 producer_done 后再发送 -> 完整小配置两端 RC=0。
```

日志：

```text
/tmp/uccl_coord_doneboundary_small_rank{0,1}.log
```

随后把 ready/done 状态放入 `atomic_tail_base` scratch，用单调
`atomicAdd_system` 发布/读取；流式 coordinator 完整小配置 correctness 通过：

```text
/tmp/uccl_coord_systemready_small_rank{0,1}.log
```

### README-like 性能与结论

```text
coordinator, chunk=16, forward slots=16, cap=0:
  node0 dispatch: 23-24 GB/s, ~2.52-2.60 ms
  node1 dispatch: 22 GB/s,    ~2.72-2.80 ms

coordinator, chunk=4, forward slots=3, cap=0:
  both nodes dispatch: ~20 GB/s, ~3.02-3.07 ms
```

日志：

```text
/tmp/uccl_coord_systemready_c16_s16_rank{0,1}.log
/tmp/uccl_coord_systemready_c4_s3_rank{0,1}.log
```

结论：

- 大 WRITE 相比 coordinator 的小 WRITE 有收益，但额外 coordinator warp、
  system-scope ready 发布和生命周期 drain 的成本远大于收益；
- 当前 V2 内核中直接照搬 V1 producer/coordinator 角色不能接近 V1 `~59 GB/s`；
- 按“低于基线就回退”规则，已删除 coordinator、forward slots 开关和配套 proxy
  quiet 改动，恢复原 `chunk=4 / slots=3` 主路径；
- 本轮无收益代码不 commit，只保留实验记录和设计约束。

### 操作与清理

- 一次测试误用了不可达的 `MASTER_ADDR=172.31.78.36`；实际管理网地址为
  `172.31.70.225`，之后已恢复。
- 每次失败后同时清理 `test_ep.py`、`spawn_main` 进程组并确认两台 GPU 无残留。
- 服务器下一步同步回退主路径，重建默认 `GIN_MAX_INFLIGHT_NORMAL=8`，确认
  `37-38 GB/s` 基线恢复。

### 回退后最终验证

回退源码后必须同时重建两类二进制：

```text
1. ep/ host proxy extension，恢复 GIN_MAX_INFLIGHT_NORMAL=8；
2. DeepEP deep_ep/_C 扩展，恢复 hybrid_dispatch_impl 的 16 参数模板 ABI。
```

第一次回退验证因 `_C` 仍保留 coordinator 版本的 17 参数生成逻辑而 NVCC 失败：

```text
/tmp/uccl_post_revert_baseline_rank{0,1}.log
```

重建并同步 `_C` 后，README-like EP8x2 两端 `RC=0`，基线完全恢复：

```text
dispatch:          37-38 GB/s, 1.60-1.64 ms
expanded dispatch: 38 GB/s,    1.61-1.63 ms
cached dispatch:   38 GB/s,    1.60-1.63 ms

logs:
  /tmp/uccl_post_revert_baseline2_rank0.log
  /tmp/uccl_post_revert_baseline2_rank1.log
```

最终检查两台机器均无 `test_ep.py` / `spawn_main` 残留，`nvidia-smi` compute process
为空。

## 2026-06-09 per-rail post->CQE profile 与 V1 queue 映射恢复

### 假设与 profile 实现

为了区分 4-token WRITE 的长 completion 延迟来自少数慢 NIC/QP，还是 EFA 小包路径的
普遍成本，在 proxy 已有 command profile 上增加独立开关：

```text
UCCL_PROXY_PROFILE_RAILS=1
```

它只在已有 WRITE post 与 CQE completion 点按 `(proxy thread, ring/QP)` 记录：

```text
posted WRITE WRs / bytes
completed WRITE WRs
post -> CQE average / max latency
```

没有改变 wire、ordering、CQ progress 或 D2H queue 行为。日志：

```text
/tmp/uccl_rail_profile_rank0.log
/tmp/uccl_rail_profile_rank1.log
```

### 原映射的 profile 结果

同一 proxy 内 8 个 QP 的负载与 CQE latency 基本均衡，但 4 个 proxy thread 出现稳定
的 `3:2` 偏载：

```text
node0 bytes:
  thread 0/1: 62.51 / 62.53 GB
  thread 2/3: 41.69 / 41.58 GB

node1 bytes:
  thread 0/1: 62.51 / 62.53 GB
  thread 2/3: 41.69 / 41.60 GB

all QP post->CQE average:
  node0: 87.3-111.1 us, overall 96.2 us
  node1: 83.1-102.6 us, overall 92.1 us
```

根因来自 queue 数组是 proxy-major，而当前 `UCCLGin::lane()` 直接执行
`channel_idx % num_queues`。README 配置有 80 个 channel、32 个 queue，因此前 16 个
queue 各承载 3 个 channel、后 16 个各承载 2 个；前两个 proxy 恰好拿到全部重载
queue。

原 V1 `uccl_ibgda.cuh` 已有正确映射：

```text
logical channel
  -> thread_idx = logical % num_proxy_threads
  -> queue_in_proxy = logical / num_proxy_threads
  -> physical queue = thread_idx * queues_per_proxy + queue_in_proxy
```

因此恢复同一映射，没有新增策略。修改后 profile 日志：

```text
/tmp/uccl_v1_queue_map_profile_rank0.log
/tmp/uccl_v1_queue_map_profile_rank1.log
```

4 个 proxy thread 的累计 bytes 变为：

```text
node0: 52.095 / 52.087 / 52.056 / 52.072 GB
node1: 52.100 / 52.092 / 52.062 / 52.075 GB
```

即 thread 层偏载被完全消除。单个 QP 仍不可避免地承载 2 或 3 个 channel，但它们属于
同一均衡 proxy/NIC 工作集，不再形成 proxy thread 的 `3:2` 偏载。

### profiling-off README-like EP8x2 验证

命令使用：

```text
--num-processes 8 --test-first-only --num-sms 20
--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256
--ignore-local-traffic
```

日志：

```text
/tmp/uccl_v1_queue_map_clean_rank0.log
/tmp/uccl_v1_queue_map_clean_rank1.log
```

两节点 correctness 全过，平均 kernel 时间：

```text
                         old baseline       V1 queue map
node0 dispatch           1629.4 us          1622.5 us
node1 dispatch           1606.5 us          1599.8 us
node0 expanded           1622.4 us          1607.8 us
node1 expanded           1618.5 us          1589.3 us
node0 combine            4047.4 us          3892.9 us
node1 combine            4125.0 us          3998.1 us
node0 reduced combine    3771.8 us          3617.9 us
node1 reduced combine    3785.5 us          3587.6 us
```

结论：

- 保留 V1 queue 映射恢复：它修复确定的 proxy 偏载，dispatch 小幅改善，combine /
  reduced combine 达到约 `3-5%` 改善。
- 排除“少数慢 rail/QP 导致 dispatch 4-token completion 长尾”：约 `90-100 us`
  post->CQE latency 在全部 proxy/NIC/QP 上普遍存在。
- 下一步 dispatch 不再调 channel->rail mapping；转向 receiver forward pipeline，
  分解 tail 可见后到 metadata-ready、payload load、scaleup store 的等待与 overlap。
- 测试结束后两台机器均无 `test_ep.py` / `spawn_main` 或 GPU compute process 残留。

## 2026-06-09: V1/V2 dispatch channel 与 warp 组织对照

低扰动 FIFO profile 排除 device->proxy push 差距后，继续逐行对照 V1
`ep/src/internode.cu` 与 V2 `hybrid_dispatch.cuh`，定位到此前计划低估的核心差异。

### V1

V1 host/kernel 事实：

```text
num_channels = config.num_sms / 2
dispatch launch blocks = num_channels * 2
kNumDispatchRDMASenderWarps = 7
NUM_MAX_NVL_PEERS = 8
```

README-like `num_sms=20` 对应：

```text
10 network channels

每个 sender channel block:
  7 x kRDMASender
  1 x kRDMASenderCoordinator
  8 x kNVLReceivers

每个 forward channel block:
  8 x kRDMAAndNVLForwarder
  1 x kForwarderCoordinator
```

sender coordinator 观察 7 个 producer 的 ready tail，默认每次发最多 16 token；
receiver 侧 8 个 forward warp 可以并行吸收到达的 chunk。

### 当前 V2

V2 host/kernel 事实：

```text
num_scaleout_warps = num_channels_per_sm
num_forward_warps = num_channels_per_sm
num_channels = num_sms * num_channels_per_sm
```

README-like 配置实测 `num_channels_per_sm=4`，对应：

```text
80 network channels

每个 channel:
  1 x scaleout warp
  1 x forward warp
```

V1/V2 的总 producer/forward warp 数接近：

```text
V1 producer:  10 channels * 7 = 70 warps
V1 forward:   10 channels * 8 = 80 warps
V2 producer:  80 channels * 1 = 80 warps
V2 forward:   80 channels * 1 = 80 warps
```

但并行度组织完全不同。V1 把 warp 聚合到少量 channel，能快速形成 16-token chunk，
也能用多个 forward warp 吸收 burst；V2 把相同 warp 数摊成大量单-warp stream，
因此 chunk=16 会让单 forward warp 出现持续气泡，chunk=4 才是当前平衡点。

### 修正后的根因与开发方向

- V1 的 per-token epoch tag 主要保证 payload freshness；receiver 仍以 tail/chunk
  获得可消费范围。它不是 V1 能用大 WRITE 的唯一原因。
- “给当前 V2 单 forward warp 增加 ready/tag，然后直接把 chunk 改 16”不足以解决
  burst 消费能力。
- “额外增加一个 coordinator warp”此前已实测更慢，因为它没有同时恢复 V1 的
  multi-producer / multi-forward channel 组织，还引入了 system-scope ready 成本。
- 下一阶段应设计 **V2-native multi-warp channel grouping**：
  - 保留 V2 compact send scratch、expanded receiver layout、handle 和 combine replay；
  - 将每 SM 的多个 scaleout/forward warp 聚合到更少的 network channel；
  - 每 channel 使用多个 producer、轻量 coordinator、多个 forward consumer；
  - producer/coordinator ordering 优先精确复用 V1 release/acquire + FIFO
    system-release 机制，不新增每 chunk system atomic；
  - 多 forward warp 必须重新设计 metadata index、linked-list tail 和 slot ownership。

### 恢复 V1 映射后的 dispatch phase profile

使用：

```text
UCCL_GIN_DISPATCH_CLOCK_PROFILE=1
```

日志：

```text
/tmp/uccl_v1_map_dispatch_clock_rank0.log
/tmp/uccl_v1_map_dispatch_clock_rank1.log
```

该 profile 每个事件执行 global atomic，令 dispatch headline 降至约 `10 GB/s`，因此
cycle 不能相加成 wall-time budget，只用于比较阶段量级。两节点累计结果：

```text
scaleout D2H push             76.1k cycles/event
scaleout tail push            13.1k cycles/event
forward tail wait             41.2k cycles/event
  stalled event fraction      24.6%
  stalled event average       165.7k cycles/event
forward metadata-ready wait    2.2k cycles/event
forward payload TMA load      14.1k cycles/event
forward scaleup store          0.5k cycles/event

fresh-read after a tail stall:
  selected source ready       44.0%
  another source ready        46.4%
  no source ready              9.6%
```

结论：

- receiver 的 per-token metadata-ready 检查不是当前主要成本；它比 stalled tail wait
  小约两个数量级。
- sender D2H emission 与 receiver tail visibility/source-selection 仍是主要候选。
- “另一个 source ready”仍很常见，但 naive ready-source-first 已经实测无 wall-time
  收益，所以后续不能只改 source selection。

### 8 proxy threads 重测与回退

旧的 8-proxy 负结果受到错误 queue 映射污染，因此在恢复 V1 mapping 后重新构建：

```text
NUM_PROXY_THREADS=8
CHANNEL_PER_PROXY=8
GIN_MAX_INFLIGHT_NORMAL=8
```

profiling-off README-like EP8x2 日志：

```text
/tmp/uccl_v1_map_8proxy_clean_rank0.log
/tmp/uccl_v1_map_8proxy_clean_rank1.log
```

对比保留的 4-proxy V1 mapping：

```text
                         4 proxy             8 proxy
node0 dispatch           1622.5 us           1620.0 us
node1 dispatch           1599.8 us           1647.3 us
node0 expanded           1607.8 us           1624.1 us
node1 expanded           1589.3 us           1654.4 us
node0 cached             1610.4 us           1636.3 us
node1 cached             1602.6 us           1657.1 us
```

8 proxy 未改善 dispatch，rank1 反而稳定回退；combine/reduced 的额外收益也低于
`3%` 保留门槛。结论是当前普遍的 post->CQE 延迟不是 proxy/QP 数不足造成，更多
proxy 会增加 CPU/CQ 竞争。实验配置已回退，服务器恢复 4 proxy 默认构建。

## 2026-06-08: dispatch 首包 4 / steady 16 自适应 chunk 实验

### 动机与改动

为了区分“大 chunk 慢”究竟来自首包启动气泡，还是持续的 receiver 可见性气泡，
临时把 dispatch compact chunk 改为：

```text
每个 channel 的第一个 remote batch: 4 token
之后的 steady remote batch:          16 token
```

该实验不新增 buffer，不改变 receiver、proxy、tail 或 V2 layout，只调整
`hybrid_dispatch.cuh` 中 compact batch flush 阈值。

### README-like EP8x2 结果

命令使用标准第一组配置：

```text
--num-processes 8 --test-first-only --num-sms 20
--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256
--ignore-local-traffic
```

日志：

```text
/tmp/uccl_adaptive_chunk_4_16_rank0.log
/tmp/uccl_adaptive_chunk_4_16_rank1.log
```

两节点 correctness 全过，但性能明显回退：

```text
                         fixed chunk=4       first=4, steady=16
node0 dispatch           1622.5 us           2064-2079 us
node1 dispatch           1599.8 us           1995-2011 us
node0 expanded           1607.8 us           2043-2058 us
node1 expanded           1589.3 us           2012-2028 us
node0 cached             1610.4 us           1999-2010 us
node1 cached             1602.6 us           2000-2014 us
```

结论：

- 小首包不足以让后续 16-token chunk 获得大包收益；steady 阶段仍持续制造
  receiver count/tail 可见性气泡。
- 当前 chunk=4 最优不是单纯的 startup latency 偶然结果，而是现有
  scaleout/forward pipeline 对持续细粒度可见性的要求。
- 若要使用 V1 式 16/32-token 大 WRITE，必须先解耦 producer 与网络 coordinator，
  并给 receiver 引入 per-token ready/tag 或等价机制；只调 flush threshold 不成立。
- 实验代码已本地和服务器回退到固定 chunk=4。测试结束后两台机器无残留
  `test_ep.py` / `spawn_main` 或 GPU compute process。

## 2026-06-09: 低扰动 dispatch FIFO / forward 联合采样

### 目标

全量 dispatch clock profile 会明显扰动 kernel。本轮新增
`DEEPEP_UCCL_GIN_DISPATCH_SAMPLE_PROFILE`，只采样每 16 个 channel 中的一个，
并在 warp 结束时一次性打印：

```text
sender:
  push events / cycles
  push 前 FIFO inflight sum / max
  push 前是否已达到 FIFO capacity

receiver:
  forward tail-wait events / cycles
```

profiling 关闭时不进入主路径。采样开启后 headline 仍保持约 `37-38 GB/s`，说明
扰动远小于旧全量 clock profile。

### 编译时发现：当前 hot path 是 MSCCL++ FIFO，不是旧 ring

第一次采样直接读取 `D2HHandle::head()/tail()`，NVCC 编译失败：

```text
/tmp/uccl_dispatch_sample_rank0.log
/tmp/uccl_dispatch_sample_rank1.log

d2hq::D2HHandle has no member head/tail
```

原因是 `ep/include/common.hpp` 默认定义 `USE_MSCCLPP_FIFO_BACKEND`。当前
`D2HHandle::atomic_set_and_commit` 实际调用：

```text
mscclpp::FifoDeviceHandle::push(trigger, maxSpinCount=-1)
```

因此旧 ring 的 `kUCCLGinMaxInflightNormal=8` 不作用于当前 device hot path。
采样随后改为读取 FIFO 自己使用的 `head`、`tailCache` 和 `size`。

### README-like EP8x2 结果

日志：

```text
/tmp/uccl_dispatch_sample2_rank0.log
/tmp/uccl_dispatch_sample2_rank1.log
```

两节点 correctness 全过，性能约 `37-38 GB/s`。采样聚合：

```text
node0:
  push events:              194007
  push avg:                 39425 cycles
  initial FIFO inflight:    1023.8 average
  at FIFO capacity:         102 events / 0.053%
  observed max:             2049
  forward tail-wait avg:    33828 cycles/event

node1:
  push events:              193727
  push avg:                 38552 cycles
  initial FIFO inflight:    1027.5 average
  at FIFO capacity:         108 events / 0.056%
  observed max:             2049
  forward tail-wait avg:    50364 cycles/event
```

`initial inflight` 使用和 `FifoDeviceHandle::push` 相同的 cached-tail 口径，因此它
可能比 host 当前真实 tail 更旧；但这正是判断 push 是否会进入 FIFO `sync()` 的正确
口径。

### 结论

- FIFO 平均约半满，但几乎从不达到 capacity；当前 V2 不是“灌到 2048 后撞墙硬等”。
- push 平均 `38-39k cycles`，与 V1 apples-to-apples profile 的约
  `38k cycles/event` 接近；device->proxy FIFO push 不是 V1/V2 gap 的主因。
- 当前差距主要仍来自 V2 的命令数量：V2 每 4 token 一次 push，而 V1 默认每
  16 token 一次；以及 V2 receiver forward 每 token 执行的 metadata/TMA/top-k/slot
  语义处理。
- 直接增大 chunk 和独立 coordinator 已被负实验否定。下一步应优先 profile/优化
  receiver forward per-token pipeline，再重试大 WRITE；不再继续调 FIFO cap/ring
  depth。
- 测试结束后两台机器均无 `test_ep.py` / `spawn_main` 或 GPU compute process 残留。
