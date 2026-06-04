# Worklog

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
