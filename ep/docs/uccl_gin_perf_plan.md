# UCCL-GIN V2 性能计划

本文档描述当前 DeepEP V2 UCCL-GIN 在 AWS EFA 上的性能现状、已完成实验和下一阶段
优化顺序。当前主线已经同时覆盖 **dispatch 与 combine**，不再只讨论 dispatch。

计划只接受能由 profile 区分、由完整 correctness 验证、由 README-like benchmark
量化的优化。不要为了“看起来像 V1”而直接照搬机制；先确认它解决的是当前 V2
critical path。

## 1. 当前范围与目标

当前实现与 benchmark 范围:

```text
硬件:        2 x p5en.48xlarge, each 8 x H200 + 16 EFA NIC
拓扑:        EP8x2 / EP16, kNumScaleoutRanks=2
配置:        tokens=8192, hidden=7168, topk=8, experts=256, num_sms=20
transport:   UCCL-GIN EFA normal-mode proxy
CUDA:        13.0
```

当前 compact dispatch 仍使用:

```cpp
remote_scaleout_rank_idx = scaleout_rank_idx ^ 1;
```

所以当前性能计划先优化 EP8x2。泛化到 `kNumScaleoutRanks > 2` 是独立的功能工作，不能
与当前性能数字混为一谈。

阶段目标:

```text
第一目标:
  dispatch 接近 V1 8192-token FP8 baseline: ~59 GB/s SO
  combine 至少追平当前 dispatch:            >= 38 GB/s SO

长期目标:
  在不改变 DeepEP V2 layout/handle 语义的前提下，继续逼近 EFA 整机有效带宽。
  90 GB/s 是 CX7/IB 官方结果，可作为长期方向，但不能未经测量地当作当前 EFA
  proxy 路径的单 rank 必达验收线。
```

贯穿性假设（dispatch 当前最高优先级）:

```text
当前 dispatch 已排除 inflight cap、ring 深度、finish dependency、receiver reorder
深度是单独主因。本轮 PT.0 把关键路径拆开后，结论更具体:

  - CPU proxy post 不慢: WRITE `ibv_post_send` 热区约 0.42 us/WR。
  - receiver atomic apply 不慢: receiver CQE -> ordered fetch_add 约 133 ns/CQE。
  - 4-token WRITE 的 post->CQE 平均约 64-67 us/WR。
  - GPU scaleout D2H push 在 clock profile 下约 518-525k cycles/4-token chunk。
  - GPU forward tail wait 在 clock profile 下约 169-181k cycles/event。
  - forward stall 后 fresh tail read 显示 53-55% 是“selected source 没 ready，
    但其他 source 已 ready”，存在明显 source-selection HOL。

所以当前 dispatch gap 更像是【V2 将 producer/forward 并行度摊到大量单-warp
channel，导致每条 stream 只能用 4-token WRITE 保持流水；EFA delivery 与单 forward
warp 的 burst 消费成为主路径】；不是 V2 proxy CPU 每 command 慢几十倍，也不是
receiver atomic apply 慢。

进一步对照 V1 代码后，V1 更快的首要结构原因已经明确：

```text
V1, num_sms=20:
  num_channels = num_sms / 2 = 10
  each channel:
    sender block:    7 producer warps + 1 sender coordinator
    forwarder block: 8 forward warps + 1 forward coordinator

V2, num_sms=20, channels_per_sm=4:
  num_channels = 80
  each channel:
    1 scaleout warp + 1 forward warp
```

两者总 producer/forward warp 数接近，但 V1 把并行度聚合到少量 channel：producer
能快速攒够 16-token chunk，receiver 又能用 8 个 forward warp 吸收大 WRITE burst。
V2 将并行度拆成 80 条单-warp stream；单 forward warp 的 per-token
metadata/TMA/top-k/slot 工作无法快速吸收 16-token burst，所以实测必须用 4-token
chunk 保持平滑流水。per-token ready/tag 仍是 correctness/landing 机制，但不能单独
解释或修复当前性能差距。
```

## 2. 已验证基线

### 2.1 当前 UCCL-GIN V2

README-like EP8x2 完整 correctness，两端均退出 `0`:

```text
dispatch:          37-38 GB/s SO, 1.60-1.64 ms
expanded dispatch: 37-38 GB/s SO, 1.61-1.64 ms
cached dispatch:   37-38 GB/s SO, 1.60-1.63 ms
combine:           28-30 GB/s SO, 3.96-4.16 ms
reduced combine:   ~31 GB/s SO, 3.75-3.82 ms
```

日志:

```text
/tmp/uccl_gin_combine_followup_rank0.log
/tmp/uccl_gin_combine_followup_rank1.log
```

### 2.2 V1 UCCL-EP apples-to-apples baseline

V1 commit `495b7221d084cce92553d6a038376358bd218a5a`，8192-token FP8 dispatch:

```text
~59 GB/s RDMA
```

这证明同一套 EFA/UCCL transport substrate 在类似负载下可以明显高于当前 V2
dispatch。它是当前第一阶段的有效对标值，但不能直接要求 V2 完全复刻 V1 的 buffer
layout 和 receiver 语义。

V1 chunk 事实:

```text
test_internode.py baseline config:
  Config(num_sms, nvl_chunk_send=8, nvl_chunk_recv=512,
         rdma_chunk_send=16, rdma_chunk_recv=512)

dispatch tuning sweep:
  rdma_chunk_size in range(4, 33, 4)
```

所以当前 apples-to-apples V1 baseline 不是固定“32 token 一次”；`32` 是论文/某些
HT 模式或 sweep 上界中的候选值。当前 V1 默认 dispatch config 是 `16` token RDMA
chunk。V2 UCCL-GIN 的 `4` token 最优也不是因为 EFA 喜欢 4 token，而是当前 V2
compact send buffer、forward 每轮消费粒度、TMA/receiver overlap 和 piggyback tail
共同作用后的实测 optimum。若 forward/receiver 调度改变，必须重跑 chunk sweep。

V1 RDMA dispatch 过程（`ep/src/internode.cu`）:

```text
producer warps:
  按 dst RDMA rank / channel 把 token 写入 V1 send_buffer[dst][slot]
  更新 rdma_send_channel_tail[dst]

coordinator warp:
  读取某个 dst/channel 已 ready 的连续 token 数
  若 ready >= rdma_chunk_send 或该 dst 已完成:
      num_tokens_to_issue = min(ready, rdma_chunk_send)   # baseline = 16
      nvshmemi_ibgda_put_nbi_warp(
          bytes = num_tokens_to_issue * bytes_per_token,
          atomic_offset = rdma_channel_tail,
          atomic_val = num_tokens_to_issue)

EFA normal path:
  payload WRITE 与 tail add piggyback 在同一个 command/WR 语义中
  receiver 端仍依靠 per-token/source tag 判断 payload readiness
```

这和当前 V2 的差异不是“V1 固定 32、V2 固定 4”这么简单，而是:

```text
V1: receiver-facing staging 已按 dst/channel 连续；同一 channel 有 7 个 producer
    warp + coordinator 发 16-token chunk，并有 8 个 forward warp 吸收 burst；
    ready/tag 负责 payload freshness。
V2: receiver 直接消费 expanded layout。当前 UCCL-GIN 用 compact send scratch 生成
    连续 payload chunk，但每个 channel 只有一个 producer/forward warp；chunk 越大，
    sender 越晚发出，单 forward warp 也越难吸收 burst。
```

### 2.3 独立 GIN / EFA 事实

已有 microbenchmark 证明:

```text
大包、整机 remote-only:
  双向 aggregate ~740 GB/s
  单向每 node ~370 GB/s

单 P2P 流:
  大包约 44 GB/s

DeepEP-like 小消息:
  4 KiB  ~2.8 GB/s
  8 KiB  ~5.1 GB/s
  16 KiB ~8.8 GB/s
  32 KiB ~12.5 GB/s
```

因此不能把 `44 GB/s` 误写成整机天花板；但 message size、并行流数量和 receiver
处理方式确实会显著影响 EFA 有效吞吐。

## 3. 当前主路径

### 3.1 Dispatch

```text
GPU scaleout warp
  -> 把 remote token TMA store 到 per-channel compact send slots
  -> 每 4 token 发一个 rail_put_tail_add
  -> 一条 WRITE_WITH_IMM 同时携带 payload 与 count delta
  -> channel 结束时发 standalone finish ATOMIC

CPU proxy / EFA
  -> post compact payload WRITE_WITH_IMM
  -> receiver CQE 按 sequence 重排并 apply count
  -> standalone finish 只依赖此前 plain WRITE

GPU forward warp
  -> poll count/finish tail
  -> 每轮最多消费 kNumSlotsPerForwardChunk=3 个 slot
  -> TMA load payload 并转发到 scale-up layout
```

关键现状:

```text
kUCCLGinCompactChunkTokens = 4
kNumSlotsPerForwardChunk   = 3
GIN_MAX_INFLIGHT_NORMAL    = 8
ordered sequence states    = fixed arrays, not unordered_map
```

### 3.2 Combine

```text
GPU combine forward warp
  -> replay token_metadata_at_forward
  -> reduce / TMA store 到 V2 scaleout send buffer
  -> 每个 remote token 立即 gin.put<Rail>
  -> 每 channel/source 最后发 standalone finish ATOMIC

CPU proxy / EFA
  -> plain payload WRITE 进入 finish dependency tracking
  -> finish 等这些 WRITE CQE 后发送

GPU receiver
  -> 等每个来源 finish
  -> 清 tail 后退出
```

Combine 当前没有 dispatch 的 compact staging 或 piggyback count。`AggregateRequests`
参数在 UCCL-GIN handle 中仍未实现，这很可能是 combine `28-30 GB/s` 低于 dispatch
`37-38 GB/s` 的主要结构性原因之一，但必须先测连续性和 merge opportunity。

## 4. 已完成或已降级方向

以下结论已经由实验支持，不应反复当作首要优化:

### 4.1 Dispatch compact chunk sweep 已完成

```text
tokens/chunk   cached dispatch SO BW   dispatch_impl
64             ~27 GB/s                2.23-2.31 ms
32             ~31-32 GB/s             1.93-2.00 ms
16             ~33-34 GB/s             1.79-1.83 ms
8              ~35-36 GB/s             1.70-1.73 ms
4              ~37-38 GB/s             1.59-1.64 ms
2              ~32 GB/s                1.90-1.93 ms
```

结论:

- 4-token 是当前配置的最佳 streaming/message-size 平衡点。
- “继续把 dispatch chunk 加到 32/64”已证实会损失 overlap。
- 未来只有在 forward 调度或 transport 行为变化后，才需要重跑 sweep。

补充实验：尝试“首个 chunk=4，后续 steady chunk=16”，希望先用小首包唤醒
receiver，再用大包提高 EFA 效率。README-like EP8x2 correctness 通过，但
dispatch 从约 `1.60-1.62 ms` 回退到 `2.00-2.08 ms`，约慢 `24%`：

```text
/tmp/uccl_adaptive_chunk_4_16_rank0.log
/tmp/uccl_adaptive_chunk_4_16_rank1.log
```

因此问题不只是首包启动延迟。当前 forward pipeline 需要持续的细粒度 count
可见性；在不改变 receiver 消费协议时，后续 16-token burst 仍会产生明显气泡。
该实验已回退。后续若要恢复 16/32-token WRITE，必须先实现 producer/coordinator
解耦与 receiver 侧 ready/tag 或等价的流式可见机制，不能只调发送阈值。

### 4.2 Compact32 未打满不是 bug

旧 profile 中平均 `25.49 tokens/WRITE` 来自每 channel 多个 full chunk 加最后 partial
chunk；`non-contiguous flushes=0`。该问题已通过 chunk/flush-reason profile 澄清。

### 4.3 Sender dependency 已收窄

当前只有 `atomic_val == 0` 的 plain WRITE 进入 standalone finish dependency。
piggyback WRITE_WITH_IMM 与 finish 共享 per-tail sequence，不再作为 sender-side
completion dependency。

结果:

```text
dependency_max:        72 -> 2
dispatch gain:         about 2-4%
```

继续微调 dependency container 不是 dispatch 首要方向；但它仍是当前 uncompacted
combine 的重要成本。

### 4.4 Proxy thread、ring size、inflight cap 已降级

已验证:

```text
8 proxy threads:       无明显改善
queue 2048 -> 4096:    无明显改善
V1-style inflight=8:   correctness pass, 性能仍约 37-38 GB/s
```

所以“GPU 灌满 2048 ring”不是当前 dispatch gap 的直接主因。cap=8 保留用于 sequence
安全和可控背压，不应被描述成性能解法。

补充澄清：当前构建启用了 `USE_MSCCLPP_FIFO_BACKEND`，device hot path 实际进入
`mscclpp::FifoDeviceHandle::push`，不是旧 `RingBuffer::atomic_set_and_commit`。
FIFO 容量仍为 2048，但旧 ring 的 `kUCCLGinMaxInflightNormal=8` 不限制这个路径。
低扰动采样结果：

```text
node0: push avg 39.4k cycles, initial FIFO inflight avg 1023.8,
       at-cap 0.053%, max 2049
node1: push avg 38.6k cycles, initial FIFO inflight avg 1027.5,
       at-cap 0.056%, max 2049

logs:
  /tmp/uccl_dispatch_sample2_rank0.log
  /tmp/uccl_dispatch_sample2_rank1.log
```

FIFO 平均约半满，但几乎从不因容量耗尽进入 `sync()`。push 的 `38-39k cycles` 又与
V1 profile 的约 `38k cycles/event` 接近，因此 device->proxy FIFO push 不是 V1/V2
性能差距的主因。后续不再用“撞 2048 ring”解释 V2 dispatch gap。

### 4.5 Receiver reorder 不是主瓶颈

4-token 代表性数据:

```text
receiver_atomic_cqes:          120528
receiver_atomic_in_order:      115181
receiver_atomic_buffered:        5347
receiver_atomic_max_buffered:        5
```

绝大多数 CQE 按序，乱序深度浅。固定数组优化已完成；继续优化 reorder lookup 预计收益
有限。

### 4.6 `rail_is_combine` 不参与当前性能路径

normal-mode ordered atomic 将 legacy `is_combine` bit 复用为 `seq[3]`。当前 receiver
只读取 sequence，WRITE receiver 也不读取 phase。该字段已删除，不得把 phase bit 当作
combine 性能或 correctness 机制。

### 4.7 不加入 host-side per-iteration quiet

当前 combine receiver 必须观察所有 finish 已 apply，随后清 tail，kernel 才退出。
finish 又依赖此前 plain payload WRITE。同步执行模型下不存在旧 finish 在下一轮 clear
后才落地的问题。粗粒度 host quiet 只会串行化 transport。

若未来支持跨 iteration async pipeline，应使用 epoch/double buffer 重新设计，而不是
补全局 quiet。

## 5. 当前最重要的未知量

### 5.1 Dispatch 的剩余时间花在哪里

本轮 PT.0 重新测了当前 `kUCCLGinCompactChunkTokens=4` 主路径。日志:

```text
/tmp/uccl_gin_pt0_proxy2_rank0.log
/tmp/uccl_gin_pt0_proxy2_rank1.log
/tmp/uccl_gin_pt0_clock_rank0.log
/tmp/uccl_gin_pt0_clock_rank1.log
/tmp/uccl_gin_hol_proxy_rank0.log
/tmp/uccl_gin_hol_proxy_rank1.log
/tmp/uccl_gin_hol_clock2_rank0.log
/tmp/uccl_gin_hol_clock2_rank1.log
```

chunk/flush profile（每 node，取每 rank 最后一条）:

```text
rank0 node:
  chunks=16616, tokens=65291, tokens/chunk=3.929
  bin_1=98, bin_2=359, bin_3_4=16159
  flush_full=15998, flush_finish=618, flush_noncontig=0

rank1 node:
  chunks=16623, tokens=65241, tokens/chunk=3.925
  bin_1=123, bin_2=379, bin_3_4=16121
  flush_full=15997, flush_finish=626, flush_noncontig=0
```

所以当前代码不是“没凑满 4”；它几乎全部都是 full 4-token chunk。无法直接把
4-token chunk 提到 16/32 而不改变 receiver 可见性/forward pipeline。

proxy 侧直接拆分:

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

这撤销了旧的“V2 proxy post hot path 约 8us/cmd 是主因”的结论。旧 `post_gpu_us`
包含空轮询、整轮 mixed loop、profile 扰动和非对称口径；不能再作为 active
per-command CPU 成本。新增 receiver-side timing 又排除了“receiver CQE apply 很慢”
这个解释；ordered sequence / fetch_add 本身是百纳秒量级。

GPU clock profile（侵入式，只看相对量级，不看 headline BW；`clock2` 修复了 CUDA
device `printf` 参数上限导致的最后一个 counter 错读问题）:

```text
rank0 node:
  scaleout_d2h:       525.1k cycles / 4-token WRITE
  forward_tail_wait:  168.6k cycles / event
  forward stall split after one fresh tail read:
    selected source ready: 32.0%
    other source ready:    53.0%
    no source ready:       15.0%

rank1 node:
  scaleout_d2h:       517.6k cycles / 4-token WRITE
  forward_tail_wait:  180.5k cycles / event
  forward stall split after one fresh tail read:
    selected source ready: 27.9%
    other source ready:    55.3%
    no source ready:       16.9%
```

这说明:

- `scaleout_d2h` 高延迟不是 CPU post 慢，而主要覆盖 GPU 等待 ring/proxy/EFA
  delivery 的下游效果；
- `forward_tail_wait` 的一半以上 stall 在 fresh read 后发现“别的 source 已 ready”，
  说明 source-selection HOL 是真实优化机会；
- 只有约 15-17% stall 是 fresh read 后所有 source 都不 ready，不能把 forward wait
  全部归因于 receiver apply 或 EFA delivery；
- V2 `4` token 最优的合理解释是:4-token 在 EFA 小包成本与 forward 首 token 可见性
  之间取得了当前最好的 pipeline 平衡。16/32 token 减少 WR，但会推迟 count/tail
  update，使 forward 更晚启动消费。

### 5.2 Combine 的 4 ms 花在哪里

首轮 combine merge-opportunity profile 已完成。README-like EP8x2 的两个实际 JIT
实例 (`expanded=0/1, kAllowMultipleReduction=true`) 都得到:

```text
remote puts / kernel:       ~8155
same dst transition:        100%
local pointer contiguous:   66.54%
remote pointer contiguous:  0%
both contiguous:            0%
run length:                 100% are 1 token
```

这不是 proxy coalescing 没实现，而是当前 V2 replay 顺序与 receiver layout 的结构性
结果:

```text
local send slot:
  scaleout_send_buffer.get_token_buffer(i)
  常随 replay i 前进，所以约 2/3 transition 连续

remote receive slot:
  scaleout_recv_buffer.get_rank_buffer(...).get_token_buffer(src_token_idx)
  由 token_metadata_at_forward 回放的 src_token_idx 决定，在当前 emission order 中
  每次都跳跃
```

EFA RDMA WRITE 不能把一个连续 local range scatter 到多个不连续 remote slot。因此:

- proxy 不能仅靠合并相邻 `TransferCmd` 减少 combine payload WR；
- `ncclGinOptFlagsAggregateRequests` 也不能把这些 WR 合成一个普通 RDMA WRITE；
- P2 的 direct contiguous multi-token WRITE 已被当前数据否定；
- 若要减少 combine WR 数，必须改变 GPU emission order、引入 scatter/gather transport，
  或改变 receiver-facing staging/layout。前两者需要先证明不破坏 V2 replay/reduce
  语义；最后一种不能未经设计退回 V1 packed layout。

首轮 clock profile 每个 token/transition 都执行 host-mapped atomic，combine headline
从约 `4 ms` 膨胀到 `132-168 ms`。所以它只能支持上述结构性结论，不能用于阶段耗时
占比。当前仍未知:

- 正常运行中 scale-up wait、reduce/TMA、D2H push、finish dependency、scale-out
  finish wait 各占多少；
- finish dependency 从 enqueue 到 post 的实际等待时间；
- reduced combine 比普通 combine 快约 `0.3 ms` 的具体来源。

### 5.3 贯穿性嫌疑：细粒度 command path（dispatch 与 combine 共有）

把 5.1 / 5.2 的数据放在一起看，dispatch 与 combine 都受细粒度 command path 影响，
但 dispatch 目前已经拆得更清楚：

```text
dispatch: proxy CPU post 约 0.42 us/WR；
          WRITE post->CQE 约 64-67 us/WR；
          GPU scaleout D2H push 约 291-309k cycles/4-token WRITE；
          forward tail wait 约 83-90k cycles/event。
combine:  per-token D2H push 约 62k cycles ≈ 31 us / event；
          finish dependency 仅约 3% / 0.12 ms；cap=8 与 cap=0 同样约 4.17 ms。
V1:       同一套 UCCL/EFA substrate，8192-token dispatch ~59 GB/s。
```

已经被实验排除、不再是首要原因的项（见 4.4 / 4.5 / P1.3 / P2.1）：

```text
- inflight cap / ring 深度       （cap=0 无改善）
- finish dependency 容器          （仅 ~3%）
- receiver reorder lookup 深度    （多数 CQE 按序）
- proxy 侧合并相邻 TransferCmd    （remote_contig=0%）
- bounded emission reorder        （W4/8/16/32 仍 remote_contig=0%）
```

因此“GPU 观察到每条 D2H push 百微秒级 stall”是下游症状。PT.0 已经排除
`proxy CPU/cmd` 是 dispatch 主因；剩余关键缺口是把 `post->CQE` 再细分为 NIC/rail
小包 delivery、receiver CQE apply、payload 到 HBM 可见和 forward source scheduling：

```text
(a) ring residency   GPU push -> proxy 取出该 command 的等待
(b) proxy CPU/cmd    proxy 取出 -> ibv_post_send 返回
                     本轮 dispatch 已测: 约 0.42 us/WR，不是主因
(c) EFA delivery     ibv_post_send -> send CQE
                     本轮 dispatch 已测: 约 64-67 us/WR，是主要嫌疑之一
(d) receiver apply   receiver CQE -> tail/count 对 forward warp 可见
                     需要继续拆 receiver CQE apply 与 GPU tail wait 的关系
```

V1 之所以更快，首要结构差异是 **multi-warp channel grouping**：少量 channel 上有
多个 producer、独立 coordinator 和多个 forward warp，因此 baseline 16-token
chunk 的形成与消费都足够快。receiver-facing staging/ready-tag 保证 freshness，但
不是单独的吞吐来源。当前 V2 若只把 chunk 设大，已经实测会降低 BW；必须在不破坏
V2 layout 的前提下重新聚合 channel 内 producer/forward 并行度，再重新评估
ready/tag/landing。

## 6. 新执行顺序

## P0: 固定基线与 profiling 纪律

每次性能修改前后都运行同一套 README-like EP8x2:

```bash
python thirdparty/DeepEP-v2-d4f41e4/tests/elastic/test_ep.py \
  --num-processes 8 --test-first-only --num-sms 20 \
  --num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 \
  --ignore-local-traffic
```

要求:

- 性能 run 必须 fresh JIT cache。
- 最终结果必须带完整 correctness；`--skip-check` 只允许 profiling。
- headline BW 使用 profiling-off 结果。
- clock profile 侵入性很强，只比较相对 counter，不能比较其 headline BW。
- 同时记录 rank0/rank1，不能只取快节点。
- 共享 EFS 产物只能单节点顺序 build/link。

每轮最少记录:

```text
dispatch / cached dispatch / combine / reduced combine:
  SO BW
  kernel latency

transport:
  WRITE / WRITE_WITH_IMM / ATOMIC command count
  bytes / command
  CQE count
  per-proxy-thread imbalance

correctness:
  exit code
  timeout / CUDA fault / assertion count
  log and JIT cache paths
```

验收门槛:

```text
有效优化: 两次重复 run 的慢节点均提升 >= 5%，且 correctness 全过。
低于 3%: 视为噪声或微优化，除非显著简化代码。
```

## PT: command path 分解与压缩（当前最高优先级）

依据 5.3：dispatch 与 combine 的共同上游嫌疑是 command path 的细粒度开销。PT 的
目标不是先假设“端到端 30-43 us/op”，而是把 GPU-observed D2H stall 拆成 proxy CPU、
EFA delivery、receiver apply 和 GPU ring/backpressure 几段。它在 dispatch 和 combine
之间共享，因此排在所有单边 kernel 微调之前。原 P4（proxy hot-path / rail 利用率）
的候选项并入 PT.2/PT.3，不再单独排到最后。

### PT.0 把 GPU-observed D2H stall 拆成各 hop（dispatch 已完成第一轮）

在不改 wire/layout 的前提下，给 proxy 与 kernel 加默认零开销、可开关的 profile，
分别量 5.3 中的 (a)-(d)。关键是**隔离纯 CPU 与各等待**，不要再用含空轮询的
`post_gpu_us / post_cmds`。

proxy 侧（已有 `dependency_*_ns` 框架，扩展到所有 command，不止 finish）:

```text
(a) ring/backpressure = GPU `atomic_set_and_commit` 进入 -> 返回，配合 head-tail
                        occupancy / spin counter 间接判断；不做跨时钟域相减。
(b) proxy CPU/cmd     = proxy 取出一批 cmd -> `ibv_post_send` 返回，纯 CPU 区间。
(c) EFA delivery      = `ibv_post_send` 返回 -> 对应 send CQE（按 wr_id 配对）。
(d) receiver apply    = receiver CQE 到达 -> 对应 tail/atomic 落入 host buffer。
```

第一版 PT.0 先不改 `TransferCmd` wire，因此不能直接测“GPU commit 时间戳 -> proxy
取出”。它能直接测 (b)/(c)，用 GPU clock profile 间接判断 (a)，再用 receiver CQE
profile 判断 (d)。

dispatch 当前结果:

```text
(b) proxy CPU/cmd:
  WRITE post 约 0.42 us/WR，ATOMIC post 约 0.53 us/WR。
  结论: 不是 dispatch gap 主因。

(c) post->CQE:
  WRITE 约 64-67 us/WR，ATOMIC 约 31 us/WR。
  结论: 4-token payload WRITE 的 delivery/queueing 是主要嫌疑。

(d) receiver apply:
  receiver atomic process 约 133 ns/CQE；
  receiver atomic commit 约 64-65 ns/fetch_add；
  结论: receiver CQE decode / ordered apply 不是 dispatch gap 主因。

(a)/GPU-observed:
  scaleout_d2h 在 clock profile 下约 518-525k cycles/WRITE；
  forward_tail_wait 在 clock profile 下约 169-181k cycles/event；
  forward stall 后 fresh read:
    selected source ready 约 28-32%，other source ready 约 53-55%，
    no source ready 约 15-17%。
  结论: forward source-selection HOL 是明确优化机会；所有 source 都未 ready 的
        事件只占少数，不能把 forward wait 全部归因于 receiver apply。
```

下一步不是继续削薄 CPU `post_send`，而是区分:

```text
1. post->CQE 长是 EFA 小包/rail 并行度问题，还是 proxy 队列里 WR 太细导致的
   NIC completion 延迟；
2. forward_tail_wait 中 source-selection HOL 该如何消除，且消除后是否能提升
   headline BW；
3. 消除 HOL 后，是否能引入 V2-native ready/tag 或 landing 机制，让 payload chunk 做大但不推迟
   receiver 流式消费。
```

kernel 侧（复用 clock-only 框架，注意 5.1 已知的 combine reduce 区间会 subsume D2H，
见 combine 评审结论）:

```text
GPU push 区间 = atomic_set_and_commit 进入 -> 返回（含 ring-full 自旋）
拆出自旋时间：记录“进入时 ring 是否已满 / 自旋圈数”，区分 (a) 与纯 commit fence。
```

PT.0 当前状态:

```text
dispatch: 第一轮完成；(b)/(d) 已排除，(c) 与 forward source-selection HOL 成为主嫌疑。
combine:  仍需同口径拆分；combine clock profile 已知有嵌套/扰动，不能直接复用为
          wall-time budget。
```

后续如果继续 PT.0，应优先做低开销、低扰动的 source/channel HOL 与 NIC/rail 分布
profile，而不是再增加全量 GPU clock atomic。

### PT.1 若 (b) proxy CPU/cmd 偏大 —— 削薄 per-command CPU（dispatch 已降级）

本轮 dispatch profile 显示 `ibv_post_send` 热区只有约 `0.42 us/WR`，所以 PT.1 对
dispatch 暂时降级。它仍可能影响 combine plain WRITE/finish dependency，但不是当前
dispatch `38 -> 59 GB/s` gap 的首要解释。

如果后续 combine 或新 profile 证明 (b) 偏大，再考虑以下候选：

```text
- atomic dependency 跟踪：atomic_dep_by_wr_ / inflight_write_wrs_ 仍是 hash 容器，
  每条 plain WRITE 都 insert，每个 CQE 都 erase/lookup。改成按 ring 的定长环形数组
  或位图（seq buffer 已经这么做过，复用同一模式）。
- 批量 drain：一次 drain 多条 D2H command 再统一 ibv_post_send / 统一打点，
  摊薄 per-command 的函数调用与 cache miss（原 P4 第一条）。
- imm 打包与 seq 分配：确认 PackAtomicWithSeq / take_next_atomic_seq 不在热路径里
  做多余分支；普通 payload WRITE 若不需要 seq，不要走 atomic 打包路径。
- flush_writes / coalesce_atomic_batch 的扫描：确认它们是 O(batch) 而不是 O(全表)。
```

每改一项都要用 PT.0 的 (b) 单独验证它确实下降，再看 headline；不接受“看起来更像
V1”但 (b) 没降的改动。

### PT.2 若 (c) EFA delivery 偏大 —— 把小包做大、把 NIC 用满

(c) 偏大有两个子因，要分别处理：

```text
子因 1：消息太小（remote-scatter 导致 1-token / 4-token WRITE）
  - dispatch 已用 compact staging，当前实测最优 `4 token/WRITE`，平均 payload
    约 30KB/WRITE。V1 baseline 是 `16 token/WRITE`，但直接把 V2 chunk 提到 16/32
    已实测回退，因为 count/tail 可见被推迟。
  - 因此 dispatch 的下一步不是单纯调大 `kUCCLGinCompactChunkTokens`，而是探索
    V2-native ready/tag 或 receiver-facing landing，让“大 payload WRITE”和“细粒度
    token ready”解耦。
  - combine remote_contig=0% 无法直接合并。
  - 唯一能把 combine 做成大包的方向是 receiver-facing staging：sender 先把一段
    连续 local range 一次性 WRITE 到 receiver 的连续 landing 区，再由 receiver
    本地（NVLink/local copy）scatter 进最终 V2 slot。必须量化这次 local copy 的
    成本 vs 减少 WR / 增大包 的收益，且不得退回 V1 packed layout 语义。
  - 先做 profile-only 估算：若 combine 改成“每 (channel,src) 一个连续 landing 段”，
    平均包大小会从 ~14KB 升到多少，按 2.3 的 size sweep 推算 BW 上界。

子因 2：NIC/rail 并行度不足
  - 2.3 microbench：单流 rails=2 即 ~44 GB/s，整机 remote-only ~370 GB/s/node 需要
    用满 16 NIC。确认当前 proxy 的 channel -> queue -> proxy thread -> EFA QP/NIC
    映射是否把命令铺到所有 NIC，还是挤在少数 QP 上。
  - 检查 OFI_NCCL_FORCE_NUM_RAILS / QP-per-thread / NIC 绑定，对照 send CQE 的
    per-NIC 分布直方图。
```

### PT.3 若 (a) ring residency / (d) receiver apply 偏大

```text
(a) 偏大且自旋圈数高 -> 是 proxy drain 跟不上 GPU 产出（即 (b)/(c) 的下游表现），
    优先回到 PT.1/PT.2，不要单纯加深 ring（cap=0 已证无效）。
(a) 偏大但自旋圈数低  -> 是 commit fence / host-mapped 写本身的成本，评估 ring 放置
    与 fence 粒度。
(d) 偏大            -> receiver seq-apply 或 host buffer cache 行为；4.5 已显示 reorder
    不深，但要确认 apply 本身的 per-CQE 成本与 cache miss（原 P4 候选）。
```

PT 验收：

```text
- PT.0 给出 dispatch/combine 的 (a)-(d) 占比，并标出最大项；
- 针对最大项的改动使该 hop 单独下降，且慢节点 headline 提升 >= 5%；
- dispatch 朝 ~59 GB/s、combine 朝 >= 38 GB/s 前进；
- correctness 全过，wire/layout 未被破坏（或破坏处有设计说明）。
```

## P1: Combine critical-path profile

PT 是当前最高优先级；P1 的阶段分解结果作为 PT.0 在 combine 侧的输入之一，二者数据
互相印证。Combine 已正确运行，但仍是最明显的未优化主路径。

### P1.1 Kernel 阶段分解

增加可开关、默认零开销的 combine clock counters:

```text
scaleup_tail_wait_cycles/events/max
reduce_cycles/events/max
tma_store_wait_cycles/events/max
scaleout_d2h_cycles/events/max
finish_d2h_cycles/events/max
scaleout_finish_wait_cycles/events/max
remote_tokens
```

要求:

- 分开统计 `kAllowMultipleReduction=true/false` 的两个 put call site。
- max counter 必须带 `(channel, queue/lane)` detail，定位长尾和 proxy imbalance。
- profiling off 时不增加 hot-path atomic 或 printf。

当前进展:

- 增加 `UCCL_GIN_COMBINE_CLOCK_PROFILE=1` 的 clock-only 采样模式；
- 关闭已完成使命的 merge-opportunity 逐 token 统计，仅采样每 8 个 SM 中一个 SM；
- 相比原 full profile 的 `132-168 ms`，普通 combine 降到约 `14.6-15.1 ms`，但仍明显
  高于正常约 `4.1 ms`，所以只能读取事件级相对量级，不能读取 headline BW；
- 代表性采样中，scale-up wait 通常远小于 forward/reduce span；但第二个
  `combine_reduce` call site 的 `Wait buffer release` callback 会调用
  `flush_last_tma_and_issue_rdma()`，所以该位置的 `reduce_cycles` 包含了
  `d2h_cycles`。这些 counters 是嵌套采样，不能相加成 wall-time budget。
- 粗略读数应改为:forward span 里 D2H emission 占主导，pure reduce compute 更接近
  `reduce_cycles - d2h_cycles` 的残差。clock64 instrumentation 也会破坏原 kernel
  试图建立的 reduce/RDMA overlap，因此只能用于阶段排序，不能用于 headline BW 或
  精确 critical-path 求和。
- finish wait 呈现明显跨节点长尾，但每 channel/source 只发生一次；P1.3 已显示
  finish dependency 平均只占约 `0.12 ms`，不是首要优化对象。

### P1.2 Merge-opportunity profile

在 combine GPU emission 或 proxy decode 侧统计连续 run:

```text
same dst
local source pointer contiguous
remote destination pointer contiguous
both contiguous
run length histogram: 1, 2, 3-4, 5-8, 9-16, 17-32, >32
run break reason:
  dst change
  local gap
  remote gap
  metadata/end-of-channel
```

必须分别统计:

```text
normal combine
reduced combine
kAllowMultipleReduction=true/false
```

Discriminator:

- `both contiguous` 的 run 大量达到 `>=4`:优先实现零额外 buffer 的 direct multi-token
  WRITE。
- local 连续但 remote 不连续:不能简单合并 RDMA WRITE；需要重排目标或额外 descriptor，
  先评估是否违背 V2 layout。
- remote 连续但 local 不连续:需要 compact send staging；先算 buffer 与 TMA 成本。
- 两边都基本不连续:不要重写半个 combine kernel，应转向 proxy/GPU overlap 和
  forward/reduce 调度。

当前原始顺序结果:

```text
状态: completed
结论: same dst=100%，remote contiguous=0%，所有 run length=1。
```

注意:当前 EP8x2/两节点下 `same dst=100%` 是拓扑 artifact，因为非本地目标只有一个
remote node；未来扩展到 `>2` nodes 时仍必须先按 `dst_rank` bucket，不能把当前数字
当成 emission order 天然同 dst。

因此原始顺序下 direct contiguous batch 已降级为不可行。接着检查了是否能在保留 V2
receiver layout 的前提下，对每 channel 的 replay work 按 `src_token_idx` 做有界调度
重排。必须证明:

- 不破坏 `token_metadata_at_forward` 的 scale-up tail / linked-list 消费顺序；
- 不改变 reduce 和 top-k weight 对应关系；
- 不增加新的全局排序 pass 或 host materialize；
- 重排窗口能实际形成足够长的 remote-contiguous run。

P2.1 有界重排 profile 结果:

```text
状态: completed
方法:
  kernel 只 dump channel 0 的前 256 个 remote put candidates；
  CPU offline 按窗口 4/8/16/32 排序模拟，不在 GPU 内做重排序。

rank0:
  W4/W8/W16/W32 remote_contig = 0.0%, both_contig = 0.0%
  local_contig 约 64.9-73.7%

rank1:
  W4/W8/W16/W32 remote_contig = 0.0%, both_contig = 0.0%
  local_contig 约 65.4-68.3%
```

结论:bounded reorder window 不能创造 remote-contiguous run。local 有一定连续性，
但 receiver-facing V2 destination slot 完全打散；仅重排 emission 顺序不能合并成
multi-token RDMA WRITE。

### P1.3 Proxy/finish dependency profile

记录 combine 专属:

```text
plain payload WRITE count
finish ATOMIC count
dependency candidates/active/max
finish enqueue -> post latency
last payload CQE -> finish post latency
receiver finish CQE -> GPU observed latency
```

目标是判断 combine 慢在 payload WR 数量，还是 finish 被 dependency/receiver apply
拖住。不要只看累计 CPU 时间。

当前已增加低开销 proxy 指标:

```text
dependency_batches
dependency_enqueue_to_post_ns / max
dependency_ready_to_post_ns / max
```

其中带 plain WRITE dependencies 的 atomic batch 在当前协议中对应 combine finish。
`enqueue -> post` 表示 finish 被 payload CQE 阻塞的总时间；`ready -> post` 表示
dependencies 已完成后，proxy progress 调度自身的延迟。

README-like EP8x2 实测聚合:

```text
node0:
  dependency batches:       72,204
  enqueue -> post avg/max:  126.5 / 518.6 us
  ready -> post avg/max:     14.6 / 461.2 us

node1:
  dependency batches:       74,087
  enqueue -> post avg/max:  123.5 / 462.3 us
  ready -> post avg/max:     13.9 / 268.1 us
```

普通 combine 仍约 `4.1-4.3 ms`。因此 finish dependency 平均只占约 `3%`，且
dependencies ready 后的 proxy 调度约 `14 us`，不是当前 combine 主瓶颈。P1.3
completed；不要优先微调 finish dependency container。

额外验证:

- 尝试仅对带 `ncclGinOptFlagsAggregateRequests` 的 payload put 放宽 D2H inflight cap，
  sweep `8/16/32/64`；跨节点最慢普通 combine 分别约
  `4.166/4.182/4.171/4.166 ms`，无收益，代码已回退；
- 尝试将所有 normal put cap 设为 `0`，只由 2048-slot ring 容量背压；最慢普通
  combine 仍约 `4.171 ms`，dispatch 仍约 `1.626 ms`；
- 因此 combine 不是被 cap=8 过度节流。不要继续把放宽 cap 或扩大 ring 当作 P2
  优化方向。

P1 输出必须直接回答:

```text
combine 4 ms 中，最大的可优化部分是什么？
```

## P2: Combine emission scheduling / payload batching

P1.2/P2.1 已证明当前 emission order 和小窗口调度重排都没有 direct merge
opportunity。P2 若继续推进，必须改变 receiver-facing emission 形态，不能直接在
proxy 侧假装 batch。

P1.3 又证明 finish dependency 平均等待只有约 `0.12 ms`，所以 P2 的目标应是减少
约 `8155` 条离散 payload WRITE，以及降低与每 token emission 绑定的 TMA/D2H 成本，
而不是继续优化最后的 finish。

优先顺序:

1. **Receiver-facing staging / layout-aware compact 评估**
   - bounded reorder 已证伪，若要减少 payload WR 数量，需要把 remote destination
     先 staging 成连续区域，或设计显式 gather/scatter descriptor。
   - 必须保持最终 V2 receiver layout；如果需要额外 copy-back，要把 TMA/store 成本和
     减少 WR 的收益一起量化。

2. **Scatter/gather descriptor 评估**
   - 若 EFA/libibverbs 路径可以低成本 post 多 SGE，评估是否能把 local contiguous 的
     片段组合起来；但 remote 端仍不连续时不能用单个普通 RDMA WRITE 表达。

3. **Direct contiguous batch**
   - 当前原始顺序与窗口 `4/8/16/32` 重排的机会均为 `0%`，不允许直接实现。
   - 必须保持最终 V2 receiver layout，量化额外 gather/TMA/buffer 成本。

4. **Finish 优化**
   - payload batching 真正改变命令形态后重新测 standalone finish dependency。
   - 只有 wire encoding 和 empty-channel 语义都明确时，才考虑把 finish 与最后 payload
   合并；当前 8-bit `atomic_val` 不能直接表示 finish delta `8192`。

   另外，EFA SRD 不保证多个离散 payload WR 按到达顺序落地。把 finish 只 piggyback
   到“最后发出的 payload”不能证明此前 payload 已全部到达；当前 sender-side
   completion dependency 正是这项保证。P1.3 又显示该等待平均仅约 `0.12 ms`，所以
   在没有等价 ordering 协议前，不应为省一个 finish WR 删除它。

禁止:

- 为了 batch 把 receiver 写回 V1 packed layout。
- 添加 Python materialize 或 semantic all-to-all。
- 未 profile 就引入新的大 staging buffer。
- 通过同步 quiet 保证 payload-before-finish。

P2 验收:

```text
combine WRITE command count 明显下降；
平均 bytes/WRITE 明显上升；
combine SO BW >= 38 GB/s；
dispatch 性能不回退；
normal/reduced combine correctness 全过。
```

## P3: Dispatch forward critical path

Dispatch 已经过 command 数、sequence array、dependency 和 chunk sweep 优化。下一步不再
优先改 proxy 容器，而是定位并减少 forward 长尾。

### P3.1 轻量 tail-to-payload latency discriminator

把采样分为 CPU 和 GPU 两个时钟域，而不是全量 clock atomic:

```text
CPU domain:
  sender proxy post -> local send CQE
  receiver CQE poll -> sequence apply 完成

GPU domain:
  scaleout warp D2H push stall
  forward 开始 poll -> 首次观察到 tail
  tail ready -> metadata ready
  metadata ready -> payload TMA load 完成
```

优先使用采样或 per-channel max，避免现有 full clock profile 将 BW 从 `38` 压到
`10-12 GB/s`。

CPU steady clock 与 GPU `clock64()` 不能直接相减。只有通过 tagged sequence 和明确的
host/device clock calibration 后，才允许做跨域端到端时间关联；否则分别判断各自区间。

判断:

- sender post -> send CQE 大:sender/NIC completion 路径；
- receiver CQE poll -> apply 大:receiver decode/reorder/apply；
- GPU tail poll 大:count 首次可见或 source scheduling；
- tail ready -> payload load 大:payload DMA/HBM visibility 或 TMA load；
- 少数 channel/source max 极大:调度或 head-of-line blocking。

### P3.2 Forward 消费粒度 sweep

当前:

```text
payload compact chunk = 4
forward max consume    = 3 slots/round
```

做独立 sweep:

```text
kNumSlotsPerForwardChunk = 3, 4, 6, 8
```

观察:

```text
tail poll 次数
tail stall fraction / max
forward load latency
dispatch wall time
register/smem occupancy
```

Sweep 结果:

```text
slots=3:
  node0 avg 1.632 ms
  node1 avg 1.624 ms

slots=4:
  node0 avg 1.692 ms
  node1 avg 1.673 ms

slots=6:
  node0 avg 1.602 ms
  node1 avg 1.583 ms

slots=8:
  node0 avg 1.636 ms
  node1 avg 1.606 ms
```

`slots=6` 相对真正默认 `slots=3` 仅改善约 `1.9%` / `2.5%`，低于计划的 `3%`
保留门槛；`slots=4` 明显回退，`slots=8` 也没有额外收益。因此恢复上游默认 `3`，
删除实验开关，不给主路径留下调参分支。Forward consume 粒度不是当前主要 gap。

### P3.3 Source/channel head-of-line profile

已确认 forward warp 会等待某个慢 source，同时跳过其他已 ready source。`clock2`
profile 在每次 stall 后做一次 fresh tail read:

```text
rank0 node:
  selected source ready: 32.0%
  other source ready:    53.0%
  no source ready:       15.0%

rank1 node:
  selected source ready: 27.9%
  other source ready:    55.3%
  no source ready:       16.9%
```

这说明超过一半 stall 是 source-selection HOL，而不是所有 source 都没 ready。下一步
曾尝试最小 ready-source-first 改动：fresh-read 后如果 selected source 仍不 ready 但
其他 source ready，就切换到该 ready source。结果:

```text
log:
  /tmp/uccl_gin_ready_source_rank0.log
  /tmp/uccl_gin_ready_source_rank1.log

dispatch:          仍约 38 GB/s
cached dispatch:   仍约 38 GB/s
combine:           rank1 出现 23-24 GB/s 低值（combine path 未改，可能是噪声，但无收益）
```

该行为改动低于保留门槛，已回退；profile-only counters 保留。HOL profile 说明现象存在，
但 naive source switch 不足以提升 wall time。后续若继续处理 HOL，需要同时验证:

```text
ready source count when selected source stalls 是否下降
forward_tail_wait_events / stall_events 是否下降
cached dispatch wall time 是否下降
copy epilogue / combine 是否不回退
是否只是把等待从 tail wait 搬到 metadata/payload load 或 scale-up store
```

实现要求:

- 不改变 V2 metadata / expanded layout；
- 不引入 Python materialize 或 host scheduling；
- 默认不开 profiling 时不留下额外 printf / atomic counter；
- 若改动低于 3% headline 收益，回退或只保留文档结论。

P3 验收:

```text
dispatch SO BW 从 37-38 GB/s 提升到 >=45 GB/s，作为第一检查点；
最终目标接近 V1 ~59 GB/s；
copy epilogue 不回退；
expanded/cached dispatch correctness 全过。
```

## P4: Proxy hot-path 与 rail 利用率（已并入 PT）

注：本节候选项已上提到 PT.1/PT.2/PT.3，并由 PT.0 的 hop 分解来决定先做哪一项。
保留下方清单作为 PT 的实现细节参考。只有 PT.0/P1/P3 显示 CPU proxy 或特定 lane
是关键路径时再做对应项。

候选:

- 批量 drain D2H commands，减少 per-command C++ container 操作。
- 对 plain combine WRITE 提供真正的 batch post，而不是只依赖
  `ncclGinOptFlagsAggregateRequests` 标志。
- 减少 CQ poll 与 receiver apply 的 cache miss。
- 调整 channel -> queue -> proxy thread -> EFA rail 映射。

必须先收集:

```text
per-thread commands/s
per-thread posted bytes
per-thread CQEs/s
active mixed ns/command
receiver apply ns/CQE
queue occupancy / max wait
selected NIC/rail mapping
```

不要再使用包含空轮询时间的 `post_gpu_us / post_cmds` 作为 active per-command cost。

## P5: 泛化与 async pipeline

性能稳定后再做:

- 泛化 dispatch compact state 到 `kNumScaleoutRanks > 2`。
- 为跨 iteration async pipeline 设计 epoch/double-buffer tail；当前同步模型不需要
  host quiet。
- 评估 sequence 4-bit wrap 的严格 inflight 上界和运行时断言。
- 清理 profiling-only scaffolding，保留默认零开销 counters。

## 7. 每个实验的标准记录格式

每次实验写入 `worklog.md`:

```text
假设:
  为什么它可能在 critical path。

改动:
  文件、关键逻辑、是否改变 wire/layout/ordering。

Discriminator:
  哪个数据能证实或证伪。

环境:
  commit、JIT flags、JIT cache、build variables、server log path。

结果:
  rank0/rank1 correctness、BW、latency、command/WR/CQE、关键 profile。

结论:
  保留 / 撤销 / 降级；下一步是什么。
```

## 8. 当前最近一步

当前结论:

```text
PT.0 dispatch 当前轮已完成。CPU proxy post 与 receiver atomic apply 都不是 dispatch
主因；当前主嫌疑是 4-token WRITE 的 post->CQE delivery/queueing，以及 forward
source-selection HOL。V1 baseline 不是固定 32-token chunk，而是默认 16-token RDMA
chunk；V2 4-token 最优来自当前 receiver 可见性与 pipeline 平衡。

最新 producer/coordinator 实验进一步确认：

  - `cap=0 + chunk=16/32 + forward slots=16` 仍只有 `33-34 / 31-32 GB/s`；
  - 独立 coordinator 必须使用正式 system-scope payload-ready 协议，否则第一次
    dispatch 就会读取损坏 metadata；
  - system-scope ready 协议 correctness 通过，但 coordinator `chunk=4` 只有约
    `20 GB/s`，`chunk=16` 只有 `22-24 GB/s`；
  - 大 WRITE 本身有收益，但 producer/coordinator ready 发布与生命周期成本更大。
```

最新 per-rail profile 与 V1 queue 映射实验确认：

```text
原 V2 queue 映射:
  proxy thread bytes = 62.5 / 62.5 / 41.7 / 41.6 GB

恢复 V1 interleaved proxy mapping:
  proxy thread bytes = 52.1 / 52.1 / 52.1 / 52.1 GB

全部 QP 的 post->CQE average:
  约 83-111 us，未发现少数 rail/QP 独占长延迟。
```

恢复 V1 映射后，profiling-off README-like EP8x2 correctness 全过；dispatch 改善
约 `0.4-1.8%`，combine 改善约 `3-4%`，reduced combine 改善约 `4-5%`。该映射
修复保留。它同时排除了“当前 dispatch 主要被少数错误映射的 NIC/QP 拖慢”；剩余
约 `90-100 us` WRITE completion latency 是所有 rail 上普遍的小包 delivery/queueing
成本。

恢复映射后的 dispatch phase profile 进一步给出：

```text
scaleout D2H push           ~76k cycles/event
stalled forward tail wait  ~166k cycles/event
forward payload TMA load    ~14k cycles/event
forward metadata wait       ~2k cycles/event
forward scaleup store      < 1k cycles/event
```

全量 clock atomic 会显著扰动 wall time，因此这些数字只用于阶段排序；它们足以排除
“per-token metadata-ready wait 是主因”。同时，恢复映射后重新做的 8-proxy sweep
仍令 dispatch 回退，说明 4-token completion latency 也不是 proxy/QP 数不足导致。

P1.2 已否定原始顺序下的 direct batch，P1.3 已否定 finish dependency 是主瓶颈。
P1.1 clock-only 采样的 reduce/D2H counters 是嵌套测量，不可相加；校正后的方向是
优先减少 D2H emission/离散 payload WR，而不是优化 pure reduce。aggregate/normal
inflight cap sweep 又否定了“放宽 D2H cap 即可改善 combine”。

下一步:

```text
dispatch:
  1. 低扰动 sender-emission / forward-tail 联合 profile 已完成第一轮:
     当前路径是 MSCCL++ FIFO；FIFO at-cap 仅约 0.05%，push 约 38-39k cycles，
     与 V1 接近。下一轮只需补齐少量 forward 处理吞吐/profile，不再继续调 FIFO cap。
  2. 设计 V2-native multi-warp channel grouping:
     保留 V2 BufferLayout、expanded dispatch、token_metadata_at_forward 和 combine
     replay 语义；把当前每 SM 的 4 个 scaleout/4 个 forward warp 从 4 条独立 channel
     聚合成更少的 network channel。每条 network channel 需要多个 producer、一个
     轻量 coordinator、多个 forward consumer。
  3. grouping 的关键 correctness 设计:
     producer->coordinator 必须复用 V1 的 release/acquire + FIFO system-release
     ordering，避免每 token/chunk system atomic；多个 forward warp 必须安全分配
     V2 metadata 顺序、expanded slot 和 linked-list tail。ready/tag 只用于 payload
     freshness，不能替代这些 ownership 规则。
  4. grouping 完成后重新 sweep:
     network channel 数、producer/forward warp 比、chunk=4/8/16/32。只有此时再次
     测大 WRITE 才有意义。

### 6.1 Multi-warp channel grouping 的实现顺序

不能直接把多个 warp 的 `channel_idx` 做整数除法。V2 的 channel 同时是 buffer、
metadata、linked-list 和 combine replay 的 ownership 单元，必须按以下顺序改：

1. **解耦 warp 数与 network channel 数**（已完成且默认映射验证通过）
   - kernel thread 数仍由 `num_warps_per_role_per_sm` 决定；
   - buffer/handle/JIT 的 `kNumChannelsPerSM` 独立表示 network channel 数；
   - 默认两者相等，README-like dispatch 仍为 `37-38 GB/s`。
2. **Sender grouping**
   - 每个 network channel 使用多个 producer warp 和一个 coordinator；
   - producer 从共享 reservation tail 领取 compact-send slot，完成 TMA store 后按
     V1 的 CTA release/window 机制发布连续 ready tail；
   - coordinator 只观察连续 ready tail，按 16/32-token chunk 发
     WRITE+piggyback tail，不参与 token pack；
   - finish 只在所有 producer 退出且连续 ready tail 全部发送后发布。
3. **Receiver grouping**
   - 多个 forward warp 从每个 source 的 arrived tail 中原子领取不重叠 slot range；
   - 为每个 claimed range 原子分配连续 `token_metadata_at_forward` 序号；
   - per-scaleup linked-list index/tail 改为 channel-shared allocation，不能继续使用
     warp-local `stored_scaleup_send_counters`；
   - 最后一个 forward coordinator 写 metadata sentinel、linked-list tail，并清理
     rail tail。
4. **重新 sweep**
   - 从 `4 warps / 2 channels` 开始，再测 `4/1`；
   - 每种 grouping sweep chunk `4/8/16/32`；
   - 保留标准是 correctness 全过且 dispatch wall time/BW 有实际提升。

sender 和 receiver grouping 必须分别做 isolated correctness，但不能把临时 fallback
接入主路径。任何共享发布机制优先复制 V1 的 `rdma_send_channel_tail/window/lock`
语义，而不是新增 system-scope per-token atomic。

combine:
  继续用同口径 PT.0 拆 proxy post、post->CQE、receiver apply，但避免全量 clock
  atomic 扰动。
```

post->CQE/rail profile 已完成并恢复 V1 queue mapping。在完成更深 HOL/landing
profile 前，不再凭直觉调 cap/ring，也不再把
`kUCCLGinCompactChunkTokens` 直接改大作为主路径优化。naive ready-source-first
和独立 producer/coordinator 都已尝试且低于保留门槛，代码已回退。

receiver-facing staging / layout-aware compact 仍是 combine 把小包做大的主要候选，
但它现在归在 PT.2 子因 1 下，必须先由 PT.0 证明 (c) EFA delivery（小包）确实是最大
hop，再投入；如果这个方向需要重写过多 V2 buffer/layout，必须先和保留当前 28-31 GB/s
combine 性能作工程收益对比。与此同时，dispatch 的 P3.2 sweep 已完成并降级；
`slots=6` 的收益低于保留门槛，主路径继续使用上游默认 `3`。

已完成并否定的额外旋钮：

```text
恢复 V1 mapping 后的 8 proxy threads:
  dispatch 无提升，rank1 约回退 3%，已恢复 4 proxy。
```
