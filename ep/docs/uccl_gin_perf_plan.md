# UCCL-GIN V2 Dispatch 性能计划

本文档聚焦 DeepEP V2 UCCL-GIN 在 AWS EFA 上的 dispatch 性能。当前结论不是
“EFA 大包带宽不够”,也不是“ring size/proxy thread 数量不够”,而是 V2 UCCL-GIN 的
command emission、semantic batching、proxy per-command 成本和 receiver wait 组合起来
没有达到 V1 UCCL-EP 已验证过的效率。

最重要的原则:

- 先测清楚根因,不要先押注某个修法。
- V1 UCCL-EP 是 substrate/设计参考,但 V1/V2 benchmark 必须尽量 apples-to-apples。
- 任何优化都要同时看 BW、kernel wall time、D2H cycles、proxy per-command cost 和
  command/WR 数量。

## 2026-06-06 最新诊断与优先级

已经完成三项前置测量:

```text
V1 8192-token FP8 dispatch:       ~59 GB/s RDMA
V2 compact chunk average:         25.49 tokens
V2 full 32-token chunk fraction:  75%
V2 non-contiguous flushes:        0

V1 proxy post_gpu:                ~166 ns/command
V2 proxy post_gpu/command:        ~7,949 ns/command（包含大量空轮询，不能视为 active cost）
V2 active mixed path:             ~0.55-0.75 us/command
V2 dependency scan:               ~35-50 ns/candidate
V2 dependency max fan-in:         ~48-93 writes
```

因此需要修正旧优先级:

- compact32 已经正常工作。平均 `25.49 tokens/WRITE` 来自每 channel 的
  `3 x 32-token full chunk + 1 x 5-8-token finish partial chunk`,不是 slot
  不连续或中途 flush。
- 旧的 V2 `post_gpu_us/post_cmds ~= 8 us` 会把没有 command 的 proxy loop 也算入
  分子,不能与 V1 active post 直接比较。新增 `mixed_ns` 后,真正处理非空 command
  batch 的成本约为 `0.55-0.75 us/command`。
- dependency vector 不会无界增长,扫描本身也不是主瓶颈:单个 finish 前最大 fan-in
  约 `48-93`,扫描约 `35-50 ns/candidate`,每线程整个测试累计通常不足 `1 ms`。
- `progress_pending_atomics ~= 10-13 us/atomic` 仍值得继续拆分,但其整个测试累计仅约
  `35-50 ms / 23 s`,不能解释 dispatch 的 `~2 ms` kernel wall time。
- dispatch 优化目标以 V1 8192-token FP8 的 `~59 GB/s` 为第一阶段对标值。

### 对 dependency-tracking 根因假设的核对

当前代码事实:

```text
flush_writes():
  post_rdma_async_batched(...)
  inflight_write_wrs_.insert(all writes)
  atomic_dependency_wrs_.append(all writes)

enqueue_pending_atomics():
  扫描 atomic_dependency_wrs_
  对仍 inflight 的 WR 建 atomic_dep_by_wr_ 映射
  deps.clear()
```

结论:

- piggyback WRITE 虽然已经把 payload 和 count 放进同一个 WRITE_WITH_IMM,仍会进入
  `atomic_dependency_wrs_`,作为后续 standalone finish ATOMIC 的依赖。这部分
  bookkeeping 是真实存在的。
- 但 `atomic_dependency_wrs_` 会在每次 `enqueue_pending_atomics()` 后清空,并非只在
  QUIET/BARRIER 清空。因此“vector 无界增长到几十万、每个 finish 都扫描全历史”
  与当前代码不符。
- standalone finish 目前仍需要保证排在该 channel 最后 payload 之后。除非把 finish
  bit piggyback 到该 channel 的最后 payload chunk,否则不能直接删除 dependency
  tracking。

### 2026-06-07 dependency / finish 实验结论

新增 profile:

```text
mixed_ns
dependency_scan_ns
dependency_candidates
dependency_active
dependency_max
merge_profile_enabled
```

代表性数据:

```text
rank0/thread0:
  post_cmds=22568
  mixed_ns=14.027 ms              => 622 ns/command
  dependency_scan_ns=0.795 ms
  dependency_candidates=18104     => 43.9 ns/candidate
  dependency_active=11399
  dependency_max=72
```

还实验过“最后一个非空 payload WRITE 同时 piggyback finish,空 channel 才发 standalone
finish”。该实验确实把 `atomic_cmds` 和 dependency 计数降为 0,但没有证明性能收益,
而且把 finish 可见性绑定到最后一个大 payload 的完成/receiver apply。该行为已撤销,
只保留 profiling。

注意 kernel clock profile 很侵入:

```text
profiling off: cached dispatch 30-31 GB/s, 2.00-2.02 ms
clock profile on: cached dispatch 11-14 GB/s, 4.4-5.6 ms
```

因此 clock counters 只能用于定位相对热点,不能拿其 headline BW 判断优化收益。

### 当前执行顺序

1. 核对 receiver software-atomic sequence/reorder 语义:
   - piggyback count 与 standalone finish 是否共享同一 `(dst, atomic_offset)` seq。
   - receiver 是否只有按序 apply 完同一 tail-word 的 payload count 后才 apply finish。
   - 若成立,删除或收窄 sender 侧过度保守的全 batch completion dependency。
2. 继续拆 `progress_pending_atomics` 与 receiver CQE/apply:
   - 区分等待 dependency、post atomic、receiver WRITE_WITH_IMM decode/reorder/apply。
   - 关注它是否延迟关键 finish,而不是只看累计 CPU 时间。
3. 测量/优化 receiver critical path:
   - forward tail wait 和 forward load 是当前更可信的 kernel 热点。
   - profile 必须使用轻量采样或单独实验,避免 clock counters 改变整体性能。
4. 每轮必须同时观察:
   - dispatch SO BW / dispatch_impl wall time。
   - `post_gpu_ns_per_cmd`, `progress_atomic_ns_per_atomic`。
   - `scaleout_d2h cycles/event`。
   - command/WR/CQE 数量。
5. dependency container fast path 和 inflight cap 都降为后续实验,不作为当前首要修复。

### 已完成:收窄 standalone finish dependency

代码核对确认:

- payload `WRITE_WITH_IMM` 和 standalone finish ATOMIC 都通过
  `next_seq_per_index[(dst_rank, tail_index)]` 分配 sequence。
- receiver 的 `SeqBuf` 只会按 sequence 顺序 apply 同一 tail-word 的 delta。
- payload count 只有在相应 payload WRITE 到达 receiver 后才进入 `SeqBuf`。

因此 finish 不需要等待已经携带 ordered piggyback count 的 payload CQE。现在只把
`atomic_val == 0` 的 plain WRITE 留作 sender-side completion dependency。

结果:

```text
dependency_candidates: 18104 -> 248   (rank0/thread0)
dependency_active:      11399 -> 63
dependency_max:         72 -> 2

恢复基线:               30-31 GB/s, 2.00-2.02 ms
收窄 dependency:         31-32 GB/s, 1.93-2.00 ms
```

完整 correctness check 通过。收益约 `2-4%`,说明过度保守 dependency 确实延迟了
finish,但它不是剩余 `~2x` V1/V2 gap 的主因。下一优先级转向 receiver
WRITE_WITH_IMM CQE/reorder/apply 与 forward tail/load critical path。

### 已完成:receiver sequence profile 与 compact chunk sweep

receiver ordered atomic profile 显示:

```text
32-token chunk, rank0/thread0:
  receiver_atomic_cqes=22320
  receiver_atomic_in_order=20438
  receiver_atomic_buffered=1882     # 8.4%
  receiver_atomic_max_buffered=2

4-token chunk, rank0/thread0:
  receiver_atomic_cqes=120528
  receiver_atomic_in_order=115181
  receiver_atomic_buffered=5347     # 4.4%
  receiver_atomic_max_buffered=5
```

即使 4-token 让 CQE 数增加约 `5.4x`,dispatch 仍明显更快。当前瓶颈不是 receiver
处理 CQE 的总吞吐,而是 payload/count 首次可见和 forward warp 流式消费的延迟。

compact chunk sweep:

```text
tokens/chunk   cached dispatch SO BW   dispatch_impl
64             ~27 GB/s                2.23-2.31 ms
32             ~31-32 GB/s             1.93-2.00 ms
16             ~33-34 GB/s             1.79-1.83 ms
8              ~35-36 GB/s             1.70-1.73 ms
4              ~37-38 GB/s             1.59-1.64 ms
2              ~32 GB/s                1.90-1.93 ms
```

4-token 是当前 README-like EP8x2/H200/EFA 配置的最佳点。2-token 开始进入 EFA
小消息效率下降区,64-token 则因等待 chunk 填满而损失 streaming overlap。

同时把 ordered atomic sender/receiver sequence 状态从 `unordered_map` 改成由
`ProxyCtx` 持有的 1024 项直接索引数组:

- 1024 项上界直接来自 wire ABI 的 13-bit byte offset / 8-byte tail word。
- 不改变 sequence、reorder 或 apply 语义。
- 避免 4-token 高命令率下每个 ordered operation 的哈希查找。

当前基线:

```text
cached dispatch: ~38 GB/s (SO), 1.60-1.63 ms
```

下一步:

- 继续优化 payload 首次可见和 forward 消费重叠,而不是减少 WR/CQE。
- 检查能否让 scaleout warp 更早发布首个 4-token chunk,并减少 compact store 到 D2H
  publish 之间的等待。
- 重新拆 4-token 配置下的 `scaleout_store_wait`, `scaleout_d2h`,
  `forward_tail_wait`, `forward_load`,但应使用采样式 profile 避免 full clock profile
  把 headline BW 改写。

## 当前数据

### V2 UCCL-GIN

README-like EP8x2 / EP16-style dispatch 当前约:

```text
dispatch SO BW:      ~30 GB/s
dispatch_impl:       ~2.05 ms
copy epilogue:       ~0.30-0.38 ms
combine_impl:        ~12.8-13.1 ms
write_cmds:          ~956k
write_bytes:         ~182.6 GB
piggyback writes:    ~952k
atomic_cmds:         ~238k
proxy quiet/barrier: 0
```

关键日志:

```text
/tmp/uccl_gin_truecost_light_rank0.log
/tmp/uccl_gin_truecost_light_rank1.log
/tmp/uccl_gin_p06b_rank0.log
/tmp/uccl_gin_p06b_rank1.log
```

P0.6 kernel clock profile:

```text
scaleout_d2h_cycles      avg ~153k cycles/event
forward_tail_wait_cycles avg  ~52k cycles/event
forward_load_cycles      avg  ~15k cycles/event
forward_meta_wait_cycles avg   ~2k cycles/event
```

已证伪或降级的方向:

```text
8 proxy threads, queue 2048:
  dispatch ~2.00-2.02 ms
  scaleout_d2h avg ~178k cycles/event

4 proxy threads, queue 4096:
  dispatch ~2.05-2.08 ms
  scaleout_d2h avg ~153k cycles/event
```

所以问题不是简单的 “ring 太小” 或 “proxy thread 太少”。

### V1 UCCL-EP Baseline

远端独立 worktree:

```text
commit: 495b7221d084cce92553d6a038376358bd218a5a
worktree: /home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang-v1-baseline-495b722
```

已跑 EP16, `4096 tokens, hidden 7168, topk 8, experts 256`:

```text
rank0 FP8 dispatch: 1202 us, 50.22 GB/s RDMA
rank1 FP8 dispatch: 1161 us, 51.95 GB/s RDMA

rank0 BF16 dispatch: 1981 us, 59.10 GB/s RDMA
rank1 BF16 dispatch: 1836 us, 63.72 GB/s RDMA

rank0 combine: 7658 us, 15.29 GB/s RDMA
rank1 combine: 7777 us, 15.04 GB/s RDMA
```

日志:

```text
/tmp/uccl_v1_495_ep16_rank0.log
/tmp/uccl_v1_495_ep16_rank1.log
```

临时 D2H profile 版 V1:

```text
rank0 put:    avg 37,766 cycles/event, max 280,691
rank0 atomic: avg 38,369 cycles/event, max 233,733

rank1 put:    avg 38,382 cycles/event, max 459,321
rank1 atomic: avg 38,441 cycles/event, max 383,944
```

日志:

```text
/tmp/uccl_v1_495_d2hprof_ep16_rank0.log
/tmp/uccl_v1_495_d2hprof_ep16_rank1.log
```

注意:这组 V1 是 `4096 tokens`,而当前 V2 主要 profile 是 `8192 tokens`。因此
`50 GB/s vs 30 GB/s` 只能说明 V1 substrate 能在这台机器上跑快,不能直接作为最终差距。
必须补一组 V1 `8192 tokens` apples-to-apples baseline。

## 已确认的结构性问题

### 1. compact32 平均没有达到 32-token chunk

当前 profile 已经给出:

```text
write_bytes / write_cmds = 182,578,798,336 / 956,288 ~= 190,923 bytes
                           ~= 186 KiB per WRITE
```

按每 token payload 约 `7.5-14 KiB` 粗估,平均每个 WRITE 只有约 `13-25 tokens`,
不是 32。由于:

```text
piggyback_atomic_write_cmds / write_cmds ~= 0.996
```

大多数 WRITE 都是 compact chunk + piggyback tail,所以这个平均值确实反映当前
chunk 粒度没打满。它不是 P3 的“待确认小项”,而是和 proxy per-command cost 同级的
核心问题。

可能原因:

- EP8x2 / 8192 tokens 分到约 160 channels,每 channel 约 51 tokens。
- 再经过 local-bypass、dedup、dst 分布、slot 连续性限制后,每个 `(channel,dst)`
  流可能凑不满 32 就 flush。
- 也可能是 device emission 中途因为 slot/local buffer 不连续而提前 flush。

必须测:

```text
rail_put_tail_add count_delta histogram:
  count=1..8 / 9..16 / 17..24 / 25..31 / 32
per channel,dst 的 chunk 数和平均 count_delta
flush 原因:
  no_more_tokens
  dst_change
  local_gap
  remote_gap
  channel_end
  scratch/buffer_limit
```

如果大多数 chunk 是 `13-25 tokens`,把它推近 32 可以直接减少 `40-60%` 的 WRITE/WR/CQE,
也会把 message size 推到更适合 EFA 的区间。这可能比 inflight cap 更接近根因。

### 2. V2 proxy per-command 成本可能高于 V1

V2 `scaleout_d2h avg ~153k cycles/event` 可以解释为 GPU 在等 proxy 推进 tail。但
GPU 等得久不一定说明 cap 是根因,也可能说明 proxy 处理每条 command 太慢。

当前 V2 proxy 相对 V1 多了不少 hot-path 状态:

```text
pending_atomic_batches_
atomic_dep_by_wr_
retire_inflight_write()
piggyback atomic decode/apply
ordered software atomic / reordering buffer
profile/coalescing scaffolding
```

V1 和 V2 共用 RDMA verbs substrate,但 V2 每条 command 的 CPU bookkeeping 更重。
如果 proxy per-command cost 变高,那么 GPU 把 ring 灌到 2048 只是结果,不是根因。

必须测:

```text
V1 vs V2 proxy per-command cost:
  post_gpu_commands_mixed total ns / command
  poll_cq ns / CQE
  remote_process_completions ns / CQE
  piggyback/atomic apply ns / update
  notify_gpu_completion ns / acked WR
  commands drained per proxy loop
  per-thread commands/s
```

如果 V2 proxy per-command cost 明显高于 V1,优先优化 proxy hot path 或减少 command 数,
而不是先做 cap。

### 3. D2H inflight policy 不同,但 cap 是实验,不是先验答案

当前 V2 UCCL-GIN Rail helper 直接调用 ring 的无界 push:

```cpp
// ep/include/uccl_gin/uccl_gin_rail.cuh
q->atomic_set_and_commit(cmd, &slot);
```

ring 只在满 2048 时背压:

```cpp
uint64_t h = ld_volatile(&head);
uint64_t t = ld_volatile(&tail);
if (h - t == Capacity) {
  __nanosleep(64);
  continue;
}
```

V1 normal path 在 push 前做:

```cpp
inflight = cur_head - cur_tail;
if (inflight < kMaxInflightNormal) {
  h->atomic_set_and_commit(cmd, &slot);
}
```

默认:

```text
kMaxInflightNormal = 8
kMaxInflightLowLatency = 32
```

当前非 FIFO EFA path 的 `tail` 是 completion/ack 语义:

```cpp
poll_cq_* -> acked_wrs_
notify_gpu_completion -> mark_acked -> advance_tail_from_mask
```

所以 V1 式 cap 在语义上可做。但 cap 不会让 proxy 本身变快。它可能:

- 减少 ring 中排队陈旧 command。
- 降低 tail 滞后和 receiver/control burst。
- 把等待从 ring-full wait 改成更平滑的 credit wait。

也可能:

- 只是把 GPU 等待点从 2048 改到 8/16。
- wall time 不变,因为 proxy/NIC 服务速率没变。

因此 cap 应作为 P2 实验,不能在 P0/P1 诊断前写成最高优先级解法。

## 新优先级

### P0: Apples-to-apples V1 baseline

先补 V1 `8192 tokens` baseline,避免用 V1 4096 tokens 和 V2 8192 tokens 直接比较。

命令:

```text
V1 @ 495b722
EP16, hidden=7168, topk=8, experts=256
num_tokens=8192
```

收集:

```text
FP8 dispatch BW / latency
BF16 dispatch BW / latency
combine BW / latency
V1 D2H put/atomic avg/max cycles
```

判断:

- 如果 V1 8192 仍有 `~50 GB/s`,V2 gap 确认。
- 如果 V1 8192 明显下降,目标和优化优先级需要重新标定。

### P1: 两个根因诊断并行前置

P1 不写最终优化,只做低成本诊断,决定下一步治哪里。

#### P1A: compact chunk 粒度 / flush 原因 profile

在 device/JIT 或 proxy command decode 侧统计:

```text
count_delta histogram
avg count_delta per channel,dst
chunks per channel,dst
flush reason histogram
write_bytes/write_cmds by dispatch mode
```

目标:

```text
确认 chunk 小是因为自然凑不满,还是被 slot/local gap/flush policy 打断。
```

可能结论:

- 如果大多数流天然只有十几个 token,需要重新设计 channel/dst batching 或减少 channel
  分裂。
- 如果 flush 被 local/remote gap 打断,需要修 compact staging/emission 顺序。
- 如果 count_delta 已接近 32,则 command 数不是首要问题。

#### P1B: V1 vs V2 proxy per-command cost profile

在 V1 和 V2 都加同一种轻量 proxy profile:

```text
post command loop:
  ns total
  commands
  ns/command

RDMA post:
  ns total
  WRs
  ns/WR

CQ poll + completion processing:
  ns total
  CQEs
  ns/CQE

notify_gpu_completion:
  ns total
  acked WRs
  ns/acked WR

software atomic / piggyback apply:
  ns total
  updates
  ns/update
```

目标:

```text
判断 V2 proxy 是否比 V1 每条 command 显著更贵。
```

如果是:

- 优先砍 proxy hot-path bookkeeping。
- 或通过 P1A/P3 减少 command 数。

如果不是:

- 再看 D2H cap / receiver wait / NIC visibility。

### P2: V1 式 D2H inflight cap sweep

只有在 P1 之后做。cap 是调度/排队形态实验,不是根因假设。

设计:

```cpp
atomic_set_and_commit_with_cap(cmd, &slot, max_inflight)
```

用于:

```text
rail_put
rail_put_tail_add
rail_red_add
```

sweep:

```text
max_inflight = 4, 8, 16, 32, 64, 128, 2048
```

记录:

```text
dispatch BW / dispatch_impl latency
scaleout_d2h avg/max
cap_wait cycles/events
proxy ns/command
forward_tail_wait / forward_load
```

成功标准只看 wall time/BW 和整体 profile,不要用 “per-push cycles 接近 V1” 作为成功。
per-push cycles 只是辅助信号。

### P3: 修 compact batching / 减少 command 数

如果 P1A 显示 chunk 未达 32,这是高优先级优化。

候选方向:

1. 减少 channel 过度分裂。
   - EP8x2 / 8192 tokens / 160 channel 导致每 channel token 少。
   - sweep channels-per-SM 或 num_sms/channel layout,看 count_delta 和 BW 是否改善。

2. 改 device emission 顺序。
   - 让同一 `(channel,dst)` 的 token 尽量连续 flush。
   - 避免 local gap / remote gap 过早打断 chunk。

3. 更晚 flush。
   - 不要在可以继续攒 token 时提前发小 chunk。
   - 需要保证 receiver expanded layout 和 metadata 语义不变。

目标:

```text
write_cmds 下降 40%+
avg count_delta 接近 32
dispatch BW 提升
```

### P4: 降低 V2 proxy hot-path per-command overhead

如果 P1B 显示 V2 proxy per-command cost 高于 V1:

优先检查:

```text
atomic_dep_by_wr_ unordered_map 查找/擦除
pending_atomic_batches_ 生命周期
retire_inflight_write()
remote_process_completions piggyback decode/apply
ordered atomic reordering buffer
profile/coalescing debug scaffolding 是否在 hot path
```

方向:

- 给 piggyback payload 的常见路径做 fast path。
- 避免每个 WRITE 都走 atomic dependency bookkeeping。
- 把 profile/merge 代码完全 gate 到 env off 路径外。
- 用数组/ring-local metadata 替代 unordered_map,如果 WR id 可直接索引。

目标:

```text
V2 proxy ns/command 接近 V1
scaleout_d2h avg 下降
dispatch BW 提升
```

### P5: channel -> queue/proxy 映射偏斜

当前 proxy 负载:

```text
thread0/1: ~54.8 GB each
thread2/3: ~36.5 GB each
```

8 proxy threads 没改善,说明瓶颈不是线程数本身,更可能是每线程 per-command cost 或
映射/命令来源偏斜。P5 在 P1B 后做:

```text
per-ring post_cmds/write_cmds/atomic_cmds/write_bytes
per-ring ns/command
per-ring d2h wait max
per-ring dst distribution
notify/control vs scaleout payload 来源
```

修复必须保持 queue 是 transport parallelism,不能引入新的 dispatch/combine 语义队列。

### P6: Receiver wait / data visibility

P1-P5 后复测:

```text
forward_tail_wait
forward_meta_wait
forward_load
```

如果 sender/proxy 侧改善但 wall time 不动,说明瓶颈转向 receiver:

- 对比 V1 epoch-tag wait。
- 检查 V2 metadata ready 是否足以替代 epoch tag。
- 确认 forward warp 是否在 TMA load 上等尚未可见的 NIC DMA payload。

### P7: Combine 单独立项

V1 combine baseline:

```text
~15-17 GB/s RDMA
```

当前 V2 combine:

```text
combine_impl ~13 ms
```

combine 需要单独按 dispatch 方法 profile:

```text
D2H push
proxy ns/command
payload WR size/count
receiver wait
reduce path
```

## 实施顺序

```text
0. 跑 V1 8192-token apples-to-apples baseline。
1. P1A: count_delta / chunk / flush reason profile。
2. P1B: V1 vs V2 proxy ns/command profile。
3. 根据 P1A/P1B 选择:
   - chunk 太小 -> P3 compact batching
   - proxy per-command 太慢 -> P4 proxy fast path
   - 两者都不是 -> P2 inflight cap sweep / P6 receiver wait
4. P2: bounded D2H cap sweep,作为排队形态优化实验。
5. P5: per-ring/proxy 映射优化。
6. P7: combine。
```

## 判断标准

不要用单一指标判断 patch 成功。

必须同时看:

```text
dispatch wall time / SO BW
write_cmds / write_bytes / avg bytes per WRITE
count_delta distribution
proxy ns/command
scaleout_d2h avg/max
forward_tail_wait / forward_load
correctness
```

短期目标:

```text
V2 UCCL-GIN dispatch 从 ~30 GB/s 提升到 40+ GB/s
```

中期目标:

```text
接近 V1 UCCL-EP apples-to-apples dispatch baseline
```

长期目标:

```text
保持 DeepEP V2 layout / JIT / handle 语义,让 UCCL-GIN Rail path 成为
NCCL-GIN 在 AWS EFA 上的高性能 backend,而不是回退成 V1 packed EP。
```
