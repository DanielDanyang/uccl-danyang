# UCCL-GIN Compact Staging 设计

## 问题

V2 的 `scaleout_send_buffer` 按 `token_idx` 索引,同 dst 的 token 在本地内存中不连续:

```
scaleout_send_buffer[token_idx]:
  token 0  (dst=1) → slot 0
  token 1  (dst=0) → slot 1     ← dst 0 和 dst 1 穿插
  token 2  (dst=1) → slot 2
  ...
```

这导致每条 `gin.put` 只能发 1 个 token (~14KB)。对比 UCCL-EP/EFA 论文(UCCL-EP
§3.3):HT 模式以 **chunk(典型 32 token)** 为粒度发 RDMA WRITE。V1 的 coordinator warp
做到这一点是因为 `send_buffer[dst][slot]` 天然连续。V2 缺的就是这个连续布局。

### EFA 小包性能

NCCL-GIN microbench 在 EFA 上的单 QP small-message 数据(agenda已记录):

```
 4 KiB →  2.8 GB/s
 8 KiB →  5.1 GB/s
16 KiB →  8.8 GB/s
32 KiB → 12.5 GB/s
...
 1 GiB → 44.8 GB/s (rails=2)
```

当前 V2 每 token ≈ 11-14 KB,处在 EFA 小包吞吐曲线的底部。论文 §3.3 明确说 HT
kernel 使用多个 ring buffer 暂存待发送 token,以 configurable chunk 发送,典型值是
**32 tokens**。因此这里不应把 6-8 token 当最终目标;6-8 只能作为调试阶段的保守
smoke 参数。最终目标应是 32-token chunk,即单 WRITE 约 350-450 KB。

---

## 核心思路

**同一个 `scaleout_send_buffer`,换一种索引方式**。不新增新的 payload buffer,不新增
额外 GPU copy。关键是把它从 `send[token_idx]` 临时 scratch 改成 sender-side
per-channel ring/window,对 EP8x2 先特化成 `send[channel][slot]`。

```
当前 (sparse):
  TMA store → send_buffer[token_idx]           // dst 穿插
  gin.put(send_buffer[token_idx], recv[slot])  // 每 token 一条小 WRITE

改为 (compact):
  compact_slot = stored_dst_slot_idx
  TMA store → send_buffer[channel][compact_slot]         // EP8x2:唯一 remote dst,源连续
  攒够 32 token 或 channel finish 时:
    gin.put(send_buffer[channel][first_slot],             // 源连续
            recv[channel][first_slot],                    // 目标连续
            N × token_bytes)                              // 一条大 WRITE
    rail_tail_add(N)                                      // 一条 tail
```

```
scaleout_send_buffer 分区:

  ┌─────────────────────────────────────────────────────────────────┐
  │                    scaleout_send_buffer                          │
  │               (kNumMaxTokensPerRank tokens)                      │
  │                                                                  │
  │  channel 0:  ┌──────────────┬──────────────┬─────┐              │
  │              │ dst 0 slots  │ dst 1 slots  │ ... │              │
  │              │  [0..N0-1]   │  [0..N1-1]   │     │              │
  │              └──────────────┴──────────────┴─────┘              │
  │  channel 1:  ┌──────────────┬──────────────┬─────┐              │
  │              │ dst 0 slots  │ dst 1 slots  │ ... │              │
  │              └──────────────┴──────────────┴─────┘              │
  │  ...                                                             │
  └─────────────────────────────────────────────────────────────────┘

  EP8x2 first:
    local dst 直接 bypass,不占 send buffer
    每个 local rank 只有 1 个 remote scaleout dst
    所以 send buffer 可解释为:
      send[channel][slot], slot in [0, kNumMaxTokensPerChannel)

  README-like EP8x2, SM=20:
    kNumChannels = num_sms * num_channels_per_sm
    若 num_channels_per_sm=4, kNumChannels=80
    kNumMaxTokensPerChannel=ceil(8192/80)=103
    一个 channel 可切 3 个 32-token chunk + 1 个 tail chunk
```

不要把 send buffer 做成 `[channel][dst][slot]` 并给每个 dst 平均分 slot。那样要么
实际扩大 buffer,要么在 skew routing 下溢出。EP8x2 先利用“两节点只有一个 remote
scaleout dst”这个事实,才能做到不新增 payload buffer 且 chunk 目标达到 32。

---

## 为什么不会破坏 V2 的其他部分

`scaleout_send_buffer` 在整个 `hybrid_dispatch.cuh` 中只有两处访问:

| 行号 | 操作 | 角色 |
|------|------|------|
| 463 | `tma_store_1d(send_buffer[token_idx], ...)` | TMA store 目标 (写) |
| 486 | `gin.put(..., send_buffer[token_idx], ...)` | RDMA 源地址 (读) |

两处都在 **scaleout warp 内部**,同一条 warp 的同一个循环。没有其他 warp、其他
kernel、或 receiver 端读这个 buffer。

关键不变量:

1. **TMA store → gin.put 的 happens-before**: 当前靠 `tma_store_wait()` (行 475)。
   改为 compact 后,`gin.put` 在 TMA store 之后才发(因为要等攒够 batch),happens-before
   自动满足。

2. **Receiver 不看 send buffer**: forward warp 从 `scaleout_recv_buffer` 读——那
   是 NIC DMA 写入的目标。send buffer 的布局变化对 receiver 完全透明。

3. **本地 bypass 不受影响**: 行 469-473,本地 rank token 直接 TMA store 到
   `scaleout_recv_buffer`,不走 `scaleout_send_buffer` 也不走 `gin.put`。不变。

4. **Buffer 总大小不变**: compact 索引不改变 buffer 的字节数。不碰 `scaleup_buffer`
   和 `scaleout_recv_buffer` 的边界。

5. **多 warp 无竞争**: 每个 channel 只有一个 scaleout warp。EP8x2 版本中每个
   channel 写自己的 `send[channel][slot]` ring/window。`stored_dst_slot_idx` 已经由
   V2 的 per-channel/dst tail 分配,不需要新的 `atomicAdd`。

6. **Forward warp chunk 兼容**: forward warp 按 `kNumSlotsPerForwardChunk` (=3) 消费
   recv slot。一个 compact batch 的 N 个 token 对应 N 个连续的 recv slot。forward
   warp 多轮消费即可,语义不受影响。

7. **Receiver metadata readiness 不变**: 每个 slot 仍然有独立的
   `src_token_global_idx`,forward warp 的 per-slot check 照旧工作。

---

## 改动范围

### 改 1: `hybrid_dispatch.cuh` — scaleout warp 循环

在 scaleout warp 的 per-token 循环中(行 418-505),引入 per-dst batch 状态:

```cpp
// 在 scaleout warp 循环开始处新增:
struct DstBatch {
    int dst_rank = -1;
    int first_compact_slot = 0;
    int count = 0;
    int first_recv_slot = -1;  // 第一个 recv slot,用于 gin.put 目标地址
};
DstBatch cur_batch;  // warp register,每 lane 持有一个

auto flush_batch = [&]() {
    if (cur_batch.count == 0) return;
    // TMA store fence: 所有 token 的 TMA store 都已完成
    // (每 token 的 tma_store_commit+wait 已在循环中完成)
    if (cur_batch.dst_rank != scaleout_rank_idx) {
        const uint32_t loff = uccl_gin::window_off(
            reinterpret_cast<uint64_t>(
                scaleout_send_buffer
                    .get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx)
                    .get_token_buffer(cur_batch.first_compact_slot)
                    .get_base_ptr()),
            res.window_base);
        const uint32_t roff = uccl_gin::window_off(
            reinterpret_cast<uint64_t>(
                scaleout_recv_buffer
                    .get_rank_buffer(cur_batch.dst_rank)
                    .get_token_buffer(cur_batch.first_recv_slot)
                    .get_base_ptr()),
            res.window_base);
        uccl_gin::rail_put(
            gin.lane(channel_idx),
            gin.rail_global_rank(cur_batch.dst_rank),
            cur_batch.count * tma_buffer.get_num_bytes<false>(),
            loff, roff);
    }
    gin.rail_tail_add(channel_idx, scaleout_rank_idx,
                      cur_batch.dst_rank, cur_batch.count,
                      /*finish=*/false, channel_idx);
    cur_batch.count = 0;
};
```

Per-token 逻辑修改:

```cpp
// 原来 (行 462-500):
if (scaleout_rank_mask ^ (1 << scaleout_rank_idx)) {
    tma_store_1d(send_buffer[token_idx], ...);   // 目标: token_idx
}
...
if (stored_dst_slot_idx >= 0 && stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
    gin.put(recv[stored_dst_slot_idx], send_buffer[token_idx], ...);  // 源: token_idx
}
update_scaleout_tail();

// 改为:
if (stored_dst_scaleout_rank_idx >= 0 &&
    stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
    // dst 变了 → flush 上一个 batch
    if (stored_dst_scaleout_rank_idx != cur_batch.dst_rank) {
        flush_batch();
        cur_batch.dst_rank = stored_dst_scaleout_rank_idx;
        cur_batch.first_recv_slot = stored_dst_slot_idx;
        cur_batch.first_compact_slot = stored_dst_slot_idx;
    }
    // TMA store 到 compact slot
    compact_slot = stored_dst_slot_idx;
    tma_store_1d(send_channel_buffer[compact_slot], ...);
    cur_batch.count++;
}
// update_scaleout_tail 移除 —— tail 在 flush_batch 里统一发
```

以及循环结束后和 finish 时 flush 残留 batch。

### 改 2: `hybrid_dispatch.cuh` — buffer 分区

将 `scaleout_send_buffer` 按 channel 分区:

```cpp
// 原来:
auto scaleout_send_buffer = BufferLayout<false>(
    token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());

// 改为:
constexpr int kNumCompactSendTokens = kNumChannels * kNumMaxTokensPerChannel;
auto scaleout_send_buffer = BufferLayout<false>(
    token_layout, 1,
    kNumCompactSendTokens,
    scaleup_buffer.get_buffer_end_ptr());
auto send_channel_buffer =
    scaleout_send_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
```

这不是新增 payload buffer。`kNumChannels * kNumMaxTokensPerChannel` 只是把
`ceil_div(num_tokens, num_channels)` 带来的 per-channel padding 显式给 send side 用。
上游 dispatch buffer size 已经在 recv side 按 `num_max_tokens_per_rank + kNumMaxChannels`
预留了 padding slack;实现中有 static assert 确认 compact send + recv 仍落在原
DeepEP V2 dispatch buffer size 内。

### 改 3: 不新增 compact tail 计数器

不新增发送端 compact slot allocator。`stored_dst_slot_idx` 已经是当前 channel/dst 的
V2 expanded recv slot,由上游 `ptx::exchange(stored_scaleout_tail, dst)` 分配并保持
唯一、单调。EP8x2 版本直接用它作为 send compact slot。

### 不改

- `scaleout_recv_buffer` — 完全不动
- `scaleup_buffer` — 不动
- `gin.put` / `rail_put` / `rail_tail_add` 的 API — 不动,只改调用参数
- proxy 端 — 不动 (仍然是 `rail_put` → D2H WRITE cmd → `ibv_post_send`)
- receiver forward warp — 不动 (仍然是 tail + per-slot metadata readiness)
- QP / EFA 传输层 — 不动

---

## 预期收益

| 指标 | 当前 | compact staging |
|------|------|-----------------|
| 每条 WRITE 大小 | ~14 KB (1 token) | ~350-450 KB (32 token) |
| 每 iter WRITE 命令数 | ~455,000 | ~14,000 量级或更低 |
| 每 iter tail atomic 数 | ~155,000 | ~14,000 量级或更低 |
| GPU 额外开销 | 0 | `atomicAdd` × 1 per token (~几个 cycle) |
| 预计 SO bandwidth | 4-8 GB/s | 目标接近 EFA 大包区间 |
| receiver 端改动 | — | 无 |

---

## 风险与验证

1. **Compact slot 溢出**:
   - EP8x2 first: local bypass,remote dst 唯一,`stored_dst_slot_idx <
     kNumMaxTokensPerChannel` 应由 V2 原有 tail/slot 逻辑保证。
   - 3+ scaleout ranks:不能复用这个不新增 buffer 的简单 layout;需要额外 send
     capacity 或 per-dst ring credit。

2. **Finish tail**: 最后一批 token 的 finish flag 需要在 `rail_tail_add` 中置位。

3. **Tail delta 限制**: 目前 `kAtomicValueMax = 16383`,batch N ≤ 16383 宽松满足。

4. **验证方案**: smoke test (64 token) → README-like EP8x2 (8192 token) → 性能对比。

---

## 与 UCCL-EP 论文的对照

| 论文结论 | 本设计 |
|----------|--------|
| HT mode chunk = 32 tokens | chunk target = 32 tokens (EP8x2 per-channel ring/window) |
| Write + piggyback atomic | tail 嵌入 `flush_batch` 中的同一次 `rail_tail_add` |
| Receiver-side ordering | 已有的 receiver metadata readiness + proxy reorder buffer |
| Per-channel FIFO ordering | 已有的 `lane(channel_idx)` |
| LL mode token packing = future work | 本设计就是 LL 粒度的 HT-style chunking |

论文 §7 特别提到 "packing tokens in a best-effort manner before sending them out
... would particularly benefit AWS EFA NICs" — compact staging 正是这个方向。

---

## 讨论记录

(来自 2026-06-06 与 Claude Code 的讨论)

**Q**: V1 chunked RDMA vs 当前 V2 的量级?
**A**: V1 每 WRITE ≈ 84 KB (6 token × 14KB),V2 每 WRITE ≈ 14 KB (1 token)。
WR 数量差 ~300 倍。V1 buffer layout 是 `send[dst][slot]` 连续,V2 是 `send[token_idx]`
稀疏。

**Q**: 在 smem 里攒 token batch 可行吗?
**A**: 不可行。H200 单 SM 228KB shared memory,16 warps/SM 同时工作,放不下 multi-token
batch。

**Q**: 改 buffer layout (compact staging) vs multi-SGE?
**A**: 论文支持 compact staging 方向。multi-SGE 需要改 QP cap,且 EFA SRD
对 multi-SGE 性能未知。V1 本身用 compact layout 而非 multi-SGE。

**Q**: compact staging 会破坏 V2 其他部分吗?
**A**: 不会。`scaleout_send_buffer` 只有 TMA store 和 gin.put 两处访问,都在 scaleout
warp 内部。Receiver、forward warp、buffer 边界都不受影响。

**Q**: V1 有 LL/HT 两种模式,V2 只有一种?
**A**: V2 的 `hybrid_dispatch` 语义是 HT (有 dedup + 转发),但传输粒度是 LL (per-token)。
论文的 chunk=32 是 HT 模式的优化,论文自己也说 LL 模式的 token packing 是 future work。
V2 虽然语义是 HT,但传输粒度没有达到 HT 的 chunk 水平。
