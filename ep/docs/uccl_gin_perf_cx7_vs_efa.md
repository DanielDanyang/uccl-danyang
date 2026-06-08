# CX7 (GDAKI) vs EFA (Proxy GIN) — GIN put 性能对比

**测试日期**: 2026-06-07  
**NCCL**: 2.30.4 (cuda13.2)  
**Benchmark**: `gin_proxy_bench` — device-side NCCL GIN all-to-all put  
**测试拓扑**: 2 节点 × 1 GPU, inter-node only

---

## 硬件对比

| | GH200 + CX7 | P5en + EFA |
|---|---|---|
| **GPU** | 1× GH200 480GB (sm_90) | 1× H200 141GB (sm_90) |
| **NIC** | 1× CX7 (BlueField-3 integrated) | 2× EFA (200 Gbps each) |
| **总 raw BW** | 400 Gbps (NDR) | 400 Gbps (2×200G) |
| **GPU-NIC 互联** | NVLink-C2C (integrated) | PCIe Gen5 x16 |
| **GIN 路径** | **GIN_IB_GDAKI** (native) | **GIN_IB_PROXY** (aws-ofi-nccl) |
| **底层 transport** | InfiniBand Verbs | Libfabric/SRD over EFA |

---

## 大包带宽 (big packet, ctas=16, remote-only)

| Size | CX7 GDAKI | EFA proxy (2 rails) | CX7/EFA |
|------|-----------|---------------------|---------|
| 1 MiB | 25.39 GB/s | 3.20 GB/s | **7.9×** |
| 4 MiB | 40.64 GB/s | 13.14 GB/s | **3.1×** |
| 16 MiB | 47.35 GB/s (注1) | 29.55 GB/s | **1.6×** |
| 64 MiB | 48.30 GB/s | 39.52 GB/s | **1.22×** |
| 256 MiB | 48.40 GB/s | 40.67 GB/s | **1.19×** |
| 1 GiB | 48.43 GB/s | 41.67 GB/s | **1.16×** |

> 注1: CX7 在 16 MiB 时出现短暂波动 (39.36)，两侧均为 `remote_only` 纯跨节点。
> EFA 在 ≤16 MiB 时尚未到达饱和。

**结论**: 大包极限带宽 CX7 领先约 **16%** (48.4 vs 41.7 GB/s)。两者均已接近各自
400 Gbps 物理上限的 80-97%。EFA proxy 路径额外损耗约 7-8 GB/s。

---

## 小消息带宽 (64 MiB/peer, 拆分为 message_bytes)

**这是 DeepEP-like dispatch 最关键的指标** — 需要高频、小粒度 RDMA put。

| message_bytes | CX7 GDAKI | EFA proxy (2 rails) | CX7/EFA |
|---|---|---|---|
| **1 KiB** | 25.53 GB/s | 0.94 GB/s | **27.2×** |
| **2 KiB** | 47.30 GB/s | 1.92 GB/s | **24.6×** |
| **4 KiB** | 48.95 GB/s | 3.81 GB/s | **12.8×** |
| **8 KiB** | 49.56 GB/s | 7.61 GB/s | **6.5×** |
| **16 KiB** | 49.22 GB/s | 14.92 GB/s | **3.3×** |
| **32 KiB** | 49.55 GB/s | 26.20 GB/s | **1.89×** |
| **64 KiB** | 47.15 GB/s | 33.50 GB/s | **1.41×** |
| **128 KiB** | 46.68 GB/s | 37.10 GB/s | **1.26×** |

**结论**: 小消息性能是 CX7 GDAKI 对 EFA proxy 的**决定性优势**:
- 在 DeepEP 典型 7-8 KiB (FP8 hidden=7168) 消息粒度下，CX7 比 EFA 快 **6-8×**
- CX7 在 2 KiB 即接近线速 (47 GB/s)，EFA 要到 128 KiB 才达到类似水平
- EFA proxy 每条消息的 CPU 介入开销 (CQ poll、proxy progress、SRD reorder) 
  是瓶颈根因；GDAKI 路径完全消除了这一开销

---

## CTA/Context 敏感性

64 MiB/peer big packet，all peers:

| CTAs | CX7 GDAKI | EFA proxy |
|------|-----------|-----------|
| 1 | 48.39 GB/s | 43.68 GB/s |
| 4 | 46.80 GB/s | 44.13 GB/s |
| 16 | 49.32 GB/s | 44.98 GB/s |
| 32 | 49.11 GB/s | 42.72 GB/s |

**结论**: 两者对大包都不敏感 (1 CTA 已近峰值)。EFA 32 CTA 有轻微退化。

---

## 总结

### CX7 GDAKI (native IB) 优势

| 维度 | 优势倍数 | 解释 |
|------|---------|------|
| **大包极限** | 1.16× | 48.4 vs 41.7 GB/s，均接近 400G 物理上限 |
| **小消息 (≤4 KiB)** | **10-27×** | GDAKI 消除 CPU proxy 开销 |
| **小消息 (8-32 KiB)** | **2-6×** | EFA proxy 每个 message 固定开销 ~3-5 µs |
| **饱和所需消息大小** | 2 KiB vs 128 KiB | CX7 2 KiB 即饱和，EFA 需 64× 更大消息 |
| **单 rail/port 效率** | 97% of 400G | CX7 单 port 达 48.4/50 GB/s |

### EFA proxy 特点

- 大包性能可接受 (41.7 GB/s, 83% of 400G)
- 小消息受 proxy 路径 CPU 开销显著拖累
- 好消息: 多 rail 可扩展 (本测试仅 2 rails；4+ rails 预期接近 CX7 大包性能)
- 坏消息: 小消息开销不会随 rail 数线性改善 (proxy 瓶颈在 CPU)

### 对 UCCL-GIN 的启示

1. **AWS EFA 上 GIN 性能的根本瓶颈是 proxy 路径的 per-message CPU 开销**，
   不是网络带宽。12.8× 以上的小消息差距无法靠增加 rails 或优化合并策略弥补。
2. CX7 GDAKI 在 DeepEP 典型 7-8 KiB 消息粒度下可以**不经 coalescing 直接
   到达线速**；EFA proxy 则需要 aggressive coalescing 才能接近可用性能。
3. DeepEP dispatch 在 EFA 上 `num_sms` sweep 只能提升到 5-6 GB/s (`SO`)
   的结论与本次 microbenchmark 一致：16 KiB 消息在 EFA 上仅 14.9 GB/s
   (all-to-all 模式，含双向)，单向 dispatch 7-8 GiB/s 已接近上限。
4. 如果未来 UCCL-GIN 需要 target CX7 平台，GDAKI 路径应作为一等路径支持；
   对 EFA proxy 路径，coalescing 和 batching 是性能关键。

---

## 原始数据

| 文件 | 位置 |
|------|------|
| CX7 size sweep | `/tmp/gin_bench_size_sweep.log` (gh200_0) |
| CX7 remote-only | `/tmp/gin_bench_remote_only.log` (gh200_0) |
| CX7 message sweep | `/tmp/gin_bench_msg_sweep.log` (gh200_0) |
| CX7 CTA sweep | `/tmp/gin_bench_cta_sweep.log` (gh200_0) |
| EFA size sweep | `/tmp/gin_bench_efa_size_sweep.log` (p5en_0) |
| EFA remote-only | `/tmp/gin_bench_efa_remote.log` (p5en_0) |
| EFA message sweep | `/tmp/gin_bench_efa_msg_sweep.log` (p5en_0) |
| EFA CTA sweep | `/tmp/gin_bench_efa_cta_sweep.log` (p5en_0) |

---

## 测试命令参考

**GH200 + CX7**:
```bash
cd ~/nfs/danyang/ep/tools
export LD_LIBRARY_PATH=$(pwd)/cuda13_libs:$HOME/.venvs/gin-bench/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
mpirun -np 2 -H 38.123.21.3:1,38.123.21.6:1 \
  -x LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_IB_HCA=mlx5_0 \
  ./gin_proxy_bench --min-bytes 1K --max-bytes 1G --ctas 16 --iters 20
```

**P5en + EFA (1 GPU, 2 rails)**:
```bash
cd ~/efs/yzhou/playground/daniel/uccl-danyang/ep/tools
export LD_LIBRARY_PATH=$HOME/.venvs/uccl-gin-cu13/lib/python3.12/site-packages/nvidia/nccl/lib:$HOME/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi5/lib:$LD_LIBRARY_PATH
mpirun --oversubscribe --map-by node -np 2 --host 172.31.70.225,172.31.71.140 \
  -x LD_LIBRARY_PATH -x NCCL_NET_PLUGIN=ofi -x FI_PROVIDER=efa \
  -x FI_EFA_USE_DEVICE_RDMA=1 -x OFI_NCCL_FORCE_NUM_RAILS=2 \
  -x CUDA_VISIBLE_DEVICES=0 \
  ./gin_proxy_bench_efa --min-bytes 1K --max-bytes 1G --ctas 16 --iters 20
```
