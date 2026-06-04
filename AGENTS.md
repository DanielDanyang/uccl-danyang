# Agents 记录

## 目标

在 AWS `p5en_0` 和 `p5en_1` 两台机器上，用隔离的 Python 虚拟环境开发和验证 `/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/` 里的 `ep/` native DeepEP V2 on AWS EFA 路径。当前目标是像原 UCCL-EP V1 一样把 transport substrate 留在 UCCL 内部，同时把 DeepEP V2 依赖作为独立 submodule 固定在仓库里，避免依赖外部 DeepEP 工作区或安装态源码。

## 远端主机

- `p5en_0`
  - hostname: `ip-172-31-78-36`
  - 内网 IP: `172.31.78.36`
  - 实例类型: `p5en.48xlarge`
- `p5en_1`
  - hostname: `ip-172-31-72-96`
  - 内网 IP: `172.31.72.96`
  - 实例类型: `p5en.48xlarge`

## 已确认设备信息

- OS: Ubuntu 24.04，kernel `6.14.0-1018-aws`
- GPU: 每台 8 张 `NVIDIA H200 SXM 141GB`
- GPU 拓扑: 同节点有 NVSwitch 设备，`nvidia-smi` 显示 MIG disabled
- NVIDIA driver: `580.105.08`
- `nvidia-smi` CUDA runtime: `13.0`
- `/usr/local/cuda` 指向 `/usr/local/cuda-12.9`
- `nvcc`: CUDA `12.9.86`
- Python: 系统 Python `3.12.3`
- 网络:
  - VPC/管理网卡: `enp71s0`
  - `p5en_0`: `172.31.78.36/20`
  - `p5en_1`: `172.31.72.96/20`
  - 每台有 16 个 AWS EFA PCI 设备，另有 1 个 ENA 设备
  - `ibv_devinfo` 可见 EFA RDMA 设备，端口状态 `PORT_ACTIVE`，`link_layer: Unspecified`
- AWS 通信栈:
  - `efa` package: `3.0.0-1.amzn1`
  - `efa-nv-peermem`: `1.2.3-1.amzn1`
  - `libfabric-aws`: `2.4.0amzn1.0`
  - `libnccl-ofi`: `1.18.0-1`
  - NCCL OFI plugin 在 `/opt/amazon/ofi-nccl/lib`
  - libfabric 在 `/opt/amazon/efa/lib`
  - 系统 OFI plugin 是 `aws-ofi-nccl 1.18.0`
  - 另已在用户目录构建：
    - release `v1.19.2`: `/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-1.19.2`
    - master `git-c8a3df2`: `/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master`

## 当前 UCCL/EP 仓库状态

- 本地路径: `/Users/daniel/Documents/code/uccl-danyang/`
- 远端路径: `/home/ubuntu/efs/yzhou/playground/daniel/uccl-danyang/`
- 当前主线: `ep/`，不是旧 `DeepEP-danyang/`，也不是已废弃的 `uccl-ep/`。
- 原 UCCL-EP V1 路径必须保持可用，不要为了 V2 修改这些路径：
  - `ep/deep_ep_wrapper/`
  - `ep/bench/test_internode.py`
  - `ep/bench/test_low_latency.py`
  - 旧 V1 static kernel 和 binding：`internode.cu`、`intranode.cu`、`internode_ll.cu`、`layout.cu`、`ep_runtime.cu`、原 V1 Python binding。
  - 旧第三方库 `thirdparty/DeepEP/`。
- DeepEP V2 依赖使用新增的独立 git submodule，不更新、不替换旧 `thirdparty/DeepEP/`：
  - 路径: `thirdparty/DeepEP-v2-d4f41e4/`
  - 上游: `https://github.com/deepseek-ai/DeepEP`
  - 固定 commit: `d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb`
  - 需要递归初始化它自己的上游 submodule `third-party/fmt`。
- V2 wrapper 默认从 `thirdparty/DeepEP-v2-d4f41e4/deep_ep` 取 DeepEP V2 Python/JIT 资源。
- V2 C++ JIT bridge 默认 include `thirdparty/DeepEP-v2-d4f41e4/csrc/jit/*`。
- 服务器构建统一使用 `/usr/local/cuda-13.0`，不要依赖 `/usr/local/cuda` 软链。

## 当前验证结果

- 单机 `p5en_0`:
  - `EP_DISABLE_GIN=1`
  - `tests/elastic/test_ep.py --num-processes 2 --test-first-only --skip-perf-test` 通过。
  - `tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test` 通过。
- 多机系统 OFI `1.18.0`:
  - NCCL 能加载 EFA/OFI，但报 `Communicator does not support symmetric memory`，不满足 DeepEP V2。
- 多机 OFI `v1.19.2`:
  - 已有 `ncclGinPlugin_v11`，但缺 master 后续 proxy GIN 的 `iget` / `iflush` 等补丁。
  - `EP_DISABLE_GIN=1` 或启用旧 GIN 路径都会在 first dispatch 附近失败。
- 多机 OFI master `git-c8a3df2`:
  - 动态符号包含 `ncclNetPlugin_v12`、`ncclGinPlugin_v13`、`ncclGinPlugin_v11`。
  - 启用 GIN proxy（不要设置 `EP_DISABLE_GIN=1`，也不要设置 `OFI_NCCL_GIN_GDAKI=1`）后通过。
  - 2 节点 x 2 卡：`tests/elastic/test_ep.py --num-processes 2 --test-first-only --skip-perf-test --num-sms 20` 通过。
  - 2 节点 x 8 卡，即 EP16：`tests/elastic/test_ep.py --num-processes 8 --test-first-only --skip-perf-test --num-sms 20` 通过。
- 性能 first-case 结果（参考 README 表格格式，只测 `--test-first-only` 的第一组配置）：
  - 单机 EP8，`EP_DISABLE_GIN=1`，`#SM=64`：dispatch bottleneck `331 GB/s (NVLink/SU)`，combine bottleneck `343 GB/s (NVLink/SU)`。
  - 双机 EP16，aws-ofi-nccl master proxy GIN，`#SM=20`：dispatch bottleneck `5 GB/s (RDMA/SO)`，combine bottleneck `15 GB/s (RDMA/SO)`。
  - 日志在 `/tmp/deepep_perf_single_ep8.log`、`/tmp/deepep_perf_dual_ep16_rank0.log`、`/tmp/deepep_perf_dual_ep16_rank1.log`。
- profiling 初步结论：
  - 普通 NCCL EFA path 不慢：`nccl-tests all_reduce_perf` EP16 1 GiB 约 `237 GB/s algbw` / `444 GB/s busbw`；`alltoall_perf` EP16 1 GiB 约 `91 GB/s algbw` / `85 GB/s busbw`。
  - DeepEP EP16 慢主要不像是底层 EFA/NCCL net bandwidth 问题，而更像 device-side NCCL GIN proxy 路径或 DeepEP GIN kernel 协议瓶颈。
  - `num_sms` sweep 中 dispatch bottleneck 只从 `3 GB/s` 提升到 `5-6 GB/s`，combine bottleneck 到 `15-21 GB/s`，说明单纯调 SM 不能把 dispatch 拉到 README CX7 EP16 的 `90 GB/s` 量级。
  - aws-ofi-nccl master 源码有 `tests/functional/gin.cpp` 可单测 `iput` / `iputSignal` / `iget` / `iflush`，但当前 build 是 `--disable-tests`；等 GPU 空闲后应继续构建并跑 functional GIN 或专门 microbenchmark。
- 下一步真正要测的是单独的 device-side NCCL GIN proxy 性能，不再通过 DeepEP 间接推断：
  - 参考 NVIDIA NCCL 2.30 Device-Initiated Communication / GIN Device Kernel 教程。
  - host 侧创建 NCCL communicator，使用 `ncclMemAlloc` + `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` 注册 symmetric window。
  - `ncclDevCommRequirements` 需要设置 `worldGinBarrierCount`、`ginSignalCount`、`ginConnectionType=NCCL_GIN_CONNECTION_FULL`，再调用 `ncclDevCommCreate`。
  - device kernel 直接用 `ncclGin`，核心路径为 `put(..., ncclGin_SignalInc{signalIndex})`、`waitSignal(...)`、`flush(...)`；这可以测纯 proxy GIN all-to-all/put 带宽，避免 DeepEP layout/copy/reduce 干扰。
  - NCCL 文档提醒 GIN kernel 对 NCCL compile/runtime 版本敏感，NCCL 升级后需要重新编译 device code。
- 纯 device-side NCCL GIN proxy microbenchmark 已添加并在服务器构建：
  - 源码：`/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/tools/gin_proxy_bench.cu`
  - 二进制：`/home/ubuntu/efs/yzhou/playground/daniel/DeepEP-danyang/tools/gin_proxy_bench`
  - 路径：NCCL 2.30.4 + aws-ofi-nccl master `git-c8a3df2`，`unset OFI_NCCL_GIN_GDAKI`，即 proxy GIN。
  - EP16，`ctas=16`，`OFI_NCCL_FORCE_NUM_RAILS=4`，大包上限约 `44.4 GB/s` per-rank remote bandwidth；对应 aggregate remote 约 `710 GB/s`。
  - size sweep 日志：`/tmp/gin_proxy_bench_ep16_cta16_1m_256m.log`。
  - CTA/context sweep：`ctas=1/2/4/8/16/32` 在 64 MiB/peer 上都约 `43.5-44.0 GB/s`，说明增加 GIN contexts 不是当前大包瓶颈。
  - rails sweep：`rails=1` 约 `22.2 GB/s`；`rails=2` 和 `rails=4` 约 `44.0-44.2 GB/s`；`rails=8` 在 `ncclCommInitRank` 里触发 aws-ofi-nccl master segfault，未继续测 `rails=16`。
  - 纯 P2P 模式使用 `--skip-self`，2 节点各 1 rank，GPU0 到 GPU0。1 GiB 大包结果：`rails=1` 约 `23.1 GB/s`，`rails=2` 约 `44.8 GB/s`，`rails=4` 约 `43.5 GB/s`。
  - P2P debug 日志 `/tmp/gin_proxy_p2p_rails4_debug.log` 确认 aws-ofi-nccl 找到每台 `16` 个 EFA NIC，组织成 `8` 个 NCCL Net devices；单个 P2P 流只看到 `4` 个 channels/rails，且 rails=2/4 性能相同，不是用满 16 个物理网卡。
  - 新增 `--remote-only` 后的 EP16 整机大包测试：`rails=2`、每 rank 发远端 8 peers，1 GiB/peer 时 per-rank `46.3 GB/s`，双向 aggregate `740 GB/s`，折合单向每 node `370 GB/s`，接近 `16 x 200 Gbps = 400 GB/s` 的 EFA 理论上限。
  - 新增 `--same-local-remote-only --message-bytes` 后的 DeepEP-like small-message 测试：每 rank 只发另一台机器同 local rank peer，64 MiB/rank 时 4 KiB message `2.8 GB/s`、8 KiB `5.1 GB/s`、16 KiB `8.8 GB/s`、32 KiB `12.5 GB/s`。这和 DeepEP dispatch `2-3 GB/s (SO)` 同量级。
- DeepEP dispatch profiling 结论：
  - README 风格 EP16 配置：`--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256 --num-sms 20 --ignore-local-traffic`。
  - 干净日志：`/tmp/deepep_profile_ep16_clean_rank0.log`、`/tmp/deepep_profile_ep16_clean_rank1.log`。
  - trace：`/tmp/deepep_profile_ep16_rank0/*.json`、`/tmp/deepep_profile_ep16_rank1/*.json`。
  - dispatch / cached dispatch 的主耗时都是 `hybrid_dispatch_impl`，rank0 约 `23.9 ms/iter`；`dispatch_copy_epilogue_impl` 约 `3.3 ms/iter`。`spin_kernel` 是 profiling barrier 的 `torch.cuda._sleep`，不是 dispatch 本体。
  - 源码 `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` 显示 scaleout warps 按 token 调用 `gin.put<ncclTeamTagRail>(..., tma_buffer.get_num_bytes<false>(), ...)`；FP8 hidden=7168 时单 token payload 约 7-8 KiB，并伴随 tail/count 的 `red_add_rel` / `put_value` 小 signal。
  - 结合 UCCL EFA programming 文章，EFA 大包吞吐强，但没有 native RDMA atomics 和 ordering，小消息/细粒度 GPU-initiated 通信会被 SRD/proxy/reordering/软件 atomic 路径限制；这解释了 DeepEP dispatch 在 AWS EFA 上明显低于 CX7/IB 官方机器。

关键多机环境变量：

```bash
export LD_LIBRARY_PATH=/home/ubuntu/efs/yzhou/playground/daniel/aws-ofi-nccl-master/lib:/opt/amazon/efa/lib:/home/ubuntu/.venvs/deepep-danyang-cu13/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
unset EP_DISABLE_GIN
unset OFI_NCCL_GIN_GDAKI
export NCCL_NET_PLUGIN=ofi
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export OFI_NCCL_FORCE_NUM_RAILS=4
export NCCL_SOCKET_IFNAME=enp71s0
```

日志确认：

- `NET/OFI Initializing aws-ofi-nccl git-c8a3df2`
- `NET/Plugin: Loaded net plugin Libfabric (v12)`
- `NET/Plugin: Loaded gin plugin Libfabric (v13)`
- `devCommCreate: creating 129 contexts`
- `GIN Proxy will not be using GDRCopy`

## Native V2 EP 当前方向和开发准则

- 当前主线是 `ep/`，不是 `uccl-ep/`。`uccl-ep/` 已废弃，不再作为开发、测试或
  benchmark 目标。
- `ep/` backend 必须以 DeepEP V2 的 JIT `.cuh`、`BufferLayout`、`TokenLayout`、
  expanded dispatch、reduced combine、handle/cache 语义为核心。
- `ep/` 必须优先复用原 `uccl/ep` 的 transport substrate：GPU 写 D2H FIFO、CPU
  proxy drain、EFA post、CQ poll、completion/ack 回传、quiet/barrier、thread
  pinning、peer metadata exchange。
- `ep/` 不应复用 V1 的 EP 语义层：`SourceMeta`、prefix matrix、packed/staged token
  buffer、V1 prepare/dispatch/combine binding、旧 static kernel。
- 当前 native V2 command 方向：
  - 主路径直接复用原 `uccl/ep` 的旧 16B `TransferCmd`，不要把差异很大的
    `V2TransferCmd` 作为 dispatch/combine 主路径 wire command。
  - V2 JIT 负责把 V2 `buffer/workspace/signal_scratch` 指针计算成 unified transport
    window offset，然后直接写旧 `TransferCmd`。
  - signal/tail/count 不新增 immediate command 字段；先把 value 写入 registered
    signal scratch slot，再用旧 `TransferCmd` 普通 WRITE 写到远端 workspace word。
  - lane 不进入 command；用 `channel_idx % num_fifo_queues` 选择 D2H queue/proxy/lane。
  - 多个 D2H queue 只表示 channel/proxy thread/NIC lane 并行，不表示 dispatch/combine
    语义分队列。V1 没有按 dispatch/combine 分队列，V2 也不应该分。
- 当前必须坚持的 V2 descriptor 方向：
  - dispatch descriptor/JIT 应直接描述和计算 `dst_rank, queue/channel, expert_id,
    src_token range, expanded_slot range, count, payload bytes`，最终落成旧
    `TransferCmd` 的 local/remote offset。
  - combine descriptor 应直接描述 V2 reduced-combine 的反向 gather/reduce/send 路径。
  - receiver 应直接写入 V2 expanded layout 或 reduced-combine 目标区域，避免回到 V1
    packed token staging。
- 如果 V2 中任何设计和原 `uccl/ep` 不同，必须在当前计划或设计文档里写明理由。默认沿用
  V1 transport 方法；只有当 V2 语义、buffer layout、JIT/cache 或 AWS EFA 约束使原方法
  无法表达时，才允许不同。
- 如果新增 queue、command 字段、buffer、metadata 或 proxy 状态，必须说明它对应的 V2
  语义来源，不能只因为实现方便而新增。
- 如果遇到 bug，不要优先自创新策略；先检查原 `uccl/ep` 是否已有类似逻辑可以复用。
  能用原逻辑就用原逻辑；如果必须自己写，必须在计划或文档里给出充分理由。
- 不要为了跑通 correctness 写 fallback、临时路径、semantic all-to-all、Python
  materialize、overlay、dummy/scaffold 路径进主代码。可以分模块做 isolated test，但
  主 dispatch/combine path 必须朝最终方案写。
- 不要 commit 一段“只跑通 correctness、性能明显不对”的 fallback 路径作为完成状态。
  代码可以分阶段，但每个阶段都应服务于最终 native V2 + old `TransferCmd` + UCCL
  proxy substrate 方案。
- `worklog.md` 需要持续记录操作、测试命令、benchmark 数据和重要设计结论。
- 重要开发进度应在本地及时 commit，使用用户的 git/GitHub 身份，不添加 Codex
  co-author。

## 约束

- 不污染别人环境：使用用户目录下专用 venv `/home/ubuntu/.venvs/deepep-danyang-cu13`。
- 不打断别人任务：在任何服务器上的构建、测试、profiling、benchmark、采样或排查操作之前，必须先确认
  `p5en_0` 和 `p5en_1` 的 GPU 空闲。至少执行：
  - `ssh p5en_0 nvidia-smi`
  - `ssh p5en_1 nvidia-smi`
  - 必要时补充 `ssh p5en_0 "ps -eo user,pid,ppid,stat,pcpu,pmem,cmd | grep -E 'python|torch|cuda|nccl|deepep|uccl' | grep -v grep"`，
    `p5en_1` 同理。
- 如果 `nvidia-smi` 或进程列表显示其他用户/未知任务正在占用 GPU，必须立即停止所有远端构建、测试、
  profiling、benchmark、采样和排查操作；不要继续跑轻量/重型实验，不要 kill、抢占或影响别人的训练/
  benchmark 进程。只允许在本地整理记录、分析代码、更新文档，并等待用户下一步指示。
- 如果只看到自己刚启动的残留进程，应先确认来源；除非用户明确允许，不要 kill 任何可能属于他人的进程。
- DeepEP JIT 默认写 `$HOME/.deep_ep`；如需长期多人共用，建议后续再切到专用 `EP_JIT_CACHE_DIR`。
- 先跑最小正确性验证，确认构建、import、单机通信和 JIT 都通，再扩大到两机。
- 多机阶段必须使用 aws-ofi-nccl master 的 GIN proxy 路径；禁用 Gin 只适合单机验证。
