from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from uccl import ep

from ..utils.event import EventOverlap


_NATIVE_V2_REWRITE_MESSAGE = (
    "uccl-ep is being rewritten as a native DeepEP V2 AWS EFA backend. "
    "The previous V1/UCCL EP transport path has been removed, and dispatch/"
    "combine must be implemented through V2 JIT .cuh kernels before this "
    "ElasticBuffer can run benchmarks."
)


def _align(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _align_2mb(value: int) -> int:
    return _align(value, 2 << 20)


def _native_workspace_bytes() -> int:
    # Mirrors deep_ep/common/layout.cuh::WorkspaceLayout::get_num_bytes(),
    # aligned like csrc/elastic/buffer.hpp so the future native fork sees the
    # same workspace shape as upstream DeepEP V2.
    max_ranks = 1024
    max_experts = 2048
    max_channels = 1024
    max_inflight_agrs = 32
    num_bytes = 16
    num_bytes += (max_ranks + max_experts) * 8
    num_bytes += max_ranks * 8 * 2
    num_bytes += max_experts * 8 * 2
    num_bytes += max_ranks * 4
    num_bytes += max_ranks * 4 * 2
    num_bytes += max_experts * 4 * 2
    num_bytes += max_ranks * max_channels * 8
    num_bytes += max_ranks * max_channels * 4
    num_bytes += 2 * 2 * 8
    num_bytes += (max_inflight_agrs + 1) * max_ranks * 4
    return _align(num_bytes, 2 << 20)


def _d2h_queue_capacity() -> int:
    capacity = int(ep.d2h_queue_capacity())
    if capacity <= 0 or capacity & (capacity - 1):
        raise RuntimeError(f"invalid UCCL D2H queue capacity: {capacity}")
    return capacity


@dataclass
class EPHandle:
    """DeepEP V2 dispatch handle shape.

    This remains as an API placeholder for the native V2 backend. It must be
    populated by V2 descriptor/JIT dispatch, not by the removed staged-token
    transport metadata.
    """

    do_expand: bool
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    num_sms: int
    topk_idx: torch.Tensor
    num_recv_tokens_per_expert_list: list
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor
    dst_buffer_slot_idx: torch.Tensor
    token_metadata_at_forward: Optional[torch.Tensor]
    channel_linked_list: Optional[torch.Tensor]
    transport_handle: Optional[object] = None


@dataclass
class V2TransportHandle:
    dispatch_segments: torch.Tensor
    dispatch_batches: torch.Tensor
    dispatch_route_offsets: torch.Tensor
    dispatch_counters: torch.Tensor
    combine_segments: Optional[torch.Tensor]
    combine_batches: Optional[torch.Tensor]
    combine_counters: Optional[torch.Tensor]
    d2h_queue: object
    combine_d2h_queue: Optional[object]
    dispatch_layout: dict
    combine_layout: Optional[dict]
    num_dispatch_batches: int
    num_dispatch_segments: int
    num_combine_batches: int
    num_combine_segments: int
    payload_bytes: int
    scale_bytes: int
    dispatch_drain_stats: Optional[dict] = None
    combine_drain_stats: Optional[dict] = None
    timings: Optional[dict] = None


class ElasticBuffer:
    """Native V2 AWS EFA buffer surface.

    Construction is intentionally lightweight so tests can inspect topology,
    descriptor sizes, and workspace plans while dispatch/combine are being
    ported to JIT kernels.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_bytes: Optional[int] = None,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        sl_idx: int = 3,
        num_allocated_qps: int = 0,
        num_cpu_timeout_secs: int = 300,
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ) -> None:
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.explicitly_destroy = explicitly_destroy
        self._destroyed = False

        local_world = int(
            os.environ.get("LOCAL_WORLD_SIZE")
            or os.environ.get("LOCAL_SIZE")
            or torch.cuda.device_count()
            or 1
        )
        self.num_scaleup_ranks = min(max(1, local_world), self.num_ranks)
        self.num_scaleout_ranks = max(1, self.num_ranks // self.num_scaleup_ranks)
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks

        self.num_experts = self.num_ranks
        self.elem_bytes = 1 if use_fp8_dispatch else 2
        self.num_sms = 0
        self.runtime = self._make_runtime(
            num_experts=self.num_experts,
            num_topk=max(1, self.num_topk),
            hidden=self.hidden,
            elem_bytes=self.elem_bytes,
            num_sms=self.num_sms,
        )
        self.num_bytes = num_bytes or self.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            use_fp8_dispatch,
            allow_hybrid_mode,
            allow_multiple_reduction,
        )
        self.num_allocated_qps = int(num_allocated_qps or 129)
        self.allow_hybrid_mode = bool(allow_hybrid_mode)
        self.allow_multiple_reduction = bool(allow_multiple_reduction)
        self.prefer_overlap_with_compute = bool(prefer_overlap_with_compute)
        # Native UCCL EFA transport: persistent UcclProxy threads over the single
        # DeepEP symmetric window (workspace+buffer registered as one MR).
        self._v2_proxies = None                 # list[ep.Proxy]
        self._v2_window_base = 0                # registered window base (raw symmetric)
        self._v2_window_bytes = 0              # registered window size
        self._v2_d2h_queue_ptrs: Optional[torch.Tensor] = None  # int64[]: DeviceToHostCmdBuffer*
        self._v2_num_d2h_queues = 0
        self._v2_signal_scratch_base = 0       # mapped scratch base (window tail)
        self._v2_efa_num_lanes = 1
        self._native_v2_resources: Optional[dict] = None

    def _make_runtime(
        self,
        num_experts: int,
        num_topk: int,
        hidden: int,
        elem_bytes: int,
        num_sms: int,
    ):
        config = ep.V2EfaRuntimeConfig()
        config.rank = self.rank_idx
        config.world_size = self.num_ranks
        config.scaleout_rank = self.scaleout_rank_idx
        config.scaleup_rank = self.scaleup_rank_idx
        config.num_scaleout_ranks = self.num_scaleout_ranks
        config.num_scaleup_ranks = self.num_scaleup_ranks
        config.num_experts = int(num_experts)
        config.num_topk = int(num_topk)
        config.hidden = int(hidden)
        config.elem_bytes = int(elem_bytes)
        config.num_sms = int(num_sms)
        return ep.V2EfaRuntime(config)

    def configure_native_v2(
        self,
        num_experts: int,
        num_topk: Optional[int] = None,
        hidden: Optional[int] = None,
        elem_bytes: Optional[int] = None,
        num_sms: int = 0,
    ) -> None:
        self.num_experts = int(num_experts)
        self.num_topk = int(self.num_topk if num_topk is None else num_topk)
        self.hidden = int(self.hidden if hidden is None else hidden)
        self.elem_bytes = int(self.elem_bytes if elem_bytes is None else elem_bytes)
        self.num_sms = int(num_sms)
        self.runtime = self._make_runtime(
            self.num_experts,
            max(1, self.num_topk),
            self.hidden,
            self.elem_bytes,
            self.num_sms,
        )

    def destroy(self) -> None:
        self._destroyed = True

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        if with_cpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier(self.group)
        if with_cpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_logical_domain_size(self) -> Tuple[int, int]:
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def get_physical_domain_size(self) -> Tuple[int, int]:
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = 200,
        sm_write_gbs: float = 50,
    ) -> int:
        del num_experts, num_topk, num_scaleout_topk, rdma_gbs, nvlink_gbs
        del sm_read_gbs, sm_write_gbs
        if self.num_sms > 0:
            return self.num_sms
        if not torch.cuda.is_available():
            return 4
        return min(20 if self.num_scaleout_ranks > 1 else 64,
                   torch.cuda.get_device_properties("cuda").multi_processor_count)

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        # DeepEP V2's upstream formula is tuned for its NCCL GIN scaleout
        # transport and can request num_sms * 16 + 1 QPs in hybrid mode.  The
        # native AWS path replaces scaleout GIN with UCCL proxy/D2H queues, so
        # QPs here only configure the remaining scaleup/NVLink GIN resource
        # sharing.  Match the original UCCL-EP default and keep it stable unless
        # the caller explicitly sweeps it.
        del num_sms
        num_qps = int(os.environ.get("UCCL_V2_NUM_QPS", "24"))
        if num_qps <= 0:
            raise RuntimeError(f"UCCL_V2_NUM_QPS must be positive, got {num_qps}")
        return min(num_qps, self.num_allocated_qps)

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
    ) -> int:
        elem_bytes = 1 if use_fp8_dispatch else 2
        token_bytes = hidden * elem_bytes
        metadata_bytes = max(num_topk, 1) * 16
        world = group.size()
        raw_bytes = max(1, world * num_max_tokens_per_rank * (token_bytes + metadata_bytes) * 4)
        return _align_2mb(raw_bytes)

    @staticmethod
    def capture() -> EventOverlap:
        return EventOverlap(None)

    def get_native_v2_status(self) -> str:
        return self.runtime.status()

    def init_native_v2_deep_ep_resources(
        self,
        *,
        nccl_dev_comm_ptr: int,
        nccl_window_ptr: int,
        buffer_ptr: int,
        buffer_bytes: int,
        workspace_ptr: int,
        workspace_bytes: int,
        mapped_host_workspace_ptr: int,
        host_workspace_ptr: int = 0,
    ) -> None:
        resources = {
            "nccl_dev_comm_ptr": int(nccl_dev_comm_ptr),
            "nccl_window_ptr": int(nccl_window_ptr),
            "buffer_ptr": int(buffer_ptr),
            "buffer_bytes": int(buffer_bytes),
            "workspace_ptr": int(workspace_ptr),
            "workspace_bytes": int(workspace_bytes),
            "mapped_host_workspace_ptr": int(mapped_host_workspace_ptr),
            "host_workspace_ptr": int(host_workspace_ptr),
        }
        missing = [
            name for name, value in resources.items()
            if value <= 0 and name != "host_workspace_ptr"
        ]
        if missing:
            raise ValueError(f"native V2 DeepEP resources contain null fields: {missing}")
        self._native_v2_resources = resources

    def _native_num_channels_per_sm(
        self,
        hidden_bytes: int,
        sf_bytes: int,
        num_topk: int,
        smem_bytes: int = 228 * 1024,
    ) -> int:
        notify_smem = 0
        if self.num_scaleout_ranks > 1:
            notify_smem = _align(self.num_ranks + self.num_experts, 4 * 32) * 4
        token_bytes = _v2_token_layout_bytes(hidden_bytes, sf_bytes, num_topk)
        channels = min(
            max(1, (int(smem_bytes) - notify_smem) // max(1, token_bytes)),
            32 - 4,
        )
        channels = min(max(1, channels // 2), 8)
        if self.num_scaleup_ranks > 1:
            channels = min(channels, 4)
        return max(1, int(channels))

    def route_expert(self, expert_id: int):
        return self.runtime.route_expert(int(expert_id))

    def init_native_v2_efa_transport(
        self,
        *,
        window_base: int,
        window_bytes: int,
        scratch_region_base: int,
        scratch_region_bytes: int,
        num_lanes: int = 1,
    ):
        """Set up the native UCCL EFA transport for the DeepEP symmetric window.

        Mirrors the V1 ``init_uccl`` setup: spin up ``get_num_proxy_threads()``
        persistent ``UcclProxy`` threads, each registering the single window
        ``[window_base, window_base+window_bytes)`` (raw symmetric address) as its
        RDMA MR, exchange peer meta (listen ports + window base) over the torch
        group, connect, and start them in dual mode.  The proxies' D2H command
        rings become the kernel's ``d2h_queues**``.  ``signal_scratch`` is placed
        in ``[scratch_region_base, scratch_region_base+scratch_region_bytes)`` (the
        mapped CPU/engram segment of the window) so it never overlaps the GPU
        dispatch/combine buffer.
        """
        if not hasattr(ep, "Proxy"):
            raise RuntimeError("uccl.ep was built without the UcclProxy transport")

        # TransferCmd WRITE offsets are 32-bit, shifted by 2 (4-byte granularity),
        # so the registered window must fit the 16 GiB encodable range and be
        # 4-byte aligned; otherwise device-side offsets silently truncate.
        if int(window_bytes) <= 0 or int(window_bytes) > (1 << 34):
            raise RuntimeError(
                f"V2 EFA window {int(window_bytes)} B exceeds the 16 GiB TransferCmd "
                f"offset-encoding range")
        if int(window_base) % 4 != 0:
            raise RuntimeError("V2 EFA window base must be 4-byte aligned")

        rank = int(self.rank_idx)
        num_ranks = int(self.num_ranks)
        local_rank = int(os.environ.get("LOCAL_RANK", self.scaleup_rank_idx))
        node_idx = int(self.scaleout_rank_idx)
        num_nodes = int(self.num_scaleout_ranks)
        is_intranode = num_nodes <= 1

        num_proxy_threads = int(ep.get_num_proxy_threads())
        proxies = [
            ep.Proxy(
                thread_idx=i,
                gpu_buffer_addr=int(window_base),
                total_size=int(window_bytes),
                rank=rank,
                node_idx=node_idx,
                local_rank=local_rank,
                num_experts=int(self.num_experts),
                num_ranks=num_ranks,
                num_nodes=num_nodes,
                use_normal_mode=True,
                is_intranode=is_intranode,
            )
            for i in range(num_proxy_threads)
        ]

        # Peer-meta exchange (mirror get_cpu_proxies_meta): each rank advertises
        # its window base/size and per-thread listen ports; the proxy completes
        # the rkey handshake over those ports during start_dual().
        my_ip = ep.get_oob_ip()
        meta = {
            "rank": rank,
            "ptr": int(window_base),
            "nbytes": int(window_bytes),
            "ip": my_ip,
            "listen_ports": [p.get_listen_port() for p in proxies],
        }
        all_meta = [None] * num_ranks
        dist.all_gather_object(all_meta, meta, group=self.group)
        rank2meta = {m["rank"]: m for m in all_meta}
        peers = [rank2meta[r] for r in range(num_ranks)]
        if not is_intranode:
            for p in proxies:
                p.set_peers_meta(peers)
        ep.register_proxies(local_rank, proxies)
        dist.barrier(self.group)
        if not is_intranode:
            for p in proxies:
                p.start_dual()
        time.sleep(1)

        # GPU-resident array of DeviceToHostCmdBuffer* (one per proxy D2H channel).
        d2h_addrs = []
        for p in proxies:
            d2h_addrs.extend(int(a) for a in p.get_d2h_channel_addrs())
        if not d2h_addrs:
            raise RuntimeError("UcclProxy exposed no D2H channels")
        self._v2_d2h_queue_ptrs = torch.tensor(d2h_addrs, dtype=torch.int64, device="cuda")
        self._v2_num_d2h_queues = len(d2h_addrs)

        # signal_scratch: one int64 slot per ring slot per queue, placed at the
        # start of the CPU/engram segment (mapped) so it never overlaps the GPU
        # dispatch/combine buffer.
        scratch_bytes = _align(self._v2_num_d2h_queues * _d2h_queue_capacity() * 8, 128)
        if int(scratch_region_base) == 0 or scratch_bytes > int(scratch_region_bytes):
            raise RuntimeError(
                f"signal scratch needs {scratch_bytes} B but the CPU/engram segment "
                f"is {int(scratch_region_bytes)} B; create the DeepEP buffer with "
                f"num_cpu_bytes >= {scratch_bytes}"
            )
        self._v2_signal_scratch_base = int(scratch_region_base)

        self._v2_proxies = proxies
        self._v2_window_base = int(window_base)
        self._v2_window_bytes = int(window_bytes)
        self._v2_efa_num_lanes = int(max(1, num_lanes))
        return proxies

    def has_native_v2_efa_transport(self) -> bool:
        return self._v2_proxies is not None

    def init_from_deep_ep_v2(
        self,
        deep_ep_buffer,
        num_lanes: int = 1,
        device_index: int = -1,
        signal_capacity: int = 65536,
    ) -> dict:
        """One-shot setup: extract V2 resources from an existing DeepEP ElasticBuffer
        and wire up both the EFA RDMA transport and the native dispatch resource table.

        Registers the WHOLE DeepEP symmetric window (workspace + GPU buffer + CPU
        segment) as one MR via UcclProxy; signal scratch lives in the CPU/engram
        segment so it never overlaps the GPU dispatch/combine buffer.

        Returns the full resource dict so callers can inspect addresses.
        """
        del device_index, signal_capacity  # legacy scaffold args, no longer used
        deep_ep_handle = getattr(deep_ep_buffer, "runtime", None)
        if deep_ep_handle is None:
            deep_ep_handle = getattr(deep_ep_buffer, "_handle", None)
        if deep_ep_handle is None:
            deep_ep_handle = deep_ep_buffer
        if not hasattr(deep_ep_handle, "get_native_v2_resources"):
            raise TypeError(
                "deep_ep_buffer must be a DeepEP V2 ElasticBuffer or native C++ handle "
                "with get_native_v2_resources()"
            )
        resources = deep_ep_handle.get_native_v2_resources()
        ws_bytes = int(resources["workspace_bytes"])
        gpu_bytes = int(resources["buffer_bytes"])
        cpu_bytes = int(resources.get("cpu_buffer_bytes", 0))
        # Full symmetric window [Workspace | GPU buffer | CPU segment].
        window_bytes = int(resources.get("rdma_window_bytes", ws_bytes + gpu_bytes + cpu_bytes))
        raw_window_base = int(resources.get("rdma_workspace_ptr", resources["workspace_ptr"]))
        workspace_ptr = int(resources["workspace_ptr"])
        buffer_ptr = int(resources["buffer_ptr"])
        mapped_host_workspace_ptr = int(resources["mapped_host_workspace_ptr"])
        expected_window_bytes = ws_bytes + gpu_bytes + cpu_bytes
        if window_bytes != expected_window_bytes:
            raise RuntimeError(
                f"DeepEP V2 window layout mismatch: rdma_window_bytes={window_bytes}, "
                f"expected workspace+buffer+cpu={expected_window_bytes}"
            )
        if buffer_ptr != workspace_ptr + ws_bytes:
            raise RuntimeError(
                f"DeepEP V2 buffer must be contiguous after workspace: "
                f"buffer_ptr=0x{buffer_ptr:x}, workspace_ptr+workspace_bytes="
                f"0x{workspace_ptr + ws_bytes:x}"
            )
        # Scratch in the CPU/engram segment (idle when engram unused); mapped base.
        scratch_region_base = int(resources.get("cpu_buffer_ptr", 0))
        if cpu_bytes > 0:
            expected_scratch_base = buffer_ptr + gpu_bytes
            if scratch_region_base != expected_scratch_base:
                raise RuntimeError(
                    f"DeepEP V2 CPU/scratch segment must follow the GPU buffer: "
                    f"cpu_buffer_ptr=0x{scratch_region_base:x}, expected="
                    f"0x{expected_scratch_base:x}"
                )
        mapped_window_end = workspace_ptr + window_bytes
        for name, ptr, size in (
            ("workspace", workspace_ptr, ws_bytes),
            ("buffer", buffer_ptr, gpu_bytes),
            ("signal_scratch_region", scratch_region_base, cpu_bytes),
        ):
            if size == 0:
                continue
            if ptr < workspace_ptr or ptr + size > mapped_window_end:
                raise RuntimeError(
                    f"DeepEP V2 {name} range [0x{ptr:x}, 0x{ptr + size:x}) "
                    f"is outside mapped window [0x{workspace_ptr:x}, "
                    f"0x{mapped_window_end:x})"
                )
        self.init_native_v2_efa_transport(
            window_base=raw_window_base,
            window_bytes=window_bytes,
            scratch_region_base=scratch_region_base,
            scratch_region_bytes=cpu_bytes,
            num_lanes=num_lanes,
        )
        self.init_native_v2_deep_ep_resources(
            nccl_dev_comm_ptr=int(resources["nccl_dev_comm_ptr"]),
            nccl_window_ptr=int(resources["nccl_window_ptr"]),
            buffer_ptr=buffer_ptr,
            buffer_bytes=gpu_bytes,
            workspace_ptr=workspace_ptr,
            workspace_bytes=ws_bytes,
            mapped_host_workspace_ptr=mapped_host_workspace_ptr,
            host_workspace_ptr=int(resources["host_workspace_ptr"]),
        )
        return dict(resources)

    def get_comm_stream(self) -> torch.Stream:
        return torch.cuda.current_stream()

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Optional[object] = None,
        previous_event_before_epilogue: Optional[object] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: bool = False,
        use_tma_aligned_col_major_sf: bool = False,
    ):
        del previous_event, previous_event_before_epilogue
        del async_with_compute_stream, allocate_on_comm_stream
        del use_tma_aligned_col_major_sf

        if handle is not None:
            if topk_idx is not None or topk_weights is not None:
                raise ValueError("topk_idx/topk_weights must be None when cached handle is used")
            topk_idx = handle.topk_idx
            num_experts = handle.num_experts
            num_max_tokens_per_rank = handle.num_max_tokens_per_rank
            expert_alignment = handle.expert_alignment
            do_expand = handle.do_expand
            do_cpu_sync = False if do_cpu_sync is None else do_cpu_sync

        if topk_idx is None:
            raise ValueError("topk_idx is required for uncached native V2 dispatch")
        if self._v2_proxies is None:
            raise RuntimeError("native V2 dispatch requires init_native_v2_efa_transport()")
        _require_cuda_contiguous(topk_idx, "topk_idx")
        if topk_idx.dtype != torch.int64:
            raise TypeError("topk_idx must be torch.int64")

        x_tensor, sf = x if isinstance(x, tuple) else (x, None)
        _require_cuda_contiguous(x_tensor, "x")
        if sf is not None:
            _require_cuda_contiguous(sf, "sf")
        if topk_weights is not None:
            _require_cuda_contiguous(topk_weights, "topk_weights")

        num_tokens, hidden = x_tensor.shape
        num_topk = topk_idx.shape[1]
        num_experts = int(num_experts if num_experts is not None else self.num_experts)
        num_max_tokens_per_rank = int(
            num_max_tokens_per_rank
            if num_max_tokens_per_rank is not None
            else self.num_max_tokens_per_rank
        )
        expert_alignment = int(expert_alignment if expert_alignment is not None else 1)
        # Default False: the native path reads counts on the GPU receiver; the
        # host-workspace CPU-sync reader is not implemented.
        do_cpu_sync = False if do_cpu_sync is None else bool(do_cpu_sync)
        num_sms = int(num_sms or self.get_theoretical_num_sms(num_experts, num_topk))
        num_qps = int(num_qps or self.get_theoretical_num_qps(num_sms))
        elem_bytes = int(x_tensor.element_size())
        scale_bytes = 0 if sf is None else int(sf.shape[1] * sf.element_size())
        self.configure_native_v2(num_experts, num_topk, hidden, elem_bytes, num_sms)

        if self._native_v2_resources is None:
            raise RuntimeError(
                "native V2 dispatch requires init_native_v2_deep_ep_resources(); "
                "the old scaffold/materialize dispatch path is not a production fallback"
            )

        return self._dispatch_native_hybrid(
            x_tensor=x_tensor,
            sf=sf,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            num_tokens=num_tokens,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            scale_bytes=scale_bytes,
            expert_alignment=expert_alignment,
            num_sms=num_sms,
            num_qps=num_qps,
            do_cpu_sync=do_cpu_sync,
            do_expand=do_expand,
            do_handle_copy=do_handle_copy,
        )

    def _dispatch_native_hybrid(
        self,
        *,
        x_tensor: torch.Tensor,
        sf: Optional[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        cumulative_local_expert_recv_stats: Optional[torch.Tensor],
        num_tokens: int,
        num_max_tokens_per_rank: int,
        scale_bytes: int,
        expert_alignment: int,
        num_sms: int,
        num_qps: int,
        do_cpu_sync: bool,
        do_expand: bool,
        do_handle_copy: bool,
    ):
        if self._native_v2_resources is None:
            raise RuntimeError("native V2 resources are not initialized")
        if self._v2_proxies is None:
            raise RuntimeError("native V2 EFA transport is not initialized")
        # Reject CPU-sync BEFORE launching anything: the host-workspace CPU
        # reader is not implemented, so a do_cpu_sync=True call would otherwise
        # launch the kernel + emit transport commands and only then fail, leaving
        # proxy/remote state mid-flight.
        if do_cpu_sync:
            raise RuntimeError(
                "native V2 do_cpu_sync=True is not supported yet (host-workspace "
                "CPU reader unimplemented); call dispatch(..., do_cpu_sync=False)"
            )

        smem_bytes = int(os.environ.get("UCCL_V2_SMEM_BYTES", str(224 * 1024)))
        hidden = int(x_tensor.shape[1])
        elem_bytes = int(x_tensor.element_size())
        num_topk = int(topk_idx.shape[1])
        num_local_experts = int(self.num_experts // self.num_ranks)
        num_sf_packs = 0 if sf is None else int(sf.shape[1])
        sf_token_stride = 0 if sf is None else int(sf.stride(0))
        sf_hidden_stride = 0 if sf is None else int(sf.stride(1))
        hidden_bytes = hidden * elem_bytes
        sf_bytes = int(scale_bytes)
        num_channels_per_sm = self._native_num_channels_per_sm(
            hidden_bytes, sf_bytes, num_topk, smem_bytes
        )
        num_channels = int(num_sms) * int(num_channels_per_sm)
        num_max_tokens_per_channel = max(
            1, (int(num_max_tokens_per_rank) + num_channels - 1) // num_channels
        )
        max_forwarded_tokens = self.num_scaleout_ranks * num_max_tokens_per_channel + 1
        forward_dims = 2 + num_topk * 2

        psum_scaleup = torch.empty((self.num_scaleup_ranks,), dtype=torch.int32, device=x_tensor.device)
        psum_expert = torch.empty((num_local_experts + 1,), dtype=torch.int32, device=x_tensor.device)
        dst_buffer_slot_idx = torch.empty(
            (num_channels, self.num_scaleout_ranks, num_max_tokens_per_channel, num_topk),
            dtype=torch.int32,
            device=x_tensor.device,
        )
        token_metadata_at_forward = torch.empty(
            (num_channels, max_forwarded_tokens, forward_dims),
            dtype=torch.int32,
            device=x_tensor.device,
        )
        channel_linked_list = torch.empty(
            (num_channels, max_forwarded_tokens, self.num_scaleup_ranks),
            dtype=torch.int32,
            device=x_tensor.device,
        )
        copied_topk_idx = topk_idx.clone() if do_handle_copy else topk_idx

        # Native UCCL transport: persistent UcclProxy D2H queues (created in
        # init_native_v2_efa_transport).  The kernel pushes old TransferCmds into
        # d2h_queues[channel % num_queues]; the owning proxy thread drains and
        # posts the RDMA write.  signal_scratch lives at the registered window
        # tail.  No per-dispatch queue/layout allocation any more.
        if self._v2_proxies is None or self._v2_d2h_queue_ptrs is None:
            raise RuntimeError("native V2 dispatch requires init_native_v2_efa_transport()")
        queues = None
        layout = None
        try:
            self.runtime.launch_native_hybrid_dispatch(
                int(x_tensor.data_ptr()),
                0 if sf is None else int(sf.data_ptr()),
                int(topk_idx.data_ptr()),
                0 if topk_weights is None else int(topk_weights.data_ptr()),
                int(copied_topk_idx.data_ptr()),
                0 if cumulative_local_expert_recv_stats is None else int(cumulative_local_expert_recv_stats.data_ptr()),
                int(psum_scaleup.data_ptr()),
                int(psum_expert.data_ptr()),
                int(dst_buffer_slot_idx.data_ptr()),
                int(token_metadata_at_forward.data_ptr()),
                int(num_tokens),
                int(num_max_tokens_per_rank),
                int(num_channels_per_sm),
                int(num_sf_packs),
                int(sf_token_stride),
                int(sf_hidden_stride),
                int(expert_alignment),
                int(num_qps),
                int(os.environ.get("UCCL_V2_GPU_TIMEOUT_CYCLES", "200000000000")),
                False,  # cached_mode
                False,  # deterministic
                bool(do_cpu_sync),
                int(smem_bytes),
                int(self._native_v2_resources["nccl_dev_comm_ptr"]),
                int(self._native_v2_resources["nccl_window_ptr"]),
                int(self._native_v2_resources["buffer_ptr"]),
                int(self._native_v2_resources["workspace_ptr"]),
                int(self._native_v2_resources["mapped_host_workspace_ptr"]),
                # EFA transport (new short ABI): D2H queue array, count, scratch.
                int(self._v2_d2h_queue_ptrs.data_ptr()),
                int(self._v2_num_d2h_queues),
                int(self._v2_signal_scratch_base),
                str(Path(__file__).resolve().parents[3] / "include"),
                _cuda_stream_ptr(torch.cuda.current_stream()),
            )

            # (do_cpu_sync=True is rejected before launch, above.)
            num_recv_tokens = int(num_max_tokens_per_rank) * int(self.num_ranks)
            num_expanded_tokens = (
                self.num_ranks * int(num_max_tokens_per_rank) * min(num_topk, num_local_experts)
            )
            num_expanded_tokens = _align(num_expanded_tokens + (expert_alignment - 1) * num_local_experts, expert_alignment)
            num_allocated_tokens = num_expanded_tokens if do_expand else num_recv_tokens
            recv_x = torch.empty((num_allocated_tokens, hidden), dtype=x_tensor.dtype, device=x_tensor.device)
            recv_sf = None
            recv_topk_idx = None if do_expand else torch.empty(
                (num_allocated_tokens, num_topk), dtype=topk_idx.dtype, device=topk_idx.device
            )
            recv_topk_weights = None
            if topk_weights is not None:
                weight_shape = (num_allocated_tokens,) if do_expand else (num_allocated_tokens, num_topk)
                recv_topk_weights = torch.empty(weight_shape, dtype=topk_weights.dtype, device=topk_weights.device)
            recv_src_metadata = torch.empty(
                (num_recv_tokens, num_topk + 2), dtype=torch.int32, device=x_tensor.device
            )
            recv_sf_token_stride = 0
            recv_sf_hidden_stride = 0
            if sf is not None:
                recv_sf = torch.empty((num_allocated_tokens, num_sf_packs), dtype=sf.dtype, device=sf.device)
                recv_sf_token_stride = int(recv_sf.stride(0))
                recv_sf_hidden_stride = int(recv_sf.stride(1))
            self.runtime.launch_dispatch_copy_epilogue(
                int(self._native_v2_resources["buffer_ptr"]),
                int(self._native_v2_resources["workspace_ptr"]),
                int(psum_scaleup.data_ptr()),
                int(psum_expert.data_ptr()),
                int(recv_x.data_ptr()),
                0 if recv_sf is None else int(recv_sf.data_ptr()),
                0 if recv_topk_idx is None else int(recv_topk_idx.data_ptr()),
                0 if recv_topk_weights is None else int(recv_topk_weights.data_ptr()),
                int(recv_src_metadata.data_ptr()),
                int(channel_linked_list.data_ptr()),
                int(num_recv_tokens),
                int(num_max_tokens_per_rank),
                int(num_channels),
                int(num_sf_packs),
                int(recv_sf_token_stride),
                int(recv_sf_hidden_stride),
                bool(do_expand),
                False,
                int(smem_bytes),
                str(Path(__file__).resolve().parents[3] / "include"),
                _cuda_stream_ptr(torch.cuda.current_stream()),
            )
            torch.cuda.current_stream().synchronize()
        finally:
            # Proxies are persistent (started in init_native_v2_efa_transport);
            # nothing to stop per-dispatch.
            pass
        transport = V2TransportHandle(
            dispatch_segments=torch.empty((0,), dtype=torch.uint8, device=x_tensor.device),
            dispatch_batches=torch.empty((0,), dtype=torch.uint8, device=x_tensor.device),
            dispatch_route_offsets=torch.empty((0,), dtype=torch.int32, device=x_tensor.device),
            dispatch_counters=torch.empty((0,), dtype=torch.int32, device=x_tensor.device),
            combine_segments=None,
            combine_batches=None,
            combine_counters=None,
            d2h_queue=None,
            combine_d2h_queue=None,
            dispatch_layout=layout,
            combine_layout=None,
            num_dispatch_batches=0,
            num_dispatch_segments=0,
            num_combine_batches=0,
            num_combine_segments=0,
            payload_bytes=hidden_bytes,
            scale_bytes=scale_bytes,
            dispatch_drain_stats=None,
            timings=None,
        )
        expert_counts = [0 for _ in range(num_local_experts)]
        handle = EPHandle(
            do_expand=do_expand,
            num_experts=self.num_experts,
            expert_alignment=expert_alignment,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_sms=num_sms,
            topk_idx=copied_topk_idx,
            num_recv_tokens_per_expert_list=expert_counts,
            psum_num_recv_tokens_per_scaleup_rank=psum_scaleup,
            psum_num_recv_tokens_per_expert=psum_expert,
            recv_src_metadata=recv_src_metadata,
            dst_buffer_slot_idx=dst_buffer_slot_idx,
            token_metadata_at_forward=token_metadata_at_forward,
            channel_linked_list=channel_linked_list,
            transport_handle=transport,
        )
        recv_x_out = (recv_x, recv_sf) if sf is not None else recv_x
        return recv_x_out, recv_topk_idx, recv_topk_weights, handle, EventOverlap(None)

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Optional[object] = None,
        previous_event_before_epilogue: Optional[object] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        del num_sms, num_qps, previous_event, previous_event_before_epilogue
        del async_with_compute_stream, allocate_on_comm_stream
        _require_cuda_contiguous(x, "x")
        if topk_weights is not None:
            _require_cuda_contiguous(topk_weights, "topk_weights")
        del bias
        raise NotImplementedError(
            "native V2 combine is intentionally not wired through a semantic "
            "all-to-all path; finish the V2 combine descriptor/data path first"
        )
