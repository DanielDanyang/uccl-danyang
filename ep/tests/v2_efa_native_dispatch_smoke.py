"""Correctness + rough-throughput test for the native V2 EFA dispatch path.

Creates a real DeepEP V2 ``ElasticBuffer`` (NCCL symmetric window + dev_comm);
the native EFA signal scratch is carved from the GPU buffer device tail (only the
device segment is registered as the EFA MR), then
drives dispatch through the native UCCL EFA wrapper (loaded under a synthetic
package name to avoid the ``deep_ep`` name collision).  Designed for a 2-node
EP16 run (8 GPUs/node).  By default each token is routed to the same local rank
on the other node so the EFA scaleout path is exercised.  ``--route-mode
spread-remote`` is a diagnostic mode that spreads tokens across all remote local
ranks to reduce repeated tail writes to the same EFA/SRD destination word.
"""
import argparse
import importlib.util
import inspect
import os
import sys
import time

import torch
import torch.distributed as dist


def _repo_root() -> str:
    return os.environ.get(
        "DEEPEP_REPO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )


def _load_pkg(name: str, path: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"),
        submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _hide_repo_root_for_installed_uccl(repo: str) -> None:
    repo_real = os.path.realpath(repo)
    cwd_real = os.path.realpath(os.getcwd())
    kept = []
    for entry in sys.path:
        if entry == "":
            entry_real = cwd_real
        else:
            entry_real = os.path.realpath(entry)
        if entry_real == repo_real:
            continue
        kept.append(entry)
    sys.path[:] = kept


def _trace(rank: int, msg: str) -> None:
    if os.environ.get("UCCL_V2_SMOKE_TRACE", "0") == "1":
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[rank {rank} local {local_rank}] {msg}", flush=True)


def _init_process_group(local_rank: int, num_local_ranks: int):
    torch.cuda.set_device(local_rank)
    if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
        dist.init_process_group("nccl")
    else:
        master = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = int(os.environ.get("MASTER_PORT", "8361"))
        num_nodes = int(os.environ.get("WORLD_SIZE", "1"))
        node_rank = int(os.environ.get("RANK", "0"))
        kwargs = dict(
            backend="nccl",
            init_method=f"tcp://{master}:{port}",
            world_size=num_nodes * num_local_ranks,
            rank=node_rank * num_local_ranks + local_rank,
        )
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**kwargs)
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(num_local_ranks)
    repo = _repo_root()
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")
    torch.cuda.set_device(local_rank)
    rank, world, group = _init_process_group(local_rank, num_local_ranks)
    local_world = num_local_ranks
    T, H = args.tokens, args.hidden
    _trace(rank, f"process group ready world={world}")

    # Real DeepEP V2 buffer.  The native EFA signal scratch is carved from the
    # GPU buffer tail, so the test either uses an explicit --window-mb or asks
    # upstream DeepEP for the exact V2 layout size and adds scratch headroom.
    # (num_cpu_bytes is the engram segment and is unused here.)
    import deep_ep as rdep
    _trace(rank, "constructing DeepEP ElasticBuffer")
    if args.window_mb > 0:
        num_buffer_bytes = args.window_mb << 20
    else:
        num_buffer_bytes = rdep.ElasticBuffer.get_buffer_size_hint(
            group,
            num_max_tokens_per_rank=T,
            hidden=H,
            num_topk=1,
            use_fp8_dispatch=False,
            allow_hybrid_mode=True,
            allow_multiple_reduction=True,
        ) + (args.extra_window_mb << 20)
    real_buf = rdep.ElasticBuffer(
        group,
        num_bytes=num_buffer_bytes,
        num_cpu_bytes=args.cpu_mb << 20,
        num_max_tokens_per_rank=T, hidden=H, num_topk=1,
        allow_hybrid_mode=True,
        explicitly_destroy=True,
    )
    _trace(rank, "DeepEP ElasticBuffer ready")

    # Native EFA wrapper (loaded under a synthetic name to dodge the collision).
    _trace(rank, "loading native EFA wrapper")
    _hide_repo_root_for_installed_uccl(repo)
    wdep = _load_pkg("uccl_deep_ep_efa",
                     os.path.join(repo, "ep", "deep_ep_v2_wrapper", "deep_ep"))
    _trace(rank, "initializing DeepEP JIT")
    wdep.init_deep_ep_jit(cuda_home_path=cuda_home, nccl_root_path=wdep.find_nccl_root())
    _trace(rank, "constructing wrapper ElasticBuffer")
    wrap = wdep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=T, hidden=H, num_topk=1,
    )
    _trace(rank, "initializing native V2 EFA transport")
    wrap.init_from_deep_ep_v2(real_buf, num_lanes=args.lanes)
    _trace(rank, "native V2 EFA transport ready")

    experts = world
    node_rank = rank // local_world
    local_idx = rank % local_world
    remote_base = (1 - node_rank) * local_world
    src_rank = (rank + local_world) % world  # who routes to us in paired mode
    dst_rank = (rank + local_world) % world  # where our tokens go in paired mode
    x = (torch.arange(T * H, dtype=torch.float32, device="cuda").reshape(T, H)
         + rank * 10000).to(torch.bfloat16)
    if args.route_mode == "paired-remote":
        topk_idx = torch.full((T, 1), dst_rank, dtype=torch.int64, device="cuda")
    elif args.route_mode == "spread-remote":
        token_ids = torch.arange(T, dtype=torch.int64, device="cuda")
        topk_idx = (remote_base + ((token_ids + local_idx) % local_world)).reshape(T, 1)
    else:
        raise ValueError(f"unknown route mode {args.route_mode}")
    topk_weights = torch.full((T, 1), rank + 0.5, dtype=torch.float32, device="cuda")

    def one_dispatch(do_expand=False):
        return wrap.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_experts=experts, num_max_tokens_per_rank=T,
            num_sms=args.sms, do_cpu_sync=False, do_expand=do_expand,
        )

    # Warmup (triggers JIT compile)
    for _ in range(2):
        _trace(rank, "entering dispatch warmup")
        dist.barrier()
        recv_x, recv_idx, recv_w, handle, _ = one_dispatch()
        torch.cuda.synchronize()
        _trace(rank, "dispatch warmup complete")

    if os.environ.get("UCCL_V2_DISPATCH_LAUNCH_ONLY", "0") == "1":
        if rank == 0:
            stage = os.environ.get("UCCL_V2_DISPATCH_DEBUG_STAGE", "full")
            print(f"native_dispatch_launch_only PASS stage={stage}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Correctness: rank R receives src_rank's T tokens; channel arrival order is
    # not part of the dispatch contract, so compare after sorting metadata.
    if args.route_mode == "paired-remote":
        expected_x = (torch.arange(T * H, dtype=torch.float32, device="cuda")
                      .reshape(T, H) + src_rank * 10000).to(torch.bfloat16)
        expected_w = torch.full((T, 1), src_rank + 0.5, dtype=torch.float32, device="cuda")
        expected_src = (torch.arange(T, dtype=torch.int32, device="cuda") + src_rank * T)
    else:
        expected_x_parts = []
        expected_w_parts = []
        expected_src_parts = []
        token_ids = torch.arange(T, dtype=torch.int64, device="cuda")
        for src in range(remote_base, remote_base + local_world):
            src_local = src % local_world
            mask = ((token_ids + src_local) % local_world) == local_idx
            src_tokens = token_ids[mask]
            expected_x_parts.append(
                (torch.arange(T * H, dtype=torch.float32, device="cuda")
                 .reshape(T, H)[src_tokens] + src * 10000).to(torch.bfloat16))
            expected_w_parts.append(
                torch.full((src_tokens.numel(), 1), src + 0.5,
                           dtype=torch.float32, device="cuda"))
            expected_src_parts.append((src_tokens.to(torch.int32) + src * T))
        expected_x = torch.cat(expected_x_parts, dim=0)
        expected_w = torch.cat(expected_w_parts, dim=0)
        expected_src = torch.cat(expected_src_parts, dim=0)

    ok = True
    msgs = []
    expected_tokens = expected_src.numel()
    psum_scaleup_cpu = handle.psum_num_recv_tokens_per_scaleup_rank.cpu().tolist()
    psum_expert_cpu = handle.psum_num_recv_tokens_per_expert.cpu().tolist()
    reported_tokens = int(psum_scaleup_cpu[-1]) if psum_scaleup_cpu else -1
    if reported_tokens != expected_tokens:
        ok = False
        msgs.append(
            f"psum token count mismatch reported={reported_tokens} "
            f"expected={expected_tokens} psum_scaleup={psum_scaleup_cpu} "
            f"psum_expert_head={psum_expert_cpu[:8]}")
    src_md = handle.recv_src_metadata[:expected_tokens, 0]
    sorted_src, order = torch.sort(src_md)
    sorted_x = recv_x[:expected_tokens][order]
    if not torch.equal(sorted_x.cpu(), expected_x.cpu()):
        ok = False; msgs.append("recv_x mismatch")
    if recv_w is not None:
        sorted_w = recv_w[:expected_tokens].reshape(expected_tokens, -1)[order, 0]
        if not torch.allclose(sorted_w.cpu(), expected_w[:, 0].cpu()):
            ok = False; msgs.append("recv_w mismatch")
    if not torch.equal(sorted_src.cpu(), expected_src.cpu()):
        ok = False
        sorted_src_cpu = sorted_src.cpu()
        expected_src_cpu = expected_src.cpu()
        mismatch = (sorted_src_cpu != expected_src_cpu).nonzero(as_tuple=False).flatten()
        first_bad = int(mismatch[0].item()) if mismatch.numel() else -1
        lo = max(first_bad - 4, 0)
        hi = min(first_bad + 8, expected_tokens)
        unique_count = int(torch.unique(sorted_src).numel())
        msgs.append(
            "recv_src mismatch "
            f"raw[:4]={src_md[:4].cpu().tolist()} "
            f"sorted[:8]={sorted_src_cpu[:8].tolist()} "
            f"sorted[-8:]={sorted_src_cpu[-8:].tolist()} "
            f"exp[:8]={expected_src_cpu[:8].tolist()} "
            f"exp[-8:]={expected_src_cpu[-8:].tolist()} "
            f"first_bad={first_bad} "
            f"got_window={sorted_src_cpu[lo:hi].tolist()} "
            f"exp_window={expected_src_cpu[lo:hi].tolist()} "
            f"unique={unique_count}/{expected_tokens} "
            f"minmax=({int(sorted_src_cpu[0].item())}, {int(sorted_src_cpu[-1].item())})")

    flag = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"native_dispatch_correctness {'PASS' if int(flag.item())==1 else 'FAIL'} "
              f"world={world} tokens={T} hidden={H} sms={args.sms} lanes={args.lanes} "
              f"route_mode={args.route_mode}",
              flush=True)
    if not ok:
        print(f"[rank {rank}] FAIL: {'; '.join(msgs)}", flush=True)

    if args.perf and int(flag.item()) == 1:
        total_dt = 0.0
        dispatch_ms_total = 0.0
        epilogue_ms_total = 0.0
        gpu_total_ms_total = 0.0
        timing_count = 0
        for _ in range(args.iters):
            dist.barrier()
            t0 = time.perf_counter()
            recv_x, recv_idx, recv_w, handle, _ = one_dispatch()
            torch.cuda.synchronize()
            total_dt += time.perf_counter() - t0
            timings = getattr(getattr(handle, "transport_handle", None), "timings", None)
            if timings:
                dispatch_ms_total += float(timings.get("dispatch_ms", 0.0))
                epilogue_ms_total += float(timings.get("copy_epilogue_ms", 0.0))
                gpu_total_ms_total += float(timings.get("gpu_total_ms", 0.0))
                timing_count += 1
        dt = total_dt / args.iters
        bw = (T * H * 2) / dt / 1e9
        if rank == 0:
            print(f"native_dispatch_perf per_iter={dt*1e3:.3f}ms approx_per_rank_BW={bw:.2f}GB/s",
                  flush=True)
            if timing_count > 0:
                print(
                    "native_dispatch_split "
                    f"dispatch={dispatch_ms_total / timing_count:.3f}ms "
                    f"copy_epilogue={epilogue_ms_total / timing_count:.3f}ms "
                    f"gpu_total={gpu_total_ms_total / timing_count:.3f}ms",
                    flush=True,
                )

    _trace(rank, "entering final barrier")
    dist.barrier()
    _trace(rank, "destroying native wrapper")
    wrap.destroy()
    _trace(rank, "destroying DeepEP ElasticBuffer")
    real_buf.destroy()
    _trace(rank, "destroying process group")
    dist.destroy_process_group()
    _trace(rank, "done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--sms", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--window-mb", type=int, default=0,
                        help="explicit DeepEP GPU buffer MB; 0 uses upstream size hint")
    parser.add_argument("--extra-window-mb", type=int, default=16,
                        help="extra GPU buffer headroom for native EFA signal scratch")
    parser.add_argument("--cpu-mb", type=int, default=0)  # Scratch lives in the GPU buffer tail.
    parser.add_argument("--route-mode", choices=("paired-remote", "spread-remote"),
                        default="paired-remote")
    parser.add_argument("--perf", action="store_true", help="also run a throughput loop")
    parsed = parser.parse_args()

    if os.environ.get("LOCAL_RANK") is not None and os.environ.get("LOCAL_WORLD_SIZE") is not None:
        run(int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"]), parsed)
    else:
        torch.multiprocessing.spawn(
            run, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
