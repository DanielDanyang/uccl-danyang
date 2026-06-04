"""Correctness + rough-throughput test for the native V2 EFA dispatch path.

Creates a real DeepEP V2 ``ElasticBuffer`` (NCCL symmetric window + dev_comm,
with a CPU/engram segment reserved for the native EFA signal scratch), then
drives dispatch through the native UCCL EFA wrapper (loaded under a synthetic
package name to avoid the ``deep_ep`` name collision).  Designed for a 2-node
EP16 run (8 GPUs/node); each token is routed to the same local rank on the other
node so the EFA scaleout path is exercised, with one remaining token kept local
to also cover the local-rank notify/tail path.
"""
import argparse
import importlib.util
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--sms", type=int, default=8)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--window-mb", type=int, default=512)
    p.add_argument("--cpu-mb", type=int, default=8)  # CPU/engram segment for signal scratch
    p.add_argument("--perf", action="store_true", help="also run a throughput loop")
    args = p.parse_args()

    repo = _repo_root()
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    T, H = args.tokens, args.hidden

    # Real DeepEP V2 buffer.  Reserve a CPU/engram segment for the EFA signal
    # scratch (engram is unused here, so this is otherwise-idle space).
    import deep_ep as rdep
    real_buf = rdep.ElasticBuffer(
        dist.group.WORLD,
        num_bytes=args.window_mb << 20,
        num_cpu_bytes=args.cpu_mb << 20,
        num_max_tokens_per_rank=T, hidden=H, num_topk=1,
        allow_hybrid_mode=True,
    )

    # Native EFA wrapper (loaded under a synthetic name to dodge the collision).
    wdep = _load_pkg("uccl_deep_ep_efa",
                     os.path.join(repo, "ep", "deep_ep_v2_wrapper", "deep_ep"))
    wdep.init_deep_ep_jit(cuda_home_path=cuda_home, nccl_root_path=wdep.find_nccl_root())
    wrap = wdep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=T, hidden=H, num_topk=1,
    )
    wrap.init_from_deep_ep_v2(real_buf, num_lanes=args.lanes)

    experts = world
    src_rank = (rank + local_world) % world  # who routes to us (remote pair)
    dst_rank = (rank + local_world) % world  # where our tokens go
    x = (torch.arange(T * H, dtype=torch.float32, device="cuda").reshape(T, H)
         + rank * 10000).to(torch.bfloat16)
    topk_idx = torch.full((T, 1), dst_rank, dtype=torch.int64, device="cuda")
    topk_weights = torch.full((T, 1), rank + 0.5, dtype=torch.float32, device="cuda")

    def one_dispatch(do_expand=False):
        return wrap.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_experts=experts, num_max_tokens_per_rank=T,
            num_sms=args.sms, do_cpu_sync=False, do_expand=do_expand,
        )

    # Warmup (triggers JIT compile)
    for _ in range(2):
        dist.barrier()
        recv_x, recv_idx, recv_w, handle, _ = one_dispatch()
        torch.cuda.synchronize()

    # Correctness: rank R receives src_rank's T tokens in order.
    expected_x = (torch.arange(T * H, dtype=torch.float32, device="cuda")
                  .reshape(T, H) + src_rank * 10000).to(torch.bfloat16)
    expected_w = torch.full((T, 1), src_rank + 0.5, dtype=torch.float32, device="cuda")
    expected_src = (torch.arange(T, dtype=torch.int32, device="cuda") + src_rank * T)

    ok = True
    msgs = []
    if not torch.equal(recv_x[:T].cpu(), expected_x.cpu()):
        ok = False; msgs.append("recv_x mismatch")
    if recv_w is not None and not torch.allclose(recv_w[:T].reshape(T, -1)[:, 0].cpu(), expected_w[:, 0].cpu()):
        ok = False; msgs.append("recv_w mismatch")
    src_md = handle.recv_src_metadata[:T, 0]
    if not torch.equal(src_md.cpu(), expected_src.cpu()):
        ok = False; msgs.append(f"recv_src mismatch got[:4]={src_md[:4].cpu().tolist()} exp[:4]={expected_src[:4].cpu().tolist()}")

    flag = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"native_dispatch_correctness {'PASS' if int(flag.item())==1 else 'FAIL'} "
              f"world={world} tokens={T} hidden={H} sms={args.sms} lanes={args.lanes}",
              flush=True)
    if not ok:
        print(f"[rank {rank}] FAIL: {'; '.join(msgs)}", flush=True)

    if args.perf and int(flag.item()) == 1:
        total_dt = 0.0
        for _ in range(args.iters):
            dist.barrier()
            t0 = time.perf_counter()
            recv_x, recv_idx, recv_w, handle, _ = one_dispatch()
            torch.cuda.synchronize()
            total_dt += time.perf_counter() - t0
        dt = total_dt / args.iters
        bw = (T * H * 2) / dt / 1e9
        if rank == 0:
            print(f"native_dispatch_perf per_iter={dt*1e3:.3f}ms approx_per_rank_BW={bw:.2f}GB/s",
                  flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
