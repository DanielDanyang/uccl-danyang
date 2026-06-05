// ===========================================================================
// UCCL-GIN vs NCCL-GIN standalone microbench (Rail / inter-node path)
// ===========================================================================
//
// Goal: exercise the few UCCL-GIN Rail ops (put + red_add_rel) in isolation,
// WITHOUT DeepEP V2, and compare against the same workload run over native
// NCCL-GIN. Both paths live in one MPI program for apples-to-apples timing.
//
// Workload (paired-remote, mirrors AGENTS.md gin_proxy_bench "same-local-remote"):
//   - 2 nodes x local_world ranks (EP16 default: 8/node).
//   - Each rank r streams `iters` messages of `msg_bytes` to its remote pair
//     peer = (r + local_world) % world (same local rank on the other node),
//     then issues a red_add to a per-(peer) counter.
//   - Per-rank achievable BW = iters*msg_bytes / elapsed; correctness checks the
//     receiver's recv buffer + the red_add counter.
//
// Two paths:
//   [NCCL-GIN]  ncclMemAlloc + window + ncclDevComm(GIN) + ncclGin.put(SignalInc)
//               (reference, modeled on nccl/docs/examples/06_device_api/02_alltoall_gin)
//   [UCCL-GIN]  UcclProxy (EFA) + D2H ring + handle::UCCLGin (gin.put<Rail> / red_add_rel<Rail>)
//
// !!! STATUS: first version, written without a local GPU/CUDA toolchain.
//     Expect to compile-iterate on p5en. The API calls follow the verified
//     headers (uccl_proxy.hpp, d2h_queue_device.cuh, ring_buffer.cuh, nccl_device.h),
//     but signatures/semantics should be double-checked on the server.
//     See README.md.
//
// Build: see Makefile (needs mpicxx/nvcc, NCCL device runtime, EFA verbs, and the
//        UCCL ep objects). Launch with mpirun across the 2 nodes.
// ===========================================================================

#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ring_buffer.cuh"        // TransferCmd, DeviceToHostCmdBuffer
#include "d2h_queue_device.cuh"   // d2hq::D2HHandle
#include "uccl_proxy.hpp"         // UcclProxy, PeerMeta
#include "uccl_gin/uccl_gin.cuh"  // uccl_gin::UCCLGin handle (mirrors handle::NCCLGin)

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define NCCL_OK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){ \
  fprintf(stderr,"[NCCL] %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Args {
  int   iters       = 50;
  int   warmup      = 5;
  bool  run_nccl    = true;
  bool  run_uccl    = true;
  const char* ifname = "enp71s0";       // NCCL_SOCKET_IFNAME on p5en
  std::vector<size_t> sizes = {4096, 16384, 65536, 262144, 1048576, 16777216};
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](){ return (i+1<argc) ? argv[++i] : nullptr; };
    if (s == "--iters") a.iters = atoi(next());
    else if (s == "--warmup") a.warmup = atoi(next());
    else if (s == "--ifname") a.ifname = next();
    else if (s == "--no-nccl") a.run_nccl = false;
    else if (s == "--no-uccl") a.run_uccl = false;
    else if (s == "--sizes") { a.sizes.clear(); char* t=strtok(next(),","); while(t){a.sizes.push_back(strtoull(t,0,10)); t=strtok(0,",");} }
  }
  return a;
}

// IPv4 address of `ifname` (for UCCL PeerMeta OOB exchange).
static std::string iface_ip(const char* ifname) {
  struct ifaddrs* ifa = nullptr; getifaddrs(&ifa);
  std::string out;
  for (auto* p = ifa; p; p = p->ifa_next) {
    if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
    if (strcmp(p->ifa_name, ifname) != 0) continue;
    char buf[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &((struct sockaddr_in*)p->ifa_addr)->sin_addr, buf, sizeof(buf));
    out = buf; break;
  }
  if (ifa) freeifaddrs(ifa);
  return out;
}

// ===========================================================================
// NCCL-GIN reference path
// ===========================================================================
//
// Paired put: rank r writes `count` floats into peer's recv buffer, signaling
// the peer's signal index. Mirrors the alltoall_gin example but 1 peer (the
// remote pair) instead of all ranks, so it matches the UCCL paired workload.

#define NCCL_CTAS 16
#define NCCL_TPB  512

__global__ void nccl_gin_paired_kernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                       size_t bytes, int peer, int iters,
                                       struct ncclDevComm devComm) {
  int ginCtx = 0;
  unsigned sig = blockIdx.x;
  ncclGin gin{devComm, ginCtx};
  uint64_t base = gin.readSignal(sig);
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  // One put per iter from this CTA (CTA 0 only, to keep it a single stream that
  // is comparable to one UCCL D2H lane; multi-CTA fan-out is a later sweep).
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      for (int it = 0; it < iters; ++it) {
        gin.put(ncclTeamWorld(devComm), peer,
                recvwin, /*recvoff=*/0, sendwin, /*sendoff=*/0,
                bytes, ncclGin_SignalInc{sig});
      }
    }
    __syncthreads();
    gin.waitSignal(ncclCoopCta(), sig, base + iters);  // wait inbound from our pair
    gin.flush(ncclCoopCta());
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

static double run_nccl_gin(ncclComm_t comm, struct ncclDevComm devComm,
                           ncclWindow_t sendwin, ncclWindow_t recvwin,
                           void* d_send, void* d_recv,
                           size_t bytes, int peer, int iters, int warmup,
                           cudaStream_t stream) {
  CUDA_OK(cudaMemset(d_recv, 0, bytes));
  // warmup
  nccl_gin_paired_kernel<<<NCCL_CTAS, NCCL_TPB, 0, stream>>>(sendwin, recvwin, bytes, peer, warmup, devComm);
  CUDA_OK(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  cudaEvent_t a, b; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b));
  CUDA_OK(cudaEventRecord(a, stream));
  nccl_gin_paired_kernel<<<NCCL_CTAS, NCCL_TPB, 0, stream>>>(sendwin, recvwin, bytes, peer, iters, devComm);
  CUDA_OK(cudaEventRecord(b, stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, a, b));
  CUDA_OK(cudaEventDestroy(a)); CUDA_OK(cudaEventDestroy(b));
  return (double)ms;
}

// ===========================================================================
// UCCL-GIN path
// ===========================================================================
//
// Kernel: a single warp streams `iters` gin.put<Rail>()s of `bytes` to the remote
// pair through handle::UCCLGin, then gin.red_add_rel<Rail>() to the peer counter.
// Layout of the registered window buffer:
//   [ send region (max_bytes) | recv region (max_bytes) ]  (offset 0 / max_bytes)
// red_add counter lives in the proxy atomic buffer at a fixed offset.

// Mirrors the NCCL-GIN kernel's call site, but through handle::UCCLGin:
// same gin.put<Rail>(...) / gin.red_add_rel<Rail>(...) shape, UCCL backend.
__global__ void uccl_gin_paired_kernel(uccl_gin::UCCLGinResources res, int peer,
                                       void* send_ptr, void* recv_ptr, uint32_t bytes,
                                       int iters, void* counter_ptr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uccl_gin::UCCLGin gin(res);  // the exact handle DeepEP kernels will use
  for (int it = 0; it < iters; ++it)
    gin.put<ncclTeamTagRail>(recv_ptr, send_ptr, (int)bytes, peer);
  gin.red_add_rel<ncclTeamTagRail>(counter_ptr, /*delta=*/1, peer);
}

struct UcclCtx {
  std::vector<UcclProxy*> proxies;
  void* d_window = nullptr;            // registered MR buffer (send|recv)
  size_t window_bytes = 0;
  d2hq::D2HHandle** d_handles = nullptr; // device array of D2H handle pointers
  int num_queues = 0;
  uccl_gin::UCCLGinResources res{};    // injected into UCCLGin (stable ABI)
};

// Bootstrap the UCCL EFA transport for this rank (one registered window + proxies).
static UcclCtx uccl_setup(int rank, int world, int local_world, size_t max_bytes,
                          const char* ifname) {
  UcclCtx c;
  int local_rank = rank % local_world;
  int node_idx   = rank / local_world;
  int num_nodes  = world / local_world;
  c.window_bytes = 2 * max_bytes;                 // send | recv
  CUDA_OK(cudaMalloc(&c.d_window, c.window_bytes));
  CUDA_OK(cudaMemset(c.d_window, 0, c.window_bytes));

  const int nproxy = kNumProxyThs;
  for (int t = 0; t < nproxy; ++t) {
    auto* p = new UcclProxy(/*thread_idx=*/t,
                            /*gpu_buffer_addr=*/(uintptr_t)c.d_window,
                            /*total_size=*/c.window_bytes,
                            /*rank=*/rank, /*node_idx=*/node_idx,
                            /*local_rank=*/local_rank,
                            /*num_experts=*/0, /*num_ranks=*/world,
                            /*num_nodes=*/num_nodes, /*use_normal_mode=*/true,
                            /*is_intranode=*/(num_nodes <= 1),
                            /*gpu_buffer_is_host_allocated=*/false,
                            /*owns_gpu_buffer=*/false);
    c.proxies.push_back(p);
  }

  // Build local PeerMeta and Allgather it across all ranks.
  PeerMeta me{};
  me.rank   = rank;
  me.ptr    = (uintptr_t)c.d_window;
  me.nbytes = c.window_bytes;
  std::string ip = iface_ip(ifname);
  for (int t = 0; t < nproxy; ++t) me.listen_ports[t] = c.proxies[t]->get_listen_port();
  // Flatten PeerMeta into a POD for MPI (ip as fixed-size char buffer).
  struct WirePeer { int rank; uintptr_t ptr; size_t nbytes; int ports[kNumProxyThs]; char ip[64]; };
  WirePeer mine{}; mine.rank = me.rank; mine.ptr = me.ptr; mine.nbytes = me.nbytes;
  memcpy(mine.ports, me.listen_ports, sizeof(mine.ports));
  strncpy(mine.ip, ip.c_str(), sizeof(mine.ip)-1);
  std::vector<WirePeer> all(world);
  MPI_Allgather(&mine, sizeof(WirePeer), MPI_BYTE, all.data(), sizeof(WirePeer), MPI_BYTE, MPI_COMM_WORLD);

  std::vector<PeerMeta> peers(world);
  for (int r = 0; r < world; ++r) {
    peers[r].rank = all[r].rank; peers[r].ptr = all[r].ptr; peers[r].nbytes = all[r].nbytes;
    peers[r].ip = all[r].ip;
    memcpy(peers[r].listen_ports, all[r].ports, sizeof(peers[r].listen_ports));
  }
  for (auto* p : c.proxies) p->set_peers_meta(peers);

  MPI_Barrier(MPI_COMM_WORLD);
  for (auto* p : c.proxies) p->start_dual();
  // Give proxies time to connect (mirrors the Python smoke's time.sleep(1)).
  MPI_Barrier(MPI_COMM_WORLD);

  // Collect device D2H ring handles into a device array.
  std::vector<uint64_t> dev_ring_addrs;
  for (auto* p : c.proxies) {
    for (auto a : p->get_d2h_channel_handle_addrs()) dev_ring_addrs.push_back(a);
  }
  c.num_queues = (int)dev_ring_addrs.size();
  std::vector<d2hq::D2HHandle*> h_handles(c.num_queues);
  for (int i = 0; i < c.num_queues; ++i)
    h_handles[i] = reinterpret_cast<d2hq::D2HHandle*>(dev_ring_addrs[i]);
  CUDA_OK(cudaMalloc(&c.d_handles, c.num_queues * sizeof(d2hq::D2HHandle*)));
  CUDA_OK(cudaMemcpy(c.d_handles, h_handles.data(), c.num_queues * sizeof(d2hq::D2HHandle*), cudaMemcpyHostToDevice));

  // All proxies share proxy[0]'s atomic buffer (mirrors the deleted Python wrapper),
  // so red_add counter offsets have a single consistent origin.
  uintptr_t atomic_base = c.proxies[0]->get_atomic_buffer_addr();
  for (auto* p : c.proxies) p->set_atomic_buffer_addr(atomic_base);

  // Bundle the stable resource view injected into UCCLGin.
  c.res.d2h_queues      = c.d_handles;
  c.res.num_queues      = (uint32_t)c.num_queues;
  c.res.window_base     = (uint64_t)c.d_window;
  c.res.atomic_tail_base = (uint64_t)atomic_base;
  c.res.num_scaleout_ranks = num_nodes;
  c.res.num_scaleup_ranks  = local_world;
  c.res.scaleout_rank   = node_idx;
  c.res.scaleup_rank    = local_rank;
  c.res.num_lanes       = (uint32_t)nproxy;
  return c;
}

static void uccl_teardown(UcclCtx& c) {
  for (auto* p : c.proxies) { p->stop(); delete p; }
  c.proxies.clear();
  if (c.d_handles) cudaFree(c.d_handles);
  if (c.d_window)  cudaFree(c.d_window);
}

static double run_uccl_gin(UcclCtx& c, size_t bytes, int peer, int iters, int warmup,
                           cudaStream_t stream, size_t max_bytes) {
  // Symmetric pointers into the registered window: send region @0, recv region
  // @max_bytes; counter @ atomic buffer base. The handle converts to offsets.
  void* send_ptr = c.d_window;
  void* recv_ptr = (char*)c.d_window + max_bytes;
  void* counter_ptr = (void*)c.res.atomic_tail_base;
  auto* local_counter = reinterpret_cast<std::atomic<int64_t>*>(counter_ptr);
  auto wait_counter = [&](int64_t target) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    int spins = 0;
    while (local_counter->load(std::memory_order_acquire) < target) {
      if ((++spins & 0xFFFF) == 0) std::this_thread::yield();
      if (clock::now() - start > std::chrono::seconds(30)) {
        fprintf(stderr,
                "[UCCL-GIN] timeout waiting local counter >= %ld (now=%ld)\n",
                (long)target,
                (long)local_counter->load(std::memory_order_relaxed));
        std::abort();
      }
    }
  };

  // warmup
  local_counter->store(0, std::memory_order_release);
  MPI_Barrier(MPI_COMM_WORLD);
  uccl_gin_paired_kernel<<<1, 32, 0, stream>>>(c.res, peer, send_ptr, recv_ptr,
                                               (uint32_t)bytes, warmup, counter_ptr);
  CUDA_OK(cudaStreamSynchronize(stream));
  wait_counter(1);
  MPI_Barrier(MPI_COMM_WORLD);

  local_counter->store(0, std::memory_order_release);
  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::steady_clock::now();
  uccl_gin_paired_kernel<<<1, 32, 0, stream>>>(c.res, peer, send_ptr, recv_ptr,
                                               (uint32_t)bytes, iters, counter_ptr);
  CUDA_OK(cudaStreamSynchronize(stream));
  wait_counter(1);
  auto t1 = std::chrono::steady_clock::now();
  MPI_Barrier(MPI_COMM_WORLD);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  Args args = parse_args(argc, argv);

  // local_world: ranks per node (from env, else assume 8). MPI launch must place
  // contiguous ranks per node (mpirun -npernode 8).
  int local_world = atoi(getenv("LOCAL_WORLD_SIZE") ? getenv("LOCAL_WORLD_SIZE") : "8");
  int local_rank  = rank % local_world;
  int peer        = (rank + local_world) % world;   // remote pair
  CUDA_OK(cudaSetDevice(local_rank));
  cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));
  size_t max_bytes = 0; for (auto s : args.sizes) max_bytes = std::max(max_bytes, s);

  if (rank == 0) {
    printf("uccl_gin_microbench world=%d local_world=%d iters=%d (paired-remote)\n",
           world, local_world, args.iters);
    printf("%-12s %-18s %-18s\n", "bytes", "NCCL-GIN GB/s", "UCCL-GIN GB/s");
  }

  // -------- NCCL-GIN setup (once) --------
  ncclComm_t comm = nullptr; struct ncclDevComm devComm{};
  ncclWindow_t sendwin = nullptr, recvwin = nullptr; void* d_send = nullptr; void* d_recv = nullptr;
  if (args.run_nccl) {
    ncclUniqueId id;
    if (rank == 0) NCCL_OK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_OK(ncclCommInitRank(&comm, world, id, rank));
    NCCL_OK(ncclMemAlloc(&d_send, max_bytes));
    NCCL_OK(ncclMemAlloc(&d_recv, max_bytes));
    NCCL_OK(ncclCommWindowRegister(comm, d_send, max_bytes, &sendwin, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_OK(ncclCommWindowRegister(comm, d_recv, max_bytes, &recvwin, NCCL_WIN_COLL_SYMMETRIC));
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.worldGinBarrierCount = NCCL_CTAS;
    reqs.ginSignalCount = NCCL_CTAS;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCL_OK(ncclDevCommCreate(comm, &reqs, &devComm));
  }

  // -------- UCCL-GIN setup (once) --------
  UcclCtx uctx;
  if (args.run_uccl) uctx = uccl_setup(rank, world, local_world, max_bytes, args.ifname);

  // -------- sweep --------
  for (size_t bytes : args.sizes) {
    double nccl_gbps = 0, uccl_gbps = 0;
    if (args.run_nccl) {
      double ms = run_nccl_gin(comm, devComm, sendwin, recvwin, d_send, d_recv,
                               bytes, peer, args.iters, args.warmup, stream);
      double per_rank = (double)bytes * args.iters / (ms * 1e-3) / 1e9;
      MPI_Reduce(rank==0?MPI_IN_PLACE:&per_rank, &per_rank, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      nccl_gbps = per_rank / world;  // avg per-rank
    }
    if (args.run_uccl) {
      double ms = run_uccl_gin(uctx, bytes, peer, args.iters, args.warmup, stream, max_bytes);
      double per_rank = (double)bytes * args.iters / (ms * 1e-3) / 1e9;
      MPI_Reduce(rank==0?MPI_IN_PLACE:&per_rank, &per_rank, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      uccl_gbps = per_rank / world;
    }
    if (rank == 0)
      printf("%-12zu %-18.2f %-18.2f\n", bytes, nccl_gbps, uccl_gbps);
  }

  // TODO(correctness): after the largest size, copy back d_recv (NCCL) and the
  // UCCL recv region + atomic counter, verify against the rank-tagged send data
  // and expected red_add count. (Left as a server-side step; see README.)

  if (args.run_uccl) uccl_teardown(uctx);
  if (args.run_nccl) {
    NCCL_OK(ncclDevCommDestroy(comm, &devComm));
    NCCL_OK(ncclCommWindowDeregister(comm, sendwin));
    NCCL_OK(ncclCommWindowDeregister(comm, recvwin));
    ncclMemFree(d_send); ncclMemFree(d_recv);
    NCCL_OK(ncclCommFinalize(comm)); NCCL_OK(ncclCommDestroy(comm));
  }
  CUDA_OK(cudaStreamDestroy(stream));
  MPI_Finalize();
  return 0;
}
