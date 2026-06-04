#include "v2_efa/runtime.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <deep_ep/common/compiled.cuh>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#include "../../thirdparty/DeepEP-v2-d4f41e4/csrc/jit/compiler.hpp"
#include "../../thirdparty/DeepEP-v2-d4f41e4/csrc/jit/handle.hpp"
#include "../../thirdparty/DeepEP-v2-d4f41e4/csrc/jit/include_parser.hpp"
#include "../../thirdparty/DeepEP-v2-d4f41e4/csrc/jit/kernel_runtime.hpp"
#include "ring_buffer.cuh"

namespace uccl::v2_efa {

namespace {

std::once_flag g_jit_init_once;
bool g_jit_initialized = false;

}  // namespace

void init_deep_ep_jit_bridge(const std::string& library_root_path,
                             const std::string& cuda_home_path,
                             const std::string& nccl_root_path) {
  std::call_once(g_jit_init_once, [&] {
    deep_ep::jit::Compiler::prepare_init(library_root_path, cuda_home_path,
                                         nccl_root_path);
    deep_ep::jit::KernelRuntime::prepare_init(cuda_home_path);
    deep_ep::jit::IncludeParser::prepare_init(library_root_path);
    g_jit_initialized = true;
  });
}

bool is_deep_ep_jit_bridge_initialized() { return g_jit_initialized; }

std::shared_ptr<deep_ep::jit::KernelRuntime> build_v2_efa_jit_runtime(
    const V2EfaJitLaunchPlan& plan) {
  if (!is_deep_ep_jit_bridge_initialized()) {
    throw std::runtime_error(
        "DeepEP JIT bridge is not initialized; call init_deep_ep_jit first");
  }
  if (plan.name.empty() || plan.source.empty()) {
    throw std::invalid_argument("empty V2 EFA JIT plan");
  }
  const auto runtime = deep_ep::jit::compiler->build(plan.name, plan.source);
  if (runtime == nullptr) {
    throw std::runtime_error("DeepEP JIT compiler returned null runtime");
  }
  return runtime;
}

deep_ep::jit::LaunchConfigHandle make_launch_config(
    const V2EfaJitLaunchPlan& plan,
    const deep_ep::jit::KernelHandle& kernel,
    std::uintptr_t cuda_stream_ptr) {
  if (plan.grid_dim_x <= 0 || plan.grid_dim_y <= 0 ||
      plan.num_threads <= 0) {
    throw std::invalid_argument("invalid V2 EFA JIT launch dimensions");
  }

  auto stream = cuda_stream_ptr == 0
                    ? at::cuda::getCurrentCUDAStream().stream()
                    : reinterpret_cast<cudaStream_t>(cuda_stream_ptr);
  const dim3 grid_dim{static_cast<unsigned>(plan.grid_dim_x),
                      static_cast<unsigned>(plan.grid_dim_y), 1};
  const dim3 block_dim{static_cast<unsigned>(plan.num_threads), 1, 1};
  return deep_ep::jit::construct_launch_config(
      kernel, stream, plan.smem_bytes, grid_dim, block_dim, plan.cluster_dim,
      plan.cooperative, plan.pdl_enabled);
}

template <typename T>
T* checked_ptr(std::uintptr_t ptr, const char* name) {
  if (ptr == 0) {
    throw std::invalid_argument(std::string(name) + " pointer is null");
  }
  return reinterpret_cast<T*>(ptr);
}

template <typename Result>
void check_jit_launch_result(Result result) {
  using deep_ep::lazy_cuGetErrorName;
  using deep_ep::lazy_cuGetErrorString;
  EP_CUDA_UNIFIED_CHECK(result);
}

bool jit_debug_enabled() {
  const char* value = std::getenv("EP_JIT_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void compile_v2_efa_jit_plan(const V2EfaJitLaunchPlan& plan) {
  (void)build_v2_efa_jit_runtime(plan);
}

// ---------------------------------------------------------------------------
// Native hybrid dispatch kernel launch
// Kernel signature (hybrid_dispatch_native.cuh) expects these args in order:
//   standard DeepEP V2 args (x, sf, topk_idx, ..., nccl_dev_comm, nccl_window,
//                             buffer, workspace, mapped_host_workspace,
//                             scaleout_rank, scaleup_rank)
//   EFA D2H args: uint64_t* d2h_channel_addrs (GPU array of d2hq::D2HHandle*
//                 values, matching the original UCCL EP transport ABI),
//                 num_queues, signal_scratch_base.
//   The kernel derives every transport offset relative to the mapped workspace
//   pointer (= registered NCCL window base), so no separate base args are passed.
// ---------------------------------------------------------------------------
void launch_v2_efa_native_hybrid_dispatch_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t x_ptr,
    std::uintptr_t sf_ptr, std::uintptr_t topk_idx_ptr,
    std::uintptr_t topk_weights_ptr, std::uintptr_t copied_topk_idx_ptr,
    std::uintptr_t cumulative_local_expert_recv_stats_ptr,
    std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
    std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
    std::uintptr_t dst_buffer_slot_idx_ptr,
    std::uintptr_t token_metadata_at_forward_ptr, int num_tokens,
    int sf_token_stride, int sf_hidden_stride,
    std::uintptr_t nccl_dev_comm_ptr, std::uintptr_t nccl_window_ptr,
    std::uintptr_t buffer_ptr, std::uintptr_t workspace_ptr,
    std::uintptr_t mapped_host_workspace_ptr, int scaleout_rank,
    int scaleup_rank,
    // EFA D2H: GPU pointer to array of d2hq::D2HHandle* values
    std::uintptr_t d2h_queues_ptr, uint32_t num_queues,
    std::uintptr_t signal_scratch_base, std::uintptr_t atomic_tail_base,
    std::uintptr_t cuda_stream_ptr) {
  if (num_tokens < 0 || num_queues == 0 || d2h_queues_ptr == 0 ||
      signal_scratch_base == 0 || nccl_dev_comm_ptr == 0 ||
      nccl_window_ptr == 0 || mapped_host_workspace_ptr == 0 ||
      atomic_tail_base == 0) {
    throw std::invalid_argument("invalid V2 EFA native hybrid dispatch launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  if (jit_debug_enabled()) {
    std::fprintf(stderr,
                 "V2 EFA dispatch args: x=%p sf=%p topk_idx=%p weights=%p "
                 "copied=%p psum_scaleup=%p psum_expert=%p dst=%p meta=%p "
                 "dev_comm_ptr=%p window=%p buffer=%p workspace=%p "
                 "host_workspace=%p queues=%p nqueues=%u scratch=0x%lx "
                 "atomic_tail=0x%lx stream=%p\n",
                 reinterpret_cast<void*>(x_ptr),
                 reinterpret_cast<void*>(sf_ptr),
                 reinterpret_cast<void*>(topk_idx_ptr),
                 reinterpret_cast<void*>(topk_weights_ptr),
                 reinterpret_cast<void*>(copied_topk_idx_ptr),
                 reinterpret_cast<void*>(
                     psum_num_recv_tokens_per_scaleup_rank_ptr),
                 reinterpret_cast<void*>(psum_num_recv_tokens_per_expert_ptr),
                 reinterpret_cast<void*>(dst_buffer_slot_idx_ptr),
                 reinterpret_cast<void*>(token_metadata_at_forward_ptr),
                 reinterpret_cast<void*>(nccl_dev_comm_ptr),
                 reinterpret_cast<void*>(nccl_window_ptr),
                 reinterpret_cast<void*>(buffer_ptr),
                 reinterpret_cast<void*>(workspace_ptr),
                 reinterpret_cast<void*>(mapped_host_workspace_ptr),
                 reinterpret_cast<void*>(d2h_queues_ptr), num_queues,
                 static_cast<unsigned long>(signal_scratch_base),
                 static_cast<unsigned long>(atomic_tail_base),
                 reinterpret_cast<void*>(cuda_stream_ptr));
  }

  auto* topk_idx = checked_ptr<deep_ep::topk_idx_t>(topk_idx_ptr, "topk_idx");
  auto* psum_scaleup =
      checked_ptr<int>(psum_num_recv_tokens_per_scaleup_rank_ptr,
                       "psum_num_recv_tokens_per_scaleup_rank");
  auto* psum_expert =
      checked_ptr<int>(psum_num_recv_tokens_per_expert_ptr,
                       "psum_num_recv_tokens_per_expert");
  auto* dst_buffer_slot_idx =
      checked_ptr<int>(dst_buffer_slot_idx_ptr, "dst_buffer_slot_idx");
  auto* token_metadata_at_forward =
      checked_ptr<int>(token_metadata_at_forward_ptr,
                       "token_metadata_at_forward");
  auto* nccl_dev_comm_host =
      checked_ptr<ncclDevComm_t>(nccl_dev_comm_ptr, "nccl_dev_comm");
  auto nccl_dev_comm = *nccl_dev_comm_host;
  auto* buffer = checked_ptr<void>(buffer_ptr, "buffer");
  auto* workspace = checked_ptr<void>(workspace_ptr, "workspace");
  auto* mapped_host_workspace =
      checked_ptr<void>(mapped_host_workspace_ptr, "mapped_host_workspace");
  if (jit_debug_enabled()) {
    std::fprintf(stderr, "V2 EFA dispatch launching kernel\n");
  }
  auto launch_result = deep_ep::jit::launch_kernel(
      runtime->kernel, config, reinterpret_cast<void*>(x_ptr),
      reinterpret_cast<deep_ep::sf_pack_t*>(sf_ptr), topk_idx,
      reinterpret_cast<float*>(topk_weights_ptr),
      reinterpret_cast<deep_ep::topk_idx_t*>(copied_topk_idx_ptr),
      reinterpret_cast<int*>(cumulative_local_expert_recv_stats_ptr),
      psum_scaleup, psum_expert, dst_buffer_slot_idx,
      token_metadata_at_forward, num_tokens, sf_token_stride,
      sf_hidden_stride, nccl_dev_comm,
      reinterpret_cast<ncclWindow_t>(nccl_window_ptr), buffer, workspace,
      mapped_host_workspace, scaleout_rank, scaleup_rank,
      reinterpret_cast<const uint64_t*>(d2h_queues_ptr), num_queues,
      static_cast<uint64_t>(signal_scratch_base),
      static_cast<uint64_t>(atomic_tail_base));
  if (jit_debug_enabled()) {
    std::fprintf(stderr, "V2 EFA dispatch launch returned\n");
  }
  check_jit_launch_result(launch_result);
}

void launch_v2_efa_dispatch_copy_epilogue_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t buffer_ptr,
    std::uintptr_t workspace_ptr,
    std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
    std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
    std::uintptr_t recv_x_ptr, std::uintptr_t recv_sf_ptr,
    std::uintptr_t recv_topk_idx_ptr,
    std::uintptr_t recv_topk_weights_ptr,
    std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
    int recv_sf_token_stride, int recv_sf_hidden_stride, int scaleout_rank,
    int scaleup_rank, std::uintptr_t cuda_stream_ptr) {
  if (num_recv_tokens < 0 || recv_sf_token_stride < 0 ||
      recv_sf_hidden_stride < 0) {
    throw std::invalid_argument("invalid V2 EFA dispatch epilogue launch");
  }

  const auto runtime = build_v2_efa_jit_runtime(plan);
  auto config = make_launch_config(plan, runtime->kernel, cuda_stream_ptr);
  check_jit_launch_result(deep_ep::jit::launch_kernel(
      runtime->kernel, config, checked_ptr<void>(buffer_ptr, "buffer"),
      checked_ptr<void>(workspace_ptr, "workspace"),
      checked_ptr<int>(psum_num_recv_tokens_per_scaleup_rank_ptr,
                       "psum_num_recv_tokens_per_scaleup_rank"),
      checked_ptr<int>(psum_num_recv_tokens_per_expert_ptr,
                       "psum_num_recv_tokens_per_expert"),
      checked_ptr<void>(recv_x_ptr, "recv_x"),
      reinterpret_cast<deep_ep::sf_pack_t*>(recv_sf_ptr),
      reinterpret_cast<deep_ep::topk_idx_t*>(recv_topk_idx_ptr),
      reinterpret_cast<float*>(recv_topk_weights_ptr),
      checked_ptr<int>(recv_src_metadata_ptr, "recv_src_metadata"),
      reinterpret_cast<int*>(channel_linked_list_ptr), num_recv_tokens,
      recv_sf_token_stride, recv_sf_hidden_stride, scaleout_rank,
      scaleup_rank));
}

// ---------------------------------------------------------------------------
// V2EfaRuntime method implementations
// ---------------------------------------------------------------------------

void V2EfaRuntime::launch_native_hybrid_dispatch(
    std::uintptr_t x_ptr, std::uintptr_t sf_ptr,
    std::uintptr_t topk_idx_ptr, std::uintptr_t topk_weights_ptr,
    std::uintptr_t copied_topk_idx_ptr,
    std::uintptr_t cumulative_local_expert_recv_stats_ptr,
    std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
    std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
    std::uintptr_t dst_buffer_slot_idx_ptr,
    std::uintptr_t token_metadata_at_forward_ptr, int num_tokens,
    int num_max_tokens_per_rank, int num_channels_per_sm, int num_sf_packs,
    int sf_token_stride, int sf_hidden_stride, int expert_alignment,
    int num_qps, int64_t num_timeout_cycles, bool cached_mode,
    bool deterministic, bool do_cpu_sync, int smem_bytes,
    std::uintptr_t nccl_dev_comm_ptr, std::uintptr_t nccl_window_ptr,
    std::uintptr_t buffer_ptr, std::uintptr_t workspace_ptr,
    std::uintptr_t mapped_host_workspace_ptr,
    std::uintptr_t d2h_queues_ptr, uint32_t num_queues,
    std::uintptr_t signal_scratch_base, std::uintptr_t atomic_tail_base,
    const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_native_hybrid_dispatch_jit_plan(
      num_max_tokens_per_rank, num_channels_per_sm, num_sf_packs,
      expert_alignment, num_qps, num_timeout_cycles, cached_mode,
      deterministic, do_cpu_sync, smem_bytes, uccl_include_path);
  launch_v2_efa_native_hybrid_dispatch_plan(
      plan, x_ptr, sf_ptr, topk_idx_ptr, topk_weights_ptr,
      copied_topk_idx_ptr, cumulative_local_expert_recv_stats_ptr,
      psum_num_recv_tokens_per_scaleup_rank_ptr,
      psum_num_recv_tokens_per_expert_ptr, dst_buffer_slot_idx_ptr,
      token_metadata_at_forward_ptr, num_tokens, sf_token_stride,
      sf_hidden_stride, nccl_dev_comm_ptr, nccl_window_ptr, buffer_ptr,
      workspace_ptr, mapped_host_workspace_ptr, cfg.scaleout_rank,
      cfg.scaleup_rank, d2h_queues_ptr, num_queues, signal_scratch_base,
      atomic_tail_base, cuda_stream_ptr);
}

void V2EfaRuntime::launch_dispatch_copy_epilogue(
    std::uintptr_t buffer_ptr, std::uintptr_t workspace_ptr,
    std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
    std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
    std::uintptr_t recv_x_ptr, std::uintptr_t recv_sf_ptr,
    std::uintptr_t recv_topk_idx_ptr,
    std::uintptr_t recv_topk_weights_ptr,
    std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
    int num_max_tokens_per_rank, int num_channels, int num_sf_packs,
    int recv_sf_token_stride, int recv_sf_hidden_stride, bool do_expand,
    bool cached_mode, int smem_bytes, const std::string& uccl_include_path,
    std::uintptr_t cuda_stream_ptr) const {
  const auto& cfg = config();
  const auto plan = build_dispatch_copy_epilogue_jit_plan(
      num_max_tokens_per_rank, num_channels, num_sf_packs, do_expand,
      cached_mode, smem_bytes, uccl_include_path);
  launch_v2_efa_dispatch_copy_epilogue_plan(
      plan, buffer_ptr, workspace_ptr,
      psum_num_recv_tokens_per_scaleup_rank_ptr,
      psum_num_recv_tokens_per_expert_ptr, recv_x_ptr, recv_sf_ptr,
      recv_topk_idx_ptr, recv_topk_weights_ptr, recv_src_metadata_ptr,
      channel_linked_list_ptr, num_recv_tokens, recv_sf_token_stride,
      recv_sf_hidden_stride, cfg.scaleout_rank, cfg.scaleup_rank,
      cuda_stream_ptr);
}

}  // namespace uccl::v2_efa
