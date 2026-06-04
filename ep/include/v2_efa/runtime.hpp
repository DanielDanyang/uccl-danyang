#pragma once

#include <cstdint>
#include <string>

#include "v2_efa/jit_plan.hpp"
#include "v2_efa/topology.hpp"
#include "v2_efa/workspace.hpp"

namespace uccl::v2_efa {

struct RuntimeConfig {
  int rank = 0;
  int world_size = 1;
  int scaleout_rank = 0;
  int scaleup_rank = 0;
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int num_experts = 1;
  int num_topk = 1;
  int hidden = 0;
  int elem_bytes = 2;
  int num_sms = 0;
};

// JIT bridge lifecycle (implemented in v2_efa_deep_ep_jit.cc).
void init_deep_ep_jit_bridge(const std::string& library_root_path,
                             const std::string& cuda_home_path,
                             const std::string& nccl_root_path);
bool is_deep_ep_jit_bridge_initialized();
void compile_v2_efa_jit_plan(const V2EfaJitLaunchPlan& plan);

// Native hybrid dispatch launch.  The scaleout transport is the UCCL D2H queue
// path: `d2h_queues_ptr` is a GPU pointer to an array of DeviceToHostCmdBuffer*
// (host-pinned rings), `signal_scratch_base` is the mapped base of the tail
// scratch region carved from the registered NCCL window.  The kernel derives
// every transport offset relative to the mapped `workspace_ptr` (window base).
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
    int scaleup_rank, std::uintptr_t d2h_queues_ptr, uint32_t num_queues,
    std::uintptr_t signal_scratch_base, std::uintptr_t cuda_stream_ptr = 0);

void launch_v2_efa_dispatch_copy_epilogue_plan(
    const V2EfaJitLaunchPlan& plan, std::uintptr_t buffer_ptr,
    std::uintptr_t workspace_ptr,
    std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
    std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
    std::uintptr_t recv_x_ptr, std::uintptr_t recv_sf_ptr,
    std::uintptr_t recv_topk_idx_ptr, std::uintptr_t recv_topk_weights_ptr,
    std::uintptr_t recv_src_metadata_ptr,
    std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
    int recv_sf_token_stride, int recv_sf_hidden_stride, int scaleout_rank,
    int scaleup_rank, std::uintptr_t cuda_stream_ptr = 0);

class V2EfaRuntime {
 public:
  explicit V2EfaRuntime(RuntimeConfig config);

  const RuntimeConfig& config() const { return config_; }
  std::string status() const;

  ExpertRoute route_expert(int expert_id) const;

  V2EfaJitLaunchPlan build_native_hybrid_dispatch_jit_plan(
      int num_max_tokens_per_rank, int num_channels_per_sm, int num_sf_packs,
      int expert_alignment, int num_qps, int64_t num_timeout_cycles,
      bool cached_mode, bool deterministic, bool do_cpu_sync, int smem_bytes,
      const std::string& uccl_include_path = "") const;
  V2EfaJitLaunchPlan build_dispatch_copy_epilogue_jit_plan(
      int num_max_tokens_per_rank, int num_channels, int num_sf_packs,
      bool do_expand, bool cached_mode, int smem_bytes,
      const std::string& uccl_include_path = "") const;

  void launch_native_hybrid_dispatch(
      std::uintptr_t x_ptr, std::uintptr_t sf_ptr, std::uintptr_t topk_idx_ptr,
      std::uintptr_t topk_weights_ptr, std::uintptr_t copied_topk_idx_ptr,
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
      std::uintptr_t mapped_host_workspace_ptr, std::uintptr_t d2h_queues_ptr,
      uint32_t num_queues, std::uintptr_t signal_scratch_base,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;
  void launch_dispatch_copy_epilogue(
      std::uintptr_t buffer_ptr, std::uintptr_t workspace_ptr,
      std::uintptr_t psum_num_recv_tokens_per_scaleup_rank_ptr,
      std::uintptr_t psum_num_recv_tokens_per_expert_ptr,
      std::uintptr_t recv_x_ptr, std::uintptr_t recv_sf_ptr,
      std::uintptr_t recv_topk_idx_ptr, std::uintptr_t recv_topk_weights_ptr,
      std::uintptr_t recv_src_metadata_ptr,
      std::uintptr_t channel_linked_list_ptr, int num_recv_tokens,
      int num_max_tokens_per_rank, int num_channels, int num_sf_packs,
      int recv_sf_token_stride, int recv_sf_hidden_stride, bool do_expand,
      bool cached_mode, int smem_bytes,
      const std::string& uccl_include_path = "",
      std::uintptr_t cuda_stream_ptr = 0) const;

 private:
  RuntimeConfig config_;
};

}  // namespace uccl::v2_efa
