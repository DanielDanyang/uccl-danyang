#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace uccl::v2_efa {

struct V2EfaJitLaunchPlan {
  std::string name;
  std::string source;
  int grid_dim_x = 0;
  int grid_dim_y = 1;
  int num_threads = 0;
  int smem_bytes = 0;
  int cluster_dim = 1;
  bool cooperative = false;
  bool pdl_enabled = false;
  int num_notify_warps = 0;
  int num_scaleout_warps = 0;
  int num_forward_warps = 0;
  int num_payload_warps = 0;
};

struct V2EfaDispatchJitConfig {
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int num_experts = 1;
  int num_topk = 1;
  int hidden = 1;
  int elem_bytes = 2;
  int num_sms = 1;
  int num_channels_per_sm = 1;
  int num_max_tokens_per_rank = 1;
  int scaleout_rank = 0;
  int scaleup_rank = 0;
  int scale_bytes = 0;
  int num_sf_packs = 0;
  int expert_alignment = 1;
  int num_qps = 1;
  int64_t num_timeout_cycles = 200000000000ll;
  bool has_topk_weight = true;
  bool cached_mode = false;
  bool deterministic = false;
  bool do_cpu_sync = false;
  int smem_bytes = 228 * 1024;
  std::string uccl_include_path;
};

inline void validate_v2_efa_jit_common(int num_scaleout_ranks,
                                       int num_scaleup_ranks,
                                       int num_experts, int num_topk,
                                       int num_sms,
                                       int num_max_tokens_per_rank) {
  if (num_scaleout_ranks <= 0 || num_scaleup_ranks <= 0 || num_experts <= 0 ||
      num_topk <= 0 || num_sms <= 0 || num_max_tokens_per_rank <= 0) {
    throw std::invalid_argument("invalid V2 EFA JIT dimensions");
  }
  const int world_size = num_scaleout_ranks * num_scaleup_ranks;
  if (num_experts % world_size != 0) {
    throw std::invalid_argument(
        "num_experts must be divisible by V2 EFA world size");
  }
}

inline int align_up_int(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline int v2_token_layout_bytes(int hidden_bytes, int sf_bytes, int num_topk,
                                 bool with_metadata, bool with_mbarrier) {
  constexpr int kTmaAlignBytes = 32;
  constexpr int kMBarrierBytes = 8;
  const int metadata_bytes =
      num_topk * (static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float))) +
      (with_metadata ? (1 + num_topk) * static_cast<int>(sizeof(int)) : 0);
  return align_up_int(hidden_bytes, kTmaAlignBytes) +
         align_up_int(sf_bytes, kTmaAlignBytes) +
         align_up_int(metadata_bytes, kTmaAlignBytes) +
         align_up_int(with_mbarrier ? kMBarrierBytes : 0, kTmaAlignBytes);
}

inline std::string quote_include(const std::string& include_root,
                                 const char* header) {
  if (include_root.empty()) {
    return std::string("<") + header + ">";
  }
  auto root = include_root;
  while (!root.empty() && root.back() == '/') {
    root.pop_back();
  }
  return std::string("\"") + root + "/" + header + "\"";
}

inline int default_dispatch_num_sms(const V2EfaDispatchJitConfig& config) {
  return config.num_sms > 0 ? config.num_sms : 1;
}

// Native hybrid dispatch: scaleout GIN replaced by D2H TransferCmd WRITEs.  The
// kernel header self-includes ring_buffer.cuh + v2_efa/workspace.hpp (relative
// to its own directory), so the generated source only includes the deep_ep
// common headers and the kernel itself.
inline V2EfaJitLaunchPlan build_v2_efa_native_hybrid_dispatch_jit_plan(
    V2EfaDispatchJitConfig config) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.num_scaleout_ranks <= 1 || config.hidden <= 0 ||
      config.elem_bytes <= 0 || config.num_channels_per_sm <= 0 ||
      config.expert_alignment <= 0 || config.num_qps <= 0 ||
      config.num_timeout_cycles <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA native hybrid dispatch JIT config");
  }
  if (config.deterministic) {
    throw std::invalid_argument(
        "native V2 EFA hybrid dispatch does not support deterministic mode");
  }

  constexpr int kNumNotifyWarps = 4;
  const int num_notify_warps = config.cached_mode ? 0 : kNumNotifyWarps;
  const int num_scaleout_warps = config.num_channels_per_sm;
  const int num_forward_warps = config.num_channels_per_sm;
  const int num_threads =
      (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
  if (num_threads <= 0 || num_threads > 1024) {
    throw std::invalid_argument(
        "invalid V2 EFA native hybrid dispatch thread count");
  }

  const int hidden_bytes = config.hidden * config.elem_bytes;
  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_native_hybrid_dispatch";
  plan.grid_dim_x = config.num_sms;
  plan.grid_dim_y = 1;
  plan.num_threads = num_threads;
  plan.smem_bytes = config.smem_bytes;
  // Native V2 EFA dispatch still uses DeepEP V2 cooperative grid sync inside
  // notify/forward barriers.  AWS CUDA 13 was crashing with the original
  // clustered cooperative launch, so keep clusters disabled but preserve the
  // cooperative launch required by cooperative_groups::this_grid().sync().
  plan.cluster_dim = 1;
  plan.cooperative = true;
  plan.pdl_enabled = false;
  plan.num_notify_warps = num_notify_warps;
  plan.num_scaleout_warps = num_scaleout_warps;
  plan.num_forward_warps = num_forward_warps;
  // Upstream DeepEP V2 uses 3 for IB/Gin streaming.  The AWS EFA path applies
  // tail updates through CPU-proxy software atomics, so a coarser semantic batch
  // is materially faster while preserving the same V2 channel/slot semantics.
  int scaleout_update_interval = 32;
  if (const char* env = std::getenv("UCCL_V2_SCALEOUT_UPDATE_INTERVAL");
      env != nullptr && env[0] != '\0') {
    scaleout_update_interval = std::atoi(env);
    if (scaleout_update_interval <= 0) {
      throw std::invalid_argument(
          "UCCL_V2_SCALEOUT_UPDATE_INTERVAL must be positive");
    }
  }

  std::ostringstream source;
  if (const char* debug_finish =
          std::getenv("UCCL_V2_DEBUG_FINISH_PRINT");
      debug_finish != nullptr && debug_finish[0] != '\0' &&
      debug_finish[0] != '0') {
    source << "#define UCCL_V2_DEBUG_FINISH_PRINT 1\n";
  }
  source << "#include <deep_ep/common/comm.cuh>\n"
         << "#include <deep_ep/common/compiled.cuh>\n"
         << "#include <deep_ep/common/exception.cuh>\n"
         << "#include <deep_ep/common/layout.cuh>\n"
         << "#include <deep_ep/common/math.cuh>\n"
         << "#include <deep_ep/common/ptx.cuh>\n"
         << "#include "
         << quote_include(config.uccl_include_path,
                          "v2_efa/hybrid_dispatch_native.cuh")
         << "\n\n"
         << "using namespace deep_ep::elastic;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "hybrid_dispatch_impl<"
         << (config.do_cpu_sync ? "true" : "false") << ", "
         << ((config.cached_mode || config.deterministic) ? "true" : "false")
         << ", "
         << config.num_sms << ", " << num_notify_warps << ", "
         << num_scaleout_warps << ", " << num_forward_warps << ", "
         << config.num_scaleout_ranks << ", " << config.num_scaleup_ranks
         << ", " << hidden_bytes << ", " << config.num_sf_packs << ", "
         << config.num_max_tokens_per_rank << ", " << config.num_experts
         << ", " << config.num_topk << ", " << config.expert_alignment
         << ", " << config.num_qps << ", " << config.num_timeout_cycles
         << ", " << "deep_ep::elastic::math::constexpr_ceil_div("
         << "static_cast<int>(" << config.num_scaleup_ranks << "), 32)"
         << ", " << num_scaleout_warps << ", "
         << (num_scaleout_warps * config.num_sms) << ", "
         << "deep_ep::elastic::math::constexpr_ceil_div("
         << config.num_max_tokens_per_rank << ", "
         << (num_scaleout_warps * config.num_sms) << "), "
         << scaleout_update_interval << ", " << scaleout_update_interval
         << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

inline V2EfaJitLaunchPlan build_v2_efa_dispatch_copy_epilogue_jit_plan(
    V2EfaDispatchJitConfig config, int num_channels, bool do_expand,
    bool cached_mode) {
  config.num_sms = default_dispatch_num_sms(config);
  validate_v2_efa_jit_common(
      config.num_scaleout_ranks, config.num_scaleup_ranks, config.num_experts,
      config.num_topk, config.num_sms, config.num_max_tokens_per_rank);
  if (config.hidden <= 0 || config.elem_bytes <= 0 ||
      config.num_sf_packs < 0 || num_channels <= 0 ||
      config.smem_bytes <= 0) {
    throw std::invalid_argument(
        "invalid V2 EFA dispatch copy-epilogue JIT config");
  }

  const int hidden_bytes = config.hidden * config.elem_bytes;
  const int token_bytes = v2_token_layout_bytes(
      hidden_bytes, config.num_sf_packs * static_cast<int>(sizeof(float)),
      config.num_topk, true, true);
  const int num_warps = std::min(config.smem_bytes / token_bytes, 32);
  if (num_warps <= 0) {
    throw std::invalid_argument(
        "insufficient shared memory for V2 dispatch copy epilogue");
  }

  V2EfaJitLaunchPlan plan;
  plan.name = "v2_efa_dispatch_copy_epilogue";
  plan.grid_dim_x = config.num_sms;
  plan.grid_dim_y = 1;
  plan.num_threads = num_warps * 32;
  plan.smem_bytes = config.smem_bytes;
  plan.cluster_dim = 1;
  plan.cooperative = false;
  plan.pdl_enabled = true;

  std::ostringstream source;
  source << "#include <deep_ep/impls/dispatch_copy_epilogue.cuh>\n\n"
         << "using namespace deep_ep::elastic;\n\n"
         << "static void __instantiate_kernel() {\n"
         << "    auto ptr = reinterpret_cast<void*>(&"
         << "dispatch_copy_epilogue_impl<"
         << (do_expand ? "true" : "false") << ", "
         << (cached_mode ? "true" : "false") << ", "
         << config.num_sms << ", " << num_channels << ", " << num_warps
         << ", " << config.num_scaleout_ranks << ", "
         << config.num_scaleup_ranks << ", " << hidden_bytes << ", "
         << config.num_sf_packs << ", " << config.num_max_tokens_per_rank
         << ", " << config.num_experts << ", " << config.num_topk
         << ">);\n"
         << "}\n";
  plan.source = source.str();
  return plan;
}

}  // namespace uccl::v2_efa
