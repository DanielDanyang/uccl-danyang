#include "v2_efa/runtime.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace uccl::v2_efa {

namespace {

void validate_non_negative(const char* name, int value) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
}

void validate_config(const RuntimeConfig& config) {
  validate_non_negative("rank", config.rank);
  validate_non_negative("world_size", config.world_size);
  validate_non_negative("scaleout_rank", config.scaleout_rank);
  validate_non_negative("scaleup_rank", config.scaleup_rank);
  validate_non_negative("num_scaleout_ranks", config.num_scaleout_ranks);
  validate_non_negative("num_scaleup_ranks", config.num_scaleup_ranks);
  validate_non_negative("num_experts", config.num_experts);
  validate_non_negative("num_topk", config.num_topk);
  validate_non_negative("hidden", config.hidden);
  validate_non_negative("elem_bytes", config.elem_bytes);
  validate_non_negative("num_sms", config.num_sms);
  if (config.world_size == 0 || config.num_scaleout_ranks == 0 ||
      config.num_scaleup_ranks == 0 || config.num_experts == 0 ||
      config.num_topk == 0 || config.elem_bytes == 0) {
    throw std::invalid_argument("V2EfaRuntimeConfig contains a zero dimension");
  }
  if (config.num_scaleout_ranks * config.num_scaleup_ranks !=
      config.world_size) {
    throw std::invalid_argument(
        "num_scaleout_ranks * num_scaleup_ranks must equal world_size");
  }
  if (config.scaleout_rank * config.num_scaleup_ranks + config.scaleup_rank !=
      config.rank) {
    throw std::invalid_argument(
        "rank must equal scaleout_rank * num_scaleup_ranks + scaleup_rank");
  }
  (void)experts_per_rank(config.num_experts, config.world_size);
}

V2EfaDispatchJitConfig make_dispatch_jit_config(const RuntimeConfig& config) {
  V2EfaDispatchJitConfig jit_config;
  jit_config.num_scaleout_ranks = config.num_scaleout_ranks;
  jit_config.num_scaleup_ranks = config.num_scaleup_ranks;
  jit_config.num_experts = config.num_experts;
  jit_config.num_topk = config.num_topk;
  jit_config.hidden = config.hidden;
  jit_config.elem_bytes = config.elem_bytes;
  jit_config.num_sms = config.num_sms > 0 ? config.num_sms : 1;
  jit_config.scaleout_rank = config.scaleout_rank;
  jit_config.scaleup_rank = config.scaleup_rank;
  return jit_config;
}

}  // namespace

V2EfaRuntime::V2EfaRuntime(RuntimeConfig config) : config_(config) {
  validate_config(config_);
}

std::string V2EfaRuntime::status() const {
  return "native V2 AWS EFA runtime: dispatch replaces scaleout GIN with UCCL "
         "D2H TransferCmd writes over a single registered NCCL window";
}

ExpertRoute V2EfaRuntime::route_expert(int expert_id) const {
  return uccl::v2_efa::route_expert(expert_id, config_.num_experts,
                                    config_.world_size,
                                    config_.num_scaleup_ranks,
                                    config_.scaleout_rank);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_native_hybrid_dispatch_jit_plan(
    int num_max_tokens_per_rank, int num_channels_per_sm, int num_sf_packs,
    int expert_alignment, int num_qps, int64_t num_timeout_cycles,
    bool cached_mode, bool deterministic, bool do_cpu_sync, int smem_bytes,
    const std::string& uccl_include_path) const {
  auto jit_config = make_dispatch_jit_config(config_);
  jit_config.num_channels_per_sm = num_channels_per_sm;
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.num_sf_packs = num_sf_packs;
  jit_config.expert_alignment = expert_alignment;
  jit_config.num_qps = num_qps;
  jit_config.num_timeout_cycles = num_timeout_cycles;
  jit_config.cached_mode = cached_mode;
  jit_config.deterministic = deterministic;
  jit_config.do_cpu_sync = do_cpu_sync;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_native_hybrid_dispatch_jit_plan(jit_config);
}

V2EfaJitLaunchPlan V2EfaRuntime::build_dispatch_copy_epilogue_jit_plan(
    int num_max_tokens_per_rank, int num_channels, int num_sf_packs,
    bool do_expand, bool cached_mode, int smem_bytes,
    const std::string& uccl_include_path) const {
  auto jit_config = make_dispatch_jit_config(config_);
  jit_config.num_channels_per_sm =
      std::max(1, num_channels / jit_config.num_sms);
  jit_config.num_max_tokens_per_rank = num_max_tokens_per_rank;
  jit_config.num_sf_packs = num_sf_packs;
  jit_config.smem_bytes = smem_bytes;
  jit_config.uccl_include_path = uccl_include_path;
  return build_v2_efa_dispatch_copy_epilogue_jit_plan(jit_config, num_channels,
                                                      do_expand, cached_mode);
}

}  // namespace uccl::v2_efa
