#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace uccl::v2_efa {

struct ExpertRoute {
  int32_t expert_id = -1;
  int32_t owner_rank = -1;
  int32_t dst_scaleout_rank = -1;
  int32_t dst_scaleup_lane = -1;
  int32_t is_remote_scaleout = 0;
};

inline void validate_positive(const char* name, int value) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

inline int experts_per_rank(int num_experts, int world_size) {
  validate_positive("num_experts", num_experts);
  validate_positive("world_size", world_size);
  if (num_experts % world_size != 0) {
    throw std::invalid_argument("num_experts must be divisible by world_size");
  }
  return num_experts / world_size;
}

inline ExpertRoute route_expert(int expert_id, int num_experts, int world_size,
                                int num_scaleup_ranks,
                                int local_scaleout_rank) {
  validate_positive("num_scaleup_ranks", num_scaleup_ranks);
  const auto per_rank = experts_per_rank(num_experts, world_size);
  if (expert_id < 0 || expert_id >= num_experts) {
    throw std::invalid_argument("expert_id is out of range");
  }
  ExpertRoute route;
  route.expert_id = expert_id;
  route.owner_rank = expert_id / per_rank;
  route.dst_scaleout_rank = route.owner_rank / num_scaleup_ranks;
  route.dst_scaleup_lane = route.owner_rank % num_scaleup_ranks;
  route.is_remote_scaleout =
      route.dst_scaleout_rank == local_scaleout_rank ? 0 : 1;
  return route;
}

}  // namespace uccl::v2_efa
