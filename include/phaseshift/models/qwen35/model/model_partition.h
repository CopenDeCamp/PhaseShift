#pragma once

#include <cstdint>

namespace ps::qwen35 {

struct ModelPartition {
    std::uint32_t layer_begin = 0;
    std::uint32_t layer_end = 0;
    bool owns_embedding = true;
    bool owns_lm_head = true;
};

inline ModelPartition resolve_model_partition(ModelPartition partition,
                                              std::uint32_t total_layers) {
    if (partition.layer_end == 0 || partition.layer_end > total_layers)
        partition.layer_end = total_layers;
    if (partition.layer_begin > partition.layer_end)
        partition.layer_begin = partition.layer_end;
    return partition;
}

}
