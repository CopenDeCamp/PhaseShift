#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/runtime/program/program.h>

#include <phaseshift/runtime/graph/primitive_graph.h>

#include <cstdint>
#include <span>

namespace ps::qwen35 {

struct Qwen35LoweredPrimitives {
    ps::runtime::PrimitiveGraph graph;
    std::vector<ps::runtime::WeightSlot> weights;
    std::vector<ps::runtime::StaticParameterSlot> parameters;
};

struct ModelPartition {
    uint32_t layer_begin = 0;
    uint32_t layer_end = 0;
    bool owns_embedding = true;
    bool owns_lm_head = true;
};

struct Qwen35LowerOptions {
    std::span<const uint32_t> hidden_taps;
    ModelPartition partition{};
    uint32_t pipeline_peer_rank = 0;
    uint32_t tensor_parallel_size = 1;
    uint32_t tensor_parallel_rank = 0;
};

Result<Qwen35LoweredPrimitives> lower_qwen35_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const Qwen35LowerOptions& options = {});

Result<Qwen35LoweredPrimitives> lower_qwen35_mtp_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights);

}
