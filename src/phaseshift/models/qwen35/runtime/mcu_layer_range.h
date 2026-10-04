#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/runtime/program/program.h>

#include <cstdint>

namespace ps::qwen35::runtime {

Status find_dispatch_range_for_graph_prefix(
    const ::ps::runtime::PrimitiveGraph& graph,
    const ::ps::runtime::Program& program, const char* prefix,
    uint32_t* out_begin, uint32_t* out_end);

}  // namespace ps::qwen35::runtime
