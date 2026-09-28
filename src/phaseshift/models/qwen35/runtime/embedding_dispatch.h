#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <cstddef>

namespace ps::qwen35::runtime {

struct HostExecutionContext;

Result<OptimizedLaunchResult> try_launch_embedding(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::qwen35::runtime
