#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::qwen35::runtime {

struct HostExecutionContext;

enum class Qwen35KernelMode : uint8_t {
    Correctness,
    Auto,
};

Result<Qwen35KernelMode> read_qwen35_kernel_mode();

enum class OptimizedLaunchKind : uint8_t {
    NotApplicable,
    Launched,
    Skipped,
};

struct OptimizedLaunchResult {
    OptimizedLaunchKind kind = OptimizedLaunchKind::NotApplicable;
    uint32_t consumed_dispatches = 0;
};

Result<OptimizedLaunchResult> try_launch_linear_psq4(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_rmsnorm(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_activation_quantize_a8(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_linear_psq8(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_optimized(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_lm_head_proxy_fusion(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

void record_correctness_fallback(::ps::runtime::KernelId id);

uint32_t embedding_optimized_count() noexcept;

}
