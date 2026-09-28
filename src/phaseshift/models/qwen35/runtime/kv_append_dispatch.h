#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::qwen35::runtime {

struct HostExecutionContext;

struct KvAppendResolvedArgs {
    ::ps::qwen35::KVCacheDType kv_dtype = ::ps::qwen35::KVCacheDType::BF16;
    ::ps::kernel::KvAppendCommonArgs common;
    bf16_t* k_bf16_pool = nullptr;
    bf16_t* v_bf16_pool = nullptr;
    ::ps::qwen35::fp8e4m3_storage_t* k_fp8_pool = nullptr;
    ::ps::qwen35::fp8e4m3_storage_t* v_fp8_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;
    uint8_t* k_psq4_code_pool = nullptr;
    uint8_t* v_psq4_code_pool = nullptr;
    bf16_t* k_psq4_scale_pool = nullptr;
    bf16_t* v_psq4_scale_pool = nullptr;
    uint32_t psq4_blocks_per_head = 0;
    ::ps::qwen35::Psq4ScaleEstimator psq4_estimator = ::ps::qwen35::Psq4ScaleEstimator::Lsq1;
    uint8_t* k_psq8_code_pool = nullptr;
    uint8_t* v_psq8_code_pool = nullptr;
    bf16_t* k_psq8_scale_pool = nullptr;
    bf16_t* v_psq8_scale_pool = nullptr;
    uint32_t psq8_blocks_per_head = 0;
    ::ps::qwen35::Psq8ScaleEstimator psq8_estimator = ::ps::qwen35::Psq8ScaleEstimator::Lsq1;
};

Result<KvAppendResolvedArgs>
resolve_kv_append_args(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx);

Result<OptimizedLaunchResult>
try_launch_kv_append(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::qwen35::runtime
