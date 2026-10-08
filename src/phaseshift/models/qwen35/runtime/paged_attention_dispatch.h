#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::qwen35::runtime {

struct HostExecutionContext;

struct PagedAttentionResolvedArgs {
    ::ps::qwen35::KVCacheDType kv_dtype = ::ps::qwen35::KVCacheDType::BF16;
    ::ps::runtime::ValueDType output_dtype = ::ps::runtime::ValueDType::BF16;
    ::ps::kernel::PagedAttentionCommonArgs common;
    const bf16_t* k_bf16_pool = nullptr;
    const bf16_t* v_bf16_pool = nullptr;
    const ::ps::qwen35::fp8e4m3_storage_t* k_fp8_pool = nullptr;
    const ::ps::qwen35::fp8e4m3_storage_t* v_fp8_pool = nullptr;
    const float* k_scale_pool = nullptr;
    const float* v_scale_pool = nullptr;
    const uint8_t* k_psq4_code_pool = nullptr;
    const uint8_t* v_psq4_code_pool = nullptr;
    const bf16_t* k_psq4_scale_pool = nullptr;
    const bf16_t* v_psq4_scale_pool = nullptr;
    uint32_t psq4_blocks_per_head = 0;
    const uint8_t* k_psq8_code_pool = nullptr;
    const uint8_t* v_psq8_code_pool = nullptr;
    const bf16_t* k_psq8_scale_pool = nullptr;
    const bf16_t* v_psq8_scale_pool = nullptr;
    uint32_t psq8_blocks_per_head = 0;
};

Result<PagedAttentionResolvedArgs>
resolve_paged_attention_args(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx);

struct PagedAttentionPlan {
    bool optimized = false;
    bool use_prefill = false;
    uint32_t splits = 1u;
};

constexpr uint32_t kPagedAttentionPrefillMinRows = 128u;
constexpr uint32_t kPagedAttentionSplitMinVisible = 2048u;

uint32_t make_paged_attention_plan_key(const PagedAttentionPlan& plan);

struct McuStaticAttentionPlanClass {
    bool has_paged_attention = false;
    bool supported = false;

    uint32_t key = 0u;

    uint32_t splits = 1u;

    uint32_t canonical_max_visible = 0u;
};

Result<McuStaticAttentionPlanClass> classify_static_attention_plan(
    const ::ps::runtime::Program& program,
    const HostExecutionContext& live_ctx,
    uint32_t row_capacity);

PagedAttentionPlan plan_paged_attention(
    const PagedAttentionResolvedArgs& args,
    const HostExecutionContext& ctx,
    bool use_prefill);

Result<uint32_t> paged_attention_plan_signature(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    bool use_prefill);

Result<OptimizedLaunchResult>
try_launch_paged_attention(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::qwen35::runtime
