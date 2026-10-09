#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compile_context.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/runtime/program/program.h>

#include <cstddef>

namespace ps::qwen35::runtime {

Result<HostResolvedValue> resolve_host_value_static(
    const ::ps::runtime::Program& program,
    ::ps::runtime::ValueId id,
    const McuPlanResolverInput& input);

Result<PhysicalResolveStatus> resolve_embedding_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalEmbeddingLaunch& out);

Result<PhysicalResolveStatus> resolve_rmsnorm_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalRmsNormLaunch& out);

Result<PhysicalResolveStatus> resolve_activation_quantize_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalActivationQuantizeLaunch& out);

Result<PhysicalResolveStatus> resolve_psq4_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalPsq4Launch& out);

Result<PhysicalResolveStatus> resolve_psq8_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalPsq8Launch& out);

Result<PhysicalResolveStatus> resolve_elementwise_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalElementwiseLaunch& out);

Result<PhysicalResolveStatus> resolve_bf16_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalBf16Launch& out);

Result<PhysicalResolveStatus> resolve_l2_normalize_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalL2NormalizeLaunch& out);

Result<PhysicalResolveStatus> resolve_gdn_conv1d_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalGdnConv1dLaunch& out);

Result<PhysicalResolveStatus> resolve_gdn_recurrence_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalGdnRecurrenceLaunch& out);

Result<PhysicalResolveStatus> resolve_rope_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input,
    PhysicalRopeLaunch& out);

Result<PhysicalResolveStatus> resolve_kv_append_physical_static(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const McuPlanResolverInput& input,
    PhysicalKvAppendLaunch& out);

Result<PhysicalResolveStatus> resolve_paged_attention_static_plan(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const McuPlanResolverInput& input,
    PhysicalPagedAttentionPlan& out);

Result<PhysicalResolveStatus> resolve_paged_attention_static_prefill(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const McuPlanResolverInput& input,
    PhysicalPagedAttentionPlan& out);

Result<PhysicalResolveStatus> resolve_paged_attention_host_immediate(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const McuPlanResolverInput& input,
    PhysicalPagedAttentionPlan& out);

}  // namespace ps::qwen35::runtime
