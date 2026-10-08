#include <phaseshift/models/qwen35/runtime/physical_launch_static.h>

namespace ps::qwen35::runtime {

Result<HostResolvedValue> resolve_host_value_static(
    const ::ps::runtime::Program& program, ::ps::runtime::ValueId id,
    const McuPlanResolverInput& input) {
    return resolve_host_value(program, id, input.ctx);
}

#define PHASESHIFT_MCU_STATIC_RESOLVE(name, launch_type)                       \
    Result<PhysicalResolveStatus> name##_static(                              \
        const ::ps::runtime::Program& program,                                \
        const ::ps::runtime::DispatchBinding& b,                              \
        const McuPlanResolverInput& input, launch_type& out) {                 \
        return name(program, b, input.ctx, out);                              \
    }

PHASESHIFT_MCU_STATIC_RESOLVE(resolve_embedding_physical, PhysicalEmbeddingLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_rmsnorm_physical, PhysicalRmsNormLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_activation_quantize_physical,
                              PhysicalActivationQuantizeLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_elementwise_physical,
                              PhysicalElementwiseLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_bf16_physical, PhysicalBf16Launch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_l2_normalize_physical,
                              PhysicalL2NormalizeLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_gdn_conv1d_physical,
                              PhysicalGdnConv1dLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_gdn_recurrence_physical,
                              PhysicalGdnRecurrenceLaunch)
PHASESHIFT_MCU_STATIC_RESOLVE(resolve_rope_physical, PhysicalRopeLaunch)

#undef PHASESHIFT_MCU_STATIC_RESOLVE

Result<PhysicalResolveStatus> resolve_psq4_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input, PhysicalPsq4Launch& out) {
    return resolve_psq4_physical(program, b, input.ctx, false, out);
}

Result<PhysicalResolveStatus> resolve_psq8_physical_static(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const McuPlanResolverInput& input, PhysicalPsq8Launch& out) {
    return resolve_psq8_physical(program, b, input.ctx, false, out);
}

Result<PhysicalResolveStatus> resolve_kv_append_physical_static(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const McuPlanResolverInput& input, PhysicalKvAppendLaunch& out) {
    return resolve_kv_append_physical(program, dispatch_index, input.ctx, out);
}

Result<PhysicalResolveStatus> resolve_paged_attention_static_plan(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const McuPlanResolverInput& input, PhysicalPagedAttentionPlan& out) {
    return resolve_paged_attention_physical_static_plan(
        program, dispatch_index, input.ctx, out);
}

Result<PhysicalResolveStatus> resolve_paged_attention_host_immediate(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    const McuPlanResolverInput& input, PhysicalPagedAttentionPlan& out) {
    return resolve_paged_attention_physical(program, dispatch_index, input.ctx,
                                            out);
}

}  // namespace ps::qwen35::runtime
