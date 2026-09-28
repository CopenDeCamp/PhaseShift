#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/sampling_common.h>

namespace ps::kernel::detail {

__device__ __forceinline__ ProgramStatus exec_gather_or_sample(
    ModelDispatchEnv& env, const DispatchBinding& b, KernelId kind) {
    if (env.context == nullptr) return ProgramStatus::INVALID_BINDING;

    ResolvedValue in{};
    if (!resolve_value(env, b.input_slots[0], in)) return ProgramStatus::INVALID_BINDING;

    const uint32_t outputs = env.context->num_outputs;

    if (kind == KernelId::OUTPUT_GATHER) {
        ResolvedValue out{};
        if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
        if (in.dtype != out.dtype || in.features != out.features)
            return ProgramStatus::INVALID_BINDING;
        if (out.row_domain != ValueRowDomain::OUTPUT_ROWS)
            return ProgramStatus::INVALID_BINDING;
        if (outputs > 0 && env.context->output_rows == nullptr)
            return ProgramStatus::INVALID_BINDING;
        const uint32_t f = out.features;
        for (uint32_t i = 0; i < outputs; ++i) {
            const uint32_t src_row = env.context->output_rows[i];
            const size_t src_base = static_cast<size_t>(src_row) * in.row_stride;
            const size_t dst_base = static_cast<size_t>(i) * out.row_stride;
            for (uint32_t c = global_tid(env); c < f; c += global_stride(env)) {
                store_elem(out, dst_base + c, load_elem(in, src_base + c));
            }
            __syncthreads();
        }
        return ProgramStatus::COMPLETE;
    }

    ResolvedValue out{};
    if (!resolve_value(env, b.output_slots[0], out)) return ProgramStatus::INVALID_BINDING;
    if (out.dtype != ValueDType::I32) return ProgramStatus::INVALID_BINDING;
    if (out.row_domain != ValueRowDomain::OUTPUT_ROWS)
        return ProgramStatus::INVALID_BINDING;
    if (outputs > 0 &&
        (out.ptr == nullptr || env.context->output_sampling_params == nullptr))
        return ProgramStatus::INVALID_BINDING;
    if (outputs > 0 && in.dtype != ValueDType::F32) return ProgramStatus::INVALID_BINDING;

    auto* sampled_tokens = static_cast<int32_t*>(out.ptr);
    const uint32_t vocab = in.features;
    const auto* logits = static_cast<const float*>(in.ptr);

    for (uint32_t row = blockIdx.x; row < outputs; row += gridDim.x) {
        const SamplingRowParams params =
            sampling_row_params(env.context->output_sampling_params[row]);
        SamplingConstraint constraint;
        if ((env.context->output_sampling_params[row].reserved[0] & 1u) != 0u &&
            env.context->constraint_masks != nullptr) {
            constraint.active = true;
            constraint.mask = env.context->constraint_masks +
                              static_cast<std::size_t>(row) *
                                  env.context->constraint_mask_words;
        }
        const float* row_logits =
            logits + static_cast<size_t>(row) * in.row_stride;
        sampling_sample_row(row_logits, vocab, params, constraint, sampled_tokens + row,
                            env.error_word);
    }
    return ProgramStatus::COMPLETE;
}

}  // namespace ps::kernel::detail
