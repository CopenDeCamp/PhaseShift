#include <phaseshift/models/qwen35/kernels/optimized/stochastic_sampling.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {
namespace {

constexpr uint32_t kThreads = 256u;

__global__ void stochastic_sampling_impl(
    const float* __restrict__ logits,
    const ::ps::runtime::DeviceSamplingParams* __restrict__ sampling,
    int32_t* __restrict__ sampled_tokens,
    uint32_t vocab_size,
    uint32_t logits_row_stride,
    const uint32_t* __restrict__ constraint_masks,
    uint32_t constraint_mask_words,
    uint32_t* error_word,
    uint32_t* __restrict__ attempts_out) {
    const uint32_t row = blockIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * logits_row_stride;
    const SamplingRowParams params = sampling_row_params(sampling[row]);
    SamplingConstraint constraint;
    if ((sampling[row].reserved[0] & 1u) != 0u && constraint_masks != nullptr) {
        constraint.active = true;
        constraint.mask =
            constraint_masks + static_cast<std::size_t>(row) * constraint_mask_words;
    }
    sampling_sample_row(row_logits, vocab_size, params, constraint, sampled_tokens + row,
                        error_word, attempts_out == nullptr ? nullptr : attempts_out + row);
}

}  // namespace

hipError_t launch_stochastic_sampling_f32_singleblock(
    const StochasticSamplingF32Args& args,
    hipStream_t stream) {
    if (args.outputs == 0u) return hipSuccess;
    if (args.vocab_size == 0u) return hipErrorInvalidValue;
    if (args.logits == nullptr || args.sampling == nullptr ||
        args.sampled_tokens == nullptr)
        return hipErrorInvalidValue;
    stochastic_sampling_impl<<<args.outputs, kThreads, 0, stream>>>(
        args.logits, args.sampling, args.sampled_tokens, args.vocab_size,
        args.logits_row_stride, args.constraint_masks, args.constraint_mask_words,
        args.error_word, args.attempts_out);
    return hipGetLastError();
}

}  // namespace ps::kernel
