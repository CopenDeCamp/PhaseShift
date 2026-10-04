#include <phaseshift/models/qwen35/kernels/optimized/verify_accept.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

__device__ __forceinline__ uint32_t verify_accept_prefix_row(
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ sampled_tokens, uint32_t count,
    uint32_t max_candidates) {
    if (count > max_candidates) count = max_candidates;
    uint32_t accepted = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        if (sampled_tokens[i] != candidates[i]) break;
        ++accepted;
    }
    return accepted;
}

}  // namespace

extern "C" __global__ __launch_bounds__(kVerifyAcceptPrefixThreads)
void phaseshift_gpu_mcu_verify_accept_prefix(VerifyAcceptPrefixArgs args) {
    for (uint32_t request = threadIdx.x; request < args.request_count;
         request += blockDim.x) {
        const uint32_t count = args.candidate_counts[request];
        const int32_t* candidates =
            args.candidates + static_cast<size_t>(request) * args.candidate_stride;
        if (args.requests != nullptr && args.output_rows != nullptr) {
            const ::ps::runtime::DeviceRequestDescriptor& descriptor =
                args.requests[request];
            if (descriptor.execution_class !=
                    ::ps::runtime::ExecutionClass::SPEC_VERIFY ||
                descriptor.output_index == 0xFFFFFFFFu) {
                args.committed_counts[request] = 0u;
                continue;
            }
            uint32_t bounded =
                count > args.max_candidates ? args.max_candidates : count;
            uint32_t accepted = 0u;
            for (uint32_t i = 0u; i < bounded; ++i) {
                const uint32_t row = args.output_rows[descriptor.output_index + i];
                if (args.sampled_tokens[row] != candidates[i]) break;
                ++accepted;
            }
            args.committed_counts[request] = accepted;
            continue;
        }
        const int32_t* sampled =
            args.sampled_tokens +
            static_cast<size_t>(request) * args.candidate_stride;
        args.committed_counts[request] = verify_accept_prefix_row(
            candidates, sampled, count, args.max_candidates);
    }
}

extern "C" __global__ __launch_bounds__(kVerifyAcceptPrefixThreads)
void phaseshift_gpu_mcu_verify_accept_batch(VerifyAcceptBatchArgs args) {
    if (args.context == nullptr) return;
    for (uint32_t request = threadIdx.x; request < args.context->num_requests;
         request += blockDim.x) {
        const ::ps::runtime::DeviceRequestDescriptor& descriptor =
            args.context->requests[request];
        if (descriptor.execution_class !=
                ::ps::runtime::ExecutionClass::SPEC_VERIFY ||
            descriptor.output_index == 0xFFFFFFFFu) {
            args.committed_counts[request] = 0u;
            continue;
        }
        uint32_t count = descriptor.row_count;
        if (count > args.max_candidates) count = args.max_candidates;
        const int32_t* candidates =
            args.context->token_ids + descriptor.row_begin;
        uint32_t accepted = 0u;
        for (uint32_t i = 0u; i < count; ++i) {
            const uint32_t row =
                args.context->output_rows[descriptor.output_index + i];
            if (args.sampled_tokens[row] != candidates[i]) break;
            ++accepted;
        }
        args.committed_counts[request] = accepted;
    }
}

hipError_t launch_verify_accept_batch(
    const ::ps::runtime::DeviceBatchContext* context,
    const int32_t* sampled_tokens, uint32_t* committed_counts,
    uint32_t max_candidates, hipStream_t stream) {
    if (context == nullptr || sampled_tokens == nullptr ||
        committed_counts == nullptr) {
        return hipErrorInvalidValue;
    }
    if (max_candidates == 0u) return hipErrorInvalidValue;
    VerifyAcceptBatchArgs args{};
    args.context = context;
    args.sampled_tokens = sampled_tokens;
    args.committed_counts = committed_counts;
    args.max_candidates = max_candidates;
    phaseshift_gpu_mcu_verify_accept_batch<<<1, kVerifyAcceptPrefixThreads, 0,
                                             stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_verify_accept_prefix(
    const int32_t* candidates, const int32_t* sampled_tokens,
    const uint32_t* candidate_counts, uint32_t* committed_counts,
    uint32_t request_count, uint32_t candidate_stride,
    uint32_t max_candidates, hipStream_t stream) {
    if (candidates == nullptr || sampled_tokens == nullptr ||
        candidate_counts == nullptr || committed_counts == nullptr) {
        return hipErrorInvalidValue;
    }
    if (request_count == 0u || candidate_stride == 0u ||
        max_candidates == 0u) {
        return hipErrorInvalidValue;
    }
    VerifyAcceptPrefixArgs args{};
    args.candidates = candidates;
    args.sampled_tokens = sampled_tokens;
    args.candidate_counts = candidate_counts;
    args.committed_counts = committed_counts;
    args.request_count = request_count;
    args.candidate_stride = candidate_stride;
    args.max_candidates = max_candidates;
    phaseshift_gpu_mcu_verify_accept_prefix<<<1, kVerifyAcceptPrefixThreads, 0,
                                              stream>>>(args);
    return hipGetLastError();
}

}  // namespace ps::kernel
