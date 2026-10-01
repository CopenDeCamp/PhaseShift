#include <phaseshift/models/qwen35/kernels/optimized/constraint_candidates.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {
namespace {

__global__ void constraint_mask_to_candidates_kernel(
    const uint32_t* __restrict__ masks,
    uint32_t mask_words,
    uint32_t vocab_size,
    uint32_t capacity,
    int32_t* __restrict__ candidate_ids) {
    const uint32_t row = blockIdx.x;
    const uint32_t* mask = masks + static_cast<std::size_t>(row) * mask_words;
    int32_t* out = candidate_ids + static_cast<std::size_t>(row) * capacity;

    const uint32_t full_words = vocab_size / 32u;
    const uint32_t remainder = vocab_size % 32u;
    const uint32_t scan_words = full_words < mask_words ? full_words : mask_words;

    uint32_t written = 0u;
    for (uint32_t w = 0u; w < scan_words && written < capacity; ++w) {
        uint32_t bits = mask[w];
        while (bits != 0u && written < capacity) {
            const uint32_t offset =
                static_cast<uint32_t>(::__ffs(static_cast<int>(bits))) - 1u;
            bits &= bits - 1u;
            out[written++] = static_cast<int32_t>(w * 32u + offset);
        }
    }
    if (remainder != 0u && scan_words < mask_words && written < capacity) {
        uint32_t bits = mask[scan_words] & ((1u << remainder) - 1u);
        while (bits != 0u && written < capacity) {
            const uint32_t offset =
                static_cast<uint32_t>(::__ffs(static_cast<int>(bits))) - 1u;
            bits &= bits - 1u;
            out[written++] = static_cast<int32_t>(scan_words * 32u + offset);
        }
    }
    if (written == 0u) {
        out[0] = 0;
        written = 1u;
    }
    const int32_t fill = out[0];
    for (uint32_t i = written; i < capacity; ++i) {
        out[i] = fill;
    }
}

}  // namespace

hipError_t launch_constraint_mask_to_candidates(
    const uint32_t* masks,
    uint32_t rows,
    uint32_t mask_words,
    uint32_t vocab_size,
    uint32_t candidate_capacity,
    int32_t* candidate_ids,
    hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (masks == nullptr || candidate_ids == nullptr) return hipErrorInvalidValue;
    if (mask_words == 0u || vocab_size == 0u || candidate_capacity == 0u)
        return hipErrorInvalidValue;
    constraint_mask_to_candidates_kernel<<<rows, kConstraintCandidateThreads, 0, stream>>>(
        masks, mask_words, vocab_size, candidate_capacity, candidate_ids);
    return hipGetLastError();
}

}
