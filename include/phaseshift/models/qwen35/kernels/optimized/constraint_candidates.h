#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kConstraintCandidateThreads = 128u;

hipError_t launch_constraint_mask_to_candidates(
    const uint32_t* masks,
    uint32_t rows,
    uint32_t mask_words,
    uint32_t vocab_size,
    uint32_t candidate_capacity,
    int32_t* candidate_ids,
    hipStream_t stream);

}
