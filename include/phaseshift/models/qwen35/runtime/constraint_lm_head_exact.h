#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::qwen35::runtime {

struct ConstraintLmHeadWeightView {
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
    uint32_t vocab = 0;
    uint32_t cols = 0;
    uint32_t k_padded = 0;
    uint32_t scale_stride_bytes = 0;
};

inline constexpr uint32_t kConstraintLmHeadExactMaxAllowed = 128u;

class ConstraintLmHeadExact {
public:
    ConstraintLmHeadExact() = default;
    ~ConstraintLmHeadExact() noexcept;
    ConstraintLmHeadExact(const ConstraintLmHeadExact&) = delete;
    ConstraintLmHeadExact& operator=(const ConstraintLmHeadExact&) = delete;
    ConstraintLmHeadExact(ConstraintLmHeadExact&& other) noexcept;
    ConstraintLmHeadExact& operator=(ConstraintLmHeadExact&& other) noexcept;

    Status init(const ConstraintLmHeadWeightView& weight, uint32_t max_rows, hipStream_t stream);

    Status select_exact(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                        const uint32_t* masks, uint32_t mask_words,
                        uint32_t candidate_capacity, int32_t* out_ids, hipStream_t stream);

    Status shutdown() noexcept;

    bool valid() const noexcept { return act_codes_ != nullptr; }
    uint32_t max_rows() const noexcept { return max_rows_; }
    uint32_t candidate_capacity() const noexcept { return candidate_capacity_; }

private:
    void move_from(ConstraintLmHeadExact& other) noexcept;

    const uint8_t* weight_codes_ = nullptr;
    const uint8_t* weight_scales_ = nullptr;
    uint32_t vocab_ = 0;
    uint32_t cols_ = 0;
    uint32_t k_padded_ = 0;
    uint32_t scale_stride_ = 0;
    uint32_t max_rows_ = 0;
    uint32_t candidate_capacity_ = 0;
    uint32_t act_code_stride_ = 0;
    uint32_t act_scale_stride_ = 0;

    uint8_t* act_codes_ = nullptr;
    float* act_scales_ = nullptr;
    int32_t* cand_ids_ = nullptr;
    float* rerank_ = nullptr;
};

bool constraint_lm_head_exact_enabled();

inline uint32_t select_constraint_lm_head_exact_capacity(
    uint32_t stochastic_outputs, uint32_t sampled_outputs, uint32_t outputs,
    const uint32_t* allowed_counts, uint32_t rows) {
    if (allowed_counts == nullptr || rows == 0u)
        return 0u;
    if (stochastic_outputs != 0u)
        return 0u;
    if (outputs == 0u || sampled_outputs != outputs)
        return 0u;
    uint32_t max_count = 0u;
    for (uint32_t i = 0u; i < rows; ++i) {
        const uint32_t count = allowed_counts[i];
        if (count == UINT32_MAX || count == 0u)
            return 0u;
        if (count > max_count)
            max_count = count;
    }
    if (max_count <= kConstraintLmHeadExactMaxAllowed)
        return max_count;
    return 0u;
}

}  // namespace ps::qwen35::runtime
