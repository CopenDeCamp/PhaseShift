#pragma once
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/execution/execution_types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::qwen35::runtime {

struct LmHeadProxyWeightView {
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
    uint32_t vocab = 0;
    uint32_t cols = 0;
    uint32_t k_padded = 0;
    uint32_t scale_stride_bytes = 0;
};

class LmHeadCandidateProxy {
public:
    LmHeadCandidateProxy() = default;
    ~LmHeadCandidateProxy() noexcept;
    LmHeadCandidateProxy(const LmHeadCandidateProxy&) = delete;
    LmHeadCandidateProxy& operator=(const LmHeadCandidateProxy&) = delete;
    LmHeadCandidateProxy(LmHeadCandidateProxy&& other) noexcept;
    LmHeadCandidateProxy& operator=(LmHeadCandidateProxy&& other) noexcept;

    Status init(const LmHeadProxyWeightView& weight, uint32_t max_rows, uint32_t pool,
                hipStream_t stream);
    Status select(const bf16_t* normed, uint32_t rows, uint32_t row_stride, int32_t* out_ids,
                  hipStream_t stream);
    Status select_shadow(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                         hipStream_t stream);
    Status select_constrained_exact(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                                    const uint32_t* masks, uint32_t mask_words,
                                    uint32_t candidate_capacity, int32_t* out_ids,
                                    hipStream_t stream);
    Status select_constrained_exact_shadow(const bf16_t* normed, uint32_t rows,
                                           uint32_t row_stride, const uint32_t* masks,
                                           uint32_t mask_words, uint32_t candidate_capacity,
                                           hipStream_t stream);
    Status select_constrained_masked(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                                     const uint32_t* masks, uint32_t mask_words,
                                     int32_t* out_ids, hipStream_t stream);
    Status select_constrained_masked_shadow(const bf16_t* normed, uint32_t rows,
                                            uint32_t row_stride, const uint32_t* masks,
                                            uint32_t mask_words, hipStream_t stream);
    Status compare_shadow(const int32_t* full_ids, uint32_t rows, hipStream_t stream);
    Status shutdown() noexcept;

    bool valid() const noexcept { return int2_codes_ != nullptr; }
    uint32_t max_rows() const noexcept { return max_rows_; }
    uint32_t candidate_capacity() const noexcept { return candidate_capacity_; }
    uint32_t coarse_pool() const noexcept { return pool_; }

private:
    void move_from(LmHeadCandidateProxy& other) noexcept;
    Status run_select(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                      int32_t* out_ids, hipStream_t stream);
    Status run_select_constrained_exact(const bf16_t* normed, uint32_t rows,
                                        uint32_t row_stride, const uint32_t* masks,
                                        uint32_t mask_words, uint32_t candidate_capacity,
                                        int32_t* out_ids, hipStream_t stream);
    Status run_select_constrained_masked(const bf16_t* normed, uint32_t rows,
                                         uint32_t row_stride, const uint32_t* masks,
                                         uint32_t mask_words, int32_t* out_ids,
                                         hipStream_t stream);

    const uint8_t* weight_codes_ = nullptr;
    const uint8_t* weight_scales_ = nullptr;
    uint32_t vocab_ = 0;
    uint32_t cols_ = 0;
    uint32_t k_padded_ = 0;
    uint32_t scale_stride_ = 0;
    uint32_t max_rows_ = 0;
    uint32_t pool_ = 0;
    uint32_t candidate_capacity_ = 0;
    uint32_t partitions_ = 0;
    uint32_t radix_partitions_ = 0;
    std::size_t radix_scratch_bytes_ = 0;
    uint32_t scratch_stride_ = 0;
    uint32_t act_code_stride_ = 0;
    uint32_t act_scale_stride_ = 0;

    uint8_t* int2_codes_ = nullptr;
    uint8_t* tables_ = nullptr;
    uint8_t* act_codes_ = nullptr;
    float* act_scales_ = nullptr;
    float* coarse_ = nullptr;
    int32_t* pool_ids_ = nullptr;
    float* pool_logits_ = nullptr;
    int32_t* cand_ids_ = nullptr;
    float* rerank_ = nullptr;
    int32_t* scratch_ids_ = nullptr;
    float* scratch_values_ = nullptr;
    uint32_t* radix_scratch_ = nullptr;
    int32_t* shadow_ids_ = nullptr;
    uint64_t* shadow_counters_ = nullptr;
};

uint32_t target_lm_head_proxy_mode();
uint32_t target_lm_head_proxy_decode_mode();
uint32_t target_lm_head_proxy_pool();
uint32_t target_lm_head_proxy_constraint_threshold();

enum class LmHeadProxyPath : uint8_t {
    None = 0,
    Fast = 1,
    Shadow = 2,
};

inline LmHeadProxyPath lm_head_proxy_path(::ps::runtime::ExecutionRole role,
                                          uint32_t stochastic_outputs, bool constrained,
                                          uint32_t mode) {
    const bool verify_path = role == ::ps::runtime::ExecutionRole::Verify;
    const bool decode_path = role == ::ps::runtime::ExecutionRole::Decode &&
                             stochastic_outputs == 0u;
    if (!verify_path && !decode_path)
        return LmHeadProxyPath::None;
    if (constrained || mode == 0u)
        return LmHeadProxyPath::None;
    if (verify_path)
        return mode == 2u ? LmHeadProxyPath::None : LmHeadProxyPath::Fast;
    return mode == 2u ? LmHeadProxyPath::Shadow : LmHeadProxyPath::Fast;
}

enum class LmHeadConstrainedPath : uint8_t {
    None = 0,
    ExactCandidates = 1,
    MaskedCoarse = 2,
};

struct LmHeadConstrainedSelection {
    LmHeadConstrainedPath path = LmHeadConstrainedPath::None;
    uint32_t candidate_capacity = 0u;
};

inline LmHeadConstrainedSelection select_lm_head_constrained(
    uint32_t stochastic_outputs, uint32_t sampled_outputs, uint32_t outputs,
    const uint32_t* allowed_counts, uint32_t rows, uint32_t small_threshold,
    uint32_t coarse_pool) {
    LmHeadConstrainedSelection selection;
    if (allowed_counts == nullptr || rows == 0u || small_threshold == 0u)
        return selection;
    if (stochastic_outputs != 0u)
        return selection;
    if (outputs == 0u || sampled_outputs != outputs)
        return selection;
    uint32_t min_count = UINT32_MAX;
    uint32_t max_count = 0u;
    for (uint32_t i = 0u; i < rows; ++i) {
        const uint32_t count = allowed_counts[i];
        if (count == UINT32_MAX || count == 0u)
            return selection;
        if (count < min_count)
            min_count = count;
        if (count > max_count)
            max_count = count;
    }
    if (max_count <= small_threshold) {
        selection.path = LmHeadConstrainedPath::ExactCandidates;
        selection.candidate_capacity = max_count;
        return selection;
    }
    if (min_count > small_threshold && coarse_pool != 0u && min_count >= coarse_pool) {
        selection.path = LmHeadConstrainedPath::MaskedCoarse;
    }
    return selection;
}

}  // namespace ps::qwen35::runtime
