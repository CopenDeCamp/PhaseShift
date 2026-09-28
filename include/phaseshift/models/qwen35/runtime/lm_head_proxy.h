#pragma once
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
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
    Status shutdown() noexcept;

    bool valid() const noexcept { return int2_codes_ != nullptr; }
    uint32_t max_rows() const noexcept { return max_rows_; }

private:
    void move_from(LmHeadCandidateProxy& other) noexcept;

    const uint8_t* weight_codes_ = nullptr;
    const uint8_t* weight_scales_ = nullptr;
    uint32_t vocab_ = 0;
    uint32_t cols_ = 0;
    uint32_t k_padded_ = 0;
    uint32_t scale_stride_ = 0;
    uint32_t max_rows_ = 0;
    uint32_t pool_ = 0;
    uint32_t partitions_ = 0;
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
};

uint32_t target_lm_head_proxy_mode();
uint32_t target_lm_head_proxy_pool();

}  // namespace ps::qwen35::runtime
