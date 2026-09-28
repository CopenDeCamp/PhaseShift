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
    Status compare_shadow(const int32_t* full_ids, uint32_t rows, hipStream_t stream);
    Status shutdown() noexcept;

    bool valid() const noexcept { return int2_codes_ != nullptr; }
    uint32_t max_rows() const noexcept { return max_rows_; }

private:
    void move_from(LmHeadCandidateProxy& other) noexcept;
    Status run_select(const bf16_t* normed, uint32_t rows, uint32_t row_stride,
                      int32_t* out_ids, hipStream_t stream);

    const uint8_t* weight_codes_ = nullptr;
    const uint8_t* weight_scales_ = nullptr;
    uint32_t vocab_ = 0;
    uint32_t cols_ = 0;
    uint32_t k_padded_ = 0;
    uint32_t scale_stride_ = 0;
    uint32_t max_rows_ = 0;
    uint32_t pool_ = 0;
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

}  // namespace ps::qwen35::runtime
