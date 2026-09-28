#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/linear/config.h>
#include <cstdint>
#include <optional>

namespace ps::qwen35::runtime {

enum class LinearComputeFamily : uint8_t {
    Psq4W4A8,
    Psq8W8A8,
    Bf16,
    Fp8W8A8,
    Mxfp4W4A8,
};

struct Bf16GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    bool weight_aligned16 = false;
    bool input_aligned16 = false;
    bool verify_exact = false;
    bool allow_exact_rows = true;
};

struct Psq4GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

struct Psq8GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

struct Fp8Block128GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

struct Mxfp4GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

bool linear_shape_supported(
    LinearComputeFamily family,
    uint32_t out_features,
    uint32_t k);

std::optional<ps::kernel::Bf16GemmConfig>
select_bf16_gemm_config(const Bf16GemmSelectorInput& input);

std::optional<ps::kernel::Psq4GemmConfig>
select_psq4_gemm_config(const Psq4GemmSelectorInput& input);

std::optional<ps::kernel::Psq8GemmConfig>
select_psq8_gemm_config(const Psq8GemmSelectorInput& input);

std::optional<ps::kernel::Fp8Block128GemmConfig>
select_fp8_block128_gemm_config(const Fp8Block128GemmSelectorInput& input);

std::optional<ps::kernel::Mxfp4GemmConfig>
select_mxfp4_gemm_config(const Mxfp4GemmSelectorInput& input);

}  // namespace ps::qwen35::runtime
