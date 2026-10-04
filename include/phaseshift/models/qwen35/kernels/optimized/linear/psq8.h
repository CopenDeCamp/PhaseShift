#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/linear/config.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {

enum class Psq8GemmOutputDType : uint8_t {
    BF16,
    F32,
};

inline constexpr const char* kPsq8Decode1Bf16U8Symbol =
    "phaseshift_qwen35_psq8_decode1_bf16_u8";

struct Psq8Decode1Bf16Args {
    const void* weight_codes = nullptr;
    const void* weight_scales = nullptr;
    const void* activation_codes = nullptr;
    const void* activation_scales = nullptr;
    void* output = nullptr;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
};

static_assert(sizeof(Psq8Decode1Bf16Args) == 48);
static_assert(alignof(Psq8Decode1Bf16Args) == 8);
static_assert(offsetof(Psq8Decode1Bf16Args, weight_codes) == 0);
static_assert(offsetof(Psq8Decode1Bf16Args, weight_scales) == 8);
static_assert(offsetof(Psq8Decode1Bf16Args, activation_codes) == 16);
static_assert(offsetof(Psq8Decode1Bf16Args, activation_scales) == 24);
static_assert(offsetof(Psq8Decode1Bf16Args, output) == 32);
static_assert(offsetof(Psq8Decode1Bf16Args, k_padded) == 40);
static_assert(offsetof(Psq8Decode1Bf16Args, weight_scale_stride) == 44);

uint32_t psq8_decode1_block_size() noexcept;
uint32_t psq8_decode1_grid(uint32_t out_features) noexcept;

hipError_t launch_gemm_psq8_w8a8_wmma(
    const Psq8GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq8GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

bool psq8_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded);

uint32_t psq8_decode1_unroll(uint32_t out_features, uint32_t k_padded);

hipError_t launch_gemm_psq8_w8a8_wmma_decode1(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq8GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    uint32_t unroll,
    hipStream_t stream);

hipError_t launch_gemm_psq8_w8a8_wmma_auto(
    const Psq8GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq8GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    bool* decode1_used = nullptr);

}  // namespace ps::kernel
