#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/linear/config.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {

enum class Psq4GemmOutputDType : uint8_t {
    BF16,
    F32,
};

inline constexpr const char* kPsq4Decode1Bf16U16Symbol =
    "phaseshift_qwen35_psq4_decode1_bf16_u16";
inline constexpr const char* kPsq4Decode1Bf16U8Symbol =
    "phaseshift_qwen35_psq4_decode1_bf16_u8";
inline constexpr const char* kPsq4RowBlock1Bf16Symbol =
    "phaseshift_qwen35_psq4_rowblock1_bf16";
inline constexpr const char* kPsq4RowBlock2Bf16Symbol =
    "phaseshift_qwen35_psq4_rowblock2_bf16";
inline constexpr const char* kPsq4RowBlock4Bf16Symbol =
    "phaseshift_qwen35_psq4_rowblock4_bf16";
inline constexpr const char* kPsq4RowBlock8Bf16Symbol =
    "phaseshift_qwen35_psq4_rowblock8_bf16";

struct Psq4MultiRowBf16Args {
    const void* weight_codes = nullptr;
    const void* weight_scales = nullptr;
    const void* activation_codes = nullptr;
    const void* activation_scales = nullptr;
    void* output = nullptr;
    uint32_t output_dtype = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
    uint32_t activation_code_stride = 0;
    uint32_t activation_scale_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(Psq4MultiRowBf16Args) == 72);
static_assert(alignof(Psq4MultiRowBf16Args) == 8);
static_assert(offsetof(Psq4MultiRowBf16Args, weight_codes) == 0);
static_assert(offsetof(Psq4MultiRowBf16Args, weight_scales) == 8);
static_assert(offsetof(Psq4MultiRowBf16Args, activation_codes) == 16);
static_assert(offsetof(Psq4MultiRowBf16Args, activation_scales) == 24);
static_assert(offsetof(Psq4MultiRowBf16Args, output) == 32);
static_assert(offsetof(Psq4MultiRowBf16Args, output_dtype) == 40);
static_assert(offsetof(Psq4MultiRowBf16Args, rows) == 44);
static_assert(offsetof(Psq4MultiRowBf16Args, out_features) == 48);
static_assert(offsetof(Psq4MultiRowBf16Args, k_padded) == 52);
static_assert(offsetof(Psq4MultiRowBf16Args, weight_scale_stride) == 56);
static_assert(offsetof(Psq4MultiRowBf16Args, activation_code_stride) == 60);
static_assert(offsetof(Psq4MultiRowBf16Args, activation_scale_stride) == 64);
static_assert(offsetof(Psq4MultiRowBf16Args, output_row_stride) == 68);

struct Psq4Decode1Bf16Args {
    const void* weight_codes = nullptr;
    const void* weight_scales = nullptr;
    const void* activation_codes = nullptr;
    const void* activation_scales = nullptr;
    void* output = nullptr;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
};

static_assert(sizeof(Psq4Decode1Bf16Args) == 48);
static_assert(alignof(Psq4Decode1Bf16Args) == 8);
static_assert(offsetof(Psq4Decode1Bf16Args, weight_codes) == 0);
static_assert(offsetof(Psq4Decode1Bf16Args, weight_scales) == 8);
static_assert(offsetof(Psq4Decode1Bf16Args, activation_codes) == 16);
static_assert(offsetof(Psq4Decode1Bf16Args, activation_scales) == 24);
static_assert(offsetof(Psq4Decode1Bf16Args, output) == 32);
static_assert(offsetof(Psq4Decode1Bf16Args, k_padded) == 40);
static_assert(offsetof(Psq4Decode1Bf16Args, weight_scale_stride) == 44);

uint32_t psq4_decode1_block_size() noexcept;
uint32_t psq4_decode1_grid(uint32_t out_features) noexcept;

__device__ __forceinline__ uint2 psq4_cb10_expand(uint32_t q4) {
    constexpr uint32_t kMags0 = 0x44403800u;
    constexpr uint32_t kMags1 = 0x52504C48u;
    const uint32_t q_odd = q4 >> 4u;
    const uint32_t mag_even =
        __builtin_amdgcn_perm(kMags1, kMags0, q4 & 0x07070707u);
    const uint32_t mag_odd =
        __builtin_amdgcn_perm(kMags1, kMags0, q_odd & 0x07070707u);
    const uint32_t sign_even = (q4 & 0x08080808u) << 4u;
    const uint32_t sign_odd = q4 & 0x80808080u;
    return make_uint2(mag_even | sign_even, mag_odd | sign_odd);
}

hipError_t launch_gemm_psq4_w4a8_wmma(
    const Psq4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales_bf16,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

bool psq4_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded);

uint32_t psq4_decode1_unroll(uint32_t out_features, uint32_t k_padded);

hipError_t launch_gemm_psq4_w4a8_wmma_decode1(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    uint32_t unroll,
    hipStream_t stream);

hipError_t launch_gemm_psq4_w4a8_wmma_auto(
    const Psq4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

}  // namespace ps::kernel
