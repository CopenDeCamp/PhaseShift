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
inline constexpr const char* kPsq8RowBlock1Bf16Symbol =
    "phaseshift_qwen35_psq8_rowblock1_bf16";
inline constexpr const char* kPsq8RowBlock2Bf16Symbol =
    "phaseshift_qwen35_psq8_rowblock2_bf16";
inline constexpr const char* kPsq8RowBlock4Bf16Symbol =
    "phaseshift_qwen35_psq8_rowblock4_bf16";
inline constexpr const char* kPsq8RowBlock8Bf16Symbol =
    "phaseshift_qwen35_psq8_rowblock8_bf16";
inline constexpr const char* kPsq8Prefill2DN64K64Symbol =
    "phaseshift_qwen35_psq8_prefill2d_n64k64";
inline constexpr const char* kPsq8Prefill2DN128K64Symbol =
    "phaseshift_qwen35_psq8_prefill2d_n128k64";
inline constexpr const char* kPsq8Prefill2DN128K128Symbol =
    "phaseshift_qwen35_psq8_prefill2d_n128k128";

struct Psq8MultiRowBf16Args {
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

static_assert(sizeof(Psq8MultiRowBf16Args) == 72);
static_assert(alignof(Psq8MultiRowBf16Args) == 8);
static_assert(offsetof(Psq8MultiRowBf16Args, weight_codes) == 0);
static_assert(offsetof(Psq8MultiRowBf16Args, weight_scales) == 8);
static_assert(offsetof(Psq8MultiRowBf16Args, activation_codes) == 16);
static_assert(offsetof(Psq8MultiRowBf16Args, activation_scales) == 24);
static_assert(offsetof(Psq8MultiRowBf16Args, output) == 32);
static_assert(offsetof(Psq8MultiRowBf16Args, output_dtype) == 40);
static_assert(offsetof(Psq8MultiRowBf16Args, rows) == 44);
static_assert(offsetof(Psq8MultiRowBf16Args, out_features) == 48);
static_assert(offsetof(Psq8MultiRowBf16Args, k_padded) == 52);
static_assert(offsetof(Psq8MultiRowBf16Args, weight_scale_stride) == 56);
static_assert(offsetof(Psq8MultiRowBf16Args, activation_code_stride) == 60);
static_assert(offsetof(Psq8MultiRowBf16Args, activation_scale_stride) == 64);
static_assert(offsetof(Psq8MultiRowBf16Args, output_row_stride) == 68);

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

inline constexpr uint32_t kPsq8RowBlockOutTile = 16u;

constexpr uint32_t psq8_rowblock_grid_x(uint32_t out_features) noexcept {
    return (out_features + kPsq8RowBlockOutTile - 1u) / kPsq8RowBlockOutTile;
}

constexpr uint32_t psq8_rowblock_grid_y(uint32_t rows,
                                        uint32_t rowblock_tiles) noexcept {
    const uint32_t rows_per_block = rowblock_tiles * kPsqGemmRowTile;
    return (rows + rows_per_block - 1u) / rows_per_block;
}

inline constexpr uint32_t kPsq8Prefill2DRowsPerBlock =
    kPsqGemmPrefill2dRowsPerBlock;
inline constexpr uint32_t kPsq8Prefill2DThreads = 256u;

constexpr uint32_t psq8_prefill2d_grid_x(uint32_t out_features,
                                         uint32_t out_block) noexcept {
    return out_features / out_block;
}

constexpr uint32_t psq8_prefill2d_grid_y(uint32_t rows) noexcept {
    return rows / kPsq8Prefill2DRowsPerBlock;
}

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
