#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/linear/config.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kBf16ExactRowsSymbol =
    "phaseshift_qwen35_gemm_bf16_exact_rows_1";

inline constexpr const char* kBf16ExactRowsSymbols[16] = {
    "phaseshift_qwen35_gemm_bf16_exact_rows_1",
    "phaseshift_qwen35_gemm_bf16_exact_rows_2",
    "phaseshift_qwen35_gemm_bf16_exact_rows_3",
    "phaseshift_qwen35_gemm_bf16_exact_rows_4",
    "phaseshift_qwen35_gemm_bf16_exact_rows_5",
    "phaseshift_qwen35_gemm_bf16_exact_rows_6",
    "phaseshift_qwen35_gemm_bf16_exact_rows_7",
    "phaseshift_qwen35_gemm_bf16_exact_rows_8",
    "phaseshift_qwen35_gemm_bf16_exact_rows_9",
    "phaseshift_qwen35_gemm_bf16_exact_rows_10",
    "phaseshift_qwen35_gemm_bf16_exact_rows_11",
    "phaseshift_qwen35_gemm_bf16_exact_rows_12",
    "phaseshift_qwen35_gemm_bf16_exact_rows_13",
    "phaseshift_qwen35_gemm_bf16_exact_rows_14",
    "phaseshift_qwen35_gemm_bf16_exact_rows_15",
    "phaseshift_qwen35_gemm_bf16_exact_rows_16",
};

struct Bf16GemmExactRowsArgs {
    const void* weight = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t output_dtype = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(Bf16GemmExactRowsArgs) == 48);
static_assert(alignof(Bf16GemmExactRowsArgs) == 8);
static_assert(offsetof(Bf16GemmExactRowsArgs, weight) == 0);
static_assert(offsetof(Bf16GemmExactRowsArgs, input) == 8);
static_assert(offsetof(Bf16GemmExactRowsArgs, output) == 16);
static_assert(offsetof(Bf16GemmExactRowsArgs, output_dtype) == 24);
static_assert(offsetof(Bf16GemmExactRowsArgs, n) == 28);
static_assert(offsetof(Bf16GemmExactRowsArgs, k) == 32);
static_assert(offsetof(Bf16GemmExactRowsArgs, input_row_stride) == 36);
static_assert(offsetof(Bf16GemmExactRowsArgs, output_row_stride) == 40);

enum class Bf16GemmOutputDType : uint8_t {
    BF16,
    F32,
};

hipError_t launch_gemm_bf16(
    const Bf16GemmConfig& config,
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    hipStream_t stream);

}  // namespace ps::kernel
