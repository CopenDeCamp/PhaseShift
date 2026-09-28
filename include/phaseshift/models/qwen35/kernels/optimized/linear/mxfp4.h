#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/linear/config.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

enum class Mxfp4GemmOutputDType : uint8_t {
    BF16,
    F32,
};

// MXFP4 W4A8 WMMA GEMM. Weight E2M1 nibbles are consumed in the gfx1201 FP8
// WMMA fragment order (16 output x 32 K chunk, half split) and expanded to
// exact E4M3 operands in registers; the per-K32 E8M0 scale is applied after
// each 16x16x16 FP8 WMMA. Activations are E4M3 with per-row FP32 scales in the
// shared fragment layout.
hipError_t launch_gemm_mxfp4_w4a8_wmma(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

}  // namespace ps::kernel
