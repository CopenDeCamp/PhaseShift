#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

hipError_t launch_dflash2_psq8_candidate_rerank(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    const int32_t* candidate_ids,
    float* logits,
    uint32_t rows,
    uint32_t pool,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t logits_row_stride,
    hipStream_t stream);

}  // namespace ps::kernel
