#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel::detail {

using rerank_int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using rerank_float8_t = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ size_t psq8_preshuffle_row_off(uint32_t r, uint32_t code_stride) {
    return static_cast<size_t>(r >> 4) * (static_cast<size_t>(code_stride) * 16u) +
           static_cast<size_t>(r & 15u) * 16u;
}

__device__ __forceinline__ float psq8_rerank_tile_body(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    uint32_t token_id,
    bool cand_valid,
    uint32_t m_lane,
    uint32_t row,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t weight_scale_stride,
    uint32_t activation_code_stride,
    uint32_t k_group) {
    const uint32_t nb = k_padded >> 5u;
    const bool row_valid = m_lane < rows;
    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(token_id >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(token_id & 15u) * 16u + k_group * 8u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(
        weight_scales + static_cast<size_t>(token_id >> 4) * weight_scale_stride +
        static_cast<size_t>(token_id & 15u) * 2u);
    const uint8_t* arow =
        activation_codes + psq8_preshuffle_row_off(row, activation_code_stride) + k_group * 8u;

    rerank_float8_t acc{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t ib = 0u; ib < nb; ++ib) {
        rerank_int32x2_t wv0 = {};
        rerank_int32x2_t wv1 = {};
        float ws = 0.0f;
        if (cand_valid) {
            const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
            wv0 = *reinterpret_cast<const rerank_int32x2_t*>(wp);
            wv1 = *reinterpret_cast<const rerank_int32x2_t*>(wp + 256u);
            ws = __bfloat162float(__ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
        }
        rerank_int32x2_t a0 = {};
        rerank_int32x2_t a1 = {};
        if (row_valid && cand_valid) {
            const uint8_t* ap = arow + static_cast<size_t>(ib) * 512u;
            a0 = *reinterpret_cast<const rerank_int32x2_t*>(ap);
            a1 = *reinterpret_cast<const rerank_int32x2_t*>(ap + 256u);
        }
        rerank_float8_t c{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, wv0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, wv1, c);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] = __builtin_fmaf(c[j], ws, acc[j]);
    }
    return acc[0];
}

}  // namespace ps::kernel::detail
