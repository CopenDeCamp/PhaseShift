#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel::detail {

using coarse_int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using coarse_float8_t = float __attribute__((ext_vector_type(8)));

inline constexpr uint32_t kCoarseHeadTileOut = 16u;
inline constexpr uint32_t kCoarseHeadThreads = 32u;
inline constexpr uint32_t kCoarseHeadTileRow = 16u;

__device__ __forceinline__ size_t coarse_preshuffle_row_off(uint32_t r, uint32_t code_stride) {
    return static_cast<size_t>(r >> 4) * (static_cast<size_t>(code_stride) * 16u) +
           static_cast<size_t>(r & 15u) * 16u;
}

__device__ __forceinline__ void coarse_head_load_lut(
    uint32_t* __restrict__ lut, const uint32_t* __restrict__ expand_table) {
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x)
        lut[i] = expand_table[i];
}

__device__ __forceinline__ void coarse_head_tile_body(
    const uint8_t* __restrict__ int2_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    const uint32_t* __restrict__ lut,
    uint32_t lane,
    uint32_t out_start,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride,
    uint32_t activation_code_stride,
    uint32_t activation_scale_stride,
    float* out) {
    const uint32_t nb = k_padded >> 5u;
    const uint32_t tid = lane & 31u;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;
    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;
    const bool row_valid = m_lane < rows;

    const uint8_t* wrow = int2_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 4u) +
        static_cast<size_t>(o_self & 15u) * 4u + k_group * 2u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(
        weight_scales + static_cast<size_t>(o_self >> 4) * weight_scale_stride +
        static_cast<size_t>(o_self & 15u) * 2u);
    const uint8_t* arow =
        activation_codes + coarse_preshuffle_row_off(m_lane, activation_code_stride) +
        k_group * 8u;

    coarse_float8_t acc{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t ib = 0u; ib < nb; ++ib) {
        coarse_int32x2_t wv0 = {};
        coarse_int32x2_t wv1 = {};
        float ws = 0.0f;
        if (out_valid) {
            const uint8_t* wp = wrow + static_cast<size_t>(ib) * 128u;
            const uint16_t w0 = *reinterpret_cast<const uint16_t*>(wp);
            const uint16_t w1 = *reinterpret_cast<const uint16_t*>(wp + 64u);
            wv0 = coarse_int32x2_t{static_cast<int32_t>(lut[w0 & 0xFFu]),
                                   static_cast<int32_t>(lut[w0 >> 8u])};
            wv1 = coarse_int32x2_t{static_cast<int32_t>(lut[w1 & 0xFFu]),
                                   static_cast<int32_t>(lut[w1 >> 8u])};
            ws = __bfloat162float(__ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
        }
        coarse_int32x2_t a0 = {};
        coarse_int32x2_t a1 = {};
        if (row_valid && out_valid) {
            const uint8_t* ap = arow + static_cast<size_t>(ib) * 512u;
            a0 = *reinterpret_cast<const coarse_int32x2_t*>(ap);
            a1 = *reinterpret_cast<const coarse_int32x2_t*>(ap + 256u);
        }
        coarse_float8_t c{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, wv0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, wv1, c);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] = __builtin_fmaf(c[j], ws, acc[j]);
    }

    const float as_self =
        row_valid ? activation_scales[static_cast<size_t>(m_lane) *
                                      (activation_scale_stride / 4u)]
                  : 0.0f;
    float rs[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        rs[j] = __shfl(as_self, 8 * static_cast<int>(k_group) + j);
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        out[j] = acc[j] * rs[j];
}

}  // namespace ps::kernel::detail
