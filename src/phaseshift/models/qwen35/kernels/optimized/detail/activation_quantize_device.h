#pragma once
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/quantization/fpx/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace ps::kernel::detail {

constexpr uint32_t kQuantThreadsPerRow = 256u;
constexpr uint32_t kQuantVecLanes = 8u;
constexpr uint32_t kQuantVecChunk = kQuantThreadsPerRow * kQuantVecLanes;

__device__ __forceinline__ float quant_vec_bf16_lane_f32(uint32_t word, uint32_t e) {
    return (e & 1u) == 0u ? __uint_as_float(word << 16)
                          : __uint_as_float(word & 0xffff0000u);
}

template <uint32_t K, uint32_t KP, uint32_t Threads>
struct QuantVecRow {
    static_assert(Threads % 32u == 0u, "quant row requires whole waves");
    static_assert(K % kQuantVecLanes == 0u, "vector path requires K aligned to lanes");
    static constexpr uint32_t kNChunk =
        (KP + Threads * kQuantVecLanes - 1u) / (Threads * kQuantVecLanes);
    static constexpr uint32_t kChunkElems = Threads * kQuantVecLanes;

    __device__ __forceinline__ static void load(
        const ::ps::bf16_t* __restrict__ xrow, uint32_t (&packed)[kNChunk][4]) {
        const uint32_t t = threadIdx.x;
#pragma unroll
        for (uint32_t it = 0u; it < kNChunk; ++it) {
            const uint32_t i0 = it * kChunkElems + t * kQuantVecLanes;
            uint32_t w0 = 0u;
            uint32_t w1 = 0u;
            uint32_t w2 = 0u;
            uint32_t w3 = 0u;
            if (i0 < K) {
                const uint4 raw = *reinterpret_cast<const uint4*>(xrow + i0);
                w0 = raw.x;
                w1 = raw.y;
                w2 = raw.z;
                w3 = raw.w;
            }
            packed[it][0] = w0;
            packed[it][1] = w1;
            packed[it][2] = w2;
            packed[it][3] = w3;
        }
    }

    __device__ __forceinline__ static float peak(const uint32_t (&packed)[kNChunk][4]) {
        const uint32_t t = threadIdx.x;
        float peak_v = 0.0f;
#pragma unroll
        for (uint32_t it = 0u; it < kNChunk; ++it) {
            const uint32_t i0 = it * kChunkElems + t * kQuantVecLanes;
            if (i0 < K) {
#pragma unroll
                for (uint32_t e = 0u; e < kQuantVecLanes; ++e) {
                    const float v = quant_vec_bf16_lane_f32(packed[it][e >> 1u], e);
                    if (isfinite(v)) {
                        const float av = std::fabs(v);
                        if (av > peak_v) peak_v = av;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            peak_v = fmaxf(peak_v, __shfl_xor_sync(0xffffffffULL, peak_v, off));
        return peak_v;
    }

    __device__ __forceinline__ static void publish_scale(
        float peak_v, float* __restrict__ scales, uint32_t m, uint32_t scale_row_stride,
        float* inv_out, bool* has_out) {
        const uint32_t t = threadIdx.x;
        __shared__ float wave_max[Threads / 32u];
        __shared__ float s_scale;
        __shared__ float s_inv_scale;
        if ((t & 31u) == 0u) wave_max[t >> 5u] = peak_v;
        __syncthreads();
        if (t < 32u) {
            float v = (t < Threads / 32u) ? wave_max[t] : 0.0f;
#pragma unroll
            for (uint32_t off = Threads / 64u; off > 0u; off >>= 1u)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffULL, v, off));
            if (t == 0u) {
                const float scale = v > 0.0f ? v / 448.0f : 0.0f;
                const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
                s_scale = scale;
                s_inv_scale = inv;
                scales[static_cast<size_t>(m) * (scale_row_stride / 4u)] = scale;
            }
        }
        __syncthreads();
        *inv_out = s_inv_scale;
        *has_out = s_scale > 0.0f;
    }

    __device__ __forceinline__ static void encode_store(
        const uint32_t (&packed)[kNChunk][4], uint8_t* __restrict__ codes, uint32_t m,
        uint32_t code_row_stride, float inv_scale, bool has_scale) {
        const uint32_t t = threadIdx.x;
        uint8_t* crow = codes +
            static_cast<size_t>(m >> 4) * (static_cast<size_t>(code_row_stride) * 16u) +
            static_cast<size_t>(m & 15u) * 16u;
#pragma unroll
        for (uint32_t it = 0u; it < kNChunk; ++it) {
            const uint32_t i0 = it * kChunkElems + t * kQuantVecLanes;
            if (i0 < KP) {
                uint64_t out = 0u;
                if (i0 < K) {
#pragma unroll
                    for (uint32_t e = 0u; e < kQuantVecLanes; ++e) {
                        float v = quant_vec_bf16_lane_f32(packed[it][e >> 1u], e);
                        v = isfinite(v) ? v : 0.0f;
                        const uint8_t c =
                            ::ps::quantization::fpx::device::e4m3_encode_u8(v * inv_scale);
                        out |= static_cast<uint64_t>(c) << (8u * e);
                    }
                }
                if (!has_scale) out = 0u;
                const uint32_t off =
                    (i0 >> 5u) * 512u + ((i0 & 31u) >> 4u) * 256u + (i0 & 15u);
                *reinterpret_cast<uint64_t*>(crow + off) = out;
            }
        }
    }
};

template <uint32_t K, uint32_t KP, uint32_t Threads>
__device__ __forceinline__ void activation_quantize_e4m3_vec_row_body_from_packed(
    const uint32_t (&packed)[QuantVecRow<K, KP, Threads>::kNChunk][4],
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    uint32_t m,
    uint32_t code_row_stride,
    uint32_t scale_row_stride) {
    using Row = QuantVecRow<K, KP, Threads>;
    const float peak_v = Row::peak(packed);
    float inv_scale = 0.0f;
    bool has_scale = false;
    Row::publish_scale(peak_v, scales, m, scale_row_stride, &inv_scale, &has_scale);
    Row::encode_store(packed, codes, m, code_row_stride, inv_scale, has_scale);
}

template <uint32_t K, uint32_t KP>
__device__ __forceinline__ void activation_quantize_e4m3_vec_row_body(
    const ::ps::bf16_t* __restrict__ input,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    uint32_t m,
    uint32_t input_row_stride,
    uint32_t code_row_stride,
    uint32_t scale_row_stride) {
    using Row = QuantVecRow<K, KP, kQuantThreadsPerRow>;
    uint32_t packed[Row::kNChunk][4];
    Row::load(input + static_cast<size_t>(m) * input_row_stride, packed);
    activation_quantize_e4m3_vec_row_body_from_packed<K, KP, kQuantThreadsPerRow>(
        packed, codes, scales, m, code_row_stride, scale_row_stride);
}

}  // namespace ps::kernel::detail
