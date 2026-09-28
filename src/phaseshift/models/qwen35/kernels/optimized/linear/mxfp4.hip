#include <phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace ps::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = kMxfp4GemmRowTile;
constexpr uint32_t kThreadsPerBlock = 32u;

// E2M1 -> E4M3 expansion, same shape as the PSQ4 cb10 path: the native layout
// puts code k in the low nibble and code k+16 in the high nibble of byte k, so
// the even (low) and odd (high) nibbles are already one per byte and can be
// gathered with v_perm_b32 without an explicit byte spread. `q4` holds 4 bytes
// = 4 even + 4 odd codes; the result is {even 0..3, odd 0..3}.
__device__ __forceinline__ int32x2_t mxfp4_low_high(uint32_t q4) {
    constexpr uint32_t mags0 = 0x3C383000u;
    constexpr uint32_t mags1 = 0x4C484440u;
    constexpr uint32_t negs0 = 0xBCB8B080u;
    constexpr uint32_t negs1 = 0xCCC8C4C0u;
    const uint32_t q_even = q4;
    const uint32_t q_odd = q4 >> 4u;
    const uint32_t v_even_low = __builtin_amdgcn_perm(mags1, mags0, q_even & 0x07070707u);
    const uint32_t v_odd_low = __builtin_amdgcn_perm(mags1, mags0, q_odd & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(negs1, negs0, q_even & 0x07070707u);
    const uint32_t v_odd_high = __builtin_amdgcn_perm(negs1, negs0, q_odd & 0x07070707u);
    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1u);
    const uint32_t mask_odd = 0x03020100u | ((q_odd & 0x08080808u) >> 1u);
    return int32x2_t{
        static_cast<int32_t>(__builtin_amdgcn_perm(v_even_high, v_even_low, mask_even)),
        static_cast<int32_t>(__builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd))};
}

__device__ __forceinline__ float mxfp4_e8m0_to_f32(uint8_t b) {
    return (b == 0xFFu) ? 0.0f : ldexpf(1.0f, static_cast<int>(b) - 127);
}

__device__ __forceinline__ size_t mxfp4_act_off(uint32_t r, uint32_t code_stride) {
    return static_cast<size_t>(r >> 4) * (static_cast<size_t>(code_stride) * 16u) +
           static_cast<size_t>(r & 15u) * 16u;
}

template <int MB>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_mxfp4_w4a8_wmma_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t output_row_stride) {
    const uint32_t nb = k_padded >> 5;
    const uint32_t tid = threadIdx.x;

    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * (MB * kTileRow);

    if (out_start >= out_features || row_start >= rows)
        return;

    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(nb) * 256u) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint8_t* wsrow = weight_scales +
        static_cast<size_t>(o_self) * weight_scale_stride;

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        const uint32_t r = row_start + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + mxfp4_act_off(r, activation_code_stride) + k_group * 8u;
    }

    float8_t acc[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const bool full_tile = (out_start + kTileOut <= out_features);

    auto run_k_loop = [&]<bool kGuarded, int kUnroll>() {
        #pragma unroll kUnroll
        for (uint32_t ib = 0; ib < nb; ++ib) {
            int32x2_t w0 = {};
            int32x2_t w1 = {};
            float ws = 0.0f;
            if constexpr (kGuarded) {
                if (out_valid) {
                    const uint8_t* wp = wrow + static_cast<size_t>(ib) * 256u;
                    const uint64_t packed = *reinterpret_cast<const uint64_t*>(wp);
                    const int32x2_t lo = mxfp4_low_high(static_cast<uint32_t>(packed));
                    const int32x2_t hi = mxfp4_low_high(static_cast<uint32_t>(packed >> 32u));
                    w0 = int32x2_t{lo[0], hi[0]};
                    w1 = int32x2_t{lo[1], hi[1]};
                    ws = mxfp4_e8m0_to_f32(wsrow[ib]);
                }
            } else {
                const uint8_t* wp = wrow + static_cast<size_t>(ib) * 256u;
                const uint64_t packed = *reinterpret_cast<const uint64_t*>(wp);
                const int32x2_t lo = mxfp4_low_high(static_cast<uint32_t>(packed));
                const int32x2_t hi = mxfp4_low_high(static_cast<uint32_t>(packed >> 32u));
                w0 = int32x2_t{lo[0], hi[0]};
                w1 = int32x2_t{lo[1], hi[1]};
                ws = mxfp4_e8m0_to_f32(wsrow[ib]);
            }
            #pragma unroll
            for (int t = 0; t < MB; ++t) {
                int32x2_t a0 = {};
                int32x2_t a1 = {};
                if constexpr (kGuarded) {
                    const bool rv =
                        (row_start + static_cast<uint32_t>(t) * kTileRow + m_lane) < rows;
                    if (rv) {
                        const uint8_t* ap = arow[t] + static_cast<size_t>(ib) * 512u;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    }
                } else {
                    const uint8_t* ap = arow[t] + static_cast<size_t>(ib) * 512u;
                    a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                }
                float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
                c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[t][j] = __builtin_fmaf(c[j], ws, acc[t][j]);
            }
        }
    };
    constexpr int kUnroll = (MB >= 8) ? 2 : 4;
    if (full_tile)
        run_k_loop.template operator()<false, kUnroll>();
    else
        run_k_loop.template operator()<true, kUnroll>();

    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        const uint32_t r_self = row_start + static_cast<uint32_t>(t) * kTileRow + m_lane;
        const float as_self = (r_self < rows)
            ? activation_scales[static_cast<size_t>(r_self) * (activation_scale_stride / 4u)]
            : 0.0f;
        float rs[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            rs[j] = __shfl(as_self, 8 * static_cast<int>(k_group) + j);
        if (!out_valid)
            continue;
        const uint32_t r_base = row_start + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t r = r_base + static_cast<uint32_t>(j);
            if (r >= rows)
                continue;
            const float v = acc[t][j] * rs[j];
            const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
            if (output_dtype == static_cast<uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                static_cast<::ps::bf16_t*>(output)[out_idx] =
                    ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

constexpr uint32_t k2dThreads = 256u;
constexpr uint32_t k2dOutBlock = kMxfp4GemmPrefill2dOutBlock;

// Prefill path: one block computes a 256 (rows) x 64 (outputs) tile with 8
// waves. Each wave stages one 16-output x 64-K weight tile (already expanded
// to E4M3) into shared memory so the weight is read and expanded once.
template <bool kGuarded>
__global__ void __launch_bounds__(k2dThreads) gemm_mxfp4_w4a8_wmma_prefill_2d_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t output_row_stride) {
    const uint32_t nb = k_padded >> 5;
    const uint32_t nb2 = k_padded >> 6;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4;
    const uint32_t wave_m = wave & 3u;
    const uint32_t wave_n = wave >> 2;

    const uint32_t out_block_start = blockIdx.x * k2dOutBlock;
    const uint32_t row_start = blockIdx.y * 16u * kTileRow;
    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t row_base = row_start + wave_m * 4u * kTileRow;
    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + mxfp4_act_off(r, activation_code_stride) + k_group * 8u;
    }

    __shared__ int32x2_t w0_lds[4][2][32];
    __shared__ int32x2_t w1_lds[4][2][32];
    __shared__ float ws_lds[4][2][16];

    const uint32_t stage_nt = wave_n * 2u + (wave & 1u);
    const uint32_t stage_ib = (wave >> 1) & 1u;
    const uint32_t o_stage = out_block_start + stage_nt * kTileOut + m_lane;
    const bool stage_valid = !kGuarded || (o_stage < out_features);
    const uint8_t* w_stage = weight_codes +
        static_cast<size_t>(o_stage >> 4) * (static_cast<size_t>(nb) * 256u) +
        static_cast<size_t>(o_stage & 15u) * 16u + k_group * 8u;
    const uint8_t* ws_stage = weight_scales +
        static_cast<size_t>(o_stage) * weight_scale_stride;

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t ib2 = 0; ib2 < nb2; ++ib2) {
        if (stage_valid) {
            const uint32_t ib = ib2 * 2u + stage_ib;
            const uint8_t* wp = w_stage + static_cast<size_t>(ib) * 256u;
            const uint64_t packed = *reinterpret_cast<const uint64_t*>(wp);
            const int32x2_t lo = mxfp4_low_high(static_cast<uint32_t>(packed));
            const int32x2_t hi = mxfp4_low_high(static_cast<uint32_t>(packed >> 32u));
            w0_lds[stage_nt][stage_ib][lane] = int32x2_t{lo[0], hi[0]};
            w1_lds[stage_nt][stage_ib][lane] = int32x2_t{lo[1], hi[1]};
            ws_lds[stage_nt][stage_ib][m_lane] = mxfp4_e8m0_to_f32(ws_stage[ib]);
        }
        __syncthreads();
        #pragma unroll 2
        for (uint32_t ib_local = 0; ib_local < 2; ++ib_local) {
            int32x2_t w0[2];
            int32x2_t w1[2];
            float ws[2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t nt_glob = wave_n * 2u + static_cast<uint32_t>(nt);
                w0[nt] = w0_lds[nt_glob][ib_local][lane];
                w1[nt] = w1_lds[nt_glob][ib_local][lane];
                ws[nt] = ws_lds[nt_glob][ib_local][m_lane];
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int32x2_t a0 = {};
                int32x2_t a1 = {};
                if constexpr (kGuarded) {
                    if (row_base + static_cast<uint32_t>(t) * kTileRow + m_lane < rows) {
                        const uint8_t* ap =
                            arow[t] + static_cast<size_t>(ib2 * 2u + ib_local) * 512u;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    }
                } else {
                    const uint8_t* ap = arow[t] + static_cast<size_t>(ib2 * 2u + ib_local) * 512u;
                    a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                }
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0[nt], c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1[nt], c);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][nt][j] = __builtin_fmaf(c[j], ws[nt], acc[t][nt][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t r_self = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        const float as_self = (r_self < rows)
            ? activation_scales[static_cast<size_t>(r_self) * (activation_scale_stride / 4u)]
            : 0.0f;
        float rs[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            rs[j] = __shfl(as_self, 8 * static_cast<int>(k_group) + j);
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = out_block_start + wave_n * 2u * kTileOut +
                                    static_cast<uint32_t>(nt) * kTileOut + m_lane;
            if (kGuarded && o_self >= out_features)
                continue;
            const uint32_t r_base = row_base + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t r = r_base + static_cast<uint32_t>(j);
                if (r >= rows)
                    continue;
                const float v = acc[t][nt][j] * rs[j];
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
                if (output_dtype == static_cast<uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                    static_cast<::ps::bf16_t*>(output)[out_idx] =
                        ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}

#define MXFP4_MB_LAUNCH(MB)                                                     \
    do {                                                                        \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,              \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);  \
        gemm_mxfp4_w4a8_wmma_kernel<(MB)>                                       \
            <<<grid, kThreadsPerBlock, 0, stream>>>(                            \
                weight_codes, weight_scales, acodes, activation_scales, output, \
                static_cast<uint32_t>(output_dtype), rows, out_features,        \
                k_padded, weight_scale_stride_bytes,                            \
                activation_code_stride_bytes, activation_scale_stride_bytes,    \
                output_row_stride);                                             \
        return hipGetLastError();                                               \
    } while (0)

hipError_t launch_gemm_mxfp4_w4a8_wmma_impl(
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
    hipStream_t stream) {
    if (rows == 0 || out_features == 0 || k_padded == 0)
        return hipSuccess;
    if ((k_padded % 32u) != 0u)
        return hipErrorNotSupported;
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    (void)acodes;
    if (config.id == Mxfp4GemmConfigId::Prefill2D) {
        if ((rows % (16u * kTileRow)) != 0u ||
            (out_features % k2dOutBlock) != 0u ||
            (k_padded % kMxfp4GemmPrefill2dKChunk) != 0u)
            return hipErrorNotSupported;
        const dim3 grid(out_features / k2dOutBlock, rows / (16u * kTileRow), 1);
        gemm_mxfp4_w4a8_wmma_prefill_2d_kernel<false><<<grid, k2dThreads, 0, stream>>>(
            weight_codes, weight_scales, acodes, activation_scales, output,
            static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride);
        return hipGetLastError();
    }
    switch (config.id) {
    case Mxfp4GemmConfigId::RowBlock8:
        MXFP4_MB_LAUNCH(8);
    case Mxfp4GemmConfigId::RowBlock4:
        MXFP4_MB_LAUNCH(4);
    case Mxfp4GemmConfigId::RowBlock2:
        MXFP4_MB_LAUNCH(2);
    case Mxfp4GemmConfigId::RowBlock1:
    default:
        MXFP4_MB_LAUNCH(1);
    }
}

#undef MXFP4_MB_LAUNCH

}  // namespace

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
    hipStream_t stream) {
    return launch_gemm_mxfp4_w4a8_wmma_impl(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream);
}

}  // namespace ps::kernel
