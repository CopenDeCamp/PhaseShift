#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/gemm_policy.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>

namespace ps::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = kPsqGemmRowTile;
constexpr uint32_t kThreadsPerBlock = 32u;

using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ size_t fp4_act_off(uint32_t r, uint32_t code_stride) {
    return static_cast<size_t>(r >> 4) * (static_cast<size_t>(code_stride) * 16u) +
           static_cast<size_t>(r & 15u) * 16u;
}

template <int MB>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_psq8_w8a8_wmma_kernel(
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
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(weight_scales +
        static_cast<size_t>(o_self >> 4) * weight_scale_stride +
        static_cast<size_t>(o_self & 15u) * 2u);

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        const uint32_t r = row_start + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + fp4_act_off(r, activation_code_stride) + k_group * 8u;
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
                    const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
                    w0 = *reinterpret_cast<const int32x2_t*>(wp);
                    w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                    ws = __bfloat162float(
                        __ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
                }
            } else {
                const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
                w0 = *reinterpret_cast<const int32x2_t*>(wp);
                w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                ws = __bfloat162float(
                    __ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
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
            if (output_dtype == static_cast<uint32_t>(Psq8GemmOutputDType::BF16)) {
                float vb = v;
                static_cast<::ps::bf16_t*>(output)[out_idx] =
                    ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(vb))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

template <class ActivationPolicy, class EpiloguePolicy>
__global__ void __launch_bounds__(kThreadsPerBlock)
gemm_psq8_w8a8_wmma_rowblock1_policy_kernel(
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
    const uint32_t output_row_stride,
    const void* __restrict__ epilogue_residual,
    const uint32_t epilogue_residual_row_stride) {
    const uint32_t nb = k_padded >> 5;
    const uint32_t tid = threadIdx.x;

    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * kTileRow;

    if (out_start >= out_features || row_start >= rows)
        return;

    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;
    const uint32_t r_self = row_start + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(weight_scales +
        static_cast<size_t>(o_self >> 4) * weight_scale_stride +
        static_cast<size_t>(o_self & 15u) * 2u);

    const ActivationPolicy act{
        reinterpret_cast<const uint8_t*>(activation_codes),
        activation_code_stride,
        activation_scales,
        activation_scale_stride};
    const uint8_t* arow = act.row_base(r_self, k_group);

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const bool full_tile = (out_start + kTileOut <= out_features);

    auto run_k_loop = [&]<bool kGuarded, int kUnroll>() {
        #pragma unroll kUnroll
        for (uint32_t ib = 0; ib < nb; ++ib) {
            int32x2_t w0 = {};
            int32x2_t w1 = {};
            float ws = 0.0f;
            if constexpr (kGuarded) {
                if (out_valid) {
                    const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
                    w0 = *reinterpret_cast<const int32x2_t*>(wp);
                    w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                    ws = __bfloat162float(
                        __ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
                }
            } else {
                const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
                w0 = *reinterpret_cast<const int32x2_t*>(wp);
                w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                ws = __bfloat162float(
                    __ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
            }
            int32x2_t a0 = {};
            int32x2_t a1 = {};
            if constexpr (kGuarded) {
                if (r_self < rows)
                    act.load_fragment(arow, ib, a0, a1);
            } else {
                act.load_fragment(arow, ib, a0, a1);
            }
            float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] = __builtin_fmaf(c[j], ws, acc[j]);
        }
    };
    constexpr int kUnroll = 4;
    if (full_tile)
        run_k_loop.template operator()<false, kUnroll>();
    else
        run_k_loop.template operator()<true, kUnroll>();

    const float as_self = (r_self < rows) ? act.row_scale(r_self) : 0.0f;
    float rs[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        rs[j] = __shfl(as_self, 8 * static_cast<int>(k_group) + j);
    if (!out_valid)
        return;
    const EpiloguePolicy epi = EpiloguePolicy::make(
        output, output_dtype == static_cast<uint32_t>(Psq8GemmOutputDType::BF16),
        epilogue_residual, epilogue_residual_row_stride);
    const uint32_t r_base = row_start + 8u * k_group;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t r = r_base + static_cast<uint32_t>(j);
        if (r >= rows)
            continue;
        const float v = acc[j] * rs[j];
        const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
        epi.store(out_idx, r, o_self, v);
    }
}

template <Psq8GemmOutputDType kOutputType, int kUnroll, class ActivationPolicy,
          class EpiloguePolicy>
__device__ __forceinline__ void psq8_decode1_body(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const void* __restrict__ epilogue_residual,
    const uint32_t epilogue_residual_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;

    const uint32_t o_self = blockIdx.x * kTileOut + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(
        weight_scales + static_cast<size_t>(o_self >> 4) * weight_scale_stride +
        static_cast<size_t>(o_self & 15u) * 2u);

    const ActivationPolicy act{
        reinterpret_cast<const uint8_t*>(activation_codes), 0u, activation_scales, 4u};
    const uint8_t* arow = act.row_base(0u, k_group);

    const uint32_t nb = k_padded >> 5;
    float acc0 = 0.0f;
    #pragma unroll kUnroll
    for (uint32_t ib = 0; ib < nb; ++ib) {
        const int32x2_t w0 =
            *reinterpret_cast<const int32x2_t*>(wrow + static_cast<size_t>(ib) * 512u);
        const int32x2_t w1 =
            *reinterpret_cast<const int32x2_t*>(wrow + static_cast<size_t>(ib) * 512u + 256u);
        int32x2_t a0 = {};
        int32x2_t a1 = {};
        act.load_fragment(arow, ib, a0, a1);
        const float ws = __bfloat162float(
            __ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
        acc0 = __builtin_fmaf(c[0], ws, acc0);
    }

    if (k_group == 0u) {
        const float v = acc0 * act.row_scale(0u);
        const EpiloguePolicy epi = EpiloguePolicy::make(
            output, kOutputType == Psq8GemmOutputDType::BF16, epilogue_residual,
            epilogue_residual_row_stride);
        epi.store(o_self, 0u, o_self, v);
    }
}

template <Psq8GemmOutputDType kOutputType, int kUnroll, class ActivationPolicy,
          class EpiloguePolicy>
__global__ void __launch_bounds__(kThreadsPerBlock)
gemv_psq8_w8a8_wmma_decode1_policy_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const void* __restrict__ epilogue_residual,
    const uint32_t epilogue_residual_row_stride) {
    psq8_decode1_body<kOutputType, kUnroll, ActivationPolicy, EpiloguePolicy>(
        weight_codes, weight_scales, activation_codes, activation_scales, output,
        k_padded, weight_scale_stride, epilogue_residual,
        epilogue_residual_row_stride);
}

using u8v8_t = uint8_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void psq8_local_barrier() {
    asm volatile("s_wait_dscnt 0\ns_barrier_signal -1\ns_barrier_wait -1\ns_wait_dscnt 0" ::: "memory");
}

constexpr uint32_t k2dThreads = 256u;

template <bool kGuarded, uint32_t kOutBlock, uint32_t kKChunk>
__global__ void __launch_bounds__(k2dThreads) gemm_psq8_w8a8_wmma_prefill_2d_kernel(
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
    constexpr uint32_t kNt = kOutBlock / kTileOut;
    constexpr uint32_t kIbc = kKChunk / 32u;
    constexpr uint32_t kNtPerWave = kNt;
    constexpr uint32_t kSlotsPerWave = (kNt * kIbc) / 8u;
    const uint32_t nb = k_padded / kKChunk;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;

    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4;
    const uint32_t wave_m = wave;

    const uint32_t out_block_start = blockIdx.x * kOutBlock;
    const uint32_t row_start = blockIdx.y * 16u * kTileRow;
    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t row_base = row_start + wave_m * 2u * kTileRow;
    const uint8_t* arow[2];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + fp4_act_off(r, activation_code_stride) + k_group * 8u;
    }

    __shared__ u8v8_t w0_lds[kNt][kIbc][32];
    __shared__ u8v8_t w1_lds[kNt][kIbc][32];
    __shared__ float ws_lds[kNt][kIbc][16];

    float8_t acc[2][kNtPerWave];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        #pragma unroll
        for (uint32_t nt = 0; nt < kNtPerWave; ++nt)
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    auto stage_slot = [&](uint32_t nt, uint32_t ib, uint32_t round) {
        const uint32_t o_stage = out_block_start + nt * kTileOut + m_lane;
        if (kGuarded && o_stage >= out_features)
            return;
        const uint32_t ibg = round * kIbc + ib;
        const uint8_t* w_stage = weight_codes +
            static_cast<size_t>(o_stage >> 4) * (static_cast<size_t>(k_padded) * 16u) +
            static_cast<size_t>(o_stage & 15u) * 16u + k_group * 8u +
            static_cast<size_t>(ibg) * 512u;
        w0_lds[nt][ib][lane] = *reinterpret_cast<const u8v8_t*>(w_stage);
        w1_lds[nt][ib][lane] = *reinterpret_cast<const u8v8_t*>(w_stage + 256u);
        ws_lds[nt][ib][m_lane] = __bfloat162float(__ushort_as_bfloat16(
            reinterpret_cast<const uint16_t*>(weight_scales +
                static_cast<size_t>(o_stage >> 4) * weight_scale_stride +
                static_cast<size_t>(o_stage & 15u) * 2u)[static_cast<size_t>(ibg) * 16u]));
    };

    auto compute_round = [&](uint32_t round) {
        #pragma unroll
        for (uint32_t ib = 0; ib < kIbc; ++ib) {
            int32x2_t w0[kNtPerWave];
            int32x2_t w1[kNtPerWave];
            float ws[kNtPerWave];
            #pragma unroll
            for (uint32_t nt = 0; nt < kNtPerWave; ++nt) {
                w0[nt] = *reinterpret_cast<const int32x2_t*>(&w0_lds[nt][ib][lane]);
                w1[nt] = *reinterpret_cast<const int32x2_t*>(&w1_lds[nt][ib][lane]);
                ws[nt] = ws_lds[nt][ib][m_lane];
            }
            const uint32_t ibg = round * kIbc + ib;
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                int32x2_t a0 = {};
                int32x2_t a1 = {};
                if constexpr (kGuarded) {
                    if (row_base + static_cast<uint32_t>(t) * kTileRow + m_lane < rows) {
                        const uint8_t* ap = arow[t] + static_cast<size_t>(ibg) * 512u;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    }
                } else {
                    const uint8_t* ap = arow[t] + static_cast<size_t>(ibg) * 512u;
                    a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                }
                #pragma unroll
                for (uint32_t nt = 0; nt < kNtPerWave; ++nt) {
                    float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0[nt], c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1[nt], c);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][nt][j] = __builtin_fmaf(c[j], ws[nt], acc[t][nt][j]);
                }
            }
        }
    };

    for (uint32_t round = 0; round < nb; ++round) {
        #pragma unroll
        for (uint32_t s = 0; s < kSlotsPerWave; ++s) {
            const uint32_t slot = wave * kSlotsPerWave + s;
            stage_slot(slot / kIbc, slot % kIbc, round);
        }
        psq8_local_barrier();
        compute_round(round);
        psq8_local_barrier();
    }

    #pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint32_t r_self = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        const float as_self = (r_self < rows)
            ? activation_scales[static_cast<size_t>(r_self) * (activation_scale_stride / 4u)]
            : 0.0f;
        float rs[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            rs[j] = __shfl(as_self, 8 * static_cast<int>(k_group) + j);
        #pragma unroll
        for (uint32_t nt = 0; nt < kNtPerWave; ++nt) {
            const uint32_t o_self = out_block_start + nt * kTileOut + m_lane;
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
                if (output_dtype == static_cast<uint32_t>(Psq8GemmOutputDType::BF16)) {
                    float vb = v;
                    static_cast<::ps::bf16_t*>(output)[out_idx] =
                        ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(vb))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}
}

extern "C" __global__ __launch_bounds__(kThreadsPerBlock)
void phaseshift_qwen35_psq8_decode1_bf16_u8(Psq8Decode1Bf16Args args) {
    psq8_decode1_body<Psq8GemmOutputDType::BF16, 8, CanonicalA8ActivationPolicy,
                      CanonicalStoreEpilogue>(
        static_cast<const uint8_t*>(args.weight_codes),
        static_cast<const uint8_t*>(args.weight_scales),
        static_cast<const uint8_t*>(args.activation_codes),
        static_cast<const float*>(args.activation_scales), args.output,
        args.k_padded, args.weight_scale_stride, nullptr, 0u);
}

namespace {

#define PSQ8_MB_LAUNCH(MB)                                                      \
    do {                                                                        \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,              \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);  \
        gemm_psq8_w8a8_wmma_kernel<(MB)>                                        \
            <<<grid, kThreadsPerBlock, 0, stream>>>(                            \
                weight_codes, weight_scales, acodes, activation_scales, output, \
                static_cast<uint32_t>(output_dtype), rows, out_features,        \
                k_padded, weight_scale_stride_bytes,                            \
                activation_code_stride_bytes, activation_scale_stride_bytes,    \
                output_row_stride);                                             \
        return hipGetLastError();                                               \
    } while (0)

#define PSQ8_ROWBLOCK1_POLICY_LAUNCH()                                           \
    do {                                                                         \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,               \
                        (rows + kTileRow - 1u) / kTileRow, 1);                   \
        gemm_psq8_w8a8_wmma_rowblock1_policy_kernel<                             \
            CanonicalA8ActivationPolicy, CanonicalStoreEpilogue>                 \
            <<<grid, kThreadsPerBlock, 0, stream>>>(                             \
                weight_codes, weight_scales, acodes, activation_scales, output,  \
                static_cast<uint32_t>(output_dtype), rows, out_features,         \
                k_padded, weight_scale_stride_bytes,                             \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride, nullptr, 0u);                                 \
        return hipGetLastError();                                                \
    } while (0)

hipError_t launch_gemm_psq8_w8a8_wmma_impl(
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
    hipStream_t stream) {
    if (rows == 0 || out_features == 0 || k_padded == 0)
        return hipSuccess;
#define PSQ8_2D_LAUNCH(OB, KC)                                                   \
    do {                                                                         \
        if ((rows % (16u * kTileRow)) != 0u || (out_features % (OB)) != 0u ||    \
            (k_padded % (KC)) != 0u)                                             \
            return hipErrorNotSupported;                                         \
        const dim3 grid(out_features / (OB), rows / (16u * kTileRow), 1);        \
        gemm_psq8_w8a8_wmma_prefill_2d_kernel<false, (OB), (KC)>                 \
            <<<grid, k2dThreads, 0, stream>>>(                                   \
                weight_codes, weight_scales, acodes, activation_scales, output,  \
                static_cast<uint32_t>(output_dtype), rows, out_features,         \
                k_padded, weight_scale_stride_bytes,                             \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride);                                              \
        return hipGetLastError();                                                \
    } while (0)

    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    switch (config.id) {
    case Psq8GemmConfigId::Prefill2D:
        PSQ8_2D_LAUNCH(kPsq8GemmPrefill2dBlock64, kPsq8GemmPrefill2dBlock64);
    case Psq8GemmConfigId::Prefill2D_K64N128:
        PSQ8_2D_LAUNCH(kPsq8GemmPrefill2dBlock128, kPsq8GemmPrefill2dBlock64);
    case Psq8GemmConfigId::Prefill2D_K128N128:
        PSQ8_2D_LAUNCH(kPsq8GemmPrefill2dBlock128, kPsq8GemmPrefill2dBlock128);
    case Psq8GemmConfigId::RowBlock8:
        PSQ8_MB_LAUNCH(8);
    case Psq8GemmConfigId::RowBlock4:
        PSQ8_MB_LAUNCH(4);
    case Psq8GemmConfigId::RowBlock2:
        PSQ8_MB_LAUNCH(2);
    case Psq8GemmConfigId::RowBlock1:
    default:
        PSQ8_ROWBLOCK1_POLICY_LAUNCH();
    }
}

#undef PSQ8_ROWBLOCK1_POLICY_LAUNCH
#undef PSQ8_2D_LAUNCH
#undef PSQ8_MB_LAUNCH

}  // namespace

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
    hipStream_t stream) {
    return launch_gemm_psq8_w8a8_wmma_impl(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream);
}

bool psq8_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded) {
    return rows == 1u && out_features != 0u && (out_features & 15u) == 0u &&
           (k_padded & 31u) == 0u;
}

namespace {

template <Psq8GemmOutputDType kOutputType, int kUnroll>
void decode1_launch(
    dim3 grid,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    hipStream_t stream) {
    if constexpr (kOutputType == Psq8GemmOutputDType::BF16 && kUnroll == 8) {
        Psq8Decode1Bf16Args args{};
        args.weight_codes = weight_codes;
        args.weight_scales = weight_scales;
        args.activation_codes = activation_codes;
        args.activation_scales = activation_scales;
        args.output = output;
        args.k_padded = k_padded;
        args.weight_scale_stride = weight_scale_stride_bytes;
        phaseshift_qwen35_psq8_decode1_bf16_u8<<<grid, kThreadsPerBlock, 0,
                                                   stream>>>(args);
    } else {
        gemv_psq8_w8a8_wmma_decode1_policy_kernel<kOutputType, kUnroll,
            CanonicalA8ActivationPolicy, CanonicalStoreEpilogue>
            <<<grid, kThreadsPerBlock, 0, stream>>>(
                weight_codes, weight_scales, activation_codes, activation_scales,
                output, k_padded, weight_scale_stride_bytes, nullptr, 0u);
    }
}

template <Psq8GemmOutputDType kOutputType>
hipError_t launch_decode1_dtype(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t unroll,
    hipStream_t stream) {
    const dim3 grid(out_features >> 4);
    switch (unroll) {
        case 2u:
            decode1_launch<kOutputType, 2>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream);
            break;
        case 8u:
            decode1_launch<kOutputType, 8>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream);
            break;
        case 16u:
            decode1_launch<kOutputType, 16>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream);
            break;
        default:
            decode1_launch<kOutputType, 4>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream);
            break;
    }
    return hipGetLastError();
}

}  // namespace

uint32_t psq8_decode1_unroll(uint32_t out_features, uint32_t k_padded) {
    (void)out_features;
    (void)k_padded;
    return 8u;
}

uint32_t psq8_decode1_block_size() noexcept {
    return 32u;
}

uint32_t psq8_decode1_grid(uint32_t out_features) noexcept {
    return out_features >> 4u;
}

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
    hipStream_t stream) {
    (void)activation_code_stride_bytes;
    (void)activation_scale_stride_bytes;
    (void)output_row_stride;
    if (!psq8_decode1_supported(rows, out_features, k_padded))
        return hipErrorNotSupported;
    if (unroll == 0u) unroll = psq8_decode1_unroll(out_features, k_padded);
    if (unroll == 0u) unroll = 4u;
    if (output_dtype == Psq8GemmOutputDType::BF16)
        return launch_decode1_dtype<Psq8GemmOutputDType::BF16>(
            weight_codes, weight_scales, activation_codes, activation_scales,
            output, out_features, k_padded, weight_scale_stride_bytes, unroll, stream);
    return launch_decode1_dtype<Psq8GemmOutputDType::F32>(
        weight_codes, weight_scales, activation_codes, activation_scales,
        output, out_features, k_padded, weight_scale_stride_bytes, unroll, stream);
}

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
    bool* decode1_used) {
    if (decode1_used != nullptr) *decode1_used = false;
    if (config.id == Psq8GemmConfigId::RowBlock1 &&
        psq8_decode1_supported(rows, out_features, k_padded)) {
        if (decode1_used != nullptr) *decode1_used = true;
        return launch_gemm_psq8_w8a8_wmma_decode1(
            weight_codes, weight_scales,
            reinterpret_cast<const uint8_t*>(activation_codes), activation_scales,
            output, output_dtype, rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride, 0u, stream);
    }
    return launch_gemm_psq8_w8a8_wmma(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream);
}


}  // namespace ps::kernel
