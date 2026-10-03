#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>

#include "detail/coarse_head_device.h"
#include "detail/coarse_topn_device.h"
#include "detail/psq8_rerank_device.h"

#include <phaseshift/quantization/fpx/e4m3.h>
#include <phaseshift/quantization/fpx/e4m3_detail.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = kDflash2Int2TileOut;
constexpr uint32_t kThreadsPerBlock = 32u;
constexpr uint32_t kTileRow = 16u;
constexpr uint32_t kCoarseHeadWaves = 16u;
constexpr uint32_t kCoarseHeadThreads = 32u * kCoarseHeadWaves;
constexpr uint32_t kCoarseHeadOutPerBlock = kTileOut * kCoarseHeadWaves;
constexpr float kHistogramScale = 1048576.0f;

__device__ __forceinline__ size_t preshuffle_row_off(uint32_t r, uint32_t code_stride) {
    return detail::coarse_preshuffle_row_off(r, code_stride);
}

__global__ void int2_code_histogram_kernel(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales,
    const uint32_t rows,
    const uint32_t k_padded,
    const uint32_t scale_stride,
    uint64_t* __restrict__ histogram) {
    const uint32_t nb = k_padded >> 5u;
    const uint32_t tasks = rows * nb;
    __shared__ uint64_t shared_hist[256];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x)
        shared_hist[i] = 0ull;
    __syncthreads();

    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < tasks;
         t += gridDim.x * blockDim.x) {
        const uint32_t o = t / nb;
        const uint32_t ib = t - o * nb;
        const uint16_t* srow = reinterpret_cast<const uint16_t*>(
            scales + static_cast<size_t>(o >> 4) * scale_stride +
            static_cast<size_t>(o & 15u) * 2u);
        const float s =
            __bfloat162float(__ushort_as_bfloat16(srow[static_cast<size_t>(ib) * 16u]));
        const uint64_t w = static_cast<uint64_t>(
            static_cast<double>(s) * static_cast<double>(s) *
                static_cast<double>(kHistogramScale) +
            0.5);
        if (w == 0ull)
            continue;
        const uint8_t* chunk = codes +
            static_cast<size_t>(o >> 4) * (static_cast<size_t>(k_padded) * 16u) +
            static_cast<size_t>(o & 15u) * 16u + static_cast<size_t>(ib) * 512u;
        #pragma unroll
        for (uint32_t q = 0u; q < 16u; ++q) {
            atomicAdd(&shared_hist[chunk[q]], w);
            atomicAdd(&shared_hist[chunk[q + 256u]], w);
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        if (shared_hist[i] != 0ull)
            atomicAdd(&histogram[i], shared_hist[i]);
    }
}

struct E4m3Candidate {
    float value = 0.0f;
    uint8_t byte = 0u;
};

std::vector<E4m3Candidate> finite_e4m3_candidates() {
    std::vector<E4m3Candidate> out;
    out.reserve(256u);
    for (uint32_t b = 0u; b < 256u; ++b) {
        const float v = ::ps::quantization::fpx::e4m3_decode_f32(static_cast<uint8_t>(b));
        if (!std::isfinite(v))
            continue;
        out.push_back(E4m3Candidate{v, static_cast<uint8_t>(b)});
    }
    std::sort(out.begin(), out.end(), [](const E4m3Candidate& a, const E4m3Candidate& b) {
        if (a.value != b.value)
            return a.value < b.value;
        return a.byte < b.byte;
    });
    return out;
}

void refine_centroids(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    std::vector<double>& centroids) {
    const size_t n = values.size();
    const size_t k = centroids.size();
    for (uint32_t iter = 0u; iter < 64u; ++iter) {
        std::vector<double> sum(k, 0.0);
        std::vector<double> mass(k, 0.0);
        for (size_t i = 0u; i < n; ++i) {
            size_t best = 0u;
            double best_dist = std::fabs(values[i] - centroids[0]);
            for (size_t c = 1u; c < k; ++c) {
                const double d = std::fabs(values[i] - centroids[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best = c;
                }
            }
            sum[best] += values[i] * weights[i];
            mass[best] += weights[i];
        }
        bool moved = false;
        for (size_t c = 0u; c < k; ++c) {
            if (mass[c] <= 0.0)
                continue;
            const double next = sum[c] / mass[c];
            if (next != centroids[c])
                moved = true;
            centroids[c] = next;
        }
        if (!moved)
            break;
    }
    std::sort(centroids.begin(), centroids.end());
}

double weighted_error(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    const std::vector<double>& centroids) {
    const size_t n = values.size();
    const size_t k = centroids.size();
    double total = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        double best_dist = std::fabs(values[i] - centroids[0]);
        for (size_t c = 1u; c < k; ++c) {
            const double d = std::fabs(values[i] - centroids[c]);
            if (d < best_dist)
                best_dist = d;
        }
        total += best_dist * best_dist * weights[i];
    }
    return total;
}

uint8_t snap_to_e4m3(const std::vector<E4m3Candidate>& candidates, double value) {
    double best_dist = -1.0;
    uint8_t best_byte = 0u;
    for (const E4m3Candidate& cand : candidates) {
        const double d = std::fabs(static_cast<double>(cand.value) - value);
        if (best_dist < 0.0 || d < best_dist) {
            best_dist = d;
            best_byte = cand.byte;
        }
    }
    return best_byte;
}

__global__ void int2_pack_kernel(
    const uint8_t* __restrict__ psq8_codes,
    const uint8_t* __restrict__ map_table,
    uint8_t* __restrict__ int2_codes,
    const uint32_t rows,
    const uint32_t k_padded) {
    const uint32_t nb = k_padded >> 5u;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t kg = tid & 1u;
    const uint32_t o = tid >> 1u;
    if (o >= rows)
        return;
    const uint8_t* src = psq8_codes +
        static_cast<size_t>(o >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o & 15u) * 16u + kg * 8u;
    uint8_t* dst = int2_codes +
        static_cast<size_t>(o >> 4) * (static_cast<size_t>(k_padded) * 4u) +
        static_cast<size_t>(o & 15u) * 4u + kg * 2u;
    for (uint32_t ib = 0u; ib < nb; ++ib) {
        const uint8_t* chunk = src + static_cast<size_t>(ib) * 512u;
        #pragma unroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint8_t* q = chunk + half * 256u;
            uint8_t packed0 = 0u;
            uint8_t packed1 = 0u;
            #pragma unroll
            for (uint32_t q4 = 0u; q4 < 4u; ++q4) {
                packed0 = static_cast<uint8_t>(packed0 | (map_table[q[q4]] << (2u * q4)));
                packed1 = static_cast<uint8_t>(packed1 | (map_table[q[q4 + 4u]] << (2u * q4)));
            }
            dst[static_cast<size_t>(ib) * 128u + half * 64u] = packed0;
            dst[static_cast<size_t>(ib) * 128u + half * 64u + 1u] = packed1;
        }
    }
}

__global__ void dflash2_gather_pack_int2_rows_kernel(
    const uint8_t* source_codes,
    const uint8_t* source_scales,
    const int32_t* selected_ids,
    const uint8_t* map_table,
    uint8_t* reduced_codes,
    uint8_t* reduced_scales,
    uint32_t selected_rows,
    uint32_t k_padded,
    uint32_t source_code_stride,
    uint32_t source_scale_stride,
    uint32_t reduced_code_stride,
    uint32_t reduced_scale_stride) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= selected_rows)
        return;
    const uint32_t source_row = static_cast<uint32_t>(selected_ids[row]);
    const uint32_t blocks_k = k_padded >> 5u;
    const uint8_t* source_row_base =
        source_codes + static_cast<std::size_t>(source_row >> 4u) * source_code_stride +
        static_cast<std::size_t>(source_row & 15u) * 16u;
    uint8_t* reduced_row_base =
        reduced_codes + static_cast<std::size_t>(row >> 4u) * reduced_code_stride +
        static_cast<std::size_t>(row & 15u) * 4u;
    for (uint32_t ib = 0u; ib < blocks_k; ++ib) {
        const uint8_t* source_block = source_row_base + static_cast<std::size_t>(ib) * 512u;
        uint8_t* reduced_block = reduced_row_base + static_cast<std::size_t>(ib) * 128u;
        for (uint32_t half = 0u; half < 2u; ++half) {
            for (uint32_t kg = 0u; kg < 2u; ++kg) {
                const uint8_t* source_half = source_block + half * 256u + kg * 8u;
                uint8_t packed0 = 0u;
                uint8_t packed1 = 0u;
                #pragma unroll
                for (uint32_t q4 = 0u; q4 < 4u; ++q4) {
                    packed0 = static_cast<uint8_t>(packed0 |
                        (map_table[source_half[q4]] << (2u * q4)));
                    packed1 = static_cast<uint8_t>(packed1 |
                        (map_table[source_half[q4 + 4u]] << (2u * q4)));
                }
                reduced_block[half * 64u + kg * 2u] = packed0;
                reduced_block[half * 64u + kg * 2u + 1u] = packed1;
            }
        }
        const uint16_t* source_scale = reinterpret_cast<const uint16_t*>(
            source_scales + static_cast<std::size_t>(source_row >> 4u) * source_scale_stride +
            static_cast<std::size_t>(source_row & 15u) * 2u);
        uint16_t* reduced_scale = reinterpret_cast<uint16_t*>(
            reduced_scales + static_cast<std::size_t>(row >> 4u) * reduced_scale_stride +
            static_cast<std::size_t>(row & 15u) * 2u);
        reduced_scale[static_cast<std::size_t>(ib) * 16u] =
            source_scale[static_cast<std::size_t>(ib) * 16u];
    }
}

__global__ void __launch_bounds__(kCoarseHeadThreads) int2_coarse_head_kernel(
    const uint8_t* __restrict__ int2_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    const uint32_t* __restrict__ expand_table,
    float* __restrict__ logits,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t logits_row_stride) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t out_start = blockIdx.x * kCoarseHeadOutPerBlock + warp * kTileOut;
    const bool tile_valid = out_start < out_features;

    __shared__ uint32_t lut[256];
    detail::coarse_head_load_lut(lut, expand_table);
    __syncthreads();

    if (!tile_valid)
        return;

    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4;
    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;

    float vals[8];
    detail::coarse_head_tile_body(int2_codes, weight_scales, activation_codes,
                                  activation_scales, lut, lane, out_start, rows, out_features,
                                  k_padded, weight_scale_stride, activation_code_stride,
                                  activation_scale_stride, vals);
    if (!out_valid)
        return;
    const uint32_t r_base = 8u * k_group;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t r = r_base + static_cast<uint32_t>(j);
        if (r >= rows)
            continue;
        logits[static_cast<size_t>(r) * logits_row_stride + o_self] = vals[j];
    }
}

__global__ void __launch_bounds__(kThreadsPerBlock) psq8_candidate_rerank_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    const int32_t* __restrict__ candidate_ids,
    float* __restrict__ logits,
    const uint32_t rows,
    const uint32_t pool,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t logits_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;
    const uint32_t row = blockIdx.y;
    if (row >= rows || blockIdx.x * kTileOut >= pool)
        return;
    const uint32_t cand = blockIdx.x * kTileOut + m_lane;
    const bool cand_valid = cand < pool;
    const uint32_t token_id =
        cand_valid ? static_cast<uint32_t>(candidate_ids[static_cast<size_t>(row) * pool + cand])
                   : 0u;
    const float acc0 = detail::psq8_rerank_tile_body(
        weight_codes, weight_scales, activation_codes, token_id, cand_valid, m_lane, row, rows,
        k_padded, weight_scale_stride, activation_code_stride, k_group);
    if (!cand_valid || k_group != 0u)
        return;
    const float as_scale =
        activation_scales[static_cast<size_t>(row) * (activation_scale_stride / 4u)];
    logits[static_cast<size_t>(row) * logits_row_stride + cand] = acc0 * as_scale;
}

__global__ void int2_remap_pool_ids_kernel(
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ pool_ids,
    int32_t* __restrict__ out_ids,
    const uint32_t rows,
    const uint32_t k,
    const uint32_t pool,
    const uint32_t out_row_stride) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * k)
        return;
    const uint32_t row = idx / k;
    const uint32_t i = idx - row * k;
    const int32_t pos = positions[idx];
    int32_t id = -1;
    if (pos >= 0 && static_cast<uint32_t>(pos) < pool)
        id = pool_ids[static_cast<std::size_t>(row) * pool + static_cast<uint32_t>(pos)];
    out_ids[static_cast<std::size_t>(row) * out_row_stride + i] = id;
}

constexpr uint32_t kProxyWarpsPerBlock = 8u;
constexpr uint32_t kProxyThreads = kProxyWarpsPerBlock * 32u;

__device__ __forceinline__ float proxy_warp_sum(float v) {
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        v += __shfl_xor_sync(0xffffffffULL, v, off);
    return v;
}

__global__ void __launch_bounds__(kProxyThreads) int2_row_error_l2_kernel(
    const uint8_t* __restrict__ psq8_codes,
    const uint8_t* __restrict__ psq8_scales,
    const float* __restrict__ error_table,
    const uint32_t rows,
    const uint32_t k_padded,
    const uint32_t scale_stride,
    float* __restrict__ out_error_l2) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * kProxyWarpsPerBlock + warp;
    if (o >= rows)
        return;
    const uint32_t nb = k_padded >> 5u;
    const uint8_t* crow = psq8_codes + detail::coarse_preshuffle_row_off(o, k_padded);
    const uint16_t* srow = reinterpret_cast<const uint16_t*>(
        psq8_scales + static_cast<std::size_t>(o >> 4) * scale_stride +
        static_cast<std::size_t>(o & 15u) * 2u);
    const uint32_t lane_off = (lane >> 4u) * 256u + (lane & 15u);
    float acc = 0.0f;
    for (uint32_t ib = 0u; ib < nb; ++ib) {
        const uint8_t code = crow[static_cast<std::size_t>(ib) * 512u + lane_off];
        const float s =
            __bfloat162float(__ushort_as_bfloat16(srow[static_cast<std::size_t>(ib) * 16u]));
        const float d = error_table[code] * s;
        acc = __builtin_fmaf(d, d, acc);
    }
    acc = proxy_warp_sum(acc);
    if (lane == 0u)
        out_error_l2[o] = __builtin_sqrtf(acc);
}

__global__ void __launch_bounds__(kProxyThreads) activation_l2_kernel(
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    const uint32_t rows,
    const uint32_t k_padded,
    const uint32_t code_stride,
    const uint32_t scale_stride,
    float* __restrict__ out_activation_l2) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t m = blockIdx.x * kProxyWarpsPerBlock + warp;
    if (m >= rows)
        return;
    const uint32_t nb = k_padded >> 5u;
    const uint8_t* crow = activation_codes + detail::coarse_preshuffle_row_off(m, code_stride);
    const float s = activation_scales[static_cast<std::size_t>(m) * (scale_stride / 4u)];
    const uint32_t lane_off = (lane >> 4u) * 256u + (lane & 15u);
    float acc = 0.0f;
    for (uint32_t ib = 0u; ib < nb; ++ib) {
        const uint8_t code = crow[static_cast<std::size_t>(ib) * 512u + lane_off];
        const float v = ::ps::quantization::fpx::device::e4m3_decode_f32(code) * s;
        acc = __builtin_fmaf(v, v, acc);
    }
    acc = proxy_warp_sum(acc);
    if (lane == 0u)
        out_activation_l2[m] = __builtin_sqrtf(acc);
}

__global__ void __launch_bounds__(256u) target_upper_logits_kernel(
    const float* __restrict__ coarse_logits,
    const float* __restrict__ activation_l2,
    const float* __restrict__ error_l2,
    float* __restrict__ upper_logits,
    const uint32_t vocab) {
    const uint32_t row = blockIdx.y;
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vocab)
        return;
    const float bound = activation_l2[row] * error_l2[v];
    const std::size_t idx = static_cast<std::size_t>(row) * vocab + v;
    upper_logits[idx] = coarse_logits[idx] + bound;
}

__global__ void __launch_bounds__(kProxyThreads) target_certified_argmax_kernel(
    const float* __restrict__ candidate_logits,
    const int32_t* __restrict__ candidate_ids,
    const float* __restrict__ omitted_upper,
    const uint32_t omitted_upper_stride,
    const float absolute_margin,
    const float relative_margin,
    int32_t* __restrict__ best_ids,
    float* __restrict__ best_logits,
    float* __restrict__ gaps,
    uint8_t* __restrict__ certified,
    const uint32_t rows,
    const uint32_t pool) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * kProxyWarpsPerBlock + warp;
    if (row >= rows)
        return;
    const float* logits = candidate_logits + static_cast<std::size_t>(row) * pool;
    const int32_t* ids = candidate_ids + static_cast<std::size_t>(row) * pool;
    detail::TopnBest b{-INFINITY, 0x7FFFFFFF, -1};
    for (uint32_t c = lane; c < pool; c += 32u) {
        if (detail::topn_better(logits[c], ids[c], b.value, b.id)) {
            b.value = logits[c];
            b.id = ids[c];
        }
    }
    b = detail::topn_warp_best<32u>(b);
    if (lane != 0u)
        return;
    const float omitted =
        omitted_upper[static_cast<std::size_t>(row) * omitted_upper_stride];
    const float gap = b.value - omitted;
    const float scale = fmaxf(__builtin_fabsf(b.value), __builtin_fabsf(omitted));
    const float margin = absolute_margin + relative_margin * scale;
    best_ids[row] = b.id;
    best_logits[row] = b.value;
    gaps[row] = gap;
    certified[row] = gap > margin ? 1u : 0u;
}

__global__ void __launch_bounds__(256u) verify_max_other_upper_kernel(
    const float* __restrict__ coarse_logits,
    const float* __restrict__ activation_l2,
    const float* __restrict__ error_l2,
    const int32_t* __restrict__ proposed_token_ids,
    float* __restrict__ out_max_other_upper,
    const uint32_t vocab) {
    const uint32_t row = blockIdx.x;
    const float scale = activation_l2[row];
    const int32_t proposed = proposed_token_ids[row];
    float local = -INFINITY;
    for (uint32_t v = threadIdx.x; v < vocab; v += blockDim.x) {
        if (static_cast<int32_t>(v) == proposed)
            continue;
        const float bound = coarse_logits[static_cast<size_t>(row) * vocab + v] +
                            scale * error_l2[v];
        local = fmaxf(local, bound);
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t off = 128u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        out_max_other_upper[row] = shared[0];
}

__global__ void verify_direct_certificate_kernel(
    const float* __restrict__ exact_proposed_logits,
    const float* __restrict__ max_other_upper,
    const float absolute_margin,
    const float relative_margin,
    uint8_t* __restrict__ certified,
    const uint32_t rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const float exact = exact_proposed_logits[row];
    const float other = max_other_upper[row];
    const float scale = fmaxf(__builtin_fabsf(exact), __builtin_fabsf(other));
    const float margin = absolute_margin + relative_margin * scale;
    certified[row] = exact > other + margin ? 1u : 0u;
}

__global__ void verify_decision_sufficiency_kernel(
    const int32_t* __restrict__ draft_tokens,
    const uint8_t* __restrict__ row_state,
    const int32_t* __restrict__ certified_tokens,
    const uint32_t num_drafts,
    uint32_t* __restrict__ out_decision_ready,
    uint32_t* __restrict__ out_num_accepted,
    uint32_t* __restrict__ out_correction_valid,
    int32_t* __restrict__ out_correction_token,
    uint32_t* __restrict__ out_bonus_valid,
    int32_t* __restrict__ out_bonus_token,
    int32_t* __restrict__ out_first_required_unknown_row) {
    uint32_t accepted = 0u;
    uint32_t ready = 0u;
    uint32_t correction_valid = 0u;
    int32_t correction_token = -1;
    uint32_t bonus_valid = 0u;
    int32_t bonus_token = -1;
    int32_t first_unknown = -1;
    for (uint32_t row = 0u; row < num_drafts; ++row) {
        const uint8_t state = row_state[row];
        if (state == kDflash2VerifyRowAcceptCertified) {
            ++accepted;
            continue;
        }
        if (state == kDflash2VerifyRowArgmaxCertified) {
            if (certified_tokens[row] != draft_tokens[row]) {
                correction_valid = 1u;
                correction_token = certified_tokens[row];
                ready = 1u;
                break;
            }
            ++accepted;
            continue;
        }
        first_unknown = static_cast<int32_t>(row);
        break;
    }
    if (ready == 0u && first_unknown < 0) {
        if (row_state[num_drafts] == kDflash2VerifyRowArgmaxCertified) {
            bonus_valid = 1u;
            bonus_token = certified_tokens[num_drafts];
            ready = 1u;
        } else {
            first_unknown = static_cast<int32_t>(num_drafts);
        }
    }
    out_decision_ready[0] = ready;
    out_num_accepted[0] = accepted;
    out_correction_valid[0] = correction_valid;
    out_correction_token[0] = correction_token;
    out_bonus_valid[0] = bonus_valid;
    out_bonus_token[0] = bonus_token;
    out_first_required_unknown_row[0] = first_unknown;
}

}  // namespace

hipError_t launch_dflash2_int2_remap_pool_ids(
    const int32_t* positions,
    const int32_t* pool_ids,
    int32_t* out_ids,
    uint32_t rows,
    uint32_t k,
    uint32_t pool,
    uint32_t out_row_stride,
    hipStream_t stream) {
    if (positions == nullptr || pool_ids == nullptr || out_ids == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || k == 0u || pool == 0u)
        return hipErrorInvalidValue;
    const uint32_t total = rows * k;
    const uint32_t threads = 128u;
    const uint32_t blocks = (total + threads - 1u) / threads;
    int2_remap_pool_ids_kernel<<<blocks, threads, 0, stream>>>(
        positions, pool_ids, out_ids, rows, k, pool, out_row_stride);
    return hipGetLastError();
}

hipError_t launch_dflash2_int2_code_histogram(
    const uint8_t* psq8_codes,
    const uint8_t* psq8_scales,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t scale_stride_bytes,
    uint64_t* histogram,
    hipStream_t stream) {
    if (psq8_codes == nullptr || psq8_scales == nullptr || histogram == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t tasks = rows * (k_padded >> 5u);
    const uint32_t threads = 256u;
    uint32_t blocks = (tasks + threads - 1u) / threads;
    if (blocks == 0u)
        blocks = 1u;
    if (blocks > 65535u)
        blocks = 65535u;
    int2_code_histogram_kernel<<<blocks, threads, 0, stream>>>(
        psq8_codes, psq8_scales, rows, k_padded, scale_stride_bytes, histogram);
    return hipGetLastError();
}

Dflash2Int2Codebook dflash2_int2_derive_codebook(const uint64_t* histogram, bool symmetric) {
    const std::vector<E4m3Candidate> candidates = finite_e4m3_candidates();
    std::vector<double> values;
    std::vector<double> weights;
    for (const E4m3Candidate& c : candidates) {
        if (histogram[c.byte] == 0ull)
            continue;
        values.push_back(static_cast<double>(c.value));
        weights.push_back(static_cast<double>(histogram[c.byte]));
    }
    Dflash2Int2Codebook out;
    if (values.empty())
        return out;

    std::vector<double> snapped(4u, 0.0);
    if (symmetric) {
        std::vector<double> magnitudes;
        for (const E4m3Candidate& c : candidates) {
            if (c.value > 0.0f)
                magnitudes.push_back(static_cast<double>(c.value));
        }
        double best_a = 1.0;
        double best_b = 1.0;
        double best_err = -1.0;
        for (size_t i = 0u; i < magnitudes.size(); ++i) {
            for (size_t j = i; j < magnitudes.size(); ++j) {
                const std::vector<double> centroids{-magnitudes[j], -magnitudes[i],
                                                    magnitudes[i], magnitudes[j]};
                const double err = weighted_error(values, weights, centroids);
                if (best_err < 0.0 || err < best_err) {
                    best_err = err;
                    best_a = magnitudes[i];
                    best_b = magnitudes[j];
                }
            }
        }
        snapped = {-best_b, -best_a, best_a, best_b};
    } else {
        std::vector<double> centroids(4u, 0.0);
        double total = 0.0;
        for (double w : weights)
            total += w;
        const size_t n = values.size();
        for (size_t c = 0u; c < 4u; ++c) {
            const double target = total * ((static_cast<double>(c) + 0.5) / 4.0);
            double running = 0.0;
            size_t idx = 0u;
            for (size_t i = 0u; i < n; ++i) {
                running += weights[i];
                idx = i;
                if (running >= target)
                    break;
            }
            centroids[c] = values[idx];
        }
        refine_centroids(values, weights, centroids);
        for (size_t c = 0u; c < 4u; ++c)
            snapped[c] = centroids[c];
    }

    for (size_t c = 0u; c < 4u; ++c) {
        out.values[c] = snap_to_e4m3(candidates, snapped[c]);
        out.decoded[c] = ::ps::quantization::fpx::e4m3_decode_f32(out.values[c]);
    }
    if (symmetric) {
        std::sort(out.decoded.begin(), out.decoded.end());
        for (size_t c = 0u; c < 4u; ++c)
            out.values[c] = snap_to_e4m3(candidates, static_cast<double>(out.decoded[c]));
        for (size_t c = 0u; c < 4u; ++c)
            out.decoded[c] = ::ps::quantization::fpx::e4m3_decode_f32(out.values[c]);
    }
    return out;
}

double dflash2_int2_codebook_weight_error(
    const Dflash2Int2Codebook& codebook, const uint64_t* histogram) {
    const std::vector<E4m3Candidate> candidates = finite_e4m3_candidates();
    std::vector<double> values;
    std::vector<double> weights;
    for (const E4m3Candidate& c : candidates) {
        if (histogram[c.byte] == 0ull)
            continue;
        values.push_back(static_cast<double>(c.value));
        weights.push_back(static_cast<double>(histogram[c.byte]));
    }
    if (values.empty())
        return 0.0;
    const std::vector<double> centroids(codebook.decoded.begin(), codebook.decoded.end());
    return weighted_error(values, weights, centroids);
}

void dflash2_int2_build_tables(
    const Dflash2Int2Codebook& codebook, uint8_t* map_table, uint32_t* expand_table) {
    for (uint32_t b = 0u; b < 256u; ++b) {
        const float v = ::ps::quantization::fpx::e4m3_decode_f32(static_cast<uint8_t>(b));
        uint32_t best = 0u;
        if (std::isfinite(v)) {
            float best_dist = std::fabs(v - codebook.decoded[0]);
            for (uint32_t c = 1u; c < kDflash2Int2CodebookValues; ++c) {
                const float d = std::fabs(v - codebook.decoded[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best = c;
                }
            }
        }
        map_table[b] = static_cast<uint8_t>(best);
    }
    for (uint32_t packed = 0u; packed < 256u; ++packed) {
        uint32_t word = 0u;
        for (uint32_t q = 0u; q < 4u; ++q) {
            const uint32_t idx = (packed >> (2u * q)) & 3u;
            word |= static_cast<uint32_t>(codebook.values[idx]) << (8u * q);
        }
        expand_table[packed] = word;
    }
}

hipError_t launch_dflash2_int2_pack(
    const uint8_t* psq8_codes,
    const uint8_t* map_table,
    uint8_t* int2_codes,
    uint32_t rows,
    uint32_t k_padded,
    hipStream_t stream) {
    if (psq8_codes == nullptr || map_table == nullptr || int2_codes == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t total = rows * 2u;
    const uint32_t threads = 128u;
    const uint32_t blocks = (total + threads - 1u) / threads;
    int2_pack_kernel<<<blocks, threads, 0, stream>>>(psq8_codes, map_table, int2_codes, rows,
                                                     k_padded);
    return hipGetLastError();
}

hipError_t launch_dflash2_gather_pack_int2_rows(
    const uint8_t* source_codes,
    const uint8_t* source_scales,
    const int32_t* selected_ids,
    const uint8_t* map_table,
    uint8_t* reduced_codes,
    uint8_t* reduced_scales,
    uint32_t selected_rows,
    uint32_t k_padded,
    uint32_t source_code_stride,
    uint32_t source_scale_stride,
    uint32_t reduced_code_stride,
    uint32_t reduced_scale_stride,
    hipStream_t stream) {
    if (source_codes == nullptr || source_scales == nullptr || selected_ids == nullptr ||
        map_table == nullptr || reduced_codes == nullptr || reduced_scales == nullptr)
        return hipErrorInvalidValue;
    if (selected_rows == 0u || k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t threads = 256u;
    const uint32_t blocks = (selected_rows + threads - 1u) / threads;
    dflash2_gather_pack_int2_rows_kernel<<<blocks, threads, 0, stream>>>(
        source_codes, source_scales, selected_ids, map_table, reduced_codes, reduced_scales,
        selected_rows, k_padded, source_code_stride, source_scale_stride, reduced_code_stride,
        reduced_scale_stride);
    return hipGetLastError();
}

hipError_t launch_dflash2_int2_coarse_head(
    const uint8_t* int2_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    const uint32_t* expand_table,
    float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t logits_row_stride,
    hipStream_t stream) {
    if (int2_codes == nullptr || weight_scales == nullptr || activation_codes == nullptr ||
        activation_scales == nullptr || expand_table == nullptr || logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || vocab == 0u || k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    if (rows > kTileRow)
        return hipErrorNotSupported;
    const dim3 grid((vocab + kCoarseHeadOutPerBlock - 1u) / kCoarseHeadOutPerBlock, 1u, 1u);
    int2_coarse_head_kernel<<<grid, kCoarseHeadThreads, 0, stream>>>(
        int2_codes, weight_scales, activation_codes, activation_scales, expand_table, logits,
        rows, vocab, k_padded, weight_scale_stride_bytes, activation_code_stride_bytes,
        activation_scale_stride_bytes, logits_row_stride);
    return hipGetLastError();
}

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
    hipStream_t stream) {
    if (weight_codes == nullptr || weight_scales == nullptr || activation_codes == nullptr ||
        activation_scales == nullptr || candidate_ids == nullptr || logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || pool == 0u || k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const dim3 grid((pool + kTileOut - 1u) / kTileOut, rows, 1u);
    psq8_candidate_rerank_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        weight_codes, weight_scales, activation_codes, activation_scales, candidate_ids,
        logits, rows, pool, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, logits_row_stride);
    return hipGetLastError();
}

void dflash2_int2_build_error_table(
    const Dflash2Int2Codebook& codebook,
    const uint8_t* map_table,
    float* error_table) {
    for (uint32_t b = 0u; b < 256u; ++b) {
        const float v = ::ps::quantization::fpx::e4m3_decode_f32(static_cast<uint8_t>(b));
        if (!std::isfinite(v)) {
            error_table[b] = 0.0f;
            continue;
        }
        error_table[b] = v - codebook.decoded[map_table[b]];
    }
}

hipError_t launch_dflash2_int2_row_error_l2(
    const uint8_t* psq8_codes,
    const uint8_t* psq8_scales,
    const float* error_table,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t scale_stride_bytes,
    float* out_error_l2,
    hipStream_t stream) {
    if (psq8_codes == nullptr || psq8_scales == nullptr || error_table == nullptr ||
        out_error_l2 == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t blocks = (rows + kProxyWarpsPerBlock - 1u) / kProxyWarpsPerBlock;
    int2_row_error_l2_kernel<<<blocks, kProxyThreads, 0, stream>>>(
        psq8_codes, psq8_scales, error_table, rows, k_padded, scale_stride_bytes,
        out_error_l2);
    return hipGetLastError();
}

hipError_t launch_dflash2_activation_l2(
    const uint8_t* activation_codes,
    const float* activation_scales,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    float* out_activation_l2,
    hipStream_t stream) {
    if (activation_codes == nullptr || activation_scales == nullptr ||
        out_activation_l2 == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    if (code_row_stride_bytes % 16u != 0u || scale_row_stride_bytes < 4u)
        return hipErrorInvalidValue;
    const uint32_t blocks = (rows + kProxyWarpsPerBlock - 1u) / kProxyWarpsPerBlock;
    activation_l2_kernel<<<blocks, kProxyThreads, 0, stream>>>(
        activation_codes, activation_scales, rows, k_padded, code_row_stride_bytes,
        scale_row_stride_bytes, out_activation_l2);
    return hipGetLastError();
}

hipError_t launch_dflash2_target_upper_logits(
    const float* coarse_logits,
    const float* activation_l2,
    const float* error_l2,
    float* upper_logits,
    uint32_t rows,
    uint32_t vocab,
    hipStream_t stream) {
    if (coarse_logits == nullptr || activation_l2 == nullptr || error_l2 == nullptr ||
        upper_logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || vocab == 0u)
        return hipSuccess;
    const dim3 grid((vocab + 255u) / 256u, rows, 1u);
    target_upper_logits_kernel<<<grid, 256u, 0, stream>>>(
        coarse_logits, activation_l2, error_l2, upper_logits, vocab);
    return hipGetLastError();
}

hipError_t launch_dflash2_target_certified_argmax(
    const float* candidate_logits,
    const int32_t* candidate_ids,
    const float* omitted_upper,
    uint32_t omitted_upper_stride,
    float absolute_margin,
    float relative_margin,
    int32_t* best_ids,
    float* best_logits,
    float* gaps,
    uint8_t* certified,
    uint32_t rows,
    uint32_t pool,
    hipStream_t stream) {
    if (candidate_logits == nullptr || candidate_ids == nullptr || omitted_upper == nullptr ||
        best_ids == nullptr || best_logits == nullptr || gaps == nullptr || certified == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (pool == 0u || pool > kDflash2TargetProxyMaxPool)
        return hipErrorInvalidValue;
    if (omitted_upper_stride == 0u)
        return hipErrorInvalidValue;
    const uint32_t blocks = (rows + kProxyWarpsPerBlock - 1u) / kProxyWarpsPerBlock;
    target_certified_argmax_kernel<<<blocks, kProxyThreads, 0, stream>>>(
        candidate_logits, candidate_ids, omitted_upper, omitted_upper_stride, absolute_margin,
        relative_margin, best_ids, best_logits, gaps, certified, rows, pool);
    return hipGetLastError();
}

hipError_t launch_dflash2_psq8_selected_token_logits(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    const int32_t* token_ids,
    float* out_logits,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    hipStream_t stream) {
    if (weight_codes == nullptr || weight_scales == nullptr || activation_codes == nullptr ||
        activation_scales == nullptr || token_ids == nullptr || out_logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const dim3 grid(1u, rows, 1u);
    psq8_candidate_rerank_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        weight_codes, weight_scales, activation_codes, activation_scales, token_ids, out_logits,
        rows, 1u, k_padded, weight_scale_stride_bytes, activation_code_stride_bytes,
        activation_scale_stride_bytes, 1u);
    return hipGetLastError();
}

hipError_t launch_dflash2_verify_max_other_upper(
    const float* coarse_logits,
    const float* activation_l2,
    const float* error_l2,
    const int32_t* proposed_token_ids,
    float* out_max_other_upper,
    uint32_t rows,
    uint32_t vocab,
    hipStream_t stream) {
    if (coarse_logits == nullptr || activation_l2 == nullptr || error_l2 == nullptr ||
        proposed_token_ids == nullptr || out_max_other_upper == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || vocab == 0u)
        return hipSuccess;
    verify_max_other_upper_kernel<<<rows, 256u, 0, stream>>>(
        coarse_logits, activation_l2, error_l2, proposed_token_ids, out_max_other_upper, vocab);
    return hipGetLastError();
}

hipError_t launch_dflash2_verify_direct_certificate(
    const float* exact_proposed_logits,
    const float* max_other_upper,
    float absolute_margin,
    float relative_margin,
    uint8_t* certified,
    uint32_t rows,
    hipStream_t stream) {
    if (exact_proposed_logits == nullptr || max_other_upper == nullptr || certified == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    const uint32_t threads = 256u;
    const uint32_t blocks = (rows + threads - 1u) / threads;
    verify_direct_certificate_kernel<<<blocks, threads, 0, stream>>>(
        exact_proposed_logits, max_other_upper, absolute_margin, relative_margin, certified,
        rows);
    return hipGetLastError();
}

hipError_t launch_dflash2_verify_decision_sufficiency(
    const int32_t* draft_tokens,
    const uint8_t* row_state,
    const int32_t* certified_tokens,
    uint32_t num_drafts,
    uint32_t* out_decision_ready,
    uint32_t* out_num_accepted,
    uint32_t* out_correction_valid,
    int32_t* out_correction_token,
    uint32_t* out_bonus_valid,
    int32_t* out_bonus_token,
    int32_t* out_first_required_unknown_row,
    hipStream_t stream) {
    if (draft_tokens == nullptr || row_state == nullptr || certified_tokens == nullptr ||
        out_decision_ready == nullptr || out_num_accepted == nullptr ||
        out_correction_valid == nullptr || out_correction_token == nullptr ||
        out_bonus_valid == nullptr || out_bonus_token == nullptr ||
        out_first_required_unknown_row == nullptr)
        return hipErrorInvalidValue;
    if (num_drafts == 0u)
        return hipSuccess;
    verify_decision_sufficiency_kernel<<<1u, 1u, 0, stream>>>(
        draft_tokens, row_state, certified_tokens, num_drafts, out_decision_ready,
        out_num_accepted, out_correction_valid, out_correction_token, out_bonus_valid,
        out_bonus_token, out_first_required_unknown_row);
    return hipGetLastError();
}

}  // namespace ps::kernel
