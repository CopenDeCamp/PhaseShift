#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/quantization/psq/psq4_kv.h>
#include <phaseshift/quantization/psq/psq8_kv.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cfloat>

namespace ps::kernel {

namespace {
namespace op_kv_append_bf16 {

using bf16 = __hip_bfloat16;

__device__ __forceinline__ void kv_append_bf16_body(const KvAppendCommonArgs& a,
                                                   bf16* k_pool, bf16* v_pool) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t row = blockIdx.x / a.kv_heads;
    const uint32_t head = blockIdx.x % a.kv_heads;

    __shared__ uint64_t token_base_shared;
    __shared__ uint32_t valid_shared;

    if (threadIdx.x == 0u) {
        uint64_t token_base = 0;
        uint32_t valid = 1u;
        const uint32_t slot = a.row_sequence_slots[row];
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t pos = a.row_positions[row];
            const uint32_t logical_block = pos / a.page_tokens;
            const uint32_t offset = pos % a.page_tokens;
            if (logical_block >= a.block_table_stride) {
                valid = 0u;
            } else {
                const uint32_t page = a.block_tables[static_cast<size_t>(slot) * a.block_table_stride + logical_block];
                if (page >= a.num_pages) {
                    valid = 0u;
                } else {
                    token_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer
                               + static_cast<uint64_t>(page) * a.elems_per_page
                               + static_cast<uint64_t>(offset) * a.elems_per_token;
                }
            }
        }
        token_base_shared = token_base;
        valid_shared = valid;
    }
    __syncthreads();
    if (valid_shared == 0u) return;

    const uint64_t dst_base = token_base_shared + static_cast<uint64_t>(head) * head_dim;
    const bf16* k_src = reinterpret_cast<const bf16*>(a.k_input) + static_cast<size_t>(row) * a.k_row_stride + static_cast<size_t>(head) * head_dim;
    const bf16* v_src = reinterpret_cast<const bf16*>(a.v_input) + static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(head) * head_dim;
    bf16* k_dst = k_pool + dst_base;
    bf16* v_dst = v_pool + dst_base;

    const uint32_t full = head_dim >> 2;
    const uint32_t rem = head_dim & 3u;
    const bool aligned8 =
        (reinterpret_cast<uintptr_t>(k_src) & 7u) == 0u &&
        (reinterpret_cast<uintptr_t>(k_dst) & 7u) == 0u &&
        (reinterpret_cast<uintptr_t>(v_src) & 7u) == 0u &&
        (reinterpret_cast<uintptr_t>(v_dst) & 7u) == 0u;

    if (static_cast<uint32_t>(threadIdx.x) < full) {
        const uint32_t i = static_cast<uint32_t>(threadIdx.x) << 2;
        if (aligned8) {
            const uint64_t kb = *reinterpret_cast<const uint64_t*>(k_src + i);
            const uint64_t vb = *reinterpret_cast<const uint64_t*>(v_src + i);
            *reinterpret_cast<uint64_t*>(k_dst + i) = kb;
            *reinterpret_cast<uint64_t*>(v_dst + i) = vb;
        } else {
            k_dst[i] = k_src[i]; k_dst[i + 1] = k_src[i + 1];
            k_dst[i + 2] = k_src[i + 2]; k_dst[i + 3] = k_src[i + 3];
            v_dst[i] = v_src[i]; v_dst[i + 1] = v_src[i + 1];
            v_dst[i + 2] = v_src[i + 2]; v_dst[i + 3] = v_src[i + 3];
        }
    } else if (static_cast<uint32_t>(threadIdx.x) == full && rem != 0u) {
        const uint32_t i = full << 2;
        for (uint32_t j = 0; j < rem; ++j) {
            k_dst[i + j] = k_src[i + j];
            v_dst[i + j] = v_src[i + j];
        }
    }
}

}  // namespace op_kv_append_bf16
}  // namespace

extern "C" __global__ __launch_bounds__(256)
void phaseshift_qwen35_kv_append_bf16(::ps::kernel::KvAppendBf16Args args) {
    op_kv_append_bf16::kv_append_bf16_body(
        args.common, reinterpret_cast<op_kv_append_bf16::bf16*>(args.k_pool),
        reinterpret_cast<op_kv_append_bf16::bf16*>(args.v_pool));
}

hipError_t launch_kv_append_bf16(const KvAppendCommonArgs& common, bf16_t* k_pool, bf16_t* v_pool, hipStream_t stream) {
    if (common.rows == 0u || common.head_dim == 0u || common.kv_heads == 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.kv_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const uint32_t chunks = (common.head_dim + 3u) / 4u;
    const uint32_t block = chunks <= 32u ? 32u : chunks <= 64u ? 64u : chunks <= 128u ? 128u : 256u;
    KvAppendBf16Args args{};
    args.common = common;
    args.k_pool = k_pool;
    args.v_pool = v_pool;
    phaseshift_qwen35_kv_append_bf16<<<dim3(static_cast<uint32_t>(grid_x)), block,
                                        0, stream>>>(args);
    return hipGetLastError();
}

namespace {
namespace op_kv_append_fp8 {

using bf16 = __hip_bfloat16;
using fp8 = ::ps::qwen35::fp8e4m3_storage_t;

constexpr float kFp8E4M3Max = 448.0f;

__device__ __forceinline__ float wave_reduce_max(float x) {
    x = fmaxf(x, __shfl_down(x, 16, 32));
    x = fmaxf(x, __shfl_down(x, 8, 32));
    x = fmaxf(x, __shfl_down(x, 4, 32));
    x = fmaxf(x, __shfl_down(x, 2, 32));
    x = fmaxf(x, __shfl_down(x, 1, 32));
    return x;
}

__device__ __forceinline__ fp8 cvt_fp8(float v) {
    return __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
}

__global__ __launch_bounds__(256)
void impl(const KvAppendCommonArgs a, fp8* k_pool, fp8* v_pool, float* k_scale_pool, float* v_scale_pool) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t row = blockIdx.x / a.kv_heads;
    const uint32_t head = blockIdx.x % a.kv_heads;

    __shared__ uint64_t token_base_shared;
    __shared__ uint32_t page_shared;
    __shared__ uint32_t offset_shared;
    __shared__ uint32_t valid_shared;
    __shared__ float k_wave_max[8];
    __shared__ float v_wave_max[8];
    __shared__ float k_scale_shared;
    __shared__ float v_scale_shared;

    if (threadIdx.x == 0u) {
        uint64_t token_base = 0;
        uint32_t page = 0, offset = 0, valid = 1u;
        const uint32_t slot = a.row_sequence_slots[row];
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t pos = a.row_positions[row];
            const uint32_t logical_block = pos / a.page_tokens;
            const uint32_t off = pos % a.page_tokens;
            if (logical_block >= a.block_table_stride) {
                valid = 0u;
            } else {
                const uint32_t pg = a.block_tables[static_cast<size_t>(slot) * a.block_table_stride + logical_block];
                if (pg >= a.num_pages) {
                    valid = 0u;
                } else {
                    page = pg;
                    offset = off;
                    token_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer
                               + static_cast<uint64_t>(page) * a.elems_per_page
                               + static_cast<uint64_t>(offset) * a.elems_per_token;
                }
            }
        }
        token_base_shared = token_base;
        page_shared = page;
        offset_shared = offset;
        valid_shared = valid;
    }
    __syncthreads();
    if (valid_shared == 0u) return;

    const uint64_t dst_base = token_base_shared + static_cast<uint64_t>(head) * head_dim;
    const bf16* k_src = reinterpret_cast<const bf16*>(a.k_input) + static_cast<size_t>(row) * a.k_row_stride + static_cast<size_t>(head) * head_dim;
    const bf16* v_src = reinterpret_cast<const bf16*>(a.v_input) + static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(head) * head_dim;
    fp8* k_dst = k_pool + dst_base;
    fp8* v_dst = v_pool + dst_base;

    float k_max = 0.0f;
    float v_max = 0.0f;
    const uint32_t step = static_cast<uint32_t>(blockDim.x) << 2;
    for (uint32_t base = static_cast<uint32_t>(threadIdx.x) << 2; base < head_dim; base += step) {
        const uint32_t n = base + 4u <= head_dim ? 4u : (head_dim - base);
        for (uint32_t j = 0; j < n; ++j) {
            const float kf = __bfloat162float(k_src[base + j]);
            const float vf = __bfloat162float(v_src[base + j]);
            k_max = fmaxf(k_max, fabsf(kf));
            v_max = fmaxf(v_max, fabsf(vf));
        }
    }
    const uint32_t num_waves = static_cast<uint32_t>(blockDim.x) >> 5;
    const uint32_t wave_id = static_cast<uint32_t>(threadIdx.x) >> 5;
    k_max = wave_reduce_max(k_max);
    v_max = wave_reduce_max(v_max);
    if ((static_cast<uint32_t>(threadIdx.x) & 31u) == 0u) {
        k_wave_max[wave_id] = k_max;
        v_wave_max[wave_id] = v_max;
    }
    __syncthreads();
    if (wave_id == 0u) {
        const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
        float kf = (tid < num_waves) ? k_wave_max[tid] : 0.0f;
        float vf = (tid < num_waves) ? v_wave_max[tid] : 0.0f;
        kf = wave_reduce_max(kf);
        vf = wave_reduce_max(vf);
        if (threadIdx.x == 0u) {
            k_scale_shared = (kf == 0.0f) ? 1.0f : kf / kFp8E4M3Max;
            v_scale_shared = (vf == 0.0f) ? 1.0f : vf / kFp8E4M3Max;
        }
    }
    __syncthreads();
    const float k_scale = k_scale_shared;
    const float v_scale = v_scale_shared;

    if (threadIdx.x == 0u) {
        const size_t si = ((static_cast<size_t>(a.layer) * a.num_pages + page_shared) * a.page_tokens + offset_shared) * a.pool_kv_heads + (head + a.kv_head_offset);
        k_scale_pool[si] = k_scale;
        v_scale_pool[si] = v_scale;
    }

    for (uint32_t base = static_cast<uint32_t>(threadIdx.x) << 2; base < head_dim; base += step) {
        const uint32_t n = base + 4u <= head_dim ? 4u : (head_dim - base);
        fp8 kb[4];
        fp8 vb[4];
        for (uint32_t j = 0; j < n; ++j) {
            kb[j] = cvt_fp8(__bfloat162float(k_src[base + j]) / k_scale);
            vb[j] = cvt_fp8(__bfloat162float(v_src[base + j]) / v_scale);
        }
        if (n == 4u &&
            (reinterpret_cast<uintptr_t>(k_dst + base) & 3u) == 0u &&
            (reinterpret_cast<uintptr_t>(v_dst + base) & 3u) == 0u) {
            const uint32_t kp = static_cast<uint32_t>(kb[0])
                | (static_cast<uint32_t>(kb[1]) << 8)
                | (static_cast<uint32_t>(kb[2]) << 16)
                | (static_cast<uint32_t>(kb[3]) << 24);
            const uint32_t vp = static_cast<uint32_t>(vb[0])
                | (static_cast<uint32_t>(vb[1]) << 8)
                | (static_cast<uint32_t>(vb[2]) << 16)
                | (static_cast<uint32_t>(vb[3]) << 24);
            *reinterpret_cast<uint32_t*>(k_dst + base) = kp;
            *reinterpret_cast<uint32_t*>(v_dst + base) = vp;
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                k_dst[base + j] = kb[j];
                v_dst[base + j] = vb[j];
            }
        }
    }
}

}  // namespace op_kv_append_fp8

namespace op_kv_append_psq4 {

using bf16 = __hip_bfloat16;

constexpr uint32_t kThreads = 256u;

__device__ __forceinline__ float wave_max_p4(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x, 8, 32));
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

__device__ __forceinline__ void wave_sum2_p4(float& x, float& y) {
    x += __shfl_xor(x, 16, 32); y += __shfl_xor(y, 16, 32);
    x += __shfl_xor(x, 8, 32);  y += __shfl_xor(y, 8, 32);
    x += __shfl_xor(x, 4, 32);  y += __shfl_xor(y, 4, 32);
    x += __shfl_xor(x, 2, 32);  y += __shfl_xor(y, 2, 32);
    x += __shfl_xor(x, 1, 32);  y += __shfl_xor(y, 1, 32);
}

template <bool kLsq1>
__device__ __forceinline__ void encode_lane(
    float x, float amax, float& scale_out, uint8_t& code_out) {
    using namespace ::ps::quantization::psq;
    float s = (amax > 0.0f) ? psq_bf16_round(amax * 0.1f) : 0.0f;
    uint8_t c = (s > 0.0f) ? cb10_encode(x, 1.0f / s) : 0u;
    if (kLsq1) {
        const float q = cb10_code_to_f32(c);
        float num = x * q;
        float den = q * q;
        wave_sum2_p4(num, den);
        const float sl = (den > 0.0f) ? (num / den) : 0.0f;
        float s1 = psq_bf16_round(sl);
        if (!(s1 > 0.0f)) {
            s1 = 0.0f;
            c = 0u;
        } else {
            c = cb10_encode(x, 1.0f / s1);
        }
        s = s1;
    }
    scale_out = s;
    code_out = c;
}

template <bool kLsq1>
__global__ __launch_bounds__(kThreads)
void impl(const KvAppendCommonArgs a,
          uint8_t* k_code_pool, uint8_t* v_code_pool,
          bf16* k_scale_pool, bf16* v_scale_pool,
          uint32_t blocks_per_head) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t row = blockIdx.x / a.kv_heads;
    const uint32_t head = blockIdx.x % a.kv_heads;

    __shared__ uint64_t token_base_shared;
    __shared__ uint32_t page_shared;
    __shared__ uint32_t offset_shared;
    __shared__ uint32_t valid_shared;

    if (threadIdx.x == 0u) {
        uint64_t token_base = 0;
        uint32_t page = 0, offset = 0, valid = 1u;
        const uint32_t slot = a.row_sequence_slots[row];
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t pos = a.row_positions[row];
            const uint32_t logical_block = pos / a.page_tokens;
            const uint32_t off = pos % a.page_tokens;
            if (logical_block >= a.block_table_stride) {
                valid = 0u;
            } else {
                const uint32_t pg = a.block_tables[static_cast<size_t>(slot) * a.block_table_stride + logical_block];
                if (pg >= a.num_pages) {
                    valid = 0u;
                } else {
                    page = pg;
                    offset = off;
                    token_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer
                               + static_cast<uint64_t>(page) * a.elems_per_page
                               + static_cast<uint64_t>(offset) * a.elems_per_token;
                }
            }
        }
        token_base_shared = token_base;
        page_shared = page;
        offset_shared = offset;
        valid_shared = valid;
    }
    __syncthreads();
    if (valid_shared == 0u) return;

    const uint32_t wave = static_cast<uint32_t>(threadIdx.x) >> 5;
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;

    const bf16* k_src = reinterpret_cast<const bf16*>(a.k_input) +
        static_cast<size_t>(row) * a.k_row_stride + static_cast<size_t>(head) * head_dim;
    const bf16* v_src = reinterpret_cast<const bf16*>(a.v_input) +
        static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(head) * head_dim;
    const uint32_t d = wave * 32u + lane;
    const float kx = __bfloat162float(k_src[d]);
    const float vx = __bfloat162float(v_src[d]);

    const uint64_t bi = (((static_cast<uint64_t>(a.layer) * a.num_pages + page_shared)
                          * a.page_tokens + offset_shared) * a.pool_kv_heads + (head + a.kv_head_offset))
                        * blocks_per_head + wave;

    const float k_amax = wave_max_p4(fabsf(kx));
    const float v_amax = wave_max_p4(fabsf(vx));

    float k_scale = 0.0f;
    float v_scale = 0.0f;
    uint8_t k_code = 0u;
    uint8_t v_code = 0u;
    encode_lane<kLsq1>(kx, k_amax, k_scale, k_code);
    encode_lane<kLsq1>(vx, v_amax, v_scale, v_code);

    const int hi_lane = static_cast<int>((lane + 16u) & 31u);
    const uint32_t k_hi = static_cast<uint32_t>(__shfl(static_cast<int>(k_code), hi_lane, 32));
    const uint32_t v_hi = static_cast<uint32_t>(__shfl(static_cast<int>(v_code), hi_lane, 32));
    if (lane < 16u) {
        k_code_pool[bi * ::ps::qwen35::kPsq4CodeBytesPerBlock + lane] =
            static_cast<uint8_t>((k_code & 0xFu) | ((k_hi & 0xFu) << 4u));
        v_code_pool[bi * ::ps::qwen35::kPsq4CodeBytesPerBlock + lane] =
            static_cast<uint8_t>((v_code & 0xFu) | ((v_hi & 0xFu) << 4u));
    }
    if (lane == 0u) {
        k_scale_pool[bi] = __float2bfloat16(k_scale);
        v_scale_pool[bi] = __float2bfloat16(v_scale);
    }
}

}  // namespace op_kv_append_psq4

namespace op_kv_append_psq8 {

using bf16 = __hip_bfloat16;

constexpr uint32_t kThreads = 256u;

__device__ __forceinline__ float wave_max_p8(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x, 8, 32));
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

__device__ __forceinline__ void wave_sum2_p8(float& x, float& y) {
    x += __shfl_xor(x, 16, 32); y += __shfl_xor(y, 16, 32);
    x += __shfl_xor(x, 8, 32);  y += __shfl_xor(y, 8, 32);
    x += __shfl_xor(x, 4, 32);  y += __shfl_xor(y, 4, 32);
    x += __shfl_xor(x, 2, 32);  y += __shfl_xor(y, 2, 32);
    x += __shfl_xor(x, 1, 32);  y += __shfl_xor(y, 1, 32);
}

template <bool kLsq1>
__device__ __forceinline__ void encode_lane(
    float x, float amax, float& scale_out, uint8_t& code_out) {
    using namespace ::ps::quantization::psq;
    float s = (amax > 0.0f) ? psq_bf16_round(amax * (1.0f / 448.0f)) : 0.0f;
    uint8_t c = (s > 0.0f) ? psq8_e4m3_encode_u8(x * (1.0f / s)) : 0u;
    if (kLsq1) {
        const float q = psq8_e4m3_decode_f32(c);
        float num = x * q;
        float den = q * q;
        wave_sum2_p8(num, den);
        const float sl = (den > 0.0f) ? (num / den) : 0.0f;
        float s1 = psq_bf16_round(sl);
        if (!(s1 > 0.0f)) {
            s1 = 0.0f;
            c = 0u;
        } else {
            c = psq8_e4m3_encode_u8(x * (1.0f / s1));
        }
        s = s1;
    }
    scale_out = s;
    code_out = c;
}

template <bool kLsq1>
__global__ __launch_bounds__(kThreads)
void impl(const KvAppendCommonArgs a,
          uint8_t* k_code_pool, uint8_t* v_code_pool,
          bf16* k_scale_pool, bf16* v_scale_pool,
          uint32_t blocks_per_head) {
    const uint32_t head_dim = a.head_dim;
    const uint32_t row = blockIdx.x / a.kv_heads;
    const uint32_t head = blockIdx.x % a.kv_heads;

    __shared__ uint64_t token_base_shared;
    __shared__ uint32_t page_shared;
    __shared__ uint32_t offset_shared;
    __shared__ uint32_t valid_shared;

    if (threadIdx.x == 0u) {
        uint64_t token_base = 0;
        uint32_t page = 0, offset = 0, valid = 1u;
        const uint32_t slot = a.row_sequence_slots[row];
        if (slot >= a.max_sequences) {
            valid = 0u;
        } else {
            const uint32_t pos = a.row_positions[row];
            const uint32_t logical_block = pos / a.page_tokens;
            const uint32_t off = pos % a.page_tokens;
            if (logical_block >= a.block_table_stride) {
                valid = 0u;
            } else {
                const uint32_t pg = a.block_tables[static_cast<size_t>(slot) * a.block_table_stride + logical_block];
                if (pg >= a.num_pages) {
                    valid = 0u;
                } else {
                    page = pg;
                    offset = off;
                    token_base = static_cast<uint64_t>(a.layer) * a.elems_per_layer
                               + static_cast<uint64_t>(page) * a.elems_per_page
                               + static_cast<uint64_t>(offset) * a.elems_per_token;
                }
            }
        }
        token_base_shared = token_base;
        page_shared = page;
        offset_shared = offset;
        valid_shared = valid;
    }
    __syncthreads();
    if (valid_shared == 0u) return;

    const uint32_t wave = static_cast<uint32_t>(threadIdx.x) >> 5;
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;

    const bf16* k_src = reinterpret_cast<const bf16*>(a.k_input) +
        static_cast<size_t>(row) * a.k_row_stride + static_cast<size_t>(head) * head_dim;
    const bf16* v_src = reinterpret_cast<const bf16*>(a.v_input) +
        static_cast<size_t>(row) * a.v_row_stride + static_cast<size_t>(head) * head_dim;
    const uint32_t d = wave * 32u + lane;
    const float kx = __bfloat162float(k_src[d]);
    const float vx = __bfloat162float(v_src[d]);

    const uint64_t bi = (((static_cast<uint64_t>(a.layer) * a.num_pages + page_shared)
                          * a.page_tokens + offset_shared) * a.pool_kv_heads + (head + a.kv_head_offset))
                        * blocks_per_head + wave;

    const float k_amax = wave_max_p8(fabsf(kx));
    const float v_amax = wave_max_p8(fabsf(vx));

    float k_scale = 0.0f;
    float v_scale = 0.0f;
    uint8_t k_code = 0u;
    uint8_t v_code = 0u;
    encode_lane<kLsq1>(kx, k_amax, k_scale, k_code);
    encode_lane<kLsq1>(vx, v_amax, v_scale, v_code);

    k_code_pool[bi * ::ps::qwen35::kPsq8CodeBytesPerBlock + lane] = k_code;
    v_code_pool[bi * ::ps::qwen35::kPsq8CodeBytesPerBlock + lane] = v_code;
    if (lane == 0u) {
        k_scale_pool[bi] = __float2bfloat16(k_scale);
        v_scale_pool[bi] = __float2bfloat16(v_scale);
    }
}

}  // namespace op_kv_append_psq8
}  // namespace

hipError_t launch_kv_append_fp8_e4m3(const KvAppendCommonArgs& common,
    ::ps::qwen35::fp8e4m3_storage_t* k_pool, ::ps::qwen35::fp8e4m3_storage_t* v_pool,
    float* k_scale_pool, float* v_scale_pool, hipStream_t stream) {
    if (common.rows == 0u || common.head_dim == 0u || common.kv_heads == 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_pool == nullptr || v_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.kv_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const uint32_t block = common.head_dim <= 128u ? 32u
                        : common.head_dim <= 256u ? 64u
                        : common.head_dim <= 512u ? 128u
                        : 256u;
    const KvAppendCommonArgs a = common;
    op_kv_append_fp8::impl<<<dim3(static_cast<uint32_t>(grid_x)), block, 0, stream>>>(
        a, reinterpret_cast<op_kv_append_fp8::fp8*>(k_pool),
        reinterpret_cast<op_kv_append_fp8::fp8*>(v_pool), k_scale_pool, v_scale_pool);
    return hipGetLastError();
}

hipError_t launch_kv_append_psq4(const KvAppendCommonArgs& common,
    uint8_t* k_code_pool, uint8_t* v_code_pool,
    bf16_t* k_scale_pool, bf16_t* v_scale_pool,
    uint32_t blocks_per_head,
    ::ps::qwen35::Psq4ScaleEstimator estimator,
    hipStream_t stream) {
    using namespace op_kv_append_psq4;
    if (common.rows == 0u || common.head_dim == 0u || common.kv_heads == 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_code_pool == nullptr || v_code_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    if (common.head_dim != ::ps::qwen35::kPsq4BlockValues * blocks_per_head)
        return hipErrorNotSupported;
    if (blocks_per_head != kThreads / 32u)
        return hipErrorNotSupported;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.kv_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const KvAppendCommonArgs a = common;
    if (estimator == ::ps::qwen35::Psq4ScaleEstimator::Lsq1) {
        op_kv_append_psq4::impl<true><<<dim3(static_cast<uint32_t>(grid_x)), kThreads, 0, stream>>>(
            a, k_code_pool, v_code_pool,
            reinterpret_cast<op_kv_append_psq4::bf16*>(k_scale_pool),
            reinterpret_cast<op_kv_append_psq4::bf16*>(v_scale_pool), blocks_per_head);
    } else {
        op_kv_append_psq4::impl<false><<<dim3(static_cast<uint32_t>(grid_x)), kThreads, 0, stream>>>(
            a, k_code_pool, v_code_pool,
            reinterpret_cast<op_kv_append_psq4::bf16*>(k_scale_pool),
            reinterpret_cast<op_kv_append_psq4::bf16*>(v_scale_pool), blocks_per_head);
    }
    return hipGetLastError();
}

hipError_t launch_kv_append_psq8(const KvAppendCommonArgs& common,
    uint8_t* k_code_pool, uint8_t* v_code_pool,
    bf16_t* k_scale_pool, bf16_t* v_scale_pool,
    uint32_t blocks_per_head,
    ::ps::qwen35::Psq8ScaleEstimator estimator,
    hipStream_t stream) {
    using namespace op_kv_append_psq8;
    if (common.rows == 0u || common.head_dim == 0u || common.kv_heads == 0u)
        return hipErrorInvalidValue;
    if (common.page_tokens == 0u || common.block_table_stride == 0u || common.num_pages == 0u)
        return hipErrorInvalidValue;
    if (common.k_input == nullptr || common.v_input == nullptr ||
        common.row_sequence_slots == nullptr || common.row_positions == nullptr ||
        common.block_tables == nullptr || k_code_pool == nullptr || v_code_pool == nullptr ||
        k_scale_pool == nullptr || v_scale_pool == nullptr)
        return hipErrorInvalidValue;
    if (common.head_dim != ::ps::qwen35::kPsq8BlockValues * blocks_per_head)
        return hipErrorNotSupported;
    if (blocks_per_head != kThreads / 32u)
        return hipErrorNotSupported;
    const uint64_t grid_x = static_cast<uint64_t>(common.rows) * common.kv_heads;
    if (grid_x == 0u) return hipErrorInvalidValue;
    const KvAppendCommonArgs a = common;
    if (estimator == ::ps::qwen35::Psq8ScaleEstimator::Lsq1) {
        op_kv_append_psq8::impl<true><<<dim3(static_cast<uint32_t>(grid_x)), kThreads, 0, stream>>>(
            a, k_code_pool, v_code_pool,
            reinterpret_cast<op_kv_append_psq8::bf16*>(k_scale_pool),
            reinterpret_cast<op_kv_append_psq8::bf16*>(v_scale_pool), blocks_per_head);
    } else {
        op_kv_append_psq8::impl<false><<<dim3(static_cast<uint32_t>(grid_x)), kThreads, 0, stream>>>(
            a, k_code_pool, v_code_pool,
            reinterpret_cast<op_kv_append_psq8::bf16*>(k_scale_pool),
            reinterpret_cast<op_kv_append_psq8::bf16*>(v_scale_pool), blocks_per_head);
    }
    return hipGetLastError();
}

}  // namespace ps::kernel
