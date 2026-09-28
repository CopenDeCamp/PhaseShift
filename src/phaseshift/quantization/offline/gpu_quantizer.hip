#include <phaseshift/quantization/offline/gpu_quantizer.h>
#include <hip/hip_runtime.h>

namespace ps::quantization::fpx {

namespace {

__global__ void fp4_weighted_row_sigma2_kernel(
    const uint8_t* __restrict__ src,
    uint64_t row_stride_bytes,
    int dtype,
    uint64_t logical_k,
    uint64_t padded_k,
    float* __restrict__ sigma2)
{
    const uint64_t rr = blockIdx.x;
    float sum = 0.0f;
    for (uint64_t j = 0; j < padded_k; ++j) {
        float x = 0.0f;
        if (j < logical_k) {
            x = gpu_load_elem(
                src + rr * row_stride_bytes + j * (dtype == kGpuSrcF32 ? 4u : 2u), dtype);
        }
        sum = __fadd_rn(sum, __fmul_rn(x, x));
    }
    sigma2[rr] = sum / static_cast<float>(padded_k);
}

constexpr uint32_t kPsqBlockThreads = 128;
constexpr int kPsqCandidates = 65;

__global__ void psq4_quantize_chunk_kernel(
    const uint8_t* __restrict__ src,
    uint64_t row_stride_bytes,
    int dtype,
    uint64_t k,
    uint64_t rb,
    const float* __restrict__ quant_weights,
    const float* __restrict__ sigma2,
    uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales)
{
    const uint32_t ib = blockIdx.x;
    const uint32_t rr = blockIdx.y;
    const uint32_t t = threadIdx.x;

    __shared__ float xs[32];
    __shared__ float mw[32];
    __shared__ float errs[kPsqCandidates];
    __shared__ float s_out;
    __shared__ float s_max;
    __shared__ uint8_t q[32];

    if (t < 32) {
        const uint64_t j = static_cast<uint64_t>(ib) * 32u + t;
        float x = 0.0f;
        if (j < k) {
            const uint64_t off = static_cast<uint64_t>(rr) * row_stride_bytes +
                                 j * (dtype == kGpuSrcF32 ? 4u : 2u);
            x = gpu_load_elem(src + off, dtype);
        }
        xs[t] = x;
        if (quant_weights != nullptr) {
            const float qw = quant_weights[static_cast<uint64_t>(ib) * 32u + t];
            mw[t] = (isfinite(qw) && qw > 0.0f)
                        ? __fmul_rn(qw, sqrtf(__fadd_rn(sigma2[rr], __fmul_rn(x, x))))
                        : 0.0f;
        } else {
            mw[t] = 1.0f;
        }
    }
    __syncthreads();

    if (t < 32) {
        float m = isfinite(xs[t]) ? fabsf(xs[t]) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_xor(m, off);
            if (o > m) m = o;
        }
        if (t == 0) s_max = m;
    }
    __syncthreads();

    const float smaxv = s_max;
    if (t < kPsqCandidates) {
        float e = FLT_MAX;
        if (smaxv > 0.0f && isfinite(smaxv)) {
            const int j = static_cast<int>(t) - 32;
            const float s = gpu_bf16_round((smaxv / 10.0f) * exp2f(static_cast<float>(j) / 64.0f));
            if (s > 0.0f && isfinite(s)) {
                const float inv = 1.0f / s;
                float err = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const uint8_t c = gpu_psq4_code_index(xs[i], inv);
                    const float y = gpu_cb10_code_to_f32(c) * s;
                    const float d = xs[i] - y;
                    err = __fadd_rn(err, __fmul_rn(__fmul_rn(mw[i], d), d));
                }
                e = err;
            }
        }
        errs[t] = e;
    }
    __syncthreads();

    if (t == 0) {
        float best = FLT_MAX;
        int bestj = -1;
        for (int i = 0; i < kPsqCandidates; ++i) {
            if (errs[i] < best) { best = errs[i]; bestj = i; }
        }
        float s = 0.0f;
        if (bestj >= 0) {
            s = gpu_bf16_round((smaxv / 10.0f) *
                               exp2f(static_cast<float>(bestj - 32) / 64.0f));
        }
        const uint16_t b = __bfloat16_as_ushort(__float2bfloat16(s));
        scales[static_cast<uint64_t>(rr) * rb * 2u + static_cast<uint64_t>(ib) * 2u] =
            static_cast<uint8_t>(b & 0xFFu);
        scales[static_cast<uint64_t>(rr) * rb * 2u + static_cast<uint64_t>(ib) * 2u + 1u] =
            static_cast<uint8_t>(b >> 8u);
        s_out = s;
    }
    __syncthreads();

    const float s = s_out;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (t < 32) {
        q[t] = inv > 0.0f ? gpu_psq4_code_index(xs[t], inv) : 0;
    }
    __syncthreads();
    if (t < 16) {
        codes[static_cast<uint64_t>(rr) * rb * 16u + static_cast<uint64_t>(ib) * 16u + t] =
            static_cast<uint8_t>(q[t] | (q[t + 16u] << 4u));
    }
}

__global__ void psq8_quantize_chunk_kernel(
    const uint8_t* __restrict__ src,
    uint64_t row_stride_bytes,
    int dtype,
    uint64_t k,
    uint64_t rb,
    const float* __restrict__ quant_weights,
    const float* __restrict__ sigma2,
    uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales)
{
    const uint32_t ib = blockIdx.x;
    const uint32_t rr = blockIdx.y;
    const uint32_t t = threadIdx.x;

    __shared__ float xs[32];
    __shared__ float mw[32];
    __shared__ float errs[kPsqCandidates];
    __shared__ float s_out;
    __shared__ float s_max;

    if (t < 32) {
        const uint64_t j = static_cast<uint64_t>(ib) * 32u + t;
        float x = 0.0f;
        if (j < k) {
            const uint64_t off = static_cast<uint64_t>(rr) * row_stride_bytes +
                                 j * (dtype == kGpuSrcF32 ? 4u : 2u);
            x = gpu_load_elem(src + off, dtype);
        }
        xs[t] = x;
        if (quant_weights != nullptr) {
            const float qw = quant_weights[static_cast<uint64_t>(ib) * 32u + t];
            mw[t] = (isfinite(qw) && qw > 0.0f)
                        ? __fmul_rn(qw, sqrtf(__fadd_rn(sigma2[rr], __fmul_rn(x, x))))
                        : 0.0f;
        } else {
            mw[t] = 1.0f;
        }
    }
    __syncthreads();

    if (t < 32) {
        float m = isfinite(xs[t]) ? fabsf(xs[t]) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_xor(m, off);
            if (o > m) m = o;
        }
        if (t == 0) s_max = m;
    }
    __syncthreads();

    const float smaxv = s_max;
    if (t < kPsqCandidates) {
        float e = FLT_MAX;
        if (smaxv > 0.0f && isfinite(smaxv)) {
            const int j = static_cast<int>(t) - 32;
            const float s = gpu_bf16_round((smaxv / 448.0f) * exp2f(static_cast<float>(j) / 64.0f));
            if (s > 0.0f && isfinite(s)) {
                const float inv = 1.0f / s;
                float err = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const uint8_t c = gpu_e4m3_encode(xs[i] * inv);
                    const float y = gpu_e4m3_decode(c) * s;
                    const float d = xs[i] - y;
                    err = __fadd_rn(err, __fmul_rn(__fmul_rn(mw[i], d), d));
                }
                e = err;
            }
        }
        errs[t] = e;
    }
    __syncthreads();

    if (t == 0) {
        float best = FLT_MAX;
        int bestj = -1;
        for (int i = 0; i < kPsqCandidates; ++i) {
            if (errs[i] < best) { best = errs[i]; bestj = i; }
        }
        float s = 0.0f;
        if (bestj >= 0) {
            s = gpu_bf16_round((smaxv / 448.0f) *
                               exp2f(static_cast<float>(bestj - 32) / 64.0f));
        }
        const uint16_t b = __bfloat16_as_ushort(__float2bfloat16(s));
        scales[static_cast<uint64_t>(rr) * rb * 2u + static_cast<uint64_t>(ib) * 2u] =
            static_cast<uint8_t>(b & 0xFFu);
        scales[static_cast<uint64_t>(rr) * rb * 2u + static_cast<uint64_t>(ib) * 2u + 1u] =
            static_cast<uint8_t>(b >> 8u);
        s_out = s;
    }
    __syncthreads();

    const float s = s_out;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (t < 32) {
        codes[static_cast<uint64_t>(rr) * rb * 32u + static_cast<uint64_t>(ib) * 32u + t] =
            inv > 0.0f ? gpu_e4m3_encode(xs[t] * inv) : 0;
    }
}

__global__ void mxfp4_quantize_chunk_kernel(
    const uint8_t* __restrict__ src,
    uint64_t row_stride_bytes,
    int dtype,
    uint64_t k,
    uint64_t rb,
    uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales)
{
    const uint32_t ib = blockIdx.x;
    const uint32_t rr = blockIdx.y;
    const uint32_t t = threadIdx.x;

    __shared__ float xs[32];
    __shared__ uint8_t q[32];
    __shared__ uint8_t s_byte;
    __shared__ float s_scale;

    if (t < 32) {
        const uint64_t j = static_cast<uint64_t>(ib) * 32u + t;
        float x = 0.0f;
        if (j < k) {
            const uint64_t off = static_cast<uint64_t>(rr) * row_stride_bytes +
                                 j * (dtype == kGpuSrcF32 ? 4u : 2u);
            x = gpu_load_elem(src + off, dtype);
        }
        xs[t] = x;
    }
    __syncthreads();

    if (t < 32) {
        float m = isfinite(xs[t]) ? fabsf(xs[t]) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_xor(m, off);
            if (o > m) m = o;
        }
        if (t == 0) {
            s_byte = gpu_e8m0_encode(m);
            s_scale = (m > 0.0f) ? gpu_e8m0_decode(s_byte) : 0.0f;
            scales[static_cast<uint64_t>(rr) * rb + ib] = s_byte;
        }
    }
    __syncthreads();

    const float scale = s_scale;
    if (t < 32) {
        q[t] = (scale > 0.0f) ? gpu_e2m1_encode(xs[t] / scale) : 0u;
    }
    __syncthreads();
    if (t < 16) {
        codes[static_cast<uint64_t>(rr) * rb * 16u + static_cast<uint64_t>(ib) * 16u + t] =
            static_cast<uint8_t>(q[2u * t] | (q[2u * t + 1u] << 4u));
    }
}

constexpr uint32_t kFp8BlockThreads = 128;

__global__ void fp8_quantize_chunk_kernel(
    const uint8_t* __restrict__ src,
    uint64_t row_stride_bytes,
    int dtype,
    uint64_t logical_k,
    uint64_t padded_k,
    uint64_t k_tiles,
    uint64_t chunk_rows,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales)
{
    const uint64_t kb = blockIdx.x;
    const uint64_t g = blockIdx.y;
    const uint32_t t = threadIdx.x;

    const uint64_t r0 = g * 128u;
    const uint64_t r1 = min(chunk_rows, r0 + static_cast<uint64_t>(128u));
    const uint64_t k0 = kb * 128u;
    const uint64_t k1 = min(padded_k, k0 + static_cast<uint64_t>(128u));

    __shared__ float red[kFp8BlockThreads];
    float local = 0.0f;
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = src + r * row_stride_bytes;
        for (uint64_t kk = k0 + t; kk < k1; kk += kFp8BlockThreads) {
            if (kk >= logical_k) continue;
            const float x = gpu_load_elem(
                row + kk * (dtype == kGpuSrcF32 ? 4u : 2u), dtype);
            const float a = fabsf(x);
            if (isfinite(a) && a > local) local = a;
        }
    }
    red[t] = local;
    __syncthreads();
    #pragma unroll
    for (int off = 64; off > 0; off >>= 1) {
        if (t < static_cast<uint32_t>(off)) {
            if (red[t + off] > red[t]) red[t] = red[t + off];
        }
        __syncthreads();
    }
    const float amax = red[0];
    const float scale = (amax > 0.0f && isfinite(amax)) ? (amax / 448.0f) : 0.0f;
    if (t == 0) scales[g * k_tiles + kb] = scale;
    __syncthreads();

    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* srow = src + r * row_stride_bytes;
        uint8_t* crow = codes + r * padded_k + k0;
        for (uint64_t kk = k0 + t; kk < k1; kk += kFp8BlockThreads) {
            if (scale == 0.0f || kk >= logical_k) {
                crow[kk - k0] = 0u;
                continue;
            }
            const float x = gpu_load_elem(
                srow + kk * (dtype == kGpuSrcF32 ? 4u : 2u), dtype);
            crow[kk - k0] = isfinite(x) ? gpu_e4m3_encode(x / scale) : 0u;
        }
    }
}

}  // namespace

Status gpu_quantizer_create(GpuQuantizer& q, std::size_t max_input_bytes,
                            std::size_t max_codes_bytes, std::size_t max_scales_bytes,
                            std::size_t max_quant_weights_bytes, std::size_t max_chunk_rows) {
    q.max_input_bytes = max_input_bytes;
    q.max_codes_bytes = max_codes_bytes;
    q.max_scales_bytes = max_scales_bytes;
    q.max_quant_weights_bytes = max_quant_weights_bytes;
    q.max_chunk_rows = max_chunk_rows;
    for (int i = 0; i < kGpuQuantSlots; ++i) {
        hipStream_t stream = nullptr;
        if (hipStreamCreate(&stream) != hipSuccess) {
            gpu_quantizer_destroy(q);
            return Status::hip_error("hipStreamCreate", "gpu_quantizer stream", __FILE__, __LINE__);
        }
        q.streams[i] = stream;
        hipEvent_t ev = nullptr;
        if (hipEventCreate(&ev) != hipSuccess) {
            gpu_quantizer_destroy(q);
            return Status::hip_error("hipEventCreate", "gpu_quantizer event", __FILE__, __LINE__);
        }
        q.events[i] = ev;
        if (hipHostMalloc(&q.h_in[i], max_input_bytes, hipHostMallocDefault) != hipSuccess ||
            hipHostMalloc(&q.h_codes[i], max_codes_bytes, hipHostMallocDefault) != hipSuccess ||
            hipHostMalloc(&q.h_scales[i], max_scales_bytes, hipHostMallocDefault) != hipSuccess) {
            gpu_quantizer_destroy(q);
            return Status::insufficient_memory("gpu_quantizer pinned host memory", __FILE__, __LINE__);
        }
        if (hipMalloc(&q.d_in[i], max_input_bytes) != hipSuccess ||
            hipMalloc(&q.d_codes[i], max_codes_bytes) != hipSuccess ||
            hipMalloc(&q.d_scales[i], max_scales_bytes) != hipSuccess ||
            hipMalloc(&q.d_row_sigma2[i], max_chunk_rows * sizeof(float)) != hipSuccess) {
            gpu_quantizer_destroy(q);
            return Status::insufficient_memory("gpu_quantizer device memory", __FILE__, __LINE__);
        }
    }
    if (max_quant_weights_bytes > 0) {
        if (hipMalloc(&q.d_quant_weights, max_quant_weights_bytes) != hipSuccess) {
            gpu_quantizer_destroy(q);
            return Status::insufficient_memory("gpu_quantizer quant_weights device memory",
                                               __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Status gpu_quantizer_destroy(GpuQuantizer& q) {
    bool first_error = false;
    for (int i = 0; i < kGpuQuantSlots; ++i) {
        if (q.h_in[i]) {
            const hipError_t r = hipHostFree(q.h_in[i]);
            q.h_in[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.h_codes[i]) {
            const hipError_t r = hipHostFree(q.h_codes[i]);
            q.h_codes[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.h_scales[i]) {
            const hipError_t r = hipHostFree(q.h_scales[i]);
            q.h_scales[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.d_in[i]) {
            const hipError_t r = hipFree(q.d_in[i]);
            q.d_in[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.d_codes[i]) {
            const hipError_t r = hipFree(q.d_codes[i]);
            q.d_codes[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.d_scales[i]) {
            const hipError_t r = hipFree(q.d_scales[i]);
            q.d_scales[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.d_row_sigma2[i]) {
            const hipError_t r = hipFree(q.d_row_sigma2[i]);
            q.d_row_sigma2[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.events[i]) {
            const hipError_t r = hipEventDestroy(static_cast<hipEvent_t>(q.events[i]));
            q.events[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
        if (q.streams[i]) {
            const hipError_t r = hipStreamDestroy(static_cast<hipStream_t>(q.streams[i]));
            q.streams[i] = nullptr;
            if (r != hipSuccess) first_error = true;
        }
    }
    if (q.d_quant_weights) {
        const hipError_t r = hipFree(q.d_quant_weights);
        q.d_quant_weights = nullptr;
        if (r != hipSuccess) first_error = true;
    }
    q.max_input_bytes = 0;
    q.max_codes_bytes = 0;
    q.max_scales_bytes = 0;
    q.max_quant_weights_bytes = 0;
    q.max_chunk_rows = 0;
    if (first_error) {
        return Status::hip_error("gpu_quantizer_destroy", "one or more resources failed to release", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

void* gpu_quantizer_host_input(GpuQuantizer& q, int slot) {
    return q.h_in[slot];
}

const void* gpu_quantizer_host_codes(const GpuQuantizer& q, int slot) {
    return q.h_codes[slot];
}

const void* gpu_quantizer_host_scales(const GpuQuantizer& q, int slot) {
    return q.h_scales[slot];
}

Status gpu_quantizer_upload_quant_weights(GpuQuantizer& q, const float* quant_weights,
                                          uint64_t count) {
    if (count == 0) {
        return Status::invalid_argument("gpu_quantizer_upload_quant_weights empty", __FILE__, __LINE__);
    }
    const std::size_t nbytes = static_cast<std::size_t>(count) * sizeof(float);
    if (nbytes > q.max_quant_weights_bytes) {
        return Status::invalid_argument("gpu_quantizer_upload_quant_weights exceeds workspace",
                                        __FILE__, __LINE__);
    }
    if (!q.d_quant_weights) {
        return Status::invalid_argument("gpu_quantizer_upload_quant_weights no workspace",
                                        __FILE__, __LINE__);
    }
    const hipError_t r = hipMemcpy(q.d_quant_weights, quant_weights, nbytes, hipMemcpyHostToDevice);
    if (r != hipSuccess) {
        return Status::hip_error("hipMemcpy quant_weights H2D",
                                 "gpu_quantizer_upload_quant_weights", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_quantize_chunk_submit(GpuQuantizer& q, int slot, GpuQuantKind kind, bool weighted,
                                 const void* src_host, const GpuTensorSource& src,
                                 uint64_t row_begin, uint64_t row_end) {
    if (slot < 0 || slot >= kGpuQuantSlots) {
        return Status::out_of_range("gpu_quantize_chunk_submit slot", __FILE__, __LINE__);
    }
    if (row_begin >= row_end) {
        return Status::invalid_argument("gpu_quantize_chunk_submit empty row range", __FILE__, __LINE__);
    }
    if (row_end > src.rows) {
        return Status::out_of_range("gpu_quantize_chunk_submit row range", __FILE__, __LINE__);
    }
    uint64_t codes_per_block = 32;
    uint64_t scale_bytes_per_block = 2;
    if (kind == GpuQuantKind::Psq4) {
        codes_per_block = 16;
    } else if (kind == GpuQuantKind::Mxfp4) {
        codes_per_block = 16;
        scale_bytes_per_block = 1;
    } else if (kind == GpuQuantKind::Fp8Block128) {
        codes_per_block = 128;
        scale_bytes_per_block = 4;
    }
    const uint64_t chunk_rows = row_end - row_begin;
    const uint64_t input_bytes = chunk_rows * src.logical_k * src.stride;
    const uint64_t codes_bytes = chunk_rows * src.blocks_per_row * codes_per_block;
    const uint64_t scales_bytes = (kind == GpuQuantKind::Fp8Block128)
        ? src.blocks_per_row * scale_bytes_per_block
        : chunk_rows * src.blocks_per_row * scale_bytes_per_block;
    if (input_bytes > q.max_input_bytes || codes_bytes > q.max_codes_bytes ||
        scales_bytes > q.max_scales_bytes) {
        return Status::invalid_argument("gpu_quantize_chunk_submit chunk exceeds workspace", __FILE__, __LINE__);
    }

    hipStream_t stream = static_cast<hipStream_t>(q.streams[slot]);
    if (hipMemcpyAsync(q.d_in[slot], src_host, input_bytes, hipMemcpyHostToDevice, stream) != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync H2D", "gpu_quantize_chunk_submit", __FILE__, __LINE__);
    }

    const uint64_t row_stride_bytes = src.logical_k * src.stride;
    const int dtype = static_cast<int>(src.dtype);
    if (kind == GpuQuantKind::Mxfp4) {
        dim3 grid(static_cast<unsigned>(src.blocks_per_row), static_cast<unsigned>(chunk_rows));
        mxfp4_quantize_chunk_kernel<<<grid, 32, 0, stream>>>(
            static_cast<const uint8_t*>(q.d_in[slot]), row_stride_bytes, dtype,
            src.logical_k, src.blocks_per_row,
            static_cast<uint8_t*>(q.d_codes[slot]), static_cast<uint8_t*>(q.d_scales[slot]));
    } else if (kind == GpuQuantKind::Fp8Block128) {
        const uint64_t groups = (chunk_rows + 127u) / 128u;
        dim3 grid(static_cast<unsigned>(src.blocks_per_row), static_cast<unsigned>(groups));
        fp8_quantize_chunk_kernel<<<grid, kFp8BlockThreads, 0, stream>>>(
            static_cast<const uint8_t*>(q.d_in[slot]), row_stride_bytes, dtype,
            src.logical_k, src.padded_k, src.blocks_per_row, chunk_rows,
            static_cast<uint8_t*>(q.d_codes[slot]), static_cast<float*>(q.d_scales[slot]));
    } else if (kind == GpuQuantKind::Psq4 || kind == GpuQuantKind::Psq8) {
        const float* qw = nullptr;
        const float* sig = nullptr;
        if (weighted) {
            if (!q.d_quant_weights || !q.d_row_sigma2[slot]) {
                return Status::invalid_argument("gpu_quantize_chunk_submit weighted workspace missing",
                                                __FILE__, __LINE__);
            }
            dim3 sgrid(static_cast<unsigned>(chunk_rows));
            dim3 sblock(1);
            fp4_weighted_row_sigma2_kernel<<<sgrid, sblock, 0, stream>>>(
                static_cast<const uint8_t*>(q.d_in[slot]), row_stride_bytes, dtype,
                src.logical_k, src.padded_k,
                static_cast<float*>(q.d_row_sigma2[slot]));
            qw = static_cast<const float*>(q.d_quant_weights);
            sig = static_cast<const float*>(q.d_row_sigma2[slot]);
        }
        dim3 grid(static_cast<unsigned>(src.blocks_per_row), static_cast<unsigned>(chunk_rows));
        if (kind == GpuQuantKind::Psq4) {
            psq4_quantize_chunk_kernel<<<grid, kPsqBlockThreads, 0, stream>>>(
                static_cast<const uint8_t*>(q.d_in[slot]), row_stride_bytes, dtype,
                src.logical_k, src.blocks_per_row, qw, sig,
                static_cast<uint8_t*>(q.d_codes[slot]), static_cast<uint8_t*>(q.d_scales[slot]));
        } else {
            psq8_quantize_chunk_kernel<<<grid, kPsqBlockThreads, 0, stream>>>(
                static_cast<const uint8_t*>(q.d_in[slot]), row_stride_bytes, dtype,
                src.logical_k, src.blocks_per_row, qw, sig,
                static_cast<uint8_t*>(q.d_codes[slot]), static_cast<uint8_t*>(q.d_scales[slot]));
        }
    }
    if (hipGetLastError() != hipSuccess) {
        return Status::hip_error("kernel launch", "gpu_quantize_chunk_submit", __FILE__, __LINE__);
    }

    if (hipMemcpyAsync(q.h_codes[slot], q.d_codes[slot], codes_bytes, hipMemcpyDeviceToHost, stream) != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync D2H codes", "gpu_quantize_chunk_submit", __FILE__, __LINE__);
    }
    if (hipMemcpyAsync(q.h_scales[slot], q.d_scales[slot], scales_bytes, hipMemcpyDeviceToHost, stream) != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync D2H scales", "gpu_quantize_chunk_submit", __FILE__, __LINE__);
    }
    if (hipEventRecord(static_cast<hipEvent_t>(q.events[slot]), stream) != hipSuccess) {
        return Status::hip_error("hipEventRecord", "gpu_quantize_chunk_submit", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_quantize_chunk_sync(GpuQuantizer& q, int slot) {
    if (slot < 0 || slot >= kGpuQuantSlots) {
        return Status::out_of_range("gpu_quantize_chunk_sync slot", __FILE__, __LINE__);
    }
    const hipError_t r = hipEventSynchronize(static_cast<hipEvent_t>(q.events[slot]));
    if (r != hipSuccess) {
        return Status::hip_error("hipEventSynchronize", "gpu_quantize_chunk_sync", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace ps::quantization::fpx
