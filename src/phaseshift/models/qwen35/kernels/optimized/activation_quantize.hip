#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include "detail/activation_quantize_device.h"
#include <phaseshift/quantization/fpx/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

__device__ __forceinline__ void activation_quantize_a8_body(
    const ::ps::bf16_t* __restrict__ input,
    int8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t rows,
    const uint32_t k,
    const uint32_t k_padded,
    const uint32_t input_row_stride,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride) {
    const uint32_t blocks = k_padded >> 5;
    const uint32_t total_waves = rows * blocks;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t local_wave = threadIdx.x >> 5;
    const uint32_t gwave = blockIdx.x * (blockDim.x >> 5) + local_wave;
    if (gwave >= total_waves)
        return;
    const uint32_t m = gwave / blocks;
    const uint32_t blk = gwave % blocks;
    const uint32_t eidx = blk * 32u + lane;
    const __hip_bfloat16* xb = reinterpret_cast<const __hip_bfloat16*>(input);
    const float v = (eidx < k)
        ? __bfloat162float(xb[static_cast<size_t>(m) * input_row_stride + eidx])
        : 0.f;
    const float val = std::isfinite(v) ? v : 0.f;
    float av = std::fabs(val);
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1)
        av = fmaxf(av, __shfl_xor_sync(0xffffffffULL, av, off));
    const float scale = av > 0.f ? av / 127.0f : 0.f;
    int8_t c;
    if (scale <= 0.f) {
        c = 0;
    } else {
        const float inv = 1.0f / scale;
        int ci = static_cast<int>(nearbyintf(val * inv));
        if (ci > 127) ci = 127;
        if (ci < -127) ci = -127;
        c = static_cast<int8_t>(ci);
    }
    codes[static_cast<size_t>(m >> 4) * (static_cast<size_t>(code_row_stride) * 16u) +
          static_cast<size_t>(blk) * 512u + static_cast<size_t>(lane >> 4) * 256u +
          static_cast<size_t>(m & 15u) * 16u + static_cast<size_t>(lane & 15u)] = c;
    if (lane == 0u)
        scales[static_cast<size_t>(m) * (scale_row_stride / 4u) + blk] = scale;
}

}

extern "C" __global__ __launch_bounds__(kThreadsPerBlock)
void phaseshift_qwen35_activation_quantize_a8(ActivationQuantizeA8Args args) {
    activation_quantize_a8_body(
        reinterpret_cast<const ::ps::bf16_t*>(args.input),
        reinterpret_cast<int8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), args.rows, args.k, args.k_padded,
        args.input_row_stride, args.code_row_stride, args.scale_row_stride);
}

uint32_t activation_quantize_a8_block_size() noexcept {
    return kThreadsPerBlock;
}

uint32_t activation_quantize_a8_grid(uint32_t rows, uint32_t k_padded) noexcept {
    const uint32_t blocks = k_padded >> 5;
    const uint64_t total_waves = static_cast<uint64_t>(rows) * blocks;
    if (total_waves == 0)
        return 0u;
    const uint32_t waves_per_block = kThreadsPerBlock >> 5;
    return static_cast<uint32_t>(
        (total_waves + waves_per_block - 1u) / waves_per_block);
}

hipError_t launch_activation_quantize_a8(
    const ::ps::bf16_t* input,
    int8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream) {
    const uint32_t grid = activation_quantize_a8_grid(rows, k_padded);
    if (grid == 0u)
        return hipSuccess;
    ActivationQuantizeA8Args args{};
    args.input = input;
    args.codes = codes;
    args.scales = scales;
    args.rows = rows;
    args.k = k;
    args.k_padded = k_padded;
    args.input_row_stride = input_row_stride;
    args.code_row_stride = code_row_stride_bytes;
    args.scale_row_stride = scale_row_stride_bytes;
    phaseshift_qwen35_activation_quantize_a8<<<grid, kThreadsPerBlock, 0, stream>>>(args);
    return hipGetLastError();
}

namespace {

constexpr uint32_t kThreadsPerRow = 256u;

__global__ void __launch_bounds__(kThreadsPerRow) activation_quantize_e4m3_kernel(
    const ::ps::bf16_t* __restrict__ input,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t k,
    const uint32_t k_padded,
    const uint32_t input_row_stride,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride) {
    const uint32_t m = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const __hip_bfloat16* xb = reinterpret_cast<const __hip_bfloat16*>(input) +
        static_cast<size_t>(m) * input_row_stride;

    float peak = 0.0f;
    for (uint32_t i = t; i < k; i += kThreadsPerRow) {
        const float v = __bfloat162float(xb[i]);
        if (isfinite(v)) {
            const float av = std::fabs(v);
            if (av > peak) peak = av;
        }
    }
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
    __shared__ float warp_max[kThreadsPerRow / 32u];
    if ((t & 31u) == 0u) warp_max[t >> 5u] = peak;
    __syncthreads();
    if (t < 32u) {
        float v = (t < kThreadsPerRow / 32u) ? warp_max[t] : 0.0f;
        #pragma unroll
        for (uint32_t off = kThreadsPerRow / 64u; off > 0u; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffULL, v, off));
        if (t == 0u)
            scales[static_cast<size_t>(m) * (scale_row_stride / 4u)] =
                v > 0.0f ? v / 448.0f : 0.0f;
    }
    __syncthreads();
    const float scale = scales[static_cast<size_t>(m) * (scale_row_stride / 4u)];
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    uint8_t* crow = codes +
        static_cast<size_t>(m >> 4) * (static_cast<size_t>(code_row_stride) * 16u) +
        static_cast<size_t>(m & 15u) * 16u;
    for (uint32_t i = t; i < k_padded; i += kThreadsPerRow) {
        float v = 0.0f;
        if (i < k) {
            const float x = __bfloat162float(xb[i]);
            v = isfinite(x) ? x : 0.0f;
        }
        const uint8_t c = scale > 0.0f
            ? ::ps::quantization::fpx::device::e4m3_encode_u8(v * inv)
            : 0u;
        const uint32_t lane = i & 31u;
        crow[static_cast<size_t>(i >> 5) * 512u +
             static_cast<size_t>(lane >> 4) * 256u + (lane & 15u)] = c;
    }
}

template <uint32_t K, uint32_t KP>
__global__ void __launch_bounds__(kThreadsPerRow)
activation_quantize_e4m3_vec_kernel(
    const ::ps::bf16_t* __restrict__ input,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t input_row_stride,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride) {
    detail::activation_quantize_e4m3_vec_row_body<K, KP>(
        input, codes, scales, blockIdx.x, input_row_stride, code_row_stride,
        scale_row_stride);
}

}  // namespace

bool activation_quantize_e4m3_vec_supported(
    const ::ps::bf16_t* input, const uint8_t* codes,
    uint32_t k, uint32_t k_padded, uint32_t input_row_stride) {
    if (k != k_padded || (k & 7u) != 0u)
        return false;
    if (k != 5120u && k != 6144u && k != 17408u)
        return false;
    if ((reinterpret_cast<uintptr_t>(input) & 15u) != 0u)
        return false;
    if ((reinterpret_cast<uintptr_t>(codes) & 7u) != 0u)
        return false;
    if ((input_row_stride & 7u) != 0u)
        return false;
    return true;
}

hipError_t launch_activation_quantize_e4m3_generic(
    const ::ps::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream) {
    if (rows == 0 || k_padded == 0)
        return hipSuccess;
    activation_quantize_e4m3_kernel<<<rows, kThreadsPerRow, 0, stream>>>(
        input, codes, scales, k, k_padded,
        input_row_stride, code_row_stride_bytes, scale_row_stride_bytes);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_qwen35_activation_quantize_e4m3_k5120(
    ActivationQuantizeE4m3K5120Args args);

hipError_t launch_activation_quantize_e4m3(
    const ::ps::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream,
    bool* specialized) {
    if (specialized != nullptr) *specialized = false;
    if (rows == 0 || k_padded == 0)
        return hipSuccess;
    if (activation_quantize_e4m3_vec_supported(input, codes, k, k_padded, input_row_stride)) {
        switch (k) {
            case 5120u: {
                ActivationQuantizeE4m3K5120Args k5120{};
                k5120.input = input;
                k5120.codes = codes;
                k5120.scales = scales;
                k5120.input_row_stride = input_row_stride;
                k5120.code_row_stride = code_row_stride_bytes;
                k5120.scale_row_stride = scale_row_stride_bytes;
                phaseshift_qwen35_activation_quantize_e4m3_k5120
                    <<<rows, 256, 0, stream>>>(k5120);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            }
            case 6144u:
                activation_quantize_e4m3_vec_kernel<6144u, 6144u>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        input, codes, scales, input_row_stride,
                        code_row_stride_bytes, scale_row_stride_bytes);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            case 17408u:
                activation_quantize_e4m3_vec_kernel<17408u, 17408u>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        input, codes, scales, input_row_stride,
                        code_row_stride_bytes, scale_row_stride_bytes);
                if (specialized != nullptr) *specialized = true;
                return hipGetLastError();
            default:
                break;
        }
    }
    return launch_activation_quantize_e4m3_generic(
        input, codes, scales, rows, k, k_padded,
        input_row_stride, code_row_stride_bytes, scale_row_stride_bytes, stream);
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_qwen35_activation_quantize_e4m3_k5120(
    ActivationQuantizeE4m3K5120Args args) {
    detail::activation_quantize_e4m3_vec_row_body<5120u, 5120u>(
        reinterpret_cast<const ::ps::bf16_t*>(args.input),
        reinterpret_cast<uint8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), blockIdx.x, args.input_row_stride,
        args.code_row_stride, args.scale_row_stride);
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_qwen35_activation_quantize_e4m3_k6144(
    ActivationQuantizeE4m3KArgs args) {
    detail::activation_quantize_e4m3_vec_row_body<6144u, 6144u>(
        reinterpret_cast<const ::ps::bf16_t*>(args.input),
        reinterpret_cast<uint8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), blockIdx.x, args.input_row_stride,
        args.code_row_stride, args.scale_row_stride);
}

extern "C" __global__ __launch_bounds__(256)
void phaseshift_qwen35_activation_quantize_e4m3_k17408(
    ActivationQuantizeE4m3KArgs args) {
    detail::activation_quantize_e4m3_vec_row_body<17408u, 17408u>(
        reinterpret_cast<const ::ps::bf16_t*>(args.input),
        reinterpret_cast<uint8_t*>(args.codes),
        reinterpret_cast<float*>(args.scales), blockIdx.x, args.input_row_stride,
        args.code_row_stride, args.scale_row_stride);
}

uint32_t activation_quantize_e4m3_block_size() noexcept { return 256u; }

namespace {

constexpr uint32_t kThreadsPerRowI8 = 256u;

__global__ void __launch_bounds__(kThreadsPerRowI8) activation_quantize_i8_row_kernel(
    const ::ps::bf16_t* __restrict__ input,
    int8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t k,
    const uint32_t k_padded,
    const uint32_t input_row_stride,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride) {
    const uint32_t m = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const __hip_bfloat16* xb = reinterpret_cast<const __hip_bfloat16*>(input) +
        static_cast<size_t>(m) * input_row_stride;
    float peak = 0.0f;
    for (uint32_t i = t; i < k; i += kThreadsPerRowI8) {
        const float v = __bfloat162float(xb[i]);
        if (isfinite(v)) {
            const float av = std::fabs(v);
            if (av > peak) peak = av;
        }
    }
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
    __shared__ float warp_max_i8[kThreadsPerRowI8 / 32u];
    if ((t & 31u) == 0u) warp_max_i8[t >> 5u] = peak;
    __syncthreads();
    if (t < 32u) {
        float v = (t < kThreadsPerRowI8 / 32u) ? warp_max_i8[t] : 0.0f;
        #pragma unroll
        for (uint32_t off = kThreadsPerRowI8 / 64u; off > 0u; off >>= 1u)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffULL, v, off));
        if (t == 0u)
            scales[static_cast<size_t>(m) * (scale_row_stride / 4u)] =
                v > 0.0f ? v / 127.0f : 0.0f;
    }
    __syncthreads();
    const float scale = scales[static_cast<size_t>(m) * (scale_row_stride / 4u)];
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int8_t* crow = codes +
        static_cast<size_t>(m >> 4) * (static_cast<size_t>(code_row_stride) * 16u) +
        static_cast<size_t>(m & 15u) * 16u;
    for (uint32_t i = t; i < k_padded; i += kThreadsPerRowI8) {
        float v = 0.0f;
        if (i < k) {
            const float x = __bfloat162float(xb[i]);
            v = isfinite(x) ? x : 0.0f;
        }
        int8_t c = 0;
        if (scale > 0.0f) {
            float ci = nearbyintf(v * inv);
            if (ci > 127.0f) ci = 127.0f;
            if (ci < -127.0f) ci = -127.0f;
            c = static_cast<int8_t>(ci);
        }
        const uint32_t lane = i & 31u;
        crow[static_cast<size_t>(i >> 5) * 512u +
             static_cast<size_t>(lane >> 4) * 256u + (lane & 15u)] = c;
    }
}

}

hipError_t launch_activation_quantize_i8_row(
    const ::ps::bf16_t* input,
    int8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream) {
    if (rows == 0 || k_padded == 0)
        return hipSuccess;
    activation_quantize_i8_row_kernel<<<rows, kThreadsPerRowI8, 0, stream>>>(
        input, codes, scales, k, k_padded,
        input_row_stride, code_row_stride_bytes, scale_row_stride_bytes);
    return hipGetLastError();
}

}
