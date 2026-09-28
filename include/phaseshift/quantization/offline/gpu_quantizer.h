#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/io/safetensors_reader.h>
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <cfloat>
#include <cmath>
#endif

namespace ps::quantization::fpx {

constexpr std::size_t kGpuQuantChunkTargetInputBytes = 64ull * 1024 * 1024;
constexpr std::size_t kGpuQuantChunkMaxRows = 8192;
constexpr int kGpuQuantSlots = 2;

enum class GpuQuantKind {
    Psq4,
    Psq8,
    Mxfp4,
    Fp8Block128,
};

struct GpuQuantizer {
    void* streams[kGpuQuantSlots] = {};
    void* events[kGpuQuantSlots] = {};
    void* d_in[kGpuQuantSlots] = {};
    void* d_codes[kGpuQuantSlots] = {};
    void* d_scales[kGpuQuantSlots] = {};
    void* h_in[kGpuQuantSlots] = {};
    void* h_codes[kGpuQuantSlots] = {};
    void* h_scales[kGpuQuantSlots] = {};
    void* d_quant_weights = nullptr;
    void* d_row_sigma2[kGpuQuantSlots] = {};
    std::size_t max_input_bytes = 0;
    std::size_t max_codes_bytes = 0;
    std::size_t max_scales_bytes = 0;
    std::size_t max_quant_weights_bytes = 0;
    std::size_t max_chunk_rows = 0;
};

struct GpuTensorSource {
    ps::io::SType dtype = ps::io::SType::BF16;
    std::size_t stride = 2;
    uint64_t logical_k = 0;
    uint64_t padded_k = 0;
    uint64_t blocks_per_row = 0;
    uint64_t rows = 0;
};

Status gpu_quantizer_create(GpuQuantizer& q, std::size_t max_input_bytes,
                            std::size_t max_codes_bytes, std::size_t max_scales_bytes,
                            std::size_t max_quant_weights_bytes, std::size_t max_chunk_rows);
Status gpu_quantizer_destroy(GpuQuantizer& q);

void* gpu_quantizer_host_input(GpuQuantizer& q, int slot);
const void* gpu_quantizer_host_codes(const GpuQuantizer& q, int slot);
const void* gpu_quantizer_host_scales(const GpuQuantizer& q, int slot);

Status gpu_quantizer_upload_quant_weights(GpuQuantizer& q, const float* quant_weights,
                                          uint64_t count);
Status gpu_quantize_chunk_submit(GpuQuantizer& q, int slot, GpuQuantKind kind, bool weighted,
                                 const void* src_host, const GpuTensorSource& src,
                                 uint64_t row_begin, uint64_t row_end);
Status gpu_quantize_chunk_sync(GpuQuantizer& q, int slot);

#if defined(__HIPCC__)

enum {
    kGpuSrcF16 = 0,
    kGpuSrcF32 = 1,
    kGpuSrcBF16 = 2,
};

__device__ __forceinline__ float gpu_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (static_cast<uint32_t>(h) & 0x7C00u) >> 10;
    const uint32_t man = static_cast<uint32_t>(h) & 0x03FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) {
            u = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x0400u) == 0);
            u = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    return __uint_as_float(u);
}

__device__ __forceinline__ float gpu_load_elem(const uint8_t* p, int dtype) {
    if (dtype == kGpuSrcF32) {
        uint32_t u;
        __builtin_memcpy(&u, p, 4);
        return __uint_as_float(u);
    }
    uint16_t h;
    __builtin_memcpy(&h, p, 2);
    if (dtype == kGpuSrcBF16) {
        return __uint_as_float(static_cast<uint32_t>(h) << 16);
    }
    return gpu_fp16_to_fp32(h);
}

__device__ __forceinline__ float gpu_bf16_round(float v) {
    const uint16_t b = __bfloat16_as_ushort(__float2bfloat16(v));
    const uint32_t f = static_cast<uint32_t>(b) << 16u;
    return __uint_as_float(f);
}

__device__ __forceinline__ float gpu_cb10_mag(uint32_t mag) {
    switch (mag & 7u) {
        case 0u: return 0.0f;
        case 1u: return 1.0f;
        case 2u: return 2.0f;
        case 3u: return 3.0f;
        case 4u: return 4.0f;
        case 5u: return 6.0f;
        case 6u: return 8.0f;
        default: return 10.0f;
    }
}

__device__ __forceinline__ uint8_t gpu_psq4_code_index(float x, float inv_scale) {
    if (!isfinite(x)) {
        return 0;
    }
    const float a = fabsf(x * inv_scale);
    uint32_t mag;
    if (a <= 0.5f) mag = 0;
    else if (a <= 1.5f) mag = 1;
    else if (a <= 2.5f) mag = 2;
    else if (a <= 3.5f) mag = 3;
    else if (a <= 5.0f) mag = 4;
    else if (a <= 7.0f) mag = 5;
    else if (a <= 9.0f) mag = 6;
    else mag = 7;
    return static_cast<uint8_t>(mag | (x < 0.0f ? 8u : 0u));
}

__device__ __forceinline__ float gpu_cb10_code_to_f32(uint8_t code) {
    const float mag = gpu_cb10_mag(static_cast<uint32_t>(code & 7u));
    return (code & 8u) ? -mag : mag;
}

__device__ __forceinline__ float gpu_e4m3_decode(uint8_t b) {
    const uint32_t sg = static_cast<uint32_t>(b) >> 7;
    const uint32_t ex = (static_cast<uint32_t>(b) >> 3) & 0xFu;
    const uint32_t mn = static_cast<uint32_t>(b) & 7u;
    const float mag = (ex == 0u)
        ? ldexpf(static_cast<float>(mn), -9)
        : ldexpf(1.0f + static_cast<float>(mn) / 8.0f, static_cast<int>(ex) - 7);
    return sg ? -mag : mag;
}

__device__ __forceinline__ int gpu_e4m3_round(int fl, float rem) {
    if (rem < 0.5f) return fl;
    if (rem > 0.5f) return fl + 1;
    return (fl & 1) ? fl + 1 : fl;
}

__device__ __forceinline__ float gpu_e2m1_mag(uint32_t idx) {
    switch (idx & 7u) {
        case 0u: return 0.0f;
        case 1u: return 0.5f;
        case 2u: return 1.0f;
        case 3u: return 1.5f;
        case 4u: return 2.0f;
        case 5u: return 3.0f;
        case 6u: return 4.0f;
        default: return 6.0f;
    }
}

__device__ __forceinline__ uint8_t gpu_e2m1_encode(float x) {
    if (!isfinite(x)) return 0u;
    const uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
    const float a = fabsf(x);
    if (!(a > 0.0f)) return sign;
    if (a >= gpu_e2m1_mag(7)) return static_cast<uint8_t>(sign | 0x7u);
    int lo = 0;
    for (int i = 0; i < 8; ++i) {
        if (gpu_e2m1_mag(static_cast<uint32_t>(i)) <= a) lo = i;
        else break;
    }
    if (lo >= 7) return static_cast<uint8_t>(sign | 0x7u);
    const float vlo = gpu_e2m1_mag(static_cast<uint32_t>(lo));
    const float vhi = gpu_e2m1_mag(static_cast<uint32_t>(lo + 1));
    const float dlo = a - vlo;
    const float dhi = vhi - a;
    int pick;
    if (dhi < dlo) pick = lo + 1;
    else if (dlo < dhi) pick = lo;
    else pick = (((lo + 1) & 1) == 0) ? lo + 1 : lo;
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(pick));
}

__device__ __forceinline__ uint8_t gpu_e8m0_encode(float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) return 0u;
    const float raw = amax / 6.0f;
    int e = 0;
    frexpf(raw, &e);
    if (raw == ldexpf(0.5f, e)) e -= 1;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float gpu_e8m0_decode(uint8_t byte) {
    return ldexpf(1.0f, static_cast<int>(byte) - 127);
}

__device__ __forceinline__ uint8_t gpu_e4m3_encode(float v) {
    if (!isfinite(v)) {
        return (v != v) ? 0x7Fu : static_cast<uint8_t>((v < 0.0f) ? 0xFEu : 0x7Eu);
    }
    const uint8_t sign = (v < 0.0f || (v == 0.0f && 1.0f / v < 0.0f)) ? 0x80u : 0x00u;
    const float a = fabsf(v);
    if (a == 0.0f) {
        return sign;
    }
    if (a >= 448.0f) {
        return static_cast<uint8_t>(sign | 0x7Eu);
    }
    if (a < 0x1p-6f) {
        const float fx = a * 512.0f;
        const int fl = static_cast<int>(floorf(fx));
        const int man = gpu_e4m3_round(fl, fx - static_cast<float>(fl));
        if (man <= 0) return sign;
        if (man == 8) return static_cast<uint8_t>(sign | 8u);
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(man));
    }
    int e = static_cast<int>(floorf(log2f(a))) + 7;
    const float fx = (a * exp2f(static_cast<float>(7 - e)) - 1.0f) * 8.0f;
    const int fl = static_cast<int>(floorf(fx));
    int man = gpu_e4m3_round(fl, fx - static_cast<float>(fl));
    if (man >= 8) { man = 0; e += 1; }
    if (e >= 16) return static_cast<uint8_t>(sign | 0x7Eu);
    return static_cast<uint8_t>(sign | static_cast<uint8_t>((e << 3) | man));
}

#endif  // defined(__HIPCC__)

}  // namespace ps::quantization::fpx
