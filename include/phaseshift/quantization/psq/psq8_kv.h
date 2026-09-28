#pragma once

#include <phaseshift/quantization/psq/psq4_kv.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace ps::quantization::psq {

PSQ_DEVICE_HD inline float psq8_e4m3_decode_f32(uint8_t x) {
#if defined(__HIP_DEVICE_COMPILE__)
    const __half_raw hr =
        __hip_cvt_fp8_to_halfraw(static_cast<__hip_fp8_storage_t>(x), __HIP_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
#else
    return ::ps::quantization::fpx::e4m3_decode_f32(x);
#endif
}

PSQ_DEVICE_HD inline uint8_t psq8_e4m3_encode_u8(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<uint8_t>(__hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3));
#else
    return ::ps::quantization::fpx::e4m3_encode_u8(x);
#endif
}

PSQ_DEVICE_HD inline void psq8_maxabs_encode_block(
    const float* x, uint32_t n, float& scale_out, uint8_t* codes_out) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    float s = 0.0f;
    if (amax > 0.0f && amax < 3.402823466e+38f) s = psq_bf16_round(amax * (1.0f / 448.0f));
    if (!(s > 0.0f)) {
        scale_out = 0.0f;
        for (uint32_t i = 0; i < n; ++i) codes_out[i] = 0u;
        return;
    }
    const float inv = 1.0f / s;
    for (uint32_t i = 0; i < n; ++i) codes_out[i] = psq8_e4m3_encode_u8(x[i] * inv);
    scale_out = s;
}

PSQ_DEVICE_HD inline void psq8_lsq1_encode_block(
    const float* x, uint32_t n, float& scale_out, uint8_t* codes_out) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    float s = 0.0f;
    if (amax > 0.0f && amax < 3.402823466e+38f) s = psq_bf16_round(amax * (1.0f / 448.0f));
    if (!(s > 0.0f)) {
        scale_out = 0.0f;
        for (uint32_t i = 0; i < n; ++i) codes_out[i] = 0u;
        return;
    }
    const float inv0 = 1.0f / s;
    float num = 0.0f;
    float den = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t c = psq8_e4m3_encode_u8(x[i] * inv0);
        const float qv = psq8_e4m3_decode_f32(c);
        num += x[i] * qv;
        den += qv * qv;
    }
    float s1 = 0.0f;
    if (den > 0.0f) s1 = psq_bf16_round(num / den);
    if (!(s1 > 0.0f)) {
        scale_out = 0.0f;
        for (uint32_t i = 0; i < n; ++i) codes_out[i] = 0u;
        return;
    }
    const float inv1 = 1.0f / s1;
    for (uint32_t i = 0; i < n; ++i) codes_out[i] = psq8_e4m3_encode_u8(x[i] * inv1);
    scale_out = s1;
}

}  // namespace ps::quantization::psq
