#pragma once

#include <phaseshift/quantization/psq/psq_device.h>

#include <cstdint>

namespace ps::quantization::psq {

PSQ_DEVICE_HD inline float psq_bf16_round(float v) {
    uint32_t u;
    __builtin_memcpy(&u, &v, sizeof(u));
    u += 0x7FFFu + ((u >> 16u) & 1u);
    u &= 0xFFFF0000u;
    float out;
    __builtin_memcpy(&out, &u, sizeof(out));
    return out;
}

PSQ_DEVICE_HD inline void psq4_maxabs_encode_block(
    const float* x, uint32_t n, float& scale_out, uint8_t* codes_out) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    float s = 0.0f;
    if (amax > 0.0f && amax < 3.402823466e+38f) s = psq_bf16_round(amax * 0.1f);
    if (!(s > 0.0f)) {
        scale_out = 0.0f;
        for (uint32_t i = 0; i < n; ++i) codes_out[i] = 0u;
        return;
    }
    const float inv = 1.0f / s;
    for (uint32_t i = 0; i < n; ++i) codes_out[i] = cb10_encode(x[i], inv);
    scale_out = s;
}

PSQ_DEVICE_HD inline void psq4_lsq1_encode_block(
    const float* x, uint32_t n, float& scale_out, uint8_t* codes_out) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    float s = 0.0f;
    if (amax > 0.0f && amax < 3.402823466e+38f) s = psq_bf16_round(amax * 0.1f);
    if (!(s > 0.0f)) {
        scale_out = 0.0f;
        for (uint32_t i = 0; i < n; ++i) codes_out[i] = 0u;
        return;
    }
    const float inv0 = 1.0f / s;
    float num = 0.0f;
    float den = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t c = cb10_encode(x[i], inv0);
        const float qv = cb10_code_to_f32(c);
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
    for (uint32_t i = 0; i < n; ++i) codes_out[i] = cb10_encode(x[i], inv1);
    scale_out = s1;
}

PSQ_DEVICE_HD inline void psq4_pack_codes(const uint8_t* codes, uint8_t* bytes) {
    for (uint32_t j = 0; j < 16u; ++j) {
        bytes[j] = static_cast<uint8_t>((codes[j] & 0xFu) | ((codes[j + 16u] & 0xFu) << 4u));
    }
}

}  // namespace ps::quantization::psq
