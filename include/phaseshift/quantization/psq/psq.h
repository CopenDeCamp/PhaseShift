#pragma once
#include <phaseshift/core/memory/types.h>
#include <phaseshift/quantization/psq/psq_device.h>
#include <cstdint>
#include <cstring>
#include <span>

namespace ps::quantization::psq {

inline float bf16_bytes_to_f32(const uint8_t* p) {
    const uint32_t bits = static_cast<uint32_t>(p[0]) |
                          (static_cast<uint32_t>(p[1]) << 8u);
    const uint32_t f = bits << 16u;
    float v;
    std::memcpy(&v, &f, sizeof(v));
    return v;
}

inline void store_bf16_bytes(uint8_t* p, float v) {
    const uint16_t b = f32_to_bf16_rne(v);
    p[0] = static_cast<uint8_t>(b & 0xFFu);
    p[1] = static_cast<uint8_t>(b >> 8u);
}

void quantize_psq4_row(
    std::span<const float> src,
    std::span<uint8_t> codes,
    std::span<uint8_t> scales,
    uint64_t logical_k,
    uint64_t padded_k);

void dequantize_psq4_row(
    std::span<const uint8_t> codes,
    std::span<const uint8_t> scales,
    std::span<float> dst,
    uint64_t logical_k,
    uint64_t padded_k);

void quantize_psq8_row(
    std::span<const float> src,
    std::span<uint8_t> codes,
    std::span<uint8_t> scales,
    uint64_t logical_k,
    uint64_t padded_k);

void dequantize_psq8_row(
    std::span<const uint8_t> codes,
    std::span<const uint8_t> scales,
    std::span<float> dst,
    uint64_t logical_k,
    uint64_t padded_k);

}  // namespace ps::quantization::psq
