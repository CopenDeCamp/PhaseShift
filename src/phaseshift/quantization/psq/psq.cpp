#include <phaseshift/quantization/psq/psq.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <algorithm>
#include <cmath>
#include <cfloat>

namespace ps::quantization::psq {

namespace {

constexpr int kHalfCandidates = 32;
constexpr uint32_t kBlock = 32;

inline float bf16_round(float v) {
    const uint16_t b = f32_to_bf16_rne(v);
    const uint32_t f = static_cast<uint32_t>(b) << 16u;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline uint8_t psq4_code_index(float x, float inv_scale) {
    if (!std::isfinite(x)) {
        return 0;
    }
    const float a = std::fabs(x * inv_scale);
    uint8_t mag;
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

void quantize_psq4_block(const float* x, float* out_scale, uint8_t* out_codes) {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < kBlock; ++i) {
        const float a = std::fabs(x[i]);
        if (std::isfinite(a) && a > max_abs) max_abs = a;
    }
    uint8_t best_codes[kBlock] = {};
    float best_scale = 0.0f;
    float best_err = FLT_MAX;
    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        const float s0 = max_abs / kPsq4Magnitudes[7];
        for (int j = -kHalfCandidates; j <= kHalfCandidates; ++j) {
            const float s = bf16_round(s0 * std::exp2(static_cast<float>(j) / 64.0f));
            if (!(s > 0.0f) || !std::isfinite(s)) continue;
            const float inv = 1.0f / s;
            float err = 0.0f;
            uint8_t codes[kBlock];
            for (uint32_t i = 0; i < kBlock; ++i) {
                const uint8_t c = psq4_code_index(x[i], inv);
                codes[i] = c;
                const float y = cb10_code_to_f32(c) * s;
                const float d = x[i] - y;
                err += d * d;
            }
            if (err < best_err) {
                best_err = err;
                best_scale = s;
                std::memcpy(best_codes, codes, sizeof(best_codes));
            }
        }
    }
    out_scale[0] = best_scale;
    for (uint32_t i = 0; i < kBlock; ++i) {
        out_codes[i] = best_codes[i];
    }
}

void quantize_psq8_block(const float* x, float* out_scale, uint8_t* out_codes) {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < kBlock; ++i) {
        const float a = std::fabs(x[i]);
        if (std::isfinite(a) && a > max_abs) max_abs = a;
    }
    uint8_t best_codes[kBlock] = {};
    float best_scale = 0.0f;
    float best_err = FLT_MAX;
    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        const float s0 = max_abs / 448.0f;
        for (int j = -kHalfCandidates; j <= kHalfCandidates; ++j) {
            const float s = bf16_round(s0 * std::exp2(static_cast<float>(j) / 64.0f));
            if (!(s > 0.0f) || !std::isfinite(s)) continue;
            const float inv = 1.0f / s;
            float err = 0.0f;
            uint8_t codes[kBlock];
            for (uint32_t i = 0; i < kBlock; ++i) {
                const uint8_t c = fpx::e4m3_encode_u8(x[i] * inv);
                codes[i] = c;
                const float y = fpx::e4m3_decode_f32(c) * s;
                const float d = x[i] - y;
                err += d * d;
            }
            if (err < best_err) {
                best_err = err;
                best_scale = s;
                std::memcpy(best_codes, codes, sizeof(best_codes));
            }
        }
    }
    out_scale[0] = best_scale;
    for (uint32_t i = 0; i < kBlock; ++i) {
        out_codes[i] = best_codes[i];
    }
}

void load_block(std::span<const float> src, uint64_t k, uint64_t ib, float* x) {
    for (uint32_t i = 0; i < kBlock; ++i) {
        const uint64_t idx = ib * kBlock + i;
        x[i] = (idx < k) ? src[static_cast<std::size_t>(idx)] : 0.0f;
    }
}

}  // namespace

void quantize_psq4_row(
    std::span<const float> src,
    std::span<uint8_t> codes,
    std::span<uint8_t> scales,
    uint64_t logical_k,
    uint64_t padded_k) {
    (void)logical_k;
    const uint64_t nb = padded_k / kBlock;
    for (uint64_t ib = 0; ib < nb; ++ib) {
        float x[kBlock];
        load_block(src, logical_k, ib, x);
        float s = 0.0f;
        uint8_t c[kBlock];
        quantize_psq4_block(x, &s, c);
        store_bf16_bytes(scales.data() + ib * 2u, s);
        uint8_t* dst = codes.data() + ib * 16u;
        for (uint32_t j = 0; j < 16u; ++j) {
            dst[j] = static_cast<uint8_t>(c[j] | (c[j + 16u] << 4u));
        }
    }
}

void dequantize_psq4_row(
    std::span<const uint8_t> codes,
    std::span<const uint8_t> scales,
    std::span<float> dst,
    uint64_t logical_k,
    uint64_t padded_k) {
    (void)logical_k;
    const uint64_t nb = padded_k / kBlock;
    for (uint64_t ib = 0; ib < nb; ++ib) {
        const float s = bf16_bytes_to_f32(scales.data() + ib * 2u);
        const uint8_t* src = codes.data() + ib * 16u;
        for (uint32_t j = 0; j < 16u; ++j) {
            const uint8_t byte = src[j];
            dst[static_cast<std::size_t>(ib * kBlock + j)] = cb10_code_to_f32(byte & 0x0Fu) * s;
            dst[static_cast<std::size_t>(ib * kBlock + 16u + j)] = cb10_code_to_f32(byte >> 4u) * s;
        }
    }
}

void quantize_psq8_row(
    std::span<const float> src,
    std::span<uint8_t> codes,
    std::span<uint8_t> scales,
    uint64_t logical_k,
    uint64_t padded_k) {
    (void)logical_k;
    const uint64_t nb = padded_k / kBlock;
    for (uint64_t ib = 0; ib < nb; ++ib) {
        float x[kBlock];
        load_block(src, logical_k, ib, x);
        float s = 0.0f;
        uint8_t c[kBlock];
        quantize_psq8_block(x, &s, c);
        store_bf16_bytes(scales.data() + ib * 2u, s);
        std::memcpy(codes.data() + ib * kBlock, c, kBlock);
    }
}

void dequantize_psq8_row(
    std::span<const uint8_t> codes,
    std::span<const uint8_t> scales,
    std::span<float> dst,
    uint64_t logical_k,
    uint64_t padded_k) {
    (void)logical_k;
    const uint64_t nb = padded_k / kBlock;
    for (uint64_t ib = 0; ib < nb; ++ib) {
        const float s = bf16_bytes_to_f32(scales.data() + ib * 2u);
        const uint8_t* src = codes.data() + ib * kBlock;
        for (uint32_t i = 0; i < kBlock; ++i) {
            dst[static_cast<std::size_t>(ib * kBlock + i)] =
                fpx::e4m3_decode_f32(src[i]) * s;
        }
    }
}


}  // namespace ps::quantization::psq
