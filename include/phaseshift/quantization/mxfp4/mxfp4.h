#pragma once

#include <phaseshift/quantization/quant_format.h>
#include <cstdint>
#include <span>
#include <vector>

namespace ps::quantization::mxfp4 {

// OCP MXFP4: E2M1 weights, E8M0 scales, K = 32 block. 2 weights / byte, one
// scale byte per block -> 17 bytes / 32 weights (4.25 bpw).
inline constexpr uint32_t kBlockK = 32;
inline constexpr uint32_t kCodesBytesPerBlock = 16;

// E2M1 representable magnitudes indexed by the 3-bit magnitude field.
inline constexpr float kE2M1Magnitudes[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

// Nibble (0..15): bit 3 is the sign, bits 0..2 index kE2M1Magnitudes.
float e2m1_decode_nibble(uint8_t nibble);
uint8_t e2m1_encode_nibble(float x);

// E8M0 byte: value = 2^(byte - 127). Byte 255 is NaN. Byte 0 is the canonical
// minimum exponent (-127).
float e8m0_decode_u8(uint8_t byte);
uint8_t e8m0_encode_f32(float x);

// Quantize one row. `src` holds `logical_k` floats; K beyond `logical_k` is
// treated as zero. `codes` is padded_k/32*16 bytes, `scales` padded_k/32 bytes.
void quantize_mxfp4_row(
    std::span<const float> src,
    std::span<uint8_t> codes,
    std::span<uint8_t> scales,
    uint64_t logical_k,
    uint64_t padded_k);

void dequantize_mxfp4_row(
    std::span<const uint8_t> codes,
    std::span<const uint8_t> scales,
    std::span<float> dst,
    uint64_t logical_k,
    uint64_t padded_k);

// Canonical on-disk payload: row-major E2M1 codes plus one E8M0 scale per K
// block. No preshuffle, directly decodable on CPU.
struct Mxfp4CanonicalView {
    uint64_t rows = 0;
    uint64_t padded_k = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const uint8_t* scales = nullptr;
    uint64_t scales_bytes = 0;

    bool validate() const;
};

struct Mxfp4CanonicalStore {
    uint64_t rows = 0;
    uint64_t logical_k = 0;
    uint64_t padded_k = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;

    void init(uint64_t rows, uint64_t logical_k, uint64_t padded_k);
    Mxfp4CanonicalView view() const;
    bool quantize_row(uint32_t row, std::span<const float> src_row);
};

void dequantize_mxfp4_canonical(const Mxfp4CanonicalView& v, std::vector<float>& out);
void dequantize_mxfp4_canonical_row(const Mxfp4CanonicalView& v, uint32_t row, std::span<float> out);

}  // namespace ps::quantization::mxfp4
