#pragma once

#include <phaseshift/quantization/quant_format.h>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace ps::quantization::psq {

// Read-only view over a canonical SoA payload: one codes stream plus four
// metadata streams (unused streams are size-0 / null). The canonical layout is
// backend-independent, has no backend-specific preshuffle, no metadata
// duplication, and is directly decodable on CPU.
struct CanonicalQuantView {
    QuantFormatId format_id = QuantFormatId::None;
    const QuantFormatDesc* desc = nullptr;
    uint64_t rows = 0;
    uint64_t padded_k = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const uint8_t* metadata1 = nullptr;
    uint64_t metadata1_bytes = 0;
    const uint8_t* metadata2 = nullptr;
    uint64_t metadata2_bytes = 0;
    const uint8_t* metadata3 = nullptr;
    uint64_t metadata3_bytes = 0;
    const uint8_t* metadata4 = nullptr;
    uint64_t metadata4_bytes = 0;

    bool validate() const;
};

// Owning canonical SoA payload for one weight. Built by the quantizer (from a
// float source) and by tests; consumed by the file writer and the preshuffle.
struct CanonicalQuantStore {
    QuantFormatId format_id = QuantFormatId::None;
    const QuantFormatDesc* desc = nullptr;
    uint64_t rows = 0;
    uint64_t logical_k = 0;
    uint64_t padded_k = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> metadata1;
    std::vector<uint8_t> metadata2;
    std::vector<uint8_t> metadata3;
    std::vector<uint8_t> metadata4;

    void init(QuantFormatId id, uint64_t rows, uint64_t logical_k, uint64_t padded_k);
    CanonicalQuantView view() const;

    // Build row `row` (0-based) of the canonical payload from `src_row`
    // (logical_k floats). Reuses the reference row math; only the output
    // stream layout changes. Returns false on unsupported format.
    bool quantize_row(uint32_t row, std::span<const float> src_row);
};

// CPU dequantize of a canonical payload (full matrix or one row).
void dequantize_canonical(const CanonicalQuantView& v, std::vector<float>& out);
void dequantize_canonical_row(const CanonicalQuantView& v, uint32_t row, std::span<float> out);

}  // namespace ps::quantization::psq
