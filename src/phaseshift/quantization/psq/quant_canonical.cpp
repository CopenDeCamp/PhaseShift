#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/psq/psq.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <cmath>
#include <cstring>

namespace ps::quantization::psq {

namespace {

constexpr uint32_t kBlock = 32;

uint64_t row_metadata_bytes(uint32_t row, uint64_t nb, const QuantFormatDesc& f, uint32_t idx) {
    const QuantMetaDesc& m = f.meta[idx];
    if (m.count_mode == MetaCountMode::PerBlock) return static_cast<uint64_t>(row) * nb * m.element_bytes;
    return 0;
}

}  // namespace

bool CanonicalQuantView::validate() const {
    if (desc == nullptr) return false;
    if (padded_k == 0 || padded_k % desc->block_elements != 0) return false;
    if (codes_bytes != quant_codes_bytes(rows, padded_k, *desc)) return false;
    if (metadata1_bytes != quant_metadata1_bytes(rows, padded_k, *desc)) return false;
    if (metadata2_bytes != quant_metadata2_bytes(rows, padded_k, *desc)) return false;
    if (metadata3_bytes != quant_metadata3_bytes(rows, padded_k, *desc)) return false;
    if (metadata4_bytes != quant_metadata4_bytes(rows, padded_k, *desc)) return false;
    return true;
}

void CanonicalQuantStore::init(QuantFormatId id, uint64_t rows, uint64_t logical_k, uint64_t padded_k) {
    format_id = id;
    desc = quant_format_desc(id);
    this->rows = rows;
    this->logical_k = logical_k;
    this->padded_k = padded_k;
    if (desc == nullptr) return;
    codes.reserve(quant_codes_bytes(rows, padded_k, *desc));
    metadata1.reserve(quant_metadata1_bytes(rows, padded_k, *desc));
    metadata2.reserve(quant_metadata2_bytes(rows, padded_k, *desc));
    metadata3.reserve(quant_metadata3_bytes(rows, padded_k, *desc));
    metadata4.reserve(quant_metadata4_bytes(rows, padded_k, *desc));
}

CanonicalQuantView CanonicalQuantStore::view() const {
    CanonicalQuantView v;
    v.format_id = format_id;
    v.desc = desc;
    v.rows = rows;
    v.padded_k = padded_k;
    v.codes = codes.data();
    v.codes_bytes = codes.size();
    v.metadata1 = metadata1.data();
    v.metadata1_bytes = metadata1.size();
    v.metadata2 = metadata2.data();
    v.metadata2_bytes = metadata2.size();
    v.metadata3 = metadata3.data();
    v.metadata3_bytes = metadata3.size();
    v.metadata4 = metadata4.data();
    v.metadata4_bytes = metadata4.size();
    return v;
}

bool CanonicalQuantStore::quantize_row(uint32_t row, std::span<const float> src_row) {
    (void)row;
    if (desc == nullptr) return false;
    const uint64_t nb = padded_k / kBlock;

    switch (format_id) {
        case QuantFormatId::Psq4: {
            std::vector<uint8_t> row_codes(nb * 16u);
            std::vector<uint8_t> scales(nb * 2u);
            quantize_psq4_row(src_row, row_codes, scales, logical_k, padded_k);
            codes.insert(codes.end(), row_codes.begin(), row_codes.end());
            metadata1.insert(metadata1.end(), scales.begin(), scales.end());
            return true;
        }
        case QuantFormatId::Psq8: {
            std::vector<uint8_t> row_codes(nb * 32u);
            std::vector<uint8_t> scales(nb * 2u);
            quantize_psq8_row(src_row, row_codes, scales, logical_k, padded_k);
            codes.insert(codes.end(), row_codes.begin(), row_codes.end());
            metadata1.insert(metadata1.end(), scales.begin(), scales.end());
            return true;
        }
        default:
            return false;
    }
}

void dequantize_canonical_row(const CanonicalQuantView& v, uint32_t row, std::span<float> out) {
    if (v.desc == nullptr) return;
    const QuantFormatDesc& f = *v.desc;
    const uint64_t nb = v.padded_k / kBlock;
    const uint64_t codes_off = static_cast<uint64_t>(row) * nb * f.code_bytes_per_block;
    const uint64_t metadata_off[4] = {
        row_metadata_bytes(row, nb, f, 0),
        row_metadata_bytes(row, nb, f, 1),
        row_metadata_bytes(row, nb, f, 2),
        row_metadata_bytes(row, nb, f, 3),
    };
    const uint8_t* codes = v.codes + codes_off;
    const uint8_t* metadata1 = v.metadata1 + metadata_off[0];

    switch (v.format_id) {
        case QuantFormatId::Psq4:
            for (uint64_t ib = 0; ib < nb; ++ib) {
                const float s = bf16_bytes_to_f32(metadata1 + ib * 2u);
                const uint8_t* src = codes + ib * 16u;
                for (uint32_t j = 0; j < 16u; ++j) {
                    const uint8_t b = src[j];
                    out[ib * kBlock + j] = cb10_code_to_f32(b & 0x0Fu) * s;
                    out[ib * kBlock + 16u + j] = cb10_code_to_f32(b >> 4u) * s;
                }
            }
            break;
        case QuantFormatId::Psq8:
            for (uint64_t ib = 0; ib < nb; ++ib) {
                const float s = bf16_bytes_to_f32(metadata1 + ib * 2u);
                const uint8_t* src = codes + ib * kBlock;
                for (uint32_t i = 0; i < kBlock; ++i)
                    out[ib * kBlock + i] = fpx::e4m3_decode_f32(src[i]) * s;
            }
            break;
        default:
            break;
    }
}

void dequantize_canonical(const CanonicalQuantView& v, std::vector<float>& out) {
    if (v.desc == nullptr) return;
    out.resize(v.rows * v.padded_k);
    std::vector<float> row(v.padded_k);
    for (uint64_t r = 0; r < v.rows; ++r) {
        dequantize_canonical_row(v, static_cast<uint32_t>(r), row);
        std::memcpy(out.data() + r * v.padded_k, row.data(), v.padded_k * sizeof(float));
    }
}

}  // namespace ps::quantization::psq
