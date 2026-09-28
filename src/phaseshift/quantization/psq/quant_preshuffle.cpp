#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <cstddef>
#include <cstring>

namespace ps::quantization::psq {

namespace {

constexpr uint32_t kTile = 16;

// Tiling of codes + metadata1 for PSQ4. `code_block_bytes` is 16. The native
// codes layout is [tile16][nb][16 outputs * code_block_bytes]; the native
// metadata1 layout is [tile16][nb][16 outputs * 2 bytes].
void tile_codes_metadata1(
    const uint8_t* codes, const uint8_t* metadata1,
    uint64_t rows, uint64_t rows_padded, uint64_t nb,
    uint64_t code_block_bytes, uint64_t metadata1_row_bytes,
    std::vector<uint8_t>& codes_out, std::vector<uint8_t>& metadata1_out)
{
    const uint64_t tiles = rows_padded / kTile;
    const uint64_t code_block = kTile * code_block_bytes;
    const uint64_t metadata1_block = kTile * 2u;
    codes_out.assign(tiles * nb * code_block, 0u);
    metadata1_out.assign(tiles * nb * metadata1_block, 0u);
    for (uint64_t t = 0; t < tiles; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t ol = 0; ol < kTile; ++ol) {
                const uint64_t o = t * kTile + ol;
                if (o >= rows) continue;
                for (uint64_t j = 0; j < code_block_bytes; ++j)
                    codes_out[(t * nb + ib) * code_block + ol * code_block_bytes + j] =
                        codes[o * nb * code_block_bytes + ib * code_block_bytes + j];
            }
        }
    }
    if (metadata1 == nullptr) return;
    for (uint64_t t = 0; t < tiles; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t ol = 0; ol < kTile; ++ol) {
                const uint64_t o = t * kTile + ol;
                if (o >= rows) continue;
                for (uint64_t h = 0; h < 2u; ++h)
                    metadata1_out[(t * nb + ib) * metadata1_block + ol * 2u + h] =
                        metadata1[o * metadata1_row_bytes + ib * 2u + h];
            }
        }
    }
}

// PSQ8 native codes: each (tile, block) 512-byte chunk is
// [half][16 outputs * 16 bytes], matching the WMMA consumer layout
// (first 16 codes of the 32-wide block, then the second 16).
void tile_codes_metadata1_psq8(
    const uint8_t* codes, const uint8_t* metadata1,
    uint64_t rows, uint64_t rows_padded, uint64_t nb,
    std::vector<uint8_t>& codes_out, std::vector<uint8_t>& metadata1_out)
{
    const uint64_t tiles = rows_padded / kTile;
    const uint64_t code_block = kTile * 32u;
    const uint64_t metadata1_block = kTile * 2u;
    codes_out.assign(tiles * nb * code_block, 0u);
    metadata1_out.assign(tiles * nb * metadata1_block, 0u);
    for (uint64_t t = 0; t < tiles; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t half = 0; half < 2u; ++half) {
                for (uint64_t ol = 0; ol < kTile; ++ol) {
                    const uint64_t o = t * kTile + ol;
                    if (o >= rows) continue;
                    std::memcpy(
                        codes_out.data() + (t * nb + ib) * code_block + half * 256u + ol * 16u,
                        codes + o * nb * 32u + ib * 32u + half * 16u, 16u);
                }
            }
        }
    }
    if (metadata1 == nullptr) return;
    for (uint64_t t = 0; t < tiles; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t ol = 0; ol < kTile; ++ol) {
                const uint64_t o = t * kTile + ol;
                if (o >= rows) continue;
                std::memcpy(
                    metadata1_out.data() + (t * nb + ib) * metadata1_block + ol * 2u,
                    metadata1 + o * nb * 2u + ib * 2u, 2u);
            }
        }
    }
}

}  // namespace

bool preshuffle_native(const CanonicalQuantView& v, NativeQuantHost& out) {
    if (v.desc == nullptr) return false;
    if (v.format_id != QuantFormatId::Psq4 && v.format_id != QuantFormatId::Psq8)
        return false;

    out.format_id = v.format_id;
    out.desc = v.desc;
    out.rows = v.rows;
    out.padded_k = v.padded_k;
    const uint64_t nb = v.padded_k / v.desc->block_elements;
    out.rows_padded = (v.rows + kTile - 1u) / kTile * kTile;

    if (v.format_id == QuantFormatId::Psq8) {
        tile_codes_metadata1_psq8(v.codes, v.metadata1, v.rows, out.rows_padded, nb,
                                  out.codes, out.metadata1);
        out.metadata2.clear();
        out.metadata3.clear();
        out.metadata4.clear();
        out.codes_row_stride_bytes = static_cast<uint32_t>(nb * kTile * 32u);
        out.metadata1_stride_bytes = static_cast<uint32_t>(nb * kTile * 2u);
        out.metadata2_stride_bytes = 0;
        out.metadata3_stride_bytes = 0;
        out.metadata4_stride_bytes = 0;
        return true;
    }

    const uint64_t code_block_bytes = 16u;
    tile_codes_metadata1(v.codes, v.metadata1, v.rows, out.rows_padded, nb,
                         code_block_bytes, nb * 2u, out.codes, out.metadata1);
    out.metadata2.clear();
    out.metadata3.clear();
    out.metadata4.clear();
    out.codes_row_stride_bytes = static_cast<uint32_t>(nb * kTile * code_block_bytes);
    out.metadata1_stride_bytes = static_cast<uint32_t>(nb * kTile * 2u);
    out.metadata2_stride_bytes = 0;
    out.metadata3_stride_bytes = 0;
    out.metadata4_stride_bytes = 0;
    return true;
}

bool inverse_preshuffle(const NativeQuantHost& n, CanonicalQuantStore& out) {
    if (n.desc == nullptr) return false;
    if (n.format_id != QuantFormatId::Psq4 && n.format_id != QuantFormatId::Psq8)
        return false;
    const uint64_t nb = n.padded_k / n.desc->block_elements;
    out.init(n.format_id, n.rows, 0, n.padded_k);
    const QuantFormatDesc& f = *n.desc;
    out.codes.resize(quant_codes_bytes(n.rows, n.padded_k, f));
    out.metadata1.resize(quant_metadata1_bytes(n.rows, n.padded_k, f));
    out.metadata2.resize(quant_metadata2_bytes(n.rows, n.padded_k, f));
    out.metadata3.resize(quant_metadata3_bytes(n.rows, n.padded_k, f));
    out.metadata4.resize(quant_metadata4_bytes(n.rows, n.padded_k, f));

    if (n.format_id == QuantFormatId::Psq8) {
        const uint64_t code_block = kTile * 32u;
        const uint64_t metadata1_block = kTile * 2u;
        for (uint64_t o = 0; o < n.rows; ++o) {
            const uint64_t t = o / kTile;
            const uint64_t ol = o % kTile;
            for (uint64_t ib = 0; ib < nb; ++ib) {
                for (uint64_t half = 0; half < 2u; ++half)
                    for (uint64_t j = 0; j < 16u; ++j)
                        out.codes[static_cast<size_t>(o) * nb * 32u + ib * 32u + half * 16u + j] =
                            n.codes[(t * nb + ib) * code_block + half * 256u + ol * 16u + j];
                for (uint64_t h = 0; h < 2u; ++h)
                    out.metadata1[static_cast<size_t>(o) * nb * 2u + ib * 2u + h] =
                        n.metadata1[(t * nb + ib) * metadata1_block + ol * 2u + h];
            }
        }
        return true;
    }

    const uint64_t code_block_bytes = 16u;
    const uint64_t code_block = kTile * code_block_bytes;
    const uint64_t metadata1_block = kTile * 2u;
    for (uint64_t o = 0; o < n.rows; ++o) {
        const uint64_t t = o / kTile;
        const uint64_t ol = o % kTile;
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t j = 0; j < code_block_bytes; ++j)
                out.codes[static_cast<size_t>(o) * nb * code_block_bytes + ib * code_block_bytes + j] =
                    n.codes[(t * nb + ib) * code_block + ol * code_block_bytes + j];
            for (uint64_t h = 0; h < 2u; ++h)
                out.metadata1[static_cast<size_t>(o) * nb * 2u + ib * 2u + h] =
                    n.metadata1[(t * nb + ib) * metadata1_block + ol * 2u + h];
        }
    }
    return true;
}

}  // namespace ps::quantization::psq
