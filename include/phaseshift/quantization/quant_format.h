#pragma once

#include <cstdint>

namespace ps::quantization {

constexpr uint32_t kQuantBlockElements = 32;
constexpr uint32_t kFp8ScaleBlockN = 128;
constexpr uint32_t kFp8ScaleBlockK = 128;

// Metadata placement relative to the codes matrix.
//
//   PerBlock        one element per (row, K block)
//   PerSuperBlock   one element per (row, superblock)
//   MatrixBlock     one element per (N block, K block) tile, shared by every
//                   row inside the tile. `block_n` / `block_k` describe the
//                   tile geometry; `block_k == 0` means "format block".
enum class MetaCountMode : uint8_t {
    None = 0,
    PerBlock = 1,
    PerSuperBlock = 2,
    MatrixBlock = 3,
};

struct QuantMetaDesc {
    uint32_t element_bytes = 0;
    MetaCountMode count_mode = MetaCountMode::None;
    uint32_t block_n = 1;
    uint32_t block_k = 0;
};

struct QuantFormatDesc {
    uint32_t block_elements = 0;
    uint32_t blocks_per_superblock = 1;
    uint32_t code_bytes_per_block = 0;
    QuantMetaDesc meta[4];
};

enum class QuantFormatId : uint8_t {
    None = 0,
    Bf16 = 1,
    Psq4 = 2,
    Psq8 = 3,
    Fp8Block128 = 4,
    Mxfp4 = 5,
};

static_assert(static_cast<uint8_t>(QuantFormatId::None) == 0);
static_assert(static_cast<uint8_t>(QuantFormatId::Bf16) == 1);
static_assert(static_cast<uint8_t>(QuantFormatId::Psq4) == 2);
static_assert(static_cast<uint8_t>(QuantFormatId::Psq8) == 3);
static_assert(static_cast<uint8_t>(QuantFormatId::Fp8Block128) == 4);
static_assert(static_cast<uint8_t>(QuantFormatId::Mxfp4) == 5);

inline constexpr QuantFormatDesc kPsq4FormatDesc = {
    32, 1, 16,
    {
        {2, MetaCountMode::PerBlock},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
    },
};

inline constexpr QuantFormatDesc kPsq8FormatDesc = {
    32, 1, 32,
    {
        {2, MetaCountMode::PerBlock},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
    },
};

inline constexpr QuantFormatDesc kBf16FormatDesc = {
    16, 1, 32,
    {
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
    },
};

// FP8 E4M3 codes are one byte per element; the FP32 scale tile is shared by
// 128 output rows and 128 K elements.
inline constexpr QuantFormatDesc kFp8Block128FormatDesc = {
    kFp8ScaleBlockK, 1, kFp8ScaleBlockK,
    {
        {4, MetaCountMode::MatrixBlock, kFp8ScaleBlockN, kFp8ScaleBlockK},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
    },
};

// MXFP4 packs two E2M1 weights per byte; one E8M0 scale per 32 K elements of
// each row.
inline constexpr QuantFormatDesc kMxfp4FormatDesc = {
    32, 1, 16,
    {
        {1, MetaCountMode::PerBlock, 1, 32},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
        {0, MetaCountMode::None},
    },
};

inline constexpr const QuantFormatDesc* quant_format_desc(QuantFormatId id) {
    switch (id) {
        case QuantFormatId::Psq4: return &kPsq4FormatDesc;
        case QuantFormatId::Psq8: return &kPsq8FormatDesc;
        case QuantFormatId::Bf16: return &kBf16FormatDesc;
        case QuantFormatId::Fp8Block128: return &kFp8Block128FormatDesc;
        case QuantFormatId::Mxfp4: return &kMxfp4FormatDesc;
        default: return nullptr;
    }
}

inline uint64_t quant_num_blocks(uint64_t padded_k, const QuantFormatDesc& f) {
    return padded_k / f.block_elements;
}

inline uint64_t quant_num_superblocks(uint64_t padded_k, const QuantFormatDesc& f) {
    const uint64_t nb = quant_num_blocks(padded_k, f);
    return (nb + f.blocks_per_superblock - 1u) / f.blocks_per_superblock;
}

inline uint64_t quant_codes_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return rows * quant_num_blocks(padded_k, f) * f.code_bytes_per_block;
}

// Number of metadata elements for a matrix batch of `batch` matrices, each with
// `rows` output rows and `padded_k` K elements.
inline uint64_t quant_metadata_count(
    uint64_t batch, uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f, uint32_t idx) {
    const QuantMetaDesc& m = f.meta[idx];
    switch (m.count_mode) {
        case MetaCountMode::None:
            return 0;
        case MetaCountMode::PerBlock:
            return batch * rows * quant_num_blocks(padded_k, f);
        case MetaCountMode::PerSuperBlock:
            return batch * rows * quant_num_superblocks(padded_k, f);
        case MetaCountMode::MatrixBlock: {
            const uint64_t bn = m.block_n != 0 ? m.block_n : 1;
            const uint64_t bk = m.block_k != 0 ? m.block_k : f.block_elements;
            return batch * ((rows + bn - 1u) / bn) * ((padded_k + bk - 1u) / bk);
        }
    }
    return 0;
}

inline uint64_t quant_metadata_bytes(
    uint64_t batch, uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f, uint32_t idx) {
    const QuantMetaDesc& m = f.meta[idx];
    if (m.count_mode == MetaCountMode::None || m.element_bytes == 0) return 0;
    return quant_metadata_count(batch, rows, padded_k, f, idx) * m.element_bytes;
}

inline uint64_t quant_metadata_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f, uint32_t idx) {
    return quant_metadata_bytes(1, rows, padded_k, f, idx);
}

inline uint64_t quant_metadata1_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 0);
}
inline uint64_t quant_metadata2_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 1);
}
inline uint64_t quant_metadata3_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 2);
}
inline uint64_t quant_metadata4_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_metadata_bytes(rows, padded_k, f, 3);
}

inline uint64_t quant_payload_bytes(
    uint64_t batch, uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_codes_bytes(batch * rows, padded_k, f) +
           quant_metadata_bytes(batch, rows, padded_k, f, 0) +
           quant_metadata_bytes(batch, rows, padded_k, f, 1) +
           quant_metadata_bytes(batch, rows, padded_k, f, 2) +
           quant_metadata_bytes(batch, rows, padded_k, f, 3);
}

inline uint64_t quant_payload_bytes(uint64_t rows, uint64_t padded_k, const QuantFormatDesc& f) {
    return quant_payload_bytes(1, rows, padded_k, f);
}

}  // namespace ps::quantization
