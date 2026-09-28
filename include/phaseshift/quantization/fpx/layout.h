#pragma once
#include <phaseshift/quantization/fpx/types.h>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace ps::quantization::fpx {

constexpr uint64_t kRegionAlignment = 64;
constexpr uint64_t kFpBlockSize = 32;
constexpr uint64_t kPsq4CodesBytesPerBlock = 16;
constexpr uint64_t kPsq8CodesBytesPerBlock = 32;
constexpr uint64_t kPsqScalesBytesPerBlock = 2;

inline uint64_t align_up64(uint64_t v) {
    return (v + 63u) & ~uint64_t{63};
}

inline uint64_t align_up(uint64_t v, uint64_t alignment) {
    const uint64_t mask = alignment - 1;
    return (v + mask) & ~mask;
}

inline uint64_t element_count(const std::vector<int64_t>& shape) {
    uint64_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<uint64_t>(d);
    }
    return n;
}

inline uint64_t row_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    uint64_t rows = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        rows *= static_cast<uint64_t>(shape[i]);
    }
    return rows;
}

inline uint64_t k_dim(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return static_cast<uint64_t>(shape.back());
}

// Matrix-batch view of a weight shape: `[..., N, K]`. `batch` is the product
// of every dimension before N. FP8 128x128 scale tiles must never straddle a
// batch boundary, so the batch dimension is kept separate from N.
struct MatrixShape {
    uint64_t batch = 1;
    uint64_t n = 0;
    uint64_t k = 0;
};

inline MatrixShape matrix_shape(const std::vector<int64_t>& shape) {
    MatrixShape m;
    if (shape.empty()) return m;
    if (shape.size() == 1) {
        m.n = 1;
        m.k = static_cast<uint64_t>(shape[0]);
        return m;
    }
    uint64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<uint64_t>(shape[i]);
    }
    m.batch = batch;
    m.n = static_cast<uint64_t>(shape[shape.size() - 2]);
    m.k = static_cast<uint64_t>(shape.back());
    return m;
}

inline uint64_t padded_k(uint64_t k, WeightEncoding enc) {
    switch (enc) {
        case WeightEncoding::BF16: return k;
        case WeightEncoding::FP8_BLOCK128: return align_up(k, 128);
        default: return align_up(k, kFpBlockSize);
    }
}

struct RegionSizes {
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t data_bytes = 0;
};

inline RegionSizes compute_region_sizes(const std::vector<int64_t>& shape, WeightEncoding enc) {
    RegionSizes r;
    const MatrixShape ms = matrix_shape(shape);
    const uint64_t rows = ms.batch * ms.n;
    const uint64_t kp = padded_k(ms.k, enc);
    const uint64_t blocks = rows * (kp / kFpBlockSize);
    if (enc == WeightEncoding::PSQ4) {
        r.codes_bytes = blocks * kPsq4CodesBytesPerBlock;
        r.scales_bytes = blocks * kPsqScalesBytesPerBlock;
    } else if (enc == WeightEncoding::PSQ8) {
        r.codes_bytes = blocks * kPsq8CodesBytesPerBlock;
        r.scales_bytes = blocks * kPsqScalesBytesPerBlock;
    } else if (enc == WeightEncoding::FP8_BLOCK128) {
        const uint64_t kblocks = kp / 128;
        const uint64_t nblocks = (ms.n + 127) / 128;
        r.codes_bytes = rows * kp;
        r.scales_bytes = ms.batch * nblocks * kblocks * 4;
    } else if (enc == WeightEncoding::MXFP4) {
        r.codes_bytes = rows * (kp / 2);
        r.scales_bytes = rows * (kp / 32);
    } else {
        r.data_bytes = rows * ms.k * 2;
    }
    return r;
}

inline uint32_t block_size_for(WeightEncoding enc) {
    switch (enc) {
        case WeightEncoding::BF16: return 0;
        case WeightEncoding::FP8_BLOCK128: return 128;
        default: return static_cast<uint32_t>(kFpBlockSize);
    }
}

inline uint32_t scale_group_size_for(WeightEncoding enc) {
    switch (enc) {
        case WeightEncoding::PSQ4:
        case WeightEncoding::PSQ8:
        case WeightEncoding::MXFP4:
            return static_cast<uint32_t>(kFpBlockSize);
        case WeightEncoding::FP8_BLOCK128:
            return 128;
        default:
            return 0;
    }
}

inline uint32_t scale_group_n_for(WeightEncoding enc) {
    return enc == WeightEncoding::FP8_BLOCK128 ? 128u : 1u;
}

inline FpxLayout layout_for(WeightEncoding enc) {
    switch (enc) {
        case WeightEncoding::BF16: return FpxLayout::Bf16RowMajorV1;
        case WeightEncoding::PSQ4: return FpxLayout::Psq4RowMajorSoAV2;
        case WeightEncoding::PSQ8: return FpxLayout::Psq8RowMajorSoAV1;
        case WeightEncoding::FP8_BLOCK128: return FpxLayout::Fp8E4m3RowMajorBlock128V1;
        case WeightEncoding::MXFP4: return FpxLayout::Mxfp4E2m1RowMajorS32V1;
    }
    return FpxLayout::Bf16RowMajorV1;
}

}  // namespace ps::quantization::fpx
