#pragma once
#include <phaseshift/core/status.h>
#include <cstdint>

namespace ps::runtime {

enum class RowBucket : std::uint16_t {
    R16 = 16,
    R32 = 32,
    R64 = 64,
    R128 = 128,
    R256 = 256,
    R512 = 512,
    R1024 = 1024,
    R2048 = 2048,
};

constexpr std::uint32_t row_bucket_limit(RowBucket bucket) noexcept {
    switch (bucket) {
        case RowBucket::R16: return 16;
        case RowBucket::R32: return 32;
        case RowBucket::R64: return 64;
        case RowBucket::R128: return 128;
        case RowBucket::R256: return 256;
        case RowBucket::R512: return 512;
        case RowBucket::R1024: return 1024;
        case RowBucket::R2048: return 2048;
    }
    return 0;
}

constexpr std::uint32_t row_bucket_rows(RowBucket bucket) noexcept {
    return row_bucket_limit(bucket);
}

constexpr bool valid_row_bucket(RowBucket bucket) noexcept {
    return row_bucket_limit(bucket) != 0;
}

constexpr bool is_optimized_gemm_row_bucket(RowBucket bucket) noexcept {
    switch (bucket) {
        case RowBucket::R16:
        case RowBucket::R32:
        case RowBucket::R64:
        case RowBucket::R128:
            return true;
        default:
            return false;
    }
}

inline Result<RowBucket> resolve_row_bucket(std::uint32_t logical_rows) {
    if (logical_rows == 0) {
        return Status::invalid_argument(
            "logical rows must be non-zero", __FILE__, __LINE__);
    }
    if (logical_rows <= 16) return RowBucket::R16;
    if (logical_rows <= 32) return RowBucket::R32;
    if (logical_rows <= 64) return RowBucket::R64;
    if (logical_rows <= 128) return RowBucket::R128;
    if (logical_rows <= 256) return RowBucket::R256;
    if (logical_rows <= 512) return RowBucket::R512;
    if (logical_rows <= 1024) return RowBucket::R1024;
    if (logical_rows <= 2048) return RowBucket::R2048;
    return Status::out_of_range(
        "logical rows exceed the production row bucket contract",
        __FILE__, __LINE__);
}

}
