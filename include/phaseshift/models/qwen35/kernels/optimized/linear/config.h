#pragma once

#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kBf16GemmExactRowsMax = 16u;
inline constexpr uint32_t kBf16GemmSplitKMaxOutFeatures = 256u;
inline constexpr uint32_t kBf16GemmWideMinRows = 65u;
inline constexpr uint32_t kBf16GemmExactKAlignment = 8u;
inline constexpr uint32_t kBf16GemmExactPointerAlignment = 16u;
inline constexpr uint32_t kBf16GemmExactRowStrideAlignment = 8u;
inline constexpr uint32_t kBf16GemmExactRowsThreads = 256u;
inline constexpr uint32_t kBf16GemmExactRowsOutFeaturesPerBlock = 8u;

constexpr uint32_t bf16_exact_rows_grid_x(uint32_t out_features) {
    return (out_features + kBf16GemmExactRowsOutFeaturesPerBlock - 1u) /
           kBf16GemmExactRowsOutFeaturesPerBlock;
}

enum class Bf16GemmConfigId : uint8_t {
    Wmma,
    WmmaWide,
    WmmaKPartition,
    ExactRows,
};

struct Bf16GemmConfig {
    Bf16GemmConfigId id = Bf16GemmConfigId::Wmma;
    uint8_t exact_rows = 0u;
};

inline constexpr uint32_t kPsqGemmRowTile = 16u;
inline constexpr uint32_t kPsqGemmRowBlockMinOutFeatures = 256u;
inline constexpr uint32_t kPsqGemmPrefill2dRowsPerBlock = 16u * kPsqGemmRowTile;
inline constexpr uint32_t kPsqGemmPrefill2dMinRows = 512u;
inline constexpr uint32_t kPsqGemmPrefill2dMinOutFeatures = 512u;

inline constexpr uint32_t kPsq4GemmPrefill2dBlock64 = 64u;
inline constexpr uint32_t kPsq4GemmPrefill2dBlock128 = 128u;
inline constexpr uint32_t kPsq4GemmPrefill2dLargeMinRows = 1024u;
inline constexpr uint32_t kPsq4GemmPrefill2dLargeMinOutFeatures = 2048u;

inline constexpr uint32_t kPsq8GemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kPsq8GemmPrefill2dKChunk = 64u;
inline constexpr uint32_t kPsq8GemmPrefill2dBlock64 = 64u;
inline constexpr uint32_t kPsq8GemmPrefill2dBlock128 = 128u;

enum class Psq4GemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    RowBlock8,
    Prefill2D_K64N64,
    Prefill2D_K128N64,
    Prefill2D_K64N128,
    Prefill2D_K128N128,
};

struct Psq4GemmConfig {
    Psq4GemmConfigId id = Psq4GemmConfigId::RowBlock1;
};

enum class Psq8GemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    RowBlock8,
    Prefill2D,
    Prefill2D_K64N128,
    Prefill2D_K128N128,
};

struct Psq8GemmConfig {
    Psq8GemmConfigId id = Psq8GemmConfigId::RowBlock1;
};

inline constexpr uint32_t kFp8Block128GemmRowTile = 16u;
inline constexpr uint32_t kFp8Block128GemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kFp8Block128GemmPrefill2dKChunk = 64u;

enum class Fp8Block128GemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    RowBlock8,
    Prefill2D,
};

struct Fp8Block128GemmConfig {
    Fp8Block128GemmConfigId id = Fp8Block128GemmConfigId::RowBlock1;
};

inline constexpr uint32_t kMxfp4GemmRowTile = 16u;
inline constexpr uint32_t kMxfp4GemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kMxfp4GemmPrefill2dKChunk = 64u;

enum class Mxfp4GemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    RowBlock8,
    Prefill2D,
};

struct Mxfp4GemmConfig {
    Mxfp4GemmConfigId id = Mxfp4GemmConfigId::RowBlock1;
};

}  // namespace ps::kernel
