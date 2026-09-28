#pragma once
#include <hip/hip_fp8.h>
#include <cstdint>

namespace ps {
namespace qwen35 {

using fp8e4m3_storage_t = __hip_fp8_storage_t;
static_assert(sizeof(fp8e4m3_storage_t) == 1, "FP8 E4M3 storage must be exactly 1 byte");

enum class KVCacheDType : uint8_t {
    BF16 = 0,
    FP8_E4M3 = 1,
    PSQ4_W32 = 3,
    PSQ8_W32 = 4,
};

constexpr uint32_t kPsq4BlockValues = 32u;
constexpr uint32_t kPsq4CodeBits = 4u;
constexpr uint32_t kPsq4CodeBytesPerBlock = 16u;
constexpr uint32_t kPsq4ScaleBytesPerBlock = 2u;
constexpr uint32_t kPsq4BlocksPerHead = 8u;
constexpr uint32_t kPsq4CodeBytesPerHead =
    kPsq4BlocksPerHead * kPsq4CodeBytesPerBlock;
constexpr uint32_t kPsq4ScaleBytesPerHead =
    kPsq4BlocksPerHead * kPsq4ScaleBytesPerBlock;

enum class Psq4ScaleEstimator : uint8_t {
    MaxAbs = 0,
    Lsq1 = 1,
};

constexpr uint32_t kPsq8BlockValues = 32u;
constexpr uint32_t kPsq8CodeBits = 8u;
constexpr uint32_t kPsq8CodeBytesPerBlock = 32u;
constexpr uint32_t kPsq8ScaleBytesPerBlock = 2u;
constexpr uint32_t kPsq8BlocksPerHead = 8u;
constexpr uint32_t kPsq8CodeBytesPerHead =
    kPsq8BlocksPerHead * kPsq8CodeBytesPerBlock;
constexpr uint32_t kPsq8ScaleBytesPerHead =
    kPsq8BlocksPerHead * kPsq8ScaleBytesPerBlock;

enum class Psq8ScaleEstimator : uint8_t {
    MaxAbs = 0,
    Lsq1 = 1,
};

}
}
