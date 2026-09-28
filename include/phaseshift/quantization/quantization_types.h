#pragma once

#include <cstdint>

namespace ps::quantization {

enum class WeightStorageFormat : uint8_t {
    BF16,
    PSQ4,
    PSQ8,
    FP8_BLOCK128,
    MXFP4,
};

enum class TensorDType : uint8_t {
    BF16,
    FP8_E4M3,
    INT8,
    INT32,
    FP32,
};

enum class KvCacheFormat : uint8_t {
    BF16,
    FP8_E4M3,
};

enum class StorageDType : uint8_t {
    NONE,
    BF16,
    U8,
    I8,
    FP16,
    FP32,
};

enum class QuantLayout : uint8_t {
    NONE,
    BF16_V1,
    PSQ4_SOA_V1,
    PSQ8_SOA_V1,
    PSQ_SCALE_BF16_S32_V1,
    FP8_E4M3_BLOCK128_V1,
    FP8_SCALE_F32_N128K128_V1,
    MXFP4_E2M1_S32_V1,
    MXFP4_SCALE_E8M0_S32_V1,
};

struct QuantDescriptor {
    WeightStorageFormat format = WeightStorageFormat::BF16;
    uint32_t group_size = 0;
    uint32_t block_size = 0;
    StorageDType storage_type = StorageDType::BF16;
    StorageDType scale_type = StorageDType::NONE;
    QuantLayout weight_layout = QuantLayout::BF16_V1;
    QuantLayout scale_layout = QuantLayout::NONE;
    uint32_t format_version = 0;
};

template<
    WeightStorageFormat FORMAT,
    uint32_t GROUP_SIZE,
    uint32_t BLOCK_SIZE,
    StorageDType STORAGE_TYPE,
    StorageDType SCALE_TYPE,
    QuantLayout WEIGHT_LAYOUT,
    QuantLayout SCALE_LAYOUT>
struct QuantSpec {
    static constexpr WeightStorageFormat format = FORMAT;
    static constexpr uint32_t group_size = GROUP_SIZE;
    static constexpr uint32_t block_size = BLOCK_SIZE;
    static constexpr StorageDType storage_type = STORAGE_TYPE;
    static constexpr StorageDType scale_type = SCALE_TYPE;
    static constexpr QuantLayout weight_layout = WEIGHT_LAYOUT;
    static constexpr QuantLayout scale_layout = SCALE_LAYOUT;
};

template<
    TensorDType ACTIVATION,
    TensorDType DOT_ACCUMULATOR,
    TensorDType REDUCTION_ACCUMULATOR,
    TensorDType OUTPUT>
struct ComputeSpec {
    static constexpr TensorDType activation = ACTIVATION;
    static constexpr TensorDType dot_accumulator = DOT_ACCUMULATOR;
    static constexpr TensorDType reduction_accumulator = REDUCTION_ACCUMULATOR;
    static constexpr TensorDType output = OUTPUT;
};

using Bf16QuantSpec = QuantSpec<
    WeightStorageFormat::BF16,
    0,
    0,
    StorageDType::BF16,
    StorageDType::NONE,
    QuantLayout::BF16_V1,
    QuantLayout::NONE>;

using Psq4QuantSpec = QuantSpec<
    WeightStorageFormat::PSQ4,
    32,
    32,
    StorageDType::U8,
    StorageDType::BF16,
    QuantLayout::PSQ4_SOA_V1,
    QuantLayout::PSQ_SCALE_BF16_S32_V1>;

using Psq8QuantSpec = QuantSpec<
    WeightStorageFormat::PSQ8,
    32,
    32,
    StorageDType::U8,
    StorageDType::BF16,
    QuantLayout::PSQ8_SOA_V1,
    QuantLayout::PSQ_SCALE_BF16_S32_V1>;

using Fp8Block128QuantSpec = QuantSpec<
    WeightStorageFormat::FP8_BLOCK128,
    128,
    128,
    StorageDType::U8,
    StorageDType::FP32,
    QuantLayout::FP8_E4M3_BLOCK128_V1,
    QuantLayout::FP8_SCALE_F32_N128K128_V1>;

using Mxfp4QuantSpec = QuantSpec<
    WeightStorageFormat::MXFP4,
    32,
    32,
    StorageDType::U8,
    StorageDType::U8,
    QuantLayout::MXFP4_E2M1_S32_V1,
    QuantLayout::MXFP4_SCALE_E8M0_S32_V1>;

using Bf16ComputeSpec = ComputeSpec<
    TensorDType::BF16,
    TensorDType::FP32,
    TensorDType::FP32,
    TensorDType::BF16>;

using W8A8ComputeSpec = ComputeSpec<
    TensorDType::INT8,
    TensorDType::INT32,
    TensorDType::FP32,
    TensorDType::BF16>;

using W4A8ComputeSpec = ComputeSpec<
    TensorDType::INT8,
    TensorDType::INT32,
    TensorDType::FP32,
    TensorDType::BF16>;

template<typename Spec>
struct QuantTraits;

template<>
struct QuantTraits<Bf16QuantSpec> {
    static constexpr bool has_codes = false;
    static constexpr bool has_scales = false;
};

template<>
struct QuantTraits<Psq4QuantSpec> {
    static constexpr bool has_codes = true;
    static constexpr bool has_scales = true;
};

template<>
struct QuantTraits<Psq8QuantSpec> {
    static constexpr bool has_codes = true;
    static constexpr bool has_scales = true;
};

template<>
struct QuantTraits<Fp8Block128QuantSpec> {
    static constexpr bool has_codes = true;
    static constexpr bool has_scales = true;
};

template<>
struct QuantTraits<Mxfp4QuantSpec> {
    static constexpr bool has_codes = true;
    static constexpr bool has_scales = true;
};

enum class QuantSpecId : uint8_t {
    BF16_V1,
    PSQ4_S32_BF16_V1,
    PSQ8_S32_BF16_V1,
    FP8_E4M3_S128X128_F32_V1,
    MXFP4_E2M1_S32_E8M0_V1,
};

enum class ComputeSpecId : uint8_t {
    BF16_F32_BF16 = 0,
    W8A8_I32_F32_BF16 = 1,
    W4A8_I32_F32_BF16 = 2,
    PSQ4_W4A8_F32_BF16 = 3,
    PSQ8_W8A8_F32_BF16 = 4,
    FP8_W8A8_F32_BF16 = 5,
    MXFP4_W4A8_F32_BF16 = 6,
};

struct ResolvedQuantSpec {
    QuantDescriptor descriptor;
    QuantSpecId spec_id = QuantSpecId::BF16_V1;
};

} // namespace ps::quantization
