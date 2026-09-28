#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/quantization/quantization_types.h>
#include <phaseshift/quantization/fpx/quantized_manifest.h>
#include <phaseshift/quantization/fpx/layout.h>

#include <cstdint>

namespace ps::quantization::fpx {

static_assert(kFpBlockSize == 32);

inline Result<ps::quantization::ResolvedQuantSpec> resolve_quant_spec(
    const QuantizedManifest& manifest,
    const QuantizedTensorMetadata& tensor)
{
    using ps::quantization::QuantDescriptor;
    using ps::quantization::QuantLayout;
    using ps::quantization::QuantSpecId;
    using ps::quantization::ResolvedQuantSpec;
    using ps::quantization::StorageDType;
    using ps::quantization::WeightStorageFormat;

    if (manifest.format != kQuantizedSafetensorsFormat) {
        return Status::invalid_argument("quantized manifest format mismatch", __FILE__, __LINE__);
    }
    if (manifest.format_version != kQuantizedSafetensorsFormatVersion) {
        return Status::invalid_argument("quantized manifest format_version mismatch", __FILE__, __LINE__);
    }

    if (tensor.logical_shape.empty()) {
        return Status::invalid_argument("quantized tensor logical_shape is empty", __FILE__, __LINE__);
    }
    for (const int64_t d : tensor.logical_shape) {
        if (d <= 0) {
            return Status::invalid_argument(
                "quantized tensor logical_shape has non-positive dim", __FILE__, __LINE__);
        }
    }
    const uint64_t logical_k = static_cast<uint64_t>(tensor.logical_shape.back());

    ResolvedQuantSpec resolved;
    QuantDescriptor& d = resolved.descriptor;
    d.format_version = manifest.format_version;

    if (tensor.encoding == QuantizedEncoding::Bf16) {
        if (!tensor.data) {
            return Status::invalid_argument("bf16 quantized tensor missing data ref", __FILE__, __LINE__);
        }
        if (tensor.codes || tensor.metadata1) {
            return Status::invalid_argument(
                "bf16 quantized tensor must not carry codes or scales", __FILE__, __LINE__);
        }
        if (tensor.data->dtype != "BF16") {
            return Status::invalid_argument(
                "bf16 quantized tensor data dtype is not BF16", __FILE__, __LINE__);
        }
        if (tensor.k_padded != logical_k) {
            return Status::invalid_argument(
                "bf16 quantized tensor k_padded does not match logical k", __FILE__, __LINE__);
        }
        d.format = WeightStorageFormat::BF16;
        d.group_size = 0;
        d.block_size = 0;
        d.storage_type = StorageDType::BF16;
        d.scale_type = StorageDType::NONE;
        d.weight_layout = QuantLayout::BF16_V1;
        d.scale_layout = QuantLayout::NONE;
        resolved.spec_id = QuantSpecId::BF16_V1;
        return resolved;
    }

    if (tensor.encoding == QuantizedEncoding::Psq4 ||
        tensor.encoding == QuantizedEncoding::Psq8) {
        if (!tensor.codes) {
            return Status::invalid_argument("quantized tensor missing codes ref", __FILE__, __LINE__);
        }
        if (!tensor.metadata1) {
            return Status::invalid_argument("quantized tensor missing scales ref", __FILE__, __LINE__);
        }
        if (tensor.data) {
            return Status::invalid_argument(
                "quantized tensor must not carry data ref", __FILE__, __LINE__);
        }
        const char* expected_codes_dtype = "U8";
        if (tensor.codes->dtype != expected_codes_dtype) {
            return Status::invalid_argument(
                "quantized tensor codes dtype mismatch", __FILE__, __LINE__);
        }
        if (tensor.metadata1->dtype != "U8") {
            return Status::invalid_argument(
                "quantized tensor scales dtype is not U8", __FILE__, __LINE__);
        }
        if (tensor.k_padded < logical_k || (tensor.k_padded % kFpBlockSize) != 0) {
            return Status::invalid_argument("quantized tensor k_padded invalid", __FILE__, __LINE__);
        }

        d.block_size = kFpBlockSize;

        if (tensor.encoding == QuantizedEncoding::Psq4) {
            d.format = WeightStorageFormat::PSQ4;
            d.group_size = kFpBlockSize;
            d.scale_type = StorageDType::BF16;
            d.weight_layout = QuantLayout::PSQ4_SOA_V1;
            d.scale_layout = QuantLayout::PSQ_SCALE_BF16_S32_V1;
            resolved.spec_id = QuantSpecId::PSQ4_S32_BF16_V1;
        } else {
            d.format = WeightStorageFormat::PSQ8;
            d.group_size = kFpBlockSize;
            d.scale_type = StorageDType::BF16;
            d.weight_layout = QuantLayout::PSQ8_SOA_V1;
            d.scale_layout = QuantLayout::PSQ_SCALE_BF16_S32_V1;
            resolved.spec_id = QuantSpecId::PSQ8_S32_BF16_V1;
        }
        return resolved;
    }

    if (tensor.encoding == QuantizedEncoding::Fp8Block128) {
        if (!tensor.codes) {
            return Status::invalid_argument("quantized tensor missing codes ref", __FILE__, __LINE__);
        }
        if (!tensor.metadata1) {
            return Status::invalid_argument("quantized tensor missing scales ref", __FILE__, __LINE__);
        }
        if (tensor.data) {
            return Status::invalid_argument(
                "quantized tensor must not carry data ref", __FILE__, __LINE__);
        }
        if (!tensor.layout.empty() && tensor.layout != kFp8Block128LayoutName) {
            return Status::invalid_argument("fp8 block128 layout mismatch", __FILE__, __LINE__);
        }
        if (tensor.codes->dtype != "U8") {
            return Status::invalid_argument(
                "quantized tensor codes dtype mismatch", __FILE__, __LINE__);
        }
        if (tensor.metadata1->dtype != "F32") {
            return Status::invalid_argument(
                "fp8 block128 scales dtype is not F32", __FILE__, __LINE__);
        }
        if (tensor.k_padded < logical_k || (tensor.k_padded % 128) != 0) {
            return Status::invalid_argument("quantized tensor k_padded invalid", __FILE__, __LINE__);
        }
        d.format = WeightStorageFormat::FP8_BLOCK128;
        d.group_size = 128;
        d.block_size = 128;
        d.storage_type = StorageDType::U8;
        d.scale_type = StorageDType::FP32;
        d.weight_layout = QuantLayout::FP8_E4M3_BLOCK128_V1;
        d.scale_layout = QuantLayout::FP8_SCALE_F32_N128K128_V1;
        resolved.spec_id = QuantSpecId::FP8_E4M3_S128X128_F32_V1;
        return resolved;
    }

    if (tensor.encoding == QuantizedEncoding::Mxfp4) {
        if (!tensor.codes) {
            return Status::invalid_argument("quantized tensor missing codes ref", __FILE__, __LINE__);
        }
        if (!tensor.metadata1) {
            return Status::invalid_argument("quantized tensor missing scales ref", __FILE__, __LINE__);
        }
        if (tensor.data) {
            return Status::invalid_argument(
                "quantized tensor must not carry data ref", __FILE__, __LINE__);
        }
        if (!tensor.layout.empty() && tensor.layout != kMxfp4LayoutName) {
            return Status::invalid_argument("mxfp4 layout mismatch", __FILE__, __LINE__);
        }
        if (tensor.codes->dtype != "U8") {
            return Status::invalid_argument(
                "quantized tensor codes dtype mismatch", __FILE__, __LINE__);
        }
        if (tensor.metadata1->dtype != "U8") {
            return Status::invalid_argument(
                "mxfp4 scales dtype is not U8", __FILE__, __LINE__);
        }
        if (tensor.k_padded < logical_k || (tensor.k_padded % 32) != 0) {
            return Status::invalid_argument("quantized tensor k_padded invalid", __FILE__, __LINE__);
        }
        d.format = WeightStorageFormat::MXFP4;
        d.group_size = 32;
        d.block_size = 32;
        d.storage_type = StorageDType::U8;
        d.scale_type = StorageDType::U8;
        d.weight_layout = QuantLayout::MXFP4_E2M1_S32_V1;
        d.scale_layout = QuantLayout::MXFP4_SCALE_E8M0_S32_V1;
        resolved.spec_id = QuantSpecId::MXFP4_E2M1_S32_E8M0_V1;
        return resolved;
    }

    return Status::invalid_argument("unsupported quantized tensor encoding", __FILE__, __LINE__);
}

// Preferred future GEMM compute contract for a resolved quant spec.
// This is not a currently bound executable kernel.
inline Result<ps::quantization::ComputeSpecId> preferred_gemm_compute_spec(
    ps::quantization::QuantSpecId quant_spec)
{
    switch (quant_spec) {
        case ps::quantization::QuantSpecId::BF16_V1:
            return ps::quantization::ComputeSpecId::BF16_F32_BF16;
        case ps::quantization::QuantSpecId::PSQ4_S32_BF16_V1:
            return ps::quantization::ComputeSpecId::PSQ4_W4A8_F32_BF16;
        case ps::quantization::QuantSpecId::PSQ8_S32_BF16_V1:
            return ps::quantization::ComputeSpecId::PSQ8_W8A8_F32_BF16;
        case ps::quantization::QuantSpecId::FP8_E4M3_S128X128_F32_V1:
            return ps::quantization::ComputeSpecId::FP8_W8A8_F32_BF16;
        case ps::quantization::QuantSpecId::MXFP4_E2M1_S32_E8M0_V1:
            return ps::quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16;
    }
    return Status::invalid_argument("unknown quant spec id", __FILE__, __LINE__);
}

// Compatibility (not preference) between a quant spec and a compute spec.
// Each quantized storage maps to exactly one executable compute contract.
inline bool is_gemm_compute_compatible(
    ps::quantization::QuantSpecId quant_spec,
    ps::quantization::ComputeSpecId compute_spec) noexcept {
    switch (quant_spec) {
        case ps::quantization::QuantSpecId::BF16_V1:
            return compute_spec == ps::quantization::ComputeSpecId::BF16_F32_BF16;
        case ps::quantization::QuantSpecId::PSQ4_S32_BF16_V1:
            return compute_spec == ps::quantization::ComputeSpecId::PSQ4_W4A8_F32_BF16;
        case ps::quantization::QuantSpecId::PSQ8_S32_BF16_V1:
            return compute_spec == ps::quantization::ComputeSpecId::PSQ8_W8A8_F32_BF16;
        case ps::quantization::QuantSpecId::FP8_E4M3_S128X128_F32_V1:
            return compute_spec == ps::quantization::ComputeSpecId::FP8_W8A8_F32_BF16;
        case ps::quantization::QuantSpecId::MXFP4_E2M1_S32_E8M0_V1:
            return compute_spec == ps::quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16;
    }
    return false;
}

inline Status validate_quantized_manifest_contract(const QuantizedManifest& manifest) {
    if (manifest.format != kQuantizedSafetensorsFormat) {
        return Status::invalid_argument("quantized manifest format mismatch", __FILE__, __LINE__);
    }
    if (manifest.format_version != kQuantizedSafetensorsFormatVersion) {
        return Status::invalid_argument("quantized manifest format_version mismatch", __FILE__, __LINE__);
    }
    if (manifest.tensors.empty()) {
        return Status::invalid_argument("quantized manifest has no tensors", __FILE__, __LINE__);
    }
    for (const auto& kv : manifest.tensors) {
        auto r = resolve_quant_spec(manifest, kv.second);
        if (!r.ok()) {
            return r.status();
        }
    }
    return Status::make_ok();
}

} // namespace ps::quantization::fpx
