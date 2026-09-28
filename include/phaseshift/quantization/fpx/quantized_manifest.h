#pragma once
#include <phaseshift/core/status.h>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ps::quantization::fpx {

constexpr const char* kQuantizedSafetensorsFormat = "phaseshift-fpx-safetensors";
constexpr uint32_t kQuantizedSafetensorsFormatVersion = 3;
constexpr const char* kQuantizedMetadataFile = "phaseshift_quantization.json";
constexpr const char* kFp8Block128EncodingName = "fp8_e4m3_block128_f32";
constexpr const char* kMxfp4EncodingName = "mxfp4_e2m1_e8m0_s32";
constexpr const char* kFp8Block128LayoutName = "fp8_e4m3_row_major_block128_v1";
constexpr const char* kMxfp4LayoutName = "mxfp4_e2m1_row_major_s32_v1";
constexpr const char* kCodesSuffix = ".__phaseshift_codes";
constexpr const char* kMetadata1Suffix = ".__phaseshift_metadata1";
constexpr const char* kMetadata2Suffix = ".__phaseshift_metadata2";
constexpr const char* kMetadata3Suffix = ".__phaseshift_metadata3";
constexpr const char* kMetadata4Suffix = ".__phaseshift_metadata4";

enum class QuantizedEncoding : uint8_t {
    Bf16 = 0,
    Psq4 = 3,
    Psq8 = 4,
    Fp8Block128 = 5,
    Mxfp4 = 6,
};

struct QuantizedTensorRef {
    std::string tensor;
    std::string dtype;
    std::string crc32;
};

struct QuantizedTensorMetadata {
    std::string role;
    QuantizedEncoding encoding = QuantizedEncoding::Bf16;
    std::string codebook;
    std::string layout;
    std::vector<int64_t> logical_shape;
    uint64_t k_padded = 0;
    std::optional<QuantizedTensorRef> codes;
    std::optional<QuantizedTensorRef> metadata1;
    std::optional<QuantizedTensorRef> metadata2;
    std::optional<QuantizedTensorRef> metadata3;
    std::optional<QuantizedTensorRef> metadata4;
    std::optional<QuantizedTensorRef> data;
};

struct QuantizedImatrixMetadata {
    bool enabled = false;
    std::string policy;
    std::string sha256;
    std::string model_fingerprint;
    std::string corpus_sha256;
    uint64_t required = 0;
    uint64_t matched = 0;
};

struct QuantizedManifest {
    std::string format = kQuantizedSafetensorsFormat;
    uint32_t format_version = kQuantizedSafetensorsFormatVersion;
    std::string architecture;
    std::string preset;
    std::string scope = "full";
    std::string source_dtype;
    std::string source_model_fingerprint;
    uint64_t logical_parameter_count = 0;
    uint64_t payload_bytes = 0;
    uint64_t padding_bytes = 0;
    double effective_payload_bpw = 0.0;
    std::map<std::string, std::string> aliases;
    std::optional<QuantizedImatrixMetadata> imatrix;
    std::map<std::string, QuantizedTensorMetadata> tensors;
};

std::string quantized_encoding_name(QuantizedEncoding enc);
Result<QuantizedEncoding> parse_quantized_encoding(const std::string& name);

Result<QuantizedManifest> parse_quantized_manifest(const std::string& json);
Result<std::string> serialize_quantized_manifest(const QuantizedManifest& manifest);

}
