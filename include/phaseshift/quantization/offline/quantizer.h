#pragma once
#include <phaseshift/quantization/fpx/types.h>
#include <phaseshift/quantization/fpx/quantized_manifest.h>
#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <cstdint>
#include <map>
#include <string>

namespace ps::quantization::fpx {

enum class ConvertBackend {
    Cpu,
    Hip,
};

enum class ImatrixPolicy {
    Strict,
    BestEffort,
};

struct ImatrixCoverage {
    bool used = false;
    ImatrixPolicy policy = ImatrixPolicy::Strict;
    uint64_t required = 0;
    uint64_t found = 0;
    uint64_t missing = 0;
    uint64_t k_mismatch = 0;
    double coverage = 0.0;
};

struct QuantizeOptions {
    std::string input_dir;
    std::string output_dir;
    FpxPreset preset = FpxPreset::Psq;
    bool text_only_scope = false;
    ConvertBackend backend = ConvertBackend::Cpu;
    std::string imatrix_path;
    ImatrixPolicy imatrix_policy = ImatrixPolicy::Strict;
    uint64_t max_shard_size = 4ull * 1024ull * 1024ull * 1024ull;
    bool overwrite = false;
    bool verify = true;
};

struct FormatSummary {
    uint64_t tensors = 0;
    uint64_t parameters = 0;
    uint64_t codes_bytes = 0;
    uint64_t metadata_bytes = 0;

    uint64_t total_bytes() const { return codes_bytes + metadata_bytes; }
};

struct QuantizeSummary {
    QuantizedManifest manifest;
    uint64_t shards = 0;
    std::map<WeightEncoding, FormatSummary> formats;
    uint64_t padding_bytes = 0;
    uint64_t skipped_tensors = 0;
    uint64_t logical_parameter_count = 0;
    double effective_bpw = 0.0;
    ImatrixCoverage imatrix;
    bool self_contained = false;
};

Result<QuantizeSummary> quantize_model(const QuantizeOptions& opts);

// Independent validation mode: opens an existing self-contained model
// directory and validates it with QuantizedModelReader.
Result<QuantizedModelReader> validate_model_dir(const std::string& output_dir);

// Parses a human-readable byte size like "4GiB", "4096MiB", "4000000000".
Result<uint64_t> parse_byte_size(const std::string& s);

}
