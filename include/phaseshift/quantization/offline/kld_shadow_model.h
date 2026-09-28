#pragma once
#include <phaseshift/core/status.h>
#include <cstddef>
#include <string>

namespace ps::quantization::fpx {

struct ShadowModelOptions {
    std::string input_dir;
    std::string quantized_dir;
    std::string output_dir;
};

struct ShadowModelResult {
    std::string shadow_dir;
    std::string source_model_fingerprint;
    std::size_t patched_tensors = 0;
    std::size_t total_tensors = 0;
};

Result<ShadowModelResult> create_qdq_shadow_model(const ShadowModelOptions& options);

}
