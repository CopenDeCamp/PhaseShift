#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/quantization/fpx/types.h>
#include <phaseshift/quantization/offline/qwen35_adapter.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ps::quantization::fpx {

Result<Architecture> detect_quantization_architecture(const std::string& config_json);

TensorInfo classify_quantization_tensor(
    Architecture architecture,
    std::string_view name,
    const std::vector<int64_t>& shape);

Result<uint32_t> quantization_num_layers(
    Architecture architecture,
    const std::string& config_json);

std::string_view quantization_main_layer_prefix(Architecture architecture);

}  // namespace ps::quantization::fpx
