#pragma once
#include <phaseshift/quantization/fpx/types.h>
#include <phaseshift/core/status.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ps::quantization::fpx {

struct TensorInfo {
    TensorRole role = TensorRole::Unknown;
    int32_t layer_index = -1;
    bool is_weight = false;
    bool is_visual = false;
    bool malformed_name = false;
};

class Qwen35Adapter {
public:
    static Result<Architecture> detect(const std::string& config_json);
    static TensorInfo classify(std::string_view name, const std::vector<int64_t>& shape);
};

}  // namespace ps::quantization::fpx
