#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/quantization/fpx/types.h>
#include <phaseshift/quantization/offline/qwen35_adapter.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ps::quantization::fpx {

class DFlash2Adapter {
public:
    static Result<Architecture> detect(const std::string& config_json);
    static TensorInfo classify(std::string_view name, const std::vector<int64_t>& shape);
};

}  // namespace ps::quantization::fpx
