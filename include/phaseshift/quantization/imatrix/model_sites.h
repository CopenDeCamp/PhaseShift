#pragma once
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <cstdint>
#include <string>
#include <vector>

namespace ps::quantization::imatrix {

struct ImatrixSitePlan {
    uint32_t tag = 0;
    uint64_t k = 0;
};

std::vector<ImatrixSitePlan> build_site_plan(const ps::qwen35::Qwen35TextConfig& config);
std::vector<std::string> tensor_names_for(uint32_t tag);

}
