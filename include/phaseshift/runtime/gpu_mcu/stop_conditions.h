#pragma once

#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct GpuMcuStopConditions {
    const int32_t* token_ids = nullptr;
    uint32_t token_count = 0u;
    uint32_t flags = 0u;
};

}  // namespace ps::runtime::gpu_mcu
