#pragma once

#include <cstdint>

namespace ps::runtime::gpu_mcu {

enum : uint32_t {
    kMcuKernargSourcePrepared = 1u << 0,
    kMcuKernargSourceSupervisorProbe = 1u << 1,
};

struct alignas(16) McuKernargSourceDesc {
    uint64_t source = 0;
    uint32_t explicit_args_bytes = 0;
    uint32_t flags = 0;
};

static_assert(sizeof(McuKernargSourceDesc) == 16);
static_assert(alignof(McuKernargSourceDesc) == 16);

}  // namespace ps::runtime::gpu_mcu
