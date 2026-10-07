#pragma once

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

inline constexpr const char* kGpuMcuFsmWorkerSymbol = "phaseshift_gpu_mcu_fsm_worker";

struct GpuMcuFsmWorkerArgs {
    uint64_t output = 0;
    uint64_t device_completion = 0;
    uint64_t input = 0;
    uint64_t timestamps = 0;
    uint32_t value = 0;
    uint32_t element_count = 0;
    uint32_t generation = 0;
    uint32_t status_detail = 0;
    uint32_t work = 0;
    uint32_t flags = 0;
};

static_assert(sizeof(GpuMcuFsmWorkerArgs) == 56);
static_assert(offsetof(GpuMcuFsmWorkerArgs, output) == 0);
static_assert(offsetof(GpuMcuFsmWorkerArgs, device_completion) == 8);
static_assert(offsetof(GpuMcuFsmWorkerArgs, input) == 16);
static_assert(offsetof(GpuMcuFsmWorkerArgs, timestamps) == 24);
static_assert(offsetof(GpuMcuFsmWorkerArgs, value) == 32);
static_assert(offsetof(GpuMcuFsmWorkerArgs, element_count) == 36);
static_assert(offsetof(GpuMcuFsmWorkerArgs, generation) == 40);
static_assert(offsetof(GpuMcuFsmWorkerArgs, status_detail) == 44);
static_assert(offsetof(GpuMcuFsmWorkerArgs, work) == 48);
static_assert(offsetof(GpuMcuFsmWorkerArgs, flags) == 52);

}  // namespace ps::runtime::gpu_mcu
