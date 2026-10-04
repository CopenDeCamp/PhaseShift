#pragma once

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

inline constexpr const char* kGpuMcuProbeWorkerSymbol = "phaseshift_gpu_mcu_probe_worker";

struct GpuMcuWorkerArgs {
    uint64_t output = 0;
    uint64_t completion = 0;
    uint32_t value = 0;
    uint32_t element_count = 0;
    uint32_t generation = 0;
    uint32_t status_detail = 0;
    uint64_t timestamps = 0;
};

static_assert(sizeof(GpuMcuWorkerArgs) == 40);
static_assert(offsetof(GpuMcuWorkerArgs, output) == 0);
static_assert(offsetof(GpuMcuWorkerArgs, completion) == 8);
static_assert(offsetof(GpuMcuWorkerArgs, value) == 16);
static_assert(offsetof(GpuMcuWorkerArgs, element_count) == 20);
static_assert(offsetof(GpuMcuWorkerArgs, generation) == 24);
static_assert(offsetof(GpuMcuWorkerArgs, status_detail) == 28);
static_assert(offsetof(GpuMcuWorkerArgs, timestamps) == 32);

struct GpuMcuWorkerImage {
    const unsigned char* data = nullptr;
    unsigned int bytes = 0;
};

GpuMcuWorkerImage gpu_mcu_worker_image();

}  // namespace ps::runtime::gpu_mcu
