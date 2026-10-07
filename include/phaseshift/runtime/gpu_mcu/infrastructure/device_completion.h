#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

enum class GpuMcuDeviceCompletionState : uint32_t {
    idle = 0,
    running = 1,
    complete = 2,
    error = 3,
};

struct alignas(64) GpuMcuDeviceCompletion {
    uint32_t generation = 0;
    uint32_t state = 0;
    uint32_t detail = 0;
    uint32_t reserved0 = 0;
    uint64_t aux = 0;
    uint64_t reserved1[5] = {};
};

static_assert(sizeof(GpuMcuDeviceCompletion) == 64);
static_assert(alignof(GpuMcuDeviceCompletion) == 64);
static_assert(offsetof(GpuMcuDeviceCompletion, generation) == 0);
static_assert(offsetof(GpuMcuDeviceCompletion, state) == 4);
static_assert(offsetof(GpuMcuDeviceCompletion, detail) == 8);

__device__ __forceinline__ void gpu_mcu_device_completion_publish(
    GpuMcuDeviceCompletion* completion,
    uint32_t generation,
    uint32_t state,
    uint32_t detail) {
    completion->detail = detail;
    __scoped_atomic_store_n(&completion->state,
                            static_cast<uint32_t>(GpuMcuDeviceCompletionState::idle),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
    __scoped_atomic_store_n(&completion->generation, generation, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_DEVICE);
    __scoped_atomic_store_n(&completion->state, state, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_DEVICE);
}

__device__ __forceinline__ uint32_t gpu_mcu_device_completion_observe(
    const GpuMcuDeviceCompletion* completion) {
    return __scoped_atomic_load_n(&completion->state, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE);
}

__device__ __forceinline__ bool gpu_mcu_device_completion_matches(
    const GpuMcuDeviceCompletion* completion,
    uint32_t expected_generation) {
    return gpu_mcu_device_completion_observe(completion) ==
               static_cast<uint32_t>(GpuMcuDeviceCompletionState::complete) &&
           __scoped_atomic_load_n(&completion->generation, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) == expected_generation;
}

}  // namespace ps::runtime::gpu_mcu
