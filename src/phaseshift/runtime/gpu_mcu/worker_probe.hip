#include <phaseshift/runtime/gpu_mcu/completion.h>
#include <phaseshift/runtime/gpu_mcu/device_completion.h>
#include <phaseshift/runtime/gpu_mcu/fsm_worker.h>
#include <phaseshift/runtime/gpu_mcu/worker_image.h>

#include <hip/hip_runtime.h>

extern "C" __global__ void phaseshift_gpu_mcu_fsm_worker(
    ps::runtime::gpu_mcu::GpuMcuFsmWorkerArgs args) {
    using ps::runtime::gpu_mcu::GpuMcuDeviceCompletion;
    using ps::runtime::gpu_mcu::GpuMcuDeviceCompletionState;

    auto* output = reinterpret_cast<uint32_t*>(args.output);
    auto* completion = reinterpret_cast<GpuMcuDeviceCompletion*>(args.device_completion);
    auto* timestamps = reinterpret_cast<unsigned long long*>(args.timestamps);
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0u) {
        if (timestamps != nullptr) {
            timestamps[0] = static_cast<unsigned long long>(wall_clock64());
        }
        ps::runtime::gpu_mcu::gpu_mcu_device_completion_publish(
            completion, args.generation,
            static_cast<uint32_t>(GpuMcuDeviceCompletionState::running), 0u);
    }

    volatile float accumulator = static_cast<float>(args.value & 0xffffu);
    for (uint32_t i = 0; i < args.work; ++i) {
        accumulator = fmaf(accumulator, 1.0000001f, 1e-7f);
    }

    const uint32_t base =
        args.input != 0u ? *reinterpret_cast<const uint32_t*>(args.input) : 0u;
    if (gid < args.element_count) {
        output[gid] = base + args.value + gid;
    }
    __syncthreads();

    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        if (accumulator == 12345.678f) {
            output[0] = 0xdeadbeefu;
        }
        if (timestamps != nullptr) {
            timestamps[1] = static_cast<unsigned long long>(wall_clock64());
        }
        ps::runtime::gpu_mcu::gpu_mcu_device_completion_publish(
            completion, args.generation,
            static_cast<uint32_t>(GpuMcuDeviceCompletionState::complete),
            args.status_detail);
    }
}

extern "C" __global__ void phaseshift_gpu_mcu_probe_worker(
    ps::runtime::gpu_mcu::GpuMcuWorkerArgs args) {
    using ps::runtime::gpu_mcu::GpuMcuCompletion;
    using ps::runtime::gpu_mcu::GpuMcuCompletionState;

    auto* output = reinterpret_cast<uint32_t*>(args.output);
    auto* completion = reinterpret_cast<GpuMcuCompletion*>(args.completion);
    auto* timestamps = reinterpret_cast<unsigned long long*>(args.timestamps);
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid == 0) {
        ps::runtime::gpu_mcu::gpu_mcu_completion_publish(
            completion, static_cast<uint32_t>(GpuMcuCompletionState::running), 0u,
            args.generation);
        if (timestamps != nullptr) {
            timestamps[0] = static_cast<unsigned long long>(wall_clock64());
        }
    }

    if (gid < args.element_count) {
        output[gid] = args.value + gid;
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (timestamps != nullptr) {
            timestamps[1] = static_cast<unsigned long long>(wall_clock64());
        }
        ps::runtime::gpu_mcu::gpu_mcu_completion_publish(
            completion, static_cast<uint32_t>(GpuMcuCompletionState::complete),
            args.status_detail, args.generation);
    }
}
