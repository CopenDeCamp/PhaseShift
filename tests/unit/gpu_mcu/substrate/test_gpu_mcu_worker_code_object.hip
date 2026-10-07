#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/worker_image.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    check(image.data != nullptr && image.bytes > 0, "worker HSACO embedded");
    check(std::memcmp(image.data, "\x7f" "ELF", 4) == 0, "worker HSACO is an ELF image");
    if (g_failed != 0) return 1;

    auto queue_res = mcu::GpuMcuAqlQueue::create(
        0, mcu::AqlQueueOwner::GpuMcu, mcu::kAqlDefaultQueueSize);
    check(queue_res.ok(), "queue created");
    if (!queue_res.ok()) {
        std::printf("queue error: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    {
        auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
            queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
        check(code_res.ok(), "worker code object loaded from memory");
        if (code_res.ok()) {
            auto code = code_res.release();
            const auto& meta = code.metadata();
            std::printf("probe worker meta: kernarg=%u group=%u private=%u kernel_object=%p\n",
                        meta.kernarg_segment_size, meta.static_group_segment_size,
                        meta.private_segment_size,
                        reinterpret_cast<void*>(meta.kernel_object));
            check(meta.kernel_object != 0, "kernel_object resolved");
            check(meta.kernarg_segment_size >= sizeof(mcu::GpuMcuWorkerArgs),
                  "kernarg segment covers the worker args");
            check(code.shutdown().ok(), "memory code object shutdown");
        } else {
            std::printf("load_memory error: %s\n", code_res.status().message().c_str());
        }
    }

#ifdef PHASESHIFT_GPU_MCU_PROBE_HSACO
    {
        auto code_res = mcu::GpuMcuAqlCodeObject::load(
            queue, PHASESHIFT_GPU_MCU_PROBE_HSACO, mcu::kGpuMcuProbeWorkerSymbol);
        check(code_res.ok(), "worker code object loaded from file");
        if (code_res.ok()) {
            auto code = code_res.release();
            check(code.metadata().kernel_object != 0, "file kernel_object resolved");
            check(code.shutdown().ok(), "file code object shutdown");
        } else {
            std::printf("load error: %s\n", code_res.status().message().c_str());
        }
    }
#endif

    check(queue.shutdown().ok(), "queue shutdown");

    {
        auto missing = mcu::GpuMcuAqlCodeObject::load_memory(
            queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
        check(!missing.ok(), "load on shut down queue rejected");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_worker_code_object: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_worker_code_object: PASS\n");
    return 0;
}
