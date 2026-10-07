#include <phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

__global__ void worker_probe_kernel(uint32_t* out, uint32_t value) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = value + gid;
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

    auto partition_res = mcu::GpuMcuCuPartition::create(0, 1u);
    check(partition_res.ok(), "partition created");
    if (!partition_res.ok()) {
        std::printf("partition error: %s\n", partition_res.status().message().c_str());
        return 1;
    }
    auto partition = partition_res.release();

    {
        std::vector<uint32_t> control(partition.hip_control_mask().size(), 0u);
        std::vector<uint32_t> worker(partition.hip_worker_mask().size(), 0u);
        check(hipExtStreamGetCUMask(partition.control_stream(),
                                    static_cast<uint32_t>(control.size()),
                                    control.data()) == hipSuccess &&
                  control == partition.hip_control_mask(),
              "control stream owns the control mask");
        check(hipExtStreamGetCUMask(partition.worker_stream(),
                                    static_cast<uint32_t>(worker.size()),
                                    worker.data()) == hipSuccess &&
                  worker == partition.hip_worker_mask(),
              "worker stream owns the worker mask");
    }

    constexpr uint32_t kElements = 32;
    uint32_t* output_dev = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&output_dev),
                    kElements * sizeof(uint32_t)) == hipSuccess,
          "probe output allocated");

    constexpr int kRounds = 3;
    for (int round = 0; round < kRounds; ++round) {
        auto mcu_res = mcu::GpuMcuPersistentMcu::create(0, partition.control_stream());
        check(mcu_res.ok(), "persistent mcu created");
        if (!mcu_res.ok()) break;
        auto mcu = mcu_res.release();

        (void)hipMemset(output_dev, 0, kElements * sizeof(uint32_t));
        check(hipStreamSynchronize(partition.worker_stream()) == hipSuccess,
              "worker stream quiesced before persistent start");

        check(mcu.start().ok(), "persistent mcu started");
        check(mcu.running(), "persistent mcu running");

        const uint64_t h0 = mcu.heartbeat();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        const uint64_t h1 = mcu.heartbeat();
        check(h1 > h0, "heartbeat advances");
        check(h1 > 0, "heartbeat observed after start");

        const uint32_t value = 500u + static_cast<uint32_t>(round) * 10u;
        worker_probe_kernel<<<1, kElements, 0, partition.worker_stream()>>>(
            output_dev, value);
        check(hipGetLastError() == hipSuccess, "worker probe launched while persistent");

        std::vector<uint32_t> host_output(kElements, 0u);
        check(hipMemcpyAsync(host_output.data(), output_dev,
                             kElements * sizeof(uint32_t), hipMemcpyDeviceToHost,
                             partition.worker_stream()) == hipSuccess,
              "probe readback enqueued on worker stream");
        check(hipStreamSynchronize(partition.worker_stream()) == hipSuccess,
              "worker stream synchronized independently");

        bool output_ok = true;
        for (uint32_t i = 0; i < kElements; ++i) {
            output_ok &= (host_output[i] == value + i);
        }
        check(output_ok, "worker probe ran while persistent mcu was live");

        check(mcu.request_stop().ok(), "stop requested");
        check(mcu.wait_stopped(5000).ok(), "persistent mcu stopped in bounded time");
        check(!mcu.running(), "persistent mcu not running after stop");
        check(mcu.shutdown().ok(), "persistent mcu shutdown");
    }

    check(hipStreamSynchronize(partition.control_stream()) == hipSuccess,
          "control stream clean after persistent shutdown");

    {
        auto idle = mcu::GpuMcuPersistentMcu::create(0, partition.control_stream());
        check(idle.ok(), "persistent mcu created without start");
        if (idle.ok()) {
            auto m = idle.release();
            check(m.shutdown().ok(), "shutdown without start");
        }
    }

    check(hipFree(output_dev) == hipSuccess, "probe output freed");
    check(partition.shutdown().ok(), "partition shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_persistent_mcu: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_persistent_mcu: PASS\n");
    return 0;
}
