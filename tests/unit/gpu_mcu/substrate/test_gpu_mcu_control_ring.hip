#include <phaseshift/runtime/gpu_mcu/io/control_ring.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr uint64_t kTornMult = 0x9E3779B97F4A7C15ull;

void encode_payload(uint64_t sequence, mcu::ControlRingPayload& payload) {
    uint64_t words[4];
    words[0] = sequence;
    words[1] = ~sequence;
    words[2] = sequence * kTornMult;
    words[3] = words[0] ^ words[1] ^ words[2];
    std::memcpy(payload.bytes, words, sizeof(words));
}

bool decode_payload(const mcu::ControlRingPayload& payload,
                    uint64_t& sequence,
                    bool& torn_ok) {
    uint64_t words[4];
    std::memcpy(words, payload.bytes, sizeof(words));
    sequence = words[0];
    torn_ok = (words[1] == ~words[0]) && (words[2] == words[0] * kTornMult) &&
              (words[3] == (words[0] ^ words[1] ^ words[2]));
    return true;
}

__global__ void ring_echo_service(
    mcu::ControlRing* input,
    mcu::ControlRing* output,
    uint32_t* stop) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t in_position = 0;
    uint64_t out_position = 0;
    while (true) {
        mcu::ControlRingPayload payload{};
        if (mcu::control_ring_try_pop(input, in_position, payload)) {
            while (!mcu::control_ring_try_push(output, out_position, payload)) {
                __builtin_amdgcn_s_sleep(4);
            }
            continue;
        }
        if (__scoped_atomic_load_n(stop, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) !=
            0u) {
            break;
        }
        __builtin_amdgcn_s_sleep(4);
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

    uint64_t records = 1000000ull;
    if (const char* env = std::getenv("PHASESHIFT_GPU_MCU_RING_STRESS")) {
        const long long parsed = std::atoll(env);
        if (parsed > 0) records = static_cast<uint64_t>(parsed);
    }

    auto ring_res = mcu::GpuMcuControlRing::create(0);
    check(ring_res.ok(), "control ring created");
    if (!ring_res.ok()) {
        std::printf("ring error: %s\n", ring_res.status().message().c_str());
        return 1;
    }
    auto ring = ring_res.release();
    check(ring.device_input() != nullptr && ring.device_events() != nullptr,
          "device rings mapped");

    uint32_t* host_stop = nullptr;
    if (hipHostMalloc(&host_stop, sizeof(uint32_t),
                      hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
        std::printf("FAIL: stop flag allocation\n");
        return 1;
    }
    *host_stop = 0;
    uint32_t* device_stop = nullptr;
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&device_stop), host_stop, 0) !=
        hipSuccess) {
        std::printf("FAIL: stop flag mapping\n");
        return 1;
    }

    hipStream_t service_stream = nullptr;
    if (hipStreamCreate(&service_stream) != hipSuccess) {
        std::printf("FAIL: service stream\n");
        return 1;
    }
    ring_echo_service<<<1, 32, 0, service_stream>>>(
        ring.device_input(), ring.device_events(), device_stop);
    check(hipGetLastError() == hipSuccess, "echo service launched");

    const uint64_t window = mcu::kControlRingCapacity - 1u;
    uint64_t pushed = 0;
    uint64_t popped = 0;
    uint64_t lost = 0;
    uint64_t duplicate = 0;
    uint64_t reorder = 0;
    uint64_t torn = 0;
    uint64_t push_backpressure = 0;

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    const auto start = std::chrono::steady_clock::now();
    while (popped < records) {
        while (pushed < records && (pushed - popped) < window) {
            mcu::ControlRingPayload payload{};
            encode_payload(pushed, payload);
            if (!ring.try_push_input(payload)) {
                ++push_backpressure;
                break;
            }
            ++pushed;
        }
        bool progress = false;
        for (uint32_t k = 0; k < mcu::kControlRingCapacity; ++k) {
            mcu::ControlRingPayload event{};
            if (!ring.try_pop_event(event)) break;
            uint64_t sequence = 0;
            bool torn_ok = false;
            decode_payload(event, sequence, torn_ok);
            if (sequence == popped) {
                ++popped;
            } else if (sequence < popped) {
                ++duplicate;
            } else {
                ++lost;
                ++reorder;
                ++popped;
            }
            if (!torn_ok) ++torn;
            progress = true;
        }
        if (!progress) {
            if (std::chrono::steady_clock::now() >= deadline) {
                check(false, "ring stress timed out");
                break;
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    check(popped == records, "all records echoed");
    check(pushed == records, "all records pushed");
    check(lost == 0, "no lost records");
    check(duplicate == 0, "no duplicate records");
    check(reorder == 0, "no reordered records");
    check(torn == 0, "no torn records");

    std::printf("ring stress: records=%llu elapsed=%.1fms rate=%.2fM/s "
                "backpressure=%llu\n",
                static_cast<unsigned long long>(records), elapsed_ms,
                records / elapsed_ms / 1000.0,
                static_cast<unsigned long long>(push_backpressure));

    *host_stop = 1;
    std::atomic_thread_fence(std::memory_order_release);
    check(hipStreamSynchronize(service_stream) == hipSuccess, "echo service stopped");

    check(ring.shutdown().ok(), "ring shutdown");
    check(hipStreamDestroy(service_stream) == hipSuccess, "service stream destroyed");
    check(hipFreeHost(host_stop) == hipSuccess, "stop flag freed");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_control_ring: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_control_ring: PASS\n");
    return 0;
}
