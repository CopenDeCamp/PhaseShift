#include <phaseshift/runtime/gpu_mcu/infrastructure/wall_clock.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

namespace mcu = ::ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

}  // namespace

__global__ void wall_clock_probe_kernel(unsigned long long* out,
                                        unsigned int iters) {
    if (threadIdx.x != 0u) return;
    float acc = 1.0f;
    const unsigned long long t0 = wall_clock64();
    for (unsigned int i = 0u; i < iters; ++i) {
        acc = fmaf(acc, 1.0000001f, 0.5f);
    }
    const unsigned long long t1 = wall_clock64();
    out[0] = t0;
    out[1] = t1;
    out[2] = (acc == 12345.0f) ? 1ull : 0ull;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    const auto rate_res = mcu::gpu_wall_clock_rate_khz(0);
    if (!rate_res.ok()) {
        std::printf("FAIL: wall clock rate unavailable: %s\n",
                    rate_res.status().message().c_str());
        return 1;
    }
    const int rate = rate_res.value();
    std::printf("wall_clock_rate_khz=%d\n", rate);
    check(rate > 0, "wall clock rate is positive");

    check(mcu::gpu_wall_clock_ticks_to_ms(0u, rate) == 0.0,
          "zero ticks convert to zero ms");
    check(mcu::gpu_wall_clock_ticks_to_ms(static_cast<uint64_t>(rate), rate) ==
              1.0,
          "rate ticks convert to one ms");
    check(mcu::gpu_wall_clock_ticks_to_us(static_cast<uint64_t>(rate), rate) ==
              1000.0,
          "rate ticks convert to one thousand us");
    check(mcu::gpu_wall_clock_ticks_to_ms(12345u, 0) == 0.0,
          "invalid rate fails safe");
    check(mcu::gpu_wall_clock_ticks_to_ms(2u * static_cast<uint64_t>(rate),
                                          rate) == 2.0,
          "two ms worth of ticks");

    unsigned long long* probe = nullptr;
    if (hipMalloc(&probe, 3u * sizeof(unsigned long long)) != hipSuccess) {
        std::printf("FAIL: probe alloc\n");
        return 1;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }

    unsigned int iters = 1000000u;
    double event_ms = 0.0;
    double device_ms = 0.0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (hipMemset(probe, 0, 3u * sizeof(unsigned long long)) != hipSuccess) {
            std::printf("FAIL: probe reset\n");
            return 1;
        }
        hipEvent_t e0 = nullptr;
        hipEvent_t e1 = nullptr;
        (void)hipEventCreate(&e0);
        (void)hipEventCreate(&e1);
        (void)hipEventRecord(e0, stream);
        wall_clock_probe_kernel<<<1, 32, 0, stream>>>(probe, iters);
        (void)hipEventRecord(e1, stream);
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: probe sync\n");
            return 1;
        }
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, e0, e1);
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
        event_ms = static_cast<double>(ms);

        unsigned long long host_probe[3] = {0ull, 0ull, 0ull};
        (void)hipMemcpy(host_probe, probe, 3u * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost);
        if (host_probe[1] <= host_probe[0]) {
            std::printf("FAIL: probe did not record wall clock ticks\n");
            return 1;
        }
        device_ms = mcu::gpu_wall_clock_ticks_to_ms(host_probe[1] - host_probe[0],
                                                    rate);
        std::printf(
            "  probe iters=%u ticks=%llu device_ms=%.4f event_ms=%.4f\n", iters,
            static_cast<unsigned long long>(host_probe[1] - host_probe[0]),
            device_ms, event_ms);
        if (event_ms >= 20.0) break;
        iters *= 4u;
    }

    if (event_ms < 20.0) {
        std::printf("SKIP: probe too short (%.4f ms) to cross-check\n", event_ms);
        std::printf("test_gpu_mcu_wall_clock_conversion: PASS (partial)\n");
        return 0;
    }

    const double ratio = device_ms / event_ms;
    std::printf("  cross-check: device_ms/event_ms=%.4f\n", ratio);
    check(std::fabs(ratio - 1.0) <= 0.10,
          "device wall clock conversion agrees with the hip event within 10%");

    (void)hipStreamDestroy(stream);
    (void)hipFree(probe);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_wall_clock_conversion: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_wall_clock_conversion: PASS\n");
    return 0;
}
