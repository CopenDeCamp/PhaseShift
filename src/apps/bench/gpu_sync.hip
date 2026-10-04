#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/cu_partition.h>

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace cg = cooperative_groups;

struct Options {
    int warmup = 5;
    int iterations = 30;
    int n = 1000;
    int device = 0;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: phaseshift-bench gpu-sync [options]\n"
                 "  --warmup N\n"
                 "  --iterations N\n"
                 "  --n N            barrier iterations per launch\n"
                 "  --device N\n");
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        q * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

__device__ unsigned long long g_sink = 0;

__global__ void loop_empty(int n) {
    unsigned long long acc = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<unsigned long long>(i) & 1u;
    }
    if (acc == 0xFFFFFFFFull) g_sink = acc;
}

__global__ void loop_syncthreads(int n) {
    unsigned long long acc = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        __syncthreads();
        acc += static_cast<unsigned long long>(i) & 1u;
    }
    if (acc == 0xFFFFFFFFull) g_sink = acc;
}

__global__ void loop_lds_barrier(int n) {
    __shared__ int shared[128];
    shared[threadIdx.x & 127] = threadIdx.x;
    unsigned long long acc = 0;
    for (int i = 0; i < n; ++i) {
        __syncthreads();
        acc += static_cast<unsigned long long>(shared[(threadIdx.x + i) & 127]);
        shared[threadIdx.x & 127] = static_cast<int>(acc);
    }
    if (acc == 0xFFFFFFFFull) g_sink = acc;
}

__global__ void loop_threadfence(int n) {
    unsigned long long acc = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        __threadfence();
        acc += static_cast<unsigned long long>(i) & 1u;
    }
    if (acc == 0xFFFFFFFFull) g_sink = acc;
}

__global__ void loop_threadfence_system(int n) {
    unsigned long long acc = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        __threadfence_system();
        acc += static_cast<unsigned long long>(i) & 1u;
    }
    if (acc == 0xFFFFFFFFull) g_sink = acc;
}

__global__ void loop_grid_sync(int n) {
    cg::grid_group grid = cg::this_grid();
    unsigned long long acc = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        grid.sync();
        acc += static_cast<unsigned long long>(i) & 1u;
    }
    if (acc == 0xFFFFFFFFull) g_sink = acc;
}

double measure(hipStream_t stream, void (*kernel)(int), int n, int blocks, int threads,
               const Options& opt, bool cooperative) {
    std::vector<double> samples;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);
        (void)hipEventRecord(start, stream);
        if (cooperative) {
            int n_arg = n;
            void* args[] = {&n_arg};
            (void)hipLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                             dim3(blocks), dim3(threads), args, 0,
                                             stream);
        } else {
            kernel<<<blocks, threads, 0, stream>>>(n);
        }
        (void)hipEventRecord(stop, stream);
        if (hipEventSynchronize(stop) != hipSuccess) {
            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);
            return -1.0;
        }
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
        if (it < opt.warmup) continue;
        samples.push_back(static_cast<double>(ms) * 1000.0);
    }
    return percentile(samples, 0.5);
}

void report(const char* name, double total_us, int n) {
    if (total_us < 0.0) {
        std::printf("  %-26s FAILED\n", name);
        return;
    }
    std::printf("  %-26s total_us=%9.3f per_iter_ns=%8.2f\n", name, total_us,
                n > 0 ? total_us * 1000.0 / n : 0.0);
}

}  // namespace

int run_gpu_sync(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int& out) {
            if (i + 1 < argc) out = std::atoi(argv[++i]);
        };
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        }
        if (a == "--warmup") next(opt.warmup);
        else if (a == "--iterations") next(opt.iterations);
        else if (a == "--n") next(opt.n);
        else if (a == "--device") next(opt.device);
        else {
            usage(stderr);
            return 2;
        }
    }
    if (hipSetDevice(opt.device) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    hipDeviceProp_t prop{};
    (void)hipGetDeviceProperties(&prop, opt.device);

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    const int blocks = 32;
    const int threads = 256;

    std::printf("gpu-sync n=%d iterations=%d blocks=%d threads=%d mp=%d\n", opt.n,
                opt.iterations, blocks, threads, prop.multiProcessorCount);
    std::printf("GPU side (median total, and per-iteration):\n");
    const double empty_us = measure(stream, loop_empty, opt.n, blocks, threads, opt, false);
    report("loop_overhead", empty_us, opt.n);
    report("syncthreads", measure(stream, loop_syncthreads, opt.n, blocks, threads, opt, false), opt.n);
    report("lds_syncthreads", measure(stream, loop_lds_barrier, opt.n, blocks, threads, opt, false), opt.n);
    report("threadfence", measure(stream, loop_threadfence, opt.n, blocks, threads, opt, false), opt.n);
    report("threadfence_system", measure(stream, loop_threadfence_system, opt.n, blocks, threads, opt, false), opt.n);

    int cooperative = 0;
    (void)hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch,
                                opt.device);
    if (cooperative != 0) {
        int per_mp = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_mp, reinterpret_cast<const void*>(loop_grid_sync), threads, 0);
        const int grid = prop.multiProcessorCount * std::max(per_mp, 1);
        std::printf("  grid_barrier resident_blocks=%d (per_mp=%d)\n", grid, per_mp);
        report("grid_sync", measure(stream, loop_grid_sync, opt.n, grid, threads, opt, true), opt.n);
    } else {
        std::printf("  grid_sync UNSUPPORTED\n");
    }

    std::printf("Host side (median per call):\n");
    {
        std::vector<double> event_us;
        std::vector<double> stream_us;
        hipEvent_t e = nullptr;
        (void)hipEventCreate(&e);
        for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
            (void)hipEventRecord(e, stream);
            const auto t0 = std::chrono::steady_clock::now();
            (void)hipEventSynchronize(e);
            const auto t1 = std::chrono::steady_clock::now();
            if (it < opt.warmup) continue;
            event_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
            (void)hipEventRecord(e, stream);
            const auto t0 = std::chrono::steady_clock::now();
            (void)hipStreamSynchronize(stream);
            const auto t1 = std::chrono::steady_clock::now();
            if (it < opt.warmup) continue;
            stream_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        (void)hipEventDestroy(e);
        std::printf("  %-26s p50_us=%9.3f p95_us=%9.3f\n", "hipEventSynchronize",
                    percentile(event_us, 0.5), percentile(event_us, 0.95));
        std::printf("  %-26s p50_us=%9.3f p95_us=%9.3f\n", "hipStreamSynchronize",
                    percentile(stream_us, 0.5), percentile(stream_us, 0.95));
    }

    if (hsa_init() == HSA_STATUS_SUCCESS) {
        hsa_signal_t signal{};
        if (hsa_signal_create(1, 0, nullptr, &signal) == HSA_STATUS_SUCCESS) {
            std::vector<double> hsa_us;
            for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
                hsa_signal_store_screlease(signal, 1);
                const auto t0 = std::chrono::steady_clock::now();
                (void)hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 1,
                                                1000000ull, HSA_WAIT_STATE_ACTIVE);
                const auto t1 = std::chrono::steady_clock::now();
                if (it < opt.warmup) continue;
                hsa_us.push_back(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::printf("  %-26s p50_us=%9.3f p95_us=%9.3f\n",
                        "hsa_signal_wait(satisfied)", percentile(hsa_us, 0.5),
                        percentile(hsa_us, 0.95));
            (void)hsa_signal_destroy(signal);
        }
    }

    (void)hipStreamDestroy(stream);
    return 0;
}
