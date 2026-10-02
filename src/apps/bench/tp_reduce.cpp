#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ps::Status;
using ps::runtime::HostMediatedTpTransport;
using ps::runtime::TpSumInvocation;
using ps::runtime::TpSumTarget;
using ps::runtime::TpTransport;
using ps::runtime::ValueDType;
using ps::runtime::create_tp_transport;

namespace {

void usage() {
    std::printf("usage: phaseshift-bench tp-reduce [--rows 1,2,...] [--features N] [--iters N]\n");
    std::printf("  measures tp sum_hidden latency for the implemented transports\n");
    std::printf("  default rows: 1,2,4,8,16,32,64,128,256,512,1024,2048\n");
    std::printf("  default features: 5120 (Qwen3.8 hidden), dtype bf16\n");
    std::printf("  bandwidth definition: 2 * payload_bytes / elapsed (one read + one write)\n");
}

std::vector<uint64_t> parse_rows(const char* text) {
    std::vector<uint64_t> rows;
    const char* p = text;
    while (*p != '\0') {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(p, &end, 10);
        if (end == p) break;
        if (value > 0) rows.push_back(value);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return rows;
}

struct BenchPair {
    void* buffer_a = nullptr;
    void* buffer_b = nullptr;
    hipStream_t stream_a = nullptr;
    hipStream_t stream_b = nullptr;
    hipEvent_t ready_a = nullptr;
    hipEvent_t ready_b = nullptr;
};

bool allocate_pair(BenchPair& pair, std::size_t bytes) {
    for (int device = 0; device < 2; ++device) {
        auto scope = ps::gpu::ScopedDevice::create(device);
        if (!scope.ok()) return false;
        void** buffer = device == 0 ? &pair.buffer_a : &pair.buffer_b;
        hipStream_t* stream = device == 0 ? &pair.stream_a : &pair.stream_b;
        hipEvent_t* ready = device == 0 ? &pair.ready_a : &pair.ready_b;
        if (hipStreamCreate(stream) != hipSuccess) return false;
        if (hipEventCreate(ready) != hipSuccess) return false;
        if (hipMalloc(buffer, bytes) != hipSuccess) return false;
    }
    return true;
}

void free_pair(BenchPair& pair) {
    for (int device = 0; device < 2; ++device) {
        auto scope = ps::gpu::ScopedDevice::create(device);
        if (!scope.ok()) continue;
        void** buffer = device == 0 ? &pair.buffer_a : &pair.buffer_b;
        hipStream_t* stream = device == 0 ? &pair.stream_a : &pair.stream_b;
        hipEvent_t* ready = device == 0 ? &pair.ready_a : &pair.ready_b;
        if (*buffer != nullptr) ps::gpu::discard_cleanup_result(hipFree(*buffer));
        if (*ready != nullptr) ps::gpu::discard_cleanup_result(hipEventDestroy(*ready));
        if (*stream != nullptr) ps::gpu::discard_cleanup_result(hipStreamDestroy(*stream));
    }
}

Status upload(int device, void* buffer, const std::vector<std::uint8_t>& host) {
    auto scope = ps::gpu::ScopedDevice::create(device);
    if (!scope.ok()) return scope.status();
    if (hipMemcpy(buffer, host.data(), host.size(), hipMemcpyHostToDevice) != hipSuccess) {
        return Status::hip_error("tp-reduce upload", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

void run_backend(TpTransport* transport, const char* backend, BenchPair& pair,
                 const std::vector<std::uint8_t>& host_a,
                 const std::vector<std::uint8_t>& host_b, uint64_t rows,
                 uint32_t features, int iters) {
    const std::size_t bytes = static_cast<std::size_t>(rows) * features * 2u;
    TpSumTarget target_a;
    target_a.ptr = pair.buffer_a;
    target_a.rows = rows;
    target_a.row_stride = features;
    target_a.feature_count = features;
    target_a.dtype = ValueDType::BF16;
    TpSumTarget target_b = target_a;
    target_b.ptr = pair.buffer_b;
    std::vector<TpSumTarget> targets{target_a, target_b};
    std::vector<hipStream_t> streams{pair.stream_a, pair.stream_b};
    std::vector<hipEvent_t> ready{pair.ready_a, pair.ready_b};
    TpSumInvocation invocation{targets, streams, ready};

    const auto run_once = [&]() -> Status {
        Status st = upload(0, pair.buffer_a, host_a);
        if (!st.ok()) return st;
        st = upload(1, pair.buffer_b, host_b);
        if (!st.ok()) return st;
        if (hipEventRecord(pair.ready_a, pair.stream_a) != hipSuccess ||
            hipEventRecord(pair.ready_b, pair.stream_b) != hipSuccess) {
            return Status::hip_error("tp-reduce record", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        st = transport->sum_hidden(invocation);
        if (!st.ok()) return st;
        if (hipStreamSynchronize(pair.stream_a) != hipSuccess ||
            hipStreamSynchronize(pair.stream_b) != hipSuccess) {
            return Status::hip_error("tp-reduce sync", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        return Status::make_ok();
    };

    Status warmup = run_once();
    if (!warmup.ok()) {
        std::printf("  %s rows=%llu FAILED: %s\n", backend,
                    static_cast<unsigned long long>(rows), warmup.message().c_str());
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        Status st = run_once();
        if (!st.ok()) {
            std::printf("  %s rows=%llu FAILED: %s\n", backend,
                        static_cast<unsigned long long>(rows), st.message().c_str());
            return;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double total_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    const double us_per_iter = total_us / iters;
    const double payload = static_cast<double>(bytes);
    const double gbps = (2.0 * payload * iters) / total_us / 1e3;
    std::printf("  %-14s rows=%-5llu payload=%-9llu us=%.1f GB/s=%.2f\n", backend,
                static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(bytes), us_per_iter, gbps);
}

}

int run_tp_reduce(int argc, char** argv) {
    std::vector<uint64_t> rows{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    uint32_t features = 5120;
    int iters = 50;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else if (arg == "--rows" && i + 1 < argc) {
            rows = parse_rows(argv[++i]);
        } else if (arg == "--features" && i + 1 < argc) {
            features = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            usage();
            return 1;
        }
    }
    if (rows.empty() || features == 0 || iters <= 0) {
        std::fprintf(stderr, "invalid rows/features/iters\n");
        return 1;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::fprintf(stderr, "tp-reduce requires 2 GPUs (found %d)\n", device_count);
        return 1;
    }

    std::uint64_t max_rows = 0;
    for (uint64_t r : rows) max_rows = r > max_rows ? r : max_rows;
    const std::size_t max_bytes =
        static_cast<std::size_t>(max_rows) * features * sizeof(std::uint16_t);

    BenchPair pair;
    if (!allocate_pair(pair, max_bytes)) {
        std::fprintf(stderr, "failed to allocate benchmark buffers\n");
        return 1;
    }

    std::vector<std::uint8_t> host_a(max_bytes);
    std::vector<std::uint8_t> host_b(max_bytes);
    for (std::size_t i = 0; i < max_bytes; ++i) {
        host_a[i] = static_cast<std::uint8_t>(i * 31u + 7u);
        host_b[i] = static_cast<std::uint8_t>(i * 17u + 3u);
    }

    const std::vector<int> devices{0, 1};
    auto selection = create_tp_transport(devices);
    if (!selection.ok()) {
        std::fprintf(stderr, "failed to create tp transport: %s\n",
                     selection.status().message().c_str());
        free_pair(pair);
        return 1;
    }
    TpTransport* selected = selection.value().transport.get();
    const Status verify = selected->verify();
    const bool peer_usable = verify.ok();
    if (!peer_usable) {
        std::printf("[tp] peer transport self-test failed (%s); measuring the "
                    "host-mediated reference transport\n",
                    verify.message().c_str());
    }

    std::printf("tp-reduce devices=0,1 features=%u dtype=bf16 iters=%d\n", features,
                iters);
    std::printf("bandwidth definition: 2 * payload_bytes / elapsed\n");

    if (peer_usable) {
        std::printf("backend: %s\n", selected->name());
        for (uint64_t r : rows) {
            run_backend(selected, selected->name(), pair, host_a, host_b, r, features,
                        iters);
        }
    }

    HostMediatedTpTransport host_transport(devices);
    if (!peer_usable || std::string(host_transport.name()) != selected->name()) {
        std::printf("backend: %s\n", host_transport.name());
        for (uint64_t r : rows) {
            run_backend(&host_transport, host_transport.name(), pair, host_a, host_b, r,
                        features, iters);
        }
    }

    free_pair(pair);
    return 0;
}
