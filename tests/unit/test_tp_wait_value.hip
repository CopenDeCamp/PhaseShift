#include <phaseshift/core/status.h>
#include <phaseshift/runtime/stream_bridge.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using ps::Status;
using ps::runtime::StreamSignal;
using ps::runtime::StreamSignalKind;
using ps::runtime::stream_signal_alloc;
using ps::runtime::stream_signal_free;
using ps::runtime::stream_signal_kind_name;
using ps::runtime::stream_wait_value32;
using ps::runtime::stream_wait_value_supported;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

__global__ void k_delay_store(unsigned int* dst, unsigned int value,
                              unsigned long long spins) {
    if (threadIdx.x == 0u) {
        for (unsigned long long i = 0; i < spins; ++i) {
            __builtin_amdgcn_s_sleep(4);
        }
        __threadfence_system();
        *dst = value;
        __threadfence_system();
    }
}

__global__ void k_delay_only(unsigned long long spins) {
    if (threadIdx.x == 0u) {
        for (unsigned long long i = 0; i < spins; ++i) {
            __builtin_amdgcn_s_sleep(4);
        }
    }
}

__global__ void k_store(unsigned int* p, unsigned int value) {
    if (threadIdx.x == 0u) {
        __threadfence_system();
        *p = value;
        __threadfence_system();
    }
}

static double host_us(std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static bool record_event(hipEvent_t e, hipStream_t s, const char* what) {
    if (hipEventRecord(e, s) != hipSuccess) {
        std::printf("FAIL event record %s\n", what);
        return false;
    }
    return true;
}

static StreamSignal make_signal(unsigned int* p) {
    StreamSignal s{};
    s.ptr = p;
    s.device_ptr = p;
    s.bytes = sizeof(unsigned int);
    s.kind = StreamSignalKind::Device;
    return s;
}

static unsigned int* g_pinned_staging = nullptr;

static bool read_u32_async(int device, const unsigned int* p, hipStream_t stream,
                           unsigned int* out, const char* what) {
    if (hipSetDevice(device) != hipSuccess) {
        std::printf("FAIL read set device (%s)\n", what);
        return false;
    }
    if (g_pinned_staging == nullptr) {
        std::printf("FAIL read: no pinned staging\n");
        return false;
    }
    if (hipMemcpyAsync(g_pinned_staging, p, sizeof(unsigned int),
                       hipMemcpyDeviceToHost, stream) != hipSuccess) {
        std::printf("FAIL read async copy (%s): %s\n", what,
                    hipGetErrorString(hipGetLastError()));
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL read stream sync (%s)\n", what);
        return false;
    }
    *out = *g_pinned_staging;
    return true;
}

static bool run_ordering_case(const char* tag, unsigned int* flag_b,
                              unsigned int* marker_b, hipStream_t stream_a,
                              hipStream_t stream_b, hipEvent_t b_before,
                              hipEvent_t b_after, int spins, int warmup, int iters,
                              bool enqueue_wait_first, float* min_out,
                              float* max_out) {
    const StreamSignal sig = make_signal(flag_b);
    float min_us = 1e30f;
    float max_us = 0.0f;
    for (int i = 0; i < warmup + iters; ++i) {
        const unsigned int seq = static_cast<unsigned int>(1000 + i);

        if (hipSetDevice(1) != hipSuccess) {
            std::printf("FAIL %s: set device 1\n", tag);
            return false;
        }
        k_store<<<1, 64, 0, stream_b>>>(flag_b, 0u);
        k_store<<<1, 64, 0, stream_b>>>(marker_b, 0u);
        if (hipStreamSynchronize(stream_b) != hipSuccess) {
            std::printf("FAIL %s: reset sync\n", tag);
            return false;
        }

        if (!record_event(b_before, stream_b, tag)) return false;

        if (enqueue_wait_first) {
            const Status st = stream_wait_value32(stream_b, sig, seq);
            if (!st.ok()) {
                std::printf("FAIL %s: wait enqueue: %s\n", tag, st.message().c_str());
                return false;
            }
            k_store<<<1, 64, 0, stream_b>>>(marker_b, seq);
            if (!record_event(b_after, stream_b, tag)) return false;
            if (hipSetDevice(0) != hipSuccess) return false;
            k_delay_store<<<1, 64, 0, stream_a>>>(
                flag_b, seq, static_cast<unsigned long long>(spins));
        } else {
            if (hipSetDevice(0) != hipSuccess) return false;
            k_delay_store<<<1, 64, 0, stream_a>>>(
                flag_b, seq, static_cast<unsigned long long>(spins));
            if (hipSetDevice(1) != hipSuccess) return false;
            const Status st = stream_wait_value32(stream_b, sig, seq);
            if (!st.ok()) {
                std::printf("FAIL %s: wait enqueue: %s\n", tag, st.message().c_str());
                return false;
            }
            k_store<<<1, 64, 0, stream_b>>>(marker_b, seq);
            if (!record_event(b_after, stream_b, tag)) return false;
        }

        if (hipSetDevice(0) != hipSuccess) return false;
        if (hipStreamSynchronize(stream_a) != hipSuccess) {
            std::printf("FAIL %s: writer sync\n", tag);
            return false;
        }
        if (hipSetDevice(1) != hipSuccess) return false;
        if (hipStreamSynchronize(stream_b) != hipSuccess) {
            std::printf("FAIL %s: waiter sync\n", tag);
            return false;
        }

        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, b_before, b_after) != hipSuccess) {
            std::printf("FAIL %s: elapsed\n", tag);
            return false;
        }
        unsigned int marker = 0;
        unsigned int flag_read = 0;
        if (!read_u32_async(1, marker_b, stream_b, &marker, tag)) return false;
        if (!read_u32_async(1, flag_b, stream_b, &flag_read, tag)) return false;
        if (marker != seq || flag_read != seq) {
            std::printf("FAIL %s: value mismatch marker=%u flag=%u seq=%u\n", tag,
                        marker, flag_read, seq);
            return false;
        }
        if (i >= warmup) {
            if (elapsed < min_us) min_us = elapsed;
            if (elapsed > max_us) max_us = elapsed;
        }
    }
    *min_out = min_us;
    *max_out = max_us;
    return true;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: requires 2 GPUs\n");
        return 77;
    }
    for (int dev : {0, 1}) {
        auto supported = stream_wait_value_supported(dev);
        if (!supported.ok() || supported.value() == 0) {
            std::printf("SKIP: hipStreamWaitValue32 unsupported on device %d\n", dev);
            return 77;
        }
        std::printf("device %d: CanUseStreamWaitValue=%d\n", dev, supported.value());
    }

    if (hipSetDevice(0) != hipSuccess) return 77;
    (void)hipDeviceEnablePeerAccess(1, 0);
    if (hipSetDevice(1) != hipSuccess) return 77;
    (void)hipDeviceEnablePeerAccess(0, 1);

    hipStream_t stream_a = nullptr;
    hipStream_t stream_b = nullptr;
    if (hipSetDevice(0) != hipSuccess) return 77;
    if (hipStreamCreate(&stream_a) != hipSuccess) return 77;
    if (hipSetDevice(1) != hipSuccess) return 77;
    if (hipStreamCreate(&stream_b) != hipSuccess) return 77;

    if (hipSetDevice(1) != hipSuccess) return 77;
    unsigned int* dev_flag = nullptr;
    if (hipMalloc(&dev_flag, sizeof(unsigned int) * 4u) != hipSuccess) {
        std::printf("FAIL device flag alloc\n");
        return 1;
    }
    if (hipMemset(dev_flag, 0, sizeof(unsigned int) * 4u) != hipSuccess) {
        std::printf("FAIL device flag memset\n");
        return 1;
    }
    unsigned int* dev_marker = dev_flag + 2;
    std::printf("flag: Device (hipMalloc) at %p on device 1\n",
                static_cast<void*>(dev_flag));

    if (hipHostMalloc(&g_pinned_staging, sizeof(unsigned int),
                      hipHostMallocDefault) != hipSuccess) {
        std::printf("FAIL pinned staging alloc\n");
        return 1;
    }
    *g_pinned_staging = 0u;

    hipEvent_t b_before = nullptr;
    hipEvent_t b_after = nullptr;
    if (hipEventCreate(&b_before) != hipSuccess) return 77;
    if (hipEventCreate(&b_after) != hipSuccess) return 77;

    float min_us = 0.0f;
    float max_us = 0.0f;

    std::printf("=== [1] wait-value blocks on a HOST-written signal (baseline) ===\n");
    {
        auto sig = stream_signal_alloc(sizeof(unsigned int) * 4u);
        if (!sig.ok()) {
            std::printf("[1] alloc failed: %s\n", sig.status().message().c_str());
            return 1;
        }
        StreamSignal s = sig.release();
        std::printf("[1] kind=%s\n", stream_signal_kind_name(s.kind));
        const unsigned int want = 31337u;
        if (hipSetDevice(1) != hipSuccess) return 77;
        k_store<<<1, 64, 0, stream_b>>>(s.device_ptr, 0u);
        if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
        if (!record_event(b_before, stream_b, "local-before")) return 1;
        const Status st1 = stream_wait_value32(stream_b, s, want);
        if (!st1.ok()) {
            std::printf("[1] wait enqueue: %s\n", st1.message().c_str());
            return 1;
        }
        k_store<<<1, 64, 0, stream_b>>>(s.device_ptr + 2, 1u);
        if (!record_event(b_after, stream_b, "local-after")) return 1;

        constexpr int kDelayUs = 3000;
        const auto h0 = std::chrono::steady_clock::now();
        while (host_us(h0, std::chrono::steady_clock::now()) < kDelayUs) {
        }
        *s.ptr = want;
        if (hipStreamSynchronize(stream_b) != hipSuccess) {
            std::printf("[1] waiter sync\n");
            return 1;
        }
        float e = 0.0f;
        if (hipEventElapsedTime(&e, b_before, b_after) != hipSuccess) return 1;
        std::printf("[1] waiter elapsed: %.0f us (host delayed %d us)\n", e,
                    kDelayUs);
        std::printf("FINDING [1] %s (elapsed %.0f us vs host delay 3000 us)\n",
                e > 1000.0f ? "wait_value BLOCKED" : "wait_value did NOT block", e);
        stream_signal_free(s);
    }

    std::printf("=== [2] host enqueue cost ===\n");
    constexpr int kCostIters = 2000;
    if (hipSetDevice(1) != hipSuccess) return 77;
    k_store<<<1, 64, 0, stream_b>>>(dev_flag, 777u);
    if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
    {
        const StreamSignal sig = make_signal(dev_flag);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kCostIters; ++i) {
            const Status st = stream_wait_value32(stream_b, sig, 777u);
            if (!st.ok()) {
                std::printf("FAIL wait_value enqueue loop: %s\n",
                            st.message().c_str());
                return 1;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("stream_wait_value32: %.3f us/call\n",
                    host_us(t0, t1) / kCostIters);
        if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
    }
    {
        hipEvent_t done = nullptr;
        if (hipEventCreate(&done) != hipSuccess) return 77;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kCostIters; ++i) {
            if (hipStreamWaitEvent(stream_b, done, 0) != hipSuccess) {
                std::printf("FAIL wait_event enqueue loop\n");
                return 1;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("hipStreamWaitEvent:   %.3f us/call\n",
                    host_us(t0, t1) / kCostIters);
        if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
        (void)hipEventDestroy(done);
    }

    std::printf("=== [3] peer write on DEVICE memory (informational) ===\n");
    if (run_ordering_case("device-peer", dev_flag, dev_marker, stream_a, stream_b,
                          b_before, b_after, 40000, 1, 3, false, &min_us, &max_us)) {
        std::printf("[3] waiter elapsed: min=%.0f us max=%.0f us%s\n", min_us, max_us,
                    min_us < 100.0f
                        ? "  (wait did NOT block: wait-value ignores plain hipMalloc)"
                        : "");
    } else {
        std::printf("[3] device-peer case failed (see messages above)\n");
    }

    std::printf("=== [4] peer hipStreamWriteValue32 -> host-coherent signal ===\n");
    {
        auto sig = stream_signal_alloc(sizeof(unsigned int) * 4u);
        if (!sig.ok()) {
            std::printf("[4] alloc failed: %s\n", sig.status().message().c_str());
            return 1;
        }
        StreamSignal s = sig.release();
        const unsigned int want = 77777u;
        bool blocked_ok = true;
        float p4_min = 1e30f;
        for (int i = 0; i < 5; ++i) {
            if (hipSetDevice(1) != hipSuccess) return 77;
            if (hipStreamWriteValue32(stream_b, static_cast<void*>(s.device_ptr), 0u, 0u) !=
                hipSuccess) {
                std::printf("[4] reset write failed\n");
                blocked_ok = false;
                break;
            }
            if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
            if (!record_event(b_before, stream_b, "p4-before")) {
                blocked_ok = false;
                break;
            }
            if (hipSetDevice(1) != hipSuccess) return 77;
            const Status st = stream_wait_value32(stream_b, s, want);
            if (!st.ok()) {
                std::printf("[4] wait enqueue: %s\n", st.message().c_str());
                blocked_ok = false;
                break;
            }
            if (hipSetDevice(0) != hipSuccess) return 77;
            if (hipStreamWriteValue32(stream_a, static_cast<void*>(s.device_ptr), want,
                                 0u) != hipSuccess) {
                std::printf("[4] peer stream write failed\n");
                blocked_ok = false;
                break;
            }
            if (hipSetDevice(1) != hipSuccess) return 77;
            k_store<<<1, 64, 0, stream_b>>>(s.device_ptr + 2, 1u);
            if (!record_event(b_after, stream_b, "p4-after")) {
                blocked_ok = false;
                break;
            }
            if (hipStreamSynchronize(stream_b) != hipSuccess) {
                std::printf("[4] waiter sync\n");
                blocked_ok = false;
                break;
            }
            float e = 0.0f;
            if (hipEventElapsedTime(&e, b_before, b_after) != hipSuccess) return 1;
            if (e < p4_min) p4_min = e;
        }
        if (blocked_ok) {
            std::printf("[4] waiter elapsed: min=%.0f us (peer stream write from device 0)\n",
                        p4_min);
            std::printf("FINDING [4] %s\n",
                        p4_min > 100.0f
                            ? "peer hipStreamWriteValue32 observed"
                            : "peer hipStreamWriteValue32 NOT observed");
        }
        stream_signal_free(s);
    }

    std::printf("=== [5] peer kernel write -> host-coherent signal (risky, last) ===\n");
    {
        auto sig = stream_signal_alloc(sizeof(unsigned int) * 4u);
        if (!sig.ok()) {
            std::printf("[5] alloc failed: %s\n", sig.status().message().c_str());
            return 1;
        }
        StreamSignal s = sig.release();
        const unsigned int want = 55555u;
        if (hipSetDevice(1) != hipSuccess) return 77;
        k_store<<<1, 64, 0, stream_b>>>(s.device_ptr, 0u);
        if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
        if (!record_event(b_before, stream_b, "p5-before")) return 1;
        const Status st5 = stream_wait_value32(stream_b, s, want);
        if (!st5.ok()) {
            std::printf("[5] wait enqueue: %s\n", st5.message().c_str());
            return 1;
        }
        k_store<<<1, 64, 0, stream_b>>>(s.device_ptr + 2, 1u);
        if (!record_event(b_after, stream_b, "p5-after")) return 1;
        if (hipSetDevice(0) != hipSuccess) return 77;
        k_delay_store<<<1, 64, 0, stream_a>>>(s.device_ptr, want, 40000ull);
        if (hipSetDevice(1) != hipSuccess) return 77;
        if (hipStreamSynchronize(stream_b) != hipSuccess) {
            std::printf("[5] waiter sync\n");
            return 1;
        }
        if (hipSetDevice(0) != hipSuccess) return 77;
        if (hipStreamSynchronize(stream_a) != hipSuccess) return 1;
        float e = 0.0f;
        if (hipEventElapsedTime(&e, b_before, b_after) != hipSuccess) return 1;
        std::printf("[5] waiter elapsed: %.0f us (peer kernel write)\n", e);
        std::printf("FINDING [5] %s\n",
                    e > 1000.0f ? "peer kernel store observed"
                                : "peer kernel store NOT observed");
        stream_signal_free(s);
    }

    std::printf("=== [6] ExtSignal + LOCAL writer (2 streams on device 1) ===\n");
    {
        if (hipSetDevice(1) != hipSuccess) return 77;
        void* ext = nullptr;
        const hipError_t e = hipExtMallocWithFlags(&ext, 8, hipMallocSignalMemory);
        std::printf("[6] hipExtMallocWithFlags(hipMallocSignalMemory): %s (%d)\n",
                    hipGetErrorString(e), static_cast<int>(e));
        if (e != hipSuccess) {
            std::printf("[6] ExtSignal unavailable on this setup\n");
        } else {
            StreamSignal sig{};
            sig.ptr = static_cast<unsigned int*>(ext);
            sig.device_ptr = static_cast<unsigned int*>(ext);
            sig.bytes = 8;
            sig.kind = ps::runtime::StreamSignalKind::ExtSignal;
            const unsigned int want = 99111u;

            hipStream_t stream_writer = nullptr;
            if (hipStreamCreate(&stream_writer) != hipSuccess) return 77;

            if (hipStreamWriteValue32(stream_b, ext, 0u, 0u) != hipSuccess) {
                std::printf("[6] reset failed\n");
                return 1;
            }
            if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
            if (!record_event(b_before, stream_b, "p6-before")) return 1;
            const Status st6 = stream_wait_value32(stream_b, sig, want);
            if (!st6.ok()) {
                std::printf("[6] wait enqueue: %s\n", st6.message().c_str());
                return 1;
            }
            k_store<<<1, 64, 0, stream_b>>>(dev_marker, 1u);
            if (!record_event(b_after, stream_b, "p6-after")) return 1;

            k_delay_only<<<1, 64, 0, stream_writer>>>(40000ull);
            if (hipStreamWriteValue32(stream_writer, ext, want, 0u) != hipSuccess) {
                std::printf("[6] local stream write failed\n");
                return 1;
            }

            const auto w0 = std::chrono::steady_clock::now();
            if (hipStreamSynchronize(stream_writer) != hipSuccess) return 1;
            const double writer_ms = host_us(w0, std::chrono::steady_clock::now()) / 1000.0;
            const auto b0 = std::chrono::steady_clock::now();
            if (hipStreamSynchronize(stream_b) != hipSuccess) {
                std::printf("[6] waiter sync\n");
                return 1;
            }
            const double waiter_ms = host_us(b0, std::chrono::steady_clock::now()) / 1000.0;
            float e6 = 0.0f;
            if (hipEventElapsedTime(&e6, b_before, b_after) != hipSuccess) return 1;
            std::printf("[6] writer sync=%.2f ms  waiter sync=%.2f ms  "
                        "GPU elapsed=%.0f us\n", writer_ms, waiter_ms, e6);
            std::printf("FINDING [6] %s\n",
                        e6 > 1000.0f ? "wait_value blocks on ExtSignal"
                                     : "wait_value does NOT block on ExtSignal");
            (void)hipFree(ext);
            (void)hipStreamDestroy(stream_writer);
        }
    }

    std::printf("=== [7] ExtSignal + PEER writer (device 0 stream) ===\n");
    {
        if (hipSetDevice(1) != hipSuccess) return 77;
        void* ext = nullptr;
        const hipError_t e = hipExtMallocWithFlags(&ext, 8, hipMallocSignalMemory);
        if (e != hipSuccess) {
            std::printf("[7] ExtSignal unavailable\n");
        } else {
            StreamSignal sig{};
            sig.ptr = static_cast<unsigned int*>(ext);
            sig.device_ptr = static_cast<unsigned int*>(ext);
            sig.bytes = 8;
            sig.kind = ps::runtime::StreamSignalKind::ExtSignal;
            const unsigned int want = 77777u;

            if (hipStreamWriteValue32(stream_b, ext, 0u, 0u) != hipSuccess) {
                std::printf("[7] reset failed\n");
                return 1;
            }
            if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
            if (!record_event(b_before, stream_b, "p7-before")) return 1;
            const Status st7 = stream_wait_value32(stream_b, sig, want);
            if (!st7.ok()) {
                std::printf("[7] wait enqueue: %s\n", st7.message().c_str());
                return 1;
            }
            k_store<<<1, 64, 0, stream_b>>>(dev_marker, 2u);
            if (!record_event(b_after, stream_b, "p7-after")) return 1;

            if (hipSetDevice(0) != hipSuccess) return 77;
            k_delay_only<<<1, 64, 0, stream_a>>>(40000ull);
            if (hipStreamWriteValue32(stream_a, ext, want, 0u) != hipSuccess) {
                std::printf("[7] peer stream write failed\n");
                return 1;
            }
            if (hipStreamSynchronize(stream_a) != hipSuccess) return 1;
            if (hipSetDevice(1) != hipSuccess) return 77;
            if (hipStreamSynchronize(stream_b) != hipSuccess) {
                std::printf("[7] waiter sync\n");
                return 1;
            }
            float e7 = 0.0f;
            if (hipEventElapsedTime(&e7, b_before, b_after) != hipSuccess) return 1;
            std::printf("[7] waiter elapsed: %.0f us (ExtSignal, peer writer)\n", e7);
            std::printf("FINDING [7] %s\n",
                        e7 > 1000.0f ? "peer stream write observed"
                                     : "peer stream write NOT observed");
            (void)hipFree(ext);
        }
    }

    std::printf("=== [8] hipStreamBatchMemOp (same config as [6]) ===\n");
    {
        if (hipSetDevice(1) != hipSuccess) return 77;
        void* ext = nullptr;
        const hipError_t e = hipExtMallocWithFlags(&ext, 8, hipMallocSignalMemory);
        if (e != hipSuccess) {
            std::printf("[8] ExtSignal unavailable\n");
        } else {
            const unsigned int want = 42424u;
            hipStream_t writer8 = nullptr;
            if (hipStreamCreate(&writer8) != hipSuccess) return 77;

            if (hipStreamWriteValue32(stream_b, ext, 0u, 0u) != hipSuccess) {
                std::printf("[8] reset failed\n");
                return 1;
            }
            if (hipStreamSynchronize(stream_b) != hipSuccess) return 1;
            if (!record_event(b_before, stream_b, "p8-before")) return 1;

            hipStreamBatchMemOpParams op{};
            op.operation = hipStreamMemOpWaitValue32;
            op.waitValue.address = ext;
            op.waitValue.value = want;
            op.waitValue.flags = hipStreamWaitValueEq;
            op.waitValue.alias = nullptr;
            const hipError_t be =
                hipStreamBatchMemOp(stream_b, 1, &op, 0u);
            std::printf("[8] hipStreamBatchMemOp enqueue: %s (%d)\n",
                        hipGetErrorString(be), static_cast<int>(be));
            if (be != hipSuccess) {
                (void)hipStreamDestroy(writer8);
                (void)hipFree(ext);
            } else {
                k_store<<<1, 64, 0, stream_b>>>(dev_marker, 3u);
                if (!record_event(b_after, stream_b, "p8-after")) return 1;

                k_delay_only<<<1, 64, 0, writer8>>>(40000ull);
                if (hipStreamWriteValue32(writer8, ext, want, 0u) != hipSuccess) {
                    std::printf("[8] writer stream write failed\n");
                    return 1;
                }
                const auto w0 = std::chrono::steady_clock::now();
                if (hipStreamSynchronize(writer8) != hipSuccess) return 1;
                const double writer_ms =
                    host_us(w0, std::chrono::steady_clock::now()) / 1000.0;
                const auto b0 = std::chrono::steady_clock::now();
                if (hipStreamSynchronize(stream_b) != hipSuccess) {
                    std::printf("[8] waiter sync\n");
                    return 1;
                }
                const double waiter_ms =
                    host_us(b0, std::chrono::steady_clock::now()) / 1000.0;
                float e8 = 0.0f;
                if (hipEventElapsedTime(&e8, b_before, b_after) != hipSuccess) return 1;
                std::printf("[8] writer sync=%.2f ms  waiter sync=%.2f ms  "
                            "GPU elapsed=%.0f us\n",
                            writer_ms, waiter_ms, e8);
                std::printf("FINDING [8] %s\n",
                            e8 > 1000.0f
                                ? "batch mem op blocks until the delayed store"
                                : "batch mem op does NOT block");
                (void)hipStreamDestroy(writer8);
                (void)hipFree(ext);
            }
        }
    }

    if (hipSetDevice(1) != hipSuccess) return 77;
    (void)hipHostFree(g_pinned_staging);
    g_pinned_staging = nullptr;
    (void)hipEventDestroy(b_before);
    (void)hipEventDestroy(b_after);
    (void)hipFree(dev_flag);
    if (hipSetDevice(0) != hipSuccess) return 77;
    (void)hipStreamDestroy(stream_a);
    if (hipSetDevice(1) != hipSuccess) return 77;
    (void)hipStreamDestroy(stream_b);

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
