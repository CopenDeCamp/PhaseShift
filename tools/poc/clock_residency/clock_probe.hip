#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string mode = "timing";
    std::string probe = "compute";
    std::string size = "short";
    int device = 1;
    int reps = 100;
    int prime = 8;
    int warmup = 4;
    std::string idle_list = "0,10,25,50,100,250,500,1000,2000,5000,10000,25000,50000,100000";
    unsigned int compute_iters_short = 8192;
    unsigned int compute_iters_medium = 98304;
    unsigned int mem_mb_short = 128;
    unsigned int mem_mb_medium = 512;
    int blocks = 256;
    int threads = 256;
    int mem_blocks = 1024;
    int mem_threads = 256;
    unsigned int ramp_iters = 0;
    std::string output;
    std::string sclk_path;
    std::string mclk_path;
    std::string power_path;
    std::string temp_path;
    int observe_samples = 40;
    int observe_interval_us = 250;
    int keepalive_us = 0;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: clock_probe --mode timing|observe [options]\n"
                 "  --probe compute|mem          (default compute)\n"
                 "  --size short|medium          (default short)\n"
                 "  --device N                   (default 1)\n"
                 "  --idle-us A,B,C,...          requested idle durations in us\n"
                 "  --reps N                     trials per idle value (default 100)\n"
                 "  --prime N                    probe launches before each trial (default 8)\n"
                 "  --warmup N                   discarded trials before each idle value (default 4)\n"
                 "  --compute-iters N            compute probe FMA iterations (short)\n"
                 "  --compute-iters-medium N     compute probe FMA iterations (medium)\n"
                 "  --mem-mb N                   memory probe size MB (short)\n"
                 "  --mem-mb-medium N            memory probe size MB (medium)\n"
                 "  --blocks N --threads N       compute probe geometry\n"
                 "  --mem-blocks N --mem-threads N  memory probe geometry\n"
                 "  --ramp-iters N               observe-mode ramp compute iterations\n"
                 "  --observe-samples N --observe-interval-us N\n"
                 "  --keepalive-us N             event-record CP keepalive interval (0=off)\n"
                 "  --sclk-path P --mclk-path P --power-path P --temp-path P\n"
                 "  --output PATH\n");
}

bool parse_u32(const char* s, unsigned int* out) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0') return false;
    *out = static_cast<unsigned int>(v);
    return true;
}

bool parse_i32(const char* s, int* out) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    *out = static_cast<int>(v);
    return true;
}

std::vector<double> parse_idle_list(const std::string& text) {
    std::vector<double> values;
    const char* p = text.c_str();
    while (*p != '\0') {
        char* end = nullptr;
        const double v = std::strtod(p, &end);
        if (end == p) break;
        values.push_back(v);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return values;
}

bool read_u64(const std::string& path, unsigned long long* value) {
    if (path.empty()) return false;
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return false;
    unsigned long long v = 0;
    const int n = std::fscanf(f, "%llu", &v);
    std::fclose(f);
    if (n != 1) return false;
    *value = v;
    return true;
}

void spin_wait_us(double us) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto target = std::chrono::duration<double, std::micro>(us);
    while (std::chrono::steady_clock::now() - t0 < target) {
        __builtin_ia32_pause();
    }
}

__global__ void probe_compute_kernel(float* out, unsigned int iters,
                                     unsigned long long* ts) {
    const unsigned long long t0 = wall_clock64();
    float a = static_cast<float>(threadIdx.x) * 0.000123f +
              static_cast<float>(blockIdx.x) * 0.000456f + 1.0f;
    const float b = 1.000001f;
    const float c = 0.999999f;
    for (unsigned int i = 0; i < iters; ++i) {
        a = fmaf(a, b, c);
        a = fmaf(a, c, b);
    }
    const unsigned long long t1 = wall_clock64();
    if (blockIdx.x == 0 && threadIdx.x == 0 && ts != nullptr) {
        ts[0] = t0;
        ts[1] = t1;
    }
    if (a == 123456.789f) out[0] = a;
}

__global__ void probe_fill_kernel(unsigned int* buf, unsigned long long n) {
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i =
             static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        buf[i] = static_cast<unsigned int>(i) * 2654435761u + 0x9e3779b9u;
    }
}

__global__ void probe_mem_kernel(const uint4* __restrict__ in,
                                 unsigned long long n4, float* out,
                                 unsigned long long* ts) {
    const unsigned long long t0 = wall_clock64();
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned long long i =
             static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n4; i += stride) {
        const uint4 v = in[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    const unsigned long long t1 = wall_clock64();
    if (blockIdx.x == 0 && threadIdx.x == 0 && ts != nullptr) {
        ts[0] = t0;
        ts[1] = t1;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xdeadbeefu) out[0] = 1.0f;
}

struct EventPair {
    hipEvent_t ref = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
};

bool create_events(EventPair* ev) {
    return hipEventCreateWithFlags(&ev->ref, hipEventDisableSystemFence) == hipSuccess &&
           hipEventCreateWithFlags(&ev->start, hipEventDisableSystemFence) == hipSuccess &&
           hipEventCreateWithFlags(&ev->stop, hipEventDisableSystemFence) == hipSuccess;
}

void destroy_events(EventPair* ev) {
    if (ev->ref) hipEventDestroy(ev->ref);
    if (ev->start) hipEventDestroy(ev->start);
    if (ev->stop) hipEventDestroy(ev->stop);
}

struct ProbeRig {
    float* out = nullptr;
    unsigned int* fill = nullptr;
    uint4* mem = nullptr;
    unsigned long long* ts = nullptr;
    unsigned long long mem_words = 0;
    std::size_t mem_bytes = 0;
    Options opt;
    unsigned int compute_iters = 0;
    unsigned long long mem_n4 = 0;
};

void launch_probe(const ProbeRig& rig, hipStream_t stream) {
    if (rig.opt.probe == "compute") {
        probe_compute_kernel<<<rig.opt.blocks, rig.opt.threads, 0, stream>>>(
            rig.out, rig.compute_iters, rig.ts);
    } else {
        probe_mem_kernel<<<rig.opt.mem_blocks, rig.opt.mem_threads, 0, stream>>>(
            rig.mem, rig.mem_n4, rig.out, rig.ts);
    }
}

struct KeepAliveCtx {
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    std::atomic<bool> stop{false};
    std::thread thread;
    bool active = false;
};

void keepalive_start(KeepAliveCtx* ka, int interval_us) {
    if (interval_us <= 0) return;
    if (hipStreamCreate(&ka->stream) != hipSuccess) return;
    if (hipEventCreateWithFlags(&ka->event, hipEventDisableSystemFence) != hipSuccess) {
        hipStreamDestroy(ka->stream);
        ka->stream = nullptr;
        return;
    }
    ka->active = true;
    ka->thread = std::thread([ka, interval_us] {
        while (!ka->stop.load(std::memory_order_relaxed)) {
            (void)hipEventRecord(ka->event, ka->stream);
            std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
        }
    });
}

void keepalive_stop(KeepAliveCtx* ka) {
    if (!ka->active) return;
    ka->stop.store(true);
    if (ka->thread.joinable()) ka->thread.join();
    (void)hipStreamSynchronize(ka->stream);
    (void)hipEventDestroy(ka->event);
    (void)hipStreamDestroy(ka->stream);
    ka->active = false;
}

int run_timing(ProbeRig& rig) {
    const Options& opt = rig.opt;
    const std::vector<double> idles = parse_idle_list(opt.idle_list);
    if (idles.empty()) {
        std::fprintf(stderr, "empty idle list\n");
        return 2;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    EventPair ev;
    if (!create_events(&ev)) {
        std::fprintf(stderr, "event create failed\n");
        return 1;
    }

    FILE* csv = nullptr;
    if (!opt.output.empty()) {
        csv = std::fopen(opt.output.c_str(), "a");
        if (csv == nullptr) {
            std::fprintf(stderr, "cannot open %s\n", opt.output.c_str());
            return 1;
        }
        if (std::ftell(csv) == 0) {
            std::fprintf(csv,
                         "mode,probe,size,device,idle_us,rep,host_idle_us,gap_us,"
                         "probe_us,exec_us,keepalive_us\n");
        }
    }

    KeepAliveCtx ka;
    keepalive_start(&ka, opt.keepalive_us);

    std::fprintf(stdout,
                 "clock_probe timing probe=%s size=%s device=%d reps=%d "
                 "keepalive_us=%d\n",
                 opt.probe.c_str(), opt.size.c_str(), opt.device, opt.reps,
                 opt.keepalive_us);

    std::vector<std::vector<double>> gap_samples(idles.size());
    std::vector<std::vector<double>> probe_samples(idles.size());
    std::vector<std::vector<double>> exec_samples(idles.size());

    const int total_rounds = opt.warmup + opt.reps;
    for (int round = 0; round < total_rounds; ++round) {
        const bool measured = round >= opt.warmup;
        for (std::size_t k = 0; k < idles.size(); ++k) {
            for (int p = 0; p < opt.prime; ++p) launch_probe(rig, stream);
            if (hipEventRecord(ev.ref, stream) != hipSuccess) return 1;
            if (hipEventSynchronize(ev.ref) != hipSuccess) return 1;

            const auto spin_begin = std::chrono::steady_clock::now();
            spin_wait_us(idles[k]);
            const auto spin_end = std::chrono::steady_clock::now();
            const double host_idle_us =
                std::chrono::duration<double, std::micro>(spin_end - spin_begin).count();

            if (hipEventRecord(ev.start, stream) != hipSuccess) return 1;
            launch_probe(rig, stream);
            if (hipEventRecord(ev.stop, stream) != hipSuccess) return 1;
            if (hipEventSynchronize(ev.stop) != hipSuccess) return 1;

            float gap_ms = 0.0f;
            float probe_ms = 0.0f;
            if (hipEventElapsedTime(&gap_ms, ev.ref, ev.start) != hipSuccess) return 1;
            if (hipEventElapsedTime(&probe_ms, ev.start, ev.stop) != hipSuccess) return 1;
            const double gap_us = static_cast<double>(gap_ms) * 1000.0;
            const double probe_us = static_cast<double>(probe_ms) * 1000.0;

            double exec_us = 0.0;
            if (rig.ts != nullptr) {
                unsigned long long ts_host[2] = {0, 0};
                if (hipMemcpy(ts_host, rig.ts, sizeof(ts_host),
                              hipMemcpyDeviceToHost) == hipSuccess) {
                    exec_us = static_cast<double>(ts_host[1] - ts_host[0]) / 100.0;
                }
            }

            if (measured) {
                gap_samples[k].push_back(gap_us);
                probe_samples[k].push_back(probe_us);
                exec_samples[k].push_back(exec_us);
                if (csv != nullptr) {
                    std::fprintf(csv,
                                 "timing,%s,%s,%d,%.0f,%d,%.3f,%.3f,%.3f,%.3f,%d\n",
                                 opt.probe.c_str(), opt.size.c_str(), opt.device,
                                 idles[k], round - opt.warmup, host_idle_us, gap_us,
                                 probe_us, exec_us, opt.keepalive_us);
                }
            }
        }
    }

    for (std::size_t k = 0; k < idles.size(); ++k) {
        auto summarize = [](std::vector<double>& v, double* median, double* p95,
                            double* minv) {
            if (v.empty()) {
                *median = *p95 = *minv = 0.0;
                return;
            }
            std::sort(v.begin(), v.end());
            const std::size_t n = v.size();
            *median = v[n / 2];
            *minv = v.front();
            const std::size_t hi = static_cast<std::size_t>(0.95 * (n - 1) + 0.5);
            *p95 = v[hi < n ? hi : n - 1];
        };
        double gap_med = 0.0, gap_p95 = 0.0, gap_min = 0.0;
        double probe_med = 0.0, probe_p95 = 0.0, probe_min = 0.0;
        double exec_med = 0.0, exec_p95 = 0.0, exec_min = 0.0;
        summarize(gap_samples[k], &gap_med, &gap_p95, &gap_min);
        summarize(probe_samples[k], &probe_med, &probe_p95, &probe_min);
        summarize(exec_samples[k], &exec_med, &exec_p95, &exec_min);
        std::fprintf(stdout,
                     "idle_us=%.0f n=%zu gap_us median=%.3f p95=%.3f min=%.3f "
                     "probe_us median=%.3f p95=%.3f min=%.3f "
                     "exec_us median=%.3f p95=%.3f min=%.3f\n",
                     idles[k], gap_samples[k].size(), gap_med, gap_p95, gap_min,
                     probe_med, probe_p95, probe_min, exec_med, exec_p95, exec_min);
    }

    if (csv != nullptr) std::fclose(csv);
    keepalive_stop(&ka);
    destroy_events(&ev);
    (void)hipStreamDestroy(stream);
    return 0;
}

int run_observe(ProbeRig& rig) {
    const Options& opt = rig.opt;
    const std::vector<double> idles = parse_idle_list(opt.idle_list);
    if (idles.empty()) {
        std::fprintf(stderr, "empty idle list\n");
        return 2;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    FILE* csv = nullptr;
    if (!opt.output.empty()) {
        csv = std::fopen(opt.output.c_str(), "a");
        if (csv == nullptr) {
            std::fprintf(stderr, "cannot open %s\n", opt.output.c_str());
            return 1;
        }
        if (std::ftell(csv) == 0) {
            std::fprintf(csv,
                         "mode,probe,size,device,idle_us,rep,sample,dt_us,sclk_hz,"
                         "mclk_hz,power_uw,temp_mc\n");
        }
    }

    unsigned int ramp_iters = opt.ramp_iters;
    if (ramp_iters == 0) ramp_iters = opt.compute_iters_medium * 4u;

    KeepAliveCtx ka;
    keepalive_start(&ka, opt.keepalive_us);

    std::fprintf(stdout,
                 "clock_probe observe device=%d samples=%d interval_us=%d "
                 "keepalive_us=%d\n",
                 opt.device, opt.observe_samples, opt.observe_interval_us,
                 opt.keepalive_us);

    for (int rep = 0; rep < opt.reps; ++rep) {
        for (std::size_t k = 0; k < idles.size(); ++k) {
            for (int p = 0; p < opt.prime; ++p) launch_probe(rig, stream);
            if (hipStreamSynchronize(stream) != hipSuccess) return 1;

            spin_wait_us(idles[k]);

            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s <= opt.observe_samples; ++s) {
                unsigned long long sclk = 0, mclk = 0, power = 0, temp = 0;
                read_u64(opt.sclk_path, &sclk);
                read_u64(opt.mclk_path, &mclk);
                read_u64(opt.power_path, &power);
                read_u64(opt.temp_path, &temp);
                const double dt_us =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
                if (csv != nullptr) {
                    std::fprintf(csv,
                                 "observe,%s,%s,%d,%.0f,%d,%d,%.3f,%llu,%llu,%llu,%llu\n",
                                 opt.probe.c_str(), opt.size.c_str(), opt.device,
                                 idles[k], rep, s, dt_us, sclk, mclk, power, temp);
                }
                if (s == 0) {
                    if (opt.probe == "compute") {
                        probe_compute_kernel<<<opt.blocks, opt.threads, 0, stream>>>(
                            rig.out, ramp_iters, nullptr);
                    } else {
                        launch_probe(rig, stream);
                    }
                }
                spin_wait_us(opt.observe_interval_us);
            }
            if (hipStreamSynchronize(stream) != hipSuccess) return 1;
        }
    }

    if (csv != nullptr) std::fclose(csv);
    keepalive_stop(&ka);
    (void)hipStreamDestroy(stream);
    return 0;
}

std::string default_path(const std::string& dir, const char* name) {
    if (dir.empty()) return std::string();
    return dir + "/" + name;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::string telem_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        } else if (a == "--mode") {
            opt.mode = value();
        } else if (a == "--probe") {
            opt.probe = value();
        } else if (a == "--size") {
            opt.size = value();
        } else if (a == "--device") {
            parse_i32(value(), &opt.device);
        } else if (a == "--idle-us") {
            opt.idle_list = value();
        } else if (a == "--reps") {
            parse_i32(value(), &opt.reps);
        } else if (a == "--prime") {
            parse_i32(value(), &opt.prime);
        } else if (a == "--warmup") {
            parse_i32(value(), &opt.warmup);
        } else if (a == "--compute-iters") {
            parse_u32(value(), &opt.compute_iters_short);
        } else if (a == "--compute-iters-medium") {
            parse_u32(value(), &opt.compute_iters_medium);
        } else if (a == "--mem-mb") {
            parse_u32(value(), &opt.mem_mb_short);
        } else if (a == "--mem-mb-medium") {
            parse_u32(value(), &opt.mem_mb_medium);
        } else if (a == "--blocks") {
            parse_i32(value(), &opt.blocks);
        } else if (a == "--threads") {
            parse_i32(value(), &opt.threads);
        } else if (a == "--mem-blocks") {
            parse_i32(value(), &opt.mem_blocks);
        } else if (a == "--mem-threads") {
            parse_i32(value(), &opt.mem_threads);
        } else if (a == "--ramp-iters") {
            parse_u32(value(), &opt.ramp_iters);
        } else if (a == "--observe-samples") {
            parse_i32(value(), &opt.observe_samples);
        } else if (a == "--observe-interval-us") {
            parse_i32(value(), &opt.observe_interval_us);
        } else if (a == "--keepalive-us") {
            parse_i32(value(), &opt.keepalive_us);
        } else if (a == "--telem-dir") {
            telem_dir = value();
        } else if (a == "--sclk-path") {
            opt.sclk_path = value();
        } else if (a == "--mclk-path") {
            opt.mclk_path = value();
        } else if (a == "--power-path") {
            opt.power_path = value();
        } else if (a == "--temp-path") {
            opt.temp_path = value();
        } else if (a == "--output") {
            opt.output = value();
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(stderr);
            return 2;
        }
    }
    if (!telem_dir.empty()) {
        if (opt.sclk_path.empty()) opt.sclk_path = default_path(telem_dir, "freq1_input");
        if (opt.mclk_path.empty()) opt.mclk_path = default_path(telem_dir, "freq2_input");
        if (opt.power_path.empty())
            opt.power_path = default_path(telem_dir, "power1_average");
        if (opt.temp_path.empty()) opt.temp_path = default_path(telem_dir, "temp1_input");
    }

    if (hipSetDevice(opt.device) != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice(%d) failed\n", opt.device);
        return 1;
    }

    ProbeRig rig;
    rig.opt = opt;
    if (opt.probe == "compute") {
        rig.compute_iters = (opt.size == "medium") ? opt.compute_iters_medium
                                                   : opt.compute_iters_short;
    } else if (opt.probe == "mem") {
        const unsigned int mb = (opt.size == "medium") ? opt.mem_mb_medium
                                                       : opt.mem_mb_short;
        rig.mem_bytes = static_cast<std::size_t>(mb) << 20;
        rig.mem_words = rig.mem_bytes / sizeof(uint4);
        rig.mem_n4 = rig.mem_words;
        if (hipMalloc(reinterpret_cast<void**>(&rig.mem), rig.mem_bytes) != hipSuccess) {
            std::fprintf(stderr, "hipMalloc %u MB failed\n", mb);
            return 1;
        }
        const unsigned long long fill_words = rig.mem_bytes / sizeof(unsigned int);
        probe_fill_kernel<<<1024, 256>>>(reinterpret_cast<unsigned int*>(rig.mem),
                                         fill_words);
        if (hipDeviceSynchronize() != hipSuccess) return 1;
    } else {
        std::fprintf(stderr, "unknown probe: %s\n", opt.probe.c_str());
        return 2;
    }
    if (hipMalloc(reinterpret_cast<void**>(&rig.out), 64 * sizeof(float)) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc out failed\n");
        return 1;
    }
    if (hipMalloc(reinterpret_cast<void**>(&rig.ts),
                  2 * sizeof(unsigned long long)) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc ts failed\n");
        return 1;
    }

    int rc = 0;
    if (opt.mode == "timing") {
        rc = run_timing(rig);
    } else if (opt.mode == "observe") {
        rc = run_observe(rig);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", opt.mode.c_str());
        rc = 2;
    }

    if (rig.ts) hipFree(rig.ts);
    if (rig.out) hipFree(rig.out);
    if (rig.mem) hipFree(rig.mem);
    return rc;
}
