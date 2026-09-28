#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>

#ifdef PHASESHIFT_HAS_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace {

struct BenchOptions {
    int prompt_tokens = 128;
    int prefill_chunk = 0;
    int gap_keepalive = 0;
    int gap_keepalive_flags = hipEventDisableSystemFence;
    std::string mode = "forward";
    int device = 0;
    std::string model_dir;
    int page_tokens = 16;
    double arena_gib = 6.0;
    int runs = 3;
    int warmup = 0;
    std::string output;
};

void print_usage(FILE* out) {
    fprintf(out,
        "Usage: phaseshift-bench pp [options]\n"
        "  --prompt-tokens N            (default 128)\n"
        "  --prefill-chunk N            (default 0 = whole prompt in one batch)\n"
        "  --gap-keepalive US           (default 0 = off; keep the CP awake)\n"
        "  --gap-keepalive-flags N      (default hipEventDisableSystemFence;\n"
        "                                -1 = hipEventCreate)\n"
        "  --mode forward|greedy        (default forward)\n"
        "  --device N                   (default 0)\n"
        "  --model-dir PATH             (required)\n"
        "  --page-tokens N              (default 16)\n"
        "  --arena-gib N                (default 6)\n"
        "  --runs N                     (default 3)\n"
        "  --warmup N                   (default 0)\n"
        "  --output PATH                (default none)\n");
}

bool parse_int(const std::string& value, int& out) {
    size_t pos = 0;
    try {
        out = std::stoi(value, &pos);
    } catch (...) {
        return false;
    }
    return pos == value.size();
}

bool parse_double(const std::string& value, double& out) {
    size_t pos = 0;
    try {
        out = std::stod(value, &pos);
    } catch (...) {
        return false;
    }
    return pos == value.size();
}

bool parse_args(int argc, char** argv, BenchOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key = arg;
        std::string value;
        bool has_value = false;
        size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
            has_value = true;
        }
        auto need_value = [&]() -> std::string {
            if (has_value) return value;
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", key.c_str());
                exit(2);
            }
            ++i;
            return std::string(argv[i]);
        };
        if (key == "--prompt-tokens") {
            if (!parse_int(need_value(), opts.prompt_tokens) || opts.prompt_tokens < 1) {
                fprintf(stderr, "invalid --prompt-tokens\n");
                return false;
            }
        } else if (key == "--prefill-chunk") {
            if (!parse_int(need_value(), opts.prefill_chunk) || opts.prefill_chunk < 0) {
                fprintf(stderr, "invalid --prefill-chunk\n");
                return false;
            }
        } else if (key == "--gap-keepalive") {
            if (!parse_int(need_value(), opts.gap_keepalive) || opts.gap_keepalive < 0) {
                fprintf(stderr, "invalid --gap-keepalive\n");
                return false;
            }
        } else if (key == "--gap-keepalive-flags") {
            if (!parse_int(need_value(), opts.gap_keepalive_flags)) {
                fprintf(stderr, "invalid --gap-keepalive-flags\n");
                return false;
            }
        } else if (key == "--mode") {
            opts.mode = need_value();
        } else if (key == "--device") {
            if (!parse_int(need_value(), opts.device)) {
                fprintf(stderr, "invalid --device\n");
                return false;
            }
        } else if (key == "--model-dir") {
            opts.model_dir = need_value();
        } else if (key == "--page-tokens") {
            if (!parse_int(need_value(), opts.page_tokens) || opts.page_tokens < 1) {
                fprintf(stderr, "invalid --page-tokens\n");
                return false;
            }
        } else if (key == "--arena-gib") {
            if (!parse_double(need_value(), opts.arena_gib) || opts.arena_gib <= 0.0) {
                fprintf(stderr, "invalid --arena-gib\n");
                return false;
            }
        } else if (key == "--runs") {
            if (!parse_int(need_value(), opts.runs) || opts.runs < 1) {
                fprintf(stderr, "invalid --runs\n");
                return false;
            }
        } else if (key == "--warmup") {
            if (!parse_int(need_value(), opts.warmup) || opts.warmup < 0) {
                fprintf(stderr, "invalid --warmup\n");
                return false;
            }
        } else if (key == "--output") {
            opts.output = need_value();
        } else if (key == "--help" || key == "-h") {
            print_usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "unknown option: %s\n", key.c_str());
            return false;
        }
    }
    if (opts.mode != "forward" && opts.mode != "greedy") {
        fprintf(stderr, "--mode must be forward or greedy\n");
        return false;
    }
    if (opts.model_dir.empty()) {
        fprintf(stderr, "--model-dir is required\n");
        return false;
    }
    return true;
}

bool roctx_available() {
#ifdef PHASESHIFT_HAS_ROCTX
    return true;
#else
    return false;
#endif
}

void roctx_resume() {
#ifdef PHASESHIFT_HAS_ROCTX
    roctxProfilerResume(0);
#endif
}

void roctx_pause() {
#ifdef PHASESHIFT_HAS_ROCTX
    roctxProfilerPause(0);
#endif
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

}  // anonymous namespace

int run_pp(int argc, char** argv) {
    BenchOptions opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(stderr);
        return 2;
    }

    roctx_pause();

    hipError_t hip_err = hipSetDevice(opts.device);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    hipStream_t stream = nullptr;
    hip_err = hipStreamCreate(&stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipStreamCreate failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    const std::size_t arena_bytes =
        static_cast<std::size_t>(opts.arena_gib * 1024.0 * 1024.0 * 1024.0);

    auto arena_result = ps::gpu::GpuArena::create(opts.device, arena_bytes);
    if (!arena_result.ok()) {
        fprintf(stderr, "GpuArena::create failed: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    auto model_result =
        ps::qwen35::Qwen35Model::load_from_safetensors(opts.model_dir, arena, stream);
    if (!model_result.ok()) {
        fprintf(stderr, "model load failed: %s\n", model_result.status().message().c_str());
        return 1;
    }
    auto model = model_result.release();

    const auto& cfg = model.text_config();
    const uint32_t kv_heads = static_cast<uint32_t>(cfg.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(cfg);

    hip_err = hipStreamSynchronize(stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "weight sync failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    const uint32_t required_seq_len =
        static_cast<uint32_t>(opts.prompt_tokens) + 16;

    const uint32_t num_pages = ceil_div(required_seq_len, static_cast<uint32_t>(opts.page_tokens)) + 1;

    auto seq_pool_result =
        ps::qwen35::SequenceSlotPool::create(arena, 1, num_pages);
    if (!seq_pool_result.ok()) {
        fprintf(stderr, "SequenceSlotPool::create failed: %s\n",
            seq_pool_result.status().message().c_str());
        return 1;
    }
    auto seq_pool = seq_pool_result.release();

    auto rec_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 1, ps::qwen35::GdnStatePoolLayout::from_text_config(model.text_config()));
    if (!rec_pool_result.ok()) {
        fprintf(stderr, "GdnStatePool::create failed: %s\n",
            rec_pool_result.status().message().c_str());
        return 1;
    }
    auto rec_pool = rec_pool_result.release();

    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, static_cast<uint32_t>(opts.page_tokens),
        num_attn_layers, kv_heads, head_dim, ps::qwen35::KVCacheDType::BF16);
    if (!kv_pool_result.ok()) {
        fprintf(stderr, "PagedKVPool::create failed: %s\n",
            kv_pool_result.status().message().c_str());
        return 1;
    }
    auto kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = std::max(opts.prompt_tokens, 16);
    exec_config.max_scheduled_requests = 1;

    auto exec_result = ps::qwen35::create_model_executor(
        model, seq_pool, rec_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        fprintf(stderr, "create_model_executor failed: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    auto executor = exec_result.release();

    const int32_t token_id = 1;
    if (token_id < 0 ||
        static_cast<std::size_t>(token_id) >= model.weights().embed_tokens.rows) {
        fprintf(stderr, "token_id out of vocab range: vocab_size=%zu\n",
                static_cast<std::size_t>(model.weights().embed_tokens.rows));
        return 1;
    }

    int32_t* prompt_tokens_host = nullptr;
    hip_err = hipHostAlloc(&prompt_tokens_host, sizeof(int32_t) * required_seq_len, hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipHostAlloc failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    for (int i = 0; i < static_cast<int>(required_seq_len); ++i) {
        prompt_tokens_host[i] = token_id;
    }

    hipEvent_t start_event = nullptr;
    hipEvent_t stop_event = nullptr;
    hip_err = hipEventCreate(&start_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    hip_err = hipEventCreate(&stop_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    std::vector<double> gpu_times_ms;
    std::vector<double> wall_times_ms;
    gpu_times_ms.reserve(static_cast<size_t>(opts.runs));
    wall_times_ms.reserve(static_cast<size_t>(opts.runs));

    auto run_once = [&](int run, bool measured) -> int {
        auto seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, rec_pool, kv_pool, required_seq_len, stream);
        if (!seq_result.ok()) {
            fprintf(stderr, "run %d create_paged_sequence_state failed: %s\n", run, seq_result.status().message().c_str());
            return 1;
        }
        ps::qwen35::PagedSequenceState seq = seq_result.release();

        const uint32_t total_tokens = static_cast<uint32_t>(opts.prompt_tokens);
        const uint32_t chunk = (opts.prefill_chunk > 0)
            ? std::min<uint32_t>(static_cast<uint32_t>(opts.prefill_chunk), total_tokens)
            : total_tokens;

        hip_err = hipStreamSynchronize(stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "run %d setup sync failed: %s\n", run, hipGetErrorString(hip_err));
            return 1;
        }

        roctx_resume();

        auto wall_start = std::chrono::high_resolution_clock::now();
        hip_err = hipEventRecord(start_event, stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventRecord failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        for (uint32_t processed = 0; processed < total_tokens;) {
            const uint32_t chunk_len = std::min(chunk, total_tokens - processed);
            const bool last = processed + chunk_len == total_tokens;
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq; req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::PREFILL;
            req.token_begin = 0;
            req.num_tokens = chunk_len;
            req.prefix_tokens = seq.position;
            req.compute_logits = last && opts.mode == "greedy";
            req.sample = last && opts.mode == "greedy";
            std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = prompt_tokens_host + processed;
            batch.requests = reqs;
            batch.num_tokens = chunk_len;
            batch.num_requests = 1;
            batch.num_decode_requests = 0;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 1;
            auto st = ps::qwen35::execute_batch(executor, batch, stream);
            if (!st.ok()) {
                fprintf(stderr, "execute_batch failed at %u: %s\n", processed,
                        st.status().message().c_str());
                return 1;
            }
            ps::qwen35::BatchExecutionOutput out = std::move(st.release());
            (void)out;
            processed += chunk_len;
        }

        hip_err = hipEventRecord(stop_event, stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventRecord failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
        hip_err = hipEventSynchronize(stop_event);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventSynchronize failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        auto wall_stop = std::chrono::high_resolution_clock::now();

        roctx_pause();

        float gpu_ms = 0.0f;
        hip_err = hipEventElapsedTime(&gpu_ms, start_event, stop_event);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventElapsedTime failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        if (measured) {
            const double wall_ms =
                std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
            gpu_times_ms.push_back(static_cast<double>(gpu_ms));
            wall_times_ms.push_back(wall_ms);
        }

        auto release_st = ps::qwen35::release_paged_sequence_state(seq, stream);
        if (!release_st.ok()) {
            fprintf(stderr, "run %d release_paged_sequence_state failed: %s\n", run, release_st.message().c_str());
            return 1;
        }
        return 0;
    };

    std::atomic<bool> keepalive_stop{false};
    hipStream_t keepalive_stream = nullptr;
    hipEvent_t keepalive_event = nullptr;
    std::thread keepalive_thread;
    if (opts.gap_keepalive > 0) {
        if (hipStreamCreate(&keepalive_stream) != hipSuccess) {
            fprintf(stderr, "keepalive hipStreamCreate failed\n");
            return 1;
        }
        const hipError_t ev_err = (opts.gap_keepalive_flags >= 0)
            ? hipEventCreateWithFlags(&keepalive_event,
                                      static_cast<unsigned>(opts.gap_keepalive_flags))
            : hipEventCreate(&keepalive_event);
        if (ev_err != hipSuccess) {
            fprintf(stderr, "keepalive event create failed: %s\n", hipGetErrorString(ev_err));
            return 1;
        }
        const int interval = opts.gap_keepalive;
        keepalive_thread = std::thread([&] {
            while (!keepalive_stop.load(std::memory_order_relaxed)) {
                const hipError_t ka_err = hipEventRecord(keepalive_event, keepalive_stream);
                (void)ka_err;
                std::this_thread::sleep_for(std::chrono::microseconds(interval));
            }
        });
    }

    for (int run = 0; run < opts.warmup; ++run) {
        if (run_once(run, false) != 0) return 1;
    }

    for (int run = 0; run < opts.runs; ++run) {
        if (run_once(run, true) != 0) return 1;
    }

    if (opts.gap_keepalive > 0) {
        keepalive_stop.store(true, std::memory_order_relaxed);
        keepalive_thread.join();
        hipError_t ka_err = hipStreamSynchronize(keepalive_stream);
        ka_err = hipEventDestroy(keepalive_event);
        ka_err = hipStreamDestroy(keepalive_stream);
        (void)ka_err;
    }

    hip_err = hipHostFree(prompt_tokens_host);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipHostFree failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    if (gpu_times_ms.empty()) {
        fprintf(stderr, "no valid measurements collected\n");
        return 1;
    }

    auto median = [](std::vector<double>& v) -> double {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        if (n == 0) return 0.0;
        if (n % 2 == 1) return v[n / 2];
        return (v[n / 2 - 1] + v[n / 2]) * 0.5;
    };

    auto percentile = [](std::vector<double>& v, double p) -> double {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        if (n == 0) return 0.0;
        double idx = p * static_cast<double>(n - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = lo + 1 < n ? lo + 1 : lo;
        double frac = idx - static_cast<double>(lo);
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };

    auto stddev = [](const std::vector<double>& v, double mean) -> double {
        if (v.size() < 2) return 0.0;
        double acc = 0.0;
        for (double x : v) {
            double d = x - mean;
            acc += d * d;
        }
        return std::sqrt(acc / static_cast<double>(v.size() - 1));
    };

    const double median_gpu_ms = median(gpu_times_ms);
    const double median_wall_ms = median(wall_times_ms);
    const double min_gpu_ms = gpu_times_ms.empty() ? 0.0 : *std::min_element(gpu_times_ms.begin(), gpu_times_ms.end());
    const double p10_gpu_ms = percentile(gpu_times_ms, 0.10);
    const double p90_gpu_ms = percentile(gpu_times_ms, 0.90);
    const double stddev_gpu_ms = stddev(gpu_times_ms, median_gpu_ms);

    const double median_gpu_us_per_token = median_gpu_ms * 1000.0 / static_cast<double>(opts.prompt_tokens);
    const double median_wall_us_per_token = median_wall_ms * 1000.0 / static_cast<double>(opts.prompt_tokens);
    const double median_gpu_tokens_per_sec =
        static_cast<double>(opts.prompt_tokens) / (median_gpu_ms / 1000.0);
    const double median_wall_tokens_per_sec =
        static_cast<double>(opts.prompt_tokens) / (median_wall_ms / 1000.0);

    const std::size_t kv_reserved_bytes = kv_pool.reserved_bytes();

    const int roctx_enabled = roctx_available() ? 1 : 0;

    printf("PhaseShift Prefill Benchmark\n");
    printf("\n");
    printf("device=%d\n", opts.device);
    printf("mode=%s\n", opts.mode.c_str());
    printf("prompt_tokens=%d\n", opts.prompt_tokens);
    printf("page_tokens=%d\n", opts.page_tokens);
    printf("runs=%d\n", opts.runs);
    printf("warmup=%d\n", opts.warmup);
    printf("\n");
    printf("gpu_ms_median=%.4f\n", median_gpu_ms);
    printf("gpu_ms_min=%.4f\n", min_gpu_ms);
    printf("gpu_ms_p10=%.4f\n", p10_gpu_ms);
    printf("gpu_ms_p90=%.4f\n", p90_gpu_ms);
    printf("gpu_ms_stddev=%.4f\n", stddev_gpu_ms);
    printf("wall_ms_median=%.4f\n", median_wall_ms);
    printf("gpu_us_per_token=%.2f\n", median_gpu_us_per_token);
    printf("wall_us_per_token=%.2f\n", median_wall_us_per_token);
    printf("gpu_tokens_per_sec=%.2f\n", median_gpu_tokens_per_sec);
    printf("wall_tokens_per_sec=%.2f\n", median_wall_tokens_per_sec);
    printf("\n");
    printf("kv_reserved_bytes=%zu\n", kv_reserved_bytes);
    printf("roctx=%s\n", roctx_enabled ? "enabled" : "disabled");
    printf("\n");
    printf("run_detail:\n");
    for (int i = 0; i < opts.runs; ++i) {
        const double run_gpu_us_per_token =
            gpu_times_ms[i] * 1000.0 / static_cast<double>(opts.prompt_tokens);
        const double run_wall_us_per_token =
            wall_times_ms[i] * 1000.0 / static_cast<double>(opts.prompt_tokens);
        const double run_gpu_tokens_per_sec =
            static_cast<double>(opts.prompt_tokens) / (gpu_times_ms[i] / 1000.0);
        const double run_wall_tokens_per_sec =
            static_cast<double>(opts.prompt_tokens) / (wall_times_ms[i] / 1000.0);
        printf("  run=%d gpu_ms=%.4f wall_ms=%.4f gpu_us_per_token=%.2f wall_us_per_token=%.2f "
               "gpu_tokens_per_sec=%.2f wall_tokens_per_sec=%.2f\n",
               i, gpu_times_ms[i], wall_times_ms[i], run_gpu_us_per_token, run_wall_us_per_token,
               run_gpu_tokens_per_sec, run_wall_tokens_per_sec);
    }

    if (!opts.output.empty()) {
        FILE* out = fopen(opts.output.c_str(), "a");
        if (out == nullptr) {
            fprintf(stderr, "cannot open output file: %s\n", opts.output.c_str());
            return 1;
        }
        if (ftell(out) == 0) {
            fprintf(out,
                "schema_version,device,mode,prompt_tokens,page_tokens,runs,warmup,"
                "gpu_elapsed_ms_median,wall_elapsed_ms_median,"
                "gpu_ms_min,gpu_ms_p10,gpu_ms_p90,gpu_ms_stddev,"
                "gpu_us_per_token,wall_us_per_token,gpu_tokens_per_sec,"
                "wall_tokens_per_sec,kv_reserved_bytes,roctx_enabled\n");
        }
        fprintf(out,
            "5,%d,%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%zu,%d\n",
            opts.device,
            opts.mode.c_str(),
            opts.prompt_tokens,
            opts.page_tokens,
            opts.runs,
            opts.warmup,
            median_gpu_ms,
            median_wall_ms,
            min_gpu_ms,
            p10_gpu_ms,
            p90_gpu_ms,
            stddev_gpu_ms,
            median_gpu_us_per_token,
            median_wall_us_per_token,
            median_gpu_tokens_per_sec,
            median_wall_tokens_per_sec,
            kv_reserved_bytes,
            roctx_enabled);
        fclose(out);
    }

    hip_err = hipEventDestroy(start_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventDestroy failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    hip_err = hipEventDestroy(stop_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventDestroy failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    auto shutdown_st = arena.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "arena shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }

    return 0;
}
