#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/runtime/parallel/comm_bridge.h>
#include <hip/hip_runtime.h>

#include "support/qwen35_synthetic_runner.h"
#include "support/synthetic_qwen35_model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace rt = ps::runtime;
namespace w8 = ps::weights;

struct BenchSettings {
    std::string model_dir;
    uint32_t total_layers = 4;
    uint64_t arena_bytes = 512ull * 1024ull * 1024ull;
    uint32_t prefill_tokens = 16;
    uint32_t decode_steps = 4;
};

struct ConfigResult {
    std::string name;
    bool ok = false;
    std::string error;
    uint64_t prefill_us = 0;
    uint64_t decode_us = 0;
    uint64_t p50_us = 0;
    uint64_t p95_us = 0;
    std::vector<uint64_t> bytes_per_rank;
    uint64_t vram_used_bytes = 0;
    std::vector<int32_t> tokens;
};

uint64_t max_u64(const std::vector<uint64_t>& values) {
    uint64_t out = 0;
    for (uint64_t v : values)
        if (v > out) out = v;
    return out;
}

uint64_t percentile(std::vector<uint64_t> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(p * static_cast<double>(values.size() - 1));
    return values[index];
}

void fill_shard(ps::qwen35::Qwen35TensorShard& shard, uint32_t rank, uint32_t world) {
    shard.column.axis = w8::ShardAxis::OutputFeatures;
    shard.column.rank = rank;
    shard.column.world_size = world;
    shard.row.axis = w8::ShardAxis::InputFeatures;
    shard.row.rank = rank;
    shard.row.world_size = world;
    shard.full_attention = true;
    shard.linear_attention = true;
    shard.mlp = true;
}

void report(const ConfigResult& r, const BenchSettings& settings) {
    if (!r.ok) {
        std::printf("%-8s FAILED %s\n", r.name.c_str(), r.error.c_str());
        return;
    }
    const double prefill_s = static_cast<double>(r.prefill_us) / 1e6;
    const double decode_s = static_cast<double>(r.decode_us) / 1e6;
    const double prefill_tps = prefill_s > 0.0 ? settings.prefill_tokens / prefill_s : 0.0;
    const double decode_tps = decode_s > 0.0 ? settings.decode_steps / decode_s : 0.0;
    const uint64_t total_tokens = settings.prefill_tokens + settings.decode_steps;
    const uint64_t bytes = max_u64(r.bytes_per_rank);
    const double bytes_per_token = total_tokens > 0
                                       ? static_cast<double>(bytes) / static_cast<double>(total_tokens)
                                       : 0.0;
    std::printf(
        "%-8s prefill_us=%-8llu prefill_tok_s=%-8.1f decode_us=%-8llu decode_tok_s=%-8.1f "
        "p50=%-7llu p95=%-7llu bytes=%-10llu bytes_per_token=%-8.0f vram_gb=%.2f\n",
        r.name.c_str(), static_cast<unsigned long long>(r.prefill_us), prefill_tps,
        static_cast<unsigned long long>(r.decode_us), decode_tps,
        static_cast<unsigned long long>(r.p50_us), static_cast<unsigned long long>(r.p95_us),
        static_cast<unsigned long long>(bytes), bytes_per_token,
        static_cast<double>(r.vram_used_bytes) / (1024.0 * 1024.0 * 1024.0));
    std::printf("          tokens:");
    for (int32_t t : r.tokens) std::printf(" %d", t);
    std::printf("\n");
}

void fill_timing(ConfigResult& out, const ps::test::SyntheticRunResult& result) {
    if (result.step_us.size() > 1) out.prefill_us = result.step_us[0];
    for (size_t i = 1; i < result.step_us.size(); ++i) out.decode_us += result.step_us[i];
    std::vector<uint64_t> decode(result.step_us.begin() + (result.step_us.empty() ? 0 : 1),
                                 result.step_us.end());
    out.p50_us = percentile(decode, 0.5);
    out.p95_us = percentile(decode, 0.95);
    out.vram_used_bytes = result.vram_used_bytes;
}

ConfigResult run_single(const BenchSettings& settings) {
    ConfigResult out;
    out.name = "1GPU";
    ps::test::SyntheticRunOptions options;
    options.model_dir = settings.model_dir;
    options.device = 0;
    options.sample = true;
    options.arena_bytes = settings.arena_bytes;
    options.prefill_tokens = settings.prefill_tokens;
    options.decode_steps = settings.decode_steps;
    ps::test::SyntheticRunResult result = ps::test::run_synthetic_model(options);
    if (!result.ok) {
        out.error = result.error;
        return out;
    }
    out.ok = true;
    fill_timing(out, result);
    out.tokens = result.tokens;
    return out;
}

ConfigResult run_parallel(const BenchSettings& settings, const char* name,
                          uint32_t pp_size, uint32_t tp_size,
                          std::vector<int> devices) {
    ConfigResult out;
    out.name = name;
    const uint32_t total_layers = settings.total_layers;
    const uint32_t stage1_begin = total_layers / 2;
    ps::test::SyntheticRunOptions base;
    base.model_dir = settings.model_dir;
    base.sample = true;
    base.arena_bytes = settings.arena_bytes;
    base.prefill_tokens = settings.prefill_tokens;
    base.decode_steps = settings.decode_steps;
    base.tensor_parallel_size = tp_size;
    base.tp_full_attention = true;
    base.tp_linear_attention = true;
    base.tp_mlp = true;
    if (pp_size > 1) {
        base.partition = {0, stage1_begin, true, false};
        base.sample = false;
        base.read_shared_tokens = true;
    }

    rt::ParallelConfig config;
    config.pp_size = pp_size;
    config.tp_size = tp_size;
    config.devices = devices;
    ps::Status cs = config.validate();
    if (!cs.ok()) {
        out.error = cs.message();
        return out;
    }
    auto transport_result = rt::RcclTransport::create(config);
    if (!transport_result.ok()) {
        out.error = transport_result.status().message();
        return out;
    }
    std::unique_ptr<rt::RcclTransport> transport = transport_result.release();

    std::vector<std::unique_ptr<rt::RcclCommEndpoint>> endpoints;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        auto endpoint = std::make_unique<rt::RcclCommEndpoint>();
        endpoint->transport = transport.get();
        endpoint->global_rank = rank;
        endpoints.push_back(std::move(endpoint));
    }

    std::vector<int32_t> shared_tokens;
    std::barrier step_barrier(static_cast<std::ptrdiff_t>(config.world_size()));

    std::vector<ps::test::SyntheticRunOptions> rank_options(config.world_size());
    const uint32_t last_rank = config.world_size() - 1;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        auto& o = rank_options[rank];
        o = base;
        o.device = static_cast<int>(rank);
        o.tensor_parallel_rank = rank % tp_size;
        o.comm_self = endpoints[rank].get();
        o.comm_launch = &rt::launch_rccl_communication;
        fill_shard(o.tensor_shard, rank % tp_size, tp_size);
        o.shared_tokens = &shared_tokens;
        o.step_barrier = &step_barrier;
        if (pp_size > 1) {
            const bool stage0 = rank / tp_size == 0;
            o.partition = stage0 ? ps::qwen35::ModelPartition{0, stage1_begin, true, false}
                                 : ps::qwen35::ModelPartition{stage1_begin, total_layers, false,
                                                              true};
            o.pipeline_peer_rank = stage0 ? rank + tp_size : rank - tp_size;
            o.sample = !stage0;
            o.read_shared_tokens = stage0;
            o.publish_shared_tokens = stage0 ? false : (rank == tp_size);
        }
    }

    std::vector<ps::test::SyntheticRunResult> results(config.world_size());
    std::vector<std::thread> workers;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        workers.emplace_back([&, rank] {
            results[rank] = ps::test::run_synthetic_model(rank_options[rank]);
        });
    }
    for (auto& w : workers) w.join();

    const uint32_t observer = pp_size > 1 ? tp_size : 0;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        if (!results[rank].ok) {
            out.error = "rank" + std::to_string(rank) + ": " + results[rank].error;
            (void)transport->shutdown();
            return out;
        }
    }
    fill_timing(out, results[observer]);
    out.tokens = results[observer].tokens;
    for (uint32_t rank = 0; rank < config.world_size(); ++rank) {
        out.bytes_per_rank.push_back(transport->issued_bytes(rank));
        if (results[rank].vram_used_bytes > out.vram_used_bytes)
            out.vram_used_bytes = results[rank].vram_used_bytes;
    }

    ps::Status shutdown = transport->shutdown();
    if (!shutdown.ok()) out.error = shutdown.message();
    out.ok = out.error.empty();
    if (!out.ok) return out;
    out.name = name;
    return out;
}

}

std::vector<std::string> split_csv(const char* value, const char* fallback) {
    std::vector<std::string> out;
    const std::string text = value != nullptr && value[0] != '\0' ? value : fallback;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;
        if (end > begin) out.push_back(text.substr(begin, end - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return out;
}

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    return std::strtoull(value, nullptr, 10);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    BenchSettings settings;
    const char* model_env = std::getenv("PHASESHIFT_BENCH_MODEL_DIR");
    const bool use_real_model = model_env != nullptr && model_env[0] != '\0';
    if (use_real_model) {
        settings.model_dir = model_env;
        settings.arena_bytes = 22ull * 1024ull * 1024ull * 1024ull;
        settings.prefill_tokens = 256;
        settings.decode_steps = 64;
    } else {
        settings.model_dir = "/tmp/opencode/ps_synthetic_qwen35_bench";
        ps::test::SyntheticQwen35Spec spec;
        spec.full_attention_interval = 2;
        spec.num_hidden_layers = 4;
        ps::Status write_status = ps::test::write_synthetic_qwen35_model(settings.model_dir, spec);
        if (!write_status.ok()) {
            std::printf("FAIL: write synthetic model: %s\n", write_status.message().c_str());
            return 1;
        }
    }
    if (const char* arena_env = std::getenv("PHASESHIFT_BENCH_ARENA_GB")) {
        if (arena_env[0] != '\0')
            settings.arena_bytes =
                std::strtoull(arena_env, nullptr, 10) * 1024ull * 1024ull * 1024ull;
    }
    settings.prefill_tokens =
        static_cast<uint32_t>(env_u64("PHASESHIFT_BENCH_PREFILL", settings.prefill_tokens));
    settings.decode_steps =
        static_cast<uint32_t>(env_u64("PHASESHIFT_BENCH_DECODE", settings.decode_steps));

    auto text_config = ps::qwen35::read_qwen35_text_config(settings.model_dir);
    if (!text_config.ok()) {
        std::printf("FAIL: read config: %s\n", text_config.status().message().c_str());
        return 1;
    }
    settings.total_layers = static_cast<uint32_t>(text_config.value().num_hidden_layers);

    const std::vector<std::string> selected =
        split_csv(std::getenv("PHASESHIFT_BENCH_CONFIGS"), "1gpu,pp2,tp2,pp2tp2");
    uint32_t needed_gpus = 1;
    for (const auto& name : selected) {
        if (name == "1gpu") continue;
        if (name == "pp2tp2") needed_gpus = 4;
        else if ((name == "pp2" || name == "tp2") && needed_gpus < 2) needed_gpus = 2;
    }
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess ||
        device_count < static_cast<int>(needed_gpus)) {
        std::printf("SKIP: need %u GPUs, have %d\n", needed_gpus, device_count);
        return 77;
    }

    std::printf("model=%s layers=%u arena_gb=%.1f prefill=%u decode=%u configs=%s\n",
                settings.model_dir.c_str(), settings.total_layers,
                static_cast<double>(settings.arena_bytes) / (1024.0 * 1024.0 * 1024.0),
                settings.prefill_tokens, settings.decode_steps,
                std::getenv("PHASESHIFT_BENCH_CONFIGS") != nullptr
                    ? std::getenv("PHASESHIFT_BENCH_CONFIGS")
                    : "1gpu,pp2,tp2,pp2tp2");

    std::vector<ConfigResult> results;
    for (const auto& name : selected) {
        if (name == "1gpu") results.push_back(run_single(settings));
        else if (name == "pp2") results.push_back(run_parallel(settings, "PP2", 2, 1, {0, 1}));
        else if (name == "tp2") results.push_back(run_parallel(settings, "TP2", 1, 2, {0, 1}));
        else if (name == "pp2tp2")
            results.push_back(run_parallel(settings, "PP2TP2", 2, 2, {0, 1, 2, 3}));
        else std::printf("unknown config: %s\n", name.c_str());
    }

    int failures = 0;
    for (const auto& r : results) {
        report(r, settings);
        if (!r.ok) ++failures;
    }

    if (failures != 0) {
        std::printf("%d configuration(s) failed\n", failures);
        return 1;
    }
    std::printf("PARALLEL_BENCH: PASS\n");
    return 0;
}
