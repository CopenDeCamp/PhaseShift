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

struct ConfigResult {
    std::string name;
    bool ok = false;
    std::string error;
    uint64_t prefill_us = 0;
    uint64_t decode_us = 0;
    uint64_t p50_us = 0;
    uint64_t p95_us = 0;
    std::vector<uint64_t> bytes_per_rank;
};

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

void report(const ConfigResult& r) {
    if (!r.ok) {
        std::printf("%-8s FAILED %s\n", r.name.c_str(), r.error.c_str());
        return;
    }
    std::printf("%-8s prefill_us=%-8llu decode_us=%-8llu p50=%-7llu p95=%-7llu bytes:",
                r.name.c_str(), static_cast<unsigned long long>(r.prefill_us),
                static_cast<unsigned long long>(r.decode_us),
                static_cast<unsigned long long>(r.p50_us),
                static_cast<unsigned long long>(r.p95_us));
    if (r.bytes_per_rank.empty()) {
        std::printf(" 0");
    } else {
        for (uint64_t b : r.bytes_per_rank)
            std::printf(" %llu", static_cast<unsigned long long>(b));
    }
    std::printf("\n");
}

ConfigResult run_single(const std::string& model_dir) {
    ConfigResult out;
    out.name = "1GPU";
    ps::test::SyntheticRunOptions options;
    options.model_dir = model_dir;
    options.device = 0;
    options.sample = true;
    ps::test::SyntheticRunResult result = ps::test::run_synthetic_model(options);
    if (!result.ok) {
        out.error = result.error;
        return out;
    }
    out.ok = true;
    if (result.step_us.size() > 1)
        out.prefill_us = result.step_us[0];
    for (size_t i = 1; i < result.step_us.size(); ++i) out.decode_us += result.step_us[i];
    std::vector<uint64_t> decode(result.step_us.begin() + (result.step_us.empty() ? 0 : 1),
                                 result.step_us.end());
    out.p50_us = percentile(decode, 0.5);
    out.p95_us = percentile(decode, 0.95);
    return out;
}

ConfigResult run_parallel(const std::string& model_dir, const char* name,
                          uint32_t pp_size, uint32_t tp_size,
                          std::vector<int> devices) {
    ConfigResult out;
    out.name = name;
    ps::test::SyntheticRunOptions base;
    base.model_dir = model_dir;
    base.sample = true;
    base.tensor_parallel_size = tp_size;
    base.tp_full_attention = true;
    base.tp_linear_attention = true;
    base.tp_mlp = true;
    if (pp_size > 1) {
        base.partition = {0, 2, true, false};
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
            o.partition = stage0 ? ps::qwen35::ModelPartition{0, 2, true, false}
                                 : ps::qwen35::ModelPartition{2, 4, false, true};
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
    const auto& result = results[observer];
    if (result.step_us.size() > 1) out.prefill_us = result.step_us[0];
    for (size_t i = 1; i < result.step_us.size(); ++i) out.decode_us += result.step_us[i];
    std::vector<uint64_t> decode(result.step_us.begin() + (result.step_us.empty() ? 0 : 1),
                                 result.step_us.end());
    out.p50_us = percentile(decode, 0.5);
    out.p95_us = percentile(decode, 0.95);
    for (uint32_t rank = 0; rank < config.world_size(); ++rank)
        out.bytes_per_rank.push_back(transport->issued_bytes(rank));

    ps::Status shutdown = transport->shutdown();
    if (!shutdown.ok()) out.error = shutdown.message();
    out.ok = out.error.empty();
    if (!out.ok) return out;
    out.name = name;
    return out;
}

}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 4) {
        std::printf("SKIP: need 4 GPUs, have %d\n", device_count);
        return 77;
    }

    const std::string model_dir = "/tmp/opencode/ps_synthetic_qwen35_bench";
    ps::test::SyntheticQwen35Spec spec;
    spec.full_attention_interval = 2;
    spec.num_hidden_layers = 4;
    ps::Status write_status = ps::test::write_synthetic_qwen35_model(model_dir, spec);
    if (!write_status.ok()) {
        std::printf("FAIL: write synthetic model: %s\n", write_status.message().c_str());
        return 1;
    }

    std::vector<ConfigResult> results;
    results.push_back(run_single(model_dir));
    results.push_back(run_parallel(model_dir, "PP2", 2, 1, {0, 1}));
    results.push_back(run_parallel(model_dir, "TP2", 1, 2, {0, 1}));
    results.push_back(run_parallel(model_dir, "PP2TP2", 2, 2, {0, 1, 2, 3}));

    int failures = 0;
    for (const auto& r : results) {
        report(r);
        if (!r.ok) ++failures;
    }

    if (failures != 0) {
        std::printf("%d configuration(s) failed\n", failures);
        return 1;
    }
    std::printf("PARALLEL_BENCH: PASS\n");
    return 0;
}
