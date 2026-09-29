#include <phaseshift/runtime/parallel/comm_bridge.h>
#include <hip/hip_runtime.h>

#include "support/qwen35_synthetic_runner.h"
#include "support/synthetic_qwen35_model.h"
#include "support/tp_linear_common.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace rt = ps::runtime;
namespace w8 = ps::weights;

using ps::test::compare_matrices;
using ps::test::HostMatrix;
using ps::test::MaxError;

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

HostMatrix as_matrix(const std::vector<float>& values, std::uint32_t rows,
                     std::uint32_t cols) {
    HostMatrix out;
    out.rows = rows;
    out.cols = cols;
    out.data.resize(static_cast<std::size_t>(rows) * cols);
    for (std::size_t i = 0; i < out.data.size() && i < values.size(); ++i)
        out.data[i] = ps::test::to_bf16(values[i]);
    return out;
}

w8::MatrixShardSpec make_spec(w8::ShardAxis axis, std::uint32_t rank) {
    w8::MatrixShardSpec spec;
    spec.axis = axis;
    spec.rank = rank;
    spec.world_size = 2;
    return spec;
}

std::size_t expected_collectives(const ps::test::SyntheticQwen35Spec& spec) {
    std::size_t expected = 0;
    for (std::uint32_t layer = 0; layer < spec.num_hidden_layers; ++layer)
        expected += 2;
    return expected;
}

}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: need 2 GPUs, have %d\n", device_count);
        return 77;
    }

    const std::string model_dir = "/tmp/opencode/ps_synthetic_qwen35_tp4";
    ps::test::SyntheticQwen35Spec spec;
    spec.full_attention_interval = 2;
    spec.num_hidden_layers = 4;
    ps::Status write_status = ps::test::write_synthetic_qwen35_model(model_dir, spec);
    if (!write_status.ok()) {
        std::printf("FAIL: write synthetic model: %s\n", write_status.message().c_str());
        return 1;
    }

    ps::test::SyntheticRunOptions baseline_options;
    baseline_options.model_dir = model_dir;
    baseline_options.device = 0;
    baseline_options.sample = true;
    ps::test::SyntheticRunResult baseline = ps::test::run_synthetic_model(baseline_options);
    if (!baseline.ok) {
        std::printf("FAIL: baseline: %s\n", baseline.error.c_str());
        return 1;
    }
    std::printf("baseline tokens:");
    for (int32_t t : baseline.tokens) std::printf(" %d", t);
    std::printf("\nbaseline collectives=%zu\n", baseline.comm_count);

    rt::ParallelConfig config;
    config.pp_size = 1;
    config.tp_size = 2;
    config.devices = {0, 1};
    ps::Status cs = config.validate();
    if (!cs.ok()) {
        std::printf("FAIL: parallel config: %s\n", cs.message().c_str());
        return 1;
    }
    auto transport_result = rt::RcclTransport::create(config);
    if (!transport_result.ok()) {
        std::printf("FAIL: rccl transport: %s\n",
                    transport_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<rt::RcclTransport> transport = transport_result.release();

    rt::RcclCommEndpoint endpoint0;
    endpoint0.transport = transport.get();
    endpoint0.global_rank = 0;
    rt::RcclCommEndpoint endpoint1;
    endpoint1.transport = transport.get();
    endpoint1.global_rank = 1;

    std::vector<ps::test::SyntheticRunOptions> rank_options(2);
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        auto& o = rank_options[rank];
        o.model_dir = model_dir;
        o.device = static_cast<int>(rank);
        o.sample = true;
        o.tensor_parallel_size = 2;
        o.tensor_parallel_rank = rank;
        o.tp_full_attention = true;
        o.tp_linear_attention = true;
        o.tp_mlp = true;
        o.tensor_shard.column = make_spec(w8::ShardAxis::OutputFeatures, rank);
        o.tensor_shard.row = make_spec(w8::ShardAxis::InputFeatures, rank);
        o.tensor_shard.full_attention = true;
        o.tensor_shard.linear_attention = true;
        o.tensor_shard.mlp = true;
        o.comm_self = rank == 0 ? &endpoint0 : &endpoint1;
        o.comm_launch = &rt::launch_rccl_communication;
    }

    std::vector<ps::test::SyntheticRunResult> rank_results(2);
    std::thread t0([&] { rank_results[0] = ps::test::run_synthetic_model(rank_options[0]); });
    std::thread t1([&] { rank_results[1] = ps::test::run_synthetic_model(rank_options[1]); });
    t0.join();
    t1.join();

    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        if (!rank_results[rank].ok) {
            std::printf("FAIL: rank%u: %s\n", rank, rank_results[rank].error.c_str());
            ++g_failed;
        }
    }
    if (g_failed != 0) {
        (void)transport->shutdown();
        return 1;
    }

    std::printf("tp4 rank0 tokens:");
    for (int32_t t : rank_results[0].tokens) std::printf(" %d", t);
    std::printf("\ntp4 rank1 tokens:");
    for (int32_t t : rank_results[1].tokens) std::printf(" %d", t);
    std::printf("\n");

    const std::size_t expected = expected_collectives(spec);
    std::printf("collectives: expected=%zu rank0=%zu rank1=%zu\n", expected,
                rank_results[0].comm_count, rank_results[1].comm_count);
    check(rank_results[0].comm_count == expected,
          "rank0 collective count matches the lowering plan");
    check(rank_results[1].comm_count == expected,
          "rank1 collective count matches the lowering plan");
    check(rank_results[0].comm_count == rank_results[1].comm_count,
          "both ranks lowered the same number of collectives");

    check(rank_results[0].tokens == rank_results[1].tokens,
          "tensor parallel ranks agree on sampled tokens");
    check(rank_results[0].logits == rank_results[1].logits,
          "tensor parallel ranks agree on logits");

    const std::uint32_t steps = static_cast<std::uint32_t>(baseline.tokens.size());
    const HostMatrix tp_logits = as_matrix(rank_results[0].logits, steps,
                                           static_cast<std::uint32_t>(spec.vocab_size));
    const HostMatrix base_logits = as_matrix(baseline.logits, steps,
                                             static_cast<std::uint32_t>(spec.vocab_size));
    const MaxError vs_baseline = compare_matrices(base_logits, tp_logits);
    std::printf(
        "tp4 vs 1GPU logits: mismatches=%zu/%zu max_abs=%.6g scale=%.6g max_abs/scale=%.6g\n",
        vs_baseline.mismatch, vs_baseline.count, vs_baseline.max_abs, vs_baseline.scale,
        vs_baseline.max_abs_ulp);
    check(vs_baseline.max_abs_ulp <= 0.05f,
          "tensor parallel logits stay within 5% of the 1GPU output scale");
    check(rank_results[0].tokens == baseline.tokens,
          "tensor parallel sampled tokens match the 1GPU baseline");

    ps::Status shutdown = transport->shutdown();
    if (!shutdown.ok()) {
        std::printf("FAIL: transport shutdown: %s\n", shutdown.message().c_str());
        ++g_failed;
    }

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("TP4_E2E: PASS\n");
    return 0;
}
