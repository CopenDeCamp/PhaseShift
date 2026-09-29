#include <phaseshift/runtime/parallel/comm_bridge.h>
#include <hip/hip_runtime.h>

#include "support/qwen35_synthetic_runner.h"
#include "support/synthetic_qwen35_model.h"
#include "support/tp_linear_common.h"

#include <barrier>
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

std::size_t expected_collectives(const ps::test::SyntheticQwen35Spec& spec,
                                 std::uint32_t layer_begin, std::uint32_t layer_end,
                                 bool boundary_comm) {
    std::size_t expected = 0;
    for (std::uint32_t layer = layer_begin; layer < layer_end; ++layer)
        expected += 2;
    if (boundary_comm) ++expected;
    return expected;
}

}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 4) {
        std::printf("SKIP: need 4 GPUs, have %d\n", device_count);
        return 77;
    }

    const std::string model_dir = "/tmp/opencode/ps_synthetic_qwen35_pp2tp2";
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
    std::printf("\n");

    rt::ParallelConfig config;
    config.pp_size = 2;
    config.tp_size = 2;
    config.devices = {0, 1, 2, 3};
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

    std::vector<std::unique_ptr<rt::RcclCommEndpoint>> endpoints;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        auto endpoint = std::make_unique<rt::RcclCommEndpoint>();
        endpoint->transport = transport.get();
        endpoint->global_rank = rank;
        endpoints.push_back(std::move(endpoint));
    }

    std::vector<int32_t> shared_tokens;
    std::barrier step_barrier(4);

    std::vector<ps::test::SyntheticRunOptions> rank_options(4);
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        auto& o = rank_options[rank];
        const bool stage0 = rank < 2;
        o.model_dir = model_dir;
        o.device = static_cast<int>(rank);
        o.pipeline_peer_rank = stage0 ? rank + 2 : rank - 2;
        o.tensor_parallel_size = 2;
        o.tensor_parallel_rank = rank % 2;
        o.tp_full_attention = true;
        o.tp_linear_attention = true;
        o.tp_mlp = true;
        o.tensor_shard.column = make_spec(w8::ShardAxis::OutputFeatures, rank % 2);
        o.tensor_shard.row = make_spec(w8::ShardAxis::InputFeatures, rank % 2);
        o.tensor_shard.full_attention = true;
        o.tensor_shard.linear_attention = true;
        o.tensor_shard.mlp = true;
        o.comm_self = endpoints[rank].get();
        o.comm_launch = &rt::launch_rccl_communication;
        o.shared_tokens = &shared_tokens;
        o.step_barrier = &step_barrier;
        if (stage0) {
            o.partition = {0, 2, true, false};
            o.sample = false;
            o.read_shared_tokens = true;
        } else {
            o.partition = {2, 4, false, true};
            o.sample = true;
            o.publish_shared_tokens = (rank == 2);
        }
    }

    std::vector<ps::test::SyntheticRunResult> rank_results(4);
    std::vector<std::thread> workers;
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        workers.emplace_back([&, rank] {
            rank_results[rank] = ps::test::run_synthetic_model(rank_options[rank]);
            std::printf("rank%u finished ok=%d error=%s\n", rank,
                        rank_results[rank].ok ? 1 : 0, rank_results[rank].error.c_str());
        });
    }
    for (auto& w : workers) w.join();

    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        if (!rank_results[rank].ok) {
            std::printf("FAIL: rank%u: %s\n", rank, rank_results[rank].error.c_str());
            ++g_failed;
        }
    }
    if (g_failed != 0) {
        (void)transport->shutdown();
        return 1;
    }

    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        std::printf("rank%u tokens:", rank);
        for (int32_t t : rank_results[rank].tokens) std::printf(" %d", t);
        std::printf(" collectives=%zu\n", rank_results[rank].comm_count);
    }

    const std::size_t expected_stage0 =
        expected_collectives(spec, 0, 2, true);
    const std::size_t expected_stage1 =
        expected_collectives(spec, 2, 4, true);
    std::printf("collectives: stage0 expected=%zu stage1 expected=%zu\n",
                expected_stage0, expected_stage1);
    check(rank_results[0].comm_count == expected_stage0,
          "stage0 rank0 collective count matches the lowering plan");
    check(rank_results[1].comm_count == expected_stage0,
          "stage0 rank1 collective count matches the lowering plan");
    check(rank_results[2].comm_count == expected_stage1,
          "stage1 rank2 collective count matches the lowering plan");
    check(rank_results[3].comm_count == expected_stage1,
          "stage1 rank3 collective count matches the lowering plan");

    check(rank_results[0].tokens == rank_results[1].tokens,
          "stage0 tensor ranks agree on sampled tokens");
    check(rank_results[2].tokens == rank_results[3].tokens,
          "stage1 tensor ranks agree on sampled tokens");
    check(rank_results[2].tokens == baseline.tokens,
          "pp2 tp2 sampled tokens match the 1GPU baseline");
    check(rank_results[0].logits == rank_results[1].logits,
          "stage0 tensor ranks agree on logits");
    check(rank_results[2].logits == rank_results[3].logits,
          "stage1 tensor ranks agree on logits");

    const std::uint32_t steps = static_cast<std::uint32_t>(baseline.tokens.size());
    const HostMatrix tp_logits = as_matrix(rank_results[2].logits, steps,
                                           static_cast<std::uint32_t>(spec.vocab_size));
    const HostMatrix base_logits = as_matrix(baseline.logits, steps,
                                             static_cast<std::uint32_t>(spec.vocab_size));
    const MaxError vs_baseline = compare_matrices(base_logits, tp_logits);
    std::printf(
        "pp2 tp2 vs 1GPU logits: mismatches=%zu/%zu max_abs=%.6g scale=%.6g "
        "max_abs/scale=%.6g\n",
        vs_baseline.mismatch, vs_baseline.count, vs_baseline.max_abs, vs_baseline.scale,
        vs_baseline.max_abs_ulp);
    check(vs_baseline.max_abs_ulp <= 0.05f,
          "pp2 tp2 logits stay within 5% of the 1GPU output scale");

    ps::Status shutdown = transport->shutdown();
    if (!shutdown.ok()) {
        std::printf("FAIL: transport shutdown: %s\n", shutdown.message().c_str());
        ++g_failed;
    }

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("PP2_TP2_E2E: PASS\n");
    return 0;
}
