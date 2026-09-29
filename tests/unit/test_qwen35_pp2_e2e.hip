#include <phaseshift/runtime/parallel/comm_bridge.h>
#include <hip/hip_runtime.h>

#include "support/qwen35_synthetic_runner.h"
#include "support/synthetic_qwen35_model.h"

#include <barrier>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace rt = ::ps::runtime;

void report(const char* label, const std::string& error) {
    std::printf("FAIL: %s: %s\n", label, error.c_str());
}

bool compare_exact(const char* label, const std::vector<float>& expected,
                   const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        report(label, "size mismatch");
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::printf("FAIL: %s mismatch at %zu: %.9g vs %.9g\n", label, i, expected[i],
                        actual[i]);
            return false;
        }
    }
    return true;
}

bool compare_exact_u16(const char* label, const std::vector<std::uint16_t>& expected,
                       const std::vector<std::uint16_t>& actual) {
    if (expected.size() != actual.size()) {
        report(label, "size mismatch");
        return false;
    }
    size_t diffs = 0;
    size_t first = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            if (diffs == 0) first = i;
            ++diffs;
        }
    }
    if (diffs != 0) {
        std::printf("FAIL: %s diffs=%zu/%zu first=%zu: %u vs %u\n", label, diffs,
                    expected.size(), first, expected[first], actual[first]);
        return false;
    }
    return true;
}

bool compare_steps(const char* label,
                   const std::vector<std::vector<std::uint16_t>>& expected,
                   const std::vector<std::vector<std::uint16_t>>& actual) {
    if (expected.size() != actual.size()) {
        std::printf("FAIL: %s step count mismatch: %zu vs %zu\n", label, expected.size(),
                    actual.size());
        return false;
    }
    bool ok = true;
    for (size_t s = 0; s < expected.size(); ++s) {
        std::string name = std::string(label) + " step " + std::to_string(s);
        if (!compare_exact_u16(name.c_str(), expected[s], actual[s])) ok = false;
    }
    return ok;
}

}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: need 2 GPUs, have %d\n", device_count);
        return 77;
    }

    const std::string model_dir = "/tmp/opencode/ps_synthetic_qwen35_pp";
    ps::test::SyntheticQwen35Spec spec;
    spec.full_attention_interval = 2;
    spec.num_hidden_layers = 4;
    ps::Status write_status = ps::test::write_synthetic_qwen35_model(model_dir, spec);
    if (!write_status.ok()) {
        report("write synthetic model", write_status.message());
        return 1;
    }

    int failures = 0;
    const std::vector<uint32_t> taps = {1};

    ps::test::SyntheticRunOptions baseline_options;
    baseline_options.model_dir = model_dir;
    baseline_options.device = 0;
    baseline_options.sample = true;
    baseline_options.hidden_taps = taps;
    ps::test::SyntheticRunResult baseline = ps::test::run_synthetic_model(baseline_options);
    if (!baseline.ok) {
        report("baseline run", baseline.error);
        return 1;
    }
    std::printf("baseline tokens:");
    for (int32_t t : baseline.tokens) std::printf(" %d", t);
    std::printf("\n");

    rt::ParallelConfig config;
    config.pp_size = 2;
    config.tp_size = 1;
    config.devices = {0, 1};
    ps::Status config_status = config.validate();
    if (!config_status.ok()) {
        report("parallel config", config_status.message());
        return 1;
    }

    auto transport_result = rt::RcclTransport::create(config);
    if (!transport_result.ok()) {
        report("rccl transport", transport_result.status().message());
        return 1;
    }
    std::unique_ptr<rt::RcclTransport> transport = transport_result.release();

    rt::RcclCommEndpoint endpoint0;
    endpoint0.transport = transport.get();
    endpoint0.global_rank = 0;
    rt::RcclCommEndpoint endpoint1;
    endpoint1.transport = transport.get();
    endpoint1.global_rank = 1;

    std::vector<int32_t> shared_tokens;
    std::barrier step_barrier(2);

    ps::test::SyntheticRunOptions stage0_options;
    stage0_options.model_dir = model_dir;
    stage0_options.device = 0;
    stage0_options.partition = {0, 2, true, false};
    stage0_options.pipeline_peer_rank = 1;
    stage0_options.comm_self = &endpoint0;
    stage0_options.comm_launch = &rt::launch_rccl_communication;
    stage0_options.sample = false;
    stage0_options.hidden_taps = taps;
    stage0_options.shared_tokens = &shared_tokens;
    stage0_options.step_barrier = &step_barrier;

    ps::test::SyntheticRunOptions stage1_options;
    stage1_options.model_dir = model_dir;
    stage1_options.device = 1;
    stage1_options.partition = {2, 4, false, true};
    stage1_options.pipeline_peer_rank = 0;
    stage1_options.comm_self = &endpoint1;
    stage1_options.comm_launch = &rt::launch_rccl_communication;
    stage1_options.sample = true;
    stage1_options.shared_tokens = &shared_tokens;
    stage1_options.step_barrier = &step_barrier;

    ps::test::SyntheticRunResult stage0_result;
    ps::test::SyntheticRunResult stage1_result;
    std::thread stage0([&] {
        stage0_result = ps::test::run_synthetic_model(stage0_options);
        std::printf("stage0 finished ok=%d error=%s\n",
                    stage0_result.ok ? 1 : 0, stage0_result.error.c_str());
    });
    std::thread stage1([&] {
        stage1_result = ps::test::run_synthetic_model(stage1_options);
        std::printf("stage1 finished ok=%d error=%s\n",
                    stage1_result.ok ? 1 : 0, stage1_result.error.c_str());
    });
    stage0.join();
    stage1.join();

    if (!stage0_result.ok) {
        report("stage0 run", stage0_result.error);
        ++failures;
    }
    if (!stage1_result.ok) {
        report("stage1 run", stage1_result.error);
        ++failures;
    }
    if (failures != 0) {
        ps::Status shutdown = transport->shutdown();
        if (!shutdown.ok()) report("transport shutdown", shutdown.message());
        return 1;
    }

    std::printf("shared tokens:");
    for (int32_t t : shared_tokens) std::printf(" %d", t);
    std::printf("\n");
    std::printf("pp2 tokens:");
    for (int32_t t : stage1_result.tokens) std::printf(" %d", t);
    std::printf("\n");

    if (!compare_steps("stage0 boundary vs baseline", baseline.tap_boundary,
                       stage0_result.tap_boundary))
        ++failures;

    if (baseline.tokens != stage1_result.tokens) {
        report("sampled tokens", "PP2 does not match the single GPU baseline");
        ++failures;
    }
    if (!compare_exact("logits", baseline.logits, stage1_result.logits)) ++failures;
    if (!compare_exact_u16("final hidden", baseline.hidden, stage1_result.hidden))
        ++failures;

    ps::Status shutdown = transport->shutdown();
    if (!shutdown.ok()) {
        report("transport shutdown", shutdown.message());
        ++failures;
    }

    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("QWEN35_PP2_E2E: PASS\n");
    return 0;
}
