#include <hip/hip_runtime.h>

#include "support/qwen35_synthetic_runner.h"
#include "support/synthetic_qwen35_model.h"

#include <cstdio>
#include <string>

namespace {

void print_result(const char* label, const ps::test::SyntheticRunResult& result) {
    for (size_t i = 0; i < result.tokens.size(); ++i)
        std::printf("%s step %zu: token=%d\n", label, i, result.tokens[i]);
}

}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    const std::string model_dir = "/tmp/opencode/ps_synthetic_qwen35";
    ps::test::SyntheticQwen35Spec spec;
    spec.full_attention_interval = 2;
    ps::Status write_status = ps::test::write_synthetic_qwen35_model(model_dir, spec);
    if (!write_status.ok()) {
        std::printf("FAIL: write synthetic model: %s\n", write_status.message().c_str());
        return 1;
    }

    ps::test::SyntheticRunOptions options;
    options.model_dir = model_dir;
    options.device = 0;

    ps::test::SyntheticRunResult first = ps::test::run_synthetic_model(options);
    if (!first.ok) {
        std::printf("FAIL: run 1: %s\n", first.error.c_str());
        return 1;
    }
    ps::test::SyntheticRunResult second = ps::test::run_synthetic_model(options);
    if (!second.ok) {
        std::printf("FAIL: run 2: %s\n", second.error.c_str());
        return 1;
    }
    if (first.tokens != second.tokens) {
        std::printf("FAIL: sampling is not deterministic\n");
        return 1;
    }
    if (first.logits != second.logits) {
        std::printf("FAIL: logits are not deterministic\n");
        return 1;
    }
    if (first.hidden != second.hidden) {
        std::printf("FAIL: final hidden is not deterministic\n");
        return 1;
    }
    if (first.logits.empty() || first.hidden.empty()) {
        std::printf("FAIL: run produced no outputs\n");
        return 1;
    }

    print_result("run", first);
    std::printf("QWEN35_SYNTHETIC_E2E: PASS\n");
    return 0;
}
