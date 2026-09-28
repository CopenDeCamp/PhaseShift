#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/quantization/offline/kld_metrics.h>
#include <cstddef>
#include <string>

namespace ps::quantization::fpx {

struct KldOptions {
    std::string input_dir;
    std::string bundle_dir;
    std::string tokens_path;
    std::string report_path;
    std::string cache_dir = "/tmp/phaseshift-kld";
    std::size_t window = 512;
    std::size_t stride = 512;
    std::size_t max_eval_tokens = 0;
    std::size_t batch_positions = 16;
    double arena_gib = 20.0;
    bool self_test = false;
    bool keep_shadow = false;
    bool keep_logit_cache = false;
};

struct KldEvaluationSummary {
    KldResult kld;
    std::size_t evaluated_positions = 0;
};

Result<KldEvaluationSummary> evaluate_kld(const KldOptions& options);

}
