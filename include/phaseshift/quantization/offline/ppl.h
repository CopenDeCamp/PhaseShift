#pragma once
#include <phaseshift/core/status.h>
#include <cstdint>
#include <string>

namespace ps::quantization::fpx {

struct PplOptions {
    std::string model_dir;
    std::string tokens_path;
    std::size_t max_tokens = 2048;
    double arena_gib = 24.0;
    int device = 0;
};

struct PplSummary {
    uint64_t positions = 0;
    double mean_nll = 0.0;
    double perplexity = 0.0;
    int argmax_first = -1;
};

Result<PplSummary> evaluate_ppl(const PplOptions& options);

}
