#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps::quantization::fpx {

struct KldResult {
    double mean = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double p99_9 = 0.0;
    double max = 0.0;
};

double kld_per_position(const float* logits_a, const float* logits_b, std::size_t vocab_size);
double nll_per_position(const float* logits, std::size_t vocab_size, int32_t target);
KldResult aggregate_kld(const std::vector<double>& per_position);
double logsumexp(const float* logits, std::size_t n);
double softmax_at(const float* logits, std::size_t n, std::size_t idx);

}
