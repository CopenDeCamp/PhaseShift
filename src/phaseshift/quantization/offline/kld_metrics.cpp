#include <phaseshift/quantization/offline/kld_metrics.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ps::quantization::fpx {

double logsumexp(const float* logits, std::size_t n) {
    if (n == 0) return 0.0;
    double max = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(logits[i]);
        if (v > max) max = v;
        if (!std::isfinite(v)) return std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(max)) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(logits[i]) - max);
    }
    return max + std::log(sum);
}

double softmax_at(const float* logits, std::size_t n, std::size_t idx) {
    if (idx >= n) return 0.0;
    const double lse = logsumexp(logits, n);
    if (!std::isfinite(lse)) return std::numeric_limits<double>::quiet_NaN();
    return std::exp(static_cast<double>(logits[idx]) - lse);
}

double kld_per_position(const float* logits_a, const float* logits_b, std::size_t vocab_size) {
    const double lse_a = logsumexp(logits_a, vocab_size);
    const double lse_b = logsumexp(logits_b, vocab_size);
    if (!std::isfinite(lse_a) || !std::isfinite(lse_b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double kl = 0.0;
    for (std::size_t i = 0; i < vocab_size; ++i) {
        const double a = static_cast<double>(logits_a[i]);
        const double log_p = a - lse_a;
        const double log_q = static_cast<double>(logits_b[i]) - lse_b;
        const double p = std::exp(log_p);
        kl += p * (log_p - log_q);
    }
    if (kl < 0.0 && kl > -1e-12) kl = 0.0;
    return kl;
}

double nll_per_position(const float* logits, std::size_t vocab_size, int32_t target) {
    if (target < 0 || static_cast<std::size_t>(target) >= vocab_size) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double lse = logsumexp(logits, vocab_size);
    if (!std::isfinite(lse)) return std::numeric_limits<double>::quiet_NaN();
    return lse - static_cast<double>(logits[target]);
}

KldResult aggregate_kld(const std::vector<double>& per_position) {
    KldResult result;
    if (per_position.empty()) return result;
    std::vector<double> sorted = per_position;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();
    auto nearest_rank = [&](double p) -> double {
        const std::size_t rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(n))) - 1;
        return sorted[std::min(rank, n - 1)];
    };
    double sum = 0.0;
    for (double value : sorted) sum += value;
    result.mean = sum / static_cast<double>(n);
    result.p50 = nearest_rank(0.50);
    result.p90 = nearest_rank(0.90);
    result.p99 = nearest_rank(0.99);
    result.p99_9 = nearest_rank(0.999);
    result.max = sorted.back();
    return result;
}

}
