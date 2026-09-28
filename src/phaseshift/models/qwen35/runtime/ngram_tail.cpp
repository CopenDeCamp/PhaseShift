#include <phaseshift/models/qwen35/runtime/ngram_tail.h>

#include <algorithm>
#include <cassert>

namespace ps::qwen35::runtime {

Result<NgramTailProposal> propose_ngram_tail(
    std::span<const int32_t> committed_history,
    const NgramTailConfig& config) {
    NgramTailProposal out;
    const std::size_t size = committed_history.size();
    out.current_position = static_cast<uint32_t>(size);

    if (config.n == 0u) {
        return Status::invalid_argument("propose_ngram_tail: n must be positive",
                                        __FILE__, __LINE__);
    }
    if (config.max_tail == 0u || config.window == 0u || size < config.n) {
        return out;
    }

    const std::size_t n = config.n;
    const std::size_t window = config.window;
    const std::size_t start = size > window ? size - window : 0u;
    const std::size_t seed_begin = size - n;
    const std::size_t candidate_limit = seed_begin;

    std::size_t candidate = 0u;
    bool found = false;
    for (std::size_t j = candidate_limit; j-- > start;) {
        bool eq = true;
        for (std::size_t t = 0u; t < n; ++t) {
            if (committed_history[j + t] != committed_history[seed_begin + t]) {
                eq = false;
                break;
            }
        }
        if (eq) {
            candidate = j;
            found = true;
            break;
        }
    }
    if (!found) {
        return out;
    }

    out.hit = true;
    out.match_position = static_cast<uint32_t>(candidate);
    out.seed_length = config.n;
    out.candidate_start = static_cast<uint32_t>(candidate);
    out.candidate_end = static_cast<uint32_t>(candidate + n);
    assert(out.candidate_start < out.current_position);
    assert(out.candidate_end <= out.current_position);

    const std::size_t available = size - (candidate + n);
    const std::size_t take = std::min<std::size_t>(config.max_tail, available);
    out.tokens.assign(
        committed_history.begin() + static_cast<std::ptrdiff_t>(candidate + n),
        committed_history.begin() + static_cast<std::ptrdiff_t>(candidate + n + take));
    assert(out.candidate_end + out.tokens.size() <= out.current_position);
    return out;
}

}  // namespace ps::qwen35::runtime
