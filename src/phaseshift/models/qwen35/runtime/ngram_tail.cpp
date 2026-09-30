#include <phaseshift/models/qwen35/runtime/ngram_tail.h>

#include <algorithm>
#include <cassert>

namespace ps::qwen35::runtime {

Status propose_ngram_tail_into(
    std::span<const int32_t> committed_history,
    std::span<const int32_t> read_only_suffix,
    const NgramTailConfig& config,
    std::span<int32_t> output,
    NgramTailMatch& out_match) {
    out_match = NgramTailMatch{};

    if (config.n == 0u) {
        return Status::invalid_argument("propose_ngram_tail: n must be positive",
                                        __FILE__, __LINE__);
    }

    const std::size_t hsize = committed_history.size();
    const std::size_t ssize = read_only_suffix.size();
    const std::size_t vsize = hsize + ssize;
    const std::size_t n = config.n;

    if (config.max_tail == 0u || config.window == 0u || vsize < n || hsize < n + 1u) {
        return Status::make_ok();
    }

    const std::size_t start = vsize > config.window ? vsize - config.window : 0u;
    const std::size_t upper = hsize - n - 1u;
    if (start > upper) {
        return Status::make_ok();
    }

    const std::size_t seed_begin = vsize - n;
    const auto seed_at = [&](std::size_t index) -> int32_t {
        if (index < hsize) return committed_history[index];
        return read_only_suffix[index - hsize];
    };

    std::size_t candidate = 0u;
    bool found = false;
    for (std::size_t j = upper + 1u; j-- > start;) {
        bool eq = true;
        for (std::size_t t = 0u; t < n; ++t) {
            if (committed_history[j + t] != seed_at(seed_begin + t)) {
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
        return Status::make_ok();
    }

    out_match.hit = true;
    out_match.candidate_start = static_cast<uint32_t>(candidate);
    out_match.candidate_end = static_cast<uint32_t>(candidate + n);
    out_match.seed_length = config.n;
    out_match.current_position = static_cast<uint32_t>(hsize);
    assert(out_match.candidate_start < out_match.current_position);
    assert(out_match.candidate_end <= out_match.current_position);

    const std::size_t candidate_end = candidate + n;
    const std::size_t available = hsize - candidate_end;
    const std::size_t take = std::min<std::size_t>(
        config.max_tail, std::min<std::size_t>(available, output.size()));
    for (std::size_t i = 0u; i < take; ++i) {
        output[i] = committed_history[candidate_end + i];
    }
    out_match.count = static_cast<uint32_t>(take);
    assert(out_match.candidate_end + out_match.count <= out_match.current_position);
    return Status::make_ok();
}

Result<NgramTailProposal> propose_ngram_tail(
    std::span<const int32_t> committed_history,
    const NgramTailConfig& config) {
    return propose_ngram_tail(committed_history, std::span<const int32_t>{}, config);
}

Result<NgramTailProposal> propose_ngram_tail(
    std::span<const int32_t> committed_history,
    std::span<const int32_t> read_only_suffix,
    const NgramTailConfig& config) {
    NgramTailMatch match;
    std::vector<int32_t> buffer(config.max_tail);
    Status status =
        propose_ngram_tail_into(committed_history, read_only_suffix, config, buffer, match);
    if (!status.ok()) return status;

    NgramTailProposal proposal;
    proposal.hit = match.hit;
    proposal.match_position = match.candidate_start;
    proposal.seed_length = match.seed_length;
    proposal.candidate_start = match.candidate_start;
    proposal.candidate_end = match.candidate_end;
    proposal.current_position = match.current_position;
    proposal.tokens.assign(buffer.begin(),
                           buffer.begin() + static_cast<std::ptrdiff_t>(match.count));
    return proposal;
}

}  // namespace ps::qwen35::runtime
