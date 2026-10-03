#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ps::qwen35::runtime {

struct NgramTailConfig {
    uint32_t n = 4u;
    uint32_t max_tail = 16u;
    uint32_t window = 2048u;
};

struct NgramTailMatch {
    bool hit = false;
    uint32_t candidate_start = 0u;
    uint32_t candidate_end = 0u;
    uint32_t seed_length = 0u;
    uint32_t current_position = 0u;
    uint32_t count = 0u;
};

struct NgramTailProposal {
    bool hit = false;
    uint32_t match_position = 0u;
    uint32_t seed_length = 0u;
    uint32_t candidate_start = 0u;
    uint32_t candidate_end = 0u;
    uint32_t current_position = 0u;
    std::vector<int32_t> tokens;
};

Status propose_ngram_tail_into(
    std::span<const int32_t> committed_history,
    std::span<const int32_t> read_only_suffix,
    const NgramTailConfig& config,
    std::span<int32_t> output,
    NgramTailMatch& out_match);

Result<NgramTailProposal> propose_ngram_tail(
    std::span<const int32_t> committed_history,
    const NgramTailConfig& config);

Result<NgramTailProposal> propose_ngram_tail(
    std::span<const int32_t> committed_history,
    std::span<const int32_t> read_only_suffix,
    const NgramTailConfig& config);

}  // namespace ps::qwen35::runtime
