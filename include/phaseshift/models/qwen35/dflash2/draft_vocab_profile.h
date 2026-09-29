#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <optional>
#include <string>

namespace ps::qwen35::dflash2 {

struct DraftVocabResolveOptions {
    std::optional<std::string> draft_vocab;
    std::optional<std::string> draft_vocab_file;
    std::optional<std::string> int2_head;
};

struct DraftVocabResolution {
    std::string vocab_file;
    std::string reason;
    uint32_t int2_head_mode = 0u;
    bool use_subset = false;
    bool explicit_file = false;
};

DraftVocabResolveOptions draft_vocab_resolve_options_from_environment();

Result<DraftVocabResolution> resolve_draft_vocab_profile(
    const std::string& model_dir,
    uint32_t target_vocab_size,
    uint32_t target_hidden_size,
    const DraftVocabResolveOptions& options = {});

}
