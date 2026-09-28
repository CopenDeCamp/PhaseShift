#include <phaseshift/models/qwen35/runtime/embedding_selector.h>

#include <array>

namespace ps::qwen35::runtime {
namespace {

const std::array<EmbeddingRule, 4> kRules = {{
    {EmbeddingStorage::Bf16, 2560u, 1u, 2048u},
    {EmbeddingStorage::Bf16, 5120u, 1u, 2048u},
    {EmbeddingStorage::Psq8, 2560u, 1u, 2048u},
    {EmbeddingStorage::Psq8, 5120u, 1u, 2048u},
}};

}  // namespace

EmbeddingImplementation select_embedding_implementation(const EmbeddingSelectorInput& in) {
    if (in.rows == 0u || in.hidden_size == 0u) return EmbeddingImplementation::Correctness;
    for (const auto& r : kRules) {
        if (r.storage == in.storage && r.hidden_size == in.hidden_size &&
            in.rows >= r.min_rows && in.rows <= r.max_rows)
            return EmbeddingImplementation::Optimized;
    }
    return EmbeddingImplementation::Correctness;
}

}  // namespace ps::qwen35::runtime
