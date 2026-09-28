#include <phaseshift/models/qwen35/runtime/paged_attention_selector.h>

namespace ps::qwen35::runtime {
namespace {

struct PagedAttentionRule {
    ::ps::qwen35::KVCacheDType kv_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t page_tokens;
    uint32_t min_rows;
    uint32_t max_rows;
    uint32_t min_visible_tokens;
    uint32_t max_visible_tokens;
};

constexpr uint32_t kMeasuredMaxRows = 2048;
constexpr uint32_t kMeasuredMaxVisible = 32768;
constexpr uint32_t kMeasuredMaxVisible24 = 262144;

constexpr PagedAttentionRule kPagedAttentionRules[] = {
    { ::ps::qwen35::KVCacheDType::BF16, ::ps::runtime::ValueDType::F32,
      16, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible },
    { ::ps::qwen35::KVCacheDType::FP8_E4M3, ::ps::runtime::ValueDType::F32,
      16, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible },
    { ::ps::qwen35::KVCacheDType::BF16, ::ps::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::ps::qwen35::KVCacheDType::FP8_E4M3, ::ps::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::ps::qwen35::KVCacheDType::PSQ4_W32, ::ps::runtime::ValueDType::F32,
      16, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible },
    { ::ps::qwen35::KVCacheDType::PSQ4_W32, ::ps::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
    { ::ps::qwen35::KVCacheDType::PSQ8_W32, ::ps::runtime::ValueDType::F32,
      16, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible },
    { ::ps::qwen35::KVCacheDType::PSQ8_W32, ::ps::runtime::ValueDType::F32,
      24, 4, 256, 16, 1, kMeasuredMaxRows, 1, kMeasuredMaxVisible24 },
};

}  // namespace

PagedAttentionImplementation
select_paged_attention_implementation(const PagedAttentionSelectorInput& input) {
    if (input.rows == 0u)
        return PagedAttentionImplementation::Correctness;

    for (const auto& rule : kPagedAttentionRules) {
        if (rule.kv_dtype != input.kv_dtype
            || rule.output_dtype != input.output_dtype
            || rule.q_heads != input.q_heads
            || rule.kv_heads != input.kv_heads
            || rule.head_dim != input.head_dim
            || rule.page_tokens != input.page_tokens) {
            continue;
        }
        if (input.rows >= rule.min_rows && input.rows <= rule.max_rows
            && input.max_visible_tokens >= rule.min_visible_tokens
            && input.max_visible_tokens <= rule.max_visible_tokens) {
            return PagedAttentionImplementation::Optimized;
        }
        return PagedAttentionImplementation::Correctness;
    }

    return PagedAttentionImplementation::Correctness;
}

}  // namespace ps::qwen35::runtime
