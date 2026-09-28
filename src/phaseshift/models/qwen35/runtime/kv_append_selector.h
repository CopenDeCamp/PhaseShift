#pragma once
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <cstdint>

namespace ps::qwen35::runtime {

enum class KvAppendImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct KvAppendSelectorInput {
    ::ps::qwen35::KVCacheDType kv_dtype = ::ps::qwen35::KVCacheDType::BF16;
    uint32_t rows = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_tokens = 0;
};

KvAppendImplementation select_kv_append_implementation(const KvAppendSelectorInput& input);

}  // namespace ps::qwen35::runtime
