#include <phaseshift/models/qwen35/runtime/mcu_plan_cache.h>

#include <cstddef>

namespace ps {
namespace qwen35 {
namespace runtime {
namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

uint64_t hash_bytes(uint64_t hash, const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<uint64_t>(p[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
uint64_t hash_vector(uint64_t hash, const std::vector<T>& values) noexcept {
    if (values.empty()) return hash;
    return hash_bytes(hash, values.data(), values.size() * sizeof(T));
}

}  // namespace

McuPlanFingerprint mcu_plan_fingerprint(const McuCompiledPlan& plan) noexcept {
    if (plan.fingerprint != 0ull) {
        McuPlanFingerprint stored{};
        stored.hash = plan.fingerprint;
        stored.dispatch_count = plan.dispatch_count;
        stored.variant_count =
            static_cast<uint32_t>(plan.variants.size());
        return stored;
    }
    return mcu_plan_raw_fingerprint(plan);
}

McuPlanFingerprint mcu_plan_raw_fingerprint(const McuCompiledPlan& plan) noexcept {
    uint64_t hash = kFnvOffset;
    hash = hash_vector(hash, plan.nodes);
    hash = hash_vector(hash, plan.variants);
    hash = hash_vector(hash, plan.rms);
    hash = hash_vector(hash, plan.quant_e4m3);
    hash = hash_vector(hash, plan.psq4);
    hash = hash_vector(hash, plan.elementwise);
    hash = hash_vector(hash, plan.rope);
    hash = hash_vector(hash, plan.kv_append);
    hash = hash_vector(hash, plan.attention_paged);
    hash = hash_vector(hash, plan.attention_paged_split);
    hash = hash_vector(hash, plan.attention_paged_reduce);
    hash = hash_vector(hash, plan.bf16);
    hash = hash_vector(hash, plan.bf16_wmma);
    hash = hash_vector(hash, plan.l2);
    hash = hash_vector(hash, plan.embedding);
    hash = hash_vector(hash, plan.embedding_psq8);
    hash = hash_vector(hash, plan.psq4_multi);
    hash = hash_vector(hash, plan.gdn_conv1d);
    hash = hash_vector(hash, plan.gdn_recurrence);
    hash = hash_vector(hash, plan.gdn_reset);

    McuPlanFingerprint fingerprint{};
    fingerprint.hash = hash;
    fingerprint.dispatch_count = plan.dispatch_count;
    fingerprint.variant_count =
        static_cast<uint32_t>(plan.variants.size());
    return fingerprint;
}

const char* mcu_plan_shape_name(McuPlanShape shape) noexcept {
    switch (shape) {
        case McuPlanShape::Direct:
            return "Direct";
        case McuPlanShape::Split:
            return "Split";
        default:
            return "unknown";
    }
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
