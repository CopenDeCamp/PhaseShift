#pragma once

#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace ps {
namespace qwen35 {

namespace runtime {

struct PrefixCheckpoint {
    std::vector<int32_t> tokens;
    uint32_t position = 0;
    std::vector<PageId> cache_pages;
    SequenceSlotId cache_state_slot = kInvalidSlot;
    uint64_t last_used = 0;
};

struct PrefixCacheStats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t inserts = 0;
    uint64_t duplicate_hits = 0;
    uint64_t evictions = 0;
    uint64_t skipped_too_large = 0;

    uint64_t restored_tokens = 0;
    uint64_t saved_tokens = 0;

    uint64_t save_d2d_bytes = 0;
    uint64_t restore_d2d_bytes = 0;
};

class PrefixCache {
 public:
    static Result<PrefixCache> create(
        gpu::GpuArena& arena,
        const PagedKVPool& active_kv,
        const GdnStatePool& active_gdn,
        uint32_t capacity_tokens,
        uint32_t max_entries,
        uint32_t min_prefix_tokens = 64);

    PrefixCache(PrefixCache&&) noexcept = default;
    PrefixCache& operator=(PrefixCache&&) noexcept = delete;
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    bool enabled() const noexcept {
        return capacity_tokens_ != 0 && max_entries_ != 0;
    }

    uint32_t capacity_tokens() const noexcept { return capacity_tokens_; }
    uint32_t max_entries() const noexcept { return max_entries_; }
    uint32_t cache_pages() const noexcept {
        return cache_kv_pool_ ? cache_kv_pool_->num_pages() : 0;
    }
    std::size_t cache_kv_bytes() const noexcept {
        return cache_kv_pool_ ? cache_kv_pool_->reserved_bytes() : 0;
    }
    std::size_t cache_gdn_bytes() const noexcept {
        return cache_gdn_pool_ ? cache_gdn_pool_->reserved_bytes() : 0;
    }
    uint32_t cache_free_pages() const noexcept {
        return cache_kv_pool_ ? cache_kv_pool_->num_free_pages() : 0;
    }
    uint32_t cache_free_slots() const noexcept {
        return static_cast<uint32_t>(free_state_slots_.size());
    }

    const PrefixCheckpoint* find_longest(
        const std::vector<int32_t>& tokens,
        bool strictly_shorter = false);

    Status save(
        const PagedSequenceState& sequence,
        const GdnStatePool& gdn_pool,
        const PagedKVPool& kv_pool,
        std::vector<int32_t> tokens,
        hipStream_t stream,
        bool prompt_boundary = false);

    Status restore(
        PagedSequenceState& sequence,
        GdnStatePool& gdn_pool,
        PagedKVPool& kv_pool,
        const PrefixCheckpoint& checkpoint,
        hipStream_t stream);

    void clear() noexcept;

    std::size_t size() const noexcept { return entries_.size(); }

    const PrefixCacheStats& stats() const noexcept { return stats_; }

 private:
    PrefixCache() = default;

    std::vector<PrefixCheckpoint>::iterator find_exact(const std::vector<int32_t>& tokens);
    void evict_lru();
    void release_checkpoint(PrefixCheckpoint& entry) noexcept;

    std::optional<PagedKVPool> cache_kv_pool_;
    std::optional<GdnStatePool> cache_gdn_pool_;

    uint32_t capacity_tokens_ = 0;
    uint32_t max_entries_ = 0;
    uint32_t min_prefix_tokens_ = 64;

    std::vector<PrefixCheckpoint> entries_;
    std::vector<uint32_t> free_state_slots_;
    uint64_t clock_ = 0;
    PrefixCacheStats stats_{};
};

}
}
}
