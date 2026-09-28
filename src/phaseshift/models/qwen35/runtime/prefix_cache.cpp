#include <phaseshift/models/qwen35/runtime/prefix_cache.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace ps {
namespace qwen35 {

namespace runtime {

namespace {

uint32_t pages_for(uint32_t position, uint32_t page_tokens) {
    if (page_tokens == 0) return 0;
    return (position + page_tokens - 1u) / page_tokens;
}

bool trace_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PHASESHIFT_PREFIX_CACHE_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

std::size_t kv_page_bytes(const PagedKVPool& pool) {
    const std::size_t elems = static_cast<std::size_t>(pool.page_tokens())
        * pool.kv_heads() * pool.head_dim();
    if (pool.dtype() == KVCacheDType::FP8_E4M3) {
        return elems * sizeof(fp8e4m3_storage_t);
    }
    if (pool.dtype() == KVCacheDType::PSQ4_W32) {
        const std::size_t blocks = static_cast<std::size_t>(pool.page_tokens())
            * pool.kv_heads() * pool.psq4_blocks_per_head();
        return blocks * kPsq4CodeBytesPerBlock;
    }
    if (pool.dtype() == KVCacheDType::PSQ8_W32) {
        const std::size_t blocks = static_cast<std::size_t>(pool.page_tokens())
            * pool.kv_heads() * pool.psq8_blocks_per_head();
        return blocks * kPsq8CodeBytesPerBlock;
    }
    return elems * sizeof(bf16_t);
}

std::size_t kv_page_scale_bytes(const PagedKVPool& pool) {
    if (pool.dtype() == KVCacheDType::FP8_E4M3) {
        return static_cast<std::size_t>(pool.page_tokens()) * pool.kv_heads() * sizeof(float);
    }
    if (pool.dtype() == KVCacheDType::PSQ4_W32) {
        const std::size_t blocks = static_cast<std::size_t>(pool.page_tokens())
            * pool.kv_heads() * pool.psq4_blocks_per_head();
        return blocks * kPsq4ScaleBytesPerBlock;
    }
    if (pool.dtype() == KVCacheDType::PSQ8_W32) {
        const std::size_t blocks = static_cast<std::size_t>(pool.page_tokens())
            * pool.kv_heads() * pool.psq8_blocks_per_head();
        return blocks * kPsq8ScaleBytesPerBlock;
    }
    return 0;
}

std::size_t gdn_slot_bytes(const GdnStatePool& pool) {
    const std::size_t conv =
        static_cast<std::size_t>(pool.num_gdn_states())
        * pool.conv_history() * pool.conv_history_stride() * sizeof(bf16_t);
    const std::size_t rec =
        static_cast<std::size_t>(pool.num_gdn_states())
        * pool.num_v_heads() * pool.head_k() * pool.head_v() * sizeof(float);
    return conv + rec;
}

}

Result<PrefixCache> PrefixCache::create(
    gpu::GpuArena& arena,
    const PagedKVPool& active_kv,
    const GdnStatePool& active_gdn,
    uint32_t capacity_tokens,
    uint32_t max_entries,
    uint32_t min_prefix_tokens) {
    if (capacity_tokens == 0) {
        return Status::invalid_argument("prefix cache capacity must be > 0", __FILE__, __LINE__);
    }
    if (max_entries == 0) {
        return Status::invalid_argument("prefix cache max_entries must be > 0", __FILE__, __LINE__);
    }
    if (active_kv.page_tokens() == 0) {
        return Status::invalid_argument("prefix cache requires page_tokens > 0", __FILE__, __LINE__);
    }

    const uint32_t cache_pages = pages_for(capacity_tokens, active_kv.page_tokens());
    if (cache_pages == 0) {
        return Status::invalid_argument("prefix cache capacity below one page", __FILE__, __LINE__);
    }

    auto kv_result = PagedKVPool::create(
        arena,
        cache_pages,
        active_kv.page_tokens(),
        active_kv.num_attention_layers(),
        active_kv.kv_heads(),
        active_kv.head_dim(),
        active_kv.dtype());
    if (!kv_result.ok()) return kv_result.status();

    auto gdn_result = GdnStatePool::create(arena, max_entries, active_gdn.layout());
    if (!gdn_result.ok()) return gdn_result.status();

    PrefixCache cache;
    cache.cache_kv_pool_.emplace(std::move(kv_result.release()));
    cache.cache_gdn_pool_.emplace(std::move(gdn_result.release()));
    cache.capacity_tokens_ = capacity_tokens;
    cache.max_entries_ = max_entries;
    cache.min_prefix_tokens_ = min_prefix_tokens;
    cache.free_state_slots_.reserve(max_entries);
    for (uint32_t i = 0; i < max_entries; ++i) {
        cache.free_state_slots_.push_back(i);
    }
    return Result<PrefixCache>(std::move(cache));
}

std::vector<PrefixCheckpoint>::iterator PrefixCache::find_exact(
    const std::vector<int32_t>& tokens) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->tokens == tokens) return it;
    }
    return entries_.end();
}

const PrefixCheckpoint* PrefixCache::find_longest(
    const std::vector<int32_t>& tokens, bool strictly_shorter) {
    if (!enabled()) return nullptr;
    ++stats_.lookups;

    PrefixCheckpoint* best = nullptr;
    std::size_t best_len = 0;
    for (PrefixCheckpoint& entry : entries_) {
        const std::size_t n = entry.tokens.size();
        if (n == 0 || n > tokens.size() || n <= best_len) {
            continue;
        }
        if (strictly_shorter && n >= tokens.size()) {
            continue;
        }
        if (!std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin())) {
            continue;
        }
        best = &entry;
        best_len = n;
    }

    if (best == nullptr) {
        ++stats_.misses;
        if (trace_enabled()) {
            std::fprintf(stderr, "PREFIX_CACHE_MISS\n");
        }
        return nullptr;
    }

    best->last_used = ++clock_;
    ++stats_.hits;
    if (trace_enabled()) {
        std::fprintf(stderr, "PREFIX_CACHE_HIT tokens=%u\n",
                     static_cast<unsigned>(best->tokens.size()));
    }
    return best;
}

void PrefixCache::release_checkpoint(PrefixCheckpoint& entry) noexcept {
    if (cache_kv_pool_ && !entry.cache_pages.empty()) {
        (void)cache_kv_pool_->release_many(entry.cache_pages);
        entry.cache_pages.clear();
    }
    if (cache_gdn_pool_ && entry.cache_state_slot != kInvalidSlot) {
        (void)cache_gdn_pool_->release(entry.cache_state_slot);
        free_state_slots_.push_back(entry.cache_state_slot);
        entry.cache_state_slot = kInvalidSlot;
    }
}

void PrefixCache::evict_lru() {
    if (entries_.empty()) return;

    std::size_t victim = 0;
    for (std::size_t i = 1; i < entries_.size(); ++i) {
        if (entries_[i].last_used < entries_[victim].last_used) {
            victim = i;
        }
    }

    PrefixCheckpoint& entry = entries_[victim];
    ++stats_.evictions;
    if (trace_enabled()) {
        std::fprintf(stderr, "PREFIX_CACHE_EVICT tokens=%u\n",
                     static_cast<unsigned>(entry.tokens.size()));
    }
    release_checkpoint(entry);
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(victim));
}

Status PrefixCache::save(
    const PagedSequenceState& sequence,
    const GdnStatePool& gdn_pool,
    const PagedKVPool& kv_pool,
    std::vector<int32_t> tokens,
    hipStream_t stream,
    bool prompt_boundary) {
    if (!enabled()) return Status::make_ok();
    if (!sequence.is_usable()) {
        return Status::invalid_state("prefix save: sequence not usable", __FILE__, __LINE__);
    }
    if (sequence.position == 0) {
        return Status::invalid_argument("prefix save: empty sequence", __FILE__, __LINE__);
    }
    if (tokens.size() != sequence.position) {
        return Status::invalid_argument(
            "prefix save: token count != sequence position", __FILE__, __LINE__);
    }
    if (tokens.size() < min_prefix_tokens_) {
        if (trace_enabled()) {
            std::fprintf(stderr, "PREFIX_CACHE_SKIP tokens=%u reason=min_prefix\n",
                         static_cast<unsigned>(tokens.size()));
        }
        return Status::make_ok();
    }
    if (cache_kv_pool_->dtype() != kv_pool.dtype()) {
        return Status::invalid_state("prefix save: KV dtype mismatch", __FILE__, __LINE__);
    }

    const uint32_t page_tokens = cache_kv_pool_->page_tokens();
    const uint32_t pages = pages_for(sequence.position, page_tokens);
    if (pages == 0 || pages > sequence.block_table.size()) {
        return Status::invalid_state(
            "prefix save: block table shorter than position", __FILE__, __LINE__);
    }
    if (pages > cache_kv_pool_->num_pages()) {
        ++stats_.skipped_too_large;
        if (trace_enabled()) {
            std::fprintf(stderr, "PREFIX_CACHE_SKIP tokens=%u pages=%u reason=too_large\n",
                         static_cast<unsigned>(tokens.size()),
                         static_cast<unsigned>(pages));
        }
        return Status::make_ok();
    }

    auto duplicate = find_exact(tokens);
    if (duplicate != entries_.end()) {
        duplicate->last_used = ++clock_;
        ++stats_.duplicate_hits;
        return Status::make_ok();
    }

    while ((cache_kv_pool_->num_free_pages() < pages || free_state_slots_.empty()) &&
           !entries_.empty()) {
        evict_lru();
    }
    if (cache_kv_pool_->num_free_pages() < pages || free_state_slots_.empty()) {
        ++stats_.skipped_too_large;
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "PREFIX_CACHE_SKIP tokens=%u pages=%u reason=no_room\n",
                         static_cast<unsigned>(tokens.size()),
                         static_cast<unsigned>(pages));
        }
        return Status::make_ok();
    }

    PrefixCheckpoint checkpoint;
    checkpoint.tokens = std::move(tokens);
    checkpoint.position = sequence.position;
    checkpoint.cache_pages.reserve(pages);
    for (uint32_t p = 0; p < pages; ++p) {
        auto page = cache_kv_pool_->allocate();
        if (!page.ok()) {
            release_checkpoint(checkpoint);
            return page.status();
        }
        checkpoint.cache_pages.push_back(page.value());
    }

    const uint32_t slot = free_state_slots_.back();
    free_state_slots_.pop_back();
    checkpoint.cache_state_slot = slot;

    Status init_st = cache_gdn_pool_->initialize(slot, stream);
    if (!init_st.ok()) {
        release_checkpoint(checkpoint);
        return init_st;
    }

    for (uint32_t p = 0; p < pages; ++p) {
        auto copy_st = cache_kv_pool_->copy_page_from(
            checkpoint.cache_pages[p], kv_pool, sequence.block_table[p], stream);
        if (!copy_st.ok()) {
            release_checkpoint(checkpoint);
            return copy_st;
        }
    }

    auto gdn_st = cache_gdn_pool_->copy_slot_from(slot, gdn_pool, sequence.slot, stream);
    if (!gdn_st.ok()) {
        release_checkpoint(checkpoint);
        return gdn_st;
    }

    const std::size_t per_page = kv_page_bytes(kv_pool) * kv_pool.num_attention_layers() * 2u
        + kv_page_scale_bytes(kv_pool) * kv_pool.num_attention_layers() * 2u;
    stats_.save_d2d_bytes +=
        per_page * pages + 2u * gdn_slot_bytes(gdn_pool);
    stats_.saved_tokens += checkpoint.position;
    ++stats_.inserts;

    checkpoint.last_used = ++clock_;
    entries_.push_back(std::move(checkpoint));

    if (trace_enabled()) {
        if (prompt_boundary) {
            std::fprintf(stderr, "PREFIX_CACHE_CHECKPOINT tokens=%u pages=%u\n",
                         static_cast<unsigned>(sequence.position),
                         static_cast<unsigned>(pages));
        } else {
            std::fprintf(stderr, "PREFIX_CACHE_INSERT tokens=%u pages=%u\n",
                         static_cast<unsigned>(sequence.position),
                         static_cast<unsigned>(pages));
        }
    }
    return Status::make_ok();
}

Status PrefixCache::restore(
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    const PrefixCheckpoint& checkpoint,
    hipStream_t stream) {
    if (!enabled()) {
        return Status::invalid_state("prefix restore: cache disabled", __FILE__, __LINE__);
    }
    if (!sequence.is_usable()) {
        return Status::invalid_state("prefix restore: sequence not usable", __FILE__, __LINE__);
    }
    if (sequence.position != 0 || !sequence.block_table.empty()) {
        return Status::invalid_state("prefix restore: sequence not empty", __FILE__, __LINE__);
    }
    if (checkpoint.position == 0) {
        return Status::invalid_argument("prefix restore: empty checkpoint", __FILE__, __LINE__);
    }
    if (checkpoint.tokens.size() != checkpoint.position) {
        return Status::invalid_argument(
            "prefix restore: checkpoint token count mismatch", __FILE__, __LINE__);
    }
    if (cache_kv_pool_->dtype() != kv_pool.dtype()) {
        return Status::invalid_state("prefix restore: KV dtype mismatch", __FILE__, __LINE__);
    }

    const uint32_t page_tokens = cache_kv_pool_->page_tokens();
    const uint32_t pages = pages_for(checkpoint.position, page_tokens);
    if (checkpoint.cache_pages.size() != pages) {
        return Status::invalid_argument(
            "prefix restore: checkpoint page count mismatch", __FILE__, __LINE__);
    }

    const ::ps::runtime::RequestHandle handle = sequence.request_handle();
    auto reserve_st = reserve_sequence_append(sequence, checkpoint.position, stream);
    if (!reserve_st.ok()) return reserve_st;
    if (sequence.block_table.size() != pages) {
        return Status::invalid_state(
            "prefix restore: allocated page count mismatch", __FILE__, __LINE__);
    }

    for (uint32_t p = 0; p < pages; ++p) {
        auto copy_st = kv_pool.copy_page_from(
            sequence.block_table[p], *cache_kv_pool_, checkpoint.cache_pages[p], stream);
        if (!copy_st.ok()) return copy_st;
    }

    auto gdn_st = gdn_pool.copy_slot_from(
        sequence.slot, *cache_gdn_pool_, checkpoint.cache_state_slot, stream);
    if (!gdn_st.ok()) return gdn_st;

    auto commit_st = commit_sequence_append(sequence, handle, checkpoint.position);
    if (!commit_st.ok()) return commit_st;

    const std::size_t per_page = kv_page_bytes(kv_pool) * kv_pool.num_attention_layers() * 2u
        + kv_page_scale_bytes(kv_pool) * kv_pool.num_attention_layers() * 2u;
    stats_.restore_d2d_bytes += per_page * pages + 2u * gdn_slot_bytes(gdn_pool);
    stats_.restored_tokens += checkpoint.position;

    if (trace_enabled()) {
        std::fprintf(stderr, "PREFIX_CACHE_RESTORE tokens=%u\n",
                     static_cast<unsigned>(checkpoint.position));
    }
    return Status::make_ok();
}

void PrefixCache::clear() noexcept {
    for (PrefixCheckpoint& entry : entries_) {
        release_checkpoint(entry);
    }
    entries_.clear();
    free_state_slots_.clear();
    if (cache_gdn_pool_) {
        for (uint32_t i = 0; i < cache_gdn_pool_->max_sequences(); ++i) {
            free_state_slots_.push_back(i);
        }
    }
}

}
}
}
