#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/runtime/prefix_cache.h>
#include <phaseshift/models/qwen35/runtime/runtime_request.h>
#include <phaseshift/models/qwen35/runtime/token_budget_scheduler.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
        std::printf("PASS: %s\n", name);
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

constexpr uint32_t kPageTokens = 4;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 4;
constexpr uint32_t kAttnLayers = 2;
constexpr uint32_t kPositions = 8;

using ps::qwen35::GDNState;
using ps::qwen35::GdnStatePool;
using ps::qwen35::GdnStatePoolLayout;
using ps::qwen35::KVCacheDType;
using ps::qwen35::PagedKVPool;
using ps::qwen35::PagedSequenceState;
using ps::qwen35::SequenceSlotPool;
using ps::qwen35::runtime::PrefixCache;
using ps::qwen35::runtime::PrefixCheckpoint;

bool is_psq(KVCacheDType dtype) {
    return dtype == KVCacheDType::PSQ4_W32 || dtype == KVCacheDType::PSQ8_W32;
}

uint32_t head_dim_for(KVCacheDType dtype) {
    return is_psq(dtype) ? 256u : kHeadDim;
}

std::size_t psq_blocks_per_head(const PagedKVPool& pool) {
    return pool.dtype() == KVCacheDType::PSQ4_W32
               ? static_cast<std::size_t>(pool.psq4_blocks_per_head())
               : static_cast<std::size_t>(pool.psq8_blocks_per_head());
}

std::size_t psq_scale_base(const PagedKVPool& pool, uint32_t layer, uint32_t page) {
    return (static_cast<std::size_t>(layer) * pool.num_pages() + page)
        * pool.page_tokens() * pool.kv_heads() * psq_blocks_per_head(pool);
}

const ps::gpu::Tensor& psq_k_scale_pool(const PagedKVPool& pool) {
    return pool.dtype() == KVCacheDType::PSQ4_W32 ? pool.k_psq4_scale_pool()
                                                  : pool.k_psq8_scale_pool();
}

const ps::gpu::Tensor& psq_v_scale_pool(const PagedKVPool& pool) {
    return pool.dtype() == KVCacheDType::PSQ4_W32 ? pool.v_psq4_scale_pool()
                                                  : pool.v_psq8_scale_pool();
}

uint16_t psq_scale_bits(uint32_t tag, uint32_t layer, uint32_t page, std::size_t i) {
    return static_cast<uint16_t>(
        (tag * 101u + layer * 13u + page * 7u + static_cast<uint32_t>(i)) & 0xFFFFu);
}

std::size_t kv_page_bytes(const PagedKVPool& pool) {
    const std::size_t elems = static_cast<std::size_t>(pool.page_tokens())
        * pool.kv_heads() * pool.head_dim();
    if (pool.dtype() == KVCacheDType::FP8_E4M3) {
        return elems * sizeof(ps::qwen35::fp8e4m3_storage_t);
    }
    if (is_psq(pool.dtype())) {
        const std::size_t blocks = static_cast<std::size_t>(pool.page_tokens())
            * pool.kv_heads() * psq_blocks_per_head(pool);
        return blocks * (pool.dtype() == KVCacheDType::PSQ4_W32
                             ? ps::qwen35::kPsq4CodeBytesPerBlock
                             : ps::qwen35::kPsq8CodeBytesPerBlock);
    }
    return elems * sizeof(ps::bf16_t);
}

std::size_t scale_page_bytes(const PagedKVPool& pool) {
    if (pool.dtype() == KVCacheDType::FP8_E4M3) {
        return static_cast<std::size_t>(pool.page_tokens()) * pool.kv_heads() * sizeof(float);
    }
    if (is_psq(pool.dtype())) {
        const std::size_t blocks = static_cast<std::size_t>(pool.page_tokens())
            * pool.kv_heads() * psq_blocks_per_head(pool);
        return blocks * (pool.dtype() == KVCacheDType::PSQ4_W32
                             ? ps::qwen35::kPsq4ScaleBytesPerBlock
                             : ps::qwen35::kPsq8ScaleBytesPerBlock);
    }
    return 0;
}

void* k_ptr(const PagedKVPool& pool, uint32_t layer, uint32_t page) {
    const auto view = pool.k_view(layer, page);
    if (pool.dtype() == KVCacheDType::FP8_E4M3) {
        return static_cast<void*>(view.data<ps::qwen35::fp8e4m3_storage_t>());
    }
    if (is_psq(pool.dtype())) {
        return static_cast<void*>(view.data<uint8_t>());
    }
    return static_cast<void*>(view.data<ps::bf16_t>());
}

void* v_ptr(const PagedKVPool& pool, uint32_t layer, uint32_t page) {
    const auto view = pool.v_view(layer, page);
    if (pool.dtype() == KVCacheDType::FP8_E4M3) {
        return static_cast<void*>(view.data<ps::qwen35::fp8e4m3_storage_t>());
    }
    if (is_psq(pool.dtype())) {
        return static_cast<void*>(view.data<uint8_t>());
    }
    return static_cast<void*>(view.data<ps::bf16_t>());
}

uint8_t kv_byte(uint32_t tag, uint32_t layer, uint32_t page, std::size_t i, bool is_v) {
    return static_cast<uint8_t>(
        (tag * 31u + layer * 97u + page * 13u + static_cast<uint32_t>(i) * (is_v ? 3u : 1u)) & 0xFFu);
}

float scale_value(uint32_t tag, uint32_t layer, uint32_t page, std::size_t i) {
    return static_cast<float>(tag) * 0.5f + static_cast<float>(layer)
        + static_cast<float>(page) * 0.25f + static_cast<float>(i) * 0.125f;
}

bool seed_kv(PagedKVPool& pool, const PagedSequenceState& seq, uint32_t tag) {
    const std::size_t kbytes = kv_page_bytes(pool);
    const std::size_t sbytes = scale_page_bytes(pool);
    for (uint32_t layer = 0; layer < kAttnLayers; ++layer) {
        for (uint32_t p = 0; p < seq.block_table.size(); ++p) {
            std::vector<uint8_t> kbuf(kbytes);
            std::vector<uint8_t> vbuf(kbytes);
            for (std::size_t i = 0; i < kbytes; ++i) {
                kbuf[i] = kv_byte(tag, layer, p, i, false);
                vbuf[i] = kv_byte(tag, layer, p, i, true);
            }
            if (hipMemcpy(k_ptr(pool, layer, seq.block_table[p]), kbuf.data(), kbytes,
                          hipMemcpyHostToDevice) != hipSuccess) return false;
            if (hipMemcpy(v_ptr(pool, layer, seq.block_table[p]), vbuf.data(), kbytes,
                          hipMemcpyHostToDevice) != hipSuccess) return false;
            if (sbytes > 0) {
                if (is_psq(pool.dtype())) {
                    const std::size_t sbase =
                        psq_scale_base(pool, layer, seq.block_table[p]);
                    const std::size_t elems = sbytes / sizeof(uint16_t);
                    std::vector<uint16_t> ks(elems);
                    std::vector<uint16_t> vs(elems);
                    for (std::size_t i = 0; i < elems; ++i) {
                        ks[i] = psq_scale_bits(tag, layer, p, i);
                        vs[i] = psq_scale_bits(tag + 1u, layer, p, i);
                    }
                    if (hipMemcpy(psq_k_scale_pool(pool).data<ps::bf16_t>() + sbase,
                                  ks.data(), sbytes,
                                  hipMemcpyHostToDevice) != hipSuccess) return false;
                    if (hipMemcpy(psq_v_scale_pool(pool).data<ps::bf16_t>() + sbase,
                                  vs.data(), sbytes,
                                  hipMemcpyHostToDevice) != hipSuccess) return false;
                    continue;
                }
                const std::size_t base =
                    (static_cast<std::size_t>(layer) * pool.num_pages() + seq.block_table[p])
                    * pool.page_tokens() * pool.kv_heads();
                std::vector<float> ks(sbytes / sizeof(float));
                std::vector<float> vs(sbytes / sizeof(float));
                for (std::size_t i = 0; i < ks.size(); ++i) {
                    ks[i] = scale_value(tag, layer, p, i);
                    vs[i] = scale_value(tag + 1u, layer, p, i);
                }
                if (hipMemcpy(pool.k_scale_pool().data<float>() + base, ks.data(), sbytes,
                              hipMemcpyHostToDevice) != hipSuccess) return false;
                if (hipMemcpy(pool.v_scale_pool().data<float>() + base, vs.data(), sbytes,
                              hipMemcpyHostToDevice) != hipSuccess) return false;
            }
        }
    }
    return true;
}

bool verify_kv(PagedKVPool& pool, const PagedSequenceState& seq, uint32_t tag) {
    const std::size_t kbytes = kv_page_bytes(pool);
    const std::size_t sbytes = scale_page_bytes(pool);
    for (uint32_t layer = 0; layer < kAttnLayers; ++layer) {
        for (uint32_t p = 0; p < seq.block_table.size(); ++p) {
            std::vector<uint8_t> kbuf(kbytes);
            std::vector<uint8_t> vbuf(kbytes);
            if (hipMemcpy(kbuf.data(), k_ptr(pool, layer, seq.block_table[p]), kbytes,
                          hipMemcpyDeviceToHost) != hipSuccess) return false;
            if (hipMemcpy(vbuf.data(), v_ptr(pool, layer, seq.block_table[p]), kbytes,
                          hipMemcpyDeviceToHost) != hipSuccess) return false;
            for (std::size_t i = 0; i < kbytes; ++i) {
                if (kbuf[i] != kv_byte(tag, layer, p, i, false)) return false;
                if (vbuf[i] != kv_byte(tag, layer, p, i, true)) return false;
            }
            if (sbytes > 0) {
                if (is_psq(pool.dtype())) {
                    const std::size_t sbase =
                        psq_scale_base(pool, layer, seq.block_table[p]);
                    const std::size_t elems = sbytes / sizeof(uint16_t);
                    std::vector<uint16_t> ks(elems, 0);
                    std::vector<uint16_t> vs(elems, 0);
                    if (hipMemcpy(ks.data(),
                                  psq_k_scale_pool(pool).data<ps::bf16_t>() + sbase,
                                  sbytes, hipMemcpyDeviceToHost) != hipSuccess) return false;
                    if (hipMemcpy(vs.data(),
                                  psq_v_scale_pool(pool).data<ps::bf16_t>() + sbase,
                                  sbytes, hipMemcpyDeviceToHost) != hipSuccess) return false;
                    for (std::size_t i = 0; i < elems; ++i) {
                        if (ks[i] != psq_scale_bits(tag, layer, p, i)) return false;
                        if (vs[i] != psq_scale_bits(tag + 1u, layer, p, i)) return false;
                    }
                    continue;
                }
                const std::size_t base =
                    (static_cast<std::size_t>(layer) * pool.num_pages() + seq.block_table[p])
                    * pool.page_tokens() * pool.kv_heads();
                std::vector<float> ks(sbytes / sizeof(float));
                std::vector<float> vs(sbytes / sizeof(float));
                if (hipMemcpy(ks.data(), pool.k_scale_pool().data<float>() + base, sbytes,
                              hipMemcpyDeviceToHost) != hipSuccess) return false;
                if (hipMemcpy(vs.data(), pool.v_scale_pool().data<float>() + base, sbytes,
                              hipMemcpyDeviceToHost) != hipSuccess) return false;
                for (std::size_t i = 0; i < ks.size(); ++i) {
                    if (ks[i] != scale_value(tag, layer, p, i)) return false;
                    if (vs[i] != scale_value(tag + 1u, layer, p, i)) return false;
                }
            }
        }
    }
    return true;
}

std::size_t gdn_slot_bytes(const GdnStatePool& pool) {
    const std::size_t conv = static_cast<std::size_t>(pool.num_gdn_states())
        * pool.conv_history() * pool.conv_history_stride() * sizeof(ps::bf16_t);
    const std::size_t rec = static_cast<std::size_t>(pool.num_gdn_states())
        * pool.num_v_heads() * pool.head_k() * pool.head_v() * sizeof(float);
    return conv + rec;
}

bool seed_gdn(GdnStatePool& pool, const PagedSequenceState& seq, uint32_t tag) {
    const GDNState st = pool.gdn_state(seq.slot, 0);
    if (st.conv_state.ndim() == 0 || st.recurrent_state.ndim() == 0) return false;
    const std::size_t conv_bytes = static_cast<std::size_t>(pool.num_gdn_states())
        * pool.conv_history() * pool.conv_history_stride() * sizeof(ps::bf16_t);
    const std::size_t rec_bytes = static_cast<std::size_t>(pool.num_gdn_states())
        * pool.num_v_heads() * pool.head_k() * pool.head_v() * sizeof(float);
    std::vector<uint8_t> conv(conv_bytes);
    for (std::size_t i = 0; i < conv.size(); ++i) {
        conv[i] = static_cast<uint8_t>((tag * 17u + i * 7u + 11u) & 0xFFu);
    }
    std::vector<float> rec(rec_bytes / sizeof(float));
    for (std::size_t i = 0; i < rec.size(); ++i) {
        rec[i] = static_cast<float>(tag) + static_cast<float>(i) * 0.25f;
    }
    if (hipMemcpy(st.conv_state.data<ps::bf16_t>(), conv.data(), conv_bytes,
                  hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(st.recurrent_state.data<float>(), rec.data(), rec_bytes,
                  hipMemcpyHostToDevice) != hipSuccess) return false;
    return true;
}

bool verify_gdn(GdnStatePool& pool, const PagedSequenceState& seq, uint32_t tag) {
    const GDNState st = pool.gdn_state(seq.slot, 0);
    const std::size_t conv_bytes = static_cast<std::size_t>(pool.num_gdn_states())
        * pool.conv_history() * pool.conv_history_stride() * sizeof(ps::bf16_t);
    const std::size_t rec_bytes = static_cast<std::size_t>(pool.num_gdn_states())
        * pool.num_v_heads() * pool.head_k() * pool.head_v() * sizeof(float);
    std::vector<uint8_t> conv(conv_bytes);
    std::vector<float> rec(rec_bytes / sizeof(float));
    if (hipMemcpy(conv.data(), st.conv_state.data<ps::bf16_t>(), conv_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(rec.data(), st.recurrent_state.data<float>(), rec_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) return false;
    for (std::size_t i = 0; i < conv.size(); ++i) {
        if (conv[i] != static_cast<uint8_t>((tag * 17u + i * 7u + 11u) & 0xFFu)) return false;
    }
    for (std::size_t i = 0; i < rec.size(); ++i) {
        if (rec[i] != static_cast<float>(tag) + static_cast<float>(i) * 0.25f) return false;
    }
    return true;
}

struct Fixture {
    std::optional<ps::gpu::GpuArena> arena;
    hipStream_t stream = nullptr;
    std::optional<SequenceSlotPool> seq_pool;
    std::optional<GdnStatePool> gdn_pool;
    std::optional<PagedKVPool> kv_pool;
    std::optional<PrefixCache> cache;
    uint32_t cache_pages_total = 0;
};

bool make_fixture(Fixture& fx, KVCacheDType dtype, uint32_t cache_tokens,
                  uint32_t max_entries, uint32_t max_sequences = 2,
                  uint32_t active_pages = 16) {
    auto arena_result = ps::gpu::GpuArena::create(0, 128u * 1024u * 1024u);
    if (!arena_result.ok()) return false;
    fx.arena.emplace(arena_result.release());
    if (hipStreamCreate(&fx.stream) != hipSuccess) return false;

    const auto layout = GdnStatePoolLayout::make(2, 8, 3, 2, 4, 4);
    auto seq = SequenceSlotPool::create(*fx.arena, max_sequences, 8);
    auto gdn = GdnStatePool::create(*fx.arena, max_sequences, layout);
    auto kv = PagedKVPool::create(
        *fx.arena, active_pages, kPageTokens, kAttnLayers, kKvHeads, head_dim_for(dtype),
        dtype);
    if (!seq.ok() || !gdn.ok() || !kv.ok()) return false;
    fx.seq_pool.emplace(seq.release());
    fx.gdn_pool.emplace(gdn.release());
    fx.kv_pool.emplace(kv.release());

    if (cache_tokens > 0) {
        auto cache_result = PrefixCache::create(
            *fx.arena, *fx.kv_pool, *fx.gdn_pool, cache_tokens, max_entries, 1u);
        if (!cache_result.ok()) return false;
        fx.cache.emplace(cache_result.release());
        fx.cache_pages_total = fx.cache->cache_pages();
    }
    return true;
}

void teardown(Fixture& fx) {
    if (fx.cache.has_value()) fx.cache->clear();
    fx.cache.reset();
    fx.kv_pool.reset();
    fx.gdn_pool.reset();
    fx.seq_pool.reset();
    if (fx.stream != nullptr) {
        hipStreamDestroy(fx.stream);
        fx.stream = nullptr;
    }
    if (fx.arena.has_value()) fx.arena->shutdown();
    fx.arena.reset();
}

bool create_sequence(Fixture& fx, PagedSequenceState& out, uint32_t positions,
                     uint32_t max_len = 16) {
    auto result = ps::qwen35::create_paged_sequence_state(
        *fx.seq_pool, *fx.gdn_pool, *fx.kv_pool, max_len, fx.stream);
    if (!result.ok()) return false;
    ps::qwen35::runtime_install_sequence_state(out, result.release());
    if (positions == 0) return true;
    if (!ps::qwen35::reserve_sequence_append(out, positions, fx.stream).ok()) return false;
    if (!ps::qwen35::commit_sequence_append(out, out.request_handle(), positions).ok()) {
        return false;
    }
    return true;
}

void test_roundtrip(KVCacheDType dtype, const char* label) {
    Fixture fx;
    if (!make_fixture(fx, dtype, 32, 4)) {
        check(false, "fixture create");
        return;
    }

    PagedSequenceState seq;
    check(create_sequence(fx, seq, kPositions), label);
    check(seed_kv(*fx.kv_pool, seq, 3u), "seed kv");
    check(verify_kv(*fx.kv_pool, seq, 3u), "seed kv self-verify");
    check(seed_gdn(*fx.gdn_pool, seq, 3u), "seed gdn");

    std::vector<int32_t> tokens(kPositions);
    for (uint32_t i = 0; i < kPositions; ++i) tokens[i] = static_cast<int32_t>(100 + i);
    auto save_st = fx.cache->save(seq, *fx.gdn_pool, *fx.kv_pool, tokens, fx.stream);
    check(save_st.ok(), "cache save");
    check(fx.cache->size() == 1, "cache entry added");
    check(ps::qwen35::release_paged_sequence_state(seq, fx.stream).ok(), "release source");

    const PrefixCheckpoint* checkpoint = fx.cache->find_longest(tokens);
    check(checkpoint != nullptr && checkpoint->position == kPositions,
          "longest match after save");

    PagedSequenceState restored;
    check(create_sequence(fx, restored, 0), "create restore sequence");
    if (checkpoint != nullptr) {
        auto restore_st = fx.cache->restore(
            restored, *fx.gdn_pool, *fx.kv_pool, *checkpoint, fx.stream);
        check(restore_st.ok(), "cache restore");
        check(restored.position == kPositions, "restored position");
        check(restored.block_table.size() == (kPositions / kPageTokens),
              "restored block table");
        check(verify_kv(*fx.kv_pool, restored, 3u), "kv round-trip");
        check(verify_gdn(*fx.gdn_pool, restored, 3u), "gdn round-trip");
    }

    check(ps::qwen35::release_paged_sequence_state(restored, fx.stream).ok(), "release restored");
    teardown(fx);
}

void test_partial_page() {
    Fixture fx;
    if (!make_fixture(fx, KVCacheDType::BF16, 32, 4)) {
        check(false, "partial fixture");
        return;
    }
    const uint32_t partial = kPageTokens + 1;
    PagedSequenceState seq;
    check(create_sequence(fx, seq, partial), "partial sequence");
    check(seed_kv(*fx.kv_pool, seq, 7u), "partial seed kv");
    check(seed_gdn(*fx.gdn_pool, seq, 7u), "partial seed gdn");

    std::vector<int32_t> tokens(partial);
    for (uint32_t i = 0; i < partial; ++i) tokens[i] = static_cast<int32_t>(200 + i);
    check(fx.cache->save(seq, *fx.gdn_pool, *fx.kv_pool, tokens, fx.stream).ok(),
          "partial save");
    check(ps::qwen35::release_paged_sequence_state(seq, fx.stream).ok(), "partial release");

    const PrefixCheckpoint* checkpoint = fx.cache->find_longest(tokens);
    PagedSequenceState restored;
    check(create_sequence(fx, restored, 0), "partial restore sequence");
    if (checkpoint != nullptr) {
        check(fx.cache->restore(restored, *fx.gdn_pool, *fx.kv_pool, *checkpoint, fx.stream).ok(),
              "partial restore");
        check(restored.position == partial, "partial position exact");
        check(verify_kv(*fx.kv_pool, restored, 7u), "partial kv bytes");

        const uint32_t append = 3;
        check(ps::qwen35::reserve_sequence_append(restored, append, fx.stream).ok(),
              "partial append reserve");
        check(ps::qwen35::commit_sequence_append(
                  restored, restored.request_handle(), append).ok(),
              "partial append commit");
        check(restored.position == partial + append, "partial append position");
    }
    check(ps::qwen35::release_paged_sequence_state(restored, fx.stream).ok(),
          "partial release restored");

    check(fx.cache->size() == 1, "cache page immutable count");
    teardown(fx);
}

bool save_tokens(Fixture& fx, const std::vector<int32_t>& tokens, uint32_t tag) {
    PagedSequenceState seq;
    if (!create_sequence(fx, seq, static_cast<uint32_t>(tokens.size()))) return false;
    if (!seed_kv(*fx.kv_pool, seq, tag)) return false;
    if (!seed_gdn(*fx.gdn_pool, seq, tag)) return false;
    auto st = fx.cache->save(seq, *fx.gdn_pool, *fx.kv_pool, tokens, fx.stream);
    (void)ps::qwen35::release_paged_sequence_state(seq, fx.stream);
    return st.ok();
}

void test_lru_and_duplicate() {
    Fixture fx;
    if (!make_fixture(fx, KVCacheDType::BF16, 32, 2)) {
        check(false, "lru fixture");
        return;
    }

    const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int32_t> b = {9, 10, 11, 12, 13, 14, 15, 16};
    const std::vector<int32_t> c = {21, 22, 23, 24, 25, 26, 27, 28};

    check(save_tokens(fx, a, 1u), "lru insert a");
    check(save_tokens(fx, b, 2u), "lru insert b");
    check(fx.cache->size() == 2, "lru size 2");
    check(fx.cache->find_longest(a) != nullptr, "lru hit a");
    check(save_tokens(fx, c, 3u), "lru insert c");
    check(fx.cache->find_longest(b) == nullptr, "lru evicted b");
    check(fx.cache->find_longest(a) != nullptr, "lru kept a");
    check(fx.cache->find_longest(c) != nullptr, "lru kept c");
    check(fx.cache->stats().evictions == 1, "lru eviction count");

    const std::size_t before = fx.cache->size();
    check(save_tokens(fx, a, 4u), "duplicate insert");
    check(fx.cache->size() == before, "duplicate no growth");
    check(fx.cache->stats().duplicate_hits == 1, "duplicate count");

    fx.cache->clear();
    check(fx.cache->cache_free_pages() == fx.cache_pages_total, "clear releases pages");
    check(fx.cache->cache_free_slots() == fx.cache->max_entries(), "clear releases slots");
    teardown(fx);
}

void test_too_large_and_eviction_release() {
    Fixture fx;
    if (!make_fixture(fx, KVCacheDType::BF16, kPageTokens, 2)) {
        check(false, "too-large fixture");
        return;
    }
    const std::vector<int32_t> big = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint32_t free_before = fx.cache->cache_free_pages();
    check(save_tokens(fx, big, 1u), "too large save status");
    check(fx.cache->size() == 0, "too large not cached");
    check(fx.cache->stats().skipped_too_large == 1, "too large counter");
    check(fx.cache->cache_free_pages() == free_before, "too large no page leak");

    const std::vector<int32_t> small = {1, 2, 3, 4};
    check(save_tokens(fx, small, 2u), "small insert");
    const std::vector<int32_t> small2 = {5, 6, 7, 8};
    check(save_tokens(fx, small2, 3u), "small insert 2");
    check(fx.cache->stats().evictions == 1, "eviction happened");
    check(fx.cache->size() == 1, "one entry remains");
    check(fx.cache->cache_free_pages() == 0, "one page held by the remaining entry");
    fx.cache->clear();
    check(fx.cache->cache_free_pages() == fx.cache_pages_total, "eviction released pages");
    check(fx.cache->cache_free_slots() == fx.cache->max_entries(), "eviction released slots");
    teardown(fx);
}

void test_scheduler_boundary() {
    Fixture fx;
    if (!make_fixture(fx, KVCacheDType::BF16, 32, 4)) {
        check(false, "scheduler fixture");
        return;
    }

    ps::qwen35::runtime::RuntimeRequest request;
    check(create_sequence(fx, request.sequence, 0, 32), "scheduler sequence");
    request.state = ps::qwen35::runtime::RequestState::Active;
    request.input_tokens.assign(28, 7);
    request.max_new_tokens = 4;
    request.prefix_cache_checkpoint_position = 12;

    std::array<const ps::qwen35::runtime::RuntimeRequest*, 1> active{&request};
    auto plan = ps::qwen35::runtime::schedule_requests(
        active, 64, 4, kPageTokens, nullptr);
    check(plan.ok(), "scheduler plan ok");
    if (plan.ok()) {
        const auto& scheduled = plan.value().scheduled_requests;
        check(scheduled.size() == 1, "scheduler one prefill");
        if (scheduled.size() == 1) {
            check(scheduled[0].num_tokens == 12, "scheduler split at checkpoint");
            check(scheduled[0].prefix_tokens == 0, "scheduler boundary prefix");
        }
    }

    request.prefix_cache_checkpoint_saved = true;
    auto plan_saved = ps::qwen35::runtime::schedule_requests(
        active, 64, 4, kPageTokens, nullptr);
    check(plan_saved.ok(), "scheduler saved plan ok");
    if (plan_saved.ok()) {
        const auto& scheduled = plan_saved.value().scheduled_requests;
        check(scheduled.size() == 1, "scheduler saved one prefill");
        if (scheduled.size() == 1) {
            check(scheduled[0].num_tokens == 28, "scheduler no split after save");
        }
    }

    check(ps::qwen35::release_paged_sequence_state(request.sequence, fx.stream).ok(),
          "scheduler release");
    teardown(fx);
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    test_roundtrip(KVCacheDType::BF16, "bf16 roundtrip");
    test_roundtrip(KVCacheDType::FP8_E4M3, "fp8 roundtrip");
    test_roundtrip(KVCacheDType::PSQ4_W32, "psq4 roundtrip");
    test_roundtrip(KVCacheDType::PSQ8_W32, "psq8 roundtrip");
    test_partial_page();
    test_lru_and_duplicate();
    test_too_large_and_eviction_release();
    test_scheduler_boundary();

    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
