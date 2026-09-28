#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/runtime/spec_decode.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

using ps::qwen35::GdnStatePool;
using ps::qwen35::GdnStatePoolLayout;
using ps::qwen35::KVCacheDType;
using ps::qwen35::PagedKVPool;
using ps::qwen35::PagedSequenceState;
using ps::qwen35::SequenceSlotPool;
using ps::qwen35::runtime::MtpKvConfig;
using ps::qwen35::runtime::MtpKvState;
using ps::qwen35::runtime::SpecPhase;
using ps::qwen35::runtime::SpecTransaction;
using ps::qwen35::runtime::SpecVerifyResult;

constexpr uint32_t kPageTokens = 4;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 4;
constexpr uint32_t kKvPages = 256;
constexpr uint32_t kMaxSeqLen = 512;
constexpr uint32_t kMaxSeq = 2;

uint32_t blocks_for(uint32_t tokens) {
    return (tokens + kPageTokens - 1u) / kPageTokens;
}

struct Scene {
    PagedKVPool* kv_pool = nullptr;
    SequenceSlotPool* seq_pool = nullptr;
    GdnStatePool* gdn_pool = nullptr;
    hipStream_t stream = nullptr;

    bool reset(PagedSequenceState& seq) {
        if (seq.is_allocated()) {
            auto st = ps::qwen35::release_paged_sequence_state(seq, stream);
            if (!st.ok()) return false;
        }
        auto created = ps::qwen35::create_paged_sequence_state(
            *seq_pool, *gdn_pool, *kv_pool, kMaxSeqLen, stream);
        if (!created.ok()) {
            std::printf("FAIL: create_paged_sequence_state: %s\n",
                        created.status().message().c_str());
            return false;
        }
        ps::qwen35::runtime_install_sequence_state(seq, created.release());
        return true;
    }

    bool advance(PagedSequenceState& seq, uint32_t tokens) {
        if (tokens == 0u) return true;
        auto r = ps::qwen35::reserve_sequence_append(seq, tokens, stream);
        if (!r.ok()) {
            std::printf("FAIL: reserve %u: %s\n", tokens, r.message().c_str());
            return false;
        }
        auto c = ps::qwen35::commit_sequence_append(seq, seq.request_handle(), tokens);
        if (!c.ok()) {
            std::printf("FAIL: commit %u: %s\n", tokens, c.message().c_str());
            return false;
        }
        return true;
    }
};

struct Scenario {
    std::vector<int32_t> drafts;
    std::vector<int32_t> target;
    uint32_t accept = 0;
    uint32_t draft_k = 4;
};

void build_scenario(Scenario& s, uint32_t accept, uint32_t draft_k) {
    s.accept = accept;
    s.draft_k = draft_k;
    s.drafts.resize(draft_k);
    for (uint32_t k = 0; k < draft_k; ++k) s.drafts[k] = static_cast<int32_t>(1000u + k);
    s.target.assign(draft_k + 1u, 0);
    for (uint32_t i = 0; i < accept; ++i) s.target[i] = s.drafts[i];
    if (accept < draft_k) s.target[accept] = static_cast<int32_t>(2000u + accept);
    s.target[draft_k] = static_cast<int32_t>(3000u + accept);
}

bool run_forced(
    Scene& scene,
    PagedSequenceState& seq,
    MtpKvState& mtp,
    SpecTransaction& txn,
    uint32_t start_position,
    uint32_t accept,
    uint32_t draft_k,
    const char* label) {

    if (!scene.reset(seq)) {
        check(false, std::string(label) + ": reset");
        return false;
    }
    if (!scene.advance(seq, start_position)) {
        check(false, std::string(label) + ": advance to start");
        return false;
    }

    auto begin = ps::qwen35::runtime::spec_transaction_begin(txn, mtp, seq);
    check(begin.ok(), std::string(label) + ": begin");
    check(txn.phase == SpecPhase::DRAFTING, std::string(label) + ": phase DRAFTING");
    check(!ps::qwen35::runtime::spec_transaction_begin(txn, mtp, seq).ok(),
          std::string(label) + ": nested begin rejected");

    const uint32_t mtp_before = mtp.logical_length;
    const uint32_t pos_before = seq.position;
    const uint32_t blocks_before = static_cast<uint32_t>(seq.block_table.size());
    check(blocks_before == blocks_for(start_position), std::string(label) + ": start blocks");

    check(ps::qwen35::runtime::mtp_kv_reserve(mtp, draft_k).ok(), std::string(label) + ": mtp append");
    check(scene.advance(seq, draft_k + 1u), std::string(label) + ": target tentative append");
    check(seq.position == pos_before + draft_k + 1u, std::string(label) + ": tentative position");

    Scenario scenario;
    build_scenario(scenario, accept, draft_k);
    SpecVerifyResult result;
    check(ps::qwen35::runtime::spec_greedy_accept(
              scenario.drafts.data(), draft_k, scenario.target.data(), txn.bonus_token_enabled, result)
              .ok(),
          std::string(label) + ": accept");
    check(result.num_accepted_drafts == accept, std::string(label) + ": accepted count");
    check(ps::qwen35::runtime::spec_transaction_begin_verify(txn).ok(), std::string(label) + ": begin_verify");
    check(!ps::qwen35::runtime::spec_transaction_begin_verify(txn).ok(),
          std::string(label) + ": double verify rejected");

    const uint32_t expected_mtp = mtp_before + ((accept + 1u < draft_k) ? accept + 1u : draft_k);
    const uint32_t expected_pos = pos_before + accept + 1u;
    const uint32_t expected_blocks = blocks_for(expected_pos);

    auto commit = ps::qwen35::runtime::spec_transaction_commit(txn, result, mtp, seq, scene.stream);
    check(commit.ok(), std::string(label) + ": commit");
    check(txn.phase == SpecPhase::IDLE, std::string(label) + ": phase IDLE after commit");
    check(mtp.logical_length == expected_mtp, std::string(label) + ": mtp length");
    check(seq.position == expected_pos, std::string(label) + ": target position");
    check(seq.block_table.size() == expected_blocks, std::string(label) + ": block count");
    check(scene.kv_pool->num_used_pages() == expected_blocks, std::string(label) + ": page accounting");
    check(!ps::qwen35::runtime::spec_transaction_commit(txn, result, mtp, seq, scene.stream).ok(),
          std::string(label) + ": double commit rejected");

    check(ps::qwen35::runtime::mtp_kv_physical_slot(mtp, mtp.logical_length - 1u).ok(),
          std::string(label) + ": tail slot visible");
    check(!ps::qwen35::runtime::mtp_kv_physical_slot(mtp, mtp.logical_length).ok(),
          std::string(label) + ": rejected slot invisible");
    return true;
}

void rollback_case(Scene& scene, PagedSequenceState& seq, MtpKvState& mtp, SpecTransaction& txn) {
    const char* label = "rollback";
    if (!scene.reset(seq)) return;
    if (!scene.advance(seq, 16u)) return;
    check(ps::qwen35::runtime::spec_transaction_begin(txn, mtp, seq).ok(), std::string(label) + ": begin");
    const uint32_t mtp_before = mtp.logical_length;
    const std::size_t blocks_before = seq.block_table.size();
    const uint32_t used_before = scene.kv_pool->num_used_pages();

    check(ps::qwen35::runtime::mtp_kv_reserve(mtp, 4u).ok(), std::string(label) + ": mtp append");
    check(scene.advance(seq, 5u), std::string(label) + ": target tentative");
    check(ps::qwen35::runtime::spec_transaction_begin_verify(txn).ok(), std::string(label) + ": begin_verify");

    auto rollback = ps::qwen35::runtime::spec_transaction_rollback(txn, mtp, seq, scene.stream);
    check(rollback.ok(), std::string(label) + ": rollback");
    check(txn.phase == SpecPhase::IDLE, std::string(label) + ": phase IDLE");
    check(mtp.logical_length == mtp_before, std::string(label) + ": mtp restored");
    check(seq.position == 16u, std::string(label) + ": position restored");
    check(seq.block_table.size() == blocks_before, std::string(label) + ": blocks restored");
    check(scene.kv_pool->num_used_pages() == used_before, std::string(label) + ": pages released");

    check(!ps::qwen35::runtime::spec_transaction_rollback(txn, mtp, seq, scene.stream).ok(),
          std::string(label) + ": rollback in IDLE rejected");
    check(!ps::qwen35::runtime::spec_transaction_abort(txn, mtp, seq, scene.stream).ok(),
          std::string(label) + ": abort in IDLE rejected");

    check(ps::qwen35::rollback_sequence_append(seq, static_cast<uint32_t>(blocks_before), scene.stream).ok(),
          std::string(label) + ": repeated rollback is no-op");
    check(scene.kv_pool->num_used_pages() == used_before, std::string(label) + ": no double free");

    check(ps::qwen35::runtime::mtp_kv_reserve(mtp, 1u).ok(), std::string(label) + ": re-append after rollback");
    check(mtp.logical_length == mtp_before + 1u, std::string(label) + ": re-append length");
    mtp.logical_length = mtp_before;
    mtp.step_index = mtp_before;
}

void gdn_case(Scene& scene, PagedSequenceState& seq) {
    const char* label = "gdn";
    if (!scene.reset(seq)) return;
    const std::size_t conv_bytes = ps::qwen35::runtime::spec_gdn_conv_bytes(*scene.gdn_pool);
    const std::size_t rec_bytes = ps::qwen35::runtime::spec_gdn_recurrent_bytes(*scene.gdn_pool);
    check(conv_bytes > 0u && rec_bytes > 0u, std::string(label) + ": sizes positive");

    void* snap_conv = nullptr;
    void* snap_rec = nullptr;
    void* work_conv = nullptr;
    void* work_rec = nullptr;
    if (hipMalloc(&snap_conv, conv_bytes) != hipSuccess ||
        hipMalloc(&snap_rec, rec_bytes) != hipSuccess ||
        hipMalloc(&work_conv, conv_bytes) != hipSuccess ||
        hipMalloc(&work_rec, rec_bytes) != hipSuccess) {
        std::printf("FAIL: %s: hipMalloc\n", label);
        return;
    }

    check(ps::qwen35::runtime::spec_gdn_snapshot(
              *scene.gdn_pool, seq.slot, snap_conv, snap_rec, conv_bytes, rec_bytes, scene.stream)
              .ok(),
          std::string(label) + ": snapshot");

    const ps::qwen35::GDNState st = scene.gdn_pool->gdn_state(seq.slot, 0);
    ps::gpu::discard_cleanup_result(
        hipMemsetAsync(st.conv_state.data<void>(), 0x5A, conv_bytes, scene.stream));
    ps::gpu::discard_cleanup_result(
        hipMemsetAsync(st.recurrent_state.data<void>(), 0x00, rec_bytes, scene.stream));
    check(hipStreamSynchronize(scene.stream) == hipSuccess, std::string(label) + ": sync after overwrite");

    check(ps::qwen35::runtime::spec_gdn_restore(
              *scene.gdn_pool, seq.slot, snap_conv, snap_rec, conv_bytes, rec_bytes, scene.stream)
              .ok(),
          std::string(label) + ": restore");
    check(hipStreamSynchronize(scene.stream) == hipSuccess, std::string(label) + ": sync after restore");

    ps::gpu::discard_cleanup_result(
        hipMemcpyAsync(work_conv, st.conv_state.data<void>(), conv_bytes, hipMemcpyDeviceToHost, scene.stream));
    ps::gpu::discard_cleanup_result(
        hipMemcpyAsync(work_rec, st.recurrent_state.data<void>(), rec_bytes, hipMemcpyDeviceToHost, scene.stream));
    check(hipStreamSynchronize(scene.stream) == hipSuccess, std::string(label) + ": sync after readback");

    std::vector<uint8_t> conv_a(conv_bytes);
    std::vector<uint8_t> conv_b(conv_bytes);
    std::vector<uint8_t> rec_a(rec_bytes);
    std::vector<uint8_t> rec_b(rec_bytes);
    ps::gpu::discard_cleanup_result(hipMemcpy(conv_a.data(), snap_conv, conv_bytes, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(conv_b.data(), work_conv, conv_bytes, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(rec_a.data(), snap_rec, rec_bytes, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(rec_b.data(), work_rec, rec_bytes, hipMemcpyDeviceToHost));
    check(std::memcmp(conv_a.data(), conv_b.data(), conv_bytes) == 0, std::string(label) + ": conv restored");
    check(std::memcmp(rec_a.data(), rec_b.data(), rec_bytes) == 0, std::string(label) + ": recurrent restored");

    check(!ps::qwen35::runtime::spec_gdn_snapshot(
              *scene.gdn_pool, seq.slot, snap_conv, snap_rec, conv_bytes - 1u, rec_bytes, scene.stream)
              .ok(),
          std::string(label) + ": size mismatch rejected");

    ps::gpu::discard_cleanup_result(hipFree(snap_conv));
    ps::gpu::discard_cleanup_result(hipFree(snap_rec));
    ps::gpu::discard_cleanup_result(hipFree(work_conv));
    ps::gpu::discard_cleanup_result(hipFree(work_rec));
}

void repeated_case(Scene& scene, PagedSequenceState& seq, MtpKvState& mtp, SpecTransaction& txn) {
    const char* label = "repeated";
    if (!scene.reset(seq)) return;
    uint32_t oracle_pos = 0u;
    uint32_t oracle_mtp = 0u;
    const uint32_t pattern[7] = {4u, 0u, 2u, 4u, 1u, 3u, 0u};
    bool ok = true;
    for (uint32_t i = 0; i < 100u && ok; ++i) {
        const uint32_t accept = pattern[i % 7u];
        Scenario scenario;
        build_scenario(scenario, accept, 4u);
        if (!ps::qwen35::runtime::spec_transaction_begin(txn, mtp, seq).ok()) { ok = false; break; }
        if (!ps::qwen35::runtime::mtp_kv_reserve(mtp, 4u).ok()) { ok = false; break; }
        if (!scene.advance(seq, 5u)) { ok = false; break; }
        SpecVerifyResult result;
        if (!ps::qwen35::runtime::spec_greedy_accept(
                 scenario.drafts.data(), 4u, scenario.target.data(), true, result).ok()) {
            ok = false;
            break;
        }
        if (!ps::qwen35::runtime::spec_transaction_begin_verify(txn).ok()) { ok = false; break; }
        if (!ps::qwen35::runtime::spec_transaction_commit(txn, result, mtp, seq, scene.stream).ok()) {
            ok = false;
            break;
        }
        oracle_pos += accept + 1u;
        oracle_mtp += (accept + 1u < 4u) ? accept + 1u : 4u;
        if (seq.position != oracle_pos || mtp.logical_length != oracle_mtp) { ok = false; break; }
        if (scene.kv_pool->num_used_pages() != blocks_for(oracle_pos)) { ok = false; break; }
    }
    check(ok, std::string(label) + ": 100 iterations consistent");
    check(txn.phase == SpecPhase::IDLE, std::string(label) + ": ends IDLE");
}

void isolation_case(Scene& scene, const MtpKvConfig& mtp_cfg, ps::gpu::GpuArena& arena) {
    const char* label = "isolation";
    PagedSequenceState seq_a;
    PagedSequenceState seq_b;
    auto mtp_a_result = ps::qwen35::runtime::create_mtp_kv_state(arena, mtp_cfg, 0u, scene.stream);
    auto mtp_b_result = ps::qwen35::runtime::create_mtp_kv_state(arena, mtp_cfg, 1u, scene.stream);
    check(mtp_a_result.ok() && mtp_b_result.ok(), std::string(label) + ": two mtp states");
    if (!mtp_a_result.ok() || !mtp_b_result.ok()) return;
    MtpKvState mtp_a = mtp_a_result.release();
    MtpKvState mtp_b = mtp_b_result.release();

    if (!scene.reset(seq_a) || !scene.reset(seq_b)) return;
    SpecTransaction txn_a;
    SpecTransaction txn_b;
    bool ok = true;
    for (uint32_t i = 0; i < 20u && ok; ++i) {
        Scenario sa;
        Scenario sb;
        build_scenario(sa, 4u, 4u);
        build_scenario(sb, 0u, 4u);
        if (!ps::qwen35::runtime::spec_transaction_begin(txn_a, mtp_a, seq_a).ok() ||
            !ps::qwen35::runtime::spec_transaction_begin(txn_b, mtp_b, seq_b).ok()) {
            ok = false;
            break;
        }
        if (!ps::qwen35::runtime::mtp_kv_reserve(mtp_a, 4u).ok() ||
            !ps::qwen35::runtime::mtp_kv_reserve(mtp_b, 4u).ok()) {
            ok = false;
            break;
        }
        if (!scene.advance(seq_a, 5u) || !scene.advance(seq_b, 5u)) { ok = false; break; }
        SpecVerifyResult ra;
        SpecVerifyResult rb;
        if (!ps::qwen35::runtime::spec_greedy_accept(
                 sa.drafts.data(), 4u, sa.target.data(), true, ra).ok() ||
            !ps::qwen35::runtime::spec_greedy_accept(
                 sb.drafts.data(), 4u, sb.target.data(), true, rb).ok()) {
            ok = false;
            break;
        }
        if (!ps::qwen35::runtime::spec_transaction_begin_verify(txn_a).ok() ||
            !ps::qwen35::runtime::spec_transaction_begin_verify(txn_b).ok()) {
            ok = false;
            break;
        }
        if (!ps::qwen35::runtime::spec_transaction_commit(txn_a, ra, mtp_a, seq_a, scene.stream).ok() ||
            !ps::qwen35::runtime::spec_transaction_commit(txn_b, rb, mtp_b, seq_b, scene.stream).ok()) {
            ok = false;
            break;
        }
        if (seq_a.position != (i + 1u) * 5u || seq_b.position != (i + 1u) * 1u) { ok = false; break; }
        if (mtp_a.logical_length != (i + 1u) * 4u || mtp_b.logical_length != (i + 1u) * 1u) {
            ok = false;
            break;
        }
        for (uint32_t pa : seq_a.block_table) {
            for (uint32_t pb : seq_b.block_table) {
                if (pa == pb) ok = false;
            }
        }
        if (!ok) break;
    }
    check(ok, std::string(label) + ": interleaved sequences isolated");
    check(scene.kv_pool->num_used_pages() ==
              static_cast<uint32_t>(seq_a.block_table.size() + seq_b.block_table.size()),
          std::string(label) + ": page accounting");

    (void)ps::qwen35::release_paged_sequence_state(seq_a, scene.stream);
    (void)ps::qwen35::release_paged_sequence_state(seq_b, scene.stream);
    (void)ps::qwen35::runtime::mtp_kv_shutdown(mtp_a);
    (void)ps::qwen35::runtime::mtp_kv_shutdown(mtp_b);
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    auto arena_result = ps::gpu::GpuArena::create(0, 256u * 1024u * 1024u);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create\n");
        return 1;
    }
    auto arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    const auto layout = GdnStatePoolLayout::make(2u, 8u, 3u, 2u, 4u, 4u);
    auto seq_pool_result = SequenceSlotPool::create(arena, kMaxSeq, 256u);
    auto gdn_pool_result = GdnStatePool::create(arena, kMaxSeq, layout);
    auto kv_pool_result = PagedKVPool::create(
        arena, kKvPages, kPageTokens, 1u, kKvHeads, kHeadDim, KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL: pool create\n");
        return 1;
    }
    SequenceSlotPool seq_pool = seq_pool_result.release();
    GdnStatePool gdn_pool = gdn_pool_result.release();
    PagedKVPool kv_pool = kv_pool_result.release();
    Scene scene;
    scene.seq_pool = &seq_pool;
    scene.gdn_pool = &gdn_pool;
    scene.kv_pool = &kv_pool;
    scene.stream = stream;

    MtpKvConfig mtp_cfg;
    mtp_cfg.num_pages = kKvPages;
    mtp_cfg.page_tokens = kPageTokens;
    mtp_cfg.num_attention_layers = 1u;
    mtp_cfg.kv_heads = kKvHeads;
    mtp_cfg.head_dim = kHeadDim;
    mtp_cfg.dtype = KVCacheDType::BF16;
    auto mtp_result = ps::qwen35::runtime::create_mtp_kv_state(arena, mtp_cfg, 0u, stream);
    if (!mtp_result.ok()) {
        std::printf("FAIL: mtp state create: %s\n", mtp_result.status().message().c_str());
        return 1;
    }
    MtpKvState mtp = mtp_result.release();

    PagedSequenceState seq;
    SpecTransaction txn;

    for (uint32_t accept = 0; accept <= 4u; ++accept) {
        const std::string label = "forced_a" + std::to_string(accept);
        (void)ps::qwen35::runtime::mtp_kv_reset(mtp, stream);
        run_forced(scene, seq, mtp, txn, 0u, accept, 4u, label.c_str());
    }

    const uint32_t boundaries[6] = {14u, 15u, 16u, 30u, 31u, 32u};
    for (uint32_t i = 0; i < 6u; ++i) {
        for (uint32_t accept = 0; accept <= 4u; ++accept) {
            const std::string label =
                "boundary_" + std::to_string(boundaries[i]) + "_a" + std::to_string(accept);
            (void)ps::qwen35::runtime::mtp_kv_reset(mtp, stream);
            run_forced(scene, seq, mtp, txn, boundaries[i], accept, 4u, label.c_str());
        }
    }

    (void)ps::qwen35::runtime::mtp_kv_reset(mtp, stream);
    rollback_case(scene, seq, mtp, txn);

    (void)ps::qwen35::runtime::mtp_kv_reset(mtp, stream);
    gdn_case(scene, seq);

    (void)ps::qwen35::runtime::mtp_kv_reset(mtp, stream);
    repeated_case(scene, seq, mtp, txn);

    if (seq.is_allocated()) {
        (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    }
    isolation_case(scene, mtp_cfg, arena);

    (void)ps::qwen35::runtime::mtp_kv_shutdown(mtp);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("test_qwen35_spec_transaction: passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
