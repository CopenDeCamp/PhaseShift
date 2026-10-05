#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2ContextState;
using ps::qwen35::dflash2::DFlash2Executor;
using ps::qwen35::dflash2::DFlash2ExecutorConfig;
using ps::qwen35::dflash2::DFlash2Weights;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

static ps::qwen35::ScheduledBatch make_batch(
    ps::qwen35::PagedSequenceState& sequence,
    std::vector<ps::qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t output_rows) {
    ps::qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ps::runtime::ExecutionClass::DECODE
                                           : ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = sequence.position;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);
    ps::qwen35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    return batch;
}

static bool sample_tokens(
    ps::qwen35::Executor& executor,
    ps::qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& tokens,
    uint32_t output_rows,
    std::vector<int32_t>& sampled,
    hipStream_t stream) {
    std::vector<ps::qwen35::ScheduledRequest> requests;
    ps::qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, tokens.data(), static_cast<uint32_t>(tokens.size()), output_rows);
    auto executed = ps::qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: execute_batch: %s\n", executed.status().message().c_str());
        return false;
    }
    ps::qwen35::BatchExecutionOutput output = executed.release();
    sampled.assign(output_rows, -1);
    const hipError_t err = hipMemcpy(sampled.data(), output.sampled_tokens.data<int32_t>(),
                                     output_rows * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::printf("FAIL: sample copy: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

static bool run_baseline(
    ps::qwen35::Executor& executor,
    ps::qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    std::vector<int32_t>& out,
    hipStream_t stream) {
    std::vector<int32_t> sampled;
    if (!sample_tokens(executor, sequence, prompt, 1u, sampled, stream)) return false;
    out.clear();
    out.push_back(sampled[0]);
    while (out.size() < max_new) {
        std::vector<int32_t> step{out.back()};
        if (!sample_tokens(executor, sequence, step, 1u, sampled, stream)) return false;
        out.push_back(sampled[0]);
    }
    return true;
}

static bool run_speculative(
    ps::qwen35::runtime::DFlash2SpecDecoder& decoder,
    ps::qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    std::vector<int32_t>& out,
    ps::qwen35::runtime::DFlash2SpecTiming& timing,
    hipStream_t stream) {
    auto prefill = ps::qwen35::runtime::dflash2_spec_prefill(
        decoder, prompt.data(), static_cast<uint32_t>(prompt.size()));
    if (!prefill.ok()) {
        std::printf("FAIL: spec prefill: %s\n", prefill.status().message().c_str());
        return false;
    }
    out.clear();
    int32_t pending = prefill.value().pending_token;
    out.push_back(pending);
    while (out.size() < max_new && pending >= 0) {
        const uint32_t remaining = max_new - static_cast<uint32_t>(out.size());
        auto step = ps::qwen35::runtime::dflash2_spec_step(decoder, pending, remaining);
        if (!step.ok()) {
            std::printf("FAIL: spec step: %s\n", step.status().message().c_str());
            return false;
        }
        auto value = step.release();
        for (uint32_t i = 0; i < value.emitted_count; ++i) {
            out.push_back(value.emitted[i]);
        }
        pending = value.pending_token;
        if (value.finished) break;
    }
    if (out.size() > max_new) out.resize(max_new);
    (void)sequence;
    (void)timing;
    return true;
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (model_dir == nullptr || model_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2 and _TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    const char* exact_env = std::getenv("PHASESHIFT_DFLASH2_VERIFY_EXACT");
    const bool verify_exact = (exact_env == nullptr) || (exact_env[0] != '0');
    const char* long_env = std::getenv("PHASESHIFT_DFLASH2_GATE9_LONG");
    const bool run_long = (long_env != nullptr && long_env[0] == '1');

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "dflash config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();
    const uint32_t tap_count = static_cast<uint32_t>(config.num_target_layer_ids);
    const uint32_t block_rows = static_cast<uint32_t>(config.block_size);
    const uint32_t max_drafts = block_rows - 1u;

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = target_dir;
    resident_request.dflash2_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 30ull * 1024ull * 1024ull * 1024ull;
    auto resident_result = ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    resident_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture resident = resident_result.release();

    auto arena_result = ps::gpu::GpuArena::create(0, resident.arena_bytes());
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    auto model_result = resident.enabled()
        ? resident.take_qwen35()
        : ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(model_result.ok(), "target model load");
    if (!model_result.ok()) return 1;
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();

    auto weights_result = resident.enabled()
        ? resident.take_dflash2(nullptr)
        : ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "dflash weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    const uint32_t max_new = 128u;
    const uint32_t max_context = run_long ? 2048u : 256u;
    const uint32_t required_seq_len = max_context + max_new + max_drafts + 8u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = 2u * ceil_div(required_seq_len, page_tokens) + 24u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 2u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 2u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    for (uint32_t t = 0; t < tap_count; ++t) {
        taps[t] = static_cast<uint32_t>(config.target_layer_ids[t]);
    }

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = 2048u;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = block_rows;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "target executor create");
    if (!exec_result.ok()) return 1;
    ps::qwen35::Executor ex = exec_result.release();

    DFlash2ExecutorConfig draft_cfg;
    draft_cfg.max_rows = block_rows;
    draft_cfg.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    draft_cfg.target_lm_head = &model.weights().lm_head;
    draft_cfg.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result =
        ps::qwen35::dflash2::create_dflash2_executor(config, weights, arena, draft_cfg);
    check(draft_result.ok(), "draft executor create");
    if (!draft_result.ok()) return 1;
    DFlash2Executor draft = draft_result.release();

    struct CaseSpec {
        std::string name;
        uint32_t context;
        uint32_t k;
        uint32_t window_seed;
    };
    std::vector<CaseSpec> cases;
    cases.push_back({"prose_k7_ctx32", 32u, 7u, 11u});
    cases.push_back({"code_k7_ctx32", 32u, 7u, 23u});
    cases.push_back({"json_k7_ctx32", 32u, 7u, 47u});
    cases.push_back({"math_k7_ctx32", 32u, 7u, 71u});
    cases.push_back({"prose_k1_ctx32", 32u, 1u, 11u});
    cases.push_back({"prose_k3_ctx32", 32u, 3u, 11u});
    if (run_long) {
        cases.push_back({"prose_k7_ctx128", 128u, 7u, 11u});
    }
    if (run_long) {
        cases.push_back({"prose_k7_ctx2048", 2048u, 7u, 11u});
    }

    for (const CaseSpec& cs : cases) {
        std::vector<int32_t> prompt(cs.context);
        for (uint32_t i = 0; i < cs.context; ++i) {
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + cs.window_seed) % 2000u);
        }
        const uint32_t seq_len = cs.context + max_new + cs.k + 8u;

        auto base_seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, seq_len, stream);
        auto spec_seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, seq_len, stream);
        check(base_seq_result.ok() && spec_seq_result.ok(), cs.name + " sequence create");
        if (!base_seq_result.ok() || !spec_seq_result.ok()) continue;
        ps::qwen35::PagedSequenceState base_seq = base_seq_result.release();
        ps::qwen35::PagedSequenceState spec_seq = spec_seq_result.release();

        std::vector<int32_t> baseline;
        bool ok = run_baseline(ex, base_seq, prompt, max_new, baseline, stream);

        auto state_result =
            ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
        check(state_result.ok(), cs.name + " context create");
        if (!state_result.ok() || !ok) {
            (void)ps::qwen35::release_paged_sequence_state(base_seq, stream);
            (void)ps::qwen35::release_paged_sequence_state(spec_seq, stream);
            continue;
        }
        DFlash2ContextState state = state_result.release();

        ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_cfg;
        decoder_cfg.num_drafts = cs.k;
        decoder_cfg.verify_numeric_mode = verify_exact
                                              ? ps::runtime::VerifyNumericMode::Exact
                                              : ps::runtime::VerifyNumericMode::Fast;
        auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
            ex, draft, state, spec_seq, gdn_pool, arena, decoder_cfg, stream);
        check(decoder_result.ok(), cs.name + " decoder create: " +
                                       (decoder_result.ok() ? ""
                                                            : decoder_result.status().message()));
        if (decoder_result.ok()) {
            ps::qwen35::runtime::DFlash2SpecDecoder decoder = decoder_result.release();
            ps::qwen35::runtime::DFlash2SpecTiming timing;
            decoder.timing = &timing;
            std::vector<int32_t> spec;
            const bool spec_ok =
                run_speculative(decoder, spec_seq, prompt, max_new, spec, timing, stream);
            check(spec_ok, cs.name + " speculative run");

            bool token_exact = false;
            std::size_t first_mismatch = 0;
            if (spec_ok) {
                token_exact = (baseline == spec);
                if (!token_exact) {
                    while (first_mismatch < baseline.size() && first_mismatch < spec.size() &&
                           baseline[first_mismatch] == spec[first_mismatch]) {
                        ++first_mismatch;
                    }
                }
                const float mean_accept = timing.rounds > 0u
                                              ? static_cast<float>(timing.accepted_drafts) /
                                                    static_cast<float>(timing.rounds)
                                              : 0.0f;
                const float full_rate = timing.rounds > 0u
                                            ? static_cast<float>(timing.full_accepts) /
                                                  static_cast<float>(timing.rounds)
                                            : 0.0f;
                std::printf(
                    "%-18s k=%u ctx=%u gen=%zu exact=%s rounds=%u mean_acc=%.3f "
                    "full=%.3f reruns=%u\n",
                    cs.name.c_str(), cs.k, cs.context, spec.size(),
                    token_exact ? "yes" : "no", timing.rounds, mean_accept, full_rate,
                    timing.reruns);
                std::printf(
                    "  timing ms draft=%.2f d2h=%.2f verify=%.2f rerun=%.2f gdn_snap=%.2f "
                    "gdn_restore=%.2f commit=%.2f round=%.2f\n",
                    timing.draft_ms, timing.draft_d2h_ms, timing.verify_ms, timing.rerun_ms,
                    timing.gdn_snapshot_ms, timing.gdn_restore_ms, timing.dflash_commit_ms,
                    timing.round_ms);
                if (!token_exact) {
                    std::printf("  FIRST_MISMATCH=token[%zu] baseline=%d spec=%d\n",
                                first_mismatch,
                                first_mismatch < baseline.size() ? baseline[first_mismatch] : -1,
                                first_mismatch < spec.size() ? spec[first_mismatch] : -1);
                }
                check(token_exact, cs.name + " token exact");
                check(spec.size() <= max_new, cs.name + " max_new_tokens");
                check(state.next_position == spec_seq.position,
                      cs.name + " context.next_position == sequence.position");
            }
            check(ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder).ok(),
                  cs.name + " decoder shutdown");
        }

        (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
        (void)ps::qwen35::release_paged_sequence_state(base_seq, stream);
        (void)ps::qwen35::release_paged_sequence_state(spec_seq, stream);
    }

    if (!run_long) {
        const uint32_t context = 32u;
        std::vector<int32_t> prompt(context);
        for (uint32_t i = 0; i < context; ++i) {
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
        }
        const uint32_t seq_len = context + max_new + max_drafts + 8u;
        auto base_seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, seq_len, stream);
        auto spec_seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, seq_len, stream);
        check(base_seq_result.ok() && spec_seq_result.ok(), "eos sequence create");
        if (base_seq_result.ok() && spec_seq_result.ok()) {
            ps::qwen35::PagedSequenceState base_seq = base_seq_result.release();
            ps::qwen35::PagedSequenceState spec_seq = spec_seq_result.release();
            std::vector<int32_t> baseline;
            bool ok = run_baseline(ex, base_seq, prompt, max_new, baseline, stream);
            std::size_t eos_index = 0u;
            if (ok) {
                for (std::size_t j = 1u; j < baseline.size(); ++j) {
                    bool seen = false;
                    for (std::size_t i = 0u; i < j; ++i) {
                        if (baseline[i] == baseline[j]) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) {
                        eos_index = j;
                        break;
                    }
                }
            }
            check(ok && eos_index > 0u, "eos baseline");
            if (ok && eos_index > 0u) {
                auto state_result =
                    ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
                DFlash2ContextState state = state_result.release();
                ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_cfg;
                decoder_cfg.num_drafts = max_drafts;
                decoder_cfg.eos_tokens = {baseline[eos_index]};
                decoder_cfg.verify_numeric_mode =
                    verify_exact ? ps::runtime::VerifyNumericMode::Exact
                                 : ps::runtime::VerifyNumericMode::Fast;
                auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
                    ex, draft, state, spec_seq, gdn_pool, arena, decoder_cfg, stream);
                check(decoder_result.ok(), "eos decoder create");
                if (decoder_result.ok()) {
                    ps::qwen35::runtime::DFlash2SpecDecoder decoder = decoder_result.release();
                    std::vector<int32_t> spec;
                    ps::qwen35::runtime::DFlash2SpecTiming timing;
                    decoder.timing = &timing;
                    const bool spec_ok =
                        run_speculative(decoder, spec_seq, prompt, max_new, spec, timing, stream);
                    const std::size_t expected = eos_index + 1u;
                    bool prefix_ok = spec_ok && spec.size() == expected;
                    if (prefix_ok) {
                        for (std::size_t i = 0u; i < expected; ++i) {
                            if (spec[i] != baseline[i]) prefix_ok = false;
                        }
                    }
                    std::printf(
                        "eos_case           eos_index=%zu eos=%d spec_len=%zu expected=%zu ok=%s\n",
                        eos_index, baseline[eos_index], spec.size(), expected,
                        prefix_ok ? "yes" : "no");
                    check(prefix_ok, "eos truncation and prefix");
                    check(!spec.empty() && spec.back() == baseline[eos_index],
                          "eos is the last emitted token");
                    check(ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder).ok(),
                          "eos decoder shutdown");
                }
                (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
            }
            (void)ps::qwen35::release_paged_sequence_state(base_seq, stream);
            (void)ps::qwen35::release_paged_sequence_state(spec_seq, stream);
        }
    }

    check(ps::qwen35::dflash2::dflash2_executor_shutdown(draft).ok(), "draft shutdown");
    check(ps::qwen35::executor_shutdown(ex).ok(), "target shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate9_e2e: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
