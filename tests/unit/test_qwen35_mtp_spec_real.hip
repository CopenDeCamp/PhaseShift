#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/runtime/spec_decoder.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace qwen35 = ps::qwen35;
namespace q35rt = ps::qwen35::runtime;

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t count_attention_layers(const qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

qwen35::ScheduledBatch make_batch(
    qwen35::PagedSequenceState& sequence,
    std::vector<qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows) {
    qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ps::runtime::ExecutionClass::DECODE
                                           : ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);
    qwen35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    return batch;
}

bool sample_tokens(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& tokens,
    uint32_t output_rows,
    std::vector<int32_t>& sampled,
    hipStream_t stream) {
    std::vector<qwen35::ScheduledRequest> requests;
    qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, tokens.data(), static_cast<uint32_t>(tokens.size()),
        sequence.position, output_rows);
    auto executed = qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: execute_batch: %s\n", executed.status().message().c_str());
        return false;
    }
    qwen35::BatchExecutionOutput output = executed.release();
    sampled.assign(output_rows, -1);
    hipError_t err = hipMemcpy(
        sampled.data(), output.sampled_tokens.data<int32_t>(),
        output_rows * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::printf("FAIL: sample copy: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

struct SpecRunResult {
    std::vector<int32_t> tokens;
    uint32_t target_position = 0u;
    uint32_t mtp_length = 0u;
    uint32_t transitions = 0u;
    uint32_t reject_events = 0u;
    uint32_t reject_at_zero = 0u;
    uint32_t total_drafts = 0u;
    uint32_t accepted_drafts = 0u;
    uint32_t full_accepts = 0u;
};

bool run_baseline(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    SpecRunResult& out,
    hipStream_t stream) {
    std::vector<int32_t> sampled;
    if (!sample_tokens(executor, sequence, prompt, 1u, sampled, stream)) return false;
    out.tokens.clear();
    out.tokens.push_back(sampled[0]);
    for (uint32_t i = 1u; i < max_new; ++i) {
        std::vector<int32_t> step{out.tokens.back()};
        if (!sample_tokens(executor, sequence, step, 1u, sampled, stream)) return false;
        out.tokens.push_back(sampled[0]);
    }
    out.target_position = sequence.position;
    return true;
}

bool run_speculative(
    q35rt::SpecDecoder& decoder,
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    q35rt::MtpExecutor& mtp,
    q35rt::MtpKvState& mtp_state,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    SpecRunResult& out,
    hipStream_t stream) {
    std::vector<qwen35::ScheduledRequest> requests;
    qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, prompt.data(), static_cast<uint32_t>(prompt.size()),
        sequence.position, 1u);
    auto executed = qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: spec prefill: %s\n", executed.status().message().c_str());
        return false;
    }
    qwen35::BatchExecutionOutput prefill = executed.release();
    int32_t first = -1;
    hipError_t err = hipMemcpy(
        &first, prefill.sampled_tokens.data<int32_t>(), sizeof(int32_t),
        hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::printf("FAIL: spec prefill sample: %s\n", hipGetErrorString(err));
        return false;
    }
    auto sync = q35rt::spec_decoder_sync_prompt(
        decoder, prompt.data(), prefill.token_hidden.data<ps::bf16_t>(),
        static_cast<uint32_t>(prompt.size()));
    if (!sync.ok()) {
        std::printf("FAIL: spec sync: %s\n", sync.message().c_str());
        return false;
    }
    const uint32_t sync_expected = static_cast<uint32_t>(prompt.size()) - 1u;
    if (mtp_state.logical_length != sync_expected) {
        std::printf("FAIL: prompt sync mtp.logical_length=%u expected=%u\n",
                    mtp_state.logical_length, sync_expected);
        return false;
    }

    out.tokens.clear();
    out.tokens.push_back(first);
    int32_t pending = first;
    uint32_t mtp_before = mtp_state.logical_length;
    while (out.tokens.size() < max_new && pending >= 0) {
        auto step = q35rt::spec_decoder_step(decoder, pending);
        if (!step.ok()) {
            std::printf("FAIL: spec step: %s\n", step.status().message().c_str());
            return false;
        }
        q35rt::SpecIterationOutput value = step.release();
        const uint32_t mtp_kept =
            value.num_mtp_drafts > 0u
                ? std::min(value.num_accepted_drafts + 1u, value.num_mtp_drafts)
                : 0u;
        if (value.mtp_length_before != mtp_before ||
            value.mtp_length_before + mtp_kept != value.mtp_length_after ||
            value.mtp_length_after != mtp_state.logical_length ||
            value.rerun != (value.num_accepted_drafts < value.num_drafts_generated) ||
            (value.num_drafts_generated > 0u &&
             (value.draft_tokens.size() != value.num_drafts_generated ||
              value.verify_sampled.size() != value.num_drafts_generated + 1u))) {
            std::printf(
                "FAIL: spec iteration state contract: before=%u kept=%u after=%u "
                "actual=%u accepted=%u drafts=%u rerun=%d draft_tokens=%zu "
                "verify_sampled=%zu\n",
                value.mtp_length_before, mtp_kept, value.mtp_length_after,
                mtp_state.logical_length, value.num_accepted_drafts,
                value.num_drafts_generated, value.rerun ? 1 : 0,
                value.draft_tokens.size(), value.verify_sampled.size());
            return false;
        }
        mtp_before = value.mtp_length_after;
        out.tokens.insert(out.tokens.end(), value.emitted_tokens.begin(), value.emitted_tokens.end());
        ++out.transitions;
        if (decoder.config.num_drafts > 0u) {
            out.total_drafts += decoder.config.num_drafts;
            out.accepted_drafts += value.num_accepted_drafts;
            if (value.num_accepted_drafts == decoder.config.num_drafts) ++out.full_accepts;
        }
        if (value.num_accepted_drafts < decoder.config.num_drafts) {
            ++out.reject_events;
            if (value.num_accepted_drafts == 0u) ++out.reject_at_zero;
        }
        pending = value.pending_token;
        if (value.finished) break;
    }
    if (out.tokens.size() > max_new) out.tokens.resize(max_new);
    out.target_position = sequence.position;
    out.mtp_length = mtp_state.logical_length;
    (void)mtp;
    return true;
}

bool run_case(
    qwen35::Qwen35Model& model,
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    uint32_t draft_k,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    bool& tokens_match,
    SpecRunResult& baseline,
    SpecRunResult& speculative) {
    const auto& tc = model.text_config();
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t context = static_cast<uint32_t>(prompt.size());
    const uint32_t required_seq_len = context + max_new + draft_k + 4u;
    const uint32_t num_pages = (ceil_div(required_seq_len, page_tokens) + 2u) * 2u;

    auto seq_pool_result = qwen35::SequenceSlotPool::create(arena, 2u, num_pages + 8u);
    auto gdn_pool_result = qwen35::GdnStatePool::create(
        arena, 2u, qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL: pool create\n");
        return false;
    }
    qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = std::max(context, 32u);
    exec_config.max_scheduled_requests = 1u;
    exec_config.max_scheduled_output_rows = draft_k + 4u;
    auto exec_result = qwen35::create_model_executor(
        model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("FAIL: create_model_executor: %s\n", exec_result.status().message().c_str());
        return false;
    }
    qwen35::Executor executor = exec_result.release();

    auto seq_off_result = qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, required_seq_len, stream);
    auto seq_on_result = qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, required_seq_len, stream);
    if (!seq_off_result.ok() || !seq_on_result.ok()) {
        std::printf("FAIL: sequence create\n");
        return false;
    }
    qwen35::PagedSequenceState seq_off = seq_off_result.release();
    qwen35::PagedSequenceState seq_on = seq_on_result.release();

    if (!run_baseline(executor, seq_off, prompt, max_new, baseline, stream)) return false;

    q35rt::MtpExecutorConfig mtp_config;
    mtp_config.num_pages = num_pages + 8u;
    mtp_config.page_tokens = page_tokens;
    mtp_config.max_rows = 16u;
    auto mtp_result = q35rt::create_mtp_executor(
        model, qwen35::KVCacheDType::BF16, arena, mtp_config, stream);
    if (!mtp_result.ok()) {
        std::printf("FAIL: create_mtp_executor: %s\n", mtp_result.status().message().c_str());
        return false;
    }
    q35rt::MtpExecutor mtp = mtp_result.release();

    q35rt::MtpKvConfig mtp_kv_config;
    mtp_kv_config.num_pages = num_pages + 8u;
    mtp_kv_config.page_tokens = page_tokens;
    mtp_kv_config.num_attention_layers = 1u;
    mtp_kv_config.kv_heads = 4u;
    mtp_kv_config.head_dim = head_dim;
    mtp_kv_config.dtype = qwen35::KVCacheDType::BF16;
    auto mtp_state_result = q35rt::create_mtp_kv_state(arena, mtp_kv_config, 0u, stream);
    if (!mtp_state_result.ok()) {
        std::printf("FAIL: create_mtp_kv_state: %s\n", mtp_state_result.status().message().c_str());
        return false;
    }
    q35rt::MtpKvState mtp_state = mtp_state_result.release();

    q35rt::SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = draft_k;
    decoder_config.bonus_token_enabled = true;
    decoder_config.eos_token = -1;
    auto decoder_result = q35rt::create_spec_decoder(
        executor, mtp, mtp_state, seq_on, gdn_pool, arena, decoder_config, stream);
    if (!decoder_result.ok()) {
        std::printf("FAIL: create_spec_decoder: %s\n", decoder_result.status().message().c_str());
        return false;
    }
    q35rt::SpecDecoder decoder = decoder_result.release();
    if (!run_speculative(decoder, executor, seq_on, mtp, mtp_state, prompt, max_new, speculative, stream)) {
        return false;
    }

    tokens_match = (baseline.tokens == speculative.tokens);
    if (!tokens_match) {
        std::size_t first = 0u;
        while (first < baseline.tokens.size() && first < speculative.tokens.size() &&
               baseline.tokens[first] == speculative.tokens[first]) {
            ++first;
        }
        std::printf("  FIRST_MISMATCH=token[%zu] off=%d on=%d\n", first,
                    first < baseline.tokens.size() ? baseline.tokens[first] : -1,
                    first < speculative.tokens.size() ? speculative.tokens[first] : -1);
    }

    (void)q35rt::spec_decoder_shutdown(decoder);
    (void)q35rt::mtp_kv_shutdown(mtp_state);
    (void)qwen35::executor_shutdown(executor);
    (void)q35rt::mtp_executor_shutdown(mtp);
    (void)qwen35::release_paged_sequence_state(seq_off, stream);
    (void)qwen35::release_paged_sequence_state(seq_on, stream);
    return true;
}

}  // namespace

int main() {
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);

    const char* model_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_env == nullptr || model_env[0] == '\0') {
        model_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    if (model_env == nullptr || model_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result = qwen35::Qwen35Model::load_from_safetensors(model_env, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL: model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    qwen35::Qwen35Model model = model_result.release();
    if (!model.weights().mtp.present) {
        std::printf("FAIL: model has no MTP weights\n");
        return 1;
    }
    const qwen35::Qwen35TextConfig& text_config = model.text_config();
    const uint32_t hidden_size = static_cast<uint32_t>(text_config.hidden_size);

    const char* dump_env = std::getenv("PHASESHIFT_MTP_GATE2_REAL_DUMP");
    if (dump_env != nullptr && dump_env[0] != '\0') {
        const uint32_t prompt_len = 64u;
        std::vector<int32_t> prompt(prompt_len);
        for (uint32_t i = 0u; i < prompt_len; ++i) {
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
        }
        const uint32_t page_tokens = 16u;
        const uint32_t max_seq_len = prompt_len + 8u;
        const uint32_t num_pages = ceil_div(max_seq_len, page_tokens) + 2u;
        auto dump_seq_pool = qwen35::SequenceSlotPool::create(arena, 1u, num_pages + 4u);
        auto dump_gdn_pool = qwen35::GdnStatePool::create(
            arena, 1u, qwen35::GdnStatePoolLayout::from_text_config(text_config));
        auto dump_kv_pool = qwen35::PagedKVPool::create(
            arena, num_pages, page_tokens, count_attention_layers(text_config),
            static_cast<uint32_t>(text_config.num_key_value_heads),
            static_cast<uint32_t>(text_config.attention_head_dim),
            qwen35::KVCacheDType::BF16);
        if (!dump_seq_pool.ok() || !dump_gdn_pool.ok() || !dump_kv_pool.ok()) {
            std::printf("FAIL: real dump pool create\n");
            return 1;
        }
        qwen35::SequenceSlotPool dsp = dump_seq_pool.release();
        qwen35::GdnStatePool dgp = dump_gdn_pool.release();
        qwen35::PagedKVPool dkp = dump_kv_pool.release();
        qwen35::ExecutorConfig dcfg;
        dcfg.max_scheduled_tokens = prompt_len;
        dcfg.max_scheduled_requests = 1u;
        dcfg.max_scheduled_output_rows = 1u;
        auto dump_exec = qwen35::create_model_executor(model, dsp, dgp, dkp, arena, dcfg);
        auto dump_seq = qwen35::create_paged_sequence_state(dsp, dgp, dkp, max_seq_len, stream);
        if (!dump_exec.ok() || !dump_seq.ok()) {
            std::printf("FAIL: real dump executor create\n");
            return 1;
        }
        qwen35::Executor dex = dump_exec.release();
        qwen35::PagedSequenceState dseq = dump_seq.release();
        std::vector<qwen35::ScheduledRequest> requests;
        qwen35::ScheduledBatch batch = make_batch(
            dseq, requests, prompt.data(), prompt_len, 0u, 1u);
        auto executed = qwen35::execute_batch(dex, batch, stream);
        if (!executed.ok()) {
            std::printf("FAIL: real dump prefill: %s\n", executed.status().message().c_str());
            return 1;
        }
        qwen35::BatchExecutionOutput out = executed.release();
        std::vector<uint8_t> hidden_host(static_cast<std::size_t>(prompt_len) * hidden_size * 2u);
        hipError_t err = hipMemcpy(
            hidden_host.data(), out.token_hidden.data<ps::bf16_t>(), hidden_host.size(),
            hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            std::printf("FAIL: real dump hidden copy: %s\n", hipGetErrorString(err));
            return 1;
        }
        std::string dir = dump_env;
        std::filesystem::create_directories(dir);
        std::ofstream hf(dir + "/real_hidden.bin", std::ios::binary);
        hf.write(reinterpret_cast<const char*>(hidden_host.data()),
                 static_cast<std::streamsize>(hidden_host.size()));
        std::ofstream tf(dir + "/real_tokens.bin", std::ios::binary);
        tf.write(reinterpret_cast<const char*>(prompt.data()),
                 static_cast<std::streamsize>(prompt.size() * sizeof(int32_t)));
        std::ofstream mf(dir + "/real_meta.json");
        mf << "{\"N\":" << prompt_len << ",\"H\":" << hidden_size << "}\n";
        std::printf("MTP_GATE2_REAL_DUMP: PASS (%s, N=%u H=%u)\n", dir.c_str(), prompt_len,
                    hidden_size);
        (void)qwen35::executor_shutdown(dex);
        (void)qwen35::release_paged_sequence_state(dseq, stream);
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 0;
    }

    const uint32_t ks[5] = {0u, 1u, 2u, 4u, 8u};
    const uint32_t contexts[2] = {32u, 128u};
    bool k_pass[5] = {true, true, true, true, true};
    bool state_ok = true;
    uint32_t reject_total = 0u;
    uint32_t reject_zero_total = 0u;
    uint32_t stat_drafts = 0u;
    uint32_t stat_accepted = 0u;
    uint32_t stat_full = 0u;
    uint32_t stat_transitions = 0u;
    for (uint32_t ci = 0u; ci < 2u; ++ci) {
        const uint32_t context = contexts[ci];
        const uint32_t max_new = context >= 128u ? 24u : 32u;
        std::vector<int32_t> prompt(context);
        for (uint32_t i = 0u; i < context; ++i) {
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
        }
        for (uint32_t ki = 0u; ki < 5u; ++ki) {
            SpecRunResult baseline;
            SpecRunResult speculative;
            bool match = false;
            if (!run_case(model, arena, stream, ks[ki], prompt, max_new, match, baseline, speculative)) {
                k_pass[ki] = false;
                state_ok = false;
                continue;
            }
            if (!match) k_pass[ki] = false;
            reject_total += speculative.reject_events;
            reject_zero_total += speculative.reject_at_zero;
            stat_drafts += speculative.total_drafts;
            stat_accepted += speculative.accepted_drafts;
            stat_full += speculative.full_accepts;
            stat_transitions += speculative.transitions;
            if (speculative.target_position < context + speculative.tokens.size() - 1u) {
                state_ok = false;
            }
            std::printf(
                "  prompt=%u K=%u off=%zu on=%zu match=%d pos=%u mtp=%u iter=%u rej=%u rej0=%u\n",
                context, ks[ki], baseline.tokens.size(), speculative.tokens.size(), match ? 1 : 0,
                speculative.target_position, speculative.mtp_length, speculative.transitions,
                speculative.reject_events, speculative.reject_at_zero);
        }
    }

    for (uint32_t ki = 0u; ki < 5u; ++ki) {
        std::printf("MTP_GATE3_K%u: %s\n", ks[ki], k_pass[ki] ? "PASS" : "FAIL");
    }
    std::printf("MTP_GATE3_ROLLBACK: %s (reject_events=%u reject_at_zero=%u)\n",
                reject_total > 0u ? "PASS" : "N/A", reject_total, reject_zero_total);
    std::printf("MTP_GATE3_TARGET_STATE: %s\n", state_ok ? "PASS" : "FAIL");
    const double avg_accept =
        stat_drafts > 0u ? static_cast<double>(stat_accepted) / static_cast<double>(stat_drafts) : 0.0;
    const std::size_t emitted_total = 0u;
    (void)emitted_total;
    std::printf(
        "MTP_GATE3_ACCEPTANCE: iterations=%u drafts=%u accepted=%u avg=%.3f full_accept=%u "
        "reject_at_zero=%u\n",
        stat_transitions, stat_drafts, stat_accepted, avg_accept, stat_full, reject_zero_total);

    const int failures = (k_pass[0] && k_pass[1] && k_pass[2] && k_pass[3] && k_pass[4] && state_ok) ? 0 : 1;
    std::printf("MTP_GATE3_GREEDY_EQUIVALENCE: %s\n", failures == 0 ? "PASS" : "FAIL");
    std::printf("MTP_GATE3: %s\n", failures == 0 ? "PASS" : "FAIL");

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failures == 0 ? 0 : 1;
}
