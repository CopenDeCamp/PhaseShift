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
#include <cstring>
#include <vector>

namespace {

namespace qwen35 = ps::qwen35;
namespace q35rt = ps::qwen35::runtime;

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", name);
    if (cond) ++passed; else ++failed;
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

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
    return hipMemcpy(sampled.data(), output.sampled_tokens.data<int32_t>(),
                     output_rows * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess;
}

bool run_serial(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t new_tokens,
    std::vector<int32_t>& out,
    hipStream_t stream) {
    std::vector<int32_t> sampled;
    if (!sample_tokens(executor, sequence, prompt, 1u, sampled, stream)) return false;
    out.clear();
    out.push_back(sampled[0]);
    for (uint32_t i = 1u; i < new_tokens; ++i) {
        std::vector<int32_t> step{out.back()};
        if (!sample_tokens(executor, sequence, step, 1u, sampled, stream)) return false;
        out.push_back(sampled[0]);
    }
    return true;
}

std::vector<uint8_t> read_tensor(const ps::gpu::Tensor& t, hipStream_t stream) {
    const std::size_t bytes = t.allocation_bytes() - t.view_offset_bytes();
    std::vector<uint8_t> host(bytes);
    if (hipStreamSynchronize(stream) != hipSuccess) return {};
    if (hipMemcpy(host.data(), t.data<uint8_t>(), bytes, hipMemcpyDeviceToHost) != hipSuccess)
        return {};
    return host;
}

bool compare_kv(
    const qwen35::PagedKVPool& pool,
    const qwen35::PagedSequenceState& a,
    const qwen35::PagedSequenceState& b,
    uint32_t committed,
    hipStream_t stream) {
    const std::vector<uint8_t> hk = read_tensor(pool.k_pool(), stream);
    const std::vector<uint8_t> hv = read_tensor(pool.v_pool(), stream);
    if (hk.empty() || hv.empty()) return false;
    const uint32_t pt = pool.page_tokens();
    const uint32_t kvh = pool.kv_heads();
    const uint32_t hd = pool.head_dim();
    const std::size_t per_page = static_cast<std::size_t>(pt) * kvh * hd;
    const std::size_t per_layer = static_cast<std::size_t>(pool.num_pages()) * per_page;
    const std::size_t bytes_per_elem = 2u;
    const std::size_t bytes_per_page = per_page * bytes_per_elem;
    for (uint32_t layer = 0u; layer < pool.num_attention_layers(); ++layer) {
        for (uint32_t t = 0u; t < committed; ++t) {
            const uint32_t pos_page = t / pt;
            const uint32_t off = t % pt;
            if (pos_page >= a.block_table.size() || pos_page >= b.block_table.size()) return false;
            const uint32_t pa = a.block_table[pos_page];
            const uint32_t pb = b.block_table[pos_page];
            for (const std::vector<uint8_t>* pool_bytes : {&hk, &hv}) {
                const std::size_t la = static_cast<std::size_t>(layer) * per_layer
                                     + static_cast<std::size_t>(pa) * per_page
                                     + static_cast<std::size_t>(off) * kvh * hd;
                const std::size_t lb = static_cast<std::size_t>(layer) * per_layer
                                     + static_cast<std::size_t>(pb) * per_page
                                     + static_cast<std::size_t>(off) * kvh * hd;
                if (std::memcmp(pool_bytes->data() + la * bytes_per_elem,
                                pool_bytes->data() + lb * bytes_per_elem,
                                static_cast<std::size_t>(kvh) * hd * bytes_per_elem) != 0) {
                    return false;
                }
            }
        }
    }
    (void)bytes_per_page;
    return true;
}

bool compare_gdn(
    const qwen35::GdnStatePool& pool,
    qwen35::SequenceSlotId a,
    qwen35::SequenceSlotId b,
    hipStream_t stream) {
    const qwen35::GdnStatePoolDeviceView v = pool.device_view();
    if (v.recurrent_base == nullptr || v.conv_base == nullptr) return false;
    const std::size_t rec_elems = static_cast<std::size_t>(v.recurrent_slot_stride);
    const std::size_t conv_elems = static_cast<std::size_t>(v.conv_slot_stride);
    std::vector<float> ra(rec_elems), rb(rec_elems);
    std::vector<ps::bf16_t> ca(conv_elems), cb(conv_elems);
    if (hipMemcpy(ra.data(), v.recurrent_base + static_cast<std::size_t>(a) * rec_elems,
                  rec_elems * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(rb.data(), v.recurrent_base + static_cast<std::size_t>(b) * rec_elems,
                  rec_elems * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(ca.data(), v.conv_base + static_cast<std::size_t>(a) * conv_elems,
                  conv_elems * sizeof(ps::bf16_t), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipMemcpy(cb.data(), v.conv_base + static_cast<std::size_t>(b) * conv_elems,
                  conv_elems * sizeof(ps::bf16_t), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    const bool rec_ok = std::memcmp(ra.data(), rb.data(), rec_elems * sizeof(float)) == 0;
    const bool conv_ok =
        std::memcmp(ca.data(), cb.data(), conv_elems * sizeof(ps::bf16_t)) == 0;
    if (!rec_ok) {
        std::size_t first = 0u, diffs = 0u;
        for (std::size_t i = 0u; i < rec_elems; ++i) {
            if (ra[i] != rb[i]) {
                if (diffs == 0u) first = i;
                ++diffs;
            }
        }
        std::printf("  [gdn] recurrent differs: first=%zu a=%.8g b=%.8g diffs=%zu/%zu\n",
                    first, static_cast<double>(ra[first]), static_cast<double>(rb[first]),
                    diffs, rec_elems);
    }
    if (!conv_ok) {
        std::size_t first = 0u, diffs = 0u;
        for (std::size_t i = 0u; i < conv_elems; ++i) {
            if (std::memcmp(&ca[i], &cb[i], sizeof(ps::bf16_t)) != 0) {
                if (diffs == 0u) first = i;
                ++diffs;
            }
        }
        uint16_t ua = 0u, ub = 0u;
        std::memcpy(&ua, &ca[first], 2u);
        std::memcpy(&ub, &cb[first], 2u);
        const uint32_t ab = static_cast<uint32_t>(ua) << 16u;
        const uint32_t bb = static_cast<uint32_t>(ub) << 16u;
        float fa = 0.0f, fb = 0.0f;
        std::memcpy(&fa, &ab, 4u);
        std::memcpy(&fb, &bb, 4u);
        std::printf("  [gdn] conv differs: first=%zu a=%.8g b=%.8g diffs=%zu/%zu\n",
                    first, static_cast<double>(fa), static_cast<double>(fb), diffs, conv_elems);
    }
    return rec_ok && conv_ok;
}

std::vector<int32_t> read_psktok(const char* path) {
    std::vector<int32_t> out;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return out;
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "PSKLDTOK", 8) != 0) {
        std::fclose(f);
        return out;
    }
    uint32_t version = 0;
    uint64_t count = 0;
    if (std::fread(&version, 4, 1, f) != 1 || std::fread(&count, 8, 1, f) != 1) {
        std::fclose(f);
        return out;
    }
    out.resize(count);
    if (std::fread(out.data(), 4, count, f) != count) out.clear();
    std::fclose(f);
    return out;
}

void apply_replay_policy(q35rt::MtpDraftPolicy& p) {
    const char* e = std::getenv("PHASESHIFT_REPLAY_DYN");
    if (e == nullptr || e[0] != '1') return;
    p.dynamic = true;
    p.min_drafts = std::getenv("PHASESHIFT_REPLAY_DYN_MIN")
        ? static_cast<uint32_t>(std::strtoul(std::getenv("PHASESHIFT_REPLAY_DYN_MIN"), nullptr, 10))
        : 1u;
    p.stop_margin = std::getenv("PHASESHIFT_REPLAY_DYN_STOP")
        ? std::strtof(std::getenv("PHASESHIFT_REPLAY_DYN_STOP"), nullptr)
        : 1.0e30f;
    if (std::getenv("PHASESHIFT_REPLAY_DYN_DISCARD") != nullptr) {
        p.enable_discard = true;
        p.discard_margin = std::getenv("PHASESHIFT_REPLAY_DYN_DISCARD_TAU")
            ? std::strtof(std::getenv("PHASESHIFT_REPLAY_DYN_DISCARD_TAU"), nullptr)
            : 1.0e30f;
    }
}

}  // namespace

int main() {
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
        std::printf("SKIP: arena\n");
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
        std::printf("FAIL: no MTP weights\n");
        return 1;
    }
    const qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t page_tokens = 16u;
    const uint32_t context = 32u;
    const uint32_t new_tokens = std::getenv("PHASESHIFT_REPLAY_NEW")
        ? static_cast<uint32_t>(std::strtoul(std::getenv("PHASESHIFT_REPLAY_NEW"), nullptr, 10))
        : 16u;
    const uint32_t k_drafts = std::getenv("PHASESHIFT_REPLAY_K")
        ? static_cast<uint32_t>(std::strtoul(std::getenv("PHASESHIFT_REPLAY_K"), nullptr, 10))
        : 4u;
    const uint32_t max_seq_len = 64u + new_tokens + k_drafts + 8u;
    const uint32_t num_pages = ceil_div(max_seq_len, page_tokens) + 8u;

    auto seq_pool_result = qwen35::SequenceSlotPool::create(arena, 4u, num_pages + 8u);
    auto gdn_pool_result = qwen35::GdnStatePool::create(
        arena, 4u, qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, count_attention_layers(tc),
        static_cast<uint32_t>(tc.num_key_value_heads),
        static_cast<uint32_t>(tc.attention_head_dim), qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL: pools\n");
        return 1;
    }
    qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = 64u;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = 16u;
    auto exec_result = qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    if (!exec_result.ok()) { std::printf("FAIL: executor\n"); return 1; }
    qwen35::Executor executor = exec_result.release();

    q35rt::MtpExecutorConfig mtp_cfg;
    mtp_cfg.num_pages = num_pages + 8u;
    mtp_cfg.page_tokens = page_tokens;
    mtp_cfg.max_rows = 16u;
    auto mtp_result = q35rt::create_mtp_executor(model, qwen35::KVCacheDType::BF16, arena, mtp_cfg, stream);
    if (!mtp_result.ok()) { std::printf("FAIL: mtp\n"); return 1; }
    q35rt::MtpExecutor mtp = mtp_result.release();

    std::vector<int32_t> prompt(context);
    const char* pf = std::getenv("PHASESHIFT_REPLAY_PROMPT_FILE");
    if (pf != nullptr && pf[0] != '\0') {
        std::vector<int32_t> ids = read_psktok(pf);
        const uint32_t off = std::getenv("PHASESHIFT_REPLAY_PROMPT_OFFSET")
            ? static_cast<uint32_t>(std::strtoul(
                  std::getenv("PHASESHIFT_REPLAY_PROMPT_OFFSET"), nullptr, 10)) : 0u;
        if (ids.size() >= off + context)
            for (uint32_t i = 0u; i < context; ++i) prompt[i] = ids[off + i];
    } else {
        for (uint32_t i = 0u; i < context; ++i)
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    }

    // --- target KV / GDN serial replay ---
    {
        auto seq_s = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        auto seq_p = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        if (!seq_s.ok() || !seq_p.ok()) { std::printf("FAIL: seq\n"); return 1; }
        qwen35::PagedSequenceState serial = seq_s.release();
        qwen35::PagedSequenceState spec = seq_p.release();

        auto mtp_state_result = q35rt::create_mtp_kv_state(arena, q35rt::MtpKvConfig{
            .num_pages = num_pages + 8u, .page_tokens = page_tokens,
            .num_attention_layers = 1u, .kv_heads = 4u,
            .head_dim = static_cast<uint32_t>(tc.attention_head_dim),
            .dtype = qwen35::KVCacheDType::BF16}, 0u, stream);
        if (!mtp_state_result.ok()) { std::printf("FAIL: mtp state\n"); return 1; }
        q35rt::MtpKvState mtp_state = mtp_state_result.release();
        q35rt::SpecDecoderConfig dcfg;
        dcfg.num_drafts = k_drafts;
        apply_replay_policy(dcfg.draft_policy);
        dcfg.bonus_token_enabled = true;
        dcfg.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
        auto dec_result = q35rt::create_spec_decoder(
            executor, mtp, mtp_state, spec, gdn_pool, arena, dcfg, stream);
        if (!dec_result.ok()) { std::printf("FAIL: spec decoder: %s\n", dec_result.status().message().c_str()); return 1; }
        q35rt::SpecDecoder decoder = dec_result.release();

        // prefill via spec sync path
        std::vector<int32_t> sampled;
        std::vector<qwen35::ScheduledRequest> reqs;
        auto pre = qwen35::execute_batch(executor,
            make_batch(spec, reqs, prompt.data(), context, 0u, 1u), stream);
        if (!pre.ok()) { std::printf("FAIL: prefill\n"); return 1; }
        qwen35::BatchExecutionOutput pout = pre.release();
        int32_t first = -1;
        hipMemcpy(&first, pout.sampled_tokens.data<int32_t>(), 4, hipMemcpyDeviceToHost);
        auto sync = q35rt::spec_decoder_sync_prompt(decoder, prompt.data(),
                                                    pout.token_hidden.data<ps::bf16_t>(), context);
        if (!sync.ok()) { std::printf("FAIL: sync: %s\n", sync.message().c_str()); return 1; }
        std::vector<int32_t> toks_spec;
        toks_spec.push_back(first);
        int32_t pending = first;
        while (toks_spec.size() < new_tokens && pending >= 0) {
            auto step = q35rt::spec_decoder_step(decoder, pending);
            if (!step.ok()) { std::printf("FAIL: step\n"); return 1; }
            q35rt::SpecIterationOutput v = step.release();
            toks_spec.insert(toks_spec.end(), v.emitted_tokens.begin(), v.emitted_tokens.end());
            pending = v.pending_token;
            if (v.finished) break;
        }
        const uint32_t spec_committed = spec.position - context;
        std::vector<int32_t> toks_serial;
        if (!run_serial(executor, serial, prompt, spec_committed + 1u, toks_serial, stream)) return 1;

        check(toks_serial == toks_spec, "replay prompt: spec ON tokens == serial tokens");
        check(serial.position == spec.position, "replay prompt: committed position equal");
        const uint32_t committed = std::min(serial.position, spec.position);
        check(compare_kv(kv_pool, serial, spec, committed, stream),
              "target canonical KV serial replay bit-exact");
        check(compare_gdn(gdn_pool, serial.slot, spec.slot, stream),
              "GDN recurrent + conv state serial replay bit-exact");

        (void)q35rt::spec_decoder_shutdown(decoder);
        (void)q35rt::mtp_kv_shutdown(mtp_state);
        (void)qwen35::release_paged_sequence_state(serial, stream);
        (void)qwen35::release_paged_sequence_state(spec, stream);
    }

    // --- third real prompt: spec ON/OFF equivalence ---
    {
        const uint32_t ctx2 = 48u;
        std::vector<int32_t> prompt2(ctx2);
        for (uint32_t i = 0u; i < ctx2; ++i)
            prompt2[i] = static_cast<int32_t>(50u + (i * 91u + 17u) % 3000u);
        auto seq_s = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        auto seq_p = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        if (!seq_s.ok() || !seq_p.ok()) { std::printf("FAIL: seq2\n"); return 1; }
        qwen35::PagedSequenceState serial = seq_s.release();
        qwen35::PagedSequenceState spec = seq_p.release();
        std::vector<int32_t> toks_serial;
        if (!run_serial(executor, serial, prompt2, new_tokens, toks_serial, stream)) return 1;

        auto mtp_state_result = q35rt::create_mtp_kv_state(arena, q35rt::MtpKvConfig{
            .num_pages = num_pages + 8u, .page_tokens = page_tokens,
            .num_attention_layers = 1u, .kv_heads = 4u,
            .head_dim = static_cast<uint32_t>(tc.attention_head_dim),
            .dtype = qwen35::KVCacheDType::BF16}, 0u, stream);
        if (!mtp_state_result.ok()) { std::printf("FAIL: mtp state2\n"); return 1; }
        q35rt::MtpKvState mtp_state = mtp_state_result.release();
        q35rt::SpecDecoderConfig dcfg;
        dcfg.num_drafts = k_drafts;
        apply_replay_policy(dcfg.draft_policy);
        dcfg.bonus_token_enabled = true;
        dcfg.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
        auto dec_result = q35rt::create_spec_decoder(executor, mtp, mtp_state, spec, gdn_pool, arena, dcfg, stream);
        if (!dec_result.ok()) { std::printf("FAIL: spec decoder2\n"); return 1; }
        q35rt::SpecDecoder decoder = dec_result.release();
        std::vector<qwen35::ScheduledRequest> reqs;
        auto pre = qwen35::execute_batch(executor,
            make_batch(spec, reqs, prompt2.data(), ctx2, 0u, 1u), stream);
        if (!pre.ok()) { std::printf("FAIL: prefill2\n"); return 1; }
        qwen35::BatchExecutionOutput pout = pre.release();
        int32_t first = -1;
        hipMemcpy(&first, pout.sampled_tokens.data<int32_t>(), 4, hipMemcpyDeviceToHost);
        auto sync = q35rt::spec_decoder_sync_prompt(decoder, prompt2.data(),
                                                    pout.token_hidden.data<ps::bf16_t>(), ctx2);
        if (!sync.ok()) { std::printf("FAIL: sync2\n"); return 1; }
        std::vector<int32_t> toks_spec;
        toks_spec.push_back(first);
        int32_t pending = first;
        while (toks_spec.size() < new_tokens && pending >= 0) {
            auto step = q35rt::spec_decoder_step(decoder, pending);
            if (!step.ok()) { std::printf("FAIL: step2\n"); return 1; }
            q35rt::SpecIterationOutput v = step.release();
            toks_spec.insert(toks_spec.end(), v.emitted_tokens.begin(), v.emitted_tokens.end());
            pending = v.pending_token;
            if (v.finished) break;
        }
        if (toks_spec.size() > new_tokens) toks_spec.resize(new_tokens);
        check(toks_serial == toks_spec, "third real prompt: spec ON/OFF token exact");
        (void)q35rt::spec_decoder_shutdown(decoder);
        (void)q35rt::mtp_kv_shutdown(mtp_state);
        (void)qwen35::release_paged_sequence_state(serial, stream);
        (void)qwen35::release_paged_sequence_state(spec, stream);
    }

    // --- real-prompt EOS truncation ---
    {
        std::vector<int32_t> reference;
        auto seq_r = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        if (!seq_r.ok()) { std::printf("FAIL: seq3\n"); return 1; }
        qwen35::PagedSequenceState refseq = seq_r.release();
        if (!run_serial(executor, refseq, prompt, 8u, reference, stream)) return 1;
        (void)qwen35::release_paged_sequence_state(refseq, stream);
        const int32_t eos = reference[3];

        auto seq_p = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        if (!seq_p.ok()) { std::printf("FAIL: seq4\n"); return 1; }
        qwen35::PagedSequenceState spec = seq_p.release();
        auto mtp_state_result = q35rt::create_mtp_kv_state(arena, q35rt::MtpKvConfig{
            .num_pages = num_pages + 8u, .page_tokens = page_tokens,
            .num_attention_layers = 1u, .kv_heads = 4u,
            .head_dim = static_cast<uint32_t>(tc.attention_head_dim),
            .dtype = qwen35::KVCacheDType::BF16}, 0u, stream);
        if (!mtp_state_result.ok()) { std::printf("FAIL: mtp state3\n"); return 1; }
        q35rt::MtpKvState mtp_state = mtp_state_result.release();
        q35rt::SpecDecoderConfig dcfg;
        dcfg.num_drafts = k_drafts;
        apply_replay_policy(dcfg.draft_policy);
        dcfg.bonus_token_enabled = true;
        dcfg.eos_token = eos;
        dcfg.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
        auto dec_result = q35rt::create_spec_decoder(executor, mtp, mtp_state, spec, gdn_pool, arena, dcfg, stream);
        if (!dec_result.ok()) { std::printf("FAIL: spec decoder3\n"); return 1; }
        q35rt::SpecDecoder decoder = dec_result.release();
        std::vector<qwen35::ScheduledRequest> reqs;
        auto pre = qwen35::execute_batch(executor,
            make_batch(spec, reqs, prompt.data(), context, 0u, 1u), stream);
        if (!pre.ok()) { std::printf("FAIL: prefill3\n"); return 1; }
        qwen35::BatchExecutionOutput pout = pre.release();
        int32_t first = -1;
        hipMemcpy(&first, pout.sampled_tokens.data<int32_t>(), 4, hipMemcpyDeviceToHost);
        auto sync = q35rt::spec_decoder_sync_prompt(decoder, prompt.data(),
                                                    pout.token_hidden.data<ps::bf16_t>(), context);
        if (!sync.ok()) { std::printf("FAIL: sync3\n"); return 1; }
        bool saw_eos = false;
        bool stopped_after_eos = false;
        std::vector<int32_t> emitted;
        int32_t pending = first;
        uint32_t guard = 0u;
        while (pending >= 0 && guard++ < 16u) {
            auto step = q35rt::spec_decoder_step(decoder, pending);
            if (!step.ok()) { std::printf("FAIL: step3\n"); return 1; }
            q35rt::SpecIterationOutput v = step.release();
            for (int32_t t : v.emitted_tokens) {
                emitted.push_back(t);
                if (t == eos) { saw_eos = true; }
                else if (saw_eos) { stopped_after_eos = true; }
            }
            pending = v.pending_token;
            if (v.finished) break;
        }
        check(saw_eos, "real prompt EOS: eos token observed in emitted stream");
        check(!stopped_after_eos, "real prompt EOS: no token emitted after EOS");
        (void)q35rt::spec_decoder_shutdown(decoder);
        (void)q35rt::mtp_kv_shutdown(mtp_state);
        (void)qwen35::release_paged_sequence_state(spec, stream);
    }

    std::printf("MTP_GATE3_REPLAY: %s (%d passed, %d failed)\n",
                failed == 0 ? "PASS" : "FAIL", passed, failed);
    (void)qwen35::executor_shutdown(executor);
    (void)q35rt::mtp_executor_shutdown(mtp);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failed == 0 ? 0 : 1;
}
