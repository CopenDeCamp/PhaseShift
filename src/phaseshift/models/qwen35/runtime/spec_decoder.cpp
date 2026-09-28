#include <phaseshift/models/qwen35/runtime/spec_decoder.h>

#include <phaseshift/models/qwen35/runtime/ngram_tail.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ps::qwen35::runtime {

namespace {

struct ScopedTimer {
    std::chrono::steady_clock::time_point t0;
    double* out;
    explicit ScopedTimer(double* o) noexcept
        : t0(std::chrono::steady_clock::now()), out(o) {}
    ~ScopedTimer() {
        if (out != nullptr) {
            *out += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        }
    }
};

ScheduledBatch make_batch(
    PagedSequenceState& sequence,
    std::vector<ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    ::ps::runtime::VerifyNumericMode numeric_mode) {
    ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ::ps::runtime::ExecutionClass::DECODE
                                           : ::ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);

    ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    batch.speculative_verify = speculative_verify;
    batch.verify_numeric_mode = numeric_mode;
    return batch;
}

uint32_t clamp_draft_depth(const SpecDecoder& decoder, uint32_t requested) {
    uint32_t k = requested;
    const uint32_t target_room =
        decoder.sequence->max_seq_len - decoder.sequence->position;
    if (target_room == 0u) return 0u;
    k = std::min(k, target_room - 1u);
    const uint32_t mtp_room =
        decoder.mtp_state->capacity_tokens - decoder.mtp_state->logical_length;
    k = std::min(k, mtp_room);
    return k;
}

Status run_single_decode(
    SpecDecoder& decoder,
    int32_t token,
    SpecIterationOutput& out) {
    std::vector<int32_t> tokens{token};
    std::vector<ScheduledRequest> requests;
    ScheduledBatch batch = make_batch(
        *decoder.sequence, requests, tokens.data(), 1u, decoder.sequence->position, 1u,
        false, ::ps::runtime::VerifyNumericMode::Fast);
    auto executed = execute_batch(*decoder.target, batch, decoder.stream);
    if (!executed.ok()) return executed.status();
    if (decoder.timing != nullptr) ++decoder.timing->target_forwards;
    BatchExecutionOutput output = executed.release();
    int32_t sampled = -1;
    hipError_t err = hipMemcpy(
        &sampled, output.sampled_tokens.data<int32_t>(), sizeof(int32_t),
        hipMemcpyDeviceToHost);
    if (decoder.timing != nullptr) ++decoder.timing->device_to_host_reads;
    if (err != hipSuccess) {
        return Status::hip_error("spec single decode sample copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    err = hipMemcpy(
        decoder.pending_hidden, output.final_hidden.data<bf16_t>(),
        static_cast<std::size_t>(decoder.hidden_size) * sizeof(bf16_t),
        hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        return Status::hip_error("spec single decode hidden copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    out.emitted_tokens.push_back(sampled);
    out.pending_token = sampled;
    return Status::make_ok();
}

}  // namespace

Result<SpecDecoder> create_spec_decoder(
    Executor& target,
    MtpExecutor& mtp,
    MtpKvState& mtp_state,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const SpecDecoderConfig& config,
    hipStream_t stream) {

    if (target.model == nullptr) {
        return Status::invalid_state("create_spec_decoder: target model is null", __FILE__, __LINE__);
    }
    if (!mtp_state.is_initialized()) {
        return Status::invalid_state("create_spec_decoder: MTP state not initialized",
                                     __FILE__, __LINE__);
    }
    if (!sequence.is_allocated()) {
        return Status::invalid_state("create_spec_decoder: sequence not allocated",
                                     __FILE__, __LINE__);
    }

    SpecDecoder decoder;
    decoder.target = &target;
    decoder.mtp = &mtp;
    decoder.mtp_state = &mtp_state;
    decoder.sequence = &sequence;
    decoder.gdn_pool = &gdn_pool;
    decoder.stream = stream;
    decoder.config = config;
    {
        const char* exact_env = std::getenv("PHASESHIFT_VERIFY_EXACT");
        if (exact_env != nullptr && exact_env[0] == '1') {
            decoder.config.verify_numeric_mode = ::ps::runtime::VerifyNumericMode::Exact;
        }
    }
    decoder.hidden_size = static_cast<uint32_t>(target.model->text_config().hidden_size);

    decoder.gdn_conv_bytes = spec_gdn_conv_bytes(gdn_pool);
    decoder.gdn_rec_bytes = spec_gdn_recurrent_bytes(gdn_pool);
    auto conv_alloc = arena.allocate_aligned(decoder.gdn_conv_bytes, 256u);
    if (!conv_alloc.ok()) return conv_alloc.status();
    decoder.gdn_conv_snapshot = conv_alloc.release().data();
    auto rec_alloc = arena.allocate_aligned(decoder.gdn_rec_bytes, 256u);
    if (!rec_alloc.ok()) return rec_alloc.status();
    decoder.gdn_rec_snapshot = rec_alloc.release().data();

    auto hidden_alloc = arena.allocate_aligned(
        static_cast<std::size_t>(decoder.hidden_size) * sizeof(bf16_t), 256u);
    if (!hidden_alloc.ok()) return hidden_alloc.status();
    decoder.pending_hidden = static_cast<bf16_t*>(hidden_alloc.release().data());

    decoder.pending_hidden_valid = false;
    decoder.initialized = true;
    return decoder;
}

Status spec_decoder_shutdown(SpecDecoder& decoder) noexcept {
    decoder.gdn_conv_snapshot = nullptr;
    decoder.gdn_rec_snapshot = nullptr;
    decoder.gdn_conv_bytes = 0u;
    decoder.gdn_rec_bytes = 0u;
    decoder.pending_hidden = nullptr;
    decoder.pending_hidden_valid = false;
    decoder.initialized = false;
    return Status::make_ok();
}

void record_token_history(SpecDecoder& decoder, const std::vector<int32_t>& emitted) {
    for (int32_t t : emitted) decoder.token_history.push_back(t);
    constexpr std::size_t kCap = 8192u;
    if (decoder.token_history.size() > kCap) {
        decoder.token_history.erase(decoder.token_history.begin(),
                                    decoder.token_history.end() - kCap / 2u);
    }
}

void append_ngram_tail(
    SpecDecoder& decoder, int32_t pending_token, SpecDraftSet& drafts, uint32_t k_max) {
    const uint32_t n = decoder.config.ngram_n;
    if (n == 0u || drafts.steps.size() >= k_max) return;
    const std::vector<int32_t>& hist = decoder.token_history;

    std::vector<int32_t> seq;
    seq.reserve(hist.size() + 1u + drafts.steps.size());
    seq.insert(seq.end(), hist.begin(), hist.end());
    seq.push_back(pending_token);
    for (const SpecDraftStep& s : drafts.steps)
        seq.push_back(static_cast<int32_t>(s.draft_token));

    NgramTailConfig cfg;
    cfg.n = decoder.config.ngram_n;
    cfg.max_tail = decoder.config.ngram_max_tail;
    cfg.window = decoder.config.ngram_window;
    auto proposal = propose_ngram_tail(seq, cfg);
    if (!proposal.ok()) return;
    const NgramTailProposal& value = proposal.value();
    if (!value.hit) return;

    for (int32_t token : value.tokens) {
        if (drafts.steps.size() >= k_max) break;
        SpecDraftStep s{};
        s.draft_token = static_cast<uint32_t>(token);
        drafts.steps.push_back(s);
    }
}

Status spec_decoder_sync_prompt(
    SpecDecoder& decoder,
    const int32_t* prompt_tokens,
    const bf16_t* prompt_hidden,
    uint32_t prompt_tokens_count) {

    if (!decoder.initialized) {
        return Status::invalid_state("spec_decoder_sync_prompt: not initialized", __FILE__, __LINE__);
    }
    if (prompt_tokens == nullptr || prompt_hidden == nullptr) {
        return Status::invalid_argument("spec_decoder_sync_prompt: inputs are null",
                                        __FILE__, __LINE__);
    }
    if (prompt_tokens_count == 0u) {
        return Status::invalid_argument("spec_decoder_sync_prompt: empty prompt",
                                        __FILE__, __LINE__);
    }

    if (prompt_tokens_count > 1u) {
        const uint32_t total_rows = prompt_tokens_count - 1u;
        uint32_t offset = 0u;
        while (offset < total_rows) {
            const uint32_t chunk = std::min(decoder.mtp->max_rows, total_rows - offset);
            auto run = mtp_forward_step(
                *decoder.mtp,
                *decoder.mtp_state,
                prompt_hidden + static_cast<std::size_t>(offset) * decoder.hidden_size,
                prompt_tokens + 1u + offset,
                chunk,
                offset,
                decoder.stream);
            if (!run.ok()) return run.status();
            offset += chunk;
        }
    }

    hipError_t err = hipMemcpy(
        decoder.pending_hidden,
        prompt_hidden + static_cast<std::size_t>(prompt_tokens_count - 1u) * decoder.hidden_size,
        static_cast<std::size_t>(decoder.hidden_size) * sizeof(bf16_t),
        hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        return Status::hip_error("spec_decoder_sync_prompt hidden copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    decoder.pending_hidden_valid = true;
    decoder.token_history.assign(prompt_tokens, prompt_tokens + prompt_tokens_count);
    return Status::make_ok();
}

Result<SpecIterationOutput> spec_decoder_step(
    SpecDecoder& decoder,
    int32_t pending_token) {

    if (!decoder.initialized) {
        return Status::invalid_state("spec_decoder_step: not initialized", __FILE__, __LINE__);
    }
    if (!decoder.pending_hidden_valid) {
        return Status::invalid_state("spec_decoder_step: pending hidden not set", __FILE__, __LINE__);
    }

    SpecIterationOutput out;
    SpecDecoderTiming* tm = decoder.timing;
    ScopedTimer total_timer(tm ? &tm->total_ms : nullptr);
    if (tm != nullptr) ++tm->iterations;
    const uint32_t k_eff = clamp_draft_depth(decoder, decoder.config.num_drafts);
    if (k_eff == 0u) {
        Status status = run_single_decode(decoder, pending_token, out);
        if (!status.ok()) return status;
        record_token_history(decoder, out.emitted_tokens);
        if (decoder.config.eos_token >= 0 && out.pending_token == decoder.config.eos_token) {
            out.finished = true;
        }
        return out;
    }

    SpecTransaction& txn = decoder.transaction;
    Status begin = spec_transaction_begin(txn, *decoder.mtp_state, *decoder.sequence);
    if (!begin.ok()) return begin;
    txn.bonus_token_enabled = decoder.config.bonus_token_enabled;

    Status gdn = ([&]() {
        ScopedTimer t(tm ? &tm->gdn_snapshot_ms : nullptr);
        return spec_gdn_snapshot(
            *decoder.gdn_pool, decoder.sequence->slot, decoder.gdn_conv_snapshot,
            decoder.gdn_rec_snapshot, decoder.gdn_conv_bytes, decoder.gdn_rec_bytes, decoder.stream);
    })();
    if (!gdn.ok()) {
        (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
        return gdn;
    }

    auto drafts_result = [&]() {
        ScopedTimer t(tm ? &tm->draft_ms : nullptr);
        return mtp_generate_drafts(
            *decoder.mtp, *decoder.mtp_state, decoder.pending_hidden, pending_token, k_eff,
            decoder.sequence->position - 1u, decoder.stream, nullptr,
            &decoder.config.draft_policy);
    }();
    if (!drafts_result.ok()) {
        (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
        return drafts_result.status();
    }
    SpecDraftSet drafts = drafts_result.release();
    const uint32_t mtp_actual_k = static_cast<uint32_t>(drafts.steps.size());
    if (mtp_actual_k == 0u) {
        Status rb = spec_transaction_rollback(
            txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
        if (!rb.ok()) return rb;
        ScopedTimer verify_timer(tm ? &tm->verify_ms : nullptr);
        Status st0 = run_single_decode(decoder, pending_token, out);
        if (!st0.ok()) return st0;
        out.num_drafts_generated = 0u;
        record_token_history(decoder, out.emitted_tokens);
        if (decoder.config.eos_token >= 0 && out.pending_token == decoder.config.eos_token) {
            out.finished = true;
        }
        return out;
    }
    append_ngram_tail(decoder, pending_token, drafts, k_eff);
    const uint32_t actual_k = static_cast<uint32_t>(drafts.steps.size());

    std::vector<int32_t> verify_tokens(actual_k + 1u);
    verify_tokens[0] = pending_token;
    for (uint32_t k = 0u; k < actual_k; ++k) {
        verify_tokens[k + 1u] = static_cast<int32_t>(drafts.steps[k].draft_token);
    }

    std::vector<int32_t> sampled(actual_k + 1u, 0);
    {
        ScopedTimer verify_timer(tm ? &tm->verify_ms : nullptr);
        std::vector<ScheduledRequest> requests;
        ScheduledBatch batch = make_batch(
            *decoder.sequence, requests, verify_tokens.data(), actual_k + 1u,
            decoder.sequence->position, actual_k + 1u, true, decoder.config.verify_numeric_mode);
        auto executed = execute_batch(*decoder.target, batch, decoder.stream);
        if (!executed.ok()) {
            (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
            return executed.status();
        }
        if (tm != nullptr) ++tm->target_forwards;
        BatchExecutionOutput output = executed.release();
        hipError_t err = hipMemcpy(
            sampled.data(), output.sampled_tokens.data<int32_t>(),
            sampled.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
            return Status::hip_error("spec verify sample copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        if (tm != nullptr) ++tm->device_to_host_reads;
        Status verify = spec_transaction_begin_verify(txn);
        if (!verify.ok()) {
            (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
            return verify;
        }

        SpecVerifyResult result;
        Status accept = spec_greedy_accept(
            reinterpret_cast<const int32_t*>(verify_tokens.data() + 1), actual_k, sampled.data(),
            decoder.config.bonus_token_enabled, result);
        if (!accept.ok()) {
            (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
            return accept;
        }
        result.num_mtp_drafts = mtp_actual_k;
        out.num_accepted_drafts = result.num_accepted_drafts;
        out.num_drafts_generated = actual_k;
        if (decoder.draft_log != nullptr) {
            for (uint32_t k = 0u; k < actual_k; ++k) {
                decoder.draft_log->emplace_back(
                    drafts.steps[k].margin,
                    static_cast<uint8_t>(k < result.num_accepted_drafts ? 1u : 0u));
            }
        }

        if (result.num_accepted_drafts == actual_k) {
            const std::size_t hid_off = static_cast<std::size_t>(actual_k);
            err = hipMemcpy(
                decoder.pending_hidden,
                output.final_hidden.data<bf16_t>() + hid_off * decoder.hidden_size,
                static_cast<std::size_t>(decoder.hidden_size) * sizeof(bf16_t),
                hipMemcpyDeviceToDevice);
            if (err != hipSuccess) {
                (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
                return Status::hip_error("spec verify hidden copy", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
            Status commit = ([&]() {
                ScopedTimer t(tm ? &tm->commit_ms : nullptr);
                return spec_transaction_commit(
                    txn, result, *decoder.mtp_state, *decoder.sequence, decoder.stream);
            })();
            if (!commit.ok()) {
                (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
                return commit;
            }
        } else {
            Status restore = ([&]() {
                ScopedTimer t(tm ? &tm->gdn_restore_ms : nullptr);
                return spec_gdn_restore(
                    *decoder.gdn_pool, decoder.sequence->slot, decoder.gdn_conv_snapshot,
                    decoder.gdn_rec_snapshot, decoder.gdn_conv_bytes, decoder.gdn_rec_bytes,
                    decoder.stream);
            })();
            if (!restore.ok()) {
                (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
                return restore;
            }
            decoder.sequence->position = txn.target_before.position;
            std::vector<int32_t> prefix(result.num_accepted_drafts + 1u);
            prefix[0] = pending_token;
            for (uint32_t i = 0u; i < result.num_accepted_drafts; ++i) {
                prefix[i + 1u] = verify_tokens[i + 1u];
            }
            std::vector<ScheduledRequest> rerun_requests;
            ScheduledBatch rerun = make_batch(
                *decoder.sequence, rerun_requests, prefix.data(),
                static_cast<uint32_t>(prefix.size()), txn.target_before.position, 1u,
                true, decoder.config.verify_numeric_mode);
            auto rerun_executed = execute_batch(*decoder.target, rerun, decoder.stream);
            if (!rerun_executed.ok()) {
                (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
                return rerun_executed.status();
            }
            if (tm != nullptr) ++tm->target_forwards;
            BatchExecutionOutput rerun_output = rerun_executed.release();
            err = hipMemcpy(
                decoder.pending_hidden, rerun_output.final_hidden.data<bf16_t>(),
                static_cast<std::size_t>(decoder.hidden_size) * sizeof(bf16_t),
                hipMemcpyDeviceToDevice);
            if (err != hipSuccess) {
                (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
                return Status::hip_error("spec rerun hidden copy", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
            Status commit = ([&]() {
                ScopedTimer t(tm ? &tm->commit_ms : nullptr);
                return spec_transaction_commit(
                    txn, result, *decoder.mtp_state, *decoder.sequence, decoder.stream);
            })();
            if (!commit.ok()) {
                (void)spec_transaction_abort(txn, *decoder.mtp_state, *decoder.sequence, decoder.stream);
                return commit;
            }
        }

        out.emitted_tokens = result.emitted_tokens;
        if (result.correction_valid) {
            out.pending_token = result.correction_token;
        } else if (result.bonus_token_valid) {
            out.pending_token = result.bonus_token;
        } else {
            out.pending_token = -1;
        }
    }

    if (decoder.config.eos_token >= 0) {
        for (std::size_t i = 0u; i < out.emitted_tokens.size(); ++i) {
            if (out.emitted_tokens[i] == decoder.config.eos_token) {
                out.emitted_tokens.resize(i + 1u);
                out.finished = true;
                break;
            }
        }
        if (out.pending_token == decoder.config.eos_token) {
            out.finished = true;
        }
    }
    record_token_history(decoder, out.emitted_tokens);
    return out;
}

}  // namespace ps::qwen35::runtime
