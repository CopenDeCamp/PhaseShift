#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/runtime/spec_decode.h>
#include <phaseshift/models/qwen35/kernels/dflash2/feature_concat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ps::qwen35::runtime {

namespace {

bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return value[0] != '0';
}

uint32_t max_target_tap(const dflash2::DFlash2Config& config) {
    uint32_t max_tap = 0u;
    for (std::size_t i = 0; i < config.num_target_layer_ids; ++i) {
        max_tap = std::max(max_tap, static_cast<uint32_t>(config.target_layer_ids[i]));
    }
    return max_tap;
}

ScheduledBatch make_prefill_batch(
    PagedSequenceState& sequence,
    std::vector<ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    bool compute_logits,
    ::ps::runtime::VerifyNumericMode numeric_mode) {
    ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ::ps::runtime::ExecutionClass::DECODE
                                           : ::ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = compute_logits;
    req.sample = compute_logits;
    req.num_output_rows = 1u;
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
    batch.speculative_verify = false;
    batch.verify_numeric_mode = numeric_mode;
    return batch;
}

}  // namespace

Result<DFlash2SpecDecoder> create_dflash2_spec_decoder(
    Executor& target,
    dflash2::DFlash2Executor& draft,
    dflash2::DFlash2ContextState& context,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream) {
    if (target.model == nullptr) {
        return Status::invalid_state("create_dflash2_spec_decoder: target model is null",
                                     __FILE__, __LINE__);
    }
    if (!draft.initialized || draft.config == nullptr) {
        return Status::invalid_state("create_dflash2_spec_decoder: draft executor not initialized",
                                     __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("create_dflash2_spec_decoder: context not initialized",
                                     __FILE__, __LINE__);
    }
    if (!sequence.is_allocated()) {
        return Status::invalid_state("create_dflash2_spec_decoder: sequence not allocated",
                                     __FILE__, __LINE__);
    }

    const dflash2::DFlash2Config& dcfg = *draft.config;
    const Qwen35TextConfig& tc = target.model->text_config();
    const ps::weights::MatrixWeight& embed = target.model->weights().embed_tokens;

    if (embed.cols != dcfg.hidden_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target hidden_size mismatch", __FILE__, __LINE__);
    }
    if (embed.rows != dcfg.vocab_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target vocab_size mismatch", __FILE__, __LINE__);
    }
    if (tc.num_hidden_layers <= max_target_tap(dcfg)) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target layer count too small for taps", __FILE__,
            __LINE__);
    }
    if (dcfg.mask_token_id >= dcfg.vocab_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: mask_token_id exceeds vocab_size", __FILE__,
            __LINE__);
    }
    if (draft.target_embed_tokens == nullptr || draft.target_lm_head == nullptr) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: draft is missing target embed_tokens/lm_head",
            __FILE__, __LINE__);
    }
    if (draft.target_lm_head->rows != dcfg.vocab_size ||
        draft.target_lm_head->cols != dcfg.hidden_size) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: target lm_head shape mismatch", __FILE__, __LINE__);
    }
    if (config.num_drafts == 0u || config.num_drafts > dcfg.max_draft_tokens()) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: num_drafts out of range", __FILE__, __LINE__);
    }
    const bool ngram_enabled = config.ngram_n > 0u && config.ngram_max_tail > 0u;
    if ((config.ngram_n > 0u) != (config.ngram_max_tail > 0u)) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: ngram_n and ngram_max_tail must be set together",
            __FILE__, __LINE__);
    }
    if (ngram_enabled && config.ngram_window == 0u) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: ngram_window must be positive", __FILE__, __LINE__);
    }
    if (static_cast<uint64_t>(config.num_drafts) + config.ngram_max_tail >
        kDFlash2SpecMaxVerifyDrafts) {
        return Status::invalid_argument(
            "create_dflash2_spec_decoder: ngram tail exceeds verify capacity", __FILE__,
            __LINE__);
    }

    DFlash2SpecDecoder decoder;
    decoder.target = &target;
    decoder.draft = &draft;
    decoder.context = &context;
    decoder.sequence = &sequence;
    decoder.gdn_pool = &gdn_pool;
    decoder.stream = stream;
    decoder.config = config;
    decoder.hidden_size = static_cast<uint32_t>(dcfg.hidden_size);
    decoder.device_token_bridge =
        env_flag_enabled("PHASESHIFT_DFLASH2_DEVICE_TOKEN_BRIDGE", true);
    decoder.host_proposal_visibility =
        env_flag_enabled("PHASESHIFT_DFLASH2_HOST_PROPOSAL_D2H", false);

    {
        auto verify_alloc = arena.allocate_aligned(
            kDFlash2SpecMaxVerifyRows * sizeof(int32_t), 256u);
        if (!verify_alloc.ok()) return verify_alloc.status();
        decoder.verify_token_ids_device =
            static_cast<int32_t*>(verify_alloc.release().data());
        auto decision_alloc = arena.allocate_aligned(
            2u * kDFlash2SpecMaxVerifyRows * sizeof(int32_t), 256u);
        if (!decision_alloc.ok()) return decision_alloc.status();
        decoder.decision_staging_device =
            static_cast<int32_t*>(decision_alloc.release().data());
    }

    {
        hipEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 4; ++i) {
            hipError_t err = hipEventCreateWithFlags(&events[i], hipEventDefault);
            if (err != hipSuccess) {
                for (int j = 0; j < i; ++j) {
                    ps::gpu::discard_cleanup_result(hipEventDestroy(events[j]));
                }
                return Status::hip_error("create_dflash2_spec_decoder hipEventCreate",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
        decoder.draft_start_event = events[0];
        decoder.draft_stop_event = events[1];
        decoder.verify_start_event = events[2];
        decoder.verify_stop_event = events[3];
    }

    decoder.gdn_conv_bytes = spec_gdn_conv_bytes(gdn_pool);
    decoder.gdn_rec_bytes = spec_gdn_recurrent_bytes(gdn_pool);
    const char* rerun_env = std::getenv("PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE");
    decoder.gdn_rerun_reference =
        rerun_env != nullptr && rerun_env[0] != '\0' && rerun_env[0] != '0';
    if (decoder.gdn_rerun_reference) {
        if (decoder.gdn_conv_bytes != 0u) {
            auto conv_alloc = arena.allocate_aligned(decoder.gdn_conv_bytes, 256u);
            if (!conv_alloc.ok()) return conv_alloc.status();
            decoder.gdn_conv_snapshot = conv_alloc.release().data();
        }
        if (decoder.gdn_rec_bytes != 0u) {
            auto rec_alloc = arena.allocate_aligned(decoder.gdn_rec_bytes, 256u);
            if (!rec_alloc.ok()) return rec_alloc.status();
            decoder.gdn_rec_snapshot = rec_alloc.release().data();
        }
    } else {
        uint32_t history_rows = config.num_drafts;
        if (ngram_enabled) {
            history_rows = std::min<uint32_t>(
                kDFlash2SpecMaxVerifyDrafts, config.num_drafts + config.ngram_max_tail);
        }
        const GdnStatePoolDeviceView pool_view = gdn_pool.device_view();
        const std::size_t per_row_bytes =
            static_cast<std::size_t>(pool_view.conv_slot_stride) * sizeof(bf16_t) +
            static_cast<std::size_t>(pool_view.recurrent_slot_stride) * sizeof(float);
        const std::size_t history_bytes =
            static_cast<std::size_t>(history_rows) * per_row_bytes;
        if (history_bytes > kDFlash2SpecHistoryBytesMax) {
            return Status::insufficient_memory(
                "create_dflash2_spec_decoder: GDN history exceeds the 2.25 GiB guard",
                __FILE__, __LINE__);
        }
        auto history_result = create_gdn_spec_history(arena, gdn_pool, history_rows);
        if (!history_result.ok()) return history_result.status();
        decoder.gdn_history = history_result.release();
        decoder.gdn_history_enabled = true;
    }

    decoder.initialized = true;
    return decoder;
}

Status dflash2_spec_decoder_shutdown(DFlash2SpecDecoder& decoder) noexcept {
    Status first_error = Status::make_ok();
    auto destroy_event = [&](hipEvent_t& event, const char* tag) noexcept {
        if (event == nullptr) return;
        hipError_t err = hipEventDestroy(event);
        if (first_error.ok() && err != hipSuccess) {
            first_error =
                Status::hip_error(tag, hipGetErrorString(err), __FILE__, __LINE__);
        }
        event = nullptr;
    };
    destroy_event(decoder.draft_start_event, "hipEventDestroy draft_start");
    destroy_event(decoder.draft_stop_event, "hipEventDestroy draft_stop");
    destroy_event(decoder.verify_start_event, "hipEventDestroy verify_start");
    destroy_event(decoder.verify_stop_event, "hipEventDestroy verify_stop");
    decoder.verify_token_ids_device = nullptr;
    decoder.decision_staging_device = nullptr;
    decoder.device_token_bridge = false;
    decoder.host_proposal_visibility = false;
    decoder.token_history.clear();
    decoder.gdn_conv_snapshot = nullptr;
    decoder.gdn_rec_snapshot = nullptr;
    decoder.gdn_conv_bytes = 0u;
    decoder.gdn_rec_bytes = 0u;
    if (decoder.gdn_history_enabled || decoder.gdn_history.initialized) {
        (void)shutdown_gdn_spec_history(decoder.gdn_history);
    }
    decoder.gdn_history_enabled = false;
    decoder.gdn_rerun_reference = false;
    decoder.hidden_size = 0u;
    decoder.target = nullptr;
    decoder.draft = nullptr;
    decoder.context = nullptr;
    decoder.sequence = nullptr;
    decoder.gdn_pool = nullptr;
    decoder.initialized = false;
    return first_error;
}

Result<DFlash2PrefillOutput> dflash2_spec_prefill(
    DFlash2SpecDecoder& decoder,
    const int32_t* prompt_tokens,
    uint32_t prompt_count) {
    if (!decoder.initialized) {
        return Status::invalid_state("dflash2_spec_prefill: not initialized", __FILE__,
                                     __LINE__);
    }
    if (prompt_tokens == nullptr || prompt_count == 0u) {
        return Status::invalid_argument("dflash2_spec_prefill: empty prompt", __FILE__,
                                        __LINE__);
    }
    const uint32_t chunk_limit = decoder.target->config.max_scheduled_tokens;
    if (chunk_limit == 0u) {
        return Status::invalid_state("dflash2_spec_prefill: target has no token capacity",
                                     __FILE__, __LINE__);
    }
    const uint32_t tap_count = static_cast<uint32_t>(decoder.draft->config->num_target_layer_ids);
    if (tap_count == 0u || tap_count != ps::kernel::kDFlash2TargetTaps) {
        return Status::invalid_state("dflash2_spec_prefill: draft tap count mismatch", __FILE__,
                                     __LINE__);
    }

    DFlash2PrefillOutput out;
    std::vector<ScheduledRequest> requests;
    uint32_t offset = 0u;
    while (offset < prompt_count) {
        const uint32_t n = std::min(chunk_limit, prompt_count - offset);
        const bool final_chunk = (offset + n == prompt_count);
        ScheduledBatch batch = make_prefill_batch(
            *decoder.sequence, requests, prompt_tokens + offset, n, offset, final_chunk,
            decoder.config.verify_numeric_mode);
        auto executed = execute_batch(*decoder.target, batch, decoder.stream);
        if (!executed.ok()) return executed.status();
        BatchExecutionOutput output = executed.release();

        std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
        for (uint32_t t = 0; t < tap_count; ++t) {
            taps[t] = decoder.target->dflash_target_hidden[t].data<bf16_t>();
        }
        Status appended = dflash2::dflash2_append_target_taps(
            *decoder.draft, *decoder.context, taps.data(), tap_count, n, offset,
            decoder.stream);
        if (!appended.ok()) return appended;

        if (final_chunk) {
            const hipError_t err = hipMemcpy(
                &out.pending_token, output.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("dflash2_spec_prefill sample copy",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
        offset += n;
    }
    decoder.token_history.assign(prompt_tokens, prompt_tokens + prompt_count);
    decoder.token_history.push_back(out.pending_token);
    return out;
}

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

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

void commit_history_tokens(DFlash2SpecDecoder& decoder, const int32_t* tokens,
                           uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) {
        decoder.token_history.push_back(tokens[i]);
    }
    constexpr std::size_t kHistoryCap = 8192u;
    if (decoder.token_history.size() > kHistoryCap) {
        decoder.token_history.erase(decoder.token_history.begin(),
                                    decoder.token_history.end() - kHistoryCap / 2u);
    }
}

ScheduledBatch make_verify_batch(
    PagedSequenceState& sequence,
    std::vector<ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    ::ps::runtime::VerifyNumericMode numeric_mode,
    TokenIdsLocation token_location = TokenIdsLocation::Host) {
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
    batch.token_ids_location = token_location;
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

void collect_target_taps(
    const Executor& target,
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps>& taps) {
    for (uint32_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        taps[t] = target.dflash_target_hidden[t].data<bf16_t>();
    }
}

Status run_single_target(
    DFlash2SpecDecoder& decoder,
    int32_t token,
    DFlash2SpecIterationOutput& out) {
    DFlash2SpecTiming* tm = decoder.timing;
    const uint32_t position = decoder.sequence->position;
    std::vector<ScheduledRequest> requests;
    int32_t single = token;
    ScheduledBatch batch = make_verify_batch(
        *decoder.sequence, requests, &single, 1u, position, 1u, false,
        ::ps::runtime::VerifyNumericMode::Fast);
    auto executed = execute_batch(*decoder.target, batch, decoder.stream);
    if (!executed.ok()) return executed.status();
    BatchExecutionOutput output = executed.release();
    int32_t sampled = -1;
    const hipError_t err =
        hipMemcpy(&sampled, output.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                  hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 single decode sample copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
    collect_target_taps(*decoder.target, taps);
    Status st = dflash2::dflash2_append_target_taps(
        *decoder.draft, *decoder.context, taps.data(), ps::kernel::kDFlash2TargetTaps, 1u,
        position, decoder.stream);
    if (!st.ok()) return st;
    out.emitted[0] = sampled;
    out.emitted_count = 1u;
    out.pending_token = sampled;
    if (tm != nullptr) ++tm->generated_tokens;
    commit_history_tokens(decoder, out.emitted.data(), 1u);
    return Status::make_ok();
}

}  // namespace

Result<DFlash2SpecIterationOutput> dflash2_spec_step(
    DFlash2SpecDecoder& decoder,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    if (!decoder.initialized) {
        return Status::invalid_state("dflash2_spec_step: not initialized", __FILE__, __LINE__);
    }
    if (decoder.target == nullptr || decoder.draft == nullptr || decoder.context == nullptr ||
        decoder.sequence == nullptr || decoder.gdn_pool == nullptr) {
        return Status::invalid_state("dflash2_spec_step: state pointers are null", __FILE__,
                                     __LINE__);
    }
    if (remaining_tokens == 0u) {
        return Status::invalid_argument("dflash2_spec_step: remaining_tokens is zero", __FILE__,
                                        __LINE__);
    }

    DFlash2SpecIterationOutput out;
    DFlash2SpecTiming* tm = decoder.timing;
    ScopedTimer round_timer(tm != nullptr ? &tm->round_ms : nullptr);
    if (tm != nullptr) ++tm->rounds;

    const DFlash2TargetSnapshot snapshot{
        decoder.sequence->position,
        static_cast<uint32_t>(decoder.sequence->block_table.size()), true};
    const uint32_t position = snapshot.position;
    const uint32_t room = decoder.sequence->max_seq_len - position;
    uint32_t room_max = 0u;
    if (remaining_tokens > 1u && room >= 2u) {
        room_max = std::min(remaining_tokens - 1u, room - 1u);
    }
    const uint32_t dflash_k = std::min(decoder.config.num_drafts, room_max);
    const bool ngram_enabled =
        decoder.config.ngram_n > 0u && decoder.config.ngram_max_tail > 0u;

    if (dflash_k == 0u) {
        Status st = run_single_target(decoder, pending_token, out);
        if (!st.ok()) return st;
        if (decoder.config.eos_token >= 0) {
            if (out.pending_token == decoder.config.eos_token) out.finished = true;
        }
        return out;
    }

    NgramTailConfig ngram_cfg;
    ngram_cfg.n = decoder.config.ngram_n;
    ngram_cfg.max_tail = decoder.config.ngram_max_tail;
    ngram_cfg.window = decoder.config.ngram_window;

    const bool bridge = decoder.device_token_bridge &&
                        decoder.verify_token_ids_device != nullptr &&
                        decoder.decision_staging_device != nullptr;
    const bool want_timing = tm != nullptr;
    int32_t* proposal_device = decoder.draft->proposal_tokens.data<int32_t>();
    std::array<int32_t, kDFlash2SpecMaxVerifyDrafts> drafts{};
    std::array<int32_t, kDFlash2SpecMaxVerifyDrafts> tail_tokens{};
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> sampled{};
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> verify_host{};
    std::array<int32_t, 2u * kDFlash2SpecMaxVerifyRows> decision_host{};
    NgramTailMatch ngram_match;
    uint32_t tail_k = 0u;
    bool draft_events_recorded = false;
    bool verify_events_recorded = false;

    {
        ScopedTimer t(want_timing ? &tm->draft_ms : nullptr);
        if (bridge) {
            const hipError_t anchor_err = hipMemcpyAsync(
                decoder.verify_token_ids_device, &pending_token, sizeof(int32_t),
                hipMemcpyHostToDevice, decoder.stream);
            if (anchor_err != hipSuccess) {
                return Status::hip_error("dflash2 anchor upload",
                                         hipGetErrorString(anchor_err), __FILE__, __LINE__);
            }
        }
        if (want_timing) {
            const hipError_t start_err =
                hipEventRecord(decoder.draft_start_event, decoder.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("dflash2 draft start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
            draft_events_recorded = true;
        }
        Status st = dflash2::dflash2_propose_cached(
            *decoder.draft, *decoder.context, pending_token, dflash_k, proposal_device,
            decoder.stream);
        if (!st.ok()) return st;
        if (bridge) {
            const hipError_t relay_err = hipMemcpyAsync(
                decoder.verify_token_ids_device + 1, proposal_device,
                static_cast<std::size_t>(dflash_k) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, decoder.stream);
            if (relay_err != hipSuccess) {
                return Status::hip_error("dflash2 draft relay",
                                         hipGetErrorString(relay_err), __FILE__, __LINE__);
            }
        }
        if (want_timing) {
            const hipError_t stop_err =
                hipEventRecord(decoder.draft_stop_event, decoder.stream);
            if (stop_err != hipSuccess) {
                return Status::hip_error("dflash2 draft stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
    }

    const bool need_host_seed = ngram_enabled || decoder.host_proposal_visibility;
    if (!bridge || need_host_seed) {
        ScopedTimer t(want_timing ? &tm->draft_d2h_ms : nullptr);
        {
            ScopedTimer tw(want_timing ? (bridge ? &tm->ngram_seed_wait_ms
                                                 : &tm->proposal_wait_ms)
                                       : nullptr);
            if (hipStreamSynchronize(decoder.stream) != hipSuccess) {
                return Status::hip_error(
                    "dflash2 draft sync", hipGetErrorString(hipGetLastError()), __FILE__,
                    __LINE__);
            }
        }
        {
            ScopedTimer tc(want_timing ? &tm->proposal_copy_ms : nullptr);
            const hipError_t err = hipMemcpy(
                drafts.data(), proposal_device,
                static_cast<std::size_t>(dflash_k) * sizeof(int32_t),
                hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                return Status::hip_error("dflash2 draft copy", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
        }
    }

    if (ngram_enabled) {
        {
            ScopedTimer t(want_timing ? &tm->ngram_lookup_ms : nullptr);
            Status st = propose_ngram_tail_into(
                std::span<const int32_t>(decoder.token_history),
                std::span<const int32_t>(drafts.data(), dflash_k), ngram_cfg,
                std::span<int32_t>(tail_tokens.data(), tail_tokens.size()), ngram_match);
            if (!st.ok()) return st;
        }
        const uint32_t max_total = std::min(room_max, kDFlash2SpecMaxVerifyDrafts);
        uint32_t proposed = ngram_match.count;
        if (dflash_k + proposed > max_total) {
            proposed = max_total - dflash_k;
        }
        if (proposed == 0u) {
            ngram_match = NgramTailMatch{};
            tail_k = 0u;
        } else {
            ngram_match.count = proposed;
            tail_k = proposed;
            for (uint32_t i = 0u; i < tail_k; ++i) {
                drafts[dflash_k + i] = tail_tokens[i];
            }
            if (bridge) {
                ScopedTimer t(want_timing ? &tm->ngram_tail_h2d_ms : nullptr);
                const hipError_t err = hipMemcpyAsync(
                    decoder.verify_token_ids_device + 1u + dflash_k, tail_tokens.data(),
                    static_cast<std::size_t>(tail_k) * sizeof(int32_t),
                    hipMemcpyHostToDevice, decoder.stream);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 ngram tail upload",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        }
    }

    out.ngram = ngram_match;
    out.num_dflash_drafts = dflash_k;
    out.num_tail_drafts = tail_k;
    const uint32_t total_k = dflash_k + tail_k;
    out.num_drafts = total_k;

    if (decoder.gdn_rerun_reference) {
        ScopedTimer t(tm != nullptr ? &tm->gdn_snapshot_ms : nullptr);
        Status st = spec_gdn_snapshot(*decoder.gdn_pool, decoder.sequence->slot,
                                      decoder.gdn_conv_snapshot, decoder.gdn_rec_snapshot,
                                      decoder.gdn_conv_bytes, decoder.gdn_rec_bytes,
                                      decoder.stream);
        if (!st.ok()) return st;
    }

    std::vector<ScheduledRequest> requests;
    {
        ScopedTimer t(want_timing ? &tm->verify_ms : nullptr);
        ScheduledBatch batch;
        if (bridge) {
            batch = make_verify_batch(
                *decoder.sequence, requests, decoder.verify_token_ids_device, total_k + 1u,
                position, total_k + 1u, true, decoder.config.verify_numeric_mode,
                TokenIdsLocation::Device);
        } else {
            verify_host[0] = pending_token;
            for (uint32_t i = 0; i < total_k; ++i) verify_host[i + 1u] = drafts[i];
            batch = make_verify_batch(
                *decoder.sequence, requests, verify_host.data(), total_k + 1u, position,
                total_k + 1u, true, decoder.config.verify_numeric_mode);
        }
        ExecuteBatchOptions options;
        if (decoder.gdn_history_enabled) {
            options.gdn_spec_history = gdn_spec_history_view(decoder.gdn_history, total_k);
        }
        if (want_timing) {
            const hipError_t start_err =
                hipEventRecord(decoder.verify_start_event, decoder.stream);
            if (start_err != hipSuccess) {
                return Status::hip_error("dflash2 verify start event",
                                         hipGetErrorString(start_err), __FILE__, __LINE__);
            }
            verify_events_recorded = true;
        }
        auto submitted = submit_batch(*decoder.target, batch, decoder.stream, options);
        if (!submitted.ok()) return submitted.status();
        PendingBatch pending = submitted.release();
        if (want_timing) {
            const hipError_t stop_err =
                hipEventRecord(decoder.verify_stop_event, decoder.stream);
            if (stop_err != hipSuccess) {
                return Status::hip_error("dflash2 verify stop event",
                                         hipGetErrorString(stop_err), __FILE__, __LINE__);
            }
        }
        if (bridge) {
            const int32_t* sampled_device = pending.output.sampled_tokens.data<int32_t>();
            hipError_t d_err = hipMemcpyAsync(
                decoder.decision_staging_device, proposal_device,
                static_cast<std::size_t>(dflash_k) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, decoder.stream);
            if (d_err != hipSuccess) {
                return Status::hip_error("dflash2 decision draft relay",
                                         hipGetErrorString(d_err), __FILE__, __LINE__);
            }
            d_err = hipMemcpyAsync(
                decoder.decision_staging_device + kDFlash2SpecMaxVerifyRows,
                sampled_device, static_cast<std::size_t>(total_k + 1u) * sizeof(int32_t),
                hipMemcpyDeviceToDevice, decoder.stream);
            if (d_err != hipSuccess) {
                return Status::hip_error("dflash2 decision sample relay",
                                         hipGetErrorString(d_err), __FILE__, __LINE__);
            }
            {
                ScopedTimer tw(want_timing ? &tm->decision_wait_ms : nullptr);
                if (hipStreamSynchronize(decoder.stream) != hipSuccess) {
                    return Status::hip_error(
                        "dflash2 decision sync", hipGetErrorString(hipGetLastError()),
                        __FILE__, __LINE__);
                }
            }
            {
                ScopedTimer tc(want_timing ? &tm->decision_copy_ms : nullptr);
                const hipError_t err = hipMemcpy(
                    decision_host.data(), decoder.decision_staging_device,
                    decision_host.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 decision copy",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
            for (uint32_t i = 0; i < dflash_k; ++i) drafts[i] = decision_host[i];
            for (uint32_t i = 0; i <= total_k; ++i) {
                sampled[i] = decision_host[kDFlash2SpecMaxVerifyRows + i];
            }
            auto completed =
                complete_batch(*decoder.target, std::move(pending), decoder.stream);
            if (!completed.ok()) return completed.status();
        } else {
            BatchExecutionOutput output;
            {
                ScopedTimer tw(want_timing ? &tm->decision_wait_ms : nullptr);
                auto completed =
                    complete_batch(*decoder.target, std::move(pending), decoder.stream);
                if (!completed.ok()) return completed.status();
                output = completed.release();
            }
            {
                ScopedTimer tc(want_timing ? &tm->decision_copy_ms : nullptr);
                const hipError_t err = hipMemcpy(
                    sampled.data(), output.sampled_tokens.data<int32_t>(),
                    static_cast<std::size_t>(total_k + 1u) * sizeof(int32_t),
                    hipMemcpyDeviceToHost);
                if (err != hipSuccess) {
                    return Status::hip_error("dflash2 verify sample copy",
                                             hipGetErrorString(err), __FILE__, __LINE__);
                }
            }
        }
    }

    if (want_timing && draft_events_recorded) {
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, decoder.draft_start_event,
                                decoder.draft_stop_event) == hipSuccess) {
            tm->draft_gpu_ms += static_cast<double>(elapsed);
        }
    }
    if (want_timing && verify_events_recorded) {
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, decoder.verify_start_event,
                                decoder.verify_stop_event) == hipSuccess) {
            tm->verify_gpu_ms += static_cast<double>(elapsed);
        }
    }

    SpecVerifyResult result;
    Status accept = spec_greedy_accept(drafts.data(), total_k, sampled.data(), true, result);
    if (!accept.ok()) return accept;
    const uint32_t accepted = result.num_accepted_drafts;
    out.num_accepted = accepted;
    out.emitted_count = std::min(static_cast<uint32_t>(result.emitted_tokens.size()),
                                 remaining_tokens);
    for (uint32_t i = 0; i < out.emitted_count; ++i) {
        out.emitted[i] = result.emitted_tokens[i];
    }
    if (result.correction_valid) {
        out.pending_token = result.correction_token;
    } else if (result.bonus_token_valid) {
        out.pending_token = result.bonus_token;
    } else {
        out.pending_token = -1;
    }

    const uint32_t page_tokens = decoder.sequence->kv_pool()->page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_state("dflash2_spec_step: page_tokens is zero", __FILE__,
                                     __LINE__);
    }

    if (accepted == total_k) {
        ScopedTimer t(tm != nullptr ? &tm->dflash_commit_ms : nullptr);
        std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
        collect_target_taps(*decoder.target, taps);
        Status st = dflash2::dflash2_append_target_taps(
            *decoder.draft, *decoder.context, taps.data(), ps::kernel::kDFlash2TargetTaps,
            total_k + 1u, position, decoder.stream);
        if (!st.ok()) return st;
        const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
        st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
        if (!st.ok()) return st;
        if (tm != nullptr) ++tm->full_accepts;
    } else if (decoder.gdn_history_enabled) {
        {
            ScopedTimer t(tm != nullptr ? &tm->gdn_restore_ms : nullptr);
            Status st = restore_gdn_spec_history(*decoder.gdn_pool, decoder.sequence->slot,
                                                 decoder.gdn_history, accepted, decoder.stream);
            if (!st.ok()) return st;
        }
        decoder.sequence->position = position + accepted + 1u;
        {
            ScopedTimer t(tm != nullptr ? &tm->dflash_commit_ms : nullptr);
            std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
            collect_target_taps(*decoder.target, taps);
            Status st = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(), ps::kernel::kDFlash2TargetTaps,
                accepted + 1u, position, decoder.stream);
            if (!st.ok()) return st;
            const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        }
        if (tm != nullptr) ++tm->partial_accepts;
    } else {
        {
            ScopedTimer t(tm != nullptr ? &tm->gdn_restore_ms : nullptr);
            Status st = spec_gdn_restore(*decoder.gdn_pool, decoder.sequence->slot,
                                         decoder.gdn_conv_snapshot, decoder.gdn_rec_snapshot,
                                         decoder.gdn_conv_bytes, decoder.gdn_rec_bytes,
                                         decoder.stream);
            if (!st.ok()) return st;
        }
        decoder.sequence->position = position;

        std::array<int32_t, kDFlash2SpecMaxVerifyRows> prefix{};
        prefix[0] = pending_token;
        for (uint32_t i = 0; i < accepted; ++i) prefix[i + 1u] = drafts[i];
        const uint32_t n = accepted + 1u;
        {
            ScopedTimer t(tm != nullptr ? &tm->rerun_ms : nullptr);
            std::vector<ScheduledRequest> requests;
            ScheduledBatch batch = make_verify_batch(
                *decoder.sequence, requests, prefix.data(), n, position, 1u, true,
                decoder.config.verify_numeric_mode);
            auto executed = execute_batch(*decoder.target, batch, decoder.stream);
            if (!executed.ok()) return executed.status();
        }
        {
            ScopedTimer t(tm != nullptr ? &tm->dflash_commit_ms : nullptr);
            std::array<const bf16_t*, ps::kernel::kDFlash2TargetTaps> taps{};
            collect_target_taps(*decoder.target, taps);
            Status st = dflash2::dflash2_append_target_taps(
                *decoder.draft, *decoder.context, taps.data(), ps::kernel::kDFlash2TargetTaps,
                n, position, decoder.stream);
            if (!st.ok()) return st;
            const uint32_t required = ceil_div_u32(decoder.sequence->position, page_tokens);
            st = rollback_sequence_append(*decoder.sequence, required, decoder.stream);
            if (!st.ok()) return st;
        }
        out.rerun = true;
        if (tm != nullptr) ++tm->reruns;
    }

    if (tm != nullptr) {
        tm->accepted_drafts += accepted;
        tm->generated_tokens += out.emitted_count;
        tm->verify_rows_total += total_k + 1u;
        if (accepted >= dflash_k) ++tm->dflash_prefix_full_accepts;
        if (ngram_enabled && tail_k > 0u) {
            ++tm->ngram_hit_rounds;
            tm->ngram_proposed_tokens += tail_k;
            const uint32_t tail_accepted = accepted > dflash_k ? accepted - dflash_k : 0u;
            tm->ngram_accepted_tokens += tail_accepted;
            if (accepted >= dflash_k) {
                ++tm->tail_reached_rounds;
            } else {
                ++tm->tail_blocked_rounds;
            }
        }
        tm->gdn_history_bytes = decoder.gdn_history_enabled
                                    ? static_cast<uint64_t>(decoder.gdn_history.rows) *
                                          static_cast<uint64_t>(
                                              decoder.gdn_history.conv_state_bytes +
                                              decoder.gdn_history.recurrent_state_bytes)
                                    : 0u;
    }

    if (decoder.config.eos_token >= 0) {
        for (uint32_t i = 0; i < out.emitted_count; ++i) {
            if (out.emitted[i] == decoder.config.eos_token) {
                out.emitted_count = i + 1u;
                out.finished = true;
                break;
            }
        }
        if (out.pending_token == decoder.config.eos_token) out.finished = true;
    }
    commit_history_tokens(decoder, out.emitted.data(), out.emitted_count);
    return out;
}

}  // namespace ps::qwen35::runtime
