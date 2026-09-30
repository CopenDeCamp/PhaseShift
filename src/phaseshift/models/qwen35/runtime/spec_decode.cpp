#include <phaseshift/models/qwen35/runtime/spec_decode.h>

#include <cstring>

namespace ps::qwen35::runtime {

namespace {

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

}  // namespace

const char* spec_phase_name(SpecPhase phase) noexcept {
    switch (phase) {
        case SpecPhase::IDLE: return "IDLE";
        case SpecPhase::DRAFTING: return "DRAFTING";
        case SpecPhase::VERIFYING: return "VERIFYING";
        case SpecPhase::COMMITTING: return "COMMITTING";
        case SpecPhase::ROLLING_BACK: return "ROLLING_BACK";
    }
    return "UNKNOWN";
}

Status spec_greedy_accept(
    const int32_t* drafts,
    uint32_t num_drafts,
    const int32_t* target_tokens,
    bool bonus_token_enabled,
    SpecVerifyResult& out) {

    if (target_tokens == nullptr) {
        return Status::invalid_argument("spec_greedy_accept target_tokens is null", __FILE__, __LINE__);
    }
    if (num_drafts > 0u && drafts == nullptr) {
        return Status::invalid_argument("spec_greedy_accept drafts is null", __FILE__, __LINE__);
    }

    out = SpecVerifyResult{};
    out.num_drafts = num_drafts;

    if (num_drafts == 0u) {
        out.first_reject_index = 0u;
        out.num_committed_target_tokens = 0u;
        return Status::make_ok();
    }

    uint32_t accepted = 0u;
    while (accepted < num_drafts && drafts[accepted] == target_tokens[accepted]) {
        ++accepted;
    }
    out.num_accepted_drafts = accepted;

    if (accepted == num_drafts) {
        out.first_reject_index = 0xFFFFFFFFu;
        if (bonus_token_enabled) {
            out.bonus_token_valid = true;
            out.bonus_token = target_tokens[num_drafts];
        }
    } else {
        out.first_reject_index = accepted;
        out.correction_valid = true;
        out.correction_token = target_tokens[accepted];
    }

    out.emitted_tokens.reserve(static_cast<std::size_t>(accepted) + 1u);
    for (uint32_t i = 0u; i < accepted; ++i) {
        out.emitted_tokens.push_back(drafts[i]);
    }
    if (out.correction_valid) {
        out.emitted_tokens.push_back(out.correction_token);
    } else if (out.bonus_token_valid) {
        out.emitted_tokens.push_back(out.bonus_token);
    }
    out.num_emitted_tokens = static_cast<uint32_t>(out.emitted_tokens.size());
    out.num_committed_target_tokens = accepted + 1u;
    return Status::make_ok();
}

Status spec_transaction_begin(
    SpecTransaction& transaction,
    const MtpKvState& mtp,
    const PagedSequenceState& target) {

    if (transaction.phase != SpecPhase::IDLE) {
        return Status::invalid_state(
            (std::string("spec transaction already active in phase ") +
             spec_phase_name(transaction.phase)).c_str(),
            __FILE__, __LINE__);
    }
    if (!mtp.is_initialized()) {
        return Status::invalid_state("spec transaction begin: MTP state not initialized",
                                     __FILE__, __LINE__);
    }
    if (!target.is_allocated()) {
        return Status::invalid_state("spec transaction begin: target sequence not allocated",
                                     __FILE__, __LINE__);
    }

    transaction.mtp_before.logical_length = mtp.logical_length;
    transaction.mtp_before.step_index = mtp.step_index;
    transaction.mtp_before.valid = true;
    transaction.target_before.position = target.position;
    transaction.target_before.block_count = static_cast<uint32_t>(target.block_table.size());
    transaction.target_before.valid = true;
    transaction.num_drafts = 0u;
    transaction.phase = SpecPhase::DRAFTING;
    return Status::make_ok();
}

Status spec_transaction_begin_verify(SpecTransaction& transaction) {
    if (transaction.phase != SpecPhase::DRAFTING) {
        return Status::invalid_state(
            (std::string("spec transaction begin_verify in phase ") +
             spec_phase_name(transaction.phase)).c_str(),
            __FILE__, __LINE__);
    }
    transaction.phase = SpecPhase::VERIFYING;
    return Status::make_ok();
}

Status spec_transaction_commit(
    SpecTransaction& transaction,
    const SpecVerifyResult& result,
    MtpKvState& mtp,
    PagedSequenceState& target,
    hipStream_t stream) {

    if (transaction.phase != SpecPhase::VERIFYING) {
        return Status::invalid_state(
            (std::string("spec transaction commit in phase ") +
             spec_phase_name(transaction.phase)).c_str(),
            __FILE__, __LINE__);
    }
    if (!transaction.mtp_before.valid || !transaction.target_before.valid) {
        return Status::invalid_state("spec transaction commit without snapshot", __FILE__, __LINE__);
    }
    if (!mtp.is_initialized()) {
        return Status::invalid_state("spec transaction commit: MTP state not initialized",
                                     __FILE__, __LINE__);
    }
    if (target.kv_pool() == nullptr) {
        return Status::invalid_state("spec transaction commit: target kv pool is null",
                                     __FILE__, __LINE__);
    }
    if (result.num_committed_target_tokens != result.num_accepted_drafts + 1u) {
        return Status::invalid_argument(
            "spec transaction commit: committed target tokens inconsistent",
            __FILE__, __LINE__);
    }

    transaction.phase = SpecPhase::COMMITTING;

    const uint32_t num_drafts = result.num_drafts;
    const uint32_t accepted = result.num_accepted_drafts;
    const uint32_t mtp_drafts =
        result.num_mtp_drafts < num_drafts ? result.num_mtp_drafts : num_drafts;
    uint32_t mtp_kept = 0u;
    if (mtp_drafts > 0u) {
        mtp_kept = (accepted + 1u < mtp_drafts) ? accepted + 1u : mtp_drafts;
    }
    mtp.logical_length = transaction.mtp_before.logical_length + mtp_kept;
    mtp.step_index = transaction.mtp_before.step_index + mtp_kept;

    const uint32_t new_position = transaction.target_before.position + result.num_committed_target_tokens;
    const uint32_t page_tokens = target.kv_pool()->page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_state("spec transaction commit: page_tokens is zero", __FILE__, __LINE__);
    }
    const uint32_t required_blocks = ceil_div_u32(new_position, page_tokens);
    const uint32_t current_blocks = static_cast<uint32_t>(target.block_table.size());
    if (required_blocks > current_blocks) {
        return Status::invalid_state(
            "spec transaction commit: required blocks exceed allocated blocks", __FILE__, __LINE__);
    }
    Status release = rollback_sequence_append(target, required_blocks, stream);
    if (!release.ok()) {
        return release;
    }
    target.position = new_position;

    transaction.phase = SpecPhase::IDLE;
    transaction.mtp_before = SpecMtpSnapshot{};
    transaction.target_before = SpecTargetSnapshot{};
    transaction.num_drafts = 0u;
    return Status::make_ok();
}

Status spec_transaction_rollback(
    SpecTransaction& transaction,
    MtpKvState& mtp,
    PagedSequenceState& target,
    hipStream_t stream) {

    if (transaction.phase == SpecPhase::IDLE) {
        return Status::invalid_state("spec transaction rollback in IDLE phase", __FILE__, __LINE__);
    }
    if (!transaction.mtp_before.valid || !transaction.target_before.valid) {
        return Status::invalid_state("spec transaction rollback without snapshot", __FILE__, __LINE__);
    }
    if (!mtp.is_initialized()) {
        return Status::invalid_state("spec transaction rollback: MTP state not initialized",
                                     __FILE__, __LINE__);
    }

    transaction.phase = SpecPhase::ROLLING_BACK;

    mtp.logical_length = transaction.mtp_before.logical_length;
    mtp.step_index = transaction.mtp_before.step_index;

    Status release = rollback_sequence_append(target, transaction.target_before.block_count, stream);
    if (!release.ok()) {
        return release;
    }
    target.position = transaction.target_before.position;

    transaction.phase = SpecPhase::IDLE;
    transaction.mtp_before = SpecMtpSnapshot{};
    transaction.target_before = SpecTargetSnapshot{};
    transaction.num_drafts = 0u;
    return Status::make_ok();
}

Status spec_transaction_abort(
    SpecTransaction& transaction,
    MtpKvState& mtp,
    PagedSequenceState& target,
    hipStream_t stream) {
    return spec_transaction_rollback(transaction, mtp, target, stream);
}

std::size_t spec_gdn_conv_bytes(const GdnStatePool& pool) noexcept {
    return static_cast<std::size_t>(pool.device_view().conv_slot_stride) * sizeof(bf16_t);
}

std::size_t spec_gdn_recurrent_bytes(const GdnStatePool& pool) noexcept {
    return static_cast<std::size_t>(pool.device_view().recurrent_slot_stride) * sizeof(float);
}

Status spec_gdn_snapshot(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    void* conv_dst,
    void* recurrent_dst,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream) {

    if (conv_dst == nullptr || recurrent_dst == nullptr) {
        return Status::invalid_argument("spec_gdn_snapshot outputs are null", __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("spec_gdn_snapshot slot out of range", __FILE__, __LINE__);
    }
    if (conv_bytes != spec_gdn_conv_bytes(pool) ||
        recurrent_bytes != spec_gdn_recurrent_bytes(pool)) {
        return Status::invalid_argument("spec_gdn_snapshot byte size mismatch", __FILE__, __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_src = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    float* rec_src = view.recurrent_base + static_cast<std::size_t>(slot) * view.recurrent_slot_stride;
    hipError_t err = hipMemcpyAsync(conv_dst, conv_src, conv_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_snapshot conv copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(recurrent_dst, rec_src, recurrent_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_snapshot recurrent copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status spec_gdn_restore(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const void* conv_src,
    const void* recurrent_src,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream) {

    if (conv_src == nullptr || recurrent_src == nullptr) {
        return Status::invalid_argument("spec_gdn_restore inputs are null", __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("spec_gdn_restore slot out of range", __FILE__, __LINE__);
    }
    if (conv_bytes != spec_gdn_conv_bytes(pool) ||
        recurrent_bytes != spec_gdn_recurrent_bytes(pool)) {
        return Status::invalid_argument("spec_gdn_restore byte size mismatch", __FILE__, __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_dst = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    float* rec_dst = view.recurrent_base + static_cast<std::size_t>(slot) * view.recurrent_slot_stride;
    hipError_t err = hipMemcpyAsync(conv_dst, conv_src, conv_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_restore conv copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(rec_dst, recurrent_src, recurrent_bytes, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("spec_gdn_restore recurrent copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Result<SpecDraftSet> mtp_generate_drafts(
    MtpExecutor& executor,
    MtpKvState& state,
    const bf16_t* first_hidden,
    int32_t first_token,
    uint32_t num_drafts,
    uint32_t first_absolute_position,
    hipStream_t stream,
    MtpStateTraceSink* state_trace,
    const MtpDraftPolicy* policy) {

    if (!state.is_initialized()) {        return Status::invalid_state("mtp_generate_drafts: MTP state not initialized",
                                     __FILE__, __LINE__);
    }
    if (first_hidden == nullptr) {
        return Status::invalid_argument("mtp_generate_drafts: first_hidden is null",
                                        __FILE__, __LINE__);
    }
    if (num_drafts == 0u) {
        return Status::invalid_argument("mtp_generate_drafts: num_drafts must be > 0",
                                        __FILE__, __LINE__);
    }
    if (num_drafts > executor.max_rows) {
        return Status::invalid_argument("mtp_generate_drafts: num_drafts exceeds max_rows",
                                        __FILE__, __LINE__);
    }
    const uint64_t after =
        static_cast<uint64_t>(state.logical_length) + static_cast<uint64_t>(num_drafts);
    if (after > static_cast<uint64_t>(state.capacity_tokens)) {
        return Status::out_of_range("mtp_generate_drafts: MTP capacity overflow", __FILE__, __LINE__);
    }

    SpecDraftSet drafts;
    drafts.mtp_length_before = state.logical_length;
    drafts.steps.reserve(num_drafts);

    executor.collect_top2 =
        policy != nullptr && (policy->dynamic || policy->enable_discard);

    const bf16_t* hidden = first_hidden;
    int32_t token = first_token;
    uint32_t position = first_absolute_position;

    for (uint32_t k = 0u; k < num_drafts; ++k) {
        const uint32_t before = state.logical_length;
        auto outcome = mtp_forward_step(
            executor, state, hidden, &token, 1u, position, stream, nullptr, state_trace);
        if (!outcome.ok()) {
            return outcome.status();
        }
        SpecDraftStep step{};
        step.input_token = token;
        step.absolute_position = position;
        step.kv_length_before = outcome.value().kv_length_before;
        step.kv_length_after = outcome.value().kv_length_after;
        step.draft_token = outcome.value().draft_token;
        step.margin = outcome.value().top1_logit - outcome.value().top2_logit;
        auto slot = mtp_kv_physical_slot(state, before);
        step.physical_slot = slot.ok() ? slot.value() : 0xFFFFFFFFu;
        drafts.steps.push_back(step);

        if (policy != nullptr && policy->enable_discard && drafts.steps.size() == 1u) {
            const float margin0 =
                outcome.value().top1_logit - outcome.value().top2_logit;
            if (margin0 < policy->discard_margin) {
                drafts.steps.clear();
                break;
            }
        }

        if (policy != nullptr && policy->dynamic) {
            const float margin =
                outcome.value().top1_logit - outcome.value().top2_logit;
            if (drafts.steps.size() >= policy->min_drafts && margin < policy->stop_margin) {
                break;
            }
        }

        const bf16_t* next_hidden =
            (policy != nullptr && policy->chain_post_norm)
                ? executor.hidden_out_normed.data<bf16_t>()
                : executor.hidden_out.data<bf16_t>();
        hidden = next_hidden;
        token = static_cast<int32_t>(outcome.value().draft_token);
        ++position;
    }

    drafts.mtp_length_after = state.logical_length;
    return drafts;
}

}  // namespace ps::qwen35::runtime
