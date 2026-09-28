#pragma once

#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps::qwen35::runtime {

enum class SpecPhase : uint8_t {
    IDLE = 0u,
    DRAFTING = 1u,
    VERIFYING = 2u,
    COMMITTING = 3u,
    ROLLING_BACK = 4u,
};

const char* spec_phase_name(SpecPhase phase) noexcept;

struct SpecVerifyResult {
    uint32_t num_drafts = 0u;
    uint32_t num_accepted_drafts = 0u;
    uint32_t first_reject_index = 0xFFFFFFFFu;
    bool correction_valid = false;
    int32_t correction_token = -1;
    bool bonus_token_valid = false;
    int32_t bonus_token = -1;
    uint32_t num_emitted_tokens = 0u;
    std::vector<int32_t> emitted_tokens;
    uint32_t num_committed_target_tokens = 0u;
    uint32_t num_mtp_drafts = 0xFFFFFFFFu;
};

Status spec_greedy_accept(
    const int32_t* drafts,
    uint32_t num_drafts,
    const int32_t* target_tokens,
    bool bonus_token_enabled,
    SpecVerifyResult& out);

struct SpecMtpSnapshot {
    uint32_t logical_length = 0u;
    uint32_t step_index = 0u;
    bool valid = false;
};

struct SpecTargetSnapshot {
    uint32_t position = 0u;
    uint32_t block_count = 0u;
    bool valid = false;
};

struct SpecTransaction {
    SpecPhase phase = SpecPhase::IDLE;
    SpecMtpSnapshot mtp_before{};
    SpecTargetSnapshot target_before{};
    uint32_t num_drafts = 0u;
    bool bonus_token_enabled = true;
};

Status spec_transaction_begin(
    SpecTransaction& transaction,
    const MtpKvState& mtp,
    const PagedSequenceState& target);

Status spec_transaction_begin_verify(SpecTransaction& transaction);

Status spec_transaction_commit(
    SpecTransaction& transaction,
    const SpecVerifyResult& result,
    MtpKvState& mtp,
    PagedSequenceState& target,
    hipStream_t stream);

Status spec_transaction_rollback(
    SpecTransaction& transaction,
    MtpKvState& mtp,
    PagedSequenceState& target,
    hipStream_t stream);

Status spec_transaction_abort(
    SpecTransaction& transaction,
    MtpKvState& mtp,
    PagedSequenceState& target,
    hipStream_t stream);

std::size_t spec_gdn_conv_bytes(const GdnStatePool& pool) noexcept;

std::size_t spec_gdn_recurrent_bytes(const GdnStatePool& pool) noexcept;

Status spec_gdn_snapshot(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    void* conv_dst,
    void* recurrent_dst,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream);

Status spec_gdn_restore(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const void* conv_src,
    const void* recurrent_src,
    std::size_t conv_bytes,
    std::size_t recurrent_bytes,
    hipStream_t stream);

struct SpecDraftStep {
    int32_t input_token = -1;
    uint32_t absolute_position = 0u;
    uint32_t kv_length_before = 0u;
    uint32_t kv_length_after = 0u;
    uint32_t draft_token = 0u;
    uint32_t physical_slot = 0xFFFFFFFFu;
    float margin = 0.0f;
};

struct SpecDraftSet {
    std::vector<SpecDraftStep> steps;
    uint32_t mtp_length_before = 0u;
    uint32_t mtp_length_after = 0u;
};

struct MtpDraftPolicy {
    bool dynamic = false;
    float stop_margin = 0.0f;
    uint32_t min_drafts = 1u;
    bool enable_discard = false;
    float discard_margin = 0.0f;
};

Result<SpecDraftSet> mtp_generate_drafts(
    MtpExecutor& executor,
    MtpKvState& state,
    const bf16_t* first_hidden,
    int32_t first_token,
    uint32_t num_drafts,
    uint32_t first_absolute_position,
    hipStream_t stream,
    MtpStateTraceSink* state_trace = nullptr,
    const MtpDraftPolicy* policy = nullptr);

}  // namespace ps::qwen35::runtime
