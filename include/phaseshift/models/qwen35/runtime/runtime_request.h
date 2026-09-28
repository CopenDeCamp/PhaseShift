#pragma once

#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/runtime/sampling_params.h>
#include <phaseshift/models/qwen35/runtime/token_constraint.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ps {
namespace qwen35 {

namespace runtime {

enum class RequestState : uint8_t {
    Queued,
    Active,
    Finished,
};

enum class FinishReason : uint8_t {
    None,
    Eos,
    MaxNewTokens,
    Cancelled,
    Error,
};

struct RuntimeRequest {
    uint64_t id = 0;

    std::vector<int32_t> input_tokens;
    uint32_t max_new_tokens = 0;

    SamplingConfig sampling{};

    uint32_t max_kv_tokens = 0;
    uint32_t max_kv_pages = 0;

    RequestState state = RequestState::Queued;
    FinishReason finish_reason = FinishReason::None;

    PagedSequenceState sequence;
    int32_t pending_decode_token = -1;

    uint32_t generated_tokens = 0;
    std::vector<int32_t> generated;

    uint32_t restored_tokens = 0;

    uint32_t prefix_cache_checkpoint_position = 0;
    bool prefix_cache_checkpoint_saved = false;

    std::unique_ptr<TokenConstraintState> constraint;
};

Status admit_request(
    RuntimeRequest& request,
    SequenceSlotPool& seq_pool,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    hipStream_t stream);

enum class CommitAction : uint8_t {
    Continue,
    Finished,
};

CommitAction commit_sampled_token(
    RuntimeRequest& request,
    int32_t token,
    int32_t eos_token_id);

CommitAction commit_final_prefill_without_sample(RuntimeRequest& request);

Status record_sampled_token(RuntimeRequest& request, int32_t token);

}
}
}
