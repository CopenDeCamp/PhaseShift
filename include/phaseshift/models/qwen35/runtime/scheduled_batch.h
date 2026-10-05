#pragma once

#include <phaseshift/models/qwen35/state/paged_types.h>
#include <phaseshift/models/qwen35/runtime/sampling_params.h>
#include <phaseshift/runtime/execution/execution_types.h>
#include <phaseshift/runtime/request_handle.h>
#include <phaseshift/core/status.h>
#include <cstdint>
#include <span>

namespace ps {
namespace qwen35 {

struct PagedSequenceState;

struct ScheduledRequest {
    PagedSequenceState* sequence = nullptr;

    ::ps::runtime::RequestHandle handle{};

    ::ps::runtime::ExecutionClass execution_class = ::ps::runtime::ExecutionClass::DECODE;

    uint32_t token_begin = 0;
    uint32_t num_tokens = 0;

    uint32_t prefix_tokens = 0;

    bool compute_logits = false;
    bool sample = false;

    runtime::SamplingConfig sampling{};
    uint64_t sampling_index = 0;

    uint32_t num_output_rows = 1;
};

enum class TokenIdsLocation : uint8_t {
    Host,
    Device,
};

struct ScheduledBatch {
    const int32_t* token_ids = nullptr;
    TokenIdsLocation token_ids_location = TokenIdsLocation::Host;

    std::span<const ScheduledRequest> requests;

    uint32_t num_tokens = 0;
    uint32_t num_requests = 0;

    uint32_t num_decode_requests = 0;
    uint32_t num_verify_requests = 0;
    uint32_t num_prefill_requests = 0;

    bool speculative_verify = false;
    ::ps::runtime::VerifyNumericMode verify_numeric_mode =
        ::ps::runtime::VerifyNumericMode::Fast;
};

Status validate_scheduled_batch(
    const ScheduledBatch& batch,
    uint32_t max_tokens,
    uint32_t max_requests);

}
}
