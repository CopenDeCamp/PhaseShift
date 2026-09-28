#pragma once

#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/core/status.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ps {
namespace qwen35 {

namespace runtime {

struct SchedulePlan {
    std::vector<int32_t> token_ids;
    std::vector<ScheduledRequest> scheduled_requests;

    ScheduledBatch batch;

    bool empty() const noexcept { return batch.num_requests == 0; }

    std::span<const ScheduledRequest> requests() const noexcept {
        return batch.requests;
    }

    Status append_decode(
        PagedSequenceState* sequence,
        int32_t token,
        uint32_t prefix_tokens,
        bool compute_logits,
        bool sample,
        const runtime::SamplingConfig& sampling,
        uint64_t sampling_index) {

        if (sequence == nullptr) {
            return Status::invalid_argument(
                "SchedulePlan decode sequence is null", __FILE__, __LINE__);
        }

        if (prefill_started_) {
            return Status::invalid_state(
                "SchedulePlan cannot append DECODE after PREFILL",
                __FILE__,
                __LINE__);
        }

        const uint32_t token_begin = static_cast<uint32_t>(token_ids.size());

        token_ids.push_back(token);

        ScheduledRequest scheduled;
        scheduled.sequence = sequence;
        scheduled.handle = sequence->request_handle();
        scheduled.execution_class = ::ps::runtime::ExecutionClass::DECODE;
        scheduled.token_begin = token_begin;
        scheduled.num_tokens = 1;
        scheduled.prefix_tokens = prefix_tokens;
        scheduled.compute_logits = compute_logits;
        scheduled.sample = sample;
        scheduled.sampling = sampling;
        scheduled.sampling_index = sampling_index;

        scheduled_requests.push_back(scheduled);

        ++decode_count_;

        return Status::make_ok();
    }

    Status append_prefill(
        PagedSequenceState* sequence,
        std::span<const int32_t> tokens,
        uint32_t prefix_tokens,
        bool compute_logits,
        bool sample,
        const runtime::SamplingConfig& sampling,
        uint64_t sampling_index) {

        if (sequence == nullptr) {
            return Status::invalid_argument(
                "SchedulePlan prefill sequence is null", __FILE__, __LINE__);
        }

        if (tokens.empty()) {
            return Status::invalid_argument(
                "SchedulePlan PREFILL tokens are empty", __FILE__, __LINE__);
        }

        prefill_started_ = true;

        const uint32_t token_begin = static_cast<uint32_t>(token_ids.size());

        token_ids.insert(token_ids.end(), tokens.begin(), tokens.end());

        ScheduledRequest scheduled;
        scheduled.sequence = sequence;
        scheduled.handle = sequence->request_handle();
        scheduled.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        scheduled.token_begin = token_begin;
        scheduled.num_tokens = static_cast<uint32_t>(tokens.size());
        scheduled.prefix_tokens = prefix_tokens;
        scheduled.compute_logits = compute_logits;
        scheduled.sample = sample;
        scheduled.sampling = sampling;
        scheduled.sampling_index = sampling_index;

        scheduled_requests.push_back(scheduled);

        ++prefill_count_;

        return Status::make_ok();
    }

    void finalize() noexcept {
        batch.token_ids =
            token_ids.empty() ? nullptr : token_ids.data();

        batch.requests = std::span<const ScheduledRequest>(
            scheduled_requests.data(),
            scheduled_requests.size());

        batch.num_tokens = static_cast<uint32_t>(token_ids.size());
        batch.num_requests =
            static_cast<uint32_t>(scheduled_requests.size());

        batch.num_decode_requests = decode_count_;
        batch.num_verify_requests = 0;
        batch.num_prefill_requests = prefill_count_;
    }

 private:
    uint32_t decode_count_ = 0;
    uint32_t prefill_count_ = 0;
    bool prefill_started_ = false;
};


}
}
}
