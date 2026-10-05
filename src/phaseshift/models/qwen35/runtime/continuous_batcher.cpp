#include <phaseshift/models/qwen35/runtime/continuous_batcher.h>
#include <phaseshift/models/qwen35/runtime/runtime_request.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace ps {
namespace qwen35 {
namespace runtime {

ContinuousBatcher::ContinuousBatcher(
    Executor& executor,
    SequenceSlotPool& seq_pool,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    ContinuousBatcherConfig config,
    hipStream_t stream,
    PrefixCache* prefix_cache)
    : executor_(executor),
      seq_pool_(seq_pool),
      gdn_pool_(gdn_pool),
      kv_pool_(kv_pool),
      config_(config),
      stream_(stream),
      capacity_(kv_pool.num_pages()),
      prefix_cache_(prefix_cache) {
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens, uint32_t max_new_tokens) {
    return submit(std::move(input_tokens), max_new_tokens, SamplingConfig{});
}

Result<uint64_t> ContinuousBatcher::submit(
    std::vector<int32_t> input_tokens,
    uint32_t max_new_tokens,
    const SamplingConfig& sampling,
    uint32_t prefix_cache_checkpoint_position) {
    if (input_tokens.empty()) {
        return Status::invalid_argument("submit requires at least one input token",
                                        __FILE__, __LINE__);
    }
    if (prefix_cache_checkpoint_position > input_tokens.size()) {
        return Status::invalid_argument(
            "prefix cache checkpoint position exceeds input token count",
            __FILE__, __LINE__);
    }
    const Status sampling_status = validate_sampling_config(sampling);
    if (!sampling_status.ok()) {
        return sampling_status;
    }
    if (static_cast<uint64_t>(input_tokens.size()) + max_new_tokens > config_.max_seq_len) {
        return Status::out_of_range("input_tokens + max_new_tokens exceeds max_seq_len",
                                    __FILE__, __LINE__);
    }
    auto max_kv_tokens = compute_max_kv_tokens(
        static_cast<uint32_t>(input_tokens.size()), max_new_tokens);
    if (!max_kv_tokens.ok()) {
        return max_kv_tokens.status();
    }
    const uint32_t max_kv_pages =
        kv_pages_for_tokens(max_kv_tokens.value(), config_.page_tokens);
    if (max_kv_pages > kv_pool_.num_pages()) {
        return Status::out_of_range(
            "request max_kv_pages exceeds kv pool capacity", __FILE__, __LINE__);
    }
    RuntimeRequest request;
    request.id = next_id_++;
    request.input_tokens = std::move(input_tokens);
    request.max_new_tokens = max_new_tokens;
    request.sampling = sampling;
    request.max_kv_tokens = max_kv_tokens.value();
    request.max_kv_pages = max_kv_pages;
    request.prefix_cache_checkpoint_position = prefix_cache_checkpoint_position;

    requests_.push_back(std::move(request));
    request_index_[requests_.back().id] = std::prev(requests_.end());
    return requests_.back().id;
}

Status ContinuousBatcher::cancel(uint64_t id) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return Status::invalid_argument("unknown request id", __FILE__, __LINE__);
    }
    RuntimeRequest& request = *it->second;
    if (request.state == RequestState::Finished) {
        return Status::make_ok();
    }

    return finish_request(request, FinishReason::Cancelled, stream_);
}

Status ContinuousBatcher::retire(uint64_t id) {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return Status::invalid_argument("unknown request id", __FILE__, __LINE__);
    }
    if (it->second->state != RequestState::Finished) {
        return Status::invalid_state("cannot retire a non-finished request",
                                     __FILE__, __LINE__);
    }
    requests_.erase(it->second);
    request_index_.erase(it);
    return Status::make_ok();
}

Status ContinuousBatcher::cancel_all() {
    Status first_error = Status::make_ok();
    for (auto& request : requests_) {
        if (request.state == RequestState::Queued || request.state == RequestState::Active) {
            auto st = cancel(request.id);
            if (!st.ok() && first_error.ok()) {
                first_error = st;
            }
        }
    }
    return first_error;
}

const RuntimeRequest* ContinuousBatcher::request(uint64_t id) const {
    auto it = request_index_.find(id);
    if (it == request_index_.end()) {
        return nullptr;
    }
    return &*it->second;
}

std::size_t ContinuousBatcher::queued_count() const noexcept {
    std::size_t n = 0;
    for (const auto& r : requests_) {
        if (r.state == RequestState::Queued) ++n;
    }
    return n;
}

std::size_t ContinuousBatcher::active_count() const noexcept {
    std::size_t n = 0;
    for (const auto& r : requests_) {
        if (r.state == RequestState::Active) ++n;
    }
    return n;
}

std::size_t ContinuousBatcher::finished_count() const noexcept {
    std::size_t n = 0;
    for (const auto& r : requests_) {
        if (r.state == RequestState::Finished) ++n;
    }
    return n;
}

bool ContinuousBatcher::has_pending() const noexcept {
    return queued_count() + active_count() > 0;
}

Status ContinuousBatcher::finish_request(
    RuntimeRequest& request, FinishReason reason, hipStream_t stream) {
    request.state = RequestState::Finished;
    request.finish_reason = reason;
    request.pending_decode_token = -1;
    if (!request.sequence.is_allocated()) {
        return Status::make_ok();
    }
    Status save_st = Status::make_ok();
    const bool explicit_checkpoint = request.prefix_cache_checkpoint_position > 0;
    const bool cacheable =
        !explicit_checkpoint &&
        (reason == FinishReason::Eos || reason == FinishReason::MaxNewTokens);
    if (cacheable && prefix_cache_ != nullptr && prefix_cache_->enabled() &&
        request.sequence.is_usable() && request.sequence.position > 0) {
        std::vector<int32_t> tokens = request.input_tokens;
        const std::size_t base = tokens.size();
        if (request.sequence.position > base) {
            const std::size_t extra = request.sequence.position - base;
            for (std::size_t i = 0; i < extra && i < request.generated.size(); ++i) {
                tokens.push_back(request.generated[i]);
            }
        }
        if (tokens.size() == request.sequence.position) {
            save_st = prefix_cache_->save(
                request.sequence, gdn_pool_, kv_pool_, std::move(tokens), stream);
        }
    }
    Status mirror_release = Status::make_ok();
    if (tp_batch_hook_ != nullptr) {
        mirror_release = tp_batch_hook_->on_sequence_released(request.sequence);
    }
    auto release_st = release_paged_sequence_state(request.sequence, stream);
    if (!release_st.ok()) {
        return release_st;
    }
    if (!mirror_release.ok()) {
        return mirror_release;
    }
    auto claim_st = capacity_.release_claim(request.id);
    if (!claim_st.ok()) {
        return claim_st;
    }
    return save_st;
}

KVBankerState ContinuousBatcher::build_kv_banker_state() const {
    KVBankerState state;
    state.available_pages = kv_pool_.num_free_pages();
    for (const auto& request : requests_) {
        if (request.state != RequestState::Active) {
            continue;
        }
        state.processes.push_back({
            request.max_kv_pages,
            static_cast<uint32_t>(request.sequence.block_table.size())});
    }
    return state;
}

KVCapacitySnapshot ContinuousBatcher::kv_capacity_snapshot() const {
    KVCapacitySnapshot snapshot;
    snapshot.total_pages = kv_pool_.num_pages();
    snapshot.physical_used_pages = kv_pool_.num_used_pages();
    snapshot.physical_free_pages = kv_pool_.num_free_pages();
    const KVBankerState state = build_kv_banker_state();
    snapshot.active_requests = static_cast<uint32_t>(state.processes.size());
    uint64_t max_claim = 0;
    uint64_t remaining_need = 0;
    for (const auto& process : state.processes) {
        max_claim += process.max_pages;
        remaining_need += process.max_pages - process.allocated_pages;
    }
    snapshot.max_claim_pages = max_claim;
    snapshot.remaining_need_pages = remaining_need;
    const uint64_t total = snapshot.total_pages;
    snapshot.conservative_headroom_pages = max_claim > total ? 0 : total - max_claim;
    snapshot.overcommit_pages = max_claim > total ? max_claim - total : 0;
    snapshot.banker_safe = is_safe(state);
    return snapshot;
}

Result<StepResult> ContinuousBatcher::step() {
    const bool banker_mode = config_.admission_policy == KVAdmissionPolicy::BankerSafe;
    KVBankerState kv_state = build_kv_banker_state();
    if (banker_mode && !is_safe(kv_state)) {
        return Status::invalid_state(
            "banker invariant violated: active KV state is unsafe", __FILE__, __LINE__);
    }

    for (auto& request : requests_) {
        if (request.state != RequestState::Queued) {
            continue;
        }
        if (seq_pool_.num_free_slots() == 0) {
            break;
        }
        const bool kv_admit_ok = banker_mode
            ? can_add_process(kv_state, request.max_kv_pages)
            : capacity_.can_admit(request.max_kv_pages);
        if (!kv_admit_ok) {
            break;
        }
        auto admit_st = admit_request(request, seq_pool_, gdn_pool_, kv_pool_, stream_);
        if (!admit_st.ok()) {
            break;
        }
        if (tp_batch_hook_ != nullptr) {
            auto mirror_st = tp_batch_hook_->on_sequence_created(request.sequence);
            if (!mirror_st.ok()) {
                (void)release_paged_sequence_state(request.sequence, stream_);
                request.state = RequestState::Queued;
                return mirror_st;
            }
        }
        if (prefix_cache_ != nullptr && prefix_cache_->enabled()) {
            const PrefixCheckpoint* checkpoint =
                prefix_cache_->find_longest(request.input_tokens, true);
            if (checkpoint != nullptr &&
                checkpoint->tokens.size() < request.input_tokens.size()) {
                auto restore_st = prefix_cache_->restore(
                    request.sequence, gdn_pool_, kv_pool_, *checkpoint, stream_);
                if (!restore_st.ok()) {
                    if (tp_batch_hook_ != nullptr) {
                        (void)tp_batch_hook_->on_sequence_released(request.sequence);
                    }
                    (void)release_paged_sequence_state(request.sequence, stream_);
                    request.state = RequestState::Queued;
                    return restore_st;
                }
                request.restored_tokens = checkpoint->position;
                if (request.restored_tokens >=
                    request.prefix_cache_checkpoint_position) {
                    request.prefix_cache_checkpoint_saved = true;
                }
            }
        }
        auto claim_st = capacity_.register_claim(request.id, request.max_kv_pages);
        if (!claim_st.ok()) {
            if (tp_batch_hook_ != nullptr) {
                (void)tp_batch_hook_->on_sequence_released(request.sequence);
            }
            (void)release_paged_sequence_state(request.sequence, stream_);
            request.state = RequestState::Queued;
            return claim_st;
        }
        kv_state.processes.push_back(
            {request.max_kv_pages,
             static_cast<uint32_t>(request.sequence.block_table.size())});
    }

    std::vector<const RuntimeRequest*> active;
    active.reserve(active_count());
    for (const auto& request : requests_) {
        if (request.state != RequestState::Active) {
            continue;
        }
        active.push_back(&request);
    }

    auto plan_result = schedule_requests(active, config_.max_scheduled_tokens, config_.max_scheduled_requests, config_.page_tokens, &kv_state);
    if (!plan_result.ok()) {
        return plan_result.status();
    }
    SchedulePlan plan = plan_result.release();
    if (plan.empty()) {
        return StepResult{false, 0, 0};
    }

    std::unordered_map<uint64_t, RuntimeRequest*> by_handle;
    for (auto& request : requests_) {
        if (!request.sequence.is_allocated()) {
            continue;
        }
        const auto handle = request.sequence.request_handle();
        by_handle[::ps::runtime::request_handle_key(handle)] = &request;
    }

    auto resolve = [&](const ScheduledRequest& scheduled) -> RuntimeRequest* {
        const uint64_t key = ::ps::runtime::request_handle_key(scheduled.handle);
        auto it = by_handle.find(key);
        if (it == by_handle.end()) {
            return nullptr;
        }
        RuntimeRequest* request = it->second;
        if (!::ps::runtime::request_handle_equal(
                request->sequence.request_handle(), scheduled.handle)) {
            return nullptr;
        }
        return request;
    };

    auto exec_result = tp_batch_hook_ != nullptr
                          ? tp_batch_hook_->on_execute(plan.batch)
                          : execute_batch(executor_, plan.batch, stream_);
    if (!exec_result.ok()) {
        Status first_error = Status::make_ok();
        for (auto& scheduled : plan.scheduled_requests) {
            if (!scheduled.sequence->is_poisoned()) {
                continue;
            }
            RuntimeRequest* request = resolve(scheduled);
            if (request != nullptr && request->state == RequestState::Active) {
                auto st = finish_request(*request, FinishReason::Error, stream_);
                if (!st.ok() && first_error.ok()) {
                    first_error = st;
                }
            }
        }
        if (!first_error.ok()) {
            return first_error;
        }
        return exec_result.status();
    }

    BatchExecutionOutput output = exec_result.release();

    std::vector<int32_t> host_sampled;
    host_sampled.resize(output.num_outputs);
    if (output.num_outputs > 0) {
        hipError_t copy_err = hipMemcpy(
            host_sampled.data(),
            output.sampled_tokens.data<int32_t>(),
            output.num_outputs * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (copy_err != hipSuccess) {
            return Status::hip_error(
                "step hipMemcpy sampled tokens",
                hipGetErrorString(copy_err), __FILE__, __LINE__);
        }
    }

    Status first_error = Status::make_ok();
    uint32_t row = 0;
    for (auto& scheduled : plan.scheduled_requests) {
        RuntimeRequest* request_ptr = resolve(scheduled);
        if (request_ptr == nullptr) {
            return Status::invalid_state("planned sequence not found", __FILE__, __LINE__);
        }
        RuntimeRequest& request = *request_ptr;
        if (prefix_cache_ != nullptr && prefix_cache_->enabled() &&
            request.prefix_cache_checkpoint_position > 0 &&
            !request.prefix_cache_checkpoint_saved &&
            scheduled.execution_class == ::ps::runtime::ExecutionClass::PREFILL &&
            scheduled.prefix_tokens + scheduled.num_tokens ==
                request.prefix_cache_checkpoint_position &&
            request.sequence.position == request.prefix_cache_checkpoint_position) {
            std::vector<int32_t> prefix_tokens(
                request.input_tokens.begin(),
                request.input_tokens.begin() + static_cast<std::ptrdiff_t>(
                    request.prefix_cache_checkpoint_position));
            auto boundary_st = prefix_cache_->save(
                request.sequence, gdn_pool_, kv_pool_, std::move(prefix_tokens),
                stream_, true);
            if (!boundary_st.ok()) {
                return boundary_st;
            }
            request.prefix_cache_checkpoint_saved = true;
        }
        int32_t token = -1;
        if (scheduled.compute_logits) {
            if (row >= output.num_outputs) {
                return Status::invalid_state(
                    "sample output index out of range", __FILE__, __LINE__);
            }
            token = host_sampled[row];
            ++row;
        }
        {
            const bool final_prefill =
                scheduled.execution_class == ::ps::runtime::ExecutionClass::PREFILL &&
                scheduled.prefix_tokens + scheduled.num_tokens ==
                    static_cast<uint32_t>(request.input_tokens.size());
            if (scheduled.compute_logits) {
                if (scheduled.sample) {
                    if (token < 0) {
                        return Status::invalid_state(
                            "greedy sampled request returned invalid token",
                            __FILE__, __LINE__);
                    }
                    const CommitAction action =
                        commit_sampled_token(request, token, config_.eos_token_ids);
                    if (action == CommitAction::Finished) {
                        auto st = finish_request(request, request.finish_reason, stream_);
                        if (!st.ok() && first_error.ok()) {
                            first_error = st;
                        }
                    }
                } else {
                    if (token != -1) {
                        return Status::invalid_state(
                            "logits-only request returned sampled token", __FILE__, __LINE__);
                    }
                }
            } else if (final_prefill) {
                const CommitAction action = commit_final_prefill_without_sample(request);
                if (action == CommitAction::Finished) {
                    auto st = finish_request(request, request.finish_reason, stream_);
                    if (!st.ok() && first_error.ok()) {
                        first_error = st;
                    }
                }
            }
        }
    }
    if (row != output.num_outputs) {
        return Status::invalid_state(
            "sample output count mismatch", __FILE__, __LINE__);
    }
    if (!first_error.ok()) {
        return first_error;
    }

    return StepResult{true, plan.batch.num_tokens, plan.batch.num_requests};
}

}
}
}
