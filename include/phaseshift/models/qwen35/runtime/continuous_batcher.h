#pragma once

#include <phaseshift/models/qwen35/runtime/runtime_request.h>
#include <phaseshift/models/qwen35/runtime/schedule_plan.h>
#include <phaseshift/models/qwen35/runtime/token_budget_scheduler.h>
#include <phaseshift/models/qwen35/runtime/kv_capacity_manager.h>
#include <phaseshift/models/qwen35/runtime/kv_banker.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/tp_batch_hook.h>
#include <phaseshift/models/qwen35/stop_tokens.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace ps {
namespace qwen35 {

struct Executor;
class SequenceSlotPool;
class GdnStatePool;
class PagedKVPool;

namespace runtime {

struct ContinuousBatcherConfig {
    uint32_t max_scheduled_tokens = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_seq_len = 0;
    uint32_t page_tokens = 0;
    StopTokens eos_token_ids;
    KVAdmissionPolicy admission_policy = KVAdmissionPolicy::BankerSafe;
};

class ContinuousBatcher {
 public:
    ContinuousBatcher(
        Executor& executor,
        SequenceSlotPool& seq_pool,
        GdnStatePool& gdn_pool,
        PagedKVPool& kv_pool,
        ContinuousBatcherConfig config,
        hipStream_t stream);

    ContinuousBatcher(const ContinuousBatcher&) = delete;
    ContinuousBatcher& operator=(const ContinuousBatcher&) = delete;

    Result<uint64_t> submit(std::vector<int32_t> input_tokens, uint32_t max_new_tokens);

    Result<uint64_t> submit(
        std::vector<int32_t> input_tokens,
        uint32_t max_new_tokens,
        const SamplingConfig& sampling);

    Status cancel(uint64_t id);

    Status retire(uint64_t id);

    Result<StepResult> step();

    Status cancel_all();

    bool has_pending() const noexcept;

    const RuntimeRequest* request(uint64_t id) const;

    std::size_t queued_count() const noexcept;
    std::size_t active_count() const noexcept;
    std::size_t finished_count() const noexcept;

    KVAdmissionPolicy admission_policy() const noexcept {
        return config_.admission_policy;
    }

    void set_tp_batch_hook(TpBatchHook* hook) noexcept {
        tp_batch_hook_ = hook;
    }

    KVCapacitySnapshot kv_capacity_snapshot() const;

 private:
    Status finish_request(RuntimeRequest& request, FinishReason reason, hipStream_t stream);

    KVBankerState build_kv_banker_state() const;

    Executor& executor_;
    SequenceSlotPool& seq_pool_;
    GdnStatePool& gdn_pool_;
    PagedKVPool& kv_pool_;
    ContinuousBatcherConfig config_;
    hipStream_t stream_;

    KVCapacityManager capacity_;
    TpBatchHook* tp_batch_hook_ = nullptr;

    uint64_t next_id_ = 0;

    using RequestList = std::list<RuntimeRequest>;
    RequestList requests_;
    std::unordered_map<uint64_t, RequestList::iterator> request_index_;
};

}
}
}
