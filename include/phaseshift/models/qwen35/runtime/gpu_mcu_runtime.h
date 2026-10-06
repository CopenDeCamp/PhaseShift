#pragma once

#include <phaseshift/models/qwen35/runtime/runtime_request.h>
#include <phaseshift/models/qwen35/runtime/sampling_params.h>
#include <phaseshift/models/qwen35/stop_tokens.h>
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/control_ring.h>
#include <phaseshift/runtime/gpu_mcu/output_ring.h>
#include <phaseshift/runtime/gpu_mcu/request_ingress.h>
#include <phaseshift/runtime/gpu_mcu/slot_runtime.h>
#include <phaseshift/runtime/gpu_mcu/slot_table.h>
#include <hip/hip_runtime.h>

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

struct GpuMcuRuntimeConfig {
    uint32_t max_scheduled_tokens = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_seq_len = 0;
    uint32_t page_tokens = 0;
    StopTokens eos_token_ids;
};

class GpuMcuRuntime {
 public:
    GpuMcuRuntime(
        Executor& executor,
        SequenceSlotPool& seq_pool,
        GdnStatePool& gdn_pool,
        PagedKVPool& kv_pool,
        GpuMcuRuntimeConfig config,
        hipStream_t stream);

    GpuMcuRuntime(const GpuMcuRuntime&) = delete;
    GpuMcuRuntime& operator=(const GpuMcuRuntime&) = delete;
    ~GpuMcuRuntime() noexcept;

    Result<uint64_t> submit(std::vector<int32_t> input_tokens,
                            uint32_t max_new_tokens,
                            const SamplingConfig& sampling);

    Result<StepResult> step();

    Status cancel(uint64_t id);

    Status retire(uint64_t id);

    bool has_pending() const noexcept;

    const RuntimeRequest* request(uint64_t id) const;

    Status cancel_all();

    Status allocate_controller_resources(int device);

    Status shutdown();

    bool controller_resources_ready() const noexcept {
        return controller_resources_ready_;
    }

    ::ps::runtime::gpu_mcu::GpuMcuControlRing& control_ring() noexcept {
        return control_ring_;
    }
    ::ps::runtime::gpu_mcu::GpuMcuRequestIngressArena& ingress_arena() noexcept {
        return ingress_;
    }
    ::ps::runtime::gpu_mcu::GpuMcuOutputRing& output_ring() noexcept {
        return output_ring_;
    }
    ::ps::runtime::gpu_mcu::GpuMcuSlotTable& slot_table() noexcept {
        return slot_table_;
    }
    ::ps::runtime::RequestHandle* runnable_handles() noexcept {
        return runnable_;
    }
    ::ps::runtime::gpu_mcu::GpuMcuSlotRuntimeState* slot_runtime() noexcept {
        return slot_runtime_;
    }
    ::ps::runtime::gpu_mcu::GpuMcuPendingAdmission* admission_queue() noexcept {
        return admission_;
    }
    uint32_t max_slots() const noexcept { return max_slots_; }
    uint32_t admission_capacity() const noexcept {
        return admission_capacity_;
    }

    bool is_shutdown() const noexcept { return shutdown_; }

    const GpuMcuRuntimeConfig& config() const noexcept {
        return config_;
    }

    Executor& executor() const noexcept { return executor_; }
    SequenceSlotPool& sequence_pool() const noexcept { return seq_pool_; }
    GdnStatePool& gdn_pool() const noexcept { return gdn_pool_; }
    PagedKVPool& kv_pool() const noexcept { return kv_pool_; }
    hipStream_t stream() const noexcept { return stream_; }

 private:
    using RequestList = std::list<RuntimeRequest>;

    Executor& executor_;
    SequenceSlotPool& seq_pool_;
    GdnStatePool& gdn_pool_;
    PagedKVPool& kv_pool_;
    GpuMcuRuntimeConfig config_;
    hipStream_t stream_;

    bool shutdown_ = false;

    RequestList requests_;
    std::unordered_map<uint64_t, RequestList::iterator> request_index_;

    ::ps::runtime::gpu_mcu::GpuMcuControlRing control_ring_;
    ::ps::runtime::gpu_mcu::GpuMcuRequestIngressArena ingress_;
    ::ps::runtime::gpu_mcu::GpuMcuOutputRing output_ring_;
    ::ps::runtime::gpu_mcu::GpuMcuSlotTable slot_table_;
    ::ps::runtime::RequestHandle* runnable_ = nullptr;
    ::ps::runtime::gpu_mcu::GpuMcuSlotRuntimeState* slot_runtime_ = nullptr;
    ::ps::runtime::gpu_mcu::GpuMcuPendingAdmission* admission_ = nullptr;
    uint32_t max_slots_ = 0u;
    uint32_t admission_capacity_ = 0u;
    bool controller_resources_ready_ = false;
};

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
