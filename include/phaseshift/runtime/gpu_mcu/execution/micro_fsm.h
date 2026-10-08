#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/binding/kernarg_source_contract.h>
#include <phaseshift/runtime/gpu_mcu/binding/plan_binding_contract.h>
#include <phaseshift/runtime/gpu_mcu/execution/fsm_contract.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/device_completion.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/retained_packet.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

constexpr uint32_t kMcuMaxNodes = 16384;
constexpr uint32_t kMcuMaxVariants = 64;
constexpr uint32_t kMcuMaxCompletionSlots = 32;
constexpr uint32_t kMcuMaxKernargSlots = kMcuMaxNodes;
constexpr uint32_t kMcuMaxTiming = 4096;
constexpr uint32_t kMcuMaxDebugRecords = 4096;

struct GpuMcuFsmRunContext {
    uint64_t iterations = 0;
    uint64_t heartbeat = 0;
    uint64_t plans_started = 0;
    uint64_t dispatches_committed = 0;
    uint64_t completions_observed = 0;
    uint64_t run_consumed = 0;
    uint64_t write_base = 0;
    uint64_t dispatch_seq = 0;
    uint64_t append_ahead_total = 0;
    uint64_t queue_empty_total = 0;
    uint64_t queue_full_total = 0;
    uint64_t refill_total = 0;
    uint64_t doorbell_total = 0;
    uint64_t max_ahead_total = 0;
    uint64_t wrap_total = 0;
    uint32_t write_base_valid = 0;
    uint32_t region_seq = 0;
    uint32_t supervisor = static_cast<uint32_t>(McuSupervisorState::idle);
};

struct alignas(64) GpuMcuFsmState {
    uint32_t stop_requested = 0;
    uint32_t started = 0;
    uint32_t supervisor = 0;
    uint32_t reserved_ctrl = 0;
    uint64_t run_request = 0;
    uint64_t heartbeat = 0;
    uint64_t iterations = 0;
    uint64_t plans_started = 0;
    uint64_t dispatches_committed = 0;
    uint64_t completions_observed = 0;
    uint32_t fault_code = 0;
    uint32_t fault_pc = 0;
    uint32_t fault_variant = 0;
    uint32_t fault_generation = 0;

    DeviceAqlQueueView queue{};
    const McuPlanNode* plan = nullptr;
    uint32_t node_count = 0;
    uint32_t plan_id = 0;
    const McuKernelVariantDesc* variants = nullptr;
    uint32_t variant_count = 0;
    McuRuntimeNodeBinding* runtime_node_bindings = nullptr;
    uint32_t runtime_node_binding_count = 0;
    const McuAttentionPathSpan* attention_paths = nullptr;
    uint32_t attention_path_count = 0;
    const McuKernargSourceDesc* kernarg_sources = nullptr;
    uint32_t kernarg_source_count = 0;
    uint32_t reserved0 = 0;
    uint64_t kernarg_base = 0;
    uint64_t kernarg_slot_stride = 0;
    uint32_t kernarg_slot_count = 0;
    uint32_t reserved1 = 0;
    GpuMcuDeviceCompletion* completions = nullptr;
    uint32_t completion_count = 0;
    uint32_t reserved2 = 0;
    uint64_t output_base = 0;
    uint64_t timestamps_base = 0;
    GpuMcuRetainedPacket* retained_vram = nullptr;
    uint32_t retained_count = 0;
    uint32_t template_source = 0;
    uint32_t copy_lanes = 1;
    uint32_t lds_budget_bytes = 0;
    uint32_t barrier = 1;
    uint32_t prepared_enabled = 0;
    uint32_t idle_sleep = 8;
    uint32_t reserved3 = 0;
    McuDispatchTiming* timing = nullptr;
    uint32_t timing_count = 0;
    uint32_t queue_ahead_depth = 0;
    uint32_t doorbell_batch = 1;
    uint32_t doorbell_mode = 0;
    uint32_t issue_wait_running = 0;
    McuDispatchRecord* debug_records = nullptr;
    uint32_t debug_record_count = 0;
    uint32_t reserved5 = 0;
    uint64_t region_start_ts = 0;
    uint64_t region_end_ts = 0;
    uint64_t append_ahead_count = 0;
    uint64_t queue_empty_count = 0;
    uint64_t mid_region_queue_empty_count = 0;
    uint64_t backpressure_wait_count = 0;
    uint64_t refill_count = 0;
    uint64_t doorbell_count = 0;
    uint64_t max_ahead = 0;
    uint64_t wrap_count = 0;
    uint32_t external_epoch_seen = 0;
    uint32_t* external_start_signal = nullptr;
    uint32_t* external_done_signal = nullptr;
    uint32_t* external_result_code = nullptr;

    uint64_t batch_input = 0;

    uint64_t log_base = 0;
    uint32_t log_head = 0;
    uint32_t log_enabled = 0;

    GpuMcuFsmRunContext run_ctx{};
};

struct GpuMcuFsmConfig {
    DeviceAqlQueueView queue{};
    const McuPlanNode* plan = nullptr;
    uint32_t node_count = 0;
    uint32_t plan_id = 0;
    const McuKernelVariantDesc* variants = nullptr;
    uint32_t variant_count = 0;
    McuRuntimeNodeBinding* runtime_node_bindings = nullptr;
    uint32_t runtime_node_binding_count = 0;
    const McuAttentionPathSpan* attention_paths = nullptr;
    uint32_t attention_path_count = 0;
    const McuKernargSourceDesc* kernarg_sources = nullptr;
    uint32_t kernarg_source_count = 0;
    const GpuMcuRetainedPacket* retained = nullptr;
    uint32_t retained_count = 0;
    uint64_t kernarg_base = 0;
    uint64_t kernarg_slot_stride = 0;
    uint32_t kernarg_slot_count = 0;
    GpuMcuDeviceCompletion* completions = nullptr;
    uint32_t completion_count = 0;
    uint64_t output_base = 0;
    uint64_t timestamps_base = 0;
    McuDispatchTiming* timing = nullptr;
    uint32_t timing_count = 0;
    uint32_t lds_budget_bytes = 0;
    uint32_t template_source = 0;
    uint32_t copy_lanes = 1;
    uint32_t barrier = 1;
    uint32_t prepared_enabled = 0;
    uint32_t idle_sleep = 8;
    uint32_t queue_ahead_depth = 0;
    uint32_t doorbell_batch = 1;
    uint32_t doorbell_mode = 0;
    uint32_t issue_wait_running = 0;
    McuDispatchRecord* debug_records = nullptr;
    uint32_t debug_record_count = 0;
    uint32_t* start_signal = nullptr;
    uint32_t* done_signal = nullptr;
    uint32_t* result_code = nullptr;
    uint64_t log_base = 0;
};

constexpr AqlMemoryPolicy kMcuAgentAqlPolicy{
    .acquire_scope = AqlFenceScope::Agent,
    .release_scope = AqlFenceScope::Agent,
    .producer_visibility_fence = true,
};

constexpr AqlMemoryPolicy kMcuSystemAqlPolicy{
    .acquire_scope = AqlFenceScope::System,
    .release_scope = AqlFenceScope::System,
    .producer_visibility_fence = true,
};

McuKernelVariantDesc make_kernel_variant(
    uint16_t variant_id,
    const GpuAqlKernelMetadata& meta,
    uint16_t kernarg_recipe,
    uint32_t workgroup_x,
    uint32_t workgroup_count_x,
    const AqlMemoryPolicy& policy,
    AqlHiddenArgsPolicy hidden_policy = AqlHiddenArgsPolicy::IfFits,
    uint32_t workgroup_count_y = 1u,
    uint32_t workgroup_count_z = 1u);

McuKernelVariantDesc make_probe_variant(
    uint16_t variant_id,
    const GpuAqlKernelMetadata& meta,
    uint32_t workgroup_x,
    uint32_t workgroup_count_x,
    const AqlMemoryPolicy& policy = kMcuAgentAqlPolicy,
    AqlHiddenArgsPolicy hidden_policy = AqlHiddenArgsPolicy::IfFits);

class GpuMcuFsm {
public:
    GpuMcuFsm() = default;
    ~GpuMcuFsm() noexcept { (void)shutdown(); }

    GpuMcuFsm(const GpuMcuFsm&) = delete;
    GpuMcuFsm& operator=(const GpuMcuFsm&) = delete;

    GpuMcuFsm(GpuMcuFsm&& other) noexcept;
    GpuMcuFsm& operator=(GpuMcuFsm&& other) noexcept;

    static Result<GpuMcuFsm> create(int device, hipStream_t control_stream);

    bool valid() const noexcept { return host_state_ != nullptr; }

    Status configure(const GpuMcuFsmConfig& config);
    Status start();

    Status request_run();
    Status wait_plans(uint64_t count, uint32_t timeout_ms) const;
    bool running() const noexcept;
    McuSupervisorState supervisor() const noexcept;
    McuFaultCode fault_code() const noexcept;
    uint32_t fault_pc() const noexcept;
    uint32_t log_head() const noexcept;
    const McuLogRecord* log_record(uint32_t slot) const noexcept;
    uint64_t plans_started() const noexcept;
    uint64_t dispatches() const noexcept;
    uint64_t completions_observed() const noexcept;
    uint64_t heartbeat() const noexcept;
    uint64_t append_ahead_count() const noexcept;
    uint64_t queue_empty_count() const noexcept;
    uint64_t mid_region_queue_empty_count() const noexcept;
    uint64_t backpressure_wait_count() const noexcept;
    uint64_t refill_count() const noexcept;
    uint64_t doorbell_count() const noexcept;
    uint64_t max_ahead() const noexcept;
    uint64_t wrap_count() const noexcept;
    uint32_t external_epoch_seen() const noexcept;
    uint64_t region_start_ts() const noexcept;
    uint64_t region_end_ts() const noexcept;

    Status request_stop();
    Status wait_stopped(uint32_t timeout_ms);
    Status shutdown() noexcept;

    GpuMcuFsmState* device_state() const noexcept { return device_state_; }
    GpuMcuFsmState* host_state() const noexcept { return host_state_; }

private:
    void move_from(GpuMcuFsm& other) noexcept;

    int device_ = -1;
    void* host_allocation_ = nullptr;
    GpuMcuFsmState* host_state_ = nullptr;
    GpuMcuFsmState* device_state_ = nullptr;
    hipStream_t control_stream_ = nullptr;
    uint32_t lds_bytes_ = 0;
    uint64_t requested_runs_ = 0;
    bool launched_ = false;
};

}  // namespace ps::runtime::gpu_mcu
