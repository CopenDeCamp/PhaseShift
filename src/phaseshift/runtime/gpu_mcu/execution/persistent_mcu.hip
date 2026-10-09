#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <utility>

namespace ps::runtime::gpu_mcu {

namespace {

uint32_t load_u32(const uint32_t* ptr) {
    return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(ptr))
        .load(std::memory_order_acquire);
}

uint64_t load_u64(const uint64_t* ptr) {
    return std::atomic_ref<uint64_t>(*const_cast<uint64_t*>(ptr))
        .load(std::memory_order_acquire);
}

void store_u64(uint64_t* ptr, uint64_t value) {
    std::atomic_ref<uint64_t>(*ptr).store(value, std::memory_order_release);
}

}  // namespace

void GpuMcuPersistentMcu::move_from(GpuMcuPersistentMcu& other) noexcept {
    device_ = other.device_;
    host_allocation_ = other.host_allocation_;
    host_state_ = other.host_state_;
    device_state_ = other.device_state_;
    control_stream_ = other.control_stream_;
    launched_ = other.launched_;
    other.device_ = -1;
    other.host_allocation_ = nullptr;
    other.host_state_ = nullptr;
    other.device_state_ = nullptr;
    other.control_stream_ = nullptr;
    other.launched_ = false;
}

Result<GpuMcuPersistentMcu> GpuMcuPersistentMcu::create(
    int device,
    hipStream_t control_stream) {
    if (device < 0) {
        return Status::invalid_argument("device must be >= 0", __FILE__, __LINE__);
    }
    if (control_stream == nullptr) {
        return Status::invalid_argument("control stream must not be null",
                                        __FILE__, __LINE__);
    }
    void* host = nullptr;
    if (hipHostMalloc(&host, sizeof(GpuMcuPersistentState),
                      hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
        return Status::hip_error("hipHostMalloc(persistent state)", "allocation failed",
                                 __FILE__, __LINE__);
    }
    void* dev = nullptr;
    if (hipHostGetDevicePointer(&dev, host, 0) != hipSuccess) {
        (void)hipFreeHost(host);
        return Status::hip_error("hipHostGetDevicePointer(persistent state)",
                                 "mapping failed", __FILE__, __LINE__);
    }
    auto* host_state = static_cast<GpuMcuPersistentState*>(host);
    *host_state = GpuMcuPersistentState{};

    GpuMcuPersistentMcu out;
    out.device_ = device;
    out.host_allocation_ = host;
    out.host_state_ = host_state;
    out.device_state_ = static_cast<GpuMcuPersistentState*>(dev);
    out.control_stream_ = control_stream;
    return out;
}

Status GpuMcuPersistentMcu::start() {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__, __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__, __LINE__);
    }
    const GpuMcuPersistentStartConfig start_config =
        gpu_mcu_capture_start_config(*host_state_);

    *host_state_ = GpuMcuPersistentState{};
    gpu_mcu_restore_start_config(*host_state_, start_config);

    gpu_mcu_persistent_loop_kernel<<<1, 32, 0, control_stream_>>>(device_state_);
    if (hipGetLastError() != hipSuccess) {
        return Status::hip_error("persistent launch", "launch failed", __FILE__, __LINE__);
    }
    launched_ = true;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (load_u32(&host_state_->started) == 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::invalid_state("persistent mcu did not start", __FILE__, __LINE__);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return Status::make_ok();
}

bool GpuMcuPersistentMcu::running() const noexcept {
    return host_state_ != nullptr && load_u32(&host_state_->started) != 0u;
}

uint64_t GpuMcuPersistentMcu::heartbeat() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->heartbeat);
}

uint64_t GpuMcuPersistentMcu::iterations() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->iterations);
}

Status GpuMcuPersistentMcu::configure_batch_binding(
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots,
    int32_t* token_ids,
    uint32_t* output_rows,
    DeviceSamplingParams* output_sampling_params,
    uint32_t max_outputs,
    uint64_t* batch_ready_epoch) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (bindings == nullptr || token_ids == nullptr || output_rows == nullptr ||
        output_sampling_params == nullptr || batch_ready_epoch == nullptr) {
        return Status::invalid_argument("batch binding pointer is null",
                                        __FILE__, __LINE__);
    }
    if (binding_max_slots == 0u || max_outputs == 0u) {
        return Status::invalid_argument("batch binding capacity is zero",
                                        __FILE__, __LINE__);
    }
    host_state_->slot_bindings = bindings;
    host_state_->binding_max_slots = binding_max_slots;
    host_state_->batch_token_ids = token_ids;
    host_state_->batch_output_rows = output_rows;
    host_state_->batch_output_sampling_params = output_sampling_params;
    host_state_->batch_max_outputs = max_outputs;
    host_state_->batch_ready_epoch = batch_ready_epoch;
    host_state_->batch_binding_enabled = 1u;
    return Status::make_ok();
}

bool GpuMcuPersistentMcu::batch_binding_enabled() const noexcept {
    return host_state_ != nullptr &&
           load_u32(&host_state_->batch_binding_enabled) != 0u;
}

uint64_t GpuMcuPersistentMcu::batch_ready_epoch() const noexcept {
    if (host_state_ == nullptr || host_state_->batch_ready_epoch == nullptr) {
        return 0u;
    }
    return __scoped_atomic_load_n(host_state_->batch_ready_epoch,
                                  __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

uint32_t GpuMcuPersistentMcu::batch_binding_bound_requests() const noexcept {
    return host_state_ != nullptr
               ? host_state_->batch_binding_telemetry.bound_requests
               : 0u;
}

uint32_t GpuMcuPersistentMcu::batch_binding_bound_outputs() const noexcept {
    return host_state_ != nullptr
               ? host_state_->batch_binding_telemetry.bound_outputs
               : 0u;
}

Status GpuMcuPersistentMcu::configure_admission(
    GpuMcuPendingAdmission* queue, uint32_t capacity) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (queue == nullptr || capacity == 0u) {
        return Status::invalid_argument("admission queue is not configured",
                                        __FILE__, __LINE__);
    }
    host_state_->admission_queue = queue;
    host_state_->admission_capacity = capacity;
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::configure_control_drain(
    uint32_t max_commands_per_boundary) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (max_commands_per_boundary == 0u) {
        return Status::invalid_argument("control drain budget is zero", __FILE__,
                                        __LINE__);
    }
    host_state_->max_control_commands_per_boundary = max_commands_per_boundary;
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::configure_kv_blocks(uint32_t* block_tables,
                                               uint32_t block_table_stride,
                                               uint32_t page_tokens) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (block_tables == nullptr || block_table_stride == 0u ||
        page_tokens == 0u) {
        return Status::invalid_argument("kv block table is not configured",
                                        __FILE__, __LINE__);
    }
    host_state_->kv_blocks.block_tables = block_tables;
    host_state_->kv_blocks.block_table_stride = block_table_stride;
    host_state_->kv_blocks.page_tokens = page_tokens;
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::configure_resources(
    const GpuMcuKvPagePoolView& kv_pages,
    const DeviceSequenceResourceManagerView& sequence_resources) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (sequence_resources.resources == nullptr ||
        sequence_resources.resource_count == 0u) {
        return Status::invalid_argument("sequence resources are not configured",
                                        __FILE__, __LINE__);
    }
    host_state_->kv_pages = kv_pages;
    host_state_->sequence_resources = sequence_resources;
    return Status::make_ok();
}

GpuMcuExecutionTelemetry GpuMcuPersistentMcu::execution_telemetry()
    const noexcept {
    if (host_state_ == nullptr) return GpuMcuExecutionTelemetry{};
    return host_state_->execution_telemetry;
}

bool GpuMcuPersistentMcu::resources_enabled() const noexcept {
    return host_state_ != nullptr &&
           host_state_->sequence_resources.resources != nullptr;
}

uint32_t GpuMcuPersistentMcu::control_drain_budget() const noexcept {
    return host_state_ != nullptr
               ? host_state_->max_control_commands_per_boundary
               : 0u;
}

bool GpuMcuPersistentMcu::admission_enabled() const noexcept {
    return host_state_ != nullptr && host_state_->admission_queue != nullptr;
}

uint32_t GpuMcuPersistentMcu::admission_count() const noexcept {
    return host_state_ != nullptr ? host_state_->admission_count : 0u;
}

uint32_t GpuMcuPersistentMcu::admission_dropped() const noexcept {
    return host_state_ != nullptr ? host_state_->admission_dropped : 0u;
}

Status GpuMcuPersistentMcu::configure_slot_runtime(
    GpuMcuSlotRuntimeState* runtime) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (runtime == nullptr) {
        return Status::invalid_argument("slot runtime state is null", __FILE__,
                                        __LINE__);
    }
    host_state_->slot_runtime = runtime;
    return Status::make_ok();
}

bool GpuMcuPersistentMcu::slot_runtime_enabled() const noexcept {
    return host_state_ != nullptr && host_state_->slot_runtime != nullptr;
}

Status GpuMcuPersistentMcu::configure_output(OutputRing* ring) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (ring == nullptr) {
        return Status::invalid_argument("output ring is null", __FILE__,
                                        __LINE__);
    }
    host_state_->output_ring = ring;
    host_state_->output_enabled = 1u;
    return Status::make_ok();
}

bool GpuMcuPersistentMcu::output_enabled() const noexcept {
    return host_state_ != nullptr && host_state_->output_enabled != 0u;
}

uint64_t GpuMcuPersistentMcu::output_tokens_published() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.output_tokens_published
               : 0u;
}

uint64_t GpuMcuPersistentMcu::output_backpressure_events() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.output_backpressure_events
               : 0u;
}

Status GpuMcuPersistentMcu::configure_execution(
    GpuMcuFsmState* fsm, uint32_t retained_count, uint64_t* batch_ready_epoch,
    const int32_t* sampled_tokens, uint32_t sampled_capacity,
    const uint32_t* verify_counts, uint32_t verify_capacity) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (fsm == nullptr || retained_count == 0u ||
        batch_ready_epoch == nullptr) {
        return Status::invalid_argument("execution fsm is not configured",
                                        __FILE__, __LINE__);
    }
    host_state_->execution_fsm = fsm;
    host_state_->batch_ready_epoch = batch_ready_epoch;
    host_state_->execution_sampled_tokens = sampled_tokens;
    host_state_->execution_sampled_capacity = sampled_capacity;
    host_state_->execution_verify_counts = verify_counts;
    host_state_->execution_verify_capacity = verify_capacity;
    host_state_->execution_retained_count = retained_count;
    host_state_->execution_enabled = 1u;
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::configure_execution_patches(
    const McuInvocationPatch* patches, uint32_t count) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (count != 0u && patches == nullptr) {
        return Status::invalid_argument("null invocation patch table", __FILE__,
                                        __LINE__);
    }
    host_state_->execution_patches = patches;
    host_state_->execution_patch_count = count;
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::configure_commit_slots(
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (slots == nullptr || bindings == nullptr) {
        return Status::invalid_argument("commit slot table is null", __FILE__,
                                        __LINE__);
    }
    if (max_slots == 0u || binding_max_slots == 0u) {
        return Status::invalid_argument("commit slot capacity is zero", __FILE__,
                                        __LINE__);
    }
    host_state_->request_slots = slots;
    host_state_->request_max_slots = max_slots;
    host_state_->slot_bindings = bindings;
    host_state_->binding_max_slots = binding_max_slots;
    return Status::make_ok();
}

const McuInvocationPatch* GpuMcuPersistentMcu::execution_patches()
    const noexcept {
    return host_state_ != nullptr ? host_state_->execution_patches : nullptr;
}

uint32_t GpuMcuPersistentMcu::execution_patch_count() const noexcept {
    return host_state_ != nullptr ? host_state_->execution_patch_count : 0u;
}

bool GpuMcuPersistentMcu::execution_enabled() const noexcept {
    return host_state_ != nullptr &&
           load_u32(&host_state_->execution_enabled) != 0u;
}

uint64_t GpuMcuPersistentMcu::execution_epoch() const noexcept {
    if (host_state_ == nullptr) return 0u;
    return __scoped_atomic_load_n(&host_state_->execution_epoch,
                                  __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

uint64_t GpuMcuPersistentMcu::batches_dispatched() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.batches_dispatched
               : 0u;
}

uint64_t GpuMcuPersistentMcu::batches_completed() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.batches_completed
               : 0u;
}

uint64_t GpuMcuPersistentMcu::batches_failed() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.batches_failed
               : 0u;
}

uint64_t GpuMcuPersistentMcu::batches_committed() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.batches_committed
               : 0u;
}

GpuMcuBatchCommitTelemetry GpuMcuPersistentMcu::commit_telemetry()
    const noexcept {
    return host_state_ != nullptr ? host_state_->commit_telemetry
                                  : GpuMcuBatchCommitTelemetry{};
}

void GpuMcuPersistentMcu::dump_state(std::FILE* out) const noexcept {
    if (out == nullptr || host_state_ == nullptr) return;
    std::fprintf(out,
                 "[mcu-persistent] started=%u iterations=%llu heartbeat=%llu "
                 "stop=%u execution_enabled=%u execution_fsm=%p "
                 "batch_context=%p batch_ready_epoch=%p "
                 "active=%llu epoch=%llu wrap_faulted=%u "
                 "dispatched=%llu completed=%llu committed=%llu failed=%llu\n",
                 static_cast<uint32_t>(host_state_->started),
                 static_cast<unsigned long long>(host_state_->iterations),
                 static_cast<unsigned long long>(host_state_->heartbeat),
                 static_cast<uint32_t>(
                     __atomic_load_n(&host_state_->stop_requested,
                                     __ATOMIC_ACQUIRE)),
                 host_state_->execution_enabled,
                 static_cast<const void*>(host_state_->execution_fsm),
                 static_cast<const void*>(host_state_->batch_context),
                 static_cast<const void*>(host_state_->batch_ready_epoch),
                 static_cast<unsigned long long>(
                     host_state_->active_batch_ready_epoch),
                 static_cast<unsigned long long>(
                     host_state_->execution_epoch),
                 host_state_->execution_wrap_faulted,
                 static_cast<unsigned long long>(
                     host_state_->execution_telemetry.batches_dispatched),
                 static_cast<unsigned long long>(
                     host_state_->execution_telemetry.batches_completed),
                 static_cast<unsigned long long>(
                     host_state_->execution_telemetry.batches_committed),
                 static_cast<unsigned long long>(
                     host_state_->execution_telemetry.batches_failed));
    const McuStageTrace& stage = host_state_->stage_trace;
    std::fprintf(out,
                 "[mcu-persistent] stage=%s(%u) seq=%u patch=%u "
                 "patch_src=%u patch_width=%u patch_target=0x%08x%08x\n",
                 mcu_stage_name(stage.stage), stage.stage, stage.seq,
                 stage.patch_index, stage.patch_source, stage.patch_width,
                 stage.patch_target_high, stage.patch_target_low);
    if (host_state_->batch_context != nullptr) {
        std::fprintf(out, "[mcu-persistent] batch_capacity=%u max_rows=%u\n",
                     host_state_->batch_max_requests, host_state_->batch_max_rows);
    }
}

uint64_t GpuMcuPersistentMcu::committed_tokens() const noexcept {
    return host_state_ != nullptr ? host_state_->commit_telemetry.committed_tokens
                                  : 0u;
}

uint64_t GpuMcuPersistentMcu::commit_stale_handles() const noexcept {
    return host_state_ != nullptr ? host_state_->commit_telemetry.stale_handles
                                  : 0u;
}

uint64_t GpuMcuPersistentMcu::commit_pending_overflow() const noexcept {
    return host_state_ != nullptr
               ? host_state_->commit_telemetry.pending_overflow
               : 0u;
}

uint64_t GpuMcuPersistentMcu::autonomous_loops() const noexcept {
    return host_state_ != nullptr
               ? host_state_->execution_telemetry.autonomous_loops
               : 0u;
}

uint64_t GpuMcuPersistentMcu::last_committed_execution_epoch() const noexcept {
    if (host_state_ == nullptr) return 0u;
    return __scoped_atomic_load_n(&host_state_->last_committed_execution_epoch,
                                  __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

Status GpuMcuPersistentMcu::configure_batch_plan(
    DeviceBatchContext* context,
    DeviceRequestDescriptor* requests,
    uint32_t* row_sequence_slots,
    uint32_t* row_positions,
    uint32_t max_requests,
    uint32_t max_rows,
    uint32_t max_prefill_rows_per_slot,
    uint64_t* batch_plan_epoch) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (context == nullptr || requests == nullptr ||
        row_sequence_slots == nullptr || row_positions == nullptr ||
        batch_plan_epoch == nullptr) {
        return Status::invalid_argument("batch plan pointer is null", __FILE__,
                                        __LINE__);
    }
    if (max_requests == 0u || max_rows == 0u) {
        return Status::invalid_argument("batch plan capacity is zero", __FILE__,
                                        __LINE__);
    }
    host_state_->batch_context = context;
    host_state_->batch_requests = requests;
    host_state_->batch_row_sequence_slots = row_sequence_slots;
    host_state_->batch_row_positions = row_positions;
    host_state_->batch_max_requests = max_requests;
    host_state_->batch_max_rows = max_rows;
    host_state_->batch_max_prefill_rows_per_slot = max_prefill_rows_per_slot;
    host_state_->batch_plan_epoch = batch_plan_epoch;
    host_state_->batch_plan_enabled = 1u;
    return Status::make_ok();
}

bool GpuMcuPersistentMcu::batch_plan_enabled() const noexcept {
    return host_state_ != nullptr &&
           load_u32(&host_state_->batch_plan_enabled) != 0u;
}

uint64_t GpuMcuPersistentMcu::batch_plan_epoch() const noexcept {
    if (host_state_ == nullptr || host_state_->batch_plan_epoch == nullptr) {
        return 0u;
    }
    return __scoped_atomic_load_n(host_state_->batch_plan_epoch,
                                  __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

uint32_t GpuMcuPersistentMcu::batch_plan_source_epoch() const noexcept {
    return host_state_ != nullptr
               ? static_cast<uint32_t>(host_state_->batch_plan_source_epoch)
               : 0u;
}

uint32_t GpuMcuPersistentMcu::batch_plan_planned_requests() const noexcept {
    return host_state_ != nullptr
               ? host_state_->batch_plan_telemetry.planned_requests
               : 0u;
}

uint32_t GpuMcuPersistentMcu::batch_plan_planned_rows() const noexcept {
    return host_state_ != nullptr ? host_state_->batch_plan_telemetry.planned_rows
                                  : 0u;
}

Status GpuMcuPersistentMcu::configure_request_runtime(
    ControlRing* input_ring,
    ControlRing* event_ring,
    GpuMcuRequestIngressEntry* ingress_entries,
    uint32_t ingress_capacity,
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    RequestHandle* runnable_handles,
    uint32_t runnable_capacity) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__,
                                     __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("persistent mcu already started", __FILE__,
                                     __LINE__);
    }
    if (input_ring == nullptr || event_ring == nullptr ||
        ingress_entries == nullptr || slots == nullptr ||
        runnable_handles == nullptr) {
        return Status::invalid_argument("request runtime pointer is null",
                                        __FILE__, __LINE__);
    }
    if (max_slots == 0u || ingress_capacity == 0u) {
        return Status::invalid_argument("request runtime capacity is zero",
                                        __FILE__, __LINE__);
    }
    if (runnable_capacity < max_slots) {
        return Status::invalid_argument(
            "runnable capacity must cover every slot", __FILE__, __LINE__);
    }
    host_state_->request_input_ring = input_ring;
    host_state_->request_event_ring = event_ring;
    host_state_->request_ingress_entries = ingress_entries;
    host_state_->request_ingress_capacity = ingress_capacity;
    host_state_->request_slots = slots;
    host_state_->request_max_slots = max_slots;
    host_state_->runnable_handles = runnable_handles;
    host_state_->runnable_capacity = runnable_capacity;
    host_state_->request_runtime_enabled = 1u;
    host_state_->scheduler_dirty = 1u;
    return Status::make_ok();
}

bool GpuMcuPersistentMcu::request_runtime_enabled() const noexcept {
    return host_state_ != nullptr && load_u32(&host_state_->request_runtime_enabled) != 0u;
}

ControlRing* GpuMcuPersistentMcu::request_input_ring() const noexcept {
    return host_state_ != nullptr ? host_state_->request_input_ring : nullptr;
}

ControlRing* GpuMcuPersistentMcu::request_event_ring() const noexcept {
    return host_state_ != nullptr ? host_state_->request_event_ring : nullptr;
}

GpuMcuSlotState* GpuMcuPersistentMcu::request_slots() const noexcept {
    return host_state_ != nullptr ? host_state_->request_slots : nullptr;
}

uint32_t GpuMcuPersistentMcu::request_max_slots() const noexcept {
    return host_state_ != nullptr ? host_state_->request_max_slots : 0u;
}

RequestHandle* GpuMcuPersistentMcu::runnable_handles() const noexcept {
    return host_state_ != nullptr ? host_state_->runnable_handles : nullptr;
}

uint32_t GpuMcuPersistentMcu::runnable_capacity() const noexcept {
    return host_state_ != nullptr ? host_state_->runnable_capacity : 0u;
}

uint32_t GpuMcuPersistentMcu::runnable_count() const noexcept {
    return host_state_ != nullptr ? load_u32(&host_state_->runnable_count) : 0u;
}

uint64_t GpuMcuPersistentMcu::scheduler_epoch() const noexcept {
    return host_state_ != nullptr
               ? __scoped_atomic_load_n(&host_state_->scheduler_epoch,
                                        __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM)
               : 0u;
}

uint64_t GpuMcuPersistentMcu::commands_consumed() const noexcept {
    return host_state_ != nullptr ? host_state_->commands_consumed : 0u;
}

uint64_t GpuMcuPersistentMcu::events_published() const noexcept {
    return host_state_ != nullptr ? host_state_->events_published : 0u;
}

bool GpuMcuPersistentMcu::pending_event_valid() const noexcept {
    return host_state_ != nullptr &&
           load_u32(&host_state_->pending_event_valid) != 0u;
}

Status GpuMcuPersistentMcu::enable_emit(
    const DeviceAqlQueueView& queue,
    const GpuAqlPacketTemplate* device_templates,
    uint32_t template_count,
    uint64_t* device_publish_ts,
    uint64_t* device_doorbell_ts,
    uint32_t idle_sleep,
    bool streaming,
    bool barrier) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__, __LINE__);
    }
    if (device_templates == nullptr || template_count == 0u) {
        return Status::invalid_argument("emit requires templates", __FILE__, __LINE__);
    }
    host_state_->queue = queue;
    host_state_->templates = device_templates;
    host_state_->template_count = template_count;
    host_state_->publish_ts = device_publish_ts;
    host_state_->doorbell_ts = device_doorbell_ts;
    host_state_->idle_sleep = idle_sleep;
    host_state_->emit_streaming = streaming ? 1u : 0u;
    host_state_->barrier = barrier ? 1u : 0u;
    host_state_->write_base_valid = 0u;
    host_state_->write_base = 0u;
    std::atomic_ref<uint32_t>(host_state_->emit_enabled)
        .store(1u, std::memory_order_release);
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::request_emit(uint64_t published_target) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__, __LINE__);
    }
    store_u64(&host_state_->submit_request, published_target);
    return Status::make_ok();
}

uint64_t GpuMcuPersistentMcu::publish_target() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->submit_request);
}

uint64_t GpuMcuPersistentMcu::published() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->published);
}

uint64_t GpuMcuPersistentMcu::publish_failures() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->publish_failures);
}

Status GpuMcuPersistentMcu::request_stop() {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__, __LINE__);
    }
    std::atomic_ref<uint32_t>(host_state_->stop_requested)
        .store(1u, std::memory_order_release);
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::wait_stopped(uint32_t timeout_ms) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("persistent mcu not created", __FILE__, __LINE__);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (load_u32(&host_state_->started) != 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::invalid_state("persistent mcu did not stop", __FILE__, __LINE__);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    launched_ = false;
    return Status::make_ok();
}

Status GpuMcuPersistentMcu::shutdown() noexcept {
    if (host_state_ == nullptr) return Status::make_ok();
    Status first = Status::make_ok();
    auto note = [&](Status s) {
        if (!s.ok() && first.ok()) first = s;
    };
    if (launched_) {
        note(request_stop());
        Status stopped = wait_stopped(5000);
        if (!stopped.ok()) {
            note(request_stop());
            stopped = wait_stopped(5000);
        }
        if (!stopped.ok()) {
            return Status::invalid_state(
                "persistent mcu did not stop; refusing to release resources "
                "while the device loop is alive",
                __FILE__, __LINE__);
        }
        if (hipStreamSynchronize(control_stream_) != hipSuccess) {
            note(Status::hip_error("hipStreamSynchronize(persistent)", "sync failed",
                                   __FILE__, __LINE__));
        }
        launched_ = false;
    }
    void* host = host_allocation_;
    host_allocation_ = nullptr;
    host_state_ = nullptr;
    device_state_ = nullptr;
    if (host != nullptr && hipFreeHost(host) != hipSuccess) {
        note(Status::hip_error("hipFreeHost(persistent state)", "free failed",
                               __FILE__, __LINE__));
    }
    return first;
}

}  // namespace ps::runtime::gpu_mcu
