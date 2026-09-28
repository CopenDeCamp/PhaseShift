#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/request_ingress.h>
#include <phaseshift/runtime/gpu_mcu/batch_binding.h>
#include <phaseshift/runtime/gpu_mcu/batch_commit.h>
#include <phaseshift/runtime/gpu_mcu/batch_planner.h>
#include <phaseshift/runtime/gpu_mcu/execution_bridge.h>
#include <phaseshift/runtime/gpu_mcu/output_ring.h>
#include <phaseshift/runtime/gpu_mcu/slot_runtime.h>
#include <phaseshift/runtime/gpu_mcu/kv_page_allocator.h>
#include <phaseshift/runtime/gpu_mcu/micro_fsm.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct alignas(64) GpuMcuPersistentState {
    uint32_t stop_requested = 0;
    uint32_t started = 0;
    uint32_t emit_enabled = 0;
    uint32_t idle_sleep = 64;
    uint64_t heartbeat = 0;
    uint64_t iterations = 0;
    uint64_t last_loop_ts = 0;
    uint64_t submit_request = 0;
    uint64_t published = 0;
    uint64_t publish_failures = 0;
    DeviceAqlQueueView queue{};
    const GpuAqlPacketTemplate* templates = nullptr;
    uint32_t template_count = 0;
    uint32_t emit_streaming = 0;
    uint32_t barrier = 1;
    uint32_t write_base_valid = 0;
    uint32_t reserved0 = 0;
    uint64_t write_base = 0;
    uint64_t* publish_ts = nullptr;
    uint64_t* doorbell_ts = nullptr;

    uint32_t request_runtime_enabled = 0;
    uint32_t request_reserved0 = 0;
    ControlRing* request_input_ring = nullptr;
    ControlRing* request_event_ring = nullptr;
    GpuMcuRequestIngressEntry* request_ingress_entries = nullptr;
    uint32_t request_ingress_capacity = 0;
    GpuMcuSlotState* request_slots = nullptr;
    uint32_t request_max_slots = 0;
    RequestHandle* runnable_handles = nullptr;
    uint32_t runnable_capacity = 0;
    uint32_t runnable_count = 0;
    uint32_t scheduler_dirty = 0;
    uint64_t request_input_position = 0;
    uint64_t request_event_position = 0;
    uint64_t scheduler_epoch = 0;
    uint64_t commands_consumed = 0;
    uint64_t events_published = 0;
    uint64_t commands_rejected = 0;
    uint32_t max_control_commands_per_boundary = 0;
    uint32_t pending_event_valid = 0;
    uint32_t request_reserved1 = 0;
    ControlRingPayload pending_event{};

    uint32_t batch_plan_enabled = 0;
    uint32_t batch_plan_reserved = 0;
    DeviceBatchContext* batch_context = nullptr;
    DeviceRequestDescriptor* batch_requests = nullptr;
    uint32_t* batch_row_sequence_slots = nullptr;
    uint32_t* batch_row_positions = nullptr;
    uint32_t batch_max_requests = 0;
    uint32_t batch_max_rows = 0;
    uint32_t batch_max_prefill_rows_per_slot = 0;
    uint64_t* batch_plan_epoch = nullptr;
    uint64_t batch_plan_source_epoch = 0;
    GpuMcuBatchPlanTelemetry batch_plan_telemetry{};

    uint32_t batch_binding_enabled = 0;
    uint32_t batch_binding_reserved = 0;
    GpuMcuSlotBinding* slot_bindings = nullptr;
    uint32_t binding_max_slots = 0;
    int32_t* batch_token_ids = nullptr;
    uint32_t* batch_output_rows = nullptr;
    DeviceSamplingParams* batch_output_sampling_params = nullptr;
    uint32_t batch_max_outputs = 0;
    uint64_t* batch_ready_epoch = nullptr;
    uint64_t batch_binding_source_epoch = 0;
    GpuMcuBatchBindingTelemetry batch_binding_telemetry{};

    uint32_t execution_enabled = 0;
    uint32_t execution_wrap_faulted = 0;
    GpuMcuFsmState* execution_fsm = nullptr;
    uint32_t execution_retained_count = 0;
    uint64_t execution_epoch = 0;
    uint64_t active_batch_ready_epoch = 0;
    uint64_t last_committed_execution_epoch = 0;
    GpuMcuExecutionTelemetry execution_telemetry{};
    const int32_t* execution_sampled_tokens = nullptr;
    uint32_t execution_sampled_capacity = 0u;
    const uint32_t* execution_verify_counts = nullptr;
    uint32_t execution_verify_capacity = 0u;
    const McuInvocationPatch* execution_patches = nullptr;
    uint32_t execution_patch_count = 0u;
    GpuMcuBatchCommitTelemetry commit_telemetry{};
    uint32_t output_enabled = 0;
    uint32_t output_reserved = 0;
    OutputRing* output_ring = nullptr;
    uint64_t output_position = 0;
    GpuMcuSlotRuntimeState* slot_runtime = nullptr;

    GpuMcuPendingAdmission* admission_queue = nullptr;
    uint32_t admission_capacity = 0u;
    uint32_t admission_head = 0u;
    uint32_t admission_count = 0u;
    uint32_t admission_dropped = 0u;

    uint32_t resources_enabled = 0u;
    uint32_t resources_reserved = 0u;
    GpuMcuKvPagePoolView kv_pages{};
    GpuMcuBlockTableView kv_blocks{};
    DeviceSequenceResourceManagerView sequence_resources{};
};

struct GpuMcuPersistentStartConfig {
    uint32_t idle_sleep = 64;
    uint32_t emit_enabled = 0;
    uint64_t submit_request = 0;
    DeviceAqlQueueView queue{};
    const GpuAqlPacketTemplate* templates = nullptr;
    uint32_t template_count = 0;
    uint32_t emit_streaming = 0;
    uint32_t barrier = 1;
    uint64_t* publish_ts = nullptr;
    uint64_t* doorbell_ts = nullptr;

    uint32_t request_runtime_enabled = 0;
    ControlRing* request_input_ring = nullptr;
    ControlRing* request_event_ring = nullptr;
    GpuMcuRequestIngressEntry* request_ingress_entries = nullptr;
    uint32_t request_ingress_capacity = 0;
    GpuMcuSlotState* request_slots = nullptr;
    uint32_t request_max_slots = 0;
    RequestHandle* runnable_handles = nullptr;
    uint32_t runnable_capacity = 0;
    uint32_t max_control_commands_per_boundary = 0;

    uint32_t batch_plan_enabled = 0;
    DeviceBatchContext* batch_context = nullptr;
    DeviceRequestDescriptor* batch_requests = nullptr;
    uint32_t* batch_row_sequence_slots = nullptr;
    uint32_t* batch_row_positions = nullptr;
    uint32_t batch_max_requests = 0;
    uint32_t batch_max_rows = 0;
    uint32_t batch_max_prefill_rows_per_slot = 0;
    uint64_t* batch_plan_epoch = nullptr;

    uint32_t batch_binding_enabled = 0;
    GpuMcuSlotBinding* slot_bindings = nullptr;
    uint32_t binding_max_slots = 0;
    int32_t* batch_token_ids = nullptr;
    uint32_t* batch_output_rows = nullptr;
    DeviceSamplingParams* batch_output_sampling_params = nullptr;
    uint32_t batch_max_outputs = 0;
    uint64_t* batch_ready_epoch = nullptr;

    uint32_t execution_enabled = 0;
    GpuMcuFsmState* execution_fsm = nullptr;
    uint32_t execution_retained_count = 0;
    const int32_t* execution_sampled_tokens = nullptr;
    uint32_t execution_sampled_capacity = 0u;
    const uint32_t* execution_verify_counts = nullptr;
    uint32_t execution_verify_capacity = 0u;
    const McuInvocationPatch* execution_patches = nullptr;
    uint32_t execution_patch_count = 0u;

    uint32_t output_enabled = 0;
    OutputRing* output_ring = nullptr;
    GpuMcuSlotRuntimeState* slot_runtime = nullptr;

    GpuMcuPendingAdmission* admission_queue = nullptr;
    uint32_t admission_capacity = 0u;
    GpuMcuKvPagePoolView kv_pages{};
    GpuMcuBlockTableView kv_blocks{};
    DeviceSequenceResourceManagerView sequence_resources{};
};

inline GpuMcuPersistentStartConfig gpu_mcu_capture_start_config(
    const GpuMcuPersistentState& state) noexcept {
    GpuMcuPersistentStartConfig config{};
    config.idle_sleep = state.idle_sleep;
    config.emit_enabled = state.emit_enabled;
    config.submit_request = state.submit_request;
    config.queue = state.queue;
    config.templates = state.templates;
    config.template_count = state.template_count;
    config.emit_streaming = state.emit_streaming;
    config.barrier = state.barrier;
    config.publish_ts = state.publish_ts;
    config.doorbell_ts = state.doorbell_ts;
    config.request_runtime_enabled = state.request_runtime_enabled;
    config.request_input_ring = state.request_input_ring;
    config.request_event_ring = state.request_event_ring;
    config.request_ingress_entries = state.request_ingress_entries;
    config.request_ingress_capacity = state.request_ingress_capacity;
    config.request_slots = state.request_slots;
    config.request_max_slots = state.request_max_slots;
    config.runnable_handles = state.runnable_handles;
    config.runnable_capacity = state.runnable_capacity;
    config.max_control_commands_per_boundary =
        state.max_control_commands_per_boundary;
    config.batch_plan_enabled = state.batch_plan_enabled;
    config.batch_context = state.batch_context;
    config.batch_requests = state.batch_requests;
    config.batch_row_sequence_slots = state.batch_row_sequence_slots;
    config.batch_row_positions = state.batch_row_positions;
    config.batch_max_requests = state.batch_max_requests;
    config.batch_max_rows = state.batch_max_rows;
    config.batch_max_prefill_rows_per_slot =
        state.batch_max_prefill_rows_per_slot;
    config.batch_plan_epoch = state.batch_plan_epoch;
    config.batch_binding_enabled = state.batch_binding_enabled;
    config.slot_bindings = state.slot_bindings;
    config.binding_max_slots = state.binding_max_slots;
    config.batch_token_ids = state.batch_token_ids;
    config.batch_output_rows = state.batch_output_rows;
    config.batch_output_sampling_params = state.batch_output_sampling_params;
    config.batch_max_outputs = state.batch_max_outputs;
    config.batch_ready_epoch = state.batch_ready_epoch;
    config.execution_enabled = state.execution_enabled;
    config.execution_fsm = state.execution_fsm;
    config.execution_retained_count = state.execution_retained_count;
    config.execution_sampled_tokens = state.execution_sampled_tokens;
    config.execution_sampled_capacity = state.execution_sampled_capacity;
    config.execution_verify_counts = state.execution_verify_counts;
    config.execution_verify_capacity = state.execution_verify_capacity;
    config.execution_patches = state.execution_patches;
    config.execution_patch_count = state.execution_patch_count;
    config.output_enabled = state.output_enabled;
    config.output_ring = state.output_ring;
    config.slot_runtime = state.slot_runtime;
    config.admission_queue = state.admission_queue;
    config.admission_capacity = state.admission_capacity;
    config.kv_pages = state.kv_pages;
    config.kv_blocks = state.kv_blocks;
    config.sequence_resources = state.sequence_resources;
    return config;
}

inline void gpu_mcu_restore_start_config(
    GpuMcuPersistentState& state,
    const GpuMcuPersistentStartConfig& config) noexcept {
    state.idle_sleep = config.idle_sleep;
    state.emit_enabled = config.emit_enabled;
    state.submit_request = config.submit_request;
    state.queue = config.queue;
    state.templates = config.templates;
    state.template_count = config.template_count;
    state.emit_streaming = config.emit_streaming;
    state.barrier = config.barrier;
    state.publish_ts = config.publish_ts;
    state.doorbell_ts = config.doorbell_ts;
    state.request_runtime_enabled = config.request_runtime_enabled;
    state.request_input_ring = config.request_input_ring;
    state.request_event_ring = config.request_event_ring;
    state.request_ingress_entries = config.request_ingress_entries;
    state.request_ingress_capacity = config.request_ingress_capacity;
    state.request_slots = config.request_slots;
    state.request_max_slots = config.request_max_slots;
    state.runnable_handles = config.runnable_handles;
    state.runnable_capacity = config.runnable_capacity;
    state.max_control_commands_per_boundary =
        config.max_control_commands_per_boundary;
    state.batch_plan_enabled = config.batch_plan_enabled;
    state.batch_context = config.batch_context;
    state.batch_requests = config.batch_requests;
    state.batch_row_sequence_slots = config.batch_row_sequence_slots;
    state.batch_row_positions = config.batch_row_positions;
    state.batch_max_requests = config.batch_max_requests;
    state.batch_max_rows = config.batch_max_rows;
    state.batch_max_prefill_rows_per_slot =
        config.batch_max_prefill_rows_per_slot;
    state.batch_plan_epoch = config.batch_plan_epoch;
    state.batch_binding_enabled = config.batch_binding_enabled;
    state.slot_bindings = config.slot_bindings;
    state.binding_max_slots = config.binding_max_slots;
    state.batch_token_ids = config.batch_token_ids;
    state.batch_output_rows = config.batch_output_rows;
    state.batch_output_sampling_params = config.batch_output_sampling_params;
    state.batch_max_outputs = config.batch_max_outputs;
    state.batch_ready_epoch = config.batch_ready_epoch;
    state.execution_enabled = config.execution_enabled;
    state.execution_fsm = config.execution_fsm;
    state.execution_retained_count = config.execution_retained_count;
    state.execution_sampled_tokens = config.execution_sampled_tokens;
    state.execution_sampled_capacity = config.execution_sampled_capacity;
    state.execution_verify_counts = config.execution_verify_counts;
    state.execution_verify_capacity = config.execution_verify_capacity;
    state.execution_patches = config.execution_patches;
    state.execution_patch_count = config.execution_patch_count;
    state.output_enabled = config.output_enabled;
    state.output_ring = config.output_ring;
    state.slot_runtime = config.slot_runtime;
    state.admission_queue = config.admission_queue;
    state.admission_capacity = config.admission_capacity;
    state.kv_pages = config.kv_pages;
    state.kv_blocks = config.kv_blocks;
    state.sequence_resources = config.sequence_resources;
    if (config.sequence_resources.resources != nullptr) {
        state.resources_enabled = 1u;
    }
    if (config.request_runtime_enabled != 0u) {
        state.scheduler_dirty = 1u;
    }
}

__global__ void gpu_mcu_persistent_loop_kernel(GpuMcuPersistentState* state);

__device__ __forceinline__ bool gpu_mcu_scheduler_flush_pending(
    GpuMcuPersistentState* state) noexcept {
    if (state->pending_event_valid == 0u) return false;
    if (state->request_event_ring == nullptr) return false;
    if (!control_ring_try_push(state->request_event_ring,
                               state->request_event_position,
                               state->pending_event)) {
        return false;
    }
    state->pending_event_valid = 0u;
    state->events_published += 1u;
    return true;
}

// Binds a freshly allocated device sequence resource to the slot the CLAIMED
// event names. The slot keeps only the opaque handle.
__device__ __forceinline__ void gpu_mcu_resource_commit_claim(
    GpuMcuPersistentState* state, const ControlRingPayload& payload) noexcept {
    if (state->resources_enabled == 0u) return;
    GpuMcuEvent event{};
    decode_event(payload, event);
    if (event.opcode !=
        static_cast<uint32_t>(GpuMcuEventOpcode::claimed)) {
        return;
    }
    if (event.slot >= state->request_max_slots) return;
    GpuMcuSlotState& slot = state->request_slots[event.slot];
    if (!gpu_mcu_slot_matches_handle(
            slot, RequestHandle{event.slot, event.generation})) {
        return;
    }
    uint64_t handle = 0u;
    if (!gpu_mcu_resource_allocate(state->sequence_resources, &handle)) {
        return;
    }
    slot.kv_sequence_handle = handle;
    slot.gdn_state_handle = handle;
    state->execution_telemetry.resources_allocated += 1u;
    __threadfence_system();
}

__device__ __forceinline__ bool gpu_mcu_admission_enqueue(
    GpuMcuPersistentState* state,
    uint64_t request_id,
    uint64_t descriptor_handle) noexcept {
    if (state->admission_queue == nullptr || state->admission_capacity == 0u) {
        return false;
    }
    if (state->admission_count >= state->admission_capacity) return false;
    const uint32_t index = (state->admission_head + state->admission_count) %
                           state->admission_capacity;
    state->admission_queue[index].request_id = request_id;
    state->admission_queue[index].descriptor_handle = descriptor_handle;
    state->admission_queue[index].descriptor_generation = 0u;
    __threadfence_system();
    state->admission_count += 1u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_admission_remove(
    GpuMcuPersistentState* state,
    uint64_t request_id,
    uint64_t descriptor_handle) noexcept {
    if (state->admission_queue == nullptr ||
        state->admission_capacity == 0u) {
        return false;
    }
    for (uint32_t i = 0; i < state->admission_count; ++i) {
        const uint32_t index =
            (state->admission_head + i) % state->admission_capacity;
        const GpuMcuPendingAdmission& pending = state->admission_queue[index];
        if (pending.request_id != request_id) continue;
        if (pending.descriptor_handle != descriptor_handle) continue;
        for (uint32_t j = i; j + 1u < state->admission_count; ++j) {
            const uint32_t from =
                (state->admission_head + j + 1u) % state->admission_capacity;
            const uint32_t to =
                (state->admission_head + j) % state->admission_capacity;
            state->admission_queue[to] = state->admission_queue[from];
        }
        state->admission_count -= 1u;
        __threadfence_system();
        return true;
    }
    return false;
}

// Reserves the kv pages the next step of every live slot needs. A slot whose
// pages cannot be grown is blocked for this loop only; the other slots continue.
__device__ __forceinline__ bool gpu_mcu_resource_reserve_pass(
    GpuMcuPersistentState* state) noexcept {
    if (state->resources_enabled == 0u || state->slot_runtime == nullptr) {
        return false;
    }
    if (state->kv_pages.capacity == 0u || state->request_slots == nullptr) {
        return false;
    }
    bool changed = false;
    for (uint32_t i = 0; i < state->request_max_slots; ++i) {
        GpuMcuSlotState& slot = state->request_slots[i];
        if (!gpu_mcu_slot_has_active_request(slot)) continue;
        if (slot.terminal_reason !=
            static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
            continue;
        }
        if (slot.kv_sequence_handle == 0u) {
            uint64_t handle = 0u;
            if (gpu_mcu_resource_allocate(state->sequence_resources,
                                          &handle)) {
                slot.kv_sequence_handle = handle;
                slot.gdn_state_handle = handle;
                state->execution_telemetry.resources_allocated += 1u;
                __threadfence_system();
                changed = true;
            }
            if (slot.kv_sequence_handle == 0u) continue;
        }
        uint32_t next_position = slot.sequence_length;
        uint32_t next_rows = 1u;
        bool verify_phase = false;
        if (slot.phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::prefill)) {
            next_position = slot.prefill_position;
            next_rows = slot.prompt_length > slot.prefill_position
                            ? slot.prompt_length - slot.prefill_position
                            : 1u;
            if (state->batch_max_prefill_rows_per_slot != 0u &&
                next_rows > state->batch_max_prefill_rows_per_slot) {
                next_rows = state->batch_max_prefill_rows_per_slot;
            }
            if (next_rows == 0u) next_rows = 1u;
        } else if (slot.phase ==
                   gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify)) {
            const uint32_t candidates = slot.verify_candidate_count;
            if (candidates == 0u ||
                candidates > kGpuMcuMaxVerifyCandidates) {
                continue;
            }
            verify_phase = true;
            next_position = slot.committed_position;
            next_rows = candidates;
        }
        const uint32_t rows_per_page =
            gpu_mcu_block_rows_per_page(state->kv_blocks);
        uint32_t required_pages =
            (next_position + next_rows + rows_per_page - 1u) / rows_per_page;
        uint32_t sequence_slot = 0u;
        (void)gpu_mcu_resource_resolve(state->sequence_resources,
                                       slot.kv_sequence_handle,
                                       &sequence_slot, nullptr);
        GpuMcuSlotRuntimeState& meta = state->slot_runtime[i];
        const uint32_t before = gpu_mcu_resource_page_count(
            state->sequence_resources, slot.kv_sequence_handle);
        const bool new_txn = verify_phase && meta.verify_txn_active == 0u;
        if (new_txn) {
            meta.verify_base_page_count = before;
            meta.verify_reserved_page_count = 0u;
        }
        const bool reserved = gpu_mcu_resource_reserve_kv_pages(
            state->sequence_resources, state->kv_pages, state->kv_blocks,
            slot.kv_sequence_handle, sequence_slot, required_pages);
        if (!reserved) {
            if (new_txn) {
                meta.verify_base_page_count = 0u;
                meta.verify_reserved_page_count = 0u;
                meta.verify_txn_epoch = 0u;
            }
            if (meta.resource_blocked == 0u) {
                meta.resource_blocked = 1u;
                state->execution_telemetry.resource_blocked_events += 1u;
                changed = true;
            }
        } else {
            if (verify_phase) {
                const uint32_t after = gpu_mcu_resource_page_count(
                    state->sequence_resources, slot.kv_sequence_handle);
                meta.verify_base_page_count =
                    new_txn ? before : meta.verify_base_page_count;
                meta.verify_reserved_page_count =
                    after > meta.verify_base_page_count
                        ? after - meta.verify_base_page_count
                        : 0u;
                meta.verify_txn_active = 1u;
                if (new_txn) meta.verify_txn_epoch = state->scheduler_epoch;
            }
            if (meta.resource_blocked != 0u) {
                meta.resource_blocked = 0u;
                state->execution_telemetry.resource_unblocked_events += 1u;
                changed = true;
            }
        }
        if (reserved) {
            state->execution_telemetry.kv_pages_reserved +=
                gpu_mcu_resource_page_count(state->sequence_resources,
                                            slot.kv_sequence_handle) -
                before;
        }
    }
    return changed;
}

__device__ __forceinline__ bool gpu_mcu_admission_admit(
    GpuMcuPersistentState* state) noexcept {
    if (state->admission_queue == nullptr || state->admission_count == 0u) {
        return false;
    }
    if (state->pending_event_valid != 0u) return false;
    const GpuMcuPendingAdmission pending =
        state->admission_queue[state->admission_head];
    const detail::GpuMcuCommandApplyResult result =
        detail::gpu_mcu_admit_queued_entry(
            state->request_ingress_entries, state->request_ingress_capacity,
            pending, state->request_slots, state->request_max_slots,
            state->slot_bindings, state->binding_max_slots);
    if (result.slot_exhausted || !result.has_event) return false;
    gpu_mcu_resource_commit_claim(state, result.event);
    if (control_ring_try_push(state->request_event_ring,
                              state->request_event_position, result.event)) {
        state->events_published += 1u;
    } else {
        state->pending_event = result.event;
        state->pending_event_valid = 1u;
    }
    state->admission_head =
        (state->admission_head + 1u) % state->admission_capacity;
    state->admission_count -= 1u;
    if (result.scheduler_dirty) state->scheduler_dirty = 1u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_scheduler_consume_one(
    GpuMcuPersistentState* state) noexcept {
    if (state->request_input_ring == nullptr) return false;
    ControlRingPayload payload{};
    if (!control_ring_try_pop(state->request_input_ring,
                              state->request_input_position, payload)) {
        return false;
    }
    state->commands_consumed += 1u;
    GpuMcuControlCommand command{};
    decode_control_command(payload, command);
    const detail::GpuMcuCommandApplyResult result =
        detail::gpu_mcu_apply_control_command(
            command, state->request_ingress_entries,
            state->request_ingress_capacity, state->request_slots,
            state->request_max_slots, state->slot_bindings,
            state->binding_max_slots);
    if (result.scheduler_dirty) state->scheduler_dirty = 1u;
    if (result.has_event) {
        gpu_mcu_resource_commit_claim(state, result.event);
    }
    if (result.slot_exhausted) {
        if (gpu_mcu_admission_enqueue(state, command.request_id,
                                      command.descriptor_handle)) {
            return true;
        }
        const detail::GpuMcuCommandApplyResult full =
            detail::gpu_mcu_make_rejected(
                command.request_id,
                static_cast<uint32_t>(
                    GpuMcuRejectReason::admission_backlog_full));
        state->admission_dropped += 1u;
        state->commands_rejected += 1u;
        if (control_ring_try_push(state->request_event_ring,
                                  state->request_event_position,
                                  full.event)) {
            state->events_published += 1u;
            return true;
        }
        state->pending_event = full.event;
        state->pending_event_valid = 1u;
        return true;
    }
    if (result.cancel_not_found) {
        if (gpu_mcu_admission_remove(state, command.request_id,
                                     command.descriptor_handle)) {
            if (command.descriptor_handle >= 1u &&
                command.descriptor_handle <= state->request_ingress_capacity) {
                state->request_ingress_entries[command.descriptor_handle - 1u]
                    .state = kGpuMcuIngressClaimed;
            }
            GpuMcuEvent event{};
            event.opcode =
                static_cast<uint32_t>(GpuMcuEventOpcode::cancel_accepted);
            event.request_id = command.request_id;
            event.slot = kInvalidRequestSlot;
            event.generation = kInvalidRequestGeneration;
            event.descriptor_handle = command.descriptor_handle;
            const ControlRingPayload payload = encode_event(event);
            if (control_ring_try_push(state->request_event_ring,
                                      state->request_event_position,
                                      payload)) {
                state->events_published += 1u;
                return true;
            }
            state->pending_event = payload;
            state->pending_event_valid = 1u;
            return true;
        }
    }
    if (!result.has_event) return true;
    if (control_ring_try_push(state->request_event_ring,
                              state->request_event_position, result.event)) {
        state->events_published += 1u;
        return true;
    }
    GpuMcuEvent rejected{};
    decode_event(result.event, rejected);
    if (rejected.opcode ==
        static_cast<uint32_t>(GpuMcuEventOpcode::rejected)) {
        state->commands_rejected += 1u;
    }
    state->pending_event = result.event;
    state->pending_event_valid = 1u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_scheduler_reconcile(
    GpuMcuPersistentState* state) noexcept {
    bool changed = false;
    for (uint32_t i = 0; i < state->request_max_slots; ++i) {
        GpuMcuSlotState& slot = state->request_slots[i];
        if (slot.cancel_requested != 0u &&
            slot.terminal_reason ==
                static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
            slot.terminal_reason =
                static_cast<uint32_t>(GpuMcuTerminalReason::cancelled);
            changed = true;
        }
    }
    if (changed) state->scheduler_dirty = 1u;
    return changed;
}

__device__ __forceinline__ bool gpu_mcu_scheduler_is_runnable(
    const GpuMcuSlotState& slot) noexcept {
    if (!gpu_mcu_slot_phase_valid(slot.phase)) return false;
    if (slot.phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::idle)) {
        return false;
    }
    if (slot.terminal_reason !=
        static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
        return false;
    }
    if (slot.cancel_requested != 0u) return false;
    if (slot.output_blocked != 0u) return false;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_scheduler_rebuild_snapshot(
    GpuMcuPersistentState* state) noexcept {
    uint32_t count = 0u;
    for (uint32_t i = 0; i < state->request_max_slots; ++i) {
        const GpuMcuSlotState& slot = state->request_slots[i];
        if (!gpu_mcu_scheduler_is_runnable(slot)) continue;
        if (state->slot_runtime != nullptr &&
            state->slot_runtime[i].resource_blocked != 0u) {
            continue;
        }
        if (count >= state->runnable_capacity) break;
        state->runnable_handles[count] = slot.handle;
        ++count;
    }
    __threadfence_system();
    state->runnable_count = count;
    __threadfence_system();
    const uint64_t epoch = state->scheduler_epoch + 1u;
    __scoped_atomic_store_n(&state->scheduler_epoch, epoch, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    state->scheduler_dirty = 0u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_scheduler_boundary(
    GpuMcuPersistentState* state) noexcept {
    bool progressed = false;
    if (state->output_enabled != 0u && state->output_ring != nullptr &&
        state->request_slots != nullptr) {
        const GpuMcuOutputFlushResult flush = gpu_mcu_flush_slot_outputs(
            state->output_ring, state->output_position, state->request_slots,
            state->request_max_slots);
        state->execution_telemetry.output_tokens_published +=
            flush.flushed_tokens;
        state->execution_telemetry.output_backpressure_events +=
            flush.blocked_slots;
        if (flush.flushed_tokens != 0u || flush.unblocked_slots != 0u ||
            flush.blocked_slots != 0u) {
            progressed = true;
        }
        if (state->slot_runtime != nullptr) {
            const GpuMcuTerminalPublishResult terminals =
                gpu_mcu_publish_slot_terminals(
                    state->output_ring, state->output_position,
                    state->request_slots, state->slot_runtime,
                    state->request_max_slots);
            if (terminals.published != 0u) {
                progressed = true;
            }
            if (state->slot_runtime != nullptr) {
                const GpuMcuSlotReleaseResult released =
                    gpu_mcu_release_terminal_slots(
                        state->request_slots, state->slot_runtime,
                        state->slot_bindings, state->request_max_slots,
                        state->binding_max_slots, state->sequence_resources,
                        state->resources_enabled, state->kv_pages,
                        state->kv_blocks);
                state->execution_telemetry.resources_released +=
                    released.released_resources;
                state->execution_telemetry.kv_pages_released +=
                    released.released_pages;
                if (released.released != 0u) {
                    progressed = true;
                    state->scheduler_dirty = 1u;
                }
            }
            if (gpu_mcu_admission_admit(state)) {
                progressed = true;
            }
        }
    }

    if (state->request_runtime_enabled == 0u) return progressed;

    uint32_t command_budget = state->max_control_commands_per_boundary;
    if (command_budget == 0u) command_budget = 1u;
    for (uint32_t drained = 0; drained < command_budget; ++drained) {
        if (gpu_mcu_scheduler_flush_pending(state)) {
            progressed = true;
            continue;
        }
        if (state->pending_event_valid != 0u) break;
        if (!gpu_mcu_scheduler_consume_one(state)) break;
        progressed = true;
    }

    if (gpu_mcu_scheduler_reconcile(state)) progressed = true;

    if (gpu_mcu_resource_reserve_pass(state)) {
        state->scheduler_dirty = 1u;
        progressed = true;
    }

    if (state->scheduler_dirty != 0u) {
        gpu_mcu_scheduler_rebuild_snapshot(state);
        progressed = true;
    }

    if (state->batch_plan_enabled != 0u &&
        state->batch_plan_source_epoch != state->scheduler_epoch) {
        GpuMcuBatchPlanLimits limits{};
        limits.max_requests = state->batch_max_requests;
        limits.max_rows = state->batch_max_rows;
        limits.max_prefill_rows_per_slot =
            state->batch_max_prefill_rows_per_slot;
        GpuMcuBatchPlanView view{};
        view.context = state->batch_context;
        view.requests = state->batch_requests;
        view.row_sequence_slots = state->batch_row_sequence_slots;
        view.row_positions = state->batch_row_positions;
        view.batch_plan_epoch = state->batch_plan_epoch;
        view.resources = state->sequence_resources;
        view.resolve_sequence_slots = state->resources_enabled;
        (void)gpu_mcu_build_batch_geometry(state->runnable_handles,
                                          state->runnable_count,
                                          state->request_slots,
                                          state->request_max_slots, limits, view,
                                          &state->batch_plan_telemetry);
        state->batch_plan_source_epoch = state->scheduler_epoch;
        progressed = true;
    }

    if (state->batch_binding_enabled != 0u &&
        state->batch_binding_source_epoch !=
            state->batch_plan_source_epoch) {
        GpuMcuBatchBindingView binding_view{};
        binding_view.context = state->batch_context;
        binding_view.requests = state->batch_requests;
        binding_view.token_ids = state->batch_token_ids;
        binding_view.output_rows = state->batch_output_rows;
        binding_view.output_sampling_params =
            state->batch_output_sampling_params;
        binding_view.slots = state->request_slots;
        binding_view.bindings = state->slot_bindings;
        binding_view.max_slots = state->binding_max_slots;
        binding_view.max_outputs = state->batch_max_outputs;
        binding_view.batch_ready_epoch = state->batch_ready_epoch;
        (void)gpu_mcu_bind_batch_io(binding_view,
                                    &state->batch_binding_telemetry);
        state->batch_binding_source_epoch = state->batch_plan_source_epoch;
        progressed = true;
    }
    return progressed;
}

class GpuMcuPersistentMcu {
public:
    GpuMcuPersistentMcu() = default;
    ~GpuMcuPersistentMcu() noexcept { (void)shutdown(); }

    GpuMcuPersistentMcu(const GpuMcuPersistentMcu&) = delete;
    GpuMcuPersistentMcu& operator=(const GpuMcuPersistentMcu&) = delete;

    GpuMcuPersistentMcu(GpuMcuPersistentMcu&& other) noexcept { move_from(other); }
    GpuMcuPersistentMcu& operator=(GpuMcuPersistentMcu&& other) noexcept {
        if (this != &other) {
            (void)shutdown();
            move_from(other);
        }
        return *this;
    }

    static Result<GpuMcuPersistentMcu> create(int device, hipStream_t control_stream);

    bool valid() const noexcept { return host_state_ != nullptr; }
    int device() const noexcept { return device_; }

    Status start();
    bool running() const noexcept;
    uint64_t heartbeat() const noexcept;
    uint64_t iterations() const noexcept;

    Status configure_batch_plan(DeviceBatchContext* context,
                                DeviceRequestDescriptor* requests,
                                uint32_t* row_sequence_slots,
                                uint32_t* row_positions,
                                uint32_t max_requests,
                                uint32_t max_rows,
                                uint32_t max_prefill_rows_per_slot,
                                uint64_t* batch_plan_epoch);

    Status configure_batch_binding(GpuMcuSlotBinding* bindings,
                                    uint32_t binding_max_slots,
                                    int32_t* token_ids,
                                    uint32_t* output_rows,
                                    DeviceSamplingParams* output_sampling_params,
                                    uint32_t max_outputs,
                                    uint64_t* batch_ready_epoch);

    Status configure_output(OutputRing* ring);
    Status configure_slot_runtime(GpuMcuSlotRuntimeState* runtime);
    Status configure_admission(GpuMcuPendingAdmission* queue,
                               uint32_t capacity);

    Status configure_control_drain(uint32_t max_commands_per_boundary);

    Status configure_kv_blocks(uint32_t* block_tables,
                               uint32_t block_table_stride,
                               uint32_t page_tokens);
    Status configure_resources(const GpuMcuKvPagePoolView& kv_pages,
                               const DeviceSequenceResourceManagerView&
                                   sequence_resources);

    Status configure_execution(GpuMcuFsmState* fsm,
                               uint32_t retained_count,
                               uint64_t* batch_ready_epoch,
                               const int32_t* sampled_tokens,
                               uint32_t sampled_capacity,
                               const uint32_t* verify_counts,
                               uint32_t verify_capacity);

    Status configure_execution_patches(const McuInvocationPatch* patches,
                                       uint32_t count);

    Status configure_commit_slots(GpuMcuSlotState* slots,
                                  uint32_t max_slots,
                                  GpuMcuSlotBinding* bindings,
                                  uint32_t binding_max_slots);

    Status configure_request_runtime(
        ControlRing* input_ring,
        ControlRing* event_ring,
        GpuMcuRequestIngressEntry* ingress_entries,
        uint32_t ingress_capacity,
        GpuMcuSlotState* slots,
        uint32_t max_slots,
        RequestHandle* runnable_handles,
        uint32_t runnable_capacity);

    bool request_runtime_enabled() const noexcept;
    ControlRing* request_input_ring() const noexcept;
    ControlRing* request_event_ring() const noexcept;
    GpuMcuSlotState* request_slots() const noexcept;
    uint32_t request_max_slots() const noexcept;
    RequestHandle* runnable_handles() const noexcept;
    uint32_t runnable_capacity() const noexcept;
    uint32_t runnable_count() const noexcept;
    uint64_t scheduler_epoch() const noexcept;
    uint64_t commands_consumed() const noexcept;
    uint64_t events_published() const noexcept;
    bool pending_event_valid() const noexcept;
    bool batch_plan_enabled() const noexcept;
    uint64_t batch_plan_epoch() const noexcept;
    uint32_t batch_plan_source_epoch() const noexcept;
    uint32_t batch_plan_planned_requests() const noexcept;
    uint32_t batch_plan_planned_rows() const noexcept;
    bool batch_binding_enabled() const noexcept;
    uint64_t batch_ready_epoch() const noexcept;
    uint32_t batch_binding_bound_requests() const noexcept;
    uint32_t batch_binding_bound_outputs() const noexcept;
    bool execution_enabled() const noexcept;
    const McuInvocationPatch* execution_patches() const noexcept;
    uint32_t execution_patch_count() const noexcept;
    uint64_t execution_epoch() const noexcept;
    uint64_t batches_dispatched() const noexcept;
    uint64_t batches_completed() const noexcept;
    uint64_t batches_failed() const noexcept;
    uint64_t batches_committed() const noexcept;
    uint64_t committed_tokens() const noexcept;
    uint64_t commit_stale_handles() const noexcept;
    uint64_t commit_pending_overflow() const noexcept;
    uint64_t last_committed_execution_epoch() const noexcept;
    uint64_t autonomous_loops() const noexcept;
    bool output_enabled() const noexcept;
    bool slot_runtime_enabled() const noexcept;
    bool admission_enabled() const noexcept;
    uint32_t admission_count() const noexcept;
    uint32_t admission_dropped() const noexcept;
    uint32_t control_drain_budget() const noexcept;
    bool resources_enabled() const noexcept;
    GpuMcuExecutionTelemetry execution_telemetry() const noexcept;
    uint64_t output_tokens_published() const noexcept;
    uint64_t output_backpressure_events() const noexcept;

    Status enable_emit(const DeviceAqlQueueView& queue,
                       const GpuAqlPacketTemplate* device_templates,
                       uint32_t template_count,
                       uint64_t* device_publish_ts,
                       uint64_t* device_doorbell_ts,
                       uint32_t idle_sleep = 8,
                       bool streaming = false,
                       bool barrier = true);
    Status request_emit(uint64_t published_target);
    uint64_t publish_target() const noexcept;
    uint64_t published() const noexcept;
    uint64_t publish_failures() const noexcept;

    Status request_stop();
    Status wait_stopped(uint32_t timeout_ms);

    GpuMcuPersistentState* device_state() const noexcept { return device_state_; }

    Status shutdown() noexcept;

private:
    void move_from(GpuMcuPersistentMcu& other) noexcept;

    int device_ = -1;
    void* host_allocation_ = nullptr;
    GpuMcuPersistentState* host_state_ = nullptr;
    GpuMcuPersistentState* device_state_ = nullptr;
    hipStream_t control_stream_ = nullptr;
    bool launched_ = false;
};

}  // namespace ps::runtime::gpu_mcu
