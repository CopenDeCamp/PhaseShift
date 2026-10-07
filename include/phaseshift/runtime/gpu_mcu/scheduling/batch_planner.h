#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/sequence_resource.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/slot_table.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ps::runtime::gpu_mcu {

inline constexpr uint32_t kGpuMcuMaxVerifyCandidates = 32u;

struct GpuMcuBatchPlanLimits {
    uint32_t max_requests = 0u;
    uint32_t max_rows = 0u;
    uint32_t max_prefill_rows_per_slot = 0u;
};

struct GpuMcuBatchPlanView {
    DeviceBatchContext* context = nullptr;
    DeviceRequestDescriptor* requests = nullptr;
    uint32_t* row_sequence_slots = nullptr;
    uint32_t* row_positions = nullptr;
    uint64_t* batch_plan_epoch = nullptr;
    DeviceSequenceResourceManagerView resources{};
    uint32_t resolve_sequence_slots = 0u;
};

struct GpuMcuBatchPlanTelemetry {
    uint32_t planned_requests = 0u;
    uint32_t planned_rows = 0u;
    uint32_t stale_handles = 0u;
    uint32_t skipped_not_plannable = 0u;
    uint32_t skipped_no_capacity = 0u;
    uint32_t rejected_invalid = 0u;
    uint32_t empty_prefill = 0u;
    uint32_t unresolved_resources = 0u;
};

__host__ __device__ __forceinline__ bool gpu_mcu_slot_is_plannable(
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

__host__ __device__ __forceinline__ ExecutionClass gpu_mcu_plan_execution_class(
    uint32_t phase) noexcept {
    if (phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::decode)) {
        return ExecutionClass::DECODE;
    }
    if (phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify)) {
        return ExecutionClass::SPEC_VERIFY;
    }
    return ExecutionClass::PREFILL;
}

__device__ __forceinline__ bool gpu_mcu_plan_resolve_row_count(
    const GpuMcuSlotState& slot,
    uint32_t phase,
    uint32_t remaining_rows,
    const GpuMcuBatchPlanLimits& limits,
    uint32_t& row_count,
    uint32_t& skipped_telemetry) noexcept {
    if (phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::decode)) {
        if (remaining_rows < 1u) {
            skipped_telemetry = 1u;
            return false;
        }
        row_count = 1u;
        return true;
    }
    if (phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify)) {
        const uint32_t candidates = slot.verify_candidate_count;
        if (candidates == 0u || candidates > kGpuMcuMaxVerifyCandidates) {
            if (candidates > kGpuMcuMaxVerifyCandidates) skipped_telemetry = 2u;
            else skipped_telemetry = 1u;
            return false;
        }
        if (remaining_rows < candidates) {
            skipped_telemetry = 1u;
            return false;
        }
        row_count = candidates;
        return true;
    }
    const uint32_t remaining =
        slot.prompt_length > slot.prefill_position
            ? slot.prompt_length - slot.prefill_position
            : 0u;
    if (remaining == 0u) {
        skipped_telemetry = 3u;
        return false;
    }
    uint32_t rows = remaining < remaining_rows ? remaining : remaining_rows;
    if (limits.max_prefill_rows_per_slot != 0u &&
        rows > limits.max_prefill_rows_per_slot) {
        rows = limits.max_prefill_rows_per_slot;
    }
    if (rows == 0u) {
        skipped_telemetry = 1u;
        return false;
    }
    row_count = rows;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_plan_append(
    uint32_t phase,
    const RequestHandle* runnable_handles,
    uint32_t runnable_count,
    const GpuMcuSlotState* slots,
    uint32_t max_slots,
    const GpuMcuBatchPlanLimits& limits,
    const GpuMcuBatchPlanView& view,
    uint32_t& request_cursor,
    uint32_t& row_cursor,
    GpuMcuBatchPlanTelemetry& telemetry) noexcept {
    for (uint32_t i = 0; i < runnable_count; ++i) {
        if (request_cursor >= limits.max_requests) {
            telemetry.skipped_no_capacity += 1u;
            return false;
        }
        const RequestHandle handle = runnable_handles[i];
        if (handle.slot >= max_slots) {
            telemetry.rejected_invalid += 1u;
            continue;
        }
        const GpuMcuSlotState& slot = slots[handle.slot];
        if (slot.phase != phase) continue;
        if (!gpu_mcu_slot_matches_handle(slot, handle)) {
            telemetry.stale_handles += 1u;
            continue;
        }
        if (!gpu_mcu_slot_is_plannable(slot)) {
            telemetry.skipped_not_plannable += 1u;
            continue;
        }

        uint32_t sequence_slot = handle.slot;
        if (view.resolve_sequence_slots != 0u) {
            if (!gpu_mcu_resource_resolve(view.resources,
                                          slot.kv_sequence_handle,
                                          &sequence_slot, nullptr)) {
                telemetry.unresolved_resources += 1u;
                continue;
            }
        }

        const uint32_t remaining_rows = limits.max_rows - row_cursor;
        uint32_t row_count = 0u;
        uint32_t skipped = 0u;
        if (!gpu_mcu_plan_resolve_row_count(
                slot, phase, remaining_rows, limits, row_count, skipped)) {
            if (skipped == 1u) telemetry.skipped_no_capacity += 1u;
            else if (skipped == 2u) telemetry.rejected_invalid += 1u;
            else telemetry.empty_prefill += 1u;
            continue;
        }

        const uint32_t prefix_length =
            phase == gpu_mcu_slot_phase_value(GpuMcuSlotPhase::prefill)
                ? slot.prefill_position
                : slot.committed_position;
        if (prefix_length > slot.sequence_length &&
            phase != gpu_mcu_slot_phase_value(GpuMcuSlotPhase::prefill)) {
            telemetry.rejected_invalid += 1u;
            continue;
        }
        const uint32_t sequence_length = prefix_length + row_count;
        if (sequence_length < prefix_length ||
            sequence_length > slot.max_sequence_length) {
            telemetry.rejected_invalid += 1u;
            continue;
        }

        DeviceRequestDescriptor& descriptor = view.requests[request_cursor];
        descriptor = DeviceRequestDescriptor{};
        descriptor.request_handle = handle;
        descriptor.row_begin = row_cursor;
        descriptor.row_count = row_count;
        descriptor.prefix_length = prefix_length;
        descriptor.sequence_length = sequence_length;
        descriptor.output_index = 0xFFFFFFFFu;
        descriptor.execution_class = gpu_mcu_plan_execution_class(phase);
        descriptor.compute_logits = 0u;
        descriptor.output_count = 0u;
        descriptor.sampling = DeviceSamplingParams{};

        for (uint32_t t = 0; t < row_count; ++t) {
            const uint32_t row = row_cursor + t;
            view.row_sequence_slots[row] = sequence_slot;
            view.row_positions[row] = prefix_length + t;
        }

        row_cursor += row_count;
        request_cursor += 1u;
        telemetry.planned_requests += 1u;
        telemetry.planned_rows += row_count;
    }
    return true;
}

__device__ __forceinline__ bool gpu_mcu_build_batch_geometry(
    const RequestHandle* runnable_handles,
    uint32_t runnable_count,
    const GpuMcuSlotState* slots,
    uint32_t max_slots,
    const GpuMcuBatchPlanLimits& limits,
    const GpuMcuBatchPlanView& view,
    GpuMcuBatchPlanTelemetry* telemetry) noexcept {
    if (view.context == nullptr || view.requests == nullptr ||
        view.row_sequence_slots == nullptr || view.row_positions == nullptr) {
        return false;
    }
    GpuMcuBatchPlanTelemetry local{};
    uint32_t request_cursor = 0u;
    uint32_t row_cursor = 0u;
    uint32_t decode_count = 0u;
    uint32_t verify_count = 0u;
    uint32_t prefill_count = 0u;

    if (runnable_handles != nullptr && slots != nullptr && runnable_count != 0u) {
        const uint32_t before = request_cursor;
        (void)gpu_mcu_plan_append(
            gpu_mcu_slot_phase_value(GpuMcuSlotPhase::decode), runnable_handles,
            runnable_count, slots, max_slots, limits, view, request_cursor,
            row_cursor, local);
        decode_count = request_cursor - before;
        const uint32_t decode_end = request_cursor;
        (void)gpu_mcu_plan_append(
            gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify), runnable_handles,
            runnable_count, slots, max_slots, limits, view, request_cursor,
            row_cursor, local);
        verify_count = request_cursor - decode_end;
        const uint32_t verify_end = request_cursor;
        (void)gpu_mcu_plan_append(
            gpu_mcu_slot_phase_value(GpuMcuSlotPhase::prefill), runnable_handles,
            runnable_count, slots, max_slots, limits, view, request_cursor,
            row_cursor, local);
        prefill_count = request_cursor - verify_end;
    }

    view.context->actual_rows = row_cursor;
    view.context->num_requests = request_cursor;
    view.context->num_decode_requests = decode_count;
    view.context->num_verify_requests = verify_count;
    view.context->num_prefill_requests = prefill_count;
    view.context->num_outputs = 0u;

    __threadfence_system();
    if (view.batch_plan_epoch != nullptr) {
        const uint64_t epoch = *view.batch_plan_epoch + 1u;
        __scoped_atomic_store_n(view.batch_plan_epoch, epoch, __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
    }
    if (telemetry != nullptr) *telemetry = local;
    return true;
}

Status gpu_mcu_build_batch_geometry_launch(
    const RequestHandle* runnable_handles,
    uint32_t runnable_count,
    const GpuMcuSlotState* slots,
    uint32_t max_slots,
    const GpuMcuBatchPlanLimits& limits,
    const GpuMcuBatchPlanView& view,
    GpuMcuBatchPlanTelemetry* telemetry,
    hipStream_t stream);

const char* gpu_mcu_batch_plan_phase_name(uint32_t phase) noexcept;

}  // namespace ps::runtime::gpu_mcu
