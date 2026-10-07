#pragma once

#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/batch_planner.h>
#include <phaseshift/runtime/gpu_mcu/binding/slot_binding.h>
#include <phaseshift/runtime/gpu_mcu/commit/slot_runtime.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/slot_table.h>
#include <phaseshift/runtime/gpu_mcu/commit/stop_conditions.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct GpuMcuBatchCommitTelemetry {
    uint32_t committed_requests = 0u;
    uint32_t committed_tokens = 0u;
    uint32_t stale_handles = 0u;
    uint32_t invalid_geometry = 0u;
    uint32_t verify_overflow = 0u;
    uint32_t pending_overflow = 0u;
    uint32_t stopped_tokens = 0u;
    uint32_t stopped_sequence = 0u;
    uint32_t verify_resource_failed = 0u;
};

struct GpuMcuBatchCommitView {
    DeviceBatchContext* context = nullptr;
    GpuMcuSlotState* slots = nullptr;
    uint32_t max_slots = 0u;
    GpuMcuSlotBinding* bindings = nullptr;
    uint32_t binding_max_slots = 0u;
    uint32_t request_limits_valid = 0u;
    const int32_t* sampled_tokens = nullptr;
    uint32_t sampled_capacity = 0u;
    const uint32_t* verify_committed_counts = nullptr;
    uint32_t verify_capacity = 0u;
    GpuMcuSlotRuntimeState* slot_runtime = nullptr;
    uint32_t resources_enabled = 0u;
    DeviceSequenceResourceManagerView resources{};
    GpuMcuKvPagePoolView kv_pages{};
    GpuMcuBlockTableView kv_blocks{};
    GpuMcuBatchCommitTelemetry* telemetry = nullptr;
};

// Finalizes the VERIFY resource transaction at the accepted boundary. The pages
// the sequence owned before the transaction are never released, so keep is the
// larger of the base and what the accepted prefix needs. Returns false and
// mutates nothing when the transaction is not active or the slot is not in the
// verify phase.
__device__ __forceinline__ bool gpu_mcu_verify_resource_commit(
    GpuMcuSlotState& slot,
    GpuMcuSlotRuntimeState& meta,
    const DeviceSequenceResourceManagerView& resources,
    const GpuMcuKvPagePoolView& kv_pages,
    const GpuMcuBlockTableView& kv_blocks,
    uint32_t accepted_count) noexcept {
    if (meta.verify_txn_active == 0u) return false;
    if (slot.phase != gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify)) {
        return false;
    }
    if (accepted_count > slot.verify_candidate_count) return false;
    if (accepted_count > kGpuMcuMaxVerifyCandidates) return false;
    if (slot.kv_sequence_handle == 0u) return false;
    uint32_t sequence_slot = 0u;
    if (!gpu_mcu_resource_resolve(resources, slot.kv_sequence_handle,
                                  &sequence_slot, nullptr)) {
        return false;
    }
    const uint32_t rows_per_page = gpu_mcu_block_rows_per_page(kv_blocks);
    const uint32_t required =
        (slot.committed_position + accepted_count + rows_per_page - 1u) /
        rows_per_page;
    const uint32_t keep =
        required > meta.verify_base_page_count ? required
                                               : meta.verify_base_page_count;
    const uint32_t current = gpu_mcu_resource_page_count(
        resources, slot.kv_sequence_handle);
    if (current > keep) {
        uint32_t freed = 0u;
        if (!gpu_mcu_resource_shrink_kv_pages(resources, kv_pages, kv_blocks,
                                              slot.kv_sequence_handle,
                                              sequence_slot, keep, &freed)) {
            return false;
        }
    }
    meta.verify_txn_active = 0u;
    meta.verify_base_page_count = 0u;
    meta.verify_reserved_page_count = 0u;
    meta.verify_txn_epoch = 0u;
    return true;
}

// Releases the whole speculative window back to the base page count. Used when
// the verify result is invalid or the transaction cannot be committed.
__device__ __forceinline__ bool gpu_mcu_verify_resource_abort(
    GpuMcuSlotState& slot,
    GpuMcuSlotRuntimeState& meta,
    const DeviceSequenceResourceManagerView& resources,
    const GpuMcuKvPagePoolView& kv_pages,
    const GpuMcuBlockTableView& kv_blocks) noexcept {
    if (meta.verify_txn_active == 0u) return false;
    if (slot.kv_sequence_handle == 0u) return false;
    uint32_t sequence_slot = 0u;
    if (!gpu_mcu_resource_resolve(resources, slot.kv_sequence_handle,
                                  &sequence_slot, nullptr)) {
        return false;
    }
    const uint32_t current =
        gpu_mcu_resource_page_count(resources, slot.kv_sequence_handle);
    if (current > meta.verify_base_page_count) {
        uint32_t freed = 0u;
        if (!gpu_mcu_resource_shrink_kv_pages(
                resources, kv_pages, kv_blocks, slot.kv_sequence_handle,
                sequence_slot, meta.verify_base_page_count, &freed)) {
            return false;
        }
    }
    meta.verify_txn_active = 0u;
    meta.verify_base_page_count = 0u;
    meta.verify_reserved_page_count = 0u;
    meta.verify_txn_epoch = 0u;
    return true;
}

__device__ __forceinline__ int32_t gpu_mcu_commit_read_token(
    const GpuMcuBatchCommitView& view,
    const DeviceRequestDescriptor& descriptor,
    uint32_t output_slot) noexcept {
    if (view.context->output_rows == nullptr) return -1;
    if (descriptor.output_index == 0xFFFFFFFFu) return -1;
    const uint32_t row =
        view.context->output_rows[descriptor.output_index + output_slot];
    if (row >= view.sampled_capacity) return -1;
    return view.sampled_tokens[row];
}

__device__ __forceinline__ const GpuMcuStopConditions* gpu_mcu_stop_conditions_for(
    const GpuMcuBatchCommitView& view, RequestHandle handle) noexcept {
    if (view.bindings == nullptr || handle.slot >= view.binding_max_slots) {
        return nullptr;
    }
    const uint64_t h = view.bindings[handle.slot].stop_conditions_handle;
    if (h == 0u) return nullptr;
    return reinterpret_cast<const GpuMcuStopConditions*>(h);
}

__device__ __forceinline__ int32_t gpu_mcu_scan_first_stop(
    const GpuMcuBatchCommitView& view,
    const DeviceRequestDescriptor& descriptor,
    const GpuMcuStopConditions* stop,
    uint32_t count) noexcept {
    if (stop == nullptr || stop->token_ids == nullptr || stop->token_count == 0u) {
        return -1;
    }
    for (uint32_t j = 0u; j < count; ++j) {
        const int32_t token = gpu_mcu_commit_read_token(view, descriptor, j);
        for (uint32_t s = 0u; s < stop->token_count; ++s) {
            if (token == stop->token_ids[s]) {
                return static_cast<int32_t>(j);
            }
        }
    }
    return -1;
}

__device__ __forceinline__ void gpu_mcu_commit_evaluate_stop(
    GpuMcuSlotState& slot, GpuMcuBatchCommitTelemetry& telemetry,
    uint32_t request_limits_valid) noexcept {
    if (request_limits_valid == 0u) return;
    if (slot.terminal_reason !=
        static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
        return;
    }
    if (slot.generated_tokens >= slot.max_new_tokens) {
        slot.terminal_reason =
            static_cast<uint32_t>(GpuMcuTerminalReason::max_new_tokens);
        telemetry.stopped_tokens += 1u;
        return;
    }
    if (slot.max_sequence_length != 0u &&
        slot.sequence_length >= slot.max_sequence_length) {
        slot.terminal_reason =
            static_cast<uint32_t>(GpuMcuTerminalReason::max_seq_len);
        telemetry.stopped_sequence += 1u;
    }
}

__device__ __forceinline__ bool gpu_mcu_commit_active_batch(
    const GpuMcuBatchCommitView& view) noexcept {
    if (view.context == nullptr || view.context->requests == nullptr ||
        view.slots == nullptr || view.sampled_tokens == nullptr) {
        return false;
    }
    GpuMcuBatchCommitTelemetry local{};
    bool ok = true;
    for (uint32_t i = 0; i < view.context->num_requests; ++i) {
        const DeviceRequestDescriptor& descriptor = view.context->requests[i];
        const RequestHandle handle = descriptor.request_handle;
        if (handle.slot >= view.max_slots) {
            local.invalid_geometry += 1u;
            ok = false;
            continue;
        }
        GpuMcuSlotState& slot = view.slots[handle.slot];
        if (!gpu_mcu_slot_matches_handle(slot, handle)) {
            local.stale_handles += 1u;
            ok = false;
            continue;
        }
        uint32_t limit_reason =
            static_cast<uint32_t>(GpuMcuTerminalReason::none);

        uint32_t tokens = 0u;
        if (descriptor.execution_class == ExecutionClass::PREFILL) {
            slot.prefill_position = descriptor.sequence_length;
            slot.sequence_length = descriptor.sequence_length;
            slot.committed_position = descriptor.sequence_length;
            if (descriptor.sequence_length < slot.prompt_length) {
                local.committed_requests += 1u;
                continue;
            }
            if (view.request_limits_valid != 0u && slot.max_new_tokens == 0u) {
                if (slot.terminal_reason ==
                    static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
                    slot.terminal_reason = static_cast<uint32_t>(
                        GpuMcuTerminalReason::max_new_tokens);
                    local.stopped_tokens += 1u;
                }
                local.committed_requests += 1u;
                continue;
            }
            slot.phase = gpu_mcu_slot_phase_value(GpuMcuSlotPhase::decode);
            tokens = descriptor.output_count != 0u ? 1u : 0u;
        } else if (descriptor.execution_class == ExecutionClass::DECODE) {
            if (descriptor.row_count != 1u) {
                local.invalid_geometry += 1u;
                ok = false;
                continue;
            }
            slot.sequence_length = descriptor.sequence_length;
            slot.committed_position = descriptor.sequence_length;
            tokens = 1u;
        } else if (descriptor.execution_class == ExecutionClass::SPEC_VERIFY) {
            uint32_t committed = 0u;
            if (view.verify_committed_counts != nullptr &&
                i < view.verify_capacity) {
                committed = view.verify_committed_counts[i];
            }
            const bool invalid = committed > descriptor.row_count ||
                                 committed > kGpuMcuMaxVerifyCandidates;
            GpuMcuSlotRuntimeState* meta =
                (view.slot_runtime != nullptr && handle.slot < view.max_slots)
                    ? &view.slot_runtime[handle.slot]
                    : nullptr;
            const bool resources_on =
                view.resources_enabled != 0u && meta != nullptr;

            uint32_t effective = committed;
            int32_t first_stop = -1;
            bool budget_limited = false;
            bool capacity_limited = false;
            if (!invalid) {
                const GpuMcuStopConditions* stop =
                    gpu_mcu_stop_conditions_for(view, handle);
                first_stop =
                    gpu_mcu_scan_first_stop(view, descriptor, stop, committed);
                if (first_stop >= 0 &&
                    static_cast<uint32_t>(first_stop) + 1u < effective) {
                    effective = static_cast<uint32_t>(first_stop) + 1u;
                }
                if (view.request_limits_valid != 0u) {
                    const uint32_t budget =
                        slot.generated_tokens >= slot.max_new_tokens
                            ? 0u
                            : slot.max_new_tokens - slot.generated_tokens;
                    if (budget < effective) {
                        effective = budget;
                        budget_limited = true;
                    }
                    const uint64_t sequence_limit = slot.max_sequence_length;
                    uint64_t capacity = ~uint64_t{0};
                    if (sequence_limit != 0u) {
                        capacity =
                            descriptor.prefix_length >= sequence_limit
                                ? 0u
                                : sequence_limit - descriptor.prefix_length;
                    }
                    if (capacity < effective) {
                        effective = static_cast<uint32_t>(capacity);
                        capacity_limited = true;
                    }
                }
                if (first_stop >= 0 &&
                    static_cast<uint32_t>(first_stop) < effective) {
                    limit_reason =
                        static_cast<uint32_t>(GpuMcuTerminalReason::eos);
                } else if (budget_limited) {
                    limit_reason = static_cast<uint32_t>(
                        GpuMcuTerminalReason::max_new_tokens);
                } else if (capacity_limited) {
                    limit_reason = static_cast<uint32_t>(
                        GpuMcuTerminalReason::max_seq_len);
                }
            }

            bool resource_ok = true;
            if (resources_on && meta->verify_txn_active != 0u) {
                if (invalid) {
                    (void)gpu_mcu_verify_resource_abort(
                        slot, *meta, view.resources, view.kv_pages,
                        view.kv_blocks);
                    resource_ok = false;
                    local.verify_resource_failed += 1u;
                } else if (!gpu_mcu_verify_resource_commit(
                               slot, *meta, view.resources, view.kv_pages,
                               view.kv_blocks, effective)) {
                    (void)gpu_mcu_verify_resource_abort(
                        slot, *meta, view.resources, view.kv_pages,
                        view.kv_blocks);
                    resource_ok = false;
                    local.verify_resource_failed += 1u;
                }
            }
            if (invalid || !resource_ok) {
                if (invalid) local.verify_overflow += 1u;
                (void)gpu_mcu_slot_mark_terminal(
                    slot, handle,
                    static_cast<uint32_t>(GpuMcuTerminalReason::error));
                ok = false;
                continue;
            }
            slot.sequence_length = descriptor.prefix_length + effective;
            slot.committed_position = slot.sequence_length;
            slot.verify_committed_count = effective;
            slot.phase = gpu_mcu_slot_phase_value(GpuMcuSlotPhase::decode);
            tokens = effective;
        } else {
            local.invalid_geometry += 1u;
            ok = false;
            continue;
        }

        uint32_t staged = 0u;
        if (tokens != 0u) {
            int32_t last = -1;
            for (uint32_t j = 0; j < tokens; ++j) {
                const int32_t token =
                    gpu_mcu_commit_read_token(view, descriptor, j);
                if (!gpu_mcu_slot_stage_token(slot, token)) {
                    local.pending_overflow += 1u;
                    slot.output_blocked = 1u;
                    ok = false;
                    break;
                }
                last = token;
                staged += 1u;
            }
            if (staged != 0u) {
                if (view.bindings != nullptr &&
                    handle.slot < view.binding_max_slots) {
                    GpuMcuSlotBinding& binding = view.bindings[handle.slot];
                    if (gpu_mcu_binding_valid_for(slot, binding)) {
                        binding.decode_input_token = last;
                        binding.decode_input_valid = 1u;
                    }
                }
                slot.generated_tokens += staged;
                local.committed_tokens += staged;
                const GpuMcuStopConditions* stop =
                    gpu_mcu_stop_conditions_for(view, handle);
                if (stop != nullptr) {
                    const int32_t first_stop =
                        gpu_mcu_scan_first_stop(view, descriptor, stop, staged);
                    if (first_stop >= 0) {
                        (void)gpu_mcu_slot_mark_terminal(
                            slot, handle,
                            static_cast<uint32_t>(GpuMcuTerminalReason::eos));
                    }
                }
            }
        }
        if (limit_reason !=
            static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
            (void)gpu_mcu_slot_mark_terminal(slot, handle, limit_reason);
        }
        if (staged != 0u) {
            gpu_mcu_commit_evaluate_stop(slot, local,
                                         view.request_limits_valid);
        }
        local.committed_requests += 1u;
    }
    if (view.telemetry != nullptr) *view.telemetry = local;
    return ok;
}

}  // namespace ps::runtime::gpu_mcu
