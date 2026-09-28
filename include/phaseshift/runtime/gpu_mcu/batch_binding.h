#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/batch_planner.h>
#include <phaseshift/runtime/gpu_mcu/slot_binding.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct GpuMcuBatchBindingView {
    DeviceBatchContext* context = nullptr;
    DeviceRequestDescriptor* requests = nullptr;
    int32_t* token_ids = nullptr;
    uint32_t* output_rows = nullptr;
    DeviceSamplingParams* output_sampling_params = nullptr;
    const GpuMcuSlotState* slots = nullptr;
    const GpuMcuSlotBinding* bindings = nullptr;
    uint32_t max_slots = 0u;
    uint32_t max_outputs = 0u;
    uint64_t* batch_ready_epoch = nullptr;
};

struct GpuMcuBatchBindingTelemetry {
    uint32_t bound_requests = 0u;
    uint32_t bound_rows = 0u;
    uint32_t bound_outputs = 0u;
    uint32_t stale_slot = 0u;
    uint32_t stale_binding = 0u;
    uint32_t missing_prompt = 0u;
    uint32_t missing_decode_token = 0u;
    uint32_t missing_verify_tokens = 0u;
    uint32_t missing_sampling = 0u;
    uint32_t unsupported_constraint = 0u;
    uint32_t output_capacity_failure = 0u;
    uint32_t invalid_geometry = 0u;
    uint32_t not_ready = 0u;
};

__device__ __forceinline__ bool gpu_mcu_bind_prefill_tokens(
    const DeviceRequestDescriptor& descriptor,
    const GpuMcuSlotState& slot,
    const GpuMcuSlotBinding& binding,
    const int32_t*& prompt,
    int32_t* token_ids) noexcept {
    prompt = gpu_mcu_resolve_device_address<int32_t>(
        binding.prompt_tokens_handle);
    if (prompt == nullptr) return false;
    if (descriptor.prefix_length + descriptor.row_count >
        slot.prompt_length) {
        return false;
    }
    for (uint32_t t = 0; t < descriptor.row_count; ++t) {
        token_ids[descriptor.row_begin + t] =
            prompt[descriptor.prefix_length + t];
    }
    return true;
}

__device__ __forceinline__ bool gpu_mcu_bind_batch_io(
    const GpuMcuBatchBindingView& view,
    GpuMcuBatchBindingTelemetry* telemetry) noexcept {
    if (view.context == nullptr || view.requests == nullptr ||
        view.token_ids == nullptr || view.slots == nullptr ||
        view.bindings == nullptr) {
        return false;
    }
    GpuMcuBatchBindingTelemetry local{};
    if (view.context->actual_rows == 0u) {
        view.context->num_outputs = 0u;
        if (telemetry != nullptr) *telemetry = local;
        return true;
    }
    if (view.output_rows == nullptr || view.output_sampling_params == nullptr) {
        return false;
    }

    view.context->constraint_masks = nullptr;
    view.context->constraint_mask_words = 0u;

    bool ready = true;
    uint32_t output_cursor = 0u;
    for (uint32_t i = 0; i < view.context->num_requests; ++i) {
        DeviceRequestDescriptor& descriptor = view.requests[i];
        const RequestHandle handle = descriptor.request_handle;
        if (handle.slot >= view.max_slots) {
            local.invalid_geometry += 1u;
            ready = false;
            continue;
        }
        const GpuMcuSlotState& slot = view.slots[handle.slot];
        if (!gpu_mcu_slot_matches_handle(slot, handle)) {
            local.stale_slot += 1u;
            ready = false;
            continue;
        }
        const GpuMcuSlotBinding& binding = view.bindings[handle.slot];
        if (!gpu_mcu_binding_valid_for(slot, binding)) {
            local.stale_binding += 1u;
            ready = false;
            continue;
        }
        if (slot.terminal_reason !=
            static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
            local.not_ready += 1u;
            ready = false;
            continue;
        }
        if (binding.constraint_state_handle != 0u) {
            local.unsupported_constraint += 1u;
            return false;
        }

        uint32_t output_count = 0u;
        if (descriptor.execution_class == ExecutionClass::PREFILL) {
            const int32_t* prompt = nullptr;
            if (!gpu_mcu_bind_prefill_tokens(descriptor, slot, binding, prompt,
                                             view.token_ids)) {
                local.missing_prompt += 1u;
                ready = false;
                continue;
            }
            const bool final_chunk =
                descriptor.sequence_length == slot.prompt_length;
            const bool wants_token =
                slot.max_new_tokens != 0u &&
                slot.generated_tokens < slot.max_new_tokens;
            output_count = (final_chunk && wants_token) ? 1u : 0u;
        } else if (descriptor.execution_class == ExecutionClass::DECODE) {
            if (descriptor.row_count != 1u) {
                local.invalid_geometry += 1u;
                ready = false;
                continue;
            }
            if (binding.decode_input_valid == 0u) {
                local.missing_decode_token += 1u;
                ready = false;
                continue;
            }
            if (slot.max_new_tokens != 0u &&
                slot.generated_tokens >= slot.max_new_tokens) {
                local.not_ready += 1u;
                ready = false;
                continue;
            }
            view.token_ids[descriptor.row_begin] = binding.decode_input_token;
            output_count = 1u;
        } else if (descriptor.execution_class == ExecutionClass::SPEC_VERIFY) {
            const int32_t* candidates = gpu_mcu_resolve_device_address<int32_t>(
                slot.verify_candidate_ref);
            if (candidates == nullptr) {
                local.missing_verify_tokens += 1u;
                ready = false;
                continue;
            }
            if (descriptor.row_count != slot.verify_candidate_count ||
                descriptor.row_count == 0u ||
                descriptor.row_count > kGpuMcuMaxVerifyCandidates) {
                local.invalid_geometry += 1u;
                ready = false;
                continue;
            }
            for (uint32_t t = 0; t < descriptor.row_count; ++t) {
                view.token_ids[descriptor.row_begin + t] = candidates[t];
            }
            output_count = descriptor.row_count;
        } else {
            local.invalid_geometry += 1u;
            ready = false;
            continue;
        }

        if (output_cursor + output_count > view.max_outputs) {
            local.output_capacity_failure += 1u;
            return false;
        }

        descriptor.compute_logits = output_count != 0u ? 1u : 0u;
        descriptor.output_count = static_cast<uint8_t>(output_count);
        if (output_count == 0u) {
            descriptor.output_index = 0xFFFFFFFFu;
        } else {
            const DeviceSamplingParams* sampling_template =
                gpu_mcu_resolve_device_address<DeviceSamplingParams>(
                    binding.sampling_params_handle);
            if (sampling_template == nullptr) {
                local.missing_sampling += 1u;
                ready = false;
                continue;
            }
            DeviceSamplingParams sampling = *sampling_template;
            sampling.sample_index = slot.generated_tokens;
            descriptor.sampling = sampling;
            descriptor.output_index = output_cursor;

            const uint32_t first =
                descriptor.row_begin + descriptor.row_count - output_count;
            for (uint32_t j = 0; j < output_count; ++j) {
                view.output_rows[output_cursor + j] = first + j;
                DeviceSamplingParams row_sampling = sampling;
                row_sampling.sample_index = sampling.sample_index + j;
                view.output_sampling_params[output_cursor + j] = row_sampling;
            }
            output_cursor += output_count;
            local.bound_outputs += output_count;
        }
        local.bound_requests += 1u;
        local.bound_rows += descriptor.row_count;
    }

    view.context->num_outputs = output_cursor;
    if (ready) {
        __threadfence_system();
    }
    if (ready && view.batch_ready_epoch != nullptr) {
        const uint64_t epoch = *view.batch_ready_epoch + 1u;
        __scoped_atomic_store_n(view.batch_ready_epoch, epoch, __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
    }
    if (telemetry != nullptr) *telemetry = local;
    return ready;
}

Status gpu_mcu_bind_batch_io_launch(const GpuMcuBatchBindingView& view,
                                    GpuMcuBatchBindingTelemetry* telemetry,
                                    hipStream_t stream);

}  // namespace ps::runtime::gpu_mcu
