#pragma once

#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/binding/plan_binding_contract.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/stage_trace.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

__host__ __device__ __forceinline__ void gpu_mcu_bind_attention_paths(
    const DeviceBatchContext* context,
    const McuAttentionPathSpan* spans,
    uint32_t span_count,
    McuRuntimeNodeBinding* bindings,
    uint32_t binding_count) noexcept {
    if (context == nullptr || spans == nullptr || bindings == nullptr) return;
    uint32_t max_visible = 0u;
    uint32_t first_row_begin = 0u;
    uint32_t first_row_count = 0u;
    if (context->requests != nullptr && context->num_requests != 0u) {
        for (uint32_t i = 0u; i < context->num_requests; ++i) {
            const uint32_t visible = context->requests[i].sequence_length;
            if (visible > max_visible) max_visible = visible;
        }
        first_row_begin = context->requests[0].row_begin;
        first_row_count = context->requests[0].row_count;
    }
    const uint32_t rows = context->actual_rows;
    for (uint32_t s = 0u; s < span_count; ++s) {
        const McuAttentionPathSpan& span = spans[s];
        const bool has_prefill = span.prefill_node_end > span.node_begin;
        const bool has_direct = span.direct_node_end > span.prefill_node_end;
        const bool has_split = span.split_node_end > span.direct_node_end;
        const bool prefill_batch =
            has_prefill && context->num_requests == 1u &&
            rows >= span.prefill_min_rows && first_row_begin == 0u &&
            first_row_count >= rows;
        const bool split_batch =
            has_split && max_visible >= span.split_min_visible;
        bool chosen = false;
        uint32_t mode = 0u;
        if (prefill_batch) {
            mode = 0u;
            chosen = true;
        } else if (split_batch) {
            mode = 2u;
            chosen = true;
        } else if (has_direct) {
            mode = 1u;
            chosen = true;
        } else if (has_split) {
            mode = 2u;
            chosen = true;
        } else if (has_prefill) {
            mode = 0u;
            chosen = true;
        }
        if (!chosen) continue;
        const uint32_t group_begin[3] = {span.node_begin, span.prefill_node_end,
                                         span.direct_node_end};
        const uint32_t group_end[3] = {span.prefill_node_end, span.direct_node_end,
                                       span.split_node_end};
        const bool present[3] = {has_prefill, has_direct, has_split};
        for (uint32_t g = 0u; g < 3u; ++g) {
            if (!present[g]) continue;
            const uint16_t enabled = g == mode ? 1u : 0u;
            for (uint32_t n = group_begin[g];
                 n < group_end[g] && n < binding_count; ++n)
                bindings[n].enabled = enabled;
        }
    }
}

__host__ __device__ __forceinline__ uint64_t gpu_mcu_patch_source_value(
    uint8_t source,
    uint32_t param,
    const DeviceBatchContext* context) noexcept {
    if (context == nullptr) return 0ull;
    switch (static_cast<McuInvocationPatchSource>(source)) {
        case McuInvocationPatchSource::ActualRows:
            return context->actual_rows;
        case McuInvocationPatchSource::NumRequests:
            return context->num_requests;
        case McuInvocationPatchSource::NumOutputs:
            return context->num_outputs;
        case McuInvocationPatchSource::ActualRowsTimesParam:
            return static_cast<uint64_t>(context->actual_rows) * param;
        case McuInvocationPatchSource::NumOutputsTimesParam:
            return static_cast<uint64_t>(context->num_outputs) * param;
        case McuInvocationPatchSource::VerifyRequests:
            return context->num_verify_requests != 0u ? 1u : 0u;
        case McuInvocationPatchSource::BatchTokenIds:
            return reinterpret_cast<uint64_t>(context->token_ids);
        case McuInvocationPatchSource::Bf16ExactRowsVariant: {
            const uint32_t rows = context->num_outputs != 0u
                                      ? context->num_outputs
                                      : context->actual_rows;
            if (rows == 0u || rows > 16u) return 0xFFFFFFFFu;
            return static_cast<uint64_t>(param) + (rows - 1u);
        }
        case McuInvocationPatchSource::AttentionRegionRows: {
            if (context->requests == nullptr) return 0u;
            const uint32_t region_begin = param & 0xFFFFu;
            const uint32_t region_end = region_begin + (param >> 16);
            uint32_t rows = 0u;
            for (uint32_t i = 0u; i < context->num_requests; ++i) {
                const DeviceRequestDescriptor& d = context->requests[i];
                const uint32_t begin = d.row_begin;
                const uint32_t end = begin + d.row_count;
                const uint32_t lo = begin > region_begin ? begin : region_begin;
                const uint32_t hi = end < region_end ? end : region_end;
                if (hi > lo) rows += hi - lo;
            }
            return rows;
        }
    }
    return 0u;
}

// Rewrites the per-loop geometry of a static skeleton. The static plan keeps
// the maximum geometry; this overlay writes the live batch's values into the
// row-global invocation fields so the same prepared plan serves every loop
// without recompiling or re-uploading it.
__host__ __device__ __forceinline__ void gpu_mcu_bind_execution_plan(
    const DeviceBatchContext* context,
    const McuInvocationPatch* patches,
    uint32_t patch_count,
    McuStageTrace* trace = nullptr) noexcept {
    if (context == nullptr || patches == nullptr) return;
    if (trace != nullptr) {
        trace->stage = kMcuStageBindPlanEnter;
        trace->seq += 1u;
    }
    for (uint32_t i = 0; i < patch_count; ++i) {
        const McuInvocationPatch& patch = patches[i];
        if (trace != nullptr) {
            trace->stage = kMcuStageBindPlanPatch;
            trace->patch_index = i;
            trace->patch_source = patch.source;
            trace->patch_width = patch.width;
            trace->patch_target_low =
                static_cast<uint32_t>(patch.target & 0xffffffffull);
            trace->patch_target_high =
                static_cast<uint32_t>(patch.target >> 32);
            trace->seq += 1u;
        }
        if (patch.target == 0u) continue;
        const uint64_t value =
            gpu_mcu_patch_source_value(patch.source, patch.param, context);
        if (patch.null_guard != 0u && value == 0ull) continue;
        if (patch.width == 8u) {
            *reinterpret_cast<uint64_t*>(patch.target) = value;
        } else {
            *reinterpret_cast<uint32_t*>(patch.target) =
                static_cast<uint32_t>(value);
        }
    }
    if (trace != nullptr) {
        trace->stage = kMcuStageBindPlanExit;
        trace->seq += 1u;
    }
#ifdef __HIP_DEVICE_COMPILE__
    __threadfence_system();
#endif
}

}  // namespace ps::runtime::gpu_mcu
