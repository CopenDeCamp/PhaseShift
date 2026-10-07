#pragma once

#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/execution/micro_fsm.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

__host__ __device__ __forceinline__ uint32_t gpu_mcu_patch_source_value(
    uint8_t source,
    uint32_t param,
    const DeviceBatchContext* context) noexcept {
    if (context == nullptr) return 0u;
    switch (static_cast<McuInvocationPatchSource>(source)) {
        case McuInvocationPatchSource::ActualRows:
            return context->actual_rows;
        case McuInvocationPatchSource::NumRequests:
            return context->num_requests;
        case McuInvocationPatchSource::NumOutputs:
            return context->num_outputs;
        case McuInvocationPatchSource::ActualRowsTimesParam:
            return context->actual_rows * param;
        case McuInvocationPatchSource::NumOutputsTimesParam:
            return context->num_outputs * param;
        case McuInvocationPatchSource::VerifyRequests:
            return context->num_verify_requests != 0u ? 1u : 0u;
        case McuInvocationPatchSource::Bf16ExactRowsVariant: {
            const uint32_t rows = context->num_outputs != 0u
                                      ? context->num_outputs
                                      : context->actual_rows;
            if (rows == 0u || rows > 16u) return 0xFFFFFFFFu;
            return param + (rows - 1u);
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
    uint32_t patch_count) noexcept {
    if (context == nullptr || patches == nullptr) return;
    for (uint32_t i = 0; i < patch_count; ++i) {
        const McuInvocationPatch& patch = patches[i];
        if (patch.target == 0u) continue;
        *reinterpret_cast<uint32_t*>(patch.target) =
            gpu_mcu_patch_source_value(patch.source, patch.param, context);
    }
#ifdef __HIP_DEVICE_COMPILE__
    __threadfence_system();
#endif
}

}  // namespace ps::runtime::gpu_mcu
