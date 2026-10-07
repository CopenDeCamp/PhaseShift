#include <phaseshift/runtime/gpu_mcu/scheduling/batch_planner.h>

namespace ps::runtime::gpu_mcu {
namespace {

__global__ void gpu_mcu_build_batch_geometry_kernel(
    const RequestHandle* runnable_handles,
    uint32_t runnable_count,
    const GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuBatchPlanLimits limits,
    GpuMcuBatchPlanView view,
    GpuMcuBatchPlanTelemetry* telemetry) {
    (void)gpu_mcu_build_batch_geometry(runnable_handles, runnable_count, slots,
                                       max_slots, limits, view, telemetry);
}

}  // namespace

Status gpu_mcu_build_batch_geometry_launch(
    const RequestHandle* runnable_handles,
    uint32_t runnable_count,
    const GpuMcuSlotState* slots,
    uint32_t max_slots,
    const GpuMcuBatchPlanLimits& limits,
    const GpuMcuBatchPlanView& view,
    GpuMcuBatchPlanTelemetry* telemetry,
    hipStream_t stream) {
    if (slots == nullptr) {
        return Status::invalid_argument("slot table is null", __FILE__,
                                        __LINE__);
    }
    if (view.context == nullptr || view.requests == nullptr ||
        view.row_sequence_slots == nullptr || view.row_positions == nullptr) {
        return Status::invalid_argument("batch plan view is incomplete",
                                        __FILE__, __LINE__);
    }
    if (limits.max_requests == 0u || limits.max_rows == 0u) {
        return Status::invalid_argument("batch plan limits are zero", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_build_batch_geometry_kernel<<<1, 1, 0u, stream>>>(
        runnable_handles, runnable_count, slots, max_slots, limits, view,
        telemetry);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("build batch geometry", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

const char* gpu_mcu_batch_plan_phase_name(uint32_t phase) noexcept {
    return gpu_mcu_slot_phase_name(phase);
}

}  // namespace ps::runtime::gpu_mcu
