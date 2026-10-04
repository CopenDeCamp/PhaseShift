#include <phaseshift/runtime/gpu_mcu/batch_binding.h>

namespace ps::runtime::gpu_mcu {
namespace {

__global__ void gpu_mcu_bind_batch_io_kernel(
    GpuMcuBatchBindingView view,
    GpuMcuBatchBindingTelemetry* telemetry) {
    (void)gpu_mcu_bind_batch_io(view, telemetry);
}

}  // namespace

Status gpu_mcu_bind_batch_io_launch(const GpuMcuBatchBindingView& view,
                                    GpuMcuBatchBindingTelemetry* telemetry,
                                    hipStream_t stream) {
    if (view.context == nullptr || view.requests == nullptr ||
        view.token_ids == nullptr || view.slots == nullptr ||
        view.bindings == nullptr) {
        return Status::invalid_argument("batch binding view is incomplete",
                                        __FILE__, __LINE__);
    }
    if (view.max_slots == 0u || view.max_outputs == 0u) {
        return Status::invalid_argument("batch binding capacity is zero",
                                        __FILE__, __LINE__);
    }
    gpu_mcu_bind_batch_io_kernel<<<1, 1, 0u, stream>>>(view, telemetry);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("bind batch io", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace ps::runtime::gpu_mcu
