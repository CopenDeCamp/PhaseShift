#include <phaseshift/runtime/gpu_mcu/gdn_reset.h>

#include <hip/hip_runtime.h>

namespace ps {
namespace runtime {
namespace gpu_mcu {

extern "C" __global__ void phaseshift_gpu_mcu_gdn_reset(
    GpuMcuGdnResetInvocation invocation) {
    if (invocation.context == nullptr || invocation.conv_base == nullptr ||
        invocation.recurrent_base == nullptr) {
        return;
    }
    const DeviceRequestDescriptor* requests = invocation.context->requests;
    if (requests == nullptr) return;
    const uint32_t stride = gridDim.x * blockDim.x;
    if (stride == 0u) return;
    // A request that starts at position zero opens a new sequence, and the
    // scheduler only hands a sequence to a batch once, so the request cannot
    // repeat and the dispatch stays idempotent when it runs every loop.
    const uint32_t count = invocation.context->num_requests;
    for (uint32_t r = 0u; r < count; ++r) {
        const DeviceRequestDescriptor& request = requests[r];
        if (request.prefix_length != 0u) continue;
        const uint32_t sequence_slot = request.request_handle.slot;
        uint16_t* conv = invocation.conv_base +
                         static_cast<std::size_t>(sequence_slot) *
                             invocation.conv_slot_stride;
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < invocation.conv_slot_count; i += stride) {
            conv[i] = 0u;
        }
        uint32_t* recurrent =
            invocation.recurrent_base +
            static_cast<std::size_t>(sequence_slot) *
                invocation.recurrent_slot_stride;
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < invocation.recurrent_slot_count; i += stride) {
            recurrent[i] = 0u;
        }
    }
}

}  // namespace gpu_mcu
}  // namespace runtime
}  // namespace ps
