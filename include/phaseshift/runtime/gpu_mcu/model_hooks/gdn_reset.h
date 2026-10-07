#pragma once

// Transitional model-specific hook.
//
// This dependency is intentionally quarantined from the generic GPU-MCU
// substrate. Do not add new model-specific operations here.
// A later architecture refactor must move this responsibility into the
// model backend or replace it with a generic invocation mechanism.

#include <phaseshift/runtime/batch/device_batch_context.h>

#include <cstdint>

namespace ps {
namespace runtime {
namespace gpu_mcu {

inline constexpr const char* kGdnResetSymbol =
    "phaseshift_gpu_mcu_gdn_reset";

// Kernarg of the gdn state reset. The kernel resets the conv and the recurrent
// state of every sequence that starts in the current batch, so a request never
// inherits the state of the request that used the slot before it.
//
// The reset has to run on the device. The state is megabytes per layer, so the
// persistent loop cannot zero it with its own threads, and once the MCU owns the
// request lifecycle the host must not issue a per request memset.
//
// The whole invocation is static per plan except the context, which the device
// scheduler rewrites every loop. The kernel reads the request descriptors the
// same way the gdn conv and recurrence kernels do, and a request whose prefix
// length is zero opens a new sequence. The scheduler hands a sequence to a batch
// once, so the dispatch stays correct when it runs every loop: there is nothing
// to reset unless a sequence starts.
struct GpuMcuGdnResetInvocation {
    uint16_t* conv_base = nullptr;
    uint32_t* recurrent_base = nullptr;
    const DeviceBatchContext* context = nullptr;
    uint64_t conv_slot_stride = 0u;
    uint64_t recurrent_slot_stride = 0u;
    uint32_t conv_slot_count = 0u;
    uint32_t recurrent_slot_count = 0u;
};

static_assert(sizeof(GpuMcuGdnResetInvocation) == 48u);
static_assert(alignof(GpuMcuGdnResetInvocation) == 8u);

}  // namespace gpu_mcu
}  // namespace runtime
}  // namespace ps
