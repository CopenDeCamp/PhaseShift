#include <phaseshift/runtime/gpu_mcu/scheduling/slot_table.h>

#include <cstddef>

namespace ps::runtime::gpu_mcu {
namespace {

__global__ void gpu_mcu_slot_init_kernel(GpuMcuSlotState* slots,
                                         uint32_t max_slots) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= max_slots) return;
    gpu_mcu_slot_init(slots[index], index);
}

__global__ void gpu_mcu_slot_claim_kernel(GpuMcuSlotState* slot,
                                          uint32_t index,
                                          uint64_t request_id,
                                          uint32_t prompt_length,
                                          uint32_t stable_prefix_boundary,
                                          uint32_t max_sequence_length,
                                          uint32_t max_new_tokens,
                                          uint32_t* applied) {
    const bool ok = gpu_mcu_slot_try_claim(
        *slot, index, request_id, prompt_length, stable_prefix_boundary,
        max_sequence_length, max_new_tokens);
    *applied = ok ? 1u : 0u;
}

__global__ void gpu_mcu_slot_cancel_kernel(GpuMcuSlotState* slot,
                                           RequestHandle handle,
                                           uint32_t* applied) {
    *applied = gpu_mcu_slot_request_cancel(*slot, handle) ? 1u : 0u;
}

__global__ void gpu_mcu_slot_output_blocked_kernel(GpuMcuSlotState* slot,
                                                   RequestHandle handle,
                                                   uint32_t blocked,
                                                   uint32_t* applied) {
    *applied = gpu_mcu_slot_mark_output_blocked(*slot, handle, blocked) ? 1u : 0u;
}

__global__ void gpu_mcu_slot_terminal_kernel(GpuMcuSlotState* slot,
                                             RequestHandle handle,
                                             uint32_t reason,
                                             uint32_t* applied) {
    *applied = gpu_mcu_slot_mark_terminal(*slot, handle, reason) ? 1u : 0u;
}

__global__ void gpu_mcu_slot_phase_kernel(GpuMcuSlotState* slot,
                                          RequestHandle handle,
                                          uint32_t phase,
                                          uint32_t* applied) {
    *applied = gpu_mcu_slot_set_phase(*slot, handle, phase) ? 1u : 0u;
}

__global__ void gpu_mcu_slot_release_kernel(GpuMcuSlotState* slot,
                                            RequestHandle handle,
                                            uint32_t* applied) {
    *applied = gpu_mcu_slot_release(*slot, handle) ? 1u : 0u;
}

}  // namespace

GpuMcuSlotTable::~GpuMcuSlotTable() noexcept {
    const Status status = shutdown();
    (void)status;
}

Result<GpuMcuSlotTable> GpuMcuSlotTable::create(int device, uint32_t max_slots) {
    if (max_slots == 0u || max_slots > kGpuMcuMaxSlots) {
        return Status::invalid_argument("slot count out of range", __FILE__,
                                        __LINE__);
    }
    GpuMcuSlotTable table{};
    table.max_slots_ = max_slots;
    table.device_ = device;

    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return Status::hip_error("hipSetDevice", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    void* ptr = nullptr;
    err = hipMalloc(&ptr, static_cast<std::size_t>(max_slots) *
                              sizeof(GpuMcuSlotState));
    if (err != hipSuccess) {
        return Status::hip_error("hipMalloc slot table",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    table.slots_ = static_cast<GpuMcuSlotState*>(ptr);

    hipStream_t stream = nullptr;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        (void)table.shutdown();
        return Status::hip_error("hipStreamCreate slot init",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    const uint32_t block = 128u;
    const uint32_t grid = (max_slots + block - 1u) / block;
    gpu_mcu_slot_init_kernel<<<grid, block, 0u, stream>>>(table.slots_,
                                                         max_slots);
    err = hipStreamSynchronize(stream);
    const hipError_t destroy_err = hipStreamDestroy(stream);
    if (err != hipSuccess) {
        (void)table.shutdown();
        return Status::hip_error("slot table init", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (destroy_err != hipSuccess) {
        (void)table.shutdown();
        return Status::hip_error("slot init stream destroy",
                                 hipGetErrorString(destroy_err), __FILE__,
                                 __LINE__);
    }
    return table;
}

Status GpuMcuSlotTable::shutdown() noexcept {
    if (slots_ == nullptr) {
        max_slots_ = 0u;
        return Status::make_ok();
    }
    hipError_t err = hipSetDevice(device_);
    if (err != hipSuccess) {
        return Status::hip_error("hipSetDevice shutdown",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    GpuMcuSlotState* slots = slots_;
    slots_ = nullptr;
    max_slots_ = 0u;
    err = hipFree(slots);
    if (err != hipSuccess) {
        return Status::hip_error("hipFree slot table", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

GpuMcuSlotTable::GpuMcuSlotTable(GpuMcuSlotTable&& other) noexcept
    : slots_(other.slots_),
      max_slots_(other.max_slots_),
      device_(other.device_) {
    other.slots_ = nullptr;
    other.max_slots_ = 0u;
}

GpuMcuSlotTable& GpuMcuSlotTable::operator=(GpuMcuSlotTable&& other) noexcept {
    if (this != &other) {
        const Status status = shutdown();
        (void)status;
        slots_ = other.slots_;
        max_slots_ = other.max_slots_;
        device_ = other.device_;
        other.slots_ = nullptr;
        other.max_slots_ = 0u;
    }
    return *this;
}

Status gpu_mcu_slot_table_claim(GpuMcuSlotTable& table,
                                uint32_t index,
                                uint64_t request_id,
                                uint32_t prompt_length,
                                uint32_t stable_prefix_boundary,
                                uint32_t max_sequence_length,
                                uint32_t max_new_tokens,
                                uint32_t* applied,
                                hipStream_t stream) {
    if (table.device_slots() == nullptr) {
        return Status::invalid_state("slot table is not created", __FILE__,
                                     __LINE__);
    }
    if (index >= table.max_slots()) {
        return Status::invalid_argument("slot index out of range", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_slot_claim_kernel<<<1, 1, 0u, stream>>>(
        table.device_slots() + index, index, request_id, prompt_length,
        stable_prefix_boundary, max_sequence_length, max_new_tokens, applied);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("slot claim", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_mcu_slot_table_request_cancel(GpuMcuSlotTable& table,
                                         uint32_t index,
                                         RequestHandle handle,
                                         uint32_t* applied,
                                         hipStream_t stream) {
    if (table.device_slots() == nullptr) {
        return Status::invalid_state("slot table is not created", __FILE__,
                                     __LINE__);
    }
    if (index >= table.max_slots()) {
        return Status::invalid_argument("slot index out of range", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_slot_cancel_kernel<<<1, 1, 0u, stream>>>(
        table.device_slots() + index, handle, applied);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("slot cancel", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_mcu_slot_table_mark_output_blocked(GpuMcuSlotTable& table,
                                              uint32_t index,
                                              RequestHandle handle,
                                              uint32_t blocked,
                                              uint32_t* applied,
                                              hipStream_t stream) {
    if (table.device_slots() == nullptr) {
        return Status::invalid_state("slot table is not created", __FILE__,
                                     __LINE__);
    }
    if (index >= table.max_slots()) {
        return Status::invalid_argument("slot index out of range", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_slot_output_blocked_kernel<<<1, 1, 0u, stream>>>(
        table.device_slots() + index, handle, blocked, applied);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("slot output blocked",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_mcu_slot_table_mark_terminal(GpuMcuSlotTable& table,
                                        uint32_t index,
                                        RequestHandle handle,
                                        uint32_t reason,
                                        uint32_t* applied,
                                        hipStream_t stream) {
    if (table.device_slots() == nullptr) {
        return Status::invalid_state("slot table is not created", __FILE__,
                                     __LINE__);
    }
    if (index >= table.max_slots()) {
        return Status::invalid_argument("slot index out of range", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_slot_terminal_kernel<<<1, 1, 0u, stream>>>(
        table.device_slots() + index, handle, reason, applied);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("slot terminal", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_mcu_slot_table_set_phase(GpuMcuSlotTable& table,
                                    uint32_t index,
                                    RequestHandle handle,
                                    uint32_t phase,
                                    uint32_t* applied,
                                    hipStream_t stream) {
    if (table.device_slots() == nullptr) {
        return Status::invalid_state("slot table is not created", __FILE__,
                                     __LINE__);
    }
    if (index >= table.max_slots()) {
        return Status::invalid_argument("slot index out of range", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_slot_phase_kernel<<<1, 1, 0u, stream>>>(
        table.device_slots() + index, handle, phase, applied);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("slot phase", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gpu_mcu_slot_table_release(GpuMcuSlotTable& table,
                                  uint32_t index,
                                  RequestHandle handle,
                                  uint32_t* applied,
                                  hipStream_t stream) {
    if (table.device_slots() == nullptr) {
        return Status::invalid_state("slot table is not created", __FILE__,
                                     __LINE__);
    }
    if (index >= table.max_slots()) {
        return Status::invalid_argument("slot index out of range", __FILE__,
                                        __LINE__);
    }
    gpu_mcu_slot_release_kernel<<<1, 1, 0u, stream>>>(
        table.device_slots() + index, handle, applied);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("slot release", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

const char* gpu_mcu_slot_phase_name(uint32_t phase) noexcept {
    switch (phase) {
        case gpu_mcu_slot_phase_value(GpuMcuSlotPhase::idle):
            return "idle";
        case gpu_mcu_slot_phase_value(GpuMcuSlotPhase::prefill):
            return "prefill";
        case gpu_mcu_slot_phase_value(GpuMcuSlotPhase::decode):
            return "decode";
        case gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify):
            return "verify";
        default:
            return "invalid";
    }
}

const char* gpu_mcu_terminal_reason_name(uint32_t reason) noexcept {
    switch (reason) {
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::none):
            return "none";
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::eos):
            return "eos";
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::stop_token):
            return "stop_token";
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::max_new_tokens):
            return "max_new_tokens";
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::max_seq_len):
            return "max_seq_len";
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::cancelled):
            return "cancelled";
        case gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::error):
            return "error";
        default:
            return "invalid";
    }
}

}  // namespace ps::runtime::gpu_mcu
