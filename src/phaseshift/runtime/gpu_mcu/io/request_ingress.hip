#include <phaseshift/runtime/gpu_mcu/io/request_ingress.h>

#include <cstddef>
#include <cstring>

namespace ps::runtime::gpu_mcu {
namespace {

__global__ void gpu_mcu_process_ingress_once_kernel(
    ControlRing* input,
    ControlRing* events,
    GpuMcuIngressCursor* cursor,
    GpuMcuRequestIngressEntry* entries,
    uint32_t capacity,
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots,
    uint32_t max_commands) {
    uint64_t event_position = cursor->event_position;
    for (uint32_t processed = 0; processed < max_commands; ++processed) {
        ControlRingPayload payload{};
        if (!control_ring_try_pop(input, cursor->input_position, payload)) {
            break;
        }
        GpuMcuControlCommand command{};
        decode_control_command(payload, command);
        const detail::GpuMcuCommandApplyResult result =
            detail::gpu_mcu_apply_control_command(command, entries, capacity,
                                                  slots, max_slots, bindings,
                                                  binding_max_slots);
        if (result.has_event) {
            (void)control_ring_try_push(events, event_position, result.event);
        }
    }
    cursor->event_position = event_position;
}

}  // namespace

GpuMcuRequestIngressArena::~GpuMcuRequestIngressArena() noexcept {
    const Status status = shutdown();
    (void)status;
}

Result<GpuMcuRequestIngressArena> GpuMcuRequestIngressArena::create(
    int device,
    uint32_t capacity) {
    if (capacity == 0u || capacity > kRequestIngressCapacity) {
        return Status::invalid_argument("ingress capacity out of range",
                                        __FILE__, __LINE__);
    }
    GpuMcuRequestIngressArena arena{};
    arena.capacity_ = capacity;
    arena.device_ = device;

    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return Status::hip_error("hipSetDevice", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    const std::size_t bytes =
        static_cast<std::size_t>(capacity) * sizeof(GpuMcuRequestIngressEntry);
    void* host = nullptr;
    err = hipHostMalloc(&host, bytes, hipHostMallocMapped | hipHostMallocCoherent);
    if (err != hipSuccess) {
        return Status::hip_error("hipHostMalloc ingress arena",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    std::memset(host, 0, bytes);
    void* dev = nullptr;
    err = hipHostGetDevicePointer(&dev, host, 0);
    if (err != hipSuccess) {
        (void)hipHostFree(host);
        return Status::hip_error("hipHostGetDevicePointer ingress arena",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    arena.allocation_ = host;
    arena.host_entries_ = static_cast<GpuMcuRequestIngressEntry*>(host);
    arena.device_entries_ = static_cast<GpuMcuRequestIngressEntry*>(dev);
    return arena;
}

Status GpuMcuRequestIngressArena::shutdown() noexcept {
    if (allocation_ == nullptr) {
        capacity_ = 0u;
        return Status::make_ok();
    }
    void* allocation = allocation_;
    allocation_ = nullptr;
    host_entries_ = nullptr;
    device_entries_ = nullptr;
    capacity_ = 0u;
    const hipError_t err = hipHostFree(allocation);
    if (err != hipSuccess) {
        return Status::hip_error("hipHostFree ingress arena",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

GpuMcuRequestIngressArena::GpuMcuRequestIngressArena(
    GpuMcuRequestIngressArena&& other) noexcept
    : host_entries_(other.host_entries_),
      device_entries_(other.device_entries_),
      allocation_(other.allocation_),
      capacity_(other.capacity_),
      device_(other.device_) {
    other.host_entries_ = nullptr;
    other.device_entries_ = nullptr;
    other.allocation_ = nullptr;
    other.capacity_ = 0u;
}

GpuMcuRequestIngressArena& GpuMcuRequestIngressArena::operator=(
    GpuMcuRequestIngressArena&& other) noexcept {
    if (this != &other) {
        const Status status = shutdown();
        (void)status;
        host_entries_ = other.host_entries_;
        device_entries_ = other.device_entries_;
        allocation_ = other.allocation_;
        capacity_ = other.capacity_;
        device_ = other.device_;
        other.host_entries_ = nullptr;
        other.device_entries_ = nullptr;
        other.allocation_ = nullptr;
        other.capacity_ = 0u;
    }
    return *this;
}

uint32_t GpuMcuRequestIngressArena::reserve() noexcept {
    if (host_entries_ == nullptr) return 0u;
    for (uint32_t i = 0; i < capacity_; ++i) {
        auto state = std::atomic_ref<uint32_t>(host_entries_[i].state);
        uint32_t expected = kGpuMcuIngressFree;
        if (state.compare_exchange_strong(expected, kGpuMcuIngressWriting)) {
            return i + 1u;
        }
    }
    return 0u;
}

GpuMcuRequestDescriptor* GpuMcuRequestIngressArena::descriptor(
    uint32_t handle) noexcept {
    if (handle == 0u || handle > capacity_) return nullptr;
    return &host_entries_[handle - 1u].descriptor;
}

Status GpuMcuRequestIngressArena::publish(uint32_t handle) noexcept {
    if (handle == 0u || handle > capacity_) {
        return Status::invalid_argument("ingress handle out of range",
                                        __FILE__, __LINE__);
    }
    GpuMcuRequestIngressEntry& entry = host_entries_[handle - 1u];
    if (entry.state != kGpuMcuIngressWriting) {
        return Status::invalid_state("ingress entry is not being written",
                                     __FILE__, __LINE__);
    }
    entry.descriptor.state = kGpuMcuIngressReady;
    entry.descriptor.generation = entry.generation;
    auto state = std::atomic_ref<uint32_t>(entry.state);
    state.store(kGpuMcuIngressReady, std::memory_order_release);
    return Status::make_ok();
}

Status GpuMcuRequestIngressArena::recycle(uint32_t handle) noexcept {
    if (handle == 0u || handle > capacity_) {
        return Status::invalid_argument("ingress handle out of range",
                                        __FILE__, __LINE__);
    }
    GpuMcuRequestIngressEntry& entry = host_entries_[handle - 1u];
    const uint32_t state = entry.state;
    if (state == kGpuMcuIngressFree) {
        return Status::invalid_state("ingress entry is already free", __FILE__,
                                     __LINE__);
    }
    ++entry.generation;
    entry.descriptor = GpuMcuRequestDescriptor{};
    auto state_ref = std::atomic_ref<uint32_t>(entry.state);
    state_ref.store(kGpuMcuIngressFree, std::memory_order_release);
    return Status::make_ok();
}

uint32_t GpuMcuRequestIngressArena::state_of(uint32_t handle) const noexcept {
    if (handle == 0u || handle > capacity_) return kGpuMcuIngressFree;
    auto state = std::atomic_ref<uint32_t>(host_entries_[handle - 1u].state);
    return state.load(std::memory_order_acquire);
}

uint32_t GpuMcuRequestIngressArena::generation_of(uint32_t handle) const
    noexcept {
    if (handle == 0u || handle > capacity_) return 0u;
    return host_entries_[handle - 1u].generation;
}

bool GpuMcuRequestIngressArena::is_claimed(uint32_t handle) const noexcept {
    return state_of(handle) == kGpuMcuIngressClaimed;
}

Result<GpuMcuIngressCursor*> gpu_mcu_ingress_cursor_create(int device) {
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return Status::hip_error("hipSetDevice cursor", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    GpuMcuIngressCursor* cursor = nullptr;
    err = hipMalloc(reinterpret_cast<void**>(&cursor), sizeof(GpuMcuIngressCursor));
    if (err != hipSuccess) {
        return Status::hip_error("hipMalloc ingress cursor",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    const hipError_t init =
        hipMemset(cursor, 0, sizeof(GpuMcuIngressCursor));
    if (init != hipSuccess) {
        (void)hipFree(cursor);
        return Status::hip_error("hipMemset ingress cursor",
                                 hipGetErrorString(init), __FILE__, __LINE__);
    }
    return cursor;
}

Status gpu_mcu_process_ingress_once(GpuMcuControlRing& ring,
                                    GpuMcuIngressCursor* cursor,
                                    GpuMcuRequestIngressArena& arena,
                                    GpuMcuSlotTable& slots,
                                    GpuMcuSlotBinding* bindings,
                                    uint32_t binding_max_slots,
                                    uint32_t max_commands,
                                    hipStream_t stream) {
    if (!ring.valid()) {
        return Status::invalid_state("control ring is not created", __FILE__,
                                     __LINE__);
    }
    if (cursor == nullptr) {
        return Status::invalid_argument("ingress cursor is null", __FILE__,
                                        __LINE__);
    }
    if (!arena.valid() || slots.device_slots() == nullptr) {
        return Status::invalid_state("ingress arena or slot table is not created",
                                     __FILE__, __LINE__);
    }
    if (max_commands == 0u) {
        return Status::invalid_argument("max_commands must be non-zero",
                                        __FILE__, __LINE__);
    }
    gpu_mcu_process_ingress_once_kernel<<<1, 1, 0u, stream>>>(
        ring.device_input(), ring.device_events(), cursor,
        arena.device_entries(), arena.capacity(), slots.device_slots(),
        slots.max_slots(), bindings, binding_max_slots, max_commands);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("process ingress once", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

const char* gpu_mcu_event_opcode_name(uint32_t opcode) noexcept {
    switch (opcode) {
        case static_cast<uint32_t>(GpuMcuEventOpcode::invalid):
            return "invalid";
        case static_cast<uint32_t>(GpuMcuEventOpcode::claimed):
            return "claimed";
        case static_cast<uint32_t>(GpuMcuEventOpcode::rejected):
            return "rejected";
        case static_cast<uint32_t>(GpuMcuEventOpcode::cancel_accepted):
            return "cancel_accepted";
        case static_cast<uint32_t>(GpuMcuEventOpcode::cancel_stale):
            return "cancel_stale";
        default:
            return "unknown";
    }
}

const char* gpu_mcu_reject_reason_name(uint32_t reason) noexcept {
    switch (reason) {
        case static_cast<uint32_t>(GpuMcuRejectReason::none):
            return "none";
        case static_cast<uint32_t>(GpuMcuRejectReason::invalid_descriptor):
            return "invalid_descriptor";
        case static_cast<uint32_t>(GpuMcuRejectReason::invalid_boundary):
            return "invalid_boundary";
        case static_cast<uint32_t>(GpuMcuRejectReason::no_idle_slot):
            return "no_idle_slot";
        case static_cast<uint32_t>(GpuMcuRejectReason::unsupported_version):
            return "unsupported_version";
        case static_cast<uint32_t>(GpuMcuRejectReason::internal_error):
            return "internal_error";
        default:
            return "unknown";
    }
}

}  // namespace ps::runtime::gpu_mcu
