#include <phaseshift/runtime/gpu_mcu/io/control_ring.h>

#include <atomic>
#include <cstddef>
#include <cstring>

namespace ps::runtime::gpu_mcu {

namespace {

struct alignas(64) MappedControlRings {
    ControlRing input;
    ControlRing events;
};

}  // namespace

void GpuMcuControlRing::move_from(GpuMcuControlRing& other) noexcept {
    host_allocation_ = other.host_allocation_;
    host_input_ = other.host_input_;
    host_events_ = other.host_events_;
    device_input_ = other.device_input_;
    device_events_ = other.device_events_;
    input_position_ = other.input_position_;
    event_position_ = other.event_position_;
    other.host_allocation_ = nullptr;
    other.host_input_ = nullptr;
    other.host_events_ = nullptr;
    other.device_input_ = nullptr;
    other.device_events_ = nullptr;
    other.input_position_ = 0;
    other.event_position_ = 0;
}

Result<GpuMcuControlRing> GpuMcuControlRing::create(int device) {
    int host_native_atomic = 0;
    hipError_t err = hipDeviceGetAttribute(
        &host_native_atomic, hipDeviceAttributeHostNativeAtomicSupported, device);
    if (err != hipSuccess) {
        return Status::hip_error("hipDeviceGetAttribute", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (host_native_atomic != 1) {
        return Status::unsupported("host native atomic unsupported",
                                   __FILE__, __LINE__);
    }

    void* host = nullptr;
    err = hipHostMalloc(&host, sizeof(MappedControlRings),
                        hipHostMallocMapped | hipHostMallocCoherent);
    if (err != hipSuccess) {
        return Status::hip_error("hipHostMalloc(mapped|coherent)",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    std::memset(host, 0, sizeof(MappedControlRings));

    void* dev = nullptr;
    err = hipHostGetDevicePointer(&dev, host, 0);
    if (err != hipSuccess) {
        (void)hipFreeHost(host);
        return Status::hip_error("hipHostGetDevicePointer", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    auto* mem = static_cast<MappedControlRings*>(host);
    for (uint32_t i = 0; i < kControlRingCapacity; ++i) {
        mem->input.slots[i].sequence = i;
        mem->events.slots[i].sequence = i;
    }

    GpuMcuControlRing out;
    out.host_allocation_ = host;
    out.host_input_ = &mem->input;
    out.host_events_ = &mem->events;

    const std::size_t input_off =
        reinterpret_cast<const char*>(out.host_input_) -
        reinterpret_cast<const char*>(host);
    const std::size_t events_off =
        reinterpret_cast<const char*>(out.host_events_) -
        reinterpret_cast<const char*>(host);
    out.device_input_ = reinterpret_cast<ControlRing*>(
        static_cast<char*>(dev) + input_off);
    out.device_events_ = reinterpret_cast<ControlRing*>(
        static_cast<char*>(dev) + events_off);
    return out;
}

bool GpuMcuControlRing::try_push_input(const ControlRingPayload& payload) noexcept {
    if (host_input_ == nullptr) return false;
    const uint64_t pos = input_position_;
    ControlRingSlot& slot = host_input_->slots[pos & (kControlRingCapacity - 1u)];
    std::atomic_ref<uint64_t> published(slot.sequence);
    if (published.load(std::memory_order_acquire) != pos) {
        return false;
    }
    slot.payload = payload;
    published.store(pos + 1u, std::memory_order_release);
    input_position_ = pos + 1u;
    return true;
}

bool GpuMcuControlRing::try_pop_event(ControlRingPayload& payload) noexcept {
    if (host_events_ == nullptr) return false;
    const uint64_t pos = event_position_;
    ControlRingSlot& slot = host_events_->slots[pos & (kControlRingCapacity - 1u)];
    std::atomic_ref<uint64_t> published(slot.sequence);
    if (published.load(std::memory_order_acquire) != pos + 1u) {
        return false;
    }
    payload = slot.payload;
    published.store(pos + kControlRingCapacity, std::memory_order_release);
    event_position_ = pos + 1u;
    return true;
}

Status GpuMcuControlRing::shutdown() noexcept {
    if (host_allocation_ == nullptr) return Status::make_ok();
    void* host = host_allocation_;
    host_allocation_ = nullptr;
    host_input_ = nullptr;
    host_events_ = nullptr;
    device_input_ = nullptr;
    device_events_ = nullptr;
    input_position_ = 0;
    event_position_ = 0;
    const hipError_t err = hipFreeHost(host);
    if (err != hipSuccess) {
        return Status::hip_error("hipFreeHost(control rings)", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace ps::runtime::gpu_mcu
