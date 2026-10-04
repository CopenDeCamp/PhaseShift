#include <phaseshift/runtime/gpu_mcu/output_ring.h>

#include <atomic>
#include <cstddef>
#include <cstring>

namespace ps::runtime::gpu_mcu {

void GpuMcuOutputRing::move_from(GpuMcuOutputRing& other) noexcept {
    host_allocation_ = other.host_allocation_;
    host_ring_ = other.host_ring_;
    device_ring_ = other.device_ring_;
    position_ = other.position_;
    other.host_allocation_ = nullptr;
    other.host_ring_ = nullptr;
    other.device_ring_ = nullptr;
    other.position_ = 0;
}

Result<GpuMcuOutputRing> GpuMcuOutputRing::create(int device) {
    int host_native_atomic = 0;
    hipError_t err = hipDeviceGetAttribute(
        &host_native_atomic, hipDeviceAttributeHostNativeAtomicSupported, device);
    if (err != hipSuccess) {
        return Status::hip_error("hipDeviceGetAttribute", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (host_native_atomic != 1) {
        return Status::unsupported("host native atomic unsupported", __FILE__,
                                   __LINE__);
    }

    void* host = nullptr;
    err = hipHostMalloc(&host, sizeof(OutputRing),
                        hipHostMallocMapped | hipHostMallocCoherent);
    if (err != hipSuccess) {
        return Status::hip_error("hipHostMalloc(mapped|coherent)",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    std::memset(host, 0, sizeof(OutputRing));

    void* dev = nullptr;
    err = hipHostGetDevicePointer(&dev, host, 0);
    if (err != hipSuccess) {
        (void)hipFreeHost(host);
        return Status::hip_error("hipHostGetDevicePointer", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    auto* mem = static_cast<OutputRing*>(host);
    for (uint32_t i = 0; i < kOutputRingCapacity; ++i) {
        mem->slots[i].sequence = i;
    }

    GpuMcuOutputRing out;
    out.host_allocation_ = host;
    out.host_ring_ = mem;
    out.device_ring_ = static_cast<OutputRing*>(dev);
    return out;
}

bool GpuMcuOutputRing::try_pop(GpuMcuOutputRecord& out) noexcept {
    if (host_ring_ == nullptr) return false;
    const uint64_t pos = position_;
    OutputRingSlot& slot = host_ring_->slots[pos & (kOutputRingCapacity - 1u)];
    std::atomic_ref<uint64_t> published(slot.sequence);
    if (published.load(std::memory_order_acquire) != pos + 1u) {
        return false;
    }
    out = slot.record;
    published.store(pos + kOutputRingCapacity, std::memory_order_release);
    position_ = pos + 1u;
    return true;
}

Status GpuMcuOutputRing::shutdown() noexcept {
    if (host_allocation_ == nullptr) return Status::make_ok();
    void* host = host_allocation_;
    host_allocation_ = nullptr;
    host_ring_ = nullptr;
    device_ring_ = nullptr;
    position_ = 0;
    const hipError_t err = hipFreeHost(host);
    if (err != hipSuccess) {
        return Status::hip_error("hipFreeHost(output ring)",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace ps::runtime::gpu_mcu
