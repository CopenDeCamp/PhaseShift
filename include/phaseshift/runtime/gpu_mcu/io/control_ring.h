#pragma once

#include <phaseshift/core/status.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

inline constexpr uint32_t kControlRingCapacity = 256u;
inline constexpr uint32_t kControlRingPayloadBytes = 56u;

static_assert((kControlRingCapacity & (kControlRingCapacity - 1u)) == 0u);
static_assert(kControlRingPayloadBytes == 64u - sizeof(uint64_t));

struct ControlRingPayload {
    std::byte bytes[kControlRingPayloadBytes] = {};
};

static_assert(sizeof(ControlRingPayload) == kControlRingPayloadBytes);

struct alignas(64) ControlRingSlot {
    uint64_t sequence = 0;
    ControlRingPayload payload{};
};

static_assert(sizeof(ControlRingSlot) == 64);
static_assert(alignof(ControlRingSlot) == 64);
static_assert(offsetof(ControlRingSlot, sequence) == 0);

struct alignas(64) ControlRing {
    ControlRingSlot slots[kControlRingCapacity];
};

static_assert(sizeof(ControlRing) == 64u * kControlRingCapacity);

__device__ __forceinline__ uint64_t control_ring_system_load_acquire(
    const uint64_t* ptr) {
    return __scoped_atomic_load_n(ptr, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void control_ring_system_store_release(
    uint64_t* ptr,
    uint64_t value) {
    __scoped_atomic_store_n(ptr, value, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ bool control_ring_try_pop(
    ControlRing* ring,
    uint64_t& position,
    ControlRingPayload& out) {
    ControlRingSlot& slot = ring->slots[position & (kControlRingCapacity - 1u)];
    if (control_ring_system_load_acquire(&slot.sequence) != position + 1u) {
        return false;
    }
    out = slot.payload;
    control_ring_system_store_release(&slot.sequence, position + kControlRingCapacity);
    ++position;
    return true;
}

__device__ __forceinline__ bool control_ring_try_push(
    ControlRing* ring,
    uint64_t& position,
    const ControlRingPayload& in) {
    ControlRingSlot& slot = ring->slots[position & (kControlRingCapacity - 1u)];
    if (control_ring_system_load_acquire(&slot.sequence) != position) {
        return false;
    }
    slot.payload = in;
    control_ring_system_store_release(&slot.sequence, position + 1u);
    ++position;
    return true;
}

class GpuMcuControlRing {
public:
    GpuMcuControlRing() = default;
    ~GpuMcuControlRing() noexcept { (void)shutdown(); }

    GpuMcuControlRing(const GpuMcuControlRing&) = delete;
    GpuMcuControlRing& operator=(const GpuMcuControlRing&) = delete;

    GpuMcuControlRing(GpuMcuControlRing&& other) noexcept { move_from(other); }
    GpuMcuControlRing& operator=(GpuMcuControlRing&& other) noexcept {
        if (this != &other) {
            (void)shutdown();
            move_from(other);
        }
        return *this;
    }

    static Result<GpuMcuControlRing> create(int device);

    bool valid() const noexcept { return host_input_ != nullptr; }

    ControlRing* host_input() const noexcept { return host_input_; }
    ControlRing* host_events() const noexcept { return host_events_; }
    ControlRing* device_input() const noexcept { return device_input_; }
    ControlRing* device_events() const noexcept { return device_events_; }

    bool try_push_input(const ControlRingPayload& payload) noexcept;
    bool try_pop_event(ControlRingPayload& payload) noexcept;

    uint64_t input_position() const noexcept { return input_position_; }
    uint64_t event_position() const noexcept { return event_position_; }

    Status shutdown() noexcept;

private:
    void move_from(GpuMcuControlRing& other) noexcept;

    void* host_allocation_ = nullptr;
    ControlRing* host_input_ = nullptr;
    ControlRing* host_events_ = nullptr;
    ControlRing* device_input_ = nullptr;
    ControlRing* device_events_ = nullptr;
    uint64_t input_position_ = 0;
    uint64_t event_position_ = 0;
};

}  // namespace ps::runtime::gpu_mcu
