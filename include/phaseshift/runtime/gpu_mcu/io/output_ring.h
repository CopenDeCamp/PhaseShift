#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/io/control_ring.h>
#include <phaseshift/runtime/gpu_mcu/slot_table.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

inline constexpr uint32_t kOutputRingCapacity = 256u;
inline constexpr uint32_t kOutputRecordFlagTerminal = 1u;

struct GpuMcuOutputRecord {
    uint64_t request_handle_bits = 0u;
    uint64_t request_id = 0u;
    int32_t token_id = -1;
    uint32_t token_index = 0u;
    uint32_t flags = 0u;
    uint32_t terminal_reason = 0u;
    uint32_t reserved[6] = {};
};

static_assert(sizeof(GpuMcuOutputRecord) == 56u);

struct alignas(64) OutputRingSlot {
    uint64_t sequence = 0;
    GpuMcuOutputRecord record{};
};

static_assert(sizeof(OutputRingSlot) == 64u);
static_assert(alignof(OutputRingSlot) == 64u);

struct alignas(64) OutputRing {
    OutputRingSlot slots[kOutputRingCapacity];
};

__device__ __forceinline__ bool gpu_mcu_output_ring_try_push(
    OutputRing* ring,
    uint64_t& position,
    const GpuMcuOutputRecord& record) noexcept {
    if (ring == nullptr) return false;
    OutputRingSlot& slot = ring->slots[position & (kOutputRingCapacity - 1u)];
    if (control_ring_system_load_acquire(&slot.sequence) != position) {
        return false;
    }
    slot.record = record;
    control_ring_system_store_release(&slot.sequence, position + 1u);
    ++position;
    return true;
}

struct GpuMcuOutputFlushResult {
    uint32_t flushed_tokens = 0u;
    uint32_t blocked_slots = 0u;
    uint32_t unblocked_slots = 0u;
};

__device__ __forceinline__ GpuMcuOutputFlushResult gpu_mcu_flush_slot_outputs(
    OutputRing* ring,
    uint64_t& position,
    GpuMcuSlotState* slots,
    uint32_t max_slots) noexcept {
    GpuMcuOutputFlushResult result{};
    if (ring == nullptr || slots == nullptr) return result;
    for (uint32_t i = 0; i < max_slots; ++i) {
        GpuMcuSlotState& slot = slots[i];
        if (slot.pending_count == 0u) {
            if (slot.output_blocked != 0u) {
                slot.output_blocked = 0u;
                result.unblocked_slots += 1u;
            }
            continue;
        }
        if (slot.pending_begin >= kGpuMcuSlotPendingTokens) {
            slot.pending_begin = 0u;
        }
        uint32_t pushed = 0u;
        while (pushed < slot.pending_count) {
            const uint32_t index =
                (slot.pending_begin + pushed) % kGpuMcuSlotPendingTokens;
            GpuMcuOutputRecord record{};
            record.request_handle_bits =
                (static_cast<uint64_t>(slot.handle.generation) << 32u) |
                static_cast<uint64_t>(slot.handle.slot);
            record.request_id = slot.request_id;
            record.token_id = slot.pending_tokens[index];
            record.token_index =
                slot.generated_tokens > slot.pending_count
                    ? slot.generated_tokens - slot.pending_count + pushed
                    : pushed;
            record.flags = 0u;
            record.terminal_reason = slot.terminal_reason;
            if (!gpu_mcu_output_ring_try_push(ring, position, record)) break;
            pushed += 1u;
        }
        result.flushed_tokens += pushed;
        if (pushed == slot.pending_count) {
            slot.pending_count = 0u;
            slot.pending_begin = 0u;
            slot.output_blocked = 0u;
        } else {
            slot.pending_begin =
                (slot.pending_begin + pushed) % kGpuMcuSlotPendingTokens;
            slot.pending_count -= pushed;
            slot.output_blocked = 1u;
            result.blocked_slots += 1u;
        }
    }
    return result;
}

class GpuMcuOutputRing {
public:
    GpuMcuOutputRing() = default;
    ~GpuMcuOutputRing() noexcept { (void)shutdown(); }

    GpuMcuOutputRing(const GpuMcuOutputRing&) = delete;
    GpuMcuOutputRing& operator=(const GpuMcuOutputRing&) = delete;

    GpuMcuOutputRing(GpuMcuOutputRing&& other) noexcept { move_from(other); }
    GpuMcuOutputRing& operator=(GpuMcuOutputRing&& other) noexcept {
        if (this != &other) {
            (void)shutdown();
            move_from(other);
        }
        return *this;
    }

    static Result<GpuMcuOutputRing> create(int device);

    bool valid() const noexcept { return host_ring_ != nullptr; }
    OutputRing* host_ring() const noexcept { return host_ring_; }
    OutputRing* device_ring() const noexcept { return device_ring_; }
    uint64_t position() const noexcept { return position_; }

    bool try_pop(GpuMcuOutputRecord& out) noexcept;

    Status shutdown() noexcept;

private:
    void move_from(GpuMcuOutputRing& other) noexcept;

    void* host_allocation_ = nullptr;
    OutputRing* host_ring_ = nullptr;
    OutputRing* device_ring_ = nullptr;
    uint64_t position_ = 0;
};

}  // namespace ps::runtime::gpu_mcu
