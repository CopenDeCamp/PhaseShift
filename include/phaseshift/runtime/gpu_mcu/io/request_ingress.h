#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/io/control_ring.h>
#include <phaseshift/runtime/gpu_mcu/binding/slot_binding.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/slot_table.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ps::runtime::gpu_mcu {

inline constexpr uint32_t kGpuMcuControlVersion = 1u;
inline constexpr uint32_t kRequestIngressCapacity = 256u;

inline constexpr uint32_t kGpuMcuIngressFree = 0u;
inline constexpr uint32_t kGpuMcuIngressWriting = 1u;
inline constexpr uint32_t kGpuMcuIngressReady = 2u;
inline constexpr uint32_t kGpuMcuIngressClaimed = 3u;
inline constexpr uint32_t kGpuMcuIngressQueued = 4u;

enum class GpuMcuControlOpcode : uint32_t {
    invalid = 0,
    submit = 1,
    cancel = 2,
    shutdown = 3,
};

enum class GpuMcuEventOpcode : uint32_t {
    invalid = 0,
    claimed = 1,
    rejected = 2,
    cancel_accepted = 3,
    cancel_stale = 4,
};

enum class GpuMcuRejectReason : uint32_t {
    none = 0,
    invalid_descriptor = 1,
    invalid_boundary = 2,
    no_idle_slot = 3,
    unsupported_version = 4,
    internal_error = 5,
    admission_backlog_full = 6,
};

struct GpuMcuControlCommand {
    uint32_t opcode = 0u;
    uint32_t version = 0u;
    uint64_t request_id = 0u;
    uint64_t descriptor_handle = 0u;
    uint64_t target_handle_bits = 0u;
    uint64_t reserved[3] = {};
};

static_assert(sizeof(GpuMcuControlCommand) <= kControlRingPayloadBytes);
static_assert(std::is_trivially_copyable_v<GpuMcuControlCommand>);

struct GpuMcuEvent {
    uint32_t opcode = 0u;
    uint32_t reason = 0u;
    uint64_t request_id = 0u;
    uint32_t slot = kInvalidRequestSlot;
    uint32_t generation = 0u;
    uint64_t descriptor_handle = 0u;
    uint64_t reserved = 0u;
};

static_assert(sizeof(GpuMcuEvent) <= kControlRingPayloadBytes);
static_assert(std::is_trivially_copyable_v<GpuMcuEvent>);

__host__ __device__ __host__ __device__ __forceinline__ void gpu_mcu_copy_bytes(void* dst,
                                                        const void* src,
                                                        unsigned long bytes) noexcept {
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    for (unsigned long i = 0; i < bytes; ++i) d[i] = s[i];
}

__host__ __device__ inline ControlRingPayload encode_control_command(
    const GpuMcuControlCommand& command) noexcept {
    ControlRingPayload payload{};
    gpu_mcu_copy_bytes(payload.bytes, &command, sizeof(command));
    return payload;
}

__host__ __device__ inline void decode_control_command(const ControlRingPayload& payload,
                                   GpuMcuControlCommand& out) noexcept {
    gpu_mcu_copy_bytes(&out, payload.bytes, sizeof(out));
}

__host__ __device__ inline ControlRingPayload encode_event(const GpuMcuEvent& event) noexcept {
    ControlRingPayload payload{};
    gpu_mcu_copy_bytes(payload.bytes, &event, sizeof(event));
    return payload;
}

__host__ __device__ inline void decode_event(const ControlRingPayload& payload,
                         GpuMcuEvent& out) noexcept {
    gpu_mcu_copy_bytes(&out, payload.bytes, sizeof(out));
}

__host__ __device__ inline uint64_t encode_target_handle(RequestHandle handle) noexcept {
    return (static_cast<uint64_t>(handle.generation) << 32) |
           static_cast<uint64_t>(handle.slot);
}

__host__ __device__ inline RequestHandle decode_target_handle(uint64_t bits) noexcept {
    RequestHandle handle{};
    handle.slot = static_cast<uint32_t>(bits & 0xFFFFFFFFull);
    handle.generation = static_cast<uint32_t>(bits >> 32);
    return handle;
}

struct alignas(64) GpuMcuRequestDescriptor {
    uint64_t request_id = 0u;

    uint64_t prompt_tokens_handle = 0u;
    uint32_t prompt_length = 0u;
    uint32_t stable_prefix_boundary = 0u;

    uint64_t sampling_params_handle = 0u;
    uint64_t stop_conditions_handle = 0u;

    uint32_t max_new_tokens = 0u;
    uint32_t max_sequence_length = 0u;

    uint32_t state = 0u;
    uint32_t generation = 0u;

    uint64_t reserved[9] = {};
};

static_assert(sizeof(GpuMcuRequestDescriptor) == 128u);
static_assert(alignof(GpuMcuRequestDescriptor) == 64u);
static_assert(std::is_trivially_copyable_v<GpuMcuRequestDescriptor>);

struct alignas(64) GpuMcuRequestIngressEntry {
    uint32_t generation = 0u;
    uint32_t state = 0u;
    uint64_t reserved = 0u;
    GpuMcuRequestDescriptor descriptor{};
};

static_assert(sizeof(GpuMcuRequestIngressEntry) % 64u == 0u);
static_assert(std::is_trivially_copyable_v<GpuMcuRequestIngressEntry>);

// Fixed capacity admission backlog entry. A submit that finds no idle slot is
// queued here instead of being rejected, and is admitted when a slot is released.
struct GpuMcuPendingAdmission {
    uint64_t request_id = 0u;
    uint64_t descriptor_handle = 0u;
    uint32_t descriptor_generation = 0u;
    uint32_t reserved = 0u;
};

static_assert(sizeof(GpuMcuPendingAdmission) == 24u);
static_assert(alignof(GpuMcuPendingAdmission) == 8u);

struct alignas(64) GpuMcuIngressCursor {
    uint64_t input_position = 0u;
    uint64_t event_position = 0u;
    uint64_t reserved[6] = {};
};

static_assert(sizeof(GpuMcuIngressCursor) == 64u);
static_assert(std::is_trivially_copyable_v<GpuMcuIngressCursor>);

__host__ __device__ __forceinline__ bool gpu_mcu_descriptor_validate(
    const GpuMcuRequestDescriptor& descriptor) noexcept {
    if (descriptor.stable_prefix_boundary > descriptor.prompt_length) {
        return false;
    }
    if (descriptor.max_sequence_length == 0u) return false;
    if (descriptor.prompt_length == 0u) return false;
    if (static_cast<uint64_t>(descriptor.prompt_length) +
            static_cast<uint64_t>(descriptor.max_new_tokens) >
        static_cast<uint64_t>(descriptor.max_sequence_length)) {
        return false;
    }
    return true;
}

class GpuMcuRequestIngressArena {
public:
    GpuMcuRequestIngressArena() = default;
    ~GpuMcuRequestIngressArena() noexcept;

    static Result<GpuMcuRequestIngressArena> create(int device,
                                                    uint32_t capacity);

    Status shutdown() noexcept;

    uint32_t capacity() const noexcept { return capacity_; }
    bool valid() const noexcept { return host_entries_ != nullptr; }
    GpuMcuRequestIngressEntry* host_entries() const noexcept {
        return host_entries_;
    }
    GpuMcuRequestIngressEntry* device_entries() const noexcept {
        return device_entries_;
    }

    uint32_t reserve() noexcept;
    GpuMcuRequestDescriptor* descriptor(uint32_t handle) noexcept;
    Status publish(uint32_t handle) noexcept;
    Status recycle(uint32_t handle) noexcept;

    uint32_t state_of(uint32_t handle) const noexcept;
    uint32_t generation_of(uint32_t handle) const noexcept;
    bool is_claimed(uint32_t handle) const noexcept;

    GpuMcuRequestIngressArena(GpuMcuRequestIngressArena&& other) noexcept;
    GpuMcuRequestIngressArena& operator=(
        GpuMcuRequestIngressArena&& other) noexcept;
    GpuMcuRequestIngressArena(const GpuMcuRequestIngressArena&) = delete;
    GpuMcuRequestIngressArena& operator=(
        const GpuMcuRequestIngressArena&) = delete;

private:
    GpuMcuRequestIngressEntry* host_entries_ = nullptr;
    GpuMcuRequestIngressEntry* device_entries_ = nullptr;
    void* allocation_ = nullptr;
    uint32_t capacity_ = 0u;
    int device_ = 0;
};

Result<GpuMcuIngressCursor*> gpu_mcu_ingress_cursor_create(int device);

Status gpu_mcu_process_ingress_once(GpuMcuControlRing& ring,
                                    GpuMcuIngressCursor* cursor,
                                    GpuMcuRequestIngressArena& arena,
                                    GpuMcuSlotTable& slots,
                                    GpuMcuSlotBinding* bindings,
                                    uint32_t binding_max_slots,
                                    uint32_t max_commands,
                                    hipStream_t stream);

const char* gpu_mcu_event_opcode_name(uint32_t opcode) noexcept;
const char* gpu_mcu_reject_reason_name(uint32_t reason) noexcept;

namespace detail {

struct GpuMcuCommandApplyResult {
    ControlRingPayload event{};
    bool has_event = false;
    bool scheduler_dirty = false;
    bool slot_exhausted = false;
    bool cancel_not_found = false;
};

__device__ __forceinline__ GpuMcuCommandApplyResult
gpu_mcu_make_rejected(uint64_t request_id, uint32_t reason) noexcept {
    GpuMcuCommandApplyResult result{};
    GpuMcuEvent event{};
    event.opcode = static_cast<uint32_t>(GpuMcuEventOpcode::rejected);
    event.reason = reason;
    event.request_id = request_id;
    result.event = encode_event(event);
    result.has_event = true;
    return result;
}

__device__ __forceinline__ uint32_t gpu_mcu_find_idle_slot(
    GpuMcuSlotState* slots,
    uint32_t max_slots) noexcept {
    for (uint32_t i = 0; i < max_slots; ++i) {
        if (!gpu_mcu_slot_has_active_request(slots[i])) return i;
    }
    return kInvalidRequestSlot;
}

// Claims an idle slot for an ingress entry that was previously deferred because
// no slot was free. Returns slot_exhausted when the table is still full.
__device__ __forceinline__ GpuMcuCommandApplyResult gpu_mcu_claim_entry(
    GpuMcuRequestIngressEntry& entry,
    uint64_t descriptor_handle,
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots) noexcept {
    const uint32_t index = gpu_mcu_find_idle_slot(slots, max_slots);
    if (index == kInvalidRequestSlot) {
        GpuMcuCommandApplyResult queued{};
        queued.scheduler_dirty = true;
        queued.slot_exhausted = true;
        return queued;
    }
    if (!gpu_mcu_slot_try_claim(
            slots[index], index, entry.descriptor.request_id,
            entry.descriptor.prompt_length,
            entry.descriptor.stable_prefix_boundary,
            entry.descriptor.max_sequence_length,
            entry.descriptor.max_new_tokens)) {
        return gpu_mcu_make_rejected(
            entry.descriptor.request_id,
            static_cast<uint32_t>(GpuMcuRejectReason::internal_error));
    }
    if (bindings != nullptr && index < binding_max_slots) {
        gpu_mcu_binding_attach(bindings[index], slots[index].handle,
                               entry.descriptor.prompt_tokens_handle,
                               entry.descriptor.sampling_params_handle,
                               entry.descriptor.stop_conditions_handle);
        __threadfence_system();
    }
    entry.state = kGpuMcuIngressClaimed;

    GpuMcuCommandApplyResult result{};
    GpuMcuEvent event{};
    event.opcode = static_cast<uint32_t>(GpuMcuEventOpcode::claimed);
    event.request_id = entry.descriptor.request_id;
    event.slot = slots[index].handle.slot;
    event.generation = slots[index].handle.generation;
    event.descriptor_handle = descriptor_handle;
    result.event = encode_event(event);
    result.has_event = true;
    result.scheduler_dirty = true;
    return result;
}

__device__ __forceinline__ GpuMcuCommandApplyResult gpu_mcu_apply_submit(
    const GpuMcuControlCommand& command,
    GpuMcuRequestIngressEntry* entries,
    uint32_t capacity,
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots) noexcept {
    if (command.descriptor_handle == 0u ||
        command.descriptor_handle > capacity) {
        return gpu_mcu_make_rejected(
            command.request_id,
            static_cast<uint32_t>(GpuMcuRejectReason::invalid_descriptor));
    }
    GpuMcuRequestIngressEntry& entry = entries[command.descriptor_handle - 1u];
    if (entry.state != kGpuMcuIngressReady ||
        entry.descriptor.request_id != command.request_id) {
        return gpu_mcu_make_rejected(
            command.request_id,
            static_cast<uint32_t>(GpuMcuRejectReason::invalid_descriptor));
    }
    if (!gpu_mcu_descriptor_validate(entry.descriptor)) {
        const bool boundary = entry.descriptor.stable_prefix_boundary >
                              entry.descriptor.prompt_length;
        return gpu_mcu_make_rejected(
            command.request_id,
            static_cast<uint32_t>(boundary
                                      ? GpuMcuRejectReason::invalid_boundary
                                      : GpuMcuRejectReason::invalid_descriptor));
    }
    const GpuMcuCommandApplyResult claim = gpu_mcu_claim_entry(
        entry, command.descriptor_handle, slots, max_slots, bindings,
        binding_max_slots);
    if (claim.slot_exhausted) {
        entry.state = kGpuMcuIngressQueued;
    }
    return claim;
}

__device__ __forceinline__ GpuMcuCommandApplyResult gpu_mcu_admit_queued_entry(
    GpuMcuRequestIngressEntry* entries,
    uint32_t capacity,
    const GpuMcuPendingAdmission& pending,
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots) noexcept {
    if (pending.descriptor_handle == 0u || pending.descriptor_handle > capacity) {
        return gpu_mcu_make_rejected(
            pending.request_id,
            static_cast<uint32_t>(GpuMcuRejectReason::invalid_descriptor));
    }
    GpuMcuRequestIngressEntry& entry = entries[pending.descriptor_handle - 1u];
    if (entry.state != kGpuMcuIngressQueued ||
        entry.descriptor.request_id != pending.request_id) {
        return gpu_mcu_make_rejected(
            pending.request_id,
            static_cast<uint32_t>(GpuMcuRejectReason::invalid_descriptor));
    }
    return gpu_mcu_claim_entry(entry, pending.descriptor_handle, slots,
                               max_slots, bindings, binding_max_slots);
}

__device__ __forceinline__ GpuMcuCommandApplyResult gpu_mcu_apply_cancel(
    const GpuMcuControlCommand& command,
    GpuMcuSlotState* slots,
    uint32_t max_slots) noexcept {
    const RequestHandle handle = decode_target_handle(command.target_handle_bits);
    GpuMcuEvent event{};
    event.request_id = command.request_id;
    event.slot = handle.slot;
    event.generation = handle.generation;
    GpuMcuCommandApplyResult result{};
    if (handle.slot < max_slots &&
        gpu_mcu_slot_matches_handle(slots[handle.slot], handle) &&
        slots[handle.slot].request_id == command.request_id) {
        slots[handle.slot].cancel_requested = 1u;
        event.opcode =
            static_cast<uint32_t>(GpuMcuEventOpcode::cancel_accepted);
        result.scheduler_dirty = true;
    } else {
        event.opcode = static_cast<uint32_t>(GpuMcuEventOpcode::cancel_stale);
        result.cancel_not_found = true;
    }
    result.event = encode_event(event);
    result.has_event = true;
    return result;
}

__device__ __forceinline__ GpuMcuCommandApplyResult gpu_mcu_apply_control_command(
    const GpuMcuControlCommand& command,
    GpuMcuRequestIngressEntry* entries,
    uint32_t capacity,
    GpuMcuSlotState* slots,
    uint32_t max_slots,
    GpuMcuSlotBinding* bindings,
    uint32_t binding_max_slots) noexcept {
    if (command.version != kGpuMcuControlVersion) {
        return gpu_mcu_make_rejected(
            command.request_id,
            static_cast<uint32_t>(GpuMcuRejectReason::unsupported_version));
    }
    if (command.opcode ==
        static_cast<uint32_t>(GpuMcuControlOpcode::submit)) {
        return gpu_mcu_apply_submit(command, entries, capacity, slots,
                                    max_slots, bindings, binding_max_slots);
    }
    if (command.opcode ==
        static_cast<uint32_t>(GpuMcuControlOpcode::cancel)) {
        return gpu_mcu_apply_cancel(command, slots, max_slots);
    }
    return gpu_mcu_make_rejected(
        command.request_id,
        static_cast<uint32_t>(GpuMcuRejectReason::unsupported_version));
}

}  // namespace detail


}  // namespace ps::runtime::gpu_mcu
