#pragma once

#include <phaseshift/runtime/gpu_mcu/io/output_ring.h>
#include <phaseshift/runtime/gpu_mcu/slot_binding.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/sequence_resource.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/slot_table.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

// Per slot lifecycle and scheduler bookkeeping that does not fit in the fixed
// 256 byte GpuMcuSlotState. Indexed by the same slot id.
struct alignas(64) GpuMcuSlotRuntimeState {
    uint32_t generation = 0u;
    uint32_t terminal_publish_pending = 0u;
    uint32_t terminal_published = 0u;
    uint32_t release_pending = 0u;
    uint32_t resource_blocked = 0u;
    uint32_t verify_txn_active = 0u;
    uint64_t terminal_epoch = 0u;
    uint64_t last_scheduled_epoch = 0u;
    uint64_t runnable_since_epoch = 0u;
    uint32_t verify_base_page_count = 0u;
    uint32_t verify_reserved_page_count = 0u;
    uint64_t verify_txn_epoch = 0u;
};

static_assert(sizeof(GpuMcuSlotRuntimeState) == 64u);
static_assert(alignof(GpuMcuSlotRuntimeState) == 64u);
static_assert(offsetof(GpuMcuSlotRuntimeState, verify_txn_active) == 20u);
static_assert(offsetof(GpuMcuSlotRuntimeState, verify_base_page_count) == 48u);
static_assert(offsetof(GpuMcuSlotRuntimeState, verify_reserved_page_count) == 52u);
static_assert(offsetof(GpuMcuSlotRuntimeState, verify_txn_epoch) == 56u);

__device__ __forceinline__ void gpu_mcu_slot_runtime_bind(
    GpuMcuSlotRuntimeState& state, uint32_t generation) noexcept {
    if (state.generation == generation) return;
    state = GpuMcuSlotRuntimeState{};
    state.generation = generation;
}

struct GpuMcuTerminalPublishResult {
    uint32_t published = 0u;
    uint32_t waiting = 0u;
};

__device__ __forceinline__ bool gpu_mcu_slot_publish_terminal(
    GpuMcuSlotRuntimeState& state,
    const GpuMcuSlotState& slot,
    OutputRing* ring,
    uint64_t& position) noexcept {
    if (slot.terminal_reason ==
        static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
        return false;
    }
    gpu_mcu_slot_runtime_bind(state, slot.handle.generation);
    if (state.terminal_published != 0u) return false;
    GpuMcuOutputRecord record{};
    record.request_handle_bits =
        (static_cast<uint64_t>(slot.handle.generation) << 32u) |
        static_cast<uint64_t>(slot.handle.slot);
    record.request_id = slot.request_id;
    record.token_id = -1;
    record.token_index = slot.generated_tokens;
    record.flags = kOutputRecordFlagTerminal;
    record.terminal_reason = slot.terminal_reason;
    if (!gpu_mcu_output_ring_try_push(ring, position, record)) {
        state.terminal_publish_pending = 1u;
        return false;
    }
    state.terminal_publish_pending = 0u;
    state.terminal_published = 1u;
    state.release_pending = 1u;
    state.terminal_epoch = slot.generated_tokens;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_publish_terminal_record(
    OutputRing* ring,
    uint64_t& position,
    uint64_t request_id,
    uint64_t request_handle_bits,
    uint32_t terminal_reason) noexcept {
    if (ring == nullptr) return false;
    if (terminal_reason ==
        static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
        return false;
    }
    GpuMcuOutputRecord record{};
    record.request_handle_bits = request_handle_bits;
    record.request_id = request_id;
    record.token_id = -1;
    record.token_index = 0u;
    record.flags = kOutputRecordFlagTerminal;
    record.terminal_reason = terminal_reason;
    return gpu_mcu_output_ring_try_push(ring, position, record);
}

__device__ __forceinline__ GpuMcuTerminalPublishResult
gpu_mcu_publish_slot_terminals(OutputRing* ring,
                               uint64_t& position,
                               const GpuMcuSlotState* slots,
                               GpuMcuSlotRuntimeState* runtime,
                               uint32_t max_slots) noexcept {
    GpuMcuTerminalPublishResult result{};
    if (ring == nullptr || slots == nullptr || runtime == nullptr) return result;
    for (uint32_t i = 0; i < max_slots; ++i) {
        const GpuMcuSlotState& slot = slots[i];
        if (slot.terminal_reason ==
            static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
            continue;
        }
        if (slot.pending_count != 0u) continue;
        if (gpu_mcu_slot_publish_terminal(runtime[i], slot, ring, position)) {
            result.published += 1u;
        } else if (runtime[i].terminal_publish_pending != 0u) {
            result.waiting += 1u;
        }
    }
    return result;
}

struct GpuMcuSlotReleaseResult {
    uint32_t released = 0u;
    uint32_t pending_output = 0u;
    uint32_t awaiting_terminal = 0u;
    uint32_t resource_held = 0u;
    uint32_t released_resources = 0u;
    uint32_t released_pages = 0u;
};

__device__ __forceinline__ bool gpu_mcu_slot_release_eligible(
    const GpuMcuSlotState& slot,
    const GpuMcuSlotRuntimeState& runtime,
    bool resources_managed) noexcept {
    if (slot.terminal_reason ==
        static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
        return false;
    }
    if (slot.pending_count != 0u) return false;
    if (runtime.terminal_published == 0u) return false;
    if (runtime.release_pending == 0u) return false;
    if (!resources_managed &&
        (slot.kv_sequence_handle != 0u || slot.gdn_state_handle != 0u)) {
        return false;
    }
    if (slot.tree_state_handle != 0u || slot.prefix_state_handle != 0u) {
        return false;
    }
    return true;
}

__device__ __forceinline__ GpuMcuSlotReleaseResult
gpu_mcu_release_terminal_slots(GpuMcuSlotState* slots,
                               GpuMcuSlotRuntimeState* runtime,
                               GpuMcuSlotBinding* bindings,
                               uint32_t max_slots,
                               uint32_t binding_max_slots,
                               const DeviceSequenceResourceManagerView& resources,
                               uint32_t resources_enabled,
                               const GpuMcuKvPagePoolView& kv_pages,
                               const GpuMcuBlockTableView& kv_blocks) noexcept {
    GpuMcuSlotReleaseResult result{};
    if (slots == nullptr || runtime == nullptr) return result;
    for (uint32_t i = 0; i < max_slots; ++i) {
        GpuMcuSlotState& slot = slots[i];
        GpuMcuSlotRuntimeState& meta = runtime[i];
        if (slot.terminal_reason ==
            static_cast<uint32_t>(GpuMcuTerminalReason::none)) {
            continue;
        }
        if (slot.pending_count != 0u) {
            result.pending_output += 1u;
            continue;
        }
        if (!gpu_mcu_slot_release_eligible(slot, meta,
                                            resources_enabled != 0u)) {
            if (meta.terminal_published == 0u) {
                result.awaiting_terminal += 1u;
            } else {
                result.resource_held += 1u;
            }
            continue;
        }
        const RequestHandle handle = slot.handle;
        if (resources_enabled != 0u) {
            if (slot.kv_sequence_handle != 0u) {
                uint32_t freed = 0u;
                if (kv_pages.capacity != 0u) {
                    uint32_t sequence_slot = 0u;
                    (void)gpu_mcu_resource_resolve(resources,
                                                   slot.kv_sequence_handle,
                                                   &sequence_slot, nullptr);
                    (void)gpu_mcu_resource_release_kv_pages(
                        resources, kv_pages, kv_blocks, slot.kv_sequence_handle,
                        sequence_slot, &freed);
                }
                result.released_pages += freed;
                if (gpu_mcu_resource_release(resources,
                                             slot.kv_sequence_handle)) {
                    result.released_resources += 1u;
                }
            }
            slot.kv_sequence_handle = 0u;
            slot.gdn_state_handle = 0u;
        }
        if (bindings != nullptr && i < binding_max_slots) {
            gpu_mcu_binding_detach(bindings[i], i);
        }
        if (gpu_mcu_slot_release(slot, handle)) {
            meta = GpuMcuSlotRuntimeState{};
            result.released += 1u;
        }
    }
    return result;
}

}  // namespace ps::runtime::gpu_mcu
