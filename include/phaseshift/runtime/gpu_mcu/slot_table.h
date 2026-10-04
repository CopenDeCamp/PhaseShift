#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/request_handle.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ps::runtime::gpu_mcu {

inline constexpr uint32_t kGpuMcuMaxSlots = 256u;
inline constexpr uint32_t kGpuMcuSlotBytes = 256u;
inline constexpr uint32_t kGpuMcuSlotPendingTokens = 32u;
inline constexpr uint32_t kGpuMcuSlotVerifyTokens = 32u;

enum class GpuMcuSlotPhase : uint32_t {
    idle = 0,
    prefill = 1,
    decode = 2,
    verify = 3,
};

enum class GpuMcuTerminalReason : uint32_t {
    none = 0,
    eos = 1,
    stop_token = 2,
    max_new_tokens = 3,
    max_seq_len = 4,
    cancelled = 5,
    error = 6,
};

struct alignas(64) GpuMcuSlotState {
    RequestHandle handle;
    uint64_t request_id;

    uint32_t phase;
    uint32_t terminal_reason;
    uint32_t cancel_requested;
    uint32_t output_blocked;

    uint32_t sequence_length;
    uint32_t committed_position;
    uint32_t generated_tokens;
    uint32_t max_sequence_length;
    uint32_t max_new_tokens;

    uint32_t prompt_length;
    uint32_t prefill_position;
    uint32_t stable_prefix_boundary;

    uint32_t verify_candidate_count;
    uint32_t verify_committed_count;

    uint64_t verify_candidate_ref;

    uint64_t kv_sequence_handle;
    uint64_t gdn_state_handle;
    uint64_t tree_state_handle;

    uint64_t prefix_state_handle;

    uint32_t prefix_restored_length;
    uint32_t prefix_checkpoint_position;

    uint32_t pending_count;
    uint32_t pending_begin;

    int32_t pending_tokens[kGpuMcuSlotPendingTokens];
};

static_assert(sizeof(GpuMcuSlotState) == kGpuMcuSlotBytes);
static_assert(alignof(GpuMcuSlotState) == 64);
static_assert(std::is_standard_layout_v<GpuMcuSlotState>);
static_assert(std::is_trivially_copyable_v<GpuMcuSlotState>);

constexpr uint32_t gpu_mcu_slot_phase_value(GpuMcuSlotPhase phase) noexcept {
    return static_cast<uint32_t>(phase);
}

constexpr uint32_t gpu_mcu_terminal_reason_value(
    GpuMcuTerminalReason reason) noexcept {
    return static_cast<uint32_t>(reason);
}

__host__ __device__ __forceinline__ bool gpu_mcu_slot_phase_valid(
    uint32_t phase) noexcept {
    return phase <= gpu_mcu_slot_phase_value(GpuMcuSlotPhase::verify);
}

__host__ __device__ __forceinline__ bool gpu_mcu_terminal_reason_valid(
    uint32_t reason) noexcept {
    return reason <=
           gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::error);
}

__host__ __device__ __forceinline__ bool gpu_mcu_slot_has_active_request(
    const GpuMcuSlotState& slot) noexcept {
    return slot.phase != gpu_mcu_slot_phase_value(GpuMcuSlotPhase::idle);
}

__host__ __device__ __forceinline__ bool gpu_mcu_slot_matches_handle(
    const GpuMcuSlotState& slot,
    RequestHandle handle) noexcept {
    return gpu_mcu_slot_has_active_request(slot) &&
           request_handle_equal(slot.handle, handle);
}

__host__ __device__ __forceinline__ bool gpu_mcu_slot_is_runnable(
    const GpuMcuSlotState& slot) noexcept {
    return gpu_mcu_slot_has_active_request(slot) &&
           slot.terminal_reason ==
               gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::none) &&
           slot.cancel_requested == 0u &&
           slot.output_blocked == 0u;
}

__host__ __device__ __forceinline__ bool gpu_mcu_slot_validate(
    const GpuMcuSlotState& slot) noexcept {
    if (!gpu_mcu_slot_phase_valid(slot.phase)) return false;
    if (!gpu_mcu_terminal_reason_valid(slot.terminal_reason)) return false;
    if (slot.verify_candidate_count > kGpuMcuSlotVerifyTokens) return false;
    if (slot.verify_committed_count > kGpuMcuSlotVerifyTokens) return false;
    if (slot.pending_count > kGpuMcuSlotPendingTokens) return false;
    if (slot.prefill_position > slot.prompt_length) return false;
    if (slot.stable_prefix_boundary > slot.prompt_length) return false;
    if (slot.committed_position > slot.sequence_length) return false;
    if (gpu_mcu_slot_has_active_request(slot)) {
        if (slot.handle.slot == kInvalidRequestSlot) return false;
        if (slot.handle.generation == kInvalidRequestGeneration) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ uint32_t gpu_mcu_slot_next_generation(
    uint32_t generation) noexcept {
    if (generation == kInvalidRequestGeneration) return 1u;
    const uint32_t next = generation + 1u;
    return next == kInvalidRequestGeneration ? 1u : next;
}

__device__ __forceinline__ bool gpu_mcu_slot_try_claim(
    GpuMcuSlotState& slot,
    uint32_t index,
    uint64_t request_id,
    uint32_t prompt_length,
    uint32_t stable_prefix_boundary,
    uint32_t max_sequence_length,
    uint32_t max_new_tokens) noexcept {
    if (gpu_mcu_slot_has_active_request(slot)) return false;
    GpuMcuSlotState fresh{};
    fresh.handle.slot = index;
    fresh.handle.generation =
        slot.handle.generation == kInvalidRequestGeneration
            ? gpu_mcu_slot_next_generation(slot.handle.generation)
            : slot.handle.generation;
    fresh.request_id = request_id;
    fresh.phase = gpu_mcu_slot_phase_value(GpuMcuSlotPhase::prefill);
    fresh.terminal_reason =
        gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::none);
    fresh.prompt_length = prompt_length;
    fresh.stable_prefix_boundary = stable_prefix_boundary;
    fresh.max_sequence_length = max_sequence_length;
    fresh.max_new_tokens = max_new_tokens;
    slot = fresh;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_slot_request_cancel(
    GpuMcuSlotState& slot,
    RequestHandle handle) noexcept {
    if (!gpu_mcu_slot_matches_handle(slot, handle)) return false;
    slot.cancel_requested = 1u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_slot_mark_output_blocked(
    GpuMcuSlotState& slot,
    RequestHandle handle,
    uint32_t blocked) noexcept {
    if (!gpu_mcu_slot_matches_handle(slot, handle)) return false;
    slot.output_blocked = blocked != 0u ? 1u : 0u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_slot_mark_terminal(
    GpuMcuSlotState& slot,
    RequestHandle handle,
    uint32_t reason) noexcept {
    if (!gpu_mcu_slot_matches_handle(slot, handle)) return false;
    if (!gpu_mcu_terminal_reason_valid(reason)) return false;
    if (reason == gpu_mcu_terminal_reason_value(GpuMcuTerminalReason::none)) {
        return false;
    }
    slot.terminal_reason = reason;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_slot_stage_token(
    GpuMcuSlotState& slot, int32_t token) noexcept {
    if (slot.pending_count >= kGpuMcuSlotPendingTokens) return false;
    const uint32_t index =
        (slot.pending_begin + slot.pending_count) % kGpuMcuSlotPendingTokens;
    slot.pending_tokens[index] = token;
    slot.pending_count += 1u;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_slot_set_phase(
    GpuMcuSlotState& slot,
    RequestHandle handle,
    uint32_t phase) noexcept {
    if (!gpu_mcu_slot_matches_handle(slot, handle)) return false;
    if (!gpu_mcu_slot_phase_valid(phase)) return false;
    slot.phase = phase;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_slot_release(
    GpuMcuSlotState& slot,
    RequestHandle handle) noexcept {
    if (!gpu_mcu_slot_matches_handle(slot, handle)) return false;
    if (slot.pending_count != 0u) return false;
    if (slot.handle.generation == UINT32_MAX) return false;
    GpuMcuSlotState idle{};
    idle.handle.slot = slot.handle.slot;
    idle.handle.generation =
        gpu_mcu_slot_next_generation(slot.handle.generation);
    slot = idle;
    return true;
}

__device__ __forceinline__ void gpu_mcu_slot_init(
    GpuMcuSlotState& slot,
    uint32_t index) noexcept {
    GpuMcuSlotState fresh{};
    fresh.handle.slot = index;
    fresh.handle.generation = kInvalidRequestGeneration;
    slot = fresh;
}

class GpuMcuSlotTable {
public:
    GpuMcuSlotTable() = default;
    ~GpuMcuSlotTable() noexcept;

    static Result<GpuMcuSlotTable> create(int device, uint32_t max_slots);

    Status shutdown() noexcept;

    uint32_t max_slots() const noexcept { return max_slots_; }
    int device() const noexcept { return device_; }
    GpuMcuSlotState* device_slots() const noexcept { return slots_; }
    bool is_shutdown() const noexcept { return slots_ == nullptr; }

    GpuMcuSlotTable(GpuMcuSlotTable&& other) noexcept;
    GpuMcuSlotTable& operator=(GpuMcuSlotTable&& other) noexcept;
    GpuMcuSlotTable(const GpuMcuSlotTable&) = delete;
    GpuMcuSlotTable& operator=(const GpuMcuSlotTable&) = delete;

private:
    GpuMcuSlotState* slots_ = nullptr;
    uint32_t max_slots_ = 0u;
    int device_ = 0;
};

Status gpu_mcu_slot_table_claim(
    GpuMcuSlotTable& table,
    uint32_t index,
    uint64_t request_id,
    uint32_t prompt_length,
    uint32_t stable_prefix_boundary,
    uint32_t max_sequence_length,
    uint32_t max_new_tokens,
    uint32_t* applied,
    hipStream_t stream);

Status gpu_mcu_slot_table_request_cancel(
    GpuMcuSlotTable& table,
    uint32_t index,
    RequestHandle handle,
    uint32_t* applied,
    hipStream_t stream);

Status gpu_mcu_slot_table_mark_output_blocked(
    GpuMcuSlotTable& table,
    uint32_t index,
    RequestHandle handle,
    uint32_t blocked,
    uint32_t* applied,
    hipStream_t stream);

Status gpu_mcu_slot_table_mark_terminal(
    GpuMcuSlotTable& table,
    uint32_t index,
    RequestHandle handle,
    uint32_t reason,
    uint32_t* applied,
    hipStream_t stream);

Status gpu_mcu_slot_table_set_phase(
    GpuMcuSlotTable& table,
    uint32_t index,
    RequestHandle handle,
    uint32_t phase,
    uint32_t* applied,
    hipStream_t stream);

Status gpu_mcu_slot_table_release(
    GpuMcuSlotTable& table,
    uint32_t index,
    RequestHandle handle,
    uint32_t* applied,
    hipStream_t stream);

const char* gpu_mcu_slot_phase_name(uint32_t phase) noexcept;
const char* gpu_mcu_terminal_reason_name(uint32_t reason) noexcept;

}  // namespace ps::runtime::gpu_mcu
