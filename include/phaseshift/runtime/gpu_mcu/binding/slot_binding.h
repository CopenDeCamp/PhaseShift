#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/scheduling/slot_table.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ps::runtime::gpu_mcu {

struct alignas(64) GpuMcuSlotBinding {
    RequestHandle handle{};

    uint64_t prompt_tokens_handle = 0u;
    uint64_t sampling_params_handle = 0u;
    uint64_t stop_conditions_handle = 0u;

    int32_t decode_input_token = -1;
    uint32_t decode_input_valid = 0u;

    uint64_t reserved[11] = {};
};

static_assert(sizeof(GpuMcuSlotBinding) == 128u);
static_assert(alignof(GpuMcuSlotBinding) == 64u);
static_assert(std::is_standard_layout_v<GpuMcuSlotBinding>);
static_assert(std::is_trivially_copyable_v<GpuMcuSlotBinding>);

__host__ __device__ __forceinline__ uint64_t gpu_mcu_encode_device_address(
    const void* pointer) noexcept {
    return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

template <typename T>
__host__ __device__ __forceinline__ const T* gpu_mcu_resolve_device_address(
    uint64_t handle) noexcept {
    if (handle == 0u) return nullptr;
    return reinterpret_cast<const T*>(static_cast<std::uintptr_t>(handle));
}

__device__ __forceinline__ void gpu_mcu_binding_init(
    GpuMcuSlotBinding& binding,
    uint32_t index) noexcept {
    GpuMcuSlotBinding fresh{};
    fresh.handle.slot = index;
    fresh.handle.generation = kInvalidRequestGeneration;
    binding = fresh;
}

__host__ __device__ __forceinline__ bool gpu_mcu_binding_valid_for(
    const GpuMcuSlotState& slot,
    const GpuMcuSlotBinding& binding) noexcept {
    if (!gpu_mcu_slot_has_active_request(slot)) return false;
    return request_handle_equal(binding.handle, slot.handle);
}

__host__ __device__ __forceinline__ void gpu_mcu_binding_detach(
    GpuMcuSlotBinding& binding, uint32_t index) noexcept {
    GpuMcuSlotBinding fresh{};
    fresh.handle.slot = index;
    fresh.handle.generation = kInvalidRequestGeneration;
    binding = fresh;
}

__device__ __forceinline__ void gpu_mcu_binding_attach(
    GpuMcuSlotBinding& binding,
    RequestHandle handle,
    uint64_t prompt_tokens_handle,
    uint64_t sampling_params_handle,
    uint64_t stop_conditions_handle) noexcept {
    binding.handle = handle;
    binding.prompt_tokens_handle = prompt_tokens_handle;
    binding.sampling_params_handle = sampling_params_handle;
    binding.stop_conditions_handle = stop_conditions_handle;
    binding.decode_input_token = -1;
    binding.decode_input_valid = 0u;
}

class GpuMcuSlotBindingTable {
public:
    GpuMcuSlotBindingTable() = default;
    ~GpuMcuSlotBindingTable() noexcept;

    static Result<GpuMcuSlotBindingTable> create(int device, uint32_t max_slots);

    Status shutdown() noexcept;

    uint32_t max_slots() const noexcept { return max_slots_; }
    int device() const noexcept { return device_; }
    GpuMcuSlotBinding* device_bindings() const noexcept { return bindings_; }
    bool is_shutdown() const noexcept { return bindings_ == nullptr; }

    GpuMcuSlotBindingTable(GpuMcuSlotBindingTable&& other) noexcept;
    GpuMcuSlotBindingTable& operator=(GpuMcuSlotBindingTable&& other) noexcept;
    GpuMcuSlotBindingTable(const GpuMcuSlotBindingTable&) = delete;
    GpuMcuSlotBindingTable& operator=(const GpuMcuSlotBindingTable&) = delete;

private:
    GpuMcuSlotBinding* bindings_ = nullptr;
    uint32_t max_slots_ = 0u;
    int device_ = 0;
};

}  // namespace ps::runtime::gpu_mcu
