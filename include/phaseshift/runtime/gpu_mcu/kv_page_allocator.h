#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

// A KV page is owned by exactly one request: the slot id plus the handle
// generation. 0 means free, so a stale request can neither free nor observe a
// page that now belongs to another request.
__host__ __device__ __forceinline__ uint64_t gpu_mcu_kv_page_owner(
    uint32_t slot, uint32_t generation) {
    return (static_cast<uint64_t>(generation) << 32u) |
           (static_cast<uint64_t>(slot) + 1u);
}

__host__ __device__ __forceinline__ uint32_t gpu_mcu_kv_page_owner_slot(
    uint64_t owner) {
    return static_cast<uint32_t>(owner & 0xffffffffu) - 1u;
}

__host__ __device__ __forceinline__ uint32_t gpu_mcu_kv_page_owner_generation(
    uint64_t owner) {
    return static_cast<uint32_t>(owner >> 32u);
}

// Device visible KV page pool. The MCU is the only writer during inference.
struct GpuMcuKvPagePoolView {
    uint32_t* free_stack = nullptr;
    uint32_t* free_count = nullptr;
    uint64_t* owner = nullptr;
    uint32_t capacity = 0u;
};

// The tag is opaque: the caller decides whether it encodes a slot or a resource.
__device__ __forceinline__ bool gpu_mcu_kv_page_alloc_tag(
    const GpuMcuKvPagePoolView& pool,
    uint64_t tag,
    uint32_t count,
    uint32_t* out_first) noexcept {
    if (pool.free_stack == nullptr || pool.free_count == nullptr ||
        pool.owner == nullptr || out_first == nullptr || tag == 0u) {
        return false;
    }
    const uint32_t free_pages = *pool.free_count;
    if (count == 0u || free_pages < count) return false;
    *out_first = pool.free_stack[free_pages - 1u];
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t page = pool.free_stack[free_pages - 1u - i];
        pool.owner[page] = tag;
    }
    __threadfence_system();
    *pool.free_count = free_pages - count;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_kv_page_alloc(
    const GpuMcuKvPagePoolView& pool,
    uint32_t slot,
    uint32_t generation,
    uint32_t count,
    uint32_t* out_first) noexcept {
    return gpu_mcu_kv_page_alloc_tag(pool, gpu_mcu_kv_page_owner(slot, generation),
                                     count, out_first);
}

// Releases every page still owned by this exact request. A stale generation or a
// duplicate release owns no page, so it frees nothing and reports zero.
__device__ __forceinline__ bool gpu_mcu_kv_page_free_tag(
    const GpuMcuKvPagePoolView& pool,
    uint64_t tag,
    uint32_t* out_freed) noexcept {
    if (pool.free_stack == nullptr || pool.free_count == nullptr ||
        pool.owner == nullptr || tag == 0u) {
        return false;
    }
    uint32_t freed = 0u;
    uint32_t cursor = *pool.free_count;
    for (uint32_t page = 0; page < pool.capacity; ++page) {
        if (pool.owner[page] != tag) continue;
        pool.owner[page] = 0u;
        if (cursor < pool.capacity) {
            pool.free_stack[cursor] = page;
            ++cursor;
        }
        ++freed;
    }
    if (out_freed != nullptr) *out_freed = freed;
    __threadfence_system();
    *pool.free_count = cursor;
    return freed != 0u;
}

__device__ __forceinline__ bool gpu_mcu_kv_page_free_owned(
    const GpuMcuKvPagePoolView& pool,
    uint32_t slot,
    uint32_t generation,
    uint32_t* out_freed) noexcept {
    return gpu_mcu_kv_page_free_tag(
        pool, gpu_mcu_kv_page_owner(slot, generation), out_freed);
}

__device__ __forceinline__ uint32_t gpu_mcu_kv_page_owned_count(
    const GpuMcuKvPagePoolView& pool) noexcept {
    if (pool.owner == nullptr) return 0u;
    uint32_t owned = 0u;
    for (uint32_t page = 0; page < pool.capacity; ++page) {
        if (pool.owner[page] != 0u) ++owned;
    }
    return owned;
}

// free + owned must always equal capacity.
__device__ __forceinline__ bool gpu_mcu_kv_page_pool_balanced(
    const GpuMcuKvPagePoolView& pool) noexcept {
    if (pool.free_count == nullptr) return false;
    return *pool.free_count + gpu_mcu_kv_page_owned_count(pool) == pool.capacity;
}

}  // namespace ps::runtime::gpu_mcu
