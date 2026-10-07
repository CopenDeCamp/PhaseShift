#pragma once

#include <hip/hip_runtime.h>

#include <phaseshift/runtime/gpu_mcu/scheduling/kv_page_allocator.h>

#include <cstdint>

namespace ps::runtime::gpu_mcu {

// The slot keeps an opaque 64 bit handle. Only the resource manager interprets
// it, and it carries BOTH the resource index and a resource generation that is
// independent from the request slot generation. A released resource therefore
// invalidates every old handle even before the slot generation moves.
//
// handle = (resource_generation << 32) | (resource_index + 1)
// 0 means "no resource".
__host__ __device__ __forceinline__ uint64_t gpu_mcu_resource_handle(
    uint32_t index, uint32_t generation) {
    return (static_cast<uint64_t>(generation) << 32u) |
           (static_cast<uint64_t>(index) + 1u);
}

__host__ __device__ __forceinline__ uint32_t gpu_mcu_resource_handle_index(
    uint64_t handle) {
    return static_cast<uint32_t>(handle & 0xffffffffu) - 1u;
}

__host__ __device__ __forceinline__ uint32_t gpu_mcu_resource_handle_generation(
    uint64_t handle) {
    return static_cast<uint32_t>(handle >> 32u);
}

// Rows covered by one KV page block. The reservation is expressed in pages so
// the resource manager stays independent from the block table layout.
inline constexpr uint32_t kGpuMcuKvRowsPerPage = 16u;

// Adapter to the block table the model side owns. The manager writes the page id
// of every block it hands out into the row the physical sequence slot owns, so
// the existing kv kernels keep reading the model layout and never see the
// resource manager. An empty view keeps the adapter off and only tracks counts.
struct GpuMcuBlockTableView {
    uint32_t* block_tables = nullptr;
    uint32_t block_table_stride = 0u;
    uint32_t page_tokens = 0u;
};

__host__ __device__ __forceinline__ bool gpu_mcu_block_table_valid(
    const GpuMcuBlockTableView& blocks) noexcept {
    return blocks.block_tables != nullptr && blocks.block_table_stride != 0u;
}

__host__ __device__ __forceinline__ uint32_t gpu_mcu_block_rows_per_page(
    const GpuMcuBlockTableView& blocks) noexcept {
    return blocks.page_tokens != 0u ? blocks.page_tokens : kGpuMcuKvRowsPerPage;
}

inline constexpr uint32_t kGpuMcuResourceFree = 0u;
inline constexpr uint32_t kGpuMcuResourceActive = 1u;

struct DeviceSequenceResourceEntry {
    uint32_t generation = 0u;
    uint32_t state = 0u;
    uint32_t sequence_slot = 0u;
    uint32_t gdn_slot = 0u;
    uint32_t kv_page_count = 0u;
    uint32_t reserved = 0u;
};

struct DeviceSequenceResourceManagerView {
    DeviceSequenceResourceEntry* resources = nullptr;
    uint32_t* free_stack = nullptr;
    uint32_t* free_count = nullptr;
    uint32_t resource_count = 0u;
};

__host__ __device__ __forceinline__ uint32_t gpu_mcu_resource_next_generation(
    uint32_t generation) {
    const uint32_t next = generation + 1u;
    return next == 0u ? 1u : next;
}

__host__ __device__ __forceinline__ bool gpu_mcu_resource_resolve(
    const DeviceSequenceResourceManagerView& mgr,
    uint64_t handle,
    uint32_t* out_sequence_slot,
    uint32_t* out_gdn_slot) {
    if (mgr.resources == nullptr || handle == 0u) return false;
    const uint32_t index = gpu_mcu_resource_handle_index(handle);
    if (index >= mgr.resource_count) return false;
    const DeviceSequenceResourceEntry& entry = mgr.resources[index];
    if (entry.state != kGpuMcuResourceActive) return false;
    if (entry.generation != gpu_mcu_resource_handle_generation(handle)) {
        return false;
    }
    if (out_sequence_slot != nullptr) *out_sequence_slot = entry.sequence_slot;
    if (out_gdn_slot != nullptr) *out_gdn_slot = entry.gdn_slot;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_resource_allocate(
    const DeviceSequenceResourceManagerView& mgr, uint64_t* out_handle) {
    if (mgr.resources == nullptr || mgr.free_stack == nullptr ||
        mgr.free_count == nullptr || out_handle == nullptr) {
        return false;
    }
    const uint32_t free_count = *mgr.free_count;
    if (free_count == 0u) return false;
    const uint32_t index = mgr.free_stack[free_count - 1u];
    if (index >= mgr.resource_count) return false;
    DeviceSequenceResourceEntry& entry = mgr.resources[index];
    entry.generation = gpu_mcu_resource_next_generation(entry.generation);
    entry.state = kGpuMcuResourceActive;
    entry.sequence_slot = index;
    entry.gdn_slot = index;
    __threadfence_system();
    *mgr.free_count = free_count - 1u;
    *out_handle = gpu_mcu_resource_handle(index, entry.generation);
    return true;
}

// Grows the resource to hold a sequence of required_pages pages. The reservation
// is incremental: only the missing pages are taken, so a long sequence does not
// reserve its whole maximum length up front.
__device__ __forceinline__ bool gpu_mcu_resource_reserve_kv_pages(
    const DeviceSequenceResourceManagerView& mgr,
    const GpuMcuKvPagePoolView& pool,
    const GpuMcuBlockTableView& blocks,
    uint64_t handle,
    uint32_t sequence_slot,
    uint32_t required_pages) {
    if (!gpu_mcu_resource_resolve(mgr, handle, nullptr, nullptr)) return false;
    const uint32_t index = gpu_mcu_resource_handle_index(handle);
    DeviceSequenceResourceEntry& entry = mgr.resources[index];
    const bool publish = gpu_mcu_block_table_valid(blocks);
    if (publish && required_pages > blocks.block_table_stride) {
        required_pages = blocks.block_table_stride;
    }
    if (entry.kv_page_count >= required_pages) return true;
    const uint32_t additional = required_pages - entry.kv_page_count;
    if (pool.free_count == nullptr || *pool.free_count < additional) {
        return false;
    }
    while (entry.kv_page_count < required_pages) {
        const uint32_t block = entry.kv_page_count;
        uint32_t page = 0u;
        if (!gpu_mcu_kv_page_alloc_tag(pool, handle, 1u, &page)) return false;
        entry.kv_page_count = block + 1u;
        if (publish) {
            blocks.block_tables[sequence_slot * blocks.block_table_stride +
                                 block] = page;
        }
        __threadfence_system();
    }
    return true;
}

__device__ __forceinline__ bool gpu_mcu_resource_release_kv_pages(
    const DeviceSequenceResourceManagerView& mgr,
    const GpuMcuKvPagePoolView& pool,
    const GpuMcuBlockTableView& blocks,
    uint64_t handle,
    uint32_t sequence_slot,
    uint32_t* out_freed) {
    if (!gpu_mcu_resource_resolve(mgr, handle, nullptr, nullptr)) return false;
    const uint32_t index = gpu_mcu_resource_handle_index(handle);
    DeviceSequenceResourceEntry& entry = mgr.resources[index];
    if (gpu_mcu_block_table_valid(blocks)) {
        uint32_t blocks_owned = entry.kv_page_count;
        if (blocks_owned > blocks.block_table_stride) {
            blocks_owned = blocks.block_table_stride;
        }
        for (uint32_t block = 0u; block < blocks_owned; ++block) {
            blocks.block_tables[sequence_slot * blocks.block_table_stride +
                                 block] = 0u;
        }
    }
    uint32_t freed = 0u;
    (void)gpu_mcu_kv_page_free_tag(pool, handle, &freed);
    entry.kv_page_count = 0u;
    __threadfence_system();
    if (out_freed != nullptr) *out_freed = freed;
    return true;
}

__device__ __forceinline__ uint32_t gpu_mcu_resource_page_count(
    const DeviceSequenceResourceManagerView& mgr, uint64_t handle) {
    if (!gpu_mcu_resource_resolve(mgr, handle, nullptr, nullptr)) return 0u;
    return mgr.resources[gpu_mcu_resource_handle_index(handle)].kv_page_count;
}

// Returns the pages a sequence reserved beyond keep_pages, newest block first.
// The pages [0, keep_pages) are left untouched so the committed prefix keeps its
// data and its block table entries. The whole tail is validated before any
// mutation, so a single bad entry leaves the pool, the block table and the
// resource page count unchanged. Page id 0 is a valid page: ownership is the
// only authority, never the zero clear value.
__device__ __forceinline__ bool gpu_mcu_resource_shrink_kv_pages(
    const DeviceSequenceResourceManagerView& mgr,
    const GpuMcuKvPagePoolView& pool,
    const GpuMcuBlockTableView& blocks,
    uint64_t handle,
    uint32_t sequence_slot,
    uint32_t keep_pages,
    uint32_t* out_freed) {
    if (out_freed != nullptr) *out_freed = 0u;
    if (!gpu_mcu_resource_resolve(mgr, handle, nullptr, nullptr)) return false;
    const uint32_t index = gpu_mcu_resource_handle_index(handle);
    DeviceSequenceResourceEntry& entry = mgr.resources[index];
    const uint32_t current = entry.kv_page_count;
    if (keep_pages > current) return false;
    if (keep_pages == current) return true;
    if (!gpu_mcu_block_table_valid(blocks)) return false;
    if (current > blocks.block_table_stride) return false;
    if (pool.free_stack == nullptr || pool.free_count == nullptr ||
        pool.owner == nullptr || pool.capacity == 0u) {
        return false;
    }
    const uint32_t to_free = current - keep_pages;
    const uint32_t free_count = *pool.free_count;
    if (free_count > pool.capacity ||
        to_free > pool.capacity - free_count) {
        return false;
    }
    const uint32_t row = sequence_slot * blocks.block_table_stride;
    for (uint32_t block = keep_pages; block < current; ++block) {
        const uint32_t page = blocks.block_tables[row + block];
        if (page >= pool.capacity) return false;
        if (pool.owner[page] != handle) return false;
    }
    uint32_t free_cursor = free_count;
    for (uint32_t block = keep_pages; block < current; ++block) {
        const uint32_t page = blocks.block_tables[row + block];
        pool.owner[page] = 0u;
        pool.free_stack[free_cursor] = page;
        ++free_cursor;
        blocks.block_tables[row + block] = 0u;
    }
    entry.kv_page_count = keep_pages;
    __threadfence_system();
    *pool.free_count = free_cursor;
    if (out_freed != nullptr) *out_freed = to_free;
    return true;
}

__device__ __forceinline__ bool gpu_mcu_resource_release(
    const DeviceSequenceResourceManagerView& mgr, uint64_t handle) {
    if (mgr.resources == nullptr || mgr.free_stack == nullptr ||
        mgr.free_count == nullptr) {
        return false;
    }
    uint32_t sequence_slot = 0u;
    if (!gpu_mcu_resource_resolve(mgr, handle, &sequence_slot, nullptr)) {
        return false;
    }
    const uint32_t index = gpu_mcu_resource_handle_index(handle);
    DeviceSequenceResourceEntry& entry = mgr.resources[index];
    entry.state = kGpuMcuResourceFree;
    entry.generation = gpu_mcu_resource_next_generation(entry.generation);
    entry.sequence_slot = 0u;
    entry.gdn_slot = 0u;
    __threadfence_system();
    const uint32_t free_count = *mgr.free_count;
    if (free_count < mgr.resource_count) {
        mgr.free_stack[free_count] = index;
    }
    *mgr.free_count = free_count + 1u;
    return true;
}

}  // namespace ps::runtime::gpu_mcu
