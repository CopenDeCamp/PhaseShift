#include <phaseshift/runtime/gpu_mcu/scheduling/sequence_resource.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace mcu = ::ps::runtime::gpu_mcu;

namespace {

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kResources = 4u;

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

__global__ void init_kernel(mcu::DeviceSequenceResourceManagerView mgr) {
    for (uint32_t i = 0; i < mgr.resource_count; ++i) {
        mgr.resources[i] = mcu::DeviceSequenceResourceEntry{};
        mgr.free_stack[i] = i;
    }
    *mgr.free_count = mgr.resource_count;
}

__global__ void allocate_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                                uint64_t* handle, uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_resource_allocate(mgr, handle) ? 1u : 0u;
}

__global__ void release_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                               uint64_t handle, uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_resource_release(mgr, handle) ? 1u : 0u;
}

__global__ void pool_init_kernel(mcu::GpuMcuKvPagePoolView pool) {
    for (uint32_t i = 0; i < pool.capacity; ++i) {
        pool.free_stack[i] = i;
        pool.owner[i] = 0u;
    }
    *pool.free_count = pool.capacity;
}

__global__ void reserve_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                               mcu::GpuMcuKvPagePoolView pool,
                               mcu::GpuMcuBlockTableView blocks, uint64_t handle,
                               uint32_t sequence_slot, uint32_t required,
                               uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_resource_reserve_kv_pages(
                mgr, pool, blocks, handle, sequence_slot, required)
                ? 1u
                : 0u;
}

__global__ void release_pages_kernel(
    mcu::DeviceSequenceResourceManagerView mgr, mcu::GpuMcuKvPagePoolView pool,
    mcu::GpuMcuBlockTableView blocks, uint64_t handle, uint32_t sequence_slot,
    uint32_t* freed, uint32_t* balanced) {
    freed[0] = 0u;
    (void)mcu::gpu_mcu_resource_release_kv_pages(mgr, pool, blocks, handle,
                                                 sequence_slot, freed);
    balanced[0] = mcu::gpu_mcu_kv_page_pool_balanced(pool) ? 1u : 0u;
}

__global__ void trim_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                            mcu::GpuMcuKvPagePoolView pool,
                            mcu::GpuMcuBlockTableView blocks, uint64_t handle,
                            uint32_t sequence_slot, uint32_t keep_pages,
                            uint32_t* ok, uint32_t* balanced) {
    ok[0] = mcu::gpu_mcu_resource_shrink_kv_pages(mgr, pool, blocks, handle,
                                                  sequence_slot, keep_pages,
                                                  nullptr)
                ? 1u
                : 0u;
    balanced[0] = mcu::gpu_mcu_kv_page_pool_balanced(pool) ? 1u : 0u;
}

__global__ void page_count_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                                  uint64_t handle, uint32_t* count) {
    count[0] = mcu::gpu_mcu_resource_page_count(mgr, handle);
}

__global__ void resolve_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                               uint64_t handle, uint32_t* ok, uint32_t* sequence,
                               uint32_t* gdn) {
    ok[0] = mcu::gpu_mcu_resource_resolve(mgr, handle, sequence, gdn) ? 1u : 0u;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "stream create");

    void* entries_host = nullptr;
    void* entries_dev = nullptr;
    void* stack_host = nullptr;
    void* stack_dev = nullptr;
    void* count_host = nullptr;
    void* count_dev = nullptr;
    void* handle_host = nullptr;
    void* handle_dev = nullptr;
    void* ok_host = nullptr;
    void* ok_dev = nullptr;
    void* sequence_host = nullptr;
    void* sequence_dev = nullptr;
    void* gdn_host = nullptr;
    void* gdn_dev = nullptr;
    void* freed_host = nullptr;
    void* freed_dev = nullptr;
    check(map_alloc(kResources * sizeof(mcu::DeviceSequenceResourceEntry),
                    &entries_host, &entries_dev),
          "entries alloc");
    check(map_alloc(kResources * sizeof(uint32_t), &stack_host, &stack_dev),
          "free stack alloc");
    check(map_alloc(sizeof(uint32_t), &count_host, &count_dev),
          "free count alloc");
    check(map_alloc(sizeof(uint64_t), &handle_host, &handle_dev),
          "handle alloc");
    check(map_alloc(sizeof(uint32_t), &ok_host, &ok_dev), "ok alloc");
    check(map_alloc(sizeof(uint32_t), &sequence_host, &sequence_dev),
          "sequence alloc");
    check(map_alloc(sizeof(uint32_t), &gdn_host, &gdn_dev), "gdn alloc");
    check(map_alloc(sizeof(uint32_t), &freed_host, &freed_dev), "freed alloc");

    auto* entries = static_cast<mcu::DeviceSequenceResourceEntry*>(entries_host);
    auto* count = static_cast<uint32_t*>(count_host);
    auto* handle = static_cast<uint64_t*>(handle_host);
    auto* ok = static_cast<uint32_t*>(ok_host);
    auto* sequence = static_cast<uint32_t*>(sequence_host);
    auto* gdn = static_cast<uint32_t*>(gdn_host);
    auto* freed = static_cast<uint32_t*>(freed_host);
    auto* freed_dev_u32 = static_cast<uint32_t*>(freed_dev);
    void* pool_stack_host = nullptr;
    void* pool_stack_dev = nullptr;
    void* pool_count_host = nullptr;
    void* pool_count_dev = nullptr;
    void* pool_owner_host = nullptr;
    void* pool_owner_dev = nullptr;
    constexpr uint32_t kPoolPages = 8u;
    check(map_alloc(kPoolPages * sizeof(uint32_t), &pool_stack_host,
                    &pool_stack_dev),
          "pool stack alloc");
    check(map_alloc(sizeof(uint32_t), &pool_count_host, &pool_count_dev),
          "pool count alloc");
    check(map_alloc(kPoolPages * sizeof(uint64_t), &pool_owner_host,
                    &pool_owner_dev),
          "pool owner alloc");
    auto* pool_count = static_cast<uint32_t*>(pool_count_host);
    auto* pool_owner = static_cast<uint64_t*>(pool_owner_host);
    auto* handle_dev_u64 = static_cast<uint64_t*>(handle_dev);
    auto* ok_dev_u32 = static_cast<uint32_t*>(ok_dev);
    auto* sequence_dev_u32 = static_cast<uint32_t*>(sequence_dev);
    auto* gdn_dev_u32 = static_cast<uint32_t*>(gdn_dev);

    mcu::DeviceSequenceResourceManagerView mgr{};
    mgr.resources = static_cast<mcu::DeviceSequenceResourceEntry*>(entries_dev);
    mgr.free_stack = static_cast<uint32_t*>(stack_dev);
    mgr.free_count = static_cast<uint32_t*>(count_dev);
    mgr.resource_count = kResources;

    init_kernel<<<1, 1, 0, stream>>>(mgr);
    check(hipStreamSynchronize(stream) == hipSuccess, "init sync");
    check(*count == kResources, "every resource starts free");
    check(*handle == 0u, "no handle is outstanding");

    allocate_kernel<<<1, 1, 0, stream>>>(mgr, handle_dev_u64, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "allocate sync");
    check(ok[0] == 1u, "a resource is allocated");
    check(*count == kResources - 1u, "the free count drops by one");
    const uint64_t first = *handle;
    const uint32_t first_index = mcu::gpu_mcu_resource_handle_index(first);
    const uint32_t first_generation = mcu::gpu_mcu_resource_handle_generation(first);
    check(first_index < kResources, "the handle carries a valid index");
    check(first_generation != 0u, "the handle carries a generation");
    check(entries[first_index].sequence_slot == first_index &&
              entries[first_index].gdn_slot == first_index,
          "the physical slots start one to one");

    resolve_kernel<<<1, 1, 0, stream>>>(mgr, first, ok_dev_u32,
                                        sequence_dev_u32, gdn_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "resolve sync");
    check(ok[0] == 1u, "the live handle resolves");
    check(*sequence == first_index && *gdn == first_index,
          "the resolved slots come from the entry");

    for (uint32_t i = 1u; i < kResources; ++i) {
        allocate_kernel<<<1, 1, 0, stream>>>(mgr, handle_dev_u64, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "allocate more sync");
        check(ok[0] == 1u, "another resource is allocated");
    }
    allocate_kernel<<<1, 1, 0, stream>>>(mgr, handle_dev_u64, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "exhaust sync");
    check(ok[0] == 0u, "an allocation beyond the pool fails");
    check(*count == 0u, "the pool is exhausted");

    release_kernel<<<1, 1, 0, stream>>>(mgr, first, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "release sync");
    check(ok[0] == 1u, "the owning handle releases the resource");
    check(*count == 1u, "the free count is restored");

    resolve_kernel<<<1, 1, 0, stream>>>(mgr, first, ok_dev_u32,
                                        sequence_dev_u32, gdn_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "stale resolve sync");
    check(ok[0] == 0u, "the released handle stops resolving");

    release_kernel<<<1, 1, 0, stream>>>(mgr, first, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "double release sync");
    check(ok[0] == 0u, "a repeated release is refused");
    check(*count == 1u, "a refused release leaves the pool alone");

    allocate_kernel<<<1, 1, 0, stream>>>(mgr, handle_dev_u64, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "reuse sync");
    check(ok[0] == 1u, "the resource is reused");
    const uint64_t second = *handle;
    check(mcu::gpu_mcu_resource_handle_index(second) == first_index,
          "the reuse picks the released index");
    check(second != first, "the reuse carries a new generation");

    resolve_kernel<<<1, 1, 0, stream>>>(mgr, first, ok_dev_u32,
                                        sequence_dev_u32, gdn_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "old resolve sync");
    check(ok[0] == 0u, "the previous handle stays stale after reuse");
    resolve_kernel<<<1, 1, 0, stream>>>(mgr, second, ok_dev_u32,
                                        sequence_dev_u32, gdn_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "new resolve sync");
    check(ok[0] == 1u, "the new handle resolves");

    resolve_kernel<<<1, 1, 0, stream>>>(mgr, 0u, ok_dev_u32, sequence_dev_u32,
                                        gdn_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "zero resolve sync");
    check(ok[0] == 0u, "an empty handle never resolves");

    // Incremental KV page reservation on top of the resource handle.
    {
        void* block_table_host = nullptr;
        void* block_table_dev = nullptr;
        constexpr uint32_t kBlockTableStride = 8u;
        constexpr uint32_t kPageTokens = 4u;
        check(map_alloc(kResources * kBlockTableStride * sizeof(uint32_t),
                        &block_table_host, &block_table_dev),
              "block table alloc");
        auto* block_table = static_cast<uint32_t*>(block_table_host);
        for (uint32_t i = 0; i < kResources * kBlockTableStride; ++i) {
            block_table[i] = 0xFFFFFFFFu;
        }
        mcu::GpuMcuBlockTableView blocks{};
        blocks.block_tables = static_cast<uint32_t*>(block_table_dev);
        blocks.block_table_stride = kBlockTableStride;
        blocks.page_tokens = kPageTokens;
        mcu::GpuMcuKvPagePoolView pool{};
        pool.free_stack = static_cast<uint32_t*>(pool_stack_dev);
        pool.free_count = static_cast<uint32_t*>(pool_count_dev);
        pool.owner = static_cast<uint64_t*>(pool_owner_dev);
        pool.capacity = kPoolPages;
        pool_init_kernel<<<1, 1, 0, stream>>>(pool);
        check(hipStreamSynchronize(stream) == hipSuccess, "pool init sync");
        check(*pool_count == kPoolPages, "the kv pool starts free");

        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, mcu::GpuMcuBlockTableView{}, second, 0u, 3u, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "reserve 3 sync");
        check(ok[0] == 1u, "three pages are reserved");
        check(*pool_count == kPoolPages - 3u, "three pages leave the pool");
        check(entries[first_index].kv_page_count == 3u,
              "the entry records the reservation");

        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, mcu::GpuMcuBlockTableView{}, second, 0u, 2u, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "reserve 2 sync");
        check(ok[0] == 1u, "a smaller request is already satisfied");
        check(*pool_count == kPoolPages - 3u, "no page is taken twice");

        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, mcu::GpuMcuBlockTableView{}, second, 0u, 6u, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "reserve 6 sync");
        check(ok[0] == 1u, "the reservation grows incrementally");
        check(*pool_count == kPoolPages - 6u, "only the difference is taken");
        check(entries[first_index].kv_page_count == 6u, "the entry is updated");

        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, mcu::GpuMcuBlockTableView{}, second, 0u, 12u, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "reserve 12 sync");
        check(ok[0] == 0u, "a request beyond the pool fails");
        check(*pool_count == kPoolPages - 6u, "a failed reservation changes nothing");
        check(entries[first_index].kv_page_count == 6u, "the entry is unchanged");

        release_pages_kernel<<<1, 1, 0, stream>>>(
            mgr, pool, mcu::GpuMcuBlockTableView{}, second, 0u, freed_dev_u32,
            ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "release pages sync");
        check(*freed == 6u, "exactly the reserved pages are released");
        check(*pool_count == kPoolPages, "the pool is whole again");
        check(entries[first_index].kv_page_count == 0u, "the entry is cleared");
        check(ok[0] == 1u, "the pool stays balanced");
        uint32_t stale_owner = 0u;
        for (uint32_t i = 0; i < kPoolPages; ++i) {
            if (pool_owner[i] != 0u) ++stale_owner;
        }
        check(stale_owner == 0u, "no page keeps an owner after the release");

        // The adapter publishes the page ids into the row of the sequence slot.
        check(mcu::gpu_mcu_block_rows_per_page(blocks) == kPageTokens,
              "adapter: the model page size wins");
        check(mcu::gpu_mcu_block_rows_per_page(mcu::GpuMcuBlockTableView{}) ==
                  mcu::kGpuMcuKvRowsPerPage,
              "adapter: the default page size applies");
        check(!mcu::gpu_mcu_block_table_valid(mcu::GpuMcuBlockTableView{}),
              "adapter: an empty view is off");
        const uint32_t row = 0u;
        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, blocks, second, 0u, 2u,
                                            ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "adapter reserve sync");
        check(ok[0] == 1u, "adapter: two blocks are reserved");
        check(block_table[row + 0u] != 0xFFFFFFFFu && block_table[row + 1u] != 0xFFFFFFFFu,
              "adapter: the pages are published");
        check(block_table[row + 0u] != block_table[row + 1u],
              "adapter: each block gets its own page");
        check(block_table[row + 2u] == 0xFFFFFFFFu,
              "adapter: nothing beyond the reservation is touched");
        const uint32_t published_first = block_table[row + 0u];
        const uint32_t published_second = block_table[row + 1u];
        check(pool_owner[published_first] == second &&
                  pool_owner[published_second] == second,
              "adapter: the published pages are owned by the resource");

        release_pages_kernel<<<1, 1, 0, stream>>>(
            mgr, pool, blocks, second, 0u, freed_dev_u32, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "adapter release sync");
        check(*freed == 2u, "adapter: the reserved pages come back");
        check(block_table[row + 0u] == 0u && block_table[row + 1u] == 0u,
              "adapter: the published block is cleared");
        check(block_table[row + 2u] == 0xFFFFFFFFu,
              "adapter: the rest of the row is untouched");

        // A reservation larger than the block table is clamped to its capacity.
        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, blocks, second, 0u, 32u,
                                            ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "clamp reserve sync");
        check(ok[0] == 1u, "adapter: the clamped reservation succeeds");
        check(entries[first_index].kv_page_count == kBlockTableStride,
              "adapter: the reservation stops at the block table capacity");
        release_pages_kernel<<<1, 1, 0, stream>>>(
            mgr, pool, blocks, second, 0u, freed_dev_u32, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "clamp release sync");
        check(*pool_count == kPoolPages, "adapter: the pool is whole again");

        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, blocks, second, 0u, 6u,
                                            ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim reserve sync");
        check(ok[0] == 1u, "trim: six blocks are reserved");
        const uint32_t kept_page_a = block_table[row + 0u];
        const uint32_t kept_page_b = block_table[row + 1u];
        const uint32_t tail_page = block_table[row + 4u];
        trim_kernel<<<1, 1, 0, stream>>>(mgr, pool, blocks, second, 0u, 2u,
                                         ok_dev_u32, sequence_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim sync");
        check(ok[0] == 1u, "trim: two pages are kept");
        check(*sequence == 1u, "trim: the pool stays balanced");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, second, gdn_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim count sync");
        check(*gdn == 2u, "trim: the entry keeps only the retained pages");
        check(block_table[row + 0u] == kept_page_a &&
                  block_table[row + 1u] == kept_page_b,
              "trim: the retained blocks keep their page ids");
        check(block_table[row + 2u] == 0u && block_table[row + 3u] == 0u &&
                  block_table[row + 4u] == 0u && block_table[row + 5u] == 0u,
              "trim: the tail blocks are cleared");
        check(pool_owner[kept_page_a] == second && pool_owner[kept_page_b] == second,
              "trim: the retained pages stay owned");
        check(pool_owner[tail_page] == 0u, "trim: a returned page loses its owner");
        check(*pool_count == kPoolPages - 2u, "trim: only the tail returns");
        release_pages_kernel<<<1, 1, 0, stream>>>(
            mgr, pool, blocks, second, 0u, freed_dev_u32, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim release sync");
        check(*freed == 2u, "trim: the retained pages come back too");
        check(*pool_count == kPoolPages, "trim: the pool is whole again");

        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool,
                                            mcu::GpuMcuBlockTableView{}, second, 0u,
                                            3u, ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim blind reserve sync");
        trim_kernel<<<1, 1, 0, stream>>>(mgr, pool, mcu::GpuMcuBlockTableView{},
                                         second, 0u, 1u, ok_dev_u32,
                                         sequence_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim blind sync");
        check(ok[0] == 0u, "trim: an unpublished trim is refused");
        check(*sequence == 1u, "trim refused: the pool stays balanced");
        release_pages_kernel<<<1, 1, 0, stream>>>(
            mgr, pool, mcu::GpuMcuBlockTableView{}, second, 0u, freed_dev_u32,
            ok_dev_u32);
        check(hipStreamSynchronize(stream) == hipSuccess, "trim blind release sync");
        check(*pool_count == kPoolPages, "trim blind: the pool is whole again");
        check(hipHostFree(block_table_host) == hipSuccess, "block table free");
    }

    check(hipStreamDestroy(stream) == hipSuccess, "stream destroy");
    check(hipHostFree(entries_host) == hipSuccess, "entries free");
    check(hipHostFree(stack_host) == hipSuccess, "stack free");
    check(hipHostFree(count_host) == hipSuccess, "count free");
    check(hipHostFree(handle_host) == hipSuccess, "handle free");
    check(hipHostFree(ok_host) == hipSuccess, "ok free");
    check(hipHostFree(sequence_host) == hipSuccess, "sequence free");
    check(hipHostFree(gdn_host) == hipSuccess, "gdn free");
    check(hipHostFree(freed_host) == hipSuccess, "freed free");
    check(hipHostFree(pool_stack_host) == hipSuccess, "pool stack free");
    check(hipHostFree(pool_count_host) == hipSuccess, "pool count free");
    check(hipHostFree(pool_owner_host) == hipSuccess, "pool owner free");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_sequence_resource: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_sequence_resource: PASS\n");
    return 0;
}
