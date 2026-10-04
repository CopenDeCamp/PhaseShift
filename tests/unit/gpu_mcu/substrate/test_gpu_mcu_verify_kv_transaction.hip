#include <phaseshift/runtime/gpu_mcu/batch_commit.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>
#include <phaseshift/runtime/gpu_mcu/sequence_resource.h>
#include <phaseshift/runtime/gpu_mcu/slot_runtime.h>

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

constexpr uint32_t kResources = 2u;
constexpr uint32_t kPoolPages = 12u;
constexpr uint32_t kBlockStride = 16u;
constexpr uint32_t kSequences = 2u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kCandidates = 32u;
constexpr uint32_t kPrefix = 15u;

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

__global__ void res_init_kernel(mcu::DeviceSequenceResourceManagerView mgr) {
    for (uint32_t i = 0; i < mgr.resource_count; ++i) {
        mgr.resources[i] = mcu::DeviceSequenceResourceEntry{};
        mgr.free_stack[i] = (i + 1u) % mgr.resource_count;
    }
    *mgr.free_count = mgr.resource_count;
}

__global__ void pool_init_kernel(mcu::GpuMcuKvPagePoolView pool,
                                 uint32_t page_zero_first) {
    for (uint32_t i = 0; i < pool.capacity; ++i) {
        pool.owner[i] = 0u;
        pool.free_stack[i] =
            page_zero_first != 0u ? (i + 1u) % pool.capacity : i;
    }
    *pool.free_count = pool.capacity;
}

__global__ void alloc_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                             uint64_t* handle, uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_resource_allocate(mgr, handle) ? 1u : 0u;
}

__global__ void release_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                               uint64_t handle, uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_resource_release(mgr, handle) ? 1u : 0u;
}

__global__ void reserve_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                               mcu::GpuMcuKvPagePoolView pool,
                               mcu::GpuMcuBlockTableView blocks, uint64_t handle,
                               uint32_t sequence_slot, uint32_t required,
                               uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_resource_reserve_kv_pages(mgr, pool, blocks, handle,
                                                   sequence_slot, required)
                ? 1u
                : 0u;
}

__global__ void prepare_txn_kernel(
    mcu::GpuMcuSlotState* slots, mcu::GpuMcuSlotRuntimeState* runtime,
    mcu::DeviceSequenceResourceManagerView mgr, mcu::GpuMcuKvPagePoolView pool,
    mcu::GpuMcuBlockTableView blocks, uint64_t handle, uint32_t slot_index,
    uint32_t commit_position, uint32_t candidates, uint32_t base_pages,
    uint32_t* ok) {
    const uint32_t rows_per_page = mcu::gpu_mcu_block_rows_per_page(blocks);
    const uint32_t required =
        (commit_position + candidates + rows_per_page - 1u) / rows_per_page;
    uint32_t sequence_slot = 0u;
    (void)mcu::gpu_mcu_resource_resolve(mgr, handle, &sequence_slot, nullptr);
    const bool reserved = mcu::gpu_mcu_resource_reserve_kv_pages(
        mgr, pool, blocks, handle, sequence_slot, required);
    mcu::GpuMcuSlotState& slot = slots[slot_index];
    mcu::GpuMcuSlotRuntimeState& meta = runtime[slot_index];
    slot = mcu::GpuMcuSlotState{};
    slot.handle = ::ps::runtime::RequestHandle{slot_index, 1u};
    slot.phase = mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::verify);
    slot.committed_position = commit_position;
    slot.verify_candidate_count = candidates;
    slot.kv_sequence_handle = handle;
    meta = mcu::GpuMcuSlotRuntimeState{};
    meta.verify_txn_active = 1u;
    meta.verify_base_page_count = base_pages;
    const uint32_t current =
        mcu::gpu_mcu_resource_page_count(mgr, handle);
    meta.verify_reserved_page_count =
        current > base_pages ? current - base_pages : 0u;
    ok[0] = reserved ? 1u : 0u;
}

__global__ void commit_txn_kernel(
    mcu::GpuMcuSlotState* slots, mcu::GpuMcuSlotRuntimeState* runtime,
    mcu::DeviceSequenceResourceManagerView mgr, mcu::GpuMcuKvPagePoolView pool,
    mcu::GpuMcuBlockTableView blocks, uint32_t slot_index, uint32_t accepted,
    uint32_t* ok, uint32_t* freed) {
    mcu::GpuMcuSlotState& slot = slots[slot_index];
    mcu::GpuMcuSlotRuntimeState& meta = runtime[slot_index];
    const uint32_t before =
        mcu::gpu_mcu_resource_page_count(mgr, slot.kv_sequence_handle);
    ok[0] = mcu::gpu_mcu_verify_resource_commit(slot, meta, mgr, pool, blocks,
                                                accepted)
                ? 1u
                : 0u;
    const uint32_t after =
        mcu::gpu_mcu_resource_page_count(mgr, slot.kv_sequence_handle);
    freed[0] = before > after ? before - after : 0u;
}

__global__ void abort_txn_kernel(
    mcu::GpuMcuSlotState* slots, mcu::GpuMcuSlotRuntimeState* runtime,
    mcu::DeviceSequenceResourceManagerView mgr, mcu::GpuMcuKvPagePoolView pool,
    mcu::GpuMcuBlockTableView blocks, uint32_t slot_index, uint32_t* ok,
    uint32_t* freed) {
    mcu::GpuMcuSlotState& slot = slots[slot_index];
    mcu::GpuMcuSlotRuntimeState& meta = runtime[slot_index];
    const uint32_t before =
        mcu::gpu_mcu_resource_page_count(mgr, slot.kv_sequence_handle);
    ok[0] = mcu::gpu_mcu_verify_resource_abort(slot, meta, mgr, pool, blocks)
                ? 1u
                : 0u;
    const uint32_t after =
        mcu::gpu_mcu_resource_page_count(mgr, slot.kv_sequence_handle);
    freed[0] = before > after ? before - after : 0u;
}

__global__ void page_count_kernel(mcu::DeviceSequenceResourceManagerView mgr,
                                  uint64_t handle, uint32_t* count) {
    count[0] = mcu::gpu_mcu_resource_page_count(mgr, handle);
}

__global__ void balanced_kernel(mcu::GpuMcuKvPagePoolView pool, uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_kv_page_pool_balanced(pool) ? 1u : 0u;
}

__global__ void corrupt_owner_kernel(mcu::GpuMcuKvPagePoolView pool,
                                     uint32_t page, uint64_t owner) {
    pool.owner[page] = owner;
}

__global__ void reserve_pass_kernel(mcu::GpuMcuPersistentState* state,
                                    uint32_t* changed) {
    changed[0] = mcu::gpu_mcu_resource_reserve_pass(state) ? 1u : 0u;
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

    void *entries_h = nullptr, *entries_d = nullptr;
    void *stack_h = nullptr, *stack_d = nullptr;
    void *count_h = nullptr, *count_d = nullptr;
    void *pool_stack_h = nullptr, *pool_stack_d = nullptr;
    void *pool_count_h = nullptr, *pool_count_d = nullptr;
    void *pool_owner_h = nullptr, *pool_owner_d = nullptr;
    void *blocks_h = nullptr, *blocks_d = nullptr;
    void *slots_h = nullptr, *slots_d = nullptr;
    void *runtime_h = nullptr, *runtime_d = nullptr;
    void *handle_h = nullptr, *handle_d = nullptr;
    void *ok_h = nullptr, *ok_d = nullptr;
    void *freed_h = nullptr, *freed_d = nullptr;
    check(map_alloc(kResources * sizeof(mcu::DeviceSequenceResourceEntry),
                    &entries_h, &entries_d),
          "entries alloc");
    check(map_alloc(kResources * sizeof(uint32_t), &stack_h, &stack_d),
          "stack alloc");
    check(map_alloc(sizeof(uint32_t), &count_h, &count_d), "count alloc");
    check(map_alloc(kPoolPages * sizeof(uint32_t), &pool_stack_h, &pool_stack_d),
          "pool stack alloc");
    check(map_alloc(sizeof(uint32_t), &pool_count_h, &pool_count_d),
          "pool count alloc");
    check(map_alloc(kPoolPages * sizeof(uint64_t), &pool_owner_h, &pool_owner_d),
          "pool owner alloc");
    check(map_alloc(kSequences * kBlockStride * sizeof(uint32_t), &blocks_h,
                    &blocks_d),
          "blocks alloc");
    check(map_alloc(2u * sizeof(mcu::GpuMcuSlotState), &slots_h, &slots_d),
          "slots alloc");
    check(map_alloc(2u * sizeof(mcu::GpuMcuSlotRuntimeState), &runtime_h,
                    &runtime_d),
          "runtime alloc");
    check(map_alloc(sizeof(uint64_t), &handle_h, &handle_d), "handle alloc");
    check(map_alloc(sizeof(uint32_t), &ok_h, &ok_d), "ok alloc");
    check(map_alloc(sizeof(uint32_t), &freed_h, &freed_d), "freed alloc");

    auto* pool_count = static_cast<uint32_t*>(pool_count_h);
    auto* pool_owner = static_cast<uint64_t*>(pool_owner_h);
    auto* blocks = static_cast<uint32_t*>(blocks_h);
    auto* handle = static_cast<uint64_t*>(handle_h);
    auto* ok = static_cast<uint32_t*>(ok_h);
    auto* freed = static_cast<uint32_t*>(freed_h);

    mcu::DeviceSequenceResourceManagerView mgr{};
    mgr.resources = static_cast<mcu::DeviceSequenceResourceEntry*>(entries_d);
    mgr.free_stack = static_cast<uint32_t*>(stack_d);
    mgr.free_count = static_cast<uint32_t*>(count_d);
    mgr.resource_count = kResources;
    mcu::GpuMcuKvPagePoolView pool{};
    pool.free_stack = static_cast<uint32_t*>(pool_stack_d);
    pool.free_count = static_cast<uint32_t*>(pool_count_d);
    pool.owner = static_cast<uint64_t*>(pool_owner_d);
    pool.capacity = kPoolPages;
    mcu::GpuMcuBlockTableView table{};
    table.block_tables = static_cast<uint32_t*>(blocks_d);
    table.block_table_stride = kBlockStride;
    table.page_tokens = kPageTokens;
    auto* slots = static_cast<mcu::GpuMcuSlotState*>(slots_d);
    auto* runtime =
        static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_d);

    auto reset = [&](uint32_t page_zero_first) {
        for (uint32_t i = 0; i < kSequences * kBlockStride; ++i) blocks[i] = 0u;
        res_init_kernel<<<1, 1, 0, stream>>>(mgr);
        pool_init_kernel<<<1, 1, 0, stream>>>(pool, page_zero_first);
    };

    // Case 1-6: prefix 15, candidate 32, accepted 0/1/2/17/18/32.
    struct BoundaryCase {
        uint32_t accepted;
        uint32_t expected_keep;
        uint32_t expected_freed;
    };
    const BoundaryCase boundaries[] = {
        {0u, 1u, 2u},  {1u, 1u, 2u},  {2u, 2u, 1u},
        {17u, 2u, 1u}, {18u, 3u, 0u}, {32u, 3u, 0u},
    };
    for (const BoundaryCase& bc : boundaries) {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "boundary alloc sync");
        const uint64_t h = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, h, 0u, kPrefix, kCandidates, 1u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "boundary prepare sync");
        check(ok[0] == 1u, "boundary prepare reserved");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "boundary count sync");
        check(ok[0] == (kPrefix + kCandidates + kPageTokens - 1u) / kPageTokens,
              "boundary speculative pages reserved");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, bc.accepted,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "boundary commit sync");
        check(ok[0] == 1u, "boundary commit succeeds");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "boundary keep sync");
        check(ok[0] == bc.expected_keep, "boundary keep pages");
        check(*freed == bc.expected_freed, "boundary rolled back pages");
        balanced_kernel<<<1, 1, 0, stream>>>(
            pool, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "boundary balanced sync");
        check(ok[0] == 1u, "boundary pool balanced");
    }

    // Case 7: capacity exhaustion is all-or-nothing.
    {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "cap alloc sync");
        const uint64_t h = *handle;
        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, table, h, 0u,
                                            kPoolPages,
                                            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "cap reserve sync");
        check(ok[0] == 1u, "cap: the whole capacity is reserved");
        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, table, h, 0u,
                                            kPoolPages + 1u,
                                            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "cap over sync");
        check(ok[0] == 0u, "cap: over reservation fails");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "cap count sync");
        check(ok[0] == kPoolPages, "cap: page count unchanged");
        balanced_kernel<<<1, 1, 0, stream>>>(
            pool, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "cap balanced sync");
        check(ok[0] == 1u, "cap: pool balanced");
    }

    // Case 8: stale handle after release.
    {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "stale alloc sync");
        const uint64_t stale = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, stale, 0u, kPrefix, kCandidates,
            1u, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "stale prepare sync");
        release_kernel<<<1, 1, 0, stream>>>(mgr, stale,
                                            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "stale release sync");
        check(ok[0] == 1u, "stale release succeeds");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 4u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "stale commit sync");
        check(ok[0] == 0u, "stale commit refused");
        check(*freed == 0u, "stale commit mutates nothing");
        balanced_kernel<<<1, 1, 0, stream>>>(
            pool, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "stale balanced sync");
        check(ok[0] == 1u, "stale: pool balanced");
    }

    // Case 9: wrong owner in the tail.
    {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "owner alloc sync");
        const uint64_t h = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, h, 0u, kPrefix, kCandidates, 1u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "owner prepare sync");
        const uint32_t tail_page = blocks[2u];
        corrupt_owner_kernel<<<1, 1, 0, stream>>>(pool, tail_page, 0xDEADBEEFull);
        check(hipStreamSynchronize(stream) == hipSuccess, "owner corrupt sync");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 0u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "owner commit sync");
        check(ok[0] == 0u, "owner mismatch refuses the commit");
        check(*freed == 0u, "owner mismatch mutates nothing");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "owner count sync");
        check(ok[0] == 3u, "owner mismatch keeps the page count");
    }

    // Case 10: duplicate finalize does not double free.
    {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup alloc sync");
        const uint64_t h = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, h, 0u, kPrefix, kCandidates, 1u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup prepare sync");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 4u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup commit sync");
        check(ok[0] == 1u, "dup first commit succeeds");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup count sync");
        const uint32_t kept = ok[0];
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 4u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup second sync");
        check(ok[0] == 0u, "dup second commit refused");
        check(*freed == 0u, "dup second commit mutates nothing");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup final count sync");
        check(ok[0] == kept, "dup page count unchanged");
        balanced_kernel<<<1, 1, 0, stream>>>(
            pool, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "dup balanced sync");
        check(ok[0] == 1u, "dup pool balanced");
    }

    // Case 11: base over-reservation is never released. prefix 31 needs two
    // pages but the resource already owns three.
    {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "base alloc sync");
        const uint64_t h = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, h, 0u, 31u, kCandidates, 3u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "base prepare sync");
        check(ok[0] == 1u, "base prepare reserved");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 0u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "base commit sync");
        check(ok[0] == 1u, "base commit succeeds");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "base count sync");
        check(ok[0] == 3u, "base pages are kept even when accepted is zero");
        uint32_t owned = 0u;
        for (uint32_t p = 0u; p < kPoolPages; ++p) {
            if (pool_owner[p] == h) ++owned;
        }
        check(owned == 3u, "base owner tags stay on the kept pages");
    }

    // Case 12: two slot isolation.
    {
        reset(0u);
        uint64_t handle_b = 0u;
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "iso a alloc sync");
        const uint64_t ha = *handle;
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "iso b alloc sync");
        handle_b = *handle;
        reserve_kernel<<<1, 1, 0, stream>>>(mgr, pool, table, handle_b, 1u, 4u,
                                            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "iso b reserve sync");
        check(ok[0] == 1u, "iso: slot b pages reserved");
        uint32_t b_owner_before = 0u;
        for (uint32_t p = 0u; p < kPoolPages; ++p) {
            if (pool_owner[p] == handle_b) ++b_owner_before;
        }
        uint32_t b_blocks_before[kBlockStride];
        for (uint32_t i = 0; i < kBlockStride; ++i)
            b_blocks_before[i] = blocks[kBlockStride + i];
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, ha, 0u, kPrefix, kCandidates, 1u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "iso a prepare sync");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 2u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "iso a commit sync");
        check(ok[0] == 1u, "iso: slot a commit succeeds");
        uint32_t b_owner_after = 0u;
        for (uint32_t p = 0u; p < kPoolPages; ++p) {
            if (pool_owner[p] == handle_b) ++b_owner_after;
        }
        check(b_owner_after == b_owner_before, "iso: slot b owner unchanged");
        bool b_blocks_same = true;
        for (uint32_t i = 0; i < kBlockStride; ++i) {
            if (blocks[kBlockStride + i] != b_blocks_before[i])
                b_blocks_same = false;
        }
        check(b_blocks_same, "iso: slot b block table unchanged");
    }

    // Case 13: physical page id 0 is a valid retained page.
    {
        reset(1u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "page0 alloc sync");
        const uint64_t h = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, h, 0u, kPrefix, 8u, 1u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "page0 prepare sync");
        commit_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u, 0u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "page0 commit sync");
        check(ok[0] == 1u, "page0 commit succeeds");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "page0 count sync");
        check(ok[0] == 1u, "page0: one base page kept");
        check(blocks[0u] == 0u && pool_owner[0] == h,
              "page0 is a valid owned page, not a clear sentinel");
        balanced_kernel<<<1, 1, 0, stream>>>(
            pool, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "page0 balanced sync");
        check(ok[0] == 1u, "page0: pool balanced");
    }

    // Case 14: abort returns exactly the base page count.
    {
        reset(0u);
        alloc_kernel<<<1, 1, 0, stream>>>(mgr, static_cast<uint64_t*>(handle_d),
                                          static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "abort alloc sync");
        const uint64_t h = *handle;
        prepare_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, h, 0u, kPrefix, kCandidates, 1u,
            static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "abort prepare sync");
        abort_txn_kernel<<<1, 1, 0, stream>>>(
            slots, runtime, mgr, pool, table, 0u,
            static_cast<uint32_t*>(ok_d), static_cast<uint32_t*>(freed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "abort sync");
        check(ok[0] == 1u, "abort succeeds");
        check(*freed == 2u, "abort rolls back the speculative pages");
        page_count_kernel<<<1, 1, 0, stream>>>(mgr, h,
                                               static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "abort count sync");
        check(ok[0] == 1u, "abort leaves exactly the base page count");
        balanced_kernel<<<1, 1, 0, stream>>>(
            pool, static_cast<uint32_t*>(ok_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "abort balanced sync");
        check(ok[0] == 1u, "abort: pool balanced");
    }

    // Case 31: the persistent reserve pass reserves the whole candidate window
    // and does not overwrite the base on a repeated boundary.
    {
        void* state_h = nullptr;
        void* state_d = nullptr;
        void *s_slots_h = nullptr, *s_slots_d = nullptr;
        void *s_runtime_h = nullptr, *s_runtime_d = nullptr;
        void *s_entries_h = nullptr, *s_entries_d = nullptr;
        void *s_stack_h = nullptr, *s_stack_d = nullptr;
        void *s_count_h = nullptr, *s_count_d = nullptr;
        void *s_pstack_h = nullptr, *s_pstack_d = nullptr;
        void *s_pcount_h = nullptr, *s_pcount_d = nullptr;
        void *s_powner_h = nullptr, *s_powner_d = nullptr;
        void *s_blocks_h = nullptr, *s_blocks_d = nullptr;
        void *changed_h = nullptr, *changed_d = nullptr;
        check(map_alloc(sizeof(mcu::GpuMcuPersistentState), &state_h, &state_d),
              "state alloc");
        check(map_alloc(sizeof(mcu::GpuMcuSlotState), &s_slots_h, &s_slots_d),
              "state slots alloc");
        check(map_alloc(sizeof(mcu::GpuMcuSlotRuntimeState), &s_runtime_h,
                        &s_runtime_d),
              "state runtime alloc");
        check(map_alloc(sizeof(mcu::DeviceSequenceResourceEntry), &s_entries_h,
                        &s_entries_d),
              "state entries alloc");
        check(map_alloc(sizeof(uint32_t), &s_stack_h, &s_stack_d),
              "state stack alloc");
        check(map_alloc(sizeof(uint32_t), &s_count_h, &s_count_d),
              "state count alloc");
        check(map_alloc(kPoolPages * sizeof(uint32_t), &s_pstack_h, &s_pstack_d),
              "state pstack alloc");
        check(map_alloc(sizeof(uint32_t), &s_pcount_h, &s_pcount_d),
              "state pcount alloc");
        check(map_alloc(kPoolPages * sizeof(uint64_t), &s_powner_h, &s_powner_d),
              "state powner alloc");
        check(map_alloc(kBlockStride * sizeof(uint32_t), &s_blocks_h, &s_blocks_d),
              "state blocks alloc");
        check(map_alloc(sizeof(uint32_t), &changed_h, &changed_d),
              "state changed alloc");
        void* s_handle_h = nullptr;
        void* s_handle_d = nullptr;
        check(map_alloc(sizeof(uint64_t), &s_handle_h, &s_handle_d),
              "state handle alloc");

        mcu::DeviceSequenceResourceManagerView s_mgr{};
        s_mgr.resources = static_cast<mcu::DeviceSequenceResourceEntry*>(s_entries_d);
        s_mgr.free_stack = static_cast<uint32_t*>(s_stack_d);
        s_mgr.free_count = static_cast<uint32_t*>(s_count_d);
        s_mgr.resource_count = 1u;
        mcu::GpuMcuKvPagePoolView s_pool{};
        s_pool.free_stack = static_cast<uint32_t*>(s_pstack_d);
        s_pool.free_count = static_cast<uint32_t*>(s_pcount_d);
        s_pool.owner = static_cast<uint64_t*>(s_powner_d);
        s_pool.capacity = kPoolPages;
        mcu::GpuMcuBlockTableView s_table{};
        s_table.block_tables = static_cast<uint32_t*>(s_blocks_d);
        s_table.block_table_stride = kBlockStride;
        s_table.page_tokens = kPageTokens;
        auto* s_slots = static_cast<mcu::GpuMcuSlotState*>(s_slots_d);
        auto* s_runtime = static_cast<mcu::GpuMcuSlotRuntimeState*>(s_runtime_d);
        auto* s_state = static_cast<mcu::GpuMcuPersistentState*>(state_h);

        for (uint32_t i = 0; i < kSequences * kBlockStride; ++i) blocks[i] = 0u;
        res_init_kernel<<<1, 1, 0, stream>>>(s_mgr);
        pool_init_kernel<<<1, 1, 0, stream>>>(s_pool, 0u);
        (void)hipMemset(s_blocks_d, 0, kBlockStride * sizeof(uint32_t));
        alloc_kernel<<<1, 1, 0, stream>>>(
            s_mgr, static_cast<uint64_t*>(s_handle_d),
            static_cast<uint32_t*>(changed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "state init sync");
        const uint64_t s_handle = *static_cast<uint64_t*>(s_handle_h);

        *s_state = mcu::GpuMcuPersistentState{};
        s_state->resources_enabled = 1u;
        s_state->slot_runtime = s_runtime;
        s_state->kv_pages = s_pool;
        s_state->request_slots = s_slots;
        s_state->request_max_slots = 1u;
        s_state->sequence_resources = s_mgr;
        s_state->kv_blocks = s_table;
        s_state->scheduler_epoch = 7u;
        s_slots[0] = mcu::GpuMcuSlotState{};
        s_slots[0].phase = mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::verify);
        s_slots[0].committed_position = kPrefix;
        s_slots[0].verify_candidate_count = kCandidates;
        s_slots[0].kv_sequence_handle = s_handle;
        s_runtime[0] = mcu::GpuMcuSlotRuntimeState{};

        reserve_pass_kernel<<<1, 1, 0, stream>>>(
            static_cast<mcu::GpuMcuPersistentState*>(state_d),
            static_cast<uint32_t*>(changed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "state reserve sync");
        check(s_runtime[0].verify_txn_active == 1u,
              "state reserve begins the transaction");
        check(s_runtime[0].verify_base_page_count == 0u,
              "state reserve records the base");
        check(s_runtime[0].verify_reserved_page_count == 3u,
              "state reserve takes the whole candidate window");
        check(s_runtime[0].verify_txn_epoch == 7u,
              "state reserve stamps the transaction epoch");
        page_count_kernel<<<1, 1, 0, stream>>>(
            s_mgr, s_handle, static_cast<uint32_t*>(changed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "state count sync");
        check(*static_cast<uint32_t*>(changed_h) == 3u,
              "state reserve grew the resource to three pages");

        reserve_pass_kernel<<<1, 1, 0, stream>>>(
            static_cast<mcu::GpuMcuPersistentState*>(state_d),
            static_cast<uint32_t*>(changed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "state repeat sync");
        check(s_runtime[0].verify_base_page_count == 0u,
              "state repeat keeps the base fixed");
        check(s_runtime[0].verify_reserved_page_count == 3u,
              "state repeat does not take extra pages");
        page_count_kernel<<<1, 1, 0, stream>>>(
            s_mgr, s_handle, static_cast<uint32_t*>(changed_d));
        check(hipStreamSynchronize(stream) == hipSuccess, "state repeat count sync");
        check(*static_cast<uint32_t*>(changed_h) == 3u,
              "state repeat keeps three pages");
    }

    check(hipStreamDestroy(stream) == hipSuccess, "stream destroy");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_verify_kv_transaction: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_verify_kv_transaction: PASS\n");
    return 0;
}
