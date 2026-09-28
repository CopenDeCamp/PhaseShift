#include <phaseshift/runtime/gpu_mcu/kv_page_allocator.h>

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

constexpr uint32_t kPages = 16u;

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

__global__ void init_kernel(mcu::GpuMcuKvPagePoolView pool) {
    for (uint32_t i = 0; i < pool.capacity; ++i) {
        pool.free_stack[i] = i;
        pool.owner[i] = 0u;
    }
    *pool.free_count = pool.capacity;
}

__global__ void alloc_kernel(mcu::GpuMcuKvPagePoolView pool, uint32_t slot,
                             uint32_t generation, uint32_t count,
                             uint32_t* first, uint32_t* ok) {
    ok[0] = mcu::gpu_mcu_kv_page_alloc(pool, slot, generation, count, first)
                ? 1u
                : 0u;
}

__global__ void free_kernel(mcu::GpuMcuKvPagePoolView pool, uint32_t slot,
                            uint32_t generation, uint32_t* freed,
                            uint32_t* balanced) {
    freed[0] = 0u;
    (void)mcu::gpu_mcu_kv_page_free_owned(pool, slot, generation, freed);
    balanced[0] = mcu::gpu_mcu_kv_page_pool_balanced(pool) ? 1u : 0u;
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

    void* stack_host = nullptr;
    void* stack_dev = nullptr;
    void* owner_host = nullptr;
    void* owner_dev = nullptr;
    void* count_host = nullptr;
    void* count_dev = nullptr;
    void* first_host = nullptr;
    void* first_dev = nullptr;
    void* ok_host = nullptr;
    void* ok_dev = nullptr;
    void* freed_host = nullptr;
    void* freed_dev = nullptr;
    void* balanced_host = nullptr;
    void* balanced_dev = nullptr;
    check(map_alloc(kPages * sizeof(uint32_t), &stack_host, &stack_dev),
          "free stack alloc");
    check(map_alloc(kPages * sizeof(uint64_t), &owner_host, &owner_dev),
          "owner alloc");
    check(map_alloc(sizeof(uint32_t), &count_host, &count_dev),
          "free count alloc");
    check(map_alloc(sizeof(uint32_t), &first_host, &first_dev), "first alloc");
    check(map_alloc(sizeof(uint32_t), &ok_host, &ok_dev), "ok alloc");
    check(map_alloc(sizeof(uint32_t), &freed_host, &freed_dev), "freed alloc");
    check(map_alloc(sizeof(uint32_t), &balanced_host, &balanced_dev),
          "balanced alloc");

    mcu::GpuMcuKvPagePoolView pool{};
    pool.free_stack = static_cast<uint32_t*>(stack_dev);
    pool.free_count = static_cast<uint32_t*>(count_dev);
    pool.owner = static_cast<uint64_t*>(owner_dev);
    pool.capacity = kPages;

    auto* count = static_cast<uint32_t*>(count_host);
    auto* first = static_cast<uint32_t*>(first_host);
    auto* ok = static_cast<uint32_t*>(ok_host);
    auto* freed = static_cast<uint32_t*>(freed_host);
    auto* balanced = static_cast<uint32_t*>(balanced_host);
    auto* owners = static_cast<uint64_t*>(owner_host);
    auto* first_dev_u32 = static_cast<uint32_t*>(first_dev);
    auto* ok_dev_u32 = static_cast<uint32_t*>(ok_dev);
    auto* freed_dev_u32 = static_cast<uint32_t*>(freed_dev);
    auto* balanced_dev_u32 = static_cast<uint32_t*>(balanced_dev);

    init_kernel<<<1, 1, 0, stream>>>(pool);
    check(hipStreamSynchronize(stream) == hipSuccess, "init sync");
    check(*count == kPages, "the pool starts completely free");
    check(mcu::gpu_mcu_kv_page_owner_slot(0u) == 0xffffffffu,
          "owner zero decodes as no slot");

    alloc_kernel<<<1, 1, 0, stream>>>(pool, 0u, 1u, 4u, first_dev_u32, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "alloc sync");
    check(ok[0] == 1u, "four pages are allocated");
    check(*count == kPages - 4u, "the free count drops by four");
    uint32_t tagged = 0u;
    for (uint32_t i = 0; i < kPages; ++i) {
        if (owners[i] == mcu::gpu_mcu_kv_page_owner(0u, 1u)) ++tagged;
    }
    check(tagged == 4u, "four pages carry the request tag");

    alloc_kernel<<<1, 1, 0, stream>>>(pool, 1u, 7u, kPages, first_dev_u32, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "over alloc sync");
    check(ok[0] == 0u, "an allocation beyond the free pages fails");
    check(*count == kPages - 4u, "a failed allocation changes nothing");

    free_kernel<<<1, 1, 0, stream>>>(pool, 0u, 2u, freed_dev_u32, balanced_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "stale free sync");
    check(freed[0] == 0u, "a stale generation frees nothing");
    check(*count == kPages - 4u, "a stale free leaves the pool alone");
    check(balanced[0] == 1u, "the pool stays balanced");

    free_kernel<<<1, 1, 0, stream>>>(pool, 0u, 1u, freed_dev_u32, balanced_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "free sync");
    check(freed[0] == 4u, "the owning request frees its four pages");
    check(*count == kPages, "the free count is restored");
    check(balanced[0] == 1u, "the pool stays balanced");

    free_kernel<<<1, 1, 0, stream>>>(pool, 0u, 1u, freed_dev_u32, balanced_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "dup free sync");
    check(freed[0] == 0u, "a duplicate free frees nothing");
    check(*count == kPages, "a duplicate free leaves the pool alone");

    alloc_kernel<<<1, 1, 0, stream>>>(pool, 0u, 3u, 8u, first_dev_u32, ok_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "reuse sync");
    check(ok[0] == 1u, "the next generation reuses the released pages");
    check(*count == kPages - 8u, "the free count drops by eight");
    free_kernel<<<1, 1, 0, stream>>>(pool, 0u, 3u, freed_dev_u32, balanced_dev_u32);
    check(hipStreamSynchronize(stream) == hipSuccess, "reuse free sync");
    check(freed[0] == 8u && *count == kPages, "the reuse is fully released");
    check(balanced[0] == 1u, "the pool stays balanced after reuse");

    check(hipStreamDestroy(stream) == hipSuccess, "stream destroy");
    check(hipHostFree(stack_host) == hipSuccess, "stack free");
    check(hipHostFree(owner_host) == hipSuccess, "owner free");
    check(hipHostFree(count_host) == hipSuccess, "count free");
    check(hipHostFree(first_host) == hipSuccess, "first free");
    check(hipHostFree(ok_host) == hipSuccess, "ok free");
    check(hipHostFree(freed_host) == hipSuccess, "freed free");
    check(hipHostFree(balanced_host) == hipSuccess, "balanced free");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_kv_page_allocator: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_kv_page_allocator: PASS\n");
    return 0;
}
