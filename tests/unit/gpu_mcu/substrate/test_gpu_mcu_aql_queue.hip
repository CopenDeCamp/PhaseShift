#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace aql = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    {
        auto queue_res = aql::GpuMcuAqlQueue::create(
            0, aql::AqlQueueOwner::GpuMcu, aql::kAqlDefaultQueueSize);
        check(queue_res.ok(), "queue created");
        if (!queue_res.ok()) {
            std::printf("create error: %s\n", queue_res.status().message().c_str());
            return 1;
        }
        auto queue = queue_res.release();

        check(queue.valid(), "queue valid");
        check(queue.size() == aql::kAqlDefaultQueueSize, "queue size");
        check(queue.owner() == aql::AqlQueueOwner::GpuMcu, "queue owner");
        check(queue.hsa_agent_handle() != 0, "hsa agent handle");
        check(std::string(queue.arch_name()) == "gfx1201", "arch name gfx1201");

        const auto& view = queue.device_queue_view();
        check(view.queue_ptr != nullptr, "queue ptr");
        check(view.ring_base != nullptr, "ring base");
        check((reinterpret_cast<uintptr_t>(view.ring_base) & 63u) == 0u, "ring 64B aligned");
        check(view.doorbell_handle != 0, "doorbell handle");
        check(view.size == aql::kAqlDefaultQueueSize, "view size");
        check(view.write_index != nullptr && view.read_index != nullptr, "queue indices");
        check(view.owner == aql::AqlQueueOwner::GpuMcu, "view owner");

        const std::size_t slot_bytes = aql::aql_kernarg_slot_stride(32);
        const uint32_t slot_count = 4;
        auto region_res = queue.allocate_kernarg(slot_bytes, slot_count);
        check(region_res.ok(), "kernarg region allocated");
        if (region_res.ok()) {
            const aql::GpuMcuKernargRegion& region = region_res.value();
            check(region.base != nullptr, "kernarg base");
            check(region.requested_slot_bytes == slot_bytes, "kernarg requested bytes");
            check(region.granule > 0, "kernarg granule reported");
            check(region.slot_count == slot_count, "kernarg slot count");
            check(region.slot_stride >= slot_bytes, "kernarg slot stride covers request");
            check((region.slot_stride % 16u) == 0u, "kernarg slot stride 16B aligned");
            check(region.actual_size >= region.slot_stride * slot_count,
                  "kernarg actual size covers all slots");
            const uintptr_t lo = reinterpret_cast<uintptr_t>(region.base);
            const uintptr_t hi = lo + region.actual_size;
            bool in_range = true;
            bool distinct = true;
            for (uint32_t i = 0; i < slot_count; ++i) {
                const uintptr_t a =
                    reinterpret_cast<uintptr_t>(region.slot(i));
                in_range &= (a >= lo && a + slot_bytes <= hi);
                for (uint32_t j = i + 1; j < slot_count; ++j) {
                    distinct &= region.slot(i) != region.slot(j);
                }
            }
            check(in_range, "all kernarg slots inside the allocation");
            check(distinct, "kernarg slots are distinct");
        }

        auto bad_zero = queue.allocate_kernarg(0, 1);
        check(!bad_zero.ok(), "zero-size kernarg request rejected");
        auto bad_count = queue.allocate_kernarg(16, 0);
        check(!bad_count.ok(), "zero-count kernarg request rejected");

        check(queue.shutdown().ok(), "queue shutdown");
    }

    {
        std::vector<uint32_t> mask{0xFFFFFFFCu, 0xFFFFFFFFu};
        auto queue_res = aql::GpuMcuAqlQueue::create_with_mask(
            0, aql::AqlQueueOwner::GpuMcu, aql::kAqlDefaultQueueSize,
            aql::AqlCuMaskView{mask.data(), 64u});
        check(queue_res.ok(), "masked queue created");
        if (!queue_res.ok()) {
            std::printf("masked queue error: %s\n",
                        queue_res.status().message().c_str());
        }
        if (queue_res.ok()) {
            auto queue = queue_res.release();
            auto requested = queue.requested_mask();
            check(requested.ok(), "requested mask readable");
            if (requested.ok()) {
                check(requested.value() == mask, "requested mask preserved");
            }
            auto readback = queue.readback_mask();
            if (readback.ok() && !readback.value().empty()) {
                std::printf("HSA queue mask readback[0]=0x%08x requested[0]=0x%08x\n",
                            readback.value()[0], mask[0]);
            }
            check(queue.shutdown().ok(), "masked queue shutdown");
        }
    }

    {
        std::vector<uint32_t> unpaired{0xFFFFFFFEu, 0xFFFFFFFFu};
        auto queue_res = aql::GpuMcuAqlQueue::create_with_mask(
            0, aql::AqlQueueOwner::GpuMcu, aql::kAqlDefaultQueueSize,
            aql::AqlCuMaskView{unpaired.data(), 64u});
        check(!queue_res.ok(), "unpaired CU mask rejected");
    }

    {
        std::vector<uint32_t> bad(1, 0xFFFFFFFFu);
        auto queue_res = aql::GpuMcuAqlQueue::create_with_mask(
            0, aql::AqlQueueOwner::GpuMcu, aql::kAqlDefaultQueueSize,
            aql::AqlCuMaskView{bad.data(), 8u});
        check(!queue_res.ok(), "non-multiple-of-32 mask bit count rejected");
    }

    {
        auto queue_res = aql::GpuMcuAqlQueue::create(0, aql::AqlQueueOwner::GpuMcu);
        check(queue_res.ok(), "queue recreate");
        if (queue_res.ok()) {
            auto queue = queue_res.release();
            check(queue.shutdown().ok(), "recreate shutdown");
        }
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_aql_queue: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_aql_queue: PASS\n");
    return 0;
}
