#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

bool all_bytes(const void* base, std::size_t offset, std::size_t count,
               uint8_t value) {
    const auto* p = static_cast<const uint8_t*>(base) + offset;
    for (std::size_t i = 0; i < count; ++i) {
        if (p[i] != value) return false;
    }
    return true;
}

bool any_non_zero(const void* base, std::size_t offset, std::size_t count) {
    const auto* p = static_cast<const uint8_t*>(base) + offset;
    for (std::size_t i = 0; i < count; ++i) {
        if (p[i] != 0u) return true;
    }
    return false;
}

void run_case(mcu::GpuMcuAqlQueue& queue, std::size_t segment,
              const char* label) {
    const std::size_t slot_bytes = mcu::aql_kernarg_slot_stride(segment);
    auto region_res = queue.allocate_kernarg(slot_bytes, 2u);
    if (!region_res.ok()) {
        std::printf("FAIL: %s region: %s\n", label,
                    region_res.status().message().c_str());
        ++g_failed;
        return;
    }
    mcu::GpuMcuKernargRegion region = region_res.release();
    check(region.slot_stride == slot_bytes, "slot stride matches the request");

    std::memset(region.base, 0x00, region.actual_size);
    std::memset(region.slot(1), 0xCD, slot_bytes);

    const std::size_t explicit_bytes = 48u;
    mcu::GpuAqlDispatchDesc desc{};
    desc.global_work_items_x = 2048;
    desc.global_work_items_y = 1;
    desc.global_work_items_z = 1;
    desc.workgroup_size_x = 32;
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;

    const bool fits =
        mcu::aql_hidden_args_fits(explicit_bytes, segment);
    const bool wrote = mcu::build_aql_hidden_args(
        region.base, explicit_bytes, segment, mcu::AqlHiddenArgsPolicy::IfFits,
        desc);
    check(wrote == fits, "IfFits reports exactly the fitted case");
    if (fits) {
        check(any_non_zero(region.base,
                           mcu::aql_hidden_args_offset(explicit_bytes),
                           mcu::kAqlHiddenArgsBytes),
              "hidden args written when the segment fits");
    } else {
        check(all_bytes(region.base, explicit_bytes,
                        mcu::aql_kernarg_slot_stride(segment) - explicit_bytes,
                        0u),
              "no write beyond the declared segment");
    }

    const bool required_ok = mcu::build_aql_hidden_args(
        region.base, explicit_bytes, segment,
        mcu::AqlHiddenArgsPolicy::Required, desc);
    check(required_ok == fits, "Required fails closed when it does not fit");

    mcu::build_aql_launch_metadata(region.base, segment, 64u, 1u, 1u, 32u);
    check(all_bytes(region.slot(1), 0, slot_bytes, 0xCD),
          "neighbour slot stays byte exact");
    std::printf("  %-22s segment=%zu slot=%zu hidden_end=%zu fits=%d wrote=%d\n",
                label, segment, slot_bytes,
                mcu::aql_hidden_args_end(explicit_bytes), fits ? 1 : 0,
                wrote ? 1 : 0);
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

    auto queue_res = mcu::GpuMcuAqlQueue::create(
        0, mcu::AqlQueueOwner::Host, mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    check(mcu::aql_hidden_args_end(48u) == 72u, "hidden args end for 48 is 72");
    run_case(queue, 48u, "psq4-like segment 48");
    run_case(queue, 304u, "rms-like segment 304");
    run_case(queue, 80u, "segment 80 exactly fits");

    check(queue.shutdown().ok(), "queue shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_kernarg_hidden_bounds: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_kernarg_hidden_bounds: PASS\n");
    return 0;
}
