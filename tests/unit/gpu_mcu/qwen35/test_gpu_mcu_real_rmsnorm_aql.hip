#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include "unit/gpu_mcu/gpu_mcu_real_rmsnorm_fixture.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

using gpu_mcu_rmsnorm::Shape;
using gpu_mcu_rmsnorm::build_input;
using gpu_mcu_rmsnorm::build_weight;
using gpu_mcu_rmsnorm::compare_to_reference;
using gpu_mcu_rmsnorm::cpu_reference;
using gpu_mcu_rmsnorm::f32_from_bf16;
using gpu_mcu_rmsnorm::kEps;
using gpu_mcu_rmsnorm::kShapes;

bool wait_queue_drained(const mcu::GpuMcuAqlQueue& queue, uint64_t target,
                        uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (queue.host_consumed_index() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
}

bool run_shape(const Shape& shape, const mcu::GpuAqlKernelMetadata& meta,
               mcu::GpuMcuAqlQueue& queue, const mcu::GpuMcuKernargRegion& region,
               const std::vector<uint16_t>& h_in, const std::vector<uint16_t>& h_wt) {
    const uint32_t rows = 1u;
    const uint32_t features = shape.features;
    const uint32_t group = shape.group;
    const uint32_t stride = features;
    const size_t in_bytes = static_cast<size_t>(rows) * stride * sizeof(uint16_t);
    const size_t out_bytes = static_cast<size_t>(rows) * stride * sizeof(uint16_t);
    const size_t wt_bytes = static_cast<size_t>(features) * sizeof(uint16_t);
    const size_t guard_bytes = 256u * sizeof(uint16_t);

    void* in_dev = nullptr;
    void* wt_dev = nullptr;
    void* ref_dev = nullptr;
    void* aql_dev = nullptr;
    if (hipMalloc(&in_dev, in_bytes) != hipSuccess ||
        hipMalloc(&wt_dev, wt_bytes) != hipSuccess ||
        hipMalloc(&ref_dev, out_bytes + guard_bytes) != hipSuccess ||
        hipMalloc(&aql_dev, out_bytes + guard_bytes) != hipSuccess) {
        std::printf("FAIL: shape F=%u alloc\n", features);
        return false;
    }

    bool ok = true;
    std::vector<uint16_t> guard(guard_bytes / 2, 0xABABu);
    ok &= hipMemcpy(in_dev, h_in.data(), in_bytes, hipMemcpyHostToDevice) == hipSuccess;
    ok &= hipMemcpy(wt_dev, h_wt.data(), wt_bytes, hipMemcpyHostToDevice) == hipSuccess;
    ok &= hipMemset(ref_dev, 0xAB, out_bytes + guard_bytes) == hipSuccess;
    ok &= hipMemset(aql_dev, 0xAB, out_bytes + guard_bytes) == hipSuccess;
    check(ok, "shape buffers uploaded");

    const kernel::RmsNormDataType bf16 = kernel::RmsNormDataType::BF16;
    const hipError_t hr = kernel::launch_rmsnorm(
        in_dev, bf16, wt_dev, bf16, kernel::RmsNormWeightLayout::PER_FEATURE,
        kernel::RmsNormWeightMode::ONE_PLUS, ref_dev, bf16, rows, features, group,
        stride, stride, kEps, nullptr);
    check(hr == hipSuccess, "host launch_rmsnorm");
    ok &= hipDeviceSynchronize() == hipSuccess;

    const uint32_t grid = kernel::rmsnorm_bf16_pf_oneplus_grid(rows, features, group);
    const uint32_t block = kernel::rmsnorm_bf16_pf_oneplus_block_size(group);
    check(grid == rows * (features / group), "shared grid helper");

    auto* slot = region.base;
    std::memset(slot, 0, region.slot_stride);
    auto* args = reinterpret_cast<kernel::RmsNormBf16PfOnePlusArgs*>(slot);
    args->input = in_dev;
    args->weight = wt_dev;
    args->output = aql_dev;
    args->rows = rows;
    args->features = features;
    args->group_size = group;
    args->input_row_stride = stride;
    args->output_row_stride = stride;
    args->eps = kEps;

    mcu::GpuAqlDispatchDesc desc{};
    desc.kernel_object = meta.kernel_object;
    desc.global_work_items_x = grid * block;
    desc.global_work_items_y = 1u;
    desc.global_work_items_z = 1u;
    desc.workgroup_size_x = static_cast<uint16_t>(block);
    desc.workgroup_size_y = 1u;
    desc.workgroup_size_z = 1u;
    desc.private_segment_size = meta.private_segment_size;
    desc.group_segment_size = meta.static_group_segment_size;
    desc.kernarg_address = slot;

    mcu::build_aql_hidden_args(slot, sizeof(kernel::RmsNormBf16PfOnePlusArgs),
                               meta.kernarg_segment_size,
                               mcu::AqlHiddenArgsPolicy::IfFits, desc);
    check(mcu::aql_hidden_args_fits(sizeof(kernel::RmsNormBf16PfOnePlusArgs),
                                    meta.kernarg_segment_size),
          "rmsnorm segment includes hidden args");
    mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid, 1u, 1u,
                                   block);

    const mcu::GpuAqlPacketTemplate tmpl =
        mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);

    check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
    const uint64_t target = queue.host_queued_index();
    const bool drained = wait_queue_drained(queue, target, 15000);
    if (!drained) {
        std::printf("  drain stalled: write=%llu read=%llu\n",
                    static_cast<unsigned long long>(target),
                    static_cast<unsigned long long>(queue.host_consumed_index()));
        std::fflush(stdout);
    }
    check(drained, "aql queue drained");

    std::vector<uint16_t> got_flat(static_cast<size_t>(rows) * stride, 0u);
    ok &= hipMemcpy(got_flat.data(), aql_dev, out_bytes, hipMemcpyDeviceToHost) ==
          hipSuccess;
    std::vector<uint16_t> ref_flat(static_cast<size_t>(rows) * stride, 0u);
    ok &= hipMemcpy(ref_flat.data(), ref_dev, out_bytes, hipMemcpyDeviceToHost) ==
          hipSuccess;
    check(ok, "shape outputs read back");

    std::vector<uint16_t> got_guard(guard_bytes / 2, 0u);
    std::vector<uint16_t> ref_guard(guard_bytes / 2, 0u);
    ok &= hipMemcpy(got_guard.data(), static_cast<unsigned char*>(aql_dev) + out_bytes,
                    guard_bytes, hipMemcpyDeviceToHost) == hipSuccess;
    ok &= hipMemcpy(ref_guard.data(), static_cast<unsigned char*>(ref_dev) + out_bytes,
                    guard_bytes, hipMemcpyDeviceToHost) == hipSuccess;
    check(got_guard == guard, "aql output sentinel intact");
    check(ref_guard == guard, "host output sentinel intact");

    std::vector<float> ref_cpu;
    cpu_reference(h_in, h_wt, rows, features, group, stride, ref_cpu);

    std::vector<float> ref_host(ref_flat.size());
    std::vector<float> ref_aql(got_flat.size());
    for (size_t i = 0; i < ref_flat.size(); ++i) {
        ref_host[i] = f32_from_bf16(ref_flat[i]);
        ref_aql[i] = f32_from_bf16(got_flat[i]);
    }
    check(compare_to_reference(ref_host, ref_cpu, features, stride,
                               "host launcher vs cpu"),
          "host launch matches the reference policy");
    check(compare_to_reference(ref_aql, ref_cpu, features, stride, "aql vs cpu"),
          "aql launch matches the reference policy");
    size_t diffs = 0;
    size_t first = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < ref_flat.size(); ++i) {
        if (ref_flat[i] != got_flat[i]) {
            if (diffs == 0) first = i;
            ++diffs;
            const double d = std::fabs(ref_aql[i] - ref_host[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    if (diffs != 0) {
        std::printf("  bitexact diff: count=%zu/%zu first=%zu host=%.9g aql=%.9g "
                    "max_abs=%.3g\n",
                    diffs, ref_flat.size(), first, ref_host[first], ref_aql[first],
                    max_abs);
    }
    check(ref_flat == got_flat, "host aql output is bit exact with host launch");

    std::printf("  shape F=%u G=%u grid=%u block=%u bitexact=%d\n", features, group,
                grid, block, ref_flat == got_flat ? 1 : 0);

    (void)hipFree(in_dev);
    (void)hipFree(wt_dev);
    (void)hipFree(ref_dev);
    (void)hipFree(aql_dev);
    return ok;
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

    auto queue_res =
        mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                    mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
        kernel::kRmsNormBf16PfOnePlusSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: rmsnorm code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();
    std::printf(
        "rmsnorm hsaco: kernel_object=%p kernarg=%u private=%u group=%u "
        "sizeof(args)=%zu\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        meta.private_segment_size, meta.static_group_segment_size,
        sizeof(kernel::RmsNormBf16PfOnePlusArgs));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::RmsNormBf16PfOnePlusArgs),
          "kernarg segment covers the stable args");
    check(meta.static_group_segment_size >= 32u, "static group segment reported");

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), 1u);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    for (const Shape& shape : kShapes) {
        std::vector<uint16_t> h_in;
        std::vector<uint16_t> h_wt;
        build_input(1u, shape.features, shape.features, h_in);
        build_weight(shape.features, h_wt);
        if (!run_shape(shape, meta, queue, region, h_in, h_wt)) ++g_failed;
    }

    check(code.shutdown().ok(), "rmsnorm code object shutdown");
    check(queue.shutdown().ok(), "aql queue shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_rmsnorm_aql: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_rmsnorm_aql: PASS\n");
    return 0;
}
