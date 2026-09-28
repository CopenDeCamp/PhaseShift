#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_rmsnorm_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace fixture = gpu_mcu_rmsnorm;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kData = 16u;

std::vector<mcu::McuPlanNode> make_rmsnorm_feed_plan() {
    std::vector<mcu::McuPlanNode> nodes(kData + 3u);
    for (uint32_t i = 0; i < kData; ++i) {
        nodes[i].variant_id = 1;
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
        nodes[i].completion_slot = 0;
        nodes[i].input_slot = mcu::kMcuNoInput;
        nodes[i].invocation_index = i;
        nodes[i].flags = mcu::kMcuNodeDispatch;
        nodes[i].next = static_cast<uint16_t>(i + 1u);
    }
    nodes[kData].variant_id = 0;
    nodes[kData].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[kData].completion_slot = 0;
    nodes[kData].input_slot = mcu::kMcuNoInput;
    nodes[kData].flags = mcu::kMcuNodeDispatch;
    nodes[kData].next = static_cast<uint16_t>(kData + 1u);
    nodes[kData + 1u].completion_slot = 0;
    nodes[kData + 1u].flags = mcu::kMcuNodeWait;
    nodes[kData + 1u].next = static_cast<uint16_t>(kData + 2u);
    nodes[kData + 2u].flags = mcu::kMcuNodeEnd;
    return nodes;
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

    const uint32_t rows = 1u;
    const uint32_t features = 5120u;
    const uint32_t group = 5120u;
    const uint32_t stride = features;
    const size_t row_bytes = static_cast<size_t>(rows) * stride * sizeof(uint16_t);
    const size_t wt_bytes = static_cast<size_t>(features) * sizeof(uint16_t);

    std::vector<uint16_t> h_in;
    std::vector<uint16_t> h_wt;
    fixture::build_input(rows, features, stride, h_in);
    fixture::build_weight(features, h_wt);

    void* mcu_a = nullptr;
    void* mcu_b = nullptr;
    void* host_a = nullptr;
    void* host_b = nullptr;
    void* wt_dev = nullptr;
    if (hipMalloc(&mcu_a, row_bytes) != hipSuccess ||
        hipMalloc(&mcu_b, row_bytes) != hipSuccess ||
        hipMalloc(&host_a, row_bytes) != hipSuccess ||
        hipMalloc(&host_b, row_bytes) != hipSuccess ||
        hipMalloc(&wt_dev, wt_bytes) != hipSuccess) {
        std::printf("FAIL: buffers\n");
        return 1;
    }
    (void)hipMemcpy(mcu_a, h_in.data(), row_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(host_a, h_in.data(), row_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(wt_dev, h_wt.data(), wt_bytes, hipMemcpyHostToDevice);

    const kernel::RmsNormDataType bf16 = kernel::RmsNormDataType::BF16;
    for (uint32_t i = 0; i < kData; ++i) {
        void* in = (i % 2u == 0u) ? host_a : host_b;
        void* out = (i % 2u == 0u) ? host_b : host_a;
        const hipError_t hr = kernel::launch_rmsnorm(
            in, bf16, wt_dev, bf16, kernel::RmsNormWeightLayout::PER_FEATURE,
            kernel::RmsNormWeightMode::ONE_PLUS, out, bf16, rows, features, group,
            stride, stride, fixture::kEps, nullptr);
        check(hr == hipSuccess, "host launch_rmsnorm feed reference");
    }
    (void)hipDeviceSynchronize();

    const uint32_t grid =
        kernel::rmsnorm_bf16_pf_oneplus_grid(rows, features, group);
    const uint32_t block = kernel::rmsnorm_bf16_pf_oneplus_block_size(group);

    Rig rig;
    if (!rig.open(0, 32u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("test_gpu_mcu_real_rmsnorm_feed: setup failed\n");
        return 1;
    }

    mcu::GpuAqlKernelMetadata rmsnorm_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                             kernel::kRmsNormBf16PfOnePlusSymbol, rmsnorm_meta)) {
        return 1;
    }
    check(rmsnorm_meta.kernel_object != 0, "rmsnorm kernel_object resolved");

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u, mcu::kMcuSystemAqlPolicy),
        mcu::make_kernel_variant(1u, rmsnorm_meta,
                                 mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus, block,
                                 grid, mcu::kMcuAgentAqlPolicy)};

    const std::vector<mcu::McuPlanNode> nodes = make_rmsnorm_feed_plan();
    if (!rig.upload(nodes, variants)) return 1;

    std::vector<mcu::McuRmsNormInvocation> invocations(kData);
    for (uint32_t i = 0; i < kData; ++i) {
        invocations[i].input = reinterpret_cast<uint64_t>(
            (i % 2u == 0u) ? mcu_a : mcu_b);
        invocations[i].weight = reinterpret_cast<uint64_t>(wt_dev);
        invocations[i].output = reinterpret_cast<uint64_t>(
            (i % 2u == 0u) ? mcu_b : mcu_a);
        invocations[i].rows = rows;
        invocations[i].features = features;
        invocations[i].group_size = group;
        invocations[i].input_row_stride = stride;
        invocations[i].output_row_stride = stride;
        invocations[i].eps = fixture::kEps;
    }
    if (!rig.set_invocations(invocations.data(), kData)) return 1;

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x52534644u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_real_rmsnorm_feed: start failed\n");
        return 1;
    }

    constexpr uint64_t kPlans = 1000u;

    check(rig.fsm.request_run().ok(), "first plan requested");
    check(rig.fsm.wait_plans(1u, 5000).ok(), "first plan completed");
    check(rig.drain_queue(30000), "queue drained after first plan");

    std::vector<uint16_t> got(row_bytes / 2, 0u);
    std::vector<uint16_t> ref(row_bytes / 2, 0u);
    rig.device_to_host(got.data(), mcu_a, row_bytes);
    rig.device_to_host(ref.data(), host_a, row_bytes);

    size_t diffs = 0;
    size_t first = 0;
    const bool exact = fixture::bit_exact(ref, got, &diffs, &first);
    if (!exact) {
        std::printf("  final diff: count=%zu first=%zu host=%.9g mcu=%.9g\n",
                    diffs, first, fixture::f32_from_bf16(ref[first]),
                    fixture::f32_from_bf16(got[first]));
    }
    check(exact, "repeated real rmsnorm feed is bit exact with host reference");

    for (uint64_t i = 2; i <= kPlans; ++i) {
        if (!rig.fsm.request_run().ok()) {
            check(false, "plan requested");
            break;
        }
        auto step = rig.fsm.wait_plans(i, 5000);
        if (!step.ok()) {
            std::printf("  wait[%llu]: %s fault=%u pc=%u\n",
                        static_cast<unsigned long long>(i),
                        step.message().c_str(),
                        static_cast<unsigned>(rig.fsm.fault_code()),
                        static_cast<unsigned>(rig.fsm.fault_pc()));
            check(false, "mcu plan completed");
            break;
        }
    }
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == kPlans * (kData + 1u), "mcu dispatch count");
    check(rig.fsm.completions_observed() == kPlans, "marker completions observed");
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.max_ahead() <= 4u, "queue occupancy respects ahead depth");
    check(rig.fsm.queue_empty_count() <= kPlans,
          "queue stayed fed inside each plan");

    std::printf(
        "  F=%u G=%u grid=%u block=%u plans=%llu ahead=%llu refill=%llu "
        "empty=%llu doorbells=%llu\n",
        features, group, grid, block,
        static_cast<unsigned long long>(rig.fsm.plans_started()),
        static_cast<unsigned long long>(rig.fsm.max_ahead()),
        static_cast<unsigned long long>(rig.fsm.refill_count()),
        static_cast<unsigned long long>(rig.fsm.queue_empty_count()),
        static_cast<unsigned long long>(rig.fsm.doorbell_count()));

    check(rig.close(), "rig teardown");
    (void)hipFree(mcu_a);
    (void)hipFree(mcu_b);
    (void)hipFree(host_a);
    (void)hipFree(host_b);
    (void)hipFree(wt_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_rmsnorm_feed: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_rmsnorm_feed: PASS\n");
    return 0;
}
