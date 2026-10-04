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

using fixture::compare_to_reference;
using fixture::cpu_reference;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

std::vector<mcu::McuPlanNode> make_rmsnorm_plan() {
    std::vector<mcu::McuPlanNode> nodes(4);
    nodes[0].variant_id = 1;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
    nodes[0].completion_slot = 0;
    nodes[0].input_slot = mcu::kMcuNoInput;
    nodes[0].invocation_index = 0;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1;

    nodes[1].variant_id = 0;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[1].completion_slot = 0;
    nodes[1].element_count = 0;
    nodes[1].input_slot = mcu::kMcuNoInput;
    nodes[1].work = 0;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2;

    nodes[2].completion_slot = 0;
    nodes[2].flags = mcu::kMcuNodeWait;
    nodes[2].next = 3;

    nodes[3].flags = mcu::kMcuNodeEnd;
    return nodes;
}

bool run_shape(const fixture::Shape& shape) {
    const uint32_t rows = 1u;
    const uint32_t features = shape.features;
    const uint32_t group = shape.group;
    const uint32_t stride = features;
    const size_t out_bytes = static_cast<size_t>(rows) * stride * sizeof(uint16_t);
    const size_t in_bytes = out_bytes;
    const size_t wt_bytes = static_cast<size_t>(features) * sizeof(uint16_t);
    const size_t guard_bytes = 256u * sizeof(uint16_t);

    std::vector<uint16_t> h_in;
    std::vector<uint16_t> h_wt;
    fixture::build_input(rows, features, stride, h_in);
    fixture::build_weight(features, h_wt);

    void* in_dev = nullptr;
    void* wt_dev = nullptr;
    void* ref_dev = nullptr;
    void* out_dev = nullptr;
    if (hipMalloc(&in_dev, in_bytes) != hipSuccess ||
        hipMalloc(&wt_dev, wt_bytes) != hipSuccess ||
        hipMalloc(&ref_dev, out_bytes + guard_bytes) != hipSuccess ||
        hipMalloc(&out_dev, out_bytes + guard_bytes) != hipSuccess) {
        std::printf("FAIL: F=%u alloc\n", features);
        return false;
    }
    std::vector<uint16_t> guard(guard_bytes / 2, 0xABABu);
    (void)hipMemcpy(in_dev, h_in.data(), in_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(wt_dev, h_wt.data(), wt_bytes, hipMemcpyHostToDevice);
    (void)hipMemset(ref_dev, 0xAB, out_bytes + guard_bytes);
    (void)hipMemset(out_dev, 0xAB, out_bytes + guard_bytes);

    const kernel::RmsNormDataType bf16 = kernel::RmsNormDataType::BF16;
    const hipError_t hr = kernel::launch_rmsnorm(
        in_dev, bf16, wt_dev, bf16, kernel::RmsNormWeightLayout::PER_FEATURE,
        kernel::RmsNormWeightMode::ONE_PLUS, ref_dev, bf16, rows, features, group,
        stride, stride, fixture::kEps, nullptr);
    check(hr == hipSuccess, "host launch_rmsnorm reference");
    (void)hipDeviceSynchronize();

    const uint32_t grid =
        kernel::rmsnorm_bf16_pf_oneplus_grid(rows, features, group);
    const uint32_t block =
        kernel::rmsnorm_bf16_pf_oneplus_block_size(group);

    Rig rig;
    if (!rig.open(0, 4u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("test_gpu_mcu_real_rmsnorm_fsm: setup failed\n");
        return false;
    }

    mcu::GpuAqlKernelMetadata rmsnorm_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                             kernel::kRmsNormBf16PfOnePlusSymbol,
                             rmsnorm_meta)) {
        return false;
    }
    check(rmsnorm_meta.kernel_object != 0, "rmsnorm kernel_object resolved");

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u, mcu::kMcuSystemAqlPolicy),
        mcu::make_kernel_variant(1u, rmsnorm_meta,
                                 mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus, block,
                                 grid, mcu::kMcuAgentAqlPolicy)};

    const std::vector<mcu::McuPlanNode> nodes = make_rmsnorm_plan();
    if (!rig.upload(nodes, variants)) return false;

    mcu::McuRmsNormInvocation invocation{};
    invocation.input = reinterpret_cast<uint64_t>(in_dev);
    invocation.weight = reinterpret_cast<uint64_t>(wt_dev);
    invocation.output = reinterpret_cast<uint64_t>(out_dev);
    invocation.rows = rows;
    invocation.features = features;
    invocation.group_size = group;
    invocation.input_row_stride = stride;
    invocation.output_row_stride = stride;
    invocation.eps = fixture::kEps;
    if (!rig.set_invocations(&invocation, 1u)) return false;

    rig.poison_kernarg_args();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x524d5346u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_real_rmsnorm_fsm: start failed\n");
        return false;
    }

    constexpr uint64_t kPlans = 1000u;
    for (uint64_t i = 1; i <= kPlans; ++i) {
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
    check(rig.fsm.dispatches() == kPlans * 2u, "mcu dispatch count");
    check(rig.fsm.completions_observed() == kPlans, "marker completions observed");
    const bool drained = rig.drain_queue(30000);
    std::fprintf(stderr, "[diag] F=%u plans done drain_ok=%d\n", features,
                 static_cast<int>(drained));
    std::fflush(stderr);
    check(drained, "queue drained");
    check(rig.fsm.max_ahead() <= 4u, "queue occupancy respects ahead depth");

    std::vector<uint16_t> ref_flat(out_bytes / 2, 0u);
    std::vector<uint16_t> got_flat(out_bytes / 2, 0u);
    std::vector<uint16_t> ref_guard(guard_bytes / 2, 0u);
    std::vector<uint16_t> got_guard(guard_bytes / 2, 0u);
    rig.device_to_host(ref_flat.data(), ref_dev, out_bytes);
    rig.device_to_host(got_flat.data(), out_dev, out_bytes);
    rig.device_to_host(ref_guard.data(),
                       static_cast<const unsigned char*>(ref_dev) + out_bytes,
                       guard_bytes);
    rig.device_to_host(got_guard.data(),
                       static_cast<const unsigned char*>(out_dev) + out_bytes,
                       guard_bytes);

    std::vector<float> ref_cpu;
    fixture::cpu_reference(h_in, h_wt, rows, features, group, stride, ref_cpu);
    std::vector<float> ref_host(ref_flat.size());
    std::vector<float> ref_mcu(got_flat.size());
    for (size_t i = 0; i < ref_flat.size(); ++i) {
        ref_host[i] = fixture::f32_from_bf16(ref_flat[i]);
        ref_mcu[i] = fixture::f32_from_bf16(got_flat[i]);
    }

    check(compare_to_reference(ref_host, ref_cpu, features, stride, "host vs cpu"),
          "host launcher matches the reference policy");
    check(compare_to_reference(ref_mcu, ref_cpu, features, stride, "mcu vs cpu"),
          "mcu chain matches the reference policy");
    size_t diffs = 0;
    size_t first = 0;
    check(gpu_mcu_rmsnorm::bit_exact(ref_flat, got_flat, &diffs, &first),
          "mcu output is bit exact with host launch");
    if (diffs != 0) {
        std::printf("  bitexact diff: count=%zu/%zu first=%zu host=%.9g mcu=%.9g\n",
                    diffs, ref_flat.size(), first, ref_host[first], ref_mcu[first]);
    }
    check(got_guard == guard, "mcu output sentinel intact");
    check(ref_guard == guard, "host output sentinel intact");

    std::printf(
        "  shape F=%u G=%u grid=%u block=%u plans=%llu ahead=%llu refill=%llu\n",
        features, group, grid, block,
        static_cast<unsigned long long>(rig.fsm.plans_started()),
        static_cast<unsigned long long>(rig.fsm.max_ahead()),
        static_cast<unsigned long long>(rig.fsm.refill_count()));

    const bool closed = rig.close();
    check(closed, "rig teardown");
    (void)hipFree(in_dev);
    (void)hipFree(wt_dev);
    (void)hipFree(ref_dev);
    (void)hipFree(out_dev);
    return closed;
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

    for (const fixture::Shape& shape : fixture::kShapes) {
        if (!run_shape(shape)) ++g_failed;
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_rmsnorm_fsm: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_rmsnorm_fsm: PASS\n");
    return 0;
}
