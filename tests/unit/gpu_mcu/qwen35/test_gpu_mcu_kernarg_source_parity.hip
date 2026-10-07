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

constexpr uint64_t kPlans = 200u;

struct RunResult {
    bool ok = false;
    uint32_t fault = 0u;
    uint64_t dispatches = 0u;
    uint64_t completions = 0u;
    bool drained = false;
    std::vector<uint16_t> output;
    std::vector<uint16_t> guard;
    std::vector<uint16_t> kernarg_slot;
};

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
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2;

    nodes[2].completion_slot = 0;
    nodes[2].flags = mcu::kMcuNodeWait;
    nodes[2].next = 3;

    nodes[3].flags = mcu::kMcuNodeEnd;
    return nodes;
}

bool run_mode(const fixture::Shape& shape, bool generic, void* in_dev,
              void* wt_dev, void* out_dev, RunResult& out) {
    const uint32_t rows = 1u;
    const uint32_t features = shape.features;
    const uint32_t group = shape.group;
    const uint32_t stride = features;
    const size_t out_bytes = static_cast<size_t>(rows) * stride * sizeof(uint16_t);
    const size_t guard_bytes = 256u * sizeof(uint16_t);

    Rig rig;
    if (!rig.open(0, 4u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("FAIL: %s rig open\n", generic ? "generic" : "legacy");
        return false;
    }

    mcu::GpuAqlKernelMetadata rmsnorm_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                             kernel::kRmsNormBf16PfOnePlusSymbol,
                             rmsnorm_meta)) {
        (void)rig.close();
        return false;
    }

    const uint32_t grid =
        kernel::rmsnorm_bf16_pf_oneplus_grid(rows, features, group);
    const uint32_t block = kernel::rmsnorm_bf16_pf_oneplus_block_size(group);

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u, mcu::kMcuSystemAqlPolicy),
        mcu::make_kernel_variant(1u, rmsnorm_meta,
                                 mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus,
                                 block, grid, mcu::kMcuAgentAqlPolicy)};

    const std::vector<mcu::McuPlanNode> nodes = make_rmsnorm_plan();
    if (!rig.upload(nodes, variants)) {
        (void)rig.close();
        return false;
    }

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
    if (!rig.set_invocations(&invocation, 1u)) {
        (void)rig.close();
        return false;
    }

    std::vector<mcu::McuKernargSourceDesc> sources;
    if (generic) {
        sources.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].kernarg_recipe == mcu::kMcuKernargRecipeProbe) {
                sources[i].flags = mcu::kMcuKernargSourceSupervisorProbe;
                sources[i].explicit_args_bytes =
                    static_cast<uint32_t>(sizeof(mcu::GpuMcuFsmWorkerArgs));
            }
        }
        sources[0].source = reinterpret_cast<uint64_t>(rig.invocations);
        sources[0].explicit_args_bytes =
            static_cast<uint32_t>(sizeof(mcu::McuRmsNormInvocation));
        sources[0].flags = mcu::kMcuKernargSourcePrepared;
    }
    if (!rig.set_kernarg_sources(sources)) {
        (void)rig.close();
        return false;
    }

    rig.poison_kernarg_args();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = generic ? 0x47454e43u : 0x4c454741u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: %s rig start\n", generic ? "generic" : "legacy");
        (void)rig.close();
        return false;
    }

    bool ran = true;
    for (uint64_t i = 1; i <= kPlans && ran; ++i) {
        if (!rig.fsm.request_run().ok()) {
            check(false, "plan requested");
            ran = false;
            break;
        }
        const auto step = rig.fsm.wait_plans(i, 5000);
        if (!step.ok()) {
            std::printf("  %s wait[%llu]: %s fault=%u pc=%u\n",
                        generic ? "generic" : "legacy",
                        static_cast<unsigned long long>(i),
                        step.message().c_str(),
                        static_cast<unsigned>(rig.fsm.fault_code()),
                        static_cast<unsigned>(rig.fsm.fault_pc()));
            check(false, "mcu plan completed");
            ran = false;
        }
    }

    out.fault = static_cast<uint32_t>(rig.fsm.fault_code());
    out.dispatches = rig.fsm.dispatches();
    out.completions = rig.fsm.completions_observed();
    out.drained = rig.drain_queue(30000);
    out.output.assign(out_bytes / 2u, 0u);
    out.guard.assign(guard_bytes / 2u, 0u);
    out.kernarg_slot.assign(rig.region.slot_stride, 0u);
    rig.device_to_host(out.output.data(), out_dev, out_bytes);
    rig.device_to_host(out.guard.data(),
                       static_cast<const unsigned char*>(out_dev) + out_bytes,
                       guard_bytes);
    rig.device_to_host(out.kernarg_slot.data(), rig.region.slot(0),
                       rig.region.slot_stride);
    out.ok = ran;
    return rig.close();
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
        const uint32_t rows = 1u;
        const uint32_t features = shape.features;
        const uint32_t group = shape.group;
        const size_t out_bytes =
            static_cast<size_t>(rows) * features * sizeof(uint16_t);
        const size_t wt_bytes = static_cast<size_t>(features) * sizeof(uint16_t);
        const size_t guard_bytes = 256u * sizeof(uint16_t);

        std::vector<uint16_t> h_in;
        std::vector<uint16_t> h_wt;
        fixture::build_input(rows, features, features, h_in);
        fixture::build_weight(features, h_wt);

        void* in_dev = nullptr;
        void* wt_dev = nullptr;
        void* out_dev = nullptr;
        if (hipMalloc(&in_dev, out_bytes) != hipSuccess ||
            hipMalloc(&wt_dev, wt_bytes) != hipSuccess ||
            hipMalloc(&out_dev, out_bytes + guard_bytes) != hipSuccess) {
            std::printf("FAIL: F=%u alloc\n", features);
            ++g_failed;
            continue;
        }
        (void)hipMemcpy(in_dev, h_in.data(), out_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(wt_dev, h_wt.data(), wt_bytes, hipMemcpyHostToDevice);

        std::vector<uint16_t> sentinel(guard_bytes / 2u, 0xABABu);

        RunResult legacy;
        (void)hipMemset(out_dev, 0xAB, out_bytes + guard_bytes);
        (void)hipMemcpy(static_cast<unsigned char*>(out_dev) + out_bytes,
                        sentinel.data(), guard_bytes, hipMemcpyHostToDevice);
        const bool legacy_ok = run_mode(shape, false, in_dev, wt_dev, out_dev,
                                        legacy);

        RunResult generic;
        (void)hipMemset(out_dev, 0xAB, out_bytes + guard_bytes);
        (void)hipMemcpy(static_cast<unsigned char*>(out_dev) + out_bytes,
                        sentinel.data(), guard_bytes, hipMemcpyHostToDevice);
        const bool generic_ok = run_mode(shape, true, in_dev, wt_dev, out_dev,
                                         generic);

        check(legacy_ok && generic_ok, "both modes ran");
        if (!legacy_ok || !generic_ok) {
            (void)hipFree(in_dev);
            (void)hipFree(wt_dev);
            (void)hipFree(out_dev);
            continue;
        }

        check(legacy.fault == 0u && generic.fault == 0u, "no fault in either mode");
        check(legacy.dispatches == kPlans * 2u, "legacy dispatch count");
        check(generic.dispatches == kPlans * 2u, "generic dispatch count");
        check(legacy.dispatches == generic.dispatches, "dispatch count parity");
        check(legacy.completions == generic.completions, "completion parity");
        check(legacy.drained && generic.drained, "queue drained in both modes");
        check(legacy.output == generic.output, "output bit parity");
        check(legacy.guard == generic.guard, "output sentinel parity");
        check(legacy.kernarg_slot.size() == generic.kernarg_slot.size(),
              "kernarg slot size parity");
        check(legacy.kernarg_slot == generic.kernarg_slot,
              "kernarg explicit byte parity");
        check(legacy.guard == sentinel, "legacy output sentinel intact");
        check(generic.guard == sentinel, "generic output sentinel intact");
        if (legacy.output.size() != generic.output.size()) {
            std::printf("  output size mismatch: %zu vs %zu\n",
                        legacy.output.size(), generic.output.size());
        }
        if (legacy.kernarg_slot.size() == generic.kernarg_slot.size() &&
            legacy.kernarg_slot != generic.kernarg_slot) {
            for (size_t i = 0; i < legacy.kernarg_slot.size(); ++i) {
                if (legacy.kernarg_slot[i] != generic.kernarg_slot[i]) {
                    std::printf("  kernarg byte %zu: legacy=0x%02x generic=0x%02x\n",
                                i, legacy.kernarg_slot[i],
                                generic.kernarg_slot[i]);
                    break;
                }
            }
        }

        std::vector<float> ref_cpu;
        cpu_reference(h_in, h_wt, rows, features, group, features, ref_cpu);
        std::vector<float> ref_host(generic.output.size());
        std::vector<float> ref_mcu(generic.output.size());
        for (size_t i = 0; i < generic.output.size(); ++i) {
            ref_host[i] = fixture::f32_from_bf16(legacy.output[i]);
            ref_mcu[i] = fixture::f32_from_bf16(generic.output[i]);
        }
        check(compare_to_reference(ref_host, ref_cpu, features, features,
                                   "legacy vs cpu"),
              "legacy path matches the reference policy");
        check(compare_to_reference(ref_mcu, ref_cpu, features, features,
                                   "generic vs cpu"),
              "generic path matches the reference policy");

        std::printf("  shape F=%u G=%u parity dispatch=%llu kernarg=%zu\n",
                    features, group,
                    static_cast<unsigned long long>(generic.dispatches),
                    generic.kernarg_slot.size());

        (void)hipFree(in_dev);
        (void)hipFree(wt_dev);
        (void)hipFree(out_dev);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_kernarg_source_parity: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_kernarg_source_parity: PASS\n");
    return 0;
}
