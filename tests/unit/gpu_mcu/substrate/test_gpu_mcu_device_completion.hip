#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
using gpu_mcu_test::Chain;
using gpu_mcu_test::make_chain;
using gpu_mcu_test::chain_expected;
namespace mcu = ps::runtime::gpu_mcu;

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

    Rig rig;
    if (!rig.open(0, 16u, 2u, 8u, 8u)) {
        std::printf("test_gpu_mcu_device_completion: setup failed\n");
        return 1;
    }
    const Chain chain = make_chain(4u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    {
        std::vector<mcu::GpuMcuDeviceCompletion> stale(8);
        for (auto& c : stale) {
            c.generation = 0xFFFF0000u;
            c.state = static_cast<uint32_t>(
                mcu::GpuMcuDeviceCompletionState::complete);
        }
        (void)hipMemcpy(rig.completions, stale.data(),
                        stale.size() * sizeof(mcu::GpuMcuDeviceCompletion),
                        hipMemcpyHostToDevice);
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 6u;
    config.template_source = 0u;
    config.lds_budget_bytes = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_device_completion: start failed\n");
        return 1;
    }

    const uint64_t kPlans = 2500u;
    for (uint64_t i = 0; i < kPlans; ++i) {
        check(rig.fsm.request_run().ok(), "plan requested");
    }
    auto waited = rig.fsm.wait_plans(kPlans, 120000);
    if (!waited.ok()) {
        std::printf("  wait: %s disp=%llu comp=%llu\n", waited.message().c_str(),
                    static_cast<unsigned long long>(rig.fsm.dispatches()),
                    static_cast<unsigned long long>(
                        rig.fsm.completions_observed()));
    }
    check(waited.ok(), "device completion transitions completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none,
          "stale generation never accepted as completion");
    check(rig.fsm.dispatches() == kPlans * 4u, "dispatch count");
    check(rig.fsm.completions_observed() == kPlans * 4u, "completion count");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    check(got == expected, "dependent outputs through device completion");

    check(rig.close(), "device completion rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_device_completion: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_device_completion: PASS\n");
    return 0;
}
