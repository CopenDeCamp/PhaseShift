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

    const uint64_t kPlans = 600u;

    {
        Rig rig;
        if (!rig.open(0, 8u, 2u, 8u, 4u, false)) {
            std::printf("test_gpu_mcu_device_kernarg: setup failed\n");
            return 1;
        }
        const Chain chain = make_chain(2u);
        std::vector<mcu::McuKernelVariantDesc> variants{
            mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
        if (!rig.upload(chain.nodes, variants)) return 1;
        rig.poison_kernarg_args();

        mcu::GpuMcuFsmConfig config{};
        config.plan_id = 4u;
        config.template_source = 0u;
        config.barrier = 1u;
        config.timing_count = 0u;
        if (!rig.start(config)) {
            std::printf("test_gpu_mcu_device_kernarg: start failed\n");
            return 1;
        }
        for (uint64_t i = 0; i < kPlans; ++i) {
            check(rig.fsm.request_run().ok(), "plan requested");
        }
        auto waited = rig.fsm.wait_plans(kPlans, 60000);
        if (!waited.ok()) {
            std::printf("  wait: %s fault=%u pc=%u disp=%llu comp=%llu\n",
                        waited.message().c_str(),
                        static_cast<unsigned>(rig.fsm.fault_code()),
                        static_cast<unsigned>(rig.fsm.fault_pc()),
                        static_cast<unsigned long long>(rig.fsm.dispatches()),
                        static_cast<unsigned long long>(
                            rig.fsm.completions_observed()));
        }
        check(waited.ok(), "device kernarg plans completed");
        check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
        check(rig.fsm.dispatches() == kPlans * 2u, "dispatch count");
        check(rig.fsm.completions_observed() == kPlans * 2u, "completion count");

        const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
        const std::vector<uint32_t> expected = chain_expected(chain);
        check(got == expected,
              "worker read device-generated kernarg despite poisoned host slot");
        check(rig.close(), "device kernarg rig teardown");
    }

    {
        Rig tight;
        if (!tight.open(0, 8u, 2u, 8u, 4u, true)) {
            std::printf("test_gpu_mcu_device_kernarg: tight setup failed\n");
            return 1;
        }
        const Chain chain = make_chain(2u);
        std::vector<mcu::McuKernelVariantDesc> variants{
            mcu::make_probe_variant(0u, tight.meta, 64u, 1u)};
        if (!tight.upload(chain.nodes, variants)) return 1;
        mcu::GpuMcuFsmConfig config{};
        config.plan_id = 4u;
        config.template_source = 0u;
        config.barrier = 1u;
        config.timing_count = 0u;
        config.kernarg_slot_count = 2u;
        if (!tight.start(config)) {
            std::printf("test_gpu_mcu_device_kernarg: tight start failed\n");
            return 1;
        }
        for (uint64_t i = 0; i < 200u; ++i) {
            check(tight.fsm.request_run().ok(), "tight plan requested");
        }
        check(tight.fsm.wait_plans(200u, 30000).ok(),
              "tight kernarg ring completed");
        check(tight.fsm.fault_code() == mcu::McuFaultCode::none,
              "no fault with tight kernarg ring");
        check(tight.fsm.dispatches() == 400u, "tight ring dispatch count");
        const std::vector<uint32_t> got = tight.read_output_at(chain.dispatch_pc);
        check(got == chain_expected(chain), "tight ring outputs");
        check(tight.close(), "tight ring teardown");
    }

    {
        auto fsm_res = mcu::GpuMcuFsm::create(0, nullptr);
        check(!fsm_res.ok(), "null control stream rejected");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_device_kernarg: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_device_kernarg: PASS\n");
    return 0;
}
