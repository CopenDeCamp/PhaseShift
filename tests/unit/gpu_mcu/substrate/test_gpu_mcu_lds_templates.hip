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
    if (!rig.open(0, 16u, 4u, 16u, 8u, true)) {
        std::printf("test_gpu_mcu_lds_templates: setup failed\n");
        return 1;
    }
    const Chain chain = make_chain(3u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
    if (!rig.upload(chain.nodes, variants)) {
        std::printf("test_gpu_mcu_lds_templates: upload failed\n");
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 3u;
    config.template_source = 1u;
    config.copy_lanes = 1u;
    config.lds_budget_bytes = 8192u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    config.idle_sleep = 8u;

    {
        auto fsm_res = mcu::GpuMcuFsm::create(0, rig.partition.control_stream());
        check(fsm_res.ok(), "probe fsm created");
        if (fsm_res.ok()) {
            auto probe = fsm_res.release();
            mcu::GpuMcuFsmConfig bad = config;
            bad.lds_budget_bytes = 32u;
            check(!probe.configure(bad).ok(),
                  "configure fail-closed when templates exceed LDS budget");
            check(probe.shutdown().ok(), "probe fsm shutdown");
        }
    }

    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_lds_templates: start failed\n");
        return 1;
    }

    const uint64_t kPlans = 400u;
    for (uint64_t i = 0; i < kPlans; ++i) {
        check(rig.fsm.request_run().ok(), "plan requested");
    }
    check(rig.fsm.wait_plans(kPlans, 60000).ok(),
          "all plans completed without fault");

    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.supervisor() == mcu::McuSupervisorState::idle,
          "supervisor idle after plans");
    check(rig.fsm.plans_started() == kPlans, "plan count");
    check(rig.fsm.dispatches() == kPlans * 3u, "dispatch count");
    check(rig.fsm.completions_observed() == kPlans * 3u, "completion count");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    check(got == expected, "chained outputs with LDS templates");

    check(rig.fsm.request_stop().ok(), "stop requested");
    check(rig.fsm.wait_stopped(5000).ok(), "stopped");
    check(rig.fsm.shutdown().ok(), "shutdown for recreate");
    rig.fsm_started = false;

    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_lds_templates: recreate failed\n");
        return 1;
    }
    check(rig.fsm.request_run().ok(), "recreated plan requested");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "recreated plan completed");
    check(rig.fsm.dispatches() == 3u, "recreated dispatch count");
    const std::vector<uint32_t> got2 = rig.read_output_at(chain.dispatch_pc);
    check(got2 == expected, "recreated chained outputs");

    if (!rig.close()) {
        std::printf("test_gpu_mcu_lds_templates: teardown failed\n");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_lds_templates: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_lds_templates: PASS\n");
    return 0;
}
