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
        std::printf("test_gpu_mcu_prepared_dispatch: setup failed\n");
        return 1;
    }
    const Chain chain = make_chain(3u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 7u;
    config.template_source = 0u;
    config.lds_budget_bytes = 0u;
    config.barrier = 1u;
    config.prepared_enabled = 1u;
    config.timing_count = 8u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_prepared_dispatch: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "plan requested");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "prepared plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == 3u, "three dispatches");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    check(got == expected,
          "prepared dispatch preserves dependency order and outputs");

    std::vector<mcu::McuDispatchTiming> timing(8);
    rig.device_to_host(timing.data(), rig.timing,
                       timing.size() * sizeof(mcu::McuDispatchTiming));

    // Dispatch ordinal 1 was staged while dispatch ordinal 0 was still
    // running: its packet_stage_end must precede ordinal 0's dependency_ready.
    check(timing[1].packet_stage_end != 0u &&
              timing[0].dependency_ready != 0u &&
              timing[1].packet_stage_end < timing[0].dependency_ready,
          "next dispatch staged before previous dependency satisfied");
    check(timing[2].packet_stage_end != 0u &&
              timing[1].dependency_ready != 0u &&
              timing[2].packet_stage_end < timing[1].dependency_ready,
          "third dispatch staged before second dependency satisfied");

    check(rig.close(), "prepared dispatch rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_prepared_dispatch: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_prepared_dispatch: PASS\n");
    return 0;
}
