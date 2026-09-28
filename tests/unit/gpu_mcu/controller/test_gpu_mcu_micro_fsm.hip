#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
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
        std::printf("test_gpu_mcu_micro_fsm: setup failed\n");
        return 1;
    }
    Chain chain = make_chain(3u);
    chain.nodes[2].variant_id = 1u;
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 128u, 1u)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 5u;
    config.template_source = 1u;
    config.copy_lanes = 1u;
    config.lds_budget_bytes = 8192u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    config.timing_count = 8u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_micro_fsm: start failed\n");
        return 1;
    }

    // Exactly one host kick drives all three nodes.
    check(rig.fsm.request_run().ok(), "single host kick issued");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "plan completed");

    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.supervisor() == mcu::McuSupervisorState::idle,
          "supervisor returns to idle");
    check(rig.fsm.plans_started() == 1u, "one plan");
    check(rig.fsm.dispatches() == 3u, "three dispatches");
    check(rig.fsm.completions_observed() == 3u, "three completions");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    check(got == expected, "micro fsm node order and dependency");

    const std::vector<uint64_t> ts = rig.read_timestamps();
    bool worker_ts_ok = true;
    for (uint32_t pc : chain.dispatch_pc) {
        if (ts[static_cast<std::size_t>(pc) * 2u] == 0u ||
            ts[static_cast<std::size_t>(pc) * 2u + 1u] <
                ts[static_cast<std::size_t>(pc) * 2u]) {
            worker_ts_ok = false;
        }
    }
    check(worker_ts_ok, "worker start/end timestamps recorded");

    check(rig.close(), "micro fsm rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_micro_fsm: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_micro_fsm: PASS\n");
    return 0;
}
