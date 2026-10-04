#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gpu_mcu_test::Chain;
using gpu_mcu_test::Rig;
using gpu_mcu_test::chain_expected;
using gpu_mcu_test::make_feed_chain_with_marker;
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

    constexpr uint32_t kDispatch = 6u;
    constexpr uint32_t kDataDispatches = kDispatch - 1u;
    Rig rig;
    if (!rig.open(0, 16u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  kDispatch + 4u)) {
        std::printf("test_gpu_mcu_append_multiple_while_running: setup failed\n");
        return 1;
    }
    const Chain chain = make_feed_chain_with_marker(kDataDispatches, 4000u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x41505032u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.issue_wait_running = 1u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_append_multiple_while_running: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "plan requested");
    check(rig.fsm.wait_plans(1u, 60000).ok(), "plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == kDispatch, "dispatch count");
    check(rig.drain_queue(30000), "queue drained before host readback");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    if (!(got == expected)) {
        std::printf("[dbg] got=");
        for (uint32_t v : got) std::printf("%u ", v);
        std::printf("\n[dbg] exp=");
        for (uint32_t v : expected) std::printf("%u ", v);
        std::printf("\n");
        std::fflush(stdout);
    }
    check(got == expected, "dependent outputs preserved in order");

    const std::vector<uint64_t> ts = rig.read_timestamps();
    const std::vector<mcu::McuDispatchRecord> rec = rig.read_records();
    const uint64_t a_start = ts[0];
    const uint64_t a_end = ts[1];
    check(a_start != 0u && a_end > a_start, "A start/end recorded");

    uint32_t appended_during_a = 0u;
    for (uint32_t i = 1u; i < kDispatch; ++i) {
        const uint64_t publish = rec[i].publish_ts;
        if (a_start < publish && publish < a_end) {
            ++appended_during_a;
        }
    }
    check(appended_during_a == kDispatch - 1u,
          "all following packets appended while A was running");
    check(rig.fsm.append_ahead_count() >= kDispatch - 1u,
          "MCU observed append-ahead for every following packet");

    check(rig.close(), "append multiple rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_append_multiple_while_running: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_append_multiple_while_running: PASS\n");
    return 0;
}
