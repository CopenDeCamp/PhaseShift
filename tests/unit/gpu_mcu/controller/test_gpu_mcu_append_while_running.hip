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

    Rig rig;
    if (!rig.open(0, 16u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize, 8u)) {
        std::printf("test_gpu_mcu_append_while_running: setup failed\n");
        return 1;
    }
    const Chain chain = make_feed_chain_with_marker(2u, 4000u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x41505031u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 0u;
    config.issue_wait_running = 1u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_append_while_running: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "plan requested");
    check(rig.fsm.wait_plans(1u, 60000).ok(), "plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == 3u, "three dispatches");
    check(rig.drain_queue(30000), "queue drained before host readback");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    check(got == expected, "dependent outputs preserved");

    const std::vector<uint64_t> ts = rig.read_timestamps();
    const std::vector<mcu::McuDispatchRecord> rec = rig.read_records();

    const uint64_t a_start = ts[0];
    const uint64_t a_end = ts[1];
    const uint64_t b_publish = rec[1].publish_ts;
    const uint64_t b_start = ts[2];
    check(a_start != 0u && a_end > a_start, "A start/end recorded");
    check(b_publish != 0u, "B publish recorded");
    check(a_start < b_publish && b_publish < a_end,
          "B published while A was running");
    check(b_start >= a_end, "B launched after A completed");
    check(rec[1].flags & mcu::kMcuRecordAppendAhead,
          "B flagged as appended ahead of A completion");
    check(rig.fsm.append_ahead_count() >= 2u,
          "MCU observed append-ahead for B and C");

    check(rig.close(), "append while running rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_append_while_running: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_append_while_running: PASS\n");
    return 0;
}
