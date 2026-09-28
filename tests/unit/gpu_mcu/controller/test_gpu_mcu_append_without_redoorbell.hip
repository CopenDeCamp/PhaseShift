#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

using gpu_mcu_test::Chain;
using gpu_mcu_test::Rig;
using gpu_mcu_test::make_feed_chain_no_wait;
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
    if (!rig.open(0, 16u, 1u, 8u, 0u, false, mcu::kAqlDefaultQueueSize, 16u)) {
        std::printf(
            "test_gpu_mcu_append_without_redoorbell: setup failed\n");
        return 1;
    }
    const Chain chain = make_feed_chain_no_wait(2u, 2000u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4e4f4442u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::first_only);
    if (!rig.start(config)) {
        std::printf(
            "test_gpu_mcu_append_without_redoorbell: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "first plan requested");
    check(rig.fsm.wait_plans(1u, 60000).ok(), "first plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault after first");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::vector<uint64_t> ts = rig.read_timestamps();
    const bool b_ran_without_redoorbell = ts[2] != 0u;
    std::printf(
        "  observed_case=%s (B %s executed before a later doorbell)\n",
        b_ran_without_redoorbell ? "case2_cp_picked_up" : "case1_not_executed",
        b_ran_without_redoorbell ? "was" : "was not");

    check(rig.fsm.request_run().ok(), "second plan requested");
    check(rig.fsm.wait_plans(2u, 60000).ok(), "second plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault after second");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ts = rig.read_timestamps();
    check(ts[2] != 0u, "B ran after a later doorbell advanced past it");
    check(ts[3] > ts[2], "B start/end recorded");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    check(got.size() >= 2u && got[0] == 1u && got[1] == 2u,
          "outputs intact after recovery doorbell");
    check(rig.fsm.doorbell_count() == 2u,
          "only the first packet of each plan rang the doorbell");
    check(rig.fsm.doorbell_count() < rig.fsm.dispatches(),
          "doorbells fewer than dispatches");

    check(rig.close(), "no redoorbell rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_append_without_redoorbell: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_append_without_redoorbell: PASS\n");
    return 0;
}
