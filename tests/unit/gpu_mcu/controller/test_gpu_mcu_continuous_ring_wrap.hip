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

    constexpr uint32_t kDispatch = 10000u;
    constexpr uint32_t kQueueSize = 64u;
    constexpr uint32_t kAhead = 8u;
    const uint32_t node_cap = kDispatch + 2u;

    Rig rig;
    if (!rig.open(0, node_cap, 2u, 8u, 0u, false, kQueueSize, 64u)) {
        std::printf("test_gpu_mcu_continuous_ring_wrap: setup failed\n");
        return 1;
    }
    const Chain chain = make_feed_chain_with_marker(kDispatch - 1u, 0u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x57524150u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = kAhead;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_continuous_ring_wrap: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "plan requested");
    auto waited = rig.fsm.wait_plans(1u, 300000);
    if (!waited.ok()) {
        std::printf("  wait: %s disp=%llu\n", waited.message().c_str(),
                    static_cast<unsigned long long>(rig.fsm.dispatches()));
    }
    check(waited.ok(), "long fed plan completed without hang");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == kDispatch, "dispatch count across wraps");
    check(rig.drain_queue(30000), "queue drained before host readback");

    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    const std::vector<uint32_t> expected = chain_expected(chain);
    check(got == expected, "no overwrite or corruption across ring wraps");

    check(rig.fsm.wrap_count() > 0u, "ring wrapped");
    check(rig.fsm.backpressure_wait_count() > 0u, "backpressure observed");
    check(rig.fsm.refill_count() > 0u, "runtime refill observed");
    std::printf("  wraps=%llu full=%llu empty=%llu refill=%llu max_ahead=%llu\n",
                static_cast<unsigned long long>(rig.fsm.wrap_count()),
                static_cast<unsigned long long>(rig.fsm.backpressure_wait_count()),
                static_cast<unsigned long long>(rig.fsm.queue_empty_count()),
                static_cast<unsigned long long>(rig.fsm.refill_count()),
                static_cast<unsigned long long>(rig.fsm.max_ahead()));

    check(rig.close(), "ring wrap rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_continuous_ring_wrap: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_continuous_ring_wrap: PASS\n");
    return 0;
}
