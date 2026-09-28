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

    constexpr uint32_t kDispatch = 8u;
    constexpr uint32_t kDataDispatches = kDispatch - 1u;
    constexpr uint32_t kBatch = 4u;
    Rig rig;
    if (!rig.open(0, 16u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  kDispatch + 4u)) {
        std::printf("test_gpu_mcu_doorbell_coalescing: setup failed\n");
        return 1;
    }
    const Chain chain = make_feed_chain_with_marker(kDataDispatches, 2000u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};
    if (!rig.upload(chain.nodes, variants)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x434f414cu;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::coalesce);
    config.doorbell_batch = kBatch;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_doorbell_coalescing: start failed\n");
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
        std::printf("\n[dbg] records:");
        const std::vector<mcu::McuDispatchRecord> rr = rig.read_records();
        for (uint32_t i = 0u; i < kDispatch; ++i) {
            std::printf(" [%u gen=%llu ts=%llu]", i,
                        static_cast<unsigned long long>(rr[i].generation),
                        static_cast<unsigned long long>(rr[i].publish_ts));
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    check(got == expected, "coalesced doorbell preserves order and outputs");

    const uint32_t expected_doorbells = kDispatch / kBatch;
    check(rig.fsm.doorbell_count() == expected_doorbells,
          "doorbell coalesced to the batch count");

    const std::vector<mcu::McuDispatchRecord> rec = rig.read_records();
    uint32_t doorbell_records = 0u;
    bool doorbell_positions_ok = true;
    for (uint32_t i = 0u; i < kDispatch; ++i) {
        const bool marked = (rec[i].flags & mcu::kMcuRecordDoorbell) != 0u;
        const bool expected_mark = ((i + 1u) % kBatch) == 0u;
        if (marked) ++doorbell_records;
        if (marked != expected_mark) doorbell_positions_ok = false;
    }
    check(doorbell_records == expected_doorbells,
          "only tail packets carry the doorbell flag");
    check(doorbell_positions_ok, "doorbell flag positions match the batch");

    check(rig.close(), "doorbell coalescing rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_doorbell_coalescing: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_doorbell_coalescing: PASS\n");
    return 0;
}
