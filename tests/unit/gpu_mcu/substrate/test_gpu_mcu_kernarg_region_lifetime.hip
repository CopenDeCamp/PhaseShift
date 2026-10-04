#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using gpu_mcu_test::Chain;
using gpu_mcu_test::Rig;
using gpu_mcu_test::chain_expected;
using gpu_mcu_test::make_feed_chain;
namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint8_t kPoison = 0xABu;

void poison_region(const Rig& rig) {
    std::memset(rig.region.base, kPoison, rig.region.actual_size);
}

std::vector<unsigned char> read_region(const Rig& rig) {
    std::vector<unsigned char> out(rig.region.actual_size, 0u);
    std::memcpy(out.data(), rig.region.base, rig.region.actual_size);
    return out;
}

bool slot_touched(const std::vector<unsigned char>& region,
                  const mcu::GpuMcuKernargRegion& r, uint32_t index) {
    const unsigned char* slot = region.data() + index * r.slot_stride;
    for (std::size_t i = 0; i < r.slot_stride; ++i) {
        if (slot[i] != kPoison) return true;
    }
    return false;
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

    constexpr uint32_t kDispatch = 4u;
    constexpr uint64_t kPlans = 1000u;
    const uint32_t node_cap = kDispatch + 2u;

    {
        Rig rig;
        if (!rig.open(0, node_cap, 1u, 8u, 0u, false)) {
            std::printf("test_gpu_mcu_kernarg_region_lifetime: setup failed\n");
            return 1;
        }
        const Chain chain = make_feed_chain(kDispatch, 0u);
        std::vector<mcu::McuKernelVariantDesc> variants{
            mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
        if (!rig.upload(chain.nodes, variants)) return 1;

        mcu::GpuMcuFsmConfig config{};
        config.plan_id = 0x4b524e47u;
        config.template_source = 0u;
        config.barrier = 1u;
        config.timing_count = 0u;
        config.kernarg_slot_count = kDispatch;
        config.doorbell_mode =
            static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
        config.queue_ahead_depth = 4u;
        if (!rig.start(config)) {
            std::printf("test_gpu_mcu_kernarg_region_lifetime: start failed\n");
            return 1;
        }

        for (uint64_t i = 1; i <= kPlans; ++i) {
            poison_region(rig);
            if (!rig.fsm.request_run().ok()) {
                check(false, "plan requested");
                break;
            }
            auto step = rig.fsm.wait_plans(i, 300000);
            if (!step.ok()) {
                std::printf("  wait[%llu]: %s fault=%u pc=%u\n",
                            static_cast<unsigned long long>(i),
                            step.message().c_str(),
                            static_cast<unsigned>(rig.fsm.fault_code()),
                            static_cast<unsigned>(rig.fsm.fault_pc()));
                check(false, "region lifetime plan completed");
                break;
            }
        }
        check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
        check(rig.fsm.dispatches() == kPlans * kDispatch,
              "region dispatch count");
        check(rig.drain_queue(30000), "queue drained");

        const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
        check(got == chain_expected(chain),
              "device kernarg regenerated after every region");

        const std::vector<unsigned char> region = read_region(rig);
        bool rewritten = true;
        for (uint32_t slot = 0; slot < kDispatch; ++slot) {
            rewritten &= slot_touched(region, rig.region, slot);
        }
        check(rewritten, "region slots rewritten");
        bool budget_untouched = true;
        for (uint32_t slot = kDispatch; slot < rig.region.slot_count; ++slot) {
            budget_untouched &= !slot_touched(region, rig.region, slot);
        }
        check(budget_untouched, "no kernarg slot beyond the region budget");
        check(rig.close(), "lifetime rig teardown");
    }

    {
        Rig rig;
        if (!rig.open(0, node_cap, 1u, 8u, 0u, false)) {
            std::printf("test_gpu_mcu_kernarg_region_lifetime: setup failed\n");
            return 1;
        }
        const Chain chain = make_feed_chain(kDispatch, 0u);
        std::vector<mcu::McuKernelVariantDesc> variants{
            mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
        if (!rig.upload(chain.nodes, variants)) return 1;

        mcu::GpuMcuFsmConfig config{};
        config.plan_id = 0x4b524e48u;
        config.template_source = 0u;
        config.barrier = 1u;
        config.timing_count = 0u;
        config.kernarg_slot_count = kDispatch - 1u;
        config.doorbell_mode =
            static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
        config.queue_ahead_depth = 4u;
        if (!rig.start(config)) {
            std::printf("test_gpu_mcu_kernarg_region_lifetime: setup failed\n");
            return 1;
        }
        check(rig.fsm.request_run().ok(), "tight plan requested");
        const bool tight_ok = rig.fsm.wait_plans(1u, 30000).ok();
        check(!tight_ok, "region budget overrun is not silently accepted");
        check(rig.fsm.fault_code() ==
                  mcu::McuFaultCode::kernarg_region_exhausted,
              "region exhaustion faults fail-closed");
        (void)rig.fsm.request_stop();
        (void)rig.fsm.wait_stopped(5000);
        check(rig.drain_queue(30000), "faulted queue drained");
        check(rig.close(), "tight rig teardown");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_kernarg_region_lifetime: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_kernarg_region_lifetime: PASS\n");
    return 0;
}
