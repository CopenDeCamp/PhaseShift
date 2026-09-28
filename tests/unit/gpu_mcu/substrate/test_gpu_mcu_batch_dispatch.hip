#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

namespace {

using gpu_mcu_test::Chain;
using gpu_mcu_test::make_chain;
using gpu_mcu_test::Rig;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

template <typename F>
bool wait_until(F predicate, uint32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

void device_to_host(void* dst, const void* src, std::size_t bytes,
                    hipStream_t stream) {
    (void)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
}

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

constexpr uint32_t kSlots = 4u;
constexpr int32_t kBoundToken = 4242;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
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
    if (!rig.open(0, 8u, 2u, 8u, 0u)) {
        std::printf("test_gpu_mcu_batch_dispatch: rig setup failed\n");
        return 1;
    }
    Chain chain = make_chain(1u);
    chain.nodes[1].variant_id = 1u;
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 128u, 1u)};
    if (!rig.upload(chain.nodes, variants)) return 1;

    mcu::GpuMcuFsmConfig fsm_config{};
    fsm_config.plan_id = 0xcb50001u;
    fsm_config.template_source = 0u;
    fsm_config.copy_lanes = 1u;
    fsm_config.barrier = 1u;
    fsm_config.prepared_enabled = 0u;
    if (!rig.prepare(fsm_config)) {
        std::printf("test_gpu_mcu_batch_dispatch: fsm prepare failed\n");
        return 1;
    }
    check(!rig.fsm_started, "the standalone fsm kernel is not started");
    check(rig.fsm.device_state() != nullptr, "fsm state is available");

    hipStream_t work = rig.partition.worker_stream();
    auto slots_result = mcu::GpuMcuSlotTable::create(0, kSlots);
    check(slots_result.ok(), "slot table create");
    if (!slots_result.ok()) return 1;
    mcu::GpuMcuSlotTable slots = slots_result.release();

    uint32_t* applied = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&applied), sizeof(uint32_t)) ==
              hipSuccess,
          "applied alloc");
    check(mcu::gpu_mcu_slot_table_claim(slots, 0u, 0xcb5u, 32u, 0u, 256u, 8u,
                                        applied, work)
              .ok(),
          "slot claim launches");
    check(hipStreamSynchronize(work) == hipSuccess, "slot claim sync");

    int32_t* tokens = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&tokens), kSlots * sizeof(int32_t)) ==
              hipSuccess,
          "tokens alloc");
    const int32_t host_tokens[kSlots] = {kBoundToken, 0, 0, 0};
    check(hipMemcpy(tokens, host_tokens, sizeof(host_tokens),
                    hipMemcpyHostToDevice) == hipSuccess,
          "tokens upload");

    void* ctx_host = nullptr;
    void* ctx_dev = nullptr;
    check(map_alloc(sizeof(rt::DeviceBatchContext), &ctx_host, &ctx_dev),
          "batch context alloc");
    void* ready_host = nullptr;
    void* ready_dev = nullptr;
    check(map_alloc(sizeof(uint64_t), &ready_host, &ready_dev),
          "ready epoch alloc");

    std::vector<mcu::GpuMcuSlotState> slot_before(kSlots);
    device_to_host(slot_before.data(), slots.device_slots(),
                   kSlots * sizeof(mcu::GpuMcuSlotState), work);

    auto* context = static_cast<rt::DeviceBatchContext*>(ctx_host);
    std::memset(context, 0, sizeof(rt::DeviceBatchContext));
    context->actual_rows = 1u;
    context->num_requests = 1u;
    context->token_ids = tokens;
    static_cast<uint64_t*>(ready_host)[0] = 1u;

    void* plan_epoch_host = nullptr;
    void* plan_epoch_dev = nullptr;
    check(map_alloc(sizeof(uint64_t), &plan_epoch_host, &plan_epoch_dev),
          "plan epoch alloc");
    rt::DeviceRequestDescriptor* requests = nullptr;
    uint32_t* row_slots = nullptr;
    uint32_t* row_positions = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&requests),
                    kSlots * sizeof(rt::DeviceRequestDescriptor)) == hipSuccess,
          "requests alloc");
    check(hipMalloc(reinterpret_cast<void**>(&row_slots),
                    kSlots * sizeof(uint32_t)) == hipSuccess,
          "row slots alloc");
    check(hipMalloc(reinterpret_cast<void**>(&row_positions),
                    kSlots * sizeof(uint32_t)) == hipSuccess,
          "row positions alloc");

    auto persistent_result =
        mcu::GpuMcuPersistentMcu::create(0, rig.partition.control_stream());
    check(persistent_result.ok(), "persistent mcu create");
    if (!persistent_result.ok()) return 1;
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();

    check(persistent
              .configure_batch_plan(static_cast<rt::DeviceBatchContext*>(ctx_dev),
                                    requests, row_slots, row_positions, kSlots,
                                    kSlots, 0u,
                                    static_cast<uint64_t*>(plan_epoch_dev))
              .ok(),
          "configure batch plan");

    check(persistent
              .configure_execution(rig.fsm.device_state(), 1u,
                                   static_cast<uint64_t*>(ready_dev), nullptr, 0u,
                                   nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent.execution_enabled(), "execution is enabled");
    check(persistent.start().ok(), "persistent mcu start");
    check(persistent.running(), "persistent mcu is running");

    check(wait_until([&] { return persistent.execution_epoch() >= 1u; }, 10000u),
          "the persistent mcu dispatches the ready batch");
    check(wait_until([&] { return persistent.batches_completed() >= 1u; }, 10000u),
          "the dispatched batch completes");
    check(persistent.batches_dispatched() >= 1u, "dispatch telemetry");
    check(persistent.batches_failed() == 0u, "no failed batch");

    check(persistent.request_stop().ok(), "stop requested");
    check(persistent.wait_stopped(5000).ok(), "persistent mcu stopped");

    const uint32_t expected =
        static_cast<uint32_t>(kBoundToken) + chain.values[0];
    const std::vector<uint32_t> got = rig.read_output_at(chain.dispatch_pc);
    check(!got.empty() && got[0] == expected,
          "the worker consumed the bound batch token");

    std::vector<mcu::GpuMcuSlotState> slot_after(kSlots);
    device_to_host(slot_after.data(), slots.device_slots(),
                   kSlots * sizeof(mcu::GpuMcuSlotState), work);
    check(std::memcmp(slot_before.data(), slot_after.data(),
                      kSlots * sizeof(mcu::GpuMcuSlotState)) == 0,
          "dispatch does not progress the slot");

    check(persistent.shutdown().ok(), "persistent mcu shutdown");
    check(slots.shutdown().ok(), "slot table shutdown");
    check(rig.close(), "rig teardown");
    check(hipFree(tokens) == hipSuccess, "tokens free");
    check(hipFree(applied) == hipSuccess, "applied free");
    check(hipHostFree(ctx_host) == hipSuccess, "batch context free");
    check(hipHostFree(ready_host) == hipSuccess, "ready epoch free");
    check(hipHostFree(plan_epoch_host) == hipSuccess, "plan epoch free");
    check(hipFree(requests) == hipSuccess, "requests free");
    check(hipFree(row_slots) == hipSuccess, "row slots free");
    check(hipFree(row_positions) == hipSuccess, "row positions free");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_batch_dispatch: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_batch_dispatch: PASS\n");
    return 0;
}
