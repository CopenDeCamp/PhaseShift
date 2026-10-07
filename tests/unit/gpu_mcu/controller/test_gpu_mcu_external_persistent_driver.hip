#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace rt = ps::runtime;
namespace qrt = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

qrt::McuCompiledPlan marker_plan() {
    qrt::McuCompiledPlan plan;
    mcu::McuPlanNode dispatch{};
    dispatch.variant_id = 0u;
    dispatch.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    dispatch.completion_slot = 0u;
    dispatch.element_count = 1u;
    dispatch.flags = mcu::kMcuNodeDispatch;
    dispatch.next = 1u;
    plan.nodes.push_back(dispatch);
    mcu::McuPlanNode wait{};
    wait.completion_slot = 0u;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = 2u;
    plan.nodes.push_back(wait);
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    plan.nodes.push_back(end);
    qrt::McuCompiledVariant variant{};
    variant.kind = qrt::McuCompiledVariantKind::CompletionMarker;
    variant.grid = 1u;
    variant.workgroup = 64u;
    plan.variants.push_back(variant);
    return plan;
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

    qrt::McuDecodeRuntimeOptions options{};
    options.device = 0;
    options.record_entries = 0u;
    options.driver_mode = qrt::McuPlanDriverMode::ExternalPersistentMcu;
    auto runtime_result = qrt::McuDecodeRuntime::create(options);
    if (!runtime_result.ok()) {
        std::printf("FAIL: runtime create: %s\n",
                    runtime_result.status().message().c_str());
        return 1;
    }
    qrt::McuDecodeRuntime runtime = runtime_result.release();

    hipStream_t persistent_stream = nullptr;
    if (hipStreamCreate(&persistent_stream) != hipSuccess) {
        std::printf("FAIL: persistent stream create\n");
        return 1;
    }

    const qrt::McuCompiledPlan plan = marker_plan();
    const ps::Status prepared = runtime.prepare_plan(plan);
    if (!prepared.ok()) {
        std::printf("FAIL: prepare_plan: %s\n", prepared.message().c_str());
        return 1;
    }
    check(runtime.ready(), "plan prepared");
    check(runtime.external_driver(), "external driver mode active");
    check(!runtime.fsm().running(), "host stream controller not started");
    check(runtime.execution_state() != nullptr, "execution state exposed");
    check(!runtime.enqueue_run(nullptr, 1u).ok(),
          "enqueue_run rejected for external driver");
    check(!runtime.enqueue_wait(nullptr, 1u).ok(),
          "enqueue_wait rejected for external driver");

    rt::DeviceBatchContext* ctx_dev = nullptr;
    rt::DeviceRequestDescriptor* req_dev = nullptr;
    int32_t* token_dev = nullptr;
    uint64_t* ready_dev = nullptr;
    uint32_t* row_seq_dev = nullptr;
    uint32_t* row_pos_dev = nullptr;
    uint64_t* plan_epoch_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&ctx_dev),
                  sizeof(rt::DeviceBatchContext)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&req_dev),
                  sizeof(rt::DeviceRequestDescriptor)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&token_dev), sizeof(int32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_seq_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_pos_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&plan_epoch_dev), sizeof(uint64_t)) !=
            hipSuccess) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(plan_epoch_dev, 0, sizeof(uint64_t));

    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = 1u;
    ctx.num_requests = 1u;
    ctx.num_decode_requests = 1u;
    ctx.token_ids = token_dev;
    ctx.requests = req_dev;
    ctx.row_sequence_slots = row_seq_dev;
    ctx.row_positions = row_pos_dev;
    rt::DeviceRequestDescriptor descriptor{};
    descriptor.request_handle = rt::RequestHandle{0u, 1u};
    descriptor.row_begin = 0u;
    descriptor.row_count = 1u;
    descriptor.sequence_length = 1u;
    descriptor.execution_class = rt::ExecutionClass::DECODE;
    const int32_t token = 7;
    (void)hipMemcpy(ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice);
    (void)hipMemcpy(req_dev, &descriptor, sizeof(descriptor),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(token_dev, &token, sizeof(token), hipMemcpyHostToDevice);

    auto persistent_result = mcu::GpuMcuPersistentMcu::create(
        0, persistent_stream);
    if (!persistent_result.ok()) {
        std::printf("FAIL: persistent create\n");
        return 1;
    }
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();

    check(persistent
              .configure_batch_plan(ctx_dev, req_dev, row_seq_dev, row_pos_dev,
                                    1u, 1u, 1u, plan_epoch_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev,
                                   nullptr, 0u, nullptr, 0u)
              .ok(),
          "configure execution with prepared fsm");
    const uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent mcu start");
    check(persistent.running(), "persistent mcu running");


    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (persistent.batches_dispatched() < 1u) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check(persistent.batches_dispatched() >= 1u,
          "persistent mcu dispatched prepared plan");
    check(persistent.batches_completed() >= 1u,
          "prepared plan completed without fault");
    check(!runtime.fsm().running(), "no second controller kernel");
    check(runtime.execution_state()->plans_started >= 1u,
          "prepared fsm recorded a plan");
    check(runtime.execution_state()->fault_code ==
              static_cast<uint32_t>(mcu::McuFaultCode::none),
          "prepared fsm has no fault");

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");
    check(runtime.shutdown().ok(), "runtime shutdown");

    (void)hipFree(ctx_dev);
    (void)hipFree(req_dev);
    (void)hipFree(token_dev);
    (void)hipFree(ready_dev);
    (void)hipFree(row_seq_dev);
    (void)hipFree(row_pos_dev);
    (void)hipFree(plan_epoch_dev);
    check(hipStreamDestroy(persistent_stream) == hipSuccess,
          "persistent stream destroy");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_external_persistent_driver: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_external_persistent_driver: PASS\n");
    return 0;
}
