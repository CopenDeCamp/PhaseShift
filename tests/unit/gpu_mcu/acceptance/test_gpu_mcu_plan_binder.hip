#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>
#include <phaseshift/runtime/gpu_mcu/binding/plan_binder.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

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

bool wait_for(const mcu::GpuMcuPersistentMcu& persistent, uint64_t target,
              uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (persistent.batches_completed() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return true;
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

    {
        rt::DeviceBatchContext host_ctx{};
        host_ctx.actual_rows = 26u;
        host_ctx.num_requests = 4u;
        host_ctx.num_outputs = 2u;
        uint32_t rows_field = 0u;
        uint32_t requests_field = 0u;
        uint32_t outputs_field = 0u;
        std::vector<mcu::McuInvocationPatch> patches(3);
        patches[0].target = reinterpret_cast<uint64_t>(&rows_field);
        patches[0].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::ActualRows);
        patches[1].target = reinterpret_cast<uint64_t>(&requests_field);
        patches[1].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::NumRequests);
        patches[2].target = reinterpret_cast<uint64_t>(&outputs_field);
        patches[2].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::NumOutputs);
        mcu::gpu_mcu_bind_execution_plan(&host_ctx, patches.data(), 3u);
        check(rows_field == 26u, "actual rows bound");
        check(requests_field == 4u, "num requests bound");
        check(outputs_field == 2u, "num outputs bound");

        host_ctx.actual_rows = 2u;
        mcu::gpu_mcu_bind_execution_plan(&host_ctx, patches.data(), 3u);
        check(rows_field == 2u, "geometry change rebinds rows");
        check(requests_field == 4u, "unchanged source stays");
    }

    {
        std::vector<rt::DeviceRequestDescriptor> descriptors(4);
        descriptors[0].execution_class = rt::ExecutionClass::DECODE;
        descriptors[0].row_begin = 0u;
        descriptors[0].row_count = 1u;
        descriptors[1].execution_class = rt::ExecutionClass::DECODE;
        descriptors[1].row_begin = 1u;
        descriptors[1].row_count = 1u;
        descriptors[2].execution_class = rt::ExecutionClass::SPEC_VERIFY;
        descriptors[2].row_begin = 2u;
        descriptors[2].row_count = 8u;
        descriptors[3].execution_class = rt::ExecutionClass::PREFILL;
        descriptors[3].row_begin = 10u;
        descriptors[3].row_count = 16u;
        rt::DeviceBatchContext region_ctx{};
        region_ctx.actual_rows = 26u;
        region_ctx.num_requests = 4u;
        region_ctx.requests = descriptors.data();

        uint32_t rows_field = 0u;
        mcu::McuInvocationPatch patch{};
        patch.target = reinterpret_cast<uint64_t>(&rows_field);
        patch.source = static_cast<uint8_t>(
            mcu::McuInvocationPatchSource::AttentionRegionRows);

        patch.param = 0u | (10u << 16);
        mcu::gpu_mcu_bind_execution_plan(&region_ctx, &patch, 1u);
        check(rows_field == 10u, "decode region rows bound");

        patch.param = 10u | (16u << 16);
        mcu::gpu_mcu_bind_execution_plan(&region_ctx, &patch, 1u);
        check(rows_field == 16u, "prefill region rows bound");

        region_ctx.num_requests = 2u;
        region_ctx.actual_rows = 2u;
        patch.param = 0u | (10u << 16);
        mcu::gpu_mcu_bind_execution_plan(&region_ctx, &patch, 1u);
        check(rows_field == 2u, "decode region rows shrink per loop");

        patch.param = 10u | (16u << 16);
        mcu::gpu_mcu_bind_execution_plan(&region_ctx, &patch, 1u);
        check(rows_field == 0u, "prefill region rows vanish per loop");
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

    const qrt::McuCompiledPlan plan = marker_plan();
    const ps::Status prepared = runtime.prepare_plan(plan);
    if (!prepared.ok()) {
        std::printf("FAIL: prepare_plan: %s\n", prepared.message().c_str());
        return 1;
    }
    check(runtime.row_global_patch_count() == 0u,
          "marker plan has no row-global patches");

    rt::DeviceBatchContext* ctx_dev = nullptr;
    rt::DeviceRequestDescriptor* req_dev = nullptr;
    uint32_t* row_seq_dev = nullptr;
    uint32_t* row_pos_dev = nullptr;
    uint64_t* ready_dev = nullptr;
    uint32_t* rows_dev = nullptr;
    mcu::McuInvocationPatch* patches_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&ctx_dev),
                  sizeof(rt::DeviceBatchContext)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&req_dev),
                  sizeof(rt::DeviceRequestDescriptor)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_seq_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_pos_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&rows_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&patches_dev),
                  sizeof(mcu::McuInvocationPatch)) != hipSuccess) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(rows_dev, 0, sizeof(uint32_t));
    (void)hipMemset(row_seq_dev, 0, sizeof(uint32_t));
    (void)hipMemset(row_pos_dev, 0, sizeof(uint32_t));

    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = 26u;
    ctx.num_requests = 1u;
    ctx.num_decode_requests = 1u;
    (void)hipMemcpy(ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice);

    mcu::McuInvocationPatch patch{};
    patch.target = reinterpret_cast<uint64_t>(rows_dev);
    patch.source =
        static_cast<uint8_t>(mcu::McuInvocationPatchSource::ActualRows);
    (void)hipMemcpy(patches_dev, &patch, sizeof(patch), hipMemcpyHostToDevice);

    hipStream_t persistent_stream = nullptr;
    hipStream_t update_stream = nullptr;
    if (hipStreamCreate(&persistent_stream) != hipSuccess ||
        hipStreamCreateWithFlags(&update_stream, hipStreamNonBlocking) !=
            hipSuccess) {
        std::printf("FAIL: stream create\n");
        return 1;
    }

    auto persistent_result = mcu::GpuMcuPersistentMcu::create(0, persistent_stream);
    if (!persistent_result.ok()) {
        std::printf("FAIL: persistent create\n");
        return 1;
    }
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();

    check(persistent
              .configure_batch_plan(ctx_dev, req_dev, row_seq_dev, row_pos_dev, 1u,
                                    26u, 16u, ready_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev, nullptr,
                                   0u, nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent.configure_execution_patches(patches_dev, 1u).ok(),
          "configure execution patches");
    check(persistent.execution_patch_count() == 1u, "patch table recorded");

    uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent start");
    check(wait_for(persistent, 1u, 5000u), "first loop completed");

    uint32_t rows_seen = 0u;
    (void)hipMemcpyAsync(&rows_seen, rows_dev, sizeof(uint32_t),
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipStreamSynchronize(update_stream);
    check(rows_seen == 26u, "first loop bound actual_rows=26");

    ctx.actual_rows = 3u;
    ready = 2u;
    (void)hipMemcpyAsync(ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice,
                         update_stream);
    (void)hipMemcpyAsync(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice,
                         update_stream);
    (void)hipStreamSynchronize(update_stream);
    check(wait_for(persistent, 2u, 5000u), "second loop completed");

    rows_seen = 0u;
    (void)hipMemcpyAsync(&rows_seen, rows_dev, sizeof(uint32_t),
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipStreamSynchronize(update_stream);
    check(rows_seen == 3u, "second loop rebound actual_rows=3 without re-prepare");
    check(runtime.execution_state()->plans_started >= 2u,
          "prepared fsm served both loops");
    check(persistent.batches_failed() == 0u, "no fault");

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");
    check(runtime.shutdown().ok(), "runtime shutdown");

    (void)hipFree(patches_dev);
    (void)hipFree(rows_dev);
    (void)hipFree(ready_dev);
    (void)hipFree(row_pos_dev);
    (void)hipFree(row_seq_dev);
    (void)hipFree(req_dev);
    (void)hipFree(ctx_dev);
    check(hipStreamDestroy(update_stream) == hipSuccess, "update stream destroy");
    check(hipStreamDestroy(persistent_stream) == hipSuccess,
          "persistent stream destroy");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_plan_binder: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_plan_binder: PASS\n");
    return 0;
}
