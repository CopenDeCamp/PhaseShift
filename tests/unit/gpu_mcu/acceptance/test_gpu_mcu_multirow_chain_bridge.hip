#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kMaxRows = 26u;
constexpr uint32_t kLoopRows = 13u;

constexpr uint32_t kFeatures = chain::kFeatures;
constexpr uint32_t kN = chain::kN;
constexpr uint32_t kKp = chain::kKp;
constexpr uint32_t kNb = chain::kNb;
constexpr uint32_t kTiles = chain::kTiles;

const rt::Int8ActivationWorkspaceLayout& act_layout() {
    static const rt::Int8ActivationWorkspaceLayout l =
        rt::Int8ActivationWorkspaceLayout::make(kFeatures, kMaxRows);
    return l;
}

uint64_t rms_out_bytes() {
    return static_cast<uint64_t>(kMaxRows) * kFeatures * sizeof(uint16_t);
}

uint64_t act_offset() { return rms_out_bytes(); }

uint64_t workspace_bytes() {
    return act_offset() + act_layout().total_bytes;
}

void build_program(rt::Program& program) {
    const auto& layout = act_layout();
    program = rt::Program{};
    program.row_bucket = rt::RowBucket::R64;
    program.execution_class = rt::ExecutionClass::DECODE;
    program.workspace.total_bytes = workspace_bytes();
    program.workspace.ranges.push_back(
        rt::WorkspaceRange{act_offset(), layout.total_bytes, 0u, 0u});
    program.weight_slot_count = 1u;
    program.parameter_slot_count = 1u;

    rt::ValueBinding in{};
    in.logical_value = rt::ValueId{0u};
    in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    in.dtype = rt::ValueDType::BF16;
    in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    in.slot = 0u;
    in.feature_count = kFeatures;
    in.row_stride = kFeatures;
    program.values.push_back(in);

    rt::ValueBinding rms_out{};
    rms_out.logical_value = rt::ValueId{1u};
    rms_out.storage = rt::ValueStorage::WORKSPACE;
    rms_out.dtype = rt::ValueDType::BF16;
    rms_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    rms_out.slot = 0u;
    rms_out.feature_count = kFeatures;
    rms_out.row_stride = kFeatures;
    rms_out.offset = 0u;
    rms_out.bytes = rms_out_bytes();
    program.values.push_back(rms_out);

    rt::ValueBinding out{};
    out.logical_value = rt::ValueId{2u};
    out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    out.dtype = rt::ValueDType::BF16;
    out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    out.slot = 0u;
    out.feature_count = kN;
    out.row_stride = kN;
    program.values.push_back(out);

    rt::DispatchBinding rms{};
    rms.kernel_id = rt::KernelId::RMS_NORM;
    rms.input_count = 1u;
    rms.input_slots[0] = 0u;
    rms.output_count = 1u;
    rms.output_slots[0] = 1u;
    rms.parameter_index0 = 0u;
    rms.group_size = kFeatures;
    rms.scalar_a = 1e-6f;
    program.dispatches.push_back(rms);
    program.dispatch_source_node.push_back(0u);

    rt::DispatchBinding quant{};
    quant.kernel_id = rt::KernelId::ACTIVATION_QUANTIZE_W4A8;
    quant.input_count = 1u;
    quant.input_slots[0] = 1u;
    quant.output_count = 0u;
    quant.group_size = 32u;
    quant.workspace_slot = 0u;
    quant.activation_max_rows = kMaxRows;
    quant.activation_kp = kFeatures;
    quant.activation_code_stride = layout.code_row_stride_bytes;
    quant.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(quant);
    program.dispatch_source_node.push_back(1u);

    rt::DispatchBinding psq4{};
    psq4.kernel_id = rt::KernelId::LINEAR_PSQ4;
    psq4.input_count = 0u;
    psq4.output_count = 1u;
    psq4.output_slots[0] = 2u;
    psq4.weight_index = 0u;
    psq4.workspace_slot = 0u;
    psq4.compute_spec = 3u;
    psq4.activation_max_rows = kMaxRows;
    psq4.activation_kp = kFeatures;
    psq4.activation_code_stride = layout.code_row_stride_bytes;
    psq4.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(psq4);
    program.dispatch_source_node.push_back(1u);
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

bool rows_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                uint32_t rows, size_t row_bytes) {
    for (uint32_t r = 0u; r < rows; ++r) {
        const size_t off = static_cast<size_t>(r) * row_bytes;
        if (std::memcmp(a.data() + off, b.data() + off, row_bytes) != 0)
            return false;
    }
    return true;
}

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

    rt::Program program;
    build_program(program);
    check(program.dispatches.size() == 3u, "program has three dispatches");

    const size_t out_bytes = static_cast<size_t>(kN) * sizeof(uint16_t);
    const size_t out_total = static_cast<size_t>(kMaxRows) * out_bytes;
    const size_t input_bytes =
        static_cast<size_t>(kMaxRows) * kFeatures * sizeof(uint16_t);
    const size_t rms_weight_bytes = static_cast<size_t>(kFeatures) * sizeof(uint16_t);
    const size_t w_codes_bytes = static_cast<size_t>(kTiles) * kKp * 8u;
    const size_t w_scales_bytes = static_cast<size_t>(kTiles) * kNb * 32u;
    const size_t ws_bytes = static_cast<size_t>(workspace_bytes());

    void* input_dev = nullptr;
    void* rms_weight_dev = nullptr;
    void* w_codes_dev = nullptr;
    void* w_scales_dev = nullptr;
    void* ws_host = nullptr;
    void* ws_mcu = nullptr;
    void* out_host = nullptr;
    void* out_mcu = nullptr;
    if (hipMalloc(&input_dev, input_bytes) != hipSuccess ||
        hipMalloc(&rms_weight_dev, rms_weight_bytes) != hipSuccess ||
        hipMalloc(&w_codes_dev, w_codes_bytes) != hipSuccess ||
        hipMalloc(&w_scales_dev, w_scales_bytes) != hipSuccess ||
        hipMalloc(&ws_host, ws_bytes) != hipSuccess ||
        hipMalloc(&ws_mcu, ws_bytes) != hipSuccess ||
        hipMalloc(&out_host, out_total) != hipSuccess ||
        hipMalloc(&out_mcu, out_total) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }

    {
        std::vector<uint16_t> in(static_cast<size_t>(kMaxRows) * kFeatures, 0u);
        for (uint32_t r = 0u; r < kMaxRows; ++r) {
            for (uint32_t c = 0u; c < kFeatures; ++c) {
                const uint32_t h = (r * 131u + c) * 2654435761u + 12345u;
                const int base = static_cast<int>(h % 2001u) - 1000;
                in[static_cast<size_t>(r) * kFeatures + c] =
                    chain::bf16_from_float(static_cast<float>(base) / 1000.0f).data;
            }
        }
        std::vector<uint16_t> wt(kFeatures, 0u);
        for (uint32_t c = 0u; c < kFeatures; ++c) {
            const uint32_t h = c * 104729u + 11u;
            const int base = static_cast<int>(h % 2001u) - 1000;
            wt[c] = chain::bf16_from_float(static_cast<float>(base) / 1000.0f).data;
        }
        uint32_t s = 0x1234567u + kKp * 131u + kN;
        std::vector<uint8_t> codes(w_codes_bytes);
        for (auto& v : codes) v = chain::code_byte(s);
        std::vector<uint8_t> scales(w_scales_bytes, 0u);
        for (size_t i = 0u; i + 1u < scales.size(); i += 2u) {
            const uint16_t v = chain::scale_bf16(s);
            scales[i] = static_cast<uint8_t>(v & 0xFFu);
            scales[i + 1u] = static_cast<uint8_t>(v >> 8u);
        }
        (void)hipMemcpy(input_dev, in.data(), input_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(rms_weight_dev, wt.data(), rms_weight_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(w_codes_dev, codes.data(), w_codes_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(w_scales_dev, scales.data(), w_scales_bytes,
                        hipMemcpyHostToDevice);
    }
    (void)hipMemset(ws_host, 0xAB, ws_bytes);
    (void)hipMemset(ws_mcu, 0xAB, ws_bytes);
    (void)hipMemset(out_host, 0xAB, out_total);
    (void)hipMemset(out_mcu, 0xAB, out_total);

    rt::StaticParameterSlot params[1];
    params[0].device_ptr = rms_weight_dev;
    params[0].elements = kFeatures;
    params[0].dtype = rt::ValueDType::BF16;

    rt::WeightSlot weights[1];
    std::memset(weights, 0, sizeof(weights));
    weights[0].rows = kN;
    weights[0].cols = kFeatures;
    weights[0].k_padded = kKp;
    weights[0].encoding = 3u;
    weights[0].compute_spec = 3u;
    weights[0].codes = w_codes_dev;
    weights[0].scales = w_scales_dev;
    weights[0].weight_scale_group = 32u;
    weights[0].storage_scale_stride_bytes = chain::kWeightScaleStride;
    weights[0].preshuffled = true;

    const void* ext_inputs[1] = {input_dev};
    void* ext_out_host[1] = {out_host};
    void* ext_out_mcu[1] = {out_mcu};

    q35::HostExecutionContext ctx_host{};
    ctx_host.workspace = static_cast<uint8_t*>(ws_host);
    ctx_host.workspace_bytes = ws_bytes;
    ctx_host.external_inputs = ext_inputs;
    ctx_host.external_input_count = 1u;
    ctx_host.external_outputs = ext_out_host;
    ctx_host.external_output_count = 1u;
    ctx_host.host_parameters = params;
    ctx_host.host_parameter_count = 1u;
    ctx_host.weights = weights;
    ctx_host.weight_count = 1u;
    ctx_host.actual_rows = kMaxRows;
    ctx_host.actual_outputs = 1u;

    q35::HostExecutionContext ctx_mcu = ctx_host;
    ctx_mcu.workspace = static_cast<uint8_t*>(ws_mcu);
    ctx_mcu.external_outputs = ext_out_mcu;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }
    const auto r1 =
        q35::try_launch_rmsnorm(program, program.dispatches[0], ctx_host, stream);
    const auto r2 = q35::try_launch_activation_quantize_a8(
        program, program.dispatches[1], ctx_host, stream);
    const auto r3 =
        q35::try_launch_linear_psq4(program, program.dispatches[2], ctx_host, stream);
    if (!(r1.ok() && r1.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r2.ok() && r2.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r3.ok() && r3.value().kind == q35::OptimizedLaunchKind::Launched) ||
        hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: host reference chain\n");
        return 1;
    }

    q35::McuPlanCompileOptions opts{};
    opts.dispatch_begin = 0u;
    opts.dispatch_end = 3u;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const ps::Status plan_status = q35::compile_mcu_plan(program, ctx_mcu, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    check(plan.physical_dispatch_count == 3u, "three physical dispatches");
    check(plan.psq4_multi.size() == 1u, "row-block linear invocation emitted");

    q35::McuDecodeRuntimeOptions options{};
    options.device = 0;
    options.record_entries = 0u;
    options.driver_mode = q35::McuPlanDriverMode::ExternalPersistentMcu;
    auto runtime_result = q35::McuDecodeRuntime::create(options);
    if (!runtime_result.ok()) {
        std::printf("FAIL: runtime create: %s\n",
                    runtime_result.status().message().c_str());
        return 1;
    }
    q35::McuDecodeRuntime runtime = runtime_result.release();
    const ps::Status prepared = runtime.prepare_plan(plan);
    if (!prepared.ok()) {
        std::printf("FAIL: prepare_plan: %s\n", prepared.message().c_str());
        return 1;
    }
    check(runtime.row_global_patch_count() == 0u,
          "row-block rows are fixed by the prepared plan");

    rt::DeviceBatchContext* ctx_dev = nullptr;
    rt::DeviceRequestDescriptor* req_dev = nullptr;
    uint32_t* row_seq_dev = nullptr;
    uint32_t* row_pos_dev = nullptr;
    uint64_t* ready_dev = nullptr;
    uint64_t* plan_epoch_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&ctx_dev),
                  sizeof(rt::DeviceBatchContext)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&req_dev),
                  sizeof(rt::DeviceRequestDescriptor)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_seq_dev),
                  kMaxRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_pos_dev),
                  kMaxRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&plan_epoch_dev), sizeof(uint64_t)) !=
            hipSuccess) {
        std::printf("FAIL: context alloc\n");
        return 1;
    }
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(plan_epoch_dev, 0, sizeof(uint64_t));
    (void)hipMemset(row_seq_dev, 0, kMaxRows * sizeof(uint32_t));
    (void)hipMemset(row_pos_dev, 0, kMaxRows * sizeof(uint32_t));

    rt::DeviceRequestDescriptor descriptor{};
    descriptor.execution_class = rt::ExecutionClass::DECODE;
    descriptor.request_handle = rt::RequestHandle{0u, 1u};
    descriptor.row_begin = 0u;
    descriptor.row_count = kMaxRows;
    (void)hipMemcpy(req_dev, &descriptor, sizeof(descriptor),
                    hipMemcpyHostToDevice);
    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = kMaxRows;
    ctx.num_requests = 1u;
    ctx.num_decode_requests = 1u;
    ctx.requests = req_dev;
    ctx.row_sequence_slots = row_seq_dev;
    ctx.row_positions = row_pos_dev;
    (void)hipMemcpy(ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice);

    hipStream_t persistent_stream = nullptr;
    hipStream_t update_stream = nullptr;
    if (hipStreamCreate(&persistent_stream) != hipSuccess ||
        hipStreamCreateWithFlags(&update_stream, hipStreamNonBlocking) !=
            hipSuccess) {
        std::printf("FAIL: persistent stream\n");
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
                                    kMaxRows, kMaxRows, plan_epoch_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev, nullptr,
                                   0u, nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent
              .configure_execution_patches(runtime.row_global_patches(),
                                           runtime.row_global_patch_count())
              .ok(),
          "configure execution patches");
    check(persistent.execution_patch_count() == 0u, "patch table recorded");

    uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent start");
    check(wait_for(persistent, 1u, 10000u), "full-row loop completed");

    std::vector<uint8_t> host_out(out_total, 0u);
    std::vector<uint8_t> mcu_out(out_total, 0u);
    (void)hipMemcpyAsync(host_out.data(), out_host, out_total,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipMemcpyAsync(mcu_out.data(), out_mcu, out_total,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipStreamSynchronize(update_stream);
    {
        size_t first = 0u;
        size_t diffs = 0u;
        for (size_t i = 0u; i < out_total; ++i) {
            if (host_out[i] == mcu_out[i]) continue;
            if (diffs == 0u) first = i;
            ++diffs;
        }
        if (diffs != 0u)
            std::printf("  out diff count=%zu first=%zu\n", diffs, first);
    }
    check(rows_equal(host_out, mcu_out, kMaxRows, out_bytes),
          "26-row chain host vs persistent mcu byte exact");

    ctx.actual_rows = kLoopRows;
    ready = 2u;
    (void)hipMemcpyAsync(ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice,
                         update_stream);
    (void)hipMemcpyAsync(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice,
                         update_stream);
    (void)hipStreamSynchronize(update_stream);
    check(wait_for(persistent, 2u, 10000u), "reduced-row loop completed");

    std::fill(mcu_out.begin(), mcu_out.end(), 0u);
    (void)hipMemcpyAsync(mcu_out.data(), out_mcu, out_total,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipStreamSynchronize(update_stream);
    check(rows_equal(host_out, mcu_out, kLoopRows, out_bytes),
          "reduced rows match the reference without re-prepare");
    check(persistent.batches_failed() == 0u, "no fault");
    check(runtime.execution_state()->plans_started >= 2u,
          "prepared fsm served both loops");
    std::printf("multirow chain: rows=%u->%u patches=%u plans=%llu\n", kMaxRows,
                kLoopRows, runtime.row_global_patch_count(),
                static_cast<unsigned long long>(
                    runtime.execution_state()->plans_started));

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");
    check(runtime.shutdown().ok(), "runtime shutdown");

    (void)hipFree(plan_epoch_dev);
    (void)hipFree(ready_dev);
    (void)hipFree(row_pos_dev);
    (void)hipFree(row_seq_dev);
    (void)hipFree(req_dev);
    (void)hipFree(ctx_dev);
    check(hipStreamDestroy(update_stream) == hipSuccess, "update stream destroy");
    check(hipStreamDestroy(persistent_stream) == hipSuccess,
          "persistent stream destroy");
    (void)hipStreamDestroy(stream);
    (void)hipFree(out_mcu);
    (void)hipFree(out_host);
    (void)hipFree(ws_mcu);
    (void)hipFree(ws_host);
    (void)hipFree(w_scales_dev);
    (void)hipFree(w_codes_dev);
    (void)hipFree(rms_weight_dev);
    (void)hipFree(input_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_multirow_chain_bridge: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_multirow_chain_bridge: PASS\n");
    return 0;
}
