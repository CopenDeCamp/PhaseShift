#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>
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
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kRows = 16u;
constexpr uint32_t kExternalOutputCount = 6u;
constexpr uint32_t kTapA = 4u;
constexpr uint32_t kTapB = 5u;

const rt::Int8ActivationWorkspaceLayout& act_layout() {
    static const rt::Int8ActivationWorkspaceLayout l =
        rt::Int8ActivationWorkspaceLayout::make(chain::kFeatures, kRows);
    return l;
}

uint64_t workspace_bytes() {
    return act_layout().total_bytes;
}

void build_program(rt::Program& program) {
    const auto& layout = act_layout();
    program = rt::Program{};
    program.row_bucket = rt::RowBucket::R16;
    program.execution_class = rt::ExecutionClass::DECODE;
    program.workspace.total_bytes = workspace_bytes();
    program.workspace.ranges.push_back(
        rt::WorkspaceRange{0u, layout.total_bytes, 0u, 0u});
    program.weight_slot_count = 1u;
    program.parameter_slot_count = 1u;
    program.external_output_count = kExternalOutputCount;

    rt::ValueBinding input{};
    input.logical_value = rt::ValueId{0u};
    input.storage = rt::ValueStorage::EXTERNAL_INPUT;
    input.dtype = rt::ValueDType::BF16;
    input.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    input.slot = 0u;
    input.feature_count = chain::kFeatures;
    input.row_stride = chain::kFeatures;
    program.values.push_back(input);

    rt::ValueBinding tap_a{};
    tap_a.logical_value = rt::ValueId{1u};
    tap_a.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    tap_a.dtype = rt::ValueDType::BF16;
    tap_a.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    tap_a.slot = kTapA;
    tap_a.feature_count = chain::kFeatures;
    tap_a.row_stride = chain::kFeatures;
    program.values.push_back(tap_a);

    rt::ValueBinding tap_b{};
    tap_b.logical_value = rt::ValueId{2u};
    tap_b.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    tap_b.dtype = rt::ValueDType::BF16;
    tap_b.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    tap_b.slot = kTapB;
    tap_b.feature_count = chain::kFeatures;
    tap_b.row_stride = chain::kFeatures;
    program.values.push_back(tap_b);

    rt::ValueBinding final_out{};
    final_out.logical_value = rt::ValueId{3u};
    final_out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    final_out.dtype = rt::ValueDType::BF16;
    final_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    final_out.slot = 0u;
    final_out.feature_count = chain::kN;
    final_out.row_stride = chain::kN;
    program.values.push_back(final_out);

    rt::DispatchBinding rms{};
    rms.kernel_id = rt::KernelId::RMS_NORM;
    rms.input_count = 1u;
    rms.input_slots[0] = 0u;
    rms.output_count = 1u;
    rms.output_slots[0] = 1u;
    rms.parameter_index0 = 0u;
    rms.group_size = chain::kFeatures;
    rms.scalar_a = 1e-6f;
    program.dispatches.push_back(rms);
    program.dispatch_source_node.push_back(0u);

    rt::DispatchBinding silu{};
    silu.kernel_id = rt::KernelId::RMS_NORM;
    silu.input_count = 1u;
    silu.input_slots[0] = 0u;
    silu.output_count = 1u;
    silu.output_slots[0] = 2u;
    silu.parameter_index0 = 0u;
    silu.group_size = chain::kFeatures;
    silu.scalar_a = 1e-6f;
    program.dispatches.push_back(silu);
    program.dispatch_source_node.push_back(0u);

    rt::DispatchBinding quant{};
    quant.kernel_id = rt::KernelId::ACTIVATION_QUANTIZE_W4A8;
    quant.input_count = 1u;
    quant.input_slots[0] = 1u;
    quant.output_count = 0u;
    quant.group_size = 32u;
    quant.workspace_slot = 0u;
    quant.activation_max_rows = kRows;
    quant.activation_kp = chain::kFeatures;
    quant.activation_code_stride = layout.code_row_stride_bytes;
    quant.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(quant);
    program.dispatch_source_node.push_back(1u);

    rt::DispatchBinding psq8{};
    psq8.kernel_id = rt::KernelId::LINEAR_PSQ8;
    psq8.input_count = 0u;
    psq8.output_count = 1u;
    psq8.output_slots[0] = 3u;
    psq8.weight_index = 0u;
    psq8.workspace_slot = 0u;
    psq8.compute_spec = 4u;
    psq8.activation_max_rows = kRows;
    psq8.activation_kp = chain::kFeatures;
    psq8.activation_code_stride = layout.code_row_stride_bytes;
    psq8.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(psq8);
    program.dispatch_source_node.push_back(2u);
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

bool bytes_match(const std::vector<uint8_t>& host,
                 const std::vector<uint8_t>& mcu_bytes, const char* label) {
    if (host.size() != mcu_bytes.size()) {
        std::printf("FAIL: %s size mismatch\n", label);
        ++g_failed;
        return false;
    }
    for (size_t i = 0u; i < host.size(); ++i) {
        if (host[i] == mcu_bytes[i]) continue;
        std::printf("FAIL: %s differs at %zu (host=%u mcu=%u)\n", label, i,
                    host[i], mcu_bytes[i]);
        ++g_failed;
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
    check(program.external_output_count == kExternalOutputCount,
          "program carries six external outputs");
    check(program.validate().ok(), "program validates");
    check(program.dispatches.size() == 4u, "program has four dispatches");

    const size_t tap_bytes_a =
        static_cast<size_t>(kRows) * chain::kFeatures * sizeof(uint16_t);
    const size_t tap_bytes_b =
        static_cast<size_t>(kRows) * chain::kFeatures * sizeof(uint16_t);
    const size_t final_bytes =
        static_cast<size_t>(kRows) * chain::kN * sizeof(uint16_t);
    const size_t input_bytes =
        static_cast<size_t>(kRows) * chain::kFeatures * sizeof(uint16_t);
    const size_t rms_weight_bytes = chain::kFeatures * sizeof(uint16_t);
    const size_t w_codes_bytes =
        static_cast<size_t>(chain::kTiles) * chain::kKp * 16u;
    const size_t w_scales_bytes =
        static_cast<size_t>(chain::kTiles) * chain::kNb * 32u;
    const size_t ws_bytes = static_cast<size_t>(workspace_bytes());

    void* input_dev = nullptr;
    void* rms_weight_dev = nullptr;
    void* w_codes_dev = nullptr;
    void* w_scales_dev = nullptr;
    void* ws_host = nullptr;
    void* ws_mcu = nullptr;
    void* final_host = nullptr;
    void* final_mcu = nullptr;
    void* tap_host_a = nullptr;
    void* tap_host_b = nullptr;
    void* tap_mcu_a = nullptr;
    void* tap_mcu_b = nullptr;
    if (hipMalloc(&input_dev, input_bytes) != hipSuccess ||
        hipMalloc(&rms_weight_dev, rms_weight_bytes) != hipSuccess ||
        hipMalloc(&w_codes_dev, w_codes_bytes) != hipSuccess ||
        hipMalloc(&w_scales_dev, w_scales_bytes) != hipSuccess ||
        hipMalloc(&ws_host, ws_bytes) != hipSuccess ||
        hipMalloc(&ws_mcu, ws_bytes) != hipSuccess ||
        hipMalloc(&final_host, final_bytes) != hipSuccess ||
        hipMalloc(&final_mcu, final_bytes) != hipSuccess ||
        hipMalloc(&tap_host_a, tap_bytes_a) != hipSuccess ||
        hipMalloc(&tap_host_b, tap_bytes_b) != hipSuccess ||
        hipMalloc(&tap_mcu_a, tap_bytes_a) != hipSuccess ||
        hipMalloc(&tap_mcu_b, tap_bytes_b) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }

    {
        std::vector<uint16_t> in(static_cast<size_t>(kRows) * chain::kFeatures,
                                 0u);
        for (uint32_t r = 0u; r < kRows; ++r) {
            for (uint32_t c = 0u; c < chain::kFeatures; ++c) {
                const uint32_t h = (r * 131u + c) * 2654435761u + 12345u;
                const int base = static_cast<int>(h % 2001u) - 1000;
                in[static_cast<size_t>(r) * chain::kFeatures + c] =
                    chain::bf16_from_float(static_cast<float>(base) / 1000.0f)
                        .data;
            }
        }
        (void)hipMemcpy(input_dev, in.data(), input_bytes,
                        hipMemcpyHostToDevice);

        std::vector<uint16_t> wt(chain::kFeatures, 0u);
        for (uint32_t c = 0u; c < chain::kFeatures; ++c) {
            const uint32_t h = c * 104729u + 11u;
            const int base = static_cast<int>(h % 2001u) - 1000;
            wt[c] = chain::bf16_from_float(static_cast<float>(base) / 1000.0f)
                        .data;
        }
        (void)hipMemcpy(rms_weight_dev, wt.data(), rms_weight_bytes,
                        hipMemcpyHostToDevice);

        uint32_t s = 0x51A7u + chain::kKp * 131u + chain::kN;
        std::vector<uint8_t> codes(w_codes_bytes);
        for (auto& v : codes) v = chain::code_byte(s);
        std::vector<uint8_t> scales(w_scales_bytes, 0u);
        for (size_t i = 0u; i + 1u < scales.size(); i += 2u) {
            const uint16_t v = chain::scale_bf16(s);
            scales[i] = static_cast<uint8_t>(v & 0xFFu);
            scales[i + 1u] = static_cast<uint8_t>(v >> 8u);
        }
        (void)hipMemcpy(w_codes_dev, codes.data(), w_codes_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(w_scales_dev, scales.data(), w_scales_bytes,
                        hipMemcpyHostToDevice);
    }
    (void)hipMemset(ws_host, 0xAB, ws_bytes);
    (void)hipMemset(ws_mcu, 0xAB, ws_bytes);
    (void)hipMemset(final_host, 0xAB, final_bytes);
    (void)hipMemset(final_mcu, 0xAB, final_bytes);
    (void)hipMemset(tap_host_a, 0xAB, tap_bytes_a);
    (void)hipMemset(tap_host_b, 0xAB, tap_bytes_b);
    (void)hipMemset(tap_mcu_a, 0xAB, tap_bytes_a);
    (void)hipMemset(tap_mcu_b, 0xAB, tap_bytes_b);

    rt::StaticParameterSlot params[1];
    params[0].device_ptr = rms_weight_dev;
    params[0].elements = chain::kFeatures;
    params[0].dtype = rt::ValueDType::BF16;

    rt::WeightSlot weights[1];
    std::memset(weights, 0, sizeof(weights));
    weights[0].rows = chain::kN;
    weights[0].cols = chain::kFeatures;
    weights[0].k_padded = chain::kKp;
    weights[0].encoding = 4u;
    weights[0].compute_spec = 3u;
    weights[0].codes = w_codes_dev;
    weights[0].scales = w_scales_dev;
    weights[0].weight_scale_group = 32u;
    weights[0].storage_scale_stride_bytes = chain::kWeightScaleStride;
    weights[0].preshuffled = true;

    const void* ext_inputs[1] = {input_dev};
    void* ext_out_host[kExternalOutputCount] = {};
    void* ext_out_mcu[kExternalOutputCount] = {};
    ext_out_host[0] = final_host;
    ext_out_host[kTapA] = tap_host_a;
    ext_out_host[kTapB] = tap_host_b;
    ext_out_mcu[0] = final_mcu;
    ext_out_mcu[kTapA] = tap_mcu_a;
    ext_out_mcu[kTapB] = tap_mcu_b;

    q35::HostExecutionContext ctx_host{};
    ctx_host.workspace = static_cast<uint8_t*>(ws_host);
    ctx_host.workspace_bytes = ws_bytes;
    ctx_host.external_inputs = ext_inputs;
    ctx_host.external_input_count = 1u;
    ctx_host.external_outputs = ext_out_host;
    ctx_host.external_output_count = kExternalOutputCount;
    ctx_host.host_parameters = params;
    ctx_host.host_parameter_count = 1u;
    ctx_host.weights = weights;
    ctx_host.weight_count = 1u;
    ctx_host.actual_rows = kRows;
    ctx_host.actual_outputs = 1u;

    q35::HostExecutionContext ctx_mcu = ctx_host;
    ctx_mcu.workspace = static_cast<uint8_t*>(ws_mcu);
    ctx_mcu.external_outputs = ext_out_mcu;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }
    const auto h1 = q35::try_launch_rmsnorm(program, program.dispatches[0],
                                            ctx_host, stream);
    const auto h2 = q35::try_launch_rmsnorm(program, program.dispatches[1],
                                            ctx_host, stream);
    const auto h3 = q35::try_launch_activation_quantize_a8(
        program, program.dispatches[2], ctx_host, stream);
    const auto h4 =
        q35::try_launch_linear_psq8(program, program.dispatches[3], ctx_host,
                                    stream);
    const bool host_ok =
        h1.ok() && h1.value().kind == q35::OptimizedLaunchKind::Launched &&
        h2.ok() && h2.value().kind == q35::OptimizedLaunchKind::Launched &&
        h3.ok() && h3.value().kind == q35::OptimizedLaunchKind::Launched &&
        h4.ok() && h4.value().kind == q35::OptimizedLaunchKind::Launched;
    if (!host_ok) {
        const auto dump = [](const char* name, const auto& r) {
            if (!r.ok()) {
                std::printf("HOST_FAIL %s: %s\n", name,
                            r.status().message().c_str());
                return;
            }
            std::printf("HOST_FAIL %s: kind=%d\n", name,
                        static_cast<int>(r.value().kind));
        };
        dump("rms", h1);
        dump("rms_tap_b", h2);
        dump("quant", h3);
        dump("psq8", h4);
    }
    check(host_ok, "host chain launched");
    if (!host_ok || hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: host chain\n");
        return 1;
    }

    q35::McuPlanCompileOptions options{};
    options.dispatch_begin = 0u;
    options.dispatch_end = 4u;
    options.marker_grid = 1u;
    options.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const q35::McuStaticPlanCompileContext ctx_mcu_static =
        q35::make_static_plan_compile_context(
            ctx_mcu, ctx_mcu.actual_rows, ctx_mcu.actual_outputs,
            ctx_mcu.staged_request_count,
            ctx_mcu.actual_stochastic_outputs);
    const ps::Status plan_status =
        q35::compile_mcu_plan(program,ctx_mcu_static, options, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    check(plan.dispatch_count == 4u, "plan carries four dispatches");
    check(plan.rms.size() == 2u && !plan.psq4_multi.empty(),
          "plan carries two rms and one linear invocation");

    q35::McuDecodeRuntimeOptions rt_options{};
    rt_options.device = 0;
    rt_options.record_entries = 0u;
    rt_options.driver_mode = q35::McuPlanDriverMode::ExternalPersistentMcu;
    auto runtime_result = q35::McuDecodeRuntime::create(rt_options);
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

    rt::DeviceRequestDescriptor* req_dev = nullptr;
    rt::DeviceBatchContext* ctx_dev = nullptr;
    uint32_t* row_seq_dev = nullptr;
    uint32_t* row_pos_dev = nullptr;
    uint64_t* ready_dev = nullptr;
    uint64_t* plan_epoch_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&req_dev),
                  sizeof(rt::DeviceRequestDescriptor)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ctx_dev),
                  sizeof(rt::DeviceBatchContext)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_seq_dev),
                  kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_pos_dev),
                  kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&plan_epoch_dev),
                  sizeof(uint64_t)) != hipSuccess) {
        std::printf("FAIL: context alloc\n");
        return 1;
    }
    (void)hipMemset(row_seq_dev, 0, kRows * sizeof(uint32_t));
    (void)hipMemset(row_pos_dev, 0, kRows * sizeof(uint32_t));
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(plan_epoch_dev, 0, sizeof(uint64_t));

    rt::DeviceRequestDescriptor descriptor{};
    descriptor.execution_class = rt::ExecutionClass::DECODE;
    descriptor.request_handle = rt::RequestHandle{0u, 1u};
    descriptor.row_begin = 0u;
    descriptor.row_count = kRows;
    (void)hipMemcpy(req_dev, &descriptor, sizeof(descriptor),
                    hipMemcpyHostToDevice);
    rt::DeviceBatchContext dev_ctx{};
    dev_ctx.actual_rows = kRows;
    dev_ctx.num_requests = 1u;
    dev_ctx.num_decode_requests = 1u;
    dev_ctx.requests = req_dev;
    dev_ctx.row_sequence_slots = row_seq_dev;
    dev_ctx.row_positions = row_pos_dev;
    (void)hipMemcpy(ctx_dev, &dev_ctx, sizeof(dev_ctx), hipMemcpyHostToDevice);

    hipStream_t persistent_stream = nullptr;
    if (hipStreamCreate(&persistent_stream) != hipSuccess) {
        std::printf("FAIL: persistent stream\n");
        return 1;
    }
    auto persistent_result =
        mcu::GpuMcuPersistentMcu::create(0, persistent_stream);
    if (!persistent_result.ok()) {
        std::printf("FAIL: persistent create: %s\n",
                    persistent_result.status().message().c_str());
        return 1;
    }
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();
    check(persistent
              .configure_batch_plan(ctx_dev, req_dev, row_seq_dev, row_pos_dev,
                                    1u, kRows, kRows, plan_epoch_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev,
                                   nullptr, 0u, nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent
              .configure_execution_patches(runtime.row_global_patches(),
                                           runtime.row_global_patch_count())
              .ok(),
          "configure execution patches");

    uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent start");
    check(wait_for(persistent, 1u, 10000u), "mcu loop completed");

    hipStream_t update_stream = nullptr;
    if (hipStreamCreateWithFlags(&update_stream, hipStreamNonBlocking) !=
        hipSuccess) {
        std::printf("FAIL: update stream\n");
        return 1;
    }
    std::vector<uint8_t> final_h(final_bytes, 0u);
    std::vector<uint8_t> final_m(final_bytes, 0u);
    std::vector<uint8_t> tap_h_a(tap_bytes_a, 0u);
    std::vector<uint8_t> tap_m_a(tap_bytes_a, 0u);
    std::vector<uint8_t> tap_h_b(tap_bytes_b, 0u);
    std::vector<uint8_t> tap_m_b(tap_bytes_b, 0u);
    (void)hipMemcpyAsync(final_h.data(), final_host, final_bytes,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipMemcpyAsync(final_m.data(), final_mcu, final_bytes,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipMemcpyAsync(tap_h_a.data(), tap_host_a, tap_bytes_a,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipMemcpyAsync(tap_m_a.data(), tap_mcu_a, tap_bytes_a,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipMemcpyAsync(tap_h_b.data(), tap_host_b, tap_bytes_b,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipMemcpyAsync(tap_m_b.data(), tap_mcu_b, tap_bytes_b,
                         hipMemcpyDeviceToHost, update_stream);
    (void)hipStreamSynchronize(update_stream);

    bytes_match(tap_h_a, tap_m_a, "tap A (slot 4) host vs mcu");
    bytes_match(tap_h_b, tap_m_b, "tap B (slot 5) host vs mcu");
    bytes_match(final_h, final_m, "final output host vs mcu");

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");

    (void)hipFree(req_dev);
    (void)hipFree(ctx_dev);
    (void)hipFree(row_seq_dev);
    (void)hipFree(row_pos_dev);
    (void)hipFree(ready_dev);
    (void)hipFree(plan_epoch_dev);
    (void)hipStreamDestroy(persistent_stream);
    (void)hipStreamDestroy(update_stream);
    (void)hipStreamDestroy(stream);
    (void)hipFree(input_dev);
    (void)hipFree(rms_weight_dev);
    (void)hipFree(w_codes_dev);
    (void)hipFree(w_scales_dev);
    (void)hipFree(ws_host);
    (void)hipFree(ws_mcu);
    (void)hipFree(final_host);
    (void)hipFree(final_mcu);
    (void)hipFree(tap_host_a);
    (void)hipFree(tap_host_b);
    (void)hipFree(tap_mcu_a);
    (void)hipFree(tap_mcu_b);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_hidden_tap_plan: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_hidden_tap_plan: PASS\n");
    return 0;
}
