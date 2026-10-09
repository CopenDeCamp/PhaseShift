#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_program_chain_fixture.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;
namespace w4 = gpu_mcu_w4a8;
namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
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

    chain::Workspace feed;
    if (!feed.allocate()) {
        std::printf("FAIL: feed alloc\n");
        return 1;
    }
    feed.fill_inputs();

    w4::W4a8Buffers ws_host;
    w4::W4a8Buffers ws_mcu;
    if (!ws_host.allocate() || !ws_mcu.allocate()) {
        std::printf("FAIL: workspace alloc\n");
        return 1;
    }
    ws_host.poison();
    ws_mcu.poison();

    rt::Program program;
    w4::build_w4a8_program(program);
    check(program.validate().ok(), "program validates");
    check(program.dispatches.size() == 3u, "program has three dispatches");
    check(program.dispatch_source_node.size() == program.dispatches.size(),
          "program carries dispatch provenance");

    rt::StaticParameterSlot params[1];
    rt::WeightSlot weights[1];
    w4::fill_w4a8_weights(params, weights, feed.rms_weight, feed.w_codes,
                          feed.w_scales);

    const void* ext_inputs[1] = {feed.input};
    void* ext_out_host[1] = {ws_host.output};
    void* ext_out_mcu[1] = {ws_mcu.output};
    q35::HostExecutionContext ctx_host;
    q35::HostExecutionContext ctx_mcu;
    w4::bind_w4a8_context(ctx_host, ws_host.workspace, ext_inputs, ext_out_host,
                          params, weights);
    w4::bind_w4a8_context(ctx_mcu, ws_mcu.workspace, ext_inputs, ext_out_mcu,
                          params, weights);

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    const auto r1 = q35::try_launch_rmsnorm(program, program.dispatches[0],
                                            ctx_host, stream);
    check(r1.ok() && r1.value().kind == q35::OptimizedLaunchKind::Launched,
          "host rmsnorm launched");
    const auto r2 = q35::try_launch_activation_quantize_a8(
        program, program.dispatches[1], ctx_host, stream);
    check(r2.ok() && r2.value().kind == q35::OptimizedLaunchKind::Launched,
          "host activation quantize launched");
    const auto r3 = q35::try_launch_linear_psq4(program, program.dispatches[2],
                                                ctx_host, stream);
    check(r3.ok() && r3.value().kind == q35::OptimizedLaunchKind::Launched,
          "host psq4 launched");
    check(hipStreamSynchronize(stream) == hipSuccess, "host chain sync");

    q35::McuPlanCompileOptions options;
    options.dispatch_begin = 0u;
    options.dispatch_end = 3u;
    options.marker_grid = 1u;
    options.marker_workgroup = 64u;

    q35::McuCompiledPlan compiled;
    const q35::McuStaticPlanCompileContext ctx_mcu_static =
        q35::make_static_plan_compile_context(
            ctx_mcu, ctx_mcu.actual_rows, ctx_mcu.actual_outputs,
            ctx_mcu.staged_request_count,
            ctx_mcu.actual_stochastic_outputs);
    const ps::Status compiled_status =
        q35::compile_mcu_plan(program,ctx_mcu_static, options, compiled);
    if (!compiled_status.ok()) {
        std::printf("FAIL: compile: %s\n", compiled_status.message().c_str());
        ++g_failed;
    }
    check(compiled.dispatch_count == 3u, "compiled dispatch count");
    check(compiled.marker_count == 1u, "compiled marker count");
    check(compiled.kernarg_slots_required == 4u, "compiled kernarg slots");
    check(compiled.variants.size() == 4u, "compiled variant count");
    check(compiled.rms.size() == 1u, "one rms invocation");
    check(compiled.quant_e4m3.size() == 1u, "one e4m3 invocation");
    check(compiled.psq4.size() == 1u, "one psq4 invocation");
    check(compiled.nodes.size() == 6u, "compiled node count");

    const std::vector<mcu::McuPlanNode> handwritten =
        w4::make_w4a8_handwritten_nodes();
    check(handwritten.size() == compiled.nodes.size(),
          "handwritten and compiled node counts match");
    if (handwritten.size() == compiled.nodes.size()) {
        const size_t bytes =
            handwritten.size() * sizeof(mcu::McuPlanNode);
        check(std::memcmp(handwritten.data(), compiled.nodes.data(), bytes) == 0,
              "compiled plan equals the handwritten plan");
        if (std::memcmp(handwritten.data(), compiled.nodes.data(), bytes) != 0) {
            for (size_t i = 0; i < handwritten.size(); ++i) {
                const auto& a = handwritten[i];
                const auto& b = compiled.nodes[i];
                if (std::memcmp(&a, &b, sizeof(a)) != 0) {
                    std::printf(
                        "  node[%zu] differs: hw(v=%u r=%u f=%u inv=%u n=%u) "
                        "compiled(v=%u r=%u f=%u inv=%u n=%u)\n",
                        i, a.variant_id, a.kernarg_recipe, a.flags,
                        a.invocation_index, a.next, b.variant_id, b.kernarg_recipe,
                        b.flags, b.invocation_index, b.next);
                    break;
                }
            }
        }
    }

    rt::Program unsupported = program;
    rt::DispatchBinding rope{};
    rope.kernel_id = rt::KernelId::ROPE;
    rope.input_count = 1u;
    rope.input_slots[0] = w4::kVInput;
    rope.output_count = 1u;
    rope.output_slots[0] = w4::kVRmsOut;
    unsupported.dispatches.push_back(rope);
    unsupported.dispatch_source_node.push_back(1u);
    q35::McuCompiledPlan rejected;
    const q35::McuStaticPlanCompileContext ctx_mcu_static2 =
        q35::make_static_plan_compile_context(
            ctx_mcu, ctx_mcu.actual_rows, ctx_mcu.actual_outputs,
            ctx_mcu.staged_request_count,
            ctx_mcu.actual_stochastic_outputs);
    const ps::Status unsupported_status = q35::compile_mcu_plan(
        unsupported,ctx_mcu_static2, q35::McuPlanCompileOptions{0u, 4u, 1u, 64u},
        rejected);
    check(!unsupported_status.ok() &&
              unsupported_status.code() == ps::Status::Code::unsupported,
          "unsupported kernel id fails closed with Status::unsupported");
    check(rejected.nodes.empty(), "unsupported compile leaves no partial plan");

    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("test_gpu_mcu_program_plan_three_primitive: setup failed\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata rms_meta{};
    mcu::GpuAqlKernelMetadata quant_meta{};
    mcu::GpuAqlKernelMetadata psq4_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                             kernel::kRmsNormBf16PfOnePlusSymbol, rms_meta)) {
        return 1;
    }
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO,
                             kernel::kActivationQuantizeE4m3K5120Symbol,
                             quant_meta)) {
        return 1;
    }
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_PSQ4_HSACO,
                             kernel::kPsq4Decode1Bf16U16Symbol, psq4_meta)) {
        return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants(compiled.variants.size());
    for (size_t i = 0; i < compiled.variants.size(); ++i) {
        const auto& v = compiled.variants[i];
        switch (v.kind) {
            case q35::McuCompiledVariantKind::CompletionMarker:
                variants[i] = mcu::make_probe_variant(
                    static_cast<uint16_t>(i), rig.meta, v.workgroup, v.grid,
                    mcu::kMcuSystemAqlPolicy);
                break;
            case q35::McuCompiledVariantKind::RmsNormBf16PfOnePlus:
                variants[i] = mcu::make_kernel_variant(
                    static_cast<uint16_t>(i), rms_meta,
                    mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus, v.workgroup,
                    v.grid, mcu::kMcuAgentAqlPolicy);
                break;
            case q35::McuCompiledVariantKind::ActivationQuantizeE4m3K5120:
                variants[i] = mcu::make_kernel_variant(
                    static_cast<uint16_t>(i), quant_meta,
                    mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120,
                    v.workgroup, v.grid, mcu::kMcuAgentAqlPolicy,
                    mcu::AqlHiddenArgsPolicy::None);
                break;
            case q35::McuCompiledVariantKind::Psq4Decode1Bf16U16:
                variants[i] = mcu::make_kernel_variant(
                    static_cast<uint16_t>(i), psq4_meta,
                    mcu::kMcuKernargRecipePsq4Decode1Bf16U16, v.workgroup,
                    v.grid, mcu::kMcuAgentAqlPolicy,
                    mcu::AqlHiddenArgsPolicy::None);
                break;
        }
    }

    if (!rig.upload(compiled.nodes, variants)) return 1;
    if (!rig.set_invocations(compiled.rms.data(),
                             static_cast<uint32_t>(compiled.rms.size()))) {
        return 1;
    }
    if (!rig.set_e4m3_invocations(
            compiled.quant_e4m3.data(),
            static_cast<uint32_t>(compiled.quant_e4m3.size()))) {
        return 1;
    }
    if (!rig.set_psq4_invocations(compiled.psq4.data(),
                                  static_cast<uint32_t>(compiled.psq4.size()))) {
        return 1;
    }
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x50434d50u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = compiled.kernarg_slots_required;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_program_plan_three_primitive: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "compiled plan requested");
    check(rig.fsm.wait_plans(1u, 20000).ok(), "compiled plan completed");
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == compiled.dispatch_count + 1u,
          "dispatch count includes the marker");

    const auto& layout = w4::act_layout();
    const size_t rms_bytes = w4::kRmsOutBytes;
    const size_t codes_bytes =
        static_cast<size_t>(layout.code_row_stride_bytes) * 16u;
    const size_t scale_bytes = layout.scale_row_stride_bytes;
    const size_t out_bytes = 1024u * 2u;
    std::vector<uint8_t> host_rms(rms_bytes);
    std::vector<uint8_t> mcu_rms(rms_bytes);
    std::vector<uint8_t> host_codes(codes_bytes);
    std::vector<uint8_t> mcu_codes(codes_bytes);
    std::vector<uint8_t> host_scales(scale_bytes);
    std::vector<uint8_t> mcu_scales(scale_bytes);
    std::vector<uint8_t> host_out(out_bytes);
    std::vector<uint8_t> mcu_out(out_bytes);

    rig.device_to_host(host_rms.data(),
                       static_cast<uint8_t*>(ws_host.workspace) + w4::kRmsOutOffset,
                       rms_bytes);
    rig.device_to_host(mcu_rms.data(),
                       static_cast<uint8_t*>(ws_mcu.workspace) + w4::kRmsOutOffset,
                       rms_bytes);
    rig.device_to_host(
        host_codes.data(),
        static_cast<uint8_t*>(ws_host.workspace) + w4::kActOffset +
            layout.codes_offset,
        codes_bytes);
    rig.device_to_host(
        mcu_codes.data(),
        static_cast<uint8_t*>(ws_mcu.workspace) + w4::kActOffset +
            layout.codes_offset,
        codes_bytes);
    rig.device_to_host(
        host_scales.data(),
        static_cast<uint8_t*>(ws_host.workspace) + w4::kActOffset +
            layout.scales_offset,
        scale_bytes);
    rig.device_to_host(
        mcu_scales.data(),
        static_cast<uint8_t*>(ws_mcu.workspace) + w4::kActOffset +
            layout.scales_offset,
        scale_bytes);
    rig.device_to_host(host_out.data(), ws_host.output, out_bytes);
    rig.device_to_host(mcu_out.data(), ws_mcu.output, out_bytes);

    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(host_rms.data(), mcu_rms.data(), rms_bytes, &first,
                             &diffs),
          "compiled plan: rms output matches the host path");
    if (diffs) std::printf("  rms diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_codes.data(), mcu_codes.data(), codes_bytes,
                             &first, &diffs),
          "compiled plan: activation codes match the host path");
    if (diffs) std::printf("  codes diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_scales.data(), mcu_scales.data(), scale_bytes,
                             &first, &diffs),
          "compiled plan: activation scales match the host path");
    if (diffs) std::printf("  scales diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_out.data(), mcu_out.data(), out_bytes, &first,
                             &diffs),
          "compiled plan: psq4 output matches the host path");
    if (diffs) std::printf("  out diff count=%zu first=%zu\n", diffs, first);

    std::printf(
        "  compiled plan: dispatch=%u variants=%u slots=%u nodes=%zu\n",
        compiled.dispatch_count,
        static_cast<unsigned>(compiled.variants.size()),
        compiled.kernarg_slots_required, compiled.nodes.size());

    check(rig.close(), "rig teardown");
    (void)hipStreamDestroy(stream);
    ws_host.free_all();
    ws_mcu.free_all();
    feed.free_all();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_program_plan_three_primitive: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_program_plan_three_primitive: PASS\n");
    return 0;
}
