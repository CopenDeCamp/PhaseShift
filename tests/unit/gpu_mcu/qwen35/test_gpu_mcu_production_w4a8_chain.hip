#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_program_chain_fixture.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/wall_clock.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>

#include <algorithm>
#include <chrono>
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

using Clock = std::chrono::steady_clock;

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

    const auto wall_rate = mcu::gpu_wall_clock_rate_khz(0);
    if (!wall_rate.ok()) {
        std::printf("FAIL: wall clock rate: %s\n",
                    wall_rate.status().message().c_str());
        return 1;
    }
    const int wall_clock_rate_khz = wall_rate.value();

    chain::Workspace feed;
    if (!feed.allocate()) {
        std::printf("FAIL: workspace alloc\n");
        return 1;
    }
    feed.fill_inputs();

    rt::Program program;
    w4::build_w4a8_program(program);
    check(program.dispatches.size() == 3u, "program has three dispatches");
    check(program.workspace.ranges.size() == 1u, "program has one workspace range");

    void* ws_host = nullptr;
    void* ws_mcu = nullptr;
    void* out_host = nullptr;
    void* out_mcu = nullptr;
    if (hipMalloc(&ws_host, w4::kWorkspaceBytes) != hipSuccess ||
        hipMalloc(&ws_mcu, w4::kWorkspaceBytes) != hipSuccess ||
        hipMalloc(&out_host, chain::Workspace::kOutBytes) != hipSuccess ||
        hipMalloc(&out_mcu, chain::Workspace::kOutBytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    (void)hipMemset(ws_host, 0xAB, w4::kWorkspaceBytes);
    (void)hipMemset(ws_mcu, 0xAB, w4::kWorkspaceBytes);
    (void)hipMemset(out_host, 0xAB, chain::Workspace::kOutBytes);
    (void)hipMemset(out_mcu, 0xAB, chain::Workspace::kOutBytes);

    const auto act_layout =
        ::ps::runtime::Int8ActivationWorkspaceLayout::make(chain::kFeatures, 1u);

    rt::StaticParameterSlot params[1];
    params[0].device_ptr = feed.rms_weight;
    params[0].elements = chain::kFeatures;
    params[0].dtype = rt::ValueDType::BF16;

    rt::WeightSlot weights[1];
    std::memset(weights, 0, sizeof(weights));
    weights[0].rows = chain::kN;
    weights[0].cols = chain::kFeatures;
    weights[0].k_padded = chain::kKp;
    weights[0].encoding = 3u;
    weights[0].compute_spec = 3u;
    weights[0].codes = feed.w_codes;
    weights[0].scales = feed.w_scales;
    weights[0].weight_scale_group = 32u;
    weights[0].storage_scale_stride_bytes = chain::kWeightScaleStride;
    weights[0].preshuffled = true;

    q35::HostExecutionContext ctx_host{};
    q35::HostExecutionContext ctx_mcu{};
    const void* ext_inputs[1] = {feed.input};
    void* ext_out_host[1] = {out_host};
    void* ext_out_mcu[1] = {out_mcu};
    w4::bind_w4a8_context(ctx_host, ws_host, ext_inputs, ext_out_host, params, weights);
    w4::bind_w4a8_context(ctx_mcu, ws_mcu, ext_inputs, ext_out_mcu, params, weights);

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    const auto host_t0 = Clock::now();
    const auto r1 = q35::try_launch_rmsnorm(program, program.dispatches[0], ctx_host, stream);
    check(r1.ok() && r1.value().kind == q35::OptimizedLaunchKind::Launched,
          "host rmsnorm launched");
    const auto r2 = q35::try_launch_activation_quantize_a8(program, program.dispatches[1], ctx_host, stream);
    check(r2.ok() && r2.value().kind == q35::OptimizedLaunchKind::Launched,
          "host activation quantize launched");
    const auto r3 = q35::try_launch_linear_psq4(program, program.dispatches[2], ctx_host, stream);
    check(r3.ok() && r3.value().kind == q35::OptimizedLaunchKind::Launched,
          "host psq4 launched");
    check(hipStreamSynchronize(stream) == hipSuccess, "host chain sync");
    const int64_t host_cold_us =
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                              host_t0)
            .count();

    for (int i = 0; i < 10; ++i) {
        (void)q35::try_launch_rmsnorm(program, program.dispatches[0], ctx_host, stream);
        (void)q35::try_launch_activation_quantize_a8(program, program.dispatches[1], ctx_host, stream);
        (void)q35::try_launch_linear_psq4(program, program.dispatches[2], ctx_host, stream);
        (void)hipStreamSynchronize(stream);
    }
    std::vector<int64_t> host_chain_us;
    host_chain_us.reserve(50);
    for (int i = 0; i < 50; ++i) {
        const auto t = Clock::now();
        (void)q35::try_launch_rmsnorm(program, program.dispatches[0], ctx_host, stream);
        (void)q35::try_launch_activation_quantize_a8(program, program.dispatches[1], ctx_host, stream);
        (void)q35::try_launch_linear_psq4(program, program.dispatches[2], ctx_host, stream);
        (void)hipStreamSynchronize(stream);
        host_chain_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t)
                .count());
    }
    std::sort(host_chain_us.begin(), host_chain_us.end());
    const int64_t host_chain_p50_us = host_chain_us[host_chain_us.size() / 2];

    q35::PhysicalRmsNormLaunch lr{};
    q35::PhysicalActivationQuantizeLaunch lq{};
    q35::PhysicalPsq4Launch lp{};
    const auto s1 = q35::resolve_rmsnorm_physical(program, program.dispatches[0], ctx_mcu, lr);
    check(s1.ok() && s1.value() == q35::PhysicalResolveStatus::Applicable,
          "rmsnorm resolved for the mcu path");
    check(lr.variant == q35::PhysicalRmsNormVariant::Bf16PfOnePlus,
          "rmsnorm resolved to the bf16/per-feature/one-plus variant");
    const auto s2 = q35::resolve_activation_quantize_physical(
        program, program.dispatches[1], ctx_mcu, lq);
    check(s2.ok() && s2.value() == q35::PhysicalResolveStatus::Applicable,
          "activation quantize resolved for the mcu path");
    check(lq.kind == q35::PhysicalActivationQuantizeKind::E4M3,
          "activation quantize resolved to the production e4m3 path");
    const auto s3 = q35::resolve_psq4_physical(program, program.dispatches[2], ctx_mcu, lp);
    check(s3.ok() && s3.value() == q35::PhysicalResolveStatus::Applicable,
          "psq4 resolved for the mcu path");
    check(lp.variant == q35::PhysicalPsq4Variant::Decode1Bf16,
          "psq4 resolved to the decode1 bf16 variant");

    std::printf(
        "  resolved rms(grid=%u block=%u) quant(grid=%u block=%u) "
        "psq4(grid=%u block=%u unroll=%u)\n",
        lr.grid, lr.workgroup, lq.grid, lq.workgroup, lp.grid, lp.workgroup,
        lp.unroll);

    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("test_gpu_mcu_production_w4a8_chain: setup failed\n");
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

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u, mcu::kMcuSystemAqlPolicy),
        mcu::make_kernel_variant(1u, rms_meta,
                                 mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus,
                                 lr.workgroup, lr.grid, mcu::kMcuAgentAqlPolicy),
        mcu::make_kernel_variant(
            2u, quant_meta, mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120,
            lq.workgroup, lq.grid, mcu::kMcuAgentAqlPolicy,
            mcu::AqlHiddenArgsPolicy::None),
        mcu::make_kernel_variant(3u, psq4_meta,
                                 mcu::kMcuKernargRecipePsq4Decode1Bf16U16,
                                 lp.workgroup, lp.grid, mcu::kMcuAgentAqlPolicy,
                                 mcu::AqlHiddenArgsPolicy::None)};

    const std::vector<mcu::McuPlanNode> nodes =
        w4::make_w4a8_handwritten_nodes();
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuRmsNormInvocation rms_inv{};
    rms_inv.input = lr.input;
    rms_inv.weight = lr.weight;
    rms_inv.output = lr.output;
    rms_inv.rows = lr.rows;
    rms_inv.features = lr.features;
    rms_inv.group_size = lr.group_size;
    rms_inv.input_row_stride = lr.input_row_stride;
    rms_inv.output_row_stride = lr.output_row_stride;
    rms_inv.eps = lr.eps;
    if (!rig.set_invocations(&rms_inv, 1u)) return 1;

    mcu::McuActivationQuantizeE4m3Invocation quant_inv{};
    quant_inv.input = lq.input;
    quant_inv.codes = lq.codes;
    quant_inv.scales = lq.scales;
    quant_inv.input_row_stride = lq.input_row_stride;
    quant_inv.code_row_stride = lq.code_row_stride;
    quant_inv.scale_row_stride = lq.scale_row_stride;
    if (!rig.set_e4m3_invocations(&quant_inv, 1u)) return 1;

    mcu::McuPsq4Decode1Invocation psq4_inv{};
    psq4_inv.weight_codes = lp.weight_codes;
    psq4_inv.weight_scales = lp.weight_scales;
    psq4_inv.activation_codes = lp.activation_codes;
    psq4_inv.activation_scales = lp.activation_scales;
    psq4_inv.output = lp.output;
    psq4_inv.k_padded = lp.k_padded;
    psq4_inv.weight_scale_stride = lp.weight_scale_stride;
    if (!rig.set_psq4_invocations(&psq4_inv, 1u)) return 1;

    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x57344138u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 8u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_production_w4a8_chain: start failed\n");
        return 1;
    }

    const auto mcu_t0 = Clock::now();
    check(rig.fsm.request_run().ok(), "chain plan requested");
    const bool mcu_ok = rig.fsm.wait_plans(1u, 20000).ok();
    check(mcu_ok, "chain plan completed");
    const int64_t mcu_host_us =
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                              mcu_t0)
            .count();
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == 4u, "four dispatches (3 real + marker)");
    const uint64_t region_start = rig.fsm.region_start_ts();
    const uint64_t region_end = rig.fsm.region_end_ts();
    const double mcu_gpu_us =
        region_end > region_start
            ? mcu::gpu_wall_clock_ticks_to_us(region_end - region_start,
                                              wall_clock_rate_khz)
            : 0.0;
    check(region_start != 0u && region_end > region_start,
          "gpu region timestamps recorded by the mcu");
    std::printf(
        "  [timing] host_cold_us=%lld host_chain_p50_us=%lld "
        "mcu_host_observed_us=%lld mcu_gpu_region_us=%.3f\n",
        static_cast<long long>(host_cold_us),
        static_cast<long long>(host_chain_p50_us),
        static_cast<long long>(mcu_host_us), mcu_gpu_us);

    const size_t rms_bytes = w4::kRmsOutBytes;
    const size_t codes_bytes =
        static_cast<size_t>(act_layout.code_row_stride_bytes) * 16u;
    const size_t scale_bytes = act_layout.scale_row_stride_bytes;
    const size_t out_bytes = chain::Workspace::kOutBytes;
    std::vector<uint8_t> host_rms(rms_bytes);
    std::vector<uint8_t> mcu_rms(rms_bytes);
    std::vector<uint8_t> host_codes(codes_bytes);
    std::vector<uint8_t> mcu_codes(codes_bytes);
    std::vector<uint8_t> host_scales(scale_bytes);
    std::vector<uint8_t> mcu_scales(scale_bytes);
    std::vector<uint8_t> host_out(out_bytes);
    std::vector<uint8_t> mcu_out(out_bytes);

    rig.device_to_host(host_rms.data(),
                       static_cast<uint8_t*>(ws_host) + w4::kRmsOutOffset, rms_bytes);
    rig.device_to_host(mcu_rms.data(),
                       static_cast<uint8_t*>(ws_mcu) + w4::kRmsOutOffset, rms_bytes);
    rig.device_to_host(host_codes.data(),
                       static_cast<uint8_t*>(ws_host) + w4::kActOffset +
                           act_layout.codes_offset,
                       codes_bytes);
    rig.device_to_host(mcu_codes.data(),
                       static_cast<uint8_t*>(ws_mcu) + w4::kActOffset +
                           act_layout.codes_offset,
                       codes_bytes);
    rig.device_to_host(host_scales.data(),
                       static_cast<uint8_t*>(ws_host) + w4::kActOffset +
                           act_layout.scales_offset,
                       scale_bytes);
    rig.device_to_host(mcu_scales.data(),
                       static_cast<uint8_t*>(ws_mcu) + w4::kActOffset +
                           act_layout.scales_offset,
                       scale_bytes);
    rig.device_to_host(host_out.data(), out_host, out_bytes);
    rig.device_to_host(mcu_out.data(), out_mcu, out_bytes);

    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(host_rms.data(), mcu_rms.data(), rms_bytes, &first,
                             &diffs),
          "rms output matches the production host path");
    if (diffs) std::printf("  rms diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_codes.data(), mcu_codes.data(), codes_bytes,
                             &first, &diffs),
          "activation codes match the production host path");
    if (diffs) std::printf("  codes diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_scales.data(), mcu_scales.data(), scale_bytes,
                             &first, &diffs),
          "activation scales match the production host path");
    if (diffs) std::printf("  scales diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_out.data(), mcu_out.data(), out_bytes, &first,
                             &diffs),
          "psq4 output matches the production host path");
    if (diffs) {
        std::printf("  out diff count=%zu first=%zu\n", diffs, first);
    }

    bool nontrivial = false;
    for (size_t i = 0; i < out_bytes; ++i) {
        if (mcu_out[i] != 0u && mcu_out[i] != 0xABu) {
            nontrivial = true;
            break;
        }
    }
    check(nontrivial, "mcu chain produced non-trivial output");

    std::printf(
        "  production chain rms(%ux%u) quant(K=%u) psq4(%ux%u u=%u) "
        "dispatches=%llu\n",
        lr.features, lr.group_size, lq.k, lp.out_features, lp.k_padded,
        lp.unroll,
        static_cast<unsigned long long>(rig.fsm.dispatches()));

    check(rig.close(), "rig teardown");

    double marker_only_us = 0.0;
    {
        Rig marker_rig;
        if (!marker_rig.open(0, 4u, 1u, 8u, 0u, false,
                             mcu::kAqlDefaultQueueSize, 0u)) {
            check(false, "marker rig open");
        } else {
            std::vector<mcu::McuKernelVariantDesc> marker_variants{
                mcu::make_probe_variant(0u, marker_rig.meta, 64u, 1u,
                                        mcu::kMcuSystemAqlPolicy)};
            std::vector<mcu::McuPlanNode> marker_nodes(3);
            marker_nodes[0].variant_id = 0;
            marker_nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
            marker_nodes[0].completion_slot = 0;
            marker_nodes[0].input_slot = mcu::kMcuNoInput;
            marker_nodes[0].flags = mcu::kMcuNodeDispatch;
            marker_nodes[0].next = 1;
            marker_nodes[1].completion_slot = 0;
            marker_nodes[1].flags = mcu::kMcuNodeWait;
            marker_nodes[1].next = 2;
            marker_nodes[2].flags = mcu::kMcuNodeEnd;
            if (marker_rig.upload(marker_nodes, marker_variants)) {
                mcu::GpuMcuFsmConfig mc{};
                mc.plan_id = 0x4d41524bu;
                mc.template_source = 0u;
                mc.barrier = 1u;
                mc.timing_count = 0u;
                mc.kernarg_slot_count = 4u;
                mc.doorbell_mode =
                    static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
                if (marker_rig.start(mc)) {
                    (void)marker_rig.fsm.request_run();
                    const bool marker_ok =
                        marker_rig.fsm.wait_plans(1u, 20000).ok();
                    check(marker_ok, "marker-only plan completed");
                    (void)marker_rig.drain_queue(30000);
                    check(marker_rig.fsm.fault_code() ==
                              mcu::McuFaultCode::none,
                          "marker-only plan no fault");
                    const uint64_t ms = marker_rig.fsm.region_start_ts();
                    const uint64_t me = marker_rig.fsm.region_end_ts();
                    if (me > ms)
                        marker_only_us = mcu::gpu_wall_clock_ticks_to_us(
                            me - ms, wall_clock_rate_khz);
                    check(marker_rig.fsm.dispatches() == 1u,
                          "marker-only plan dispatch count");
                } else {
                    check(false, "marker-only rig start");
                }
            } else {
                check(false, "marker-only upload");
            }
            check(marker_rig.close(), "marker rig teardown");
        }
    }
    std::printf("  [timing] marker_only_us=%.3f\n", marker_only_us);
    (void)hipStreamDestroy(stream);
    (void)hipFree(ws_host);
    (void)hipFree(ws_mcu);
    (void)hipFree(out_host);
    (void)hipFree(out_mcu);
    feed.free_all();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_production_w4a8_chain: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_production_w4a8_chain: PASS\n");
    return 0;
}
