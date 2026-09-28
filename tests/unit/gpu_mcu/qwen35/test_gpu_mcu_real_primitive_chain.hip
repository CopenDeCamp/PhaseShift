#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

using Clock = std::chrono::steady_clock;
Clock::time_point g_start;
Clock::time_point g_phase;

int64_t ms_since(Clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                 from)
        .count();
}

int64_t ms_total() { return ms_since(g_start); }

void mark_phase() { g_phase = Clock::now(); }

int64_t ms_phase() { return ms_since(g_phase); }

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

std::vector<mcu::McuPlanNode> make_chain_plan() {
    std::vector<mcu::McuPlanNode> nodes(6);
    nodes[0].variant_id = 1;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1;

    nodes[1].variant_id = 2;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeActivationQuantizeA8;
    nodes[1].completion_slot = 0;
    nodes[1].invocation_index = 0;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2;

    nodes[2].variant_id = 3;
    nodes[2].kernarg_recipe = mcu::kMcuKernargRecipePsq4Decode1Bf16U16;
    nodes[2].completion_slot = 0;
    nodes[2].invocation_index = 0;
    nodes[2].flags = mcu::kMcuNodeDispatch;
    nodes[2].next = 3;

    nodes[3].variant_id = 0;
    nodes[3].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[3].completion_slot = 0;
    nodes[3].input_slot = mcu::kMcuNoInput;
    nodes[3].flags = mcu::kMcuNodeDispatch;
    nodes[3].next = 4;

    nodes[4].completion_slot = 0;
    nodes[4].flags = mcu::kMcuNodeWait;
    nodes[4].next = 5;

    nodes[5].flags = mcu::kMcuNodeEnd;
    return nodes;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    g_start = Clock::now();
    g_phase = g_start;
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    chain::Workspace host_ws;
    chain::Workspace mcu_ws;
    if (!host_ws.allocate() || !mcu_ws.allocate()) {
        std::printf("FAIL: workspace alloc\n");
        return 1;
    }
    host_ws.fill_inputs();
    mcu_ws.fill_inputs();

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    const auto host_chain_t0 = Clock::now();
    const hipError_t hr = chain::run_host_chain(host_ws, stream);
    check(hr == hipSuccess, "host reference chain launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "host chain sync");
    const int64_t host_chain_cold_us =
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                              host_chain_t0)
            .count();

    constexpr int kWarmup = 10;
    constexpr int kHostIters = 50;
    for (int i = 0; i < kWarmup; ++i) {
        const hipError_t w = chain::run_host_chain(host_ws, stream);
        check(w == hipSuccess, "host chain warmup launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host chain warmup sync");
    }
    std::vector<int64_t> host_chain_us;
    host_chain_us.reserve(kHostIters);
    for (int i = 0; i < kHostIters; ++i) {
        const auto t0 = Clock::now();
        const hipError_t w = chain::run_host_chain(host_ws, stream);
        check(w == hipSuccess, "host chain measured launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host chain measured sync");
        host_chain_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0)
                .count());
    }
    std::sort(host_chain_us.begin(), host_chain_us.end());
    const int64_t host_chain_p50 = host_chain_us[kHostIters / 2];
    const int64_t host_chain_min = host_chain_us.front();

    const uint32_t rms_block =
        kernel::rmsnorm_bf16_pf_oneplus_block_size(chain::kFeatures);
    const uint32_t rms_grid =
        kernel::rmsnorm_bf16_pf_oneplus_grid(chain::kRows, chain::kFeatures,
                                             chain::kFeatures);
    const uint32_t quant_block = kernel::activation_quantize_a8_block_size();
    const uint32_t quant_grid =
        kernel::activation_quantize_a8_grid(chain::kRows, chain::kKp);
    const uint32_t psq4_block = kernel::psq4_decode1_block_size();
    const uint32_t psq4_grid = kernel::psq4_decode1_grid(chain::kN);

    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("test_gpu_mcu_real_primitive_chain: setup failed\n");
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
                             kernel::kActivationQuantizeA8Symbol, quant_meta)) {
        return 1;
    }
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_PSQ4_HSACO,
                             kernel::kPsq4Decode1Bf16U16Symbol, psq4_meta)) {
        return 1;
    }
    check(rms_meta.kernel_object != 0, "rmsnorm kernel_object resolved");
    check(quant_meta.kernel_object != 0, "quantize kernel_object resolved");
    check(psq4_meta.kernel_object != 0, "psq4 kernel_object resolved");

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u, mcu::kMcuSystemAqlPolicy),
        mcu::make_kernel_variant(1u, rms_meta,
                                 mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus, rms_block,
                                 rms_grid, mcu::kMcuAgentAqlPolicy),
        mcu::make_kernel_variant(2u, quant_meta,
                                 mcu::kMcuKernargRecipeActivationQuantizeA8, quant_block,
                                 quant_grid, mcu::kMcuAgentAqlPolicy),
        mcu::make_kernel_variant(3u, psq4_meta,
                                 mcu::kMcuKernargRecipePsq4Decode1Bf16U16, psq4_block,
                                 psq4_grid, mcu::kMcuAgentAqlPolicy,
                                 mcu::AqlHiddenArgsPolicy::None)};

    const std::vector<mcu::McuPlanNode> nodes = make_chain_plan();
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuRmsNormInvocation rms{};
    rms.input = reinterpret_cast<uint64_t>(mcu_ws.input);
    rms.weight = reinterpret_cast<uint64_t>(mcu_ws.rms_weight);
    rms.output = reinterpret_cast<uint64_t>(mcu_ws.rms_out);
    rms.rows = chain::kRows;
    rms.features = chain::kFeatures;
    rms.group_size = chain::kFeatures;
    rms.input_row_stride = chain::kFeatures;
    rms.output_row_stride = chain::kFeatures;
    rms.eps = chain::kEps;
    if (!rig.set_invocations(&rms, 1u)) return 1;

    mcu::McuActivationQuantizeInvocation quant{};
    quant.input = reinterpret_cast<uint64_t>(mcu_ws.rms_out);
    quant.codes = reinterpret_cast<uint64_t>(mcu_ws.codes);
    quant.scales = reinterpret_cast<uint64_t>(mcu_ws.scales);
    quant.rows = chain::kRows;
    quant.k = chain::kFeatures;
    quant.k_padded = chain::kKp;
    quant.input_row_stride = chain::kFeatures;
    quant.code_row_stride = chain::Workspace::code_row_stride();
    quant.scale_row_stride = chain::Workspace::scale_row_stride();
    if (!rig.set_quantize_invocations(&quant, 1u)) return 1;

    mcu::McuPsq4Decode1Invocation psq4{};
    psq4.weight_codes = reinterpret_cast<uint64_t>(mcu_ws.w_codes);
    psq4.weight_scales = reinterpret_cast<uint64_t>(mcu_ws.w_scales);
    psq4.activation_codes = reinterpret_cast<uint64_t>(mcu_ws.codes);
    psq4.activation_scales = reinterpret_cast<uint64_t>(mcu_ws.scales);
    psq4.output = reinterpret_cast<uint64_t>(mcu_ws.final_out);
    psq4.k_padded = chain::kKp;
    psq4.weight_scale_stride = chain::kWeightScaleStride;
    if (!rig.set_psq4_invocations(&psq4, 1u)) return 1;

    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4348414eu;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 8u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_real_primitive_chain: start failed\n");
        return 1;
    }
    mark_phase();
    const auto plan_t0 = Clock::now();
    check(rig.fsm.request_run().ok(), "chain plan requested");
    const bool plan_ok = rig.fsm.wait_plans(1u, 20000).ok();
    check(plan_ok, "chain plan completed");
    const int64_t plan_us =
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                              plan_t0)
            .count();
    const int64_t plan_ms = ms_phase();
    mark_phase();
    const bool drain1_ok = rig.drain_queue(30000);
    const int64_t drain1_ms = ms_phase();
    check(drain1_ok, "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == 4u, "four dispatches (3 real + marker)");
    const uint64_t ahead_first = rig.fsm.max_ahead();
    const uint64_t doorbells_first = rig.fsm.doorbell_count();
    const uint64_t refill_first = rig.fsm.refill_count();
    const uint64_t empty_first = rig.fsm.queue_empty_count();
    std::printf(
        "  [diag] first plan ahead=%llu doorbells=%llu refill=%llu empty=%llu\n",
        static_cast<unsigned long long>(ahead_first),
        static_cast<unsigned long long>(doorbells_first),
        static_cast<unsigned long long>(refill_first),
        static_cast<unsigned long long>(empty_first));
    check(ahead_first >= 2u, "later packets queued ahead while the first ran");
    check(doorbells_first == 4u, "one doorbell per packet");
    check(empty_first <= 1u, "queue stayed fed inside the plan");
    mark_phase();

    const size_t rms_bytes = chain::Workspace::kInputBytes;
    std::vector<uint16_t> host_rms(rms_bytes / 2, 0u);
    std::vector<uint16_t> mcu_rms(rms_bytes / 2, 0u);
    rig.device_to_host(host_rms.data(), host_ws.rms_out, rms_bytes);
    rig.device_to_host(mcu_rms.data(), mcu_ws.rms_out, rms_bytes);

    const size_t codes_bytes =
        static_cast<size_t>(chain::Workspace::code_row_stride()) * 16u;
    const size_t scale_bytes = chain::Workspace::scale_row_stride();
    std::vector<uint8_t> host_codes(codes_bytes);
    std::vector<uint8_t> mcu_codes(codes_bytes);
    std::vector<uint8_t> host_scales(scale_bytes);
    std::vector<uint8_t> mcu_scales(scale_bytes);
    rig.device_to_host(host_codes.data(), host_ws.codes, codes_bytes);
    rig.device_to_host(mcu_codes.data(), mcu_ws.codes, codes_bytes);
    rig.device_to_host(host_scales.data(), host_ws.scales, scale_bytes);
    rig.device_to_host(mcu_scales.data(), mcu_ws.scales, scale_bytes);

    const size_t out_bytes = chain::Workspace::kOutBytes;
    std::vector<uint16_t> host_out(out_bytes / 2, 0u);
    std::vector<uint16_t> mcu_out(out_bytes / 2, 0u);
    rig.device_to_host(host_out.data(), host_ws.final_out, out_bytes);
    rig.device_to_host(mcu_out.data(), mcu_ws.final_out, out_bytes);

    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(host_rms.data(), mcu_rms.data(), rms_bytes, &first, &diffs),
          "rms output matches host chain");
    if (diffs) {
        std::printf("  rms diff: count=%zu first=%zu host=%.9g mcu=%.9g\n", diffs,
                    first, chain::f32_from_bf16(host_rms[first]),
                    chain::f32_from_bf16(mcu_rms[first]));
    }
    check(chain::bytes_equal(host_codes.data(), mcu_codes.data(), codes_bytes,
                             &first, &diffs),
          "a8 codes match host chain");
    if (diffs) std::printf("  codes diff: count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_scales.data(), mcu_scales.data(), scale_bytes,
                             &first, &diffs),
          "a8 scales match host chain");
    if (diffs) std::printf("  scales diff: count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_out.data(), mcu_out.data(), out_bytes, &first,
                             &diffs),
          "psq4 final output matches host chain");
    if (diffs) {
        std::printf("  final diff: count=%zu first=%zu host=%.9g mcu=%.9g\n", diffs,
                    first, chain::f32_from_bf16(host_out[first]),
                    chain::f32_from_bf16(mcu_out[first]));
    }

    bool host_nonzero = false;
    for (uint16_t v : host_out) {
        if (v != 0u && v != 0xABABu) { host_nonzero = true; break; }
    }
    check(host_nonzero, "host chain produced non-trivial output");

    std::printf(
        "  chain R=%u F=%u N=%u rms=%ux%u quant=%ux%u psq4=%ux%u\n",
        chain::kRows, chain::kFeatures, chain::kN, rms_grid, rms_block, quant_grid,
        quant_block, psq4_grid, psq4_block);
    mark_phase();
    for (uint64_t i = 2; i <= 1000u; ++i) {
        if (!rig.fsm.request_run().ok()) {
            check(false, "plan requested");
            break;
        }
        auto step = rig.fsm.wait_plans(i, 5000);
        if (!step.ok()) {
            std::printf("  wait[%llu]: %s fault=%u pc=%u\n",
                        static_cast<unsigned long long>(i),
                        step.message().c_str(),
                        static_cast<unsigned>(rig.fsm.fault_code()),
                        static_cast<unsigned>(rig.fsm.fault_pc()));
            check(false, "chain plan completed");
            break;
        }
    }
    const int64_t longrun_ms = ms_phase();
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault over 1000 plans");
    check(rig.fsm.dispatches() == 1000u * 4u, "dispatch count over 1000 plans");
    check(rig.fsm.completions_observed() == 1000u, "marker completions over 1000 plans");
    check(rig.drain_queue(30000), "queue drained over long run");

    const uint64_t final_dispatches = rig.fsm.dispatches();
    const uint64_t final_completions = rig.fsm.completions_observed();
    const uint64_t final_doorbells = rig.fsm.doorbell_count();
    const uint64_t final_max_ahead = rig.fsm.max_ahead();
    const uint64_t final_refill = rig.fsm.refill_count();
    const uint64_t final_empty = rig.fsm.queue_empty_count();
    const uint64_t final_backpressure = rig.fsm.backpressure_wait_count();
    const uint64_t final_wrap = rig.fsm.wrap_count();
    const unsigned final_fault = static_cast<unsigned>(rig.fsm.fault_code());
    const unsigned final_fault_pc = static_cast<unsigned>(rig.fsm.fault_pc());

    mark_phase();
    const bool close_ok = rig.close();
    const int64_t close_ms = ms_phase();
    check(close_ok, "rig teardown");
    (void)hipStreamDestroy(stream);
    host_ws.free_all();
    mcu_ws.free_all();

    const int64_t mcu_steady_us = longrun_ms * 1000 / 999;
    std::printf(
        "  [diag] host_chain_cold_us=%lld host_chain_p50_us=%lld "
        "host_chain_min_us=%lld mcu_plan_cold_us=%lld mcu_plan_p50_us=%lld "
        "first_plan_ms=%lld drain1_ms=%lld longrun_ms=%lld plans_per_sec=%.0f "
        "dispatches=%llu completions=%llu fault=%u pc=%u "
        "doorbells=%llu max_ahead=%llu refill=%llu empty=%llu full=%llu "
        "wraps=%llu close_ms=%lld total_ms=%lld\n",
        static_cast<long long>(host_chain_cold_us),
        static_cast<long long>(host_chain_p50),
        static_cast<long long>(host_chain_min),
        static_cast<long long>(plan_us), static_cast<long long>(mcu_steady_us),
        static_cast<long long>(plan_ms), static_cast<long long>(drain1_ms),
        static_cast<long long>(longrun_ms),
        longrun_ms > 0 ? 999.0 * 1000.0 / static_cast<double>(longrun_ms) : 0.0,
        static_cast<unsigned long long>(final_dispatches),
        static_cast<unsigned long long>(final_completions), final_fault,
        final_fault_pc, static_cast<unsigned long long>(final_doorbells),
        static_cast<unsigned long long>(final_max_ahead),
        static_cast<unsigned long long>(final_refill),
        static_cast<unsigned long long>(final_empty),
        static_cast<unsigned long long>(final_backpressure),
        static_cast<unsigned long long>(final_wrap),
        static_cast<long long>(close_ms), static_cast<long long>(ms_total()));

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_primitive_chain: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_primitive_chain: PASS\n");
    return 0;
}
