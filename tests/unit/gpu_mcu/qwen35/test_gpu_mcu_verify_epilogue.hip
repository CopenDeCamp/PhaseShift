#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace rt = ::ps::runtime;
namespace q35 = ::ps::qwen35::runtime;

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

constexpr uint32_t kN = 256u;
constexpr uint32_t kK = 5120u;

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    void* weight_dev = nullptr;
    void* input_dev = nullptr;
    void* output_dev = nullptr;
    void* sampled_dev = nullptr;
    void* counts_dev = nullptr;
    if (hipMalloc(&weight_dev, static_cast<std::size_t>(kN) * kK * 2u) !=
            hipSuccess ||
        hipMalloc(&input_dev, 16u * kK * 2u) != hipSuccess ||
        hipMalloc(&output_dev, 16u * kN * 4u) != hipSuccess ||
        hipMalloc(&sampled_dev, 16u * sizeof(int32_t)) != hipSuccess ||
        hipMalloc(&counts_dev, 8u * sizeof(uint32_t)) != hipSuccess) {
        std::printf("FAIL: buffers\n");
        return 1;
    }

    rt::Program program{};
    program.row_bucket = rt::RowBucket::R16;
    program.execution_class = rt::ExecutionClass::DECODE;
    program.weight_slot_count = 1u;

    rt::ValueBinding in{};
    in.logical_value = rt::ValueId{0u};
    in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    in.dtype = rt::ValueDType::BF16;
    in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    in.slot = 0u;
    in.feature_count = kK;
    in.row_stride = kK;
    program.values.push_back(in);

    rt::ValueBinding out{};
    out.logical_value = rt::ValueId{1u};
    out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    out.dtype = rt::ValueDType::F32;
    out.row_domain = rt::ValueRowDomain::OUTPUT_ROWS;
    out.slot = 0u;
    out.feature_count = kN;
    out.row_stride = kN;
    program.values.push_back(out);

    rt::DispatchBinding lin{};
    lin.kernel_id = rt::KernelId::LINEAR_BF16;
    lin.input_count = 1u;
    lin.input_slots[0] = 0u;
    lin.output_count = 1u;
    lin.output_slots[0] = 1u;
    lin.weight_index = 0u;
    program.dispatches.push_back(lin);
    program.dispatch_source_node.push_back(0u);

    const std::size_t value_count = program.values.size();
    const std::size_t dispatch_count = program.dispatches.size();

    rt::WeightSlot weights[1]{};
    weights[0].encoding = 0u;
    weights[0].device_ptr = weight_dev;
    weights[0].rows = kN;
    weights[0].cols = kK;

    const void* ext_in[1] = {input_dev};
    void* ext_out[1] = {output_dev};
    rt::DeviceBatchContext batch_ctx{};
    batch_ctx.actual_rows = 4u;
    batch_ctx.num_requests = 1u;
    batch_ctx.num_outputs = 1u;
    batch_ctx.requests =
        reinterpret_cast<const rt::DeviceRequestDescriptor*>(weight_dev);
    batch_ctx.token_ids = static_cast<const int32_t*>(sampled_dev);
    batch_ctx.output_rows = static_cast<uint32_t*>(counts_dev);

    q35::HostExecutionContext ctx{};
    ctx.external_inputs = ext_in;
    ctx.external_input_count = 1u;
    ctx.external_outputs = ext_out;
    ctx.external_output_count = 1u;
    ctx.weights = weights;
    ctx.weight_count = 1u;
    ctx.actual_rows = 4u;
    ctx.actual_outputs = 1u;
    ctx.role = rt::ExecutionRole::Verify;
    ctx.numeric_mode = rt::VerifyNumericMode::Exact;
    ctx.batch_context = &batch_ctx;
    ctx.device_requests = batch_ctx.requests;
    ctx.device_output_rows = batch_ctx.output_rows;

    q35::McuPlanCompileOptions plain_opts{};
    plain_opts.dispatch_begin = 0u;
    plain_opts.dispatch_end = static_cast<uint32_t>(dispatch_count);

    q35::McuCompiledPlan plain{};
    const ps::Status plain_st = q35::compile_mcu_plan(program, ctx, plain_opts, plain);
    if (!plain_st.ok()) {
        std::printf("FAIL: plain compile: %s\n", plain_st.message().c_str());
        return 1;
    }

    q35::McuPlanCompileOptions verify_opts = plain_opts;
    verify_opts.verify_sampled_tokens = static_cast<const int32_t*>(sampled_dev);
    verify_opts.verify_committed_counts = static_cast<uint32_t*>(counts_dev);
    verify_opts.verify_max_candidates = 32u;

    q35::McuCompiledPlan verify{};
    const ps::Status verify_st =
        q35::compile_mcu_plan(program, ctx, verify_opts, verify);
    if (!verify_st.ok()) {
        std::printf("FAIL: verify compile: %s\n", verify_st.message().c_str());
        return 1;
    }

    check(program.values.size() == value_count,
          "program values are unchanged by the epilogue");
    check(program.dispatches.size() == dispatch_count,
          "program dispatches are unchanged by the epilogue");

    check(!plain.epilogue.verify_accept_capable,
          "no verify accept epilogue without runtime sources");
    check(!plain.epilogue.gdn_restore_capable,
          "no gdn restore epilogue without runtime sources");
    check(plain.nodes.size() > dispatch_count,
          "plain plan carries the model dispatches and a marker tail");
    check(plain.nodes.size() == verify.nodes.size() - 1u ||
              verify.nodes.size() == plain.nodes.size(),
          "the verify epilogue adds at most one node");

    check(verify.epilogue.verify_accept_capable,
          "verify accept epilogue is enabled with runtime sources");
    check(verify.epilogue.verify_accept_node >= dispatch_count,
          "verify accept node follows the model dispatches");
    check(verify.epilogue.verify_accept_node < verify.nodes.size(),
          "verify accept node index is inside the plan");
    check(verify.nodes[verify.epilogue.verify_accept_node].invocation_index ==
              verify.epilogue.verify_accept_invocation,
          "verify accept node points at its invocation");
    check(verify.bf16_variant_catalogs.size() == 1u,
          "the lm-head catalog is present in both plans");
    check(plain.bf16_variant_catalogs.size() == 1u,
          "the plain plan keeps the same lm-head catalog");

    bool model_prefix_equal = true;
    for (std::size_t i = 0; i < dispatch_count; ++i) {
        const rt::gpu_mcu::McuPlanNode& a = plain.nodes[i];
        const rt::gpu_mcu::McuPlanNode& b = verify.nodes[i];
        if (a.variant_id != b.variant_id || a.invocation_index != b.invocation_index ||
            a.kernarg_recipe != b.kernarg_recipe) {
            model_prefix_equal = false;
        }
    }
    check(model_prefix_equal, "model dispatch prefix is identical in both plans");

    (void)hipFree(weight_dev);
    (void)hipFree(input_dev);
    (void)hipFree(output_dev);
    (void)hipFree(sampled_dev);
    (void)hipFree(counts_dev);
    std::printf("test_gpu_mcu_verify_epilogue: %s (%d failed)\n",
                g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
