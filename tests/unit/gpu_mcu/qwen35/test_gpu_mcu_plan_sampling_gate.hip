#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/program.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

struct SamplingProgram {
    rt::Program program;
    float logits[8] = {};
    int32_t sampled[4] = {};
    rt::DeviceSamplingParams sampling_params[4] = {};
    uint32_t output_rows[4] = {0u, 1u, 2u, 3u};
    void* external_outputs[2] = {};
};

void build_sampling_program(SamplingProgram& p) {
    p.program = rt::Program{};
    p.program.row_bucket = rt::RowBucket::R16;
    p.program.execution_class = rt::ExecutionClass::DECODE;

    rt::ValueBinding logits{};
    logits.logical_value = rt::ValueId{0u};
    logits.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    logits.dtype = rt::ValueDType::F32;
    logits.row_domain = rt::ValueRowDomain::OUTPUT_ROWS;
    logits.slot = 0u;
    logits.feature_count = 8u;
    logits.row_stride = 8u;
    p.program.values.push_back(logits);

    rt::ValueBinding sampled{};
    sampled.logical_value = rt::ValueId{1u};
    sampled.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    sampled.dtype = rt::ValueDType::I32;
    sampled.row_domain = rt::ValueRowDomain::OUTPUT_ROWS;
    sampled.slot = 1u;
    sampled.feature_count = 1u;
    sampled.row_stride = 1u;
    p.program.values.push_back(sampled);

    rt::DispatchBinding sampling{};
    sampling.kernel_id = rt::KernelId::SAMPLING;
    sampling.input_count = 1u;
    sampling.input_slots[0] = 0u;
    sampling.output_count = 1u;
    sampling.output_slots[0] = 1u;
    p.program.dispatches.push_back(sampling);
    p.program.dispatch_source_node.push_back(0u);
    p.program.external_input_count = 0u;
    p.program.external_output_count = 2u;

    p.external_outputs[0] = p.logits;
    p.external_outputs[1] = p.sampled;
}

ps::Status compile_once(SamplingProgram& p, uint32_t stochastic_outputs,
                        q35::McuCompiledPlan& out) {
    rt::DeviceBatchContext bc{};
    bc.actual_rows = 1u;
    bc.num_requests = 1u;
    bc.num_outputs = 1u;
    bc.output_rows = p.output_rows;
    bc.output_sampling_params = p.sampling_params;

    q35::HostExecutionContext ctx{};
    ctx.batch_context = &bc;
    ctx.external_outputs = p.external_outputs;
    ctx.external_output_count = 2u;
    ctx.actual_rows = 1u;
    ctx.actual_outputs = 1u;
    ctx.actual_sampled_outputs = 1u;
    ctx.actual_stochastic_outputs = stochastic_outputs;

    q35::McuPlanCompileOptions options{};
    options.dispatch_begin = 0u;
    options.dispatch_end = 1u;
    options.marker_grid = 1u;
    options.marker_workgroup = 64u;
    return q35::compile_mcu_plan(p.program, ctx, options, out);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    SamplingProgram p;
    build_sampling_program(p);

    {
        q35::McuCompiledPlan plan;
        const ps::Status status = compile_once(p, 0u, plan);
        if (!status.ok()) {
            std::printf("FAIL: greedy sampling compile: %s\n",
                        status.message().c_str());
            ++g_failed;
        }
        check(plan.dispatch_count == 1u, "greedy sampling compiles one dispatch");
        check(plan.nodes.size() == 4u, "greedy sampling plan node count");
        check(plan.variants.size() == 2u, "greedy sampling plan variant count");
        check(plan.argmax_f32.size() == 1u, "greedy sampling argmax invocation");
        check(plan.marker_count == 1u, "greedy sampling marker count");
    }

    {
        q35::McuCompiledPlan plan;
        const ps::Status status = compile_once(p, 1u, plan);
        check(!status.ok(), "stochastic sampling is rejected at plan compile");
        check(status.code() == ps::Status::Code::unsupported,
              "stochastic sampling reports unsupported");
        check(status.message().find("stochastic sampling has no mcu entrypoint") !=
                  std::string::npos,
              "stochastic sampling message names the missing capability");
        check(plan.nodes.empty(), "rejected plan is cleared");
        check(plan.dispatch_count == 0u, "rejected plan has no dispatch");
        check(plan.variants.empty(), "rejected plan has no variant");
    }

    {
        q35::McuCompiledPlan plan;
        const ps::Status status = compile_once(p, 2u, plan);
        check(!status.ok() && status.code() == ps::Status::Code::unsupported,
              "a batch with two stochastic rows is rejected too");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_plan_sampling_gate: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_plan_sampling_gate: PASS\n");
    return 0;
}
