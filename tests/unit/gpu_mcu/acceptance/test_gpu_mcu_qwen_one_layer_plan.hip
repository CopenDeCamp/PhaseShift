#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/models/qwen35/kernels/optimized/l2_normalize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/mcu_layer_range.h>
#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

bool directory_exists(const char* path) {
    struct stat info;
    return path != nullptr && stat(path, &info) == 0 && S_ISDIR(info.st_mode);
}

namespace mcu = ::ps::runtime::gpu_mcu;
namespace q35 = ::ps::qwen35::runtime;
namespace kernel = ::ps::kernel;
using gpu_mcu_test::Rig;

struct SymbolSpec {
    const void* hsaco = nullptr;
    std::size_t hsaco_size = 0;
    const char* symbol = nullptr;
};

bool symbol_for_kind(q35::McuCompiledVariantKind kind, SymbolSpec* out) {
    const auto entry = q35::mcu_kernel_code_object(kind);
    if (!entry.ok() || entry.value().blob.blob == nullptr ||
        entry.value().blob.size == 0u) {
        return false;
    }
    out->hsaco = entry.value().blob.blob;
    out->hsaco_size = entry.value().blob.size;
    out->symbol = entry.value().symbol;
    return true;
}

uint16_t recipe_for(q35::McuCompiledVariantKind kind) {
    const auto entry = q35::mcu_kernel_code_object(kind);
    if (!entry.ok()) return mcu::kMcuKernargRecipeNone;
    return entry.value().kernarg_recipe;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* from_env = std::getenv("PHASESHIFT_MODEL_DIR");
    if (from_env == nullptr || from_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR to a Qwen3.5 model dir\n");
        return 77;
    }
    const std::string model_dir(from_env);
    if (!directory_exists(model_dir.c_str())) {
        std::printf("SKIP: model dir not found: %s\n", model_dir.c_str());
        return 77;
    }
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }
    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 30ull * 1024ull * 1024ull * 1024ull;
    auto resident_result = ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    resident_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture resident = resident_result.release();

    auto arena_result = ps::gpu::GpuArena::create(0, resident.arena_bytes());
    if (!arena_result.ok()) {
        std::printf("FAIL: arena: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    auto model_result = resident.enabled()
        ? resident.take_qwen35()
        : ps::qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL: model: %s\n", model_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();

    auto lower_result = ps::qwen35::lower_qwen35_to_primitives(
        model.text_config(), model.weights());
    if (!lower_result.ok()) {
        std::printf("FAIL: lower: %s\n", lower_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Qwen35LoweredPrimitives lowered = lower_result.release();

    ::ps::runtime::WeightTableView weight_table;
    weight_table.slots = lowered.weights.data();
    weight_table.count = lowered.weights.size();
    ::ps::runtime::ProgramBuildOptions build_options;
    auto prog_result = ps::runtime::build_program(
        lowered.graph, weight_table, ::ps::runtime::RowBucket::R16,
        ::ps::runtime::ExecutionClass::DECODE, build_options);
    if (!prog_result.ok()) {
        std::printf("FAIL: build_program: %s\n",
                    prog_result.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    ::ps::runtime::Program program = prog_result.release();

    void* ext_in_buf = nullptr;
    void* ext_out_buf = nullptr;
    if (hipMalloc(&ext_in_buf, 16u * 1024u * 1024u) != hipSuccess ||
        hipMalloc(&ext_out_buf, 16u * 1024u * 1024u) != hipSuccess) {
        std::printf("FAIL: external buffer alloc\n");
        return 1;
    }
    (void)hipMemset(ext_in_buf, 0, 16u * 1024u * 1024u);
    (void)hipMemset(ext_out_buf, 0, 16u * 1024u * 1024u);
    std::vector<const void*> external_inputs(program.external_input_count,
                                             ext_in_buf);
    std::vector<void*> external_outputs(program.external_output_count, ext_out_buf);

    void* workspace = nullptr;
    if (program.workspace.total_bytes != 0u) {
        if (hipMalloc(&workspace, program.workspace.total_bytes) != hipSuccess) {
            std::printf("FAIL: workspace alloc\n");
            return 1;
        }
        (void)hipMemset(workspace, 0, program.workspace.total_bytes);
    }
    uint32_t* error_word = nullptr;
    if (hipMalloc(&error_word, sizeof(uint32_t)) != hipSuccess) {
        std::printf("FAIL: error word alloc\n");
        return 1;
    }
    (void)hipMemset(error_word, 0, sizeof(uint32_t));

    q35::HostExecutionContext ctx;
    ctx.workspace = static_cast<uint8_t*>(workspace);
    ctx.workspace_bytes = program.workspace.total_bytes;
    ctx.external_inputs = external_inputs.data();
    ctx.external_input_count = program.external_input_count;
    ctx.external_outputs = external_outputs.data();
    ctx.external_output_count = program.external_output_count;
    ctx.host_parameters =
        lowered.parameters.empty() ? nullptr : lowered.parameters.data();
    ctx.host_parameter_count = static_cast<uint32_t>(lowered.parameters.size());
    ctx.weights = lowered.weights.empty() ? nullptr : lowered.weights.data();
    ctx.weight_count = static_cast<uint32_t>(lowered.weights.size());
    ctx.actual_rows = 1u;
    ctx.actual_outputs = 1u;
    ctx.error_word = error_word;

    const auto gdn_layout =
        ps::qwen35::GdnStatePoolLayout::from_text_config(model.text_config());
    auto gdn_pool_res = ps::qwen35::GdnStatePool::create(arena, 1u, gdn_layout);
    if (!gdn_pool_res.ok()) {
        std::printf("FAIL: gdn state pool: %s\n",
                    gdn_pool_res.status().message().c_str());
        return 1;
    }
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_res.release();
    ctx.model_state.gdn = gdn_pool.device_view();

    auto bcs_res = ps::runtime::create_device_batch_context(arena, 1u, 1024u);
    if (!bcs_res.ok()) {
        std::printf("FAIL: batch context: %s\n", bcs_res.status().message().c_str());
        return 1;
    }
    ps::runtime::DeviceBatchContextStorage batch_storage = bcs_res.release();
    {
        ::ps::runtime::DeviceRequestDescriptor req;
        std::memset(&req, 0, sizeof(req));
        req.request_handle.slot = 0u;
        req.row_begin = 0u;
        req.row_count = 1u;
        if (hipMemcpy(batch_storage.requests
                          .data<::ps::runtime::DeviceRequestDescriptor>(),
                      &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: batch request upload\n");
            return 1;
        }
        ::ps::runtime::DeviceBatchContext bc;
        std::memset(&bc, 0, sizeof(bc));
        bc.actual_rows = 1u;
        bc.num_requests = 1u;
        bc.requests = batch_storage.requests
                          .data<::ps::runtime::DeviceRequestDescriptor>();
        if (hipMemcpy(batch_storage.device_struct
                          .data<::ps::runtime::DeviceBatchContext>(),
                      &bc, sizeof(bc), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: batch context upload\n");
            return 1;
        }
    }
    ctx.batch_context = batch_storage.device_struct
                            .data<::ps::runtime::DeviceBatchContext>();
    ctx.device_requests =
        batch_storage.requests.data<::ps::runtime::DeviceRequestDescriptor>();

    uint32_t begin = 0u;
    uint32_t end = 0u;
    const ps::Status range_status =
        ps::qwen35::runtime::find_dispatch_range_for_graph_prefix(
            lowered.graph, program, "L0", &begin, &end);
    if (!range_status.ok()) {
        std::printf("FAIL: L0 range: %s\n", range_status.message().c_str());
        return 1;
    }
    const uint32_t dispatch_count = end - begin;
    std::printf("L0 range=[%u,%u) dispatches=%u\n", begin, end, dispatch_count);
    check(dispatch_count > 0u, "L0 range is non-empty");

    q35::McuPlanCompileOptions opts;
    opts.dispatch_begin = begin;
    opts.dispatch_end = end;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const ps::Status plan_status = q35::compile_mcu_plan(program, ctx, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: L0 plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    check(plan.variants.size() <= mcu::kMcuMaxVariants,
          "one layer variants within kMcuMaxVariants");
    check(plan.nodes.size() <= mcu::kMcuMaxNodes,
          "one layer nodes within kMcuMaxNodes");
    const size_t inv_total =
        plan.rms.size() + plan.quant_e4m3.size() + plan.elementwise.size() +
        plan.rope.size() + plan.kv_append.size() + plan.attention_paged.size() +
        plan.bf16.size() + plan.bf16_wmma.size() + plan.l2.size() +
        plan.gdn_conv1d.size() + plan.gdn_recurrence.size() + plan.psq4.size() +
        plan.psq4_multi.size() + plan.attention_paged_split.size() +
        plan.attention_paged_reduce.size() + plan.embedding.size() +
        plan.embedding_psq8.size() + plan.output_gather.size() +
        plan.argmax_f32.size() + plan.verify_accept.size() +
        plan.verify_accept_batch.size() + plan.gdn_spec_restore.size() +
        plan.gdn_spec_restore_from_counts.size();
    std::printf(
        "plan: nodes=%zu variants=%zu logical=%u physical=%u "
        "kernarg_slots=%u invocations=%zu\n",
        plan.nodes.size(), plan.variants.size(), plan.logical_dispatch_count,
        plan.physical_dispatch_count, plan.kernarg_slots_required, inv_total);
    check(inv_total + plan.gdn_reset.size() ==
              static_cast<size_t>(plan.physical_dispatch_count),
          "one layer invocation per dispatch plus the gdn reset node");
    std::printf("plan: nodes=%zu variants=%zu kernarg_slots=%u invocations=%zu\n",
                plan.nodes.size(), plan.variants.size(),
                plan.kernarg_slots_required, inv_total);

    const auto layer_input = q35::resolve_host_value(
        program, ::ps::runtime::ValueId{program.dispatches[begin].input_slots[0]},
        ctx);
    if (!layer_input.ok()) {
        std::printf("FAIL: layer input resolve: %s\n",
                    layer_input.status().message().c_str());
        return 1;
    }
    const auto last_out = q35::resolve_host_value(
        program, ::ps::runtime::ValueId{program.dispatches[end - 1u].output_slots[0]},
        ctx);
    if (!last_out.ok()) {
        std::printf("FAIL: layer output resolve: %s\n",
                    last_out.status().message().c_str());
        return 1;
    }
    const size_t out_bytes =
        static_cast<size_t>(last_out.value().feature_count) *
        ((last_out.value().dtype == ::ps::runtime::ValueDType::F32) ? 4u : 2u);
    std::printf("layer input: F=%u dtype=%d  layer output: F=%u dtype=%d bytes=%zu\n",
                layer_input.value().feature_count,
                static_cast<int>(layer_input.value().dtype),
                last_out.value().feature_count,
                static_cast<int>(last_out.value().dtype), out_bytes);
    check(out_bytes > 0u, "layer output is non-empty");

    const bool input_is_f32 =
        layer_input.value().dtype == ::ps::runtime::ValueDType::F32;
    const size_t input_elems = static_cast<size_t>(layer_input.value().feature_count) *
                               ctx.actual_rows;
    std::vector<uint16_t> h_input_bf16(input_elems);
    std::vector<float> h_input_f32(input_elems);
    for (size_t i = 0; i < input_elems; ++i) {
        h_input_bf16[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1400u));
        h_input_f32[i] = static_cast<float>(static_cast<int>((i * 37u) % 201u) - 100) /
                         64.0f;
    }
    const size_t input_bytes = input_elems * (input_is_f32 ? 4u : 2u);
    auto fill_input = [&]() -> bool {
        if (layer_input.value().ptr == nullptr) return false;
        if (hipMemcpyAsync(layer_input.value().ptr,
                           input_is_f32
                               ? static_cast<const void*>(h_input_f32.data())
                               : static_cast<const void*>(h_input_bf16.data()),
                           input_bytes, hipMemcpyHostToDevice,
                           stream) != hipSuccess) {
            return false;
        }
        return hipStreamSynchronize(stream) == hipSuccess;
    };

    if (!gdn_pool.initialize(0u, stream).ok()) {
        std::printf("FAIL: gdn state initialize\n");
        return 1;
    }

    std::vector<uint8_t> host_out(out_bytes, 0u);
    auto run_host = [&]() -> bool {
        if (!gdn_pool.reset(0u, stream).ok()) {
            std::printf("FAIL: gdn state reset\n");
            return false;
        }
        if (!fill_input()) {
            std::printf("FAIL: fill layer input\n");
            return false;
        }
        (void)hipMemsetAsync(ctx.error_word, 0, sizeof(uint32_t), stream);
        for (uint32_t i = begin; i < end; ++i) {
            auto r = q35::try_launch_optimized(program, i, ctx, stream);
            if (!r.ok()) {
                std::printf("FAIL: host dispatch %u: %s\n", i,
                            r.status().message().c_str());
                return false;
            }
            if (r.value().kind != ::ps::qwen35::runtime::OptimizedLaunchKind::Launched) {
                std::printf("FAIL: host dispatch %u not launched (kind=%d)\n", i,
                            static_cast<int>(r.value().kind));
                return false;
            }
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: host sync\n");
            return false;
        }
        (void)hipMemcpy(host_out.data(), last_out.value().ptr, out_bytes,
                        hipMemcpyDeviceToHost);
        return true;
    };

    if (!run_host()) return 1;
    std::printf("host L0: launched %u dispatches\n", dispatch_count);

    Rig rig;
    std::vector<mcu::McuKernelVariantDesc> variants(plan.variants.size());
    std::vector<mcu::GpuAqlKernelMetadata> metas(plan.variants.size());

    uint32_t needed_segment = 0u;
    {
        auto hq_res = mcu::GpuMcuAqlQueue::create(
            0, mcu::AqlQueueOwner::Host, mcu::kAqlDefaultQueueSize);
        if (!hq_res.ok()) {
            std::printf("FAIL: host queue: %s\n",
                        hq_res.status().message().c_str());
            return 1;
        }
        auto hqueue = hq_res.release();
        for (size_t i = 0; i < plan.variants.size(); ++i) {
            if (plan.variants[i].kind == q35::McuCompiledVariantKind::CompletionMarker)
                continue;
            SymbolSpec spec{};
            if (!symbol_for_kind(plan.variants[i].kind, &spec)) {
                std::printf("FAIL: no symbol for kind=%d\n",
                            static_cast<int>(plan.variants[i].kind));
                return 1;
            }
            auto co_res = mcu::GpuMcuAqlCodeObject::load_memory(hqueue, spec.hsaco,
                                              spec.hsaco_size, spec.symbol);
            if (!co_res.ok()) {
                std::printf("FAIL: load %s: %s\n", spec.symbol,
                            co_res.status().message().c_str());
                return 1;
            }
            auto co = co_res.release();
            metas[i] = co.metadata();
            if (metas[i].kernarg_segment_size > needed_segment)
                needed_segment = metas[i].kernarg_segment_size;
        }
        if (!hqueue.shutdown().ok()) return 1;
    }
    std::printf("needed_segment=%u (probe=312)\n", needed_segment);

    if (!rig.open(0, static_cast<uint32_t>(plan.nodes.size()) + 4u,
                  static_cast<uint32_t>(plan.variants.size()) + 2u, 8u, 0u,
                  false, mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  needed_segment)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }

    for (size_t i = 0; i < plan.variants.size(); ++i) {
        if (plan.variants[i].kind == q35::McuCompiledVariantKind::CompletionMarker)
            continue;
        SymbolSpec spec{};
        (void)symbol_for_kind(plan.variants[i].kind, &spec);
        if (!rig.load_extra_code_memory(spec.hsaco, spec.hsaco_size, spec.symbol,
                                  metas[i])) return 1;
    }

    for (size_t i = 0; i < plan.variants.size(); ++i) {
        const auto& v = plan.variants[i];
        if (v.kind == q35::McuCompiledVariantKind::CompletionMarker) {
            variants[i] = mcu::make_probe_variant(
                static_cast<uint16_t>(i), rig.meta, v.workgroup, v.grid,
                mcu::kMcuSystemAqlPolicy);
        } else {
            variants[i] = mcu::make_kernel_variant(
                static_cast<uint16_t>(i), metas[i], recipe_for(v.kind), v.workgroup,
                v.grid, mcu::kMcuAgentAqlPolicy,
                mcu::AqlHiddenArgsPolicy::IfFits, v.grid_y, v.grid_z);
        }
    }

    if (!rig.upload(plan.nodes, variants)) return 1;
    if (!rig.set_invocations(plan.rms.data(),
                             static_cast<uint32_t>(plan.rms.size())))
        return 1;
    if (!rig.set_e4m3_invocations(
            plan.quant_e4m3.data(),
            static_cast<uint32_t>(plan.quant_e4m3.size())))
        return 1;
    if (!rig.set_elementwise_invocations(
            plan.elementwise.data(),
            static_cast<uint32_t>(plan.elementwise.size())))
        return 1;
    if (!rig.set_bf16_invocations(plan.bf16.data(),
                                  static_cast<uint32_t>(plan.bf16.size())))
        return 1;
    if (!rig.set_l2_invocations(plan.l2.data(),
                                static_cast<uint32_t>(plan.l2.size())))
        return 1;
    if (!rig.set_gdn_conv1d_invocations(
            plan.gdn_conv1d.data(),
            static_cast<uint32_t>(plan.gdn_conv1d.size())))
        return 1;
    if (!rig.set_gdn_recurrence_invocations(
            plan.gdn_recurrence.data(),
            static_cast<uint32_t>(plan.gdn_recurrence.size())))
        return 1;
    if (!rig.set_gdn_reset_invocations(
            plan.gdn_reset.data(),
            static_cast<uint32_t>(plan.gdn_reset.size())))
        return 1;
    if (!rig.set_psq4_invocations(
            plan.psq4.data(),
            static_cast<uint32_t>(plan.psq4.size())))
        return 1;
    if (!rig.set_psq4_multi_invocations(
            plan.psq4_multi.data(),
            static_cast<uint32_t>(plan.psq4_multi.size())))
        return 1;
    if (!rig.set_rope_invocations(
            plan.rope.data(),
            static_cast<uint32_t>(plan.rope.size())))
        return 1;
    if (!rig.set_kv_append_invocations(
            plan.kv_append.data(),
            static_cast<uint32_t>(plan.kv_append.size())))
        return 1;
    if (!rig.set_attention_paged_invocations(
            plan.attention_paged.data(),
            static_cast<uint32_t>(plan.attention_paged.size())))
        return 1;
    if (!rig.set_attention_paged_split_invocations(
            plan.attention_paged_split.data(),
            static_cast<uint32_t>(plan.attention_paged_split.size())))
        return 1;
    if (!rig.set_attention_paged_reduce_invocations(
            plan.attention_paged_reduce.data(),
            static_cast<uint32_t>(plan.attention_paged_reduce.size())))
        return 1;
    if (!rig.set_bf16_wmma_invocations(
            plan.bf16_wmma.data(),
            static_cast<uint32_t>(plan.bf16_wmma.size())))
        return 1;
    if (!rig.set_embedding_invocations(
            plan.embedding.data(),
            static_cast<uint32_t>(plan.embedding.size())))
        return 1;
    if (!rig.set_embedding_psq8_invocations(
            plan.embedding_psq8.data(),
            static_cast<uint32_t>(plan.embedding_psq8.size())))
        return 1;
    if (!rig.set_verify_accept_invocations(
            plan.verify_accept.data(),
            static_cast<uint32_t>(plan.verify_accept.size())))
        return 1;
    if (!rig.set_verify_accept_batch_invocations(
            plan.verify_accept_batch.data(),
            static_cast<uint32_t>(plan.verify_accept_batch.size())))
        return 1;
    if (!rig.set_gdn_spec_restore_invocations(
            plan.gdn_spec_restore.data(),
            static_cast<uint32_t>(plan.gdn_spec_restore.size())))
        return 1;
    if (!rig.set_gdn_spec_restore_from_counts_invocations(
            plan.gdn_spec_restore_from_counts.data(),
            static_cast<uint32_t>(plan.gdn_spec_restore_from_counts.size())))
        return 1;
    if (!rig.set_argmax_f32_invocations(
            plan.argmax_f32.data(),
            static_cast<uint32_t>(plan.argmax_f32.size())))
        return 1;
    if (!rig.set_output_gather_invocations(
            plan.output_gather.data(),
            static_cast<uint32_t>(plan.output_gather.size())))
        return 1;
    rig.poison_all_kernarg();

    const uint32_t template_source =
        std::getenv("PHASESHIFT_MCU_LDS_TEMPLATES") != nullptr ? 1u : 0u;
    int iterations = 1000;
    if (const char* it_env = std::getenv("PHASESHIFT_MCU_PLAN_ITERS")) {
        const long v = std::strtol(it_env, nullptr, 10);
        if (v > 0) iterations = static_cast<int>(v);
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4c30504cu;
    config.template_source = template_source;
    config.lds_budget_bytes = template_source != 0u ? 8192u : 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = plan.kernarg_slots_required;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }

    std::vector<uint8_t> mcu_out(out_bytes, 0u);
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
        if (!gdn_pool.reset(0u, stream).ok()) {
            std::printf("FAIL: gdn state reset (mcu)\n");
            return 1;
        }
        if (!fill_input()) {
            std::printf("FAIL: fill layer input (mcu)\n");
            return 1;
        }
        if (!rig.fsm.request_run().ok()) {
            std::printf("FAIL: mcu plan requested\n");
            return 1;
        }
        if (!rig.fsm.wait_plans(static_cast<uint64_t>(it) + 1u, 60000).ok()) {
            std::printf("FAIL: mcu plan completed at iter %d\n", it);
            return 1;
        }
        if (!rig.drain_queue(60000)) {
            std::printf("FAIL: queue drained at iter %d\n", it);
            return 1;
        }
        if (it + 1 == iterations) {
            rig.device_to_host(mcu_out.data(), last_out.value().ptr, out_bytes);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double plan_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no mcu fault");
    check(rig.fsm.dispatches() ==
              static_cast<uint64_t>(plan.physical_dispatch_count + 1u) *
                  static_cast<uint64_t>(iterations),
          "mcu dispatch count = (physical + marker) * iterations");
    std::printf(
        "mcu: iters=%d template_source=%u dispatches=%llu fault=%u "
        "avg_iter=%.4f ms/iter\n",
        iterations, template_source,
        static_cast<unsigned long long>(rig.fsm.dispatches()),
        static_cast<unsigned>(rig.fsm.fault_code()), plan_ms);

    size_t first = 0;
    size_t diffs = 0;
    if (host_out.size() == mcu_out.size()) {
        for (size_t i = 0; i < host_out.size(); ++i) {
            if (host_out[i] != mcu_out[i]) {
                if (first == 0u) first = i;
                ++diffs;
            }
        }
    } else {
        diffs = 1;
    }
    if (diffs != 0u) {
        std::printf("FAIL: L0 output mismatch: bytes=%zu first=%zu count=%zu\n",
                    out_bytes, first, diffs);
        const size_t w = first / 4u;
        std::printf("  host[0..3]=%02x %02x %02x %02x  mcu[0..3]=%02x %02x %02x %02x\n",
                    host_out[w * 4], host_out[w * 4 + 1], host_out[w * 4 + 2],
                    host_out[w * 4 + 3], mcu_out[w * 4], mcu_out[w * 4 + 1],
                    mcu_out[w * 4 + 2], mcu_out[w * 4 + 3]);
        ++g_failed;
    } else {
        std::printf("L0 host vs mcu: %zu bytes byte-exact\n", out_bytes);
    }

    check(rig.close(), "rig teardown");

    const ps::Status bc_destroy =
        ps::runtime::destroy_device_batch_context(batch_storage);
    check(bc_destroy.ok(), "device batch context destroyed");
    if (workspace) (void)hipFree(workspace);
    (void)hipFree(error_word);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    arena.shutdown();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_qwen_one_layer_plan: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_qwen_one_layer_plan: PASS\n");
    return 0;
}
