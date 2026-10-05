#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/mcu_layer_range.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>

#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    const unsigned long parsed = std::strtoul(v, nullptr, 10);
    return parsed > 0xFFFFFFFFul ? fallback : static_cast<uint32_t>(parsed);
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

    std::printf("=== Qwen layer coverage inventory ===\n");
    std::printf("model_dir: %s\n", model_dir.c_str());

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
        std::printf("FAIL: model load: %s\n",
                    model_result.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();

    auto lower_result = ps::qwen35::lower_qwen35_to_primitives(
        model.text_config(), model.weights());
    if (!lower_result.ok()) {
        std::printf("FAIL: lowering: %s\n",
                    lower_result.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    ps::qwen35::Qwen35LoweredPrimitives lowered = lower_result.release();

    ps::runtime::WeightTableView weight_table;
    weight_table.slots = lowered.weights.data();
    weight_table.count = lowered.weights.size();

    ps::runtime::ProgramBuildOptions options;
    auto program_result = ps::runtime::build_program(
        lowered.graph, weight_table, ps::runtime::RowBucket::R16,
        ps::runtime::ExecutionClass::DECODE, options);
    if (!program_result.ok()) {
        std::printf("FAIL: build_program: %s\n",
                    program_result.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    ps::runtime::Program program = program_result.release();
    check(program.dispatch_source_node.size() == program.dispatches.size(),
          "program carries dispatch provenance");

    const size_t layer_nodes =
        lowered.graph.nodes.empty() ? 0u : 1u;
    if (layer_nodes != 0u) {
        std::printf("first source node: %s\n",
                    lowered.graph.nodes[0].debug_name.c_str());
    }
    std::printf("graph_nodes=%zu dispatches=%zu weights=%zu params=%zu\n",
                lowered.graph.nodes.size(), program.dispatches.size(),
                lowered.weights.size(), lowered.parameters.size());

    void* workspace = nullptr;
    if (program.workspace.total_bytes != 0u &&
        hipMalloc(&workspace, program.workspace.total_bytes) != hipSuccess) {
        std::printf("FAIL: workspace alloc\n");
        arena.shutdown();
        return 1;
    }
    std::vector<const void*> external_inputs(program.external_input_count,
                                             reinterpret_cast<const void*>(0x1000));
    std::vector<void*> external_outputs(program.external_output_count,
                                        reinterpret_cast<void*>(0x2000));

    ps::qwen35::runtime::HostExecutionContext ctx;
    ctx.workspace = static_cast<uint8_t*>(workspace);
    ctx.workspace_bytes = program.workspace.total_bytes;
    ctx.external_inputs = external_inputs.data();
    ctx.external_input_count = program.external_input_count;
    ctx.external_outputs = external_outputs.data();
    ctx.external_output_count = program.external_output_count;
    ctx.host_parameters = lowered.parameters.empty() ? nullptr
                                                     : lowered.parameters.data();
    ctx.host_parameter_count = static_cast<uint32_t>(lowered.parameters.size());
    ctx.weights = lowered.weights.empty() ? nullptr : lowered.weights.data();
    ctx.weight_count = static_cast<uint32_t>(lowered.weights.size());
    ctx.actual_rows = 1u;
    ctx.actual_outputs = 1u;

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
    std::printf(
        "gdn state: num_states=%u conv_dim=%u history=%u head_k=%u head_v=%u "
        "num_v_heads=%u conv_base=%p recurrent_base=%p\n",
        gdn_pool.num_gdn_states(), gdn_pool.conv_dim(), gdn_pool.conv_history(),
        gdn_pool.head_k(), gdn_pool.head_v(), gdn_pool.num_v_heads(),
        static_cast<void*>(ctx.model_state.gdn.conv_base),
        static_cast<void*>(ctx.model_state.gdn.recurrent_base));

    const auto& cfg = model.text_config();
    const uint32_t kv_heads = static_cast<uint32_t>(cfg.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
    uint32_t attention_layers = 0u;
    for (int lt : cfg.layer_types) {
        if (lt == 1) ++attention_layers;
    }
    const uint32_t page_tokens = 16u;

    auto kv_res = ps::qwen35::PagedKVPool::create(
        arena, 512u, page_tokens, attention_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    if (!kv_res.ok()) {
        std::printf("FAIL: kv pool: %s\n", kv_res.status().message().c_str());
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    ps::qwen35::PagedKVPool kv_pool = kv_res.release();

    auto seq_res = ps::qwen35::SequenceSlotPool::create(arena, 8u, 2048u);
    if (!seq_res.ok()) {
        std::printf("FAIL: sequence slot pool: %s\n",
                    seq_res.status().message().c_str());
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    ps::qwen35::SequenceSlotPool seq_pool = seq_res.release();

    void* row_positions = nullptr;
    void* row_slots = nullptr;
    void* rope_positions = nullptr;
    void* rope_inv_freq = nullptr;
    if (hipMalloc(&row_positions, 64u * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_slots, 64u * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&rope_positions, 64u * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&rope_inv_freq, 32u * sizeof(double)) != hipSuccess) {
        std::printf("FAIL: row metadata alloc\n");
        if (row_positions) (void)hipFree(row_positions);
        if (row_slots) (void)hipFree(row_slots);
        if (rope_positions) (void)hipFree(rope_positions);
        if (rope_inv_freq) (void)hipFree(rope_inv_freq);
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    std::vector<uint32_t> h_rows(64u, 0u);
    std::vector<uint32_t> h_slots(64u, 0u);
    for (uint32_t i = 0; i < 64u; ++i) h_rows[i] = i;
    if (hipMemcpy(row_positions, h_rows.data(),
                  h_rows.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(row_slots, h_slots.data(), h_slots.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(rope_positions, h_rows.data(),
                  h_rows.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: row metadata upload\n");
        if (row_positions) (void)hipFree(row_positions);
        if (row_slots) (void)hipFree(row_slots);
        if (rope_positions) (void)hipFree(rope_positions);
        if (rope_inv_freq) (void)hipFree(rope_inv_freq);
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    {
        float rope_theta = cfg.rope_theta;
        if (rope_theta <= 0.0f) rope_theta = 10000000.0f;
        const hipError_t rerr = ::ps::kernel::launch_rope_inv_freq_init(
            static_cast<double*>(rope_inv_freq), 32u, rope_theta, stream);
        if (rerr != hipSuccess) {
            std::printf("FAIL: rope inv_freq init: %s\n",
                        hipGetErrorString(rerr));
            if (row_positions) (void)hipFree(row_positions);
            if (row_slots) (void)hipFree(row_slots);
            if (rope_positions) (void)hipFree(rope_positions);
            if (rope_inv_freq) (void)hipFree(rope_inv_freq);
            if (workspace) (void)hipFree(workspace);
            arena.shutdown();
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return 1;
        }
    }
    ctx.row_positions = static_cast<const uint32_t*>(row_positions);
    ctx.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx.rope_positions = static_cast<const uint32_t*>(rope_positions);
    ctx.rope_inv_freq = static_cast<const double*>(rope_inv_freq);
    ctx.max_visible_tokens = env_u32("PHASESHIFT_ATTN_MAX_VISIBLE", 4096u);
    ctx.model_state.block_table = seq_pool.device_view();
    ctx.model_state.kv_bf16 = ps::qwen35::make_paged_kv_bf16_view(kv_pool);
    ctx.model_state.kv_dtype =
        static_cast<uint32_t>(ps::qwen35::KVCacheDType::BF16);
    std::printf(
        "kv pool: dtype=BF16 page_tokens=%u num_pages=%u layers=%u kv_heads=%u "
        "head_dim=%u block_table_stride=%u\n",
        kv_pool.page_tokens(), kv_pool.num_pages(), kv_pool.num_attention_layers(),
        kv_pool.kv_heads(), kv_pool.head_dim(),
        seq_pool.max_blocks_per_sequence());

    auto bcs_res = ps::runtime::create_device_batch_context(arena, 1u, 1024u);
    if (!bcs_res.ok()) {
        std::printf("FAIL: batch context: %s\n",
                    bcs_res.status().message().c_str());
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
                        &req, sizeof(req),
                      hipMemcpyHostToDevice) != hipSuccess) {
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

    std::printf(
        "index source debug_name kernel_id compute_spec in out weight ws mcu\n");
    uint32_t mcu_supported = 0u;
    uint32_t mcu_unsupported = 0u;
    uint32_t mcu_error = 0u;
    for (size_t i = 0; i < program.dispatches.size(); ++i) {
        const auto& b = program.dispatches[i];
        const uint32_t source =
            i < program.dispatch_source_node.size()
                ? program.dispatch_source_node[i]
                : 0xFFFFFFFFu;
        const char* debug_name =
            source < lowered.graph.nodes.size()
                ? lowered.graph.nodes[source].debug_name.c_str()
                : "<none>";

        ps::qwen35::runtime::McuPlanCompileOptions probe_options;
        probe_options.dispatch_begin = static_cast<uint32_t>(i);
        probe_options.dispatch_end = static_cast<uint32_t>(i) + 1u;
        probe_options.marker_grid = 1u;
        probe_options.marker_workgroup = 64u;
        ps::qwen35::runtime::McuCompiledPlan probe_plan;
        const ps::Status status = ps::qwen35::runtime::compile_mcu_plan(
            program, ctx, probe_options, probe_plan);
        const char* verdict = nullptr;
        if (status.ok()) {
            verdict = "yes";
            ++mcu_supported;
        } else if (status.code() == ps::Status::Code::unsupported) {
            verdict = "no";
            ++mcu_unsupported;
        } else {
            verdict = "error";
            ++mcu_error;
            std::printf("  compile[%zu]: %s\n", i, status.message().c_str());
        }
        if (b.kernel_id == ps::runtime::KernelId::RMS_NORM) {
            std::printf("  rmsnorm[%zu] %s verdict=%s%s%s\n", i, debug_name,
                        verdict, status.ok() ? "" : " reason=",
                        status.ok() ? "" : status.message().c_str());
            ps::qwen35::runtime::PhysicalRmsNormLaunch resolved{};
            const auto rstatus =
                ps::qwen35::runtime::resolve_rmsnorm_physical(program, b, ctx,
                                                              resolved);
            if (rstatus.ok()) {
                std::printf(
                    "      resolve=%d variant=%d in/out/w=%d/%d/%d layout=%d "
                    "mode=%d rows=%u F=%u G=%u\n",
                    static_cast<int>(rstatus.value()),
                    static_cast<int>(resolved.variant),
                    static_cast<int>(resolved.input_dtype),
                    static_cast<int>(resolved.output_dtype),
                    static_cast<int>(resolved.weight_dtype),
                    static_cast<int>(resolved.weight_layout),
                    static_cast<int>(resolved.weight_mode), resolved.rows,
                    resolved.features, resolved.group_size);
            } else {
                std::printf("      resolve failed: %s\n",
                            rstatus.status().message().c_str());
            }
        }

        if (b.kernel_id == ps::runtime::KernelId::SILU ||
            b.kernel_id == ps::runtime::KernelId::MUL ||
            b.kernel_id == ps::runtime::KernelId::SCALE ||
            b.kernel_id == ps::runtime::KernelId::RESIDUAL_ADD ||
            b.kernel_id == ps::runtime::KernelId::SWIGLU) {
            std::printf("  ew[%zu] %s %s\n", i, debug_name,
                        ps::runtime::kernel_to_name(b.kernel_id));
            for (uint32_t k = 0u; k < b.input_count; ++k) {
                const auto v = ps::qwen35::runtime::resolve_host_value(
                    program, ps::runtime::ValueId{b.input_slots[k]}, ctx);
                if (!v.ok()) {
                    std::printf("      in%u resolve failed\n", k);
                    continue;
                }
                const auto& x = v.value();
                std::printf("      in%u dtype=%d F=%u stride=%u\n", k,
                            static_cast<int>(x.dtype), x.feature_count,
                            x.row_stride);
            }
            for (uint32_t k = 0u; k < b.output_count; ++k) {
                const auto v = ps::qwen35::runtime::resolve_host_value(
                    program, ps::runtime::ValueId{b.output_slots[k]}, ctx);
                if (!v.ok()) {
                    std::printf("      out%u resolve failed\n", k);
                    continue;
                }
                const auto& x = v.value();
                std::printf("      out%u dtype=%d F=%u stride=%u scalar_a=%.6g\n",
                            k, static_cast<int>(x.dtype), x.feature_count,
                            x.row_stride, b.scalar_a);
            }
        }

        if (b.kernel_id == ps::runtime::KernelId::LINEAR_PSQ8) {
            ps::qwen35::runtime::PhysicalPsq8Launch resolved{};
            const auto rstatus = ps::qwen35::runtime::resolve_psq8_physical(
                program, b, ctx, resolved);
            std::printf("  psq8[%zu] %s verdict=%s%s%s\n", i, debug_name, verdict,
                        status.ok() ? "" : " reason=",
                        status.ok() ? "" : status.message().c_str());
            if (rstatus.ok()) {
                std::printf(
                    "      resolve=%d variant=%d use_decode1=%d unroll=%u "
                    "rows=%u out=%u k_padded=%u odt=%d grid=%u wg=%u\n",
                    static_cast<int>(rstatus.value()),
                    static_cast<int>(resolved.variant),
                    static_cast<int>(resolved.use_decode1), resolved.unroll,
                    resolved.rows, resolved.out_features, resolved.k_padded,
                    static_cast<int>(resolved.output_dtype), resolved.grid,
                    resolved.workgroup);
            } else {
                std::printf("      resolve failed: %s\n",
                            rstatus.status().message().c_str());
            }
        }

        if (b.kernel_id == ps::runtime::KernelId::LINEAR_BF16) {
            ps::qwen35::runtime::PhysicalBf16Launch resolved{};
            const auto rstatus = ps::qwen35::runtime::resolve_bf16_physical(
                program, b, ctx, resolved);
            std::printf("  bf16[%zu] %s verdict=%s%s%s\n", i, debug_name, verdict,
                        status.ok() ? "" : " reason=",
                        status.ok() ? "" : status.message().c_str());
            if (rstatus.ok()) {
                std::printf(
                    "      resolve=%d variant=%d cfg=%d exact_rows=%u rows=%u "
                    "n=%u k=%u in_rs=%u out_rs=%u odt=%d grid_x=%u wg=%u\n",
                    static_cast<int>(rstatus.value()),
                    static_cast<int>(resolved.variant),
                    static_cast<int>(resolved.config.id),
                    resolved.config.exact_rows, resolved.rows,
                    resolved.out_features, resolved.k, resolved.input_row_stride,
                    resolved.output_row_stride,
                    static_cast<int>(resolved.output_dtype), resolved.grid_x,
                    resolved.workgroup);
            } else {
                std::printf("      resolve failed: %s\n",
                            rstatus.status().message().c_str());
            }
        }

        if (b.kernel_id == ps::runtime::KernelId::L2_NORMALIZE) {
            const auto in = ps::qwen35::runtime::resolve_host_value(
                program, ps::runtime::ValueId{b.input_slots[0]}, ctx);
            const auto out = ps::qwen35::runtime::resolve_host_value(
                program, ps::runtime::ValueId{b.output_slots[0]}, ctx);
            std::printf("  l2[%zu] %s verdict=%s%s%s\n", i, debug_name, verdict,
                        status.ok() ? "" : " reason=",
                        status.ok() ? "" : status.message().c_str());
            if (in.ok() && out.ok()) {
                const auto& a = in.value();
                const auto& c = out.value();
                const uint32_t f = a.feature_count;
                const uint32_t g = (b.group_size != 0u && b.group_size <= f)
                                       ? b.group_size
                                       : f;
                std::printf(
                    "      in/out=%d/%d F=%u G=%u in_rs=%u out_rs=%u "
                    "scalar_a=%.6g rows=%u\n",
                    static_cast<int>(a.dtype), static_cast<int>(c.dtype), f, g,
                    a.row_stride, c.row_stride, b.scalar_a, ctx.actual_rows);
            } else {
                std::printf("      value resolve failed\n");
            }
        }

        if (b.kernel_id == ps::runtime::KernelId::GDN_RECURRENCE) {
            ps::qwen35::runtime::PhysicalGdnRecurrenceLaunch resolved{};
            const auto rstatus = ps::qwen35::runtime::resolve_gdn_recurrence_physical(
                program, b, ctx, resolved);
            std::printf("  rec[%zu] %s verdict=%s%s%s\n", i, debug_name, verdict,
                        status.ok() ? "" : " reason=",
                        status.ok() ? "" : status.message().c_str());
            if (rstatus.ok()) {
                std::printf(
                    "      resolve=%d variant=%d rows=%u nreq=%u max_req_rows=%u "
                    "head_k=%u head_v=%u grid_x=%u wg=%u\n",
                    static_cast<int>(rstatus.value()),
                    static_cast<int>(resolved.variant), resolved.rows,
                    resolved.num_requests, resolved.max_request_rows,
                    resolved.kernel.head_k, resolved.kernel.head_v,
                    resolved.grid_x, resolved.workgroup);
            } else {
                std::printf("      resolve failed: %s\n",
                            rstatus.status().message().c_str());
            }
        }

        if (b.kernel_id == ps::runtime::KernelId::EMBEDDING_LOOKUP) {
            const auto tok = ps::qwen35::runtime::resolve_host_value(
                program, ps::runtime::ValueId{b.input_slots[0]}, ctx);
            const auto outv = ps::qwen35::runtime::resolve_host_value(
                program, ps::runtime::ValueId{b.output_slots[0]}, ctx);
            std::printf(
                "  emb[%zu] %s verdict=%s weight_index=%u encoding=%u rows=%u "
                "cols=%u out_dtype=%d out_F=%u\n",
                i, debug_name, verdict, b.weight_index,
                b.weight_index < ctx.weight_count
                    ? ctx.weights[b.weight_index].encoding
                    : 0xFFFFFFFFu,
                b.weight_index < ctx.weight_count
                    ? ctx.weights[b.weight_index].rows
                    : 0u,
                b.weight_index < ctx.weight_count
                    ? ctx.weights[b.weight_index].cols
                    : 0u,
                outv.ok() ? static_cast<int>(outv.value().dtype) : -1,
                outv.ok() ? outv.value().feature_count : 0u);
        }

        std::printf(
            "%zu %u %s %s %u %u %u %u %u %s\n", i, source, debug_name,
            ps::runtime::kernel_to_name(b.kernel_id),
            static_cast<unsigned>(b.compute_spec), b.input_count,
            b.output_count,
            b.weight_index == ps::runtime::kNoWeight
                ? static_cast<unsigned>(-1)
                : b.weight_index,
            b.workspace_slot == ps::runtime::kNoWorkspace
                ? static_cast<unsigned>(-1)
                : b.workspace_slot,
            verdict);
    }
    std::printf("mcu_supported=%u mcu_unsupported=%u mcu_error=%u\n",
                mcu_supported, mcu_unsupported, mcu_error);

    uint32_t one_begin = 0u;
    uint32_t one_end = 0u;
    const ps::Status range_status = ps::qwen35::runtime::find_dispatch_range_for_graph_prefix(
        lowered.graph, program, "L0", &one_begin, &one_end);
    if (!range_status.ok()) {
        std::printf("FAIL: L0 range: %s\n", range_status.message().c_str());
        ++g_failed;
    } else {
        ps::qwen35::runtime::McuPlanCompileOptions one_opts;
        one_opts.dispatch_begin = one_begin;
        one_opts.dispatch_end = one_end;
        one_opts.marker_grid = 1u;
        one_opts.marker_workgroup = 64u;
        ps::qwen35::runtime::McuCompiledPlan one_plan;
        const ps::Status one_status = ps::qwen35::runtime::compile_mcu_plan(
            program, ctx, one_opts, one_plan);
        if (!one_status.ok()) {
            std::printf("FAIL: L0 plan: %s\n", one_status.message().c_str());
            ++g_failed;
        } else {
            const size_t inv_total = one_plan.rms.size() +
                                     one_plan.quant_e4m3.size() +
                                     one_plan.elementwise.size() +
                                     one_plan.rope.size() +
                                     one_plan.kv_append.size() +
                                     one_plan.bf16.size() +
                                     one_plan.l2.size() + one_plan.gdn_conv1d.size() +
                                     one_plan.gdn_recurrence.size() +
                                     one_plan.psq4.size();
            std::printf(
                "one-layer plan: dispatch=[%u,%u) nodes=%zu variants=%zu "
                "kernarg_slots=%u marker=%u\n",
                one_begin, one_end, one_plan.nodes.size(),
                one_plan.variants.size(), one_plan.kernarg_slots_required,
                one_plan.marker_count);
            std::printf(
                "  invocations: rms=%zu e4m3=%zu elementwise=%zu rope=%zu kv=%zu "
                "bf16=%zu l2=%zu conv1d=%zu recurrence=%zu psq4=%zu (total=%zu)\n",
                one_plan.rms.size(), one_plan.quant_e4m3.size(),
                one_plan.elementwise.size(), one_plan.rope.size(),
                one_plan.kv_append.size(), one_plan.bf16.size(),
                one_plan.l2.size(), one_plan.gdn_conv1d.size(),
                one_plan.gdn_recurrence.size(), one_plan.psq4.size(), inv_total);
            const uint64_t bytes_nodes =
                static_cast<uint64_t>(one_plan.nodes.size()) * 28u;
            std::printf("  plan bytes: nodes=%llu\n",
                        static_cast<unsigned long long>(bytes_nodes));
            check(one_plan.variants.size() <= ::ps::runtime::gpu_mcu::kMcuMaxVariants,
                  "one layer variants within kMcuMaxVariants");
            check(one_plan.nodes.size() <= ::ps::runtime::gpu_mcu::kMcuMaxNodes,
                  "one layer nodes within kMcuMaxNodes");
            check(one_plan.kernarg_slots_required <= one_plan.nodes.size(),
                  "one layer kernarg slots bounded by node count");
            check(inv_total == static_cast<size_t>(one_end - one_begin),
                  "one layer invocation per dispatch plus the gdn reset node");
        }
    }

    const ps::Status bc_destroy =
        ps::runtime::destroy_device_batch_context(batch_storage);
    check(bc_destroy.ok(), "device batch context destroyed");
    if (rope_inv_freq) (void)hipFree(rope_inv_freq);
    if (rope_positions) (void)hipFree(rope_positions);
    if (row_slots) (void)hipFree(row_slots);
    if (row_positions) (void)hipFree(row_positions);
    if (workspace) (void)hipFree(workspace);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_qwen_one_layer_inventory: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_qwen_one_layer_inventory: PASS\n");
    return 0;
}
