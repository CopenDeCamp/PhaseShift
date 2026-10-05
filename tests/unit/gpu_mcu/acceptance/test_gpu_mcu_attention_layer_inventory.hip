#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/elementwise_selector.h>
#include <phaseshift/models/qwen35/runtime/kv_append_dispatch.h>
#include <phaseshift/models/qwen35/runtime/kv_append_selector.h>
#include <phaseshift/models/qwen35/runtime/mcu_layer_range.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

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

const char* default_model_dir() {
#ifdef PS_MODEL_DIR_4B
    return PS_MODEL_DIR_4B;
#else
    return "";
#endif
}

const char* elementwise_profile_name(int p) {
    using EP = ::ps::qwen35::runtime::ElementwiseProfile;
    switch (static_cast<EP>(p)) {
        case EP::ResidualBf16: return "ResidualBf16";
        case EP::SiluBf16ToF32: return "SiluBf16ToF32";
        case EP::SiluF32ToBf16: return "SiluF32ToBf16";
        case EP::SigmoidBf16ToF32: return "SigmoidBf16ToF32";
        case EP::MulF32F32ToBf16: return "MulF32F32Bf16";
        case EP::SwigluBf16: return "SwigluBf16";
        case EP::ScaleF32: return "ScaleF32";
        case EP::SplitBf16Halves: return "SplitBf16Halves";
        case EP::SplitBf16InterleavedHeads: return "SplitBf16InterleavedHeads";
        default: return "Count";
    }
}

const char* rmsnorm_variant_name(int v) {
    using RV = ::ps::qwen35::runtime::PhysicalRmsNormVariant;
    switch (static_cast<RV>(v)) {
        case RV::Bf16PfOnePlus: return "Bf16PfOnePlus";
        case RV::F32PgDirect: return "F32PgDirect";
        case RV::Bf16F32PgOnePlus: return "Bf16F32PgOnePlus";
        default: return "Unsupported";
    }
}

const char* bf16_variant_name(int v) {
    using BV = ::ps::qwen35::runtime::PhysicalBf16Variant;
    switch (static_cast<BV>(v)) {
        case BV::ExactRows: return "ExactRows";
        case BV::WmmaKPartition: return "WmmaKPartition";
        case BV::WmmaWide: return "WmmaWide";
        case BV::Wmma: return "Wmma";
        default: return "Unsupported";
    }
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

    const char* from_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    const std::string model_dir =
        (from_env != nullptr && from_env[0] != '\0') ? from_env
                                                     : default_model_dir();
    if (!directory_exists(model_dir.c_str())) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_4B to a Qwen3.5 model dir\n");
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

    std::printf("=== standard attention layer inventory ===\n");
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
        std::printf("FAIL: model load: %s\n", model_result.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();

    auto lower_result = ps::qwen35::lower_qwen35_to_primitives(
        model.text_config(), model.weights());
    if (!lower_result.ok()) {
        std::printf("FAIL: lowering: %s\n", lower_result.status().message().c_str());
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
        std::printf("FAIL: build_program: %s\n", program_result.status().message().c_str());
        arena.shutdown();
        return 1;
    }
    ps::runtime::Program program = program_result.release();
    check(program.dispatch_source_node.size() == program.dispatches.size(),
          "program carries dispatch provenance");

    const auto& cfg = model.text_config();

    std::printf("graph_nodes=%zu dispatches=%zu weights=%zu params=%zu\n",
                lowered.graph.nodes.size(), program.dispatches.size(),
                lowered.weights.size(), lowered.parameters.size());

    // ---- Gate A0: first standard attention layer -------------------------
    int first_attention_layer = -1;
    for (std::size_t i = 0; i < cfg.layer_types.size(); ++i) {
        if (cfg.layer_types[i] == 1) {
            first_attention_layer = static_cast<int>(i);
            break;
        }
    }
    if (first_attention_layer < 0) {
        std::printf("FAIL: model has no standard attention layer\n");
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    const std::string prefix =
        "L" + std::to_string(first_attention_layer) + ".";
    std::printf("first_attention_layer=%d prefix=%s\n", first_attention_layer,
                prefix.c_str());

    uint32_t range_begin = 0u;
    uint32_t range_end = 0u;
    const ps::Status range_status =
        ps::qwen35::runtime::find_dispatch_range_for_graph_prefix(
            lowered.graph, program, prefix.c_str(), &range_begin, &range_end);
    if (!range_status.ok()) {
        std::printf("FAIL: attention range: %s\n", range_status.message().c_str());
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    std::printf("attention range=[%u,%u) dispatches=%u\n", range_begin, range_end,
                range_end - range_begin);

    // ---- Gate A0.2: attention shape from the real config ------------------
    const uint32_t hidden_size = static_cast<uint32_t>(cfg.hidden_size);
    const uint32_t q_heads = static_cast<uint32_t>(cfg.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(cfg.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
    const uint32_t q_features = q_heads * head_dim;
    const uint32_t kv_features = kv_heads * head_dim;
    const uint32_t rotary_dim =
        head_dim > 0u
            ? static_cast<uint32_t>(static_cast<double>(head_dim) *
                                    static_cast<double>(cfg.partial_rotary_factor))
            : 0u;
    uint32_t attention_layers = 0u;
    for (int lt : cfg.layer_types)
        if (lt == 1) ++attention_layers;
    const uint32_t page_tokens = 16u;
    const uint32_t max_visible_tokens =
        env_u32("PHASESHIFT_ATTN_MAX_VISIBLE", 4096u);
    const uint32_t rows = 1u;

    std::printf("A0.2 hidden=%u q_features=%u kv_features=%u q_heads=%u kv_heads=%u "
                "head_dim=%u rotary_dim=%u rope_theta=%.6g partial_rotary=%.6g\n",
                hidden_size, q_features, kv_features, q_heads, kv_heads, head_dim,
                rotary_dim, cfg.rope_theta, cfg.partial_rotary_factor);
    std::printf("A0.2 rows=%u page_tokens=%u attention_layers=%u "
                "max_visible_tokens=%u intermediate=%zu layers=%zu\n",
                rows, page_tokens, attention_layers, max_visible_tokens,
                cfg.intermediate_size, cfg.num_hidden_layers);

    // ---- workspace / external slots --------------------------------------
    void* workspace = nullptr;
    if (program.workspace.total_bytes != 0u &&
        hipMalloc(&workspace, program.workspace.total_bytes) != hipSuccess) {
        std::printf("FAIL: workspace alloc\n");
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    std::vector<const void*> external_inputs(program.external_input_count,
                                             reinterpret_cast<const void*>(0x1000));
    std::vector<void*> external_outputs(program.external_output_count,
                                        reinterpret_cast<void*>(0x2000));

    // ---- Gate A8 state: KV pool + block table + row metadata -------------
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

    auto seq_res =
        ps::qwen35::SequenceSlotPool::create(arena, 8u, 2048u);
    if (!seq_res.ok()) {
        std::printf("FAIL: sequence slot pool: %s\n", seq_res.status().message().c_str());
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    ps::qwen35::SequenceSlotPool seq_pool = seq_res.release();

    void* partials = nullptr;
    if (hipMalloc(&partials, ::ps::qwen35::runtime::kDecodeAttnPartialBytes) !=
        hipSuccess) {
        std::printf("FAIL: decode attention partials alloc\n");
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }

    void* row_positions = nullptr;
    void* row_slots = nullptr;
    void* rope_positions = nullptr;
    void* inv_freq = nullptr;
    if (hipMalloc(&row_positions, 64u * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_slots, 64u * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&rope_positions, 64u * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&inv_freq, 128u * sizeof(double)) != hipSuccess) {
        std::printf("FAIL: row metadata alloc\n");
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    std::vector<uint32_t> h_rows(64u, 0u);
    std::vector<uint32_t> h_slots(64u, 0u);
    for (uint32_t i = 0; i < 64u; ++i) h_rows[i] = i;
    if (hipMemcpy(row_positions, h_rows.data(), h_rows.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(row_slots, h_slots.data(), h_slots.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(rope_positions, h_rows.data(), h_rows.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: row metadata upload\n");
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    // ---- batch context ---------------------------------------------------
    const auto gdn_layout =
        ps::qwen35::GdnStatePoolLayout::from_text_config(model.text_config());
    auto gdn_pool_res = ps::qwen35::GdnStatePool::create(arena, 1u, gdn_layout);
    if (!gdn_pool_res.ok()) {
        std::printf("FAIL: gdn state pool: %s\n",
                    gdn_pool_res.status().message().c_str());
        return 1;
    }
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_res.release();

    auto bcs_res = ps::runtime::create_device_batch_context(arena, 1u, 1024u);
    if (!bcs_res.ok()) {
        std::printf("FAIL: batch context: %s\n", bcs_res.status().message().c_str());
        if (workspace) (void)hipFree(workspace);
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    ps::runtime::DeviceBatchContextStorage batch_storage = bcs_res.release();
    {
        ::ps::runtime::DeviceRequestDescriptor req;
        std::memset(&req, 0, sizeof(req));
        req.request_handle.slot = 0u;
        req.row_begin = 0u;
        req.row_count = rows;
        if (hipMemcpy(batch_storage.requests
                          .data<::ps::runtime::DeviceRequestDescriptor>(),
                      &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: batch request upload\n");
            if (workspace) (void)hipFree(workspace);
            arena.shutdown();
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return 1;
        }
        ::ps::runtime::DeviceBatchContext bc;
        std::memset(&bc, 0, sizeof(bc));
        bc.actual_rows = rows;
        bc.num_requests = 1u;
        bc.requests = batch_storage.requests
                          .data<::ps::runtime::DeviceRequestDescriptor>();
        if (hipMemcpy(batch_storage.device_struct
                          .data<::ps::runtime::DeviceBatchContext>(),
                      &bc, sizeof(bc), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: batch context upload\n");
            if (workspace) (void)hipFree(workspace);
            arena.shutdown();
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            return 1;
        }
    }

    ps::qwen35::runtime::HostExecutionContext ctx;
    ctx.batch_context =
        batch_storage.device_struct.data<::ps::runtime::DeviceBatchContext>();
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
    ctx.actual_rows = rows;
    ctx.actual_outputs = rows;
    ctx.row_positions = static_cast<const uint32_t*>(row_positions);
    ctx.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx.rope_positions = static_cast<const uint32_t*>(rope_positions);
    ctx.rope_inv_freq = static_cast<const double*>(inv_freq);
    ctx.max_visible_tokens = max_visible_tokens;
    ctx.decode_attn_partials = static_cast<uint8_t*>(partials);
    ctx.decode_attn_partial_bytes =
        static_cast<uint32_t>(::ps::qwen35::runtime::kDecodeAttnPartialBytes);

    {
        ::ps::kernel::ModelDispatchStateView ms{};
        ms.block_table = seq_pool.device_view();
        ms.gdn = gdn_pool.device_view();
        ms.kv_bf16 = ps::qwen35::make_paged_kv_bf16_view(kv_pool);
        ms.kv_dtype = static_cast<uint32_t>(ps::qwen35::KVCacheDType::BF16);
        ctx.model_state = ms;
    }

    std::printf(
        "kv pool: dtype=BF16 page_tokens=%u num_pages=%u layers=%u kv_heads=%u "
        "head_dim=%u\n",
        kv_pool.page_tokens(), kv_pool.num_pages(),
        kv_pool.num_attention_layers(), kv_pool.kv_heads(), kv_pool.head_dim());
    std::printf("block table: max_sequences=%u stride=%u capacity=%u\n",
                seq_pool.max_sequences(), seq_pool.max_blocks_per_sequence(),
                seq_pool.max_blocks_per_sequence() * page_tokens);

    // ---- Gate A0.1: per dispatch dump ------------------------------------
    std::printf("A0.1 index source debug_name kernel_id compute_spec desc in out "
                "weight ws mcu\n");
    uint32_t mcu_supported = 0u;
    uint32_t mcu_unsupported = 0u;
    uint32_t mcu_error = 0u;
    uint32_t pa_dispatches = 0u;
    for (uint32_t i = range_begin; i < range_end; ++i) {
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
        probe_options.dispatch_begin = i;
        probe_options.dispatch_end = i + 1u;
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
        }

        std::printf("%u %u %s %s spec=%u desc=%u", i, source, debug_name,
                    ps::runtime::kernel_to_name(b.kernel_id),
                    b.compute_spec, b.descriptor_slot);
        std::printf(" in=");
        for (uint32_t k = 0u; k < b.input_count; ++k) {
            const auto v = ps::qwen35::runtime::resolve_host_value(
                program, ps::runtime::ValueId{b.input_slots[k]}, ctx);
            if (v.ok()) {
                const auto& x = v.value();
                std::printf("%d/F%u/S%u ", static_cast<int>(x.dtype),
                            x.feature_count, x.row_stride);
            } else {
                std::printf("? ");
            }
        }
        std::printf("out=");
        for (uint32_t k = 0u; k < b.output_count; ++k) {
            const auto v = ps::qwen35::runtime::resolve_host_value(
                program, ps::runtime::ValueId{b.output_slots[k]}, ctx);
            if (v.ok()) {
                const auto& x = v.value();
                std::printf("%d/F%u/S%u ", static_cast<int>(x.dtype),
                            x.feature_count, x.row_stride);
            } else {
                std::printf("? ");
            }
        }
        std::printf("weight=%u ws=%u mcu=%s",
                    b.weight_index == ps::runtime::kNoWeight
                        ? 0xFFFFFFFFu
                        : b.weight_index,
                    b.workspace_slot == ps::runtime::kNoWorkspace
                        ? 0xFFFFFFFFu
                        : b.workspace_slot,
                    verdict);
        if (!status.ok()) std::printf(" reason=%s", status.message().c_str());

        if (b.kernel_id == ps::runtime::KernelId::RMS_NORM) {
            ps::qwen35::runtime::PhysicalRmsNormLaunch r{};
            const auto rs = ps::qwen35::runtime::resolve_rmsnorm_physical(
                program, b, ctx, r);
            if (rs.ok()) {
                std::printf(" |rms resolve=%d variant=%s in/out/wdt=%d/%d/%d "
                            "wlayout=%d mode=%d rows=%u F=%u G=%u grid=%u wg=%u",
                            static_cast<int>(rs.value()),
                            rmsnorm_variant_name(static_cast<int>(r.variant)),
                            static_cast<int>(r.input_dtype),
                            static_cast<int>(r.output_dtype),
                            static_cast<int>(r.weight_dtype),
                            static_cast<int>(r.weight_layout),
                            static_cast<int>(r.weight_mode), r.rows, r.features,
                            r.group_size, r.grid, r.workgroup);
            } else {
                std::printf(" |rms resolve=failed(%s)",
                            rs.status().message().c_str());
            }
        } else if (b.kernel_id == ps::runtime::KernelId::LINEAR_BF16) {
            ps::qwen35::runtime::PhysicalBf16Launch r{};
            const auto rs = ps::qwen35::runtime::resolve_bf16_physical(
                program, b, ctx, r);
            if (rs.ok()) {
                std::printf(" |bf16 resolve=%d variant=%s rows=%u out=%u k=%u "
                            "grid=%u,%u wg=%u",
                            static_cast<int>(rs.value()),
                            bf16_variant_name(static_cast<int>(r.variant)),
                            r.rows, r.out_features, r.k, r.grid_x, r.grid_y,
                            r.workgroup);
            } else {
                std::printf(" |bf16 resolve=failed(%s)",
                            rs.status().message().c_str());
            }
        } else if (b.kernel_id == ps::runtime::KernelId::SPLIT ||
                   b.kernel_id == ps::runtime::KernelId::SIGMOID ||
                   b.kernel_id == ps::runtime::KernelId::SILU ||
                   b.kernel_id == ps::runtime::KernelId::MUL ||
                   b.kernel_id == ps::runtime::KernelId::SCALE ||
                   b.kernel_id == ps::runtime::KernelId::RESIDUAL_ADD ||
                   b.kernel_id == ps::runtime::KernelId::SWIGLU ||
                   b.kernel_id == ps::runtime::KernelId::CONCAT) {
            ps::qwen35::runtime::PhysicalElementwiseLaunch r{};
            const auto rs = ps::qwen35::runtime::resolve_elementwise_physical(
                program, b, ctx, r);
            if (rs.ok()) {
                std::printf(" |ew resolve=%d profile=%s rows=%u F=%u Fb=%u aux=%u "
                            "grid=%u,%u wg=%u",
                            static_cast<int>(rs.value()),
                            elementwise_profile_name(static_cast<int>(r.profile)),
                            r.rows, r.features, r.features_b, r.aux, r.grid_x,
                            r.grid_y, r.workgroup);
            } else {
                std::printf(" |ew resolve=failed(%s)",
                            rs.status().message().c_str());
            }
        } else if (b.kernel_id == ps::runtime::KernelId::ROPE) {
            ps::qwen35::runtime::PhysicalRopeLaunch r{};
            const auto rs = ps::qwen35::runtime::resolve_rope_physical(
                program, b, ctx, r);
            if (rs.ok()) {
                const char* vname =
                    r.variant == ps::qwen35::runtime::PhysicalRopeVariant::F32Bf16Pair
                        ? "F32Bf16Pair"
                        : (r.variant ==
                                   ps::qwen35::runtime::PhysicalRopeVariant::F32Bf16Generic
                               ? "F32Bf16Generic"
                               : "Unsupported");
                std::printf(
                    " |rope resolve=%d variant=%s rows=%u F=%u head_dim=%u "
                    "rotary_dim=%u theta=%.6g grid=%u wg=%u",
                    static_cast<int>(rs.value()), vname, r.rows, r.features,
                    r.head_dim, r.rotary_dim, r.theta, r.grid, r.workgroup);
            } else {
                std::printf(" |rope resolve=failed(%s)",
                            rs.status().message().c_str());
            }
        } else if (b.kernel_id == ps::runtime::KernelId::KV_APPEND) {
            ps::qwen35::runtime::PhysicalKvAppendLaunch ka{};
            const auto ks = ps::qwen35::runtime::resolve_kv_append_physical(
                program, i, ctx, ka);
            if (ks.ok()) {
                const char* vname =
                    ka.variant == ps::qwen35::runtime::PhysicalKvAppendVariant::Bf16
                        ? "Bf16"
                        : (ka.variant ==
                                   ps::qwen35::runtime::PhysicalKvAppendVariant::Fp8E4M3
                               ? "Fp8E4M3"
                               : (ka.variant == ps::qwen35::runtime::PhysicalKvAppendVariant::Psq4W32
                                      ? "Psq4W32"
                                      : "Unsupported"));
                std::printf(
                    " |kv_append resolve=%d variant=%s layer=%u rows=%u kv_heads=%u "
                    "head_dim=%u page_tokens=%u grid=%u wg=%u",
                    static_cast<int>(ks.value()), vname, ka.common.layer,
                    ka.common.rows, ka.common.kv_heads, ka.common.head_dim,
                    ka.common.page_tokens, ka.grid, ka.workgroup);
            } else {
                std::printf(" |kv_append resolve=failed(%s)",
                            ks.status().message().c_str());
            }
        } else if (b.kernel_id == ps::runtime::KernelId::PAGED_ATTENTION) {
            ++pa_dispatches;
            const auto pa =
                ps::qwen35::runtime::resolve_paged_attention_args(program, i, ctx);
            if (pa.ok()) {
                const ps::qwen35::runtime::PagedAttentionPlan p =
                    ps::qwen35::runtime::plan_paged_attention(pa.value(), ctx,
                                                              false);
                std::printf(
                    " |pa optimized=%d use_prefill=%d splits=%u kv_dtype=%d "
                    "out_dtype=%d rows=%u q_heads=%u kv_heads=%u head_dim=%u "
                    "page_tokens=%u layer=%u",
                    p.optimized ? 1 : 0, p.use_prefill ? 1 : 0, p.splits,
                    static_cast<int>(pa.value().kv_dtype),
                    static_cast<int>(pa.value().output_dtype),
                    pa.value().common.rows, pa.value().common.q_heads,
                    pa.value().common.kv_heads, pa.value().common.head_dim,
                    pa.value().common.page_tokens, pa.value().common.layer);
            } else {
                std::printf(" |pa resolve=failed(%s)",
                            pa.status().message().c_str());
            }
        }
        std::printf("\n");
    }
    std::printf("range mcu_supported=%u mcu_unsupported=%u mcu_error=%u\n",
                mcu_supported, mcu_unsupported, mcu_error);
    check(pa_dispatches == 1u, "attention layer has exactly one paged attention");

    // ---- Gate A0.3: paged attention plan under a max_visible sweep -------
    std::vector<uint32_t> pa_index;
    for (uint32_t i = range_begin; i < range_end; ++i) {
        if (program.dispatches[i].kernel_id ==
            ps::runtime::KernelId::PAGED_ATTENTION) {
            pa_index.push_back(i);
        }
    }
    if (pa_index.empty()) {
        std::printf("FAIL: no PAGED_ATTENTION dispatch in the attention layer\n");
        ++g_failed;
    } else {
        const uint32_t saved_max_visible = ctx.max_visible_tokens;
        const uint32_t sweep[] = {1u, 1024u, 2047u, 2048u, 4096u, 32768u};
        std::printf("A0.3 sweep: max_visible -> optimized/use_prefill/splits\n");
        for (uint32_t mv : sweep) {
            ctx.max_visible_tokens = mv;
            const auto pa = ps::qwen35::runtime::resolve_paged_attention_args(
                program, pa_index[0], ctx);
            if (!pa.ok()) {
                std::printf("  max_visible=%u resolve failed: %s\n", mv,
                            pa.status().message().c_str());
                continue;
            }
            const ps::qwen35::runtime::PagedAttentionPlan p =
                ps::qwen35::runtime::plan_paged_attention(pa.value(), ctx, false);
            std::printf("  max_visible=%u optimized=%d use_prefill=%d splits=%u\n",
                        mv, p.optimized ? 1 : 0, p.use_prefill ? 1 : 0, p.splits);
        }
        ctx.max_visible_tokens = saved_max_visible;
    }

    // ---- coverage at the Gate decode condition (splits == 1) --------------
    {
        const uint32_t saved = ctx.max_visible_tokens;
        ctx.max_visible_tokens = 512u;
        uint32_t ok_count = 0u;
        uint32_t no_count = 0u;
        for (uint32_t i = range_begin; i < range_end; ++i) {
            ps::qwen35::runtime::McuPlanCompileOptions probe;
            probe.dispatch_begin = i;
            probe.dispatch_end = i + 1u;
            probe.marker_grid = 1u;
            probe.marker_workgroup = 64u;
            ps::qwen35::runtime::McuCompiledPlan p;
            const ps::Status st =
                ps::qwen35::runtime::compile_mcu_plan(program, ctx, probe, p);
            if (st.ok()) {
                ++ok_count;
            } else if (st.code() == ps::Status::Code::unsupported) {
                ++no_count;
                std::printf("  decode-condition unsupported[%u]: %s\n", i,
                            st.message().c_str());
            }
        }
        std::printf(
            "decode condition max_visible=512: mcu_supported=%u mcu_unsupported=%u\n",
            ok_count, no_count);
        check(no_count == 0u,
              "attention layer fully supported at the decode condition");
        ctx.max_visible_tokens = saved;
    }

    // ---- whole range compile --------------------------------------------
    {
        ps::qwen35::runtime::McuPlanCompileOptions opts;
        opts.dispatch_begin = range_begin;
        opts.dispatch_end = range_end;
        opts.marker_grid = 1u;
        opts.marker_workgroup = 64u;
        ps::qwen35::runtime::McuCompiledPlan plan;
        const ps::Status st =
            ps::qwen35::runtime::compile_mcu_plan(program, ctx, opts, plan);
        if (st.ok()) {
            const size_t inv_total = plan.rms.size() +
                                     plan.quant_e4m3.size() +
                                     plan.elementwise.size() + plan.rope.size() + plan.kv_append.size() +
                                     plan.attention_paged.size() + plan.bf16.size() +
                                     plan.l2.size() + plan.gdn_conv1d.size() +
                                     plan.gdn_recurrence.size();
            std::printf(
                "attention plan: nodes=%zu variants=%zu kernarg_slots=%u "
                "marker=%u logical=%u physical=%u invocations=%zu\n",
                plan.nodes.size(), plan.variants.size(),
                plan.kernarg_slots_required, plan.marker_count,
                plan.logical_dispatch_count, plan.physical_dispatch_count,
                inv_total);
            std::printf(
                "  invocations: rms=%zu e4m3=%zu elementwise=%zu rope=%zu kv=%zu "
                "pa=%zu bf16=%zu l2=%zu conv1d=%zu recurrence=%zu\n",
                plan.rms.size(), plan.quant_e4m3.size(), plan.elementwise.size(),
                plan.rope.size(), plan.kv_append.size(),
                plan.attention_paged.size(), plan.bf16.size(), plan.l2.size(),
                plan.gdn_conv1d.size(), plan.gdn_recurrence.size());
        } else {
            std::printf("attention plan: compile failed: %s\n",
                        st.message().c_str());
        }
    }

    // ---- Gate A15: transformer body whole-range compile-only -------------
    {
        uint32_t body_begin = 0u;
        uint32_t body_end = 0u;
        const ps::Status bst = ps::qwen35::runtime::find_dispatch_range_for_graph_prefix(
            lowered.graph, program, "L", &body_begin, &body_end);
        if (!bst.ok()) {
            std::printf("FAIL: transformer body range: %s\n",
                        bst.message().c_str());
            ++g_failed;
        } else {
            const uint32_t saved = ctx.max_visible_tokens;
            ctx.max_visible_tokens = 512u;
            ps::qwen35::runtime::McuPlanCompileOptions bo;
            bo.dispatch_begin = body_begin;
            bo.dispatch_end = body_end;
            bo.marker_grid = 1u;
            bo.marker_workgroup = 64u;
            ps::qwen35::runtime::McuCompiledPlan bp;
            const ps::Status bps =
                ps::qwen35::runtime::compile_mcu_plan(program, ctx, bo, bp);
            std::printf(
                "body compile: logical=%u physical=%u nodes=%zu variants=%zu "
                "kernarg_slots=%u invocations=%zu\n",
                body_end - body_begin, bp.physical_dispatch_count,
                bp.nodes.size(), bp.variants.size(), bp.kernarg_slots_required,
                bp.rms.size() + bp.elementwise.size() + bp.rope.size() +
                    bp.kv_append.size() + bp.attention_paged.size() +
                    bp.bf16.size() + bp.l2.size() + bp.gdn_conv1d.size() +
                    bp.gdn_recurrence.size() + bp.quant_e4m3.size() +
                    bp.psq4.size());
            if (bps.ok()) {
                std::printf(
                    "body compile: ok variants_vs_limit=%zu/%u nodes_vs_limit=%zu/%u\n",
                    bp.variants.size(), ::ps::runtime::gpu_mcu::kMcuMaxVariants,
                    bp.nodes.size(), ::ps::runtime::gpu_mcu::kMcuMaxNodes);
            } else {
                std::printf("body compile unsupported: %s\n",
                            bps.message().c_str());
            }
            ctx.max_visible_tokens = saved;
        }
    }

    const ps::Status bc_destroy =
        ps::runtime::destroy_device_batch_context(batch_storage);
    check(bc_destroy.ok(), "device batch context destroyed");
    if (inv_freq) (void)hipFree(inv_freq);
    if (rope_positions) (void)hipFree(rope_positions);
    if (row_slots) (void)hipFree(row_slots);
    if (row_positions) (void)hipFree(row_positions);
    if (partials) (void)hipFree(partials);
    if (workspace) (void)hipFree(workspace);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_attention_layer_inventory: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_attention_layer_inventory: PASS\n");
    return 0;
}
