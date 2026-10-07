#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/embedding.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
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
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/wall_clock.h>

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

namespace mcu = ::ps::runtime::gpu_mcu;
namespace q35 = ::ps::qwen35::runtime;
namespace kernel = ::ps::kernel;
namespace psq = ::ps::qwen35;
using gpu_mcu_test::Rig;

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

bool bytes_equal(const void* a, const void* b, size_t bytes, size_t* first,
                 size_t* diffs) {
    const auto* x = static_cast<const unsigned char*>(a);
    const auto* y = static_cast<const unsigned char*>(b);
    *first = 0u;
    *diffs = 0u;
    for (size_t i = 0; i < bytes; ++i) {
        if (x[i] == y[i]) continue;
        if (*diffs == 0u) *first = i;
        ++(*diffs);
    }
    return *diffs == 0u;
}

size_t dtype_bytes(::ps::runtime::ValueDType dt) {
    return dt == ::ps::runtime::ValueDType::F32 ? 4u : 2u;
}

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

constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kMaxSequences = 2u;
constexpr uint32_t kPosition = 100u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;

struct Checkpoint {
    uint32_t layer = 0u;
    uint32_t dispatch = 0u;
    void* ptr = nullptr;
    size_t bytes = 0u;
    std::vector<uint8_t> host;
    std::vector<uint8_t> mcu;
};

void compare_or_report(const char* what, const std::vector<uint8_t>& host,
                       const std::vector<uint8_t>& mcu, int* failed) {
    if (host.size() != mcu.size()) {
        std::printf("FAIL: %s size host=%zu mcu=%zu\n", what, host.size(),
                    mcu.size());
        ++(*failed);
        return;
    }
    if (host.empty()) return;
    size_t first = 0;
    size_t diffs = 0;
    if (bytes_equal(host.data(), mcu.data(), host.size(), &first, &diffs)) return;
    std::printf("FAIL: %s differs bytes=%zu first=%zu count=%zu\n", what,
                host.size(), first, diffs);
    ++(*failed);
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

    const auto wall_rate = mcu::gpu_wall_clock_rate_khz(0);
    if (!wall_rate.ok()) {
        std::printf("FAIL: wall clock rate: %s\n",
                    wall_rate.status().message().c_str());
        return 1;
    }
    const int wall_clock_rate_khz = wall_rate.value();
    std::printf("wall_clock_rate_khz=%d\n", wall_clock_rate_khz);

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
        return 1;
    }
    ps::runtime::Program program = program_result.release();

    const auto& ltc = model.text_config().layer_types;
    uint32_t attention_layers = 0u;
    uint32_t gdn_layers = 0u;
    for (uint32_t t : ltc) {
        if (t == 1) ++attention_layers;
        else ++gdn_layers;
    }
    std::printf("model: layers=%zu attention=%u linear=%u\n", ltc.size(),
                attention_layers, gdn_layers);

    uint32_t begin = 0u;
    uint32_t end = 0u;
    const ps::Status range_status =
        q35::find_dispatch_range_for_graph_prefix(lowered.graph, program, "L",
                                                  &begin, &end);
    if (!range_status.ok()) {
        std::printf("FAIL: body range: %s\n", range_status.message().c_str());
        return 1;
    }
    const uint32_t body_dispatch_count = end - begin;
    uint32_t plan_begin = begin;
    if (begin >= 1u &&
        program.dispatches[begin - 1u].kernel_id ==
            ps::runtime::KernelId::EMBEDDING_LOOKUP) {
        plan_begin = begin - 1u;
    }
    const uint32_t dispatch_count = end - plan_begin;
    std::printf("transformer body range=[%u,%u) plan_begin=%u dispatches=%u\n",
                begin, end, plan_begin, dispatch_count);
    (void)body_dispatch_count;

    void* workspace = nullptr;
    if (program.workspace.total_bytes != 0u &&
        hipMalloc(&workspace, program.workspace.total_bytes) != hipSuccess) {
        std::printf("FAIL: workspace alloc\n");
        return 1;
    }
    constexpr size_t kExternalBytes = 8u * 1024u * 1024u;
    std::vector<void*> external_alloc;
    std::vector<const void*> external_inputs(program.external_input_count, nullptr);
    std::vector<void*> external_outputs(program.external_output_count, nullptr);
    for (uint32_t i = 0; i < program.external_input_count; ++i) {
        void* p = nullptr;
        if (hipMalloc(&p, kExternalBytes) != hipSuccess ||
            hipMemset(p, 0, kExternalBytes) != hipSuccess) {
            std::printf("FAIL: external input alloc\n");
            return 1;
        }
        external_alloc.push_back(p);
        external_inputs[i] = p;
    }
    for (uint32_t i = 0; i < program.external_output_count; ++i) {
        void* p = nullptr;
        if (hipMalloc(&p, kExternalBytes) != hipSuccess ||
            hipMemset(p, 0, kExternalBytes) != hipSuccess) {
            std::printf("FAIL: external output alloc\n");
            return 1;
        }
        external_alloc.push_back(p);
        external_outputs[i] = p;
    }
    std::printf("external inputs=%zu outputs=%zu buffers=%zu\n",
                external_inputs.size(), external_outputs.size(),
                external_alloc.size());

    uint32_t max_visible = 512u;
    if (const char* mv = std::getenv("PHASESHIFT_BODY_MAX_VISIBLE")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= static_cast<long>(kPageTokens)) {
            max_visible = static_cast<uint32_t>(v);
        }
    }
    const uint32_t block_table_stride = max_visible / kPageTokens;
    const uint32_t num_pages = block_table_stride;
    std::printf("max_visible=%u block_table_stride=%u num_pages=%u\n", max_visible,
                block_table_stride, num_pages);
    const size_t pool_bytes = static_cast<size_t>(attention_layers) *
                              num_pages * kElemsPerPage * sizeof(uint16_t);

    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* partials = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* rope_pos = nullptr;
    void* inv_freq = nullptr;
    void* block_tables = nullptr;
    void* error_word = nullptr;
    if (hipMalloc(&k_pool, pool_bytes) != hipSuccess ||
        hipMalloc(&v_pool, pool_bytes) != hipSuccess ||
        hipMalloc(&partials, q35::kDecodeAttnPartialBytes) != hipSuccess ||
        hipMalloc(&row_slots, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&rope_pos, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&inv_freq, 64u * sizeof(double)) != hipSuccess ||
        hipMalloc(&error_word, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables,
                  static_cast<size_t>(kMaxSequences) * block_table_stride *
                      sizeof(uint32_t)) != hipSuccess) {
        std::printf("FAIL: state alloc\n");
        return 1;
    }
    {
        const uint32_t slot = 0u;
        std::vector<uint32_t> bt(static_cast<size_t>(kMaxSequences) *
                                 block_table_stride);
        for (size_t i = 0; i < bt.size(); ++i)
            bt[i] = static_cast<uint32_t>(i) % num_pages;
        if (hipMemcpy(row_slots, &slot, sizeof(slot), hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(row_pos, &kPosition, sizeof(uint32_t),
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(rope_pos, &kPosition, sizeof(uint32_t),
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(error_word, &slot, sizeof(slot), hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(block_tables, bt.data(), bt.size() * sizeof(uint32_t),
                      hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: state upload\n");
            return 1;
        }
        if (kernel::launch_rope_inv_freq_init(static_cast<double*>(inv_freq), 32u,
                                              10000000.0f, stream) != hipSuccess ||
            hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: rope inv freq init\n");
            return 1;
        }
    }

    const auto gdn_layout =
        psq::GdnStatePoolLayout::from_text_config(model.text_config());
    auto gdn_pool_res = psq::GdnStatePool::create(arena, 1u, gdn_layout);
    if (!gdn_pool_res.ok()) {
        std::printf("FAIL: gdn state pool: %s\n",
                    gdn_pool_res.status().message().c_str());
        return 1;
    }
    psq::GdnStatePool gdn_pool = gdn_pool_res.release();
    const psq::GdnStatePoolDeviceView gdn_view = gdn_pool.device_view();
    const size_t conv_bytes = static_cast<size_t>(gdn_pool.max_sequences()) *
                              gdn_view.conv_slot_stride * sizeof(uint16_t);
    const size_t recurrent_bytes =
        static_cast<size_t>(gdn_pool.max_sequences()) *
        gdn_view.recurrent_slot_stride * sizeof(float);
    std::printf("state: gdn_states=%u conv_bytes=%zu recurrent_bytes=%zu "
                "kv_pool_bytes=%zu\n",
                gdn_layout.num_gdn_states, conv_bytes, recurrent_bytes,
                pool_bytes);

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
        if (hipMemcpy(
                batch_storage.requests
                    .data<::ps::runtime::DeviceRequestDescriptor>(),
                &req, sizeof(req), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: batch request upload\n");
            return 1;
        }
        ::ps::runtime::DeviceBatchContext bc;
        std::memset(&bc, 0, sizeof(bc));
        bc.actual_rows = 1u;
        bc.num_requests = 1u;
        bc.requests =
            batch_storage.requests.data<::ps::runtime::DeviceRequestDescriptor>();
        if (hipMemcpy(batch_storage.device_struct
                          .data<::ps::runtime::DeviceBatchContext>(),
                      &bc, sizeof(bc), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: batch context upload\n");
            return 1;
        }
    }

    q35::HostExecutionContext ctx;
    ctx.batch_context =
        batch_storage.device_struct.data<::ps::runtime::DeviceBatchContext>();
    ctx.device_requests =
        batch_storage.requests.data<::ps::runtime::DeviceRequestDescriptor>();
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
    ctx.error_word = static_cast<uint32_t*>(error_word);
    ctx.row_positions = static_cast<const uint32_t*>(row_pos);
    ctx.rope_positions = static_cast<const uint32_t*>(rope_pos);
    ctx.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx.rope_inv_freq = static_cast<const double*>(inv_freq);
    ctx.max_visible_tokens = max_visible;
    ctx.decode_attn_partials = static_cast<uint8_t*>(partials);
    ctx.decode_attn_partial_bytes =
        static_cast<uint32_t>(q35::kDecodeAttnPartialBytes);
    ctx.model_state.gdn = gdn_view;
    ctx.model_state.block_table = psq::SequenceBlockTableDeviceView{
        .block_tables = static_cast<uint32_t*>(block_tables),
        .max_sequences = kMaxSequences,
        .block_table_stride = block_table_stride};
    ctx.model_state.kv_bf16 = psq::PagedKVPoolBF16View{
        .k_pool = static_cast<ps::bf16_t*>(k_pool),
        .v_pool = static_cast<ps::bf16_t*>(v_pool),
        .page_tokens = kPageTokens,
        .num_pages = num_pages,
        .num_attention_layers = attention_layers,
        .kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .elems_per_token = kElemsPerToken,
        .elems_per_page = kElemsPerPage,
        .elems_per_layer = num_pages * kElemsPerPage};
    ctx.model_state.kv_dtype = static_cast<uint32_t>(psq::KVCacheDType::BF16);

    std::vector<Checkpoint> checkpoints;
    for (uint32_t layer = 0; layer < ltc.size(); ++layer) {
        const std::string prefix = "L" + std::to_string(layer) + ".";
        uint32_t lb = 0u;
        uint32_t le = 0u;
        const ps::Status st = q35::find_dispatch_range_for_graph_prefix(
            lowered.graph, program, prefix.c_str(), &lb, &le);
        if (!st.ok()) {
            std::printf("FAIL: %s range: %s\n", prefix.c_str(),
                        st.message().c_str());
            return 1;
        }
        const uint32_t last = le - 1u;
        const auto v = q35::resolve_host_value(
            program, ::ps::runtime::ValueId{program.dispatches[last].output_slots[0]},
            ctx);
        if (!v.ok() || v.value().ptr == nullptr) {
            std::printf("FAIL: %s output resolve\n", prefix.c_str());
            return 1;
        }
        Checkpoint cp;
        cp.layer = layer;
        cp.dispatch = last;
        cp.ptr = v.value().ptr;
        cp.bytes = static_cast<size_t>(v.value().feature_count) *
                   dtype_bytes(v.value().dtype);
        checkpoints.push_back(std::move(cp));
    }
    std::printf("layer checkpoints=%zu\n", checkpoints.size());

    q35::McuPlanCompileOptions opts;
    opts.dispatch_begin = plan_begin;
    opts.dispatch_end = end;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const q35::McuStaticPlanCompileContext ctx_static =
        q35::make_static_plan_compile_context(
            ctx, ctx.actual_rows, ctx.actual_outputs,
            ctx.staged_request_count,
            ctx.actual_stochastic_outputs);
    const ps::Status plan_status = q35::compile_mcu_plan(program,ctx_static, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: body plan: %s\n", plan_status.message().c_str());
        return 1;
    }
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
        "plan: nodes=%zu variants=%zu logical=%u physical=%u kernarg_slots=%u "
        "invocations=%zu\n",
        plan.nodes.size(), plan.variants.size(), plan.logical_dispatch_count,
        plan.physical_dispatch_count, plan.kernarg_slots_required, inv_total);
    std::printf(
        "plan: rms=%zu e4m3=%zu elementwise=%zu rope=%zu kv_append=%zu "
        "paged=%zu paged_split=%zu paged_reduce=%zu bf16=%zu bf16_wmma=%zu "
        "l2=%zu conv1d=%zu recurrence=%zu psq4=%zu psq4_multi=%zu "
        "embed=%zu embed_psq8=%zu gather=%zu argmax=%zu\n",
        plan.rms.size(), plan.quant_e4m3.size(), plan.elementwise.size(),
        plan.rope.size(), plan.kv_append.size(), plan.attention_paged.size(),
        plan.attention_paged_split.size(), plan.attention_paged_reduce.size(),
        plan.bf16.size(), plan.bf16_wmma.size(), plan.l2.size(),
        plan.gdn_conv1d.size(), plan.gdn_recurrence.size(), plan.psq4.size(),
        plan.psq4_multi.size(), plan.embedding.size(),
        plan.embedding_psq8.size(), plan.output_gather.size(),
        plan.argmax_f32.size());
    const size_t plan_node_bytes =
        plan.nodes.size() * sizeof(mcu::McuPlanNode);
    const size_t invocation_bytes =
        plan.rms.size() * sizeof(mcu::McuRmsNormInvocation) +
        plan.quant_e4m3.size() *
            sizeof(mcu::McuActivationQuantizeE4m3Invocation) +
        plan.elementwise.size() * sizeof(mcu::McuElementwiseInvocation) +
        plan.rope.size() * sizeof(mcu::McuRopeInvocation) +
        plan.kv_append.size() * sizeof(mcu::McuKvAppendInvocation) +
        plan.attention_paged.size() *
            sizeof(mcu::McuPagedAttentionInvocation) +
        plan.attention_paged_split.size() *
            sizeof(mcu::McuPagedAttentionSplitInvocation) +
        plan.attention_paged_reduce.size() *
            sizeof(mcu::McuPagedAttentionReduceInvocation) +
        plan.bf16.size() * sizeof(mcu::McuBf16ExactRowsInvocation) +
        plan.l2.size() * sizeof(mcu::McuL2NormalizeInvocation) +
        plan.gdn_conv1d.size() * sizeof(mcu::McuGdnConv1dInvocation) +
        plan.gdn_recurrence.size() * sizeof(mcu::McuGdnRecurrenceInvocation) +
        plan.psq4.size() * sizeof(mcu::McuPsq4Decode1Invocation) +
        plan.psq4_multi.size() * sizeof(mcu::McuPsq4MultiRowInvocation) +
        plan.bf16_wmma.size() * sizeof(mcu::McuBf16WmmaInvocation) +
        plan.embedding.size() * sizeof(mcu::McuEmbeddingBf16Invocation) +
        plan.embedding_psq8.size() * sizeof(mcu::McuEmbeddingPsq8Invocation);
    std::printf(
        "plan memory: node_bytes=%zu (%zu x %zu) invocation_bytes=%zu "
        "variant_bytes=%zu\n",
        plan_node_bytes, plan.nodes.size(), sizeof(mcu::McuPlanNode),
        invocation_bytes,
        plan.variants.size() * sizeof(mcu::McuKernelVariantDesc));
    check(plan.logical_dispatch_count == dispatch_count,
          "logical dispatch count matches the range");
    check(plan.physical_dispatch_count >= dispatch_count,
          "physical dispatch count covers every logical dispatch");
    check(plan.physical_dispatch_count ==
              dispatch_count + plan.attention_paged_reduce.size() +
                  plan.gdn_reset.size(),
          "the body plan adds one physical node per reduce and one gdn reset");
    check(plan.gdn_reset.size() == 1u, "the body plan carries one gdn reset");
    const bool embed_psq8 =
        model.weights().embed_tokens.encoding != ps::weights::MatrixEncoding::Bf16;
    const uint16_t embed_recipe =
        embed_psq8 ? mcu::kMcuKernargRecipeEmbeddingPsq8
                   : mcu::kMcuKernargRecipeEmbeddingBf16;
    check(plan.nodes[0].kernarg_recipe == embed_recipe,
          "the embedding is the first node of the region plan");
    if (plan.gdn_reset.size() == 1u) {
        const mcu::GpuMcuGdnResetInvocation& reset = plan.gdn_reset[0];
        check(reset.conv_base ==
                      reinterpret_cast<const uint16_t*>(gdn_view.conv_base) &&
                  reset.recurrent_base ==
                      reinterpret_cast<const uint32_t*>(gdn_view.recurrent_base) &&
                  reset.conv_slot_stride == gdn_view.conv_slot_stride &&
                  reset.recurrent_slot_stride ==
                      gdn_view.recurrent_slot_stride &&
                  reset.conv_slot_count ==
                      static_cast<uint32_t>(gdn_view.conv_slot_stride) &&
                  reset.recurrent_slot_count ==
                      static_cast<uint32_t>(gdn_view.recurrent_slot_stride),
              "the gdn reset covers the pool slot of the view");
        check(reset.context == ctx.batch_context,
              "the gdn reset reads the batch context of this plan");
    }
    const size_t embedding_invocations =
        embed_psq8 ? plan.embedding_psq8.size() : plan.embedding.size();
    check(embedding_invocations == 1u,
          "the plan carries one embedding invocation");
    if (embed_psq8 && plan.embedding_psq8.size() == 1u) {
        const mcu::McuEmbeddingPsq8Invocation& e = plan.embedding_psq8[0];
        check(e.rows == 1u &&
                  e.hidden_size == static_cast<std::uint32_t>(
                                       model.text_config().hidden_size),
              "embedding invocation geometry");
        check(e.codes != 0u && e.scales != 0u && e.token_ids != 0u &&
                  e.output != 0u,
              "embedding invocation pointers");
        bool has_embedding_variant = false;
        for (const auto& v : plan.variants) {
            if (v.kind == q35::McuCompiledVariantKind::EmbeddingPsq8)
                has_embedding_variant = true;
        }
        check(has_embedding_variant, "the plan carries an embedding variant");
    } else if (!embed_psq8 && plan.embedding.size() == 1u) {
        const mcu::McuEmbeddingBf16Invocation& e = plan.embedding[0];
        check(e.rows == 1u &&
                  e.hidden_size == static_cast<std::uint32_t>(
                                       model.text_config().hidden_size),
              "embedding invocation geometry");
        check(e.table != 0u && e.token_ids != 0u && e.output != 0u,
              "embedding invocation pointers");
        bool has_embedding_variant = false;
        for (const auto& v : plan.variants) {
            if (v.kind == q35::McuCompiledVariantKind::EmbeddingBf16)
                has_embedding_variant = true;
        }
        check(has_embedding_variant, "the plan carries an embedding variant");
    }
    check(inv_total + plan.gdn_reset.size() ==
              static_cast<size_t>(plan.physical_dispatch_count),
          "one invocation per physical dispatch");
    check(plan.variants.size() <= mcu::kMcuMaxVariants,
          "variants within kMcuMaxVariants");
    check(plan.nodes.size() <= mcu::kMcuMaxNodes, "nodes within kMcuMaxNodes");
    check(plan.kernarg_slots_required <= mcu::kMcuMaxKernargSlots,
          "kernarg slots within kMcuMaxKernargSlots");
    check(plan.kernarg_slots_required >= plan.physical_dispatch_count + 1u,
          "kernarg slots cover every physical dispatch plus the marker");

    const auto body_input = q35::resolve_host_value(
        program, ::ps::runtime::ValueId{program.dispatches[begin].input_slots[0]},
        ctx);
    if (!body_input.ok() || body_input.value().ptr == nullptr) {
        std::printf("FAIL: body input resolve: %s\n",
                    body_input.status().message().c_str());
        return 1;
    }
    const size_t input_bytes = static_cast<size_t>(body_input.value().feature_count) *
                               dtype_bytes(body_input.value().dtype);
    std::printf("body input F=%u dtype=%d bytes=%zu\n",
                body_input.value().feature_count,
                static_cast<int>(body_input.value().dtype), input_bytes);

    std::vector<uint8_t> h_input(input_bytes, 0u);
    for (size_t i = 0; i + 1 < h_input.size(); i += 2u) {
        const uint16_t v = static_cast<uint16_t>(
            0x3A00u + ((static_cast<unsigned>(i) * 2654435761u) % 1400u));
        std::memcpy(&h_input[i], &v, sizeof(v));
    }
    std::vector<uint8_t> h_kv(2u * pool_bytes, 0u);
    for (size_t i = 0; i + 1 < h_kv.size(); i += 2u) {
        const uint16_t v = static_cast<uint16_t>(
            0x3800u + ((static_cast<unsigned>(i) * 40503u) % 1200u));
        std::memcpy(&h_kv[i], &v, sizeof(v));
    }

    auto fill_input = [&]() -> bool {
        if (hipMemcpyAsync(body_input.value().ptr, h_input.data(), input_bytes,
                           hipMemcpyHostToDevice, stream) != hipSuccess)
            return false;
        return true;
    };
    auto reset_state = [&]() -> bool {
        if (!gdn_pool.reset(0u, stream).ok()) return false;
        if (hipMemcpyAsync(k_pool, h_kv.data(), pool_bytes,
                           hipMemcpyHostToDevice, stream) != hipSuccess)
            return false;
        if (hipMemcpyAsync(v_pool, h_kv.data() + pool_bytes, pool_bytes,
                           hipMemcpyHostToDevice, stream) != hipSuccess)
            return false;
        if (!fill_input()) return false;
        if (hipMemsetAsync(error_word, 0, sizeof(uint32_t), stream) != hipSuccess)
            return false;
        return hipStreamSynchronize(stream) == hipSuccess;
    };
    if (!gdn_pool.initialize(0u, stream).ok()) {
        std::printf("FAIL: gdn state initialize\n");
        return 1;
    }

    std::vector<uint8_t> host_conv(conv_bytes, 0u);
    std::vector<uint8_t> host_recurrent(recurrent_bytes, 0u);
    std::vector<uint8_t> host_kv(2u * pool_bytes, 0u);
    uint32_t host_fallbacks = 0u;

    hipEvent_t host_ev0 = nullptr;
    hipEvent_t host_ev1 = nullptr;
    (void)hipEventCreate(&host_ev0);
    (void)hipEventCreate(&host_ev1);
    if (!reset_state()) {
        std::printf("FAIL: reset state\n");
        return 1;
    }
    (void)hipEventRecord(host_ev0, stream);
    for (uint32_t i = plan_begin; i < end; ++i) {
        auto r = q35::try_launch_optimized(program, i, ctx, stream);
        if (!r.ok()) {
            std::printf("FAIL: host dispatch %u: %s\n", i,
                        r.status().message().c_str());
            return 1;
        }
        if (r.value().kind != q35::OptimizedLaunchKind::Launched)
            ++host_fallbacks;
    }
    (void)hipEventRecord(host_ev1, stream);
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: host sync\n");
        return 1;
    }
    float host_gpu_ms = 0.0f;
    (void)hipEventElapsedTime(&host_gpu_ms, host_ev0, host_ev1);
    (void)hipEventDestroy(host_ev0);
    (void)hipEventDestroy(host_ev1);
    for (auto& cp : checkpoints) {
        cp.host.assign(cp.bytes, 0u);
        (void)hipMemcpy(cp.host.data(), cp.ptr, cp.bytes, hipMemcpyDeviceToHost);
    }
    (void)hipMemcpy(host_conv.data(), gdn_view.conv_base, conv_bytes,
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_recurrent.data(), gdn_view.recurrent_base,
                    recurrent_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_kv.data(), k_pool, pool_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(host_kv.data() + pool_bytes, v_pool, pool_bytes,
                    hipMemcpyDeviceToHost);
    std::printf("host body: launched %u dispatches, fallbacks=%u, gpu=%.3f ms\n",
                dispatch_count, host_fallbacks, static_cast<double>(host_gpu_ms));
    check(host_fallbacks == 0u, "host body ran with no correctness fallback");
    const uint32_t host_embedding_dispatches =
        q35::embedding_optimized_count();
    check(host_embedding_dispatches >= 1u,
          "host body launched the optimized embedding");

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
            if (plan.variants[i].kind ==
                q35::McuCompiledVariantKind::CompletionMarker)
                continue;
            SymbolSpec spec{};
            if (!symbol_for_kind(plan.variants[i].kind, &spec)) {
                std::printf("FAIL: no symbol for kind=%d\n",
                            static_cast<int>(plan.variants[i].kind));
                return 1;
            }
            auto co_res =
                mcu::GpuMcuAqlCodeObject::load_memory(hqueue, spec.hsaco,
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
    std::printf("needed_segment=%u\n", needed_segment);

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
            variants[i] =
                mcu::make_probe_variant(static_cast<uint16_t>(i), rig.meta,
                                        v.workgroup, v.grid,
                                        mcu::kMcuSystemAqlPolicy);
        } else {
            variants[i] = mcu::make_kernel_variant(
                static_cast<uint16_t>(i), metas[i], recipe_for(v.kind),
                v.workgroup, v.grid, mcu::kMcuAgentAqlPolicy,
                mcu::AqlHiddenArgsPolicy::IfFits, v.grid_y, v.grid_z);
        }
    }
    if (!rig.upload(plan.nodes, variants)) return 1;
    if (!rig.set_invocations(plan.rms.data(),
                             static_cast<uint32_t>(plan.rms.size())))
        return 1;
    if (!rig.set_e4m3_invocations(plan.quant_e4m3.data(),
                                  static_cast<uint32_t>(plan.quant_e4m3.size())))
        return 1;
    if (!rig.set_psq4_invocations(plan.psq4.data(),
                                  static_cast<uint32_t>(plan.psq4.size())))
        return 1;
    if (!rig.set_psq4_multi_invocations(
            plan.psq4_multi.data(),
            static_cast<uint32_t>(plan.psq4_multi.size())))
        return 1;
    if (!rig.set_elementwise_invocations(
            plan.elementwise.data(),
            static_cast<uint32_t>(plan.elementwise.size())))
        return 1;
    if (!rig.set_rope_invocations(plan.rope.data(),
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
    if (!rig.set_bf16_invocations(plan.bf16.data(),
                                  static_cast<uint32_t>(plan.bf16.size())))
        return 1;
    if (!rig.set_bf16_wmma_invocations(
            plan.bf16_wmma.data(),
            static_cast<uint32_t>(plan.bf16_wmma.size())))
        return 1;
    if (!rig.set_l2_invocations(plan.l2.data(),
                                static_cast<uint32_t>(plan.l2.size())))
        return 1;
    if (!rig.set_embedding_invocations(
            plan.embedding.data(),
            static_cast<uint32_t>(plan.embedding.size())))
        return 1;
    if (!rig.set_embedding_psq8_invocations(
            plan.embedding_psq8.data(),
            static_cast<uint32_t>(plan.embedding_psq8.size())))
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

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x424f4459u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = plan.kernarg_slots_required;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }

    int full_iters = 100;
    int endurance_iters = 1000;
    if (const char* e = std::getenv("PHASESHIFT_BODY_ITERS")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v > 0) full_iters = static_cast<int>(v);
    }
    if (const char* e = std::getenv("PHASESHIFT_BODY_ENDURANCE")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v >= 0) endurance_iters = static_cast<int>(v);
    }

    std::vector<uint8_t> mcu_conv(conv_bytes, 0u);
    std::vector<uint8_t> mcu_recurrent(recurrent_bytes, 0u);
    std::vector<uint8_t> mcu_kv(2u * pool_bytes, 0u);
    uint32_t mismatch_iters = 0u;

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < full_iters; ++it) {
        if (!reset_state()) {
            std::printf("FAIL: reset state (mcu)\n");
            return 1;
        }
        if (!rig.fsm.request_run().ok()) {
            std::printf("FAIL: mcu plan requested\n");
            return 1;
        }
        if (!rig.fsm.wait_plans(static_cast<uint64_t>(it) + 1u, 120000).ok()) {
            std::printf("FAIL: mcu plan completed at iter %d\n", it);
            return 1;
        }
        if (!rig.drain_queue(120000)) {
            std::printf("FAIL: queue drained at iter %d\n", it);
            return 1;
        }
        if (rig.fsm.fault_code() != mcu::McuFaultCode::none) {
            std::printf("FAIL: fault at iter %d code=%u\n", it,
                        static_cast<unsigned>(rig.fsm.fault_code()));
            return 1;
        }
        int iter_failed = 0;
        for (auto& cp : checkpoints) {
            cp.mcu.assign(cp.bytes, 0u);
            rig.device_to_host(cp.mcu.data(), cp.ptr, cp.bytes);
            const std::string label =
                "L" + std::to_string(cp.layer) + ".output";
            compare_or_report(label.c_str(), cp.host, cp.mcu, &iter_failed);
        }
        rig.device_to_host(mcu_kv.data(), k_pool, pool_bytes);
        rig.device_to_host(mcu_kv.data() + pool_bytes, v_pool, pool_bytes);
        compare_or_report("kv state", host_kv, mcu_kv, &iter_failed);
        rig.device_to_host(mcu_conv.data(), gdn_view.conv_base, conv_bytes);
        compare_or_report("gdn conv state", host_conv, mcu_conv, &iter_failed);
        rig.device_to_host(mcu_recurrent.data(), gdn_view.recurrent_base,
                           recurrent_bytes);
        compare_or_report("gdn recurrent state", host_recurrent, mcu_recurrent,
                          &iter_failed);
        if (iter_failed != 0) {
            std::printf("FAIL: iteration %d had %d mismatching checkpoints\n", it,
                        iter_failed);
            ++g_failed;
            break;
        }
        ++mismatch_iters;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double full_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    check(q35::embedding_optimized_count() == host_embedding_dispatches,
          "mcu regions issued no host embedding dispatch");

    uint32_t endurance_faults = 0u;
    const uint64_t before_endurance = rig.fsm.dispatches();
    const auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < endurance_iters; ++it) {
        if (!reset_state()) {
            std::printf("FAIL: reset state (endurance)\n");
            return 1;
        }
        if (!rig.fsm.request_run().ok()) {
            std::printf("FAIL: endurance request\n");
            return 1;
        }
        if (!rig.fsm
                 .wait_plans(static_cast<uint64_t>(full_iters + it) + 1u, 120000)
                 .ok()) {
            std::printf("FAIL: endurance plan completed at iter %d\n", it);
            return 1;
        }
        if (!rig.drain_queue(120000)) {
            std::printf("FAIL: endurance drain at iter %d\n", it);
            return 1;
        }
        if (rig.fsm.fault_code() != mcu::McuFaultCode::none) {
            ++endurance_faults;
            std::printf("FAIL: endurance fault at iter %d code=%u\n", it,
                        static_cast<unsigned>(rig.fsm.fault_code()));
            break;
        }
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double endurance_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    const uint64_t regions =
        static_cast<uint64_t>(full_iters) + static_cast<uint64_t>(endurance_iters);
    check(rig.fsm.dispatches() ==
              (static_cast<uint64_t>(plan.physical_dispatch_count) + 1u) * regions,
          "mcu dispatch count = (physical + marker) * regions");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(endurance_faults == 0u, "no endurance fault");
    (void)before_endurance;

    const uint64_t region_ticks =
        rig.fsm.region_end_ts() - rig.fsm.region_start_ts();
    const double region_ms =
        mcu::gpu_wall_clock_ticks_to_ms(region_ticks, wall_clock_rate_khz);
    std::printf(
        "timing: host_hip_body=%.3f ms mcu_gpu_region=%.3f ms "
        "mcu_host_observed=%.3f ms/region (endurance, includes state reset) "
        "dispatches_per_region=%u\n",
        static_cast<double>(host_gpu_ms), region_ms,
        endurance_ms /
            static_cast<double>(endurance_iters > 0 ? endurance_iters : 1),
        plan.physical_dispatch_count);
    std::printf(
        "mcu: regions=%llu dispatches=%llu fault=%u full_iters=%d "
        "full_ms_per_region=%.3f endurance_iters=%d endurance_ms_per_region=%.3f "
        "gpu_region=%.3f ms\n",
        static_cast<unsigned long long>(regions),
        static_cast<unsigned long long>(rig.fsm.dispatches()),
        static_cast<unsigned>(rig.fsm.fault_code()), full_iters,
        full_ms / static_cast<double>(full_iters > 0 ? full_iters : 1),
        endurance_iters,
        endurance_ms /
            static_cast<double>(endurance_iters > 0 ? endurance_iters : 1),
        region_ms);
    std::printf(
        "queue: depth=4 doorbell=per_packet doorbell_count=%llu refill=%llu "
        "empty=%llu mid_region_empty=%llu backpressure=%llu max_ahead=%llu "
        "wrap=%llu\n",
        static_cast<unsigned long long>(rig.fsm.doorbell_count()),
        static_cast<unsigned long long>(rig.fsm.refill_count()),
        static_cast<unsigned long long>(rig.fsm.queue_empty_count()),
        static_cast<unsigned long long>(rig.fsm.mid_region_queue_empty_count()),
        static_cast<unsigned long long>(rig.fsm.backpressure_wait_count()),
        static_cast<unsigned long long>(rig.fsm.max_ahead()),
        static_cast<unsigned long long>(rig.fsm.wrap_count()));
    std::printf(
        "note: mid_region_queue_empty=%llu regions=%llu "
        "(region starvation metric, bounded target <= regions; "
        "docs/rnd/gpu_mcu/full_transformer_static_region.md)\n",
        static_cast<unsigned long long>(rig.fsm.mid_region_queue_empty_count()),
        static_cast<unsigned long long>(regions));
    std::printf(
        "correctness: iterations_ok=%d/%d (layer outputs + kv state + gdn state "
        "compared on every iteration)\n",
        mismatch_iters, full_iters);

    check(rig.close(), "rig teardown");

    const ps::Status bc_destroy =
        ps::runtime::destroy_device_batch_context(batch_storage);
    check(bc_destroy.ok(), "device batch context destroyed");
    (void)hipFree(error_word);
    (void)hipFree(inv_freq);    (void)hipFree(rope_pos);
    (void)hipFree(row_pos);
    (void)hipFree(row_slots);
    (void)hipFree(block_tables);
    (void)hipFree(partials);
    (void)hipFree(v_pool);
    (void)hipFree(k_pool);
    for (void* p : external_alloc) (void)hipFree(p);
    if (workspace) (void)hipFree(workspace);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_full_transformer_body_plan: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_full_transformer_body_plan: PASS\n");
    return 0;
}
