#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/models/qwen35/kernels/optimized/l2_normalize.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/mcu_layer_range.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>

#include "support/resident_model_fixture.h"

#include <chrono>
#include <cmath>
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

struct SymbolSpec {
    const char* hsaco;
    const char* symbol;
};

bool symbol_for_kind(q35::McuCompiledVariantKind kind, SymbolSpec* out) {
    switch (kind) {
        case q35::McuCompiledVariantKind::RmsNormBf16PfOnePlus:
            *out = {PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                    kernel::kRmsNormBf16PfOnePlusSymbol};
            return true;
        case q35::McuCompiledVariantKind::RmsNormF32PgDirect:
            *out = {PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                    kernel::kRmsNormF32PgDirectSymbol};
            return true;
        case q35::McuCompiledVariantKind::RmsNormBf16F32PgOnePlus:
            *out = {PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                    kernel::kRmsNormBf16F32PgOnePlusSymbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseResidualBf16:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseResidualAddBf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseSwigluBf16:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseSwigluBf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseMulF32F32Bf16:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseMulF32F32Bf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseScaleF32:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseScaleF32Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseSiluBf16ToF32:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseSiluBf16ToF32Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseSiluF32ToBf16:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseSiluF32ToBf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseSigmoidBf16ToF32:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseSigmoidBf16ToF32Symbol};
            return true;
        case q35::McuCompiledVariantKind::ElementwiseSplitBf16:
            *out = {PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                    kernel::kElementwiseSplitBf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::Bf16ExactRows:
            *out = {PHASESHIFT_GPU_MCU_BF16_HSACO, kernel::kBf16ExactRowsSymbol};
            return true;
        case q35::McuCompiledVariantKind::L2NormalizeBf16F32:
            *out = {PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO,
                    kernel::kL2NormalizeBf16F32Symbol};
            return true;
        case q35::McuCompiledVariantKind::RopeF32Bf16Pair:
            *out = {PHASESHIFT_GPU_MCU_ROPE_HSACO,
                    kernel::kRopeF32Bf16PairSymbol};
            return true;
        case q35::McuCompiledVariantKind::KvAppendBf16:
            *out = {PHASESHIFT_GPU_MCU_KV_APPEND_HSACO,
                    kernel::kKvAppendBf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::AttentionPagedBf16:
            *out = {PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO,
                    kernel::kPagedAttentionBf16Symbol};
            return true;
        case q35::McuCompiledVariantKind::AttentionPagedPrefillBf16:
            *out = {PHASESHIFT_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO,
                    kernel::kPagedAttentionPrefillBf16Symbol};
            return true;
        default:
            return false;
    }
}

uint16_t recipe_for(q35::McuCompiledVariantKind kind) {
    switch (kind) {
        case q35::McuCompiledVariantKind::RmsNormBf16PfOnePlus:
        case q35::McuCompiledVariantKind::RmsNormF32PgDirect:
        case q35::McuCompiledVariantKind::RmsNormBf16F32PgOnePlus:
            return mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
        case q35::McuCompiledVariantKind::ElementwiseResidualBf16:
        case q35::McuCompiledVariantKind::ElementwiseSwigluBf16:
        case q35::McuCompiledVariantKind::ElementwiseMulF32F32Bf16:
        case q35::McuCompiledVariantKind::ElementwiseScaleF32:
        case q35::McuCompiledVariantKind::ElementwiseSiluBf16ToF32:
        case q35::McuCompiledVariantKind::ElementwiseSiluF32ToBf16:
        case q35::McuCompiledVariantKind::ElementwiseSigmoidBf16ToF32:
        case q35::McuCompiledVariantKind::ElementwiseSplitBf16:
            return mcu::kMcuKernargRecipeElementwise;
        case q35::McuCompiledVariantKind::Bf16ExactRows:
            return mcu::kMcuKernargRecipeBf16ExactRows;
        case q35::McuCompiledVariantKind::L2NormalizeBf16F32:
            return mcu::kMcuKernargRecipeL2Normalize;
        case q35::McuCompiledVariantKind::RopeF32Bf16Pair:
            return mcu::kMcuKernargRecipeRope;
        case q35::McuCompiledVariantKind::KvAppendBf16:
            return mcu::kMcuKernargRecipeKvAppend;
        case q35::McuCompiledVariantKind::AttentionPagedBf16:
            return mcu::kMcuKernargRecipeAttentionPaged;
        case q35::McuCompiledVariantKind::AttentionPagedPrefillBf16:
            return mcu::kMcuKernargRecipeAttentionPaged;
        default:
            return mcu::kMcuKernargRecipeProbe;
    }
}

constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kBlockTableStride = 32u;
constexpr uint32_t kMaxSequences = 2u;
constexpr uint32_t kPosition = 100u;
constexpr uint32_t kMaxVisible = 512u;
constexpr uint32_t kAttnLayers = 8u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;

struct TraceEntry {
    uint32_t dispatch = 0u;
    std::string name;
    void* ptr = nullptr;
    size_t bytes = 0u;
    std::vector<uint8_t> host;
    std::vector<uint8_t> mcu;
};

void run_prefill_scenario() {
    constexpr uint32_t kRows = 256u;
    constexpr uint32_t kQHeads = 16u;
    constexpr uint32_t kKvHeads = 4u;
    constexpr uint32_t kPfHeadDim = 256u;
    constexpr uint32_t kPfPageTokens = 16u;
    constexpr uint32_t kPfNumPages = 32u;
    constexpr uint32_t kPfLayers = 8u;
    constexpr uint32_t kPfBtStride = 32u;
    constexpr uint32_t kPfMaxSeq = 2u;
    constexpr uint32_t kPfMaxVisible = 512u;
    constexpr uint32_t kQFeatures = kQHeads * kPfHeadDim;
    constexpr uint32_t kElemsPerToken = kKvHeads * kPfHeadDim;
    constexpr uint32_t kElemsPerPage = kPfPageTokens * kElemsPerToken;
    constexpr uint32_t kElemsPerLayer = kPfNumPages * kElemsPerPage;
    constexpr size_t kQBytes = static_cast<size_t>(kRows) * kQFeatures * sizeof(uint16_t);
    constexpr size_t kOutBytes = static_cast<size_t>(kRows) * kQFeatures * sizeof(float);
    constexpr size_t kPoolBytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);

    std::printf("prefill scenario: rows=%u q_heads=%u kv_heads=%u head_dim=%u "
                "page_tokens=%u\n",
                kRows, kQHeads, kKvHeads, kPfHeadDim, kPfPageTokens);

    void* q_dev = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* out_host = nullptr;
    void* out_mcu = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    if (hipMalloc(&q_dev, kQBytes) != hipSuccess ||
        hipMalloc(&k_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&out_host, kOutBytes) != hipSuccess ||
        hipMalloc(&out_mcu, kOutBytes) != hipSuccess ||
        hipMalloc(&row_slots, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables, kPfMaxSeq * kPfBtStride * sizeof(uint32_t)) !=
            hipSuccess) {
        check(false, "prefill buffers allocated");
        return;
    }

    std::vector<uint16_t> h_q(static_cast<size_t>(kRows) * kQFeatures);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
    std::vector<uint16_t> h_pool(kElemsPerLayer);
    for (size_t i = 0; i < h_pool.size(); ++i)
        h_pool[i] = static_cast<uint16_t>(0x3900u + ((i * 40503u) % 1700u));
    std::vector<uint32_t> h_slots(kRows, 0u);
    std::vector<uint32_t> h_pos(kRows);
    for (uint32_t r = 0; r < kRows; ++r) h_pos[r] = r;
    std::vector<uint32_t> h_bt(kPfMaxSeq * kPfBtStride);
    for (uint32_t i = 0; i < h_bt.size(); ++i) h_bt[i] = i % kPfNumPages;

    (void)hipMemcpy(q_dev, h_q.data(), kQBytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(k_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
    for (size_t i = 0; i < h_pool.size(); ++i)
        h_pool[i] = static_cast<uint16_t>(0x3B00u + ((i * 2654435761u) % 1500u));
    (void)hipMemcpy(v_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(row_slots, h_slots.data(), h_slots.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(row_pos, h_pos.data(), h_pos.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(block_tables, h_bt.data(), h_bt.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemset(out_host, 0xAB, kOutBytes);
    (void)hipMemset(out_mcu, 0xAB, kOutBytes);

    ::ps::runtime::DeviceRequestDescriptor req{};
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = kRows;
    req.prefix_length = 0u;
    req.sequence_length = kRows;
    ::ps::runtime::DeviceBatchContext bc{};
    bc.actual_rows = kRows;
    bc.num_requests = 1u;
    bc.num_prefill_requests = 1u;
    bc.requests = &req;
    bc.row_sequence_slots = static_cast<uint32_t*>(row_slots);
    bc.row_positions = static_cast<uint32_t*>(row_pos);

    ::ps::runtime::Program program{};
    program.row_bucket = ::ps::runtime::RowBucket::R256;
    program.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
    {
        ::ps::runtime::ValueBinding vq{};
        vq.logical_value = ::ps::runtime::ValueId{0u};
        vq.storage = ::ps::runtime::ValueStorage::EXTERNAL_INPUT;
        vq.dtype = ::ps::runtime::ValueDType::BF16;
        vq.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
        vq.slot = 0u;
        vq.feature_count = kQFeatures;
        vq.row_stride = kQFeatures;
        program.values.push_back(vq);

        ::ps::runtime::ValueBinding vo{};
        vo.logical_value = ::ps::runtime::ValueId{1u};
        vo.storage = ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT;
        vo.dtype = ::ps::runtime::ValueDType::F32;
        vo.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
        vo.slot = 0u;
        vo.feature_count = kQFeatures;
        vo.row_stride = kQFeatures;
        program.values.push_back(vo);

        ::ps::runtime::StateBinding st{};
        st.state = ::ps::runtime::StateId{0u};
        st.kind = ::ps::runtime::PrimitiveStateKind::KV_CACHE;
        st.state_index = 0u;
        st.slot = 0u;
        program.states.push_back(st);

        ::ps::runtime::DispatchBinding ad{};
        ad.kernel_id = ::ps::runtime::KernelId::PAGED_ATTENTION;
        ad.descriptor_slot = 1u;
        ad.input_count = 1u;
        ad.input_slots[0] = 0u;
        ad.output_count = 1u;
        ad.output_slots[0] = 1u;
        ad.group_size = kPfHeadDim;
        ad.scalar_a = 1.0f / std::sqrt(static_cast<float>(kPfHeadDim));
        program.dispatches.push_back(ad);
        program.dispatch_source_node.push_back(0u);
        program.external_input_count = 1u;
        program.external_output_count = 1u;
    }

    const void* input_ptrs[1] = {q_dev};
    void* output_ptrs[1] = {out_host};

    q35::HostExecutionContext ctx{};
    ctx.batch_context = &bc;
    ctx.external_inputs = input_ptrs;
    ctx.external_input_count = 1u;
    ctx.external_outputs = output_ptrs;
    ctx.external_output_count = 1u;
    ctx.actual_rows = kRows;
    ctx.actual_outputs = 1u;
    ctx.row_positions = static_cast<const uint32_t*>(row_pos);
    ctx.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx.max_visible_tokens = kPfMaxVisible;
    {
        ::ps::kernel::ModelDispatchStateView ms{};
        ms.block_table = ::ps::qwen35::SequenceBlockTableDeviceView{
            .block_tables = static_cast<uint32_t*>(block_tables),
            .max_sequences = kPfMaxSeq,
            .block_table_stride = kPfBtStride};
        ms.kv_bf16 = ::ps::qwen35::PagedKVPoolBF16View{
            .k_pool = static_cast<ps::bf16_t*>(k_pool),
            .v_pool = static_cast<ps::bf16_t*>(v_pool),
            .page_tokens = kPfPageTokens,
            .num_pages = kPfNumPages,
            .num_attention_layers = kPfLayers,
            .kv_heads = kKvHeads,
            .head_dim = kPfHeadDim,
            .elems_per_token = kElemsPerToken,
            .elems_per_page = kElemsPerPage,
            .elems_per_layer = kElemsPerLayer};
        ms.kv_dtype = static_cast<uint32_t>(::ps::qwen35::KVCacheDType::BF16);
        ctx.model_state = ms;
    }

    q35::McuPlanCompileOptions opts;
    opts.dispatch_begin = 0u;
    opts.dispatch_end = 1u;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const ps::Status plan_status = q35::compile_mcu_plan(program, ctx, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: prefill plan: %s\n", plan_status.message().c_str());
        check(false, "prefill plan compiled");
        return;
    }
    check(plan.logical_dispatch_count == 1u, "prefill logical dispatch count");
    check(plan.physical_dispatch_count == 1u, "prefill physical dispatch count");
    check(plan.attention_paged.size() == 1u, "prefill attention invocation");
    bool has_prefill_kind = false;
    for (const auto& v : plan.variants) {
        if (v.kind != q35::McuCompiledVariantKind::AttentionPagedPrefillBf16) continue;
        has_prefill_kind = true;
        std::printf("prefill variant: grid=%u grid_y=%u workgroup=%u\n", v.grid,
                    v.grid_y, v.workgroup);
        check(v.grid == (kRows + kernel::kPagedAttentionPrefillTileRows - 1u) /
                            kernel::kPagedAttentionPrefillTileRows,
              "prefill grid is ceil(rows/tile_rows)");
        check(v.grid_y == kQHeads, "prefill grid_y is q_heads");
        check(v.workgroup == kernel::kPagedAttentionPrefillBlock,
              "prefill workgroup is the kernel block");
    }
    check(has_prefill_kind, "prefill variant emitted");

    ::ps::runtime::DeviceRequestDescriptor mixed_reqs[2]{};
    mixed_reqs[0].execution_class = ::ps::runtime::ExecutionClass::DECODE;
    mixed_reqs[0].row_begin = 0u;
    mixed_reqs[0].row_count = 1u;
    mixed_reqs[1].execution_class = ::ps::runtime::ExecutionClass::PREFILL;
    mixed_reqs[1].row_begin = 1u;
    mixed_reqs[1].row_count = kRows - 1u;
    bc.requests = mixed_reqs;
    bc.num_requests = 2u;
    bc.num_decode_requests = 1u;
    bc.num_prefill_requests = 1u;
    q35::McuCompiledPlan mixed_plan;
    const ps::Status mixed_status =
        q35::compile_mcu_plan(program, ctx, opts, mixed_plan);
    bc.requests = &req;
    bc.num_requests = 1u;
    bc.num_decode_requests = 0u;
    if (mixed_status.ok()) {
        bool mixed_prefill = false;
        bool mixed_decode = false;
        for (const auto& v : mixed_plan.variants) {
            mixed_prefill = mixed_prefill ||
                (v.kind == q35::McuCompiledVariantKind::AttentionPagedPrefillBf16);
            mixed_decode = mixed_decode ||
                (v.kind == q35::McuCompiledVariantKind::AttentionPagedBf16);
        }
        check(mixed_prefill, "mixed batch emits the prefill region");
        check(mixed_decode, "mixed batch emits the decode region");
        check(mixed_plan.physical_dispatch_count == 2u,
              "mixed batch has two attention regions");
        std::printf(
            "mixed batch (num_requests=2): prefill=%d decode=%d physical=%u\n",
            mixed_prefill ? 1 : 0, mixed_decode ? 1 : 0,
            mixed_plan.physical_dispatch_count);
    } else {
        std::printf("mixed batch (num_requests=2) compile failed: %s\n",
                    mixed_status.message().c_str());
        check(false, "mixed batch compiles");
    }

    auto resolved = q35::resolve_paged_attention_args(program, 0u, ctx);
    if (!resolved.ok()) {
        std::printf("FAIL: prefill resolve: %s\n",
                    resolved.status().message().c_str());
        check(false, "prefill resolve");
        return;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        check(false, "prefill stream");
        return;
    }
    const auto& ra = resolved.value();
    check(kernel::launch_attention_paged_prefill_bf16(
              ra.common, ra.k_bf16_pool, ra.v_bf16_pool, stream) == hipSuccess,
          "host prefill launch");
    check(hipStreamSynchronize(stream) == hipSuccess, "host prefill sync");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    plan.attention_paged[0].output = reinterpret_cast<uint64_t>(out_mcu);

    Rig rig;
    std::vector<mcu::McuKernelVariantDesc> variants(plan.variants.size());
    std::vector<mcu::GpuAqlKernelMetadata> metas(plan.variants.size());
    uint32_t needed_segment = 0u;
    {
        auto hq_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                  mcu::kAqlDefaultQueueSize);
        if (!hq_res.ok()) {
            check(false, "prefill host queue");
            return;
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
                check(false, "prefill symbol");
                return;
            }
            auto co_res =
                mcu::GpuMcuAqlCodeObject::load(hqueue, spec.hsaco, spec.symbol);
            if (!co_res.ok()) {
                std::printf("FAIL: load %s: %s\n", spec.symbol,
                            co_res.status().message().c_str());
                check(false, "prefill code object");
                return;
            }
            auto co = co_res.release();
            metas[i] = co.metadata();
            if (metas[i].kernarg_segment_size > needed_segment)
                needed_segment = metas[i].kernarg_segment_size;
        }
        if (!hqueue.shutdown().ok()) {
            check(false, "prefill host queue shutdown");
            return;
        }
    }

    if (!rig.open(0, static_cast<uint32_t>(plan.nodes.size()) + 4u,
                  static_cast<uint32_t>(plan.variants.size()) + 2u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  needed_segment)) {
        check(false, "prefill rig open");
        return;
    }
    for (size_t i = 0; i < plan.variants.size(); ++i) {
        if (plan.variants[i].kind == q35::McuCompiledVariantKind::CompletionMarker)
            continue;
        SymbolSpec spec{};
        (void)symbol_for_kind(plan.variants[i].kind, &spec);
        if (!rig.load_extra_code(spec.hsaco, spec.symbol, metas[i])) {
            check(false, "prefill extra code");
            return;
        }
    }
    for (size_t i = 0; i < plan.variants.size(); ++i) {
        const auto& v = plan.variants[i];
        if (v.kind == q35::McuCompiledVariantKind::CompletionMarker) {
            variants[i] = mcu::make_probe_variant(static_cast<uint16_t>(i), rig.meta,
                                                  v.workgroup, v.grid,
                                                  mcu::kMcuSystemAqlPolicy);
        } else {
            variants[i] = mcu::make_kernel_variant(
                static_cast<uint16_t>(i), metas[i], recipe_for(v.kind), v.workgroup,
                v.grid, mcu::kMcuAgentAqlPolicy, mcu::AqlHiddenArgsPolicy::IfFits,
                v.grid_y, v.grid_z);
        }
    }
    if (!rig.upload(plan.nodes, variants)) {
        check(false, "prefill upload");
        return;
    }
    if (!rig.set_attention_paged_invocations(
            plan.attention_paged.data(),
            static_cast<uint32_t>(plan.attention_paged.size()))) {
        check(false, "prefill invocations");
        return;
    }
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x5052464Cu;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = plan.kernarg_slots_required;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        check(false, "prefill fsm start");
        return;
    }
    if (!rig.fsm.request_run().ok()) {
        check(false, "prefill run requested");
        (void)rig.close();
        return;
    }
    if (!rig.fsm.wait_plans(1u, 60000).ok()) {
        check(false, "prefill plan completed");
        (void)rig.close();
        return;
    }
    if (!rig.drain_queue(60000)) {
        check(false, "prefill queue drained");
        (void)rig.close();
        return;
    }
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "prefill no fault");
    check(rig.fsm.dispatches() == 2u, "prefill two dispatches (attention + marker)");

    std::vector<uint8_t> h_host(kOutBytes, 0u);
    std::vector<uint8_t> h_mcu(kOutBytes, 0u);
    rig.device_to_host(h_host.data(), out_host, kOutBytes);
    rig.device_to_host(h_mcu.data(), out_mcu, kOutBytes);
    size_t first = 0;
    size_t diffs = 0;
    const bool same =
        bytes_equal(h_host.data(), h_mcu.data(), kOutBytes, &first, &diffs);
    check(same, "prefill output host vs mcu byte exact");
    if (diffs)
        std::printf("  prefill output diff count=%zu first=%zu\n", diffs, first);
    std::printf("prefill scenario: diffs=%zu dispatches=%llu\n", diffs,
                static_cast<unsigned long long>(rig.fsm.dispatches()));

    check(rig.close(), "prefill rig teardown");
    (void)hipFree(block_tables);
    (void)hipFree(row_pos);
    (void)hipFree(row_slots);
    (void)hipFree(out_mcu);
    (void)hipFree(out_host);
    (void)hipFree(v_pool);
    (void)hipFree(k_pool);
    (void)hipFree(q_dev);
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
    if (model.weights().embed_tokens.encoding != ps::weights::MatrixEncoding::Bf16) {
        std::printf("SKIP: mcu plan compile requires BF16 embedding storage\n");
        return 77;
    }

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
    int attention_layer = -1;
    for (std::size_t i = 0; i < ltc.size(); ++i) {
        if (ltc[i] == 1) {
            attention_layer = static_cast<int>(i);
            break;
        }
    }
    if (attention_layer < 0) {
        std::printf("FAIL: no standard attention layer\n");
        return 1;
    }
    const std::string prefix = "L" + std::to_string(attention_layer) + ".";
    std::printf("attention layer=%s\n", prefix.c_str());

    void* workspace = nullptr;
    if (program.workspace.total_bytes != 0u &&
        hipMalloc(&workspace, program.workspace.total_bytes) != hipSuccess) {
        std::printf("FAIL: workspace alloc\n");
        return 1;
    }
    std::vector<const void*> external_inputs(program.external_input_count,
                                             reinterpret_cast<const void*>(0x1000));
    std::vector<void*> external_outputs(program.external_output_count,
                                        reinterpret_cast<void*>(0x2000));

    const uint32_t num_pages = kBlockTableStride;
    const size_t pool_bytes =
        static_cast<size_t>(num_pages) * kElemsPerPage * sizeof(uint16_t);

    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* partials = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* rope_pos = nullptr;
    void* inv_freq = nullptr;
    void* block_tables = nullptr;
    if (hipMalloc(&k_pool, pool_bytes) != hipSuccess ||
        hipMalloc(&v_pool, pool_bytes) != hipSuccess ||
        hipMalloc(&partials, q35::kDecodeAttnPartialBytes) != hipSuccess ||
        hipMalloc(&row_slots, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&rope_pos, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&inv_freq, 64u * sizeof(double)) != hipSuccess ||
        hipMalloc(&block_tables,
                  kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess) {
        std::printf("FAIL: attention state alloc\n");
        return 1;
    }
    {
        const uint32_t slot = 0u;
        const uint32_t pos = kPosition;
        const uint32_t rp = kPosition;
        uint32_t bt[kMaxSequences * kBlockTableStride];
        for (uint32_t i = 0; i < kMaxSequences * kBlockTableStride; ++i)
            bt[i] = i % num_pages;
        if (hipMemcpy(row_slots, &slot, sizeof(slot), hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(row_pos, &pos, sizeof(pos), hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(rope_pos, &rp, sizeof(rp), hipMemcpyHostToDevice) !=
                hipSuccess ||
            hipMemcpy(block_tables, bt, sizeof(bt), hipMemcpyHostToDevice) !=
                hipSuccess) {
            std::printf("FAIL: attention state upload\n");
            return 1;
        }
        if (kernel::launch_rope_inv_freq_init(static_cast<double*>(inv_freq), 32u,
                                              10000000.0f, stream) != hipSuccess ||
            hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: rope inv freq init\n");
            return 1;
        }
    }

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
    ctx.row_positions = static_cast<const uint32_t*>(row_pos);
    ctx.rope_positions = static_cast<const uint32_t*>(rope_pos);
    ctx.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx.rope_inv_freq = static_cast<const double*>(inv_freq);
    ctx.max_visible_tokens = kMaxVisible;
    ctx.decode_attn_partials = static_cast<uint8_t*>(partials);
    ctx.decode_attn_partial_bytes =
        static_cast<uint32_t>(q35::kDecodeAttnPartialBytes);
    {
        ::ps::kernel::ModelDispatchStateView ms{};
        ms.block_table = ::ps::qwen35::SequenceBlockTableDeviceView{
            .block_tables = static_cast<uint32_t*>(block_tables),
            .max_sequences = kMaxSequences,
            .block_table_stride = kBlockTableStride};
        ms.kv_bf16 = ::ps::qwen35::PagedKVPoolBF16View{
            .k_pool = static_cast<ps::bf16_t*>(k_pool),
            .v_pool = static_cast<ps::bf16_t*>(v_pool),
            .page_tokens = kPageTokens,
            .num_pages = num_pages,
            .num_attention_layers = kAttnLayers,
            .kv_heads = kKvHeads,
            .head_dim = kHeadDim,
            .elems_per_token = kElemsPerToken,
            .elems_per_page = kElemsPerPage,
            .elems_per_layer = num_pages * kElemsPerPage};
        ms.kv_dtype = static_cast<uint32_t>(::ps::qwen35::KVCacheDType::BF16);
        ctx.model_state = ms;
    }

    uint32_t begin = 0u;
    uint32_t end = 0u;
    const ps::Status range_status =
        q35::find_dispatch_range_for_graph_prefix(lowered.graph, program,
                                                  prefix.c_str(), &begin, &end);
    if (!range_status.ok()) {
        std::printf("FAIL: attention range: %s\n", range_status.message().c_str());
        return 1;
    }
    const uint32_t dispatch_count = end - begin;
    std::printf("%s range=[%u,%u) dispatches=%u\n", prefix.c_str(), begin, end,
                dispatch_count);

    q35::McuPlanCompileOptions opts;
    opts.dispatch_begin = begin;
    opts.dispatch_end = end;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const ps::Status plan_status = q35::compile_mcu_plan(program, ctx, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: attention plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    const size_t inv_total =
        plan.rms.size() + plan.quant_e4m3.size() + plan.elementwise.size() +
        plan.rope.size() + plan.kv_append.size() + plan.attention_paged.size() +
        plan.bf16.size() + plan.l2.size() + plan.gdn_conv1d.size() +
        plan.gdn_recurrence.size();
    std::printf(
        "plan: nodes=%zu variants=%zu kernarg_slots=%u logical=%u physical=%u "
        "invocations=%zu\n",
        plan.nodes.size(), plan.variants.size(), plan.kernarg_slots_required,
        plan.logical_dispatch_count, plan.physical_dispatch_count, inv_total);
    check(plan.logical_dispatch_count == dispatch_count,
          "logical dispatch count matches the range");
    check(plan.physical_dispatch_count == dispatch_count,
          "physical dispatch count matches (splits == 1)");
    check(inv_total == static_cast<size_t>(dispatch_count),
          "one invocation per dispatch");
    check(plan.variants.size() <= mcu::kMcuMaxVariants,
          "variants within kMcuMaxVariants");

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
    std::printf("layer F=%u dtype=%d out_bytes=%zu\n",
                layer_input.value().feature_count,
                static_cast<int>(last_out.value().dtype), out_bytes);

    // Gate A11: trace every dispatch output in the layer.
    std::vector<TraceEntry> traces;
    for (uint32_t i = begin; i < end; ++i) {
        if (program.dispatches[i].output_count == 0u) continue;
        const auto v = q35::resolve_host_value(
            program, ::ps::runtime::ValueId{program.dispatches[i].output_slots[0]},
            ctx);
        if (!v.ok() || v.value().ptr == nullptr || v.value().feature_count == 0u)
            continue;
        TraceEntry t;
        t.dispatch = i;
        const uint32_t src = i < program.dispatch_source_node.size()
                                 ? program.dispatch_source_node[i]
                                 : 0u;
        t.name = src < lowered.graph.nodes.size()
                     ? lowered.graph.nodes[src].debug_name
                     : std::string("<none>");
        t.ptr = v.value().ptr;
        t.bytes = static_cast<size_t>(v.value().feature_count) *
                  ((v.value().dtype == ::ps::runtime::ValueDType::F32) ? 4u : 2u);
        traces.push_back(std::move(t));
    }
    std::printf("traces=%zu\n", traces.size());

    const size_t input_bytes =
        static_cast<size_t>(layer_input.value().feature_count) * sizeof(uint16_t);
    std::vector<uint16_t> h_input(layer_input.value().feature_count);
    for (size_t i = 0; i < h_input.size(); ++i)
        h_input[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1400u));
    std::vector<uint8_t> h_kv(pool_bytes, 0u);
    for (size_t i = 0; i < pool_bytes / 2u; ++i) {
        const uint16_t v = static_cast<uint16_t>(0x3800u + ((i * 40503u) % 1200u));
        std::memcpy(&h_kv[i * 2u], &v, sizeof(v));
    }

    auto fill_input = [&]() -> bool {
        if (layer_input.value().ptr == nullptr) return false;
        if (hipMemcpyAsync(layer_input.value().ptr, h_input.data(), input_bytes,
                           hipMemcpyHostToDevice, stream) != hipSuccess)
            return false;
        return hipStreamSynchronize(stream) == hipSuccess;
    };
    auto reset_state = [&]() -> bool {
        if (hipMemcpyAsync(k_pool, h_kv.data(), pool_bytes,
                           hipMemcpyHostToDevice, stream) != hipSuccess)
            return false;
        if (hipMemcpyAsync(v_pool, h_kv.data(), pool_bytes,
                           hipMemcpyHostToDevice, stream) != hipSuccess)
            return false;
        if (!fill_input()) return false;
        return hipStreamSynchronize(stream) == hipSuccess;
    };

    std::vector<uint8_t> host_out(out_bytes, 0u);
    std::vector<uint8_t> host_kv(2u * pool_bytes, 0u);
    uint32_t host_physical_launches = 0u;
    auto run_host = [&]() -> bool {
        if (!reset_state()) {
            std::printf("FAIL: reset state\n");
            return false;
        }
        for (uint32_t i = begin; i < end; ++i) {
            auto r = q35::try_launch_optimized(program, i, ctx, stream);
            if (!r.ok()) {
                std::printf("FAIL: host dispatch %u: %s\n", i,
                            r.status().message().c_str());
                return false;
            }
            if (r.value().kind != q35::OptimizedLaunchKind::Launched) {
                std::printf("FAIL: host dispatch %u not launched (kind=%d)\n", i,
                            static_cast<int>(r.value().kind));
                return false;
            }
            host_physical_launches += 1u;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: host sync\n");
            return false;
        }
        (void)hipMemcpy(host_out.data(), last_out.value().ptr, out_bytes,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(host_kv.data(), k_pool, pool_bytes,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(host_kv.data() + pool_bytes, v_pool, pool_bytes,
                        hipMemcpyDeviceToHost);
        for (auto& t : traces) {
            t.host.assign(t.bytes, 0u);
            (void)hipMemcpy(t.host.data(), t.ptr, t.bytes, hipMemcpyDeviceToHost);
        }
        return true;
    };
    if (!run_host()) return 1;
    std::printf("host %s: launched %u dispatches (%u physical launches)\n",
                prefix.c_str(), dispatch_count, host_physical_launches);

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
                mcu::GpuMcuAqlCodeObject::load(hqueue, spec.hsaco, spec.symbol);
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
        if (!rig.load_extra_code(spec.hsaco, spec.symbol, metas[i])) return 1;
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
    if (!rig.set_bf16_invocations(plan.bf16.data(),
                                  static_cast<uint32_t>(plan.bf16.size())))
        return 1;
    if (!rig.set_l2_invocations(plan.l2.data(),
                                static_cast<uint32_t>(plan.l2.size())))
        return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x41544c31u;
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

    std::vector<uint8_t> mcu_out(out_bytes, 0u);
    std::vector<uint8_t> mcu_kv(2u * pool_bytes, 0u);
    int iterations = 100;
    if (const char* it_env = std::getenv("PHASESHIFT_ATTN_ITERS")) {
        const long v = std::strtol(it_env, nullptr, 10);
        if (v > 0) iterations = static_cast<int>(v);
    }
    std::vector<uint8_t> check_buf(out_bytes, 0u);
    uint32_t mismatched_iters = 0u;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
        if (!reset_state()) {
            std::printf("FAIL: reset state (mcu)\n");
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
        if (rig.fsm.fault_code() != mcu::McuFaultCode::none) {
            std::printf("FAIL: fault at iter %d code=%u\n", it,
                        static_cast<unsigned>(rig.fsm.fault_code()));
            return 1;
        }
        if (it + 1 < iterations) {
            rig.device_to_host(check_buf.data(), last_out.value().ptr, out_bytes);
            size_t f3 = 0;
            size_t d3 = 0;
            if (!bytes_equal(host_out.data(), check_buf.data(), out_bytes, &f3,
                             &d3)) {
                ++mismatched_iters;
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    rig.device_to_host(mcu_out.data(), last_out.value().ptr, out_bytes);
    rig.device_to_host(mcu_kv.data(), k_pool, pool_bytes);
    rig.device_to_host(mcu_kv.data() + pool_bytes, v_pool, pool_bytes);
    for (auto& t : traces) {
        t.mcu.assign(t.bytes, 0u);
        rig.device_to_host(t.mcu.data(), t.ptr, t.bytes);
    }

    std::printf(
        "mcu: iterations=%d dispatches=%llu fault=%u physical=%u elapsed=%.3f ms "
        "(%.4f ms/iter) mismatch_iters=%u\n",
        iterations, static_cast<unsigned long long>(rig.fsm.dispatches()),
        static_cast<unsigned>(rig.fsm.fault_code()), plan.physical_dispatch_count,
        ms, ms / static_cast<double>(iterations), mismatched_iters);
    check(rig.fsm.dispatches() ==
              (static_cast<uint64_t>(plan.physical_dispatch_count) + 1u) *
                  static_cast<uint64_t>(iterations),
          "mcu dispatch count");
    check(mismatched_iters == 0u, "no output mismatch across iterations");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");

    size_t first = 0;
    size_t diffs = 0;
    check(bytes_equal(host_out.data(), mcu_out.data(), out_bytes,
                                    &first, &diffs),
          "layer output host vs mcu byte exact");
    if (diffs)
        std::printf("  output diff count=%zu first=%zu\n", diffs, first);
    check(bytes_equal(host_kv.data(), mcu_kv.data(),
                                    2u * pool_bytes, &first, &diffs),
          "kv state host vs mcu byte exact");
    if (diffs)
        std::printf("  kv diff count=%zu first=%zu\n", diffs, first);
    for (const auto& t : traces) {
        if (t.host == t.mcu) continue;
        size_t f2 = 0;
        size_t d2 = 0;
        (void)bytes_equal(t.host.data(), t.mcu.data(), t.bytes, &f2,
                                        &d2);
        std::printf("FAIL: trace %s (dispatch %u) differs %zu bytes first=%zu\n",
                    t.name.c_str(), t.dispatch, d2, f2);
        ++g_failed;
    }
    std::printf("intermediates compared=%zu\n", traces.size());
    std::printf(
        "queue: depth=%u doorbell=per_packet doorbell_count=%llu refill=%llu "
        "empty=%llu mid_region_empty=%llu backpressure=%llu append_ahead=%llu "
        "max_ahead=%llu wrap=%llu\n",
        config.queue_ahead_depth,
        static_cast<unsigned long long>(rig.fsm.doorbell_count()),
        static_cast<unsigned long long>(rig.fsm.refill_count()),
        static_cast<unsigned long long>(rig.fsm.queue_empty_count()),
        static_cast<unsigned long long>(
            rig.fsm.mid_region_queue_empty_count()),
        static_cast<unsigned long long>(rig.fsm.backpressure_wait_count()),
        static_cast<unsigned long long>(rig.fsm.append_ahead_count()),
        static_cast<unsigned long long>(rig.fsm.max_ahead()),
        static_cast<unsigned long long>(rig.fsm.wrap_count()));

    check(rig.close(), "rig teardown");

    run_prefill_scenario();

    const ps::Status bc_destroy =
        ps::runtime::destroy_device_batch_context(batch_storage);
    check(bc_destroy.ok(), "device batch context destroyed");
    (void)hipFree(inv_freq);
    (void)hipFree(rope_pos);
    (void)hipFree(row_pos);
    (void)hipFree(row_slots);
    (void)hipFree(block_tables);
    (void)hipFree(partials);
    (void)hipFree(v_pool);
    (void)hipFree(k_pool);
    if (workspace) (void)hipFree(workspace);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_attention_one_layer_plan: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_attention_one_layer_plan: PASS\n");
    return 0;
}
