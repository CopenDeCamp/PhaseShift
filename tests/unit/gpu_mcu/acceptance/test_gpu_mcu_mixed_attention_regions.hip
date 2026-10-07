#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kBlockTableStride = 16u;
constexpr uint32_t kNumPages = 64u;
constexpr uint32_t kMaxSequences = 4u;
constexpr uint32_t kLayers = 1u;
constexpr uint32_t kQHeads = 16u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kMaxVisible = 256u;

constexpr uint32_t kQFeatures = kQHeads * kHeadDim;
constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;
constexpr uint32_t kElemsPerLayer = kNumPages * kElemsPerPage;
constexpr size_t kPoolBytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);

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

uint16_t recipe_for(q35::McuCompiledVariantKind) {
    return mcu::kMcuKernargRecipeAttentionPaged;
}

struct Scenario {
    const char* name;
    std::vector<::ps::runtime::DeviceRequestDescriptor> requests;
    std::vector<uint32_t> slots;
    std::vector<uint32_t> positions;
    uint32_t total_rows = 0u;
    uint32_t expected_physical = 0u;
    bool expect_prefill = false;
    bool expect_decode = false;
};

bool build_program(::ps::runtime::Program* program) {
    *program = ::ps::runtime::Program{};
    program->row_bucket = ::ps::runtime::RowBucket::R256;
    program->execution_class = ::ps::runtime::ExecutionClass::PREFILL;

    ::ps::runtime::ValueBinding vq{};
    vq.logical_value = ::ps::runtime::ValueId{0u};
    vq.storage = ::ps::runtime::ValueStorage::EXTERNAL_INPUT;
    vq.dtype = ::ps::runtime::ValueDType::BF16;
    vq.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    vq.slot = 0u;
    vq.feature_count = kQFeatures;
    vq.row_stride = kQFeatures;
    program->values.push_back(vq);

    ::ps::runtime::ValueBinding vo{};
    vo.logical_value = ::ps::runtime::ValueId{1u};
    vo.storage = ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT;
    vo.dtype = ::ps::runtime::ValueDType::F32;
    vo.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    vo.slot = 0u;
    vo.feature_count = kQFeatures;
    vo.row_stride = kQFeatures;
    program->values.push_back(vo);

    ::ps::runtime::StateBinding st{};
    st.state = ::ps::runtime::StateId{0u};
    st.kind = ::ps::runtime::PrimitiveStateKind::KV_CACHE;
    st.state_index = 0u;
    st.slot = 0u;
    program->states.push_back(st);

    ::ps::runtime::DispatchBinding ad{};
    ad.kernel_id = ::ps::runtime::KernelId::PAGED_ATTENTION;
    ad.descriptor_slot = 1u;
    ad.input_count = 1u;
    ad.input_slots[0] = 0u;
    ad.output_count = 1u;
    ad.output_slots[0] = 1u;
    ad.group_size = kHeadDim;
    ad.scalar_a = 1.0f / 1.6e1f;
    program->dispatches.push_back(ad);
    program->dispatch_source_node.push_back(0u);
    program->external_input_count = 1u;
    program->external_output_count = 1u;
    return true;
}

bool run_scenario(const Scenario& s) {
    const uint32_t rows = s.total_rows;
    const size_t q_bytes = static_cast<size_t>(rows) * kQFeatures * sizeof(uint16_t);
    const size_t out_bytes = static_cast<size_t>(rows) * kQFeatures * sizeof(float);

    std::printf("[%s] rows=%u requests=%zu expected_physical=%u\n", s.name, rows,
                s.requests.size(), s.expected_physical);

    void* q_dev = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* out_ref = nullptr;
    void* out_mcu = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    if (hipMalloc(&q_dev, q_bytes) != hipSuccess ||
        hipMalloc(&k_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&out_ref, out_bytes) != hipSuccess ||
        hipMalloc(&out_mcu, out_bytes) != hipSuccess ||
        hipMalloc(&row_slots, rows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, rows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables,
                  kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess) {
        check(false, "scenario buffers allocated");
        return false;
    }

    auto release_buffers = [&]() {
        (void)hipFree(block_tables);
        (void)hipFree(row_pos);
        (void)hipFree(row_slots);
        (void)hipFree(out_mcu);
        (void)hipFree(out_ref);
        (void)hipFree(v_pool);
        (void)hipFree(k_pool);
        (void)hipFree(q_dev);
    };

    std::vector<uint16_t> h_q(static_cast<size_t>(rows) * kQFeatures);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
    std::vector<uint16_t> h_pool(kElemsPerLayer);
    for (size_t i = 0; i < h_pool.size(); ++i)
        h_pool[i] = static_cast<uint16_t>(0x3900u + ((i * 40503u) % 1700u));
    std::vector<uint32_t> h_bt(kMaxSequences * kBlockTableStride);
    for (uint32_t slot = 0u; slot < kMaxSequences; ++slot)
        for (uint32_t b = 0u; b < kBlockTableStride; ++b)
            h_bt[slot * kBlockTableStride + b] = slot * kBlockTableStride + b;

    if (hipMemcpy(q_dev, h_q.data(), q_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(k_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
        check(false, "scenario q/pool uploaded");
        release_buffers();
        return false;
    }
    for (size_t i = 0; i < h_pool.size(); ++i)
        h_pool[i] = static_cast<uint16_t>(0x3B00u + ((i * 2654435761u) % 1500u));
    if (hipMemcpy(v_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(row_slots, s.slots.data(), rows * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(row_pos, s.positions.data(), rows * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(block_tables, h_bt.data(), h_bt.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        check(false, "scenario metadata uploaded");
        release_buffers();
        return false;
    }
    (void)hipMemset(out_ref, 0xAB, out_bytes);
    (void)hipMemset(out_mcu, 0xAB, out_bytes);

    ::ps::runtime::DeviceBatchContext bc{};
    bc.actual_rows = rows;
    bc.num_requests = static_cast<uint32_t>(s.requests.size());
    for (const auto& r : s.requests) {
        if (r.execution_class == ::ps::runtime::ExecutionClass::PREFILL)
            bc.num_prefill_requests += 1u;
        else
            bc.num_decode_requests += 1u;
    }
    bc.requests = s.requests.data();
    bc.row_sequence_slots = static_cast<uint32_t*>(row_slots);
    bc.row_positions = static_cast<uint32_t*>(row_pos);

    ::ps::runtime::Program program;
    build_program(&program);

    const void* input_ptrs[1] = {q_dev};
    void* output_ptrs[1] = {out_ref};

    q35::HostExecutionContext ctx{};
    ctx.batch_context = &bc;
    ctx.external_inputs = input_ptrs;
    ctx.external_input_count = 1u;
    ctx.external_outputs = output_ptrs;
    ctx.external_output_count = 1u;
    ctx.actual_rows = rows;
    ctx.actual_outputs = 1u;
    ctx.row_positions = static_cast<const uint32_t*>(row_pos);
    ctx.row_sequence_slots = static_cast<const uint32_t*>(row_slots);
    ctx.max_visible_tokens = kMaxVisible;
    ctx.decode_attn_partials = nullptr;
    ctx.decode_attn_partial_bytes = 0u;
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
            .num_pages = kNumPages,
            .num_attention_layers = kLayers,
            .kv_heads = kKvHeads,
            .head_dim = kHeadDim,
            .elems_per_token = kElemsPerToken,
            .elems_per_page = kElemsPerPage,
            .elems_per_layer = kElemsPerLayer};
        ms.kv_dtype = static_cast<uint32_t>(::ps::qwen35::KVCacheDType::BF16);
        ctx.model_state = ms;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        check(false, "scenario stream");
        release_buffers();
        return false;
    }

    q35::PhysicalPagedAttentionPlan host_plan{};
    const auto resolved =
        q35::resolve_paged_attention_physical(program, 0u, ctx, host_plan);
    if (!resolved.ok() ||
        resolved.value() != q35::PhysicalResolveStatus::Applicable) {
        check(false, "host region plan applicable");
        (void)hipStreamDestroy(stream);
        release_buffers();
        return false;
    }
    check(host_plan.physical_count == s.expected_physical,
          "host physical region count");
    bool host_prefill = false;
    bool host_decode = false;
    for (uint32_t i = 0u; i < host_plan.physical_count; ++i) {
        host_prefill = host_prefill ||
            host_plan.kernels[i].variant ==
                q35::PhysicalPagedAttentionVariant::Bf16Prefill;
        host_decode = host_decode ||
            host_plan.kernels[i].variant ==
                q35::PhysicalPagedAttentionVariant::Bf16FullHead;
    }
    check(host_prefill == s.expect_prefill, "host plan prefill region present");
    check(host_decode == s.expect_decode, "host plan decode region present");
    uint32_t covered_rows = 0u;
    for (uint32_t i = 0u; i < host_plan.physical_count; ++i)
        covered_rows += host_plan.kernels[i].common.rows;
    check(covered_rows == rows, "host region coverage exact");
    if (q35::launch_paged_attention_physical(host_plan, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        check(false, "host region launch");
        (void)hipStreamDestroy(stream);
        release_buffers();
        return false;
    }

    q35::McuPlanCompileOptions opts;
    opts.dispatch_begin = 0u;
    opts.dispatch_end = 1u;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    output_ptrs[0] = out_mcu;
    q35::McuCompiledPlan plan;
    const ps::Status plan_status = q35::compile_mcu_plan(program, ctx, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: %s plan: %s\n", s.name, plan_status.message().c_str());
        check(false, "mcu region plan compiled");
        (void)hipStreamDestroy(stream);
        release_buffers();
        return false;
    }
    check(plan.logical_dispatch_count == 1u, "one logical attention dispatch");
    check(plan.physical_dispatch_count == s.expected_physical,
          "mcu physical region count");
    check(plan.attention_paged.size() == s.expected_physical,
          "one attention invocation per region");

    Rig rig;
    std::vector<mcu::McuKernelVariantDesc> variants(plan.variants.size());
    std::vector<mcu::GpuAqlKernelMetadata> metas(plan.variants.size());
    uint32_t needed_segment = 0u;
    {
        auto hq_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                  mcu::kAqlDefaultQueueSize);
        if (!hq_res.ok()) {
            check(false, "host metadata queue");
            (void)hipStreamDestroy(stream);
            release_buffers();
            return false;
        }
        auto hqueue = hq_res.release();
        for (size_t i = 0; i < plan.variants.size(); ++i) {
            if (plan.variants[i].kind ==
                q35::McuCompiledVariantKind::CompletionMarker)
                continue;
            SymbolSpec spec{};
            if (!symbol_for_kind(plan.variants[i].kind, &spec)) {
                check(false, "symbol for variant kind");
                (void)hqueue.shutdown();
                (void)hipStreamDestroy(stream);
                release_buffers();
                return false;
            }
            auto co_res =
                mcu::GpuMcuAqlCodeObject::load(hqueue, spec.hsaco, spec.symbol);
            if (!co_res.ok()) {
                check(false, "attention code object loaded");
                (void)hqueue.shutdown();
                (void)hipStreamDestroy(stream);
                release_buffers();
                return false;
            }
            auto co = co_res.release();
            metas[i] = co.metadata();
            if (metas[i].kernarg_segment_size > needed_segment)
                needed_segment = metas[i].kernarg_segment_size;
        }
        if (!hqueue.shutdown().ok()) {
            check(false, "host metadata queue shutdown");
            (void)hipStreamDestroy(stream);
            release_buffers();
            return false;
        }
    }

    if (!rig.open(0, static_cast<uint32_t>(plan.nodes.size()) + 4u,
                  static_cast<uint32_t>(plan.variants.size()) + 2u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  needed_segment)) {
        check(false, "rig open");
        (void)hipStreamDestroy(stream);
        release_buffers();
        return false;
    }
    for (size_t i = 0; i < plan.variants.size(); ++i) {
        if (plan.variants[i].kind == q35::McuCompiledVariantKind::CompletionMarker)
            continue;
        SymbolSpec spec{};
        (void)symbol_for_kind(plan.variants[i].kind, &spec);
        if (!rig.load_extra_code(spec.hsaco, spec.symbol, metas[i])) {
            check(false, "rig extra code");
            (void)rig.close();
            (void)hipStreamDestroy(stream);
            release_buffers();
            return false;
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
    if (!rig.upload(plan.nodes, variants) ||
        !rig.set_attention_paged_invocations(
            plan.attention_paged.data(),
            static_cast<uint32_t>(plan.attention_paged.size()))) {
        check(false, "rig upload");
        (void)rig.close();
        (void)hipStreamDestroy(stream);
        release_buffers();
        return false;
    }
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4D495831u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = plan.kernarg_slots_required;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config) || !rig.fsm.request_run().ok() ||
        !rig.fsm.wait_plans(1u, 60000).ok() || !rig.drain_queue(60000)) {
        check(false, "mcu plan completed");
        (void)rig.close();
        (void)hipStreamDestroy(stream);
        release_buffers();
        return false;
    }
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "mcu no fault");
    check(rig.fsm.dispatches() ==
              static_cast<uint64_t>(plan.physical_dispatch_count) + 1u,
          "mcu dispatch count");

    std::vector<uint8_t> h_ref(out_bytes, 0u);
    std::vector<uint8_t> h_mcu(out_bytes, 0u);
    rig.device_to_host(h_ref.data(), out_ref, out_bytes);
    rig.device_to_host(h_mcu.data(), out_mcu, out_bytes);
    size_t first = 0u;
    size_t diffs = 0u;
    const bool same = bytes_equal(h_ref.data(), h_mcu.data(), out_bytes, &first,
                                  &diffs);
    check(same, "region output host vs mcu byte exact");
    if (diffs != 0u) {
        std::printf("  %s output diff count=%zu first=%zu\n", s.name, diffs, first);
        for (uint32_t r = 0u; r < rows; r += (rows > 8u ? 8u : 1u)) {
            const size_t off = static_cast<size_t>(r) * kQFeatures * sizeof(float);
            size_t rf = 0u;
            size_t rd = 0u;
            (void)bytes_equal(h_ref.data() + off, h_mcu.data() + off,
                              static_cast<size_t>(kQFeatures) * sizeof(float), &rf,
                              &rd);
            std::printf("    row=%u diffs=%zu\n", r, rd);
        }
        for (uint32_t c = 0u; c < 4u; ++c) {
            float a = 0.0f;
            float b = 0.0f;
            std::memcpy(&a, h_ref.data() + c * sizeof(float), sizeof(float));
            std::memcpy(&b, h_mcu.data() + c * sizeof(float), sizeof(float));
            std::printf("    word[%u] ref=%g mcu=%g\n", c, a, b);
        }
    }
    std::printf("[%s] regions=%u dispatches=%llu diffs=%zu\n", s.name,
                plan.physical_dispatch_count,
                static_cast<unsigned long long>(rig.fsm.dispatches()), diffs);

    check(rig.close(), "rig teardown");
    (void)hipStreamDestroy(stream);
    release_buffers();
    return same;
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

    Scenario decode_prefill_decode;
    decode_prefill_decode.name = "decode_prefill_decode";
    decode_prefill_decode.total_rows = 162u;
    decode_prefill_decode.expected_physical = 2u;
    decode_prefill_decode.expect_prefill = true;
    decode_prefill_decode.expect_decode = true;
    {
        ::ps::runtime::DeviceRequestDescriptor d0{};
        d0.execution_class = ::ps::runtime::ExecutionClass::DECODE;
        d0.request_handle.slot = 0u;
        d0.row_begin = 0u;
        d0.row_count = 1u;
        d0.prefix_length = 100u;
        d0.sequence_length = 101u;
        ::ps::runtime::DeviceRequestDescriptor d2{};
        d2.execution_class = ::ps::runtime::ExecutionClass::DECODE;
        d2.request_handle.slot = 2u;
        d2.row_begin = 1u;
        d2.row_count = 1u;
        d2.prefix_length = 100u;
        d2.sequence_length = 101u;
        ::ps::runtime::DeviceRequestDescriptor d1{};
        d1.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        d1.request_handle.slot = 1u;
        d1.row_begin = 2u;
        d1.row_count = 160u;
        d1.prefix_length = 0u;
        d1.sequence_length = 160u;
        decode_prefill_decode.requests = {d0, d2, d1};
        decode_prefill_decode.slots.assign(162u, 1u);
        decode_prefill_decode.positions.resize(162u);
        decode_prefill_decode.slots[0] = 0u;
        decode_prefill_decode.slots[1] = 2u;
        decode_prefill_decode.positions[0] = 100u;
        decode_prefill_decode.positions[1] = 100u;
        for (uint32_t r = 0u; r < 160u; ++r)
            decode_prefill_decode.positions[2u + r] = r;
    }
    (void)run_scenario(decode_prefill_decode);

    Scenario prefill_prefill;
    prefill_prefill.name = "prefill_prefill";
    prefill_prefill.total_rows = 288u;
    prefill_prefill.expected_physical = 2u;
    prefill_prefill.expect_prefill = true;
    prefill_prefill.expect_decode = false;
    {
        ::ps::runtime::DeviceRequestDescriptor a{};
        a.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        a.request_handle.slot = 1u;
        a.row_begin = 0u;
        a.row_count = 160u;
        a.prefix_length = 0u;
        a.sequence_length = 160u;
        ::ps::runtime::DeviceRequestDescriptor b{};
        b.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        b.request_handle.slot = 3u;
        b.row_begin = 160u;
        b.row_count = 128u;
        b.prefix_length = 0u;
        b.sequence_length = 128u;
        prefill_prefill.requests = {a, b};
        prefill_prefill.slots.assign(288u, 1u);
        prefill_prefill.positions.resize(288u);
        for (uint32_t r = 160u; r < 288u; ++r) {
            prefill_prefill.slots[r] = 3u;
            prefill_prefill.positions[r] = r - 160u;
        }
        for (uint32_t r = 0u; r < 160u; ++r)
            prefill_prefill.positions[r] = r;
    }
    (void)run_scenario(prefill_prefill);

    Scenario decode_verify_prefill;
    decode_verify_prefill.name = "decode_verify_prefill";
    decode_verify_prefill.total_rows = 137u;
    decode_verify_prefill.expected_physical = 2u;
    decode_verify_prefill.expect_prefill = true;
    decode_verify_prefill.expect_decode = true;
    {
        ::ps::runtime::DeviceRequestDescriptor d0{};
        d0.execution_class = ::ps::runtime::ExecutionClass::DECODE;
        d0.request_handle.slot = 0u;
        d0.row_begin = 0u;
        d0.row_count = 1u;
        d0.prefix_length = 100u;
        d0.sequence_length = 101u;
        ::ps::runtime::DeviceRequestDescriptor dv{};
        dv.execution_class = ::ps::runtime::ExecutionClass::SPEC_VERIFY;
        dv.request_handle.slot = 2u;
        dv.row_begin = 1u;
        dv.row_count = 8u;
        dv.prefix_length = 100u;
        dv.sequence_length = 108u;
        ::ps::runtime::DeviceRequestDescriptor dp{};
        dp.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        dp.request_handle.slot = 1u;
        dp.row_begin = 9u;
        dp.row_count = 128u;
        dp.prefix_length = 0u;
        dp.sequence_length = 128u;
        decode_verify_prefill.requests = {d0, dv, dp};
        decode_verify_prefill.slots.assign(137u, 1u);
        decode_verify_prefill.positions.resize(137u);
        decode_verify_prefill.slots[0] = 0u;
        decode_verify_prefill.positions[0] = 100u;
        for (uint32_t r = 0u; r < 8u; ++r) {
            decode_verify_prefill.slots[1u + r] = 2u;
            decode_verify_prefill.positions[1u + r] = 100u + r;
        }
        for (uint32_t r = 0u; r < 128u; ++r)
            decode_verify_prefill.positions[9u + r] = r;
    }
    (void)run_scenario(decode_verify_prefill);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_mixed_attention_regions: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_mixed_attention_regions: PASS\n");
    return 0;
}
