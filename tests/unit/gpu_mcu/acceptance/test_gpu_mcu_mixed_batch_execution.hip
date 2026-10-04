#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;
namespace q35 = ::ps::qwen35::runtime;
namespace kernel = ::ps::kernel;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kRows = 26u;
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

bool build_program(rt::Program* program) {
    *program = rt::Program{};
    program->row_bucket = rt::RowBucket::R256;
    program->execution_class = rt::ExecutionClass::PREFILL;

    rt::ValueBinding vq{};
    vq.logical_value = rt::ValueId{0u};
    vq.storage = rt::ValueStorage::EXTERNAL_INPUT;
    vq.dtype = rt::ValueDType::BF16;
    vq.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    vq.slot = 0u;
    vq.feature_count = kQFeatures;
    vq.row_stride = kQFeatures;
    program->values.push_back(vq);

    rt::ValueBinding vo{};
    vo.logical_value = rt::ValueId{1u};
    vo.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    vo.dtype = rt::ValueDType::F32;
    vo.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    vo.slot = 0u;
    vo.feature_count = kQFeatures;
    vo.row_stride = kQFeatures;
    program->values.push_back(vo);

    rt::StateBinding st{};
    st.state = rt::StateId{0u};
    st.kind = rt::PrimitiveStateKind::KV_CACHE;
    st.state_index = 0u;
    st.slot = 0u;
    program->states.push_back(st);

    rt::DispatchBinding ad{};
    ad.kernel_id = rt::KernelId::PAGED_ATTENTION;
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

    const size_t q_bytes = static_cast<size_t>(kRows) * kQFeatures * sizeof(uint16_t);
    const size_t out_bytes = static_cast<size_t>(kRows) * kQFeatures * sizeof(float);

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
        hipMalloc(&row_slots, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables,
                  kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess) {
        std::printf("FAIL: buffers\n");
        return 1;
    }

    std::vector<uint16_t> h_q(static_cast<size_t>(kRows) * kQFeatures);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
    std::vector<uint16_t> h_pool(kElemsPerLayer);
    for (size_t i = 0; i < h_pool.size(); ++i)
        h_pool[i] = static_cast<uint16_t>(0x3900u + ((i * 40503u) % 1700u));
    std::vector<uint32_t> h_bt(kMaxSequences * kBlockTableStride);
    for (uint32_t slot = 0u; slot < kMaxSequences; ++slot)
        for (uint32_t b = 0u; b < kBlockTableStride; ++b)
            h_bt[slot * kBlockTableStride + b] = slot * kBlockTableStride + b;

    (void)hipMemcpy(q_dev, h_q.data(), q_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(k_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
    for (size_t i = 0; i < h_pool.size(); ++i)
        h_pool[i] = static_cast<uint16_t>(0x3B00u + ((i * 2654435761u) % 1500u));
    (void)hipMemcpy(v_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);

    std::vector<uint32_t> slots(kRows, 1u);
    std::vector<uint32_t> positions(kRows, 0u);
    slots[0] = 0u;
    positions[0] = 100u;
    slots[1] = 1u;
    positions[1] = 100u;
    for (uint32_t r = 0u; r < 8u; ++r) {
        slots[2u + r] = 2u;
        positions[2u + r] = 100u + r;
    }
    for (uint32_t r = 0u; r < 16u; ++r) {
        slots[10u + r] = 3u;
        positions[10u + r] = r;
    }
    (void)hipMemcpy(row_slots, slots.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(row_pos, positions.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(block_tables, h_bt.data(), h_bt.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemset(out_ref, 0xAB, out_bytes);
    (void)hipMemset(out_mcu, 0xAB, out_bytes);

    std::vector<rt::DeviceRequestDescriptor> requests;
    {
        rt::DeviceRequestDescriptor d0{};
        d0.execution_class = rt::ExecutionClass::DECODE;
        d0.request_handle = rt::RequestHandle{0u, 1u};
        d0.row_begin = 0u;
        d0.row_count = 1u;
        d0.prefix_length = 100u;
        d0.sequence_length = 101u;
        rt::DeviceRequestDescriptor d1{};
        d1.execution_class = rt::ExecutionClass::DECODE;
        d1.request_handle = rt::RequestHandle{1u, 1u};
        d1.row_begin = 1u;
        d1.row_count = 1u;
        d1.prefix_length = 100u;
        d1.sequence_length = 101u;
        rt::DeviceRequestDescriptor dv{};
        dv.execution_class = rt::ExecutionClass::SPEC_VERIFY;
        dv.request_handle = rt::RequestHandle{2u, 1u};
        dv.row_begin = 2u;
        dv.row_count = 8u;
        dv.prefix_length = 100u;
        dv.sequence_length = 108u;
        rt::DeviceRequestDescriptor dp{};
        dp.execution_class = rt::ExecutionClass::PREFILL;
        dp.request_handle = rt::RequestHandle{3u, 1u};
        dp.row_begin = 10u;
        dp.row_count = 16u;
        dp.prefix_length = 0u;
        dp.sequence_length = 16u;
        requests = {d0, d1, dv, dp};
    }

    rt::DeviceBatchContext bc{};
    bc.actual_rows = kRows;
    bc.num_requests = static_cast<uint32_t>(requests.size());
    bc.num_decode_requests = 2u;
    bc.num_verify_requests = 1u;
    bc.num_prefill_requests = 1u;
    bc.requests = requests.data();
    bc.row_sequence_slots = static_cast<uint32_t*>(row_slots);
    bc.row_positions = static_cast<uint32_t*>(row_pos);

    rt::Program program;
    build_program(&program);

    const void* input_ptrs[1] = {q_dev};
    void* output_ptrs[1] = {out_ref};

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
    ctx.max_visible_tokens = kMaxVisible;
    ctx.decode_attn_partials = nullptr;
    ctx.decode_attn_partial_bytes = 0u;
    {
        kernel::ModelDispatchStateView ms{};
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
        std::printf("FAIL: stream\n");
        return 1;
    }

    q35::PhysicalPagedAttentionPlan host_plan{};
    const auto resolved =
        q35::resolve_paged_attention_physical(program, 0u, ctx, host_plan);
    if (!resolved.ok() ||
        resolved.value() != q35::PhysicalResolveStatus::Applicable) {
        std::printf("FAIL: host region plan applicable\n");
        return 1;
    }
    check(host_plan.physical_count == 2u, "host physical region count");
    uint32_t covered_rows = 0u;
    for (uint32_t i = 0u; i < host_plan.physical_count; ++i)
        covered_rows += host_plan.kernels[i].common.rows;
    check(covered_rows == kRows, "host region coverage exact");
    check(host_plan.kernels[0].common.rows == 10u,
          "decode+verify region covers 10 rows");
    check(host_plan.kernels[1].common.rows == 16u,
          "prefill region covers 16 rows");
    if (q35::launch_paged_attention_physical(host_plan, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: host region launch\n");
        return 1;
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
        std::printf("FAIL: plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    check(plan.logical_dispatch_count == 1u, "one logical attention dispatch");
    check(plan.physical_dispatch_count == 2u, "two physical regions");
    check(plan.attention_paged.size() == 2u, "one invocation per region");

    q35::McuDecodeRuntimeOptions options{};
    options.device = 0;
    options.record_entries = 0u;
    options.driver_mode = q35::McuPlanDriverMode::ExternalPersistentMcu;
    auto runtime_result = q35::McuDecodeRuntime::create(options);
    if (!runtime_result.ok()) {
        std::printf("FAIL: runtime create: %s\n",
                    runtime_result.status().message().c_str());
        return 1;
    }
    q35::McuDecodeRuntime runtime = runtime_result.release();
    const ps::Status prepared = runtime.prepare_plan(plan);
    if (!prepared.ok()) {
        std::printf("FAIL: prepare_plan: %s\n", prepared.message().c_str());
        return 1;
    }
    check(runtime.ready(), "plan prepared");
    check(runtime.external_driver(), "external driver mode active");
    check(!runtime.fsm().running(), "host stream controller not started");

    rt::DeviceBatchContext* ctx_dev = nullptr;
    rt::DeviceRequestDescriptor* req_dev = nullptr;
    uint64_t* ready_dev = nullptr;
    uint32_t* row_seq_dev = nullptr;
    uint32_t* row_pos_dev = nullptr;
    uint64_t* plan_epoch_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&ctx_dev),
                  sizeof(rt::DeviceBatchContext)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&req_dev),
                  requests.size() * sizeof(rt::DeviceRequestDescriptor)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_seq_dev),
                  kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_pos_dev),
                  kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&plan_epoch_dev), sizeof(uint64_t)) !=
            hipSuccess) {
        std::printf("FAIL: context allocation\n");
        return 1;
    }
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(plan_epoch_dev, 0, sizeof(uint64_t));
    (void)hipMemcpy(row_seq_dev, slots.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(row_pos_dev, positions.data(), kRows * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(req_dev, requests.data(),
                    requests.size() * sizeof(rt::DeviceRequestDescriptor),
                    hipMemcpyHostToDevice);
    rt::DeviceBatchContext ctx_copy = bc;
    ctx_copy.requests = req_dev;
    ctx_copy.row_sequence_slots = row_seq_dev;
    ctx_copy.row_positions = row_pos_dev;
    (void)hipMemcpy(ctx_dev, &ctx_copy, sizeof(ctx_copy), hipMemcpyHostToDevice);

    hipStream_t persistent_stream = nullptr;
    if (hipStreamCreate(&persistent_stream) != hipSuccess) {
        std::printf("FAIL: persistent stream\n");
        return 1;
    }
    auto persistent_result = mcu::GpuMcuPersistentMcu::create(0, persistent_stream);
    if (!persistent_result.ok()) {
        std::printf("FAIL: persistent create\n");
        return 1;
    }
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();

    check(persistent
              .configure_batch_plan(ctx_dev, req_dev, row_seq_dev, row_pos_dev,
                                    static_cast<uint32_t>(requests.size()), kRows,
                                    16u, plan_epoch_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev, nullptr,
                                   0u, nullptr, 0u)
              .ok(),
          "configure execution with prepared fsm");
    const uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent mcu start");
    check(persistent.running(), "persistent mcu running");

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (persistent.batches_completed() < 1u) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check(persistent.batches_dispatched() >= 1u, "mixed plan dispatched");
    check(persistent.batches_completed() >= 1u, "mixed plan completed");
    check(persistent.batches_failed() == 0u, "mixed plan did not fault");
    check(!runtime.fsm().running(), "no second controller kernel");
    check(runtime.execution_state()->plans_started >= 1u,
          "prepared fsm recorded a plan");
    check(runtime.execution_state()->dispatches_committed >= 3u,
          "decode + prefill + marker committed");
    check(runtime.execution_state()->fault_code ==
              static_cast<uint32_t>(mcu::McuFaultCode::none),
          "prepared fsm has no fault");

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");

    std::vector<uint8_t> h_ref(out_bytes, 0u);
    std::vector<uint8_t> h_mcu(out_bytes, 0u);
    (void)hipMemcpy(h_ref.data(), out_ref, out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_mcu.data(), out_mcu, out_bytes, hipMemcpyDeviceToHost);
    size_t first = 0u;
    size_t diffs = 0u;
    const bool same =
        bytes_equal(h_ref.data(), h_mcu.data(), out_bytes, &first, &diffs);
    check(same, "canonical mixed output host vs persistent mcu byte exact");
    if (diffs != 0u)
        std::printf("  output diff count=%zu first=%zu\n", diffs, first);
    std::printf(
        "canonical_mixed_26: regions=%u committed=%llu diffs=%zu\n",
        plan.physical_dispatch_count,
        static_cast<unsigned long long>(persistent.batches_committed()), diffs);

    check(runtime.shutdown().ok(), "runtime shutdown");

    (void)hipFree(plan_epoch_dev);
    (void)hipFree(row_pos_dev);
    (void)hipFree(row_seq_dev);
    (void)hipFree(ready_dev);
    (void)hipFree(req_dev);
    (void)hipFree(ctx_dev);
    check(hipStreamDestroy(persistent_stream) == hipSuccess,
          "persistent stream destroy");
    (void)hipStreamDestroy(stream);
    (void)hipFree(block_tables);
    (void)hipFree(row_pos);
    (void)hipFree(row_slots);
    (void)hipFree(out_mcu);
    (void)hipFree(out_ref);
    (void)hipFree(v_pool);
    (void)hipFree(k_pool);
    (void)hipFree(q_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_mixed_batch_execution: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_mixed_batch_execution: PASS\n");
    return 0;
}
