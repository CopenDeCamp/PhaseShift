#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/runtime/gdn_conv_dispatch.h>
#include <phaseshift/models/qwen35/runtime/gdn_recurrence_dispatch.h>
#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
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

namespace mcu = ps::runtime::gpu_mcu;
namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kRows = 26u;
constexpr uint32_t kConvDim = 8192u;
constexpr uint32_t kConvHistory = 3u;
constexpr uint32_t kConvHistoryStride = kConvDim;
constexpr uint32_t kConvLayerStride = kConvHistory * kConvHistoryStride;
constexpr uint32_t kConvSlotStride = kConvLayerStride;
constexpr uint32_t kKeyHeads = 16u;
constexpr uint32_t kNumVHeads = 32u;
constexpr uint32_t kHeadK = 128u;
constexpr uint32_t kHeadV = 128u;
constexpr uint32_t kQkDim = kKeyHeads * kHeadK;
constexpr uint32_t kValueDim = kNumVHeads * kHeadV;
constexpr uint32_t kRecurrentLayerStride = kNumVHeads * kHeadK * kHeadV;
constexpr uint32_t kRecurrentSlotStride = kRecurrentLayerStride;
constexpr uint32_t kMaxSequences = 4u;

void build_program(rt::Program& program) {
    program = rt::Program{};
    program.row_bucket = rt::RowBucket::R64;
    program.execution_class = rt::ExecutionClass::DECODE;
    program.parameter_slot_count = 3u;

    auto add_value = [&](uint32_t id, rt::ValueStorage storage, rt::ValueDType dtype,
                         uint32_t slot, uint32_t features) {
        rt::ValueBinding v{};
        v.logical_value = rt::ValueId{id};
        v.storage = storage;
        v.dtype = dtype;
        v.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
        v.slot = slot;
        v.feature_count = features;
        v.row_stride = features;
        program.values.push_back(v);
    };
    add_value(0u, rt::ValueStorage::EXTERNAL_INPUT, rt::ValueDType::BF16, 0u,
              kConvDim);
    add_value(1u, rt::ValueStorage::EXTERNAL_OUTPUT, rt::ValueDType::F32, 0u,
              kConvDim);
    add_value(2u, rt::ValueStorage::EXTERNAL_INPUT, rt::ValueDType::F32, 1u,
              kQkDim);
    add_value(3u, rt::ValueStorage::EXTERNAL_INPUT, rt::ValueDType::F32, 2u,
              kQkDim);
    add_value(4u, rt::ValueStorage::EXTERNAL_INPUT, rt::ValueDType::BF16, 3u,
              kValueDim);
    add_value(5u, rt::ValueStorage::EXTERNAL_INPUT, rt::ValueDType::BF16, 4u,
              kNumVHeads);
    add_value(6u, rt::ValueStorage::EXTERNAL_INPUT, rt::ValueDType::BF16, 5u,
              kNumVHeads);
    add_value(7u, rt::ValueStorage::EXTERNAL_OUTPUT, rt::ValueDType::F32, 1u,
              kValueDim);

    rt::StateBinding conv_state{};
    conv_state.state = rt::StateId{0u};
    conv_state.kind = rt::PrimitiveStateKind::GDN_CONV_STATE;
    conv_state.state_index = 0u;
    conv_state.slot = 0u;
    program.states.push_back(conv_state);

    rt::StateBinding rec_state{};
    rec_state.state = rt::StateId{1u};
    rec_state.kind = rt::PrimitiveStateKind::GDN_RECURRENCE_STATE;
    rec_state.state_index = 0u;
    rec_state.slot = 1u;
    program.states.push_back(rec_state);

    rt::DispatchBinding conv{};
    conv.kernel_id = rt::KernelId::STATEFUL_CAUSAL_CONV1D;
    conv.descriptor_slot = 1u;
    conv.input_count = 1u;
    conv.input_slots[0] = 0u;
    conv.output_count = 1u;
    conv.output_slots[0] = 1u;
    conv.parameter_index0 = 0u;
    program.dispatches.push_back(conv);
    program.dispatch_source_node.push_back(0u);

    rt::DispatchBinding rec{};
    rec.kernel_id = rt::KernelId::GDN_RECURRENCE;
    rec.descriptor_slot = 2u;
    rec.input_count = 5u;
    rec.input_slots[0] = 2u;
    rec.input_slots[1] = 3u;
    rec.input_slots[2] = 4u;
    rec.input_slots[3] = 5u;
    rec.input_slots[4] = 6u;
    rec.output_count = 1u;
    rec.output_slots[0] = 7u;
    rec.parameter_index0 = 1u;
    rec.parameter_index1 = 2u;
    program.dispatches.push_back(rec);
    program.dispatch_source_node.push_back(1u);

    program.external_input_count = 6u;
    program.external_output_count = 2u;
}

::ps::qwen35::GdnStatePoolDeviceView make_gdn_view(void* conv_base,
                                                   void* recurrent_base) {
    ::ps::qwen35::GdnStatePoolDeviceView gdn{};
    gdn.conv_base = static_cast<ps::bf16_t*>(conv_base);
    gdn.recurrent_base = static_cast<float*>(recurrent_base);
    gdn.num_gdn_states = 1u;
    gdn.max_sequences = kMaxSequences;
    gdn.conv_slot_stride = kConvSlotStride;
    gdn.conv_layer_stride = kConvLayerStride;
    gdn.conv_history_stride = kConvHistoryStride;
    gdn.recurrent_slot_stride = kRecurrentSlotStride;
    gdn.recurrent_layer_stride = kRecurrentLayerStride;
    gdn.conv_dim = kConvDim;
    gdn.conv_dim_padded = kConvDim;
    gdn.conv_history = kConvHistory;
    gdn.num_v_heads = kNumVHeads;
    gdn.head_k = kHeadK;
    gdn.head_v = kHeadV;
    return gdn;
}

bool wait_for(const mcu::GpuMcuPersistentMcu& persistent, uint64_t target,
              uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (persistent.batches_completed() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return true;
}

bool bytes_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                 size_t* first, size_t* diffs) {
    *first = 0u;
    *diffs = 0u;
    for (size_t i = 0u; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        if (*diffs == 0u) *first = i;
        ++(*diffs);
    }
    return *diffs == 0u;
}

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    void* p = nullptr;
    if (hipMallocManaged(&p, bytes) != hipSuccess) return false;
    *host = p;
    *dev = p;
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

    rt::Program program;
    build_program(program);
    check(program.dispatches.size() == 2u, "program has two dispatches");

    const size_t conv_in_bytes =
        static_cast<size_t>(kRows) * kConvDim * sizeof(uint16_t);
    const size_t conv_out_bytes =
        static_cast<size_t>(kRows) * kConvDim * sizeof(float);
    const size_t qk_bytes = static_cast<size_t>(kRows) * kQkDim * sizeof(float);
    const size_t v_bytes = static_cast<size_t>(kRows) * kValueDim * sizeof(uint16_t);
    const size_t ab_bytes = static_cast<size_t>(kRows) * kNumVHeads * sizeof(uint16_t);
    const size_t gdn_out_bytes =
        static_cast<size_t>(kRows) * kValueDim * sizeof(float);
    const size_t conv_weight_bytes = static_cast<size_t>(kConvDim) * 4u * sizeof(uint16_t);
    const size_t small_param_bytes = static_cast<size_t>(kNumVHeads) * sizeof(uint16_t);
    const size_t conv_state_bytes =
        static_cast<size_t>(kMaxSequences) * kConvSlotStride * sizeof(uint16_t);
    const size_t rec_state_bytes =
        static_cast<size_t>(kMaxSequences) * kRecurrentSlotStride * sizeof(float);

    void* conv_in = nullptr;
    void* conv_weight = nullptr;
    void* dt_bias = nullptr;
    void* a_log = nullptr;
    void* q_dev = nullptr;
    void* k_dev = nullptr;
    void* v_dev = nullptr;
    void* a_dev = nullptr;
    void* b_dev = nullptr;
    void* conv_out_host = nullptr;
    void* conv_out_mcu = nullptr;
    void* gdn_out_host = nullptr;
    void* gdn_out_mcu = nullptr;
    void* conv_state_host = nullptr;
    void* conv_state_mcu = nullptr;
    void* rec_state_host = nullptr;
    void* rec_state_mcu = nullptr;
    if (hipMalloc(&conv_in, conv_in_bytes) != hipSuccess ||
        hipMalloc(&conv_weight, conv_weight_bytes) != hipSuccess ||
        hipMalloc(&dt_bias, small_param_bytes) != hipSuccess ||
        hipMalloc(&a_log, small_param_bytes) != hipSuccess ||
        hipMalloc(&q_dev, qk_bytes) != hipSuccess ||
        hipMalloc(&k_dev, qk_bytes) != hipSuccess ||
        hipMalloc(&v_dev, v_bytes) != hipSuccess ||
        hipMalloc(&a_dev, ab_bytes) != hipSuccess ||
        hipMalloc(&b_dev, ab_bytes) != hipSuccess ||
        hipMalloc(&conv_out_host, conv_out_bytes) != hipSuccess ||
        hipMalloc(&conv_out_mcu, conv_out_bytes) != hipSuccess ||
        hipMalloc(&gdn_out_host, gdn_out_bytes) != hipSuccess ||
        hipMalloc(&gdn_out_mcu, gdn_out_bytes) != hipSuccess ||
        hipMalloc(&conv_state_host, conv_state_bytes) != hipSuccess ||
        hipMalloc(&conv_state_mcu, conv_state_bytes) != hipSuccess ||
        hipMalloc(&rec_state_host, rec_state_bytes) != hipSuccess ||
        hipMalloc(&rec_state_mcu, rec_state_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }

    {
        std::vector<uint16_t> h_conv_in(static_cast<size_t>(kRows) * kConvDim);
        for (size_t i = 0u; i < h_conv_in.size(); ++i)
            h_conv_in[i] = static_cast<uint16_t>(0x3C00u + ((i * 2654435761u) % 900u));
        std::vector<uint16_t> h_w(static_cast<size_t>(kConvDim) * 4u);
        for (size_t i = 0u; i < h_w.size(); ++i)
            h_w[i] = static_cast<uint16_t>(0x3B00u + ((i * 40503u) % 800u));
        std::vector<uint16_t> h_dt(kNumVHeads, 0x3A00u);
        std::vector<uint16_t> h_al(kNumVHeads, 0x3D00u);
        (void)hipMemcpy(conv_in, h_conv_in.data(), conv_in_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(conv_weight, h_w.data(), conv_weight_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(dt_bias, h_dt.data(), small_param_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(a_log, h_al.data(), small_param_bytes,
                        hipMemcpyHostToDevice);

        std::vector<float> h_q(static_cast<size_t>(kRows) * kQkDim);
        for (size_t i = 0u; i < h_q.size(); ++i)
            h_q[i] = static_cast<float>((i % 97u) - 48u) * 0.01f;
        (void)hipMemcpy(q_dev, h_q.data(), qk_bytes, hipMemcpyHostToDevice);
        for (size_t i = 0u; i < h_q.size(); ++i)
            h_q[i] = static_cast<float>((i % 89u) - 44u) * 0.02f;
        (void)hipMemcpy(k_dev, h_q.data(), qk_bytes, hipMemcpyHostToDevice);

        std::vector<uint16_t> h_v(static_cast<size_t>(kRows) * kValueDim);
        for (size_t i = 0u; i < h_v.size(); ++i)
            h_v[i] = static_cast<uint16_t>(0x3C00u + ((i * 2246822519u) % 700u));
        (void)hipMemcpy(v_dev, h_v.data(), v_bytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> h_a(static_cast<size_t>(kRows) * kNumVHeads);
        for (size_t i = 0u; i < h_a.size(); ++i)
            h_a[i] = static_cast<uint16_t>(0x3C00u + ((i * 3266489917u) % 400u));
        (void)hipMemcpy(a_dev, h_a.data(), ab_bytes, hipMemcpyHostToDevice);
        for (size_t i = 0u; i < h_a.size(); ++i)
            h_a[i] = static_cast<uint16_t>(0x3D00u + ((i * 668265263u) % 300u));
        (void)hipMemcpy(b_dev, h_a.data(), ab_bytes, hipMemcpyHostToDevice);

        std::vector<uint16_t> h_cs(static_cast<size_t>(kMaxSequences) * kConvSlotStride);
        for (size_t i = 0u; i < h_cs.size(); ++i)
            h_cs[i] = static_cast<uint16_t>(0x3900u + ((i * 2654435761u) % 600u));
        for (size_t i = 3u * kConvSlotStride; i < 4u * kConvSlotStride; ++i)
            h_cs[i] = 0u;
        (void)hipMemcpy(conv_state_host, h_cs.data(), conv_state_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(conv_state_mcu, h_cs.data(), conv_state_bytes,
                        hipMemcpyHostToDevice);
        std::vector<float> h_rs(static_cast<size_t>(kMaxSequences) * kRecurrentSlotStride);
        for (size_t i = 0u; i < h_rs.size(); ++i)
            h_rs[i] = static_cast<float>((i % 53u) - 26u) * 0.001f;
        for (size_t i = 3u * kRecurrentSlotStride; i < 4u * kRecurrentSlotStride; ++i)
            h_rs[i] = 0.0f;
        (void)hipMemcpy(rec_state_host, h_rs.data(), rec_state_bytes,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(rec_state_mcu, h_rs.data(), rec_state_bytes,
                        hipMemcpyHostToDevice);
    }
    (void)hipMemset(conv_out_host, 0xAB, conv_out_bytes);
    (void)hipMemset(conv_out_mcu, 0xAB, conv_out_bytes);
    (void)hipMemset(gdn_out_host, 0xAB, gdn_out_bytes);
    (void)hipMemset(gdn_out_mcu, 0xAB, gdn_out_bytes);

    std::vector<rt::DeviceRequestDescriptor> request_seed;
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
        request_seed = {d0, d1, dv, dp};
    }
    void* requests_host = nullptr;
    void* requests_dev = nullptr;
    if (!map_alloc(request_seed.size() * sizeof(rt::DeviceRequestDescriptor),
                   &requests_host, &requests_dev)) {
        std::printf("FAIL: request map alloc\n");
        return 1;
    }
    std::memcpy(requests_host, request_seed.data(),
                request_seed.size() * sizeof(rt::DeviceRequestDescriptor));

    rt::DeviceBatchContext bc{};
    bc.actual_rows = kRows;
    bc.num_requests = static_cast<uint32_t>(request_seed.size());
    bc.num_decode_requests = 2u;
    bc.num_verify_requests = 1u;
    bc.num_prefill_requests = 1u;
    bc.requests =
        static_cast<const rt::DeviceRequestDescriptor*>(requests_dev);

    void* ctx_host_mem = nullptr;
    void* ctx_dev_mem = nullptr;
    if (!map_alloc(sizeof(rt::DeviceBatchContext), &ctx_host_mem, &ctx_dev_mem)) {
        std::printf("FAIL: context map alloc\n");
        return 1;
    }
    std::memcpy(ctx_host_mem, &bc, sizeof(bc));
    rt::DeviceBatchContext* ctx_dev =
        static_cast<rt::DeviceBatchContext*>(ctx_dev_mem);

    rt::StaticParameterSlot params[3];
    params[0].device_ptr = conv_weight;
    params[0].elements = static_cast<uint64_t>(kConvDim) * 4u;
    params[0].dtype = rt::ValueDType::BF16;
    params[1].device_ptr = dt_bias;
    params[1].elements = kNumVHeads;
    params[1].dtype = rt::ValueDType::BF16;
    params[2].device_ptr = a_log;
    params[2].elements = kNumVHeads;
    params[2].dtype = rt::ValueDType::BF16;

    const void* ext_inputs[6] = {conv_in, q_dev, k_dev, v_dev, a_dev, b_dev};
    void* ext_out_host[2] = {conv_out_host, gdn_out_host};
    void* ext_out_mcu[2] = {conv_out_mcu, gdn_out_mcu};

    kernel::ModelDispatchStateView ms_host{};
    ms_host.gdn = make_gdn_view(conv_state_host, rec_state_host);
    kernel::ModelDispatchStateView ms_mcu{};
    ms_mcu.gdn = make_gdn_view(conv_state_mcu, rec_state_mcu);

    q35::HostExecutionContext ctx_host{};
    ctx_host.batch_context = ctx_dev;
    ctx_host.device_requests = bc.requests;
    ctx_host.external_inputs = ext_inputs;
    ctx_host.external_input_count = 6u;
    ctx_host.external_outputs = ext_out_host;
    ctx_host.external_output_count = 2u;
    ctx_host.host_parameters = params;
    ctx_host.host_parameter_count = 3u;
    ctx_host.actual_rows = kRows;
    ctx_host.actual_outputs = 2u;
    ctx_host.model_state = ms_host;

    q35::HostExecutionContext ctx_mcu = ctx_host;
    ctx_mcu.external_outputs = ext_out_mcu;
    ctx_mcu.model_state = ms_mcu;

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }
    const auto r1 = q35::try_launch_gdn_conv1d(program, 0u, ctx_host, stream);
    const auto r2 = q35::try_launch_gdn_recurrence(program, 1u, ctx_host, stream);
    if (!(r1.ok() && r1.value().kind == q35::OptimizedLaunchKind::Launched) ||
        !(r2.ok() && r2.value().kind == q35::OptimizedLaunchKind::Launched) ||
        hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: host gdn chain r1=%d r2=%d\n", static_cast<int>(r1.ok()),
                    static_cast<int>(r2.ok()));
        if (!r1.ok()) std::printf("  conv: %s\n", r1.status().message().c_str());
        if (!r2.ok()) std::printf("  rec: %s\n", r2.status().message().c_str());
        return 1;
    }

    q35::McuPlanCompileOptions opts{};
    opts.dispatch_begin = 0u;
    opts.dispatch_end = 2u;
    opts.marker_grid = 1u;
    opts.marker_workgroup = 64u;
    q35::McuCompiledPlan plan;
    const ps::Status plan_status = q35::compile_mcu_plan(program, ctx_mcu, opts, plan);
    if (!plan_status.ok()) {
        std::printf("FAIL: plan: %s\n", plan_status.message().c_str());
        return 1;
    }
    check(plan.gdn_conv1d.size() == 1u, "gdn conv1d emitted");
    check(plan.gdn_recurrence.size() == 1u, "gdn recurrence emitted");
    check(plan.gdn_reset.size() == 1u, "gdn reset emitted");

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
    check(runtime.row_global_patch_count() == 6u,
          "runtime derives row-global patches for gdn conv and recurrence");

    uint32_t* row_seq_dev = nullptr;
    uint32_t* row_pos_dev = nullptr;
    uint64_t* ready_dev = nullptr;
    uint64_t* plan_epoch_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&row_seq_dev),
                  kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&row_pos_dev),
                  kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&ready_dev), sizeof(uint64_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&plan_epoch_dev), sizeof(uint64_t)) !=
            hipSuccess) {
        std::printf("FAIL: context alloc\n");
        return 1;
    }
    (void)hipMemset(ready_dev, 0, sizeof(uint64_t));
    (void)hipMemset(plan_epoch_dev, 0, sizeof(uint64_t));
    (void)hipMemset(row_seq_dev, 0, kRows * sizeof(uint32_t));
    (void)hipMemset(row_pos_dev, 0, kRows * sizeof(uint32_t));

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
              .configure_batch_plan(ctx_dev,
                                    static_cast<rt::DeviceRequestDescriptor*>(
                                        requests_dev),
                                    row_seq_dev, row_pos_dev,
                                    static_cast<uint32_t>(request_seed.size()),
                                    kRows, 16u, plan_epoch_dev)
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_execution(runtime.execution_state(),
                                   runtime.plan_variant_count(), ready_dev, nullptr,
                                   0u, nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent
              .configure_execution_patches(runtime.row_global_patches(),
                                           runtime.row_global_patch_count())
              .ok(),
          "configure execution patches");
    const uint64_t ready = 1u;
    (void)hipMemcpy(ready_dev, &ready, sizeof(ready), hipMemcpyHostToDevice);
    check(persistent.start().ok(), "persistent start");
    check(wait_for(persistent, 1u, 15000u), "gdn loop completed");
    check(persistent.batches_failed() == 0u, "no fault");

    check(persistent.request_stop().ok(), "persistent stop requested");
    check(persistent.wait_stopped(5000u).ok(), "persistent stopped");
    check(persistent.shutdown().ok(), "persistent shutdown");

    std::vector<uint8_t> ch(conv_out_bytes, 0u);
    std::vector<uint8_t> cm(conv_out_bytes, 0u);
    std::vector<uint8_t> gh(gdn_out_bytes, 0u);
    std::vector<uint8_t> gm(gdn_out_bytes, 0u);
    (void)hipMemcpy(ch.data(), conv_out_host, conv_out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(cm.data(), conv_out_mcu, conv_out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(gh.data(), gdn_out_host, gdn_out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(gm.data(), gdn_out_mcu, gdn_out_bytes, hipMemcpyDeviceToHost);
    size_t first = 0u;
    size_t diffs = 0u;
    check(bytes_equal(ch, cm, &first, &diffs),
          "gdn conv1d output host vs persistent mcu byte exact");
    if (diffs != 0u) std::printf("  conv diff count=%zu first=%zu\n", diffs, first);
    check(bytes_equal(gh, gm, &first, &diffs),
          "gdn recurrence output host vs persistent mcu byte exact");
    if (diffs != 0u) std::printf("  rec diff count=%zu first=%zu\n", diffs, first);
    std::printf("gdn chain: dispatches=%llu plans=%llu\n",
                static_cast<unsigned long long>(
                    runtime.execution_state()->dispatches_committed),
                static_cast<unsigned long long>(
                    runtime.execution_state()->plans_started));

    check(runtime.shutdown().ok(), "runtime shutdown");

    (void)hipFree(plan_epoch_dev);
    (void)hipFree(ready_dev);
    (void)hipFree(row_pos_dev);
    (void)hipFree(row_seq_dev);
    check(hipFree(ctx_host_mem) == hipSuccess, "context map free");
    check(hipFree(requests_host) == hipSuccess, "request map free");
    check(hipStreamDestroy(persistent_stream) == hipSuccess,
          "persistent stream destroy");
    (void)hipStreamDestroy(stream);
    (void)hipFree(rec_state_mcu);
    (void)hipFree(rec_state_host);
    (void)hipFree(conv_state_mcu);
    (void)hipFree(conv_state_host);
    (void)hipFree(gdn_out_mcu);
    (void)hipFree(gdn_out_host);
    (void)hipFree(conv_out_mcu);
    (void)hipFree(conv_out_host);
    (void)hipFree(b_dev);
    (void)hipFree(a_dev);
    (void)hipFree(v_dev);
    (void)hipFree(k_dev);
    (void)hipFree(q_dev);
    (void)hipFree(a_log);
    (void)hipFree(dt_bias);
    (void)hipFree(conv_weight);
    (void)hipFree(conv_in);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_gdn_chain_bridge: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_gdn_chain_bridge: PASS\n");
    return 0;
}
