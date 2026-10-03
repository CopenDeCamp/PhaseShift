#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/staging_layout.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;
namespace q35rt = ::ps::qwen35::runtime;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

std::uint16_t bf16_rne_bits(float f) {
    return ps::f32_to_bf16_rne(f);
}

float bf16_to_f32(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float bf16_ulp(float v) {
    if (!std::isfinite(v) || v == 0.f) return 0.f;
    const std::uint16_t b = bf16_rne_bits(v);
    const int e = static_cast<int>(b >> 7) - 127;
    return std::ldexp(1.0f, e - 7);
}

struct Allocs {
    gpu::GpuArena* arena = nullptr;
    bool ok = true;

    gpu::Tensor bytes(std::size_t n_bytes, const char* name) {
        auto view = arena->allocate_aligned(n_bytes == 0 ? 1 : n_bytes, 128);
        if (!view.ok()) {
            std::printf("FAIL: alloc %s: %s\n", name, view.status().message().c_str());
            ok = false;
            return gpu::Tensor{};
        }
        auto t = gpu::Tensor::create<ps::bf16_t>(view.release(), {n_bytes / 2}, {1});
        if (!t.ok()) {
            std::printf("FAIL: tensor %s: %s\n", name, t.status().message().c_str());
            ok = false;
            return gpu::Tensor{};
        }
        return t.release();
    }
};

void fill_bf16(gpu::Tensor& t, const std::vector<float>& values, hipStream_t stream) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = bf16_rne_bits(values[i]);
    if (hipMemcpyAsync(t.data<void>(), bits.data(), bits.size() * sizeof(std::uint16_t),
                       hipMemcpyHostToDevice, stream) != hipSuccess) {
        std::printf("FAIL: fill %zu values\n", values.size());
        std::exit(1);
    }
}

std::vector<float> read_value(const rt::Program& prog, rt::ValueId id,
                              q35rt::HostExecutionContext& ctx) {
    auto r = q35rt::resolve_host_value(prog, id, ctx);
    if (!r.ok()) {
        std::printf("FAIL: resolve value %u: %s\n", id.id, r.status().message().c_str());
        return {};
    }
    const auto& v = r.value();
    const std::size_t elem = v.dtype == rt::ValueDType::BF16 ? 2u : 4u;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(v.feature_count) * elem);
    if (raw.size() > 0 &&
        hipMemcpy(raw.data(), v.ptr, raw.size(), hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: d2h value %u\n", id.id);
        return {};
    }
    std::vector<float> out(v.feature_count);
    for (std::size_t i = 0; i < v.feature_count; ++i) {
        if (v.dtype == rt::ValueDType::BF16) {
            const std::uint16_t b =
                static_cast<std::uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
            out[i] = bf16_to_f32(b);
        } else {
            float f;
            std::memcpy(&f, &raw[4 * i], sizeof(f));
            out[i] = f;
        }
    }
    return out;
}

bool close_abs(float act, double ref, double tol, const std::string& msg) {
    if (std::fabs(static_cast<double>(act) - ref) > tol) {
        check(false, msg + " act=" + std::to_string(act) + " ref=" + std::to_string(ref));
        return false;
    }
    return true;
}

} // namespace

int main() {
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    auto ar = gpu::GpuArena::create(0, 16ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("SKIP: no GPU available (%s)\n", ar.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = ar.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream create\n");
        arena.shutdown();
        return 1;
    }
    Allocs allocs{&arena};

    const std::vector<float> x_norm = {
        1.0f, -1.0f, 2.0f, -2.0f,
        0.25f, 0.25f, 0.25f, 0.25f,
        0.5f, 0.5f, 0.5f, 0.5f,
        0.125f, 0.125f, 0.125f, 0.125f};
    const std::vector<float> x_split = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f};
    const std::vector<float> a_gate = {1.5f, -0.5f, 0.25f, -2.0f, 3.0f, 0.125f, -0.75f, 2.0f};
    const std::vector<float> g_gate = {-2.0f, 0.0f, 2.0f, -1.0f, 1.0f, -0.5f, 0.5f, 3.0f};
    const std::vector<float> c1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const std::vector<float> c2 = {100.f, 101.f, 102.f, 103.f, 104.f, 105.f, 106.f, 107.f};
    const std::vector<float> r1 = {0.25f, -0.5f, 1.5f, -2.5f, 0.125f, -3.5f, 4.25f, -0.0625f};
    const std::vector<float> r2 = {-0.125f, 0.25f, -2.25f, 1.25f, 0.0625f, -0.75f, 3.75f, -1.875f};
    std::vector<float> w_full(16);
    for (std::size_t i = 0; i < 16; ++i) w_full[i] = 0.25f + 0.0625f * static_cast<float>(i % 4);
    const std::vector<float> w_group = {0.25f, 0.3125f, 0.375f, 0.4375f};

    gpu::Tensor t_x_norm = allocs.bytes(32, "x_norm");
    gpu::Tensor t_x_split = allocs.bytes(32, "x_split");
    gpu::Tensor t_a_gate = allocs.bytes(16, "a_gate");
    gpu::Tensor t_g_gate = allocs.bytes(16, "g_gate");
    gpu::Tensor t_c1 = allocs.bytes(16, "c1");
    gpu::Tensor t_c2 = allocs.bytes(16, "c2");
    gpu::Tensor t_r1 = allocs.bytes(16, "r1");
    gpu::Tensor t_r2 = allocs.bytes(16, "r2");
    gpu::Tensor t_w_full = allocs.bytes(32, "w_full");
    gpu::Tensor t_w_group = allocs.bytes(8, "w_group");
    gpu::Tensor t_y1 = allocs.bytes(64, "y1");
    gpu::Tensor t_y2 = allocs.bytes(64, "y2");
    gpu::Tensor t_y3 = allocs.bytes(64, "y3");
    gpu::Tensor t_a4 = allocs.bytes(16, "a4");
    gpu::Tensor t_b4 = allocs.bytes(16, "b4");
    gpu::Tensor t_s5 = allocs.bytes(32, "s5");
    gpu::Tensor t_m6 = allocs.bytes(16, "m6");
    gpu::Tensor t_c7 = allocs.bytes(32, "c7");
    gpu::Tensor t_r8 = allocs.bytes(16, "r8");
    if (!allocs.ok) {
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }
    fill_bf16(t_x_norm, x_norm, stream);
    fill_bf16(t_x_split, x_split, stream);
    fill_bf16(t_a_gate, a_gate, stream);
    fill_bf16(t_g_gate, g_gate, stream);
    fill_bf16(t_c1, c1, stream);
    fill_bf16(t_c2, c2, stream);
    fill_bf16(t_r1, r1, stream);
    fill_bf16(t_r2, r2, stream);
    fill_bf16(t_w_full, w_full, stream);
    fill_bf16(t_w_group, w_group, stream);

    rt::StaticParameterSlot param_slots[2] = {
        {t_w_full.data<void>(), 16, rt::ValueDType::BF16},
        {t_w_group.data<void>(), 4, rt::ValueDType::BF16},
    };

    rt::PrimitiveGraph g;
    auto val = [&](std::uint32_t features, rt::ValueDType dt) {
        return g.alloc_value(features, dt, rt::ValueRowDomain::TOKEN_ROWS);
    };
    const rt::ValueId v_x_norm = val(16, rt::ValueDType::BF16);
    const rt::ValueId v_x_split = val(16, rt::ValueDType::BF16);
    const rt::ValueId v_a_gate = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_g_gate = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_c1 = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_c2 = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_r1 = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_r2 = val(8, rt::ValueDType::BF16);
    g.external_inputs = {v_x_norm, v_x_split, v_a_gate, v_g_gate, v_c1, v_c2, v_r1, v_r2};
    const rt::ValueId v_y1 = val(16, rt::ValueDType::F32);
    const rt::ValueId v_y2 = val(16, rt::ValueDType::F32);
    const rt::ValueId v_y3 = val(16, rt::ValueDType::F32);
    const rt::ValueId v_a4 = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_b4 = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_s5 = val(8, rt::ValueDType::F32);
    const rt::ValueId v_m6 = val(8, rt::ValueDType::BF16);
    const rt::ValueId v_c7 = val(16, rt::ValueDType::BF16);
    const rt::ValueId v_r8 = val(8, rt::ValueDType::BF16);
    g.external_outputs = {v_y1, v_y2, v_y3, v_a4, v_b4, v_s5, v_m6, v_c7, v_r8};

    auto add_node = [&](rt::PrimitiveNode node, std::vector<rt::ValueId> ins,
                        std::vector<rt::ValueId> outs) {
        auto& n = g.add_node(std::move(node));
        n.inputs = std::move(ins);
        n.outputs = std::move(outs);
    };
    {
        rt::RmsNormNode n;
        n.shape = rt::RowwiseShapeKey{16};
        n.eps = 1e-6f;
        n.group_size = 4;
        n.parameter_index = 0;
        n.weight_mode = rt::RmsNormWeightMode::ONE_PLUS;
        add_node(rt::PrimitiveNode{n}, {v_x_norm}, {v_y1});
    }
    {
        rt::RmsNormNode n;
        n.shape = rt::RowwiseShapeKey{16};
        n.eps = 1e-6f;
        n.group_size = 4;
        n.parameter_index = 1;
        n.weight_mode = rt::RmsNormWeightMode::ONE_PLUS;
        add_node(rt::PrimitiveNode{n}, {v_x_norm}, {v_y2});
    }
    {
        rt::RmsNormNode n;
        n.shape = rt::RowwiseShapeKey{16};
        n.eps = 1e-6f;
        n.group_size = 4;
        n.parameter_index = 0;
        n.weight_mode = rt::RmsNormWeightMode::DIRECT;
        add_node(rt::PrimitiveNode{n}, {v_x_norm}, {v_y3});
    }
    {
        rt::SplitNode n;
        n.input_shape = rt::RowwiseShapeKey{16};
        n.output_shape_a = rt::RowwiseShapeKey{8};
        n.output_shape_b = rt::RowwiseShapeKey{8};
        n.split_dim = 1;
        n.layout = rt::SplitLayout::INTERLEAVED_HEADS;
        n.head_dim = 4;
        add_node(rt::PrimitiveNode{n}, {v_x_split}, {v_a4, v_b4});
    }
    {
        rt::SigmoidNode n;
        n.shape = rt::RowwiseShapeKey{8};
        add_node(rt::PrimitiveNode{n}, {v_g_gate}, {v_s5});
    }
    {
        rt::MulNode n;
        n.shape = rt::RowwiseShapeKey{8};
        add_node(rt::PrimitiveNode{n}, {v_a_gate, v_s5}, {v_m6});
    }
    {
        rt::ConcatNode n;
        n.shape = rt::RowwiseShapeKey{16};
        n.input_shape_a = rt::RowwiseShapeKey{8};
        n.input_shape_b = rt::RowwiseShapeKey{8};
        add_node(rt::PrimitiveNode{n}, {v_c1, v_c2}, {v_c7});
    }
    {
        rt::ResidualAddNode n;
        n.shape = rt::RowwiseShapeKey{8};
        add_node(rt::PrimitiveNode{n}, {v_r1, v_r2}, {v_r8});
    }

    rt::WeightTableView wv{nullptr, 0};
    rt::StaticParameterTableView pv{param_slots, 2};
    rt::ProgramBuildOptions opts;
    opts.max_token_rows = 16;
    opts.max_output_rows = 1;
    auto set_res = rt::build_program_set(g, wv, pv, rt::ExecutionClass::DECODE, opts);
    if (!set_res.ok()) {
        std::printf("FAIL: build_program_set: %s\n", set_res.status().message().c_str());
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }
    auto set = set_res.release();
    const auto* prog = set.resolve(rt::RowBucket::R16);
    if (prog == nullptr) {
        std::printf("FAIL: program R16 not resolved\n");
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }

    auto bctx = rt::create_device_batch_context(arena, 1u, 1u);
    if (!bctx.ok()) {
        std::printf("FAIL: batch context: %s\n", bctx.status().message().c_str());
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }
    auto bstorage = bctx.release();
    auto t_rope = allocs.bytes(4, "rope_positions");
    if (!allocs.ok) {
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }
    {
        rt::DeviceRequestDescriptor desc{};
        desc.row_begin = 0;
        desc.row_count = 1;
        desc.prefix_length = 0;
        desc.sequence_length = 1;
        desc.output_index = 0;
        desc.execution_class = rt::ExecutionClass::DECODE;
        if (hipMemcpyAsync(bstorage.requests.data<rt::DeviceRequestDescriptor>(), &desc,
                           sizeof(desc), hipMemcpyHostToDevice, stream) != hipSuccess) {
            std::printf("FAIL: request desc copy\n");
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        const std::uint32_t zero = 0;
        if (hipMemcpyAsync(bstorage.row_positions.data<std::uint32_t>(), &zero,
                           sizeof(zero), hipMemcpyHostToDevice, stream) != hipSuccess ||
            hipMemcpyAsync(t_rope.data<void>(), &zero, sizeof(zero),
                           hipMemcpyHostToDevice, stream) != hipSuccess) {
            std::printf("FAIL: position copy\n");
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        rt::DeviceBatchContext hc{};
        hc.actual_rows = 1;
        hc.num_requests = 1;
        hc.num_outputs = 1;
        hc.num_decode_requests = 1;
        hc.token_ids = bstorage.token_ids.data<int32_t>();
        hc.requests = bstorage.requests.data<rt::DeviceRequestDescriptor>();
        hc.row_sequence_slots = bstorage.row_sequence_slots.data<std::uint32_t>();
        hc.row_positions = bstorage.row_positions.data<std::uint32_t>();
        hc.rope_positions = t_rope.data<std::uint32_t>();
        hc.output_rows = bstorage.output_rows.data<std::uint32_t>();
        hc.output_sampling_params =
            bstorage.output_sampling_params
                .data<::ps::runtime::DeviceSamplingParams>();
        hc.device_ptr = bstorage.device_struct.data<rt::DeviceBatchContext>();
        if (hipMemcpyAsync(bstorage.device_struct.data<rt::DeviceBatchContext>(), &hc,
                           sizeof(hc), hipMemcpyHostToDevice, stream) != hipSuccess) {
            std::printf("FAIL: batch context copy\n");
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
    }
    {
        auto pstat = rt::launch_prepare_batch_descriptor(
            bstorage.device_struct.data<rt::DeviceBatchContext>(), 1u, stream);
        if (!pstat.ok()) {
            std::printf("FAIL: prepare batch descriptor: %s\n", pstat.message().c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
    }

    const std::size_t ws_bytes =
        (static_cast<std::size_t>(prog->workspace.total_bytes) + 255ull) & ~255ull;
    constexpr std::size_t kScratchFloats = 8192u;
    const std::size_t scratch_bytes = kScratchFloats * sizeof(float);
    std::uint8_t* ws = nullptr;
    std::uint32_t* error_word = nullptr;
    void* staging_dev = nullptr;
    void* staging_host = nullptr;
    const std::uint64_t staging_bytes = rt::max_host_staging_bytes(*prog);
    if (hipMalloc(reinterpret_cast<void**>(&ws), ws_bytes + scratch_bytes) != hipSuccess ||
        hipMemsetAsync(ws, 0, ws_bytes + scratch_bytes, stream) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&error_word), sizeof(std::uint32_t)) != hipSuccess ||
        hipMemsetAsync(error_word, 0, sizeof(std::uint32_t), stream) != hipSuccess ||
        hipMalloc(&staging_dev, staging_bytes) != hipSuccess ||
        hipHostMalloc(&staging_host, staging_bytes) != hipSuccess) {
        std::printf("FAIL: scratch allocation\n");
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }

    rt::StaticParameterSlot* param_dev = nullptr;
    {
        auto pt = q35rt::upload_host_parameter_table(pv, &param_dev, stream);
        if (!pt.ok()) {
            std::printf("FAIL: upload parameter table: %s\n", pt.message().c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
    }

    const void* inputs[8] = {
        t_x_norm.data<void>(), t_x_split.data<void>(), t_a_gate.data<void>(),
        t_g_gate.data<void>(), t_c1.data<void>(), t_c2.data<void>(),
        t_r1.data<void>(), t_r2.data<void>()};
    void* outputs[9] = {
        t_y1.data<void>(), t_y2.data<void>(), t_y3.data<void>(), t_a4.data<void>(),
        t_b4.data<void>(), t_s5.data<void>(), t_m6.data<void>(), t_c7.data<void>(),
        t_r8.data<void>()};

    q35rt::HostExecutionContext ctx{};
    ctx.batch_context = bstorage.device_struct.data<rt::DeviceBatchContext>();
    ctx.parameters = param_dev;
    ctx.parameter_count = prog->parameter_slot_count;
    ctx.host_parameters = param_slots;
    ctx.host_parameter_count = 2;
    ctx.row_positions = bstorage.row_positions.data<std::uint32_t>();
    ctx.rope_positions = t_rope.data<std::uint32_t>();
    ctx.row_sequence_slots = bstorage.row_sequence_slots.data<std::uint32_t>();
    ctx.external_inputs = inputs;
    ctx.external_input_count = 8;
    ctx.external_outputs = outputs;
    ctx.external_output_count = 9;
    ctx.workspace = ws;
    ctx.workspace_bytes = ws_bytes + scratch_bytes;
    ctx.scratch = reinterpret_cast<float*>(ws + ws_bytes);
    ctx.scratch_floats = static_cast<std::uint32_t>(kScratchFloats);
    ctx.error_word = error_word;
    ctx.bucket_m = rt::row_bucket_limit(rt::RowBucket::R16);
    ctx.staging = staging_dev;
    ctx.staging_bytes = staging_bytes;
    ctx.host_staging = staging_host;
    ctx.actual_rows = 1;
    ctx.actual_outputs = 1;

    {
        auto est = q35rt::execute_program(*prog, ctx, stream);
        if (!est.ok()) {
            std::printf("FAIL: execute_program: %s\n", est.message().c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: stream sync\n");
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        std::uint32_t ew = 1;
        ps::gpu::discard_cleanup_result(
            hipMemcpy(&ew, error_word, sizeof(ew), hipMemcpyDeviceToHost));
        check(ew == 0u, "device error word == 0");
    }

    const std::vector<float> y1 = read_value(*prog, v_y1, ctx);
    const std::vector<float> y2 = read_value(*prog, v_y2, ctx);
    const std::vector<float> y3 = read_value(*prog, v_y3, ctx);
    check(!y1.empty() && !y2.empty() && !y3.empty(), "norm outputs read");

    {
        constexpr double kEpsRef = 1e-6;
        const double x[16] = {
            1.0, -1.0, 2.0, -2.0,
            0.25, 0.25, 0.25, 0.25,
            0.5, 0.5, 0.5, 0.5,
            0.125, 0.125, 0.125, 0.125};
        double inv[4] = {0.0, 0.0, 0.0, 0.0};
        for (int gi = 0; gi < 4; ++gi) {
            double sum = 0.0;
            for (int h = 0; h < 4; ++h) sum += x[gi * 4 + h] * x[gi * 4 + h];
            inv[gi] = 1.0 / std::sqrt(sum / 4.0 + kEpsRef);
        }
        check(inv[0] != inv[1] && inv[1] != inv[2] && inv[2] != inv[3],
              "per-group inv differs across groups (test sensitivity)");
        for (int i = 0; i < 16; ++i) {
            const double wf = 0.25 + 0.0625 * (i % 4);
            const double one_plus = x[i] * inv[i / 4] * (1.0 + wf);
            const double direct = x[i] * inv[i / 4] * wf;
            if (!y1.empty())
                close_abs(y1[i], one_plus, 1e-5, "y1[" + std::to_string(i) + "] ONE_PLUS per-row weight");
            if (!y2.empty())
                close_abs(y2[i], one_plus, 1e-5, "y2[" + std::to_string(i) + "] ONE_PLUS per-group weight");
            if (!y3.empty())
                close_abs(y3[i], direct, 1e-5, "y3[" + std::to_string(i) + "] DIRECT weight");
        }
        double max_diff = 0.0;
        if (!y1.empty() && !y3.empty())
            for (int i = 0; i < 16; ++i)
                max_diff = std::max(max_diff, std::fabs(static_cast<double>(y1[i]) - y3[i]));
        check(max_diff > 0.1, "ONE_PLUS result distinguishable from DIRECT (negative control)");
    }

    const std::vector<float> a4 = read_value(*prog, v_a4, ctx);
    const std::vector<float> b4 = read_value(*prog, v_b4, ctx);
    check(!a4.empty() && !b4.empty(), "split outputs read");
    {
        const float a_exp[8] = {1.f, 2.f, 3.f, 4.f, 9.f, 10.f, 11.f, 12.f};
        const float b_exp[8] = {5.f, 6.f, 7.f, 8.f, 13.f, 14.f, 15.f, 16.f};
        const float a_halves[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        bool a_ok = true;
        bool b_ok = true;
        bool differs_from_halves = false;
        if (a4.size() == 8 && b4.size() == 8) {
            for (int i = 0; i < 8; ++i) {
                if (a4[i] != a_exp[i]) a_ok = false;
                if (b4[i] != b_exp[i]) b_ok = false;
                if (a4[i] != a_halves[i]) differs_from_halves = true;
            }
        } else {
            a_ok = false;
            b_ok = false;
        }
        check(a_ok, "split a == per-head first half (INTERLEAVED_HEADS)");
        check(b_ok, "split b == per-head second half (INTERLEAVED_HEADS)");
        check(differs_from_halves, "split is not global halves (negative control)");
    }

    const std::vector<float> s5 = read_value(*prog, v_s5, ctx);
    const std::vector<float> m6 = read_value(*prog, v_m6, ctx);
    check(!s5.empty() && !m6.empty(), "gate outputs read");
    {
        const double g[8] = {-2.0, 0.0, 2.0, -1.0, 1.0, -0.5, 0.5, 3.0};
        const double a[8] = {1.5, -0.5, 0.25, -2.0, 3.0, 0.125, -0.75, 2.0};
        for (int i = 0; i < 8; ++i) {
            const double sig = 1.0 / (1.0 + std::exp(-g[i]));
            if (!s5.empty())
                close_abs(s5[i], sig, 1e-6, "sigmoid[" + std::to_string(i) + "]");
            if (!m6.empty()) {
                const double ref = a[i] * sig;
                const double tol =
                    static_cast<double>(bf16_ulp(m6[i])) + static_cast<double>(bf16_ulp(static_cast<float>(ref))) + 1e-9;
                check(std::fabs(static_cast<double>(m6[i]) - ref) <= tol,
                      "gate mul[" + std::to_string(i) + "] == attention_core * sigmoid(gate)");
            }
        }
    }

    const std::vector<float> c7 = read_value(*prog, v_c7, ctx);
    check(!c7.empty(), "concat output read");
    {
        bool ok_order = c7.size() == 16;
        if (ok_order) {
            for (int i = 0; i < 8; ++i) {
                if (c7[i] != c1[i] || c7[8 + i] != c2[i]) ok_order = false;
            }
        }
        check(ok_order, "concat == [embedding_norm, hidden_norm] order");
        check(c7.size() == 16 && c7[0] != c2[0], "concat is not reversed (negative control)");
    }

    const std::vector<float> r8 = read_value(*prog, v_r8, ctx);
    check(!r8.empty(), "residual output read");
    {
        if (r8.size() == 8) {
            for (int i = 0; i < 8; ++i) {
                const float ref = r1[i] + r2[i];
                const double tol = static_cast<double>(bf16_ulp(r8[i])) + 1e-9;
                check(std::fabs(static_cast<double>(r8[i]) - static_cast<double>(ref)) <= tol,
                      "residual[" + std::to_string(i) + "] == a + b");
            }
        }
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("QWEN35_MTP_PRIMITIVES: PASS\n");
    }

    ps::gpu::discard_cleanup_result(hipFree(ws));
    ps::gpu::discard_cleanup_result(hipFree(error_word));
    ps::gpu::discard_cleanup_result(hipFree(staging_dev));
    ps::gpu::discard_cleanup_result(hipHostFree(staging_host));
    ps::gpu::discard_cleanup_result(hipFree(param_dev));
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)rt::destroy_device_batch_context(bstorage);
    arena.shutdown();
    return g_fail == 0 ? 0 : 1;
}
