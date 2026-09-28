#include <phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/fp8_block128.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.h>
#include <phaseshift/quantization/fp8/block128.h>
#include <phaseshift/quantization/fp8/block128_native.h>
#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/mxfp4/mxfp4_native.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/program/device_program.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#define HIP_ASSERT(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        std::printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err)); \
        return false; \
    } \
} while (0)

namespace rt = ::ps::runtime;

namespace {

int g_pass = 0;
int g_fail = 0;

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui = 0;
    std::memcpy(&ui, &f, sizeof(float));
    return ps::bf16_t{static_cast<uint16_t>((ui + 0x8000u) >> 16)};
}

float bf16_to_f32(ps::bf16_t b) {
    const uint32_t ui = static_cast<uint32_t>(b.data) << 16;
    float f = 0.0f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

uint64_t align16(uint64_t v) { return (v + 15u) & ~15ull; }

uint64_t a8_scales_offset(uint32_t code_stride, uint32_t max_rows) {
    return align16(static_cast<uint64_t>(code_stride) * align16(max_rows));
}

struct Payload {
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t kp = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<float> dequant;
};

bool build_payload(bool fp8, const std::vector<float>& w, uint32_t n, uint32_t k, Payload& p) {
    p.n = n;
    p.k = k;
    if (fp8) {
        p.kp = static_cast<uint32_t>((k + 127u) & ~127u);
        ps::quantization::fp8::Fp8BlockCanonicalStore store;
        store.init(1, n, k, p.kp);
        store.quantize_matrix(0, w.data(), k);
        ps::quantization::fp8::Fp8Block128NativeHost native;
        if (!ps::quantization::fp8::preshuffle_fp8_block128_native(store.view(), native)) return false;
        p.codes = native.codes;
        p.scales.assign(native.scales.size() * 4u, 0u);
        std::memcpy(p.scales.data(), native.scales.data(), p.scales.size());
        p.dequant.assign(static_cast<std::size_t>(n) * p.kp, 0.0f);
        ps::quantization::fp8::dequantize_fp8_block128_canonical_matrix(
            store.view(), 0, p.dequant.data(), p.kp);
    } else {
        p.kp = static_cast<uint32_t>((k + 31u) & ~31u);
        ps::quantization::mxfp4::Mxfp4CanonicalStore store;
        store.init(n, k, p.kp);
        for (uint32_t r = 0; r < n; ++r)
            store.quantize_row(r, std::span<const float>(w.data() + static_cast<std::size_t>(r) * k, k));
        ps::quantization::mxfp4::Mxfp4NativeHost native;
        if (!ps::quantization::mxfp4::preshuffle_mxfp4_native(store.view(), native)) return false;
        p.codes = native.codes;
        p.scales = native.scales;
        p.dequant.assign(static_cast<std::size_t>(n) * p.kp, 0.0f);
        ps::quantization::mxfp4::dequantize_mxfp4_canonical(store.view(), p.dequant);
    }
    return true;
}

bool run_case(ps::gpu::GpuArena& arena, bool fp8, uint32_t rows, uint32_t n, uint32_t k,
              const char* tag) {
    std::mt19937 rng(20260923u + rows * 31u + n * 7u + k + (fp8 ? 1u : 2u));
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : w) v = wdist(rng);
    for (auto& v : x) v = xdist(rng);

    Payload p;
    if (!build_payload(fp8, w, n, k, p)) { std::printf("FAIL: %s payload\n", tag); return false; }
    const uint32_t kp = p.kp;
    const uint32_t code_stride = kp;
    const uint32_t scale_stride = 64u;
    const uint32_t max_rows = rows;

    std::vector<ps::bf16_t> hx(static_cast<std::size_t>(rows) * k);
    std::vector<float> xr(x.size());
    for (std::size_t i = 0; i < hx.size(); ++i) {
        hx[i] = f32_to_bf16(x[i]);
        xr[i] = bf16_to_f32(hx[i]);
    }
    std::vector<float> adec(static_cast<std::size_t>(rows) * kp, 0.0f);
    for (uint32_t m = 0; m < rows; ++m) {
        float peak = 0.0f;
        for (uint32_t i = 0; i < k; ++i)
            peak = std::max(peak, std::fabs(xr[static_cast<std::size_t>(m) * k + i]));
        const float scale = peak > 0.0f ? peak / 448.0f : 0.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (uint32_t i = 0; i < kp; ++i) {
            const float v = (i < k) ? xr[static_cast<std::size_t>(m) * k + i] : 0.0f;
            const uint8_t c = scale > 0.0f ? ps::quantization::fpx::e4m3_encode_u8(v * inv) : 0u;
            adec[static_cast<std::size_t>(m) * kp + i] =
                ps::quantization::fpx::e4m3_decode_f32(c) * scale;
        }
    }

    const uint64_t ws_bytes = a8_scales_offset(code_stride, max_rows) +
                              static_cast<uint64_t>(max_rows) * scale_stride + 64u;
    const std::size_t out_ref_bytes = static_cast<std::size_t>(rows) * n * 2u;
    const std::size_t out_opt_bytes = static_cast<std::size_t>(rows) * n * 4u;

    auto wc = arena.allocate_aligned(p.codes.size(), 16);
    auto ws = arena.allocate_aligned(p.scales.size(), 16);
    auto xa = arena.allocate_aligned(hx.size() * sizeof(ps::bf16_t), 16);
    auto ora = arena.allocate_aligned(out_ref_bytes, 16);
    auto ooa = arena.allocate_aligned(out_opt_bytes, 16);
    auto wsa = arena.allocate_aligned(ws_bytes, 256);
    auto sca = arena.allocate_aligned(64u * 2048u * sizeof(float), 16);
    auto era = arena.allocate_aligned(sizeof(uint32_t), 16);
    auto sla = arena.allocate_aligned(sizeof(rt::WeightSlot), 16);
    auto vala = arena.allocate_aligned(2 * sizeof(rt::DeviceValueBinding), 16);
    auto rnga = arena.allocate_aligned(sizeof(rt::DeviceWorkspaceRange), 16);
    auto ctxa = arena.allocate_aligned(sizeof(rt::DeviceBatchContext), 16);
    if (!wc.ok() || !ws.ok() || !xa.ok() || !ora.ok() || !ooa.ok() || !wsa.ok() || !sca.ok() ||
        !era.ok() || !sla.ok() || !vala.ok() || !rnga.ok() || !ctxa.ok()) {
        std::printf("FAIL: %s alloc\n", tag);
        return false;
    }
    auto* d_wc = static_cast<uint8_t*>(wc.release().data());
    auto* d_ws = static_cast<uint8_t*>(ws.release().data());
    auto* d_x = static_cast<ps::bf16_t*>(xa.release().data());
    auto* d_out_ref = static_cast<ps::bf16_t*>(ora.release().data());
    auto* d_out_opt = static_cast<float*>(ooa.release().data());
    auto* d_ws_ws = static_cast<uint8_t*>(wsa.release().data());
    auto* d_scratch = static_cast<float*>(sca.release().data());
    auto* d_err = static_cast<uint32_t*>(era.release().data());
    auto* d_slot = static_cast<rt::WeightSlot*>(sla.release().data());
    auto* d_values = static_cast<rt::DeviceValueBinding*>(vala.release().data());
    auto* d_ranges = static_cast<rt::DeviceWorkspaceRange*>(rnga.release().data());
    auto* d_ctx = static_cast<rt::DeviceBatchContext*>(ctxa.release().data());

    HIP_ASSERT(hipMemcpy(d_wc, p.codes.data(), p.codes.size(), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_ws, p.scales.data(), p.scales.size(), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemset(d_ws_ws, 0, ws_bytes));
    HIP_ASSERT(hipMemset(d_out_ref, 0, out_ref_bytes));
    HIP_ASSERT(hipMemset(d_out_opt, 0, out_opt_bytes));
    HIP_ASSERT(hipMemset(d_err, 0, sizeof(uint32_t)));

    rt::WeightSlot slot{};
    slot.rows = n;
    slot.cols = k;
    slot.k_padded = kp;
    slot.encoding = fp8 ? 5u : 6u;
    slot.compute_spec = fp8 ? 5u : 6u;
    slot.codes = d_wc;
    slot.scales = d_ws;
    slot.storage_scale_stride_bytes = fp8 ? (kp >> 7) * 4u : (kp >> 5);
    slot.weight_scale_group = fp8 ? 128u : 32u;
    slot.codes_row_stride_bytes = fp8 ? (kp >> 5) * 512u : (kp >> 5) * 256u;
    slot.preshuffled = true;
    HIP_ASSERT(hipMemcpy(d_slot, &slot, sizeof(slot), hipMemcpyHostToDevice));

    rt::DeviceValueBinding values[2]{};
    values[0].logical_value = 0;
    values[0].storage = rt::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = rt::ValueDType::BF16;
    values[0].row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    values[0].slot = 0;
    values[0].feature_count = k;
    values[0].row_stride = k;
    values[0].offset = reinterpret_cast<uint64_t>(d_x);
    values[0].bytes = hx.size() * sizeof(ps::bf16_t);
    values[1].logical_value = 1;
    values[1].storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    values[1].dtype = rt::ValueDType::BF16;
    values[1].row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    values[1].slot = 0;
    values[1].feature_count = n;
    values[1].row_stride = n;
    values[1].offset = reinterpret_cast<uint64_t>(d_out_ref);
    values[1].bytes = out_ref_bytes;
    HIP_ASSERT(hipMemcpy(d_values, values, sizeof(values), hipMemcpyHostToDevice));

    rt::DeviceWorkspaceRange range{};
    range.offset = 0;
    range.bytes = ws_bytes;
    HIP_ASSERT(hipMemcpy(d_ranges, &range, sizeof(range), hipMemcpyHostToDevice));

    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_outputs = 0;
    HIP_ASSERT(hipMemcpy(d_ctx, &ctx, sizeof(ctx), hipMemcpyHostToDevice));

    rt::DeviceProgramView prog{};
    prog.values = d_values;
    prog.value_count = 2;
    prog.weight_table = d_slot;
    prog.weight_slot_count = 1;
    prog.workspace_ranges = d_ranges;
    prog.workspace_range_count = 1;
    prog.workspace_base = reinterpret_cast<uint64_t>(d_ws_ws);

    ps::kernel::ModelDispatchStateView state{};

    rt::DispatchBinding aq{};
    aq.kernel_id = rt::KernelId::ACTIVATION_QUANTIZE_W4A8;
    aq.workspace_slot = 0;
    aq.group_size = 32;
    aq.activation_kp = kp;
    aq.activation_max_rows = max_rows;
    aq.activation_code_stride = code_stride;
    aq.activation_scale_stride = scale_stride;
    aq.input_slots[0] = 0;
    aq.input_count = 1;
    auto st = ps::kernel::launch_model_dispatch_correctness(
        prog, aq, d_ctx, state, d_scratch, d_err, 0);
    if (!st.ok()) { std::printf("FAIL: %s aq: %s\n", tag, st.message().c_str()); return false; }

    rt::DispatchBinding lin{};
    lin.kernel_id = fp8 ? rt::KernelId::LINEAR_FP8 : rt::KernelId::LINEAR_MXFP4;
    lin.workspace_slot = 0;
    lin.weight_index = 0;
    lin.compute_spec = fp8 ? 5u : 6u;
    lin.activation_kp = kp;
    lin.activation_max_rows = max_rows;
    lin.activation_code_stride = code_stride;
    lin.activation_scale_stride = scale_stride;
    lin.output_slots[0] = 1;
    lin.output_count = 1;
    st = ps::kernel::launch_model_dispatch_correctness(prog, lin, d_ctx, state, d_scratch, d_err, 0);
    if (!st.ok()) { std::printf("FAIL: %s linear: %s\n", tag, st.message().c_str()); return false; }
    HIP_ASSERT(hipStreamSynchronize(0));
    uint32_t err = 0;
    HIP_ASSERT(hipMemcpy(&err, d_err, sizeof(err), hipMemcpyDeviceToHost));
    if (err != 0u) { std::printf("FAIL: %s program status=%u\n", tag, err); return false; }

    const uint8_t* acodes = d_ws_ws;
    const float* ascales = reinterpret_cast<const float*>(
        d_ws_ws + a8_scales_offset(code_stride, max_rows));
    hipError_t hk;
    if (fp8) {
        hk = ps::kernel::launch_gemm_fp8_block128_w8a8_wmma(
            ps::kernel::Fp8Block128GemmConfig{}, d_wc, reinterpret_cast<const float*>(d_ws),
            reinterpret_cast<const int8_t*>(acodes), ascales, d_out_opt,
            ps::kernel::Fp8Block128GemmOutputDType::F32, rows, n, kp,
            (kp >> 7) * 4u, code_stride, scale_stride, n, 0);
    } else {
        hk = ps::kernel::launch_gemm_mxfp4_w4a8_wmma(
            ps::kernel::Mxfp4GemmConfig{}, d_wc, d_ws,
            reinterpret_cast<const int8_t*>(acodes), ascales, d_out_opt,
            ps::kernel::Mxfp4GemmOutputDType::F32, rows, n, kp,
            (kp >> 5), code_stride, scale_stride, n, 0);
    }
    if (hk != hipSuccess) { std::printf("FAIL: %s opt launch\n", tag); return false; }
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<ps::bf16_t> ref(static_cast<std::size_t>(rows) * n);
    std::vector<float> opt(static_cast<std::size_t>(rows) * n);
    HIP_ASSERT(hipMemcpy(ref.data(), d_out_ref, out_ref_bytes, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(opt.data(), d_out_opt, out_opt_bytes, hipMemcpyDeviceToHost));

    float max_mag = 0.0f;
    float max_ref = 0.0f;
    float max_opt = 0.0f;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t o = 0; o < n; ++o) {
            double expected = 0.0;
            for (uint32_t kk = 0; kk < kp; ++kk)
                expected += static_cast<double>(p.dequant[static_cast<std::size_t>(o) * kp + kk]) *
                            static_cast<double>(adec[static_cast<std::size_t>(m) * kp + kk]);
            const float e = static_cast<float>(expected);
            const std::size_t idx = static_cast<std::size_t>(m) * n + o;
            const float r = bf16_to_f32(ref[idx]);
            const float g = opt[idx];
            max_mag = std::max(max_mag, std::fabs(e));
            max_ref = std::max(max_ref, std::fabs(r - e));
            max_opt = std::max(max_opt, std::fabs(g - e));
        }
    }
    const float ref_atol = std::max(1.0f, max_mag) * 1.5e-2f;
    const float opt_atol = std::max(1.0f, max_mag) * 1e-4f;
    const bool ok = (max_ref <= ref_atol) && (max_opt <= opt_atol);
    if (!ok)
        std::printf("FAIL: %s rows=%u N=%u K=%u correctness_abs=%.6g (atol %.4f) opt_abs=%.6g (atol %.4f)\n",
                    tag, rows, n, k, max_ref, ref_atol, max_opt, opt_atol);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== Block Scaled correctness-dispatch vs optimized Tests ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    auto ar = ps::gpu::GpuArena::create(0, 1024ull * 1024 * 1024);
    if (!ar.ok()) { std::printf("FAIL: arena\n"); return 1; }
    auto arena = ar.release();

    const uint32_t rows_list[] = {1u, 16u, 17u, 64u};
    const uint32_t n_list[] = {128u, 130u};
    const uint32_t k_list[] = {256u, 2560u};
    for (uint32_t rows : rows_list) {
        for (uint32_t n : n_list) {
            for (uint32_t k : k_list) {
                char tag[96];
                std::snprintf(tag, sizeof(tag), "fp8 rows=%u n=%u k=%u", rows, n, k);
                if (run_case(arena, true, rows, n, k, tag)) ++g_pass; else ++g_fail;
                std::snprintf(tag, sizeof(tag), "mxfp4 rows=%u n=%u k=%u", rows, n, k);
                if (run_case(arena, false, rows, n, k, tag)) ++g_pass; else ++g_fail;
            }
        }
    }

    // Qwen3.5-4B linear geometries that are inside the selector allowlist.
    const uint32_t big_n[] = {9216u, 2560u, 8192u, 4096u, 1024u, 2560u, 32u};
    const uint32_t big_k[] = {2560u, 9216u, 2560u, 2560u, 2560u, 4096u, 2560u};
    for (std::size_t i = 0; i < sizeof(big_n) / sizeof(big_n[0]); ++i) {
        for (uint32_t rows : {1u, 17u, 128u}) {
            char tag[96];
            std::snprintf(tag, sizeof(tag), "fp8 big rows=%u n=%u k=%u", rows, big_n[i], big_k[i]);
            if (run_case(arena, true, rows, big_n[i], big_k[i], tag)) ++g_pass; else ++g_fail;
            std::snprintf(tag, sizeof(tag), "mxfp4 big rows=%u n=%u k=%u", rows, big_n[i], big_k[i]);
            if (run_case(arena, false, rows, big_n[i], big_k[i], tag)) ++g_pass; else ++g_fail;
        }
    }

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
