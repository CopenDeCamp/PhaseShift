#include <phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/mxfp4/mxfp4_native.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_ASSERT(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        std::printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err)); \
        return false; \
    } \
} while (0)

using ps::kernel::Mxfp4GemmConfig;
using ps::kernel::Mxfp4GemmConfigId;

namespace {

int g_pass = 0;
int g_fail = 0;
double g_worst_rel = 0.0;

ps::bf16_t f32_to_bf16(float f) {
    uint32_t ui = 0;
    std::memcpy(&ui, &f, sizeof(float));
    const uint16_t bf = static_cast<uint16_t>((ui + 0x8000u) >> 16);
    return ps::bf16_t{bf};
}

float bf16_to_f32(ps::bf16_t b) {
    const uint32_t ui = static_cast<uint32_t>(b.data) << 16;
    float f = 0.0f;
    std::memcpy(&f, &ui, sizeof(float));
    return f;
}

Mxfp4GemmConfig config_for(uint32_t rows) {
    if (rows >= 8u * ps::kernel::kMxfp4GemmRowTile) return {Mxfp4GemmConfigId::RowBlock8};
    if (rows >= 4u * ps::kernel::kMxfp4GemmRowTile) return {Mxfp4GemmConfigId::RowBlock4};
    if (rows > ps::kernel::kMxfp4GemmRowTile) return {Mxfp4GemmConfigId::RowBlock2};
    return {Mxfp4GemmConfigId::RowBlock1};
}

const char* config_name(Mxfp4GemmConfigId id) {
    switch (id) {
        case Mxfp4GemmConfigId::RowBlock1: return "MB1";
        case Mxfp4GemmConfigId::RowBlock2: return "MB2";
        case Mxfp4GemmConfigId::RowBlock4: return "MB4";
        case Mxfp4GemmConfigId::RowBlock8: return "MB8";
        case Mxfp4GemmConfigId::Prefill2D: return "2D";
    }
    return "?";
}

uint8_t nibble_at(const std::vector<uint8_t>& codes, uint32_t row, uint32_t kk, uint32_t kp) {
    const uint8_t byte = codes[static_cast<std::size_t>(row) * (kp / 2u) + kk / 2u];
    return (kk & 1u) ? static_cast<uint8_t>(byte >> 4u) : static_cast<uint8_t>(byte & 0x0Fu);
}

bool run_case(ps::gpu::GpuArena& arena, Mxfp4GemmConfig config, uint32_t rows,
              uint32_t n, uint32_t k, const char* label) {
    std::mt19937 rng(20260922u + rows * 31u + n * 7u + k);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);

    std::vector<float> w(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : w) v = wdist(rng);
    for (auto& v : x) v = xdist(rng);

    const uint32_t kp = static_cast<uint32_t>((k + 31u) & ~31u);
    ps::quantization::mxfp4::Mxfp4CanonicalStore store;
    store.init(n, k, kp);
    for (uint32_t r = 0; r < n; ++r)
        store.quantize_row(r, std::span<const float>(w.data() + static_cast<std::size_t>(r) * k, k));

    ps::quantization::mxfp4::Mxfp4NativeHost native;
    if (!ps::quantization::mxfp4::preshuffle_mxfp4_native(store.view(), native)) {
        std::printf("FAIL: %s preshuffle\n", label);
        return false;
    }

    const uint32_t nb = kp / 32u;
    const uint32_t code_stride = kp;
    const uint32_t as_floats = 16u;
    const uint32_t as_bytes = as_floats * 4u;
    const uint32_t weight_scale_stride_bytes = native.scale_row_stride_bytes;

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

    const std::size_t act_codes_bytes = ((rows + 15u) / 16u) * static_cast<std::size_t>(code_stride) * 16u;
    const std::size_t act_scales_bytes = static_cast<std::size_t>(rows) * as_bytes;
    const std::size_t out_f32_bytes = static_cast<std::size_t>(rows) * n * 4u;

    auto wc = arena.allocate_aligned(native.codes.size(), 16);
    auto ws = arena.allocate_aligned(native.scales.size(), 16);
    auto xa = arena.allocate_aligned(hx.size() * sizeof(ps::bf16_t), 16);
    auto ac = arena.allocate_aligned(act_codes_bytes, 16);
    auto as = arena.allocate_aligned(act_scales_bytes, 16);
    auto ya = arena.allocate_aligned(out_f32_bytes, 16);
    if (!wc.ok() || !ws.ok() || !xa.ok() || !ac.ok() || !as.ok() || !ya.ok()) {
        std::printf("FAIL: %s alloc\n", label);
        return false;
    }
    auto* d_wc = static_cast<uint8_t*>(wc.release().data());
    auto* d_ws = static_cast<uint8_t*>(ws.release().data());
    auto* d_x = static_cast<ps::bf16_t*>(xa.release().data());
    auto* d_ac = static_cast<uint8_t*>(ac.release().data());
    auto* d_as = static_cast<float*>(as.release().data());
    auto* d_y = static_cast<float*>(ya.release().data());

    HIP_ASSERT(hipMemset(d_ac, 0, act_codes_bytes));
    HIP_ASSERT(hipMemset(d_as, 0, act_scales_bytes));
    HIP_ASSERT(hipMemcpy(d_wc, native.codes.data(), native.codes.size(), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_ws, native.scales.data(), native.scales.size(), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    hipError_t aq = ps::kernel::launch_activation_quantize_e4m3(
        d_x, d_ac, d_as, rows, k, kp, k, code_stride, as_bytes, 0);
    if (aq != hipSuccess) {
        std::printf("FAIL: %s activation quantize: %s\n", label, hipGetErrorString(aq));
        return false;
    }
    hipError_t hk = ps::kernel::launch_gemm_mxfp4_w4a8_wmma(
        config, d_wc, d_ws, reinterpret_cast<const int8_t*>(d_ac), d_as, d_y,
        ps::kernel::Mxfp4GemmOutputDType::F32, rows, n, kp,
        weight_scale_stride_bytes, code_stride, as_bytes, n, 0);
    if (hk != hipSuccess) {
        std::printf("FAIL: %s launch: %s\n", label, hipGetErrorString(hk));
        return false;
    }
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<float> got(static_cast<std::size_t>(rows) * n);
    HIP_ASSERT(hipMemcpy(got.data(), d_y, out_f32_bytes, hipMemcpyDeviceToHost));

    const std::vector<uint8_t>& codes = store.codes;
    float max_abs = 0.0f;
    double max_rel = 0.0;
    float max_mag = 0.0f;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t o = 0; o < n; ++o) {
            double expected = 0.0;
            for (uint32_t ib = 0; ib < nb; ++ib) {
                const double s = static_cast<double>(
                    ps::quantization::mxfp4::e8m0_decode_u8(store.scales[static_cast<std::size_t>(o) * nb + ib]));
                double p = 0.0;
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t kk = ib * 32u + j;
                    p += static_cast<double>(
                             ps::quantization::mxfp4::e2m1_decode_nibble(nibble_at(codes, o, kk, kp))) *
                         static_cast<double>(adec[static_cast<std::size_t>(m) * kp + kk]);
                }
                expected += p * s;
            }
            const float e = static_cast<float>(expected);
            const float g = got[static_cast<std::size_t>(m) * n + o];
            max_mag = std::max(max_mag, std::fabs(e));
            const float diff = std::fabs(g - e);
            max_abs = std::max(max_abs, diff);
            if (std::fabs(e) > 1e-3f)
                max_rel = std::max(max_rel, static_cast<double>(diff) / std::fabs(e));
        }
    }
    if (max_rel > g_worst_rel) g_worst_rel = max_rel;
    const float atol = std::max(1.0f, max_mag) * 1e-4f;
    const bool ok = max_abs <= atol;
    if (!ok)
        std::printf("FAIL: %s rows=%u N=%u K=%u Kp=%u %s max_abs=%.6g max_rel=%.4f atol=%.4f\n",
                    label, rows, n, k, kp, config_name(config.id), max_abs, max_rel, atol);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== MXFP4 W4A8 WMMA GEMM Tests ===\n");

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    auto arena_result = ps::gpu::GpuArena::create(0, 1024ull * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    const uint32_t rows_list[] = {1, 2, 4, 8, 16, 17, 33, 64, 128};
    const uint32_t n_list[] = {16, 17, 128, 129, 130, 256};
    const uint32_t k_list[] = {32, 64, 128, 130, 256, 2560};

    char label[128];
    for (uint32_t rows : rows_list) {
        for (uint32_t n : n_list) {
            for (uint32_t k : k_list) {
                if (static_cast<std::size_t>(rows) * n * k > 4000000u) continue;
                if (run_case(arena, config_for(rows), rows, n, k, "mxfp4")) ++g_pass; else ++g_fail;
            }
        }
    }

    for (uint32_t rows : {16u, 17u, 64u, 128u, 129u}) {
        for (Mxfp4GemmConfigId id : {Mxfp4GemmConfigId::RowBlock1, Mxfp4GemmConfigId::RowBlock2,
                                     Mxfp4GemmConfigId::RowBlock4, Mxfp4GemmConfigId::RowBlock8}) {
            std::snprintf(label, sizeof(label), "mxfp4-force-%s", config_name(id));
            if (run_case(arena, Mxfp4GemmConfig{id}, rows, 130u, 256u, label)) ++g_pass; else ++g_fail;
        }
    }

    for (uint32_t rows : {512u, 2048u}) {
        if (run_case(arena, config_for(rows), rows, 64u, 256u, "mxfp4-large")) ++g_pass; else ++g_fail;
        if (run_case(arena, Mxfp4GemmConfig{Mxfp4GemmConfigId::RowBlock1},
                     rows, 64u, 256u, "mxfp4-large-mb1")) ++g_pass; else ++g_fail;
    }

    for (uint32_t rows : {256u, 512u}) {
        for (uint32_t n : {64u, 128u}) {
            if (run_case(arena, Mxfp4GemmConfig{Mxfp4GemmConfigId::Prefill2D},
                         rows, n, 256u, "mxfp4-2d")) ++g_pass; else ++g_fail;
        }
    }

    std::printf("\nWorst relative error vs exact host reference: %.3e\n", g_worst_rel);
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
