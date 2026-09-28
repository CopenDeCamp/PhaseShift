#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_block_scaled_correctness.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/quantization/fp8/block128.h>
#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#define HIP_ASSERT(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        std::printf("HIP error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err)); \
        return 1; \
    } \
} while (0)

using ps::kernel::detail::BlockScaledKind;
using ps::kernel::detail::BlockScaledWeightView;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& name) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL: %s\n", name.c_str());
    }
}

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

uint64_t align_up(uint64_t v, uint64_t a) {
    return ((v + a - 1u) / a) * a;
}

struct HostPayload {
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t kp = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<float> dequant;
};

HostPayload build_fp8(const std::vector<float>& w, uint32_t n, uint32_t k) {
    const uint32_t kp = static_cast<uint32_t>(align_up(k, 128));
    HostPayload p;
    p.n = n;
    p.k = k;
    p.kp = kp;
    ps::quantization::fp8::Fp8BlockCanonicalStore store;
    store.init(1, n, k, kp);
    store.quantize_matrix(0, w.data(), k);
    p.codes = store.codes;
    p.scales.resize(store.scales.size() * sizeof(float));
    std::memcpy(p.scales.data(), store.scales.data(), p.scales.size());
    p.dequant.assign(static_cast<std::size_t>(n) * kp, 0.0f);
    ps::quantization::fp8::dequantize_fp8_block128_canonical_matrix(
        store.view(), 0, p.dequant.data(), kp);
    return p;
}

HostPayload build_mxfp4(const std::vector<float>& w, uint32_t n, uint32_t k) {
    const uint32_t kp = static_cast<uint32_t>(align_up(k, 32));
    HostPayload p;
    p.n = n;
    p.k = k;
    p.kp = kp;
    ps::quantization::mxfp4::Mxfp4CanonicalStore store;
    store.init(n, k, kp);
    for (uint32_t r = 0; r < n; ++r) {
        store.quantize_row(r, std::span<const float>(w.data() + static_cast<std::size_t>(r) * k, k));
    }
    p.codes = store.codes;
    p.scales = store.scales;
    p.dequant.assign(static_cast<std::size_t>(n) * kp, 0.0f);
    ps::quantization::mxfp4::dequantize_mxfp4_canonical(store.view(), p.dequant);
    return p;
}

std::vector<float> quantize_activation_row(const std::vector<float>& x, uint32_t rows,
                                           uint32_t k, uint32_t kp) {
    std::vector<float> out(static_cast<std::size_t>(rows) * kp, 0.0f);
    for (uint32_t m = 0; m < rows; ++m) {
        float peak = 0.0f;
        for (uint32_t i = 0; i < k; ++i) {
            const float v = x[static_cast<std::size_t>(m) * k + i];
            if (std::isfinite(v)) peak = std::max(peak, std::fabs(v));
        }
        const float scale = peak > 0.0f ? peak / 448.0f : 0.0f;
        for (uint32_t i = 0; i < kp; ++i) {
            float v = 0.0f;
            if (i < k) v = x[static_cast<std::size_t>(m) * k + i];
            const uint8_t code = scale > 0.0f
                ? ps::quantization::fpx::e4m3_encode_u8(v / scale)
                : 0u;
            out[static_cast<std::size_t>(m) * kp + i] =
                ps::quantization::fpx::e4m3_decode_f32(code) * scale;
        }
    }
    return out;
}

int run_case(ps::gpu::GpuArena& arena, BlockScaledKind kind, uint32_t n, uint32_t k,
             uint32_t rows, const char* label) {
    const uint32_t kp = static_cast<uint32_t>(
        kind == BlockScaledKind::Fp8Block128 ? align_up(k, 128) : align_up(k, 32));

    std::mt19937 rng(20260920u + n * 7u + k + rows);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(rows) * k);
    for (auto& v : w) v = wdist(rng);
    for (auto& v : x) v = xdist(rng);

    HostPayload payload = (kind == BlockScaledKind::Fp8Block128)
        ? build_fp8(w, n, k) : build_mxfp4(w, n, k);

    std::vector<ps::bf16_t> hx(static_cast<std::size_t>(rows) * k);
    std::vector<float> xr(x.size());
    for (std::size_t i = 0; i < hx.size(); ++i) {
        hx[i] = f32_to_bf16(x[i]);
        xr[i] = bf16_to_f32(hx[i]);
    }

    auto wc = arena.allocate_aligned(payload.codes.size(), 16);
    auto ws = arena.allocate_aligned(payload.scales.size(), 16);
    auto xa = arena.allocate_aligned(hx.size() * sizeof(ps::bf16_t), 16);
    auto ya = arena.allocate_aligned(static_cast<std::size_t>(rows) * n * sizeof(ps::bf16_t), 16);
    if (!wc.ok() || !ws.ok() || !xa.ok() || !ya.ok()) {
        std::printf("FAIL: %s alloc\n", label);
        return 1;
    }
    auto* d_codes = static_cast<uint8_t*>(wc.release().data());
    auto* d_scales = static_cast<uint8_t*>(ws.release().data());
    auto* d_x = static_cast<ps::bf16_t*>(xa.release().data());
    auto* d_y = static_cast<ps::bf16_t*>(ya.release().data());

    HIP_ASSERT(hipMemcpy(d_codes, payload.codes.data(), payload.codes.size(), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_scales, payload.scales.data(), payload.scales.size(), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice));

    BlockScaledWeightView view;
    view.kind = kind;
    view.codes = d_codes;
    view.scales = d_scales;
    view.out_features = n;
    view.k = k;
    view.k_padded = kp;

    hipError_t launch = ps::kernel::detail::launch_gemm_block_scaled_reference(
        view, d_x, d_y, rows, 0);
    check(launch == hipSuccess, std::string(label) + " launch");
    HIP_ASSERT(hipStreamSynchronize(0));

    std::vector<ps::bf16_t> hy(static_cast<std::size_t>(rows) * n);
    HIP_ASSERT(hipMemcpy(hy.data(), d_y, hy.size() * sizeof(ps::bf16_t), hipMemcpyDeviceToHost));

    const std::vector<float> adec = quantize_activation_row(xr, rows, k, kp);
    float max_abs = 0.0f;
    double max_rel = 0.0;
    float max_mag = 0.0f;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t o = 0; o < n; ++o) {
            double expected = 0.0;
            for (uint32_t kk = 0; kk < kp; ++kk) {
                expected += static_cast<double>(payload.dequant[static_cast<std::size_t>(o) * kp + kk]) *
                            static_cast<double>(adec[static_cast<std::size_t>(m) * kp + kk]);
            }
            const float got = bf16_to_f32(hy[static_cast<std::size_t>(m) * n + o]);
            const float ref = bf16_to_f32(f32_to_bf16(static_cast<float>(expected)));
            max_mag = std::max(max_mag, static_cast<float>(std::fabs(expected)));
            const float diff = static_cast<float>(std::fabs(got - ref));
            max_abs = std::max(max_abs, diff);
            if (std::fabs(ref) > 1e-3)
                max_rel = std::max(max_rel, static_cast<double>(diff) / std::fabs(ref));
        }
    }
    const float atol = std::max(1.0f, max_mag) * 2e-3f;
    std::printf("%s rows=%u N=%u K=%u Kp=%u  max_abs=%.6f max_rel=%.4f (atol=%.4f)\n",
                label, rows, n, k, kp, max_abs, max_rel, atol);
    check(max_abs <= atol, std::string(label) + " atol");
    return 0;
}

}  // namespace

int main() {
    std::printf("=== Block Scaled GEMM/GEMV Reference Tests ===\n");

    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    HIP_ASSERT(hipSetDevice(0));

    auto arena_result = ps::gpu::GpuArena::create(0, 512ull * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    run_case(arena, BlockScaledKind::Fp8Block128, 130, 200, 4, "fp8 gemm n130 k200");
    run_case(arena, BlockScaledKind::Fp8Block128, 128, 256, 4, "fp8 gemm n128 k256");
    run_case(arena, BlockScaledKind::Fp8Block128, 129, 128, 1, "fp8 gemv n129 k128");
    run_case(arena, BlockScaledKind::Mxfp4, 130, 200, 4, "mxfp4 gemm n130 k200");
    run_case(arena, BlockScaledKind::Mxfp4, 256, 256, 4, "mxfp4 gemm n256 k256");
    run_case(arena, BlockScaledKind::Mxfp4, 129, 128, 1, "mxfp4 gemv n129 k128");

    std::printf(g_fail == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
