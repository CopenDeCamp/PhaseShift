#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_block_scaled_correctness.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

uint32_t h32(uint64_t x) {
    uint32_t h = static_cast<uint32_t>(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

ps::bf16_t rand_bf16(uint64_t seed) {
    const uint32_t h = h32(seed);
    const float u = static_cast<float>(h & 0xFFFFu) / 65535.0f;
    const float v = (u * 2.0f - 1.0f) * 0.25f;
    return ps::bf16_t{ps::f32_to_bf16_rne(v)};
}

ps::kernel::Psq4GemmConfig selector_or_clone(uint32_t rows, uint32_t n, uint32_t k,
                                             uint32_t kp) {
    const auto sel = ps::qwen35::runtime::select_psq4_gemm_config(
        ps::qwen35::runtime::Psq4GemmSelectorInput{rows, n, k, kp});
    if (sel) return *sel;
    using Id = ps::kernel::Psq4GemmConfigId;
    const bool use_2d =
        (rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        n >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d) {
        const bool k128 = (kp % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        const bool n128 = (n % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        uint32_t cfg;
        if (rows >= ps::kernel::kPsq4GemmPrefill2dLargeMinRows &&
            n >= ps::kernel::kPsq4GemmPrefill2dLargeMinOutFeatures && k128 && n128)
            cfg = 3u;
        else if (k128)
            cfg = 1u;
        else if (n128)
            cfg = 2u;
        else
            cfg = 0u;
        const uint32_t ob = (cfg == 2u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                                     : ps::kernel::kPsq4GemmPrefill2dBlock64;
        const uint32_t kc = (cfg == 1u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                                     : ps::kernel::kPsq4GemmPrefill2dBlock64;
        if ((kp % kc) == 0u && (n % ob) == 0u)
            return ps::kernel::Psq4GemmConfig{
                static_cast<Id>(static_cast<uint8_t>(Id::Prefill2D_K64N64) + cfg)};
    }
    if (rows >= 8u * ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock2};
    return ps::kernel::Psq4GemmConfig{Id::RowBlock1};
}

bool bf16_finite(const std::vector<uint16_t>& v) {
    for (const uint16_t bits : v) {
        uint32_t fbits = static_cast<uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &fbits, 4);
        if (!std::isfinite(f)) return false;
    }
    return true;
}

float bf16_f(uint16_t bits) {
    uint32_t fbits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &fbits, 4);
    return f;
}

struct ShapeCase {
    uint32_t n;
    uint32_t k;
    bool newly_allowed;
};

void* xalloc(size_t bytes) {
    void* p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess || p == nullptr) {
        std::fprintf(stderr, "hipMalloc(%zu) failed\n", bytes);
        std::abort();
    }
    return p;
}

void run_shape(hipStream_t stream, const ShapeCase& sc) {
    const uint32_t n = sc.n;
    const uint32_t k = sc.k;
    const uint32_t kp = (k + 31u) & ~31u;
    char label[128];

    std::vector<ps::bf16_t> hw(static_cast<size_t>(n) * k);
    for (size_t i = 0; i < hw.size(); ++i)
        hw[i] = rand_bf16(i * 131u + n * 7u + k);

    ps::quantization::psq::CanonicalQuantStore store;
    store.init(ps::quantization::QuantFormatId::Psq4, n, k, kp);
    {
        std::vector<float> row(k);
        for (uint32_t o = 0; o < n; ++o) {
            for (uint32_t i = 0; i < k; ++i) {
                uint32_t bits = static_cast<uint32_t>(hw[static_cast<size_t>(o) * k + i].data) << 16u;
                std::memcpy(&row[i], &bits, 4);
            }
            if (!store.quantize_row(o, row)) {
                std::snprintf(label, sizeof(label), "canonical_quantize n=%u k=%u", n, k);
                check(false, label);
                return;
            }
        }
    }
    ps::quantization::psq::NativeQuantHost native;
    if (!ps::quantization::psq::preshuffle_native(store.view(), native)) {
        std::snprintf(label, sizeof(label), "preshuffle n=%u k=%u", n, k);
        check(false, label);
        return;
    }

    uint8_t* d_canon_codes = static_cast<uint8_t*>(xalloc(store.codes.size()));
    uint8_t* d_canon_scales = static_cast<uint8_t*>(xalloc(store.metadata1.size()));
    uint8_t* d_native_codes = static_cast<uint8_t*>(xalloc(native.codes.size()));
    uint8_t* d_native_scales = static_cast<uint8_t*>(xalloc(native.metadata1.size()));
    (void)hipMemcpy(d_canon_codes, store.codes.data(), store.codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_canon_scales, store.metadata1.data(), store.metadata1.size(),
              hipMemcpyHostToDevice);
    (void)hipMemcpy(d_native_codes, native.codes.data(), native.codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_native_scales, native.metadata1.data(), native.metadata1.size(),
              hipMemcpyHostToDevice);

    const uint32_t max_rows = 8u;
    const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(k, max_rows);
    uint8_t* a8_base = static_cast<uint8_t*>(xalloc(layout.total_bytes));
    uint8_t* a_codes = a8_base + layout.codes_offset;
    float* a_scales = reinterpret_cast<float*>(a8_base + layout.scales_offset);

    for (const uint32_t rows : {1u, 2u, 4u, 8u}) {
        std::vector<ps::bf16_t> ha(static_cast<size_t>(rows) * k);
        for (size_t i = 0; i < ha.size(); ++i)
            ha[i] = rand_bf16(i * 977u + rows * 31u + k * 3u);
        ps::bf16_t* d_input = static_cast<ps::bf16_t*>(xalloc(ha.size() * 2));
        ps::bf16_t* d_opt = static_cast<ps::bf16_t*>(xalloc(ha.size() * 2));
        ps::bf16_t* d_ref = static_cast<ps::bf16_t*>(xalloc(ha.size() * 2));
        (void)hipMemcpy(d_input, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);

        (void)hipGetLastError();
        hipError_t err = ps::kernel::launch_activation_quantize_e4m3(
            d_input, a_codes, a_scales, rows, k, kp, k, layout.code_row_stride_bytes,
            layout.scale_row_stride_bytes, stream);
        std::snprintf(label, sizeof(label), "activation quantize launch n=%u k=%u rows=%u (%s)",
                      n, k, rows, hipGetErrorString(err));
        check(err == hipSuccess, label);
        check(hipStreamSynchronize(stream) == hipSuccess, "quantize sync");

        const ps::kernel::Psq4GemmConfig config = selector_or_clone(rows, n, k, kp);
        err = ps::kernel::launch_gemm_psq4_w4a8_wmma_auto(
            config, d_native_codes, d_native_scales,
            reinterpret_cast<const int8_t*>(a_codes), a_scales, d_opt,
            ps::kernel::Psq4GemmOutputDType::BF16, rows, n, kp,
            native.metadata1_stride_bytes, layout.code_row_stride_bytes,
            layout.scale_row_stride_bytes, n, stream);
        check(err == hipSuccess, "psq4 gemm launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "gemm sync");

        ps::kernel::detail::BlockScaledWeightView wv;
        wv.kind = ps::kernel::detail::BlockScaledKind::Psq4;
        wv.codes = d_canon_codes;
        wv.scales = d_canon_scales;
        wv.out_features = n;
        wv.k = k;
        wv.k_padded = kp;
        err = ps::kernel::detail::launch_gemm_block_scaled_reference(wv, d_input, d_ref, rows,
                                                                    stream);
        check(err == hipSuccess, "psq4 reference launch");

        std::vector<uint16_t> ho(static_cast<size_t>(rows) * n);
        std::vector<uint16_t> hr(ho.size());
        (void)hipMemcpy(ho.data(), d_opt, ho.size() * 2, hipMemcpyDeviceToHost);
        (void)hipMemcpy(hr.data(), d_ref, hr.size() * 2, hipMemcpyDeviceToHost);
        (void)hipStreamSynchronize(stream);

        std::snprintf(label, sizeof(label), "finite n=%u k=%u rows=%u", n, k, rows);
        check(bf16_finite(ho) && bf16_finite(hr), label);

        double max_mag = 0.0;
        double max_diff = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < ho.size(); ++i) {
            const float r = bf16_f(hr[i]);
            const float x = bf16_f(ho[i]);
            const double d = std::fabs(static_cast<double>(x) - static_cast<double>(r));
            if (std::fabs(r) > max_mag) max_mag = std::fabs(r);
            if (d > max_diff) { max_diff = d; worst = i; }
        }
        const double atol = std::max(1.0, max_mag) * 1.5e-2;
        std::snprintf(label, sizeof(label), "correctness n=%u k=%u rows=%u (%.4f vs %.4f)", n, k,
                      rows, max_diff, atol);
        if (max_diff > atol) {
            std::printf("  worst idx=%zu opt=%g ref=%g\n", worst, bf16_f(ho[worst]),
                        bf16_f(hr[worst]));
        }
        check(max_diff <= atol, label);

        (void)hipFree(d_input);
        (void)hipFree(d_opt);
        (void)hipFree(d_ref);
    }

    (void)hipFree(d_canon_codes);
    (void)hipFree(d_canon_scales);
    (void)hipFree(d_native_codes);
    (void)hipFree(d_native_scales);
    (void)hipFree(a8_base);
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    const ShapeCase shapes[] = {
        {1280u, 5120u, true},
        {4096u, 5120u, true},
        {5120u, 4096u, true},
        {5120u, 25600u, true},
        {1024u, 5120u, false},
        {17408u, 5120u, false},
        {5120u, 17408u, false},
    };
    for (const ShapeCase& sc : shapes) {
        if (!sc.newly_allowed) continue;
        const ps::qwen35::runtime::Psq4GemmSelectorInput lin{8u, sc.n, sc.k, sc.k};
        char label[96];
        std::snprintf(label, sizeof(label), "psq4 allowlist n=%u k=%u", sc.n, sc.k);
        check(ps::qwen35::runtime::select_psq4_gemm_config(lin).has_value(), label);
    }
    for (const ShapeCase& sc : shapes) run_shape(stream, sc);

    (void)hipStreamDestroy(stream);
    std::printf("test_dflash2_psq4_shapes: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
