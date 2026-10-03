#include <phaseshift/core/gpu/gfx_arch.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

std::vector<uint8_t> ws_frag_pack(const std::vector<uint8_t>& ws, uint32_t n, uint32_t kp, uint32_t wss) {
    const uint32_t nb = kp / 32u, tiles = (n + 15u) / 16u;
    std::vector<uint8_t> out(static_cast<size_t>(tiles) * nb * 32u, 0);
    for (uint32_t t = 0; t < tiles; ++t)
        for (uint32_t ib = 0; ib < nb; ++ib)
            for (uint32_t ol = 0; ol < 16u; ++ol) {
                const uint32_t o = t * 16u + ol;
                if (o >= n) continue;
                out[static_cast<size_t>(t) * nb * 32u + ib * 32u + ol * 2u] =
                    ws[static_cast<size_t>(o) * wss + ib * 2u];
                out[static_cast<size_t>(t) * nb * 32u + ib * 32u + ol * 2u + 1u] =
                    ws[static_cast<size_t>(o) * wss + ib * 2u + 1u];
            }
    return out;
}

int passed = 0;
int failed = 0;

std::vector<int8_t> act_frag_pack(const std::vector<int8_t>& ac, uint32_t rows, uint32_t code_stride) {
    const uint32_t nb = code_stride / 32u;
    const uint32_t tiles = (rows + 15u) / 16u;
    std::vector<int8_t> out(static_cast<size_t>(tiles) * code_stride * 16u, 0);
    for (uint32_t t = 0; t < tiles; ++t)
        for (uint32_t blk = 0; blk < nb; ++blk)
            for (uint32_t ol = 0; ol < 16u; ++ol) {
                const uint32_t m = t * 16u + ol;
                if (m >= rows) continue;
                for (uint32_t half = 0; half < 2u; ++half)
                    for (uint32_t jj = 0; jj < 16u; ++jj)
                        out[static_cast<size_t>(t) * (code_stride * 16u) +
                            static_cast<size_t>(blk) * 512u + half * 256u + ol * 16u + jj] =
                            ac[static_cast<size_t>(m) * code_stride + blk * 32u +
                               half * 16u + jj];
            }
    return out;
}

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t h32(uint64_t x) {
    uint32_t h = static_cast<uint32_t>(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

float cb10_nibble_to_float_h(int n) {
    const float kMags[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    const float v = kMags[n & 7];
    return (n & 8) ? -v : v;
}

int int_to_fp4_nibble(int v) {
    switch (v) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 6: return 5;
        case 8: return 6;
        case 10: return 7;
        case -1: return 9;
        case -2: return 10;
        case -3: return 11;
        case -4: return 12;
        case -6: return 13;
        case -8: return 14;
        case -10: return 15;
        default: return 0;
    }
}

float e4m3_decode_h(uint8_t b) {
    const uint32_t sg = static_cast<uint32_t>(b) >> 7;
    const uint32_t ex = (static_cast<uint32_t>(b) >> 3) & 0xFu;
    const uint32_t mn = static_cast<uint32_t>(b) & 7u;
    const float mag = (ex == 0u)
        ? std::ldexp(static_cast<float>(mn), -9)
        : std::ldexp(1.0f + static_cast<float>(mn) / 8.0f, static_cast<int>(ex) - 7);
    return sg ? -mag : mag;
}

uint8_t e4_fixture(uint32_t h) {
    uint8_t b = static_cast<uint8_t>(h & 0xFFu);
    if ((b & 0x78u) == 0x78u) b = static_cast<uint8_t>(b & ~0x40u);
    return b;
}

float bf16_scale_h(const std::vector<uint8_t>& ws, size_t off) {
    const uint32_t bits = static_cast<uint32_t>(ws[off]) |
                          (static_cast<uint32_t>(ws[off + 1u]) << 8u);
    const uint32_t fbits = bits << 16u;
    float v;
    std::memcpy(&v, &fbits, sizeof(v));
    return v;
}

float fp4_codebook_h(uint32_t code) {
    return cb10_nibble_to_float_h(static_cast<int>(code));
}


struct Strides {
    uint32_t k_padded;
    uint32_t nb;
    uint32_t wcp;
    uint32_t wss;
    uint32_t wss_frag;
    uint32_t cstride;
    uint32_t as_floats;
};

Strides mk_strides(uint32_t k) {
    const uint32_t kp = (k + 31u) & ~31u;
    const uint32_t nb = kp >> 5;
    Strides s;
    s.k_padded = kp;
    s.nb = nb;
    s.wcp = kp >> 1;
    s.wss = ((nb * 2u + 15u) & ~15u);
    s.wss_frag = nb * 32u;
    s.cstride = kp;
    s.as_floats = ((nb * 4u + 15u) & ~15u) / 4u;
    return s;
}

std::vector<uint8_t> fp4_frag_pack(const std::vector<uint8_t>& wc, uint32_t N, uint32_t k_padded) {
    const uint32_t tiles = (N + 15u) / 16u;
    const uint32_t ibs = k_padded / 32u;
    const size_t src_k2 = k_padded / 2u;
    std::vector<uint8_t> out(static_cast<size_t>(tiles) * ibs * 256u, 0u);
    for (uint32_t t = 0; t < tiles; ++t)
        for (uint32_t ib = 0; ib < ibs; ++ib)
            for (uint32_t ol = 0; ol < 16u; ++ol) {
                const uint32_t o = t * 16u + ol;
                if (o >= N) continue;
                for (uint32_t j = 0; j < 16u; ++j)
                    out[(static_cast<size_t>(t) * ibs + ib) * 256u + ol * 16u + j] =
                        wc[static_cast<size_t>(o) * src_k2 + ib * 16u + j];
            }
    return out;
}

void cpu_ref(const std::vector<uint8_t>& wc, const std::vector<uint8_t>& ws,
             const std::vector<int8_t>& ac, const std::vector<float>& as,
             uint32_t rows, uint32_t N, uint32_t k_padded, uint32_t wcp,
             uint32_t wss, uint32_t cstride, uint32_t as_floats,
             std::vector<float>& ref, std::vector<double>& cond) {
    const uint32_t nb = k_padded >> 5;
    const size_t total = static_cast<size_t>(rows) * N;
    ref.assign(total, 0.f);
    cond.assign(total, 0.0);
    for (uint32_t m = 0; m < rows; ++m) {
        const float as0 = as[static_cast<size_t>(m) * as_floats];
        for (uint32_t o = 0; o < N; ++o) {
            double p = 0.0;
            double c = 0.0;
            for (uint32_t ib = 0; ib < nb; ++ib) {
                const double wsb = static_cast<double>(
                    bf16_scale_h(ws, static_cast<size_t>(o) * wss + 2u * ib));
                for (uint32_t j = 0; j < 16u; ++j) {
                    const uint8_t byte = wc[static_cast<size_t>(o) * wcp + ib * 16u + j];
                    const double t0 = static_cast<double>(fp4_codebook_h(byte & 0x0Fu)) * wsb *
                        static_cast<double>(e4m3_decode_h(
                            static_cast<uint8_t>(ac[static_cast<size_t>(m) * cstride + ib * 32u + j])));
                    const double t1 = static_cast<double>(fp4_codebook_h(byte >> 4)) * wsb *
                        static_cast<double>(e4m3_decode_h(
                            static_cast<uint8_t>(ac[static_cast<size_t>(m) * cstride + ib * 32u + 16u + j])));
                    p += t0 + t1;
                    c += std::fabs(t0 * as0) + std::fabs(t1 * as0);
                }
            }
            ref[static_cast<size_t>(m) * N + o] = static_cast<float>(p) * as0;
            cond[static_cast<size_t>(m) * N + o] = c;
        }
    }
}

bool finite_ok(float v) {
    return !(std::isnan(v) || std::isinf(v));
}

bool f32_ok(const std::vector<float>& got, const std::vector<float>& ref,
            const std::vector<double>& cond) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (!finite_ok(got[i]))
            return false;
        const double ad =
            std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        if (ad > 1e-4 * (cond[i] + 1e-6))
            return false;
    }
    return true;
}

float bf16_bits_to_float(uint16_t bits) {
    const uint32_t fbits = static_cast<uint32_t>(bits) << 16u;
    float v;
    std::memcpy(&v, &fbits, sizeof(v));
    return v;
}

bool bf16_ok(const std::vector<uint16_t>& got, const std::vector<float>& ref,
             const std::vector<double>& cond) {
    for (size_t i = 0; i < got.size(); ++i) {
        const float v = bf16_bits_to_float(got[i]);
        if (!finite_ok(v))
            return false;
        const double ad =
            std::fabs(static_cast<double>(v) - static_cast<double>(ref[i]));
        if (ad > 1e-2 * (cond[i] + 1e-6))
            return false;
    }
    return true;
}

struct Bufs {
    uint8_t* d_wc; uint8_t* d_ws; int8_t* d_ac; float* d_as;
    float* d_of32; ::ps::bf16_t* d_obf16;
};

struct Case {
    std::vector<uint8_t> wc, ws;
    std::vector<int8_t> ac;
    std::vector<float> as, ref;
    std::vector<double> cond;
    uint32_t rows, N, k_padded, wcp, wss, wss_frag, cstride, as_floats;
};

void gen_random(uint32_t rows, uint32_t N, uint32_t k, bool zero_block1,
                uint32_t ws_even, uint32_t ws_odd, uint32_t as_base, Case& c) {
    Strides s = mk_strides(k);
    c.rows = rows; c.N = N; c.k_padded = s.k_padded; c.wcp = s.wcp;
    c.wss = s.wss; c.wss_frag = s.wss_frag; c.cstride = s.cstride; c.as_floats = s.as_floats;
    c.wc.assign(static_cast<size_t>(N) * s.wcp, 0);
    for (size_t i = 0; i < c.wc.size(); ++i) {
        const uint32_t h = h32(i * 2654435761u + 77u);
        c.wc[i] = static_cast<uint8_t>((h & 0x0Fu) | ((h >> 4u) & 0xF0u));
    }
    c.ws.assign(static_cast<size_t>(N) * s.wss, 0);
    for (uint32_t o = 0; o < N; ++o) {
        for (uint32_t q = 0; q < s.nb; ++q) {
            const uint32_t he = h32(o * 131u + q * 17u);
            const uint32_t ho = h32(o * 197u + q * 23u);
            const uint16_t bits = ws_even
                ? static_cast<uint16_t>(((0x3Cu + (he % 3u)) << 8) | (he & 0xFFu))
                : static_cast<uint16_t>((0x3Cu << 8) | (ho & 0xFFu));
            c.ws[static_cast<size_t>(o) * s.wss + 2u * q] = static_cast<uint8_t>(bits & 0xFFu);
            c.ws[static_cast<size_t>(o) * s.wss + 2u * q + 1u] = static_cast<uint8_t>(bits >> 8u);
        }
    }
    c.ac.assign(static_cast<size_t>(rows) * s.cstride, 0);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t cc = 0; cc < s.cstride; ++cc) {
            if (cc < k) {
                c.ac[static_cast<size_t>(m) * s.cstride + cc] =
                    static_cast<int8_t>(e4_fixture(h32(m * 7919u + cc * 31u)));
            }
        }
    }
    if (zero_block1 && s.nb >= 2) {
        for (uint32_t m = 0; m < rows; ++m)
            for (uint32_t cc = 32u; cc < 64u && cc < s.cstride; ++cc)
                c.ac[static_cast<size_t>(m) * s.cstride + cc] = 0;
    }
    c.as.assign(static_cast<size_t>(rows) * s.as_floats, 0.f);
    for (uint32_t m = 0; m < rows; ++m) {
        const float v = as_base
            ? (0.1f + static_cast<float>(h32(m * 31u) % 1000u) / 1000.0f)
            : 1.0f;
        c.as[static_cast<size_t>(m) * s.as_floats] = v;
    }
    cpu_ref(c.wc, c.ws, c.ac, c.as, rows, N, s.k_padded, s.wcp, s.wss, s.cstride, s.as_floats, c.ref, c.cond);
}

ps::kernel::Psq4GemmConfig psq4_row_block_config(uint32_t rows, uint32_t N) {
    using Id = ps::kernel::Psq4GemmConfigId;
    if (rows >= 8u * ps::kernel::kPsqGemmRowTile &&
        N >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsqGemmRowTile &&
        N >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock4};
    if (rows > ps::kernel::kPsqGemmRowTile &&
        N >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock2};
    return {Id::RowBlock1};
}

ps::kernel::Psq4GemmConfig psq4_config_for(uint32_t rows, uint32_t N, uint32_t k_padded) {
    using Id = ps::kernel::Psq4GemmConfigId;
    const bool use_2d =
        (rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        N >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d) {
        const bool k128 = (k_padded % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        const bool n128 = (N % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        uint32_t cfg;
        if (rows >= ps::kernel::kPsq4GemmPrefill2dLargeMinRows &&
            N >= ps::kernel::kPsq4GemmPrefill2dLargeMinOutFeatures && k128 && n128)
            cfg = 3u;
        else if (k128)
            cfg = 1u;
        else if (n128)
            cfg = 2u;
        else
            cfg = 0u;
        const uint32_t ob =
            (cfg == 2u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                     : ps::kernel::kPsq4GemmPrefill2dBlock64;
        const uint32_t kc =
            (cfg == 1u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                     : ps::kernel::kPsq4GemmPrefill2dBlock64;
        if ((k_padded % kc) == 0u && (N % ob) == 0u) {
            switch (cfg) {
                case 1u: return {Id::Prefill2D_K128N64};
                case 2u: return {Id::Prefill2D_K64N128};
                case 3u: return {Id::Prefill2D_K128N128};
                default: return {Id::Prefill2D_K64N64};
            }
        }
        if ((N % ps::kernel::kPsq4GemmPrefill2dBlock64) == 0u &&
            (k_padded % ps::kernel::kPsq4GemmPrefill2dBlock64) == 0u)
            return {Id::Prefill2D_K64N64};
    }
    return psq4_row_block_config(rows, N);
}

bool run_case(Case& c, ps::gpu::GpuArena& arena, hipStream_t stream, const char* label) {
    const ps::kernel::Psq4GemmConfig config = psq4_config_for(c.rows, c.N, c.k_padded);
    const size_t out_bytes = static_cast<size_t>(c.rows) * c.N * 4;
    const std::vector<uint8_t> pwc = fp4_frag_pack(c.wc, c.N, c.k_padded);
    const std::vector<int8_t> ac_frag = act_frag_pack(c.ac, c.rows, c.cstride);
    auto wcr = arena.allocate_aligned(pwc.size(), 16);
    const std::vector<uint8_t> pws = ws_frag_pack(c.ws, c.N, c.k_padded, c.wss);
    auto wsr = arena.allocate_aligned(pws.size(), 16);
    auto acr = arena.allocate_aligned(ac_frag.size(), 16);
    auto asr = arena.allocate_aligned(c.as.size() * 4, 16);
    auto of32r = arena.allocate_aligned(out_bytes, 16);
    auto obf16r = arena.allocate_aligned(static_cast<size_t>(c.rows) * c.N * 2, 16);
    if (!wcr.ok() || !wsr.ok() || !acr.ok() || !asr.ok() || !of32r.ok() || !obf16r.ok()) {
        std::printf("FAIL: %s alloc\n", label);
        return false;
    }
    Bufs b;
    b.d_wc = static_cast<uint8_t*>(wcr.release().data());
    b.d_ws = static_cast<uint8_t*>(wsr.release().data());
    b.d_ac = static_cast<int8_t*>(acr.release().data());
    b.d_as = static_cast<float*>(asr.release().data());
    b.d_of32 = static_cast<float*>(of32r.release().data());
    b.d_obf16 = static_cast<::ps::bf16_t*>(obf16r.release().data());
    if (hipMemcpy(b.d_wc, pwc.data(), pwc.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(b.d_ws, pws.data(), pws.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(b.d_ac, ac_frag.data(), ac_frag.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(b.d_as, c.as.data(), c.as.size() * 4, hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: %s h2d\n", label);
        return false;
    }
    hipError_t hf = ps::kernel::launch_gemm_psq4_w4a8_wmma(
        config, b.d_wc, b.d_ws, b.d_ac, b.d_as, b.d_of32, ps::kernel::Psq4GemmOutputDType::F32,
        c.rows, c.N, c.k_padded, c.wss_frag, c.cstride, c.as_floats * 4, c.N, stream);
    if (hf != hipSuccess) {
        std::printf("FAIL: %s f32 launch: %s\n", label, hipGetErrorString(hf));
        return false;
    }
    hipError_t hb = ps::kernel::launch_gemm_psq4_w4a8_wmma(
        config, b.d_wc, b.d_ws, b.d_ac, b.d_as, b.d_obf16, ps::kernel::Psq4GemmOutputDType::BF16,
        c.rows, c.N, c.k_padded, c.wss_frag, c.cstride, c.as_floats * 4, c.N, stream);
    if (hb != hipSuccess) {
        std::printf("FAIL: %s bf16 launch: %s\n", label, hipGetErrorString(hb));
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: %s sync\n", label);
        return false;
    }
    std::vector<float> got32(static_cast<size_t>(c.rows) * c.N, 0.f);
    std::vector<uint16_t> gotbf16(static_cast<size_t>(c.rows) * c.N, 0);
    if (hipMemcpy(got32.data(), b.d_of32, out_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(gotbf16.data(), b.d_obf16, gotbf16.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: %s d2h\n", label);
        return false;
    }
    bool f32 = f32_ok(got32, c.ref, c.cond);
    bool bf16 = bf16_ok(gotbf16, c.ref, c.cond);
    const bool ok = f32 && bf16;
    if (!ok) {
        double maxad = 0; size_t bad_i = 0;
        for (size_t i = 0; i < got32.size(); ++i) {
            const double ad =
                std::fabs(static_cast<double>(got32[i]) - static_cast<double>(c.ref[i]));
            if (ad > maxad) { maxad = ad; bad_i = i; }
        }
        std::fprintf(stderr, "DBG_FAIL %s: maxad=%.6g cond_bad=%.6g got=%.6g ref=%.6g\n",
                     label, maxad, (double)c.cond[bad_i], (double)got32[bad_i],
                     (double)c.ref[bad_i]);
        std::fflush(stderr);
    }
    check(ok, label);
    return ok;
}

// WMMA C-mapping / lane mapping sanity: 16x16, K=32, all scales = 1, K16..31 = 0.
// Weights and activations use distinct, non-symmetric deterministic patterns so
// any row/column transposition or 8*k_group+j mis-mapping changes the result.
bool run_layout_sanity(ps::gpu::GpuArena& arena, hipStream_t stream) {
    Case c;
    c.rows = 16; c.N = 16; c.k_padded = 32;
    Strides s = mk_strides(32);
    c.wcp = s.wcp; c.wss = s.wss; c.wss_frag = s.wss_frag; c.cstride = s.cstride; c.as_floats = s.as_floats;
    c.wc.assign(static_cast<size_t>(16) * s.wcp, 0);
    for (uint32_t o = 0; o < 16; ++o) {
        for (uint32_t k = 0; k < 16; ++k) {
            const int w = static_cast<int>((o + k) % 7u) - 3;
            c.wc[static_cast<size_t>(o) * s.wcp + k] =
                static_cast<uint8_t>(int_to_fp4_nibble(w));
        }
    }
    c.ws.assign(static_cast<size_t>(16) * s.wss, 0);
    for (uint32_t o = 0; o < 16; ++o) {
        c.ws[static_cast<size_t>(o) * s.wss + 0] = 0x80;
        c.ws[static_cast<size_t>(o) * s.wss + 1] = 0x3F;
    }
    c.ac.assign(static_cast<size_t>(16) * s.cstride, 0);
    for (uint32_t m = 0; m < 16; ++m) {
        for (uint32_t k = 0; k < 16; ++k) {
            const int a = static_cast<int>((m * 2u + k) % 5u) - 2;
            c.ac[static_cast<size_t>(m) * s.cstride + k] = static_cast<int8_t>(a);
        }
    }
    c.as.assign(static_cast<size_t>(16) * s.as_floats, 1.0f);
    cpu_ref(c.wc, c.ws, c.ac, c.as, 16, 16, 32, s.wcp, s.wss, s.cstride, s.as_floats, c.ref, c.cond);

    const size_t out_bytes = 16u * 16u * 4u;
    const std::vector<uint8_t> pwc = fp4_frag_pack(c.wc, c.N, c.k_padded);
    const std::vector<int8_t> ac_frag = act_frag_pack(c.ac, 16u, c.cstride);
    auto wcr = arena.allocate_aligned(pwc.size(), 16);
    const std::vector<uint8_t> pws = ws_frag_pack(c.ws, c.N, c.k_padded, c.wss);
    auto wsr = arena.allocate_aligned(pws.size(), 16);
    auto acr = arena.allocate_aligned(ac_frag.size(), 16);
    auto asr = arena.allocate_aligned(c.as.size() * 4, 16);
    auto of32r = arena.allocate_aligned(out_bytes, 16);
    if (!wcr.ok() || !wsr.ok() || !acr.ok() || !asr.ok() || !of32r.ok())
        return false;
    uint8_t* d_wc = static_cast<uint8_t*>(wcr.release().data());
    uint8_t* d_ws = static_cast<uint8_t*>(wsr.release().data());
    int8_t* d_ac = static_cast<int8_t*>(acr.release().data());
    float* d_as = static_cast<float*>(asr.release().data());
    float* d_out = static_cast<float*>(of32r.release().data());
    if (hipMemcpy(d_wc, pwc.data(), pwc.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_ws, pws.data(), pws.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_ac, ac_frag.data(), ac_frag.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_as, c.as.data(), c.as.size() * 4, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    hipError_t hf = ps::kernel::launch_gemm_psq4_w4a8_wmma(
        psq4_config_for(16, 16, 32),
        d_wc, d_ws, d_ac, d_as, d_out, ps::kernel::Psq4GemmOutputDType::F32,
        16, 16, 32, s.wss_frag, s.cstride, s.as_floats * 4, 16, stream);
    if (hf != hipSuccess)
        return false;
    if (hipStreamSynchronize(stream) != hipSuccess)
        return false;
    std::vector<float> got(256, 0.f);
    if (hipMemcpy(got.data(), d_out, out_bytes, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    bool ok = true;
    for (int i = 0; i < 256; ++i) {
        if (std::fabs(got[i] - c.ref[i]) > 1e-4f) {
            ok = false;
            break;
        }
    }
    check(ok, "layout_sanity 16x16 K=32 integer-exact");
    return ok;
}

}  // namespace

int main() {
    std::printf("=== PSQ4 (CB10 + BF16/32) W4A8 WMMA GEMM Tests ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = ps::gpu::GpuArena::create(0, 1024 * 1024 * 1024);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    run_layout_sanity(arena, stream);

    // scale geometry: ws0 != ws1, as != 1, both K halves non-zero
    {
        Case c;
        gen_random(16, 16, 32, false, 1, 1, 1, c);
        char label[128];
        std::snprintf(label, sizeof(label), "scale_geometry rows=16 N=16 K=32");
        run_case(c, arena, stream, label);
    }

    // direct matrix (representative, not full product)
    const uint32_t rows_list[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128};
    const uint32_t N_list[] = {1, 15, 16, 17, 31, 32, 33, 64, 256, 1024};
    const uint32_t K_list[] = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 2560, 9216};
    for (uint32_t rows : rows_list) {
        for (uint32_t N : N_list) {
            for (uint32_t k : K_list) {
                if (static_cast<size_t>(rows) * N * k > 2000000u) continue;
                Case c;
                gen_random(rows, N, k, false, 1, 1, 1, c);
                char label[128];
                std::snprintf(label, sizeof(label), "f32+bf16 rows=%u N=%u K=%u", rows, N, k);
                run_case(c, arena, stream, label);
            }
        }
    }

    // negative decoded FP4 + negative A8 (covered by random nibbles/codes)
    // zero activation group (block 1 zeroed)
    {
        Case c;
        gen_random(2, 17, 64, true, 1, 1, 1, c);
        char label[128];
        std::snprintf(label, sizeof(label), "zero_block1 rows=2 N=17 K=64");
        run_case(c, arena, stream, label);
    }
    // weight scale0 != scale1 across a wider shape
    {
        Case c;
        gen_random(33, 65, 65, false, 1, 1, 1, c);
        char label[128];
        std::snprintf(label, sizeof(label), "ws_neq rows=33 N=65 K=65");
        run_case(c, arena, stream, label);
    }
    // actual Qwen-shaped K with small rows/N tail
    for (uint32_t k : {2560u, 9216u}) {
        for (uint32_t N : {17u, 256u, 1024u}) {
            for (uint32_t rows : {1u, 15u, 17u}) {
                Case c;
                gen_random(rows, N, k, false, 1, 1, 1, c);
                char label[128];
                std::snprintf(label, sizeof(label), "qwenK rows=%u N=%u K=%u", rows, N, k);
                run_case(c, arena, stream, label);
            }
        }
    }

    // Qwen3.8-27B TP=2 local geometries (optimized selector allowlist).
    {
        const uint32_t tp2_n[] = {8704u, 5120u, 3072u, 5120u, 512u, 5120u};
        const uint32_t tp2_k[] = {5120u, 8704u, 5120u, 3072u, 5120u, 5120u};
        for (std::size_t i = 0; i < 6; ++i) {
            Case c;
            gen_random(1, tp2_n[i], tp2_k[i], false, 1, 1, 1, c);
            char label[128];
            std::snprintf(label, sizeof(label), "tp2 rows=1 N=%u K=%u", tp2_n[i], tp2_k[i]);
            run_case(c, arena, stream, label);
        }
        Case c;
        gen_random(64, 8704, 5120, false, 1, 1, 1, c);
        char label[128];
        std::snprintf(label, sizeof(label), "tp2 rows=64 N=8704 K=5120");
        run_case(c, arena, stream, label);
    }

    // production 2D prefill dispatch (rows>=512, N>=512, k%64==0)
    for (uint32_t cfg : {0u, 1u, 2u}) {
        const uint32_t rows = cfg == 2u ? 1024u : 512u;
        const uint32_t N = cfg == 0u ? 512u : (cfg == 1u ? 1024u : 2048u);
        const uint32_t k = cfg == 0u ? 256u : (cfg == 1u ? 2560u : 256u);
        Case c;
        gen_random(rows, N, k, false, 1, 1, 1, c);
        char label[128];
        std::snprintf(label, sizeof(label), "disp2d rows=%u N=%u K=%u", rows, N, k);
        run_case(c, arena, stream, label);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
