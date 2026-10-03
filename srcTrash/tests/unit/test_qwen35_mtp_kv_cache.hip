#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "support/mtp_kv_harness.h"

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

using ps::mtp_test::Geometry;
using ps::mtp_test::KvHarness;

std::vector<float> pattern_k(const Geometry& g, uint32_t t) {
    std::vector<float> k(g.kv_features());
    for (uint32_t h = 0; h < g.kv_heads; ++h) {
        for (uint32_t i = 0; i < g.head_dim; ++i) {
            k[h * g.head_dim + i] =
                2.0f * static_cast<float>(h + 1u) + 0.01f * static_cast<float>(t * 13 + h * 7 + i * 3);
        }
    }
    return k;
}

std::vector<float> pattern_v(const Geometry& g, uint32_t t) {
    std::vector<float> v(g.kv_features());
    for (uint32_t h = 0; h < g.kv_heads; ++h) {
        for (uint32_t i = 0; i < g.head_dim; ++i) {
            v[h * g.head_dim + i] =
                -3.0f * static_cast<float>(h + 1u) - 0.02f * static_cast<float>(t * 11 + h * 5 + i);
        }
    }
    return v;
}

std::vector<float> pattern_q(const Geometry& g, uint32_t t) {
    std::vector<float> q(g.q_features());
    for (uint32_t h = 0; h < g.q_heads; ++h) {
        for (uint32_t i = 0; i < g.head_dim; ++i) {
            q[h * g.head_dim + i] = 0.5f + 0.1f * static_cast<float>((t + h + i) % 5);
        }
    }
    return q;
}

bool close_abs(float act, double ref, double tol) { return std::fabs(act - ref) <= tol; }

bool slice_matches(const std::vector<float>& flat, uint32_t index, uint32_t features,
                   const std::vector<float>& row) {
    if (flat.size() < (index + 1u) * features || row.size() != features) return false;
    for (uint32_t i = 0; i < features; ++i) {
        if (flat[index * features + i] != row[i]) return false;
    }
    return true;
}

}  // namespace

int main() {
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    auto ar = ps::gpu::GpuArena::create(0, 256ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("SKIP: no GPU available (%s)\n", ar.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = ar.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream create\n");
        arena.shutdown();
        return 1;
    }

    Geometry geom;
    KvHarness h;
    std::string err;
    if (!h.init(arena, stream, geom, err)) {
        std::printf("FAIL: harness init: %s\n", err.c_str());
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }

    const uint32_t B = geom.page_tokens;
    const uint32_t N = 10u;
    h.poison(50.0f, -40.0f);

    std::vector<std::vector<float>> k_hist;
    std::vector<std::vector<float>> v_hist;
    std::vector<std::vector<float>> out_hist;
    std::vector<std::vector<float>> q_hist;

    for (uint32_t t = 0; t < N; ++t) {
        const std::vector<float> k = pattern_k(geom, t);
        const std::vector<float> v = pattern_v(geom, t);
        const std::vector<float> q = pattern_q(geom, t);
        std::vector<float> out;
        if (!h.run(k, v, q, 1u, t, t, out, err)) {
            std::printf("FAIL: run step %u: %s\n", t, err.c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        k_hist.push_back(ps::mtp_test::to_f32(ps::mtp_test::to_bits(k)));
        v_hist.push_back(ps::mtp_test::to_f32(ps::mtp_test::to_bits(v)));
        q_hist.push_back(ps::mtp_test::to_f32(ps::mtp_test::to_bits(q)));
        out_hist.push_back(out);
    }

    for (uint32_t t = 0; t < N; ++t) {
        const std::vector<std::vector<float>> kh(k_hist.begin(), k_hist.begin() + t + 1);
        const std::vector<std::vector<float>> vh(v_hist.begin(), v_hist.begin() + t + 1);
        const std::vector<float> ref =
            ps::mtp_test::reference_attention(geom, q_hist[t], kh, vh);
        bool ok = ref.size() == out_hist[t].size();
        double worst = 0.0;
        if (ok) {
            for (std::size_t i = 0; i < ref.size(); ++i) {
                const double d = std::fabs(static_cast<double>(out_hist[t][i]) - ref[i]);
                if (d > worst) worst = d;
                if (!close_abs(out_hist[t][i], ref[i], 5e-3)) ok = false;
            }
        }
        check(ok, "attention step " + std::to_string(t) + " matches full-history reference (worst=" +
                      std::to_string(worst) + ")");
        if (t >= 1u) {
            const std::vector<std::vector<float>> kh1 = {k_hist[t]};
            const std::vector<std::vector<float>> vh1 = {v_hist[t]};
            const std::vector<float> cur_only =
                ps::mtp_test::reference_attention(geom, q_hist[t], kh1, vh1);
            double diff = 0.0;
            for (std::size_t i = 0; i < cur_only.size() && i < out_hist[t].size(); ++i) {
                diff = std::max(diff, std::fabs(static_cast<double>(out_hist[t][i]) - cur_only[i]));
            }
            check(diff > 1e-2, "step " + std::to_string(t) +
                                   " output is not current-token-only (negative control)");
        }
    }

    {
        const std::vector<float> dk = h.dump_k(N);
        const std::vector<float> dv = h.dump_v(N);
        check(dk.size() == N * geom.kv_features(), "canonical K dump size");
        check(dv.size() == N * geom.kv_features(), "canonical V dump size");
        bool k_ok = true;
        bool v_ok = true;
        for (uint32_t t = 0; t < N; ++t) {
            if (!slice_matches(dk, t, geom.kv_features(), k_hist[t])) k_ok = false;
            if (!slice_matches(dv, t, geom.kv_features(), v_hist[t])) v_ok = false;
        }
        check(k_ok, "canonical K history bit-exact (per-token, per-head layout)");
        check(v_ok, "canonical V history bit-exact (per-token, per-head layout)");
        check(dk.size() >= geom.kv_features() && dk[0] != 50.0f,
              "poison overwritten at logical slot 0");
    }

    for (uint32_t idx : {B - 1u, B, B + 1u, 2u * B - 1u, 2u * B, 2u * B + 1u}) {
        if (idx >= N) continue;
        const std::vector<float> dk = h.dump_k(N);
        check(slice_matches(dk, idx, geom.kv_features(), k_hist[idx]),
              "block boundary slot " + std::to_string(idx) + " canonical K intact");
    }

    {
        const std::vector<std::vector<float>> kh1 = {k_hist[0]};
        const std::vector<std::vector<float>> vh1 = {v_hist[0]};
        const std::vector<float> ref = ps::mtp_test::reference_attention(geom, q_hist[0], kh1, vh1);
        bool gqa_ok = ref.size() == out_hist[0].size();
        if (gqa_ok) {
            for (std::size_t i = 0; i < ref.size(); ++i) {
                if (!close_abs(out_hist[0][i], ref[i], 1e-6)) gqa_ok = false;
            }
        }
        check(gqa_ok, "single-key attention equals V (GQA kv_head = q_head / group)");
    }

    {
        const uint32_t t = N;
        const std::vector<float> k = pattern_k(geom, t);
        const std::vector<float> v = pattern_v(geom, t);
        const std::vector<float> q = pattern_q(geom, t);
        std::vector<float> out;
        const uint32_t rope_position = 1000u;
        if (!h.run(k, v, q, 1u, t, rope_position, out, err)) {
            check(false, "position offset run: " + err);
        } else {
            const std::vector<float> dk = h.dump_k(N + 1u);
            check(slice_matches(dk, t, geom.kv_features(),
                                ps::mtp_test::to_f32(ps::mtp_test::to_bits(k))),
                  "rope position != kv_index: canonical K written at kv_index (offset 1000)");
            std::vector<std::vector<float>> kh = k_hist;
            std::vector<std::vector<float>> vh = v_hist;
            kh.push_back(ps::mtp_test::to_f32(ps::mtp_test::to_bits(k)));
            vh.push_back(ps::mtp_test::to_f32(ps::mtp_test::to_bits(v)));
            const std::vector<float> ref =
                ps::mtp_test::reference_attention(geom, ps::mtp_test::to_f32(ps::mtp_test::to_bits(q)),
                                                  kh, vh);
            bool ok = ref.size() == out.size();
            for (std::size_t i = 0; i < ref.size() && ok; ++i) {
                if (!close_abs(out[i], ref[i], 5e-3)) ok = false;
            }
            check(ok, "position offset run attention uses kv_index history, not rope position");
        }
    }

    check(h.state().logical_length == N + 1u, "logical length tracks appends");
    check(h.state().step_index >= 1u, "step index advanced");

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    arena.shutdown();

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) std::printf("QWEN35_MTP_KV_CACHE: PASS\n");
    return g_fail == 0 ? 0 : 1;
}
