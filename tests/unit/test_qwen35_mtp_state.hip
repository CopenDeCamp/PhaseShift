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
using ps::bf16_t;
namespace q35rt = ps::qwen35::runtime;

std::vector<float> pattern_k(const Geometry& g, uint32_t t) {
    std::vector<float> k(g.kv_features());
    for (uint32_t h = 0; h < g.kv_heads; ++h) {
        for (uint32_t i = 0; i < g.head_dim; ++i) {
            k[h * g.head_dim + i] =
                1.5f * static_cast<float>(h + 1u) + 0.017f * static_cast<float>(t * 9 + h * 3 + i);
        }
    }
    return k;
}

std::vector<float> pattern_v(const Geometry& g, uint32_t t) {
    std::vector<float> v(g.kv_features());
    for (uint32_t h = 0; h < g.kv_heads; ++h) {
        for (uint32_t i = 0; i < g.head_dim; ++i) {
            v[h * g.head_dim + i] =
                -2.0f * static_cast<float>(h + 1u) - 0.023f * static_cast<float>(t * 7 + h * 5 + i);
        }
    }
    return v;
}

std::vector<float> pattern_q(const Geometry& g, uint32_t t) {
    std::vector<float> q(g.q_features());
    for (uint32_t h = 0; h < g.q_heads; ++h) {
        for (uint32_t i = 0; i < g.head_dim; ++i) {
            q[h * g.head_dim + i] = 0.75f + 0.05f * static_cast<float>((t + 2 * h + i) % 6);
        }
    }
    return q;
}

std::vector<std::uint16_t> bits_of(const std::vector<float>& v) {
    return ps::mtp_test::to_bits(v);
}

struct SeqResult {
    std::vector<std::vector<float>> outputs;
    std::vector<float> canonical_k;
    std::vector<float> canonical_v;
};

bool run_serial(KvHarness& h, const Geometry& g, uint32_t steps, uint32_t pattern_offset,
                uint32_t position_offset, SeqResult& out, std::string& err) {
    out.outputs.clear();
    for (uint32_t t = 0; t < steps; ++t) {
        std::vector<float> o;
        if (!h.run(pattern_k(g, t + pattern_offset), pattern_v(g, t + pattern_offset),
                   pattern_q(g, t + pattern_offset), 1u, t, position_offset + t, o, err)) {
            return false;
        }
        out.outputs.push_back(std::move(o));
    }
    out.canonical_k = h.dump_k(steps);
    out.canonical_v = h.dump_v(steps);
    return true;
}

bool run_chunks(KvHarness& h, const Geometry& g, const std::vector<uint32_t>& chunks,
                uint32_t pattern_offset, uint32_t position_offset, SeqResult& out,
                std::string& err) {
    out.outputs.clear();
    uint32_t kv = 0;
    for (uint32_t c : chunks) {
        std::vector<float> k, v, q;
        for (uint32_t i = 0; i < c; ++i) {
            const uint32_t t = kv + i;
            auto ki = pattern_k(g, t + pattern_offset);
            auto vi = pattern_v(g, t + pattern_offset);
            auto qi = pattern_q(g, t + pattern_offset);
            k.insert(k.end(), ki.begin(), ki.end());
            v.insert(v.end(), vi.begin(), vi.end());
            q.insert(q.end(), qi.begin(), qi.end());
        }
        std::vector<float> o;
        if (!h.run(k, v, q, c, kv, position_offset + kv, o, err)) return false;
        for (uint32_t i = 0; i < c; ++i) {
            out.outputs.push_back(std::vector<float>(
                o.begin() + static_cast<std::ptrdiff_t>(i) * g.q_features(),
                o.begin() + static_cast<std::ptrdiff_t>(i + 1u) * g.q_features()));
        }
        kv += c;
    }
    out.canonical_k = h.dump_k(kv);
    out.canonical_v = h.dump_v(kv);
    return true;
}

bool same_floats(const std::vector<float>& a, const std::vector<float>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(static_cast<double>(a[i]) - b[i]) > tol) return false;
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
    std::string err;

    {
        q35rt::MtpKvState uninit;
        check(!q35rt::mtp_kv_reset(uninit, stream).ok(), "reset on uninitialized state fails");
        check(!q35rt::mtp_kv_reserve(uninit, 1u).ok(), "reserve on uninitialized state fails");
        check(!q35rt::mtp_kv_physical_slot(uninit, 0u).ok(),
              "physical slot on uninitialized state fails");
        bf16_t buf[8];
        check(!q35rt::mtp_kv_dump_canonical(uninit, 0u, buf, buf, stream).ok(),
              "dump on uninitialized state fails");
        check(q35rt::mtp_kv_shutdown(uninit).ok(), "shutdown on uninitialized state is safe");
    }

    {
        q35rt::MtpKvConfig bad;
        bad.num_pages = 0u;
        check(!q35rt::create_mtp_kv_state(arena, bad, 0u, stream).ok(),
              "create rejects zero geometry");
        q35rt::MtpKvConfig fp8;
        fp8.dtype = ps::qwen35::KVCacheDType::FP8_E4M3;
        check(!q35rt::create_mtp_kv_state(arena, fp8, 0u, stream).ok(),
              "create rejects non-BF16 dtype in Gate 2");
    }

    {
        q35rt::MtpKvConfig cfg;
        cfg.num_pages = 3u;
        cfg.page_tokens = 4u;
        cfg.kv_heads = 2u;
        cfg.head_dim = 4u;
        auto st_res = q35rt::create_mtp_kv_state(arena, cfg, 7u, stream);
        check(st_res.ok(), "create raw state for capacity test");
        if (st_res.ok()) {
            q35rt::MtpKvState st = st_res.release();
            check(st.capacity_tokens == 12u, "capacity == num_pages * page_tokens");
            check(st.sequence_id == 7u, "sequence id preserved");
            check(!q35rt::mtp_kv_reserve(st, 0u).ok(), "reserve(0) rejected");
            check(q35rt::mtp_kv_reserve(st, 12u).ok(), "reserve to capacity accepted");
            check(st.logical_length == 12u, "logical length == capacity");
            check(!q35rt::mtp_kv_reserve(st, 1u).ok(), "reserve beyond capacity rejected");
            check(!q35rt::mtp_kv_physical_slot(st, 12u).ok(),
                  "physical slot beyond logical length rejected");
            auto slot0 = q35rt::mtp_kv_physical_slot(st, 0u);
            auto slot4 = q35rt::mtp_kv_physical_slot(st, 4u);
            auto slot8 = q35rt::mtp_kv_physical_slot(st, 8u);
            check(slot0.ok() && slot4.ok() && slot8.ok(), "physical slot mapping resolvable");
            if (slot0.ok() && slot4.ok() && slot8.ok()) {
                check(slot0.value() != slot4.value() && slot4.value() != slot8.value(),
                      "physical slots differ per block");
                check(slot0.value() / 4u != slot4.value() / 4u,
                      "block id changes at block boundary");
            }
            bf16_t kbuf[12 * 8];
            bf16_t vbuf[12 * 8];
            check(!q35rt::mtp_kv_dump_canonical(st, 13u, kbuf, vbuf, stream).ok(),
                  "dump beyond logical length rejected");
            check(q35rt::mtp_kv_reset(st, stream).ok(), "reset accepted");
            check(st.logical_length == 0u, "logical length reset to 0");
            check(q35rt::mtp_kv_reserve(st, 4u).ok(), "reserve after reset accepted");
            check(q35rt::mtp_kv_shutdown(st).ok(), "shutdown accepted");
            check(!q35rt::mtp_kv_reserve(st, 1u).ok(), "reserve after shutdown rejected");
        }
    }

    Geometry geom;
    KvHarness h;
    if (!h.init(arena, stream, geom, err)) {
        std::printf("FAIL: harness init: %s\n", err.c_str());
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        arena.shutdown();
        return 1;
    }

    {
        h.poison(70.0f, -60.0f);
        SeqResult serial;
        if (!run_serial(h, geom, 8u, 0u, 0u, serial, err)) {
            std::printf("FAIL: serial run: %s\n", err.c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }

        std::vector<std::vector<float>> kh;
        std::vector<std::vector<float>> vh;
        for (uint32_t t = 0; t < 8u; ++t) {
            kh.push_back(ps::mtp_test::to_f32(bits_of(pattern_k(geom, t))));
            vh.push_back(ps::mtp_test::to_f32(bits_of(pattern_v(geom, t))));
        }
        bool history_ok = true;
        for (uint32_t t = 0; t < 8u; ++t) {
            std::vector<std::vector<float>> kh_t(kh.begin(), kh.begin() + t + 1);
            std::vector<std::vector<float>> vh_t(vh.begin(), vh.begin() + t + 1);
            const auto ref = ps::mtp_test::reference_attention(
                geom, ps::mtp_test::to_f32(bits_of(pattern_q(geom, t))), kh_t, vh_t);
            if (!same_floats(serial.outputs[t], ref, 5e-3)) history_ok = false;
        }
        check(history_ok, "serial 8-step attention matches reference");

        h.poison(70.0f, -60.0f);
        check(q35rt::mtp_kv_reset(h.state(), stream).ok(), "reset sequence for chunk test");
        check(h.state().logical_length == 0u, "logical length 0 after reset");
        SeqResult chunked;
        if (!run_chunks(h, geom, {3u, 2u, 3u}, 0u, 0u, chunked, err)) {
            std::printf("FAIL: chunked run: %s\n", err.c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        bool out_match = chunked.outputs.size() == serial.outputs.size();
        if (out_match) {
            for (std::size_t i = 0; i < serial.outputs.size(); ++i) {
                if (!same_floats(chunked.outputs[i], serial.outputs[i], 5e-3)) out_match = false;
            }
        }
        check(out_match, "chunked [3,2,3] per-row outputs match serial");
        check(same_floats(chunked.canonical_k, serial.canonical_k, 0.0),
              "chunked canonical K equals serial (bit-exact)");
        check(same_floats(chunked.canonical_v, serial.canonical_v, 0.0),
              "chunked canonical V equals serial (bit-exact)");

        h.poison(70.0f, -60.0f);
        check(q35rt::mtp_kv_reset(h.state(), stream).ok(), "reset sequence for single-chunk test");
        SeqResult one_chunk;
        if (!run_chunks(h, geom, {8u}, 0u, 0u, one_chunk, err)) {
            std::printf("FAIL: one-chunk run: %s\n", err.c_str());
            ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
            arena.shutdown();
            return 1;
        }
        check(same_floats(one_chunk.canonical_k, serial.canonical_k, 0.0),
              "single chunk [8] canonical K equals serial");
        bool one_ok = one_chunk.outputs.size() == serial.outputs.size();
        if (one_ok) {
            for (std::size_t i = 0; i < serial.outputs.size(); ++i) {
                if (!same_floats(one_chunk.outputs[i], serial.outputs[i], 5e-3)) one_ok = false;
            }
        }
        check(one_ok, "single chunk [8] per-row outputs match serial");
    }

    {
        h.poison(70.0f, -60.0f);
        check(q35rt::mtp_kv_reset(h.state(), stream).ok(), "reset before contamination test");
        SeqResult first;
        if (!run_serial(h, geom, 6u, 0u, 0u, first, err)) {
            check(false, "contamination first run: " + err);
        }
        h.poison(70.0f, -60.0f);
        check(q35rt::mtp_kv_reset(h.state(), stream).ok(), "reset after first run");
        check(h.state().logical_length == 0u, "logical length 0 after second reset");
        SeqResult second;
        if (!run_serial(h, geom, 4u, 50u, 0u, second, err)) {
            check(false, "contamination second run: " + err);
        }
        std::vector<std::vector<float>> kh;
        std::vector<std::vector<float>> vh;
        for (uint32_t t = 0; t < 4u; ++t) {
            kh.push_back(ps::mtp_test::to_f32(bits_of(pattern_k(geom, t + 50u))));
            vh.push_back(ps::mtp_test::to_f32(bits_of(pattern_v(geom, t + 50u))));
        }
        bool clean = true;
        for (uint32_t t = 0; t < 4u; ++t) {
            std::vector<std::vector<float>> kh_t(kh.begin(), kh.begin() + t + 1);
            std::vector<std::vector<float>> vh_t(vh.begin(), vh.begin() + t + 1);
            const auto ref = ps::mtp_test::reference_attention(
                geom, ps::mtp_test::to_f32(bits_of(pattern_q(geom, t + 50u))), kh_t, vh_t);
            if (!same_floats(second.outputs[t], ref, 5e-3)) clean = false;
        }
        check(clean, "reset removes old KV: second sequence matches fresh reference");
        bool ref_has_old = false;
        for (uint32_t t = 0; t < 4u; ++t) {
            if (!same_floats(second.outputs[t], first.outputs[t], 1e-2)) ref_has_old = true;
        }
        check(ref_has_old, "second sequence differs from first (negative control)");
        check(second.canonical_k.size() == 4u * geom.kv_features(),
              "second sequence canonical K length after reset");
    }

    {
        for (uint32_t offset : {0u, 17u, 127u}) {
            h.poison(70.0f, -60.0f);
            check(q35rt::mtp_kv_reset(h.state(), stream).ok(), "reset for position offset test");
            SeqResult r;
            if (!run_serial(h, geom, 3u, 0u, offset, r, err)) {
                check(false, "position offset run: " + err);
                continue;
            }
            const auto dk = h.dump_k(3u);
            bool ok = true;
            for (uint32_t t = 0; t < 3u; ++t) {
                const auto expect = ps::mtp_test::to_f32(bits_of(pattern_k(geom, t)));
                for (uint32_t i = 0; i < geom.kv_features(); ++i) {
                    if (dk[t * geom.kv_features() + i] != expect[i]) ok = false;
                }
            }
            check(ok, "position offset " + std::to_string(offset) +
                          " writes KV at kv_index (not rope position)");
        }
    }

    {
        KvHarness a;
        KvHarness b;
        KvHarness a_ref;
        KvHarness b_ref;
        bool init_ok = a.init(arena, stream, geom, err) && b.init(arena, stream, geom, err) &&
                       a_ref.init(arena, stream, geom, err) && b_ref.init(arena, stream, geom, err);
        check(init_ok, "two-sequence harnesses init");
        if (init_ok) {
            a.poison(70.0f, -60.0f);
            b.poison(70.0f, -60.0f);
            const uint32_t steps = 5u;
            std::vector<std::vector<float>> a_out;
            std::vector<std::vector<float>> b_out;
            std::string ierr;
            for (uint32_t t = 0; t < steps; ++t) {
                std::vector<float> oa, ob;
                if (!a.run(pattern_k(geom, t), pattern_v(geom, t), pattern_q(geom, t), 1u, t, t,
                           oa, ierr) ||
                    !b.run(pattern_k(geom, t + 90u), pattern_v(geom, t + 90u),
                           pattern_q(geom, t + 90u), 1u, t, t, ob, ierr)) {
                    check(false, "interleaved run: " + ierr);
                    break;
                }
                a_out.push_back(std::move(oa));
                b_out.push_back(std::move(ob));
            }
            SeqResult a_solo;
            SeqResult b_solo;
            const bool solo_ok =
                run_serial(a_ref, geom, steps, 0u, 0u, a_solo, ierr) &&
                run_serial(b_ref, geom, steps, 90u, 0u, b_solo, ierr);
            check(solo_ok, "solo reference sequences run");
            if (solo_ok && a_out.size() == steps && b_out.size() == steps) {
                bool a_ok = true;
                bool b_ok = true;
                for (uint32_t t = 0; t < steps; ++t) {
                    if (!same_floats(a_out[t], a_solo.outputs[t], 5e-3)) a_ok = false;
                    if (!same_floats(b_out[t], b_solo.outputs[t], 5e-3)) b_ok = false;
                }
                check(a_ok, "interleaved sequence A matches solo");
                check(b_ok, "interleaved sequence B matches solo");
                check(same_floats(a.dump_k(steps), a_solo.canonical_k, 0.0),
                      "interleaved A canonical K unchanged");
                check(same_floats(b.dump_k(steps), b_solo.canonical_k, 0.0),
                      "interleaved B canonical K unchanged");
            }
        }
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    arena.shutdown();

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) std::printf("QWEN35_MTP_STATE: PASS\n");
    return g_fail == 0 ? 0 : 1;
}
