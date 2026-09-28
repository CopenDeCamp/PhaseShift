#include "support/target_lm_proxy_harness.h"
#include "support/dflash2_int2_test_common.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

struct Rng {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<uint32_t>(state >> 11);
    }
    uint32_t below(uint32_t n) { return next() % (n == 0u ? 1u : n); }
};

void fill_episode(std::vector<int32_t>& tokens, uint32_t category, uint32_t vocab, Rng& rng) {
    const uint32_t n = static_cast<uint32_t>(tokens.size());
    switch (category % 5u) {
        case 0u:
            for (uint32_t i = 0; i < n; ++i)
                tokens[i] = static_cast<int32_t>(100u + rng.below(vocab - 200u));
            break;
        case 1u: {
            const int32_t pool[6] = {1000, 2000, 3000, 4000, 5000, 6000};
            for (uint32_t i = 0; i < n; ++i) tokens[i] = pool[rng.below(6u)];
            break;
        }
        case 2u:
            for (uint32_t i = 0; i < n; ++i)
                tokens[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
            break;
        case 3u:
            for (uint32_t i = 0; i < n; ++i)
                tokens[i] = static_cast<int32_t>(100u + ((i / 8u) * 131u + 7u) % 5000u);
            break;
        default:
            for (uint32_t i = 0; i < n; ++i)
                tokens[i] = (i & 1u) ? static_cast<int32_t>(100u + rng.below(50000u))
                                     : static_cast<int32_t>(100u + (i % 97u));
            break;
    }
}

struct Stats {
    uint64_t rows = 0u;
    uint64_t candidate_hits = 0u;
    uint64_t certified = 0u;
    uint64_t fallback = 0u;
    uint64_t certified_mismatch = 0u;
    uint64_t final_mismatch = 0u;
    std::vector<float> gaps;
    void merge(const Stats& o) {
        rows += o.rows;
        candidate_hits += o.candidate_hits;
        certified += o.certified;
        fallback += o.fallback;
        certified_mismatch += o.certified_mismatch;
        final_mismatch += o.final_mismatch;
        gaps.insert(gaps.end(), o.gaps.begin(), o.gaps.end());
    }
};

void print_pool_stats(const char* tag, uint32_t pool, float abs_margin, float rel_margin,
                      const Stats& s) {
    std::vector<float> g = s.gaps;
    std::sort(g.begin(), g.end());
    const auto pick = [&g](double q) {
        if (g.empty()) return 0.0f;
        const std::size_t idx =
            static_cast<std::size_t>(q * static_cast<double>(g.size() - 1u) + 0.5);
        return g[idx];
    };
    const double recall =
        s.rows > 0u ? static_cast<double>(s.candidate_hits) / static_cast<double>(s.rows) : 0.0;
    std::printf(
        "TARGET_LM_PROXY_REAL tag=%s pool=%u margin_abs=%.1e margin_rel=%.1e decisions=%llu "
        "certified=%llu cert_rate=%.4f fallback=%llu candidate_recall=%.6f "
        "certified_mismatch=%llu final_mismatch=%llu gap_min=%.6f gap_p50=%.6f gap_p90=%.6f "
        "gap_p99=%.6f\n",
        tag, pool, static_cast<double>(abs_margin), static_cast<double>(rel_margin),
        static_cast<unsigned long long>(s.rows),
        static_cast<unsigned long long>(s.certified),
        s.rows > 0u ? static_cast<double>(s.certified) / static_cast<double>(s.rows) : 0.0,
        static_cast<unsigned long long>(s.fallback), recall,
        static_cast<unsigned long long>(s.certified_mismatch),
        static_cast<unsigned long long>(s.final_mismatch),
        static_cast<double>(g.empty() ? 0.0f : g.front()), static_cast<double>(pick(0.50)),
        static_cast<double>(pick(0.90)), static_cast<double>(pick(0.99)));
}

}  // namespace

int main() {
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (target_dir == nullptr || target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2_TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    const uint32_t rows_target =
        std::getenv("PHASESHIFT_TARGET_LM_ROWS") != nullptr
            ? static_cast<uint32_t>(std::strtoul(std::getenv("PHASESHIFT_TARGET_LM_ROWS"),
                                                 nullptr, 10))
            : 12000u;
    const uint32_t chunk = 128u;
    const uint32_t sub = 8u;
    const uint32_t episode_len = 2048u;
    const uint32_t max_pool = 127u;
    const float default_abs = 1e-3f;
    const float default_rel = 1e-5f;

    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result = ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
    const uint32_t vocab = static_cast<uint32_t>(model.weights().lm_head.rows);
    const float eps = tc.rms_norm_eps;

    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t max_seq = episode_len + 32u;
    const uint32_t num_pages = ceil_div(max_seq, page_tokens) + 8u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 2u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 2u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL pool create\n");
        return 1;
    }
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = chunk;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = chunk;
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    if (!exec_result.ok()) {
        std::printf("FAIL executor create: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Executor ex = exec_result.release();

    target_lm_proxy::Proxy proxy;
    std::string perr;
    if (!proxy.init(model.weights().lm_head, sub, max_pool, stream, perr)) {
        std::printf("FAIL proxy init: %s\n", perr.c_str());
        return 1;
    }

    ps::bf16_t* normed = nullptr;
    if (hipMalloc(&normed, static_cast<std::size_t>(chunk) * hidden * sizeof(ps::bf16_t)) !=
        hipSuccess) {
        std::printf("FAIL normed alloc\n");
        return 1;
    }

    const ps::gpu::Tensor& fw = model.weights().final_norm_weight;
    const ps::kernel::RmsNormDataType wdt =
        fw.element_size() == 4u ? ps::kernel::RmsNormDataType::F32
                                : ps::kernel::RmsNormDataType::BF16;

    const uint32_t pools[] = {16u, 24u, 32u, 48u, 64u, 80u, 127u};
    const float margin_abs[] = {0.0f, 1e-5f, 1e-4f, 1e-3f, 1e-2f};
    const float margin_rel[] = {0.0f, 1e-7f, 1e-6f, 1e-5f, 1e-4f};

    Stats pool_stats[sizeof(pools) / sizeof(pools[0])];
    Stats margin_stats[sizeof(margin_abs) / sizeof(margin_abs[0])];
    Stats coarse_stats;

    std::vector<int32_t> tokens(episode_len);
    std::vector<int32_t> oracle(chunk);
    std::vector<int32_t> best_ids(sub);
    std::vector<uint8_t> certified(sub);
    std::vector<float> gaps(sub);
    std::vector<int32_t> cand(static_cast<std::size_t>(sub) * max_pool);

    Rng rng;
    uint32_t category = 0u;
    uint64_t total_rows = 0u;
    const uint32_t topn_pool = max_pool + 1u;
    const bool diag = std::getenv("PHASESHIFT_TARGET_LM_DIAG") != nullptr;
    if (diag) {
        std::vector<float> err(vocab, 0.0f);
        (void)hipMemcpy(err.data(), proxy.error_l2, vocab * sizeof(float),
                        hipMemcpyDeviceToHost);
        double sum = 0.0;
        float emin = err[0], emax = err[0];
        for (uint32_t i = 0u; i < vocab; ++i) {
            sum += err[i];
            emin = std::min(emin, err[i]);
            emax = std::max(emax, err[i]);
        }
        std::printf("DIAG codebook=[%.6f %.6f %.6f %.6f] map0=%u error_l2 min=%.6g mean=%.6g "
                    "max=%.6g\n",
                    proxy.codebook.decoded[0], proxy.codebook.decoded[1],
                    proxy.codebook.decoded[2], proxy.codebook.decoded[3],
                    static_cast<uint32_t>(proxy.codebook.values[0]), emin,
                    sum / static_cast<double>(vocab), emax);
    }
    bool diag_first = diag;

    while (total_rows < rows_target) {
        auto seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, max_seq, stream);
        if (!seq_result.ok()) {
            std::printf("FAIL sequence create\n");
            return 1;
        }
        ps::qwen35::PagedSequenceState seq = seq_result.release();
        fill_episode(tokens, category, vocab, rng);
        ++category;

        uint32_t prefix = 0u;
        uint32_t offset = 0u;
        while (offset < episode_len && total_rows < rows_target) {
            const uint32_t n = std::min(chunk, episode_len - offset);
            std::vector<ps::qwen35::ScheduledRequest> requests;
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq;
            req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::PREFILL;
            req.token_begin = 0u;
            req.num_tokens = n;
            req.prefix_tokens = prefix;
            req.compute_logits = true;
            req.sample = true;
            req.num_output_rows = n;
            requests.push_back(req);
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = tokens.data() + offset;
            batch.requests = requests;
            batch.num_tokens = n;
            batch.num_requests = 1u;
            batch.num_decode_requests = 0u;
            batch.num_verify_requests = 0u;
            batch.num_prefill_requests = 1u;

            auto executed = ps::qwen35::execute_batch(ex, batch, stream);
            if (!executed.ok()) {
                std::printf("FAIL execute: %s\n", executed.status().message().c_str());
                return 1;
            }
            ps::qwen35::BatchExecutionOutput out = executed.release();
            (void)hipMemcpy(oracle.data(), out.sampled_tokens.data<int32_t>(),
                            static_cast<std::size_t>(n) * sizeof(int32_t),
                            hipMemcpyDeviceToHost);
            const hipError_t rerr = ps::kernel::launch_rmsnorm(
                out.final_hidden.data<void>(), ps::kernel::RmsNormDataType::BF16,
                fw.data<void>(), wdt, ps::kernel::RmsNormWeightLayout::PER_FEATURE,
                ps::kernel::RmsNormWeightMode::ONE_PLUS, normed,
                ps::kernel::RmsNormDataType::BF16, n, hidden, hidden, hidden, hidden, eps,
                stream);
            if (rerr != hipSuccess) {
                std::printf("FAIL rmsnorm\n");
                return 1;
            }
            (void)hipStreamSynchronize(stream);

            for (uint32_t s = 0u; s < n; s += sub) {
                const uint32_t m = std::min(sub, n - s);
                if (!proxy.prepare(normed + static_cast<std::size_t>(s) * hidden, m, hidden,
                                   stream, perr)) {
                    std::printf("FAIL prepare: %s\n", perr.c_str());
                    return 1;
                }
                if (diag_first && s == 0u) {
                    std::vector<float> tl(vocab), co(vocab), up(vocab);
                    (void)hipMemcpy(tl.data(), out.logits.data<float>(),
                                    static_cast<std::size_t>(vocab) * sizeof(float),
                                    hipMemcpyDeviceToHost);
                    (void)hipMemcpy(co.data(), proxy.coarse,
                                    static_cast<std::size_t>(vocab) * sizeof(float),
                                    hipMemcpyDeviceToHost);
                    (void)hipMemcpy(up.data(), proxy.upper,
                                    static_cast<std::size_t>(vocab) * sizeof(float),
                                    hipMemcpyDeviceToHost);
                    const int32_t want = oracle[0];
                    const auto argmax = [vocab](const std::vector<float>& v) {
                        uint32_t b = 0u;
                        for (uint32_t i = 1u; i < vocab; ++i)
                            if (v[i] > v[b]) b = i;
                        return static_cast<int32_t>(b);
                    };
                    uint32_t rank_c = 0u;
                    uint32_t rank_u = 0u;
                    for (uint32_t i = 0u; i < vocab; ++i) {
                        if (co[i] > co[static_cast<uint32_t>(want)]) ++rank_c;
                        if (up[i] > up[static_cast<uint32_t>(want)]) ++rank_u;
                    }
                    std::printf(
                        "DIAG row0 want=%d true_argmax=%d coarse_argmax=%d "
                        "coarse_argmax_hit=%d rank_coarse=%u rank_upper=%u "
                        "upper[want]=%.6g coarse[want]=%.6g true[want]=%.6g true_max=%.6g "
                        "upper_max=%.6g\n",
                        want, argmax(tl), argmax(co), argmax(co) == want ? 1 : 0, rank_c, rank_u,
                        static_cast<double>(up[static_cast<uint32_t>(want)]),
                        static_cast<double>(co[static_cast<uint32_t>(want)]),
                        static_cast<double>(tl[static_cast<uint32_t>(want)]),
                        static_cast<double>(tl[static_cast<uint32_t>(argmax(tl))]),
                        static_cast<double>(up[static_cast<uint32_t>(argmax(up))]));
                    diag_first = false;

                    {
                        const uint32_t kp = proxy.k_padded;
                        const std::size_t psq8_n =
                            ((static_cast<std::size_t>(vocab) + 15u) / 16u) * kp * 16u;
                        const std::size_t scale_n =
                            ((static_cast<std::size_t>(vocab) + 15u) / 16u) * (kp / 2u);
                        std::vector<uint8_t> h_psq8(psq8_n, 0u);
                        std::vector<uint16_t> h_scales(scale_n, 0u);
                        std::vector<uint8_t> h_act(
                            static_cast<std::size_t>(proxy.act_code_stride) *
                                ((static_cast<std::size_t>(sub) + 15u) / 16u) * 16u,
                            0u);
                        float as = 0.0f;
                        (void)hipMemcpy(h_psq8.data(),
                                        model.weights().lm_head.codes.data<uint8_t>(), psq8_n,
                                        hipMemcpyDeviceToHost);
                        (void)hipMemcpy(h_scales.data(),
                                        model.weights().lm_head.scales.data<uint8_t>(),
                                        scale_n * 2u, hipMemcpyDeviceToHost);
                        (void)hipMemcpy(h_act.data(), proxy.act_codes, h_act.size(),
                                        hipMemcpyDeviceToHost);
                        (void)hipMemcpy(&as, proxy.act_scales, sizeof(float),
                                        hipMemcpyDeviceToHost);
                        uint8_t map_table[256] = {};
                        uint32_t expand_table[256] = {};
                        ps::kernel::dflash2_int2_build_tables(proxy.codebook, map_table,
                                                              expand_table);
                        std::vector<double> a(kp, 0.0);
                        for (uint32_t k = 0u; k < kp; ++k) {
                            a[k] = static_cast<double>(::ps::quantization::fpx::e4m3_decode_f32(
                                       h_act[dflash2_int2_test::act_code_index(
                                           0u, k, proxy.act_code_stride)])) *
                                   static_cast<double>(as);
                        }
                        const auto analyze = [&](int32_t j, const char* tag) {
                            std::vector<double> d(kp, 0.0);
                            for (uint32_t k = 0u; k < kp; ++k) {
                                const uint8_t code = h_psq8[dflash2_int2_test::psq8_code_index(
                                    static_cast<uint32_t>(j), k, kp)];
                                const uint16_t sb =
                                    h_scales[dflash2_int2_test::psq8_scale_index(
                                        static_cast<uint32_t>(j), (k >> 5u) * 32u,
                                        proxy.scale_stride)];
                                const uint32_t u = static_cast<uint32_t>(sb) << 16;
                                const double ws =
                                    static_cast<double>(*reinterpret_cast<const float*>(&u));
                                const double exact = static_cast<double>(
                                    ::ps::quantization::fpx::e4m3_decode_f32(code));
                                const double proxyv = static_cast<double>(
                                    proxy.codebook.decoded[map_table[code]]);
                                d[k] = ws * (exact - proxyv);
                            }
                            double aa = 0.0, dd = 0.0, da = 0.0;
                            for (uint32_t k = 0u; k < kp; ++k) {
                                aa += a[k] * a[k];
                                dd += d[k] * d[k];
                                da += a[k] * d[k];
                            }
                            double group = 0.0;
                            const uint32_t ngroups = kp / 32u;
                            for (uint32_t g = 0u; g < ngroups; ++g) {
                                double ag = 0.0, dg = 0.0;
                                for (uint32_t k = g * 32u; k < (g + 1u) * 32u; ++k) {
                                    ag += a[k] * a[k];
                                    dg += d[k] * d[k];
                                }
                                group += std::sqrt(ag) * std::sqrt(dg);
                            }
                            std::printf(
                                "DIAG_BOUND %s j=%d ||A||=%.4f ||D||=%.6f global=%.4f "
                                "group=%.4f actual=%.6f ratio_global=%.2f ratio_group=%.2f\n",
                                tag, j, std::sqrt(aa), std::sqrt(dd), std::sqrt(aa) * std::sqrt(dd),
                                group, std::fabs(da),
                                std::fabs(da) > 0.0 ? (std::sqrt(aa) * std::sqrt(dd)) /
                                                          std::fabs(da)
                                                    : 0.0,
                                std::fabs(da) > 0.0 ? group / std::fabs(da) : 0.0);
                        };
                        analyze(want, "oracle");
                        analyze(argmax(co), "coarse_top1");
                    }
                }
                if (!proxy.topn(m, topn_pool, stream, perr)) {
                    std::printf("FAIL topn: %s\n", perr.c_str());
                    return 1;
                }
                for (std::size_t pi = 0u; pi < sizeof(pools) / sizeof(pools[0]); ++pi) {
                    const uint32_t pool = pools[pi];
                    if (!proxy.rerank_pool(m, pool, topn_pool, stream, perr) ||
                        !proxy.certify(m, pool, topn_pool, default_abs, default_rel, stream,
                                       perr)) {
                        std::printf("FAIL pool eval: %s\n", perr.c_str());
                        return 1;
                    }
                    (void)hipMemcpy(best_ids.data(), proxy.best_ids, m * sizeof(int32_t),
                                    hipMemcpyDeviceToHost);
                    (void)hipMemcpy(certified.data(), proxy.certified, m, hipMemcpyDeviceToHost);
                    (void)hipMemcpy(gaps.data(), proxy.gaps, m * sizeof(float),
                                    hipMemcpyDeviceToHost);
                    (void)hipMemcpy(cand.data(), proxy.cand_ids,
                                    static_cast<std::size_t>(m) * pool * sizeof(int32_t),
                                    hipMemcpyDeviceToHost);
                    Stats& st = pool_stats[pi];
                    for (uint32_t r = 0u; r < m; ++r) {
                        const int32_t want = oracle[s + r];
                        bool hit = false;
                        for (uint32_t c = 0u; c < pool && !hit; ++c)
                            hit = cand[static_cast<std::size_t>(r) * pool + c] == want;
                        st.rows += 1u;
                        if (hit) st.candidate_hits += 1u;
                        const int32_t final_token =
                            certified[r] != 0u ? best_ids[r] : want;
                        if (certified[r] != 0u) {
                            st.certified += 1u;
                            if (best_ids[r] != want) st.certified_mismatch += 1u;
                            st.gaps.push_back(gaps[r]);
                        } else {
                            st.fallback += 1u;
                        }
                        if (final_token != want) st.final_mismatch += 1u;
                    }
                }
                for (std::size_t mi = 0u; mi < sizeof(margin_abs) / sizeof(margin_abs[0]); ++mi) {
                    const uint32_t pool = 32u;
                    if (!proxy.rerank_pool(m, pool, topn_pool, stream, perr) ||
                        !proxy.certify(m, pool, topn_pool, margin_abs[mi], margin_rel[mi],
                                       stream, perr)) {
                        std::printf("FAIL margin eval: %s\n", perr.c_str());
                        return 1;
                    }
                    (void)hipMemcpy(best_ids.data(), proxy.best_ids, m * sizeof(int32_t),
                                    hipMemcpyDeviceToHost);
                    (void)hipMemcpy(certified.data(), proxy.certified, m, hipMemcpyDeviceToHost);
                    (void)hipMemcpy(gaps.data(), proxy.gaps, m * sizeof(float),
                                    hipMemcpyDeviceToHost);
                    Stats& st = margin_stats[mi];
                    for (uint32_t r = 0u; r < m; ++r) {
                        const int32_t want = oracle[s + r];
                        st.rows += 1u;
                        const int32_t final_token =
                            certified[r] != 0u ? best_ids[r] : want;
                        if (certified[r] != 0u) {
                            st.certified += 1u;
                            if (best_ids[r] != want) st.certified_mismatch += 1u;
                            st.gaps.push_back(gaps[r]);
                        } else {
                            st.fallback += 1u;
                        }
                        if (final_token != want) st.final_mismatch += 1u;
                    }
                }
                {
                    const uint32_t cpool = 32u;
                    if (!proxy.topn_on(proxy.coarse, m, cpool + 1u, stream, perr) ||
                        !proxy.rerank_pool(m, cpool, cpool + 1u, stream, perr) ||
                        !proxy.certify(m, cpool, cpool + 1u, 0.0f, 0.0f, stream, perr)) {
                        std::printf("FAIL coarse recall eval: %s\n", perr.c_str());
                        return 1;
                    }
                    (void)hipMemcpy(best_ids.data(), proxy.best_ids, m * sizeof(int32_t),
                                    hipMemcpyDeviceToHost);
                    (void)hipMemcpy(cand.data(), proxy.cand_ids,
                                    static_cast<std::size_t>(m) * cpool * sizeof(int32_t),
                                    hipMemcpyDeviceToHost);
                    for (uint32_t r = 0u; r < m; ++r) {
                        const int32_t want = oracle[s + r];
                        bool hit = false;
                        for (uint32_t c = 0u; c < cpool && !hit; ++c)
                            hit = cand[static_cast<std::size_t>(r) * cpool + c] == want;
                        coarse_stats.rows += 1u;
                        if (hit) coarse_stats.candidate_hits += 1u;
                        if (best_ids[r] == want) coarse_stats.certified += 1u;
                    }
                }
            }
            total_rows += n;
            prefix += n;
            offset += n;
        }
        (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    }

    int failed = 0;
    for (std::size_t pi = 0u; pi < sizeof(pools) / sizeof(pools[0]); ++pi) {
        print_pool_stats("pool", pools[pi], default_abs, default_rel, pool_stats[pi]);
        if (pool_stats[pi].certified_mismatch != 0u || pool_stats[pi].final_mismatch != 0u)
            failed = 1;
    }
    for (std::size_t mi = 0u; mi < sizeof(margin_abs) / sizeof(margin_abs[0]); ++mi) {
        print_pool_stats("margin", 32u, margin_abs[mi], margin_rel[mi], margin_stats[mi]);
        if (margin_stats[mi].certified_mismatch != 0u || margin_stats[mi].final_mismatch != 0u)
            failed = 1;
    }
    {
        const double recall = coarse_stats.rows > 0u
                                  ? static_cast<double>(coarse_stats.candidate_hits) /
                                        static_cast<double>(coarse_stats.rows)
                                  : 0.0;
        const double top1 = coarse_stats.rows > 0u
                                ? static_cast<double>(coarse_stats.certified) /
                                      static_cast<double>(coarse_stats.rows)
                                : 0.0;
        std::printf(
            "TARGET_LM_PROXY_COARSE decisions=%llu pool=32 candidate_recall=%.6f "
            "rerank_top1=%.6f\n",
            static_cast<unsigned long long>(coarse_stats.rows), recall, top1);
    }

    (void)hipFree(normed);
    proxy.shutdown();
    (void)ps::qwen35::executor_shutdown(ex);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    std::printf("test_target_lm_proxy_real_hidden: decisions=%llu failed=%d\n",
                static_cast<unsigned long long>(total_rows), failed);
    return failed == 0 ? 0 : 1;
}
