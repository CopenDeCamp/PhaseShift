#include "support/dflash2_int2_test_common.h"

#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/quantization/fpx/e4m3.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using dflash2_int2_test::Rng;
using ps::kernel::Dflash2Int2Codebook;

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

std::size_t psq8_bytes(uint32_t rows, uint32_t k_padded) {
    return ((static_cast<std::size_t>(rows) + 15u) / 16u) *
           static_cast<std::size_t>(k_padded) * 16u;
}

struct Stats {
    uint64_t rows = 0u;
    uint64_t certified = 0u;
    uint64_t fallback = 0u;
    uint64_t certified_mismatch = 0u;
    uint64_t all_mismatch = 0u;
    double gap_sum = 0.0;
    uint64_t gap_count = 0u;

    void merge(const Stats& o) {
        rows += o.rows;
        certified += o.certified;
        fallback += o.fallback;
        certified_mismatch += o.certified_mismatch;
        all_mismatch += o.all_mismatch;
        gap_sum += o.gap_sum;
        gap_count += o.gap_count;
    }
};

struct Case {
    uint32_t rows;
    uint32_t vocab;
    uint32_t k_padded;
    uint32_t pool;
    bool perturb;
};

void cpu_argmax(const float* row, uint32_t vocab, int32_t& best_id, float& best_value) {
    best_id = 0;
    best_value = row[0];
    for (uint32_t v = 1u; v < vocab; ++v) {
        if (row[v] > best_value ||
            (row[v] == best_value && static_cast<int32_t>(v) < best_id)) {
            best_value = row[v];
            best_id = static_cast<int32_t>(v);
        }
    }
}

struct Device {
    uint32_t rows = 0u;
    uint32_t vocab = 0u;
    uint32_t k_padded = 0u;
    uint32_t pool = 0u;
    uint32_t topn_pool = 0u;
    uint32_t partitions = 0u;
    uint32_t scratch_stride = 0u;
    uint32_t act_code_stride = 0u;
    uint32_t act_scale_stride = 0u;
    std::size_t wscale_count = 0u;
    std::size_t elems = 0u;

    uint8_t* psq8 = nullptr;
    uint8_t* wscale = nullptr;
    uint8_t* int2 = nullptr;
    uint8_t* tables = nullptr;
    uint8_t* act = nullptr;
    float* act_scales = nullptr;
    ps::bf16_t* hidden = nullptr;
    float* error = nullptr;
    float* error_table = nullptr;
    float* act_l2 = nullptr;
    float* coarse = nullptr;
    float* upper = nullptr;
    int32_t* top_ids = nullptr;
    float* top_values = nullptr;
    int32_t* candidate_ids = nullptr;
    float* rerank = nullptr;
    int32_t* scratch_ids = nullptr;
    float* scratch_values = nullptr;
    int32_t* best_ids = nullptr;
    float* best_logits = nullptr;
    float* gaps = nullptr;
    uint8_t* certified = nullptr;
    float* full = nullptr;

    bool alloc(const Case& c) {
        rows = c.rows;
        vocab = c.vocab;
        k_padded = c.k_padded;
        pool = c.pool;
        topn_pool = c.pool + 1u;
        partitions = ps::kernel::dflash2_int2_default_partitions(vocab);
        scratch_stride = partitions * ps::kernel::kDflash2Int2MaxPool;
        const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(k_padded, rows);
        act_code_stride = layout.code_row_stride_bytes;
        act_scale_stride = layout.scale_row_stride_bytes;
        wscale_count = ((static_cast<std::size_t>(vocab) + 15u) / 16u) * (k_padded / 2u);
        elems = static_cast<std::size_t>(rows) * vocab;

        bool ok = true;
#define PS_ALLOC(p, bytes) ok = ok && (hipMalloc(&(p), (bytes)) == hipSuccess)
        PS_ALLOC(psq8, psq8_bytes(vocab, k_padded));
        PS_ALLOC(wscale, wscale_count * 2u);
        PS_ALLOC(int2, psq8_bytes(vocab, k_padded) / 4u);
        PS_ALLOC(tables, 1280u);
        PS_ALLOC(act, static_cast<std::size_t>(act_code_stride) * ((rows + 15u) / 16u) * 16u);
        PS_ALLOC(act_scales, static_cast<std::size_t>(act_scale_stride / 4u) * rows * 4u);
        PS_ALLOC(hidden, static_cast<std::size_t>(rows) * k_padded * sizeof(ps::bf16_t));
        PS_ALLOC(error, vocab * sizeof(float));
        PS_ALLOC(error_table, 256u * sizeof(float));
        PS_ALLOC(act_l2, rows * sizeof(float));
        PS_ALLOC(coarse, elems * sizeof(float));
        PS_ALLOC(upper, elems * sizeof(float));
        PS_ALLOC(top_ids, static_cast<std::size_t>(rows) * topn_pool * sizeof(int32_t));
        PS_ALLOC(top_values, static_cast<std::size_t>(rows) * topn_pool * sizeof(float));
        PS_ALLOC(candidate_ids, static_cast<std::size_t>(rows) * pool * sizeof(int32_t));
        PS_ALLOC(rerank, static_cast<std::size_t>(rows) * pool * sizeof(float));
        PS_ALLOC(scratch_ids, static_cast<std::size_t>(rows) * scratch_stride * sizeof(int32_t));
        PS_ALLOC(scratch_values,
                 static_cast<std::size_t>(rows) * scratch_stride * sizeof(float));
        PS_ALLOC(best_ids, rows * sizeof(int32_t));
        PS_ALLOC(best_logits, rows * sizeof(float));
        PS_ALLOC(gaps, rows * sizeof(float));
        PS_ALLOC(certified, rows);
        PS_ALLOC(full, elems * sizeof(float));
#undef PS_ALLOC
        return ok;
    }

    void free_all() {
#define PS_FREE(p) (void)hipFree(p)
        PS_FREE(psq8);
        PS_FREE(wscale);
        PS_FREE(int2);
        PS_FREE(tables);
        PS_FREE(act);
        PS_FREE(act_scales);
        PS_FREE(hidden);
        PS_FREE(error);
        PS_FREE(error_table);
        PS_FREE(act_l2);
        PS_FREE(coarse);
        PS_FREE(upper);
        PS_FREE(top_ids);
        PS_FREE(top_values);
        PS_FREE(candidate_ids);
        PS_FREE(rerank);
        PS_FREE(scratch_ids);
        PS_FREE(scratch_values);
        PS_FREE(best_ids);
        PS_FREE(best_logits);
        PS_FREE(gaps);
        PS_FREE(certified);
        PS_FREE(full);
#undef PS_FREE
    }
};

Stats run_config(const Case& c, uint32_t base_seed, uint32_t iterations, float abs_margin,
                 float rel_margin, bool report_once, std::vector<float>& gap_samples) {
    Stats stats;
    const Dflash2Int2Codebook codebook = [] {
        Dflash2Int2Codebook cb;
        cb.values = {0x00u, 0x30u, 0x38u, 0x3Cu};
        for (uint32_t i = 0u; i < 4u; ++i)
            cb.decoded[i] = ps::quantization::fpx::e4m3_decode_f32(cb.values[i]);
        return cb;
    }();
    uint8_t map_table[256] = {};
    uint32_t expand_table[256] = {};
    float error_table[256];
    ps::kernel::dflash2_int2_build_tables(codebook, map_table, expand_table);
    ps::kernel::dflash2_int2_build_error_table(codebook, map_table, error_table);
    std::vector<uint8_t> tables(1280u, 0u);
    std::memcpy(tables.data(), map_table, 256u);
    std::memcpy(tables.data() + 256u, expand_table, 1024u);

    Device dev;
    if (!dev.alloc(c)) {
        std::printf("FAIL: shadow allocation\n");
        ++failed;
        dev.free_all();
        return stats;
    }
    (void)hipMemcpy(dev.tables, tables.data(), tables.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev.error_table, error_table, 256u * sizeof(float),
                    hipMemcpyHostToDevice);

    const std::vector<uint8_t> exact_codes = {0x00u, 0x30u, 0x38u, 0x3Cu};
    const std::vector<uint8_t> extra_codes = {0x34u, 0xB8u};
    std::vector<uint8_t> psq8(psq8_bytes(c.vocab, c.k_padded), 0u);
    std::vector<uint16_t> wscales(dev.wscale_count, 0u);
    std::vector<ps::bf16_t> hidden(static_cast<std::size_t>(c.rows) * c.k_padded);
    std::vector<int32_t> top_ids(static_cast<std::size_t>(c.rows) * dev.topn_pool, 0);
    std::vector<int32_t> candidate_ids(static_cast<std::size_t>(c.rows) * c.pool, 0);
    std::vector<float> full(dev.elems, 0.0f);
    std::vector<int32_t> best_ids(c.rows, 0);
    std::vector<float> best_logits(c.rows, 0.0f);
    std::vector<float> gaps(c.rows, 0.0f);
    std::vector<uint8_t> certified(c.rows, 0u);

    for (uint32_t it = 0u; it < iterations; ++it) {
        Rng rng;
        rng.state = base_seed + it * 2654435761u;
        for (uint32_t o = 0u; o < c.vocab; ++o) {
            for (uint32_t k = 0u; k < c.k_padded; ++k) {
                uint8_t code = exact_codes[rng.below(4u)];
                if (c.perturb && rng.below(4u) == 0u)
                    code = extra_codes[rng.below(2u)];
                psq8[dflash2_int2_test::psq8_code_index(o, k, c.k_padded)] = code;
            }
        }
        for (std::size_t i = 0u; i < wscales.size(); ++i)
            wscales[i] = 0u;
        for (uint32_t o = 0u; o < c.vocab; ++o) {
            for (uint32_t ib = 0u; ib < c.k_padded / 32u; ++ib)
                wscales[dflash2_int2_test::psq8_scale_index(o, ib * 32u, c.k_padded)] =
                    dflash2_int2_test::bf16_bits(0.02f + 0.4f * rng.unit());
        }
        for (auto& h : hidden)
            h.data = ps::f32_to_bf16_rne(rng.sym());

        (void)hipMemcpy(dev.psq8, psq8.data(), psq8.size(), hipMemcpyHostToDevice);
        (void)hipMemcpy(dev.wscale, wscales.data(), dev.wscale_count * 2u,
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(dev.hidden, hidden.data(), hidden.size() * sizeof(ps::bf16_t),
                        hipMemcpyHostToDevice);

        hipError_t err = ps::kernel::launch_dflash2_int2_pack(
            dev.psq8, dev.tables, dev.int2, c.vocab, c.k_padded, nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_activation_quantize_e4m3(
                                      dev.hidden, dev.act, dev.act_scales, c.rows, c.k_padded,
                                      c.k_padded, c.k_padded, dev.act_code_stride,
                                      dev.act_scale_stride, nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_dflash2_activation_l2(
                                      dev.act, dev.act_scales, c.rows, c.k_padded,
                                      dev.act_code_stride, dev.act_scale_stride, dev.act_l2,
                                      nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_dflash2_int2_row_error_l2(
                                      dev.psq8, dev.wscale, dev.error_table, c.vocab,
                                      c.k_padded, c.k_padded, dev.error, nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_dflash2_int2_coarse_head(
                                      dev.int2, dev.wscale, dev.act, dev.act_scales,
                                      reinterpret_cast<const uint32_t*>(dev.tables + 256u),
                                      dev.coarse, c.rows, c.vocab, c.k_padded, c.k_padded,
                                      dev.act_code_stride, dev.act_scale_stride, c.vocab,
                                      nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_dflash2_target_upper_logits(
                                      dev.coarse, dev.act_l2, dev.error, dev.upper, c.rows,
                                      c.vocab, nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_dflash2_coarse_topn(
                                      dev.upper, c.rows, c.vocab, c.vocab, dev.topn_pool,
                                      dev.top_ids, dev.top_values, dev.scratch_ids,
                                      dev.scratch_values, dev.scratch_stride, dev.partitions,
                                      nullptr);
        if (err == hipSuccess)
            err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::printf("FAIL: pipeline launch %s\n", hipGetErrorString(err));
            ++failed;
            break;
        }
        (void)hipMemcpy(top_ids.data(), dev.top_ids, top_ids.size() * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        for (uint32_t r = 0u; r < c.rows; ++r)
            for (uint32_t i = 0u; i < c.pool; ++i)
                candidate_ids[static_cast<std::size_t>(r) * c.pool + i] =
                    top_ids[static_cast<std::size_t>(r) * dev.topn_pool + i];
        (void)hipMemcpy(dev.candidate_ids, candidate_ids.data(),
                        candidate_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice);

        err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
            dev.psq8, dev.wscale, dev.act, dev.act_scales, dev.candidate_ids, dev.rerank,
            c.rows, c.pool, c.k_padded, c.k_padded, dev.act_code_stride,
            dev.act_scale_stride, c.pool, nullptr);
        err = err != hipSuccess ? err
                                : ps::kernel::launch_dflash2_target_certified_argmax(
                                      dev.rerank, dev.candidate_ids, dev.top_values + c.pool,
                                      dev.topn_pool, abs_margin, rel_margin, dev.best_ids,
                                      dev.best_logits, dev.gaps, dev.certified, c.rows, c.pool,
                                      nullptr);
        ps::kernel::Psq8GemmConfig cfg;
        cfg.id = ps::kernel::Psq8GemmConfigId::RowBlock1;
        err = err != hipSuccess
                  ? err
                  : ps::kernel::launch_gemm_psq8_w8a8_wmma(
                        cfg, dev.psq8, dev.wscale, reinterpret_cast<const int8_t*>(dev.act),
                        dev.act_scales, dev.full, ps::kernel::Psq8GemmOutputDType::F32,
                        c.rows, c.vocab, c.k_padded, c.k_padded, dev.act_code_stride,
                        dev.act_scale_stride, c.vocab, nullptr);
        if (err == hipSuccess)
            err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::printf("FAIL: rerank/oracle launch %s\n", hipGetErrorString(err));
            ++failed;
            break;
        }

        (void)hipMemcpy(full.data(), dev.full, full.size() * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(best_ids.data(), dev.best_ids, c.rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(best_logits.data(), dev.best_logits, c.rows * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(gaps.data(), dev.gaps, c.rows * sizeof(float),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(certified.data(), dev.certified, c.rows, hipMemcpyDeviceToHost);

        stats.rows += c.rows;
        for (uint32_t r = 0u; r < c.rows; ++r) {
            int32_t oracle_id = 0;
            float oracle_value = 0.0f;
            cpu_argmax(full.data() + static_cast<std::size_t>(r) * c.vocab, c.vocab,
                       oracle_id, oracle_value);
            (void)oracle_value;
            const bool proxy_match = best_ids[r] == oracle_id;
            if (!proxy_match)
                ++stats.all_mismatch;
            if (certified[r] != 0u) {
                ++stats.certified;
                stats.gap_sum += static_cast<double>(gaps[r]);
                ++stats.gap_count;
                gap_samples.push_back(gaps[r]);
                if (!proxy_match) {
                    ++stats.certified_mismatch;
                    if (report_once)
                        std::printf(
                            "CERTIFIED MISMATCH rows=%u vocab=%u k=%u pool=%u r=%u proxy=%d "
                            "oracle=%d best=%.6f gap=%.6f\n",
                            c.rows, c.vocab, c.k_padded, c.pool, r, best_ids[r], oracle_id,
                            static_cast<double>(best_logits[r]), static_cast<double>(gaps[r]));
                }
            } else {
                ++stats.fallback;
            }
        }
    }

    dev.free_all();
    return stats;
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;

    const Case configs[] = {
        {8u, 128u, 64u, 32u, false}, {8u, 256u, 128u, 64u, false},
        {4u, 96u, 64u, 48u, false},  {8u, 128u, 64u, 32u, true},
        {8u, 256u, 128u, 64u, true}, {4u, 96u, 64u, 48u, true},
    };
    const uint32_t pools[] = {16u, 32u, 48u, 64u};

    const float abs_margin = 1e-2f;
    const float rel_margin = 1e-4f;

    Stats total;
    std::vector<float> gap_samples;
    uint32_t seed = 0x51ed2701u;
    for (const Case& c : configs) {
        for (uint32_t pool : pools) {
            if (pool > c.vocab)
                continue;
            Case sc = c;
            sc.pool = pool;
            const uint32_t iterations = c.rows >= 8u ? 512u : 1088u;
            const Stats s =
                run_config(sc, seed, iterations, abs_margin, rel_margin, true, gap_samples);
            total.merge(s);
            seed = seed * 1664525u + 1013904223u;
            std::printf(
                "TARGET_LM_PROXY rows=%u vocab=%u k=%u pool=%u perturb=%d decisions=%llu "
                "certified=%llu fallback=%llu cert_mismatch=%llu\n",
                sc.rows, sc.vocab, sc.k_padded, sc.pool, c.perturb ? 1 : 0,
                static_cast<unsigned long long>(s.rows),
                static_cast<unsigned long long>(s.certified),
                static_cast<unsigned long long>(s.fallback),
                static_cast<unsigned long long>(s.certified_mismatch));
        }
    }

    check(total.certified_mismatch == 0u, "certified decisions never mismatch oracle");
    check(total.certified > 0u, "at least some decisions certified");
    check(total.rows >= 100000u, "at least 100k decisions");

    float gap_p50 = 0.0f;
    float gap_p90 = 0.0f;
    float gap_p99 = 0.0f;
    if (!gap_samples.empty()) {
        std::sort(gap_samples.begin(), gap_samples.end());
        const auto pick = [&gap_samples](double q) {
            const std::size_t idx = static_cast<std::size_t>(
                q * static_cast<double>(gap_samples.size() - 1u) + 0.5);
            return gap_samples[idx];
        };
        gap_p50 = pick(0.50);
        gap_p90 = pick(0.90);
        gap_p99 = pick(0.99);
    }
    std::printf(
        "TARGET_LM_PROXY_SUMMARY decisions=%llu certified=%llu fallback=%llu "
        "certified_mismatch=%llu all_mismatch=%llu cert_rate=%.4f gap_p50=%.6f "
        "gap_p90=%.6f gap_p99=%.6f\n",
        static_cast<unsigned long long>(total.rows),
        static_cast<unsigned long long>(total.certified),
        static_cast<unsigned long long>(total.fallback),
        static_cast<unsigned long long>(total.certified_mismatch),
        static_cast<unsigned long long>(total.all_mismatch),
        total.rows > 0u ? static_cast<double>(total.certified) /
                              static_cast<double>(total.rows)
                        : 0.0,
        static_cast<double>(gap_p50), static_cast<double>(gap_p90),
        static_cast<double>(gap_p99));

    {
        const Case diag_case = {8u, 128u, 64u, 32u, true};
        const float margin_pairs[][2] = {
            {0.0f, 0.0f}, {1.0e-4f, 1.0e-6f}, {1.0e-2f, 1.0e-4f}};
        for (const auto& m : margin_pairs) {
            std::vector<float> diag_gaps;
            const Stats s = run_config(diag_case, 0x0badc0deu, 256u, m[0], m[1], true,
                                       diag_gaps);
            std::printf(
                "TARGET_LM_PROXY_MARGIN abs=%.1e rel=%.1e decisions=%llu certified=%llu "
                "cert_mismatch=%llu\n",
                static_cast<double>(m[0]), static_cast<double>(m[1]),
                static_cast<unsigned long long>(s.rows),
                static_cast<unsigned long long>(s.certified),
                static_cast<unsigned long long>(s.certified_mismatch));
        }
    }
    std::printf("test_target_lm_proxy_fallback: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
