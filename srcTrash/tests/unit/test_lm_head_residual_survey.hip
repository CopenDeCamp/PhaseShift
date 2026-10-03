#include "support/dflash2_int2_test_common.h"
#include "support/target_lm_proxy_harness.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/quantization/fpx/e4m3.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using ps::kernel::Dflash2Int2Codebook;

constexpr uint32_t kW32 = 32u;

struct Survey {
    uint32_t vocab = 0u;
    uint32_t k_padded = 0u;
    uint32_t groups = 0u;
    uint32_t scale_stride = 0u;
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
    float error_table[256] = {};
    uint8_t map_table[256] = {};
    Dflash2Int2Codebook codebook{};

    inline uint8_t code(uint32_t j, uint32_t k) const {
        return codes[dflash2_int2_test::psq8_code_index(j, k, k_padded)];
    }
    inline float scale(uint32_t j, uint32_t g) const {
        const std::size_t idx = dflash2_int2_test::psq8_scale_index(j, g * kW32, scale_stride);
        const float v = [&] {
            const uint32_t u = static_cast<uint32_t>(scales[idx]) << 16;
            return *reinterpret_cast<const float*>(&u);
        }();
        return v;
    }
    inline double residual(uint32_t j, uint32_t k) const {
        return static_cast<double>(error_table[code(j, k)]) *
               static_cast<double>(scale(j, k / kW32));
    }
    inline double exact_weight(uint32_t j, uint32_t k) const {
        const double c = static_cast<double>(
            ps::quantization::fpx::e4m3_decode_f32(code(j, k)));
        return c * static_cast<double>(scale(j, k / kW32));
    }
};

uint64_t g_seed = 0x243f6a8885a308d3ull;
double rng_normal() {
    g_seed = g_seed * 6364136223846793005ull + 1442695040888963407ull;
    const double u1 = ((g_seed >> 11) & ((1ull << 53) - 1)) / static_cast<double>(1ull << 53);
    g_seed = g_seed * 6364136223846793005ull + 1442695040888963407ull;
    const double u2 = ((g_seed >> 11) & ((1ull << 53) - 1)) / static_cast<double>(1ull << 53);
    const double r = std::sqrt(-2.0 * std::log(u1 + 1e-300));
    return r * std::cos(2.0 * M_PI * u2);
}

template <typename Fn>
void parallel_for(uint32_t begin, uint32_t end, Fn&& fn) {
    const uint32_t n = end - begin;
    if (n == 0u) return;
    const uint32_t threads = std::min<uint32_t>(16u, std::max<uint32_t>(1u, n));
    std::vector<std::thread> pool;
    std::atomic<uint32_t> next{begin};
    for (uint32_t t = 0u; t < threads; ++t) {
        pool.emplace_back([&] {
            for (;;) {
                const uint32_t i = next.fetch_add(1u);
                if (i >= end) break;
                fn(i);
            }
        });
    }
    for (auto& th : pool) th.join();
}

float quantile(const std::vector<float>& v, double q) {
    if (v.empty()) return 0.0f;
    const std::size_t idx =
        static_cast<std::size_t>(q * static_cast<double>(v.size() - 1u) + 0.5);
    return v[idx];
}

void finish_dist(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
}

void print_dist(const char* tag, const std::vector<float>& sorted, double ratio_ref) {
    const double p50 = quantile(sorted, 0.50);
    const double p90 = quantile(sorted, 0.90);
    const double p99 = quantile(sorted, 0.99);
    const double mx = sorted.empty() ? 0.0 : sorted.back();
    std::printf("%s tail_l2 p50=%.5f p90=%.5f p99=%.5f max=%.5f ratio_p50=%.4f "
                "ratio_p99=%.4f\n",
                tag, p50, p90, p99, mx, ratio_ref > 0.0 ? p50 / ratio_ref : 0.0,
                ratio_ref > 0.0 ? p99 / ratio_ref : 0.0);
}

// ---------------------------------------------------------------- Part 1

struct SurveyResult {
    std::vector<float> orig_l2;
    std::vector<float> orig_l1;
    std::vector<float> max_abs;
    std::vector<std::vector<float>> topm_tail;
    std::vector<std::vector<float>> topg_tail;
    std::vector<uint32_t> topm_list;
    std::vector<uint32_t> topg_list;
};

void weight_survey(const Survey& s, SurveyResult& out) {
    const uint32_t V = s.vocab;
    const uint32_t K = s.k_padded;
    const uint32_t G = s.groups;
    out.topm_list = {4u, 8u, 16u, 32u, 64u, 128u, 256u};
    out.topg_list = {1u, 2u, 4u, 8u, 16u, 32u};
    out.orig_l2.assign(V, 0.0f);
    out.orig_l1.assign(V, 0.0f);
    out.max_abs.assign(V, 0.0f);
    out.topm_tail.resize(out.topm_list.size());
    out.topg_tail.resize(out.topg_list.size());
    for (auto& v : out.topm_tail) v.assign(V, 0.0f);
    for (auto& v : out.topg_tail) v.assign(V, 0.0f);

    const uint32_t max_m = out.topm_list.back();
    const uint32_t max_g = out.topg_list.back();

    parallel_for(0u, V, [&](uint32_t j) {
        std::vector<double> d(K);
        double l2 = 0.0, l1 = 0.0, mx = 0.0;
        std::vector<double> group_energy(G, 0.0);
        for (uint32_t k = 0u; k < K; ++k) {
            const double v = s.residual(j, k);
            d[k] = v;
            const double a = std::fabs(v);
            l2 += v * v;
            l1 += a;
            if (a > mx) mx = a;
            group_energy[k / kW32] += v * v;
        }
        out.orig_l2[j] = static_cast<float>(std::sqrt(l2));
        out.orig_l1[j] = static_cast<float>(l1);
        out.max_abs[j] = static_cast<float>(mx);

        std::vector<uint32_t> idx(K);
        for (uint32_t k = 0u; k < K; ++k) idx[k] = k;
        const uint32_t mcap = std::min<uint32_t>(max_m, K);
        std::partial_sort(idx.begin(), idx.begin() + mcap, idx.end(),
                          [&d](uint32_t a, uint32_t b) {
                              return std::fabs(d[a]) > std::fabs(d[b]);
                          });
        double captured = 0.0;
        uint32_t mi = 0u;
        for (std::size_t slot = 0u; slot < out.topm_list.size(); ++slot) {
            const uint32_t m = std::min<uint32_t>(out.topm_list[slot], K);
            while (mi < m) {
                captured += d[idx[mi]] * d[idx[mi]];
                ++mi;
            }
            const double tail = std::sqrt(std::max(0.0, l2 - captured));
            out.topm_tail[slot][j] = static_cast<float>(tail);
        }

        std::vector<uint32_t> gidx(G);
        for (uint32_t g = 0u; g < G; ++g) gidx[g] = g;
        const uint32_t gcap = std::min<uint32_t>(max_g, G);
        std::partial_sort(gidx.begin(), gidx.begin() + gcap, gidx.end(),
                          [&group_energy](uint32_t a, uint32_t b) {
                              return group_energy[a] > group_energy[b];
                          });
        double gcaptured = 0.0;
        uint32_t gi = 0u;
        for (std::size_t slot = 0u; slot < out.topg_list.size(); ++slot) {
            const uint32_t g = std::min<uint32_t>(out.topg_list[slot], G);
            while (gi < g) {
                gcaptured += group_energy[gidx[gi]];
                ++gi;
            }
            const double tail = std::sqrt(std::max(0.0, l2 - gcaptured));
            out.topg_tail[slot][j] = static_cast<float>(tail);
        }
    });
}

// ---------------------------------------------------------------- low-rank

struct LowRank {
    uint32_t rank = 0u;
    uint32_t k_padded = 0u;
    std::vector<float> v_basis;  // [k_padded, rank]
    std::vector<double> sigma2;
    double total_energy = 0.0;
    std::vector<std::vector<float>> tail;  // [rank slot][subset]
};

void jacobi_eigen(std::vector<double>& a, uint32_t n, std::vector<double>& evec) {
    evec.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (uint32_t i = 0u; i < n; ++i) evec[static_cast<std::size_t>(i) * n + i] = 1.0;
    for (uint32_t sweep = 0u; sweep < 100u; ++sweep) {
        double off = 0.0;
        for (uint32_t p = 0u; p < n; ++p)
            for (uint32_t q = p + 1u; q < n; ++q)
                off += a[static_cast<std::size_t>(p) * n + q] *
                       a[static_cast<std::size_t>(p) * n + q];
        if (off < 1e-20) break;
        for (uint32_t p = 0u; p < n; ++p) {
            for (uint32_t q = p + 1u; q < n; ++q) {
                const double apq = a[static_cast<std::size_t>(p) * n + q];
                if (std::fabs(apq) < 1e-30) continue;
                const double app = a[static_cast<std::size_t>(p) * n + p];
                const double aqq = a[static_cast<std::size_t>(q) * n + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double sn = t * c;
                for (uint32_t i = 0u; i < n; ++i) {
                    const double aip = a[static_cast<std::size_t>(i) * n + p];
                    const double aiq = a[static_cast<std::size_t>(i) * n + q];
                    a[static_cast<std::size_t>(i) * n + p] = c * aip - sn * aiq;
                    a[static_cast<std::size_t>(i) * n + q] = sn * aip + c * aiq;
                }
                for (uint32_t i = 0u; i < n; ++i) {
                    const double api = a[static_cast<std::size_t>(p) * n + i];
                    const double aqi = a[static_cast<std::size_t>(q) * n + i];
                    a[static_cast<std::size_t>(p) * n + i] = c * api - sn * aqi;
                    a[static_cast<std::size_t>(q) * n + i] = sn * api + c * aqi;
                }
                for (uint32_t i = 0u; i < n; ++i) {
                    const double vip = evec[static_cast<std::size_t>(i) * n + p];
                    const double viq = evec[static_cast<std::size_t>(i) * n + q];
                    evec[static_cast<std::size_t>(i) * n + p] = c * vip - sn * viq;
                    evec[static_cast<std::size_t>(i) * n + q] = sn * vip + c * viq;
                }
            }
        }
    }
}

LowRank low_rank_survey(const Survey& s, uint32_t subset_rows, uint32_t p_max,
                        const std::vector<uint32_t>& rank_list) {
    const uint32_t K = s.k_padded;
    const uint32_t S = std::min(subset_rows, s.vocab);
    std::vector<float> dsub(static_cast<std::size_t>(S) * K, 0.0f);
    std::vector<double> row_l2(S, 0.0);
    for (uint32_t j = 0u; j < S; ++j) {
        double l2 = 0.0;
        for (uint32_t k = 0u; k < K; ++k) {
            const double v = s.residual(j, k);
            dsub[static_cast<std::size_t>(j) * K + k] = static_cast<float>(v);
            l2 += v * v;
        }
        row_l2[j] = l2;
    }

    std::vector<double> omega(static_cast<std::size_t>(K) * p_max);
    for (auto& v : omega) v = rng_normal();

    std::vector<double> y(static_cast<std::size_t>(S) * p_max, 0.0);
    parallel_for(0u, S, [&](uint32_t j) {
        const float* row = dsub.data() + static_cast<std::size_t>(j) * K;
        for (uint32_t r = 0u; r < p_max; ++r) {
            double acc = 0.0;
            for (uint32_t k = 0u; k < K; ++k)
                acc += static_cast<double>(row[k]) * omega[static_cast<std::size_t>(k) * p_max + r];
            y[static_cast<std::size_t>(j) * p_max + r] = acc;
        }
    });

    for (uint32_t r = 0u; r < p_max; ++r) {
        for (uint32_t prev = 0u; prev < r; ++prev) {
            double dot = 0.0;
            for (uint32_t j = 0u; j < S; ++j)
                dot += y[static_cast<std::size_t>(j) * p_max + r] *
                       y[static_cast<std::size_t>(j) * p_max + prev];
            for (uint32_t j = 0u; j < S; ++j)
                y[static_cast<std::size_t>(j) * p_max + r] -=
                    dot * y[static_cast<std::size_t>(j) * p_max + prev];
        }
        double nrm = 0.0;
        for (uint32_t j = 0u; j < S; ++j) {
            const double v = y[static_cast<std::size_t>(j) * p_max + r];
            nrm += v * v;
        }
        nrm = std::sqrt(nrm);
        if (nrm < 1e-30) nrm = 1.0;
        for (uint32_t j = 0u; j < S; ++j)
            y[static_cast<std::size_t>(j) * p_max + r] /= nrm;
    }

    std::vector<double> bmat(static_cast<std::size_t>(p_max) * K, 0.0);
    parallel_for(0u, p_max, [&](uint32_t r) {
        for (uint32_t k = 0u; k < K; ++k) {
            double acc = 0.0;
            for (uint32_t j = 0u; j < S; ++j)
                acc += y[static_cast<std::size_t>(j) * p_max + r] *
                       static_cast<double>(dsub[static_cast<std::size_t>(j) * K + k]);
            bmat[static_cast<std::size_t>(r) * K + k] = acc;
        }
    });

    std::vector<double> c(static_cast<std::size_t>(p_max) * p_max, 0.0);
    parallel_for(0u, p_max, [&](uint32_t r) {
        for (uint32_t q = 0u; q < p_max; ++q) {
            double acc = 0.0;
            for (uint32_t k = 0u; k < K; ++k)
                acc += bmat[static_cast<std::size_t>(r) * K + k] *
                       bmat[static_cast<std::size_t>(q) * K + k];
            c[static_cast<std::size_t>(r) * p_max + q] = acc;
        }
    });

    std::vector<double> evec;
    jacobi_eigen(c, p_max, evec);
    std::vector<std::pair<double, uint32_t>> eig(p_max);
    for (uint32_t i = 0u; i < p_max; ++i)
        eig[i] = {c[static_cast<std::size_t>(i) * p_max + i], i};
    std::sort(eig.begin(), eig.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    LowRank lr;
    lr.k_padded = K;
    lr.total_energy = 0.0;
    for (uint32_t j = 0u; j < S; ++j) lr.total_energy += row_l2[j];

    const uint32_t rmax = rank_list.back();
    lr.v_basis.assign(static_cast<std::size_t>(K) * rmax, 0.0f);
    lr.sigma2.assign(rmax, 0.0);
    for (uint32_t r = 0u; r < rmax; ++r) {
        const double lam = std::max(0.0, eig[r].first);
        lr.sigma2[r] = lam;
        const double sg = std::sqrt(lam);
        const uint32_t col = eig[r].second;
        if (sg < 1e-30) continue;
        for (uint32_t k = 0u; k < K; ++k) {
            double acc = 0.0;
            for (uint32_t i = 0u; i < p_max; ++i)
                acc += bmat[static_cast<std::size_t>(i) * K + k] *
                       evec[static_cast<std::size_t>(i) * p_max + col];
            lr.v_basis[static_cast<std::size_t>(k) * rmax + r] =
                static_cast<float>(acc / sg);
        }
    }

    lr.rank = rmax;
    lr.tail.resize(rank_list.size());
    for (std::size_t si = 0u; si < rank_list.size(); ++si) {
        lr.tail[si].assign(S, 0.0f);
    }
    parallel_for(0u, S, [&](uint32_t j) {
        const float* row = dsub.data() + static_cast<std::size_t>(j) * K;
        for (std::size_t si = 0u; si < rank_list.size(); ++si) {
            const uint32_t r = rank_list[si];
            double proj = 0.0;
            for (uint32_t t = 0u; t < r; ++t) {
                double acc = 0.0;
                for (uint32_t k = 0u; k < K; ++k)
                    acc += static_cast<double>(row[k]) *
                           lr.v_basis[static_cast<std::size_t>(k) * rmax + t];
                proj += acc * acc;
            }
            lr.tail[si][j] =
                static_cast<float>(std::sqrt(std::max(0.0, row_l2[j] - proj)));
        }
    });
    return lr;
}

// ---------------------------------------------------------------- Part 2 (R3)

struct MethodParams {
    enum Type { NONE = 0, TOPM = 1, TOPG = 2 };
    Type type = NONE;
    uint32_t param = 0u;
    std::vector<uint16_t> m_idx;
    std::vector<float> m_res;
    std::vector<uint8_t> g_idx;
    std::vector<float> g_res;
    std::vector<float> tail_l2;
    const char* name = "none";
};

void build_meta(const Survey& s, MethodParams& mp) {
    const uint32_t V = s.vocab;
    const uint32_t K = s.k_padded;
    const uint32_t G = s.groups;
    mp.tail_l2.assign(V, 0.0f);
    if (mp.type == MethodParams::TOPM) {
        const uint32_t M = mp.param;
        mp.m_idx.assign(static_cast<std::size_t>(V) * M, 0u);
        mp.m_res.assign(static_cast<std::size_t>(V) * M, 0.0f);
        parallel_for(0u, V, [&](uint32_t j) {
            std::vector<double> d(K);
            double l2 = 0.0;
            for (uint32_t k = 0u; k < K; ++k) {
                d[k] = s.residual(j, k);
                l2 += d[k] * d[k];
            }
            std::vector<uint32_t> idx(K);
            for (uint32_t k = 0u; k < K; ++k) idx[k] = k;
            std::partial_sort(idx.begin(), idx.begin() + M, idx.end(),
                              [&d](uint32_t a, uint32_t b) {
                                  return std::fabs(d[a]) > std::fabs(d[b]);
                              });
            double captured = 0.0;
            for (uint32_t m = 0u; m < M; ++m) {
                mp.m_idx[static_cast<std::size_t>(j) * M + m] =
                    static_cast<uint16_t>(idx[m]);
                mp.m_res[static_cast<std::size_t>(j) * M + m] =
                    static_cast<float>(d[idx[m]]);
                captured += d[idx[m]] * d[idx[m]];
            }
            mp.tail_l2[j] = static_cast<float>(std::sqrt(std::max(0.0, l2 - captured)));
        });
    } else if (mp.type == MethodParams::TOPG) {
        const uint32_t P = mp.param;
        mp.g_idx.assign(static_cast<std::size_t>(V) * P, 0u);
        mp.g_res.assign(static_cast<std::size_t>(V) * P * kW32, 0.0f);
        parallel_for(0u, V, [&](uint32_t j) {
            std::vector<double> d(K);
            std::vector<double> ge(G, 0.0);
            double l2 = 0.0;
            for (uint32_t k = 0u; k < K; ++k) {
                d[k] = s.residual(j, k);
                l2 += d[k] * d[k];
                ge[k / kW32] += d[k] * d[k];
            }
            std::vector<uint32_t> gx(G);
            for (uint32_t g = 0u; g < G; ++g) gx[g] = g;
            std::partial_sort(gx.begin(), gx.begin() + P, gx.end(),
                              [&ge](uint32_t a, uint32_t b) { return ge[a] > ge[b]; });
            double captured = 0.0;
            for (uint32_t p = 0u; p < P; ++p) {
                const uint32_t g = gx[p];
                mp.g_idx[static_cast<std::size_t>(j) * P + p] = static_cast<uint8_t>(g);
                captured += ge[g];
                for (uint32_t t = 0u; t < kW32; ++t) {
                    mp.g_res[(static_cast<std::size_t>(j) * P + p) * kW32 + t] =
                        static_cast<float>(d[g * kW32 + t]);
                }
            }
            mp.tail_l2[j] = static_cast<float>(std::sqrt(std::max(0.0, l2 - captured)));
        });
    }
}

double exact_dot(const Survey& s, const double* a, uint32_t j) {
    double acc = 0.0;
    for (uint32_t k = 0u; k < s.k_padded; ++k) acc += a[k] * s.exact_weight(j, k);
    return acc;
}

struct ReplayMetrics {
    uint64_t rows = 0u;
    uint64_t direct_cert = 0u;
    uint64_t cert[3] = {0u, 0u, 0u};
    uint64_t mismatch[3] = {0u, 0u, 0u};
    uint64_t recall_upper[3] = {0u, 0u, 0u};
    uint64_t recall_score[3] = {0u, 0u, 0u};
    uint64_t bound_violation = 0u;
    std::vector<float> slack;
};

void replay_method(const Survey& s, const MethodParams& mp, uint32_t rows,
                   const std::vector<double>& act, const std::vector<float>& coarse,
                   const std::vector<int32_t>& oracle, const std::vector<float>& error_l2,
                   const std::vector<float>& act_l2, const float margin_abs,
                   const float margin_rel, ReplayMetrics& out) {
    const uint32_t V = s.vocab;
    const uint32_t K = s.k_padded;
    const uint32_t pools[3] = {32u, 64u, 128u};
    const bool parallel_method = mp.type != MethodParams::NONE;

    for (uint32_t r = 0u; r < rows; ++r) {
        const double* A = act.data() + static_cast<std::size_t>(r) * K;
        const float* co = coarse.data() + static_cast<std::size_t>(r) * V;
        double norm_a2 = 0.0;
        for (uint32_t k = 0u; k < K; ++k) norm_a2 += A[k] * A[k];

        std::vector<float> upper(V, 0.0f);
        std::vector<float> score(V, 0.0f);
        std::vector<float> corr(V, 0.0f);
        std::vector<float> tailb(V, 0.0f);

        if (mp.type == MethodParams::NONE) {
            for (uint32_t j = 0u; j < V; ++j) {
                const float b = act_l2[r] * error_l2[j];
                tailb[j] = b;
                upper[j] = co[j] + b;
                score[j] = co[j];
            }
        } else if (mp.type == MethodParams::TOPM) {
            const uint32_t M = mp.param;
            parallel_for(0u, V, [&](uint32_t j) {
                const uint16_t* idx = mp.m_idx.data() + static_cast<std::size_t>(j) * M;
                const float* res = mp.m_res.data() + static_cast<std::size_t>(j) * M;
                double c = 0.0;
                double sel2 = 0.0;
                for (uint32_t m = 0u; m < M; ++m) {
                    const double av = A[idx[m]];
                    c += av * static_cast<double>(res[m]);
                    sel2 += av * av;
                }
                const double atail = std::sqrt(std::max(0.0, norm_a2 - sel2));
                const double tb = atail * static_cast<double>(mp.tail_l2[j]);
                corr[j] = static_cast<float>(c);
                tailb[j] = static_cast<float>(tb);
                score[j] = co[j] + static_cast<float>(c);
                upper[j] = static_cast<float>(co[j] + c + tb);
            });
        } else {
            const uint32_t P = mp.param;
            parallel_for(0u, V, [&](uint32_t j) {
                const uint8_t* gi = mp.g_idx.data() + static_cast<std::size_t>(j) * P;
                const float* gr = mp.g_res.data() + static_cast<std::size_t>(j) * P * kW32;
                double c = 0.0;
                double sel2 = 0.0;
                for (uint32_t p = 0u; p < P; ++p) {
                    double ga = 0.0;
                    for (uint32_t t = 0u; t < kW32; ++t) {
                        const double av = A[static_cast<std::size_t>(gi[p]) * kW32 + t];
                        ga += av * av;
                        c += av * static_cast<double>(gr[static_cast<std::size_t>(p) * kW32 + t]);
                    }
                    sel2 += ga;
                }
                const double atail = std::sqrt(std::max(0.0, norm_a2 - sel2));
                const double tb = atail * static_cast<double>(mp.tail_l2[j]);
                corr[j] = static_cast<float>(c);
                tailb[j] = static_cast<float>(tb);
                score[j] = co[j] + static_cast<float>(c);
                upper[j] = static_cast<float>(co[j] + c + tb);
            });
        }
        (void)parallel_method;

        std::vector<uint32_t> order(V);
        for (uint32_t j = 0u; j < V; ++j) order[j] = j;
        std::partial_sort(order.begin(), order.begin() + 129, order.end(),
                          [&upper](uint32_t a, uint32_t b) { return upper[a] > upper[b]; });
        std::vector<double> cand_exact(129, 0.0);
        for (uint32_t c = 0u; c < 129u; ++c) {
            cand_exact[c] = exact_dot(s, A, order[c]);
            if (cand_exact[c] > static_cast<double>(upper[order[c]]) + 1e-2)
                ++out.bound_violation;
        }
        std::vector<uint32_t> order_score(V);
        for (uint32_t j = 0u; j < V; ++j) order_score[j] = j;
        std::partial_sort(order_score.begin(), order_score.begin() + 128, order_score.end(),
                          [&score](uint32_t a, uint32_t b) { return score[a] > score[b]; });
        const int32_t want = oracle[r];

        double best = -1e30;
        uint32_t bi = 0u;
        for (uint32_t si = 0u; si < 3u; ++si) {
            const uint32_t N = pools[si];
            while (bi < N) {
                if (cand_exact[bi] > best) best = cand_exact[bi];
                ++bi;
            }
            const double omitted = static_cast<double>(upper[order[N]]);
            const double scale = std::max(std::fabs(best), std::fabs(omitted));
            const double margin = margin_abs + margin_rel * scale;
            if (best > omitted + margin) {
                ++out.cert[si];
                if (order[0] != static_cast<uint32_t>(want)) {
                    int32_t be = 0;
                    for (uint32_t c = 1u; c < N; ++c)
                        if (cand_exact[c] > cand_exact[be]) be = static_cast<int32_t>(c);
                    if (order[be] != static_cast<uint32_t>(want)) ++out.mismatch[si];
                }
            }
            bool hit = false;
            for (uint32_t c = 0u; c < N && !hit; ++c) hit = order[c] == static_cast<uint32_t>(want);
            if (hit) ++out.recall_upper[si];
            bool hit2 = false;
            for (uint32_t c = 0u; c < N && !hit2; ++c)
                hit2 = order_score[c] == static_cast<uint32_t>(want);
            if (hit2) ++out.recall_score[si];
        }

        const double exact_o = exact_dot(s, A, static_cast<uint32_t>(want));
        double max_other = -1e30;
        for (uint32_t j = 0u; j < V; ++j) {
            if (j == static_cast<uint32_t>(want)) continue;
            if (upper[j] > max_other) max_other = upper[j];
        }
        const double scale_o = std::max(std::fabs(exact_o), std::fabs(max_other));
        if (exact_o > max_other + (margin_abs + margin_rel * scale_o)) ++out.direct_cert;

        const double actual_res = exact_o - static_cast<double>(co[want]);
        const double actual_tail = actual_res - static_cast<double>(corr[want]);
        const double slack = static_cast<double>(tailb[want]) - std::fabs(actual_tail);
        out.slack.push_back(static_cast<float>(slack));
    }
    out.rows += rows;
}

void run_replay(const Survey& s, ps::qwen35::Qwen35Model& model, ps::gpu::GpuArena& arena,
                hipStream_t stream, uint32_t probe_rows, uint32_t ctx,
                std::vector<MethodParams>& methods, const std::vector<float>& error_l2) {
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = [] (const ps::qwen35::Qwen35TextConfig& t) {
        uint32_t n = 0u;
        for (int lt : t.layer_types)
            if (lt != 0) ++n;
        return n;
    }(tc);
    const uint32_t max_seq = ctx + probe_rows + 64u;
    const uint32_t num_pages = (max_seq + page_tokens - 1u) / page_tokens + 8u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 1u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 1u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("REPLAY FAIL pool create\n");
        return;
    }
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = std::max(ctx + 16u, probe_rows + 16u);
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = probe_rows;
    auto exec_result = ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool,
                                                         arena, ec);
    if (!exec_result.ok()) {
        std::printf("REPLAY FAIL executor: %s\n", exec_result.status().message().c_str());
        return;
    }
    ps::qwen35::Executor ex = exec_result.release();
    auto seq_result =
        ps::qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!seq_result.ok()) {
        std::printf("REPLAY FAIL sequence\n");
        return;
    }
    ps::qwen35::PagedSequenceState seq = seq_result.release();

    auto run_prefill = [&](const std::vector<int32_t>& toks, uint32_t output_rows,
                           ps::qwen35::BatchExecutionOutput* out) -> bool {
        std::vector<ps::qwen35::ScheduledRequest> requests;
        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = ps::runtime::ExecutionClass::PREFILL;
        req.token_begin = 0u;
        req.num_tokens = static_cast<uint32_t>(toks.size());
        req.prefix_tokens = seq.position;
        req.compute_logits = true;
        req.sample = true;
        req.num_output_rows = output_rows;
        requests.push_back(req);
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = toks.data();
        batch.requests = requests;
        batch.num_tokens = static_cast<uint32_t>(toks.size());
        batch.num_requests = 1u;
        batch.num_prefill_requests = 1u;
        auto executed = ps::qwen35::execute_batch(ex, batch, stream);
        if (!executed.ok()) {
            std::printf("REPLAY FAIL execute: %s\n", executed.status().message().c_str());
            return false;
        }
        if (out != nullptr) *out = executed.release();
        else (void)executed.release();
        return true;
    };

    std::vector<int32_t> prompt(ctx);
    for (uint32_t i = 0u; i < ctx; ++i)
        prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    if (!run_prefill(prompt, 1u, nullptr)) return;

    std::vector<int32_t> probe(probe_rows);
    for (uint32_t i = 0u; i < probe_rows; ++i)
        probe[i] = static_cast<int32_t>(100u + (i * 131u + 7u) % 5000u);
    ps::qwen35::BatchExecutionOutput out;
    if (!run_prefill(probe, probe_rows, &out)) return;

    std::vector<int32_t> oracle(probe_rows, -1);
    (void)hipMemcpy(oracle.data(), out.sampled_tokens.data<int32_t>(),
                    probe_rows * sizeof(int32_t), hipMemcpyDeviceToHost);

    ps::bf16_t* normed = nullptr;
    (void)hipMalloc(&normed, static_cast<std::size_t>(probe_rows) * hidden * sizeof(ps::bf16_t));
    const ps::gpu::Tensor& fw = model.weights().final_norm_weight;
    const ps::kernel::RmsNormDataType wdt =
        fw.element_size() == 4u ? ps::kernel::RmsNormDataType::F32
                                : ps::kernel::RmsNormDataType::BF16;
    if (ps::kernel::launch_rmsnorm(
            out.final_hidden.data<void>(), ps::kernel::RmsNormDataType::BF16, fw.data<void>(),
            wdt, ps::kernel::RmsNormWeightLayout::PER_FEATURE,
            ps::kernel::RmsNormWeightMode::ONE_PLUS, normed,
            ps::kernel::RmsNormDataType::BF16, probe_rows, hidden, hidden, hidden, hidden,
            tc.rms_norm_eps, stream) != hipSuccess) {
        std::printf("REPLAY FAIL rmsnorm\n");
        return;
    }

    target_lm_proxy::Proxy proxy;
    std::string perr;
    if (!proxy.init(model.weights().lm_head, 8u, 32u, stream, perr)) {
        std::printf("REPLAY FAIL proxy: %s\n", perr.c_str());
        return;
    }
    const uint32_t V = s.vocab;
    const uint32_t K = s.k_padded;

    std::vector<double> act(static_cast<std::size_t>(probe_rows) * K, 0.0);
    std::vector<float> coarse(static_cast<std::size_t>(probe_rows) * V, 0.0f);
    std::vector<float> act_l2(probe_rows, 0.0f);
    for (uint32_t s0 = 0u; s0 < probe_rows; s0 += 8u) {
        const uint32_t m = std::min(8u, probe_rows - s0);
        if (!proxy.prepare(normed + static_cast<std::size_t>(s0) * hidden, m, hidden, stream,
                           perr)) {
            std::printf("REPLAY FAIL prepare: %s\n", perr.c_str());
            return;
        }
        if (hipMemcpy(coarse.data() + static_cast<std::size_t>(s0) * V, proxy.coarse,
                      static_cast<std::size_t>(m) * V * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("REPLAY FAIL coarse copy\n");
            return;
        }
        if (hipMemcpy(act_l2.data() + s0, proxy.act_l2, m * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("REPLAY FAIL act_l2 copy\n");
            return;
        }
        std::vector<uint8_t> ac(static_cast<std::size_t>(proxy.act_code_stride) * 16u);
        std::vector<float> as(static_cast<std::size_t>(proxy.act_scale_stride / 4u) * m);
        (void)hipMemcpy(ac.data(), proxy.act_codes, ac.size(), hipMemcpyDeviceToHost);
        (void)hipMemcpy(as.data(), proxy.act_scales, as.size() * sizeof(float),
                        hipMemcpyDeviceToHost);
        for (uint32_t r = 0u; r < m; ++r) {
            const double sc = as[static_cast<std::size_t>(r) * (proxy.act_scale_stride / 4u)];
            for (uint32_t k = 0u; k < K; ++k) {
                const uint8_t c = ac[dflash2_int2_test::act_code_index(r, k,
                                                                       proxy.act_code_stride)];
                act[static_cast<std::size_t>(s0 + r) * K + k] =
                    static_cast<double>(ps::quantization::fpx::e4m3_decode_f32(c)) * sc;
            }
        }
    }

    for (MethodParams& mp : methods) {
        ReplayMetrics rm;
        replay_method(s, mp, probe_rows, act, coarse, oracle, error_l2, act_l2, 1e-3f, 1e-5f,
                      rm);
        std::vector<float> slack = rm.slack;
        finish_dist(slack);
        std::printf(
            "REPLAY method=%s param=%u rows=%llu direct_cert=%llu "
            "pool32=%.4f pool64=%.4f pool128=%.4f mismatch=%llu/%llu/%llu "
            "recall_upper=%.4f/%.4f/%.4f recall_score=%.4f/%.4f/%.4f "
            "bound_violation=%llu slack_p50=%.4f slack_p90=%.4f\n",
            mp.name, mp.param, static_cast<unsigned long long>(rm.rows),
            static_cast<unsigned long long>(rm.direct_cert),
            rm.rows ? static_cast<double>(rm.cert[0]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.cert[1]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.cert[2]) / rm.rows : 0.0,
            static_cast<unsigned long long>(rm.mismatch[0]),
            static_cast<unsigned long long>(rm.mismatch[1]),
            static_cast<unsigned long long>(rm.mismatch[2]),
            rm.rows ? static_cast<double>(rm.recall_upper[0]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.recall_upper[1]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.recall_upper[2]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.recall_score[0]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.recall_score[1]) / rm.rows : 0.0,
            rm.rows ? static_cast<double>(rm.recall_score[2]) / rm.rows : 0.0,
            static_cast<unsigned long long>(rm.bound_violation), quantile(slack, 0.50),
            quantile(slack, 0.90));
    }

    (void)hipFree(normed);
    proxy.shutdown();
    (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    (void)ps::qwen35::executor_shutdown(ex);
}

}  // namespace

int main() {    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (target_dir == nullptr || target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2_TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    const uint32_t subset_rows = std::getenv("PHASESHIFT_RESIDUAL_SUBSET") != nullptr
                                     ? static_cast<uint32_t>(std::strtoul(
                                           std::getenv("PHASESHIFT_RESIDUAL_SUBSET"), nullptr, 10))
                                     : 8192u;

    auto arena_result = ps::gpu::GpuArena::create(0, 28ull * 1024ull * 1024ull * 1024ull);
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
    const ps::weights::MatrixWeight& w = model.weights().lm_head;

    Survey s;
    s.vocab = w.rows;
    s.k_padded = w.k_padded;
    s.scale_stride = w.storage_scale_stride_bytes;
    s.groups = s.k_padded / kW32;
    const std::size_t codes_bytes =
        ((static_cast<std::size_t>(s.vocab) + 15u) / 16u) * s.k_padded * 16u;
    const std::size_t scale_elems =
        ((static_cast<std::size_t>(s.vocab) + 15u) / 16u) * (s.k_padded / 2u);
    s.codes.resize(codes_bytes);
    s.scales.resize(scale_elems);
    if (hipMemcpy(s.codes.data(), w.codes.data<uint8_t>(), codes_bytes, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(s.scales.data(), w.scales.data<uint8_t>(), scale_elems * 2u,
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL weight copy\n");
        return 1;
    }

    uint64_t* hist = nullptr;
    (void)hipMalloc(&hist, 256u * sizeof(uint64_t));
    (void)hipMemset(hist, 0, 256u * sizeof(uint64_t));
    (void)ps::kernel::launch_dflash2_int2_code_histogram(
        w.codes.data<uint8_t>(), w.scales.data<uint8_t>(), s.vocab, s.k_padded,
        s.scale_stride, hist, stream);
    std::vector<uint64_t> host_hist(256u, 0ull);
    (void)hipMemcpy(host_hist.data(), hist, 256u * sizeof(uint64_t), hipMemcpyDeviceToHost);
    (void)hipFree(hist);
    s.codebook = ps::kernel::dflash2_int2_derive_codebook(host_hist.data(), false);
    uint32_t expand_table[256] = {};
    ps::kernel::dflash2_int2_build_tables(s.codebook, s.map_table, expand_table);
    ps::kernel::dflash2_int2_build_error_table(s.codebook, s.map_table, s.error_table);

    std::printf("SURVEY vocab=%u k=%u groups=%u subset=%u codebook=[%.4f %.4f %.4f %.4f]\n",
                s.vocab, s.k_padded, s.groups, subset_rows, s.codebook.decoded[0],
                s.codebook.decoded[1], s.codebook.decoded[2], s.codebook.decoded[3]);
    (void)hipStreamSynchronize(stream);

    SurveyResult sr;
    weight_survey(s, sr);
    double ref = 0.0;
    for (float v : sr.orig_l2) ref += v;
    ref /= sr.orig_l2.size();
    {
        std::vector<float> sorted = sr.orig_l2;
        finish_dist(sorted);
        print_dist("ORIG", sorted, ref);
    }
    for (std::size_t i = 0u; i < sr.topm_list.size(); ++i) {
        std::vector<float> sorted = sr.topm_tail[i];
        finish_dist(sorted);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "TOPM M=%u", sr.topm_list[i]);
        print_dist(tag, sorted, ref);
    }
    for (std::size_t i = 0u; i < sr.topg_list.size(); ++i) {
        std::vector<float> sorted = sr.topg_tail[i];
        finish_dist(sorted);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "TOPG G=%u", sr.topg_list[i]);
        print_dist(tag, sorted, ref);
    }

    std::vector<uint32_t> ranks = {4u, 8u, 16u, 32u, 64u};
    LowRank lr = low_rank_survey(s, subset_rows, 80u, ranks);
    std::printf("LOWRANK subset=%u total_energy=%.4f\n", subset_rows, lr.total_energy);
    double cumulative = 0.0;
    for (std::size_t i = 0u; i < ranks.size(); ++i) {
        cumulative += lr.sigma2[i];
        const double energy = lr.total_energy > 0.0 ? cumulative / lr.total_energy : 0.0;
        std::vector<float> sorted = lr.tail[i];
        finish_dist(sorted);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "LOWRANK r=%u", ranks[i]);
        print_dist(tag, sorted, ref);
        std::printf("LOWRANK_ENERGY r=%u captured=%.6f\n", ranks[i], energy);
    }

    const bool replay = std::getenv("PHASESHIFT_RESIDUAL_REPLAY") == nullptr ||
                        std::strtol(std::getenv("PHASESHIFT_RESIDUAL_REPLAY"), nullptr, 10) != 0;
    if (replay) {
        const uint32_t probe_rows = std::getenv("PHASESHIFT_RESIDUAL_PROBE") != nullptr
                                        ? static_cast<uint32_t>(std::strtoul(
                                              std::getenv("PHASESHIFT_RESIDUAL_PROBE"),
                                              nullptr, 10))
                                        : 128u;
        const uint32_t ctx = std::getenv("PHASESHIFT_RESIDUAL_CTX") != nullptr
                                 ? static_cast<uint32_t>(std::strtoul(
                                       std::getenv("PHASESHIFT_RESIDUAL_CTX"), nullptr, 10))
                                 : 512u;
        std::vector<MethodParams> methods;
        auto add_method = [&](MethodParams::Type t, uint32_t p, const char* n) {
            MethodParams mp;
            mp.type = t;
            mp.param = p;
            mp.name = n;
            methods.push_back(std::move(mp));
        };
        add_method(MethodParams::NONE, 0u, "none");
        add_method(MethodParams::TOPM, 64u, "topm64");
        add_method(MethodParams::TOPM, 256u, "topm256");
        add_method(MethodParams::TOPG, 16u, "topg16");
        add_method(MethodParams::TOPG, 32u, "topg32");
        for (MethodParams& mp : methods) {
            if (mp.type != MethodParams::NONE) build_meta(s, mp);
        }
        run_replay(s, model, arena, stream, probe_rows, ctx, methods, sr.orig_l2);
    }

    (void)hipStreamSynchronize(stream);
    (void)arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("test_lm_head_residual_survey: done\n");
    return 0;
}
