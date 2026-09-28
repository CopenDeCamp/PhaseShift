#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

float bf16_bits_to_f32(uint16_t bits) {
    const uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float r;
    std::memcpy(&r, &ui, 4);
    return r;
}

float e4m3_to_f32(uint8_t x) {
    const int sign = (x & 0x80u) ? -1 : 1;
    const int exp = (x >> 3) & 0x0Fu;
    const int man = x & 0x07u;
    const float val = (exp == 0) ? (man * 0.125f) * std::ldexp(1.0f, -6)
                                 : (1.0f + man * 0.125f) * std::ldexp(1.0f, exp - 7);
    return static_cast<float>(sign) * val;
}

bool read_all(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) {
        std::fclose(f);
        return false;
    }
    out.resize(static_cast<std::size_t>(n));
    const bool ok = out.empty() || std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

struct ProbeMeta {
    bool fp8 = false;
    uint32_t layer = 0;
    uint32_t rows = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t q_features = 0;
    uint32_t q_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
    uint32_t scale_elems_per_layer = 0;
    double scale = 0.0;
};

bool parse_meta(const std::string& path, ProbeMeta& m) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return false;
    char line[256];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        char key[64];
        char val[160];
        if (std::sscanf(line, "%63[^=]=%159s", key, val) != 2) continue;
        auto u = [&]() { return static_cast<uint32_t>(std::strtoul(val, nullptr, 10)); };
        if (std::strcmp(key, "kv_dtype") == 0) m.fp8 = (std::strcmp(val, "fp8") == 0);
        else if (std::strcmp(key, "layer") == 0) m.layer = u();
        else if (std::strcmp(key, "rows") == 0) m.rows = u();
        else if (std::strcmp(key, "q_heads") == 0) m.q_heads = u();
        else if (std::strcmp(key, "kv_heads") == 0) m.kv_heads = u();
        else if (std::strcmp(key, "head_dim") == 0) m.head_dim = u();
        else if (std::strcmp(key, "page_tokens") == 0) m.page_tokens = u();
        else if (std::strcmp(key, "num_pages") == 0) m.num_pages = u();
        else if (std::strcmp(key, "num_attention_layers") == 0) m.num_attention_layers = u();
        else if (std::strcmp(key, "q_features") == 0) m.q_features = u();
        else if (std::strcmp(key, "q_row_stride") == 0) m.q_row_stride = u();
        else if (std::strcmp(key, "output_row_stride") == 0) m.output_row_stride = u();
        else if (std::strcmp(key, "block_table_stride") == 0) m.block_table_stride = u();
        else if (std::strcmp(key, "max_sequences") == 0) m.max_sequences = u();
        else if (std::strcmp(key, "elems_per_token") == 0) m.elems_per_token = u();
        else if (std::strcmp(key, "elems_per_page") == 0) m.elems_per_page = u();
        else if (std::strcmp(key, "elems_per_layer") == 0) m.elems_per_layer = u();
        else if (std::strcmp(key, "scale_elems_per_layer") == 0) m.scale_elems_per_layer = u();
        else if (std::strcmp(key, "scale") == 0) m.scale = std::strtod(val, nullptr);
    }
    std::fclose(f);
    return m.rows != 0 && m.q_heads != 0 && m.kv_heads != 0 && m.head_dim != 0 && m.page_tokens != 0;
}

struct Dump {
    ProbeMeta meta;
    std::vector<uint16_t> q;         // bf16 bits [rows * q_features]
    std::vector<float> k;            // decoded [elems_per_layer]
    std::vector<float> v;            // decoded [elems_per_layer]
    std::vector<uint32_t> block_tables;
    std::vector<uint32_t> row_positions;
    std::vector<uint32_t> row_sequence_slots;
    std::vector<float> dense_out;    // [rows * q_features] if present
    bool has_dense_out = false;
};

bool load_dump(const std::string& dir, Dump& d) {
    if (!parse_meta(dir + "/meta.txt", d.meta)) {
        std::fprintf(stderr, "failed to parse %s/meta.txt\n", dir.c_str());
        return false;
    }
    const ProbeMeta& m = d.meta;
    std::vector<uint8_t> raw;
    if (!read_all(dir + "/q.bin", raw) || raw.size() != static_cast<std::size_t>(m.rows) * m.q_features * 2) {
        std::fprintf(stderr, "q.bin missing/size mismatch\n");
        return false;
    }
    d.q.resize(raw.size() / 2);
    std::memcpy(d.q.data(), raw.data(), raw.size());

    std::vector<uint8_t> kraw, vraw;
    if (!read_all(dir + "/k.bin", kraw) || !read_all(dir + "/v.bin", vraw)) {
        std::fprintf(stderr, "k.bin/v.bin missing\n");
        return false;
    }
    d.k.resize(m.elems_per_layer);
    d.v.resize(m.elems_per_layer);
    if (m.fp8) {
        if (kraw.size() != m.elems_per_layer || vraw.size() != m.elems_per_layer) return false;
        std::vector<uint8_t> ksraw, vsraw;
        if (!read_all(dir + "/k_scale.bin", ksraw) || !read_all(dir + "/v_scale.bin", vsraw)) return false;
        const float* ks = reinterpret_cast<const float*>(ksraw.data());
        const float* vs = reinterpret_cast<const float*>(vsraw.data());
        const uint32_t per_token_scale = m.kv_heads;
        for (uint32_t t = 0; t < m.elems_per_layer / m.elems_per_token; ++t) {
            for (uint32_t h = 0; h < m.kv_heads; ++h) {
                const uint32_t si = t * per_token_scale + h;
                const uint32_t base = t * m.elems_per_token + h * m.head_dim;
                const float kscale = ks[si];
                const float vscale = vs[si];
                for (uint32_t dd = 0; dd < m.head_dim; ++dd) {
                    d.k[base + dd] = e4m3_to_f32(kraw[base + dd]) * kscale;
                    d.v[base + dd] = e4m3_to_f32(vraw[base + dd]) * vscale;
                }
            }
        }
    } else {
        if (kraw.size() != static_cast<std::size_t>(m.elems_per_layer) * 2 ||
            vraw.size() != static_cast<std::size_t>(m.elems_per_layer) * 2)
            return false;
        const uint16_t* kb = reinterpret_cast<const uint16_t*>(kraw.data());
        const uint16_t* vb = reinterpret_cast<const uint16_t*>(vraw.data());
        for (std::size_t i = 0; i < d.k.size(); ++i) {
            d.k[i] = bf16_bits_to_f32(kb[i]);
            d.v[i] = bf16_bits_to_f32(vb[i]);
        }
    }

    std::vector<uint8_t> braw, praw, sraw;
    if (!read_all(dir + "/block_tables.bin", braw) || !read_all(dir + "/row_positions.bin", praw) ||
        !read_all(dir + "/row_sequence_slots.bin", sraw))
        return false;
    d.block_tables.resize(braw.size() / 4);
    std::memcpy(d.block_tables.data(), braw.data(), braw.size());
    d.row_positions.resize(praw.size() / 4);
    std::memcpy(d.row_positions.data(), praw.data(), praw.size());
    d.row_sequence_slots.resize(sraw.size() / 4);
    std::memcpy(d.row_sequence_slots.data(), sraw.data(), sraw.size());

    std::vector<uint8_t> draw;
    if (read_all(dir + "/dense_out.bin", draw)) {
        const std::size_t n = static_cast<std::size_t>(m.rows) * m.q_features;
        d.dense_out.resize(n);
        if (draw.size() == n * 4) {
            std::memcpy(d.dense_out.data(), draw.data(), n * 4);
            d.has_dense_out = true;
        } else if (draw.size() == n * 2) {
            const uint16_t* db = reinterpret_cast<const uint16_t*>(draw.data());
            for (std::size_t i = 0; i < n; ++i) d.dense_out[i] = bf16_bits_to_f32(db[i]);
            d.has_dense_out = true;
        }
    }
    return true;
}

struct PagePart {
    float pmax = 0.0f;
    float psum = 0.0f;   // sum exp(s_i - pmax)
    float plse = 0.0f;   // logsumexp
    std::vector<float> vsum;  // sum exp(s_i-pmax) * V_i[0..hd)
};

struct ConfigResult {
    double rel_l2_sum = 0.0;
    double rel_l2_max = 0.0;
    double cos_min = 1.0;
    double cos_sum = 0.0;
    double mass_sum = 0.0;
    double mass_min = 1.0;
    double sel_ratio_sum = 0.0;
    double old_keep_sum = 0.0;
    double page_recall_sum = 0.0;
    double sel_pages_sum = 0.0;
    double total_old_sum = 0.0;
    std::vector<double> rel_l2_vals;
    std::vector<double> cos_vals;
    size_t heads = 0;
    size_t nonfinite = 0;
};

double percentile_of(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min<size_t>(static_cast<size_t>(std::ceil(idx)), v.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct Config {
    const char* policy = "oracle";  // oracle|per_q_head|group_max
    const char* budget_kind = "frac";  // frac|fixed
    double frac = 0.0;
    uint32_t fixed_budget = 0;
};

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / std::sqrt(na * nb);
}

}  // namespace

int run_paged_prune(int argc, char** argv);

int run_paged_prune(int argc, char** argv) {
    std::string dir;
    std::string out_path;
    uint32_t recent = 4096;
    uint32_t sink = 1;
    uint32_t max_rows = 8;
    uint32_t min_visible = 2048;
    std::vector<std::string> policies;
    std::vector<double> fracs;
    std::vector<uint32_t> fixed_budgets;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing arg for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--dir") dir = next();
        else if (a == "--out") out_path = next();
        else if (a == "--recent") recent = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--sink") sink = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--rows") max_rows = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--min-visible") min_visible = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--policies") {
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                policies.push_back(s.substr(p, q - p));
                p = q + 1;
            }
        } else if (a == "--fractions") {
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                fracs.push_back(std::stod(s.substr(p, q - p)));
                p = q + 1;
            }
        } else if (a == "--budgets") {
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                fixed_budgets.push_back(static_cast<uint32_t>(std::stoul(s.substr(p, q - p))));
                p = q + 1;
            }
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: phaseshift-bench paged-prune --dir DIR [--recent N] [--sink N]\n"
                        "       [--policies oracle,per_q_head,group_max] [--fractions 0.5,...]\n"
                        "       [--budgets 64,128,...] [--rows N] [--min-visible N] [--out PATH]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    if (dir.empty()) {
        std::fprintf(stderr, "--dir required\n");
        return 2;
    }
    if (policies.empty()) policies = {"oracle", "per_q_head", "group_max"};
    if (fracs.empty()) fracs = {0.5, 0.25, 0.125, 0.0625, 0.03125};

    Dump d;
    if (!load_dump(dir, d)) return 1;
    const ProbeMeta& m = d.meta;
    const uint32_t q_per_kv = m.q_heads / m.kv_heads;
    const uint32_t hd = m.head_dim;
    const uint32_t sf = m.scale_elems_per_layer;

    std::vector<uint32_t> rows;
    for (uint32_t r = 0; r < m.rows; ++r) {
        if (d.row_positions[r] + 1u >= min_visible) rows.push_back(r);
    }
    if (rows.empty()) {
        std::fprintf(stderr, "no rows with visible >= %u\n", min_visible);
        return 1;
    }
    if (rows.size() > max_rows) {
        std::vector<uint32_t> picked;
        for (uint32_t k = 0; k < max_rows; ++k) {
            const std::size_t idx = (rows.size() - 1) * k / (max_rows - 1 == 0 ? 1 : (max_rows - 1));
            picked.push_back(rows[idx]);
        }
        std::sort(picked.begin(), picked.end());
        picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
        rows = picked;
    }

    std::vector<Config> configs;
    for (const auto& pol : policies) {
        for (double fr : fracs) {
            Config c;
            c.policy = pol == "oracle" ? "oracle" : (pol == "group_max" ? "group_max" : "per_q_head");
            c.budget_kind = "frac";
            c.frac = fr;
            configs.push_back(c);
        }
        for (uint32_t b : fixed_budgets) {
            Config c;
            c.policy = pol == "oracle" ? "oracle" : (pol == "group_max" ? "group_max" : "per_q_head");
            c.budget_kind = "fixed";
            c.fixed_budget = b;
            configs.push_back(c);
        }
    }

    std::vector<ConfigResult> results(configs.size());
    double ref_rel_l2_max = 0.0;
    double ref_cos_min = 1.0;

    for (uint32_t ri : rows) {
        const uint32_t visible = d.row_positions[ri] + 1u;
        const uint32_t slot = d.row_sequence_slots[ri];
        const uint32_t n_pages = (visible + m.page_tokens - 1) / m.page_tokens;
        const uint32_t* bt = d.block_tables.data() + static_cast<std::size_t>(slot) * m.block_table_stride;
        const uint16_t* qrow = d.q.data() + static_cast<std::size_t>(ri) * m.q_features;

        // Per (head, page) exact partials. Only visible tokens contribute.
        std::vector<std::vector<PagePart>> parts(m.q_heads, std::vector<PagePart>(n_pages));
        std::vector<std::vector<float>> dense_out(m.q_heads, std::vector<float>(hd, 0.0f));
        std::vector<float> global_lse(m.q_heads, 0.0f);

        for (uint32_t h = 0; h < m.q_heads; ++h) {
            const uint32_t kh = h / q_per_kv;
            std::vector<float> qf(hd);
            for (uint32_t dd = 0; dd < hd; ++dd) qf[dd] = bf16_bits_to_f32(qrow[h * hd + dd]);

            float gmax = -INFINITY;
            for (uint32_t p = 0; p < n_pages; ++p) {
                const uint32_t phys = bt[p];
                const uint32_t lo = p * m.page_tokens;
                const uint32_t hi = std::min(lo + m.page_tokens, visible);
                PagePart part;
                float pmax = -INFINITY;
                for (uint32_t t = lo; t < hi; ++t) {
                    const float* kt = d.k.data() + (static_cast<std::size_t>(phys) * m.page_tokens + (t - lo)) * m.elems_per_token + kh * hd;
                    float s = 0.0f;
                    for (uint32_t dd = 0; dd < hd; ++dd) s += qf[dd] * kt[dd];
                    s *= static_cast<float>(m.scale);
                    if (s > pmax) pmax = s;
                }
                part.pmax = pmax;
                float psum = 0.0f;
                std::vector<float> vsum(hd, 0.0f);
                for (uint32_t t = lo; t < hi; ++t) {
                    const float* kt = d.k.data() + (static_cast<std::size_t>(phys) * m.page_tokens + (t - lo)) * m.elems_per_token + kh * hd;
                    const float* vt = d.v.data() + (static_cast<std::size_t>(phys) * m.page_tokens + (t - lo)) * m.elems_per_token + kh * hd;
                    float s = 0.0f;
                    for (uint32_t dd = 0; dd < hd; ++dd) s += qf[dd] * kt[dd];
                    s *= static_cast<float>(m.scale);
                    const float w = std::exp(s - pmax);
                    psum += w;
                    for (uint32_t dd = 0; dd < hd; ++dd) vsum[dd] += w * vt[dd];
                }
                part.psum = psum;
                part.plse = pmax + std::log(psum);
                part.vsum = std::move(vsum);
                parts[h][p] = std::move(part);
                if (part.plse > gmax) gmax = part.plse;
            }
            double gsum = 0.0;
            std::vector<double> vacc(hd, 0.0);
            for (uint32_t p = 0; p < n_pages; ++p) {
                const double w = std::exp(static_cast<double>(parts[h][p].pmax) - gmax);
                gsum += w * parts[h][p].psum;
                for (uint32_t dd = 0; dd < hd; ++dd) vacc[dd] += w * parts[h][p].vsum[dd];
            }
            global_lse[h] = static_cast<float>(gmax + std::log(gsum));
            for (uint32_t dd = 0; dd < hd; ++dd) dense_out[h][dd] = static_cast<float>(vacc[dd] / gsum);

            if (d.has_dense_out) {
                std::vector<float> ref(hd);
                for (uint32_t dd = 0; dd < hd; ++dd) ref[dd] = d.dense_out[static_cast<std::size_t>(ri) * m.q_features + h * hd + dd];
                double num = 0.0, den = 0.0;
                for (uint32_t dd = 0; dd < hd; ++dd) {
                    const double e = dense_out[h][dd] - ref[dd];
                    num += e * e;
                    den += static_cast<double>(ref[dd]) * ref[dd];
                }
                const double rel = std::sqrt(num / std::max(den, 1e-30));
                ref_rel_l2_max = std::max(ref_rel_l2_max, rel);
                ref_cos_min = std::min(ref_cos_min, cosine_similarity(dense_out[h], ref));
            }
        }

        // Always-on pages for this row.
        std::vector<char> always_on(n_pages, 0);
        always_on[n_pages - 1] = 1;  // current/partial page
        const uint32_t recent_start = visible > recent ? visible - recent : 0u;
        for (uint32_t p = recent_start / m.page_tokens; p < n_pages; ++p) always_on[p] = 1;
        const uint32_t sink_last = std::min<uint32_t>(sink, n_pages);
        for (uint32_t p = 0; p < sink_last; ++p) always_on[p] = 1;

        std::vector<uint32_t> eligible;
        for (uint32_t p = 0; p < n_pages; ++p) {
            const bool full = (p + 1u) * m.page_tokens <= visible;
            if (full && always_on[p] == 0) eligible.push_back(p);
        }
        const uint32_t eligible_count = static_cast<uint32_t>(eligible.size());

        // Min/max upper bounds per (head, page) for eligible pages.
        std::vector<std::vector<float>> upper(m.q_heads, std::vector<float>(eligible_count, 0.0f));
        for (uint32_t h = 0; h < m.q_heads; ++h) {
            const uint32_t kh = h / q_per_kv;
            std::vector<float> qf(hd);
            for (uint32_t dd = 0; dd < hd; ++dd) qf[dd] = bf16_bits_to_f32(qrow[h * hd + dd]);
            for (uint32_t e = 0; e < eligible_count; ++e) {
                const uint32_t p = eligible[e];
                const uint32_t phys = bt[p];
                std::vector<float> kmin(hd, INFINITY), kmax(hd, -INFINITY);
                for (uint32_t o = 0; o < m.page_tokens; ++o) {
                    const float* kt = d.k.data() + (static_cast<std::size_t>(phys) * m.page_tokens + o) * m.elems_per_token + kh * hd;
                    for (uint32_t dd = 0; dd < hd; ++dd) {
                        kmin[dd] = std::min(kmin[dd], kt[dd]);
                        kmax[dd] = std::max(kmax[dd], kt[dd]);
                    }
                }
                double acc = 0.0;
                for (uint32_t dd = 0; dd < hd; ++dd) {
                    acc += static_cast<double>(qf[dd]) * (qf[dd] >= 0.0f ? kmax[dd] : kmin[dd]);
                }
                upper[h][e] = static_cast<float>(acc * m.scale);
            }
        }
        std::vector<std::vector<float>> group_upper(m.kv_heads, std::vector<float>(eligible_count, 0.0f));
        for (uint32_t g = 0; g < m.kv_heads; ++g) {
            for (uint32_t e = 0; e < eligible_count; ++e) {
                float mx = -INFINITY;
                for (uint32_t j = 0; j < q_per_kv; ++j) mx = std::max(mx, upper[g * q_per_kv + j][e]);
                group_upper[g][e] = mx;
            }
        }

        // Oracle ranking: page_lse per head.
        for (std::size_t ci = 0; ci < configs.size(); ++ci) {
            const Config& cfg = configs[ci];
            ConfigResult& res = results[ci];
            for (uint32_t h = 0; h < m.q_heads; ++h) {
                const uint32_t g = h / q_per_kv;
                uint32_t budget = 0;
                if (eligible_count > 0) {
                    if (cfg.budget_kind == std::string("fixed"))
                        budget = std::min(cfg.fixed_budget, eligible_count);
                    else
                        budget = std::min<uint32_t>(eligible_count,
                                                    static_cast<uint32_t>(std::ceil(eligible_count * cfg.frac)));
                }
                std::vector<uint32_t> order(eligible_count);
                for (uint32_t e = 0; e < eligible_count; ++e) order[e] = e;
                if (std::string(cfg.policy) == "oracle") {
                    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
                        return parts[h][eligible[x]].plse > parts[h][eligible[y]].plse;
                    });
                } else if (std::string(cfg.policy) == "group_max") {
                    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
                        return group_upper[g][x] > group_upper[g][y];
                    });
                } else {
                    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
                        return upper[h][x] > upper[h][y];
                    });
                }
                std::vector<char> selected(n_pages, 0);
                for (uint32_t p = 0; p < n_pages; ++p) selected[p] = always_on[p];
                for (uint32_t k = 0; k < budget; ++k) selected[eligible[order[k]]] = 1;

                float smax = -INFINITY;
                for (uint32_t p = 0; p < n_pages; ++p)
                    if (selected[p]) smax = std::max(smax, parts[h][p].pmax);
                double denom = 0.0;
                std::vector<double> vout(hd, 0.0);
                double retained = 0.0;
                uint32_t selected_count = 0;
                for (uint32_t p = 0; p < n_pages; ++p) {
                    if (!selected[p]) continue;
                    ++selected_count;
                    const double w = std::exp(static_cast<double>(parts[h][p].pmax) - smax);
                    denom += w * parts[h][p].psum;
                    for (uint32_t dd = 0; dd < hd; ++dd) vout[dd] += w * parts[h][p].vsum[dd];
                    retained += std::exp(static_cast<double>(parts[h][p].plse) - global_lse[h]);
                }
                std::vector<float> out(hd);
                bool finite = true;
                for (uint32_t dd = 0; dd < hd; ++dd) {
                    out[dd] = static_cast<float>(vout[dd] / denom);
                    if (!std::isfinite(out[dd])) finite = false;
                }
                double num = 0.0, den = 0.0, max_abs = 0.0;
                for (uint32_t dd = 0; dd < hd; ++dd) {
                    const double e = out[dd] - dense_out[h][dd];
                    num += e * e;
                    den += static_cast<double>(dense_out[h][dd]) * dense_out[h][dd];
                    max_abs = std::max(max_abs, std::fabs(e));
                }
                const double rel = std::sqrt(num / std::max(den, 1e-30));
                const double cos = cosine_similarity(out, dense_out[h]);
                res.rel_l2_sum += rel;
                res.rel_l2_max = std::max(res.rel_l2_max, rel);
                res.cos_sum += cos;
                res.cos_min = std::min(res.cos_min, cos);
                res.mass_sum += retained;
                res.mass_min = std::min(res.mass_min, retained);
                res.sel_ratio_sum += static_cast<double>(selected_count) / std::max<uint32_t>(n_pages, 1u);
                res.old_keep_sum += eligible_count > 0
                    ? static_cast<double>(budget) / static_cast<double>(eligible_count) : 0.0;
                res.sel_pages_sum += selected_count;
                res.total_old_sum += eligible_count;
                res.rel_l2_vals.push_back(rel);
                res.cos_vals.push_back(cos);
                res.heads += 1;

                if (std::string(cfg.policy) != "oracle" && budget > 0) {
                    std::vector<uint32_t> oracle_order(eligible_count);
                    for (uint32_t e = 0; e < eligible_count; ++e) oracle_order[e] = e;
                    std::sort(oracle_order.begin(), oracle_order.end(), [&](uint32_t x, uint32_t y) {
                        return parts[h][eligible[x]].plse > parts[h][eligible[y]].plse;
                    });
                    std::vector<char> oracle_top(eligible_count, 0);
                    for (uint32_t k = 0; k < budget; ++k) oracle_top[oracle_order[k]] = 1;
                    uint32_t hit = 0;
                    for (uint32_t k = 0; k < budget; ++k)
                        if (oracle_top[order[k]]) ++hit;
                    res.page_recall_sum += static_cast<double>(hit) / std::max<uint32_t>(budget, 1u);
                } else {
                    res.page_recall_sum += 1.0;
                }
                if (!finite) res.nonfinite += 1;
            }
        }
    }

    uint32_t visible_max = 0u;
    for (uint32_t ri : rows) visible_max = std::max(visible_max, d.row_positions[ri] + 1u);

    const char* kHeader =
        "dir,layer,kv_dtype,visible,rows_analyzed,recent,sink,policy,budget_kind,budget_value,"
        "rel_l2_mean,rel_l2_p50,rel_l2_p90,rel_l2_p95,rel_l2_p99,rel_l2_max,cos_mean,cos_min,"
        "mass_mean,mass_min,old_keep,sel_ratio,sel_pages_mean,total_old_pages_mean,page_recall,nonfinite";

    auto emit = [&](FILE* f) {
        for (std::size_t ci = 0; ci < configs.size(); ++ci) {
            const ConfigResult& r = results[ci];
            if (r.heads == 0) continue;
            const double n = static_cast<double>(r.heads);
            char val[32];
            if (configs[ci].budget_kind == std::string("fixed"))
                std::snprintf(val, sizeof val, "%u", configs[ci].fixed_budget);
            else
                std::snprintf(val, sizeof val, "%.4f", configs[ci].frac);
            std::fprintf(f,
                         "%s,%u,%s,%u,%zu,%u,%u,%s,%s,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,"
                         "%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.6f,%zu\n",
                         dir.c_str(), m.layer, m.fp8 ? "fp8" : "bf16", visible_max, rows.size(), recent,
                         sink, configs[ci].policy, configs[ci].budget_kind, val,
                         r.rel_l2_sum / n,
                         percentile_of(r.rel_l2_vals, 0.50),
                         percentile_of(r.rel_l2_vals, 0.90),
                         percentile_of(r.rel_l2_vals, 0.95),
                         percentile_of(r.rel_l2_vals, 0.99),
                         r.rel_l2_max,
                         r.cos_sum / n, r.cos_min,
                         r.mass_sum / n, r.mass_min,
                         r.old_keep_sum / n, r.sel_ratio_sum / n,
                         r.sel_pages_sum / n, r.total_old_sum / n,
                         r.page_recall_sum / n, r.nonfinite);
        }
    };

    std::printf("probe dir=%s layer=%u kv=%s rows=%u q_heads=%u kv_heads=%u hd=%u pt=%u num_pages=%u\n",
                dir.c_str(), m.layer, m.fp8 ? "fp8" : "bf16", m.rows, m.q_heads, m.kv_heads, hd,
                m.page_tokens, m.num_pages);
    std::printf("analyzed_rows=%zu recent=%u sink=%u min_visible=%u visible_max=%u\n", rows.size(),
                recent, sink, min_visible, visible_max);
    if (d.has_dense_out)
        std::printf("dense_ref_check: max_rel_l2=%.3e min_cosine=%.6f\n", ref_rel_l2_max, ref_cos_min);
    std::printf("%s\n", kHeader);
    emit(stdout);

    if (!out_path.empty()) {
        FILE* f = std::fopen(out_path.c_str(), "a");
        if (f != nullptr) {
            if (std::ftell(f) == 0) std::fprintf(f, "%s\n", kHeader);
            emit(f);
            std::fclose(f);
        }
    }
    return 0;
}
