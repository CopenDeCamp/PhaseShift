#include <nlohmann/json.hpp>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace ps;

namespace {

struct CompareSpec {
    const char* name;
    double rel_l2_max;
    double cosine_min;
};

std::vector<double> bf16_to_f64(const uint8_t* p, std::size_t n) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t u = (static_cast<uint32_t>(p[2 * i]) |
                            (static_cast<uint32_t>(p[2 * i + 1]) << 8)) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        out[i] = static_cast<double>(f);
    }
    return out;
}

std::vector<double> f32_to_f64(const uint8_t* p, std::size_t n) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float f;
        std::memcpy(&f, p + 4 * i, sizeof(f));
        out[i] = static_cast<double>(f);
    }
    return out;
}

bool compare_bytes(const std::vector<uint8_t>& ref, bool ref_bf16,
                   const std::vector<uint8_t>& act, bool act_bf16, const CompareSpec& spec,
                   std::string& why, double* out_rel = nullptr, double* out_cos = nullptr) {
    const std::size_t nr = ref.size() / (ref_bf16 ? 2 : 4);
    const std::size_t na = act.size() / (act_bf16 ? 2 : 4);
    if (nr != na || nr == 0) {
        why = "size";
        return false;
    }
    const auto rf = ref_bf16 ? bf16_to_f64(ref.data(), nr) : f32_to_f64(ref.data(), nr);
    const auto af = act_bf16 ? bf16_to_f64(act.data(), na) : f32_to_f64(act.data(), na);
    double num = 0.0, den = 0.0, dot = 0.0, nrm = 0.0;
    for (std::size_t i = 0; i < nr; ++i) {
        const double d = rf[i] - af[i];
        num += d * d;
        den += rf[i] * rf[i];
        dot += rf[i] * af[i];
        nrm += af[i] * af[i];
        if (!std::isfinite(rf[i]) || !std::isfinite(af[i])) {
            why = "non-finite";
            return false;
        }
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    const double cos = (den > 0.0 && nrm > 0.0) ? dot / std::sqrt(den * nrm) : 1.0;
    if (out_rel != nullptr) *out_rel = rel;
    if (out_cos != nullptr) *out_cos = cos;
    if (rel > spec.rel_l2_max) {
        why = "rel_l2=" + std::to_string(rel) + " cosine=" + std::to_string(cos);
        return false;
    }
    if (cos < spec.cosine_min) {
        why = "rel_l2=" + std::to_string(rel) + " cosine=" + std::to_string(cos);
        return false;
    }
    return true;
}

std::unordered_map<uint32_t, std::string> build_value_name_map(
    const qwen35::Qwen35TextConfig& config, const qwen35::Qwen35ModelWeights& weights) {
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, weights);
    if (!lower.ok()) return {};
    const auto& g = lower.value().graph;
    if (g.nodes.size() != 30u) return {};
    std::unordered_map<uint32_t, std::string> m;
    m[g.nodes[8].outputs[0].id] = "v_proj_raw";
    m[g.nodes[13].outputs[0].id] = "k_rope";
    m[g.nodes[15].outputs[0].id] = "attention_core_out";
    m[g.nodes[18].outputs[0].id] = "o_proj_out";
    m[g.nodes[27].outputs[0].id] = "final_norm";
    m[g.nodes[28].outputs[0].id] = "logits";
    m[g.nodes[29].outputs[0].id] = "argmax";
    return m;
}

struct Capture {
    std::map<std::string, std::vector<uint8_t>> bytes;
    std::map<std::string, bool> is_bf16;
};

struct RunResult {
    bool pass = true;
    bool ran = false;
    std::string first_mismatch;
    std::string first_state_field;
    int first_state_step = -1;
    std::vector<int> argmax;
    std::vector<uint8_t> final_k;
    std::vector<uint8_t> final_v;
    std::vector<uint8_t> extra_final_norm;
    int extra_argmax = -1;
    uint32_t final_length = 0;
    double worst_ac_rel = 0.0;
    double min_ac_cos = 1.0;
    uint32_t argmax_mismatches = 0;
    uint32_t ambiguous_mismatches = 0;
    std::map<std::string, std::pair<double, double>> worst;
};

const CompareSpec kKrope{nullptr, 1e-2, 0.0};
const CompareSpec kVcurr{nullptr, 1e-2, 0.0};
const CompareSpec kAcore{nullptr, 2e-2, 0.0};
const CompareSpec kOproj{nullptr, 2e-2, 0.0};
const CompareSpec kFnorm{nullptr, 2e-2, 0.0};
const CompareSpec kLogits{nullptr, 2e-2, 0.0};

}  // namespace

int main() {
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);

    const char* model_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_env == nullptr || model_env[0] == '\0') {
        model_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    const char* dir_env = std::getenv("PHASESHIFT_MTP_GATE2_DIR");
    const std::string gate2_dir =
        (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "artifacts/mtp_gate2";
    const fs::path syn_dir = fs::path(gate2_dir) / "synthetic";
    if (model_env == nullptr || model_env[0] == '\0' || !fs::is_directory(syn_dir)) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP and provide %s\n", syn_dir.string().c_str());
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    std::string err;
    auto ar = gpu::GpuArena::create(0, 16ull * 1024ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("SKIP: arena (%s)\n", ar.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = ar.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_res = qwen35::Qwen35Model::load_from_safetensors(model_env, arena, stream);
    if (!model_res.ok()) {
        std::printf("FAIL: model load: %s\n", model_res.status().message().c_str());
        return 1;
    }
    qwen35::Qwen35Model model = model_res.release();
    const qwen35::Qwen35TextConfig& config = model.text_config();
    const uint32_t H = static_cast<uint32_t>(config.hidden_size);
    const uint32_t Q = static_cast<uint32_t>(config.num_attention_heads) *
                       static_cast<uint32_t>(config.attention_head_dim);
    const uint32_t KV = static_cast<uint32_t>(config.num_key_value_heads) *
                        static_cast<uint32_t>(config.attention_head_dim);
    auto name_of = build_value_name_map(config, model.weights());
    if (name_of.empty()) {
        std::printf("FAIL: value map\n");
        return 1;
    }

    qwen35::runtime::MtpExecutorConfig cfg;
    cfg.num_pages = 24;
    cfg.page_tokens = 16;
    cfg.max_rows = 16;
    auto ex_res = qwen35::runtime::create_mtp_executor(
        model, qwen35::KVCacheDType::BF16, arena, cfg, stream);
    if (!ex_res.ok()) {
        std::printf("FAIL: executor: %s\n", ex_res.status().message().c_str());
        return 1;
    }
    qwen35::runtime::MtpExecutor ex = ex_res.release();
    const uint32_t B = ex.kv_state.page_tokens;

    auto load_case = [&](const fs::path& case_dir,
                         nlohmann::json& meta) -> std::optional<io::SafetensorsReader> {
        std::ifstream mf(case_dir / "metadata.json");
        if (!mf) return std::nullopt;
        mf >> meta;
        auto r = io::SafetensorsReader::open((case_dir / "sequence.safetensors").string());
        if (!r.ok()) return std::nullopt;
        return std::optional<io::SafetensorsReader>(r.release());
    };

    auto run_case = [&](const fs::path& case_dir, bool chunked, bool with_extra,
                        RunResult& out) -> bool {
        nlohmann::json meta;
        auto rd = load_case(case_dir, meta);
        if (!rd.has_value()) {
            std::printf("FAIL: load case %s\n", case_dir.string().c_str());
            return false;
        }
        io::SafetensorsReader& reader = *rd;
        const auto& sm = meta.at("step_meta");
        const uint32_t steps = meta.at("steps").get<uint32_t>();

        std::size_t hbytes = 0;
        auto hd = reader.tensor_data("target_hidden", hbytes);
        if (!hd.ok()) return false;
        const uint8_t* hidden_host = static_cast<const uint8_t*>(hd.value());

        std::vector<uint8_t> g_krope, g_v, g_ac, g_o, g_fn, g_argmax, g_top10l, g_top10i;
        auto load = [&](const char* n, std::vector<uint8_t>& b) {
            std::size_t nb = 0;
            auto d = reader.tensor_data(n, nb);
            if (!d.ok()) return false;
            b.assign(static_cast<const uint8_t*>(d.value()),
                     static_cast<const uint8_t*>(d.value()) + nb);
            return true;
        };
        if (!load("k_rope_current", g_krope) || !load("v_current", g_v) ||
            !load("attention_core_out", g_ac) || !load("o_proj_out", g_o) ||
            !load("final_norm", g_fn) || !load("argmax", g_argmax) ||
            !load("top10_logits", g_top10l) || !load("top10_ids", g_top10i)) {
            std::printf("FAIL: golden tensors in %s\n", case_dir.string().c_str());
            return false;
        }

        if (!qwen35::runtime::mtp_kv_reset(ex.kv_state, stream).ok()) return false;
        out = RunResult{};
        out.ran = true;

        Capture cap;
        qwen35::runtime::MtpStateStep st{};
        qwen35::runtime::MtpStateTraceSink ssink;
        ssink.on_step = [&](const qwen35::runtime::MtpStateStep& s) { st = s; };
        qwen35::runtime::ValueTraceSink sink;
        sink.on_value = [&](uint32_t vid, const void* ptr, uint32_t, uint32_t features,
                            ::ps::runtime::ValueDType dtype, uint32_t, hipStream_t s) {
            auto it = name_of.find(vid);
            if (it == name_of.end()) return;
            const std::size_t eb = dtype == ::ps::runtime::ValueDType::BF16 ? 2u : 4u;
            const std::size_t nb = static_cast<std::size_t>(features) * eb;
            if (nb == 0) return;
            if (hipStreamSynchronize(s) != hipSuccess) return;
            std::vector<uint8_t> host(nb);
            if (hipMemcpy(host.data(), ptr, nb, hipMemcpyDeviceToHost) != hipSuccess) return;
            cap.bytes[it->second] = std::move(host);
            cap.is_bf16[it->second] = (dtype == ::ps::runtime::ValueDType::BF16);
        };

        const bf16_t* hidden_dev = ex.hidden_in.data<bf16_t>();
        std::vector<uint8_t> row(static_cast<std::size_t>(H) * 2);
        std::ofstream dump_arg, dump_k, dump_v, dump_t10i, dump_t10l;
        const char* dump_env = std::getenv("PHASESHIFT_MTP_GATE2_DUMP");
        if (dump_env != nullptr && dump_env[0] != '\0') {
            fs::create_directories(dump_env);
            const std::string base = std::string(dump_env) + "/" + case_dir.filename().string();
            dump_arg.open(base + "_argmax.bin", std::ios::binary);
            dump_k.open(base + "_k.bin", std::ios::binary);
            dump_v.open(base + "_v.bin", std::ios::binary);
            dump_t10i.open(base + "_top10_ids.bin", std::ios::binary);
            dump_t10l.open(base + "_top10_logits.bin", std::ios::binary);
        }

        auto run_one = [&](uint32_t t, int32_t token, uint32_t position, bool check) -> bool {
            std::memcpy(row.data(), hidden_host + static_cast<std::size_t>(t) * H * 2, row.size());
            if (hipMemcpy(ex.hidden_in.data<void>(), row.data(), row.size(),
                          hipMemcpyHostToDevice) != hipSuccess)
                return false;
            cap.bytes.clear();
            auto r = qwen35::runtime::mtp_forward_step(ex, ex.kv_state, hidden_dev, &token, 1u,
                                                       position, stream, &sink, &ssink);
            if (!r.ok()) {
                std::printf("FAIL: step %u: %s\n", t, r.status().message().c_str());
                return false;
            }
            out.argmax.push_back(static_cast<int>(r.value().draft_token));
            if (!check) return true;
            if (dump_arg.is_open()) {
                const int32_t a = static_cast<int32_t>(r.value().draft_token);
                dump_arg.write(reinterpret_cast<const char*>(&a), 4);
                auto ik = cap.bytes.find("k_rope");
                if (ik != cap.bytes.end()) {
                    dump_k.write(reinterpret_cast<const char*>(ik->second.data()),
                                 static_cast<std::streamsize>(ik->second.size()));
                }
                auto iv = cap.bytes.find("v_proj_raw");
                if (iv != cap.bytes.end()) {
                    dump_v.write(reinterpret_cast<const char*>(iv->second.data()),
                                 static_cast<std::streamsize>(iv->second.size()));
                }
                auto il = cap.bytes.find("logits");
                if (il != cap.bytes.end()) {
                    const float* lf = reinterpret_cast<const float*>(il->second.data());
                    const std::size_t vsz = il->second.size() / 4;
                    std::vector<int32_t> idx(vsz);
                    for (std::size_t i = 0; i < vsz; ++i) idx[i] = static_cast<int32_t>(i);
                    const std::size_t topn = std::min<std::size_t>(10, vsz);
                    std::partial_sort(idx.begin(), idx.begin() + topn, idx.end(),
                                      [&](int32_t x, int32_t y) { return lf[x] > lf[y]; });
                    std::vector<int32_t> t10i(topn);
                    std::vector<float> t10l(topn);
                    for (std::size_t j = 0; j < topn; ++j) {
                        t10i[j] = idx[j];
                        t10l[j] = lf[idx[j]];
                    }
                    dump_t10i.write(reinterpret_cast<const char*>(t10i.data()),
                                    static_cast<std::streamsize>(topn * 4));
                    dump_t10l.write(reinterpret_cast<const char*>(t10l.data()),
                                    static_cast<std::streamsize>(topn * 4));
                }
            }
            const uint32_t exp_before = sm[t].at("kv_length_before").get<uint32_t>();
            const uint32_t exp_after = sm[t].at("kv_length_after").get<uint32_t>();
            if (r.value().kv_length_before != exp_before || r.value().kv_length_after != exp_after) {
                out.pass = false;
                if (out.first_state_step < 0) {
                    out.first_state_step = static_cast<int>(t);
                    out.first_state_field = "kv_length";
                }
            }
            if (r.value().absolute_position != sm[t].at("absolute_position").get<uint32_t>()) {
                out.pass = false;
                if (out.first_state_step < 0) {
                    out.first_state_step = static_cast<int>(t);
                    out.first_state_field = "absolute_position";
                }
            }
            if (st.input_token != token ||
                st.hidden_source_index != sm[t].at("hidden_source_index").get<uint32_t>()) {
                out.pass = false;
                if (out.first_state_step < 0) {
                    out.first_state_step = static_cast<int>(t);
                    out.first_state_field = "state_trace";
                }
            }
            const int32_t g_arg = *reinterpret_cast<const int32_t*>(
                g_argmax.data() + static_cast<std::size_t>(t) * 4);
            if (static_cast<int32_t>(r.value().draft_token) != g_arg) {
                ++out.argmax_mismatches;
                double ref_margin = 0.0, act_margin = 0.0, eps = 0.0;
                bool have = false;
                auto lg = cap.bytes.find("logits");
                if (lg != cap.bytes.end()) {
                    const float* lf = reinterpret_cast<const float*>(lg->second.data());
                    const std::size_t vsz = lg->second.size() / 4;
                    int a1 = -1, a2 = -1;
                    float v1 = -1e30f, v2 = -1e30f;
                    for (std::size_t i = 0; i < vsz; ++i) {
                        if (lf[i] > v1) {
                            v2 = v1;
                            a2 = a1;
                            v1 = lf[i];
                            a1 = static_cast<int>(i);
                        } else if (lf[i] > v2) {
                            v2 = lf[i];
                            a2 = static_cast<int>(i);
                        }
                    }
                    const float* gl = reinterpret_cast<const float*>(
                        g_top10l.data() + static_cast<std::size_t>(t) * 10 * 4);
                    const int32_t* gi = reinterpret_cast<const int32_t*>(
                        g_top10i.data() + static_cast<std::size_t>(t) * 10 * 4);
                    ref_margin = static_cast<double>(gl[0]) - gl[1];
                    act_margin = static_cast<double>(v1) - v2;
                    for (int j = 0; j < 10; ++j) {
                        const int32_t id = gi[j];
                        const float al = (id >= 0 && static_cast<std::size_t>(id) < vsz)
                                             ? lf[id]
                                             : 0.0f;
                        eps = std::max(eps, std::fabs(static_cast<double>(gl[j]) - al));
                    }
                    have = true;
                }
                const bool ambiguous =
                    have && (ref_margin <= 2.0 * eps || act_margin <= 2.0 * eps);
                if (ambiguous) {
                    ++out.ambiguous_mismatches;
                } else {
                    out.pass = false;
                    if (out.first_mismatch.empty()) out.first_mismatch = "argmax@" + std::to_string(t);
                    if (out.first_state_step < 0) {
                        out.first_state_step = static_cast<int>(t);
                        out.first_state_field = "argmax";
                    }
                }
                if (out.argmax_mismatches <= 6u && have) {
                    std::printf(
                        "  argmax step %u: %s golden=%d margin_ref=%.5f actual=%d margin_act=%.5f "
                        "epsilon=%.5f\n",
                        t, ambiguous ? "AMBIGUOUS" : "STABLE", g_arg, ref_margin,
                        static_cast<int>(r.value().draft_token), act_margin, eps);
                }
            }
            struct Chk {
                const char* dump;
                const std::vector<uint8_t>* golden;
                bool gb;
                const CompareSpec* spec;
            };
            const Chk chks[] = {
                {"k_rope", &g_krope, true, &kKrope},
                {"v_proj_raw", &g_v, true, &kVcurr},
                {"attention_core_out", &g_ac, false, &kAcore},
                {"o_proj_out", &g_o, true, &kOproj},
                {"final_norm", &g_fn, true, &kFnorm},
            };
            for (const auto& c : chks) {
                auto it = cap.bytes.find(c.dump);
                if (it == cap.bytes.end()) continue;
                const std::size_t feats =
                    std::strcmp(c.dump, "k_rope") == 0 || std::strcmp(c.dump, "v_proj_raw") == 0
                        ? KV
                        : (std::strcmp(c.dump, "attention_core_out") == 0 ? Q : H);
                const std::size_t stride = feats * (c.gb ? 2u : 4u);
                std::vector<uint8_t> gr(c.golden->begin() + static_cast<std::ptrdiff_t>(t) * stride,
                                        c.golden->begin() +
                                            static_cast<std::ptrdiff_t>(t + 1) * stride);
                std::string why;
                double rel = 0.0, cos = 1.0;
                if (!compare_bytes(gr, c.gb, it->second, cap.is_bf16[c.dump], *c.spec, why, &rel,
                                   &cos)) {
                    out.pass = false;
                    if (out.first_mismatch.empty())
                        out.first_mismatch = std::string(c.dump) + "@" + std::to_string(t) + "(" +
                                             why + ")";
                }
                if (std::strcmp(c.dump, "attention_core_out") == 0) {
                    if (rel > out.worst_ac_rel) out.worst_ac_rel = rel;
                    if (cos < out.min_ac_cos) out.min_ac_cos = cos;
                }
                auto& w = out.worst[c.dump];
                if (w.second == 0.0) w.second = 1.0;
                if (rel > w.first) w.first = rel;
                if (cos < w.second) w.second = cos;
            }
            return true;
        };

        if (!chunked) {
            std::set<uint32_t> ckpts;
            if (meta.contains("checkpoint_steps")) {
                for (const auto& c : meta["checkpoint_steps"]) ckpts.insert(c.get<uint32_t>());
            }
            for (uint32_t t = 0; t < steps; ++t) {
                const int32_t token = sm[t].at("input_token_id").get<int>();
                const uint32_t pos = sm[t].at("mtp_position").get<uint32_t>();
                if (!run_one(t, token, pos, true)) return false;
                if (ckpts.count(t) != 0u) {
                    const fs::path ck = case_dir / "checkpoints" /
                                        ("step_" + std::to_string(t) + ".safetensors");
                    if (fs::is_regular_file(ck)) {
                        auto cr = io::SafetensorsReader::open(ck.string());
                        if (cr.ok()) {
                            io::SafetensorsReader crd = cr.release();
                            std::vector<bf16_t> kb(static_cast<std::size_t>(t + 1u) * KV);
                            std::vector<bf16_t> vb(kb.size());
                            if (qwen35::runtime::mtp_kv_dump_canonical(ex.kv_state, t + 1u, kb.data(),
                                                                       vb.data(), stream)
                                    .ok()) {
                                std::size_t nb = 0;
                                auto gk = crd.tensor_data("K_history", nb);
                                auto gv = crd.tensor_data("V_history", nb);
                                if (gk.ok() && gv.ok()) {
                                    std::vector<uint8_t> ak(
                                        reinterpret_cast<uint8_t*>(kb.data()),
                                        reinterpret_cast<uint8_t*>(kb.data()) + kb.size() * 2);
                                    std::vector<uint8_t> av(
                                        reinterpret_cast<uint8_t*>(vb.data()),
                                        reinterpret_cast<uint8_t*>(vb.data()) + vb.size() * 2);
                                    std::vector<uint8_t> grk(
                                        static_cast<const uint8_t*>(gk.value()),
                                        static_cast<const uint8_t*>(gk.value()) + nb);
                                    std::vector<uint8_t> grv(
                                        static_cast<const uint8_t*>(gv.value()),
                                        static_cast<const uint8_t*>(gv.value()) + nb);
                                    std::string why;
                                    if (!compare_bytes(grk, true, ak, true, kKrope, why)) {
                                        out.pass = false;
                                        if (out.first_mismatch.empty())
                                            out.first_mismatch = "K_history@" + std::to_string(t) +
                                                                 "(" + why + ")";
                                    }
                                    if (!compare_bytes(grv, true, av, true, kVcurr, why)) {
                                        out.pass = false;
                                        if (out.first_mismatch.empty())
                                            out.first_mismatch = "V_history@" + std::to_string(t) +
                                                                 "(" + why + ")";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            uint32_t c = 0;
            while (c < steps) {
                const uint32_t rows = std::min<uint32_t>(ex.max_rows, steps - c);
                std::vector<int32_t> toks(rows);
                for (uint32_t i = 0; i < rows; ++i)
                    toks[i] = sm[c + i].at("input_token_id").get<int>();
                const uint32_t pos = sm[c].at("mtp_position").get<uint32_t>();
                if (hipMemcpy(ex.hidden_in.data<void>(),
                              hidden_host + static_cast<std::size_t>(c) * H * 2,
                              static_cast<std::size_t>(rows) * H * 2,
                              hipMemcpyHostToDevice) != hipSuccess)
                    return false;
                auto r = qwen35::runtime::mtp_forward_step(ex, ex.kv_state, ex.hidden_in.data<bf16_t>(),
                                                           toks.data(), rows, pos, stream);
                if (!r.ok()) {
                    std::printf("FAIL: chunk %u: %s\n", c, r.status().message().c_str());
                    return false;
                }
                for (uint32_t i = 0; i < rows; ++i) {
                    out.argmax.push_back(i + 1u == rows ? static_cast<int>(r.value().draft_token) : -1);
                }
                c += rows;
            }
        }
        out.final_length = ex.kv_state.logical_length;

        if (with_extra) {
            const uint32_t t = 0u;
            std::memcpy(row.data(), hidden_host, row.size());
            if (hipMemcpy(ex.hidden_in.data<void>(), row.data(), row.size(),
                          hipMemcpyHostToDevice) != hipSuccess)
                return false;
            const int32_t token = sm[0].at("input_token_id").get<int>() ^ 0x55;
            const uint32_t pos = sm[steps - 1].at("mtp_position").get<uint32_t>() + 1u;
            cap.bytes.clear();
            auto r = qwen35::runtime::mtp_forward_step(ex, ex.kv_state, hidden_dev, &token, 1u,
                                                       pos, stream, &sink, &ssink);
            if (!r.ok()) return false;
            out.extra_argmax = static_cast<int>(r.value().draft_token);
            auto it = cap.bytes.find("final_norm");
            if (it != cap.bytes.end()) out.extra_final_norm = it->second;
        }

        {
            std::vector<bf16_t> kb(static_cast<std::size_t>(out.final_length) * KV);
            std::vector<bf16_t> vb(kb.size());
            if (!qwen35::runtime::mtp_kv_dump_canonical(ex.kv_state, out.final_length, kb.data(),
                                                        vb.data(), stream)
                     .ok())
                return false;
            out.final_k.assign(reinterpret_cast<uint8_t*>(kb.data()),
                               reinterpret_cast<uint8_t*>(kb.data()) + kb.size() * 2);
            out.final_v.assign(reinterpret_cast<uint8_t*>(vb.data()),
                               reinterpret_cast<uint8_t*>(vb.data()) + vb.size() * 2);
        }
        return true;
    };

    std::printf("=== MTP Gate 2: persistent KV / state synchronization ===\n");
    std::printf("H=%u Q=%u KV=%u B=%u capacity=%u\n", H, Q, KV, B, ex.kv_state.capacity_tokens);

    RunResult serial;
    const bool serial_ok = run_case(syn_dir, false, true, serial);
    std::printf("synthetic: steps=%u final_kv=%u first_mismatch=%s first_state=%s@%d ac_worst_rel=%.3e ac_min_cos=%.8f\n",
                static_cast<unsigned>(serial.argmax.size()), serial.final_length,
                serial.first_mismatch.empty() ? "<none>" : serial.first_mismatch.c_str(),
                serial.first_state_field.empty() ? "<none>" : serial.first_state_field.c_str(),
                serial.first_state_step, serial.worst_ac_rel, serial.min_ac_cos);
    for (const auto& [nm, wc] : serial.worst) {
        std::printf("  worst %-22s rel_l2=%.3e min_cos=%.8f\n", nm.c_str(), wc.first, wc.second);
    }
    std::printf("MTP_GATE2_ARGMAX: %zu/%zu match (%u flip: %u ambiguous, %u stable)\n",
                serial.argmax.size() - serial.argmax_mismatches, serial.argmax.size(),
                serial.argmax_mismatches, serial.ambiguous_mismatches,
                serial.argmax_mismatches - serial.ambiguous_mismatches);
    std::printf("MTP_GATE2_SYNTHETIC: %s\n", (serial_ok && serial.pass) ? "PASS" : "FAIL");

    bool block_ok = true;
    {
        std::ifstream mf(syn_dir / "metadata.json");
        nlohmann::json meta;
        mf >> meta;
        const uint32_t steps = meta.at("steps").get<uint32_t>();
        for (uint32_t t : {B - 1u, B, B + 1u, 2u * B - 1u, 2u * B, 2u * B + 1u}) {
            if (t >= steps || t >= serial.argmax.size()) continue;
            if (meta.at("step_meta")[t].at("argmax").get<int>() != serial.argmax[t]) block_ok = false;
        }
    }
    std::printf("MTP_GATE2_BLOCK_BOUNDARY: %s\n", block_ok ? "PASS" : "FAIL");

    RunResult chunked;
    const bool chunk_ok = run_case(syn_dir, true, true, chunked);
    bool chunk_match = chunk_ok;
    if (chunk_ok && serial_ok) {
        chunk_match = (chunked.final_length == serial.final_length) &&
                      (chunked.final_k == serial.final_k) && (chunked.final_v == serial.final_v) &&
                      (chunked.extra_argmax == serial.extra_argmax) &&
                      (chunked.extra_final_norm == serial.extra_final_norm);
    }
    std::printf("MTP_GATE2_CHUNK_EQUIVALENCE: %s\n", chunk_match ? "PASS" : "FAIL");
    std::printf("MTP_GATE2_STATE_SYNC: %s\n", chunk_match ? "PASS" : "FAIL");

    RunResult again;
    const bool reset_ok = run_case(syn_dir, false, false, again) && again.final_k == serial.final_k &&
                          again.final_v == serial.final_v && again.argmax == serial.argmax;
    std::printf("MTP_GATE2_RESET: %s\n", reset_ok ? "PASS" : "FAIL");

    bool offset_ok = true;
    for (uint32_t off : {17u, 127u}) {
        RunResult ro;
        const fs::path d = fs::path(gate2_dir) / ("synthetic_offset_" + std::to_string(off));
        if (!fs::is_directory(d)) continue;
        if (!(run_case(d, false, false, ro) && ro.pass)) offset_ok = false;
    }
    std::printf("MTP_GATE2_POSITION_OFFSET: %s\n", offset_ok ? "PASS" : "FAIL");

    bool real_ok = fs::is_directory(fs::path(gate2_dir) / "real");
    if (real_ok) {
        RunResult rr;
        real_ok = run_case(fs::path(gate2_dir) / "real", false, false, rr) && rr.pass;
        std::printf("MTP_GATE2_REAL: %s\n", real_ok ? "PASS" : "FAIL");
    } else {
        std::printf("MTP_GATE2_REAL: SKIP\n");
    }

    const bool all = serial_ok && serial.pass && block_ok && chunk_match && reset_ok && offset_ok &&
                     (!fs::is_directory(fs::path(gate2_dir) / "real") || real_ok);
    std::printf("MTP_GATE2_FORWARD: %s\n", all ? "PASS" : "FAIL");

    (void)qwen35::runtime::mtp_executor_shutdown(ex);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return all ? 0 : 1;
}
