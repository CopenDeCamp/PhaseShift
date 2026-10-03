#include <nlohmann/json.hpp>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace ps;

namespace {

namespace qwen35 = ps::qwen35;
namespace q35rt = ps::qwen35::runtime;

const char* kModelEnvNames[] = {"PHASESHIFT_MODEL_DIR_MTP", "PHASESHIFT_MODEL_DIR_4B"};

double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    if (den <= 0.0) return num <= 0.0 ? 0.0 : 1e30;
    return std::sqrt(num / den);
}

std::vector<double> bf16_bytes_to_f64(const uint8_t* p, std::size_t n) {
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

std::vector<double> f32_bytes_to_f64(const uint8_t* p, std::size_t n) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float f;
        std::memcpy(&f, p + 4 * i, sizeof(f));
        out[i] = static_cast<double>(f);
    }
    return out;
}

struct ArgmaxClass {
    bool ambiguous = true;
    double margin_ref = 0.0;
    double epsilon = 0.0;
};

ArgmaxClass classify_argmax(
    const std::vector<double>& actual_logits,
    const int32_t* ref_top10_ids,
    const double* ref_top10_logits) {
    int a1 = -1;
    int a2 = -1;
    double v1 = -1e30;
    double v2 = -1e30;
    for (std::size_t i = 0; i < actual_logits.size(); ++i) {
        const double v = actual_logits[i];
        if (v > v1) {
            v2 = v1;
            a2 = a1;
            v1 = v;
            a1 = static_cast<int>(i);
        } else if (v > v2) {
            v2 = v;
            a2 = static_cast<int>(i);
        }
    }
    (void)a1;
    (void)a2;
    const double act_margin = v1 - v2;
    const double ref_margin = ref_top10_logits[0] - ref_top10_logits[1];
    double eps = 0.0;
    for (int j = 0; j < 10; ++j) {
        const int32_t id = ref_top10_ids[j];
        const double actual = (id >= 0 && static_cast<std::size_t>(id) < actual_logits.size())
                                  ? actual_logits[id]
                                  : 0.0;
        eps = std::max(eps, std::fabs(ref_top10_logits[j] - actual));
    }
    ArgmaxClass c;
    c.margin_ref = ref_margin;
    c.epsilon = eps;
    c.ambiguous = (ref_margin <= 2.0 * eps) || (act_margin <= 2.0 * eps);
    return c;
}

struct Loaded {
    std::vector<uint8_t> committed_tokens;
    std::vector<uint8_t> committed_hidden;
    std::vector<uint8_t> first_token;
    std::vector<uint8_t> first_hidden;
    std::vector<uint8_t> input_token;
    std::vector<uint8_t> position;
    std::vector<uint8_t> kv_before;
    std::vector<uint8_t> kv_after;
    std::vector<uint8_t> draft;
    std::vector<uint8_t> hidden_out;
    std::vector<uint8_t> final_norm;
    std::vector<uint8_t> k_rope;
    std::vector<uint8_t> v;
    std::vector<uint8_t> top10_ids;
    std::vector<uint8_t> top10_logits;
    std::vector<uint8_t> committed_k;
    std::vector<uint8_t> committed_v;
};

}  // namespace

int main() {
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);

    const char* model_env = nullptr;
    for (const char* name : kModelEnvNames) {
        const char* v = std::getenv(name);
        if (v != nullptr && v[0] != '\0') {
            model_env = v;
            break;
        }
    }
    const char* dir_env = std::getenv("PHASESHIFT_MTP_GATE3_DIR");
    const std::string gate3_dir =
        (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "artifacts/mtp_gate3";
    if (model_env == nullptr || !fs::is_directory(gate3_dir)) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP and provide %s\n", gate3_dir.c_str());
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto arena_result = gpu::GpuArena::create(0, 16ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result = qwen35::Qwen35Model::load_from_safetensors(model_env, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL: model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    qwen35::Qwen35Model model = model_result.release();
    const qwen35::Qwen35TextConfig& text = model.text_config();
    const uint32_t head_dim = static_cast<uint32_t>(text.attention_head_dim);

    q35rt::MtpExecutorConfig ex_config;
    ex_config.num_pages = 16u;
    ex_config.page_tokens = 16u;
    ex_config.max_rows = 16u;
    auto ex_result = q35rt::create_mtp_executor(
        model, qwen35::KVCacheDType::BF16, arena, ex_config, stream);
    if (!ex_result.ok()) {
        std::printf("FAIL: create_mtp_executor: %s\n", ex_result.status().message().c_str());
        return 1;
    }
    q35rt::MtpExecutor executor = ex_result.release();

    const q35rt::MtpKvConfig kv_config{
        .num_pages = 16u,
        .page_tokens = 16u,
        .num_attention_layers = 1u,
        .kv_heads = 4u,
        .head_dim = head_dim,
        .dtype = qwen35::KVCacheDType::BF16,
    };

    int failures = 0;
    int cases_run = 0;
    for (const auto& entry : fs::directory_iterator(gate3_dir)) {
        if (!entry.is_directory()) continue;
        const fs::path case_dir = entry.path();
        const fs::path meta_path = case_dir / "metadata.json";
        const fs::path seq_path = case_dir / "sequence.safetensors";
        if (!fs::exists(meta_path) || !fs::exists(seq_path)) continue;

        std::ifstream mf(meta_path);
        nlohmann::json meta;
        mf >> meta;
        auto reader_result = io::SafetensorsReader::open(seq_path.string());
        if (!reader_result.ok()) {
            std::printf("FAIL: %s: open safetensors\n", case_dir.string().c_str());
            ++failures;
            continue;
        }
        io::SafetensorsReader reader = reader_result.release();

        const uint32_t committed_count = meta.at("committed_count").get<uint32_t>();
        const uint32_t draft_k = meta.at("draft_k").get<uint32_t>();
        const uint32_t kv_f = meta.at("kv_heads").get<uint32_t>() *
                              meta.at("head_dim").get<uint32_t>();

        Loaded data;
        auto load = [&](const char* name, std::vector<uint8_t>& out) -> bool {
            std::size_t bytes = 0;
            auto d = reader.tensor_data(name, bytes);
            if (!d.ok()) return false;
            out.assign(static_cast<const uint8_t*>(d.value()),
                       static_cast<const uint8_t*>(d.value()) + bytes);
            return true;
        };
        const bool loaded = load("first_token", data.first_token) &&
                            load("first_hidden", data.first_hidden) &&
                            load("input_token", data.input_token) &&
                            load("position", data.position) &&
                            load("kv_before", data.kv_before) &&
                            load("kv_after", data.kv_after) &&
                            load("draft", data.draft) &&
                            load("hidden_out", data.hidden_out) &&
                            load("final_norm", data.final_norm) &&
                            load("k_rope", data.k_rope) &&
                            load("v", data.v) &&
                            load("top10_ids", data.top10_ids) &&
                            load("top10_logits", data.top10_logits) &&
                            (committed_count == 0u ||
                             (load("committed_tokens", data.committed_tokens) &&
                              load("committed_hidden", data.committed_hidden) &&
                              load("committed_k", data.committed_k) &&
                              load("committed_v", data.committed_v)));
        if (!loaded) {
            std::printf("FAIL: %s: missing tensor\n", case_dir.string().c_str());
            ++failures;
            continue;
        }
        ++cases_run;

        auto state_result = q35rt::create_mtp_kv_state(arena, kv_config, 0u, stream);
        if (!state_result.ok()) {
            std::printf("FAIL: %s: create state\n", case_dir.string().c_str());
            ++failures;
            continue;
        }
        q35rt::MtpKvState state = state_result.release();

        const std::size_t hidden_elems = static_cast<std::size_t>(meta.at("H").get<uint32_t>());
        auto malloc_bytes = [&](std::size_t bytes, void*& out) -> bool {
            return hipMalloc(&out, bytes == 0 ? 1 : bytes) == hipSuccess;
        };
        void* committed_hidden_dev = nullptr;
        void* first_hidden_dev = nullptr;
        if (!malloc_bytes(hidden_elems * 2u * std::max<uint32_t>(committed_count, 1u),
                          committed_hidden_dev) ||
            !malloc_bytes(hidden_elems * 2u, first_hidden_dev)) {
            std::printf("FAIL: %s: hipMalloc\n", case_dir.string().c_str());
            ++failures;
            continue;
        }
        if (committed_count > 0) {
            hipError_t err = hipMemcpy(
                committed_hidden_dev, data.committed_hidden.data(),
                committed_count * hidden_elems * 2u, hipMemcpyHostToDevice);
            if (err != hipSuccess) {
                std::printf("FAIL: %s: committed hidden upload\n", case_dir.string().c_str());
                ++failures;
                continue;
            }
            std::vector<int32_t> committed_tokens(committed_count);
            std::memcpy(committed_tokens.data(), data.committed_tokens.data(),
                        committed_count * sizeof(int32_t));
            auto st = q35rt::mtp_forward_step(
                executor, state, static_cast<const ps::bf16_t*>(committed_hidden_dev),
                committed_tokens.data(), committed_count, 0u, stream);
            if (!st.ok()) {
                std::printf("FAIL: %s: teacher-force committed: %s\n", case_dir.string().c_str(),
                            st.status().message().c_str());
                ++failures;
                continue;
            }
        }
        hipError_t err = hipMemcpy(
            first_hidden_dev, data.first_hidden.data(), hidden_elems * 2u,
            hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            std::printf("FAIL: %s: first hidden upload\n", case_dir.string().c_str());
            ++failures;
            continue;
        }

        int32_t token = 0;
        std::memcpy(&token, data.first_token.data(), sizeof(int32_t));
        const bf16_t* hidden_dev = static_cast<const bf16_t*>(first_hidden_dev);

        std::vector<const int32_t*> ref_input(draft_k);
        std::vector<uint32_t> ref_position(draft_k);
        std::vector<const int32_t*> ref_kv_before(draft_k);
        std::vector<const int32_t*> ref_draft(draft_k);
        for (uint32_t k = 0; k < draft_k; ++k) {
            ref_input[k] = reinterpret_cast<const int32_t*>(
                data.input_token.data() + static_cast<std::size_t>(k) * 4);
            ref_position[k] = *reinterpret_cast<const int32_t*>(
                data.position.data() + static_cast<std::size_t>(k) * 4);
            ref_kv_before[k] = reinterpret_cast<const int32_t*>(
                data.kv_before.data() + static_cast<std::size_t>(k) * 4);
            ref_draft[k] = reinterpret_cast<const int32_t*>(
                data.draft.data() + static_cast<std::size_t>(k) * 4);
        }

        std::vector<double> prod_hidden(draft_k * hidden_elems);
        std::vector<double> prod_norm(draft_k * hidden_elems);
        std::vector<double> prod_logits(draft_k * static_cast<std::size_t>(text.vocab_size));
        bool case_ok = true;
        std::string first_state;
        int first_step = -1;
        std::vector<int32_t> actual_draft(draft_k, -1);

        for (uint32_t k = 0; k < draft_k; ++k) {
            const uint32_t position = committed_count + k;
            int32_t input_token = token;
            auto st = q35rt::mtp_forward_step(
                executor, state, hidden_dev, &input_token, 1u, position, stream);
            if (!st.ok()) {
                std::printf("FAIL: %s: draft step %u: %s\n", case_dir.string().c_str(), k,
                            st.status().message().c_str());
                case_ok = false;
                break;
            }
            actual_draft[k] = static_cast<int32_t>(st.value().draft_token);
            if (input_token != *ref_input[k] ||
                st.value().absolute_position != ref_position[k] ||
                st.value().kv_length_before != *reinterpret_cast<const uint32_t*>(ref_kv_before[k])) {
                if (case_ok) {
                    first_state = "step_alignment";
                    first_step = static_cast<int>(k);
                }
                case_ok = false;
            }
            std::vector<uint8_t> hidden_host(hidden_elems * 2u);
            std::vector<uint8_t> norm_host(hidden_elems * 2u);
            std::vector<float> logits_host(static_cast<std::size_t>(text.vocab_size));
            err = hipMemcpy(hidden_host.data(), executor.hidden_out.data<bf16_t>(),
                            hidden_elems * 2u, hipMemcpyDeviceToHost);
            err = err == hipSuccess
                      ? hipMemcpy(norm_host.data(), executor.hidden_out_normed.data<bf16_t>(),
                                  hidden_elems * 2u, hipMemcpyDeviceToHost)
                      : err;
            err = err == hipSuccess
                      ? hipMemcpy(logits_host.data(), executor.logits.data<float>(),
                                  logits_host.size() * sizeof(float), hipMemcpyDeviceToHost)
                      : err;
            if (err != hipSuccess) {
                std::printf("FAIL: %s: step %u copy: %s\n", case_dir.string().c_str(), k,
                            hipGetErrorString(err));
                case_ok = false;
                break;
            }
            const std::vector<double> ref_h = bf16_bytes_to_f64(
                data.hidden_out.data() + static_cast<std::size_t>(k) * hidden_elems * 2u,
                hidden_elems);
            const std::vector<double> ref_n = bf16_bytes_to_f64(
                data.final_norm.data() + static_cast<std::size_t>(k) * hidden_elems * 2u,
                hidden_elems);
            const std::vector<double> act_h = bf16_bytes_to_f64(hidden_host.data(), hidden_elems);
            const std::vector<double> act_n = bf16_bytes_to_f64(norm_host.data(), hidden_elems);
            const double rl_h = rel_l2(act_h, ref_h);
            const double rl_n = rel_l2(act_n, ref_n);
            for (uint32_t i = 0; i < hidden_elems; ++i) {
                prod_hidden[static_cast<std::size_t>(k) * hidden_elems + i] = act_h[i];
                prod_norm[static_cast<std::size_t>(k) * hidden_elems + i] = act_n[i];
            }
            std::vector<double> act_logits(logits_host.begin(), logits_host.end());
            std::copy(act_logits.begin(), act_logits.end(),
                      prod_logits.begin() + static_cast<std::size_t>(k) * text.vocab_size);
            const int32_t* ref_top10_ids = reinterpret_cast<const int32_t*>(
                data.top10_ids.data() + static_cast<std::size_t>(k) * 10 * 4);
            std::vector<double> ref_top10(10);
            for (int j = 0; j < 10; ++j) {
                float f;
                std::memcpy(&f, data.top10_logits.data() +
                                   (static_cast<std::size_t>(k) * 10 + j) * 4, sizeof(f));
                ref_top10[j] = static_cast<double>(f);
            }
            const ArgmaxClass cls = classify_argmax(act_logits, ref_top10_ids, ref_top10.data());
            if (actual_draft[k] != *ref_draft[k] && !cls.ambiguous) {
                if (case_ok) {
                    first_state = "argmax_stable";
                    first_step = static_cast<int>(k);
                }
                case_ok = false;
            }
            if (rl_h > 2e-2 || rl_n > 2e-2) {
                if (case_ok) {
                    first_state = (rl_h > 2e-2) ? "hidden_out" : "final_norm";
                    first_step = static_cast<int>(k);
                }
                case_ok = false;
            }
            hidden_dev = executor.hidden_out.data<bf16_t>();
            token = actual_draft[k];
        }

        std::vector<uint8_t> can_k(static_cast<std::size_t>(committed_count + draft_k) * kv_f * 2u);
        std::vector<uint8_t> can_v(can_k.size());
        auto dump = q35rt::mtp_kv_dump_canonical(
            state, committed_count + draft_k,
            reinterpret_cast<bf16_t*>(can_k.data()),
            reinterpret_cast<bf16_t*>(can_v.data()), stream);
        if (!dump.ok()) {
            std::printf("FAIL: %s: canonical dump\n", case_dir.string().c_str());
            ++failures;
            continue;
        }
        std::vector<double> exp_k;
        std::vector<double> exp_v;
        if (committed_count > 0) {
            const std::vector<double> ck = bf16_bytes_to_f64(
                data.committed_k.data(), committed_count * kv_f);
            const std::vector<double> cv = bf16_bytes_to_f64(
                data.committed_v.data(), committed_count * kv_f);
            exp_k.insert(exp_k.end(), ck.begin(), ck.end());
            exp_v.insert(exp_v.end(), cv.begin(), cv.end());
        }
        const std::vector<double> dk = bf16_bytes_to_f64(data.k_rope.data(), draft_k * kv_f);
        const std::vector<double> dv = bf16_bytes_to_f64(data.v.data(), draft_k * kv_f);
        exp_k.insert(exp_k.end(), dk.begin(), dk.end());
        exp_v.insert(exp_v.end(), dv.begin(), dv.end());
        const std::vector<double> act_k = bf16_bytes_to_f64(can_k.data(), exp_k.size());
        const std::vector<double> act_v = bf16_bytes_to_f64(can_v.data(), exp_v.size());
        const double rk = rel_l2(act_k, exp_k);
        const double rv = rel_l2(act_v, exp_v);
        if (rk > 1e-2 || rv > 1e-2) {
            if (case_ok) {
                first_state = (rk > 1e-2) ? "canonical_k" : "canonical_v";
                first_step = 0;
            }
            case_ok = false;
        }

        int argmax_match = 0;
        for (uint32_t k = 0; k < draft_k; ++k) {
            if (actual_draft[k] == *ref_draft[k]) ++argmax_match;
        }
        std::printf(
            "%s: %s committed=%u K=%u argmax=%d/%u k_rel=%.3e v_rel=%.3e\n",
            case_dir.filename().string().c_str(), case_ok ? "PASS" : "FAIL", committed_count,
            draft_k, argmax_match, draft_k, rk, rv);
        if (!case_ok) {
            std::printf("  FIRST_FAILED_STEP=%d FIRST_FAILED_STATE=%s\n", first_step,
                        first_state.c_str());
            ++failures;
        }
        (void)prod_hidden;
        (void)prod_norm;
        (void)prod_logits;
        ps::gpu::discard_cleanup_result(hipFree(committed_hidden_dev));
        ps::gpu::discard_cleanup_result(hipFree(first_hidden_dev));
    }

    if (cases_run == 0) {
        std::printf("SKIP: no cases in %s\n", gate3_dir.c_str());
        return 77;
    }
    std::printf("MTP_GATE3_MULTISTEP: %s (%d cases)\n", failures == 0 ? "PASS" : "FAIL",
                cases_run);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failures == 0 ? 0 : 1;
}
