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
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace ps;

namespace {

struct CompareSpec {
    const char* name;
    double rel_l2_max;
    double cosine_min;
};

const std::vector<CompareSpec> kCompareOrder = {
    {"target_hidden", 0.0, 1.0},
    {"embedding_raw", 0.0, 1.0},
    {"embedding_norm", 1e-2, 0.9999},
    {"target_hidden_norm", 1e-2, 0.9999},
    {"concat_embedding_hidden", 1e-2, 0.9999},
    {"fc_out", 1e-2, 0.9999},
    {"input_layernorm_out", 1e-2, 0.9999},
    {"q_proj_raw", 1e-2, 0.9999},
    {"k_proj_raw", 1e-2, 0.9999},
    {"v_proj_raw", 1e-2, 0.9999},
    {"q_pre_norm", 1e-2, 0.9999},
    {"gate_pre_sigmoid", 1e-2, 0.9999},
    {"q_norm", 1e-2, 0.99999},
    {"k_norm", 1e-2, 0.99999},
    {"q_rope", 1e-2, 0.9999},
    {"k_rope", 1e-2, 0.9999},
    {"attention_core_out", 1e-2, 0.99999},
    {"gate_sigmoid", 1e-2, 0.99999},
    {"attention_gated", 1e-2, 0.9999},
    {"o_proj_out", 2e-2, 0.9999},
    {"residual1", 2e-2, 0.9999},
    {"post_attention_norm_out", 2e-2, 0.9999},
    {"mlp_gate_proj", 2e-2, 0.9999},
    {"mlp_up_proj", 2e-2, 0.9999},
    {"mlp_silu_gate", 2e-2, 0.9999},
    {"mlp_mul", 2e-2, 0.9999},
    {"mlp_down_proj", 2e-2, 0.9999},
    {"final_residual", 2e-2, 0.9999},
    {"selected_hidden", 2e-2, 0.9999},
    {"final_norm", 2e-2, 0.9999},
    {"logits", 2e-2, 0.9999},
};

const char* kCaseNames[] = {"position_0", "position_1", "position_127", "token_alt"};

struct TensorRef {
    std::vector<uint8_t> bytes;
    std::size_t features = 0;
    io::SType dtype = io::SType::BF16;
};

struct TensorDump {
    std::vector<uint8_t> bytes;
    uint32_t features = 0;
    ::ps::runtime::ValueDType dtype = ::ps::runtime::ValueDType::BF16;
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

std::vector<double> ref_to_f64(const TensorRef& t) {
    if (t.dtype == io::SType::BF16) {
        return bf16_to_f64(t.bytes.data(), t.features);
    }
    return f32_to_f64(t.bytes.data(), t.features);
}

std::vector<double> dump_to_f64(const TensorDump& t) {
    if (t.dtype == ::ps::runtime::ValueDType::BF16) {
        return bf16_to_f64(t.bytes.data(), t.features);
    }
    return f32_to_f64(t.bytes.data(), t.features);
}

std::string dump_dtype_name(::ps::runtime::ValueDType d) {
    switch (d) {
        case ::ps::runtime::ValueDType::BF16: return "bf16";
        case ::ps::runtime::ValueDType::F32: return "f32";
        case ::ps::runtime::ValueDType::I32: return "i32";
    }
    return "unknown";
}

struct CaseResult {
    bool pass = false;
    std::string first_mismatch;
    std::vector<std::pair<std::string, std::string>> failures;
    int ref_top1 = -1;
    int act_top1 = -1;
    int act_argmax = -1;
    bool top10_set_match = false;
    bool top10_rank_match = false;
    std::vector<int> ref_top10;
    std::vector<int> act_top10;

    void fail(const std::string& name, const std::string& why) {
        if (first_mismatch.empty()) first_mismatch = name;
        failures.push_back({name, why});
    }
};

std::unordered_map<uint32_t, std::string> build_value_name_map(
    const qwen35::Qwen35TextConfig& config,
    const qwen35::Qwen35ModelWeights& weights,
    std::string& err) {
    using ::ps::runtime::PrimitiveKind;
    auto lower = qwen35::lower_qwen35_mtp_to_primitives(config, weights);
    if (!lower.ok()) {
        err = lower.status().message();
        return {};
    }
    const auto& g = lower.value().graph;
    std::unordered_map<uint32_t, std::string> m;
    auto expect = [&](std::size_t idx, PrimitiveKind kind) -> bool {
        if (idx >= g.nodes.size()) {
            err = "graph has fewer nodes than expected at index " + std::to_string(idx);
            return false;
        }
        const auto k = ::ps::runtime::primitive_kind_of(g.nodes[idx].node);
        if (k != kind) {
            err = "node " + std::to_string(idx) + " kind mismatch: " +
                  ::ps::runtime::to_string(k) + " != " + ::ps::runtime::to_string(kind);
            return false;
        }
        return true;
    };
    auto name_out = [&](std::size_t idx, std::size_t out_idx, const char* name) -> bool {
        if (out_idx >= g.nodes[idx].outputs.size()) {
            err = "node " + std::to_string(idx) + " has no output " + std::to_string(out_idx);
            return false;
        }
        m[g.nodes[idx].outputs[out_idx].id] = name;
        return true;
    };

    if (!expect(0, PrimitiveKind::EMBEDDING_LOOKUP) || !name_out(0, 0, "embedding_raw")) return {};
    if (!expect(1, PrimitiveKind::RMS_NORM) || !name_out(1, 0, "embedding_norm")) return {};
    if (!expect(2, PrimitiveKind::RMS_NORM) || !name_out(2, 0, "target_hidden_norm")) return {};
    if (!expect(3, PrimitiveKind::CONCAT) || !name_out(3, 0, "concat_embedding_hidden")) return {};
    if (!expect(4, PrimitiveKind::LINEAR) || !name_out(4, 0, "fc_out")) return {};
    if (!expect(5, PrimitiveKind::RMS_NORM) || !name_out(5, 0, "input_layernorm_out")) return {};
    if (!expect(6, PrimitiveKind::LINEAR) || !name_out(6, 0, "q_proj_raw")) return {};
    if (!expect(7, PrimitiveKind::LINEAR) || !name_out(7, 0, "k_proj_raw")) return {};
    if (!expect(8, PrimitiveKind::LINEAR) || !name_out(8, 0, "v_proj_raw")) return {};
    if (!expect(9, PrimitiveKind::SPLIT) || !name_out(9, 0, "q_pre_norm") ||
        !name_out(9, 1, "gate_pre_sigmoid")) {
        return {};
    }
    if (!expect(10, PrimitiveKind::RMS_NORM) || !name_out(10, 0, "q_norm")) return {};
    if (!expect(11, PrimitiveKind::ROPE) || !name_out(11, 0, "q_rope")) return {};
    if (!expect(12, PrimitiveKind::RMS_NORM) || !name_out(12, 0, "k_norm")) return {};
    if (!expect(13, PrimitiveKind::ROPE) || !name_out(13, 0, "k_rope")) return {};
    if (!expect(14, PrimitiveKind::KV_APPEND)) return {};
    if (!expect(15, PrimitiveKind::PAGED_ATTENTION) || !name_out(15, 0, "attention_core_out")) return {};
    if (!expect(16, PrimitiveKind::SIGMOID) || !name_out(16, 0, "gate_sigmoid")) return {};
    if (!expect(17, PrimitiveKind::MUL) || !name_out(17, 0, "attention_gated")) return {};
    if (!expect(18, PrimitiveKind::LINEAR) || !name_out(18, 0, "o_proj_out")) return {};
    if (!expect(19, PrimitiveKind::RESIDUAL_ADD) || !name_out(19, 0, "residual1")) return {};
    if (!expect(20, PrimitiveKind::RMS_NORM) || !name_out(20, 0, "post_attention_norm_out")) return {};
    if (!expect(21, PrimitiveKind::LINEAR) || !name_out(21, 0, "mlp_gate_proj")) return {};
    if (!expect(22, PrimitiveKind::LINEAR) || !name_out(22, 0, "mlp_up_proj")) return {};
    if (!expect(23, PrimitiveKind::SWIGLU) || !name_out(23, 0, "mlp_mul")) return {};
    if (!expect(24, PrimitiveKind::LINEAR) || !name_out(24, 0, "mlp_down_proj")) return {};
    if (!expect(25, PrimitiveKind::RESIDUAL_ADD) || !name_out(25, 0, "final_residual")) return {};
    if (!expect(26, PrimitiveKind::OUTPUT_GATHER) || !name_out(26, 0, "selected_hidden")) return {};
    if (!expect(27, PrimitiveKind::RMS_NORM) || !name_out(27, 0, "final_norm")) return {};
    if (!expect(28, PrimitiveKind::LINEAR) || !name_out(28, 0, "logits")) return {};
    if (!expect(29, PrimitiveKind::SAMPLING) || !name_out(29, 0, "argmax")) return {};
    return m;
}

struct CaseInput {
    nlohmann::json meta;
    std::map<std::string, TensorRef> tensors;
};

bool load_case(const fs::path& case_dir, CaseInput& out, std::string& err) {
    const fs::path meta_path = case_dir / "metadata.json";
    const fs::path fixture_path = case_dir / "fixture.safetensors";
    if (!fs::is_regular_file(meta_path) || !fs::is_regular_file(fixture_path)) {
        err = "fixture files missing in " + case_dir.string() +
              " (run tools/reference/export_qwen35_mtp_gate1.py first)";
        return false;
    }
    {
        std::ifstream f(meta_path);
        try {
            f >> out.meta;
        } catch (const std::exception& e) {
            err = std::string("bad metadata.json: ") + e.what();
            return false;
        }
        for (const char* k : {"token_id", "position", "top1_token_id", "top10_token_ids"}) {
            if (!out.meta.contains(k)) {
                err = std::string("metadata.json missing key: ") + k;
                return false;
            }
        }
    }
    auto reader_res = io::SafetensorsReader::open(fixture_path.string());
    if (!reader_res.ok()) {
        err = "open fixture: " + reader_res.status().message();
        return false;
    }
    io::SafetensorsReader reader = reader_res.release();
    for (const auto& spec : kCompareOrder) {
        const std::string name = spec.name;
        auto s = reader.tensor_spec(name);
        if (!s.ok()) {
            err = "fixture missing tensor " + name + ": " + s.status().message();
            return false;
        }
        std::size_t bytes = 0;
        auto d = reader.tensor_data(name, bytes);
        if (!d.ok()) {
            err = "fixture read " + name + ": " + d.status().message();
            return false;
        }
        TensorRef t;
        t.dtype = s.value().dtype;
        t.features = bytes / (t.dtype == io::SType::BF16 ? 2u : 4u);
        t.bytes.assign(static_cast<const uint8_t*>(d.value()),
                       static_cast<const uint8_t*>(d.value()) + bytes);
        out.tensors[name] = std::move(t);
    }
    return true;
}

void compare_tensor(
    const std::string& name,
    const TensorRef& ref,
    const TensorDump* dump,
    const CompareSpec& spec,
    CaseResult& result) {
    if (dump == nullptr) {
        std::printf("  %-26s FAIL missing in dump\n", name.c_str());
        result.fail(name, "missing in dump");
        return;
    }
    if (spec.rel_l2_max == 0.0) {
        const bool exact =
            dump->bytes.size() == ref.bytes.size() &&
            std::memcmp(dump->bytes.data(), ref.bytes.data(), ref.bytes.size()) == 0;
        std::printf("  %-26s %s (bit-exact check)\n", name.c_str(), exact ? "PASS" : "FAIL");
        if (!exact) result.fail(name, "bit-exact mismatch");
        return;
    }
    const auto rf = ref_to_f64(ref);
    const auto af = dump_to_f64(*dump);
    if (rf.size() != af.size()) {
        std::printf("  %-26s FAIL size got=%zu want=%zu\n", name.c_str(), af.size(), rf.size());
        result.fail(name, "size mismatch");
        return;
    }
    double num = 0.0, den = 0.0, dot = 0.0, na = 0.0;
    double max_abs = 0.0;
    std::size_t worst = 0;
    bool finite = true;
    for (std::size_t i = 0; i < rf.size(); ++i) {
        const double d = rf[i] - af[i];
        num += d * d;
        den += rf[i] * rf[i];
        dot += rf[i] * af[i];
        na += af[i] * af[i];
        if (!std::isfinite(rf[i]) || !std::isfinite(af[i])) finite = false;
        if (std::abs(d) > max_abs) {
            max_abs = std::abs(d);
            worst = i;
        }
    }
    if (!finite) {
        std::printf("  %-26s FAIL non-finite\n", name.c_str());
        result.fail(name, "non-finite value");
        return;
    }
    const double rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    const double cosine = (den > 0.0 && na > 0.0) ? dot / std::sqrt(den * na) : 1.0;
    const bool ok = rel_l2 <= spec.rel_l2_max && cosine >= spec.cosine_min;
    char line[256];
    std::snprintf(line, sizeof(line),
                  "  %-26s %-7s rel_l2=%.3e max_abs=%.3e cosine=%.8f worst=%zu ref=%.6f act=%.6f",
                  name.c_str(), ok ? "PASS" : "FAIL", rel_l2, max_abs, cosine, worst,
                  rf[worst], af[worst]);
    std::printf("%s\n", line);
    if (!ok) {
        result.fail(name,
                    "rel_l2=" + std::to_string(rel_l2) + " cosine=" + std::to_string(cosine));
    }
}

std::string join_ints(const std::vector<int>& v) {
    std::string s;
    for (int x : v) {
        if (!s.empty()) s += ",";
        s += std::to_string(x);
    }
    return s;
}

std::vector<int> topk_of(const std::vector<double>& v, int k) {
    std::vector<std::size_t> idx(v.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::size_t>(k), idx.end(),
                      [&](std::size_t a, std::size_t b) { return v[a] > v[b]; });
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) out.push_back(static_cast<int>(idx[static_cast<std::size_t>(i)]));
    return out;
}

void write_dump(const fs::path& dump_dir, const std::map<std::string, TensorDump>& dumps) {
    fs::create_directories(dump_dir);
    std::ofstream info(dump_dir / "info.json");
    info << "{\n";
    bool first = true;
    for (const auto& [name, t] : dumps) {
        if (!first) info << ",\n";
        first = false;
        info << "  \"" << name << "\": {\"dtype\": \"" << dump_dtype_name(t.dtype)
             << "\", \"features\": " << t.features << "}";
        std::ofstream bin(dump_dir / (name + ".bin"), std::ios::binary);
        if (!t.bytes.empty()) {
            bin.write(reinterpret_cast<const char*>(t.bytes.data()),
                       static_cast<std::streamsize>(t.bytes.size()));
        }
    }
    info << "\n}\n";
}

}  // namespace

int main() {
    setenv("PHASESHIFT_QWEN35_KERNEL_MODE", "correctness", 1);

    const char* model_dir_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_dir_env == nullptr || model_dir_env[0] == '\0') {
        model_dir_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    if (model_dir_env == nullptr || model_dir_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP or PHASESHIFT_MODEL_DIR_4B to a "
                    "Qwen3.5 model directory\n");
        return 77;
    }
    const std::string model_dir = model_dir_env;
    const char* gate1_dir_env = std::getenv("PHASESHIFT_MTP_GATE1_DIR");
    const std::string gate1_dir =
        (gate1_dir_env != nullptr && gate1_dir_env[0] != '\0') ? gate1_dir_env : "artifacts/mtp_gate1";

    if (!fs::is_directory(model_dir)) {
        std::printf("FAIL: model dir not found: %s\n", model_dir.c_str());
        std::printf("MTP_GATE1_FORWARD: FAIL\n");
        return 1;
    }
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::string err;
    auto ar = gpu::GpuArena::create(0, 16ull * 1024ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("FAIL: arena: %s\n", ar.status().message().c_str());
        std::printf("MTP_GATE1_FORWARD: FAIL\n");
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    gpu::GpuArena arena = ar.release();

    std::printf("=== MTP Gate 1: 1-step draft forward vs vLLM reference ===\n");
    std::printf("model_dir: %s\n", model_dir.c_str());
    std::printf("gate1_dir: %s\n", gate1_dir.c_str());

    auto model_res = qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream);
    if (!model_res.ok()) {
        std::printf("FAIL: model load: %s\n", model_res.status().message().c_str());
        std::printf("MTP_GATE1_FORWARD: FAIL\n");
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    qwen35::Qwen35Model model = model_res.release();
    const qwen35::Qwen35TextConfig& config = model.text_config();
    const uint32_t H = static_cast<uint32_t>(config.hidden_size);
    std::printf("H=%u I=%u heads=%u kv_heads=%u head_dim=%u vocab=%zu\n",
                H, static_cast<uint32_t>(config.intermediate_size),
                static_cast<uint32_t>(config.num_attention_heads),
                static_cast<uint32_t>(config.num_key_value_heads),
                static_cast<uint32_t>(config.attention_head_dim), config.vocab_size);

    std::unordered_map<uint32_t, std::string> name_of =
        build_value_name_map(config, model.weights(), err);
    if (name_of.empty()) {
        std::printf("FAIL: value name map: %s\n", err.c_str());
        std::printf("MTP_GATE1_FORWARD: FAIL\n");
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }

    qwen35::runtime::MtpExecutorConfig ex_config;
    auto ex_res = qwen35::runtime::create_mtp_executor(
        model, qwen35::KVCacheDType::BF16, arena, ex_config, stream);
    if (!ex_res.ok()) {
        std::printf("FAIL: mtp executor: %s\n", ex_res.status().message().c_str());
        std::printf("MTP_GATE1_FORWARD: FAIL\n");
        arena.shutdown();
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    qwen35::runtime::MtpExecutor ex = ex_res.release();

    std::vector<CaseResult> case_results;
    bool any_case = false;
    for (const char* case_name : kCaseNames) {
        const fs::path case_dir = fs::path(gate1_dir) / case_name;
        CaseInput ci;
        if (!load_case(case_dir, ci, err)) {
            std::printf("CASE %s: FAIL (%s)\n", case_name, err.c_str());
            CaseResult r;
            r.fail(case_name, err);
            case_results.push_back(std::move(r));
            continue;
        }
        any_case = true;
        const int token = ci.meta.at("token_id").get<int>();
        const uint32_t position = ci.meta.at("position").get<uint32_t>();
        const int ref_top1 = ci.meta.at("top1_token_id").get<int>();
        std::vector<int> ref_top10 = ci.meta.at("top10_token_ids").get<std::vector<int>>();

        std::printf("\n--- case %s (token=%d position=%u) ---\n", case_name, token, position);

        CaseResult result;
        result.ref_top1 = ref_top1;
        result.ref_top10 = ref_top10;

        auto hidden_ref = ci.tensors.find("target_hidden");
        if (hidden_ref == ci.tensors.end() || hidden_ref->second.features != H) {
            std::printf("CASE %s: FAIL (bad target_hidden in fixture)\n", case_name);
            result.fail("target_hidden", "bad fixture");
            case_results.push_back(std::move(result));
            continue;
        }
        const TensorRef& hidden_ref_t = hidden_ref->second;

        std::vector<bf16_t> hidden_host;
        if (hidden_ref_t.dtype == io::SType::BF16) {
            hidden_host.resize(H);
            std::memcpy(hidden_host.data(), hidden_ref_t.bytes.data(), H * sizeof(bf16_t));
        } else {
            std::printf("CASE %s: FAIL (target_hidden dtype not BF16)\n", case_name);
            result.fail("target_hidden", "dtype");
            case_results.push_back(std::move(result));
            continue;
        }

        std::map<std::string, TensorDump> dumps;

        qwen35::runtime::ValueTraceSink sink;
        sink.on_value = [&](uint32_t value_id, const void* device_ptr, uint32_t row_stride,
                            uint32_t feature_count, ::ps::runtime::ValueDType dtype,
                            uint32_t rows, hipStream_t s) {
            (void)row_stride;
            (void)rows;
            auto it = name_of.find(value_id);
            if (it == name_of.end()) return;
            const std::size_t elem = (dtype == ::ps::runtime::ValueDType::BF16) ? 2u : 4u;
            const std::size_t bytes = static_cast<std::size_t>(feature_count) * elem;
            std::vector<uint8_t> host(bytes);
            if (bytes > 0) {
                if (hipStreamSynchronize(s) != hipSuccess) return;
                if (hipMemcpy(host.data(), device_ptr, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
                    return;
                }
            }
            TensorDump td;
            td.bytes = std::move(host);
            td.features = feature_count;
            td.dtype = dtype;
            dumps[it->second] = std::move(td);
        };

        const bf16_t* hidden_dev = nullptr;
        if (hipMemcpy(ex.hidden_in.data<bf16_t>(), hidden_host.data(),
                      H * sizeof(bf16_t), hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("CASE %s: FAIL (hidden upload)\n", case_name);
            result.fail("target_hidden", "upload");
            case_results.push_back(std::move(result));
            continue;
        }
        hidden_dev = ex.hidden_in.data<bf16_t>();

        auto run = qwen35::runtime::run_mtp_rows(
            ex, hidden_dev, &token, 1u, position, stream, &sink, /*kv_position_start=*/0u);
        if (!run.ok()) {
            std::printf("CASE %s: FAIL (run_mtp_rows: %s)\n", case_name,
                        run.status().message().c_str());
            result.fail("(run)", run.status().message());
            case_results.push_back(std::move(result));
            continue;
        }
        const int actual_draft = static_cast<int>(run.value());

        std::vector<uint8_t> hidden_check(H * sizeof(bf16_t));
        if (hipMemcpy(hidden_check.data(), ex.hidden_in.data<bf16_t>(), hidden_check.size(),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("CASE %s: FAIL (hidden check)\n", case_name);
            result.fail("target_hidden", "check");
            case_results.push_back(std::move(result));
            continue;
        }
        TensorDump hidden_dump;
        hidden_dump.bytes = std::move(hidden_check);
        hidden_dump.features = H;
        hidden_dump.dtype = ::ps::runtime::ValueDType::BF16;
        compare_tensor("target_hidden", hidden_ref_t, &hidden_dump,
                       kCompareOrder.front(), result);
        dumps["target_hidden"] = hidden_dump;

        for (const auto& spec : kCompareOrder) {
            if (std::strcmp(spec.name, "target_hidden") == 0) continue;
            if (std::strcmp(spec.name, "mlp_silu_gate") == 0) {
                auto g_it = dumps.find("mlp_gate_proj");
                auto ref_it = ci.tensors.find("mlp_silu_gate");
                if (g_it == dumps.end() || ref_it == ci.tensors.end()) {
                    std::printf("  %-26s FAIL missing (derived)\n", spec.name);
                    result.fail(spec.name, "missing (derived)");
                    continue;
                }
                const auto g = dump_to_f64(g_it->second);
                std::vector<double> silu(g.size());
                for (std::size_t i = 0; i < g.size(); ++i) {
                    silu[i] = g[i] / (1.0 + std::exp(-g[i]));
                }
                TensorDump derived;
                derived.features = static_cast<uint32_t>(silu.size());
                derived.dtype = ::ps::runtime::ValueDType::F32;
                derived.bytes.resize(silu.size() * 4);
                for (std::size_t i = 0; i < silu.size(); ++i) {
                    const float f = static_cast<float>(silu[i]);
                    std::memcpy(derived.bytes.data() + 4 * i, &f, sizeof(f));
                }
                compare_tensor(spec.name, ref_it->second, &derived, spec, result);
                continue;
            }
            auto d_it = dumps.find(spec.name);
            auto r_it = ci.tensors.find(spec.name);
            if (d_it == dumps.end()) {
                std::printf("  %-26s FAIL missing in dump\n", spec.name);
                result.fail(spec.name, "missing in dump");
                continue;
            }
            if (r_it == ci.tensors.end()) {
                std::printf("  %-26s FAIL missing in fixture\n", spec.name);
                result.fail(spec.name, "missing in fixture");
                continue;
            }
            compare_tensor(spec.name, r_it->second, &d_it->second, spec, result);
        }

        auto logits_it = dumps.find("logits");
        if (logits_it != dumps.end()) {
            const auto lg = dump_to_f64(logits_it->second);
            result.act_top10 = topk_of(lg, 10);
            result.act_top1 = result.act_top10.front();
            result.top10_set_match =
                std::set<int>(result.act_top10.begin(), result.act_top10.end()) ==
                std::set<int>(ref_top10.begin(), ref_top10.end());
            result.top10_rank_match = (result.act_top10 == ref_top10);
        }
        auto argmax_it = dumps.find("argmax");
        if (argmax_it != dumps.end() && argmax_it->second.bytes.size() >= 4) {
            int32_t a;
            std::memcpy(&a, argmax_it->second.bytes.data(), 4);
            result.act_argmax = static_cast<int>(a);
        }

        const std::string dump_dir =
            (fs::path(gate1_dir) / case_name / "pn_dump").string();
        write_dump(fs::path(dump_dir), dumps);

        std::printf("  DRAFT_TOKEN reference=%d actual=%d %s\n", ref_top1, actual_draft,
                    actual_draft == ref_top1 ? "MATCH" : "MISMATCH");
        if (result.act_argmax >= 0) {
            std::printf("  ARGMAX(sampling) reference=%d actual=%d %s\n", ref_top1,
                        result.act_argmax, result.act_argmax == ref_top1 ? "MATCH" : "MISMATCH");
        }
        if (result.act_top10.size() == 10) {
            std::printf("  TOP10 reference=%s actual=%s set=%s rank=%s\n",
                        join_ints(ref_top10).c_str(), join_ints(result.act_top10).c_str(),
                        result.top10_set_match ? "MATCH" : "MISMATCH",
                        result.top10_rank_match ? "MATCH" : "DIFF");
        }
        if (actual_draft != ref_top1) {
            result.fail("argmax", "draft token mismatch");
        }
        if (!result.top10_set_match && result.act_top10.size() == 10) {
            result.fail("top10", "top10 set mismatch");
        }
        result.pass = result.failures.empty();
        std::printf("CASE %s: %s\n", case_name, result.pass ? "PASS" : "FAIL");
        case_results.push_back(std::move(result));
    }

    bool all_pass = any_case;
    std::string first_mismatch;
    for (const auto& r : case_results) {
        if (!r.pass) {
            all_pass = false;
            if (first_mismatch.empty() && !r.first_mismatch.empty()) {
                first_mismatch = r.first_mismatch;
            }
        }
    }
    std::printf("\n");
    if (!all_pass) {
        for (const auto& r : case_results) {
            for (const auto& [name, why] : r.failures) {
                std::printf("FAIL %s: %s\n", name.c_str(), why.c_str());
            }
        }
        std::printf("FIRST_MISMATCH=%s\n", first_mismatch.c_str());
    }
    std::printf("MTP_GATE1_FORWARD: %s\n", all_pass ? "PASS" : "FAIL");

    (void)qwen35::runtime::mtp_executor_shutdown(ex);
    arena.shutdown();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return all_pass ? 0 : 1;
}
