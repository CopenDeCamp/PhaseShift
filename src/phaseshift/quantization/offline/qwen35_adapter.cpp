#include <phaseshift/quantization/offline/qwen35_adapter.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

namespace ps::quantization::fpx {

Result<Architecture> Qwen35Adapter::detect(const std::string& config_json) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(config_json);
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument("bad config.json", __FILE__, __LINE__);
    }
    const std::string model_type = root.value("model_type", std::string());
    std::string text_type;
    if (root.contains("text_config") && root["text_config"].is_object()) {
        text_type = root["text_config"].value("model_type", std::string());
    }
    if (model_type == "qwen3_5" && text_type == "qwen3_5_text") {
        return Architecture::Qwen35Dense;
    }
    return Status::invalid_argument(
        ("unsupported architecture: model_type=" + model_type + " text_model_type=" + text_type).c_str(),
        __FILE__, __LINE__);
}

namespace {

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

bool parse_uint32_strict(const char* begin, const char* end, uint32_t& out) {
    if (begin == end) return false;
    uint32_t v = 0;
    for (const char* p = begin; p != end; ++p) {
        if (*p < '0' || *p > '9') return false;
        const uint32_t digit = static_cast<uint32_t>(*p - '0');
        if (v > (UINT32_MAX - digit) / 10u) return false;
        v = v * 10u + digit;
    }
    out = v;
    return true;
}

bool parse_layer_index(std::string_view idx_str, int32_t& out) {
    uint32_t layer = 0;
    if (!parse_uint32_strict(idx_str.data(), idx_str.data() + idx_str.size(), layer) ||
        layer > static_cast<uint32_t>(INT32_MAX)) {
        return false;
    }
    out = static_cast<int32_t>(layer);
    return true;
}

TensorInfo layer_tensor(std::string_view name, const std::vector<int64_t>& shape) {
    TensorInfo info;
    const std::string_view prefix = "model.language_model.layers.";
    const size_t pos = prefix.size();
    const size_t dot = name.find('.', pos);
    if (dot == std::string_view::npos) {
        return info;
    }
    const std::string_view idx_str = name.substr(pos, dot - pos);
    if (!parse_layer_index(idx_str, info.layer_index)) {
        info.malformed_name = true;
        return info;
    }
    const std::string rest(name.substr(dot + 1));
    info.is_weight = !shape.empty();
    if (rest == "input_layernorm.weight" || rest == "post_attention_layernorm.weight" ||
        rest == "self_attn.q_norm.weight" || rest == "self_attn.k_norm.weight" ||
        rest == "linear_attn.norm.weight") {
        info.role = TensorRole::Norm;
        return info;
    }
    if (rest == "linear_attn.dt_bias" || rest == "linear_attn.A_log") {
        info.role = TensorRole::Bias;
        return info;
    }
    if (rest == "self_attn.q_proj.weight") { info.role = TensorRole::AttnQ; return info; }
    if (rest == "self_attn.k_proj.weight") { info.role = TensorRole::AttnK; return info; }
    if (rest == "self_attn.v_proj.weight") { info.role = TensorRole::AttnV; return info; }
    if (rest == "self_attn.o_proj.weight") { info.role = TensorRole::AttnO; return info; }
    if (rest == "mlp.gate_proj.weight") { info.role = TensorRole::FfnGate; return info; }
    if (rest == "mlp.up_proj.weight") { info.role = TensorRole::FfnUp; return info; }
    if (rest == "mlp.down_proj.weight") { info.role = TensorRole::FfnDown; return info; }
    if (rest == "linear_attn.in_proj_qkv.weight" || rest == "linear_attn.in_proj_z.weight") {
        info.role = TensorRole::GdnQkvza;
        return info;
    }
    if (rest == "linear_attn.in_proj_a.weight" || rest == "linear_attn.in_proj_b.weight" ||
        rest == "linear_attn.conv1d.weight") {
        info.role = TensorRole::GdnSmall;
        return info;
    }
    if (rest == "linear_attn.out_proj.weight") { info.role = TensorRole::GdnOut; return info; }
    return info;
}

TensorInfo mtp_tensor(std::string_view name, const std::vector<int64_t>& shape) {
    TensorInfo info;
    info.is_weight = !shape.empty();
    const std::string_view prefix = "mtp.layers.";
    if (starts_with(name, prefix)) {
        const size_t pos = prefix.size();
        const size_t dot = name.find('.', pos);
        if (dot == std::string_view::npos) return info;
        const std::string_view idx_str = name.substr(pos, dot - pos);
        if (!parse_layer_index(idx_str, info.layer_index)) {
            info.malformed_name = true;
            return info;
        }
        const std::string rest(name.substr(dot + 1));
        if (rest == "self_attn.q_proj.weight" || rest == "self_attn.k_proj.weight" ||
            rest == "self_attn.v_proj.weight" || rest == "self_attn.o_proj.weight") {
            info.role = TensorRole::MtpAttention;
        } else if (rest == "mlp.gate_proj.weight" || rest == "mlp.up_proj.weight" ||
                   rest == "mlp.down_proj.weight") {
            info.role = TensorRole::MtpFfn;
        } else if (rest == "input_layernorm.weight" || rest == "post_attention_layernorm.weight" ||
                   rest == "self_attn.q_norm.weight" || rest == "self_attn.k_norm.weight") {
            info.role = TensorRole::MtpNorm;
        }
        return info;
    }
    if (name == "mtp.fc.weight") {
        info.role = TensorRole::MtpOutput;
        return info;
    }
    if (name == "mtp.pre_fc_norm_embedding.weight" || name == "mtp.pre_fc_norm_hidden.weight" ||
        name == "mtp.norm.weight") {
        info.role = TensorRole::MtpNorm;
        return info;
    }
    return info;
}

}  // namespace

TensorInfo Qwen35Adapter::classify(std::string_view name, const std::vector<int64_t>& shape) {
    TensorInfo info;
    info.is_weight = !shape.empty();
    if (starts_with(name, "model.visual.")) {
        info.is_visual = true;
        return info;
    }
    if (starts_with(name, "model.language_model.layers.")) {
        return layer_tensor(name, shape);
    }
    if (starts_with(name, "mtp.")) {
        return mtp_tensor(name, shape);
    }
    if (name == "model.language_model.embed_tokens.weight") {
        info.role = TensorRole::TokenEmbedding;
        return info;
    }
    if (name == "model.language_model.norm.weight") {
        info.role = TensorRole::Norm;
        return info;
    }
    if (name == "model.language_model.lm_head.weight" || name == "lm_head.weight") {
        info.role = TensorRole::Output;
        return info;
    }
    return info;
}

}  // namespace ps::quantization::fpx
