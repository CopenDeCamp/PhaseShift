#include <phaseshift/quantization/offline/dflash2_adapter.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

namespace ps::quantization::fpx {

Result<Architecture> DFlash2Adapter::detect(const std::string& config_json) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(config_json);
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument("bad config.json", __FILE__, __LINE__);
    }
    if (!root.contains("architectures") || !root["architectures"].is_array() ||
        root["architectures"].empty() || !root["architectures"].at(0).is_string()) {
        return Status::invalid_argument(
            "DFlash2 config missing valid architectures[0]", __FILE__, __LINE__);
    }
    const std::string arch = root["architectures"].at(0).get<std::string>();
    if (arch == "DFlash2DraftModel") {
        return Architecture::DFlash2Draft;
    }
    return Status::invalid_argument(
        ("unsupported dflash architecture: " + arch).c_str(), __FILE__, __LINE__);
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

TensorInfo backbone_linear(std::string_view rest) {
    TensorInfo info;
    if (rest == "self_attn.q_proj.weight" || rest == "self_attn.k_proj.weight" ||
        rest == "self_attn.v_proj.weight" || rest == "self_attn.o_proj.weight" ||
        rest == "mlp.gate_proj.weight" || rest == "mlp.up_proj.weight" ||
        rest == "mlp.down_proj.weight" ||
        rest == "attention_conv.kernel_projection.weight" ||
        rest == "mlp_conv.kernel_projection.weight") {
        info.role = TensorRole::DFlashBackboneLinear;
        return info;
    }
    return info;
}

TensorInfo dflash_small(std::string_view rest) {
    TensorInfo info;
    if (rest == "hidden_norm.weight" || rest == "norm.weight" ||
        rest == "input_layernorm.weight" || rest == "post_attention_layernorm.weight" ||
        rest == "self_attn.q_norm.weight" || rest == "self_attn.k_norm.weight" ||
        rest == "attention_conv.base_kernel" || rest == "mlp_conv.base_kernel" ||
        rest == "candidate_selector.predecessor_codebook" ||
        rest == "candidate_selector.successor_codebook") {
        info.role = TensorRole::DFlashSmall;
        return info;
    }
    if (rest == "candidate_selector.hidden_projection.weight") {
        info.role = TensorRole::DFlashSelectorLinear;
        return info;
    }
    return info;
}

TensorInfo layer_tensor(std::string_view name) {
    TensorInfo info;
    const std::string_view prefix = "layers.";
    const size_t pos = prefix.size();
    const size_t dot = name.find('.', pos);
    if (dot == std::string_view::npos) {
        return info;
    }
    uint32_t layer = 0;
    if (!parse_uint32_strict(name.data() + pos, name.data() + dot, layer) ||
        layer > static_cast<uint32_t>(INT32_MAX)) {
        info.malformed_name = true;
        return info;
    }
    info.layer_index = static_cast<int32_t>(layer);
    const std::string_view rest = name.substr(dot + 1);
    TensorInfo lin = backbone_linear(rest);
    if (lin.role != TensorRole::Unknown) {
        lin.layer_index = info.layer_index;
        return lin;
    }
    TensorInfo small = dflash_small(rest);
    small.layer_index = info.layer_index;
    return small;
}

}  // namespace

TensorInfo DFlash2Adapter::classify(std::string_view name, const std::vector<int64_t>& shape) {
    TensorInfo info;
    if (starts_with(name, "layers.")) {
        info = layer_tensor(name);
    } else if (name == "fc.weight") {
        info.role = TensorRole::DFlashBackboneLinear;
    } else {
        info = dflash_small(name);
    }
    info.is_weight = !shape.empty();
    return info;
}

}  // namespace ps::quantization::fpx
