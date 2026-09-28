#include <phaseshift/quantization/offline/adapter_dispatch.h>
#include <phaseshift/quantization/offline/dflash2_adapter.h>
#include <phaseshift/quantization/offline/qwen35_adapter.h>
#include <nlohmann/json.hpp>

namespace ps::quantization::fpx {

Result<Architecture> detect_quantization_architecture(const std::string& config_json) {
    auto dflash = DFlash2Adapter::detect(config_json);
    if (dflash.ok()) return dflash;
    return Qwen35Adapter::detect(config_json);
}

TensorInfo classify_quantization_tensor(
    Architecture architecture,
    std::string_view name,
    const std::vector<int64_t>& shape) {
    if (architecture == Architecture::DFlash2Draft) {
        return DFlash2Adapter::classify(name, shape);
    }
    return Qwen35Adapter::classify(name, shape);
}

Result<uint32_t> quantization_num_layers(
    Architecture architecture,
    const std::string& config_json) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(config_json);
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument("bad config.json", __FILE__, __LINE__);
    }
    const bool dflash = architecture == Architecture::DFlash2Draft;
    const char* where = dflash ? "DFlash2 config missing valid num_hidden_layers"
                               : "Qwen3.5 config missing valid text_config.num_hidden_layers";
    const nlohmann::json* holder = &root;
    if (!dflash) {
        if (!root.contains("text_config") || !root["text_config"].is_object()) {
            return Status::invalid_argument(where, __FILE__, __LINE__);
        }
        holder = &root["text_config"];
    }
    if (!holder->contains("num_hidden_layers") ||
        !(*holder)["num_hidden_layers"].is_number_unsigned() ||
        (*holder)["num_hidden_layers"].get<uint64_t>() == 0 ||
        (*holder)["num_hidden_layers"].get<uint64_t>() > UINT32_MAX) {
        return Status::invalid_argument(where, __FILE__, __LINE__);
    }
    return static_cast<uint32_t>((*holder)["num_hidden_layers"].get<uint64_t>());
}

std::string_view quantization_main_layer_prefix(Architecture architecture) {
    if (architecture == Architecture::DFlash2Draft) return "layers.";
    return "model.language_model.layers.";
}

}  // namespace ps::quantization::fpx
