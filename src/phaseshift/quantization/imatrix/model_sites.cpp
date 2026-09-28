#include <phaseshift/quantization/imatrix/model_sites.h>
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <cstdint>

namespace ps::quantization::imatrix {

namespace {

bool layer_is_full_attention(const ps::qwen35::Qwen35TextConfig& config, std::size_t index) {
    if (!config.layer_types.empty() && config.layer_types.size() == config.num_hidden_layers)
        return config.layer_types[index] == 1;
    const std::size_t interval = config.full_attention_interval > 0 ? config.full_attention_interval : 4;
    return (index + 1) % interval == 0;
}

std::string layer_prefix(std::size_t layer_index) {
    return "model.language_model.layers." + std::to_string(layer_index) + ".";
}

}

std::vector<ImatrixSitePlan> build_site_plan(const ps::qwen35::Qwen35TextConfig& config) {
    const uint64_t hidden_k = config.hidden_size;
    const uint64_t inter_k = config.intermediate_size;
    const uint64_t attn_o_k = config.num_attention_heads * config.attention_head_dim;
    const uint64_t gdn_out_k = config.linear_num_value_heads * config.linear_value_head_dim;

    std::vector<ImatrixSitePlan> plan;
    plan.reserve(config.num_hidden_layers * 4u + 1u);
    for (std::size_t li = 0; li < config.num_hidden_layers; ++li) {
        const uint32_t layer = static_cast<uint32_t>(li);
        if (layer_is_full_attention(config, li)) {
            plan.push_back({imatrix_tag(layer, ImatrixSite::FullAttnQkvInput), hidden_k});
            plan.push_back({imatrix_tag(layer, ImatrixSite::FullAttnOInput), attn_o_k});
        } else {
            plan.push_back({imatrix_tag(layer, ImatrixSite::GdnProjectionInput), hidden_k});
            plan.push_back({imatrix_tag(layer, ImatrixSite::GdnOutputInput), gdn_out_k});
        }
        plan.push_back({imatrix_tag(layer, ImatrixSite::MlpGateUpInput), hidden_k});
        plan.push_back({imatrix_tag(layer, ImatrixSite::MlpDownInput), inter_k});
    }
    plan.push_back({imatrix_tag(kImatrixLayerNone, ImatrixSite::LmHeadInput), hidden_k});
    return plan;
}

std::vector<std::string> tensor_names_for(uint32_t tag) {
    const uint32_t layer = imatrix_tag_layer(tag);
    const ImatrixSite site = imatrix_tag_site(tag);
    if (site == ImatrixSite::LmHeadInput) return {"lm_head.weight"};
    const std::string prefix = layer_prefix(layer);
    switch (site) {
        case ImatrixSite::FullAttnQkvInput:
            return {prefix + "self_attn.q_proj.weight",
                    prefix + "self_attn.k_proj.weight",
                    prefix + "self_attn.v_proj.weight"};
        case ImatrixSite::FullAttnOInput:
            return {prefix + "self_attn.o_proj.weight"};
        case ImatrixSite::GdnProjectionInput:
            return {prefix + "linear_attn.in_proj_qkv.weight",
                    prefix + "linear_attn.in_proj_z.weight"};
        case ImatrixSite::GdnOutputInput:
            return {prefix + "linear_attn.out_proj.weight"};
        case ImatrixSite::MlpGateUpInput:
            return {prefix + "mlp.gate_proj.weight", prefix + "mlp.up_proj.weight"};
        case ImatrixSite::MlpDownInput:
            return {prefix + "mlp.down_proj.weight"};
        case ImatrixSite::LmHeadInput:
            break;
    }
    return {};
}

}
