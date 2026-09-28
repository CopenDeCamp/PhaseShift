#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/io/safetensors_writer.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ps::test {

struct SyntheticQwen35Spec {
    std::uint32_t vocab_size = 64;
    std::uint32_t hidden_size = 128;
    std::uint32_t intermediate_size = 64;
    std::uint32_t num_hidden_layers = 4;
    std::uint32_t linear_num_key_heads = 4;
    std::uint32_t linear_key_head_dim = 32;
    std::uint32_t linear_num_value_heads = 4;
    std::uint32_t linear_value_head_dim = 32;
    std::uint32_t linear_conv_kernel_dim = 4;
    std::uint32_t num_attention_heads = 4;
    std::uint32_t num_key_value_heads = 2;
    std::uint32_t attention_head_dim = 32;
    std::size_t full_attention_interval = 2;
    bool tie_word_embeddings = true;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;

    bool layer_is_gdn(std::uint32_t layer) const {
        if (full_attention_interval == 0) return layer % 4 < 3;
        return layer % full_attention_interval < full_attention_interval - 1;
    }

    std::uint32_t query_features() const {
        return num_attention_heads * attention_head_dim;
    }

    std::uint32_t kv_features() const {
        return num_key_value_heads * attention_head_dim;
    }

    std::uint32_t gdn_qk_features() const {
        return linear_num_key_heads * linear_key_head_dim;
    }

    std::uint32_t gdn_value_features() const {
        return linear_num_value_heads * linear_value_head_dim;
    }

    std::uint32_t gdn_conv_features() const {
        return 2 * gdn_qk_features() + gdn_value_features();
    }

    std::uint32_t attention_layer_count() const {
        std::uint32_t n = 0;
        for (std::uint32_t i = 0; i < num_hidden_layers; ++i)
            if (!layer_is_gdn(i)) ++n;
        return n;
    }

    std::uint32_t gdn_layer_count() const {
        return num_hidden_layers - attention_layer_count();
    }
};

namespace detail {

struct SyntheticTensor {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<std::uint16_t> bytes;
};

inline std::uint16_t to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

inline SyntheticTensor make_bf16(const std::string& name,
                                 const std::vector<std::size_t>& shape, float scale,
                                 std::uint32_t seed) {
    SyntheticTensor t;
    t.name = name;
    t.shape = shape;
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    t.bytes.resize(count);
    std::uint32_t state = seed * 2654435761u + 1013904223u;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f;
        t.bytes[i] = to_bf16((unit * 2.0f - 1.0f) * scale);
    }
    return t;
}

}

inline Status write_synthetic_qwen35_model(const std::string& model_dir,
                                           const SyntheticQwen35Spec& spec) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(model_dir, ec);
    if (ec)
        return Status::invalid_argument("cannot create synthetic model directory", __FILE__,
                                        __LINE__);

    {
        std::ofstream f(model_dir + "/config.json", std::ios::trunc);
        if (!f.is_open())
            return Status::invalid_argument("cannot write synthetic config.json", __FILE__,
                                            __LINE__);
        f << "{\n  \"tie_word_embeddings\": " << (spec.tie_word_embeddings ? "true" : "false")
          << ",\n  \"text_config\": {\n";
        f << "    \"vocab_size\": " << spec.vocab_size << ",\n";
        f << "    \"hidden_size\": " << spec.hidden_size << ",\n";
        f << "    \"intermediate_size\": " << spec.intermediate_size << ",\n";
        f << "    \"num_hidden_layers\": " << spec.num_hidden_layers << ",\n";
        f << "    \"full_attention_interval\": " << spec.full_attention_interval << ",\n";
        f << "    \"linear_num_key_heads\": " << spec.linear_num_key_heads << ",\n";
        f << "    \"linear_key_head_dim\": " << spec.linear_key_head_dim << ",\n";
        f << "    \"linear_num_value_heads\": " << spec.linear_num_value_heads << ",\n";
        f << "    \"linear_value_head_dim\": " << spec.linear_value_head_dim << ",\n";
        f << "    \"linear_conv_kernel_dim\": " << spec.linear_conv_kernel_dim << ",\n";
        f << "    \"num_attention_heads\": " << spec.num_attention_heads << ",\n";
        f << "    \"num_key_value_heads\": " << spec.num_key_value_heads << ",\n";
        f << "    \"head_dim\": " << spec.attention_head_dim << ",\n";
        f << "    \"rms_norm_eps\": " << spec.rms_norm_eps << ",\n";
        f << "    \"rope_parameters\": { \"rope_theta\": " << spec.rope_theta
          << ", \"partial_rotary_factor\": " << spec.partial_rotary_factor << " },\n";
        f << "    \"layer_types\": [";
        for (std::uint32_t i = 0; i < spec.num_hidden_layers; ++i) {
            if (i != 0) f << ", ";
            f << (spec.layer_is_gdn(i) ? "\"linear_attention\"" : "\"full_attention\"");
        }
        f << "]\n  }\n}\n";
        if (!f.good())
            return Status::invalid_argument("synthetic config.json write failed", __FILE__,
                                            __LINE__);
    }

    using detail::make_bf16;
    const std::uint32_t H = spec.hidden_size;
    const std::uint32_t I = spec.intermediate_size;
    const std::uint32_t V = spec.vocab_size;
    const std::uint32_t qf = spec.query_features();
    const std::uint32_t kvf = spec.kv_features();
    const std::uint32_t gqk = spec.gdn_qk_features();
    const std::uint32_t gval = spec.gdn_value_features();
    const std::uint32_t gconv = spec.gdn_conv_features();
    const std::uint32_t vg = spec.linear_num_value_heads;

    std::vector<detail::SyntheticTensor> tensors;
    std::uint32_t seed = 7;
    tensors.push_back(make_bf16("model.language_model.embed_tokens.weight", {V, H},
                                0.05f, seed++));
    tensors.push_back(make_bf16("model.language_model.norm.weight", {H}, 0.1f, seed++));

    const std::string lpfx = "model.language_model.layers.";
    for (std::uint32_t l = 0; l < spec.num_hidden_layers; ++l) {
        const std::string p = lpfx + std::to_string(l) + ".";
        tensors.push_back(make_bf16(p + "input_layernorm.weight", {H}, 0.1f, seed++));
        tensors.push_back(make_bf16(p + "post_attention_layernorm.weight", {H}, 0.1f, seed++));
        tensors.push_back(make_bf16(p + "mlp.gate_proj.weight", {I, H}, 0.05f, seed++));
        tensors.push_back(make_bf16(p + "mlp.up_proj.weight", {I, H}, 0.05f, seed++));
        tensors.push_back(make_bf16(p + "mlp.down_proj.weight", {H, I}, 0.05f, seed++));
        if (spec.layer_is_gdn(l)) {
            tensors.push_back(make_bf16(p + "linear_attn.in_proj_qkv.weight", {gconv, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.in_proj_z.weight", {gval, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.in_proj_b.weight", {vg, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.in_proj_a.weight", {vg, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.out_proj.weight", {H, gval}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.conv1d.weight",
                                        {gconv, spec.linear_conv_kernel_dim}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.norm.weight", {gval}, 0.1f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.dt_bias", {vg}, 0.1f, seed++));
            tensors.push_back(make_bf16(p + "linear_attn.A_log", {vg}, 0.1f, seed++));
        } else {
            tensors.push_back(make_bf16(p + "self_attn.q_proj.weight", {2 * qf, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "self_attn.k_proj.weight", {kvf, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "self_attn.v_proj.weight", {kvf, H}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "self_attn.o_proj.weight", {H, qf}, 0.05f, seed++));
            tensors.push_back(make_bf16(p + "self_attn.q_norm.weight",
                                        {spec.attention_head_dim}, 0.1f, seed++));
            tensors.push_back(make_bf16(p + "self_attn.k_norm.weight",
                                        {spec.attention_head_dim}, 0.1f, seed++));
        }
    }

    const std::string path = model_dir + "/model.safetensors";
    auto created = io::SafetensorsWriter::create(path);
    if (!created.ok()) return created.status();
    io::SafetensorsWriter writer = created.release();
    for (const auto& t : tensors) {
        Status st = writer.plan_tensor(t.name, io::SType::BF16, t.shape);
        if (!st.ok()) return st;
    }
    auto header = writer.write_header();
    if (!header.ok()) return header.status();
    for (const auto& t : tensors) {
        Status st = writer.write_tensor(t.name, t.bytes.data(),
                                        t.bytes.size() * sizeof(std::uint16_t));
        if (!st.ok()) return st;
    }
    auto finished = writer.finish();
    if (!finished.ok()) return finished.status();
    return Status::make_ok();
}

}
