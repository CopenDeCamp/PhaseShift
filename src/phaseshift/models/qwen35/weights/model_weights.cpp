#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <phaseshift/weights/weight_loader.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace ps {
namespace qwen35 {
namespace {

using ps::weights::MatrixWeight;
using ps::weights::SafetensorsCollection;
using Tensor = gpu::Tensor;

std::string layer_prefix(std::size_t layer) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "model.language_model.layers.%zu.", layer);
    return buf;
}

const std::vector<std::string>& mtp_required_tensors() {
    static const std::vector<std::string> names = {
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    };
    return names;
}

Result<std::size_t> count_mtp_layers(const std::vector<std::string>& names) {
    std::vector<std::size_t> indices;
    const std::string prefix = "mtp.layers.";
    for (const auto& name : names) {
        if (name.rfind(prefix, 0) != 0) continue;
        std::size_t i = prefix.size();
        std::size_t j = i;
        while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) ++j;
        if (j == i || j >= name.size() || name[j] != '.') continue;
        indices.push_back(std::stoull(name.substr(i, j - i)));
    }
    if (indices.empty()) return std::size_t{0};
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i) {
            std::string msg = "unsupported MTP layer index layout:";
            for (auto v : indices) msg += " " + std::to_string(v);
            return Status::unsupported(msg.c_str(), __FILE__, __LINE__);
        }
    }
    return indices.size();
}

Status check_mtp_required_tensors(const std::vector<std::string>& names) {
    const std::set<std::string> have(names.begin(), names.end());
    std::string missing;
    for (const auto& name : mtp_required_tensors()) {
        if (have.find(name) == have.end()) {
            if (!missing.empty()) missing += ", ";
            missing += name;
        }
    }
    if (!missing.empty()) {
        return Status::invalid_argument(
            ("missing required MTP tensors: " + missing).c_str(), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status check_mtp_layer_count(const std::vector<std::string>& names) {
    auto layer_count = count_mtp_layers(names);
    if (!layer_count.ok()) return layer_count.status();
    if (layer_count.value() != 1) {
        return Status::unsupported(
            ("checkpoint contains " + std::to_string(layer_count.value()) +
             " MTP layers, runtime currently supports 1").c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status load_matrix(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ps::weights::WeightLoadOptions& weight_options,
    MatrixWeight& dst)
{
    auto m = ps::weights::load_bf16_matrix(collection, name, arena, stream, weight_options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_vector(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    Tensor& dst)
{
    auto t = ps::weights::load_bf16_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_norm_vector(
    const SafetensorsCollection& collection,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    Tensor& dst)
{
    auto t = ps::weights::load_bf16_or_f32_tensor(collection, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

Status load_q_matrix(
    const ps::quantization::fpx::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ps::weights::WeightLoadOptions& weight_options,
    MatrixWeight& dst)
{
    auto m = ps::weights::load_quantized_matrix(reader, name, arena, stream, weight_options);
    if (!m.ok()) return m.status();
    dst = m.release();
    return Status::make_ok();
}

Status load_q_vector(
    const ps::quantization::fpx::QuantizedModelReader& reader,
    const std::string& name,
    gpu::GpuArena& arena,
    hipStream_t stream,
    Tensor& dst)
{
    auto t = ps::weights::load_quantized_small(reader, name, arena, stream);
    if (!t.ok()) return t.status();
    dst = t.release();
    return Status::make_ok();
}

bool shard_applies_to_layer(const Qwen35LoadOptions& options, bool is_gdn) {
    return is_gdn ? options.tensor_shard.linear_attention
                  : options.tensor_shard.full_attention;
}

ps::weights::WeightLoadOptions layer_matrix_options(const Qwen35LoadOptions& options,
                                                    bool applies,
                                                    const ps::weights::MatrixShardSpec& axis) {
    ps::weights::WeightLoadOptions out = options.weights;
    out.shard = applies ? axis : ps::weights::MatrixShardSpec{};
    return out;
}

Status load_layer_bf16(
    gpu::GpuArena& arena, hipStream_t stream,
    const SafetensorsCollection& collection, std::size_t layer,
    const Qwen35LoadOptions& options,
    Qwen35LayerWeights& w)
{
    const std::string prefix = layer_prefix(layer);
    const bool is_gdn = !collection.tensor_spec(prefix + "self_attn.q_proj.weight").ok();
    w.is_gdn = is_gdn;
    const bool applies = shard_applies_to_layer(options, is_gdn);
    const auto column_opts = layer_matrix_options(options, applies, options.tensor_shard.column);
    const auto row_opts = layer_matrix_options(options, applies, options.tensor_shard.row);
    const bool mlp_applies = options.tensor_shard.mlp;
    const auto mlp_column_opts =
        layer_matrix_options(options, mlp_applies, options.tensor_shard.column);
    const auto mlp_row_opts =
        layer_matrix_options(options, mlp_applies, options.tensor_shard.row);

    Status st = load_norm_vector(collection, prefix + "input_layernorm.weight", arena, stream,
                                 w.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_norm_vector(collection, prefix + "post_attention_layernorm.weight", arena, stream,
                          w.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_matrix(collection, prefix + "mlp.gate_proj.weight", arena, stream, mlp_column_opts, w.mlp_gate_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, prefix + "mlp.up_proj.weight", arena, stream, mlp_column_opts, w.mlp_up_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, prefix + "mlp.down_proj.weight", arena, stream, mlp_row_opts, w.mlp_down_proj);
    if (!st.ok()) return st;

    if (is_gdn) {
        st = load_matrix(collection, prefix + "linear_attn.in_proj_a.weight", arena, stream, column_opts, w.attn_in_proj_a);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.in_proj_b.weight", arena, stream, column_opts, w.attn_in_proj_b);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.in_proj_qkv.weight", arena, stream, column_opts, w.attn_in_proj_qkv);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.in_proj_z.weight", arena, stream, column_opts, w.attn_in_proj_z);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.conv1d.weight", arena, stream, w.attn_conv1d_weight);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "linear_attn.out_proj.weight", arena, stream, row_opts, w.attn_out_proj);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.norm.weight", arena, stream, w.attn_norm_weight);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.dt_bias", arena, stream, w.attn_dt_bias);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "linear_attn.A_log", arena, stream, w.attn_A_log);
        if (!st.ok()) return st;
    } else {
        st = load_matrix(collection, prefix + "self_attn.q_proj.weight", arena, stream, column_opts, w.attn_q_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "self_attn.k_proj.weight", arena, stream, column_opts, w.attn_k_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "self_attn.v_proj.weight", arena, stream, column_opts, w.attn_v_proj);
        if (!st.ok()) return st;
        st = load_matrix(collection, prefix + "self_attn.o_proj.weight", arena, stream, row_opts, w.attn_o_proj);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "self_attn.q_norm.weight", arena, stream, w.attn_q_norm_weight);
        if (!st.ok()) return st;
        st = load_vector(collection, prefix + "self_attn.k_norm.weight", arena, stream, w.attn_k_norm_weight);
        if (!st.ok()) return st;
    }

    return Status::make_ok();
}

Status load_mtp_bf16(
    gpu::GpuArena& arena, hipStream_t stream,
    const SafetensorsCollection& collection,
    const Qwen35LoadOptions& options,
    Qwen35ModelWeights& weights)
{
    if (!collection.tensor_spec("mtp.fc.weight").ok()) {
        return Status::make_ok();
    }

    auto names = collection.tensor_names();
    if (!names.ok()) return names.status();
    Status mtp_contract = check_mtp_layer_count(names.value());
    if (!mtp_contract.ok()) return mtp_contract;
    mtp_contract = check_mtp_required_tensors(names.value());
    if (!mtp_contract.ok()) return mtp_contract;

    Qwen35MtpWeights& m = weights.mtp;
    m.present = true;
    m.layer.is_gdn = false;

    Status st = load_vector(collection, "mtp.pre_fc_norm_embedding.weight", arena, stream,
                            m.pre_fc_norm_embedding_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.pre_fc_norm_hidden.weight", arena, stream, m.pre_fc_norm_hidden_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.norm.weight", arena, stream, m.final_norm_weight);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.fc.weight", arena, stream, options.weights, m.fc);
    if (!st.ok()) return st;

    st = load_vector(collection, "mtp.layers.0.input_layernorm.weight", arena, stream,
                     m.layer.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.layers.0.post_attention_layernorm.weight", arena, stream,
                     m.layer.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.mlp.gate_proj.weight", arena, stream, options.weights,
                     m.layer.mlp_gate_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.mlp.up_proj.weight", arena, stream, options.weights,
                     m.layer.mlp_up_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.mlp.down_proj.weight", arena, stream, options.weights,
                     m.layer.mlp_down_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.q_proj.weight", arena, stream, options.weights,
                     m.layer.attn_q_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.k_proj.weight", arena, stream, options.weights,
                     m.layer.attn_k_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.v_proj.weight", arena, stream, options.weights,
                     m.layer.attn_v_proj);
    if (!st.ok()) return st;
    st = load_matrix(collection, "mtp.layers.0.self_attn.o_proj.weight", arena, stream, options.weights,
                     m.layer.attn_o_proj);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.layers.0.self_attn.q_norm.weight", arena, stream,
                     m.layer.attn_q_norm_weight);
    if (!st.ok()) return st;
    st = load_vector(collection, "mtp.layers.0.self_attn.k_norm.weight", arena, stream,
                     m.layer.attn_k_norm_weight);
    if (!st.ok()) return st;

    return Status::make_ok();
}

Result<Qwen35ModelWeights> load_qwen35_weights_bf16(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto collection_result = SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    SafetensorsCollection collection = collection_result.release();

    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    const Qwen35TextConfig config = config_result.release();

    Qwen35ModelWeights weights;
    weights.layers.resize(config.num_hidden_layers);

    ps::weights::WeightLoadOptions embed_load;
    embed_load.preshuffle = false;

    {
        auto m = ps::weights::load_bf16_matrix(
            collection, "model.language_model.embed_tokens.weight", arena, stream, embed_load);
        if (!m.ok()) return m.status();
        weights.embed_tokens = m.release();
    }

    if (config.tie_word_embeddings) {
        weights.lm_head = weights.embed_tokens;
        weights.lm_head_tied = true;
    } else {
        auto m = ps::weights::load_bf16_matrix(
            collection, "model.language_model.lm_head.weight", arena, stream, embed_load);
        if (!m.ok()) return m.status();
        weights.lm_head = m.release();
        weights.lm_head_tied = false;
    }

    {
        auto t = ps::weights::load_bf16_tensor(
            collection, "model.language_model.norm.weight", arena, stream);
        if (!t.ok()) return t.status();
        weights.final_norm_weight = t.release();
    }

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        Status st = load_layer_bf16(arena, stream, collection, l, options, weights.layers[l]);
        if (!st.ok()) return st;
    }

    Status mtp = load_mtp_bf16(arena, stream, collection, options, weights);
    if (!mtp.ok()) return mtp;

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after weight load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }

    return weights;
}

Status load_layer_quantized(
    gpu::GpuArena& arena, hipStream_t stream,
    const ps::quantization::fpx::QuantizedModelReader& reader, std::size_t layer,
    bool is_gdn,
    const ps::weights::WeightLoadOptions& weight_options,
    Qwen35LayerWeights& w)
{
    const std::string prefix = layer_prefix(layer);
    w.is_gdn = is_gdn;

    Status st = load_q_vector(reader, prefix + "input_layernorm.weight", arena, stream,
                              w.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, prefix + "post_attention_layernorm.weight", arena, stream,
                       w.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, prefix + "mlp.gate_proj.weight", arena, stream, weight_options, w.mlp_gate_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, prefix + "mlp.up_proj.weight", arena, stream, weight_options, w.mlp_up_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, prefix + "mlp.down_proj.weight", arena, stream, weight_options, w.mlp_down_proj);
    if (!st.ok()) return st;

    if (is_gdn) {
        st = load_q_matrix(reader, prefix + "linear_attn.in_proj_a.weight", arena, stream, weight_options, w.attn_in_proj_a);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "linear_attn.in_proj_b.weight", arena, stream, weight_options, w.attn_in_proj_b);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "linear_attn.in_proj_qkv.weight", arena, stream, weight_options, w.attn_in_proj_qkv);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "linear_attn.in_proj_z.weight", arena, stream, weight_options, w.attn_in_proj_z);
        if (!st.ok()) return st;
        st = load_q_vector(reader, prefix + "linear_attn.conv1d.weight", arena, stream, w.attn_conv1d_weight);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "linear_attn.out_proj.weight", arena, stream, weight_options, w.attn_out_proj);
        if (!st.ok()) return st;
        st = load_q_vector(reader, prefix + "linear_attn.norm.weight", arena, stream, w.attn_norm_weight);
        if (!st.ok()) return st;
        st = load_q_vector(reader, prefix + "linear_attn.dt_bias", arena, stream, w.attn_dt_bias);
        if (!st.ok()) return st;
        st = load_q_vector(reader, prefix + "linear_attn.A_log", arena, stream, w.attn_A_log);
        if (!st.ok()) return st;
    } else {
        st = load_q_matrix(reader, prefix + "self_attn.q_proj.weight", arena, stream, weight_options, w.attn_q_proj);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "self_attn.k_proj.weight", arena, stream, weight_options, w.attn_k_proj);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "self_attn.v_proj.weight", arena, stream, weight_options, w.attn_v_proj);
        if (!st.ok()) return st;
        st = load_q_matrix(reader, prefix + "self_attn.o_proj.weight", arena, stream, weight_options, w.attn_o_proj);
        if (!st.ok()) return st;
        st = load_q_vector(reader, prefix + "self_attn.q_norm.weight", arena, stream, w.attn_q_norm_weight);
        if (!st.ok()) return st;
        st = load_q_vector(reader, prefix + "self_attn.k_norm.weight", arena, stream, w.attn_k_norm_weight);
        if (!st.ok()) return st;
    }

    return Status::make_ok();
}

Status load_mtp_quantized(
    gpu::GpuArena& arena, hipStream_t stream,
    const ps::quantization::fpx::QuantizedModelReader& reader,
    const ps::weights::WeightLoadOptions& weight_options,
    Qwen35ModelWeights& weights)
{
    if (reader.manifest().tensors.count("mtp.fc.weight") == 0u) {
        return Status::make_ok();
    }

    auto names = reader.list_logical_tensors();
    if (!names.ok()) return names.status();
    Status mtp_contract = check_mtp_layer_count(names.value());
    if (!mtp_contract.ok()) return mtp_contract;
    mtp_contract = check_mtp_required_tensors(names.value());
    if (!mtp_contract.ok()) return mtp_contract;

    Qwen35MtpWeights& m = weights.mtp;
    m.present = true;
    m.layer.is_gdn = false;

    Status st = load_q_vector(reader, "mtp.pre_fc_norm_embedding.weight", arena, stream,
                              m.pre_fc_norm_embedding_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.pre_fc_norm_hidden.weight", arena, stream, m.pre_fc_norm_hidden_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.norm.weight", arena, stream, m.final_norm_weight);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.fc.weight", arena, stream, weight_options, m.fc);
    if (!st.ok()) return st;

    st = load_q_vector(reader, "mtp.layers.0.input_layernorm.weight", arena, stream,
                       m.layer.input_layernorm_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.layers.0.post_attention_layernorm.weight", arena, stream,
                       m.layer.post_attention_layernorm_weight);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.mlp.gate_proj.weight", arena, stream, weight_options, m.layer.mlp_gate_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.mlp.up_proj.weight", arena, stream, weight_options, m.layer.mlp_up_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.mlp.down_proj.weight", arena, stream, weight_options, m.layer.mlp_down_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.q_proj.weight", arena, stream, weight_options, m.layer.attn_q_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.k_proj.weight", arena, stream, weight_options, m.layer.attn_k_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.v_proj.weight", arena, stream, weight_options, m.layer.attn_v_proj);
    if (!st.ok()) return st;
    st = load_q_matrix(reader, "mtp.layers.0.self_attn.o_proj.weight", arena, stream, weight_options, m.layer.attn_o_proj);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.layers.0.self_attn.q_norm.weight", arena, stream,
                       m.layer.attn_q_norm_weight);
    if (!st.ok()) return st;
    st = load_q_vector(reader, "mtp.layers.0.self_attn.k_norm.weight", arena, stream,
                       m.layer.attn_k_norm_weight);
    if (!st.ok()) return st;

    return Status::make_ok();
}

}
Result<Qwen35ModelWeights> load_qwen35_weights_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options) {

    if (ps::weights::is_quantized_model_dir(model_dir)) {
        return load_qwen35_weights_from_quantized_safetensors(model_dir, arena, stream, options);
    }
    return load_qwen35_weights_bf16(model_dir, arena, stream, options);
}

Result<Qwen35ModelWeights> load_qwen35_weights_from_quantized_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto reader_result = ps::quantization::fpx::QuantizedModelReader::open(
        model_dir, options.verify_quantized_payload_crc);
    if (!reader_result.ok()) return reader_result.status();
    ps::quantization::fpx::QuantizedModelReader reader = reader_result.release();

    Status contract = ps::weights::validate_quantized_model(reader);
    if (!contract.ok()) {
        return contract;
    }

    const auto& manifest = reader.manifest();
    if (manifest.architecture != "qwen35_dense") {
        return Status::unsupported(
            ("unsupported quantized model architecture: " + manifest.architecture).c_str(),
            __FILE__, __LINE__);
    }

    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) return config_result.status();
    Qwen35TextConfig config = config_result.release();

    std::vector<int> layer_types = config.layer_types;
    if (layer_types.size() != config.num_hidden_layers) {
        layer_types.clear();
        for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
            const bool gdn = (config.full_attention_interval == 0)
                ? (l % 4 < 3)
                : (l % config.full_attention_interval < config.full_attention_interval - 1);
            layer_types.push_back(gdn ? 0 : 1);
        }
    }

    Qwen35ModelWeights weights;
    weights.layers.resize(config.num_hidden_layers);

    {
        auto m = ps::weights::load_quantized_matrix(reader, "model.language_model.embed_tokens.weight", arena, stream, options.weights);
        if (!m.ok()) return m.status();
        weights.embed_tokens = m.release();
    }

    if (config.tie_word_embeddings) {
        weights.lm_head = weights.embed_tokens;
        weights.lm_head_tied = true;
    } else {
        Result<MatrixWeight> m = ps::weights::load_quantized_matrix(reader, "model.language_model.lm_head.weight", arena, stream, options.weights);
        if (!m.ok()) {
            m = ps::weights::load_quantized_matrix(reader, "lm_head.weight", arena, stream, options.weights);
            if (!m.ok()) return m.status();
        }
        weights.lm_head = m.release();
        weights.lm_head_tied = false;
    }

    {
        auto t = ps::weights::load_quantized_small(reader, "model.language_model.norm.weight", arena, stream);
        if (!t.ok()) return t.status();
        weights.final_norm_weight = t.release();
    }

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        Status st = load_layer_quantized(arena, stream, reader, l,
                                         layer_types[l] == 0, options.weights, weights.layers[l]);
        if (!st.ok()) return st;
    }

    Status mtp = load_mtp_quantized(arena, stream, reader, options.weights, weights);
    if (!mtp.ok()) return mtp;

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error(
            "hipStreamSynchronize after quantized weight load",
            hipGetErrorString(sync_err), __FILE__, __LINE__);
    }

    return weights;
}
}
}
