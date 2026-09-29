#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/runtime/graph/primitive_node.h>
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>

namespace ps::qwen35 {

namespace {

namespace RT = ::ps::runtime;
using MW = ::ps::weights::MatrixWeight;
using ::ps::quantization::imatrix::imatrix_tag;
using ::ps::quantization::imatrix::ImatrixSite;
using ::ps::quantization::imatrix::kImatrixLayerNone;

Status checked_u32(std::size_t value, const char* name, std::uint32_t& out) {
    if (value == 0) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return Status::out_of_range(name, __FILE__, __LINE__);
    }
    out = static_cast<std::uint32_t>(value);
    return Status::make_ok();
}

Status checked_product(
    std::size_t left,
    std::size_t right,
    const char* name,
    std::uint32_t& out) {

    if (left == 0 || right == 0) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (left > std::numeric_limits<std::uint32_t>::max() / right) {
        return Status::out_of_range(name, __FILE__, __LINE__);
    }
    out = static_cast<std::uint32_t>(left * right);
    return Status::make_ok();
}

Status checked_double(
    std::uint32_t value,
    const char* name,
    std::uint32_t& out) {

    if (value > std::numeric_limits<std::uint32_t>::max() / 2) {
        return Status::out_of_range(name, __FILE__, __LINE__);
    }
    out = value * 2;
    return Status::make_ok();
}

Status validate_matrix(
    const ps::weights::MatrixWeight& weight,
    RT::MatrixwiseShapeKey expected,
    const char* name) {

    if (weight.rows == 0 || weight.cols == 0) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    if (weight.rows != expected.output_features ||
        weight.cols != expected.input_features) {
        return Status::invalid_argument(name, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::size_t tensor_element_count(const gpu::Tensor& t) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) {
        n *= t.dim(i);
    }
    return n;
}

Status validate_mtp_matrix(
    const ps::weights::MatrixWeight& weight,
    std::uint32_t expected_rows,
    std::uint32_t expected_cols,
    const char* tensor_name) {

    if (weight.rows == expected_rows && weight.cols == expected_cols) {
        return Status::make_ok();
    }
    char message[512];
    std::snprintf(
        message, sizeof(message),
        "invalid MTP matrix %s: expected [%u,%u], got [%u,%u]",
        tensor_name,
        expected_rows, expected_cols,
        weight.rows, weight.cols);
    return Status::invalid_argument(message, __FILE__, __LINE__);
}

Status validate_mtp_vector(
    const gpu::Tensor& tensor,
    std::uint32_t expected_elements,
    const char* tensor_name) {

    const std::size_t got = tensor.ndim() == 0 ? 0 : tensor_element_count(tensor);
    if (tensor.ndim() != 0 && got == expected_elements) {
        return Status::make_ok();
    }
    char message[512];
    std::snprintf(
        message, sizeof(message),
        "invalid MTP vector %s: expected [%u], got [%zu]",
        tensor_name, expected_elements, got);
    return Status::invalid_argument(message, __FILE__, __LINE__);
}

Status validate_mtp_geometry(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights) {

    if (!weights.mtp.present) {
        return Status::unsupported("model has no MTP weights", __FILE__, __LINE__);
    }

    std::uint32_t hidden_size = 0;
    std::uint32_t fc_input_features = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t query_features = 0;
    std::uint32_t kv_features = 0;
    std::uint32_t q_projection_features = 0;
    std::uint32_t head_dim = 0;
    auto status = checked_u32(config.hidden_size, "invalid MTP hidden size", hidden_size);
    if (!status.ok()) return status;
    status = checked_double(
        hidden_size,
        "invalid MTP fc geometry",
        fc_input_features);
    if (!status.ok()) return status;
    status = checked_u32(config.intermediate_size, "invalid MTP intermediate size", intermediate_size);
    if (!status.ok()) return status;
    status = checked_product(
        config.num_attention_heads,
        config.attention_head_dim,
        "invalid MTP attention query geometry",
        query_features);
    if (!status.ok()) return status;
    status = checked_product(
        config.num_key_value_heads,
        config.attention_head_dim,
        "invalid MTP attention KV geometry",
        kv_features);
    if (!status.ok()) return status;
    status = checked_double(
        query_features,
        "invalid MTP attention gated query geometry",
        q_projection_features);
    if (!status.ok()) return status;
    status = checked_u32(config.attention_head_dim, "invalid MTP attention head dimension", head_dim);
    if (!status.ok()) return status;

    const Qwen35MtpWeights& m = weights.mtp;

    status = validate_mtp_vector(
        m.pre_fc_norm_embedding_weight, hidden_size, "mtp.pre_fc_norm_embedding.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        m.pre_fc_norm_hidden_weight, hidden_size, "mtp.pre_fc_norm_hidden.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        m.final_norm_weight, hidden_size, "mtp.norm.weight");
    if (!status.ok()) return status;

    status = validate_mtp_matrix(
        m.fc, hidden_size, fc_input_features, "mtp.fc.weight");
    if (!status.ok()) return status;

    const Qwen35LayerWeights& layer = m.layer;
    status = validate_mtp_vector(
        layer.input_layernorm_weight, hidden_size, "mtp.layers.0.input_layernorm.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        layer.post_attention_layernorm_weight, hidden_size, "mtp.layers.0.post_attention_layernorm.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.mlp_gate_proj, intermediate_size, hidden_size, "mtp.layers.0.mlp.gate_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.mlp_up_proj, intermediate_size, hidden_size, "mtp.layers.0.mlp.up_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.mlp_down_proj, hidden_size, intermediate_size, "mtp.layers.0.mlp.down_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_q_proj, q_projection_features, hidden_size, "mtp.layers.0.self_attn.q_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_k_proj, kv_features, hidden_size, "mtp.layers.0.self_attn.k_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_v_proj, kv_features, hidden_size, "mtp.layers.0.self_attn.v_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_matrix(
        layer.attn_o_proj, hidden_size, query_features, "mtp.layers.0.self_attn.o_proj.weight");
    if (!status.ok()) return status;
    status = validate_mtp_vector(
        layer.attn_q_norm_weight, head_dim, "mtp.layers.0.self_attn.q_norm.weight");
    if (!status.ok()) return status;
    return validate_mtp_vector(
        layer.attn_k_norm_weight, head_dim, "mtp.layers.0.self_attn.k_norm.weight");
}

Status validate_mlp_shapes(
    const Qwen35LayerWeights& weights,
    std::uint32_t hidden_size,
    std::uint32_t intermediate_size,
    std::uint32_t tensor_parallel) {

    if (tensor_parallel > 1 && intermediate_size % tensor_parallel != 0) {
        return Status::invalid_argument(
            "intermediate size is not divisible by the tensor parallel size",
            __FILE__, __LINE__);
    }
    const RT::MatrixwiseShapeKey gate{intermediate_size / tensor_parallel, hidden_size};
    const RT::MatrixwiseShapeKey down{hidden_size, intermediate_size / tensor_parallel};

    auto status = validate_matrix(weights.mlp_gate_proj, gate, "invalid MLP gate projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.mlp_up_proj, gate, "invalid MLP up projection");
    if (!status.ok()) return status;
    return validate_matrix(weights.mlp_down_proj, down, "invalid MLP down projection");
}

Status validate_full_attention_shapes(
    const Qwen35LayerWeights& weights,
    const Qwen35TextConfig& config,
    std::uint32_t hidden_size,
    std::uint32_t tensor_parallel) {

    std::uint32_t query_features = 0;
    std::uint32_t kv_features = 0;
    auto status = checked_product(
        config.num_attention_heads,
        config.attention_head_dim,
        "invalid full attention query geometry",
        query_features);
    if (!status.ok()) return status;
    status = checked_product(
        config.num_key_value_heads,
        config.attention_head_dim,
        "invalid full attention KV geometry",
        kv_features);
    if (!status.ok()) return status;
    if (tensor_parallel > 1) {
        if (config.num_attention_heads % tensor_parallel != 0 ||
            config.num_key_value_heads % tensor_parallel != 0) {
            return Status::invalid_argument(
                "attention head counts are not divisible by the tensor parallel size",
                __FILE__, __LINE__);
        }
        query_features /= tensor_parallel;
        kv_features /= tensor_parallel;
    }

    std::uint32_t q_projection_features = 0;
    status = checked_double(
        query_features,
        "invalid full attention gated query geometry",
        q_projection_features);
    if (!status.ok()) return status;

    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    status = checked_u32(config.num_attention_heads, "invalid full attention query heads", query_heads);
    if (!status.ok()) return status;
    status = checked_u32(config.num_key_value_heads, "invalid full attention KV heads", kv_heads);
    if (!status.ok()) return status;
    status = checked_u32(config.attention_head_dim, "invalid full attention head dimension", head_dim);
    if (!status.ok()) return status;

    if (!(config.partial_rotary_factor > 0.0f &&
          config.partial_rotary_factor <= 1.0f)) {
        return Status::invalid_argument(
            "invalid full attention rotary factor", __FILE__, __LINE__);
    }
    const auto rotary_dim = static_cast<std::size_t>(
        static_cast<double>(config.attention_head_dim) *
        static_cast<double>(config.partial_rotary_factor));
    std::uint32_t rotary_dim_u32 = 0;
    status = checked_u32(rotary_dim, "invalid full attention rotary geometry", rotary_dim_u32);
    if (!status.ok()) return status;

    const RT::MatrixwiseShapeKey q_proj{q_projection_features, hidden_size};
    const RT::MatrixwiseShapeKey k_proj{kv_features, hidden_size};
    const RT::MatrixwiseShapeKey v_proj{kv_features, hidden_size};
    const RT::MatrixwiseShapeKey o_proj{hidden_size, query_features};

    status = validate_matrix(weights.attn_q_proj, q_proj, "invalid full attention q projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.attn_k_proj, k_proj, "invalid full attention k projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.attn_v_proj, v_proj, "invalid full attention v projection");
    if (!status.ok()) return status;
    return validate_matrix(weights.attn_o_proj, o_proj, "invalid full attention o projection");
}

Status validate_gdn_shapes(
    const Qwen35LayerWeights& weights,
    const Qwen35TextConfig& config,
    std::uint32_t hidden_size,
    std::uint32_t tensor_parallel) {

    std::uint32_t qk_features = 0;
    std::uint32_t value_features = 0;
    auto status = checked_product(
        config.linear_num_key_heads,
        config.linear_key_head_dim,
        "invalid GDN key geometry",
        qk_features);
    if (!status.ok()) return status;
    status = checked_product(
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        "invalid GDN value geometry",
        value_features);
    if (!status.ok()) return status;
    if (tensor_parallel > 1) {
        if (config.linear_num_key_heads % tensor_parallel != 0 ||
            config.linear_num_value_heads % tensor_parallel != 0) {
            return Status::invalid_argument(
                "GDN head counts are not divisible by the tensor parallel size",
                __FILE__, __LINE__);
        }
        qk_features /= tensor_parallel;
        value_features /= tensor_parallel;
    }
    if (qk_features > (std::numeric_limits<std::uint32_t>::max() - value_features) / 2) {
        return Status::out_of_range("invalid GDN convolution geometry", __FILE__, __LINE__);
    }
    const std::uint32_t conv_features = 2 * qk_features + value_features;
    std::uint32_t value_heads = 0;
    std::uint32_t key_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0;
    status = checked_u32(config.linear_num_value_heads, "invalid GDN value heads", value_heads);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_num_key_heads, "invalid GDN key heads", key_heads);
    if (!status.ok()) return status;
    if (tensor_parallel > 1) {
        value_heads /= tensor_parallel;
        key_heads /= tensor_parallel;
    }
    status = checked_u32(config.linear_key_head_dim, "invalid GDN key head dimension", key_head_dim);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_value_head_dim, "invalid GDN value head dimension", value_head_dim);
    if (!status.ok()) return status;
    status = checked_u32(config.linear_conv_kernel_dim, "invalid GDN convolution kernel", conv_kernel);
    if (!status.ok()) return status;

    const RT::MatrixwiseShapeKey qkv{conv_features, hidden_size};
    const RT::MatrixwiseShapeKey z{value_features, hidden_size};
    const RT::MatrixwiseShapeKey b{value_heads, hidden_size};
    const RT::MatrixwiseShapeKey a{value_heads, hidden_size};
    const RT::MatrixwiseShapeKey out{hidden_size, value_features};

    status = validate_matrix(weights.attn_in_proj_qkv, qkv, "invalid GDN qkv projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.attn_in_proj_z, z, "invalid GDN z projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.attn_in_proj_b, b, "invalid GDN b projection");
    if (!status.ok()) return status;
    status = validate_matrix(weights.attn_in_proj_a, a, "invalid GDN a projection");
    if (!status.ok()) return status;
    return validate_matrix(weights.attn_out_proj, out, "invalid GDN output projection");
}

Status validate_geometry(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const ModelPartition& raw_partition,
    std::uint32_t tp_full_attention,
    std::uint32_t tp_linear_attention,
    std::uint32_t tp_mlp) {

    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    auto status = checked_u32(config.hidden_size, "invalid hidden size", hidden_size);
    if (!status.ok()) return status;
    status = checked_u32(config.intermediate_size, "invalid intermediate size", intermediate_size);
    if (!status.ok()) return status;
    if (config.num_hidden_layers == 0 ||
        weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument(
            "Qwen3.5 layer metadata count mismatch", __FILE__, __LINE__);
    }
    if (!config.layer_types.empty() &&
        config.layer_types.size() != config.num_hidden_layers) {
        return Status::invalid_argument(
            "Qwen3.5 layer type count mismatch", __FILE__, __LINE__);
    }

    const ModelPartition partition =
        resolve_model_partition(raw_partition, config.num_hidden_layers);
    if (partition.owns_embedding) {
        auto embed_status = validate_matrix(
            weights.embed_tokens,
            RT::MatrixwiseShapeKey{weights.embed_tokens.rows, hidden_size},
            "invalid embedding weight");
        if (!embed_status.ok()) return embed_status;
    }
    if (partition.owns_lm_head && !weights.lm_head_tied) {
        auto lm_status = validate_matrix(
            weights.lm_head, RT::MatrixwiseShapeKey{weights.lm_head.rows, hidden_size},
            "invalid lm head weight");
        if (!lm_status.ok()) return lm_status;
    }

    for (std::size_t layer_index = partition.layer_begin;
         layer_index < partition.layer_end;
         ++layer_index) {

        const auto& layer_weights = weights.layers[layer_index];
        bool is_gdn = layer_weights.is_gdn;
        if (!config.layer_types.empty()) {
            const int layer_type = config.layer_types[layer_index];
            if (layer_type != 0 && layer_type != 1) {
                return Status::unsupported(
                    "unknown Qwen3.5 layer type", __FILE__, __LINE__);
            }
            const bool config_is_gdn = layer_type == 0;
            if (config_is_gdn != is_gdn) {
                return Status::invalid_argument(
                    "Qwen3.5 layer type conflicts with weight metadata",
                    __FILE__, __LINE__);
            }
            is_gdn = config_is_gdn;
        }

        status = validate_mlp_shapes(layer_weights, hidden_size, intermediate_size, tp_mlp);
        if (!status.ok()) return status;

        if (is_gdn) {
            status = validate_gdn_shapes(layer_weights, config, hidden_size,
                                         tp_linear_attention);
        } else {
            status = validate_full_attention_shapes(layer_weights, config, hidden_size,
                                                    tp_full_attention);
        }
        if (!status.ok()) return status;
    }
    return Status::make_ok();
}

RT::WeightSlot make_weight_slot(const MW& w) {
    RT::WeightSlot s{};
    s.rows = w.rows;
    s.cols = w.cols;
    s.k_padded = w.k_padded;
    s.encoding = static_cast<uint8_t>(w.encoding);
    s.compute_spec = static_cast<uint8_t>(w.compute_spec);
    if (w.encoding == ps::weights::MatrixEncoding::Bf16) {
        s.device_ptr = w.bf16.data<void>();
        s.bytes = static_cast<uint64_t>(w.rows) * w.cols * sizeof(uint16_t);
        s.codes = nullptr;
        s.scales = nullptr;
    } else {
        s.device_ptr = nullptr;
        s.bytes = 0;
        s.codes = w.codes.ndim() != 0 ? w.codes.data<void>() : nullptr;
        s.scales = w.scales.ndim() != 0 ? w.scales.data<void>() : nullptr;
        s.storage_scale_stride_bytes = w.storage_scale_stride_bytes;
        s.compute_codes = w.compute_codes.ndim() != 0 ? w.compute_codes.data<void>() : nullptr;
        s.compute_scales_bf16 =
            w.compute_scales_bf16.ndim() != 0 ? w.compute_scales_bf16.data<void>() : nullptr;
        s.codes_row_stride_bytes = w.codes_row_stride_bytes;
        s.scale_row_stride_bytes = w.scale_row_stride_bytes;
        s.weight_scale_group = w.weight_scale_group;
        s.preshuffled = w.preshuffled;
    }
    return s;
}

struct LayerWeightIndices {
    uint32_t attn_q = 0;
    uint32_t attn_k = 0;
    uint32_t attn_v = 0;
    uint32_t attn_o = 0;
    uint32_t gdn_qkv = 0;
    uint32_t gdn_z = 0;
    uint32_t gdn_b = 0;
    uint32_t gdn_a = 0;
    uint32_t gdn_out = 0;
    uint32_t mlp_gate = 0;
    uint32_t mlp_up = 0;
    uint32_t mlp_down = 0;
};

struct Emitter {
    RT::PrimitiveGraph& pg;

    RT::ValueId value(uint32_t features,
                      RT::ValueDType dtype = RT::ValueDType::BF16,
                      RT::ValueRowDomain rows = RT::ValueRowDomain::TOKEN_ROWS) {
        return pg.alloc_value(features, dtype, rows);
    }

    void emit(const RT::PrimitiveNode& node, std::vector<RT::ValueId> ins,
              std::vector<RT::ValueId> outs,
              std::vector<RT::StateId> state_ins = {},
              std::vector<RT::StateId> state_outs = {},
              uint32_t imatrix_tag = 0,
              std::vector<RT::ValueId> keep_alive = {}) {
        auto& gn = pg.add_node(node);
        gn.inputs = std::move(ins);
        gn.outputs = std::move(outs);
        gn.state_inputs = std::move(state_ins);
        gn.state_outputs = std::move(state_outs);
        gn.keep_alive = std::move(keep_alive);
        gn.imatrix_tag = imatrix_tag;
    }

    RT::ValueId linear(const MW& w, uint32_t weight_index, RT::ValueId x,
                       RT::ValueDType out_dtype = RT::ValueDType::BF16,
                       uint32_t imatrix_tag = 0) {
        RT::LinearNode n;
        n.shape = RT::MatrixwiseShapeKey{w.rows, w.cols};
        n.weight_index = weight_index;
        n.out_dtype = out_dtype;
        RT::ValueId y = value(w.rows, out_dtype, pg.value_row_domain(x));
        emit(RT::PrimitiveNode{n}, {x}, {y}, {}, {}, imatrix_tag);
        return y;
    }

    RT::ValueId unary(const RT::PrimitiveNode& node, RT::ValueId x,
                      uint32_t out_features,
                      RT::ValueDType dtype = RT::ValueDType::BF16) {
        RT::ValueId y = value(out_features, dtype);
        emit(node, {x}, {y});
        return y;
    }
};

}

namespace {
void lower_stage_debug(const char* what) {
    const char* env = std::getenv("PHASESHIFT_EXECUTOR_DEBUG");
    if (env != nullptr && env[0] != '\0' && env[0] != '0') {
        int dev = -1;
        (void)hipGetDevice(&dev);
        std::fprintf(stderr, "LOWERDBG dev=%d %s\n", dev, what);
    }
}
}  // namespace

Result<Qwen35LoweredPrimitives> lower_qwen35_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const Qwen35LowerOptions& options) {
    if (options.tensor_parallel_size == 0)
        return Status::invalid_argument("tensor parallel size must be >= 1", __FILE__,
                                        __LINE__);
    if (options.tensor_parallel_rank >= options.tensor_parallel_size)
        return Status::out_of_range("tensor parallel rank out of range", __FILE__, __LINE__);
    const uint32_t tp_attn =
        options.tensor_parallel_size > 1 && options.tp_full_attention
            ? options.tensor_parallel_size
            : 1u;
    const uint32_t tp_gdn =
        options.tensor_parallel_size > 1 && options.tp_linear_attention
            ? options.tensor_parallel_size
            : 1u;
    const uint32_t tp_mlp =
        options.tensor_parallel_size > 1 && options.tp_mlp ? options.tensor_parallel_size
                                                           : 1u;
    auto geometry_status =
        validate_geometry(config, weights, options.partition, tp_attn, tp_gdn, tp_mlp);
    if (!geometry_status.ok()) return geometry_status;

    ModelPartition partition = options.partition;
    const std::uint32_t total_layers = config.num_hidden_layers;
    if (partition.layer_end == 0) partition.layer_end = total_layers;
    if (partition.layer_end > total_layers)
        return Status::invalid_argument("model partition exceeds the layer count", __FILE__,
                                        __LINE__);
    lower_stage_debug("partition_resolved");
        if (partition.layer_begin > partition.layer_end)
        return Status::invalid_argument("model partition layer range is inverted", __FILE__,
                                        __LINE__);
    if (partition.owns_embedding != (partition.layer_begin == 0))
        return Status::invalid_argument(
            "only the first layer stage may own the embedding", __FILE__, __LINE__);
    if (partition.owns_lm_head != (partition.layer_end == total_layers))
        return Status::invalid_argument(
            "only the last layer stage may own the lm head", __FILE__, __LINE__);

    for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
        const std::size_t layer = options.hidden_taps[t];
        if (layer >= weights.layers.size()) {
            return Status::invalid_argument(
                "hidden tap layer index out of range", __FILE__, __LINE__);
        }
        if (layer < partition.layer_begin || layer >= partition.layer_end) {
            return Status::invalid_argument(
                "hidden tap layer lies outside the model partition", __FILE__, __LINE__);
        }
        for (std::size_t u = 0; u < t; ++u) {
            if (static_cast<std::size_t>(options.hidden_taps[u]) == layer) {
                return Status::invalid_argument(
                    "duplicate hidden tap layer index", __FILE__, __LINE__);
            }
        }
    }

    Qwen35LoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;
    std::vector<RT::ValueId> hidden_tap_values(options.hidden_taps.size());
    std::vector<uint8_t> hidden_tap_found(options.hidden_taps.size(), 0);

    std::vector<const MW*> weight_pointers;
    auto add_weight = [&](const MW& w) -> uint32_t {
        weight_pointers.push_back(&w);
        lowered.weights.push_back(make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<uint32_t>(weight_pointers.size() - 1);
    };

    const bool needs_embed_weight = partition.owns_embedding ||
                                    (weights.lm_head_tied && partition.owns_lm_head);
    const uint32_t embed_index =
        needs_embed_weight ? add_weight(weights.embed_tokens) : 0u;
    std::vector<LayerWeightIndices> layer_weights(weights.layers.size());
    {
        size_t cursor = 1;
        for (size_t li = 0; li < weights.layers.size(); ++li) {
            const auto& lw = weights.layers[li];
            auto& idx = layer_weights[li];
            if (li < partition.layer_begin || li >= partition.layer_end) continue;
            if (lw.is_gdn) {
                idx.gdn_qkv = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_qkv);
                idx.gdn_z = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_z);
                idx.gdn_b = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_b);
                idx.gdn_a = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_in_proj_a);
                idx.gdn_out = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_out_proj);
            } else {
                idx.attn_q = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_q_proj);
                idx.attn_k = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_k_proj);
                idx.attn_v = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_v_proj);
                idx.attn_o = static_cast<uint32_t>(cursor++);
                (void)add_weight(lw.attn_o_proj);
            }
            idx.mlp_gate = static_cast<uint32_t>(cursor++);
            (void)add_weight(lw.mlp_gate_proj);
            idx.mlp_up = static_cast<uint32_t>(cursor++);
            (void)add_weight(lw.mlp_up_proj);
            idx.mlp_down = static_cast<uint32_t>(cursor++);
            (void)add_weight(lw.mlp_down_proj);
        }
    }
    const bool lm_tied = weights.lm_head_tied;
    const uint32_t lm_index =
        lm_tied ? embed_index
                : (partition.owns_lm_head ? add_weight(weights.lm_head) : 0u);

    uint32_t key_heads = static_cast<uint32_t>(config.linear_num_key_heads);
    uint32_t value_heads = static_cast<uint32_t>(config.linear_num_value_heads);
    uint32_t key_head_dim = static_cast<uint32_t>(config.linear_key_head_dim);
    uint32_t value_head_dim = static_cast<uint32_t>(config.linear_value_head_dim);
    uint32_t conv_kernel = static_cast<uint32_t>(config.linear_conv_kernel_dim);
    uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    uint32_t head_dim = static_cast<uint32_t>(config.attention_head_dim);
    float rope_theta = config.rope_theta;
    float eps = config.rms_norm_eps;
    float partial = config.partial_rotary_factor;
    if (rope_theta == 0) rope_theta = 10000000.0f;
    if (eps == 0) eps = 1e-6f;
    if (partial == 0) partial = 0.25f;
    uint32_t rotary_dim = static_cast<uint32_t>((double)head_dim * (double)partial);

    auto reg_param = [&](const gpu::Tensor& t) -> uint32_t {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        std::size_t n = 1;
        for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
        s.elements = static_cast<uint32_t>(n);
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        lowered.parameters.push_back(s);
        return static_cast<uint32_t>(lowered.parameters.size() - 1);
    };
    struct LayerParamIndices {
        uint32_t input_norm = UINT32_MAX;
        uint32_t post_norm = UINT32_MAX;
        uint32_t q_norm = UINT32_MAX;
        uint32_t k_norm = UINT32_MAX;
        uint32_t conv = UINT32_MAX;
        uint32_t gdn_norm = UINT32_MAX;
        uint32_t dt_bias = UINT32_MAX;
        uint32_t a_log = UINT32_MAX;
    };
    std::vector<LayerParamIndices> layer_params(weights.layers.size());
    {
        for (size_t li = 0; li < weights.layers.size(); ++li) {
            const auto& lw = weights.layers[li];
            auto& lp = layer_params[li];
            if (lw.input_layernorm_weight.ndim() > 0)
                lp.input_norm = reg_param(lw.input_layernorm_weight);
            if (lw.post_attention_layernorm_weight.ndim() > 0)
                lp.post_norm = reg_param(lw.post_attention_layernorm_weight);
            if (lw.is_gdn) {
                if (lw.attn_conv1d_weight.ndim() > 0) lp.conv = reg_param(lw.attn_conv1d_weight);
                if (lw.attn_norm_weight.ndim() > 0) lp.gdn_norm = reg_param(lw.attn_norm_weight);
                if (lw.attn_dt_bias.ndim() > 0) lp.dt_bias = reg_param(lw.attn_dt_bias);
                if (lw.attn_A_log.ndim() > 0) lp.a_log = reg_param(lw.attn_A_log);
            } else {
                if (lw.attn_q_norm_weight.ndim() > 0) lp.q_norm = reg_param(lw.attn_q_norm_weight);
                if (lw.attn_k_norm_weight.ndim() > 0) lp.k_norm = reg_param(lw.attn_k_norm_weight);
            }
        }
    }
    const uint32_t final_norm_param =
        weights.final_norm_weight.ndim() > 0 ? reg_param(weights.final_norm_weight) : UINT32_MAX;

    uint32_t next_gdn_state_index = 0;
    uint32_t next_attention_state_index = 0;

    Emitter em{pg};

    const uint32_t tensor_rank = options.tensor_parallel_rank;
    const uint32_t hidden_size = static_cast<uint32_t>(config.hidden_size);
    const RT::RowwiseShapeKey hidden_key{hidden_size};

    RT::ValueId tokens{0};
    RT::ValueId layer_input{0};
    if (partition.owns_embedding) {
        tokens = em.value(0, RT::ValueDType::I32);
        layer_input = em.value(hidden_size);
        RT::EmbeddingLookupNode n;
        n.vocab_size = weights.embed_tokens.rows;
        n.hidden_size = hidden_size;
        n.weight_shape =
            RT::MatrixwiseShapeKey{weights.embed_tokens.rows, weights.embed_tokens.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {layer_input});
        pg.nodes.back().debug_name = "embedding";
        pg.external_inputs.push_back(tokens);
    } else {
        layer_input = em.value(hidden_size);
        RT::CommRecvNode n;
        n.group = RT::CommGroup::Pipeline;
        n.peer = options.pipeline_peer_rank;
        n.dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {}, {layer_input});
        pg.nodes.back().debug_name = "pp_recv";
    }

    lower_stage_debug("graph_loop_begin");
    for (size_t li = partition.layer_begin; li < partition.layer_end; ++li) {
        const auto& lw = weights.layers[li];
        const auto& lw_idx = layer_weights[li];
        const std::string layer_prefix = "L" + std::to_string(li) + ".";
        const auto name_last = [&](const char* suffix) {
            pg.nodes.back().debug_name = layer_prefix + suffix;
        };

        const auto& lp = layer_params[li];
        RT::RmsNormNode input_rms;
        input_rms.shape = hidden_key;
        input_rms.eps = eps;
        input_rms.group_size = hidden_size;
        input_rms.parameter_index = lp.input_norm;
        input_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        RT::ValueId normed = em.unary(RT::PrimitiveNode{input_rms}, layer_input, hidden_size);
        name_last("input_norm");

        RT::ValueId block_out{0};
        if (lw.is_gdn) {
            const uint32_t local_key_heads = key_heads / tp_gdn;
            const uint32_t local_value_heads = value_heads / tp_gdn;
            const MW& w_qkv = *weight_pointers[lw_idx.gdn_qkv];
            const MW& w_z = *weight_pointers[lw_idx.gdn_z];
            const MW& w_b = *weight_pointers[lw_idx.gdn_b];
            const MW& w_a = *weight_pointers[lw_idx.gdn_a];
            const MW& w_out = *weight_pointers[lw_idx.gdn_out];

            uint32_t qk_features = local_key_heads * key_head_dim;
            uint32_t value_features = local_value_heads * value_head_dim;
            uint32_t conv_features = 2 * qk_features + value_features;
            RT::GdnShapeKey gdn_key{local_key_heads, local_value_heads, key_head_dim,
                                    value_head_dim, conv_kernel};

            RT::ValueId v_qkv = em.linear(w_qkv, lw_idx.gdn_qkv, normed, RT::ValueDType::BF16,
                                          imatrix_tag(static_cast<uint32_t>(li),
                                                      ImatrixSite::GdnProjectionInput));
            name_last("gdn_qkv");
            RT::ValueId v_z = em.linear(w_z, lw_idx.gdn_z, normed);
            name_last("gdn_z");
            RT::ValueId v_b = em.linear(w_b, lw_idx.gdn_b, normed);
            name_last("gdn_b");
            RT::ValueId v_a = em.linear(w_a, lw_idx.gdn_a, normed);
            name_last("gdn_a");

            RT::ValueId v_conv = em.value(conv_features, RT::ValueDType::F32);
            const uint32_t gdn_state_index = next_gdn_state_index++;
            RT::StateId conv_state =
                pg.alloc_state(RT::PrimitiveStateKind::GDN_CONV_STATE, gdn_state_index);
            RT::StatefulCausalConv1DNode cn;
            cn.gdn = gdn_key;
            cn.parameter_index = lp.conv;
            cn.state_index = gdn_state_index;
            em.emit(RT::PrimitiveNode{cn}, {v_qkv}, {v_conv},
                    {conv_state}, {conv_state});
            name_last("gdn_conv");

            RT::ValueId v_act = em.unary(
                RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{conv_features}}},
                v_conv, conv_features);
            name_last("gdn_act");

            RT::ValueId v_q_view = em.value(qk_features);
            RT::ValueId v_k_view = em.value(qk_features);
            RT::ValueId v_v_view = em.value(value_features);
            pg.views.push_back(RT::PrimitiveValueView{v_q_view, v_act, 0, qk_features});
            pg.views.push_back(
                RT::PrimitiveValueView{v_k_view, v_act, qk_features, qk_features});
            pg.views.push_back(
                RT::PrimitiveValueView{v_v_view, v_act, 2 * qk_features, value_features});

            RT::ValueId v_nq = em.unary(
                RT::PrimitiveNode{RT::L2NormalizeNode{
                    gdn_key, RT::RowwiseShapeKey{qk_features}, 1e-6f, key_head_dim}},
                v_q_view, qk_features, RT::ValueDType::F32);
            name_last("gdn_q_normalize");
            RT::ValueId v_nk = em.unary(
                RT::PrimitiveNode{RT::L2NormalizeNode{
                    gdn_key, RT::RowwiseShapeKey{qk_features}, 1e-6f, key_head_dim}},
                v_k_view, qk_features, RT::ValueDType::F32);
            name_last("gdn_k_normalize");
            float scale = key_head_dim ? 1.0f / sqrtf(static_cast<float>(key_head_dim))
                                       : 1.0f;
            RT::ValueId v_sq = em.unary(
                RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{qk_features}, scale}},
                v_nq, qk_features, RT::ValueDType::F32);
            name_last("gdn_q_scale");

            RT::ValueId v_rec = em.value(value_features, RT::ValueDType::F32);
            RT::StateId rec_state =
                pg.alloc_state(RT::PrimitiveStateKind::GDN_RECURRENCE_STATE, gdn_state_index);
            RT::GdnRecurrenceNode rn;
            rn.gdn = gdn_key;
            rn.dt_bias_parameter_index = lp.dt_bias;
            rn.a_log_parameter_index = lp.a_log;
            rn.state_index = gdn_state_index;
            em.emit(RT::PrimitiveNode{rn}, {v_sq, v_nk, v_v_view, v_a, v_b}, {v_rec},
                    {rec_state}, {rec_state}, 0, {v_qkv});
            name_last("gdn_recurrence");

            RT::RmsNormNode gdn_rms;
            gdn_rms.shape = RT::RowwiseShapeKey{value_features};
            gdn_rms.eps = eps;
            gdn_rms.group_size = value_head_dim;
            gdn_rms.parameter_index = lp.gdn_norm;
            gdn_rms.weight_mode = RT::RmsNormWeightMode::DIRECT;
            RT::ValueId v_rms = em.unary(RT::PrimitiveNode{gdn_rms}, v_rec, value_features,
                                         RT::ValueDType::F32);
            name_last("gdn_norm");
            RT::ValueId v_gated_z = em.unary(
                RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{value_features}}},
                v_z, value_features, RT::ValueDType::F32);
            name_last("gdn_z_silu");
            RT::ValueId v_gated{0};
            {
                RT::MulNode mn{RT::RowwiseShapeKey{value_features}};
                v_gated = em.value(value_features);
                em.emit(RT::PrimitiveNode{mn}, {v_rms, v_gated_z}, {v_gated});
                name_last("gdn_gated");
            }
            block_out = em.linear(w_out, lw_idx.gdn_out, v_gated, RT::ValueDType::BF16,
                                  imatrix_tag(static_cast<uint32_t>(li),
                                              ImatrixSite::GdnOutputInput));
            name_last("gdn_out");
            if (tp_gdn > 1) {
                RT::ValueId reduced = em.value(hidden_size);
                RT::CommAllReduceNode ar;
                ar.group = RT::CommGroup::Tensor;
                ar.dtype = RT::ValueDType::BF16;
                em.emit(RT::PrimitiveNode{ar}, {block_out}, {reduced});
                name_last("gdn_out_all_reduce");
                block_out = reduced;
            }
        } else {
            const uint32_t local_q_heads = q_heads / tp_attn;
            const uint32_t local_kv_heads = kv_heads / tp_attn;
            uint32_t q_features = local_q_heads * head_dim;
            uint32_t kv_features = local_kv_heads * head_dim;
            uint32_t q_proj_features = 2 * q_features;
            RT::AttentionShapeKey attn_key{local_q_heads, local_kv_heads, head_dim, rotary_dim,
                                           tp_attn > 1 ? tensor_rank * local_kv_heads : 0u};

            const MW& w_q = *weight_pointers[lw_idx.attn_q];
            const MW& w_k = *weight_pointers[lw_idx.attn_k];
            const MW& w_v = *weight_pointers[lw_idx.attn_v];
            const MW& w_o = *weight_pointers[lw_idx.attn_o];

            RT::ValueId v_qproj = em.linear(
                w_q, lw_idx.attn_q, normed, RT::ValueDType::BF16,
                imatrix_tag(static_cast<uint32_t>(li), ImatrixSite::FullAttnQkvInput));
            name_last("q_proj");
            RT::ValueId v_k = em.linear(w_k, lw_idx.attn_k, normed);
            name_last("k_proj");
            RT::ValueId v_v = em.linear(w_v, lw_idx.attn_v, normed);
            name_last("v_proj");
            RT::ValueId v_q = em.value(q_features);
            RT::ValueId v_gate = em.value(q_features);
            {
                RT::SplitNode sn;
                sn.input_shape = RT::RowwiseShapeKey{q_proj_features};
                sn.output_shape_a = RT::RowwiseShapeKey{q_features};
                sn.output_shape_b = RT::RowwiseShapeKey{q_features};
                sn.split_dim = 1;
                sn.layout = RT::SplitLayout::INTERLEAVED_HEADS;
                sn.head_dim = head_dim;
                em.emit(RT::PrimitiveNode{sn}, {v_qproj}, {v_q, v_gate});
                name_last("q_gate_split");
            }
            RT::RmsNormNode q_rms;
            q_rms.shape = RT::RowwiseShapeKey{q_features};
            q_rms.eps = eps;
            q_rms.group_size = head_dim;
            q_rms.parameter_index = lp.q_norm;
            q_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
            RT::ValueId v_qn = em.unary(RT::PrimitiveNode{q_rms}, v_q, q_features,
                                        RT::ValueDType::F32);
            name_last("q_norm");
            RT::ValueId v_qr = em.unary(
                RT::PrimitiveNode{RT::RoPENode{attn_key, rotary_dim, head_dim, rope_theta}},
                v_qn, q_features);
            name_last("q_rope");

            RT::RmsNormNode k_rms;
            k_rms.shape = RT::RowwiseShapeKey{kv_features};
            k_rms.eps = eps;
            k_rms.group_size = head_dim;
            k_rms.parameter_index = lp.k_norm;
            k_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
            RT::ValueId v_kn = em.unary(RT::PrimitiveNode{k_rms}, v_k, kv_features,
                                        RT::ValueDType::F32);
            name_last("k_norm");
            RT::ValueId v_kr = em.unary(
                RT::PrimitiveNode{RT::RoPENode{attn_key, rotary_dim, head_dim, rope_theta}},
                v_kn, kv_features);
            name_last("k_rope");

            const uint32_t attn_state_index = next_attention_state_index++;
            RT::StateId kv_state = pg.alloc_state(
                RT::PrimitiveStateKind::KV_CACHE, attn_state_index);
            RT::KvAppendNode kan{attn_key, attn_state_index};
            em.emit(RT::PrimitiveNode{kan}, {v_kr, v_v}, {}, {}, {kv_state});
            name_last("kv_append");

            RT::ValueId v_ctx{0};
            {
                RT::PagedAttentionNode pan{
                    attn_key, head_dim ? 1.0f / sqrtf(static_cast<float>(head_dim)) : 1.0f,
                    attn_state_index};
                v_ctx = em.value(q_features, RT::ValueDType::F32);
                auto& pgn = pg.add_node(RT::PrimitiveNode{pan});
                pgn.inputs = {v_qr};
                pgn.outputs = {v_ctx};
                pgn.state_inputs = {kv_state};
                name_last("attn_core");
            }

            RT::ValueId v_gs = em.unary(
                RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{q_features}}},
                v_gate, q_features, RT::ValueDType::F32);
            name_last("attn_gate");
            RT::ValueId v_att{0};
            {
                RT::MulNode mn{RT::RowwiseShapeKey{q_features}};
                v_att = em.value(q_features);
                em.emit(RT::PrimitiveNode{mn}, {v_ctx, v_gs}, {v_att});
                name_last("attn_gated");
            }
            block_out = em.linear(w_o, lw_idx.attn_o, v_att, RT::ValueDType::BF16,
                                  imatrix_tag(static_cast<uint32_t>(li),
                                              ImatrixSite::FullAttnOInput));
            name_last("o_proj");
            if (tp_attn > 1) {
                RT::ValueId reduced = em.value(hidden_size);
                RT::CommAllReduceNode ar;
                ar.group = RT::CommGroup::Tensor;
                ar.dtype = RT::ValueDType::BF16;
                em.emit(RT::PrimitiveNode{ar}, {block_out}, {reduced});
                name_last("o_proj_all_reduce");
                block_out = reduced;
            }
        }

        RT::ValueId attn_residual{0};
        {
            RT::ResidualAddNode ra{hidden_key};
            attn_residual = em.value(hidden_size);
            em.emit(RT::PrimitiveNode{ra}, {block_out, layer_input}, {attn_residual});
            name_last("residual_attn");
        }

        RT::RmsNormNode post_rms;
        post_rms.shape = hidden_key;
        post_rms.eps = eps;
        post_rms.group_size = hidden_size;
        post_rms.parameter_index = lp.post_norm;
        post_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        RT::ValueId post_norm = em.unary(RT::PrimitiveNode{post_rms}, attn_residual,
                                         hidden_size);
        name_last("post_norm");

        const MW& w_gate = *weight_pointers[lw_idx.mlp_gate];
        const MW& w_up = *weight_pointers[lw_idx.mlp_up];
        const MW& w_down = *weight_pointers[lw_idx.mlp_down];

        RT::ValueId v_mlp_gate = em.linear(
            w_gate, lw_idx.mlp_gate, post_norm, RT::ValueDType::BF16,
            imatrix_tag(static_cast<uint32_t>(li), ImatrixSite::MlpGateUpInput));
        name_last("mlp_gate");
        RT::ValueId v_mlp_up = em.linear(w_up, lw_idx.mlp_up, post_norm);
        name_last("mlp_up");
        const uint32_t local_intermediate = config.intermediate_size / tp_mlp;
        RT::ValueId v_swiglu = em.value(local_intermediate);
        {
            RT::SwiGluNode sn{RT::RowwiseShapeKey{local_intermediate}};
            em.emit(RT::PrimitiveNode{sn}, {v_mlp_gate, v_mlp_up}, {v_swiglu});
            name_last("mlp_swiglu");
        }
        RT::ValueId v_down = em.linear(w_down, lw_idx.mlp_down, v_swiglu, RT::ValueDType::BF16,
                                       imatrix_tag(static_cast<uint32_t>(li),
                                                   ImatrixSite::MlpDownInput));
        name_last("mlp_down");
        if (tp_mlp > 1) {
            RT::ValueId reduced = em.value(hidden_size);
            RT::CommAllReduceNode ar;
            ar.group = RT::CommGroup::Tensor;
            ar.dtype = RT::ValueDType::BF16;
            em.emit(RT::PrimitiveNode{ar}, {v_down}, {reduced});
            name_last("mlp_down_all_reduce");
            v_down = reduced;
        }

        RT::ValueId layer_output{0};
        {
            RT::ResidualAddNode ra{hidden_key};
            layer_output = em.value(hidden_size);
            em.emit(RT::PrimitiveNode{ra}, {v_down, attn_residual}, {layer_output});
            name_last("output");
        }
        layer_input = layer_output;
        for (std::size_t t = 0; t < options.hidden_taps.size(); ++t) {
            if (static_cast<std::size_t>(options.hidden_taps[t]) == li) {
                hidden_tap_values[t] = layer_output;
                hidden_tap_found[t] = 1;
            }
        }
    }

    lower_stage_debug("head_begin");
    if (partition.owns_lm_head) {
        const MW* lmw = lm_tied ? &weights.embed_tokens : &weights.lm_head;
        const uint32_t out_vocab = lmw->rows;

        RT::ValueId selected_hidden = em.value(
            hidden_size, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
        RT::OutputGatherNode og{RT::RowwiseShapeKey{hidden_size}};
        em.emit(RT::PrimitiveNode{og}, {layer_input}, {selected_hidden});
        pg.nodes.back().debug_name = "selected_hidden";

        RT::RmsNormNode head_rms;
        head_rms.shape = hidden_key;
        head_rms.eps = eps;
        head_rms.group_size = hidden_size;
        head_rms.parameter_index = final_norm_param;
        head_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        RT::ValueId normed_hidden = em.value(hidden_size);
        em.emit(RT::PrimitiveNode{head_rms}, {selected_hidden}, {normed_hidden}, {}, {},
                imatrix_tag(kImatrixLayerNone, ImatrixSite::LmHeadInput));
        pg.nodes.back().debug_name = "final_norm";

        RT::ValueId selected_logits = em.value(
            out_vocab, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
        RT::LinearNode ln;
        ln.shape = RT::MatrixwiseShapeKey{lmw->rows, lmw->cols};
        ln.weight_index = lm_index;
        ln.out_dtype = RT::ValueDType::F32;
        em.emit(RT::PrimitiveNode{ln}, {normed_hidden}, {selected_logits});
        pg.nodes.back().debug_name = "logits";

        RT::ValueId sampled = em.value(
            1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
        em.emit(RT::PrimitiveNode{RT::SamplingNode{out_vocab}}, {selected_logits}, {sampled});
        pg.nodes.back().debug_name = "sampled";

        pg.external_outputs.push_back(selected_hidden);
        pg.external_outputs.push_back(selected_logits);
        pg.external_outputs.push_back(sampled);
        pg.external_outputs.push_back(layer_input);
    } else {
        RT::CommSendNode n;
        n.group = RT::CommGroup::Pipeline;
        n.peer = options.pipeline_peer_rank;
        n.dtype = RT::ValueDType::BF16;
        em.emit(RT::PrimitiveNode{n}, {layer_input}, {});
        pg.nodes.back().debug_name = "pp_send";
    }

    for (std::size_t t = 0; t < hidden_tap_values.size(); ++t) {
        if (hidden_tap_found[t] == 0) {
            return Status::invalid_state(
                "hidden tap layer was not lowered", __FILE__, __LINE__);
        }
        pg.external_outputs.push_back(hidden_tap_values[t]);
    }

    auto st = RT::validate_primitive_graph(pg);
    if (!st.ok()) return st;

    return lowered;
}

Result<Qwen35LoweredPrimitives> lower_qwen35_mtp_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights) {

    auto mtp_geometry_status = validate_mtp_geometry(config, weights);
    if (!mtp_geometry_status.ok()) return mtp_geometry_status;

    const uint32_t hidden_size = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(config.attention_head_dim);
    const uint32_t intermediate = static_cast<uint32_t>(config.intermediate_size);
    const uint32_t q_features = q_heads * head_dim;
    const uint32_t kv_features = kv_heads * head_dim;
    const uint32_t q_proj_features = 2u * q_features;

    float eps = config.rms_norm_eps;
    float rope_theta = config.rope_theta;
    float partial = config.partial_rotary_factor;
    if (eps == 0) eps = 1e-6f;
    if (rope_theta == 0) rope_theta = 10000000.0f;
    if (partial == 0) partial = 0.25f;
    const uint32_t rotary_dim = static_cast<uint32_t>((double)head_dim * (double)partial);

    Qwen35LoweredPrimitives lowered;
    RT::PrimitiveGraph& pg = lowered.graph;

    auto add_weight = [&](const MW& w) -> uint32_t {
        lowered.weights.push_back(make_weight_slot(w));
        pg.weight_table.push_back(RT::MatrixwiseShapeKey{w.rows, w.cols});
        return static_cast<uint32_t>(lowered.weights.size() - 1u);
    };
    auto reg_param = [&](const gpu::Tensor& t) -> uint32_t {
        RT::StaticParameterSlot s{};
        s.device_ptr = t.data<void>();
        std::size_t n = 1;
        for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
        s.elements = static_cast<uint32_t>(n);
        s.dtype = t.element_size() == 2 ? RT::ValueDType::BF16 : RT::ValueDType::F32;
        lowered.parameters.push_back(s);
        return static_cast<uint32_t>(lowered.parameters.size() - 1u);
    };

    lower_stage_debug("weight_table_begin");
    const uint32_t embed_index = add_weight(weights.embed_tokens);
    const uint32_t lm_index =
        weights.lm_head_tied ? embed_index : add_weight(weights.lm_head);
    const uint32_t fc_index = add_weight(weights.mtp.fc);

    const auto& lw = weights.mtp.layer;
    const uint32_t q_i = add_weight(lw.attn_q_proj);
    const uint32_t k_i = add_weight(lw.attn_k_proj);
    const uint32_t v_i = add_weight(lw.attn_v_proj);
    const uint32_t o_i = add_weight(lw.attn_o_proj);
    const uint32_t gate_i = add_weight(lw.mlp_gate_proj);
    const uint32_t up_i = add_weight(lw.mlp_up_proj);
    const uint32_t down_i = add_weight(lw.mlp_down_proj);

    const uint32_t p_enorm = reg_param(weights.mtp.pre_fc_norm_embedding_weight);
    const uint32_t p_hnorm = reg_param(weights.mtp.pre_fc_norm_hidden_weight);
    const uint32_t p_inorm = reg_param(lw.input_layernorm_weight);
    const uint32_t p_postnorm = reg_param(lw.post_attention_layernorm_weight);
    const uint32_t p_qnorm = reg_param(lw.attn_q_norm_weight);
    const uint32_t p_knorm = reg_param(lw.attn_k_norm_weight);
    const uint32_t p_finalnorm = reg_param(weights.mtp.final_norm_weight);

    Emitter em{pg};
    const RT::RowwiseShapeKey hidden_key{hidden_size};
    const RT::RowwiseShapeKey concat_key{2u * hidden_size};

    RT::ValueId tokens = em.value(0, RT::ValueDType::I32);
    RT::ValueId hidden_in = em.value(hidden_size);
    pg.external_inputs.push_back(tokens);
    pg.external_inputs.push_back(hidden_in);

    RT::ValueId v_embed = em.value(hidden_size);
    {
        RT::EmbeddingLookupNode n;
        n.vocab_size = weights.embed_tokens.rows;
        n.hidden_size = hidden_size;
        n.weight_shape =
            RT::MatrixwiseShapeKey{weights.embed_tokens.rows, weights.embed_tokens.cols};
        n.weight_index = embed_index;
        em.emit(RT::PrimitiveNode{n}, {tokens}, {v_embed});
    }

    RT::RmsNormNode enorm;
    enorm.shape = hidden_key;
    enorm.eps = eps;
    enorm.group_size = hidden_size;
    enorm.parameter_index = p_enorm;
    enorm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_en = em.unary(RT::PrimitiveNode{enorm}, v_embed, hidden_size);

    RT::RmsNormNode hnorm;
    hnorm.shape = hidden_key;
    hnorm.eps = eps;
    hnorm.group_size = hidden_size;
    hnorm.parameter_index = p_hnorm;
    hnorm.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_hn = em.unary(RT::PrimitiveNode{hnorm}, hidden_in, hidden_size);

    RT::ValueId v_cat = em.value(concat_key.features);
    {
        RT::ConcatNode cn;
        cn.shape = concat_key;
        cn.input_shape_a = hidden_key;
        cn.input_shape_b = hidden_key;
        em.emit(RT::PrimitiveNode{cn}, {v_en, v_hn}, {v_cat});
    }

    RT::ValueId x = em.linear(weights.mtp.fc, fc_index, v_cat);

    RT::RmsNormNode input_rms;
    input_rms.shape = hidden_key;
    input_rms.eps = eps;
    input_rms.group_size = hidden_size;
    input_rms.parameter_index = p_inorm;
    input_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId normed = em.unary(RT::PrimitiveNode{input_rms}, x, hidden_size);

    RT::AttentionShapeKey attn_key{q_heads, kv_heads, head_dim, rotary_dim};
    RT::ValueId v_qproj = em.linear(lw.attn_q_proj, q_i, normed);
    RT::ValueId v_k = em.linear(lw.attn_k_proj, k_i, normed);
    RT::ValueId v_v = em.linear(lw.attn_v_proj, v_i, normed);

    RT::ValueId v_q = em.value(q_features);
    RT::ValueId v_gate = em.value(q_features);
    {
        RT::SplitNode sn;
        sn.input_shape = RT::RowwiseShapeKey{q_proj_features};
        sn.output_shape_a = RT::RowwiseShapeKey{q_features};
        sn.output_shape_b = RT::RowwiseShapeKey{q_features};
        sn.split_dim = 1;
        sn.layout = RT::SplitLayout::INTERLEAVED_HEADS;
        sn.head_dim = head_dim;
        em.emit(RT::PrimitiveNode{sn}, {v_qproj}, {v_q, v_gate});
    }

    RT::RmsNormNode q_rms;
    q_rms.shape = RT::RowwiseShapeKey{q_features};
    q_rms.eps = eps;
    q_rms.group_size = head_dim;
    q_rms.parameter_index = p_qnorm;
    q_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_qn = em.unary(RT::PrimitiveNode{q_rms}, v_q, q_features, RT::ValueDType::F32);
    RT::ValueId v_qr = em.unary(
        RT::PrimitiveNode{RT::RoPENode{attn_key, rotary_dim, head_dim, rope_theta}},
        v_qn, q_features);

    RT::RmsNormNode k_rms;
    k_rms.shape = RT::RowwiseShapeKey{kv_features};
    k_rms.eps = eps;
    k_rms.group_size = head_dim;
    k_rms.parameter_index = p_knorm;
    k_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId v_kn = em.unary(RT::PrimitiveNode{k_rms}, v_k, kv_features, RT::ValueDType::F32);
    RT::ValueId v_kr = em.unary(
        RT::PrimitiveNode{RT::RoPENode{attn_key, rotary_dim, head_dim, rope_theta}},
        v_kn, kv_features);

    const uint32_t mtp_attn_state_index = 0;
    RT::StateId kv_state =
        pg.alloc_state(RT::PrimitiveStateKind::KV_CACHE, mtp_attn_state_index);
    RT::KvAppendNode kan{attn_key, mtp_attn_state_index};
    em.emit(RT::PrimitiveNode{kan}, {v_kr, v_v}, {}, {}, {kv_state});

    RT::ValueId v_ctx = em.value(q_features, RT::ValueDType::F32);
    {
        RT::PagedAttentionNode pan{
            attn_key, head_dim ? 1.0f / sqrtf(static_cast<float>(head_dim)) : 1.0f,
            mtp_attn_state_index};
        auto& pgn = pg.add_node(RT::PrimitiveNode{pan});
        pgn.inputs = {v_qr};
        pgn.outputs = {v_ctx};
        pgn.state_inputs = {kv_state};
    }

    RT::ValueId v_gs = em.unary(
        RT::PrimitiveNode{RT::SigmoidNode{RT::RowwiseShapeKey{q_features}}},
        v_gate, q_features, RT::ValueDType::F32);
    RT::ValueId v_att = em.value(q_features);
    {
        RT::MulNode mn{RT::RowwiseShapeKey{q_features}};
        em.emit(RT::PrimitiveNode{mn}, {v_ctx, v_gs}, {v_att});
    }
    RT::ValueId block_out = em.linear(lw.attn_o_proj, o_i, v_att);

    RT::ValueId attn_residual = em.value(hidden_size);
    {
        RT::ResidualAddNode ra{hidden_key};
        em.emit(RT::PrimitiveNode{ra}, {block_out, x}, {attn_residual});
    }

    RT::RmsNormNode post_rms;
    post_rms.shape = hidden_key;
    post_rms.eps = eps;
    post_rms.group_size = hidden_size;
    post_rms.parameter_index = p_postnorm;
    post_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId post_norm = em.unary(RT::PrimitiveNode{post_rms}, attn_residual, hidden_size);

    RT::ValueId v_mlp_gate = em.linear(lw.mlp_gate_proj, gate_i, post_norm);
    RT::ValueId v_mlp_up = em.linear(lw.mlp_up_proj, up_i, post_norm);
    RT::ValueId v_swiglu = em.value(intermediate);
    {
        RT::SwiGluNode sn{RT::RowwiseShapeKey{intermediate}};
        em.emit(RT::PrimitiveNode{sn}, {v_mlp_gate, v_mlp_up}, {v_swiglu});
    }
    RT::ValueId v_down = em.linear(lw.mlp_down_proj, down_i, v_swiglu);

    RT::ValueId layer_output = em.value(hidden_size);
    {
        RT::ResidualAddNode ra{hidden_key};
        em.emit(RT::PrimitiveNode{ra}, {v_down, attn_residual}, {layer_output});
    }

    RT::RmsNormNode final_rms;
    final_rms.shape = hidden_key;
    final_rms.eps = eps;
    final_rms.group_size = hidden_size;
    final_rms.parameter_index = p_finalnorm;
    final_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;

    RT::ValueId selected_hidden = em.value(
        hidden_size, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
    RT::OutputGatherNode og{RT::RowwiseShapeKey{hidden_size}};
    em.emit(RT::PrimitiveNode{og}, {layer_output}, {selected_hidden});

    RT::ValueId normed_hidden =
        em.unary(RT::PrimitiveNode{final_rms}, selected_hidden, hidden_size);

    const MW* lmw = weights.lm_head_tied ? &weights.embed_tokens : &weights.lm_head;
    RT::ValueId selected_logits = em.value(
        lmw->rows, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
    {
        RT::LinearNode ln;
        ln.shape = RT::MatrixwiseShapeKey{lmw->rows, lmw->cols};
        ln.weight_index = lm_index;
        ln.out_dtype = RT::ValueDType::F32;
        em.emit(RT::PrimitiveNode{ln}, {normed_hidden}, {selected_logits});
    }

    RT::ValueId sampled = em.value(1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::SamplingNode{lmw->rows}}, {selected_logits}, {sampled});

    pg.external_outputs.push_back(selected_hidden);
    pg.external_outputs.push_back(selected_logits);
    pg.external_outputs.push_back(sampled);
    pg.external_outputs.push_back(normed_hidden);

    auto st = RT::validate_primitive_graph(pg);
    if (!st.ok()) return st;

    return lowered;
}

}
