#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/runtime/graph/primitive_node.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;
namespace q35 = ::ps::qwen35;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

void check_status(const ps::Status& status, const std::string& msg) {
    if (status.ok()) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s: %s\n", msg.c_str(), status.message().c_str());
    }
}

ps::Result<gpu::Tensor> alloc_bf16(gpu::GpuArena& arena,
                                   const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(std::uint16_t), 128);
    if (!view.ok()) return view.status();
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = 0; i + 1 < shape.size(); ++i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return gpu::Tensor::create<ps::bf16_t>(view.release(), shape, strides);
}

ps::weights::MatrixWeight make_matrix(gpu::Tensor t, std::uint32_t rows,
                                      std::uint32_t cols) {
    ps::weights::MatrixWeight w;
    w.bf16 = std::move(t);
    w.rows = rows;
    w.cols = cols;
    return w;
}

struct ModelFixture {
    q35::Qwen35TextConfig config;
    q35::Qwen35ModelWeights weights;
    bool ok = false;
};

constexpr std::uint32_t kV = 8;
constexpr std::uint32_t kH = 16;
constexpr std::uint32_t kI = 8;
constexpr std::uint32_t kLayers = 4;

ModelFixture make_fixture(gpu::GpuArena& arena) {
    ModelFixture fx;
    const auto alloc = [&](const std::vector<std::size_t>& shape) -> gpu::Tensor {
        auto r = alloc_bf16(arena, shape);
        if (!r.ok()) {
            std::printf("FAIL: alloc: %s\n", r.status().message().c_str());
            fx.ok = false;
            return gpu::Tensor{};
        }
        return r.release();
    };

    fx.ok = true;
    fx.config.vocab_size = kV;
    fx.config.hidden_size = kH;
    fx.config.intermediate_size = kI;
    fx.config.num_hidden_layers = kLayers;
    fx.config.linear_num_key_heads = 2;
    fx.config.linear_num_value_heads = 2;
    fx.config.linear_key_head_dim = 4;
    fx.config.linear_value_head_dim = 4;
    fx.config.linear_conv_kernel_dim = 4;
    fx.config.num_attention_heads = 2;
    fx.config.num_key_value_heads = 1;
    fx.config.attention_head_dim = 4;
    fx.config.rms_norm_eps = 1e-6f;
    fx.config.rope_theta = 1e7f;
    fx.config.partial_rotary_factor = 0.25f;
    fx.config.tie_word_embeddings = true;
    fx.config.layer_types = {0, 1, 0, 1};

    auto& w = fx.weights;
    w.lm_head_tied = true;
    w.embed_tokens = make_matrix(alloc({kV, kH}), kV, kH);
    w.final_norm_weight = alloc({kH});

    for (std::uint32_t li = 0; li < kLayers; ++li) {
        q35::Qwen35LayerWeights layer;
        layer.input_layernorm_weight = alloc({kH});
        layer.post_attention_layernorm_weight = alloc({kH});
        layer.mlp_gate_proj = make_matrix(alloc({kI, kH}), kI, kH);
        layer.mlp_up_proj = make_matrix(alloc({kI, kH}), kI, kH);
        layer.mlp_down_proj = make_matrix(alloc({kH, kI}), kH, kI);
        if (fx.config.layer_types[li] == 0) {
            layer.is_gdn = true;
            layer.attn_in_proj_qkv = make_matrix(alloc({24, kH}), 24, kH);
            layer.attn_in_proj_z = make_matrix(alloc({8, kH}), 8, kH);
            layer.attn_in_proj_b = make_matrix(alloc({2, kH}), 2, kH);
            layer.attn_in_proj_a = make_matrix(alloc({2, kH}), 2, kH);
            layer.attn_out_proj = make_matrix(alloc({kH, 8}), kH, 8);
            layer.attn_conv1d_weight = alloc({24, 4});
            layer.attn_norm_weight = alloc({8});
            layer.attn_dt_bias = alloc({2});
            layer.attn_A_log = alloc({2});
        } else {
            layer.is_gdn = false;
            layer.attn_q_proj = make_matrix(alloc({16, kH}), 16, kH);
            layer.attn_k_proj = make_matrix(alloc({4, kH}), 4, kH);
            layer.attn_v_proj = make_matrix(alloc({4, kH}), 4, kH);
            layer.attn_o_proj = make_matrix(alloc({kH, 8}), kH, 8);
            layer.attn_q_norm_weight = alloc({4});
            layer.attn_k_norm_weight = alloc({4});
        }
        w.layers.push_back(std::move(layer));
    }
    return fx;
}

std::size_t count_debug_prefix(const rt::PrimitiveGraph& g, const std::string& prefix) {
    std::size_t n = 0;
    for (const auto& node : g.nodes) {
        if (node.debug_name.rfind(prefix, 0) == 0) ++n;
    }
    return n;
}

bool has_debug_name(const rt::PrimitiveGraph& g, const std::string& name) {
    for (const auto& node : g.nodes) {
        if (node.debug_name == name) return true;
    }
    return false;
}

void check_partition(const q35::Qwen35TextConfig& config,
                     const q35::Qwen35ModelWeights& weights,
                     const q35::ModelPartition& partition, std::uint32_t peer,
                     const char* label) {
    q35::ModelPartition effective = partition;
    if (effective.layer_end == 0) effective.layer_end = kLayers;
    q35::Qwen35LowerOptions options;
    options.partition = partition;
    options.pipeline_peer_rank = peer;
    auto lowered = q35::lower_qwen35_to_primitives(config, weights, options);
    if (!lowered.ok()) {
        ++g_fail;
        std::printf("FAIL: %s lowering: %s\n", label, lowered.status().message().c_str());
        return;
    }
    const auto& g = lowered.value().graph;
    const std::string tag = label;

    if (partition.owns_embedding) {
        check(g.nodes.front().debug_name == "embedding", tag + ": first node is embedding");
        check(g.external_inputs.size() == 1, tag + ": one external input");
    } else {
        check(g.nodes.front().debug_name == "pp_recv", tag + ": first node is pp_recv");
        check(rt::primitive_kind_of(g.nodes.front().node) == rt::PrimitiveKind::COMM_RECV,
              tag + ": first node kind is COMM_RECV");
        check(g.external_inputs.empty(), tag + ": no external input");
    }

    if (partition.owns_lm_head) {
        check(g.nodes.back().debug_name == "sampled", tag + ": last node is sampled");
        check(g.external_outputs.size() == 4, tag + ": four external outputs");
    } else {
        check(g.nodes.back().debug_name == "pp_send", tag + ": last node is pp_send");
        check(rt::primitive_kind_of(g.nodes.back().node) == rt::PrimitiveKind::COMM_SEND,
              tag + ": last node kind is COMM_SEND");
        check(g.external_outputs.empty(), tag + ": no external output");
    }

    for (std::uint32_t li = 0; li < kLayers; ++li) {
        const std::string prefix = "L" + std::to_string(li) + ".";
        const bool inside =
            li >= effective.layer_begin && li < effective.layer_end;
        const std::size_t found = count_debug_prefix(g, prefix);
        if (inside) {
            check(found > 0, tag + ": lowered " + prefix);
        } else {
            check(found == 0, tag + ": omitted " + prefix);
        }
    }

    check(has_debug_name(g, "selected_hidden") == partition.owns_lm_head,
          tag + ": output gather only on the head stage");
    check(has_debug_name(g, "final_norm") == partition.owns_lm_head,
          tag + ": final norm only on the head stage");
}

void check_program(const q35::Qwen35TextConfig& config,
                   const q35::Qwen35ModelWeights& weights,
                   const q35::ModelPartition& partition, std::uint32_t peer,
                   rt::KernelId expected_edge_kernel, std::uint32_t expected_peer,
                   const char* label) {
    q35::Qwen35LowerOptions options;
    options.partition = partition;
    options.pipeline_peer_rank = peer;
    auto lowered = q35::lower_qwen35_to_primitives(config, weights, options);
    if (!lowered.ok()) {
        ++g_fail;
        std::printf("FAIL: %s lowering: %s\n", label, lowered.status().message().c_str());
        return;
    }
    auto built = lowered.release();
    rt::WeightTableView wview{built.weights.data(), built.weights.size()};
    rt::StaticParameterTableView pview{built.parameters.data(), built.parameters.size()};
    auto set = rt::build_program_set(built.graph, wview, pview,
                                     rt::ExecutionClass::DECODE, {});
    if (!set.ok()) {
        ++g_fail;
        std::printf("FAIL: %s program: %s\n", label, set.status().message().c_str());
        return;
    }
    const rt::Program& program = *set.value().resolve(rt::RowBucket::R16);
    check(program.dispatch_comm.size() == program.dispatches.size(),
          std::string(label) + ": dispatch communication metadata is dense");
    check(program.comms.size() == 1, std::string(label) + ": one communication descriptor");

    const bool send = expected_edge_kernel == rt::KernelId::COMM_SEND;
    const std::size_t edge = send ? program.dispatches.size() - 1 : 0;
    check(program.dispatches[edge].kernel_id == expected_edge_kernel,
          std::string(label) + ": stage edge dispatch kind");
    check(program.dispatch_comm[edge] != rt::kNoComm,
          std::string(label) + ": stage edge carries a descriptor");
    if (program.comms.empty()) return;
    const rt::CommDescriptor& d = program.comms[0];
    check(d.group == rt::CommGroup::Pipeline, std::string(label) + ": pipeline group");
    check(d.operation == (send ? rt::CommOperation::Send : rt::CommOperation::Recv),
          std::string(label) + ": operation matches the stage edge");
    check(d.peer == expected_peer, std::string(label) + ": peer rank recorded");
    const rt::ValueBinding* carried = program.find_value(send ? d.input : d.output);
    check(carried != nullptr, std::string(label) + ": carried value is bound");
    if (carried != nullptr)
        check(carried->storage == rt::ValueStorage::WORKSPACE,
              std::string(label) + ": carried value lives in the workspace");
    if (send) {
        check(program.dispatches[edge].input_slots[0] == d.input.id,
              std::string(label) + ": send input matches descriptor");
    } else {
        check(program.dispatches[edge].output_slots[0] == d.output.id,
              std::string(label) + ": receive output matches descriptor");
    }
}

void check_rejected(const q35::Qwen35TextConfig& config,
                    const q35::Qwen35ModelWeights& weights,
                    const q35::ModelPartition& partition, const char* label) {
    q35::Qwen35LowerOptions options;
    options.partition = partition;
    auto lowered = q35::lower_qwen35_to_primitives(config, weights, options);
    check(!lowered.ok(), std::string("rejected: ") + label);
    if (lowered.ok()) return;
    check(lowered.status().code() == ps::Status::Code::invalid_argument,
          std::string("rejected with invalid_argument: ") + label);
}

}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    auto ar = gpu::GpuArena::create(0, 16ull * 1024ull * 1024ull);
    if (!ar.ok()) {
        std::printf("SKIP: no GPU available (%s)\n", ar.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = ar.release();

    ModelFixture fx = make_fixture(arena);
    if (!fx.ok) {
        arena.shutdown();
        return 1;
    }

    check_partition(fx.config, fx.weights, q35::ModelPartition{}, 0, "full");
    check_partition(fx.config, fx.weights, q35::ModelPartition{0, 2, true, false}, 1,
                    "stage0");
    check_partition(fx.config, fx.weights, q35::ModelPartition{2, 4, false, true}, 0,
                    "stage1");

    check_program(fx.config, fx.weights, q35::ModelPartition{0, 2, true, false}, 1,
                  rt::KernelId::COMM_SEND, 1, "stage0");
    check_program(fx.config, fx.weights, q35::ModelPartition{2, 4, false, true}, 0,
                  rt::KernelId::COMM_RECV, 0, "stage1");

    check_rejected(fx.config, fx.weights, q35::ModelPartition{1, 3, true, true},
                   "embedding stage not starting at layer 0");
    check_rejected(fx.config, fx.weights, q35::ModelPartition{0, 3, true, true},
                   "lm head stage not ending at the last layer");
    check_rejected(fx.config, fx.weights, q35::ModelPartition{0, 4, true, false},
                   "sending stage ending at the last layer");
    check_rejected(fx.config, fx.weights, q35::ModelPartition{3, 2, false, true},
                   "inverted layer range");
    check_rejected(fx.config, fx.weights, q35::ModelPartition{0, 5, true, true},
                   "layer range beyond the model");

    {
        q35::Qwen35LowerOptions options;
        options.partition = q35::ModelPartition{2, 4, false, true};
        const std::uint32_t tap = 0;
        options.hidden_taps = std::span<const std::uint32_t>(&tap, 1);
        auto lowered = q35::lower_qwen35_to_primitives(fx.config, fx.weights, options);
        check(!lowered.ok(), "rejected: hidden tap outside the partition");
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) std::printf("QWEN35_PP_BOUNDARY: PASS\n");
    arena.shutdown();
    return g_fail == 0 ? 0 : 1;
}
