#include <phaseshift/models/qwen35/model/tensor_parallel_context.h>
#include <phaseshift/runtime/tp/tp_execution.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using ps::Result;
using ps::Status;
using ps::qwen35::Qwen35TextConfig;
using ps::qwen35::Qwen35TensorParallelContext;
using ps::qwen35::make_qwen35_tensor_parallel_context;
using ps::runtime::LinearNode;
using ps::runtime::MatrixwiseShapeKey;
using ps::runtime::PrimitiveGraph;
using ps::runtime::PrimitiveGraphNode;
using ps::runtime::Program;
using ps::runtime::TpBarrierKind;
using ps::runtime::ValueBinding;
using ps::runtime::ValueDType;
using ps::runtime::ValueId;
using ps::runtime::ValueRowDomain;
using ps::runtime::ValueStorage;
using ps::runtime::build_tp_execution_schedule;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static Qwen35TextConfig base_config() {
    Qwen35TextConfig c;
    c.vocab_size = 128;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.num_hidden_layers = 4;
    c.full_attention_interval = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.attention_head_dim = 16;
    c.linear_num_key_heads = 4;
    c.linear_num_value_heads = 8;
    c.linear_key_head_dim = 8;
    c.linear_value_head_dim = 8;
    c.linear_conv_kernel_dim = 4;
    c.layer_types = {0, 1, 0, 1};
    return c;
}

static void test_context_valid() {
    Qwen35TextConfig c = base_config();

    auto single = make_qwen35_tensor_parallel_context(c, 1, 0);
    check(single.ok(), "tp_size=1 context builds");
    if (single.ok()) {
        const Qwen35TensorParallelContext& ctx = single.value();
        check(ctx.tp_size == 1 && ctx.tp_rank == 0, "tp=1 geometry");
        check(ctx.local_intermediate_size == 128, "tp=1 local intermediate");
        check(ctx.local_attention_heads == 4, "tp=1 local q heads");
        check(ctx.local_key_value_heads == 2, "tp=1 local kv heads");
        check(ctx.local_gdn_key_heads == 4, "tp=1 local gdn key heads");
        check(ctx.local_gdn_value_heads == 8, "tp=1 local gdn value heads");
    }

    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        auto ctx = make_qwen35_tensor_parallel_context(c, 2, rank);
        check(ctx.ok(), "tp=2 context builds");
        if (!ctx.ok()) continue;
        check(ctx.value().tp_size == 2 && ctx.value().tp_rank == rank, "tp=2 geometry");
        check(ctx.value().local_intermediate_size == 64, "tp=2 local intermediate");
        check(ctx.value().local_attention_heads == 2, "tp=2 local q heads");
        check(ctx.value().local_key_value_heads == 1, "tp=2 local kv heads");
        check(ctx.value().local_gdn_key_heads == 2, "tp=2 local gdn key heads");
        check(ctx.value().local_gdn_value_heads == 4, "tp=2 local gdn value heads");
    }
}

static void test_context_invalid() {
    Qwen35TextConfig c = base_config();
    check(!make_qwen35_tensor_parallel_context(c, 0, 0).ok(), "tp_size=0 rejected");
    check(!make_qwen35_tensor_parallel_context(c, 2, 2).ok(), "tp_rank>=tp_size rejected");

    Qwen35TextConfig odd_i = base_config();
    odd_i.intermediate_size = 131;
    check(!make_qwen35_tensor_parallel_context(odd_i, 2, 0).ok(),
          "indivisible intermediate rejected");

    Qwen35TextConfig odd_q = base_config();
    odd_q.num_attention_heads = 3;
    check(!make_qwen35_tensor_parallel_context(odd_q, 2, 0).ok(),
          "indivisible query heads rejected");

    Qwen35TextConfig odd_kv = base_config();
    odd_kv.num_key_value_heads = 3;
    check(!make_qwen35_tensor_parallel_context(odd_kv, 2, 0).ok(),
          "indivisible kv heads rejected");

    Qwen35TextConfig odd_gdn_k = base_config();
    odd_gdn_k.linear_num_key_heads = 3;
    check(!make_qwen35_tensor_parallel_context(odd_gdn_k, 2, 0).ok(),
          "indivisible gdn key heads rejected");

    Qwen35TextConfig odd_gdn_v = base_config();
    odd_gdn_v.linear_num_value_heads = 5;
    check(!make_qwen35_tensor_parallel_context(odd_gdn_v, 2, 0).ok(),
          "indivisible gdn value heads rejected");
}

namespace {

std::uint32_t add_node(PrimitiveGraph& graph, std::string name, bool combine,
                       std::uint32_t output_value) {
    PrimitiveGraphNode node;
    LinearNode linear;
    linear.shape = MatrixwiseShapeKey{64, 64};
    linear.weight_index = 0;
    node.node = linear;
    if (output_value != 0xFFFFFFFFu) {
        node.outputs.push_back(ValueId{output_value});
    }
    node.debug_name = std::move(name);
    node.tp_combine = combine ? 1u : 0u;
    graph.nodes.push_back(std::move(node));
    return static_cast<std::uint32_t>(graph.nodes.size() - 1);
}

void bind_value(Program& program, std::uint32_t value) {
    ValueBinding binding;
    binding.logical_value = ValueId{value};
    binding.storage = ValueStorage::WORKSPACE;
    binding.dtype = ValueDType::BF16;
    binding.row_domain = ValueRowDomain::TOKEN_ROWS;
    binding.slot = 0;
    binding.feature_count = 64;
    binding.row_stride = 64;
    binding.offset = 0;
    binding.bytes = 64u * 64u * 2u;
    program.values.push_back(binding);
}

}

static void test_schedule_build() {
    PrimitiveGraph graph;
    Program program;

    const std::uint32_t n0 = add_node(graph, "L0.input_norm", false, 10);
    const std::uint32_t n1 = add_node(graph, "L0.o_proj", true, 11);
    const std::uint32_t n2 = add_node(graph, "L0.mlp_gate", false, 12);
    const std::uint32_t n3 = add_node(graph, "L0.mlp_down", true, 13);
    (void)n0;
    (void)n2;

    program.dispatches.resize(5);
    program.dispatch_source_node = {n0, n0, n1, n2, n3};
    bind_value(program, 11);
    bind_value(program, 13);

    auto schedule = build_tp_execution_schedule(graph, program);
    check(schedule.ok(), "schedule builds");
    if (!schedule.ok()) return;
    const auto& segments = schedule.value().segments;
    check(schedule.value().active(), "schedule is active");
    check(segments.size() == 2, "two barrier segments");
    if (segments.size() != 2) return;

    check(segments[0].dispatch_begin == 0 && segments[0].dispatch_end == 3,
          "first segment covers dispatches [0,3)");
    check(segments[0].barrier_after == TpBarrierKind::SumHidden,
          "first segment barrier after o_proj");
    check(segments[0].combine_value.id == 11, "first segment combine value");

    check(segments[1].dispatch_begin == 3 && segments[1].dispatch_end == 5,
          "second segment covers dispatches [3,5)");
    check(segments[1].barrier_after == TpBarrierKind::SumHidden,
          "second segment barrier after mlp_down");
    check(segments[1].combine_value.id == 13, "second segment combine value");

    Program contiguous = program;
    contiguous.dispatch_source_node = {n0, n0, n1, n2};
    auto short_map = build_tp_execution_schedule(graph, contiguous);
    check(!short_map.ok(), "source map size mismatch rejected");

    PrimitiveGraph no_combine_graph;
    Program plain;
    const std::uint32_t plain_node = add_node(no_combine_graph, "L0.input_norm", false, 20);
    plain.dispatches.resize(2);
    plain.dispatch_source_node = {plain_node, plain_node};
    auto plain_schedule = build_tp_execution_schedule(no_combine_graph, plain);
    check(plain_schedule.ok(), "schedule without combine nodes builds");
    if (plain_schedule.ok()) {
        check(plain_schedule.value().segments.size() == 1,
              "no combine nodes yields a single segment");
        check(plain_schedule.value().segments[0].barrier_after == TpBarrierKind::None,
              "no barrier without combine nodes");
        check(plain_schedule.value().segments[0].dispatch_end == 2,
              "single segment covers all dispatches");
    }

    PrimitiveGraph bad_index;
    Program index_program;
    index_program.dispatches.resize(1);
    index_program.dispatch_source_node = {7};
    auto bad = build_tp_execution_schedule(bad_index, index_program);
    check(!bad.ok(), "out-of-range source node rejected");

    PrimitiveGraph empty_out;
    PrimitiveGraphNode node;
    LinearNode linear;
    linear.shape = MatrixwiseShapeKey{64, 64};
    node.node = linear;
    node.tp_combine = 1;
    empty_out.nodes.push_back(node);
    Program empty_program;
    empty_program.dispatches.resize(1);
    empty_program.dispatch_source_node = {0};
    auto no_output = build_tp_execution_schedule(empty_out, empty_program);
    check(!no_output.ok(), "combine node without outputs rejected");

    PrimitiveGraph unbound;
    const std::uint32_t unbound_node = add_node(unbound, "L0.o_proj", true, 99);
    Program unbound_program;
    unbound_program.dispatches.resize(1);
    unbound_program.dispatch_source_node = {unbound_node};
    auto unbound_result = build_tp_execution_schedule(unbound, unbound_program);
    check(!unbound_result.ok(), "combine value without binding rejected");
}

int main() {
    test_context_valid();
    test_context_invalid();
    test_schedule_build();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
