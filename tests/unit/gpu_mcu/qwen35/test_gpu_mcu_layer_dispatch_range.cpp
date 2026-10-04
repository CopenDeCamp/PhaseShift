#include <phaseshift/models/qwen35/runtime/mcu_layer_range.h>
#include <phaseshift/runtime/program/program.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

rt::WeightSlot make_weight_slot(uint32_t rows, uint32_t k) {
    rt::WeightSlot slot;
    slot.rows = rows;
    slot.cols = k;
    slot.k_padded = k;
    slot.encoding = 3u;
    slot.compute_spec = 3u;
    slot.codes = reinterpret_cast<const void*>(0x1000);
    slot.scales = reinterpret_cast<const void*>(0x2000);
    slot.storage_scale_stride_bytes = k;
    slot.weight_scale_group = 32u;
    slot.codes_row_stride_bytes = k;
    slot.preshuffled = true;
    return slot;
}

rt::PrimitiveGraph make_layer_graph(const std::vector<std::string>& names,
                                    uint32_t k, uint32_t out_features) {
    rt::PrimitiveGraph g;
    const rt::ValueId in = g.alloc_value(k, rt::ValueDType::BF16,
                                         rt::ValueRowDomain::TOKEN_ROWS);
    rt::ValueId current = in;
    for (size_t i = 0; i < names.size(); ++i) {
        const rt::ValueId next =
            g.alloc_value(out_features, rt::ValueDType::BF16,
                          rt::ValueRowDomain::TOKEN_ROWS);
        rt::LinearNode linear;
        linear.shape = rt::MatrixwiseShapeKey{out_features, k};
        linear.weight_index = static_cast<uint32_t>(i);
        linear.out_dtype = rt::ValueDType::BF16;
        auto& node = g.add_node(rt::PrimitiveNode{linear});
        node.inputs = {current};
        node.outputs = {next};
        node.debug_name = names[i];
        g.register_weight(linear.shape);
        current = next;
    }
    g.external_inputs = {in};
    g.external_outputs = {current};
    return g;
}

}  // namespace

int main() {
    constexpr uint32_t kK = 5120u;
    constexpr uint32_t kOut = 5120u;

    std::vector<rt::WeightSlot> slots(3);
    for (uint32_t i = 0; i < slots.size(); ++i) slots[i] = make_weight_slot(kOut, kK);
    rt::WeightTableView table;
    table.slots = slots.data();
    table.count = slots.size();

    rt::ProgramBuildOptions options;
    const rt::PrimitiveGraph two_nodes = make_layer_graph(
        {"L0.proj", "L1.proj"}, kK, kOut);
    auto two_prog = rt::build_program(two_nodes, table, rt::RowBucket::R16,
                                      rt::ExecutionClass::DECODE, options);
    check(two_prog.ok(), "two node graph builds");
    if (!two_prog.ok()) {
        std::printf("  build_program: %s\n",
                    two_prog.status().message().c_str());
        std::printf("test_gpu_mcu_layer_dispatch_range: failed\n");
        return 1;
    }
    const rt::Program& p2 = two_prog.value();
    check(p2.dispatches.size() == 4u, "each graph node lowers into two dispatches");
    check(p2.dispatch_source_node.size() == p2.dispatches.size(),
          "provenance size matches dispatch count");
    if (p2.dispatch_source_node.size() == 4u) {
        check(p2.dispatch_source_node[0] == 0u &&
                  p2.dispatch_source_node[1] == 0u,
              "both L0 dispatches share graph node 0");
        check(p2.dispatch_source_node[2] == 1u &&
                  p2.dispatch_source_node[3] == 1u,
              "both L1 dispatches share graph node 1");
        check(rt::kernel_to_name(p2.dispatches[0].kernel_id) ==
                  std::string("ACTIVATION_QUANTIZE_W4A8"),
              "first dispatch quantizes");
        check(rt::kernel_to_name(p2.dispatches[1].kernel_id) ==
                  std::string("LINEAR_PSQ4"),
              "second dispatch is the gemm");
    }

    uint32_t begin = 0;
    uint32_t end = 0;
    check(q35::find_dispatch_range_for_graph_prefix(two_nodes, p2, "L0.", &begin,
                                                    &end)
              .ok(),
          "L0 range resolves");
    check(begin == 0u && end == 2u, "L0 range is [0, 2)");
    check(q35::find_dispatch_range_for_graph_prefix(two_nodes, p2, "L1.", &begin,
                                                    &end)
              .ok(),
          "L1 range resolves");
    check(begin == 2u && end == 4u, "L1 range is [2, 4)");
    check(q35::find_dispatch_range_for_graph_prefix(two_nodes, p2, "L", &begin,
                                                    &end)
              .ok(),
          "prefix L resolves the whole set");
    check(begin == 0u && end == 4u, "prefix L range is [0, 4)");
    check(!q35::find_dispatch_range_for_graph_prefix(two_nodes, p2, "L2.", &begin,
                                                     &end)
               .ok(),
          "missing layer prefix fails closed");

    const rt::PrimitiveGraph split_nodes = make_layer_graph(
        {"L0.head", "L1.head", "L0.tail"}, kK, kOut);
    auto split_prog = rt::build_program(split_nodes, table, rt::RowBucket::R16,
                                        rt::ExecutionClass::DECODE, options);
    check(split_prog.ok(), "split graph builds");
    if (split_prog.ok()) {
        check(split_prog.value().dispatches.size() == 6u, "six dispatches");
        const ps::Status split_status = q35::find_dispatch_range_for_graph_prefix(
            split_nodes, split_prog.value(), "L0.", &begin, &end);
        check(!split_status.ok(), "scattered layer prefix fails closed");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_layer_dispatch_range: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_layer_dispatch_range: PASS\n");
    return 0;
}
