#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>

#include <phaseshift/models/qwen35/runtime/mcu_layer_range.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_value.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/kernels/optimized/verify_accept.h>
#include <phaseshift/models/qwen35/kernels/optimized/embedding.h>
#include <phaseshift/models/qwen35/kernels/optimized/output_gather.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.h>
#include <phaseshift/models/qwen35/kernels/optimized/sampling.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps::qwen35::runtime {

using ::ps::runtime::DispatchBinding;
using ::ps::runtime::KernelId;
using ::ps::runtime::Program;
namespace mcu = ::ps::runtime::gpu_mcu;

static_assert(sizeof(mcu::McuElementwiseInvocation) ==
              sizeof(::ps::kernel::ElementwiseArgs));
static_assert(alignof(mcu::McuElementwiseInvocation) ==
              alignof(::ps::kernel::ElementwiseArgs));
static_assert(offsetof(mcu::McuElementwiseInvocation, in0) ==
              offsetof(::ps::kernel::ElementwiseArgs, in0));
static_assert(offsetof(mcu::McuElementwiseInvocation, in0_row_stride) ==
              offsetof(::ps::kernel::ElementwiseArgs, in0_row_stride));
static_assert(offsetof(mcu::McuElementwiseInvocation, in1) ==
              offsetof(::ps::kernel::ElementwiseArgs, in1));
static_assert(offsetof(mcu::McuElementwiseInvocation, in1_row_stride) ==
              offsetof(::ps::kernel::ElementwiseArgs, in1_row_stride));
static_assert(offsetof(mcu::McuElementwiseInvocation, out0) ==
              offsetof(::ps::kernel::ElementwiseArgs, out0));
static_assert(offsetof(mcu::McuElementwiseInvocation, out0_row_stride) ==
              offsetof(::ps::kernel::ElementwiseArgs, out0_row_stride));
static_assert(offsetof(mcu::McuElementwiseInvocation, out1) ==
              offsetof(::ps::kernel::ElementwiseArgs, out1));
static_assert(offsetof(mcu::McuElementwiseInvocation, out1_row_stride) ==
              offsetof(::ps::kernel::ElementwiseArgs, out1_row_stride));
static_assert(offsetof(mcu::McuElementwiseInvocation, features) ==
              offsetof(::ps::kernel::ElementwiseArgs, features));
static_assert(offsetof(mcu::McuElementwiseInvocation, features_b) ==
              offsetof(::ps::kernel::ElementwiseArgs, features_b));
static_assert(offsetof(mcu::McuElementwiseInvocation, aux) ==
              offsetof(::ps::kernel::ElementwiseArgs, aux));
static_assert(offsetof(mcu::McuElementwiseInvocation, scalar_a) ==
              offsetof(::ps::kernel::ElementwiseArgs, scalar_a));
static_assert(offsetof(mcu::McuElementwiseInvocation, rows) ==
              offsetof(::ps::kernel::ElementwiseArgs, rows));
static_assert(sizeof(mcu::McuBf16ExactRowsInvocation) ==
              sizeof(::ps::kernel::Bf16GemmExactRowsArgs));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, weight) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, weight));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, input) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, input));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, output) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, output));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, output_dtype) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, output_dtype));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, n) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, n));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, k) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, k));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, input_row_stride) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, input_row_stride));
static_assert(offsetof(mcu::McuBf16ExactRowsInvocation, output_row_stride) ==
              offsetof(::ps::kernel::Bf16GemmExactRowsArgs, output_row_stride));
static_assert(sizeof(mcu::McuBf16WmmaInvocation) ==
              sizeof(::ps::kernel::Bf16GemmWmmaArgs));
static_assert(alignof(mcu::McuBf16WmmaInvocation) ==
              alignof(::ps::kernel::Bf16GemmWmmaArgs));
static_assert(offsetof(mcu::McuBf16WmmaInvocation, rows) ==
              offsetof(::ps::kernel::Bf16GemmWmmaArgs, rows));
static_assert(offsetof(mcu::McuBf16WmmaInvocation, out_features) ==
              offsetof(::ps::kernel::Bf16GemmWmmaArgs, out_features));
static_assert(offsetof(mcu::McuBf16WmmaInvocation, output_row_stride) ==
              offsetof(::ps::kernel::Bf16GemmWmmaArgs, output_row_stride));
static_assert(sizeof(mcu::McuL2NormalizeInvocation) ==
              sizeof(::ps::kernel::L2NormalizeArgs));
static_assert(offsetof(mcu::McuL2NormalizeInvocation, input) ==
              offsetof(::ps::kernel::L2NormalizeArgs, input));
static_assert(offsetof(mcu::McuL2NormalizeInvocation, output) ==
              offsetof(::ps::kernel::L2NormalizeArgs, output));
static_assert(offsetof(mcu::McuL2NormalizeInvocation, input_row_stride) ==
              offsetof(::ps::kernel::L2NormalizeArgs, input_row_stride));
static_assert(offsetof(mcu::McuL2NormalizeInvocation, output_row_stride) ==
              offsetof(::ps::kernel::L2NormalizeArgs, output_row_stride));
static_assert(offsetof(mcu::McuL2NormalizeInvocation, group_size) ==
              offsetof(::ps::kernel::L2NormalizeArgs, group_size));
static_assert(offsetof(mcu::McuL2NormalizeInvocation, eps) ==
              offsetof(::ps::kernel::L2NormalizeArgs, eps));
static_assert(sizeof(mcu::McuEmbeddingBf16Invocation) ==
              sizeof(::ps::kernel::EmbeddingBf16Args));
static_assert(alignof(mcu::McuEmbeddingBf16Invocation) ==
              alignof(::ps::kernel::EmbeddingBf16Args));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, table) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, table));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, token_ids) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, token_ids));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, output) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, output));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, error_word) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, error_word));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, rows) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, rows));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, vocab_size) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, vocab_size));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, hidden_size) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, hidden_size));
static_assert(offsetof(mcu::McuEmbeddingBf16Invocation, output_row_stride) ==
              offsetof(::ps::kernel::EmbeddingBf16Args, output_row_stride));
static_assert(sizeof(mcu::McuEmbeddingPsq8Invocation) ==
              sizeof(::ps::kernel::EmbeddingPsq8Args));
static_assert(alignof(mcu::McuEmbeddingPsq8Invocation) ==
              alignof(::ps::kernel::EmbeddingPsq8Args));
static_assert(offsetof(mcu::McuEmbeddingPsq8Invocation, codes) ==
              offsetof(::ps::kernel::EmbeddingPsq8Args, codes));
static_assert(offsetof(mcu::McuEmbeddingPsq8Invocation, error_word) ==
              offsetof(::ps::kernel::EmbeddingPsq8Args, error_word));
static_assert(offsetof(mcu::McuEmbeddingPsq8Invocation, rows) ==
              offsetof(::ps::kernel::EmbeddingPsq8Args, rows));
static_assert(offsetof(mcu::McuEmbeddingPsq8Invocation, output_row_stride) ==
              offsetof(::ps::kernel::EmbeddingPsq8Args, output_row_stride));
static_assert(sizeof(mcu::McuOutputGatherBf16Invocation) ==
              sizeof(::ps::kernel::OutputGatherBf16McuArgs));
static_assert(alignof(mcu::McuOutputGatherBf16Invocation) ==
              alignof(::ps::kernel::OutputGatherBf16McuArgs));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, input) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs, input));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, output) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs, output));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, output_rows) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs, output_rows));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, input_row_stride) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs,
                       input_row_stride));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, output_row_stride) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs,
                       output_row_stride));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, num_outputs) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs, num_outputs));
static_assert(offsetof(mcu::McuOutputGatherBf16Invocation, features) ==
              offsetof(::ps::kernel::OutputGatherBf16McuArgs, features));
static_assert(sizeof(mcu::McuVerifyAcceptPrefixInvocation) ==
              sizeof(::ps::kernel::VerifyAcceptPrefixArgs));
static_assert(alignof(mcu::McuVerifyAcceptPrefixInvocation) ==
              alignof(::ps::kernel::VerifyAcceptPrefixArgs));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation, candidates) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, candidates));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation, sampled_tokens) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, sampled_tokens));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation,
                       candidate_counts) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, candidate_counts));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation,
                       committed_counts) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, committed_counts));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation, request_count) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, request_count));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation,
                       candidate_stride) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, candidate_stride));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation,
                       max_candidates) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, max_candidates));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation, requests) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, requests));
static_assert(offsetof(mcu::McuVerifyAcceptPrefixInvocation, output_rows) ==
              offsetof(::ps::kernel::VerifyAcceptPrefixArgs, output_rows));
static_assert(sizeof(mcu::McuVerifyAcceptBatchInvocation) ==
              sizeof(::ps::kernel::VerifyAcceptBatchArgs));
static_assert(alignof(mcu::McuVerifyAcceptBatchInvocation) ==
              alignof(::ps::kernel::VerifyAcceptBatchArgs));
static_assert(offsetof(mcu::McuVerifyAcceptBatchInvocation, context) ==
              offsetof(::ps::kernel::VerifyAcceptBatchArgs, context));
static_assert(offsetof(mcu::McuVerifyAcceptBatchInvocation, sampled_tokens) ==
              offsetof(::ps::kernel::VerifyAcceptBatchArgs, sampled_tokens));
static_assert(offsetof(mcu::McuVerifyAcceptBatchInvocation, committed_counts) ==
              offsetof(::ps::kernel::VerifyAcceptBatchArgs, committed_counts));
static_assert(offsetof(mcu::McuVerifyAcceptBatchInvocation, max_candidates) ==
              offsetof(::ps::kernel::VerifyAcceptBatchArgs, max_candidates));
static_assert(sizeof(mcu::McuGdnSpecRestoreFromCountsInvocation) ==
              sizeof(::ps::kernel::GdnSpecRestoreFromCountsArgs));
static_assert(alignof(mcu::McuGdnSpecRestoreFromCountsInvocation) ==
              alignof(::ps::kernel::GdnSpecRestoreFromCountsArgs));
static_assert(offsetof(mcu::McuGdnSpecRestoreFromCountsInvocation, context) ==
              offsetof(::ps::kernel::GdnSpecRestoreFromCountsArgs, context));
static_assert(offsetof(mcu::McuGdnSpecRestoreFromCountsInvocation,
                       committed_counts) ==
              offsetof(::ps::kernel::GdnSpecRestoreFromCountsArgs,
                       committed_counts));
static_assert(offsetof(mcu::McuGdnSpecRestoreFromCountsInvocation,
                       conv_history) ==
              offsetof(::ps::kernel::GdnSpecRestoreFromCountsArgs,
                       conv_history));
static_assert(offsetof(mcu::McuGdnSpecRestoreFromCountsInvocation, conv_elems) ==
              offsetof(::ps::kernel::GdnSpecRestoreFromCountsArgs, conv_elems));
static_assert(offsetof(mcu::McuGdnSpecRestoreFromCountsInvocation,
                       recurrent_elems) ==
              offsetof(::ps::kernel::GdnSpecRestoreFromCountsArgs,
                       recurrent_elems));
static_assert(offsetof(mcu::McuGdnSpecRestoreFromCountsInvocation,
                       max_candidates) ==
              offsetof(::ps::kernel::GdnSpecRestoreFromCountsArgs,
                       max_candidates));
static_assert(sizeof(mcu::McuGdnSpecRestoreInvocation) ==
              sizeof(::ps::kernel::GdnSpecRestoreArgs));
static_assert(alignof(mcu::McuGdnSpecRestoreInvocation) ==
              alignof(::ps::kernel::GdnSpecRestoreArgs));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, conv_history) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, conv_history));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, conv_pool) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, conv_pool));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, recurrent_history) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, recurrent_history));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, recurrent_pool) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, recurrent_pool));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, conv_slot_stride) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, conv_slot_stride));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation,
                       recurrent_slot_stride) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs,
                       recurrent_slot_stride));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation,
                       conv_history_stride) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs,
                       conv_history_stride));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation,
                       recurrent_history_stride) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs,
                       recurrent_history_stride));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, sequence_slot) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, sequence_slot));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, history_row) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, history_row));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, conv_elems) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, conv_elems));
static_assert(offsetof(mcu::McuGdnSpecRestoreInvocation, recurrent_elems) ==
              offsetof(::ps::kernel::GdnSpecRestoreArgs, recurrent_elems));
static_assert(sizeof(mcu::McuArgmaxF32Invocation) ==
              sizeof(::ps::kernel::ArgmaxF32Args));
static_assert(alignof(mcu::McuArgmaxF32Invocation) ==
              alignof(::ps::kernel::ArgmaxF32Args));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, logits) ==
              offsetof(::ps::kernel::ArgmaxF32Args, logits));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, sampling) ==
              offsetof(::ps::kernel::ArgmaxF32Args, sampling));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, sampled_tokens) ==
              offsetof(::ps::kernel::ArgmaxF32Args, sampled_tokens));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, outputs) ==
              offsetof(::ps::kernel::ArgmaxF32Args, outputs));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, vocab_size) ==
              offsetof(::ps::kernel::ArgmaxF32Args, vocab_size));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, logits_row_stride) ==
              offsetof(::ps::kernel::ArgmaxF32Args, logits_row_stride));
static_assert(offsetof(mcu::McuArgmaxF32Invocation, output_rows) ==
              offsetof(::ps::kernel::ArgmaxF32Args, output_rows));
static_assert(sizeof(mcu::McuGdnConv1dInvocation) ==
              sizeof(::ps::kernel::GdnConv1dAqlArgs));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, input) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, input));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, output) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, output));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, conv_weight) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, conv_weight));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, conv_state) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, conv_state));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, requests) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, requests));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, actual_rows) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, actual_rows));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, conv_slot_stride) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, conv_slot_stride));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, row_tile) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, row_tile));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, conv_history_store) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, conv_history_store));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, capture_rows) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, capture_rows));
static_assert(offsetof(mcu::McuGdnConv1dInvocation, capture_verify_only) ==
              offsetof(::ps::kernel::GdnConv1dAqlArgs, capture_verify_only));
static_assert(offsetof(mcu::McuGdnRecurrenceInvocation, capture_verify_only) ==
              offsetof(::ps::kernel::GdnRecurrenceArgs, capture_verify_only));
static_assert(sizeof(mcu::McuGdnRecurrenceInvocation) ==
              sizeof(::ps::kernel::GdnRecurrenceArgs));
static_assert(alignof(mcu::McuGdnRecurrenceInvocation) ==
              alignof(::ps::kernel::GdnRecurrenceArgs));
static_assert(sizeof(mcu::McuPsq4MultiRowInvocation) ==
              sizeof(::ps::kernel::Psq4MultiRowBf16Args));
static_assert(alignof(mcu::McuPsq4MultiRowInvocation) ==
              alignof(::ps::kernel::Psq4MultiRowBf16Args));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, weight_codes) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, weight_codes));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, output) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, output));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, output_dtype) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, output_dtype));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, rows) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, rows));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, out_features) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, out_features));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, k_padded) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, k_padded));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, weight_scale_stride) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, weight_scale_stride));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, activation_code_stride) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, activation_code_stride));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, activation_scale_stride) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, activation_scale_stride));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, output_row_stride) ==
              offsetof(::ps::kernel::Psq4MultiRowBf16Args, output_row_stride));
static_assert(sizeof(mcu::McuPsq4MultiRowInvocation) ==
              sizeof(::ps::kernel::Psq8MultiRowBf16Args));
static_assert(alignof(mcu::McuPsq4MultiRowInvocation) ==
              alignof(::ps::kernel::Psq8MultiRowBf16Args));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, output_dtype) ==
              offsetof(::ps::kernel::Psq8MultiRowBf16Args, output_dtype));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, rows) ==
              offsetof(::ps::kernel::Psq8MultiRowBf16Args, rows));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, activation_code_stride) ==
              offsetof(::ps::kernel::Psq8MultiRowBf16Args, activation_code_stride));
static_assert(offsetof(mcu::McuPsq4MultiRowInvocation, output_row_stride) ==
              offsetof(::ps::kernel::Psq8MultiRowBf16Args, output_row_stride));

namespace {

// The gdn reset zeroes megabytes per sequence, so it is a dispatch of its own.
// The grid only has to give the grid stride loop enough lanes to keep the
// memory system busy; the kernel walks the whole slot from any grid.
constexpr uint32_t kGdnResetGrid = 128u;
constexpr uint32_t kGdnResetWorkgroup = 256u;

int find_variant(const McuCompiledPlan& plan, McuCompiledVariantKind kind,
                 uint32_t grid, uint32_t grid_y, uint32_t grid_z,
                 uint32_t workgroup) {
    for (size_t i = 0; i < plan.variants.size(); ++i) {
        const McuCompiledVariant& v = plan.variants[i];
        if (v.kind == kind && v.grid == grid && v.grid_y == grid_y &&
            v.grid_z == grid_z && v.workgroup == workgroup)
            return static_cast<int>(i);
    }
    return -1;
}

uint16_t mark_variant(McuCompiledPlan& plan, McuCompiledVariantKind kind,
                      uint32_t grid, uint32_t workgroup, uint32_t grid_y = 1u,
                      uint32_t grid_z = 1u) {
    const int found = find_variant(plan, kind, grid, grid_y, grid_z, workgroup);
    if (found >= 0) return static_cast<uint16_t>(found);
    auto v = mcu_plan_value<McuCompiledVariant>();
    v.kind = kind;
    v.grid = grid;
    v.grid_y = grid_y;
    v.grid_z = grid_z;
    v.workgroup = workgroup;
    plan.variants.push_back(v);
    return static_cast<uint16_t>(plan.variants.size() - 1u);
}

mcu::McuPlanNode make_dispatch_node(size_t current_size, uint16_t variant_id,
                                    uint16_t recipe, uint32_t invocation_index) {
    mcu::McuPlanNode node{};
    node.variant_id = variant_id;
    node.kernarg_recipe = recipe;
    node.completion_slot = 0;
    node.invocation_index = invocation_index;
    node.flags = mcu::kMcuNodeDispatch;
    node.next = static_cast<uint16_t>(current_size + 1u);
    return node;
}

bool name_starts_with(const char* name, const char* prefix) {
    if (name == nullptr || prefix == nullptr) return false;
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

}  // namespace

Status find_dispatch_range_for_graph_prefix(
    const ::ps::runtime::PrimitiveGraph& graph, const Program& program,
    const char* prefix, uint32_t* out_begin, uint32_t* out_end) {
    if (out_begin == nullptr || out_end == nullptr)
        return Status::invalid_argument("null output range", __FILE__, __LINE__);
    *out_begin = 0u;
    *out_end = 0u;
    if (program.dispatch_source_node.size() != program.dispatches.size())
        return Status::invalid_state("program has no dispatch provenance", __FILE__,
                                     __LINE__);
    uint32_t begin = 0u;
    uint32_t end = 0u;
    uint32_t matched = 0u;
    bool inside = false;
    const uint32_t total = static_cast<uint32_t>(program.dispatches.size());
    for (uint32_t i = 0; i < total; ++i) {
        const uint32_t source = program.dispatch_source_node[i];
        if (source >= graph.nodes.size())
            return Status::invalid_state("dispatch source node out of range",
                                         __FILE__, __LINE__);
        const bool matches =
            name_starts_with(graph.nodes[source].debug_name.c_str(), prefix);
        if (matches) {
            if (!inside) {
                if (matched != 0u)
                    return Status::invalid_state(
                        "layer dispatch range is not contiguous", __FILE__,
                        __LINE__);
                inside = true;
                begin = i;
            }
            matched += 1u;
            end = i + 1u;
        } else {
            inside = false;
        }
    }
    if (matched == 0u)
        return Status::invalid_argument("no dispatch matches the layer prefix",
                                        __FILE__, __LINE__);
    *out_begin = begin;
    *out_end = end;
    return Status::make_ok();
}

static Status compile_mcu_plan_impl(const Program& program,
                                    const McuStaticPlanCompileContext& sc,
                                    const McuPlanCompileOptions& options,
                                    McuCompiledPlan& out) {
    out = McuCompiledPlan{};
    if (options.dispatch_begin > options.dispatch_end ||
        options.dispatch_end > program.dispatches.size())
        return Status::invalid_argument("dispatch range out of bounds", __FILE__,
                                        __LINE__);
    if (options.dispatch_begin == options.dispatch_end)
        return Status::invalid_argument("empty dispatch range", __FILE__, __LINE__);

    const McuStaticResolveView resolve_view =
        make_static_resolve_view(sc, options.static_plan);
    const HostExecutionContext& ctx = resolve_view.ctx;

    const uint16_t marker_variant =
        mark_variant(out, McuCompiledVariantKind::CompletionMarker,
                     options.marker_grid, options.marker_workgroup);
    (void)marker_variant;

    uint32_t physical_nodes = 0u;
    const auto push_dispatch = [&out, &physical_nodes](mcu::McuPlanNode node) {
        out.nodes.push_back(node);
        ++physical_nodes;
    };

    const auto record_geometry = [&out](uint32_t node_index, McuGeometryAxis axis,
                                        uint8_t source, uint32_t param) {
        McuGeometryPatch patch;
        patch.node_index = node_index;
        patch.axis = axis;
        patch.source = source;
        patch.param = param;
        out.geometry_patches.push_back(patch);
    };

    const auto record_row_patch = [&out](McuInvocationTable table,
                                         uint32_t index, uint8_t source) {
        McuInvocationRowPatch patch;
        patch.table = table;
        patch.source = source;
        patch.index = index;
        out.invocation_row_patches.push_back(patch);
    };

    const auto row_source_for = [&](::ps::runtime::ValueId value,
                                    uint8_t& source) -> Status {
        auto resolved = resolve_host_value(program, value, ctx);
        if (!resolved.ok()) return resolved.status();
        source = static_cast<uint8_t>(
            resolved.value().row_domain ==
                    ::ps::runtime::ValueRowDomain::OUTPUT_ROWS
                ? mcu::McuInvocationPatchSource::NumOutputs
                : mcu::McuInvocationPatchSource::ActualRows);
        return Status::make_ok();
    };

    const auto scaled_row_source = [](uint8_t source) -> uint8_t {
        return source == static_cast<uint8_t>(
                             mcu::McuInvocationPatchSource::NumOutputs)
                   ? static_cast<uint8_t>(
                         mcu::McuInvocationPatchSource::NumOutputsTimesParam)
                   : static_cast<uint8_t>(
                         mcu::McuInvocationPatchSource::ActualRowsTimesParam);
    };

    const auto emit_embedding = [&](const DispatchBinding& b) -> Status {
        PhysicalEmbeddingLaunch launch{};
        const auto resolved = resolve_embedding_physical(program, b, ctx, launch);
        if (!resolved.ok()) return resolved.status();
        if (resolved.value() != PhysicalResolveStatus::Applicable)
            return Status::unsupported(
                "embedding dispatch is not applicable on the mcu", __FILE__,
                __LINE__);
        if (launch.storage == EmbeddingStorage::Psq8) {
            auto inv = mcu_plan_value<mcu::McuEmbeddingPsq8Invocation>();
            inv.codes = launch.codes;
            inv.scales = launch.scales;
            inv.token_ids = launch.token_ids;
            inv.output = launch.output;
            inv.error_word = reinterpret_cast<uint64_t>(sc.error_word);
            inv.codes_row_stride_bytes = launch.codes_row_stride_bytes;
            inv.scale_row_stride_bytes = launch.scale_row_stride_bytes;
            inv.rows = launch.rows;
            inv.vocab_size = launch.vocab_size;
            inv.hidden_size = launch.hidden_size;
            inv.output_row_stride = launch.output_row_stride;
            const uint32_t index =
                static_cast<uint32_t>(out.embedding_psq8.size());
            out.embedding_psq8.push_back(inv);
            record_row_patch(McuInvocationTable::EmbeddingPsq8, index,
                             static_cast<uint8_t>(
                                 mcu::McuInvocationPatchSource::ActualRows));
            const uint16_t variant_id = mark_variant(
                out, McuCompiledVariantKind::EmbeddingPsq8, launch.rows,
                ::ps::kernel::kEmbeddingThreads);
            const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
            push_dispatch(make_dispatch_node(
                out.nodes.size(), variant_id,
                mcu::kMcuKernargRecipeEmbeddingPsq8, index));
            record_geometry(node_index, McuGeometryAxis::X,
                            static_cast<uint8_t>(
                                mcu::McuInvocationPatchSource::ActualRows),
                            0u);
            return Status::make_ok();
        }
        if (launch.storage != EmbeddingStorage::Bf16)
            return Status::unsupported(
                "embedding storage has no mcu entrypoint", __FILE__, __LINE__);
        auto inv = mcu_plan_value<mcu::McuEmbeddingBf16Invocation>();
        inv.table = launch.table;
        inv.token_ids = launch.token_ids;
        inv.output = launch.output;
        inv.error_word = reinterpret_cast<uint64_t>(sc.error_word);
        inv.rows = launch.rows;
        inv.vocab_size = launch.vocab_size;
        inv.hidden_size = launch.hidden_size;
        inv.output_row_stride = launch.output_row_stride;
        const uint32_t index = static_cast<uint32_t>(out.embedding.size());
        out.embedding.push_back(inv);
        record_row_patch(McuInvocationTable::EmbeddingBf16, index,
                         static_cast<uint8_t>(
                             mcu::McuInvocationPatchSource::ActualRows));
        const uint16_t variant_id =
            mark_variant(out, McuCompiledVariantKind::EmbeddingBf16, launch.rows,
                         ::ps::kernel::kEmbeddingThreads);
        const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
        push_dispatch(make_dispatch_node(out.nodes.size(), variant_id,
                                         mcu::kMcuKernargRecipeEmbeddingBf16,
                                         index));
        record_geometry(node_index, McuGeometryAxis::X,
                        static_cast<uint8_t>(
                            mcu::McuInvocationPatchSource::ActualRows),
                        0u);
        return Status::make_ok();
    };

    const auto emit_output_gather = [&](const DispatchBinding& b) -> Status {
        if (b.input_count != 1u || b.output_count != 1u)
            return Status::unsupported("output gather io count is invalid",
                                       __FILE__, __LINE__);
        auto in_res = resolve_host_value(
            program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
        if (!in_res.ok()) return in_res.status();
        auto out_res = resolve_host_value(
            program, ::ps::runtime::ValueId{b.output_slots[0]}, ctx);
        if (!out_res.ok()) return out_res.status();
        const HostResolvedValue src = in_res.value();
        const HostResolvedValue dst = out_res.value();
        if (src.ptr == nullptr || dst.ptr == nullptr)
            return Status::unsupported("output gather value is invalid",
                                       __FILE__, __LINE__);
        if (src.dtype != ::ps::runtime::ValueDType::BF16)
            return Status::unsupported("output gather input is not BF16",
                                       __FILE__, __LINE__);
        if (dst.row_domain != ::ps::runtime::ValueRowDomain::OUTPUT_ROWS)
            return Status::unsupported("output gather output is not output rows",
                                       __FILE__, __LINE__);
        if (src.feature_count == 0u)
            return Status::unsupported("output gather feature count is zero",
                                       __FILE__, __LINE__);
        if (sc.device_output_rows == nullptr)
            return Status::unsupported("output gather requires output rows",
                                       __FILE__, __LINE__);
        const uint32_t num_outputs = sc.output_capacity;
        if (num_outputs == 0u)
            return Status::unsupported("output gather output count is zero",
                                       __FILE__, __LINE__);
        auto inv = mcu_plan_value<mcu::McuOutputGatherBf16Invocation>();
        inv.input = reinterpret_cast<uint64_t>(src.ptr);
        inv.output = reinterpret_cast<uint64_t>(dst.ptr);
        inv.output_rows =
            reinterpret_cast<uint64_t>(sc.device_output_rows);
        inv.input_row_stride = src.row_stride;
        inv.output_row_stride = dst.row_stride;
        inv.num_outputs = num_outputs;
        inv.features = src.feature_count;
        const uint32_t index = static_cast<uint32_t>(out.output_gather.size());
        out.output_gather.push_back(inv);
        const uint32_t blocks_x =
            (src.feature_count + ::ps::kernel::kOutputGatherMcuThreads - 1u) /
            ::ps::kernel::kOutputGatherMcuThreads;
        const uint16_t variant_id = mark_variant(
            out, McuCompiledVariantKind::OutputGatherBf16, blocks_x,
            ::ps::kernel::kOutputGatherMcuThreads, num_outputs);
        const uint32_t gather_node = static_cast<uint32_t>(out.nodes.size());
        push_dispatch(make_dispatch_node(
            out.nodes.size(), variant_id,
            mcu::kMcuKernargRecipeOutputGatherBf16, index));
        record_geometry(gather_node, McuGeometryAxis::Y,
                        static_cast<uint8_t>(
                            mcu::McuInvocationPatchSource::NumOutputs),
                        0u);
        return Status::make_ok();
    };

    uint32_t loop_begin = options.dispatch_begin;
    while (loop_begin < options.dispatch_end &&
           program.dispatches[loop_begin].kernel_id ==
               KernelId::EMBEDDING_LOOKUP) {
        const Status s = emit_embedding(program.dispatches[loop_begin]);
        if (!s.ok()) return s;
        ++loop_begin;
    }

    // The gdn state of a sequence that starts in this batch. A plan is static per
    // bucket and the device scheduler rewrites the batch context every loop, so
    // the node goes in whenever the range touches the gdn state, and the kernel
    // decides from the positions whether there is anything to reset: a row whose
    // position is zero is the first row of a new sequence. A decode only loop
    // therefore runs the node as a no-op, and a loop that starts a prefill clears
    // the state before the first layer reads it.
    bool range_touches_gdn_state = false;
    for (uint32_t i = loop_begin; i < options.dispatch_end; ++i) {
        const KernelId id = program.dispatches[i].kernel_id;
        if (id == KernelId::STATEFUL_CAUSAL_CONV1D ||
            id == KernelId::GDN_RECURRENCE) {
            range_touches_gdn_state = true;
            break;
        }
    }
    if (range_touches_gdn_state) {
        const ::ps::qwen35::GdnStatePoolDeviceView& gdn = sc.model_state.gdn;
        if (sc.batch_context != nullptr && gdn.conv_base != nullptr &&
            gdn.recurrent_base != nullptr && gdn.conv_slot_stride != 0u &&
            gdn.recurrent_slot_stride != 0u &&
            gdn.conv_slot_stride <= 0xFFFFFFFFull &&
            gdn.recurrent_slot_stride <= 0xFFFFFFFFull &&
            sc.device_requests != nullptr) {
            auto inv = mcu_plan_value<mcu::GpuMcuGdnResetInvocation>();
            inv.conv_base = reinterpret_cast<uint16_t*>(gdn.conv_base);
            inv.recurrent_base = reinterpret_cast<uint32_t*>(gdn.recurrent_base);
            inv.context = sc.batch_context;
            inv.conv_slot_stride = gdn.conv_slot_stride;
            inv.recurrent_slot_stride = gdn.recurrent_slot_stride;
            inv.conv_slot_count = static_cast<uint32_t>(gdn.conv_slot_stride);
            inv.recurrent_slot_count =
                static_cast<uint32_t>(gdn.recurrent_slot_stride);
            const uint32_t index = static_cast<uint32_t>(out.gdn_reset.size());
            out.gdn_reset.push_back(inv);
            const uint16_t variant_id =
                mark_variant(out, McuCompiledVariantKind::GdnResetZero,
                             kGdnResetGrid, kGdnResetWorkgroup);
            push_dispatch(make_dispatch_node(
                out.nodes.size(), variant_id,
                mcu::kMcuKernargRecipeGdnReset, index));
        }
    }

    for (uint32_t i = loop_begin; i < options.dispatch_end; ++i) {
        const DispatchBinding& b = program.dispatches[i];
        switch (b.kernel_id) {
            case KernelId::RMS_NORM: {
                PhysicalRmsNormLaunch launch{};
                const auto resolved =
                    resolve_rmsnorm_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "rmsnorm dispatch is not applicable on the mcu", __FILE__,
                        __LINE__);
                if (launch.variant != PhysicalRmsNormVariant::Bf16PfOnePlus &&
                    launch.variant != PhysicalRmsNormVariant::F32PgDirect &&
                    launch.variant != PhysicalRmsNormVariant::Bf16F32PgOnePlus)
                    return Status::unsupported(
                        "rmsnorm physical variant has no mcu entrypoint", __FILE__,
                        __LINE__);
                McuCompiledVariantKind rms_kind =
                    McuCompiledVariantKind::RmsNormF32PgDirect;
                if (launch.variant == PhysicalRmsNormVariant::Bf16PfOnePlus) {
                    rms_kind = McuCompiledVariantKind::RmsNormBf16PfOnePlus;
                } else if (launch.variant ==
                           PhysicalRmsNormVariant::Bf16F32PgOnePlus) {
                    rms_kind = McuCompiledVariantKind::RmsNormBf16F32PgOnePlus;
                }
                auto inv = mcu_plan_value<mcu::McuRmsNormInvocation>();
                inv.input = launch.input;
                inv.weight = launch.weight;
                inv.output = launch.output;
                inv.rows = launch.rows;
                inv.features = launch.features;
                inv.group_size = launch.group_size;
                inv.input_row_stride = launch.input_row_stride;
                inv.output_row_stride = launch.output_row_stride;
                inv.eps = launch.eps;
                const uint32_t index = static_cast<uint32_t>(out.rms.size());
                out.rms.push_back(inv);
                uint8_t rms_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.input_slots[0]}, rms_row_source);
                    if (!src_st.ok()) return src_st;
                }
                record_row_patch(McuInvocationTable::RmsNorm, index,
                                 rms_row_source);
                const uint16_t variant_id =
                    mark_variant(out, rms_kind, launch.grid, launch.workgroup);
                const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus, index));
                if (launch.group_size != 0u) {
                    record_geometry(node_index, McuGeometryAxis::X,
                                    scaled_row_source(rms_row_source),
                                    launch.features / launch.group_size);
                }
                break;
            }
            case KernelId::ACTIVATION_QUANTIZE_W4A8: {
                PhysicalActivationQuantizeLaunch launch{};
                const auto resolved =
                    resolve_activation_quantize_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "activation quantize dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                if (launch.kind != PhysicalActivationQuantizeKind::E4M3)
                    return Status::unsupported(
                        "activation quantize physical kernel has no mcu entrypoint",
                        __FILE__, __LINE__);
                McuCompiledVariantKind quant_kind =
                    McuCompiledVariantKind::ActivationQuantizeE4m3K5120;
                if (launch.k != launch.k_padded)
                    return Status::unsupported(
                        "activation quantize physical variant has no mcu entrypoint",
                        __FILE__, __LINE__);
                switch (launch.k) {
                    case 5120u:
                        quant_kind =
                            McuCompiledVariantKind::ActivationQuantizeE4m3K5120;
                        break;
                    case 6144u:
                        quant_kind =
                            McuCompiledVariantKind::ActivationQuantizeE4m3K6144;
                        break;
                    case 17408u:
                        quant_kind =
                            McuCompiledVariantKind::ActivationQuantizeE4m3K17408;
                        break;
                    default:
                        return Status::unsupported(
                            "activation quantize physical variant has no mcu entrypoint",
                            __FILE__, __LINE__);
                }
                auto inv = mcu_plan_value<mcu::McuActivationQuantizeE4m3Invocation>();
                inv.input = launch.input;
                inv.codes = launch.codes;
                inv.scales = launch.scales;
                inv.input_row_stride = launch.input_row_stride;
                inv.code_row_stride = launch.code_row_stride;
                inv.scale_row_stride = launch.scale_row_stride;
                const uint32_t index =
                    static_cast<uint32_t>(out.quant_e4m3.size());
                out.quant_e4m3.push_back(inv);
                uint8_t quant_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.input_slots[0]},
                        quant_row_source);
                    if (!src_st.ok()) return src_st;
                }
                const uint16_t variant_id = mark_variant(
                    out, quant_kind, launch.grid, launch.workgroup);
                const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120, index));
                record_geometry(node_index, McuGeometryAxis::X, quant_row_source,
                                0u);
                break;
            }
            case KernelId::LINEAR_PSQ4: {
                PhysicalPsq4Launch launch{};
                const auto resolved = resolve_psq4_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "psq4 dispatch is not applicable on the mcu", __FILE__,
                        __LINE__);
                if (launch.variant == PhysicalPsq4Variant::Decode1Bf16) {
                    if (launch.unroll != 16u && launch.unroll != 8u)
                        return Status::unsupported(
                            "psq4 physical variant has no mcu entrypoint", __FILE__,
                            __LINE__);
                    const bool psq4_u16 = launch.unroll == 16u;
                    auto inv = mcu_plan_value<mcu::McuPsq4Decode1Invocation>();
                    inv.weight_codes = launch.weight_codes;
                    inv.weight_scales = launch.weight_scales;
                    inv.activation_codes = launch.activation_codes;
                    inv.activation_scales = launch.activation_scales;
                    inv.output = launch.output;
                    inv.k_padded = launch.k_padded;
                    inv.weight_scale_stride = launch.weight_scale_stride;
                    const uint32_t index = static_cast<uint32_t>(out.psq4.size());
                    out.psq4.push_back(inv);
                    const uint16_t variant_id = mark_variant(
                        out,
                        psq4_u16 ? McuCompiledVariantKind::Psq4Decode1Bf16U16
                                 : McuCompiledVariantKind::Psq4Decode1Bf16U8,
                        launch.grid, launch.workgroup);
                    push_dispatch(make_dispatch_node(
                        out.nodes.size(), variant_id,
                        psq4_u16 ? mcu::kMcuKernargRecipePsq4Decode1Bf16U16
                                 : mcu::kMcuKernargRecipePsq4Decode1Bf16U8,
                        index));
                    break;
                }
                McuCompiledVariantKind multi_kind{};
                switch (launch.variant) {
                    case PhysicalPsq4Variant::RowBlock1Bf16:
                        multi_kind = McuCompiledVariantKind::Psq4RowBlock1Bf16;
                        break;
                    case PhysicalPsq4Variant::RowBlock2Bf16:
                        multi_kind = McuCompiledVariantKind::Psq4RowBlock2Bf16;
                        break;
                    case PhysicalPsq4Variant::RowBlock4Bf16:
                        multi_kind = McuCompiledVariantKind::Psq4RowBlock4Bf16;
                        break;
                    case PhysicalPsq4Variant::RowBlock8Bf16:
                        multi_kind = McuCompiledVariantKind::Psq4RowBlock8Bf16;
                        break;
                    case PhysicalPsq4Variant::Prefill2DK64N64:
                        multi_kind = McuCompiledVariantKind::Psq4Prefill2DK64N64;
                        break;
                    case PhysicalPsq4Variant::Prefill2DK128N64:
                        multi_kind =
                            McuCompiledVariantKind::Psq4Prefill2DK128N64;
                        break;
                    case PhysicalPsq4Variant::Prefill2DK64N128:
                        multi_kind =
                            McuCompiledVariantKind::Psq4Prefill2DK64N128;
                        break;
                    case PhysicalPsq4Variant::Prefill2DK128N128:
                        multi_kind =
                            McuCompiledVariantKind::Psq4Prefill2DK128N128;
                        break;
                    default:
                        return Status::unsupported(
                            "psq4 physical variant has no mcu entrypoint", __FILE__,
                            __LINE__);
                }
                auto minv = mcu_plan_value<mcu::McuPsq4MultiRowInvocation>();
                minv.weight_codes = launch.weight_codes;
                minv.weight_scales = launch.weight_scales;
                minv.activation_codes = launch.activation_codes;
                minv.activation_scales = launch.activation_scales;
                minv.output = launch.output;
                minv.output_dtype = static_cast<uint32_t>(launch.output_dtype);
                minv.rows = launch.rows;
                minv.out_features = launch.out_features;
                minv.k_padded = launch.k_padded;
                minv.weight_scale_stride = launch.weight_scale_stride;
                minv.activation_code_stride = launch.activation_code_stride;
                minv.activation_scale_stride = launch.activation_scale_stride;
                minv.output_row_stride = launch.output_row_stride;
                const uint32_t mindex =
                    static_cast<uint32_t>(out.psq4_multi.size());
                out.psq4_multi.push_back(minv);
                uint8_t multi_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.output_slots[0]},
                        multi_row_source);
                    if (!src_st.ok()) return src_st;
                }
                record_row_patch(McuInvocationTable::Psq4MultiRow, mindex,
                                 multi_row_source);
                const uint16_t mid = mark_variant(
                    out, multi_kind, launch.grid, launch.workgroup, launch.grid_y,
                    launch.grid_z);
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), mid,
                    mcu::kMcuKernargRecipePsq4MultiRowBf16, mindex));
                break;
            }
            case KernelId::LINEAR_PSQ8: {
                PhysicalPsq8Launch launch{};
                const auto resolved = resolve_psq8_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "psq8 dispatch is not applicable on the mcu", __FILE__,
                        __LINE__);
                if (launch.variant == PhysicalPsq8Variant::Decode1Bf16) {
                    if (launch.unroll != 8u)
                        return Status::unsupported(
                            "psq8 physical variant has no mcu entrypoint", __FILE__,
                            __LINE__);
                    auto inv = mcu_plan_value<mcu::McuPsq4Decode1Invocation>();
                    inv.weight_codes = launch.weight_codes;
                    inv.weight_scales = launch.weight_scales;
                    inv.activation_codes = launch.activation_codes;
                    inv.activation_scales = launch.activation_scales;
                    inv.output = launch.output;
                    inv.k_padded = launch.k_padded;
                    inv.weight_scale_stride = launch.weight_scale_stride;
                    const uint32_t index = static_cast<uint32_t>(out.psq4.size());
                    out.psq4.push_back(inv);
                    const uint16_t variant_id = mark_variant(
                        out, McuCompiledVariantKind::Psq8Decode1Bf16U8,
                        launch.grid, launch.workgroup);
                    push_dispatch(make_dispatch_node(
                        out.nodes.size(), variant_id,
                        mcu::kMcuKernargRecipePsq8Decode1Bf16U8, index));
                    break;
                }
                McuCompiledVariantKind multi_kind{};
                switch (launch.variant) {
                    case PhysicalPsq8Variant::RowBlock1Bf16:
                        multi_kind = McuCompiledVariantKind::Psq8RowBlock1Bf16;
                        break;
                    case PhysicalPsq8Variant::RowBlock2Bf16:
                        multi_kind = McuCompiledVariantKind::Psq8RowBlock2Bf16;
                        break;
                    case PhysicalPsq8Variant::RowBlock4Bf16:
                        multi_kind = McuCompiledVariantKind::Psq8RowBlock4Bf16;
                        break;
                    case PhysicalPsq8Variant::RowBlock8Bf16:
                        multi_kind = McuCompiledVariantKind::Psq8RowBlock8Bf16;
                        break;
                    case PhysicalPsq8Variant::Prefill2DN64K64:
                        multi_kind = McuCompiledVariantKind::Psq8Prefill2DN64K64;
                        break;
                    case PhysicalPsq8Variant::Prefill2DN128K64:
                        multi_kind = McuCompiledVariantKind::Psq8Prefill2DN128K64;
                        break;
                    case PhysicalPsq8Variant::Prefill2DN128K128:
                        multi_kind =
                            McuCompiledVariantKind::Psq8Prefill2DN128K128;
                        break;
                    default:
                        return Status::unsupported(
                            "psq8 physical variant has no mcu entrypoint", __FILE__,
                            __LINE__);
                }
                auto minv = mcu_plan_value<mcu::McuPsq4MultiRowInvocation>();
                minv.weight_codes = launch.weight_codes;
                minv.weight_scales = launch.weight_scales;
                minv.activation_codes = launch.activation_codes;
                minv.activation_scales = launch.activation_scales;
                minv.output = launch.output;
                minv.output_dtype = static_cast<uint32_t>(launch.output_dtype);
                minv.rows = launch.rows;
                minv.out_features = launch.out_features;
                minv.k_padded = launch.k_padded;
                minv.weight_scale_stride = launch.weight_scale_stride;
                minv.activation_code_stride = launch.activation_code_stride;
                minv.activation_scale_stride = launch.activation_scale_stride;
                minv.output_row_stride = launch.output_row_stride;
                const uint32_t mindex =
                    static_cast<uint32_t>(out.psq4_multi.size());
                out.psq4_multi.push_back(minv);
                uint8_t multi_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.output_slots[0]},
                        multi_row_source);
                    if (!src_st.ok()) return src_st;
                }
                record_row_patch(McuInvocationTable::Psq4MultiRow, mindex,
                                 multi_row_source);
                const uint16_t mid = mark_variant(
                    out, multi_kind, launch.grid, launch.workgroup, launch.grid_y);
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), mid,
                    mcu::kMcuKernargRecipePsq4MultiRowBf16, mindex));
                break;
            }
            case KernelId::SILU:
            case KernelId::MUL:
            case KernelId::SCALE:
            case KernelId::RESIDUAL_ADD:
            case KernelId::SWIGLU:
            case KernelId::SPLIT:
            case KernelId::SIGMOID: {
                PhysicalElementwiseLaunch launch{};
                const auto resolved =
                    resolve_elementwise_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "elementwise dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                McuCompiledVariantKind kind =
                    McuCompiledVariantKind::ElementwiseResidualBf16;
                switch (launch.profile) {
                    case ElementwiseProfile::ResidualBf16:
                        kind = McuCompiledVariantKind::ElementwiseResidualBf16;
                        break;
                    case ElementwiseProfile::SwigluBf16:
                        kind = McuCompiledVariantKind::ElementwiseSwigluBf16;
                        break;
                    case ElementwiseProfile::MulF32F32ToBf16:
                        kind = McuCompiledVariantKind::ElementwiseMulF32F32Bf16;
                        break;
                    case ElementwiseProfile::ScaleF32:
                        kind = McuCompiledVariantKind::ElementwiseScaleF32;
                        break;
                    case ElementwiseProfile::SiluBf16ToF32:
                        kind = McuCompiledVariantKind::ElementwiseSiluBf16ToF32;
                        break;
                    case ElementwiseProfile::SiluF32ToBf16:
                        kind = McuCompiledVariantKind::ElementwiseSiluF32ToBf16;
                        break;
                    case ElementwiseProfile::SigmoidBf16ToF32:
                        kind = McuCompiledVariantKind::ElementwiseSigmoidBf16ToF32;
                        break;
                    case ElementwiseProfile::SplitBf16Halves:
                    case ElementwiseProfile::SplitBf16InterleavedHeads:
                        kind = McuCompiledVariantKind::ElementwiseSplitBf16;
                        break;
                    default:
                        return Status::unsupported(
                            "elementwise physical variant has no mcu entrypoint",
                            __FILE__, __LINE__);
                }
                auto inv = mcu_plan_value<mcu::McuElementwiseInvocation>();
                inv.in0 = launch.in0;
                inv.in0_row_stride = launch.in0_stride;
                inv.in1 = launch.in1;
                inv.in1_row_stride = launch.in1_stride;
                inv.out0 = launch.out0;
                inv.out0_row_stride = launch.out0_stride;
                inv.out1 = launch.out1;
                inv.out1_row_stride = launch.out1_stride;
                inv.features = launch.features;
                inv.features_b = launch.features_b;
                inv.aux = launch.aux;
                inv.scalar_a = launch.scalar_a;
                inv.rows = launch.rows;
                if (launch.profile == ElementwiseProfile::SplitBf16InterleavedHeads)
                    inv.reserved = 1u;
                const uint32_t index = static_cast<uint32_t>(out.elementwise.size());
                out.elementwise.push_back(inv);
                uint8_t ew_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.output_slots[0]}, ew_row_source);
                    if (!src_st.ok()) return src_st;
                }
                record_row_patch(McuInvocationTable::Elementwise, index,
                                 ew_row_source);
                const uint16_t variant_id =
                    mark_variant(out, kind, launch.grid_x, launch.workgroup,
                                 launch.grid_y);
                const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeElementwise, index));
                record_geometry(node_index, McuGeometryAxis::Y, ew_row_source,
                                0u);
                break;
            }
            case KernelId::PAGED_ATTENTION: {
                PhysicalPagedAttentionPlan plan{};
                const auto resolved =
                    options.static_plan
                        ? resolve_paged_attention_physical_static_plan(
                              program, i, ctx, plan)
                        : resolve_paged_attention_physical(program, i, ctx,
                                                          plan);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "paged attention dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                const auto fill_common =
                    [](const ::ps::kernel::PagedAttentionCommonArgs& c, auto& inv) {
                        inv.q = reinterpret_cast<uint64_t>(c.q);
                        inv.output = reinterpret_cast<uint64_t>(c.output);
                        inv.output_dtype = static_cast<uint32_t>(c.output_dtype);
                        inv.q_row_stride = c.q_row_stride;
                        inv.output_row_stride = c.output_row_stride;
                        inv.row_sequence_slots =
                            reinterpret_cast<uint64_t>(c.row_sequence_slots);
                        inv.row_positions =
                            reinterpret_cast<uint64_t>(c.row_positions);
                        inv.block_tables =
                            reinterpret_cast<uint64_t>(c.block_tables);
                        inv.block_table_stride = c.block_table_stride;
                        inv.max_sequences = c.max_sequences;
                        inv.rows = c.rows;
                        inv.layer = c.layer;
                        inv.page_tokens = c.page_tokens;
                        inv.num_pages = c.num_pages;
                        inv.num_attention_layers = c.num_attention_layers;
                        inv.q_heads = c.q_heads;
                        inv.kv_heads = c.kv_heads;
                        inv.head_dim = c.head_dim;
                        inv.elems_per_token = c.elems_per_token;
                        inv.elems_per_page = c.elems_per_page;
                        inv.elems_per_layer = c.elems_per_layer;
                        inv.scale = c.scale;
                    };
                for (uint32_t region = 0u; region < plan.physical_count; ++region) {
                    const PhysicalPagedAttentionKernel& k = plan.kernels[region];
                    switch (k.variant) {
                        case PhysicalPagedAttentionVariant::Bf16Prefill: {
                            if (k.common.head_dim !=
                                    ::ps::kernel::kPagedAttentionHeadLimit ||
                                k.common.page_tokens != 16u)
                                return Status::unsupported(
                                    "paged attention prefill shape has no mcu "
                                    "entrypoint",
                                    __FILE__, __LINE__);
                            auto inv =
                                mcu_plan_value<mcu::McuPagedAttentionInvocation>();
                            fill_common(k.common, inv);
                            inv.k_pool = k.k_pool;
                            inv.v_pool = k.v_pool;
                            const uint32_t index =
                                static_cast<uint32_t>(out.attention_paged.size());
                            out.attention_paged.push_back(inv);
                            out.attention_regions.push_back(
                                McuAttentionRegion{0u, index, k.row_begin,
                                                   k.common.rows});
                            const uint32_t tiles =
                                (k.common.rows +
                                 ::ps::kernel::kPagedAttentionPrefillTileRows - 1u) /
                                ::ps::kernel::kPagedAttentionPrefillTileRows;
                            const uint16_t variant_id = mark_variant(
                                out,
                                McuCompiledVariantKind::AttentionPagedPrefillBf16,
                                tiles, ::ps::kernel::kPagedAttentionPrefillBlock,
                                k.common.q_heads);
                            push_dispatch(make_dispatch_node(
                                out.nodes.size(), variant_id,
                                mcu::kMcuKernargRecipeAttentionPaged, index));
                            break;
                        }
                        case PhysicalPagedAttentionVariant::Bf16FullHead: {
                            auto inv =
                                mcu_plan_value<mcu::McuPagedAttentionInvocation>();
                            fill_common(k.common, inv);
                            inv.k_pool = k.k_pool;
                            inv.v_pool = k.v_pool;
                            const uint32_t index =
                                static_cast<uint32_t>(out.attention_paged.size());
                            out.attention_paged.push_back(inv);
                            out.attention_regions.push_back(
                                McuAttentionRegion{0u, index, k.row_begin,
                                                   k.common.rows});
                            const uint16_t variant_id = mark_variant(
                                out, McuCompiledVariantKind::AttentionPagedBf16,
                                k.grid, k.workgroup);
                            push_dispatch(make_dispatch_node(
                                out.nodes.size(), variant_id,
                                mcu::kMcuKernargRecipeAttentionPaged, index));
                            break;
                        }
                        case PhysicalPagedAttentionVariant::Bf16SplitQ4Full:
                        case PhysicalPagedAttentionVariant::Bf16SplitOther: {
                            McuCompiledVariantKind split_kind{};
                            if (k.variant ==
                                PhysicalPagedAttentionVariant::Bf16SplitQ4Full) {
                                split_kind =
                                    McuCompiledVariantKind::AttentionPagedSplitQ4Full;
                            } else {
                                if (k.q_per_kv != 6u ||
                                    k.common.head_dim !=
                                        ::ps::kernel::kPagedAttentionHeadLimit)
                                    return Status::unsupported(
                                        "paged attention split other shape has no "
                                        "mcu entrypoint",
                                        __FILE__, __LINE__);
                                split_kind =
                                    McuCompiledVariantKind::AttentionPagedSplitOther;
                            }
                            auto inv = mcu_plan_value<
                                mcu::McuPagedAttentionSplitInvocation>();
                            fill_common(k.common, inv);
                            inv.k_pool = k.k_pool;
                            inv.v_pool = k.v_pool;
                            inv.partials = k.partials;
                            inv.splits = k.splits;
                            const uint32_t index = static_cast<uint32_t>(
                                out.attention_paged_split.size());
                            out.attention_paged_split.push_back(inv);
                            out.attention_regions.push_back(
                                McuAttentionRegion{1u, index, k.row_begin,
                                                   k.common.rows});
                            const uint16_t variant_id =
                                mark_variant(out, split_kind, k.grid, k.workgroup);
                            push_dispatch(make_dispatch_node(
                                out.nodes.size(), variant_id,
                                mcu::kMcuKernargRecipeAttentionPagedSplit, index));
                            break;
                        }
                        case PhysicalPagedAttentionVariant::Bf16ReduceQ4FullQh1:
                        case PhysicalPagedAttentionVariant::Bf16ReduceQ4FullQh2:
                        case PhysicalPagedAttentionVariant::Bf16ReduceQ4FullBase:
                        case PhysicalPagedAttentionVariant::Bf16ReduceOther: {
                            McuCompiledVariantKind reduce_kind{};
                            switch (k.variant) {
                                case PhysicalPagedAttentionVariant::
                                    Bf16ReduceQ4FullQh1:
                                    reduce_kind = McuCompiledVariantKind::
                                        AttentionPagedReduceQ4FullQh1;
                                    break;
                                case PhysicalPagedAttentionVariant::
                                    Bf16ReduceQ4FullQh2:
                                    reduce_kind = McuCompiledVariantKind::
                                        AttentionPagedReduceQ4FullQh2;
                                    break;
                                case PhysicalPagedAttentionVariant::
                                    Bf16ReduceQ4FullBase:
                                    reduce_kind = McuCompiledVariantKind::
                                        AttentionPagedReduceQ4FullBase;
                                    break;
                                default:
                                    if (k.q_per_kv != 6u ||
                                        k.common.head_dim !=
                                            ::ps::kernel::kPagedAttentionHeadLimit)
                                        return Status::unsupported(
                                            "paged attention reduce other shape "
                                            "has no mcu entrypoint",
                                            __FILE__, __LINE__);
                                    reduce_kind = McuCompiledVariantKind::
                                        AttentionPagedReduceOther;
                                    break;
                            }
                            auto inv = mcu_plan_value<
                                mcu::McuPagedAttentionReduceInvocation>();
                            fill_common(k.common, inv);
                            inv.partials = k.partials;
                            inv.splits = k.splits;
                            const uint32_t index = static_cast<uint32_t>(
                                out.attention_paged_reduce.size());
                            out.attention_paged_reduce.push_back(inv);
                            out.attention_regions.push_back(
                                McuAttentionRegion{2u, index, k.row_begin,
                                                   k.common.rows});
                            const uint16_t variant_id =
                                mark_variant(out, reduce_kind, k.grid, k.workgroup);
                            push_dispatch(make_dispatch_node(
                                out.nodes.size(), variant_id,
                                mcu::kMcuKernargRecipeAttentionPagedReduce, index));
                            break;
                        }
                        default:
                            return Status::unsupported(
                                "paged attention physical variant has no mcu "
                                "entrypoint",
                                __FILE__, __LINE__);
                    }
                }
                break;
            }
            case KernelId::KV_APPEND: {
                PhysicalKvAppendLaunch launch{};
                const auto resolved =
                    resolve_kv_append_physical(program, i, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "kv append dispatch is not applicable on the mcu", __FILE__,
                        __LINE__);
                if (launch.variant != PhysicalKvAppendVariant::Bf16)
                    return Status::unsupported(
                        "kv append physical variant has no mcu entrypoint",
                        __FILE__, __LINE__);
                auto inv = mcu_plan_value<mcu::McuKvAppendInvocation>();
                inv.k_input = reinterpret_cast<uint64_t>(launch.common.k_input);
                inv.v_input = reinterpret_cast<uint64_t>(launch.common.v_input);
                inv.k_row_stride = launch.common.k_row_stride;
                inv.v_row_stride = launch.common.v_row_stride;
                inv.row_sequence_slots =
                    reinterpret_cast<uint64_t>(launch.common.row_sequence_slots);
                inv.row_positions =
                    reinterpret_cast<uint64_t>(launch.common.row_positions);
                inv.block_tables =
                    reinterpret_cast<uint64_t>(launch.common.block_tables);
                inv.block_table_stride = launch.common.block_table_stride;
                inv.max_sequences = launch.common.max_sequences;
                inv.rows = launch.common.rows;
                inv.layer = launch.common.layer;
                inv.page_tokens = launch.common.page_tokens;
                inv.num_pages = launch.common.num_pages;
                inv.num_attention_layers = launch.common.num_attention_layers;
                inv.kv_heads = launch.common.kv_heads;
                inv.head_dim = launch.common.head_dim;
                inv.elems_per_token = launch.common.elems_per_token;
                inv.elems_per_page = launch.common.elems_per_page;
                inv.elems_per_layer = launch.common.elems_per_layer;
                inv.k_pool = launch.k_pool;
                inv.v_pool = launch.v_pool;
                const uint32_t index = static_cast<uint32_t>(out.kv_append.size());
                out.kv_append.push_back(inv);
                record_row_patch(McuInvocationTable::KvAppend, index,
                                 static_cast<uint8_t>(
                                     mcu::McuInvocationPatchSource::ActualRows));
                const uint16_t variant_id =
                    mark_variant(out, McuCompiledVariantKind::KvAppendBf16,
                                 launch.grid, launch.workgroup);
                const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id, mcu::kMcuKernargRecipeKvAppend,
                    index));
                if (launch.common.kv_heads != 0u) {
                    record_geometry(
                        node_index, McuGeometryAxis::X,
                        static_cast<uint8_t>(
                            mcu::McuInvocationPatchSource::ActualRowsTimesParam),
                        launch.common.kv_heads);
                }
                break;
            }
            case KernelId::ROPE: {
                PhysicalRopeLaunch launch{};
                const auto resolved =
                    resolve_rope_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "rope dispatch is not applicable on the mcu", __FILE__,
                        __LINE__);
                if (launch.variant != PhysicalRopeVariant::F32Bf16Pair)
                    return Status::unsupported(
                        "rope physical variant has no mcu entrypoint", __FILE__,
                        __LINE__);
                auto inv = mcu_plan_value<mcu::McuRopeInvocation>();
                inv.input = launch.input;
                inv.input_row_stride = launch.input_row_stride;
                inv.output = launch.output;
                inv.output_row_stride = launch.output_row_stride;
                inv.positions = launch.positions;
                inv.inv_freq = launch.inv_freq;
                inv.features = launch.features;
                const uint32_t index = static_cast<uint32_t>(out.rope.size());
                out.rope.push_back(inv);
                uint8_t rope_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.output_slots[0]}, rope_row_source);
                    if (!src_st.ok()) return src_st;
                }
                const uint16_t variant_id =
                    mark_variant(out, McuCompiledVariantKind::RopeF32Bf16Pair,
                                 launch.grid, launch.workgroup);
                const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id, mcu::kMcuKernargRecipeRope,
                    index));
                record_geometry(node_index, McuGeometryAxis::X, rope_row_source,
                                0u);
                break;
            }
            case KernelId::LINEAR_BF16: {
                PhysicalBf16Launch launch{};
                const auto resolved =
                    resolve_bf16_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "bf16 linear dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                if (launch.variant != PhysicalBf16Variant::ExactRows) {
                    McuCompiledVariantKind wmma_kind{};
                    switch (launch.variant) {
                        case PhysicalBf16Variant::Wmma:
                            wmma_kind = McuCompiledVariantKind::Bf16Wmma;
                            break;
                        case PhysicalBf16Variant::WmmaWide:
                            wmma_kind = McuCompiledVariantKind::Bf16WmmaWide;
                            break;
                        case PhysicalBf16Variant::WmmaKPartition:
                            wmma_kind = McuCompiledVariantKind::Bf16SplitKWmma;
                            break;
                        default:
                            return Status::unsupported(
                                "bf16 physical variant has no mcu entrypoint",
                                __FILE__, __LINE__);
                    }
                    if (launch.grid_x == 0u || launch.workgroup == 0u)
                        return Status::unsupported(
                            "bf16 physical variant has no mcu geometry",
                            __FILE__, __LINE__);
                    auto winv = mcu_plan_value<mcu::McuBf16WmmaInvocation>();
                    winv.weight = launch.weight;
                    winv.input = launch.input;
                    winv.output = launch.output;
                    winv.output_dtype =
                        static_cast<uint32_t>(launch.output_dtype);
                    winv.rows = launch.rows;
                    winv.out_features = launch.out_features;
                    winv.k = launch.k;
                    winv.input_row_stride = launch.input_row_stride;
                    winv.output_row_stride = launch.output_row_stride;
                    const uint32_t windex =
                        static_cast<uint32_t>(out.bf16_wmma.size());
                    out.bf16_wmma.push_back(winv);
                    uint8_t wmma_row_source = 0u;
                    {
                        const Status src_st = row_source_for(
                            ::ps::runtime::ValueId{b.output_slots[0]},
                            wmma_row_source);
                        if (!src_st.ok()) return src_st;
                    }
                    record_row_patch(McuInvocationTable::Bf16Wmma, windex,
                                     wmma_row_source);
                    const uint16_t wid = mark_variant(
                        out, wmma_kind, launch.grid_x, launch.workgroup,
                        launch.grid_y);
                    push_dispatch(make_dispatch_node(
                        out.nodes.size(), wid,
                        mcu::kMcuKernargRecipeBf16Wmma, windex));
                    break;
                }
                const uint32_t live_rows = launch.config.exact_rows;
                if (live_rows < 1u || live_rows > 16u)
                    return Status::unsupported(
                        "bf16 exact rows is outside the mcu catalog", __FILE__,
                        __LINE__);
                auto inv = mcu_plan_value<mcu::McuBf16ExactRowsInvocation>();
                inv.weight = launch.weight;
                inv.input = launch.input;
                inv.output = launch.output;
                inv.output_dtype = static_cast<uint32_t>(launch.output_dtype);
                inv.n = launch.out_features;
                inv.k = launch.k;
                inv.input_row_stride = launch.input_row_stride;
                inv.output_row_stride = launch.output_row_stride;
                const uint32_t index = static_cast<uint32_t>(out.bf16.size());
                out.bf16.push_back(inv);
                if (launch.output_rows_domain) {
                    uint16_t ids[16] = {};
                    for (uint32_t r = 1u; r <= 16u; ++r) {
                        ids[r - 1u] = mark_variant(
                            out, kMcuBf16ExactRowsKinds[r - 1u], launch.grid_x,
                            launch.workgroup, launch.grid_y);
                    }
                    for (uint32_t r = 2u; r <= 16u; ++r) {
                        if (ids[r - 1u] !=
                            static_cast<uint16_t>(ids[0] + (r - 1u))) {
                            return Status::unsupported(
                                "bf16 exact rows catalog is not contiguous",
                                __FILE__, __LINE__);
                        }
                    }
                    const uint16_t live_variant = ids[live_rows - 1u];
                    const uint32_t node_index =
                        static_cast<uint32_t>(out.nodes.size());
                    push_dispatch(make_dispatch_node(
                        node_index, live_variant,
                        mcu::kMcuKernargRecipeBf16ExactRows, index));
                    McuBf16VariantCatalog catalog{};
                    catalog.node_index = node_index;
                    catalog.invocation_index = index;
                    catalog.variant_base = ids[0];
                    out.bf16_variant_catalogs.push_back(catalog);
                } else {
                    const uint16_t live_variant = mark_variant(
                        out, kMcuBf16ExactRowsKinds[live_rows - 1u],
                        launch.grid_x, launch.workgroup, launch.grid_y);
                    push_dispatch(make_dispatch_node(
                        out.nodes.size(), live_variant,
                        mcu::kMcuKernargRecipeBf16ExactRows, index));
                }
                break;
            }
            case KernelId::L2_NORMALIZE: {
                PhysicalL2NormalizeLaunch launch{};
                const auto resolved =
                    resolve_l2_normalize_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "l2 normalize dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                auto inv = mcu_plan_value<mcu::McuL2NormalizeInvocation>();
                inv.input = launch.input;
                inv.output = launch.output;
                inv.input_row_stride = launch.input_row_stride;
                inv.output_row_stride = launch.output_row_stride;
                inv.group_size = launch.group_size;
                inv.eps = launch.eps;
                const uint32_t index = static_cast<uint32_t>(out.l2.size());
                out.l2.push_back(inv);
                uint8_t l2_row_source = 0u;
                {
                    const Status src_st = row_source_for(
                        ::ps::runtime::ValueId{b.output_slots[0]}, l2_row_source);
                    if (!src_st.ok()) return src_st;
                }
                const uint16_t variant_id = mark_variant(
                    out, McuCompiledVariantKind::L2NormalizeBf16F32,
                    launch.grid_x, launch.workgroup, launch.grid_y);
                const uint32_t node_index = static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeL2Normalize, index));
                record_geometry(node_index, McuGeometryAxis::Y, l2_row_source,
                                0u);
                break;
            }
            case KernelId::EMBEDDING_LOOKUP: {
                const Status s = emit_embedding(b);
                if (!s.ok()) return s;
                break;
            }
            case KernelId::OUTPUT_GATHER: {
                const Status s = emit_output_gather(b);
                if (!s.ok()) return s;
                break;
            }
            case KernelId::STATEFUL_CAUSAL_CONV1D: {
                PhysicalGdnConv1dLaunch launch{};
                const auto resolved =
                    resolve_gdn_conv1d_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "gdn conv1d dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                const ::ps::kernel::GdnConv1dArgs& a = launch.args;
                auto inv = mcu_plan_value<mcu::McuGdnConv1dInvocation>();
                inv.input = reinterpret_cast<uint64_t>(a.input);
                inv.output = reinterpret_cast<uint64_t>(a.output);
                inv.conv_weight = reinterpret_cast<uint64_t>(a.conv_weight);
                inv.conv_state = reinterpret_cast<uint64_t>(a.conv_state);
                inv.requests = reinterpret_cast<uint64_t>(a.requests);
                inv.actual_rows = a.actual_rows;
                inv.input_row_stride = a.input_row_stride;
                inv.output_row_stride = a.output_row_stride;
                inv.conv_dim = a.conv_dim;
                inv.state_index = a.state_index;
                inv.max_sequences = a.max_sequences;
                inv.conv_slot_stride = a.conv_slot_stride;
                inv.conv_layer_stride = a.conv_layer_stride;
                inv.conv_history_stride = a.conv_history_stride;
                inv.row_tile = ::ps::kernel::gdn_conv1d_resolved_row_tile(
                    launch.row_tile, a.actual_rows);
                inv.conv_history_store =
                    reinterpret_cast<uint64_t>(a.conv_history_store);
                inv.conv_history_row_stride = a.conv_history_row_stride;
                inv.capture_rows = a.capture_rows;
                inv.capture_verify_only = a.capture_verify_only;
                const uint32_t index = static_cast<uint32_t>(out.gdn_conv1d.size());
                out.gdn_conv1d.push_back(inv);
                const uint16_t variant_id = mark_variant(
                    out, McuCompiledVariantKind::GdnConv1dBf16F32, launch.grid_x,
                    launch.workgroup, launch.grid_y, launch.grid_z);
                const uint32_t node_index =
                    static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeGdnConv1d, index));
                record_geometry(node_index, McuGeometryAxis::Y,
                                static_cast<uint8_t>(
                                    mcu::McuInvocationPatchSource::NumRequests),
                                0u);
                break;
            }
            case KernelId::GDN_RECURRENCE: {
                PhysicalGdnRecurrenceLaunch launch{};
                const auto resolved =
                    resolve_gdn_recurrence_physical(program, b, ctx, launch);
                if (!resolved.ok()) return resolved.status();
                if (resolved.value() != PhysicalResolveStatus::Applicable)
                    return Status::unsupported(
                        "gdn recurrence dispatch is not applicable on the mcu",
                        __FILE__, __LINE__);
                McuCompiledVariantKind rec_kind;
                if (launch.variant == PhysicalGdnRecurrenceVariant::WmmaDecode1) {
                    rec_kind = McuCompiledVariantKind::GdnRecurrenceWmmaDecode1;
                } else if (launch.variant ==
                           PhysicalGdnRecurrenceVariant::DecodeRowsExact) {
                    switch (launch.kernel.actual_rows) {
                        case 2u:
                            rec_kind =
                                McuCompiledVariantKind::
                                    GdnRecurrenceDecodeRowsExactR2;
                            break;
                        case 4u:
                            rec_kind =
                                McuCompiledVariantKind::
                                    GdnRecurrenceDecodeRowsExactR4;
                            break;
                        case 8u:
                            rec_kind =
                                McuCompiledVariantKind::
                                    GdnRecurrenceDecodeRowsExactR8;
                            break;
                        default:
                            return Status::unsupported(
                                "gdn decode rows variant has no mcu entrypoint",
                                __FILE__, __LINE__);
                    }
                } else if (launch.variant == PhysicalGdnRecurrenceVariant::Wmma) {
                    rec_kind = launch.lossy
                                   ? McuCompiledVariantKind::GdnRecurrenceWmmaLossy
                                   : McuCompiledVariantKind::GdnRecurrenceWmmaExact;
                } else if (launch.variant ==
                               PhysicalGdnRecurrenceVariant::Decode1Serial ||
                           launch.variant ==
                               PhysicalGdnRecurrenceVariant::WmmaSerial) {
                    rec_kind = launch.lossy
                                   ? McuCompiledVariantKind::
                                         GdnRecurrenceWmmaSerialLossy
                                   : McuCompiledVariantKind::
                                         GdnRecurrenceWmmaSerialExact;
                } else {
                    return Status::unsupported(
                        "gdn recurrence physical variant has no mcu entrypoint",
                        __FILE__, __LINE__);
                }
                auto inv = mcu_plan_value<mcu::McuGdnRecurrenceInvocation>();
                std::memcpy(&inv, &launch.kernel, sizeof(inv));
                const uint32_t index =
                    static_cast<uint32_t>(out.gdn_recurrence.size());
                out.gdn_recurrence.push_back(inv);
                const uint16_t variant_id = mark_variant(
                    out, rec_kind, launch.grid_x, launch.workgroup, launch.grid_y,
                    launch.grid_z);
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeGdnRecurrence, index));
                break;
            }
            case KernelId::SAMPLING: {
                if (sc.stochastic_output_capacity != 0u)
                    return Status::unsupported(
                        "stochastic sampling has no mcu entrypoint", __FILE__,
                        __LINE__);
                auto logits_res = resolve_host_value(
                    program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
                if (!logits_res.ok()) return logits_res.status();
                const HostResolvedValue logits = logits_res.value();
                auto sampled_res = resolve_host_value(
                    program, ::ps::runtime::ValueId{b.output_slots[0]}, ctx);
                if (!sampled_res.ok()) return sampled_res.status();
                const HostResolvedValue sampled = sampled_res.value();
                if (logits.dtype != ::ps::runtime::ValueDType::F32 ||
                    logits.row_domain !=
                        ::ps::runtime::ValueRowDomain::OUTPUT_ROWS ||
                    logits.ptr == nullptr || logits.feature_count == 0u ||
                    logits.row_stride < logits.feature_count)
                    return Status::unsupported("sampling logits value is invalid",
                                               __FILE__, __LINE__);
                if (sampled.dtype != ::ps::runtime::ValueDType::I32 ||
                    sampled.ptr == nullptr)
                    return Status::unsupported("sampling output value is invalid",
                                               __FILE__, __LINE__);
                if (sc.output_sampling_params == nullptr)
                    return Status::unsupported(
                        "sampling requires output sampling params", __FILE__,
                        __LINE__);
                const uint32_t outputs = sc.output_capacity;
                if (outputs == 0u)
                    return Status::unsupported("sampling output count is zero",
                                               __FILE__, __LINE__);
                auto inv = mcu_plan_value<mcu::McuArgmaxF32Invocation>();
                inv.logits = reinterpret_cast<uint64_t>(logits.ptr);
                inv.sampling =
                    reinterpret_cast<uint64_t>(sc.output_sampling_params);
                inv.sampled_tokens = reinterpret_cast<uint64_t>(sampled.ptr);
                inv.outputs = outputs;
                inv.vocab_size = logits.feature_count;
                inv.logits_row_stride = logits.row_stride;
                inv.output_rows =
                    reinterpret_cast<uint64_t>(sc.device_output_rows);
                const uint32_t index =
                    static_cast<uint32_t>(out.argmax_f32.size());
                out.argmax_f32.push_back(inv);
                const uint16_t variant_id = mark_variant(
                    out, McuCompiledVariantKind::SamplingArgmaxF32, outputs,
                    ::ps::kernel::kArgmaxF32Threads);
                const uint32_t node_index =
                    static_cast<uint32_t>(out.nodes.size());
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeArgmaxF32, index));
                record_geometry(node_index, McuGeometryAxis::X,
                                static_cast<uint8_t>(
                                    mcu::McuInvocationPatchSource::NumOutputs),
                                0u);
                break;
            }
            case KernelId::VERIFY_ACCEPT: {
                if (b.input_count != 3u || b.output_count != 1u)
                    return Status::unsupported(
                        "verify accept io count is invalid", __FILE__, __LINE__);
                if (sc.batch_context == nullptr ||
                    sc.device_requests == nullptr ||
                    sc.device_output_rows == nullptr)
                    return Status::unsupported(
                        "verify accept requires a batch context", __FILE__,
                        __LINE__);
                const uint32_t request_count = sc.request_capacity;
                const uint32_t max_candidates = b.rotary_dim;
                if (request_count == 0u || max_candidates == 0u)
                    return Status::unsupported(
                        "verify accept geometry is invalid", __FILE__, __LINE__);
                auto cand_res = resolve_host_value(
                    program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
                if (!cand_res.ok()) return cand_res.status();
                auto samp_res = resolve_host_value(
                    program, ::ps::runtime::ValueId{b.input_slots[1]}, ctx);
                if (!samp_res.ok()) return samp_res.status();
                auto count_res = resolve_host_value(
                    program, ::ps::runtime::ValueId{b.input_slots[2]}, ctx);
                if (!count_res.ok()) return count_res.status();
                auto committed_res = resolve_host_value(
                    program, ::ps::runtime::ValueId{b.output_slots[0]}, ctx);
                if (!committed_res.ok()) return committed_res.status();
                const HostResolvedValue cand = cand_res.value();
                const HostResolvedValue samp = samp_res.value();
                const HostResolvedValue counts = count_res.value();
                const HostResolvedValue committed = committed_res.value();
                if (cand.ptr == nullptr || samp.ptr == nullptr ||
                    counts.ptr == nullptr || committed.ptr == nullptr)
                    return Status::unsupported(
                        "verify accept value is invalid", __FILE__, __LINE__);
                auto inv = mcu_plan_value<mcu::McuVerifyAcceptPrefixInvocation>();
                inv.candidates = reinterpret_cast<uint64_t>(cand.ptr);
                inv.sampled_tokens = reinterpret_cast<uint64_t>(samp.ptr);
                inv.candidate_counts = reinterpret_cast<uint64_t>(counts.ptr);
                inv.committed_counts = reinterpret_cast<uint64_t>(committed.ptr);
                inv.request_count = request_count;
                inv.candidate_stride = max_candidates;
                inv.max_candidates = max_candidates;
                inv.requests = reinterpret_cast<uint64_t>(sc.device_requests);
                inv.output_rows =
                    reinterpret_cast<uint64_t>(sc.device_output_rows);
                const uint32_t index =
                    static_cast<uint32_t>(out.verify_accept.size());
                out.verify_accept.push_back(inv);
                const uint16_t variant_id = mark_variant(
                    out, McuCompiledVariantKind::VerifyAcceptPrefix, 1u,
                    ::ps::kernel::kVerifyAcceptPrefixThreads);
                push_dispatch(make_dispatch_node(
                    out.nodes.size(), variant_id,
                    mcu::kMcuKernargRecipeVerifyAcceptPrefix, index));
                break;
            }
            default:
                return Status::unsupported(
                    ::ps::runtime::kernel_to_name(b.kernel_id), __FILE__,
                    __LINE__);
        }
    }
    if (options.verify_committed_counts != nullptr &&
        options.verify_sampled_tokens != nullptr &&
        options.verify_max_candidates != 0u && sc.batch_context != nullptr &&
        sc.device_requests != nullptr) {
        auto inv = mcu_plan_value<mcu::McuVerifyAcceptBatchInvocation>();
        inv.context = reinterpret_cast<uint64_t>(sc.batch_context);
        inv.sampled_tokens =
            reinterpret_cast<uint64_t>(options.verify_sampled_tokens);
        inv.committed_counts =
            reinterpret_cast<uint64_t>(options.verify_committed_counts);
        inv.max_candidates = options.verify_max_candidates;
        const uint32_t index =
            static_cast<uint32_t>(out.verify_accept_batch.size());
        out.verify_accept_batch.push_back(inv);
        const uint16_t variant_id = mark_variant(
            out, McuCompiledVariantKind::VerifyAcceptBatch, 1u,
            ::ps::kernel::kVerifyAcceptPrefixThreads);
        out.epilogue.verify_accept_capable = true;
        out.epilogue.verify_accept_variant = variant_id;
        out.epilogue.verify_accept_invocation = index;
        out.epilogue.verify_accept_node =
            static_cast<uint32_t>(out.nodes.size());
        push_dispatch(make_dispatch_node(
            out.nodes.size(), variant_id,
            mcu::kMcuKernargRecipeVerifyAcceptBatch, index));
    }

    if (options.verify_conv_history != nullptr &&
        options.verify_recurrent_history != nullptr && sc.batch_context != nullptr &&
        sc.model_state.gdn.conv_base != nullptr &&
        sc.model_state.gdn.recurrent_base != nullptr) {
        auto inv =
            mcu_plan_value<mcu::McuGdnSpecRestoreFromCountsInvocation>();
        inv.context = reinterpret_cast<uint64_t>(sc.batch_context);
        inv.committed_counts =
            reinterpret_cast<uint64_t>(options.verify_committed_counts);
        inv.conv_history = reinterpret_cast<uint64_t>(options.verify_conv_history);
        inv.recurrent_history =
            reinterpret_cast<uint64_t>(options.verify_recurrent_history);
        inv.conv_pool = reinterpret_cast<uint64_t>(sc.model_state.gdn.conv_base);
        inv.recurrent_pool =
            reinterpret_cast<uint64_t>(sc.model_state.gdn.recurrent_base);
        inv.conv_slot_stride = sc.model_state.gdn.conv_slot_stride;
        inv.recurrent_slot_stride = sc.model_state.gdn.recurrent_slot_stride;
        inv.conv_history_stride = options.verify_conv_history_stride;
        inv.recurrent_history_stride = options.verify_recurrent_history_stride;
        inv.conv_elems = static_cast<uint32_t>(sc.model_state.gdn.conv_slot_stride);
        inv.recurrent_elems =
            static_cast<uint32_t>(sc.model_state.gdn.recurrent_slot_stride);
        inv.max_candidates = options.verify_max_candidates;
        const uint32_t index = static_cast<uint32_t>(
            out.gdn_spec_restore_from_counts.size());
        out.gdn_spec_restore_from_counts.push_back(inv);
        const uint32_t elems = inv.conv_elems > inv.recurrent_elems
                                   ? inv.conv_elems
                                   : inv.recurrent_elems;
        const uint16_t variant_id = mark_variant(
            out, McuCompiledVariantKind::GdnSpecRestoreFromCounts,
            (elems + ::ps::kernel::kGdnSpecRestoreThreads - 1u) /
                ::ps::kernel::kGdnSpecRestoreThreads,
            ::ps::kernel::kGdnSpecRestoreThreads);
        out.epilogue.gdn_restore_capable = true;
        out.epilogue.gdn_restore_variant = variant_id;
        out.epilogue.gdn_restore_invocation = index;
        out.epilogue.gdn_restore_node = static_cast<uint32_t>(out.nodes.size());
        push_dispatch(make_dispatch_node(
            out.nodes.size(), variant_id,
            mcu::kMcuKernargRecipeGdnSpecRestoreFromCounts, index));
    }

    mcu::McuPlanNode marker{};
    marker.variant_id = marker_variant;    marker.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    marker.completion_slot = 0;
    marker.input_slot = mcu::kMcuNoInput;
    marker.flags = mcu::kMcuNodeDispatch;
    marker.next = static_cast<uint16_t>(out.nodes.size() + 1u);
    out.nodes.push_back(marker);

    mcu::McuPlanNode wait{};
    wait.completion_slot = 0;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = static_cast<uint16_t>(out.nodes.size() + 1u);
    out.nodes.push_back(wait);

    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    out.nodes.push_back(end);

    if (out.variants.size() > mcu::kMcuMaxVariants)
        return Status::unsupported("mcu plan exceeds the variant capacity",
                                   __FILE__, __LINE__);

    out.logical_dispatch_count = options.dispatch_end - options.dispatch_begin;
    out.dispatch_count = out.logical_dispatch_count;
    out.physical_dispatch_count = physical_nodes;
    out.kernarg_slots_required = physical_nodes + 1u;
    out.marker_count = 1u;
    return Status::make_ok();
}

Status compile_mcu_plan(const Program& program,
                        const McuStaticPlanCompileContext& static_ctx,
                        const McuPlanCompileOptions& options,
                        McuCompiledPlan& out) {
    const Status status =
        compile_mcu_plan_impl(program, static_ctx, options, out);
    const char* plan_dump = std::getenv("PHASESHIFT_MCU_PLAN_DUMP");
    if (plan_dump != nullptr && plan_dump[0] != '\0') {
        std::fprintf(stderr, "MCU_PLAN_DUMP status=%s variants=%zu nodes=%zu\n",
                     status.ok() ? "ok" : "unsupported", out.variants.size(),
                     out.nodes.size());
        for (const McuCompiledVariant& v : out.variants) {
            std::fprintf(stderr, "MCU_PLAN_VARIANT %s grid=%u gy=%u gz=%u wg=%u\n",
                         mcu_kernel_variant_kind_name(v.kind), v.grid, v.grid_y,
                         v.grid_z, v.workgroup);
        }
        uint32_t dispatch_index = 0u;
        for (const mcu::McuPlanNode& node : out.nodes) {
            if (node.flags != mcu::kMcuNodeDispatch ||
                node.variant_id >= out.variants.size())
                continue;
            std::fprintf(stderr, "MCU_DISPATCH %u %s\n", dispatch_index,
                         mcu_kernel_variant_kind_name(
                             out.variants[node.variant_id].kind));
            ++dispatch_index;
        }
    }
    if (!status.ok()) out = McuCompiledPlan{};
    return status;
}

}  // namespace ps::qwen35::runtime
