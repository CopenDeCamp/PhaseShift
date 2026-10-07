#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/runtime/gpu_mcu/invocation_abi.h>
#include <phaseshift/models/qwen35/runtime/gpu_mcu/kernarg_recipe.h>
#include <phaseshift/runtime/gpu_mcu/execution/micro_fsm.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/reset.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compile_context.h>
#include <phaseshift/runtime/program/program.h>

#include <cstdint>
#include <vector>

namespace ps::qwen35::runtime {

enum class McuCompiledVariantKind : uint8_t {
    CompletionMarker = 0,
    RmsNormBf16PfOnePlus = 1,
    ActivationQuantizeE4m3K5120 = 2,
    Psq4Decode1Bf16U16 = 3,
    RmsNormF32PgDirect = 4,
    ElementwiseResidualBf16 = 5,
    ElementwiseSwigluBf16 = 6,
    ElementwiseMulF32F32Bf16 = 7,
    ElementwiseScaleF32 = 8,
    ElementwiseSiluBf16ToF32 = 9,
    ElementwiseSiluF32ToBf16 = 10,
    Bf16ExactRows = 11,
    L2NormalizeBf16F32 = 12,
    GdnConv1dBf16F32 = 13,
    GdnRecurrenceWmmaDecode1 = 14,
    GdnRecurrenceWmmaLossy = 15,
    GdnRecurrenceWmmaExact = 16,
    ElementwiseSigmoidBf16ToF32 = 17,
    ElementwiseSplitBf16 = 18,
    RmsNormBf16F32PgOnePlus = 19,
    RopeF32Bf16Pair = 20,
    KvAppendBf16 = 21,
    AttentionPagedBf16 = 22,
    AttentionPagedSplitQ4Full = 23,
    AttentionPagedReduceQ4FullQh1 = 24,
    AttentionPagedReduceQ4FullQh2 = 25,
    AttentionPagedReduceQ4FullBase = 26,
    GdnResetZero = 27,
    AttentionPagedPrefillBf16 = 28,
    ActivationQuantizeE4m3K6144 = 29,
    ActivationQuantizeE4m3K17408 = 30,
    Psq4Decode1Bf16U8 = 31,
    Psq8Decode1Bf16U8 = 32,
    AttentionPagedSplitOther = 33,
    AttentionPagedReduceOther = 34,
    Psq4RowBlock1Bf16 = 35,
    Psq4RowBlock2Bf16 = 36,
    Psq4RowBlock4Bf16 = 37,
    Psq4RowBlock8Bf16 = 38,
    VerifyAcceptPrefix = 39,
    GdnSpecRestore = 40,
    SamplingArgmaxF32 = 41,
    EmbeddingBf16 = 42,
    OutputGatherBf16 = 43,
    VerifyAcceptBatch = 44,
    GdnSpecRestoreFromCounts = 45,
    Bf16ExactRows2 = 46,
    Bf16ExactRows3 = 47,
    Bf16ExactRows4 = 48,
    Bf16ExactRows5 = 49,
    Bf16ExactRows6 = 50,
    Bf16ExactRows7 = 51,
    Bf16ExactRows8 = 52,
    Bf16ExactRows9 = 53,
    Bf16ExactRows10 = 54,
    Bf16ExactRows11 = 55,
    Bf16ExactRows12 = 56,
    Bf16ExactRows13 = 57,
    Bf16ExactRows14 = 58,
    Bf16ExactRows15 = 59,
    Bf16ExactRows16 = 60,
    Psq8RowBlock1Bf16 = 61,
    Psq8RowBlock2Bf16 = 62,
    Psq8RowBlock4Bf16 = 63,
    Psq8RowBlock8Bf16 = 64,
    EmbeddingPsq8 = 65,
    Bf16Wmma = 66,
    Bf16WmmaWide = 67,
    Bf16SplitKWmma = 68,
    GdnRecurrenceDecodeRowsExactR2 = 69,
    GdnRecurrenceDecodeRowsExactR4 = 70,
    GdnRecurrenceDecodeRowsExactR8 = 71,
    GdnRecurrenceWmmaSerialLossy = 72,
    GdnRecurrenceWmmaSerialExact = 73,
    Psq8Prefill2DN64K64 = 74,
    Psq8Prefill2DN128K64 = 75,
    Psq8Prefill2DN128K128 = 76,
    Psq4Prefill2DK64N64 = 77,
    Psq4Prefill2DK128N64 = 78,
    Psq4Prefill2DK64N128 = 79,
    Psq4Prefill2DK128N128 = 80,
};

inline constexpr McuCompiledVariantKind kMcuBf16ExactRowsKinds[16] = {
    McuCompiledVariantKind::Bf16ExactRows,
    McuCompiledVariantKind::Bf16ExactRows2,
    McuCompiledVariantKind::Bf16ExactRows3,
    McuCompiledVariantKind::Bf16ExactRows4,
    McuCompiledVariantKind::Bf16ExactRows5,
    McuCompiledVariantKind::Bf16ExactRows6,
    McuCompiledVariantKind::Bf16ExactRows7,
    McuCompiledVariantKind::Bf16ExactRows8,
    McuCompiledVariantKind::Bf16ExactRows9,
    McuCompiledVariantKind::Bf16ExactRows10,
    McuCompiledVariantKind::Bf16ExactRows11,
    McuCompiledVariantKind::Bf16ExactRows12,
    McuCompiledVariantKind::Bf16ExactRows13,
    McuCompiledVariantKind::Bf16ExactRows14,
    McuCompiledVariantKind::Bf16ExactRows15,
    McuCompiledVariantKind::Bf16ExactRows16,
};

struct McuCompiledVariant {
    McuCompiledVariantKind kind = McuCompiledVariantKind::CompletionMarker;
    uint32_t grid = 0;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t workgroup = 0;
};

struct McuAttentionRegion {
    uint32_t invocation_kind = 0;
    uint32_t invocation_index = 0;
    uint32_t row_begin = 0;
    uint32_t rows = 0;
};

struct McuBf16VariantCatalog {
    uint32_t node_index = 0;
    uint32_t invocation_index = 0;
    uint32_t variant_base = 0;
};

enum class McuGeometryAxis : uint8_t {
    X = 0,
    Y = 1,
    Z = 2,
};

struct McuGeometryPatch {
    uint32_t node_index = 0;
    McuGeometryAxis axis = McuGeometryAxis::X;
    uint8_t source = 0;
    uint32_t param = 0;
};

enum class McuInvocationTable : uint8_t {
    RmsNorm = 0,
    Elementwise = 1,
    KvAppend = 2,
    EmbeddingBf16 = 3,
    EmbeddingPsq8 = 4,
    Bf16Wmma = 5,
    Psq4MultiRow = 6,
};

struct McuInvocationRowPatch {
    McuInvocationTable table = McuInvocationTable::RmsNorm;
    uint8_t source = 0;
    uint32_t index = 0;
};

struct McuRuntimeEpilogue {
    bool verify_accept_capable = false;
    bool gdn_restore_capable = false;
    uint32_t verify_accept_variant = 0;
    uint32_t gdn_restore_variant = 0;
    uint32_t verify_accept_invocation = 0;
    uint32_t gdn_restore_invocation = 0;
    uint32_t verify_accept_node = 0;
    uint32_t gdn_restore_node = 0;
};

struct McuCompiledPlan {
    std::vector<::ps::runtime::gpu_mcu::McuPlanNode> nodes;
    std::vector<::ps::runtime::gpu_mcu::McuRmsNormInvocation> rms;
    std::vector<::ps::runtime::gpu_mcu::McuActivationQuantizeE4m3Invocation>
        quant_e4m3;
    std::vector<::ps::runtime::gpu_mcu::McuPsq4Decode1Invocation> psq4;
    std::vector<::ps::runtime::gpu_mcu::McuPsq4MultiRowInvocation> psq4_multi;
    std::vector<::ps::runtime::gpu_mcu::McuVerifyAcceptPrefixInvocation>
        verify_accept;
    std::vector<::ps::runtime::gpu_mcu::McuGdnSpecRestoreInvocation>
        gdn_spec_restore;
    std::vector<::ps::runtime::gpu_mcu::McuArgmaxF32Invocation> argmax_f32;
    std::vector<::ps::runtime::gpu_mcu::McuElementwiseInvocation> elementwise;
    std::vector<::ps::runtime::gpu_mcu::McuRopeInvocation> rope;
    std::vector<::ps::runtime::gpu_mcu::McuKvAppendInvocation> kv_append;
    std::vector<::ps::runtime::gpu_mcu::McuPagedAttentionInvocation>
        attention_paged;
    std::vector<::ps::runtime::gpu_mcu::McuPagedAttentionSplitInvocation>
        attention_paged_split;
    std::vector<::ps::runtime::gpu_mcu::McuPagedAttentionReduceInvocation>
        attention_paged_reduce;
    std::vector<::ps::runtime::gpu_mcu::McuBf16ExactRowsInvocation> bf16;
    std::vector<::ps::runtime::gpu_mcu::McuBf16WmmaInvocation> bf16_wmma;
    std::vector<::ps::runtime::gpu_mcu::McuL2NormalizeInvocation> l2;
    std::vector<::ps::runtime::gpu_mcu::McuEmbeddingBf16Invocation> embedding;
    std::vector<::ps::runtime::gpu_mcu::McuEmbeddingPsq8Invocation>
        embedding_psq8;
    std::vector<::ps::runtime::gpu_mcu::McuOutputGatherBf16Invocation>
        output_gather;
    std::vector<::ps::runtime::gpu_mcu::McuVerifyAcceptBatchInvocation>
        verify_accept_batch;
    std::vector<::ps::runtime::gpu_mcu::McuGdnSpecRestoreFromCountsInvocation>
        gdn_spec_restore_from_counts;
    std::vector<McuAttentionRegion> attention_regions;
    std::vector<McuBf16VariantCatalog> bf16_variant_catalogs;
    std::vector<McuGeometryPatch> geometry_patches;
    std::vector<McuInvocationRowPatch> invocation_row_patches;
    McuRuntimeEpilogue epilogue;
    std::vector<::ps::runtime::gpu_mcu::McuGdnConv1dInvocation> gdn_conv1d;
    std::vector<::ps::runtime::gpu_mcu::McuGdnRecurrenceInvocation> gdn_recurrence;
    std::vector<::ps::runtime::gpu_mcu::GpuMcuGdnResetInvocation> gdn_reset;
    std::vector<McuCompiledVariant> variants;
    uint32_t dispatch_count = 0;
    uint32_t logical_dispatch_count = 0;
    uint32_t physical_dispatch_count = 0;
    uint32_t kernarg_slots_required = 0;
    uint32_t marker_count = 0;
};

struct McuPlanCompileOptions {
    uint32_t dispatch_begin = 0;
    uint32_t dispatch_end = 0;
    uint32_t marker_grid = 1;
    uint32_t marker_workgroup = 64;
    const int32_t* verify_sampled_tokens = nullptr;
    uint32_t* verify_committed_counts = nullptr;
    uint32_t verify_max_candidates = 0;
    const void* verify_conv_history = nullptr;
    const void* verify_recurrent_history = nullptr;
    uint64_t verify_conv_history_stride = 0;
    uint64_t verify_recurrent_history_stride = 0;
    bool static_plan = false;
};

Status compile_mcu_plan(const ::ps::runtime::Program& program,
                        const McuStaticPlanCompileContext& static_ctx,
                        const McuPlanCompileOptions& options,
                        McuCompiledPlan& out);

}  // namespace ps::qwen35::runtime
