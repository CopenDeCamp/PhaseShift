#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>

#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/embedding.h>
#include <phaseshift/models/qwen35/kernels/optimized/output_gather.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/models/qwen35/kernels/optimized/l2_normalize.h>
#include <phaseshift/models/qwen35/kernels/optimized/verify_accept.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.h>
#include <phaseshift/models/qwen35/kernels/optimized/sampling.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>

namespace ps {
namespace qwen35 {
namespace runtime {

namespace mcu = ::ps::runtime::gpu_mcu;
namespace kernel = ::ps::kernel;

const char* mcu_kernel_variant_kind_name(McuCompiledVariantKind kind) {
    switch (kind) {
        case McuCompiledVariantKind::CompletionMarker:
            return "CompletionMarker";
        case McuCompiledVariantKind::GdnResetZero:
            return "GdnResetZero";
        case McuCompiledVariantKind::RmsNormBf16PfOnePlus:
            return "RmsNormBf16PfOnePlus";
        case McuCompiledVariantKind::ActivationQuantizeE4m3K5120:
            return "ActivationQuantizeE4m3K5120";
        case McuCompiledVariantKind::ActivationQuantizeE4m3K6144:
            return "ActivationQuantizeE4m3K6144";
        case McuCompiledVariantKind::ActivationQuantizeE4m3K17408:
            return "ActivationQuantizeE4m3K17408";
        case McuCompiledVariantKind::Psq4Decode1Bf16U16:
            return "Psq4Decode1Bf16U16";
        case McuCompiledVariantKind::Psq4Decode1Bf16U8:
            return "Psq4Decode1Bf16U8";
        case McuCompiledVariantKind::Psq8Decode1Bf16U8:
            return "Psq8Decode1Bf16U8";
        case McuCompiledVariantKind::Psq8RowBlock1Bf16:
            return "Psq8RowBlock1Bf16";
        case McuCompiledVariantKind::Psq8RowBlock2Bf16:
            return "Psq8RowBlock2Bf16";
        case McuCompiledVariantKind::Psq8RowBlock4Bf16:
            return "Psq8RowBlock4Bf16";
        case McuCompiledVariantKind::Psq8RowBlock8Bf16:
            return "Psq8RowBlock8Bf16";
        case McuCompiledVariantKind::Psq8Prefill2DN64K64:
            return "Psq8Prefill2DN64K64";
        case McuCompiledVariantKind::Psq8Prefill2DN128K64:
            return "Psq8Prefill2DN128K64";
        case McuCompiledVariantKind::Psq8Prefill2DN128K128:
            return "Psq8Prefill2DN128K128";
        case McuCompiledVariantKind::Psq4RowBlock1Bf16:
            return "Psq4RowBlock1Bf16";
        case McuCompiledVariantKind::Psq4RowBlock2Bf16:
            return "Psq4RowBlock2Bf16";
        case McuCompiledVariantKind::Psq4RowBlock4Bf16:
            return "Psq4RowBlock4Bf16";
        case McuCompiledVariantKind::Psq4RowBlock8Bf16:
            return "Psq4RowBlock8Bf16";
        case McuCompiledVariantKind::VerifyAcceptPrefix:
            return "VerifyAcceptPrefix";
        case McuCompiledVariantKind::GdnSpecRestore:
            return "GdnSpecRestore";
        case McuCompiledVariantKind::SamplingArgmaxF32:
            return "SamplingArgmaxF32";
        case McuCompiledVariantKind::EmbeddingBf16:
            return "EmbeddingBf16";
        case McuCompiledVariantKind::EmbeddingPsq8:
            return "EmbeddingPsq8";
        case McuCompiledVariantKind::OutputGatherBf16:
            return "OutputGatherBf16";
        case McuCompiledVariantKind::VerifyAcceptBatch:
            return "VerifyAcceptBatch";
        case McuCompiledVariantKind::GdnSpecRestoreFromCounts:
            return "GdnSpecRestoreFromCounts";
        case McuCompiledVariantKind::RmsNormF32PgDirect:
            return "RmsNormF32PgDirect";
        case McuCompiledVariantKind::ElementwiseResidualBf16:
            return "ElementwiseResidualBf16";
        case McuCompiledVariantKind::ElementwiseSwigluBf16:
            return "ElementwiseSwigluBf16";
        case McuCompiledVariantKind::ElementwiseMulF32F32Bf16:
            return "ElementwiseMulF32F32Bf16";
        case McuCompiledVariantKind::ElementwiseScaleF32:
            return "ElementwiseScaleF32";
        case McuCompiledVariantKind::ElementwiseSiluBf16ToF32:
            return "ElementwiseSiluBf16ToF32";
        case McuCompiledVariantKind::ElementwiseSiluF32ToBf16:
            return "ElementwiseSiluF32ToBf16";
        case McuCompiledVariantKind::Bf16ExactRows:
            return "Bf16ExactRows";
        case McuCompiledVariantKind::Bf16Wmma:
            return "Bf16Wmma";
        case McuCompiledVariantKind::Bf16WmmaWide:
            return "Bf16WmmaWide";
        case McuCompiledVariantKind::Bf16SplitKWmma:
            return "Bf16SplitKWmma";
        case McuCompiledVariantKind::Bf16ExactRows2:
            return "Bf16ExactRows2";
        case McuCompiledVariantKind::Bf16ExactRows3:
            return "Bf16ExactRows3";
        case McuCompiledVariantKind::Bf16ExactRows4:
            return "Bf16ExactRows4";
        case McuCompiledVariantKind::Bf16ExactRows5:
            return "Bf16ExactRows5";
        case McuCompiledVariantKind::Bf16ExactRows6:
            return "Bf16ExactRows6";
        case McuCompiledVariantKind::Bf16ExactRows7:
            return "Bf16ExactRows7";
        case McuCompiledVariantKind::Bf16ExactRows8:
            return "Bf16ExactRows8";
        case McuCompiledVariantKind::Bf16ExactRows9:
            return "Bf16ExactRows9";
        case McuCompiledVariantKind::Bf16ExactRows10:
            return "Bf16ExactRows10";
        case McuCompiledVariantKind::Bf16ExactRows11:
            return "Bf16ExactRows11";
        case McuCompiledVariantKind::Bf16ExactRows12:
            return "Bf16ExactRows12";
        case McuCompiledVariantKind::Bf16ExactRows13:
            return "Bf16ExactRows13";
        case McuCompiledVariantKind::Bf16ExactRows14:
            return "Bf16ExactRows14";
        case McuCompiledVariantKind::Bf16ExactRows15:
            return "Bf16ExactRows15";
        case McuCompiledVariantKind::Bf16ExactRows16:
            return "Bf16ExactRows16";
        case McuCompiledVariantKind::L2NormalizeBf16F32:
            return "L2NormalizeBf16F32";
        case McuCompiledVariantKind::GdnConv1dBf16F32:
            return "GdnConv1dBf16F32";
        case McuCompiledVariantKind::GdnRecurrenceWmmaDecode1:
            return "GdnRecurrenceWmmaDecode1";
        case McuCompiledVariantKind::GdnRecurrenceDecodeRowsExactR2:
            return "GdnRecurrenceDecodeRowsExactR2";
        case McuCompiledVariantKind::GdnRecurrenceDecodeRowsExactR4:
            return "GdnRecurrenceDecodeRowsExactR4";
        case McuCompiledVariantKind::GdnRecurrenceDecodeRowsExactR8:
            return "GdnRecurrenceDecodeRowsExactR8";
        case McuCompiledVariantKind::GdnRecurrenceWmmaSerialLossy:
            return "GdnRecurrenceWmmaSerialLossy";
        case McuCompiledVariantKind::GdnRecurrenceWmmaSerialExact:
            return "GdnRecurrenceWmmaSerialExact";
        case McuCompiledVariantKind::GdnRecurrenceWmmaLossy:
            return "GdnRecurrenceWmmaLossy";
        case McuCompiledVariantKind::GdnRecurrenceWmmaExact:
            return "GdnRecurrenceWmmaExact";
        case McuCompiledVariantKind::ElementwiseSigmoidBf16ToF32:
            return "ElementwiseSigmoidBf16ToF32";
        case McuCompiledVariantKind::ElementwiseSplitBf16:
            return "ElementwiseSplitBf16";
        case McuCompiledVariantKind::RmsNormBf16F32PgOnePlus:
            return "RmsNormBf16F32PgOnePlus";
        case McuCompiledVariantKind::RopeF32Bf16Pair:
            return "RopeF32Bf16Pair";
        case McuCompiledVariantKind::KvAppendBf16:
            return "KvAppendBf16";
        case McuCompiledVariantKind::AttentionPagedBf16:
            return "AttentionPagedBf16";
        case McuCompiledVariantKind::AttentionPagedSplitQ4Full:
            return "AttentionPagedSplitQ4Full";
        case McuCompiledVariantKind::AttentionPagedReduceQ4FullQh1:
            return "AttentionPagedReduceQ4FullQh1";
        case McuCompiledVariantKind::AttentionPagedReduceQ4FullQh2:
            return "AttentionPagedReduceQ4FullQh2";
        case McuCompiledVariantKind::AttentionPagedReduceQ4FullBase:
            return "AttentionPagedReduceQ4FullBase";
        case McuCompiledVariantKind::AttentionPagedPrefillBf16:
            return "AttentionPagedPrefillBf16";
        case McuCompiledVariantKind::AttentionPagedSplitOther:
            return "AttentionPagedSplitOther";
        case McuCompiledVariantKind::AttentionPagedReduceOther:
            return "AttentionPagedReduceOther";
        default:
            return "unknown";
    }
}

Result<McuKernelCodeObject> mcu_kernel_code_object(McuCompiledVariantKind kind) {
    McuKernelCodeObject out{};
    const char* blob_name = nullptr;
    switch (kind) {
        case McuCompiledVariantKind::RmsNormBf16PfOnePlus:
            blob_name = "rmsnorm";
            out.symbol = kernel::kRmsNormBf16PfOnePlusSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
            break;
        case McuCompiledVariantKind::RmsNormF32PgDirect:
            blob_name = "rmsnorm";
            out.symbol = kernel::kRmsNormF32PgDirectSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
            break;
        case McuCompiledVariantKind::RmsNormBf16F32PgOnePlus:
            blob_name = "rmsnorm";
            out.symbol = kernel::kRmsNormBf16F32PgOnePlusSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
            break;
        case McuCompiledVariantKind::ElementwiseResidualBf16:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseResidualAddBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseSwigluBf16:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseSwigluBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseMulF32F32Bf16:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseMulF32F32Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseScaleF32:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseScaleF32Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseSiluBf16ToF32:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseSiluBf16ToF32Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseSiluF32ToBf16:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseSiluF32ToBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseSigmoidBf16ToF32:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseSigmoidBf16ToF32Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::ElementwiseSplitBf16:
            blob_name = "elementwise";
            out.symbol = kernel::kElementwiseSplitBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
            break;
        case McuCompiledVariantKind::Bf16ExactRows:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16Wmma:
            blob_name = "bf16";
            out.symbol = kernel::kBf16WmmaSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16Wmma;
            break;
        case McuCompiledVariantKind::Bf16WmmaWide:
            blob_name = "bf16";
            out.symbol = kernel::kBf16WmmaWideSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16Wmma;
            break;
        case McuCompiledVariantKind::Bf16SplitKWmma:
            blob_name = "bf16";
            out.symbol = kernel::kBf16SplitKWmmaSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16Wmma;
            break;
        case McuCompiledVariantKind::Bf16ExactRows2:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[1];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows3:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[2];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows4:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[3];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows5:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[4];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows6:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[5];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows7:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[6];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows8:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[7];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows9:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[8];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows10:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[9];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows11:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[10];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows12:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[11];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows13:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[12];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows14:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[13];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows15:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[14];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::Bf16ExactRows16:
            blob_name = "bf16";
            out.symbol = kernel::kBf16ExactRowsSymbols[15];
            out.kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
            break;
        case McuCompiledVariantKind::L2NormalizeBf16F32:
            blob_name = "l2_normalize";
            out.symbol = kernel::kL2NormalizeBf16F32Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeL2Normalize;
            break;
        case McuCompiledVariantKind::GdnConv1dBf16F32:
            blob_name = "gdn_conv1d";
            out.symbol = kernel::kGdnConv1dSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnConv1d;
            break;
        case McuCompiledVariantKind::GdnRecurrenceWmmaDecode1:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceWmmaDecode1Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceWmmaLossy:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceWmmaLossySymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceWmmaExact:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceWmmaExactSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceDecodeRowsExactR2:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceDecodeRowsExactR2Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceDecodeRowsExactR4:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceDecodeRowsExactR4Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceDecodeRowsExactR8:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceDecodeRowsExactR8Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceWmmaSerialLossy:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceWmmaSerialLossySymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnRecurrenceWmmaSerialExact:
            blob_name = "gdn_recurrence";
            out.symbol = kernel::kGdnRecurrenceWmmaSerialExactSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
            break;
        case McuCompiledVariantKind::GdnResetZero:
            blob_name = "gdn_reset";
            out.symbol = mcu::kGdnResetSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnReset;
            break;
        case McuCompiledVariantKind::RopeF32Bf16Pair:
            blob_name = "rope";
            out.symbol = kernel::kRopeF32Bf16PairSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeRope;
            break;
        case McuCompiledVariantKind::KvAppendBf16:
            blob_name = "kv_append";
            out.symbol = kernel::kKvAppendBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeKvAppend;
            break;
        case McuCompiledVariantKind::AttentionPagedBf16:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPaged;
            break;
        case McuCompiledVariantKind::AttentionPagedSplitQ4Full:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionSplitQ4FullSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedSplit;
            break;
        case McuCompiledVariantKind::AttentionPagedReduceQ4FullQh1:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionReduceQ4FullQh1Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedReduce;
            break;
        case McuCompiledVariantKind::AttentionPagedReduceQ4FullQh2:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionReduceQ4FullQh2Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedReduce;
            break;
        case McuCompiledVariantKind::AttentionPagedReduceQ4FullBase:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionReduceQ4FullBaseSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedReduce;
            break;
        case McuCompiledVariantKind::AttentionPagedPrefillBf16:
            blob_name = "attention_paged_prefill";
            out.symbol = kernel::kPagedAttentionPrefillBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPaged;
            break;
        case McuCompiledVariantKind::AttentionPagedSplitOther:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionSplitOtherSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedSplit;
            break;
        case McuCompiledVariantKind::AttentionPagedReduceOther:
            blob_name = "attention_paged";
            out.symbol = kernel::kPagedAttentionReduceOtherSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedReduce;
            break;
        case McuCompiledVariantKind::ActivationQuantizeE4m3K5120:
            blob_name = "activation_quantize";
            out.symbol = kernel::kActivationQuantizeE4m3K5120Symbol;
            out.kernarg_recipe =
                mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120;
            break;
        case McuCompiledVariantKind::ActivationQuantizeE4m3K6144:
            blob_name = "activation_quantize";
            out.symbol = kernel::kActivationQuantizeE4m3K6144Symbol;
            out.kernarg_recipe =
                mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120;
            break;
        case McuCompiledVariantKind::ActivationQuantizeE4m3K17408:
            blob_name = "activation_quantize";
            out.symbol = kernel::kActivationQuantizeE4m3K17408Symbol;
            out.kernarg_recipe =
                mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120;
            break;
        case McuCompiledVariantKind::Psq4Decode1Bf16U16:
            blob_name = "psq4";
            out.symbol = kernel::kPsq4Decode1Bf16U16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4Decode1Bf16U16;
            break;
        case McuCompiledVariantKind::Psq4Decode1Bf16U8:
            blob_name = "psq4";
            out.symbol = kernel::kPsq4Decode1Bf16U8Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4Decode1Bf16U8;
            break;
        case McuCompiledVariantKind::Psq8Decode1Bf16U8:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8Decode1Bf16U8Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq8Decode1Bf16U8;
            break;
        case McuCompiledVariantKind::Psq8RowBlock1Bf16:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8RowBlock1Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq8RowBlock2Bf16:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8RowBlock2Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq8RowBlock4Bf16:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8RowBlock4Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq8RowBlock8Bf16:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8RowBlock8Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq8Prefill2DN64K64:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8Prefill2DN64K64Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq8Prefill2DN128K64:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8Prefill2DN128K64Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq8Prefill2DN128K128:
            blob_name = "psq8";
            out.symbol = kernel::kPsq8Prefill2DN128K128Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq4RowBlock1Bf16:
            blob_name = "psq4";
            out.symbol = kernel::kPsq4RowBlock1Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq4RowBlock2Bf16:
            blob_name = "psq4";
            out.symbol = kernel::kPsq4RowBlock2Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq4RowBlock4Bf16:
            blob_name = "psq4";
            out.symbol = kernel::kPsq4RowBlock4Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::Psq4RowBlock8Bf16:
            blob_name = "psq4";
            out.symbol = kernel::kPsq4RowBlock8Bf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipePsq4MultiRowBf16;
            break;
        case McuCompiledVariantKind::VerifyAcceptPrefix:
            blob_name = "verify_accept";
            out.symbol = kernel::kVerifyAcceptPrefixSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeVerifyAcceptPrefix;
            break;
        case McuCompiledVariantKind::GdnSpecRestore:
            blob_name = "gdn_spec_restore";
            out.symbol = kernel::kGdnSpecRestoreSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeGdnSpecRestore;
            break;
        case McuCompiledVariantKind::SamplingArgmaxF32:
            blob_name = "sampling";
            out.symbol = kernel::kArgmaxF32Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeArgmaxF32;
            break;
        case McuCompiledVariantKind::EmbeddingBf16:
            blob_name = "embedding";
            out.symbol = kernel::kEmbeddingBf16Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeEmbeddingBf16;
            break;
        case McuCompiledVariantKind::EmbeddingPsq8:
            blob_name = "embedding";
            out.symbol = kernel::kEmbeddingPsq8Symbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeEmbeddingPsq8;
            break;
        case McuCompiledVariantKind::OutputGatherBf16:
            blob_name = "output_gather";
            out.symbol = kernel::kOutputGatherBf16McuSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeOutputGatherBf16;
            break;
        case McuCompiledVariantKind::VerifyAcceptBatch:
            blob_name = "verify_accept";
            out.symbol = kernel::kVerifyAcceptBatchSymbol;
            out.kernarg_recipe = mcu::kMcuKernargRecipeVerifyAcceptBatch;
            break;
        case McuCompiledVariantKind::GdnSpecRestoreFromCounts:
            blob_name = "gdn_spec_restore";
            out.symbol = kernel::kGdnSpecRestoreFromCountsSymbol;
            out.kernarg_recipe =
                mcu::kMcuKernargRecipeGdnSpecRestoreFromCounts;
            break;
        case McuCompiledVariantKind::CompletionMarker:
            return Status::unsupported(
                "completion marker is served by the supervisor probe image",
                __FILE__, __LINE__);
        default:
            return Status::unsupported("unknown mcu compiled variant kind",
                                       __FILE__, __LINE__);
    }

    out.blob = mcu::gpu_mcu_embedded_kernel(blob_name);
    if (out.blob.blob == nullptr || out.blob.size == 0u) {
        return Status::invalid_state("embedded kernel blob missing", __FILE__,
                                     __LINE__);
    }
    return out;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
