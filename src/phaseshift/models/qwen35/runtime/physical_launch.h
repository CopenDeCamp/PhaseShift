#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/models/qwen35/kernels/optimized/l2_normalize.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/runtime/elementwise_selector.h>
#include <phaseshift/models/qwen35/runtime/embedding_selector.h>
#include <phaseshift/runtime/program/program.h>

#include <cstdint>

namespace ps::qwen35::runtime {

struct HostExecutionContext;

::ps::kernel::RmsNormDataType rmsnorm_kd(::ps::runtime::ValueDType d);

enum class PhysicalResolveStatus : uint8_t {
    Applicable = 0,
    NotApplicable = 1,
    Skipped = 2,
};

enum class PhysicalActivationQuantizeKind : uint8_t {
    E4M3 = 0,
    I8Row = 1,
};

enum class PhysicalRmsNormVariant : uint8_t {
    Bf16PfOnePlus = 0,
    Unsupported = 1,
    F32PgDirect = 2,
    Bf16F32PgOnePlus = 3,
};

struct PhysicalRmsNormLaunch {
    uint64_t input = 0;
    uint64_t weight = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    float eps = 0.0f;
    ::ps::kernel::RmsNormDataType input_dtype =
        ::ps::kernel::RmsNormDataType::BF16;
    ::ps::kernel::RmsNormDataType weight_dtype =
        ::ps::kernel::RmsNormDataType::BF16;
    ::ps::kernel::RmsNormDataType output_dtype =
        ::ps::kernel::RmsNormDataType::BF16;
    ::ps::kernel::RmsNormWeightLayout weight_layout =
        ::ps::kernel::RmsNormWeightLayout::NONE;
    ::ps::kernel::RmsNormWeightMode weight_mode =
        ::ps::kernel::RmsNormWeightMode::ONE_PLUS;
    PhysicalRmsNormVariant variant = PhysicalRmsNormVariant::Unsupported;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

struct PhysicalActivationQuantizeLaunch {
    PhysicalActivationQuantizeKind kind = PhysicalActivationQuantizeKind::E4M3;
    uint64_t input = 0;
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

enum class PhysicalPsq4Variant : uint8_t {
    Decode1Bf16 = 0,
    Unsupported = 1,
    RowBlock1Bf16 = 2,
    RowBlock2Bf16 = 3,
    RowBlock4Bf16 = 4,
    RowBlock8Bf16 = 5,
    Prefill2DK64N64 = 6,
    Prefill2DK128N64 = 7,
    Prefill2DK64N128 = 8,
    Prefill2DK128N128 = 9,
};

struct PhysicalPsq4Launch {
    uint64_t weight_codes = 0;
    uint64_t weight_scales = 0;
    uint64_t activation_codes = 0;
    uint64_t activation_scales = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
    uint32_t activation_code_stride = 0;
    uint32_t activation_scale_stride = 0;
    uint32_t output_row_stride = 0;
    ::ps::kernel::Psq4GemmOutputDType output_dtype =
        ::ps::kernel::Psq4GemmOutputDType::BF16;
    ::ps::kernel::Psq4GemmConfigId config_id =
        ::ps::kernel::Psq4GemmConfigId::RowBlock1;
    PhysicalPsq4Variant variant = PhysicalPsq4Variant::Unsupported;
    bool use_decode1 = false;
    uint32_t unroll = 0;
    uint32_t grid = 0;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t workgroup = 0;
};

enum class PhysicalPsq8Variant : uint8_t {
    Decode1Bf16 = 0,
    Unsupported = 1,
    RowBlock1Bf16 = 2,
    RowBlock2Bf16 = 3,
    RowBlock4Bf16 = 4,
    RowBlock8Bf16 = 5,
    Prefill2DN64K64 = 6,
    Prefill2DN128K64 = 7,
    Prefill2DN128K128 = 8,
};

struct PhysicalPsq8Launch {
    uint64_t weight_codes = 0;
    uint64_t weight_scales = 0;
    uint64_t activation_codes = 0;
    uint64_t activation_scales = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
    uint32_t activation_code_stride = 0;
    uint32_t activation_scale_stride = 0;
    uint32_t output_row_stride = 0;
    ::ps::kernel::Psq8GemmOutputDType output_dtype =
        ::ps::kernel::Psq8GemmOutputDType::BF16;
    ::ps::kernel::Psq8GemmConfigId config_id =
        ::ps::kernel::Psq8GemmConfigId::RowBlock1;
    PhysicalPsq8Variant variant = PhysicalPsq8Variant::Unsupported;
    bool use_decode1 = false;
    uint32_t unroll = 0;
    uint32_t grid = 0;
    uint32_t grid_y = 1;
    uint32_t workgroup = 0;
};

struct PhysicalElementwiseLaunch {
    ElementwiseProfile profile = ElementwiseProfile::Count;
    uint64_t in0 = 0;
    uint64_t in1 = 0;
    uint64_t out0 = 0;
    uint64_t out1 = 0;
    uint32_t in0_stride = 0;
    uint32_t in1_stride = 0;
    uint32_t out0_stride = 0;
    uint32_t out1_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t features_b = 0;
    uint32_t aux = 0;
    float scalar_a = 0.0f;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t workgroup = 0;
};

enum class PhysicalBf16Variant : uint8_t {
    ExactRows = 0,
    WmmaKPartition = 1,
    WmmaWide = 2,
    Wmma = 3,
    Unsupported = 255,
};

struct PhysicalBf16Launch {
    uint64_t weight = 0;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    ::ps::kernel::Bf16GemmOutputDType output_dtype =
        ::ps::kernel::Bf16GemmOutputDType::BF16;
    ::ps::kernel::Bf16GemmConfig config{};
    PhysicalBf16Variant variant = PhysicalBf16Variant::Unsupported;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t workgroup = 0;
    bool output_rows_domain = false;
};

struct PhysicalL2NormalizeLaunch {
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    float eps = 0.0f;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t workgroup = 0;
};

enum class PhysicalGdnRecurrenceVariant : uint8_t {
    DecodeRowsExact = 0,
    Decode1Serial = 1,
    WmmaSerial = 2,
    WmmaDecode1 = 3,
    Wmma = 4,
    F32 = 5,
    Unsupported = 255,
};

struct PhysicalGdnConv1dLaunch {
    ::ps::kernel::GdnConv1dArgs args{};
    uint32_t row_tile = 0;
    uint32_t tiles = 1;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t workgroup = 0;
};

struct PhysicalGdnRecurrenceLaunch {
    ::ps::kernel::GdnRecurrenceArgs kernel{};
    bool lossy = false;
    uint32_t rows = 0;
    uint32_t num_requests = 0;
    uint32_t max_request_rows = 0;
    PhysicalGdnRecurrenceVariant variant = PhysicalGdnRecurrenceVariant::Unsupported;
    uint32_t grid_x = 0;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t workgroup = 0;
};

struct PhysicalEmbeddingLaunch {
    EmbeddingStorage storage = EmbeddingStorage::Bf16;
    uint64_t table = 0;
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t k_padded = 0;
    uint64_t token_ids = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
};

Result<PhysicalResolveStatus> resolve_embedding_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalEmbeddingLaunch& out);

Result<PhysicalResolveStatus> resolve_rmsnorm_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalRmsNormLaunch& out);

Result<PhysicalResolveStatus> resolve_activation_quantize_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalActivationQuantizeLaunch& out);

Result<PhysicalResolveStatus> resolve_psq4_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    bool allow_unguarded_prefill2d,
    PhysicalPsq4Launch& out);

Result<PhysicalResolveStatus> resolve_psq8_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    bool allow_unguarded_prefill2d,
    PhysicalPsq8Launch& out);

Result<PhysicalResolveStatus> resolve_elementwise_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalElementwiseLaunch& out);

Result<PhysicalResolveStatus> resolve_bf16_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalBf16Launch& out);

Result<PhysicalResolveStatus> resolve_l2_normalize_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalL2NormalizeLaunch& out);

Result<PhysicalResolveStatus> resolve_gdn_conv1d_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalGdnConv1dLaunch& out);

Result<PhysicalResolveStatus> resolve_gdn_recurrence_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalGdnRecurrenceLaunch& out);

enum class PhysicalRopeVariant : uint8_t {
    F32Bf16Pair = 0,
    F32Bf16Generic = 1,
    Unsupported = 255,
};

struct PhysicalRopeLaunch {
    PhysicalRopeVariant variant = PhysicalRopeVariant::Unsupported;
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t positions = 0;
    uint64_t inv_freq = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    float theta = 0.0f;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

Result<PhysicalResolveStatus> resolve_rope_physical(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalRopeLaunch& out);

hipError_t launch_rope_physical(const PhysicalRopeLaunch& d, hipStream_t stream);

enum class PhysicalKvAppendVariant : uint8_t {
    Bf16 = 0,
    Fp8E4M3 = 1,
    Psq4W32 = 3,
    Psq8W32 = 4,
    Unsupported = 255,
};

struct PhysicalKvAppendLaunch {
    PhysicalKvAppendVariant variant = PhysicalKvAppendVariant::Unsupported;
    ::ps::kernel::KvAppendCommonArgs common{};
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
    uint64_t k_scale_pool = 0;
    uint64_t v_scale_pool = 0;
    uint64_t k_extra_pool = 0;
    uint64_t v_extra_pool = 0;
    uint32_t blocks_per_head = 0;
    ::ps::qwen35::Psq4ScaleEstimator psq4_estimator =
        ::ps::qwen35::Psq4ScaleEstimator::Lsq1;
    ::ps::qwen35::Psq8ScaleEstimator psq8_estimator =
        ::ps::qwen35::Psq8ScaleEstimator::Lsq1;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
};

Result<PhysicalResolveStatus> resolve_kv_append_physical(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalKvAppendLaunch& out);

hipError_t launch_kv_append_physical(const PhysicalKvAppendLaunch& d,
                                     hipStream_t stream);

enum class PhysicalPagedAttentionVariant : uint8_t {
    Bf16FullHead = 0,
    Bf16PartialHead = 1,
    Bf16SplitQ4Full = 2,
    Bf16SplitOther = 3,
    Bf16ReduceQ4FullQh1 = 4,
    Bf16ReduceQ4FullQh2 = 5,
    Bf16ReduceQ4FullBase = 6,
    Bf16ReduceOther = 7,
    Fp8E4M3 = 8,
    Psq4W32 = 10,
    Psq4Split = 11,
    Psq8Split = 12,
    Psq8W32 = 13,
    Bf16Prefill = 14,
    Fp8E4M3Prefill = 15,
    Psq4Prefill = 16,
    Psq8Prefill = 17,
    Unsupported = 255,
};

struct PhysicalPagedAttentionKernel {
    PhysicalPagedAttentionVariant variant = PhysicalPagedAttentionVariant::Unsupported;
    ::ps::kernel::PagedAttentionCommonArgs common{};
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
    uint64_t k_scale_pool = 0;
    uint64_t v_scale_pool = 0;
    uint64_t k_extra_pool = 0;
    uint64_t v_extra_pool = 0;
    uint64_t partials = 0;
    uint32_t splits = 0;
    uint32_t q_per_kv = 0;
    uint32_t qh_per_wg = 0;
    uint32_t blocks_per_head = 0;
    uint32_t grid = 0;
    uint32_t workgroup = 0;
    uint32_t row_begin = 0;
};

inline constexpr uint32_t kPhysicalPagedAttentionMaxPrefillRegions = 16u;

inline constexpr uint32_t kPhysicalPagedAttentionMaxKernels =
    2u + kPhysicalPagedAttentionMaxPrefillRegions;

struct PhysicalPagedAttentionPlan {
    bool optimized = false;
    bool use_prefill = false;
    ::ps::qwen35::KVCacheDType kv_dtype = ::ps::qwen35::KVCacheDType::BF16;
    uint32_t splits = 1u;
    uint32_t physical_count = 0u;
    PhysicalPagedAttentionKernel kernels[kPhysicalPagedAttentionMaxKernels];
};

Result<PhysicalResolveStatus> resolve_paged_attention_physical(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalPagedAttentionPlan& out);

Result<PhysicalResolveStatus> resolve_paged_attention_physical_static_plan(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    const HostExecutionContext& ctx,
    PhysicalPagedAttentionPlan& out);

hipError_t launch_paged_attention_physical(const PhysicalPagedAttentionPlan& plan,
                                           hipStream_t stream);

}  // namespace ps::qwen35::runtime
