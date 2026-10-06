#include <phaseshift/models/qwen35/runtime/physical_launch.h>

#include <phaseshift/models/qwen35/runtime/decode_perf_stats.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/rmsnorm_selector.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

namespace ps::qwen35::runtime {

using ::ps::runtime::DispatchBinding;
using ::ps::runtime::Int8ActivationWorkspaceLayout;
using ::ps::runtime::KernelId;
using ::ps::runtime::Program;
using ::ps::runtime::ValueDType;

namespace {

uint32_t rows_for(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ::ps::runtime::ValueRowDomain::OUTPUT_ROWS)
        return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

}  // namespace

::ps::kernel::RmsNormDataType rmsnorm_kd(ValueDType d) {
    return d == ValueDType::F32 ? ::ps::kernel::RmsNormDataType::F32
                                : ::ps::kernel::RmsNormDataType::BF16;
}

Result<PhysicalResolveStatus> resolve_rmsnorm_physical(
    const Program& program,
    const DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalRmsNormLaunch& out) {
    out = PhysicalRmsNormLaunch{};
    if (b.input_count != 1 || b.output_count != 1)
        return Status::invalid_state("rmsnorm optimized: io count != 1", __FILE__,
                                     __LINE__);
    auto in_res =
        resolve_host_value(program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    auto out_res = resolve_host_value(program,
                                      ::ps::runtime::ValueId{b.output_slots[0]},
                                      ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue dst = out_res.value();
    const uint32_t F = in.feature_count;
    if (F == 0 || in.feature_count != dst.feature_count)
        return Status::invalid_state("rmsnorm optimized: features mismatch",
                                     __FILE__, __LINE__);
    const uint32_t G = (b.group_size != 0 && b.group_size <= F) ? b.group_size : F;
    if (F % G != 0)
        return Status::invalid_state("rmsnorm optimized: F % G != 0", __FILE__,
                                     __LINE__);
    if (in.dtype != ValueDType::BF16 && in.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    if (dst.dtype != ValueDType::BF16 && dst.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    ::ps::kernel::RmsNormWeightLayout wlayout =
        ::ps::kernel::RmsNormWeightLayout::NONE;
    const void* wptr = nullptr;
    ValueDType wdt = ValueDType::BF16;
    if (b.parameter_index0 != ::ps::runtime::kNoParameter) {
        if (ctx.host_parameters == nullptr ||
            b.parameter_index0 >= ctx.host_parameter_count)
            return PhysicalResolveStatus::NotApplicable;
        const auto slot = ctx.host_parameters[b.parameter_index0];
        if (slot.dtype != ValueDType::BF16)
            return PhysicalResolveStatus::NotApplicable;
        wptr = slot.device_ptr;
        wdt = slot.dtype;
        if (slot.elements == F)
            wlayout = ::ps::kernel::RmsNormWeightLayout::PER_FEATURE;
        else if (slot.elements == G)
            wlayout = ::ps::kernel::RmsNormWeightLayout::PER_GROUP;
        else
            return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(in, ctx);
    if (rows == 0) return PhysicalResolveStatus::Skipped;
    RmsNormSelectorInput si;
    si.input_dtype = in.dtype;
    si.output_dtype = dst.dtype;
    si.weight_dtype = wdt;
    si.weight_layout = wlayout == ::ps::kernel::RmsNormWeightLayout::PER_FEATURE
                           ? RmsNormSelectorWeightLayout::PerFeature
                       : wlayout == ::ps::kernel::RmsNormWeightLayout::PER_GROUP
                           ? RmsNormSelectorWeightLayout::PerGroup
                           : RmsNormSelectorWeightLayout::None;
    si.weight_mode = b.flags;
    si.features = F;
    si.group_size = G;
    si.rows = rows;
    if (select_rmsnorm_implementation(si) != RmsNormImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.weight = reinterpret_cast<uint64_t>(wptr);
    out.output = reinterpret_cast<uint64_t>(dst.ptr);
    out.rows = rows;
    out.features = F;
    out.group_size = G;
    out.input_row_stride = in.row_stride;
    out.output_row_stride = dst.row_stride;
    out.eps = b.scalar_a;
    out.input_dtype = rmsnorm_kd(in.dtype);
    out.weight_dtype = rmsnorm_kd(wdt);
    out.output_dtype = rmsnorm_kd(dst.dtype);
    out.weight_layout = wlayout;
    out.weight_mode = b.flags == 0u
                          ? ::ps::kernel::RmsNormWeightMode::ONE_PLUS
                          : ::ps::kernel::RmsNormWeightMode::DIRECT;
    const bool bf16_pf_one_plus =
        in.dtype == ValueDType::BF16 && dst.dtype == ValueDType::BF16 &&
        wdt == ValueDType::BF16 &&
        wlayout == ::ps::kernel::RmsNormWeightLayout::PER_FEATURE &&
        out.weight_mode == ::ps::kernel::RmsNormWeightMode::ONE_PLUS;
    const bool f32_pg_direct =
        in.dtype == ValueDType::F32 && dst.dtype == ValueDType::F32 &&
        wdt == ValueDType::BF16 &&
        wlayout == ::ps::kernel::RmsNormWeightLayout::PER_GROUP &&
        out.weight_mode == ::ps::kernel::RmsNormWeightMode::DIRECT;
    const bool bf16_f32_pg_one_plus =
        in.dtype == ValueDType::BF16 && dst.dtype == ValueDType::F32 &&
        wdt == ValueDType::BF16 &&
        wlayout == ::ps::kernel::RmsNormWeightLayout::PER_GROUP &&
        out.weight_mode == ::ps::kernel::RmsNormWeightMode::ONE_PLUS;
    out.variant = bf16_pf_one_plus
                      ? PhysicalRmsNormVariant::Bf16PfOnePlus
                      : (f32_pg_direct
                             ? PhysicalRmsNormVariant::F32PgDirect
                             : (bf16_f32_pg_one_plus
                                    ? PhysicalRmsNormVariant::Bf16F32PgOnePlus
                                    : PhysicalRmsNormVariant::Unsupported));
    if (out.variant != PhysicalRmsNormVariant::Unsupported) {
        out.grid = ::ps::kernel::rmsnorm_bf16_pf_oneplus_grid(rows, F, G);
        out.workgroup = ::ps::kernel::rmsnorm_bf16_pf_oneplus_block_size(G);
    }
    return PhysicalResolveStatus::Applicable;
}

Result<PhysicalResolveStatus> resolve_activation_quantize_physical(
    const Program& program,
    const DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalActivationQuantizeLaunch& out) {
    out = PhysicalActivationQuantizeLaunch{};
    if (b.input_count != 1)
        return Status::invalid_state("act-quant optimized: input_count != 1",
                                     __FILE__, __LINE__);
    if (b.group_size != 32)
        return Status::invalid_state("act-quant optimized: group_size != 32",
                                     __FILE__, __LINE__);
    if (b.workspace_slot >= program.workspace.ranges.size())
        return Status::invalid_state("act-quant optimized: workspace_slot invalid",
                                     __FILE__, __LINE__);
    if (b.activation_max_rows == 0)
        return Status::invalid_state("act-quant optimized: activation_max_rows == 0",
                                     __FILE__, __LINE__);
    auto in_res =
        resolve_host_value(program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::BF16)
        return Status::invalid_state("act-quant optimized: input dtype != BF16",
                                     __FILE__, __LINE__);
    if (in.feature_count == 0)
        return Status::invalid_state("act-quant optimized: input feature_count == 0",
                                     __FILE__, __LINE__);
    if (b.activation_kp < in.feature_count || (b.activation_kp & 31u) != 0u)
        return Status::invalid_state("act-quant optimized: activation_kp invalid",
                                     __FILE__, __LINE__);
    const uint32_t rows = rows_for(in, ctx);
    if (rows == 0) return PhysicalResolveStatus::NotApplicable;
    if (rows > b.activation_max_rows)
        return Status::invalid_state("act-quant optimized: rows out of range",
                                     __FILE__, __LINE__);
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(in.feature_count,
                                            b.activation_max_rows);
    if (layout.k_padded != b.activation_kp ||
        layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return Status::invalid_state("act-quant optimized: workspace layout mismatch",
                                     __FILE__, __LINE__);
    if (ctx.workspace == nullptr)
        return Status::invalid_state("act-quant optimized: workspace not bound",
                                     __FILE__, __LINE__);
    const uint64_t range_off = program.workspace.ranges[b.workspace_slot].offset;
    uint8_t* range_base = ctx.workspace + range_off;

    out.kind = b.kernel_id == KernelId::ACTIVATION_QUANTIZE_W4A8
                   ? PhysicalActivationQuantizeKind::E4M3
                   : PhysicalActivationQuantizeKind::I8Row;
    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.codes = reinterpret_cast<uint64_t>(range_base + layout.codes_offset);
    out.scales = reinterpret_cast<uint64_t>(range_base + layout.scales_offset);
    out.rows = rows;
    out.k = in.feature_count;
    out.k_padded = b.activation_kp;
    out.input_row_stride = in.row_stride;
    out.code_row_stride = layout.code_row_stride_bytes;
    out.scale_row_stride = layout.scale_row_stride_bytes;
    out.grid = rows;
    out.workgroup = 256u;
    return PhysicalResolveStatus::Applicable;
}

Result<PhysicalResolveStatus> resolve_psq4_physical(
    const Program& program,
    const DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalPsq4Launch& out) {
    out = PhysicalPsq4Launch{};
    if (b.output_count != 1 || b.input_count != 0)
        return Status::invalid_state("psq4 linear: bad io count", __FILE__, __LINE__);
    if (b.compute_spec != 3u)
        return Status::invalid_state("psq4 linear: compute_spec != 3", __FILE__,
                                     __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("psq4 linear: weight_index invalid", __FILE__,
                                     __LINE__);
    if (b.workspace_slot >= program.workspace.ranges.size())
        return Status::invalid_state("psq4 linear: workspace_slot invalid", __FILE__,
                                     __LINE__);
    const ::ps::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 3u)
        return Status::invalid_state("psq4 linear: encoding != Psq4", __FILE__,
                                     __LINE__);
    if (e.codes == nullptr || e.scales == nullptr)
        return Status::invalid_state("psq4 linear: weight view missing", __FILE__,
                                     __LINE__);
    if (e.rows == 0 || e.cols == 0)
        return Status::invalid_state("psq4 linear: weight dims == 0", __FILE__,
                                     __LINE__);
    const uint32_t wkp = e.k_padded;
    if (wkp < e.cols || (wkp & 31u) != 0u)
        return Status::invalid_state("psq4 linear: weight k_padded invalid",
                                     __FILE__, __LINE__);
    if (e.weight_scale_group != 32u)
        return Status::invalid_state("psq4 linear: weight_scale_group != 32",
                                     __FILE__, __LINE__);
    const uint32_t sstride = e.storage_scale_stride_bytes;
    if (sstride < wkp || (sstride & 15u) != 0u)
        return Status::invalid_state("psq4 linear: storage_scale_stride invalid",
                                     __FILE__, __LINE__);
    const uint32_t akp = b.activation_kp;
    if (akp < wkp || (akp & 31u) != 0u)
        return Status::invalid_state("psq4 linear: activation_kp invalid", __FILE__,
                                     __LINE__);
    if (b.activation_code_stride < akp || (b.activation_code_stride & 15u) != 0u)
        return Status::invalid_state("psq4 linear: activation_code_stride invalid",
                                     __FILE__, __LINE__);
    if (b.activation_scale_stride < (akp >> 5) * 4u ||
        (b.activation_scale_stride & 15u) != 0u)
        return Status::invalid_state("psq4 linear: activation_scale_stride invalid",
                                     __FILE__, __LINE__);
    if (b.activation_max_rows == 0)
        return Status::invalid_state("psq4 linear: activation_max_rows == 0",
                                     __FILE__, __LINE__);
    if (ctx.workspace == nullptr)
        return Status::invalid_state("psq4 linear: workspace not bound", __FILE__,
                                     __LINE__);
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(e.cols, b.activation_max_rows);
    if (layout.k_padded != akp ||
        layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return Status::invalid_state("psq4 linear: workspace layout mismatch",
                                     __FILE__, __LINE__);
    const uint32_t out_vid = b.output_slots[0];
    auto out_res =
        resolve_host_value(program, ::ps::runtime::ValueId{out_vid}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue ov = out_res.value();
    if (ov.feature_count != e.rows)
        return Status::invalid_state(
            "psq4 linear: output features != weight rows", __FILE__, __LINE__);
    const bool out_bf16 = ov.dtype == ValueDType::BF16;
    const bool out_f32 = ov.dtype == ValueDType::F32;
    if (!out_bf16 && !out_f32)
        return Status::invalid_state("psq4 linear: output dtype not BF16/F32",
                                     __FILE__, __LINE__);
    const uint32_t rows = rows_for(ov, ctx);
    if (rows == 0) return PhysicalResolveStatus::Skipped;
    if (rows > b.activation_max_rows)
        return Status::invalid_state("psq4 linear: rows out of range", __FILE__,
                                     __LINE__);
    const Psq4GemmSelectorInput lin{rows, e.rows, e.cols, wkp};
    const auto config = select_psq4_gemm_config(lin);
    if (!config) {
        linear_record_fallback(LinearFamilyRef::Fp4,
                               FallbackReason::SelectorRejected);
        return PhysicalResolveStatus::NotApplicable;
    }
    if (!e.preshuffled)
        return Status::invalid_state(
            "psq4 linear: weight not preshuffled to native layout", __FILE__,
            __LINE__);
    const uint64_t range_off = program.workspace.ranges[b.workspace_slot].offset;
    uint8_t* range_base = ctx.workspace + range_off;

    out.weight_codes = reinterpret_cast<uint64_t>(e.codes);
    out.weight_scales = reinterpret_cast<uint64_t>(e.scales);
    out.activation_codes =
        reinterpret_cast<uint64_t>(range_base + layout.codes_offset);
    out.activation_scales =
        reinterpret_cast<uint64_t>(range_base + layout.scales_offset);
    out.output = reinterpret_cast<uint64_t>(ov.ptr);
    out.rows = rows;
    out.out_features = e.rows;
    out.k_padded = wkp;
    out.weight_scale_stride = sstride;
    out.activation_code_stride = b.activation_code_stride;
    out.activation_scale_stride = b.activation_scale_stride;
    out.output_row_stride = ov.row_stride;
    out.output_dtype = out_bf16 ? ::ps::kernel::Psq4GemmOutputDType::BF16
                                : ::ps::kernel::Psq4GemmOutputDType::F32;
    out.config_id = config->id;
    out.use_decode1 =
        out.config_id == ::ps::kernel::Psq4GemmConfigId::RowBlock1 &&
        ::ps::kernel::psq4_decode1_supported(rows, e.rows, wkp);
    out.unroll = out.use_decode1 ? ::ps::kernel::psq4_decode1_unroll(e.rows, wkp)
                                 : 0u;
    out.grid = out.use_decode1 ? ::ps::kernel::psq4_decode1_grid(e.rows) : 0u;
    out.workgroup = out.use_decode1 ? ::ps::kernel::psq4_decode1_block_size() : 0u;
    if (out.use_decode1 && out_bf16) {
        out.variant = PhysicalPsq4Variant::Decode1Bf16;
    } else if (out_bf16) {
        const uint32_t out_tiles = (e.rows + 15u) / 16u;
        switch (out.config_id) {
            case ::ps::kernel::Psq4GemmConfigId::RowBlock1:
                out.variant = PhysicalPsq4Variant::RowBlock1Bf16;
                out.grid = out_tiles;
                out.grid_y = (rows + 15u) / 16u;
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq4GemmConfigId::RowBlock2:
                out.variant = PhysicalPsq4Variant::RowBlock2Bf16;
                out.grid = out_tiles;
                out.grid_y = (rows + 31u) / 32u;
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq4GemmConfigId::RowBlock4:
                out.variant = PhysicalPsq4Variant::RowBlock4Bf16;
                out.grid = out_tiles;
                out.grid_y = (rows + 63u) / 64u;
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq4GemmConfigId::RowBlock8:
                out.variant = PhysicalPsq4Variant::RowBlock8Bf16;
                out.grid = out_tiles;
                out.grid_y = (rows + 127u) / 128u;
                out.workgroup = 32u;
                break;
            default:
                out.variant = PhysicalPsq4Variant::Unsupported;
                break;
        }
    } else {
        out.variant = PhysicalPsq4Variant::Unsupported;
    }
    return PhysicalResolveStatus::Applicable;
}

Result<PhysicalResolveStatus> resolve_psq8_physical(
    const Program& program,
    const DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalPsq8Launch& out) {
    out = PhysicalPsq8Launch{};
    if (b.output_count != 1 || b.input_count != 0)
        return Status::invalid_state("psq8 linear: bad io count", __FILE__, __LINE__);
    if (b.compute_spec != 4u)
        return Status::invalid_state("psq8 linear: compute_spec != 4", __FILE__,
                                     __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("psq8 linear: weight_index invalid", __FILE__,
                                     __LINE__);
    if (b.workspace_slot >= program.workspace.ranges.size())
        return Status::invalid_state("psq8 linear: workspace_slot invalid", __FILE__,
                                     __LINE__);
    const ::ps::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 4u)
        return Status::invalid_state("psq8 linear: encoding != Psq8", __FILE__,
                                     __LINE__);
    if (e.codes == nullptr || e.scales == nullptr)
        return Status::invalid_state("psq8 linear: weight view missing", __FILE__,
                                     __LINE__);
    if (e.rows == 0 || e.cols == 0)
        return Status::invalid_state("psq8 linear: weight dims == 0", __FILE__,
                                     __LINE__);
    const uint32_t wkp = e.k_padded;
    if (wkp < e.cols || (wkp & 31u) != 0u)
        return Status::invalid_state("psq8 linear: weight k_padded invalid",
                                     __FILE__, __LINE__);
    if (e.weight_scale_group != 32u)
        return Status::invalid_state("psq8 linear: weight_scale_group != 32",
                                     __FILE__, __LINE__);
    const uint32_t sstride = e.storage_scale_stride_bytes;
    if (sstride < wkp || (sstride & 15u) != 0u)
        return Status::invalid_state("psq8 linear: storage_scale_stride invalid",
                                     __FILE__, __LINE__);
    const uint32_t akp = b.activation_kp;
    if (akp < wkp || (akp & 31u) != 0u)
        return Status::invalid_state("psq8 linear: activation_kp invalid", __FILE__,
                                     __LINE__);
    if (b.activation_code_stride < akp || (b.activation_code_stride & 15u) != 0u)
        return Status::invalid_state("psq8 linear: activation_code_stride invalid",
                                     __FILE__, __LINE__);
    if (b.activation_scale_stride < (akp >> 5) * 4u ||
        (b.activation_scale_stride & 15u) != 0u)
        return Status::invalid_state("psq8 linear: activation_scale_stride invalid",
                                     __FILE__, __LINE__);
    if (b.activation_max_rows == 0)
        return Status::invalid_state("psq8 linear: activation_max_rows == 0",
                                     __FILE__, __LINE__);
    if (ctx.workspace == nullptr)
        return Status::invalid_state("psq8 linear: workspace not bound", __FILE__,
                                     __LINE__);
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(e.cols, b.activation_max_rows);
    if (layout.k_padded != akp ||
        layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return Status::invalid_state("psq8 linear: workspace layout mismatch",
                                     __FILE__, __LINE__);
    const uint32_t out_vid = b.output_slots[0];
    auto out_res =
        resolve_host_value(program, ::ps::runtime::ValueId{out_vid}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue ov = out_res.value();
    if (ov.feature_count != e.rows)
        return Status::invalid_state(
            "psq8 linear: output features != weight rows", __FILE__, __LINE__);
    const bool out_bf16 = ov.dtype == ValueDType::BF16;
    const bool out_f32 = ov.dtype == ValueDType::F32;
    if (!out_bf16 && !out_f32)
        return Status::invalid_state("psq8 linear: output dtype not BF16/F32",
                                     __FILE__, __LINE__);
    const uint32_t rows = rows_for(ov, ctx);
    if (rows == 0) return PhysicalResolveStatus::Skipped;
    if (rows > b.activation_max_rows)
        return Status::invalid_state("psq8 linear: rows out of range", __FILE__,
                                     __LINE__);
    const Psq8GemmSelectorInput lin{rows, e.rows, e.cols, wkp};
    const auto config = select_psq8_gemm_config(lin);
    if (!config) {
        linear_record_fallback(LinearFamilyRef::Fp8,
                               FallbackReason::SelectorRejected);
        return PhysicalResolveStatus::NotApplicable;
    }
    if (!e.preshuffled)
        return Status::invalid_state(
            "psq8 linear: weight not preshuffled to native layout", __FILE__,
            __LINE__);
    const uint64_t range_off = program.workspace.ranges[b.workspace_slot].offset;
    uint8_t* range_base = ctx.workspace + range_off;

    out.weight_codes = reinterpret_cast<uint64_t>(e.codes);
    out.weight_scales = reinterpret_cast<uint64_t>(e.scales);
    out.activation_codes =
        reinterpret_cast<uint64_t>(range_base + layout.codes_offset);
    out.activation_scales =
        reinterpret_cast<uint64_t>(range_base + layout.scales_offset);
    out.output = reinterpret_cast<uint64_t>(ov.ptr);
    out.rows = rows;
    out.out_features = e.rows;
    out.k_padded = wkp;
    out.weight_scale_stride = sstride;
    out.activation_code_stride = b.activation_code_stride;
    out.activation_scale_stride = b.activation_scale_stride;
    out.output_row_stride = ov.row_stride;
    out.output_dtype = out_bf16 ? ::ps::kernel::Psq8GemmOutputDType::BF16
                                : ::ps::kernel::Psq8GemmOutputDType::F32;
    out.config_id = config->id;
    out.use_decode1 =
        out.config_id == ::ps::kernel::Psq8GemmConfigId::RowBlock1 &&
        ::ps::kernel::psq8_decode1_supported(rows, e.rows, wkp);
    out.unroll = out.use_decode1 ? ::ps::kernel::psq8_decode1_unroll(e.rows, wkp)
                                 : 0u;
    if (out.use_decode1 && out_bf16) {
        out.variant = PhysicalPsq8Variant::Decode1Bf16;
        out.grid = ::ps::kernel::psq8_decode1_grid(e.rows);
        out.workgroup = ::ps::kernel::psq8_decode1_block_size();
    } else {
        const uint32_t out_tiles = ::ps::kernel::psq8_rowblock_grid_x(e.rows);
        switch (out.config_id) {
            case ::ps::kernel::Psq8GemmConfigId::RowBlock1:
                out.variant = PhysicalPsq8Variant::RowBlock1Bf16;
                out.grid = out_tiles;
                out.grid_y = ::ps::kernel::psq8_rowblock_grid_y(rows, 1u);
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq8GemmConfigId::RowBlock2:
                out.variant = PhysicalPsq8Variant::RowBlock2Bf16;
                out.grid = out_tiles;
                out.grid_y = ::ps::kernel::psq8_rowblock_grid_y(rows, 2u);
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq8GemmConfigId::RowBlock4:
                out.variant = PhysicalPsq8Variant::RowBlock4Bf16;
                out.grid = out_tiles;
                out.grid_y = ::ps::kernel::psq8_rowblock_grid_y(rows, 4u);
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq8GemmConfigId::RowBlock8:
                out.variant = PhysicalPsq8Variant::RowBlock8Bf16;
                out.grid = out_tiles;
                out.grid_y = ::ps::kernel::psq8_rowblock_grid_y(rows, 8u);
                out.workgroup = 32u;
                break;
            case ::ps::kernel::Psq8GemmConfigId::Prefill2D:
                out.variant = PhysicalPsq8Variant::Prefill2DN64K64;
                out.grid = ::ps::kernel::psq8_prefill2d_grid_x(
                    e.rows, ::ps::kernel::kPsq8GemmPrefill2dBlock64);
                out.grid_y = ::ps::kernel::psq8_prefill2d_grid_y(rows);
                out.workgroup = ::ps::kernel::kPsq8Prefill2DThreads;
                break;
            case ::ps::kernel::Psq8GemmConfigId::Prefill2D_K64N128:
                out.variant = PhysicalPsq8Variant::Prefill2DN128K64;
                out.grid = ::ps::kernel::psq8_prefill2d_grid_x(
                    e.rows, ::ps::kernel::kPsq8GemmPrefill2dBlock128);
                out.grid_y = ::ps::kernel::psq8_prefill2d_grid_y(rows);
                out.workgroup = ::ps::kernel::kPsq8Prefill2DThreads;
                break;
            case ::ps::kernel::Psq8GemmConfigId::Prefill2D_K128N128:
                out.variant = PhysicalPsq8Variant::Prefill2DN128K128;
                out.grid = ::ps::kernel::psq8_prefill2d_grid_x(
                    e.rows, ::ps::kernel::kPsq8GemmPrefill2dBlock128);
                out.grid_y = ::ps::kernel::psq8_prefill2d_grid_y(rows);
                out.workgroup = ::ps::kernel::kPsq8Prefill2DThreads;
                break;
            default:
                out.variant = PhysicalPsq8Variant::Unsupported;
                break;
        }
    }
    return PhysicalResolveStatus::Applicable;
}

}  // namespace ps::qwen35::runtime
