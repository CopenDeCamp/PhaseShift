#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/gpu_mcu/execution/micro_fsm.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <phaseshift/runtime/program/program.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace gpu_mcu_w4a8 {

namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;

inline constexpr uint32_t kRmsOutOffset = 0u;
inline constexpr uint32_t kRmsOutBytes = 5120u * 2u;
inline constexpr uint32_t kActOffset = kRmsOutOffset + kRmsOutBytes;
inline constexpr uint32_t kActBytes = 82560u;
inline constexpr uint32_t kWorkspaceBytes = kActOffset + kActBytes;

inline constexpr uint32_t kVInput = 0u;
inline constexpr uint32_t kVRmsOut = 1u;
inline constexpr uint32_t kVPsq4Out = 2u;

inline const rt::Int8ActivationWorkspaceLayout& act_layout() {
    static const rt::Int8ActivationWorkspaceLayout l =
        rt::Int8ActivationWorkspaceLayout::make(5120u, 1u);
    return l;
}

inline void build_w4a8_program(rt::Program& program) {
    program = rt::Program{};
    program.row_bucket = rt::RowBucket::R16;
    program.execution_class = rt::ExecutionClass::DECODE;
    program.workspace.total_bytes = kWorkspaceBytes;
    program.workspace.ranges.push_back(
        rt::WorkspaceRange{kActOffset, kActBytes, 0u, 0u});
    program.weight_slot_count = 1u;
    program.parameter_slot_count = 1u;

    rt::ValueBinding in{};
    in.logical_value = rt::ValueId{kVInput};
    in.storage = rt::ValueStorage::EXTERNAL_INPUT;
    in.dtype = rt::ValueDType::BF16;
    in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    in.slot = 0u;
    in.feature_count = 5120u;
    in.row_stride = 5120u;
    program.values.push_back(in);

    rt::ValueBinding rms_out{};
    rms_out.logical_value = rt::ValueId{kVRmsOut};
    rms_out.storage = rt::ValueStorage::WORKSPACE;
    rms_out.dtype = rt::ValueDType::BF16;
    rms_out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    rms_out.slot = 0u;
    rms_out.feature_count = 5120u;
    rms_out.row_stride = 5120u;
    rms_out.offset = kRmsOutOffset;
    rms_out.bytes = kRmsOutBytes;
    program.values.push_back(rms_out);

    rt::ValueBinding out{};
    out.logical_value = rt::ValueId{kVPsq4Out};
    out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    out.dtype = rt::ValueDType::BF16;
    out.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
    out.slot = 0u;
    out.feature_count = 1024u;
    out.row_stride = 1024u;
    program.values.push_back(out);

    const auto& layout = act_layout();

    rt::DispatchBinding rms{};
    rms.kernel_id = rt::KernelId::RMS_NORM;
    rms.input_count = 1u;
    rms.input_slots[0] = kVInput;
    rms.output_count = 1u;
    rms.output_slots[0] = kVRmsOut;
    rms.parameter_index0 = 0u;
    rms.group_size = 5120u;
    rms.scalar_a = 1e-6f;
    rms.flags = 0u;
    program.dispatches.push_back(rms);
    program.dispatch_source_node.push_back(0u);

    rt::DispatchBinding quant{};
    quant.kernel_id = rt::KernelId::ACTIVATION_QUANTIZE_W4A8;
    quant.input_count = 1u;
    quant.input_slots[0] = kVRmsOut;
    quant.output_count = 0u;
    quant.group_size = 32u;
    quant.workspace_slot = 0u;
    quant.activation_max_rows = 1u;
    quant.activation_kp = 5120u;
    quant.activation_code_stride = layout.code_row_stride_bytes;
    quant.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(quant);
    program.dispatch_source_node.push_back(1u);

    rt::DispatchBinding psq4{};
    psq4.kernel_id = rt::KernelId::LINEAR_PSQ4;
    psq4.input_count = 0u;
    psq4.output_count = 1u;
    psq4.output_slots[0] = kVPsq4Out;
    psq4.weight_index = 0u;
    psq4.workspace_slot = 0u;
    psq4.compute_spec = 3u;
    psq4.activation_max_rows = 1u;
    psq4.activation_kp = 5120u;
    psq4.activation_code_stride = layout.code_row_stride_bytes;
    psq4.activation_scale_stride = layout.scale_row_stride_bytes;
    program.dispatches.push_back(psq4);
    program.dispatch_source_node.push_back(1u);
}

struct W4a8Buffers {
    void* workspace = nullptr;
    void* output = nullptr;

    bool allocate() {
        return hipMalloc(&workspace, kWorkspaceBytes) == hipSuccess &&
               hipMalloc(&output, 1024u * 2u) == hipSuccess;
    }
    void poison() {
        (void)hipMemset(workspace, 0xAB, kWorkspaceBytes);
        (void)hipMemset(output, 0xAB, 1024u * 2u);
    }
    void free_all() {
        if (workspace) (void)hipFree(workspace);
        if (output) (void)hipFree(output);
        workspace = nullptr;
        output = nullptr;
    }
};

inline void bind_w4a8_context(q35::HostExecutionContext& ctx, void* workspace,
                              const void* const* ext_inputs, void** ext_outputs,
                              rt::StaticParameterSlot* params,
                              rt::WeightSlot* weights) {
    ctx = q35::HostExecutionContext{};
    ctx.workspace = static_cast<uint8_t*>(workspace);
    ctx.workspace_bytes = kWorkspaceBytes;
    ctx.external_inputs = ext_inputs;
    ctx.external_input_count = 1u;
    ctx.external_outputs = ext_outputs;
    ctx.external_output_count = 1u;
    ctx.host_parameters = params;
    ctx.host_parameter_count = 1u;
    ctx.weights = weights;
    ctx.weight_count = 1u;
    ctx.actual_rows = 1u;
    ctx.actual_outputs = 1u;
}

inline void fill_w4a8_weights(rt::StaticParameterSlot* params,
                              rt::WeightSlot* weights, const void* rms_weight,
                              const void* w_codes, const void* w_scales) {
    params[0].device_ptr = rms_weight;
    params[0].elements = 5120u;
    params[0].dtype = rt::ValueDType::BF16;
    std::memset(weights, 0, sizeof(rt::WeightSlot));
    weights[0].rows = 1024u;
    weights[0].cols = 5120u;
    weights[0].k_padded = 5120u;
    weights[0].encoding = 3u;
    weights[0].compute_spec = 3u;
    weights[0].codes = w_codes;
    weights[0].scales = w_scales;
    weights[0].weight_scale_group = 32u;
    weights[0].storage_scale_stride_bytes = 5120u;
    weights[0].preshuffled = true;
}

inline std::vector<::ps::runtime::gpu_mcu::McuPlanNode>
make_w4a8_handwritten_nodes() {
    namespace mcu = ::ps::runtime::gpu_mcu;
    std::vector<mcu::McuPlanNode> nodes(6);
    nodes[0].variant_id = 1;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1;
    nodes[1].variant_id = 2;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120;
    nodes[1].completion_slot = 0;
    nodes[1].invocation_index = 0;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2;
    nodes[2].variant_id = 3;
    nodes[2].kernarg_recipe = mcu::kMcuKernargRecipePsq4Decode1Bf16U16;
    nodes[2].completion_slot = 0;
    nodes[2].invocation_index = 0;
    nodes[2].flags = mcu::kMcuNodeDispatch;
    nodes[2].next = 3;
    nodes[3].variant_id = 0;
    nodes[3].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[3].completion_slot = 0;
    nodes[3].input_slot = mcu::kMcuNoInput;
    nodes[3].flags = mcu::kMcuNodeDispatch;
    nodes[3].next = 4;
    nodes[4].completion_slot = 0;
    nodes[4].flags = mcu::kMcuNodeWait;
    nodes[4].next = 5;
    nodes[5].flags = mcu::kMcuNodeEnd;
    return nodes;
}

}  // namespace gpu_mcu_w4a8
