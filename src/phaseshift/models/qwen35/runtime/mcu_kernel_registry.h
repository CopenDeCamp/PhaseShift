#pragma once

#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/models/qwen35/runtime/gpu_mcu/embedded_kernels.h>

namespace ps {
namespace qwen35 {
namespace runtime {

struct McuKernelCodeObject {
    ::ps::runtime::gpu_mcu::GpuMcuEmbeddedKernel blob{};
    const char* symbol = nullptr;
    uint16_t kernarg_recipe = 0;
    ::ps::runtime::gpu_mcu::AqlHiddenArgsPolicy hidden_args_policy =
        ::ps::runtime::gpu_mcu::AqlHiddenArgsPolicy::IfFits;
};

Result<McuKernelCodeObject> mcu_kernel_code_object(McuCompiledVariantKind kind);

const char* mcu_kernel_variant_kind_name(McuCompiledVariantKind kind);

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
