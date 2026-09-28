#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/graph/value_type.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kOutputGatherBf16McuSymbol =
    "phaseshift_gpu_mcu_output_gather_bf16";

inline constexpr uint32_t kOutputGatherMcuThreads = 256u;

struct OutputGatherBf16McuArgs {
    const bf16_t* input = nullptr;
    bf16_t* output = nullptr;
    const uint32_t* output_rows = nullptr;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t num_outputs = 0;
    uint32_t features = 0;
};

static_assert(sizeof(OutputGatherBf16McuArgs) == 40);
static_assert(alignof(OutputGatherBf16McuArgs) == 8);
static_assert(offsetof(OutputGatherBf16McuArgs, input) == 0);
static_assert(offsetof(OutputGatherBf16McuArgs, output) == 8);
static_assert(offsetof(OutputGatherBf16McuArgs, output_rows) == 16);
static_assert(offsetof(OutputGatherBf16McuArgs, input_row_stride) == 24);
static_assert(offsetof(OutputGatherBf16McuArgs, output_row_stride) == 28);
static_assert(offsetof(OutputGatherBf16McuArgs, num_outputs) == 32);
static_assert(offsetof(OutputGatherBf16McuArgs, features) == 36);

hipError_t launch_output_gather(
    const void* input, uint32_t input_row_stride,
    void* output, uint32_t output_row_stride,
    const ::ps::runtime::DeviceBatchContext* ctx,
    uint32_t num_outputs,
    uint32_t features, ::ps::runtime::ValueDType dtype, hipStream_t stream);

hipError_t launch_output_gather_bf16_mcu(
    const bf16_t* input, bf16_t* output, const uint32_t* output_rows,
    uint32_t input_row_stride, uint32_t output_row_stride, uint32_t num_outputs,
    uint32_t features, hipStream_t stream);

}  // namespace ps::kernel

