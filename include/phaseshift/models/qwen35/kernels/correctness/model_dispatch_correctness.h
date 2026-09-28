#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/device_program.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

struct ModelDispatchStateView {
    ::ps::qwen35::SequenceBlockTableDeviceView block_table{};
    ::ps::qwen35::GdnStatePoolDeviceView gdn{};
    ::ps::qwen35::PagedKVPoolBF16View kv_bf16{};
    ::ps::qwen35::PagedKVPoolFP8View kv_fp8{};
    ::ps::qwen35::PagedKVPoolPSQ4View kv_psq4{};
    ::ps::qwen35::PagedKVPoolPSQ8View kv_psq8{};
    uint32_t kv_dtype = 0;
};

Status launch_model_dispatch_correctness(
    const ::ps::runtime::DeviceProgramView& program,
    const ::ps::runtime::DispatchBinding& binding,
    ::ps::runtime::DeviceBatchContext* batch_context,
    const ModelDispatchStateView& state,
    float* scratch,
    uint32_t* error_word,
    hipStream_t stream);

}
