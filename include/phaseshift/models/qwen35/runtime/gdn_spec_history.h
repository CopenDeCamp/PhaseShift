#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::qwen35::runtime {

struct GdnSpecHistoryDeviceView {
    bf16_t* conv_history = nullptr;
    float* recurrent_history = nullptr;

    uint32_t capture_rows = 0u;

    uint64_t conv_row_stride = 0u;
    uint64_t recurrent_row_stride = 0u;
};

struct GdnCompactLogDeviceView {
    float* delta = nullptr;
    float* k = nullptr;
    float* a = nullptr;

    uint64_t delta_layer_stride = 0u;
    uint64_t k_layer_stride = 0u;
    uint64_t a_layer_stride = 0u;
};

struct GdnSpecHistory {
    gpu::Tensor conv;
    gpu::Tensor recurrent;

    uint32_t rows = 0u;
    std::size_t conv_state_bytes = 0u;
    std::size_t recurrent_state_bytes = 0u;
    bool initialized = false;
};

Result<GdnSpecHistory> create_gdn_spec_history(
    gpu::GpuArena& arena,
    const GdnStatePool& pool,
    uint32_t rows);

GdnSpecHistoryDeviceView gdn_spec_history_view(
    const GdnSpecHistory& history,
    uint32_t capture_rows) noexcept;

Status shutdown_gdn_spec_history(GdnSpecHistory& history) noexcept;

Status restore_gdn_spec_history(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream);

Status restore_gdn_spec_conv(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream);

Status commit_gdn_compact_log(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnCompactLogDeviceView& log,
    const ::ps::runtime::DeviceRequestDescriptor* requests,
    uint32_t rows,
    uint32_t key_heads,
    hipStream_t stream);

}  // namespace ps::qwen35::runtime
