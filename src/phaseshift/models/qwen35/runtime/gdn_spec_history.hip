#include <phaseshift/models/qwen35/runtime/gdn_spec_history.h>

#include <vector>

namespace ps::qwen35::runtime {

Result<GdnSpecHistory> create_gdn_spec_history(
    gpu::GpuArena& arena,
    const GdnStatePool& pool,
    uint32_t rows) {
    if (rows == 0u) {
        return Status::invalid_argument("create_gdn_spec_history: rows is zero", __FILE__,
                                        __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    if (view.conv_slot_stride == 0u || view.recurrent_slot_stride == 0u) {
        return Status::invalid_state("create_gdn_spec_history: pool geometry is zero", __FILE__,
                                     __LINE__);
    }

    GdnSpecHistory history;
    history.rows = rows;
    history.conv_state_bytes = static_cast<std::size_t>(view.conv_slot_stride) * sizeof(bf16_t);
    history.recurrent_state_bytes =
        static_cast<std::size_t>(view.recurrent_slot_stride) * sizeof(float);

    {
        const std::size_t count = static_cast<std::size_t>(rows) * view.conv_slot_stride;
        auto conv_view = arena.allocate_aligned(count * sizeof(bf16_t), 256u);
        if (!conv_view.ok()) return conv_view.status();
        auto backing = conv_view.release();
        auto tensor = gpu::Tensor::create<bf16_t>(
            backing, {rows, view.conv_slot_stride}, {view.conv_slot_stride, 1u});
        if (!tensor.ok()) return tensor.status();
        history.conv = tensor.release();
    }
    {
        const std::size_t count = static_cast<std::size_t>(rows) * view.recurrent_slot_stride;
        auto rec_view = arena.allocate_aligned(count * sizeof(float), 256u);
        if (!rec_view.ok()) return rec_view.status();
        auto backing = rec_view.release();
        auto tensor = gpu::Tensor::create<float>(
            backing, {rows, view.recurrent_slot_stride},
            {view.recurrent_slot_stride, 1u});
        if (!tensor.ok()) return tensor.status();
        history.recurrent = tensor.release();
    }

    history.initialized = true;
    return history;
}

GdnSpecHistoryDeviceView gdn_spec_history_view(
    const GdnSpecHistory& history,
    uint32_t capture_rows) noexcept {
    GdnSpecHistoryDeviceView view;
    if (!history.initialized || capture_rows == 0u || capture_rows > history.rows) {
        return view;
    }
    view.conv_history = history.conv.data<bf16_t>();
    view.recurrent_history = history.recurrent.data<float>();
    view.capture_rows = capture_rows;
    view.conv_row_stride = static_cast<uint64_t>(history.conv.dim(1));
    view.recurrent_row_stride = static_cast<uint64_t>(history.recurrent.dim(1));
    return view;
}

Status shutdown_gdn_spec_history(GdnSpecHistory& history) noexcept {
    history.conv = gpu::Tensor{};
    history.recurrent = gpu::Tensor{};
    history.rows = 0u;
    history.conv_state_bytes = 0u;
    history.recurrent_state_bytes = 0u;
    history.initialized = false;
    return Status::make_ok();
}

Status restore_gdn_spec_history(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream) {
    if (!history.initialized) {
        return Status::invalid_state("restore_gdn_spec_history: not initialized", __FILE__,
                                     __LINE__);
    }
    if (history_row >= history.rows) {
        return Status::out_of_range("restore_gdn_spec_history: history row out of range",
                                    __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("restore_gdn_spec_history: slot out of range", __FILE__,
                                    __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_dst = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    float* rec_dst =
        view.recurrent_base + static_cast<std::size_t>(slot) * view.recurrent_slot_stride;
    const bf16_t* conv_src =
        history.conv.data<bf16_t>() +
        static_cast<std::size_t>(history_row) * history.conv.dim(1);
    const float* rec_src =
        history.recurrent.data<float>() +
        static_cast<std::size_t>(history_row) * history.recurrent.dim(1);

    hipError_t err = hipMemcpyAsync(conv_dst, conv_src, history.conv_state_bytes,
                                    hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("restore_gdn_spec_history conv copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(rec_dst, rec_src, history.recurrent_state_bytes,
                         hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("restore_gdn_spec_history recurrent copy",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace ps::qwen35::runtime
