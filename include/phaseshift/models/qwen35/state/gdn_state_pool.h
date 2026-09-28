#pragma once

#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/state/paged_types.h>
#include <phaseshift/models/qwen35/state/gdn_state.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace ps {
namespace qwen35 {

struct GdnStatePoolDeviceView {
    bf16_t* conv_base = nullptr;
    float* recurrent_base = nullptr;

    uint32_t num_gdn_states = 0;
    uint32_t max_sequences = 0;

    uint64_t conv_slot_stride = 0;
    uint64_t conv_layer_stride = 0;
    uint64_t conv_history_stride = 0;

    uint64_t recurrent_slot_stride = 0;
    uint64_t recurrent_layer_stride = 0;

    uint32_t conv_dim = 0;
    uint32_t conv_dim_padded = 0;
    uint32_t conv_history = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_k = 0;
    uint32_t head_v = 0;
};

struct GdnStatePoolLayout {
    uint32_t num_gdn_states;
    uint32_t conv_dim;
    uint32_t conv_dim_padded;
    uint32_t conv_history;
    uint32_t conv_history_stride;
    uint32_t num_v_heads;
    uint32_t head_k;
    uint32_t head_v;

    static GdnStatePoolLayout make(
        uint32_t gdn_layers, uint32_t conv_dim, uint32_t conv_history,
        uint32_t num_v_heads, uint32_t head_k, uint32_t head_v) {
        const uint32_t padded = (conv_dim + 7u) & ~7u;
        return GdnStatePoolLayout{
            .num_gdn_states = gdn_layers,
            .conv_dim = conv_dim,
            .conv_dim_padded = padded,
            .conv_history = conv_history,
            .conv_history_stride = padded,
            .num_v_heads = num_v_heads,
            .head_k = head_k,
            .head_v = head_v,
        };
    }

    static GdnStatePoolLayout from_text_config(const Qwen35TextConfig& tc) {
        uint32_t gdn_layers = 0;
        for (uint32_t t : tc.layer_types) {
            if (t == 0) ++gdn_layers;
        }
        if (gdn_layers == 0) gdn_layers = 1;
        const uint32_t kh = tc.linear_num_key_heads != 0 ? tc.linear_num_key_heads : 16;
        const uint32_t vh = tc.linear_num_value_heads != 0 ? tc.linear_num_value_heads : 16;
        const uint32_t khd = tc.linear_key_head_dim != 0 ? tc.linear_key_head_dim : 128;
        const uint32_t vhd = tc.linear_value_head_dim != 0 ? tc.linear_value_head_dim : 128;
        uint32_t kern = tc.linear_conv_kernel_dim != 0 ? tc.linear_conv_kernel_dim : 4;
        if (kern == 0) kern = 4;
        return make(
            gdn_layers, 2 * kh * khd + vh * vhd, kern - 1, vh, khd, vhd);
    }

    static GdnStatePoolLayout qwen35_0_8b() {
        return make(18, 6144, 3, 16, 128, 128);
    }
};

class GdnStatePool {
 public:
    static Result<GdnStatePool> create(
        gpu::GpuArena& arena,
        uint32_t max_sequences);

    static Result<GdnStatePool> create(
        gpu::GpuArena& arena,
        uint32_t max_sequences,
        const GdnStatePoolLayout& layout);

    Status initialize(
        SequenceSlotId slot,
        hipStream_t stream);

    Status reset(
        SequenceSlotId slot,
        hipStream_t stream);

    Status release(
        SequenceSlotId slot);

    Status copy_slot_from(
        SequenceSlotId dst_slot,
        const GdnStatePool& src,
        SequenceSlotId src_slot,
        hipStream_t stream);

    GDNState gdn_state(
        SequenceSlotId slot,
        uint32_t state_index) const;

    uint32_t max_sequences() const noexcept { return max_slots_; }

    uint32_t num_gdn_states() const noexcept { return layout_.num_gdn_states; }

    uint32_t conv_dim() const noexcept { return layout_.conv_dim; }

    uint32_t conv_dim_padded() const noexcept { return layout_.conv_dim_padded; }

    uint32_t conv_history() const noexcept { return layout_.conv_history; }

    uint32_t conv_history_stride() const noexcept { return layout_.conv_history_stride; }

    uint32_t num_v_heads() const noexcept { return layout_.num_v_heads; }

    uint32_t head_k() const noexcept { return layout_.head_k; }

    uint32_t head_v() const noexcept { return layout_.head_v; }

    const GdnStatePoolLayout& layout() const noexcept { return layout_; }

    GdnStatePoolDeviceView device_view() const noexcept {
        return GdnStatePoolDeviceView{
            .conv_base = conv_memory_.data<bf16_t>(),
            .recurrent_base = recurrent_memory_.data<float>(),
            .num_gdn_states = layout_.num_gdn_states,
            .max_sequences = max_slots_,
            .conv_slot_stride =
                static_cast<uint64_t>(layout_.num_gdn_states)
                * layout_.conv_history * layout_.conv_history_stride,
            .conv_layer_stride =
                static_cast<uint64_t>(layout_.conv_history) * layout_.conv_history_stride,
            .conv_history_stride = layout_.conv_history_stride,
            .recurrent_slot_stride =
                static_cast<uint64_t>(layout_.num_gdn_states)
                * layout_.num_v_heads * layout_.head_k * layout_.head_v,
            .recurrent_layer_stride =
                static_cast<uint64_t>(layout_.num_v_heads) * layout_.head_k * layout_.head_v,
            .conv_dim = layout_.conv_dim,
            .conv_dim_padded = layout_.conv_dim_padded,
            .conv_history = layout_.conv_history,
            .num_v_heads = layout_.num_v_heads,
            .head_k = layout_.head_k,
            .head_v = layout_.head_v,
        };
    }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    std::size_t bytes_per_slot() const noexcept {
        return reserved_bytes_ / static_cast<std::size_t>(max_slots_);
    }

    uint32_t num_used_slots() const noexcept { return used_count_; }

    uint32_t num_free_slots() const noexcept { return max_slots_ - used_count_; }

    std::size_t used_bytes() const noexcept {
        return static_cast<std::size_t>(used_count_) * bytes_per_slot();
    }

    std::size_t free_bytes() const noexcept {
        return reserved_bytes_ - used_bytes();
    }

  private:
    gpu::Tensor conv_memory_;
    gpu::Tensor recurrent_memory_;
    int device_;
    uint32_t max_slots_;
    GdnStatePoolLayout layout_;
    std::vector<bool> initialized_;
    std::size_t reserved_bytes_;
    uint32_t used_count_;

    GdnStatePool() = default;
};

}
}
