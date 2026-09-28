#pragma once

#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/runtime/request_handle.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace ps::qwen35::runtime {

struct MtpKvConfig {
    uint32_t num_pages = 4;
    uint32_t page_tokens = 16;
    uint32_t num_attention_layers = 1;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    KVCacheDType dtype = KVCacheDType::BF16;
};

struct MtpStateStep {
    uint32_t sequence_id = 0;
    uint32_t step = 0;
    int32_t input_token = -1;
    uint32_t hidden_source_index = 0;
    uint32_t absolute_position = 0;
    uint32_t kv_length_before = 0;
    uint32_t kv_length_after = 0;
    uint32_t block_id = 0;
    uint32_t physical_slot = 0;
    uint32_t offset_in_block = 0;
    uint32_t draft_argmax = 0;
    uint32_t execution_status = 0;
};

struct MtpStateTraceSink {
    std::function<void(const MtpStateStep&)> on_step = nullptr;
};

struct MtpKvState {
    std::optional<PagedKVPool> pool;
    std::optional<SequenceSlotPool> slots;
    ::ps::runtime::RequestHandle handle{};
    std::vector<uint32_t> block_to_page;

    uint32_t sequence_id = 0;
    uint32_t logical_length = 0;
    uint32_t capacity_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t step_index = 0;
    KVCacheDType dtype = KVCacheDType::BF16;
    bool initialized = false;

    bool is_initialized() const noexcept { return initialized; }
};

Result<MtpKvState> create_mtp_kv_state(
    gpu::GpuArena& arena,
    const MtpKvConfig& config,
    uint32_t sequence_id,
    hipStream_t stream);

Status mtp_kv_reset(MtpKvState& state, hipStream_t stream);

Status mtp_kv_reserve(MtpKvState& state, uint32_t rows);

Status mtp_kv_shutdown(MtpKvState& state) noexcept;

Result<uint32_t> mtp_kv_physical_slot(const MtpKvState& state, uint32_t logical_index);

::ps::kernel::ModelDispatchStateView mtp_kv_state_view(const MtpKvState& state);

Status mtp_kv_dump_canonical(
    const MtpKvState& state,
    uint32_t length,
    bf16_t* k_out,
    bf16_t* v_out,
    hipStream_t stream);

}  // namespace ps::qwen35::runtime
