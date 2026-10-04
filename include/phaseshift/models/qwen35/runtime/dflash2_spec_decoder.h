#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/gdn_spec_history.h>
#include <phaseshift/models/qwen35/runtime/token_constraint.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/stop_tokens.h>
#include <phaseshift/runtime/execution/execution_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include <array>
#include <vector>

namespace ps::qwen35::runtime {

constexpr uint32_t kDFlash2SpecMaxVerifyRows = 64u;
constexpr uint32_t kDFlash2SpecMaxVerifyDrafts = kDFlash2SpecMaxVerifyRows - 1u;
constexpr uint64_t kDFlash2SpecHistoryBytesMax = (9ull * 1024ull * 1024ull * 1024ull) / 4ull;

struct DFlash2SpecDecoderConfig {
    uint32_t num_drafts = 7u;
    StopTokens eos_tokens;
    ::ps::runtime::VerifyNumericMode verify_numeric_mode =
        ::ps::runtime::VerifyNumericMode::Exact;
};

struct DFlash2SpecTiming;

struct DFlash2GdnRankState {
    GdnStatePool* pool = nullptr;
    hipStream_t stream = nullptr;
    int device = -1;
    void* conv_snapshot = nullptr;
    void* rec_snapshot = nullptr;
    std::size_t conv_bytes = 0u;
    std::size_t rec_bytes = 0u;
    GdnSpecHistory history;
    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    uint64_t compact_delta_layer_stride = 0u;
    uint64_t compact_k_layer_stride = 0u;
    uint64_t compact_a_layer_stride = 0u;
    ::ps::runtime::DeviceRequestDescriptor* commit_request = nullptr;
    float* compact_scratch = nullptr;
};

struct DFlash2SpecDecoder {
    Executor* target = nullptr;
    dflash2::DFlash2Executor* draft = nullptr;
    dflash2::DFlash2ContextState* context = nullptr;
    PagedSequenceState* sequence = nullptr;
    hipStream_t stream = nullptr;
    DFlash2SpecDecoderConfig config;

    std::vector<DFlash2GdnRankState> gdn_ranks;
    bool gdn_history_enabled = false;
    bool gdn_rerun_reference = false;
    uint64_t gdn_snapshot_bytes = 0u;
    uint64_t gdn_compact_bytes = 0u;

    bool gdn_compact_commit = false;
    uint32_t gdn_key_heads = 0u;
    uint32_t gdn_compact_compare_remaining = 0u;
    std::vector<float> gdn_compact_host_a;
    std::vector<float> gdn_compact_host_b;

    uint32_t hidden_size = 0u;
    int32_t* verify_token_ids_device = nullptr;
    int32_t* decision_staging_device = nullptr;
    bool device_token_bridge = false;
    bool host_proposal_visibility = false;
    std::vector<int32_t> token_history;
    hipEvent_t draft_start_event = nullptr;
    hipEvent_t draft_stop_event = nullptr;
    hipEvent_t verify_start_event = nullptr;
    hipEvent_t verify_stop_event = nullptr;
    bool initialized = false;
    DFlash2SpecTiming* timing = nullptr;

    TokenConstraintState* constraint = nullptr;
    uint32_t constraint_mask_words = 0u;
    std::vector<uint32_t> constraint_mask_host;
    uint32_t* proposal_mask_device = nullptr;

    SamplingConfig sampling{};
    uint64_t sample_index = 0u;
};

Result<DFlash2SpecDecoder> create_dflash2_spec_decoder(
    Executor& target,
    dflash2::DFlash2Executor& draft,
    dflash2::DFlash2ContextState& context,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream);

Status dflash2_spec_decoder_shutdown(DFlash2SpecDecoder& decoder) noexcept;

Status dflash2_spec_decoder_set_constraint(DFlash2SpecDecoder& decoder,
                                           TokenConstraintState* state,
                                           uint32_t mask_words,
                                           gpu::GpuArena* arena);

void dflash2_spec_decoder_set_sampling(DFlash2SpecDecoder& decoder,
                                       const SamplingConfig& sampling);

struct DFlash2PrefillOutput {
    int32_t pending_token = -1;
};

Result<DFlash2PrefillOutput> dflash2_spec_prefill(
    DFlash2SpecDecoder& decoder,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    uint32_t restored_tokens = 0u,
    uint32_t checkpoint_position = 0u);

struct DFlash2TargetSnapshot {
    uint32_t position = 0u;
    uint32_t block_count = 0u;
    bool valid = false;
};

struct DFlash2SpecTiming {
    double prompt_sync_ms = 0.0;
    double draft_ms = 0.0;
    double draft_d2h_ms = 0.0;
    double draft_gpu_ms = 0.0;
    double proposal_wait_ms = 0.0;
    double proposal_copy_ms = 0.0;
    double gdn_snapshot_ms = 0.0;
    double verify_ms = 0.0;
    double verify_gpu_ms = 0.0;
    double decision_wait_ms = 0.0;
    double decision_copy_ms = 0.0;
    double gdn_restore_ms = 0.0;
    double rerun_ms = 0.0;
    double dflash_commit_ms = 0.0;
    double round_ms = 0.0;
    uint32_t rounds = 0u;
    uint32_t full_accepts = 0u;
    uint32_t reruns = 0u;
    uint32_t partial_accepts = 0u;
    uint32_t accepted_drafts = 0u;
    uint32_t generated_tokens = 0u;
    uint32_t dflash_prefix_full_accepts = 0u;
    uint64_t verify_rows_total = 0u;
    uint64_t gdn_history_bytes = 0u;
};

struct DFlash2SpecIterationOutput {
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> emitted{};
    uint32_t emitted_count = 0u;
    int32_t pending_token = -1;
    uint32_t num_drafts = 0u;
    uint32_t num_dflash_drafts = 0u;
    uint32_t num_accepted = 0u;
    bool rerun = false;
    bool finished = false;
};

Result<DFlash2SpecIterationOutput> dflash2_spec_step(
    DFlash2SpecDecoder& decoder,
    int32_t pending_token,
    uint32_t remaining_tokens);

}  // namespace ps::qwen35::runtime
