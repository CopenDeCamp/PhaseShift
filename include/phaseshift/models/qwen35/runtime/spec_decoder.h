#pragma once

#include <phaseshift/models/qwen35/runtime/spec_decode.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps::qwen35::runtime {

struct SpecDecoderConfig {
    uint32_t num_drafts = 4u;
    bool bonus_token_enabled = true;
    int32_t eos_token = -1;
    ::ps::runtime::VerifyNumericMode verify_numeric_mode =
        ::ps::runtime::VerifyNumericMode::Fast;
    MtpDraftPolicy draft_policy;
    uint32_t ngram_n = 0u;
    uint32_t ngram_max_tail = 0u;
    uint32_t ngram_window = 2048u;
};

struct SpecDecoderTiming {
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double gdn_snapshot_ms = 0.0;
    double gdn_restore_ms = 0.0;
    double commit_ms = 0.0;
    double input_sync_ms = 0.0;
    double total_ms = 0.0;
    uint32_t iterations = 0u;
    uint32_t target_forwards = 0u;
    uint32_t device_to_host_reads = 0u;
};

struct SpecDecoder {
    Executor* target = nullptr;
    MtpExecutor* mtp = nullptr;
    MtpKvState* mtp_state = nullptr;
    PagedSequenceState* sequence = nullptr;
    GdnStatePool* gdn_pool = nullptr;
    hipStream_t stream = nullptr;
    SpecDecoderConfig config;
    uint32_t hidden_size = 0u;
    SpecDecoderTiming* timing = nullptr;

    void* gdn_conv_snapshot = nullptr;
    void* gdn_rec_snapshot = nullptr;
    std::size_t gdn_conv_bytes = 0u;
    std::size_t gdn_rec_bytes = 0u;

    bf16_t* pending_hidden = nullptr;
    bool pending_hidden_valid = false;

    SpecTransaction transaction;
    bool initialized = false;

    std::vector<std::pair<float, uint8_t>>* draft_log = nullptr;

    std::vector<int32_t> token_history;
};

Result<SpecDecoder> create_spec_decoder(
    Executor& target,
    MtpExecutor& mtp,
    MtpKvState& mtp_state,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const SpecDecoderConfig& config,
    hipStream_t stream);

Status spec_decoder_shutdown(SpecDecoder& decoder) noexcept;

Status spec_decoder_sync_prompt(
    SpecDecoder& decoder,
    const int32_t* prompt_tokens,
    const bf16_t* prompt_hidden,
    uint32_t prompt_tokens_count);

struct SpecIterationOutput {
    std::vector<int32_t> emitted_tokens;
    int32_t pending_token = -1;
    bool finished = false;
    uint32_t num_accepted_drafts = 0u;
    uint32_t num_drafts_generated = 0u;
    uint32_t num_mtp_drafts = 0u;
    uint32_t mtp_length_before = 0u;
    uint32_t mtp_length_after = 0u;
    bool rerun = false;
    std::vector<int32_t> draft_tokens;
    std::vector<int32_t> verify_sampled;
};

Result<SpecIterationOutput> spec_decoder_step(
    SpecDecoder& decoder,
    int32_t pending_token);

}  // namespace ps::qwen35::runtime
