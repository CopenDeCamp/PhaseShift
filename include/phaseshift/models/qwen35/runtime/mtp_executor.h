#pragma once
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace ps::qwen35::runtime {

struct ValueTraceSink;

struct MtpExecutorConfig {
    uint32_t num_pages = 4;
    uint32_t page_tokens = 16;
    uint32_t max_rows = 16;
};

struct MtpExecutor {
    const Qwen35Model* model = nullptr;
    KVCacheDType kv_dtype = KVCacheDType::BF16;

    ::ps::runtime::ProgramSet program_set;
    bool program_set_ready = false;

    ::ps::runtime::WeightSlot* host_weight_table = nullptr;
    ::ps::runtime::StaticParameterSlot* host_parameter_table = nullptr;
    std::vector<::ps::runtime::StaticParameterSlot> host_parameter_slots;

    void* dispatch_staging = nullptr;
    uint64_t dispatch_staging_bytes = 0;
    void* host_dispatch_staging = nullptr;
    DispatchStagingPool staging_pool;
    std::vector<ProgramStagingMeta> program_staging_meta;

    gpu::Tensor workspace;
    gpu::Tensor execution_status;
    gpu::Tensor rope_inv_freq;

    gpu::Tensor hidden_in;
    gpu::Tensor hidden_out;
    gpu::Tensor hidden_out_normed;
    gpu::Tensor logits;
    gpu::Tensor sampled;
    gpu::Tensor top2;

    float last_top1_logit = 0.0f;
    float last_top2_logit = 0.0f;
    uint32_t last_top2_token = 0xFFFFFFFFu;
    bool collect_top2 = false;

    MtpKvState kv_state;
    ::ps::runtime::DeviceBatchContextStorage batch_storage;
    ::ps::runtime::DeviceBatchContext* batch_context = nullptr;
    ::ps::kernel::ModelDispatchStateView model_state{};
    uint32_t max_rows = 16;

    bool is_shutdown() const noexcept { return model == nullptr; }
};

struct MtpStepOutcome {
    uint32_t draft_token = 0;
    uint32_t kv_length_before = 0;
    uint32_t kv_length_after = 0;
    uint32_t absolute_position = 0;
    float top1_logit = 0.0f;
    float top2_logit = 0.0f;
    uint32_t top2_token = 0xFFFFFFFFu;
};

Result<MtpExecutor> create_mtp_executor(
    const Qwen35Model& model,
    KVCacheDType kv_dtype,
    gpu::GpuArena& arena,
    const MtpExecutorConfig& config,
    hipStream_t stream);

Status mtp_executor_shutdown(MtpExecutor& executor) noexcept;

Result<uint32_t> run_mtp_rows(
    MtpExecutor& executor,
    const bf16_t* hidden,
    const int32_t* tokens,
    uint32_t rows,
    uint32_t position_start,
    hipStream_t stream,
    ValueTraceSink* trace = nullptr,
    uint32_t kv_position_start = 0xFFFFFFFFu);

Result<MtpStepOutcome> mtp_forward_step(
    MtpExecutor& executor,
    MtpKvState& state,
    const bf16_t* hidden,
    const int32_t* tokens,
    uint32_t rows,
    uint32_t absolute_position_start,
    hipStream_t stream,
    ValueTraceSink* trace = nullptr,
    MtpStateTraceSink* state_trace = nullptr);

Result<uint32_t> run_mtp_head(
    MtpExecutor& executor,
    const bf16_t* hidden,
    int32_t token,
    uint32_t position,
    hipStream_t stream);

}  // namespace ps::qwen35::runtime
