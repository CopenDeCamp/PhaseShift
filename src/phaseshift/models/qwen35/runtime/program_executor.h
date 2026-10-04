#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/staging_layout.h>
#include <phaseshift/runtime/tp/tp_execution.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/gdn_spec_history.h>
#include <phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ps::qwen35::runtime {

class ConstraintLmHeadExact;

constexpr uint64_t kDecodeAttnPartialBytes = 32u * 1024u * 1024u;

struct HostResolvedValue {
    void* ptr = nullptr;
    uint32_t row_stride = 0;
    uint32_t feature_count = 0;
    ::ps::runtime::ValueDType dtype = ::ps::runtime::ValueDType::BF16;
    ::ps::runtime::ValueRowDomain row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
};

struct ValueTraceSink {
    std::function<void(uint32_t value_id, const void* device_ptr, uint32_t row_stride,
                       uint32_t feature_count, ::ps::runtime::ValueDType dtype,
                       uint32_t rows, hipStream_t stream)> on_value = nullptr;
};

struct HostExecutionContext {
    ::ps::runtime::DeviceBatchContext* batch_context = nullptr;    ::ps::kernel::ModelDispatchStateView model_state{};
    const ::ps::runtime::WeightSlot* weights = nullptr;
    uint32_t weight_count = 0;
    const ::ps::runtime::StaticParameterSlot* parameters = nullptr;
    uint32_t parameter_count = 0;
    const ::ps::runtime::StaticParameterSlot* host_parameters = nullptr;
    uint32_t host_parameter_count = 0;
    const uint32_t* row_positions = nullptr;
    const uint32_t* rope_positions = nullptr;
    const double* rope_inv_freq = nullptr;
    const uint32_t* row_sequence_slots = nullptr;
    uint32_t min_visible_tokens = 0;
    uint32_t max_visible_tokens = 0;
    const void* const* external_inputs = nullptr;
    uint32_t external_input_count = 0;
    void* const* external_outputs = nullptr;
    uint32_t external_output_count = 0;
    uint8_t* workspace = nullptr;
    uint64_t workspace_bytes = 0;
    float* scratch = nullptr;
    uint32_t scratch_floats = 0;
    uint8_t* decode_attn_partials = nullptr;
    uint32_t decode_attn_partial_bytes = 0;
    uint32_t* error_word = nullptr;
    uint32_t bucket_m = 0;
    void* staging = nullptr;
    uint64_t staging_bytes = 0;
    void* host_staging = nullptr;
    uint32_t actual_rows = 0;
    uint32_t actual_outputs = 0;
    uint32_t actual_sampled_outputs = 0;
    uint32_t actual_stochastic_outputs = 0;
    const ::ps::runtime::DeviceSamplingParams* output_sampling_params = nullptr;
    const uint32_t* constraint_masks = nullptr;
    uint32_t constraint_mask_words = 0;
    const uint32_t* constraint_allowed_counts = nullptr;
    bool stochastic_topk_eligible = false;
    uint32_t stochastic_top_k = 0;
    int32_t* stochastic_topk_ids = nullptr;
    float* stochastic_topk_logits = nullptr;
    void* stochastic_topk_scratch = nullptr;
    uint32_t stochastic_topk_partitions = 0;
    uint32_t stochastic_topk_scratch_bytes = 0;
    uint32_t* stochastic_topk_active_counts = nullptr;
    ConstraintLmHeadExact* constraint_lm_head = nullptr;
    DispatchStagingPool* staging_pool = nullptr;
    const ProgramStagingMeta* program_meta = nullptr;
    ::ps::quantization::imatrix::ImatrixCollector* imatrix_collector = nullptr;
    ValueTraceSink* value_trace = nullptr;
    ::ps::runtime::ExecutionRole role = ::ps::runtime::ExecutionRole::Decode;
    ::ps::runtime::VerifyNumericMode numeric_mode =
        ::ps::runtime::VerifyNumericMode::Fast;
    GdnSpecHistoryDeviceView gdn_spec_history{};
    GdnCompactLogDeviceView gdn_compact{};
};

inline bool verify_exact_active(const HostExecutionContext& ctx) noexcept {
    return ctx.role == ::ps::runtime::ExecutionRole::Verify &&
           ctx.numeric_mode == ::ps::runtime::VerifyNumericMode::Exact;
}

Status execute_program(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream);

Status execute_program_range(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    size_t begin,
    size_t end);

Status execute_program_tp_ranges(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    const ::ps::runtime::TpExecutionSchedule& schedule,
    ::ps::runtime::TpBarrierHook* barrier,
    int rank);

Result<HostResolvedValue> resolve_host_value(
    const ::ps::runtime::Program& program,
    ::ps::runtime::ValueId id,
    const HostExecutionContext& ctx);

Status upload_host_weight_table(
    const ::ps::runtime::WeightTableView& weights,
    ::ps::runtime::WeightSlot** out,
    hipStream_t stream);

Status upload_host_parameter_table(
    const ::ps::runtime::StaticParameterTableView& params,
    ::ps::runtime::StaticParameterSlot** out,
    hipStream_t stream);

}
