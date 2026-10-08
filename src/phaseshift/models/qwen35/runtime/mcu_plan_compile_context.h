#pragma once

#include <phaseshift/models/qwen35/runtime/program_executor.h>

#include <cstdint>
#include <vector>

namespace ps::qwen35::runtime {

struct McuStaticPlanCompileContext {
    ::ps::runtime::DeviceBatchContext* batch_context = nullptr;
    ::ps::kernel::ModelDispatchStateView model_state{};
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
    const void* external_input_storage[kHostExternalValueCapacity] = {};
    void* external_output_storage[kHostExternalValueCapacity] = {};
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
    const ::ps::runtime::DeviceSamplingParams* output_sampling_params = nullptr;
    const ::ps::runtime::DeviceRequestDescriptor* device_requests = nullptr;
    uint32_t* device_output_rows = nullptr;
    bool stochastic_topk_eligible = false;
    uint32_t stochastic_top_k = 0;
    int32_t* stochastic_topk_ids = nullptr;
    float* stochastic_topk_logits = nullptr;
    void* stochastic_topk_scratch = nullptr;
    uint32_t stochastic_topk_partitions = 0;
    uint32_t stochastic_topk_scratch_bytes = 0;
    uint32_t* stochastic_topk_active_counts = nullptr;
    DispatchStagingPool* staging_pool = nullptr;
    const ProgramStagingMeta* program_meta = nullptr;
    ::ps::quantization::imatrix::ImatrixCollector* imatrix_collector = nullptr;
    ValueTraceSink* value_trace = nullptr;
    ::ps::runtime::VerifyNumericMode numeric_mode =
        ::ps::runtime::VerifyNumericMode::Fast;
    GdnSpecHistoryDeviceView gdn_spec_history{};
    GdnCompactLogDeviceView gdn_compact{};
    bool verify_exact = false;

    uint32_t row_capacity = 0;
    uint32_t output_capacity = 0;
    uint32_t sampled_output_capacity = 0;
    uint32_t stochastic_output_capacity = 0;
    uint32_t request_capacity = 0;
    uint32_t slot_capacity = 0;
};

inline McuStaticPlanCompileContext make_static_plan_compile_context(
    const HostExecutionContext& legacy_ctx, uint32_t row_capacity,
    uint32_t output_capacity, uint32_t request_capacity,
    uint32_t stochastic_output_capacity) {
    McuStaticPlanCompileContext out{};
    out.batch_context = legacy_ctx.batch_context;
    out.model_state = legacy_ctx.model_state;
    out.weights = legacy_ctx.weights;
    out.weight_count = legacy_ctx.weight_count;
    out.parameters = legacy_ctx.parameters;
    out.parameter_count = legacy_ctx.parameter_count;
    out.host_parameters = legacy_ctx.host_parameters;
    out.host_parameter_count = legacy_ctx.host_parameter_count;
    out.row_positions = legacy_ctx.row_positions;
    out.rope_positions = legacy_ctx.rope_positions;
    out.rope_inv_freq = legacy_ctx.rope_inv_freq;
    out.row_sequence_slots = legacy_ctx.row_sequence_slots;
    out.min_visible_tokens = legacy_ctx.min_visible_tokens;
    out.max_visible_tokens = legacy_ctx.max_visible_tokens;
    out.external_inputs = legacy_ctx.external_inputs;
    out.external_input_count = legacy_ctx.external_input_count;
    out.external_outputs = legacy_ctx.external_outputs;
    out.external_output_count = legacy_ctx.external_output_count;
    for (uint32_t i = 0u; i < kHostExternalValueCapacity; ++i) {
        out.external_input_storage[i] = legacy_ctx.external_input_storage[i];
        out.external_output_storage[i] = legacy_ctx.external_output_storage[i];
    }
    out.workspace = legacy_ctx.workspace;
    out.workspace_bytes = legacy_ctx.workspace_bytes;
    out.scratch = legacy_ctx.scratch;
    out.scratch_floats = legacy_ctx.scratch_floats;
    out.decode_attn_partials = legacy_ctx.decode_attn_partials;
    out.decode_attn_partial_bytes = legacy_ctx.decode_attn_partial_bytes;
    out.error_word = legacy_ctx.error_word;
    out.bucket_m = legacy_ctx.bucket_m;
    out.staging = legacy_ctx.staging;
    out.staging_bytes = legacy_ctx.staging_bytes;
    out.host_staging = legacy_ctx.host_staging;
    out.output_sampling_params = legacy_ctx.output_sampling_params;
    out.device_requests = legacy_ctx.device_requests;
    out.device_output_rows = legacy_ctx.device_output_rows;
    out.stochastic_topk_eligible = legacy_ctx.stochastic_topk_eligible;
    out.stochastic_top_k = legacy_ctx.stochastic_top_k;
    out.stochastic_topk_ids = legacy_ctx.stochastic_topk_ids;
    out.stochastic_topk_logits = legacy_ctx.stochastic_topk_logits;
    out.stochastic_topk_scratch = legacy_ctx.stochastic_topk_scratch;
    out.stochastic_topk_partitions = legacy_ctx.stochastic_topk_partitions;
    out.stochastic_topk_scratch_bytes = legacy_ctx.stochastic_topk_scratch_bytes;
    out.stochastic_topk_active_counts = legacy_ctx.stochastic_topk_active_counts;
    out.staging_pool = legacy_ctx.staging_pool;
    out.program_meta = legacy_ctx.program_meta;
    out.imatrix_collector = legacy_ctx.imatrix_collector;
    out.value_trace = legacy_ctx.value_trace;
    out.numeric_mode = legacy_ctx.numeric_mode;
    out.gdn_spec_history = legacy_ctx.gdn_spec_history;
    out.gdn_compact = legacy_ctx.gdn_compact;
    out.verify_exact = verify_exact_active(legacy_ctx);
    out.sampled_output_capacity = legacy_ctx.actual_sampled_outputs;
    out.row_capacity = row_capacity;
    out.output_capacity = output_capacity;
    out.request_capacity = request_capacity;
    out.stochastic_output_capacity = stochastic_output_capacity;
    return out;
}

struct McuPlanResolverInput {
    HostExecutionContext ctx;
    std::vector<::ps::runtime::DeviceRequestDescriptor> requests;
};

inline McuPlanResolverInput make_plan_resolver_input(
    const McuStaticPlanCompileContext& sc, bool static_plan,
    uint32_t visible_tokens) {
    McuPlanResolverInput view{};
    HostExecutionContext& ctx = view.ctx;
    ctx.batch_context = sc.batch_context;
    ctx.model_state = sc.model_state;
    ctx.weights = sc.weights;
    ctx.weight_count = sc.weight_count;
    ctx.parameters = sc.parameters;
    ctx.parameter_count = sc.parameter_count;
    ctx.host_parameters = sc.host_parameters;
    ctx.host_parameter_count = sc.host_parameter_count;
    ctx.row_positions = sc.row_positions;
    ctx.rope_positions = sc.rope_positions;
    ctx.rope_inv_freq = sc.rope_inv_freq;
    ctx.row_sequence_slots = sc.row_sequence_slots;
    ctx.min_visible_tokens = sc.min_visible_tokens;
    ctx.max_visible_tokens = visible_tokens;
    ctx.external_inputs = sc.external_inputs;
    ctx.external_input_count = sc.external_input_count;
    ctx.external_outputs = sc.external_outputs;
    ctx.external_output_count = sc.external_output_count;
    for (uint32_t i = 0u; i < kHostExternalValueCapacity; ++i) {
        ctx.external_input_storage[i] = sc.external_input_storage[i];
        ctx.external_output_storage[i] = sc.external_output_storage[i];
    }
    ctx.workspace = sc.workspace;
    ctx.workspace_bytes = sc.workspace_bytes;
    ctx.scratch = sc.scratch;
    ctx.scratch_floats = sc.scratch_floats;
    ctx.decode_attn_partials = sc.decode_attn_partials;
    ctx.decode_attn_partial_bytes = sc.decode_attn_partial_bytes;
    ctx.error_word = sc.error_word;
    ctx.bucket_m = sc.bucket_m;
    ctx.staging = sc.staging;
    ctx.staging_bytes = sc.staging_bytes;
    ctx.host_staging = sc.host_staging;
    ctx.output_sampling_params = sc.output_sampling_params;
    ctx.device_requests = sc.device_requests;
    ctx.device_output_rows = sc.device_output_rows;
    ctx.stochastic_topk_eligible = sc.stochastic_topk_eligible;
    ctx.stochastic_top_k = sc.stochastic_top_k;
    ctx.stochastic_topk_ids = sc.stochastic_topk_ids;
    ctx.stochastic_topk_logits = sc.stochastic_topk_logits;
    ctx.stochastic_topk_scratch = sc.stochastic_topk_scratch;
    ctx.stochastic_topk_partitions = sc.stochastic_topk_partitions;
    ctx.stochastic_topk_scratch_bytes = sc.stochastic_topk_scratch_bytes;
    ctx.stochastic_topk_active_counts = sc.stochastic_topk_active_counts;
    ctx.staging_pool = sc.staging_pool;
    ctx.program_meta = sc.program_meta;
    ctx.imatrix_collector = sc.imatrix_collector;
    ctx.value_trace = sc.value_trace;
    ctx.numeric_mode = sc.numeric_mode;
    ctx.gdn_spec_history = sc.gdn_spec_history;
    ctx.gdn_compact = sc.gdn_compact;
    ctx.static_plan_compile = static_plan;
    ctx.static_attention_use_prefill = false;
    ctx.role = sc.verify_exact ? ::ps::runtime::ExecutionRole::Verify
                               : ::ps::runtime::ExecutionRole::Decode;
    ctx.actual_rows = sc.row_capacity;
    ctx.actual_outputs = sc.output_capacity;
    ctx.actual_sampled_outputs = sc.sampled_output_capacity;
    ctx.actual_stochastic_outputs = sc.stochastic_output_capacity;
    ctx.staged_request_count = sc.request_capacity;
    ctx.staged_verify_request_count = 0u;
    if (sc.request_capacity != 0u) {
        view.requests.resize(sc.request_capacity);
        for (uint32_t i = 0u; i < sc.request_capacity; ++i) {
            view.requests[i].request_handle.slot =
                sc.slot_capacity != 0u ? i % sc.slot_capacity : 0u;
            view.requests[i].row_count = i == 0u ? sc.row_capacity : 1u;
        }
        ctx.staged_requests = view.requests.data();
    }
    return view;
}

}  // namespace ps::qwen35::runtime
