#pragma once
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/runtime/gdn_spec_history.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/runtime/lm_head_proxy.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/program/comm_launcher.h>
#include <phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/runtime/execution/row_bucket.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/status.h>
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <phaseshift/models/qwen35/runtime/decode_backend.h>
#include <phaseshift/runtime/stream_bridge.h>

namespace ps {
namespace qwen35 {

namespace runtime {
struct ValueTraceSink;
}

constexpr uint32_t kMaxTargetHiddenTaps = 8;

enum class DecodeStatus : uint8_t {
    Pending = 0,
    Complete = 1,
    Error = 2,
};

struct ExecutorConfig {
    uint32_t max_scheduled_tokens = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    uint32_t constraint_mask_words = 0;
    runtime::DecodeBackend backend = runtime::DecodeBackend::Host;
    ModelPartition partition{};
    uint32_t pipeline_peer_rank = 0;
    uint32_t tensor_parallel_size = 1;
    uint32_t tensor_parallel_rank = 0;
    bool tp_full_attention = true;
    bool tp_linear_attention = false;
    bool tp_mlp = false;
};

struct DispatchStagingPool {
    uint32_t max_bindings = 0;
    uint64_t pool_bytes = 0;
    void* device = nullptr;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> bytes;
    std::vector<uint8_t> uploaded;
    std::vector<uint8_t> run_seen;
    std::vector<uint8_t> bucket_ran;
    uint64_t slot(uint32_t bucket_index, uint32_t binding_index) const {
        return static_cast<uint64_t>(bucket_index) * max_bindings + binding_index;
    }
};

struct ProgramStagingMeta {
    void* ranges = nullptr;
    uint32_t range_count = 0;
    void* states = nullptr;
    uint32_t state_count = 0;
    uint64_t device_bytes = 0;
};

struct GraphCacheSlot {
    hipGraphExec_t exec = nullptr;
    uint32_t token_count = 0;
    uint32_t num_outputs = 0;
    uint32_t num_sampled = 0;
    uint32_t num_stochastic = 0;
    uint32_t attention_plan_key = 0;
    uint8_t verify_exact = 0;
    uint8_t exec_role = 0;
    uint8_t has_spec_history = 0;
    uint32_t capture_rows = 0;
    const void* gdn_conv_history = nullptr;
    const void* gdn_recurrent_history = nullptr;
    const void* token_device_src = nullptr;
    bool valid = false;
};

struct Executor {
    Executor();
    ~Executor() noexcept;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&& other) noexcept;
    Executor& operator=(Executor&& other) noexcept;
    bool is_shutdown() const noexcept;
    const Qwen35Model* model = nullptr;

    SequenceSlotPool* sequence_pool = nullptr;
    GdnStatePool* gdn_state_pool = nullptr;
    PagedKVPool* kv_pool = nullptr;

    ExecutorConfig config;

    ::ps::runtime::DeviceBatchContextStorage batch_context_storage;
    ::ps::runtime::DeviceBatchContext* batch_context = nullptr;
    ::ps::quantization::imatrix::ImatrixCollector* imatrix_collector = nullptr;
    runtime::ValueTraceSink* value_trace = nullptr;
    runtime::LmHeadCandidateProxy lm_head_proxy;
    void* host_request_staging = nullptr;
    size_t host_staging_bytes = 0;
    void* host_pending_staging = nullptr;
    size_t host_pending_staging_bytes = 0;
    void* host_sampled_staging = nullptr;
    size_t host_sampled_staging_bytes = 0;
    void* host_status_staging = nullptr;
    uint64_t next_submission_id = 0;
    uint64_t active_submission_id = 0;
    ::ps::runtime::ProgramSet program_set;
    bool program_set_ready = false;
    void* comm_self = nullptr;
    ::ps::runtime::CommLaunchFn comm_launch = nullptr;
    gpu::Tensor execution_status;
    gpu::Tensor execution_workspace;
    ::ps::runtime::WeightSlot* host_weight_table = nullptr;
    ::ps::runtime::StaticParameterSlot* host_parameter_table = nullptr;
    std::vector<::ps::runtime::StaticParameterSlot> host_parameter_slots;
    ::ps::kernel::ModelDispatchStateView host_model_state{};
    void* dispatch_staging = nullptr;
    uint64_t dispatch_staging_bytes = 0;
    void* host_dispatch_staging = nullptr;
    DispatchStagingPool staging_pool;
    std::vector<ProgramStagingMeta> program_staging_meta;
    std::array<GraphCacheSlot, ::ps::runtime::ProgramSet::kRowBucketCount> graph_cache{};
    void* host_token_graph_staging = nullptr;
    size_t host_token_graph_staging_bytes = 0;
    ::ps::runtime::DeviceBatchContext* host_ctx_graph_staging = nullptr;
    bool graph_failed = false;

    gpu::Tensor input_ids;

    gpu::Tensor rope_inv_freq;

    gpu::Tensor output_hidden;
    gpu::Tensor logits;
    gpu::Tensor sampled_tokens;
    gpu::Tensor token_hidden;
    std::array<gpu::Tensor, kMaxTargetHiddenTaps> dflash_target_hidden;
    gpu::Tensor constraint_mask_device;
    std::vector<uint32_t> constraint_mask_host;

    void* keepalive = nullptr;

    ::ps::runtime::StreamSignal completion_signal{};
};

Result<Executor> create_model_executor(
    const Qwen35Model& model,
    SequenceSlotPool& sequence_pool,
    GdnStatePool& gdn_state_pool,
    PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const ExecutorConfig& config);

DecodeStatus poll_decode(const Executor& executor, uint64_t generation) noexcept;

Status executor_shutdown(Executor& ex) noexcept;

struct BatchExecutionOutput {
    uint32_t num_outputs = 0;
    gpu::Tensor final_hidden;
    gpu::Tensor logits;
    gpu::Tensor sampled_tokens;
    uint32_t num_token_rows = 0;
    gpu::Tensor token_hidden;
};

struct PendingRequestCommit {
    PagedSequenceState* sequence = nullptr;
    ::ps::runtime::RequestHandle handle{};
    uint32_t committed_tokens = 0;
};

struct PendingBatch {
    hipEvent_t done = nullptr;
    BatchExecutionOutput output;
    const PendingRequestCommit* commits = nullptr;
    uint32_t commit_count = 0;
    uint64_t submission_id = 0;
    uint64_t generation = 0;
    PendingBatch() = default;
    ~PendingBatch() noexcept;
    PendingBatch(const PendingBatch&) = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;
    PendingBatch(PendingBatch&& other) noexcept;
    PendingBatch& operator=(PendingBatch&& other) noexcept;
    void destroy() noexcept;
    bool has_event() const noexcept {
        return done != nullptr;
    }
};

struct ExecuteBatchOptions {
    runtime::GdnSpecHistoryDeviceView gdn_spec_history{};
};

Result<PendingBatch> submit_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream);

Result<PendingBatch> submit_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options);

Result<BatchExecutionOutput> complete_batch(
    Executor& executor,
    PendingBatch&& pending,
    hipStream_t stream);

Result<BatchExecutionOutput> execute_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream);

Result<BatchExecutionOutput> execute_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options);

}
}
