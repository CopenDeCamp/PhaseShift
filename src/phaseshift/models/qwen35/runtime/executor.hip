#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/runtime/stream_bridge.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
#include <cstdlib>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/runtime/execution/row_bucket.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <new>
#include <phaseshift/models/qwen35/runtime/embedding_selector.h>

namespace ps {
namespace qwen35 {
namespace reference {
namespace {
__device__ __forceinline__ float b16f_local(const bf16_t x) {
    return __bfloat162float(__ushort_as_bfloat16(x.data));
}
__global__ void embedding_lookup_batched_kernel_local(
    const bf16_t* __restrict__ embed,
    const int32_t* __restrict__ token_ids,
    bf16_t* __restrict__ out,
    uint32_t N,
    uint32_t hidden) {
    const uint32_t row = blockIdx.x;
    if (row >= N) return;
    const int32_t tok = token_ids[row];
    const bf16_t* src = embed + (size_t)tok * hidden;
    bf16_t* dst = out + (size_t)row * hidden;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        dst[i] = src[i];
    }
}
__global__ void output_gather_kernel_local(
    const bf16_t* __restrict__ hidden,
    const uint32_t* __restrict__ output_rows,
    bf16_t* __restrict__ out,
    uint32_t S,
    uint32_t hidden_size) {
    const uint32_t row = blockIdx.x;
    if (row >= S) return;
    const uint32_t src_row = output_rows[row];
    const bf16_t* src = hidden + (size_t)src_row * hidden_size;
    bf16_t* dst = out + (size_t)row * hidden_size;
    for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}
__global__ void lm_head_rows_reference_kernel_local(
    const bf16_t* __restrict__ weight,
    const bf16_t* __restrict__ hidden,
    float* __restrict__ logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t hidden_size) {
    const uint32_t r = blockIdx.y;
    if (r >= rows) return;
    const uint32_t v = blockIdx.x;
    if (v >= vocab) return;
    const bf16_t* wrow = weight + (size_t)v * hidden_size;
    const bf16_t* hrow = hidden + (size_t)r * hidden_size;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        acc = fmaf(b16f_local(hrow[i]), b16f_local(wrow[i]), acc);
    }
    __shared__ float sacc[256];
    const uint32_t lane = threadIdx.x;
    sacc[lane] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sacc[lane] += sacc[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        logits[(size_t)r * vocab + v] = sacc[0];
    }
}
}
inline Status launch_embedding_lookup_batched(
    const ps::weights::MatrixWeight& embed_weight,
    const gpu::Tensor& token_ids,
    uint32_t token_count,
    gpu::Tensor& output,
    hipStream_t stream) {
    if (embed_weight.encoding != ps::weights::MatrixEncoding::Bf16) {
        return Status::unsupported("quantized embedding is not supported", __FILE__, __LINE__);
    }
    if (embed_weight.bf16.ndim() != 2) {
        return Status::invalid_argument("embedding weight must be 2D", __FILE__, __LINE__);
    }
    const uint32_t N = token_count;
    const uint32_t hidden = static_cast<uint32_t>(embed_weight.bf16.dim(1));
    embedding_lookup_batched_kernel_local<<<N, 128, 0, stream>>>(
        embed_weight.bf16.data<bf16_t>(), token_ids.data<int32_t>(),
        output.data<bf16_t>(), N, hidden);
    return Status::make_ok();
}
inline Status launch_output_gather(
    const gpu::Tensor& hidden,
    const gpu::Tensor& output_rows,
    uint32_t row_count,
    gpu::Tensor& output_hidden,
    hipStream_t stream) {
    const uint32_t S = row_count;
    const uint32_t hidden_size = static_cast<uint32_t>(hidden.dim(1));
    output_gather_kernel_local<<<S, 128, 0, stream>>>(
        hidden.data<bf16_t>(), output_rows.data<uint32_t>(),
        output_hidden.data<bf16_t>(), S, hidden_size);
    return Status::make_ok();
}
inline Status launch_lm_head_rows_reference(
    const ps::weights::MatrixWeight& weight,
    const gpu::Tensor& hidden,
    gpu::Tensor& logits,
    uint32_t rows,
    hipStream_t stream) {
    if (hidden.ndim() != 2) {
        return Status::invalid_argument("lm head hidden must be 2D", __FILE__, __LINE__);
    }
    if (logits.ndim() != 2) {
        return Status::invalid_argument("lm head logits must be 2D", __FILE__, __LINE__);
    }
    if (rows == 0) {
        return Status::invalid_argument("lm head rows must be non-zero", __FILE__, __LINE__);
    }
    const uint32_t vocab = static_cast<uint32_t>(weight.rows);
    if (weight.encoding != ps::weights::MatrixEncoding::Bf16) {
        return Status::unsupported("quantized lm_head is not supported", __FILE__, __LINE__);
    }
    if (weight.bf16.ndim() != 2) {
        return Status::invalid_argument("lm head weight must be 2D", __FILE__, __LINE__);
    }
    const uint32_t hidden_size = static_cast<uint32_t>(weight.bf16.dim(1));
    if (hidden.dim(1) != hidden_size) {
        return Status::invalid_argument("lm head hidden/features mismatch", __FILE__, __LINE__);
    }
    if (hidden.dim(0) < rows) {
        return Status::invalid_argument("lm head hidden rows insufficient", __FILE__, __LINE__);
    }
    if (logits.dim(0) < rows || logits.dim(1) != vocab) {
        return Status::invalid_argument("lm head logits shape mismatch", __FILE__, __LINE__);
    }
    dim3 grid(vocab, rows);
    lm_head_rows_reference_kernel_local<<<grid, 256, 0, stream>>>(
        weight.bf16.data<bf16_t>(), hidden.data<bf16_t>(), logits.data<float>(),
        rows, vocab, hidden_size);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("launch_lm_head_rows_reference", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}
}

namespace {

struct PreparedBatch {
    uint32_t num_tokens = 0;
    uint32_t num_requests = 0;
    uint32_t num_outputs = 0;
    uint32_t num_sampled_outputs = 0;
    uint32_t num_stochastic_outputs = 0;
};

struct ReservationEntry {
    PagedSequenceState* seq = nullptr;
    ::ps::runtime::RequestHandle handle{};
    uint32_t old_block_size = 0;
};

void poison_if_current(
    PagedSequenceState* sequence,
    ::ps::runtime::RequestHandle handle) noexcept {
    if (sequence == nullptr) {
        return;
    }
    if (!::ps::runtime::request_handle_equal(
            sequence->request_handle(), handle)) {
        return;
    }
    sequence->mark_poisoned();
}

constexpr uint32_t kHostExternalCapacity = 16u;
static_assert(4u + kMaxTargetHiddenTaps <= kHostExternalCapacity,
              "host external output capacity must cover the fixed outputs plus taps");

Status build_host_execution_context(
    Executor& executor,
    const ::ps::runtime::Program& program,
    uint32_t* status_ptr,
    uint32_t actual_rows,
    uint32_t actual_outputs,
    uint32_t actual_sampled_outputs,
    uint32_t actual_stochastic_outputs,
    uint32_t min_visible_tokens,
    uint32_t max_visible_tokens,
    ::ps::runtime::ExecutionRole role,
    ::ps::runtime::VerifyNumericMode numeric_mode,
    const runtime::GdnSpecHistoryDeviceView& gdn_spec_history,
    const uint32_t* constraint_masks,
    uint32_t constraint_mask_words,
    runtime::HostExecutionContext& ctx) {
    if (status_ptr == nullptr) {
        return Status::invalid_argument("status pointer must be non-null", __FILE__, __LINE__);
    }
    if (executor.batch_context == nullptr) {
        return Status::invalid_state("batch context not bound", __FILE__, __LINE__);
    }
    if (executor.host_weight_table == nullptr) {
        return Status::invalid_state("host weight table not uploaded", __FILE__, __LINE__);
    }
    if (executor.host_parameter_table == nullptr && program.parameter_slot_count != 0) {
        return Status::invalid_state("host parameter table not uploaded", __FILE__, __LINE__);
    }
    if (program.external_input_count > kHostExternalCapacity ||
        program.external_output_count > kHostExternalCapacity) {
        return Status::unsupported("host backend external count exceeds static capacity",
                                   __FILE__, __LINE__);
    }
    if (executor.dispatch_staging == nullptr || executor.host_dispatch_staging == nullptr ||
        executor.dispatch_staging_bytes == 0) {
        return Status::invalid_state("dispatch staging not allocated", __FILE__, __LINE__);
    }
    const void* inputs[kHostExternalCapacity] = {};
    void* outputs[kHostExternalCapacity] = {};
    auto* token_ids = executor.batch_context_storage.token_ids.data<int32_t>();
    for (uint32_t i = 0; i < program.external_input_count; ++i) inputs[i] = token_ids;
    for (uint32_t i = 0; i < program.external_output_count; ++i) {
        if (i == 0u) {
            outputs[i] = executor.output_hidden.data<void>();
        } else if (i == 1u) {
            outputs[i] = executor.logits.data<void>();
        } else if (i == 2u) {
            outputs[i] = executor.sampled_tokens.data<void>();
        } else if (i == 3u) {
            outputs[i] = executor.token_hidden.data<void>();
        } else {
            const uint32_t tap = i - 4u;
            if (tap >= executor.config.target_hidden_tap_count) {
                return Status::invalid_state(
                    "external output has no bound buffer", __FILE__, __LINE__);
            }
            outputs[i] = executor.dflash_target_hidden[tap].data<void>();
        }
    }
    auto* ws_base = executor.execution_workspace.data<uint8_t>();
    if (ws_base == nullptr) {
        return Status::invalid_state("execution workspace not allocated", __FILE__, __LINE__);
    }
    constexpr uint64_t kScratchFloats = 8192;
    const uint64_t scratch_offset =
        (static_cast<uint64_t>(program.workspace.total_bytes) + 255ull) & ~255ull;
    if (scratch_offset + kScratchFloats * sizeof(float) >
        executor.execution_workspace.allocation_bytes()) {
        return Status::insufficient_memory("execution workspace too small for scratch",
                                           __FILE__, __LINE__);
    }
    ctx.batch_context = executor.batch_context;
    ctx.imatrix_collector = executor.imatrix_collector;
    ctx.value_trace = executor.value_trace;
    ctx.role = role;
    ctx.numeric_mode = numeric_mode;
    ctx.gdn_spec_history = gdn_spec_history;
    ctx.model_state = executor.host_model_state;
    ctx.weights = executor.host_weight_table;
    ctx.weight_count = program.weight_slot_count;
    ctx.parameters = executor.host_parameter_table;
    ctx.parameter_count = program.parameter_slot_count;
    ctx.host_parameters = executor.host_parameter_slots.data();
    ctx.host_parameter_count = static_cast<uint32_t>(executor.host_parameter_slots.size());
    ctx.row_positions = executor.batch_context_storage.row_positions.data<uint32_t>();
    ctx.rope_inv_freq = executor.rope_inv_freq.data<double>();
    ctx.row_sequence_slots = executor.batch_context_storage.row_sequence_slots.data<uint32_t>();
    ctx.external_inputs = inputs;
    ctx.external_input_count = program.external_input_count;
    ctx.external_outputs = outputs;
    ctx.external_output_count = program.external_output_count;
    ctx.workspace = ws_base;
    ctx.workspace_bytes = executor.execution_workspace.allocation_bytes();
    ctx.scratch = reinterpret_cast<float*>(ws_base + scratch_offset);
    ctx.scratch_floats = static_cast<uint32_t>(kScratchFloats);
    const uint64_t attn_partial_offset =
        (scratch_offset + kScratchFloats * sizeof(float) + 255ull) & ~255ull;
    if (attn_partial_offset + runtime::kDecodeAttnPartialBytes >
        executor.execution_workspace.allocation_bytes()) {
        return Status::insufficient_memory("execution workspace too small for attention partials",
                                           __FILE__, __LINE__);
    }
    ctx.decode_attn_partials = ws_base + attn_partial_offset;
    ctx.decode_attn_partial_bytes = static_cast<uint32_t>(runtime::kDecodeAttnPartialBytes);
    ctx.error_word = status_ptr;
    ctx.comm_self = executor.comm_self;
    ctx.comm_launch = executor.comm_launch;
    ctx.bucket_m = static_cast<uint32_t>(program.row_bucket);
    ctx.staging = executor.dispatch_staging;
    ctx.staging_bytes = executor.dispatch_staging_bytes;
    ctx.host_staging = executor.host_dispatch_staging;
    ctx.staging_pool = &executor.staging_pool;
    const size_t meta_idx =
        static_cast<size_t>(&program - &executor.program_set.programs[0]);
    ctx.program_meta =
        (meta_idx < executor.program_staging_meta.size())
            ? &executor.program_staging_meta[meta_idx]
            : nullptr;
    ctx.actual_rows = actual_rows;
    ctx.actual_outputs = actual_outputs;
    ctx.actual_sampled_outputs = actual_sampled_outputs;
    ctx.actual_stochastic_outputs = actual_stochastic_outputs;
    ctx.output_sampling_params =
        executor.batch_context_storage.output_sampling_params
            .data<::ps::runtime::DeviceSamplingParams>();
    ctx.constraint_masks = constraint_masks;
    ctx.constraint_mask_words = constraint_mask_words;
    ctx.lm_head_proxy = executor.lm_head_proxy.valid() ? &executor.lm_head_proxy : nullptr;
    ctx.min_visible_tokens = min_visible_tokens;
    ctx.max_visible_tokens = max_visible_tokens;
    return Status::make_ok();
}

Status launch_host_backend_range(
    Executor& executor,
    const ::ps::runtime::Program& program,
    runtime::HostExecutionContext& ctx,
    hipStream_t stream,
    uint32_t begin,
    uint32_t end) {
    (void)executor;
    if (ctx.error_word == nullptr) {
        return Status::invalid_state("error word not bound", __FILE__, __LINE__);
    }
    if (begin > end ||
        static_cast<std::size_t>(end) > program.dispatches.size()) {
        return Status::invalid_argument("dispatch range out of bounds", __FILE__,
                                        __LINE__);
    }
    return runtime::execute_program_range(program, ctx, stream, begin, end);
}

template <typename T>
Status allocate_contiguous_tensor(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    gpu::Tensor& out)
{
    if (shape.empty()) {
        return Status::invalid_argument("tensor shape must be non-empty", __FILE__, __LINE__);
    }
    std::size_t count = 1;
    for (std::size_t d : shape) {
        if (std::numeric_limits<std::size_t>::max() / count < d) {
            return Status::invalid_argument("tensor element count overflow", __FILE__, __LINE__);
        }
        count *= d;
    }
    if (std::numeric_limits<std::size_t>::max() / count < sizeof(T)) {
        return Status::invalid_argument("tensor byte count overflow", __FILE__, __LINE__);
    }
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto result = gpu::Tensor::create<T>(backing, shape, strides);
    if (!result.ok()) return result.status();
    out = result.release();
    return Status::make_ok();
}

}

#ifdef PHASESHIFT_HIP_GRAPH
static int graph_slot_index(::ps::runtime::RowBucket bucket) {
    switch (bucket) {
        case ::ps::runtime::RowBucket::R16: return 0;
        case ::ps::runtime::RowBucket::R32: return 1;
        case ::ps::runtime::RowBucket::R64: return 2;
        case ::ps::runtime::RowBucket::R128: return 3;
        case ::ps::runtime::RowBucket::R256: return 4;
        case ::ps::runtime::RowBucket::R512: return 5;
        case ::ps::runtime::RowBucket::R1024: return 6;
        case ::ps::runtime::RowBucket::R2048: return 7;
    }
    return -1;
}

static bool graph_debug_enabled() {
    static const bool enabled = std::getenv("PHASESHIFT_GRAPH_DEBUG") != nullptr;
    return enabled;
}

static bool graph_debug_first(uint32_t bucket_index) {
    static std::array<uint8_t, ::ps::runtime::ProgramSet::kRowBucketCount> seen{};
    if (!graph_debug_enabled() || bucket_index >= seen.size())
        return false;
    if (seen[bucket_index] != 0)
        return false;
    seen[bucket_index] = 1;
    return true;
}

static bool program_pool_warm(
    const Executor& executor,
    const ::ps::runtime::Program& program,
    uint32_t bidx) {
    if (executor.staging_pool.device == nullptr || executor.staging_pool.max_bindings == 0)
        return false;
    if (bidx >= executor.staging_pool.bucket_ran.size() ||
        executor.staging_pool.bucket_ran[bidx] == 0)
        return false;
    for (uint32_t i = 0; i < static_cast<uint32_t>(program.dispatches.size()); ++i) {
        const uint64_t slot = executor.staging_pool.slot(bidx, i);
        if (slot >= executor.staging_pool.bytes.size())
            return false;
        if (executor.staging_pool.bytes[slot] == 0)
            continue;
        if (slot >= executor.staging_pool.run_seen.size() ||
            executor.staging_pool.run_seen[slot] == 0)
            continue;
        if (executor.staging_pool.uploaded[slot] == 0)
            return false;
    }
    return true;
}

static bool attention_plan_key_for(
    Executor& executor,
    const ::ps::runtime::Program& program,
    uint32_t actual_rows,
    uint32_t max_visible_tokens,
    bool use_prefill,
    uint32_t& out_key) {
    size_t attention_index = program.dispatches.size();
    for (size_t i = 0; i < program.dispatches.size(); ++i) {
        if (program.dispatches[i].kernel_id == ::ps::runtime::KernelId::PAGED_ATTENTION) {
            attention_index = i;
            break;
        }
    }
    if (attention_index == program.dispatches.size()) {
        out_key = 0u;
        return true;
    }
    if (program.external_input_count > kHostExternalCapacity ||
        program.external_output_count > kHostExternalCapacity)
        return false;
    const void* inputs[kHostExternalCapacity] = {};
    void* outputs[kHostExternalCapacity] = {};
    auto* token_ids = executor.batch_context_storage.token_ids.data<int32_t>();
    for (uint32_t i = 0; i < program.external_input_count; ++i) inputs[i] = token_ids;
    for (uint32_t i = 0; i < program.external_output_count; ++i) {
        if (i == 0u) {
            outputs[i] = executor.output_hidden.data<void>();
        } else if (i == 1u) {
            outputs[i] = executor.logits.data<void>();
        } else if (i == 2u) {
            outputs[i] = executor.sampled_tokens.data<void>();
        } else if (i == 3u) {
            outputs[i] = executor.token_hidden.data<void>();
        } else {
            const uint32_t tap = i - 4u;
            if (tap >= executor.config.target_hidden_tap_count) return false;
            outputs[i] = executor.dflash_target_hidden[tap].data<void>();
        }
    }
    runtime::HostExecutionContext ctx{};
    ctx.model_state = executor.host_model_state;
    ctx.comm_self = executor.comm_self;
    ctx.comm_launch = executor.comm_launch;
    ctx.workspace = executor.execution_workspace.data<uint8_t>();
    ctx.external_inputs = inputs;
    ctx.external_input_count = program.external_input_count;
    ctx.external_outputs = outputs;
    ctx.external_output_count = program.external_output_count;
    ctx.row_sequence_slots = executor.batch_context_storage.row_sequence_slots.data<uint32_t>();
    ctx.row_positions = executor.batch_context_storage.row_positions.data<uint32_t>();
    ctx.decode_attn_partials = executor.execution_workspace.data<uint8_t>();
    ctx.decode_attn_partial_bytes = static_cast<uint32_t>(runtime::kDecodeAttnPartialBytes);
    ctx.actual_rows = actual_rows;
    ctx.max_visible_tokens = max_visible_tokens;
    ctx.batch_context = executor.batch_context;
    auto sig = runtime::paged_attention_plan_signature(
        program, attention_index, ctx, use_prefill);
    if (!sig.ok()) return false;
    out_key = sig.value();
    return true;
}
#endif

Result<PendingBatch> submit_co_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options) {

    if (executor.active_submission_id != 0u) {
        return Status::invalid_state(
            "executor already has an in-flight batch", __FILE__, __LINE__);
    }

    auto vst = validate_scheduled_batch(batch, executor.config.max_scheduled_tokens, executor.config.max_scheduled_requests);
    if (!vst.ok()) return vst;

    const uint32_t N = batch.num_tokens;
    const uint32_t R = batch.num_requests;

    PreparedBatch prep;
    prep.num_tokens = N;
    prep.num_requests = R;

    auto* input_ids = executor.input_ids.data<int32_t>();
    auto* staging = reinterpret_cast<::ps::runtime::DeviceRequestDescriptor*>(executor.host_request_staging);
    auto* pending_commits = static_cast<PendingRequestCommit*>(executor.host_pending_staging);
    uint32_t output_count = 0;
    uint32_t sampled_count = 0;
    uint32_t stochastic_count = 0;
    uint32_t num_decode = 0;
    uint32_t num_verify = 0;
    uint32_t num_prefill = 0;
    uint32_t row_offset = 0;
    for (uint32_t i = 0; i < R; ++i) {
        const ScheduledRequest& req = batch.requests[i];
        pending_commits[i].sequence = req.sequence;
        pending_commits[i].handle = req.handle;
        pending_commits[i].committed_tokens = req.num_tokens;
        staging[i].request_handle = req.handle;
        staging[i].row_begin = row_offset;
        staging[i].row_count = req.num_tokens;
        staging[i].prefix_length = req.prefix_tokens;
        staging[i].sequence_length = req.prefix_tokens + req.num_tokens;
        staging[i].execution_class = req.execution_class;
        staging[i].compute_logits = req.compute_logits ? 1 : 0;
        staging[i].sampling = ::ps::runtime::DeviceSamplingParams{};
        if (req.compute_logits) {
            const runtime::SamplingRequest sampling = runtime::make_sampling_request(
                req.sampling, req.sample, req.sampling_index);
            staging[i].sampling.mode = static_cast<uint8_t>(sampling.mode);
            staging[i].sampling.temperature = sampling.temperature;
            staging[i].sampling.top_p = sampling.top_p;
            staging[i].sampling.top_k = sampling.top_k;
            staging[i].sampling.seed = sampling.seed;
            staging[i].sampling.sample_index = sampling.sample_index;
            staging[i].sampling.reserved[0] = req.token_constraint ? 1u : 0u;
            uint32_t oc = req.num_output_rows ? req.num_output_rows : 1u;
            staging[i].output_count = static_cast<uint8_t>(oc < 255u ? oc : 255u);
            staging[i].output_index = output_count;
            output_count += oc;
            if (sampling.mode != runtime::SamplingMode::None) sampled_count += oc;
            if (sampling.mode == runtime::SamplingMode::Stochastic) stochastic_count += oc;
        } else {
            staging[i].output_count = 0u;
            staging[i].output_index = 0xFFFFFFFFu;
        }
        if (req.execution_class == ::ps::runtime::ExecutionClass::DECODE) ++num_decode;
        else if (req.execution_class == ::ps::runtime::ExecutionClass::SPEC_VERIFY) ++num_verify;
        else if (req.execution_class == ::ps::runtime::ExecutionClass::PREFILL) ++num_prefill;
        row_offset += req.num_tokens;
    }
    if (output_count > executor.config.max_scheduled_output_rows) {
        return Status::invalid_argument(
            "scheduled output rows exceed max_scheduled_output_rows", __FILE__, __LINE__);
    }
    if (sampled_count > output_count) {
        return Status::invalid_argument(
            "sampled output rows exceed output rows", __FILE__, __LINE__);
    }
    if (stochastic_count > sampled_count) {
        return Status::invalid_argument(
            "stochastic output rows exceed sampled output rows", __FILE__, __LINE__);
    }
    prep.num_outputs = output_count;
    prep.num_sampled_outputs = sampled_count;
    prep.num_stochastic_outputs = stochastic_count;
    const bool any_constraint =
        batch.constraint_masks != nullptr && batch.constraint_mask_words > 0u &&
        executor.config.constraint_mask_words == batch.constraint_mask_words &&
        executor.constraint_mask_device.data<uint32_t>() != nullptr;
    if (any_constraint) {
        std::memcpy(
            executor.constraint_mask_host.data(), batch.constraint_masks,
            static_cast<std::size_t>(output_count) * batch.constraint_mask_words *
                sizeof(uint32_t));
    }
    uint64_t min_vis = UINT64_MAX;
    uint64_t max_vis = 0;
    for (uint32_t i = 0; i < R; ++i) {
        if (staging[i].row_count == 0u) continue;
        const uint64_t req_min = static_cast<uint64_t>(staging[i].prefix_length) + 1u;
        const uint64_t req_max = static_cast<uint64_t>(staging[i].sequence_length);
        if (req_min < min_vis) min_vis = req_min;
        if (req_max > max_vis) max_vis = req_max;
    }
    const uint32_t min_visible_tokens = (min_vis == UINT64_MAX) ? 0u : static_cast<uint32_t>(min_vis);
    const uint32_t max_visible_tokens = static_cast<uint32_t>(max_vis);
    auto upload = [&](void* dst, const void* src, std::size_t bytes) -> Status {
        hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("execute_batch hipMemcpyAsync",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        return Status::make_ok();
    };
    const bool token_ids_on_device =
        batch.token_ids_location == TokenIdsLocation::Device;
    auto upload_tokens = [&](void* dst, const void* src) -> Status {
        const hipMemcpyKind kind =
            token_ids_on_device ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice;
        hipError_t err = hipMemcpyAsync(dst, src, N * sizeof(int32_t), kind, stream);
        if (err != hipSuccess) {
            return Status::hip_error("execute_batch token upload",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        return Status::make_ok();
    };
    ReservationEntry* reservations = static_cast<ReservationEntry*>(__builtin_alloca(sizeof(ReservationEntry) * R));
    uint32_t reservation_count = 0;
    auto rollback_reservations_array = [&](uint32_t count, hipStream_t s) -> Status {
        for (uint32_t r = 0; r < count; ++r) {
            if (!reservations[r].seq) continue;
            if (!::ps::runtime::request_handle_equal(
                    reservations[r].seq->request_handle(), reservations[r].handle)) {
                return Status::invalid_state(
                    "stale request handle during reservation rollback", __FILE__, __LINE__);
            }
            auto st2 = rollback_sequence_append(*reservations[r].seq, reservations[r].old_block_size, s);
            if (!st2.ok()) return st2;
        }
        return Status::make_ok();
    };
    auto rollback_with_original_array = [&](uint32_t count, hipStream_t s, const Status& orig) -> Status {
        auto st2 = rollback_reservations_array(count, s);
        if (!st2.ok()) {
            return Status::invalid_state(
                (std::string("reservation rollback failed after '") + orig.message() +
                 "': " + st2.message()).c_str(),
                __FILE__, __LINE__);
        }
        return orig;
    };
    for (uint32_t i = 0; i < R; ++i) {
        reservations[reservation_count].seq = batch.requests[i].sequence;
        reservations[reservation_count].handle = batch.requests[i].handle;
        reservations[reservation_count].old_block_size = static_cast<uint32_t>(batch.requests[i].sequence->block_table.size());
        auto reserve_st = reserve_sequence_append(*reservations[reservation_count].seq, batch.requests[i].num_tokens, stream);
        if (!reserve_st.ok()) {
            return rollback_with_original_array(reservation_count, stream, reserve_st);
        }
        ++reservation_count;
    }
    auto* bctx_token_ids = executor.batch_context_storage.token_ids.data<int32_t>();
    auto* bctx_requests = executor.batch_context_storage.requests.data<::ps::runtime::DeviceRequestDescriptor>();
    ::ps::runtime::DeviceBatchContext host_ctx{};
    host_ctx.actual_rows = N;
    host_ctx.num_requests = R;
    host_ctx.num_outputs = output_count;
    host_ctx.num_decode_requests = num_decode;
    host_ctx.num_verify_requests = num_verify;
    host_ctx.num_prefill_requests = num_prefill;
    host_ctx.token_ids = bctx_token_ids;
    host_ctx.requests = bctx_requests;
    host_ctx.row_sequence_slots = executor.batch_context_storage.row_sequence_slots.data<uint32_t>();
    host_ctx.row_positions = executor.batch_context_storage.row_positions.data<uint32_t>();
    host_ctx.output_rows = executor.batch_context_storage.output_rows.data<uint32_t>();
    host_ctx.output_sampling_params =
        executor.batch_context_storage.output_sampling_params
            .data<::ps::runtime::DeviceSamplingParams>();
    if (any_constraint) {
        host_ctx.constraint_masks = executor.constraint_mask_device.data<uint32_t>();
        host_ctx.constraint_mask_words = batch.constraint_mask_words;
    }
    host_ctx.device_ptr = executor.batch_context;

    bool gpu_started = false;
    const ::ps::runtime::ExecutionRole exec_role =
        (batch.speculative_verify || num_verify != 0u)
            ? ::ps::runtime::ExecutionRole::Verify
            : (num_prefill == 0u ? ::ps::runtime::ExecutionRole::Decode
                                 : ::ps::runtime::ExecutionRole::Prefill);
    const ::ps::runtime::VerifyNumericMode exec_numeric_mode = batch.verify_numeric_mode;
    const uint8_t verify_exact =
        (exec_role == ::ps::runtime::ExecutionRole::Verify &&
         exec_numeric_mode == ::ps::runtime::VerifyNumericMode::Exact)
            ? 1u
            : 0u;
    (void)verify_exact;
    auto on_error = [&](const Status& s) -> Status {
        if (gpu_started) {
            for (uint32_t i = 0; i < R; ++i) {
                poison_if_current(pending_commits[i].sequence, pending_commits[i].handle);
            }
            return s;
        }
        return rollback_with_original_array(reservation_count, stream, s);
    };
    if (!executor.program_set_ready) {
        return on_error(Status::invalid_state("program set not ready", __FILE__, __LINE__));
    }
    ::ps::runtime::RowBucket bucket = ::ps::runtime::RowBucket::R16;
    uint32_t bucket_m = N;
    if (N != 0) {
        auto br = ::ps::runtime::resolve_row_bucket(N);
        if (!br.ok()) return on_error(br.status());
        bucket = br.value();
        bucket_m = ::ps::runtime::row_bucket_limit(bucket);
        (void)bucket_m;
    }
    auto* program = executor.program_set.resolve(bucket);
    if (program == nullptr) {
        return on_error(Status::invalid_state("program set not resolved", __FILE__, __LINE__));
    }
    uint32_t* status_ptr = executor.execution_status.data<uint32_t>();
    if (status_ptr == nullptr) {
        return on_error(Status::invalid_state(
            "execution status buffer not allocated", __FILE__, __LINE__));
    }

    BatchExecutionOutput output;
    output.num_outputs = prep.num_outputs;
    output.num_token_rows = N;
    if (N > 0) {
        auto token_hidden_slice = executor.token_hidden.slice(0, 0, N);
        if (!token_hidden_slice.ok()) return on_error(Status::invalid_argument("token_hidden slice", __FILE__, __LINE__));
        output.token_hidden = token_hidden_slice.release();
    }
    if (prep.num_outputs > 0) {
        auto hidden_slice = executor.output_hidden.slice(0, 0, prep.num_outputs);
        if (!hidden_slice.ok()) return on_error(Status::invalid_argument("final_hidden slice", __FILE__, __LINE__));
        auto logits_slice = executor.logits.slice(0, 0, prep.num_outputs);
        if (!logits_slice.ok()) return on_error(Status::invalid_argument("logits slice", __FILE__, __LINE__));
        auto sampled_slice = executor.sampled_tokens.slice(0, 0, prep.num_outputs);
        if (!sampled_slice.ok()) return on_error(Status::invalid_argument("sampled_tokens slice", __FILE__, __LINE__));
        output.final_hidden = hidden_slice.release();
        output.logits = logits_slice.release();
        output.sampled_tokens = sampled_slice.release();
    }

    uint64_t submission_id = ++executor.next_submission_id;
    if (submission_id == 0u) {
        submission_id = ++executor.next_submission_id;
    }

    PendingBatch pending;
    pending.output = std::move(output);
    pending.commits = pending_commits;
    pending.commit_count = R;
    pending.submission_id = submission_id;

    hipEvent_t done = nullptr;
    hipError_t ce2 = hipEventCreateWithFlags(&done, hipEventDisableTiming);
    if (ce2 != hipSuccess) {
        return on_error(Status::hip_error("hipEventCreate", hipGetErrorString(ce2), __FILE__, __LINE__));
    }

    const void* token_src = batch.token_ids;
    const void* ctx_src = &host_ctx;
#ifdef PHASESHIFT_HIP_GRAPH
    // Graph 用 staging は capture / replay を実際に行う時だけ通す。
    const int graph_slot_idx = graph_slot_index(bucket);
    const bool spec_verify_without_history =
        batch.speculative_verify && options.gdn_spec_history.recurrent_history == nullptr;
    const bool graph_eligible =
        !executor.graph_failed &&
        N > 0 &&
        graph_slot_idx >= 0 &&
        executor.imatrix_collector == nullptr &&
        executor.value_trace == nullptr &&
        !any_constraint &&
        !spec_verify_without_history;
    auto stage_graph_inputs = [&]() {
        if (!token_ids_on_device) {
            std::memcpy(executor.host_token_graph_staging, batch.token_ids,
                        N * sizeof(int32_t));
            token_src = executor.host_token_graph_staging;
        }
        std::memcpy(executor.host_ctx_graph_staging, &host_ctx, sizeof(host_ctx));
        ctx_src = executor.host_ctx_graph_staging;
    };
    uint32_t attention_plan_key = 0u;
    bool graph_active = graph_eligible;
    if (graph_active) {
        const bool attn_use_prefill =
            exec_role == ::ps::runtime::ExecutionRole::Prefill && R == 1u &&
            N >= runtime::kPagedAttentionPrefillMinRows;
        if (!attention_plan_key_for(
                executor, *program, N, max_visible_tokens, attn_use_prefill,
                attention_plan_key))
            graph_active = false;
    }
    const uint8_t graph_exec_role = static_cast<uint8_t>(exec_role);
    const uint8_t graph_has_spec_history =
        options.gdn_spec_history.recurrent_history != nullptr ? 1u : 0u;
    const uint32_t graph_capture_rows = options.gdn_spec_history.capture_rows;
    const void* graph_gdn_conv_history = options.gdn_spec_history.conv_history;
    const void* graph_gdn_recurrent_history = options.gdn_spec_history.recurrent_history;
#endif
    auto enqueue_batch = [&]() -> Status {
        auto stq = upload_tokens(input_ids, token_src);
        if (!stq.ok()) return stq;
        stq = upload_tokens(bctx_token_ids, token_src);
        if (!stq.ok()) return stq;
        stq = upload(bctx_requests, staging, R * sizeof(::ps::runtime::DeviceRequestDescriptor));
        if (!stq.ok()) return stq;
        if (any_constraint) {
            stq = upload(
                executor.constraint_mask_device.data<uint32_t>(),
                executor.constraint_mask_host.data(),
                static_cast<std::size_t>(output_count) * batch.constraint_mask_words *
                    sizeof(uint32_t));
            if (!stq.ok()) return stq;
        }
        if (ctx_src == &host_ctx) {
            hipError_t ctx_err = hipMemcpyWithStream(
                executor.batch_context, ctx_src, sizeof(host_ctx), hipMemcpyHostToDevice, stream);
            if (ctx_err != hipSuccess) {
                return Status::hip_error("execute_batch hipMemcpyWithStream batch_context",
                                         hipGetErrorString(ctx_err), __FILE__, __LINE__);
            }
        } else {
            stq = upload(executor.batch_context, ctx_src, sizeof(host_ctx));
            if (!stq.ok()) return stq;
        }
        stq = ::ps::runtime::launch_prepare_batch_descriptor(
            executor.batch_context, executor.config.max_scheduled_requests, stream);
        if (!stq.ok()) return stq;
        runtime::HostExecutionContext ctx{};
        stq = build_host_execution_context(
            executor, *program, status_ptr, N, prep.num_outputs,
            prep.num_sampled_outputs, prep.num_stochastic_outputs,
            min_visible_tokens, max_visible_tokens, exec_role,
            exec_numeric_mode, options.gdn_spec_history,
            any_constraint ? executor.constraint_mask_device.data<uint32_t>() : nullptr,
            any_constraint ? batch.constraint_mask_words : 0u, ctx);
        if (!stq.ok()) return stq;

        const uint32_t dispatch_count =
            static_cast<uint32_t>(program->dispatches.size());
        const hipError_t word_err =
            hipMemsetAsync(ctx.error_word, 0, sizeof(uint32_t), stream);
        if (word_err != hipSuccess) {
            return Status::hip_error("hipMemsetAsync error word",
                                     hipGetErrorString(word_err), __FILE__, __LINE__);
        }
        stq = launch_host_backend_range(executor, *program, ctx, stream, 0u,
                                        dispatch_count);
        if (!stq.ok()) return stq;
        auto* host_status = static_cast<uint32_t*>(executor.host_status_staging);
        if (host_status == nullptr) {
            return Status::invalid_state("host status staging not allocated",
                                         __FILE__, __LINE__);
        }
        const hipError_t stage_err = hipMemcpyAsync(
            host_status, executor.execution_status.data<uint32_t>(),
            sizeof(uint32_t), hipMemcpyDeviceToHost, stream);
        if (stage_err != hipSuccess) {
            return Status::hip_error("hipMemcpyAsync execution_status",
                                     hipGetErrorString(stage_err), __FILE__,
                                     __LINE__);
        }
        return Status::make_ok();
    };
    bool enqueued = false;
#ifdef PHASESHIFT_HIP_GRAPH
    if (graph_active) {
        const int gslot_idx = graph_slot_idx;
        if (gslot_idx >= 0) {
            GraphCacheSlot* gslot = &executor.graph_cache[gslot_idx];
            if (gslot->valid && gslot->token_count == N &&
                gslot->num_outputs == prep.num_outputs &&
                gslot->num_sampled == prep.num_sampled_outputs &&
                gslot->num_stochastic == prep.num_stochastic_outputs &&
                gslot->attention_plan_key == attention_plan_key &&
                gslot->verify_exact == verify_exact &&
                gslot->exec_role == graph_exec_role &&
                gslot->has_spec_history == graph_has_spec_history &&
                gslot->capture_rows == graph_capture_rows &&
                gslot->gdn_conv_history == graph_gdn_conv_history &&
                gslot->gdn_recurrent_history == graph_gdn_recurrent_history &&
                gslot->token_device_src ==
                    (token_ids_on_device ? batch.token_ids : nullptr)) {
                stage_graph_inputs();
                if (hipGraphLaunch(gslot->exec, stream) == hipSuccess) {
                    enqueued = true;
                    if (graph_debug_first(static_cast<uint32_t>(gslot_idx)))
                        std::fprintf(stderr, "[graph] replay bucket=%d N=%u\n", gslot_idx, N);
                } else {
                    if (graph_debug_enabled())
                        std::fprintf(stderr, "[graph] replay launch failed bucket=%d N=%u\n",
                                     gslot_idx, N);
                    executor.graph_failed = true;
                }
            }
            const bool graph_warm = !enqueued && !executor.graph_failed &&
                program_pool_warm(executor, *program, static_cast<uint32_t>(gslot_idx));
            if (!graph_warm && !enqueued && !executor.graph_failed &&
                graph_debug_first(static_cast<uint32_t>(gslot_idx)))
                std::fprintf(stderr,
                             "[graph] skip: staging pool not warm bucket=%d N=%u dispatches=%zu\n",
                             gslot_idx, N, program->dispatches.size());
            if (graph_warm) {
                if (gslot->valid) {
                    ps::gpu::discard_cleanup_result(hipGraphExecDestroy(gslot->exec));
                    *gslot = GraphCacheSlot{};
                }
                stage_graph_inputs();
                if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) == hipSuccess) {
                    Status cap_st = enqueue_batch();
                    hipGraph_t graph = nullptr;
                    hipError_t ec = hipStreamEndCapture(stream, &graph);
                    if (cap_st.ok() && ec == hipSuccess && graph != nullptr) {
                        hipGraphExec_t ge = nullptr;
                        hipError_t ic = hipGraphInstantiate(&ge, graph, nullptr, nullptr, 0);
                        ps::gpu::discard_cleanup_result(hipGraphDestroy(graph));
                        if (ic == hipSuccess && ge != nullptr) {
                            gslot->exec = ge;
                            gslot->token_count = N;
                            gslot->num_outputs = prep.num_outputs;
                            gslot->num_sampled = prep.num_sampled_outputs;
                            gslot->num_stochastic = prep.num_stochastic_outputs;
                            gslot->attention_plan_key = attention_plan_key;
                            gslot->verify_exact = verify_exact;
                            gslot->exec_role = graph_exec_role;
                            gslot->has_spec_history = graph_has_spec_history;
                            gslot->capture_rows = graph_capture_rows;
                            gslot->gdn_conv_history = graph_gdn_conv_history;
                            gslot->gdn_recurrent_history = graph_gdn_recurrent_history;
                            gslot->token_device_src =
                                token_ids_on_device ? batch.token_ids : nullptr;
                            gslot->valid = true;
                            if (hipGraphLaunch(ge, stream) == hipSuccess) {
                                enqueued = true;
                                if (graph_debug_enabled())
                                    std::fprintf(stderr, "[graph] capture ok bucket=%d N=%u\n",
                                                 gslot_idx, N);
                            } else {
                                if (graph_debug_enabled())
                                    std::fprintf(stderr, "[graph] capture launch failed bucket=%d N=%u\n",
                                                 gslot_idx, N);
                                executor.graph_failed = true;
                            }
                        } else {
                            if (graph_debug_enabled())
                                std::fprintf(stderr, "[graph] instantiate failed ic=%s\n",
                                             hipGetErrorString(ic));
                            executor.graph_failed = true;
                        }
                    } else {
                        if (graph_debug_enabled())
                            std::fprintf(stderr,
                                         "[graph] end capture failed ec=%s cap_ok=%d graph=%p\n",
                                         hipGetErrorString(ec), (int)cap_st.ok(), (void*)graph);
                        executor.graph_failed = true;
                    }
                } else {
                    if (graph_debug_enabled())
                        std::fprintf(stderr, "[graph] begin capture failed bucket=%d N=%u\n",
                                     gslot_idx, N);
                    executor.graph_failed = true;
                }
            }
        }
    }
#endif
    if (!enqueued) {
        Status plain_st = enqueue_batch();
        if (!plain_st.ok()) {
            ps::gpu::discard_cleanup_result(hipEventDestroy(done));
            return on_error(plain_st);
        }
    }

    executor.active_submission_id = submission_id;
    gpu_started = true;

    const Status completion_publish =
        ::ps::runtime::stream_write_value32(
            stream, executor.completion_signal,
            static_cast<uint32_t>(submission_id));
    if (!completion_publish.ok()) {
        ps::gpu::discard_cleanup_result(hipStreamSynchronize(stream));
        for (uint32_t i = 0; i < R; ++i) {
            poison_if_current(pending_commits[i].sequence,
                              pending_commits[i].handle);
        }
        executor.active_submission_id = 0u;
        ps::gpu::discard_cleanup_result(hipEventDestroy(done));
        return completion_publish;
    }
    pending.generation = submission_id;

    hipError_t re = hipEventRecord(done, stream);
    if (re != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipStreamSynchronize(stream));
        for (uint32_t i = 0; i < R; ++i) {
            poison_if_current(pending_commits[i].sequence, pending_commits[i].handle);
        }
        executor.active_submission_id = 0u;
        ps::gpu::discard_cleanup_result(hipEventDestroy(done));
        return Status::hip_error("hipEventRecord", hipGetErrorString(re), __FILE__, __LINE__);
    }

    pending.done = done;
    return pending;
}

namespace {

constexpr uint32_t kKeepAliveIntervalUs = 1000u;

struct KeepAlive {
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    std::atomic<bool> stop{false};
    std::thread thread;
};

bool keepalive_requested() {
    const char* e = std::getenv("PHASESHIFT_KEEPALIVE");
    if (e == nullptr) return true;
    return e[0] != '0';
}

KeepAlive* keepalive_start() {
    if (!keepalive_requested()) return nullptr;
    auto* ka = new (std::nothrow) KeepAlive();
    if (ka == nullptr) return nullptr;
    if (hipStreamCreate(&ka->stream) != hipSuccess) {
        delete ka;
        return nullptr;
    }
    if (hipEventCreateWithFlags(&ka->event, hipEventDisableSystemFence) != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipStreamDestroy(ka->stream));
        delete ka;
        return nullptr;
    }
    ka->thread = std::thread([ka] {
        while (!ka->stop.load(std::memory_order_relaxed)) {
            (void)hipEventRecord(ka->event, ka->stream);
            std::this_thread::sleep_for(std::chrono::microseconds(kKeepAliveIntervalUs));
        }
    });
    return ka;
}

Status keepalive_stop(KeepAlive* ka) {
    if (ka == nullptr) return Status::make_ok();
    Status first_error = Status::make_ok();
    ka->stop.store(true, std::memory_order_relaxed);
    if (ka->thread.joinable()) ka->thread.join();
    hipError_t err = hipStreamSynchronize(ka->stream);
    if (first_error.ok() && err != hipSuccess) {
        first_error = Status::hip_error("hipStreamSynchronize keepalive",
                                        hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipEventDestroy(ka->event);
    if (first_error.ok() && err != hipSuccess) {
        first_error = Status::hip_error("hipEventDestroy keepalive",
                                        hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipStreamDestroy(ka->stream);
    if (first_error.ok() && err != hipSuccess) {
        first_error = Status::hip_error("hipStreamDestroy keepalive",
                                        hipGetErrorString(err), __FILE__, __LINE__);
    }
    delete ka;
    return first_error;
}

}  // namespace

Executor::~Executor() noexcept {
    auto st = executor_shutdown(*this);
    (void)st;
}

Executor::Executor() = default;

Executor::Executor(Executor&& other) noexcept
    : model(other.model),
      sequence_pool(other.sequence_pool),
      gdn_state_pool(other.gdn_state_pool),
      kv_pool(other.kv_pool),
      config(other.config),
      batch_context_storage(std::move(other.batch_context_storage)),
      batch_context(nullptr),
      lm_head_proxy(std::move(other.lm_head_proxy)),
      host_request_staging(other.host_request_staging),
      host_staging_bytes(other.host_staging_bytes),
      host_pending_staging(other.host_pending_staging),
      host_pending_staging_bytes(other.host_pending_staging_bytes),
      host_sampled_staging(other.host_sampled_staging),
      host_sampled_staging_bytes(other.host_sampled_staging_bytes),
      host_status_staging(other.host_status_staging),
      completion_signal(other.completion_signal),
      next_submission_id(other.next_submission_id),
      active_submission_id(other.active_submission_id),
      program_set(std::move(other.program_set)),
      program_set_ready(other.program_set_ready),
      comm_self(other.comm_self),
      comm_launch(other.comm_launch),
      execution_status(std::move(other.execution_status)),
      execution_workspace(std::move(other.execution_workspace)),
      host_weight_table(other.host_weight_table),
       host_parameter_table(other.host_parameter_table),
       host_parameter_slots(std::move(other.host_parameter_slots)),
       host_model_state(other.host_model_state),
      dispatch_staging(other.dispatch_staging),
      dispatch_staging_bytes(other.dispatch_staging_bytes),
      host_dispatch_staging(other.host_dispatch_staging),
      staging_pool(std::move(other.staging_pool)),
      program_staging_meta(std::move(other.program_staging_meta)),
      graph_cache(other.graph_cache),
      host_token_graph_staging(other.host_token_graph_staging),
      host_token_graph_staging_bytes(other.host_token_graph_staging_bytes),
      host_ctx_graph_staging(other.host_ctx_graph_staging),
      graph_failed(other.graph_failed),
      input_ids(std::move(other.input_ids)),
      rope_inv_freq(std::move(other.rope_inv_freq)),
      output_hidden(std::move(other.output_hidden)),
      logits(std::move(other.logits)),
      sampled_tokens(std::move(other.sampled_tokens)),
      token_hidden(std::move(other.token_hidden)),
      dflash_target_hidden(std::move(other.dflash_target_hidden)),
      constraint_mask_device(std::move(other.constraint_mask_device)),
      constraint_mask_host(std::move(other.constraint_mask_host)),
      keepalive(other.keepalive) {
    if (batch_context_storage.device_struct.allocation_base() != nullptr) {
        batch_context = batch_context_storage.device_struct.data<::ps::runtime::DeviceBatchContext>();
    } else {
        batch_context = nullptr;
    }
    other.batch_context = nullptr;
    other.host_request_staging = nullptr;
    other.host_staging_bytes = 0;
    other.host_pending_staging = nullptr;
    other.host_pending_staging_bytes = 0;
    other.host_sampled_staging = nullptr;
    other.host_sampled_staging_bytes = 0;
    other.host_status_staging = nullptr;
    other.completion_signal = ::ps::runtime::StreamSignal{};
    other.next_submission_id = 0;
    other.active_submission_id = 0;
    other.program_set = ::ps::runtime::ProgramSet{};
    other.program_set_ready = false;
    other.host_weight_table = nullptr;
    other.host_parameter_table = nullptr;
    other.host_model_state = ::ps::kernel::ModelDispatchStateView{};
    other.dispatch_staging = nullptr;
    other.dispatch_staging_bytes = 0;
    other.host_dispatch_staging = nullptr;
    other.staging_pool = DispatchStagingPool{};
    other.program_staging_meta.clear();
    other.graph_cache = {};
    other.host_token_graph_staging = nullptr;
    other.host_token_graph_staging_bytes = 0;
    other.host_ctx_graph_staging = nullptr;
    other.graph_failed = false;
    other.execution_status = gpu::Tensor{};
    other.execution_workspace = gpu::Tensor{};
    other.model = nullptr;
    other.sequence_pool = nullptr;
    other.gdn_state_pool = nullptr;
    other.kv_pool = nullptr;
    other.batch_context_storage = ::ps::runtime::DeviceBatchContextStorage{};
    other.input_ids = gpu::Tensor{};
    other.rope_inv_freq = gpu::Tensor{};
    other.output_hidden = gpu::Tensor{};
    other.logits = gpu::Tensor{};
    other.sampled_tokens = gpu::Tensor{};
    other.token_hidden = gpu::Tensor{};
    other.dflash_target_hidden = {};
    other.constraint_mask_device = gpu::Tensor{};
    other.constraint_mask_host.clear();
    other.keepalive = nullptr;
}

Executor& Executor::operator=(Executor&& other) noexcept {
    if (this != &other) {
        auto st = executor_shutdown(*this);
        (void)st;
        model = other.model;
        sequence_pool = other.sequence_pool;
        gdn_state_pool = other.gdn_state_pool;
        kv_pool = other.kv_pool;
        config = other.config;
        batch_context_storage = std::move(other.batch_context_storage);
        if (batch_context_storage.device_struct.allocation_base() != nullptr) {
            batch_context = batch_context_storage.device_struct.data<::ps::runtime::DeviceBatchContext>();
        } else {
            batch_context = nullptr;
        }
        host_request_staging = other.host_request_staging;
        host_staging_bytes = other.host_staging_bytes;
        other.host_request_staging = nullptr;
        other.host_staging_bytes = 0;
        host_pending_staging = other.host_pending_staging;
        host_pending_staging_bytes = other.host_pending_staging_bytes;
        other.host_pending_staging = nullptr;
        other.host_pending_staging_bytes = 0;
        host_sampled_staging = other.host_sampled_staging;
        host_sampled_staging_bytes = other.host_sampled_staging_bytes;
        other.host_sampled_staging = nullptr;
        other.host_sampled_staging_bytes = 0;
        host_status_staging = other.host_status_staging;
        other.host_status_staging = nullptr;
        completion_signal = other.completion_signal;
        other.completion_signal = ::ps::runtime::StreamSignal{};
        next_submission_id = other.next_submission_id;
        active_submission_id = other.active_submission_id;
        program_set = std::move(other.program_set);
        program_set_ready = other.program_set_ready;
        host_weight_table = other.host_weight_table;
        other.host_weight_table = nullptr;
        host_parameter_table = other.host_parameter_table;
        other.host_parameter_table = nullptr;
        host_parameter_slots = std::move(other.host_parameter_slots);
        host_model_state = other.host_model_state;
        lm_head_proxy = std::move(other.lm_head_proxy);
        dispatch_staging = other.dispatch_staging;
        other.dispatch_staging = nullptr;
        dispatch_staging_bytes = other.dispatch_staging_bytes;
        other.dispatch_staging_bytes = 0;
        host_dispatch_staging = other.host_dispatch_staging;
        other.host_dispatch_staging = nullptr;
        staging_pool = std::move(other.staging_pool);
        other.staging_pool = DispatchStagingPool{};
        program_staging_meta = std::move(other.program_staging_meta);
        other.program_staging_meta.clear();
        graph_cache = other.graph_cache;
        host_token_graph_staging = other.host_token_graph_staging;
        host_token_graph_staging_bytes = other.host_token_graph_staging_bytes;
        host_ctx_graph_staging = other.host_ctx_graph_staging;
        graph_failed = other.graph_failed;
        execution_status = std::move(other.execution_status);
        execution_workspace = std::move(other.execution_workspace);
        input_ids = std::move(other.input_ids);
        rope_inv_freq = std::move(other.rope_inv_freq);
        output_hidden = std::move(other.output_hidden);
        logits = std::move(other.logits);
        sampled_tokens = std::move(other.sampled_tokens);
        token_hidden = std::move(other.token_hidden);
        dflash_target_hidden = std::move(other.dflash_target_hidden);
        constraint_mask_device = std::move(other.constraint_mask_device);
        constraint_mask_host = std::move(other.constraint_mask_host);
        other.batch_context = nullptr;
        other.host_request_staging = nullptr;
        other.host_staging_bytes = 0;
        other.host_pending_staging = nullptr;
        other.host_pending_staging_bytes = 0;
        other.next_submission_id = 0;
        other.active_submission_id = 0;
        other.program_set = ::ps::runtime::ProgramSet{};
        other.program_set_ready = false;
        other.host_weight_table = nullptr;
        other.host_parameter_table = nullptr;
        other.host_model_state = ::ps::kernel::ModelDispatchStateView{};
        other.dispatch_staging = nullptr;
        other.dispatch_staging_bytes = 0;
        other.host_dispatch_staging = nullptr;
        other.graph_cache = {};
        other.host_token_graph_staging = nullptr;
        other.host_token_graph_staging_bytes = 0;
        other.host_ctx_graph_staging = nullptr;
        other.graph_failed = false;
        other.execution_status = gpu::Tensor{};
        other.execution_workspace = gpu::Tensor{};
        other.model = nullptr;
        other.sequence_pool = nullptr;
        other.gdn_state_pool = nullptr;
        other.kv_pool = nullptr;
        other.batch_context_storage = ::ps::runtime::DeviceBatchContextStorage{};
        other.input_ids = gpu::Tensor{};
        other.rope_inv_freq = gpu::Tensor{};
        other.output_hidden = gpu::Tensor{};
        other.logits = gpu::Tensor{};
        other.sampled_tokens = gpu::Tensor{};
        other.token_hidden = gpu::Tensor{};
        other.dflash_target_hidden = {};
        other.constraint_mask_device = gpu::Tensor{};
        other.constraint_mask_host.clear();
        keepalive = other.keepalive;
        other.keepalive = nullptr;
    }
    return *this;
}

bool Executor::is_shutdown() const noexcept {
    return host_request_staging == nullptr && host_pending_staging == nullptr &&
           batch_context == nullptr &&
           batch_context_storage.device_struct.allocation_base() == nullptr;
}

Status executor_shutdown(Executor& ex) noexcept {
    Status first_error = Status::make_ok();
    if (ex.keepalive != nullptr) {
        Status ka_status = keepalive_stop(static_cast<KeepAlive*>(ex.keepalive));
        ex.keepalive = nullptr;
        if (first_error.ok() && !ka_status.ok()) first_error = ka_status;
    }
    {
        Status proxy_status = ex.lm_head_proxy.shutdown();
        if (first_error.ok() && !proxy_status.ok()) first_error = proxy_status;
    }
    if (ex.host_sampled_staging != nullptr) {
        hipError_t err = hipHostFree(ex.host_sampled_staging);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipHostFree host_sampled_staging",
                                             hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.host_sampled_staging = nullptr;
        ex.host_sampled_staging_bytes = 0;
    } else {
        ex.host_sampled_staging_bytes = 0;
    }
    {
        const Status sig_st =
            ::ps::runtime::stream_signal_free(ex.completion_signal);
        if (!sig_st.ok() && first_error.ok()) first_error = sig_st;
        ex.completion_signal = ::ps::runtime::StreamSignal{};
    }
    if (ex.host_status_staging != nullptr) {
        const hipError_t err = hipHostFree(ex.host_status_staging);
        if (err != hipSuccess && first_error.ok()) {
            first_error = Status::hip_error("hipHostFree host_status_staging",
                                            hipGetErrorString(err), __FILE__,
                                            __LINE__);
        }
        ex.host_status_staging = nullptr;
    }
    if (ex.host_weight_table != nullptr) {
        hipError_t err = hipFree(ex.host_weight_table);
        ex.host_weight_table = nullptr;
        if (first_error.ok() && err != hipSuccess)
            first_error = Status::hip_error("hipFree host_weight_table",
                                            hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (ex.host_parameter_table != nullptr) {
        hipError_t err = hipFree(ex.host_parameter_table);
        ex.host_parameter_table = nullptr;
        if (first_error.ok() && err != hipSuccess)
            first_error = Status::hip_error("hipFree host_parameter_table",
                                            hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (ex.dispatch_staging != nullptr) {
        hipError_t err = hipFree(ex.dispatch_staging);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipFree dispatch_staging",
                                            hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.dispatch_staging = nullptr;
        ex.dispatch_staging_bytes = 0;
    } else {
        ex.dispatch_staging_bytes = 0;
    }
    if (ex.host_dispatch_staging != nullptr) {
        hipError_t err = hipHostFree(ex.host_dispatch_staging);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipHostFree host_dispatch_staging",
                                            hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.host_dispatch_staging = nullptr;
    }
    if (ex.staging_pool.device != nullptr) {
        hipError_t err = hipFree(ex.staging_pool.device);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipFree staging_pool",
                                             hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.staging_pool.device = nullptr;
        ex.staging_pool.pool_bytes = 0;
    }
#ifdef PHASESHIFT_HIP_GRAPH
    for (auto& slot : ex.graph_cache) {
        if (slot.valid) {
            hipError_t err = hipGraphExecDestroy(slot.exec);
            if (first_error.ok() && err != hipSuccess) {
                first_error = Status::hip_error("hipGraphExecDestroy graph_cache",
                                                 hipGetErrorString(err), __FILE__, __LINE__);
            }
            slot = GraphCacheSlot{};
        }
    }
#endif
    if (ex.host_token_graph_staging != nullptr) {
        hipError_t err = hipHostFree(ex.host_token_graph_staging);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipHostFree host_token_graph_staging",
                                             hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.host_token_graph_staging = nullptr;
        ex.host_token_graph_staging_bytes = 0;
    }
    if (ex.host_ctx_graph_staging != nullptr) {
        hipError_t err = hipHostFree(ex.host_ctx_graph_staging);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipHostFree host_ctx_graph_staging",
                                             hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.host_ctx_graph_staging = nullptr;
    }
    for (auto& meta : ex.program_staging_meta) {
        if (meta.device_bytes > 0) {
            void* base = meta.ranges != nullptr ? meta.ranges : meta.states;
            hipError_t err = hipFree(base);
            if (first_error.ok() && err != hipSuccess) {
                first_error = Status::hip_error("hipFree program_staging_meta",
                                                 hipGetErrorString(err), __FILE__, __LINE__);
            }
            meta = ProgramStagingMeta{};
        }
    }
    ex.program_staging_meta.clear();
    if (ex.program_set_ready) {
        ex.program_set_ready = false;
    }
    if (ex.host_request_staging != nullptr) {
        hipError_t err = hipHostFree(ex.host_request_staging);
        if (err != hipSuccess) {
            first_error = Status::hip_error("hipHostFree host_request_staging", hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.host_request_staging = nullptr;
        ex.host_staging_bytes = 0;
    } else {
        ex.host_staging_bytes = 0;
    }
    if (ex.host_pending_staging != nullptr) {
        hipError_t err = hipHostFree(ex.host_pending_staging);
        if (first_error.ok() && err != hipSuccess) {
            first_error = Status::hip_error("hipHostFree host_pending_staging", hipGetErrorString(err), __FILE__, __LINE__);
        }
        ex.host_pending_staging = nullptr;
        ex.host_pending_staging_bytes = 0;
    } else {
        ex.host_pending_staging_bytes = 0;
    }
    {
        Status st = ::ps::runtime::destroy_device_batch_context(ex.batch_context_storage);
        if (first_error.ok() && !st.ok()) {
            first_error = st;
        }
        ex.batch_context = nullptr;
    }
    ex.sampled_tokens = gpu::Tensor{};
    ex.logits = gpu::Tensor{};
    ex.output_hidden = gpu::Tensor{};
    ex.token_hidden = gpu::Tensor{};
    for (auto& tap : ex.dflash_target_hidden) tap = gpu::Tensor{};
    ex.input_ids = gpu::Tensor{};
    ex.rope_inv_freq = gpu::Tensor{};
    ex.model = nullptr;
    ex.sequence_pool = nullptr;
    ex.gdn_state_pool = nullptr;
    ex.kv_pool = nullptr;
    ex.host_model_state = ::ps::kernel::ModelDispatchStateView{};
    ex.execution_status = gpu::Tensor{};
    ex.execution_workspace = gpu::Tensor{};
    ex.constraint_mask_device = gpu::Tensor{};
    ex.constraint_mask_host.clear();
    return first_error;
}

PendingBatch::~PendingBatch() noexcept {
    destroy();
}

PendingBatch::PendingBatch(PendingBatch&& other) noexcept
    : done(other.done),
      output(std::move(other.output)),
      commits(other.commits),
      commit_count(other.commit_count),
      submission_id(other.submission_id),
      generation(other.generation) {
    other.done = nullptr;
    other.commits = nullptr;
    other.commit_count = 0;
    other.submission_id = 0;
    other.generation = 0;
}

PendingBatch& PendingBatch::operator=(PendingBatch&& other) noexcept {
    if (this != &other) {
        destroy();
        done = other.done;
        output = std::move(other.output);
        commits = other.commits;
        commit_count = other.commit_count;
        submission_id = other.submission_id;
        generation = other.generation;
        other.done = nullptr;
        other.commits = nullptr;
        other.commit_count = 0;
        other.submission_id = 0;
    }
    return *this;
}

void PendingBatch::destroy() noexcept {
    if (done != nullptr) {
        ps::gpu::discard_cleanup_result(hipEventSynchronize(done));
        for (uint32_t i = 0; i < commit_count; ++i) {
            poison_if_current(commits[i].sequence, commits[i].handle);
        }
        ps::gpu::discard_cleanup_result(hipEventDestroy(done));
    }
    done = nullptr;
    commits = nullptr;
    commit_count = 0;
    submission_id = 0;
    output = BatchExecutionOutput{};
}

Result<Executor> create_model_executor(
    const Qwen35Model& model,
    SequenceSlotPool& sequence_pool,
    GdnStatePool& gdn_state_pool,
    PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const ExecutorConfig& config) {

    if (const Status backend_status = runtime::validate_decode_backend(config.backend);
        !backend_status.ok()) {
        return backend_status;
    }

    if (config.max_scheduled_tokens == 0 || config.max_scheduled_requests == 0) {
        return Status::invalid_argument(
            "max_scheduled_tokens and max_scheduled_requests must be > 0",
            __FILE__, __LINE__);
    }

    if (kv_pool.dtype() != KVCacheDType::BF16
        && kv_pool.dtype() != KVCacheDType::FP8_E4M3
        && kv_pool.dtype() != KVCacheDType::PSQ4_W32
        && kv_pool.dtype() != KVCacheDType::PSQ8_W32) {
        return Status::unsupported(
            "Qwen35 target executor supports BF16, FP8_E4M3, PSQ4_W32 or PSQ8_W32 KV only",
            __FILE__, __LINE__);
    }
    if (kv_pool.dtype() == KVCacheDType::PSQ4_W32) {
        const std::size_t total_blocks =
            static_cast<std::size_t>(kv_pool.num_attention_layers())
            * static_cast<std::size_t>(kv_pool.num_pages())
            * static_cast<std::size_t>(kv_pool.page_tokens())
            * static_cast<std::size_t>(kv_pool.kv_heads())
            * static_cast<std::size_t>(kv_pool.psq4_blocks_per_head());
        const std::size_t code_bytes = total_blocks * kPsq4CodeBytesPerBlock;
        if (kv_pool.k_psq4_code_pool().dim(0) != code_bytes
            || kv_pool.v_psq4_code_pool().dim(0) != code_bytes
            || kv_pool.k_psq4_scale_pool().dim(0) != total_blocks
            || kv_pool.v_psq4_scale_pool().dim(0) != total_blocks) {
            return Status::invalid_state(
                "PSQ4 KV pool element count mismatch", __FILE__, __LINE__);
        }
        const int device = kv_pool.k_psq4_code_pool().physical_device();
        if (device != kv_pool.v_psq4_code_pool().physical_device()
            || device != kv_pool.k_psq4_scale_pool().physical_device()
            || device != kv_pool.v_psq4_scale_pool().physical_device()
            || device != arena.device()) {
            return Status::invalid_state(
                "PSQ4 KV pools must share the executor arena device",
                __FILE__, __LINE__);
        }
    }
    if (kv_pool.dtype() == KVCacheDType::PSQ8_W32) {
        const std::size_t total_blocks =
            static_cast<std::size_t>(kv_pool.num_attention_layers())
            * static_cast<std::size_t>(kv_pool.num_pages())
            * static_cast<std::size_t>(kv_pool.page_tokens())
            * static_cast<std::size_t>(kv_pool.kv_heads())
            * static_cast<std::size_t>(kv_pool.psq8_blocks_per_head());
        const std::size_t code_bytes = total_blocks * kPsq8CodeBytesPerBlock;
        if (kv_pool.k_psq8_code_pool().dim(0) != code_bytes
            || kv_pool.v_psq8_code_pool().dim(0) != code_bytes
            || kv_pool.k_psq8_scale_pool().dim(0) != total_blocks
            || kv_pool.v_psq8_scale_pool().dim(0) != total_blocks) {
            return Status::invalid_state(
                "PSQ8 KV pool element count mismatch", __FILE__, __LINE__);
        }
        const int device = kv_pool.k_psq8_code_pool().physical_device();
        if (device != kv_pool.v_psq8_code_pool().physical_device()
            || device != kv_pool.k_psq8_scale_pool().physical_device()
            || device != kv_pool.v_psq8_scale_pool().physical_device()
            || device != arena.device()) {
            return Status::invalid_state(
                "PSQ8 KV pools must share the executor arena device",
                __FILE__, __LINE__);
        }
    }
    if (kv_pool.dtype() == KVCacheDType::FP8_E4M3) {
        const std::size_t body_elems =
            static_cast<std::size_t>(kv_pool.num_attention_layers())
            * static_cast<std::size_t>(kv_pool.num_pages())
            * static_cast<std::size_t>(kv_pool.page_tokens())
            * static_cast<std::size_t>(kv_pool.kv_heads())
            * static_cast<std::size_t>(kv_pool.head_dim());
        const std::size_t scale_elems = body_elems / kv_pool.head_dim();
        if (kv_pool.k_pool().dim(0) != body_elems
            || kv_pool.v_pool().dim(0) != body_elems
            || kv_pool.k_scale_pool().dim(0) != scale_elems
            || kv_pool.v_scale_pool().dim(0) != scale_elems) {
            return Status::invalid_state(
                "FP8 KV pool body/scale element count mismatch",
                __FILE__, __LINE__);
        }
        const int device = kv_pool.k_pool().physical_device();
        if (device != kv_pool.v_pool().physical_device()
            || device != kv_pool.k_scale_pool().physical_device()
            || device != kv_pool.v_scale_pool().physical_device()
            || device != arena.device()) {
            return Status::invalid_state(
                "FP8 KV body/scale pools must share the executor arena device",
                __FILE__, __LINE__);
        }
    }

    const Qwen35TextConfig& tc = model.text_config();
    {
        const auto expected_gdn =
            GdnStatePoolLayout::from_text_config(tc, config.tensor_parallel_size);
        const auto& actual_gdn = gdn_state_pool.layout();
        if (actual_gdn.num_gdn_states != expected_gdn.num_gdn_states ||
            actual_gdn.conv_dim != expected_gdn.conv_dim ||
            actual_gdn.conv_history != expected_gdn.conv_history ||
            actual_gdn.num_v_heads != expected_gdn.num_v_heads ||
            actual_gdn.head_k != expected_gdn.head_k ||
            actual_gdn.head_v != expected_gdn.head_v) {
            std::string msg =
                "GDN state pool geometry does not match model config: actual("
                + std::to_string(actual_gdn.num_gdn_states) + ","
                + std::to_string(actual_gdn.conv_dim) + ","
                + std::to_string(actual_gdn.conv_history) + ","
                + std::to_string(actual_gdn.num_v_heads) + ","
                + std::to_string(actual_gdn.head_k) + ","
                + std::to_string(actual_gdn.head_v) + ") expected("
                + std::to_string(expected_gdn.num_gdn_states) + ","
                + std::to_string(expected_gdn.conv_dim) + ","
                + std::to_string(expected_gdn.conv_history) + ","
                + std::to_string(expected_gdn.num_v_heads) + ","
                + std::to_string(expected_gdn.head_k) + ","
                + std::to_string(expected_gdn.head_v) + ")";
            return Status::invalid_argument(msg.c_str(), __FILE__, __LINE__);
        }
        uint32_t expected_attn_layers = 0;
        for (int lt : tc.layer_types) {
            if (lt == 1) ++expected_attn_layers;
        }
        const uint32_t expected_kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
        const uint32_t expected_head_dim = static_cast<uint32_t>(tc.attention_head_dim);
        if (kv_pool.num_attention_layers() != expected_attn_layers ||
            kv_pool.kv_heads() != expected_kv_heads ||
            kv_pool.head_dim() != expected_head_dim) {
            std::string msg =
                "KV pool geometry does not match model config: actual("
                + std::to_string(kv_pool.num_attention_layers()) + ","
                + std::to_string(kv_pool.kv_heads()) + ","
                + std::to_string(kv_pool.head_dim()) + ") expected("
                + std::to_string(expected_attn_layers) + ","
                + std::to_string(expected_kv_heads) + ","
                + std::to_string(expected_head_dim) + ")";
            return Status::invalid_argument(msg.c_str(), __FILE__, __LINE__);
        }
    }
    const std::size_t H = tc.hidden_size;
    const std::size_t V = tc.vocab_size;
    const uint32_t N = config.max_scheduled_tokens;
    const uint32_t R = config.max_scheduled_requests;
    uint32_t Npad = N;
    {
        auto br = ::ps::runtime::resolve_row_bucket(N);
        if (br.ok()) {
            const uint32_t lim = ::ps::runtime::row_bucket_limit(br.value());
            if (lim > Npad) Npad = lim;
        }
    }

    Executor ex;
    ex.model = &model;
    ex.sequence_pool = &sequence_pool;
    ex.gdn_state_pool = &gdn_state_pool;
    ex.kv_pool = &kv_pool;
    ex.config = config;
    if (config.target_hidden_tap_count > kMaxTargetHiddenTaps) {
        return Status::invalid_argument(
            "target hidden tap count exceeds capacity", __FILE__, __LINE__);
    }
    uint32_t MO = config.max_scheduled_output_rows > R
                      ? config.max_scheduled_output_rows
                      : R;
    if (MO > N) MO = N;
    ex.config.max_scheduled_output_rows = MO;

    if (runtime::target_lm_head_proxy_mode() != 0u) {
        const ps::weights::MatrixWeight& lm = model.weights().lm_head;
        if (lm.encoding == ps::weights::MatrixEncoding::Psq8 && lm.preshuffled &&
            lm.weight_scale_group == 32u) {
            runtime::LmHeadProxyWeightView view{};
            view.codes = lm.codes.data<uint8_t>();
            view.scales = lm.scales.data<uint8_t>();
            view.vocab = lm.rows;
            view.cols = lm.cols;
            view.k_padded = lm.k_padded;
            view.scale_stride_bytes = lm.storage_scale_stride_bytes;
            Status pst = ex.lm_head_proxy.init(view, MO, runtime::target_lm_head_proxy_pool(),
                                               nullptr);
            if (!pst.ok()) return pst;
        }
    }

    auto st = allocate_contiguous_tensor<int32_t>(arena, {Npad}, ex.input_ids);
    if (!st.ok()) return st;



    st = allocate_contiguous_tensor<bf16_t>(arena, {MO, H}, ex.output_hidden);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<float>(arena, {MO, V}, ex.logits);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<int32_t>(arena, {MO}, ex.sampled_tokens);
    if (!st.ok()) return st;
    if (ex.config.constraint_mask_words > 0) {
        st = allocate_contiguous_tensor<uint32_t>(
            arena, {MO, ex.config.constraint_mask_words}, ex.constraint_mask_device);
        if (!st.ok()) return st;
        ex.constraint_mask_host.assign(
            static_cast<std::size_t>(MO) * ex.config.constraint_mask_words, 0xFFFFFFFFu);
    }
    st = allocate_contiguous_tensor<bf16_t>(arena, {Npad, H}, ex.token_hidden);
    if (!st.ok()) return st;
    for (uint32_t t = 0; t < config.target_hidden_tap_count; ++t) {
        st = allocate_contiguous_tensor<bf16_t>(
            arena, {Npad, H}, ex.dflash_target_hidden[t]);
        if (!st.ok()) return st;
    }


    auto bctx_res = ::ps::runtime::create_device_batch_context(arena, R, Npad);
    if (!bctx_res.ok()) return bctx_res.status();
    ex.batch_context_storage = bctx_res.release();
    ex.batch_context = ex.batch_context_storage.device_struct.data<::ps::runtime::DeviceBatchContext>();

    std::size_t staging_bytes = static_cast<std::size_t>(R) * sizeof(::ps::runtime::DeviceRequestDescriptor);
    void* host_ptr = nullptr;
    hipError_t herr = hipHostMalloc(&host_ptr, staging_bytes);
    if (herr != hipSuccess) {
        return Status::hip_error("hipHostMalloc host_request_staging", hipGetErrorString(herr), __FILE__, __LINE__);
    }
    ex.host_request_staging = host_ptr;
    ex.host_staging_bytes = staging_bytes;

    std::size_t pending_bytes = static_cast<std::size_t>(R) * sizeof(PendingRequestCommit);
    void* pending_host = nullptr;
    hipError_t perr = hipHostMalloc(&pending_host, pending_bytes);
    if (perr != hipSuccess) {
        return Status::hip_error("hipHostMalloc host_pending_staging", hipGetErrorString(perr), __FILE__, __LINE__);
    }
    ex.host_pending_staging = pending_host;
    ex.host_pending_staging_bytes = pending_bytes;

    {
        const std::size_t sampled_bytes =
            static_cast<std::size_t>(R) * sizeof(int32_t);
        void* sampled_host = nullptr;
        hipError_t smerr = hipHostMalloc(&sampled_host, sampled_bytes);
        if (smerr != hipSuccess) {
            return Status::hip_error("hipHostMalloc host_sampled_staging",
                                       hipGetErrorString(smerr), __FILE__, __LINE__);
        }
        ex.host_sampled_staging = sampled_host;
        ex.host_sampled_staging_bytes = sampled_bytes;
    }

    {
        void* status_host = nullptr;
        hipError_t sterr = hipHostMalloc(&status_host, sizeof(uint32_t));
        if (sterr != hipSuccess) {
            return Status::hip_error("hipHostMalloc host_status_staging",
                                     hipGetErrorString(sterr), __FILE__, __LINE__);
        }
        *static_cast<uint32_t*>(status_host) = 0u;
        ex.host_status_staging = status_host;
    }
    {
        auto completion = ::ps::runtime::stream_signal_alloc(64u);
        if (!completion.ok()) return completion.status();
        ex.completion_signal = completion.value();
    }

#ifdef PHASESHIFT_HIP_GRAPH
    {
        const std::size_t token_graph_bytes =
            static_cast<std::size_t>(N) * sizeof(int32_t);
        void* token_graph_host = nullptr;
        hipError_t tgrerr = hipHostMalloc(&token_graph_host, token_graph_bytes);
        if (tgrerr != hipSuccess) {
            return Status::hip_error("hipHostMalloc host_token_graph_staging",
                                       hipGetErrorString(tgrerr), __FILE__, __LINE__);
        }
        ex.host_token_graph_staging = token_graph_host;
        ex.host_token_graph_staging_bytes = token_graph_bytes;
    }
    {
        void* ctx_graph_host = nullptr;
        hipError_t cgrerr = hipHostMalloc(&ctx_graph_host,
            sizeof(::ps::runtime::DeviceBatchContext));
        if (cgrerr != hipSuccess) {
            return Status::hip_error("hipHostMalloc host_ctx_graph_staging",
                                       hipGetErrorString(cgrerr), __FILE__, __LINE__);
        }
        ex.host_ctx_graph_staging =
            static_cast<::ps::runtime::DeviceBatchContext*>(ctx_graph_host);
    }
#endif

    {
        ::ps::kernel::ModelDispatchStateView ms{};
        ms.block_table = sequence_pool.device_view();
        ms.gdn = gdn_state_pool.device_view();
        if (kv_pool.dtype() == KVCacheDType::FP8_E4M3) {
            ms.kv_fp8 = make_paged_kv_fp8_view(kv_pool);
        } else if (kv_pool.dtype() == KVCacheDType::PSQ4_W32) {
            ms.kv_psq4 = make_paged_kv_psq4_view(kv_pool);
        } else if (kv_pool.dtype() == KVCacheDType::PSQ8_W32) {
            ms.kv_psq8 = make_paged_kv_psq8_view(kv_pool);
        } else {
            ms.kv_bf16 = make_paged_kv_bf16_view(kv_pool);
        }
        ms.kv_dtype = static_cast<uint32_t>(kv_pool.dtype());
        ex.host_model_state = ms;
    }

    st = allocate_contiguous_tensor<uint32_t>(arena, {1}, ex.execution_status);
    if (!st.ok()) return st;
    {
        uint32_t zero = 0;
        hipError_t ce = hipMemcpy(ex.execution_status.data<uint32_t>(), &zero, sizeof(uint32_t), hipMemcpyHostToDevice);
        if (ce != hipSuccess) {
            return Status::hip_error("hipMemcpy execution_status init", hipGetErrorString(ce), __FILE__, __LINE__);
        }
    }

    {
        hipStream_t upload_stream = nullptr;
        hipError_t herr2 = hipStreamCreate(&upload_stream);
        if (herr2 != hipSuccess) {
            return Status::hip_error("hipStreamCreate program upload", hipGetErrorString(herr2), __FILE__, __LINE__);
        }
        ::ps::runtime::ProgramSet built_set;
        Qwen35LowerOptions lower_options;
        lower_options.hidden_taps = std::span<const uint32_t>(
            ex.config.target_hidden_taps.data(), ex.config.target_hidden_tap_count);
        lower_options.partition = ex.config.partition;
        lower_options.pipeline_peer_rank = ex.config.pipeline_peer_rank;
        lower_options.tensor_parallel_size = ex.config.tensor_parallel_size;
        lower_options.tensor_parallel_rank = ex.config.tensor_parallel_rank;
        lower_options.tp_full_attention = ex.config.tp_full_attention;
        lower_options.tp_linear_attention = ex.config.tp_linear_attention;
        lower_options.tp_mlp = ex.config.tp_mlp;
        auto lower_res =
            lower_qwen35_to_primitives(model.text_config(), model.weights(), lower_options);
        if (!lower_res.ok()) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
            return lower_res.status();
        }
        auto lowered = lower_res.release();
        std::vector<::ps::runtime::WeightSlot> weight_slots =
            std::move(lowered.weights);
        std::vector<::ps::runtime::StaticParameterSlot> param_slots =
            std::move(lowered.parameters);
        ex.host_parameter_slots = param_slots;
        ::ps::runtime::PrimitiveGraph lowered_graph = std::move(lowered.graph);
        ::ps::runtime::WeightTableView weight_view{nullptr, 0};
        weight_view.slots = weight_slots.data();
        weight_view.count = weight_slots.size();
        {
            ::ps::runtime::ProgramBuildOptions opts;
            opts.validate_workspace = true;
            opts.max_token_rows = Npad;
            opts.max_output_rows = MO;
            if (const char* nwr = std::getenv("PSQ_NO_WORKSPACE_REUSE")) {
                if (nwr[0] != '\0') opts.no_workspace_reuse = true;
            }
            ::ps::runtime::StaticParameterTableView pview;
            pview.slots = param_slots.data();
            pview.count = param_slots.size();
            auto set_res = ::ps::runtime::build_program_set(
                lowered_graph, weight_view, pview,
                ::ps::runtime::ExecutionClass::DECODE, opts);
            if (!set_res.ok()) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return set_res.status();
            }
            built_set = set_res.release();

            std::size_t max_workspace_bytes = 0;
            uint64_t max_staging_bytes = 0;
            for (auto& prog : built_set.programs) {
                max_workspace_bytes =
                    std::max(max_workspace_bytes, prog.workspace.total_bytes);
                max_staging_bytes =
                    std::max(max_staging_bytes, ::ps::runtime::max_host_staging_bytes(prog));
            }
            constexpr std::size_t kScratchBytes = 8192 * sizeof(float);
            const std::size_t exec_workspace_bytes =
                max_workspace_bytes + kScratchBytes + runtime::kDecodeAttnPartialBytes;
            if (exec_workspace_bytes > 0) {
                st = allocate_contiguous_tensor<uint8_t>(
                    arena, {exec_workspace_bytes}, ex.execution_workspace);
                if (!st.ok()) {
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return st;
                }
                hipError_t werr = hipMemsetAsync(
                    ex.execution_workspace.data<uint8_t>(), 0, exec_workspace_bytes, upload_stream);
                if (werr != hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("hipMemsetAsync execution_workspace",
                                             hipGetErrorString(werr), __FILE__, __LINE__);
                }
            }

            st = allocate_contiguous_tensor<double>(arena, {32}, ex.rope_inv_freq);
            if (!st.ok()) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return st;
            }
            {
                float rope_theta = model.text_config().rope_theta;
                if (rope_theta <= 0.0f) rope_theta = 10000000.0f;
                hipError_t rerr = ::ps::kernel::launch_rope_inv_freq_init(
                    ex.rope_inv_freq.data<double>(), 32, rope_theta, upload_stream);
                if (rerr != hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("rope inv_freq init launch",
                                              hipGetErrorString(rerr), __FILE__, __LINE__);
                }
            }

            auto wtable_res = runtime::upload_host_weight_table(
                weight_view, &ex.host_weight_table, upload_stream);
            if (!wtable_res.ok()) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return wtable_res;
            }

            {
                auto pres = runtime::upload_host_parameter_table(
                    pview, &ex.host_parameter_table, upload_stream);
                if (!pres.ok()) {
                    if (ex.host_weight_table != nullptr) {
                        ps::gpu::discard_cleanup_result(hipFree(ex.host_weight_table));
                        ex.host_weight_table = nullptr;
                    }
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return pres;
                }
            }

            {
                constexpr size_t kMetaBucketCount =
                    ::ps::runtime::ProgramSet::kRowBucketCount;
                ex.program_staging_meta.resize(kMetaBucketCount);
                bool meta_upload_ok = true;
                for (size_t b = 0; b < kMetaBucketCount && meta_upload_ok; ++b) {
                    auto& prog = built_set.programs[b];
                    const uint64_t rc = prog.workspace.ranges.size();
                    const uint64_t sc = prog.states.size();
                    const uint64_t mbytes =
                        ::ps::runtime::program_staging_meta_bytes(
                            static_cast<uint32_t>(rc), static_cast<uint32_t>(sc));
                    auto& meta = ex.program_staging_meta[b];
                    meta.range_count = static_cast<uint32_t>(rc);
                    meta.state_count = static_cast<uint32_t>(sc);
                    if (mbytes == 0) continue;
                    void* mb = nullptr;
                    hipError_t merr = hipMalloc(&mb, mbytes);
                    if (merr != hipSuccess) {
                        meta_upload_ok = false;
                        break;
                    }
                    uint8_t* p = static_cast<uint8_t*>(mb);
                    if (rc > 0) {
                        std::vector<::ps::runtime::DeviceWorkspaceRange> hr(rc);
                        for (uint64_t i = 0; i < rc; ++i) {
                            hr[i].offset = prog.workspace.ranges[i].offset;
                            hr[i].bytes = prog.workspace.ranges[i].bytes;
                        }
                        merr = hipMemcpy(p, hr.data(),
                                         rc * sizeof(::ps::runtime::DeviceWorkspaceRange),
                                         hipMemcpyHostToDevice);
                        if (merr != hipSuccess) {
                            ps::gpu::discard_cleanup_result(hipFree(mb));
                            meta_upload_ok = false;
                            break;
                        }
                        p += (rc * sizeof(::ps::runtime::DeviceWorkspaceRange) + 15ull) & ~15ull;
                    }
                    if (sc > 0) {
                        std::vector<::ps::runtime::DeviceStateBinding> hs(sc);
                        for (uint64_t i = 0; i < sc; ++i) {
                            hs[i].slot = prog.states[i].slot;
                            hs[i].kind = static_cast<uint8_t>(prog.states[i].kind);
                            hs[i].state_index = prog.states[i].state_index;
                        }
                        merr = hipMemcpy(p, hs.data(),
                                         sc * sizeof(::ps::runtime::DeviceStateBinding),
                                         hipMemcpyHostToDevice);
                        if (merr != hipSuccess) {
                            ps::gpu::discard_cleanup_result(hipFree(mb));
                            meta_upload_ok = false;
                            break;
                        }
                    }
                    const uint64_t ranges_bytes =
                        (rc * sizeof(::ps::runtime::DeviceWorkspaceRange) + 15ull) & ~15ull;
                    meta.ranges = (rc > 0) ? mb : nullptr;
                    meta.states = (rc > 0)
                        ? (static_cast<uint8_t*>(mb) + ranges_bytes)
                        : ((sc > 0) ? mb : nullptr);
                    meta.device_bytes = mbytes;
                }
                if (!meta_upload_ok) {
                    for (auto& meta : ex.program_staging_meta) {
                        if (meta.device_bytes > 0) {
                            ps::gpu::discard_cleanup_result(hipFree(meta.ranges != nullptr ? meta.ranges : meta.states));
                            meta = ProgramStagingMeta{};
                        }
                    }
                    ex.program_staging_meta.clear();
                    if (ex.host_parameter_table != nullptr) {
                        ps::gpu::discard_cleanup_result(hipFree(ex.host_parameter_table));
                        ex.host_parameter_table = nullptr;
                    }
                    if (ex.host_weight_table != nullptr) {
                        ps::gpu::discard_cleanup_result(hipFree(ex.host_weight_table));
                        ex.host_weight_table = nullptr;
                    }
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("program staging meta upload",
                                              hipGetErrorString(hipGetLastError()),
                                              __FILE__, __LINE__);
                }
            }

            if (max_staging_bytes > 0) {
                hipError_t serr2 = hipMalloc(&ex.dispatch_staging, max_staging_bytes);
                if (serr2 != hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("hipMalloc dispatch_staging",
                                             hipGetErrorString(serr2), __FILE__, __LINE__);
                }
                ex.dispatch_staging_bytes = max_staging_bytes;
                void* host_stage = nullptr;
                hipError_t herr3 = hipHostMalloc(&host_stage, max_staging_bytes);
                if (herr3 != hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipFree(ex.dispatch_staging));
                    ex.dispatch_staging = nullptr;
                    ex.dispatch_staging_bytes = 0;
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("hipHostMalloc host_dispatch_staging",
                                             hipGetErrorString(herr3), __FILE__, __LINE__);
                }
                ex.host_dispatch_staging = host_stage;
            }

            {
                constexpr size_t kBucketCount =
                    ::ps::runtime::ProgramSet::kRowBucketCount;
                uint32_t max_bindings = 0;
                for (size_t b = 0; b < kBucketCount; ++b) {
                    max_bindings = std::max<uint32_t>(
                        max_bindings, static_cast<uint32_t>(
                            built_set.programs[b].dispatches.size()));
                }
                if (max_bindings > 0) {
                    ex.staging_pool.max_bindings = max_bindings;
                    const size_t nslots = kBucketCount * max_bindings;
                    ex.staging_pool.offsets.assign(nslots, 0);
                    ex.staging_pool.bytes.assign(nslots, 0);
                    ex.staging_pool.uploaded.assign(nslots, 0);
                    ex.staging_pool.run_seen.assign(nslots, 0);
                    ex.staging_pool.bucket_ran.assign(kBucketCount, 0);
                    uint64_t cur = 0;
                    for (size_t b = 0; b < kBucketCount; ++b) {
                        auto& prog = built_set.programs[b];
                        for (size_t i = 0; i < prog.dispatches.size(); ++i) {
                            const uint64_t need =
                                ::ps::runtime::staging_bytes_for(prog.dispatches[i]);
                            const uint64_t slot =
                                ex.staging_pool.slot(static_cast<uint32_t>(b),
                                                     static_cast<uint32_t>(i));
                            ex.staging_pool.offsets[slot] = cur;
                            ex.staging_pool.bytes[slot] = static_cast<uint32_t>(need);
                            cur += (need + 15ull) & ~15ull;
                        }
                    }
                    if (cur > 0) {
                        hipError_t perr = hipMalloc(&ex.staging_pool.device, cur);
                        if (perr != hipSuccess) {
                            if (ex.dispatch_staging != nullptr) {
                                ps::gpu::discard_cleanup_result(hipFree(ex.dispatch_staging));
                                ex.dispatch_staging = nullptr;
                                ex.dispatch_staging_bytes = 0;
                            }
                            if (ex.host_dispatch_staging != nullptr) {
                                ps::gpu::discard_cleanup_result(hipHostFree(ex.host_dispatch_staging));
                                ex.host_dispatch_staging = nullptr;
                            }
                            ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                            return Status::hip_error("hipMalloc staging_pool",
                                                     hipGetErrorString(perr), __FILE__, __LINE__);
                        }
                        ex.staging_pool.pool_bytes = cur;
                    }
                }
            }
        }

        ex.program_set = std::move(built_set);
        ex.program_set_ready = true;
        hipError_t serr = hipStreamSynchronize(upload_stream);
        if (serr != hipSuccess) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
            return Status::hip_error("hipStreamSynchronize program upload",
                                      hipGetErrorString(serr), __FILE__, __LINE__);
        }
        ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
    }

    ex.keepalive = keepalive_start();

    return ex;
}

Result<BatchExecutionOutput> complete_batch(
    Executor& executor,
    PendingBatch&& pending,
    hipStream_t stream) {
    (void)stream;
    if (pending.done == nullptr) {
        return Status::invalid_state("pending has no completion", __FILE__, __LINE__);
    }
    if (pending.submission_id == 0u ||
        executor.active_submission_id != pending.submission_id) {
        return Status::invalid_state(
            "pending batch does not belong to active executor submission",
            __FILE__, __LINE__);
    }
    auto poison_pending = [&]() noexcept {
        for (uint32_t i = 0; i < pending.commit_count; ++i) {
            poison_if_current(pending.commits[i].sequence, pending.commits[i].handle);
        }
    };
    hipEvent_t done = pending.done;
    if (done == nullptr) {
        return Status::invalid_state("pending has no completion event", __FILE__, __LINE__);
    }
    hipError_t se = hipStreamSynchronize(stream);
    if (se != hipSuccess) {
        poison_pending();
        executor.active_submission_id = 0u;
        pending.done = nullptr;
        return Status::hip_error("hipEventSynchronize", hipGetErrorString(se), __FILE__, __LINE__);
    }
    hipError_t de = hipEventDestroy(done);
    pending.done = nullptr;
    if (de != hipSuccess) {
        poison_pending();
        executor.active_submission_id = 0u;
        return Status::hip_error("hipEventDestroy", hipGetErrorString(de), __FILE__, __LINE__);
    }

    auto* host_status = static_cast<uint32_t*>(executor.host_status_staging);
    if (host_status == nullptr) {
        return Status::invalid_state("host status staging not allocated", __FILE__,
                                     __LINE__);
    }
    const uint32_t status = *host_status;
    if (status != static_cast<uint32_t>(::ps::runtime::ProgramStatus::COMPLETE)) {
        poison_pending();
        executor.active_submission_id = 0u;
        std::string message = "host program execution failed: status=";
        message += std::to_string(status);
        return Status::invalid_state(message.c_str(), __FILE__, __LINE__);
    }

    for (uint32_t i = 0; i < pending.commit_count; ++i) {
        const PendingRequestCommit& c = pending.commits[i];
        if (c.sequence == nullptr) {
            poison_pending();
            executor.active_submission_id = 0u;
            return Status::invalid_state("pending request sequence is null", __FILE__, __LINE__);
        }
        auto st = validate_sequence_append_commit(*c.sequence, c.handle, c.committed_tokens);
        if (!st.ok()) {
            poison_pending();
            executor.active_submission_id = 0u;
            return st;
        }
    }

    for (uint32_t i = 0; i < pending.commit_count; ++i) {
        const PendingRequestCommit& c = pending.commits[i];
        auto st = commit_sequence_append(*c.sequence, c.handle, c.committed_tokens);
        if (!st.ok()) {
            poison_pending();
            executor.active_submission_id = 0u;
            return st;
        }
    }

    executor.active_submission_id = 0u;
    BatchExecutionOutput out = std::move(pending.output);
    pending.commits = nullptr;
    pending.commit_count = 0;
    pending.submission_id = 0;
    pending.output = BatchExecutionOutput{};
    return out;
}

Result<PendingBatch> submit_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream) {
    ExecuteBatchOptions options;
    return submit_co_batch(
        executor, batch, stream, options);
}

Result<PendingBatch> submit_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options) {
    return submit_co_batch(
        executor, batch, stream, options);
}

Result<BatchExecutionOutput> execute_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream) {
    ExecuteBatchOptions options;
    return execute_batch(executor, batch, stream, options);
}

Result<BatchExecutionOutput> execute_batch(
    Executor& executor,
    const ScheduledBatch& batch,
    hipStream_t stream,
    const ExecuteBatchOptions& options) {
    auto pr = submit_batch(executor, batch, stream, options);
    if (!pr.ok()) return pr.status();
    PendingBatch pending = pr.release();
    return complete_batch(executor, std::move(pending), stream);
}


DecodeStatus poll_decode(const Executor& executor, uint64_t generation) noexcept {
    if (generation == 0u) return DecodeStatus::Pending;
    const uint32_t published = ::ps::runtime::stream_signal_load(
        executor.completion_signal);
    if (published != static_cast<uint32_t>(generation)) {
        return DecodeStatus::Pending;
    }
    const auto* host_status =
        static_cast<const uint32_t*>(executor.host_status_staging);
    if (host_status == nullptr) return DecodeStatus::Error;
    return *host_status ==
                   static_cast<uint32_t>(::ps::runtime::ProgramStatus::COMPLETE)
               ? DecodeStatus::Complete
               : DecodeStatus::Error;
}

}
}
