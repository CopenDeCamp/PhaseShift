#include "compute_runtime.h"

#include <phaseshift/runtime/execution/row_bucket.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <utility>

namespace ps {
namespace app {

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt == 1) ++n;
    }
    return n;
}

}  // namespace

Result<std::unique_ptr<Qwen35ComputeRuntime>> Qwen35ComputeRuntime::create(
    const Qwen35RuntimeConfig& config) {
    std::unique_ptr<Qwen35ComputeRuntime> runtime(new Qwen35ComputeRuntime());
    Status status = runtime->initialize(config);
    if (!status.ok()) {
        return status;
    }
    return runtime;
}

Qwen35ComputeRuntime::~Qwen35ComputeRuntime() noexcept {
    if (!shutdown_) {
        auto st = shutdown();
        (void)st;
    }
    if (stream_ != nullptr) {
        (void)hipStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

Status Qwen35ComputeRuntime::initialize(const Qwen35RuntimeConfig& config) {
    config_ = config;
    if (config_.max_seq_len == 0) {
        return Status::invalid_argument("max_seq_len must be positive", __FILE__, __LINE__);
    }
    if (config_.max_concurrent_requests == 0) {
        return Status::invalid_argument(
            "max_concurrent_requests must be positive", __FILE__, __LINE__);
    }

    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        return Status::invalid_state("no GPU available", __FILE__, __LINE__);
    }
    if (config_.device < 0 || config_.device >= dev_count) {
        return Status::out_of_range("device out of range", __FILE__, __LINE__);
    }

    hipError_t herr = hipSetDevice(config_.device);
    if (herr != hipSuccess) {
        return Status::hip_error("hipSetDevice", hipGetErrorString(herr), __FILE__, __LINE__);
    }

    herr = hipStreamCreate(&stream_);
    if (herr != hipSuccess) {
        return Status::hip_error("hipStreamCreate", hipGetErrorString(herr), __FILE__, __LINE__);
    }

    auto arena_result = ps::gpu::GpuArena::create(config_.device, config_.arena_bytes);
    if (!arena_result.ok()) {
        return arena_result.status();
    }
    arena_.emplace(arena_result.release());

    qwen35::Qwen35LoadOptions options;
    options.verify_quantized_payload_crc = config_.verify_weights;

    const auto t_load_start = std::chrono::steady_clock::now();
    auto model_result = qwen35::Qwen35Model::load_from_safetensors(
        config_.model_dir, *arena_, stream_, options);
    if (!model_result.ok()) {
        return model_result.status();
    }
    model_ = std::make_unique<qwen35::Qwen35Model>(model_result.release());
    const auto t_load_end = std::chrono::steady_clock::now();
    model_load_ms_ =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    const auto& text_config = model_->text_config();
    const qwen35::StopTokens& generation_stop_tokens = text_config.stop_tokens;
    const uint32_t kv_heads = static_cast<uint32_t>(text_config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(text_config.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(text_config);
    const auto gdn_layout = qwen35::GdnStatePoolLayout::from_text_config(text_config);

    const uint32_t max_seq = config_.max_seq_len;
    const uint32_t max_blocks_per_sequence =
        ceil_div(max_seq + 1, config_.page_tokens) + 1;
    const uint32_t physical_capacity_tokens =
        config_.kv_cache_capacity_tokens == 0
            ? max_seq + 1
            : config_.kv_cache_capacity_tokens;
    if (physical_capacity_tokens < max_seq + 1) {
        return Status::invalid_argument(
            "kv_cache_capacity_tokens must be at least max_seq_len + 1",
            __FILE__, __LINE__);
    }
    const uint32_t kv_pool_pages =
        ceil_div(physical_capacity_tokens, config_.page_tokens) + 1;
    const uint32_t max_concurrent = config_.max_concurrent_requests;

    auto slot_pool_result =
        qwen35::SequenceSlotPool::create(*arena_, max_concurrent, max_blocks_per_sequence);
    if (!slot_pool_result.ok()) {
        return slot_pool_result.status();
    }
    slot_pool_.emplace(std::move(slot_pool_result.release()));

    auto gdn_pool_result = qwen35::GdnStatePool::create(*arena_, max_concurrent, gdn_layout);
    if (!gdn_pool_result.ok()) {
        return gdn_pool_result.status();
    }
    gdn_pool_.emplace(std::move(gdn_pool_result.release()));

    auto kv_pool_result = qwen35::PagedKVPool::create(
        *arena_, kv_pool_pages, config_.page_tokens, num_attn_layers, kv_heads, head_dim,
        config_.kv_cache_dtype);
    if (!kv_pool_result.ok()) {
        return kv_pool_result.status();
    }
    kv_pool_.emplace(std::move(kv_pool_result.release()));

    exec_max_tokens_ = config_.max_scheduled_tokens;
    if (exec_max_tokens_ == 0) {
        exec_max_tokens_ = max_seq;
    }
    const uint32_t row_bucket_max = ps::runtime::row_bucket_limit(
        ps::runtime::RowBucket::R2048);
    if (exec_max_tokens_ > row_bucket_max) {
        exec_max_tokens_ = row_bucket_max;
    }
    if (exec_max_tokens_ == 0) {
        exec_max_tokens_ = 1;
    }

    qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = exec_max_tokens_;
    exec_config.max_scheduled_requests = max_concurrent;
    exec_config.max_scheduled_output_rows = config_.max_scheduled_output_rows;
    exec_config.target_hidden_taps = config_.target_hidden_taps;
    exec_config.target_hidden_tap_count = config_.target_hidden_tap_count;

    auto exec_result = qwen35::create_model_executor(
        *model_, *slot_pool_, *gdn_pool_, *kv_pool_, *arena_, exec_config);
    if (!exec_result.ok()) {
        return exec_result.status();
    }
    executor_ = std::move(exec_result.release());

    qwen35::runtime::ContinuousBatcherConfig batcher_config;
    batcher_config.max_scheduled_tokens = exec_config.max_scheduled_tokens;
    batcher_config.max_scheduled_requests = exec_config.max_scheduled_requests;
    batcher_config.max_seq_len = max_seq;
    batcher_config.page_tokens = config_.page_tokens;
    batcher_config.eos_token_ids = text_config.stop_tokens;

    batcher_ = std::make_unique<qwen35::runtime::ContinuousBatcher>(
        executor_, *slot_pool_, *gdn_pool_, *kv_pool_, batcher_config, stream_);

    return Status::make_ok();
}

Result<uint64_t> Qwen35ComputeRuntime::submit(
    std::vector<int32_t> input_ids,
    uint32_t max_new_tokens,
    const qwen35::runtime::SamplingConfig& sampling) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->submit(std::move(input_ids), max_new_tokens, sampling);
}

Result<qwen35::runtime::StepResult> Qwen35ComputeRuntime::step() {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->step();
}

bool Qwen35ComputeRuntime::has_pending() const {
    return batcher_ != nullptr && batcher_->has_pending();
}

const qwen35::runtime::RuntimeRequest* Qwen35ComputeRuntime::request(uint64_t id) const {
    if (batcher_ == nullptr) {
        return nullptr;
    }
    return batcher_->request(id);
}

Status Qwen35ComputeRuntime::cancel_all() {
    if (batcher_ == nullptr) {
        return Status::make_ok();
    }
    return batcher_->cancel_all();
}

Status Qwen35ComputeRuntime::cancel(uint64_t id) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->cancel(id);
}

Status Qwen35ComputeRuntime::retire(uint64_t id) {
    if (batcher_ == nullptr) {
        return Status::invalid_state("runtime not initialized", __FILE__, __LINE__);
    }
    return batcher_->retire(id);
}

Status Qwen35ComputeRuntime::shutdown() {
    if (shutdown_) {
        return Status::make_ok();
    }
    Status first_error = Status::make_ok();

    if (batcher_ != nullptr) {
        Status st = batcher_->cancel_all();
        if (!st.ok() && first_error.ok()) first_error = st;
    }

    auto exec_st = qwen35::executor_shutdown(executor_);
    if (!exec_st.ok() && first_error.ok()) first_error = exec_st;

    if (stream_ != nullptr) {
        hipError_t herr = hipStreamDestroy(stream_);
        if (herr != hipSuccess && first_error.ok()) {
            first_error = Status::hip_error(
                "hipStreamDestroy", hipGetErrorString(herr), __FILE__, __LINE__);
        }
        stream_ = nullptr;
    }

    if (arena_.has_value()) {
        auto arena_st = arena_->shutdown();
        if (!arena_st.ok() && first_error.ok()) first_error = arena_st;
    }

    shutdown_ = true;
    return first_error;
}

}  // namespace app
}  // namespace ps
