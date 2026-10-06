#pragma once

#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/continuous_batcher.h>
#include <phaseshift/models/qwen35/runtime/decode_backend.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/gpu_mcu_runtime.h>
#include <phaseshift/models/qwen35/runtime/sampling_params.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/model_source.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ps {
namespace app {

struct Qwen35RuntimeConfig {
    std::string model_dir;
    std::string dflash2_model_dir;
    std::optional<std::string> model_host_socket;
    uint32_t max_seq_len = 512;
    uint32_t max_scheduled_tokens = 0;
    uint32_t max_concurrent_requests = 1;
    uint32_t kv_cache_capacity_tokens = 0;
    std::size_t arena_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    uint32_t page_tokens = 16;
    int device = 0;
    qwen35::KVCacheDType kv_cache_dtype = qwen35::KVCacheDType::BF16;
    bool verify_weights = false;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, qwen35::kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    qwen35::runtime::DecodeBackend decode_backend =
        qwen35::runtime::DecodeBackend::Host;
};

class Qwen35ComputeRuntime {
 public:
    static Result<std::unique_ptr<Qwen35ComputeRuntime>> create(
        const Qwen35RuntimeConfig& config);

    Qwen35ComputeRuntime(const Qwen35ComputeRuntime&) = delete;
    Qwen35ComputeRuntime& operator=(const Qwen35ComputeRuntime&) = delete;
    ~Qwen35ComputeRuntime() noexcept;

    Result<uint64_t> submit(
        std::vector<int32_t> input_ids,
        uint32_t max_new_tokens,
        const qwen35::runtime::SamplingConfig& sampling);

    Result<qwen35::runtime::StepResult> step();

    bool has_pending() const;

    const qwen35::runtime::RuntimeRequest* request(uint64_t id) const;

    Status cancel(uint64_t id);

    Status retire(uint64_t id);

    Status cancel_all();

    Status shutdown();

    bool is_shutdown() const noexcept { return shutdown_; }

    const Qwen35RuntimeConfig& config() const noexcept { return config_; }

    const qwen35::Qwen35Model& model() const { return *model_; }

    qwen35::Executor& executor() { return executor_; }

    qwen35::SequenceSlotPool& slot_pool() { return *slot_pool_; }

    qwen35::GdnStatePool& gdn_pool() { return *gdn_pool_; }

    qwen35::PagedKVPool& kv_pool() { return *kv_pool_; }

    gpu::GpuArena& arena() { return *arena_; }

    hipStream_t stream() const noexcept { return stream_; }

    double model_load_ms() const noexcept { return model_load_ms_; }

    bool resident_model() const noexcept
    {
        return source_ != nullptr && source_->resident();
    }

    std::size_t resident_bytes() const noexcept
    {
        return source_ == nullptr ? 0 : source_->persistent_bytes();
    }

    Result<qwen35::dflash2::DFlash2Weights> load_dflash2_weights(
        const qwen35::dflash2::DFlash2Config& config,
        const ps::weights::WeightLoadOptions& options = {});

    uint32_t max_scheduled_tokens() const noexcept { return exec_max_tokens_; }

 private:
    Qwen35ComputeRuntime() = default;

    Status initialize(const Qwen35RuntimeConfig& config);

    Qwen35RuntimeConfig config_;
    uint32_t exec_max_tokens_ = 0;
    double model_load_ms_ = 0.0;
    bool shutdown_ = false;

    hipStream_t stream_ = nullptr;
    std::optional<gpu::GpuArena> arena_;
    std::unique_ptr<ps::models::ModelSource> source_;
    std::unique_ptr<qwen35::Qwen35Model> model_;
    std::optional<qwen35::SequenceSlotPool> slot_pool_;
    std::optional<qwen35::GdnStatePool> gdn_pool_;
    std::optional<qwen35::PagedKVPool> kv_pool_;
    qwen35::Executor executor_;
    std::unique_ptr<qwen35::runtime::ContinuousBatcher> batcher_;
    std::unique_ptr<qwen35::runtime::GpuMcuRuntime> gpu_mcu_runtime_;
};

}  // namespace app
}  // namespace ps
