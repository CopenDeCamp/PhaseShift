#pragma once

#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/fsm_worker.h>
#include <phaseshift/runtime/gpu_mcu/worker_image.h>
#include <phaseshift/runtime/gpu_mcu/micro_fsm.h>
#include <phaseshift/runtime/stream_bridge.h>

#include <memory>
#include <cstdio>
#include <vector>

namespace ps {
namespace qwen35 {
namespace runtime {

enum class McuPlanDriverMode : uint8_t {
    HostStream,
    ExternalPersistentMcu,
};

struct McuDecodeRuntimeOptions {
    int device = 0;
    uint32_t queue_size = ::ps::runtime::gpu_mcu::kAqlDefaultQueueSize;
    uint32_t queue_ahead_depth = 4u;
    uint32_t record_entries = ::ps::runtime::gpu_mcu::kMcuMaxDebugRecords;
    uint32_t cu_reservation = 1u;
    McuPlanDriverMode driver_mode = McuPlanDriverMode::HostStream;
};

class McuDecodeRuntime {
public:
    McuDecodeRuntime() = default;
    ~McuDecodeRuntime();

    static Result<McuDecodeRuntime> create(const McuDecodeRuntimeOptions& options);

    Status shutdown() noexcept;

    Status prepare_plan(const McuCompiledPlan& plan);

    Status enqueue_run(hipStream_t stream, uint32_t epoch) const;
    Status enqueue_wait(hipStream_t stream, uint32_t epoch) const;

    bool ready() const noexcept { return fsm_active_; }
    bool external_driver() const noexcept {
        return driver_mode_ == McuPlanDriverMode::ExternalPersistentMcu;
    }
    uint32_t plan_node_count() const noexcept { return plan_node_count_; }
    uint32_t plan_variant_count() const noexcept { return plan_variant_count_; }
    const ::ps::runtime::gpu_mcu::McuInvocationPatch* row_global_patches()
        const noexcept {
        return row_global_patches_;
    }
    uint32_t row_global_patch_count() const noexcept {
        return row_global_patch_count_;
    }
    void dump_log(std::FILE* out) const noexcept;
    uint32_t result_code() const noexcept;
    uint32_t embedded_kernel_count() const noexcept {
        return static_cast<uint32_t>(code_objects_.size());
    }
    const ::ps::runtime::gpu_mcu::GpuMcuFsm& fsm() const noexcept { return fsm_; }
    ::ps::runtime::gpu_mcu::GpuMcuFsmState* execution_state() const noexcept {
        return fsm_.device_state();
    }
    const ::ps::runtime::StreamSignal& start_signal() const noexcept {
        return start_signal_;
    }
    const ::ps::runtime::StreamSignal& done_signal() const noexcept {
        return done_signal_;
    }

    McuDecodeRuntime(McuDecodeRuntime&& other) noexcept;
    McuDecodeRuntime& operator=(McuDecodeRuntime&& other) noexcept;
    McuDecodeRuntime(const McuDecodeRuntime&) = delete;
    McuDecodeRuntime& operator=(const McuDecodeRuntime&) = delete;

private:
    void move_from(McuDecodeRuntime&& other) noexcept;
    Status release_invocations() noexcept;
    Status clear_plan() noexcept;
    Status rebuild_row_global_patches() noexcept;

    ::ps::runtime::gpu_mcu::GpuMcuCuPartition partition_{};
    ::ps::runtime::gpu_mcu::GpuMcuAqlQueue queue_{};
    ::ps::runtime::gpu_mcu::GpuMcuFsm fsm_{};
    ::ps::runtime::gpu_mcu::GpuMcuKernargRegion kernarg_{};
    ::ps::runtime::gpu_mcu::GpuMcuKernargRegion log_region_{};
    ::ps::runtime::gpu_mcu::GpuMcuAqlCodeObject probe_code_{};
    ::ps::runtime::gpu_mcu::GpuAqlKernelMetadata probe_meta_{};

    std::vector<std::unique_ptr<::ps::runtime::gpu_mcu::GpuMcuAqlCodeObject>>
        code_objects_;

    ::ps::runtime::gpu_mcu::McuPlanNode* plan_nodes_ = nullptr;
    ::ps::runtime::gpu_mcu::McuKernelVariantDesc* plan_variants_ = nullptr;
    ::ps::runtime::gpu_mcu::GpuMcuRetainedPacket* plan_retained_ = nullptr;
    ::ps::runtime::gpu_mcu::GpuMcuDeviceCompletion* completions_ = nullptr;
    ::ps::runtime::gpu_mcu::McuDispatchRecord* records_ = nullptr;
    uint32_t* plan_output_ = nullptr;
    uint64_t* plan_timestamps_ = nullptr;

    uint32_t plan_node_count_ = 0;
    uint32_t plan_variant_count_ = 0;
    uint32_t completion_count_ = 0;
    uint32_t record_count_ = 0;

    ::ps::runtime::gpu_mcu::McuInvocationPatch* row_global_patches_ = nullptr;
    uint32_t row_global_patch_count_ = 0;
    std::vector<McuAttentionRegion> attention_regions_;
    std::vector<McuBf16VariantCatalog> bf16_catalogs_;
    void* dynamic_bindings_ = nullptr;
    uint32_t dynamic_binding_count_ = 0;

    void* rms_ = nullptr;
    uint32_t rms_count_ = 0;
    void* quant_ = nullptr;
    uint32_t quant_count_ = 0;
    void* e4m3_ = nullptr;
    uint32_t e4m3_count_ = 0;
    void* psq4_ = nullptr;
    uint32_t psq4_count_ = 0;
    void* psq4_multi_ = nullptr;
    uint32_t psq4_multi_count_ = 0;
    void* verify_accept_ = nullptr;
    uint32_t verify_accept_count_ = 0;
    void* verify_accept_batch_ = nullptr;
    uint32_t verify_accept_batch_count_ = 0;
    void* gdn_spec_restore_ = nullptr;
    uint32_t gdn_spec_restore_count_ = 0;
    void* gdn_spec_restore_from_counts_ = nullptr;
    uint32_t gdn_spec_restore_from_counts_count_ = 0;
    void* argmax_f32_ = nullptr;
    uint32_t argmax_f32_count_ = 0;
    void* elementwise_ = nullptr;
    uint32_t elementwise_count_ = 0;
    void* rope_ = nullptr;
    uint32_t rope_count_ = 0;
    void* kv_append_ = nullptr;
    uint32_t kv_append_count_ = 0;
    void* paged_ = nullptr;
    uint32_t paged_count_ = 0;
    void* paged_split_ = nullptr;
    uint32_t paged_split_count_ = 0;
    void* paged_reduce_ = nullptr;
    uint32_t paged_reduce_count_ = 0;
    void* bf16_ = nullptr;
    uint32_t bf16_count_ = 0;
    void* bf16_wmma_ = nullptr;
    uint32_t bf16_wmma_count_ = 0;
    void* l2_ = nullptr;
    uint32_t l2_count_ = 0;
    void* embedding_ = nullptr;
    uint32_t embedding_count_ = 0;
    void* embedding_psq8_ = nullptr;
    uint32_t embedding_psq8_count_ = 0;
    void* output_gather_ = nullptr;
    uint32_t output_gather_count_ = 0;
    void* gdn_conv_ = nullptr;
    uint32_t gdn_conv_count_ = 0;
    void* gdn_recur_ = nullptr;
    uint32_t gdn_recur_count_ = 0;
    void* gdn_reset_ = nullptr;
    uint32_t gdn_reset_count_ = 0;

    ::ps::runtime::StreamSignal start_signal_{};
    ::ps::runtime::StreamSignal done_signal_{};
    ::ps::runtime::StreamSignal result_signal_{};

    int device_ = 0;
    uint32_t queue_ahead_depth_ = 4u;
    McuPlanDriverMode driver_mode_ = McuPlanDriverMode::HostStream;
    bool queue_active_ = false;
    bool fsm_active_ = false;
    bool partition_active_ = false;
};

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
