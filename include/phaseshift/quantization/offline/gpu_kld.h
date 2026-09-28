#pragma once
#include <phaseshift/core/status.h>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

namespace ps::quantization::fpx {

struct GpuPositionMetrics {
    double kld = 0.0;
    double nll_reference = 0.0;
    double nll_candidate = 0.0;
    double logit_mse = 0.0;
    double reference_top1_probability = 0.0;
    double candidate_probability_at_reference_top1 = 0.0;
    uint32_t reference_top1 = 0;
    uint32_t candidate_top1 = 0;
    uint32_t flags = 0;
};

constexpr uint32_t kKldFlagNonFinite = 1u;

struct GpuKldWorkspace {
    GpuKldWorkspace() noexcept = default;
    ~GpuKldWorkspace() noexcept;
    GpuKldWorkspace(const GpuKldWorkspace&) = delete;
    GpuKldWorkspace& operator=(const GpuKldWorkspace&) = delete;
    GpuKldWorkspace(GpuKldWorkspace&&) = delete;
    GpuKldWorkspace& operator=(GpuKldWorkspace&&) = delete;

    void* d_max_partial = nullptr;
    void* d_max_final = nullptr;
    void* d_stats_partial = nullptr;
    std::vector<int32_t> host_targets;
    std::size_t vocab_size = 0;
    std::size_t batch_capacity = 0;
    std::size_t num_blocks = 0;
};

Status gpu_kld_workspace_create(GpuKldWorkspace& workspace, std::size_t vocab_size,
                                std::size_t batch_capacity);
Status gpu_kld_workspace_destroy(GpuKldWorkspace& workspace);
Status gpu_kld_metrics(GpuKldWorkspace& workspace, const float* baseline, const float* candidate,
                       const int32_t* targets, std::size_t positions,
                       GpuPositionMetrics* metrics_out, hipStream_t stream);

}
