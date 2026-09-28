#include <phaseshift/quantization/offline/gpu_kld.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace ps::quantization::fpx {

namespace {

constexpr uint32_t kThreads = 256;

struct Maxima {
    float baseline;
    float candidate;
    uint32_t baseline_index;
    uint32_t candidate_index;
};

struct Partial {
    double exp_baseline;
    double exp_candidate;
    double weighted_delta;
    double squared_delta;
    uint32_t nonfinite;
};

__device__ bool better(float value, uint32_t index, float best, uint32_t best_index) {
    return value > best || (value == best && index < best_index);
}

__global__ void maxima_partial_kernel(const float* baseline, const float* candidate,
                                      std::size_t vocab, std::size_t blocks, Maxima* output) {
    const std::size_t position = blockIdx.y;
    const std::size_t block = blockIdx.x;
    __shared__ Maxima values[kThreads];
    Maxima best{-INFINITY, -INFINITY, 0, 0};
    const std::size_t offset = position * vocab;
    for (std::size_t index = block * blockDim.x + threadIdx.x; index < vocab;
         index += blocks * blockDim.x) {
        const float a = baseline[offset + index];
        const float b = candidate[offset + index];
        if (better(a, static_cast<uint32_t>(index), best.baseline, best.baseline_index)) {
            best.baseline = a;
            best.baseline_index = static_cast<uint32_t>(index);
        }
        if (better(b, static_cast<uint32_t>(index), best.candidate, best.candidate_index)) {
            best.candidate = b;
            best.candidate_index = static_cast<uint32_t>(index);
        }
    }
    values[threadIdx.x] = best;
    __syncthreads();
    for (uint32_t width = kThreads / 2; width != 0; width /= 2) {
        if (threadIdx.x < width) {
            const Maxima other = values[threadIdx.x + width];
            if (better(other.baseline, other.baseline_index, values[threadIdx.x].baseline, values[threadIdx.x].baseline_index)) values[threadIdx.x].baseline = other.baseline, values[threadIdx.x].baseline_index = other.baseline_index;
            if (better(other.candidate, other.candidate_index, values[threadIdx.x].candidate, values[threadIdx.x].candidate_index)) values[threadIdx.x].candidate = other.candidate, values[threadIdx.x].candidate_index = other.candidate_index;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[position * blocks + block] = values[0];
}

__global__ void maxima_finalize_kernel(const Maxima* partial, std::size_t blocks, Maxima* output) {
    if (threadIdx.x != 0) return;
    const std::size_t position = blockIdx.x;
    Maxima best{-INFINITY, -INFINITY, 0, 0};
    for (std::size_t block = 0; block < blocks; ++block) {
        const Maxima value = partial[position * blocks + block];
        if (better(value.baseline, value.baseline_index, best.baseline, best.baseline_index)) best.baseline = value.baseline, best.baseline_index = value.baseline_index;
        if (better(value.candidate, value.candidate_index, best.candidate, best.candidate_index)) best.candidate = value.candidate, best.candidate_index = value.candidate_index;
    }
    output[position] = best;
}

__global__ void stats_partial_kernel(const float* baseline, const float* candidate, std::size_t vocab,
                                     std::size_t blocks, const Maxima* maxima, Partial* output) {
    const std::size_t position = blockIdx.y;
    const std::size_t block = blockIdx.x;
    const Maxima max = maxima[position];
    const std::size_t offset = position * vocab;
    __shared__ Partial values[kThreads];
    Partial value{};
    for (std::size_t index = block * blockDim.x + threadIdx.x; index < vocab;
         index += blocks * blockDim.x) {
        const float a = baseline[offset + index];
        const float b = candidate[offset + index];
        if (!isfinite(a) || !isfinite(b)) { value.nonfinite = 1; continue; }
        const double pa = exp(static_cast<double>(a - max.baseline));
        value.exp_baseline += pa;
        value.exp_candidate += exp(static_cast<double>(b - max.candidate));
        value.weighted_delta += pa * static_cast<double>(a - b);
        const double delta = static_cast<double>(a) - static_cast<double>(b);
        value.squared_delta += delta * delta;
    }
    values[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t width = kThreads / 2; width != 0; width /= 2) {
        if (threadIdx.x < width) {
            const Partial other = values[threadIdx.x + width];
            values[threadIdx.x].exp_baseline += other.exp_baseline;
            values[threadIdx.x].exp_candidate += other.exp_candidate;
            values[threadIdx.x].weighted_delta += other.weighted_delta;
            values[threadIdx.x].squared_delta += other.squared_delta;
            values[threadIdx.x].nonfinite |= other.nonfinite;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[position * blocks + block] = values[0];
}

__global__ void metrics_finalize_kernel(const float* baseline, const float* candidate, std::size_t vocab,
                                        std::size_t blocks, const int32_t* targets, const Maxima* maxima,
                                        const Partial* partial, GpuPositionMetrics* output) {
    if (threadIdx.x != 0) return;
    const std::size_t position = blockIdx.x;
    const Maxima max = maxima[position];
    Partial total{};
    for (std::size_t block = 0; block < blocks; ++block) {
        const Partial value = partial[position * blocks + block];
        total.exp_baseline += value.exp_baseline;
        total.exp_candidate += value.exp_candidate;
        total.weighted_delta += value.weighted_delta;
        total.squared_delta += value.squared_delta;
        total.nonfinite |= value.nonfinite;
    }
    const std::size_t offset = position * vocab;
    const int32_t target = targets[position];
    const double log_a = static_cast<double>(max.baseline) + log(total.exp_baseline);
    const double log_b = static_cast<double>(max.candidate) + log(total.exp_candidate);
    GpuPositionMetrics result;
    result.kld = total.weighted_delta / total.exp_baseline + log_b - log_a;
    result.nll_reference = log_a - baseline[offset + target];
    result.nll_candidate = log_b - candidate[offset + target];
    result.logit_mse = total.squared_delta / static_cast<double>(vocab);
    result.reference_top1 = max.baseline_index;
    result.candidate_top1 = max.candidate_index;
    result.reference_top1_probability = exp(baseline[offset + max.baseline_index] - log_a);
    result.candidate_probability_at_reference_top1 = exp(candidate[offset + max.baseline_index] - log_b);
    result.flags = total.nonfinite ? kKldFlagNonFinite : 0;
    output[position] = result;
}

Status hip_status(hipError_t error, const char* expression) {
    return error == hipSuccess ? Status::make_ok() : Status::hip_error(expression, hipGetErrorString(error), __FILE__, __LINE__);
}

Status allocation_failure(Status primary, GpuKldWorkspace& workspace) {
    const Status cleanup = gpu_kld_workspace_destroy(workspace);
    if (cleanup.ok()) return primary;
    const std::string message = primary.message() + "; workspace cleanup failed: " + cleanup.message();
    return Status::invalid_state(message.c_str(), __FILE__, __LINE__);
}

}

GpuKldWorkspace::~GpuKldWorkspace() noexcept {
    (void)gpu_kld_workspace_destroy(*this);
}

Status gpu_kld_workspace_create(GpuKldWorkspace& workspace, std::size_t vocab_size,
                                std::size_t batch_capacity) {
    if (workspace.d_max_partial || workspace.d_max_final || workspace.d_stats_partial ||
        !workspace.host_targets.empty() || workspace.vocab_size != 0 || workspace.batch_capacity != 0 ||
        workspace.num_blocks != 0) {
        return Status::invalid_state("KLD workspace is not empty", __FILE__, __LINE__);
    }
    if (vocab_size == 0 || batch_capacity == 0 ||
        vocab_size > std::numeric_limits<uint32_t>::max() ||
        batch_capacity > std::numeric_limits<unsigned int>::max()) {
        return Status::invalid_argument("invalid KLD workspace dimension", __FILE__, __LINE__);
    }
    workspace.vocab_size = vocab_size;
    workspace.batch_capacity = batch_capacity;
    const std::size_t block_span = kThreads * 8;
    workspace.num_blocks = std::min<std::size_t>(256, (vocab_size - 1) / block_span + 1);
    if (batch_capacity > std::numeric_limits<std::size_t>::max() / workspace.num_blocks ||
        batch_capacity * workspace.num_blocks > std::numeric_limits<std::size_t>::max() / sizeof(Maxima) ||
        batch_capacity * workspace.num_blocks > std::numeric_limits<std::size_t>::max() / sizeof(Partial) ||
        batch_capacity > std::numeric_limits<std::size_t>::max() / sizeof(Maxima)) {
        workspace.vocab_size = 0;
        workspace.batch_capacity = 0;
        workspace.num_blocks = 0;
        return Status::overflow("KLD workspace allocation size overflow", __FILE__, __LINE__);
    }
    try {
        workspace.host_targets.resize(batch_capacity);
    } catch (const std::bad_alloc& error) {
        workspace.vocab_size = 0;
        workspace.batch_capacity = 0;
        workspace.num_blocks = 0;
        return Status::insufficient_memory(error.what(), __FILE__, __LINE__);
    }
    Status status = hip_status(hipMalloc(&workspace.d_max_partial, batch_capacity * workspace.num_blocks * sizeof(Maxima)), "hipMalloc(d_max_partial)");
    if (!status.ok()) return allocation_failure(status, workspace);
    status = hip_status(hipMalloc(&workspace.d_max_final, batch_capacity * sizeof(Maxima)), "hipMalloc(d_max_final)");
    if (!status.ok()) return allocation_failure(status, workspace);
    status = hip_status(hipMalloc(&workspace.d_stats_partial, batch_capacity * workspace.num_blocks * sizeof(Partial)), "hipMalloc(d_stats_partial)");
    if (!status.ok()) return allocation_failure(status, workspace);
    return Status::make_ok();
}

Status gpu_kld_workspace_destroy(GpuKldWorkspace& workspace) {
    Status first = Status::make_ok();
    const auto release = [&](void*& pointer, const char* name) {
        if (!pointer) return;
        const Status status = hip_status(hipFree(pointer), name);
        if (status.ok()) pointer = nullptr;
        if (first.ok() && !status.ok()) first = status;
    };
    release(workspace.d_stats_partial, "hipFree(d_stats_partial)");
    release(workspace.d_max_final, "hipFree(d_max_final)");
    release(workspace.d_max_partial, "hipFree(d_max_partial)");
    if (!workspace.d_stats_partial && !workspace.d_max_final && !workspace.d_max_partial) {
        workspace.host_targets.clear();
        workspace.vocab_size = 0;
        workspace.batch_capacity = 0;
        workspace.num_blocks = 0;
    }
    return first;
}

Status gpu_kld_metrics(GpuKldWorkspace& workspace, const float* baseline, const float* candidate,
                       const int32_t* targets, std::size_t positions,
                       GpuPositionMetrics* metrics_out, hipStream_t stream) {
    if (!baseline || !candidate || !targets || !metrics_out || positions == 0 ||
        workspace.vocab_size == 0 || workspace.num_blocks == 0 ||
        positions > workspace.batch_capacity || !workspace.d_max_partial || !workspace.d_max_final ||
        !workspace.d_stats_partial || positions > std::numeric_limits<unsigned int>::max()) {
        return Status::invalid_argument("invalid KLD metric input", __FILE__, __LINE__);
    }
    if (workspace.host_targets.size() < positions) {
        return Status::invalid_state("KLD workspace target capacity is invalid", __FILE__, __LINE__);
    }
    Status status = hip_status(hipMemcpyAsync(workspace.host_targets.data(), targets,
                                              positions * sizeof(int32_t),
                                              hipMemcpyDeviceToHost, stream),
                               "hipMemcpyAsync(targets)");
    if (!status.ok()) return status;
    status = hip_status(hipStreamSynchronize(stream), "hipStreamSynchronize(targets)");
    if (!status.ok()) return status;
    for (std::size_t position = 0; position < positions; ++position) {
        const int32_t target = workspace.host_targets[position];
        if (target < 0 || static_cast<std::size_t>(target) >= workspace.vocab_size) {
            return Status::invalid_argument("KLD target is out of range", __FILE__, __LINE__);
        }
    }
    dim3 partial_grid(workspace.num_blocks, positions);
    maxima_partial_kernel<<<partial_grid, kThreads, 0, stream>>>(baseline, candidate, workspace.vocab_size,
                                                                   workspace.num_blocks, static_cast<Maxima*>(workspace.d_max_partial));
    status = hip_status(hipGetLastError(), "maxima_partial_kernel");
    if (!status.ok()) return status;
    maxima_finalize_kernel<<<positions, 1, 0, stream>>>(static_cast<const Maxima*>(workspace.d_max_partial),
                                                          workspace.num_blocks, static_cast<Maxima*>(workspace.d_max_final));
    status = hip_status(hipGetLastError(), "maxima_finalize_kernel");
    if (!status.ok()) return status;
    stats_partial_kernel<<<partial_grid, kThreads, 0, stream>>>(baseline, candidate, workspace.vocab_size,
                                                                  workspace.num_blocks, static_cast<const Maxima*>(workspace.d_max_final), static_cast<Partial*>(workspace.d_stats_partial));
    status = hip_status(hipGetLastError(), "stats_partial_kernel");
    if (!status.ok()) return status;
    metrics_finalize_kernel<<<positions, 1, 0, stream>>>(baseline, candidate, workspace.vocab_size,
                                                           workspace.num_blocks, targets, static_cast<const Maxima*>(workspace.d_max_final), static_cast<const Partial*>(workspace.d_stats_partial), metrics_out);
    status = hip_status(hipGetLastError(), "metrics_finalize_kernel");
    if (!status.ok()) return status;
    return hip_status(hipStreamSynchronize(stream), "hipStreamSynchronize(metrics)");
}

}
