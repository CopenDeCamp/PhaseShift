#include <phaseshift/quantization/imatrix/gpu_collector.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ps::quantization::imatrix {

namespace {

constexpr uint32_t kThreads = 256;

__device__ __forceinline__ float load_activation(const void* data, uint64_t index,
                                                 ImatrixDType dtype) {
    if (dtype == ImatrixDType::BF16)
        return __bfloat162float(
            __ushort_as_bfloat16(static_cast<const ::ps::bf16_t*>(data)[index].data));
    return static_cast<const float*>(data)[index];
}

__global__ void imatrix_accumulate_kernel(const void* __restrict__ data,
                                          uint32_t row_stride, uint32_t features,
                                          uint32_t rows, ImatrixDType dtype,
                                          float* __restrict__ sum_sq,
                                          DeviceImatrixDiagnostics* __restrict__ diag,
                                          uint64_t tag) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= features) return;
    float acc = 0.0f;
    uint64_t bad = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const float v = load_activation(data, static_cast<uint64_t>(r) * row_stride + col, dtype);
        const float sq = v * v;
        if (!isfinite(v) || !isfinite(sq)) {
            ++bad;
            continue;
        }
        acc += sq;
    }
    if (bad != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&diag->nonfinite_count),
                  static_cast<unsigned long long>(bad));
        atomicMin(&diag->first_bad_tag, tag);
    }
    sum_sq[col] += acc;
}

}

Status GpuImatrixCollector::initialize(ps::gpu::GpuArena& arena,
                                       const std::vector<uint32_t>& tags,
                                       const std::vector<uint64_t>& k_vals) {
    if (tags.size() != k_vals.size())
        return Status::invalid_argument("imatrix collector: tags/k size mismatch", __FILE__, __LINE__);
    for (std::size_t i = 0; i < tags.size(); ++i) {
        const uint64_t k = k_vals[i];
        if (k == 0 || k > UINT32_MAX)
            return Status::invalid_argument("imatrix collector: bad k", __FILE__, __LINE__);
        const uint64_t sum_sq_bytes = k * sizeof(float);
        const uint64_t diag_offset = (sum_sq_bytes + 7u) & ~uint64_t(7);
        auto alloc = arena.allocate_aligned(diag_offset + sizeof(DeviceImatrixDiagnostics), 16);
        if (!alloc.ok()) return alloc.status();
        void* base = alloc.release().data();
        DeviceAcc acc;
        acc.sum_sq = reinterpret_cast<float*>(base);
        acc.diag = reinterpret_cast<DeviceImatrixDiagnostics*>(
            reinterpret_cast<std::byte*>(base) + diag_offset);
        acc.k = static_cast<uint32_t>(k);
        hipError_t err = hipMemset(acc.sum_sq, 0, sum_sq_bytes);
        if (err != hipSuccess)
            return Status::hip_error("imatrix collector sum_sq memset", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        err = hipMemset(acc.diag, 0, sizeof(DeviceImatrixDiagnostics));
        if (err != hipSuccess)
            return Status::hip_error("imatrix collector diag memset", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        err = hipMemset(&acc.diag->first_bad_tag, 0xFF, sizeof(uint64_t));
        if (err != hipSuccess)
            return Status::hip_error("imatrix collector diag key memset", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        if (device_.find(tags[i]) != device_.end())
            return Status::invalid_argument("imatrix collector: duplicate tag", __FILE__, __LINE__);
        device_[tags[i]] = acc;
        host_[tags[i]].sum_sq.assign(static_cast<std::size_t>(k), 0.0);
    }
    return Status::make_ok();
}

Status GpuImatrixCollector::record(uint32_t tag, const void* data, uint32_t row_stride,
                                   uint32_t features, uint32_t rows, ImatrixDType dtype,
                                   hipStream_t stream) {
    auto it = device_.find(tag);
    if (it == device_.end())
        return Status::invalid_argument("imatrix collector: unknown tag", __FILE__, __LINE__);
    if (data == nullptr)
        return Status::invalid_argument("imatrix collector: null activation", __FILE__, __LINE__);
    if (features != it->second.k) {
        std::fprintf(stderr, "imatrix k mismatch: tag=%u expected=%u actual=%u\n", tag,
                     it->second.k, features);
        return Status::invalid_argument("imatrix collector: k mismatch", __FILE__, __LINE__);
    }
    if (rows == 0 || features == 0) return Status::make_ok();
    const uint32_t grid = (features + kThreads - 1u) / kThreads;
    imatrix_accumulate_kernel<<<grid, kThreads, 0, stream>>>(
        data, row_stride, features, rows, dtype, it->second.sum_sq, it->second.diag,
        static_cast<uint64_t>(tag));
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return Status::hip_error("imatrix accumulate launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    host_[tag].count += rows;
    return Status::make_ok();
}

Status GpuImatrixCollector::flush(hipStream_t stream) {
    for (auto& kv : device_) {
        const uint32_t tag = kv.first;
        const DeviceAcc& acc = kv.second;
        std::vector<float> h(acc.k);
        hipError_t err = hipMemcpyAsync(h.data(), acc.sum_sq, acc.k * sizeof(float),
                                        hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
            return Status::hip_error("imatrix flush D2H", hipGetErrorString(err), __FILE__, __LINE__);
        DeviceImatrixDiagnostics diag{};
        err = hipMemcpyAsync(&diag, acc.diag, sizeof(DeviceImatrixDiagnostics),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
            return Status::hip_error("imatrix flush diag D2H", hipGetErrorString(err), __FILE__,
                                     __LINE__);
        const hipError_t sync = hipStreamSynchronize(stream);
        if (sync != hipSuccess)
            return Status::hip_error("imatrix flush sync", hipGetErrorString(sync), __FILE__,
                                     __LINE__);
        ImatrixAccumulator& host = host_[tag];
        for (uint32_t i = 0; i < acc.k; ++i)
            host.sum_sq[i] += static_cast<double>(h[i]);
        DeviceImatrixDiagnostics& total = diagnostics_[tag];
        total.nonfinite_count += diag.nonfinite_count;
        if (diag.first_bad_tag < total.first_bad_tag) total.first_bad_tag = diag.first_bad_tag;
        err = hipMemsetAsync(acc.sum_sq, 0, acc.k * sizeof(float), stream);
        if (err != hipSuccess)
            return Status::hip_error("imatrix flush zero", hipGetErrorString(err), __FILE__, __LINE__);
        err = hipMemsetAsync(acc.diag, 0, sizeof(DeviceImatrixDiagnostics), stream);
        if (err != hipSuccess)
            return Status::hip_error("imatrix flush diag zero", hipGetErrorString(err), __FILE__,
                                     __LINE__);
        err = hipMemsetAsync(&acc.diag->first_bad_tag, 0xFF, sizeof(uint64_t), stream);
        if (err != hipSuccess)
            return Status::hip_error("imatrix flush diag key zero", hipGetErrorString(err),
                                     __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}
