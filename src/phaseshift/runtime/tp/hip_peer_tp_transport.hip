#include <phaseshift/runtime/tp/tp_transports.h>

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/types.h>

#include "tp_transport_internal.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ps::runtime {

namespace {

constexpr std::size_t kProbeMaxBytes = 1u << 20;
constexpr std::size_t kProbeSizes[] = {8192u, 131072u, 1048576u};
constexpr int kProbeRounds = 4;

__device__ inline float tp_bf16_to_f32(uint16_t v) {
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ inline uint16_t tp_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        if ((u & 0x007FFFFFu) != 0u) return 0x7FC0u;
        return static_cast<uint16_t>(u >> 16);
    }
    const uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

__global__ void tp_sum_multi_bf16_kernel(const uint16_t* const* srcs, uint32_t nsrcs,
                                         uint16_t* out, uint16_t* bcast, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        float acc = tp_bf16_to_f32(srcs[0][i]);
        for (uint32_t r = 1; r < nsrcs; ++r) {
            acc += tp_bf16_to_f32(srcs[r][i]);
        }
        const uint16_t value = tp_f32_to_bf16(acc);
        out[i] = value;
        bcast[i] = value;
    }
}

__global__ void tp_sum_multi_f32_kernel(const float* const* srcs, uint32_t nsrcs,
                                        float* out, float* bcast, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        float acc = srcs[0][i];
        for (uint32_t r = 1; r < nsrcs; ++r) {
            acc += srcs[r][i];
        }
        out[i] = acc;
        bcast[i] = acc;
    }
}

__global__ void tp_sum2_bf16_kernel(const uint16_t* a, const uint16_t* b, uint16_t* out,
                                    uint16_t* bcast, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const uint16_t value =
            tp_f32_to_bf16(tp_bf16_to_f32(a[i]) + tp_bf16_to_f32(b[i]));
        out[i] = value;
        bcast[i] = value;
    }
}

__global__ void tp_sum2_f32_kernel(const float* a, const float* b, float* out,
                                   float* bcast, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const float value = a[i] + b[i];
        out[i] = value;
        bcast[i] = value;
    }
}

__global__ void tp_copy_bf16_kernel(uint16_t* out, const uint16_t* in, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = in[i];
    }
}

__global__ void tp_copy_f32_kernel(float* out, const float* in, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = in[i];
    }
}

uint32_t tp_grid_for(uint64_t count) {
    const uint64_t blocks = (count + 255u) / 256u;
    if (blocks == 0) return 1;
    return static_cast<uint32_t>(blocks > 1024u ? 1024u : blocks);
}

Status launch_multi_sum(const void* srcs_device, uint32_t nsrcs, void* out, void* bcast,
                        std::size_t bytes, ValueDType dtype, hipStream_t stream) {
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_sum_multi_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            reinterpret_cast<const float* const*>(srcs_device), nsrcs,
            static_cast<float*>(out), static_cast<float*>(bcast), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_sum_multi_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t* const*>(srcs_device), nsrcs,
            static_cast<uint16_t*>(out), static_cast<uint16_t*>(bcast), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp multi-sum kernel launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status launch_sum2(const void* a, const void* b, void* out, void* bcast,
                   std::size_t bytes, ValueDType dtype, hipStream_t stream) {
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_sum2_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const float*>(a), static_cast<const float*>(b),
            static_cast<float*>(out), static_cast<float*>(bcast), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_sum2_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
            static_cast<uint16_t*>(out), static_cast<uint16_t*>(bcast), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp_sum2 kernel launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status launch_copy(void* dst, const void* src, std::size_t bytes, ValueDType dtype,
                   hipStream_t stream) {
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_copy_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<float*>(dst), static_cast<const float*>(src), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_copy_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp_copy kernel launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}

HipPeerTpTransport::HipPeerTpTransport(std::vector<int> devices)
    : devices_(std::move(devices)) {}

HipPeerTpTransport::~HipPeerTpTransport() {
    {
        auto scope = ps::gpu::ScopedDevice::create(devices_.empty() ? 0 : devices_[0]);
        if (scope.ok()) {
            for (void* buffer : scratch_) {
                if (buffer != nullptr) ps::gpu::discard_cleanup_result(hipFree(buffer));
            }
            if (leader_result_ != nullptr) {
                ps::gpu::discard_cleanup_result(hipFree(leader_result_));
                leader_result_ = nullptr;
            }
            if (src_ptr_dev_ != nullptr) {
                ps::gpu::discard_cleanup_result(hipFree(src_ptr_dev_));
                src_ptr_dev_ = nullptr;
            }
            if (reduce_done_ != nullptr) {
                ps::gpu::discard_cleanup_result(hipEventDestroy(reduce_done_));
                reduce_done_ = nullptr;
            }
            for (hipEvent_t event : bcast_done_) {
                if (event != nullptr) ps::gpu::discard_cleanup_result(hipEventDestroy(event));
            }
            if (transport_stream_ != nullptr) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(transport_stream_));
                transport_stream_ = nullptr;
            }
        }
    }
    for (std::size_t r = 0; r < probe_buffers_.size(); ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) continue;
        if (probe_buffers_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipFree(probe_buffers_[r]));
            probe_buffers_[r] = nullptr;
        }
        if (r < probe_streams_.size() && probe_streams_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(probe_streams_[r]));
            probe_streams_[r] = nullptr;
        }
        if (r < probe_ready_.size() && probe_ready_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipEventDestroy(probe_ready_[r]));
            probe_ready_[r] = nullptr;
        }
        if (r < scratch_done_.size() && scratch_done_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipEventDestroy(scratch_done_[r]));
            scratch_done_[r] = nullptr;
        }
    }
    for (void* host_buffer : src_ptr_host_) {
        if (host_buffer != nullptr) ps::gpu::discard_cleanup_result(hipHostFree(host_buffer));
    }
}

Result<std::unique_ptr<HipPeerTpTransport>> HipPeerTpTransport::create(
    const std::vector<int>& devices) {
    if (devices.size() < 2) {
        return Status::invalid_argument("hip peer transport requires at least 2 devices",
                                        __FILE__, __LINE__);
    }
    auto transport = std::unique_ptr<HipPeerTpTransport>(
        new HipPeerTpTransport(devices));
    Status enable_status = transport->enable_peer_access();
    if (!enable_status.ok()) return enable_status;
    const std::size_t rank_count = devices.size();
    transport->bcast_done_.assign(rank_count, nullptr);
    transport->scratch_done_.assign(rank_count, nullptr);
    transport->scratch_.assign(rank_count, nullptr);
    transport->probe_buffers_.assign(rank_count, nullptr);
    transport->probe_streams_.assign(rank_count, nullptr);
    transport->probe_ready_.assign(rank_count, nullptr);
    {
        auto scope = ps::gpu::ScopedDevice::create(devices[0]);
        if (!scope.ok()) return scope.status();
        if (hipStreamCreate(&transport->transport_stream_) != hipSuccess) {
            return Status::hip_error("hipStreamCreate transport",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        if (hipEventCreateWithFlags(&transport->reduce_done_, hipEventDisableTiming) !=
            hipSuccess) {
            return Status::hip_error("hipEventCreate reduce",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        for (std::size_t r = 1; r < rank_count; ++r) {
            if (hipEventCreateWithFlags(&transport->bcast_done_[r],
                                        hipEventDisableTiming) != hipSuccess) {
                return Status::hip_error("hipEventCreate broadcast",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
    }
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices[r]);
        if (!scope.ok()) return scope.status();
        if (hipMalloc(&transport->probe_buffers_[r], kProbeMaxBytes) != hipSuccess) {
            return Status::hip_error("hipMalloc tp probe", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        if (hipStreamCreate(&transport->probe_streams_[r]) != hipSuccess) {
            return Status::hip_error("hipStreamCreate tp probe",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        if (hipEventCreateWithFlags(&transport->probe_ready_[r],
                                    hipEventDisableTiming) != hipSuccess) {
            return Status::hip_error("hipEventCreate tp probe",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        if (r > 0 && hipEventCreateWithFlags(&transport->scratch_done_[r],
                                             hipEventDisableTiming) != hipSuccess) {
            return Status::hip_error("hipEventCreate tp scratch",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    return transport;
}

Status HipPeerTpTransport::enable_peer_access() {
    if (peer_access_enabled_) return Status::make_ok();
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[i]);
        if (!scope.ok()) return scope.status();
        for (std::size_t j = 0; j < devices_.size(); ++j) {
            if (i == j) continue;
            const hipError_t e = hipDeviceEnablePeerAccess(devices_[j], 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                return Status::hip_error("hipDeviceEnablePeerAccess", hipGetErrorString(e),
                                         __FILE__, __LINE__);
            }
            ps::gpu::discard_cleanup_result(hipGetLastError());
        }
    }
    peer_access_enabled_ = true;
    return Status::make_ok();
}

Status HipPeerTpTransport::ensure_buffers(std::size_t bytes, std::size_t rank_count) {
    auto scope = ps::gpu::ScopedDevice::create(devices_[0]);
    if (!scope.ok()) return scope.status();
    if (leader_result_bytes_ < bytes) {
        if (leader_result_ != nullptr) {
            ps::gpu::discard_cleanup_result(hipFree(leader_result_));
            leader_result_ = nullptr;
        }
        if (hipMalloc(&leader_result_, bytes) != hipSuccess) {
            return Status::hip_error("hipMalloc leader result",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        for (std::size_t r = 1; r < rank_count; ++r) {
            if (scratch_[r] != nullptr) {
                ps::gpu::discard_cleanup_result(hipFree(scratch_[r]));
                scratch_[r] = nullptr;
            }
            if (hipMalloc(&scratch_[r], bytes) != hipSuccess) {
                return Status::hip_error("hipMalloc tp scratch",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        leader_result_bytes_ = bytes;
    }
    if (src_ptr_capacity_ < rank_count) {
        if (src_ptr_dev_ != nullptr) {
            ps::gpu::discard_cleanup_result(hipFree(src_ptr_dev_));
            src_ptr_dev_ = nullptr;
        }
        for (void*& host_buffer : src_ptr_host_) {
            if (host_buffer != nullptr) {
                ps::gpu::discard_cleanup_result(hipHostFree(host_buffer));
                host_buffer = nullptr;
            }
        }
        const std::size_t ptr_bytes = rank_count * sizeof(const void*);
        if (hipMalloc(&src_ptr_dev_, ptr_bytes) != hipSuccess) {
            return Status::hip_error("hipMalloc src pointers",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        for (void*& host_buffer : src_ptr_host_) {
            if (hipHostMalloc(&host_buffer, ptr_bytes, hipHostMallocDefault) != hipSuccess) {
                return Status::hip_error("hipHostMalloc src pointers",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        src_ptr_capacity_ = rank_count;
        host_src_ptrs_.assign(rank_count, nullptr);
    }
    return Status::make_ok();
}

Status HipPeerTpTransport::enqueue_sum(const TpSumInvocation& invocation) {
    const std::size_t rank_count = devices_.size();
    const ValueDType dtype = invocation.targets[0].dtype;
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, rank_count, bytes);
    if (!st.ok()) return st;

    for (std::size_t r = 1; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        if (hipStreamWaitEvent(invocation.streams[r], invocation.ready[r], 0) !=
            hipSuccess) {
            return Status::hip_error("hipStreamWaitEvent tp segment",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        st = launch_copy(scratch_[r], invocation.targets[r].ptr, bytes, dtype,
                         invocation.streams[r]);
        if (!st.ok()) return st;
        if (hipEventRecord(scratch_done_[r], invocation.streams[r]) != hipSuccess) {
            return Status::hip_error("hipEventRecord tp scratch",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }

    const bool two_rank = rank_count == 2;
    if (!two_rank) {
        host_src_ptrs_.resize(rank_count);
        host_src_ptrs_[0] = invocation.targets[0].ptr;
        for (std::size_t r = 1; r < rank_count; ++r) {
            host_src_ptrs_[r] = scratch_[r];
        }
    }
    const std::size_t parity = src_ptr_parity_;
    src_ptr_parity_ ^= 1u;

    auto leader = ps::gpu::ScopedDevice::create(devices_[0]);
    if (!leader.ok()) return leader.status();
    if (!two_rank) {
        std::memcpy(src_ptr_host_[parity], host_src_ptrs_.data(),
                    rank_count * sizeof(const void*));
        if (hipMemcpyAsync(src_ptr_dev_, src_ptr_host_[parity],
                           rank_count * sizeof(const void*), hipMemcpyHostToDevice,
                           transport_stream_) != hipSuccess) {
            return Status::hip_error("hipMemcpyAsync src pointers",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    if (hipStreamWaitEvent(transport_stream_, invocation.ready[0], 0) != hipSuccess) {
        return Status::hip_error("hipStreamWaitEvent tp segment",
                                 hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
    }
    for (std::size_t r = 1; r < rank_count; ++r) {
        if (hipStreamWaitEvent(transport_stream_, scratch_done_[r], 0) != hipSuccess) {
            return Status::hip_error("hipStreamWaitEvent tp scratch",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    if (two_rank) {
        st = launch_sum2(invocation.targets[0].ptr, scratch_[1],
                         invocation.targets[0].ptr, leader_result_, bytes, dtype,
                         transport_stream_);
    } else {
        st = launch_multi_sum(src_ptr_dev_, static_cast<uint32_t>(rank_count),
                              invocation.targets[0].ptr, leader_result_, bytes, dtype,
                              transport_stream_);
    }
    if (!st.ok()) return st;
    if (hipEventRecord(reduce_done_, transport_stream_) != hipSuccess) {
        return Status::hip_error("hipEventRecord tp reduce",
                                 hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
    }
    for (std::size_t r = 1; r < rank_count; ++r) {
        st = launch_copy(invocation.targets[r].ptr, leader_result_, bytes, dtype,
                         transport_stream_);
        if (!st.ok()) return st;
        if (hipEventRecord(bcast_done_[r], transport_stream_) != hipSuccess) {
            return Status::hip_error("hipEventRecord tp broadcast",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    if (hipStreamWaitEvent(invocation.streams[0], reduce_done_, 0) != hipSuccess) {
        return Status::hip_error("hipStreamWaitEvent tp reduce result",
                                 hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
    }
    for (std::size_t r = 1; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        if (hipStreamWaitEvent(invocation.streams[r], bcast_done_[r], 0) != hipSuccess) {
            return Status::hip_error("hipStreamWaitEvent tp broadcast result",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    return Status::make_ok();
}

Status HipPeerTpTransport::verify() {
    if (verified_) return Status::make_ok();
    Status st = enable_peer_access();
    if (!st.ok()) return st;

    const std::size_t rank_count = devices_.size();
    st = ensure_buffers(kProbeMaxBytes, rank_count);
    if (!st.ok()) return st;

    for (std::size_t probe_size : kProbeSizes) {
        const std::size_t count = probe_size / sizeof(std::uint16_t);
        std::vector<std::vector<std::uint16_t>> host_in(rank_count,
                                                        std::vector<std::uint16_t>(count));
        std::vector<std::uint16_t> want(count);
        for (std::size_t r = 0; r < rank_count; ++r) {
            for (std::size_t i = 0; i < count; ++i) {
                host_in[r][i] = static_cast<std::uint16_t>(
                    0x4000u + ((static_cast<std::uint32_t>(r) * 7u + i) % 512u));
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            float acc = 0.0f;
            for (std::size_t r = 0; r < rank_count; ++r) {
                acc += tp_bf16_bits_to_f32(host_in[r][i]);
            }
            want[i] = ps::f32_to_bf16_rne(acc);
        }

        std::vector<TpSumTarget> targets(rank_count);
        std::vector<hipStream_t> streams(rank_count);
        std::vector<hipEvent_t> ready(rank_count);
        for (std::size_t r = 0; r < rank_count; ++r) {
            targets[r].ptr = probe_buffers_[r];
            targets[r].rows = 1;
            targets[r].row_stride = static_cast<std::uint32_t>(probe_size / 2u);
            targets[r].feature_count = static_cast<std::uint32_t>(probe_size / 2u);
            targets[r].dtype = ValueDType::BF16;
            streams[r] = probe_streams_[r];
            ready[r] = probe_ready_[r];
        }
        TpSumInvocation invocation{targets, streams, ready};

        for (int round = 0; round < kProbeRounds; ++round) {
            for (std::size_t r = 0; r < rank_count; ++r) {
                auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
                if (!scope.ok()) return scope.status();
                if (hipMemcpy(probe_buffers_[r], host_in[r].data(), probe_size,
                              hipMemcpyHostToDevice) != hipSuccess) {
                    return Status::hip_error("tp probe upload",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
                if (hipEventRecord(probe_ready_[r], probe_streams_[r]) != hipSuccess) {
                    return Status::hip_error("tp probe record",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
            st = enqueue_sum(invocation);
            if (!st.ok()) return st;
            for (std::size_t r = 0; r < rank_count; ++r) {
                auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
                if (!scope.ok()) return scope.status();
                if (hipStreamSynchronize(probe_streams_[r]) != hipSuccess) {
                    return Status::hip_error("tp probe sync",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
            std::vector<std::uint16_t> got(count);
            bool round_ok = true;
            for (std::size_t r = 0; r < rank_count && round_ok; ++r) {
                auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
                if (!scope.ok()) return scope.status();
                if (hipMemcpy(got.data(), probe_buffers_[r], probe_size,
                              hipMemcpyDeviceToHost) != hipSuccess) {
                    return Status::hip_error("tp probe download",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
                for (std::size_t i = 0; i < count; ++i) {
                    if (got[i] != want[i]) {
                        round_ok = false;
                        break;
                    }
                }
            }
            if (!round_ok) {
                return Status::unsupported(
                    "hip peer transport self-test failed (peer writes are unreliable)",
                    __FILE__, __LINE__);
            }
        }
    }
    verified_ = true;
    return Status::make_ok();
}

Status HipPeerTpTransport::sum_hidden(const TpSumInvocation& invocation) {
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, devices_.size(), bytes);
    if (!st.ok()) return st;
    if (!verified_) {
        st = verify();
        if (!st.ok()) return st;
    }
    st = ensure_buffers(bytes, devices_.size());
    if (!st.ok()) return st;
    return enqueue_sum(invocation);
}

}  // namespace ps::runtime
