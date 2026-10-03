#include <phaseshift/runtime/tp/tp_transports.h>

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/types.h>

#include "tp_transport_internal.h"

#include <cstring>
#include <vector>

namespace ps::runtime {

namespace {

constexpr std::size_t kProbeBytes = 8192;
constexpr int kProbeRounds = 16;

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

__global__ void tp_sum_bf16_kernel(const uint16_t* a, const uint16_t* b, uint16_t* out,
                                   uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = tp_f32_to_bf16(tp_bf16_to_f32(a[i]) + tp_bf16_to_f32(b[i]));
    }
}

__global__ void tp_sum_f32_kernel(const float* a, const float* b, float* out,
                                  uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = a[i] + b[i];
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

float tp_bf16_bits_to_f32(std::uint16_t v) {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

Status launch_pair_sum(void* dst, const void* a, const void* b, std::size_t bytes,
                       ValueDType dtype, hipStream_t stream) {
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_sum_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const float*>(a), static_cast<const float*>(b),
            static_cast<float*>(dst), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_sum_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
            static_cast<uint16_t*>(dst), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp_sum kernel launch", hipGetErrorString(err),
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
    for (std::size_t r = 0; r < probe_buffers_.size(); ++r) {
        if (probe_buffers_[r] == nullptr) continue;
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) continue;
        ps::gpu::discard_cleanup_result(hipFree(probe_buffers_[r]));
        probe_buffers_[r] = nullptr;
        if (probe_streams_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(probe_streams_[r]));
            probe_streams_[r] = nullptr;
        }
        if (r < stage_events_.size() && stage_events_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipEventDestroy(stage_events_[r]));
            stage_events_[r] = nullptr;
        }
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
    const std::size_t rank_count = devices.size();
    transport->stage_events_.assign(rank_count, nullptr);
    transport->probe_buffers_.assign(rank_count, nullptr);
    transport->probe_streams_.assign(rank_count, nullptr);
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices[r]);
        if (!scope.ok()) return scope.status();
        hipEvent_t event = nullptr;
        if (hipEventCreate(&event) != hipSuccess) {
            return Status::hip_error("hipEventCreate", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        transport->stage_events_[r] = event;
        hipStream_t stream = nullptr;
        if (hipStreamCreate(&stream) != hipSuccess) {
            return Status::hip_error("hipStreamCreate", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        transport->probe_streams_[r] = stream;
        void* buffer = nullptr;
        if (hipMalloc(&buffer, kProbeBytes) != hipSuccess) {
            return Status::hip_error("hipMalloc tp probe", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        transport->probe_buffers_[r] = buffer;
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

Status HipPeerTpTransport::verify() {
    if (verified_) return Status::make_ok();
    Status st = enable_peer_access();
    if (!st.ok()) return st;

    const std::size_t leader = 0;
    const std::size_t peer = 1;
    const std::size_t count = kProbeBytes / sizeof(uint16_t);
    std::vector<std::uint16_t> host_a(count);
    std::vector<std::uint16_t> host_b(count);
    std::vector<std::uint16_t> got_a(count);
    std::vector<std::uint16_t> got_b(count);

    for (int round = 0; round < kProbeRounds; ++round) {
        for (std::size_t i = 0; i < count; ++i) {
            host_a[i] = static_cast<std::uint16_t>(16384u + ((i * 3u + round) % 4096u));
            host_b[i] = static_cast<std::uint16_t>(16640u + ((i * 7u + round) % 4096u));
        }
        {
            auto scope = ps::gpu::ScopedDevice::create(devices_[peer]);
            if (!scope.ok()) return scope.status();
            if (hipMemcpy(probe_buffers_[peer], host_b.data(), kProbeBytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                return Status::hip_error("tp probe upload", hipGetErrorString(hipGetLastError()),
                                         __FILE__, __LINE__);
            }
            if (hipEventRecord(stage_events_[peer], probe_streams_[peer]) != hipSuccess) {
                return Status::hip_error("tp probe record",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        {
            auto scope = ps::gpu::ScopedDevice::create(devices_[leader]);
            if (!scope.ok()) return scope.status();
            if (hipMemcpy(probe_buffers_[leader], host_a.data(), kProbeBytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                return Status::hip_error("tp probe upload", hipGetErrorString(hipGetLastError()),
                                         __FILE__, __LINE__);
            }
            if (hipStreamWaitEvent(probe_streams_[leader], stage_events_[peer], 0) !=
                hipSuccess) {
                return Status::hip_error("tp probe wait",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            st = launch_pair_sum(probe_buffers_[leader], probe_buffers_[leader],
                                 probe_buffers_[peer], kProbeBytes, ValueDType::BF16,
                                 probe_streams_[leader]);
            if (!st.ok()) return st;
            st = launch_copy(probe_buffers_[peer], probe_buffers_[leader], kProbeBytes,
                             ValueDType::BF16, probe_streams_[leader]);
            if (!st.ok()) return st;
            if (hipEventRecord(stage_events_[leader], probe_streams_[leader]) !=
                hipSuccess) {
                return Status::hip_error("tp probe record",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        {
            auto scope = ps::gpu::ScopedDevice::create(devices_[peer]);
            if (!scope.ok()) return scope.status();
            if (hipStreamWaitEvent(probe_streams_[peer], stage_events_[leader], 0) !=
                hipSuccess) {
                return Status::hip_error("tp probe wait",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            if (hipStreamSynchronize(probe_streams_[peer]) != hipSuccess) {
                return Status::hip_error("tp probe sync",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            if (hipMemcpy(got_b.data(), probe_buffers_[peer], kProbeBytes,
                          hipMemcpyDeviceToHost) != hipSuccess) {
                return Status::hip_error("tp probe download",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        {
            auto scope = ps::gpu::ScopedDevice::create(devices_[leader]);
            if (!scope.ok()) return scope.status();
            if (hipStreamSynchronize(probe_streams_[leader]) != hipSuccess) {
                return Status::hip_error("tp probe sync",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            if (hipMemcpy(got_a.data(), probe_buffers_[leader], kProbeBytes,
                          hipMemcpyDeviceToHost) != hipSuccess) {
                return Status::hip_error("tp probe download",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }

        bool round_ok = true;
        for (std::size_t i = 0; i < count && round_ok; ++i) {
            const float fa = tp_bf16_bits_to_f32(host_a[i]);
            const float fb = tp_bf16_bits_to_f32(host_b[i]);
            const std::uint16_t want = ps::f32_to_bf16_rne(fa + fb);
            if (got_a[i] != want || got_b[i] != want) round_ok = false;
        }
        if (!round_ok) {
            return Status::unsupported(
                "hip peer transport self-test failed (peer reads are unreliable)",
                __FILE__, __LINE__);
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
    const std::size_t leader_rank = 0;

    {
        auto scope = ps::gpu::ScopedDevice::create(devices_[leader_rank]);
        if (!scope.ok()) return scope.status();
        for (std::size_t r = 1; r < devices_.size(); ++r) {
            const hipError_t wait =
                hipStreamWaitEvent(invocation.streams[leader_rank], invocation.ready[r], 0);
            if (wait != hipSuccess) {
                return Status::hip_error("hipStreamWaitEvent peer segment",
                                         hipGetErrorString(wait), __FILE__, __LINE__);
            }
        }
        for (std::size_t r = 1; r < devices_.size(); ++r) {
            st = launch_pair_sum(invocation.targets[leader_rank].ptr,
                                 invocation.targets[leader_rank].ptr,
                                 invocation.targets[r].ptr, bytes,
                                 invocation.targets[0].dtype,
                                 invocation.streams[leader_rank]);
            if (!st.ok()) return st;
        }
        for (std::size_t r = 1; r < devices_.size(); ++r) {
            st = launch_copy(invocation.targets[r].ptr,
                             invocation.targets[leader_rank].ptr, bytes,
                             invocation.targets[0].dtype,
                             invocation.streams[leader_rank]);
            if (!st.ok()) return st;
        }
        const hipError_t record =
            hipEventRecord(stage_events_[leader_rank], invocation.streams[leader_rank]);
        if (record != hipSuccess) {
            return Status::hip_error("hipEventRecord tp combine", hipGetErrorString(record),
                                     __FILE__, __LINE__);
        }
    }

    for (std::size_t r = 1; r < devices_.size(); ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        const hipError_t wait =
            hipStreamWaitEvent(invocation.streams[r], stage_events_[leader_rank], 0);
        if (wait != hipSuccess) {
            return Status::hip_error("hipStreamWaitEvent tp combine result",
                                     hipGetErrorString(wait), __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

}  // namespace ps::runtime
