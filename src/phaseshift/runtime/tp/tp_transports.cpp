#include <phaseshift/runtime/tp/tp_transports.h>

#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/types.h>

#include "tp_transport_internal.h"

#include <cstring>

namespace ps::runtime {

namespace {

void add_host(void* dst, const void* src, std::size_t bytes, ValueDType dtype) {
    if (dtype == ValueDType::F32) {
        auto* d = static_cast<float*>(dst);
        const auto* s = static_cast<const float*>(src);
        const std::size_t n = bytes / sizeof(float);
        for (std::size_t i = 0; i < n; ++i) d[i] += s[i];
        return;
    }
    auto* d = static_cast<uint16_t*>(dst);
    const auto* s = static_cast<const uint16_t*>(src);
    const std::size_t n = bytes / sizeof(uint16_t);
    for (std::size_t i = 0; i < n; ++i) {
        float a;
        float b;
        const uint32_t ua = static_cast<uint32_t>(d[i]) << 16;
        const uint32_t ub = static_cast<uint32_t>(s[i]) << 16;
        std::memcpy(&a, &ua, sizeof(a));
        std::memcpy(&b, &ub, sizeof(b));
        d[i] = f32_to_bf16_rne(a + b);
    }
}

}

HostMediatedTpTransport::HostMediatedTpTransport(std::vector<int> devices)
    : devices_(std::move(devices)) {}

HostMediatedTpTransport::~HostMediatedTpTransport() = default;

Status HostMediatedTpTransport::sum_hidden(const TpSumInvocation& invocation) {
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, devices_.size(), bytes);
    if (!st.ok()) return st;

    std::vector<std::vector<uint8_t>> host(devices_.size());
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        const hipError_t sync = hipStreamSynchronize(invocation.streams[r]);
        if (sync != hipSuccess) {
            return Status::hip_error("host transport stream sync", hipGetErrorString(sync),
                                     __FILE__, __LINE__);
        }
        host[r].resize(bytes);
        const hipError_t copy = hipMemcpy(host[r].data(), invocation.targets[r].ptr, bytes,
                                          hipMemcpyDeviceToHost);
        if (copy != hipSuccess) {
            return Status::hip_error("host transport D2H", hipGetErrorString(copy),
                                     __FILE__, __LINE__);
        }
    }
    for (std::size_t r = 1; r < devices_.size(); ++r) {
        add_host(host[0].data(), host[r].data(), bytes, invocation.targets[0].dtype);
    }
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        const hipError_t copy = hipMemcpy(invocation.targets[r].ptr, host[0].data(), bytes,
                                          hipMemcpyHostToDevice);
        if (copy != hipSuccess) {
            return Status::hip_error("host transport H2D", hipGetErrorString(copy),
                                     __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices) {
    if (devices.size() < 2) {
        return Status::invalid_argument("tp transport requires at least 2 devices",
                                        __FILE__, __LINE__);
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        for (std::size_t j = 0; j < devices.size(); ++j) {
            if (i == j) continue;
            int can_peer = 0;
            const hipError_t e = hipDeviceCanAccessPeer(&can_peer, devices[i], devices[j]);
            if (e != hipSuccess) {
                return Status::hip_error("hipDeviceCanAccessPeer", hipGetErrorString(e),
                                         __FILE__, __LINE__);
            }
            if (can_peer == 0) {
                TpTransportSelection selection;
                selection.peer_access_available = false;
                selection.transport =
                    std::make_unique<HostMediatedTpTransport>(devices);
                return selection;
            }
        }
    }
    auto peer = HipPeerTpTransport::create(devices);
    if (!peer.ok()) return peer.status();
    TpTransportSelection selection;
    selection.peer_access_available = true;
    selection.transport = peer.release();
    return selection;
}

}  // namespace ps::runtime
