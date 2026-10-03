#pragma once

#include <phaseshift/runtime/tp/tp_execution.h>

#include <memory>
#include <vector>

namespace ps::runtime {

class HipPeerTpTransport final : public TpTransport {
public:
    static Result<std::unique_ptr<HipPeerTpTransport>> create(
        const std::vector<int>& devices);
    ~HipPeerTpTransport() override;

    const char* name() const noexcept override { return "hip-p2p"; }
    Status verify() override;
    Status sum_hidden(const TpSumInvocation& invocation) override;

private:
    explicit HipPeerTpTransport(std::vector<int> devices);

    Status enable_peer_access();

    std::vector<int> devices_;
    std::vector<hipEvent_t> stage_events_;
    std::vector<void*> probe_buffers_;
    std::vector<hipStream_t> probe_streams_;
    bool peer_access_enabled_ = false;
    bool verified_ = false;
};

class HostMediatedTpTransport final : public TpTransport {
public:
    explicit HostMediatedTpTransport(std::vector<int> devices);
    ~HostMediatedTpTransport() override;

    const char* name() const noexcept override { return "host-mediated"; }
    Status sum_hidden(const TpSumInvocation& invocation) override;

private:
    std::vector<int> devices_;
};

struct TpTransportSelection {
    std::unique_ptr<TpTransport> transport;
    bool peer_access_available = false;
};

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices);

}  // namespace ps::runtime
