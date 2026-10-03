#pragma once

#include <phaseshift/runtime/tp/tp_execution.h>

#include <cstddef>
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
    Status ensure_buffers(std::size_t bytes, std::size_t rank_count);
    Status enqueue_sum(const TpSumInvocation& invocation);

    std::vector<int> devices_;
    hipStream_t transport_stream_ = nullptr;
    hipEvent_t reduce_done_ = nullptr;
    std::vector<hipEvent_t> bcast_done_;
    std::vector<hipEvent_t> scratch_done_;
    std::vector<void*> scratch_;
    void* leader_result_ = nullptr;
    std::size_t leader_result_bytes_ = 0;
    void* src_ptr_dev_ = nullptr;
    void* src_ptr_host_[2] = {nullptr, nullptr};
    std::size_t src_ptr_capacity_ = 0;
    std::size_t src_ptr_parity_ = 0;
    std::vector<const void*> host_src_ptrs_;
    std::vector<void*> probe_buffers_;
    std::vector<hipStream_t> probe_streams_;
    std::vector<hipEvent_t> probe_ready_;
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
    Status ensure_staging(std::size_t bytes, std::size_t count);

    std::vector<int> devices_;
    std::vector<void*> staging_;
    std::vector<float> accum_;
    std::size_t staging_bytes_ = 0;
};

struct TpTransportSelection {
    std::unique_ptr<TpTransport> transport;
    bool peer_access_available = false;
};

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices);

}  // namespace ps::runtime
