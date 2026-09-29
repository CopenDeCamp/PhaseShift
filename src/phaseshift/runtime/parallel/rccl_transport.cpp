#include <phaseshift/runtime/parallel/rccl_transport.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ps::runtime {

namespace {

constexpr uint32_t kNoPeer = 0xFFFFFFFFu;

constexpr ncclDataType_t kRcclDtypes[] = {
    ncclBfloat16,
    ncclFloat32,
    ncclFloat16,
    ncclInt32,
};

constexpr uint64_t kRcclDtypeBytes[] = {2, 4, 2, 4};

bool rccl_debug_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("PHASESHIFT_RCCL_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void log_comm(const char* phase, CommOperation operation, uint32_t global_rank,
              uint64_t sequence) {
    if (!rccl_debug_enabled()) return;
    std::fprintf(stderr, "rank=%u seq=%llu op=%s %s\n", global_rank,
                 static_cast<unsigned long long>(sequence), comm_operation_name(operation),
                 phase);
}

Status validate_dtype(RcclDataType dtype) {
    const uint32_t index = static_cast<uint32_t>(dtype);
    if (index >= sizeof(kRcclDtypes) / sizeof(kRcclDtypes[0]))
        return Status::invalid_argument("unsupported RCCL datatype", __FILE__, __LINE__);
    return Status::make_ok();
}

Status make_rccl_error(const char* expr, CommOperation operation,
                       const ParallelConfig& config, uint32_t global_rank,
                       uint64_t sequence, uint32_t peer_rank, ncclResult_t result,
                       const char* file, int line) {
    char detail[320];
    std::snprintf(detail, sizeof(detail),
                  "op=%s seq=%llu device=%d pp_rank=%u tp_rank=%u rank=%u peer=%u rccl=%s",
                  comm_operation_name(operation),
                  static_cast<unsigned long long>(sequence),
                  config.devices[global_rank], config.pp_rank_of(global_rank),
                  config.tp_rank_of(global_rank), global_rank, peer_rank,
                  ncclGetErrorString(result));
    return Status::rccl_error(expr, detail, file, line);
}

void discard_status(const Status&) {}

void record_bytes(std::atomic<uint64_t>* counters, std::size_t count,
                  uint32_t global_rank, uint64_t elements, RcclDataType dtype) {
    if (counters == nullptr || global_rank >= count) return;
    const uint64_t bytes = elements * rccl_dtype_bytes(dtype);
    counters[global_rank].fetch_add(bytes, std::memory_order_relaxed);
}

}

const char* rccl_dtype_name(RcclDataType dtype) noexcept {
    switch (dtype) {
        case RcclDataType::BF16: return "BF16";
        case RcclDataType::FP32: return "FP32";
        case RcclDataType::FP16: return "FP16";
        case RcclDataType::INT32: return "INT32";
    }
    return "UNKNOWN";
}

uint64_t rccl_dtype_bytes(RcclDataType dtype) noexcept {
    const uint32_t index = static_cast<uint32_t>(dtype);
    if (index >= sizeof(kRcclDtypeBytes) / sizeof(kRcclDtypeBytes[0])) return 0;
    return kRcclDtypeBytes[index];
}

Result<RcclDataType> rccl_dtype_from_value(ValueDType dtype) {
    switch (dtype) {
        case ValueDType::BF16: return RcclDataType::BF16;
        case ValueDType::F32: return RcclDataType::FP32;
        case ValueDType::I32: return RcclDataType::INT32;
    }
    return Status::unsupported("ValueDType has no RCCL datatype mapping", __FILE__, __LINE__);
}

uint32_t parallel_group_size(const ParallelConfig& config, CommGroup group) noexcept {
    return group == CommGroup::Tensor ? config.tp_size : config.pp_size;
}

Result<std::unique_ptr<RcclTransport>> RcclTransport::create(const ParallelConfig& config) {
    Status vs = config.validate();
    if (!vs.ok()) return vs;

    int previous_device = -1;
    hipError_t herr = hipGetDevice(&previous_device);
    if (herr != hipSuccess)
        return Status::hip_error("hipGetDevice", hipGetErrorString(herr), __FILE__, __LINE__);

    std::unique_ptr<RcclTransport> transport(new RcclTransport());
    transport->config_ = config;
    const std::size_t world = static_cast<std::size_t>(config.world_size());
    transport->tensor_comms_.assign(world, NCCL_COMM_NULL);
    transport->pipeline_comms_.assign(world, NCCL_COMM_NULL);
    transport->bytes_rank_count_ = world;
    transport->bytes_per_rank_ = std::make_unique<std::atomic<uint64_t>[]>(world);
    transport->initialized_ = true;

    auto fail = [&](Status cause, const char* expr) -> Status {
        Status cleanup = transport->shutdown();
        if (cleanup.ok()) return cause;
        std::string combined = cause.message() + "; cleanup failed: " + cleanup.message();
        return Status::rccl_error(expr, combined.c_str(), __FILE__, __LINE__);
    };

    for (uint32_t pp_rank = 0; pp_rank < config.pp_size; ++pp_rank) {
        std::vector<int> devs = config.tensor_group_devices(pp_rank);
        ncclComm_t* base =
            transport->tensor_comms_.data() + static_cast<std::size_t>(pp_rank) * config.tp_size;
        ncclResult_t r =
            ncclCommInitAll(base, static_cast<int>(config.tp_size), devs.data());
        if (r != ncclSuccess) {
            char detail[256];
            std::snprintf(detail, sizeof(detail),
                          "tensor clique pp_rank=%u ndev=%u rccl=%s", pp_rank, config.tp_size,
                          ncclGetErrorString(r));
            return fail(Status::rccl_error("ncclCommInitAll(tensor)", detail, __FILE__,
                                           __LINE__),
                        "ncclCommInitAll(tensor)");
        }
    }

    for (uint32_t tp_rank = 0; tp_rank < config.tp_size; ++tp_rank) {
        std::vector<int> devs = config.pipeline_group_devices(tp_rank);
        ncclComm_t* base =
            transport->pipeline_comms_.data() + static_cast<std::size_t>(tp_rank) * config.pp_size;
        ncclResult_t r =
            ncclCommInitAll(base, static_cast<int>(config.pp_size), devs.data());
        if (r != ncclSuccess) {
            char detail[256];
            std::snprintf(detail, sizeof(detail),
                          "pipeline clique tp_rank=%u ndev=%u rccl=%s", tp_rank, config.pp_size,
                          ncclGetErrorString(r));
            return fail(Status::rccl_error("ncclCommInitAll(pipeline)", detail, __FILE__,
                                           __LINE__),
                        "ncclCommInitAll(pipeline)");
        }
    }

    herr = hipSetDevice(previous_device);
    if (herr != hipSuccess)
        return Status::hip_error("hipSetDevice", hipGetErrorString(herr), __FILE__, __LINE__);
    return transport;
}

Status RcclTransport::shutdown() {
    if (!initialized_) return Status::make_ok();
    Status first = Status::make_ok();
    auto destroy = [&](ncclComm_t& comm, const char* expr, std::size_t index,
                       const char* table) {
        if (comm == NCCL_COMM_NULL) return;
        ncclResult_t r = ncclCommDestroy(comm);
        comm = NCCL_COMM_NULL;
        if (r != ncclSuccess && first.ok()) {
            char detail[256];
            std::snprintf(detail, sizeof(detail), "%s[%zu] rccl=%s", table, index,
                          ncclGetErrorString(r));
            first = Status::rccl_error(expr, detail, __FILE__, __LINE__);
        }
    };
    for (std::size_t i = 0; i < tensor_comms_.size(); ++i)
        destroy(tensor_comms_[i], "ncclCommDestroy(tensor)", i, "tensor_comms");
    for (std::size_t i = 0; i < pipeline_comms_.size(); ++i)
        destroy(pipeline_comms_[i], "ncclCommDestroy(pipeline)", i, "pipeline_comms");
    tensor_comms_.clear();
    pipeline_comms_.clear();
    initialized_ = false;
    return first;
}

RcclTransport::~RcclTransport() {
    discard_status(shutdown());
}

Result<RcclTransport::GroupRef> RcclTransport::resolve(uint32_t global_rank,
                                                       CommGroup group) const {
    if (!initialized_)
        return Status::invalid_state("RCCL transport is not initialized", __FILE__, __LINE__);
    if (global_rank >= config_.world_size())
        return Status::out_of_range("global rank out of range", __FILE__, __LINE__);

    GroupRef ref;
    ref.device = config_.devices[global_rank];
    if (group == CommGroup::Tensor) {
        ref.comm = tensor_comms_[global_rank];
        ref.local_rank = static_cast<int>(config_.tp_rank_of(global_rank));
    } else {
        const std::size_t index = static_cast<std::size_t>(config_.tp_rank_of(global_rank)) *
                                      config_.pp_size +
                                  config_.pp_rank_of(global_rank);
        ref.comm = pipeline_comms_[index];
        ref.local_rank = static_cast<int>(config_.pp_rank_of(global_rank));
    }
    if (ref.comm == NCCL_COMM_NULL)
        return Status::invalid_state("communicator is not initialized", __FILE__, __LINE__);

    int current = -1;
    hipError_t herr = hipGetDevice(&current);
    if (herr != hipSuccess)
        return Status::hip_error("hipGetDevice", hipGetErrorString(herr), __FILE__, __LINE__);
    if (current != ref.device)
        return Status::invalid_state(
            "current HIP device does not match the rank device", __FILE__, __LINE__);
    return ref;
}

Result<int> RcclTransport::peer_local_rank(uint32_t global_rank, CommGroup group,
                                           uint32_t peer_rank) const {
    if (peer_rank >= config_.world_size())
        return Status::out_of_range("peer rank out of range", __FILE__, __LINE__);
    if (peer_rank == global_rank)
        return Status::invalid_argument("peer rank equals the local rank", __FILE__, __LINE__);
    const bool same = group == CommGroup::Tensor ? config_.same_tensor_group(global_rank, peer_rank)
                                                 : config_.same_pipeline_group(global_rank, peer_rank);
    if (!same)
        return Status::invalid_argument("peer is not a member of the requested group",
                                        __FILE__, __LINE__);
    if (group == CommGroup::Tensor)
        return static_cast<int>(config_.tp_rank_of(peer_rank));
    return static_cast<int>(config_.pp_rank_of(peer_rank));
}

Status RcclTransport::all_reduce_sum(uint32_t global_rank, CommGroup group, const void* send,
                                     void* recv, uint64_t elements, RcclDataType dtype,
                                     hipStream_t stream) {
    Status ds = validate_dtype(dtype);
    if (!ds.ok()) return ds;
    if (send == nullptr || recv == nullptr)
        return Status::invalid_argument("all_reduce buffer is null", __FILE__, __LINE__);
    if (elements == 0)
        return Status::invalid_argument("all_reduce elements must be > 0", __FILE__, __LINE__);
    auto self = resolve(global_rank, group);
    if (!self.ok()) return self.status();

    const uint64_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    log_comm("begin", CommOperation::AllReduceSum, global_rank, sequence);
    ncclResult_t r = ncclAllReduce(send, recv, elements, kRcclDtypes[static_cast<uint32_t>(dtype)],
                                   ncclSum, self.value().comm, stream);
    if (r != ncclSuccess)
        return make_rccl_error("ncclAllReduce", CommOperation::AllReduceSum, config_,
                               global_rank, sequence, kNoPeer, r, __FILE__, __LINE__);
    record_bytes(bytes_per_rank_.get(), bytes_rank_count_, global_rank, elements,
                 dtype);
    log_comm("enqueued", CommOperation::AllReduceSum, global_rank, sequence);
    return Status::make_ok();
}

Status RcclTransport::send(uint32_t global_rank, CommGroup group, uint32_t peer_rank,
                           const void* send_buffer, uint64_t elements, RcclDataType dtype,
                           hipStream_t stream) {
    Status ds = validate_dtype(dtype);
    if (!ds.ok()) return ds;
    if (send_buffer == nullptr)
        return Status::invalid_argument("send buffer is null", __FILE__, __LINE__);
    if (elements == 0)
        return Status::invalid_argument("send elements must be > 0", __FILE__, __LINE__);
    auto self = resolve(global_rank, group);
    if (!self.ok()) return self.status();
    auto peer = peer_local_rank(global_rank, group, peer_rank);
    if (!peer.ok()) return peer.status();

    const uint64_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    log_comm("begin", CommOperation::Send, global_rank, sequence);
    ncclResult_t r = ncclSend(send_buffer, elements, kRcclDtypes[static_cast<uint32_t>(dtype)],
                              peer.value(), self.value().comm, stream);
    if (r != ncclSuccess)
        return make_rccl_error("ncclSend", CommOperation::Send, config_, global_rank, sequence,
                               peer_rank, r, __FILE__, __LINE__);
    record_bytes(bytes_per_rank_.get(), bytes_rank_count_, global_rank, elements,
                 dtype);
    log_comm("enqueued", CommOperation::Send, global_rank, sequence);
    return Status::make_ok();
}

Status RcclTransport::recv(uint32_t global_rank, CommGroup group, uint32_t peer_rank,
                           void* recv_buffer, uint64_t elements, RcclDataType dtype,
                           hipStream_t stream) {
    Status ds = validate_dtype(dtype);
    if (!ds.ok()) return ds;
    if (recv_buffer == nullptr)
        return Status::invalid_argument("recv buffer is null", __FILE__, __LINE__);
    if (elements == 0)
        return Status::invalid_argument("recv elements must be > 0", __FILE__, __LINE__);
    auto self = resolve(global_rank, group);
    if (!self.ok()) return self.status();
    auto peer = peer_local_rank(global_rank, group, peer_rank);
    if (!peer.ok()) return peer.status();

    const uint64_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    log_comm("begin", CommOperation::Recv, global_rank, sequence);
    ncclResult_t r = ncclRecv(recv_buffer, elements, kRcclDtypes[static_cast<uint32_t>(dtype)],
                              peer.value(), self.value().comm, stream);
    if (r != ncclSuccess)
        return make_rccl_error("ncclRecv", CommOperation::Recv, config_, global_rank, sequence,
                               peer_rank, r, __FILE__, __LINE__);
    record_bytes(bytes_per_rank_.get(), bytes_rank_count_, global_rank, elements,
                 dtype);
    log_comm("enqueued", CommOperation::Recv, global_rank, sequence);
    return Status::make_ok();
}

Status RcclTransport::broadcast(uint32_t global_rank, CommGroup group, uint32_t root_rank,
                                void* buffer, uint64_t elements, RcclDataType dtype,
                                hipStream_t stream) {
    Status ds = validate_dtype(dtype);
    if (!ds.ok()) return ds;
    if (buffer == nullptr)
        return Status::invalid_argument("broadcast buffer is null", __FILE__, __LINE__);
    if (elements == 0)
        return Status::invalid_argument("broadcast elements must be > 0", __FILE__, __LINE__);
    auto self = resolve(global_rank, group);
    if (!self.ok()) return self.status();

    int root_local = -1;
    if (group == CommGroup::Tensor) {
        if (!config_.same_tensor_group(global_rank, root_rank))
            return Status::invalid_argument("root is not a member of the tensor group",
                                            __FILE__, __LINE__);
        root_local = static_cast<int>(config_.tp_rank_of(root_rank));
    } else {
        if (!config_.same_pipeline_group(global_rank, root_rank))
            return Status::invalid_argument("root is not a member of the pipeline group",
                                            __FILE__, __LINE__);
        root_local = static_cast<int>(config_.pp_rank_of(root_rank));
    }

    const uint64_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    log_comm("begin", CommOperation::Broadcast, global_rank, sequence);
    ncclResult_t r = ncclBroadcast(buffer, buffer, elements,
                                   kRcclDtypes[static_cast<uint32_t>(dtype)], root_local,
                                   self.value().comm, stream);
    if (r != ncclSuccess)
        return make_rccl_error("ncclBroadcast", CommOperation::Broadcast, config_, global_rank,
                               sequence, root_rank, r, __FILE__, __LINE__);
    record_bytes(bytes_per_rank_.get(), bytes_rank_count_, global_rank, elements,
                 dtype);
    log_comm("enqueued", CommOperation::Broadcast, global_rank, sequence);
    return Status::make_ok();
}

}
