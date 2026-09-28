#include <phaseshift/runtime/parallel/comm_bridge.h>

namespace ps::runtime {

Status launch_rccl_communication(void* self, const CommDescriptor& descriptor,
                                 const void* send, void* recv, uint64_t elements,
                                 hipStream_t stream) {
    if (self == nullptr)
        return Status::invalid_state("communication endpoint is not bound", __FILE__,
                                     __LINE__);
    auto* endpoint = static_cast<RcclCommEndpoint*>(self);
    if (endpoint->transport == nullptr)
        return Status::invalid_state("communication transport is not bound", __FILE__,
                                     __LINE__);
    auto dtype = rccl_dtype_from_value(descriptor.dtype);
    if (!dtype.ok()) return dtype.status();
    RcclTransport& transport = *endpoint->transport;
    const uint32_t rank = endpoint->global_rank;
    switch (descriptor.operation) {
        case CommOperation::AllReduceSum:
            return transport.all_reduce_sum(rank, descriptor.group, send, recv, elements,
                                             dtype.value(), stream);
        case CommOperation::Send:
            return transport.send(rank, descriptor.group, descriptor.peer, send, elements,
                                  dtype.value(), stream);
        case CommOperation::Recv:
            return transport.recv(rank, descriptor.group, descriptor.peer, recv, elements,
                                  dtype.value(), stream);
        case CommOperation::Broadcast:
            return transport.broadcast(rank, descriptor.group, descriptor.peer, recv,
                                       elements, dtype.value(), stream);
    }
    return Status::unsupported("unknown communication operation", __FILE__, __LINE__);
}

}
