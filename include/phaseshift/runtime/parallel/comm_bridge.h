#pragma once

#include <phaseshift/runtime/parallel/rccl_transport.h>
#include <phaseshift/runtime/program/comm_launcher.h>

#include <cstdint>

namespace ps::runtime {

struct RcclCommEndpoint {
    RcclTransport* transport = nullptr;
    uint32_t global_rank = 0;
};

Status launch_rccl_communication(void* self, const CommDescriptor& descriptor,
                                 const void* send, void* recv, uint64_t elements,
                                 hipStream_t stream);

}
