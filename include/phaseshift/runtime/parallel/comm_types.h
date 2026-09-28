#pragma once

#include <cstdint>

namespace ps::runtime {

enum class CommGroup : uint8_t {
    Tensor = 0,
    Pipeline = 1,
};

enum class CommOperation : uint8_t {
    AllReduceSum = 0,
    Send = 1,
    Recv = 2,
    Broadcast = 3,
};

inline const char* comm_group_name(CommGroup group) noexcept {
    switch (group) {
        case CommGroup::Tensor: return "TENSOR";
        case CommGroup::Pipeline: return "PIPELINE";
    }
    return "UNKNOWN";
}

inline const char* comm_operation_name(CommOperation operation) noexcept {
    switch (operation) {
        case CommOperation::AllReduceSum: return "ALLREDUCE";
        case CommOperation::Send: return "SEND";
        case CommOperation::Recv: return "RECV";
        case CommOperation::Broadcast: return "BROADCAST";
    }
    return "UNKNOWN";
}

}
