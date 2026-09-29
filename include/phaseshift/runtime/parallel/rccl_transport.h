#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/graph/value_type.h>
#include <phaseshift/runtime/parallel/comm_types.h>
#include <phaseshift/runtime/parallel/parallel_config.h>

#include <hip/hip_runtime.h>
#include <nccl.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace ps::runtime {

enum class RcclDataType : uint8_t {
    BF16 = 0,
    FP32 = 1,
    FP16 = 2,
    INT32 = 3,
};

const char* comm_group_name(CommGroup group) noexcept;
const char* comm_operation_name(CommOperation operation) noexcept;
const char* rccl_dtype_name(RcclDataType dtype) noexcept;

uint64_t rccl_dtype_bytes(RcclDataType dtype) noexcept;

Result<RcclDataType> rccl_dtype_from_value(ValueDType dtype);

uint32_t parallel_group_size(const ParallelConfig& config, CommGroup group) noexcept;

class RcclTransport {
public:
    static Result<std::unique_ptr<RcclTransport>> create(const ParallelConfig& config);

    RcclTransport(const RcclTransport&) = delete;
    RcclTransport& operator=(const RcclTransport&) = delete;

    const ParallelConfig& config() const noexcept { return config_; }

    uint64_t issued_operations() const noexcept {
        return sequence_.load(std::memory_order_relaxed);
    }

    uint64_t issued_bytes(uint32_t global_rank) const noexcept {
        if (global_rank >= bytes_rank_count_) return 0;
        return bytes_per_rank_[global_rank].load(std::memory_order_relaxed);
    }

    Status all_reduce_sum(uint32_t global_rank, CommGroup group, const void* send,
                          void* recv, uint64_t elements, RcclDataType dtype,
                          hipStream_t stream);

    Status send(uint32_t global_rank, CommGroup group, uint32_t peer_rank,
                const void* send, uint64_t elements, RcclDataType dtype, hipStream_t stream);

    Status recv(uint32_t global_rank, CommGroup group, uint32_t peer_rank, void* recv,
                uint64_t elements, RcclDataType dtype, hipStream_t stream);

    Status broadcast(uint32_t global_rank, CommGroup group, uint32_t root_rank,
                     void* buffer, uint64_t elements, RcclDataType dtype, hipStream_t stream);

    Status shutdown();

    ~RcclTransport();

private:
    RcclTransport() = default;

    struct GroupRef {
        ncclComm_t comm = NCCL_COMM_NULL;
        int local_rank = 0;
        int device = 0;
    };

    Result<GroupRef> resolve(uint32_t global_rank, CommGroup group) const;
    Result<int> peer_local_rank(uint32_t global_rank, CommGroup group,
                                uint32_t peer_rank) const;

    ParallelConfig config_;
    std::vector<ncclComm_t> tensor_comms_;
    std::vector<ncclComm_t> pipeline_comms_;
    std::unique_ptr<std::atomic<uint64_t>[]> bytes_per_rank_;
    std::size_t bytes_rank_count_ = 0;
    std::atomic<uint64_t> sequence_{0};
    bool initialized_ = false;
};

}
