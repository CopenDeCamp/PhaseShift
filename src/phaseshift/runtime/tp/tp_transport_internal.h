#pragma once

#include <phaseshift/runtime/tp/tp_execution.h>

namespace ps::runtime {

inline uint64_t tp_target_bytes(const TpSumTarget& target) {
    return target.rows * static_cast<uint64_t>(target.row_stride) *
           value_dtype_bytes(target.dtype);
}

inline Status validate_tp_sum_invocation(const TpSumInvocation& invocation,
                                         std::size_t rank_count,
                                         std::size_t& bytes_out) {
    if (rank_count < 2) {
        return Status::invalid_argument("tp transport requires at least 2 ranks",
                                        __FILE__, __LINE__);
    }
    if (invocation.targets.size() != rank_count ||
        invocation.streams.size() != rank_count ||
        invocation.ready.size() != rank_count) {
        return Status::invalid_argument("tp sum invocation rank count mismatch",
                                        __FILE__, __LINE__);
    }
    const TpSumTarget& first = invocation.targets[0];
    if (first.ptr == nullptr || first.rows == 0 || first.row_stride == 0 ||
        first.feature_count == 0) {
        return Status::invalid_argument("tp sum target is empty", __FILE__, __LINE__);
    }
    if (first.dtype != ValueDType::BF16 && first.dtype != ValueDType::F32) {
        return Status::unsupported("tp sum supports BF16 and F32 targets only",
                                   __FILE__, __LINE__);
    }
    const uint64_t bytes = tp_target_bytes(first);
    for (std::size_t r = 1; r < rank_count; ++r) {
        const TpSumTarget& t = invocation.targets[r];
        if (t.ptr == nullptr || t.rows != first.rows ||
            t.row_stride != first.row_stride || t.feature_count != first.feature_count ||
            t.dtype != first.dtype) {
            return Status::invalid_argument("tp sum targets do not match", __FILE__, __LINE__);
        }
    }
    bytes_out = static_cast<std::size_t>(bytes);
    return Status::make_ok();
}

}  // namespace ps::runtime
