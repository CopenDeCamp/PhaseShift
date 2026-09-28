#pragma once

#include <phaseshift/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kVerifyAcceptPrefixSymbol =
    "phaseshift_gpu_mcu_verify_accept_prefix";

inline constexpr const char* kVerifyAcceptBatchSymbol =
    "phaseshift_gpu_mcu_verify_accept_batch";

inline constexpr uint32_t kVerifyAcceptPrefixThreads = 32u;
inline constexpr uint32_t kVerifyAcceptPrefixMaxCandidates = 32u;

struct VerifyAcceptBatchArgs {
    const ::ps::runtime::DeviceBatchContext* context = nullptr;
    const int32_t* sampled_tokens = nullptr;
    uint32_t* committed_counts = nullptr;
    uint32_t max_candidates = 0;
};

static_assert(sizeof(VerifyAcceptBatchArgs) == 32);
static_assert(alignof(VerifyAcceptBatchArgs) == 8);
static_assert(offsetof(VerifyAcceptBatchArgs, context) == 0);
static_assert(offsetof(VerifyAcceptBatchArgs, sampled_tokens) == 8);
static_assert(offsetof(VerifyAcceptBatchArgs, committed_counts) == 16);
static_assert(offsetof(VerifyAcceptBatchArgs, max_candidates) == 24);

struct VerifyAcceptPrefixArgs {
    const int32_t* candidates = nullptr;
    const int32_t* sampled_tokens = nullptr;
    const uint32_t* candidate_counts = nullptr;
    uint32_t* committed_counts = nullptr;
    uint32_t request_count = 0;
    uint32_t candidate_stride = 0;
    uint32_t max_candidates = 0;
    uint32_t reserved = 0;
    const ::ps::runtime::DeviceRequestDescriptor* requests = nullptr;
    const uint32_t* output_rows = nullptr;
};

static_assert(sizeof(VerifyAcceptPrefixArgs) == 64);
static_assert(alignof(VerifyAcceptPrefixArgs) == 8);
static_assert(offsetof(VerifyAcceptPrefixArgs, candidates) == 0);
static_assert(offsetof(VerifyAcceptPrefixArgs, sampled_tokens) == 8);
static_assert(offsetof(VerifyAcceptPrefixArgs, candidate_counts) == 16);
static_assert(offsetof(VerifyAcceptPrefixArgs, committed_counts) == 24);
static_assert(offsetof(VerifyAcceptPrefixArgs, request_count) == 32);
static_assert(offsetof(VerifyAcceptPrefixArgs, candidate_stride) == 36);
static_assert(offsetof(VerifyAcceptPrefixArgs, max_candidates) == 40);
static_assert(offsetof(VerifyAcceptPrefixArgs, reserved) == 44);
static_assert(offsetof(VerifyAcceptPrefixArgs, requests) == 48);
static_assert(offsetof(VerifyAcceptPrefixArgs, output_rows) == 56);

hipError_t launch_verify_accept_prefix(
    const int32_t* candidates, const int32_t* sampled_tokens,
    const uint32_t* candidate_counts, uint32_t* committed_counts,
    uint32_t request_count, uint32_t candidate_stride,
    uint32_t max_candidates, hipStream_t stream);

hipError_t launch_verify_accept_batch(
    const ::ps::runtime::DeviceBatchContext* context,
    const int32_t* sampled_tokens, uint32_t* committed_counts,
    uint32_t max_candidates, hipStream_t stream);

}  // namespace ps::kernel
