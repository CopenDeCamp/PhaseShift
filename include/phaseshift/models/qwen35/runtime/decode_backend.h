#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>

namespace ps {
namespace qwen35 {
namespace runtime {

enum class DecodeBackend : uint8_t {
    Host = 0,
    GpuMcu = 1,
};

enum class DecodeBackendReason : uint8_t {
    Selected = 0,
    BackendIsHost = 1,
    NotDecode = 2,
    MultipleRequests = 3,
    NotSingleToken = 4,
    NotSingleRow = 5,
    SpeculativeVerify = 6,
    PrefillPresent = 7,
    KvDtypeNotBf16 = 8,
    ImatrixCollector = 9,
    ValueTrace = 10,
    StreamWaitUnsupported = 12,
    McuBodyRangeUnavailable = 13,
    TensorParallel = 14,
    StochasticSampling = 15,
};

struct DecodeBackendInputs {
    DecodeBackend requested = DecodeBackend::Host;
    bool role_is_decode = true;
    uint32_t num_requests = 1u;
    uint32_t num_tokens = 1u;
    uint32_t actual_rows = 1u;
    bool speculative_verify = false;
    bool prefill_present = false;
    bool kv_dtype_is_bf16 = true;
    bool tensor_parallel_configured = false;
    uint32_t stochastic_output_count = 0u;
    bool imatrix_collector_present = false;
    bool value_trace_present = false;
    bool stream_wait_value_supported = true;
    bool mcu_body_range_valid = true;
    bool persistent_ready = false;
};

struct DecodeBackendDecision {
    DecodeBackend backend = DecodeBackend::Host;
    DecodeBackendReason reason = DecodeBackendReason::BackendIsHost;
    bool eligible = false;
};

struct DecodeBackendConfigInputs {
    DecodeBackend requested = DecodeBackend::Host;
    bool kv_dtype_is_bf16 = true;
    bool tensor_parallel_configured = false;
};

struct DecodeRuntimeCounters {
    uint64_t plan_compile_count = 0;
    uint64_t plan_upload_count = 0;
    uint64_t plan_switch_count = 0;
    uint64_t mcu_run_count = 0;
    uint64_t host_fallback_count = 0;
};

DecodeBackendDecision decide_decode_backend(const DecodeBackendInputs& inputs);

DecodeBackendDecision decide_decode_backend_config(
    const DecodeBackendConfigInputs& inputs);

Status decode_backend_execution_error(const DecodeBackendDecision& decision);

const char* decode_backend_name(DecodeBackend backend);
const char* decode_backend_reason_name(DecodeBackendReason reason);

bool parse_decode_backend(const char* text, DecodeBackend* out);

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
