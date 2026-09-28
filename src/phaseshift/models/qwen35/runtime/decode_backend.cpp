#include <phaseshift/models/qwen35/runtime/decode_backend.h>

#include <strings.h>

#include <string>

namespace ps {
namespace qwen35 {
namespace runtime {

const char* decode_backend_name(DecodeBackend backend) {
    switch (backend) {
        case DecodeBackend::Host:
            return "Host";
        case DecodeBackend::GpuMcu:
            return "GpuMcu";
    }
    return "unknown";
}

bool parse_decode_backend(const char* text, DecodeBackend* out) {
    if (text == nullptr || out == nullptr) {
        return false;
    }
    if (strcasecmp(text, "host") == 0) {
        *out = DecodeBackend::Host;
        return true;
    }
    if (strcasecmp(text, "gpu-mcu") == 0) {
        *out = DecodeBackend::GpuMcu;
        return true;
    }
    return false;
}

const char* decode_backend_reason_name(DecodeBackendReason reason) {
    switch (reason) {
        case DecodeBackendReason::Selected:
            return "Selected";
        case DecodeBackendReason::BackendIsHost:
            return "BackendIsHost";
        case DecodeBackendReason::NotDecode:
            return "NotDecode";
        case DecodeBackendReason::MultipleRequests:
            return "MultipleRequests";
        case DecodeBackendReason::NotSingleToken:
            return "NotSingleToken";
        case DecodeBackendReason::NotSingleRow:
            return "NotSingleRow";
        case DecodeBackendReason::SpeculativeVerify:
            return "SpeculativeVerify";
        case DecodeBackendReason::PrefillPresent:
            return "PrefillPresent";
        case DecodeBackendReason::KvDtypeNotBf16:
            return "KvDtypeNotBf16";
        case DecodeBackendReason::ImatrixCollector:
            return "ImatrixCollector";
        case DecodeBackendReason::ValueTrace:
            return "ValueTrace";
        case DecodeBackendReason::TargetHiddenTaps:
            return "TargetHiddenTaps";
        case DecodeBackendReason::StreamWaitUnsupported:
            return "StreamWaitUnsupported";
        case DecodeBackendReason::McuBodyRangeUnavailable:
            return "McuBodyRangeUnavailable";
        default:
            return "unknown";
    }
}

DecodeBackendDecision decide_decode_backend(const DecodeBackendInputs& inputs) {
    DecodeBackendDecision decision{};
    decision.backend = inputs.requested;
    if (inputs.requested == DecodeBackend::Host) {
        decision.reason = DecodeBackendReason::BackendIsHost;
        return decision;
    }

    DecodeBackendReason reason = DecodeBackendReason::Selected;
    if (!inputs.persistent_ready && !inputs.role_is_decode) {
        reason = DecodeBackendReason::NotDecode;
    } else if (!inputs.persistent_ready && inputs.speculative_verify) {
        reason = DecodeBackendReason::SpeculativeVerify;
    } else if (!inputs.persistent_ready && inputs.num_requests != 1u) {
        reason = DecodeBackendReason::MultipleRequests;
    } else if (!inputs.persistent_ready && inputs.num_tokens != 1u) {
        reason = DecodeBackendReason::NotSingleToken;
    } else if (!inputs.persistent_ready && inputs.actual_rows != 1u) {
        reason = DecodeBackendReason::NotSingleRow;
    } else if (!inputs.persistent_ready && inputs.prefill_present) {
        reason = DecodeBackendReason::PrefillPresent;
    } else if (!inputs.kv_dtype_is_bf16) {
        reason = DecodeBackendReason::KvDtypeNotBf16;
    } else if (inputs.imatrix_collector_present) {
        reason = DecodeBackendReason::ImatrixCollector;
    } else if (inputs.value_trace_present) {
        reason = DecodeBackendReason::ValueTrace;
    } else if (inputs.target_hidden_tap_count != 0u) {
        reason = DecodeBackendReason::TargetHiddenTaps;
    } else if (!inputs.stream_wait_value_supported) {
        reason = DecodeBackendReason::StreamWaitUnsupported;
    } else if (!inputs.mcu_body_range_valid) {
        reason = DecodeBackendReason::McuBodyRangeUnavailable;
    }

    decision.reason = reason;
    decision.eligible = reason == DecodeBackendReason::Selected;
    return decision;
}

Status decode_backend_execution_error(const DecodeBackendDecision& decision) {
    if (decision.backend == DecodeBackend::Host || decision.eligible) {
        return Status::make_ok();
    }
    std::string message = "GPU-MCU backend cannot execute this request: ";
    message += decode_backend_reason_name(decision.reason);
    return Status::unsupported(message.c_str(), __FILE__, __LINE__);
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
