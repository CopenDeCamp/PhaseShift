#include <phaseshift/models/qwen35/runtime/decode_backend.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

namespace q35 = ::ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

q35::DecodeBackendInputs requested_gpu_mcu() {
    q35::DecodeBackendInputs in{};
    in.requested = q35::DecodeBackend::GpuMcu;
    return in;
}

bool parse_is(const char* text, q35::DecodeBackend expected) {
    q35::DecodeBackend parsed = q35::DecodeBackend::Host;
    if (!q35::parse_decode_backend(text, &parsed)) {
        return false;
    }
    return parsed == expected;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    check(std::strcmp(q35::decode_backend_name(q35::DecodeBackend::Host), "Host") == 0,
          "Host name is Host");
    check(std::strcmp(q35::decode_backend_name(q35::DecodeBackend::GpuMcu), "GpuMcu") == 0,
          "GpuMcu name is GpuMcu");

    const q35::DecodeBackend default_backend{};
    check(default_backend == q35::DecodeBackend::Host, "default backend is Host");

    check(parse_is("host", q35::DecodeBackend::Host), "host parses to Host");
    check(parse_is("gpu-mcu", q35::DecodeBackend::GpuMcu), "gpu-mcu parses to GpuMcu");

    const char* legacy_tokens[] = {"gpu-mcu-auto", "gpu-mcu-required",
                                   "GpuMcuAuto", "GpuMcuRequired", "gpu_mcu"};
    for (const char* token : legacy_tokens) {
        q35::DecodeBackend parsed = q35::DecodeBackend::Host;
        const bool ok = q35::parse_decode_backend(token, &parsed);
        const std::string label =
            std::string(token) + " is rejected as an invalid value";
        check(!ok && parsed == q35::DecodeBackend::Host, label.c_str());
    }

    {
        const q35::DecodeBackendDecision d =
            q35::decide_decode_backend(q35::DecodeBackendInputs{});
        check(d.backend == q35::DecodeBackend::Host, "Host request keeps Host");
        check(d.reason == q35::DecodeBackendReason::BackendIsHost,
              "Host request reason");
        check(!d.eligible, "Host request is not mcu eligible");
        check(q35::decode_backend_execution_error(d).ok(),
              "Host request is executable");
    }

    {
        const q35::DecodeBackendDecision d =
            q35::decide_decode_backend(requested_gpu_mcu());
        check(d.backend == q35::DecodeBackend::GpuMcu,
              "GpuMcu request is not rewritten");
        check(d.reason == q35::DecodeBackendReason::Selected,
              "GpuMcu request reason is selected");
        check(d.eligible, "GpuMcu request is eligible");
        check(q35::decode_backend_execution_error(d).ok(),
              "eligible GpuMcu request is executable");
    }

    struct Rejection {
        q35::DecodeBackendReason reason;
        void (*mutate)(q35::DecodeBackendInputs&);
    };
    const Rejection rejections[] = {
        {q35::DecodeBackendReason::NotDecode,
         [](q35::DecodeBackendInputs& in) { in.role_is_decode = false; }},
        {q35::DecodeBackendReason::MultipleRequests,
         [](q35::DecodeBackendInputs& in) { in.num_requests = 2u; }},
        {q35::DecodeBackendReason::NotSingleToken,
         [](q35::DecodeBackendInputs& in) { in.num_tokens = 4u; }},
        {q35::DecodeBackendReason::NotSingleRow,
         [](q35::DecodeBackendInputs& in) { in.actual_rows = 8u; }},
        {q35::DecodeBackendReason::SpeculativeVerify,
         [](q35::DecodeBackendInputs& in) { in.speculative_verify = true; }},
        {q35::DecodeBackendReason::PrefillPresent,
         [](q35::DecodeBackendInputs& in) { in.prefill_present = true; }},
        {q35::DecodeBackendReason::KvDtypeNotBf16,
         [](q35::DecodeBackendInputs& in) { in.kv_dtype_is_bf16 = false; }},
        {q35::DecodeBackendReason::ImatrixCollector,
         [](q35::DecodeBackendInputs& in) {
             in.imatrix_collector_present = true;
         }},
        {q35::DecodeBackendReason::ValueTrace,
         [](q35::DecodeBackendInputs& in) { in.value_trace_present = true; }},
        {q35::DecodeBackendReason::TargetHiddenTaps,
         [](q35::DecodeBackendInputs& in) { in.target_hidden_tap_count = 2u; }},
        {q35::DecodeBackendReason::StreamWaitUnsupported,
         [](q35::DecodeBackendInputs& in) {
             in.stream_wait_value_supported = false;
         }},
        {q35::DecodeBackendReason::McuBodyRangeUnavailable,
         [](q35::DecodeBackendInputs& in) { in.mcu_body_range_valid = false; }},
    };

    for (const Rejection& r : rejections) {
        q35::DecodeBackendInputs in = requested_gpu_mcu();
        r.mutate(in);
        const q35::DecodeBackendDecision d = q35::decide_decode_backend(in);
        check(d.backend == q35::DecodeBackend::GpuMcu,
              "rejected request keeps GpuMcu instead of falling back to Host");
        check(d.reason == r.reason, "rejected request reports the reason");
        check(!d.eligible, "rejected request is not eligible");

        const ::ps::Status err = q35::decode_backend_execution_error(d);
        std::string label = q35::decode_backend_reason_name(r.reason);
        label += " returns an error instead of a Host execution";
        check(!err.ok() && err.code() == ::ps::Status::Code::unsupported,
              label.c_str());
        label = q35::decode_backend_reason_name(r.reason);
        label += " reports the reason in its message";
        check(err.message().find(
                  "GPU-MCU backend cannot execute this request: ") !=
                  std::string::npos,
              label.c_str());
        label = q35::decode_backend_reason_name(r.reason);
        label += " message names the reason";
        check(err.message().find(q35::decode_backend_reason_name(r.reason)) !=
                  std::string::npos,
              label.c_str());
    }

    {
        q35::DecodeBackendInputs in = requested_gpu_mcu();
        in.persistent_ready = true;
        in.num_requests = 2u;
        in.num_tokens = 17u;
        in.actual_rows = 17u;
        in.prefill_present = true;
        const q35::DecodeBackendDecision d = q35::decide_decode_backend(in);
        check(d.backend == q35::DecodeBackend::GpuMcu,
              "persistent ready selects the mcu body for a mixed batch");
        check(d.reason == q35::DecodeBackendReason::Selected,
              "persistent ready mixed batch reason is selected");
        check(d.eligible, "persistent ready mixed batch is eligible");
        check(q35::decode_backend_execution_error(d).ok(),
              "persistent ready mixed batch is executable");
    }

    {
        q35::DecodeBackendInputs in = requested_gpu_mcu();
        in.persistent_ready = true;
        in.speculative_verify = true;
        const q35::DecodeBackendDecision d = q35::decide_decode_backend(in);
        check(d.backend == q35::DecodeBackend::GpuMcu,
              "persistent ready accepts speculative verify");
        check(d.reason == q35::DecodeBackendReason::Selected,
              "persistent ready verify reason is selected");
        check(d.eligible, "persistent ready verify is eligible");
    }

    {
        q35::DecodeBackendInputs in = requested_gpu_mcu();
        in.persistent_ready = true;
        in.role_is_decode = false;
        const q35::DecodeBackendDecision d = q35::decide_decode_backend(in);
        check(d.backend == q35::DecodeBackend::GpuMcu,
              "persistent ready accepts a non decode role");
        check(d.eligible, "persistent ready non decode role is eligible");
    }

    {
        q35::DecodeBackendInputs in = requested_gpu_mcu();
        in.prefill_present = true;
        const q35::DecodeBackendDecision d = q35::decide_decode_backend(in);
        check(d.backend == q35::DecodeBackend::GpuMcu,
              "a rejected prefill request keeps GpuMcu");
        check(!d.eligible, "a rejected prefill request is not eligible");
        const ::ps::Status err = q35::decode_backend_execution_error(d);
        check(!err.ok() && err.code() == ::ps::Status::Code::unsupported,
              "a rejected prefill request returns an error");
    }

    std::printf("decode backend policy: cases=%zu\n",
                sizeof(rejections) / sizeof(rejections[0]) + 8u);
    if (g_failed != 0) {
        std::printf("test_gpu_mcu_decode_backend_policy: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_decode_backend_policy: PASS\n");
    return 0;
}
