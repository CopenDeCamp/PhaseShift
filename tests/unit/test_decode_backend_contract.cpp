#include <phaseshift/models/qwen35/runtime/decode_backend.h>
#include <phaseshift/core/status.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

bool parse_is(const char* text, ::ps::qwen35::runtime::DecodeBackend expected) {
    ::ps::qwen35::runtime::DecodeBackend parsed =
        ::ps::qwen35::runtime::DecodeBackend::Host;
    if (!::ps::qwen35::runtime::parse_decode_backend(text, &parsed)) {
        return false;
    }
    return parsed == expected;
}

}  // namespace

int main() {
    namespace backend = ::ps::qwen35::runtime;
    using DecodeBackend = backend::DecodeBackend;

    check(std::strcmp(backend::decode_backend_name(DecodeBackend::Host), "Host") == 0,
          "Host name is Host");
    check(std::strcmp(backend::decode_backend_name(DecodeBackend::GpuMcu), "GpuMcu") == 0,
          "GpuMcu name is GpuMcu");

    const DecodeBackend default_backend{};
    check(default_backend == DecodeBackend::Host, "default backend is Host");

    check(parse_is("host", DecodeBackend::Host), "host parses to Host");
    check(parse_is("HOST", DecodeBackend::Host), "HOST parses to Host");
    check(parse_is("gpu-mcu", DecodeBackend::GpuMcu), "gpu-mcu parses to GpuMcu");
    check(parse_is("GPU-MCU", DecodeBackend::GpuMcu), "GPU-MCU parses to GpuMcu");

    const char* legacy_tokens[] = {"gpu-mcu-auto", "gpu-mcu-required",
                                   "GpuMcuAuto", "GpuMcuRequired", "gpu_mcu"};
    for (const char* token : legacy_tokens) {
        DecodeBackend parsed = DecodeBackend::Host;
        const bool ok = backend::parse_decode_backend(token, &parsed);
        std::string label = std::string(token) + " is rejected as an invalid value";
        check(!ok && parsed == DecodeBackend::Host, label.c_str());
    }

    DecodeBackend parsed = DecodeBackend::Host;
    check(!backend::parse_decode_backend("cuda", &parsed), "unknown backend is rejected");
    check(!backend::parse_decode_backend("", &parsed), "empty backend is rejected");
    check(!backend::parse_decode_backend(nullptr, &parsed), "null backend is rejected");

    const ps::Status host_status = backend::validate_decode_backend(DecodeBackend::Host);
    check(host_status.ok(), "Host backend validates");

    const ps::Status mcu_status = backend::validate_decode_backend(DecodeBackend::GpuMcu);
    check(!mcu_status.ok() && mcu_status.code() == ps::Status::Code::unsupported,
          "GpuMcu is unsupported");
    check(mcu_status.message().find(backend::kGpuMcuUnavailableMessage)
              != std::string::npos,
          "GpuMcu reports the unavailable message");

    DecodeBackend requested = DecodeBackend::Host;
    const bool requested_ok = backend::parse_decode_backend("gpu-mcu", &requested);
    check(requested_ok && requested == DecodeBackend::GpuMcu,
          "a gpu-mcu request stays GpuMcu");
    check(!backend::validate_decode_backend(requested).ok(),
          "a gpu-mcu request never validates as runnable");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
