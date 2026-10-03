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

    const DecodeBackend default_backend{};
    check(default_backend == DecodeBackend::Host, "default backend is Host");

    check(parse_is("host", DecodeBackend::Host), "host parses to Host");
    check(parse_is("HOST", DecodeBackend::Host), "HOST parses to Host");

    const char* legacy_tokens[] = {"gpu_mcu", "localhost", "host-auto"};
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

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
