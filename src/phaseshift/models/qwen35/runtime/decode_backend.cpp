#include <phaseshift/models/qwen35/runtime/decode_backend.h>

#include <strings.h>

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

Status validate_decode_backend(DecodeBackend backend) {
    if (backend == DecodeBackend::Host) {
        return Status::make_ok();
    }
    return Status::unsupported(kGpuMcuUnavailableMessage, __FILE__, __LINE__);
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
