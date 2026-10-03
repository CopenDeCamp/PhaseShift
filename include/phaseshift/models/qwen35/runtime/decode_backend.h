#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>

namespace ps {
namespace qwen35 {
namespace runtime {

enum class DecodeBackend : uint8_t {
    Host = 0,
};

const char* decode_backend_name(DecodeBackend backend);

bool parse_decode_backend(const char* text, DecodeBackend* out);

Status validate_decode_backend(DecodeBackend backend);

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
