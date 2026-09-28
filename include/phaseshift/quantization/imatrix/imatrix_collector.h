#pragma once
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::quantization::imatrix {

enum class ImatrixSite : uint8_t {
    FullAttnQkvInput = 1,
    FullAttnOInput = 2,
    GdnProjectionInput = 3,
    GdnOutputInput = 4,
    MlpGateUpInput = 5,
    MlpDownInput = 6,
    LmHeadInput = 7,
};

enum class ImatrixDType : uint8_t {
    BF16 = 0,
    F32 = 1,
};

constexpr uint32_t kImatrixLayerNone = 0xFFu;

constexpr uint32_t imatrix_tag(uint32_t layer_index, ImatrixSite site) noexcept {
    return (layer_index << 8) | static_cast<uint32_t>(site);
}

constexpr uint32_t imatrix_tag_layer(uint32_t tag) noexcept {
    return tag >> 8;
}

constexpr ImatrixSite imatrix_tag_site(uint32_t tag) noexcept {
    return static_cast<ImatrixSite>(tag & 0xFFu);
}

class ImatrixCollector {
public:
    virtual ~ImatrixCollector() = default;

    virtual Status record(uint32_t tag, const void* data, uint32_t row_stride,
                          uint32_t features, uint32_t rows, ImatrixDType dtype,
                          hipStream_t stream) = 0;
};

}
