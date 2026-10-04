#pragma once

#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <cstdint>
#include <vector>

namespace ps::quantization::mxfp4 {

// GPU-native (WMMA fragment-order) payload for an MXFP4 weight matrix. The
// E2M1 nibbles are tiled in 16-output x 32-K chunks; within a chunk each output
// owns 16 bytes where byte b holds code[k=b] in the low nibble and
// code[k=16+b] in the high nibble (the PSQ4 cb10 layout, so the low/high
// nibbles are one per byte and the E2M1->E4M3 perm needs no byte spread). The
// E8M0 scales stay row-major [N][Kp/32]. Padded output rows are zero-filled.
struct Mxfp4NativeView {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t nb = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_row_stride_bytes = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const uint8_t* scales = nullptr;
    uint64_t scales_bytes = 0;
};

struct Mxfp4NativeHost {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t nb = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_row_stride_bytes = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;

    Mxfp4NativeView view() const;
};

// Input contract: the canonical view is the whole logical tensor owned by the
// caller. Under tensor parallelism that tensor is already rank-local; callers
// partition the global canonical payload before calling this function, and
// this function never sees tp_rank / tp_size.
bool preshuffle_mxfp4_native(const Mxfp4CanonicalView& v, Mxfp4NativeHost& out);

}  // namespace ps::quantization::mxfp4
