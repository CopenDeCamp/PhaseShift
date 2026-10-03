#pragma once

#include <phaseshift/quantization/fp8/block128.h>
#include <cstdint>
#include <vector>

namespace ps::quantization::fp8 {

// GPU-native (WMMA fragment-order) payload for a single FP8 E4M3 block128
// weight matrix. Codes are tiled in the exact 16-output x 32-K chunk order the
// gfx1201 FP8 WMMA consumer reads; the FP32 scales stay row-major
// [ceil(N/128)][Kp/128]. Padded output rows are zero-filled so the guarded
// consumer can load full 16-row tiles.
struct Fp8Block128NativeView {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_n = 0;
    uint64_t scale_k = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const float* scales = nullptr;
    uint64_t scales_bytes = 0;
};

struct Fp8Block128NativeHost {
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_n = 0;
    uint64_t scale_k = 0;
    std::vector<uint8_t> codes;
    std::vector<float> scales;

    Fp8Block128NativeView view() const;
};

// Batched canonical matrices are not supported: 128x128 scale tiles must not
// straddle a matrix boundary, and the consumer addresses one matrix at a time.
// Input contract: the canonical view is the whole logical tensor owned by the
// caller; this function never partitions and never sees tp_rank / tp_size.
bool preshuffle_fp8_block128_native(const Fp8BlockCanonicalView& v,
                                    Fp8Block128NativeHost& out);

}  // namespace ps::quantization::fp8
