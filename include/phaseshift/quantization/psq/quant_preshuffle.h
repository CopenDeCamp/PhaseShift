#pragma once

#include <phaseshift/quantization/psq/quant_canonical.h>
#include <cstdint>
#include <vector>

namespace ps::quantization::psq {

// GPU-native (host) SoA payload: the result of a backend preshuffle of a
// canonical payload. The preshuffle is a pure permutation (plus 16-row tile
// padding); it never duplicates data. The backend kernel reads this layout.
struct NativeQuantHost {
    QuantFormatId format_id = QuantFormatId::None;
    const QuantFormatDesc* desc = nullptr;
    uint64_t rows = 0;
    uint64_t rows_padded = 0;
    uint64_t padded_k = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> metadata1;
    std::vector<uint8_t> metadata2;
    std::vector<uint8_t> metadata3;
    std::vector<uint8_t> metadata4;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t metadata1_stride_bytes = 0;
    uint32_t metadata2_stride_bytes = 0;
    uint32_t metadata3_stride_bytes = 0;
    uint32_t metadata4_stride_bytes = 0;
};

// Mandatory backend preshuffle: canonical SoA -> native SoA (permutation).
// Supports the PSQ4 and PSQ8 layouts consumed by the gfx1201 WMMA kernels.
//
// Input contract: the CanonicalQuantView passed here is the whole logical
// tensor owned by the caller. Under tensor parallelism that logical tensor is
// the rank-local canonical tensor, i.e. callers must run
//     global canonical -> logical partition -> local CanonicalQuantView
// before invoking this function. The preshuffle never sees tp_rank / tp_size
// and never partitions its input.
bool preshuffle_native(const CanonicalQuantView& v, NativeQuantHost& out);

// Test-only inverse of the preshuffle (native -> canonical) for round-trip
// verification. Not part of the production load path.
bool inverse_preshuffle(const NativeQuantHost& n, CanonicalQuantStore& out);

}  // namespace ps::quantization::psq
