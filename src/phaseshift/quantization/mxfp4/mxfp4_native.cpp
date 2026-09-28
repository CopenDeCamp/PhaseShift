#include <phaseshift/quantization/mxfp4/mxfp4_native.h>
#include <cstring>

namespace ps::quantization::mxfp4 {

Mxfp4NativeView Mxfp4NativeHost::view() const {
    Mxfp4NativeView v;
    v.rows = rows;
    v.n = n;
    v.padded_k = padded_k;
    v.nb = nb;
    v.codes_row_stride_bytes = codes_row_stride_bytes;
    v.scale_row_stride_bytes = scale_row_stride_bytes;
    v.codes = codes.data();
    v.codes_bytes = codes.size();
    v.scales = scales.data();
    v.scales_bytes = scales.size();
    return v;
}

bool preshuffle_mxfp4_native(const Mxfp4CanonicalView& v, Mxfp4NativeHost& out) {
    if (!v.validate()) return false;
    const uint64_t n = v.rows;
    const uint64_t kp = v.padded_k;
    if ((kp % kBlockK) != 0u) return false;
    const uint64_t nb = kp / kBlockK;
    const uint64_t tiles = (n + 15u) / 16u;

    out.n = n;
    out.padded_k = kp;
    out.nb = nb;
    out.rows = tiles * 16u;
    out.codes_row_stride_bytes = nb * 256u;
    out.scale_row_stride_bytes = nb;
    out.codes.assign(tiles * nb * 256u, 0u);
    out.scales.assign(n * nb, 0u);
    std::memcpy(out.scales.data(), v.scales, out.scales.size());

    for (uint64_t t = 0; t < tiles; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t ol = 0; ol < 16u; ++ol) {
                const uint64_t o = t * 16u + ol;
                if (o >= n) continue;
                const uint8_t* row = v.codes + o * (kp / 2u) + ib * kCodesBytesPerBlock;
                uint8_t* dst = out.codes.data() + (t * nb + ib) * 256u + ol * 16u;
                for (uint64_t b = 0; b < 16u; ++b) {
                    const uint64_t lo_k = b;
                    const uint64_t hi_k = 16u + b;
                    const uint8_t lo = static_cast<uint8_t>(
                        (row[lo_k >> 1u] >> (4u * (lo_k & 1u))) & 0x0Fu);
                    const uint8_t hi = static_cast<uint8_t>(
                        (row[hi_k >> 1u] >> (4u * (hi_k & 1u))) & 0x0Fu);
                    dst[b] = static_cast<uint8_t>(lo | (hi << 4u));
                }
            }
        }
    }
    return true;
}

}  // namespace ps::quantization::mxfp4
