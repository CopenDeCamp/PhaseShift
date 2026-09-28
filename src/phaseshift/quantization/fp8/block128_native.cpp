#include <phaseshift/quantization/fp8/block128_native.h>
#include <cstring>

namespace ps::quantization::fp8 {

Fp8Block128NativeView Fp8Block128NativeHost::view() const {
    Fp8Block128NativeView v;
    v.rows = rows;
    v.n = n;
    v.padded_k = padded_k;
    v.codes_row_stride_bytes = codes_row_stride_bytes;
    v.scale_n = scale_n;
    v.scale_k = scale_k;
    v.codes = codes.data();
    v.codes_bytes = codes.size();
    v.scales = scales.data();
    v.scales_bytes = scales.size() * sizeof(float);
    return v;
}

bool preshuffle_fp8_block128_native(const Fp8BlockCanonicalView& v,
                                    Fp8Block128NativeHost& out) {
    if (!v.validate()) return false;
    if (v.batch != 1) return false;
    const uint64_t n = v.n;
    const uint64_t kp = v.padded_k;
    if ((kp % 32u) != 0u) return false;
    const uint64_t nb = kp / 32u;
    const uint64_t tiles = (n + 15u) / 16u;

    out.n = n;
    out.padded_k = kp;
    out.rows = tiles * 16u;
    out.codes_row_stride_bytes = nb * 512u;
    out.scale_n = v.scale_n;
    out.scale_k = v.scale_k;
    out.codes.assign(tiles * nb * 512u, 0u);
    out.scales.assign(v.scale_n * v.scale_k, 0.0f);
    std::memcpy(out.scales.data(), v.scales, out.scales.size() * sizeof(float));

    for (uint64_t t = 0; t < tiles; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            for (uint64_t half = 0; half < 2u; ++half) {
                for (uint64_t ol = 0; ol < 16u; ++ol) {
                    const uint64_t o = t * 16u + ol;
                    if (o >= n) continue;
                    std::memcpy(
                        out.codes.data() + (t * nb + ib) * 512u + half * 256u + ol * 16u,
                        v.codes + o * kp + ib * 32u + half * 16u, 16u);
                }
            }
        }
    }
    return true;
}

}  // namespace ps::quantization::fp8
