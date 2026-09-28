#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace ps::quantization::mxfp4 {

namespace {

int ceil_log2_positive(double x) {
    int e = 0;
    const double m = std::frexp(x, &e);
    if (m == 0.5) return e - 1;
    return e;
}

uint8_t encode_scale_byte(float amax) {
    if (!(amax > 0.0f) || !std::isfinite(amax)) return 0;
    const float raw = amax / kE2M1Magnitudes[7];
    int exponent = ceil_log2_positive(static_cast<double>(raw));
    if (exponent < -127) exponent = -127;
    if (exponent > 127) exponent = 127;
    return static_cast<uint8_t>(exponent + 127);
}

void quantize_block(const float* x, uint8_t* codes, uint8_t* scale_byte) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < kBlockK; ++i) {
        const float a = std::fabs(x[i]);
        if (std::isfinite(a) && a > amax) amax = a;
    }
    if (!(amax > 0.0f)) {
        std::memset(codes, 0, kCodesBytesPerBlock);
        *scale_byte = 0;
        return;
    }
    *scale_byte = encode_scale_byte(amax);
    const int exponent = static_cast<int>(*scale_byte) - 127;
    const float scale = std::ldexp(1.0f, exponent);
    for (uint32_t j = 0; j < kCodesBytesPerBlock; ++j) {
        const uint8_t low = e2m1_encode_nibble(x[2u * j] / scale);
        const uint8_t high = e2m1_encode_nibble(x[2u * j + 1u] / scale);
        codes[j] = static_cast<uint8_t>(low | (high << 4u));
    }
}

void dequantize_block(const uint8_t* codes, uint8_t scale_byte, float* dst) {
    const float s = e8m0_decode_u8(scale_byte);
    for (uint32_t j = 0; j < kCodesBytesPerBlock; ++j) {
        const uint8_t byte = codes[j];
        dst[2u * j] = e2m1_decode_nibble(byte & 0x0Fu) * s;
        dst[2u * j + 1u] = e2m1_decode_nibble(byte >> 4u) * s;
    }
}

void load_block(std::span<const float> src, uint64_t k, uint64_t ib, float* x) {
    for (uint32_t i = 0; i < kBlockK; ++i) {
        const uint64_t idx = ib * kBlockK + i;
        x[i] = (idx < k) ? src[static_cast<std::size_t>(idx)] : 0.0f;
    }
}

const QuantFormatDesc* mxfp4_desc() {
    return quant_format_desc(QuantFormatId::Mxfp4);
}

}  // namespace

float e2m1_decode_nibble(uint8_t nibble) {
    const float mag = kE2M1Magnitudes[nibble & 0x07u];
    return (nibble & 0x08u) ? -mag : mag;
}

uint8_t e2m1_encode_nibble(float x) {
    if (!std::isfinite(x)) return 0;
    const uint8_t sign = std::signbit(x) ? 0x08u : 0x00u;
    const float a = std::fabs(x);
    if (!(a > 0.0f)) return sign;
    if (a >= kE2M1Magnitudes[7]) return static_cast<uint8_t>(sign | 0x07u);
    int lo = 0;
    for (int i = 0; i < 8; ++i) {
        if (kE2M1Magnitudes[i] <= a) lo = i;
        else break;
    }
    if (lo >= 7) return static_cast<uint8_t>(sign | 0x07u);
    const float vlo = kE2M1Magnitudes[lo];
    const float vhi = kE2M1Magnitudes[lo + 1];
    const float dlo = a - vlo;
    const float dhi = vhi - a;
    int pick;
    if (dhi < dlo) pick = lo + 1;
    else if (dlo < dhi) pick = lo;
    else pick = (((lo + 1) % 2) == 0) ? lo + 1 : lo;
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(pick));
}

float e8m0_decode_u8(uint8_t byte) {
    if (byte == 0xFFu) return std::nanf("");
    return std::ldexp(1.0f, static_cast<int>(byte) - 127);
}

uint8_t e8m0_encode_f32(float x) {
    if (!(x > 0.0f) || std::isnan(x)) return 0;
    if (std::isinf(x)) return 254;
    int exponent = ceil_log2_positive(static_cast<double>(x));
    if (exponent < -127) exponent = -127;
    if (exponent > 127) exponent = 127;
    return static_cast<uint8_t>(exponent + 127);
}

void quantize_mxfp4_row(
    std::span<const float> src,
    std::span<uint8_t> codes,
    std::span<uint8_t> scales,
    uint64_t logical_k,
    uint64_t padded_k) {
    const uint64_t nb = padded_k / kBlockK;
    for (uint64_t ib = 0; ib < nb; ++ib) {
        float x[kBlockK];
        load_block(src, logical_k, ib, x);
        quantize_block(x, codes.data() + ib * kCodesBytesPerBlock, scales.data() + ib);
    }
}

void dequantize_mxfp4_row(
    std::span<const uint8_t> codes,
    std::span<const uint8_t> scales,
    std::span<float> dst,
    uint64_t logical_k,
    uint64_t padded_k) {
    (void)logical_k;
    const uint64_t nb = padded_k / kBlockK;
    for (uint64_t ib = 0; ib < nb; ++ib) {
        dequantize_block(codes.data() + ib * kCodesBytesPerBlock, scales[static_cast<std::size_t>(ib)],
                         dst.data() + ib * kBlockK);
    }
}

bool Mxfp4CanonicalView::validate() const {
    const QuantFormatDesc* desc = mxfp4_desc();
    if (desc == nullptr) return false;
    if (padded_k == 0 || padded_k % kBlockK != 0) return false;
    if (codes_bytes != quant_codes_bytes(rows, padded_k, *desc)) return false;
    if (scales_bytes != quant_metadata1_bytes(rows, padded_k, *desc)) return false;
    return true;
}

void Mxfp4CanonicalStore::init(uint64_t rows_in, uint64_t logical_k_in, uint64_t padded_k_in) {
    rows = rows_in;
    logical_k = logical_k_in;
    padded_k = padded_k_in;
    const QuantFormatDesc* desc = mxfp4_desc();
    if (desc == nullptr) return;
    codes.resize(quant_codes_bytes(rows, padded_k, *desc));
    scales.resize(quant_metadata1_bytes(rows, padded_k, *desc));
}

Mxfp4CanonicalView Mxfp4CanonicalStore::view() const {
    Mxfp4CanonicalView v;
    v.rows = rows;
    v.padded_k = padded_k;
    v.codes = codes.data();
    v.codes_bytes = codes.size();
    v.scales = scales.data();
    v.scales_bytes = scales.size();
    return v;
}

bool Mxfp4CanonicalStore::quantize_row(uint32_t row, std::span<const float> src_row) {
    const uint64_t nb = padded_k / kBlockK;
    if (static_cast<uint64_t>(row) >= rows) return false;
    if (src_row.size() < logical_k) return false;
    if (codes.size() != rows * nb * kCodesBytesPerBlock) return false;
    if (scales.size() != rows * nb) return false;
    quantize_mxfp4_row(
        src_row,
        std::span<uint8_t>(codes.data() + static_cast<uint64_t>(row) * nb * kCodesBytesPerBlock,
                           nb * kCodesBytesPerBlock),
        std::span<uint8_t>(scales.data() + static_cast<uint64_t>(row) * nb, nb),
        logical_k,
        padded_k);
    return true;
}

void dequantize_mxfp4_canonical_row(const Mxfp4CanonicalView& v, uint32_t row, std::span<float> out) {
    const uint64_t nb = v.padded_k / kBlockK;
    dequantize_mxfp4_row(
        std::span<const uint8_t>(v.codes + static_cast<uint64_t>(row) * nb * kCodesBytesPerBlock,
                                 nb * kCodesBytesPerBlock),
        std::span<const uint8_t>(v.scales + static_cast<uint64_t>(row) * nb, nb),
        out,
        0,
        v.padded_k);
}

void dequantize_mxfp4_canonical(const Mxfp4CanonicalView& v, std::vector<float>& out) {
    out.assign(v.rows * v.padded_k, 0.0f);
    std::vector<float> row(v.padded_k);
    for (uint64_t r = 0; r < v.rows; ++r) {
        dequantize_mxfp4_canonical_row(v, static_cast<uint32_t>(r), std::span<float>(row));
        std::memcpy(out.data() + r * v.padded_k, row.data(), v.padded_k * sizeof(float));
    }
}

}  // namespace ps::quantization::mxfp4
