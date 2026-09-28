#include <phaseshift/quantization/mxfp4/mxfp4.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace ps::quantization::mxfp4;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static void check_near(float a, float b, const std::string& msg) {
    if (std::isnan(a) || std::isnan(b)) return;
    if (a == b) return;
    const float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    if (std::fabs(a - b) > 1e-6f * scale) fail(msg);
}

static void test_e2m1_table() {
    const float expected[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint8_t code = 0; code < 16; ++code) {
        const float want = (code & 8u) ? -expected[code & 7u] : expected[code & 7u];
        check_near(e2m1_decode_nibble(code), want, "e2m1 decode " + std::to_string(code));
        check(e2m1_encode_nibble(want) == code, "e2m1 roundtrip " + std::to_string(code));
        check(e2m1_encode_nibble(e2m1_decode_nibble(code)) == code,
              "e2m1 encode(decode) " + std::to_string(code));
    }
}

static void test_e2m1_thresholds() {
    check(e2m1_encode_nibble(0.0f) == 0x0, "e2m1 0");
    check(e2m1_encode_nibble(0.25f) == 0x0, "e2m1 tie 0.25 -> 0");
    check(e2m1_encode_nibble(0.26f) == 0x1, "e2m1 0.26 -> 0.5");
    check(e2m1_encode_nibble(0.75f) == 0x2, "e2m1 tie 0.75 -> 1.0");
    check(e2m1_encode_nibble(1.25f) == 0x2, "e2m1 tie 1.25 -> 1.0");
    check(e2m1_encode_nibble(1.75f) == 0x4, "e2m1 tie 1.75 -> 2.0");
    check(e2m1_encode_nibble(2.5f) == 0x4, "e2m1 tie 2.5 -> 2.0");
    check(e2m1_encode_nibble(3.5f) == 0x6, "e2m1 tie 3.5 -> 4.0");
    check(e2m1_encode_nibble(5.0f) == 0x6, "e2m1 tie 5.0 -> 4.0");
    check(e2m1_encode_nibble(7.0f) == 0x7, "e2m1 saturate 7 -> 6");
    check(e2m1_encode_nibble(100.0f) == 0x7, "e2m1 saturate 100 -> 6");
    check(e2m1_encode_nibble(-0.75f) == 0xA, "e2m1 negative tie");
    check(e2m1_encode_nibble(-6.0f) == 0xF, "e2m1 -6");
    check(e2m1_encode_nibble(std::nanf("")) == 0x0, "e2m1 nan -> 0");
    check(e2m1_encode_nibble(INFINITY) == 0x0, "e2m1 inf -> 0");
}

static void test_e8m0() {
    check_near(e8m0_decode_u8(127), 1.0f, "e8m0 1");
    check_near(e8m0_decode_u8(128), 2.0f, "e8m0 2");
    check_near(e8m0_decode_u8(126), 0.5f, "e8m0 0.5");
    check_near(e8m0_decode_u8(0), std::ldexp(1.0f, -127), "e8m0 min");
    check(std::isnan(e8m0_decode_u8(255)), "e8m0 nan byte");
    check(e8m0_encode_f32(1.0f) == 127, "e8m0 encode 1");
    check(e8m0_encode_f32(2.0f) == 128, "e8m0 encode 2");
    check(e8m0_encode_f32(0.5f) == 126, "e8m0 encode 0.5");
    check(e8m0_encode_f32(3.0f) == 129, "e8m0 encode 3");
    check(e8m0_encode_f32(4.0f) == 129, "e8m0 encode 4");
    check(e8m0_encode_f32(0.0f) == 0, "e8m0 encode 0");
    check(e8m0_encode_f32(-1.0f) == 0, "e8m0 encode negative");
    check(e8m0_encode_f32(INFINITY) == 254, "e8m0 encode inf");
    check(e8m0_encode_f32(1e-40f) == 0, "e8m0 encode underflow");
}

static void test_quantize_zero_block() {
    std::vector<float> src(32, 0.0f);
    std::vector<uint8_t> codes(16, 0xAA);
    std::vector<uint8_t> scales(1, 0xAA);
    quantize_mxfp4_row(std::span<const float>(src), std::span<uint8_t>(codes),
                       std::span<uint8_t>(scales), 32, 32);
    for (uint8_t c : codes) check(c == 0, "mxfp4 zero codes");
    check(scales[0] == 0, "mxfp4 zero scale");
}

static void test_quantize_packing() {
    std::vector<float> src(32);
    for (uint32_t i = 0; i < 32; ++i) src[i] = e2m1_decode_nibble(static_cast<uint8_t>(i % 16));
    std::vector<uint8_t> codes(16);
    std::vector<uint8_t> scales(1);
    quantize_mxfp4_row(std::span<const float>(src), std::span<uint8_t>(codes),
                       std::span<uint8_t>(scales), 32, 32);
    check(scales[0] == 127, "mxfp4 scale 1");
    for (uint32_t j = 0; j < 16; ++j) {
        const uint8_t low = static_cast<uint8_t>(j * 2u % 16u);
        const uint8_t high = static_cast<uint8_t>((j * 2u + 1u) % 16u);
        const uint8_t want = static_cast<uint8_t>(low | (high << 4u));
        check(codes[j] == want, "mxfp4 packing byte " + std::to_string(j));
    }
}

static void test_quantize_nonfinite() {
    std::vector<float> src(32, 0.0f);
    src[0] = std::nanf("");
    src[1] = INFINITY;
    src[2] = -INFINITY;
    src[3] = 3.0f;
    std::vector<uint8_t> codes(16);
    std::vector<uint8_t> scales(1);
    quantize_mxfp4_row(std::span<const float>(src), std::span<uint8_t>(codes),
                       std::span<uint8_t>(scales), 32, 32);
    check(scales[0] == 126, "mxfp4 nonfinite scale from 3.0");
    check(e2m1_decode_nibble(codes[0] & 0x0Fu) == 0.0f, "mxfp4 nan code");
    check(e2m1_decode_nibble(codes[0] >> 4u) == 0.0f, "mxfp4 inf code");
    check(e2m1_decode_nibble(codes[1] & 0x0Fu) == 0.0f, "mxfp4 -inf code");
    check(e2m1_decode_nibble(codes[1] >> 4u) == 6.0f, "mxfp4 3.0 code");
    check(e2m1_decode_nibble(codes[1] >> 4u) * e8m0_decode_u8(scales[0]) == 3.0f,
          "mxfp4 3.0 dequant");
}

static void test_padding() {
    const uint64_t logical_k = 40;
    const uint64_t padded_k = 64;
    std::vector<float> src(logical_k);
    for (uint64_t i = 0; i < logical_k; ++i) src[i] = (i < 8) ? 6.0f : 0.0f;
    std::vector<uint8_t> codes(padded_k / 32 * 16, 0xAA);
    std::vector<uint8_t> scales(padded_k / 32, 0xAA);
    quantize_mxfp4_row(std::span<const float>(src), std::span<uint8_t>(codes),
                       std::span<uint8_t>(scales), logical_k, padded_k);
    check(scales[0] == 127, "mxfp4 pad block0 scale");
    check(scales[1] == 0, "mxfp4 pad block1 zero scale");
    for (uint32_t j = 0; j < 16; ++j) {
        const uint8_t b = codes[16 + j];
        check(b == 0, "mxfp4 pad block1 codes zero");
    }
    std::vector<float> dst(padded_k, -1.0f);
    dequantize_mxfp4_row(std::span<const uint8_t>(codes), std::span<const uint8_t>(scales),
                         std::span<float>(dst), logical_k, padded_k);
    for (uint64_t i = 0; i < 8; ++i) check_near(dst[i], 6.0f, "mxfp4 pad deq real");
    for (uint64_t i = 8; i < padded_k; ++i) check(dst[i] == 0.0f, "mxfp4 pad deq zero");
}

static void test_random_roundtrip() {
    const uint64_t k = 96;
    std::vector<float> src(k);
    uint32_t state = 12345u;
    auto next = [&]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f * 2.0f - 1.0f;
    };
    for (uint64_t i = 0; i < k; ++i) src[i] = next() * 8.0f;
    std::vector<uint8_t> codes(k / 32 * 16);
    std::vector<uint8_t> scales(k / 32);
    quantize_mxfp4_row(std::span<const float>(src), std::span<uint8_t>(codes),
                       std::span<uint8_t>(scales), k, k);
    std::vector<float> dst(k, 0.0f);
    dequantize_mxfp4_row(std::span<const uint8_t>(codes), std::span<const uint8_t>(scales),
                         std::span<float>(dst), k, k);
    for (uint64_t ib = 0; ib < k / 32; ++ib) {
        const float scale = e8m0_decode_u8(scales[ib]);
        for (uint64_t i = ib * 32; i < ib * 32 + 32; ++i) {
            check(std::fabs(dst[i] - src[i]) <= scale + 1e-6f, "mxfp4 roundtrip error");
        }
    }
}

int main() {
    test_e2m1_table();
    test_e2m1_thresholds();
    test_e8m0();
    test_quantize_zero_block();
    test_quantize_packing();
    test_quantize_nonfinite();
    test_padding();
    test_random_roundtrip();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
