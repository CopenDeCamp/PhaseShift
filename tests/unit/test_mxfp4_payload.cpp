#include <phaseshift/quantization/mxfp4/mxfp4.h>
#include <phaseshift/quantization/fpx/runtime_resolution.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace ps::quantization;
using namespace ps::quantization::mxfp4;
using namespace ps::quantization::fpx;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static void test_byte_counts() {
    const uint64_t rows = 7;
    for (uint64_t k : {32u, 64u, 100u}) {
        const uint64_t kp = ((k + 31u) / 32u) * 32u;
        check(quant_codes_bytes(rows, kp, kMxfp4FormatDesc) == rows * (kp / 32u) * 16u,
              "mxfp4 codes bytes");
        check(quant_metadata1_bytes(rows, kp, kMxfp4FormatDesc) == rows * (kp / 32u),
              "mxfp4 scale bytes");
        check(quant_payload_bytes(rows, kp, kMxfp4FormatDesc) == rows * (kp / 32u) * 17u,
              "mxfp4 payload bytes");
    }
    check(quant_payload_bytes(1, 32, kMxfp4FormatDesc) == 17, "mxfp4 17 bytes / 32 weights");
}

static void test_canonical_roundtrip() {
    const uint64_t rows = 5;
    for (uint64_t logical_k : {32u, 64u, 100u}) {
        const uint64_t padded_k = ((logical_k + 31u) / 32u) * 32u;
        std::vector<float> src(rows * logical_k);
        uint32_t state = 99u;
        for (auto& f : src) {
            state = state * 1664525u + 1013904223u;
            f = (static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f * 2.0f - 1.0f) * 7.0f;
        }

        Mxfp4CanonicalStore store;
        store.init(rows, logical_k, padded_k);
        for (uint64_t r = 0; r < rows; ++r) {
            check(store.quantize_row(static_cast<uint32_t>(r),
                                     std::span<const float>(src.data() + r * logical_k, logical_k)),
                  "mxfp4 quantize_row");
        }
        const Mxfp4CanonicalView v = store.view();
        check(v.validate(), "mxfp4 canonical validate");
        check(store.codes.size() == rows * (padded_k / 32u) * 16u, "mxfp4 canonical codes size");
        check(store.scales.size() == rows * (padded_k / 32u), "mxfp4 canonical scales size");

        std::vector<float> row(padded_k);
        for (uint64_t r = 0; r < rows; ++r) {
            dequantize_mxfp4_canonical_row(v, static_cast<uint32_t>(r), std::span<float>(row));
            for (uint64_t ib = 0; ib < padded_k / 32u; ++ib) {
                const float scale = e8m0_decode_u8(store.scales[r * (padded_k / 32u) + ib]);
                for (uint64_t i = ib * 32u; i < ib * 32u + 32u; ++i) {
                    const float want = (i < logical_k) ? src[r * logical_k + i] : 0.0f;
                    check(std::fabs(row[i] - want) <= scale + 1e-6f, "mxfp4 canonical deq");
                }
            }
        }
    }
}

static QuantizedTensorMetadata mxfp4_tensor() {
    QuantizedTensorMetadata t;
    t.encoding = QuantizedEncoding::Mxfp4;
    t.layout = kMxfp4LayoutName;
    t.logical_shape = {8, 96};
    t.k_padded = 96;
    t.codes = QuantizedTensorRef{"w.__phaseshift_codes", "U8", "0x00000000"};
    t.metadata1 = QuantizedTensorRef{"w.__phaseshift_metadata1", "U8", "0x00000000"};
    return t;
}

static void test_manifest_contract() {
    QuantizedManifest m;
    m.tensors["w"] = mxfp4_tensor();
    check(validate_quantized_manifest_contract(m).ok(), "mxfp4 manifest contract");
    auto resolved = resolve_quant_spec(m, m.tensors["w"]);
    check(resolved.ok(), "mxfp4 resolve");
    if (resolved.ok()) {
        check(resolved.value().spec_id == QuantSpecId::MXFP4_E2M1_S32_E8M0_V1, "mxfp4 spec id");
        check(resolved.value().descriptor.group_size == 32 &&
                  resolved.value().descriptor.block_size == 32, "mxfp4 group size");
        check(resolved.value().descriptor.scale_type == StorageDType::U8, "mxfp4 scale dtype");
    }
    check(preferred_gemm_compute_spec(QuantSpecId::MXFP4_E2M1_S32_E8M0_V1).value() ==
              ComputeSpecId::MXFP4_W4A8_F32_BF16, "mxfp4 compute spec");

    QuantizedManifest bad_dtype = m;
    bad_dtype.tensors["w"].metadata1->dtype = "BF16";
    check(!validate_quantized_manifest_contract(bad_dtype).ok(), "mxfp4 manifest rejects scale dtype");

    QuantizedManifest bad_k = m;
    bad_k.tensors["w"].k_padded = 100;
    check(!validate_quantized_manifest_contract(bad_k).ok(), "mxfp4 manifest rejects k_padded");

    QuantizedManifest bad_layout = m;
    bad_layout.tensors["w"].layout = "wrong";
    check(!validate_quantized_manifest_contract(bad_layout).ok(), "mxfp4 manifest rejects layout");
}

static void test_encoding_names() {
    check(quantized_encoding_name(QuantizedEncoding::Mxfp4) == kMxfp4EncodingName,
          "mxfp4 encoding name");
    auto parsed = parse_quantized_encoding(kMxfp4EncodingName);
    check(parsed.ok() && parsed.value() == QuantizedEncoding::Mxfp4, "mxfp4 parse");
}

int main() {
    test_byte_counts();
    test_canonical_roundtrip();
    test_manifest_contract();
    test_encoding_names();

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
