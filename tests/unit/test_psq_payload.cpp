#include <phaseshift/quantization/quant_format.h>
#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <phaseshift/quantization/psq/psq.h>
#include <phaseshift/quantization/fpx/e4m3.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace ps::quantization;
using namespace ps::quantization::psq;

static int g_fail = 0;

static void fail(const std::string& msg) {
    g_fail++;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check_bytes_eq(const std::string& tag, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) { fail(tag + " size " + std::to_string(a.size()) + " vs " + std::to_string(b.size())); return; }
    if (!a.empty() && std::memcmp(a.data(), b.data(), a.size()) != 0) fail(tag + " (" + std::to_string(a.size()) + " bytes)");
}

static void check_float_row(const std::string& tag, const float* a, const float* b, uint64_t n) {
    int bad = 0;
    for (uint64_t i = 0; i < n; ++i) {
        const float x = a[i], y = b[i];
        if (x == 0.0f && y == 0.0f) continue;
        if (std::memcmp(&x, &y, 4) != 0) bad++;
    }
    if (bad > 0) fail(tag + " (" + std::to_string(bad) + "/" + std::to_string(n) + ")");
}

static void test_size_single_source() {
    const uint64_t rows = 100, k = 832;
    const uint64_t kp = (k + 31) / 32 * 32;
    const QuantFormatDesc* d4 = &kPsq4FormatDesc;
    const QuantFormatDesc* d8 = &kPsq8FormatDesc;
    const uint64_t nb = kp / 32;

    if (quant_payload_bytes(rows, kp, *d4) != rows * nb * 18) fail("size psq4 density");
    if (quant_payload_bytes(rows, kp, *d8) != rows * nb * 34) fail("size psq8 density");

    if (quant_payload_bytes(1, 32, *d4) != 18) fail("size psq4 18/32");
    if (quant_payload_bytes(1, 32, *d8) != 34) fail("size psq8 34/32");
}

static void test_canonical_dequant(const QuantFormatId id, const std::string& name) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    const uint64_t rows = 7;
    for (uint64_t logical_k : {32u, 64u, 100u, 400u}) {
        const uint64_t padded_k = ((logical_k + 31) / 32) * 32;
        std::vector<float> src(rows * logical_k);
        for (auto& f : src) f = dist(rng);

        CanonicalQuantStore store;
        store.init(id, rows, logical_k, padded_k);
        for (uint64_t r = 0; r < rows; ++r)
            store.quantize_row((uint32_t)r, std::span<const float>(src.data() + r * logical_k, logical_k));
        if (!store.view().validate()) { fail(name + " k=" + std::to_string(logical_k) + " validate"); continue; }
        std::vector<float> can;
        dequantize_canonical(store.view(), can);

        std::vector<float> r2(padded_k);
        for (uint64_t r = 0; r < rows; ++r) {
            std::span<const float> sr(src.data() + r * logical_k, logical_k);
            std::vector<uint8_t> c2, s2;
            if (id == QuantFormatId::Psq4) {
                c2.resize(padded_k / 2); s2.resize(padded_k / 16);
                quantize_psq4_row(sr, c2, s2, logical_k, padded_k);
                dequantize_psq4_row(c2, s2, r2, logical_k, padded_k);
            } else {
                c2.resize(padded_k); s2.resize(padded_k / 16);
                quantize_psq8_row(sr, c2, s2, logical_k, padded_k);
                dequantize_psq8_row(c2, s2, r2, logical_k, padded_k);
            }
            check_float_row(name + " k=" + std::to_string(logical_k) + " r=" + std::to_string(r),
                           can.data() + r * padded_k, r2.data(), padded_k);
        }
    }
}

static void test_preshuffle_roundtrip(const QuantFormatId id, const std::string& name) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-12.0f, 12.0f);
    const uint64_t rows = 7;
    for (uint64_t logical_k : {32u, 64u, 100u, 400u}) {
        const uint64_t padded_k = ((logical_k + 31) / 32) * 32;
        std::vector<float> src(rows * logical_k);
        for (auto& f : src) f = dist(rng);

        CanonicalQuantStore store;
        store.init(id, rows, logical_k, padded_k);
        for (uint64_t r = 0; r < rows; ++r)
            store.quantize_row((uint32_t)r, std::span<const float>(src.data() + r * logical_k, logical_k));
        const CanonicalQuantView v = store.view();

        NativeQuantHost native;
        if (!preshuffle_native(v, native)) { fail(name + " k=" + std::to_string(logical_k) + " preshuffle"); continue; }

        CanonicalQuantStore restored;
        if (!inverse_preshuffle(native, restored)) { fail(name + " k=" + std::to_string(logical_k) + " inverse"); continue; }

        const std::string tag = name + " k=" + std::to_string(logical_k);
        check_bytes_eq(tag + " codes", store.codes, restored.codes);
        check_bytes_eq(tag + " metadata1", store.metadata1, restored.metadata1);
        check_bytes_eq(tag + " metadata2", store.metadata2, restored.metadata2);
        check_bytes_eq(tag + " metadata3", store.metadata3, restored.metadata3);
        check_bytes_eq(tag + " metadata4", store.metadata4, restored.metadata4);

        const uint64_t native_bytes = native.codes.size() + native.metadata1.size() + native.metadata2.size() + native.metadata3.size() + native.metadata4.size();
        const uint64_t canon_bytes = store.codes.size() + store.metadata1.size() + store.metadata2.size() + store.metadata3.size() + store.metadata4.size();
        const uint64_t allowed = canon_bytes * native.rows_padded / rows;
        if (native_bytes > allowed) fail(tag + " VRAM " + std::to_string(native_bytes) + " > " + std::to_string(allowed));
    }
}

// Bank/sector mapping: a single WMMA fragment load (16 output rows x
// code_block_bytes) reads `block_bytes` of the native codes layout. Verify the
// 32-byte sectors it touches are 32-byte aligned, inside the block, and map to
// distinct consecutive banks (one sector per bank, no bank conflicts).
static void test_bank_mapping(const QuantFormatId id, const std::string& name) {
    const uint32_t code_block_bytes = (id == QuantFormatId::Psq4) ? 16u : 32u;
    const uint64_t block_bytes = 16u * code_block_bytes;
    const uint64_t nb = 8;
    for (uint64_t t = 0; t < 4; ++t) {
        for (uint64_t ib = 0; ib < nb; ++ib) {
            const uint64_t block_base = (t * nb + ib) * block_bytes;
            bool aligned = true, inside = true, conflict = false;
            uint32_t banks[16] = {0};
            uint64_t nsectors = block_bytes / 32u;
            for (uint64_t s = 0; s < nsectors; ++s) {
                const uint64_t addr = block_base + s * 32u;
                if ((addr % 32u) != 0) aligned = false;
                if (addr < block_base || addr + 32u > block_base + block_bytes) inside = false;
                const uint32_t bank = static_cast<uint32_t>((addr / 32u) % 32u);
                for (uint64_t p = 0; p < s; ++p)
                    if (banks[p] == bank) conflict = true;
                banks[s] = bank;
            }
            if (!aligned) fail(name + " sector not 32B aligned");
            if (!inside) fail(name + " sector out of block");
            if (conflict) fail(name + " bank conflict");
        }
    }
}

// Native layout contract for the gfx1201 consumers. The PSQ4 WMMA kernels read
// each (tile, block) 256B chunk as [16 rows * 16B]; the PSQ8 kernels read each
// 512B chunk as [half][16 rows * 16B] (offset (k>>4)*256 + ol*16 + (k&15)
// within the chunk). Pin these exact byte positions so producer/consumer layout
// drift fails here instead of in E2E.
static void test_native_layout_contract(const QuantFormatId id, const std::string& name) {
    const bool psq4 = (id == QuantFormatId::Psq4);
    const uint32_t code_block_bytes = psq4 ? 16u : 32u;
    const uint32_t chunk_bytes = 16u * code_block_bytes;
    const uint64_t rows = 7;
    for (uint64_t logical_k : {32u, 64u, 100u}) {
        const uint64_t padded_k = ((logical_k + 31) / 32) * 32;
        const uint64_t nb = padded_k / 32;
        CanonicalQuantStore store;
        store.init(id, rows, logical_k, padded_k);
        store.codes.resize(rows * nb * code_block_bytes);
        store.metadata1.resize(rows * nb * 2);
        const uint64_t crow = store.codes.size() / rows;
        for (uint64_t o = 0; o < rows; ++o)
            for (uint64_t i = 0; i < crow; ++i)
                store.codes[o * crow + i] = static_cast<uint8_t>((o * 37u + i * 11u) & 0xFFu);
        const uint64_t mrow = store.metadata1.size() / rows;
        for (uint64_t o = 0; o < rows; ++o)
            for (uint64_t i = 0; i < mrow; ++i)
                store.metadata1[o * mrow + i] = static_cast<uint8_t>((o * 53u + i * 7u) & 0xFFu);
        const CanonicalQuantView v = store.view();
        if (!v.validate()) { fail(name + " k=" + std::to_string(logical_k) + " validate"); continue; }
        NativeQuantHost native;
        if (!preshuffle_native(v, native)) {
            fail(name + " k=" + std::to_string(logical_k) + " preshuffle");
            continue;
        }
        const std::string tag = name + " k=" + std::to_string(logical_k);
        int bad = 0;
        const auto report = [&]() {
            bad++;
            if (bad <= 4) fail(tag + " native layout");
        };
        auto native_pos = [&](uint64_t t, uint64_t ol, uint64_t ib, uint64_t j) {
            return (t * nb + ib) * chunk_bytes +
                   (psq4 ? ol * 16u + j
                         : (j >> 4) * 256u + ol * 16u + (j & 15u));
        };
        for (uint64_t o = 0; o < rows; ++o) {
            const uint64_t t = o / 16u, ol = o % 16u;
            for (uint64_t ib = 0; ib < nb; ++ib) {
                for (uint64_t j = 0; j < code_block_bytes; ++j)
                    if (native.codes[native_pos(t, ol, ib, j)] !=
                        store.codes[o * nb * code_block_bytes + ib * code_block_bytes + j])
                        report();
                for (uint64_t h = 0; h < 2u; ++h)
                    if (native.metadata1[(t * nb + ib) * 32u + ol * 2u + h] !=
                        store.metadata1[o * nb * 2u + ib * 2u + h])
                        report();
            }
        }
        for (uint64_t ol = rows; ol < 16u; ++ol)
            for (uint64_t ib = 0; ib < nb; ++ib)
                for (uint64_t j = 0; j < code_block_bytes; ++j)
                    if (native.codes[native_pos(0, ol, ib, j)] != 0u)
                        report();
    }
}

static void test_format_ids() {
    if (static_cast<uint8_t>(QuantFormatId::None) != 0) fail("QuantFormatId::None != 0");
    if (static_cast<uint8_t>(QuantFormatId::Bf16) != 1) fail("QuantFormatId::Bf16 != 1");
    if (static_cast<uint8_t>(QuantFormatId::Psq4) != 2) fail("QuantFormatId::Psq4 != 2");
    if (static_cast<uint8_t>(QuantFormatId::Psq8) != 3) fail("QuantFormatId::Psq8 != 3");
}

static void test_format_desc() {
    if (quant_format_desc(QuantFormatId::None) != nullptr) fail("desc(None) must be null");

    const QuantFormatDesc* d = quant_format_desc(QuantFormatId::Bf16);
    if (d != &kBf16FormatDesc) { fail("desc(Bf16)"); return; }
    if (d->block_elements != 16 || d->code_bytes_per_block != 32) fail("Bf16 desc geometry");
    if (d->meta[0].count_mode != MetaCountMode::None) fail("Bf16 desc meta");

    d = quant_format_desc(QuantFormatId::Psq4);
    if (d != &kPsq4FormatDesc) { fail("desc(Psq4)"); return; }
    if (d->block_elements != 32 || d->code_bytes_per_block != 16) fail("Psq4 desc geometry");
    if (d->meta[0].count_mode != MetaCountMode::PerBlock || d->meta[0].element_bytes != 2) fail("Psq4 desc metadata1");
    if (d->meta[1].count_mode != MetaCountMode::None) fail("Psq4 desc metadata2");

    d = quant_format_desc(QuantFormatId::Psq8);
    if (d != &kPsq8FormatDesc) { fail("desc(Psq8)"); return; }
    if (d->block_elements != 32 || d->code_bytes_per_block != 32) fail("Psq8 desc geometry");
    if (d->meta[0].count_mode != MetaCountMode::PerBlock || d->meta[0].element_bytes != 2) fail("Psq8 desc metadata1");
}

int main() {
    test_format_ids();
    test_format_desc();
    test_size_single_source();
    test_canonical_dequant(QuantFormatId::Psq4, "psq4");
    test_canonical_dequant(QuantFormatId::Psq8, "psq8");

    test_preshuffle_roundtrip(QuantFormatId::Psq4, "psq4");
    test_preshuffle_roundtrip(QuantFormatId::Psq8, "psq8");

    test_native_layout_contract(QuantFormatId::Psq4, "psq4 layout");
    test_native_layout_contract(QuantFormatId::Psq8, "psq8 layout");

    test_bank_mapping(QuantFormatId::Psq4, "bank psq4");
    test_bank_mapping(QuantFormatId::Psq8, "bank psq8");

    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
