#include "support/dflash2_int2_test_common.h"

#include <phaseshift/quantization/fpx/e4m3.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using dflash2_int2_test::Rng;
using ps::kernel::Dflash2Int2Codebook;

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

uint32_t checksum(const uint8_t* data, std::size_t n) {
    uint32_t h = 2166136261u;
    for (std::size_t i = 0u; i < n; ++i) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

std::size_t psq8_bytes(uint32_t rows, uint32_t k_padded) {
    return ((static_cast<std::size_t>(rows) + 15u) / 16u) *
           static_cast<std::size_t>(k_padded) * 16u;
}

std::size_t int2_bytes(uint32_t rows, uint32_t k_padded) {
    return psq8_bytes(rows, k_padded) / 4u;
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;

    const uint32_t rows = 32u;
    const uint32_t k_padded = 64u;
    const uint32_t scale_stride = k_padded;

    Dflash2Int2Codebook codebook;
    codebook.values = {0x00u, 0x38u, 0x3Cu, 0x40u};
    for (uint32_t c = 0u; c < 4u; ++c)
        codebook.decoded[c] = ps::quantization::fpx::e4m3_decode_f32(codebook.values[c]);

    std::vector<uint8_t> map_table(256u, 0u);
    std::vector<uint32_t> expand_table(256u, 0u);
    ps::kernel::dflash2_int2_build_tables(codebook, map_table.data(), expand_table.data());

    for (uint32_t b = 0u; b < 256u; ++b) {
        const float v = ps::quantization::fpx::e4m3_decode_f32(static_cast<uint8_t>(b));
        uint32_t expect = 0u;
        if (std::isfinite(v)) {
            float best = std::fabs(v - codebook.decoded[0]);
            for (uint32_t c = 1u; c < 4u; ++c) {
                const float d = std::fabs(v - codebook.decoded[c]);
                if (d < best) {
                    best = d;
                    expect = c;
                }
            }
        }
        check(map_table[b] == expect, "map table nearest value");
    }
    for (uint32_t packed = 0u; packed < 256u; ++packed) {
        uint32_t expect = 0u;
        for (uint32_t q = 0u; q < 4u; ++q)
            expect |= static_cast<uint32_t>(codebook.values[(packed >> (2u * q)) & 3u])
                      << (8u * q);
        check(expand_table[packed] == expect, "expand table layout");
    }

    Rng rng;
    rng.state = 0x1234u;
    const std::vector<uint8_t> finite_codes = {0x00u, 0x38u, 0x3Cu, 0x40u,
                                               0x30u, 0x34u, 0xB8u, 0xBCu};
    const std::size_t code_count = psq8_bytes(rows, k_padded);
    std::vector<uint8_t> codes(code_count, 0u);
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t k = 0u; k < k_padded; ++k) {
            codes[dflash2_int2_test::psq8_code_index(r, k, k_padded)] =
                finite_codes[rng.below(static_cast<uint32_t>(finite_codes.size()))];
        }
    }

    std::vector<uint8_t> expected_int2(int2_bytes(rows, k_padded), 0u);
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t k = 0u; k < k_padded; ++k) {
            const std::size_t idx = dflash2_int2_test::int2_code_index(r, k, k_padded);
            const uint32_t shift = dflash2_int2_test::int2_shift(k);
            const uint8_t mapped = map_table[codes[dflash2_int2_test::psq8_code_index(
                r, k, k_padded)]];
            expected_int2[idx] = static_cast<uint8_t>(expected_int2[idx] | (mapped << shift));
        }
    }

    uint8_t* dev_codes = nullptr;
    uint8_t* dev_map = nullptr;
    uint8_t* dev_int2 = nullptr;
    uint8_t* dev_int2_again = nullptr;
    (void)hipMalloc(&dev_codes, code_count);
    (void)hipMalloc(&dev_map, map_table.size());
    (void)hipMalloc(&dev_int2, int2_bytes(rows, k_padded));
    (void)hipMalloc(&dev_int2_again, int2_bytes(rows, k_padded));
    (void)hipMemcpy(dev_codes, codes.data(), code_count, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_map, map_table.data(), map_table.size(), hipMemcpyHostToDevice);
    (void)hipMemset(dev_int2, 0, int2_bytes(rows, k_padded));
    (void)hipMemset(dev_int2_again, 0, int2_bytes(rows, k_padded));

    hipError_t err = ps::kernel::launch_dflash2_int2_pack(
        dev_codes, dev_map, dev_int2, rows, k_padded, nullptr);
    check(err == hipSuccess, "pack launch");
    err = ps::kernel::launch_dflash2_int2_pack(
        dev_codes, dev_map, dev_int2_again, rows, k_padded, nullptr);
    check(err == hipSuccess, "pack launch repeat");
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> packed(int2_bytes(rows, k_padded), 0u);
    std::vector<uint8_t> packed_again(int2_bytes(rows, k_padded), 0u);
    (void)hipMemcpy(packed.data(), dev_int2, packed.size(), hipMemcpyDeviceToHost);
    (void)hipMemcpy(packed_again.data(), dev_int2_again, packed_again.size(),
                    hipMemcpyDeviceToHost);
    check(std::memcmp(packed.data(), expected_int2.data(), packed.size()) == 0,
          "packed bytes match expected preshuffle order");
    check(std::memcmp(packed.data(), packed_again.data(), packed.size()) == 0,
          "pack deterministic");

    bool round_trip = true;
    for (uint32_t r = 0u; r < rows && round_trip; ++r) {
        for (uint32_t k = 0u; k < k_padded; ++k) {
            const std::size_t idx = dflash2_int2_test::int2_code_index(r, k, k_padded);
            const uint32_t shift = dflash2_int2_test::int2_shift(k);
            const uint32_t code = (packed[idx] >> shift) & 3u;
            const uint8_t original = codes[dflash2_int2_test::psq8_code_index(r, k, k_padded)];
            if (code != map_table[original])
                round_trip = false;
        }
    }
    check(round_trip, "pack round trip through map");
    std::printf("INFO: pack checksum=0x%08x bytes=%zu\n",
                checksum(packed.data(), packed.size()), packed.size());

    const std::size_t scale_count =
        ((static_cast<std::size_t>(rows) + 15u) / 16u) * (scale_stride / 2u);
    std::vector<uint16_t> host_scales(scale_count, dflash2_int2_test::bf16_bits(0.5f));
    uint8_t* dev_scale_bytes = nullptr;
    (void)hipMalloc(&dev_scale_bytes, scale_count * 2u);
    (void)hipMemcpy(dev_scale_bytes, host_scales.data(), scale_count * 2u,
                    hipMemcpyHostToDevice);

    uint64_t* dev_hist = nullptr;
    (void)hipMalloc(&dev_hist, 256u * sizeof(uint64_t));
    (void)hipMemset(dev_hist, 0, 256u * sizeof(uint64_t));
    err = ps::kernel::launch_dflash2_int2_code_histogram(
        dev_codes, dev_scale_bytes, rows, k_padded, scale_stride, dev_hist, nullptr);
    check(err == hipSuccess, "histogram launch");
    std::vector<uint64_t> histogram(256u, 0ull);
    (void)hipMemcpy(histogram.data(), dev_hist, 256u * sizeof(uint64_t), hipMemcpyDeviceToHost);
    uint64_t total = 0ull;
    for (uint64_t v : histogram)
        total += v;
    const uint64_t expected_total = static_cast<uint64_t>(rows) * k_padded * 262144ull;
    check(total == expected_total, "histogram total weight");

    const Dflash2Int2Codebook derived =
        ps::kernel::dflash2_int2_derive_codebook(histogram.data(), false);
    const Dflash2Int2Codebook derived2 =
        ps::kernel::dflash2_int2_derive_codebook(histogram.data(), false);
    check(derived.values == derived2.values, "codebook derivation deterministic");
    for (uint32_t c = 0u; c < 3u; ++c)
        check(derived.decoded[c] <= derived.decoded[c + 1u], "codebook ascending");
    const Dflash2Int2Codebook sym =
        ps::kernel::dflash2_int2_derive_codebook(histogram.data(), true);
    check(sym.decoded[0] == -sym.decoded[3] && sym.decoded[1] == -sym.decoded[2],
          "symmetric codebook symmetric");
    std::printf("INFO: derived lloyd=[%.6f %.6f %.6f %.6f] sym=[%.6f %.6f %.6f %.6f]\n",
                derived.decoded[0], derived.decoded[1], derived.decoded[2], derived.decoded[3],
                sym.decoded[0], sym.decoded[1], sym.decoded[2], sym.decoded[3]);

    (void)hipFree(dev_codes);
    (void)hipFree(dev_map);
    (void)hipFree(dev_int2);
    (void)hipFree(dev_int2_again);
    (void)hipFree(dev_hist);
    (void)hipFree(dev_scale_bytes);

    std::printf("test_dflash2_int2_pack: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
