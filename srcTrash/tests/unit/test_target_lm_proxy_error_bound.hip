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

std::size_t psq8_bytes(uint32_t rows, uint32_t k_padded) {
    return ((static_cast<std::size_t>(rows) + 15u) / 16u) *
           static_cast<std::size_t>(k_padded) * 16u;
}

Dflash2Int2Codebook fixed_codebook() {
    Dflash2Int2Codebook cb;
    cb.values = {0x00u, 0x30u, 0x38u, 0x3Cu};
    for (uint32_t i = 0u; i < 4u; ++i)
        cb.decoded[i] = ps::quantization::fpx::e4m3_decode_f32(cb.values[i]);
    return cb;
}

struct Case {
    uint32_t rows;
    uint32_t k_padded;
};

bool run_case(const Case& c, uint32_t seed) {
    const Dflash2Int2Codebook codebook = fixed_codebook();
    std::vector<uint8_t> map_table(256u, 0u);
    std::vector<uint32_t> expand_table(256u, 0u);
    ps::kernel::dflash2_int2_build_tables(codebook, map_table.data(), expand_table.data());

    float error_table[256];
    ps::kernel::dflash2_int2_build_error_table(codebook, map_table.data(), error_table);

    Rng rng;
    rng.state = seed;

    const std::size_t scale_count =
        ((static_cast<std::size_t>(c.rows) + 15u) / 16u) * (c.k_padded / 2u);
    const std::vector<uint8_t> finite = {0x00u, 0x30u, 0x34u, 0x38u, 0x3Cu, 0xB0u, 0xB8u};
    std::vector<uint8_t> psq8(psq8_bytes(c.rows, c.k_padded), 0u);
    std::vector<uint16_t> scales(scale_count, 0u);
    for (uint32_t o = 0u; o < c.rows; ++o) {
        for (uint32_t k = 0u; k < c.k_padded; ++k)
            psq8[dflash2_int2_test::psq8_code_index(o, k, c.k_padded)] =
                finite[rng.below(static_cast<uint32_t>(finite.size()))];
        for (uint32_t ib = 0u; ib < c.k_padded / 32u; ++ib)
            scales[dflash2_int2_test::psq8_scale_index(o, ib * 32u, c.k_padded)] =
                dflash2_int2_test::bf16_bits(0.01f + 0.9f * rng.unit());
    }

    uint8_t* dev_codes = nullptr;
    uint8_t* dev_scale = nullptr;
    float* dev_error_table = nullptr;
    float* dev_error = nullptr;
    const bool alloc_ok =
        hipMalloc(&dev_codes, psq8.size()) == hipSuccess &&
        hipMalloc(&dev_scale, scale_count * 2u) == hipSuccess &&
        hipMalloc(&dev_error_table, 256u * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev_error, c.rows * sizeof(float)) == hipSuccess;
    check(alloc_ok, "error bound allocation");
    if (!alloc_ok) {
        (void)hipFree(dev_codes);
        (void)hipFree(dev_scale);
        (void)hipFree(dev_error_table);
        (void)hipFree(dev_error);
        return false;
    }

    (void)hipMemcpy(dev_codes, psq8.data(), psq8.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_scale, scales.data(), scale_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_error_table, error_table, 256u * sizeof(float),
                    hipMemcpyHostToDevice);

    const hipError_t err = ps::kernel::launch_dflash2_int2_row_error_l2(
        dev_codes, dev_scale, dev_error_table, c.rows, c.k_padded, c.k_padded, dev_error,
        nullptr);
    check(err == hipSuccess, "row error l2 launch");
    (void)hipDeviceSynchronize();

    std::vector<float> gpu_error(c.rows, 0.0f);
    (void)hipMemcpy(gpu_error.data(), dev_error, c.rows * sizeof(float),
                    hipMemcpyDeviceToHost);

    bool all_finite = true;
    for (uint32_t o = 0u; o < c.rows; ++o) {
        if (!std::isfinite(gpu_error[o]))
            all_finite = false;
    }
    check(all_finite, "row error l2 finite");

    bool matches = true;
    for (uint32_t o = 0u; o < c.rows && matches; ++o) {
        double sum = 0.0;
        for (uint32_t k = 0u; k < c.k_padded; ++k) {
            const uint8_t code =
                psq8[dflash2_int2_test::psq8_code_index(o, k, c.k_padded)];
            const uint16_t sbits =
                scales[dflash2_int2_test::psq8_scale_index(o, (k >> 5u) * 32u, c.k_padded)];
            const float sval = [&] {
                const uint32_t u = static_cast<uint32_t>(sbits) << 16;
                return *reinterpret_cast<const float*>(&u);
            }();
            const double d = static_cast<double>(error_table[code]) *
                             static_cast<double>(sval);
            sum += d * d;
        }
        const double ref = std::sqrt(sum);
        const double tol = 1e-4 * (1.0 + ref);
        if (std::fabs(ref - static_cast<double>(gpu_error[o])) > tol)
            matches = false;
    }
    check(matches, "row error l2 matches cpu reference");

    (void)hipFree(dev_codes);
    (void)hipFree(dev_scale);
    (void)hipFree(dev_error_table);
    (void)hipFree(dev_error);
    return true;
}

bool run_bound_case(uint32_t rows, uint32_t vocab, uint32_t k_padded, uint32_t seed) {
    const Dflash2Int2Codebook codebook = fixed_codebook();
    std::vector<uint8_t> map_table(256u, 0u);
    std::vector<uint32_t> expand_table(256u, 0u);
    ps::kernel::dflash2_int2_build_tables(codebook, map_table.data(), expand_table.data());
    float error_table[256];
    ps::kernel::dflash2_int2_build_error_table(codebook, map_table.data(), error_table);

    Rng rng;
    rng.state = seed;
    const std::vector<uint8_t> finite = {0x00u, 0x30u, 0x34u, 0x38u, 0x3Cu, 0xB0u, 0xB8u};
    const std::size_t scale_count =
        ((static_cast<std::size_t>(vocab) + 15u) / 16u) * (k_padded / 2u);
    std::vector<uint8_t> psq8(psq8_bytes(vocab, k_padded), 0u);
    std::vector<uint16_t> scales(scale_count, 0u);
    for (uint32_t o = 0u; o < vocab; ++o) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            psq8[dflash2_int2_test::psq8_code_index(o, k, k_padded)] =
                finite[rng.below(static_cast<uint32_t>(finite.size()))];
        for (uint32_t ib = 0u; ib < k_padded / 32u; ++ib)
            scales[dflash2_int2_test::psq8_scale_index(o, ib * 32u, k_padded)] =
                dflash2_int2_test::bf16_bits(0.01f + 0.9f * rng.unit());
    }

    uint8_t* dev_codes = nullptr;
    uint8_t* dev_scale = nullptr;
    float* dev_error_table = nullptr;
    float* dev_error = nullptr;
    if (hipMalloc(&dev_codes, psq8.size()) != hipSuccess ||
        hipMalloc(&dev_scale, scale_count * 2u) != hipSuccess ||
        hipMalloc(&dev_error_table, 256u * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_error, vocab * sizeof(float)) != hipSuccess)
        return false;
    (void)hipMemcpy(dev_codes, psq8.data(), psq8.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_scale, scales.data(), scale_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_error_table, error_table, 256u * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)ps::kernel::launch_dflash2_int2_row_error_l2(
        dev_codes, dev_scale, dev_error_table, vocab, k_padded, k_padded, dev_error, nullptr);
    (void)hipDeviceSynchronize();
    std::vector<float> gpu_error(vocab, 0.0f);
    (void)hipMemcpy(gpu_error.data(), dev_error, vocab * sizeof(float),
                    hipMemcpyDeviceToHost);

    bool held = true;
    for (uint32_t m = 0u; m < rows && held; ++m) {
        std::vector<double> a(k_padded, 0.0);
        for (uint32_t k = 0u; k < k_padded; ++k) {
            const uint8_t code = finite[rng.below(static_cast<uint32_t>(finite.size()))];
            a[k] = static_cast<double>(ps::quantization::fpx::e4m3_decode_f32(code)) *
                   (0.3 + rng.unit());
        }
        double a2 = 0.0;
        for (uint32_t k = 0u; k < k_padded; ++k)
            a2 += a[k] * a[k];
        const double al2 = std::sqrt(a2);
        for (uint32_t o = 0u; o < vocab && held; ++o) {
            double z_exact = 0.0;
            double z_proxy = 0.0;
            for (uint32_t k = 0u; k < k_padded; ++k) {
                const uint8_t code =
                    psq8[dflash2_int2_test::psq8_code_index(o, k, k_padded)];
                const uint16_t sbits =
                    scales[dflash2_int2_test::psq8_scale_index(o, (k >> 5u) * 32u, k_padded)];
                const uint32_t u = static_cast<uint32_t>(sbits) << 16;
                const double s = static_cast<double>(*reinterpret_cast<const float*>(&u));
                const double we = static_cast<double>(
                                     ps::quantization::fpx::e4m3_decode_f32(code)) *
                                 s;
                const double wp =
                    static_cast<double>(codebook.decoded[map_table[code]]) * s;
                z_exact += a[k] * we;
                z_proxy += a[k] * wp;
            }
            const double bound =
                al2 * static_cast<double>(gpu_error[o]) + 1e-3 * (1.0 + std::fabs(z_exact));
            if (std::fabs(z_exact - z_proxy) > bound) {
                held = false;
                std::printf("FAIL: bound violated m=%u o=%u diff=%.6g bound=%.6g\n", m, o,
                            std::fabs(z_exact - z_proxy), bound);
            }
        }
    }
    check(held, "cauchy-schwarz bound holds");

    (void)hipFree(dev_codes);
    (void)hipFree(dev_scale);
    (void)hipFree(dev_error_table);
    (void)hipFree(dev_error);
    return true;
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

    const Case cases[] = {
        {1u, 32u}, {1u, 64u}, {5u, 64u}, {7u, 128u}, {17u, 256u}, {31u, 32u},
    };
    uint32_t seed = 0x9e3779b9u;
    for (const Case& c : cases) {
        if (!run_case(c, seed))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }

    check(run_bound_case(4u, 40u, 64u, 0x2545f491u), "bound case 4x40x64");
    check(run_bound_case(2u, 80u, 128u, 0x1b873593u), "bound case 2x80x128");
    check(run_bound_case(3u, 33u, 32u, 0x85ebca6bu), "bound case 3x33x32");

    std::printf("test_target_lm_proxy_error_bound: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
