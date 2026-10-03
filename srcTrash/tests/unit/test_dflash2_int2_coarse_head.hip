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

struct Case {
    uint32_t rows;
    uint32_t vocab;
    uint32_t k_padded;
};

bool run_case(const Case& c, uint32_t seed) {
    const Dflash2Int2Codebook codebook = [] {
        Dflash2Int2Codebook cb;
        cb.values = {0x00u, 0x30u, 0x38u, 0x3Cu};
        for (uint32_t i = 0u; i < 4u; ++i)
            cb.decoded[i] = ps::quantization::fpx::e4m3_decode_f32(cb.values[i]);
        return cb;
    }();
    std::vector<uint8_t> map_table(256u, 0u);
    std::vector<uint32_t> expand_table(256u, 0u);
    ps::kernel::dflash2_int2_build_tables(codebook, map_table.data(), expand_table.data());

    const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(c.k_padded, c.rows);
    const uint32_t act_code_stride = layout.code_row_stride_bytes;
    const uint32_t act_scale_stride = layout.scale_row_stride_bytes;
    const uint32_t weight_scale_stride = c.k_padded;

    Rng rng;
    rng.state = seed;

    std::vector<uint8_t> int2(psq8_bytes(c.vocab, c.k_padded) / 4u, 0u);
    std::vector<uint8_t> psq8(psq8_bytes(c.vocab, c.k_padded), 0u);
    for (uint32_t o = 0u; o < c.vocab; ++o) {
        for (uint32_t k = 0u; k < c.k_padded; ++k) {
            const uint32_t idx = rng.below(4u);
            const std::size_t i2 = dflash2_int2_test::int2_code_index(o, k, c.k_padded);
            const uint32_t shift = dflash2_int2_test::int2_shift(k);
            int2[i2] = static_cast<uint8_t>(int2[i2] | (idx << shift));
            psq8[dflash2_int2_test::psq8_code_index(o, k, c.k_padded)] =
                codebook.values[idx];
        }
    }

    const std::size_t scale_count =
        ((static_cast<std::size_t>(c.vocab) + 15u) / 16u) * (weight_scale_stride / 2u);
    std::vector<uint16_t> scales(scale_count, 0u);
    for (uint32_t o = 0u; o < c.vocab; ++o) {
        for (uint32_t ib = 0u; ib < c.k_padded / 32u; ++ib) {
            scales[dflash2_int2_test::psq8_scale_index(o, ib * 32u, weight_scale_stride)] =
                dflash2_int2_test::bf16_bits(0.05f + 0.5f * rng.unit());
        }
    }

    std::vector<uint8_t> act_codes(act_code_stride * ((c.rows + 15u) / 16u) * 16u, 0u);
    std::vector<float> act_scales(act_scale_stride / 4u * c.rows, 0.0f);
    const std::vector<uint8_t> finite = {0x00u, 0x30u, 0x34u, 0x38u, 0x3Cu, 0xB0u, 0xB8u};
    for (uint32_t r = 0u; r < c.rows; ++r) {
        for (uint32_t k = 0u; k < c.k_padded; ++k)
            act_codes[dflash2_int2_test::act_code_index(r, k, act_code_stride)] =
                finite[rng.below(static_cast<uint32_t>(finite.size()))];
        act_scales[static_cast<std::size_t>(r) * (act_scale_stride / 4u)] =
            0.2f + rng.unit();
    }

    uint8_t* dev_int2 = nullptr;
    uint8_t* dev_psq8 = nullptr;
    uint8_t* dev_scale_bytes = nullptr;
    uint8_t* dev_act = nullptr;
    float* dev_act_scales = nullptr;
    uint32_t* dev_expand = nullptr;
    float* dev_coarse = nullptr;
    float* dev_ref = nullptr;
    float* dev_coarse2 = nullptr;
    const std::size_t out_elems = static_cast<std::size_t>(c.rows) * c.vocab;
    if (hipMalloc(&dev_int2, int2.size()) != hipSuccess ||
        hipMalloc(&dev_psq8, psq8.size()) != hipSuccess ||
        hipMalloc(&dev_scale_bytes, scale_count * 2u) != hipSuccess ||
        hipMalloc(&dev_act, act_codes.size()) != hipSuccess ||
        hipMalloc(&dev_act_scales, act_scales.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_expand, expand_table.size() * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&dev_coarse, out_elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_coarse2, out_elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_ref, out_elems * sizeof(float)) != hipSuccess)
        return false;

    (void)hipMemcpy(dev_int2, int2.data(), int2.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_psq8, psq8.data(), psq8.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_scale_bytes, scales.data(), scale_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act, act_codes.data(), act_codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act_scales, act_scales.data(), act_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_expand, expand_table.data(), expand_table.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemset(dev_coarse, 0, out_elems * sizeof(float));
    (void)hipMemset(dev_coarse2, 0, out_elems * sizeof(float));
    (void)hipMemset(dev_ref, 0, out_elems * sizeof(float));

    hipError_t err = ps::kernel::launch_dflash2_int2_coarse_head(
        dev_int2, dev_scale_bytes, dev_act, dev_act_scales, dev_expand, dev_coarse, c.rows,
        c.vocab, c.k_padded, weight_scale_stride, act_code_stride, act_scale_stride, c.vocab,
        nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: coarse launch rows=%u vocab=%u k=%u err=%s\n", c.rows, c.vocab,
                    c.k_padded, hipGetErrorString(err));
        return false;
    }
    (void)ps::kernel::launch_dflash2_int2_coarse_head(
        dev_int2, dev_scale_bytes, dev_act, dev_act_scales, dev_expand, dev_coarse2, c.rows,
        c.vocab, c.k_padded, weight_scale_stride, act_code_stride, act_scale_stride, c.vocab,
        nullptr);

    ps::kernel::Psq8GemmConfig cfg;
    cfg.id = ps::kernel::Psq8GemmConfigId::RowBlock1;
    err = ps::kernel::launch_gemm_psq8_w8a8_wmma(
        cfg, dev_psq8, dev_scale_bytes, reinterpret_cast<const int8_t*>(dev_act),
        dev_act_scales, dev_ref, ps::kernel::Psq8GemmOutputDType::F32, c.rows, c.vocab,
        c.k_padded, weight_scale_stride, act_code_stride, act_scale_stride, c.vocab, nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: psq8 reference launch err=%s\n", hipGetErrorString(err));
        return false;
    }
    (void)hipDeviceSynchronize();

    std::vector<float> coarse(out_elems, 0.0f);
    std::vector<float> coarse2(out_elems, 0.0f);
    std::vector<float> ref(out_elems, 0.0f);
    (void)hipMemcpy(coarse.data(), dev_coarse, out_elems * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(coarse2.data(), dev_coarse2, out_elems * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(ref.data(), dev_ref, out_elems * sizeof(float), hipMemcpyDeviceToHost);

    bool exact = true;
    bool all_finite = true;
    bool repeat = true;
    for (std::size_t i = 0u; i < out_elems; ++i) {
        if (std::memcmp(&coarse[i], &ref[i], sizeof(float)) != 0)
            exact = false;
        if (std::memcmp(&coarse[i], &coarse2[i], sizeof(float)) != 0)
            repeat = false;
        if (!std::isfinite(coarse[i]))
            all_finite = false;
    }
    char label[160];
    std::snprintf(label, sizeof(label), "coarse==psq8 bit exact rows=%u vocab=%u k=%u",
                  c.rows, c.vocab, c.k_padded);
    check(exact, label);
    std::snprintf(label, sizeof(label), "coarse finite rows=%u vocab=%u k=%u", c.rows,
                  c.vocab, c.k_padded);
    check(all_finite, label);
    std::snprintf(label, sizeof(label), "coarse deterministic rows=%u vocab=%u k=%u", c.rows,
                  c.vocab, c.k_padded);
    check(repeat, label);

    (void)hipFree(dev_int2);
    (void)hipFree(dev_psq8);
    (void)hipFree(dev_scale_bytes);
    (void)hipFree(dev_act);
    (void)hipFree(dev_act_scales);
    (void)hipFree(dev_expand);
    (void)hipFree(dev_coarse);
    (void)hipFree(dev_coarse2);
    (void)hipFree(dev_ref);
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
        {1u, 32u, 64u}, {3u, 48u, 64u}, {7u, 64u, 128u}, {7u, 80u, 128u}, {7u, 256u, 256u},
    };
    uint32_t seed = 0x51ed2701u;
    for (const Case& c : cases) {
        if (!run_case(c, seed))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }

    std::printf("test_dflash2_int2_coarse_head: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
