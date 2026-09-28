#include "support/dflash2_int2_test_common.h"

#include <phaseshift/quantization/fpx/e4m3.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

using dflash2_int2_test::Rng;

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

bool run_case(uint32_t rows, uint32_t vocab, uint32_t k_padded, uint32_t pool, uint32_t seed) {
    const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(k_padded, rows);
    const uint32_t act_code_stride = layout.code_row_stride_bytes;
    const uint32_t act_scale_stride = layout.scale_row_stride_bytes;
    const uint32_t weight_scale_stride = k_padded;

    Rng rng;
    rng.state = seed;

    std::vector<uint8_t> psq8(psq8_bytes(vocab, k_padded), 0u);
    const std::vector<uint8_t> finite = {0x00u, 0x30u, 0x34u, 0x38u, 0x3Cu, 0xB0u, 0xB8u, 0xBCu};
    for (uint32_t o = 0u; o < vocab; ++o) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            psq8[dflash2_int2_test::psq8_code_index(o, k, k_padded)] =
                finite[rng.below(static_cast<uint32_t>(finite.size()))];
    }
    const std::size_t scale_count =
        ((static_cast<std::size_t>(vocab) + 15u) / 16u) * (weight_scale_stride / 2u);
    std::vector<uint16_t> scales(scale_count, 0u);
    for (uint32_t o = 0u; o < vocab; ++o) {
        for (uint32_t ib = 0u; ib < k_padded / 32u; ++ib)
            scales[dflash2_int2_test::psq8_scale_index(o, ib * 32u, weight_scale_stride)] =
                dflash2_int2_test::bf16_bits(0.05f + 0.4f * rng.unit());
    }
    std::vector<uint8_t> act_codes(act_code_stride * ((rows + 15u) / 16u) * 16u, 0u);
    std::vector<float> act_scales(static_cast<std::size_t>(act_scale_stride / 4u) * rows, 0.0f);
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t k = 0u; k < k_padded; ++k)
            act_codes[dflash2_int2_test::act_code_index(r, k, act_code_stride)] =
                finite[rng.below(static_cast<uint32_t>(finite.size()))];
        act_scales[static_cast<std::size_t>(r) * (act_scale_stride / 4u)] = 0.3f + rng.unit();
    }

    std::vector<int32_t> candidates(static_cast<std::size_t>(rows) * pool, 0);
    for (uint32_t r = 0u; r < rows; ++r) {
        std::vector<int32_t> all(vocab);
        for (uint32_t i = 0u; i < vocab; ++i)
            all[i] = static_cast<int32_t>(i);
        for (uint32_t i = vocab; i > 1u; --i) {
            const uint32_t j = rng.below(i);
            const int32_t t = all[i - 1u];
            all[i - 1u] = all[j];
            all[j] = t;
        }
        for (uint32_t i = 0u; i < pool; ++i)
            candidates[static_cast<std::size_t>(r) * pool + i] = all[i];
        candidates[static_cast<std::size_t>(r) * pool] = 0;
        candidates[static_cast<std::size_t>(r) * pool + 1u] = static_cast<int32_t>(vocab - 1u);
    }

    uint8_t* dev_psq8 = nullptr;
    uint8_t* dev_scale_bytes = nullptr;
    uint8_t* dev_act = nullptr;
    float* dev_act_scales = nullptr;
    int32_t* dev_cand = nullptr;
    float* dev_full = nullptr;
    float* dev_rerank = nullptr;
    float* dev_rerank2 = nullptr;
    const std::size_t full_elems = static_cast<std::size_t>(rows) * vocab;
    const std::size_t rerank_elems = static_cast<std::size_t>(rows) * pool;
    if (hipMalloc(&dev_psq8, psq8.size()) != hipSuccess ||
        hipMalloc(&dev_scale_bytes, scale_count * 2u) != hipSuccess ||
        hipMalloc(&dev_act, act_codes.size()) != hipSuccess ||
        hipMalloc(&dev_act_scales, act_scales.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_cand, candidates.size() * sizeof(int32_t)) != hipSuccess ||
        hipMalloc(&dev_full, full_elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_rerank, rerank_elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_rerank2, rerank_elems * sizeof(float)) != hipSuccess)
        return false;
    (void)hipMemcpy(dev_psq8, psq8.data(), psq8.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_scale_bytes, scales.data(), scale_count * 2u, hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act, act_codes.data(), act_codes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act_scales, act_scales.data(), act_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_cand, candidates.data(), candidates.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemset(dev_full, 0, full_elems * sizeof(float));
    (void)hipMemset(dev_rerank, 0, rerank_elems * sizeof(float));
    (void)hipMemset(dev_rerank2, 0, rerank_elems * sizeof(float));

    ps::kernel::Psq8GemmConfig cfg;
    cfg.id = ps::kernel::Psq8GemmConfigId::RowBlock1;
    hipError_t err = ps::kernel::launch_gemm_psq8_w8a8_wmma(
        cfg, dev_psq8, dev_scale_bytes, reinterpret_cast<const int8_t*>(dev_act),
        dev_act_scales, dev_full, ps::kernel::Psq8GemmOutputDType::F32, rows, vocab, k_padded,
        weight_scale_stride, act_code_stride, act_scale_stride, vocab, nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: psq8 launch rows=%u vocab=%u err=%s\n", rows, vocab,
                    hipGetErrorString(err));
        return false;
    }
    err = ps::kernel::launch_dflash2_psq8_candidate_rerank(
        dev_psq8, dev_scale_bytes, dev_act, dev_act_scales, dev_cand, dev_rerank, rows, pool,
        k_padded, weight_scale_stride, act_code_stride, act_scale_stride, pool, nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: rerank launch rows=%u vocab=%u pool=%u err=%s\n", rows, vocab, pool,
                    hipGetErrorString(err));
        return false;
    }
    (void)ps::kernel::launch_dflash2_psq8_candidate_rerank(
        dev_psq8, dev_scale_bytes, dev_act, dev_act_scales, dev_cand, dev_rerank2, rows, pool,
        k_padded, weight_scale_stride, act_code_stride, act_scale_stride, pool, nullptr);
    (void)hipDeviceSynchronize();

    std::vector<float> full(full_elems, 0.0f);
    std::vector<float> rerank(rerank_elems, 0.0f);
    std::vector<float> rerank2(rerank_elems, 0.0f);
    (void)hipMemcpy(full.data(), dev_full, full_elems * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(rerank.data(), dev_rerank, rerank_elems * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(rerank2.data(), dev_rerank2, rerank_elems * sizeof(float),
                    hipMemcpyDeviceToHost);

    bool exact = true;
    bool repeat = true;
    float max_abs = 0.0f;
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t i = 0u; i < pool; ++i) {
            const std::size_t idx = static_cast<std::size_t>(r) * pool + i;
            const float got = rerank[idx];
            const float want = full[static_cast<std::size_t>(r) * vocab +
                                    static_cast<std::size_t>(candidates[idx])];
            if (std::memcmp(&got, &want, sizeof(float)) != 0)
                exact = false;
            if (std::memcmp(&got, &rerank2[idx], sizeof(float)) != 0)
                repeat = false;
            const float d = std::fabs(got - want);
            if (d > max_abs)
                max_abs = d;
        }
    }
    char label[160];
    std::snprintf(label, sizeof(label), "rerank bit exact rows=%u vocab=%u k=%u pool=%u", rows,
                  vocab, k_padded, pool);
    check(exact, label);
    std::snprintf(label, sizeof(label), "rerank deterministic rows=%u vocab=%u pool=%u", rows,
                  vocab, pool);
    check(repeat, label);
    if (!exact)
        std::printf("INFO: rerank max_abs=%.9g\n", static_cast<double>(max_abs));

    (void)hipFree(dev_psq8);
    (void)hipFree(dev_scale_bytes);
    (void)hipFree(dev_act);
    (void)hipFree(dev_act_scales);
    (void)hipFree(dev_cand);
    (void)hipFree(dev_full);
    (void)hipFree(dev_rerank);
    (void)hipFree(dev_rerank2);
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

    uint32_t seed = 0x2545f491u;
    const uint32_t pools[] = {16u, 32u, 64u};
    for (uint32_t pool : pools) {
        if (!run_case(7u, 64u, 128u, pool, seed))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }
    if (!run_case(7u, 32u, 64u, 32u, seed))
        ++failed;
    seed = seed * 1664525u + 1013904223u;
    if (!run_case(7u, 256u, 256u, 80u, seed))
        ++failed;
    seed = seed * 1664525u + 1013904223u;
    if (!run_case(1u, 64u, 128u, 32u, seed))
        ++failed;
    seed = seed * 1664525u + 1013904223u;
    if (!run_case(3u, 128u, 128u, 48u, seed))
        ++failed;

    std::printf("test_dflash2_psq8_rerank: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
