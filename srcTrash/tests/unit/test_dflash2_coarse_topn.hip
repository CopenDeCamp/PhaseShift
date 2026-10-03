#include "support/dflash2_int2_test_common.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

using dflash2_int2_test::Rng;
using dflash2_int2_test::TopnOracle;

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

bool run_case(uint32_t rows, uint32_t vocab, uint32_t pool, bool tie, uint32_t seed) {
    const uint32_t partitions = ps::kernel::dflash2_int2_default_partitions(vocab);
    const uint32_t scratch_row_stride = partitions * ps::kernel::kDflash2Int2MaxPool;
    Rng rng;
    rng.state = seed;

    std::vector<float> logits(static_cast<std::size_t>(rows) * vocab, 0.0f);
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t v = 0u; v < vocab; ++v) {
            float value;
            if (tie) {
                value = 1.0f - static_cast<float>(v % 3u) * 0.25f;
            } else {
                value = rng.sym();
            }
            logits[static_cast<std::size_t>(r) * vocab + v] = value;
        }
    }

    float* dev_logits = nullptr;
    int32_t* dev_ids = nullptr;
    float* dev_values = nullptr;
    int32_t* dev_scratch_ids = nullptr;
    float* dev_scratch_values = nullptr;
    const std::size_t out_elems = static_cast<std::size_t>(rows) * pool;
    const std::size_t scratch_elems = static_cast<std::size_t>(rows) * scratch_row_stride;
    if (hipMalloc(&dev_logits, logits.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_ids, out_elems * sizeof(int32_t)) != hipSuccess ||
        hipMalloc(&dev_values, out_elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&dev_scratch_ids, scratch_elems * sizeof(int32_t)) != hipSuccess ||
        hipMalloc(&dev_scratch_values, scratch_elems * sizeof(float)) != hipSuccess)
        return false;
    (void)hipMemcpy(dev_logits, logits.data(), logits.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemset(dev_ids, 0, out_elems * sizeof(int32_t));
    (void)hipMemset(dev_values, 0, out_elems * sizeof(float));
    (void)hipMemset(dev_scratch_ids, 0, scratch_elems * sizeof(int32_t));
    (void)hipMemset(dev_scratch_values, 0, scratch_elems * sizeof(float));

    hipError_t err = ps::kernel::launch_dflash2_coarse_topn(
        dev_logits, rows, vocab, vocab, pool, dev_ids, dev_values, dev_scratch_ids,
        dev_scratch_values, scratch_row_stride, partitions, nullptr);
    if (err != hipSuccess) {
        std::printf("FAIL: topn launch vocab=%u pool=%u err=%s\n", vocab, pool,
                    hipGetErrorString(err));
        return false;
    }
    (void)hipDeviceSynchronize();

    std::vector<int32_t> ids(out_elems, 0);
    std::vector<float> values(out_elems, 0.0f);
    (void)hipMemcpy(ids.data(), dev_ids, out_elems * sizeof(int32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(values.data(), dev_values, out_elems * sizeof(float),
                    hipMemcpyDeviceToHost);

    bool ids_exact = true;
    bool values_exact = true;
    bool unique = true;
    char label[160];
    for (uint32_t r = 0u; r < rows; ++r) {
        std::vector<float> row(logits.begin() + static_cast<std::size_t>(r) * vocab,
                               logits.begin() + static_cast<std::size_t>(r + 1u) * vocab);
        const TopnOracle oracle = dflash2_int2_test::cpu_topn(row, pool);
        for (uint32_t i = 0u; i < pool; ++i) {
            const std::size_t idx = static_cast<std::size_t>(r) * pool + i;
            if (ids[idx] != oracle.ids[i])
                ids_exact = false;
            if (std::memcmp(&values[idx], &oracle.values[i], sizeof(float)) != 0)
                values_exact = false;
            for (uint32_t j = i + 1u; j < pool; ++j) {
                if (ids[static_cast<std::size_t>(r) * pool + j] == ids[idx])
                    unique = false;
            }
        }
    }
    std::snprintf(label, sizeof(label), "topn ids exact vocab=%u pool=%u tie=%d", vocab, pool,
                  tie ? 1 : 0);
    check(ids_exact, label);
    std::snprintf(label, sizeof(label), "topn logits bit exact vocab=%u pool=%u tie=%d", vocab,
                  pool, tie ? 1 : 0);
    check(values_exact, label);
    std::snprintf(label, sizeof(label), "topn ids unique vocab=%u pool=%u", vocab, pool);
    check(unique, label);

    (void)hipFree(dev_logits);
    (void)hipFree(dev_ids);
    (void)hipFree(dev_values);
    (void)hipFree(dev_scratch_ids);
    (void)hipFree(dev_scratch_values);
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

    const uint32_t pools[] = {16u, 24u, 32u, 48u, 64u, 80u, 128u};
    uint32_t seed = 0x9e3779b9u;
    for (uint32_t pool : pools) {
        for (bool tie : {false, true}) {
            if (!run_case(7u, 1024u, pool, tie, seed))
                ++failed;
            seed = seed * 1664525u + 1013904223u;
        }
    }
    for (uint32_t pool : pools) {
        if (!run_case(7u, 4096u, pool, false, seed))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }
    for (uint32_t pool : {16u, 32u, 80u, 128u}) {
        for (bool tie : {false, true}) {
            if (!run_case(7u, 248320u, pool, tie, seed))
                ++failed;
            seed = seed * 1664525u + 1013904223u;
        }
    }
    for (uint32_t rows : {1u, 3u}) {
        if (!run_case(rows, 4096u, 32u, false, seed))
            ++failed;
        seed = seed * 1664525u + 1013904223u;
    }

    std::printf("test_dflash2_coarse_topn: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
