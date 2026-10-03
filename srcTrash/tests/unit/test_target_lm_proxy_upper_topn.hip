#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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

struct Rng {
    uint32_t state = 0x1234567u;
    uint32_t next_u32() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    float unit() { return static_cast<float>(next_u32()) / static_cast<float>(1u << 24); }
    float sym() { return unit() * 2.0f - 1.0f; }
};

struct Case {
    uint32_t rows;
    uint32_t vocab;
    uint32_t pool;
};

void run_case(const Case& c, uint32_t seed, float margin) {
    Rng rng;
    rng.state = seed;

    const std::size_t elems = static_cast<std::size_t>(c.rows) * c.vocab;
    std::vector<float> coarse(elems, 0.0f);
    std::vector<float> act_l2(c.rows, 0.0f);
    std::vector<float> err_l2(c.vocab, 0.0f);
    for (std::size_t i = 0u; i < elems; ++i)
        coarse[i] = 3.0f * rng.sym();
    for (uint32_t r = 0u; r < c.rows; ++r)
        act_l2[r] = 0.2f + 2.0f * rng.unit();
    for (uint32_t v = 0u; v < c.vocab; ++v)
        err_l2[v] = 0.05f * rng.unit();

    std::vector<float> expect_upper(elems, 0.0f);
    for (uint32_t r = 0u; r < c.rows; ++r) {
        for (uint32_t v = 0u; v < c.vocab; ++v) {
            expect_upper[static_cast<std::size_t>(r) * c.vocab + v] =
                coarse[static_cast<std::size_t>(r) * c.vocab + v] + act_l2[r] * err_l2[v];
        }
    }

    float* dev_coarse = nullptr;
    float* dev_act_l2 = nullptr;
    float* dev_err_l2 = nullptr;
    float* dev_upper = nullptr;
    int32_t* dev_ids = nullptr;
    float* dev_values = nullptr;
    int32_t* dev_scratch_ids = nullptr;
    float* dev_scratch_values = nullptr;

    const uint32_t partitions = ps::kernel::dflash2_int2_default_partitions(c.vocab);
    const uint32_t scratch_stride = partitions * ps::kernel::kDflash2Int2MaxPool;
    const uint32_t topn_pool = c.pool + 1u;

    const bool alloc_ok =
        hipMalloc(&dev_coarse, elems * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev_act_l2, c.rows * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev_err_l2, c.vocab * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev_upper, elems * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev_ids, static_cast<std::size_t>(c.rows) * topn_pool * sizeof(int32_t)) ==
            hipSuccess &&
        hipMalloc(&dev_values,
                  static_cast<std::size_t>(c.rows) * topn_pool * sizeof(float)) == hipSuccess &&
        hipMalloc(&dev_scratch_ids,
                  static_cast<std::size_t>(c.rows) * scratch_stride * sizeof(int32_t)) ==
            hipSuccess &&
        hipMalloc(&dev_scratch_values,
                  static_cast<std::size_t>(c.rows) * scratch_stride * sizeof(float)) ==
            hipSuccess;
    check(alloc_ok, "upper topn allocation");
    if (!alloc_ok) {
        return;
    }

    (void)hipMemcpy(dev_coarse, coarse.data(), elems * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_act_l2, act_l2.data(), c.rows * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_err_l2, err_l2.data(), c.vocab * sizeof(float), hipMemcpyHostToDevice);

    hipError_t err = ps::kernel::launch_dflash2_target_upper_logits(
        dev_coarse, dev_act_l2, dev_err_l2, dev_upper, c.rows, c.vocab, nullptr);
    check(err == hipSuccess, "upper logits launch");
    err = ps::kernel::launch_dflash2_coarse_topn(
        dev_upper, c.rows, c.vocab, c.vocab, topn_pool, dev_ids, dev_values,
        dev_scratch_ids, dev_scratch_values, scratch_stride, partitions, nullptr);
    check(err == hipSuccess, "upper topn launch");
    (void)hipDeviceSynchronize();

    std::vector<float> gpu_upper(elems, 0.0f);
    (void)hipMemcpy(gpu_upper.data(), dev_upper, elems * sizeof(float),
                    hipMemcpyDeviceToHost);

    bool upper_match = true;
    for (std::size_t i = 0u; i < elems; ++i) {
        if (std::fabs(gpu_upper[i] - expect_upper[i]) > 1e-5f * (1.0f + std::fabs(expect_upper[i])))
            upper_match = false;
    }
    check(upper_match, "upper elementwise matches cpu");

    std::vector<int32_t> gpu_ids(static_cast<std::size_t>(c.rows) * topn_pool, 0);
    std::vector<float> gpu_values(static_cast<std::size_t>(c.rows) * topn_pool, 0.0f);
    (void)hipMemcpy(gpu_ids.data(), dev_ids, gpu_ids.size() * sizeof(int32_t),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(gpu_values.data(), dev_values, gpu_values.size() * sizeof(float),
                    hipMemcpyDeviceToHost);

    bool topn_match = true;
    for (uint32_t r = 0u; r < c.rows && topn_match; ++r) {
        const float* row = gpu_upper.data() + static_cast<std::size_t>(r) * c.vocab;
        std::vector<int32_t> order(c.vocab);
        for (uint32_t i = 0u; i < c.vocab; ++i)
            order[i] = static_cast<int32_t>(i);
        std::stable_sort(order.begin(), order.end(), [row](int32_t a, int32_t b) {
            if (row[a] != row[b])
                return row[a] > row[b];
            return a < b;
        });
        for (uint32_t i = 0u; i < topn_pool; ++i) {
            const int32_t want_id = order[i];
            const float want_value = row[want_id];
            const int32_t got_id = gpu_ids[static_cast<std::size_t>(r) * topn_pool + i];
            const float got_value = gpu_values[static_cast<std::size_t>(r) * topn_pool + i];
            if (want_id != got_id ||
                std::fabs(want_value - got_value) > 1e-5f * (1.0f + std::fabs(want_value))) {
                topn_match = false;
                std::printf("FAIL: topn r=%u i=%u want=(%d,%.6f) got=(%d,%.6f)\n", r, i,
                            want_id, want_value, got_id, got_value);
                break;
            }
        }
    }
    check(topn_match, "top-(n+1) matches cpu oracle");

    (void)margin;
    (void)hipFree(dev_coarse);
    (void)hipFree(dev_act_l2);
    (void)hipFree(dev_err_l2);
    (void)hipFree(dev_upper);
    (void)hipFree(dev_ids);
    (void)hipFree(dev_values);
    (void)hipFree(dev_scratch_ids);
    (void)hipFree(dev_scratch_values);
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
        {1u, 32u, 4u},   {1u, 50u, 7u},   {3u, 64u, 8u},
        {7u, 48u, 16u},  {2u, 129u, 32u}, {1u, 17u, 16u},
    };
    uint32_t seed = 0x1234abcdu;
    for (const Case& c : cases) {
        run_case(c, seed, 1e-4f);
        seed = seed * 1664525u + 1013904223u;
    }

    std::printf("test_target_lm_proxy_upper_topn: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
