#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

struct Expect {
    int32_t best_id;
    float best_logit;
    uint8_t certified;
};

void run_case(const char* label, const std::vector<std::vector<float>>& logits,
              const std::vector<std::vector<int32_t>>& ids, const std::vector<float>& omitted,
              float abs_margin, float rel_margin, const std::vector<Expect>& expect) {
    const uint32_t rows = static_cast<uint32_t>(logits.size());
    uint32_t pool = 0u;
    for (const auto& row : logits)
        pool = std::max<uint32_t>(pool, static_cast<uint32_t>(row.size()));

    std::vector<float> cl(static_cast<std::size_t>(rows) * pool, -INFINITY);
    std::vector<int32_t> ci(static_cast<std::size_t>(rows) * pool, 0);
    for (uint32_t r = 0u; r < rows; ++r) {
        for (uint32_t c = 0u; c < logits[r].size(); ++c) {
            cl[static_cast<std::size_t>(r) * pool + c] = logits[r][c];
            ci[static_cast<std::size_t>(r) * pool + c] = ids[r][c];
        }
    }

    float* dev_logits = nullptr;
    int32_t* dev_ids = nullptr;
    float* dev_omitted = nullptr;
    int32_t* dev_best_ids = nullptr;
    float* dev_best_logits = nullptr;
    float* dev_gaps = nullptr;
    uint8_t* dev_certified = nullptr;
    (void)hipMalloc(&dev_logits, cl.size() * sizeof(float));
    (void)hipMalloc(&dev_ids, ci.size() * sizeof(int32_t));
    (void)hipMalloc(&dev_omitted, rows * sizeof(float));
    (void)hipMalloc(&dev_best_ids, rows * sizeof(int32_t));
    (void)hipMalloc(&dev_best_logits, rows * sizeof(float));
    (void)hipMalloc(&dev_gaps, rows * sizeof(float));
    (void)hipMalloc(&dev_certified, rows);
    (void)hipMemcpy(dev_logits, cl.data(), cl.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_ids, ci.data(), ci.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(dev_omitted, omitted.data(), rows * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemset(dev_certified, 0, rows);

    const hipError_t err = ps::kernel::launch_dflash2_target_certified_argmax(
        dev_logits, dev_ids, dev_omitted, 1u, abs_margin, rel_margin, dev_best_ids,
        dev_best_logits, dev_gaps, dev_certified, rows, pool, nullptr);
    check(err == hipSuccess, label);
    (void)hipDeviceSynchronize();

    std::vector<int32_t> best_ids(rows, 0);
    std::vector<float> best_logits(rows, 0.0f);
    std::vector<float> gaps(rows, 0.0f);
    std::vector<uint8_t> certified(rows, 0u);
    (void)hipMemcpy(best_ids.data(), dev_best_ids, rows * sizeof(int32_t),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(best_logits.data(), dev_best_logits, rows * sizeof(float),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(gaps.data(), dev_gaps, rows * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(certified.data(), dev_certified, rows, hipMemcpyDeviceToHost);

    for (uint32_t r = 0u; r < rows; ++r) {
        char tag[192];
        std::snprintf(tag, sizeof(tag), "%s row=%u best_id", label, r);
        check(best_ids[r] == expect[r].best_id, tag);
        std::snprintf(tag, sizeof(tag), "%s row=%u best_logit", label, r);
        check(std::fabs(best_logits[r] - expect[r].best_logit) <=
                  1e-5f * (1.0f + std::fabs(expect[r].best_logit)),
              tag);
        std::snprintf(tag, sizeof(tag), "%s row=%u certified", label, r);
        check(certified[r] == expect[r].certified, tag);
    }

    (void)hipFree(dev_logits);
    (void)hipFree(dev_ids);
    (void)hipFree(dev_omitted);
    (void)hipFree(dev_best_ids);
    (void)hipFree(dev_best_logits);
    (void)hipFree(dev_gaps);
    (void)hipFree(dev_certified);
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

    run_case("A clear pass", {{5.0f, 1.0f, 2.0f}}, {{10, 11, 12}}, {0.5f}, 0.0f, 0.0f,
             {{10, 5.0f, 1u}});

    run_case("B candidate outside dominates", {{5.0f, 4.9f}}, {{10, 11}}, {5.1f}, 0.0f, 0.0f,
             {{10, 5.0f, 0u}});

    run_case("C nearly tied", {{5.0f}}, {{7}}, {4.9999f}, 1e-3f, 1e-5f, {{7, 5.0f, 0u}});

    run_case("D exact tie", {{5.0f, 5.0f}}, {{10, 3}}, {5.0f}, 0.0f, 0.0f, {{3, 5.0f, 0u}});

    run_case("E zero activation", {{0.0f, 0.0f, 0.0f}}, {{0, 1, 2}}, {0.0f}, 0.0f, 0.0f,
             {{0, 0.0f, 0u}});

    run_case("F relative margin fail", {{1.0e6f}}, {{42}}, {1.0e6f - 50.0f}, 0.0f, 1e-4f,
             {{42, 1.0e6f, 0u}});
    run_case("F relative margin pass", {{1.0e6f}}, {{42}}, {1.0e6f - 200.0f}, 0.0f, 1e-4f,
             {{42, 1.0e6f, 1u}});

    {
        std::vector<std::vector<float>> logits(5);
        std::vector<std::vector<int32_t>> ids(5);
        std::vector<float> omitted(5);
        std::vector<Expect> expect(5);
        for (uint32_t r = 0u; r < 5u; ++r) {
            logits[r].resize(33);
            ids[r].resize(33);
            for (uint32_t c = 0u; c < 33u; ++c) {
                logits[r][c] = 10.0f - 0.01f * static_cast<float>(c) - 0.001f * r;
                ids[r][c] = static_cast<int32_t>(100u + c * 5u + r);
            }
            omitted[r] = 9.0f;
            expect[r] = {static_cast<int32_t>(100u + r), logits[r][0], 1u};
        }
        run_case("G non tile pool33", logits, ids, omitted, 1e-3f, 1e-5f, expect);
    }

    std::printf("test_target_lm_proxy_certificate: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
