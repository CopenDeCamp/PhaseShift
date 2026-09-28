#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static bool alloc_f32(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(float), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<float>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_i32(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(int32_t), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<int32_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

struct Expected {
    std::vector<int32_t> ids;
    std::vector<float> logits;
};

static Expected oracle(
    const std::vector<float>& logits, uint32_t rows, uint32_t vocab, uint32_t k) {
    Expected out;
    out.ids.resize(static_cast<std::size_t>(rows) * k);
    out.logits.resize(static_cast<std::size_t>(rows) * k);
    std::vector<std::pair<float, int32_t>> items(vocab);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t v = 0; v < vocab; ++v) {
            items[v] = {logits[static_cast<std::size_t>(r) * vocab + v],
                        static_cast<int32_t>(v)};
        }
        std::stable_sort(items.begin(), items.end(),
                         [](const std::pair<float, int32_t>& a,
                            const std::pair<float, int32_t>& b) {
                             if (a.first != b.first) return a.first > b.first;
                             return a.second < b.second;
                         });
        for (uint32_t j = 0; j < k; ++j) {
            out.logits[static_cast<std::size_t>(r) * k + j] = items[j].first;
            out.ids[static_cast<std::size_t>(r) * k + j] = items[j].second;
        }
    }
    return out;
}

static double time_median_us(hipStream_t stream, uint32_t iters, const std::function<void()>& fn) {
    for (uint32_t i = 0; i < 20u; ++i) fn();
    std::vector<hipEvent_t> starts(iters);
    std::vector<hipEvent_t> stops(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        (void)hipEventCreate(&starts[i]);
        (void)hipEventCreate(&stops[i]);
        (void)hipEventRecord(starts[i], stream);
        fn();
        (void)hipEventRecord(stops[i], stream);
    }
    (void)hipStreamSynchronize(stream);
    std::vector<double> xs(iters, 0.0);
    for (uint32_t i = 0; i < iters; ++i) {
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, starts[i], stops[i]);
        xs[i] = static_cast<double>(ms) * 1000.0;
        (void)hipEventDestroy(starts[i]);
        (void)hipEventDestroy(stops[i]);
    }
    std::sort(xs.begin(), xs.end());
    return xs[iters / 2u];
}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    const uint32_t k = 16u;
    const uint32_t vocab = 248320u;
    const uint32_t max_rows = 7u;

    ps::gpu::Tensor logits;
    ps::gpu::Tensor ref_ids;
    ps::gpu::Tensor ref_logits;
    ps::gpu::Tensor opt_ids;
    ps::gpu::Tensor opt_logits;
    ps::gpu::Tensor scratch_ids;
    ps::gpu::Tensor scratch_logits;
    const bool alloc_ok =
        alloc_f32(arena, {max_rows, vocab}, logits) &&
        alloc_i32(arena, {max_rows, k}, ref_ids) &&
        alloc_f32(arena, {max_rows, k}, ref_logits) &&
        alloc_i32(arena, {max_rows, k}, opt_ids) &&
        alloc_f32(arena, {max_rows, k}, opt_logits) &&
        alloc_i32(arena, {max_rows, ps::kernel::kDflash2TopKScratchPerRow}, scratch_ids) &&
        alloc_f32(arena, {max_rows, ps::kernel::kDflash2TopKScratchPerRow}, scratch_logits);
    check(alloc_ok, "alloc");
    if (!alloc_ok) return 1;

    std::vector<ps::kernel::DFlash2TopKPlan> plans;
    for (uint32_t p : {8u, 16u, 32u, 64u}) {
        for (uint32_t th : {64u, 128u, 256u}) {
            plans.push_back(ps::kernel::DFlash2TopKPlan{p, th});
        }
    }

    struct Fixture {
        std::string name;
        std::vector<float> data;
        uint32_t rows;
    };
    std::vector<Fixture> fixtures;
    {
        Fixture f;
        f.name = "random";
        f.rows = max_rows;
        f.data.resize(static_cast<std::size_t>(max_rows) * vocab);
        uint32_t seed = 0x12345678u;
        for (float& x : f.data) {
            seed = seed * 1664525u + 1013904223u;
            x = (static_cast<float>(seed >> 8) / static_cast<float>(1u << 24)) * 20.0f - 10.0f;
        }
        fixtures.push_back(f);
    }
    {
        Fixture f;
        f.name = "ties";
        f.rows = max_rows;
        f.data.resize(static_cast<std::size_t>(max_rows) * vocab);
        for (uint32_t r = 0; r < max_rows; ++r) {
            for (uint32_t v = 0; v < vocab; ++v) {
                f.data[static_cast<std::size_t>(r) * vocab + v] =
                    static_cast<float>(static_cast<int32_t>(v % 13u) - 6) * 0.5f;
            }
        }
        fixtures.push_back(f);
    }
    {
        Fixture f;
        f.name = "boundary";
        f.rows = max_rows;
        f.data.assign(static_cast<std::size_t>(max_rows) * vocab, -100.0f);
        for (uint32_t part = 0; part < ps::kernel::kDflash2TopKMaxPartitions; ++part) {
            const uint32_t start = (part * vocab) / ps::kernel::kDflash2TopKMaxPartitions;
            for (uint32_t j = 0; j < k && start + j < vocab; ++j) {
                for (uint32_t r = 0; r < max_rows; ++r) {
                    f.data[static_cast<std::size_t>(r) * vocab + start + j] =
                        static_cast<float>(part) + static_cast<float>(j) * 0.01f;
                }
            }
        }
        fixtures.push_back(f);
    }
    {
        Fixture f;
        f.name = "masked";
        f.rows = max_rows;
        f.data.resize(static_cast<std::size_t>(max_rows) * vocab);
        uint32_t seed = 0x9e3779b9u;
        for (std::size_t i = 0; i < f.data.size(); ++i) {
            seed = seed * 1664525u + 1013904223u;
            if ((seed >> 8) % 5u == 0u) {
                f.data[i] = -INFINITY;
            } else {
                f.data[i] = (static_cast<float>(seed >> 8) / static_cast<float>(1u << 24)) * 5.0f;
            }
        }
        fixtures.push_back(f);
    }

    std::uint64_t plan_checks = 0u;
    std::uint64_t plan_failures = 0u;
    for (const Fixture& fx : fixtures) {
        check(hipMemcpy(logits.data<float>(), fx.data.data(),
                        static_cast<std::size_t>(fx.rows) * vocab * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              fx.name + " upload");
        const Expected want = oracle(fx.data, fx.rows, vocab, k);
        for (const auto& plan : plans) {
            const hipError_t err = ps::kernel::launch_dflash2_topk_f32_optimized(
                logits.data<float>(), fx.rows, vocab, vocab, k, opt_ids.data<int32_t>(),
                opt_logits.data<float>(), scratch_ids.data<int32_t>(),
                scratch_logits.data<float>(), plan, stream);
            check(err == hipSuccess, fx.name + " optimized launch");
            (void)hipStreamSynchronize(stream);
            std::vector<int32_t> ids(static_cast<std::size_t>(fx.rows) * k);
            std::vector<float> lg(static_cast<std::size_t>(fx.rows) * k);
            (void)hipMemcpy(ids.data(), opt_ids.data<int32_t>(), ids.size() * sizeof(int32_t),
                            hipMemcpyDeviceToHost);
            (void)hipMemcpy(lg.data(), opt_logits.data<float>(), lg.size() * sizeof(float),
                            hipMemcpyDeviceToHost);
            std::size_t mismatch = 0u;
            for (std::size_t i = 0; i < want.ids.size(); ++i) {
                if (ids[i] != want.ids[i]) ++mismatch;
                if (lg[i] != want.logits[i]) ++mismatch;
            }
            ++plan_checks;
            if (mismatch != 0u) {
                ++plan_failures;
                std::printf("FAIL %s p=%u th=%u mismatch=%zu\n", fx.name.c_str(),
                            plan.partitions, plan.threads, mismatch);
            }
        }
    }
    check(plan_failures == 0u, "optimized exactness across plans");
    std::printf("GATE11B_CORRECTNESS fixtures=%zu plans=%zu checks=%llu failures=%llu\n",
                fixtures.size(), plans.size(),
                static_cast<unsigned long long>(plan_checks),
                static_cast<unsigned long long>(plan_failures));

    {
        const Fixture& fx = fixtures[0];
        (void)hipMemcpy(logits.data<float>(), fx.data.data(),
                        static_cast<std::size_t>(fx.rows) * vocab * sizeof(float),
                        hipMemcpyHostToDevice);
        const auto plan = ps::kernel::dflash2_topk_default_plan();
        std::vector<int32_t> first(static_cast<std::size_t>(fx.rows) * k);
        (void)ps::kernel::launch_dflash2_topk_f32_optimized(
            logits.data<float>(), fx.rows, vocab, vocab, k, opt_ids.data<int32_t>(),
            opt_logits.data<float>(), scratch_ids.data<int32_t>(),
            scratch_logits.data<float>(), plan, stream);
        (void)hipStreamSynchronize(stream);
        (void)hipMemcpy(first.data(), opt_ids.data<int32_t>(), first.size() * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        std::size_t det_mismatch = 0u;
        for (uint32_t rep = 0; rep < 100u; ++rep) {
            (void)ps::kernel::launch_dflash2_topk_f32_optimized(
                logits.data<float>(), fx.rows, vocab, vocab, k, opt_ids.data<int32_t>(),
                opt_logits.data<float>(), scratch_ids.data<int32_t>(),
                scratch_logits.data<float>(), plan, stream);
            (void)hipStreamSynchronize(stream);
            std::vector<int32_t> ids(first.size());
            (void)hipMemcpy(ids.data(), opt_ids.data<int32_t>(), ids.size() * sizeof(int32_t),
                            hipMemcpyDeviceToHost);
            if (ids != first) ++det_mismatch;
        }
        check(det_mismatch == 0u, "deterministic 100/100");
        std::printf("GATE11B_DETERMINISTIC reps=100 mismatches=%zu\n", det_mismatch);
    }

    {
        const Fixture& fx = fixtures[0];
        (void)hipMemcpy(logits.data<float>(), fx.data.data(),
                        static_cast<std::size_t>(fx.rows) * vocab * sizeof(float),
                        hipMemcpyHostToDevice);
        const double ref_us = time_median_us(stream, 200u, [&]() {
            (void)ps::kernel::launch_dflash2_topk_f32(
                logits.data<float>(), fx.rows, vocab, vocab, k, ref_ids.data<int32_t>(),
                ref_logits.data<float>(), scratch_ids.data<int32_t>(),
                scratch_logits.data<float>(), stream);
        });
        std::printf("GATE11B_PERF impl=reference rows=%u partitions=0 threads=0 us=%.2f\n",
                    fx.rows, ref_us);
        for (const auto& plan : plans) {
            const double us = time_median_us(stream, 200u, [&]() {
                (void)ps::kernel::launch_dflash2_topk_f32_optimized(
                    logits.data<float>(), fx.rows, vocab, vocab, k, opt_ids.data<int32_t>(),
                    opt_logits.data<float>(), scratch_ids.data<int32_t>(),
                    scratch_logits.data<float>(), plan, stream);
            });
            std::printf(
                "GATE11B_PERF impl=optimized rows=%u partitions=%u threads=%u us=%.2f\n",
                fx.rows, plan.partitions, plan.threads, us);
        }
    }

    for (uint32_t rows : {1u, 3u}) {
        const Fixture& fx = fixtures[0];
        (void)hipMemcpy(logits.data<float>(), fx.data.data(),
                        static_cast<std::size_t>(fx.rows) * vocab * sizeof(float),
                        hipMemcpyHostToDevice);
        const double ref_us = time_median_us(stream, 200u, [&]() {
            (void)ps::kernel::launch_dflash2_topk_f32(
                logits.data<float>(), rows, vocab, vocab, k, ref_ids.data<int32_t>(),
                ref_logits.data<float>(), scratch_ids.data<int32_t>(),
                scratch_logits.data<float>(), stream);
        });
        std::printf("GATE11B_PERF impl=reference rows=%u partitions=0 threads=0 us=%.2f\n",
                    rows, ref_us);
        for (const auto& plan : plans) {
            const double us = time_median_us(stream, 200u, [&]() {
                (void)ps::kernel::launch_dflash2_topk_f32_optimized(
                    logits.data<float>(), rows, vocab, vocab, k, opt_ids.data<int32_t>(),
                    opt_logits.data<float>(), scratch_ids.data<int32_t>(),
                    scratch_logits.data<float>(), plan, stream);
            });
            std::printf(
                "GATE11B_PERF impl=optimized rows=%u partitions=%u threads=%u us=%.2f\n",
                rows, plan.partitions, plan.threads, us);
        }
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate11b_topk: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
