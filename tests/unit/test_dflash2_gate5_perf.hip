#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rmsnorm.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2Executor;
using ps::qwen35::dflash2::DFlash2ExecutorConfig;
using ps::qwen35::dflash2::DFlash2Weights;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static double time_events_us(hipStream_t stream, uint32_t iters, const std::function<void()>& fn) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
        return -1.0;
    }
    for (uint32_t i = 0; i < 5u; ++i) fn();
    (void)hipEventRecord(start, stream);
    for (uint32_t i = 0; i < iters; ++i) fn();
    (void)hipEventRecord(stop, stream);
    (void)hipEventSynchronize(stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return static_cast<double>(ms) * 1000.0 / static_cast<double>(iters);
}

static double time_wall_us(hipStream_t stream, uint32_t iters, const std::function<void()>& fn) {
    for (uint32_t i = 0; i < 5u; ++i) fn();
    (void)hipStreamSynchronize(stream);
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iters; ++i) fn();
    (void)hipStreamSynchronize(stream);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 /
           static_cast<double>(iters);
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    if (model_dir == nullptr || model_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();

    ps::resident::ResidentTestRequest resident_request;
    resident_request.dflash2_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    auto resident_result = ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    resident_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture resident = resident_result.release();

    auto arena_result = ps::gpu::GpuArena::create(0, resident.arena_bytes());
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

    auto weights_result = resident.enabled()
        ? resident.take_dflash2(nullptr)
        : ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = 8u;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t block_rows = 8u;
    const uint32_t context_rows = 8u;
    const uint32_t context_start = 4096u;
    const uint32_t block_start = context_start + context_rows;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);

    ps::gpu::Tensor noise;
    ps::gpu::Tensor feature;
    ps::gpu::Tensor out;
    check(alloc_bf16(arena, {block_rows, hidden}, noise), "noise alloc");
    check(alloc_bf16(arena, {context_rows, hidden}, feature), "feature alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, out), "out alloc");

    constexpr uint32_t kIters = 20u;
    double layer_us[5] = {};
    const bf16_t* src = noise.data<bf16_t>();
    for (uint32_t li = 0; li < 5u; ++li) {
        bf16_t* dst = (li % 2u == 0u) ? ex.backbone_hidden_a.data<bf16_t>()
                                      : ex.backbone_hidden_b.data<bf16_t>();
        const bf16_t* layer_src = src;
        layer_us[li] = time_events_us(stream, kIters, [&]() {
            (void)ps::qwen35::dflash2::dflash2_forward_layer_stateless(
                ex, li, layer_src, feature.data<bf16_t>(), context_rows, context_start,
                block_start, block_rows, dst, stream);
        });
        src = dst;
    }
    const double final_norm_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            src, weights.final_norm_weight.data<bf16_t>(), out.data<bf16_t>(), block_rows,
            hidden, hidden, config.rms_norm_eps, stream);
    });
    const double backbone_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
            ex, noise.data<bf16_t>(), feature.data<bf16_t>(), context_rows, context_start,
            block_start, block_rows, out.data<bf16_t>(), stream);
    });
    const double backbone_wall_us = time_wall_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
            ex, noise.data<bf16_t>(), feature.data<bf16_t>(), context_rows, context_start,
            block_start, block_rows, out.data<bf16_t>(), stream);
    });

    std::printf(
        "backbone ctx=%u blk=%u: layer0=%.2f layer1=%.2f layer2=%.2f layer3=%.2f layer4=%.2f "
        "final_norm=%.2f | full_backbone=%.2fus (wall %.2fus)\n",
        context_rows, block_rows, layer_us[0], layer_us[1], layer_us[2], layer_us[3],
        layer_us[4], final_norm_us, backbone_us, backbone_wall_us);

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");

    std::printf("test_dflash2_gate5_perf: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
