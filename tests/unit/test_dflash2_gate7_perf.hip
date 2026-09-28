#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/attention.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::f32_to_bf16_rne;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2ContextState;
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
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
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

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (model_dir == nullptr || model_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2 and _TARGET\n");
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

    auto arena_result = ps::gpu::GpuArena::create(0, 24ull * 1024ull * 1024ull * 1024ull);
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

    auto target_result = ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(target_result.ok(), "target load");
    if (!target_result.ok()) return 1;
    ps::qwen35::Qwen35Model target = target_result.release();

    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    exec_config.target_lm_head = &target.weights().lm_head;
    auto exec_result =
        ps::qwen35::dflash2::create_dflash2_executor(config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t q_features = static_cast<uint32_t>(config.attention_q_rows());
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const uint32_t capacity = static_cast<uint32_t>(config.sliding_window);
    const uint32_t block_rows = 8u;
    constexpr uint32_t kIters = 20u;

    ps::gpu::Tensor feature;
    ps::gpu::Tensor noise;
    check(alloc_bf16(arena, {capacity, hidden}, feature), "feature alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, noise), "noise alloc");
    {
        std::vector<bf16_t> host(static_cast<std::size_t>(capacity) * hidden);
        uint32_t seed = 24681357u;
        for (bf16_t& x : host) {
            seed = seed * 1664525u + 1013904223u;
            const float u = static_cast<float>(seed >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.25f)};
        }
        check(hipMemcpy(feature.data<bf16_t>(), host.data(), host.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "feature upload");
        std::vector<bf16_t> host_noise(static_cast<std::size_t>(block_rows) * hidden);
        for (bf16_t& x : host_noise) {
            seed = seed * 1664525u + 1013904223u;
            const float u = static_cast<float>(seed >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.25f)};
        }
        check(hipMemcpy(noise.data<bf16_t>(), host_noise.data(),
                        host_noise.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
              "noise upload");
    }

    auto state_result = ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
    check(state_result.ok(), "context create");
    if (!state_result.ok()) return 1;
    DFlash2ContextState state = state_result.release();

    std::printf("dflash2 gate7 perf (device 0)\n");

    std::printf("append (target feature -> all 5 layer K/V + scatter)\n");
    for (uint32_t rows : {1u, 8u, 128u, capacity}) {
        const double us = time_events_us(stream, kIters, [&]() {
            (void)ps::qwen35::dflash2::dflash2_context_reset(state, 0u);
            (void)ps::qwen35::dflash2::dflash2_context_append(
                ex, state, feature.data<bf16_t>(), rows, 0u, stream);
        });
        std::printf("  rows=%4u : %9.2f us\n", rows, us);
    }

    {
        check(ps::qwen35::dflash2::dflash2_context_reset(state, 0u).ok(), "timing reset");
        (void)ps::qwen35::dflash2::dflash2_context_append(
            ex, state, feature.data<bf16_t>(), capacity, 0u, stream);
        check(hipStreamSynchronize(stream) == hipSuccess, "ring fill sync");
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const bf16_t* k_ring0 = state.k_ring.data<bf16_t>();
    const bf16_t* v_ring0 = state.v_ring.data<bf16_t>();

    std::printf("ring attention kernel (block_rows=8)\n");
    for (uint32_t context : {8u, 512u, 2047u, 2048u}) {
        const double us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_dflash2_attention_ring_bf16(
                ex.q_rope.data<bf16_t>(), q_features, k_ring0, v_ring0, capacity, context, 0u,
                kv_features, ex.k_noise_rope.data<bf16_t>(), ex.v_noise.data<bf16_t>(),
                block_rows, kv_features, ex.attention_context.data<bf16_t>(), q_features,
                static_cast<uint32_t>(config.num_attention_heads),
                static_cast<uint32_t>(config.num_key_value_heads), head_dim, context, capacity,
                scale, stream);
        });
        std::printf("  context=%4u block=%u : %9.2f us\n", context, block_rows, us);
    }

    std::printf("cached backbone (block_rows=8, final norm)\n");
    for (uint32_t context : {8u, 512u, 2048u}) {
        check(ps::qwen35::dflash2::dflash2_context_reset(state, 0u).ok(), "reset");
        (void)ps::qwen35::dflash2::dflash2_context_append(
            ex, state, feature.data<bf16_t>(), context, 0u, stream);
        check(hipStreamSynchronize(stream) == hipSuccess, "fill sync");
        const uint32_t block_start = context;
        const double us = time_events_us(stream, kIters, [&]() {
            (void)ps::qwen35::dflash2::dflash2_forward_backbone_cached(
                ex, state, noise.data<bf16_t>(), block_start, block_rows,
                ex.backbone_output.data<bf16_t>(), stream);
        });
        std::printf("  context=%4u : %9.2f us\n", context, us);
    }

    std::printf("full cached proposer (backbone + lm_head + top16 + selector)\n");
    for (uint32_t context : {8u, 2048u}) {
        check(ps::qwen35::dflash2::dflash2_context_reset(state, 0u).ok(), "reset");
        (void)ps::qwen35::dflash2::dflash2_context_append(
            ex, state, feature.data<bf16_t>(), context, 0u, stream);
        check(hipStreamSynchronize(stream) == hipSuccess, "fill sync");
        const uint32_t block_start = context;
        const double us = time_events_us(stream, kIters, [&]() {
            (void)ps::qwen35::dflash2::dflash2_forward_and_select_cached(
                ex, state, noise.data<bf16_t>(), block_start, block_rows, 0,
                ex.proposal_tokens.data<int32_t>(), stream);
        });
        std::printf("  context=%4u : %9.2f us\n", context, us);
    }

    std::printf("stateless vs cached backbone (context=2048, block_rows=8)\n");
    {
        check(ps::qwen35::dflash2::dflash2_context_reset(state, 0u).ok(), "reset");
        (void)ps::qwen35::dflash2::dflash2_context_append(
            ex, state, feature.data<bf16_t>(), capacity, 0u, stream);
        check(hipStreamSynchronize(stream) == hipSuccess, "fill sync");
        const double stateless_us = time_events_us(stream, kIters, [&]() {
            (void)ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
                ex, noise.data<bf16_t>(), feature.data<bf16_t>(), capacity, 0u, capacity,
                block_rows, ex.backbone_output.data<bf16_t>(), stream);
        });
        const double cached_us = time_events_us(stream, kIters, [&]() {
            (void)ps::qwen35::dflash2::dflash2_forward_backbone_cached(
                ex, state, noise.data<bf16_t>(), capacity, block_rows,
                ex.backbone_output.data<bf16_t>(), stream);
        });
        std::printf("  stateless ctx2048 : %9.2f us\n", stateless_us);
        std::printf("  cached    ctx2048 : %9.2f us\n", cached_us);
    }

    check(ps::qwen35::dflash2::dflash2_context_shutdown(state).ok(), "context shutdown");
    check(ps::qwen35::dflash2::dflash2_executor_shutdown(ex).ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate7_perf: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
