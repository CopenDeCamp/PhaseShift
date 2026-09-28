#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/attention.h>
#include <phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rope.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rmsnorm.h>
#include <phaseshift/models/qwen35/kernels/dflash2/swiglu.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
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

static std::optional<ps::kernel::Bf16GemmConfig> pick(
    const ps::weights::MatrixWeight& w,
    uint32_t rows,
    uint32_t input_row_stride,
    const void* input) {
    ps::qwen35::runtime::Bf16GemmSelectorInput sel;
    sel.rows = rows;
    sel.out_features = w.rows;
    sel.k = w.cols;
    sel.input_row_stride = input_row_stride;
    sel.weight_aligned16 = (reinterpret_cast<uintptr_t>(w.bf16.data<void>()) & 15u) == 0u;
    sel.input_aligned16 = (reinterpret_cast<uintptr_t>(input) & 15u) == 0u;
    sel.verify_exact = false;
    return ps::qwen35::runtime::select_bf16_gemm_config(sel);
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

    auto arena_result = ps::gpu::GpuArena::create(0, 8ull * 1024ull * 1024ull * 1024ull);
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

    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = 2048u;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_features = static_cast<uint32_t>(config.attention_q_rows());
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t intermediate = static_cast<uint32_t>(config.intermediate_size);
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    const uint32_t block_rows = 8u;
    const uint32_t context_rows = 8u;
    const uint32_t context_start = 4096u;
    const uint32_t block_start = context_start + context_rows;
    const uint32_t window = static_cast<uint32_t>(config.sliding_window);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ps::gpu::Tensor hidden_in;
    ps::gpu::Tensor target_feature;
    ps::gpu::Tensor output;
    check(alloc_bf16(arena, {block_rows, hidden}, hidden_in), "hidden alloc");
    check(alloc_bf16(arena, {context_rows, hidden}, target_feature), "target alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, output), "out alloc");

    constexpr uint32_t kIters = 50u;
    const double layer_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_forward_layer_stateless(
            ex, 0u, hidden_in.data<bf16_t>(), target_feature.data<bf16_t>(), context_rows,
            context_start, block_start, block_rows, output.data<bf16_t>(), stream);
    });
    const double layer_wall_us = time_wall_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_forward_layer_stateless(
            ex, 0u, hidden_in.data<bf16_t>(), target_feature.data<bf16_t>(), context_rows,
            context_start, block_start, block_rows, output.data<bf16_t>(), stream);
    });

    const auto& layer = weights.layers[0];
    const auto q_cfg = pick(layer.attn_q_proj, block_rows, hidden, ex.attention_prepared.data<void>());
    const auto k_cfg = pick(layer.attn_k_proj, block_rows, hidden, ex.attention_prepared.data<void>());
    const auto o_cfg = pick(layer.attn_o_proj, block_rows, q_features, ex.attention_context.data<void>());
    const auto g_cfg = pick(layer.mlp_gate_proj, block_rows, hidden, ex.mlp_prepared.data<void>());
    const auto d_cfg = pick(layer.mlp_down_proj, block_rows, intermediate, ex.mlp_swiglu.data<void>());
    check(q_cfg && k_cfg && o_cfg && g_cfg && d_cfg, "gemm configs");
    if (!q_cfg || !k_cfg || !o_cfg || !g_cfg || !d_cfg) return 1;

    const double input_norm_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            hidden_in.data<bf16_t>(), layer.input_layernorm_weight.data<bf16_t>(),
            ex.layer_input_norm.data<bf16_t>(), block_rows, hidden, hidden,
            config.rms_norm_eps, stream);
    });
    const double attn_prepare_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_grouped_conv_prepare(
            ex, ex.layer_input_norm.data<bf16_t>(), layer, true, block_rows,
            ex.attention_prepared.data<bf16_t>(), stream);
    });
    const double q_proj_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_gemm_bf16(
            *q_cfg, layer.attn_q_proj.bf16.data<bf16_t>(), ex.attention_prepared.data<bf16_t>(),
            ex.q_raw.data<bf16_t>(), ps::kernel::Bf16GemmOutputDType::BF16, block_rows,
            layer.attn_q_proj.rows, layer.attn_q_proj.cols, hidden, q_features, stream);
    });
    const double k_proj_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_gemm_bf16(
            *k_cfg, layer.attn_k_proj.bf16.data<bf16_t>(), ex.attention_prepared.data<bf16_t>(),
            ex.k_noise_raw.data<bf16_t>(), ps::kernel::Bf16GemmOutputDType::BF16, block_rows,
            layer.attn_k_proj.rows, layer.attn_k_proj.cols, hidden, kv_features, stream);
    });
    const double q_norm_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            ex.q_raw.data<bf16_t>(), layer.attn_q_norm_weight.data<bf16_t>(),
            ex.q_norm.data<bf16_t>(), block_rows, q_features, head_dim, config.rms_norm_eps,
            stream);
    });
    const double rope_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_rope_bf16(
            ex.q_norm.data<bf16_t>(), q_features, ex.q_rope.data<bf16_t>(), q_features,
            block_rows, q_features, head_dim, block_start, config.rope_theta, stream);
    });
    const double attention_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_attention_bf16(
            ex.q_rope.data<bf16_t>(), q_features, ex.k_ctx_rope.data<bf16_t>(),
            ex.v_ctx.data<bf16_t>(), context_rows, kv_features,
            ex.k_noise_rope.data<bf16_t>(), ex.v_noise.data<bf16_t>(), block_rows,
            kv_features, ex.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads,
            head_dim, context_start, block_start, window, scale, stream);
    });
    const double o_proj_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_gemm_bf16(
            *o_cfg, layer.attn_o_proj.bf16.data<bf16_t>(), ex.attention_context.data<bf16_t>(),
            ex.attention_o.data<bf16_t>(), ps::kernel::Bf16GemmOutputDType::BF16, block_rows,
            layer.attn_o_proj.rows, layer.attn_o_proj.cols, q_features, hidden, stream);
    });
    const double attn_finish_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_grouped_conv_finish(
            ex, ex.attention_o.data<bf16_t>(), layer, true, block_rows,
            ex.attention_finished.data<bf16_t>(), stream);
    });
    const double residual_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_residual_add_bf16(
            hidden_in.data<bf16_t>(), hidden, ex.attention_finished.data<bf16_t>(), hidden,
            ex.attention_residual.data<bf16_t>(), hidden, block_rows, hidden, stream);
    });
    const double post_norm_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            ex.attention_residual.data<bf16_t>(), layer.post_attention_layernorm_weight.data<bf16_t>(),
            ex.post_attention_norm.data<bf16_t>(), block_rows, hidden, hidden,
            config.rms_norm_eps, stream);
    });
    const double mlp_prepare_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_grouped_conv_prepare(
            ex, ex.post_attention_norm.data<bf16_t>(), layer, false, block_rows,
            ex.mlp_prepared.data<bf16_t>(), stream);
    });
    const double gate_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_gemm_bf16(
            *g_cfg, layer.mlp_gate_proj.bf16.data<bf16_t>(), ex.mlp_prepared.data<bf16_t>(),
            ex.mlp_gate.data<bf16_t>(), ps::kernel::Bf16GemmOutputDType::BF16, block_rows,
            layer.mlp_gate_proj.rows, layer.mlp_gate_proj.cols, hidden, intermediate, stream);
    });
    const double swiglu_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_swiglu_bf16(
            ex.mlp_gate.data<bf16_t>(), intermediate, ex.mlp_up.data<bf16_t>(), intermediate,
            ex.mlp_swiglu.data<bf16_t>(), intermediate, block_rows, intermediate, stream);
    });
    const double down_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_gemm_bf16(
            *d_cfg, layer.mlp_down_proj.bf16.data<bf16_t>(), ex.mlp_swiglu.data<bf16_t>(),
            ex.mlp_down.data<bf16_t>(), ps::kernel::Bf16GemmOutputDType::BF16, block_rows,
            layer.mlp_down_proj.rows, layer.mlp_down_proj.cols, intermediate, hidden, stream);
    });
    const double mlp_finish_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_grouped_conv_finish(
            ex, ex.mlp_down.data<bf16_t>(), layer, false, block_rows,
            ex.mlp_finished.data<bf16_t>(), stream);
    });

    std::printf(
        "layer0 ctx=%u blk=%u: input_norm=%.2f attn_prepare=%.2f q_proj=%.2f k_proj=%.2f "
        "q_norm=%.2f rope=%.2f attention=%.2f o_proj=%.2f attn_finish=%.2f residual=%.2f "
        "post_norm=%.2f mlp_prepare=%.2f gate=%.2f swiglu=%.2f down=%.2f mlp_finish=%.2f "
        "| composed=%.2fus (wall %.2fus)\n",
        context_rows, block_rows, input_norm_us, attn_prepare_us, q_proj_us, k_proj_us,
        q_norm_us, rope_us, attention_us, o_proj_us, attn_finish_us, residual_us,
        post_norm_us, mlp_prepare_us, gate_us, swiglu_us, down_us, mlp_finish_us, layer_us,
        layer_wall_us);

    const uint32_t long_context = 2048u;
    ps::gpu::Tensor long_k_ctx;
    ps::gpu::Tensor long_v_ctx;
    check(alloc_bf16(arena, {long_context, kv_features}, long_k_ctx), "long k alloc");
    check(alloc_bf16(arena, {long_context, kv_features}, long_v_ctx), "long v alloc");
    const double attention_long_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_attention_bf16(
            ex.q_rope.data<bf16_t>(), q_features, long_k_ctx.data<bf16_t>(),
            long_v_ctx.data<bf16_t>(), long_context, kv_features,
            ex.k_noise_rope.data<bf16_t>(), ex.v_noise.data<bf16_t>(), block_rows,
            kv_features, ex.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads,
            head_dim, 0u, long_context, window, scale, stream);
    });
    std::printf("attention ctx=%u blk=%u: %.2fus\n", long_context, block_rows,
                attention_long_us);

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");

    std::printf("test_dflash2_gate4_perf: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
