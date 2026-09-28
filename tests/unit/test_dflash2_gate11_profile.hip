#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/attention.h>
#include <phaseshift/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <phaseshift/models/qwen35/kernels/dflash2/feature_concat.h>
#include <phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.h>
#include <phaseshift/models/qwen35/kernels/dflash2/noise_input.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rmsnorm.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rope.h>
#include <phaseshift/models/qwen35/kernels/dflash2/swiglu.h>
#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2ContextState;
using ps::qwen35::dflash2::DFlash2Executor;
using ps::qwen35::dflash2::DFlash2ExecutorConfig;
using ps::qwen35::dflash2::DFlash2Weights;

static int g_fail = 0;
static constexpr uint32_t kWarmup = 50u;
static constexpr uint32_t kIters = 200u;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

struct Stat {
    double median = 0.0;
    double p10 = 0.0;
    double p90 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static Stat time_stats(hipStream_t stream, const std::function<void()>& fn) {
    for (uint32_t i = 0; i < kWarmup; ++i) fn();
    std::vector<hipEvent_t> starts(kIters);
    std::vector<hipEvent_t> stops(kIters);
    for (uint32_t i = 0; i < kIters; ++i) {
        if (hipEventCreate(&starts[i]) != hipSuccess ||
            hipEventCreate(&stops[i]) != hipSuccess) {
            return Stat{};
        }
        (void)hipEventRecord(starts[i], stream);
        fn();
        (void)hipEventRecord(stops[i], stream);
    }
    if (hipStreamSynchronize(stream) != hipSuccess) return Stat{};
    std::vector<double> xs(kIters, 0.0);
    for (uint32_t i = 0; i < kIters; ++i) {
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, starts[i], stops[i]);
        xs[i] = static_cast<double>(ms) * 1000.0;
        (void)hipEventDestroy(starts[i]);
        (void)hipEventDestroy(stops[i]);
    }
    std::sort(xs.begin(), xs.end());
    Stat s;
    s.min = xs.front();
    s.max = xs.back();
    s.median = xs[kIters / 2u];
    s.p10 = xs[kIters / 10u];
    s.p90 = xs[(kIters * 9u) / 10u];
    return s;
}

static void emit(const char* stage, int layer, uint32_t rows, uint32_t ctx, const Stat& s,
                 uint64_t bytes) {
    const double gbps = s.median > 0.0
                            ? static_cast<double>(bytes) / (s.median * 1.0e3)
                            : 0.0;
    std::printf(
        "PROFILE stage=%s layer=%d rows=%u ctx=%u median=%.3f p10=%.3f p90=%.3f min=%.3f "
        "max=%.3f bytes=%llu gbps=%.2f\n",
        stage, layer, rows, ctx, s.median, s.p10, s.p90, s.min, s.max,
        static_cast<unsigned long long>(bytes), gbps);
}

static uint64_t weight_bytes(const ps::weights::MatrixWeight& w) {
    using E = ps::weights::MatrixEncoding;
    if (w.encoding == E::Bf16) return static_cast<uint64_t>(w.rows) * w.cols * 2u;
    if (w.encoding == E::Psq8) {
        return static_cast<uint64_t>(w.rows) *
               (static_cast<uint64_t>(w.k_padded) + w.storage_scale_stride_bytes +
                w.scale_row_stride_bytes);
    }
    if (w.encoding == E::Psq4) {
        return static_cast<uint64_t>(w.rows) *
               (static_cast<uint64_t>(w.k_padded) / 2u + w.storage_scale_stride_bytes +
                w.scale_row_stride_bytes);
    }
    return 0u;
}

static std::optional<ps::kernel::Bf16GemmConfig> cfg_bf16(
    const ps::weights::MatrixWeight& w,
    uint32_t rows,
    uint32_t input_row_stride,
    bool exact) {
    ps::qwen35::runtime::Bf16GemmSelectorInput lin;
    lin.rows = rows;
    lin.out_features = w.rows;
    lin.k = w.cols;
    lin.input_row_stride = input_row_stride;
    lin.weight_aligned16 = true;
    lin.input_aligned16 = true;
    lin.verify_exact = exact;
    return ps::qwen35::runtime::select_bf16_gemm_config(lin);
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

    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
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

    const auto weight_load_start = std::chrono::steady_clock::now();
    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    const auto weight_load_end = std::chrono::steady_clock::now();
    check(weights_result.ok(), "weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();
    std::printf("PROFILE_LOAD dflash2_load_ms=%.2f backbone_psq4=%d\n",
                std::chrono::duration<double, std::milli>(weight_load_end - weight_load_start)
                    .count(),
                weights.backbone_psq4 ? 1 : 0);

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    exec_config.target_lm_head = &target.weights().lm_head;
    exec_config.target_embed_tokens = &target.weights().embed_tokens;
    auto exec_result =
        ps::qwen35::dflash2::create_dflash2_executor(config, weights, arena, exec_config);
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
    const uint32_t conv_group = static_cast<uint32_t>(config.conv_group_size);
    const uint32_t capacity = static_cast<uint32_t>(config.sliding_window);
    const uint32_t vocab = static_cast<uint32_t>(config.vocab_size);
    const uint32_t rank = static_cast<uint32_t>(config.selector_rank);
    const uint32_t top_k = static_cast<uint32_t>(config.selector_top_k);
    const float eps = config.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const uint32_t layers = static_cast<uint32_t>(config.num_hidden_layers);

    ps::gpu::Tensor feature;
    ps::gpu::Tensor noise;
    check(alloc_bf16(arena, {capacity, hidden}, feature), "feature alloc");
    check(alloc_bf16(arena, {8u, hidden}, noise), "noise alloc");
    {
        std::vector<bf16_t> host(static_cast<std::size_t>(capacity) * hidden);
        uint32_t seed = 24681357u;
        for (bf16_t& x : host) {
            seed = seed * 1664525u + 1013904223u;
            const float u = static_cast<float>(seed >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{ps::f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.25f)};
        }
        check(hipMemcpy(feature.data<bf16_t>(), host.data(), host.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "feature upload");
        std::vector<bf16_t> hn(static_cast<std::size_t>(8u) * hidden);
        for (bf16_t& x : hn) {
            seed = seed * 1664525u + 1013904223u;
            const float u = static_cast<float>(seed >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{ps::f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.25f)};
        }
        check(hipMemcpy(noise.data<bf16_t>(), hn.data(), hn.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "noise upload");
    }

    auto state_result = ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
    check(state_result.ok(), "context create");
    if (!state_result.ok()) return 1;
    DFlash2ContextState state = state_result.release();

    auto fill_context = [&](uint32_t ctx) {
        (void)ps::qwen35::dflash2::dflash2_context_reset(state, 0u);
        if (ctx > 0u) {
            (void)ps::qwen35::dflash2::dflash2_context_append(
                ex, state, feature.data<bf16_t>(), ctx, 0u, stream);
        }
        (void)hipStreamSynchronize(stream);
    };

    const uint32_t base_rows = 8u;
    const uint32_t base_ctx = 2048u;

    std::printf("GATE11_PROFILE_BEGIN warmup=%u iters=%u\n", kWarmup, kIters);
    std::printf("GATE11_CONFIG hidden=%u q=%u kv=%u inter=%u vocab=%u rank=%u top_k=%u "
                "block=%zu window=%zu layers=%u\n",
                hidden, q_features, kv_features, intermediate, vocab, rank, top_k,
                config.block_size, config.sliding_window, layers);

    fill_context(base_ctx);

    const bf16_t* hidden_in = noise.data<bf16_t>();
    const bf16_t* k_ring0 = state.k_ring.data<bf16_t>();
    const bf16_t* v_ring0 = state.v_ring.data<bf16_t>();
    const bool psq4_backbone = ex.backbone_psq4;

    auto time_linear_stage = [&](const char* stage, int L, const ps::weights::MatrixWeight& w,
                                 const bf16_t* input, uint32_t istride, uint32_t stage_rows,
                                 uint32_t emit_ctx, bf16_t* out, uint32_t ostride,
                                 bool quantize_here) {
        if (psq4_backbone) {
            if (quantize_here) {
                const std::string qn = std::string(stage) + "_quantize";
                emit(qn.c_str(), L, stage_rows, emit_ctx,
                     time_stats(stream, [&]() {
                         (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                             ex, input, stage_rows, w.cols, istride, stage, stream);
                     }),
                     0u);
            }
            emit(stage, L, stage_rows, emit_ctx,
                 time_stats(stream, [&]() {
                     (void)ps::qwen35::dflash2::dflash2_gemm_psq4(
                         ex, w, out, ostride, stage_rows, stage, stream);
                 }),
                 weight_bytes(w));
            return;
        }
        const auto c = cfg_bf16(w, stage_rows, istride, false);
        emit(stage, L, stage_rows, emit_ctx,
             time_stats(stream, [&]() {
                 if (c.has_value()) {
                     (void)ps::kernel::launch_gemm_bf16(
                         *c, w.bf16.data<bf16_t>(), input, out,
                         ps::kernel::Bf16GemmOutputDType::BF16, stage_rows, w.rows, w.cols,
                         istride, ostride, stream);
                 }
             }),
             weight_bytes(w));
    };

    emit("target_feature_concat", 0, base_rows, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::kernel::launch_dflash2_feature_concat(
                 ex.target_feature.data<bf16_t>(), ex.target_feature.data<bf16_t>(),
                 ex.target_feature.data<bf16_t>(), ex.target_feature.data<bf16_t>(),
                 ex.target_feature.data<bf16_t>(), hidden, base_rows,
                 ex.target_concat.data<bf16_t>(),
                 static_cast<uint32_t>(ex.target_concat.dim(1)), stream);
         }),
         0u);
    time_linear_stage("target_fc_gemm", 0, weights.fc, ex.target_concat.data<bf16_t>(),
                      static_cast<uint32_t>(config.tap_feature_size()), base_rows, base_ctx,
                      ex.target_fc.data<bf16_t>(),
                      static_cast<uint32_t>(ex.target_fc.dim(1)), true);
    if (psq4_backbone) {
        emit("act_quantize_per_layer_total", 0, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 for (const auto& layer : weights.layers) {
                     (void)layer;
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, ex.layer_input_norm.data<bf16_t>(), base_rows, hidden, hidden,
                         "attn_conv", stream);
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, ex.attention_prepared.data<bf16_t>(), base_rows, hidden, hidden,
                         "attention_prepared", stream);
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, ex.attention_context.data<bf16_t>(), base_rows, q_features,
                         q_features, "o_proj", stream);
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, ex.post_attention_norm.data<bf16_t>(), base_rows, hidden, hidden,
                         "mlp_conv", stream);
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, ex.mlp_prepared.data<bf16_t>(), base_rows, hidden, hidden,
                         "mlp_gate", stream);
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, ex.mlp_swiglu.data<bf16_t>(), base_rows, intermediate,
                         intermediate, "mlp_down", stream);
                 }
             }),
             0u);
    }
    emit("target_hidden_norm", 0, base_rows, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
                 ex.target_fc.data<bf16_t>(), weights.hidden_norm_weight.data<bf16_t>(),
                 ex.target_feature.data<bf16_t>(), base_rows, hidden, hidden, eps, stream);
         }),
         0u);

    for (uint32_t li = 0; li < layers; ++li) {
        const auto& layer = weights.layers[li];
        const int L = static_cast<int>(li);

        emit("input_norm", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
                     hidden_in, layer.input_layernorm_weight.data<bf16_t>(),
                     ex.layer_input_norm.data<bf16_t>(), base_rows, hidden, hidden, eps,
                     stream);
             }),
             0u);

        time_linear_stage("attention_dynamic_projection", L,
                          layer.attention_conv_kernel_projection,
                          ex.layer_input_norm.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.conv_dynamic.data<bf16_t>(),
                          static_cast<uint32_t>(ex.conv_dynamic.dim(1)), true);
        emit("attention_conv_prepare", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_grouped_dynamic_conv(
                     ex.layer_input_norm.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                     layer.attention_conv_base_kernel.data<bf16_t>(),
                     ex.attention_prepared.data<bf16_t>(), base_rows, hidden, conv_group, 0u,
                     stream);
             }),
             0u);

        time_linear_stage("q_proj", L, layer.attn_q_proj,
                          ex.attention_prepared.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.q_raw.data<bf16_t>(), q_features, true);
        time_linear_stage("k_noise_proj", L, layer.attn_k_proj,
                          ex.attention_prepared.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.k_noise_raw.data<bf16_t>(), kv_features, false);
        time_linear_stage("v_noise_proj", L, layer.attn_v_proj,
                          ex.attention_prepared.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.v_noise.data<bf16_t>(), kv_features, false);

        emit("q_norm", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
                     ex.q_raw.data<bf16_t>(), layer.attn_q_norm_weight.data<bf16_t>(),
                     ex.q_norm.data<bf16_t>(), base_rows, q_features, head_dim, eps, stream);
             }),
             0u);
        emit("k_noise_norm", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
                     ex.k_noise_raw.data<bf16_t>(), layer.attn_k_norm_weight.data<bf16_t>(),
                     ex.k_noise_norm.data<bf16_t>(), base_rows, kv_features, head_dim, eps,
                     stream);
             }),
             0u);

        emit("q_rope", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_rope_bf16(
                     ex.q_norm.data<bf16_t>(), q_features, ex.q_rope.data<bf16_t>(), q_features,
                     base_rows, q_features, head_dim, base_ctx, config.rope_theta, stream);
             }),
             0u);
        emit("k_noise_rope", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_rope_bf16(
                     ex.k_noise_norm.data<bf16_t>(), kv_features,
                     ex.k_noise_rope.data<bf16_t>(), kv_features, base_rows, kv_features,
                     head_dim, base_ctx, config.rope_theta, stream);
             }),
             0u);

        emit("ring_attention", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_attention_ring_bf16(
                     ex.q_rope.data<bf16_t>(), q_features, k_ring0, v_ring0, capacity,
                     base_ctx, 0u, kv_features, ex.k_noise_rope.data<bf16_t>(),
                     ex.v_noise.data<bf16_t>(), base_rows, kv_features,
                     ex.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads,
                     head_dim, base_ctx, capacity, scale, stream);
             }),
             0u);

        time_linear_stage("o_proj", L, layer.attn_o_proj,
                          ex.attention_context.data<bf16_t>(), q_features, base_rows, base_ctx,
                          ex.attention_o.data<bf16_t>(), hidden, true);
        emit("attention_conv_finish", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_grouped_dynamic_conv(
                     ex.attention_o.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                     layer.attention_conv_base_kernel.data<bf16_t>(),
                     ex.attention_finished.data<bf16_t>(), base_rows, hidden, conv_group, 1u,
                     stream);
             }),
             0u);
        emit("attention_residual", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_residual_add_bf16(
                     hidden_in, hidden, ex.attention_finished.data<bf16_t>(), hidden,
                     ex.attention_residual.data<bf16_t>(), hidden, base_rows, hidden, stream);
             }),
             0u);
        emit("post_attention_norm", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
                     ex.attention_residual.data<bf16_t>(),
                     layer.post_attention_layernorm_weight.data<bf16_t>(),
                     ex.post_attention_norm.data<bf16_t>(), base_rows, hidden, hidden, eps,
                     stream);
             }),
             0u);

        time_linear_stage("mlp_dynamic_projection", L, layer.mlp_conv_kernel_projection,
                          ex.post_attention_norm.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.conv_dynamic.data<bf16_t>(),
                          static_cast<uint32_t>(ex.conv_dynamic.dim(1)), true);
        emit("mlp_conv_prepare", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_grouped_dynamic_conv(
                     ex.post_attention_norm.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                     layer.mlp_conv_base_kernel.data<bf16_t>(),
                     ex.mlp_prepared.data<bf16_t>(), base_rows, hidden, conv_group, 0u, stream);
             }),
             0u);

        time_linear_stage("mlp_gate_proj", L, layer.mlp_gate_proj,
                          ex.mlp_prepared.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.mlp_gate.data<bf16_t>(), intermediate, true);
        time_linear_stage("mlp_up_proj", L, layer.mlp_up_proj,
                          ex.mlp_prepared.data<bf16_t>(), hidden, base_rows, base_ctx,
                          ex.mlp_up.data<bf16_t>(), intermediate, false);
        emit("mlp_swiglu", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_swiglu_bf16(
                     ex.mlp_gate.data<bf16_t>(), intermediate, ex.mlp_up.data<bf16_t>(),
                     intermediate, ex.mlp_swiglu.data<bf16_t>(), intermediate, base_rows,
                     intermediate, stream);
             }),
             0u);
        time_linear_stage("mlp_down_proj", L, layer.mlp_down_proj,
                          ex.mlp_swiglu.data<bf16_t>(), intermediate, base_rows, base_ctx,
                          ex.mlp_down.data<bf16_t>(), hidden, true);
        emit("mlp_conv_finish", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_grouped_dynamic_conv(
                     ex.mlp_down.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                     layer.mlp_conv_base_kernel.data<bf16_t>(),
                     ex.mlp_finished.data<bf16_t>(), base_rows, hidden, conv_group, 1u, stream);
             }),
             0u);
        emit("mlp_residual", L, base_rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_residual_add_bf16(
                     ex.attention_residual.data<bf16_t>(), hidden,
                     ex.mlp_finished.data<bf16_t>(), hidden, ex.backbone_output.data<bf16_t>(),
                     hidden, base_rows, hidden, stream);
             }),
             0u);
    }

    emit("final_norm", 0, base_rows, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
                 ex.backbone_hidden_a.data<bf16_t>(), weights.final_norm_weight.data<bf16_t>(),
                 ex.backbone_output.data<bf16_t>(), base_rows, hidden, hidden, eps, stream);
         }),
         0u);

    const bf16_t* draft_hidden = ex.backbone_output.data<bf16_t>() + hidden;
    emit("target_lm_head", 0, base_rows - 1u, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::qwen35::dflash2::dflash2_compute_target_logits(
                 ex, draft_hidden, base_rows - 1u, ex.proposal_logits.data<float>(), vocab,
                 stream);
         }),
         weight_bytes(*ex.target_lm_head));
    emit("top16", 0, base_rows - 1u, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::kernel::launch_dflash2_topk_f32(
                 ex.proposal_logits.data<float>(), base_rows - 1u, vocab, vocab, top_k,
                 ex.proposal_topk_ids.data<int32_t>(), ex.proposal_topk_logits.data<float>(),
                 ex.topk_scratch_ids.data<int32_t>(), ex.topk_scratch_logits.data<float>(),
                 stream);
         }),
         static_cast<uint64_t>(base_rows - 1u) * vocab * sizeof(float));
    {
        const auto& proj = weights.selector.hidden_projection;
        const auto c = cfg_bf16(proj, base_rows - 1u, hidden, true);
        emit("selector_hidden_projection", 0, base_rows - 1u, base_ctx,
             time_stats(stream, [&]() {
                 if (c.has_value()) {
                     (void)ps::kernel::launch_gemm_bf16(
                         *c, proj.bf16.data<bf16_t>(), draft_hidden,
                         ex.selector_hidden_proj.data<bf16_t>(),
                         ps::kernel::Bf16GemmOutputDType::BF16, base_rows - 1u, proj.rows,
                         proj.cols, hidden, rank, stream);
                 }
             }),
             weight_bytes(proj));
    }
    emit("candidate_selector", 0, base_rows - 1u, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::qwen35::dflash2::dflash2_run_candidate_selector(
                 ex, ex.selector_hidden_proj.data<bf16_t>(),
                 ex.proposal_topk_ids.data<int32_t>(),
                 ex.proposal_topk_logits.data<float>(), base_rows - 1u, 0,
                 ex.proposal_tokens.data<int32_t>(), stream);
         }),
         0u);
    emit("noise_token_ids", 0, base_rows, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::kernel::launch_dflash2_noise_token_ids(
                 0, static_cast<int32_t>(config.mask_token_id),
                 ex.noise_token_ids.data<int32_t>(), base_rows, stream);
         }),
         0u);
    emit("noise_embedding", 0, base_rows, base_ctx,
         time_stats(stream, [&]() {
             (void)ps::qwen35::dflash2::dflash2_prepare_noise_embedding(ex, 0, base_rows, stream);
         }),
         weight_bytes(*ex.target_embed_tokens));

    for (uint32_t ctx : {8u, 128u, 512u, 2048u}) {
        fill_context(ctx);
        emit("ring_attention_ctxsweep", 0, base_rows, ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_attention_ring_bf16(
                     ex.q_rope.data<bf16_t>(), q_features, k_ring0, v_ring0, capacity, ctx, 0u,
                     kv_features, ex.k_noise_rope.data<bf16_t>(), ex.v_noise.data<bf16_t>(),
                     base_rows, kv_features, ex.attention_context.data<bf16_t>(), q_features,
                     q_heads, kv_heads, head_dim, ctx, capacity, scale, stream);
             }),
             0u);
        emit("ring_attention_gqa_ctxsweep", 0, base_rows, ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_attention_ring_gqa_bf16(
                     ex.q_rope.data<bf16_t>(), q_features, k_ring0, v_ring0, capacity, ctx, 0u,
                     kv_features, ex.k_noise_rope.data<bf16_t>(), ex.v_noise.data<bf16_t>(),
                     base_rows, kv_features, ex.attention_context.data<bf16_t>(), q_features,
                     q_heads, kv_heads, head_dim, ctx, capacity, scale, stream);
             }),
             0u);
        if (psq4_backbone) {
            emit("ctx_feature_quantize", 0, 0u, ctx,
                 time_stats(stream, [&]() {
                     (void)ps::qwen35::dflash2::dflash2_quantize_a8(
                         ex, feature.data<bf16_t>(), ctx, hidden, hidden, "ctx", stream);
                 }),
                 0u);
            emit("ctx_kv_gemms", 0, ctx, ctx,
                 time_stats(stream, [&]() {
                     for (uint32_t li = 0; li < layers; ++li) {
                         const auto& layer = weights.layers[li];
                         (void)ps::qwen35::dflash2::dflash2_gemm_psq4(
                             ex, layer.attn_k_proj, ex.k_ctx_raw.data<bf16_t>(), kv_features,
                             ctx, "k_ctx_proj", stream);
                         (void)ps::qwen35::dflash2::dflash2_gemm_psq4(
                             ex, layer.attn_v_proj, ex.v_ctx.data<bf16_t>(), kv_features, ctx,
                             "v_ctx_proj", stream);
                     }
                 }),
                 weight_bytes(weights.layers[0].attn_k_proj) * 2u * layers);
        } else {
            emit("ctx_kv_gemms", 0, ctx, ctx,
                 time_stats(stream, [&]() {
                     for (uint32_t li = 0; li < layers; ++li) {
                         const auto& layer = weights.layers[li];
                         const auto ck = cfg_bf16(layer.attn_k_proj, ctx, hidden, false);
                         const auto cv = cfg_bf16(layer.attn_v_proj, ctx, hidden, false);
                         if (ck.has_value())
                             (void)ps::kernel::launch_gemm_bf16(
                                 *ck, layer.attn_k_proj.bf16.data<bf16_t>(),
                                 feature.data<bf16_t>(), ex.k_ctx_raw.data<bf16_t>(),
                                 ps::kernel::Bf16GemmOutputDType::BF16, ctx,
                                 layer.attn_k_proj.rows, layer.attn_k_proj.cols, hidden,
                                 kv_features, stream);
                         if (cv.has_value())
                             (void)ps::kernel::launch_gemm_bf16(
                                 *cv, layer.attn_v_proj.bf16.data<bf16_t>(),
                                 feature.data<bf16_t>(), ex.v_ctx.data<bf16_t>(),
                                 ps::kernel::Bf16GemmOutputDType::BF16, ctx,
                                 layer.attn_v_proj.rows, layer.attn_v_proj.cols, hidden,
                                 kv_features, stream);
                     }
                 }),
                 weight_bytes(weights.layers[0].attn_k_proj) * 2u * layers);
        }
        (void)ps::qwen35::dflash2::dflash2_prepare_noise_embedding(ex, 0, base_rows, stream);
        emit("cached_backbone", 0, base_rows, ctx,
             time_stats(stream, [&]() {
                 (void)ps::qwen35::dflash2::dflash2_forward_backbone_cached(
                     ex, state, ex.noise_embedding.data<bf16_t>(), ctx, base_rows,
                     ex.backbone_output.data<bf16_t>(), stream);
             }),
             0u);
        emit("proposer_cached", 0, base_rows, ctx,
             time_stats(stream, [&]() {
                 (void)ps::qwen35::dflash2::dflash2_propose_cached(
                     ex, state, 0, base_rows - 1u, ex.proposal_tokens.data<int32_t>(), stream);
             }),
             0u);
    }

    for (uint32_t rows : {1u, 3u, 7u}) {
        emit("target_lm_head_rows", 0, rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::qwen35::dflash2::dflash2_compute_target_logits(
                     ex, draft_hidden, rows, ex.proposal_logits.data<float>(), vocab, stream);
             }),
             weight_bytes(*ex.target_lm_head));
        emit("top16_rows", 0, rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_topk_f32(
                     ex.proposal_logits.data<float>(), rows, vocab, vocab, top_k,
                     ex.proposal_topk_ids.data<int32_t>(),
                     ex.proposal_topk_logits.data<float>(),
                     ex.topk_scratch_ids.data<int32_t>(), ex.topk_scratch_logits.data<float>(),
                     stream);
             }),
             static_cast<uint64_t>(rows) * vocab * sizeof(float));
    }
    for (uint32_t rows : {2u, 4u, 8u}) {
        const auto& layer = weights.layers[0];
        time_linear_stage("q_proj_rows", 0, layer.attn_q_proj,
                          ex.attention_prepared.data<bf16_t>(), hidden, rows, base_ctx,
                          ex.q_raw.data<bf16_t>(), q_features, true);
        emit("ring_attention_rows", 0, rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_attention_ring_bf16(
                     ex.q_rope.data<bf16_t>(), q_features, k_ring0, v_ring0, capacity,
                     base_ctx, 0u, kv_features, ex.k_noise_rope.data<bf16_t>(),
                     ex.v_noise.data<bf16_t>(), rows, kv_features,
                     ex.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads,
                     head_dim, base_ctx, capacity, scale, stream);
             }),
             0u);
        emit("ring_attention_gqa_rows", 0, rows, base_ctx,
             time_stats(stream, [&]() {
                 (void)ps::kernel::launch_dflash2_attention_ring_gqa_bf16(
                     ex.q_rope.data<bf16_t>(), q_features, k_ring0, v_ring0, capacity,
                     base_ctx, 0u, kv_features, ex.k_noise_rope.data<bf16_t>(),
                     ex.v_noise.data<bf16_t>(), rows, kv_features,
                     ex.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads,
                     head_dim, base_ctx, capacity, scale, stream);
             }),
             0u);
    }

    std::printf("GATE11_PROFILE_END\n");
    check(ps::qwen35::dflash2::dflash2_context_shutdown(state).ok(), "context shutdown");
    check(ps::qwen35::dflash2::dflash2_executor_shutdown(ex).ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate11_profile: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
