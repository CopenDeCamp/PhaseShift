#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/feature_concat.h>
#include <phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
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
    const double us =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return us / static_cast<double>(iters);
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
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const std::size_t groups = config.conv_groups();

    ps::gpu::Tensor taps[ps::kernel::kDFlash2TargetTaps];
    ps::gpu::Tensor conv_input;
    ps::gpu::Tensor conv_out;
    for (std::size_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
        check(alloc_bf16(arena, {8u, hidden}, taps[t]), "tap alloc");
    }
    check(alloc_bf16(arena, {8u, hidden}, conv_input), "conv input alloc");
    check(alloc_bf16(arena, {8u, hidden}, conv_out), "conv out alloc");

    ps::qwen35::runtime::Bf16GemmSelectorInput fc_sel;
    fc_sel.rows = 8u;
    fc_sel.out_features = static_cast<uint32_t>(weights.fc.rows);
    fc_sel.k = static_cast<uint32_t>(weights.fc.cols);
    fc_sel.input_row_stride = static_cast<uint32_t>(weights.fc.cols);
    fc_sel.weight_aligned16 =
        (reinterpret_cast<uintptr_t>(weights.fc.bf16.data<void>()) & 15u) == 0u;
    fc_sel.input_aligned16 =
        (reinterpret_cast<uintptr_t>(ex.target_concat.data<void>()) & 15u) == 0u;

    const auto& layer0 = weights.layers[0];
    ps::qwen35::runtime::Bf16GemmSelectorInput kp_sel;
    kp_sel.rows = 8u;
    kp_sel.out_features = static_cast<uint32_t>(layer0.attention_conv_kernel_projection.rows);
    kp_sel.k = static_cast<uint32_t>(layer0.attention_conv_kernel_projection.cols);
    kp_sel.input_row_stride = hidden;
    kp_sel.weight_aligned16 =
        (reinterpret_cast<uintptr_t>(layer0.attention_conv_kernel_projection.bf16.data<void>()) &
         15u) == 0u;
    kp_sel.input_aligned16 =
        (reinterpret_cast<uintptr_t>(conv_input.data<void>()) & 15u) == 0u;

    constexpr uint32_t kIters = 50u;
    for (uint32_t rows : {1u, 8u}) {
        ps::qwen35::runtime::Bf16GemmSelectorInput fc_rows = fc_sel;
        fc_rows.rows = rows;
        const auto fc_config = ps::qwen35::runtime::select_bf16_gemm_config(fc_rows);
        ps::qwen35::runtime::Bf16GemmSelectorInput kp_rows = kp_sel;
        kp_rows.rows = rows;
        const auto kp_config = ps::qwen35::runtime::select_bf16_gemm_config(kp_rows);
        check(fc_config.has_value(), "fc gemm config rows=" + std::to_string(rows));
        check(kp_config.has_value(), "kp gemm config rows=" + std::to_string(rows));
        if (!fc_config.has_value() || !kp_config.has_value()) return 1;

        const bf16_t* tap_ptrs[ps::kernel::kDFlash2TargetTaps] = {};
        for (std::size_t t = 0; t < ps::kernel::kDFlash2TargetTaps; ++t) {
            tap_ptrs[t] = taps[t].data<bf16_t>();
        }

        const double concat_us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_dflash2_feature_concat(
                tap_ptrs[0], tap_ptrs[1], tap_ptrs[2], tap_ptrs[3], tap_ptrs[4], hidden, rows,
                ex.target_concat.data<bf16_t>(), static_cast<uint32_t>(ex.target_concat.dim(1)),
                stream);
        });
        const double fc_us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_gemm_bf16(
                *fc_config, weights.fc.bf16.data<bf16_t>(), ex.target_concat.data<bf16_t>(),
                ex.target_fc.data<bf16_t>(), ps::kernel::Bf16GemmOutputDType::BF16, rows,
                static_cast<uint32_t>(weights.fc.rows), static_cast<uint32_t>(weights.fc.cols),
                static_cast<uint32_t>(ex.target_concat.dim(1)),
                static_cast<uint32_t>(ex.target_fc.dim(1)), stream);
        });
        const double norm_us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_rmsnorm(
                ex.target_fc.data<void>(), ps::kernel::RmsNormDataType::BF16,
                weights.hidden_norm_weight.data<void>(), ps::kernel::RmsNormDataType::BF16,
                ps::kernel::RmsNormWeightLayout::PER_FEATURE,
                ps::kernel::RmsNormWeightMode::DIRECT, ex.target_feature.data<void>(),
                ps::kernel::RmsNormDataType::BF16, rows, hidden, hidden,
                static_cast<uint32_t>(ex.target_fc.dim(1)),
                static_cast<uint32_t>(ex.target_feature.dim(1)), config.rms_norm_eps, stream);
        });
        const double kp_us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_gemm_bf16(
                *kp_config, layer0.attention_conv_kernel_projection.bf16.data<bf16_t>(),
                conv_input.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                ps::kernel::Bf16GemmOutputDType::BF16, rows,
                static_cast<uint32_t>(layer0.attention_conv_kernel_projection.rows),
                static_cast<uint32_t>(layer0.attention_conv_kernel_projection.cols), hidden,
                static_cast<uint32_t>(ex.conv_dynamic.dim(1)), stream);
        });
        const double conv0_us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_dflash2_grouped_dynamic_conv(
                conv_input.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                layer0.attention_conv_base_kernel.data<bf16_t>(), conv_out.data<bf16_t>(),
                rows, hidden, static_cast<uint32_t>(config.conv_group_size), 0u, stream);
        });
        const double conv1_us = time_events_us(stream, kIters, [&]() {
            (void)ps::kernel::launch_dflash2_grouped_dynamic_conv(
                conv_input.data<bf16_t>(), ex.conv_dynamic.data<bf16_t>(),
                layer0.attention_conv_base_kernel.data<bf16_t>(), conv_out.data<bf16_t>(),
                rows, hidden, static_cast<uint32_t>(config.conv_group_size), 1u, stream);
        });

        const double project_us = time_events_us(stream, kIters, [&]() {
            ps::qwen35::dflash2::dflash2_project_target_features(
                ex, tap_ptrs, ps::kernel::kDFlash2TargetTaps, rows, stream);
        });
        const double project_wall_us = time_wall_us(stream, kIters, [&]() {
            ps::qwen35::dflash2::dflash2_project_target_features(
                ex, tap_ptrs, ps::kernel::kDFlash2TargetTaps, rows, stream);
        });
        const double prepare_us = time_events_us(stream, kIters, [&]() {
            ps::qwen35::dflash2::dflash2_grouped_conv_prepare(
                ex, conv_input.data<bf16_t>(), layer0, true, rows, conv_out.data<bf16_t>(),
                stream);
        });
        const double prepare_wall_us = time_wall_us(stream, kIters, [&]() {
            ps::qwen35::dflash2::dflash2_grouped_conv_prepare(
                ex, conv_input.data<bf16_t>(), layer0, true, rows, conv_out.data<bf16_t>(),
                stream);
        });
        const double finish_us = time_events_us(stream, kIters, [&]() {
            ps::qwen35::dflash2::dflash2_grouped_conv_finish(
                ex, conv_input.data<bf16_t>(), layer0, true, rows, conv_out.data<bf16_t>(),
                stream);
        });
        const double finish_wall_us = time_wall_us(stream, kIters, [&]() {
            ps::qwen35::dflash2::dflash2_grouped_conv_finish(
                ex, conv_input.data<bf16_t>(), layer0, true, rows, conv_out.data<bf16_t>(),
                stream);
        });

        std::printf(
            "rows=%u concat=%.2fus fc=%.2fus hidden_norm=%.2fus kernel_projection=%.2fus "
            "conv_prepare=%.2fus conv_finish=%.2fus | project=%.2fus(wall %.2fus) "
            "prepare=%.2fus(wall %.2fus) finish=%.2fus(wall %.2fus)\n",
            rows, concat_us, fc_us, norm_us, kp_us, conv0_us, conv1_us, project_us,
            project_wall_us, prepare_us, prepare_wall_us, finish_us, finish_wall_us);
        (void)groups;
    }

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");

    std::printf("test_dflash2_gate3_perf: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
