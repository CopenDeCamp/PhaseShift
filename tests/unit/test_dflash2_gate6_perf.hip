#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
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

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = target_dir;
    resident_request.dflash2_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 30ull * 1024ull * 1024ull * 1024ull;
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

    auto target_result = resident.enabled()
        ? resident.take_qwen35()
        : ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(target_result.ok(), "target load");
    if (!target_result.ok()) return 1;
    ps::qwen35::Qwen35Model target = target_result.release();

    auto weights_result = resident.enabled()
        ? resident.take_dflash2(nullptr)
        : ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = 8u;
    exec_config.target_lm_head = &target.weights().lm_head;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t block_rows = 8u;
    const uint32_t draft_rows = block_rows - 1u;
    const uint32_t context_rows = 8u;
    const uint32_t context_start = 4096u;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t rank = config.selector_rank;
    const uint32_t vocab = config.vocab_size;
    const uint32_t top_k = config.selector_top_k;

    ps::gpu::Tensor noise;
    ps::gpu::Tensor feature;
    ps::gpu::Tensor final_hidden;
    check(alloc_bf16(arena, {block_rows, hidden}, noise), "noise alloc");
    check(alloc_bf16(arena, {context_rows, hidden}, feature), "feature alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, final_hidden), "final alloc");

    constexpr uint32_t kIters = 20u;
    const bf16_t* draft_hidden = final_hidden.data<bf16_t>() + hidden;

    const double lm_head_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_compute_target_logits(
            ex, draft_hidden, draft_rows, ex.proposal_logits.data<float>(), vocab, stream);
    });

    const double topk_us = time_events_us(stream, kIters, [&]() {
        (void)ps::kernel::launch_dflash2_topk_f32(
            ex.proposal_logits.data<float>(), draft_rows, vocab, vocab, top_k,
            ex.proposal_topk_ids.data<int32_t>(),
            ex.proposal_topk_logits.data<float>(),
            ex.topk_scratch_ids.data<int32_t>(),
            ex.topk_scratch_logits.data<float>(), stream);
    });

    const ps::weights::MatrixWeight& projection = weights.selector.hidden_projection;
    ps::qwen35::runtime::Bf16GemmSelectorInput lin;
    lin.rows = draft_rows;
    lin.out_features = projection.rows;
    lin.k = projection.cols;
    lin.input_row_stride = hidden;
    lin.weight_aligned16 = true;
    lin.input_aligned16 = true;
    lin.verify_exact = true;
    const auto projection_config =
        ps::qwen35::runtime::select_bf16_gemm_config(lin);
    check(projection_config.has_value(), "hidden projection config");
    const double projection_us = projection_config.has_value()
        ? time_events_us(stream, kIters, [&]() {
              (void)ps::kernel::launch_gemm_bf16(
                  *projection_config, projection.bf16.data<bf16_t>(), draft_hidden,
                  ex.selector_hidden_proj.data<bf16_t>(),
                  ps::kernel::Bf16GemmOutputDType::BF16, draft_rows, projection.rows,
                  projection.cols, hidden, rank, stream);
          })
        : -1.0;

    const double selector_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_run_candidate_selector(
            ex, ex.selector_hidden_proj.data<bf16_t>(),
            ex.proposal_topk_ids.data<int32_t>(),
            ex.proposal_topk_logits.data<float>(), draft_rows, 0,
            ex.proposal_tokens.data<int32_t>(), stream);
    });

    const double select_total_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_select_draft_tokens(
            ex, final_hidden.data<bf16_t>(), block_rows, 0,
            ex.proposal_tokens.data<int32_t>(), stream);
    });

    const double composed_us = time_events_us(stream, kIters, [&]() {
        (void)ps::qwen35::dflash2::dflash2_forward_and_select_stateless(
            ex, noise.data<bf16_t>(), feature.data<bf16_t>(), context_rows, context_start,
            context_start + context_rows, block_rows, 0,
            ex.proposal_tokens.data<int32_t>(), stream);
    });

    std::printf("dflash2 gate6 perf (block_rows=%u draft_rows=%u, device 0)\n", block_rows,
                draft_rows);
    std::printf("  target lm_head (M=%u)          : %8.2f us\n", draft_rows, lm_head_us);
    std::printf("  top-16 (vocab=%u)             : %8.2f us\n", vocab, topk_us);
    std::printf("  selector hidden projection     : %8.2f us\n", projection_us);
    std::printf("  candidate selector             : %8.2f us\n", selector_us);
    std::printf("  selector stage total           : %8.2f us\n", select_total_us);
    std::printf("  composed (backbone+selector)   : %8.2f us\n", composed_us);

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");
    std::printf("test_dflash2_gate6_perf: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
