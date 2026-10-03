#include "support/target_lm_proxy_harness.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/kernels/optimized/sampling.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

struct Bench {
    hipStream_t stream = nullptr;
    hipEvent_t a = nullptr;
    hipEvent_t b = nullptr;
    void start() {
        if (a == nullptr) {
            (void)hipEventCreate(&a);
            (void)hipEventCreate(&b);
        }
        (void)hipEventRecord(a, stream);
    }
    double stop_ms() {
        (void)hipEventRecord(b, stream);
        (void)hipStreamSynchronize(stream);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, a, b);
        return static_cast<double>(ms);
    }
    ~Bench() {
        if (a != nullptr) (void)hipEventDestroy(a);
        if (b != nullptr) (void)hipEventDestroy(b);
    }
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2u];
}

}  // namespace

int main() {
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (target_dir == nullptr || target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2_TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    const uint32_t perf_rows = 1024u;
    const uint32_t chunk = 128u;
    const uint32_t proxy_rows = 8u;
    const uint32_t proxy_pool = 32u;
    const uint32_t proxy_topn = 128u;
    const uint32_t warmup = 30u;
    const uint32_t iters = 300u;
    const double cert_rate = std::getenv("PHASESHIFT_TARGET_LM_CERT_RATE") != nullptr
                                 ? std::strtod(std::getenv("PHASESHIFT_TARGET_LM_CERT_RATE"),
                                               nullptr)
                                 : 0.0;

    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result = ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
    const uint32_t vocab = static_cast<uint32_t>(model.weights().lm_head.rows);
    const uint32_t cols = static_cast<uint32_t>(model.weights().lm_head.cols);
    const uint32_t k_padded = static_cast<uint32_t>(model.weights().lm_head.k_padded);
    const uint32_t wscale_stride =
        static_cast<uint32_t>(model.weights().lm_head.storage_scale_stride_bytes);
    const float eps = tc.rms_norm_eps;

    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t max_seq = perf_rows + 32u;
    const uint32_t num_pages = ceil_div(max_seq, page_tokens) + 8u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 2u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 2u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL pool create\n");
        return 1;
    }
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = chunk;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = chunk;
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    if (!exec_result.ok()) {
        std::printf("FAIL executor create: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Executor ex = exec_result.release();

    ps::bf16_t* perf_hidden = nullptr;
    ps::bf16_t* normed = nullptr;
    if (hipMalloc(&perf_hidden, static_cast<std::size_t>(perf_rows) * hidden *
                                    sizeof(ps::bf16_t)) != hipSuccess ||
        hipMalloc(&normed, static_cast<std::size_t>(chunk) * hidden * sizeof(ps::bf16_t)) !=
            hipSuccess) {
        std::printf("FAIL hidden alloc\n");
        return 1;
    }

    const ps::gpu::Tensor& fw = model.weights().final_norm_weight;
    const ps::kernel::RmsNormDataType wdt =
        fw.element_size() == 4u ? ps::kernel::RmsNormDataType::F32
                                : ps::kernel::RmsNormDataType::BF16;
    {
        auto seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, max_seq, stream);
        if (!seq_result.ok()) {
            std::printf("FAIL sequence create\n");
            return 1;
        }
        ps::qwen35::PagedSequenceState seq = seq_result.release();
        std::vector<int32_t> tokens(perf_rows);
        for (uint32_t i = 0u; i < perf_rows; ++i)
            tokens[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
        for (uint32_t offset = 0u; offset < perf_rows; offset += chunk) {
            const uint32_t n = std::min(chunk, perf_rows - offset);
            std::vector<ps::qwen35::ScheduledRequest> requests;
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq;
            req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::PREFILL;
            req.token_begin = 0u;
            req.num_tokens = n;
            req.prefix_tokens = offset;
            req.compute_logits = true;
            req.sample = true;
            req.num_output_rows = n;
            requests.push_back(req);
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = tokens.data() + offset;
            batch.requests = requests;
            batch.num_tokens = n;
            batch.num_requests = 1u;
            batch.num_decode_requests = 0u;
            batch.num_verify_requests = 0u;
            batch.num_prefill_requests = 1u;
            auto executed = ps::qwen35::execute_batch(ex, batch, stream);
            if (!executed.ok()) {
                std::printf("FAIL execute: %s\n", executed.status().message().c_str());
                return 1;
            }
            ps::qwen35::BatchExecutionOutput out = executed.release();
            const hipError_t rerr = ps::kernel::launch_rmsnorm(
                out.final_hidden.data<void>(), ps::kernel::RmsNormDataType::BF16,
                fw.data<void>(), wdt, ps::kernel::RmsNormWeightLayout::PER_FEATURE,
                ps::kernel::RmsNormWeightMode::ONE_PLUS, normed,
                ps::kernel::RmsNormDataType::BF16, n, hidden, hidden, hidden, hidden, eps,
                stream);
            if (rerr != hipSuccess) {
                std::printf("FAIL rmsnorm\n");
                return 1;
            }
            (void)hipMemcpy(perf_hidden + static_cast<std::size_t>(offset) * hidden, normed,
                            static_cast<std::size_t>(n) * hidden * sizeof(ps::bf16_t),
                            hipMemcpyDeviceToDevice);
        }
        (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    }

    target_lm_proxy::Proxy proxy;
    std::string perr;
    if (!proxy.init(model.weights().lm_head, proxy_rows, 127u, stream, perr)) {
        std::printf("FAIL proxy init: %s\n", perr.c_str());
        return 1;
    }

    const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(cols, proxy_rows);
    uint8_t* full_act = nullptr;
    float* full_act_scales = nullptr;
    float* full_logits = nullptr;
    int32_t* full_sampled = nullptr;
    if (hipMalloc(&full_act, layout.codes_bytes()) != hipSuccess ||
        hipMalloc(&full_act_scales, layout.scales_bytes()) != hipSuccess ||
        hipMalloc(&full_logits, static_cast<std::size_t>(proxy_rows) * vocab * sizeof(float)) !=
            hipSuccess ||
        hipMalloc(&full_sampled, proxy_rows * sizeof(int32_t)) != hipSuccess) {
        std::printf("FAIL full alloc\n");
        return 1;
    }

    const uint32_t buckets[] = {1u, 2u, 4u, 8u};
    for (uint32_t rows : buckets) {
        const auto pick_config = [](uint32_t r) {
            (void)r;
            ps::kernel::Psq8GemmConfig cfg;
            cfg.id = ps::kernel::Psq8GemmConfigId::RowBlock1;
            return cfg;
        };
        ps::kernel::Psq8GemmConfig cfg = pick_config(rows);
        const auto full_once = [&]() {
            (void)ps::kernel::launch_activation_quantize_e4m3(
                perf_hidden, full_act, full_act_scales, rows, cols, k_padded, hidden,
                layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
            (void)ps::kernel::launch_gemm_psq8_w8a8_wmma_auto(
                cfg, model.weights().lm_head.codes.data<uint8_t>(),
                model.weights().lm_head.scales.data<uint8_t>(),
                reinterpret_cast<const int8_t*>(full_act), full_act_scales, full_logits,
                ps::kernel::Psq8GemmOutputDType::F32, rows, vocab, k_padded, wscale_stride,
                layout.code_row_stride_bytes, layout.scale_row_stride_bytes, vocab, stream);
            ps::kernel::ArgmaxF32Args aa;
            aa.logits = full_logits;
            aa.sampling = nullptr;
            aa.sampled_tokens = full_sampled;
            aa.outputs = rows;
            aa.vocab_size = vocab;
            aa.logits_row_stride = vocab;
            (void)ps::kernel::launch_argmax_f32_single(aa, stream);
        };
        const auto proxy_once = [&]() {
            (void)proxy.prepare(perf_hidden, rows, hidden, stream, perr);
            (void)proxy.topn(rows, proxy_topn, stream, perr);
            (void)proxy.rerank_pool(rows, proxy_pool, proxy_topn, stream, perr);
            (void)proxy.certify(rows, proxy_pool, proxy_topn, 1e-3f, 1e-5f, stream, perr);
        };

        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL sync\n");
            return 1;
        }
        full_once();
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL full head launch (rows=%u)\n", rows);
            return 1;
        }

        Bench bench;
        bench.stream = stream;
        std::vector<double> full_ms;
        std::vector<double> proxy_ms;
        std::vector<double> fallback_ms;
        for (uint32_t i = 0u; i < warmup; ++i) {
            full_once();
            proxy_once();
        }
        (void)hipStreamSynchronize(stream);
        for (uint32_t i = 0u; i < iters; ++i) {
            bench.start();
            full_once();
            full_ms.push_back(bench.stop_ms());

            bench.start();
            proxy_once();
            proxy_ms.push_back(bench.stop_ms());

            bench.start();
            proxy_once();
            full_once();
            fallback_ms.push_back(bench.stop_ms());
        }
        const double f = median(full_ms);
        const double p = median(proxy_ms);
        const double fb = median(fallback_ms);
        const double expected = p + (1.0 - cert_rate) * f;
        std::printf(
            "TARGET_LM_PROXY_PERF rows=%u pool=%u full_ms=%.4f proxy_ms=%.4f "
            "measured_fallback_ms=%.4f cert_rate=%.4f expected_ms=%.4f speedup=%.4f\n",
            rows, proxy_pool, f, p, fb, cert_rate, expected,
            expected > 0.0 ? f / expected : 0.0);
    }

    {
        const uint32_t pool_list[] = {8u, 16u, 24u, 32u};
        const uint32_t nrows = proxy_rows;
        for (uint32_t pool_n : pool_list) {
            const auto score_once = [&]() {
                (void)ps::kernel::launch_rmsnorm(
                    perf_hidden, ps::kernel::RmsNormDataType::BF16, fw.data<void>(), wdt,
                    ps::kernel::RmsNormWeightLayout::PER_FEATURE,
                    ps::kernel::RmsNormWeightMode::ONE_PLUS, normed,
                    ps::kernel::RmsNormDataType::BF16, nrows, hidden, hidden, hidden, hidden, eps,
                    stream);
                (void)proxy.prepare(normed, nrows, hidden, stream, perr);
            };
            const auto topn_once = [&]() { (void)proxy.topn(nrows, pool_n, stream, perr); };
            const auto exact_once = [&]() {
                (void)proxy.rerank_pool(nrows, pool_n, pool_n, stream, perr);
            };
            const auto cert_once = [&]() {
                (void)proxy.certify(nrows, pool_n, pool_n, 1e-3f, 1e-5f, stream, perr);
            };
            for (uint32_t i = 0u; i < warmup; ++i) {
                score_once();
                topn_once();
                exact_once();
                cert_once();
            }
            (void)hipStreamSynchronize(stream);

            Bench nb;
            nb.stream = stream;
            std::vector<double> c1_ms;
            std::vector<double> c2_ms;
            std::vector<double> c3_ms;
            std::vector<double> c4_ms;
            for (uint32_t i = 0u; i < iters; ++i) {
                nb.start();
                score_once();
                c1_ms.push_back(nb.stop_ms());
                nb.start();
                score_once();
                topn_once();
                c2_ms.push_back(nb.stop_ms());
                nb.start();
                score_once();
                topn_once();
                exact_once();
                c3_ms.push_back(nb.stop_ms());
                nb.start();
                score_once();
                topn_once();
                exact_once();
                cert_once();
                c4_ms.push_back(nb.stop_ms());
            }
            const double c1 = median(c1_ms);
            const double c2 = median(c2_ms);
            const double c3 = median(c3_ms);
            const double c4 = median(c4_ms);
            std::printf(
                "TARGET_LM_PROXY_N_PERF rows=%u pool=%u proxy_score_us=%.3f topn_us=%.3f "
                "exact_candidate_us=%.3f certify_us=%.3f total_us=%.3f\n",
                nrows, pool_n, c1 * 1000.0, (c2 - c1) * 1000.0, (c3 - c2) * 1000.0,
                (c4 - c3) * 1000.0, c4 * 1000.0);
        }
    }

    {
        const auto quant_once = [&](uint32_t nrows) {
            (void)ps::kernel::launch_activation_quantize_e4m3(
                perf_hidden, proxy.act_codes, proxy.act_scales, nrows, cols, k_padded, hidden,
                proxy.act_code_stride, proxy.act_scale_stride, stream);
        };
        const auto l2_once = [&](uint32_t nrows) {
            (void)ps::kernel::launch_dflash2_activation_l2(
                proxy.act_codes, proxy.act_scales, nrows, k_padded, proxy.act_code_stride,
                proxy.act_scale_stride, proxy.act_l2, stream);
        };
        const auto coarse_once = [&](uint32_t nrows) {
            (void)ps::kernel::launch_dflash2_int2_coarse_head(
                proxy.int2_codes, model.weights().lm_head.scales.data<uint8_t>(),
                proxy.act_codes, proxy.act_scales, proxy.expand_table(), proxy.coarse, nrows,
                vocab, k_padded, proxy.scale_stride, proxy.act_code_stride,
                proxy.act_scale_stride, vocab, stream);
        };
        const auto upper_once = [&](uint32_t nrows) {
            (void)ps::kernel::launch_dflash2_target_upper_logits(
                proxy.coarse, proxy.act_l2, proxy.error_l2, proxy.upper, nrows, vocab, stream);
        };
        const uint32_t row_list[] = {1u, 2u, 4u, 8u};
        for (uint32_t nrows : row_list) {
            for (uint32_t i = 0u; i < warmup; ++i) {
                quant_once(nrows);
                l2_once(nrows);
                coarse_once(nrows);
                upper_once(nrows);
            }
            (void)hipStreamSynchronize(stream);
            Bench sb;
            sb.stream = stream;
            std::vector<double> q1_ms;
            std::vector<double> q2_ms;
            std::vector<double> q3_ms;
            std::vector<double> q4_ms;
            for (uint32_t i = 0u; i < iters; ++i) {
                sb.start();
                quant_once(nrows);
                q1_ms.push_back(sb.stop_ms());
                sb.start();
                quant_once(nrows);
                l2_once(nrows);
                q2_ms.push_back(sb.stop_ms());
                sb.start();
                quant_once(nrows);
                l2_once(nrows);
                coarse_once(nrows);
                q3_ms.push_back(sb.stop_ms());
                sb.start();
                quant_once(nrows);
                l2_once(nrows);
                coarse_once(nrows);
                upper_once(nrows);
                q4_ms.push_back(sb.stop_ms());
            }
            const double q1 = median(q1_ms);
            const double q2 = median(q2_ms);
            const double q3 = median(q3_ms);
            const double q4 = median(q4_ms);
            std::printf(
                "TARGET_LM_PROXY_SCORE_PERF rows=%u quant_us=%.3f l2_us=%.3f "
                "coarse_head_us=%.3f upper_us=%.3f sum_us=%.3f\n",
                nrows, q1 * 1000.0, (q2 - q1) * 1000.0, (q3 - q2) * 1000.0,
                (q4 - q3) * 1000.0, q4 * 1000.0);
        }
    }

    {
        const uint32_t nrows = proxy_rows;
        const uint32_t pool_n = 32u;
        const auto topn_once = [&]() {
            (void)ps::kernel::launch_dflash2_coarse_topn(
                proxy.upper, nrows, vocab, vocab, pool_n, proxy.top_ids, proxy.top_values,
                proxy.scratch_ids, proxy.scratch_values, proxy.scratch_stride,
                proxy.partitions, stream);
        };
        const auto merge_once = [&]() {
            (void)ps::kernel::launch_dflash2_coarse_topn_merge(
                proxy.scratch_ids, proxy.scratch_values, nrows, proxy.partitions, pool_n,
                proxy.scratch_stride, proxy.top_ids, proxy.top_values, stream);
        };
        for (uint32_t i = 0u; i < warmup; ++i)
            topn_once();
        (void)hipStreamSynchronize(stream);
        Bench tb;
        tb.stream = stream;
        std::vector<double> full_ms;
        std::vector<double> merge_ms;
        for (uint32_t i = 0u; i < iters; ++i) {
            tb.start();
            topn_once();
            full_ms.push_back(tb.stop_ms());
            topn_once();
            tb.start();
            merge_once();
            merge_ms.push_back(tb.stop_ms());
        }
        std::printf(
            "TARGET_LM_PROXY_TOPN_PERF rows=%u pool=%u topn_us=%.3f merge_us=%.3f\n", nrows,
            pool_n, median(full_ms) * 1000.0, median(merge_ms) * 1000.0);
    }

    (void)hipFree(full_act);
    (void)hipFree(full_act_scales);
    (void)hipFree(full_logits);
    (void)hipFree(full_sampled);
    (void)hipFree(perf_hidden);
    (void)hipFree(normed);
    proxy.shutdown();
    (void)ps::qwen35::executor_shutdown(ex);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    std::printf("test_target_lm_proxy_real_perf: done\n");
    return 0;
}
