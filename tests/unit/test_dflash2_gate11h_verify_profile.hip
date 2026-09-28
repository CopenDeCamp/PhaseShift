#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/decode_perf_stats.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/gdn_spec_history.h>
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

using ps::qwen35::dflash2::DFlash2Config;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

static std::vector<uint32_t> parse_u32_list(const char* text) {
    std::vector<uint32_t> out;
    if (text == nullptr) return out;
    std::string s(text);
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(static_cast<uint32_t>(std::stoul(cur)));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(static_cast<uint32_t>(std::stoul(cur)));
    return out;
}

static double time_median_us(hipStream_t stream, uint32_t iters, const std::function<void()>& fn) {
    for (uint32_t i = 0; i < 10u; ++i) fn();
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

static ps::qwen35::ScheduledBatch make_batch(
    ps::qwen35::PagedSequenceState& sequence,
    std::vector<ps::qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    ps::runtime::VerifyNumericMode numeric_mode) {
    ps::qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ps::runtime::ExecutionClass::DECODE
                                           : ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = sequence.position;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);
    ps::qwen35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = num_tokens == 1u ? 1u : 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = num_tokens == 1u ? 0u : 1u;
    batch.speculative_verify = speculative_verify;
    batch.verify_numeric_mode = numeric_mode;
    return batch;
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

    std::vector<uint32_t> ctx_list = parse_u32_list(std::getenv("PHASESHIFT_GATE11H_CTX"));
    if (ctx_list.empty()) ctx_list = {2048u};
    std::vector<uint32_t> m_list = parse_u32_list(std::getenv("PHASESHIFT_GATE11H_M"));
    if (m_list.empty()) m_list = {1u, 2u, 4u, 8u};
    const char* iters_env = std::getenv("PHASESHIFT_GATE11H_ITERS");
    const uint32_t iters =
        iters_env != nullptr ? static_cast<uint32_t>(std::stoul(iters_env)) : 30u;
    const char* capture_env = std::getenv("PHASESHIFT_GATE11H_CAPTURE");
    const bool capture =
        capture_env != nullptr && capture_env[0] != '\0' && capture_env[0] != '0';
    const char* dump_env = std::getenv("PHASESHIFT_GATE11H_DUMP");

    const uint32_t max_ctx = *std::max_element(ctx_list.begin(), ctx_list.end());
    auto arena_result = ps::gpu::GpuArena::create(0, 31ull * 1024ull * 1024ull * 1024ull);
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

    auto model_result = ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(model_result.ok(), "target load");
    if (!model_result.ok()) return 1;
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();

    const uint32_t required_seq_len = max_ctx + 16u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = ceil_div(required_seq_len, page_tokens) + 24u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 1u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 1u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    for (uint32_t t = 0; t < config.num_target_layer_ids; ++t) {
        taps[t] = static_cast<uint32_t>(config.target_layer_ids[t]);
    }
    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = std::max(256u, max_ctx + 16u);
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = static_cast<uint32_t>(config.block_size);
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = static_cast<uint32_t>(config.num_target_layer_ids);
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    ps::qwen35::Executor ex = exec_result.release();

    auto seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, required_seq_len + 8u, stream);
    check(seq_result.ok(), "sequence create");
    if (!seq_result.ok()) return 1;
    ps::qwen35::PagedSequenceState sequence = seq_result.release();

    ps::qwen35::runtime::GdnSpecHistory history{};
    if (capture) {
        auto history_result = ps::qwen35::runtime::create_gdn_spec_history(arena, gdn_pool, 8u);
        check(history_result.ok(), "history create");
        if (!history_result.ok()) return 1;
        history = history_result.release();
    }

    std::printf("GATE11H_PROFILE attn_layers=%u hidden=%u inter=%u capture=%u iters=%u\n",
                num_attn_layers, static_cast<unsigned>(tc.hidden_size),
                static_cast<unsigned>(tc.intermediate_size), capture ? 1u : 0u, iters);

    std::vector<int32_t> verify_tokens(8u);
    for (uint32_t i = 0; i < 8u; ++i) {
        verify_tokens[i] = static_cast<int32_t>(200u + i);
    }

    for (uint32_t ctx : ctx_list) {
        {
            std::vector<int32_t> prompt(ctx);
            for (uint32_t i = 0; i < ctx; ++i) {
                prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
            }
            std::vector<ps::qwen35::ScheduledRequest> requests;
            ps::qwen35::ScheduledBatch batch =
                make_batch(sequence, requests, prompt.data(), ctx, 1u, false,
                           ps::runtime::VerifyNumericMode::Fast);
            auto executed = ps::qwen35::execute_batch(ex, batch, stream);
            if (!executed.ok()) {
                std::printf("FAIL prefill ctx=%u %s\n", ctx,
                            executed.status().message().c_str());
            }
            (void)executed.release();
            (void)hipStreamSynchronize(stream);
        }

        for (uint32_t m : m_list) {
            const uint32_t saved = sequence.position;
            const double us = time_median_us(stream, iters, [&]() {
                sequence.position = saved;
                std::vector<ps::qwen35::ScheduledRequest> requests;
                ps::qwen35::ScheduledBatch batch =
                    make_batch(sequence, requests, verify_tokens.data(), m, m, true,
                               ps::runtime::VerifyNumericMode::Exact);
                if (capture) {
                    ps::qwen35::ExecuteBatchOptions options;
                    options.gdn_spec_history =
                        ps::qwen35::runtime::gdn_spec_history_view(history, m);
                    auto executed = ps::qwen35::execute_batch(ex, batch, stream, options);
                    (void)executed.status();
                } else {
                    auto executed = ps::qwen35::execute_batch(ex, batch, stream);
                    (void)executed.status();
                }
            });
            std::printf("GATE11H_TOTAL ctx=%u M=%u capture=%u verify_us=%.2f\n", ctx, m,
                        capture ? 1u : 0u, us);

            ps::qwen35::runtime::perf_stats_reset();
            {
                sequence.position = saved;
                std::vector<ps::qwen35::ScheduledRequest> requests;
                ps::qwen35::ScheduledBatch batch =
                    make_batch(sequence, requests, verify_tokens.data(), m, m, true,
                               ps::runtime::VerifyNumericMode::Exact);
                if (capture) {
                    ps::qwen35::ExecuteBatchOptions options;
                    options.gdn_spec_history =
                        ps::qwen35::runtime::gdn_spec_history_view(history, m);
                    auto executed = ps::qwen35::execute_batch(ex, batch, stream, options);
                    (void)executed.status();
                } else {
                    auto executed = ps::qwen35::execute_batch(ex, batch, stream);
                    (void)executed.status();
                }
                (void)hipStreamSynchronize(stream);
            }
            const ps::qwen35::runtime::DecodePerfStats& st =
                ps::qwen35::runtime::perf_stats_instance();
            std::printf("GATE11H_LAUNCH ctx=%u M=%u logical=%llu optimized=%llu "
                        "fallback=%llu physical=%llu\n",
                        ctx, m, static_cast<unsigned long long>(st.logical_dispatches),
                        static_cast<unsigned long long>(st.optimized_operations),
                        static_cast<unsigned long long>(st.correctness_fallbacks),
                        static_cast<unsigned long long>(st.physical_kernel_launches));
        }
    }

    if (dump_env != nullptr && dump_env[0] != '\0') {
        ps::qwen35::runtime::perf_stats_dump(dump_env, 0u);
    }

    if (capture) {
        check(ps::qwen35::runtime::shutdown_gdn_spec_history(history).ok(), "history shutdown");
    }
    check(ps::qwen35::release_paged_sequence_state(sequence, stream).ok(), "sequence release");
    check(ps::qwen35::executor_shutdown(ex).ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate11h_verify_profile: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
