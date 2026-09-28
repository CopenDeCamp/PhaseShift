#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

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

    const uint32_t chunk = 64u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t num_pages = ceil_div(chunk + 32u, page_tokens) + 8u;

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
    auto exec_result = ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool,
                                                         arena, ec);
    if (!exec_result.ok()) {
        std::printf("FAIL executor create: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Executor ex = exec_result.release();

    std::vector<int32_t> prompt(chunk);
    for (uint32_t i = 0; i < chunk; ++i)
        prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);

    auto seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, chunk + 32u, stream);
    if (!seq_result.ok()) {
        std::printf("FAIL sequence create\n");
        return 1;
    }
    ps::qwen35::PagedSequenceState seq = seq_result.release();

    std::vector<ps::qwen35::ScheduledRequest> requests;
    ps::qwen35::ScheduledRequest req;
    req.sequence = &seq;
    req.handle = seq.request_handle();
    req.execution_class = ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = chunk;
    req.prefix_tokens = 0u;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = chunk;
    requests.push_back(req);
    ps::qwen35::ScheduledBatch batch;
    batch.token_ids = prompt.data();
    batch.requests = requests;
    batch.num_tokens = chunk;
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
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
    const uint32_t vocab = out.logits.dim(1);
    std::printf("PROBE num_outputs=%u num_token_rows=%u\n", out.num_outputs, out.num_token_rows);
    std::printf("PROBE final_hidden dims=%u,%u  logits dims=%u,%u  sampled=%u\n",
                out.final_hidden.dim(0), out.final_hidden.dim(1), out.logits.dim(0),
                out.logits.dim(1), out.sampled_tokens.dim(0));

    std::vector<ps::bf16_t> hidden_host(static_cast<std::size_t>(out.final_hidden.dim(0)) * hidden);
    (void)hipMemcpy(hidden_host.data(), out.final_hidden.data<ps::bf16_t>(),
                    hidden_host.size() * sizeof(ps::bf16_t), hipMemcpyDeviceToHost);
    std::vector<float> logits(static_cast<std::size_t>(out.logits.dim(0)) * vocab);
    (void)hipMemcpy(logits.data(), out.logits.data<float>(), logits.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    std::vector<int32_t> sampled(out.sampled_tokens.dim(0));
    (void)hipMemcpy(sampled.data(), out.sampled_tokens.data<int32_t>(),
                    sampled.size() * sizeof(int32_t), hipMemcpyDeviceToHost);

    uint32_t match = 0u;
    for (uint32_t r = 0u; r < out.logits.dim(0); ++r) {
        const float* row = logits.data() + static_cast<std::size_t>(r) * vocab;
        int32_t best = 0;
        for (uint32_t v = 1u; v < vocab; ++v) {
            if (row[v] > row[best]) best = static_cast<int32_t>(v);
        }
        if (best == sampled[r]) ++match;
    }
    std::printf("PROBE sampled==logits_argmax %u/%u\n", match, out.logits.dim(0));

    (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    (void)ps::qwen35::executor_shutdown(ex);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    std::printf("test_target_lm_proxy_real_probe: done\n");
    return 0;
}
