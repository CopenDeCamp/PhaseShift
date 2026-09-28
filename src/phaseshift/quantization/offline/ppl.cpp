#include <phaseshift/quantization/offline/ppl.h>
#include <phaseshift/quantization/offline/token_corpus.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/core/memory/arena.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace ps::quantization::fpx {

namespace {

namespace RT = ::ps::runtime;

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types)
        if (lt != 0) ++n;
    return n;
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

double log_softmax_at(const std::vector<float>& logits, int32_t target) {
    double max_value = logits[0];
    for (float v : logits) {
        if (v > max_value) max_value = v;
    }
    double sum = 0.0;
    for (float v : logits) sum += std::exp(static_cast<double>(v) - max_value);
    const double target_value = static_cast<double>(logits[static_cast<std::size_t>(target)]);
    return target_value - max_value - std::log(sum);
}

}

Result<PplSummary> evaluate_ppl(const PplOptions& options) {
    if (options.model_dir.empty())
        return Status::invalid_argument("model dir is required", __FILE__, __LINE__);
    if (options.tokens_path.empty())
        return Status::invalid_argument("tokens path is required", __FILE__, __LINE__);

    auto corpus_res = read_token_corpus(options.tokens_path);
    if (!corpus_res.ok()) return corpus_res.status();
    TokenCorpus corpus = corpus_res.release();
    if (corpus.tokens.size() < 2)
        return Status::invalid_argument("corpus has too few tokens", __FILE__, __LINE__);

    auto config_res = ps::qwen35::read_qwen35_text_config(options.model_dir);
    if (!config_res.ok()) return config_res.status();
    const ps::qwen35::Qwen35TextConfig tc = config_res.value();

    if (hipSetDevice(options.device) != hipSuccess)
        return Status::hip_error("hipSetDevice", hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return Status::hip_error("hipStreamCreate", hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);

    auto arena_res = ps::gpu::GpuArena::create(
        options.device, static_cast<std::size_t>(options.arena_gib * 1024.0 * 1024.0 * 1024.0));
    if (!arena_res.ok()) return arena_res.status();
    ps::gpu::GpuArena arena = arena_res.release();

    ps::qwen35::Qwen35LoadOptions load_opts;
    auto model_res = ps::qwen35::Qwen35Model::load_from_safetensors(options.model_dir, arena,
                                                                   stream, load_opts);
    if (!model_res.ok()) return model_res.status();
    ps::qwen35::Qwen35Model model = model_res.release();

    const uint32_t max_seq_len = static_cast<uint32_t>(
        std::min<std::size_t>(options.max_tokens + 2u, corpus.tokens.size()));
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = ceil_div(max_seq_len, 16u) + 2u;

    auto seq_pool_res = ps::qwen35::SequenceSlotPool::create(arena, 1, num_pages);
    if (!seq_pool_res.ok()) return seq_pool_res.status();
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_res.release();

    auto gdn_pool_res = ps::qwen35::GdnStatePool::create(
        arena, 1, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    if (!gdn_pool_res.ok()) return gdn_pool_res.status();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_res.release();

    auto kv_pool_res = ps::qwen35::PagedKVPool::create(arena, num_pages, 16u, attn_layers, kv_heads,
                                                       head_dim, ps::qwen35::KVCacheDType::BF16);
    if (!kv_pool_res.ok()) return kv_pool_res.status();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_res.release();

    ps::qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = 16;
    exec_config.max_scheduled_requests = 1;

    auto exec_res =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_res.ok()) return exec_res.status();
    ps::qwen35::Executor executor = exec_res.release();

    auto seq_res = ps::qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq_len,
                                                          stream);
    if (!seq_res.ok()) return seq_res.status();
    ps::qwen35::PagedSequenceState seq = seq_res.release();

    const std::size_t vocab_size = tc.vocab_size;
    std::vector<float> logits(vocab_size);
    std::vector<float> prev_logits(vocab_size);
    double nll_sum = 0.0;
    uint64_t counted = 0;
    PplSummary summary;

    std::printf("[ppl] tokens=%zu model=%s\n", static_cast<std::size_t>(max_seq_len),
                options.model_dir.c_str());
    for (uint32_t i = 0; i < max_seq_len; ++i) {
        std::vector<int32_t> token_ids = {corpus.tokens[i]};
        const bool want_logits = i + 1u < max_seq_len;
        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = RT::ExecutionClass::DECODE;
        req.token_begin = 0;
        req.num_tokens = 1;
        req.prefix_tokens = seq.position;
        req.compute_logits = want_logits;
        req.sample = false;
        std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.requests = reqs;
        batch.num_tokens = 1;
        batch.num_requests = 1;
        batch.num_decode_requests = 1;
        batch.num_verify_requests = 0;
        batch.num_prefill_requests = 0;
        auto res = ps::qwen35::execute_batch(executor, batch, stream);
        if (!res.ok()) return res.status();
        ps::qwen35::BatchExecutionOutput out = res.release();
        if (i > 0) {
            nll_sum -= log_softmax_at(prev_logits, corpus.tokens[i]);
            ++counted;
        }
        if (want_logits) {
            if (out.num_outputs != 1)
                return Status::invalid_state("ppl expects a single output row", __FILE__, __LINE__);
            if (hipMemcpy(logits.data(), out.logits.data<float>(), vocab_size * sizeof(float),
                          hipMemcpyDeviceToHost) != hipSuccess)
                return Status::hip_error("ppl logits copy", hipGetErrorString(hipGetLastError()),
                                         __FILE__, __LINE__);
            prev_logits.swap(logits);
            if (i == 0) {
                int argmax = 0;
                for (std::size_t v = 1; v < vocab_size; ++v)
                    if (prev_logits[v] > prev_logits[static_cast<std::size_t>(argmax)]) argmax = int(v);
                summary.argmax_first = argmax;
            }
        }
        if ((i + 1u) % 512u == 0u) {
            std::printf("[ppl] %u / %u mean_nll=%.6f\n", i + 1u, max_seq_len,
                        counted > 0 ? nll_sum / static_cast<double>(counted) : 0.0);
            std::fflush(stdout);
        }
    }

    summary.positions = counted;
    summary.mean_nll = counted > 0 ? nll_sum / static_cast<double>(counted) : 0.0;
    summary.perplexity = std::exp(summary.mean_nll);
    return summary;
}

}
