#include "support/target_lm_proxy_harness.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/runtime/spec_decode.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

struct Rng {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<uint32_t>(state >> 11);
    }
    uint32_t below(uint32_t n) { return next() % (n == 0u ? 1u : n); }
};

struct Metrics {
    uint64_t rounds = 0u;
    uint64_t draft_rows = 0u;
    uint64_t direct_accept_rows = 0u;
    uint64_t oracle_accepted = 0u;
    uint64_t generic32_rows = 0u;
    uint64_t generic128_rows = 0u;
    uint64_t unknown_rows = 0u;
    uint64_t ready_rounds = 0u;
    uint64_t full_accept_rounds = 0u;
    uint64_t fallback_rounds = 0u;
    uint64_t false_accept = 0u;
    uint64_t argmax_mismatch = 0u;
    uint64_t decision_mismatch = 0u;
    uint64_t sample_argmax_mismatch = 0u;
    uint64_t proxy_rows = 0u;
    uint64_t proxy_top1_mismatch = 0u;
    uint64_t proxy_top1_mismatch127 = 0u;
    uint64_t proxy_false_accept = 0u;
    uint64_t proxy_false_reject = 0u;
};

int g_fail = 0;
void check(bool cond, const char* label) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL: %s\n", label);
    }
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

    const uint32_t k = std::getenv("PHASESHIFT_VERIFY_SHADOW_K") != nullptr
                           ? static_cast<uint32_t>(std::strtoul(
                                 std::getenv("PHASESHIFT_VERIFY_SHADOW_K"), nullptr, 10))
                           : 7u;
    const uint32_t rounds = std::getenv("PHASESHIFT_VERIFY_SHADOW_ROUNDS") != nullptr
                                ? static_cast<uint32_t>(std::strtoul(
                                      std::getenv("PHASESHIFT_VERIFY_SHADOW_ROUNDS"), nullptr, 10))
                                : 256u;
    const uint32_t ctx = std::getenv("PHASESHIFT_VERIFY_SHADOW_CTX") != nullptr
                             ? static_cast<uint32_t>(std::strtoul(
                                   std::getenv("PHASESHIFT_VERIFY_SHADOW_CTX"), nullptr, 10))
                             : 512u;
    const float perturb = std::getenv("PHASESHIFT_VERIFY_SHADOW_PERTURB") != nullptr
                              ? static_cast<float>(std::strtod(
                                    std::getenv("PHASESHIFT_VERIFY_SHADOW_PERTURB"), nullptr))
                              : 0.12f;
    const float margin_abs = 1e-3f;
    const float margin_rel = 1e-5f;
    const uint32_t pool0 = 32u;
    const uint32_t rows = k + 1u;
    if (k == 0u || k > 15u) {
        std::printf("FAIL: k out of range\n");
        return 1;
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
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);
    const uint32_t vocab = static_cast<uint32_t>(model.weights().lm_head.rows);
    const float eps = tc.rms_norm_eps;

    const uint32_t max_seq = ctx + rounds * rows + 64u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = ceil_div(max_seq, page_tokens) + 8u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 1u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 1u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
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
    ec.max_scheduled_tokens = std::max(ctx + 16u, rows + 16u);
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = rows;
    auto exec_result = ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    if (!exec_result.ok()) {
        std::printf("FAIL executor create: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Executor ex = exec_result.release();

    auto seq_result =
        ps::qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!seq_result.ok()) {
        std::printf("FAIL sequence create\n");
        return 1;
    }
    ps::qwen35::PagedSequenceState seq = seq_result.release();

    target_lm_proxy::Proxy proxy;
    std::string perr;
    if (!proxy.init(model.weights().lm_head, rows, 127u, stream, perr)) {
        std::printf("FAIL proxy init: %s\n", perr.c_str());
        return 1;
    }

    ps::bf16_t* normed = nullptr;
    if (hipMalloc(&normed, static_cast<std::size_t>(rows) * hidden * sizeof(ps::bf16_t)) !=
        hipSuccess) {
        std::printf("FAIL normed alloc\n");
        return 1;
    }

    const ps::gpu::Tensor& fw = model.weights().final_norm_weight;
    const ps::kernel::RmsNormDataType wdt =
        fw.element_size() == 4u ? ps::kernel::RmsNormDataType::F32
                                : ps::kernel::RmsNormDataType::BF16;

    std::vector<int32_t> prompt(ctx);
    const char* prompt_file = std::getenv("PHASESHIFT_VERIFY_SHADOW_PROMPT");
    if (prompt_file != nullptr && prompt_file[0] != '\0') {
        FILE* pf = std::fopen(prompt_file, "r");
        if (pf == nullptr) {
            std::printf("FAIL open prompt %s\n", prompt_file);
            return 1;
        }
        for (uint32_t i = 0u; i < ctx; ++i) {
            int32_t t = 0;
            if (std::fscanf(pf, "%d", &t) != 1) {
                std::fclose(pf);
                std::printf("FAIL short prompt\n");
                return 1;
            }
            prompt[i] = t;
        }
        std::fclose(pf);
    } else {
        for (uint32_t i = 0u; i < ctx; ++i)
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    }

    {
        std::vector<ps::qwen35::ScheduledRequest> requests;
        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = ps::runtime::ExecutionClass::PREFILL;
        req.token_begin = 0u;
        req.num_tokens = ctx;
        req.prefix_tokens = 0u;
        req.compute_logits = true;
        req.sample = true;
        req.num_output_rows = 1u;
        requests.push_back(req);
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = prompt.data();
        batch.requests = requests;
        batch.num_tokens = ctx;
        batch.num_requests = 1u;
        batch.num_prefill_requests = 1u;
        auto executed = ps::qwen35::execute_batch(ex, batch, stream);
        if (!executed.ok()) {
            std::printf("FAIL prefill: %s\n", executed.status().message().c_str());
            return 1;
        }
        (void)executed.release();
        (void)hipStreamSynchronize(stream);
    }

    Rng rng;
    int32_t pending = static_cast<int32_t>(100u + rng.below(vocab - 200u));
    std::vector<int32_t> drafts(k, 0);
    for (uint32_t i = 0u; i < k; ++i)
        drafts[i] = static_cast<int32_t>(100u + rng.below(vocab - 200u));

    std::vector<int32_t> verify_tokens(rows, 0);
    std::vector<int32_t> proposed(rows, 0);
    std::vector<int32_t> samples(rows, 0);
    std::vector<int32_t> oracle(rows, 0);
    std::vector<int32_t> best_ids(rows, 0);
    std::vector<uint8_t> generic_cert(rows, 0u);
    std::vector<uint8_t> direct_cert(rows, 0u);
    std::vector<uint8_t> row_state(rows, 0u);
    std::vector<int32_t> cert_tokens(rows, 0);
    std::vector<int32_t> cand_ids(rows, 0);
    std::vector<int32_t> cand_ids127(rows, 0);

    Metrics m;
    bool probe_done = false;

    for (uint32_t round = 0u; round < rounds; ++round) {
        verify_tokens[0] = pending;
        for (uint32_t i = 0u; i < k; ++i) verify_tokens[i + 1u] = drafts[i];

        std::vector<ps::qwen35::ScheduledRequest> requests;
        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = ps::runtime::ExecutionClass::PREFILL;
        req.token_begin = 0u;
        req.num_tokens = rows;
        req.prefix_tokens = seq.position;
        req.compute_logits = true;
        req.sample = true;
        req.num_output_rows = rows;
        requests.push_back(req);
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = verify_tokens.data();
        batch.requests = requests;
        batch.num_tokens = rows;
        batch.num_requests = 1u;
        batch.num_prefill_requests = 1u;
        batch.speculative_verify = true;
        batch.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;

        auto executed = ps::qwen35::execute_batch(ex, batch, stream);
        if (!executed.ok()) {
            std::printf("FAIL verify: %s\n", executed.status().message().c_str());
            return 1;
        }
        ps::qwen35::BatchExecutionOutput out = executed.release();
        const uint32_t out_rows = out.sampled_tokens.dim(0);
        if (out_rows < rows) {
            std::printf("FAIL verify out rows=%u want=%u\n", out_rows, rows);
            return 1;
        }
        (void)hipMemcpy(samples.data(), out.sampled_tokens.data<int32_t>(),
                        static_cast<std::size_t>(rows) * sizeof(int32_t), hipMemcpyDeviceToHost);

        std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
        if (out.logits.data<float>() != nullptr && out.logits.dim(1) == vocab) {
            (void)hipMemcpy(logits.data(), out.logits.data<float>(),
                            logits.size() * sizeof(float), hipMemcpyDeviceToHost);
        }

        const hipError_t rerr = ps::kernel::launch_rmsnorm(
            out.final_hidden.data<void>(), ps::kernel::RmsNormDataType::BF16, fw.data<void>(),
            wdt, ps::kernel::RmsNormWeightLayout::PER_FEATURE,
            ps::kernel::RmsNormWeightMode::ONE_PLUS, normed,
            ps::kernel::RmsNormDataType::BF16, rows, hidden, hidden, hidden, hidden, eps, stream);
        if (rerr != hipSuccess) {
            std::printf("FAIL rmsnorm\n");
            return 1;
        }
        if (!proxy.prepare(normed, rows, hidden, stream, perr)) {
            std::printf("FAIL proxy prepare: %s\n", perr.c_str());
            return 1;
        }

        for (uint32_t i = 0u; i < k; ++i) proposed[i] = drafts[i];
        proposed[k] = 0;
        if (!proxy.verify_direct(rows, proposed.data(), margin_abs, margin_rel, stream, perr)) {
            std::printf("FAIL proxy verify_direct: %s\n", perr.c_str());
            return 1;
        }
        (void)hipMemcpy(direct_cert.data(), proxy.certified, rows, hipMemcpyDeviceToHost);

        if (!proxy.topn(rows, pool0 + 1u, stream, perr) ||
            !proxy.rerank_pool(rows, pool0, pool0 + 1u, stream, perr) ||
            !proxy.certify(rows, pool0, pool0 + 1u, margin_abs, margin_rel, stream, perr)) {
            std::printf("FAIL proxy generic: %s\n", perr.c_str());
            return 1;
        }
        (void)hipMemcpy(best_ids.data(), proxy.best_ids, rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(generic_cert.data(), proxy.certified, rows, hipMemcpyDeviceToHost);

        if (!proxy.topn(rows, 128u, stream, perr) ||
            !proxy.rerank_pool(rows, 127u, 128u, stream, perr) ||
            !proxy.certify(rows, 127u, 128u, margin_abs, margin_rel, stream, perr)) {
            std::printf("FAIL proxy generic128: %s\n", perr.c_str());
            return 1;
        }
        std::vector<int32_t> best_ids127(rows, 0);
        std::vector<uint8_t> generic_cert127(rows, 0u);
        (void)hipMemcpy(best_ids127.data(), proxy.best_ids, rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(generic_cert127.data(), proxy.certified, rows, hipMemcpyDeviceToHost);

        if (!proxy.topn_on(proxy.coarse, rows, pool0, stream, perr) ||
            !proxy.rerank_pool(rows, pool0, pool0, stream, perr) ||
            !proxy.certify(rows, pool0, pool0, margin_abs, margin_rel, stream, perr)) {
            std::printf("FAIL proxy cand32: %s\n", perr.c_str());
            return 1;
        }
        (void)hipMemcpy(cand_ids.data(), proxy.best_ids, rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);

        if (!proxy.topn_on(proxy.coarse, rows, 128u, stream, perr) ||
            !proxy.rerank_pool(rows, 127u, 128u, stream, perr) ||
            !proxy.certify(rows, 127u, 128u, margin_abs, margin_rel, stream, perr)) {
            std::printf("FAIL proxy cand128: %s\n", perr.c_str());
            return 1;
        }
        (void)hipMemcpy(cand_ids127.data(), proxy.best_ids, rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);

        for (uint32_t i = 0u; i < rows; ++i) {
            if (logits.empty()) {
                oracle[i] = samples[i];
            } else {
                const float* row = logits.data() + static_cast<std::size_t>(i) * vocab;
                int32_t best = 0;
                for (uint32_t v = 1u; v < vocab; ++v)
                    if (row[v] > row[best]) best = static_cast<int32_t>(v);
                oracle[i] = best;
                if (best != samples[i]) ++m.sample_argmax_mismatch;
            }
        }

        for (uint32_t i = 0u; i < k; ++i) {
            ++m.draft_rows;
            if (direct_cert[i] != 0u) {
                ++m.direct_accept_rows;
                if (oracle[i] != drafts[i]) {
                    ++m.false_accept;
                }
                row_state[i] = ps::kernel::kDflash2VerifyRowAcceptCertified;
                cert_tokens[i] = drafts[i];
            } else if (generic_cert[i] != 0u) {
                ++m.generic32_rows;
                row_state[i] = ps::kernel::kDflash2VerifyRowArgmaxCertified;
                cert_tokens[i] = best_ids[i];
                if (best_ids[i] != oracle[i]) ++m.argmax_mismatch;
            } else if (generic_cert127[i] != 0u) {
                ++m.generic128_rows;
                row_state[i] = ps::kernel::kDflash2VerifyRowArgmaxCertified;
                cert_tokens[i] = best_ids127[i];
                if (best_ids127[i] != oracle[i]) ++m.argmax_mismatch;
            } else {
                ++m.unknown_rows;
                row_state[i] = ps::kernel::kDflash2VerifyRowUnknown;
                cert_tokens[i] = -1;
            }
        }
        if (generic_cert[k] != 0u) {
            ++m.generic32_rows;
            row_state[k] = ps::kernel::kDflash2VerifyRowArgmaxCertified;
            cert_tokens[k] = best_ids[k];
            if (best_ids[k] != oracle[k]) ++m.argmax_mismatch;
        } else if (generic_cert127[k] != 0u) {
            ++m.generic128_rows;
            row_state[k] = ps::kernel::kDflash2VerifyRowArgmaxCertified;
            cert_tokens[k] = best_ids127[k];
            if (best_ids127[k] != oracle[k]) ++m.argmax_mismatch;
        } else {
            ++m.unknown_rows;
            row_state[k] = ps::kernel::kDflash2VerifyRowUnknown;
            cert_tokens[k] = -1;
        }

        for (uint32_t i = 0u; i < k; ++i) {
            ++m.proxy_rows;
            if (cand_ids[i] != oracle[i])
                ++m.proxy_top1_mismatch;
            if (cand_ids127[i] != oracle[i])
                ++m.proxy_top1_mismatch127;
            if (cand_ids[i] == drafts[i]) {
                if (oracle[i] != drafts[i])
                    ++m.proxy_false_accept;
            } else if (oracle[i] == drafts[i]) {
                ++m.proxy_false_reject;
            }
        }

        int32_t* d_drafts = nullptr;
        uint8_t* d_state = nullptr;
        int32_t* d_cert = nullptr;
        uint32_t* d_ready = nullptr;
        uint32_t* d_accepted = nullptr;
        uint32_t* d_corr_valid = nullptr;
        int32_t* d_corr = nullptr;
        uint32_t* d_bonus_valid = nullptr;
        int32_t* d_bonus = nullptr;
        int32_t* d_first = nullptr;
        (void)hipMalloc(&d_drafts, k * sizeof(int32_t));
        (void)hipMalloc(&d_state, rows);
        (void)hipMalloc(&d_cert, rows * sizeof(int32_t));
        (void)hipMalloc(&d_ready, sizeof(uint32_t));
        (void)hipMalloc(&d_accepted, sizeof(uint32_t));
        (void)hipMalloc(&d_corr_valid, sizeof(uint32_t));
        (void)hipMalloc(&d_corr, sizeof(int32_t));
        (void)hipMalloc(&d_bonus_valid, sizeof(uint32_t));
        (void)hipMalloc(&d_bonus, sizeof(int32_t));
        (void)hipMalloc(&d_first, sizeof(int32_t));
        (void)hipMemcpy(d_drafts, drafts.data(), k * sizeof(int32_t), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_state, row_state.data(), rows, hipMemcpyHostToDevice);
        (void)hipMemcpy(d_cert, cert_tokens.data(), rows * sizeof(int32_t),
                        hipMemcpyHostToDevice);
        (void)ps::kernel::launch_dflash2_verify_decision_sufficiency(
            d_drafts, d_state, d_cert, k, d_ready, d_accepted, d_corr_valid, d_corr,
            d_bonus_valid, d_bonus, d_first, stream);
        uint32_t ready = 0u, accepted = 0u, corr_valid = 0u, bonus_valid = 0u;
        int32_t corr = -1, bonus = -1;
        (void)hipMemcpy(&ready, d_ready, sizeof(uint32_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&accepted, d_accepted, sizeof(uint32_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&corr_valid, d_corr_valid, sizeof(uint32_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&corr, d_corr, sizeof(int32_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&bonus_valid, d_bonus_valid, sizeof(uint32_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(&bonus, d_bonus, sizeof(int32_t), hipMemcpyDeviceToHost);
        (void)hipFree(d_drafts);
        (void)hipFree(d_state);
        (void)hipFree(d_cert);
        (void)hipFree(d_ready);
        (void)hipFree(d_accepted);
        (void)hipFree(d_corr_valid);
        (void)hipFree(d_corr);
        (void)hipFree(d_bonus_valid);
        (void)hipFree(d_bonus);
        (void)hipFree(d_first);

        ps::qwen35::runtime::SpecVerifyResult prod{};
        ps::Status st = ps::qwen35::runtime::spec_greedy_accept(drafts.data(), k, samples.data(),
                                                                true, prod);
        if (!st.ok()) {
            std::printf("FAIL spec accept: %s\n", st.message().c_str());
            return 1;
        }
        ++m.rounds;
        m.oracle_accepted += prod.num_accepted_drafts;
        if (prod.num_accepted_drafts == k) ++m.full_accept_rounds;

        if (ready != 0u) {
            ++m.ready_rounds;
            bool ok = accepted == prod.num_accepted_drafts;
            if (prod.correction_valid) {
                ok = ok && corr_valid != 0u && corr == prod.correction_token &&
                     bonus_valid == 0u;
            } else {
                ok = ok && bonus_valid != 0u && bonus == prod.bonus_token &&
                     corr_valid == 0u;
            }
            if (!ok) ++m.decision_mismatch;
        } else {
            ++m.fallback_rounds;
        }

        if (!probe_done) {
            std::printf("VERIFY_SHADOW_PROBE k=%u rows=%u logits_rows=%u hidden_rows=%u "
                        "sample_argmax_mismatch=%llu\n",
                        k, rows, out.logits.dim(0), out.final_hidden.dim(0),
                        static_cast<unsigned long long>(m.sample_argmax_mismatch));
            probe_done = true;
        }

        pending = samples[0];
        for (uint32_t i = 0u; i < k; ++i) drafts[i] = samples[i + 1u];
        for (uint32_t i = 0u; i < k; ++i) {
            if (perturb <= 0.0f || logits.empty())
                continue;
            if (static_cast<float>(rng.below(1000u)) / 1000.0f >= perturb)
                continue;
            const float* row = logits.data() + static_cast<std::size_t>(i + 1u) * vocab;
            const int32_t best = oracle[i + 1u];
            int32_t second = -1;
            float second_v = -INFINITY;
            for (uint32_t v = 0u; v < vocab; ++v) {
                if (static_cast<int32_t>(v) == best)
                    continue;
                if (row[v] > second_v) {
                    second_v = row[v];
                    second = static_cast<int32_t>(v);
                }
            }
            if (second >= 0)
                drafts[i] = second;
        }
    }

    (void)hipStreamSynchronize(stream);

    const double direct_rate =
        m.draft_rows > 0u ? static_cast<double>(m.direct_accept_rows) / m.draft_rows : 0.0;
    const double ready_rate =
        m.rounds > 0u ? static_cast<double>(m.ready_rounds) / m.rounds : 0.0;
    std::printf(
        "VERIFY_LM_PROXY rounds=%llu draft_rows=%llu direct_cert_rows=%llu direct_cert_rate=%.4f "
        "oracle_accepted=%llu generic32_rows=%llu generic128_rows=%llu unknown_rows=%llu "
        "proxy_decision_ready_rounds=%llu proxy_decision_ready_rate=%.4f "
        "full_accept_rounds=%llu full_fallback_rounds=%llu full_head_avoid_rate=%.4f "
        "false_accept=%llu argmax_mismatch=%llu decision_mismatch=%llu "
        "sample_argmax_mismatch=%llu k=%u rows=%u pool0=%u\n",
        static_cast<unsigned long long>(m.rounds),
        static_cast<unsigned long long>(m.draft_rows),
        static_cast<unsigned long long>(m.direct_accept_rows), direct_rate,
        static_cast<unsigned long long>(m.oracle_accepted),
        static_cast<unsigned long long>(m.generic32_rows),
        static_cast<unsigned long long>(m.generic128_rows),
        static_cast<unsigned long long>(m.unknown_rows),
        static_cast<unsigned long long>(m.ready_rounds), ready_rate,
        static_cast<unsigned long long>(m.full_accept_rounds),
        static_cast<unsigned long long>(m.fallback_rounds), ready_rate,
        static_cast<unsigned long long>(m.false_accept),
        static_cast<unsigned long long>(m.argmax_mismatch),
        static_cast<unsigned long long>(m.decision_mismatch),
        static_cast<unsigned long long>(m.sample_argmax_mismatch), k, rows, pool0);

    const double proxy_mismatch_rate =
        m.proxy_rows > 0u ? static_cast<double>(m.proxy_top1_mismatch) / m.proxy_rows : 0.0;
    const double proxy_mismatch127_rate =
        m.proxy_rows > 0u ? static_cast<double>(m.proxy_top1_mismatch127) / m.proxy_rows : 0.0;
    const double proxy_fa_rate =
        m.proxy_rows > 0u ? static_cast<double>(m.proxy_false_accept) / m.proxy_rows : 0.0;
    const double proxy_fr_rate =
        m.proxy_rows > 0u ? static_cast<double>(m.proxy_false_reject) / m.proxy_rows : 0.0;
    std::printf(
        "VERIFY_LM_PROXY_PROXY proxy_rows=%llu cand32_mismatch=%llu cand32_mismatch_rate=%.6f "
        "cand128_mismatch=%llu cand128_mismatch_rate=%.6f proxy_false_accept=%llu "
        "proxy_false_accept_rate=%.6f proxy_false_reject=%llu proxy_false_reject_rate=%.6f "
        "pool=%u k=%u rows=%u\n",
        static_cast<unsigned long long>(m.proxy_rows),
        static_cast<unsigned long long>(m.proxy_top1_mismatch), proxy_mismatch_rate,
        static_cast<unsigned long long>(m.proxy_top1_mismatch127), proxy_mismatch127_rate,
        static_cast<unsigned long long>(m.proxy_false_accept), proxy_fa_rate,
        static_cast<unsigned long long>(m.proxy_false_reject), proxy_fr_rate, pool0, k, rows);

    check(m.false_accept == 0u, "false_accept");
    check(m.argmax_mismatch == 0u, "argmax_mismatch");
    check(m.decision_mismatch == 0u, "decision_mismatch");
    check(m.sample_argmax_mismatch == 0u, "sample_argmax_mismatch");

    (void)hipFree(normed);
    proxy.shutdown();
    (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    (void)ps::qwen35::executor_shutdown(ex);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    std::printf("test_verify_lm_proxy_real_verify: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
