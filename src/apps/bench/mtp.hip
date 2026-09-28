#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
        "Usage: phaseshift-bench mtp [options]\n"
        "  --model-dir PATH             (required)\n"
        "  --device N                   (default 0)\n"
        "  --arena-gib N                (default 8)\n"
        "  --context N                  prompt tokens (default 0)\n"
        "  --steps N                    target decode steps (default 32)\n"
        "  --draft-k N                  drafts per verify round (default 4)\n"
        "  --token-id N                 (default 1)\n"
        "  --tokens-file PATH           PSKLDTOK corpus used as the prompt\n"
        "  --tokens-file PATH           PSKLDTOK corpus used as the prompt\n"
        "  --tokens-offset N            corpus position to start the prompt (default 0)\n"
        "  --chain pre|post             hidden fed to draft step >=2 (default post)\n"
        "  --spec                       run the real greedy draft/verify/accept loop\n"
        "  --replay-check               replay prefix + sequential decode vs reference\n"
        "  --spec-only                  skip the acceptance sim + verify-check phases\n"
        "  --page-tokens N              (default 16)\n");
}

bool parse_u32(const char* s, unsigned long& out) {
    char* end = nullptr;
    out = std::strtoul(s, &end, 10);
    return end != s && *end == '\0';
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

std::vector<int32_t> read_token_corpus(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    char magic[8] = {};
    uint32_t version = 0;
    uint64_t count = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "PSKLDTOK", 8) == 0;
    ok = ok && std::fread(&version, 4, 1, f) == 1 && version == 1u;
    ok = ok && std::fread(&count, 8, 1, f) == 1 && count > 0u && count < (1u << 22);
    std::vector<int32_t> tokens;
    if (ok) {
        tokens.resize(static_cast<std::size_t>(count));
        ok = std::fread(tokens.data(), sizeof(int32_t), tokens.size(), f) == tokens.size();
    }
    std::fclose(f);
    if (!ok) return {};
    return tokens;
}

}  // namespace

int run_mtp(int argc, char** argv) {
    std::string model_dir;
    unsigned long device = 0;
    unsigned long arena_gib = 8;
    unsigned long context = 0;
    unsigned long steps = 32;
    unsigned long draft_k = 4;
    unsigned long token_id = 1;
    unsigned long page_tokens = 16;
    std::string tokens_file;
    unsigned long tokens_offset = 0;
    bool chain_post = true;
    bool spec_mode = false;
    bool replay_check = false;
    bool spec_only = false;
    bool mrow_check = false;
    int vc_level = 3;
    unsigned long mrow_offset = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") { const char* v = next("--model-dir"); if (!v) return 2; model_dir = v; }
        else if (a == "--device") { const char* v = next("--device"); if (!v || !parse_u32(v, device)) return 2; }
        else if (a == "--arena-gib") { const char* v = next("--arena-gib"); if (!v || !parse_u32(v, arena_gib)) return 2; }
        else if (a == "--context") { const char* v = next("--context"); if (!v || !parse_u32(v, context)) return 2; }
        else if (a == "--steps") { const char* v = next("--steps"); if (!v || !parse_u32(v, steps)) return 2; }
        else if (a == "--draft-k") { const char* v = next("--draft-k"); if (!v || !parse_u32(v, draft_k)) return 2; }
        else if (a == "--token-id") { const char* v = next("--token-id"); if (!v || !parse_u32(v, token_id)) return 2; }
        else if (a == "--page-tokens") { const char* v = next("--page-tokens"); if (!v || !parse_u32(v, page_tokens)) return 2; }
        else if (a == "--tokens-file") { const char* v = next("--tokens-file"); if (!v) return 2; tokens_file = v; }
        else if (a == "--tokens-offset") { const char* v = next("--tokens-offset"); if (!v || !parse_u32(v, tokens_offset)) return 2; }
        else if (a == "--chain") { const char* v = next("--chain"); if (!v) return 2; chain_post = (std::strcmp(v, "post") == 0); }
        else if (a == "--spec") { spec_mode = true; spec_only = true; }
        else if (a == "--replay-check") { replay_check = true; }
        else if (a == "--spec-only") { spec_mode = true; spec_only = true; }
        else if (a == "--mrow-check") { mrow_check = true; }
        else if (a == "--vc-level") { const char* v = next("--vc-level"); if (!v) return 2; vc_level = (int)strtol(v, nullptr, 10); }
        else if (a == "--mrow-offset") { const char* v = next("--mrow-offset"); if (!v || !parse_u32(v, mrow_offset)) return 2; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 2; }
    }
    if (model_dir.empty()) { usage(); return 2; }
    if (draft_k < 1u) draft_k = 1u;

    std::vector<int32_t> corpus;
    if (!tokens_file.empty()) {
        corpus = read_token_corpus(tokens_file);
        if (corpus.empty()) {
            std::fprintf(stderr, "failed to read token corpus: %s\n", tokens_file.c_str());
            return 1;
        }
        if (tokens_offset >= corpus.size()) {
            std::fprintf(stderr, "tokens-offset %lu >= corpus size %zu\n",
                         tokens_offset, corpus.size());
            return 1;
        }
        if (context == 0) context = corpus.size() - tokens_offset;
        if (tokens_offset + context > corpus.size()) {
            std::fprintf(stderr, "tokens-offset %lu + context %lu exceeds corpus %zu\n",
                         tokens_offset, context, corpus.size());
            return 1;
        }
    }

    if (hipSetDevice(static_cast<int>(device)) != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice failed\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::fprintf(stderr, "hipStreamCreate failed\n");
        return 1;
    }

    const std::size_t arena_bytes =
        static_cast<std::size_t>(arena_gib * 1024.0 * 1024.0 * 1024.0);
    auto arena_result = ps::gpu::GpuArena::create(static_cast<int>(device), arena_bytes);
    if (!arena_result.ok()) {
        std::fprintf(stderr, "GpuArena::create failed: %s\n",
                     arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    ps::qwen35::Qwen35LoadOptions load_options;
    auto model_result = ps::qwen35::Qwen35Model::load_from_safetensors(
        model_dir, arena, stream, load_options);
    if (!model_result.ok()) {
        std::fprintf(stderr, "model load failed: %s\n",
                     model_result.status().message().c_str());
        return 1;
    }
    auto model = model_result.release();
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::fprintf(stderr, "weight sync failed\n");
        return 1;
    }
    if (!model.weights().mtp.present) {
        std::fprintf(stderr, "model has no MTP weights\n");
        return 1;
    }

    const auto& tc = model.text_config();
    const uint32_t H = static_cast<uint32_t>(tc.hidden_size);
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t vocab_size = static_cast<uint32_t>(tc.vocab_size);

    const uint32_t hidden_rows = static_cast<uint32_t>(context) + static_cast<uint32_t>(steps);
    const uint32_t token_rows = hidden_rows + 1u;
    const uint32_t required_seq_len = token_rows + static_cast<uint32_t>(draft_k) + 2u;
    const uint32_t num_pages =
        ceil_div(required_seq_len, static_cast<uint32_t>(page_tokens)) + 1;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 1, num_pages);
    if (!seq_pool_result.ok()) {
        std::fprintf(stderr, "SequenceSlotPool::create failed: %s\n",
                     seq_pool_result.status().message().c_str());
        return 1;
    }
    auto seq_pool = seq_pool_result.release();

    auto rec_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 1, ps::qwen35::GdnStatePoolLayout::from_text_config(model.text_config()));
    if (!rec_pool_result.ok()) {
        std::fprintf(stderr, "GdnStatePool::create failed: %s\n",
                     rec_pool_result.status().message().c_str());
        return 1;
    }
    auto rec_pool = rec_pool_result.release();

    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, static_cast<uint32_t>(page_tokens),
        num_attn_layers, kv_heads, head_dim, ps::qwen35::KVCacheDType::BF16);
    if (!kv_pool_result.ok()) {
        std::fprintf(stderr, "PagedKVPool::create failed: %s\n",
                     kv_pool_result.status().message().c_str());
        return 1;
    }
    auto kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = required_seq_len;
    exec_config.max_scheduled_requests = 1;
    exec_config.max_scheduled_output_rows = static_cast<uint32_t>(draft_k) + 4u;

    auto exec_result = ps::qwen35::create_model_executor(
        model, seq_pool, rec_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::fprintf(stderr, "create_model_executor failed: %s\n",
                     exec_result.status().message().c_str());
        return 1;
    }
    auto executor = exec_result.release();

    auto seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, rec_pool, kv_pool, required_seq_len, stream);
    if (!seq_result.ok()) {
        std::fprintf(stderr, "create_paged_sequence_state failed: %s\n",
                     seq_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::PagedSequenceState seq = seq_result.release();

    ps::qwen35::runtime::MtpExecutorConfig mtp_config;
    mtp_config.page_tokens = static_cast<uint32_t>(page_tokens);
    mtp_config.num_pages = num_pages;
    mtp_config.max_rows = 16u;
    auto mtp_result = ps::qwen35::runtime::create_mtp_executor(
        model, ps::qwen35::KVCacheDType::BF16, arena, mtp_config, stream);
    if (!mtp_result.ok()) {
        std::fprintf(stderr, "create_mtp_executor failed: %s\n",
                     mtp_result.status().message().c_str());
        return 1;
    }
    auto mtp = mtp_result.release();

    auto hh_alloc = arena.allocate_aligned(
        static_cast<std::size_t>(hidden_rows) * H * sizeof(ps::bf16_t), 256);
    if (!hh_alloc.ok()) {
        std::fprintf(stderr, "hidden ring alloc failed\n");
        return 1;
    }
    auto hh_backing = hh_alloc.release();
    auto hh_tensor = ps::gpu::Tensor::create<ps::bf16_t>(
        hh_backing, {hidden_rows, H}, {H, 1});
    if (!hh_tensor.ok()) {
        std::fprintf(stderr, "hidden ring tensor failed\n");
        return 1;
    }
    auto hidden_ring = hh_tensor.release();
    ps::bf16_t* hidden_dev = hidden_ring.data<ps::bf16_t>();
    std::printf("addrs: tgt.token_hidden=%p+%zu tgt.workspace=%p+%zu mtp.workspace=%p+%zu mtp.logits=%p mtp.hidden_out=%p ring=%p\n",
                executor.token_hidden.data<void>(), executor.token_hidden.allocation_bytes(),
                executor.execution_workspace.data<void>(), executor.execution_workspace.allocation_bytes(),
                mtp.workspace.data<void>(), mtp.workspace.allocation_bytes(),
                mtp.logits.data<void>(), mtp.hidden_out.data<void>(), (void*)hidden_dev);

    std::vector<int32_t> tokens(token_rows, 0);
    std::vector<int32_t> prompt;

    // ---- phase 1: target greedy decode, collect per-position hidden + tokens ----
    int32_t cur = static_cast<int32_t>(token_id);
    double target_step_ms = 0.0;

    if (context > 0) {
        prompt.resize(static_cast<std::size_t>(context));
        if (!corpus.empty()) {
            for (std::size_t i = 0; i < prompt.size(); ++i) {
                prompt[i] = corpus[static_cast<std::size_t>(tokens_offset) + i];
            }
        } else {
            for (std::size_t i = 0; i < prompt.size(); ++i) {
                prompt[i] = static_cast<int32_t>(
                    (static_cast<uint64_t>(token_id) + static_cast<uint64_t>(i) * 7919ull) %
                    vocab_size);
            }
        }
        for (std::size_t i = 0; i < prompt.size(); ++i) {
            if (prompt[i] < 0 || static_cast<uint32_t>(prompt[i]) >= vocab_size) {
                std::fprintf(stderr, "prompt token %d out of vocab %u at %zu\n",
                             prompt[i], vocab_size, i);
                return 1;
            }
        }
        for (std::size_t i = 0; i < prompt.size(); ++i) tokens[i] = prompt[i];

        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = ps::runtime::ExecutionClass::PREFILL;
        req.token_begin = 0;
        req.num_tokens = static_cast<uint32_t>(context);
        req.prefix_tokens = seq.position;
        req.compute_logits = true;
        req.sample = true;
        std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = prompt.data();
        batch.requests = reqs;
        batch.num_tokens = static_cast<uint32_t>(context);
        batch.num_requests = 1;
        batch.num_decode_requests = 0;
        batch.num_verify_requests = 0;
        batch.num_prefill_requests = 1;

        auto res = ps::qwen35::execute_batch(executor, batch, stream);
        if (!res.ok()) {
            std::fprintf(stderr, "target prefill failed: %s\n",
                         res.status().message().c_str());
            return 1;
        }
        ps::qwen35::BatchExecutionOutput pout = std::move(res.release());
        if (pout.num_token_rows != static_cast<uint32_t>(context) || pout.num_outputs != 1) {
            std::fprintf(stderr, "prefill output mismatch: rows=%u outputs=%u\n",
                         pout.num_token_rows, pout.num_outputs);
            return 1;
        }
        if (hipMemcpy(&cur, pout.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "prefill sample copy failed\n");
            return 1;
        }
        if (hipMemcpyAsync(hidden_dev, pout.token_hidden.data<void>(),
                           static_cast<std::size_t>(context) * H * sizeof(ps::bf16_t),
                           hipMemcpyDeviceToDevice, stream) != hipSuccess) {
            std::fprintf(stderr, "prefill hidden copy failed\n");
            return 1;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::fprintf(stderr, "prefill hidden sync failed\n");
            return 1;
        }
        tokens[static_cast<std::size_t>(context)] = cur;
    }

    for (unsigned long step = 0; step < steps; ++step) {
        const uint32_t pos = seq.position;
        std::vector<int32_t> token_ids = {cur};
        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = ps::runtime::ExecutionClass::DECODE;
        req.token_begin = 0;
        req.num_tokens = 1;
        req.prefix_tokens = pos;
        req.compute_logits = true;
        req.sample = true;
        std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.requests = reqs;
        batch.num_tokens = 1;
        batch.num_requests = 1;
        batch.num_decode_requests = 1;
        batch.num_verify_requests = 0;
        batch.num_prefill_requests = 0;

        auto t0 = std::chrono::steady_clock::now();
        auto res = ps::qwen35::execute_batch(executor, batch, stream);
        if (!res.ok()) {
            std::fprintf(stderr, "target decode failed at %lu: %s\n", step,
                         res.status().message().c_str());
            return 1;
        }
        ps::qwen35::BatchExecutionOutput out = std::move(res.release());
        if (out.num_outputs != 1) {
            std::fprintf(stderr, "expected 1 output, got %u\n", out.num_outputs);
            return 1;
        }
        int32_t sampled = -1;
        if (hipMemcpy(&sampled, out.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "sample copy failed\n");
            return 1;
        }
        if (hipMemcpy(hidden_dev + static_cast<std::size_t>(pos) * H,
                      out.final_hidden.data<void>(),
                      static_cast<std::size_t>(H) * sizeof(ps::bf16_t),
                      hipMemcpyDeviceToDevice) != hipSuccess) {
            std::fprintf(stderr, "hidden copy failed\n");
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        target_step_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (pos >= hidden_rows) {
            std::fprintf(stderr, "hidden ring overflow at pos %u\n", pos);
            return 1;
        }
        tokens[static_cast<std::size_t>(pos) + 1u] = sampled;
        cur = sampled;
    }
    target_step_ms /= static_cast<double>(steps);

    std::printf("target greedy stream:");
    for (uint32_t i = static_cast<uint32_t>(context); i < token_rows; ++i) {
        std::printf(" %d", tokens[i]);
    }
    std::printf("\n");

    if (!spec_only) {
    // ---- phase 2: MTP KV over the prompt, then draft-loop acceptance simulation ----
    if (context > 1) {
        const uint32_t kv_rows = static_cast<uint32_t>(context) - 1u;
        std::vector<int32_t> chunk_tokens;
        for (uint32_t c = 0; c < kv_rows; c += mtp.max_rows) {
            const uint32_t rows = std::min<uint32_t>(mtp.max_rows, kv_rows - c);
            chunk_tokens.resize(rows);
            for (uint32_t i = 0; i < rows; ++i) chunk_tokens[i] = prompt[c + 1u + i];
            auto kr = ps::qwen35::runtime::run_mtp_rows(
                mtp, hidden_dev + static_cast<std::size_t>(c) * H,
                chunk_tokens.data(), rows, c, stream);
            if (!kr.ok()) {
                std::fprintf(stderr, "mtp kv prefill failed at %u: %s\n", c,
                             kr.status().message().c_str());
                return 1;
            }
        }
    }

    std::vector<int> depth_agree(static_cast<std::size_t>(draft_k) + 1u, 0);
    std::vector<int> depth_total(static_cast<std::size_t>(draft_k) + 1u, 0);
    std::vector<int> accept_hist(static_cast<std::size_t>(draft_k) + 1u, 0);
    int rounds = 0;
    long long committed = 0;
    double mtp_call_ms = 0.0;
    long long mtp_calls = 0;

    uint32_t pos = context > 0 ? static_cast<uint32_t>(context) - 1u : 0u;
    const uint32_t last_hidden = hidden_rows - 1u;
    const uint32_t last_token = token_rows - 1u;

    std::printf("\nround  pos   accept  drafts(expected)\n");
    while (pos < last_hidden && pos + 1u + draft_k <= last_token) {
        const ps::bf16_t* hid = hidden_dev + static_cast<std::size_t>(pos) * H;
        int32_t in_tok = tokens[static_cast<std::size_t>(pos) + 1u];
        int accept = 0;
        bool matched = true;
        std::vector<uint32_t> drafts;
        drafts.reserve(draft_k);

        for (uint32_t k = 0; k < draft_k; ++k) {
            auto t0 = std::chrono::steady_clock::now();
            auto dr = ps::qwen35::runtime::run_mtp_head(
                mtp, hid, in_tok, pos + k, stream);
            auto t1 = std::chrono::steady_clock::now();
            if (!dr.ok()) {
                std::fprintf(stderr, "mtp draft failed pos=%u k=%u: %s\n", pos, k,
                             dr.status().message().c_str());
                return 1;
            }
            mtp_call_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++mtp_calls;
            const uint32_t draft = dr.value();
            drafts.push_back(draft);
            const bool ok = draft == static_cast<uint32_t>(
                tokens[static_cast<std::size_t>(pos) + 2u + k]);
            ++depth_total[k];
            if (ok) ++depth_agree[k];
            if (matched && ok) ++accept;
            else matched = false;
            hid = chain_post ? mtp.hidden_out_normed.data<ps::bf16_t>()
                             : mtp.hidden_out.data<ps::bf16_t>();
            in_tok = static_cast<int32_t>(draft);
        }
        ++accept_hist[std::min<uint32_t>(static_cast<uint32_t>(accept),
                                         static_cast<uint32_t>(draft_k))];
        ++rounds;
        committed += accept + 1;

        std::printf("%5d %4u %6d  ", rounds, pos, accept);
        for (uint32_t k = 0; k < draft_k; ++k) {
            std::printf(" %6u%s", drafts[k],
                        drafts[k] == static_cast<uint32_t>(tokens[pos + 2u + k]) ? "" : "!");
        }
        std::printf("\n");

        pos += static_cast<uint32_t>(accept) + 1u;
    }
    mtp_call_ms /= static_cast<double>(mtp_calls > 0 ? mtp_calls : 1);

    std::printf("\ntarget decode: %u steps  %.3f ms/step\n",
                static_cast<uint32_t>(steps), target_step_ms);
    std::printf("mtp head:      %lld calls %.3f ms/call (%.1f%% of a decode step)\n",
                mtp_calls, mtp_call_ms, 100.0 * mtp_call_ms / target_step_ms);
    std::printf("\ndraft top-1 agreement by depth:\n");
    for (uint32_t k = 0; k < draft_k; ++k) {
        if (depth_total[k] == 0) continue;
        std::printf("  depth %u: %d / %d (%.1f%%)\n", k + 1u, depth_agree[k],
                    depth_total[k],
                    100.0 * static_cast<double>(depth_agree[k]) /
                        static_cast<double>(depth_total[k]));
    }
    std::printf("\naccept length histogram (drafts accepted per round):\n");
    for (uint32_t a = 0; a <= draft_k; ++a) {
        if (accept_hist[a] == 0) continue;
        std::printf("  %2u: %d\n", a, accept_hist[a]);
    }
    if (rounds > 0) {
        const double mean = static_cast<double>(committed) / static_cast<double>(rounds);
        const double step = target_step_ms + static_cast<double>(draft_k) * mtp_call_ms;
        std::printf("\nverify rounds: %d  tokens/round: %.2f\n", rounds, mean);
        std::printf("estimated speedup (no verify cost): %.2fx over %u decode steps\n",
                    mean * target_step_ms / step, static_cast<uint32_t>(draft_k));

        std::printf("\nconditional acceptance (given prefix matched):\n");
        double prev_p = 1.0;
        for (uint32_t k = 1; k <= draft_k; ++k) {
            int ge = 0;
            for (uint32_t a = k; a <= draft_k; ++a) ge += accept_hist[a];
            const double p = static_cast<double>(ge) / static_cast<double>(rounds);
            const double cond = prev_p > 0.0 ? p / prev_p : 0.0;
            std::printf("  P(draft %u correct | 1..%u accepted) = %5.1f%%   cumulative %5.1f%%"
                        "  (%d/%d)\n", k, k - 1u, 100.0 * cond, 100.0 * p, ge, rounds);
            prev_p = p;
        }

        std::printf("\nper draft count K: expected tokens/round and estimated speedup\n");
        std::printf("  K   accept   tokens/round   ms/token   speedup\n");
        const double plain = target_step_ms;
        for (uint32_t kp = 1; kp <= draft_k; ++kp) {
            double ea = 0.0;
            double pge = 1.0;
            for (uint32_t k = 1u; k <= kp; ++k) {
                int ge = 0;
                for (uint32_t a = k; a <= draft_k; ++a) ge += accept_hist[a];
                const double cond = static_cast<double>(ge) /
                    static_cast<double>(std::max(rounds, 1));
                ea += pge * (ge == 0 ? 0.0 : cond / std::max(pge, 1e-9)) ;
                pge = cond;
            }
            const double tpr = 1.0 + ea;
            const double ms_tok = (target_step_ms + static_cast<double>(kp) * mtp_call_ms) / tpr;
            std::printf("  %2u   %6.2f      %6.2f       %7.3f    %5.2fx\n",
                        kp, ea, tpr, ms_tok, plain / ms_tok);
        }
    }

    if (!spec_only) {
        {
            const uint32_t pp = hidden_rows - 2u;
            auto pr = ps::qwen35::runtime::run_mtp_head(
                mtp, hidden_dev + static_cast<std::size_t>(pp) * H,
                tokens[static_cast<std::size_t>(pp) + 1u], pp, stream);
            uint32_t st = 0;
            float lg[4] = {0.f, 0.f, 0.f, 0.f};
            if (hipMemcpy(&st, mtp.execution_status.data<uint32_t>(), 4, hipMemcpyDeviceToHost) != hipSuccess ||
                hipMemcpy(lg, mtp.logits.data<float>(), 16, hipMemcpyDeviceToHost) != hipSuccess) {
                std::fprintf(stderr, "mtp-probe after-phase2 copy failed\n");
                return 1;
            }
            std::printf("mtp-probe after-phase2: %s st=%u lg=[%.3f %.3f]\n",
                        pr.ok() ? ("draft=" + std::to_string(pr.value())).c_str()
                                : ("ERR " + pr.status().message()).c_str(),
                        st, lg[0], lg[1]);
            std::fflush(stdout);
        }

    // ---- phase 3: multi-row verify equivalence against the sequential reference ----
    if (vc_level > 0) {
    {
        const uint32_t vrows = static_cast<uint32_t>(draft_k) + 2u;
        const uint32_t win = hidden_rows > vrows ? hidden_rows - vrows : 0u;
        if (win >= 1u && static_cast<uint64_t>(win) + vrows <= token_rows) {
            std::printf("\nverify-check: window at pos %u, %u rows\n", win, vrows);
            ps::Status rst = rec_pool.reset(seq.slot, stream);
            if (!rst.ok()) {
                std::fprintf(stderr, "gdn reset failed: %s\n", rst.message().c_str());
                return 1;
            }
            seq.position = 0u;

            auto run_req = [&](uint32_t begin, uint32_t nt, uint32_t nout, bool logits,
                               ps::runtime::ExecutionClass cls,
                               ps::qwen35::BatchExecutionOutput* dst) -> bool {
                ps::qwen35::ScheduledRequest req;
                req.sequence = &seq;
                req.handle = seq.request_handle();
                req.execution_class = cls;
                req.token_begin = 0;
                req.num_tokens = nt;
                req.prefix_tokens = seq.position;
                req.compute_logits = logits;
                req.sample = logits;
                req.num_output_rows = nout;
                std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
                ps::qwen35::ScheduledBatch batch;
                batch.token_ids = tokens.data() + begin;
                batch.requests = reqs;
                batch.num_tokens = nt;
                batch.num_requests = 1;
                batch.num_decode_requests = cls == ps::runtime::ExecutionClass::DECODE ? 1u : 0u;
                batch.num_verify_requests = 0;
                batch.num_prefill_requests = cls == ps::runtime::ExecutionClass::DECODE ? 0u : 1u;
                auto res = ps::qwen35::execute_batch(executor, batch, stream);
                if (!res.ok()) {
                    std::fprintf(stderr, "verify-check request failed: %s\n",
                                 res.status().message().c_str());
                    return false;
                }
                if (dst != nullptr) *dst = std::move(res.release());
                else (void)res.release();
                return true;
            };

            ps::qwen35::BatchExecutionOutput vout;
            bool ok = true;
            const uintptr_t ws_end =
                reinterpret_cast<uintptr_t>(executor.execution_workspace.data<uint8_t>()) +
                executor.execution_workspace.allocation_bytes();
            std::vector<uint8_t> ring_saved;
            if (vc_level == 2) {
                const size_t ring_bytes = static_cast<std::size_t>(hidden_rows) * H * sizeof(ps::bf16_t);
                ring_saved.resize(ring_bytes);
                if (hipMemcpy(ring_saved.data(), hidden_dev, ring_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
                    std::fprintf(stderr, "ring save failed\n");
                    return 1;
                }
                hipError_t pe = hipMemsetAsync(reinterpret_cast<void*>(ws_end), 0xCD, 8u << 20, stream);
                if (pe != hipSuccess) { std::fprintf(stderr, "paint failed\n"); return 1; }
            }
            if (win > 0u) ok = run_req(0u, win, 1u, false, ps::runtime::ExecutionClass::PREFILL,
                                        nullptr) && ok;
            if (vc_level == 2) {
                std::vector<uint8_t> sp(8u << 20, 0xCD);
                if (hipMemcpy(sp.data(), reinterpret_cast<void*>(ws_end), sp.size(), hipMemcpyDeviceToHost) != hipSuccess) {
                    std::fprintf(stderr, "paint readback failed\n");
                    return 1;
                }
                size_t first = sp.size(), last = 0, runs = 0; bool inrun = false;
                for (size_t z = 0; z < sp.size(); ++z) {
                    if (sp[z] != 0xCD) {
                        if (!inrun) { ++runs; inrun = true; }
                        if (first == sp.size()) first = z;
                        last = z;
                    } else inrun = false;
                }
                const size_t lg_off = reinterpret_cast<uintptr_t>(mtp.logits.data<float>()) - ws_end;
                const size_t hw_off = reinterpret_cast<uintptr_t>(mtp.workspace.data<uint8_t>()) - ws_end;
                const size_t ring_off = reinterpret_cast<uintptr_t>(hidden_dev) - ws_end;
                const size_t hi_off = reinterpret_cast<uintptr_t>(mtp.hidden_in.data<void>()) - ws_end;
                const size_t ho_off = reinterpret_cast<uintptr_t>(mtp.hidden_out.data<void>()) - ws_end;
                std::printf("paint: first=+%zu last=+%zu runs=%zu | hidden_in=+%zu hidden_out=+%zu logits=+%zu mtp.ws=+%zu ring=+%zu\n",
                            first, last, runs, hi_off, ho_off, lg_off, hw_off, ring_off);
                if (first < sp.size()) {
                    std::printf("  @first:");
                    for (size_t z = first; z + 1 < first + 16 && z < sp.size(); ++z) std::printf(" %02x", sp[z]);
                    std::printf("\n");
                }
                if (!ring_saved.empty()) {
                    if (hipMemcpy(hidden_dev, ring_saved.data(), ring_saved.size(), hipMemcpyHostToDevice) != hipSuccess) {
                        std::fprintf(stderr, "ring restore failed\n");
                        return 1;
                    }
                }
                std::fflush(stdout);
            }
            if (vc_level >= 3)
                ok = ok && run_req(win, vrows, vrows, true, ps::runtime::ExecutionClass::PREFILL,
                                   &vout);
            if (!ok) return 1;
            {
                const uint32_t pp = hidden_rows - 2u;
                auto pr = ps::qwen35::runtime::run_mtp_head(
                    mtp, hidden_dev + static_cast<std::size_t>(pp) * H,
                    tokens[static_cast<std::size_t>(pp) + 1u], pp, stream);
                uint32_t st = 0;
                std::vector<float> lgfull(static_cast<std::size_t>(vocab_size), 0.f);
                uint16_t hin[8] = {};
                const hipError_t pe0 =
                    hipMemcpy(&st, mtp.execution_status.data<uint32_t>(), 4, hipMemcpyDeviceToHost);
                const hipError_t pe1 = hipMemcpy(
                    lgfull.data(), mtp.logits.data<float>(),
                    static_cast<std::size_t>(vocab_size) * sizeof(float),
                    hipMemcpyDeviceToHost);
                const hipError_t pe2 = hipMemcpy(
                    hin, hidden_dev + static_cast<std::size_t>(pp) * H, sizeof(hin),
                    hipMemcpyDeviceToHost);
                if (pe0 != hipSuccess || pe1 != hipSuccess || pe2 != hipSuccess) {
                    std::fprintf(stderr, "mtp-probe after-replay copy failed\n");
                    return 1;
                }
                size_t nzc = 0;
                for (float v : lgfull) nzc += (v != 0.f);
                std::printf("mtp-probe after-replay(win=%u): %s st=%u logits_nz=%zu ring=%04x %04x\n",
                            win,
                            pr.ok() ? ("draft=" + std::to_string(pr.value())).c_str()
                                    : ("ERR " + pr.status().message()).c_str(),
                            st, nzc, hin[0], hin[1]);
                std::fflush(stdout);
            }
            if (vc_level == 2) return 0;
            if (vout.num_outputs != vrows) {
                std::fprintf(stderr, "verify-check: got %u outputs, expected %u\n",
                             vout.num_outputs, vrows);
                return 1;
            }
            std::vector<int32_t> vsamp(vrows, -1);
            if (hipMemcpy(vsamp.data(), vout.sampled_tokens.data<int32_t>(),
                          static_cast<std::size_t>(vrows) * sizeof(int32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                std::fprintf(stderr, "verify-check sample copy failed\n");
                return 1;
            }
            std::printf("  row  expect  got    ok\n");
            int mismatches = 0;
            for (uint32_t r = 0; r < vrows; ++r) {
                const int32_t expect = tokens[static_cast<std::size_t>(win) + r + 1u];
                const bool m = vsamp[r] == expect;
                if (!m) ++mismatches;
                std::printf("  %3u  %6d %6d %4s\n", r, expect, vsamp[r], m ? "ok" : "NO");
            }
            std::printf("verify-check: %s (%d/%u rows mismatched)\n",
                        mismatches == 0 ? "PASS" : "FAIL", mismatches, vrows);
            std::fflush(stdout);
        }
    }
    }

    }

    }

    // ---- phase 3b: does replay + sequential decode reproduce the reference? ----
    if (replay_check && context >= 1u) {
        std::printf("\nreplay-check: reset + prefill %u tokens, then %lu sequential decodes\n",
                    static_cast<uint32_t>(context), std::min<unsigned long>(steps, 8ul));
        ps::Status rst = rec_pool.reset(seq.slot, stream);
        if (!rst.ok()) {
            std::fprintf(stderr, "replay-check reset failed\n");
            return 1;
        }
        seq.position = 0u;
        std::vector<ps::qwen35::ScheduledRequest> reqs(1);
        reqs[0].sequence = &seq;
        reqs[0].handle = seq.request_handle();
        reqs[0].execution_class = ps::runtime::ExecutionClass::PREFILL;
        reqs[0].num_tokens = static_cast<uint32_t>(context);
        reqs[0].prefix_tokens = seq.position;
        reqs[0].compute_logits = true;
        reqs[0].sample = true;
        reqs[0].num_output_rows = 1;
        ps::qwen35::ScheduledBatch pb;
        pb.token_ids = prompt.data();
        pb.requests = reqs;
        pb.num_tokens = static_cast<uint32_t>(context);
        pb.num_requests = 1;
        pb.num_prefill_requests = 1;
        auto pres = ps::qwen35::execute_batch(executor, pb, stream);
        if (!pres.ok()) {
            std::fprintf(stderr, "replay-check prefill failed: %s\n",
                         pres.status().message().c_str());
            return 1;
        }
        ps::qwen35::BatchExecutionOutput po = std::move(pres.release());
        int32_t rcur = -1;
        if (hipMemcpy(&rcur, po.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "replay-check copy failed\n");
            return 1;
        }
        int bad = -1;
        if (rcur != tokens[static_cast<std::size_t>(context)]) {
            bad = 0;
        }
        const uint32_t nseq = static_cast<uint32_t>(std::min<unsigned long>(steps, 8ul));
        for (uint32_t i = 0; i < nseq && bad < 0; ++i) {
            const uint32_t rpos = seq.position;
            std::vector<int32_t> tin = {rcur};
            std::vector<ps::qwen35::ScheduledRequest> r2(1);
            r2[0].sequence = &seq;
            r2[0].handle = seq.request_handle();
            r2[0].execution_class = ps::runtime::ExecutionClass::DECODE;
            r2[0].num_tokens = 1;
            r2[0].prefix_tokens = rpos;
            r2[0].compute_logits = true;
            r2[0].sample = true;
            r2[0].num_output_rows = 1;
            ps::qwen35::ScheduledBatch b2;
            b2.token_ids = tin.data();
            b2.requests = r2;
            b2.num_tokens = 1;
            b2.num_requests = 1;
            b2.num_decode_requests = 1;
            auto rs = ps::qwen35::execute_batch(executor, b2, stream);
            if (!rs.ok()) {
                std::fprintf(stderr, "replay-check decode failed\n");
                return 1;
            }
            ps::qwen35::BatchExecutionOutput ro = std::move(rs.release());
            if (hipMemcpy(&rcur, ro.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                std::fprintf(stderr, "replay-check decode copy failed\n");
                return 1;
            }
            const int32_t ref = tokens[static_cast<std::size_t>(context) + 1u + i];
            std::printf("  pos %u replay=%8d reference=%8d %s\n", rpos + 1u, rcur, ref,
                        rcur == ref ? "ok" : "DIFF");
            if (rcur != ref) bad = static_cast<int>(i) + 1;
        }
        std::printf("replay-check: %s\n", bad < 0 ? "PASS (replay is exact)" : "FAIL (replay drifts)");
        std::fflush(stdout);
    }

        {
            const uint32_t pp = hidden_rows - 2u;
            auto pr = ps::qwen35::runtime::run_mtp_head(
                mtp, hidden_dev + static_cast<std::size_t>(pp) * H,
                tokens[static_cast<std::size_t>(pp) + 1u], pp, stream);
            uint32_t st = 0;
            float lg[4] = {0.f, 0.f, 0.f, 0.f};
            if (hipMemcpy(&st, mtp.execution_status.data<uint32_t>(), 4, hipMemcpyDeviceToHost) != hipSuccess ||
                hipMemcpy(lg, mtp.logits.data<float>(), 16, hipMemcpyDeviceToHost) != hipSuccess) {
                std::fprintf(stderr, "mtp-probe after-phase3 copy failed\n");
                return 1;
            }
            std::printf("mtp-probe after-phase3: %s st=%u lg=[%.3f %.3f]\n",
                        pr.ok() ? ("draft=" + std::to_string(pr.value())).c_str()
                                : ("ERR " + pr.status().message()).c_str(),
                        st, lg[0], lg[1]);
            std::fflush(stdout);
        }

    // ---- phase 5: prefill(context) then ONE reference-only multi-row verify ----
    if (mrow_check && context >= 1u) {
        const uint32_t mr = 4u;
        const uint32_t moff = static_cast<uint32_t>(mrow_offset);
        std::printf("\nmrow-check: prefill %u tokens then %u-row verify at pos %u\n",
                    static_cast<uint32_t>(context), mr, static_cast<uint32_t>(context) + moff);
        ps::Status rst = rec_pool.reset(seq.slot, stream);
        if (!rst.ok()) { std::fprintf(stderr, "mrow reset failed\n"); return 1; }
        seq.position = 0u;
        std::vector<ps::qwen35::ScheduledRequest> rq(1);
        rq[0].sequence = &seq;
        rq[0].handle = seq.request_handle();
        rq[0].execution_class = ps::runtime::ExecutionClass::PREFILL;
        rq[0].num_tokens = static_cast<uint32_t>(context);
        rq[0].prefix_tokens = 0u;
        rq[0].compute_logits = false;
        rq[0].num_output_rows = 1;
        ps::qwen35::ScheduledBatch b1;
        b1.token_ids = prompt.data();
        b1.requests = rq;
        b1.num_tokens = static_cast<uint32_t>(context);
        b1.num_requests = 1;
        b1.num_prefill_requests = 1;
        auto r1 = ps::qwen35::execute_batch(executor, b1, stream);
        if (!r1.ok()) {
            std::fprintf(stderr, "mrow prefill failed: %s\n", r1.status().message().c_str());
            return 1;
        }
        (void)r1.release();
        std::vector<int32_t> vin(mr);
        for (uint32_t i = 0; i < mr; ++i) vin[i] = tokens[static_cast<std::size_t>(context) + moff + i];
        std::vector<ps::qwen35::ScheduledRequest> rq2(1);
        rq2[0].sequence = &seq;
        rq2[0].handle = seq.request_handle();
        rq2[0].execution_class = ps::runtime::ExecutionClass::PREFILL;
        rq2[0].num_tokens = mr;
        rq2[0].prefix_tokens = seq.position;
        rq2[0].compute_logits = true;
        rq2[0].sample = true;
        rq2[0].num_output_rows = mr;
        ps::qwen35::ScheduledBatch b2;
        b2.token_ids = vin.data();
        b2.requests = rq2;
        b2.num_tokens = mr;
        b2.num_requests = 1;
        b2.num_prefill_requests = 1;
        auto r2 = ps::qwen35::execute_batch(executor, b2, stream);
        if (!r2.ok()) {
            std::fprintf(stderr, "mrow verify failed: %s\n", r2.status().message().c_str());
            return 1;
        }
        ps::qwen35::BatchExecutionOutput vo = std::move(r2.release());
        std::vector<int32_t> vs(mr, -1);
        if (hipMemcpy(vs.data(), vo.sampled_tokens.data<int32_t>(),
                      static_cast<std::size_t>(mr) * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "mrow sample copy failed\n");
            return 1;
        }
        std::vector<float> lgrow(static_cast<std::size_t>(vocab_size), 0.f);
        int bad = 0;
        for (uint32_t i = 0; i + 1u < mr; ++i) {
            const int32_t ref = tokens[static_cast<std::size_t>(context) + 1u + i];
            const bool ok = vs[i] == ref;
            if (!ok) ++bad;
            std::printf("  row %u (pos %u) in=%8d got=%8d ref=%8d %s",
                        i, seq.position - mr + i, vin[i], vs[i], ref, ok ? "ok" : "DIFF");
            if (hipMemcpy(lgrow.data(),
                          vo.logits.data<float>() + static_cast<std::size_t>(i) * vocab_size,
                          static_cast<std::size_t>(vocab_size) * sizeof(float),
                          hipMemcpyDeviceToHost) == hipSuccess) {
                float m1 = -1e30f, m2 = -1e30f;
                for (uint32_t z = 0; z < vocab_size; ++z) {
                    const float v = lgrow[z];
                    if (v > m1) { m2 = m1; m1 = v; }
                    else if (v > m2) { m2 = v; }
                }
                std::printf("  top1=%.4f top2=%.4f gap=%.4f", m1, m2, m1 - m2);
            }
            std::printf("\n");
        }
        std::printf("mrow-check: %s (%d mismatched)\n", bad == 0 ? "PASS" : "FAIL", bad);
        std::fflush(stdout);
    }

    // ---- phase 4: real greedy speculative decode loop (draft -> verify -> accept) ----
    if (spec_mode && context < 1u) {
        std::fprintf(stderr, "--spec requires --context >= 1\n");
        return 2;
    }
    if (spec_mode) {
        std::printf("\n--- spec decode (K=%lu) ---\n", draft_k);
        ps::Status rst = rec_pool.reset(seq.slot, stream);
        if (!rst.ok()) {
            std::fprintf(stderr, "spec: gdn reset failed: %s\n", rst.message().c_str());
            return 1;
        }
        seq.position = 0u;

        auto hid_alloc = arena.allocate_aligned(static_cast<std::size_t>(H) * sizeof(ps::bf16_t), 128);
        if (!hid_alloc.ok()) {
            std::fprintf(stderr, "spec hidden alloc failed\n");
            return 1;
        }
        auto hb = hid_alloc.release();
        auto ht = ps::gpu::Tensor::create<ps::bf16_t>(hb, {H}, {1});
        if (!ht.ok()) {
            std::fprintf(stderr, "spec hidden tensor failed\n");
            return 1;
        }
        auto spec_h = ht.release();
        ps::bf16_t* hlast = spec_h.data<ps::bf16_t>();

        auto run_one = [&](const int32_t* toks, uint32_t nt, uint32_t nout,
                           bool logits, ps::qwen35::BatchExecutionOutput* dst) -> bool {
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq;
            req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::PREFILL;
            req.token_begin = 0;
            req.num_tokens = nt;
            req.prefix_tokens = seq.position;
            req.compute_logits = logits;
            req.sample = logits;
            req.num_output_rows = nout;
            std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = toks;
            batch.requests = reqs;
            batch.num_tokens = nt;
            batch.num_requests = 1;
            batch.num_decode_requests = 0;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 1;
            auto res = ps::qwen35::execute_batch(executor, batch, stream);
            if (!res.ok()) {
                std::fprintf(stderr, "spec forward failed: %s\n", res.status().message().c_str());
                return false;
            }
            if (dst != nullptr) *dst = std::move(res.release());
            else (void)res.release();
            return true;
        };

        ps::qwen35::BatchExecutionOutput pout;
        if (!run_one(prompt.data(), static_cast<uint32_t>(context), 1u, true, &pout) ||
            pout.num_outputs != 1u) {
            return 1;
        }
        int32_t cur = -1;
        if (hipMemcpy(&cur, pout.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "spec prefill sample copy failed\n");
            return 1;
        }
        if (hipMemcpy(hlast, pout.final_hidden.data<void>(),
                      static_cast<std::size_t>(H) * sizeof(ps::bf16_t),
                      hipMemcpyDeviceToDevice) != hipSuccess) {
            std::fprintf(stderr, "spec prefill hidden copy failed\n");
            return 1;
        }

        uint32_t P = static_cast<uint32_t>(context) - 1u;
        std::vector<int32_t> emitted;
        std::vector<int32_t> vtok(draft_k + 1u);
        std::vector<int32_t> vsamp(draft_k + 1u);
        std::vector<uint32_t> drafts(draft_k);
        long long rounds = 0;
        long long reruns = 0;
        long long sum_accept = 0;

        auto t0 = std::chrono::steady_clock::now();
        while (emitted.size() < steps) {
            ps::bf16_t* hid = hlast;
            int32_t in_tok = cur;
            for (uint32_t k = 0; k < draft_k; ++k) {
                auto dr = ps::qwen35::runtime::run_mtp_head(
                    mtp, hid, in_tok, P + k, stream);
                if (!dr.ok()) {
                    std::fprintf(stderr, "spec draft failed at pos %u: %s\n", P + k,
                                 dr.status().message().c_str());
                    return 1;
                }
                drafts[k] = dr.value();
                if (rounds == 0 && k == 0u) {
                    std::vector<float> lg(8, 0.f);
                    if (hipMemcpy(lg.data(), mtp.logits.data<float>(), lg.size() * sizeof(float),
                                  hipMemcpyDeviceToHost) != hipSuccess) {
                        std::fprintf(stderr, "spec debug copy failed\n");
                        return 1;
                    }
                    std::printf("  dbg draft0 pos=%u tok=%d logits[0..7]=", P, in_tok);
                    for (uint32_t z = 0; z < lg.size(); ++z) std::printf(" %.3f", lg[z]);
                    std::printf(" -> %u\n", drafts[k]);
                    std::fflush(stdout);
                }
                hid = chain_post ? mtp.hidden_out_normed.data<ps::bf16_t>()
                                 : mtp.hidden_out.data<ps::bf16_t>();
                in_tok = static_cast<int32_t>(drafts[k]);
            }

            vtok[0] = cur;
            for (uint32_t k = 0; k < draft_k; ++k) vtok[k + 1u] = static_cast<int32_t>(drafts[k]);
            ps::qwen35::BatchExecutionOutput vout;
            if (!run_one(vtok.data(), static_cast<uint32_t>(draft_k) + 1u,
                         static_cast<uint32_t>(draft_k) + 1u, true, &vout) ||
                vout.num_outputs != static_cast<uint32_t>(draft_k) + 1u) {
                return 1;
            }
            if (hipMemcpy(vsamp.data(), vout.sampled_tokens.data<int32_t>(),
                          vtok.size() * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
                std::fprintf(stderr, "spec verify sample copy failed\n");
                return 1;
            }

            uint32_t a = 0;
            while (a < draft_k && vsamp[a] == static_cast<int32_t>(drafts[a])) ++a;
            if (rounds < 6) {
                std::printf("  rnd%lld P=%u cur=%d drafts=", rounds, P, cur);
                for (uint32_t k = 0; k < draft_k; ++k) std::printf(" %d", drafts[k]);
                std::printf("  verify=");
                for (uint32_t k = 0; k <= draft_k; ++k) std::printf(" %d", vsamp[k]);
                std::printf("  accept=%u%s\n", a, a < draft_k ? " RERUN" : "");
            }
            const int32_t bonus = vsamp[a];
            for (uint32_t k = 0; k < a; ++k) emitted.push_back(static_cast<int32_t>(drafts[k]));
            emitted.push_back(bonus);
            ++rounds;
            sum_accept += a;

            const uint32_t newP = P + 1u + a;
            if (a == draft_k) {
                const ps::bf16_t* row =
                    vout.token_hidden.data<ps::bf16_t>() + static_cast<std::size_t>(a) * H;
                if (hipMemcpy(hlast, row, static_cast<std::size_t>(H) * sizeof(ps::bf16_t),
                              hipMemcpyDeviceToDevice) != hipSuccess) {
                    std::fprintf(stderr, "spec hidden copy failed\n");
                    return 1;
                }
            } else {
                ++reruns;
                seq.position = P + 1u;
                ps::qwen35::BatchExecutionOutput rout;
                if (!run_one(vtok.data(), a + 1u, 1u, true, &rout) || rout.num_outputs != 1u) {
                    return 1;
                }
                if (hipMemcpy(hlast, rout.final_hidden.data<void>(),
                              static_cast<std::size_t>(H) * sizeof(ps::bf16_t),
                              hipMemcpyDeviceToDevice) != hipSuccess) {
                    std::fprintf(stderr, "spec rerun hidden copy failed\n");
                    return 1;
                }
            }
            P = newP;
            cur = bonus;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double spec_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() /
            static_cast<double>(emitted.size());

        std::printf("reference (tokens[context+1+i]):");
        for (uint32_t i = 0; i < 10u; ++i) {
            std::printf(" %d", tokens[static_cast<std::size_t>(context) + 1u + i]);
        }
        std::printf("\nemitted                          :");
        for (std::size_t i = 0; i < std::min<size_t>(emitted.size(), 11u); ++i) {
            std::printf(" %d", emitted[i]);
        }
        std::printf("\n");

        int first_bad = -1;
        const std::size_t cmp_n = std::min(emitted.size(), static_cast<std::size_t>(steps));
        for (std::size_t i = 0; i < cmp_n; ++i) {
            const int32_t ref = tokens[static_cast<std::size_t>(context) + 1u + i];
            if (emitted[i] != ref) {
                first_bad = static_cast<int>(i);
                break;
            }
        }
        std::printf("spec: %zu tokens in %lld rounds (reruns %lld, mean accept %.2f)\n",
                    emitted.size(), rounds, reruns,
                    static_cast<double>(sum_accept) / static_cast<double>(rounds));
        std::printf("spec decode: %.3f ms/token   plain decode: %.3f ms/token   speedup %.2fx\n",
                    spec_ms, target_step_ms, target_step_ms / spec_ms);
        std::printf("greedy equivalence: %s", first_bad < 0 ? "PASS" : "FAIL");
        if (first_bad >= 0) {
            std::printf("  (diverges at emitted index %d: got %d expected %d)",
                        first_bad, emitted[first_bad],
                        tokens[static_cast<std::size_t>(context) + 1u + first_bad]);
        }
        std::printf("\n");
    }

    ps::qwen35::runtime::mtp_executor_shutdown(mtp);
    if (hipStreamSynchronize(stream) != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 1;
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return 0;
}
