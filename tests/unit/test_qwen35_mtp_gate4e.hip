#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/runtime/spec_decoder.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

namespace qwen35 = ps::qwen35;
namespace q35rt = ps::qwen35::runtime;

double wall_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

uint32_t count_attention_layers(const qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

std::vector<int32_t> read_corpus(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    char magic[8];
    f.read(magic, 8);
    if (f.gcount() != 8 || std::memcmp(magic, "PSKLDTOK", 8) != 0) return {};
    uint32_t version = 0;
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&count), 8);
    if (count == 0 || count > 4000000u) return {};
    std::vector<int32_t> out(count);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * 4));
    return out;
}

qwen35::ScheduledBatch make_batch(
    qwen35::PagedSequenceState& sequence,
    std::vector<qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    ps::runtime::VerifyNumericMode numeric_mode) {
    qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? ps::runtime::ExecutionClass::DECODE
                                           : ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = output_rows;
    requests.clear();
    requests.push_back(req);
    qwen35::ScheduledBatch batch;
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

bool sample_once(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& tokens,
    uint32_t output_rows,
    std::vector<int32_t>& sampled,
    hipStream_t stream) {
    std::vector<qwen35::ScheduledRequest> requests;
    qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, tokens.data(), static_cast<uint32_t>(tokens.size()),
        sequence.position, output_rows, false, ps::runtime::VerifyNumericMode::Fast);
    auto executed = qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: execute_batch: %s\n", executed.status().message().c_str());
        return false;
    }
    qwen35::BatchExecutionOutput output = executed.release();
    sampled.assign(output_rows, -1);
    return hipMemcpy(sampled.data(), output.sampled_tokens.data<int32_t>(),
                     output_rows * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess;
}

struct IterRec {
    uint32_t prompt_id = 0u;
    std::string category;
    uint32_t ctx = 0u;
    uint32_t k = 0u;
    uint32_t iteration = 0u;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double snap_ms = 0.0;
    double restore_ms = 0.0;
    double commit_ms = 0.0;
    double total_ms = 0.0;
    uint32_t num_drafts = 0u;
    uint32_t num_accepted = 0u;
    uint32_t num_emitted = 0u;
};

struct TargetStats {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    uint32_t decode_tokens = 0u;
    std::vector<int32_t> tokens;
};

bool run_target_only(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t gen_tokens,
    TargetStats& out,
    hipStream_t stream) {
    std::vector<int32_t> sampled;
    auto t0 = std::chrono::steady_clock::now();
    if (!sample_once(executor, sequence, prompt, 1u, sampled, stream)) return false;
    out.prefill_ms = wall_ms(t0);
    out.tokens.clear();
    out.tokens.push_back(sampled[0]);
    out.decode_tokens = 1u;
    auto td0 = std::chrono::steady_clock::now();
    for (uint32_t i = 1u; i < gen_tokens; ++i) {
        std::vector<int32_t> step{out.tokens.back()};
        if (!sample_once(executor, sequence, step, 1u, sampled, stream)) return false;
        out.tokens.push_back(sampled[0]);
    }
    out.decode_ms = wall_ms(td0);
    out.decode_tokens = static_cast<uint32_t>(out.tokens.size() - 1u);
    return true;
}

bool run_spec(
    q35rt::SpecDecoder& decoder,
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t gen_tokens,
    uint32_t k,
    uint32_t prompt_id,
    const std::string& category,
    uint32_t ctx,
    const std::vector<int32_t>& reference,
    std::vector<IterRec>& records,
    std::vector<int32_t>& out_tokens,
    hipStream_t stream) {
    std::vector<qwen35::ScheduledRequest> requests;
    qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, prompt.data(), static_cast<uint32_t>(prompt.size()),
        sequence.position, 1u, false, ps::runtime::VerifyNumericMode::Fast);
    auto executed = qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: spec prefill: %s\n", executed.status().message().c_str());
        return false;
    }
    qwen35::BatchExecutionOutput prefill = executed.release();
    int32_t first = -1;
    hipError_t err = hipMemcpy(&first, prefill.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                               hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::printf("FAIL: spec prefill sample: %s\n", hipGetErrorString(err));
        return false;
    }
    auto sync = q35rt::spec_decoder_sync_prompt(
        decoder, prompt.data(), prefill.token_hidden.data<ps::bf16_t>(),
        static_cast<uint32_t>(prompt.size()));
    if (!sync.ok()) {
        std::printf("FAIL: spec sync: %s\n", sync.message().c_str());
        return false;
    }
    out_tokens.clear();
    out_tokens.push_back(first);
    int32_t pending = first;
    uint32_t iteration = 0u;
    while (out_tokens.size() < gen_tokens && pending >= 0) {
        const q35rt::SpecDecoderTiming before = decoder.timing ? *decoder.timing
                                                               : q35rt::SpecDecoderTiming{};
        auto step = q35rt::spec_decoder_step(decoder, pending);
        if (!step.ok()) {
            std::printf("FAIL: spec step: %s\n", step.status().message().c_str());
            return false;
        }
        const q35rt::SpecDecoderTiming after = decoder.timing ? *decoder.timing
                                                              : q35rt::SpecDecoderTiming{};
        q35rt::SpecIterationOutput value = step.release();
        if (k > 0u) {
            IterRec rec;
            rec.prompt_id = prompt_id;
            rec.category = category;
            rec.ctx = ctx;
            rec.k = k;
            rec.iteration = iteration;
            rec.draft_ms = after.draft_ms - before.draft_ms;
            rec.verify_ms = after.verify_ms - before.verify_ms;
            rec.snap_ms = after.gdn_snapshot_ms - before.gdn_snapshot_ms;
            rec.restore_ms = after.gdn_restore_ms - before.gdn_restore_ms;
            rec.commit_ms = after.commit_ms - before.commit_ms;
            rec.total_ms = after.total_ms - before.total_ms;
            rec.num_drafts = value.num_drafts_generated;
            rec.num_accepted = value.num_accepted_drafts;
            rec.num_emitted = static_cast<uint32_t>(value.emitted_tokens.size());
            records.push_back(rec);
        }
        out_tokens.insert(out_tokens.end(), value.emitted_tokens.begin(),
                          value.emitted_tokens.end());
        for (std::size_t j = 0u; j < value.emitted_tokens.size(); ++j) {
            const std::size_t g = out_tokens.size() - value.emitted_tokens.size() + j;
            if (g < reference.size() && value.emitted_tokens[j] != reference[g]) {
                std::printf("GATE4E_ITR %s ctx=%u K=%u itr=%u accepted=%u emitted=%zu "
                            "gpos=%zu spec=%d ref=%d\n",
                            category.c_str(), ctx, k, iteration, value.num_accepted_drafts,
                            value.emitted_tokens.size(), g, value.emitted_tokens[j],
                            reference[g]);
                break;
            }
        }
        pending = value.pending_token;
        ++iteration;
        if (value.finished) break;
    }
    if (out_tokens.size() > gen_tokens) out_tokens.resize(gen_tokens);
    return true;
}

struct Key {
    std::string category;
    uint32_t ctx = 0u;
    uint32_t k = 0u;
    bool operator==(const Key& o) const {
        return category == o.category && ctx == o.ctx && k == o.k;
    }
};

}  // namespace

int main() {
    const char* model_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_env == nullptr || model_env[0] == '\0') {
        model_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    const char* dir_env = std::getenv("PHASESHIFT_MTP_GATE4E_DIR");
    const std::string root =
        (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "artifacts/mtp_gate4e";
    const std::string corpus_dir = root + "/corpus";
    if (model_env == nullptr || !fs::is_directory(corpus_dir)) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP and provide %s\n", corpus_dir.c_str());
        return 77;
    }
    const uint32_t gen_tokens =
        std::getenv("PHASESHIFT_MTP_GATE4E_GEN")
            ? static_cast<uint32_t>(std::strtoul(std::getenv("PHASESHIFT_MTP_GATE4E_GEN"), nullptr, 10))
            : 256u;
    const uint32_t max_per_bucket =
        std::getenv("PHASESHIFT_MTP_GATE4E_PROMPTS_PER_BUCKET")
            ? static_cast<uint32_t>(std::strtoul(
                  std::getenv("PHASESHIFT_MTP_GATE4E_PROMPTS_PER_BUCKET"), nullptr, 10))
            : 2u;

    std::vector<uint32_t> lengths = {32u, 128u, 512u, 2048u};
    if (const char* len_env = std::getenv("PHASESHIFT_MTP_GATE4E_LENGTHS")) {
        lengths.clear();
        std::string s = len_env;
        std::size_t pos = 0;
        while (pos < s.size()) {
            std::size_t comma = s.find(',', pos);
            std::string part = s.substr(pos, comma == std::string::npos ? std::string::npos
                                                                        : comma - pos);
            if (!part.empty()) lengths.push_back(static_cast<uint32_t>(std::strtoul(part.c_str(), nullptr, 10)));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    uint32_t max_used_len = 32u;
    for (uint32_t l : lengths) max_used_len = std::max(max_used_len, l);
    const uint32_t max_len = max_used_len + gen_tokens + 16u;
    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena\n");
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result = qwen35::Qwen35Model::load_from_safetensors(model_env, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL: model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    qwen35::Qwen35Model model = model_result.release();
    std::fprintf(stderr, "[g4e] after model: arena used=%.1f GB cap=%.1f GB\n",
                 static_cast<double>(arena.used()) / 1e9,
                 static_cast<double>(arena.capacity()) / 1e9);
    if (!model.weights().mtp.present) {
        std::printf("FAIL: model has no MTP weights\n");
        return 1;
    }
    const qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t page_tokens = 16u;
    const uint32_t num_pages = (ceil_div(max_len, page_tokens) + 4u) * 4u;

    auto seq_pool_result = qwen35::SequenceSlotPool::create(arena, 2u, num_pages + 16u);
    auto gdn_pool_result = qwen35::GdnStatePool::create(
        arena, 2u, qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, count_attention_layers(tc),
        static_cast<uint32_t>(tc.num_key_value_heads),
        static_cast<uint32_t>(tc.attention_head_dim), qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL: pool create\n");
        return 1;
    }
    qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = max_len;
    exec_config.max_scheduled_requests = 1u;
    exec_config.max_scheduled_output_rows = 16u;
    auto exec_result = qwen35::create_model_executor(
        model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("FAIL: executor: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    qwen35::Executor executor = exec_result.release();
    std::fprintf(stderr, "[g4e] after executor: arena used=%.1f GB\n",
                 static_cast<double>(arena.used()) / 1e9);

    q35rt::MtpExecutorConfig mtp_config;
    mtp_config.num_pages = num_pages + 16u;
    mtp_config.page_tokens = page_tokens;
    mtp_config.max_rows = 16u;
    auto mtp_result = q35rt::create_mtp_executor(model, qwen35::KVCacheDType::BF16, arena,
                                                 mtp_config, stream);
    if (!mtp_result.ok()) {
        std::printf("FAIL: mtp executor\n");
        return 1;
    }
    q35rt::MtpExecutor mtp = mtp_result.release();
    std::fprintf(stderr, "[g4e] after mtp: arena used=%.1f GB\n",
                 static_cast<double>(arena.used()) / 1e9);
    std::fprintf(stderr, "[g4e] gdn conv=%zu rec=%zu hidden=%zu mtp_pages=%u num_pages=%u\n",
                 q35rt::spec_gdn_conv_bytes(gdn_pool), q35rt::spec_gdn_recurrent_bytes(gdn_pool),
                 static_cast<std::size_t>(tc.hidden_size), mtp_config.num_pages, num_pages);

    const char* exact_env = std::getenv("PHASESHIFT_VERIFY_EXACT");
    const bool verify_exact = exact_env != nullptr && exact_env[0] == '1';

    std::vector<std::string> categories;
    const char* only_env = std::getenv("PHASESHIFT_MTP_GATE4E_ONLY");
    for (const auto& e : fs::directory_iterator(corpus_dir)) {
        if (e.path().extension() == ".psktok") {
            const std::string name = e.path().stem().string();
            if (only_env != nullptr && only_env[0] != '\0' && name != only_env) continue;
            categories.push_back(name);
        }
    }
    std::sort(categories.begin(), categories.end());

    std::vector<IterRec> raw;
    struct Agg {
        std::string category;
        uint32_t ctx = 0u;
        uint32_t k = 0u;
        uint32_t updates = 0u;
        uint32_t emitted = 0u;
        uint32_t accepted = 0u;
        uint32_t full = 0u;
        uint32_t reject0 = 0u;
        double total_ms = 0.0;
        double draft_ms = 0.0;
        double verify_ms = 0.0;
        std::vector<uint32_t> accept_hist;
    };
    std::vector<Agg> aggs;
    struct KMode {
        uint32_t kmax;
        bool dynamic;
    };
    std::vector<KMode> k_modes = {{1u, false}, {2u, false}, {4u, false}, {8u, false}};
    {
        const char* dyn_env = std::getenv("PHASESHIFT_MTP_GATE4E_DYNAMIC");
        if (dyn_env != nullptr && dyn_env[0] != '\0') {
            uint32_t kmax = 8u;
            if (const char* e = std::getenv("PHASESHIFT_MTP_DRAFT_KMAX"))
                kmax = static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
            k_modes.push_back({kmax, true});
        }
    }
    std::vector<std::pair<std::string, double>> target_cost;  // "cat:ctx" -> ms/token
    std::vector<std::pair<std::string, uint32_t>> token_matches;
    std::vector<std::pair<float, uint8_t>> draft_log;
    const bool want_draft_log =
        std::getenv("PHASESHIFT_MTP_GATE4E_DRAFT_LOG") != nullptr;

    q35rt::MtpKvConfig mtp_kv_config;
    mtp_kv_config.num_pages = num_pages + 16u;
    mtp_kv_config.page_tokens = page_tokens;
    mtp_kv_config.num_attention_layers = 1u;
    mtp_kv_config.kv_heads = 4u;
    mtp_kv_config.head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    mtp_kv_config.dtype = qwen35::KVCacheDType::BF16;

    std::fprintf(stderr, "[g4e] categories=%zu lengths=%zu max_per_bucket=%u gen=%u max_len=%u\n",
                 categories.size(), lengths.size(), max_per_bucket, gen_tokens, max_len);
    for (uint32_t l : lengths) std::fprintf(stderr, "[g4e] length=%u\n", l);

    uint32_t prompt_id = 0u;
    for (const std::string& cat : categories) {
        std::vector<int32_t> corpus = read_corpus(fs::path(corpus_dir) / (cat + ".psktok"));
        if (corpus.size() < 32u) continue;
        for (uint32_t len : lengths) {
            if (corpus.size() < len) continue;
            uint32_t taken = 0u;
            for (uint32_t off = 0u; off + len <= corpus.size() && taken < max_per_bucket;
                 off += len) {
                ++taken;
                std::vector<int32_t> prompt(corpus.begin() + off, corpus.begin() + off + len);
                const uint32_t id = prompt_id++;
                const uint32_t required = len + gen_tokens + 16u;

                TargetStats target;
                auto wt0 = std::chrono::steady_clock::now();
                {
                    auto seq_result = qwen35::create_paged_sequence_state(
                        seq_pool, gdn_pool, kv_pool, required, stream);
                    if (!seq_result.ok()) { std::printf("FAIL: seq\n"); return 1; }
                    qwen35::PagedSequenceState seq = seq_result.release();
                    if (!run_target_only(executor, seq, prompt, gen_tokens, target, stream)) return 1;
                    (void)qwen35::release_paged_sequence_state(seq, stream);
                }
                std::fprintf(stderr, "[g4e] target id=%u cat=%s ctx=%u ms=%.1f toks=%u\n", id,
                             cat.c_str(), len, wall_ms(wt0), target.decode_tokens);
                target_cost.emplace_back(cat + ":" + std::to_string(len),
                                         target.decode_ms / std::max(1u, target.decode_tokens));

                for (const KMode& km : k_modes) {
                    const uint32_t k = km.kmax;
                    const bool is_dyn = km.dynamic;
                    auto mtp_state_result = q35rt::create_mtp_kv_state(arena, mtp_kv_config, 0u, stream);
                    if (!mtp_state_result.ok()) { std::printf("FAIL: mtp state\n"); return 1; }
                    q35rt::MtpKvState mtp_state = mtp_state_result.release();
                    auto seq_result = qwen35::create_paged_sequence_state(
                        seq_pool, gdn_pool, kv_pool, required, stream);
                    if (!seq_result.ok()) { std::printf("FAIL: seq\n"); return 1; }
                    qwen35::PagedSequenceState seq = seq_result.release();
                    q35rt::SpecDecoderConfig dcfg_k;
                    dcfg_k.num_drafts = k;
                    dcfg_k.bonus_token_enabled = true;
                    if (is_dyn) {
                        dcfg_k.draft_policy.dynamic = true;
                        dcfg_k.draft_policy.min_drafts =
                            std::getenv("PHASESHIFT_MTP_DRAFT_MIN")
                                ? static_cast<uint32_t>(std::strtoul(
                                      std::getenv("PHASESHIFT_MTP_DRAFT_MIN"), nullptr, 10))
                                : 1u;
                        dcfg_k.draft_policy.stop_margin =
                            std::getenv("PHASESHIFT_MTP_DRAFT_TAU")
                                ? std::strtof(std::getenv("PHASESHIFT_MTP_DRAFT_TAU"), nullptr)
                                : 0.0f;
                        dcfg_k.draft_policy.enable_discard =
                            std::getenv("PHASESHIFT_MTP_DRAFT_DISCARD") != nullptr;
                        dcfg_k.draft_policy.discard_margin =
                            std::getenv("PHASESHIFT_MTP_DRAFT_DISCARD_TAU")
                                ? std::strtof(
                                      std::getenv("PHASESHIFT_MTP_DRAFT_DISCARD_TAU"), nullptr)
                                : -1.0e30f;
                    }
                    if (const char* ng = std::getenv("PHASESHIFT_MTP_NGRAM_N"))
                        dcfg_k.ngram_n = static_cast<uint32_t>(std::strtoul(ng, nullptr, 10));
                    if (const char* nt = std::getenv("PHASESHIFT_MTP_NGRAM_TAIL"))
                        dcfg_k.ngram_max_tail =
                            static_cast<uint32_t>(std::strtoul(nt, nullptr, 10));
                    dcfg_k.verify_numeric_mode = verify_exact
                        ? ps::runtime::VerifyNumericMode::Exact
                        : ps::runtime::VerifyNumericMode::Fast;
                    auto decoder_result = q35rt::create_spec_decoder(
                        executor, mtp, mtp_state, seq, gdn_pool, arena, dcfg_k, stream);
                    if (!decoder_result.ok()) {
                        std::printf("FAIL: spec decoder: %s\n",
                                    decoder_result.status().message().c_str());
                        return 1;
                    }
                    q35rt::SpecDecoder decoder = decoder_result.release();
                    q35rt::SpecDecoderTiming timing{};
                    decoder.timing = &timing;
                    if (want_draft_log) decoder.draft_log = &draft_log;
                    std::vector<int32_t> spec_tokens;
                    std::size_t raw_before = raw.size();
                    auto wk0 = std::chrono::steady_clock::now();
                    if (!run_spec(decoder, executor, seq, prompt, gen_tokens, k, id, cat, len,
                                  target.tokens, raw, spec_tokens, stream)) {
                        return 1;
                    }
                    std::fprintf(stderr, "[g4e] spec id=%u k=%u iters=%zu ms=%.1f\n", id, k,
                                 raw.size() - raw_before, wall_ms(wk0));
                    const bool matched = spec_tokens == target.tokens;
                    if (!matched) {
                        std::size_t d = 0u;
                        while (d < spec_tokens.size() && d < target.tokens.size() &&
                               spec_tokens[d] == target.tokens[d]) ++d;
                        std::printf("GATE4E_DIVERGE %s ctx=%u K=%u idx=%zu spec=%d target=%d\n",
                                    cat.c_str(), len, k, d,
                                    d < spec_tokens.size() ? spec_tokens[d] : -1,
                                    d < target.tokens.size() ? target.tokens[d] : -1);
                    }
                    token_matches.emplace_back(
                        cat + ":" + std::to_string(len) +
                            (is_dyn ? (":DYN" + std::to_string(k)) : (":K" + std::to_string(k))),
                        matched ? 1u : 0u);
                    Agg agg;
                    agg.category = cat;
                    agg.ctx = len;
                    agg.k = is_dyn ? 100u : k;
                    agg.accept_hist.assign(k + 1u, 0u);
                    for (std::size_t ri = raw_before; ri < raw.size(); ++ri) {
                        const IterRec& r = raw[ri];
                        agg.updates += 1u;
                        agg.emitted += r.num_emitted;
                        agg.accepted += r.num_accepted;
                        agg.total_ms += r.total_ms;
                        agg.draft_ms += r.draft_ms;
                        agg.verify_ms += r.verify_ms;
                        if (r.num_accepted == k) ++agg.full;
                        if (r.num_accepted == 0u) ++agg.reject0;
                        if (r.num_accepted <= k) agg.accept_hist[r.num_accepted] += 1u;
                    }
                    aggs.push_back(std::move(agg));
                    (void)q35rt::spec_decoder_shutdown(decoder);
                    (void)q35rt::mtp_kv_shutdown(mtp_state);
                    (void)qwen35::release_paged_sequence_state(seq, stream);
                }
            }
        }
    }

    fs::create_directories(root);
    if (want_draft_log) {
        std::ofstream dl(root + "/draft_margin_log.csv");
        dl << "margin,accepted\n";
        for (const auto& p : draft_log)
            dl << p.first << "," << static_cast<int>(p.second) << "\n";
    }
    std::ofstream raw_csv(root + "/raw_iterations.csv");
    raw_csv << "prompt_id,category,context,k,iteration,draft_ms,verify_ms,snapshot_ms,"
               "restore_ms,commit_ms,total_ms,num_drafts,num_accepted,num_emitted,reject_at,"
               "full_accept\n";
    for (const IterRec& r : raw) {
        raw_csv << r.prompt_id << ',' << r.category << ',' << r.ctx << ',' << r.k << ','
                << r.iteration << ',' << r.draft_ms << ',' << r.verify_ms << ',' << r.snap_ms
                << ',' << r.restore_ms << ',' << r.commit_ms << ',' << r.total_ms << ','
                << r.num_drafts << ',' << r.num_accepted << ',' << r.num_emitted << ','
                << (r.num_accepted < r.num_drafts ? static_cast<int>(r.num_accepted) : -1) << ','
                << (r.num_accepted == r.num_drafts ? 1 : 0) << '\n';
    }
    raw_csv.close();

    std::ofstream agg_csv(root + "/aggregate.csv");
    agg_csv << "category,context,k,updates,emitted,accepted,emitted_per_update,"
               "accepted_per_update,ms_per_update,ms_per_token,draft_ms_per_update,"
               "verify_ms_per_update,reject0,full_accept,target_ms_per_token,speedup,"
               "required_emitted_per_update,required_cost_reduction\n";
    std::printf("GATE4E_AGG context k updates emitted_per_update ms_per_token target speedup "
                "req_E req_cost_red\n");
    for (const Agg& a : aggs) {
        if (a.updates == 0u) continue;
        double target_ms = 0.0;
        for (const auto& tc : target_cost) {
            if (tc.first == a.category + ":" + std::to_string(a.ctx)) target_ms = tc.second;
        }
        const double e_per_update = static_cast<double>(a.emitted) / a.updates;
        const double ms_per_update = a.total_ms / a.updates;
        const double ms_per_token = a.emitted > 0u ? a.total_ms / a.emitted : 0.0;
        const double speedup = ms_per_token > 0.0 ? target_ms / ms_per_token : 0.0;
        const double req_E = target_ms > 0.0 ? ms_per_update / target_ms : 0.0;
        const double req_red = (target_ms > 0.0 && ms_per_update > 0.0)
            ? 1.0 - (target_ms * e_per_update) / ms_per_update : 0.0;
        agg_csv << a.category << ',' << a.ctx << ',' << a.k << ',' << a.updates << ',' << a.emitted
                << ',' << a.accepted << ',' << e_per_update << ','
                << static_cast<double>(a.accepted) / a.updates << ',' << ms_per_update << ','
                << ms_per_token << ',' << a.draft_ms / a.updates << ','
                << a.verify_ms / a.updates << ',' << a.reject0 << ',' << a.full << ',' << target_ms
                << ',' << speedup << ',' << req_E << ',' << req_red << '\n';
        std::printf("GATE4E_AGG %u %u %u %.3f %.3f %.3f %.4f %.3f %.4f\n", a.ctx, a.k,
                    a.updates, e_per_update, ms_per_token, target_ms, speedup, req_E, req_red);
    }
    agg_csv.close();

    std::ofstream env(root + "/environment.json");
    env << "{\n  \"gpu\": \"AMD Radeon AI PRO R9700\",\n  \"kernel_mode\": \"optimized\",\n"
        << "  \"verify_exact\": " << (verify_exact ? "true" : "false") << ",\n"
        << "  \"gen_tokens\": " << gen_tokens << ",\n"
        << "  \"prompts_per_bucket\": " << max_per_bucket << ",\n"
        << "  \"prompts\": " << prompt_id << ",\n"
        << "  \"categories\": [";
    for (std::size_t i = 0; i < categories.size(); ++i) {
        env << (i ? ", " : "") << '"' << categories[i] << '"';
    }
    env << "]\n}\n";
    env.close();

    std::printf("GATE4E_CATEGORIES=%zu PROMPTS=%u\n", categories.size(), prompt_id);
    uint32_t match_count = 0u;
    for (const auto& tm : token_matches) {
        match_count += tm.second;
        if (tm.second == 0u) std::printf("GATE4E_MISMATCH %s\n", tm.first.c_str());
    }
    std::printf("GATE4E_TOKEN_MATCH=%u/%zu\n", match_count, token_matches.size());
    std::printf("GATE4E_DONE\n");
    (void)qwen35::executor_shutdown(executor);
    (void)q35rt::mtp_executor_shutdown(mtp);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return 0;
}
