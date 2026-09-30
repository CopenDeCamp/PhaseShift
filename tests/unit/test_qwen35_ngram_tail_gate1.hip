#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/ngram_tail.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
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

int g_leak_failures = 0;

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

std::vector<uint32_t> parse_u32_list(const char* value, const char* fallback) {
    std::string s = (value != nullptr && value[0] != '\0') ? value : fallback;
    std::vector<uint32_t> out;
    std::size_t pos = 0;
    while (pos < s.size()) {
        std::size_t comma = s.find(',', pos);
        std::string part = s.substr(pos, comma == std::string::npos ? std::string::npos
                                                                    : comma - pos);
        if (!part.empty()) out.push_back(static_cast<uint32_t>(std::strtoul(part.c_str(), nullptr, 10)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
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

struct GenStream {
    std::string category;
    uint32_t ctx = 0u;
    uint32_t prompt_id = 0u;
    double target_ms_per_token = 0.0;
    std::vector<int32_t> prompt;
    std::vector<int32_t> tokens;
};

struct RawRow {
    std::string category;
    uint32_t ctx = 0u;
    uint32_t prompt_id = 0u;
    uint32_t position = 0u;
    uint32_t n = 0u;
    uint32_t max_tail = 0u;
    uint32_t window = 0u;
    bool hit = false;
    uint32_t candidate_position = 0u;
    uint32_t proposed = 0u;
    uint32_t accepted = 0u;
};

struct Agg {
    std::string category;
    uint32_t ctx = 0u;
    uint32_t n = 0u;
    uint32_t max_tail = 0u;
    uint32_t window = 0u;
    uint64_t positions = 0u;
    uint64_t hits = 0u;
    uint64_t proposed_total = 0u;
    uint64_t accepted_total = 0u;
    uint64_t ideal_sum = 0u;
    uint64_t ge4 = 0u;
    uint64_t ge8 = 0u;
    uint64_t ge16 = 0u;
    uint64_t accept_hist[7] = {0, 0, 0, 0, 0, 0, 0};
    double lookup_ms = 0.0;

    double hit_rate() const {
        return positions > 0u ? static_cast<double>(hits) / static_cast<double>(positions) : 0.0;
    }
    double acceptance_rate() const {
        return proposed_total > 0u
            ? static_cast<double>(accepted_total) / static_cast<double>(proposed_total) : 0.0;
    }
    double accepted_per_position() const {
        return positions > 0u ? static_cast<double>(accepted_total) / static_cast<double>(positions)
                              : 0.0;
    }
    double accepted_per_hit() const {
        return hits > 0u ? static_cast<double>(accepted_total) / static_cast<double>(hits) : 0.0;
    }
    double ideal_emitted_per_update() const {
        return positions > 0u ? static_cast<double>(ideal_sum) / static_cast<double>(positions)
                              : 0.0;
    }
    double ge_rate(uint64_t count) const {
        return hits > 0u ? static_cast<double>(count) / static_cast<double>(hits) : 0.0;
    }
};

void fail_leak(const char* what, const q35rt::NgramTailProposal& p, std::size_t history_size) {
    std::printf("FAIL: future leakage (%s) candidate_start=%u candidate_end=%u "
                "current_position=%u history_size=%zu\n",
                what, p.candidate_start, p.candidate_end, p.current_position, history_size);
    ++g_leak_failures;
}

void replay(
    const GenStream& stream,
    const q35rt::NgramTailConfig& cfg,
    std::vector<RawRow>& rows,
    Agg& agg) {
    std::vector<int32_t> history = stream.prompt;
    history.reserve(stream.prompt.size() + stream.tokens.size());

    for (std::size_t p = 0u; p < stream.tokens.size(); ++p) {
        if (history.size() != stream.prompt.size() + p) {
            fail_leak("history size", q35rt::NgramTailProposal{}, history.size());
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        auto result = q35rt::propose_ngram_tail(history, cfg);
        agg.lookup_ms += wall_ms(t0);
        if (!result.ok()) {
            std::printf("FAIL: propose_ngram_tail: %s\n", result.status().message().c_str());
            ++g_leak_failures;
            return;
        }
        const q35rt::NgramTailProposal& prop = result.value();
        ++agg.positions;

        RawRow row;
        row.category = stream.category;
        row.ctx = stream.ctx;
        row.prompt_id = stream.prompt_id;
        row.position = static_cast<uint32_t>(p);
        row.n = cfg.n;
        row.max_tail = cfg.max_tail;
        row.window = cfg.window;

        if (prop.hit) {
            if (prop.current_position != history.size()) fail_leak("position", prop, history.size());
            if (prop.candidate_start >= prop.current_position) fail_leak("start", prop, history.size());
            if (prop.candidate_end > prop.current_position) fail_leak("end", prop, history.size());
            if (prop.candidate_end + prop.tokens.size() > prop.current_position) {
                fail_leak("continuation", prop, history.size());
            }
            bool verbatim = true;
            for (std::size_t i = 0u; i < prop.tokens.size(); ++i) {
                if (prop.candidate_end + i >= history.size() ||
                    history[prop.candidate_end + i] != prop.tokens[i]) {
                    verbatim = false;
                    break;
                }
            }
            if (!verbatim) fail_leak("verbatim", prop, history.size());

            const std::size_t available = stream.tokens.size() - p;
            const uint32_t proposed =
                static_cast<uint32_t>(std::min<std::size_t>(prop.tokens.size(), available));
            uint32_t accepted = 0u;
            while (accepted < proposed && prop.tokens[accepted] == stream.tokens[p + accepted]) {
                ++accepted;
            }

            row.hit = true;
            row.candidate_position = prop.candidate_start;
            row.proposed = proposed;
            row.accepted = accepted;

            ++agg.hits;
            agg.proposed_total += proposed;
            agg.accepted_total += accepted;
            agg.ideal_sum += accepted + 1u;
            if (accepted >= 4u) ++agg.ge4;
            if (accepted >= 8u) ++agg.ge8;
            if (accepted >= 16u) ++agg.ge16;
            if (accepted == 0u) {
                ++agg.accept_hist[0];
            } else if (accepted == 1u) {
                ++agg.accept_hist[1];
            } else if (accepted == 2u) {
                ++agg.accept_hist[2];
            } else if (accepted <= 4u) {
                ++agg.accept_hist[3];
            } else if (accepted <= 8u) {
                ++agg.accept_hist[4];
            } else if (accepted <= 16u) {
                ++agg.accept_hist[5];
            } else {
                ++agg.accept_hist[6];
            }
        } else {
            row.hit = false;
            row.candidate_position = 0u;
            row.proposed = 0u;
            row.accepted = 0u;
            agg.ideal_sum += 1u;
        }

        rows.push_back(row);
        history.push_back(stream.tokens[p]);
    }
}

Agg* find_agg(
    std::vector<Agg>& aggs,
    const std::string& category,
    uint32_t ctx,
    uint32_t n,
    uint32_t max_tail,
    uint32_t window) {
    for (Agg& a : aggs) {
        if (a.category == category && a.ctx == ctx && a.n == n && a.max_tail == max_tail &&
            a.window == window) {
            return &a;
        }
    }
    aggs.push_back(Agg{});
    Agg& a = aggs.back();
    a.category = category;
    a.ctx = ctx;
    a.n = n;
    a.max_tail = max_tail;
    a.window = window;
    return &a;
}

}  // namespace

int main() {
    const char* model_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_env == nullptr || model_env[0] == '\0') {
        model_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    const char* corpus_env = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_CORPUS");
    const std::string corpus_dir =
        (corpus_env != nullptr && corpus_env[0] != '\0') ? corpus_env : "tests/data/mtp_perf";
    const char* dir_env = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_DIR");
    const std::string root =
        (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "artifacts/ngram_tail_gate1";
    if (model_env == nullptr || !fs::is_directory(corpus_dir)) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP and provide %s\n", corpus_dir.c_str());
        return 77;
    }

    const uint32_t gen_tokens = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_GEN") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_GEN") : "256",
        nullptr, 10));
    const uint32_t prompts_per_bucket = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_PROMPTS") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_PROMPTS") : "2",
        nullptr, 10));
    const std::vector<uint32_t> contexts = parse_u32_list(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_CONTEXTS"), "512");
    const std::vector<uint32_t> n_values =
        parse_u32_list(std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_N"), "3,4,5,8");
    const std::vector<uint32_t> tail_values =
        parse_u32_list(std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_TAIL"), "8,16,32");
    const std::vector<uint32_t> window_values =
        parse_u32_list(std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_WINDOW"), "2048");
    const char* only_env = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE1_ONLY");

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    uint32_t max_used_len = 32u;
    for (uint32_t l : contexts) max_used_len = std::max(max_used_len, l);
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
    auto exec_result =
        qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("FAIL: executor: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    qwen35::Executor executor = exec_result.release();

    std::vector<std::string> categories;
    for (const auto& e : fs::directory_iterator(corpus_dir)) {
        if (e.path().extension() == ".psktok") {
            const std::string name = e.path().stem().string();
            if (only_env != nullptr && only_env[0] != '\0' && name != only_env) continue;
            categories.push_back(name);
        }
    }
    std::sort(categories.begin(), categories.end());
    if (categories.empty()) {
        std::printf("FAIL: no corpus in %s\n", corpus_dir.c_str());
        return 1;
    }

    std::fprintf(stderr,
                 "[g1] corpus=%s gen=%u prompts=%u contexts=%zu n=%zu tail=%zu window=%zu\n",
                 corpus_dir.c_str(), gen_tokens, prompts_per_bucket, contexts.size(),
                 n_values.size(), tail_values.size(), window_values.size());

    std::vector<GenStream> streams;
    uint32_t prompt_id = 0u;
    for (const std::string& cat : categories) {
        const std::vector<int32_t> corpus = read_corpus(fs::path(corpus_dir) / (cat + ".psktok"));
        if (corpus.empty()) {
            std::printf("FAIL: unreadable corpus %s\n", cat.c_str());
            return 1;
        }
        for (uint32_t len : contexts) {
            if (corpus.size() < len) {
                std::printf("FAIL: corpus %s has %zu tokens < context %u\n", cat.c_str(),
                            corpus.size(), len);
                return 1;
            }
            uint32_t taken = 0u;
            for (uint32_t off = 0u; off + len <= corpus.size() && taken < prompts_per_bucket;
                 off += len) {
                ++taken;
                GenStream stream_rec;
                stream_rec.category = cat;
                stream_rec.ctx = len;
                stream_rec.prompt_id = prompt_id++;
                stream_rec.prompt.assign(corpus.begin() + off, corpus.begin() + off + len);
                const uint32_t required = len + gen_tokens + 16u;
                TargetStats target;
                auto wt0 = std::chrono::steady_clock::now();
                {
                    auto seq_result =
                        qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, required,
                                                            stream);
                    if (!seq_result.ok()) {
                        std::printf("FAIL: seq\n");
                        return 1;
                    }
                    qwen35::PagedSequenceState seq = seq_result.release();
                    if (!run_target_only(executor, seq, stream_rec.prompt, gen_tokens, target,
                                         stream)) {
                        return 1;
                    }
                    (void)qwen35::release_paged_sequence_state(seq, stream);
                }
                stream_rec.tokens = target.tokens;
                stream_rec.target_ms_per_token =
                    target.decode_ms / std::max(1u, target.decode_tokens);
                std::fprintf(stderr, "[g1] target id=%u cat=%s ctx=%u ms=%.1f toks=%u\n",
                             stream_rec.prompt_id, cat.c_str(), len, wall_ms(wt0),
                             target.decode_tokens);
                streams.push_back(std::move(stream_rec));
            }
        }
    }

    std::vector<RawRow> rows;
    std::vector<Agg> aggs;
    for (const GenStream& s : streams) {
        for (uint32_t n : n_values) {
            for (uint32_t tail : tail_values) {
                for (uint32_t window : window_values) {
                    q35rt::NgramTailConfig cfg;
                    cfg.n = n;
                    cfg.max_tail = tail;
                    cfg.window = window;
                    Agg* agg = find_agg(aggs, s.category, s.ctx, n, tail, window);
                    replay(s, cfg, rows, *agg);
                }
            }
        }
    }

    if (g_leak_failures > 0) {
        std::printf("FAIL: future leakage detected (%d)\n", g_leak_failures);
        return 1;
    }

    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) {
        std::printf("FAIL: create %s: %s\n", root.c_str(), ec.message().c_str());
        return 1;
    }

    std::ofstream raw_csv(root + "/raw.csv");
    raw_csv << "category,context,prompt_id,generation_position,n,max_tail,window,hit,"
               "candidate_position,proposed,accepted\n";
    for (const RawRow& r : rows) {
        raw_csv << r.category << ',' << r.ctx << ',' << r.prompt_id << ',' << r.position << ','
                << r.n << ',' << r.max_tail << ',' << r.window << ',' << (r.hit ? 1 : 0) << ','
                << r.candidate_position << ',' << r.proposed << ',' << r.accepted << '\n';
    }
    raw_csv.close();

    std::ofstream agg_csv(root + "/aggregate.csv");
    agg_csv << "category,context,n,max_tail,window,positions,hits,hit_rate,proposed_total,"
               "accepted_total,acceptance_rate,accepted_per_position,accepted_per_hit,"
               "ideal_emitted_per_update,accept_ge4_rate,accept_ge8_rate,accept_ge16_rate,"
               "accept_0,accept_1,accept_2,accept_3_4,accept_5_8,accept_9_16,accept_17_32,"
               "lookup_ms_total,lookup_ns_per_position\n";
    std::printf("GATE1_AGG category context n tail window positions hits hit_rate "
                "acceptance_rate accepted_per_position accepted_per_hit ge4 ge8 ge16 ideal\n");
    for (const Agg& a : aggs) {
        if (a.positions == 0u) continue;
        agg_csv << a.category << ',' << a.ctx << ',' << a.n << ',' << a.max_tail << ','
                << a.window << ',' << a.positions << ',' << a.hits << ',' << a.hit_rate() << ','
                << a.proposed_total << ',' << a.accepted_total << ',' << a.acceptance_rate() << ','
                << a.accepted_per_position() << ',' << a.accepted_per_hit() << ','
                << a.ideal_emitted_per_update() << ',' << a.ge_rate(a.ge4) << ','
                << a.ge_rate(a.ge8) << ',' << a.ge_rate(a.ge16);
        for (uint64_t h : a.accept_hist) agg_csv << ',' << h;
        agg_csv << ',' << a.lookup_ms << ','
                << (a.lookup_ms * 1e6 / static_cast<double>(a.positions)) << '\n';
        std::printf("GATE1_AGG %s %u %u %u %u %llu %llu %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                    a.category.c_str(), a.ctx, a.n, a.max_tail, a.window,
                    static_cast<unsigned long long>(a.positions),
                    static_cast<unsigned long long>(a.hits), a.hit_rate(), a.acceptance_rate(),
                    a.accepted_per_position(), a.accepted_per_hit(), a.ge_rate(a.ge4),
                    a.ge_rate(a.ge8), a.ge_rate(a.ge16), a.ideal_emitted_per_update());
    }
    agg_csv.close();

    const bool quantized = fs::exists(fs::path(model_env) / "phaseshift_quantization.json");
    std::ofstream env(root + "/environment.json");
    env << "{\n"
        << "  \"gpu\": \"AMD Radeon AI PRO R9700\",\n"
        << "  \"arch\": \"gfx1201\",\n"
        << "  \"model_dir\": \"" << model_env << "\",\n"
        << "  \"quantization\": \"" << (quantized ? "psq" : "bf16") << "\",\n"
        << "  \"corpus_dir\": \"" << corpus_dir << "\",\n"
        << "  \"gen_tokens\": " << gen_tokens << ",\n"
        << "  \"prompts_per_bucket\": " << prompts_per_bucket << ",\n"
        << "  \"streams\": " << streams.size() << ",\n"
        << "  \"raw_rows\": " << rows.size() << ",\n"
        << "  \"verify_numeric_mode\": \"fast\",\n"
        << "  \"contexts\": [";
    for (std::size_t i = 0u; i < contexts.size(); ++i) {
        env << (i ? ", " : "") << contexts[i];
    }
    env << "],\n  \"n\": [";
    for (std::size_t i = 0u; i < n_values.size(); ++i) env << (i ? ", " : "") << n_values[i];
    env << "],\n  \"max_tail\": [";
    for (std::size_t i = 0u; i < tail_values.size(); ++i) env << (i ? ", " : "") << tail_values[i];
    env << "],\n  \"window\": [";
    for (std::size_t i = 0u; i < window_values.size(); ++i) {
        env << (i ? ", " : "") << window_values[i];
    }
    env << "],\n  \"categories\": [";
    for (std::size_t i = 0u; i < categories.size(); ++i) {
        env << (i ? ", " : "") << '"' << categories[i] << '"';
    }
    env << "]\n}\n";
    env.close();

    std::printf("GATE1_STREAMS=%zu ROWS=%zu AGGS=%zu\n", streams.size(), rows.size(), aggs.size());
    std::printf("GATE1_DONE\n");
    (void)qwen35::executor_shutdown(executor);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return 0;
}
