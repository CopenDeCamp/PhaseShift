#include <phaseshift/quantization/offline/kld.h>
#include <phaseshift/quantization/offline/kld_logit_cache.h>
#include <phaseshift/quantization/offline/kld_metrics.h>
#include <phaseshift/quantization/offline/kld_shadow_model.h>
#include <phaseshift/quantization/offline/gpu_kld.h>
#include <phaseshift/quantization/offline/model_fingerprint.h>
#include <phaseshift/quantization/offline/token_corpus.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/schedule_plan.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/quantization/fpx/quantized_model_reader.h>
#include <openssl/sha.h>
#include <nlohmann/json.hpp>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

namespace ps::quantization::fpx {

namespace fs = std::filesystem;

namespace {

bool is_not_found_error(const std::error_code& ec) {
    return ec == std::errc::no_such_file_or_directory;
}

Result<std::string> read_shadow_fingerprint(const std::string& shadow_dir) {
    fs::path meta = fs::path(shadow_dir) / "phaseshift_qdq_shadow.json";
    std::ifstream f(meta);
    if (!f) return Status::invalid_argument(("cannot open shadow metadata: " + meta.string()).c_str(), __FILE__, __LINE__);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    try {
        auto j = nlohmann::json::parse(text);
        if (!j.contains("source_model_fingerprint") || !j["source_model_fingerprint"].is_string()) return Status::invalid_argument("shadow metadata fingerprint missing", __FILE__, __LINE__);
        return j["source_model_fingerprint"].get<std::string>();
    } catch (const std::exception& e) {
        return Status::invalid_argument(e.what(), __FILE__, __LINE__);
    }
}

struct PredictionMeta {
    uint32_t window_index = 0;
    uint32_t position_in_window = 0;
    uint64_t global_token_offset = 0;
    int32_t input_token = 0;
    int32_t target_token = 0;
};

struct PredictionPlan {
    std::vector<PredictionMeta> predictions;
    std::vector<uint32_t> window_starts;
    std::vector<uint32_t> window_counts;
    std::size_t windows = 0;
};

PredictionPlan build_prediction_plan(const TokenCorpus& corpus, std::size_t window, std::size_t stride) {
    PredictionPlan plan;
    std::size_t n = corpus.tokens.size();
    std::size_t start = 0;
    uint32_t wi = 0;
    while (start < n) {
        std::size_t end = std::min(start + window, n);
        std::size_t cnt = end - start;
        if (cnt < 2) break;
        plan.window_starts.push_back(static_cast<uint32_t>(start));
        plan.window_counts.push_back(static_cast<uint32_t>(cnt));
        for (std::size_t i = 0; i < cnt - 1; ++i) {
            PredictionMeta m;
            m.window_index = wi;
            m.position_in_window = static_cast<uint32_t>(i);
            m.global_token_offset = start + i;
            m.input_token = corpus.tokens[start + i];
            m.target_token = corpus.tokens[start + i + 1];
            plan.predictions.push_back(m);
        }
        ++wi;
        start += stride;
    }
    plan.windows = plan.window_starts.size();
    return plan;
}

std::string runtime_config_hash_value(std::size_t window, std::size_t stride, std::size_t batch) {
    return "paged:win" + std::to_string(window) + ":str" + std::to_string(stride) + ":batch" + std::to_string(batch);
}

Status hip_status(hipError_t err, const char* expr) {
    return err == hipSuccess ? Status::make_ok() : Status::hip_error(expr, hipGetErrorString(err), __FILE__, __LINE__);
}

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) if (lt != 0) ++n;
    return n;
}

Status ensure_cache_dir(const std::string& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return Status::invalid_argument(("cannot create cache dir: " + ec.message()).c_str(), __FILE__, __LINE__);
    return Status::make_ok();
}

}

Result<KldEvaluationSummary> evaluate_kld(const KldOptions& options) {
    if (options.input_dir.empty()) return Status::invalid_argument("input_dir is required", __FILE__, __LINE__);
    if (options.tokens_path.empty()) return Status::invalid_argument("tokens_path is required", __FILE__, __LINE__);
    if (options.report_path.empty()) return Status::invalid_argument("report_path is required", __FILE__, __LINE__);
    if (options.window == 0) return Status::invalid_argument("window must be > 0", __FILE__, __LINE__);
    if (options.stride == 0) return Status::invalid_argument("stride must be > 0", __FILE__, __LINE__);
    if (options.batch_positions == 0) return Status::invalid_argument("batch_positions must be > 0", __FILE__, __LINE__);
    if (options.arena_gib <= 0.0) return Status::invalid_argument("arena_gib must be > 0", __FILE__, __LINE__);
    if (!options.self_test && options.bundle_dir.empty()) return Status::invalid_argument("bundle_dir is required unless self_test", __FILE__, __LINE__);

    auto corpus_res = read_token_corpus(options.tokens_path);
    if (!corpus_res.ok()) return corpus_res.status();
    TokenCorpus corpus = corpus_res.release();
    if (options.max_eval_tokens > 0 && corpus.tokens.size() > options.max_eval_tokens) corpus.tokens.resize(options.max_eval_tokens);

    PredictionPlan plan = build_prediction_plan(corpus, options.window, options.stride);
    if (plan.predictions.empty()) return Status::invalid_argument("no predictions (corpus shorter than window?)", __FILE__, __LINE__);
    std::size_t prediction_count = plan.predictions.size();

    std::string model_fp = compute_model_fingerprint(options.input_dir);
    if (model_fp.empty()) return Status::invalid_argument("cannot compute model fingerprint", __FILE__, __LINE__);

    std::string corpus_sha = corpus.sha256_hex;
    std::string rc_hash = runtime_config_hash_value(options.window, options.stride, options.batch_positions);

    auto tc_res = ps::qwen35::read_qwen35_text_config(options.input_dir);
    if (!tc_res.ok()) return tc_res.status();
    ps::qwen35::Qwen35TextConfig text_config = tc_res.release();
    std::size_t vocab_size = text_config.vocab_size;
    if (vocab_size == 0) return Status::invalid_argument("vocab_size is zero", __FILE__, __LINE__);

    Status st = ensure_cache_dir(options.cache_dir);
    if (!st.ok()) return st;

    std::string sanitized_rc = rc_hash;
    for (char& c : sanitized_rc) if (c == ':') c = '_';
    std::string base_cache = (fs::path(options.cache_dir) / ("baseline_" + model_fp.substr(0, 12) + "_" + corpus_sha.substr(0, 12) + "_" + sanitized_rc + ".bin")).string();

    LogitCacheKey cache_key;
    cache_key.model_fingerprint = model_fp;
    cache_key.corpus_sha256 = corpus_sha;
    cache_key.vocab_size = vocab_size;
    cache_key.window = options.window;
    cache_key.stride = options.stride;
    cache_key.prediction_count = prediction_count;
    cache_key.runtime_config_hash = rc_hash;

    bool cache_valid = false;
    {
        auto m = logit_cache_matches(base_cache, cache_key);
        if (m.ok() && m.value()) cache_valid = true;
    }

    auto t_total_start = std::chrono::steady_clock::now();
    double t_shadow = 0, t_baseline_load = 0, t_baseline_infer = 0, t_candidate_load = 0, t_candidate_infer = 0, t_cpu_agg = 0;

    std::string shadow_dir;
    bool shadow_created = false;
    std::string candidate_dir;
    if (options.self_test) {
        candidate_dir = options.input_dir;
    } else {
        auto t0 = std::chrono::steady_clock::now();
        auto reader_res = QuantizedModelReader::open(options.bundle_dir);
        if (!reader_res.ok()) return reader_res.status();
        QuantizedModelReader reader = reader_res.release();
        if (reader.manifest().source_model_fingerprint != model_fp) return Status::invalid_argument("quantized model source fingerprint mismatch", __FILE__, __LINE__);
        {
            fs::path input_tok = fs::path(options.input_dir) / "tokenizer.json";
            fs::path bundle_tok = fs::path(options.bundle_dir) / "tokenizer.json";
            if (fs::exists(input_tok) && fs::exists(bundle_tok)) {
                auto h1_res = sha256_hex_file(input_tok.string());
                if (!h1_res.ok()) return h1_res.status();
                auto h2_res = sha256_hex_file(bundle_tok.string());
                if (!h2_res.ok()) return h2_res.status();
                if (h1_res.value() != h2_res.value()) return Status::invalid_argument("quantized model tokenizer mismatch", __FILE__, __LINE__);
            }
        }
        std::string shadow_base = (fs::path(options.cache_dir) / ("shadow_" + model_fp.substr(0, 12))).string();
        if (options.keep_shadow) {
            shadow_dir = shadow_base;
        } else {
            char tmpl[4096];
            std::snprintf(tmpl, sizeof(tmpl), "%s/shadow_XXXXXX", options.cache_dir.c_str());
            std::vector<char> buf(tmpl, tmpl + std::strlen(tmpl) + 1);
            char* d = mkdtemp(buf.data());
            if (!d) return Status::invalid_argument("cannot create shadow temp dir", __FILE__, __LINE__);
            shadow_dir = d;
        }
        if (fs::exists(shadow_dir)) {
            if (options.keep_shadow) {
                auto fp_res = read_shadow_fingerprint(shadow_dir);
                bool valid = fp_res.ok() && fp_res.value() == model_fp;
                if (valid) {
                    fs::path input_tok = fs::path(options.input_dir) / "tokenizer.json";
                    fs::path bundle_tok = fs::path(options.bundle_dir) / "tokenizer.json";
                    fs::path shadow_tok = fs::path(shadow_dir) / "tokenizer.json";
                    if (fs::exists(input_tok) && fs::exists(bundle_tok)) {
                        auto h1_res = sha256_hex_file(input_tok.string());
                        auto h2_res = sha256_hex_file(bundle_tok.string());
                        if (!h1_res.ok() || !h2_res.ok() || h1_res.value() != h2_res.value()) valid = false;
                    }
                    if (valid && fs::exists(shadow_tok) && fs::exists(input_tok)) {
                        auto h_in = sha256_hex_file(input_tok.string());
                        auto h_sh = sha256_hex_file(shadow_tok.string());
                        if (!h_in.ok() || !h_sh.ok() || h_in.value() != h_sh.value()) valid = false;
                    }
                    if (!valid) {
                        std::error_code ec;
                        fs::remove_all(shadow_dir, ec);
                    }
                } else {
                    std::error_code ec;
                    fs::remove_all(shadow_dir, ec);
                }
            } else {
                std::error_code ec;
                fs::remove_all(shadow_dir, ec);
            }
        }
        if (!fs::exists(shadow_dir)) {
            ShadowModelOptions sopts;
            sopts.input_dir = options.input_dir;
            sopts.quantized_dir = options.bundle_dir;
            sopts.output_dir = shadow_dir;
            auto sr = create_qdq_shadow_model(sopts);
            if (!sr.ok()) {
                if (!options.keep_shadow) { std::error_code ec; fs::remove_all(shadow_dir, ec); }
                return sr.status();
            }
            shadow_created = true;
        } else {
            shadow_created = true;
        }
        candidate_dir = shadow_dir;
        t_shadow = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    hipStream_t baseline_stream = nullptr;
    std::optional<ps::gpu::GpuArena> baseline_arena;
    std::optional<ps::qwen35::SequenceSlotPool> baseline_slot_pool;
    std::optional<ps::qwen35::GdnStatePool> baseline_gdn_pool;
    std::optional<ps::qwen35::PagedKVPool> baseline_kv_pool;
    ps::qwen35::Executor baseline_executor;
    bool baseline_executor_created = false;
    LogitCacheWrite cache_write;
    bool cache_write_begun = false;
    Status baseline_error = Status::make_ok();
    std::size_t baseline_sync_count = 0;

    auto cleanup_baseline = [&](Status primary) -> Status {
        Status first = primary;
        auto combine = [&](Status s) {
            if (first.ok() && !s.ok()) first = s;
            else if (!first.ok() && !s.ok()) {
                std::string msg = first.message() + "; " + s.message();
                first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
            }
        };
        if (cache_write_begun) {
            if (first.ok()) {
                Status we = cache_write.write_end();
                if (!we.ok()) combine(we);
                else cache_write_begun = false;
            } else {
                cache_write = LogitCacheWrite();
                cache_write_begun = false;
            }
        }
        if (baseline_executor_created) {
            baseline_executor_created = false;
        }
        if (baseline_arena.has_value()) {
            Status s = baseline_arena->shutdown();
            combine(s);
            if (s.ok()) baseline_arena.reset();
        }
        if (baseline_stream) {
            hipError_t e = hipStreamDestroy(baseline_stream);
            Status s = hip_status(e, "hipStreamDestroy(baseline)");
            combine(s);
            if (s.ok()) baseline_stream = nullptr;
        }
        if (!first.ok() && !options.keep_shadow && shadow_created && !shadow_dir.empty()) {
            std::error_code ec;
            fs::remove_all(shadow_dir, ec);
            if (ec && !is_not_found_error(ec)) {
                std::string msg = first.message() + "; shadow cleanup failed: " + ec.message();
                first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
            }
        }
        if (!options.keep_logit_cache && !first.ok() && !base_cache.empty()) {
            std::error_code ec;
            fs::remove(base_cache, ec);
            if (ec && !is_not_found_error(ec)) {
                std::string msg = first.message() + "; logit cache cleanup failed: " + ec.message();
                first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
            }
        }
        return first;
    };

    if (!cache_valid) {
        auto t0 = std::chrono::steady_clock::now();
        int dev = 0;
        hipError_t he = hipSetDevice(dev);
        if (he != hipSuccess) { Status s = hip_status(he, "hipSetDevice"); return cleanup_baseline(s); }
        he = hipStreamCreate(&baseline_stream);
        if (he != hipSuccess) { Status s = hip_status(he, "hipStreamCreate"); return cleanup_baseline(s); }

        std::size_t arena_bytes = static_cast<std::size_t>(options.arena_gib * 1024.0 * 1024.0 * 1024.0);
        auto arena_res = ps::gpu::GpuArena::create(dev, arena_bytes);
        if (!arena_res.ok()) { Status s = arena_res.status(); return cleanup_baseline(s); }
        baseline_arena.emplace(arena_res.release());
        t_baseline_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        auto t1 = std::chrono::steady_clock::now();
        ps::qwen35::Qwen35LoadOptions lopts;
        auto model_res = ps::qwen35::Qwen35Model::load_from_safetensors(options.input_dir, *baseline_arena, baseline_stream, lopts);
        if (!model_res.ok()) return cleanup_baseline(model_res.status());
        ps::qwen35::Qwen35Model baseline_model = model_res.release();
        vocab_size = baseline_model.text_config().vocab_size;
        cache_key.vocab_size = vocab_size;
        {
            auto m = logit_cache_matches(base_cache, cache_key);
            if (m.ok() && m.value()) {
                cache_valid = true;
                t_baseline_infer = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                baseline_error = cleanup_baseline(Status::make_ok());
                if (!baseline_error.ok()) return baseline_error;
                goto baseline_done;
            }
        }

        uint32_t page_tokens = 16;
        uint32_t max_seq = static_cast<uint32_t>(options.window);
        uint32_t max_blocks = (max_seq + page_tokens - 1) / page_tokens + 1;
        uint32_t num_pages = max_blocks + 2;
        uint32_t num_attn = count_attention_layers(baseline_model.text_config());
        uint32_t kv_heads = static_cast<uint32_t>(baseline_model.text_config().num_key_value_heads);
        uint32_t head_dim = static_cast<uint32_t>(baseline_model.text_config().attention_head_dim);

        auto slot_res = ps::qwen35::SequenceSlotPool::create(*baseline_arena, 1, max_blocks);
        if (!slot_res.ok()) return cleanup_baseline(slot_res.status());
        baseline_slot_pool.emplace(slot_res.release());
        auto gdn_res = ps::qwen35::GdnStatePool::create(
            *baseline_arena, 1,
            ps::qwen35::GdnStatePoolLayout::from_text_config(baseline_model.text_config()));
        if (!gdn_res.ok()) return cleanup_baseline(gdn_res.status());
        baseline_gdn_pool.emplace(gdn_res.release());
        auto kv_res = ps::qwen35::PagedKVPool::create(*baseline_arena, num_pages, page_tokens, num_attn, kv_heads, head_dim);
        if (!kv_res.ok()) return cleanup_baseline(kv_res.status());
        baseline_kv_pool.emplace(kv_res.release());

        ps::qwen35::ExecutorConfig ecfg;
        ecfg.max_scheduled_tokens = std::max<uint32_t>(static_cast<uint32_t>(options.window), 16);
        ecfg.max_scheduled_requests = 1;
        auto exec_res = ps::qwen35::create_model_executor(baseline_model, *baseline_slot_pool, *baseline_gdn_pool, *baseline_kv_pool, *baseline_arena, ecfg);
        if (!exec_res.ok()) return cleanup_baseline(exec_res.status());
        baseline_executor = exec_res.release();
        baseline_executor_created = true;

        cache_write.path = base_cache;
        cache_write.key = cache_key;
        Status wb = cache_write.write_begin();
        if (!wb.ok()) return cleanup_baseline(wb);
        cache_write_begun = true;

        {
            std::size_t next_pred = 0;
            for (uint32_t wi = 0; wi < plan.windows; ++wi) {
                uint32_t cnt = plan.window_counts[wi];
                auto seq_res = ps::qwen35::create_paged_sequence_state(*baseline_slot_pool, *baseline_gdn_pool, *baseline_kv_pool, cnt, baseline_stream);
                if (!seq_res.ok()) return cleanup_baseline(seq_res.status());
                ps::qwen35::PagedSequenceState seq = seq_res.release();
                Status loop_st = Status::make_ok();
                for (uint32_t pi = 0; pi + 1 < cnt; ++pi) {
                    const PredictionMeta& meta = plan.predictions[next_pred];
                    ps::qwen35::runtime::SchedulePlan sched;
                    Status bs = sched.append_decode(&seq, meta.input_token, seq.position, true, false,
                                               ps::qwen35::runtime::SamplingConfig{},
                                               0u);
                    if (!bs.ok()) { loop_st = bs; break; }
                    sched.finalize();
                    auto eres = ps::qwen35::execute_batch(baseline_executor, sched.batch, baseline_stream);
                    if (!eres.ok()) { loop_st = eres.status(); break; }
                    float* d_logits = baseline_executor.logits.data<float>();
                    std::vector<float> h_logits(vocab_size);
                    hipError_t herr = hipMemcpyAsync(h_logits.data(), d_logits, vocab_size * sizeof(float), hipMemcpyDeviceToHost, baseline_stream);
                    if (herr != hipSuccess) { loop_st = hip_status(herr, "hipMemcpyAsync baseline logits"); break; }
                    herr = hipStreamSynchronize(baseline_stream);
                    if (herr != hipSuccess) { loop_st = hip_status(herr, "hipStreamSynchronize baseline logits"); break; }
                    Status ws = cache_write.write_row(h_logits.data(), vocab_size);
                    if (!ws.ok()) { loop_st = ws; break; }
                    ++next_pred;
                    if (next_pred % options.batch_positions == 0) ++baseline_sync_count;
                }
                if (next_pred % options.batch_positions != 0 && loop_st.ok()) ++baseline_sync_count;
                Status rel = ps::qwen35::release_paged_sequence_state(seq, baseline_stream);
                if (!rel.ok() && loop_st.ok()) loop_st = rel;
                if (!loop_st.ok()) return cleanup_baseline(loop_st);
            }
            if (baseline_sync_count == 0) baseline_sync_count = (prediction_count + options.batch_positions - 1) / options.batch_positions;
        }

        t_baseline_infer = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        Status cs = cleanup_baseline(Status::make_ok());
        if (!cs.ok()) return cs;
    }
baseline_done:

    hipStream_t candidate_stream = nullptr;
    std::optional<ps::gpu::GpuArena> candidate_arena;
    std::optional<ps::qwen35::SequenceSlotPool> cand_slot_pool;
    std::optional<ps::qwen35::GdnStatePool> cand_gdn_pool;
    std::optional<ps::qwen35::PagedKVPool> cand_kv_pool;
    ps::qwen35::Executor cand_executor;
    bool cand_executor_created = false;
    LogitCacheReader cache_reader;
    bool cache_reader_open = false;
    GpuKldWorkspace kld_ws;
    bool kld_ws_created = false;
    float* d_baseline = nullptr;
    int32_t* d_targets = nullptr;
    GpuPositionMetrics* d_metrics = nullptr;
    Status candidate_error = Status::make_ok();
    std::vector<GpuPositionMetrics> position_metrics;
    std::size_t candidate_sync_count = 0;

    auto cleanup_candidate = [&](Status primary) -> Status {
        Status first = primary;
        auto combine = [&](Status s) {
            if (first.ok() && !s.ok()) first = s;
            else if (!first.ok() && !s.ok()) {
                std::string msg = first.message() + "; " + s.message();
                first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
            }
        };
        if (d_metrics) { hipError_t e = hipFree(d_metrics); Status s = hip_status(e, "hipFree(d_metrics)"); combine(s); if (s.ok()) d_metrics = nullptr; }
        if (d_targets) { hipError_t e = hipFree(d_targets); Status s = hip_status(e, "hipFree(d_targets)"); combine(s); if (s.ok()) d_targets = nullptr; }
        if (d_baseline) { hipError_t e = hipFree(d_baseline); Status s = hip_status(e, "hipFree(d_baseline)"); combine(s); if (s.ok()) d_baseline = nullptr; }
        if (kld_ws_created) { Status s = gpu_kld_workspace_destroy(kld_ws); combine(s); if (s.ok()) kld_ws_created = false; }
        if (cache_reader_open) { cache_reader.close(); cache_reader_open = false; }
        if (cand_executor_created) cand_executor_created = false;
        if (candidate_arena.has_value()) { Status s = candidate_arena->shutdown(); combine(s); if (s.ok()) candidate_arena.reset(); }
        if (candidate_stream) { hipError_t e = hipStreamDestroy(candidate_stream); Status s = hip_status(e, "hipStreamDestroy(candidate)"); combine(s); if (s.ok()) candidate_stream = nullptr; }
        if (!first.ok() && !options.keep_shadow && shadow_created && !shadow_dir.empty()) {
            std::error_code ec; fs::remove_all(shadow_dir, ec);
            if (ec && !is_not_found_error(ec)) {
                std::string msg = first.message() + "; shadow cleanup failed: " + ec.message();
                first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
            }
        }
        if (!options.keep_logit_cache && !first.ok() && !base_cache.empty()) {
            std::error_code ec;
            fs::remove(base_cache, ec);
            if (ec && !is_not_found_error(ec)) {
                std::string msg = first.message() + "; logit cache cleanup failed: " + ec.message();
                first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
            }
        }
        return first;
    };

    {
        auto t0 = std::chrono::steady_clock::now();
        int dev = 0;
        hipError_t he = hipSetDevice(dev);
        if (he != hipSuccess) return cleanup_candidate(hip_status(he, "hipSetDevice"));
        he = hipStreamCreate(&candidate_stream);
        if (he != hipSuccess) return cleanup_candidate(hip_status(he, "hipStreamCreate candidate"));
        std::size_t arena_bytes = static_cast<std::size_t>(options.arena_gib * 1024.0 * 1024.0 * 1024.0);
        auto arena_res = ps::gpu::GpuArena::create(dev, arena_bytes);
        if (!arena_res.ok()) return cleanup_candidate(arena_res.status());
        candidate_arena.emplace(arena_res.release());
        t_candidate_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        auto t1 = std::chrono::steady_clock::now();
        ps::qwen35::Qwen35LoadOptions lopts;
        auto cand_model_res = ps::qwen35::Qwen35Model::load_from_safetensors(candidate_dir, *candidate_arena, candidate_stream, lopts);
        if (!cand_model_res.ok()) return cleanup_candidate(cand_model_res.status());
        ps::qwen35::Qwen35Model cand_model = cand_model_res.release();
        vocab_size = cand_model.text_config().vocab_size;
        cache_key.vocab_size = vocab_size;

        uint32_t page_tokens = 16;
        uint32_t max_seq = static_cast<uint32_t>(options.window);
        uint32_t max_blocks = (max_seq + page_tokens - 1) / page_tokens + 1;
        uint32_t num_pages = max_blocks + 2;
        uint32_t num_attn = count_attention_layers(cand_model.text_config());
        uint32_t kv_heads = static_cast<uint32_t>(cand_model.text_config().num_key_value_heads);
        uint32_t head_dim = static_cast<uint32_t>(cand_model.text_config().attention_head_dim);

        auto slot_res = ps::qwen35::SequenceSlotPool::create(*candidate_arena, 1, max_blocks);
        if (!slot_res.ok()) return cleanup_candidate(slot_res.status());
        cand_slot_pool.emplace(slot_res.release());
        auto gdn_res = ps::qwen35::GdnStatePool::create(
            *candidate_arena, 1,
            ps::qwen35::GdnStatePoolLayout::from_text_config(cand_model.text_config()));
        if (!gdn_res.ok()) return cleanup_candidate(gdn_res.status());
        cand_gdn_pool.emplace(gdn_res.release());
        auto kv_res = ps::qwen35::PagedKVPool::create(*candidate_arena, num_pages, page_tokens, num_attn, kv_heads, head_dim);
        if (!kv_res.ok()) return cleanup_candidate(kv_res.status());
        cand_kv_pool.emplace(kv_res.release());

        ps::qwen35::ExecutorConfig ecfg;
        ecfg.max_scheduled_tokens = std::max<uint32_t>(static_cast<uint32_t>(options.window), 16);
        ecfg.max_scheduled_requests = 1;
        auto exec_res = ps::qwen35::create_model_executor(cand_model, *cand_slot_pool, *cand_gdn_pool, *cand_kv_pool, *candidate_arena, ecfg);
        if (!exec_res.ok()) return cleanup_candidate(exec_res.status());
        cand_executor = exec_res.release();
        cand_executor_created = true;

        cache_reader.path = base_cache;
        cache_reader.key = cache_key;
        Status rs = cache_reader.open();
        if (!rs.ok()) return cleanup_candidate(rs);
        cache_reader_open = true;

        Status ws = gpu_kld_workspace_create(kld_ws, vocab_size, options.batch_positions);
        if (!ws.ok()) return cleanup_candidate(ws);
        kld_ws_created = true;

        hipError_t herr = hipMalloc(&d_baseline, vocab_size * sizeof(float));
        if (herr != hipSuccess) return cleanup_candidate(hip_status(herr, "hipMalloc d_baseline"));
        herr = hipMalloc(&d_targets, sizeof(int32_t));
        if (herr != hipSuccess) return cleanup_candidate(hip_status(herr, "hipMalloc d_targets"));
        herr = hipMalloc(&d_metrics, sizeof(GpuPositionMetrics));
        if (herr != hipSuccess) return cleanup_candidate(hip_status(herr, "hipMalloc d_metrics"));

        position_metrics.reserve(prediction_count);
        std::size_t next_pred = 0;

        for (uint32_t wi = 0; wi < plan.windows; ++wi) {
            uint32_t cnt = plan.window_counts[wi];
            auto seq_res = ps::qwen35::create_paged_sequence_state(*cand_slot_pool, *cand_gdn_pool, *cand_kv_pool, cnt, candidate_stream);
            if (!seq_res.ok()) return cleanup_candidate(seq_res.status());
            ps::qwen35::PagedSequenceState seq = seq_res.release();
            Status loop_st = Status::make_ok();
            for (uint32_t pi = 0; pi + 1 < cnt; ++pi) {
                const PredictionMeta& meta = plan.predictions[next_pred];
                std::vector<float> h_baseline(vocab_size);
                Status rr = cache_reader.read_row(h_baseline.data(), vocab_size);
                if (!rr.ok()) { loop_st = rr; break; }
                herr = hipMemcpyAsync(d_baseline, h_baseline.data(), vocab_size * sizeof(float), hipMemcpyHostToDevice, candidate_stream);
                if (herr != hipSuccess) { loop_st = hip_status(herr, "hipMemcpy H2D baseline"); break; }
                herr = hipMemcpyAsync(d_targets, &meta.target_token, sizeof(int32_t), hipMemcpyHostToDevice, candidate_stream);
                if (herr != hipSuccess) { loop_st = hip_status(herr, "hipMemcpy H2D target"); break; }
                ps::qwen35::runtime::SchedulePlan sched;
                Status bs = sched.append_decode(&seq, meta.input_token, seq.position, true, false,
                                               ps::qwen35::runtime::SamplingConfig{},
                                               0u);
                if (!bs.ok()) { loop_st = bs; break; }
                sched.finalize();
                auto eres = ps::qwen35::execute_batch(cand_executor, sched.batch, candidate_stream);
                if (!eres.ok()) { loop_st = eres.status(); break; }
                float* d_candidate = cand_executor.logits.data<float>();
                Status ks = gpu_kld_metrics(kld_ws, d_baseline, d_candidate, d_targets, 1, d_metrics, candidate_stream);
                if (!ks.ok()) { loop_st = ks; break; }
                GpuPositionMetrics h_metric;
                herr = hipMemcpyAsync(&h_metric, d_metrics, sizeof(GpuPositionMetrics), hipMemcpyDeviceToHost, candidate_stream);
                if (herr != hipSuccess) { loop_st = hip_status(herr, "hipMemcpyAsync D2H metrics"); break; }
                herr = hipStreamSynchronize(candidate_stream);
                if (herr != hipSuccess) { loop_st = hip_status(herr, "hipStreamSynchronize D2H metrics"); break; }
                position_metrics.push_back(h_metric);
                ++next_pred;
                if (next_pred % options.batch_positions == 0) ++candidate_sync_count;
            }
            if (next_pred % options.batch_positions != 0 && loop_st.ok() && next_pred > 0) ++candidate_sync_count;
            Status rel = ps::qwen35::release_paged_sequence_state(seq, candidate_stream);
            if (!rel.ok() && loop_st.ok()) loop_st = rel;
            if (!loop_st.ok()) return cleanup_candidate(loop_st);
        }
        if (candidate_sync_count == 0) candidate_sync_count = (prediction_count + options.batch_positions - 1) / options.batch_positions;
        t_candidate_infer = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

        Status cs = Status::make_ok();
        if (kld_ws_created) { Status s = gpu_kld_workspace_destroy(kld_ws); if (!s.ok()) cs = s; else kld_ws_created = false; }
        if (d_metrics) { hipError_t e = hipFree(d_metrics); Status s = hip_status(e, "hipFree d_metrics"); if (!cs.ok() && !s.ok()) { std::string msg = cs.message() + "; " + s.message(); cs = Status::invalid_state(msg.c_str(), __FILE__, __LINE__); } else if (cs.ok() && !s.ok()) cs = s; if (s.ok()) d_metrics = nullptr; }
        if (d_targets) { hipError_t e = hipFree(d_targets); Status s = hip_status(e, "hipFree d_targets"); if (!cs.ok() && !s.ok()) { std::string msg = cs.message() + "; " + s.message(); cs = Status::invalid_state(msg.c_str(), __FILE__, __LINE__); } else if (cs.ok() && !s.ok()) cs = s; if (s.ok()) d_targets = nullptr; }
        if (d_baseline) { hipError_t e = hipFree(d_baseline); Status s = hip_status(e, "hipFree d_baseline"); if (!cs.ok() && !s.ok()) { std::string msg = cs.message() + "; " + s.message(); cs = Status::invalid_state(msg.c_str(), __FILE__, __LINE__); } else if (cs.ok() && !s.ok()) cs = s; if (s.ok()) d_baseline = nullptr; }
        if (cache_reader_open) { cache_reader.close(); cache_reader_open = false; }
        if (cand_executor_created) cand_executor_created = false;
        if (candidate_arena.has_value()) { Status s = candidate_arena->shutdown(); if (!cs.ok() && !s.ok()) { std::string msg = cs.message() + "; " + s.message(); cs = Status::invalid_state(msg.c_str(), __FILE__, __LINE__); } else if (cs.ok() && !s.ok()) cs = s; if (s.ok()) candidate_arena.reset(); }
        if (candidate_stream) { hipError_t e = hipStreamDestroy(candidate_stream); Status s = hip_status(e, "hipStreamDestroy candidate"); if (!cs.ok() && !s.ok()) { std::string msg = cs.message() + "; " + s.message(); cs = Status::invalid_state(msg.c_str(), __FILE__, __LINE__); } else if (cs.ok() && !s.ok()) cs = s; if (s.ok()) candidate_stream = nullptr; }
        if (!cs.ok()) {
            Status first = cs;
            if (!options.keep_shadow && shadow_created && !shadow_dir.empty()) {
                std::error_code ec; fs::remove_all(shadow_dir, ec);
                if (ec && !is_not_found_error(ec)) {
                    std::string msg = first.message() + "; shadow cleanup failed: " + ec.message();
                    first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
                }
            }
            if (!options.keep_logit_cache && !base_cache.empty()) {
                std::error_code ec; fs::remove(base_cache, ec);
                if (ec && !is_not_found_error(ec)) {
                    std::string msg = first.message() + "; logit cache cleanup failed: " + ec.message();
                    first = Status::invalid_state(msg.c_str(), __FILE__, __LINE__);
                }
            }
            return first;
        }
    }

    auto t_cpu_start = std::chrono::steady_clock::now();
    std::vector<double> per_kld;
    per_kld.reserve(position_metrics.size());
    std::vector<double> nll_b, nll_c;
    nll_b.reserve(position_metrics.size());
    nll_c.reserve(position_metrics.size());
    double sum_nll_b = 0, sum_nll_c = 0;
    double sum_abs = 0, sum_sq = 0, max_delta = 0;
    double sum_rmse = 0;
    std::size_t agree = 0;
    std::size_t nonfinite = 0;
    for (auto& m : position_metrics) {
        per_kld.push_back(m.kld);
        nll_b.push_back(m.nll_reference);
        nll_c.push_back(m.nll_candidate);
        sum_nll_b += m.nll_reference;
        sum_nll_c += m.nll_candidate;
        if (m.reference_top1 == m.candidate_top1) ++agree;
        double delta = std::fabs(m.reference_top1_probability - m.candidate_probability_at_reference_top1);
        sum_abs += delta;
        sum_sq += delta * delta;
        max_delta = std::max(max_delta, delta);
        sum_rmse += std::sqrt(m.logit_mse);
        if (m.flags & kKldFlagNonFinite) ++nonfinite;
    }
    KldResult kld_res = aggregate_kld(per_kld);
    double ppl_b = per_kld.empty() ? 0.0 : std::exp(sum_nll_b / static_cast<double>(per_kld.size()));
    double ppl_c = per_kld.empty() ? 0.0 : std::exp(sum_nll_c / static_cast<double>(per_kld.size()));
    double ppl_delta = ppl_c - ppl_b;
    double ppl_rel = ppl_b > 0 ? (ppl_delta / ppl_b) * 100.0 : 0.0;
    double agree_pct = per_kld.empty() ? 0.0 : 100.0 * static_cast<double>(agree) / static_cast<double>(per_kld.size());
    double mean_abs = per_kld.empty() ? 0.0 : sum_abs / static_cast<double>(per_kld.size());
    double rms = per_kld.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(per_kld.size()));
    double mean_rmse = per_kld.empty() ? 0.0 : sum_rmse / static_cast<double>(per_kld.size());
    if (baseline_sync_count == 0) baseline_sync_count = (prediction_count + options.batch_positions - 1) / options.batch_positions;
    if (candidate_sync_count == 0) candidate_sync_count = (prediction_count + options.batch_positions - 1) / options.batch_positions;

    nlohmann::json by_bucket = nlohmann::json::array();
    {
        std::array<std::pair<std::size_t,std::size_t>,5> ranges = {{{0,31},{32,63},{64,127},{128,255},{256,511}}};
        for (auto& rng : ranges) {
            std::vector<double> vals;
            for (std::size_t i = 0; i < per_kld.size() && i < plan.predictions.size(); ++i) {
                uint32_t pos = plan.predictions[i].position_in_window;
                if (pos >= rng.first && pos <= rng.second) vals.push_back(per_kld[i]);
            }
            nlohmann::json b;
            b["range"] = {rng.first, rng.second};
            if (!vals.empty()) {
                b["count"] = vals.size();
                b["mean_kld"] = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
                std::vector<double> s = vals; std::sort(s.begin(), s.end());
                b["p99_kld"] = s[static_cast<std::size_t>(0.99 * static_cast<double>(s.size()))];
            } else {
                b["count"] = 0;
                b["mean_kld"] = 0.0;
                b["p99_kld"] = 0.0;
            }
            by_bucket.push_back(std::move(b));
        }
    }

    nlohmann::json by_window = nlohmann::json::array();
    {
        for (uint32_t wi = 0; wi < plan.windows; ++wi) {
            std::vector<double> vals;
            std::size_t ag = 0;
            double s_b = 0, s_c = 0;
            for (std::size_t i = 0; i < per_kld.size(); ++i) {
                if (plan.predictions[i].window_index != wi) continue;
                vals.push_back(per_kld[i]);
                if (position_metrics[i].reference_top1 == position_metrics[i].candidate_top1) ++ag;
                s_b += nll_b[i];
                s_c += nll_c[i];
            }
            nlohmann::json w;
            w["window_index"] = wi;
            w["start_token_offset"] = plan.window_starts[wi];
            w["positions"] = vals.size();
            if (!vals.empty()) {
                w["mean_kld"] = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
                std::vector<double> s = vals; std::sort(s.begin(), s.end());
                w["p99_kld"] = s[static_cast<std::size_t>(0.99 * static_cast<double>(s.size()))];
                w["max_kld"] = s.back();
                w["top1_agreement"] = 100.0 * static_cast<double>(ag) / static_cast<double>(vals.size());
                w["mean_nll_reference"] = s_b / static_cast<double>(vals.size());
                w["mean_nll_candidate"] = s_c / static_cast<double>(vals.size());
            } else {
                w["mean_kld"] = 0.0; w["p99_kld"] = 0.0; w["max_kld"] = 0.0;
                w["top1_agreement"] = 0.0; w["mean_nll_reference"] = 0.0; w["mean_nll_candidate"] = 0.0;
            }
            by_window.push_back(std::move(w));
        }
    }

    nlohmann::json worst = nlohmann::json::array();
    {
        std::vector<std::size_t> idx(per_kld.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b){ return per_kld[a] > per_kld[b]; });
        std::size_t nw = std::min<std::size_t>(50, per_kld.size());
        for (std::size_t k = 0; k < nw; ++k) {
            std::size_t i = idx[k];
            const auto& pm = plan.predictions[i];
            const auto& m = position_metrics[i];
            nlohmann::json e;
            e["window"] = pm.window_index;
            e["position"] = pm.position_in_window;
            e["global_token_offset"] = pm.global_token_offset;
            e["input_token"] = pm.input_token;
            e["target_token"] = pm.target_token;
            e["reference_top1"] = m.reference_top1;
            e["candidate_top1"] = m.candidate_top1;
            e["reference_top1_probability"] = m.reference_top1_probability;
            e["candidate_probability_at_reference_top1"] = m.candidate_probability_at_reference_top1;
            e["kld"] = m.kld;
            worst.push_back(std::move(e));
        }
    }

    t_cpu_agg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cpu_start).count();
    double t_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total_start).count();

    nlohmann::json report;
    report["format"] = "phaseshift-fpx-kld";
    report["format_version"] = 2;
    std::string preset = options.self_test ? "self" : "balanced";
    report["model"] = {{"architecture","qwen35_dense"},{"preset",preset}};
    report["comparison"] = {{"reference","original-bf16-runtime"},{"candidate", options.self_test ? "self" : "fpx-qdq-bf16-shadow"}};
    report["dataset"] = {{"token_corpus_sha256",corpus_sha},{"token_count",corpus.tokens.size()},{"window",options.window},{"stride",options.stride},{"evaluated_windows",plan.windows},{"evaluated_positions",per_kld.size()}};
    report["runtime"] = {{"backend","paged"},{"wave32_attention",false},{"batch_positions",options.batch_positions},{"model_fingerprint",model_fp}};
    report["kld"] = {{"direction","D_KL(P_BF16||P_QDQ)"},{"mean",kld_res.mean},{"p50",kld_res.p50},{"p90",kld_res.p90},{"p99",kld_res.p99},{"p99_9",kld_res.p99_9},{"max",kld_res.max}};
    report["perplexity"] = {{"bf16",ppl_b},{"qdq",ppl_c},{"absolute_delta",ppl_delta},{"relative_delta_percent",ppl_rel}};
    report["top_token"] = {{"agreement",agree_pct},{"mean_abs_probability_delta",mean_abs},{"rms_probability_delta",rms},{"max_probability_delta",max_delta}};
    report["logit"] = {{"mean_rmse",mean_rmse}};
    report["numerical"] = {{"nonfinite_positions",nonfinite}};
    report["by_position_bucket"] = by_bucket;
    report["by_window"] = by_window;
    report["worst_tokens"] = worst;
    nlohmann::json transfer;
    transfer["baseline_cache_d2h_bytes"] = prediction_count * vocab_size * sizeof(float);
    transfer["baseline_cache_h2d_bytes"] = prediction_count * vocab_size * sizeof(float);
    transfer["candidate_full_logit_d2h_bytes"] = 0;
    transfer["candidate_metric_d2h_bytes"] = position_metrics.size() * sizeof(GpuPositionMetrics);
    transfer["candidate_disk_cache_bytes"] = 0;
    report["transfer"] = transfer;
    report["synchronization"] = {{"positions",per_kld.size()},{"batch_size",options.batch_positions},{"baseline_sync_count",baseline_sync_count},{"candidate_sync_count",candidate_sync_count}};
    report["timing"] = {{"shadow_materialize_seconds",t_shadow},{"baseline_load_seconds",t_baseline_load},{"baseline_inference_seconds",t_baseline_infer},{"candidate_load_seconds",t_candidate_load},{"candidate_inference_seconds",t_candidate_infer},{"cpu_aggregate_seconds",t_cpu_agg},{"total_seconds",t_total},{"baseline_positions_per_sec", t_baseline_infer > 0 ? per_kld.size() / t_baseline_infer : 0.0},{"candidate_positions_per_sec", t_candidate_infer > 0 ? per_kld.size() / t_candidate_infer : 0.0}};

    {
        fs::path rp(options.report_path);
        std::error_code ec;
        fs::create_directories(rp.parent_path(), ec);
        std::ofstream rf(options.report_path, std::ios::trunc);
        if (!rf) return Status::invalid_argument("cannot open report for write", __FILE__, __LINE__);
        rf << report.dump(2);
        if (!rf) return Status::invalid_state("cannot write report", __FILE__, __LINE__);
    }

    if (!options.keep_shadow && shadow_created && !shadow_dir.empty()) {
        std::error_code ec; fs::remove_all(shadow_dir, ec);
        if (ec && !is_not_found_error(ec)) return Status::invalid_state(("shadow cleanup failed: " + ec.message()).c_str(), __FILE__, __LINE__);
    }
    if (!options.keep_logit_cache && !base_cache.empty()) {
        std::error_code ec; fs::remove(base_cache, ec);
        if (ec && !is_not_found_error(ec)) return Status::invalid_state(("logit cache cleanup failed: " + ec.message()).c_str(), __FILE__, __LINE__);
    }

    KldEvaluationSummary summary;
    summary.kld = kld_res;
    summary.evaluated_positions = per_kld.size();
    return summary;
}

}
