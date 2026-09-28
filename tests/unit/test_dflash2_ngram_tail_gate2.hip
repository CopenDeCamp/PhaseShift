#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/ngram_tail.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using ps::bf16_t;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2ContextState;
using ps::qwen35::dflash2::DFlash2Executor;
using ps::qwen35::dflash2::DFlash2ExecutorConfig;
using ps::qwen35::dflash2::DFlash2Weights;
using ps::qwen35::runtime::DFlash2SpecDecoder;
using ps::qwen35::runtime::DFlash2SpecDecoderConfig;
using ps::qwen35::runtime::DFlash2SpecIterationOutput;
using ps::qwen35::runtime::DFlash2SpecTiming;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static double wall_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

static std::vector<std::string> split_list(const char* value, const char* fallback) {
    std::string s = (value != nullptr && value[0] != '\0') ? value : fallback;
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (pos < s.size()) {
        std::size_t comma = s.find(',', pos);
        std::string part =
            s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!part.empty()) out.push_back(part);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

static std::vector<int32_t> read_corpus(const fs::path& path) {
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

static ps::qwen35::ScheduledBatch make_batch(
    ps::qwen35::PagedSequenceState& sequence,
    std::vector<ps::qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t output_rows,
    ps::runtime::VerifyNumericMode mode) {
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
    batch.verify_numeric_mode = mode;
    return batch;
}

static bool sample_tokens(
    ps::qwen35::Executor& executor,
    ps::qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& tokens,
    uint32_t output_rows,
    std::vector<int32_t>& sampled,
    hipStream_t stream) {
    std::vector<ps::qwen35::ScheduledRequest> requests;
    ps::qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, tokens.data(), static_cast<uint32_t>(tokens.size()), output_rows,
        ps::runtime::VerifyNumericMode::Exact);
    auto executed = ps::qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: execute_batch: %s\n", executed.status().message().c_str());
        return false;
    }
    ps::qwen35::BatchExecutionOutput output = executed.release();
    sampled.assign(output_rows, -1);
    const hipError_t err = hipMemcpy(sampled.data(), output.sampled_tokens.data<int32_t>(),
                                     output_rows * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::printf("FAIL: sample copy: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

struct TargetOnlyRun {
    std::vector<int32_t> tokens;
    double decode_ms = 0.0;
    bool ok = false;
};

static TargetOnlyRun run_target_only(
    ps::qwen35::Executor& executor,
    ps::qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    hipStream_t stream) {
    TargetOnlyRun out;
    std::vector<int32_t> sampled;
    if (!sample_tokens(executor, sequence, prompt, 1u, sampled, stream)) return out;
    out.tokens.push_back(sampled[0]);
    const auto t0 = std::chrono::steady_clock::now();
    while (out.tokens.size() < max_new) {
        std::vector<int32_t> step{out.tokens.back()};
        if (!sample_tokens(executor, sequence, step, 1u, sampled, stream)) return out;
        out.tokens.push_back(sampled[0]);
    }
    out.decode_ms = wall_ms(t0);
    out.ok = true;
    return out;
}

struct SpecRun {
    std::vector<int32_t> tokens;
    double decode_ms = 0.0;
    uint32_t violations = 0u;
    uint32_t max_total_drafts = 0u;
    bool ok = false;
};

static SpecRun run_speculative(
    DFlash2SpecDecoder& decoder,
    ps::qwen35::PagedSequenceState& sequence,
    DFlash2ContextState& state,
    const std::vector<int32_t>& prompt,
    uint32_t max_new,
    uint32_t case_total_drafts,
    DFlash2SpecTiming& timing,
    hipStream_t stream) {
    SpecRun out;
    auto prefill = ps::qwen35::runtime::dflash2_spec_prefill(
        decoder, prompt.data(), static_cast<uint32_t>(prompt.size()));
    if (!prefill.ok()) {
        std::printf("FAIL: spec prefill: %s\n", prefill.status().message().c_str());
        return out;
    }
    int32_t pending = prefill.value().pending_token;
    out.tokens.push_back(pending);
    const auto t0 = std::chrono::steady_clock::now();
    while (out.tokens.size() < max_new && pending >= 0) {
        const uint32_t remaining = max_new - static_cast<uint32_t>(out.tokens.size());
        auto step = ps::qwen35::runtime::dflash2_spec_step(decoder, pending, remaining);
        if (!step.ok()) {
            std::printf("FAIL: spec step: %s\n", step.status().message().c_str());
            return out;
        }
        DFlash2SpecIterationOutput value = step.release();
        const std::size_t history_at_lookup = decoder.token_history.size() - value.emitted_count;

        if (value.num_drafts != value.num_dflash_drafts + value.num_tail_drafts) {
            std::printf("FAIL: total_k != dflash_k + tail_k\n");
            ++out.violations;
        }
        if (value.num_drafts > case_total_drafts) {
            std::printf("FAIL: total_k %u exceeds case capacity %u\n", value.num_drafts,
                        case_total_drafts);
            ++out.violations;
        }
        if (value.num_accepted > value.num_drafts) {
            std::printf("FAIL: accepted %u > total_k %u\n", value.num_accepted,
                        value.num_drafts);
            ++out.violations;
        }
        const uint32_t tail_accepted = value.num_accepted > value.num_dflash_drafts
                                           ? value.num_accepted - value.num_dflash_drafts
                                           : 0u;
        if (tail_accepted > value.num_tail_drafts) {
            std::printf("FAIL: tail accepted %u > tail_k %u\n", tail_accepted,
                        value.num_tail_drafts);
            ++out.violations;
        }
        if (tail_accepted > 0u && value.num_accepted < value.num_dflash_drafts + 1u) {
            std::printf("FAIL: tail accepted without DFlash prefix full accept\n");
            ++out.violations;
        }
        if (value.ngram.hit) {
            if (value.ngram.candidate_end > history_at_lookup) {
                std::printf("FAIL: candidate_end %u beyond committed history %zu\n",
                            value.ngram.candidate_end, history_at_lookup);
                ++out.violations;
            }
            if (value.ngram.candidate_start >= value.ngram.current_position) {
                std::printf("FAIL: candidate_start not before current_position\n");
                ++out.violations;
            }
            if (value.ngram.candidate_end + value.ngram.count > value.ngram.current_position) {
                std::printf("FAIL: tail continuation beyond committed history\n");
                ++out.violations;
            }
        }
        if (value.num_drafts > out.max_total_drafts) out.max_total_drafts = value.num_drafts;

        for (uint32_t i = 0; i < value.emitted_count; ++i) {
            out.tokens.push_back(value.emitted[i]);
        }
        pending = value.pending_token;
        if (value.finished) break;
    }
    out.decode_ms = wall_ms(t0);
    if (out.tokens.size() > max_new) out.tokens.resize(max_new);
    out.ok = state.next_position == sequence.position;
    if (!out.ok) {
        std::printf("FAIL: context.next_position %u != sequence.position %u\n",
                    state.next_position, sequence.position);
    }
    return out;
}

struct CaseConfig {
    std::string name;
    uint32_t k = 7u;
    uint32_t tail = 0u;
    bool host_d2h = false;
};

struct RunSlot {
    CaseConfig config;
    std::optional<ps::qwen35::PagedSequenceState> sequence;
    DFlash2SpecDecoder decoder;
    bool decoder_valid = false;
};

struct PerfRow {
    std::string category;
    std::string config;
    uint32_t context = 0u;
    uint32_t prompt_id = 0u;
    uint32_t run = 0u;
    uint32_t dflash_k = 0u;
    uint32_t ngram_n = 0u;
    uint32_t ngram_tail = 0u;
    uint32_t window = 0u;
    std::string bridge_mode;
    std::string gdn_mode;
    uint32_t generated_tokens = 0u;
    uint32_t rounds = 0u;
    uint32_t dflash_accepted = 0u;
    uint32_t ngram_hits = 0u;
    uint32_t tail_reached = 0u;
    uint32_t tail_blocked = 0u;
    uint32_t ngram_proposed = 0u;
    uint32_t ngram_accepted = 0u;
    uint64_t verify_rows = 0u;
    double draft_gpu_ms = 0.0;
    double proposal_wait_ms = 0.0;
    double ngram_seed_wait_ms = 0.0;
    double ngram_lookup_ms = 0.0;
    double ngram_tail_h2d_ms = 0.0;
    double verify_gpu_ms = 0.0;
    double rerun_ms = 0.0;
    double restore_ms = 0.0;
    double commit_ms = 0.0;
    double round_ms = 0.0;
    double decode_ms = 0.0;
    double tok_s = 0.0;
    bool token_exact = false;
};

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

    const char* mode_env = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_MODE");
    const std::string mode =
        (mode_env != nullptr && mode_env[0] != '\0') ? mode_env : "correctness";
    const bool perf_mode = mode == "perf";

    const char* corpus_env = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_CORPUS");
    std::string corpus_dir = (corpus_env != nullptr && corpus_env[0] != '\0') ? corpus_env : "";
    if (corpus_dir.empty()) {
        corpus_dir = fs::exists("artifacts/ngram_tail_gate1/corpus")
                         ? "artifacts/ngram_tail_gate1/corpus"
                         : "tests/data/mtp_perf";
    }
    const char* dir_env = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_DIR");
    const std::string root =
        (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "artifacts/ngram_tail_gate2";

    const uint32_t gen_tokens = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_GEN") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_GEN")
            : (perf_mode ? "512" : "256"),
        nullptr, 10));
    const uint32_t runs = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_RUNS") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_RUNS")
            : (perf_mode ? "5" : "1"),
        nullptr, 10));
    const uint32_t prompts = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_PROMPTS") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_PROMPTS")
            : "1",
        nullptr, 10));
    const uint32_t ngram_n = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_N") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_N")
            : "5",
        nullptr, 10));
    const uint32_t ngram_window = static_cast<uint32_t>(std::strtoul(
        std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_WINDOW") != nullptr
            ? std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_WINDOW")
            : "2048",
        nullptr, 10));

    std::vector<CaseConfig> all_configs;
    if (perf_mode) {
        all_configs.push_back({"A_k7_t0", 7u, 0u, false});
        all_configs.push_back({"B_k7_t0_hostd2h", 7u, 0u, true});
        all_configs.push_back({"C_k7_t8", 7u, 8u, false});
        if (std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_DIAG") != nullptr) {
            all_configs.push_back({"C_k7_t9", 7u, 9u, false});
        }
        all_configs.push_back({"C_k7_t16", 7u, 16u, false});
        all_configs.push_back({"C_k7_t32", 7u, 32u, false});
        setenv("PHASESHIFT_DFLASH2_GDN_RERUN_REFERENCE", "1", 1);
    } else {
        all_configs.push_back({"k1_t6", 1u, 6u, false});
        all_configs.push_back({"k3_t4", 3u, 4u, false});
        all_configs.push_back({"k7_t0", 7u, 0u, false});
    }
    std::vector<CaseConfig> configs;
    const std::vector<std::string> case_filter =
        split_list(std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_CASES"), "");
    for (const CaseConfig& cc : all_configs) {
        if (case_filter.empty()) {
            configs.push_back(cc);
            continue;
        }
        for (const std::string& want : case_filter) {
            if (want == cc.name) configs.push_back(cc);
        }
    }
    if (configs.empty()) {
        std::printf("FAIL: no case configuration selected\n");
        return 1;
    }

    std::vector<std::pair<std::string, uint32_t>> workload_spec;
    const std::vector<std::string> workload_env =
        split_list(std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_WORKLOADS"), "");
    if (!workload_env.empty()) {
        for (const std::string& w : workload_env) {
            const std::size_t sep = w.find(':');
            if (sep == std::string::npos) continue;
            workload_spec.push_back({w.substr(0, sep),
                                     static_cast<uint32_t>(std::strtoul(
                                         w.substr(sep + 1).c_str(), nullptr, 10))});
        }
    } else if (perf_mode) {
        workload_spec = {{"code", 512u},  {"code", 2048u},   {"json", 512u},
                         {"json", 2048u}, {"prose", 2048u},  {"reasoning", 2048u}};
    } else {
        for (const std::string& cat : {"code", "json", "prose", "reasoning"}) {
            for (const char* ctx : {"512", "2048"}) {
                workload_spec.push_back(
                    {cat, static_cast<uint32_t>(std::strtoul(ctx, nullptr, 10))});
            }
        }
        const char* only = std::getenv("PHASESHIFT_NGRAM_TAIL_GATE2_ONLY");
        if (only != nullptr && only[0] != '\0') {
            std::vector<std::pair<std::string, uint32_t>> filtered;
            for (const auto& w : workload_spec) {
                if (w.first == only) filtered.push_back(w);
            }
            workload_spec = filtered;
        }
    }

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "dflash config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();
    const uint32_t tap_count = static_cast<uint32_t>(config.num_target_layer_ids);
    const uint32_t block_rows = static_cast<uint32_t>(config.block_size);
    const uint32_t max_drafts = block_rows - 1u;

    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
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
    check(model_result.ok(), "target model load");
    if (!model_result.ok()) return 1;
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();
    std::fprintf(stderr, "[g2] after model arena used=%.1f GB\n",
                 static_cast<double>(arena.used()) / 1e9);

    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "dflash weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    uint32_t max_context = 32u;
    for (const auto& w : workload_spec) max_context = std::max(max_context, w.second);
    const uint32_t required_seq_len = max_context + gen_tokens + max_drafts + 64u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t seq_slots = static_cast<uint32_t>(configs.size()) + 1u;
    const uint32_t blocks_per_seq = ceil_div(required_seq_len, page_tokens) + 8u;
    const uint32_t num_pages =
        seq_slots * ceil_div(required_seq_len, page_tokens) + 16u;

    auto seq_pool_result =
        ps::qwen35::SequenceSlotPool::create(arena, seq_slots, blocks_per_seq);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, seq_slots, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    for (uint32_t t = 0; t < tap_count; ++t) {
        taps[t] = static_cast<uint32_t>(config.target_layer_ids[t]);
    }

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = 2048u;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = ps::qwen35::runtime::kDFlash2SpecMaxVerifyRows;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "target executor create");
    if (!exec_result.ok()) return 1;
    ps::qwen35::Executor ex = exec_result.release();
    std::fprintf(stderr, "[g2] after executor arena used=%.1f GB\n",
                 static_cast<double>(arena.used()) / 1e9);

    DFlash2ExecutorConfig draft_cfg;
    draft_cfg.max_rows = block_rows;
    draft_cfg.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    draft_cfg.target_lm_head = &model.weights().lm_head;
    draft_cfg.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result =
        ps::qwen35::dflash2::create_dflash2_executor(config, weights, arena, draft_cfg);
    check(draft_result.ok(), "draft executor create");
    if (!draft_result.ok()) return 1;
    DFlash2Executor draft = draft_result.release();
    std::fprintf(stderr, "[g2] after draft arena used=%.1f GB\n",
                 static_cast<double>(arena.used()) / 1e9);

    auto state_result = ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
    check(state_result.ok(), "dflash context create");
    if (!state_result.ok()) return 1;
    DFlash2ContextState state = state_result.release();

    struct Workload {
        std::string category;
        uint32_t context = 0u;
        uint32_t prompt_id = 0u;
        std::vector<int32_t> prompt;
        std::vector<int32_t> baseline;
        bool have_baseline = false;
    };
    std::vector<Workload> workloads;
    for (const auto& spec : workload_spec) {
        const std::vector<int32_t> corpus =
            read_corpus(fs::path(corpus_dir) / (spec.first + ".psktok"));
        if (corpus.empty()) {
            std::printf("FAIL: unreadable corpus %s in %s\n", spec.first.c_str(),
                        corpus_dir.c_str());
            return 1;
        }
        if (corpus.size() < spec.second) {
            std::printf("FAIL: corpus %s has %zu tokens < context %u\n", spec.first.c_str(),
                        corpus.size(), spec.second);
            return 1;
        }
        for (uint32_t p = 0u; p < prompts; ++p) {
            const uint32_t off = p * spec.second;
            if (off + spec.second > corpus.size()) break;
            Workload w;
            w.category = spec.first;
            w.context = spec.second;
            w.prompt_id = p;
            w.prompt.assign(corpus.begin() + off, corpus.begin() + off + spec.second);
            workloads.push_back(std::move(w));
        }
    }
    if (workloads.empty()) {
        std::printf("FAIL: no workload\n");
        return 1;
    }

    std::vector<RunSlot> slots(configs.size());
    for (std::size_t i = 0u; i < configs.size(); ++i) slots[i].config = configs[i];

    std::vector<PerfRow> perf_rows;
    std::vector<std::string> correctness_lines;
    uint32_t ngram_hit_rounds_total = 0u;
    uint32_t ngram_accepted_total = 0u;
    uint32_t parity_ok = 0u;
    uint32_t parity_total = 0u;

    const bool do_smoke = perf_mode;
    const int phases = do_smoke ? 2 : 1;

    for (int phase = 0; phase < phases; ++phase) {
        const uint32_t phase_gen = do_smoke ? (phase == 0 ? 256u : gen_tokens) : gen_tokens;
        const uint32_t phase_runs = do_smoke ? (phase == 0 ? 1u : runs) : runs;
        const bool record = !do_smoke || phase == 1;
        if (do_smoke) std::printf("PHASE gen=%u runs=%u\n", phase_gen, phase_runs);

        for (Workload& w : workloads) {
            w.have_baseline = false;
            const uint32_t seq_len = w.context + phase_gen + max_drafts + 64u;

            auto base_seq_result = ps::qwen35::create_paged_sequence_state(
                seq_pool, gdn_pool, kv_pool, seq_len, stream);
            if (!base_seq_result.ok()) {
                std::printf("FAIL: baseline sequence create\n");
                return 1;
            }
            ps::qwen35::PagedSequenceState base_seq = base_seq_result.release();
            TargetOnlyRun base = run_target_only(ex, base_seq, w.prompt, phase_gen, stream);
            (void)ps::qwen35::release_paged_sequence_state(base_seq, stream);
            if (!base.ok) {
                std::printf("FAIL: target-only baseline\n");
                return 1;
            }
            w.baseline = std::move(base.tokens);
            w.have_baseline = true;

            for (uint32_t sample = 0u; sample < phase_runs; ++sample) {
                for (RunSlot& slot : slots) {
                    if (slot.config.host_d2h && do_smoke && phase == 0) continue;
                    if (slot.config.host_d2h) {
                        setenv("PHASESHIFT_DFLASH2_HOST_PROPOSAL_D2H", "1", 1);
                    } else {
                        unsetenv("PHASESHIFT_DFLASH2_HOST_PROPOSAL_D2H");
                    }

                    if (slot.sequence.has_value()) {
                        (void)ps::qwen35::release_paged_sequence_state(*slot.sequence, stream);
                        slot.sequence.reset();
                    }
                    auto seq_result = ps::qwen35::create_paged_sequence_state(
                        seq_pool, gdn_pool, kv_pool, seq_len, stream);
                    if (!seq_result.ok()) {
                        std::printf("FAIL: spec sequence create\n");
                        return 1;
                    }
                    slot.sequence.emplace(seq_result.release());

                    if (!slot.decoder_valid) {
                        DFlash2SpecDecoderConfig decoder_cfg;
                        decoder_cfg.num_drafts = slot.config.k;
                        decoder_cfg.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
                        decoder_cfg.ngram_n = ngram_n;
                        decoder_cfg.ngram_max_tail = slot.config.tail;
                        decoder_cfg.ngram_window = ngram_window;
                        auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
                            ex, draft, state, *slot.sequence, gdn_pool, arena, decoder_cfg,
                            stream);
                        check(decoder_result.ok(),
                              slot.config.name + " decoder create: " +
                                  (decoder_result.ok()
                                       ? ""
                                       : decoder_result.status().message()));
                        if (!decoder_result.ok()) return 1;
                        slot.decoder = decoder_result.release();
                        slot.decoder_valid = true;
                        std::fprintf(stderr, "[g2] after decoder %s arena used=%.1f GB\n",
                                     slot.config.name.c_str(),
                                     static_cast<double>(arena.used()) / 1e9);
                    }

                    (void)ps::qwen35::dflash2::dflash2_context_reset(state, 0u);
                    DFlash2SpecTiming timing;
                    slot.decoder.timing = &timing;

                    SpecRun spec = run_speculative(
                        slot.decoder, *slot.sequence, state, w.prompt, phase_gen,
                        slot.config.k + slot.config.tail, timing, stream);
                    const bool exact = spec.ok && spec.tokens == w.baseline;
                    ++parity_total;
                    if (exact) ++parity_ok;
                    ngram_hit_rounds_total += timing.ngram_hit_rounds;
                    ngram_accepted_total += timing.ngram_accepted_tokens;

                    if (record) {
                        std::printf(
                            "%-11s %-15s ctx=%u run=%u exact=%s rounds=%u gen=%zu "
                            "acc/round=%.3f hits=%u reached=%u blocked=%u tail_acc=%u "
                            "rows=%.1f tok/s=%.2f\n",
                            w.category.c_str(), slot.config.name.c_str(), w.context, sample,
                            exact ? "yes" : "NO", timing.rounds, spec.tokens.size(),
                            timing.rounds > 0u
                                ? static_cast<double>(timing.accepted_drafts) /
                                      static_cast<double>(timing.rounds)
                                : 0.0,
                            timing.ngram_hit_rounds, timing.tail_reached_rounds,
                            timing.tail_blocked_rounds, timing.ngram_accepted_tokens,
                            timing.rounds > 0u
                                ? static_cast<double>(timing.verify_rows_total) /
                                      static_cast<double>(timing.rounds)
                                : 0.0,
                            spec.decode_ms > 0.0
                                ? static_cast<double>(spec.tokens.size() - 1u) /
                                      (spec.decode_ms / 1000.0)
                                : 0.0);
                    }

                    if (record) {
                        PerfRow row;
                        row.category = w.category;
                        row.config = slot.config.name;
                        row.context = w.context;
                        row.prompt_id = w.prompt_id;
                        row.run = sample;
                        row.dflash_k = slot.config.k;
                        row.ngram_n = ngram_n;
                        row.ngram_tail = slot.config.tail;
                        row.window = ngram_window;
                        row.bridge_mode = slot.decoder.device_token_bridge ? "device" : "host";
                        row.gdn_mode = slot.decoder.gdn_history_enabled ? "history" : "rerun";
                        row.generated_tokens = timing.generated_tokens;
                        row.rounds = timing.rounds;
                        row.dflash_accepted = timing.accepted_drafts;
                        row.ngram_hits = timing.ngram_hit_rounds;
                        row.tail_reached = timing.tail_reached_rounds;
                        row.tail_blocked = timing.tail_blocked_rounds;
                        row.ngram_proposed = timing.ngram_proposed_tokens;
                        row.ngram_accepted = timing.ngram_accepted_tokens;
                        row.verify_rows = timing.verify_rows_total;
                        row.draft_gpu_ms = timing.draft_gpu_ms;
                        row.proposal_wait_ms = timing.proposal_wait_ms;
                        row.ngram_seed_wait_ms = timing.ngram_seed_wait_ms;
                        row.ngram_lookup_ms = timing.ngram_lookup_ms;
                        row.ngram_tail_h2d_ms = timing.ngram_tail_h2d_ms;
                        row.verify_gpu_ms = timing.verify_gpu_ms;
                        row.rerun_ms = timing.rerun_ms;
                        row.restore_ms = timing.gdn_restore_ms;
                        row.commit_ms = timing.dflash_commit_ms;
                        row.round_ms = timing.round_ms;
                        row.decode_ms = spec.decode_ms;
                        row.tok_s = spec.decode_ms > 0.0
                                        ? static_cast<double>(spec.tokens.size() - 1u) /
                                              (spec.decode_ms / 1000.0)
                                        : 0.0;
                        row.token_exact = exact;
                        perf_rows.push_back(row);
                    }

                    if (!exact) {
                        std::size_t idx = 0;
                        while (idx < w.baseline.size() && idx < spec.tokens.size() &&
                               w.baseline[idx] == spec.tokens[idx]) {
                            ++idx;
                        }
                        std::printf(
                            "GATE2_DIVERGE %s %s ctx=%u gen=%u idx=%zu baseline=%d spec=%d\n",
                            w.category.c_str(), slot.config.name.c_str(), w.context, phase_gen,
                            idx, idx < w.baseline.size() ? w.baseline[idx] : -1,
                            idx < spec.tokens.size() ? spec.tokens[idx] : -1);
                    }
                    check(exact, w.category + " " + slot.config.name + " token exact");
                    check(spec.ok, w.category + " " + slot.config.name + " position invariant");
                    check(spec.max_total_drafts <= slot.config.k + slot.config.tail,
                          w.category + " " + slot.config.name + " total draft capacity");
                    check(spec.violations == 0u,
                          w.category + " " + slot.config.name + " per-round assertions");

                    correctness_lines.push_back(
                        w.category + "," + std::to_string(w.context) + "," +
                        slot.config.name + "," + std::to_string(sample) + "," +
                        (exact ? "1" : "0") + "," + std::to_string(timing.rounds) + "," +
                        std::to_string(timing.ngram_hit_rounds) + "," +
                        std::to_string(timing.ngram_accepted_tokens) + "," +
                        std::to_string(spec.violations) + "," +
                        std::to_string(spec.max_total_drafts));
                }
            }
        }
    }

    unsetenv("PHASESHIFT_DFLASH2_HOST_PROPOSAL_D2H");

    std::error_code ec_fs;
    fs::create_directories(root, ec_fs);

    std::ofstream corr(root + "/correctness.csv");
    corr << "category,context,config,sample,token_exact,rounds,ngram_hits,"
            "ngram_accepted,violations,max_total_drafts\n";
    for (const std::string& line : correctness_lines) corr << line << '\n';
    corr.close();

    std::ofstream raw(root + "/raw_perf.csv");
    raw << "category,context,prompt_id,run,config,dflash_k,ngram_n,ngram_tail,window,"
           "bridge_mode,gdn_mode,generated_tokens,rounds,dflash_accepted,ngram_hits,"
           "tail_reached,tail_blocked,ngram_proposed,ngram_accepted,verify_rows,"
           "draft_gpu_ms,proposal_wait_ms,ngram_seed_wait_ms,ngram_lookup_ms,"
           "ngram_tail_h2d_ms,verify_gpu_ms,"
           "rerun_ms,restore_ms,commit_ms,round_ms,decode_ms,tok_s,token_exact\n";
    for (const PerfRow& r : perf_rows) {
        raw << r.category << ',' << r.context << ',' << r.prompt_id << ',' << r.run << ','
            << r.config << ',' << r.dflash_k << ',' << r.ngram_n << ',' << r.ngram_tail << ','
            << r.window << ',' << r.bridge_mode << ',' << r.gdn_mode << ','
            << r.generated_tokens << ',' << r.rounds << ',' << r.dflash_accepted << ','
            << r.ngram_hits << ',' << r.tail_reached << ',' << r.tail_blocked << ','
            << r.ngram_proposed << ',' << r.ngram_accepted << ',' << r.verify_rows << ','
            << r.draft_gpu_ms << ',' << r.proposal_wait_ms << ','
            << r.ngram_seed_wait_ms << ',' << r.ngram_lookup_ms << ','
            << r.ngram_tail_h2d_ms << ',' << r.verify_gpu_ms << ',' << r.rerun_ms << ','
            << r.restore_ms << ',' << r.commit_ms << ',' << r.round_ms << ',' << r.decode_ms
            << ',' << r.tok_s << ',' << (r.token_exact ? 1 : 0) << '\n';
    }
    raw.close();

    struct AggKey {
        std::string category;
        uint32_t context = 0u;
        std::string config;
        uint32_t dflash_k = 0u;
        uint32_t ngram_n = 0u;
        uint32_t ngram_tail = 0u;
        uint32_t window = 0u;
        bool operator==(const AggKey& o) const {
            return category == o.category && context == o.context && config == o.config &&
                   dflash_k == o.dflash_k && ngram_n == o.ngram_n &&
                   ngram_tail == o.ngram_tail && window == o.window;
        }
    };
    struct AggVal {
        std::vector<double> tok_s;
        double rounds = 0.0;
        double generated = 0.0;
        double dflash_accepted = 0.0;
        double ngram_hits = 0.0;
        double tail_reached = 0.0;
        double tail_blocked = 0.0;
        double ngram_proposed = 0.0;
        double ngram_accepted = 0.0;
        double verify_rows = 0.0;
        double round_ms = 0.0;
        double verify_gpu_ms = 0.0;
        double rerun_ms = 0.0;
        double ngram_lookup_ms = 0.0;
        double ngram_seed_wait_ms = 0.0;
        double ngram_tail_h2d_ms = 0.0;
        uint64_t samples = 0u;
    };
    std::vector<std::pair<AggKey, AggVal>> aggs;
    for (const PerfRow& r : perf_rows) {
        AggKey key{r.category, r.context, r.config, r.dflash_k, r.ngram_n, r.ngram_tail,
                   r.window};
        AggVal* value = nullptr;
        for (auto& kv : aggs) {
            if (kv.first == key) {
                value = &kv.second;
                break;
            }
        }
        if (value == nullptr) {
            aggs.push_back({key, AggVal{}});
            value = &aggs.back().second;
        }
        value->tok_s.push_back(r.tok_s);
        value->rounds += r.rounds;
        value->generated += r.generated_tokens;
        value->dflash_accepted += r.dflash_accepted;
        value->ngram_hits += r.ngram_hits;
        value->tail_reached += r.tail_reached;
        value->tail_blocked += r.tail_blocked;
        value->ngram_proposed += r.ngram_proposed;
        value->ngram_accepted += r.ngram_accepted;
        value->verify_rows += r.verify_rows;
        value->round_ms += r.round_ms;
        value->verify_gpu_ms += r.verify_gpu_ms;
        value->rerun_ms += r.rerun_ms;
        value->ngram_lookup_ms += r.ngram_lookup_ms;
        value->ngram_seed_wait_ms += r.ngram_seed_wait_ms;
        value->ngram_tail_h2d_ms += r.ngram_tail_h2d_ms;
        ++value->samples;
    }

    std::ofstream agg(root + "/aggregate_perf.csv");
    agg << "category,context,config,dflash_k,ngram_n,ngram_tail,window,samples,"
           "median_tok_s,emitted_per_round,dflash_accepted_per_round,tail_hit_rate,"
           "tail_reach_rate,tail_accepted_per_round,tail_accepted_per_reached,"
           "tail_blocked_rate,mean_verify_rows,mean_round_ms,mean_verify_gpu_ms,"
           "mean_rerun_ms,mean_ngram_seed_wait_ms,mean_ngram_lookup_ms,"
           "mean_ngram_tail_h2d_ms\n";
    for (auto& kv : aggs) {
        const AggKey& key = kv.first;
        AggVal& v = kv.second;
        std::sort(v.tok_s.begin(), v.tok_s.end());
        const double median = v.tok_s.empty()
                                  ? 0.0
                                  : v.tok_s[v.tok_s.size() / 2u];
        const double rounds = std::max(1.0, v.rounds);
        const double reached = std::max(1.0, v.tail_reached);
        const double rate = v.rounds > 0.0 ? 1.0 / v.rounds : 0.0;
        agg << key.category << ',' << key.context << ',' << key.config << ','
            << key.dflash_k << ',' << key.ngram_n << ',' << key.ngram_tail << ','
            << key.window << ',' << v.samples << ',' << median << ','
            << v.generated / rounds << ',' << v.dflash_accepted / rounds << ','
            << v.ngram_hits * rate << ',' << v.tail_reached * rate << ','
            << v.ngram_accepted / rounds << ',' << v.ngram_accepted / reached << ','
            << v.tail_blocked * rate << ',' << v.verify_rows / rounds << ','
            << v.round_ms / rounds << ',' << v.verify_gpu_ms / rounds << ','
            << v.rerun_ms / rounds << ',' << v.ngram_seed_wait_ms / rounds << ','
            << v.ngram_lookup_ms / rounds << ',' << v.ngram_tail_h2d_ms / rounds << '\n';
        std::printf(
            "GATE2_AGG %s ctx=%u %s tok/s=%.2f E/round=%.3f dflash_acc/round=%.3f "
            "hit=%.3f reach=%.3f tail_acc/round=%.3f tail_acc/reached=%.3f blocked=%.3f "
            "rows=%.1f round_ms=%.2f verify_gpu_ms=%.2f rerun_ms=%.2f\n",
            key.category.c_str(), key.context, key.config.c_str(), median,
            v.generated / rounds, v.dflash_accepted / rounds, v.ngram_hits * rate,
            v.tail_reached * rate, v.ngram_accepted / rounds, v.ngram_accepted / reached,
            v.tail_blocked * rate, v.verify_rows / rounds, v.round_ms / rounds,
            v.verify_gpu_ms / rounds, v.rerun_ms / rounds);
    }
    agg.close();

    std::ofstream env(root + "/environment.json");
    env << "{\n"
        << "  \"gpu\": \"AMD Radeon AI PRO R9700\",\n"
        << "  \"arch\": \"gfx1201\",\n"
        << "  \"mode\": \"" << mode << "\",\n"
        << "  \"model_dir\": \"" << target_dir << "\",\n"
        << "  \"draft_dir\": \"" << model_dir << "\",\n"
        << "  \"corpus_dir\": \"" << corpus_dir << "\",\n"
        << "  \"gen_tokens\": " << gen_tokens << ",\n"
        << "  \"runs\": " << runs << ",\n"
        << "  \"prompts\": " << prompts << ",\n"
        << "  \"ngram_n\": " << ngram_n << ",\n"
        << "  \"ngram_window\": " << ngram_window << ",\n"
        << "  \"verify_numeric_mode\": \"exact\",\n"
        << "  \"configs\": [";
    for (std::size_t i = 0u; i < configs.size(); ++i) {
        env << (i ? ", " : "") << '"' << configs[i].name << '"';
    }
    env << "],\n  \"workloads\": [";
    for (std::size_t i = 0u; i < workloads.size(); ++i) {
        env << (i ? ", " : "") << '"' << workloads[i].category << ':'
            << workloads[i].context << '"';
    }
    env << "]\n}\n";
    env.close();

    std::printf("GATE2 parity=%u/%u ngram_hit_rounds=%u ngram_accepted=%u rows=%zu\n",
                parity_ok, parity_total, ngram_hit_rounds_total, ngram_accepted_total,
                perf_rows.size());
    check(parity_ok == parity_total, "target parity");
    if (!perf_mode) {
        check(ngram_hit_rounds_total > 0u, "real Ngram hit > 0");
        check(ngram_accepted_total > 0u, "real Ngram accepted > 0");
    }

    for (RunSlot& slot : slots) {
        if (slot.decoder_valid) {
            (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(slot.decoder);
        }
        if (slot.sequence.has_value()) {
            (void)ps::qwen35::release_paged_sequence_state(*slot.sequence, stream);
            slot.sequence.reset();
        }
    }
    (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
    (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
    (void)ps::qwen35::executor_shutdown(ex);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    (void)arena.shutdown();
    std::printf("test_dflash2_ngram_tail_gate2: mode=%s failed=%d\n", mode.c_str(), g_fail);
    return g_fail == 0 ? 0 : 1;
}
