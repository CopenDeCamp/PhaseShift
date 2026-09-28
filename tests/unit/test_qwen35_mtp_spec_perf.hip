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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace ps;

namespace {

namespace qwen35 = ps::qwen35;
namespace q35rt = ps::qwen35::runtime;

double wall_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

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
    uint32_t output_rows) {
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
        sequence.position, output_rows);
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

struct ModeStats {
    std::vector<int32_t> tokens;
    double ms = 0.0;
    uint32_t emitted = 0u;
    uint32_t iterations = 0u;
    uint32_t drafts = 0u;
    uint32_t accepted = 0u;
    uint32_t full_accepts = 0u;
    uint32_t reject0 = 0u;
    uint32_t target_forwards = 0u;
    uint32_t d2h_reads = 0u;
    q35rt::SpecDecoderTiming timing{};
};

bool run_greedy(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const std::vector<int32_t>& prompt,
    uint32_t gen_tokens,
    ModeStats& out,
    hipStream_t stream) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> sampled;
    if (!sample_once(executor, sequence, prompt, 1u, sampled, stream)) return false;
    out.tokens.clear();
    out.tokens.push_back(sampled[0]);
    for (uint32_t i = 1u; i < gen_tokens; ++i) {
        std::vector<int32_t> step{out.tokens.back()};
        if (!sample_once(executor, sequence, step, 1u, sampled, stream)) return false;
        out.tokens.push_back(sampled[0]);
    }
    out.ms = wall_ms(t0);
    out.emitted = static_cast<uint32_t>(out.tokens.size());
    out.iterations = out.emitted;
    out.drafts = out.emitted;
    out.accepted = out.emitted;
    out.target_forwards = out.emitted;
    out.d2h_reads = out.emitted;
    return true;
}

bool run_spec(
    q35rt::SpecDecoder& decoder,
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    q35rt::MtpExecutor& mtp,
    q35rt::MtpKvState& mtp_state,
    const std::vector<int32_t>& prompt,
    uint32_t gen_tokens,
    ModeStats& out,
    hipStream_t stream) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<qwen35::ScheduledRequest> requests;
    qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, prompt.data(), static_cast<uint32_t>(prompt.size()),
        sequence.position, 1u);
    auto executed = qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: spec prefill: %s\n", executed.status().message().c_str());
        return false;
    }
    qwen35::BatchExecutionOutput prefill = executed.release();
    int32_t first = -1;
    if (hipMemcpy(&first, prefill.sampled_tokens.data<int32_t>(), sizeof(int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    auto sync = q35rt::spec_decoder_sync_prompt(
        decoder, prompt.data(), prefill.token_hidden.data<ps::bf16_t>(),
        static_cast<uint32_t>(prompt.size()));
    if (!sync.ok()) {
        std::printf("FAIL: spec sync: %s\n", sync.message().c_str());
        return false;
    }
    out.tokens.clear();
    out.tokens.push_back(first);
    int32_t pending = first;
    while (out.tokens.size() < gen_tokens && pending >= 0) {
        auto step = q35rt::spec_decoder_step(decoder, pending);
        if (!step.ok()) {
            std::printf("FAIL: spec step: %s\n", step.status().message().c_str());
            return false;
        }
        q35rt::SpecIterationOutput value = step.release();
        out.tokens.insert(out.tokens.end(), value.emitted_tokens.begin(), value.emitted_tokens.end());
        ++out.iterations;
        if (decoder.config.num_drafts > 0u) {
            out.drafts += decoder.config.num_drafts;
            out.accepted += value.num_accepted_drafts;
            if (value.num_accepted_drafts == decoder.config.num_drafts) ++out.full_accepts;
            if (value.num_accepted_drafts == 0u) ++out.reject0;
        }
        pending = value.pending_token;
        if (value.finished) break;
    }
    if (out.tokens.size() > gen_tokens) out.tokens.resize(gen_tokens);
    out.ms = wall_ms(t0);
    out.emitted = static_cast<uint32_t>(out.tokens.size());
    out.target_forwards = decoder.timing ? decoder.timing->target_forwards : 0u;
    out.d2h_reads = decoder.timing ? decoder.timing->device_to_host_reads : 0u;
    (void)mtp;
    (void)mtp_state;
    return true;
}

void print_aggregate(const char* label, const std::vector<ModeStats>& runs) {
    double ms = 0.0;
    uint64_t emitted = 0u;
    uint64_t iterations = 0u;
    uint64_t drafts = 0u;
    uint64_t accepted = 0u;
    uint64_t full = 0u;
    uint64_t reject0 = 0u;
    uint64_t target_fwd = 0u;
    uint64_t d2h = 0u;
    for (const ModeStats& r : runs) {
        ms += r.ms;
        emitted += r.emitted;
        iterations += r.iterations;
        drafts += r.drafts;
        accepted += r.accepted;
        full += r.full_accepts;
        reject0 += r.reject0;
        target_fwd += r.target_forwards;
        d2h += r.d2h_reads;
    }
    const double ms_per_token = emitted > 0u ? ms / static_cast<double>(emitted) : 0.0;
    const double tok_per_s = ms > 0.0 ? static_cast<double>(emitted) * 1000.0 / ms : 0.0;
    const double tok_per_update = iterations > 0u ? static_cast<double>(emitted) / iterations : 0.0;
    const double avg_accept = drafts > 0u ? static_cast<double>(accepted) / drafts : 0.0;
    std::printf(
        "%s: prompts=%zu ms=%.1f emitted=%llu ms/token=%.3f tok/s=%.2f "
        "updates=%llu tok/update=%.3f avg_accept=%.3f full=%llu reject0=%llu "
        "fwd=%llu d2h=%llu\n",
        label, runs.size(), ms, static_cast<unsigned long long>(emitted), ms_per_token, tok_per_s,
        static_cast<unsigned long long>(iterations), tok_per_update, avg_accept,
        static_cast<unsigned long long>(full), static_cast<unsigned long long>(reject0),
        static_cast<unsigned long long>(target_fwd), static_cast<unsigned long long>(d2h));
}

}  // namespace

int main() {
    const char* model_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_env == nullptr || model_env[0] == '\0') {
        model_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    const char* dir_env = std::getenv("PHASESHIFT_MTP_PERF_DIR");
    const std::string perf_dir =
        (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "tests/data/mtp_perf";
    if (model_env == nullptr || !fs::is_directory(perf_dir)) {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP and provide %s\n", perf_dir.c_str());
        return 77;
    }
    const char* ctx_env = std::getenv("PHASESHIFT_MTP_PERF_CONTEXT");
    const char* gen_env = std::getenv("PHASESHIFT_MTP_PERF_GEN");
    const char* ppc_env = std::getenv("PHASESHIFT_MTP_PERF_PROMPTS");
    const uint32_t context = ctx_env ? static_cast<uint32_t>(std::atoi(ctx_env)) : 128u;
    const uint32_t gen_tokens = gen_env ? static_cast<uint32_t>(std::atoi(gen_env)) : 128u;
    const uint32_t max_prompts = ppc_env ? static_cast<uint32_t>(std::atoi(ppc_env)) : 2u;
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    {
        hipDeviceProp_t prop{};
        (void)hipGetDeviceProperties(&prop, 0);
        int runtime_version = 0;
        (void)hipRuntimeGetVersion(&runtime_version);
        const char* km = std::getenv("PHASESHIFT_QWEN35_KERNEL_MODE");
        std::printf("env: gpu=%s hip_runtime=%d kernel_mode=%s\n", prop.name, runtime_version,
                    km ? km : "default(optimized)");
    }

    auto arena_result = gpu::GpuArena::create(0, 24ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result = qwen35::Qwen35Model::load_from_safetensors(model_env, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL: model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    qwen35::Qwen35Model model = model_result.release();
    if (!model.weights().mtp.present) {
        std::printf("FAIL: model has no MTP weights\n");
        return 1;
    }
    const qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t page_tokens = 16u;
    const uint32_t required_seq_len = context + gen_tokens + 16u;
    const uint32_t num_pages = ceil_div(required_seq_len, page_tokens) + 8u;

    auto seq_pool_result = qwen35::SequenceSlotPool::create(arena, 1u, num_pages + 8u);
    auto gdn_pool_result = qwen35::GdnStatePool::create(
        arena, 1u, qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, count_attention_layers(tc),
        static_cast<uint32_t>(tc.num_key_value_heads), head_dim, qwen35::KVCacheDType::BF16);
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("FAIL: pool create\n");
        return 1;
    }
    qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = std::max(context, 64u);
    exec_config.max_scheduled_requests = 1u;
    exec_config.max_scheduled_output_rows = 16u;
    auto exec_result = qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("FAIL: create_model_executor: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    qwen35::Executor executor = exec_result.release();

    q35rt::MtpExecutorConfig mtp_config;
    mtp_config.num_pages = num_pages + 8u;
    mtp_config.page_tokens = page_tokens;
    mtp_config.max_rows = 16u;
    auto mtp_result = q35rt::create_mtp_executor(model, qwen35::KVCacheDType::BF16, arena, mtp_config, stream);
    if (!mtp_result.ok()) {
        std::printf("FAIL: create_mtp_executor: %s\n", mtp_result.status().message().c_str());
        return 1;
    }
    q35rt::MtpExecutor mtp = mtp_result.release();

    std::vector<std::string> categories = {"prose", "code", "json", "reasoning"};
    std::vector<std::string> prompt_sets;
    std::vector<std::vector<int32_t>> corpus;
    for (const std::string& cat : categories) {
        auto tokens = read_corpus(fs::path(perf_dir) / (cat + ".psktok"));
        if (tokens.size() < context) continue;
        corpus.push_back(std::move(tokens));
        prompt_sets.push_back(cat);
    }
    if (corpus.empty()) {
        std::printf("SKIP: no usable corpus in %s\n", perf_dir.c_str());
        return 77;
    }

    struct Prompt {
        std::string category;
        std::vector<int32_t> tokens;
    };
    std::vector<Prompt> prompts;
    for (std::size_t c = 0; c < corpus.size(); ++c) {
        uint32_t taken = 0u;
        for (uint32_t off = 0u; off + context <= corpus[c].size() && taken < max_prompts;
             off += context) {
            Prompt p;
            p.category = prompt_sets[c];
            p.tokens.assign(corpus[c].begin() + off, corpus[c].begin() + off + context);
            prompts.push_back(std::move(p));
            ++taken;
        }
    }

    const uint32_t ks[4] = {1u, 2u, 4u, 8u};

    const char* k_only_env = std::getenv("PHASESHIFT_MTP_PERF_K");
    const int k_only = (k_only_env != nullptr) ? std::atoi(k_only_env) : -1;
    const bool skip_off = k_only > 0;

    std::vector<ModeStats> off_runs;
    if (!skip_off) {
        for (const Prompt& p : prompts) {
        auto seq_result = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, required_seq_len, stream);
        if (!seq_result.ok()) {
            std::printf("FAIL: seq create\n");
            return 1;
        }
        qwen35::PagedSequenceState seq = seq_result.release();
        ModeStats stats;
        if (!run_greedy(executor, seq, p.tokens, gen_tokens, stats, stream)) return 1;
        off_runs.push_back(std::move(stats));
        (void)qwen35::release_paged_sequence_state(seq, stream);
    }
        print_aggregate("spec_off", off_runs);
    }

    for (uint32_t ki = 0u; ki < 4u; ++ki) {
        const uint32_t k = ks[ki];
        if (k_only >= 0 && static_cast<int>(k) != k_only) continue;
        q35rt::MtpKvConfig mtp_kv_config;
        mtp_kv_config.num_pages = num_pages + 8u;
        mtp_kv_config.page_tokens = page_tokens;
        mtp_kv_config.num_attention_layers = 1u;
        mtp_kv_config.kv_heads = 4u;
        mtp_kv_config.head_dim = head_dim;
        mtp_kv_config.dtype = qwen35::KVCacheDType::BF16;
        auto mtp_state_result = q35rt::create_mtp_kv_state(arena, mtp_kv_config, 0u, stream);
        if (!mtp_state_result.ok()) {
            std::printf("FAIL: mtp state create\n");
            return 1;
        }
        q35rt::MtpKvState mtp_state = mtp_state_result.release();
        std::vector<ModeStats> runs;
        uint32_t matched = 0u;
        for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
            const Prompt& p = prompts[pi];
            (void)q35rt::mtp_kv_reset(mtp_state, stream);
            auto seq_result = qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, required_seq_len, stream);
            if (!seq_result.ok()) {
                std::printf("FAIL: seq create\n");
                return 1;
            }
            qwen35::PagedSequenceState seq = seq_result.release();
            q35rt::SpecDecoderConfig dcfg;
            dcfg.num_drafts = k;
            dcfg.bonus_token_enabled = true;
            auto decoder_result = q35rt::create_spec_decoder(
                executor, mtp, mtp_state, seq, gdn_pool, arena, dcfg, stream);
            if (!decoder_result.ok()) {
                std::printf("FAIL: create_spec_decoder: %s\n", decoder_result.status().message().c_str());
                return 1;
            }
            q35rt::SpecDecoder decoder = decoder_result.release();
            ModeStats stats;
            decoder.timing = &stats.timing;
            if (!run_spec(decoder, executor, seq, mtp, mtp_state, p.tokens, gen_tokens, stats, stream)) return 1;
            if (!skip_off) {
                if (stats.tokens == off_runs[pi].tokens) {
                    ++matched;
                } else {
                    std::size_t fi = 0u;
                    while (fi < stats.tokens.size() && fi < off_runs[pi].tokens.size() &&
                           stats.tokens[fi] == off_runs[pi].tokens[fi]) {
                        ++fi;
                    }
                    std::printf("  MISMATCH k=%u prompt=%zu cat=%s ctx=%u first=%zu off=%d on=%d\n",
                                k, pi, p.category.c_str(), context, fi,
                                fi < off_runs[pi].tokens.size() ? off_runs[pi].tokens[fi] : -1,
                                fi < stats.tokens.size() ? stats.tokens[fi] : -1);
                }
            }
            runs.push_back(std::move(stats));
            (void)q35rt::spec_decoder_shutdown(decoder);
            (void)qwen35::release_paged_sequence_state(seq, stream);
        }
        char label[32];
        std::snprintf(label, sizeof(label), "spec_k%u", k);
        print_aggregate(label, runs);
        std::printf("  token_match=%u/%zu\n", matched, prompts.size());
        if (k == 4u) {
            double draft = 0.0, verify = 0.0, snap = 0.0, rest = 0.0, commit = 0.0, total = 0.0;
            for (const ModeStats& r : runs) {
                draft += r.timing.draft_ms;
                verify += r.timing.verify_ms;
                snap += r.timing.gdn_snapshot_ms;
                rest += r.timing.gdn_restore_ms;
                commit += r.timing.commit_ms;
                total += r.timing.total_ms;
            }
            std::printf(
                "  K4 stage: total=%.1f draft=%.1f verify=%.1f gdn_snap=%.1f gdn_restore=%.1f "
                "commit=%.1f unattributed=%.1f\n",
                total, draft, verify, snap, rest, commit,
                total - draft - verify - snap - rest - commit);
        }
        (void)q35rt::mtp_kv_shutdown(mtp_state);
    }

    (void)qwen35::executor_shutdown(executor);
    (void)q35rt::mtp_executor_shutdown(mtp);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("MTP_GATE4_BASELINE: DONE\n");
    return 0;
}
