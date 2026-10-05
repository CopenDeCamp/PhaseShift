#include "compute_runtime.h"

#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <phaseshift/models/qwen35/runtime/runtime_request.h>
#include <phaseshift/models/qwen35/runtime/sampling_params.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt == 1) ++n;
    }
    return n;
}

void usage(const char* prog) {
    printf("usage: %s [options]\n", prog);
    printf("\nPhaseShift inference runtime for Qwen 3.5 Dense on AMD RDNA4\n");
    printf("options:\n");
    printf("  --model-dir PATH       BF16 or quantized model directory (auto-detected via\n");
    printf("                         phaseshift_quantization.json) (required)\n");
    printf("  --input-ids-file PATH  whitespace separated prompt token ids\n");
    printf("  --max-new-tokens N     max tokens to generate (default 32)\n");
    printf("  --max-seq-len N        max sequence length (default 512)\n");
    printf("  --arena-gib N          arena size in GiB (default 16)\n");
    printf("  --page-tokens N        paged KV page size in tokens (default 16)\n");
    printf("  --max-concurrent-requests N  max in-flight requests (default 1)\n");
    printf("  --kv-cache-capacity-tokens N physical KV token capacity (0 = one context)\n");
    printf("  --device N             GPU device (default 0)\n");
    printf("  --kv-cache-dtype TYPE  bf16 | fp8_e4m3 | psq4 | psq8 (default bf16)\n");
    printf("  --verify-weights 0|1      verify quantized payload CRC32 on load (default 0)\n");
    printf("  --dump-logits PATH      append per-step sampled logits rows (raw f32) to PATH\n");
    printf("                          row k aligns with GENERATED_IDS[k] (prefill row first)\n");
    printf("  --temperature F        sampling temperature (default 0 = greedy)\n");
    printf("  --top-p F              nucleus sampling threshold in (0, 1] (default 1)\n");
    printf("  --top-k N              keep the N highest logits (default 0 = disabled)\n");
    printf("  --seed N               sampling seed (default 0)\n");
    printf("  --dflash2-model-dir PATH  DFlash2 draft model directory; enables greedy\n");
    printf("                            speculative decoding (requires --temperature 0)\n");
    printf("  --dflash2-drafts N     speculative draft tokens per round (default 7)\n");
    printf("  --dflash2-stats 0|1    print speculative decode statistics (default 1)\n");
    printf("  --model-host PATH      attach model weights from a resident model host\n");
    printf("                          (default: PHASESHIFT_MODEL_HOST_SOCKET, empty = local load)\n");
    printf("  --serve-stdio          serve JSON Lines generation requests on stdin/stdout\n");
    printf("  --help                 show this help\n");
}

bool parse_uint(const char* s, unsigned long& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    out = std::strtoul(s, &end, 10);
    return end != s && *end == '\0';
}

bool parse_float(const char* s, float& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    out = std::strtof(s, &end);
    return end != s && *end == '\0';
}

bool parse_u64(const char* s, uint64_t& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    out = std::strtoull(s, &end, 10);
    return end != s && *end == '\0';
}

std::vector<int32_t> read_input_ids(const std::string& path) {
    std::vector<int32_t> ids;
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return ids;
    long v = 0;
    while (std::fscanf(f, "%ld", &v) == 1) {
        ids.push_back(static_cast<int32_t>(v));
    }
    std::fclose(f);
    return ids;
}

const char* finish_reason_name(ps::qwen35::runtime::FinishReason reason) {
    using ps::qwen35::runtime::FinishReason;
    switch (reason) {
        case FinishReason::Eos: return "stop";
        case FinishReason::MaxNewTokens: return "length";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::Error: return "error";
        case FinishReason::None: return "none";
    }
    return "none";
}

void print_model_geometry(const ps::qwen35::Qwen35Model& model) {
    const auto& text_config = model.text_config();
    const uint32_t kv_heads = static_cast<uint32_t>(text_config.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(text_config.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(text_config);
    const auto gdn_layout =
        ps::qwen35::GdnStatePoolLayout::from_text_config(text_config);

    printf("MODEL_HIDDEN_SIZE=%zu\n", text_config.hidden_size);
    printf("MODEL_INTERMEDIATE_SIZE=%zu\n", text_config.intermediate_size);
    printf("MODEL_LAYERS=%zu\n", text_config.num_hidden_layers);
    printf("MODEL_GDN_LAYERS=%u\n", gdn_layout.num_gdn_states);
    printf("MODEL_ATTENTION_LAYERS=%u\n", num_attn_layers);
    printf("MODEL_Q_HEADS=%zu\n", text_config.num_attention_heads);
    printf("MODEL_KV_HEADS=%u\n", kv_heads);
    printf("MODEL_HEAD_DIM=%u\n", head_dim);
    printf("MODEL_GDN_KEY_HEADS=%zu\n", text_config.linear_num_key_heads);
    printf("MODEL_GDN_VALUE_HEADS=%u\n", gdn_layout.num_v_heads);
    printf("MODEL_GDN_KEY_HEAD_DIM=%u\n", gdn_layout.head_k);
    printf("MODEL_GDN_VALUE_HEAD_DIM=%u\n", gdn_layout.head_v);
    printf("MODEL_GDN_CONV_DIM=%u\n", gdn_layout.conv_dim);
    printf("MODEL_GDN_CONV_HISTORY=%u\n", gdn_layout.conv_history);
    fflush(stdout);
}

void dump_mem_debug(ps::app::Qwen35ComputeRuntime& runtime) {
    const auto& model = runtime.model();
    auto& executor = runtime.executor();
    const auto& gdn_pool = runtime.gdn_pool();
    const auto& wts = model.weights();
    uint64_t bf16_bytes = 0, w4_codes = 0, w4_sscales = 0, w4_cscales = 0;
    uint64_t w8_codes = 0, w8_sscales = 0, w8_cscales = 0;
    auto add_matrix = [&](const ps::weights::MatrixWeight& m) {
        using E = ps::weights::MatrixEncoding;
        if (m.rows == 0) return;
        const uint64_t r = m.rows;
        if (m.encoding == E::Bf16) {
            bf16_bytes += r * m.cols * 2u;
        } else if (m.encoding == E::Psq4) {
            w4_codes += r * (m.k_padded / 2u);
            w4_sscales += r * m.storage_scale_stride_bytes;
            w4_cscales += r * m.scale_row_stride_bytes;
        } else if (m.encoding == E::Psq8) {
            w8_codes += r * m.k_padded;
            w8_sscales += r * m.storage_scale_stride_bytes;
            w8_cscales += r * m.scale_row_stride_bytes;
        }
    };
    add_matrix(wts.embed_tokens);
    if (!wts.lm_head_tied) add_matrix(wts.lm_head);
    for (const auto& l : wts.layers) {
        add_matrix(l.mlp_gate_proj);
        add_matrix(l.mlp_up_proj);
        add_matrix(l.mlp_down_proj);
        if (l.is_gdn) {
            add_matrix(l.attn_in_proj_a);
            add_matrix(l.attn_in_proj_b);
            add_matrix(l.attn_in_proj_qkv);
            add_matrix(l.attn_in_proj_z);
            add_matrix(l.attn_out_proj);
        } else {
            add_matrix(l.attn_q_proj);
            add_matrix(l.attn_k_proj);
            add_matrix(l.attn_v_proj);
            add_matrix(l.attn_o_proj);
        }
    }
    uint64_t staging_dispatch_count = 0;
    uint64_t staging_max_image_bytes = 0;
    uint64_t staging_shared_program_bytes = 0;
    for (size_t b = 0; b < ps::runtime::ProgramSet::kRowBucketCount; ++b) {
        const auto& p = executor.program_set.programs[b];
        printf("ROWBUCKET_%zu_WORKSPACE_BYTES=%zu\n", b, p.workspace.total_bytes);
        staging_dispatch_count += p.dispatches.size();
        const uint64_t slot_base = executor.staging_pool.slot(static_cast<uint32_t>(b), 0);
        for (size_t i = 0; i < p.dispatches.size(); ++i) {
            const uint64_t slot = slot_base + i;
            if (slot < executor.staging_pool.bytes.size()) {
                const uint64_t image = executor.staging_pool.bytes[slot];
                if (image > staging_max_image_bytes) staging_max_image_bytes = image;
            }
        }
        if (b < executor.program_staging_meta.size()) {
            staging_shared_program_bytes += executor.program_staging_meta[b].device_bytes;
        }
    }
    printf("WEIGHTS_BF16_BYTES=%llu\n", (unsigned long long)bf16_bytes);
    printf("WEIGHTS_W4_CODES_BYTES=%llu\n", (unsigned long long)w4_codes);
    printf("WEIGHTS_W4_STORAGE_SCALE_BYTES=%llu\n", (unsigned long long)w4_sscales);
    printf("WEIGHTS_W4_COMPUTE_SCALE_BYTES=%llu\n", (unsigned long long)w4_cscales);
    printf("WEIGHTS_W8_CODES_BYTES=%llu\n", (unsigned long long)w8_codes);
    printf("WEIGHTS_W8_STORAGE_SCALE_BYTES=%llu\n", (unsigned long long)w8_sscales);
    printf("WEIGHTS_W8_COMPUTE_SCALE_BYTES=%llu\n", (unsigned long long)w8_cscales);
    printf("DISPATCH_STAGING_DEVICE_BYTES=%llu\n",
           (unsigned long long)executor.staging_pool.pool_bytes);
    printf("DISPATCH_STAGING_HOST_BYTES=%llu\n",
           (unsigned long long)executor.dispatch_staging_bytes);
    printf("STAGING_DISPATCH_COUNT=%llu\n", (unsigned long long)staging_dispatch_count);
    printf("STAGING_MAX_IMAGE_BYTES=%llu\n", (unsigned long long)staging_max_image_bytes);
    printf("STAGING_SHARED_PROGRAM_BYTES=%llu\n", (unsigned long long)staging_shared_program_bytes);
    printf("EXECUTION_WORKSPACE_BYTES=%zu\n", executor.execution_workspace.allocation_bytes());
    printf("GDN_STATE_BYTES=%zu\n", gdn_pool.reserved_bytes());
    printf("ARENA_USED_BYTES=%zu\n", runtime.arena().used());
    printf("ARENA_CAPACITY_BYTES=%zu\n", runtime.arena().capacity());
    printf("ARENA_COMMITTED_BYTES=%zu\n", runtime.arena().committed());
    printf("ARENA_VMM=%d\n", runtime.arena().is_vmm() ? 1 : 0);
}

struct LogitsDumper {
    FILE* file = nullptr;
    std::size_t vocab = 0;
    std::vector<std::size_t> step_rows;

    bool open(const std::string& path, std::size_t vocab_size) {
        vocab = vocab_size;
        file = std::fopen(path.c_str(), "wb");
        return file != nullptr;
    }

    bool step(ps::qwen35::Executor& executor) {
        if (file == nullptr) return true;
        const auto& lg = executor.logits;
        if (lg.ndim() != 2 || lg.dim(1) != vocab || lg.element_size() != sizeof(float)) {
            fprintf(stderr,
                    "dump-logits: unexpected logits shape ndim=%zu dim0=%zu dim1=%zu elem=%zu (vocab=%zu)\n",
                    lg.ndim(), lg.ndim() > 0 ? lg.dim(0) : 0,
                    lg.ndim() > 1 ? lg.dim(1) : 0, lg.element_size(), vocab);
            return false;
        }
        const std::size_t rows = lg.dim(0);
        std::vector<float> host(rows * vocab);
        hipError_t herr = hipMemcpy(host.data(), lg.data<float>(),
                                    rows * vocab * sizeof(float), hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            fprintf(stderr, "dump-logits copy failed: %s\n", hipGetErrorString(herr));
            return false;
        }
        if (std::fwrite(host.data(), sizeof(float), rows * vocab, file) != rows * vocab) {
            fprintf(stderr, "dump-logits write failed\n");
            return false;
        }
        step_rows.push_back(rows);
        return true;
    }

    void finish() {
        if (file == nullptr) return;
        std::fclose(file);
        file = nullptr;
        std::size_t total_rows = 0;
        printf("LOGITS_VOCAB=%zu\n", vocab);
        printf("LOGITS_STEP_ROWS=");
        for (std::size_t i = 0; i < step_rows.size(); ++i) {
            if (i > 0) printf(",");
            printf("%zu", step_rows[i]);
            total_rows += step_rows[i];
        }
        printf("\n");
        printf("LOGITS_TOTAL_ROWS=%zu\n", total_rows);
    }
};

void dump_hidden_debug(ps::qwen35::Executor& executor, const char* tag) {
    const char* h = std::getenv("PSQ_DEBUG_HIDDEN");
    if (h == nullptr || h[0] == '\0') return;
    for (int t = 0; t < 2; ++t) {
        const auto& ten = (t == 0) ? executor.output_hidden : executor.token_hidden;
        if (ten.ndim() != 2 || ten.element_size() != sizeof(std::uint16_t)) {
            fprintf(stderr, "HIDDENDBG %s t=%d bad shape ndim=%zu\n", tag, t, ten.ndim());
            continue;
        }
        const std::size_t rows = ten.dim(0);
        const std::size_t cols = ten.dim(1);
        std::vector<std::uint16_t> raw(rows * cols);
        hipError_t he = hipMemcpy(raw.data(), ten.data<void>(),
                                rows * cols * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
        if (he != hipSuccess) {
            fprintf(stderr, "HIDDENDBG %s t=%d copy failed: %s\n", tag, t, hipGetErrorString(he));
            continue;
        }
        double mx = 0.0;
        std::size_t nz = 0;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            const std::uint32_t u = static_cast<std::uint32_t>(raw[i]) << 16;
            float fv;
            std::memcpy(&fv, &u, sizeof(fv));
            const double a = std::abs(fv);
            if (a > mx) mx = a;
            if (u != 0) ++nz;
        }
        fprintf(stderr, "HIDDENDBG %s t=%d rows=%zu cols=%zu max=%.6e nonzero=%zu\n",
                tag, t, rows, cols, mx, nz);
        const char* dbg_file = std::getenv("PSQ_DEBUG_HIDDEN_FILE");
        if (dbg_file != nullptr && dbg_file[0] != '\0') {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%s_t%d.bin", dbg_file, tag, t);
            FILE* hf = std::fopen(path, "wb");
            if (hf != nullptr) {
                std::vector<float> fv(rows * cols);
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    const std::uint32_t u = static_cast<std::uint32_t>(raw[i]) << 16;
                    std::memcpy(&fv[i], &u, sizeof(float));
                }
                std::fwrite(fv.data(), sizeof(float), fv.size(), hf);
                std::fclose(hf);
            }
        }
    }
}

int run_single_shot(ps::app::Qwen35ComputeRuntime& runtime,
                    const std::vector<int32_t>& prompt_ids,
                    unsigned long max_new_tokens,
                    const ps::qwen35::runtime::SamplingConfig& sampling,
                    const std::string& kv_cache_dtype,
                    const std::string& dump_logits_path) {
    auto& executor = runtime.executor();
    auto& kv_pool = runtime.kv_pool();
    print_model_geometry(runtime.model());
    if (std::getenv("PHASESHIFT_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }

    LogitsDumper dumper;
    if (!dump_logits_path.empty()) {
        if (!dumper.open(dump_logits_path, runtime.model().text_config().vocab_size)) {
            fprintf(stderr, "cannot open --dump-logits path %s\n", dump_logits_path.c_str());
            return 1;
        }
    }

    auto submit_result = runtime.submit(prompt_ids, static_cast<uint32_t>(max_new_tokens), sampling);
    if (!submit_result.ok()) {
        fprintf(stderr, "submit failed: %s\n", submit_result.status().message().c_str());
        return 1;
    }
    const uint64_t request_id = submit_result.release();

    auto t_prefill_start = std::chrono::steady_clock::now();
    auto first_step = runtime.step();
    if (!first_step.ok()) {
        fprintf(stderr, "prefill step failed: %s\n", first_step.status().message().c_str());
        auto cleanup_st = runtime.cancel_all();
        if (!cleanup_st.ok()) {
            fprintf(stderr, "cleanup after prefill failure: %s\n", cleanup_st.message().c_str());
        }
        return 1;
    }
    if (!first_step.value().executed) {
        fprintf(stderr, "prefill step did not execute\n");
        auto cleanup_st = runtime.cancel_all();
        if (!cleanup_st.ok()) {
            fprintf(stderr, "cleanup after prefill failure: %s\n", cleanup_st.message().c_str());
        }
        return 1;
    }
    hipError_t herr = hipStreamSynchronize(runtime.stream());
    if (herr != hipSuccess) {
        fprintf(stderr, "prefill sync failed: %s\n", hipGetErrorString(herr));
        return 1;
    }
    if (!dumper.step(executor)) return 1;
    dump_hidden_debug(executor, "prefill");
    auto t_prefill_end = std::chrono::steady_clock::now();

    auto t_decode_start = std::chrono::steady_clock::now();
    uint32_t decode_tokens = 0;
    while (runtime.has_pending()) {
        auto step_result = runtime.step();
        if (!step_result.ok()) {
            fprintf(stderr, "decode step failed: %s\n", step_result.status().message().c_str());
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after decode failure: %s\n", cleanup_st.message().c_str());
            }
            return 1;
        }
        if (!step_result.value().executed) {
            fprintf(stderr, "decode stuck: pending request remains but step did not execute\n");
            auto cleanup_st = runtime.cancel_all();
            if (!cleanup_st.ok()) {
                fprintf(stderr, "cleanup after stuck decode: %s\n", cleanup_st.message().c_str());
            }
            return 1;
        }
        if (dumper.file != nullptr) {
            herr = hipStreamSynchronize(runtime.stream());
            if (herr != hipSuccess) {
                fprintf(stderr, "decode sync failed: %s\n", hipGetErrorString(herr));
                return 1;
            }
            if (!dumper.step(executor)) return 1;
        }
        if (std::getenv("PSQ_DEBUG_HIDDEN") != nullptr) {
            herr = hipStreamSynchronize(runtime.stream());
            if (herr != hipSuccess) {
                fprintf(stderr, "decode sync failed: %s\n", hipGetErrorString(herr));
                return 1;
            }
            char tag[32];
            std::snprintf(tag, sizeof(tag), "decode%d", static_cast<int>(decode_tokens));
            dump_hidden_debug(executor, tag);
        }
        decode_tokens += step_result.value().num_tokens;
    }
    herr = hipStreamSynchronize(runtime.stream());
    if (herr != hipSuccess) {
        fprintf(stderr, "decode sync failed: %s\n", hipGetErrorString(herr));
        return 1;
    }
    auto t_decode_end = std::chrono::steady_clock::now();

    const auto* request = runtime.request(request_id);
    if (request == nullptr ||
        request->state != ps::qwen35::runtime::RequestState::Finished) {
        fprintf(stderr, "request did not finish\n");
        return 1;
    }
    std::vector<int32_t> generated(request->generated);

    const auto prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    const auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const double decode_tps = decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;

    printf("MODEL_LAYERS=%zu\n", runtime.model().text_config().num_hidden_layers);
    printf("PROMPT_TOKENS=%zu\n", prompt_ids.size());
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    printf("ARENA_USED_GB=%.3f\n", static_cast<double>(runtime.arena().used()) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(runtime.arena().committed()) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%zu\n", kv_pool.reserved_bytes());

    printf("GENERATED_IDS=");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) printf(",");
        printf("%d", generated[i]);
    }
    printf("\n");
    dumper.finish();
    fflush(stdout);

    const uint32_t used_kv_pages = kv_pool.num_used_pages();
    const uint32_t used_seq_slots = runtime.slot_pool().num_used_slots();
    const uint32_t used_gdn_slots = runtime.gdn_pool().num_used_slots();
    if (used_kv_pages != 0 || used_seq_slots != 0 || used_gdn_slots != 0) {
        fprintf(stderr,
                "pool leak after generation: kv_pages=%u seq_slots=%u gdn_slots=%u\n",
                used_kv_pages, used_seq_slots, used_gdn_slots);
        return 1;
    }
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

int run_dflash2_shot(ps::app::Qwen35ComputeRuntime& runtime,
                     const std::vector<int32_t>& prompt_ids,
                     unsigned long max_new_tokens,
                     const ps::qwen35::dflash2::DFlash2Config& dflash_config,
                     const std::string& dflash2_model_dir,
                     unsigned long dflash2_drafts,
                     int dflash2_stats,
                     const std::string& kv_cache_dtype,
                     const ps::qwen35::runtime::SamplingConfig& sampling) {
    auto& executor = runtime.executor();
    const auto& model = runtime.model();
    auto& slot_pool = runtime.slot_pool();
    auto& gdn_pool = runtime.gdn_pool();
    auto& kv_pool = runtime.kv_pool();
    auto& arena = runtime.arena();
    const hipStream_t stream = runtime.stream();
    const auto& text_config = model.text_config();
    print_model_geometry(model);
    if (std::getenv("PHASESHIFT_MEM_DEBUG") != nullptr) {
        dump_mem_debug(runtime);
    }

    auto dflash_weights_result = runtime.load_dflash2_weights(dflash_config);
    if (!dflash_weights_result.ok()) {
        fprintf(stderr, "dflash2 weight load failed: %s\n",
                dflash_weights_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::dflash2::DFlash2Weights dflash_weights = dflash_weights_result.release();

    const uint32_t max_seq = runtime.config().max_seq_len;

    ps::qwen35::dflash2::DFlash2ExecutorConfig draft_config;
    draft_config.max_rows = static_cast<uint32_t>(dflash_config.block_size);
    draft_config.max_context_rows = static_cast<uint32_t>(dflash_config.sliding_window);
    draft_config.target_lm_head = &model.weights().lm_head;
    draft_config.target_embed_tokens = &model.weights().embed_tokens;
    auto draft_result = ps::qwen35::dflash2::create_dflash2_executor(
        dflash_config, dflash_weights, arena, draft_config);
    if (!draft_result.ok()) {
        fprintf(stderr, "dflash2 executor create failed: %s\n",
                draft_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::dflash2::DFlash2Executor draft = draft_result.release();

    auto sequence_result = ps::qwen35::create_paged_sequence_state(
        slot_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!sequence_result.ok()) {
        fprintf(stderr, "dflash2 sequence create failed: %s\n",
                sequence_result.status().message().c_str());
        (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    ps::qwen35::PagedSequenceState sequence = sequence_result.release();

    auto context_result =
        ps::qwen35::dflash2::create_dflash2_context_state(dflash_config, arena, 0u);
    if (!context_result.ok()) {
        fprintf(stderr, "dflash2 context create failed: %s\n",
                context_result.status().message().c_str());
        (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
        (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    ps::qwen35::dflash2::DFlash2ContextState context = context_result.release();

    ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = static_cast<uint32_t>(dflash2_drafts);
    decoder_config.eos_tokens = text_config.stop_tokens;
    decoder_config.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
    auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
        executor, draft, context, sequence, gdn_pool, arena, decoder_config, stream);
    if (!decoder_result.ok()) {
        fprintf(stderr, "dflash2 spec decoder create failed: %s\n",
                decoder_result.status().message().c_str());
        (void)ps::qwen35::dflash2::dflash2_context_shutdown(context);
        (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
        (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    ps::qwen35::runtime::DFlash2SpecDecoder decoder = decoder_result.release();
    ps::qwen35::runtime::DFlash2SpecTiming timing;
    decoder.timing = &timing;
    ps::qwen35::runtime::dflash2_spec_decoder_set_sampling(decoder, sampling);

    std::vector<int32_t> generated;
    uint32_t decode_tokens = 0u;

    const auto t_prefill_start = std::chrono::steady_clock::now();
    auto prefill_result = ps::qwen35::runtime::dflash2_spec_prefill(
        decoder, prompt_ids.data(), static_cast<uint32_t>(prompt_ids.size()));
    if (!prefill_result.ok()) {
        fprintf(stderr, "dflash2 prefill failed: %s\n",
                prefill_result.status().message().c_str());
        (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
        (void)ps::qwen35::dflash2::dflash2_context_shutdown(context);
        (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
        (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
        return 1;
    }
    int32_t pending = prefill_result.value().pending_token;
    generated.push_back(pending);
    const auto t_prefill_end = std::chrono::steady_clock::now();

    const auto t_decode_start = std::chrono::steady_clock::now();
    bool finished = ps::qwen35::is_stop_token(decoder_config.eos_tokens, pending);
    while (!finished && generated.size() < max_new_tokens) {
        const uint32_t remaining =
            static_cast<uint32_t>(max_new_tokens - generated.size());
        auto step_result =
            ps::qwen35::runtime::dflash2_spec_step(decoder, pending, remaining);
        if (!step_result.ok()) {
            fprintf(stderr, "dflash2 step failed: %s\n",
                    step_result.status().message().c_str());
            (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(context);
            (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
            (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);
            return 1;
        }
        auto step = step_result.release();
        for (uint32_t i = 0; i < step.emitted_count; ++i) {
            generated.push_back(step.emitted[i]);
            ++decode_tokens;
        }
        pending = step.pending_token;
        finished = step.finished;
    }
    hipError_t herr = hipStreamSynchronize(stream);
    if (herr != hipSuccess) {
        fprintf(stderr, "dflash2 sync failed: %s\n", hipGetErrorString(herr));
        return 1;
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    if (dflash2_stats != 0) {
        const float mean_accepted = timing.rounds > 0u
            ? static_cast<float>(timing.accepted_drafts) / static_cast<float>(timing.rounds)
            : 0.0f;
        const float full_rate = timing.rounds > 0u
            ? static_cast<float>(timing.full_accepts) / static_cast<float>(timing.rounds)
            : 0.0f;
        const float emitted_per_round = timing.rounds > 0u
            ? static_cast<float>(timing.generated_tokens) / static_cast<float>(timing.rounds)
            : 0.0f;
        printf("DFLASH2_DRAFTS=%u\n", decoder_config.num_drafts);
        printf("DFLASH2_BLOCK_SIZE=%zu\n", dflash_config.block_size);
        printf("DFLASH2_VERIFY_NUMERIC_MODE=Exact\n");
        printf("DFLASH2_ROUNDS=%u\n", timing.rounds);
        printf("DFLASH2_FULL_ACCEPTS=%u\n", timing.full_accepts);
        printf("DFLASH2_RERUNS=%u\n", timing.reruns);
        printf("DFLASH2_PARTIAL_ACCEPTS=%u\n", timing.partial_accepts);
        printf("DFLASH2_GDN_HISTORY_BYTES=%llu\n",
               static_cast<unsigned long long>(timing.gdn_history_bytes));
        printf("DFLASH2_GDN_SNAPSHOT_BYTES=%llu\n",
               static_cast<unsigned long long>(decoder.gdn_snapshot_bytes));
        printf("DFLASH2_GDN_COMPACT_BYTES=%llu\n",
               static_cast<unsigned long long>(decoder.gdn_compact_bytes));
        printf("DFLASH2_GDN_MODE=%s\n",
               decoder.gdn_compact_commit
                   ? "compact"
                   : (decoder.gdn_history_enabled ? "history" : "rerun"));
        printf("DFLASH2_ACCEPTED_DRAFTS=%u\n", timing.accepted_drafts);
        printf("DFLASH2_MEAN_ACCEPTED=%.3f\n", mean_accepted);
        printf("DFLASH2_FULL_ACCEPT_RATE=%.3f\n", full_rate);
        printf("DFLASH2_EMITTED_PER_ROUND=%.3f\n", emitted_per_round);
        printf("DFLASH2_DRAFT_MS=%.2f\n", timing.draft_ms);
        printf("DFLASH2_D2H_MS=%.2f\n", timing.draft_d2h_ms);
        printf("DFLASH2_DRAFT_GPU_MS=%.2f\n", timing.draft_gpu_ms);
        printf("DFLASH2_PROPOSAL_WAIT_MS=%.2f\n", timing.proposal_wait_ms);
        printf("DFLASH2_PROPOSAL_COPY_MS=%.2f\n", timing.proposal_copy_ms);
        printf("DFLASH2_VERIFY_MS=%.2f\n", timing.verify_ms);
        printf("DFLASH2_VERIFY_GPU_MS=%.2f\n", timing.verify_gpu_ms);
        printf("DFLASH2_DECISION_WAIT_MS=%.2f\n", timing.decision_wait_ms);
        printf("DFLASH2_DECISION_D2H_MS=%.2f\n", timing.decision_copy_ms);
        printf("DFLASH2_DEVICE_TOKEN_BRIDGE=%d\n", decoder.device_token_bridge ? 1 : 0);
        printf("DFLASH2_RERUN_MS=%.2f\n", timing.rerun_ms);
        printf("DFLASH2_GDN_SNAPSHOT_MS=%.2f\n", timing.gdn_snapshot_ms);
        printf("DFLASH2_GDN_RESTORE_MS=%.2f\n", timing.gdn_restore_ms);
        printf("DFLASH2_COMMIT_MS=%.2f\n", timing.dflash_commit_ms);
        printf("DFLASH2_ROUND_MS=%.2f\n", timing.round_ms);
        fflush(stdout);
    }

    (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
    (void)ps::qwen35::dflash2::dflash2_context_shutdown(context);
    (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
    (void)ps::qwen35::dflash2::dflash2_executor_shutdown(draft);

    const auto prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    const auto decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    const double decode_tps = decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;

    printf("MODEL_LAYERS=%zu\n", text_config.num_hidden_layers);
    printf("PROMPT_TOKENS=%zu\n", prompt_ids.size());
    printf("GENERATED_TOKENS=%zu\n", generated.size());
    printf("ARENA_USED_GB=%.3f\n", static_cast<double>(arena.used()) / (1024.0 * 1024.0 * 1024.0));
    printf("ARENA_COMMITTED_GB=%.3f\n",
           static_cast<double>(arena.committed()) / (1024.0 * 1024.0 * 1024.0));
    printf("MODEL_LOAD_MS=%.1f\n", runtime.model_load_ms());
    printf("PREFILL_MS=%.1f\n", prefill_ms);
    printf("DECODE_MS=%.1f\n", decode_ms);
    printf("DECODE_TOKENS=%u\n", decode_tokens);
    printf("DECODE_TOKENS_PER_SEC=%.2f\n", decode_tps);
    printf("KV_CACHE_DTYPE=%s\n", kv_cache_dtype.c_str());
    printf("KV_POOL_RESERVED_BYTES=%zu\n", kv_pool.reserved_bytes());

    printf("GENERATED_IDS=");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) printf(",");
        printf("%d", generated[i]);
    }
    printf("\n");
    fflush(stdout);

    const uint32_t used_kv_pages = kv_pool.num_used_pages();
    const uint32_t used_seq_slots = slot_pool.num_used_slots();
    const uint32_t used_gdn_slots = gdn_pool.num_used_slots();
    if (used_kv_pages != 0 || used_seq_slots != 0 || used_gdn_slots != 0) {
        fprintf(stderr,
                "pool leak after generation: kv_pages=%u seq_slots=%u gdn_slots=%u\n",
                used_kv_pages, used_seq_slots, used_gdn_slots);
        return 1;
    }
    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

void emit_json(const nlohmann::json& value) {
    const std::string line = value.dump();
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void emit_error(const nlohmann::json& request_id, const std::string& message) {
    nlohmann::json ev;
    ev["event"] = "error";
    if (!request_id.is_null()) ev["request_id"] = request_id;
    ev["message"] = message;
    emit_json(ev);
}

void emit_error_with_code(
    const nlohmann::json& request_id, const std::string& code,
    const std::string& message) {
    nlohmann::json ev;
    ev["event"] = "error";
    if (!request_id.is_null()) ev["request_id"] = request_id;
    if (!code.empty()) ev["code"] = code;
    ev["message"] = message;
    emit_json(ev);
}

const char* status_error_code(ps::Status::Code code) {
    switch (code) {
        case ps::Status::Code::invalid_argument:
        case ps::Status::Code::out_of_range:
        case ps::Status::Code::unsupported:
            return "invalid_argument";
        default:
            return "internal";
    }
}

void emit_token(int64_t request_id, int32_t token_id) {
    emit_json(nlohmann::json{
        {"event", "token"}, {"request_id", request_id}, {"token_id", token_id}});
}

constexpr std::size_t kRecentTerminalLimit = 256;

struct DFlashServeSession {
    std::unique_ptr<ps::qwen35::PagedSequenceState> sequence;
    std::unique_ptr<ps::qwen35::dflash2::DFlash2ContextState> context;
    std::unique_ptr<ps::qwen35::runtime::DFlash2SpecDecoder> decoder;
    ps::qwen35::runtime::DFlash2SpecTiming timing{};
    int32_t pending_token = 0;
    bool ready = false;
};

struct ServeGeneration {
    int64_t external_request_id = 0;
    uint64_t internal_request_id = 0;
    std::size_t emitted_tokens = 0;
    uint32_t max_new_tokens = 0;
    bool dflash = false;
    bool finished = false;
    ps::qwen35::runtime::FinishReason finish_reason =
        ps::qwen35::runtime::FinishReason::None;
    uint32_t prompt_tokens = 0;
    std::vector<int32_t> generated;
    std::unique_ptr<DFlashServeSession> dflash_session;
};

struct DFlashServeContext {
    bool enabled = false;
    ps::qwen35::dflash2::DFlash2Config config{};
    std::string model_dir;
    uint32_t drafts = 7u;
    ps::qwen35::dflash2::DFlash2Weights weights{};
    ps::qwen35::dflash2::DFlash2Executor executor{};
    bool executor_ready = false;
};

struct ServeState {
    std::map<int64_t, ServeGeneration> inflight;
    std::deque<int64_t> recent_terminal_ids;
    uint32_t max_inflight = 1;
    bool exit_requested = false;
    bool dflash_enabled = false;
};

struct JsonlInput {
    std::string buffer;
    bool eof = false;
};

void jsonl_poll_and_read(JsonlInput& input, bool block) {
    if (input.eof) return;
    struct pollfd descriptor;
    descriptor.fd = STDIN_FILENO;
    descriptor.events = POLLIN;
    descriptor.revents = 0;
    const int timeout_ms = block ? -1 : 0;
    const int ready = ::poll(&descriptor, 1, timeout_ms);
    if (ready <= 0) return;
    if ((descriptor.revents & (POLLIN | POLLHUP | POLLERR)) == 0) return;
    char chunk[4096];
    const ssize_t count = ::read(STDIN_FILENO, chunk, sizeof(chunk));
    if (count > 0) {
        input.buffer.append(chunk, static_cast<std::size_t>(count));
    } else if (count == 0) {
        input.eof = true;
    } else if (errno != EAGAIN && errno != EINTR) {
        input.eof = true;
    }
}

bool jsonl_take_line(JsonlInput& input, std::string& line) {
    const std::size_t newline = input.buffer.find('\n');
    if (newline == std::string::npos) return false;
    line = input.buffer.substr(0, newline);
    input.buffer.erase(0, newline + 1);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

void serve_emit_tokens(ps::app::Qwen35ComputeRuntime& runtime,
                       ServeGeneration& generation) {
    if (generation.dflash) {
        while (generation.emitted_tokens < generation.generated.size()) {
            emit_token(generation.external_request_id,
                       generation.generated[generation.emitted_tokens]);
            ++generation.emitted_tokens;
        }
        return;
    }
    const auto* request = runtime.request(generation.internal_request_id);
    if (request == nullptr) return;
    while (generation.emitted_tokens < request->generated.size()) {
        emit_token(generation.external_request_id,
                   request->generated[generation.emitted_tokens]);
        ++generation.emitted_tokens;
    }
}

void serve_push_terminal(ServeState& state, int64_t external_request_id) {
    state.recent_terminal_ids.push_back(external_request_id);
    while (state.recent_terminal_ids.size() > kRecentTerminalLimit) {
        state.recent_terminal_ids.pop_front();
    }
}

bool serve_is_recent_terminal(const ServeState& state, int64_t external_request_id) {
    return std::find(state.recent_terminal_ids.begin(),
                     state.recent_terminal_ids.end(),
                     external_request_id) != state.recent_terminal_ids.end();
}

void serve_emit_done(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                     const ServeGeneration& generation) {
    const auto* request = runtime.request(generation.internal_request_id);
    nlohmann::json done;
    done["event"] = "done";
    done["request_id"] = generation.external_request_id;
    if (generation.dflash) {
        done["generated_ids"] = generation.generated;
        done["finish_reason"] = finish_reason_name(generation.finish_reason);
        done["prompt_tokens"] = generation.prompt_tokens;
        done["restored_tokens"] = 0;
        done["prefill_tokens"] = generation.prompt_tokens;
        done["cache_checkpoint_tokens"] = 0;
    } else if (request != nullptr) {
        done["generated_ids"] = request->generated;
        done["finish_reason"] = finish_reason_name(request->finish_reason);
        done["prompt_tokens"] = request->input_tokens.size();
        done["restored_tokens"] = 0;
        done["prefill_tokens"] = request->input_tokens.size();
        done["cache_checkpoint_tokens"] = 0;
    } else {
        done["generated_ids"] = std::vector<int32_t>{};
        done["finish_reason"] = "error";
    }
    emit_json(done);
    serve_push_terminal(state, generation.external_request_id);
}

void serve_dflash_close_session(ServeGeneration& generation, hipStream_t stream) {
    if (generation.dflash_session == nullptr) return;
    if (generation.dflash_session->ready) {
        const char* stats_env = std::getenv("PHASESHIFT_DFLASH2_SERVE_STATS");
        if (stats_env != nullptr && stats_env[0] != '\0' && stats_env[0] != '0') {
            const auto& t = generation.dflash_session->timing;
            const float rounds = static_cast<float>(t.rounds);
            const float mean = t.rounds > 0u
                ? static_cast<float>(t.accepted_drafts) / rounds : 0.0f;
            const float full_rate = t.rounds > 0u
                ? static_cast<float>(t.full_accepts) / rounds : 0.0f;
            const float emitted = t.rounds > 0u
                ? static_cast<float>(t.generated_tokens) / rounds : 0.0f;
            const auto& dec = *generation.dflash_session->decoder;
            fprintf(stderr, "DFLASH2_GDN_MODE=%s\n",
                    dec.gdn_compact_commit
                        ? "compact"
                        : (dec.gdn_history_enabled ? "history" : "rerun"));
            fprintf(stderr, "DFLASH2_GDN_HISTORY_BYTES=%llu\n",
                    static_cast<unsigned long long>(t.gdn_history_bytes));
            fprintf(stderr, "DFLASH2_GDN_SNAPSHOT_BYTES=%llu\n",
                    static_cast<unsigned long long>(dec.gdn_snapshot_bytes));
            fprintf(stderr, "DFLASH2_GDN_COMPACT_BYTES=%llu\n",
                    static_cast<unsigned long long>(dec.gdn_compact_bytes));
            fprintf(stderr, "DFLASH2_ROUNDS=%u\n", t.rounds);
            fprintf(stderr, "DFLASH2_ACCEPTED_DRAFTS=%u\n", t.accepted_drafts);
            fprintf(stderr, "DFLASH2_MEAN_ACCEPTED=%.3f\n", mean);
            fprintf(stderr, "DFLASH2_FULL_ACCEPT_RATE=%.3f\n", full_rate);
            fprintf(stderr, "DFLASH2_EMITTED_PER_ROUND=%.3f\n", emitted);
            fprintf(stderr, "DFLASH2_GENERATED_TOKENS=%u\n", t.generated_tokens);
            fprintf(stderr, "DFLASH2_ROUND_MS=%.3f\n", t.round_ms);
            fflush(stderr);
        }
        (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(
            *generation.dflash_session->decoder);
        (void)ps::qwen35::dflash2::dflash2_context_shutdown(
            *generation.dflash_session->context);
        (void)ps::qwen35::release_paged_sequence_state(
            *generation.dflash_session->sequence, stream);
    }
    generation.dflash_session.reset();
}

void serve_retire(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                  ServeGeneration& generation) {
    if (generation.dflash) {
        serve_dflash_close_session(generation, runtime.stream());
    } else {
        auto retire_status = runtime.retire(generation.internal_request_id);
        if (!retire_status.ok()) {
            fprintf(stderr, "retire request %lld failed: %s\n",
                    static_cast<long long>(generation.external_request_id),
                    retire_status.message().c_str());
        }
    }
    state.inflight.erase(generation.external_request_id);
}

void serve_fail_all(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                    const std::string& message) {
    auto cleanup = runtime.cancel_all();
    if (!cleanup.ok()) {
        fprintf(stderr, "fail-safe cancel_all failed: %s\n", cleanup.message().c_str());
    }
    std::vector<int64_t> ids;
    ids.reserve(state.inflight.size());
    for (const auto& entry : state.inflight) {
        ids.push_back(entry.first);
    }
    for (int64_t id : ids) {
        emit_error(nlohmann::json(id), message);
        serve_push_terminal(state, id);
    }
    for (int64_t id : ids) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        ServeGeneration& generation = it->second;
        if (generation.dflash) {
            serve_dflash_close_session(generation, runtime.stream());
            state.inflight.erase(it);
            continue;
        }
        auto retire_status = runtime.retire(generation.internal_request_id);
        if (!retire_status.ok()) {
            fprintf(stderr, "retire after failure failed: %s\n",
                    retire_status.message().c_str());
        }
        state.inflight.erase(it);
    }
}

void serve_cancel_all_inflight(ps::app::Qwen35ComputeRuntime& runtime,
                               ServeState& state) {
    auto cancel_status = runtime.cancel_all();
    if (!cancel_status.ok()) {
        fprintf(stderr, "cancel_all failed: %s\n", cancel_status.message().c_str());
    }
    std::vector<int64_t> ids;
    ids.reserve(state.inflight.size());
    for (const auto& entry : state.inflight) {
        ids.push_back(entry.first);
    }
    for (int64_t id : ids) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        ServeGeneration& generation = it->second;
        if (generation.dflash) {
            generation.finish_reason = ps::qwen35::runtime::FinishReason::Error;
        }
        serve_emit_tokens(runtime, generation);
        serve_emit_done(runtime, state, generation);
        serve_retire(runtime, state, generation);
    }
}

void serve_step_dflash(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state) {
    std::vector<int64_t> finished;
    for (auto& entry : state.inflight) {
        ServeGeneration& generation = entry.second;
        if (generation.finished) continue;
        if (generation.dflash_session == nullptr || !generation.dflash_session->ready) {
            serve_fail_all(runtime, state, "dflash2 session is not available");
            return;
        }
        if (generation.generated.size() >= generation.max_new_tokens) {
            generation.finished = true;
            generation.finish_reason = ps::qwen35::runtime::FinishReason::MaxNewTokens;
            finished.push_back(entry.first);
            continue;
        }
        auto& session = *generation.dflash_session;
        const uint32_t remaining = static_cast<uint32_t>(
            generation.max_new_tokens - generation.generated.size());
        auto step_result = ps::qwen35::runtime::dflash2_spec_step(
            *session.decoder, session.pending_token, remaining);
        if (!step_result.ok()) {
            serve_fail_all(runtime, state, step_result.status().message());
            return;
        }
        auto step = step_result.release();
        for (uint32_t i = 0; i < step.emitted_count; ++i) {
            generation.generated.push_back(step.emitted[i]);
        }
        session.pending_token = step.pending_token;
        if (step.finished) {
            generation.finished = true;
            generation.finish_reason = ps::qwen35::runtime::FinishReason::Eos;
            finished.push_back(entry.first);
        }
    }
    for (int64_t id : finished) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        serve_emit_tokens(runtime, it->second);
        serve_emit_done(runtime, state, it->second);
        serve_retire(runtime, state, it->second);
    }
}

void serve_step(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state) {
    if (state.inflight.empty()) return;
    if (state.dflash_enabled) {
        serve_step_dflash(runtime, state);
        return;
    }
    auto step_result = runtime.step();
    if (!step_result.ok()) {
        serve_fail_all(runtime, state, step_result.status().message());
        return;
    }
    if (!step_result.value().executed) {
        serve_fail_all(runtime, state,
                       "decode stuck: pending request remains but step did not execute");
        return;
    }
    if (std::getenv("PHASESHIFT_BATCH_TRACE") != nullptr) {
        fprintf(stderr, "PHASESHIFT_BATCH_STEP requests=%u tokens=%u\n",
                step_result.value().num_requests, step_result.value().num_tokens);
    }
    std::vector<int64_t> finished;
    for (auto& entry : state.inflight) {
        serve_emit_tokens(runtime, entry.second);
        const auto* request = runtime.request(entry.second.internal_request_id);
        if (request != nullptr &&
            request->state == ps::qwen35::runtime::RequestState::Finished) {
            finished.push_back(entry.first);
        }
    }
    for (int64_t id : finished) {
        auto it = state.inflight.find(id);
        if (it == state.inflight.end()) continue;
        serve_emit_done(runtime, state, it->second);
        serve_retire(runtime, state, it->second);
    }
}

bool serve_cancel_one(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                      int64_t external_request_id) {
    auto it = state.inflight.find(external_request_id);
    if (it == state.inflight.end()) return false;
    ServeGeneration& generation = it->second;
    if (generation.dflash) {
        generation.finish_reason = ps::qwen35::runtime::FinishReason::Cancelled;
        serve_emit_tokens(runtime, generation);
        serve_emit_done(runtime, state, generation);
        serve_retire(runtime, state, generation);
        return true;
    }
    const ps::Status cancel_status = runtime.cancel(generation.internal_request_id);
    if (!cancel_status.ok()) {
        serve_fail_all(runtime, state, cancel_status.message());
        return true;
    }
    serve_emit_tokens(runtime, generation);
    const auto* request = runtime.request(generation.internal_request_id);
    if (request == nullptr ||
        request->state != ps::qwen35::runtime::RequestState::Finished) {
        serve_fail_all(runtime, state, "cancel did not finish the request");
        return true;
    }
    serve_emit_done(runtime, state, generation);
    serve_retire(runtime, state, generation);
    return true;
}

ps::Status serve_dflash_open_session(DFlashServeContext& dflash,
                                     ps::app::Qwen35ComputeRuntime& runtime,
                                     ServeGeneration& generation,
                                     const std::vector<int32_t>& input_ids,
                                     const ps::qwen35::runtime::SamplingConfig& sampling) {
    auto& slot_pool = runtime.slot_pool();
    auto& gdn_pool = runtime.gdn_pool();
    auto& kv_pool = runtime.kv_pool();
    auto& arena = runtime.arena();
    const hipStream_t stream = runtime.stream();
    const uint32_t max_seq = runtime.config().max_seq_len;
    const ps::qwen35::StopTokens& eos_tokens =
        runtime.model().text_config().stop_tokens;

    auto session = std::make_unique<DFlashServeSession>();
    auto sequence_result = ps::qwen35::create_paged_sequence_state(
        slot_pool, gdn_pool, kv_pool, max_seq, stream);
    if (!sequence_result.ok()) return sequence_result.status();
    session->sequence =
        std::make_unique<ps::qwen35::PagedSequenceState>(sequence_result.release());

    auto context_result =
        ps::qwen35::dflash2::create_dflash2_context_state(dflash.config, arena, 0u);
    if (!context_result.ok()) {
        (void)ps::qwen35::release_paged_sequence_state(*session->sequence, stream);
        return context_result.status();
    }
    session->context =
        std::make_unique<ps::qwen35::dflash2::DFlash2ContextState>(
            context_result.release());

    ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_config;
    decoder_config.num_drafts = dflash.drafts;
    decoder_config.eos_tokens = eos_tokens;
    decoder_config.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
    auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
        runtime.executor(), dflash.executor, *session->context, *session->sequence,
        gdn_pool, arena, decoder_config, stream);
    if (!decoder_result.ok()) {
        (void)ps::qwen35::dflash2::dflash2_context_shutdown(*session->context);
        (void)ps::qwen35::release_paged_sequence_state(*session->sequence, stream);
        return decoder_result.status();
    }
    session->decoder =
        std::make_unique<ps::qwen35::runtime::DFlash2SpecDecoder>(
            decoder_result.release());
    session->decoder->timing = &session->timing;
    ps::qwen35::runtime::dflash2_spec_decoder_set_sampling(*session->decoder, sampling);

    auto prefill_result = ps::qwen35::runtime::dflash2_spec_prefill(
        *session->decoder, input_ids.data(), static_cast<uint32_t>(input_ids.size()));
    if (!prefill_result.ok()) {
        (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(*session->decoder);
        (void)ps::qwen35::dflash2::dflash2_context_shutdown(*session->context);
        (void)ps::qwen35::release_paged_sequence_state(*session->sequence, stream);
        return prefill_result.status();
    }

    const int32_t pending = prefill_result.value().pending_token;
    session->pending_token = pending;
    session->ready = true;
    generation.dflash_session = std::move(session);
    generation.generated.push_back(pending);
    if (ps::qwen35::is_stop_token(eos_tokens, pending)) {
        generation.finished = true;
        generation.finish_reason = ps::qwen35::runtime::FinishReason::Eos;
    }
    return ps::Status::make_ok();
}

bool serve_parse_generate(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                          DFlashServeContext& dflash,
                          const nlohmann::json& request) {
    if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
        emit_error(nullptr, "missing or invalid request_id");
        return false;
    }
    const nlohmann::json request_id = request["request_id"];
    const int64_t external_request_id = request_id.get<int64_t>();
    if (state.inflight.count(external_request_id) > 0 ||
        serve_is_recent_terminal(state, external_request_id)) {
        emit_error(request_id, "duplicate request_id");
        return false;
    }
    if (state.inflight.size() >= state.max_inflight) {
        emit_error(request_id, "compute concurrency limit reached");
        return false;
    }

    for (const auto& entry : request.items()) {
        const std::string& key = entry.key();
        if (key != "op" && key != "request_id" && key != "input_ids" &&
            key != "max_new_tokens" && key != "temperature" && key != "top_p" &&
            key != "top_k" && key != "seed" &&
            key != "prefix_cache_checkpoint_position") {
            emit_error_with_code(request_id, "invalid_argument",
                                 "unknown request field: " + key);
            return false;
        }
    }

    if (!request.contains("input_ids") || !request["input_ids"].is_array() ||
        request["input_ids"].empty()) {
        emit_error(request_id, "input_ids must be a non-empty array");
        return false;
    }
    std::vector<int32_t> input_ids;
    input_ids.reserve(request["input_ids"].size());
    bool ids_ok = true;
    for (const auto& value : request["input_ids"]) {
        if (!value.is_number_integer()) { ids_ok = false; break; }
        input_ids.push_back(value.get<int32_t>());
    }
    if (!ids_ok) {
        emit_error(request_id, "input_ids must contain integers");
        return false;
    }

    if (!request.contains("max_new_tokens") || !request["max_new_tokens"].is_number_integer() ||
        request["max_new_tokens"].get<long long>() <= 0) {
        emit_error(request_id, "max_new_tokens must be a positive integer");
        return false;
    }
    const uint32_t max_new_tokens =
        static_cast<uint32_t>(request["max_new_tokens"].get<long long>());

    if (request.contains("prefix_cache_checkpoint_position")) {
        if (!request["prefix_cache_checkpoint_position"].is_number_integer()) {
            emit_error_with_code(
                request_id, "invalid_argument",
                "prefix_cache_checkpoint_position must be an integer");
            return false;
        }
        const long long checkpoint_value =
            request["prefix_cache_checkpoint_position"].get<long long>();
        if (checkpoint_value != 0) {
            emit_error_with_code(
                request_id, "invalid_argument",
                "prefix cache is not available in this build");
            return false;
        }
    }

    if (static_cast<uint64_t>(input_ids.size()) + max_new_tokens >
        static_cast<uint64_t>(runtime.config().max_seq_len)) {
        emit_error(request_id, "input_ids + max_new_tokens exceeds max_seq_len");
        return false;
    }

    ps::qwen35::runtime::SamplingConfig sampling;
    if (request.contains("temperature")) {
        if (!request["temperature"].is_number()) {
            emit_error(request_id, "temperature must be a number");
            return false;
        }
        sampling.temperature = request["temperature"].get<float>();
    }
    if (request.contains("top_p")) {
        if (!request["top_p"].is_number()) {
            emit_error(request_id, "top_p must be a number");
            return false;
        }
        sampling.top_p = request["top_p"].get<float>();
    }
    if (request.contains("top_k")) {
        if (!request["top_k"].is_number_integer()) {
            emit_error(request_id, "top_k must be an integer");
            return false;
        }
        sampling.top_k = static_cast<uint32_t>(request["top_k"].get<long long>());
    }
    if (request.contains("seed")) {
        if (!request["seed"].is_number_integer()) {
            emit_error(request_id, "seed must be an integer");
            return false;
        }
        sampling.seed = static_cast<uint64_t>(request["seed"].get<long long>());
    }
    const ps::Status sampling_status = ps::qwen35::runtime::validate_sampling_config(sampling);
    if (!sampling_status.ok()) {
        emit_error(request_id, sampling_status.message());
        return false;
    }

    ServeGeneration generation;
    generation.external_request_id = external_request_id;
    generation.emitted_tokens = 0;
    generation.max_new_tokens = max_new_tokens;
    generation.prompt_tokens = static_cast<uint32_t>(input_ids.size());

    if (dflash.enabled) {
        generation.dflash = true;
        const ps::Status open_status =
            serve_dflash_open_session(dflash, runtime, generation, input_ids, sampling);
        if (!open_status.ok()) {
            emit_error_with_code(request_id, status_error_code(open_status.code()),
                                 open_status.message());
            return false;
        }
        state.inflight.emplace(external_request_id, std::move(generation));
        return true;
    }

    ps::Result<uint64_t> submit_result =
        runtime.submit(std::move(input_ids), max_new_tokens, sampling);
    if (!submit_result.ok()) {
        emit_error_with_code(request_id,
                             status_error_code(submit_result.status().code()),
                             submit_result.status().message());
        return false;
    }
    generation.internal_request_id = submit_result.release();
    state.inflight.emplace(external_request_id, std::move(generation));
    return true;
}

void serve_handle_command(ps::app::Qwen35ComputeRuntime& runtime, ServeState& state,
                          DFlashServeContext& dflash, const std::string& line) {
    nlohmann::json request;
    try {
        request = nlohmann::json::parse(line);
    } catch (const std::exception&) {
        emit_error(nullptr, "malformed json");
        return;
    }
    if (!request.is_object()) {
        emit_error(nullptr, "request must be a json object");
        return;
    }

    const std::string op = request.value("op", std::string());
    if (op == "ping") {
        emit_json(nlohmann::json{
            {"event", "pong"},
            {"protocol_version", 1},
            {"capabilities", {{"prefix_cache", false}}},
        });
        return;
    }
    if (op == "shutdown") {
        serve_cancel_all_inflight(runtime, state);
        emit_json(nlohmann::json{{"event", "shutdown"}});
        state.exit_requested = true;
        return;
    }
    if (op == "cancel") {
        if (!request.contains("request_id") || !request["request_id"].is_number_integer()) {
            emit_error(nullptr, "missing or invalid request_id");
            return;
        }
        const nlohmann::json request_id = request["request_id"];
        const int64_t id = request_id.get<int64_t>();
        if (state.inflight.count(id) > 0) {
            serve_cancel_one(runtime, state, id);
        } else if (serve_is_recent_terminal(state, id)) {
            return;
        } else {
            emit_error(request_id, "unknown or mismatched request_id");
        }
        return;
    }
    if (op == "generate") {
        serve_parse_generate(runtime, state, dflash, request);
        return;
    }
    emit_error(request.contains("request_id") ? request["request_id"] : nlohmann::json(),
               "unknown op");
}

int run_serve_stdio(ps::app::Qwen35ComputeRuntime& runtime,
                    DFlashServeContext& dflash) {
    if (dflash.enabled) {
        auto weights_result = runtime.load_dflash2_weights(dflash.config);
        if (!weights_result.ok()) {
            fprintf(stderr, "dflash2 weight load failed: %s\n",
                    weights_result.status().message().c_str());
            return 1;
        }
        dflash.weights = weights_result.release();

        const auto& text_config = runtime.model().text_config();
        ps::qwen35::dflash2::DFlash2ExecutorConfig draft_config;
        draft_config.max_rows = static_cast<uint32_t>(dflash.config.block_size);
        draft_config.max_context_rows = static_cast<uint32_t>(dflash.config.sliding_window);
        draft_config.target_lm_head = &runtime.model().weights().lm_head;
        draft_config.target_embed_tokens = &runtime.model().weights().embed_tokens;
        auto draft_result = ps::qwen35::dflash2::create_dflash2_executor(
            dflash.config, dflash.weights, runtime.arena(), draft_config);
        if (!draft_result.ok()) {
            fprintf(stderr, "dflash2 executor create failed: %s\n",
                    draft_result.status().message().c_str());
            return 1;
        }
        dflash.executor = draft_result.release();
        dflash.executor_ready = true;
        fprintf(stderr, "DFLASH2_ENABLED=1\n");
        fprintf(stderr, "DFLASH2_DRAFTS=%u\n", dflash.drafts);
        fprintf(stderr, "DFLASH2_BLOCK_SIZE=%zu\n", dflash.config.block_size);
    }
    fprintf(stderr, "PHASESHIFT_SERVE_READY\n");
    fflush(stderr);
    JsonlInput input;
    ServeState state;
    state.max_inflight = runtime.config().max_concurrent_requests;
    state.dflash_enabled = dflash.enabled;
    while (!state.exit_requested) {
        jsonl_poll_and_read(input, state.inflight.empty());
        std::string line;
        while (jsonl_take_line(input, line)) {
            if (line.empty()) continue;
            serve_handle_command(runtime, state, dflash, line);
            if (state.exit_requested) break;
        }
        if (state.exit_requested) break;
        if (!state.inflight.empty()) {
            if (input.eof) {
                serve_cancel_all_inflight(runtime, state);
            } else {
                serve_step(runtime, state);
            }
        } else if (input.eof) {
            break;
        }
    }
    if (!state.inflight.empty()) {
        serve_cancel_all_inflight(runtime, state);
    }

    if (dflash.executor_ready) {
        const ps::Status draft_st =
            ps::qwen35::dflash2::dflash2_executor_shutdown(dflash.executor);
        dflash.executor_ready = false;
        if (!draft_st.ok()) {
            fprintf(stderr, "dflash2 executor shutdown failed: %s\n",
                    draft_st.message().c_str());
        }
    }

    auto shutdown_st = runtime.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "runtime shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string input_ids_file;
    unsigned long max_new_tokens = 32;
    unsigned long max_seq_len = 512;
    unsigned long arena_gib = 16;
    unsigned long page_tokens = 16;
    unsigned long max_concurrent_requests = 1;
    unsigned long kv_cache_capacity_tokens = 0;
    unsigned long device = 0;
    std::string kv_cache_dtype = "bf16";
    int verify_weights = 0;
    std::string dump_logits_path;
    bool serve_stdio = false;
    ps::qwen35::runtime::SamplingConfig sampling;
    unsigned long top_k = 0;
    std::string dflash2_model_dir;
    std::string model_host;
    unsigned long dflash2_drafts = 7;
    int dflash2_stats = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 ||
            (argv[i][0] == '-' && argv[i][1] == 'h' && argv[i][2] == '\0')) {
            usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--model-host") == 0 && i + 1 < argc) {
            model_host = argv[++i];
        } else if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--input-ids-file") == 0 && i + 1 < argc) {
            input_ids_file = argv[++i];
        } else if (std::strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_new_tokens)) { fprintf(stderr, "bad --max-new-tokens\n"); return 2; }
        } else if (std::strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_seq_len)) { fprintf(stderr, "bad --max-seq-len\n"); return 2; }
        } else if (std::strcmp(argv[i], "--arena-gib") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], arena_gib)) { fprintf(stderr, "bad --arena-gib\n"); return 2; }
        } else if (std::strcmp(argv[i], "--page-tokens") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], page_tokens) || page_tokens == 0) { fprintf(stderr, "bad --page-tokens\n"); return 2; }
        } else if (std::strcmp(argv[i], "--max-concurrent-requests") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], max_concurrent_requests) || max_concurrent_requests == 0) {
                fprintf(stderr, "bad --max-concurrent-requests\n"); return 2;
            }
        } else if (std::strcmp(argv[i], "--kv-cache-capacity-tokens") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], kv_cache_capacity_tokens)) {
                fprintf(stderr, "bad --kv-cache-capacity-tokens\n"); return 2;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], device)) { fprintf(stderr, "bad --device\n"); return 2; }
        } else if (std::strcmp(argv[i], "--kv-cache-dtype") == 0 && i + 1 < argc) {
            kv_cache_dtype = argv[++i];
            if (kv_cache_dtype != "bf16" && kv_cache_dtype != "fp8_e4m3" &&
                kv_cache_dtype != "psq4" && kv_cache_dtype != "psq8") {
                fprintf(stderr, "bad --kv-cache-dtype: %s (expected bf16 | fp8_e4m3 | psq4 | psq8)\n", kv_cache_dtype.c_str());
                return 2;
            }
        } else if (std::strcmp(argv[i], "--verify-weights") == 0 && i + 1 < argc) {
            verify_weights = std::atoi(argv[++i]);
            if (verify_weights != 0 && verify_weights != 1) {
                fprintf(stderr, "bad --verify-weights (expected 0 or 1)\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits_path = argv[++i];
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], sampling.temperature)) { fprintf(stderr, "bad --temperature\n"); return 2; }
        } else if (std::strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], sampling.top_p)) { fprintf(stderr, "bad --top-p\n"); return 2; }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], top_k)) { fprintf(stderr, "bad --top-k\n"); return 2; }
            sampling.top_k = static_cast<uint32_t>(top_k);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], sampling.seed)) { fprintf(stderr, "bad --seed\n"); return 2; }
        } else if (std::strcmp(argv[i], "--dflash2-model-dir") == 0 && i + 1 < argc) {
            dflash2_model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--dflash2-drafts") == 0 && i + 1 < argc) {
            if (!parse_uint(argv[++i], dflash2_drafts) || dflash2_drafts == 0) {
                fprintf(stderr, "bad --dflash2-drafts\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--dflash2-stats") == 0 && i + 1 < argc) {
            dflash2_stats = std::atoi(argv[++i]);
            if (dflash2_stats != 0 && dflash2_stats != 1) {
                fprintf(stderr, "bad --dflash2-stats (expected 0 or 1)\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--serve-stdio") == 0) {
            serve_stdio = true;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (model_dir.empty()) {
        fprintf(stderr, "model-dir required\n");
        return 2;
    }

    const bool dflash_enabled = !dflash2_model_dir.empty();
    if (dflash_enabled) {
        if (!dump_logits_path.empty()) {
            fprintf(stderr, "dflash2 speculative decoding does not support --dump-logits\n");
            return 2;
        }
    }

    {
        const ps::Status sampling_status =
            ps::qwen35::runtime::validate_sampling_config(sampling);
        if (!sampling_status.ok()) {
            fprintf(stderr, "invalid sampling option: %s\n",
                    sampling_status.message().c_str());
            return 2;
        }
    }

    ps::app::Qwen35RuntimeConfig config;
    config.model_dir = model_dir;
    config.dflash2_model_dir = dflash_enabled ? dflash2_model_dir : std::string();
    config.model_host_socket = model_host.empty()
        ? ps::models::model_host_socket_from_env()
        : std::optional<std::string>(model_host);
    config.max_seq_len = static_cast<uint32_t>(max_seq_len);
    config.max_concurrent_requests = static_cast<uint32_t>(max_concurrent_requests);
    config.kv_cache_capacity_tokens = static_cast<uint32_t>(kv_cache_capacity_tokens);
    config.arena_bytes = static_cast<std::size_t>(arena_gib) * 1024ull * 1024ull * 1024ull;
    config.page_tokens = static_cast<uint32_t>(page_tokens);
    config.device = static_cast<int>(device);
    config.kv_cache_dtype = kv_cache_dtype == "psq4"
        ? ps::qwen35::KVCacheDType::PSQ4_W32
        : kv_cache_dtype == "psq8"
              ? ps::qwen35::KVCacheDType::PSQ8_W32
              : kv_cache_dtype == "fp8_e4m3"
                    ? ps::qwen35::KVCacheDType::FP8_E4M3
                    : ps::qwen35::KVCacheDType::BF16;
    config.verify_weights = verify_weights != 0;

    ps::qwen35::dflash2::DFlash2Config dflash_config;
    if (dflash_enabled) {
        auto dflash_config_result =
            ps::qwen35::dflash2::read_dflash2_config(dflash2_model_dir);
        if (!dflash_config_result.ok()) {
            fprintf(stderr, "dflash2 config read failed: %s\n",
                    dflash_config_result.status().message().c_str());
            return 1;
        }
        dflash_config = dflash_config_result.release();
        if (dflash2_drafts > dflash_config.max_draft_tokens()) {
            fprintf(stderr, "--dflash2-drafts %lu exceeds block_size-1 %zu\n", dflash2_drafts,
                    dflash_config.max_draft_tokens());
            return 2;
        }
    }

    std::vector<int32_t> prompt_ids;
    if (serve_stdio) {
        config.max_scheduled_tokens = config.max_seq_len;
        if (dflash_enabled) {
            config.max_scheduled_tokens = config.max_seq_len;
            config.max_scheduled_output_rows =
                static_cast<uint32_t>(dflash_config.block_size);
            config.target_hidden_tap_count =
                static_cast<uint32_t>(dflash_config.num_target_layer_ids);
            for (std::size_t t = 0; t < dflash_config.num_target_layer_ids; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(dflash_config.target_layer_ids[t]);
            }
        }
    } else {
        if (!input_ids_file.empty()) {
            prompt_ids = read_input_ids(input_ids_file);
            if (prompt_ids.empty()) {
                fprintf(stderr, "no prompt ids read from %s\n", input_ids_file.c_str());
                return 1;
            }
        } else {
            prompt_ids = {248041, 77091};
        }
        if (static_cast<uint64_t>(prompt_ids.size()) + max_new_tokens > max_seq_len) {
            fprintf(stderr, "prompt + generation exceeds max_seq_len %lu\n", max_seq_len);
            return 2;
        }
        if (dflash_enabled) {
            const unsigned long dflash_rows =
                static_cast<unsigned long>(dflash_config.block_size);
            unsigned long dflash_tokens =
                std::max<unsigned long>(prompt_ids.size(), dflash_rows);
            dflash_tokens = std::min<unsigned long>(dflash_tokens, 2048UL);
            config.max_scheduled_tokens = static_cast<uint32_t>(dflash_tokens);
            config.max_scheduled_output_rows = static_cast<uint32_t>(dflash_rows);
            config.target_hidden_tap_count =
                static_cast<uint32_t>(dflash_config.num_target_layer_ids);
            for (std::size_t t = 0; t < dflash_config.num_target_layer_ids; ++t) {
                config.target_hidden_taps[t] =
                    static_cast<uint32_t>(dflash_config.target_layer_ids[t]);
            }
        } else {
            config.max_scheduled_tokens = static_cast<uint32_t>(
                std::max<unsigned long>(prompt_ids.size() + 1UL, 16UL));
        }
    }

    auto runtime_result = ps::app::Qwen35ComputeRuntime::create(config);
    if (!runtime_result.ok()) {
        fprintf(stderr, "runtime create failed: %s\n",
                runtime_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<ps::app::Qwen35ComputeRuntime> runtime = runtime_result.release();

    DFlashServeContext dflash_serve;
    if (dflash_enabled) {
        dflash_serve.enabled = true;
        dflash_serve.config = dflash_config;
        dflash_serve.model_dir = dflash2_model_dir;
        dflash_serve.drafts = static_cast<uint32_t>(dflash2_drafts);
    }

    if (serve_stdio) {
        return run_serve_stdio(*runtime, dflash_serve);
    }
    if (dflash_enabled) {
        return run_dflash2_shot(*runtime, prompt_ids, max_new_tokens, dflash_config,
                                dflash2_model_dir, dflash2_drafts, dflash2_stats,
                                kv_cache_dtype, sampling);
    }
    return run_single_shot(*runtime, prompt_ids, max_new_tokens, sampling,
                           kv_cache_dtype, dump_logits_path);
}
