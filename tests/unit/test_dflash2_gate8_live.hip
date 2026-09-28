#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/context_state.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using ps::bf16_t;
using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::DFlash2ContextState;
using ps::qwen35::dflash2::DFlash2Executor;
using ps::qwen35::dflash2::DFlash2ExecutorConfig;
using ps::qwen35::dflash2::DFlash2Weights;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena, const std::vector<std::size_t>& shape, ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 128);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool read_prompt(const fs::path& path, std::vector<int32_t>& out) {
    std::FILE* fp = std::fopen(path.string().c_str(), "r");
    if (fp == nullptr) return false;
    out.clear();
    int value = 0;
    int ch = 0;
    bool have = false;
    while ((ch = std::fgetc(fp)) != EOF) {
        if (ch >= '0' && ch <= '9') {
            value = value * 10 + (ch - '0');
            have = true;
        } else if (have) {
            out.push_back(value);
            value = 0;
            have = false;
        }
    }
    if (have) out.push_back(value);
    std::fclose(fp);
    return !out.empty();
}

static bool dump_blob(const fs::path& dir, const std::string& name, const void* data,
                      std::size_t bytes) {
    std::FILE* fp = std::fopen((dir / name).string().c_str(), "wb");
    if (fp == nullptr) return false;
    const std::size_t written = std::fwrite(data, 1, bytes, fp);
    std::fclose(fp);
    return written == bytes;
}

static bool read_reference_tokens(
    const fs::path& dir, std::vector<int32_t>& out) {
    const fs::path fixture = dir / "fixture.safetensors";
    if (!fs::is_regular_file(fixture)) return false;
    auto reader_result = ps::io::SafetensorsReader::open(fixture.string());
    if (!reader_result.ok()) return false;
    ps::io::SafetensorsReader reader = reader_result.release();
    auto spec = reader.tensor_spec("draft_tokens");
    if (!spec.ok() || spec.value().dtype != ps::io::SType::I32) return false;
    std::size_t bytes = 0;
    auto data = reader.tensor_data("draft_tokens", bytes);
    if (!data.ok()) return false;
    out.resize(bytes / sizeof(int32_t));
    std::memcpy(out.data(), data.value(), bytes);
    return true;
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (model_dir == nullptr || model_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2 and _TARGET\n");
        return 77;
    }
    const char* prompt_root_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_FIXTURE");
    const fs::path prompt_root = (prompt_root_env != nullptr && prompt_root_env[0] != '\0')
                                     ? fs::path(prompt_root_env)
                                     : fs::path("build/dflash2-gate51/real");
    if (!fs::is_directory(prompt_root)) {
        std::printf("SKIP: prompt dir missing: %s\n", prompt_root.string().c_str());
        return 77;
    }
    const char* cases_env = std::getenv("PHASESHIFT_DFLASH2_GATE8_CASES");
    const std::string cases = (cases_env != nullptr && cases_env[0] != '\0')
                                  ? std::string(cases_env)
                                  : std::string("real_prose,real_code,real_math");
    const char* dump_env = std::getenv("PHASESHIFT_DFLASH2_GATE8_DUMP");
    const fs::path dump_root = (dump_env != nullptr && dump_env[0] != '\0')
                                   ? fs::path(dump_env)
                                   : fs::path();
    const char* ref_env = std::getenv("PHASESHIFT_DFLASH2_GATE8_FIXTURE");
    const fs::path ref_root = (ref_env != nullptr && ref_env[0] != '\0')
                                  ? fs::path(ref_env)
                                  : fs::path("build/dflash2-gate8-reference");
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "dflash config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();
    const uint32_t tap_count = static_cast<uint32_t>(config.num_target_layer_ids);
    const uint32_t block_rows = static_cast<uint32_t>(config.block_size);
    const uint32_t num_drafts = block_rows - 1u;

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    for (uint32_t t = 0; t < tap_count && t < taps.size(); ++t) {
        taps[t] = static_cast<uint32_t>(config.target_layer_ids[t]);
    }

    auto arena_result = ps::gpu::GpuArena::create(0, 24ull * 1024ull * 1024ull * 1024ull);
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

    auto model_result =
        ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(model_result.ok(), "target model load");
    if (!model_result.ok()) return 1;
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();

    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "dflash weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    const uint32_t max_seq_len = 2116u;
    const uint32_t max_batch = 256u;
    const uint32_t page_tokens = 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t num_pages = ceil_div(max_seq_len + 16u, page_tokens) + 8u;

    auto seq_pool_result = ps::qwen35::SequenceSlotPool::create(arena, 2u, num_pages + 8u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 2u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(), "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) return 1;
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig ec;
    ec.max_scheduled_tokens = max_batch;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = block_rows;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, ec);
    check(exec_result.ok(), "target executor create");
    if (!exec_result.ok()) return 1;
    ps::qwen35::Executor ex = exec_result.release();

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

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);

    std::size_t begin = 0;
    while (begin <= cases.size()) {
        const std::size_t comma = cases.find(',', begin);
        const std::string name = cases.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (name.empty()) {
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        const fs::path case_dir = prompt_root / name;
        std::vector<int32_t> prompt;
        if (!fs::is_directory(case_dir) || !read_prompt(case_dir / "prompt.txt", prompt)) {
            std::printf("SKIP case %s (no prompt)\n", name.c_str());
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        const uint32_t prompt_count = static_cast<uint32_t>(prompt.size());
        if (prompt_count > max_batch) {
            std::printf("SKIP case %s (prompt %u > %u)\n", name.c_str(), prompt_count, max_batch);
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }

        std::printf("=== %s (prompt=%u block=%u drafts=%u)\n", name.c_str(), prompt_count,
                    block_rows, num_drafts);

        auto state_result = ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
        check(state_result.ok(), name + " context create");
        if (!state_result.ok()) {
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        DFlash2ContextState state = state_result.release();

        auto seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, max_seq_len + 16u, stream);
        check(seq_result.ok(), name + " sequence create");
        if (!seq_result.ok()) {
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        ps::qwen35::PagedSequenceState sequence = seq_result.release();

        ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_cfg;
        decoder_cfg.num_drafts = num_drafts;
        decoder_cfg.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
        auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
            ex, draft, state, sequence, gdn_pool, arena, decoder_cfg, stream);
        check(decoder_result.ok(), name + " decoder create");
        if (!decoder_result.ok()) {
            (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        ps::qwen35::runtime::DFlash2SpecDecoder decoder = decoder_result.release();

        auto prefill_result = ps::qwen35::runtime::dflash2_spec_prefill(
            decoder, prompt.data(), prompt_count);
        check(prefill_result.ok(), name + " prefill: " +
                                       (prefill_result.ok() ? "" : prefill_result.status().message()));
        if (!prefill_result.ok()) {
            (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
            (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        const int32_t anchor = prefill_result.value().pending_token;

        check(sequence.position == prompt_count, name + " sequence position after prefill");
        check(state.next_position == prompt_count, name + " context next after prefill");
        check(state.length == std::min<uint32_t>(prompt_count,
                                                 static_cast<uint32_t>(config.sliding_window)),
              name + " context length after prefill");
        check(decoder.hidden_size == hidden, name + " decoder hidden");

        std::vector<int32_t> cached_drafts(num_drafts, 0);
        ps::Status prop = ps::qwen35::dflash2::dflash2_propose_cached(
            draft, state, anchor, num_drafts, draft.proposal_tokens.data<int32_t>(), stream);
        check(prop.ok(), name + " propose cached: " + prop.message());
        check(hipStreamSynchronize(stream) == hipSuccess, name + " propose sync");
        check(hipMemcpy(cached_drafts.data(), draft.proposal_tokens.data<int32_t>(),
                        cached_drafts.size() * sizeof(int32_t), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              name + " propose read");
        std::printf("  anchor=%d live drafts:", anchor);
        for (int32_t t : cached_drafts) std::printf(" %d", t);
        std::printf("\n");

        check(sequence.position == prompt_count, name + " proposal mutated sequence position");
        check(state.next_position == prompt_count, name + " proposal mutated next_position");
        check(state.length == std::min<uint32_t>(prompt_count,
                                                 static_cast<uint32_t>(config.sliding_window)),
              name + " proposal mutated length");

        {
            auto oracle_seq_result = ps::qwen35::create_paged_sequence_state(
                seq_pool, gdn_pool, kv_pool, max_seq_len + 16u, stream);
            check(oracle_seq_result.ok(), name + " oracle sequence create");
            if (oracle_seq_result.ok()) {
                ps::qwen35::PagedSequenceState oracle_seq = oracle_seq_result.release();
                std::vector<ps::qwen35::ScheduledRequest> requests;
                ps::qwen35::ScheduledRequest req;
                req.sequence = &oracle_seq;
                req.handle = oracle_seq.request_handle();
                req.execution_class = ps::runtime::ExecutionClass::PREFILL;
                req.token_begin = 0u;
                req.num_tokens = prompt_count;
                req.prefix_tokens = 0u;
                req.compute_logits = true;
                req.sample = true;
                req.num_output_rows = 1u;
                requests.push_back(req);
                ps::qwen35::ScheduledBatch batch;
                batch.token_ids = prompt.data();
                batch.requests = requests;
                batch.num_tokens = prompt_count;
                batch.num_requests = 1u;
                batch.num_decode_requests = 0u;
                batch.num_verify_requests = 0u;
                batch.num_prefill_requests = 1u;
                batch.speculative_verify = false;
                batch.verify_numeric_mode = ps::runtime::VerifyNumericMode::Exact;
                auto executed = ps::qwen35::execute_batch(ex, batch, stream);
                check(executed.ok(), name + " oracle target run");
                if (executed.ok()) {
                    ps::qwen35::BatchExecutionOutput out = executed.release();
                    int32_t oracle_anchor = -1;
                    check(hipMemcpy(&oracle_anchor, out.sampled_tokens.data<int32_t>(),
                                    sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess,
                          name + " oracle anchor read");
                    check(oracle_anchor == anchor, name + " anchor matches target sample");

                    ps::gpu::Tensor full_feature;
                    check(alloc_bf16(arena, {prompt_count, hidden}, full_feature),
                          name + " full feature alloc");
                    bool projected = true;
                    for (uint32_t offset = 0; offset < prompt_count; offset += draft.max_rows) {
                        const uint32_t n = std::min(draft.max_rows, prompt_count - offset);
                        std::array<const bf16_t*, ps::qwen35::kMaxTargetHiddenTaps> tap_ptrs{};
                        for (uint32_t t = 0; t < tap_count; ++t) {
                            tap_ptrs[t] = ex.dflash_target_hidden[t].data<bf16_t>() +
                                          static_cast<std::size_t>(offset) * hidden;
                        }
                        ps::Status st = ps::qwen35::dflash2::dflash2_project_target_features(
                            draft, tap_ptrs.data(), tap_count, n, stream);
                        if (!st.ok()) {
                            check(false, name + " oracle project: " + st.message());
                            projected = false;
                            break;
                        }
                        const hipError_t err = hipMemcpy(
                            full_feature.data<bf16_t>() + static_cast<std::size_t>(offset) * hidden,
                            draft.target_feature.data<bf16_t>(),
                            static_cast<std::size_t>(n) * hidden * sizeof(bf16_t),
                            hipMemcpyDeviceToDevice);
                        if (err != hipSuccess) {
                            check(false, name + " oracle feature copy");
                            projected = false;
                            break;
                        }
                    }
                    if (projected) {
                        ps::Status st = ps::qwen35::dflash2::dflash2_forward_and_select_stateless(
                            draft, draft.noise_embedding.data<bf16_t>(),
                            full_feature.data<bf16_t>(), prompt_count, 0u, prompt_count,
                            block_rows, anchor, draft.proposal_tokens.data<int32_t>(), stream);
                        check(st.ok(), name + " oracle stateless propose: " + st.message());
                        check(hipStreamSynchronize(stream) == hipSuccess,
                              name + " oracle propose sync");
                        std::vector<int32_t> oracle_drafts(num_drafts, 0);
                        check(hipMemcpy(oracle_drafts.data(), draft.proposal_tokens.data<int32_t>(),
                                        oracle_drafts.size() * sizeof(int32_t),
                                        hipMemcpyDeviceToHost) == hipSuccess,
                              name + " oracle propose read");
                        const bool exact = oracle_drafts == cached_drafts;
                        std::printf("  oracle(stateless) drafts:");
                        for (int32_t t : oracle_drafts) std::printf(" %d", t);
                        std::printf("  exact=%s\n", exact ? "yes" : "no");
                        check(exact, name + " live proposal matches stateless oracle");
                    }
                }
                (void)ps::qwen35::release_paged_sequence_state(oracle_seq, stream);
            }
        }

        if (!dump_root.empty()) {
            const fs::path case_dump = dump_root / name;
            std::error_code ec;
            fs::create_directories(case_dump, ec);
            bool dumped = true;
            std::vector<bf16_t> host_tap(static_cast<std::size_t>(prompt_count) * hidden);
            for (uint32_t t = 0; t < tap_count; ++t) {
                const auto& tap = ex.dflash_target_hidden[t];
                check(hipMemcpy(host_tap.data(), tap.data<bf16_t>(),
                                host_tap.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) ==
                          hipSuccess,
                      name + " dump tap read");
                dumped &= dump_blob(case_dump, "tap" + std::to_string(t) + ".bin",
                                    host_tap.data(), host_tap.size() * sizeof(bf16_t));
            }
            std::vector<bf16_t> host_noise(static_cast<std::size_t>(block_rows) * hidden);
            check(hipMemcpy(host_noise.data(), draft.noise_embedding.data<bf16_t>(),
                            host_noise.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  name + " dump noise read");
            dumped &= dump_blob(case_dump, "noise_embedding.bin", host_noise.data(),
                                host_noise.size() * sizeof(bf16_t));
            std::FILE* fp = std::fopen((case_dump / "metadata.json").string().c_str(), "w");
            if (fp != nullptr) {
                std::fprintf(fp, "{\n  \"prompt_count\": %u,\n  \"block_rows\": %u,\n",
                             prompt_count, block_rows);
                std::fprintf(fp, "  \"hidden_size\": %u,\n  \"tap_count\": %u,\n", hidden,
                             tap_count);
                std::fprintf(fp, "  \"anchor_token_id\": %d,\n", anchor);
                std::fprintf(fp, "  \"sliding_window\": %u,\n",
                             static_cast<uint32_t>(config.sliding_window));
                std::fprintf(fp, "  \"target_layer_ids\": [");
                for (uint32_t t = 0; t < tap_count; ++t) {
                    std::fprintf(fp, "%s%u", t == 0u ? "" : ", ", taps[t]);
                }
                std::fprintf(fp, "]\n}\n");
                std::fclose(fp);
            }
            check(dumped, name + " dump");
        }

        std::vector<int32_t> ref_tokens;
        if (read_reference_tokens(ref_root / name, ref_tokens)) {
            const bool ref_exact = ref_tokens == cached_drafts;
            std::printf("  official reference drafts:");
            for (int32_t t : ref_tokens) std::printf(" %d", t);
            std::printf("  exact=%s\n", ref_exact ? "yes" : "no");
            if (ref_exact) {
                std::printf("  live proposal exact vs official\n");
            } else {
                std::printf("  live proposal differs from official (Gate 5.1/7 numeric "
                            "contract)\n");
            }
        }

        (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
        (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
        (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }

    {
        const uint32_t long_count = 2100u;
        std::vector<int32_t> long_prompt(long_count);
        for (uint32_t i = 0; i < long_count; ++i) {
            long_prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
        }
        std::printf("=== long_prompt (prompt=%u block=%u)\n", long_count, block_rows);

        auto state_result =
            ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
        auto seq_result = ps::qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, max_seq_len + 16u, stream);
        check(state_result.ok() && seq_result.ok(), "long context/sequence create");
        if (state_result.ok() && seq_result.ok()) {
            DFlash2ContextState state = state_result.release();
            ps::qwen35::PagedSequenceState sequence = seq_result.release();
            ps::qwen35::runtime::DFlash2SpecDecoderConfig decoder_cfg;
            decoder_cfg.num_drafts = num_drafts;
            auto decoder_result = ps::qwen35::runtime::create_dflash2_spec_decoder(
                ex, draft, state, sequence, gdn_pool, arena, decoder_cfg, stream);
            check(decoder_result.ok(), "long decoder create");
            if (decoder_result.ok()) {
                ps::qwen35::runtime::DFlash2SpecDecoder decoder = decoder_result.release();
                auto prefill_result = ps::qwen35::runtime::dflash2_spec_prefill(
                    decoder, long_prompt.data(), long_count);
                check(prefill_result.ok(), "long prefill: " +
                                               (prefill_result.ok()
                                                    ? ""
                                                    : prefill_result.status().message()));
                if (prefill_result.ok()) {
                    check(sequence.position == long_count, "long sequence position");
                    check(state.length == static_cast<uint32_t>(config.sliding_window),
                          "long ring length clamped to capacity");
                    check(state.next_position == long_count, "long next_position");
                    const int32_t anchor = prefill_result.value().pending_token;
                    std::vector<int32_t> long_drafts(num_drafts, 0);
                    ps::Status prop = ps::qwen35::dflash2::dflash2_propose_cached(
                        draft, state, anchor, num_drafts,
                        draft.proposal_tokens.data<int32_t>(), stream);
                    check(prop.ok(), "long propose: " + prop.message());
                    check(hipStreamSynchronize(stream) == hipSuccess, "long propose sync");
                    check(hipMemcpy(long_drafts.data(), draft.proposal_tokens.data<int32_t>(),
                                    long_drafts.size() * sizeof(int32_t),
                                    hipMemcpyDeviceToHost) == hipSuccess,
                          "long propose read");
                    std::printf("  long anchor=%d drafts:", anchor);
                    for (int32_t t : long_drafts) std::printf(" %d", t);
                    std::printf("\n");
                    check(state.next_position == long_count && state.length ==
                              static_cast<uint32_t>(config.sliding_window),
                          "long proposal mutated context");
                }
                (void)ps::qwen35::runtime::dflash2_spec_decoder_shutdown(decoder);
            }
            (void)ps::qwen35::release_paged_sequence_state(sequence, stream);
            (void)ps::qwen35::dflash2::dflash2_context_shutdown(state);
        }
    }

    check(ps::qwen35::dflash2::dflash2_executor_shutdown(draft).ok(), "draft shutdown");
    check(ps::qwen35::executor_shutdown(ex).ok(), "target shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate8_live: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
