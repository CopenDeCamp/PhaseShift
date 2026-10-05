#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

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

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

static ps::qwen35::ScheduledBatch make_batch(
    ps::qwen35::PagedSequenceState& sequence,
    std::vector<ps::qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens) {
    ps::qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = ps::runtime::ExecutionClass::PREFILL;
    req.token_begin = 0u;
    req.num_tokens = num_tokens;
    req.prefix_tokens = 0u;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = 1u;
    requests.clear();
    requests.push_back(req);
    ps::qwen35::ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests = 0u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = 1u;
    return batch;
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

static bool dump_bf16(const fs::path& dir, const std::string& name, const bf16_t* data,
                      std::size_t count) {
    std::FILE* fp = std::fopen((dir / name).string().c_str(), "wb");
    if (fp == nullptr) return false;
    const std::size_t written = std::fwrite(data, sizeof(bf16_t), count, fp);
    std::fclose(fp);
    return written == count;
}

int main() {
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (target_dir == nullptr || target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2_TARGET\n");
        return 77;
    }
    const char* prompt_env = std::getenv("PHASESHIFT_DFLASH2_GATE51_PROMPT_FILE");
    if (prompt_env == nullptr || prompt_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_DFLASH2_GATE51_PROMPT_FILE\n");
        return 77;
    }
    const char* out_env = std::getenv("PHASESHIFT_DFLASH2_GATE51_OUT");
    const fs::path out_dir = (out_env != nullptr && out_env[0] != '\0')
                                 ? fs::path(out_env)
                                 : fs::path("build/dflash2-gate51/real");
    fs::create_directories(out_dir);
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    std::array<uint32_t, ps::qwen35::kMaxTargetHiddenTaps> taps{};
    uint32_t tap_count = 0;
    {
        const char* dflash_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
        if (dflash_dir != nullptr && dflash_dir[0] != '\0') {
            auto cfg = ps::qwen35::dflash2::read_dflash2_config(dflash_dir);
            if (cfg.ok() && cfg.value().num_target_layer_ids <= taps.size()) {
                for (std::size_t i = 0; i < cfg.value().num_target_layer_ids; ++i) {
                    taps[i] = static_cast<uint32_t>(cfg.value().target_layer_ids[i]);
                }
                tap_count = static_cast<uint32_t>(cfg.value().num_target_layer_ids);
            }
        }
        if (tap_count == 0u) {
            taps = {5u, 19u, 33u, 47u, 61u};
            tap_count = 5u;
        }
    }

    std::vector<int32_t> prompt;
    if (!read_prompt(prompt_env, prompt)) {
        std::printf("FAIL: cannot read prompt\n");
        return 1;
    }
    const uint32_t num_tokens = static_cast<uint32_t>(prompt.size());
    if (num_tokens < 8u) {
        std::printf("FAIL: prompt too short (%u)\n", num_tokens);
        return 1;
    }

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = target_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 30ull * 1024ull * 1024ull * 1024ull;
    auto resident_result = ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    resident_result.status().message().c_str());
        return 1;
    }
    ps::resident::ResidentModelFixture resident = resident_result.release();

    auto arena_result = ps::gpu::GpuArena::create(0, resident.arena_bytes());
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

    auto model_result = resident.enabled()
        ? resident.take_qwen35()
        : ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(model_result.ok(), "model load");
    if (!model_result.ok()) return 1;
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();
    const uint32_t hidden = static_cast<uint32_t>(tc.hidden_size);

    const uint32_t max_tokens = num_tokens + 16u;
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t num_pages = ceil_div(max_tokens + 16u, page_tokens) + 8u;

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
    ec.max_scheduled_tokens = max_tokens;
    ec.max_scheduled_requests = 1u;
    ec.max_scheduled_output_rows = 1u;
    ec.target_hidden_taps = taps;
    ec.target_hidden_tap_count = tap_count;
    auto exec_result = ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool,
                                                         arena, ec);
    check(exec_result.ok(), "create executor");
    if (!exec_result.ok()) return 1;
    ps::qwen35::Executor ex = exec_result.release();

    for (uint32_t t = 0; t < tap_count; ++t) {
        const auto& tap = ex.dflash_target_hidden[t];
        check(tap.ndim() == 2u && tap.dim(0) >= num_tokens && tap.dim(1) == hidden,
              "tap buffer shape");
    }

    auto seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool, max_tokens + 16u, stream);
    check(seq_result.ok(), "sequence create");
    if (!seq_result.ok()) return 1;
    ps::qwen35::PagedSequenceState seq = seq_result.release();

    std::vector<ps::qwen35::ScheduledRequest> requests;
    ps::qwen35::ScheduledBatch batch = make_batch(seq, requests, prompt.data(), num_tokens);
    auto executed = ps::qwen35::execute_batch(ex, batch, stream);
    check(executed.ok(), "execute");
    if (!executed.ok()) {
        std::printf("test_dflash2_gate51_target_fixture: failed=%d\n", g_fail);
        return 1;
    }
    ps::qwen35::BatchExecutionOutput out = executed.release();
    check(hipStreamSynchronize(stream) == hipSuccess, "sync");

    const uint32_t context_rows = 8u;
    const uint32_t context_start = num_tokens - context_rows;
    std::vector<bf16_t> host(static_cast<std::size_t>(context_rows) * hidden);
    for (uint32_t t = 0; t < tap_count; ++t) {
        const auto& tap = ex.dflash_target_hidden[t];
        const bf16_t* base = tap.data<bf16_t>();
        const uint32_t row_stride = static_cast<uint32_t>(tap.dim(1));
        check(hipMemcpy(host.data(),
                        base + static_cast<std::size_t>(context_start) * row_stride,
                        host.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
              "tap copy");
        check(dump_bf16(out_dir, "tap" + std::to_string(t) + ".bin", host.data(), host.size()),
              "tap dump");
    }

    {
        std::FILE* fp = std::fopen((out_dir / "target_meta.txt").string().c_str(), "w");
        if (fp != nullptr) {
            std::fprintf(fp, "{\n  \"context_rows\": %u,\n  \"block_rows\": 8,\n",
                         context_rows);
            std::fprintf(fp, "  \"context_start\": %u,\n  \"block_start\": %u,\n",
                         context_start, num_tokens);
            std::fprintf(fp, "  \"anchor_token_id\": %d,\n  \"num_tokens\": %u,\n",
                         prompt[num_tokens - 1u], num_tokens);
            std::fprintf(fp, "  \"hidden_size\": %u,\n  \"tap_count\": %u,\n",
                         hidden, tap_count);
            std::fprintf(fp, "  \"target_layer_ids\": [");
            for (uint32_t t = 0; t < tap_count; ++t) {
                std::fprintf(fp, "%s%u", t == 0u ? "" : ", ", taps[t]);
            }
            std::fprintf(fp, "]\n}\n");
            std::fclose(fp);
        }
    }

    (void)ps::qwen35::release_paged_sequence_state(seq, stream);
    const ps::Status st = ps::qwen35::executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");

    std::printf("test_dflash2_gate51_target_fixture: failed=%d tokens=%u out=%s\n", g_fail,
                num_tokens, out_dir.string().c_str());
    return g_fail == 0 ? 0 : 1;
}
