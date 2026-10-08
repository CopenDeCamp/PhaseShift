#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/mcu_decode_state.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/runtime/gpu_mcu/execution/micro_fsm.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

namespace mcu = ::ps::runtime::gpu_mcu;

int g_fail = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++g_fail;
        std::printf("FAIL %s\n", message.c_str());
    }
}

bool directory_exists(const char* path) {
    struct stat info;
    return path != nullptr && stat(path, &info) == 0 && S_ISDIR(info.st_mode);
}

const char* model_dir() {
    const char* from_env = std::getenv("PHASESHIFT_MODEL_DIR");
    return (from_env != nullptr && from_env[0] != '\0') ? from_env : "";
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

ps::qwen35::ScheduledBatch make_decode_batch(
    ps::qwen35::PagedSequenceState& sequence,
    std::vector<ps::qwen35::ScheduledRequest>& requests,
    const int32_t* token,
    uint32_t prefix_tokens) {
    ps::qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = ps::runtime::ExecutionClass::DECODE;
    req.token_begin = 0u;
    req.num_tokens = 1u;
    req.prefix_tokens = prefix_tokens;
    req.compute_logits = true;
    req.sample = true;
    req.num_output_rows = 1u;
    requests.clear();
    requests.push_back(req);
    ps::qwen35::ScheduledBatch batch;
    batch.token_ids = token;
    batch.requests = requests;
    batch.num_tokens = 1u;
    batch.num_requests = 1u;
    batch.num_decode_requests = 1u;
    batch.num_verify_requests = 0u;
    batch.num_prefill_requests = 0u;
    return batch;
}

ps::qwen35::ScheduledBatch make_prefill_batch(
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

bool bytes_equal(const void* a, const void* b, std::size_t bytes,
                 std::size_t* first, std::size_t* diffs) {
    const auto* x = static_cast<const unsigned char*>(a);
    const auto* y = static_cast<const unsigned char*>(b);
    *first = 0u;
    *diffs = 0u;
    for (std::size_t i = 0; i < bytes; ++i) {
        if (x[i] != y[i]) {
            if (*diffs == 0u) *first = i;
            ++(*diffs);
        }
    }
    return *diffs == 0u;
}

struct DecodeResult {
    std::vector<unsigned char> hidden;
    uint32_t rows = 0u;
    uint32_t row_bytes = 0u;
    std::vector<int32_t> sampled;
};

bool run_prefill_then_decode(
    ps::qwen35::Executor& executor,
    ps::qwen35::SequenceSlotPool& seq_pool,
    ps::qwen35::GdnStatePool& gdn_pool,
    ps::qwen35::PagedKVPool& kv_pool,
    hipStream_t stream,
    const std::vector<int32_t>& prompt,
    int32_t decode_token,
    ps::qwen35::runtime::DecodeBackend backend,
    const char* label,
    DecodeResult* out) {
    executor.config.backend = backend;

    auto seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool,
        static_cast<uint32_t>(prompt.size()) + 32u, stream);
    check(seq_result.ok(), std::string(label) + ": sequence create");
    if (!seq_result.ok()) return false;
    ps::qwen35::PagedSequenceState seq = seq_result.release();

    std::vector<ps::qwen35::ScheduledRequest> requests;

    {
        ps::qwen35::ScheduledBatch batch = make_prefill_batch(
            seq, requests, prompt.data(), static_cast<uint32_t>(prompt.size()));
        auto executed = ps::qwen35::execute_batch(executor, batch, stream);
        check(executed.ok(), std::string(label) + ": prefill " +
                                 (executed.ok() ? ""
                                                : executed.status().message()));
        if (!executed.ok()) return false;
        (void)executed.release();
    }

    executor.config.backend = backend;
    {
        ps::qwen35::ScheduledBatch batch = make_decode_batch(
            seq, requests, &decode_token, static_cast<uint32_t>(prompt.size()));
        auto executed = ps::qwen35::execute_batch(executor, batch, stream);
        check(executed.ok(), std::string(label) + ": decode " +
                                 (executed.ok() ? ""
                                                : executed.status().message()));
        if (!executed.ok()) return false;
        ps::qwen35::BatchExecutionOutput output = executed.release();
        const auto& hidden = output.final_hidden;
        check(hidden.ndim() >= 1u, std::string(label) + ": final_hidden dim");
        out->rows = hidden.ndim() >= 1u ? static_cast<uint32_t>(hidden.dim(0)) : 0u;
        out->row_bytes = static_cast<uint32_t>(
            hidden.ndim() >= 2u ? hidden.dim(1) * hidden.element_size()
                                : hidden.element_size());
        const std::size_t bytes = static_cast<std::size_t>(out->rows) *
                                  out->row_bytes;
        out->hidden.resize(bytes);
        if (bytes != 0u) {
            const hipError_t err = hipMemcpy(out->hidden.data(),
                                             hidden.data<unsigned char>(), bytes,
                                             hipMemcpyDeviceToHost);
            check(err == hipSuccess, std::string(label) + ": hidden readback");
            if (err != hipSuccess) return false;
        }
        const uint32_t num_out = output.num_outputs;
        out->sampled.resize(num_out, -1);
        if (num_out != 0u) {
            const hipError_t serr = hipMemcpy(
                out->sampled.data(), output.sampled_tokens.data<int32_t>(),
                static_cast<std::size_t>(num_out) * sizeof(int32_t),
                hipMemcpyDeviceToHost);
            check(serr == hipSuccess, std::string(label) + ": sampled readback");
            if (serr != hipSuccess) return false;
        }
    }
    return true;
}


bool run_mixed_decode_prefill(
    ps::qwen35::Executor& executor,
    ps::qwen35::SequenceSlotPool& seq_pool,
    ps::qwen35::GdnStatePool& gdn_pool,
    ps::qwen35::PagedKVPool& kv_pool,
    hipStream_t stream,
    const std::vector<int32_t>& prompt,
    int32_t decode_token,
    ps::qwen35::runtime::DecodeBackend backend,
    const char* label,
    DecodeResult* out) {
    executor.config.backend = backend;

    auto decode_seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool,
        static_cast<uint32_t>(prompt.size()) + 32u, stream);
    check(decode_seq_result.ok(), std::string(label) + ": decode sequence create");
    if (!decode_seq_result.ok()) return false;
    ps::qwen35::PagedSequenceState decode_seq = decode_seq_result.release();

    std::vector<ps::qwen35::ScheduledRequest> requests;
    {
        ps::qwen35::ScheduledBatch batch = make_prefill_batch(
            decode_seq, requests, prompt.data(),
            static_cast<uint32_t>(prompt.size()));
        auto executed = ps::qwen35::execute_batch(executor, batch, stream);
        check(executed.ok(), std::string(label) + ": decode sequence prefill " +
                                 (executed.ok() ? ""
                                                : executed.status().message()));
        if (!executed.ok()) return false;
        (void)executed.release();
    }

    auto prefill_seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool,
        static_cast<uint32_t>(prompt.size()) + 32u, stream);
    check(prefill_seq_result.ok(), std::string(label) + ": prefill sequence create");
    if (!prefill_seq_result.ok()) return false;
    ps::qwen35::PagedSequenceState prefill_seq = prefill_seq_result.release();

    executor.config.backend = backend;
    {
        requests.clear();
        ps::qwen35::ScheduledRequest decode_req;
        decode_req.sequence = &decode_seq;
        decode_req.handle = decode_seq.request_handle();
        decode_req.execution_class = ps::runtime::ExecutionClass::DECODE;
        decode_req.token_begin = 0u;
        decode_req.num_tokens = 1u;
        decode_req.prefix_tokens = static_cast<uint32_t>(prompt.size());
        decode_req.compute_logits = true;
        decode_req.sample = true;
        decode_req.num_output_rows = 1u;
        requests.push_back(decode_req);

        ps::qwen35::ScheduledRequest prefill_req;
        prefill_req.sequence = &prefill_seq;
        prefill_req.handle = prefill_seq.request_handle();
        prefill_req.execution_class = ps::runtime::ExecutionClass::PREFILL;
        prefill_req.token_begin = 1u;
        prefill_req.num_tokens = static_cast<uint32_t>(prompt.size());
        prefill_req.prefix_tokens = 0u;
        prefill_req.compute_logits = true;
        prefill_req.sample = true;
        prefill_req.num_output_rows = 1u;
        requests.push_back(prefill_req);

        std::vector<int32_t> tokens(1u + prompt.size(), 0);
        tokens[0] = decode_token;
        for (std::size_t i = 0; i < prompt.size(); ++i) {
            tokens[1u + i] = prompt[i];
        }
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = tokens.data();
        batch.requests = requests;
        batch.num_tokens = static_cast<uint32_t>(tokens.size());
        batch.num_requests = 2u;
        batch.num_decode_requests = 1u;
        batch.num_verify_requests = 0u;
        batch.num_prefill_requests = 1u;
        auto executed = ps::qwen35::execute_batch(executor, batch, stream);
        check(executed.ok(), std::string(label) + ": mixed batch " +
                                 (executed.ok() ? ""
                                                : executed.status().message()));
        if (!executed.ok()) return false;
        ps::qwen35::BatchExecutionOutput output = executed.release();
        const auto& hidden = output.final_hidden;
        out->rows = hidden.ndim() >= 1u ? static_cast<uint32_t>(hidden.dim(0)) : 0u;
        out->row_bytes = static_cast<uint32_t>(
            hidden.ndim() >= 2u ? hidden.dim(1) * hidden.element_size()
                                : hidden.element_size());
        const std::size_t bytes =
            static_cast<std::size_t>(out->rows) * out->row_bytes;
        out->hidden.resize(bytes);
        if (bytes != 0u) {
            const hipError_t err = hipMemcpy(out->hidden.data(),
                                             hidden.data<unsigned char>(), bytes,
                                             hipMemcpyDeviceToHost);
            check(err == hipSuccess, std::string(label) + ": mixed hidden readback");
            if (err != hipSuccess) return false;
        }
    }
    return true;
}


bool probe_mcu_decode(
    ps::qwen35::Executor& executor,
    ps::qwen35::SequenceSlotPool& seq_pool,
    ps::qwen35::GdnStatePool& gdn_pool,
    ps::qwen35::PagedKVPool& kv_pool,
    hipStream_t stream,
    const std::vector<int32_t>& prompt,
    int32_t decode_token) {
    executor.config.backend = ps::qwen35::runtime::DecodeBackend::Host;
    auto seq_result = ps::qwen35::create_paged_sequence_state(
        seq_pool, gdn_pool, kv_pool,
        static_cast<uint32_t>(prompt.size()) + 32u, stream);
    check(seq_result.ok(), "probe: sequence create");
    if (!seq_result.ok()) return false;
    ps::qwen35::PagedSequenceState seq = seq_result.release();
    std::vector<ps::qwen35::ScheduledRequest> requests;
    {
        ps::qwen35::ScheduledBatch batch = make_prefill_batch(
            seq, requests, prompt.data(), static_cast<uint32_t>(prompt.size()));
        auto executed = ps::qwen35::execute_batch(executor, batch, stream);
        check(executed.ok(), "probe: prefill");
        if (!executed.ok()) return false;
        (void)executed.release();
    }

    executor.config.backend = ps::qwen35::runtime::DecodeBackend::GpuMcu;
    ps::qwen35::ScheduledBatch batch = make_decode_batch(
        seq, requests, &decode_token, static_cast<uint32_t>(prompt.size()));
    auto submitted = ps::qwen35::submit_batch(executor, batch, stream);
    check(submitted.ok(), "probe: submit_batch " +
                              (submitted.ok() ? ""
                                              : submitted.status().message()));
    if (!submitted.ok()) return false;
    ps::qwen35::PendingBatch pending = submitted.release();

    std::printf("probe: submitted; query=%d\n",
                static_cast<int>(hipStreamQuery(stream)));
    const uint64_t generation = pending.generation;
    if (generation == 0u) {
        std::printf("probe: FAIL generation is zero\n");
        return true;
    }
    const auto before =
        ps::qwen35::poll_decode(executor, generation);
    const auto stale = ps::qwen35::poll_decode(executor, generation + 1000000u);
    std::printf("probe: poll-before=%d stale=%d\n",
                static_cast<int>(before), static_cast<int>(stale));
    std::printf("probe: stream-sync start\n");
    {
        const hipError_t se = hipStreamSynchronize(stream);
        std::printf("probe: stream-sync rc=%d query=%d\n",
                    static_cast<int>(se),
                    static_cast<int>(hipStreamQuery(stream)));
    }
    std::printf("probe: poll-after=%d\n",
                static_cast<int>(ps::qwen35::poll_decode(executor, generation)));
    if (executor.mcu_state != nullptr) {
        const mcu::GpuMcuFsmState* st = executor.mcu_state->runtime.fsm().host_state();
        if (st != nullptr) {
            std::printf(
                "probe: after-sync epoch_seen=%u plans=%llu disp=%llu comp=%llu "
                "fault=%u result=%u\n",
                st->external_epoch_seen,
                static_cast<unsigned long long>(st->plans_started),
                static_cast<unsigned long long>(st->dispatches_committed),
                static_cast<unsigned long long>(st->completions_observed),
                st->fault_code, executor.mcu_state->runtime.result_code());
        }
    }
    {
        std::printf("probe: complete_batch start\n");
        auto completed =
            ps::qwen35::complete_batch(executor, std::move(pending), stream);
        std::printf("probe: complete_batch ok=%d\n",
                    static_cast<int>(completed.ok()));
    }
    return true;

    bool stream_done = false;
    for (int i = 0; i < 30; ++i) {
        struct timespec ts {};
        ts.tv_sec = 0;
        ts.tv_nsec = 500000000L;
        nanosleep(&ts, nullptr);
        const mcu::GpuMcuFsmState* state = nullptr;
        if (executor.mcu_state != nullptr) {
            state = executor.mcu_state->runtime.fsm().host_state();
        }
        const hipError_t query = hipStreamQuery(stream);
        const uint32_t result =
            executor.mcu_state != nullptr ? executor.mcu_state->runtime.result_code()
                                          : 0u;
        if (state == nullptr) {
            std::printf("probe t=%d stream=%d state=null result=%u\n", i,
                        static_cast<int>(query), result);
        } else {
            std::printf(
                "probe t=%d stream=%d epoch_seen=%u run_request=%llu started=%u "
                "plans=%llu disp=%llu comp=%llu iter=%llu resp=%u fault=%u "
                "pc=%u variant=%u q_empty=%llu backpressure=%llu doorbell=%llu "
                "max_ahead=%llu append_ahead=%llu node_count=%u result=%u\n",
                i, static_cast<int>(query), state->external_epoch_seen,
                static_cast<unsigned long long>(state->run_request),
                state->started,
                static_cast<unsigned long long>(state->plans_started),
                static_cast<unsigned long long>(state->dispatches_committed),
                static_cast<unsigned long long>(state->completions_observed),
                static_cast<unsigned long long>(state->iterations),
                state->reserved_ctrl, state->fault_code, state->fault_pc,
                state->fault_variant,
                static_cast<unsigned long long>(state->queue_empty_count),
                static_cast<unsigned long long>(state->backpressure_wait_count),
                static_cast<unsigned long long>(state->doorbell_count),
                static_cast<unsigned long long>(state->max_ahead),
                static_cast<unsigned long long>(state->append_ahead_count),
                state->node_count, result);
        }
        if (query == hipSuccess) {
            std::printf("probe: stream completed after %d polls\n", i + 1);
            stream_done = true;
            break;
        }
    }
    if (stream_done) {
        std::printf("probe: calling complete_batch\n");
        auto completed =
            ps::qwen35::complete_batch(executor, std::move(pending), stream);
        std::printf("probe: complete_batch ok=%d %s\n",
                    static_cast<int>(completed.ok()),
                    completed.ok() ? "" : completed.status().message().c_str());
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const char* dir = model_dir();
    if (dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR to a Qwen3.5 model dir\n");
        return 77;
    }
    if (!directory_exists(dir)) {
        std::printf("SKIP: model dir not found: %s\n", dir);
        return 77;
    }

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 30ull * 1024ull * 1024ull * 1024ull;
    auto resident_result = ps::resident::ResidentModelFixture::acquire(resident_request);
    if (!resident_result.ok()) {
        std::printf("FAIL resident acquire: %s\n",
                    resident_result.status().message().c_str());
        return 77;
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
        : ps::qwen35::Qwen35Model::load_from_safetensors(dir, arena, stream);
    check(model_result.ok(), "model load: " +
                                 (model_result.ok()
                                      ? ""
                                      : model_result.status().message()));
    if (!model_result.ok()) {
        std::printf("test_gpu_mcu_production_decode: failed=%d\n", g_fail);
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::qwen35::Qwen35TextConfig& tc = model.text_config();

    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t page_tokens = 16u;
    const uint32_t prompt_tokens = 64u;
    const uint32_t num_pages =
        ceil_div(prompt_tokens + 64u, page_tokens) * 4u + 16u;

    auto seq_pool_result =
        ps::qwen35::SequenceSlotPool::create(arena, 8u, num_pages + 16u);
    auto gdn_pool_result = ps::qwen35::GdnStatePool::create(
        arena, 8u, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = ps::qwen35::PagedKVPool::create(
        arena, num_pages, page_tokens, num_attn_layers, kv_heads, head_dim,
        ps::qwen35::KVCacheDType::BF16);
    check(seq_pool_result.ok() && gdn_pool_result.ok() && kv_pool_result.ok(),
          "pool create");
    if (!seq_pool_result.ok() || !gdn_pool_result.ok() || !kv_pool_result.ok()) {
        std::printf("test_gpu_mcu_production_decode: failed=%d\n", g_fail);
        return 1;
    }
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_result.release();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_result.release();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig config;
    config.max_scheduled_tokens = 2u * prompt_tokens;
    config.max_scheduled_requests = 2u;
    config.max_scheduled_output_rows = 2u;
    config.backend = ps::qwen35::runtime::DecodeBackend::Host;

    auto executor_result = ps::qwen35::create_model_executor(
        model, seq_pool, gdn_pool, kv_pool, arena, config);
    check(executor_result.ok(),
          "create executor: " +
              (executor_result.ok() ? "" : executor_result.status().message()));
    if (!executor_result.ok()) {
        std::printf("test_gpu_mcu_production_decode: failed=%d\n", g_fail);
        return 1;
    }
    ps::qwen35::Executor executor = executor_result.release();

    bool range_valid = false;
    for (const auto& range : executor.mcu_body_range) {
        if (range.valid) range_valid = true;
    }
    check(range_valid, "mcu body range survives executor creation");

    std::vector<int32_t> prompt(prompt_tokens);
    for (uint32_t i = 0; i < prompt_tokens; ++i) {
        prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
    }
    const int32_t decode_token = 4242;

    DecodeResult host{};
    const bool host_ok = run_prefill_then_decode(
        executor, seq_pool, gdn_pool, kv_pool, stream, prompt, decode_token,
        ps::qwen35::runtime::DecodeBackend::Host, "host", &host);

    const char* probe_flag = std::getenv("PHASESHIFT_MCU_PROBE");
    if (probe_flag != nullptr && probe_flag[0] != '\0') {
        const bool probe_ok = probe_mcu_decode(executor, seq_pool, gdn_pool,
                                               kv_pool, stream, prompt,
                                               decode_token);
        (void)probe_ok;
        return 0;
    }

    const char* mcu_flag = std::getenv("PHASESHIFT_MCU_E2E");
    const bool mcu_enabled = mcu_flag != nullptr && mcu_flag[0] != '\0';
    DecodeResult mcu{};
    bool mcu_ok = false;
    if (!mcu_enabled) {
        std::printf(
            "production decode: mcu mixed decode is not run "
            "(set PHASESHIFT_MCU_E2E=1). The persistent MCU drives the "
            "production body plan; the host stream controller is no longer "
            "used.\n");
    } else {
        mcu_ok = run_prefill_then_decode(
            executor, seq_pool, gdn_pool, kv_pool, stream, prompt, decode_token,
            ps::qwen35::runtime::DecodeBackend::GpuMcu, "mcu", &mcu);
    }

    if (mcu_enabled) {
        check(executor.mcu_state != nullptr,
              "mcu backend created the decode runtime");
        if (executor.mcu_state != nullptr) {
            const auto& counters = executor.mcu_state->cache.counters();
            std::printf(
                "production decode: compiles=%llu uploads=%llu switches=%llu "
                "runs=%llu fallbacks=%llu\n",
                static_cast<unsigned long long>(counters.plan_compile_count),
                static_cast<unsigned long long>(counters.plan_upload_count),
                static_cast<unsigned long long>(counters.plan_switch_count),
                static_cast<unsigned long long>(counters.mcu_run_count),
                static_cast<unsigned long long>(counters.host_fallback_count));
            std::printf(
                "production decode: full_plan=%d plan=[%u,%u) of %u "
                "host_prefix=%u host_suffix=%u\n",
                executor.mcu_state->full_plan ? 1 : 0,
                executor.mcu_state->plan_begin, executor.mcu_state->plan_end,
                executor.mcu_state->plan_total, executor.mcu_state->plan_begin,
                executor.mcu_state->plan_total - executor.mcu_state->plan_end);
            check(counters.mcu_run_count >= 1u, "mcu hybrid path actually ran");
            check(counters.plan_upload_count >= 1u,
                  "mcu plan was uploaded to the fsm");
            check(counters.host_fallback_count == 0u,
                  "mcu batch did not fall back to the host backend");
            std::printf("production decode: static plans=%zu",
                        executor.mcu_state->static_plans.size());
            for (const auto& entry : executor.mcu_state->static_plans) {
                std::printf(" [slot=%u key=%u rows=%u]", entry.program_slot,
                            entry.attention_plan_key, entry.plan_end);
            }
            std::printf("\n");
            check(counters.plan_compile_count == 1u,
                  "one static plan compile for the whole session");
            check(counters.plan_compile_count ==
                      executor.mcu_state->static_plans.size(),
                  "no static plan is compiled twice");
            uint32_t plan_key = executor.mcu_state->static_plans.empty()
                                    ? 0u
                                    : executor.mcu_state->static_plans[0]
                                          .attention_plan_key;
            bool plan_key_uniform = true;
            for (const auto& entry : executor.mcu_state->static_plans) {
                if (entry.attention_plan_key != plan_key)
                    plan_key_uniform = false;
            }
            check(plan_key_uniform,
                  "static plan identity ignores the live batch");

            check(executor.mcu_state->persistent.batches_committed() >= 1u,
                  "mcu persistent execution committed the batch");
        }
    }

    if (mcu_enabled && host_ok && mcu_ok) {
        check(host.rows == mcu.rows && host.row_bytes == mcu.row_bytes,
              "host and mcu produce the same output shape");
        if (host.rows == mcu.rows && host.row_bytes == mcu.row_bytes &&
            !host.hidden.empty()) {
            std::size_t first = 0u;
            std::size_t diffs = 0u;
            const bool equal = bytes_equal(host.hidden.data(), mcu.hidden.data(),
                                           host.hidden.size(), &first, &diffs);
            if (!equal) {
                std::printf(
                    "production decode: final_hidden differs bytes=%zu first=%zu\n",
                    diffs, first);
            }
            check(equal, "mcu decode matches host decode byte for byte");
        }
        check(executor.mcu_state != nullptr && executor.mcu_state->full_plan,
              "mcu decode ran the whole program plan");
        check(!host.sampled.empty() && host.sampled == mcu.sampled,
              "mcu decode samples inside the mcu plan");
    }

    DecodeResult mixed_host{};
    const bool mixed_host_ok = run_mixed_decode_prefill(
        executor, seq_pool, gdn_pool, kv_pool, stream, prompt, decode_token,
        ps::qwen35::runtime::DecodeBackend::Host, "mixed-host", &mixed_host);
    DecodeResult mixed_mcu{};
    bool mixed_mcu_ok = false;
    if (mcu_enabled) {
        mixed_mcu_ok = run_mixed_decode_prefill(
            executor, seq_pool, gdn_pool, kv_pool, stream, prompt, decode_token,
            ps::qwen35::runtime::DecodeBackend::GpuMcu, "mixed-mcu",
            &mixed_mcu);
        if (executor.mcu_state != nullptr) {
            check(executor.mcu_state->persistent.batches_committed() >= 1u,
                  "mcu persistent execution committed the mixed batch");
        }
    }
    if (mcu_enabled && mixed_host_ok && mixed_mcu_ok) {
        check(mixed_host.rows == mixed_mcu.rows &&
                  mixed_host.row_bytes == mixed_mcu.row_bytes,
              "mixed host and mcu produce the same output shape");
        if (mixed_host.rows == mixed_mcu.rows &&
            mixed_host.row_bytes == mixed_mcu.row_bytes &&
            !mixed_host.hidden.empty()) {
            std::size_t first = 0u;
            std::size_t diffs = 0u;
            const bool equal = bytes_equal(
                mixed_host.hidden.data(), mixed_mcu.hidden.data(),
                mixed_host.hidden.size(), &first, &diffs);
            if (!equal) {
                std::printf(
                    "production decode: mixed final_hidden differs bytes=%zu "
                    "first=%zu\n",
                    diffs, first);
            }
            check(equal,
                  "mcu mixed batch matches host mixed batch byte for byte");
        }
    }

    if (mcu_enabled && executor.mcu_state != nullptr) {
        const auto& late = executor.mcu_state->cache.counters();
        std::printf(
            "production decode: after mixed compiles=%llu uploads=%llu\n",
            static_cast<unsigned long long>(late.plan_compile_count),
            static_cast<unsigned long long>(late.plan_upload_count));
        check(late.plan_compile_count ==
                  executor.mcu_state->static_plans.size(),
              "mixed batch never compiles a static plan twice");
        uint32_t late_key = executor.mcu_state->static_plans.empty()
                                ? 0u
                                : executor.mcu_state->static_plans[0]
                                      .attention_plan_key;
        bool late_uniform = true;
        for (const auto& entry : executor.mcu_state->static_plans) {
            if (entry.attention_plan_key != late_key) late_uniform = false;
        }
        check(late_uniform,
              "mixed batch keeps one static plan identity");
        check(late.plan_compile_count == 1u,
              "prefill/decode/mixed share one static plan");
    }

    const auto shutdown_status = ps::qwen35::executor_shutdown(executor);
    check(shutdown_status.ok(), "executor shutdown");
    (void)ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    if (g_fail != 0) {
        std::printf("test_gpu_mcu_production_decode: failed=%d\n", g_fail);
        return 1;
    }
    std::printf("test_gpu_mcu_production_decode: PASS\n");
    return 0;
}
