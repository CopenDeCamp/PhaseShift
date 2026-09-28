#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/runtime/graph/primitive_node.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace qwen35 = ps::qwen35;
namespace RT = ps::runtime;

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

qwen35::ScheduledBatch make_batch(
    qwen35::PagedSequenceState& sequence,
    std::vector<qwen35::ScheduledRequest>& requests,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t prefix_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    RT::VerifyNumericMode numeric_mode) {
    qwen35::ScheduledRequest req;
    req.sequence = &sequence;
    req.handle = sequence.request_handle();
    req.execution_class = num_tokens == 1u ? RT::ExecutionClass::DECODE
                                           : RT::ExecutionClass::PREFILL;
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

bool run_batch(
    qwen35::Executor& executor,
    qwen35::PagedSequenceState& sequence,
    const int32_t* tokens,
    uint32_t num_tokens,
    uint32_t output_rows,
    bool speculative_verify,
    RT::VerifyNumericMode numeric_mode,
    std::vector<int32_t>& sampled,
    hipStream_t stream) {
    std::vector<qwen35::ScheduledRequest> requests;
    qwen35::ScheduledBatch batch = make_batch(
        sequence, requests, tokens, num_tokens, sequence.position, output_rows,
        speculative_verify, numeric_mode);
    auto executed = qwen35::execute_batch(executor, batch, stream);
    if (!executed.ok()) {
        std::printf("FAIL: execute_batch: %s\n", executed.status().message().c_str());
        return false;
    }
    qwen35::BatchExecutionOutput output = executed.release();
    sampled.assign(output_rows, -1);
    hipError_t err = hipMemcpy(
        sampled.data(), output.sampled_tokens.data<int32_t>(),
        output_rows * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::printf("FAIL: sample copy: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

struct Rec {
    uint32_t feature_count = 0u;
    uint32_t row_stride = 0u;
    uint32_t dtype = 0u;
    std::vector<uint8_t> data;
};

struct Tracer {
    std::unordered_map<uint32_t, Rec> map;
    std::vector<uint32_t> order;
    void reset() {
        map.clear();
        order.clear();
    }
};

std::size_t elem_bytes(uint32_t dtype) {
    return dtype == static_cast<uint32_t>(RT::ValueDType::BF16) ? 2u : 4u;
}

int parse_layer(const std::string& name) {
    if (name.size() < 3u || name[0] != 'L') return -1;
    std::size_t i = 1u;
    int value = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        value = value * 10 + (name[i] - '0');
        ++i;
    }
    if (i == 1u || i >= name.size() || name[i] != '.') return -1;
    return value;
}

const char* dtype_name(uint32_t dtype) {
    switch (static_cast<RT::ValueDType>(dtype)) {
        case RT::ValueDType::BF16: return "bf16";
        case RT::ValueDType::F32: return "f32";
        case RT::ValueDType::I32: return "i32";
        default: return "?";
    }
}

std::vector<int32_t> read_psktok(const char* path) {
    std::vector<int32_t> out;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return out;
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "PSKLDTOK", 8) != 0) {
        std::fclose(f);
        return out;
    }
    uint32_t version = 0;
    uint64_t count = 0;
    if (std::fread(&version, 4, 1, f) != 1 || std::fread(&count, 8, 1, f) != 1) {
        std::fclose(f);
        return out;
    }
    out.resize(count);
    if (std::fread(out.data(), 4, count, f) != count) out.clear();
    std::fclose(f);
    return out;
}

}  // namespace

int main() {
    const bool safe_only = std::getenv("PHASESHIFT_DIVERGENCE_FULL") == nullptr;

    const char* model_env = std::getenv("PHASESHIFT_MODEL_DIR_MTP");
    if (model_env == nullptr || model_env[0] == '\0') {
        model_env = std::getenv("PHASESHIFT_MODEL_DIR_4B");
    }
    if (model_env == nullptr || model_env[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_MTP\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
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

    uint32_t k_drafts = 4u;
    if (const char* k_env = std::getenv("PHASESHIFT_DIVERGENCE_K")) {
        k_drafts = static_cast<uint32_t>(std::strtoul(k_env, nullptr, 10));
    }
    uint32_t prompt_len = 32u;
    if (const char* p_env = std::getenv("PHASESHIFT_DIVERGENCE_PROMPT")) {
        prompt_len = static_cast<uint32_t>(std::strtoul(p_env, nullptr, 10));
    }
    const uint32_t verify_rows = k_drafts + 1u;

    std::vector<int32_t> prompt(prompt_len);
    const char* prompt_file = std::getenv("PHASESHIFT_DIVERGENCE_PROMPT_FILE");
    if (prompt_file != nullptr && prompt_file[0] != '\0') {
        std::vector<int32_t> ids = read_psktok(prompt_file);
        const uint32_t offset = std::getenv("PHASESHIFT_DIVERGENCE_PROMPT_OFFSET")
            ? static_cast<uint32_t>(std::strtoul(
                  std::getenv("PHASESHIFT_DIVERGENCE_PROMPT_OFFSET"), nullptr, 10))
            : 0u;
        if (ids.size() >= offset + prompt_len) {
            for (uint32_t i = 0u; i < prompt_len; ++i) prompt[i] = ids[offset + i];
        }
    } else {
        for (uint32_t i = 0u; i < prompt_len; ++i) {
            prompt[i] = static_cast<int32_t>(100u + (i * 37u + 11u) % 2000u);
        }
    }

    const uint32_t page_tokens = 16u;
    const uint32_t max_seq_len = prompt_len + k_drafts + 16u;
    const uint32_t pages_per_seq = ceil_div(max_seq_len, page_tokens) + 2u;
    const uint32_t total_pages = pages_per_seq * 4u + 16u;

    auto seq_pool_result = qwen35::SequenceSlotPool::create(arena, 4u, total_pages + 8u);
    auto gdn_pool_result = qwen35::GdnStatePool::create(
        arena, 4u, qwen35::GdnStatePoolLayout::from_text_config(tc));
    auto kv_pool_result = qwen35::PagedKVPool::create(
        arena, total_pages, page_tokens, count_attention_layers(tc),
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
    exec_config.max_scheduled_tokens = std::max(prompt_len, 32u);
    exec_config.max_scheduled_requests = 1u;
    exec_config.max_scheduled_output_rows = 16u;
    auto exec_result = qwen35::create_model_executor(
        model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::printf("FAIL: create_model_executor: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    qwen35::Executor executor = exec_result.release();

    auto make_sequence = [&](qwen35::PagedSequenceState& out) -> bool {
        auto seq = qwen35::create_paged_sequence_state(
            seq_pool, gdn_pool, kv_pool, max_seq_len, stream);
        if (!seq.ok()) {
            std::printf("FAIL: sequence create: %s\n", seq.status().message().c_str());
            return false;
        }
        qwen35::PagedSequenceState state = seq.release();
        std::vector<int32_t> sampled;
        if (!run_batch(executor, state, prompt.data(), prompt_len, 1u, false,
                       RT::VerifyNumericMode::Fast, sampled, stream)) {
            return false;
        }
        qwen35::runtime_install_sequence_state(out, std::move(state));
        return true;
    };

    qwen35::PagedSequenceState seq_ref;
    qwen35::PagedSequenceState seq_a;
    qwen35::PagedSequenceState seq_b;
    if (!make_sequence(seq_ref) || !make_sequence(seq_a) || !make_sequence(seq_b)) {
        return 1;
    }

    std::vector<int32_t> verify_tokens(verify_rows, 0);
    {
        std::vector<int32_t> sampled;
        for (uint32_t r = 0u; r < verify_rows; ++r) {
            const int32_t token = (r == 0u) ? prompt.back() : verify_tokens[r - 1u];
            if (!run_batch(executor, seq_ref, &token, 1u, 1u, false,
                           RT::VerifyNumericMode::Fast, sampled, stream)) return 1;
            verify_tokens[r] = sampled[0];
        }
    }
    if (std::getenv("PHASESHIFT_DIVERGENCE_RANDOM") != nullptr) {
        uint32_t seed = 2654435761u;
        for (uint32_t r = 1u; r < verify_rows; ++r) {
            seed = seed * 1664525u + 1013904223u;
            verify_tokens[r] = static_cast<int32_t>(seed % 200000u);
        }
    }

    auto lp_result = qwen35::lower_qwen35_to_primitives(tc, model.weights());
    if (!lp_result.ok()) {
        std::printf("FAIL: lowering: %s\n", lp_result.status().message().c_str());
        return 1;
    }
    qwen35::Qwen35LoweredPrimitives lowered = lp_result.release();
    std::unordered_map<uint32_t, std::string> names;
    for (const auto& node : lowered.graph.nodes) {
        for (std::size_t j = 0u; j < node.outputs.size(); ++j) {
            std::string name = node.debug_name;
            if (name.empty()) {
                name = std::string("<op") +
                       std::to_string(static_cast<int>(RT::primitive_kind_of(node.node))) + ">";
            }
            if (node.outputs.size() > 1u) {
                name += "#" + std::to_string(j);
            }
            names[node.outputs[j].id] = name;
        }
    }

    Tracer tracer;
    qwen35::runtime::ValueTraceSink sink;
    sink.on_value = [&](uint32_t vid, const void* ptr, uint32_t row_stride,
                        uint32_t feature_count, RT::ValueDType dtype, uint32_t rows,
                        hipStream_t s) {
        if (feature_count == 0u || rows == 0u || row_stride == 0u) return;
        const std::size_t elem = elem_bytes(static_cast<uint32_t>(dtype));
        const std::size_t span = static_cast<std::size_t>(rows - 1u) * row_stride + feature_count;
        Rec rec;
        rec.feature_count = feature_count;
        rec.row_stride = row_stride;
        rec.dtype = static_cast<uint32_t>(dtype);
        rec.data.resize(span * elem);
        if (hipStreamSynchronize(s) != hipSuccess) return;
        if (hipMemcpy(rec.data.data(), ptr, rec.data.size(), hipMemcpyDeviceToHost) !=
            hipSuccess) {
            return;
        }
        if (tracer.map.find(vid) == tracer.map.end()) tracer.order.push_back(vid);
        tracer.map[vid] = std::move(rec);
    };

    if (std::getenv("PHASESHIFT_DIVERGENCE_TIMING") != nullptr) {
        const int reps = 6;
        std::vector<int32_t> s;
        for (int i = 0; i < 2; ++i) {
            if (!run_batch(executor, seq_a, &verify_tokens[0], 1u, 1u, false,
                           RT::VerifyNumericMode::Fast, s, stream)) {
                return 1;
            }
            if (!run_batch(executor, seq_b, verify_tokens.data(), verify_rows, verify_rows, true,
                           RT::VerifyNumericMode::Exact, s, stream)) {
                return 1;
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            if (!run_batch(executor, seq_a, &verify_tokens[0], 1u, 1u, false,
                           RT::VerifyNumericMode::Fast, s, stream)) {
                return 1;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            if (!run_batch(executor, seq_b, verify_tokens.data(), verify_rows, verify_rows, true,
                           RT::VerifyNumericMode::Exact, s, stream)) {
                return 1;
            }
        }
        auto t2 = std::chrono::steady_clock::now();
        const double a = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        const double b = std::chrono::duration<double, std::milli>(t2 - t1).count() / reps;
        std::printf("DIVERG_TIMING M1=%.3f M%u=%.3f ratio=%.3f\n", a, verify_rows, b, b / a);
    }

    std::vector<Tracer> path_a(verify_rows);
    std::vector<int32_t> path_a_sampled(verify_rows, -1);
    executor.value_trace = &sink;
    for (uint32_t r = 0u; r < verify_rows; ++r) {
        tracer = Tracer{};
        std::vector<int32_t> sampled;
        if (!run_batch(executor, seq_a, &verify_tokens[r], 1u, 1u, false,
                       RT::VerifyNumericMode::Fast, sampled, stream)) {
            executor.value_trace = nullptr;
            return 1;
        }
        path_a_sampled[r] = sampled[0];
        path_a[r] = std::move(tracer);
    }
    Tracer path_b;
    {
        tracer = Tracer{};
        std::vector<int32_t> sampled;
        if (!run_batch(executor, seq_b, verify_tokens.data(), verify_rows, verify_rows, true,
                       RT::VerifyNumericMode::Exact, sampled, stream)) {
            executor.value_trace = nullptr;
            return 1;
        }
        path_b = std::move(tracer);
        std::printf("VERIFY_BATCH_SAMPLED:");
        for (uint32_t r = 0u; r < verify_rows; ++r) std::printf(" %d", sampled[r]);
        std::printf("\n");
    }
    executor.value_trace = nullptr;

    std::printf("VERIFY_DIVERGENCE_ENV: prompt=%u K=%u rows=%u tokens=", prompt_len, k_drafts,
                verify_rows);
    for (uint32_t r = 0u; r < verify_rows; ++r) std::printf(" %d", verify_tokens[r]);
    std::printf("\n");
    std::printf("SERIAL_SAMPLED:");
    for (uint32_t r = 0u; r < verify_rows; ++r) std::printf(" %d", path_a_sampled[r]);
    std::printf("\n");

    bool found = false;
    uint32_t first_row = 0u;
    std::string first_name;
    uint32_t first_vid = 0u;
    std::size_t first_offset = 0u;
    uint32_t first_dtype = 0u;
    std::size_t first_rel = 0u;
    double first_ulp = 0.0;

    std::vector<int> layer_exact(static_cast<std::size_t>(tc.num_hidden_layers) + 1, 1);

    const auto safe_name = [](const std::string& n) {
        if (n == "embedding" || n == "final_norm" || n == "logits") return true;
        const auto ends = [&](const char* s) {
            const std::size_t l = std::strlen(s);
            return n.size() >= l && n.compare(n.size() - l, l, s) == 0;
        };
        return ends(".input_norm") || ends(".output");
    };

    for (uint32_t r = 0u; r < verify_rows; ++r) {
        for (uint32_t vid : path_b.order) {
            auto it_a = path_a[r].map.find(vid);
            auto it_b = path_b.map.find(vid);
            if (it_a == path_a[r].map.end() || it_b == path_b.map.end()) continue;
            const Rec& ra = it_a->second;
            const Rec& rb = it_b->second;
            if (rb.dtype == static_cast<uint32_t>(RT::ValueDType::I32)) continue;
            if (safe_only) {
                auto nit = names.find(vid);
                if (nit == names.end() || !safe_name(nit->second)) continue;
            }
            const std::size_t elem = elem_bytes(rb.dtype);
            if (ra.row_stride != rb.row_stride) {
                std::printf("NOTE: row_stride mismatch vid=%u a=%u b=%u\n", vid, ra.row_stride,
                            rb.row_stride);
                continue;
            }
            const std::size_t cmp = static_cast<std::size_t>(rb.feature_count) * elem;
            const std::size_t a_off = 0u;
            const std::size_t b_off = static_cast<std::size_t>(r) * rb.row_stride * elem;
            if (ra.data.size() < a_off + cmp || rb.data.size() < b_off + cmp) continue;
            if (std::memcmp(ra.data.data() + a_off, rb.data.data() + b_off, cmp) == 0) continue;

            std::size_t rel = 0u;
            while (rel < cmp && ra.data[a_off + rel] == rb.data[b_off + rel]) ++rel;
            std::size_t elem_index = 0u;
            if (elem != 0u) elem_index = (rel / elem) % rb.feature_count;
            double delta = 0.0;
            double serial_value = 0.0;
            double batched_value = 0.0;
            if (rb.dtype == static_cast<uint32_t>(RT::ValueDType::BF16)) {
                uint16_t ua = 0u;
                uint16_t ub = 0u;
                std::memcpy(&ua, ra.data.data() + a_off + elem_index * 2u, 2u);
                std::memcpy(&ub, rb.data.data() + b_off + elem_index * 2u, 2u);
                const uint32_t fa_bits = static_cast<uint32_t>(ua) << 16u;
                const uint32_t fb_bits = static_cast<uint32_t>(ub) << 16u;
                float fa = 0.0f;
                float fb = 0.0f;
                std::memcpy(&fa, &fa_bits, 4u);
                std::memcpy(&fb, &fb_bits, 4u);
                serial_value = static_cast<double>(fa);
                batched_value = static_cast<double>(fb);
                const int da = (ua & 0x8000u) ? -static_cast<int>(ua & 0x7FFFu)
                                              : static_cast<int>(ua);
                const int db = (ub & 0x8000u) ? -static_cast<int>(ub & 0x7FFFu)
                                              : static_cast<int>(ub);
                delta = static_cast<double>(da - db);
            } else if (rb.dtype == static_cast<uint32_t>(RT::ValueDType::F32)) {
                float fa = 0.0f;
                float fb = 0.0f;
                std::memcpy(&fa, ra.data.data() + a_off + elem_index * 4u, 4u);
                std::memcpy(&fb, rb.data.data() + b_off + elem_index * 4u, 4u);
                serial_value = static_cast<double>(fa);
                batched_value = static_cast<double>(fb);
                const double denom = std::max(1e-12, static_cast<double>(std::fabs(fb)));
                delta = (serial_value - batched_value) / denom;
            } else if (rb.dtype == static_cast<uint32_t>(RT::ValueDType::I32)) {
                int32_t ia = 0;
                int32_t ib = 0;
                std::memcpy(&ia, ra.data.data() + a_off + elem_index * 4u, 4u);
                std::memcpy(&ib, rb.data.data() + b_off + elem_index * 4u, 4u);
                serial_value = static_cast<double>(ia);
                batched_value = static_cast<double>(ib);
                delta = static_cast<double>(ia - ib);
            }
            std::printf(
                "MISMATCH row=%u value=%u name=%s dtype=%s elem=%zu serial=%.8g batched=%.8g "
                "delta=%.3e\n",
                r, vid, names.count(vid) ? names[vid].c_str() : "<unknown>",
                dtype_name(rb.dtype), elem_index, serial_value, batched_value, delta);
            if (!found) {
                found = true;
                first_row = r;
                first_name = names.count(vid) ? names[vid] : "<unknown>";
                first_vid = vid;
                first_offset = elem_index;
                first_dtype = rb.dtype;
                first_rel = rel;
                first_ulp = delta;
            }
            const int layer = names.count(vid) ? parse_layer(names[vid]) : -1;
            if (layer >= 0 && static_cast<std::size_t>(layer) < layer_exact.size()) {
                layer_exact[static_cast<std::size_t>(layer)] = 0;
            } else if (layer < 0) {
                layer_exact.back() = 0;
            }
        }
    }

    std::printf("LAYER_BITEXACT:");
    for (std::size_t li = 0u; li < layer_exact.size(); ++li) {
        std::printf(" L%zu=%d", li, layer_exact[li]);
    }
    std::printf(" final=%d\n", layer_exact.back());

    if (found) {
        std::printf("FIRST_MISMATCH_ROW=%u\n", first_row);
        std::printf("FIRST_MISMATCH_LAYER=%d\n",
                    parse_layer(first_name));
        std::printf("FIRST_MISMATCH_OP=%s\n", first_name.c_str());
        std::printf("FIRST_MISMATCH_VALUE=%u\n", first_vid);
        std::printf("FIRST_MISMATCH_DTYPE=%s\n", dtype_name(first_dtype));
        std::printf("FIRST_MISMATCH_ELEM=%zu\n", first_offset);
        std::printf("FIRST_MISMATCH_BYTEOFFSET=%zu\n", first_rel);
        std::printf("FIRST_MISMATCH_DELTA=%.6e\n", first_ulp);
        std::printf("MTP_VERIFY_DIVERGENCE: FOUND\n");
    } else {
        std::printf("FIRST_MISMATCH=none\n");
        std::printf("MTP_VERIFY_DIVERGENCE: NONE\n");
    }
    std::printf("MTP_VERIFY_EXACT: %s\n", found ? "FAIL" : "PASS");
    (void)qwen35::executor_shutdown(executor);
    (void)qwen35::release_paged_sequence_state(seq_ref, stream);
    (void)qwen35::release_paged_sequence_state(seq_a, stream);
    (void)qwen35::release_paged_sequence_state(seq_b, stream);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return 0;
}
