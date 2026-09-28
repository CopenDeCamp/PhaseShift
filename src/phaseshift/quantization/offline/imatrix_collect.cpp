#include <phaseshift/quantization/offline/imatrix_collect.h>
#include <phaseshift/quantization/offline/imatrix_format.h>
#include <phaseshift/quantization/offline/model_fingerprint.h>
#include <phaseshift/quantization/offline/token_corpus.h>
#include <phaseshift/quantization/fpx/crc32.h>
#include <phaseshift/quantization/imatrix/gpu_collector.h>
#include <phaseshift/quantization/imatrix/model_sites.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/core/memory/arena.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace ps::quantization::imatrix {

namespace {

namespace RT = ::ps::runtime;

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types)
        if (lt != 0) ++n;
    return n;
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

std::string crc_hex8(const uint8_t* data, std::size_t len) {
    const uint32_t crc = ps::quantization::fpx::crc32_iso_hdlc(data, len);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", crc);
    return buf;
}

}

Result<ImatrixSummary> collect_imatrix(const ImatrixOptions& options) {
    if (options.input_dir.empty())
        return Status::invalid_argument("input dir is required", __FILE__, __LINE__);
    if (options.tokens_path.empty())
        return Status::invalid_argument("tokens path is required", __FILE__, __LINE__);
    if (options.output_path.empty())
        return Status::invalid_argument("output path is required", __FILE__, __LINE__);
    if (options.window < 2 || options.stride == 0)
        return Status::invalid_argument("window/stride invalid", __FILE__, __LINE__);
    const std::string model_dir = options.model_dir.empty() ? options.input_dir : options.model_dir;

    auto corpus_res = fpx::read_token_corpus(options.tokens_path);
    if (!corpus_res.ok()) return corpus_res.status();
    fpx::TokenCorpus corpus = corpus_res.release();
    if (corpus.tokens.size() < 2)
        return Status::invalid_argument("corpus has too few tokens", __FILE__, __LINE__);

    auto config_res = ps::qwen35::read_qwen35_text_config(model_dir);
    if (!config_res.ok()) return config_res.status();
    const ps::qwen35::Qwen35TextConfig tc = config_res.value();

    const std::vector<ImatrixSitePlan> site_plan = build_site_plan(tc);
    std::vector<uint32_t> tags;
    std::vector<uint64_t> k_vals;
    tags.reserve(site_plan.size());
    k_vals.reserve(site_plan.size());
    for (const auto& site : site_plan) {
        tags.push_back(site.tag);
        k_vals.push_back(site.k);
    }

    const std::string model_fp = fpx::compute_model_fingerprint(options.input_dir);

    if (hipSetDevice(options.device) != hipSuccess)
        return Status::hip_error("hipSetDevice", hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return Status::hip_error("hipStreamCreate", hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);

    auto arena_res = ps::gpu::GpuArena::create(
        options.device, static_cast<std::size_t>(options.arena_gib * 1024.0 * 1024.0 * 1024.0));
    if (!arena_res.ok()) return arena_res.status();
    ps::gpu::GpuArena arena = arena_res.release();

    GpuImatrixCollector collector;
    Status init_st = collector.initialize(arena, tags, k_vals);
    if (!init_st.ok()) return init_st;

    ps::qwen35::Qwen35LoadOptions load_opts;
    auto model_res = ps::qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream,
                                                                    load_opts);
    if (!model_res.ok()) return model_res.status();
    ps::qwen35::Qwen35Model model = model_res.release();

    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t attn_layers = count_attention_layers(tc);
    const uint32_t window = static_cast<uint32_t>(options.window);
    const uint32_t num_pages = ceil_div(window, 16u) + 2u;

    auto seq_pool_res = ps::qwen35::SequenceSlotPool::create(arena, 1, num_pages);
    if (!seq_pool_res.ok()) return seq_pool_res.status();
    ps::qwen35::SequenceSlotPool seq_pool = seq_pool_res.release();

    auto gdn_pool_res = ps::qwen35::GdnStatePool::create(
        arena, 1, ps::qwen35::GdnStatePoolLayout::from_text_config(tc));
    if (!gdn_pool_res.ok()) return gdn_pool_res.status();
    ps::qwen35::GdnStatePool gdn_pool = gdn_pool_res.release();

    auto kv_pool_res = ps::qwen35::PagedKVPool::create(arena, num_pages, 16u, attn_layers, kv_heads,
                                                       head_dim, ps::qwen35::KVCacheDType::BF16);
    if (!kv_pool_res.ok()) return kv_pool_res.status();
    ps::qwen35::PagedKVPool kv_pool = kv_pool_res.release();

    ps::qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens = window;
    exec_config.max_scheduled_requests = 1;

    auto exec_res =
        ps::qwen35::create_model_executor(model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_res.ok()) return exec_res.status();
    ps::qwen35::Executor executor = exec_res.release();
    executor.imatrix_collector = &collector;

    ImatrixSummary summary;
    const std::size_t total = corpus.tokens.size();
    std::size_t start = 0;
    std::size_t processed_tokens = 0;
    std::size_t pending_flush = 0;
    while (start + 1 < total) {
        const std::size_t count = std::min<std::size_t>(window, total - start);
        if (count < 2) break;

        auto seq_res = ps::qwen35::create_paged_sequence_state(seq_pool, gdn_pool, kv_pool, window,
                                                               stream);
        if (!seq_res.ok()) return seq_res.status();
        ps::qwen35::PagedSequenceState seq = seq_res.release();

        std::size_t done = 0;
        while (done < count) {
            const uint32_t chunk = static_cast<uint32_t>(std::min<std::size_t>(window, count - done));
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq;
            req.handle = seq.request_handle();
            req.execution_class = RT::ExecutionClass::PREFILL;
            req.token_begin = 0;
            req.num_tokens = chunk;
            req.prefix_tokens = seq.position;
            req.compute_logits = false;
            req.sample = false;
            std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = corpus.tokens.data() + start + done;
            batch.requests = reqs;
            batch.num_tokens = chunk;
            batch.num_requests = 1;
            batch.num_decode_requests = 0;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 1;
            auto st = ps::qwen35::execute_batch(executor, batch, stream);
            if (!st.ok()) return st.status();
            done += chunk;
        }
        processed_tokens += count;
        summary.positions += count;
        ++summary.windows;

        Status rel = ps::qwen35::release_paged_sequence_state(seq, stream);
        if (!rel.ok()) return rel;

        if (++pending_flush >= options.flush_windows) {
            Status fl = collector.flush(stream);
            if (!fl.ok()) return fl;
            pending_flush = 0;
            std::printf("[imatrix] %zu / %zu tokens\n", processed_tokens,
                        options.max_calibration_tokens > 0
                            ? std::min(options.max_calibration_tokens, total)
                            : total);
            std::fflush(stdout);
        }
        if (options.max_calibration_tokens > 0 &&
            processed_tokens >= options.max_calibration_tokens)
            break;
        start += options.stride;
    }

    Status flush_st = collector.flush(stream);
    if (!flush_st.ok()) return flush_st;
    if (hipStreamSynchronize(stream) != hipSuccess)
        return Status::hip_error("hipStreamSynchronize", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);

    for (const auto& kv : collector.diagnostics()) {
        if (kv.second.nonfinite_count == 0) continue;
        std::fprintf(stderr, "imatrix nonfinite values: tag=%u count=%llu\n", kv.first,
                     static_cast<unsigned long long>(kv.second.nonfinite_count));
        return Status::invalid_argument("imatrix calibration produced nonfinite values", __FILE__,
                                        __LINE__);
    }

    PsimHeader header;
    header.format = "phaseshift-imatrix";
    header.version = 1;
    header.model.fingerprint = model_fp;
    header.model.config_sha256 = "";
    header.model.architecture = "qwen35_dense";
    header.calibration.corpus_sha256 = corpus.sha256_hex;
    header.calibration.corpus_token_count = corpus.tokens.size();
    header.calibration.calibration_position_count = summary.positions;
    header.calibration.window = options.window;
    header.calibration.stride = options.stride;
    header.calibration.runtime_config = "quantized:prefill:gdn_nonpersistent:fusion0";
    header.components.push_back(
        {corpus.sha256_hex, corpus.tokens.size(), summary.positions});

    std::vector<uint8_t> payload;
    const auto& accs = collector.accumulators();
    for (const auto& kv : accs) {
        const uint32_t tag = kv.first;
        const ImatrixAccumulator& acc = kv.second;
        const uint64_t k = acc.sum_sq.size();
        for (const auto& name : tensor_names_for(tag)) {
            PsimEntryDesc e;
            e.name = name;
            e.k = k;
            e.count = acc.count;
            e.dtype = "f64_sum_sq";
            e.data_offset = payload.size();
            e.data_bytes = k * sizeof(double);
            for (uint64_t i = 0; i < k; ++i) {
                double v = acc.sum_sq[static_cast<std::size_t>(i)];
                uint8_t buf[8];
                std::memcpy(buf, &v, 8);
                payload.insert(payload.end(), buf, buf + 8);
            }
            e.crc32 = crc_hex8(payload.data() + e.data_offset,
                               static_cast<std::size_t>(e.data_bytes));
            header.entries.push_back(e);
        }
    }
    summary.entries = header.entries.size();

    Status write_st = write_psim(header, payload, options.output_path);
    if (!write_st.ok()) return write_st;
    return summary;
}

}
