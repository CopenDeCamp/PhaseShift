#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

static double bf16_to_f64(bf16_t x) {
    const uint32_t u = static_cast<uint32_t>(x.data) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return static_cast<double>(f);
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

struct FixtureTensor {
    std::vector<bf16_t> bf16;
    std::vector<int32_t> i32;
    std::vector<float> f32;
};

static bool read_bf16(ps::io::SafetensorsReader& r, const std::string& name,
                      FixtureTensor& out) {
    auto spec = r.tensor_spec(name);
    if (!spec.ok() || spec.value().dtype != ps::io::SType::BF16) return false;
    std::size_t bytes = 0;
    auto data = r.tensor_data(name, bytes);
    if (!data.ok()) return false;
    out.bf16.resize(bytes / sizeof(bf16_t));
    std::memcpy(out.bf16.data(), data.value(), bytes);
    return true;
}

static bool read_i32(ps::io::SafetensorsReader& r, const std::string& name,
                     FixtureTensor& out) {
    auto spec = r.tensor_spec(name);
    if (!spec.ok() || spec.value().dtype != ps::io::SType::I32) return false;
    std::size_t bytes = 0;
    auto data = r.tensor_data(name, bytes);
    if (!data.ok()) return false;
    out.i32.resize(bytes / sizeof(int32_t));
    std::memcpy(out.i32.data(), data.value(), bytes);
    return true;
}

static bool read_f32(ps::io::SafetensorsReader& r, const std::string& name,
                     FixtureTensor& out) {
    auto spec = r.tensor_spec(name);
    if (!spec.ok() || spec.value().dtype != ps::io::SType::F32) return false;
    std::size_t bytes = 0;
    auto data = r.tensor_data(name, bytes);
    if (!data.ok()) return false;
    out.f32.resize(bytes / sizeof(float));
    std::memcpy(out.f32.data(), data.value(), bytes);
    return true;
}

static double rel_l2_f32(const std::vector<float>& ref, const std::vector<float>& got) {
    double diff = 0.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
        diff += d * d;
        norm += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return norm > 0.0 ? std::sqrt(diff / norm) : 0.0;
}

static double cosine_f32(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na <= 0.0 || nb <= 0.0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static bool read_meta_i64(const fs::path& path, const std::string& key, int64_t& out) {
    std::FILE* fp = std::fopen(path.string().c_str(), "r");
    if (fp == nullptr) return false;
    std::string text;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) text.append(buf, n);
    std::fclose(fp);
    const std::string needle = "\"" + key + "\"";
    const std::size_t pos = text.find(needle);
    if (pos == std::string::npos) return false;
    const std::size_t colon = text.find(':', pos + needle.size());
    if (colon == std::string::npos) return false;
    out = std::strtoll(text.c_str() + colon + 1, nullptr, 10);
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
    const char* fix_env = std::getenv("PHASESHIFT_DFLASH2_GATE6_FIXTURE");
    const fs::path fixture_root = (fix_env != nullptr && fix_env[0] != '\0')
                                      ? fs::path(fix_env)
                                      : fs::path("build/dflash2-gate6-reference");
    const char* g51_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_FIXTURE");
    const fs::path gate51_root = (g51_env != nullptr && g51_env[0] != '\0')
                                     ? fs::path(g51_env)
                                     : fs::path("build/dflash2-gate51/real");
    const char* cases_env = std::getenv("PHASESHIFT_DFLASH2_GATE6_CASES");
    const std::string cases = (cases_env != nullptr && cases_env[0] != '\0')
                                  ? std::string(cases_env)
                                  : std::string("real_prose,real_code,real_math");
    if (!fs::is_directory(fixture_root)) {
        std::printf("SKIP: fixture dir missing: %s\n", fixture_root.string().c_str());
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();

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

    auto target_result = ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(target_result.ok(), "target model load");
    if (!target_result.ok()) return 1;
    ps::qwen35::Qwen35Model target = target_result.release();

    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "dflash2 weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = 8u;
    exec_config.target_lm_head = &target.weights().lm_head;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t vocab = config.vocab_size;
    const uint32_t top_k = config.selector_top_k;
    const uint32_t rank = config.selector_rank;
    const uint32_t draft_cap = config.block_size - 1u;

    ps::gpu::Tensor score_buf;
    {
        auto view = arena.allocate_aligned(
            static_cast<std::size_t>(draft_cap) * top_k * sizeof(float), 128);
        check(view.ok(), "score alloc");
        if (!view.ok()) return 1;
        auto backing = view.release();
        auto t = ps::gpu::Tensor::create<float>(
            backing, {draft_cap, top_k},
            {static_cast<std::size_t>(top_k), 1u});
        check(t.ok(), "score tensor");
        if (!t.ok()) return 1;
        score_buf = t.release();
    }

    ps::gpu::Tensor noise;
    ps::gpu::Tensor feature;
    check(alloc_bf16(arena, {config.block_size, hidden}, noise), "noise alloc");
    check(alloc_bf16(arena, {8u, hidden}, feature), "feature alloc");

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
        const fs::path g6_dir = fixture_root / name;
        const fs::path g51_dir = gate51_root / name;
        if (!fs::is_directory(g6_dir)) {
            std::printf("SKIP case %s (no fixture)\n", name.c_str());
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }

        int64_t block_rows = 8;
        int64_t anchor = 0;
        check(read_meta_i64(g6_dir / "metadata.json", "block_rows", block_rows),
              name + " block_rows");
        check(read_meta_i64(g6_dir / "metadata.json", "anchor_token_id", anchor),
              name + " anchor");
        const uint32_t block = static_cast<uint32_t>(block_rows);
        const uint32_t draft = block - 1u;
        check(draft <= draft_cap, name + " draft rows");

        auto g6_reader = ps::io::SafetensorsReader::open(
            (g6_dir / "fixture.safetensors").string());
        check(g6_reader.ok(), name + " gate6 fixture open");
        if (!g6_reader.ok()) {
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        ps::io::SafetensorsReader g6 = g6_reader.release();
        FixtureTensor ref_hidden_proj;
        FixtureTensor ref_topk_ids;
        FixtureTensor ref_topk_logits;
        FixtureTensor ref_scores;
        FixtureTensor ref_tokens;
        FixtureTensor ref_logits;
        FixtureTensor ref_draft_hidden;
        const bool g6_ok =
            read_bf16(g6, "hidden_projection", ref_hidden_proj) &&
            read_i32(g6, "top16_ids", ref_topk_ids) &&
            read_f32(g6, "top16_logits", ref_topk_logits) &&
            read_f32(g6, "selector_scores", ref_scores) &&
            read_i32(g6, "draft_tokens", ref_tokens) &&
            read_f32(g6, "lm_logits", ref_logits) &&
            read_bf16(g6, "draft_hidden", ref_draft_hidden);
        check(g6_ok, name + " gate6 fixture tensors");
        if (!g6_ok) {
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }

        std::printf("=== %s (block=%u draft=%u anchor=%lld)\n", name.c_str(), block, draft,
                    static_cast<long long>(anchor));

        check(hipMemcpy(ex.selector_hidden_proj.data<bf16_t>(), ref_hidden_proj.bf16.data(),
                        ref_hidden_proj.bf16.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              name + " hidden_proj upload");
        check(hipMemcpy(ex.proposal_topk_ids.data<int32_t>(), ref_topk_ids.i32.data(),
                        ref_topk_ids.i32.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              name + " topk ids upload");
        check(hipMemcpy(ex.proposal_topk_logits.data<float>(), ref_topk_logits.f32.data(),
                        ref_topk_logits.f32.size() * sizeof(float),
                        hipMemcpyHostToDevice) == hipSuccess,
              name + " topk logits upload");

        ps::Status st = ps::qwen35::dflash2::dflash2_run_candidate_selector(
            ex, ex.selector_hidden_proj.data<bf16_t>(),
            ex.proposal_topk_ids.data<int32_t>(),
            ex.proposal_topk_logits.data<float>(), draft, static_cast<int32_t>(anchor),
            ex.proposal_tokens.data<int32_t>(), stream);
        check(st.ok(), name + " selector: " + st.message());
        check(hipStreamSynchronize(stream) == hipSuccess, name + " selector sync");
        std::vector<int32_t> got_tokens(draft);
        check(hipMemcpy(got_tokens.data(), ex.proposal_tokens.data<int32_t>(),
                        got_tokens.size() * sizeof(int32_t), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              name + " selector read");
        std::vector<int32_t> want_tokens(ref_tokens.i32.begin(), ref_tokens.i32.end());
        const bool selector_exact = got_tokens == want_tokens;
        check(selector_exact, name + " selector tokens mismatch");
        std::printf("  selector isolated: got");
        for (int32_t t : got_tokens) std::printf(" %d", t);
        std::printf("  ref");
        for (int32_t t : want_tokens) std::printf(" %d", t);
        std::printf("  exact=%s\n", selector_exact ? "yes" : "no");

        {
            ps::kernel::DFlash2CandidateSelectorArgs args;
            args.hidden_proj = ex.selector_hidden_proj.data<bf16_t>();
            args.topk_ids = ex.proposal_topk_ids.data<int32_t>();
            args.topk_logits = ex.proposal_topk_logits.data<float>();
            args.predecessor_codebook =
                ex.weights->selector.predecessor_codebook.data<bf16_t>();
            args.successor_codebook =
                ex.weights->selector.successor_codebook.data<bf16_t>();
            args.anchor_token = static_cast<int32_t>(anchor);
            args.output_tokens = ex.proposal_tokens.data<int32_t>();
            args.output_scores = score_buf.data<float>();
            args.draft_rows = draft;
            args.rank = rank;
            args.top_k = top_k;
            args.vocab_size = config.vocab_size;
            const hipError_t err = ps::kernel::launch_dflash2_candidate_selector(args, stream);
            check(err == hipSuccess, name + " selector scores launch");
            check(hipStreamSynchronize(stream) == hipSuccess, name + " score sync");
            std::vector<float> got_scores(static_cast<std::size_t>(draft) * top_k);
            check(hipMemcpy(got_scores.data(), score_buf.data<float>(),
                            got_scores.size() * sizeof(float), hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  name + " score read");
            std::vector<float> want_scores(ref_scores.f32.begin(), ref_scores.f32.end());
            const double score_rel = rel_l2_f32(want_scores, got_scores);
            std::printf("  selector scores: rel_l2=%.3e cosine=%.6f\n", score_rel,
                        cosine_f32(want_scores, got_scores));
            check(score_rel <= 1.0e-2, name + " selector score rel_l2");
        }

        std::vector<bf16_t> full_hidden(static_cast<std::size_t>(block) * hidden);
        std::memcpy(full_hidden.data() + hidden, ref_draft_hidden.bf16.data(),
                    ref_draft_hidden.bf16.size() * sizeof(bf16_t));
        ps::gpu::Tensor full;
        check(alloc_bf16(arena, {block, hidden}, full), name + " full alloc");
        check(hipMemcpy(full.data<bf16_t>(), full_hidden.data(),
                        full_hidden.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
                  hipSuccess,
              name + " full hidden copy");

        st = ps::qwen35::dflash2::dflash2_select_draft_tokens(
            ex, full.data<bf16_t>(), block, static_cast<int32_t>(anchor),
            ex.proposal_tokens.data<int32_t>(), stream);
        check(st.ok(), name + " select: " + st.message());
        check(hipStreamSynchronize(stream) == hipSuccess, name + " select sync");
        std::vector<int32_t> got_sel(draft, 0);
        check(hipMemcpy(got_sel.data(), ex.proposal_tokens.data<int32_t>(),
                        got_sel.size() * sizeof(int32_t), hipMemcpyDeviceToHost) ==
                  hipSuccess,
              name + " select read");

        {
            std::vector<float> got_logits(static_cast<std::size_t>(draft) * config.vocab_size);
            check(hipMemcpy(got_logits.data(), ex.proposal_logits.data<float>(),
                            got_logits.size() * sizeof(float), hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  name + " logits read");
            std::vector<float> want_logits(ref_logits.f32.begin(), ref_logits.f32.end());
            std::size_t nonfinite = 0;
            for (float v : got_logits) {
                if (!std::isfinite(v)) ++nonfinite;
            }
            check(nonfinite == 0, name + " logits non-finite");
            const double lrel = rel_l2_f32(want_logits, got_logits);
            std::printf("  lm_head logits (vs bf16 oracle): rel_l2=%.3e cosine=%.6f\n", lrel,
                        cosine_f32(want_logits, got_logits));
            check(lrel <= 5.0e-2, name + " logits rel_l2 (cross-precision sanity)");
            FixtureTensor ref_psq;
            if (read_f32(g6, "psq_logits", ref_psq)) {
                std::vector<float> want_psq(ref_psq.f32.begin(), ref_psq.f32.end());
                const double prel = rel_l2_f32(want_psq, got_logits);
                std::printf("  lm_head logits (vs psq8 oracle): rel_l2=%.3e cosine=%.6f\n",
                            prel, cosine_f32(want_psq, got_logits));
                check(prel <= 5.0e-3, name + " psq8 logits rel_l2");
            }
        }

        {
            std::vector<int32_t> got_ids(static_cast<std::size_t>(draft) * top_k);
            check(hipMemcpy(got_ids.data(), ex.proposal_topk_ids.data<int32_t>(),
                            got_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  name + " topk ids read");
            std::size_t mismatch = 0;
            for (std::size_t i = 0; i < got_ids.size(); ++i) {
                if (got_ids[i] != ref_topk_ids.i32[i]) ++mismatch;
            }
            std::printf("  lm_head top16: ids exact=%s (%zu/%zu)\n",
                        mismatch == 0 ? "yes" : "no", got_ids.size() - mismatch,
                        got_ids.size());
            check(mismatch == 0, name + " top16 ids exact");
        }

        for (uint32_t block_try : {2u, 4u}) {
            const uint32_t draft_try = block_try - 1u;
            st = ps::qwen35::dflash2::dflash2_select_draft_tokens(
                ex, full.data<bf16_t>(), block_try, static_cast<int32_t>(anchor),
                ex.proposal_tokens.data<int32_t>(), stream);
            check(st.ok(), name + " select block=" + std::to_string(block_try) + ": " +
                                st.message());
            check(hipStreamSynchronize(stream) == hipSuccess,
                  name + " select block sync");
            std::vector<int32_t> got_try(draft_try, 0);
            check(hipMemcpy(got_try.data(), ex.proposal_tokens.data<int32_t>(),
                            got_try.size() * sizeof(int32_t), hipMemcpyDeviceToHost) ==
                      hipSuccess,
                  name + " select block read");
            const std::vector<int32_t> want_try(want_tokens.begin(),
                                                want_tokens.begin() + draft_try);
            const bool try_exact = got_try == want_try;
            std::printf("  block_rows=%u isolated: got", block_try);
            for (int32_t t : got_try) std::printf(" %d", t);
            std::printf("  ref");
            for (int32_t t : want_try) std::printf(" %d", t);
            std::printf("  exact=%s\n", try_exact ? "yes" : "no");
            check(try_exact, name + " block_rows=" + std::to_string(block_try) + " exact");
        }

        if (fs::is_directory(g51_dir)) {
            auto g51_reader = ps::io::SafetensorsReader::open(
                (g51_dir / "fixture.safetensors").string());
            check(g51_reader.ok(), name + " gate51 fixture open");
            if (g51_reader.ok()) {
                ps::io::SafetensorsReader g51 = g51_reader.release();
                FixtureTensor f_noise;
                FixtureTensor f_feature;
                const bool ok51 = read_bf16(g51, "noise_embedding", f_noise) &&
                                  read_bf16(g51, "target_feature", f_feature);
                check(ok51, name + " gate51 tensors");
                if (ok51) {
                    check(hipMemcpy(noise.data<bf16_t>(), f_noise.bf16.data(),
                                    f_noise.bf16.size() * sizeof(bf16_t),
                                    hipMemcpyHostToDevice) == hipSuccess,
                          name + " noise upload");
                    check(hipMemcpy(feature.data<bf16_t>(), f_feature.bf16.data(),
                                    f_feature.bf16.size() * sizeof(bf16_t),
                                    hipMemcpyHostToDevice) == hipSuccess,
                          name + " feature upload");
                    const uint32_t context_rows =
                        static_cast<uint32_t>(f_feature.bf16.size() / hidden);
                    int64_t context_start = 0;
                    check(read_meta_i64(g51_dir / "metadata.json", "context_start",
                                        context_start),
                          name + " context_start");
                    st = ps::qwen35::dflash2::dflash2_forward_and_select_stateless(
                        ex, noise.data<bf16_t>(), feature.data<bf16_t>(), context_rows,
                        static_cast<uint32_t>(context_start),
                        static_cast<uint32_t>(context_start) + context_rows, block,
                        static_cast<int32_t>(anchor), ex.proposal_tokens.data<int32_t>(),
                        stream);
                    check(st.ok(), name + " composed: " + st.message());
                    check(hipStreamSynchronize(stream) == hipSuccess, name + " composed sync");
                    std::vector<int32_t> got_composed(draft);
                    check(hipMemcpy(got_composed.data(), ex.proposal_tokens.data<int32_t>(),
                                    got_composed.size() * sizeof(int32_t),
                                    hipMemcpyDeviceToHost) == hipSuccess,
                          name + " composed read");
                    std::size_t match = 0;
                    for (uint32_t i = 0; i < draft; ++i) {
                        if (got_composed[i] == want_tokens[i]) ++match;
                    }
                    std::printf("  composed proposal: got");
                    for (int32_t t : got_composed) std::printf(" %d", t);
                    std::printf("  ref");
                    for (int32_t t : want_tokens) std::printf(" %d", t);
                    std::printf("  match=%zu/%u\n", match, draft);
                    if (match == draft) {
                        std::printf("  composed proposal exact vs official\n");
                    } else {
                        std::printf("  composed proposal differs at %zu position(s): "
                                    "Gate 5.1 backbone numeric contract\n",
                                    draft - match);
                    }

                    std::vector<bf16_t> composed_hidden(static_cast<std::size_t>(block) * hidden);
                    check(hipMemcpy(composed_hidden.data(),
                                    ex.backbone_output.data<bf16_t>(),
                                    composed_hidden.size() * sizeof(bf16_t),
                                    hipMemcpyDeviceToHost) == hipSuccess,
                          name + " composed hidden read");
                    std::vector<float> composed_f32(composed_hidden.size());
                    for (std::size_t i = 0; i < composed_hidden.size(); ++i) {
                        composed_f32[i] = static_cast<float>(bf16_to_f64(composed_hidden[i]));
                    }
                    {
                        std::vector<float> want_full(static_cast<std::size_t>(block) * hidden);
                        FixtureTensor ref_final;
                        if (read_bf16(g51, "final_hidden", ref_final)) {
                            for (std::size_t i = 0; i < want_full.size(); ++i) {
                                want_full[i] =
                                    static_cast<float>(bf16_to_f64(ref_final.bf16[i]));
                            }
                            const double hrel = rel_l2_f32(want_full, composed_f32);
                            std::printf("  composed final_hidden vs official: rel_l2=%.3e\n",
                                        hrel);
                            check(hrel <= 2.5e-2, name + " composed final_hidden rel_l2");
                        }
                    }

                    std::vector<int32_t> first(draft);
                    bool stable = true;
                    for (uint32_t run = 0; run < 100u; ++run) {
                        st = ps::qwen35::dflash2::dflash2_forward_and_select_stateless(
                            ex, noise.data<bf16_t>(), feature.data<bf16_t>(), context_rows,
                            static_cast<uint32_t>(context_start),
                            static_cast<uint32_t>(context_start) + context_rows, block,
                            static_cast<int32_t>(anchor),
                            ex.proposal_tokens.data<int32_t>(), stream);
                        check(st.ok(), name + " repeat: " + st.message());
                        if (!st.ok()) break;
                        check(hipStreamSynchronize(stream) == hipSuccess, name + " repeat sync");
                        std::vector<int32_t> got(draft);
                        check(hipMemcpy(got.data(), ex.proposal_tokens.data<int32_t>(),
                                        got.size() * sizeof(int32_t),
                                        hipMemcpyDeviceToHost) == hipSuccess,
                              name + " repeat read");
                        if (run == 0u) {
                            first = got;
                        } else if (got != first) {
                            stable = false;
                        }
                    }
                    std::printf("  determinism 100 runs bit-exact=%s\n",
                                stable ? "yes" : "no");
                    check(stable, name + " 100 repeats bit exact");
                }
            }
        }

        if (comma == std::string::npos) break;
        begin = comma + 1;
    }

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");
    std::printf("test_dflash2_gate6_real: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
