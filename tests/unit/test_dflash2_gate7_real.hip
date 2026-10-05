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
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

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

static bool read_bf16(
    ps::io::SafetensorsReader& r, const std::string& name, FixtureTensor& out) {
    auto spec = r.tensor_spec(name);
    if (!spec.ok() || spec.value().dtype != ps::io::SType::BF16) return false;
    std::size_t bytes = 0;
    auto data = r.tensor_data(name, bytes);
    if (!data.ok()) return false;
    out.bf16.resize(bytes / sizeof(bf16_t));
    std::memcpy(out.bf16.data(), data.value(), bytes);
    return true;
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

static uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static double rel_l2_bf16(const std::vector<bf16_t>& ref, const std::vector<bf16_t>& got) {
    double diff = 0.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const uint32_t ur = static_cast<uint32_t>(ref[i].data) << 16;
        const uint32_t ug = static_cast<uint32_t>(got[i].data) << 16;
        float fr;
        float fg;
        std::memcpy(&fr, &ur, sizeof(fr));
        std::memcpy(&fg, &ug, sizeof(fg));
        const double d = static_cast<double>(fr) - static_cast<double>(fg);
        diff += d * d;
        norm += static_cast<double>(fr) * static_cast<double>(fr);
    }
    return norm > 0.0 ? std::sqrt(diff / norm) : 0.0;
}

static double cosine_bf16(const std::vector<bf16_t>& a, const std::vector<bf16_t>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const uint32_t ua = static_cast<uint32_t>(a[i].data) << 16;
        const uint32_t ub = static_cast<uint32_t>(b[i].data) << 16;
        float fa;
        float fb;
        std::memcpy(&fa, &ua, sizeof(fa));
        std::memcpy(&fb, &ub, sizeof(fb));
        dot += static_cast<double>(fa) * static_cast<double>(fb);
        na += static_cast<double>(fa) * static_cast<double>(fa);
        nb += static_cast<double>(fb) * static_cast<double>(fb);
    }
    if (na <= 0.0 || nb <= 0.0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static std::size_t bit_mismatch(
    const std::vector<bf16_t>& a, const std::vector<bf16_t>& b) {
    std::size_t mismatch = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].data != b[i].data) ++mismatch;
    }
    return mismatch;
}

static bool same_bits(const std::vector<bf16_t>& a, const std::vector<bf16_t>& b) {
    return a.size() == b.size() && bit_mismatch(a, b) == 0u;
}

struct DeviceFixture {
    ps::gpu::Tensor noise;
    ps::gpu::Tensor feature;
    uint32_t context_rows = 0;
    uint32_t context_start = 0;
    uint32_t block_rows = 0;
    uint32_t block_start = 0;
    int32_t anchor = 0;
};

static bool load_real_fixture(
    ps::gpu::GpuArena& arena,
    const fs::path& g51_dir,
    DeviceFixture& out,
    uint32_t hidden) {
    auto reader_result =
        ps::io::SafetensorsReader::open((g51_dir / "fixture.safetensors").string());
    if (!reader_result.ok()) return false;
    ps::io::SafetensorsReader reader = reader_result.release();
    FixtureTensor noise;
    FixtureTensor feature;
    if (!read_bf16(reader, "noise_embedding", noise) || !read_bf16(reader, "target_feature", feature)) {
        return false;
    }
    int64_t context_rows = 0;
    int64_t context_start = 0;
    int64_t block_start = 0;
    int64_t block_rows = 0;
    int64_t anchor = 0;
    if (!read_meta_i64(g51_dir / "metadata.json", "context_rows", context_rows) ||
        !read_meta_i64(g51_dir / "metadata.json", "context_start", context_start) ||
        !read_meta_i64(g51_dir / "metadata.json", "block_start", block_start) ||
        !read_meta_i64(g51_dir / "metadata.json", "block_rows", block_rows) ||
        !read_meta_i64(g51_dir / "metadata.json", "anchor_token_id", anchor)) {
        return false;
    }
    out.context_rows = static_cast<uint32_t>(context_rows);
    out.context_start = static_cast<uint32_t>(context_start);
    out.block_start = static_cast<uint32_t>(block_start);
    out.block_rows = static_cast<uint32_t>(block_rows);
    out.anchor = static_cast<int32_t>(anchor);
    if (feature.bf16.size() != static_cast<std::size_t>(out.context_rows) * hidden) return false;
    if (noise.bf16.size() != static_cast<std::size_t>(out.block_rows) * hidden) return false;
    if (!alloc_bf16(arena, {out.block_rows, hidden}, out.noise)) return false;
    if (!alloc_bf16(arena, {out.context_rows, hidden}, out.feature)) return false;
    if (hipMemcpy(out.noise.data<bf16_t>(), noise.bf16.data(),
                  noise.bf16.size() * sizeof(bf16_t), hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }
    if (hipMemcpy(out.feature.data<bf16_t>(), feature.bf16.data(),
                  feature.bf16.size() * sizeof(bf16_t), hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }
    return true;
}

static std::vector<bf16_t> gather_logical_kv(
    const DFlash2ContextState& state,
    uint32_t layer,
    hipStream_t stream) {
    const uint32_t capacity = state.capacity;
    const uint32_t kv_features = state.kv_features;
    const uint32_t length = state.length;
    const uint32_t start = state.next_position - state.length;
    const std::size_t layer_stride = static_cast<std::size_t>(capacity) * kv_features;
    const bf16_t* base =
        state.k_ring.data<bf16_t>() + static_cast<std::size_t>(layer) * layer_stride;
    std::vector<bf16_t> physical(static_cast<std::size_t>(capacity) * kv_features);
    if (hipMemcpy(physical.data(), base, physical.size() * sizeof(bf16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return {};
    }
    std::vector<bf16_t> logical(static_cast<std::size_t>(length) * kv_features);
    for (uint32_t i = 0; i < length; ++i) {
        const uint32_t slot = (start + i) % capacity;
        for (uint32_t d = 0; d < kv_features; ++d) {
            logical[static_cast<std::size_t>(i) * kv_features + d] =
                physical[static_cast<std::size_t>(slot) * kv_features + d];
        }
    }
    return logical;
}

static void read_ring_physical(
    const DFlash2ContextState& state, std::vector<bf16_t>& k, std::vector<bf16_t>& v) {
    const std::size_t total =
        static_cast<std::size_t>(state.num_layers) * state.capacity * state.kv_features;
    k.resize(total);
    v.resize(total);
    (void)hipMemcpy(k.data(), state.k_ring.data<bf16_t>(), total * sizeof(bf16_t),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(v.data(), state.v_ring.data<bf16_t>(), total * sizeof(bf16_t),
                    hipMemcpyDeviceToHost);
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (model_dir == nullptr || model_dir[0] == '\0' || target_dir == nullptr ||
        target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2 and _TARGET\n");
        return 77;
    }
    const char* fix_env = std::getenv("PHASESHIFT_DFLASH2_GATE7_FIXTURE");
    const fs::path fixture_root = (fix_env != nullptr && fix_env[0] != '\0')
                                      ? fs::path(fix_env)
                                      : fs::path("build/dflash2-gate7-reference");
    const char* g51_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_FIXTURE");
    const fs::path gate51_root = (g51_env != nullptr && g51_env[0] != '\0')
                                     ? fs::path(g51_env)
                                     : fs::path("build/dflash2-gate51/real");
    const char* cases_env = std::getenv("PHASESHIFT_DFLASH2_GATE7_CASES");
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

    ps::resident::ResidentTestRequest resident_request;
    resident_request.qwen_model_dir = target_dir;
    resident_request.dflash2_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 24ull * 1024ull * 1024ull * 1024ull;
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

    auto target_result = resident.enabled()
        ? resident.take_qwen35()
        : ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    check(target_result.ok(), "target model load");
    if (!target_result.ok()) return 1;
    ps::qwen35::Qwen35Model target = target_result.release();

    auto weights_result = resident.enabled()
        ? resident.take_dflash2(nullptr)
        : ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "dflash2 weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = static_cast<uint32_t>(config.sliding_window);
    exec_config.target_lm_head = &target.weights().lm_head;
    auto exec_result =
        ps::qwen35::dflash2::create_dflash2_executor(config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t num_layers = static_cast<uint32_t>(config.num_hidden_layers);

    ps::gpu::Tensor out_cached;
    ps::gpu::Tensor out_stateless;
    check(alloc_bf16(arena, {8u, hidden}, out_cached), "cached out alloc");
    check(alloc_bf16(arena, {8u, hidden}, out_stateless), "stateless out alloc");

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
        const fs::path g7_dir = fixture_root / name;
        const fs::path g51_dir = gate51_root / name;
        if (!fs::is_directory(g7_dir) || !fs::is_directory(g51_dir)) {
            std::printf("SKIP case %s (no fixture)\n", name.c_str());
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }

        DeviceFixture fx;
        if (!load_real_fixture(arena, g51_dir, fx, hidden)) {
            check(false, name + " gate51 fixture load");
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        const uint32_t block = fx.block_rows;
        const uint32_t draft = block - 1u;

        std::printf("=== %s (ctx=%u start=%u blk=%u anchor=%d)\n", name.c_str(),
                    fx.context_rows, fx.context_start, block, static_cast<int>(fx.anchor));

        auto state_result = ps::qwen35::dflash2::create_dflash2_context_state(
            config, arena, fx.context_start);
        check(state_result.ok(), name + " context create");
        if (!state_result.ok()) {
            if (comma == std::string::npos) break;
            begin = comma + 1;
            continue;
        }
        DFlash2ContextState state = state_result.release();

        ps::Status st = ps::qwen35::dflash2::dflash2_context_append(
            ex, state, fx.feature.data<bf16_t>(), fx.context_rows, fx.context_start, stream);
        check(st.ok(), name + " append: " + st.message());
        check(state.length == fx.context_rows && state.next_position == fx.block_start,
              name + " append metadata");
        check(hipStreamSynchronize(stream) == hipSuccess, name + " append sync");

        auto g7_reader_result =
            ps::io::SafetensorsReader::open((g7_dir / "fixture.safetensors").string());
        check(g7_reader_result.ok(), name + " gate7 fixture open");
        if (g7_reader_result.ok()) {
            ps::io::SafetensorsReader g7 = g7_reader_result.release();
            double max_k_rel = 0.0;
            double max_v_rel = 0.0;
            double min_k_cos = 1.0;
            double min_v_cos = 1.0;
            for (uint32_t li = 0; li < num_layers; ++li) {
                FixtureTensor ref_k;
                FixtureTensor ref_v;
                check(read_bf16(g7, "layer" + std::to_string(li) + "_k_rope", ref_k),
                      name + " ref k" + std::to_string(li));
                check(read_bf16(g7, "layer" + std::to_string(li) + "_v", ref_v),
                      name + " ref v" + std::to_string(li));
                const std::vector<bf16_t> got_k = gather_logical_kv(state, li, stream);
                check(got_k.size() == ref_k.bf16.size(),
                      name + " k size layer" + std::to_string(li));
                if (got_k.size() != ref_k.bf16.size()) continue;
                const double krel = rel_l2_bf16(ref_k.bf16, got_k);
                const double kcos = cosine_bf16(ref_k.bf16, got_k);
                if (li == 0u || li == num_layers - 1u) {
                    std::printf("  layer%u K ref: rel_l2=%.3e cosine=%.6f\n", li, krel, kcos);
                }
                check(krel <= 1.0e-2, name + " layer" + std::to_string(li) + " K rel_l2");
                check(kcos >= 0.9999, name + " layer" + std::to_string(li) + " K cosine");

                std::vector<bf16_t> ring_v(static_cast<std::size_t>(state.capacity) *
                                           kv_features);
                const std::size_t layer_stride =
                    static_cast<std::size_t>(state.capacity) * kv_features;
                check(hipMemcpy(ring_v.data(),
                                state.v_ring.data<bf16_t>() +
                                    static_cast<std::size_t>(li) * layer_stride,
                                ring_v.size() * sizeof(bf16_t),
                                hipMemcpyDeviceToHost) == hipSuccess,
                      name + " v read");
                std::vector<bf16_t> got_v(
                    static_cast<std::size_t>(state.length) * kv_features);
                const uint32_t start = state.next_position - state.length;
                for (uint32_t i = 0; i < state.length; ++i) {
                    const uint32_t slot = (start + i) % state.capacity;
                    for (uint32_t d = 0; d < kv_features; ++d) {
                        got_v[static_cast<std::size_t>(i) * kv_features + d] =
                            ring_v[static_cast<std::size_t>(slot) * kv_features + d];
                    }
                }
                const double vrel = rel_l2_bf16(ref_v.bf16, got_v);
                const double vcos = cosine_bf16(ref_v.bf16, got_v);
                if (li == 0u || li == num_layers - 1u) {
                    std::printf("  layer%u V ref: rel_l2=%.3e cosine=%.6f\n", li, vrel, vcos);
                }
                check(vrel <= 1.0e-2, name + " layer" + std::to_string(li) + " V rel_l2");
                check(vcos >= 0.9999, name + " layer" + std::to_string(li) + " V cosine");
                max_k_rel = std::max(max_k_rel, krel);
                max_v_rel = std::max(max_v_rel, vrel);
                min_k_cos = std::min(min_k_cos, kcos);
                min_v_cos = std::min(min_v_cos, vcos);
            }
            std::printf("  K/V ref max: K rel_l2=%.3e (cos>=%.6f) V rel_l2=%.3e (cos>=%.6f)\n",
                        max_k_rel, min_k_cos, max_v_rel, min_v_cos);
        }

        {
            std::vector<bf16_t> ring_k_before;
            std::vector<bf16_t> ring_v_before;
            read_ring_physical(state, ring_k_before, ring_v_before);

            st = ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
                ex, fx.noise.data<bf16_t>(), fx.feature.data<bf16_t>(), fx.context_rows,
                fx.context_start, fx.block_start, block, out_stateless.data<bf16_t>(), stream);
            check(st.ok(), name + " stateless backbone: " + st.message());
            st = ps::qwen35::dflash2::dflash2_forward_backbone_cached(
                ex, state, fx.noise.data<bf16_t>(), fx.block_start, block,
                out_cached.data<bf16_t>(), stream);
            check(st.ok(), name + " cached backbone: " + st.message());
            check(hipStreamSynchronize(stream) == hipSuccess, name + " backbone sync");

            std::vector<bf16_t> sc(static_cast<std::size_t>(block) * hidden);
            std::vector<bf16_t> cc(static_cast<std::size_t>(block) * hidden);
            check(hipMemcpy(sc.data(), out_stateless.data<bf16_t>(),
                            sc.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
                  name + " stateless read");
            check(hipMemcpy(cc.data(), out_cached.data<bf16_t>(), cc.size() * sizeof(bf16_t),
                            hipMemcpyDeviceToHost) == hipSuccess,
                  name + " cached read");
            const double rel = rel_l2_bf16(sc, cc);
            const double cos = cosine_bf16(sc, cc);
            const std::size_t mismatch = bit_mismatch(sc, cc);
            std::printf("  cached-vs-stateless final: bit-exact=%s rel_l2=%.3e cosine=%.6f\n",
                        mismatch == 0 ? "yes" : "no", rel, cos);
            check(mismatch == 0, name + " cached-vs-stateless final bit exact");
            check(rel <= 1.0e-6, name + " cached-vs-stateless final rel_l2");
            check(cos >= 0.999999, name + " cached-vs-stateless final cosine");

            for (uint32_t block_try : {8u, 4u, 2u}) {
                const uint32_t draft_try = block_try - 1u;
                check(hipMemset(ex.proposal_tokens.data<int32_t>(), 0,
                                draft_try * sizeof(int32_t)) == hipSuccess,
                      name + " clear stateless tokens");
                st = ps::qwen35::dflash2::dflash2_forward_and_select_stateless(
                    ex, fx.noise.data<bf16_t>(), fx.feature.data<bf16_t>(), fx.context_rows,
                    fx.context_start, fx.block_start, block_try, fx.anchor,
                    ex.proposal_tokens.data<int32_t>(), stream);
                check(st.ok(), name + " stateless proposer: " + st.message());
                check(hipStreamSynchronize(stream) == hipSuccess,
                      name + " stateless proposer sync");
                std::vector<int32_t> want(draft_try, 0);
                check(hipMemcpy(want.data(), ex.proposal_tokens.data<int32_t>(),
                                want.size() * sizeof(int32_t),
                                hipMemcpyDeviceToHost) == hipSuccess,
                      name + " stateless read");
                check(hipMemset(ex.proposal_tokens.data<int32_t>(), 0,
                                draft_try * sizeof(int32_t)) == hipSuccess,
                      name + " clear cached tokens");
                st = ps::qwen35::dflash2::dflash2_forward_and_select_cached(
                    ex, state, fx.noise.data<bf16_t>(), fx.block_start, block_try, fx.anchor,
                    ex.proposal_tokens.data<int32_t>(), stream);
                check(st.ok(), name + " cached proposer: " + st.message());
                check(hipStreamSynchronize(stream) == hipSuccess,
                      name + " cached proposer sync");
                std::vector<int32_t> got(draft_try, 0);
                check(hipMemcpy(got.data(), ex.proposal_tokens.data<int32_t>(),
                                got.size() * sizeof(int32_t),
                                hipMemcpyDeviceToHost) == hipSuccess,
                      name + " cached read");
                const bool exact = got == want;
                std::printf("  block_rows=%u cached-vs-stateless draft: exact=%s",
                            block_try, exact ? "yes" : "no");
                std::printf(" got");
                for (int32_t t : got) std::printf(" %d", t);
                std::printf(" ref");
                for (int32_t t : want) std::printf(" %d", t);
                std::printf("\n");
                check(exact, name + " cached-vs-stateless draft block=" +
                                 std::to_string(block_try));
            }

            std::vector<int32_t> first(draft);
            bool stable = true;
            for (uint32_t run = 0; run < 100u; ++run) {
                st = ps::qwen35::dflash2::dflash2_forward_and_select_cached(
                    ex, state, fx.noise.data<bf16_t>(), fx.block_start, block, fx.anchor,
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
            std::printf("  cached proposer determinism 100x bit-exact=%s\n",
                        stable ? "yes" : "no");
            check(stable, name + " cached 100 repeats bit exact");

            check(state.length == fx.context_rows && state.next_position == fx.block_start,
                  name + " proposal mutated metadata");

            std::vector<bf16_t> ring_k_after;
            std::vector<bf16_t> ring_v_after;
            read_ring_physical(state, ring_k_after, ring_v_after);
            check(same_bits(ring_k_after, ring_k_before) &&
                      same_bits(ring_v_after, ring_v_before),
                  name + " proposal mutated ring K/V");
        }

        check(ps::qwen35::dflash2::dflash2_context_shutdown(state).ok(),
              name + " context shutdown");
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }

    {
        const uint32_t max_total = 2056u;
        std::vector<bf16_t> host_f(static_cast<std::size_t>(max_total) * hidden);
        uint32_t seed = 987654321u;
        for (bf16_t& x : host_f) {
            const float u = static_cast<float>(lcg(seed) >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{ps::f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.5f)};
        }
        ps::gpu::Tensor synth;
        check(alloc_bf16(arena, {max_total, hidden}, synth), "synth feature alloc");
        check(hipMemcpy(synth.data<bf16_t>(), host_f.data(),
                        host_f.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
              "synth feature upload");

        std::vector<bf16_t> host_noise(static_cast<std::size_t>(8u) * hidden);
        for (bf16_t& x : host_noise) {
            const float u = static_cast<float>(lcg(seed) >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{ps::f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.5f)};
        }
        ps::gpu::Tensor synth_noise;
        check(alloc_bf16(arena, {8u, hidden}, synth_noise), "synth noise alloc");
        check(hipMemcpy(synth_noise.data<bf16_t>(), host_noise.data(),
                        host_noise.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
                  hipSuccess,
              "synth noise upload");

        auto bstate_result =
            ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
        check(bstate_result.ok(), "boundary context create");
        if (bstate_result.ok()) {
            DFlash2ContextState bstate = bstate_result.release();
            uint32_t consumed = 0;
            for (uint32_t total : {2047u, 2048u, 2049u, 2056u}) {
                const uint32_t rows = total - consumed;
                ps::Status ast = ps::qwen35::dflash2::dflash2_context_append(
                    ex, bstate, synth.data<bf16_t>() + static_cast<std::size_t>(consumed) * hidden,
                    rows, consumed, stream);
                check(ast.ok(), "boundary append " + std::to_string(total) + ": " + ast.message());
                consumed = total;
                const uint32_t expected_length =
                    std::min<uint32_t>(static_cast<uint32_t>(config.sliding_window), total);
                const uint32_t expected_start = total - expected_length;
                check(bstate.length == expected_length && bstate.next_position == total &&
                          bstate.next_position - bstate.length == expected_start,
                      "boundary metadata total=" + std::to_string(total));
                check(hipStreamSynchronize(stream) == hipSuccess, "boundary sync");

                ps::Status sst = ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
                    ex, synth_noise.data<bf16_t>(), synth.data<bf16_t>() +
                                                        static_cast<std::size_t>(expected_start) * hidden,
                    expected_length, expected_start, total, 8u, out_stateless.data<bf16_t>(),
                    stream);
                check(sst.ok(), "boundary stateless " + std::to_string(total) + ": " +
                                    sst.message());
                check(ps::qwen35::dflash2::dflash2_forward_backbone_cached(
                          ex, bstate, synth_noise.data<bf16_t>(), total, 8u,
                          out_cached.data<bf16_t>(), stream)
                          .ok(),
                      "boundary cached " + std::to_string(total));
                check(hipStreamSynchronize(stream) == hipSuccess, "boundary backbone sync");
                std::vector<bf16_t> sc(static_cast<std::size_t>(8u) * hidden);
                std::vector<bf16_t> cc(static_cast<std::size_t>(8u) * hidden);
                check(hipMemcpy(sc.data(), out_stateless.data<bf16_t>(),
                                sc.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
                      "boundary stateless read");
                check(hipMemcpy(cc.data(), out_cached.data<bf16_t>(),
                                cc.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
                      "boundary cached read");
                const std::size_t mismatch = bit_mismatch(sc, cc);
                const double rel = rel_l2_bf16(sc, cc);
                std::printf(
                    "  boundary total=%u length=%u start=%u wrap=%s bit-exact=%s rel_l2=%.3e\n",
                    total, bstate.length, bstate.next_position - bstate.length,
                    total > static_cast<uint32_t>(config.sliding_window) ? "yes" : "no",
                    mismatch == 0 ? "yes" : "no", rel);
                check(mismatch == 0, "boundary cached-vs-stateless total=" +
                                         std::to_string(total) + " bit exact");
                check(rel <= 1.0e-6, "boundary cached-vs-stateless rel_l2 total=" +
                                         std::to_string(total));
            }
            check(ps::qwen35::dflash2::dflash2_context_shutdown(bstate).ok(),
                  "boundary context shutdown");
        }
    }

    {
        const uint32_t n = 2048u;
        std::vector<bf16_t> host_f(static_cast<std::size_t>(n) * hidden);
        uint32_t seed = 1357911u;
        for (bf16_t& x : host_f) {
            const float u = static_cast<float>(lcg(seed) >> 8) / static_cast<float>(1u << 24);
            x = bf16_t{ps::f32_to_bf16_rne((u * 2.0f - 1.0f) * 0.25f)};
        }
        ps::gpu::Tensor synth;
        check(alloc_bf16(arena, {n, hidden}, synth), "chunk feature alloc");
        check(hipMemcpy(synth.data<bf16_t>(), host_f.data(), host_f.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              "chunk feature upload");

        ps::gpu::Tensor ref_k;
        ps::gpu::Tensor ref_v;
        bool first = true;
        for (uint32_t chunk : {1u, 8u, 128u, 2048u}) {
            auto st_result =
                ps::qwen35::dflash2::create_dflash2_context_state(config, arena, 0u);
            check(st_result.ok(), "chunk context create");
            if (!st_result.ok()) continue;
            DFlash2ContextState s = st_result.release();
            uint32_t pos = 0;
            while (pos < n) {
                const uint32_t rows = std::min(chunk, n - pos);
                ps::Status ast = ps::qwen35::dflash2::dflash2_context_append(
                    ex, s, synth.data<bf16_t>() + static_cast<std::size_t>(pos) * hidden,
                    rows, pos, stream);
                check(ast.ok(), "chunk append");
                pos += rows;
            }
            check(hipStreamSynchronize(stream) == hipSuccess, "chunk sync");
            check(s.length == n && s.next_position == n, "chunk metadata size" +
                                                             std::to_string(chunk));
            ps::gpu::Tensor ck;
            ps::gpu::Tensor cv;
            check(alloc_bf16(arena, {s.num_layers, s.capacity, s.kv_features}, ck),
                  "chunk k alloc");
            check(alloc_bf16(arena, {s.num_layers, s.capacity, s.kv_features}, cv),
                  "chunk v alloc");
            const std::size_t bytes =
                static_cast<std::size_t>(s.num_layers) * s.capacity * s.kv_features *
                sizeof(bf16_t);
            check(hipMemcpy(ck.data<bf16_t>(), s.k_ring.data<bf16_t>(), bytes,
                            hipMemcpyDeviceToDevice) == hipSuccess,
                  "chunk k copy");
            check(hipMemcpy(cv.data<bf16_t>(), s.v_ring.data<bf16_t>(), bytes,
                            hipMemcpyDeviceToDevice) == hipSuccess,
                  "chunk v copy");
            if (first) {
                ref_k = ck;
                ref_v = cv;
                first = false;
            } else {
                check(hipStreamSynchronize(stream) == hipSuccess, "chunk compare sync");
                std::vector<bf16_t> a(bytes / sizeof(bf16_t));
                std::vector<bf16_t> b(bytes / sizeof(bf16_t));
                check(hipMemcpy(a.data(), ref_k.data<bf16_t>(), bytes,
                                hipMemcpyDeviceToHost) == hipSuccess,
                      "chunk ref k read");
                check(hipMemcpy(b.data(), ck.data<bf16_t>(), bytes,
                                hipMemcpyDeviceToHost) == hipSuccess,
                      "chunk k read");
                check(same_bits(a, b), "chunk K independence chunk=" + std::to_string(chunk));
                check(hipMemcpy(a.data(), ref_v.data<bf16_t>(), bytes,
                                hipMemcpyDeviceToHost) == hipSuccess,
                      "chunk ref v read");
                check(hipMemcpy(b.data(), cv.data<bf16_t>(), bytes,
                                hipMemcpyDeviceToHost) == hipSuccess,
                      "chunk v read");
                check(same_bits(a, b), "chunk V independence chunk=" + std::to_string(chunk));
            }
            check(ps::qwen35::dflash2::dflash2_context_shutdown(s).ok(),
                  "chunk context shutdown");
        }
        std::printf("  chunk size 1/8/128/2048 final ring bit-exact\n");
    }

    check(ps::qwen35::dflash2::dflash2_executor_shutdown(ex).ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_dflash2_gate7_real: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
