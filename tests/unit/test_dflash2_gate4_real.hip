#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

struct FixtureTensor {
    std::vector<bf16_t> values;
    std::size_t rows = 0;
    std::size_t features = 0;
};

static bool read_fixture(
    const ps::io::SafetensorsReader& reader,
    const std::string& name,
    FixtureTensor& out,
    std::string& err) {
    auto spec = reader.tensor_spec(name);
    if (!spec.ok()) {
        err = "fixture missing " + name;
        return false;
    }
    std::size_t bytes = 0;
    auto data = reader.tensor_data(name, bytes);
    if (!data.ok()) {
        err = "fixture read " + name;
        return false;
    }
    const ps::io::SType dtype = spec.value().dtype;
    const std::size_t elem = (dtype == ps::io::SType::BF16) ? 2u : 4u;
    const std::size_t count = bytes / elem;
    out.values.resize(count);
    out.rows = spec.value().shape.size() >= 2 ? spec.value().shape[0] : 1;
    out.features = out.rows > 0 ? count / out.rows : count;
    if (dtype == ps::io::SType::BF16) {
        std::memcpy(out.values.data(), data.value(), bytes);
    } else if (dtype == ps::io::SType::F32) {
        const auto* src = static_cast<const float*>(data.value());
        for (std::size_t i = 0; i < count; ++i) {
            out.values[i] = bf16_t{ps::f32_to_bf16_rne(src[i])};
        }
    } else {
        err = "fixture tensor " + name + " has unsupported dtype";
        return false;
    }
    return true;
}

static bool alloc_bf16(
    ps::gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    ps::gpu::Tensor& out) {
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

static const bf16_t* device_buffer(DFlash2Executor& ex, const std::string& key) {
    if (key == "input_norm") return ex.layer_input_norm.data<bf16_t>();
    if (key == "attention_prepared") return ex.attention_prepared.data<bf16_t>();
    if (key == "q_raw") return ex.q_raw.data<bf16_t>();
    if (key == "k_ctx_raw") return ex.k_ctx_raw.data<bf16_t>();
    if (key == "v_ctx") return ex.v_ctx.data<bf16_t>();
    if (key == "k_noise_raw") return ex.k_noise_raw.data<bf16_t>();
    if (key == "v_noise") return ex.v_noise.data<bf16_t>();
    if (key == "q_norm") return ex.q_norm.data<bf16_t>();
    if (key == "k_ctx_norm") return ex.k_ctx_norm.data<bf16_t>();
    if (key == "k_noise_norm") return ex.k_noise_norm.data<bf16_t>();
    if (key == "q_rope") return ex.q_rope.data<bf16_t>();
    if (key == "k_ctx_rope") return ex.k_ctx_rope.data<bf16_t>();
    if (key == "k_noise_rope") return ex.k_noise_rope.data<bf16_t>();
    if (key == "attention_context") return ex.attention_context.data<bf16_t>();
    if (key == "attention_o") return ex.attention_o.data<bf16_t>();
    if (key == "attention_finished") return ex.attention_finished.data<bf16_t>();
    if (key == "attention_residual") return ex.attention_residual.data<bf16_t>();
    if (key == "post_attention_norm") return ex.post_attention_norm.data<bf16_t>();
    if (key == "mlp_prepared") return ex.mlp_prepared.data<bf16_t>();
    if (key == "mlp_gate") return ex.mlp_gate.data<bf16_t>();
    if (key == "mlp_up") return ex.mlp_up.data<bf16_t>();
    if (key == "mlp_swiglu") return ex.mlp_swiglu.data<bf16_t>();
    if (key == "mlp_down") return ex.mlp_down.data<bf16_t>();
    if (key == "mlp_finished") return ex.mlp_finished.data<bf16_t>();
    if (key == "mlp_dynamic") return ex.conv_dynamic.data<bf16_t>();
    return nullptr;
}

struct StageResult {
    std::string name;
    double rel_l2 = 0.0;
    double cosine = 1.0;
};

static StageResult compare_stage(
    const std::string& name,
    const bf16_t* device_ptr,
    const FixtureTensor& ref,
    double rel_l2_max,
    double cosine_min) {
    StageResult r;
    r.name = name;
    std::vector<bf16_t> got(ref.values.size());
    if (!got.empty()) {
        const hipError_t e = hipMemcpy(got.data(), device_ptr, got.size() * sizeof(bf16_t),
                                       hipMemcpyDeviceToHost);
        check(e == hipSuccess, name + " readback");
        if (e != hipSuccess) return r;
    }
    double diff = 0.0;
    double ref_norm = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double max_abs = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < ref.values.size(); ++i) {
        const double a = bf16_to_f64(ref.values[i]);
        const double b = bf16_to_f64(got[i]);
        check(std::isfinite(a) && std::isfinite(b), name + " finite");
        const double d = a - b;
        if (std::fabs(d) > max_abs) {
            max_abs = std::fabs(d);
            worst = i;
        }
        diff += d * d;
        ref_norm += a * a;
        dot += a * b;
        got_norm += b * b;
    }
    r.rel_l2 = ref_norm > 0.0 ? std::sqrt(diff / ref_norm) : 0.0;
    r.cosine = (ref_norm > 0.0 && got_norm > 0.0)
                   ? dot / (std::sqrt(ref_norm) * std::sqrt(got_norm))
                   : 1.0;
    std::printf("  %-28s rel_l2=%.3e cosine=%.6f max_abs=%.3e worst=%zu\n", name.c_str(),
                r.rel_l2, r.cosine, max_abs, worst);
    check(r.rel_l2 <= rel_l2_max, name + " rel_l2 " + std::to_string(r.rel_l2));
    check(r.cosine >= cosine_min, name + " cosine " + std::to_string(r.cosine));
    return r;
}

static bool read_metadata_int(const fs::path& case_dir, const std::string& key, uint32_t& out) {
    std::ifstream f(case_dir / "metadata.json");
    if (!f.is_open()) return false;
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string needle = "\"" + key + "\"";
    const std::size_t pos = text.find(needle);
    if (pos == std::string::npos) return false;
    const std::size_t colon = text.find(':', pos + needle.size());
    if (colon == std::string::npos) return false;
    out = static_cast<uint32_t>(std::strtoul(text.c_str() + colon + 1, nullptr, 10));
    return true;
}

static int run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    DFlash2Executor& ex,
    const DFlash2Config& config,
    const fs::path& case_dir) {
    const std::string tag = case_dir.filename().string();
    auto reader_result = ps::io::SafetensorsReader::open(
        (case_dir / "fixture.safetensors").string());
    check(reader_result.ok(), tag + " open fixture");
    if (!reader_result.ok()) return 1;
    ps::io::SafetensorsReader reader = reader_result.release();

    uint32_t context_rows = 0;
    uint32_t block_rows = 0;
    uint32_t context_start = 0;
    uint32_t block_start = 0;
    if (!read_metadata_int(case_dir, "context_rows", context_rows) ||
        !read_metadata_int(case_dir, "block_rows", block_rows) ||
        !read_metadata_int(case_dir, "context_start", context_start) ||
        !read_metadata_int(case_dir, "block_start", block_start)) {
        check(false, tag + " metadata");
        return 1;
    }

    std::string err;
    FixtureTensor hidden_in;
    FixtureTensor target_feature;
    if (!read_fixture(reader, "hidden_in", hidden_in, err) ||
        !read_fixture(reader, "target_feature", target_feature, err)) {
        check(false, tag + " " + err);
        return 1;
    }

    FixtureTensor ref_mask;
    if (!read_fixture(reader, "attention_mask", ref_mask, err)) {
        check(false, tag + " " + err);
        return 1;
    }
    {
        const uint32_t window = static_cast<uint32_t>(config.sliding_window);
        std::size_t mismatch = 0;
        for (std::size_t r = 0; r < block_rows; ++r) {
            for (std::size_t k = 0; k < context_rows + block_rows; ++k) {
                const bool is_ctx = k < context_rows;
                const int64_t key_pos = is_ctx ? static_cast<int64_t>(context_start) + k
                                               : static_cast<int64_t>(block_start) + (k - context_rows);
                const int64_t query_pos = static_cast<int64_t>(block_start) + r;
                const int64_t distance = key_pos - query_pos;
                const int64_t abs_distance = distance < 0 ? -distance : distance;
                const bool visible = abs_distance < static_cast<int64_t>(window);
                const bool ref_visible = bf16_to_f64(ref_mask.values[r * ref_mask.features + k]) != 0.0;
                if (visible != ref_visible) ++mismatch;
            }
        }
        check(mismatch == 0, tag + " official attention mask equals |q-k| < window");
    }

    ps::gpu::Tensor dev_hidden;
    ps::gpu::Tensor dev_target;
    ps::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {block_rows, config.hidden_size}, dev_hidden), tag + " hidden alloc");
    check(alloc_bf16(arena, {context_rows, config.hidden_size}, dev_target),
          tag + " target alloc");
    check(alloc_bf16(arena, {block_rows, config.hidden_size}, dev_out), tag + " out alloc");
    check(hipMemcpy(dev_hidden.data<bf16_t>(), hidden_in.values.data(),
                    hidden_in.values.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
              hipSuccess,
          tag + " hidden upload");
    check(hipMemcpy(dev_target.data<bf16_t>(), target_feature.values.data(),
                    target_feature.values.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
              hipSuccess,
          tag + " target upload");

    const ps::Status st = ps::qwen35::dflash2::dflash2_forward_layer_stateless(
        ex, 0u, dev_hidden.data<bf16_t>(), dev_target.data<bf16_t>(), context_rows,
        context_start, block_start, block_rows, dev_out.data<bf16_t>(), stream);
    check(st.ok(), tag + " layer forward: " + st.message());
    if (!st.ok()) return 1;
    check(hipStreamSynchronize(stream) == hipSuccess, tag + " layer sync");

    struct Entry {
        const char* fixture;
        const char* device;
        double rel_l2_max;
        double cosine_min;
        bool uses_context;
    };
    const Entry entries[] = {
        {"input_norm", "input_norm", 1.0e-2, 0.9999, false},
        {"attention_prepared", "attention_prepared", 1.0e-2, 0.9999, false},
        {"q_raw", "q_raw", 1.0e-2, 0.9999, false},
        {"k_ctx_raw", "k_ctx_raw", 1.0e-2, 0.9999, true},
        {"v_ctx", "v_ctx", 1.0e-2, 0.9999, true},
        {"k_noise_raw", "k_noise_raw", 1.0e-2, 0.9999, false},
        {"v_noise", "v_noise", 1.0e-2, 0.9999, false},
        {"q_norm", "q_norm", 1.0e-2, 0.9999, false},
        {"k_ctx_norm", "k_ctx_norm", 1.0e-2, 0.9999, true},
        {"k_noise_norm", "k_noise_norm", 1.0e-2, 0.9999, false},
        {"q_rope", "q_rope", 1.0e-2, 0.9999, false},
        {"k_ctx_rope", "k_ctx_rope", 1.0e-2, 0.9999, true},
        {"k_noise_rope", "k_noise_rope", 1.0e-2, 0.9999, false},
        {"attention_context", "attention_context", 1.5e-2, 0.9995, false},
        {"attention_o", "attention_o", 1.5e-2, 0.9995, false},
        {"attention_finished", "attention_finished", 1.5e-2, 0.9995, false},
        {"attention_residual", "attention_residual", 1.5e-2, 0.9995, false},
        {"post_attention_norm", "post_attention_norm", 1.0e-2, 0.9999, false},
        {"mlp_prepared", "mlp_prepared", 1.0e-2, 0.9999, false},
        {"mlp_gate", "mlp_gate", 1.0e-2, 0.9999, false},
        {"mlp_up", "mlp_up", 1.0e-2, 0.9999, false},
        {"mlp_swiglu", "mlp_swiglu", 1.0e-2, 0.9999, false},
        {"mlp_down", "mlp_down", 1.0e-2, 0.9999, false},
        {"mlp_finished", "mlp_finished", 1.5e-2, 0.9995, false},
        {"mlp_dynamic", "mlp_dynamic", 1.0e-2, 0.9999, false},
    };

    double rope_max = 0.0;
    double attention_context_rel = 0.0;
    double layer_output_rel = 0.0;
    for (const Entry& e : entries) {
        FixtureTensor ref;
        if (!read_fixture(reader, e.fixture, ref, err)) {
            check(false, tag + " " + err);
            return 1;
        }
        const bf16_t* ptr = device_buffer(ex, e.device);
        if (ptr == nullptr) {
            check(false, tag + " device buffer " + e.device);
            continue;
        }
        const StageResult r =
            compare_stage(tag + " " + e.fixture, ptr, ref, e.rel_l2_max, e.cosine_min);
        if (std::string(e.fixture).find("rope") != std::string::npos) {
            rope_max = r.rel_l2 > rope_max ? r.rel_l2 : rope_max;
        }
        if (std::string(e.fixture) == "attention_context") attention_context_rel = r.rel_l2;
        (void)e.uses_context;
    }

    FixtureTensor ref_output;
    if (!read_fixture(reader, "layer_output", ref_output, err)) {
        check(false, tag + " " + err);
        return 1;
    }
    const StageResult out_result = compare_stage(tag + " layer_output", dev_out.data<bf16_t>(),
                                                 ref_output, 2.0e-2, 0.9995);
    layer_output_rel = out_result.rel_l2;

    const double attention_limit = 10.0 * rope_max + 1.0e-3;
    check(attention_context_rel <= attention_limit,
          tag + " attention error amplification: context " +
              std::to_string(attention_context_rel) + " vs rope max " + std::to_string(rope_max));
    const double layer_limit = 10.0 * attention_context_rel + 2.0e-3;
    check(layer_output_rel <= layer_limit,
          tag + " layer error amplification: output " + std::to_string(layer_output_rel) +
              " vs context " + std::to_string(attention_context_rel));
    return 0;
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    if (model_dir == nullptr || model_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2\n");
        return 77;
    }
    const char* fixture_env = std::getenv("PHASESHIFT_DFLASH2_GATE4_FIXTURE");
    const fs::path fixture_root = (fixture_env != nullptr && fixture_env[0] != '\0')
                                      ? fs::path(fixture_env)
                                      : fs::path("build/dflash2-gate4-reference");
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
    resident_request.dflash2_model_dir = model_dir;
    resident_request.device = 0;
    resident_request.arena_bytes = 8ull * 1024ull * 1024ull * 1024ull;
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

    auto weights_result = resident.enabled()
        ? resident.take_dflash2(nullptr)
        : ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
    check(weights_result.ok(), "weight load");
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    exec_config.max_context_rows = 32u;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    for (const auto& entry : fs::directory_iterator(fixture_root)) {
        if (!entry.is_directory()) continue;
        if (entry.path().filename().string().rfind("ctx", 0) != 0) continue;
        std::printf("=== %s\n", entry.path().filename().string().c_str());
        run_case(arena, stream, ex, config, entry.path());
    }

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown: " + st.message());
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_gate4_real: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
