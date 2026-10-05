#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/feature_concat.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

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

struct FixtureTensor {
    std::vector<bf16_t> values;
};

static bool read_fixture(
    const ps::io::SafetensorsReader& reader,
    const std::string& name,
    std::size_t expected_elems,
    FixtureTensor& out,
    std::string& err) {
    auto spec = reader.tensor_spec(name);
    if (!spec.ok()) {
        err = "fixture missing " + name;
        return false;
    }
    if (spec.value().dtype != ps::io::SType::BF16) {
        err = "fixture tensor " + name + " is not BF16";
        return false;
    }
    std::size_t bytes = 0;
    auto data = reader.tensor_data(name, bytes);
    if (!data.ok()) {
        err = "fixture read " + name;
        return false;
    }
    if (bytes != expected_elems * sizeof(bf16_t)) {
        err = "fixture tensor " + name + " size mismatch";
        return false;
    }
    out.values.resize(expected_elems);
    std::memcpy(out.values.data(), data.value(), bytes);
    return true;
}

static void compare_device(
    const std::string& name,
    const bf16_t* device_ptr,
    const FixtureTensor& ref,
    hipStream_t stream,
    bool bit_exact,
    double rel_l2_max,
    double cosine_min) {
    std::vector<bf16_t> got(ref.values.size());
    if (!got.empty()) {
        const hipError_t e = hipMemcpy(got.data(), device_ptr,
                                       got.size() * sizeof(bf16_t), hipMemcpyDeviceToHost);
        check(e == hipSuccess, name + " readback");
        if (e != hipSuccess) return;
    }
    if (bit_exact) {
        const bool same = std::memcmp(got.data(), ref.values.data(),
                                      ref.values.size() * sizeof(bf16_t)) == 0;
        std::printf("  %-30s %s (bit-exact)\n", name.c_str(), same ? "PASS" : "FAIL");
        check(same, name + " bit-exact");
        return;
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
    const double rel_l2 = ref_norm > 0.0 ? std::sqrt(diff / ref_norm) : 0.0;
    const double cosine = (ref_norm > 0.0 && got_norm > 0.0)
                              ? dot / (std::sqrt(ref_norm) * std::sqrt(got_norm))
                              : 1.0;
    std::printf("  %-30s rel_l2=%.3e cosine=%.6f max_abs=%.3e worst=%zu\n", name.c_str(),
                rel_l2, cosine, max_abs, worst);
    check(rel_l2 <= rel_l2_max, name + " rel_l2 " + std::to_string(rel_l2));
    check(cosine >= cosine_min, name + " cosine " + std::to_string(cosine));
}

static int run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    DFlash2Executor& ex,
    const DFlash2Config& config,
    const fs::path& case_dir,
    uint32_t rows) {
    const std::string tag = "rows" + std::to_string(rows);
    auto reader_result = ps::io::SafetensorsReader::open(
        (case_dir / "fixture.safetensors").string());
    check(reader_result.ok(), tag + " open fixture");
    if (!reader_result.ok()) return 1;
    ps::io::SafetensorsReader reader = reader_result.release();

    const std::size_t hidden = config.hidden_size;
    const std::size_t concat_features = config.tap_feature_size();
    const std::size_t dynamic_features = config.conv_projection_rows();

    std::string err;
    std::vector<FixtureTensor> taps(ps::kernel::kDFlash2TargetTaps);
    for (std::size_t t = 0; t < taps.size(); ++t) {
        if (!read_fixture(reader, "tap" + std::to_string(t), rows * hidden, taps[t], err)) {
            check(false, tag + " " + err);
            return 1;
        }
    }
    FixtureTensor ref_concat;
    FixtureTensor ref_fc;
    FixtureTensor ref_feature;
    if (!read_fixture(reader, "target_concat", rows * concat_features, ref_concat, err) ||
        !read_fixture(reader, "target_fc", rows * hidden, ref_fc, err) ||
        !read_fixture(reader, "target_feature", rows * hidden, ref_feature, err)) {
        check(false, tag + " " + err);
        return 1;
    }

    ps::gpu::Tensor dev_taps[ps::kernel::kDFlash2TargetTaps];
    const bf16_t* tap_ptrs[ps::kernel::kDFlash2TargetTaps] = {};
    for (std::size_t t = 0; t < taps.size(); ++t) {
        if (!alloc_bf16(arena, {rows, hidden}, dev_taps[t])) {
            check(false, tag + " tap alloc");
            return 1;
        }
        check(hipMemcpy(dev_taps[t].data<bf16_t>(), taps[t].values.data(),
                        taps[t].values.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              tag + " tap upload");
        tap_ptrs[t] = dev_taps[t].data<bf16_t>();
    }

    ps::Status st = ps::qwen35::dflash2::dflash2_project_target_features(
        ex, tap_ptrs, ps::kernel::kDFlash2TargetTaps, rows, stream);
    check(st.ok(), tag + " project target features: " + st.message());
    if (!st.ok()) return 1;
    check(hipStreamSynchronize(stream) == hipSuccess, tag + " project sync");

    compare_device(tag + " target_concat", ex.target_concat.data<bf16_t>(), ref_concat, stream,
                   true, 0.0, 1.0);
    compare_device(tag + " target_fc", ex.target_fc.data<bf16_t>(), ref_fc, stream, false,
                   1.0e-2, 0.9999);
    compare_device(tag + " target_feature", ex.target_feature.data<bf16_t>(), ref_feature,
                   stream, false, 1.0e-2, 0.9999);

    ps::gpu::Tensor dev_hidden;
    ps::gpu::Tensor dev_finish_input;
    ps::gpu::Tensor dev_out;
    check(alloc_bf16(arena, {rows, hidden}, dev_hidden), tag + " hidden alloc");
    check(alloc_bf16(arena, {rows, hidden}, dev_finish_input), tag + " finish input alloc");
    check(alloc_bf16(arena, {rows, hidden}, dev_out), tag + " out alloc");

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        for (const auto& kind : {std::string("attn"), std::string("mlp")}) {
            const bool attention_conv = kind == "attn";
            const std::string prefix = "L" + std::to_string(layer) + "_" + kind;
            FixtureTensor in;
            FixtureTensor ref_dynamic;
            FixtureTensor ref_prepare;
            FixtureTensor fin_in;
            FixtureTensor ref_finish;
            if (!read_fixture(reader, prefix + "_input", rows * hidden, in, err) ||
                !read_fixture(reader, prefix + "_dynamic", rows * dynamic_features,
                              ref_dynamic, err) ||
                !read_fixture(reader, prefix + "_prepare", rows * hidden, ref_prepare, err) ||
                !read_fixture(reader, prefix + "_finish_input", rows * hidden, fin_in, err) ||
                !read_fixture(reader, prefix + "_finish", rows * hidden, ref_finish, err)) {
                check(false, tag + " " + err);
                return 1;
            }
            check(hipMemcpy(dev_hidden.data<bf16_t>(), in.values.data(),
                            in.values.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
                      hipSuccess,
                  tag + " " + prefix + " input upload");

            st = ps::qwen35::dflash2::dflash2_grouped_conv_prepare(
                ex, dev_hidden.data<bf16_t>(), ex.weights->layers[layer], attention_conv,
                rows, dev_out.data<bf16_t>(), stream);
            check(st.ok(), tag + " " + prefix + " prepare: " + st.message());
            if (!st.ok()) return 1;
            check(hipStreamSynchronize(stream) == hipSuccess, tag + " " + prefix + " sync");

            compare_device(tag + " " + prefix + "_dynamic", ex.conv_dynamic.data<bf16_t>(),
                           ref_dynamic, stream, false, 1.0e-2, 0.9999);
            compare_device(tag + " " + prefix + "_prepare", dev_out.data<bf16_t>(), ref_prepare,
                           stream, false, 1.0e-2, 0.9999);

            check(hipMemcpy(dev_finish_input.data<bf16_t>(), fin_in.values.data(),
                            fin_in.values.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
                      hipSuccess,
                  tag + " " + prefix + " finish input upload");
            st = ps::qwen35::dflash2::dflash2_grouped_conv_finish(
                ex, dev_finish_input.data<bf16_t>(), ex.weights->layers[layer], attention_conv,
                rows, dev_out.data<bf16_t>(), stream);
            check(st.ok(), tag + " " + prefix + " finish: " + st.message());
            if (!st.ok()) return 1;
            check(hipStreamSynchronize(stream) == hipSuccess, tag + " " + prefix + " finish sync");
            compare_device(tag + " " + prefix + "_finish", dev_out.data<bf16_t>(), ref_finish,
                           stream, false, 1.0e-2, 0.9999);
        }
    }
    return 0;
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    if (model_dir == nullptr || model_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2\n");
        return 77;
    }
    const char* fixture_env = std::getenv("PHASESHIFT_DFLASH2_GATE3_FIXTURE");
    const fs::path fixture_root =
        (fixture_env != nullptr && fixture_env[0] != '\0')
            ? fs::path(fixture_env)
            : fs::path("build/dflash2-gate3-reference");
    if (!fs::is_directory(fixture_root)) {
        std::printf("SKIP: fixture dir missing: %s\n", fixture_root.string().c_str());
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "config: " +
                                  (config_result.ok() ? "" : config_result.status().message()));
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
    check(weights_result.ok(), "weight load: " +
                                   (weights_result.ok()
                                        ? ""
                                        : weights_result.status().message()));
    if (!weights_result.ok()) return 1;
    DFlash2Weights weights = weights_result.release();

    DFlash2ExecutorConfig exec_config;
    exec_config.max_rows = 8u;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create: " +
                                (exec_result.ok() ? "" : exec_result.status().message()));
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    for (uint32_t rows : {1u, 2u, 8u}) {
        run_case(arena, stream, ex, config, fixture_root / ("rows" + std::to_string(rows)),
                 rows);
    }

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown: " + st.message());
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_gate3_real: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
