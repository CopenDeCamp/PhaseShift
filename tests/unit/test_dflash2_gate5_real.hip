#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/dflash2/weights.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rmsnorm.h>
#include <hip/hip_runtime.h>

#include "support/resident_model_fixture.h"

#include <cmath>
#include <cstdio>
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
    out.values.resize(bytes / sizeof(bf16_t));
    std::memcpy(out.values.data(), data.value(), bytes);
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

struct StageResult {
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
    std::printf("  %-26s rel_l2=%.3e cosine=%.6f max_abs=%.3e worst=%zu\n", name.c_str(),
                r.rel_l2, r.cosine, max_abs, worst);
    check(r.rel_l2 <= rel_l2_max, name + " rel_l2 " + std::to_string(r.rel_l2));
    check(r.cosine >= cosine_min, name + " cosine " + std::to_string(r.cosine));
    return r;
}

struct CaseInputs {
    std::vector<FixtureTensor> taps;
    FixtureTensor target_feature;
    FixtureTensor target_concat;
    FixtureTensor noise_embedding;
    std::vector<FixtureTensor> layer_outputs;
    FixtureTensor final_hidden;
    uint32_t context_rows = 0;
    uint32_t block_rows = 0;
    uint32_t context_start = 0;
    uint32_t block_start = 0;
};

static bool load_case(
    const fs::path& case_dir,
    uint32_t num_layers,
    CaseInputs& out,
    std::string& err) {
    auto reader_result = ps::io::SafetensorsReader::open(
        (case_dir / "fixture.safetensors").string());
    if (!reader_result.ok()) {
        err = "open fixture";
        return false;
    }
    ps::io::SafetensorsReader reader = reader_result.release();
    if (!read_metadata_int(case_dir, "context_rows", out.context_rows) ||
        !read_metadata_int(case_dir, "block_rows", out.block_rows) ||
        !read_metadata_int(case_dir, "context_start", out.context_start) ||
        !read_metadata_int(case_dir, "block_start", out.block_start)) {
        err = "metadata";
        return false;
    }
    out.taps.resize(5);
    for (std::size_t i = 0; i < out.taps.size(); ++i) {
        if (!read_fixture(reader, "tap" + std::to_string(i), out.taps[i], err)) return false;
    }
    if (!read_fixture(reader, "target_concat", out.target_concat, err)) return false;
    if (!read_fixture(reader, "target_feature", out.target_feature, err)) return false;
    if (!read_fixture(reader, "noise_embedding", out.noise_embedding, err)) return false;
    if (!read_fixture(reader, "final_hidden", out.final_hidden, err)) return false;
    out.layer_outputs.resize(num_layers);
    for (uint32_t li = 0; li < num_layers; ++li) {
        if (!read_fixture(reader, "layer" + std::to_string(li) + "_output",
                          out.layer_outputs[li], err)) {
            return false;
        }
    }
    return true;
}

struct DeviceCase {
    ps::gpu::Tensor noise;
    ps::gpu::Tensor target_feature;
    ps::gpu::Tensor feature_out;
    std::vector<ps::gpu::Tensor> taps;
    ps::gpu::Tensor backbone_out;
    ps::gpu::Tensor final_norm_out;
    ps::gpu::Tensor chain_a;
    ps::gpu::Tensor chain_b;
};

static bool upload_case(
    ps::gpu::GpuArena& arena,
    const DFlash2Config& config,
    const CaseInputs& in,
    DeviceCase& out) {
    const std::size_t hidden = config.hidden_size;
    if (!alloc_bf16(arena, {in.block_rows, hidden}, out.noise)) return false;
    if (!alloc_bf16(arena, {in.context_rows, hidden}, out.target_feature)) return false;
    if (!alloc_bf16(arena, {in.block_rows, hidden}, out.feature_out)) return false;
    if (!alloc_bf16(arena, {in.block_rows, hidden}, out.backbone_out)) return false;
    if (!alloc_bf16(arena, {in.block_rows, hidden}, out.final_norm_out)) return false;
    if (!alloc_bf16(arena, {in.block_rows, hidden}, out.chain_a)) return false;
    if (!alloc_bf16(arena, {in.block_rows, hidden}, out.chain_b)) return false;
    out.taps.resize(in.taps.size());
    for (std::size_t i = 0; i < in.taps.size(); ++i) {
        if (!alloc_bf16(arena, {in.context_rows, hidden}, out.taps[i])) return false;
    }
    const auto copy = [](ps::gpu::Tensor& dst, const FixtureTensor& src) {
        return hipMemcpy(dst.data<bf16_t>(), src.values.data(),
                         src.values.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
               hipSuccess;
    };
    if (!copy(out.noise, in.noise_embedding)) return false;
    if (!copy(out.target_feature, in.target_feature)) return false;
    for (std::size_t i = 0; i < in.taps.size(); ++i) {
        if (!copy(out.taps[i], in.taps[i])) return false;
    }
    return true;
}

static int run_case(
    ps::gpu::GpuArena& arena,
    hipStream_t stream,
    DFlash2Executor& ex,
    const DFlash2Config& config,
    const fs::path& case_dir,
    bool isolated,
    bool composed,
    const std::string& tag) {
    CaseInputs in;
    std::string err;
    if (!load_case(case_dir, static_cast<uint32_t>(config.num_hidden_layers), in, err)) {
        check(false, tag + " " + err);
        return 1;
    }
    DeviceCase dev;
    if (!upload_case(arena, config, in, dev)) {
        check(false, tag + " upload");
        return 1;
    }

    const bf16_t* tap_ptrs[5] = {};
    for (std::size_t i = 0; i < dev.taps.size(); ++i) tap_ptrs[i] = dev.taps[i].data<bf16_t>();

    if (isolated) {
        std::printf("=== %s isolated (ctx=%u blk=%u)\n", tag.c_str(), in.context_rows,
                    in.block_rows);
        const bf16_t* src = dev.noise.data<bf16_t>();
        for (uint32_t li = 0; li < config.num_hidden_layers; ++li) {
            bf16_t* dst = (li % 2u == 0u) ? dev.chain_a.data<bf16_t>()
                                          : dev.chain_b.data<bf16_t>();
            const ps::Status st = ps::qwen35::dflash2::dflash2_forward_layer_stateless(
                ex, li, src, dev.target_feature.data<bf16_t>(), in.context_rows,
                in.context_start, in.block_start, in.block_rows, dst, stream);
            check(st.ok(), tag + " isolated layer" + std::to_string(li) + ": " + st.message());
            if (!st.ok()) return 1;
            check(hipStreamSynchronize(stream) == hipSuccess, tag + " isolated sync");
            compare_stage(tag + " isolated layer" + std::to_string(li), dst,
                          in.layer_outputs[li], 3.0e-2, 0.9995);
            src = dst;
        }
        const hipError_t e = ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            src, ex.weights->final_norm_weight.data<bf16_t>(),
            dev.final_norm_out.data<bf16_t>(), in.block_rows,
            static_cast<uint32_t>(config.hidden_size),
            static_cast<uint32_t>(config.hidden_size), config.rms_norm_eps, stream);
        check(e == hipSuccess, tag + " isolated final norm");
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " isolated final sync");
        compare_stage(tag + " isolated final_hidden", dev.final_norm_out.data<bf16_t>(),
                      in.final_hidden, 2.0e-2, 0.9995);

        const ps::Status bst = ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
            ex, dev.noise.data<bf16_t>(), dev.target_feature.data<bf16_t>(), in.context_rows,
            in.context_start, in.block_start, in.block_rows, dev.backbone_out.data<bf16_t>(),
            stream);
        check(bst.ok(), tag + " isolated backbone: " + bst.message());
        if (!bst.ok()) return 1;
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " isolated backbone sync");
        compare_stage(tag + " isolated backbone final_hidden",
                      dev.backbone_out.data<bf16_t>(), in.final_hidden, 2.0e-2, 0.9995);
    }

    if (composed) {
        const ps::Status pst = ps::qwen35::dflash2::dflash2_project_target_features(
            ex, tap_ptrs, 5u, in.context_rows, stream);
        check(pst.ok(), tag + " composed projection: " + pst.message());
        if (!pst.ok()) return 1;
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " projection sync");
        std::printf("=== %s composed (ctx=%u blk=%u)\n", tag.c_str(), in.context_rows,
                    in.block_rows);
        compare_stage(tag + " composed target_feature", ex.target_feature.data<bf16_t>(),
                      in.target_feature, 1.0e-2, 0.9999);
        const bf16_t* src = dev.noise.data<bf16_t>();
        for (uint32_t li = 0; li < config.num_hidden_layers; ++li) {
            bf16_t* dst = (li % 2u == 0u) ? dev.chain_a.data<bf16_t>()
                                          : dev.chain_b.data<bf16_t>();
            const ps::Status st = ps::qwen35::dflash2::dflash2_forward_layer_stateless(
                ex, li, src, ex.target_feature.data<bf16_t>(), in.context_rows,
                in.context_start, in.block_start, in.block_rows, dst, stream);
            check(st.ok(), tag + " composed layer" + std::to_string(li) + ": " + st.message());
            if (!st.ok()) return 1;
            check(hipStreamSynchronize(stream) == hipSuccess, tag + " composed sync");
            compare_stage(tag + " composed layer" + std::to_string(li), dst,
                          in.layer_outputs[li], 4.0e-2, 0.9990);
            src = dst;
        }
        const hipError_t e = ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            src, ex.weights->final_norm_weight.data<bf16_t>(),
            dev.final_norm_out.data<bf16_t>(), in.block_rows,
            static_cast<uint32_t>(config.hidden_size),
            static_cast<uint32_t>(config.hidden_size), config.rms_norm_eps, stream);
        check(e == hipSuccess, tag + " composed final norm");
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " composed final sync");
        compare_stage(tag + " composed final_hidden", dev.final_norm_out.data<bf16_t>(),
                      in.final_hidden, 3.0e-2, 0.9990);

        const ps::Status bst = ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
            ex, dev.noise.data<bf16_t>(), ex.target_feature.data<bf16_t>(), in.context_rows,
            in.context_start, in.block_start, in.block_rows, dev.backbone_out.data<bf16_t>(),
            stream);
        check(bst.ok(), tag + " composed backbone: " + bst.message());
        if (!bst.ok()) return 1;
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " composed backbone sync");
        compare_stage(tag + " composed backbone final_hidden",
                      dev.backbone_out.data<bf16_t>(), in.final_hidden, 3.0e-2, 0.9990);
    }

    {
        const std::size_t hidden = config.hidden_size;
        check(hipMemcpy(dev.final_norm_out.data<bf16_t>(), in.layer_outputs.back().values.data(),
                        in.layer_outputs.back().values.size() * sizeof(bf16_t),
                        hipMemcpyHostToDevice) == hipSuccess,
              tag + " final norm input upload");
        const hipError_t e = ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
            dev.final_norm_out.data<bf16_t>(), ex.weights->final_norm_weight.data<bf16_t>(),
            dev.feature_out.data<bf16_t>(), in.block_rows, static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(hidden), config.rms_norm_eps, stream);
        check(e == hipSuccess, tag + " final norm launch");
        check(hipStreamSynchronize(stream) == hipSuccess, tag + " final norm sync");
        compare_stage(tag + " final_norm_from_reference_layer4",
                      dev.feature_out.data<bf16_t>(), in.final_hidden, 1.0e-2, 0.9999);
    }
    return 0;
}

int main() {
    const char* model_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2");
    if (model_dir == nullptr || model_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2\n");
        return 77;
    }
    const char* fixture_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_FIXTURE");
    const fs::path fixture_root = (fixture_env != nullptr && fixture_env[0] != '\0')
                                      ? fs::path(fixture_env)
                                      : fs::path("build/dflash2-gate5-reference");
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
    exec_config.max_context_rows = 8u;
    auto exec_result = ps::qwen35::dflash2::create_dflash2_executor(
        config, weights, arena, exec_config);
    check(exec_result.ok(), "executor create");
    if (!exec_result.ok()) return 1;
    DFlash2Executor ex = exec_result.release();

    {
        const char* cases_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_CASES");
        const std::string cases =
            (cases_env != nullptr && cases_env[0] != '\0') ? std::string(cases_env)
                                                            : std::string("ctx1_blk2,ctx4_blk4,ctx8_blk8");
        std::size_t begin = 0;
        while (begin <= cases.size()) {
            const std::size_t comma = cases.find(',', begin);
            const std::string name = cases.substr(
                begin, comma == std::string::npos ? std::string::npos : comma - begin);
            if (!name.empty() && fs::is_directory(fixture_root / name)) {
                run_case(arena, stream, ex, config, fixture_root / name, true, true, name);
            }
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }
    }

    {
        const char* det_env = std::getenv("PHASESHIFT_DFLASH2_GATE5_DET_CASE");
        const std::string det_case =
            (det_env != nullptr && det_env[0] != '\0') ? std::string(det_env)
                                                        : std::string("ctx8_blk8");
        if (!fs::is_directory(fixture_root / det_case)) return g_fail == 0 ? 0 : 1;
        CaseInputs in;
        std::string err;
        check(load_case(fixture_root / det_case,
                        static_cast<uint32_t>(config.num_hidden_layers), in, err),
              "determinism fixture: " + err);
        DeviceCase dev;
        check(upload_case(arena, config, in, dev), "determinism upload");
        std::vector<bf16_t> first;
        bool exact = true;
        for (uint32_t run = 0; run < 10u; ++run) {
            const ps::Status st = ps::qwen35::dflash2::dflash2_forward_backbone_stateless(
                ex, dev.noise.data<bf16_t>(), dev.target_feature.data<bf16_t>(),
                in.context_rows, in.context_start, in.block_start, in.block_rows,
                dev.backbone_out.data<bf16_t>(), stream);
            check(st.ok(), "determinism run " + std::to_string(run) + ": " + st.message());
            if (!st.ok()) break;
            check(hipStreamSynchronize(stream) == hipSuccess, "determinism sync");
            std::vector<bf16_t> got(in.final_hidden.values.size());
            check(hipMemcpy(got.data(), dev.backbone_out.data<bf16_t>(),
                            got.size() * sizeof(bf16_t), hipMemcpyDeviceToHost) == hipSuccess,
                  "determinism readback");
            if (run == 0u) {
                first = got;
            } else if (std::memcmp(first.data(), got.data(),
                                   first.size() * sizeof(bf16_t)) != 0) {
                exact = false;
            }
        }
        std::printf("determinism: 10 runs bit-exact=%s\n", exact ? "yes" : "no");
        check(exact, "10 repeated backbone runs must be bit-exact");
    }

    if (fs::is_directory(fixture_root / "ctx8_blk8") &&
        fs::is_directory(fixture_root / "ctx1_blk2") &&
        fs::is_directory(fixture_root / "ctx4_blk4")) {
        const char* order[] = {"ctx8_blk8", "ctx1_blk2", "ctx4_blk4", "ctx8_blk8"};
        std::printf("=== row sequence 8 -> 2 -> 4 -> 8\n");
        for (const char* name : order) {
            run_case(arena, stream, ex, config, fixture_root / name, true, false, name);
        }
    }

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown: " + st.message());
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown: " + shutdown.message());

    std::printf("test_dflash2_gate5_real: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
