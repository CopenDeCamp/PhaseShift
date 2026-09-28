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

static bool read_fixture_tensor(
    ps::io::SafetensorsReader& reader,
    const std::string& name,
    std::vector<bf16_t>& out) {
    auto spec = reader.tensor_spec(name);
    if (!spec.ok()) return false;
    if (spec.value().dtype != ps::io::SType::BF16) return false;
    std::size_t bytes = 0;
    auto data = reader.tensor_data(name, bytes);
    if (!data.ok()) return false;
    out.resize(bytes / sizeof(bf16_t));
    std::memcpy(out.data(), data.value(), bytes);
    return true;
}

struct StageSpec {
    const char* name;
    uint32_t features;
    bool context_rows;
};

static const StageSpec kStages[] = {
    {"input_norm", 5120u, false},
    {"attention_prepared", 5120u, false},
    {"q_raw", 4096u, false},
    {"q_norm", 4096u, false},
    {"q_rope", 4096u, false},
    {"k_ctx_raw", 1024u, true},
    {"k_ctx_norm", 1024u, true},
    {"k_ctx_rope", 1024u, true},
    {"v_ctx", 1024u, true},
    {"k_noise_raw", 1024u, false},
    {"k_noise_norm", 1024u, false},
    {"k_noise_rope", 1024u, false},
    {"v_noise", 1024u, false},
    {"attention_context", 4096u, false},
    {"attention_o", 5120u, false},
    {"attention_finished", 5120u, false},
    {"attention_residual", 5120u, false},
    {"post_attention_norm", 5120u, false},
    {"mlp_prepared", 5120u, false},
    {"mlp_gate", 17408u, false},
    {"mlp_up", 17408u, false},
    {"mlp_swiglu", 17408u, false},
    {"mlp_down", 5120u, false},
    {"mlp_finished", 5120u, false},
};

static void dump_tensor(
    const fs::path& dir,
    const std::string& name,
    const bf16_t* device_ptr,
    std::size_t count,
    hipStream_t stream) {
    if (count == 0 || device_ptr == nullptr) return;
    std::vector<bf16_t> host(count);
    if (hipMemcpy(host.data(), device_ptr, count * sizeof(bf16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        ++g_fail;
        std::printf("FAIL dump %s\n", name.c_str());
        return;
    }
    const std::string path = (dir / (name + ".bin")).string();
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        ++g_fail;
        std::printf("FAIL open %s\n", path.c_str());
        return;
    }
    std::fwrite(host.data(), sizeof(bf16_t), host.size(), fp);
    std::fclose(fp);
    (void)stream;
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
    const char* case_env = std::getenv("PHASESHIFT_DFLASH2_GATE51_CASE");
    const std::string case_name = (case_env != nullptr && case_env[0] != '\0')
                                      ? std::string(case_env)
                                      : std::string("ctx8_blk8");
    const fs::path case_dir = fixture_root / case_name;
    if (!fs::is_directory(case_dir)) {
        std::printf("SKIP: fixture case missing: %s\n", case_dir.string().c_str());
        return 77;
    }
    const char* out_env = std::getenv("PHASESHIFT_DFLASH2_GATE51_OUT");
    const fs::path ps_dir = (out_env != nullptr && out_env[0] != '\0')
                                ? fs::path(out_env)
                                : fs::path("build/dflash2-gate51/ps");
    fs::create_directories(ps_dir);

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto config_result = ps::qwen35::dflash2::read_dflash2_config(model_dir);
    check(config_result.ok(), "config");
    if (!config_result.ok()) return 1;
    const DFlash2Config config = config_result.value();

    auto arena_result = ps::gpu::GpuArena::create(0, 8ull * 1024ull * 1024ull * 1024ull);
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

    auto weights_result =
        ps::qwen35::dflash2::load_dflash2_weights(model_dir, config, arena, stream);
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

    auto reader_result = ps::io::SafetensorsReader::open(
        (case_dir / "fixture.safetensors").string());
    check(reader_result.ok(), "fixture open");
    if (!reader_result.ok()) return 1;
    ps::io::SafetensorsReader reader = reader_result.release();

    std::vector<bf16_t> noise_host;
    std::vector<bf16_t> feature_host;
    check(read_fixture_tensor(reader, "noise_embedding", noise_host), "noise");
    check(read_fixture_tensor(reader, "target_feature", feature_host), "target_feature");

    const uint32_t context_rows = 8u;
    const uint32_t block_rows = 8u;
    const uint32_t context_start = 4096u;
    const uint32_t block_start = context_start + context_rows;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);

    ps::gpu::Tensor noise;
    ps::gpu::Tensor feature;
    ps::gpu::Tensor chain_a;
    ps::gpu::Tensor chain_b;
    ps::gpu::Tensor final_out;
    check(alloc_bf16(arena, {block_rows, hidden}, noise), "noise alloc");
    check(alloc_bf16(arena, {context_rows, hidden}, feature), "feature alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, chain_a), "chain_a alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, chain_b), "chain_b alloc");
    check(alloc_bf16(arena, {block_rows, hidden}, final_out), "final alloc");
    check(hipMemcpy(noise.data<bf16_t>(), noise_host.data(),
                    noise_host.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "noise upload");
    check(hipMemcpy(feature.data<bf16_t>(), feature_host.data(),
                    feature_host.size() * sizeof(bf16_t), hipMemcpyHostToDevice) == hipSuccess,
          "feature upload");

    std::printf("gate51 trace case=%s block=%u context=%u\n", case_name.c_str(), block_rows,
                context_rows);

    const DFlash2Executor& cex = ex;
    const auto stage_ptr = [&](const std::string& name) -> const bf16_t* {
        if (name == "input_norm") return cex.layer_input_norm.data<bf16_t>();
        if (name == "attention_prepared") return cex.attention_prepared.data<bf16_t>();
        if (name == "q_raw") return cex.q_raw.data<bf16_t>();
        if (name == "q_norm") return cex.q_norm.data<bf16_t>();
        if (name == "q_rope") return cex.q_rope.data<bf16_t>();
        if (name == "k_ctx_raw") return cex.k_ctx_raw.data<bf16_t>();
        if (name == "k_ctx_norm") return cex.k_ctx_norm.data<bf16_t>();
        if (name == "k_ctx_rope") return cex.k_ctx_rope.data<bf16_t>();
        if (name == "v_ctx") return cex.v_ctx.data<bf16_t>();
        if (name == "k_noise_raw") return cex.k_noise_raw.data<bf16_t>();
        if (name == "k_noise_norm") return cex.k_noise_norm.data<bf16_t>();
        if (name == "k_noise_rope") return cex.k_noise_rope.data<bf16_t>();
        if (name == "v_noise") return cex.v_noise.data<bf16_t>();
        if (name == "attention_context") return cex.attention_context.data<bf16_t>();
        if (name == "attention_o") return cex.attention_o.data<bf16_t>();
        if (name == "attention_finished") return cex.attention_finished.data<bf16_t>();
        if (name == "attention_residual") return cex.attention_residual.data<bf16_t>();
        if (name == "post_attention_norm") return cex.post_attention_norm.data<bf16_t>();
        if (name == "mlp_prepared") return cex.mlp_prepared.data<bf16_t>();
        if (name == "mlp_gate") return cex.mlp_gate.data<bf16_t>();
        if (name == "mlp_up") return cex.mlp_up.data<bf16_t>();
        if (name == "mlp_swiglu") return cex.mlp_swiglu.data<bf16_t>();
        if (name == "mlp_down") return cex.mlp_down.data<bf16_t>();
        if (name == "mlp_finished") return cex.mlp_finished.data<bf16_t>();
        return nullptr;
    };

    const bf16_t* src = noise.data<bf16_t>();
    for (uint32_t li = 0; li < 5u; ++li) {
        bf16_t* dst = (li % 2u == 0u) ? chain_a.data<bf16_t>() : chain_b.data<bf16_t>();
        const ps::Status st = ps::qwen35::dflash2::dflash2_forward_layer_stateless(
            ex, li, src, feature.data<bf16_t>(), context_rows, context_start, block_start,
            block_rows, dst, stream);
        check(st.ok(), "layer " + std::to_string(li) + ": " + st.message());
        if (!st.ok()) return 1;
        check(hipStreamSynchronize(stream) == hipSuccess, "sync");
        const std::string prefix = "layer" + std::to_string(li) + "_";
        for (const StageSpec& spec : kStages) {
            const uint32_t rows = spec.context_rows ? context_rows : block_rows;
            const std::size_t count = static_cast<std::size_t>(rows) * spec.features;
            dump_tensor(ps_dir, prefix + spec.name, stage_ptr(spec.name), count, stream);
        }
        dump_tensor(ps_dir, prefix + "layer_output", dst,
                    static_cast<std::size_t>(block_rows) * hidden, stream);
        dump_tensor(ps_dir, prefix + "layer_input", src,
                    static_cast<std::size_t>(block_rows) * hidden, stream);
        src = dst;
    }

    const hipError_t e = ps::kernel::launch_dflash2_rmsnorm_direct_bf16(
        src, ex.weights->final_norm_weight.data<bf16_t>(), final_out.data<bf16_t>(),
        block_rows, hidden, hidden, config.rms_norm_eps, stream);
    check(e == hipSuccess, "final norm");
    check(hipStreamSynchronize(stream) == hipSuccess, "final sync");
    dump_tensor(ps_dir, "final_hidden", final_out.data<bf16_t>(),
                static_cast<std::size_t>(block_rows) * hidden, stream);

    const ps::Status st = ps::qwen35::dflash2::dflash2_executor_shutdown(ex);
    check(st.ok(), "executor shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    const ps::Status shutdown = arena.shutdown();
    check(shutdown.ok(), "arena shutdown");

    std::printf("test_dflash2_gate51_trace: failed=%d out=%s\n", g_fail,
                ps_dir.string().c_str());
    return g_fail == 0 ? 0 : 1;
}
