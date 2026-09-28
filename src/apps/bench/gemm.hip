#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/fp8_block128.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/kernels/dflash2/swiglu.h>
#include <phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

namespace {

struct Options {
    std::string dtype = "bf16";
    std::string variant = "auto";
    std::vector<uint32_t> rows;
    uint32_t n = 5120;
    uint32_t k = 5120;
    int warmup = 100;
    int samples = 50;
    int launches = 20;
    uint32_t seed = 1234;
    std::string out;
    int device = 0;
    bool check = false;
    std::string model_dir;
    bool actual_qwen_shapes = false;
    std::string impl = "auto";
    std::string shapes_spec;
    bool psq8_decode1 = false;
    uint32_t psq8_unroll = 0;
    bool psq4_decode1 = false;
    uint32_t psq4_unroll = 0;
    int psq4_prefill2d = -1;
    int psq8_prefill2d = -1;
    uint32_t weight_copies = 0;
    bool verify_small_m = false;
};

void usage(FILE* f) {
    fprintf(f,
        "phaseshift-bench gemm [options]\n"
        "\n"
        "  --dtype bf16|psq4|psq8|fp8|mxfp4   (logical weight/compute family, default bf16)\n"
        "  --variant correctness|gemm|auto\n"
        "      correctness: production correctness path (default auto)\n"
        "      gemm:        force Big-M GEMM compute kind\n"
        "      auto:        production linear selector\n"
        "  --rows 1,2,4,...               (comma list, default 1,16,32,64,128)\n"
        "  --n N                          (default 5120)\n"
        "  --k N                          (default 5120)\n"
        "  --shapes N:K,N:K,...           (explicit shape list for bf16/psq sweeps)\n"
        "  --impl auto|wmma|wide|splitk|exact   (bf16 forced implementation)\n"
        "  --model-dir PATH               (with --actual-qwen-shapes: read Qwen config)\n"
        "  --actual-qwen-shapes           (benchmark all deduped Qwen linear shapes)\n"
        "  --warmup N                     (default 100)\n"
        "  --samples N                    (default 50)\n"
        "  --launches N                   (default 20)\n"
        "  --seed N                       (default 1234)\n"
        "  --device N                     (default 0)\n"
        "  --psq8-decode1                 force the rows=1 decode1 PSQ8 kernel (A/B vs default)\n"
        "  --psq8-unroll N                decode1 unroll factor: 0=auto, 2, 4, 8, 16 (default 0)\n"
        "  --psq4-decode1                 force the rows=1 decode1 PSQ4 kernel (A/B vs default)\n"
        "  --psq4-unroll N                PSQ4 decode1 unroll factor: 0=auto, 2, 4, 8, 16 (default 0)\n"
        "  --psq4-prefill2d N             force PSQ4 prefill 2d config: 0=64x64, 1=K128N64, 2=K64N128, 3=K128N128 (-1=auto)\n"
        "  --psq8-prefill2d N             force PSQ8 prefill 2d config: 0=64x64, 1=K64N128, 2=K128N128 (-1=auto)\n"
        "  --weight-copies N              weight buffer copies cycled per launch (0=auto >=96MB, default 0)\n"
        "  --verify-small-m               benchmark the speculative verify rows 1,2,4,8\n"
        "  --check                        verify selected path against the production correctness path\n"
        "  --out PATH                     (append CSV rows)\n");
}

bool parse_rows(const std::string& s, std::vector<uint32_t>& out) {
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (cur.empty()) return false;
            uint32_t v = (uint32_t)std::stoul(cur);
            if (v == 0) return false;
            out.push_back(v);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        uint32_t v = (uint32_t)std::stoul(cur);
        if (v == 0) return false;
        out.push_back(v);
    }
    return !out.empty();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

float bf16_bits_to_f32(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

}  // namespace

struct Shape {
    std::string name;
    uint32_t n = 0;
    uint32_t k = 0;
};

std::vector<Shape> build_qwen_shapes(const ps::qwen35::Qwen35TextConfig& c) {
    const uint32_t hidden = (uint32_t)c.hidden_size;
    const uint32_t inter = (uint32_t)c.intermediate_size;
    const uint32_t qa = (uint32_t)c.num_attention_heads * (uint32_t)c.attention_head_dim;
    const uint32_t kva = (uint32_t)c.num_key_value_heads * (uint32_t)c.attention_head_dim;
    const uint32_t gqk = (uint32_t)c.linear_num_key_heads * (uint32_t)c.linear_key_head_dim;
    const uint32_t gv = (uint32_t)c.linear_num_value_heads * (uint32_t)c.linear_value_head_dim;
    const uint32_t nv = (uint32_t)c.linear_num_value_heads;
    const std::vector<Shape> raw = {
        {"mlp_gate_up", inter, hidden},
        {"mlp_down", hidden, inter},
        {"attn_q", qa, hidden},
        {"attn_kv", kva, hidden},
        {"attn_o", hidden, qa},
        {"gdn_qkv", 2u * gqk + gv, hidden},
        {"gdn_z", gv, hidden},
        {"gdn_ba", nv, hidden},
        {"gdn_out", hidden, gv},
    };
    std::vector<Shape> dedup;
    for (const auto& s : raw) {
        if (s.n == 0 || s.k == 0) continue;
        bool seen = false;
        for (const auto& d : dedup)
            if (d.n == s.n && d.k == s.k) {
                seen = true;
                break;
            }
        if (!seen) dedup.push_back(s);
    }
    return dedup;
}

struct CorrectnessEnv {
    void* weight_slot_dev = nullptr;
    void* values_dev = nullptr;
    void* ranges_dev = nullptr;
    void* ctx_dev = nullptr;
    float* scratch = nullptr;
    uint32_t* error_word = nullptr;
    ::ps::runtime::DeviceProgramView program;
    ~CorrectnessEnv() {
        ps::gpu::discard_cleanup_result(hipFree(weight_slot_dev));
        ps::gpu::discard_cleanup_result(hipFree(values_dev));
        ps::gpu::discard_cleanup_result(hipFree(ranges_dev));
        ps::gpu::discard_cleanup_result(hipFree(ctx_dev));
        ps::gpu::discard_cleanup_result(hipFree(scratch));
        ps::gpu::discard_cleanup_result(hipFree(error_word));
    }
};

bool gen_bf16_input(uint32_t rows, uint32_t k, uint32_t seed, ps::bf16_t** d_input) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ps::bf16_t> h((size_t)rows * k);
    for (auto& v : h) {
        const float fv = dist(rng);
        uint32_t ui;
        std::memcpy(&ui, &fv, 4);
        ui = (ui + 0x8000u) & 0xFFFF0000u;
        v = ps::bf16_t{(uint16_t)(ui >> 16)};
    }
    if (hipMalloc((void**)d_input, (size_t)rows * k * 2) != hipSuccess) return false;
    if (hipMemcpy(*d_input, h.data(), (size_t)rows * k * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    return true;
}

struct Bf16Weights {
    ps::bf16_t* d_weight = nullptr;
    ~Bf16Weights() {
        ps::gpu::discard_cleanup_result(hipFree(d_weight));
    }
};

bool gen_bf16_weights(uint32_t n, uint32_t k, uint32_t seed, Bf16Weights& w) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.02f, 0.02f);
    std::vector<uint16_t> h((size_t)n * k);
    auto f32b = [](float f) {
        uint32_t ui;
        std::memcpy(&ui, &f, 4);
        ui = (ui + 0x8000u) & 0xFFFF0000u;
        return (uint16_t)(ui >> 16);
    };
    for (auto& v : h) v = f32b(wd(rng));
    if (hipMalloc((void**)&w.d_weight, h.size() * 2) != hipSuccess) return false;
    if (hipMemcpy(w.d_weight, h.data(), h.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    return true;
}

bool setup_correctness_bf16(const Bf16Weights& w, const ps::bf16_t* d_input, ps::bf16_t* d_output,
                            uint32_t rows, uint32_t n, uint32_t k, CorrectnessEnv& env) {
    ::ps::runtime::WeightSlot slot{};
    slot.rows = n;
    slot.cols = k;
    slot.k_padded = k;
    slot.device_ptr = w.d_weight;
    slot.encoding = 0u;
    slot.compute_spec = 0u;

    if (hipMalloc(&env.weight_slot_dev, sizeof(::ps::runtime::WeightSlot)) != hipSuccess) return false;
    if (hipMemcpy(env.weight_slot_dev, &slot, sizeof(::ps::runtime::WeightSlot), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ::ps::runtime::DeviceValueBinding values[2];
    values[0] = {};
    values[0].logical_value = 0;
    values[0].storage = ::ps::runtime::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = ::ps::runtime::ValueDType::BF16;
    values[0].row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    values[0].slot = 0;
    values[0].feature_count = k;
    values[0].row_stride = k;
    values[0].offset = reinterpret_cast<uint64_t>(d_input);
    values[0].bytes = (uint64_t)rows * k * sizeof(ps::bf16_t);
    values[1] = {};
    values[1].logical_value = 1;
    values[1].storage = ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT;
    values[1].dtype = ::ps::runtime::ValueDType::BF16;
    values[1].row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    values[1].slot = 0;
    values[1].feature_count = n;
    values[1].row_stride = n;
    values[1].offset = reinterpret_cast<uint64_t>(d_output);
    values[1].bytes = (uint64_t)rows * n * sizeof(ps::bf16_t);

    if (hipMalloc(&env.values_dev, 2 * sizeof(::ps::runtime::DeviceValueBinding)) != hipSuccess) return false;
    if (hipMemcpy(env.values_dev, values, 2 * sizeof(::ps::runtime::DeviceValueBinding), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    env.program = {};
    env.program.values = static_cast<const ::ps::runtime::DeviceValueBinding*>(env.values_dev);
    env.program.value_count = 2;
    env.program.weight_table = env.weight_slot_dev;
    env.program.weight_slot_count = 1;

    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_outputs = 0;
    if (hipMalloc(&env.ctx_dev, sizeof(::ps::runtime::DeviceBatchContext)) != hipSuccess) return false;
    if (hipMemcpy(env.ctx_dev, &ctx, sizeof(::ps::runtime::DeviceBatchContext), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    if (hipMalloc(&env.scratch, (size_t)64 * 2048 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&env.error_word, sizeof(uint32_t)) != hipSuccess) return false;
    if (hipMemset(env.error_word, 0, sizeof(uint32_t)) != hipSuccess) return false;
    return true;
}

ps::Status launch_correctness_linear_bf16(const CorrectnessEnv& env, uint32_t rows, hipStream_t stream) {
    ::ps::runtime::DispatchBinding lbind{};
    lbind.kernel_id = ::ps::runtime::KernelId::LINEAR_BF16;
    lbind.weight_index = 0;
    lbind.input_slots[0] = 0;
    lbind.input_count = 1;
    lbind.output_slots[0] = 1;
    lbind.output_count = 1;
    ps::gpu::discard_cleanup_result(hipMemset(env.error_word, 0, sizeof(uint32_t)));
    return ps::kernel::launch_model_dispatch_correctness(
        env.program, lbind, static_cast<::ps::runtime::DeviceBatchContext*>(env.ctx_dev),
        ps::kernel::ModelDispatchStateView{}, env.scratch, env.error_word, stream);
}

ps::kernel::Bf16GemmConfig bf16_config_for(uint32_t rows, uint32_t n, uint32_t k) {
    using Id = ps::kernel::Bf16GemmConfigId;
    if (rows == 1u && (k & 7u) == 0u)
        return ps::kernel::Bf16GemmConfig{Id::ExactRows, 1u};
    if (n <= ps::kernel::kBf16GemmSplitKMaxOutFeatures)
        return ps::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
    if (rows >= ps::kernel::kBf16GemmWideMinRows)
        return ps::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
    return ps::kernel::Bf16GemmConfig{Id::Wmma, 0u};
}

const char* psq4_config_name(ps::kernel::Psq4GemmConfigId id) {
    using Id = ps::kernel::Psq4GemmConfigId;
    switch (id) {
        case Id::Prefill2D_K64N64: return "2d_ob64_kc64";
        case Id::Prefill2D_K128N64: return "2d_ob64_kc128";
        case Id::Prefill2D_K64N128: return "2d_ob128_kc64";
        case Id::Prefill2D_K128N128: return "2d_ob128_kc128";
        case Id::RowBlock8: return "rowblock8";
        case Id::RowBlock4: return "rowblock4";
        case Id::RowBlock2: return "rowblock2";
        default: return "rowblock1";
    }
}

const char* psq8_config_name(ps::kernel::Psq8GemmConfigId id) {
    using Id = ps::kernel::Psq8GemmConfigId;
    switch (id) {
        case Id::Prefill2D: return "2d_ob64_kc64";
        case Id::Prefill2D_K64N128: return "2d_ob128_kc64";
        case Id::Prefill2D_K128N128: return "2d_ob128_kc128";
        case Id::RowBlock8: return "rowblock8";
        case Id::RowBlock4: return "rowblock4";
        case Id::RowBlock2: return "rowblock2";
        default: return "rowblock1";
    }
}

ps::kernel::Psq4GemmConfig psq4_gemm_config_for(uint32_t rows, uint32_t n, uint32_t k_padded) {
    using Id = ps::kernel::Psq4GemmConfigId;
    const bool use_2d =
        (rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        n >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d) {
        const bool k128 = (k_padded % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        const bool n128 = (n % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        uint32_t cfg;
        if (rows >= ps::kernel::kPsq4GemmPrefill2dLargeMinRows &&
            n >= ps::kernel::kPsq4GemmPrefill2dLargeMinOutFeatures && k128 && n128)
            cfg = 3u;
        else if (k128)
            cfg = 1u;
        else if (n128)
            cfg = 2u;
        else
            cfg = 0u;
        const uint32_t ob =
            (cfg == 2u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                     : ps::kernel::kPsq4GemmPrefill2dBlock64;
        const uint32_t kc =
            (cfg == 1u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                     : ps::kernel::kPsq4GemmPrefill2dBlock64;
        if ((k_padded % kc) == 0u && (n % ob) == 0u) {
            switch (cfg) {
                case 1u: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K128N64};
                case 2u: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K64N128};
                case 3u: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K128N128};
                default: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K64N64};
            }
        }
        if ((n % ps::kernel::kPsq4GemmPrefill2dBlock64) == 0u &&
            (k_padded % ps::kernel::kPsq4GemmPrefill2dBlock64) == 0u)
            return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K64N64};
    }
    if (rows >= 8u * ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock2};
    return ps::kernel::Psq4GemmConfig{Id::RowBlock1};
}

ps::kernel::Psq8GemmConfig psq8_gemm_config_for(uint32_t rows, uint32_t n, uint32_t k_padded) {
    using Id = ps::kernel::Psq8GemmConfigId;
    const bool use_2d =
        (k_padded % ps::kernel::kPsq8GemmPrefill2dKChunk) == 0u &&
        (rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        (n % ps::kernel::kPsq8GemmPrefill2dOutBlock) == 0u &&
        rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        n >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d)
        return ps::kernel::Psq8GemmConfig{Id::Prefill2D};
    if (rows >= 8u * ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq8GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq8GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kPsqGemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq8GemmConfig{Id::RowBlock2};
    return ps::kernel::Psq8GemmConfig{Id::RowBlock1};
}

ps::kernel::Fp8Block128GemmConfig fp8_gemm_config_for(uint32_t rows, uint32_t n) {
    using Id = ps::kernel::Fp8Block128GemmConfigId;
    if (rows >= 8u * ps::kernel::kFp8Block128GemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kFp8Block128GemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock4};
    if (rows > ps::kernel::kFp8Block128GemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock2};
    return {Id::RowBlock1};
}

ps::kernel::Mxfp4GemmConfig mxfp4_gemm_config_for(uint32_t rows, uint32_t n) {
    using Id = ps::kernel::Mxfp4GemmConfigId;
    if (rows >= 8u * ps::kernel::kMxfp4GemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kMxfp4GemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock4};
    if (rows > ps::kernel::kMxfp4GemmRowTile &&
        n >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return {Id::RowBlock2};
    return {Id::RowBlock1};
}

int parse_shapes_spec(const std::string& spec, std::vector<Shape>& shapes) {
    std::string cur;
    std::vector<std::string> parts;
    for (char c : spec) {
        if (c == ',') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    parts.push_back(cur);
    for (const std::string& p : parts) {
        const size_t colon = p.find(':');
        if (colon == std::string::npos) {
            std::fprintf(stderr, "bad --shapes entry '%s'\n", p.c_str());
            return 2;
        }
        Shape s;
        s.name = p;
        s.n = (uint32_t)std::stoul(p.substr(0, colon));
        s.k = (uint32_t)std::stoul(p.substr(colon + 1));
        if (s.n == 0u || s.k == 0u) {
            std::fprintf(stderr, "bad --shapes entry '%s'\n", p.c_str());
            return 2;
        }
        shapes.push_back(s);
    }
    return 0;
}

int run_gemm_bf16(Options& o) {
    if (o.variant != "correctness" && o.variant != "gemm" && o.variant != "auto") {
        std::fprintf(stderr, "bf16 --variant must be correctness|gemm|auto\n");
        return 2;
    }
    std::vector<Shape> shapes;
    if (o.actual_qwen_shapes) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--actual-qwen-shapes requires --model-dir\n");
            return 2;
        }
        auto cfg_res = ps::qwen35::read_qwen35_text_config(o.model_dir);
        if (!cfg_res.ok()) {
            std::fprintf(stderr, "read_qwen35_text_config: %s\n", cfg_res.status().message().c_str());
            return 2;
        }
        shapes = build_qwen_shapes(cfg_res.release());
    } else if (!o.shapes_spec.empty()) {
        const int pr = parse_shapes_spec(o.shapes_spec, shapes);
        if (pr != 0) return pr;
    } else {
        shapes.push_back({"custom", o.n, o.k});
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes to benchmark\n");
        return 2;
    }

    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "dtype,requested_variant,selected_variant,shape_name,rows,n,k,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const auto& shape : shapes) {
        Bf16Weights w;
        if (!gen_bf16_weights(shape.n, shape.k, o.seed, w)) {
            std::fprintf(stderr, "gen weights failed %s\n", shape.name.c_str());
            rc = 1;
            break;
        }
        for (uint32_t rows : o.rows) {
            ps::bf16_t* d_input = nullptr;
            ps::bf16_t* d_output = nullptr;
            bool ok = gen_bf16_input(rows, shape.k, o.seed + 7777u, &d_input);
            ok = ok && hipMalloc((void**)&d_output, (size_t)rows * shape.n * 2) == hipSuccess;
            if (!ok) {
                std::fprintf(stderr, "alloc failed %s rows=%u\n", shape.name.c_str(), rows);
                rc = 1;
                break;
            }
            CorrectnessEnv env;
            if (!setup_correctness_bf16(w, d_input, d_output, rows, shape.n, shape.k, env)) {
                std::fprintf(stderr, "setup correctness failed %s rows=%u\n", shape.name.c_str(), rows);
                rc = 1;
                break;
            }
            std::string eff = o.variant;
            ps::kernel::Bf16GemmConfig bf16_cfg = bf16_config_for(rows, shape.n, shape.k);
            if (o.impl != "auto") {
                using Id = ps::kernel::Bf16GemmConfigId;
                if (o.impl == "wmma")
                    bf16_cfg = ps::kernel::Bf16GemmConfig{Id::Wmma, 0u};
                else if (o.impl == "wide")
                    bf16_cfg = ps::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
                else if (o.impl == "splitk")
                    bf16_cfg = ps::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
                else
                    bf16_cfg = ps::kernel::Bf16GemmConfig{Id::ExactRows,
                                                          static_cast<uint8_t>(rows)};
                eff = "gemm";
            } else if (eff == "auto") {
                const bool supported = ps::qwen35::runtime::linear_shape_supported(
                    ps::qwen35::runtime::LinearComputeFamily::Bf16, shape.n, shape.k);
                eff = supported ? "gemm" : "correctness";
            }
            auto launch_one = [&]() -> hipError_t {
                if (eff == "gemm")
                    return ps::kernel::launch_gemm_bf16(
                        bf16_cfg, w.d_weight, d_input, d_output,
                        ps::kernel::Bf16GemmOutputDType::BF16, rows, shape.n, shape.k,
                        shape.k, shape.n, stream);
                ps::Status st = launch_correctness_linear_bf16(env, rows, stream);
                return st.ok() ? hipSuccess : hipErrorUnknown;
            };
            if (o.check) {
                const size_t outn = (size_t)rows * shape.n;
                std::vector<uint16_t> h_c(outn), h_g(outn);
                ps::Status stc = launch_correctness_linear_bf16(env, rows, stream);
                uint32_t ew = 0;
                if (!stc.ok() || hipStreamSynchronize(stream) != hipSuccess ||
                    hipMemcpy(&ew, env.error_word, sizeof(ew), hipMemcpyDeviceToHost) != hipSuccess || ew != 0) {
                    std::fprintf(stderr, "CHECK FAIL %s rows=%u correctness (err=%u)\n", shape.name.c_str(), rows, ew);
                    rc = 1;
                    break;
                }
                ps::gpu::discard_cleanup_result(hipMemcpy(h_c.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                if (launch_one() != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
                    std::fprintf(stderr, "CHECK FAIL %s rows=%u %s launch\n", shape.name.c_str(), rows, o.variant.c_str());
                    rc = 1;
                    break;
                }
                ps::gpu::discard_cleanup_result(hipMemcpy(h_g.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                uint64_t mismatch = 0, nan = 0, inf = 0;
                float max_rel = 0.f;
                for (size_t i = 0; i < outn; ++i) {
                    if (h_c[i] != h_g[i]) ++mismatch;
                    const float fc = bf16_bits_to_f32(h_c[i]);
                    const float fg = bf16_bits_to_f32(h_g[i]);
                    if (std::isnan(fc) || std::isnan(fg)) ++nan;
                    if (std::isinf(fc) || std::isinf(fg)) ++inf;
                    const float rel = std::fabs(fc - fg) / std::max(1.0f, std::fabs(fc));
                    if (rel > max_rel) max_rel = rel;
                }
                const bool okc = nan == 0 && inf == 0 && max_rel <= 2e-2f;
                std::printf("CHECK %-12s rows=%-4u n=%-6u k=%-6u mismatch=%llu max_rel=%.4g nan=%llu inf=%llu %s\n",
                            shape.name.c_str(), rows, shape.n, shape.k, (unsigned long long)mismatch,
                            max_rel, (unsigned long long)nan, (unsigned long long)inf, okc ? "OK" : "FAIL");
                if (!okc) rc = 1;
            }
            for (int i = 0; i < o.warmup; ++i) {
                if (launch_one() != hipSuccess) {
                    std::fprintf(stderr, "warmup launch failed %s rows=%u\n", shape.name.c_str(), rows);
                    rc = 1;
                    break;
                }
            }
            if (rc) break;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; break; }
            std::vector<double> per_launch_us;
            per_launch_us.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) { rc = 1; break; }
                for (int l = 0; l < o.launches; ++l) {
                    if (launch_one() != hipSuccess) { rc = 1; break; }
                }
                if (rc) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) { rc = 1; break; }
                if (hipEventSynchronize(ev1) != hipSuccess) { rc = 1; break; }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { rc = 1; break; }
                per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
            }
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            if (rc) break;
            if (!per_launch_us.empty()) {
                const double mn = percentile(per_launch_us, 0.0);
                const double p50 = percentile(per_launch_us, 0.5);
                const double p95 = percentile(per_launch_us, 0.95);
                double mean = 0.0;
                for (double v : per_launch_us) mean += v;
                mean /= (double)per_launch_us.size();
                std::printf("bf16 %-12s rows=%-4u n=%-6u k=%-6u requested=%s selected=%s min=%.3f p50=%.3f p95=%.3f mean=%.3f us\n",
                            shape.name.c_str(), rows, shape.n, shape.k, o.variant.c_str(), eff.c_str(), mn, p50, p95, mean);
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "%s,%s,%s,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                     o.dtype.c_str(), o.variant.c_str(), eff.c_str(), shape.name.c_str(),
                                     rows, shape.n, shape.k,
                                     o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                        std::fclose(f);
                    }
                }
            }
            ps::gpu::discard_cleanup_result(hipFree(d_input));
            ps::gpu::discard_cleanup_result(hipFree(d_output));
            if (rc) break;
        }
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}

int run_gemm_psq(Options& o, bool psq4) {
    std::vector<Shape> shapes;
    if (o.actual_qwen_shapes) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--actual-qwen-shapes requires --model-dir\n");
            return 2;
        }
        auto cfg_res = ps::qwen35::read_qwen35_text_config(o.model_dir);
        if (!cfg_res.ok()) {
            std::fprintf(stderr, "read_qwen35_text_config: %s\n", cfg_res.status().message().c_str());
            return 2;
        }
        shapes = build_qwen_shapes(cfg_res.release());
    } else if (!o.shapes_spec.empty()) {
        const int pr = parse_shapes_spec(o.shapes_spec, shapes);
        if (pr != 0) return pr;
    } else {
        shapes.push_back({"custom", o.n, o.k});
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes to benchmark\n");
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "dtype,requested_variant,selected_variant,shape_name,rows,n,k,warmup,samples,launches_per_sample,min_us,p10_us,p50_us,p90_us,p95_us,mean_us,code_GBs,code_scale_GBs\n");
            std::fclose(f);
        }
    }

    const char* dtype_name = psq4 ? "psq4" : "psq8";
    int rc = 0;
    for (const auto& shape : shapes) {
        const uint32_t kp = (shape.k + 31u) & ~31u;
        const uint32_t nb = kp >> 5;
        const uint32_t n = shape.n;
        const uint32_t n_tiled = ((n + 15u) & ~15u);
        const size_t codes_bytes = (size_t)n_tiled * (psq4 ? (kp >> 1) : kp);
        const size_t scales_bytes = (size_t)((n + 15u) / 16u) * nb * 32u;
        const size_t per_copy = codes_bytes + scales_bytes;
        uint32_t copies = o.weight_copies;
        if (copies == 0u) {
            const size_t target = 96ull * 1024ull * 1024ull;
            copies = (uint32_t)((target + per_copy - 1u) / per_copy);
            if (copies < 1u) copies = 1u;
            if (copies > 8u) copies = 8u;
        }
        uint8_t* d_codes = nullptr;
        uint8_t* d_scales = nullptr;
        if (hipMalloc((void**)&d_codes, (size_t)copies * codes_bytes) != hipSuccess ||
            hipMalloc((void**)&d_scales, (size_t)copies * scales_bytes) != hipSuccess) {
            return 1;
        }
        {
            std::mt19937 rng(o.seed + (psq4 ? 11u : 22u));
            std::vector<uint8_t> hc(codes_bytes);
            for (auto& v : hc) v = (uint8_t)(rng() & 0xFFu);
            std::vector<uint8_t> hs(scales_bytes, 0);
            for (size_t i = 0; i + 1 < hs.size(); i += 2) {
                const uint16_t b = (uint16_t)(0x3C00u | (rng() & 0x00FFu));
                hs[i] = (uint8_t)(b & 0xFFu);
                hs[i + 1] = (uint8_t)(b >> 8u);
            }
            for (uint32_t cp = 0; cp < copies; ++cp) {
                if (hipMemcpy(d_codes + (size_t)cp * codes_bytes, hc.data(), hc.size(),
                              hipMemcpyHostToDevice) != hipSuccess ||
                    hipMemcpy(d_scales + (size_t)cp * scales_bytes, hs.data(), hs.size(),
                              hipMemcpyHostToDevice) != hipSuccess) {
                    return 1;
                }
            }
        }
        for (uint32_t rows : o.rows) {
            ps::bf16_t* d_input = nullptr;
            ps::bf16_t* d_output = nullptr;
            void* a8_base = nullptr;
            const uint32_t ws_rows = (rows + 127u) & ~127u;
            const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(shape.k, ws_rows);
            bool ok = gen_bf16_input(rows, shape.k, o.seed + 7777u, &d_input);
            ok = ok && hipMalloc(&a8_base, layout.total_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_output, (size_t)rows * n * 2) == hipSuccess;
            if (!ok) { rc = 1; break; }
            uint8_t* a8_codes = reinterpret_cast<uint8_t*>(static_cast<char*>(a8_base) + layout.codes_offset);
            float* a8_scales = reinterpret_cast<float*>(static_cast<char*>(a8_base) + layout.scales_offset);
            hipError_t af = ps::kernel::launch_activation_quantize_e4m3(
                d_input, a8_codes, a8_scales, rows, shape.k, kp, shape.k,
                layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
            if (af != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            const auto psq4_sel = ps::qwen35::runtime::select_psq4_gemm_config(
                ps::qwen35::runtime::Psq4GemmSelectorInput{rows, n, shape.k, kp});
            const auto psq8_sel = ps::qwen35::runtime::select_psq8_gemm_config(
                ps::qwen35::runtime::Psq8GemmSelectorInput{rows, n, shape.k, kp});
            const ps::kernel::Psq4GemmConfig psq4_cfg =
                o.psq4_prefill2d >= 0
                    ? ps::kernel::Psq4GemmConfig{
                          static_cast<ps::kernel::Psq4GemmConfigId>(
                              static_cast<uint8_t>(ps::kernel::Psq4GemmConfigId::Prefill2D_K64N64) +
                              static_cast<uint8_t>(o.psq4_prefill2d))}
                    : (psq4_sel ? *psq4_sel : psq4_gemm_config_for(rows, n, kp));
            const ps::kernel::Psq8GemmConfig psq8_cfg =
                o.psq8_prefill2d >= 0
                    ? ps::kernel::Psq8GemmConfig{
                          static_cast<ps::kernel::Psq8GemmConfigId>(
                              static_cast<uint8_t>(ps::kernel::Psq8GemmConfigId::Prefill2D) +
                              static_cast<uint8_t>(o.psq8_prefill2d))}
                    : (psq8_sel ? *psq8_sel : psq8_gemm_config_for(rows, n, kp));
            uint32_t copy_idx = 0;
            auto launch_one = [&]() -> hipError_t {
                const uint8_t* wc = d_codes + (size_t)(copy_idx % copies) * codes_bytes;
                const uint8_t* wsc = d_scales + (size_t)(copy_idx % copies) * scales_bytes;
                ++copy_idx;
                if (psq4)
                    return o.psq4_decode1
                        ? ps::kernel::launch_gemm_psq4_w4a8_wmma_decode1(
                              wc, wsc, a8_codes, a8_scales, d_output,
                              ps::kernel::Psq4GemmOutputDType::BF16, rows, n, kp,
                              nb * 32u, layout.code_row_stride_bytes,
                              layout.scale_row_stride_bytes, n, o.psq4_unroll, stream)
                        : ps::kernel::launch_gemm_psq4_w4a8_wmma(
                              psq4_cfg, wc, wsc, reinterpret_cast<const int8_t*>(a8_codes),
                              a8_scales, d_output, ps::kernel::Psq4GemmOutputDType::BF16,
                              rows, n, kp, nb * 32u, layout.code_row_stride_bytes,
                              layout.scale_row_stride_bytes, n, stream);
                if (o.psq8_decode1)
                    return ps::kernel::launch_gemm_psq8_w8a8_wmma_decode1(
                        wc, wsc, a8_codes, a8_scales, d_output,
                        ps::kernel::Psq8GemmOutputDType::BF16, rows, n, kp,
                        nb * 32u, layout.code_row_stride_bytes, layout.scale_row_stride_bytes,
                        n, o.psq8_unroll, stream);
                return ps::kernel::launch_gemm_psq8_w8a8_wmma(
                    psq8_cfg, wc, wsc, reinterpret_cast<const int8_t*>(a8_codes),
                    a8_scales, d_output, ps::kernel::Psq8GemmOutputDType::BF16, rows, n, kp,
                    nb * 32u, layout.code_row_stride_bytes, layout.scale_row_stride_bytes,
                    n, stream);
            };
            for (int i = 0; i < o.warmup; ++i)
                if (launch_one() != hipSuccess) { rc = 1; break; }
            if (rc) break;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; break; }
            std::vector<double> per_launch_us;
            per_launch_us.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) { rc = 1; break; }
                for (int l = 0; l < o.launches; ++l)
                    if (launch_one() != hipSuccess) { rc = 1; break; }
                if (rc) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) { rc = 1; break; }
                if (hipEventSynchronize(ev1) != hipSuccess) { rc = 1; break; }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { rc = 1; break; }
                per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
            }
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            ps::gpu::discard_cleanup_result(hipFree(d_input));
            ps::gpu::discard_cleanup_result(hipFree(a8_base));
            ps::gpu::discard_cleanup_result(hipFree(d_output));
            if (rc) break;
            if (!per_launch_us.empty()) {
                const double mn = percentile(per_launch_us, 0.0);
                const double p10 = percentile(per_launch_us, 0.1);
                const double p50 = percentile(per_launch_us, 0.5);
                const double p90 = percentile(per_launch_us, 0.9);
                const double p95 = percentile(per_launch_us, 0.95);
                double mean = 0.0;
                for (double v : per_launch_us) mean += v;
                mean /= (double)per_launch_us.size();
                const double code_gbs = (p50 > 0.0)
                    ? (double)codes_bytes / (p50 * 1.0e-6) / 1.0e9 : 0.0;
                const double both_gbs = (p50 > 0.0)
                    ? (double)(codes_bytes + scales_bytes) / (p50 * 1.0e-6) / 1.0e9 : 0.0;
                std::printf("%s %-12s rows=%-4u n=%-6u k=%-6u requested=%s selected=%s p10=%.3f p50=%.3f p90=%.3f mean=%.3f us "
                            "code_MB=%.2f scale_MB=%.2f code_GBs=%.0f code+scale_GBs=%.0f\n",
                            dtype_name, shape.name.c_str(), rows, shape.n, shape.k,
                            o.variant.c_str(),
                            psq4 ? psq4_config_name(psq4_cfg.id) : psq8_config_name(psq8_cfg.id),
                            p10, p50, p90, mean,
                            (double)codes_bytes / 1.0e6, (double)scales_bytes / 1.0e6,
                            code_gbs, both_gbs);
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "%s,%s,%s,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f\n",
                                     dtype_name, o.variant.c_str(),
                                     psq4 ? psq4_config_name(psq4_cfg.id) : psq8_config_name(psq8_cfg.id),
                                     shape.name.c_str(),
                                     rows, shape.n, shape.k,
                                     o.warmup, o.samples, o.launches, mn, p10, p50, p90, p95, mean,
                                     code_gbs, both_gbs);
                        std::fclose(f);
                    }
                }
            }
        }
        ps::gpu::discard_cleanup_result(hipFree(d_codes));
        ps::gpu::discard_cleanup_result(hipFree(d_scales));
        if (rc) break;
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}

enum class BlockScaledBenchKind { Fp8, Mxfp4 };

int run_gemm_block_scaled(Options& o, BlockScaledBenchKind kind) {
    std::vector<Shape> shapes;
    if (o.actual_qwen_shapes) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--actual-qwen-shapes requires --model-dir\n");
            return 2;
        }
        auto cfg_res = ps::qwen35::read_qwen35_text_config(o.model_dir);
        if (!cfg_res.ok()) {
            std::fprintf(stderr, "read_qwen35_text_config: %s\n", cfg_res.status().message().c_str());
            return 2;
        }
        shapes = build_qwen_shapes(cfg_res.release());
    } else {
        shapes.push_back({"custom", o.n, o.k});
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes to benchmark\n");
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "dtype,requested_variant,selected_variant,shape_name,rows,n,k,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    const bool is_fp8 = kind == BlockScaledBenchKind::Fp8;
    const char* dtype_name = is_fp8 ? "fp8" : "mxfp4";
    int rc = 0;
    for (const auto& shape : shapes) {
        const uint32_t kp = is_fp8 ? ((shape.k + 127u) & ~127u) : ((shape.k + 31u) & ~31u);
        const uint32_t nb = kp >> 5;
        const uint32_t n = shape.n;
        const uint32_t n_tiled = (n + 15u) & ~15u;
        const size_t codes_bytes = is_fp8
            ? static_cast<size_t>(n_tiled) * nb * 512u
            : static_cast<size_t>(n_tiled) * nb * 256u;
        const size_t scales_bytes = is_fp8
            ? static_cast<size_t>(((n + 127u) / 128u)) * (kp / 128u) * 4u
            : static_cast<size_t>(n) * nb;
        uint8_t* d_codes = nullptr;
        uint8_t* d_scales = nullptr;
        if (hipMalloc((void**)&d_codes, codes_bytes) != hipSuccess ||
            hipMalloc((void**)&d_scales, scales_bytes) != hipSuccess) {
            return 1;
        }
        {
            std::mt19937 rng(o.seed + (is_fp8 ? 33u : 44u));
            std::vector<uint8_t> hc(codes_bytes);
            for (auto& v : hc) v = static_cast<uint8_t>(rng() & 0xFFu);
            std::vector<uint8_t> hs(scales_bytes, 0);
            if (is_fp8) {
                float* hsf = reinterpret_cast<float*>(hs.data());
                for (size_t i = 0; i < hs.size() / 4u; ++i)
                    hsf[i] = 0.01f + static_cast<float>(rng() % 1000u) / 100000.0f;
            } else {
                for (auto& v : hs) v = static_cast<uint8_t>(120u + (rng() % 16u));
            }
            if (hipMemcpy(d_codes, hc.data(), hc.size(), hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(d_scales, hs.data(), hs.size(), hipMemcpyHostToDevice) != hipSuccess) {
                return 1;
            }
        }
        for (uint32_t rows : o.rows) {
            ps::bf16_t* d_input = nullptr;
            ps::bf16_t* d_output = nullptr;
            uint8_t* d_acodes = nullptr;
            float* d_ascales = nullptr;
            const size_t acodes_bytes = static_cast<size_t>((rows + 15u) / 16u) * kp * 16u;
            const size_t ascales_bytes = static_cast<size_t>(rows) * 64u;
            bool ok = gen_bf16_input(rows, shape.k, o.seed + 7777u, &d_input);
            ok = ok && hipMalloc((void**)&d_acodes, acodes_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_ascales, ascales_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_output, static_cast<size_t>(rows) * n * 2) == hipSuccess;
            if (!ok) { rc = 1; break; }
            hipError_t af = ps::kernel::launch_activation_quantize_e4m3(
                d_input, d_acodes, d_ascales, rows, shape.k, kp, shape.k, kp, 64u, stream);
            if (af != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            const int8_t* a8 = reinterpret_cast<const int8_t*>(d_acodes);
            const auto fp8_sel = ps::qwen35::runtime::select_fp8_block128_gemm_config(
                ps::qwen35::runtime::Fp8Block128GemmSelectorInput{rows, n, shape.k, kp});
            const auto mxfp4_sel = ps::qwen35::runtime::select_mxfp4_gemm_config(
                ps::qwen35::runtime::Mxfp4GemmSelectorInput{rows, n, shape.k, kp});
            const ps::kernel::Fp8Block128GemmConfig fp8_cfg =
                fp8_sel ? *fp8_sel : fp8_gemm_config_for(rows, n);
            const ps::kernel::Mxfp4GemmConfig mxfp4_cfg =
                mxfp4_sel ? *mxfp4_sel : mxfp4_gemm_config_for(rows, n);
            auto launch_one = [&]() -> hipError_t {
                if (is_fp8) {
                    return ps::kernel::launch_gemm_fp8_block128_w8a8_wmma(
                        fp8_cfg, d_codes,
                        reinterpret_cast<const float*>(d_scales), a8, d_ascales, d_output,
                        ps::kernel::Fp8Block128GemmOutputDType::BF16, rows, n, kp,
                        (kp / 128u) * 4u, kp, 64u, n, stream);
                }
                return ps::kernel::launch_gemm_mxfp4_w4a8_wmma(
                    mxfp4_cfg, d_codes, d_scales, a8, d_ascales, d_output,
                    ps::kernel::Mxfp4GemmOutputDType::BF16, rows, n, kp, nb, kp, 64u, n, stream);
            };
            for (int i = 0; i < o.warmup; ++i)
                if (launch_one() != hipSuccess) { rc = 1; break; }
            if (rc) break;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; break; }
            std::vector<double> per_launch_us;
            per_launch_us.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) { rc = 1; break; }
                for (int l = 0; l < o.launches; ++l)
                    if (launch_one() != hipSuccess) { rc = 1; break; }
                if (rc) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) { rc = 1; break; }
                if (hipEventSynchronize(ev1) != hipSuccess) { rc = 1; break; }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { rc = 1; break; }
                per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
            }
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            ps::gpu::discard_cleanup_result(hipFree(d_input));
            ps::gpu::discard_cleanup_result(hipFree(d_acodes));
            ps::gpu::discard_cleanup_result(hipFree(d_ascales));
            ps::gpu::discard_cleanup_result(hipFree(d_output));
            if (rc) break;
            if (!per_launch_us.empty()) {
                const double mn = percentile(per_launch_us, 0.0);
                const double p50 = percentile(per_launch_us, 0.5);
                const double p95 = percentile(per_launch_us, 0.95);
                double mean = 0.0;
                for (double v : per_launch_us) mean += v;
                mean /= (double)per_launch_us.size();
                std::printf("%s %-12s rows=%-4u n=%-6u k=%-6u requested=%s selected=gemm min=%.3f p50=%.3f p95=%.3f mean=%.3f us\n",
                            dtype_name, shape.name.c_str(), rows, shape.n, shape.k,
                            o.variant.c_str(), mn, p50, p95, mean);
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "%s,%s,gemm,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                     dtype_name, o.variant.c_str(), shape.name.c_str(),
                                     rows, shape.n, shape.k,
                                     o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                        std::fclose(f);
                    }
                }
            }
        }
        ps::gpu::discard_cleanup_result(hipFree(d_codes));
        ps::gpu::discard_cleanup_result(hipFree(d_scales));
        if (rc) break;
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}

int run_gemm(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--dtype") o.dtype = next();
        else if (a == "--variant") o.variant = next();
        else if (a == "--rows") { if (!parse_rows(next(), o.rows)) { fprintf(stderr, "bad --rows\n"); return 2; } }
        else if (a == "--n") o.n = (uint32_t)std::stoul(next());
        else if (a == "--k") o.k = (uint32_t)std::stoul(next());
        else if (a == "--impl") o.impl = next();
        else if (a == "--shapes") o.shapes_spec = next();
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--samples") o.samples = std::stoi(next());
        else if (a == "--launches") o.launches = std::stoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--out") o.out = next();
        else if (a == "--psq8-decode1") o.psq8_decode1 = true;
        else if (a == "--psq8-unroll") o.psq8_unroll = (uint32_t)std::stoul(next());
        else if (a == "--psq4-decode1") o.psq4_decode1 = true;
        else if (a == "--psq4-unroll") o.psq4_unroll = (uint32_t)std::stoul(next());
        else if (a == "--psq4-prefill2d") o.psq4_prefill2d = std::stoi(next());
        else if (a == "--psq8-prefill2d") o.psq8_prefill2d = std::stoi(next());
        else if (a == "--weight-copies") o.weight_copies = (uint32_t)std::stoul(next());
        else if (a == "--verify-small-m") o.verify_small_m = true;
        else if (a == "--check") o.check = true;
        else if (a == "--model-dir") o.model_dir = next();
        else if (a == "--actual-qwen-shapes") o.actual_qwen_shapes = true;
        else if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(stderr); return 2; }
    }
    if (o.verify_small_m) o.rows = {1, 2, 4, 8};
    if (o.rows.empty()) { o.rows = {1, 16, 32, 64, 128}; }
    if (o.warmup < 0 || o.samples <= 0 || o.launches <= 0) {
        fprintf(stderr, "bad --warmup/--samples/--launches\n");
        return 2;
    }
    if (o.dtype != "bf16" && o.dtype != "psq4" && o.dtype != "psq8" &&
        o.dtype != "fp8" && o.dtype != "mxfp4") {
        fprintf(stderr, "valid --dtype: bf16|psq4|psq8|fp8|mxfp4 (got %s)\n", o.dtype.c_str());
        return 2;
    }
    if (o.variant != "correctness" && o.variant != "gemm" && o.variant != "auto") {
        fprintf(stderr, "valid --variant: correctness|gemm|auto (got %s)\n", o.variant.c_str());
        return 2;
    }
    if (o.dtype == "psq4") return run_gemm_psq(o, true);
    if (o.dtype == "psq8") return run_gemm_psq(o, false);
    if (o.dtype == "fp8") return run_gemm_block_scaled(o, BlockScaledBenchKind::Fp8);
    if (o.dtype == "mxfp4") return run_gemm_block_scaled(o, BlockScaledBenchKind::Mxfp4);
    return run_gemm_bf16(o);
}
