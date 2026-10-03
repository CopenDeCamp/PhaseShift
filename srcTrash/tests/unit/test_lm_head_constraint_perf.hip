#include "support/target_lm_proxy_harness.h"

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/sampling.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/lm_head_proxy.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* label) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", label);
    }
}

struct Bench {
    hipStream_t stream = nullptr;
    hipEvent_t a = nullptr;
    hipEvent_t b = nullptr;
    void begin() {
        if (a == nullptr) {
            (void)hipEventCreate(&a);
            (void)hipEventCreate(&b);
        }
        (void)hipEventRecord(a, stream);
    }
    double end_ms() {
        (void)hipEventRecord(b, stream);
        (void)hipStreamSynchronize(stream);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, a, b);
        return static_cast<double>(ms);
    }
    ~Bench() {
        if (a != nullptr) (void)hipEventDestroy(a);
        if (b != nullptr) (void)hipEventDestroy(b);
    }
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2u];
}

void build_mask(std::vector<uint32_t>& mask, uint32_t vocab, uint32_t count,
                uint32_t seed) {
    const uint32_t words = (vocab + 31u) / 32u;
    mask.assign(words, 0u);
    uint32_t state = seed * 1664525u + 1013904223u;
    uint32_t placed = 0u;
    while (placed < count) {
        state = state * 1664525u + 1013904223u;
        const uint32_t token = (state >> 8) % vocab;
        const uint32_t bit = 1u << (token & 31u);
        if ((mask[token >> 5u] & bit) == 0u) {
            mask[token >> 5u] |= bit;
            ++placed;
        }
    }
}

uint32_t mask_count(const std::vector<uint32_t>& mask, uint32_t vocab) {
    uint32_t n = 0u;
    for (uint32_t t = 0u; t < vocab; ++t)
        if (((mask[t >> 5u] >> (t & 31u)) & 1u) != 0u) ++n;
    return n;
}

}  // namespace

int main() {
    const char* target_dir = std::getenv("PHASESHIFT_MODEL_DIR_DFLASH2_TARGET");
    if (target_dir == nullptr || target_dir[0] == '\0') {
        std::printf("SKIP: set PHASESHIFT_MODEL_DIR_DFLASH2_TARGET\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    const uint32_t rows = 4u;
    const uint32_t warmup = 20u;
    const uint32_t iters = 100u;
    const uint32_t counts[] = {1u,    4u,    8u,    16u,   32u,    48u,
                               64u,   96u,   128u,  129u,  256u,   512u,
                               1024u, 4096u, 16384u};

    auto arena_result = ps::gpu::GpuArena::create(0, 30ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    auto model_result =
        ps::qwen35::Qwen35Model::load_from_safetensors(target_dir, arena, stream);
    if (!model_result.ok()) {
        std::printf("FAIL model load: %s\n", model_result.status().message().c_str());
        return 1;
    }
    ps::qwen35::Qwen35Model model = model_result.release();
    const ps::weights::MatrixWeight& lm = model.weights().lm_head;
    const uint32_t vocab = lm.rows;
    const uint32_t cols = lm.cols;
    const uint32_t k_padded = lm.k_padded;
    const uint32_t wscale_stride = lm.storage_scale_stride_bytes;

    if (lm.encoding != ps::weights::MatrixEncoding::Psq8 || !lm.preshuffled ||
        lm.weight_scale_group != 32u || (k_padded % 32u) != 0u) {
        std::printf("SKIP: lm_head is not preshuffled psq8 w32\n");
        return 77;
    }

    ps::qwen35::runtime::LmHeadCandidateProxy proxy;
    ps::qwen35::runtime::LmHeadProxyWeightView view;
    view.codes = lm.codes.data<uint8_t>();
    view.scales = lm.scales.data<uint8_t>();
    view.vocab = vocab;
    view.cols = cols;
    view.k_padded = k_padded;
    view.scale_stride_bytes = wscale_stride;
    const uint32_t pool = ps::qwen35::runtime::target_lm_head_proxy_pool();
    const ps::Status ist = proxy.init(view, rows, pool, stream);
    if (!ist.ok()) {
        std::printf("FAIL proxy init: %s\n", ist.message().c_str());
        return 1;
    }

    const auto layout = ps::runtime::Int8ActivationWorkspaceLayout::make(k_padded, rows);
    const uint32_t act_code_stride = layout.code_row_stride_bytes;
    const uint32_t act_scale_stride = layout.scale_row_stride_bytes;
    const uint32_t mask_words = (vocab + 31u) / 32u;

    ps::bf16_t* normed = nullptr;
    float* full_logits = nullptr;
    int32_t* out_a = nullptr;
    int32_t* out_b = nullptr;
    uint32_t* dev_mask = nullptr;
    uint8_t* act_codes = nullptr;
    float* act_scales = nullptr;
    ::ps::runtime::DeviceSamplingParams* dev_sampling = nullptr;
    bool alloc_ok =
        hipMalloc(&normed, static_cast<std::size_t>(rows) * cols * sizeof(ps::bf16_t)) ==
        hipSuccess;
    alloc_ok = alloc_ok &&
               hipMalloc(&full_logits, static_cast<std::size_t>(rows) * vocab * sizeof(float)) ==
                   hipSuccess;
    alloc_ok = alloc_ok && hipMalloc(&out_a, rows * sizeof(int32_t)) == hipSuccess;
    alloc_ok = alloc_ok && hipMalloc(&out_b, rows * sizeof(int32_t)) == hipSuccess;
    alloc_ok = alloc_ok &&
               hipMalloc(&dev_mask, rows * mask_words * sizeof(uint32_t)) == hipSuccess;
    alloc_ok = alloc_ok &&
               hipMalloc(&act_codes,
                         static_cast<std::size_t>(act_code_stride) *
                             ((rows + 15u) / 16u) * 16u) == hipSuccess;
    alloc_ok = alloc_ok &&
               hipMalloc(&act_scales, static_cast<std::size_t>(act_scale_stride) * rows) ==
                   hipSuccess;
    alloc_ok = alloc_ok &&
               hipMalloc(&dev_sampling,
                         rows * sizeof(::ps::runtime::DeviceSamplingParams)) == hipSuccess;
    if (!alloc_ok) {
        std::printf("FAIL alloc\n");
        return 1;
    }
    {
        std::vector<::ps::runtime::DeviceSamplingParams> host_s(rows);
        for (auto& p : host_s) {
            p.mode = 1u;
            p.temperature = 0.0f;
            p.reserved[0] = 1u;
        }
        (void)hipMemcpy(dev_sampling, host_s.data(),
                        rows * sizeof(::ps::runtime::DeviceSamplingParams),
                        hipMemcpyHostToDevice);
    }

    {
        std::vector<ps::bf16_t> host(static_cast<std::size_t>(rows) * cols);
        uint32_t state = 0x12345678u;
        for (auto& h : host) {
            state = state * 1664525u + 1013904223u;
            const float v =
                static_cast<float>(static_cast<int32_t>((state >> 8) % 2001u) - 1000u) /
                500.0f;
            h = ps::bf16_t(v);
        }
        (void)hipMemcpy(normed, host.data(),
                        host.size() * sizeof(ps::bf16_t), hipMemcpyHostToDevice);
    }

    ps::kernel::Psq8GemmConfig cfg;
    cfg.id = ps::kernel::Psq8GemmConfigId::RowBlock1;
    Bench bench;
    bench.stream = stream;

    std::printf("rows=%u vocab=%u cols=%u pool=%u\n", rows, vocab, cols, pool);
    std::printf("%-8s %-14s %-14s %-10s %-10s %-6s\n", "allowed", "full_psq8_us",
                "exact_cand_us", "speedup", "token_ok", "path");

    bool all_exact = true;
    for (uint32_t count : counts) {
        std::vector<uint32_t> mask;
        build_mask(mask, vocab, count, 77u + count);
        const uint32_t actual = mask_count(mask, vocab);
        if (count <= 4u) {
            std::printf("  allowed count=%u:", actual);
            for (uint32_t t = 0u; t < vocab; ++t)
                if (((mask[t >> 5u] >> (t & 31u)) & 1u) != 0u) std::printf(" %u", t);
            std::printf("\n");
        }
        std::vector<uint32_t> stacked(static_cast<std::size_t>(rows) * mask_words, 0u);
        for (uint32_t r = 0u; r < rows; ++r)
            std::copy(mask.begin(), mask.end(),
                      stacked.begin() + static_cast<std::size_t>(r) * mask_words);
        (void)hipMemcpy(dev_mask, stacked.data(),
                        stacked.size() * sizeof(uint32_t), hipMemcpyHostToDevice);

        const hipError_t qerr = ps::kernel::launch_activation_quantize_e4m3(
            normed, act_codes, act_scales, rows, cols, k_padded, cols, act_code_stride,
            act_scale_stride, stream);
        if (qerr != hipSuccess) {
            std::printf("FAIL quantize: %s\n", hipGetErrorString(qerr));
            return 1;
        }

        auto run_full = [&]() -> hipError_t {
            hipError_t err = ps::kernel::launch_gemm_psq8_w8a8_wmma(
                cfg, lm.codes.data<uint8_t>(), lm.scales.data<uint8_t>(),
                reinterpret_cast<const int8_t*>(act_codes), act_scales, full_logits,
                ps::kernel::Psq8GemmOutputDType::F32, rows, vocab, k_padded, wscale_stride,
                act_code_stride, act_scale_stride, vocab, stream);
            if (err != hipSuccess) return err;
            ::ps::kernel::ArgmaxF32Args a;
            a.logits = full_logits;
            a.sampling = dev_sampling;
            a.sampled_tokens = out_a;
            a.outputs = rows;
            a.vocab_size = vocab;
            a.logits_row_stride = vocab;
            a.constraint_masks = dev_mask;
            a.constraint_mask_words = mask_words;
            return ::ps::kernel::launch_argmax_f32_single(a, stream);
        };

        const bool small_path = actual <= proxy.candidate_capacity();
        auto run_exact = [&]() -> hipError_t {
            const ps::Status st = small_path
                                      ? proxy.select_constrained_exact(
                                            normed, rows, cols, dev_mask, mask_words,
                                            actual, out_b, stream)
                                      : proxy.select_constrained_masked(
                                            normed, rows, cols, dev_mask, mask_words,
                                            out_b, stream);
            if (!st.ok()) {
                std::printf("  path error: %s\n", st.message().c_str());
                return hipErrorUnknown;
            }
            return hipSuccess;
        };

        for (uint32_t i = 0; i < warmup; ++i) {
            (void)run_full();
            (void)run_exact();
        }
        (void)hipStreamSynchronize(stream);

        std::vector<double> full_ms;
        std::vector<double> exact_ms;
        full_ms.reserve(iters);
        exact_ms.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            bench.begin();
            const hipError_t ferr = run_full();
            if (ferr != hipSuccess) {
                std::printf("FAIL full launch: %s\n", hipGetErrorString(ferr));
                return 1;
            }
            full_ms.push_back(bench.end_ms());
        }
        for (uint32_t i = 0; i < iters; ++i) {
            bench.begin();
            const hipError_t eerr = run_exact();
            if (eerr != hipSuccess) {
                std::printf("FAIL exact launch: %s\n", hipGetErrorString(eerr));
                return 1;
            }
            exact_ms.push_back(bench.end_ms());
        }

        std::vector<int32_t> tok_a(rows, -1);
        std::vector<int32_t> tok_b(rows, -1);
        (void)hipMemcpy(tok_a.data(), out_a, rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(tok_b.data(), out_b, rows * sizeof(int32_t),
                        hipMemcpyDeviceToHost);
        const bool same = tok_a == tok_b;
        if (!same) all_exact = false;
        char label[64];
        std::snprintf(label, sizeof(label), "token exact at allowed=%u", actual);
        if (small_path) check(same, label);
        if (!same) {
            for (uint32_t r = 0u; r < rows; ++r)
                if (tok_a[r] != tok_b[r])
                    std::printf("  row=%u full=%d exact=%d\n", r, tok_a[r], tok_b[r]);
        }
        const double full_med = median(full_ms);
        const double exact_med = median(exact_ms);
        std::printf("%-8u %-14.4f %-14.4f %-10.2f %-10s %-6s\n", actual, full_med,
                    exact_med, exact_med > 0.0 ? full_med / exact_med : 0.0,
                    same ? "yes" : "NO", small_path ? "exact" : "masked");
    }

    (void)hipFree(normed);
    (void)hipFree(full_logits);
    (void)hipFree(out_a);
    (void)hipFree(out_b);
    (void)hipFree(dev_mask);
    (void)hipFree(act_codes);
    (void)hipFree(act_scales);
    (void)hipFree(dev_sampling);
    (void)proxy.shutdown();
    (void)hipStreamDestroy(stream);
    std::printf("test_lm_head_constraint_perf: passed=%d failed=%d\n", passed, failed);
    (void)all_exact;
    return failed > 0 ? 1 : 0;
}
