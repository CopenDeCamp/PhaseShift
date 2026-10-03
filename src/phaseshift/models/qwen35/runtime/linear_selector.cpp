#include <phaseshift/models/qwen35/runtime/linear_selector.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ps::qwen35::runtime {

namespace {

// Kernel capability allowlist: the (out_features, k) geometries the optimized
// WMMA linear kernels have been validated against. The bits identify which
// compute family can execute a geometry; Qwen model geometry validation is a
// separate, model-load-time concern.
constexpr uint8_t kFamilyBf16 = 1u << 0u;
constexpr uint8_t kFamilyPsq4 = 1u << 1u;
constexpr uint8_t kFamilyPsq8 = 1u << 2u;
constexpr uint8_t kFamilyFp8 = 1u << 3u;
constexpr uint8_t kFamilyMxfp4 = 1u << 4u;

struct LinearShapeRule {
    uint32_t out_features;
    uint32_t k;
    uint8_t families;
};

constexpr LinearShapeRule kLinearShapeRules[] = {
    { 32, 2560, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 48, 5120, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 1024, 2560, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 1024, 5120, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 1280, 5120, kFamilyBf16 | kFamilyPsq4 },
    { 2560, 4096, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 2560, 5120, kFamilyBf16 | kFamilyPsq4 },
    { 2560, 9216, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 4096, 2560, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 4096, 5120, kFamilyBf16 | kFamilyPsq4 },
    { 5120, 4096, kFamilyBf16 | kFamilyPsq4 },
    { 5120, 6144, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 5120, 10240, kFamilyBf16 | kFamilyPsq4 },
    { 5120, 17408, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 5120, 25600, kFamilyBf16 | kFamilyPsq4 },
    { 6144, 5120, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 8192, 2560, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 9216, 2560, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 256, 5120, kFamilyBf16 },
    { 10240, 5120, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 12288, 5120, kFamilyBf16 | kFamilyPsq4 },
    { 17408, 5120, kFamilyBf16 | kFamilyPsq4 | kFamilyPsq8 | kFamilyFp8 | kFamilyMxfp4 },
    { 248320, 2560, kFamilyBf16 | kFamilyPsq8 },
    { 248320, 5120, kFamilyBf16 | kFamilyPsq8 },
    { 8704, 5120, kFamilyPsq4 },
    { 5120, 8704, kFamilyPsq4 | kFamilyPsq8 },
    { 3072, 5120, kFamilyPsq4 | kFamilyPsq8 },
    { 512, 5120, kFamilyPsq4 },
    { 5120, 3072, kFamilyPsq4 | kFamilyPsq8 },
    { 5120, 5120, kFamilyPsq4 | kFamilyPsq8 },
    { 24, 5120, kFamilyBf16 },
};

uint8_t family_bit(LinearComputeFamily family) {
    switch (family) {
        case LinearComputeFamily::Bf16: return kFamilyBf16;
        case LinearComputeFamily::Psq4W4A8: return kFamilyPsq4;
        case LinearComputeFamily::Psq8W8A8: return kFamilyPsq8;
        case LinearComputeFamily::Fp8W8A8: return kFamilyFp8;
        case LinearComputeFamily::Mxfp4W4A8: return kFamilyMxfp4;
    }
    return 0u;
}

void linear_debug_miss(LinearComputeFamily family, uint32_t out_features, uint32_t k,
                       uint32_t rows) {
    static const bool linear_debug = []() {
        const char* e = std::getenv("PHASESHIFT_LINEAR_DEBUG");
        return e != nullptr && e[0] != '\0';
    }();
    if (linear_debug) {
        std::fprintf(stderr, "LINEAR_SELECT_MISS family=%u out=%u k=%u rows=%u\n",
                     static_cast<unsigned>(family), out_features, k, rows);
    }
}

int32_t psq4_prefill_2d_config_override() {
    const char* e = std::getenv("PHASESHIFT_PSQ_PREFILL_2D");
    if (e == nullptr) return -1;
    const std::string v(e);
    if (v == "bk64bn64") return 0;
    if (v == "bk128") return 1;
    if (v == "bn128") return 2;
    if (v == "bk128bn128") return 3;
    return -1;
}

int32_t psq8_prefill_2d_config_override() {
    const char* e = std::getenv("PHASESHIFT_PSQ8_PREFILL_2D");
    if (e == nullptr) return -1;
    const std::string v(e);
    if (v == "bk64bn64") return 0;
    if (v == "bn128") return 1;
    if (v == "bk128bn128") return 2;
    return -1;
}

ps::kernel::Psq4GemmConfig psq4_row_block_config(uint32_t rows, uint32_t out_features) {
    using Id = ps::kernel::Psq4GemmConfigId;
    if (rows >= 8u * ps::kernel::kPsqGemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsqGemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kPsqGemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq4GemmConfig{Id::RowBlock2};
    return ps::kernel::Psq4GemmConfig{Id::RowBlock1};
}

ps::kernel::Psq8GemmConfig psq8_row_block_config(uint32_t rows, uint32_t out_features) {
    using Id = ps::kernel::Psq8GemmConfigId;
    if (rows >= 8u * ps::kernel::kPsqGemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq8GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kPsqGemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq8GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kPsqGemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Psq8GemmConfig{Id::RowBlock2};
    return ps::kernel::Psq8GemmConfig{Id::RowBlock1};
}

ps::kernel::Fp8Block128GemmConfig fp8_row_block_config(uint32_t rows, uint32_t out_features) {
    using Id = ps::kernel::Fp8Block128GemmConfigId;
    if (rows >= 8u * ps::kernel::kFp8Block128GemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Fp8Block128GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kFp8Block128GemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Fp8Block128GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kFp8Block128GemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Fp8Block128GemmConfig{Id::RowBlock2};
    return ps::kernel::Fp8Block128GemmConfig{Id::RowBlock1};
}

ps::kernel::Mxfp4GemmConfig mxfp4_row_block_config(uint32_t rows, uint32_t out_features) {
    using Id = ps::kernel::Mxfp4GemmConfigId;
    if (rows >= 8u * ps::kernel::kMxfp4GemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Mxfp4GemmConfig{Id::RowBlock8};
    if (rows >= 4u * ps::kernel::kMxfp4GemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Mxfp4GemmConfig{Id::RowBlock4};
    if (rows > ps::kernel::kMxfp4GemmRowTile &&
        out_features >= ps::kernel::kPsqGemmRowBlockMinOutFeatures)
        return ps::kernel::Mxfp4GemmConfig{Id::RowBlock2};
    return ps::kernel::Mxfp4GemmConfig{Id::RowBlock1};
}

}  // namespace

bool linear_shape_supported(
    LinearComputeFamily family,
    uint32_t out_features,
    uint32_t k) {
    const uint8_t bit = family_bit(family);
    for (const auto& rule : kLinearShapeRules) {
        if (rule.out_features == out_features
            && rule.k == k
            && (rule.families & bit) != 0u) {
            return true;
        }
    }
    return false;
}

std::optional<ps::kernel::Bf16GemmConfig>
select_bf16_gemm_config(const Bf16GemmSelectorInput& input) {
    using Id = ps::kernel::Bf16GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Bf16, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Bf16, input.out_features, input.k, input.rows);
        return std::nullopt;
    }

    const bool k_aligned = (input.k & (ps::kernel::kBf16GemmExactKAlignment - 1u)) == 0u;
    if (input.allow_exact_rows && input.rows == 1u && k_aligned && input.weight_aligned16 &&
        input.input_aligned16)
        return ps::kernel::Bf16GemmConfig{Id::ExactRows, 1u};

    if (input.allow_exact_rows && input.verify_exact
        && input.rows >= 2u
        && input.rows <= ps::kernel::kBf16GemmExactRowsMax
        && k_aligned
        && input.weight_aligned16
        && input.input_aligned16
        && (input.input_row_stride & (ps::kernel::kBf16GemmExactRowStrideAlignment - 1u)) == 0u)
        return ps::kernel::Bf16GemmConfig{Id::ExactRows, static_cast<uint8_t>(input.rows)};

    if (input.out_features <= ps::kernel::kBf16GemmSplitKMaxOutFeatures)
        return ps::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
    if (input.rows >= ps::kernel::kBf16GemmWideMinRows)
        return ps::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
    return ps::kernel::Bf16GemmConfig{Id::Wmma, 0u};
}

std::optional<ps::kernel::Psq4GemmConfig>
select_psq4_gemm_config(const Psq4GemmSelectorInput& input) {
    using Id = ps::kernel::Psq4GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Psq4W4A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Psq4W4A8, input.out_features, input.k,
                          input.rows);
        return std::nullopt;
    }

    const bool use_2d_prefill =
        (input.rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        input.rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (!use_2d_prefill)
        return psq4_row_block_config(input.rows, input.out_features);

    const int32_t ov = psq4_prefill_2d_config_override();
    uint32_t cfg;
    if (ov >= 0) {
        cfg = static_cast<uint32_t>(ov);
    } else {
        const bool k128 =
            (input.k_padded % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        const bool n128 =
            (input.out_features % ps::kernel::kPsq4GemmPrefill2dBlock128) == 0u;
        if (input.rows >= ps::kernel::kPsq4GemmPrefill2dLargeMinRows &&
            input.out_features >= ps::kernel::kPsq4GemmPrefill2dLargeMinOutFeatures &&
            k128 && n128)
            cfg = 3u;
        else if (k128)
            cfg = 1u;
        else if (n128)
            cfg = 2u;
        else
            cfg = 0u;
    }

    const uint32_t ob =
        (cfg == 2u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                 : ps::kernel::kPsq4GemmPrefill2dBlock64;
    const uint32_t kc =
        (cfg == 1u || cfg == 3u) ? ps::kernel::kPsq4GemmPrefill2dBlock128
                                 : ps::kernel::kPsq4GemmPrefill2dBlock64;
    if ((input.k_padded % kc) != 0u || (input.out_features % ob) != 0u) {
        cfg = 0u;
        if ((input.out_features % ps::kernel::kPsq4GemmPrefill2dBlock64) != 0u ||
            (input.k_padded % ps::kernel::kPsq4GemmPrefill2dBlock64) != 0u)
            return psq4_row_block_config(input.rows, input.out_features);
    }

    switch (cfg) {
        case 1u: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K128N64};
        case 2u: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K64N128};
        case 3u: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K128N128};
        default: return ps::kernel::Psq4GemmConfig{Id::Prefill2D_K64N64};
    }
}

std::optional<ps::kernel::Psq8GemmConfig>
select_psq8_gemm_config(const Psq8GemmSelectorInput& input) {
    using Id = ps::kernel::Psq8GemmConfigId;
    if (input.rows == 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Psq8W8A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Psq8W8A8, input.out_features, input.k,
                          input.rows);
        return std::nullopt;
    }

    const bool use_2d_prefill =
        (input.k_padded % ps::kernel::kPsq8GemmPrefill2dKChunk) == 0u &&
        (input.rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        (input.out_features % ps::kernel::kPsq8GemmPrefill2dOutBlock) == 0u &&
        input.rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d_prefill) {
        const int32_t ov = psq8_prefill_2d_config_override();
        uint32_t cfg;
        if (ov >= 0) {
            cfg = static_cast<uint32_t>(ov);
        } else {
            const bool ob128 =
                (input.out_features % ps::kernel::kPsq8GemmPrefill2dBlock128) == 0u;
            const bool kc128 =
                (input.k_padded % ps::kernel::kPsq8GemmPrefill2dBlock128) == 0u;
            cfg = ob128 ? (kc128 ? 2u : 1u) : 0u;
        }
        const uint32_t ob = (cfg >= 1u) ? ps::kernel::kPsq8GemmPrefill2dBlock128
                                        : ps::kernel::kPsq8GemmPrefill2dBlock64;
        const uint32_t kc = (cfg == 2u) ? ps::kernel::kPsq8GemmPrefill2dBlock128
                                        : ps::kernel::kPsq8GemmPrefill2dBlock64;
        if ((input.k_padded % kc) != 0u || (input.out_features % ob) != 0u) {
            cfg = 0u;
            if ((input.out_features % ps::kernel::kPsq8GemmPrefill2dBlock64) != 0u ||
                (input.k_padded % ps::kernel::kPsq8GemmPrefill2dBlock64) != 0u)
                return psq8_row_block_config(input.rows, input.out_features);
        }
        switch (cfg) {
            case 1u: return ps::kernel::Psq8GemmConfig{Id::Prefill2D_K64N128};
            case 2u: return ps::kernel::Psq8GemmConfig{Id::Prefill2D_K128N128};
            default: return ps::kernel::Psq8GemmConfig{Id::Prefill2D};
        }
    }
    return psq8_row_block_config(input.rows, input.out_features);
}

std::optional<ps::kernel::Fp8Block128GemmConfig>
select_fp8_block128_gemm_config(const Fp8Block128GemmSelectorInput& input) {
    if (input.rows == 0u)
        return std::nullopt;
    if ((input.k_padded & 127u) != 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Fp8W8A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Fp8W8A8, input.out_features, input.k, input.rows);
        return std::nullopt;
    }
    const bool use_2d =
        (input.k_padded % ps::kernel::kFp8Block128GemmPrefill2dKChunk) == 0u &&
        (input.rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        (input.out_features % ps::kernel::kFp8Block128GemmPrefill2dOutBlock) == 0u &&
        input.rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d)
        return ps::kernel::Fp8Block128GemmConfig{ps::kernel::Fp8Block128GemmConfigId::Prefill2D};
    return fp8_row_block_config(input.rows, input.out_features);
}

std::optional<ps::kernel::Mxfp4GemmConfig>
select_mxfp4_gemm_config(const Mxfp4GemmSelectorInput& input) {
    if (input.rows == 0u)
        return std::nullopt;
    if ((input.k_padded & 31u) != 0u)
        return std::nullopt;
    if (!linear_shape_supported(LinearComputeFamily::Mxfp4W4A8, input.out_features, input.k)) {
        linear_debug_miss(LinearComputeFamily::Mxfp4W4A8, input.out_features, input.k, input.rows);
        return std::nullopt;
    }
    const bool use_2d =
        (input.k_padded % ps::kernel::kMxfp4GemmPrefill2dKChunk) == 0u &&
        (input.rows % ps::kernel::kPsqGemmPrefill2dRowsPerBlock) == 0u &&
        (input.out_features % ps::kernel::kMxfp4GemmPrefill2dOutBlock) == 0u &&
        input.rows >= ps::kernel::kPsqGemmPrefill2dMinRows &&
        input.out_features >= ps::kernel::kPsqGemmPrefill2dMinOutFeatures;
    if (use_2d)
        return ps::kernel::Mxfp4GemmConfig{ps::kernel::Mxfp4GemmConfigId::Prefill2D};
    return mxfp4_row_block_config(input.rows, input.out_features);
}

}  // namespace ps::qwen35::runtime
