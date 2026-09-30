#pragma once
#include <cstdint>

namespace ps::quantization::fpx {

enum class WeightEncoding : uint8_t {
    BF16 = 0,
    PSQ4 = 3,
    PSQ8 = 4,
    FP8_BLOCK128 = 5,
    MXFP4 = 6,
};

enum class FpxPreset : uint8_t {
    Psq = 0,
    Fp8 = 1,
    Mxfp4 = 2,
};

enum class FpxLayout : uint16_t {
    Bf16RowMajorV1 = 0,
    Psq4RowMajorSoAV2 = 3,
    Psq8RowMajorSoAV1 = 4,
    Fp8E4m3RowMajorBlock128V1 = 5,
    Mxfp4E2m1RowMajorS32V1 = 6,
};

enum class TensorRole : uint16_t {
    Unknown = 0,
    TokenEmbedding,
    Output,
    Norm,
    Bias,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    FfnGate,
    FfnUp,
    FfnDown,
    GdnQkvza,
    GdnOut,
    GdnSmall,
    MoeRouter,
    MoeRouterBias,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    MtpEmbedding,
    MtpFc,
    MtpAttention,
    MtpFfn,
    MtpNorm,
    DFlashBackboneLinear,
    DFlashSelectorLinear,
    DFlashSmall,
};

enum class Architecture : uint8_t {
    Qwen35Dense = 0,
    DFlash2Draft = 1,
};

}  // namespace ps::quantization::fpx
