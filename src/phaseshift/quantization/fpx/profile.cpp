#include <phaseshift/quantization/fpx/profile.h>

namespace ps::quantization::fpx {

namespace {

WeightEncoding psq(TensorRole role, uint32_t li) {
    switch (role) {
        case TensorRole::TokenEmbedding:
        case TensorRole::SharedExpertDown:
        case TensorRole::Output:
            return WeightEncoding::PSQ8;
        case TensorRole::Norm:
        case TensorRole::Bias:
        case TensorRole::GdnSmall:
        case TensorRole::MoeRouter:
        case TensorRole::MoeRouterBias:
        case TensorRole::MtpNorm:
        case TensorRole::MtpEmbedding:
            return WeightEncoding::BF16;
        case TensorRole::MtpOutput:
        case TensorRole::AttnQ:
        case TensorRole::AttnO:
        case TensorRole::AttnK:
        case TensorRole::AttnV:
        case TensorRole::FfnGate:
        case TensorRole::FfnUp:
        case TensorRole::MtpAttention:
        case TensorRole::MtpFfn:
        case TensorRole::ExpertGate:
        case TensorRole::ExpertUp:
        case TensorRole::SharedExpertGate:
        case TensorRole::SharedExpertUp:
            return WeightEncoding::PSQ4;
        case TensorRole::FfnDown:
        case TensorRole::ExpertDown:
        case TensorRole::GdnQkvza:
        case TensorRole::GdnOut:
            return (li % 4 == 0) ? WeightEncoding::PSQ8 : WeightEncoding::PSQ4;
        case TensorRole::Unknown:
        case TensorRole::DFlashBackboneLinear:
        case TensorRole::DFlashSelectorLinear:
        case TensorRole::DFlashSmall:
            return WeightEncoding::BF16;
    }
    return WeightEncoding::BF16;
}

bool is_block_scaled_linear(TensorRole role) {
    switch (role) {
        case TensorRole::AttnQ:
        case TensorRole::AttnK:
        case TensorRole::AttnV:
        case TensorRole::AttnO:
        case TensorRole::FfnGate:
        case TensorRole::FfnUp:
        case TensorRole::FfnDown:
        case TensorRole::ExpertGate:
        case TensorRole::ExpertUp:
        case TensorRole::ExpertDown:
        case TensorRole::SharedExpertGate:
        case TensorRole::SharedExpertUp:
        case TensorRole::SharedExpertDown:
        case TensorRole::GdnQkvza:
        case TensorRole::GdnOut:
            return true;
        case TensorRole::TokenEmbedding:
        case TensorRole::Output:
        case TensorRole::Norm:
        case TensorRole::Bias:
        case TensorRole::GdnSmall:
        case TensorRole::MoeRouter:
        case TensorRole::MoeRouterBias:
        case TensorRole::MtpEmbedding:
        case TensorRole::MtpOutput:
        case TensorRole::MtpAttention:
        case TensorRole::MtpFfn:
        case TensorRole::MtpNorm:
        case TensorRole::DFlashBackboneLinear:
        case TensorRole::DFlashSelectorLinear:
        case TensorRole::DFlashSmall:
        case TensorRole::Unknown:
            return false;
    }
    return false;
}

WeightEncoding fp8_block128(TensorRole role) {
    return is_block_scaled_linear(role) ? WeightEncoding::FP8_BLOCK128 : WeightEncoding::BF16;
}

WeightEncoding mxfp4(TensorRole role) {
    return is_block_scaled_linear(role) ? WeightEncoding::MXFP4 : WeightEncoding::BF16;
}

WeightEncoding choose_dflash_encoding(FpxPreset preset, TensorRole role) {
    switch (preset) {
        case FpxPreset::Psq:
            switch (role) {
                case TensorRole::DFlashBackboneLinear: return WeightEncoding::PSQ4;
                case TensorRole::DFlashSelectorLinear:
                case TensorRole::DFlashSmall: return WeightEncoding::BF16;
                default: return WeightEncoding::BF16;
            }
        case FpxPreset::Fp8:
        case FpxPreset::Mxfp4:
            return WeightEncoding::BF16;
    }
    return WeightEncoding::BF16;
}

}  // namespace

WeightEncoding choose_weight_encoding(
    FpxPreset preset,
    Architecture architecture,
    TensorRole role,
    uint32_t layer_index,
    uint32_t) {
    if (architecture == Architecture::DFlash2Draft)
        return choose_dflash_encoding(preset, role);
    switch (preset) {
        case FpxPreset::Psq: return psq(role, layer_index);
        case FpxPreset::Fp8: return fp8_block128(role);
        case FpxPreset::Mxfp4: return mxfp4(role);
    }
    return WeightEncoding::BF16;
}

const char* to_string(WeightEncoding e) {
    switch (e) {
        case WeightEncoding::BF16: return "bf16";
        case WeightEncoding::PSQ4: return "psq4";
        case WeightEncoding::PSQ8: return "psq8";
        case WeightEncoding::FP8_BLOCK128: return "fp8_block128";
        case WeightEncoding::MXFP4: return "mxfp4";
    }
    return "unknown";
}

const char* to_string(FpxPreset p) {
    switch (p) {
        case FpxPreset::Psq: return "psq";
        case FpxPreset::Fp8: return "fp8";
        case FpxPreset::Mxfp4: return "mxfp4";
    }
    return "unknown";
}

const char* to_string(FpxLayout l) {
    switch (l) {
        case FpxLayout::Bf16RowMajorV1: return "bf16_row_major_v1";
        case FpxLayout::Psq4RowMajorSoAV2: return "psq4_row_major_soa_v2";
        case FpxLayout::Psq8RowMajorSoAV1: return "psq8_row_major_soa_v1";
        case FpxLayout::Fp8E4m3RowMajorBlock128V1: return "fp8_e4m3_row_major_block128_v1";
        case FpxLayout::Mxfp4E2m1RowMajorS32V1: return "mxfp4_e2m1_row_major_s32_v1";
    }
    return "unknown";
}

const char* to_string(TensorRole r) {
    switch (r) {
        case TensorRole::Unknown: return "unknown";
        case TensorRole::TokenEmbedding: return "token_embedding";
        case TensorRole::Output: return "output";
        case TensorRole::Norm: return "norm";
        case TensorRole::Bias: return "bias";
        case TensorRole::AttnQ: return "attn_q";
        case TensorRole::AttnK: return "attn_k";
        case TensorRole::AttnV: return "attn_v";
        case TensorRole::AttnO: return "attn_o";
        case TensorRole::FfnGate: return "ffn_gate";
        case TensorRole::FfnUp: return "ffn_up";
        case TensorRole::FfnDown: return "ffn_down";
        case TensorRole::GdnQkvza: return "gdn_qkvza";
        case TensorRole::GdnOut: return "gdn_out";
        case TensorRole::GdnSmall: return "gdn_small";
        case TensorRole::MoeRouter: return "moe_router";
        case TensorRole::MoeRouterBias: return "moe_router_bias";
        case TensorRole::ExpertGate: return "expert_gate";
        case TensorRole::ExpertUp: return "expert_up";
        case TensorRole::ExpertDown: return "expert_down";
        case TensorRole::SharedExpertGate: return "shared_expert_gate";
        case TensorRole::SharedExpertUp: return "shared_expert_up";
        case TensorRole::SharedExpertDown: return "shared_expert_down";
        case TensorRole::MtpEmbedding: return "mtp_embedding";
        case TensorRole::MtpOutput: return "mtp_output";
        case TensorRole::MtpAttention: return "mtp_attention";
        case TensorRole::MtpFfn: return "mtp_ffn";
        case TensorRole::MtpNorm: return "mtp_norm";
        case TensorRole::DFlashBackboneLinear: return "dflash_backbone_linear";
        case TensorRole::DFlashSelectorLinear: return "dflash_selector_linear";
        case TensorRole::DFlashSmall: return "dflash_small";
    }
    return "unknown";
}

const char* to_string(Architecture a) {
    switch (a) {
        case Architecture::Qwen35Dense: return "qwen35_dense";
        case Architecture::DFlash2Draft: return "dflash2_draft";
    }
    return "unknown";
}

std::optional<TensorRole> try_parse_tensor_role(std::string_view s) {
    if (s == "token_embedding") return TensorRole::TokenEmbedding;
    if (s == "output") return TensorRole::Output;
    if (s == "norm") return TensorRole::Norm;
    if (s == "bias") return TensorRole::Bias;
    if (s == "attn_q") return TensorRole::AttnQ;
    if (s == "attn_k") return TensorRole::AttnK;
    if (s == "attn_v") return TensorRole::AttnV;
    if (s == "attn_o") return TensorRole::AttnO;
    if (s == "ffn_gate") return TensorRole::FfnGate;
    if (s == "ffn_up") return TensorRole::FfnUp;
    if (s == "ffn_down") return TensorRole::FfnDown;
    if (s == "gdn_qkvza") return TensorRole::GdnQkvza;
    if (s == "gdn_out") return TensorRole::GdnOut;
    if (s == "gdn_small") return TensorRole::GdnSmall;
    if (s == "moe_router") return TensorRole::MoeRouter;
    if (s == "moe_router_bias") return TensorRole::MoeRouterBias;
    if (s == "expert_gate") return TensorRole::ExpertGate;
    if (s == "expert_up") return TensorRole::ExpertUp;
    if (s == "expert_down") return TensorRole::ExpertDown;
    if (s == "shared_expert_gate") return TensorRole::SharedExpertGate;
    if (s == "shared_expert_up") return TensorRole::SharedExpertUp;
    if (s == "shared_expert_down") return TensorRole::SharedExpertDown;
    if (s == "mtp_embedding") return TensorRole::MtpEmbedding;
    if (s == "mtp_output") return TensorRole::MtpOutput;
    if (s == "mtp_attention") return TensorRole::MtpAttention;
    if (s == "mtp_ffn") return TensorRole::MtpFfn;
    if (s == "mtp_norm") return TensorRole::MtpNorm;
    if (s == "dflash_backbone_linear") return TensorRole::DFlashBackboneLinear;
    if (s == "dflash_selector_linear") return TensorRole::DFlashSelectorLinear;
    if (s == "dflash_small") return TensorRole::DFlashSmall;
    return std::nullopt;
}

std::optional<WeightEncoding> try_parse_weight_encoding(std::string_view s) {
    if (s == "bf16") return WeightEncoding::BF16;
    if (s == "psq4") return WeightEncoding::PSQ4;
    if (s == "psq8") return WeightEncoding::PSQ8;
    if (s == "fp8_block128") return WeightEncoding::FP8_BLOCK128;
    if (s == "mxfp4") return WeightEncoding::MXFP4;
    return std::nullopt;
}

std::optional<FpxLayout> try_parse_fpx_layout(std::string_view s) {
    if (s == "bf16_row_major_v1") return FpxLayout::Bf16RowMajorV1;
    if (s == "psq4_row_major_soa_v2") return FpxLayout::Psq4RowMajorSoAV2;
    if (s == "psq8_row_major_soa_v1") return FpxLayout::Psq8RowMajorSoAV1;
    if (s == "fp8_e4m3_row_major_block128_v1") return FpxLayout::Fp8E4m3RowMajorBlock128V1;
    if (s == "mxfp4_e2m1_row_major_s32_v1") return FpxLayout::Mxfp4E2m1RowMajorS32V1;
    return std::nullopt;
}

}  // namespace ps::quantization::fpx
