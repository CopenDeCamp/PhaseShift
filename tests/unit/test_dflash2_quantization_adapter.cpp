#include <phaseshift/quantization/offline/adapter_dispatch.h>
#include <phaseshift/quantization/offline/dflash2_adapter.h>
#include <phaseshift/quantization/fpx/profile.h>
#include <phaseshift/quantization/offline/qwen35_adapter.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace fpx = ps::quantization::fpx;

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

void check_role(std::string_view name, fpx::TensorRole want) {
    fpx::TensorInfo info = fpx::DFlash2Adapter::classify(name, {4, 8});
    check(info.role == want, name.data());
}

constexpr char kDFlashConfig[] = R"JSON({
  "architectures": ["DFlash2DraftModel"],
  "num_hidden_layers": 5,
  "hidden_size": 5120
})JSON";

constexpr char kQwenConfig[] = R"JSON({
  "model_type": "qwen3_5",
  "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64}
})JSON";

void test_detect() {
    auto d = fpx::detect_quantization_architecture(kDFlashConfig);
    check(d.ok() && d.value() == fpx::Architecture::DFlash2Draft, "detect_dflash");
    auto q = fpx::detect_quantization_architecture(kQwenConfig);
    check(q.ok() && q.value() == fpx::Architecture::Qwen35Dense, "detect_qwen");
    const char* bad = R"JSON({"model_type": "llama"})JSON";
    check(!fpx::detect_quantization_architecture(bad).ok(), "detect_rejects_unknown");
}

void test_num_layers() {
    auto d = fpx::quantization_num_layers(fpx::Architecture::DFlash2Draft, kDFlashConfig);
    check(d.ok() && d.value() == 5u, "num_layers_dflash");
    auto q = fpx::quantization_num_layers(fpx::Architecture::Qwen35Dense, kQwenConfig);
    check(q.ok() && q.value() == 64u, "num_layers_qwen");
    const char* missing = R"JSON({"architectures": ["DFlash2DraftModel"]})JSON";
    check(!fpx::quantization_num_layers(fpx::Architecture::DFlash2Draft, missing).ok(),
          "num_layers_dflash_missing");
    const char* zero = R"JSON({"architectures": ["DFlash2DraftModel"], "num_hidden_layers": 0})JSON";
    check(!fpx::quantization_num_layers(fpx::Architecture::DFlash2Draft, zero).ok(),
          "num_layers_dflash_zero");
    const char* neg = R"JSON({"architectures": ["DFlash2DraftModel"], "num_hidden_layers": -1})JSON";
    check(!fpx::quantization_num_layers(fpx::Architecture::DFlash2Draft, neg).ok(),
          "num_layers_dflash_negative");
}

void test_classify_dflash() {
    check_role("fc.weight", fpx::TensorRole::DFlashBackboneLinear);
    const char* layer_backbones[] = {
        "layers.0.self_attn.q_proj.weight",
        "layers.1.self_attn.k_proj.weight",
        "layers.2.self_attn.v_proj.weight",
        "layers.3.self_attn.o_proj.weight",
        "layers.4.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.1.mlp.down_proj.weight",
        "layers.2.attention_conv.kernel_projection.weight",
        "layers.3.mlp_conv.kernel_projection.weight",
    };
    for (const char* n : layer_backbones) check_role(n, fpx::TensorRole::DFlashBackboneLinear);
    check_role("candidate_selector.hidden_projection.weight",
               fpx::TensorRole::DFlashSelectorLinear);
    check_role("hidden_norm.weight", fpx::TensorRole::DFlashSmall);
    check_role("norm.weight", fpx::TensorRole::DFlashSmall);
    check_role("layers.4.input_layernorm.weight", fpx::TensorRole::DFlashSmall);
    check_role("layers.4.post_attention_layernorm.weight", fpx::TensorRole::DFlashSmall);
    check_role("layers.4.self_attn.q_norm.weight", fpx::TensorRole::DFlashSmall);
    check_role("layers.4.self_attn.k_norm.weight", fpx::TensorRole::DFlashSmall);
    check_role("layers.4.attention_conv.base_kernel", fpx::TensorRole::DFlashSmall);
    check_role("layers.4.mlp_conv.base_kernel", fpx::TensorRole::DFlashSmall);
    check_role("candidate_selector.predecessor_codebook", fpx::TensorRole::DFlashSmall);
    check_role("candidate_selector.successor_codebook", fpx::TensorRole::DFlashSmall);
}

void test_classify_dflash_rejects() {
    {
        fpx::TensorInfo info = fpx::DFlash2Adapter::classify("layers.x.self_attn.q_proj.weight", {4});
        check(info.malformed_name && info.role == fpx::TensorRole::Unknown,
              "reject_malformed_layer");
    }
    {
        fpx::TensorInfo info = fpx::DFlash2Adapter::classify("layers.01.self_attn.q_proj.weight", {4});
        check(!info.malformed_name && info.role == fpx::TensorRole::DFlashBackboneLinear &&
                  info.layer_index == 1,
              "leading_zero_layer_parses");
    }
    {
        fpx::TensorInfo info = fpx::DFlash2Adapter::classify("lm_head.weight", {4, 8});
        check(info.role == fpx::TensorRole::Unknown && !info.malformed_name,
              "unknown_tensor_fails_closed");
    }
    {
        fpx::TensorInfo info = fpx::DFlash2Adapter::classify("layers.9.bogus.weight", {4, 8});
        check(info.role == fpx::TensorRole::Unknown && info.layer_index == 9,
              "unknown_layer_suffix_fails_closed");
    }
    {
        fpx::TensorInfo info = fpx::DFlash2Adapter::classify("layers.123456789012345678.q", {4});
        check(info.malformed_name, "reject_overflow_layer");
    }
}

void test_layer_index() {
    fpx::TensorInfo info = fpx::DFlash2Adapter::classify("layers.3.mlp.down_proj.weight", {4, 8});
    check(info.layer_index == 3, "layer_index_parsed");
    fpx::TensorInfo top = fpx::DFlash2Adapter::classify("fc.weight", {4, 8});
    check(top.layer_index == -1, "layer_index_default");
}

void test_encoding_contract() {
    auto enc = [&](fpx::TensorRole role, fpx::Architecture arch) {
        return fpx::choose_weight_encoding(fpx::FpxPreset::Psq, arch, role, 2u, 5u);
    };
    check(enc(fpx::TensorRole::DFlashBackboneLinear, fpx::Architecture::DFlash2Draft) ==
              fpx::WeightEncoding::PSQ4,
          "dflash_backbone_is_psq4");
    check(enc(fpx::TensorRole::DFlashSelectorLinear, fpx::Architecture::DFlash2Draft) ==
              fpx::WeightEncoding::BF16,
          "dflash_selector_is_bf16");
    check(enc(fpx::TensorRole::DFlashSmall, fpx::Architecture::DFlash2Draft) ==
              fpx::WeightEncoding::BF16,
          "dflash_small_is_bf16");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Fp8, fpx::Architecture::DFlash2Draft,
                                      fpx::TensorRole::DFlashBackboneLinear, 0u, 5u) ==
              fpx::WeightEncoding::BF16,
          "dflash_fp8_preset_no_psq4");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Mxfp4, fpx::Architecture::DFlash2Draft,
                                      fpx::TensorRole::DFlashBackboneLinear, 0u, 5u) ==
              fpx::WeightEncoding::BF16,
          "dflash_mxfp4_preset_no_psq4");
}

struct QwenContractCase {
    const char* name;
    fpx::TensorRole role;
    int32_t layer;
};

void test_qwen_contract_unchanged() {
    const QwenContractCase cases[] = {
        {"model.language_model.embed_tokens.weight", fpx::TensorRole::TokenEmbedding, -1},
        {"model.language_model.norm.weight", fpx::TensorRole::Norm, -1},
        {"model.language_model.lm_head.weight", fpx::TensorRole::Output, -1},
        {"lm_head.weight", fpx::TensorRole::Output, -1},
        {"model.language_model.layers.3.input_layernorm.weight", fpx::TensorRole::Norm, 3},
        {"model.language_model.layers.3.self_attn.q_proj.weight", fpx::TensorRole::AttnQ, 3},
        {"model.language_model.layers.3.self_attn.q_norm.weight", fpx::TensorRole::Norm, 3},
        {"model.language_model.layers.4.self_attn.o_proj.weight", fpx::TensorRole::AttnO, 4},
        {"model.language_model.layers.5.mlp.gate_proj.weight", fpx::TensorRole::FfnGate, 5},
        {"model.language_model.layers.6.mlp.down_proj.weight", fpx::TensorRole::FfnDown, 6},
        {"model.language_model.layers.7.linear_attn.in_proj_qkv.weight", fpx::TensorRole::GdnQkvza, 7},
        {"model.language_model.layers.8.linear_attn.out_proj.weight", fpx::TensorRole::GdnOut, 8},
        {"model.language_model.layers.9.linear_attn.conv1d.weight", fpx::TensorRole::GdnSmall, 9},
        {"mtp.layers.1.self_attn.q_proj.weight", fpx::TensorRole::MtpAttention, 1},
        {"mtp.layers.2.mlp.up_proj.weight", fpx::TensorRole::MtpFfn, 2},
        {"mtp.fc.weight", fpx::TensorRole::MtpFc, -1},
        {"mtp.norm.weight", fpx::TensorRole::MtpNorm, -1},
        {"model.visual.blocks.0.attn.q.weight", fpx::TensorRole::Unknown, -1},
        {"totally.unknown.weight", fpx::TensorRole::Unknown, -1},
    };
    for (const auto& c : cases) {
        fpx::TensorInfo info = fpx::classify_quantization_tensor(
            fpx::Architecture::Qwen35Dense, c.name, {4, 8});
        check(info.role == c.role && info.layer_index == c.layer, c.name);
        if (std::string(c.name) == "model.visual.blocks.0.attn.q.weight")
            check(info.is_visual, "qwen_visual_flag");
    }
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::AttnQ, 2u, 64u) ==
              fpx::WeightEncoding::PSQ4,
          "qwen_attn_q_stays_psq4");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::FfnDown, 4u, 64u) ==
              fpx::WeightEncoding::PSQ8,
          "qwen_ffn_down_layer4_stays_psq8");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::FfnDown, 1u, 64u) ==
              fpx::WeightEncoding::PSQ4,
          "qwen_ffn_down_layer1_stays_psq4");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::Output, 0u, 64u) ==
              fpx::WeightEncoding::PSQ8,
          "qwen_output_stays_psq8");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::MtpFc, 0u, 64u) ==
              fpx::WeightEncoding::PSQ4,
          "qwen_mtp_fc_psq4");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::MtpAttention, 0u, 64u) ==
              fpx::WeightEncoding::PSQ4,
          "qwen_mtp_attn_psq4");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::MtpFfn, 0u, 64u) ==
              fpx::WeightEncoding::PSQ4,
          "qwen_mtp_ffn_psq4");
    check(fpx::choose_weight_encoding(fpx::FpxPreset::Psq, fpx::Architecture::Qwen35Dense,
                                      fpx::TensorRole::MtpNorm, 0u, 64u) ==
              fpx::WeightEncoding::BF16,
          "qwen_mtp_norm_bf16");
}

void test_role_strings() {
    check(std::string(fpx::to_string(fpx::Architecture::DFlash2Draft)) == "dflash2_draft",
          "arch_string");
    check(fpx::try_parse_tensor_role("dflash_backbone_linear") ==
              fpx::TensorRole::DFlashBackboneLinear,
          "role_roundtrip_backbone");
    check(fpx::try_parse_tensor_role("dflash_selector_linear") ==
              fpx::TensorRole::DFlashSelectorLinear,
          "role_roundtrip_selector");
    check(fpx::try_parse_tensor_role("dflash_small") == fpx::TensorRole::DFlashSmall,
          "role_roundtrip_small");
}

void test_backbone_matrix_count() {
    std::size_t count = 0;
    const char* suffixes[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "mlp.gate_proj.weight",    "mlp.up_proj.weight",
        "mlp.down_proj.weight",    "attention_conv.kernel_projection.weight",
        "mlp_conv.kernel_projection.weight",
    };
    for (uint32_t layer = 0; layer < 5u; ++layer) {
        for (const char* s : suffixes) {
            const std::string name = "layers." + std::to_string(layer) + "." + s;
            fpx::TensorInfo info = fpx::DFlash2Adapter::classify(name, {4, 8});
            if (info.role == fpx::TensorRole::DFlashBackboneLinear) ++count;
        }
    }
    fpx::TensorInfo fc = fpx::DFlash2Adapter::classify("fc.weight", {4, 8});
    if (fc.role == fpx::TensorRole::DFlashBackboneLinear) ++count;
    check(count == 46u, "backbone_psq4_matrix_count_46");
}

}  // namespace

int main() {
    test_detect();
    test_num_layers();
    test_classify_dflash();
    test_classify_dflash_rejects();
    test_layer_index();
    test_encoding_contract();
    test_qwen_contract_unchanged();
    test_role_strings();
    test_backbone_matrix_count();
    std::printf("test_dflash2_quantization_adapter: passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
