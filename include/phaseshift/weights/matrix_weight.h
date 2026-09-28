#pragma once
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/quantization/quantization_types.h>
#include <cstdint>

namespace ps::weights {

enum class MatrixEncoding : uint8_t {
    Bf16 = 0,
    Psq4 = 3,
    Psq8 = 4,
    Fp8Block128 = 5,
    Mxfp4 = 6,
};

struct MatrixWeight {
    quantization::QuantSpecId quant_spec = quantization::QuantSpecId::BF16_V1;
    quantization::ComputeSpecId compute_spec = quantization::ComputeSpecId::BF16_F32_BF16;
    MatrixEncoding encoding = MatrixEncoding::Bf16;
    gpu::Tensor bf16;
    gpu::Tensor data;
    gpu::Tensor codes;
    gpu::Tensor scales;
    gpu::Tensor compute_codes;
    gpu::Tensor compute_scales_bf16;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t k_padded = 0;
    uint32_t storage_scale_stride_bytes = 0;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t weight_scale_group = 0;
    bool preshuffled = false;
};

inline quantization::QuantSpecId encoding_to_quant_spec(MatrixEncoding enc) noexcept {
    switch (enc) {
        case MatrixEncoding::Bf16: return quantization::QuantSpecId::BF16_V1;
        case MatrixEncoding::Psq4: return quantization::QuantSpecId::PSQ4_S32_BF16_V1;
        case MatrixEncoding::Psq8: return quantization::QuantSpecId::PSQ8_S32_BF16_V1;
        case MatrixEncoding::Fp8Block128: return quantization::QuantSpecId::FP8_E4M3_S128X128_F32_V1;
        case MatrixEncoding::Mxfp4: return quantization::QuantSpecId::MXFP4_E2M1_S32_E8M0_V1;
    }
    return quantization::QuantSpecId::BF16_V1;
}

inline quantization::ComputeSpecId encoding_to_compute_spec(MatrixEncoding enc) noexcept {
    switch (enc) {
        case MatrixEncoding::Bf16: return quantization::ComputeSpecId::BF16_F32_BF16;
        case MatrixEncoding::Psq4: return quantization::ComputeSpecId::PSQ4_W4A8_F32_BF16;
        case MatrixEncoding::Psq8: return quantization::ComputeSpecId::PSQ8_W8A8_F32_BF16;
        case MatrixEncoding::Fp8Block128: return quantization::ComputeSpecId::FP8_W8A8_F32_BF16;
        case MatrixEncoding::Mxfp4: return quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16;
    }
    return quantization::ComputeSpecId::BF16_F32_BF16;
}

inline void apply_default_compute_spec(MatrixWeight& w) noexcept {
    w.quant_spec = encoding_to_quant_spec(w.encoding);
    w.compute_spec = encoding_to_compute_spec(w.encoding);
}

inline MatrixEncoding quant_spec_to_encoding(quantization::QuantSpecId id) noexcept {
    switch (id) {
        case quantization::QuantSpecId::BF16_V1: return MatrixEncoding::Bf16;
        case quantization::QuantSpecId::PSQ4_S32_BF16_V1: return MatrixEncoding::Psq4;
        case quantization::QuantSpecId::PSQ8_S32_BF16_V1: return MatrixEncoding::Psq8;
        case quantization::QuantSpecId::FP8_E4M3_S128X128_F32_V1: return MatrixEncoding::Fp8Block128;
        case quantization::QuantSpecId::MXFP4_E2M1_S32_E8M0_V1: return MatrixEncoding::Mxfp4;
        default: break;
    }
    return MatrixEncoding::Bf16;
}

}
