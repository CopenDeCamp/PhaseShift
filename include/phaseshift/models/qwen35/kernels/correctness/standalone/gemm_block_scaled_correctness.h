#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps {
namespace kernel {
namespace detail {

enum class BlockScaledKind : uint32_t {
    Fp8Block128 = 0,
    Mxfp4 = 1,
    Psq4 = 2,
};

// Reference view over a canonical block-scaled weight. `codes` is the raw
// canonical code stream (fp8: out_features x k_padded U8; mxfp4/psq4:
// out_features x k_padded/2 U8) and `scales` its metadata stream
// (fp8: ceil(out_features/128) x k_padded/128 F32; mxfp4:
// out_features x k_padded/32 U8 E8M0; psq4: out_features x k_padded/16
// bytes BF16 per 32-weight block).
struct BlockScaledWeightView {
    BlockScaledKind kind = BlockScaledKind::Fp8Block128;
    const uint8_t* codes = nullptr;
    const void* scales = nullptr;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

// Exact reference kernel: the activation row is quantized to E4M3 with a
// per-row scale (peak / 448) and the weight is dequantized from its canonical
// codes and scales. Accumulation is FP32 with the scale factors applied per
// K tile, matching the intended optimized data path. Output is BF16.
hipError_t launch_gemm_block_scaled_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    uint32_t rows,
    hipStream_t stream);

hipError_t launch_gemv_block_scaled_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    hipStream_t stream);

hipError_t launch_gemm_fp8_block128_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    uint32_t rows,
    hipStream_t stream);

hipError_t launch_gemv_fp8_block128_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    hipStream_t stream);

hipError_t launch_gemm_mxfp4_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    uint32_t rows,
    hipStream_t stream);

hipError_t launch_gemv_mxfp4_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    hipStream_t stream);

hipError_t launch_gemm_psq4_reference(
    const BlockScaledWeightView& weight,
    const ::ps::bf16_t* input,
    ::ps::bf16_t* output,
    uint32_t rows,
    hipStream_t stream);

}
}
}
