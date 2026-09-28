#include <phaseshift/models/qwen35/kernels/optimized/embedding.h>
#include <phaseshift/quantization/fpx/e4m3_detail.h>
#include <phaseshift/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

namespace ps::kernel {

namespace {
namespace op_embedding_bf16 {

constexpr uint32_t kThreads = kEmbeddingThreads;
static_assert(kThreads == 256u);
constexpr uint32_t kVec = 8u;

__device__ __forceinline__ void body(const EmbeddingBf16Args& args) {
    const uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    const int32_t token = args.token_ids[row];
    const bool valid =
        token >= 0 && static_cast<uint32_t>(token) < args.vocab_size;
    if (!valid) {
        if (threadIdx.x == 0u && args.error_word != nullptr) {
            atomicCAS(args.error_word,
                      static_cast<uint32_t>(::ps::runtime::ProgramStatus::COMPLETE),
                      static_cast<uint32_t>(::ps::runtime::ProgramStatus::INVALID_BINDING));
        }
        return;
    }
    const bf16_t* __restrict__ src =
        args.table + static_cast<size_t>(token) * args.hidden_size;
    bf16_t* __restrict__ dst =
        args.output + static_cast<size_t>(row) * args.output_row_stride;
    for (uint32_t base = threadIdx.x * kVec; base < args.hidden_size;
         base += static_cast<uint32_t>(blockDim.x) * kVec) {
        if (base + kVec <= args.hidden_size &&
            (reinterpret_cast<uintptr_t>(src + base) & 15u) == 0u &&
            (reinterpret_cast<uintptr_t>(dst + base) & 15u) == 0u) {
            const uint4 v = *reinterpret_cast<const uint4*>(src + base);
            *reinterpret_cast<uint4*>(dst + base) = v;
        } else {
            const uint32_t end = min(base + kVec, args.hidden_size);
            for (uint32_t i = base; i < end; ++i)
                dst[i] = src[i];
        }
    }
}

}  // namespace op_embedding_bf16
}  // namespace

extern "C" __global__ void phaseshift_qwen35_embedding_bf16(
    EmbeddingBf16Args args) {
    op_embedding_bf16::body(args);
}

hipError_t launch_embedding_bf16(
    const bf16_t* table, const int32_t* token_ids, bf16_t* output,
    uint32_t rows, uint32_t vocab_size, uint32_t hidden_size,
    uint32_t output_row_stride, uint32_t* error_word, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (hidden_size == 0u) return hipErrorInvalidValue;
    if (table == nullptr || token_ids == nullptr || output == nullptr)
        return hipErrorInvalidValue;
    EmbeddingBf16Args args{};
    args.table = table;
    args.token_ids = token_ids;
    args.output = output;
    args.error_word = error_word;
    args.rows = rows;
    args.vocab_size = vocab_size;
    args.hidden_size = hidden_size;
    args.output_row_stride = output_row_stride;
    phaseshift_qwen35_embedding_bf16<<<rows, op_embedding_bf16::kThreads, 0, stream>>>(
        args);
    return hipGetLastError();
}

namespace {
namespace op_embedding_psq8 {

using bf16 = __hip_bfloat16;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kWavesPerBlock = 8u;

__global__ void impl(const uint8_t* __restrict__ codes,
                     const uint8_t* __restrict__ scales,
                     uint32_t codes_row_stride_bytes,
                     uint32_t scale_row_stride_bytes,
                     const int32_t* __restrict__ token_ids,
                     bf16_t* __restrict__ output,
                     uint32_t vocab_size, uint32_t hidden_size,
                     uint32_t output_row_stride,
                     uint32_t* __restrict__ error_word) {
    const uint32_t row = blockIdx.x;
    const int32_t token = token_ids[row];
    const bool valid =
        token >= 0 && static_cast<uint32_t>(token) < vocab_size;
    if (!valid) {
        if (threadIdx.x == 0u && error_word != nullptr) {
            atomicCAS(error_word,
                      static_cast<uint32_t>(::ps::runtime::ProgramStatus::COMPLETE),
                      static_cast<uint32_t>(::ps::runtime::ProgramStatus::INVALID_BINDING));
        }
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5;
    const uint32_t groups = (hidden_size + 31u) >> 5;
    bf16* dst = reinterpret_cast<bf16*>(output) +
        static_cast<size_t>(row) * output_row_stride;
    for (uint32_t g = wave; g < groups; g += kWavesPerBlock) {
        const uint32_t h = g * 32u + lane;
        float scale = 0.0f;
        if (lane == 0u) {
            const uint8_t* sb = scales +
                static_cast<size_t>(token >> 4) * scale_row_stride_bytes +
                static_cast<size_t>(token & 15u) * 2u +
                static_cast<size_t>(g) * 32u;
            const uint16_t bits = static_cast<uint16_t>(sb[0]) |
                                  (static_cast<uint16_t>(sb[1]) << 8u);
            scale = __bfloat162float(__ushort_as_bfloat16(bits));
        }
        scale = __shfl(scale, 0, 32);
        if (h < hidden_size) {
            const uint8_t* cp = codes +
                static_cast<size_t>(token >> 4) * codes_row_stride_bytes +
                static_cast<size_t>(g) * 512u +
                static_cast<size_t>(lane >> 4) * 256u +
                static_cast<size_t>(token & 15u) * 16u +
                static_cast<size_t>(lane & 15u);
            const float value =
                ::ps::quantization::fpx::device::e4m3_decode_f32(*cp) * scale;
            dst[h] = __float2bfloat16(value);
        }
    }
}

}  // namespace op_embedding_psq8
}  // namespace

hipError_t launch_embedding_psq8(
    const uint8_t* codes, const uint8_t* scales,
    uint32_t codes_row_stride_bytes, uint32_t scale_row_stride_bytes,
    const int32_t* token_ids, bf16_t* output,
    uint32_t rows, uint32_t vocab_size, uint32_t hidden_size,
    uint32_t k_padded, uint32_t output_row_stride,
    uint32_t* error_word, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (hidden_size == 0u || k_padded == 0u) return hipErrorInvalidValue;
    if (codes == nullptr || scales == nullptr || token_ids == nullptr ||
        output == nullptr)
        return hipErrorInvalidValue;
    op_embedding_psq8::impl<<<rows, op_embedding_psq8::kThreads, 0, stream>>>(
        codes, scales, codes_row_stride_bytes, scale_row_stride_bytes, token_ids,
        output, vocab_size, hidden_size, output_row_stride, error_word);
    return hipGetLastError();
}

}  // namespace ps::kernel
