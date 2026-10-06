#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kEmbeddingBf16Symbol =
    "phaseshift_qwen35_embedding_bf16";
inline constexpr const char* kEmbeddingPsq8Symbol =
    "phaseshift_qwen35_embedding_psq8";

inline constexpr uint32_t kEmbeddingThreads = 256u;

struct EmbeddingBf16Args {
    const bf16_t* table = nullptr;
    const int32_t* token_ids = nullptr;
    bf16_t* output = nullptr;
    uint32_t* error_word = nullptr;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(EmbeddingBf16Args) == 48);
static_assert(alignof(EmbeddingBf16Args) == 8);
static_assert(offsetof(EmbeddingBf16Args, table) == 0);
static_assert(offsetof(EmbeddingBf16Args, token_ids) == 8);
static_assert(offsetof(EmbeddingBf16Args, output) == 16);
static_assert(offsetof(EmbeddingBf16Args, error_word) == 24);
static_assert(offsetof(EmbeddingBf16Args, rows) == 32);
static_assert(offsetof(EmbeddingBf16Args, vocab_size) == 36);
static_assert(offsetof(EmbeddingBf16Args, hidden_size) == 40);
static_assert(offsetof(EmbeddingBf16Args, output_row_stride) == 44);

struct EmbeddingPsq8Args {
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
    const int32_t* token_ids = nullptr;
    bf16_t* output = nullptr;
    uint32_t* error_word = nullptr;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(EmbeddingPsq8Args) == 64);
static_assert(alignof(EmbeddingPsq8Args) == 8);
static_assert(offsetof(EmbeddingPsq8Args, codes) == 0);
static_assert(offsetof(EmbeddingPsq8Args, scales) == 8);
static_assert(offsetof(EmbeddingPsq8Args, token_ids) == 16);
static_assert(offsetof(EmbeddingPsq8Args, output) == 24);
static_assert(offsetof(EmbeddingPsq8Args, error_word) == 32);
static_assert(offsetof(EmbeddingPsq8Args, codes_row_stride_bytes) == 40);
static_assert(offsetof(EmbeddingPsq8Args, scale_row_stride_bytes) == 44);
static_assert(offsetof(EmbeddingPsq8Args, rows) == 48);
static_assert(offsetof(EmbeddingPsq8Args, vocab_size) == 52);
static_assert(offsetof(EmbeddingPsq8Args, hidden_size) == 56);
static_assert(offsetof(EmbeddingPsq8Args, output_row_stride) == 60);

hipError_t launch_embedding_bf16(
    const bf16_t* table,
    const int32_t* token_ids,
    bf16_t* output,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t hidden_size,
    uint32_t output_row_stride,
    uint32_t* error_word,
    hipStream_t stream);

hipError_t launch_embedding_psq8(
    const uint8_t* codes,
    const uint8_t* scales,
    uint32_t codes_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    const int32_t* token_ids,
    bf16_t* output,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t hidden_size,
    uint32_t k_padded,
    uint32_t output_row_stride,
    uint32_t* error_word,
    hipStream_t stream);

}  // namespace ps::kernel
