#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kPagedAttentionHeadLimit = 256u;
inline constexpr uint32_t kPagedAttentionBlock = 256u;
inline constexpr uint32_t kPagedAttentionSplitBlock = 128u;
inline constexpr uint32_t kPagedAttentionPrefillBlock = 256u;
inline constexpr uint32_t kPagedAttentionPrefillTileRows = 32u;

inline constexpr const char* kPagedAttentionBf16Symbol =
    "phaseshift_qwen35_attention_paged_bf16";
inline constexpr const char* kPagedAttentionPrefillBf16Symbol =
    "phaseshift_qwen35_attention_paged_prefill_bf16";
inline constexpr const char* kPagedAttentionSplitQ4FullSymbol =
    "phaseshift_qwen35_attention_paged_bf16_split_q4_full";
inline constexpr const char* kPagedAttentionReduceQ4FullQh1Symbol =
    "phaseshift_qwen35_attention_paged_bf16_reduce_q4_full_qh1";
inline constexpr const char* kPagedAttentionReduceQ4FullQh2Symbol =
    "phaseshift_qwen35_attention_paged_bf16_reduce_q4_full_qh2";
inline constexpr const char* kPagedAttentionReduceQ4FullBaseSymbol =
    "phaseshift_qwen35_attention_paged_bf16_reduce_q4_full_base";
inline constexpr const char* kPagedAttentionSplitOtherSymbol =
    "phaseshift_qwen35_attention_paged_bf16_split_other";
inline constexpr const char* kPagedAttentionReduceOtherSymbol =
    "phaseshift_qwen35_attention_paged_bf16_reduce_other";

uint32_t paged_attention_split_reduce_qh_per_wg();

enum class PagedAttentionOutputDType : uint8_t {
    BF16,
    F32,
};

struct PagedAttentionCommonArgs {
    const bf16_t* q = nullptr;

    void* output = nullptr;

    PagedAttentionOutputDType
        output_dtype = PagedAttentionOutputDType::BF16;

    uint32_t q_row_stride = 0;
    uint32_t output_row_stride = 0;

    const uint32_t* row_sequence_slots = nullptr;
    const uint32_t* row_positions = nullptr;

    const uint32_t* block_tables = nullptr;

    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;

    uint32_t rows = 0;

    uint32_t layer = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;

    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;

    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;

    float scale = 0.0f;

    uint32_t pool_kv_heads = 0;
    uint32_t kv_head_offset = 0;
};

struct PagedAttentionBf16Args {
    PagedAttentionCommonArgs common{};
    const bf16_t* k_pool = nullptr;
    const bf16_t* v_pool = nullptr;
};

static_assert(sizeof(PagedAttentionCommonArgs) == 120);
static_assert(alignof(PagedAttentionCommonArgs) == 8);
static_assert(offsetof(PagedAttentionCommonArgs, q) == 0);
static_assert(offsetof(PagedAttentionCommonArgs, output) == 8);
static_assert(offsetof(PagedAttentionCommonArgs, output_dtype) == 16);
static_assert(offsetof(PagedAttentionCommonArgs, q_row_stride) == 20);
static_assert(offsetof(PagedAttentionCommonArgs, output_row_stride) == 24);
static_assert(offsetof(PagedAttentionCommonArgs, row_sequence_slots) == 32);
static_assert(offsetof(PagedAttentionCommonArgs, row_positions) == 40);
static_assert(offsetof(PagedAttentionCommonArgs, block_tables) == 48);
static_assert(offsetof(PagedAttentionCommonArgs, block_table_stride) == 56);
static_assert(offsetof(PagedAttentionCommonArgs, max_sequences) == 60);
static_assert(offsetof(PagedAttentionCommonArgs, rows) == 64);
static_assert(offsetof(PagedAttentionCommonArgs, layer) == 68);
static_assert(offsetof(PagedAttentionCommonArgs, page_tokens) == 72);
static_assert(offsetof(PagedAttentionCommonArgs, num_pages) == 76);
static_assert(offsetof(PagedAttentionCommonArgs, num_attention_layers) == 80);
static_assert(offsetof(PagedAttentionCommonArgs, q_heads) == 84);
static_assert(offsetof(PagedAttentionCommonArgs, kv_heads) == 88);
static_assert(offsetof(PagedAttentionCommonArgs, head_dim) == 92);
static_assert(offsetof(PagedAttentionCommonArgs, elems_per_token) == 96);
static_assert(offsetof(PagedAttentionCommonArgs, elems_per_page) == 100);
static_assert(offsetof(PagedAttentionCommonArgs, elems_per_layer) == 104);
static_assert(offsetof(PagedAttentionCommonArgs, scale) == 108);
static_assert(offsetof(PagedAttentionCommonArgs, pool_kv_heads) == 112);
static_assert(offsetof(PagedAttentionCommonArgs, kv_head_offset) == 116);
static_assert(sizeof(PagedAttentionBf16Args) == 136);
static_assert(alignof(PagedAttentionBf16Args) == 8);
static_assert(offsetof(PagedAttentionBf16Args, k_pool) == 120);
static_assert(offsetof(PagedAttentionBf16Args, v_pool) == 128);

struct PagedAttentionSplitArgs {
    PagedAttentionCommonArgs common{};
    const bf16_t* k_pool = nullptr;
    const bf16_t* v_pool = nullptr;
    float* partials = nullptr;
    uint32_t splits = 0;
};

struct PagedAttentionSplitReduceArgs {
    PagedAttentionCommonArgs common{};
    const float* partials = nullptr;
    uint32_t splits = 0;
};

static_assert(sizeof(PagedAttentionSplitArgs) == 152);
static_assert(alignof(PagedAttentionSplitArgs) == 8);
static_assert(offsetof(PagedAttentionSplitArgs, k_pool) == 120);
static_assert(offsetof(PagedAttentionSplitArgs, v_pool) == 128);
static_assert(offsetof(PagedAttentionSplitArgs, partials) == 136);
static_assert(offsetof(PagedAttentionSplitArgs, splits) == 144);
static_assert(sizeof(PagedAttentionSplitReduceArgs) == 136);
static_assert(alignof(PagedAttentionSplitReduceArgs) == 8);
static_assert(offsetof(PagedAttentionSplitReduceArgs, partials) == 120);
static_assert(offsetof(PagedAttentionSplitReduceArgs, splits) == 128);

hipError_t
launch_attention_paged_bf16(
    const PagedAttentionCommonArgs& common,

    const bf16_t* k_pool,
    const bf16_t* v_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_prefill_bf16(
    const PagedAttentionCommonArgs& common,

    const bf16_t* k_pool,
    const bf16_t* v_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_prefill_fp8_e4m3(
    const PagedAttentionCommonArgs& common,

    const ::ps::qwen35::fp8e4m3_storage_t* k_pool,
    const ::ps::qwen35::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_fp8_e4m3(
    const PagedAttentionCommonArgs& common,

    const ::ps::qwen35::fp8e4m3_storage_t* k_pool,
    const ::ps::qwen35::fp8e4m3_storage_t* v_pool,

    const float* k_scale_pool,
    const float* v_scale_pool,

    hipStream_t stream);

hipError_t
launch_attention_paged_bf16_split(
    const PagedAttentionCommonArgs& common,

    const bf16_t* k_pool,
    const bf16_t* v_pool,

    float* partials,
    uint32_t splits,

    hipStream_t stream);

hipError_t
launch_attention_paged_bf16_split_reduce(
    const PagedAttentionCommonArgs& common,

    const float* partials,
    uint32_t splits,

    hipStream_t stream);

hipError_t
launch_attention_paged_psq4(
    const PagedAttentionCommonArgs& common,

    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16_t* k_scale_pool,
    const bf16_t* v_scale_pool,

    uint32_t blocks_per_head,

    hipStream_t stream);

hipError_t
launch_attention_paged_psq8(
    const PagedAttentionCommonArgs& common,

    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16_t* k_scale_pool,
    const bf16_t* v_scale_pool,

    uint32_t blocks_per_head,

    hipStream_t stream);

hipError_t
launch_attention_paged_prefill_psq4(
    const PagedAttentionCommonArgs& common,

    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16_t* k_scale_pool,
    const bf16_t* v_scale_pool,

    uint32_t blocks_per_head,

    hipStream_t stream);

hipError_t
launch_attention_paged_prefill_psq8(
    const PagedAttentionCommonArgs& common,

    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16_t* k_scale_pool,
    const bf16_t* v_scale_pool,

    uint32_t blocks_per_head,

    hipStream_t stream);

hipError_t
launch_attention_paged_psq8_split(
    const PagedAttentionCommonArgs& common,

    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16_t* k_scale_pool,
    const bf16_t* v_scale_pool,

    uint32_t blocks_per_head,
    float* partials,
    uint32_t splits,

    hipStream_t stream);

hipError_t
launch_attention_paged_psq4_split(
    const PagedAttentionCommonArgs& common,

    const uint8_t* k_code_pool,
    const uint8_t* v_code_pool,
    const bf16_t* k_scale_pool,
    const bf16_t* v_scale_pool,

    uint32_t blocks_per_head,
    float* partials,
    uint32_t splits,

    hipStream_t stream);

}  // namespace ps::kernel
