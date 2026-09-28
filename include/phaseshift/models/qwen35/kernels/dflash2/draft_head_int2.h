#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kDflash2Int2CodebookValues = 4u;
inline constexpr uint32_t kDflash2Int2TileOut = 16u;
inline constexpr uint32_t kDflash2Int2MaxPool = 128u;
inline constexpr uint32_t kDflash2Int2TopnThreads = 512u;
inline constexpr uint32_t kDflash2Int2TopnPerThread = 16u;
inline constexpr uint32_t kDflash2Int2MaxPartitions = 128u;

struct Dflash2Int2Codebook {
    std::array<uint8_t, kDflash2Int2CodebookValues> values{};
    std::array<float, kDflash2Int2CodebookValues> decoded{};
};

inline uint32_t dflash2_int2_default_partitions(uint32_t vocab) noexcept {
    const uint32_t per_partition = kDflash2Int2TopnThreads * kDflash2Int2TopnPerThread;
    uint32_t partitions = (vocab + per_partition - 1u) / per_partition;
    if (partitions == 0u)
        partitions = 1u;
    if (partitions > kDflash2Int2MaxPartitions)
        partitions = kDflash2Int2MaxPartitions;
    return partitions;
}

hipError_t launch_dflash2_int2_code_histogram(
    const uint8_t* psq8_codes,
    const uint8_t* psq8_scales,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t scale_stride_bytes,
    uint64_t* histogram,
    hipStream_t stream);

Dflash2Int2Codebook dflash2_int2_derive_codebook(
    const uint64_t* histogram, bool symmetric);

double dflash2_int2_codebook_weight_error(
    const Dflash2Int2Codebook& codebook, const uint64_t* histogram);

void dflash2_int2_build_tables(
    const Dflash2Int2Codebook& codebook, uint8_t* map_table, uint32_t* expand_table);

hipError_t launch_dflash2_coarse_topn_merge(
    const int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t rows,
    uint32_t partitions,
    uint32_t pool,
    uint32_t scratch_row_stride,
    int32_t* out_ids,
    float* out_logits,
    hipStream_t stream);

hipError_t launch_dflash2_int2_pack(
    const uint8_t* psq8_codes,
    const uint8_t* map_table,
    uint8_t* int2_codes,
    uint32_t rows,
    uint32_t k_padded,
    hipStream_t stream);

hipError_t launch_dflash2_int2_coarse_head(
    const uint8_t* int2_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    const uint32_t* expand_table,
    float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t logits_row_stride,
    hipStream_t stream);

hipError_t launch_dflash2_coarse_topn(
    const float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t pool,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t scratch_row_stride,
    uint32_t partitions,
    hipStream_t stream);

hipError_t launch_dflash2_int2_remap_pool_ids(
    const int32_t* positions,
    const int32_t* pool_ids,
    int32_t* out_ids,
    uint32_t rows,
    uint32_t k,
    uint32_t pool,
    uint32_t out_row_stride,
    hipStream_t stream);

hipError_t launch_dflash2_psq8_candidate_rerank(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    const int32_t* candidate_ids,
    float* logits,
    uint32_t rows,
    uint32_t pool,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t logits_row_stride,
    hipStream_t stream);

inline constexpr uint32_t kDflash2TargetProxyMaxPool = kDflash2Int2MaxPool;

void dflash2_int2_build_error_table(
    const Dflash2Int2Codebook& codebook,
    const uint8_t* map_table,
    float* error_table);

hipError_t launch_dflash2_int2_row_error_l2(
    const uint8_t* psq8_codes,
    const uint8_t* psq8_scales,
    const float* error_table,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t scale_stride_bytes,
    float* out_error_l2,
    hipStream_t stream);

hipError_t launch_dflash2_activation_l2(
    const uint8_t* activation_codes,
    const float* activation_scales,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    float* out_activation_l2,
    hipStream_t stream);

hipError_t launch_dflash2_target_upper_logits(
    const float* coarse_logits,
    const float* activation_l2,
    const float* error_l2,
    float* upper_logits,
    uint32_t rows,
    uint32_t vocab,
    hipStream_t stream);

hipError_t launch_dflash2_target_certified_argmax(
    const float* candidate_logits,
    const int32_t* candidate_ids,
    const float* omitted_upper,
    uint32_t omitted_upper_stride,
    float absolute_margin,
    float relative_margin,
    int32_t* best_ids,
    float* best_logits,
    float* gaps,
    uint8_t* certified,
    uint32_t rows,
    uint32_t pool,
    hipStream_t stream);

hipError_t launch_dflash2_psq8_selected_token_logits(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    const int32_t* token_ids,
    float* out_logits,
    uint32_t rows,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    hipStream_t stream);

hipError_t launch_dflash2_verify_max_other_upper(
    const float* coarse_logits,
    const float* activation_l2,
    const float* error_l2,
    const int32_t* proposed_token_ids,
    float* out_max_other_upper,
    uint32_t rows,
    uint32_t vocab,
    hipStream_t stream);

hipError_t launch_dflash2_verify_direct_certificate(
    const float* exact_proposed_logits,
    const float* max_other_upper,
    float absolute_margin,
    float relative_margin,
    uint8_t* certified,
    uint32_t rows,
    hipStream_t stream);

inline constexpr uint8_t kDflash2VerifyRowUnknown = 0u;
inline constexpr uint8_t kDflash2VerifyRowAcceptCertified = 1u;
inline constexpr uint8_t kDflash2VerifyRowArgmaxCertified = 2u;

hipError_t launch_dflash2_verify_decision_sufficiency(
    const int32_t* draft_tokens,
    const uint8_t* row_state,
    const int32_t* certified_tokens,
    uint32_t num_drafts,
    uint32_t* out_decision_ready,
    uint32_t* out_num_accepted,
    uint32_t* out_correction_valid,
    int32_t* out_correction_token,
    uint32_t* out_bonus_valid,
    int32_t* out_bonus_token,
    int32_t* out_first_required_unknown_row,
    hipStream_t stream);

}  // namespace ps::kernel
