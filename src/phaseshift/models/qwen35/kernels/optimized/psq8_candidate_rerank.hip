#include <phaseshift/models/qwen35/kernels/optimized/psq8_candidate_rerank.h>

#include <phaseshift/models/qwen35/kernels/dflash2/detail/psq8_rerank_device.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kThreadsPerBlock = 32u;

__global__ void __launch_bounds__(kThreadsPerBlock) psq8_candidate_rerank_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    const int32_t* __restrict__ candidate_ids,
    float* __restrict__ logits,
    const uint32_t rows,
    const uint32_t pool,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t logits_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;
    const uint32_t row = blockIdx.y;
    if (row >= rows || blockIdx.x * kTileOut >= pool)
        return;
    const uint32_t cand = blockIdx.x * kTileOut + m_lane;
    const bool cand_valid = cand < pool;
    const uint32_t token_id =
        cand_valid ? static_cast<uint32_t>(candidate_ids[static_cast<size_t>(row) * pool + cand])
                   : 0u;
    const float acc0 = detail::psq8_rerank_tile_body(
        weight_codes, weight_scales, activation_codes, token_id, cand_valid, m_lane, row, rows,
        k_padded, weight_scale_stride, activation_code_stride, k_group);
    if (!cand_valid || k_group != 0u)
        return;
    const float as_scale =
        activation_scales[static_cast<size_t>(row) * (activation_scale_stride / 4u)];
    logits[static_cast<size_t>(row) * logits_row_stride + cand] = acc0 * as_scale;
}

}  // namespace

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
    hipStream_t stream) {
    if (weight_codes == nullptr || weight_scales == nullptr || activation_codes == nullptr ||
        activation_scales == nullptr || candidate_ids == nullptr || logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || pool == 0u || k_padded == 0u || (k_padded & 31u) != 0u)
        return hipErrorInvalidValue;
    const dim3 grid((pool + kTileOut - 1u) / kTileOut, rows, 1u);
    psq8_candidate_rerank_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        weight_codes, weight_scales, activation_codes, activation_scales, candidate_ids,
        logits, rows, pool, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, logits_row_stride);
    return hipGetLastError();
}

}  // namespace ps::kernel
