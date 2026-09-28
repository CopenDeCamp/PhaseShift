#include <phaseshift/models/qwen35/kernels/dflash2/noise_input.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kNoiseInputThreads = 256u;

namespace {

__global__ void dflash2_noise_token_ids_kernel(
    int32_t anchor_token,
    int32_t mask_token,
    int32_t* output,
    uint32_t block_rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= block_rows) return;
    output[row] = row == 0u ? anchor_token : mask_token;
}

}  // namespace

hipError_t launch_dflash2_noise_token_ids(
    int32_t anchor_token,
    int32_t mask_token,
    int32_t* output,
    uint32_t block_rows,
    hipStream_t stream) {
    if (output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (block_rows == 0u) {
        return hipSuccess;
    }
    const uint32_t blocks = (block_rows + kNoiseInputThreads - 1u) / kNoiseInputThreads;
    dflash2_noise_token_ids_kernel<<<blocks, kNoiseInputThreads, 0, stream>>>(
        anchor_token, mask_token, output, block_rows);
    return hipGetLastError();
}

}  // namespace ps::kernel
