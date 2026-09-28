#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

hipError_t launch_dflash2_noise_token_ids(
    int32_t anchor_token,
    int32_t mask_token,
    int32_t* output,
    uint32_t block_rows,
    hipStream_t stream);

}  // namespace ps::kernel
