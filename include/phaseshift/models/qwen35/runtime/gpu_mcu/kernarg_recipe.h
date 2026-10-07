#pragma once

#include <cstdint>

namespace ps::runtime::gpu_mcu {

enum : uint16_t {
    kMcuKernargRecipeNone = 0,
    kMcuKernargRecipeProbe = 1,
    kMcuKernargRecipeRmsNormBf16PfOnePlus = 2,
    kMcuKernargRecipeActivationQuantizeA8 = 3,
    kMcuKernargRecipePsq4Decode1Bf16U16 = 4,
    kMcuKernargRecipePsq4Decode1Bf16U8 = 5,
    kMcuKernargRecipeActivationQuantizeE4m3K5120 = 6,
    kMcuKernargRecipeElementwise = 7,
    kMcuKernargRecipeBf16ExactRows = 8,
    kMcuKernargRecipeL2Normalize = 9,
    kMcuKernargRecipeGdnConv1d = 10,
    kMcuKernargRecipeGdnRecurrence = 11,
    kMcuKernargRecipeRope = 12,
    kMcuKernargRecipeKvAppend = 13,
    kMcuKernargRecipeAttentionPaged = 14,
    kMcuKernargRecipeAttentionPagedSplit = 15,
    kMcuKernargRecipeAttentionPagedReduce = 16,
    kMcuKernargRecipeGdnReset = 17,
    kMcuKernargRecipePsq8Decode1Bf16U8 = 18,
    kMcuKernargRecipePsq4MultiRowBf16 = 19,
    kMcuKernargRecipeVerifyAcceptPrefix = 20,
    kMcuKernargRecipeGdnSpecRestore = 21,
    kMcuKernargRecipeArgmaxF32 = 22,
    kMcuKernargRecipeEmbeddingBf16 = 23,
    kMcuKernargRecipeOutputGatherBf16 = 24,
    kMcuKernargRecipeVerifyAcceptBatch = 25,
    kMcuKernargRecipeGdnSpecRestoreFromCounts = 26,
    kMcuKernargRecipeEmbeddingPsq8 = 27,
    kMcuKernargRecipeBf16Wmma = 28,
    kMcuKernargRecipeCount = 29,
};

}  // namespace ps::runtime::gpu_mcu
