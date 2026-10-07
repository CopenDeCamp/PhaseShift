#pragma once

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct alignas(32) McuDynamicNodeBinding {
    uint16_t enabled = 0;
    uint16_t variant_id = 0;
    uint32_t workgroup_count_x = 0;
    uint32_t workgroup_count_y = 0;
    uint32_t workgroup_count_z = 0;
    uint32_t invocation_index = 0;
    uint32_t variant_override = 0xFFFFFFFFu;
    uint32_t reserved1 = 0;
};

static_assert(sizeof(McuDynamicNodeBinding) == 32);
static_assert(alignof(McuDynamicNodeBinding) == 32);
static_assert(offsetof(McuDynamicNodeBinding, enabled) == 0);
static_assert(offsetof(McuDynamicNodeBinding, variant_id) == 2);
static_assert(offsetof(McuDynamicNodeBinding, workgroup_count_x) == 4);
static_assert(offsetof(McuDynamicNodeBinding, workgroup_count_y) == 8);
static_assert(offsetof(McuDynamicNodeBinding, workgroup_count_z) == 12);
static_assert(offsetof(McuDynamicNodeBinding, invocation_index) == 16);
static_assert(offsetof(McuDynamicNodeBinding, variant_override) == 20);

enum class McuInvocationPatchSource : uint8_t {
    ActualRows = 0,
    NumRequests = 1,
    NumOutputs = 2,
    AttentionRegionRows = 3,
    VerifyRequests = 4,
    Bf16ExactRowsVariant = 5,
    ActualRowsTimesParam = 6,
    NumOutputsTimesParam = 7,
};

struct alignas(16) McuInvocationPatch {
    uint64_t target = 0;
    uint8_t source = 0;
    uint8_t reserved[3] = {};
    uint32_t param = 0;
};

static_assert(sizeof(McuInvocationPatch) == 16);
static_assert(offsetof(McuInvocationPatch, param) == 12);
static_assert(alignof(McuInvocationPatch) == 16);
static_assert(offsetof(McuInvocationPatch, target) == 0);
static_assert(offsetof(McuInvocationPatch, source) == 8);

}  // namespace ps::runtime::gpu_mcu
