#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ps::runtime::gpu_mcu {

struct alignas(32) McuRuntimeNodeBinding {
    uint16_t enabled = 0;
    uint16_t variant_id = 0;
    uint32_t workgroup_count_x = 0;
    uint32_t workgroup_count_y = 0;
    uint32_t workgroup_count_z = 0;
    uint32_t invocation_index = 0;
    uint32_t variant_override = 0xFFFFFFFFu;
    uint32_t reserved1 = 0;
};

static_assert(sizeof(McuRuntimeNodeBinding) == 32);
static_assert(alignof(McuRuntimeNodeBinding) == 32);
static_assert(offsetof(McuRuntimeNodeBinding, enabled) == 0);
static_assert(offsetof(McuRuntimeNodeBinding, variant_id) == 2);
static_assert(offsetof(McuRuntimeNodeBinding, workgroup_count_x) == 4);
static_assert(offsetof(McuRuntimeNodeBinding, workgroup_count_y) == 8);
static_assert(offsetof(McuRuntimeNodeBinding, workgroup_count_z) == 12);
static_assert(offsetof(McuRuntimeNodeBinding, invocation_index) == 16);
static_assert(offsetof(McuRuntimeNodeBinding, variant_override) == 20);

enum class McuInvocationPatchSource : uint8_t {
    ActualRows = 0,
    NumRequests = 1,
    NumOutputs = 2,
    AttentionRegionRows = 3,
    VerifyRequests = 4,
    Bf16ExactRowsVariant = 5,
    ActualRowsTimesParam = 6,
    NumOutputsTimesParam = 7,
    BatchTokenIds = 8,
};

struct alignas(16) McuInvocationPatch {
    uint64_t target = 0;
    uint8_t source = 0;
    uint8_t width = 0;
    uint8_t null_guard = 0;
    uint8_t reserved = 0;
    uint32_t param = 0;
};

static_assert(sizeof(McuInvocationPatch) == 16);
static_assert(offsetof(McuInvocationPatch, param) == 12);
static_assert(alignof(McuInvocationPatch) == 16);
static_assert(offsetof(McuInvocationPatch, target) == 0);
static_assert(offsetof(McuInvocationPatch, source) == 8);
static_assert(offsetof(McuInvocationPatch, width) == 9);
static_assert(offsetof(McuInvocationPatch, null_guard) == 10);

struct McuAttentionPathSpan {
    uint32_t node_begin = 0;
    uint32_t prefill_node_end = 0;
    uint32_t direct_node_end = 0;
    uint32_t split_node_end = 0;
    uint32_t split_min_visible = 0;
    uint32_t prefill_min_rows = 0;
};

static_assert(sizeof(McuAttentionPathSpan) == 24);
static_assert(std::is_trivially_copyable_v<McuAttentionPathSpan>);

}  // namespace ps::runtime::gpu_mcu
