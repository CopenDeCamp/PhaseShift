#pragma once

#include <phaseshift/runtime/gpu_mcu/infrastructure/retained_packet.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

enum class McuSupervisorState : uint32_t {
    boot = 0,
    idle = 1,
    running = 2,
    fault = 3,
    stopping = 4,
};

enum class McuFaultCode : uint32_t {
    none = 0,
    invalid_plan = 1,
    invalid_node = 2,
    invalid_variant = 3,
    kernarg_slot_unavailable = 4,
    queue_stage_failure = 5,
    completion_generation_mismatch = 6,
    kernarg_region_exhausted = 7,
    kernarg_build_failed = 8,
    completion_wait_state_mismatch = 9,
};

enum : uint32_t {
    kMcuNodeDispatch = 1u << 0,
    kMcuNodeWait = 1u << 1,
    kMcuNodeEnd = 1u << 2,
};

enum : uint32_t {
    kMcuLogLineBytes = 256u,
    kMcuLogLineCount = 4096u,
};

constexpr uint32_t kMcuLogBytes = kMcuLogLineBytes * kMcuLogLineCount;

enum class McuLogEvent : uint32_t {
    PlanBegin = 1,
    DispatchPrepare = 2,
    DispatchPublish = 3,
    DispatchDoorbell = 4,
    WaitBegin = 5,
    WaitComplete = 6,
    WaitSpinTimeout = 7,
    CompletionSlotMismatch = 8,
    KernargBuildFailed = 9,
    Fault = 10,
    PlanEnd = 11,
    ExecutionSkip = 12,
    ExecutionEpoch = 13,
};

struct alignas(16) McuLogRecord {
    uint64_t sequence = 0;
    uint64_t clock = 0;
    uint32_t event = 0;
    uint32_t pc = 0;
    uint32_t variant = 0;
    uint32_t completion_slot = 0;
    uint64_t generation = 0;
    uint64_t value0 = 0;
    uint64_t value1 = 0;
    char message[192] = {};
};

static_assert(sizeof(McuLogRecord) == 256);
static_assert(alignof(McuLogRecord) == 16);

enum class McuDoorbellMode : uint32_t {
    per_packet = 0,
    coalesce = 1,
    first_only = 2,
};

enum : uint32_t {
    kMcuRecordDoorbell = 1u << 0,
    kMcuRecordAppendAhead = 1u << 1,
    kMcuRecordQueueEmpty = 1u << 2,
    kMcuRecordRefill = 1u << 3,
    kMcuRecordBackpressure = 1u << 4,
    kMcuRecordWrap = 1u << 5,
};

struct alignas(16) McuDispatchRecord {
    uint64_t dispatch_seq = 0;
    uint64_t publish_ts = 0;
    uint64_t doorbell_ts = 0;
    uint64_t generation = 0;
    uint32_t ring_slot = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

static_assert(sizeof(McuDispatchRecord) == 48);
static_assert(alignof(McuDispatchRecord) == 16);

enum : uint16_t {
    kMcuNoNext = 0xffffu,
    kMcuNoInput = 0xffffu,
};

struct McuPlanNode {
    uint16_t variant_id = 0;
    uint16_t next = kMcuNoNext;
    uint16_t kernarg_recipe = 0;
    uint16_t completion_slot = 0;
    uint32_t value = 0;
    uint16_t element_count = 0;
    uint16_t input_slot = kMcuNoInput;
    uint32_t work = 0;
    uint32_t flags = 0;
    uint32_t invocation_index = 0;
};

static_assert(sizeof(McuPlanNode) == 28);

struct McuKernelVariantDesc {
    uint16_t variant_id = 0;
    uint16_t kernarg_recipe = 0;
    uint32_t kernarg_size = 0;
    uint32_t workgroup_count_x = 1;
    uint32_t workgroup_count_y = 1;
    uint32_t workgroup_count_z = 1;
    uint32_t workgroup_x = 1;
    uint32_t workgroup_y = 1;
    uint32_t workgroup_z = 1;
    uint32_t hidden_args_policy = 1;
    GpuMcuRetainedPacket packet{};
};

static_assert(alignof(McuKernelVariantDesc) == 16);

struct alignas(64) McuDispatchTiming {
    uint64_t node_fetch_start = 0;
    uint64_t kernarg_build_end = 0;
    uint64_t packet_stage_end = 0;
    uint64_t dependency_ready = 0;
    uint64_t commit_start = 0;
    uint64_t dw0_publish = 0;
    uint64_t doorbell = 0;
    uint64_t reserved = 0;
};

static_assert(sizeof(McuDispatchTiming) == 64);

}  // namespace ps::runtime::gpu_mcu
