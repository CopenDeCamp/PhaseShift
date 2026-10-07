#pragma once

#include <cstdint>

namespace ps::runtime::gpu_mcu {

enum : uint32_t {
    kMcuStageIdle = 0,
    kMcuStageLoopBegin = 1,
    kMcuStageIteration = 2,
    kMcuStageSchedulerEnter = 3,
    kMcuStageSchedulerExit = 4,
    kMcuStageExecutionStepEnter = 5,
    kMcuStageBindPlanEnter = 6,
    kMcuStageBindPlanPatch = 7,
    kMcuStageBindPlanExit = 8,
    kMcuStageExecutionStepExit = 9,
    kMcuStageRunOnceEnter = 10,
    kMcuStageRunOnceExit = 11,
    kMcuStageEmitEnter = 12,
    kMcuStageLoopEnd = 13,
};

struct alignas(16) McuStageTrace {
    uint32_t stage = kMcuStageIdle;
    uint32_t seq = 0;
    uint32_t patch_index = 0;
    uint32_t patch_source = 0;
    uint32_t patch_width = 0;
    uint32_t patch_target_low = 0;
    uint32_t patch_target_high = 0;
    uint32_t reserved0 = 0;
};

static_assert(sizeof(McuStageTrace) == 32);
static_assert(alignof(McuStageTrace) == 16);

constexpr const char* mcu_stage_name(uint32_t stage) {
    switch (stage) {
        case kMcuStageIdle:
            return "idle";
        case kMcuStageLoopBegin:
            return "loop_begin";
        case kMcuStageIteration:
            return "iteration";
        case kMcuStageSchedulerEnter:
            return "scheduler_enter";
        case kMcuStageSchedulerExit:
            return "scheduler_exit";
        case kMcuStageExecutionStepEnter:
            return "execution_step_enter";
        case kMcuStageBindPlanEnter:
            return "bind_plan_enter";
        case kMcuStageBindPlanPatch:
            return "bind_plan_patch";
        case kMcuStageBindPlanExit:
            return "bind_plan_exit";
        case kMcuStageExecutionStepExit:
            return "execution_step_exit";
        case kMcuStageRunOnceEnter:
            return "run_once_enter";
        case kMcuStageRunOnceExit:
            return "run_once_exit";
        case kMcuStageEmitEnter:
            return "emit_enter";
        case kMcuStageLoopEnd:
            return "loop_end";
        default:
            return "unknown";
    }
}

}  // namespace ps::runtime::gpu_mcu
