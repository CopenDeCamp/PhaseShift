#pragma once

#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_cache.h>
#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>

#include <cstdint>
#include <vector>

namespace ps {
namespace qwen35 {
namespace runtime {

struct McuStaticPlanEntry {
    bool valid = false;
    uint32_t program_slot = 0;
    uint32_t attention_plan_key = 0;
    uint32_t plan_begin = 0;
    uint32_t plan_end = 0;
    McuCompiledPlan plan;
};

struct McuDecodeState {
    McuDecodeRuntime runtime;
    McuPlanCache cache;
    std::vector<McuStaticPlanEntry> static_plans;
    uint32_t epoch = 0;
    uint32_t fault_code = 0;
    uint32_t fault_epoch = 0;

    ::ps::runtime::gpu_mcu::GpuMcuPersistentMcu persistent;
    hipStream_t control_stream = nullptr;
    ::ps::runtime::gpu_mcu::GpuMcuSlotState* commit_slots = nullptr;
    ::ps::runtime::gpu_mcu::GpuMcuSlotBinding* commit_bindings = nullptr;
    std::vector<::ps::runtime::gpu_mcu::GpuMcuSlotState> commit_slots_host;
    std::vector<::ps::runtime::gpu_mcu::GpuMcuSlotBinding> commit_bindings_host;
    void* stop_conditions = nullptr;
    void* stop_token_ids = nullptr;
    uint32_t stop_token_count = 0;
    uint64_t* plan_epoch = nullptr;
    uint64_t* ready_epoch = nullptr;
    uint64_t ready_epoch_value = 0;
    uint32_t* verify_counts = nullptr;
    uint32_t commit_max_slots = 0;
    uint32_t commit_max_requests = 0;
    bool persistent_configured = false;
    bool full_plan = false;
    uint32_t plan_begin = 0;
    uint32_t plan_end = 0;
    uint32_t plan_total = 0;
    uint64_t commits_before_run = 0;
};

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
