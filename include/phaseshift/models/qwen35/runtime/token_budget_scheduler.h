#pragma once

#include <phaseshift/models/qwen35/runtime/runtime_request.h>
#include <phaseshift/models/qwen35/runtime/schedule_plan.h>
#include <phaseshift/models/qwen35/runtime/kv_banker.h>
#include <phaseshift/models/qwen35/runtime/kv_capacity_manager.h>
#include <phaseshift/core/status.h>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ps {
namespace qwen35 {

namespace runtime {

Result<SchedulePlan> schedule_requests(
    std::span<const RuntimeRequest*> active,
    uint32_t max_scheduled_tokens,
    uint32_t max_scheduled_requests,
    uint32_t page_tokens,
    KVBankerState* kv);

}
}
}
