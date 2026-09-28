#pragma once

#include <cstdint>

namespace ps::runtime::gpu_mcu {

struct GpuMcuExecutionTelemetry {
    uint64_t batches_dispatched = 0;
    uint64_t batches_completed = 0;
    uint64_t batches_failed = 0;
    uint64_t batches_committed = 0;
    uint64_t autonomous_loops = 0;
    uint64_t output_tokens_staged = 0;
    uint64_t output_tokens_published = 0;
    uint64_t output_backpressure_events = 0;
    uint64_t resources_allocated = 0;
    uint64_t resources_released = 0;
    uint64_t kv_pages_reserved = 0;
    uint64_t kv_pages_released = 0;
    uint64_t resource_blocked_events = 0;
    uint64_t resource_unblocked_events = 0;
};

}  // namespace ps::runtime::gpu_mcu
