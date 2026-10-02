#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>

namespace ps {
namespace qwen35 {
namespace runtime {

class TpBatchHook {
public:
    virtual ~TpBatchHook() = default;

    virtual Status on_sequence_created(const PagedSequenceState& source) = 0;

    virtual Status on_sequence_released(const PagedSequenceState& source) = 0;

    virtual Result<BatchExecutionOutput> on_execute(const ScheduledBatch& batch) = 0;
};

}
}
}
