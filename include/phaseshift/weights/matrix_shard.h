#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/weights/matrix_weight.h>

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::weights {

enum class ShardAxis : uint8_t {
    None = 0,
    OutputFeatures = 1,
    InputFeatures = 2,
};

struct MatrixShardSpec {
    ShardAxis axis = ShardAxis::None;
    uint32_t rank = 0;
    uint32_t world_size = 1;

    bool enabled() const noexcept {
        return axis != ShardAxis::None && world_size > 1;
    }

    Status validate(uint32_t rows, uint32_t cols) const;
};

Result<MatrixWeight> shard_matrix_weight(const MatrixWeight& full,
                                       const MatrixShardSpec& spec,
                                       gpu::GpuArena& arena,
                                       hipStream_t stream);

Status shard_head_tensor(gpu::Tensor& tensor, const MatrixShardSpec& spec);

}
