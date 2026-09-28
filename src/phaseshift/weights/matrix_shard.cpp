#include <phaseshift/weights/matrix_shard.h>

namespace ps::weights {

namespace {

Status validate_spec(const MatrixShardSpec& spec, uint32_t rows, uint32_t cols) {
    if (spec.world_size == 0)
        return Status::invalid_argument("shard world size must be >= 1", __FILE__, __LINE__);
    if (spec.rank >= spec.world_size)
        return Status::out_of_range("shard rank out of range", __FILE__, __LINE__);
    if (!spec.enabled()) return Status::make_ok();
    if (spec.axis == ShardAxis::OutputFeatures && rows % spec.world_size != 0)
        return Status::invalid_argument("row count is not divisible by the shard world size",
                                        __FILE__, __LINE__);
    if (spec.axis == ShardAxis::InputFeatures && cols % spec.world_size != 0)
        return Status::invalid_argument("column count is not divisible by the shard world size",
                                        __FILE__, __LINE__);
    return Status::make_ok();
}

Result<MatrixWeight> copy_input_shard(const MatrixWeight& full,
                                      const MatrixShardSpec& spec,
                                      gpu::GpuArena& arena,
                                      hipStream_t stream) {
    const uint32_t local_k = full.cols / spec.world_size;
    const uint32_t offset = local_k * spec.rank;
    const std::size_t row_bytes = static_cast<std::size_t>(local_k) * sizeof(ps::bf16_t);
    const std::size_t src_row_bytes = static_cast<std::size_t>(full.cols) * sizeof(ps::bf16_t);

    auto allocation = arena.allocate_aligned(
        static_cast<std::size_t>(full.rows) * row_bytes, 16);
    if (!allocation.ok()) return allocation.status();

    auto dst_tensor = gpu::Tensor::create<ps::bf16_t>(
        allocation.value(), std::vector<std::size_t>{full.rows, local_k},
        std::vector<std::size_t>{local_k, 1});
    if (!dst_tensor.ok()) return dst_tensor.status();

    const auto* src = full.bf16.data<const std::byte>() +
                      static_cast<std::size_t>(offset) * sizeof(ps::bf16_t);
    hipError_t err = hipMemcpy2DAsync(
        dst_tensor.value().data<void>(), row_bytes, src, src_row_bytes, row_bytes,
        full.rows, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess)
        return Status::hip_error("hipMemcpy2DAsync input shard", hipGetErrorString(err),
                                 __FILE__, __LINE__);

    MatrixWeight out = full;
    out.bf16 = dst_tensor.release();
    out.cols = local_k;
    return out;
}

}

Status MatrixShardSpec::validate(uint32_t rows, uint32_t cols) const {
    return validate_spec(*this, rows, cols);
}

Result<MatrixWeight> shard_bf16_weight(const MatrixWeight& full,
                                       const MatrixShardSpec& spec,
                                       gpu::GpuArena& arena,
                                       hipStream_t stream) {
    if (full.encoding != MatrixEncoding::Bf16)
        return Status::unsupported("matrix sharding is only implemented for BF16 weights",
                                   __FILE__, __LINE__);
    Status vs = validate_spec(spec, full.rows, full.cols);
    if (!vs.ok()) return vs;
    if (!spec.enabled()) return full;
    if (full.bf16.ndim() != 2)
        return Status::invalid_argument("matrix weight must be rank 2", __FILE__, __LINE__);

    if (spec.axis == ShardAxis::InputFeatures) return copy_input_shard(full, spec, arena, stream);

    const uint32_t extent = full.rows / spec.world_size;
    const uint32_t offset = extent * spec.rank;
    auto sliced = full.bf16.slice(0, offset, extent);
    if (!sliced.ok()) return sliced.status();
    MatrixWeight out = full;
    out.bf16 = sliced.release();
    out.rows = extent;
    return out;
}

}
