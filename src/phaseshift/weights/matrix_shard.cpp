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

Result<MatrixWeight> shard_quantized_rows(const MatrixWeight& full,
                                          const MatrixShardSpec& spec) {
    if (spec.axis != ShardAxis::OutputFeatures)
        return Status::unsupported(
            spec.axis == ShardAxis::InputFeatures
                ? "quantized input feature sharding requires a repack"
                : "quantized sharding requires an output feature axis",
            __FILE__, __LINE__);
    if (!full.preshuffled)
        return Status::unsupported("quantized row sharding requires the preshuffled layout",
                                   __FILE__, __LINE__);
    if (full.codes.ndim() != 2 || full.scales.ndim() != 2)
        return Status::invalid_argument("quantized weight needs 2D codes and scales",
                                        __FILE__, __LINE__);
    const uint32_t extent = full.rows / spec.world_size;
    const uint32_t offset = extent * spec.rank;
    if ((offset % 16u) != 0u || (extent % 16u) != 0u)
        return Status::invalid_argument(
            "quantized row shard must align to the 16-row native tile", __FILE__, __LINE__);
    const std::size_t tile_offset = offset / 16u;
    const std::size_t tile_extent = extent / 16u;
    if (full.codes.dim(0) < tile_offset + tile_extent ||
        full.scales.dim(0) < tile_offset + tile_extent)
        return Status::invalid_argument("quantized tile count does not cover the shard",
                                        __FILE__, __LINE__);
    auto codes = full.codes.slice(0, tile_offset, tile_extent);
    if (!codes.ok()) return codes.status();
    auto scales = full.scales.slice(0, tile_offset, tile_extent);
    if (!scales.ok()) return scales.status();
    MatrixWeight out = full;
    out.codes = codes.release();
    out.scales = scales.release();
    out.rows = extent;
    return out;
}

Result<MatrixWeight> shard_quantized_columns(const MatrixWeight& full,
                                             const MatrixShardSpec& spec,
                                             gpu::GpuArena& arena,
                                             hipStream_t stream) {
    if (!full.preshuffled)
        return Status::unsupported("quantized column sharding requires the preshuffled layout",
                                   __FILE__, __LINE__);
    if (full.codes.ndim() != 2 || full.scales.ndim() != 2)
        return Status::invalid_argument("quantized weight needs 2D codes and scales",
                                        __FILE__, __LINE__);
    if (full.k_padded % 32u != 0u)
        return Status::invalid_argument("quantized k_padded must be a multiple of 32",
                                        __FILE__, __LINE__);
    const uint32_t local_kp = full.k_padded / spec.world_size;
    const uint32_t local_k = full.cols / spec.world_size;
    const uint32_t offset_kp = local_kp * spec.rank;
    if (local_kp % 32u != 0u || offset_kp % 32u != 0u)
        return Status::invalid_argument(
            "quantized column shard must align to the 32-elem scale group", __FILE__, __LINE__);
    if (local_kp < local_k || offset_kp + local_kp > full.k_padded)
        return Status::out_of_range("quantized column shard exceeds the packed k range",
                                    __FILE__, __LINE__);
    const uint32_t nb_full = full.k_padded / 32u;
    const uint32_t nb_local = local_kp / 32u;
    const uint32_t ib0 = offset_kp / 32u;
    if (nb_full == 0u || full.codes_row_stride_bytes % nb_full != 0u ||
        full.storage_scale_stride_bytes % nb_full != 0u)
        return Status::invalid_argument("quantized tile strides do not divide into k blocks",
                                        __FILE__, __LINE__);
    const uint32_t code_block_bytes = full.codes_row_stride_bytes / nb_full;
    const uint32_t scale_block_bytes = full.storage_scale_stride_bytes / nb_full;
    const uint32_t local_code_stride = nb_local * code_block_bytes;
    const uint32_t local_scale_stride = nb_local * scale_block_bytes;

    const std::size_t tiles = full.codes.dim(0);
    auto dst_codes = arena.allocate_aligned(tiles * local_code_stride, 16);
    if (!dst_codes.ok()) return dst_codes.status();
    auto dst_scales = arena.allocate_aligned(tiles * local_scale_stride, 16);
    if (!dst_scales.ok()) return dst_scales.status();

    const auto* codes_src =
        static_cast<const std::byte*>(full.codes.data<const std::byte>()) +
        static_cast<std::size_t>(ib0) * code_block_bytes;
    hipError_t err =
        hipMemcpy2DAsync(dst_codes.value().data(), local_code_stride, codes_src,
                         full.codes_row_stride_bytes, local_code_stride, tiles,
                         hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess)
        return Status::hip_error("hipMemcpy2DAsync quantized codes shard",
                                 hipGetErrorString(err), __FILE__, __LINE__);

    const auto* scales_src =
        static_cast<const std::byte*>(full.scales.data<const std::byte>()) +
        static_cast<std::size_t>(ib0) * scale_block_bytes;
    err = hipMemcpy2DAsync(dst_scales.value().data(), local_scale_stride, scales_src,
                           full.storage_scale_stride_bytes, local_scale_stride, tiles,
                           hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess)
        return Status::hip_error("hipMemcpy2DAsync quantized scales shard",
                                 hipGetErrorString(err), __FILE__, __LINE__);

    std::vector<std::size_t> codes_shape{tiles, local_code_stride};
    std::vector<std::size_t> codes_strides{local_code_stride, 1};
    auto codes_tensor =
        gpu::Tensor::create<std::uint8_t>(dst_codes.release(), codes_shape, codes_strides);
    if (!codes_tensor.ok()) return codes_tensor.status();
    std::vector<std::size_t> scales_shape{tiles, local_scale_stride};
    std::vector<std::size_t> scales_strides{local_scale_stride, 1};
    auto scales_tensor =
        gpu::Tensor::create<std::uint8_t>(dst_scales.release(), scales_shape, scales_strides);
    if (!scales_tensor.ok()) return scales_tensor.status();

    MatrixWeight out = full;
    out.codes = codes_tensor.release();
    out.scales = scales_tensor.release();
    out.cols = local_k;
    out.k_padded = local_kp;
    out.codes_row_stride_bytes = local_code_stride;
    out.storage_scale_stride_bytes = local_scale_stride;
    return out;
}

Result<MatrixWeight> shard_matrix_weight(const MatrixWeight& full,
                                        const MatrixShardSpec& spec,
                                        gpu::GpuArena& arena,
                                        hipStream_t stream) {
    Status vs = validate_spec(spec, full.rows, full.cols);
    if (!vs.ok()) return vs;
    if (!spec.enabled()) return full;

    if (full.encoding == MatrixEncoding::Psq4 || full.encoding == MatrixEncoding::Psq8) {
        if (spec.axis == ShardAxis::InputFeatures)
            return shard_quantized_columns(full, spec, arena, stream);
        return shard_quantized_rows(full, spec);
    }

    if (full.encoding != MatrixEncoding::Bf16)
        return Status::unsupported(
            "matrix sharding is only implemented for BF16 and PSQ weights",
            __FILE__, __LINE__);
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

Status shard_head_tensor(gpu::Tensor& tensor, const MatrixShardSpec& spec) {
    if (!spec.enabled()) return Status::make_ok();
    if (spec.axis != ShardAxis::OutputFeatures)
        return Status::unsupported("head tensors are split along the output axis",
                                   __FILE__, __LINE__);
    if (tensor.ndim() == 0 || tensor.ndim() > 2)
        return Status::invalid_argument("head tensor must have one or two dimensions",
                                        __FILE__, __LINE__);
    const auto full = static_cast<uint32_t>(tensor.dim(0));
    if (full % spec.world_size != 0)
        return Status::invalid_argument("head count is not divisible by the shard world size",
                                        __FILE__, __LINE__);
    const uint32_t extent = full / spec.world_size;
    if (extent == 0)
        return Status::invalid_argument("shard extent must be non-zero", __FILE__, __LINE__);
    auto sliced = tensor.slice(0, static_cast<std::size_t>(extent) * spec.rank, extent);
    if (!sliced.ok()) return sliced.status();
    tensor = sliced.release();
    return Status::make_ok();
}

}
