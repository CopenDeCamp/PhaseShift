#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <phaseshift/quantization/psq/quant_canonical.h>
#include <phaseshift/quantization/psq/quant_preshuffle.h>
#include <phaseshift/weights/matrix_shard.h>
#include <hip/hip_runtime.h>

#include "support/tp_linear_common.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

namespace w8 = ps::weights;
namespace psq = ps::quantization::psq;

using ps::test::compare_matrices;
using ps::test::HostMatrix;
using ps::test::MaxError;
using ps::test::random_matrix;
using ps::test::to_bf16;

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

float bf16_to_float(ps::bf16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value.data) << 16u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

ps::Result<ps::gpu::DeviceAllocationView> alloc_bytes(ps::gpu::GpuArena& arena,
                                                      std::size_t bytes) {
    auto allocation = arena.allocate_aligned(bytes, 256);
    if (!allocation.ok()) return allocation.status();
    if (hipMemset(allocation.value().data(), 0, bytes) != hipSuccess)
        return ps::Status::hip_error("hipMemset device buffer", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);
    return allocation.release();
}

ps::Result<ps::gpu::DeviceAllocationView> upload_bytes(ps::gpu::GpuArena& arena,
                                                       const void* src, std::size_t bytes) {
    auto allocation = arena.allocate_aligned(bytes, 256);
    if (!allocation.ok()) return allocation.status();
    if (hipMemcpy(allocation.value().data(), src, bytes, hipMemcpyHostToDevice) != hipSuccess)
        return ps::Status::hip_error("hipMemcpy host to device", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);
    return allocation.release();
}

w8::MatrixWeight make_native_weight(ps::gpu::GpuArena& arena,
                                    const psq::NativeQuantHost& native,
                                    ps::quantization::QuantFormatId format,
                                    std::uint32_t rows, std::uint32_t cols,
                                    std::uint32_t k_padded) {
    w8::MatrixWeight w;
    w.rows = rows;
    w.cols = cols;
    w.k_padded = k_padded;
    w.encoding = format == ps::quantization::QuantFormatId::Psq4
                     ? ps::weights::MatrixEncoding::Psq4
                     : ps::weights::MatrixEncoding::Psq8;
    w.preshuffled = true;
    w.weight_scale_group = 32;
    w.codes_row_stride_bytes = native.codes_row_stride_bytes;
    w.storage_scale_stride_bytes = native.metadata1_stride_bytes;
    ps::weights::apply_default_compute_spec(w);

    const std::size_t tiles = native.rows_padded / 16u;
    std::vector<std::size_t> codes_shape{tiles, native.codes_row_stride_bytes};
    std::vector<std::size_t> codes_strides{native.codes_row_stride_bytes, 1};
    auto codes = upload_bytes(arena, native.codes.data(), native.codes.size());
    if (!codes.ok()) return w;
    auto codes_tensor =
        ps::gpu::Tensor::create<std::uint8_t>(codes.release(), codes_shape, codes_strides);
    if (codes_tensor.ok()) w.codes = codes_tensor.release();

    std::vector<std::size_t> scales_shape{tiles, native.metadata1_stride_bytes};
    std::vector<std::size_t> scales_strides{native.metadata1_stride_bytes, 1};
    auto scales = upload_bytes(arena, native.metadata1.data(), native.metadata1.size());
    if (!scales.ok()) return w;
    auto scales_tensor =
        ps::gpu::Tensor::create<std::uint8_t>(scales.release(), scales_shape, scales_strides);
    if (scales_tensor.ok()) w.scales = scales_tensor.release();
    return w;
}

HostMatrix concat_columns(const HostMatrix& left, const HostMatrix& right) {
    HostMatrix out;
    out.rows = left.rows;
    out.cols = left.cols + right.cols;
    out.data.resize(static_cast<std::size_t>(out.rows) * out.cols);
    for (std::uint32_t r = 0; r < left.rows; ++r) {
        for (std::uint32_t c = 0; c < left.cols; ++c) out.data[r * out.cols + c] = left.at(r, c);
        for (std::uint32_t c = 0; c < right.cols; ++c)
            out.data[r * out.cols + left.cols + c] = right.at(r, c);
    }
    return out;
}

HostMatrix row_range(const HostMatrix& source, std::uint32_t begin, std::uint32_t count) {
    HostMatrix out;
    out.rows = count;
    out.cols = source.cols;
    out.data.resize(static_cast<std::size_t>(count) * source.cols);
    for (std::uint32_t r = 0; r < count; ++r)
        for (std::uint32_t c = 0; c < source.cols; ++c)
            out.data[r * out.cols + c] = source.at(begin + r, c);
    return out;
}

bool run_format(ps::quantization::QuantFormatId format, const char* label,
                ps::gpu::GpuArena& arena, hipStream_t stream, const HostMatrix& input) {
    constexpr std::uint32_t kOut = 64;
    constexpr std::uint32_t kIn = 64;
    constexpr std::uint32_t kPadded = 64;
    constexpr std::uint32_t kShard = 2;

    const HostMatrix weight =
        random_matrix(kOut, kIn, format == ps::quantization::QuantFormatId::Psq4 ? 91u : 92u);

    psq::CanonicalQuantStore store;
    store.init(format, kOut, kIn, kPadded);
    for (std::uint32_t row = 0; row < kOut; ++row) {
        std::array<float, kIn> values{};
        for (std::uint32_t c = 0; c < kIn; ++c)
            values[c] = bf16_to_float(weight.at(row, c));
        if (!store.quantize_row(row, std::span<const float>(values))) {
            std::printf("FAIL: %s quantize_row\n", label);
            return false;
        }
    }

    std::vector<float> dequantized(static_cast<std::size_t>(kOut) * kIn);
    psq::dequantize_canonical(store.view(), dequantized);

    psq::NativeQuantHost native;
    if (!psq::preshuffle_native(store.view(), native)) {
        std::printf("FAIL: %s preshuffle_native\n", label);
        return false;
    }

    w8::MatrixWeight full = make_native_weight(arena, native, format, kOut, kIn, kPadded);
    if (full.codes.ndim() != 2 || full.scales.ndim() != 2) {
        std::printf("FAIL: %s native weight upload\n", label);
        return false;
    }

    auto input_upload =
        upload_bytes(arena, input.data.data(), input.data.size() * sizeof(ps::bf16_t));
    if (!input_upload.ok()) {
        std::printf("FAIL: %s input upload\n", label);
        return false;
    }
    const auto* input_device = static_cast<const ps::bf16_t*>(input_upload.value().data());

    HostMatrix full_weight_bf16;
    full_weight_bf16.rows = kOut;
    full_weight_bf16.cols = kIn;
    full_weight_bf16.data.resize(static_cast<std::size_t>(kOut) * kIn);
    for (std::size_t i = 0; i < full_weight_bf16.data.size(); ++i)
        full_weight_bf16.data[i] = to_bf16(dequantized[i]);
    auto weight_upload = upload_bytes(arena, full_weight_bf16.data.data(),
                                      full_weight_bf16.data.size() * sizeof(ps::bf16_t));
    if (!weight_upload.ok()) {
        std::printf("FAIL: %s weight upload\n", label);
        return false;
    }
    const auto* weight_device = static_cast<const ps::bf16_t*>(weight_upload.value().data());
    std::array<ps::bf16_t*, 2> local_device_outputs{nullptr, nullptr};
    std::array<std::size_t, 2> local_output_bytes{0, 0};

    const std::size_t full_out_bytes =
        static_cast<std::size_t>(input.rows) * kOut * sizeof(ps::bf16_t);
    auto full_out_upload = alloc_bytes(arena, full_out_bytes);
    if (!full_out_upload.ok()) {
        std::printf("FAIL: %s full output alloc\n", label);
        return false;
    }
    auto* full_output_device = static_cast<ps::bf16_t*>(full_out_upload.value().data());
    if (hipMemset(full_output_device, 0, full_out_bytes) != hipSuccess) {
        std::printf("FAIL: %s full output memset\n", label);
        return false;
    }
    hipError_t err = ps::kernel::detail::launch_gemm_bf16_baseline(
        weight_device, input_device, full_output_device, kOut, input.rows, kIn, stream);
    if (err != hipSuccess) {
        std::printf("FAIL: %s full gemm launch: %s\n", label, hipGetErrorString(err));
        return false;
    }

    const std::uint32_t extent = kOut / kShard;
    std::vector<HostMatrix> local_outputs(kShard);
    for (std::uint32_t rank = 0; rank < kShard; ++rank) {
        w8::MatrixShardSpec spec;
        spec.axis = w8::ShardAxis::OutputFeatures;
        spec.rank = rank;
        spec.world_size = kShard;
        auto shard = w8::shard_matrix_weight(full, spec, arena, stream);
        if (!shard.ok()) {
            std::printf("FAIL: %s shard rank%u: %s\n", label, rank,
                        shard.status().message().c_str());
            return false;
        }
        check(shard.value().rows == extent, "shard row count");

        const std::size_t tile_offset = (extent * rank) / 16u;
        const std::size_t tile_extent = extent / 16u;
        const std::size_t codes_bytes = tile_extent * native.codes_row_stride_bytes;
        std::vector<std::uint8_t> codes_host(codes_bytes);
        if (hipMemcpy(codes_host.data(), shard.value().codes.data<void>(), codes_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s codes download rank%u\n", label, rank);
            return false;
        }
        check(std::memcmp(codes_host.data(),
                          native.codes.data() + tile_offset * native.codes_row_stride_bytes,
                          codes_bytes) == 0,
              "shard codes match the native tile range");

        const std::size_t scales_bytes = tile_extent * native.metadata1_stride_bytes;
        std::vector<std::uint8_t> scales_host(scales_bytes);
        if (hipMemcpy(scales_host.data(), shard.value().scales.data<void>(), scales_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s scales download rank%u\n", label, rank);
            return false;
        }
        check(std::memcmp(scales_host.data(),
                          native.metadata1.data() + tile_offset * native.metadata1_stride_bytes,
                          scales_bytes) == 0,
              "shard scales match the native tile range");

        const HostMatrix local_weight =
            row_range(full_weight_bf16, extent * rank, extent);
        auto local_upload = upload_bytes(arena, local_weight.data.data(),
                                         local_weight.data.size() * sizeof(ps::bf16_t));
        if (!local_upload.ok()) {
            std::printf("FAIL: %s local weight upload rank%u\n", label, rank);
            return false;
        }
        const auto* local_device = static_cast<const ps::bf16_t*>(local_upload.value().data());
        const std::size_t local_out_bytes =
            static_cast<std::size_t>(input.rows) * extent * sizeof(ps::bf16_t);
        auto local_out_upload = alloc_bytes(arena, local_out_bytes);
        if (!local_out_upload.ok()) {
            std::printf("FAIL: %s local output alloc rank%u\n", label, rank);
            return false;
        }
        auto* local_output_device = static_cast<ps::bf16_t*>(local_out_upload.value().data());
        err = ps::kernel::detail::launch_gemm_bf16_baseline(
            local_device, input_device, local_output_device, extent, input.rows, kIn, stream);
        if (err != hipSuccess) {
            std::printf("FAIL: %s local gemm launch rank%u: %s\n", label, rank,
                        hipGetErrorString(err));
            return false;
        }
        local_device_outputs[rank] = local_output_device;
        local_output_bytes[rank] = local_out_bytes;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: %s stream sync\n", label);
        return false;
    }

    std::vector<ps::bf16_t> full_output(static_cast<std::size_t>(input.rows) * kOut);
    if (hipMemcpy(full_output.data(), full_output_device, full_out_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: %s full output download\n", label);
        return false;
    }
    for (std::uint32_t rank = 0; rank < kShard; ++rank) {
        HostMatrix& out = local_outputs[rank];
        out.rows = input.rows;
        out.cols = extent;
        out.data.resize(static_cast<std::size_t>(input.rows) * extent);
        if (hipMemcpy(out.data.data(), local_device_outputs[rank], local_output_bytes[rank],
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s local output download rank%u\n", label, rank);
            return false;
        }
    }

    HostMatrix host_full;
    host_full.rows = input.rows;
    host_full.cols = kOut;
    host_full.data = full_output;
    const HostMatrix concatenated = concat_columns(local_outputs[0], local_outputs[1]);
    const MaxError gemm_error = compare_matrices(host_full, concatenated);
    std::printf(
        "%s column shard gemm: mismatches=%zu/%zu max_abs=%.6g max_rel=%.6g\n", label,
        gemm_error.mismatch, gemm_error.count, gemm_error.max_abs, gemm_error.max_rel);
    check(gemm_error.mismatch == 0,
          "column shard GEMM matches the full GEMM exactly");

    w8::MatrixShardSpec input_spec;
    input_spec.axis = w8::ShardAxis::InputFeatures;
    input_spec.rank = 0;
    input_spec.world_size = kShard;
    auto input_shard = w8::shard_matrix_weight(full, input_spec, arena, stream);
    check(!input_shard.ok(), "quantized input feature sharding stays unsupported");
    if (!input_shard.ok())
        std::printf("%s input shard rejected: %s\n", label,
                    input_shard.status().message().c_str());
    return true;
}

}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: need 1 GPU, have %d\n", device_count);
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    constexpr std::uint32_t kIn = 64;
    constexpr std::uint32_t kRows = 3;
    const HostMatrix input = random_matrix(kRows, kIn, 20261002u);

    auto arena_result = ps::gpu::GpuArena::create(0, 64ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: hipStreamCreate\n");
        arena.shutdown();
        return 1;
    }

    if (!run_format(ps::quantization::QuantFormatId::Psq4, "PSQ4", arena, stream, input))
        ++g_failed;
    if (!run_format(ps::quantization::QuantFormatId::Psq8, "PSQ8", arena, stream, input))
        ++g_failed;

    (void)hipStreamSynchronize(stream);
    (void)hipStreamDestroy(stream);
    arena.shutdown();

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("TP_COLUMN_QUANTIZED: PASS\n");
    return 0;
}
