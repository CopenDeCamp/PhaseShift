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
    auto allocation = arena.allocate_aligned(bytes, 16);
    if (!allocation.ok()) return allocation.status();
    if (hipMemset(allocation.value().data(), 0, bytes) != hipSuccess)
        return ps::Status::hip_error("hipMemset device buffer",
                                     hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
    return allocation.release();
}

ps::Result<ps::gpu::DeviceAllocationView> upload_bytes(ps::gpu::GpuArena& arena,
                                                       const void* src, std::size_t bytes) {
    auto allocation = arena.allocate_aligned(bytes, 16);
    if (!allocation.ok()) return allocation.status();
    if (hipMemcpy(allocation.value().data(), src, bytes, hipMemcpyHostToDevice) != hipSuccess)
        return ps::Status::hip_error("hipMemcpy host to device",
                                     hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
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
    if (codes.ok()) {
        auto t = ps::gpu::Tensor::create<std::uint8_t>(codes.release(), codes_shape, codes_strides);
        if (t.ok()) w.codes = t.release();
    }

    std::vector<std::size_t> scales_shape{tiles, native.metadata1_stride_bytes};
    std::vector<std::size_t> scales_strides{native.metadata1_stride_bytes, 1};
    auto scales = upload_bytes(arena, native.metadata1.data(), native.metadata1.size());
    if (scales.ok()) {
        auto t =
            ps::gpu::Tensor::create<std::uint8_t>(scales.release(), scales_shape, scales_strides);
        if (t.ok()) w.scales = t.release();
    }
    return w;
}

HostMatrix column_range(const HostMatrix& source, std::uint32_t begin, std::uint32_t count) {
    HostMatrix out;
    out.rows = source.rows;
    out.cols = count;
    out.data.resize(static_cast<std::size_t>(source.rows) * count);
    for (std::uint32_t r = 0; r < source.rows; ++r)
        for (std::uint32_t c = 0; c < count; ++c)
            out.data[r * count + c] = source.at(r, begin + c);
    return out;
}

bool run_format(ps::quantization::QuantFormatId format, const char* label,
                ps::gpu::GpuArena& arena, hipStream_t stream, const HostMatrix& input) {
    constexpr std::uint32_t kOut = 64;
    constexpr std::uint32_t kIn = 64;
    constexpr std::uint32_t kPadded = 64;
    constexpr std::uint32_t kShard = 2;

    const HostMatrix weight =
        random_matrix(kOut, kIn, format == ps::quantization::QuantFormatId::Psq4 ? 41u : 42u);

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

    HostMatrix full_weight_bf16;
    full_weight_bf16.rows = kOut;
    full_weight_bf16.cols = kIn;
    full_weight_bf16.data.resize(static_cast<std::size_t>(kOut) * kIn);
    for (std::size_t i = 0; i < full_weight_bf16.data.size(); ++i)
        full_weight_bf16.data[i] = to_bf16(dequantized[i]);

    auto input_upload =
        upload_bytes(arena, input.data.data(), input.data.size() * sizeof(ps::bf16_t));
    if (!input_upload.ok()) {
        std::printf("FAIL: %s input upload\n", label);
        return false;
    }
    const auto* input_device = static_cast<const ps::bf16_t*>(input_upload.value().data());

    auto weight_upload = upload_bytes(arena, full_weight_bf16.data.data(),
                                      full_weight_bf16.data.size() * sizeof(ps::bf16_t));
    if (!weight_upload.ok()) {
        std::printf("FAIL: %s weight upload\n", label);
        return false;
    }
    const auto* weight_device = static_cast<const ps::bf16_t*>(weight_upload.value().data());

    const std::size_t full_out_bytes =
        static_cast<std::size_t>(input.rows) * kOut * sizeof(ps::bf16_t);
    auto full_out_alloc = alloc_bytes(arena, full_out_bytes);
    if (!full_out_alloc.ok()) {
        std::printf("FAIL: %s full output alloc\n", label);
        return false;
    }
    auto* full_output_device = static_cast<ps::bf16_t*>(full_out_alloc.value().data());
    hipError_t err = ps::kernel::detail::launch_gemm_bf16_baseline(
        weight_device, input_device, full_output_device, kOut, input.rows, kIn, stream);
    if (err != hipSuccess) {
        std::printf("FAIL: %s full gemm launch: %s\n", label, hipGetErrorString(err));
        return false;
    }

    const std::uint32_t local_k = kIn / kShard;
    const std::uint32_t ib0 = local_k / 32u;
    const std::uint32_t nb_local = local_k / 32u;
    const std::uint32_t code_block_bytes = native.codes_row_stride_bytes / (kPadded / 32u);
    const std::size_t tiles = native.rows_padded / 16u;
    const std::size_t local_code_bytes = nb_local * code_block_bytes;

    HostMatrix partial_sum;
    partial_sum.rows = input.rows;
    partial_sum.cols = kOut;
    partial_sum.data.assign(static_cast<std::size_t>(input.rows) * kOut, ps::bf16_t{});

    for (std::uint32_t rank = 0; rank < kShard; ++rank) {
        w8::MatrixShardSpec spec;
        spec.axis = w8::ShardAxis::InputFeatures;
        spec.rank = rank;
        spec.world_size = kShard;
        auto shard = w8::shard_matrix_weight(full, spec, arena, stream);
        if (!shard.ok()) {
            std::printf("FAIL: %s input shard rank%u: %s\n", label, rank,
                        shard.status().message().c_str());
            return false;
        }
        check(shard.value().cols == local_k, "shard column count");
        check(shard.value().k_padded == local_k, "shard k_padded follows the shard");
        check(shard.value().codes_row_stride_bytes == local_code_bytes,
              "shard codes tile stride follows the shard");

        std::vector<std::uint8_t> codes_host(local_code_bytes * tiles);
        if (hipMemcpy(codes_host.data(), shard.value().codes.data<void>(), codes_host.size(),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s codes download rank%u\n", label, rank);
            return false;
        }
        std::vector<std::uint8_t> expected_codes(local_code_bytes * tiles);
        for (std::size_t t = 0; t < tiles; ++t) {
            const std::uint8_t* src = native.codes.data() +
                                      t * native.codes_row_stride_bytes +
                                      static_cast<std::size_t>(rank == 0 ? 0 : ib0) * code_block_bytes;
            std::memcpy(expected_codes.data() + t * local_code_bytes, src, local_code_bytes);
        }
        check(std::memcmp(codes_host.data(), expected_codes.data(), codes_host.size()) == 0,
              "shard codes match the packed k range tile by tile");

        const std::uint32_t scale_block_bytes =
            native.metadata1_stride_bytes / (kPadded / 32u);
        const std::size_t local_scale_bytes = nb_local * scale_block_bytes;
        std::vector<std::uint8_t> scales_host(local_scale_bytes * tiles);
        if (hipMemcpy(scales_host.data(), shard.value().scales.data<void>(), scales_host.size(),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s scales download rank%u\n", label, rank);
            return false;
        }
        std::vector<std::uint8_t> expected_scales(local_scale_bytes * tiles);
        for (std::size_t t = 0; t < tiles; ++t) {
            const std::uint8_t* src =
                native.metadata1.data() + t * native.metadata1_stride_bytes +
                static_cast<std::size_t>(rank == 0 ? 0 : ib0) * scale_block_bytes;
            std::memcpy(expected_scales.data() + t * local_scale_bytes, src, local_scale_bytes);
        }
        check(std::memcmp(scales_host.data(), expected_scales.data(), scales_host.size()) == 0,
              "shard scales match the packed k range tile by tile");

        const HostMatrix local_weight = column_range(full_weight_bf16, local_k * rank, local_k);
        auto local_weight_upload = upload_bytes(arena, local_weight.data.data(),
                                                local_weight.data.size() * sizeof(ps::bf16_t));
        if (!local_weight_upload.ok()) {
            std::printf("FAIL: %s local weight upload rank%u\n", label, rank);
            return false;
        }
        const auto* local_weight_device =
            static_cast<const ps::bf16_t*>(local_weight_upload.value().data());

        const HostMatrix local_input = column_range(input, local_k * rank, local_k);
        auto local_input_upload = upload_bytes(arena, local_input.data.data(),
                                               local_input.data.size() * sizeof(ps::bf16_t));
        if (!local_input_upload.ok()) {
            std::printf("FAIL: %s local input upload rank%u\n", label, rank);
            return false;
        }
        const auto* local_input_device =
            static_cast<const ps::bf16_t*>(local_input_upload.value().data());

        const std::size_t partial_bytes =
            static_cast<std::size_t>(input.rows) * kOut * sizeof(ps::bf16_t);
        auto partial_alloc = alloc_bytes(arena, partial_bytes);
        if (!partial_alloc.ok()) {
            std::printf("FAIL: %s partial output alloc rank%u\n", label, rank);
            return false;
        }
        auto* partial_device = static_cast<ps::bf16_t*>(partial_alloc.value().data());
        err = ps::kernel::detail::launch_gemm_bf16_baseline(
            local_weight_device, local_input_device, partial_device, kOut, input.rows, local_k,
            stream);
        if (err != hipSuccess) {
            std::printf("FAIL: %s partial gemm launch rank%u: %s\n", label, rank,
                        hipGetErrorString(err));
            return false;
        }
        std::vector<ps::bf16_t> partial_host(static_cast<std::size_t>(input.rows) * kOut);
        if (hipMemcpy(partial_host.data(), partial_device, partial_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL: %s partial download rank%u\n", label, rank);
            return false;
        }
        for (std::size_t i = 0; i < partial_host.size() && i < partial_sum.data.size(); ++i) {
            const float a = bf16_to_float(partial_sum.data[i]);
            const float b = bf16_to_float(partial_host[i]);
            partial_sum.data[i] = to_bf16(a + b);
        }
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
    HostMatrix host_full;
    host_full.rows = input.rows;
    host_full.cols = kOut;
    host_full.data = full_output;
    const MaxError row_error = compare_matrices(host_full, partial_sum);
    std::printf(
        "%s row shard gemm (sum of partials): mismatches=%zu/%zu max_abs=%.6g scale=%.6g "
        "max_abs/scale=%.6g\n",
        label, row_error.mismatch, row_error.count, row_error.max_abs, row_error.scale,
        row_error.max_abs_ulp);
    check(row_error.max_abs_ulp <= 0.01f,
          "row shard partial sums stay within 1% of the full GEMM scale");

    w8::MatrixShardSpec column_spec;
    column_spec.axis = w8::ShardAxis::OutputFeatures;
    column_spec.rank = 0;
    column_spec.world_size = kShard;
    auto column_shard = w8::shard_matrix_weight(full, column_spec, arena, stream);
    check(column_shard.ok(), "quantized output feature sharding stays available");
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
    const HostMatrix input = random_matrix(kRows, kIn, 20261003u);

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
    std::printf("TP_ROW_QUANTIZED: PASS\n");
    return 0;
}
