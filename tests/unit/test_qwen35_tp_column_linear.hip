#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <phaseshift/weights/matrix_shard.h>
#include <hip/hip_runtime.h>

#include "support/tp_linear_common.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace w8 = ps::weights;
using ps::test::compare_matrices;
using ps::test::download_matrix;
using ps::test::free_matrix;
using ps::test::GpuMatrix;
using ps::test::HostMatrix;
using ps::test::MaxError;
using ps::test::random_matrix;
using ps::test::reference_gemm;

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

struct RankOutput {
    bool ok = false;
    std::string error;
    HostMatrix output;
};

HostMatrix concat_columns(const HostMatrix& left, const HostMatrix& right) {
    HostMatrix out;
    out.rows = left.rows;
    out.cols = left.cols + right.cols;
    out.data.resize(static_cast<std::size_t>(out.rows) * out.cols);
    for (std::uint32_t r = 0; r < out.rows; ++r) {
        for (std::uint32_t c = 0; c < left.cols; ++c) out.at(r, c) = left.at(r, c);
        for (std::uint32_t c = 0; c < right.cols; ++c)
            out.at(r, left.cols + c) = right.at(r, c);
    }
    return out;
}

w8::MatrixWeight make_device_weight(ps::gpu::GpuArena& arena, const HostMatrix& host) {
    w8::MatrixWeight weight;
    auto view = arena.allocate_aligned(host.data.size() * sizeof(ps::bf16_t), 16);
    if (!view.ok()) return weight;
    auto tensor = ps::gpu::Tensor::create<ps::bf16_t>(
        view.value(), std::vector<std::size_t>{host.rows, host.cols},
        std::vector<std::size_t>{host.cols, 1});
    if (!tensor.ok()) return weight;
    weight.bf16 = tensor.release();
    weight.rows = host.rows;
    weight.cols = host.cols;
    weight.encoding = w8::MatrixEncoding::Bf16;
    if (hipMemcpy(weight.bf16.data<void>(), host.data.data(),
                  host.data.size() * sizeof(ps::bf16_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        weight.rows = 0;
        weight.cols = 0;
    }
    return weight;
}

RankOutput run_rank(int device, const HostMatrix& weight_host, const HostMatrix& input_host,
                    std::uint32_t rank) {
    RankOutput out;
    if (hipSetDevice(device) != hipSuccess) {
        out.error = "hipSetDevice failed";
        return out;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        out.error = "hipStreamCreate failed";
        return out;
    }
    auto arena_result = ps::gpu::GpuArena::create(device, 64ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        out.error = arena_result.status().message();
        (void)hipStreamDestroy(stream);
        return out;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    w8::MatrixWeight full = make_device_weight(arena, weight_host);
    GpuMatrix input;
    if (!upload_matrix(input_host, input)) {
        out.error = "input upload failed";
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }

    w8::MatrixShardSpec spec;
    spec.axis = w8::ShardAxis::OutputFeatures;
    spec.rank = rank;
    spec.world_size = 2;
    auto shard = w8::shard_bf16_weight(full, spec, arena, stream);
    if (!shard.ok()) {
        out.error = shard.status().message();
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }

    GpuMatrix local_out;
    void* raw = nullptr;
    const std::size_t count =
        static_cast<std::size_t>(shard.value().rows) * input.rows;
    if (hipMalloc(&raw, count * sizeof(ps::bf16_t)) != hipSuccess) {
        out.error = "output alloc failed";
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }
    local_out.ptr = static_cast<ps::bf16_t*>(raw);
    local_out.rows = input.rows;
    local_out.cols = shard.value().rows;

    const hipError_t launch = ps::kernel::detail::launch_gemm_bf16_baseline(
        shard.value().bf16.data<const ps::bf16_t>(), input.ptr, local_out.ptr,
        shard.value().rows, input.rows, shard.value().cols, stream);
    if (launch != hipSuccess) {
        out.error = "gemm launch failed";
        free_matrix(local_out);
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        out.error = "stream sync failed";
        free_matrix(local_out);
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }
    out.ok = download_matrix(local_out, out.output);

    free_matrix(local_out);
    free_matrix(input);
    arena.shutdown();
    (void)hipStreamDestroy(stream);
    return out;
}

}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        std::printf("SKIP: need 2 GPUs, have %d\n", device_count);
        return 77;
    }

    constexpr std::uint32_t kRows = 7;
    constexpr std::uint32_t kOut = 64;
    constexpr std::uint32_t kIn = 48;

    const HostMatrix weight = random_matrix(kOut, kIn, 20260929u);
    const HostMatrix input = random_matrix(kRows, kIn, 777u);
    const HostMatrix cpu_reference = reference_gemm(input, weight);

    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: device 0 unavailable\n");
        return 77;
    }
    HostMatrix full_gpu;
    {
        hipStream_t stream = nullptr;
        if (hipStreamCreate(&stream) != hipSuccess) {
            std::printf("FAIL: baseline stream create\n");
            return 1;
        }
        GpuMatrix d_weight;
        GpuMatrix d_input;
        GpuMatrix d_out;
        if (!upload_matrix(weight, d_weight) || !upload_matrix(input, d_input)) {
            std::printf("FAIL: baseline upload\n");
            return 1;
        }
        void* raw = nullptr;
        const std::size_t count = static_cast<std::size_t>(kRows) * kOut;
        if (hipMalloc(&raw, count * sizeof(ps::bf16_t)) != hipSuccess) {
            std::printf("FAIL: baseline output alloc\n");
            return 1;
        }
        d_out.ptr = static_cast<ps::bf16_t*>(raw);
        d_out.rows = kRows;
        d_out.cols = kOut;
        const hipError_t launch = ps::kernel::detail::launch_gemm_bf16_baseline(
            d_weight.ptr, d_input.ptr, d_out.ptr, kOut, kRows, kIn, stream);
        if (launch != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: baseline gemm\n");
            return 1;
        }
        if (!download_matrix(d_out, full_gpu)) {
            std::printf("FAIL: baseline download\n");
            return 1;
        }
        free_matrix(d_out);
        free_matrix(d_input);
        free_matrix(d_weight);
        (void)hipStreamDestroy(stream);
    }

    RankOutput rank0;
    RankOutput rank1;
    std::thread t0([&] { rank0 = run_rank(0, weight, input, 0); });
    std::thread t1([&] { rank1 = run_rank(1, weight, input, 1); });
    t0.join();
    t1.join();

    if (!rank0.ok) {
        std::printf("FAIL: rank0: %s\n", rank0.error.c_str());
        ++g_failed;
    }
    if (!rank1.ok) {
        std::printf("FAIL: rank1: %s\n", rank1.error.c_str());
        ++g_failed;
    }
    if (g_failed != 0) return 1;

    check(rank0.output.rows == kRows && rank0.output.cols == kOut / 2,
          "rank0 output shape");
    check(rank1.output.rows == kRows && rank1.output.cols == kOut / 2,
          "rank1 output shape");

    const HostMatrix tp = concat_columns(rank0.output, rank1.output);
    const MaxError vs_gpu = compare_matrices(full_gpu, tp);
    std::printf("column parallel: vs 1GPU gpu-gemm mismatches=%zu/%zu max_abs=%.6g max_rel=%.6g\n",
                vs_gpu.mismatch, vs_gpu.count, vs_gpu.max_abs, vs_gpu.max_rel);
    check(vs_gpu.mismatch == 0, "column parallel matches the single GPU GEMM exactly");

    const MaxError vs_cpu = compare_matrices(cpu_reference, full_gpu);
    std::printf("1GPU gpu-gemm vs CPU reference: mismatches=%zu/%zu max_abs=%.6g max_rel=%.6g\n",
                vs_cpu.mismatch, vs_cpu.count, vs_cpu.max_abs, vs_cpu.max_rel);

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("TP_COLUMN_LINEAR: PASS\n");
    return 0;
}
