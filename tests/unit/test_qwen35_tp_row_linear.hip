#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <phaseshift/runtime/parallel/comm_bridge.h>
#include <phaseshift/weights/matrix_shard.h>
#include <hip/hip_runtime.h>

#include "support/tp_linear_common.h"

#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace rt = ps::runtime;
namespace w8 = ps::weights;
using ps::test::compare_matrices;
using ps::test::download_matrix;
using ps::test::free_matrix;
using ps::test::GpuMatrix;
using ps::test::HostMatrix;
using ps::test::MaxError;
using ps::test::random_matrix;
using ps::test::reference_gemm;

using ps::Status;

int g_failed = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        std::printf("FAIL: %s\n", name);
        ++g_failed;
    }
}

HostMatrix slice_columns(const HostMatrix& src, std::uint32_t offset,
                         std::uint32_t count) {
    HostMatrix out;
    out.rows = src.rows;
    out.cols = count;
    out.data.resize(static_cast<std::size_t>(src.rows) * count);
    for (std::uint32_t r = 0; r < src.rows; ++r)
        for (std::uint32_t c = 0; c < count; ++c) out.at(r, c) = src.at(r, offset + c);
    return out;
}

struct RankOutput {
    bool ok = false;
    std::string error;
    HostMatrix output;
};

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

RankOutput run_rank(int device, std::uint32_t rank, const HostMatrix& weight_host,
                    const HostMatrix& input_host, rt::RcclTransport& transport,
                    rt::RcclCommEndpoint& endpoint) {
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
    spec.axis = w8::ShardAxis::InputFeatures;
    spec.rank = rank;
    spec.world_size = 2;
    auto shard = w8::shard_matrix_weight(full, spec, arena, stream);
    if (!shard.ok()) {
        out.error = shard.status().message();
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }

    GpuMatrix partial;
    void* raw = nullptr;
    const std::size_t count = static_cast<std::size_t>(input.rows) * shard.value().rows;
    if (hipMalloc(&raw, count * sizeof(ps::bf16_t)) != hipSuccess) {
        out.error = "output alloc failed";
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }
    partial.ptr = static_cast<ps::bf16_t*>(raw);
    partial.rows = input.rows;
    partial.cols = shard.value().rows;

    const hipError_t launch = ps::kernel::detail::launch_gemm_bf16_baseline(
        shard.value().bf16.data<const ps::bf16_t>(), input.ptr, partial.ptr,
        shard.value().rows, input.rows, shard.value().cols, stream);
    if (launch != hipSuccess) {
        out.error = "gemm launch failed";
        free_matrix(partial);
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }

    Status st = transport.all_reduce_sum(endpoint.global_rank, rt::CommGroup::Tensor,
                                         partial.ptr, partial.ptr, count,
                                         rt::RcclDataType::BF16, stream);
    if (!st.ok()) {
        out.error = "all_reduce_sum: " + st.message();
        free_matrix(partial);
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        out.error = "stream sync failed";
        free_matrix(partial);
        free_matrix(input);
        arena.shutdown();
        (void)hipStreamDestroy(stream);
        return out;
    }
    out.ok = download_matrix(partial, out.output);

    free_matrix(partial);
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

    const HostMatrix weight = random_matrix(kOut, kIn, 4242u);
    const HostMatrix input = random_matrix(kRows, kIn, 909u);
    const HostMatrix input_rank0 = slice_columns(input, 0, kIn / 2);
    const HostMatrix input_rank1 = slice_columns(input, kIn / 2, kIn / 2);
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

    rt::ParallelConfig config;
    config.pp_size = 1;
    config.tp_size = 2;
    config.devices = {0, 1};
    Status cs = config.validate();
    if (!cs.ok()) {
        std::printf("FAIL: parallel config: %s\n", cs.message().c_str());
        return 1;
    }
    auto transport_result = rt::RcclTransport::create(config);
    if (!transport_result.ok()) {
        std::printf("FAIL: rccl transport: %s\n",
                    transport_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<rt::RcclTransport> transport = transport_result.release();

    rt::RcclCommEndpoint endpoint0;
    endpoint0.transport = transport.get();
    endpoint0.global_rank = 0;
    rt::RcclCommEndpoint endpoint1;
    endpoint1.transport = transport.get();
    endpoint1.global_rank = 1;

    RankOutput rank0;
    RankOutput rank1;
    std::thread t0([&] {
        rank0 = run_rank(0, 0, weight, input_rank0, *transport, endpoint0);
    });
    std::thread t1([&] {
        rank1 = run_rank(1, 1, weight, input_rank1, *transport, endpoint1);
    });
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
    if (g_failed == 0) {
        check(rank0.output.rows == kRows && rank0.output.cols == kOut,
              "rank0 partial shape");
        check(rank1.output.rows == kRows && rank1.output.cols == kOut,
              "rank1 partial shape");
        for (std::size_t i = 0; i < rank0.output.data.size(); ++i) {
            if (rank0.output.data[i].data != rank1.output.data[i].data) {
                check(false, "all reduced results agree across ranks");
                break;
            }
        }

        const MaxError vs_gpu = compare_matrices(full_gpu, rank0.output);
        std::printf(
            "row parallel: vs 1GPU gpu-gemm mismatches=%zu/%zu max_abs=%.6g "
            "scale=%.6g max_abs/scale=%.6g max_rel=%.6g max_rel_active=%.6g\n",
            vs_gpu.mismatch, vs_gpu.count, vs_gpu.max_abs, vs_gpu.scale,
            vs_gpu.max_abs_ulp, vs_gpu.max_rel, vs_gpu.max_rel_active);
        check(vs_gpu.max_abs_ulp <= 0.01f,
              "row parallel stays within 1% of the single GPU GEMM output scale");

        const MaxError vs_cpu = compare_matrices(cpu_reference, full_gpu);
        std::printf(
            "1GPU gpu-gemm vs CPU reference: mismatches=%zu/%zu max_abs=%.6g max_rel=%.6g\n",
            vs_cpu.mismatch, vs_cpu.count, vs_cpu.max_abs, vs_cpu.max_rel);
    }

    Status shutdown = transport->shutdown();
    if (!shutdown.ok()) {
        std::printf("FAIL: transport shutdown: %s\n", shutdown.message().c_str());
        ++g_failed;
    }

    if (g_failed != 0) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("TP_ROW_LINEAR: PASS\n");
    return 0;
}
