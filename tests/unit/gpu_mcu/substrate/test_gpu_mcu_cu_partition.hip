#include <phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

__global__ void mask_scale_kernel(float* out, int iters) {
    float a = threadIdx.x * 1.0f + blockIdx.x * 0.001f;
    const float b = 1.0001f;
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, b, 0.5f);
    }
    if (threadIdx.x == 0) out[blockIdx.x & 1023] = a;
}

double time_stream(hipStream_t stream, float* out, int iters, int blocks) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    mask_scale_kernel<<<blocks, 64, 0, stream>>>(out, iters);
    (void)hipStreamSynchronize(stream);
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        (void)hipEventRecord(start, stream);
        mask_scale_kernel<<<blocks, 64, 0, stream>>>(out, iters);
        (void)hipEventRecord(stop, stream);
        (void)hipEventSynchronize(stop);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        best = std::min(best, static_cast<double>(ms));
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return best;
}

void measure_mask_scaling(uint32_t units, uint32_t words, float* out) {
    const int blocks = 4096;
    const int iters = 2000;
    const uint32_t counts[] = {units, units / 2u, 8u, 4u, 1u};
    for (uint32_t bits : counts) {
        if (bits == 0 || bits > units) continue;
        std::vector<uint32_t> mask(words, 0u);
        for (uint32_t i = 0; i < bits; ++i) {
            mask[i / 32u] |= (1u << (i % 32u));
        }
        hipStream_t stream = nullptr;
        if (hipExtStreamCreateWithCUMask(&stream, words, mask.data()) != hipSuccess) {
            continue;
        }
        const double ms = time_stream(stream, out, iters, blocks);
        std::printf("hip mask popcount=%u: %.3f ms\n", bits, ms);
        (void)hipStreamDestroy(stream);
    }
}

bool disjoint(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & b[i]) != 0u) return false;
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        std::printf("FAIL: hipGetDeviceProperties\n");
        return 1;
    }
    const uint32_t units = static_cast<uint32_t>(prop.multiProcessorCount);
    std::printf("device mp=%u name=%s\n", units, prop.name);

    auto semantics = mcu::gpu_mcu_probe_mask_semantics(0);
    check(semantics.ok(), "mask semantics probe");
    if (!semantics.ok()) {
        std::printf("probe error: %s\n", semantics.status().message().c_str());
        return 1;
    }
    const mcu::GpuMcuMaskSemantics& sem = semantics.value();
    std::printf("semantics: hip_units=%u hsa_cus=%u\n", sem.hip_mask_unit_count,
                sem.hsa_compute_unit_count);
    check(sem.reported_multi_processor_count == units, "probe mp matches device");
    check(sem.hip_mask_unit_count == units, "hip mask unit count equals mp");
    check(sem.hsa_compute_unit_count == units * mcu::kGpuMcuCusPerWgp,
          "hsa CU count equals 2x hip units");
    check(sem.full_mask_readback_exact, "full hip mask readback exact");
    check(sem.single_bit_readback_exact, "single bit readback exact");
    check(sem.unpaired_bits_readback_exact, "unpaired bits readback exact");

    auto partition_res = mcu::GpuMcuCuPartition::create(0, 1u);
    check(partition_res.ok(), "partition created");
    if (!partition_res.ok()) {
        std::printf("partition error: %s\n", partition_res.status().message().c_str());
        return 1;
    }
    auto partition = partition_res.release();

    check(partition.hip_mask_unit_count() == units, "partition hip mask unit count");
    check(partition.hsa_cu_count() == units * mcu::kGpuMcuCusPerWgp,
          "partition hsa CU count");
    check(partition.control_wgp_count() == 1u, "control is one WGP");
    check(partition.worker_wgp_count() == units - 1u, "worker WGP count");
    check(partition.control_cu_count() == 2u, "control is two CUs");
    check(partition.worker_cu_count() == units * 2u - 2u, "worker CU count");
    check(partition.hip_mask_bit_count() == (units + 31u) / 32u * 32u,
          "hip mask bit count");
    check(partition.hsa_cu_bit_count() == (units * 2u + 31u) / 32u * 32u,
          "hsa mask bit count");

    check(disjoint(partition.hip_control_mask(), partition.hip_worker_mask()),
          "hip masks disjoint");
    check(disjoint(partition.hsa_control_cu_mask(), partition.hsa_worker_cu_mask()),
          "hsa CU masks disjoint");
    check(mcu::aql_cu_mask_is_wgp_paired(partition.hsa_control_cu_mask().data(),
                                         partition.hsa_cu_count()),
          "hsa control CU mask is WGP paired");
    check(mcu::aql_cu_mask_is_wgp_paired(partition.hsa_worker_cu_mask().data(),
                                         partition.hsa_cu_count()),
          "hsa worker CU mask is WGP paired");

    {
        uint32_t hip_used = 0;
        for (uint32_t word : partition.hip_control_mask()) {
            hip_used += __builtin_popcount(word);
        }
        for (uint32_t word : partition.hip_worker_mask()) {
            hip_used += __builtin_popcount(word);
        }
        check(hip_used == units, "hip masks cover every scheduling unit");

        uint32_t cu_used = 0;
        for (uint32_t word : partition.hsa_control_cu_mask()) {
            cu_used += __builtin_popcount(word);
        }
        for (uint32_t word : partition.hsa_worker_cu_mask()) {
            cu_used += __builtin_popcount(word);
        }
        check(cu_used == partition.hsa_cu_count(), "hsa CU masks cover every CU");
    }

    {
        std::vector<uint32_t> actual(partition.hip_control_mask().size(), 0u);
        check(hipExtStreamGetCUMask(partition.control_stream(),
                                    static_cast<uint32_t>(actual.size()),
                                    actual.data()) == hipSuccess,
              "control stream mask readback");
        check(actual == partition.hip_control_mask(), "control stream mask exact");
        std::vector<uint32_t> worker_actual(partition.hip_worker_mask().size(), 0u);
        check(hipExtStreamGetCUMask(partition.worker_stream(),
                                    static_cast<uint32_t>(worker_actual.size()),
                                    worker_actual.data()) == hipSuccess,
              "worker stream mask readback");
        check(worker_actual == partition.hip_worker_mask(), "worker stream mask exact");
    }

    {
        float* out = nullptr;
        if (hipMalloc(&out, 1024 * sizeof(float)) == hipSuccess) {
            measure_mask_scaling(units, (units + 31u) / 32u, out);
            (void)hipFree(out);
        }
    }

    check(partition.shutdown().ok(), "partition shutdown");

    auto recreate = mcu::GpuMcuCuPartition::create(0, 1u);
    check(recreate.ok(), "partition recreate");
    if (recreate.ok()) {
        auto p = recreate.release();
        check(p.shutdown().ok(), "recreated partition shutdown");
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_cu_partition: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_cu_partition: PASS\n");
    return 0;
}
