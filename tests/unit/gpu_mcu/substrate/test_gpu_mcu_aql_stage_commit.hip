#include <phaseshift/runtime/gpu_mcu/completion.h>
#include <phaseshift/runtime/gpu_mcu/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/retained_packet.h>
#include <phaseshift/runtime/gpu_mcu/worker_image.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

struct StageCommitArgs {
    mcu::DeviceAqlQueueView queue;
    mcu::GpuMcuRetainedPacket templ;
    uint64_t queue_slot;
    unsigned int* staged_flag;
    unsigned int* go_flag;
    unsigned int* published_dw0;
    unsigned int barrier;
};

__global__ void stage_then_commit_kernel(
    const mcu::GpuMcuRetainedPacket* packet_table, StageCommitArgs args) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    mcu::GpuMcuRetainedPacket templ = *packet_table;
    mcu::GpuMcuStagedPacket staged = mcu::gpu_mcu_aql_stage_retained_packet(
        args.queue, args.queue_slot, templ, 0ull);
    __threadfence_system();
    __scoped_atomic_store_n(args.staged_flag, 1u, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(args.go_flag, __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) == 0u) {
        __builtin_amdgcn_s_sleep(8);
    }
    const uint32_t dw0 = mcu::gpu_mcu_aql_commit_packet(
        args.queue, staged, args.queue_slot + 1ull, true, args.barrier != 0u,
        true);
    *args.published_dw0 = dw0;
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

    auto partition_res = mcu::GpuMcuCuPartition::create(0, 1u);
    check(partition_res.ok(), "partition created");
    if (!partition_res.ok()) return 1;
    auto partition = partition_res.release();

    auto queue_res = mcu::GpuMcuAqlQueue::create_with_mask(
        0, mcu::AqlQueueOwner::GpuMcu, mcu::kAqlDefaultQueueSize,
        partition.hsa_worker_cu_mask_view());
    check(queue_res.ok(), "worker queue created");
    if (!queue_res.ok()) return 1;
    auto queue = queue_res.release();

    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, mcu::gpu_mcu_worker_image().data,
        mcu::gpu_mcu_worker_image().bytes, mcu::kGpuMcuProbeWorkerSymbol);
    check(code_res.ok(), "probe worker loaded");
    if (!code_res.ok()) return 1;
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata& meta = code.metadata();

    auto completion_res = mcu::GpuMcuCompletionWord::create();
    check(completion_res.ok(), "completion created");
    if (!completion_res.ok()) return 1;
    auto completion = completion_res.release();

    uint32_t* output_dev = nullptr;
    uint32_t* dw0_dev = nullptr;
    mcu::GpuMcuRetainedPacket* retained_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&output_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&dw0_dev), sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&retained_dev),
                  sizeof(mcu::GpuMcuRetainedPacket)) != hipSuccess) {
        std::printf("FAIL: device buffers\n");
        return 1;
    }
    (void)hipMemset(output_dev, 0, sizeof(uint32_t));
    (void)hipMemset(dw0_dev, 0, sizeof(uint32_t));

    const std::size_t slot_bytes = mcu::aql_kernarg_slot_stride(
        meta.kernarg_segment_size);
    auto region_res = queue.allocate_kernarg(slot_bytes, 1u);
    check(region_res.ok(), "kernarg region allocated");
    if (!region_res.ok()) return 1;
    const mcu::GpuMcuKernargRegion region = region_res.release();

    unsigned char* slot = region.slot(0);
    std::memset(slot, 0, region.slot_stride);
    mcu::GpuMcuWorkerArgs args{};
    args.output = reinterpret_cast<uint64_t>(output_dev);
    args.completion = reinterpret_cast<uint64_t>(completion.device_ptr());
    args.value = 7u;
    args.element_count = 1u;
    args.generation = 1u;
    std::memcpy(slot, &args, sizeof(args));
    mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, 1u, 1u, 1u, 64u);

    mcu::GpuAqlDispatchDesc desc{};
    desc.kernel_object = meta.kernel_object;
    desc.global_work_items_x = 1u * 64u;
    desc.global_work_items_y = 1u;
    desc.global_work_items_z = 1u;
    desc.workgroup_size_x = 64;
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;
    desc.private_segment_size = meta.private_segment_size;
    desc.group_segment_size = meta.static_group_segment_size;
    desc.kernarg_address = slot;
    mcu::GpuAqlPacketTemplate templ =
        mcu::build_aql_packet_template(desc, mcu::kMcuAqlFastPolicy);
    mcu::GpuMcuRetainedPacket retained = mcu::make_retained_packet(templ);
    (void)hipMemcpy(retained_dev, &retained, sizeof(retained),
                    hipMemcpyHostToDevice);

    void* flags_host = nullptr;
    if (hipHostMalloc(&flags_host, 64u,
                      hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
        std::printf("FAIL: flags alloc\n");
        return 1;
    }
    auto* staged_flag = static_cast<unsigned int*>(flags_host);
    auto* go_flag = staged_flag + 1;
    *staged_flag = 0u;
    *go_flag = 0u;
    unsigned int* flags_dev = nullptr;
    (void)hipHostGetDevicePointer(reinterpret_cast<void**>(&flags_dev), flags_host, 0);

    StageCommitArgs sc_args{};
    sc_args.queue = queue.device_queue_view();
    sc_args.templ = retained;
    sc_args.queue_slot = 0ull;
    sc_args.staged_flag = flags_dev;
    sc_args.go_flag = flags_dev + 1;
    sc_args.published_dw0 = dw0_dev;
    sc_args.barrier = 1u;
    stage_then_commit_kernel<<<1, 1, 0, partition.control_stream()>>>(
        retained_dev, sc_args);
    check(hipGetLastError() == hipSuccess, "stage/commit kernel launched");

    const auto staged_deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (*staged_flag == 0u &&
           std::chrono::steady_clock::now() < staged_deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    check(*staged_flag != 0u, "producer staged the packet");

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(!completion.observe().is_complete(),
          "worker does not start from a staged-only packet");

    *go_flag = 1u;
    auto snapshot = completion.wait_for(mcu::GpuMcuCompletionState::complete, 5000);
    check(snapshot.ok(), "worker completed after commit");
    if (snapshot.ok()) {
        check(snapshot.value().sequence == 1u, "completion generation matched");
    }

    uint32_t host_output = 0;
    (void)hipMemcpy(&host_output, output_dev, sizeof(host_output),
                    hipMemcpyDeviceToHost);
    check(host_output == 7u, "worker output value");

    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL: device sync\n");
    }

    unsigned char published[64];
    std::memcpy(published,
                static_cast<unsigned char*>(queue.device_queue_view().ring_base),
                64u);
    bool body_ok = true;
    for (uint32_t i = 4; i < 64; ++i) {
        if (published[i] != static_cast<unsigned char>(templ.bytes[i])) {
            body_ok = false;
        }
    }
    check(body_ok, "committed packet body matches the traditional path");
    uint32_t published_dw0 = 0;
    (void)hipMemcpy(&published_dw0, dw0_dev, sizeof(published_dw0),
                    hipMemcpyDeviceToHost);
    check(published_dw0 == templ.publish_dw0, "committed DW0 matches publish_dw0");

    check(hipFreeHost(flags_host) == hipSuccess, "flags freed");
    check(hipFree(output_dev) == hipSuccess, "output freed");
    check(hipFree(dw0_dev) == hipSuccess, "dw0 freed");
    check(hipFree(retained_dev) == hipSuccess, "retained freed");
    check(completion.shutdown().ok(), "completion shutdown");
    check(code.shutdown().ok(), "code shutdown");
    check(queue.shutdown().ok(), "queue shutdown");
    check(partition.shutdown().ok(), "partition shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_aql_stage_commit: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_aql_stage_commit: PASS\n");
    return 0;
}
