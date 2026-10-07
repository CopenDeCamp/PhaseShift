#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/completion.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/worker_image.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

constexpr uint32_t kElements = 64;
constexpr uint32_t kWorkgroup = 64;

__global__ void mcu_producer_kernel(
    mcu::GpuMcuPublishBatchArgs args,
    uint32_t* status_word) {
    if (threadIdx.x != 0) return;
    const mcu::GpuMcuAqlStatus status = mcu::gpu_mcu_aql_publish_batch(args);
    bool caught = false;
    if (status == mcu::GpuMcuAqlStatus::COMPLETE) {
        caught = mcu::aql_wait_queue_caught_up(
            args.queue, mcu::kAqlQueueCaughtUpSpinLimit);
    }
    uint32_t code = static_cast<uint32_t>(status);
    if (status == mcu::GpuMcuAqlStatus::COMPLETE && caught) code |= 0x100u;
    __threadfence_system();
    *status_word = code;
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
    if (!partition_res.ok()) {
        std::printf("partition error: %s\n", partition_res.status().message().c_str());
        return 1;
    }
    auto partition = partition_res.release();

    auto queue_res = mcu::GpuMcuAqlQueue::create_with_mask(
        0, mcu::AqlQueueOwner::GpuMcu, mcu::kAqlDefaultQueueSize,
        partition.hsa_worker_cu_mask_view());
    check(queue_res.ok(), "worker aql queue created");
    if (!queue_res.ok()) {
        std::printf("queue error: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
    check(code_res.ok(), "worker code object loaded");
    if (!code_res.ok()) {
        std::printf("code error: %s\n", code_res.status().message().c_str());
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata& meta = code.metadata();
    check(meta.kernel_object != 0, "kernel_object resolved");

    auto completion_res = mcu::GpuMcuCompletionWord::create();
    check(completion_res.ok(), "completion word created");
    if (!completion_res.ok()) return 1;
    auto completion = completion_res.release();

    uint32_t* output_dev = nullptr;
    uint32_t* status_dev = nullptr;
    mcu::GpuAqlPacketTemplate* templates_dev = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&output_dev), kElements * sizeof(uint32_t)) ==
              hipSuccess &&
              hipMalloc(reinterpret_cast<void**>(&status_dev), sizeof(uint32_t)) ==
                  hipSuccess &&
              hipMalloc(reinterpret_cast<void**>(&templates_dev),
                        sizeof(mcu::GpuAqlPacketTemplate)) == hipSuccess,
          "device buffers allocated");
    if (g_failed != 0) return 1;
    (void)hipMemset(output_dev, 0, kElements * sizeof(uint32_t));
    (void)hipMemset(status_dev, 0, sizeof(uint32_t));

    const std::size_t slot_bytes = std::max(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
        mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuWorkerArgs)));
    auto region_res = queue.allocate_kernarg(slot_bytes, 1);
    check(region_res.ok(), "kernarg slot allocated");
    if (!region_res.ok()) return 1;
    const mcu::GpuMcuKernargRegion region = region_res.release();
    unsigned char* slot = region.slot(0);
    std::memset(slot, 0, region.slot_stride);

    constexpr uint32_t kIterations = 4;
    bool ok = true;
    for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        const uint32_t value = 100u * (iteration + 1u);
        const uint32_t generation = 7u + iteration;

        completion.reset();
        (void)hipMemset(status_dev, 0, sizeof(uint32_t));

        mcu::GpuMcuWorkerArgs args{};
        args.output = reinterpret_cast<uint64_t>(output_dev);
        args.completion = reinterpret_cast<uint64_t>(completion.device_ptr());
        args.value = value;
        args.element_count = kElements;
        args.generation = generation;
        args.status_detail = 0xABCDu;
        std::memcpy(slot, &args, sizeof(args));
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, kElements, 1u,
                                       1u, kWorkgroup);

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = kElements * kWorkgroup;
        desc.global_work_items_y = 1;
        desc.global_work_items_z = 1;
        desc.workgroup_size_x = kWorkgroup;
        desc.workgroup_size_y = 1;
        desc.workgroup_size_z = 1;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kMcuAqlFastPolicy);
        check(tmpl.publish_dw0 == 0x10102u, "packet template fast policy");
        if (hipMemcpy(templates_dev, &tmpl, sizeof(tmpl), hipMemcpyHostToDevice) !=
            hipSuccess) {
            check(false, "template upload");
            break;
        }

        mcu::GpuMcuPublishBatchArgs publish{};
        publish.queue = queue.device_queue_view();
        publish.packet_templates = templates_dev;
        publish.packet_count = 1;
        mcu_producer_kernel<<<1, 32, 0, partition.control_stream()>>>(publish, status_dev);
        if (hipGetLastError() != hipSuccess) {
            check(false, "producer launch");
            break;
        }
        if (hipStreamSynchronize(partition.control_stream()) != hipSuccess) {
            check(false, "producer stream sync");
            break;
        }

        uint32_t producer_status = 0;
        (void)hipMemcpy(&producer_status, status_dev, sizeof(uint32_t),
                        hipMemcpyDeviceToHost);
        check(producer_status == (static_cast<uint32_t>(mcu::GpuMcuAqlStatus::COMPLETE) |
                                  0x100u),
              "producer published and the CP consumed the packet");
        if (producer_status !=
            (static_cast<uint32_t>(mcu::GpuMcuAqlStatus::COMPLETE) | 0x100u)) {
            ok = false;
            break;
        }

        auto snapshot_res =
            completion.wait_for(mcu::GpuMcuCompletionState::complete, 10000);
        check(snapshot_res.ok(), "worker completion observed");
        if (!snapshot_res.ok()) {
            std::printf("completion error: %s\n",
                        snapshot_res.status().message().c_str());
            ok = false;
            break;
        }
        const mcu::GpuMcuCompletionSnapshot snap = snapshot_res.value();
        check(snap.is_complete(), "completion state complete");
        check(snap.sequence == generation, "completion generation matches dispatch");
        check(snap.detail == 0xABCDu, "completion detail propagated");

        std::vector<uint32_t> host_output(kElements, 0u);
        (void)hipMemcpy(host_output.data(), output_dev, kElements * sizeof(uint32_t),
                        hipMemcpyDeviceToHost);
        bool output_ok = true;
        for (uint32_t i = 0; i < kElements; ++i) {
            output_ok &= (host_output[i] == value + i);
        }
        check(output_ok, "worker output values");
        if (!output_ok) {
            ok = false;
            break;
        }
    }

    check(hipFree(output_dev) == hipSuccess, "output freed");
    check(hipFree(status_dev) == hipSuccess, "status freed");
    check(hipFree(templates_dev) == hipSuccess, "templates freed");
    check(completion.shutdown().ok(), "completion shutdown");
    check(code.shutdown().ok(), "code shutdown");
    check(queue.shutdown().ok(), "queue shutdown");
    check(partition.shutdown().ok(), "partition shutdown");

    if (!ok || g_failed != 0) {
        std::printf("test_gpu_mcu_device_enqueue: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_device_enqueue: PASS\n");
    return 0;
}
