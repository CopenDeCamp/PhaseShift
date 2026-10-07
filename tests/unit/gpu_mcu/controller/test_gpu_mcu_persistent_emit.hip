#include <phaseshift/runtime/gpu_mcu/infrastructure/completion.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/worker_image.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
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

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        q * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

constexpr uint32_t kPackets = 64;
constexpr uint32_t kWorkgroup = 64;

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
    check(queue_res.ok(), "worker aql queue created");
    if (!queue_res.ok()) return 1;
    auto queue = queue_res.release();

    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
    check(code_res.ok(), "worker code object loaded");
    if (!code_res.ok()) return 1;
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata& meta = code.metadata();

    std::vector<mcu::GpuMcuCompletionWord> completions;
    completions.reserve(kPackets);
    for (uint32_t i = 0; i < kPackets; ++i) {
        auto w = mcu::GpuMcuCompletionWord::create();
        if (!w.ok()) {
            std::printf("FAIL: completion alloc\n");
            return 1;
        }
        completions.push_back(w.release());
    }

    uint32_t* output_dev = nullptr;
    uint64_t* worker_ts_dev = nullptr;
    uint64_t* publish_ts_dev = nullptr;
    uint64_t* doorbell_ts_dev = nullptr;
    mcu::GpuAqlPacketTemplate* templates_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&output_dev), kPackets * sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&worker_ts_dev),
                  kPackets * 2 * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&publish_ts_dev),
                  kPackets * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&doorbell_ts_dev),
                  kPackets * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&templates_dev),
                  kPackets * sizeof(mcu::GpuAqlPacketTemplate)) != hipSuccess) {
        std::printf("FAIL: device buffers\n");
        return 1;
    }
    (void)hipMemset(worker_ts_dev, 0, kPackets * 2 * sizeof(uint64_t));
    (void)hipMemset(publish_ts_dev, 0, kPackets * sizeof(uint64_t));
    (void)hipMemset(doorbell_ts_dev, 0, kPackets * sizeof(uint64_t));

    const std::size_t slot_bytes = std::max(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
        mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuWorkerArgs)));
    auto region_res = queue.allocate_kernarg(slot_bytes, kPackets);
    check(region_res.ok(), "kernarg region allocated");
    if (!region_res.ok()) return 1;
    const mcu::GpuMcuKernargRegion region = region_res.release();

    std::vector<mcu::GpuAqlPacketTemplate> host_templates(kPackets);
    for (uint32_t i = 0; i < kPackets; ++i) {
        unsigned char* slot = region.slot(i);
        std::memset(slot, 0, region.slot_stride);
        mcu::GpuMcuWorkerArgs args{};
        args.output = reinterpret_cast<uint64_t>(output_dev) + i * sizeof(uint32_t);
        args.completion = reinterpret_cast<uint64_t>(completions[i].device_ptr());
        args.value = i;
        args.element_count = 1u;
        args.generation = i + 1u;
        args.status_detail = 0u;
        args.timestamps = reinterpret_cast<uint64_t>(worker_ts_dev) +
                          static_cast<uint64_t>(i) * 2ull * sizeof(unsigned long long);
        std::memcpy(slot, &args, sizeof(args));
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, 1u, 1u, 1u,
                                       kWorkgroup);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = 1u * kWorkgroup;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = kWorkgroup;
        desc.workgroup_size_y = 1;
        desc.workgroup_size_z = 1;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        host_templates[i] = mcu::build_aql_packet_template(desc, mcu::kMcuAqlFastPolicy);
    }
    if (hipMemcpy(templates_dev, host_templates.data(),
                  kPackets * sizeof(mcu::GpuAqlPacketTemplate),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: template upload\n");
        return 1;
    }

    bool ok = true;
    std::vector<double> kick_us;
    std::vector<double> doorbell_us;
    for (int round = 0; round < 2 && ok; ++round) {
        for (auto& c : completions) c.reset();
        (void)hipMemset(worker_ts_dev, 0, kPackets * 2 * sizeof(uint64_t));
        (void)hipMemset(publish_ts_dev, 0, kPackets * sizeof(uint64_t));
        (void)hipMemset(doorbell_ts_dev, 0, kPackets * sizeof(uint64_t));
        (void)hipMemset(output_dev, 0, kPackets * sizeof(uint32_t));

        auto mcu_res = mcu::GpuMcuPersistentMcu::create(0, partition.control_stream());
        check(mcu_res.ok(), "persistent mcu created");
        if (!mcu_res.ok()) {
            ok = false;
            break;
        }
        auto persistent = mcu_res.release();
        const bool barrier = (round % 2) == 0;
        check(persistent.enable_emit(queue.device_queue_view(), templates_dev, kPackets,
                                     publish_ts_dev, doorbell_ts_dev, 8, true, barrier)
                  .ok(),
              "persistent emit enabled (streaming)");
        check(persistent.start().ok(), "persistent mcu started");

        const uint64_t h0 = persistent.heartbeat();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        check(persistent.heartbeat() > h0, "heartbeat advances while emitting");

        check(persistent.request_emit(kPackets).ok(), "emit batch requested");
        bool all_complete = false;
        uint32_t observed = 0;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
        while (std::chrono::steady_clock::now() < deadline) {
            observed = 0;
            for (uint32_t i = 0; i < kPackets; ++i) {
                if (completions[i].observe().is_complete()) ++observed;
            }
            if (observed == kPackets) {
                all_complete = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (!all_complete) {
            std::printf("completed=%u/%u published=%llu failures=%llu heartbeat=%llu\n",
                        observed, kPackets,
                        static_cast<unsigned long long>(persistent.published()),
                        static_cast<unsigned long long>(persistent.publish_failures()),
                        static_cast<unsigned long long>(persistent.heartbeat()));
        }
        check(all_complete, "every worker completion observed");
        check(persistent.published() == kPackets, "persistent published all packets");
        check(persistent.publish_failures() == 0, "no publish failures");

        std::vector<uint64_t> worker_ts(kPackets * 2, 0);
        std::vector<uint64_t> publish_ts(kPackets, 0);
        std::vector<uint64_t> doorbell_ts(kPackets, 0);
        std::vector<uint32_t> host_output(kPackets, 0);
        hipStream_t read_stream = partition.worker_stream();
        (void)hipMemcpyAsync(worker_ts.data(), worker_ts_dev,
                             worker_ts.size() * sizeof(uint64_t),
                             hipMemcpyDeviceToHost, read_stream);
        (void)hipMemcpyAsync(publish_ts.data(), publish_ts_dev,
                             publish_ts.size() * sizeof(uint64_t),
                             hipMemcpyDeviceToHost, read_stream);
        (void)hipMemcpyAsync(doorbell_ts.data(), doorbell_ts_dev,
                             doorbell_ts.size() * sizeof(uint64_t),
                             hipMemcpyDeviceToHost, read_stream);
        (void)hipMemcpyAsync(host_output.data(), output_dev,
                             kPackets * sizeof(uint32_t), hipMemcpyDeviceToHost,
                             read_stream);
        (void)hipStreamSynchronize(read_stream);

        bool output_ok = true;
        bool ordering_ok = true;
        bool timestamps_ok = true;
        uint64_t previous_worker = 0;
        for (uint32_t i = 0; i < kPackets; ++i) {
            output_ok &= (host_output[i] == i);
            ordering_ok &= (publish_ts[i] != 0) && (worker_ts[i * 2] != 0);
            if (worker_ts[i * 2] < previous_worker) timestamps_ok = false;
            previous_worker = worker_ts[i * 2];
            kick_us.push_back(
                static_cast<double>(worker_ts[i * 2] - publish_ts[i]) / 100.0);
            doorbell_us.push_back(
                static_cast<double>(doorbell_ts[i] - publish_ts[i]) / 100.0);
        }
        check(output_ok, "worker output values");
        check(ordering_ok, "publish and worker timestamps recorded");
        check(timestamps_ok, "worker start timestamps monotonic");

        check(persistent.request_stop().ok(), "stop requested");
        check(persistent.wait_stopped(5000).ok(), "persistent mcu stopped");
        check(persistent.shutdown().ok(), "persistent mcu shutdown");
        if (!all_complete || !output_ok || !ordering_ok || !timestamps_ok) ok = false;
    }

    if (!kick_us.empty()) {
        std::printf("persistent emit (streaming): n=%zu "
                    "publish_to_doorbell_us p50=%.3f "
                    "publish_to_worker_start_us p50=%.3f\n",
                    kick_us.size(), percentile(doorbell_us, 0.5),
                    percentile(kick_us, 0.5));
    }

    check(hipFree(output_dev) == hipSuccess, "output freed");
    check(hipFree(worker_ts_dev) == hipSuccess, "worker ts freed");
    check(hipFree(publish_ts_dev) == hipSuccess, "publish ts freed");
    check(hipFree(doorbell_ts_dev) == hipSuccess, "doorbell ts freed");
    check(hipFree(templates_dev) == hipSuccess, "templates freed");
    check(code.shutdown().ok(), "code shutdown");
    check(queue.shutdown().ok(), "queue shutdown");
    check(partition.shutdown().ok(), "partition shutdown");

    if (!ok || g_failed != 0) {
        std::printf("test_gpu_mcu_persistent_emit: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_persistent_emit: PASS\n");
    return 0;
}
