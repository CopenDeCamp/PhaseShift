#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/completion.h>
#include <phaseshift/runtime/gpu_mcu/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/device_completion.h>
#include <phaseshift/runtime/gpu_mcu/fsm_worker.h>
#include <phaseshift/runtime/gpu_mcu/micro_fsm.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>
#include <phaseshift/runtime/gpu_mcu/retained_packet.h>
#include <phaseshift/runtime/gpu_mcu/worker_image.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

struct Options {
    std::string variant = "hip";
    int warmup = 20;
    int iterations = 200;
    int n = 64;
    int batch = 1;
    int work = 0;
    int barrier = 1;
    int copy_lanes = 1;
    int queue_ahead = 0;
    int doorbell_batch = 1;
    bool debug_records = false;
    std::string fence_scope = "agent";
    int device = 0;
    bool worker_mask = true;
    bool producer_on_control = true;
};

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: phaseshift-bench gpu-dispatch [options]\n"
                 "  --variant hip|hip-graph|device-aql|device-aql-chain|host-aql|queue-mem|"
                 "device-aql-fsm|device-aql-fsm-lds|device-aql-fsm-feed|"
                 "device-aql-fsm-feed-lds|device-aql-fsm-feed-batch\n"
                 "  --warmup N\n"
                 "  --iterations N\n"
                 "  --n N            kernels per chain / dispatches per fed plan\n"
                 "  --batch N        packets per doorbell (device-aql)\n"
                 "  --work N         in-kernel fma iterations (hip / hip-graph / fsm)\n"
                 "  --copy-lanes N   LDS template copy lanes (fsm-lds: 1 or 32)\n"
                 "  --queue-ahead N  target AQL queue depth for feed variants (0 = ring)\n"
                 "  --doorbell-batch N  packets per doorbell (feed-batch)\n"
                 "  --debug-records  enable MCU queue counters (adds overhead)\n"
                 "  --no-worker-mask\n"
                 "  --producer control|default\n"
                 "  --device N\n");
}

void report_node_validity(const std::vector<unsigned long long>& ts,
                          int n,
                          const char* label) {
    int ran = 0;
    bool monotonic = true;
    for (int i = 0; i < n; ++i) {
        if (ts[static_cast<std::size_t>(i) * 3u + 1u] >= 1u) ++ran;
        if (i > 0 &&
            ts[static_cast<std::size_t>(i - 1) * 3u] >= ts[static_cast<std::size_t>(i) * 3u]) {
            monotonic = false;
        }
    }
    std::printf("  %s nodes_ran=%d/%d start_monotonic=%s\n", label, ran, n,
                monotonic ? "yes" : "NO");
}

double percentile(std::vector<double> values, double q) {    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        q * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

__global__ void empty_kernel(unsigned long long* timestamps, unsigned int slot,
                             int work) {
    const unsigned long long start = wall_clock64();
    timestamps[slot * 3u] = start;
    float x = static_cast<float>(threadIdx.x);
    for (int i = 0; i < work; ++i) {
        x = fmaf(x, 1.0001f, 0.5f);
    }
    timestamps[slot * 3u + 1u] = 1u + static_cast<unsigned long long>(x == 123456.0f);
    timestamps[slot * 3u + 2u] = wall_clock64();
}

__global__ void aql_producer_kernel(
    mcu::GpuMcuPublishBatchArgs args,
    unsigned long long* stages,
    unsigned int* status) {
    if (threadIdx.x != 0) return;
    stages[0] = wall_clock64();
    uint64_t base_slot = 0;
    if (!mcu::aql_reserve_packets(args.queue, args.packet_count, &base_slot,
                                  mcu::kAqlQueueFullSpinLimit)) {
        *status = 1u;
        return;
    }
    stages[1] = wall_clock64();
    for (uint32_t i = 0; i < args.packet_count; ++i) {
        mcu::construct_aql_packet_nonvolatile(
            mcu::aql_packet_at(args.queue, base_slot + i), args.packet_templates[i]);
    }
    stages[2] = wall_clock64();
    for (uint32_t i = 0; i < args.packet_count; ++i) {
        mcu::aql_publish_dw0(mcu::aql_packet_at(args.queue, base_slot + i),
                             args.packet_templates[i].publish_dw0);
    }
    stages[3] = wall_clock64();
    mcu::ring_aql_doorbell(args.queue, base_slot + args.packet_count - 1u);
    stages[4] = wall_clock64();
    *status = 0u;
}


__global__ void aql_stage_kernel(mcu::GpuMcuPublishBatchArgs args,
                                 uint32_t reps,
                                 uint32_t mode,
                                 unsigned long long* s,
                                 unsigned long long* out) {
    if (threadIdx.x != 0) return;
    const mcu::DeviceAqlQueueView q = args.queue;
    const bool read_read = (mode & 1u) != 0u;
    const bool read_write = (mode & 2u) != 0u;
    const bool seq_cst = (mode & 4u) != 0u;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        s[0] = wall_clock64();
        const uint64_t w = read_write ? *q.write_index : 0ull;
        s[1] = wall_clock64();
        const uint64_t r = read_read ? *q.read_index : 0ull;
        s[2] = wall_clock64();
        const uint64_t base = seq_cst
            ? __atomic_fetch_add(q.write_index,
                                 static_cast<unsigned long long>(args.packet_count),
                                 __ATOMIC_SEQ_CST)
            : __atomic_fetch_add(q.write_index,
                                 static_cast<unsigned long long>(args.packet_count),
                                 __ATOMIC_RELAXED);
        s[3] = wall_clock64();
        for (uint32_t i = 0; i < args.packet_count; ++i) {
            mcu::construct_aql_packet_nonvolatile(
                mcu::aql_packet_at(q, base + i), args.packet_templates[i]);
        }
        s[4] = wall_clock64();
        for (uint32_t i = 0; i < args.packet_count; ++i) {
            mcu::aql_publish_dw0(mcu::aql_packet_at(q, base + i),
                                 args.packet_templates[i].publish_dw0);
        }
        s[5] = wall_clock64();
        mcu::ring_aql_doorbell(q, base + args.packet_count - 1u);
        s[6] = wall_clock64();
        out[0] = w - r;
        out[1] = base;
    }
}

struct DispatchMetrics {
    double producer_submit_us = 0.0;
    double dispatch_latency_us = 0.0;
    double worker_duration_us = 0.0;
    double host_submit_us = 0.0;
    double host_wall_us = 0.0;
};

int run_hip_chain(const Options& opt) {
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    unsigned long long* ts_dev = nullptr;
    const std::size_t entries = static_cast<std::size_t>(opt.n) * 3u;
    if (hipMalloc(&ts_dev, entries * sizeof(unsigned long long)) != hipSuccess) return 1;
    (void)hipMemset(ts_dev, 0, entries * sizeof(unsigned long long));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);

    std::vector<double> totals;
    std::vector<double> per_kernel;
    std::vector<double> gaps;
    std::vector<double> periods;
    std::vector<unsigned long long> last_ts;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        (void)hipMemsetAsync(ts_dev, 0, entries * sizeof(unsigned long long), stream);
        (void)hipEventRecord(start, stream);
        for (int i = 0; i < opt.n; ++i) {
            empty_kernel<<<1, 32, 0, stream>>>(ts_dev, static_cast<unsigned int>(i), opt.work);
        }
        (void)hipEventRecord(stop, stream);
        if (hipStreamSynchronize(stream) != hipSuccess) return 1;
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        if (it < opt.warmup) continue;
        totals.push_back(static_cast<double>(ms) * 1000.0);

        std::vector<unsigned long long> ts(entries, 0);
        (void)hipMemcpy(ts.data(), ts_dev, entries * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost);
        last_ts = ts;
        for (int i = 0; i < opt.n; ++i) {
            const unsigned long long a = ts[static_cast<std::size_t>(i) * 3u];
            const unsigned long long b = ts[static_cast<std::size_t>(i) * 3u + 2u];
            per_kernel.push_back(static_cast<double>(b - a) / 100.0);
            if (i > 0) {
                const unsigned long long prev =
                    ts[static_cast<std::size_t>(i - 1) * 3u + 2u];
                gaps.push_back(static_cast<double>(a - prev) / 100.0);
                periods.push_back(static_cast<double>(
                    a - ts[static_cast<std::size_t>(i - 1) * 3u]) / 100.0);
            }
        }
    }
    std::printf("gpu-dispatch variant=hip n=%d iterations=%d work=%d\n", opt.n,
                opt.iterations, opt.work);
    report_node_validity(last_ts, opt.n, "chain:");
    std::printf("  chain_total_us p50=%.2f p95=%.2f min=%.2f\n",
                percentile(totals, 0.5), percentile(totals, 0.95),
                percentile(totals, 0.0));
    std::printf("  per_kernel_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(per_kernel, 0.5), percentile(per_kernel, 0.95),
                percentile(per_kernel, 0.0));
    std::printf("  inter_kernel_gap_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(gaps, 0.5), percentile(gaps, 0.95), percentile(gaps, 0.0));
    std::printf("  start_to_start_period_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(periods, 0.5), percentile(periods, 0.95),
                percentile(periods, 0.0));

    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    (void)hipFree(ts_dev);
    (void)hipStreamDestroy(stream);
    return 0;
}

int run_hip_graph(const Options& opt) {
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    unsigned long long* ts_dev = nullptr;
    const std::size_t entries = static_cast<std::size_t>(opt.n) * 3u;
    if (hipMalloc(&ts_dev, entries * sizeof(unsigned long long)) != hipSuccess) return 1;
    (void)hipMemset(ts_dev, 0, entries * sizeof(unsigned long long));

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    if (hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) != hipSuccess) {
        std::printf("FAIL: capture unsupported\n");
        return 1;
    }
    for (int i = 0; i < opt.n; ++i) {
        empty_kernel<<<1, 32, 0, stream>>>(ts_dev, static_cast<unsigned int>(i), opt.work);
    }
    if (hipStreamEndCapture(stream, &graph) != hipSuccess) {
        std::printf("FAIL: end capture\n");
        return 1;
    }
    if (hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess) {
        std::printf("FAIL: graph instantiate\n");
        return 1;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);

    std::vector<double> totals;
    std::vector<double> per_kernel;
    std::vector<double> gaps;
    std::vector<double> periods;
    std::vector<unsigned long long> last_ts;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        (void)hipMemsetAsync(ts_dev, 0, entries * sizeof(unsigned long long), stream);
        (void)hipEventRecord(start, stream);
        if (hipGraphLaunch(exec, stream) != hipSuccess) return 1;
        (void)hipEventRecord(stop, stream);
        if (hipStreamSynchronize(stream) != hipSuccess) return 1;
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        if (it < opt.warmup) continue;
        totals.push_back(static_cast<double>(ms) * 1000.0);

        std::vector<unsigned long long> ts(entries, 0);
        (void)hipMemcpy(ts.data(), ts_dev, entries * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost);
        last_ts = ts;
        for (int i = 0; i < opt.n; ++i) {
            const unsigned long long a = ts[static_cast<std::size_t>(i) * 3u];
            const unsigned long long b = ts[static_cast<std::size_t>(i) * 3u + 2u];
            per_kernel.push_back(static_cast<double>(b - a) / 100.0);
            if (i > 0) {
                const unsigned long long prev =
                    ts[static_cast<std::size_t>(i - 1) * 3u + 2u];
                gaps.push_back(static_cast<double>(a - prev) / 100.0);
                periods.push_back(static_cast<double>(
                    a - ts[static_cast<std::size_t>(i - 1) * 3u]) / 100.0);
            }
        }
    }
    std::printf("gpu-dispatch variant=hip-graph n=%d iterations=%d work=%d\n", opt.n,
                opt.iterations, opt.work);
    report_node_validity(last_ts, opt.n, "graph:");
    std::printf("  replay_total_us p50=%.2f p95=%.2f min=%.2f\n",
                percentile(totals, 0.5), percentile(totals, 0.95),
                percentile(totals, 0.0));
    std::printf("  per_node_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(per_kernel, 0.5), percentile(per_kernel, 0.95),
                percentile(per_kernel, 0.0));
    std::printf("  inter_node_gap_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(gaps, 0.5), percentile(gaps, 0.95), percentile(gaps, 0.0));
    std::printf("  start_to_start_period_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(periods, 0.5), percentile(periods, 0.95),
                percentile(periods, 0.0));

    (void)hipGraphExecDestroy(exec);
    (void)hipGraphDestroy(graph);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    (void)hipFree(ts_dev);
    (void)hipStreamDestroy(stream);
    return 0;
}

struct DeviceAqlRig {
    mcu::GpuMcuCuPartition partition{};
    mcu::GpuMcuAqlQueue queue{};
    mcu::GpuMcuAqlCodeObject code{};
    mcu::GpuMcuCompletionWord completion{};
    mcu::GpuMcuKernargRegion region{};
    uint32_t* output_dev = nullptr;
    unsigned long long* worker_ts_dev = nullptr;
    unsigned long long* stages_dev = nullptr;
    unsigned int* status_dev = nullptr;
    mcu::GpuAqlPacketTemplate* templates_dev = nullptr;
    bool ok = false;
};

bool device_aql_setup(const Options& opt, DeviceAqlRig& rig) {
    auto partition_res = mcu::GpuMcuCuPartition::create(opt.device, 1u);
    if (!partition_res.ok()) return false;
    rig.partition = partition_res.release();

    auto queue_res = opt.worker_mask
                         ? mcu::GpuMcuAqlQueue::create_with_mask(
                               opt.device, mcu::AqlQueueOwner::GpuMcu,
                               mcu::kAqlDefaultQueueSize,
                               rig.partition.hsa_worker_cu_mask_view())
                         : mcu::GpuMcuAqlQueue::create(
                               opt.device, mcu::AqlQueueOwner::GpuMcu,
                               mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) return false;
    rig.queue = queue_res.release();

    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        rig.queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
    if (!code_res.ok()) return false;
    rig.code = code_res.release();

    auto completion_res = mcu::GpuMcuCompletionWord::create();
    if (!completion_res.ok()) return false;
    rig.completion = completion_res.release();

    const auto& meta = rig.code.metadata();
    const std::size_t slot_bytes = std::max(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
        mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuWorkerArgs)));
    auto region_res = rig.queue.allocate_kernarg(slot_bytes, 1);
    if (!region_res.ok()) return false;
    rig.region = region_res.release();

    if (hipMalloc(reinterpret_cast<void**>(&rig.output_dev), 64 * sizeof(uint32_t)) !=
        hipSuccess) {
        return false;
    }
    if (hipMalloc(reinterpret_cast<void**>(&rig.worker_ts_dev),
                  2 * sizeof(unsigned long long)) != hipSuccess) {
        return false;
    }
    if (hipMalloc(reinterpret_cast<void**>(&rig.stages_dev),
                  6 * sizeof(unsigned long long)) != hipSuccess) {
        return false;
    }
    if (hipMalloc(reinterpret_cast<void**>(&rig.status_dev), sizeof(unsigned int)) !=
        hipSuccess) {
        return false;
    }
    if (hipMalloc(reinterpret_cast<void**>(&rig.templates_dev),
                  sizeof(mcu::GpuAqlPacketTemplate)) != hipSuccess) {
        return false;
    }

    mcu::GpuMcuWorkerArgs args{};
    args.output = reinterpret_cast<uint64_t>(rig.output_dev);
    args.completion = reinterpret_cast<uint64_t>(rig.completion.device_ptr());
    args.value = 7u;
    args.element_count = 64u;
    args.generation = 1u;
    args.status_detail = 0u;
    args.timestamps = reinterpret_cast<uint64_t>(rig.worker_ts_dev);
    std::memcpy(rig.region.slot(0), &args, sizeof(args));
    mcu::build_aql_launch_metadata(rig.region.slot(0), meta.kernarg_segment_size, 64u,
                                   1u, 1u, 64u);

    mcu::GpuAqlDispatchDesc desc{};
    desc.kernel_object = meta.kernel_object;
    desc.global_work_items_x = 64u * 64u;
    desc.global_work_items_y = 1u;
    desc.global_work_items_z = 1u;
    desc.workgroup_size_x = 64;
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;
    desc.private_segment_size = meta.private_segment_size;
    desc.group_segment_size = meta.static_group_segment_size;
    desc.kernarg_address = rig.region.slot(0);
    const mcu::GpuAqlPacketTemplate tmpl =
        mcu::build_aql_packet_template(desc, mcu::kMcuAqlFastPolicy);
    if (hipMemcpy(rig.templates_dev, &tmpl, sizeof(tmpl), hipMemcpyHostToDevice) !=
        hipSuccess) {
        return false;
    }
    rig.ok = true;
    return true;
}

void device_aql_teardown(DeviceAqlRig& rig) {
    if (rig.templates_dev) (void)hipFree(rig.templates_dev);
    if (rig.status_dev) (void)hipFree(rig.status_dev);
    if (rig.stages_dev) (void)hipFree(rig.stages_dev);
    if (rig.worker_ts_dev) (void)hipFree(rig.worker_ts_dev);
    if (rig.output_dev) (void)hipFree(rig.output_dev);
    (void)rig.completion.shutdown();
    (void)rig.code.shutdown();
    (void)rig.queue.shutdown();
    (void)rig.partition.shutdown();
}

int run_device_aql(const Options& opt) {
    DeviceAqlRig rig{};
    if (!device_aql_setup(opt, rig)) {
        std::printf("FAIL: device-aql setup\n");
        return 1;
    }
    std::vector<double> submit_us;
    std::vector<double> dispatch_us;
    std::vector<double> worker_us;
    std::vector<double> reserve_us;
    std::vector<double> body_us;
    std::vector<double> publish_us;
    std::vector<double> doorbell_us;
    std::vector<double> doorbell_to_worker_us;
    hipStream_t producer_stream =
        opt.producer_on_control ? rig.partition.control_stream() : nullptr;
    bool ok = true;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        rig.completion.reset();
        (void)hipMemset(rig.status_dev, 0, sizeof(unsigned int));
        (void)hipMemset(rig.stages_dev, 0, 6 * sizeof(unsigned long long));
        (void)hipMemset(rig.worker_ts_dev, 0, 2 * sizeof(unsigned long long));

        mcu::GpuMcuPublishBatchArgs publish{};
        publish.queue = rig.queue.device_queue_view();
        publish.packet_templates = rig.templates_dev;
        publish.packet_count = static_cast<uint32_t>(opt.batch);
        aql_producer_kernel<<<1, 32, 0, producer_stream>>>(
            publish, rig.stages_dev, rig.status_dev);
        if (hipStreamSynchronize(producer_stream) != hipSuccess) {
            ok = false;
            break;
        }
        auto snapshot = rig.completion.wait_for(mcu::GpuMcuCompletionState::complete,
                                                10000);
        if (!snapshot.ok()) {
            ok = false;
            break;
        }
        if (it < opt.warmup) continue;

        unsigned long long stages[6] = {};
        unsigned long long worker_ts[2] = {};
        (void)hipMemcpy(stages, rig.stages_dev, sizeof(stages), hipMemcpyDeviceToHost);
        (void)hipMemcpy(worker_ts, rig.worker_ts_dev, sizeof(worker_ts),
                        hipMemcpyDeviceToHost);
        submit_us.push_back(static_cast<double>(stages[4] - stages[0]) / 100.0);
        dispatch_us.push_back(static_cast<double>(worker_ts[0] - stages[0]) / 100.0);
        worker_us.push_back(static_cast<double>(worker_ts[1] - worker_ts[0]) / 100.0);
        reserve_us.push_back(static_cast<double>(stages[1] - stages[0]) / 100.0);
        body_us.push_back(static_cast<double>(stages[2] - stages[1]) / 100.0);
        publish_us.push_back(static_cast<double>(stages[3] - stages[2]) / 100.0);
        doorbell_us.push_back(static_cast<double>(stages[4] - stages[3]) / 100.0);
        doorbell_to_worker_us.push_back(
            static_cast<double>(worker_ts[0] - stages[4]) / 100.0);
    }
    if (!ok) {
        device_aql_teardown(rig);
        std::printf("FAIL: device-aql dispatch\n");
        return 1;
    }
    std::printf("gpu-dispatch variant=device-aql batch=%d iterations=%d\n", opt.batch,
                opt.iterations);
    std::printf("  producer_submit_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(submit_us, 0.5), percentile(submit_us, 0.95),
                percentile(submit_us, 0.0));
    std::printf("  producer_to_worker_start_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(dispatch_us, 0.5), percentile(dispatch_us, 0.95),
                percentile(dispatch_us, 0.0));
    std::printf("  worker_duration_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(worker_us, 0.5), percentile(worker_us, 0.95),
                percentile(worker_us, 0.0));
    std::printf("  stage breakdown p50: reserve=%.3f body=%.3f publish=%.3f "
                "doorbell=%.3f doorbell_to_worker=%.3f\n",
                percentile(reserve_us, 0.5), percentile(body_us, 0.5),
                percentile(publish_us, 0.5), percentile(doorbell_us, 0.5),
                percentile(doorbell_to_worker_us, 0.5));
    device_aql_teardown(rig);
    return 0;
}

int run_device_aql_chain(const Options& opt) {
    const uint32_t n = static_cast<uint32_t>(std::max(std::min(opt.n, 64), 1));
    auto partition_res = mcu::GpuMcuCuPartition::create(opt.device, 1u);
    if (!partition_res.ok()) {
        std::printf("FAIL: partition\n");
        return 1;
    }
    auto partition = partition_res.release();
    auto queue_res = mcu::GpuMcuAqlQueue::create_with_mask(
        opt.device, mcu::AqlQueueOwner::GpuMcu, mcu::kAqlDefaultQueueSize,
        partition.hsa_worker_cu_mask_view());
    if (!queue_res.ok()) {
        std::printf("FAIL: queue\n");
        return 1;
    }
    auto queue = queue_res.release();
    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: code\n");
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata& meta = code.metadata();

    std::vector<mcu::GpuMcuCompletionWord> completions;
    completions.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto w = mcu::GpuMcuCompletionWord::create();
        if (!w.ok()) {
            std::printf("FAIL: completion\n");
            return 1;
        }
        completions.push_back(w.release());
    }

    uint32_t* output_dev = nullptr;
    unsigned long long* worker_ts_dev = nullptr;
    mcu::GpuAqlPacketTemplate* templates_dev = nullptr;
    unsigned long long* stages_dev = nullptr;
    unsigned int* status_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&output_dev), n * sizeof(uint32_t)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&worker_ts_dev),
                  n * 2 * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&templates_dev),
                  n * sizeof(mcu::GpuAqlPacketTemplate)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&stages_dev),
                  6 * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&status_dev), sizeof(unsigned int)) !=
            hipSuccess) {
        std::printf("FAIL: device buffers\n");
        return 1;
    }

    const std::size_t slot_bytes = std::max(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
        mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuWorkerArgs)));
    auto region_res = queue.allocate_kernarg(slot_bytes, n);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg\n");
        return 1;
    }
    const mcu::GpuMcuKernargRegion region = region_res.release();

    std::vector<mcu::GpuAqlPacketTemplate> host_templates(n);
    for (uint32_t i = 0; i < n; ++i) {
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
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, 1u, 1u, 1u, 32u);

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = 1u * 32u;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = 32;
        desc.workgroup_size_y = 1;
        desc.workgroup_size_z = 1;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        host_templates[i] = mcu::build_aql_packet_template(desc, mcu::kMcuAqlFastPolicy);
    }
    if (hipMemcpy(templates_dev, host_templates.data(),
                  n * sizeof(mcu::GpuAqlPacketTemplate),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: template upload\n");
        return 1;
    }

    std::vector<double> cold_us;
    std::vector<double> hot_gap_us;
    std::vector<double> worker_us;
    bool ok = true;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        for (auto& c : completions) c.reset();
        (void)hipMemset(stages_dev, 0, 6 * sizeof(unsigned long long));
        (void)hipMemset(worker_ts_dev, 0, n * 2 * sizeof(unsigned long long));
        mcu::GpuMcuPublishBatchArgs publish{};
        publish.queue = queue.device_queue_view();
        publish.packet_templates = templates_dev;
        publish.packet_count = n;
        aql_producer_kernel<<<1, 32, 0, partition.control_stream()>>>(
            publish, stages_dev, status_dev);
        if (hipStreamSynchronize(partition.control_stream()) != hipSuccess) {
            ok = false;
            break;
        }
        for (uint32_t i = 0; i < n; ++i) {
            auto snap = completions[i].wait_for(mcu::GpuMcuCompletionState::complete,
                                                10000);
            if (!snap.ok()) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
        if (it < opt.warmup) continue;

        unsigned long long stages[6] = {};
        std::vector<unsigned long long> ts(n * 2, 0);
        (void)hipMemcpy(stages, stages_dev, sizeof(stages), hipMemcpyDeviceToHost);
        (void)hipMemcpy(ts.data(), worker_ts_dev, ts.size() * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost);
        cold_us.push_back(static_cast<double>(ts[0] - stages[4]) / 100.0);
        for (uint32_t i = 1; i < n; ++i) {
            hot_gap_us.push_back(
                static_cast<double>(ts[i * 2] - ts[(i - 1) * 2 + 1]) / 100.0);
        }
        for (uint32_t i = 0; i < n; ++i) {
            worker_us.push_back(
                static_cast<double>(ts[i * 2 + 1] - ts[i * 2]) / 100.0);
        }
    }
    if (!ok) {
        std::printf("FAIL: device-aql-chain dispatch\n");
        (void)code.shutdown();
        (void)queue.shutdown();
        (void)partition.shutdown();
        return 1;
    }
    std::printf("gpu-dispatch variant=device-aql-chain n=%u iterations=%d\n", n,
                opt.iterations);
    std::printf("  cold_doorbell_to_first_worker_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(cold_us, 0.5), percentile(cold_us, 0.95),
                percentile(cold_us, 0.0));
    std::printf("  hot_prev_worker_end_to_next_start_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(hot_gap_us, 0.5), percentile(hot_gap_us, 0.95),
                percentile(hot_gap_us, 0.0));
    std::printf("  worker_duration_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(worker_us, 0.5), percentile(worker_us, 0.95),
                percentile(worker_us, 0.0));

    (void)hipFree(output_dev);
    (void)hipFree(worker_ts_dev);
    (void)hipFree(templates_dev);
    (void)hipFree(stages_dev);
    (void)hipFree(status_dev);
    (void)code.shutdown();
    (void)queue.shutdown();
    (void)partition.shutdown();
    return 0;
}

__global__ void flush_l2_kernel(unsigned long long* buf, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        buf[i] = i;
    }
}

__global__ void timed_load_kernel(const unsigned long long* target,
                                  unsigned long long* out) {
    __threadfence();
    const unsigned long long a = wall_clock64();
    const unsigned long long v =
        *reinterpret_cast<const volatile unsigned long long*>(target);
    const unsigned long long b = wall_clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = b - a;
        out[1] = v;
    }
}

double cold_load_ns(const volatile void* target,
                    unsigned long long* flush,
                    std::size_t flush_words,
                    unsigned long long* out) {
    const unsigned long long* t = (const unsigned long long*)target;
    std::vector<double> samples;
    for (int r = 0; r < 40; ++r) {
        flush_l2_kernel<<<1024, 256>>>(flush, flush_words);
        timed_load_kernel<<<1, 1>>>(t, out);
        (void)hipDeviceSynchronize();
        unsigned long long t[2] = {};
        (void)hipMemcpy(t, out, sizeof(t), hipMemcpyDeviceToHost);
        if (r < 8) continue;
        samples.push_back(static_cast<double>(t[0]) * 10.21);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int run_queue_mem(const Options& opt) {
    auto queue_res = mcu::GpuMcuAqlQueue::create(opt.device, mcu::AqlQueueOwner::GpuMcu,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue\n");
        return 1;
    }
    auto queue = queue_res.release();
    const auto& view = queue.device_queue_view();

    unsigned long long* out = nullptr;
    unsigned long long* flush = nullptr;
    unsigned long long* dev = nullptr;
    unsigned long long* mapped = nullptr;
    unsigned long long* host = nullptr;
    const std::size_t flush_words = 64ull * 1024 * 1024 / sizeof(unsigned long long);
    if (hipMalloc(reinterpret_cast<void**>(&out), 2 * sizeof(unsigned long long)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&flush),
                  flush_words * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&dev), 4096) != hipSuccess ||
        hipHostMalloc(reinterpret_cast<void**>(&mapped), 4096,
                      hipHostMallocMapped) != hipSuccess ||
        hipHostMalloc(reinterpret_cast<void**>(&host), 4096, 0) != hipSuccess) {
        std::printf("FAIL: allocations\n");
        return 1;
    }
    *dev = 0;
    *mapped = 0;
    *host = 0;

    std::printf("gpu-dispatch variant=queue-mem (cold single load, L2 flushed)\n");
    std::printf("  hipMalloc VRAM            %8.1f ns\n",
                cold_load_ns(dev, flush, flush_words, out));
    std::printf("  hipHostMallocMapped sys   %8.1f ns\n",
                cold_load_ns(mapped, flush, flush_words, out));
    std::printf("  hipHostMalloc pinned sys  %8.1f ns\n",
                cold_load_ns(host, flush, flush_words, out));
    std::printf("  hsa_queue ring_base       %8.1f ns\n",
                cold_load_ns(view.ring_base, flush, flush_words, out));
    std::printf("  hsa_queue write_index     %8.1f ns\n",
                cold_load_ns(view.write_index, flush, flush_words, out));
    std::printf("  hsa_queue read_index      %8.1f ns\n",
                cold_load_ns(view.read_index, flush, flush_words, out));
    std::printf("  hsa_queue_t               %8.1f ns\n",
                cold_load_ns(view.queue_ptr, flush, flush_words, out));

    (void)hipFree(out);
    (void)hipFree(flush);
    (void)hipFree(dev);
    (void)hipFreeHost(mapped);
    (void)hipFreeHost(host);
    (void)queue.shutdown();
    return 0;
}


int run_device_aql_stage(const Options& opt) {
    DeviceAqlRig rig{};
    if (!device_aql_setup(opt, rig)) {
        std::printf("FAIL: device-aql-stage setup\n");
        return 1;
    }
    unsigned long long* stage_dev = nullptr;
    unsigned long long* out_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&stage_dev), 8 * sizeof(unsigned long long)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&out_dev), 2 * sizeof(unsigned long long)) !=
            hipSuccess) {
        std::printf("FAIL: stage buffers\n");
        return 1;
    }
    std::vector<double> read_w, read_r, fetch, body, publish, doorbell, total;
    bool ok = true;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        rig.completion.reset();
        (void)hipMemset(stage_dev, 0, 8 * sizeof(unsigned long long));
        mcu::GpuMcuPublishBatchArgs publish_args{};
        publish_args.queue = rig.queue.device_queue_view();
        publish_args.packet_templates = rig.templates_dev;
        publish_args.packet_count = 1;
        aql_stage_kernel<<<1, 32, 0, rig.partition.control_stream()>>>(
            publish_args, static_cast<uint32_t>(std::max(opt.batch, 1)),
            static_cast<uint32_t>(std::max(opt.work, 0)), stage_dev, out_dev);
        if (hipStreamSynchronize(rig.partition.control_stream()) != hipSuccess) {
            ok = false;
            break;
        }
        auto snapshot = rig.completion.wait_for(mcu::GpuMcuCompletionState::complete,
                                                10000);
        if (!snapshot.ok()) {
            ok = false;
            break;
        }
        if (it < opt.warmup) continue;
        unsigned long long st[8] = {};
        (void)hipMemcpy(st, stage_dev, sizeof(st), hipMemcpyDeviceToHost);
        read_w.push_back(static_cast<double>(st[1] - st[0]) / 100.0);
        read_r.push_back(static_cast<double>(st[2] - st[1]) / 100.0);
        fetch.push_back(static_cast<double>(st[3] - st[2]) / 100.0);
        body.push_back(static_cast<double>(st[4] - st[3]) / 100.0);
        publish.push_back(static_cast<double>(st[5] - st[4]) / 100.0);
        doorbell.push_back(static_cast<double>(st[6] - st[5]) / 100.0);
        total.push_back(static_cast<double>(st[6] - st[0]) / 100.0);
    }
    if (!ok) {
        device_aql_teardown(rig);
        std::printf("FAIL: device-aql-stage\n");
        return 1;
    }
    std::printf("gpu-dispatch variant=device-aql-stage iterations=%d (p50 us)\n",
                opt.iterations);
    std::printf("  read_write_index=%.3f read_read_index=%.3f fetch_add=%.3f\n",
                percentile(read_w, 0.5), percentile(read_r, 0.5),
                percentile(fetch, 0.5));
    std::printf("  body=%.3f publish=%.3f doorbell=%.3f total_enqueue=%.3f\n",
                percentile(body, 0.5), percentile(publish, 0.5),
                percentile(doorbell, 0.5), percentile(total, 0.5));
    (void)hipFree(stage_dev);
    (void)hipFree(out_dev);
    device_aql_teardown(rig);
    return 0;
}


int run_device_aql_plc(const Options& opt) {
    const uint32_t n = static_cast<uint32_t>(std::max(std::min(opt.n, 64), 1));
    auto partition_res = mcu::GpuMcuCuPartition::create(opt.device, 1u);
    if (!partition_res.ok()) { std::printf("FAIL: partition\n"); return 1; }
    auto partition = partition_res.release();
    auto queue_res = mcu::GpuMcuAqlQueue::create_with_mask(
        opt.device, mcu::AqlQueueOwner::GpuMcu, mcu::kAqlDefaultQueueSize,
        partition.hsa_worker_cu_mask_view());
    if (!queue_res.ok()) { std::printf("FAIL: queue\n"); return 1; }
    auto queue = queue_res.release();
    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
    if (!code_res.ok()) { std::printf("FAIL: code\n"); return 1; }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata& meta = code.metadata();

    std::vector<mcu::GpuMcuCompletionWord> completions;
    completions.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto w = mcu::GpuMcuCompletionWord::create();
        if (!w.ok()) { std::printf("FAIL: completion\n"); return 1; }
        completions.push_back(w.release());
    }
    uint32_t* output_dev = nullptr;
    uint64_t* worker_ts_dev = nullptr;
    uint64_t* publish_ts_dev = nullptr;
    uint64_t* doorbell_ts_dev = nullptr;
    mcu::GpuAqlPacketTemplate* templates_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&output_dev), n * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&worker_ts_dev), n * 2 * sizeof(uint64_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&publish_ts_dev), n * sizeof(uint64_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&doorbell_ts_dev), n * sizeof(uint64_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&templates_dev), n * sizeof(mcu::GpuAqlPacketTemplate)) != hipSuccess) {
        std::printf("FAIL: buffers\n"); return 1;
    }
    const std::size_t slot_bytes = std::max(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
        mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuWorkerArgs)));
    auto region_res = queue.allocate_kernarg(slot_bytes, n);
    if (!region_res.ok()) { std::printf("FAIL: kernarg\n"); return 1; }
    const mcu::GpuMcuKernargRegion region = region_res.release();

    std::vector<mcu::GpuAqlPacketTemplate> host_templates(n);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char* slot = region.slot(i);
        std::memset(slot, 0, region.slot_stride);
        mcu::GpuMcuWorkerArgs args{};
        args.output = reinterpret_cast<uint64_t>(output_dev) + i * sizeof(uint32_t);
        args.completion = reinterpret_cast<uint64_t>(completions[i].device_ptr());
        args.value = i;
        args.element_count = 1u;
        args.generation = i + 1u;
        args.timestamps = reinterpret_cast<uint64_t>(worker_ts_dev) +
                          static_cast<uint64_t>(i) * 2ull * sizeof(uint64_t);
        std::memcpy(slot, &args, sizeof(args));
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, 1u, 1u, 1u, 32u);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = 1u * 32u;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = 32;
        desc.workgroup_size_y = 1;
        desc.workgroup_size_z = 1;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        host_templates[i] = mcu::build_aql_packet_template(desc, mcu::kMcuAqlFastPolicy);
    }
    if (hipMemcpy(templates_dev, host_templates.data(),
                  n * sizeof(mcu::GpuAqlPacketTemplate), hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: templates\n"); return 1;
    }

    std::vector<double> pubdb, kick, worker_dur, hot_gap;
    bool ok = true;
    for (int it = 0; it < opt.warmup + opt.iterations && ok; ++it) {
        for (auto& c : completions) c.reset();
        (void)hipMemset(worker_ts_dev, 0, n * 2 * sizeof(uint64_t));
        (void)hipMemset(publish_ts_dev, 0, n * sizeof(uint64_t));
        (void)hipMemset(doorbell_ts_dev, 0, n * sizeof(uint64_t));
        auto mcu_res = mcu::GpuMcuPersistentMcu::create(opt.device, partition.control_stream());
        if (!mcu_res.ok()) { ok = false; break; }
        auto persistent = mcu_res.release();
        if (!persistent.enable_emit(queue.device_queue_view(), templates_dev, n,
                                    publish_ts_dev, doorbell_ts_dev, 8, true,
                                    opt.barrier != 0).ok()) {
            ok = false; break;
        }
        if (!persistent.start().ok()) { ok = false; break; }
        if (!persistent.request_emit(n).ok()) { ok = false; break; }
        bool all = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
        while (std::chrono::steady_clock::now() < deadline) {
            uint32_t done = 0;
            for (uint32_t i = 0; i < n; ++i) if (completions[i].observe().is_complete()) ++done;
            if (done == n) { all = true; break; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (!all) {
            std::printf("incomplete: published=%llu failures=%llu\n",
                        (unsigned long long)persistent.published(),
                        (unsigned long long)persistent.publish_failures());
            ok = false;
            (void)persistent.shutdown();
            break;
        }
        if (it >= opt.warmup) {
            std::vector<uint64_t> wts(n * 2, 0), pts(n, 0), dts(n, 0);
            (void)hipMemcpyAsync(wts.data(), worker_ts_dev, n * 2 * sizeof(uint64_t),
                                 hipMemcpyDeviceToHost, partition.worker_stream());
            (void)hipMemcpyAsync(pts.data(), publish_ts_dev, n * sizeof(uint64_t),
                                 hipMemcpyDeviceToHost, partition.worker_stream());
            (void)hipMemcpyAsync(dts.data(), doorbell_ts_dev, n * sizeof(uint64_t),
                                 hipMemcpyDeviceToHost, partition.worker_stream());
            (void)hipStreamSynchronize(partition.worker_stream());
            for (uint32_t i = 0; i < n; ++i) {
                pubdb.push_back(static_cast<double>(dts[i] - pts[i]) / 100.0);
                kick.push_back(static_cast<double>(wts[i * 2] - pts[i]) / 100.0);
                worker_dur.push_back(static_cast<double>(wts[i * 2 + 1] - wts[i * 2]) / 100.0);
                if (i > 0) hot_gap.push_back(static_cast<double>(wts[i * 2] - wts[(i - 1) * 2]) / 100.0);
            }
        }
        (void)persistent.request_stop();
        (void)persistent.wait_stopped(5000);
        (void)persistent.shutdown();
    }

    if (ok) {
        std::printf("gpu-dispatch variant=device-aql-plc n=%u barrier=%d iterations=%d\n",
                    n, opt.barrier, opt.iterations);
        std::printf("  publish_to_doorbell_us p50=%.3f p95=%.3f min=%.3f\n",
                    percentile(pubdb, 0.5), percentile(pubdb, 0.95), percentile(pubdb, 0.0));
        std::printf("  publish_to_worker_start_us p50=%.3f p95=%.3f min=%.3f\n",
                    percentile(kick, 0.5), percentile(kick, 0.95), percentile(kick, 0.0));
        std::printf("  worker_duration_us p50=%.3f p95=%.3f min=%.3f\n",
                    percentile(worker_dur, 0.5), percentile(worker_dur, 0.95),
                    percentile(worker_dur, 0.0));
        std::printf("  worker_start_interval_us p50=%.3f p95=%.3f min=%.3f\n",
                    percentile(hot_gap, 0.5), percentile(hot_gap, 0.95),
                    percentile(hot_gap, 0.0));
    }
    (void)hipFree(output_dev);
    (void)hipFree(worker_ts_dev);
    (void)hipFree(publish_ts_dev);
    (void)hipFree(doorbell_ts_dev);
    (void)hipFree(templates_dev);
    (void)code.shutdown();
    (void)queue.shutdown();
    (void)partition.shutdown();
    return ok ? 0 : 1;
}

enum class FsmFeedMode : uint32_t {
    wait = 0u,
    feed = 1u,
    feed_batch = 2u,
};

int run_device_aql_fsm(const Options& opt, bool use_lds, FsmFeedMode mode) {
    const bool feed = mode != FsmFeedMode::wait;
    const uint32_t dispatch_count =
        feed ? static_cast<uint32_t>(std::max(opt.n, 1)) : 3u;
    const uint32_t node_count =
        feed ? dispatch_count + 2u : dispatch_count * 2u + 1u;
    const uint32_t kernarg_ring = feed ? dispatch_count : 8u;
    const uint64_t plans = static_cast<uint64_t>(std::max(opt.iterations, 1));
    uint32_t timing_count = static_cast<uint32_t>(
        std::min<uint64_t>(plans * dispatch_count, mcu::kMcuMaxTiming));
    if (timing_count == 0u) timing_count = 1u;

    auto partition_res = mcu::GpuMcuCuPartition::create(opt.device, 1u);
    if (!partition_res.ok()) { std::printf("FAIL: partition\n"); return 1; }
    auto partition = partition_res.release();
    auto queue_res = mcu::GpuMcuAqlQueue::create_with_mask(
        opt.device, mcu::AqlQueueOwner::GpuMcu, mcu::kAqlDefaultQueueSize,
        partition.hsa_worker_cu_mask_view());
    if (!queue_res.ok()) { std::printf("FAIL: queue\n"); return 1; }
    auto queue = queue_res.release();
    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, image.data, image.bytes, mcu::kGpuMcuFsmWorkerSymbol);
    if (!code_res.ok()) { std::printf("FAIL: code\n"); return 1; }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();

    std::vector<mcu::McuPlanNode> plan;
    std::vector<uint32_t> dispatch_pcs;
    if (!feed) {
        for (uint32_t i = 0; i < dispatch_count; ++i) {
            mcu::McuPlanNode dispatch{};
            dispatch.variant_id = 0u;
            dispatch.completion_slot = static_cast<uint16_t>(i);
            dispatch.value = 100u + i;
            dispatch.element_count = 1u;
            dispatch.input_slot = mcu::kMcuNoInput;
            dispatch.work = static_cast<uint32_t>(std::max(opt.work, 0));
            dispatch.flags = mcu::kMcuNodeDispatch;
            dispatch.next = static_cast<uint16_t>(plan.size() + 1u);
            dispatch_pcs.push_back(static_cast<uint32_t>(plan.size()));
            plan.push_back(dispatch);
            mcu::McuPlanNode wait{};
            wait.completion_slot = static_cast<uint16_t>(i);
            wait.flags = mcu::kMcuNodeWait;
            wait.next = static_cast<uint16_t>(plan.size() + 1u);
            plan.push_back(wait);
        }
    } else {
        for (uint32_t i = 0; i < dispatch_count; ++i) {
            mcu::McuPlanNode dispatch{};
            dispatch.variant_id = 0u;
            dispatch.completion_slot = 0u;
            dispatch.value = 1u;
            dispatch.element_count = 1u;
            dispatch.input_slot = static_cast<uint16_t>(
                i == 0u ? mcu::kMcuNoInput : i - 1u);
            dispatch.work = static_cast<uint32_t>(std::max(opt.work, 0));
            dispatch.flags = mcu::kMcuNodeDispatch;
            dispatch.next = static_cast<uint16_t>(i + 1u);
            dispatch_pcs.push_back(i);
            plan.push_back(dispatch);
        }
        mcu::McuPlanNode wait{};
        wait.completion_slot = 0u;
        wait.flags = mcu::kMcuNodeWait;
        wait.next = static_cast<uint16_t>(dispatch_count + 1u);
        plan.push_back(wait);
    }
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    plan.push_back(end);

    const mcu::AqlMemoryPolicy policy = opt.fence_scope == "system"
                                           ? mcu::kMcuSystemAqlPolicy
                                           : mcu::kMcuAgentAqlPolicy;
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, meta, 64u, 1u, policy)};
    std::vector<mcu::GpuMcuRetainedPacket> retained{variants[0].packet};

    mcu::McuPlanNode* plan_dev = nullptr;
    mcu::McuKernelVariantDesc* variants_dev = nullptr;
    mcu::GpuMcuRetainedPacket* retained_dev = nullptr;
    mcu::GpuMcuDeviceCompletion* completions_dev = nullptr;
    uint32_t* output_dev = nullptr;
    uint64_t* timestamps_dev = nullptr;
    mcu::McuDispatchTiming* timing_dev = nullptr;
    mcu::McuDispatchRecord* debug_records_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&plan_dev),
                  plan.size() * sizeof(mcu::McuPlanNode)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&variants_dev),
                  variants.size() * sizeof(mcu::McuKernelVariantDesc)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&retained_dev),
                  retained.size() * sizeof(mcu::GpuMcuRetainedPacket)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&completions_dev),
                  dispatch_count * sizeof(mcu::GpuMcuDeviceCompletion)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&output_dev),
                  node_count * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&timestamps_dev),
                  node_count * 2u * sizeof(uint64_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&timing_dev),
                  timing_count * sizeof(mcu::McuDispatchTiming)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&debug_records_dev),
                  mcu::kMcuMaxDebugRecords *
                      sizeof(mcu::McuDispatchRecord)) != hipSuccess) {
        std::printf("FAIL: buffers\n");
        return 1;
    }
    (void)hipMemcpy(plan_dev, plan.data(), plan.size() * sizeof(mcu::McuPlanNode),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(variants_dev, variants.data(),
                    variants.size() * sizeof(mcu::McuKernelVariantDesc),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(retained_dev, retained.data(),
                    retained.size() * sizeof(mcu::GpuMcuRetainedPacket),
                    hipMemcpyHostToDevice);
    (void)hipMemset(completions_dev, 0,
                    dispatch_count * sizeof(mcu::GpuMcuDeviceCompletion));
    (void)hipMemset(timing_dev, 0, timing_count * sizeof(mcu::McuDispatchTiming));

    const std::size_t slot_bytes = std::max(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
        mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuFsmWorkerArgs)));
    auto region_res = queue.allocate_kernarg(slot_bytes, kernarg_ring);
    if (!region_res.ok()) { std::printf("FAIL: kernarg\n"); return 1; }
    const mcu::GpuMcuKernargRegion region = region_res.release();

    mcu::GpuMcuFsmConfig config{};
    config.queue = queue.device_queue_view();
    config.plan = plan_dev;
    config.node_count = static_cast<uint32_t>(plan.size());
    config.plan_id = 0x46534d54u;
    config.variants = variants_dev;
    config.variant_count = static_cast<uint32_t>(variants.size());
    config.retained = retained_dev;
    config.retained_count = static_cast<uint32_t>(retained.size());
    config.kernarg_base = reinterpret_cast<uint64_t>(region.base);
    config.kernarg_slot_stride = region.slot_stride;
    config.kernarg_slot_count = kernarg_ring;
    config.completions = completions_dev;
    config.completion_count = feed ? 1u : dispatch_count;
    config.output_base = reinterpret_cast<uint64_t>(output_dev);
    config.timestamps_base = reinterpret_cast<uint64_t>(timestamps_dev);
    config.timing = timing_dev;
    config.timing_count = timing_count;
    config.lds_budget_bytes = use_lds ? 8192u : 0u;
    config.template_source = use_lds ? 1u : 0u;
    config.copy_lanes = static_cast<uint32_t>(std::max(opt.copy_lanes, 1));
    config.barrier = opt.barrier != 0 ? 1u : 0u;
    config.prepared_enabled = 0u;
    config.queue_ahead_depth =
        feed ? static_cast<uint32_t>(std::max(opt.queue_ahead, 0)) : 0u;
    config.doorbell_mode =
        mode == FsmFeedMode::feed_batch
            ? static_cast<uint32_t>(mcu::McuDoorbellMode::coalesce)
            : static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.doorbell_batch =
        mode == FsmFeedMode::feed_batch
            ? static_cast<uint32_t>(std::max(opt.doorbell_batch, 1))
            : 1u;
    config.debug_records = opt.debug_records ? debug_records_dev : nullptr;
    config.debug_record_count =
        opt.debug_records ? mcu::kMcuMaxDebugRecords : 0u;

    auto fsm_res = mcu::GpuMcuFsm::create(opt.device, partition.control_stream());
    if (!fsm_res.ok()) { std::printf("FAIL: fsm create\n"); return 1; }
    auto fsm = fsm_res.release();
    if (!fsm.configure(config).ok()) { std::printf("FAIL: fsm configure\n"); return 1; }
    if (!fsm.start().ok()) { std::printf("FAIL: fsm start\n"); return 1; }
    for (uint64_t i = 0; i < plans; ++i) {
        if (!fsm.request_run().ok()) { std::printf("FAIL: request\n"); return 1; }
    }
    auto waited = fsm.wait_plans(plans, 60000);
    if (!waited.ok()) {
        std::printf("FAIL: %s fault=%u\n", waited.message().c_str(),
                    static_cast<unsigned>(fsm.fault_code()));
        return 1;
    }

    std::vector<mcu::McuDispatchTiming> timing(timing_count);
    (void)hipMemcpyAsync(timing.data(), timing_dev,
                         timing_count * sizeof(mcu::McuDispatchTiming),
                         hipMemcpyDeviceToHost, partition.worker_stream());
    (void)hipStreamSynchronize(partition.worker_stream());

    std::vector<unsigned long long> ts_host(
        static_cast<std::size_t>(node_count) * 2u, 0ull);
    (void)hipMemcpyAsync(ts_host.data(), timestamps_dev,
                         ts_host.size() * sizeof(unsigned long long),
                         hipMemcpyDeviceToHost, partition.worker_stream());
    (void)hipStreamSynchronize(partition.worker_stream());

    std::vector<double> gaps, durations;
    for (std::size_t j = 0; j < dispatch_pcs.size(); ++j) {
        const std::size_t pc = dispatch_pcs[j];
        const unsigned long long start = ts_host[pc * 2u];
        const unsigned long long stop = ts_host[pc * 2u + 1u];
        if (start != 0ull && stop > start) {
            durations.push_back(
                static_cast<double>(stop - start) / 100.0);
        }
        if (j + 1u < dispatch_pcs.size()) {
            const std::size_t next = dispatch_pcs[j + 1u];
            const unsigned long long next_start = ts_host[next * 2u];
            if (stop != 0ull && next_start >= stop) {
                gaps.push_back(
                    static_cast<double>(next_start - stop) / 100.0);
            }
        }
    }

    std::vector<double> kernarg_build, packet_stage, prepare_total, commit, wait;
    for (const mcu::McuDispatchTiming& t : timing) {
        if (t.packet_stage_end == 0u) continue;
        kernarg_build.push_back(
            static_cast<double>(t.kernarg_build_end - t.node_fetch_start) / 100.0);
        packet_stage.push_back(
            static_cast<double>(t.packet_stage_end - t.kernarg_build_end) / 100.0);
        prepare_total.push_back(
            static_cast<double>(t.packet_stage_end - t.node_fetch_start) / 100.0);
        commit.push_back(static_cast<double>(t.dw0_publish - t.commit_start) / 100.0);
        if (t.dependency_ready != 0u &&
            t.dependency_ready >= t.dw0_publish) {
            wait.push_back(
                static_cast<double>(t.dependency_ready - t.dw0_publish) / 100.0);
        }
    }

    const double prepare_p50 = percentile(prepare_total, 0.5);
    const double wait_p50 = percentile(wait, 0.5);
    const char* mode_name = mode == FsmFeedMode::wait
                                ? "wait"
                                : (mode == FsmFeedMode::feed ? "feed"
                                                             : "feed-batch");
    std::printf(
        "gpu-dispatch variant=%s mode=%s plans=%llu dispatch=%u work=%d "
        "barrier=%d scope=%s template=%s depth=%d batch=%d\n",
        use_lds ? "device-aql-fsm-lds" : "device-aql-fsm", mode_name,
        static_cast<unsigned long long>(plans), dispatch_count, opt.work,
        opt.barrier, opt.fence_scope.c_str(), use_lds ? "lds" : "vram",
        opt.queue_ahead, opt.doorbell_batch);
    std::printf("  kernarg_build_us p50=%.3f p95=%.3f\n",
                percentile(kernarg_build, 0.5), percentile(kernarg_build, 0.95));
    std::printf("  packet_stage_us p50=%.3f p95=%.3f\n",
                percentile(packet_stage, 0.5), percentile(packet_stage, 0.95));
    std::printf("  prepare_total_us p50=%.3f p95=%.3f\n", prepare_p50,
                percentile(prepare_total, 0.95));
    std::printf("  commit_us p50=%.3f p95=%.3f\n", percentile(commit, 0.5),
                percentile(commit, 0.95));
    std::printf("  dependency_wait_us p50=%.3f p95=%.3f\n", wait_p50,
                percentile(wait, 0.95));
    std::printf("  prepare_total/dependency_wait p50_ratio=%.3f\n",
                wait_p50 > 0.0 ? prepare_p50 / wait_p50 : 0.0);
    std::printf("  inter_kernel_gap_us p50=%.3f p95=%.3f max=%.3f\n",
                percentile(gaps, 0.5), percentile(gaps, 0.95),
                percentile(gaps, 1.0));
    std::printf("  worker_duration_us p50=%.3f p95=%.3f\n",
                percentile(durations, 0.5), percentile(durations, 0.95));
    std::printf(
        "  doorbells=%llu queue_empty=%llu queue_full=%llu refill=%llu "
        "max_ahead=%llu wraps=%llu\n",
        static_cast<unsigned long long>(fsm.doorbell_count()),
        static_cast<unsigned long long>(fsm.queue_empty_count()),
        static_cast<unsigned long long>(fsm.backpressure_wait_count()),
        static_cast<unsigned long long>(fsm.refill_count()),
        static_cast<unsigned long long>(fsm.max_ahead()),
        static_cast<unsigned long long>(fsm.wrap_count()));

    (void)fsm.request_stop();
    (void)fsm.wait_stopped(5000);
    (void)fsm.shutdown();

    (void)hipFree(plan_dev);
    (void)hipFree(variants_dev);
    (void)hipFree(retained_dev);
    (void)hipFree(completions_dev);
    (void)hipFree(output_dev);
    (void)hipFree(timestamps_dev);
    (void)hipFree(timing_dev);
    (void)hipFree(debug_records_dev);
    (void)code.shutdown();
    (void)queue.shutdown();
    (void)partition.shutdown();
    return 0;
}

int run_host_aql(const Options& opt) {    DeviceAqlRig rig{};
    if (!device_aql_setup(opt, rig)) {
        std::printf("FAIL: host-aql setup\n");
        return 1;
    }
    mcu::GpuAqlPacketTemplate tmpl{};
    (void)hipMemcpy(&tmpl, rig.templates_dev, sizeof(tmpl), hipMemcpyDeviceToHost);

    std::vector<double> submit_us;
    std::vector<double> wall_us;
    for (int it = 0; it < opt.warmup + opt.iterations; ++it) {
        rig.completion.reset();
        const auto t0 = std::chrono::steady_clock::now();
        if (!rig.queue.submit_host(&tmpl, 1).ok()) {
            std::printf("FAIL: host submit\n");
            device_aql_teardown(rig);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        auto snapshot = rig.completion.wait_for(mcu::GpuMcuCompletionState::complete,
                                                10000);
        const auto t2 = std::chrono::steady_clock::now();
        if (!snapshot.ok()) {
            device_aql_teardown(rig);
            std::printf("FAIL: host aql completion\n");
            return 1;
        }
        if (it < opt.warmup) continue;
        submit_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
        wall_us.push_back(
            std::chrono::duration<double, std::micro>(t2 - t1).count());
    }
    std::printf("gpu-dispatch variant=host-aql iterations=%d\n", opt.iterations);
    std::printf("  HOST_AQL_SUBMIT_COST_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(submit_us, 0.5), percentile(submit_us, 0.95),
                percentile(submit_us, 0.0));
    std::printf("  HOST_AQL_SUBMIT_TO_COMPLETION_us p50=%.3f p95=%.3f min=%.3f\n",
                percentile(wall_us, 0.5), percentile(wall_us, 0.95),
                percentile(wall_us, 0.0));
    device_aql_teardown(rig);
    return 0;
}

}  // namespace

int run_gpu_dispatch(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int& out) {
            if (i + 1 < argc) out = std::atoi(argv[++i]);
        };
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        }
        if (a == "--variant" && i + 1 < argc) opt.variant = argv[++i];
        else if (a == "--warmup") next(opt.warmup);
        else if (a == "--iterations") next(opt.iterations);
        else if (a == "--n") next(opt.n);
        else if (a == "--batch") next(opt.batch);
        else if (a == "--work") next(opt.work);
        else if (a == "--barrier") next(opt.barrier);
        else if (a == "--copy-lanes") next(opt.copy_lanes);
        else if (a == "--queue-ahead") next(opt.queue_ahead);
        else if (a == "--doorbell-batch") next(opt.doorbell_batch);
        else if (a == "--debug-records") opt.debug_records = true;
        else if (a == "--fence-scope") opt.fence_scope = argv[++i];
        else if (a == "--device") next(opt.device);
        else if (a == "--no-worker-mask") opt.worker_mask = false;
        else if (a == "--producer" && i + 1 < argc) {
            opt.producer_on_control = std::string(argv[++i]) != "default";
        }
        else {
            usage(stderr);
            return 2;
        }
    }
    if (hipSetDevice(opt.device) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    if (opt.variant == "hip") return run_hip_chain(opt);
    if (opt.variant == "hip-graph") return run_hip_graph(opt);
    if (opt.variant == "device-aql") return run_device_aql(opt);
    if (opt.variant == "device-aql-chain") return run_device_aql_chain(opt);
    if (opt.variant == "queue-mem") return run_queue_mem(opt);
    if (opt.variant == "device-aql-plc") return run_device_aql_plc(opt);
    if (opt.variant == "device-aql-fsm") {
        return run_device_aql_fsm(opt, false, FsmFeedMode::wait);
    }
    if (opt.variant == "device-aql-fsm-lds") {
        return run_device_aql_fsm(opt, true, FsmFeedMode::wait);
    }
    if (opt.variant == "device-aql-fsm-feed") {
        return run_device_aql_fsm(opt, false, FsmFeedMode::feed);
    }
    if (opt.variant == "device-aql-fsm-feed-lds") {
        return run_device_aql_fsm(opt, true, FsmFeedMode::feed);
    }
    if (opt.variant == "device-aql-fsm-feed-batch") {
        return run_device_aql_fsm(opt, false, FsmFeedMode::feed_batch);
    }
    if (opt.variant == "device-aql-stage") return run_device_aql_stage(opt);
    if (opt.variant == "host-aql") return run_host_aql(opt);
    std::fprintf(stderr, "unknown variant: %s\n", opt.variant.c_str());
    return 2;
}
