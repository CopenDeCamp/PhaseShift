#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/completion.h>
#include <phaseshift/runtime/gpu_mcu/worker_image.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

namespace mcu = ps::runtime::gpu_mcu;

void usage(FILE* f) {
    std::fprintf(f,
                 "usage: phaseshift-bench gpu-ext-dispatch [options]\n"
                 "  Gate 0 probe for HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH\n"
                 "  --device N\n"
                 "  --timeout-ms N\n  --header-mode N (0=vendor+barrier+acq+rel, 1=vendor only)\n  --dep-zero 0|1\n");
}

int probe_child(int device, int timeout_ms, int header_mode, int dep_zero,
                int amd_format) {
    if (hipSetDevice(device) != hipSuccess) return 4;

    auto queue_res = mcu::GpuMcuAqlQueue::create(device, mcu::AqlQueueOwner::GpuMcu,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) return 5;
    auto queue = queue_res.release();

    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto code_res = mcu::GpuMcuAqlCodeObject::load_memory(
        queue, image.data, image.bytes, mcu::kGpuMcuProbeWorkerSymbol);
    if (!code_res.ok()) return 5;
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata& meta = code.metadata();

    auto completion_res = mcu::GpuMcuCompletionWord::create();
    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), 1);
    if (!completion_res.ok() || !region_res.ok()) return 5;
    auto completion = completion_res.release();
    const mcu::GpuMcuKernargRegion region = region_res.release();

    mcu::GpuMcuWorkerArgs args{};
    args.output = 0;
    args.completion = reinterpret_cast<uint64_t>(completion.device_ptr());
    args.value = 1u;
    args.generation = 1u;
    std::memcpy(region.slot(0), &args, sizeof(args));
    mcu::build_aql_launch_metadata(region.slot(0), meta.kernarg_segment_size, 1u, 1u,
                                   1u, 32u);

    hsa_signal_t dep_signal{};
    hsa_signal_t done_signal{};
    if (hsa_signal_create(0, 0, nullptr, &dep_signal) != HSA_STATUS_SUCCESS ||
        hsa_signal_create(0, 0, nullptr, &done_signal) != HSA_STATUS_SUCCESS) {
        return 5;
    }

    if (amd_format == 2) {
        hsa_amd_barrier_value_packet_t bp{};
        bp.header.header = static_cast<uint16_t>(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
        bp.header.AmdFormat = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
        bp.signal = hsa_signal_t{0};
        bp.value = 0;
        bp.mask = ~static_cast<hsa_signal_value_t>(0);
        bp.cond = HSA_SIGNAL_CONDITION_EQ;
        bp.completion_signal = done_signal;
        auto* bq = static_cast<hsa_queue_t*>(queue.device_queue_view().queue_ptr);
        const uint64_t bidx = hsa_queue_add_write_index_scacq_screl(bq, 1);
        auto* bring = static_cast<unsigned char*>(bq->base_address);
        unsigned char* bslot =
            bring + static_cast<uint32_t>(bidx % bq->size) * mcu::kAqlPacketBytes;
        const uint32_t invalid = mcu::kAqlInvalidDw0;
        std::memcpy(bslot, &invalid, sizeof(invalid));
        std::memcpy(bslot + 4, reinterpret_cast<const unsigned char*>(&bp) + 4, 60);
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(bslot, &bp, sizeof(uint32_t));
        std::atomic_thread_fence(std::memory_order_release);
        hsa_signal_store_screlease(bq->doorbell_signal, bidx);
        const hsa_signal_value_t seen = hsa_signal_wait_scacquire(
            done_signal, HSA_SIGNAL_CONDITION_EQ, 1,
            static_cast<uint64_t>(timeout_ms) * 1000000ull, HSA_WAIT_STATE_BLOCKED);
        return seen == 1 ? 0 : 3;
    }

    hsa_amd_ext_kernel_dispatch_packet_t packet{};
    packet.header = static_cast<uint16_t>(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    if (header_mode == 0) {
        packet.header |= static_cast<uint16_t>(1u << 8) |
                         static_cast<uint16_t>(2u << 9) |
                         static_cast<uint16_t>(2u << 11);
    }
    packet.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
    packet.setup = 1u;
    packet.workgroup_size_x = 32;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.cluster_count_x = 1;
    packet.cluster_count_y = 1;
    packet.cluster_count_z = 1;
    packet.cluster_size_x = 1;
    packet.cluster_size_y = 1;
    packet.cluster_size_z = 1;
    packet.private_segment_size = meta.private_segment_size;
    packet.group_segment_size = meta.static_group_segment_size;
    packet.kernel_object = meta.kernel_object;
    packet.kernarg_address = region.slot(0);
    packet.dep_signal = dep_zero != 0 ? hsa_signal_t{0} : dep_signal;
    packet.completion_signal = done_signal;

    auto* hsa_queue = static_cast<hsa_queue_t*>(queue.device_queue_view().queue_ptr);
    const uint64_t base = hsa_queue_add_write_index_scacq_screl(hsa_queue, 1);
    auto* ring = static_cast<unsigned char*>(hsa_queue->base_address);
    unsigned char* slot =
        ring + static_cast<uint32_t>(base % hsa_queue->size) * mcu::kAqlPacketBytes;
    const uint32_t invalid = mcu::kAqlInvalidDw0;
    std::memcpy(slot, &invalid, sizeof(invalid));
    std::memcpy(slot + 4, reinterpret_cast<const unsigned char*>(&packet) + 4, 60);
    std::atomic_thread_fence(std::memory_order_release);
    std::memcpy(slot, &packet, sizeof(uint32_t));
    std::atomic_thread_fence(std::memory_order_release);
    hsa_signal_store_screlease(hsa_queue->doorbell_signal, base);

    const hsa_signal_value_t observed = hsa_signal_wait_scacquire(
        done_signal, HSA_SIGNAL_CONDITION_EQ, 1,
        static_cast<uint64_t>(timeout_ms) * 1000000ull, HSA_WAIT_STATE_BLOCKED);
    return observed == 1 ? 0 : 3;
}

}  // namespace

int run_gpu_ext_dispatch(int argc, char** argv) {
    int device = 0;
    int timeout_ms = 2000;
    int header_mode = 0;
    int dep_zero = 0;
    int amd_format = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        }
        if (a == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (a == "--timeout-ms" && i + 1 < argc) timeout_ms = std::atoi(argv[++i]);
        else if (a == "--header-mode" && i + 1 < argc) header_mode = std::atoi(argv[++i]);
        else if (a == "--dep-zero" && i + 1 < argc) dep_zero = std::atoi(argv[++i]);
        else if (a == "--amd-format" && i + 1 < argc) amd_format = std::atoi(argv[++i]);
        else {
            usage(stderr);
            return 2;
        }
    }

    std::printf("EXT_KERNEL_DISPATCH Gate 0\n");
    std::printf("  header: <hsa/hsa_ext_amd.h>\n");
    std::printf("  HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH=%d\n",
                static_cast<int>(HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH));
    std::printf("  hsa_amd_ext_kernel_dispatch_packet_t size=%zu\n",
                sizeof(hsa_amd_ext_kernel_dispatch_packet_t));
    std::printf("  dep_signal offset=%zu completion_signal offset=%zu\n",
                offsetof(hsa_amd_ext_kernel_dispatch_packet_t, dep_signal),
                offsetof(hsa_amd_ext_kernel_dispatch_packet_t, completion_signal));
    std::printf("  HEADER: AVAILABLE\n");
    std::fflush(stdout);

    const pid_t pid = fork();
    if (pid < 0) {
        std::printf("  RUNTIME: UNKNOWN (fork failed)\n");
        return 0;
    }
    if (pid == 0) {
        const int code = probe_child(device, timeout_ms, header_mode, dep_zero, amd_format);
        std::_Exit(code);
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        const int code = WEXITSTATUS(status);
        if (code == 0) {
            std::printf("  RUNTIME: ACCEPTED (EXT dispatch executed)\n");
        } else if (code == 3) {
            std::printf("  RUNTIME: UNAVAILABLE_AT_RUNTIME (no completion within %d ms)\n",
                        timeout_ms);
        } else {
            std::printf("  RUNTIME: UNAVAILABLE_AT_RUNTIME (probe setup failure %d)\n",
                        code);
        }
    } else if (WIFSIGNALED(status)) {
        std::printf("  RUNTIME: UNAVAILABLE_AT_RUNTIME (runtime aborted on packet: signal %d)\n",
                    WTERMSIG(status));
    } else {
        std::printf("  RUNTIME: UNKNOWN\n");
    }
    return 0;
}
