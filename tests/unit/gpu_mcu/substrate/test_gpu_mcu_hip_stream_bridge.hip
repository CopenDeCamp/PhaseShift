#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/stream_bridge.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace {

namespace mcu = ::ps::runtime::gpu_mcu;
namespace kernel = ::ps::kernel;
using gpu_mcu_test::Rig;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kFeatures = 256u;
constexpr uint32_t kRows = 1u;
constexpr float kScale = 2.0f;

}  // namespace

__global__ void bridge_prefix_kernel(float* a, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = static_cast<float>(i) * 0.5f + 1.0f;
}

__global__ void bridge_suffix_kernel(const float* b, float* c, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = b[i] + 1.0f;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    const auto wait_supported = ::ps::runtime::stream_wait_value_supported(0);
    if (!wait_supported.ok()) {
        std::printf("FAIL: capability query: %s\n",
                    wait_supported.status().message().c_str());
        return 1;
    }
    std::printf("can_use_stream_wait_value=%d\n", wait_supported.value());
    if (wait_supported.value() == 0) {
        std::printf("SKIP: hipDeviceAttributeCanUseStreamWaitValue is not set\n");
        return 77;
    }

    const auto start_mem_res = ::ps::runtime::stream_signal_alloc(64u);
    const auto done_mem_res = ::ps::runtime::stream_signal_alloc(64u);
    const auto result_mem_res = ::ps::runtime::stream_signal_alloc(64u);
    if (!start_mem_res.ok() || !done_mem_res.ok() || !result_mem_res.ok()) {
        std::printf("FAIL: signal memory allocation: start=%s done=%s result=%s\n",
                    start_mem_res.status().message().c_str(),
                    done_mem_res.status().message().c_str(),
                    result_mem_res.status().message().c_str());
        return 1;
    }
    std::printf("signal_memory=%s bytes=%zu\n",
                ::ps::runtime::stream_signal_kind_name(start_mem_res.value().kind),
                start_mem_res.value().bytes);
    const ::ps::runtime::StreamSignal start_mem = start_mem_res.value();
    const ::ps::runtime::StreamSignal done_mem = done_mem_res.value();
    const ::ps::runtime::StreamSignal result_mem = result_mem_res.value();

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    const size_t bytes = static_cast<size_t>(kFeatures) * kRows * sizeof(float);
    if (hipMalloc(&a, bytes) != hipSuccess || hipMalloc(&b, bytes) != hipSuccess ||
        hipMalloc(&c, bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: host queue: %s\n",
                    queue_res.status().message().c_str());
        return 1;
    }
    auto host_queue = queue_res.release();
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        host_queue, PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
        kernel::kElementwiseScaleF32Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: scale code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    (void)code->shutdown();
    (void)host_queue.shutdown();
    std::printf("scale args=%zu kernarg=%u grid_x=%u wg=%u\n",
                sizeof(kernel::ElementwiseArgs), meta.kernarg_segment_size,
                kernel::elementwise_grid_x(kFeatures), kernel::kElementwiseBlock);

    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords, meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                             kernel::kElementwiseScaleF32Symbol, mcu_meta))
        return 1;

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, mcu_meta, mcu::kMcuKernargRecipeElementwise,
                                 kernel::kElementwiseBlock,
                                 kernel::elementwise_grid_x(kFeatures),
                                 mcu::kMcuAgentAqlPolicy,
                                 mcu::AqlHiddenArgsPolicy::IfFits, kRows),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};

    std::vector<mcu::McuPlanNode> nodes(4);
    nodes[0].variant_id = 0u;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0u;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1u;
    nodes[1].variant_id = 1u;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[1].completion_slot = 0;
    nodes[1].input_slot = mcu::kMcuNoInput;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2u;
    nodes[2].completion_slot = 0;
    nodes[2].flags = mcu::kMcuNodeWait;
    nodes[2].next = 3u;
    nodes[3].flags = mcu::kMcuNodeEnd;
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuElementwiseInvocation inv{};
    inv.in0 = reinterpret_cast<uint64_t>(a);
    inv.in0_row_stride = kFeatures;
    inv.out0 = reinterpret_cast<uint64_t>(b);
    inv.out0_row_stride = kFeatures;
    inv.features = kFeatures;
    inv.scalar_a = kScale;
    inv.rows = kRows;
    if (!rig.set_elementwise_invocations(&inv, 1u)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x42524944u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 3u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    config.start_signal = start_mem.device_ptr;
    config.done_signal = done_mem.device_ptr;
    config.result_code = result_mem.device_ptr;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }

    int iters = 100;
    if (const char* e = std::getenv("PHASESHIFT_BRIDGE_ITERS")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v > 0) iters = static_cast<int>(v);
    }

    std::vector<float> host_a(kFeatures), host_c(kFeatures);
    uint32_t stale = 0u;
    uint32_t mismatched = 0u;
    uint32_t result_errors = 0u;
    for (int it = 0; it < iters; ++it) {
        const uint32_t epoch = static_cast<uint32_t>(it) + 1u;
        bridge_prefix_kernel<<<1, kFeatures, 0, stream>>>(a, kFeatures);
        if (!::ps::runtime::stream_write_value32(stream, start_mem, epoch).ok()) {
            std::printf("FAIL: start signal write at iter %d\n", it);
            return 1;
        }
        if (!::ps::runtime::stream_wait_value32(stream, done_mem, epoch).ok()) {
            std::printf("FAIL: done signal wait enqueue at iter %d\n", it);
            return 1;
        }
        bridge_suffix_kernel<<<1, kFeatures, 0, stream>>>(b, c, kFeatures);
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: stream sync at iter %d: %s\n", it,
                        hipGetErrorString(hipGetLastError()));
            return 1;
        }
        if (::ps::runtime::stream_signal_load(done_mem) != epoch) {
            ++stale;
            std::printf("FAIL: stale done epoch at iter %d (done=%u expect=%u)\n",
                        it, ::ps::runtime::stream_signal_load(done_mem), epoch);
            if (it > 0) break;
        }
        const uint32_t rc = ::ps::runtime::stream_signal_load(result_mem);
        if (rc != 0u) {
            ++result_errors;
            std::printf("FAIL: result_code=%u at iter %d\n", rc, it);
            break;
        }
        (void)hipMemcpyAsync(host_a.data(), a, bytes, hipMemcpyDeviceToHost,
                             stream);
        (void)hipMemcpyAsync(host_c.data(), c, bytes, hipMemcpyDeviceToHost,
                             stream);
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::printf("FAIL: readback sync at iter %d\n", it);
            return 1;
        }
        bool ok = true;
        for (uint32_t i = 0; i < kFeatures; ++i) {
            const float expect = host_a[i] * kScale + 1.0f;
            if (host_c[i] != expect) {
                ok = false;
                if (mismatched == 0u)
                    std::printf("FAIL: c[%u]=%f expect=%f (a=%f)\n", i,
                                static_cast<double>(host_c[i]),
                                static_cast<double>(expect),
                                static_cast<double>(host_a[i]));
            }
        }
        if (!ok) ++mismatched;
        if (mismatched != 0u) break;
    }

    std::printf(
        "bridge: iters=%d signal_memory=%s stale=%u mismatched=%u "
        "result_errors=%u mcu_dispatches=%llu fault=%u\n",
        iters, ::ps::runtime::stream_signal_kind_name(start_mem.kind), stale,
        mismatched, result_errors,
        static_cast<unsigned long long>(rig.fsm.dispatches()),
        static_cast<unsigned>(rig.fsm.fault_code()));
    check(stale == 0u, "no stale done epoch");
    check(mismatched == 0u, "hip -> aql -> hip sandwich is exact");
    check(result_errors == 0u, "result code stays zero");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == static_cast<uint64_t>(iters) * 2u,
          "one scale dispatch plus one marker per iteration");

    check(rig.close(), "rig teardown");
    check(::ps::runtime::stream_signal_free(start_mem).ok(), "start signal freed");
    check(::ps::runtime::stream_signal_free(done_mem).ok(), "done signal freed");
    check(::ps::runtime::stream_signal_free(result_mem).ok(), "result signal freed");
    (void)hipStreamDestroy(stream);
    (void)hipFree(a);
    (void)hipFree(b);
    (void)hipFree(c);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_hip_stream_bridge: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_hip_stream_bridge: PASS\n");
    return 0;
}
