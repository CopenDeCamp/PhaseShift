#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/stream_bridge.h>

#include <cstdint>
#include <cstdio>
#include <memory>
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
constexpr uint32_t kBadRecipe = 99u;

}  // namespace

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
    if (!wait_supported.ok() || wait_supported.value() == 0) {
        std::printf("SKIP: hipDeviceAttributeCanUseStreamWaitValue is not set\n");
        return 77;
    }

    const auto start_mem_res = ::ps::runtime::stream_signal_alloc(64u);
    const auto done_mem_res = ::ps::runtime::stream_signal_alloc(64u);
    const auto result_mem_res = ::ps::runtime::stream_signal_alloc(64u);
    if (!start_mem_res.ok() || !done_mem_res.ok() || !result_mem_res.ok()) {
        std::printf("FAIL: signal memory allocation\n");
        return 1;
    }
    const ::ps::runtime::StreamSignal start_mem = start_mem_res.value();
    const ::ps::runtime::StreamSignal done_mem = done_mem_res.value();
    const ::ps::runtime::StreamSignal result_mem = result_mem_res.value();

    float* a = nullptr;
    float* b = nullptr;
    const size_t bytes = static_cast<size_t>(kFeatures) * sizeof(float);
    if (hipMalloc(&a, bytes) != hipSuccess || hipMalloc(&b, bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: host queue\n");
        return 1;
    }
    auto host_queue = queue_res.release();
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        host_queue, PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
        kernel::kElementwiseScaleF32Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: scale code object\n");
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    (void)code->shutdown();
    (void)host_queue.shutdown();

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
                                 mcu::AqlHiddenArgsPolicy::IfFits, 1u),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};

    std::vector<mcu::McuPlanNode> nodes(4);
    nodes[0].variant_id = 0u;
    nodes[0].kernarg_recipe = static_cast<uint16_t>(kBadRecipe);
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
    inv.scalar_a = 2.0f;
    inv.rows = 1u;
    if (!rig.set_elementwise_invocations(&inv, 1u)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4641554cu;
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

    const uint32_t epoch = 1u;
    if (!::ps::runtime::stream_write_value32(stream, start_mem, epoch).ok()) {
        std::printf("FAIL: start signal write\n");
        return 1;
    }
    if (!::ps::runtime::stream_wait_value32(stream, done_mem, epoch).ok()) {
        std::printf("FAIL: done signal wait enqueue\n");
        return 1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: stream sync: %s\n",
                    hipGetErrorString(hipGetLastError()));
        return 1;
    }

    const uint32_t done = ::ps::runtime::stream_signal_load(done_mem);
    const uint32_t rc = ::ps::runtime::stream_signal_load(result_mem);
    const auto fault = static_cast<uint32_t>(rig.fsm.fault_code());
    std::printf(
        "fault bridge: epoch=%u done=%u result_code=%u fault_code=%u "
        "dispatches=%llu invalid_node=%u\n",
        epoch, done, rc, fault,
        static_cast<unsigned long long>(rig.fsm.dispatches()),
        static_cast<uint32_t>(mcu::McuFaultCode::invalid_node));
    check(done == epoch, "done signal released despite the mcu fault");
    check(rc == fault, "result code carries the mcu fault code");
    check(rc == static_cast<uint32_t>(mcu::McuFaultCode::invalid_node),
          "fault is invalid_node for an unknown kernarg recipe");
    check(rig.fsm.dispatches() == 0u, "no dispatch committed after the fault");

    check(rig.close(), "rig teardown");
    check(::ps::runtime::stream_signal_free(start_mem).ok(), "start signal freed");
    check(::ps::runtime::stream_signal_free(done_mem).ok(), "done signal freed");
    check(::ps::runtime::stream_signal_free(result_mem).ok(), "result signal freed");
    (void)hipStreamDestroy(stream);
    (void)hipFree(a);
    (void)hipFree(b);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_decode_fault_bridge: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_decode_fault_bridge: PASS\n");
    return 0;
}
