#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/gdn/reset.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

namespace {

using gpu_mcu_test::Rig;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSequences = 2u;
constexpr uint32_t kConvCount = 64u;
constexpr uint32_t kRecurrentCount = 256u;
constexpr uint32_t kThreads = 64u;
constexpr uint32_t kBlocks = 4u;

constexpr uint16_t kConvPoison = 0xABCDu;
constexpr uint32_t kRecurrentPoison = 0xDEADBEEFu;

__global__ void poison_kernel(uint16_t* conv, uint32_t* recurrent,
                              uint32_t conv_count, uint32_t recurrent_count) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < conv_count;
         i += stride) {
        conv[i] = kConvPoison;
    }
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < recurrent_count; i += stride) {
        recurrent[i] = kRecurrentPoison;
    }
}

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

    Rig rig;
    if (!rig.open(0, 8u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords, 512u)) {
        std::printf("test_gpu_mcu_gdn_reset: rig setup failed\n");
        return 1;
    }

    mcu::GpuAqlKernelMetadata reset_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_GDN_RESET_HSACO,
                             mcu::kGdnResetSymbol, reset_meta)) {
        return 1;
    }

    // A real primitive does not write a completion itself, so the reset is
    // bracketed by probe dispatches exactly like the other real primitive
    // tests: the trailing probe signals the completion the final wait observes.
    std::vector<mcu::McuPlanNode> nodes(6u);
    nodes[0].variant_id = 0u;
    nodes[0].kernarg_recipe =
        static_cast<uint16_t>(mcu::kMcuKernargRecipeProbe);
    nodes[0].completion_slot = 0u;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1u;
    nodes[1].completion_slot = 0u;
    nodes[1].flags = mcu::kMcuNodeWait;
    nodes[1].next = 2u;
    nodes[2].variant_id = 1u;
    nodes[2].kernarg_recipe =
        static_cast<uint16_t>(mcu::kMcuKernargRecipeGdnReset);
    nodes[2].completion_slot = 0u;
    nodes[2].invocation_index = 0u;
    nodes[2].flags = mcu::kMcuNodeDispatch;
    nodes[2].next = 3u;
    nodes[3].variant_id = 0u;
    nodes[3].kernarg_recipe =
        static_cast<uint16_t>(mcu::kMcuKernargRecipeProbe);
    nodes[3].completion_slot = 0u;
    nodes[3].input_slot = mcu::kMcuNoInput;
    nodes[3].flags = mcu::kMcuNodeDispatch;
    nodes[3].next = 4u;
    nodes[4].completion_slot = 0u;
    nodes[4].flags = mcu::kMcuNodeWait;
    nodes[4].next = 5u;
    nodes[5].flags = mcu::kMcuNodeEnd;
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, kThreads, 1u),
        mcu::make_kernel_variant(1u, reset_meta, mcu::kMcuKernargRecipeGdnReset,
                                 kThreads, kBlocks, mcu::kMcuAgentAqlPolicy,
                                 mcu::AqlHiddenArgsPolicy::IfFits, 1u)};
    if (!rig.upload(nodes, variants)) return 1;

    uint16_t* conv_dev = nullptr;
    uint32_t* recurrent_dev = nullptr;
    rt::DeviceRequestDescriptor* requests_dev = nullptr;
    rt::DeviceBatchContext* context_dev = nullptr;
    const std::size_t conv_bytes = kSequences * kConvCount * sizeof(uint16_t);
    const std::size_t recurrent_bytes =
        kSequences * kRecurrentCount * sizeof(uint32_t);
    check(hipMalloc(reinterpret_cast<void**>(&conv_dev), conv_bytes) ==
              hipSuccess,
          "conv alloc");
    check(hipMalloc(reinterpret_cast<void**>(&recurrent_dev),
                    recurrent_bytes) == hipSuccess,
          "recurrent alloc");
    check(hipMalloc(reinterpret_cast<void**>(&requests_dev),
                    kSequences * sizeof(rt::DeviceRequestDescriptor)) ==
              hipSuccess,
          "requests alloc");
    check(hipMalloc(reinterpret_cast<void**>(&context_dev),
                    sizeof(rt::DeviceBatchContext)) == hipSuccess,
          "context alloc");

    hipStream_t work = nullptr;
    check(hipStreamCreate(&work) == hipSuccess, "work stream create");
    poison_kernel<<<8u, 128u, 0, work>>>(
        conv_dev, recurrent_dev, kSequences * kConvCount,
        kSequences * kRecurrentCount);
    check(hipStreamSynchronize(work) == hipSuccess, "poison sync");

    // Request 0 continues a sequence that already owns slot 1 and request 1 opens
    // a new sequence in slot 0, so only slot 0 may be cleared.
    rt::DeviceRequestDescriptor requests[kSequences] = {};
    requests[0].request_handle.slot = 1u;
    requests[0].row_begin = 0u;
    requests[0].row_count = 1u;
    requests[0].prefix_length = 5u;
    requests[1].request_handle.slot = 0u;
    requests[1].row_begin = 1u;
    requests[1].row_count = 1u;
    requests[1].prefix_length = 0u;
    check(hipMemcpy(requests_dev, requests, sizeof(requests),
                    hipMemcpyHostToDevice) == hipSuccess,
          "requests upload");

    rt::DeviceBatchContext context{};
    context.actual_rows = kSequences;
    context.num_requests = kSequences;
    context.requests = requests_dev;
    check(hipMemcpy(context_dev, &context, sizeof(context),
                    hipMemcpyHostToDevice) == hipSuccess,
          "context upload");

    mcu::GpuMcuGdnResetInvocation invocation{};
    invocation.conv_base = conv_dev;
    invocation.recurrent_base = recurrent_dev;
    invocation.context = context_dev;
    invocation.conv_slot_stride = kConvCount;
    invocation.recurrent_slot_stride = kRecurrentCount;
    invocation.conv_slot_count = kConvCount;
    invocation.recurrent_slot_count = kRecurrentCount;
    check(rig.set_gdn_reset_invocations(&invocation, 1u),
          "reset invocation uploaded");

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x9d0u;
    config.template_source = 1u;
    config.lds_budget_bytes = 8192u;
    config.copy_lanes = 1u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 0u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("test_gpu_mcu_gdn_reset: start failed\n");
        return 1;
    }

    check(rig.fsm.request_run().ok(), "host kick issued");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.supervisor() == mcu::McuSupervisorState::idle,
          "supervisor returns to idle");
    check(rig.fsm.plans_started() == 1u, "one plan");
    check(rig.fsm.dispatches() == 3u, "probe, reset and probe dispatched");

    // The persistent wave has to stop before a device readback, otherwise the
    // copy waits for a wave that never ends.
    check(rig.fsm.request_stop().ok(), "stop requested");
    check(rig.fsm.wait_stopped(5000).ok(), "fsm stopped");
    std::vector<uint16_t> conv(kSequences * kConvCount, 0u);
    std::vector<uint32_t> recurrent(kSequences * kRecurrentCount, 0u);
    check(hipMemcpyAsync(conv.data(), conv_dev, conv_bytes,
                         hipMemcpyDeviceToHost, work) == hipSuccess,
          "conv readback queued");
    check(hipMemcpyAsync(recurrent.data(), recurrent_dev, recurrent_bytes,
                         hipMemcpyDeviceToHost, work) == hipSuccess,
          "recurrent readback queued");
    check(hipStreamSynchronize(work) == hipSuccess, "readback sync");

    bool conv_cleared = true;
    bool conv_other_intact = true;
    for (uint32_t i = 0; i < kConvCount; ++i) {
        if (conv[i] != 0u) conv_cleared = false;
        if (conv[kConvCount + i] != kConvPoison) conv_other_intact = false;
    }
    check(conv_cleared, "the new sequence conv state is zeroed");
    check(conv_other_intact, "the running sequence conv state is untouched");

    bool recurrent_cleared = true;
    bool recurrent_other_intact = true;
    for (uint32_t i = 0; i < kRecurrentCount; ++i) {
        if (recurrent[i] != 0u) recurrent_cleared = false;
        if (recurrent[kRecurrentCount + i] != kRecurrentPoison) {
            recurrent_other_intact = false;
        }
    }
    check(recurrent_cleared, "the new sequence recurrent state is zeroed");
    check(recurrent_other_intact,
          "the running sequence recurrent state is untouched");

    check(hipStreamDestroy(work) == hipSuccess, "work stream destroy");
    check(hipFree(conv_dev) == hipSuccess, "conv free");
    check(hipFree(recurrent_dev) == hipSuccess, "recurrent free");
    check(hipFree(requests_dev) == hipSuccess, "requests free");
    check(hipFree(context_dev) == hipSuccess, "context free");
    check(rig.close(), "rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_gdn_reset: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_gdn_reset: PASS\n");
    return 0;
}
