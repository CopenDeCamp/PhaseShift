#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSlots = 4u;
constexpr uint32_t kRows = 16u;
constexpr uint32_t kStride = 16u;
constexpr uint32_t kElems = 8u;

struct RestoreCase {
    uint32_t slot;
    uint32_t accepted;
};

const RestoreCase kCases[kSlots] = {{0u, 0u}, {1u, 1u}, {2u, 4u}, {3u, 8u}};

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

    Rig rig;
    if (!rig.open(0, 16u, 2u, 4u, 0u, false, mcu::kAqlDefaultQueueSize, 0u,
                  1024u)) {
        std::printf("test_gpu_mcu_gdn_verify_restore: rig setup failed\n");
        return 1;
    }

    mcu::GpuAqlKernelMetadata restore_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_GDN_SPEC_RESTORE_HSACO,
                             kernel::kGdnSpecRestoreSymbol, restore_meta)) {
        return 1;
    }

    std::vector<uint16_t> conv_history(kRows * kStride, 0u);
    std::vector<float> rec_history(kRows * kStride, 0.0f);
    for (uint32_t r = 0u; r < kRows; ++r) {
        for (uint32_t i = 0u; i < kStride; ++i) {
            const size_t index = static_cast<size_t>(r) * kStride + i;
            conv_history[index] = static_cast<uint16_t>(1000u + r * 16u + i);
            rec_history[index] = static_cast<float>(r) * 0.5f + 0.25f * i;
        }
    }
    const std::vector<uint16_t> conv_pool_init(kSlots * kStride, 0xABCDu);
    const std::vector<float> rec_pool_init(kSlots * kStride, -1234.5f);

    uint16_t* conv_history_dev = nullptr;
    float* rec_history_dev = nullptr;
    uint16_t* conv_pool_dev = nullptr;
    float* rec_pool_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&conv_history_dev),
                  conv_history.size() * sizeof(uint16_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&rec_history_dev),
                  rec_history.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&conv_pool_dev),
                  conv_pool_init.size() * sizeof(uint16_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&rec_pool_dev),
                  rec_pool_init.size() * sizeof(float)) != hipSuccess) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    (void)hipMemcpy(conv_history_dev, conv_history.data(),
                    conv_history.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(rec_history_dev, rec_history.data(),
                    rec_history.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(conv_pool_dev, conv_pool_init.data(),
                    conv_pool_init.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(rec_pool_dev, rec_pool_init.data(),
                    rec_pool_init.size() * sizeof(float),
                    hipMemcpyHostToDevice);

    std::vector<mcu::McuPlanNode> nodes;
    std::vector<mcu::McuGdnSpecRestoreInvocation> invocations;
    for (uint32_t i = 0u; i < kSlots; ++i) {
        mcu::McuPlanNode dispatch{};
        dispatch.variant_id = 0u;
        dispatch.kernarg_recipe = mcu::kMcuKernargRecipeGdnSpecRestore;
        dispatch.completion_slot = 0u;
        dispatch.invocation_index = i;
        dispatch.flags = mcu::kMcuNodeDispatch;
        dispatch.next = static_cast<uint16_t>(i + 1u);
        nodes.push_back(dispatch);

        mcu::McuGdnSpecRestoreInvocation invocation{};
        invocation.conv_history = reinterpret_cast<uint64_t>(conv_history_dev);
        invocation.conv_pool = reinterpret_cast<uint64_t>(conv_pool_dev);
        invocation.recurrent_history =
            reinterpret_cast<uint64_t>(rec_history_dev);
        invocation.recurrent_pool = reinterpret_cast<uint64_t>(rec_pool_dev);
        invocation.conv_slot_stride = kStride;
        invocation.recurrent_slot_stride = kStride;
        invocation.conv_history_stride = kStride;
        invocation.recurrent_history_stride = kStride;
        invocation.sequence_slot = kCases[i].slot;
        invocation.history_row = kCases[i].accepted;
        invocation.conv_elems = kElems;
        invocation.recurrent_elems = kElems;
        invocations.push_back(invocation);
    }
    mcu::McuPlanNode marker{};
    marker.variant_id = 1u;
    marker.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    marker.completion_slot = 0u;
    marker.element_count = 1u;
    marker.flags = mcu::kMcuNodeDispatch;
    marker.next = static_cast<uint16_t>(nodes.size() + 1u);
    nodes.push_back(marker);
    mcu::McuPlanNode wait{};
    wait.completion_slot = 0u;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = static_cast<uint16_t>(nodes.size() + 1u);
    nodes.push_back(wait);
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    nodes.push_back(end);

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, restore_meta,
                                 mcu::kMcuKernargRecipeGdnSpecRestore,
                                 kernel::kGdnSpecRestoreThreads, 1u,
                                 mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u)};
    if (!rig.upload(nodes, variants)) return 1;
    if (!rig.set_gdn_spec_restore_invocations(invocations.data(), kSlots)) {
        return 1;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x47445253u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    if (!rig.start(config)) return 1;

    check(rig.fsm.request_run().ok(), "plan requested");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");

    std::vector<uint16_t> conv_pool(kSlots * kStride, 0u);
    std::vector<float> rec_pool(kSlots * kStride, 0.0f);
    rig.device_to_host(conv_pool.data(), conv_pool_dev,
                       conv_pool.size() * sizeof(uint16_t));
    rig.device_to_host(rec_pool.data(), rec_pool_dev,
                       rec_pool.size() * sizeof(float));

    for (uint32_t c = 0u; c < kSlots; ++c) {
        const uint32_t slot = kCases[c].slot;
        const uint32_t row = kCases[c].accepted;
        for (uint32_t i = 0u; i < kElems; ++i) {
            const size_t dst = static_cast<size_t>(slot) * kStride + i;
            const size_t src = static_cast<size_t>(row) * kStride + i;
            if (conv_pool[dst] != conv_history[src]) {
                std::printf("FAIL: conv slot=%u row=%u i=%u got=%u expected=%u\n",
                            slot, row, i, conv_pool[dst], conv_history[src]);
                ++g_failed;
            }
            if (rec_pool[dst] != rec_history[src]) {
                std::printf("FAIL: rec slot=%u row=%u i=%u\n", slot, row, i);
                ++g_failed;
            }
        }
        for (uint32_t i = kElems; i < kStride; ++i) {
            const size_t dst = static_cast<size_t>(slot) * kStride + i;
            if (conv_pool[dst] != conv_pool_init[dst] ||
                rec_pool[dst] != rec_pool_init[dst]) {
                std::printf("FAIL: slot=%u tail i=%u overwritten\n", slot, i);
                ++g_failed;
            }
        }
    }

    check(rig.close(), "rig teardown");
    (void)hipFree(conv_history_dev);
    (void)hipFree(rec_history_dev);
    (void)hipFree(conv_pool_dev);
    (void)hipFree(rec_pool_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_gdn_verify_restore: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_gdn_verify_restore: PASS\n");
    return 0;
}
