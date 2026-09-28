#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/sampling.h>
#include <phaseshift/models/qwen35/runtime/sampling_params.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace rt = ps::runtime;
namespace q35 = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kOutputs = 3u;
constexpr uint32_t kVocab = 4096u;
constexpr uint32_t kThreads = 256u;

uint32_t host_argmax(const std::vector<float>& logits, uint32_t row) {
    const float* row_logits = logits.data() + static_cast<size_t>(row) * kVocab;
    float best_value = -1.0f / 0.0f;
    uint32_t best_token = 0xffffffffu;
    for (uint32_t v = 0u; v < kVocab; ++v) {
        const float cv = row_logits[v];
        if (cv > best_value || (cv == best_value && v < best_token)) {
            best_value = cv;
            best_token = v;
        }
    }
    return best_token;
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

    Rig rig;
    if (!rig.open(0, 8u, 2u, 4u, 0u, false, mcu::kAqlDefaultQueueSize, 0u,
                  1024u)) {
        std::printf("test_gpu_mcu_sampling_argmax: rig setup failed\n");
        return 1;
    }

    mcu::GpuAqlKernelMetadata argmax_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_SAMPLING_HSACO,
                             kernel::kArgmaxF32Symbol, argmax_meta)) {
        return 1;
    }

    std::vector<float> logits(kOutputs * kVocab, 0.0f);
    std::vector<uint32_t> expected(kOutputs, 0u);
    const uint32_t winners[kOutputs] = {1234u, 7u, 4095u};
    for (uint32_t r = 0u; r < kOutputs; ++r) {
        float* row = logits.data() + static_cast<size_t>(r) * kVocab;
        for (uint32_t v = 0u; v < kVocab; ++v) {
            row[v] = -1.0f - static_cast<float>(v) * 1e-4f;
        }
        row[winners[r]] = 5.0f + static_cast<float>(r);
        if (r == 1u) {
            row[winners[r]] = 5.0f;
            row[8u] = 5.0f;
        }
        expected[r] = host_argmax(logits, r);
    }
    check(expected[1] == 7u, "host tie-break picks the smallest token");

    std::vector<rt::DeviceSamplingParams> sampling(kOutputs);
    for (uint32_t r = 0u; r < kOutputs; ++r) {
        sampling[r].mode =
            static_cast<uint8_t>(q35::SamplingMode::Greedy);
    }

    float* logits_dev = nullptr;
    rt::DeviceSamplingParams* sampling_dev = nullptr;
    int32_t* sampled_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&logits_dev),
                  logits.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&sampling_dev),
                  sampling.size() * sizeof(rt::DeviceSamplingParams)) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&sampled_dev),
                  kOutputs * sizeof(int32_t)) != hipSuccess) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    (void)hipMemcpy(logits_dev, logits.data(), logits.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(sampling_dev, sampling.data(),
                    sampling.size() * sizeof(rt::DeviceSamplingParams),
                    hipMemcpyHostToDevice);
    (void)hipMemset(sampled_dev, 0, kOutputs * sizeof(int32_t));

    mcu::McuPlanNode dispatch{};
    dispatch.variant_id = 0u;
    dispatch.kernarg_recipe = mcu::kMcuKernargRecipeArgmaxF32;
    dispatch.completion_slot = 0u;
    dispatch.flags = mcu::kMcuNodeDispatch;
    dispatch.next = 1u;
    mcu::McuPlanNode marker{};
    marker.variant_id = 1u;
    marker.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    marker.completion_slot = 0u;
    marker.element_count = 1u;
    marker.flags = mcu::kMcuNodeDispatch;
    marker.next = 2u;
    mcu::McuPlanNode wait{};
    wait.completion_slot = 0u;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = 3u;
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    std::vector<mcu::McuPlanNode> nodes{dispatch, marker, wait, end};

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, argmax_meta, mcu::kMcuKernargRecipeArgmaxF32,
                                 kThreads, kOutputs, mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u)};
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuArgmaxF32Invocation invocation{};
    invocation.logits = reinterpret_cast<uint64_t>(logits_dev);
    invocation.sampling = reinterpret_cast<uint64_t>(sampling_dev);
    invocation.sampled_tokens = reinterpret_cast<uint64_t>(sampled_dev);
    invocation.outputs = kOutputs;
    invocation.vocab_size = kVocab;
    invocation.logits_row_stride = kVocab;
    invocation.constraint_masks = 0u;
    invocation.constraint_mask_words = 0u;
    if (!rig.set_argmax_f32_invocations(&invocation, 1u)) return 1;

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x53414d50u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    if (!rig.start(config)) return 1;

    check(rig.fsm.request_run().ok(), "plan requested");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");

    std::vector<int32_t> got(kOutputs, -1);
    rig.device_to_host(got.data(), sampled_dev, kOutputs * sizeof(int32_t));
    for (uint32_t r = 0u; r < kOutputs; ++r) {
        if (got[r] != static_cast<int32_t>(expected[r])) {
            std::printf("FAIL: row %u sampled=%d expected=%u\n", r, got[r],
                        expected[r]);
            ++g_failed;
        }
    }

    check(rig.close(), "rig teardown");
    (void)hipFree(logits_dev);
    (void)hipFree(sampling_dev);
    (void)hipFree(sampled_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_sampling_argmax: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_sampling_argmax: PASS\n");
    return 0;
}
