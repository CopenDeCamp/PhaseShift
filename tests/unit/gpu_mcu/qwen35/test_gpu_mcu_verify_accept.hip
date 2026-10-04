#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/verify_accept.h>

#include <cstdint>
#include <cstdio>
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

constexpr uint32_t kStride = 32u;
constexpr uint32_t kRequests = 5u;

bool run_context_accept_case(uint32_t accepted_target, const char* name) {
    namespace rt = ::ps::runtime;
    rt::DeviceBatchContext* ctx = nullptr;
    rt::DeviceRequestDescriptor* requests = nullptr;
    int32_t* tokens = nullptr;
    uint32_t* output_rows = nullptr;
    int32_t* sampled = nullptr;
    uint32_t* counts = nullptr;
    if (hipMallocManaged(reinterpret_cast<void**>(&ctx), sizeof(rt::DeviceBatchContext)) != hipSuccess ||
        hipMallocManaged(reinterpret_cast<void**>(&requests), 3u * sizeof(rt::DeviceRequestDescriptor)) != hipSuccess ||
        hipMallocManaged(reinterpret_cast<void**>(&tokens), 32u * sizeof(int32_t)) != hipSuccess ||
        hipMallocManaged(reinterpret_cast<void**>(&output_rows), 8u * sizeof(uint32_t)) != hipSuccess ||
        hipMallocManaged(reinterpret_cast<void**>(&sampled), 8u * sizeof(int32_t)) != hipSuccess ||
        hipMallocManaged(reinterpret_cast<void**>(&counts), 3u * sizeof(uint32_t)) != hipSuccess) {
        check(false, name);
        return false;
    }
    for (uint32_t i = 0u; i < 3u; ++i) counts[i] = 0xFFFFFFFFu;
    requests[0].execution_class = rt::ExecutionClass::DECODE;
    requests[0].row_begin = 0u;
    requests[0].row_count = 1u;
    requests[1].execution_class = rt::ExecutionClass::SPEC_VERIFY;
    requests[1].row_begin = 1u;
    requests[1].row_count = 8u;
    requests[1].output_index = 0u;
    requests[2].execution_class = rt::ExecutionClass::PREFILL;
    requests[2].row_begin = 9u;
    requests[2].row_count = 16u;
    for (uint32_t i = 0u; i < 8u; ++i) {
        tokens[1u + i] = static_cast<int32_t>(100u + i);
        output_rows[i] = i;
        sampled[i] = i < accepted_target ? tokens[1u + i]
                                         : static_cast<int32_t>(999u + i);
    }
    ctx->num_requests = 3u;
    ctx->requests = requests;
    ctx->token_ids = tokens;
    ctx->output_rows = output_rows;

    ps::kernel::VerifyAcceptBatchArgs args{};
    args.context = ctx;
    args.sampled_tokens = sampled;
    args.committed_counts = counts;
    args.max_candidates = 32u;
    hipError_t err = ps::kernel::launch_verify_accept_batch(
        ctx, sampled, counts, 32u, nullptr);
    (void)args;
    check(err == hipSuccess, "verify accept batch launch");
    if (hipDeviceSynchronize() != hipSuccess) {
        check(false, "verify accept batch sync");
    }
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s: accepted count", name);
    check(counts[1] == accepted_target, msg);
    std::snprintf(msg, sizeof(msg), "%s: non verify stays zero", name);
    check(counts[0] == 0u && counts[2] == 0u, msg);
    (void)hipFree(ctx);
    (void)hipFree(requests);
    (void)hipFree(tokens);
    (void)hipFree(output_rows);
    (void)hipFree(sampled);
    (void)hipFree(counts);
    return true;
}

uint32_t host_accept(const std::vector<int32_t>& candidates,
                     const std::vector<int32_t>& sampled, uint32_t request,
                     uint32_t count) {
    uint32_t accepted = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        const size_t index = static_cast<size_t>(request) * kStride + i;
        if (sampled[index] != candidates[index]) break;
        ++accepted;
    }
    return accepted;
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

    for (uint32_t target : {0u, 1u, 4u, 8u}) {
        run_context_accept_case(target, "context accept");
    }

    Rig rig;
    if (!rig.open(0, 8u, 2u, 4u, 0u)) {
        std::printf("test_gpu_mcu_verify_accept: rig setup failed\n");
        return 1;
    }

    mcu::GpuAqlKernelMetadata verify_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_VERIFY_ACCEPT_HSACO,
                             kernel::kVerifyAcceptPrefixSymbol, verify_meta)) {
        return 1;
    }

    std::vector<int32_t> candidates(kRequests * kStride, 0);
    std::vector<int32_t> sampled(kRequests * kStride, 0);
    std::vector<uint32_t> counts(kRequests, 0u);
    const uint32_t expected_accepted[kRequests] = {0u, 1u, 4u, 8u, 32u};
    const uint32_t case_counts[kRequests] = {8u, 8u, 8u, 8u, 32u};

    for (uint32_t r = 0u; r < kRequests; ++r) {
        counts[r] = case_counts[r];
        for (uint32_t i = 0u; i < case_counts[r]; ++i) {
            const size_t index = static_cast<size_t>(r) * kStride + i;
            candidates[index] = static_cast<int32_t>(10u + i + r * 100u);
            sampled[index] = candidates[index];
        }
        for (uint32_t i = 0u; i < expected_accepted[r]; ++i) {
            const size_t index = static_cast<size_t>(r) * kStride + i;
            sampled[index] = candidates[index];
        }
        if (expected_accepted[r] < case_counts[r]) {
            const size_t index =
                static_cast<size_t>(r) * kStride + expected_accepted[r];
            sampled[index] = candidates[index] + 1;
        }
    }

    for (uint32_t r = 0u; r < kRequests; ++r) {
        const uint32_t reference =
            host_accept(candidates, sampled, r, counts[r]);
        check(reference == expected_accepted[r], "host reference accepted count");
    }

    int32_t* candidates_dev = nullptr;
    int32_t* sampled_dev = nullptr;
    uint32_t* counts_dev = nullptr;
    uint32_t* committed_dev = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&candidates_dev),
                  candidates.size() * sizeof(int32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&sampled_dev),
                  sampled.size() * sizeof(int32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&counts_dev),
                  counts.size() * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&committed_dev),
                  kRequests * sizeof(uint32_t)) != hipSuccess) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    (void)hipMemcpy(candidates_dev, candidates.data(),
                    candidates.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(sampled_dev, sampled.data(),
                    sampled.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(counts_dev, counts.data(), counts.size() * sizeof(uint32_t),
                    hipMemcpyHostToDevice);
    (void)hipMemset(committed_dev, 0, kRequests * sizeof(uint32_t));

    mcu::McuPlanNode dispatch{};
    dispatch.variant_id = 0u;
    dispatch.kernarg_recipe = mcu::kMcuKernargRecipeVerifyAcceptPrefix;
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
        mcu::make_kernel_variant(0u, verify_meta,
                                 mcu::kMcuKernargRecipeVerifyAcceptPrefix,
                                 kernel::kVerifyAcceptPrefixThreads, 1u,
                                 mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u)};
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuVerifyAcceptPrefixInvocation invocation{};
    invocation.candidates = reinterpret_cast<uint64_t>(candidates_dev);
    invocation.sampled_tokens = reinterpret_cast<uint64_t>(sampled_dev);
    invocation.candidate_counts = reinterpret_cast<uint64_t>(counts_dev);
    invocation.committed_counts = reinterpret_cast<uint64_t>(committed_dev);
    invocation.request_count = kRequests;
    invocation.candidate_stride = kStride;
    invocation.max_candidates = kernel::kVerifyAcceptPrefixMaxCandidates;
    if (!rig.set_verify_accept_invocations(&invocation, 1u)) return 1;

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x56414343u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    if (!rig.start(config)) return 1;

    check(rig.fsm.request_run().ok(), "plan requested");
    check(rig.fsm.wait_plans(1u, 10000).ok(), "plan completed");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");

    std::vector<uint32_t> got(kRequests, 0u);
    rig.device_to_host(got.data(), committed_dev, kRequests * sizeof(uint32_t));
    for (uint32_t r = 0u; r < kRequests; ++r) {
        if (got[r] != expected_accepted[r]) {
            std::printf("FAIL: request %u accepted got=%u expected=%u\n", r,
                        got[r], expected_accepted[r]);
            ++g_failed;
        }
    }

    check(rig.close(), "rig teardown");
    (void)hipFree(candidates_dev);
    (void)hipFree(sampled_dev);
    (void)hipFree(counts_dev);
    (void)hipFree(committed_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_verify_accept: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_verify_accept: PASS\n");
    return 0;
}
