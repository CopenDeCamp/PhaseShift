#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

namespace mcu = ::ps::runtime::gpu_mcu;
namespace kernel = ::ps::kernel;
namespace q35 = ::ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kFeatures = 256u;
constexpr float kScale = 3.0f;

q35::McuCompiledPlan make_plan(const float* in, float* out) {
    q35::McuCompiledPlan plan;
    mcu::McuPlanNode dispatch{};
    dispatch.variant_id = 0u;
    dispatch.kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
    dispatch.completion_slot = 0u;
    dispatch.invocation_index = 0u;
    dispatch.flags = mcu::kMcuNodeDispatch;
    dispatch.next = 1u;
    plan.nodes.push_back(dispatch);

    mcu::McuPlanNode marker{};
    marker.variant_id = 1u;
    marker.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    marker.completion_slot = 0u;
    marker.input_slot = mcu::kMcuNoInput;
    marker.flags = mcu::kMcuNodeDispatch;
    marker.next = 2u;
    plan.nodes.push_back(marker);

    mcu::McuPlanNode wait{};
    wait.completion_slot = 0u;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = 3u;
    plan.nodes.push_back(wait);

    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    plan.nodes.push_back(end);

    q35::McuCompiledVariant scale{};
    scale.kind = q35::McuCompiledVariantKind::ElementwiseScaleF32;
    scale.grid = kernel::elementwise_grid_x(kFeatures);
    scale.grid_y = 1u;
    scale.grid_z = 1u;
    scale.workgroup = kernel::kElementwiseBlock;
    plan.variants.push_back(scale);

    q35::McuCompiledVariant completion{};
    completion.kind = q35::McuCompiledVariantKind::CompletionMarker;
    completion.grid = 1u;
    completion.grid_y = 1u;
    completion.grid_z = 1u;
    completion.workgroup = 64u;
    plan.variants.push_back(completion);

    mcu::McuElementwiseInvocation invocation{};
    invocation.in0 = reinterpret_cast<uint64_t>(in);
    invocation.in0_row_stride = kFeatures;
    invocation.out0 = reinterpret_cast<uint64_t>(out);
    invocation.out0_row_stride = kFeatures;
    invocation.features = kFeatures;
    invocation.scalar_a = kScale;
    invocation.rows = 1u;
    plan.elementwise.push_back(invocation);
    return plan;
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

    const auto wait_supported = ::ps::runtime::stream_wait_value_supported(0);
    if (!wait_supported.ok() || wait_supported.value() == 0) {
        std::printf("SKIP: hipDeviceAttributeCanUseStreamWaitValue is not set\n");
        return 77;
    }

    const size_t bytes = static_cast<size_t>(kFeatures) * sizeof(float);
    float* in = nullptr;
    float* out = nullptr;
    if (hipMalloc(&in, bytes) != hipSuccess || hipMalloc(&out, bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    std::vector<float> host_in(kFeatures);
    std::vector<float> host_out(kFeatures, 0.0f);
    for (uint32_t i = 0; i < kFeatures; ++i)
        host_in[i] = static_cast<float>(i) * 0.25f - 3.0f;
    (void)hipMemcpy(in, host_in.data(), bytes, hipMemcpyHostToDevice);

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }

    const int cycles = 100;
    const int epochs_per_cycle = 3;
    uint32_t mismatched = 0u;
    uint32_t faults = 0u;
    uint32_t result_errors = 0u;
    uint64_t total_dispatches = 0u;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        auto runtime_res =
            q35::McuDecodeRuntime::create(q35::McuDecodeRuntimeOptions{});
        if (!runtime_res.ok()) {
            std::printf("FAIL: runtime create: %s\n",
                        runtime_res.status().message().c_str());
            return 1;
        }
        q35::McuDecodeRuntime runtime = runtime_res.release();

        const q35::McuCompiledPlan plan = make_plan(in, out);
        const ps::Status prepared = runtime.prepare_plan(plan);
        if (!prepared.ok()) {
            std::printf("FAIL: prepare_plan: %s\n", prepared.message().c_str());
            return 1;
        }
        if (cycle == 0) {
            std::printf(
                "runtime: nodes=%u variants=%u embedded_code_objects=%u "
                "start_memory=%s\n",
                runtime.plan_node_count(), runtime.plan_variant_count(),
                runtime.embedded_kernel_count(),
                ::ps::runtime::stream_signal_kind_name(runtime.start_signal().kind));
        }

        for (int e = 0; e < epochs_per_cycle; ++e) {
            const uint32_t epoch =
                static_cast<uint32_t>(cycle * epochs_per_cycle + e) + 1u;
            if (!runtime.enqueue_run(stream, epoch).ok()) {
                std::printf("FAIL: enqueue_run\n");
                return 1;
            }
            if (!runtime.enqueue_wait(stream, epoch).ok()) {
                std::printf("FAIL: enqueue_wait\n");
                return 1;
            }
            if (hipStreamSynchronize(stream) != hipSuccess) {
                std::printf("FAIL: stream sync: %s\n",
                            hipGetErrorString(hipGetLastError()));
                return 1;
            }
            if (runtime.result_code() != 0u) {
                ++result_errors;
                break;
            }
            if (hipMemcpyAsync(host_out.data(), out, bytes,
                               hipMemcpyDeviceToHost, stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess) {
                std::printf("FAIL: readback\n");
                return 1;
            }
            bool ok = true;
            for (uint32_t i = 0; i < kFeatures; ++i) {
                if (host_out[i] != host_in[i] * kScale) {
                    ok = false;
                    break;
                }
            }
            if (!ok) ++mismatched;
        }
        faults += runtime.fsm().fault_code() == mcu::McuFaultCode::none ? 0u : 1u;
        total_dispatches += runtime.fsm().dispatches();
        (void)runtime.shutdown();
    }

    std::printf(
        "runtime lifecycle: cycles=%d epochs_per_cycle=%d faults=%u "
        "mismatched=%u result_errors=%u dispatches=%llu\n",
        cycles, epochs_per_cycle, faults, mismatched, result_errors,
        static_cast<unsigned long long>(total_dispatches));
    check(faults == 0u, "no fault across lifecycle cycles");
    check(mismatched == 0u, "no mismatch across lifecycle cycles");
    check(result_errors == 0u, "no result code error across lifecycle cycles");
    check(total_dispatches ==
              static_cast<uint64_t>(cycles) *
                  static_cast<uint64_t>(epochs_per_cycle) * 2u,
          "dispatch count = (scale + marker) per epoch");

    (void)hipStreamDestroy(stream);
    (void)hipFree(in);
    (void)hipFree(out);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_runtime_lifecycle: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_runtime_lifecycle: PASS\n");
    return 0;
}
