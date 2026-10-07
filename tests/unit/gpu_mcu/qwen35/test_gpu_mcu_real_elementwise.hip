#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/elementwise.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kRows = 1u;
constexpr float kScale = 0.0883883f;
constexpr uint32_t kGuardBytes = 64u;

enum class EwKind : uint32_t {
    ResidualAdd = 0,
    Swiglu = 1,
    Mul = 2,
    Scale = 3,
    SiluBf16ToF32 = 4,
    SiluF32ToBf16 = 5,
    SigmoidBf16ToF32 = 6,
};

struct EwCase {
    EwKind kind;
    const char* name;
    const char* symbol;
    uint32_t features;
    bool out_f32;
    bool binary;
    bool in_f32;
};

const EwCase kCases[] = {
    {EwKind::ResidualAdd, "residual_add_bf16",
     kernel::kElementwiseResidualAddBf16Symbol, 2560u, false, true, false},
    {EwKind::Swiglu, "swiglu_bf16", kernel::kElementwiseSwigluBf16Symbol,
     9216u, false, true, false},
    {EwKind::Mul, "mul_f32_f32_bf16", kernel::kElementwiseMulF32F32Bf16Symbol,
     4096u, false, true, true},
    {EwKind::Scale, "scale_f32", kernel::kElementwiseScaleF32Symbol, 2048u,
     true, false, true},
    {EwKind::SiluBf16ToF32, "silu_bf16_f32",
     kernel::kElementwiseSiluBf16ToF32Symbol, 4096u, true, false, false},
    {EwKind::SiluF32ToBf16, "silu_f32_bf16",
     kernel::kElementwiseSiluF32ToBf16Symbol, 8192u, false, false, true},
    {EwKind::SigmoidBf16ToF32, "sigmoid_bf16_f32",
     kernel::kElementwiseSigmoidBf16ToF32Symbol, 4096u, true, false, false},
};

constexpr uint32_t kCaseCount = static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

hipError_t host_launch(const EwCase& c, void* in0, void* in1, void* out,
                       hipStream_t stream) {
    switch (c.kind) {
        case EwKind::ResidualAdd:
            return kernel::launch_residual_add_bf16(
                static_cast<const ps::bf16_t*>(in0), c.features,
                static_cast<const ps::bf16_t*>(in1), c.features,
                static_cast<ps::bf16_t*>(out), c.features, kRows, c.features,
                stream);
        case EwKind::Swiglu:
            return kernel::launch_swiglu_bf16(
                static_cast<const ps::bf16_t*>(in0), c.features,
                static_cast<const ps::bf16_t*>(in1), c.features,
                static_cast<ps::bf16_t*>(out), c.features, kRows, c.features,
                stream);
        case EwKind::Mul:
            return kernel::launch_mul_f32_f32_bf16(
                static_cast<const float*>(in0), c.features,
                static_cast<const float*>(in1), c.features,
                static_cast<ps::bf16_t*>(out), c.features, kRows, c.features,
                stream);
        case EwKind::Scale:
            return kernel::launch_scale_f32(
                static_cast<const float*>(in0), c.features,
                static_cast<float*>(out), c.features, kRows, c.features, kScale,
                stream);
        case EwKind::SiluBf16ToF32:
            return kernel::launch_silu_bf16_f32(
                static_cast<const ps::bf16_t*>(in0), c.features,
                static_cast<float*>(out), c.features, kRows, c.features, stream);
        case EwKind::SiluF32ToBf16:
            return kernel::launch_silu_f32_bf16(
                static_cast<const float*>(in0), c.features,
                static_cast<ps::bf16_t*>(out), c.features, kRows, c.features,
                stream);
        case EwKind::SigmoidBf16ToF32:
            return kernel::launch_sigmoid_bf16_f32(
                static_cast<const ps::bf16_t*>(in0), c.features,
                static_cast<float*>(out), c.features, kRows, c.features,
                stream);
    }
    return hipErrorInvalidValue;
}

void fill_inputs(const EwCase& c, void* in0, void* in1) {
    const size_t elems = c.features;
    if (c.in_f32) {
        auto* a = static_cast<float*>(in0);
        for (size_t i = 0; i < elems; ++i)
            a[i] = static_cast<float>(static_cast<int>((i * 37u + 11u) % 401u) - 200) / 64.0f;
        if (c.binary) {
            auto* b = static_cast<float*>(in1);
            for (size_t i = 0; i < elems; ++i)
                b[i] = static_cast<float>(static_cast<int>((i * 91u + 5u) % 397u) - 198) / 64.0f;
        }
    } else {
        auto* a = static_cast<uint16_t*>(in0);
        for (size_t i = 0; i < elems; ++i)
            a[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
        if (c.binary) {
            auto* b = static_cast<uint16_t*>(in1);
            for (size_t i = 0; i < elems; ++i)
                b[i] = static_cast<uint16_t>(0x3A00u + ((i * 40503u) % 1900u));
        }
    }
}

size_t out_bytes(const EwCase& c) {
    return static_cast<size_t>(c.features) * (c.out_f32 ? 4u : 2u);
}

void fill_args(const EwCase& c, void* in0, void* in1, void* out,
               kernel::ElementwiseArgs& a) {
    a = kernel::ElementwiseArgs{};
    a.in0 = in0;
    a.in0_row_stride = c.features;
    a.in1 = in1;
    a.in1_row_stride = c.features;
    a.out0 = out;
    a.out0_row_stride = c.features;
    a.features = c.features;
    a.scalar_a = c.kind == EwKind::Scale ? kScale : 0.0f;
    a.rows = kRows;
}

void fill_invocation(const EwCase& c, void* in0, void* in1, void* out,
                     mcu::McuElementwiseInvocation& v) {
    v = mcu::McuElementwiseInvocation{};
    v.in0 = reinterpret_cast<uint64_t>(in0);
    v.in0_row_stride = c.features;
    v.in1 = reinterpret_cast<uint64_t>(in1);
    v.in1_row_stride = c.features;
    v.out0 = reinterpret_cast<uint64_t>(out);
    v.out0_row_stride = c.features;
    v.features = c.features;
    v.scalar_a = c.kind == EwKind::Scale ? kScale : 0.0f;
    v.rows = kRows;
}

bool wait_drained(const mcu::GpuMcuAqlQueue& queue, uint64_t target,
                  uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (queue.host_consumed_index() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
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

    const size_t max_bf16 = 9216u * 2u;
    const size_t max_f32 = 8192u * 4u;
    void* bf16_in0 = nullptr;
    void* bf16_in1 = nullptr;
    void* f32_in0 = nullptr;
    void* f32_in1 = nullptr;
    std::vector<void*> out_ref(kCaseCount, nullptr);
    std::vector<void*> out_cand(kCaseCount, nullptr);
    if (hipMalloc(&bf16_in0, max_bf16) != hipSuccess ||
        hipMalloc(&bf16_in1, max_bf16) != hipSuccess ||
        hipMalloc(&f32_in0, max_f32) != hipSuccess ||
        hipMalloc(&f32_in1, max_f32) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const size_t bytes = out_bytes(kCases[i]) + kGuardBytes;
        if (hipMalloc(&out_ref[i], bytes) != hipSuccess ||
            hipMalloc(&out_cand[i], bytes) != hipSuccess) {
            std::printf("FAIL: output buffer alloc\n");
            return 1;
        }
    }

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EwCase& c = kCases[i];
        void* in0 = c.in_f32 ? f32_in0 : bf16_in0;
        void* in1 = c.in_f32 ? f32_in1 : bf16_in1;
        fill_inputs(c, in0, in1);
    }

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    std::vector<std::unique_ptr<mcu::GpuMcuAqlCodeObject>> codes;
    std::vector<mcu::GpuAqlKernelMetadata> metas(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        auto res = mcu::GpuMcuAqlCodeObject::load(queue, PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                                                  kCases[i].symbol);
        if (!res.ok()) {
            std::printf("FAIL: code object %s: %s\n", kCases[i].name,
                        res.status().message().c_str());
            return 1;
        }
        codes.push_back(std::unique_ptr<mcu::GpuMcuAqlCodeObject>(
            new mcu::GpuMcuAqlCodeObject(res.release())));
        metas[i] = codes.back()->metadata();
        std::printf("%s: kernarg=%u group=%u hidden_fits=%d\n", kCases[i].name,
                    metas[i].kernarg_segment_size, metas[i].static_group_segment_size,
                    static_cast<int>(mcu::aql_hidden_args_fits(
                        sizeof(kernel::ElementwiseArgs),
                        metas[i].kernarg_segment_size)));
        check(metas[i].kernel_object != 0, "kernel_object resolved");
        check(metas[i].kernarg_segment_size >= sizeof(kernel::ElementwiseArgs),
              "kernarg segment covers ElementwiseArgs");
        check(mcu::aql_hidden_args_fits(sizeof(kernel::ElementwiseArgs),
                                        metas[i].kernarg_segment_size),
              "elementwise segment includes hidden args");
    }

    uint32_t max_stride = 0u;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const uint32_t s = mcu::aql_kernarg_slot_stride(metas[i].kernarg_segment_size);
        if (s > max_stride) max_stride = s;
    }
    auto region_res = queue.allocate_kernarg(max_stride, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    std::vector<std::vector<uint32_t>> refs(kCaseCount);
    std::vector<std::vector<uint32_t>> guards(kCaseCount);

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EwCase& c = kCases[i];
        void* in0 = c.in_f32 ? f32_in0 : bf16_in0;
        void* in1 = c.in_f32 ? f32_in1 : bf16_in1;
        const size_t bytes = out_bytes(c);

        (void)hipMemset(out_ref[i], 0xAB, bytes + kGuardBytes);
        (void)hipMemset(out_cand[i], 0xAB, bytes + kGuardBytes);
        check(host_launch(c, in0, in1, out_ref[i], stream) == hipSuccess,
              "host launcher");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        unsigned char* slot = region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        kernel::ElementwiseArgs args{};
        fill_args(c, in0, in1, out_cand[i], args);
        std::memcpy(slot, &args, sizeof(args));

        const uint32_t grid_x = kernel::elementwise_grid_x(c.features);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = metas[i].kernel_object;
        desc.global_work_items_x = grid_x * kernel::kElementwiseBlock;
        desc.global_work_items_y = kRows;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(kernel::kElementwiseBlock);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = metas[i].private_segment_size;
        desc.group_segment_size = metas[i].static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::ElementwiseArgs),
                                   metas[i].kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, metas[i].kernarg_segment_size, grid_x,
                                       kRows, 1u, kernel::kElementwiseBlock);

        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        const bool drained = wait_drained(queue, queue.host_queued_index(), 15000);
        check(drained, "aql queue drained");

        const size_t words = bytes / 4u;
        refs[i].assign(words, 0u);
        guards[i].assign(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> cand(words, 0u);
        std::vector<uint32_t> cand_guard(kGuardBytes / 4u, 0u);
        (void)hipMemcpy(refs[i].data(), out_ref[i], bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand.data(), out_cand[i], bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(guards[i].data(),
                        static_cast<unsigned char*>(out_ref[i]) + bytes, kGuardBytes,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand_guard.data(),
                        static_cast<unsigned char*>(out_cand[i]) + bytes, kGuardBytes,
                        hipMemcpyDeviceToHost);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), cand.data(), bytes, &first, &diffs),
              "host produced aql is bit exact with the host launch");
        if (diffs)
            std::printf("  %s aql diff count=%zu first=%zu\n", c.name, diffs, first);
        check(cand_guard == guards[i], "aql output sentinel intact");
    }

    uint32_t needed_segment = 0u;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (metas[i].kernarg_segment_size > needed_segment)
            needed_segment = metas[i].kernarg_segment_size;
    }

    for (auto& code : codes) (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");

    Rig rig;
    if (!rig.open(0, kCaseCount + 4u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  needed_segment)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    std::printf("rig slot_stride=%u needed_segment=%u\n", rig.region.slot_stride,
                needed_segment);
    std::vector<mcu::GpuAqlKernelMetadata> mcu_metas(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO, kCases[i].symbol,
                                 mcu_metas[i]))
            return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_metas[i],
            mcu::kMcuKernargRecipeElementwise, kernel::kElementwiseBlock,
            kernel::elementwise_grid_x(kCases[i].features), mcu::kMcuAgentAqlPolicy));
    }
    variants.push_back(mcu::make_probe_variant(static_cast<uint16_t>(kCaseCount),
                                               rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
        nodes[i].completion_slot = 0;
        nodes[i].invocation_index = i;
        nodes[i].flags = mcu::kMcuNodeDispatch;
        nodes[i].next = static_cast<uint16_t>(i + 1u);
    }
    nodes[kCaseCount].variant_id = static_cast<uint16_t>(kCaseCount);
    nodes[kCaseCount].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[kCaseCount].completion_slot = 0;
    nodes[kCaseCount].input_slot = mcu::kMcuNoInput;
    nodes[kCaseCount].flags = mcu::kMcuNodeDispatch;
    nodes[kCaseCount].next = static_cast<uint16_t>(kCaseCount + 1u);
    nodes[kCaseCount + 1u].completion_slot = 0;
    nodes[kCaseCount + 1u].flags = mcu::kMcuNodeWait;
    nodes[kCaseCount + 1u].next = static_cast<uint16_t>(kCaseCount + 2u);
    nodes[kCaseCount + 2u].flags = mcu::kMcuNodeEnd;

    if (!rig.upload(nodes, variants)) return 1;

    std::vector<mcu::McuElementwiseInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EwCase& c = kCases[i];
        void* in0 = c.in_f32 ? f32_in0 : bf16_in0;
        void* in1 = c.in_f32 ? f32_in1 : bf16_in1;
        const size_t bytes = out_bytes(c);
        (void)hipMemset(out_cand[i], 0xAB, bytes + kGuardBytes);
        fill_invocation(c, in0, in1, out_cand[i], invocations[i]);
    }
    if (!rig.set_elementwise_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x45574c31u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 0u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }
    check(rig.fsm.request_run().ok(), "mcu plan requested");
    check(rig.fsm.wait_plans(1u, 30000).ok(), "mcu plan completed");
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == static_cast<uint64_t>(kCaseCount) + 1u,
          "dispatch count");

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const EwCase& c = kCases[i];
        const size_t bytes = out_bytes(c);
        const size_t words = bytes / 4u;
        std::vector<uint32_t> mcu_out(words, 0u);
        std::vector<uint32_t> mcu_guard(kGuardBytes / 4u, 0u);
        rig.device_to_host(mcu_out.data(), out_cand[i], bytes);
        rig.device_to_host(mcu_guard.data(),
                           static_cast<unsigned char*>(out_cand[i]) + bytes,
                           kGuardBytes);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), mcu_out.data(), bytes, &first, &diffs),
              "mcu elementwise matches the host launch");
        if (diffs)
            std::printf("  %s mcu diff count=%zu first=%zu\n", c.name, diffs, first);
        check(mcu_guard == guards[i], "mcu output sentinel intact");
        std::printf("  %s F=%u grid_x=%u ok=%d\n", c.name, c.features,
                    kernel::elementwise_grid_x(c.features),
                    static_cast<int>(diffs == 0u));
    }

    check(rig.close(), "rig teardown");

    (void)hipStreamDestroy(stream);
    (void)hipFree(bf16_in0);
    (void)hipFree(bf16_in1);
    (void)hipFree(f32_in0);
    (void)hipFree(f32_in1);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(out_ref[i]);
        (void)hipFree(out_cand[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_elementwise: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_elementwise: PASS\n");
    return 0;
}
