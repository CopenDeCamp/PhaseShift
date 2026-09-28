#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/l2_normalize.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>

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

constexpr uint32_t kFeatures = 2048u;
constexpr uint32_t kGroup = 128u;
constexpr uint32_t kInStride = 8192u;
constexpr float kEps = 1e-6f;
constexpr uint32_t kGuardBytes = 64u;

struct L2Case {
    uint32_t rows;
};

const L2Case kCases[] = {{1u}, {4u}};
constexpr uint32_t kCaseCount =
    static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

hipError_t host_launch(const L2Case& c, const void* input, void* output,
                       hipStream_t stream) {
    return kernel::launch_l2_normalize_bf16_f32(
        static_cast<const ps::bf16_t*>(input), kInStride,
        static_cast<float*>(output), kFeatures, c.rows, kFeatures, kGroup, kEps,
        stream);
}

void fill_args(const L2Case& c, const void* input, void* output,
               kernel::L2NormalizeArgs& a) {
    a = kernel::L2NormalizeArgs{};
    a.input = input;
    a.output = output;
    a.input_row_stride = kInStride;
    a.output_row_stride = kFeatures;
    a.group_size = kGroup;
    a.eps = kEps;
}

void fill_invocation(const L2Case& c, const void* input, void* output,
                     mcu::McuL2NormalizeInvocation& v) {
    v = mcu::McuL2NormalizeInvocation{};
    v.input = reinterpret_cast<uint64_t>(input);
    v.output = reinterpret_cast<uint64_t>(output);
    v.input_row_stride = kInStride;
    v.output_row_stride = kFeatures;
    v.group_size = kGroup;
    v.eps = kEps;
    (void)c;
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

    constexpr uint32_t kMaxRows = 4u;
    const size_t in_bytes = static_cast<size_t>(kMaxRows) * kInStride * 2u;
    const size_t out_bytes =
        static_cast<size_t>(kMaxRows) * kFeatures * 4u + kGuardBytes;

    void* input = nullptr;
    if (hipMalloc(&input, in_bytes) != hipSuccess) {
        std::printf("FAIL: input alloc\n");
        return 1;
    }
    std::vector<uint16_t> h_in(static_cast<size_t>(kMaxRows) * kInStride);
    for (size_t i = 0; i < h_in.size(); ++i)
        h_in[i] = static_cast<uint16_t>(0x3C00u + ((i * 2654435761u) % 900u));
    (void)hipMemcpy(input, h_in.data(), in_bytes, hipMemcpyHostToDevice);

    std::vector<void*> out_ref(kCaseCount, nullptr);
    std::vector<void*> out_cand(kCaseCount, nullptr);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (hipMalloc(&out_ref[i], out_bytes) != hipSuccess ||
            hipMalloc(&out_cand[i], out_bytes) != hipSuccess) {
            std::printf("FAIL: output alloc\n");
            return 1;
        }
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
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO,
        kernel::kL2NormalizeBf16F32Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: l2 code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf(
        "l2 hsaco: kernel_object=%p kernarg=%u group=%u sizeof(args)=%zu "
        "hidden_fits=%d\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        meta.static_group_segment_size, sizeof(kernel::L2NormalizeArgs),
        static_cast<int>(mcu::aql_hidden_args_fits(
            sizeof(kernel::L2NormalizeArgs), meta.kernarg_segment_size)));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::L2NormalizeArgs),
          "kernarg segment covers L2NormalizeArgs");

    const uint32_t slot_stride =
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size);
    auto region_res = queue.allocate_kernarg(slot_stride, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    std::vector<std::vector<uint32_t>> refs(kCaseCount);
    std::vector<std::vector<uint32_t>> guards(kCaseCount);
    const size_t groups = kernel::l2_normalize_groups(kFeatures, kGroup);

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const L2Case& c = kCases[i];
        const size_t bytes = static_cast<size_t>(c.rows) * kFeatures * 4u;

        (void)hipMemset(out_ref[i], 0xAB, out_bytes);
        (void)hipMemset(out_cand[i], 0xAB, out_bytes);
        check(host_launch(c, input, out_ref[i], stream) == hipSuccess,
              "host l2 launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        kernel::L2NormalizeArgs args{};
        fill_args(c, input, out_cand[i], args);
        std::memcpy(slot, &args, sizeof(args));

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = groups * kernel::kL2NormalizeThreads;
        desc.global_work_items_y = c.rows;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(kernel::kL2NormalizeThreads);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::L2NormalizeArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, groups,
                                       c.rows, 1u, kernel::kL2NormalizeThreads);

        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_drained(queue, queue.host_queued_index(), 20000),
              "aql queue drained");

        const size_t words = bytes / 4u;
        refs[i].assign(words, 0u);
        guards[i].assign(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> cand(words, 0u);
        std::vector<uint32_t> cand_guard(kGuardBytes / 4u, 0u);
        (void)hipMemcpy(refs[i].data(), out_ref[i], bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand.data(), out_cand[i], bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(guards[i].data(),
                        static_cast<unsigned char*>(out_ref[i]) + bytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand_guard.data(),
                        static_cast<unsigned char*>(out_cand[i]) + bytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), cand.data(), bytes, &first,
                                 &diffs),
              "host produced aql is bit exact with the host launch");
        if (diffs)
            std::printf("  rows=%u aql diff count=%zu first=%zu\n", c.rows, diffs,
                        first);
        check(cand_guard == guards[i], "aql output sentinel intact");
        std::printf("  rows=%u groups=%zu ok=%d\n", c.rows, groups,
                    static_cast<int>(diffs == 0u));
    }

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");

    Rig rig;
    if (!rig.open(0, kCaseCount + 3u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO,
                             kernel::kL2NormalizeBf16F32Symbol, mcu_meta))
        return 1;

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_meta, mcu::kMcuKernargRecipeL2Normalize,
            kernel::kL2NormalizeThreads, static_cast<uint32_t>(groups),
            mcu::kMcuAgentAqlPolicy, mcu::AqlHiddenArgsPolicy::IfFits,
            kCases[i].rows, 1u));
    }
    variants.push_back(mcu::make_probe_variant(static_cast<uint16_t>(kCaseCount),
                                               rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeL2Normalize;
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

    std::vector<mcu::McuL2NormalizeInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipMemset(out_cand[i], 0xAB, out_bytes);
        fill_invocation(kCases[i], input, out_cand[i], invocations[i]);
    }
    if (!rig.set_l2_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4c324e58u;
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
        const L2Case& c = kCases[i];
        const size_t bytes = static_cast<size_t>(c.rows) * kFeatures * 4u;
        const size_t words = bytes / 4u;
        std::vector<uint32_t> mcu_out(words, 0u);
        std::vector<uint32_t> mcu_guard(kGuardBytes / 4u, 0u);
        rig.device_to_host(mcu_out.data(), out_cand[i], bytes);
        rig.device_to_host(mcu_guard.data(),
                           static_cast<unsigned char*>(out_cand[i]) + bytes,
                           kGuardBytes);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), mcu_out.data(), bytes, &first,
                                 &diffs),
              "mcu l2 normalize matches the host launch");
        if (diffs)
            std::printf("  rows=%u mcu diff count=%zu first=%zu\n", c.rows, diffs,
                        first);
        check(mcu_guard == guards[i], "mcu output sentinel intact");
    }

    check(rig.close(), "rig teardown");

    (void)hipStreamDestroy(stream);
    (void)hipFree(input);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(out_ref[i]);
        (void)hipFree(out_cand[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_l2_normalize: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_l2_normalize: PASS\n");
    return 0;
}
