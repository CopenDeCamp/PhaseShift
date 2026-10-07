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
constexpr uint32_t kInFeatures = 8192u;
constexpr uint32_t kOutFeatures = 4096u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kGuardBytes = 64u;

struct SplitCase {
    const char* name;
    kernel::ElementwiseSplitLayout layout;
    uint32_t head_dim;
    uint32_t reserved;
};

const SplitCase kCases[] = {
    {"split_interleaved_heads", kernel::ElementwiseSplitLayout::InterleavedHeads,
     kHeadDim, 1u},
    {"split_halves", kernel::ElementwiseSplitLayout::Halves, kHeadDim, 0u},
};

constexpr uint32_t kCaseCount = static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));
constexpr size_t kInBytes = static_cast<size_t>(kInFeatures) * sizeof(uint16_t);
constexpr size_t kOutBytes = static_cast<size_t>(kOutFeatures) * sizeof(uint16_t);

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

void fill_args(const SplitCase& c, void* in, void* o0, void* o1,
               kernel::ElementwiseArgs& a) {
    a = kernel::ElementwiseArgs{};
    a.in0 = in;
    a.in0_row_stride = kInFeatures;
    a.out0 = o0;
    a.out0_row_stride = kOutFeatures;
    a.out1 = o1;
    a.out1_row_stride = kOutFeatures;
    a.features = kOutFeatures;
    a.aux = c.head_dim;
    a.reserved = c.reserved;
    a.rows = kRows;
}

void fill_invocation(const SplitCase& c, void* in, void* o0, void* o1,
                     mcu::McuElementwiseInvocation& v) {
    v = mcu::McuElementwiseInvocation{};
    v.in0 = reinterpret_cast<uint64_t>(in);
    v.in0_row_stride = kInFeatures;
    v.out0 = reinterpret_cast<uint64_t>(o0);
    v.out0_row_stride = kOutFeatures;
    v.out1 = reinterpret_cast<uint64_t>(o1);
    v.out1_row_stride = kOutFeatures;
    v.features = kOutFeatures;
    v.aux = c.head_dim;
    v.reserved = c.reserved;
    v.rows = kRows;
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

    void* in_buf = nullptr;
    std::vector<void*> o0_ref(kCaseCount, nullptr);
    std::vector<void*> o0_cand(kCaseCount, nullptr);
    std::vector<void*> o1_ref(kCaseCount, nullptr);
    std::vector<void*> o1_cand(kCaseCount, nullptr);
    if (hipMalloc(&in_buf, kInBytes) != hipSuccess) {
        std::printf("FAIL: input alloc\n");
        return 1;
    }
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const size_t bytes = kOutBytes + kGuardBytes;
        if (hipMalloc(&o0_ref[i], bytes) != hipSuccess ||
            hipMalloc(&o0_cand[i], bytes) != hipSuccess ||
            hipMalloc(&o1_ref[i], bytes) != hipSuccess ||
            hipMalloc(&o1_cand[i], bytes) != hipSuccess) {
            std::printf("FAIL: output alloc\n");
            return 1;
        }
    }

    std::vector<uint16_t> h_in(kInFeatures);
    for (uint32_t i = 0; i < kInFeatures; ++i) {
        const uint32_t h = i * 2654435761u + 7919u;
        h_in[i] = static_cast<uint16_t>(0x3A00u + (h % 1900u));
    }
    if (hipMemcpy(in_buf, h_in.data(), kInBytes, hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: input upload\n");
        return 1;
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
        queue, PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
        kernel::kElementwiseSplitBf16Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf("split: kernarg=%u group=%u private=%u hidden_fits=%d\n",
                meta.kernarg_segment_size, meta.static_group_segment_size,
                meta.private_segment_size,
                static_cast<int>(mcu::aql_hidden_args_fits(
                    sizeof(kernel::ElementwiseArgs), meta.kernarg_segment_size)));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::ElementwiseArgs),
          "kernarg segment covers ElementwiseArgs");
    check(mcu::aql_hidden_args_fits(sizeof(kernel::ElementwiseArgs),
                                    meta.kernarg_segment_size),
          "split segment includes hidden args");

    const uint32_t slot_stride =
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size);
    auto region_res = queue.allocate_kernarg(slot_stride, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    std::vector<std::vector<uint32_t>> o0_refs(kCaseCount);
    std::vector<std::vector<uint32_t>> o1_refs(kCaseCount);
    std::vector<std::vector<uint32_t>> o0_guards(kCaseCount);
    std::vector<std::vector<uint32_t>> o1_guards(kCaseCount);

    const uint32_t grid_x = kernel::elementwise_grid_x(kOutFeatures);
    std::printf("grid_x=%u rows=%u in_f=%u out_f=%u head_dim=%u\n", grid_x, kRows,
                kInFeatures, kOutFeatures, kHeadDim);

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const SplitCase& c = kCases[i];

        (void)hipMemset(o0_ref[i], 0xAB, kOutBytes + kGuardBytes);
        (void)hipMemset(o1_ref[i], 0xAB, kOutBytes + kGuardBytes);
        (void)hipMemset(o0_cand[i], 0xAB, kOutBytes + kGuardBytes);
        (void)hipMemset(o1_cand[i], 0xAB, kOutBytes + kGuardBytes);

        const hipError_t herr = kernel::launch_split_bf16(
            static_cast<const ps::bf16_t*>(in_buf), kInFeatures,
            static_cast<ps::bf16_t*>(o0_ref[i]), kOutFeatures,
            static_cast<ps::bf16_t*>(o1_ref[i]), kOutFeatures, kRows, kOutFeatures,
            c.layout, c.head_dim, stream);
        check(herr == hipSuccess, "host production launcher");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        kernel::ElementwiseArgs args{};
        fill_args(c, in_buf, o0_cand[i], o1_cand[i], args);
        std::memcpy(slot, &args, sizeof(args));

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = grid_x * kernel::kElementwiseBlock;
        desc.global_work_items_y = kRows;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(kernel::kElementwiseBlock);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::ElementwiseArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid_x,
                                       kRows, 1u, kernel::kElementwiseBlock);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_drained(queue, queue.host_queued_index(), 15000),
              "aql queue drained");

        o0_refs[i].assign(kOutBytes / 4u, 0u);
        o1_refs[i].assign(kOutBytes / 4u, 0u);
        o0_guards[i].assign(kGuardBytes / 4u, 0u);
        o1_guards[i].assign(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> o0_c(kOutBytes / 4u, 0u);
        std::vector<uint32_t> o1_c(kOutBytes / 4u, 0u);
        std::vector<uint32_t> o0_cg(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> o1_cg(kGuardBytes / 4u, 0u);
        (void)hipMemcpy(o0_refs[i].data(), o0_ref[i], kOutBytes,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(o1_refs[i].data(), o1_ref[i], kOutBytes,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(o0_c.data(), o0_cand[i], kOutBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(o1_c.data(), o1_cand[i], kOutBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(o0_guards[i].data(),
                        static_cast<unsigned char*>(o0_ref[i]) + kOutBytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(o1_guards[i].data(),
                        static_cast<unsigned char*>(o1_ref[i]) + kOutBytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(o0_refs[i].data(), o0_c.data(), kOutBytes, &first,
                                 &diffs),
              "aql q output is bit exact with the host launch");
        if (diffs)
            std::printf("  %s aql out0 diff count=%zu first=%zu\n", c.name, diffs,
                        first);
        check(chain::bytes_equal(o1_refs[i].data(), o1_c.data(), kOutBytes, &first,
                                 &diffs),
              "aql gate output is bit exact with the host launch");
        if (diffs)
            std::printf("  %s aql out1 diff count=%zu first=%zu\n", c.name, diffs,
                        first);
        std::vector<uint32_t> o0_hg(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> o1_hg(kGuardBytes / 4u, 0u);
        (void)hipMemcpy(o0_hg.data(),
                        static_cast<unsigned char*>(o0_cand[i]) + kOutBytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(o1_hg.data(),
                        static_cast<unsigned char*>(o1_cand[i]) + kOutBytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        check(o0_hg == o0_guards[i] && o1_hg == o1_guards[i],
              "aql output sentinel intact");
    }

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");

    Rig rig;
    if (!rig.open(0, 8u, 4u, 4u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords, meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    std::printf("rig slot_stride=%u needed_segment=%u\n", rig.region.slot_stride,
                meta.kernarg_segment_size);
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO,
                              kernel::kElementwiseSplitBf16Symbol, mcu_meta))
        return 1;

    std::vector<mcu::McuKernelVariantDesc> variants;
    variants.push_back(mcu::make_kernel_variant(
        0u, mcu_meta, mcu::kMcuKernargRecipeElementwise, kernel::kElementwiseBlock,
        grid_x, mcu::kMcuAgentAqlPolicy));
    variants.push_back(mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = 0u;
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeElementwise;
        nodes[i].completion_slot = 0;
        nodes[i].invocation_index = i;
        nodes[i].flags = mcu::kMcuNodeDispatch;
        nodes[i].next = static_cast<uint16_t>(i + 1u);
    }
    nodes[kCaseCount].variant_id = 1u;
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
        const SplitCase& c = kCases[i];
        (void)hipMemset(o0_cand[i], 0xAB, kOutBytes + kGuardBytes);
        (void)hipMemset(o1_cand[i], 0xAB, kOutBytes + kGuardBytes);
        fill_invocation(c, in_buf, o0_cand[i], o1_cand[i], invocations[i]);
    }
    if (!rig.set_elementwise_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x53504c31u;
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
        const SplitCase& c = kCases[i];
        std::vector<uint32_t> o0_m(kOutBytes / 4u, 0u);
        std::vector<uint32_t> o1_m(kOutBytes / 4u, 0u);
        std::vector<uint32_t> o0_mg(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> o1_mg(kGuardBytes / 4u, 0u);
        rig.device_to_host(o0_m.data(), o0_cand[i], kOutBytes);
        rig.device_to_host(o1_m.data(), o1_cand[i], kOutBytes);
        rig.device_to_host(o0_mg.data(),
                           static_cast<unsigned char*>(o0_cand[i]) + kOutBytes,
                           kGuardBytes);
        rig.device_to_host(o1_mg.data(),
                           static_cast<unsigned char*>(o1_cand[i]) + kOutBytes,
                           kGuardBytes);
        size_t first = 0;
        size_t diffs0 = 0;
        size_t diffs1 = 0;
        check(chain::bytes_equal(o0_refs[i].data(), o0_m.data(), kOutBytes, &first,
                                 &diffs0),
              "mcu q output matches the host launch");
        check(chain::bytes_equal(o1_refs[i].data(), o1_m.data(), kOutBytes, &first,
                                 &diffs1),
              "mcu gate output matches the host launch");
        if (diffs0)
            std::printf("  %s mcu out0 diff count=%zu first=%zu\n", c.name, diffs0,
                        first);
        if (diffs1)
            std::printf("  %s mcu out1 diff count=%zu first=%zu\n", c.name, diffs1,
                        first);
        check(o0_mg == o0_guards[i] && o1_mg == o1_guards[i],
              "mcu output sentinel intact");
        std::printf("  %s layout=%s head_dim=%u reserved=%u grid_x=%u ok=%d\n",
                    c.name,
                    c.reserved != 0u ? "InterleavedHeads" : "Halves", c.head_dim,
                    c.reserved, grid_x,
                    static_cast<int>(diffs0 == 0u && diffs1 == 0u));
    }

    check(rig.close(), "rig teardown");

    (void)hipStreamDestroy(stream);
    (void)hipFree(in_buf);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(o0_ref[i]);
        (void)hipFree(o0_cand[i]);
        (void)hipFree(o1_ref[i]);
        (void)hipFree(o1_cand[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_split: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_split: PASS\n");
    return 0;
}
