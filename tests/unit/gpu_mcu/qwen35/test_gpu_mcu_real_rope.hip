#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
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

constexpr uint32_t kHeadDim = kernel::kRopePairHeadDim;
constexpr uint32_t kRotary = kernel::kRopePairRotary;
constexpr uint32_t kPairHalf = kRotary / 2u;
constexpr float kTheta = 10000000.0f;
constexpr uint32_t kGuardBytes = 64u;

struct RopeCase {
    const char* name;
    uint32_t features;
    uint32_t rows;
    const uint32_t* positions;
};

const uint32_t kPos1[1] = {0u};
const uint32_t kPos3[3] = {0u, 31u, 1024u};

const RopeCase kCases[] = {
    {"q_rope_rows1", 4096u, 1u, kPos1},
    {"q_rope_rows3", 4096u, 3u, kPos3},
    {"k_rope_rows3", 1024u, 3u, kPos3},
};

constexpr uint32_t kCaseCount = static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

void build_input(uint32_t features, uint32_t rows, std::vector<float>& out) {
    out.assign(static_cast<size_t>(features) * rows, 0.0f);
    for (size_t i = 0; i < out.size(); ++i) {
        const uint32_t h = static_cast<uint32_t>(i) * 2654435761u + 12345u;
        out[i] = static_cast<float>(static_cast<int>(h % 4001u) - 2000) / 1000.0f;
    }
}

bool wait_queue_drained(const mcu::GpuMcuAqlQueue& queue, uint64_t target,
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

    void* inv_freq = nullptr;
    if (hipMalloc(&inv_freq, 64u * sizeof(double)) != hipSuccess) {
        std::printf("FAIL: inv_freq alloc\n");
        return 1;
    }
    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    check(kernel::launch_rope_inv_freq_init(static_cast<double*>(inv_freq), kPairHalf,
                                            kTheta, stream) == hipSuccess,
          "inv_freq init");
    check(hipStreamSynchronize(stream) == hipSuccess, "inv_freq sync");

    std::vector<std::vector<float>> h_in(kCaseCount);
    std::vector<void*> in_dev(kCaseCount, nullptr);
    std::vector<void*> pos_dev(kCaseCount, nullptr);
    std::vector<void*> ref_dev(kCaseCount, nullptr);
    std::vector<void*> aql_dev(kCaseCount, nullptr);
    std::vector<void*> mcu_dev(kCaseCount, nullptr);
    std::vector<size_t> in_bytes(kCaseCount, 0u);
    std::vector<size_t> out_bytes(kCaseCount, 0u);

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RopeCase& c = kCases[i];
        build_input(c.features, c.rows, h_in[i]);
        in_bytes[i] = static_cast<size_t>(c.features) * c.rows * sizeof(float);
        out_bytes[i] = static_cast<size_t>(c.features) * c.rows * sizeof(uint16_t);
        const size_t guarded = out_bytes[i] + kGuardBytes;
        if (hipMalloc(&in_dev[i], in_bytes[i]) != hipSuccess ||
            hipMalloc(&pos_dev[i], c.rows * sizeof(uint32_t)) != hipSuccess ||
            hipMalloc(&ref_dev[i], guarded) != hipSuccess ||
            hipMalloc(&aql_dev[i], guarded) != hipSuccess ||
            hipMalloc(&mcu_dev[i], guarded) != hipSuccess) {
            std::printf("FAIL: buffer alloc\n");
            return 1;
        }
        (void)hipMemcpy(in_dev[i], h_in[i].data(), in_bytes[i],
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(pos_dev[i], c.positions, c.rows * sizeof(uint32_t),
                        hipMemcpyHostToDevice);
        (void)hipMemset(ref_dev[i], 0xAB, guarded);
        (void)hipMemset(aql_dev[i], 0xAB, guarded);
        (void)hipMemset(mcu_dev[i], 0xAB, guarded);
    }

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RopeCase& c = kCases[i];
        const hipError_t herr = kernel::launch_rope_f32_bf16(
            static_cast<const float*>(in_dev[i]), c.features,
            static_cast<ps::bf16_t*>(ref_dev[i]), c.features,
            static_cast<const uint32_t*>(pos_dev[i]), c.rows, c.features, kHeadDim,
            kRotary, kTheta, static_cast<const double*>(inv_freq), stream);
        check(herr == hipSuccess, "host production rope launcher");
    }
    check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_ROPE_HSACO, kernel::kRopeF32Bf16PairSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: rope code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf("rope pair hsaco: kernel_object=%p kernarg=%u private=%u group=%u "
                "sizeof(args)=%zu\n",
                reinterpret_cast<void*>(meta.kernel_object),
                meta.kernarg_segment_size, meta.private_segment_size,
                meta.static_group_segment_size,
                sizeof(kernel::RopeF32Bf16PairArgs));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::RopeF32Bf16PairArgs),
          "kernarg segment covers the stable args");
    const bool hidden_fits = mcu::aql_hidden_args_fits(
        sizeof(kernel::RopeF32Bf16PairArgs), meta.kernarg_segment_size);
    std::printf("  hidden_args_fits=%d (IfFits -> both sides skip\n",
                static_cast<int>(hidden_fits));

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    std::vector<std::vector<uint32_t>> refs(kCaseCount);
    std::vector<std::vector<uint32_t>> guards(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RopeCase& c = kCases[i];
        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        auto* args = reinterpret_cast<kernel::RopeF32Bf16PairArgs*>(slot);
        args->input = static_cast<const float*>(in_dev[i]);
        args->input_row_stride = c.features;
        args->output = static_cast<ps::bf16_t*>(aql_dev[i]);
        args->output_row_stride = c.features;
        args->positions = static_cast<const uint32_t*>(pos_dev[i]);
        args->inv_freq = static_cast<const double*>(inv_freq);
        args->features = c.features;

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = c.rows * kernel::kRopePairBlock;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(kernel::kRopePairBlock);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::RopeF32Bf16PairArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, c.rows, 1u,
                                       1u, kernel::kRopePairBlock);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_queue_drained(queue, queue.host_queued_index(), 15000),
              "aql queue drained");

        refs[i].assign(out_bytes[i] / 4u, 0u);
        guards[i].assign(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> aql(out_bytes[i] / 4u, 0u);
        std::vector<uint32_t> aql_g(kGuardBytes / 4u, 0u);
        (void)hipMemcpy(refs[i].data(), ref_dev[i], out_bytes[i],
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(guards[i].data(),
                        static_cast<unsigned char*>(ref_dev[i]) + out_bytes[i],
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(aql.data(), aql_dev[i], out_bytes[i],
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(aql_g.data(),
                        static_cast<unsigned char*>(aql_dev[i]) + out_bytes[i],
                        kGuardBytes, hipMemcpyDeviceToHost);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), aql.data(), out_bytes[i], &first,
                                 &diffs),
              "host produced aql is bit exact with the host launch");
        if (diffs)
            std::printf("  %s aql diff count=%zu first=%zu\n", c.name, diffs, first);
        check(aql_g == guards[i], "aql output sentinel intact");
    }

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");
    (void)hipStreamDestroy(stream);

    Rig rig;
    if (!rig.open(0, kCaseCount + 3u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ROPE_HSACO,
                              kernel::kRopeF32Bf16PairSymbol, mcu_meta)) {
        return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_meta, mcu::kMcuKernargRecipeRope,
            kernel::kRopePairBlock, kCases[i].rows, mcu::kMcuAgentAqlPolicy));
    }
    variants.push_back(mcu::make_probe_variant(
        static_cast<uint16_t>(kCaseCount), rig.meta, 64u, 1u,
        mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeRope;
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

    std::vector<mcu::McuRopeInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RopeCase& c = kCases[i];
        (void)hipMemset(mcu_dev[i], 0xAB, out_bytes[i] + kGuardBytes);
        invocations[i] = mcu::McuRopeInvocation{};
        invocations[i].input = reinterpret_cast<uint64_t>(in_dev[i]);
        invocations[i].input_row_stride = c.features;
        invocations[i].output = reinterpret_cast<uint64_t>(mcu_dev[i]);
        invocations[i].output_row_stride = c.features;
        invocations[i].positions = reinterpret_cast<uint64_t>(pos_dev[i]);
        invocations[i].inv_freq = reinterpret_cast<uint64_t>(inv_freq);
        invocations[i].features = c.features;
    }
    if (!rig.set_rope_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x524f5045u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = kCaseCount + 1u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }
    check(rig.fsm.request_run().ok(), "mcu plan requested");
    check(rig.fsm.wait_plans(1u, 20000).ok(), "mcu plan completed");
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == static_cast<uint64_t>(kCaseCount) + 1u,
          "dispatch count");

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RopeCase& c = kCases[i];
        std::vector<uint32_t> mcu_out(out_bytes[i] / 4u, 0u);
        std::vector<uint32_t> mcu_g(kGuardBytes / 4u, 0u);
        rig.device_to_host(mcu_out.data(), mcu_dev[i], out_bytes[i]);
        rig.device_to_host(mcu_g.data(),
                           static_cast<unsigned char*>(mcu_dev[i]) + out_bytes[i],
                           kGuardBytes);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), mcu_out.data(), out_bytes[i],
                                 &first, &diffs),
              "mcu rope matches the host launch");
        if (diffs)
            std::printf("  %s mcu diff count=%zu first=%zu\n", c.name, diffs, first);
        check(mcu_g == guards[i], "mcu output sentinel intact");
        std::printf("  %s rows=%u F=%u pos0=%u poslast=%u grid=%u wg=%u ok=%d\n",
                    c.name, c.rows, c.features, c.positions[0],
                    c.positions[c.rows - 1u], c.rows, kernel::kRopePairBlock,
                    static_cast<int>(diffs == 0u));
    }

    check(rig.close(), "rig teardown");
    (void)hipFree(inv_freq);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(in_dev[i]);
        (void)hipFree(pos_dev[i]);
        (void)hipFree(ref_dev[i]);
        (void)hipFree(aql_dev[i]);
        (void)hipFree(mcu_dev[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_rope: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_rope: PASS\n");
    return 0;
}
