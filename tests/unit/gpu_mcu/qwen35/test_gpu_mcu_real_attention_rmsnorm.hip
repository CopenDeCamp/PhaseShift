#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
constexpr uint32_t kGroup = 256u;
constexpr float kEps = 1e-6f;
constexpr uint32_t kGuardBytes = 64u;

struct NormCase {
    const char* name;
    uint32_t features;
};

const NormCase kCases[] = {
    {"q_norm", 4096u},
    {"k_norm", 1024u},
};

constexpr uint32_t kCaseCount = static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

void build_inputs(uint32_t features, std::vector<uint16_t>& input,
                  std::vector<uint16_t>& weight) {
    input.assign(static_cast<size_t>(kRows) * features, 0u);
    for (uint32_t i = 0; i < features; ++i) {
        const uint32_t h = i * 2654435761u + 12345u;
        const int base = static_cast<int>(h % 2001u) - 1000;
        const float f = static_cast<float>(base) / 1000.0f;
        uint32_t ui;
        std::memcpy(&ui, &f, 4);
        input[i] = static_cast<uint16_t>((ui + 0x8000u) >> 16);
    }
    weight.assign(kGroup, 0u);
    for (uint32_t i = 0; i < kGroup; ++i) {
        const uint32_t h = i * 104729u + 11u;
        const int base = static_cast<int>(h % 2001u) - 1000;
        const float f = static_cast<float>(base) / 1000.0f;
        uint32_t ui;
        std::memcpy(&ui, &f, 4);
        weight[i] = static_cast<uint16_t>((ui + 0x8000u) >> 16);
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

    std::vector<std::vector<uint16_t>> h_in(kCaseCount);
    std::vector<std::vector<uint16_t>> h_wt(kCaseCount);
    std::vector<void*> in_dev(kCaseCount, nullptr);
    std::vector<void*> wt_dev(kCaseCount, nullptr);
    std::vector<void*> ref_dev(kCaseCount, nullptr);
    std::vector<void*> aql_dev(kCaseCount, nullptr);
    std::vector<void*> mcu_dev(kCaseCount, nullptr);
    std::vector<size_t> row_bytes(kCaseCount, 0u);
    std::vector<size_t> weight_bytes(kCaseCount, 0u);

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const NormCase& c = kCases[i];
        build_inputs(c.features, h_in[i], h_wt[i]);
        row_bytes[i] = static_cast<size_t>(kRows) * c.features * sizeof(float);
        weight_bytes[i] = static_cast<size_t>(kGroup) * sizeof(uint16_t);
        const size_t out_bytes = row_bytes[i] + kGuardBytes;
        if (hipMalloc(&in_dev[i], row_bytes[i]) != hipSuccess ||
            hipMalloc(&wt_dev[i], weight_bytes[i]) != hipSuccess ||
            hipMalloc(&ref_dev[i], out_bytes) != hipSuccess ||
            hipMalloc(&aql_dev[i], out_bytes) != hipSuccess ||
            hipMalloc(&mcu_dev[i], out_bytes) != hipSuccess) {
            std::printf("FAIL: buffer alloc\n");
            return 1;
        }
        (void)hipMemcpy(in_dev[i], h_in[i].data(), row_bytes[i],
                        hipMemcpyHostToDevice);
        (void)hipMemcpy(wt_dev[i], h_wt[i].data(), weight_bytes[i],
                        hipMemcpyHostToDevice);
        (void)hipMemset(ref_dev[i], 0xAB, out_bytes);
        (void)hipMemset(aql_dev[i], 0xAB, out_bytes);
        (void)hipMemset(mcu_dev[i], 0xAB, out_bytes);
    }

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const NormCase& c = kCases[i];
        const hipError_t herr = kernel::launch_rmsnorm(
            in_dev[i], kernel::RmsNormDataType::BF16, wt_dev[i],
            kernel::RmsNormDataType::BF16, kernel::RmsNormWeightLayout::PER_GROUP,
            kernel::RmsNormWeightMode::ONE_PLUS, ref_dev[i],
            kernel::RmsNormDataType::F32, kRows, c.features, kGroup, c.features,
            c.features, kEps, stream);
        check(herr == hipSuccess, "host launch_rmsnorm bf16/f32/pg/one_plus");
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
        queue, PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
        kernel::kRmsNormBf16F32PgOnePlusSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: attention rmsnorm code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();
    std::printf(
        "rmsnorm bf16/f32/pg/one_plus hsaco: kernel_object=%p kernarg=%u "
        "private=%u group=%u sizeof(args)=%zu\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        meta.private_segment_size, meta.static_group_segment_size,
        sizeof(kernel::RmsNormBf16F32PgOnePlusArgs));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::RmsNormBf16F32PgOnePlusArgs),
          "kernarg segment covers the stable args");
    check(mcu::aql_hidden_args_fits(sizeof(kernel::RmsNormBf16F32PgOnePlusArgs),
                                    meta.kernarg_segment_size),
          "attention rmsnorm segment includes hidden args");

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    std::vector<std::vector<uint32_t>> ref_bytes(kCaseCount);
    std::vector<std::vector<uint32_t>> ref_guard(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const NormCase& c = kCases[i];
        const uint32_t grid =
            kernel::rmsnorm_bf16_pf_oneplus_grid(kRows, c.features, kGroup);
        const uint32_t block = kernel::rmsnorm_bf16_pf_oneplus_block_size(kGroup);

        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        auto* args = reinterpret_cast<kernel::RmsNormBf16F32PgOnePlusArgs*>(slot);
        args->input = in_dev[i];
        args->weight = wt_dev[i];
        args->output = aql_dev[i];
        args->rows = kRows;
        args->features = c.features;
        args->group_size = kGroup;
        args->input_row_stride = c.features;
        args->output_row_stride = c.features;
        args->eps = kEps;

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = grid * block;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(block);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot,
                                   sizeof(kernel::RmsNormBf16F32PgOnePlusArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid, 1u,
                                       1u, block);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_queue_drained(queue, queue.host_queued_index(), 15000),
              "aql queue drained");

        ref_bytes[i].assign(row_bytes[i] / 4u, 0u);
        ref_guard[i].assign(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> aql(row_bytes[i] / 4u, 0u);
        std::vector<uint32_t> aql_g(kGuardBytes / 4u, 0u);
        (void)hipMemcpy(ref_bytes[i].data(), ref_dev[i], row_bytes[i],
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(ref_guard[i].data(),
                        static_cast<unsigned char*>(ref_dev[i]) + row_bytes[i],
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(aql.data(), aql_dev[i], row_bytes[i],
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(aql_g.data(),
                        static_cast<unsigned char*>(aql_dev[i]) + row_bytes[i],
                        kGuardBytes, hipMemcpyDeviceToHost);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(ref_bytes[i].data(), aql.data(), row_bytes[i],
                                 &first, &diffs),
              "host produced aql is bit exact with the host launch");
        if (diffs)
            std::printf("  %s aql diff count=%zu first=%zu\n", c.name, diffs, first);
        check(aql_g == ref_guard[i], "aql output sentinel intact");
        std::printf("  %s F=%u G=%u grid=%u block=%u\n", c.name, c.features, kGroup,
                    grid, block);
    }

    (void)hipStreamDestroy(stream);
    check(code.shutdown().ok(), "code object shutdown");
    check(queue.shutdown().ok(), "queue shutdown");

    Rig rig;
    if (!rig.open(0, kCaseCount + 3u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                              kernel::kRmsNormBf16F32PgOnePlusSymbol, mcu_meta)) {
        return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const NormCase& c = kCases[i];
        const uint32_t grid =
            kernel::rmsnorm_bf16_pf_oneplus_grid(kRows, c.features, kGroup);
        const uint32_t block = kernel::rmsnorm_bf16_pf_oneplus_block_size(kGroup);
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_meta,
            mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus, block, grid,
            mcu::kMcuAgentAqlPolicy));
    }
    variants.push_back(mcu::make_probe_variant(
        static_cast<uint16_t>(kCaseCount), rig.meta, 64u, 1u,
        mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
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

    std::vector<mcu::McuRmsNormInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const NormCase& c = kCases[i];
        const size_t out_bytes = row_bytes[i] + kGuardBytes;
        (void)hipMemset(mcu_dev[i], 0xAB, out_bytes);
        invocations[i] = mcu::McuRmsNormInvocation{};
        invocations[i].input = reinterpret_cast<uint64_t>(in_dev[i]);
        invocations[i].weight = reinterpret_cast<uint64_t>(wt_dev[i]);
        invocations[i].output = reinterpret_cast<uint64_t>(mcu_dev[i]);
        invocations[i].rows = kRows;
        invocations[i].features = c.features;
        invocations[i].group_size = kGroup;
        invocations[i].input_row_stride = c.features;
        invocations[i].output_row_stride = c.features;
        invocations[i].eps = kEps;
    }
    if (!rig.set_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4154524eu;
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
    if (rig.fsm.fault_code() != mcu::McuFaultCode::none) {
        std::printf("  fault code=%u pc=%u\n",
                    static_cast<unsigned>(rig.fsm.fault_code()),
                    rig.fsm.fault_pc());
    }
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == static_cast<uint64_t>(kCaseCount) + 1u,
          "dispatch count");

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const NormCase& c = kCases[i];
        std::vector<uint32_t> mcu(row_bytes[i] / 4u, 0u);
        std::vector<uint32_t> mcu_g(kGuardBytes / 4u, 0u);
        rig.device_to_host(mcu.data(), mcu_dev[i], row_bytes[i]);
        rig.device_to_host(mcu_g.data(),
                           static_cast<unsigned char*>(mcu_dev[i]) + row_bytes[i],
                           kGuardBytes);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(ref_bytes[i].data(), mcu.data(), row_bytes[i],
                                 &first, &diffs),
              "mcu attention rmsnorm matches the host launch");
        if (diffs)
            std::printf("  %s mcu diff count=%zu first=%zu\n", c.name, diffs, first);
        check(mcu_g == ref_guard[i], "mcu output sentinel intact");
        std::printf("  %s R=%u F=%u G=%u ok=%d\n", c.name, kRows, c.features, kGroup,
                    static_cast<int>(diffs == 0u));
    }

    check(rig.close(), "rig teardown");
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(in_dev[i]);
        (void)hipFree(wt_dev[i]);
        (void)hipFree(ref_dev[i]);
        (void)hipFree(aql_dev[i]);
        (void)hipFree(mcu_dev[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_attention_rmsnorm: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_attention_rmsnorm: PASS\n");
    return 0;
}
