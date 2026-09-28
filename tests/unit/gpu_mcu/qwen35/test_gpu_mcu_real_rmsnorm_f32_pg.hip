#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>

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
constexpr uint32_t kFeatures = 4096u;
constexpr uint32_t kGroup = 128u;
constexpr float kEps = 1e-6f;

void build_inputs(std::vector<float>& input, std::vector<uint16_t>& weight) {
    input.assign(static_cast<size_t>(kRows) * kFeatures, 0.f);
    for (uint32_t i = 0; i < kFeatures; ++i) {
        const int base = static_cast<int>((i * 2654435761u + 12345u) % 2001u) - 1000;
        input[i] = static_cast<float>(base) / 1000.0f;
    }
    weight.assign(kFeatures, 0u);
    for (uint32_t i = 0; i < kFeatures; ++i) {
        const int base = static_cast<int>((i * 104729u + 11u) % 2001u) - 1000;
        weight[i] = static_cast<uint16_t>(
            (static_cast<uint32_t>(static_cast<float>(base) / 1000.0f) + 0u) |
            0u);
        uint32_t ui;
        const float f = static_cast<float>(base) / 1000.0f;
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

hipError_t host_launch(const void* input, const void* weight, void* output,
                       hipStream_t stream) {
    return kernel::launch_rmsnorm(
        input, kernel::RmsNormDataType::F32, weight,
        kernel::RmsNormDataType::BF16, kernel::RmsNormWeightLayout::PER_GROUP,
        kernel::RmsNormWeightMode::DIRECT, output, kernel::RmsNormDataType::F32,
        kRows, kFeatures, kGroup, kFeatures, kFeatures, kEps, stream);
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

    std::vector<float> h_in;
    std::vector<uint16_t> h_wt;
    build_inputs(h_in, h_wt);

    const size_t row_bytes = static_cast<size_t>(kRows) * kFeatures * sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(kFeatures) * sizeof(uint16_t);
    const size_t guard_bytes = 64u * sizeof(float);

    void* in_dev = nullptr;
    void* wt_dev = nullptr;
    void* ref_dev = nullptr;
    void* aql_dev = nullptr;
    if (hipMalloc(&in_dev, row_bytes) != hipSuccess ||
        hipMalloc(&wt_dev, weight_bytes) != hipSuccess ||
        hipMalloc(&ref_dev, row_bytes + guard_bytes) != hipSuccess ||
        hipMalloc(&aql_dev, row_bytes + guard_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    std::vector<uint32_t> guard(guard_bytes / 4u, 0xDEADBEEFu);
    (void)hipMemcpy(in_dev, h_in.data(), row_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(wt_dev, h_wt.data(), weight_bytes, hipMemcpyHostToDevice);
    (void)hipMemset(ref_dev, 0xAB, row_bytes + guard_bytes);
    (void)hipMemset(aql_dev, 0xAB, row_bytes + guard_bytes);

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    check(host_launch(in_dev, wt_dev, ref_dev, stream) == hipSuccess,
          "host launch_rmsnorm f32/pg/direct");
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
        kernel::kRmsNormF32PgDirectSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: rmsnorm f32 code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();
    std::printf(
        "rmsnorm f32/pg/direct hsaco: kernel_object=%p kernarg=%u private=%u "
        "group=%u sizeof(args)=%zu\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        meta.private_segment_size, meta.static_group_segment_size,
        sizeof(kernel::RmsNormF32PgDirectArgs));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::RmsNormF32PgDirectArgs),
          "kernarg segment covers the stable args");
    check(mcu::aql_hidden_args_fits(sizeof(kernel::RmsNormF32PgDirectArgs),
                                    meta.kernarg_segment_size),
          "rmsnorm f32 segment includes hidden args");

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), 1u);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    const uint32_t grid =
        kernel::rmsnorm_bf16_pf_oneplus_grid(kRows, kFeatures, kGroup);
    const uint32_t block = kernel::rmsnorm_bf16_pf_oneplus_block_size(kGroup);

    unsigned char* slot = region.base;
    std::memset(slot, 0, region.slot_stride);
    auto* args = reinterpret_cast<kernel::RmsNormF32PgDirectArgs*>(slot);
    args->input = in_dev;
    args->weight = wt_dev;
    args->output = aql_dev;
    args->rows = kRows;
    args->features = kFeatures;
    args->group_size = kGroup;
    args->input_row_stride = kFeatures;
    args->output_row_stride = kFeatures;
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
    mcu::build_aql_hidden_args(slot, sizeof(kernel::RmsNormF32PgDirectArgs),
                               meta.kernarg_segment_size,
                               mcu::AqlHiddenArgsPolicy::IfFits, desc);
    mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid, 1u, 1u,
                                   block);

    const mcu::GpuAqlPacketTemplate tmpl =
        mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
    check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
    const uint64_t target = queue.host_queued_index();
    const bool drained = wait_queue_drained(queue, target, 15000);
    check(drained, "aql queue drained");

    std::vector<uint32_t> ref_bytes(row_bytes / 4u, 0u);
    std::vector<uint32_t> aql_bytes(row_bytes / 4u, 0u);
    std::vector<uint32_t> ref_guard(guard_bytes / 4u, 0u);
    std::vector<uint32_t> aql_guard(guard_bytes / 4u, 0u);
    (void)hipMemcpy(ref_bytes.data(), ref_dev, row_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(aql_bytes.data(), aql_dev, row_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(ref_guard.data(), static_cast<unsigned char*>(ref_dev) + row_bytes,
                    guard_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(aql_guard.data(), static_cast<unsigned char*>(aql_dev) + row_bytes,
                    guard_bytes, hipMemcpyDeviceToHost);
    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(ref_bytes.data(), aql_bytes.data(), row_bytes,
                             &first, &diffs),
          "host produced aql is bit exact with the host launch");
    if (diffs)
        std::printf("  out diff count=%zu first=%zu\n", diffs, first);
    check(ref_guard == aql_guard, "aql output sentinel intact");

    (void)hipStreamDestroy(stream);
    check(code.shutdown().ok(), "code object shutdown");
    check(queue.shutdown().ok(), "queue shutdown");

    void* mcu_dev = nullptr;
    if (hipMalloc(&mcu_dev, row_bytes + guard_bytes) != hipSuccess) {
        std::printf("FAIL: mcu buffer alloc\n");
        return 1;
    }
    (void)hipMemset(mcu_dev, 0xAB, row_bytes + guard_bytes);

    Rig rig;
    if (!rig.open(0, 4u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_RMSNORM_HSACO,
                             kernel::kRmsNormF32PgDirectSymbol, mcu_meta)) {
        return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, mcu_meta,
                                 mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus,
                                 block, grid, mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};

    std::vector<mcu::McuPlanNode> nodes(4);
    nodes[0].variant_id = 0;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1;
    nodes[1].variant_id = 1;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[1].completion_slot = 0;
    nodes[1].input_slot = mcu::kMcuNoInput;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2;
    nodes[2].completion_slot = 0;
    nodes[2].flags = mcu::kMcuNodeWait;
    nodes[2].next = 3;
    nodes[3].flags = mcu::kMcuNodeEnd;
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuRmsNormInvocation inv{};
    inv.input = reinterpret_cast<uint64_t>(in_dev);
    inv.weight = reinterpret_cast<uint64_t>(wt_dev);
    inv.output = reinterpret_cast<uint64_t>(mcu_dev);
    inv.rows = kRows;
    inv.features = kFeatures;
    inv.group_size = kGroup;
    inv.input_row_stride = kFeatures;
    inv.output_row_stride = kFeatures;
    inv.eps = kEps;
    if (!rig.set_invocations(&inv, 1u)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x524d5346u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 2u;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return 1;
    }
    check(rig.fsm.request_run().ok(), "mcu plan requested");
    check(rig.fsm.wait_plans(1u, 20000).ok(), "mcu plan completed");
    check(rig.drain_queue(30000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == 2u, "two dispatches (rmsnorm + marker)");

    std::vector<uint32_t> mcu_bytes(row_bytes / 4u, 0u);
    std::vector<uint32_t> mcu_guard(guard_bytes / 4u, 0u);
    rig.device_to_host(mcu_bytes.data(), mcu_dev, row_bytes);
    rig.device_to_host(mcu_guard.data(),
                       static_cast<unsigned char*>(mcu_dev) + row_bytes,
                       guard_bytes);
    first = 0;
    diffs = 0;
    check(chain::bytes_equal(ref_bytes.data(), mcu_bytes.data(), row_bytes,
                            &first, &diffs),
          "mcu rmsnorm f32/pg/direct matches the host launch");
    if (diffs)
        std::printf("  mcu out diff count=%zu first=%zu ref=%g mcu=%g\n", diffs,
                    first, ref_bytes[first], mcu_bytes[first]);
    check(mcu_guard == ref_guard, "mcu output sentinel intact");

    std::printf("  rmsnorm f32/pg/direct R=%u F=%u G=%u grid=%u block=%u\n",
                kRows, kFeatures, kGroup, grid, block);

    check(rig.close(), "rig teardown");
    (void)hipFree(in_dev);
    (void)hipFree(wt_dev);
    (void)hipFree(ref_dev);
    (void)hipFree(aql_dev);
    (void)hipFree(mcu_dev);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_rmsnorm_f32_pg: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_rmsnorm_f32_pg: PASS\n");
    return 0;
}
