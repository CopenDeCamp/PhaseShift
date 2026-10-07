#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
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
constexpr uint32_t kGuardBytes = 64u;

struct GemmCase {
    uint32_t n;
    uint32_t k;
};

const GemmCase kCases[] = {
    {8192u, 2560u}, {4096u, 2560u}, {32u, 2560u},
    {2560u, 4096u}, {9216u, 2560u}, {2560u, 9216u},
};

constexpr uint32_t kCaseCount =
    static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

hipError_t host_launch(const GemmCase& c, const void* weight, const void* input,
                       void* output, hipStream_t stream) {
    const kernel::Bf16GemmConfig config{kernel::Bf16GemmConfigId::ExactRows, 1u};
    return kernel::launch_gemm_bf16(
        config, static_cast<const ps::bf16_t*>(weight),
        static_cast<const ps::bf16_t*>(input), output,
        kernel::Bf16GemmOutputDType::BF16, kRows, c.n, c.k, c.k, c.n, stream);
}

void fill_args(const GemmCase& c, const void* weight, const void* input,
               void* output, kernel::Bf16GemmExactRowsArgs& a) {
    a = kernel::Bf16GemmExactRowsArgs{};
    a.weight = weight;
    a.input = input;
    a.output = output;
    a.output_dtype = static_cast<uint32_t>(kernel::Bf16GemmOutputDType::BF16);
    a.n = c.n;
    a.k = c.k;
    a.input_row_stride = c.k;
    a.output_row_stride = c.n;
}

void fill_invocation(const GemmCase& c, const void* weight, const void* input,
                     void* output, mcu::McuBf16ExactRowsInvocation& v) {
    v = mcu::McuBf16ExactRowsInvocation{};
    v.weight = reinterpret_cast<uint64_t>(weight);
    v.input = reinterpret_cast<uint64_t>(input);
    v.output = reinterpret_cast<uint64_t>(output);
    v.output_dtype = static_cast<uint32_t>(kernel::Bf16GemmOutputDType::BF16);
    v.n = c.n;
    v.k = c.k;
    v.input_row_stride = c.k;
    v.output_row_stride = c.n;
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

    constexpr uint32_t kMaxN = 9216u;
    constexpr uint32_t kMaxK = 9216u;
    const size_t weight_bytes = static_cast<size_t>(kMaxN) * kMaxK * 2u;
    const size_t input_bytes = static_cast<size_t>(kMaxK) * 2u;

    void* weight = nullptr;
    void* input = nullptr;
    if (hipMalloc(&weight, weight_bytes) != hipSuccess ||
        hipMalloc(&input, input_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    std::vector<uint16_t> h_weight(static_cast<size_t>(kMaxN) * kMaxK);
    for (size_t i = 0; i < h_weight.size(); ++i)
        h_weight[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1500u));
    std::vector<uint16_t> h_input(kMaxK);
    for (size_t i = 0; i < h_input.size(); ++i)
        h_input[i] = static_cast<uint16_t>(0x3A00u + ((i * 40503u) % 1500u));
    (void)hipMemcpy(weight, h_weight.data(), weight_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(input, h_input.data(), input_bytes, hipMemcpyHostToDevice);

    std::vector<void*> out_ref(kCaseCount, nullptr);
    std::vector<void*> out_cand(kCaseCount, nullptr);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const size_t bytes = static_cast<size_t>(kCases[i].n) * 2u + kGuardBytes;
        if (hipMalloc(&out_ref[i], bytes) != hipSuccess ||
            hipMalloc(&out_cand[i], bytes) != hipSuccess) {
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
        queue, PHASESHIFT_GPU_MCU_BF16_HSACO, kernel::kBf16ExactRowsSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: bf16 code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf("bf16 exact_rows hsaco: kernel_object=%p kernarg=%u group=%u "
                "sizeof(args)=%zu hidden_fits=%d\n",
                reinterpret_cast<void*>(meta.kernel_object),
                meta.kernarg_segment_size, meta.static_group_segment_size,
                sizeof(kernel::Bf16GemmExactRowsArgs),
                static_cast<int>(mcu::aql_hidden_args_fits(
                    sizeof(kernel::Bf16GemmExactRowsArgs),
                    meta.kernarg_segment_size)));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::Bf16GemmExactRowsArgs),
          "kernarg segment covers Bf16GemmExactRowsArgs");
    check(meta.kernarg_segment_size <=
              mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size) - 16u,
          "kernarg segment leaves room for launch metadata");

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

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const GemmCase& c = kCases[i];
        const size_t bytes = static_cast<size_t>(c.n) * 2u;

        (void)hipMemset(out_ref[i], 0xAB, bytes + kGuardBytes);
        (void)hipMemset(out_cand[i], 0xAB, bytes + kGuardBytes);
        check(host_launch(c, weight, input, out_ref[i], stream) == hipSuccess,
              "host gemm launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        kernel::Bf16GemmExactRowsArgs args{};
        fill_args(c, weight, input, out_cand[i], args);
        std::memcpy(slot, &args, sizeof(args));

        const uint32_t grid_x = kernel::bf16_exact_rows_grid_x(c.n);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = grid_x * kernel::kBf16GemmExactRowsThreads;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x =
            static_cast<uint16_t>(kernel::kBf16GemmExactRowsThreads);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::Bf16GemmExactRowsArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid_x,
                                       1u, 1u, kernel::kBf16GemmExactRowsThreads);

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
            std::printf("  n=%u k=%u aql diff count=%zu first=%zu\n", c.n, c.k,
                        diffs, first);
        check(cand_guard == guards[i], "aql output sentinel intact");
        std::printf("  n=%u k=%u grid_x=%u ok=%d\n", c.n, c.k, grid_x,
                    static_cast<int>(diffs == 0u));
    }

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");

    uint32_t needed_segment = meta.kernarg_segment_size;

    Rig rig;
    if (!rig.open(0, kCaseCount + 3u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  needed_segment)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_BF16_HSACO,
                             kernel::kBf16ExactRowsSymbol, mcu_meta))
        return 1;

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_meta,
            mcu::kMcuKernargRecipeBf16ExactRows,
            kernel::kBf16GemmExactRowsThreads,
            kernel::bf16_exact_rows_grid_x(kCases[i].n), mcu::kMcuAgentAqlPolicy));
    }
    variants.push_back(mcu::make_probe_variant(static_cast<uint16_t>(kCaseCount),
                                               rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeBf16ExactRows;
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

    std::vector<mcu::McuBf16ExactRowsInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const size_t bytes = static_cast<size_t>(kCases[i].n) * 2u;
        (void)hipMemset(out_cand[i], 0xAB, bytes + kGuardBytes);
        fill_invocation(kCases[i], weight, input, out_cand[i], invocations[i]);
    }
    if (!rig.set_bf16_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x42463131u;
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
        const GemmCase& c = kCases[i];
        const size_t bytes = static_cast<size_t>(c.n) * 2u;
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
              "mcu bf16 gemm matches the host launch");
        if (diffs)
            std::printf("  n=%u k=%u mcu diff count=%zu first=%zu\n", c.n, c.k,
                        diffs, first);
        check(mcu_guard == guards[i], "mcu output sentinel intact");
    }

    check(rig.close(), "rig teardown");

    (void)hipStreamDestroy(stream);
    (void)hipFree(weight);
    (void)hipFree(input);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(out_ref[i]);
        (void)hipFree(out_cand[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_bf16_exact_rows: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_bf16_exact_rows: PASS\n");
    return 0;
}
