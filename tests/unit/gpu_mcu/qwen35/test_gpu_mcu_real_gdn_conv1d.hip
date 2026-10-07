#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h>
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

constexpr uint32_t kConvDim = 2560u;
constexpr uint32_t kHistory = 3u;
constexpr uint32_t kNumGdnStates = 1u;
constexpr uint32_t kMaxSequences = 2u;
constexpr uint32_t kStateIndex = 0u;
constexpr uint32_t kGuardBytes = 64u;

struct ConvCase {
    uint32_t rows;
};

const ConvCase kCases[] = {{1u}, {64u}};
constexpr uint32_t kCaseCount =
    static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

constexpr uint32_t hist_stride() { return (kConvDim + 7u) & ~7u; }
constexpr uint64_t layer_stride() { return (uint64_t)kHistory * hist_stride(); }
constexpr uint64_t slot_stride() {
    return (uint64_t)kNumGdnStates * layer_stride();
}
constexpr uint64_t state_total() { return (uint64_t)kMaxSequences * slot_stride(); }

kernel::GdnConv1dArgs make_args(const void* input, void* output,
                                const void* weight, void* state,
                                const void* requests, uint32_t rows) {
    kernel::GdnConv1dArgs a;
    a.input = static_cast<const ps::bf16_t*>(input);
    a.output = static_cast<float*>(output);
    a.conv_weight = static_cast<const ps::bf16_t*>(weight);
    a.conv_state = static_cast<ps::bf16_t*>(state);
    a.requests = static_cast<const ::ps::runtime::DeviceRequestDescriptor*>(requests);
    a.num_requests = 1u;
    a.actual_rows = rows;
    a.max_request_rows = rows;
    a.input_row_stride = kConvDim;
    a.output_row_stride = kConvDim;
    a.conv_dim = kConvDim;
    a.conv_history = kHistory;
    a.state_index = kStateIndex;
    a.max_sequences = kMaxSequences;
    a.conv_slot_stride = slot_stride();
    a.conv_layer_stride = layer_stride();
    a.conv_history_stride = hist_stride();
    return a;
}

kernel::GdnConv1dAqlArgs to_aql(const kernel::GdnConv1dArgs& a) {
    kernel::GdnConv1dAqlArgs v{};
    v.input = a.input;
    v.output = a.output;
    v.conv_weight = a.conv_weight;
    v.conv_state = a.conv_state;
    v.requests = a.requests;
    v.actual_rows = a.actual_rows;
    v.input_row_stride = a.input_row_stride;
    v.output_row_stride = a.output_row_stride;
    v.conv_dim = a.conv_dim;
    v.state_index = a.state_index;
    v.max_sequences = a.max_sequences;
    v.conv_slot_stride = a.conv_slot_stride;
    v.conv_layer_stride = a.conv_layer_stride;
    v.conv_history_stride = a.conv_history_stride;
    v.row_tile = kernel::gdn_conv1d_resolved_row_tile(kernel::gdn_conv1d_row_tile(),
                                                      a.actual_rows);
    v.conv_history_store = a.conv_history_store;
    v.conv_history_row_stride = a.conv_history_row_stride;
    v.capture_rows = a.capture_rows;
    return v;
}

void fill_invocation(const kernel::GdnConv1dAqlArgs& v,
                     mcu::McuGdnConv1dInvocation& out) {
    out.input = reinterpret_cast<uint64_t>(v.input);
    out.output = reinterpret_cast<uint64_t>(v.output);
    out.conv_weight = reinterpret_cast<uint64_t>(v.conv_weight);
    out.conv_state = reinterpret_cast<uint64_t>(v.conv_state);
    out.requests = reinterpret_cast<uint64_t>(v.requests);
    out.actual_rows = v.actual_rows;
    out.input_row_stride = v.input_row_stride;
    out.output_row_stride = v.output_row_stride;
    out.conv_dim = v.conv_dim;
    out.state_index = v.state_index;
    out.max_sequences = v.max_sequences;
    out.conv_slot_stride = v.conv_slot_stride;
    out.conv_layer_stride = v.conv_layer_stride;
    out.conv_history_stride = v.conv_history_stride;
    out.row_tile = v.row_tile;
    out.conv_history_store = reinterpret_cast<uint64_t>(v.conv_history_store);
    out.conv_history_row_stride = v.conv_history_row_stride;
    out.capture_rows = v.capture_rows;
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

    constexpr uint32_t kMaxRows = 64u;
    const size_t in_bytes = (size_t)kMaxRows * kConvDim * 2u;
    const size_t weight_bytes = (size_t)kConvDim * 4u * 2u;
    const size_t out_bytes = (size_t)kMaxRows * kConvDim * 4u + kGuardBytes;
    const size_t state_bytes = state_total() * 2u + kGuardBytes;
    const size_t req_bytes = sizeof(::ps::runtime::DeviceRequestDescriptor);

    void* input = nullptr;
    void* weight = nullptr;
    std::vector<void*> requests(kCaseCount, nullptr);
    void* out_ref = nullptr;
    void* out_aql = nullptr;
    void* st_ref = nullptr;
    void* st_aql = nullptr;
    std::vector<void*> out_mcu(kCaseCount, nullptr);
    std::vector<void*> st_mcu(kCaseCount, nullptr);
    if (hipMalloc(&input, in_bytes) != hipSuccess ||
        hipMalloc(&weight, weight_bytes) != hipSuccess ||
        hipMalloc(&out_ref, out_bytes) != hipSuccess ||
        hipMalloc(&out_aql, out_bytes) != hipSuccess ||
        hipMalloc(&st_ref, state_bytes) != hipSuccess ||
        hipMalloc(&st_aql, state_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (hipMalloc(&out_mcu[i], out_bytes) != hipSuccess ||
            hipMalloc(&st_mcu[i], state_bytes) != hipSuccess ||
            hipMalloc(&requests[i], req_bytes) != hipSuccess) {
            std::printf("FAIL: mcu buffer alloc\n");
            return 1;
        }
    }

    std::vector<uint16_t> h_in((size_t)kMaxRows * kConvDim);
    std::vector<uint16_t> h_w((size_t)kConvDim * 4u);
    for (size_t i = 0; i < h_in.size(); ++i)
        h_in[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1500u));
    for (size_t i = 0; i < h_w.size(); ++i)
        h_w[i] = static_cast<uint16_t>(0x3A00u + ((i * 40503u) % 1500u));
    if (hipMemcpy(input, h_in.data(), in_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(weight, h_w.data(), weight_bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
        std::printf("FAIL: input upload\n");
        return 1;
    }
    const uint32_t kSeedState = 0x3C00u;

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    std::vector<std::vector<uint32_t>> refs(kCaseCount);
    std::vector<std::vector<uint32_t>> guards(kCaseCount);
    std::vector<std::vector<uint32_t>> state_refs(kCaseCount);
    std::vector<kernel::GdnConv1dAqlArgs> aql_args(kCaseCount);

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_GDN_CONV1D_HSACO, kernel::kGdnConv1dSymbol);
    if (!code_res.ok()) {
        std::printf("FAIL: conv1d code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf(
        "conv1d hsaco: kernarg=%u group=%u sizeof(args)=%zu hidden_fits=%d "
        "row_tile=%u\n",
        meta.kernarg_segment_size, meta.static_group_segment_size,
        sizeof(kernel::GdnConv1dAqlArgs),
        static_cast<int>(mcu::aql_hidden_args_fits(
            sizeof(kernel::GdnConv1dAqlArgs), meta.kernarg_segment_size)),
        kernel::gdn_conv1d_row_tile());
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::GdnConv1dAqlArgs),
          "kernarg segment covers GdnConv1dAqlArgs");

    const uint32_t slot_stride_bytes =
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size);
    auto region_res = queue.allocate_kernarg(slot_stride_bytes, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const ConvCase& c = kCases[i];
        const size_t bytes = (size_t)c.rows * kConvDim * 4u;

        ::ps::runtime::DeviceRequestDescriptor req;
        std::memset(&req, 0, sizeof(req));
        req.request_handle.slot = 0u;
        req.row_begin = 0u;
        req.row_count = c.rows;
        if (hipMemcpy(requests[i], &req, req_bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
            std::printf("FAIL: request upload\n");
            return 1;
        }

        for (void* p : {st_ref, st_aql})
            (void)hipMemset(p, static_cast<int>(kSeedState & 0xFF), state_bytes);
        (void)hipMemset(out_ref, 0xAB, out_bytes);
        (void)hipMemset(out_aql, 0xAB, out_bytes);

        kernel::GdnConv1dArgs a = make_args(input, out_ref, weight, st_ref,
                                            requests[i], c.rows);
        check(kernel::launch_gdn_conv1d_bf16_f32(a, stream) == hipSuccess,
              "host conv1d launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        aql_args[i] = to_aql(make_args(input, out_aql, weight, st_aql,
                                       requests[i], c.rows));

        const size_t slot_off = (size_t)i * region.slot_stride;
        unsigned char* slot = region.base + slot_off;
        std::memset(slot, 0, region.slot_stride);
        std::memcpy(slot, &aql_args[i], sizeof(kernel::GdnConv1dAqlArgs));

        const uint32_t blocks_x = kernel::gdn_conv1d_blocks_x(kConvDim);
        const uint32_t row_tile = kernel::gdn_conv1d_row_tile();
        const uint32_t tiles = kernel::gdn_conv1d_tiles(row_tile, c.rows);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = blocks_x * kernel::kGdnConv1dThreads;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = tiles;
        desc.workgroup_size_x =
            static_cast<uint16_t>(kernel::kGdnConv1dThreads);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::GdnConv1dAqlArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, blocks_x,
                                       1u, tiles, kernel::kGdnConv1dThreads);

        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        {
            const unsigned char* pb =
                reinterpret_cast<const unsigned char*>(tmpl.bytes);
            uint32_t gx = 0, gy = 0, gz = 0;
            std::memcpy(&gx, pb + 12, 4);
            std::memcpy(&gy, pb + 16, 4);
            std::memcpy(&gz, pb + 20, 4);
            const uint32_t dw0 = tmpl.publish_dw0;
            std::printf("  rows=%u grid=(%u,%u,%u) dims_field=%u\n", c.rows, gx,
                        gy, gz, (dw0 >> 16) & 3u);
            check(gx == blocks_x * kernel::kGdnConv1dThreads, "packet grid_x");
            check(gy == 1u, "packet grid_y");
            check(gz == tiles, "packet grid_z");
            check(((dw0 >> 16) & 3u) == (gz > 1u ? 3u : 1u), "packet dimensions field");
        }
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_drained(queue, queue.host_queued_index(), 20000),
              "aql queue drained");

        const size_t words = bytes / 4u;
        refs[i].assign(words, 0u);
        guards[i].assign(kGuardBytes / 4u, 0u);
        state_refs[i].assign(state_total(), 0u);
        std::vector<uint32_t> cand(words, 0u);
        std::vector<uint32_t> cand_guard(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> st_aql_h(state_total(), 0u);
        (void)hipMemcpy(refs[i].data(), out_ref, bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand.data(), out_aql, bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(guards[i].data(),
                        static_cast<unsigned char*>(out_ref) + bytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand_guard.data(),
                        static_cast<unsigned char*>(out_aql) + bytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(state_refs[i].data(), st_ref, state_total() * 2u,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(st_aql_h.data(), st_aql, state_total() * 2u,
                        hipMemcpyDeviceToHost);

        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), cand.data(), bytes, &first,
                                 &diffs),
              "host produced aql output is bit exact");
        if (diffs)
            std::printf("  rows=%u aql out diff=%zu first=%zu\n", c.rows, diffs,
                        first);
        check(cand_guard == guards[i], "aql output sentinel intact");
        size_t sfirst = 0;
        size_t sdiffs = 0;
        check(chain::bytes_equal(state_refs[i].data(), st_aql_h.data(),
                                 state_total() * 2u, &sfirst, &sdiffs),
              "host produced aql state is bit exact");
        if (sdiffs)
            std::printf("  rows=%u aql state diff=%zu first=%zu\n", c.rows, sdiffs,
                        sfirst);
        std::printf("  rows=%u grid_z=%u out_ok=%d state_ok=%d\n", c.rows, tiles,
                    static_cast<int>(diffs == 0u), static_cast<int>(sdiffs == 0u));
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
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_GDN_CONV1D_HSACO,
                             kernel::kGdnConv1dSymbol, mcu_meta))
        return 1;

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const ConvCase& c = kCases[i];
        const uint32_t blocks_x = kernel::gdn_conv1d_blocks_x(kConvDim);
        const uint32_t tiles =
            kernel::gdn_conv1d_tiles(kernel::gdn_conv1d_row_tile(), c.rows);
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_meta, mcu::kMcuKernargRecipeGdnConv1d,
            kernel::kGdnConv1dThreads, blocks_x, mcu::kMcuAgentAqlPolicy,
            mcu::AqlHiddenArgsPolicy::IfFits, 1u, tiles));
    }
    variants.push_back(mcu::make_probe_variant(static_cast<uint16_t>(kCaseCount),
                                               rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeGdnConv1d;
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

    std::vector<mcu::McuGdnConv1dInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const ConvCase& c = kCases[i];
        (void)hipMemset(out_mcu[i], 0xAB, out_bytes);
        (void)hipMemset(st_mcu[i], static_cast<int>(kSeedState & 0xFF),
                        state_bytes);
        kernel::GdnConv1dArgs a = make_args(input, out_mcu[i], weight, st_mcu[i],
                                            requests[i], c.rows);
        fill_invocation(to_aql(a), invocations[i]);
    }
    if (!rig.set_gdn_conv1d_invocations(invocations.data(), kCaseCount)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x434f4e56u;
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
        const ConvCase& c = kCases[i];
        const size_t bytes = (size_t)c.rows * kConvDim * 4u;
        const size_t words = bytes / 4u;
        std::vector<uint32_t> mcu_out(words, 0u);
        std::vector<uint32_t> mcu_guard(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> st_mcu_h(state_total(), 0u);
        rig.device_to_host(mcu_out.data(), out_mcu[i], bytes);
        rig.device_to_host(mcu_guard.data(),
                           static_cast<unsigned char*>(out_mcu[i]) + bytes,
                           kGuardBytes);
        rig.device_to_host(st_mcu_h.data(), st_mcu[i], state_total() * 2u);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(refs[i].data(), mcu_out.data(), bytes, &first,
                                 &diffs),
              "mcu conv1d output matches the host launch");
        if (diffs)
            std::printf("  rows=%u mcu out diff=%zu first=%zu\n", c.rows, diffs,
                        first);
        check(mcu_guard == guards[i], "mcu output sentinel intact");
        size_t sfirst = 0;
        size_t sdiffs = 0;
        check(chain::bytes_equal(state_refs[i].data(), st_mcu_h.data(),
                                 state_total() * 2u, &sfirst, &sdiffs),
              "mcu conv1d state matches the host launch");
        if (sdiffs)
            std::printf("  rows=%u mcu state diff=%zu first=%zu\n", c.rows, sdiffs,
                        sfirst);
    }

    check(rig.close(), "rig teardown");

    (void)hipStreamDestroy(stream);
    for (void* p : {input, weight, out_ref, out_aql, st_ref, st_aql})
        (void)hipFree(p);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(out_mcu[i]);
        (void)hipFree(st_mcu[i]);
        (void)hipFree(requests[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_gdn_conv1d: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_gdn_conv1d: PASS\n");
    return 0;
}
