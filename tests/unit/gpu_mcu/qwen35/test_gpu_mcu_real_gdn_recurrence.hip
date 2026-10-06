#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>
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

constexpr uint32_t kKeyHeads = 16u;
constexpr uint32_t kHeadK = 128u;
constexpr uint32_t kHeadV = 128u;
constexpr uint32_t kNReq = 1u;
constexpr uint32_t kRows = 8u;
constexpr uint32_t kGuardBytes = 64u;
constexpr uint32_t kMaxNumVHeads = 48u;

struct RecCase {
    const char* name;
    uint32_t num_v_heads;
    bool decode1;
    bool decode_rows;
    uint32_t rows;
};

const RecCase kCases[] = {
    {"decode1", 48u, true, false, kRows},
    {"wmma-model-shape", 32u, false, false, kRows},
    {"decode-rows-r8", 48u, false, true, 8u},
};
constexpr uint32_t kCaseCount =
    static_cast<uint32_t>(sizeof(kCases) / sizeof(kCases[0]));

constexpr uint32_t qk() { return kKeyHeads * kHeadK; }
constexpr uint32_t vdim(uint32_t nvh) { return nvh * kHeadV; }
constexpr uint64_t slot_stride(uint32_t nvh) {
    return (uint64_t)nvh * kHeadK * kHeadV;
}

kernel::GdnRecurrenceArgs make_args(const RecCase& c, const void* q,
                                    const void* k, const void* v,
                                    const void* ga, const void* gb,
                                    const void* dt_bias, const void* a_log,
                                    void* state, void* out,
                                    const void* requests) {
    kernel::GdnRecurrenceArgs a;
    a.q = static_cast<const float*>(q);
    a.k = static_cast<const float*>(k);
    a.v = static_cast<const ps::bf16_t*>(v);
    a.a = static_cast<const ps::bf16_t*>(ga);
    a.b = static_cast<const ps::bf16_t*>(gb);
    a.dt_bias = static_cast<const ps::bf16_t*>(dt_bias);
    a.a_log = static_cast<const ps::bf16_t*>(a_log);
    a.recurrent_state = static_cast<float*>(state);
    a.output = static_cast<float*>(out);
    a.requests =
        static_cast<const ::ps::runtime::DeviceRequestDescriptor*>(requests);
    a.num_requests = kNReq;
    a.actual_rows = c.rows;
    a.q_row_stride = qk();
    a.k_row_stride = qk();
    a.v_row_stride = vdim(c.num_v_heads);
    a.a_row_stride = c.num_v_heads;
    a.b_row_stride = c.num_v_heads;
    a.output_row_stride = vdim(c.num_v_heads);
    a.state_index = 0u;
    a.max_sequences = kNReq;
    a.num_gdn_states = 1u;
    a.key_heads = kKeyHeads;
    a.num_v_heads = c.num_v_heads;
    a.head_k = kHeadK;
    a.head_v = kHeadV;
    a.recurrent_slot_stride = slot_stride(c.num_v_heads);
    a.recurrent_layer_stride = slot_stride(c.num_v_heads);
    return a;
}

void fill_invocation(const kernel::GdnRecurrenceArgs& a,
                     mcu::McuGdnRecurrenceInvocation& out) {
    std::memcpy(&out, &a, sizeof(out));
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

    const size_t max_vdim = vdim(kMaxNumVHeads);
    const size_t max_slot = slot_stride(kMaxNumVHeads);
    const size_t qk_bytes = (size_t)kNReq * kRows * qk() * 4u;
    const size_t v_bytes = (size_t)kNReq * kRows * max_vdim * 2u;
    const size_t gate_bytes = (size_t)kNReq * kRows * kMaxNumVHeads * 2u;
    const size_t bias_bytes = (size_t)kMaxNumVHeads * 2u;
    const size_t out_bytes =
        (size_t)kNReq * kRows * max_vdim * 4u + kGuardBytes;
    const size_t state_bytes = (size_t)kNReq * max_slot * 4u;

    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* ga = nullptr;
    void* gb = nullptr;
    void* dt_bias = nullptr;
    void* a_log = nullptr;
    void* requests = nullptr;
    void* out_ref = nullptr;
    void* out_aql = nullptr;
    void* st_ref = nullptr;
    void* st_aql = nullptr;
    std::vector<void*> out_mcu(kCaseCount, nullptr);
    std::vector<void*> st_mcu(kCaseCount, nullptr);
    if (hipMalloc(&q, qk_bytes) != hipSuccess ||
        hipMalloc(&k, qk_bytes) != hipSuccess ||
        hipMalloc(&v, v_bytes) != hipSuccess ||
        hipMalloc(&ga, gate_bytes) != hipSuccess ||
        hipMalloc(&gb, gate_bytes) != hipSuccess ||
        hipMalloc(&dt_bias, bias_bytes) != hipSuccess ||
        hipMalloc(&a_log, bias_bytes) != hipSuccess ||
        hipMalloc(&requests, sizeof(::ps::runtime::DeviceRequestDescriptor)) !=
            hipSuccess ||
        hipMalloc(&out_ref, out_bytes) != hipSuccess ||
        hipMalloc(&out_aql, out_bytes) != hipSuccess ||
        hipMalloc(&st_ref, state_bytes) != hipSuccess ||
        hipMalloc(&st_aql, state_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (hipMalloc(&out_mcu[i], out_bytes) != hipSuccess ||
            hipMalloc(&st_mcu[i], state_bytes) != hipSuccess) {
            std::printf("FAIL: mcu buffer alloc\n");
            return 1;
        }
    }

    std::vector<float> h_q((size_t)kNReq * kRows * qk()),
        h_k((size_t)kNReq * kRows * qk());
    std::vector<uint16_t> h_v((size_t)kNReq * kRows * max_vdim);
    std::vector<uint16_t> h_ga((size_t)kNReq * kRows * kMaxNumVHeads);
    std::vector<uint16_t> h_gb((size_t)kNReq * kRows * kMaxNumVHeads);
    std::vector<uint16_t> h_dt(kMaxNumVHeads), h_al(kMaxNumVHeads);
    for (size_t i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<float>(static_cast<int>((i * 37u) % 201u) - 100) / 64.0f;
    for (size_t i = 0; i < h_k.size(); ++i)
        h_k[i] = static_cast<float>(static_cast<int>((i * 53u) % 199u) - 99) / 64.0f;
    for (size_t i = 0; i < h_v.size(); ++i)
        h_v[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1200u));
    for (size_t i = 0; i < h_ga.size(); ++i)
        h_ga[i] = static_cast<uint16_t>(0x3800u + ((i * 40503u) % 1600u));
    for (size_t i = 0; i < h_gb.size(); ++i)
        h_gb[i] = static_cast<uint16_t>(0x3800u + ((i * 1103515245u) % 1600u));
    for (uint32_t i = 0; i < kMaxNumVHeads; ++i) {
        h_dt[i] = static_cast<uint16_t>(0x3800u + ((i * 97u) % 1024u));
        h_al[i] = static_cast<uint16_t>(0x3900u + ((i * 131u) % 1024u));
    }
    if (hipMemcpy(q, h_q.data(), qk_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(k, h_k.data(), qk_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(v, h_v.data(), v_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(ga, h_ga.data(), gate_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(gb, h_gb.data(), gate_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(dt_bias, h_dt.data(), bias_bytes, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(a_log, h_al.data(), bias_bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
        std::printf("FAIL: input upload\n");
        return 1;
    }

    ::ps::runtime::DeviceRequestDescriptor req;
    std::memset(&req, 0, sizeof(req));
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = kRows;
    if (hipMemcpy(requests, &req, sizeof(req), hipMemcpyHostToDevice) !=
        hipSuccess) {
        std::printf("FAIL: request upload\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    const bool lossy = kernel::gdn_recurrence_lossy_enabled();
    auto case_symbol = [&](const RecCase& c) -> const char* {
        if (c.decode1) return kernel::kGdnRecurrenceWmmaDecode1Symbol;
        if (c.decode_rows) {
            if (c.rows == 2u) return kernel::kGdnRecurrenceDecodeRowsExactR2Symbol;
            if (c.rows == 4u) return kernel::kGdnRecurrenceDecodeRowsExactR4Symbol;
            return kernel::kGdnRecurrenceDecodeRowsExactR8Symbol;
        }
        return lossy ? kernel::kGdnRecurrenceWmmaLossySymbol
                     : kernel::kGdnRecurrenceWmmaExactSymbol;
    };

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    std::vector<std::unique_ptr<mcu::GpuMcuAqlCodeObject>> codes;
    std::vector<mcu::GpuAqlKernelMetadata> metas(kCaseCount);
    uint32_t needed_segment = 0u;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        auto res = mcu::GpuMcuAqlCodeObject::load(
            queue, PHASESHIFT_GPU_MCU_GDN_RECURRENCE_HSACO, case_symbol(kCases[i]));
        if (!res.ok()) {
            std::printf("FAIL: code object %s: %s\n", kCases[i].name,
                        res.status().message().c_str());
            return 1;
        }
        codes.push_back(std::unique_ptr<mcu::GpuMcuAqlCodeObject>(
            new mcu::GpuMcuAqlCodeObject(res.release())));
        metas[i] = codes.back()->metadata();
        if (metas[i].kernarg_segment_size > needed_segment)
            needed_segment = metas[i].kernarg_segment_size;
        std::printf("%s: symbol=%s kernarg=%u group=%u sizeof(args)=%zu\n",
                    kCases[i].name, case_symbol(kCases[i]),
                    metas[i].kernarg_segment_size,
                    metas[i].static_group_segment_size,
                    sizeof(kernel::GdnRecurrenceArgs));
        check(metas[i].kernel_object != 0, "kernel_object resolved");
        check(metas[i].kernarg_segment_size == sizeof(kernel::GdnRecurrenceArgs),
              "compiler arg area matches GdnRecurrenceArgs layout");
    }

    const uint32_t slot_bytes = mcu::aql_kernarg_slot_stride(needed_segment);
    auto region_res = queue.allocate_kernarg(slot_bytes, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    std::vector<std::vector<uint32_t>> refs(kCaseCount);
    std::vector<std::vector<uint32_t>> guards(kCaseCount);
    std::vector<std::vector<uint32_t>> state_refs(kCaseCount);
    std::vector<uint32_t> ref_guard_h(kGuardBytes / 4u);

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RecCase& c = kCases[i];
        const size_t out_elems =
            (size_t)kNReq * c.rows * vdim(c.num_v_heads);
        const size_t state_elems = (size_t)kNReq * slot_stride(c.num_v_heads);
        for (void* p : {st_ref, st_aql})
            (void)hipMemset(p, 0x5A, state_elems * 4u);
        for (void* p : {out_ref, out_aql})
            (void)hipMemset(p, 0xAB, out_bytes);

        kernel::GdnRecurrenceArgs href =
            make_args(c, q, k, v, ga, gb, dt_bias, a_log, st_ref, out_ref, requests);
        const hipError_t herr =
            c.decode_rows
                ? kernel::launch_gdn_recurrence_f32_wmma_decode_rows_exact(href,
                                                                          stream)
                : (c.decode1
                       ? kernel::launch_gdn_recurrence_f32_wmma_decode1(href,
                                                                      stream)
                       : kernel::launch_gdn_recurrence_f32_wmma(href, stream));
        check(herr == hipSuccess, "host recurrence launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        kernel::GdnRecurrenceArgs hargs =
            make_args(c, q, k, v, ga, gb, dt_bias, a_log, st_aql, out_aql, requests);
        unsigned char* slot = region.base + (size_t)i * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        std::memcpy(slot, &hargs, sizeof(hargs));

        const uint32_t grid_x =
            kernel::gdn_recurrence_wmma_grid_x(kNReq, c.num_v_heads, kHeadV);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = metas[i].kernel_object;
        desc.global_work_items_x = grid_x * kernel::kGdnRecurrenceWmmaDecode1Threads;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x =
            static_cast<uint16_t>(kernel::kGdnRecurrenceWmmaDecode1Threads);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = metas[i].private_segment_size;
        desc.group_segment_size = metas[i].static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::GdnRecurrenceArgs),
                                   metas[i].kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, metas[i].kernarg_segment_size, grid_x,
                                       1u, 1u,
                                       kernel::kGdnRecurrenceWmmaDecode1Threads);

        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        {
            const unsigned char* pb =
                reinterpret_cast<const unsigned char*>(tmpl.bytes);
            uint32_t gx = 0, gy = 0, gz = 0;
            std::memcpy(&gx, pb + 12, 4);
            std::memcpy(&gy, pb + 16, 4);
            std::memcpy(&gz, pb + 20, 4);
            std::printf("  %s: grid=(%u,%u,%u) dims_field=%u hidden_fits=%d\n",
                        c.name, gx, gy, gz, (tmpl.publish_dw0 >> 16) & 3u,
                        static_cast<int>(mcu::aql_hidden_args_fits(
                            sizeof(kernel::GdnRecurrenceArgs),
                            metas[i].kernarg_segment_size)));
            check(gx == grid_x * kernel::kGdnRecurrenceWmmaDecode1Threads,
                  "packet grid_x");
            check(gy == 1u && gz == 1u, "packet is 1d");
            check(((tmpl.publish_dw0 >> 16) & 3u) == 1u,
                  "packet dimensions field = 1");
        }
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_drained(queue, queue.host_queued_index(), 40000),
              "aql queue drained");

        refs[i].assign(out_elems, 0u);
        guards[i].assign(kGuardBytes / 4u, 0u);
        state_refs[i].assign(state_elems, 0u);
        std::vector<uint32_t> cand(out_elems, 0u);
        std::vector<uint32_t> cand_guard(kGuardBytes / 4u, 0u);
        std::vector<uint32_t> st_aql_h(state_elems, 0u);
        (void)hipMemcpy(refs[i].data(), out_ref, out_elems * 4u,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand.data(), out_aql, out_elems * 4u,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(guards[i].data(),
                        static_cast<unsigned char*>(out_ref) + out_elems * 4u,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand_guard.data(),
                        static_cast<unsigned char*>(out_aql) + out_elems * 4u,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(state_refs[i].data(), st_ref, state_elems * 4u,
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(st_aql_h.data(), st_aql, state_elems * 4u,
                        hipMemcpyDeviceToHost);

        size_t first = 0, diffs = 0;
        check(chain::bytes_equal(refs[i].data(), cand.data(), out_elems * 4u,
                                 &first, &diffs),
              "host produced aql output is bit exact");
        if (diffs)
            std::printf("  %s aql out diff=%zu first=%zu\n", c.name, diffs, first);
        check(cand_guard == guards[i], "aql output sentinel intact");
        size_t sfirst = 0, sdiffs = 0;
        check(chain::bytes_equal(state_refs[i].data(), st_aql_h.data(),
                                 state_elems * 4u, &sfirst, &sdiffs),
              "host produced aql state is bit exact");
        if (sdiffs)
            std::printf("  %s aql state diff=%zu first=%zu\n", c.name, sdiffs, sfirst);
    }

    for (auto& code : codes) (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");

    Rig rig;
    if (!rig.open(0, kCaseCount + 3u, kCaseCount + 1u, 8u, 0u, false,
                  mcu::kAqlDefaultQueueSize, mcu::kMcuMaxDebugRecords,
                  needed_segment)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    rig.log_enabled = true;
    std::vector<mcu::GpuAqlKernelMetadata> mcu_metas(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_GDN_RECURRENCE_HSACO,
                                 case_symbol(kCases[i]), mcu_metas[i]))
            return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants;
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const uint32_t gx = kernel::gdn_recurrence_wmma_grid_x(
            kNReq, kCases[i].num_v_heads, kHeadV);
        variants.push_back(mcu::make_kernel_variant(
            static_cast<uint16_t>(i), mcu_metas[i],
            mcu::kMcuKernargRecipeGdnRecurrence,
            kernel::kGdnRecurrenceWmmaDecode1Threads, gx,
            mcu::kMcuAgentAqlPolicy));
    }
    variants.push_back(mcu::make_probe_variant(static_cast<uint16_t>(kCaseCount),
                                               rig.meta, 64u, 1u,
                                               mcu::kMcuSystemAqlPolicy));

    std::vector<mcu::McuPlanNode> nodes(kCaseCount + 3u);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        nodes[i].variant_id = static_cast<uint16_t>(i);
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeGdnRecurrence;
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
    std::vector<mcu::McuGdnRecurrenceInvocation> invocations(kCaseCount);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RecCase& c = kCases[i];
        const size_t state_elems = (size_t)kNReq * slot_stride(c.num_v_heads);
        (void)hipMemset(out_mcu[i], 0xAB, out_bytes);
        (void)hipMemset(st_mcu[i], 0x5A, state_elems * 4u);
        kernel::GdnRecurrenceArgs margs =
            make_args(c, q, k, v, ga, gb, dt_bias, a_log, st_mcu[i], out_mcu[i],
                      requests);
        fill_invocation(margs, invocations[i]);
    }
    if (!rig.set_gdn_recurrence_invocations(invocations.data(), kCaseCount))
        return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x52454332u;
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
    if (!rig.fsm.wait_plans(1u, 60000).ok()) {
        const mcu::GpuMcuFsmState* fstate = rig.fsm.host_state();
        std::printf("FAIL: mcu plan completed fault=%u pc=%u variant=%u "
                    "gen=%llu dispatches=%llu\n",
                    static_cast<uint32_t>(rig.fsm.fault_code()),
                    rig.fsm.fault_pc(),
                    fstate != nullptr ? fstate->fault_variant : 0u,
                    fstate != nullptr
                        ? static_cast<unsigned long long>(
                              fstate->fault_generation)
                        : 0ull,
                    static_cast<unsigned long long>(rig.fsm.dispatches()));
        ++g_failed;
    }
    check(rig.drain_queue(60000), "queue drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == static_cast<uint64_t>(kCaseCount) + 1u,
          "dispatch count");

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const RecCase& c = kCases[i];
        const size_t out_elems =
            (size_t)kNReq * c.rows * vdim(c.num_v_heads);
        const size_t state_elems = (size_t)kNReq * slot_stride(c.num_v_heads);
        std::vector<uint32_t> mcu_out(out_elems), mcu_guard(kGuardBytes / 4u);
        std::vector<uint32_t> st_mcu_h(state_elems);
        rig.device_to_host(mcu_out.data(), out_mcu[i], out_elems * 4u);
        rig.device_to_host(mcu_guard.data(),
                           static_cast<unsigned char*>(out_mcu[i]) + out_elems * 4u,
                           kGuardBytes);
        rig.device_to_host(st_mcu_h.data(), st_mcu[i], state_elems * 4u);
        size_t first = 0, diffs = 0;
        check(chain::bytes_equal(refs[i].data(), mcu_out.data(), out_elems * 4u,
                                 &first, &diffs),
              "mcu recurrence output matches the host launch");
        if (diffs)
            std::printf("  %s mcu out diff=%zu first=%zu\n", c.name, diffs, first);
        check(mcu_guard == guards[i], "mcu output sentinel intact");
        size_t sfirst = 0, sdiffs = 0;
        check(chain::bytes_equal(state_refs[i].data(), st_mcu_h.data(),
                                 state_elems * 4u, &sfirst, &sdiffs),
              "mcu recurrence state matches the host launch");
        if (sdiffs)
            std::printf("  %s mcu state diff=%zu first=%zu\n", c.name, sdiffs, sfirst);
        std::printf("  %s: out_ok=%d state_ok=%d\n", c.name,
                    static_cast<int>(diffs == 0u), static_cast<int>(sdiffs == 0u));
    }

    {
        const uint32_t used = rig.fsm.log_head();
        uint32_t hist[16] = {};
        for (uint32_t i = 0u; i < used && i < mcu::kMcuLogLineCount; ++i) {
            const mcu::McuLogRecord* rec = rig.fsm.log_record(i);
            if (rec == nullptr || rec->event >= 16u) continue;
            if (rec->sequence >= static_cast<uint64_t>(used)) continue;
            hist[rec->event] += 1u;
        }
        std::printf("  mcu log records=%u", used);
        for (uint32_t e = 1u; e < 16u; ++e) {
            if (hist[e] != 0u) std::printf(" ev%u=%u", e, hist[e]);
        }
        std::printf("\n");
        check(hist[static_cast<uint32_t>(mcu::McuLogEvent::PlanBegin)] >= 1u &&
                  hist[static_cast<uint32_t>(mcu::McuLogEvent::
                                                 DispatchPublish)] >=
                      static_cast<uint32_t>(kCaseCount),
              "mcu log captured plan and dispatch publish events");
    }
    check(rig.close(), "rig teardown");

    (void)hipStreamDestroy(stream);
    for (void* p : {q, k, v, ga, gb, dt_bias, a_log, requests, out_ref, out_aql,
                    st_ref, st_aql})
        (void)hipFree(p);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        (void)hipFree(out_mcu[i]);
        (void)hipFree(st_mcu[i]);
    }

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_gdn_recurrence: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_gdn_recurrence: PASS\n");
    return 0;
}
