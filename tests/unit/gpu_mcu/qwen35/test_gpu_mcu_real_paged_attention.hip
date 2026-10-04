#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
#include <phaseshift/models/qwen35/runtime/paged_attention_dispatch.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>

#include <chrono>
#include <cmath>
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
constexpr uint32_t kQHeads = 16u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kNumPages = 32u;
constexpr uint32_t kLayers = 1u;
constexpr uint32_t kBlockTableStride = 32u;
constexpr uint32_t kMaxSequences = 2u;
constexpr uint32_t kPosition = 100u;
constexpr uint32_t kMaxVisible = 512u;

constexpr uint32_t kQFeatures = kQHeads * kHeadDim;
constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;
constexpr uint32_t kElemsPerLayer = kNumPages * kElemsPerPage;
constexpr size_t kPoolBytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);
constexpr size_t kQBytes = static_cast<size_t>(kQFeatures) * sizeof(uint16_t);
constexpr size_t kOutBytes = static_cast<size_t>(kQFeatures) * sizeof(float);
constexpr size_t kGuardBytes = 64u;

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

void fill_common(kernel::PagedAttentionCommonArgs& c, const void* q, void* out,
                 const uint32_t* row_slots, const uint32_t* row_pos,
                 const uint32_t* block_tables) {
    c.q = static_cast<const ps::bf16_t*>(q);
    c.output = out;
    c.output_dtype = kernel::PagedAttentionOutputDType::F32;
    c.q_row_stride = kQFeatures;
    c.output_row_stride = kQFeatures;
    c.row_sequence_slots = row_slots;
    c.row_positions = row_pos;
    c.block_tables = block_tables;
    c.block_table_stride = kBlockTableStride;
    c.max_sequences = kMaxSequences;
    c.rows = kRows;
    c.layer = 0u;
    c.page_tokens = kPageTokens;
    c.num_pages = kNumPages;
    c.num_attention_layers = kLayers;
    c.q_heads = kQHeads;
    c.kv_heads = kKvHeads;
    c.head_dim = kHeadDim;
    c.elems_per_token = kElemsPerToken;
    c.elems_per_page = kElemsPerPage;
    c.elems_per_layer = kElemsPerLayer;
    c.scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
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

    void* q_dev = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* out_ref = nullptr;
    void* out_aql = nullptr;
    void* out_mcu = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    if (hipMalloc(&q_dev, kQBytes) != hipSuccess ||
        hipMalloc(&k_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&out_ref, kOutBytes + kGuardBytes) != hipSuccess ||
        hipMalloc(&out_aql, kOutBytes + kGuardBytes) != hipSuccess ||
        hipMalloc(&out_mcu, kOutBytes + kGuardBytes) != hipSuccess ||
        hipMalloc(&row_slots, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables,
                  kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }

    std::vector<uint16_t> h_q(kQFeatures);
    for (uint32_t i = 0; i < kQFeatures; ++i)
        h_q[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
    std::vector<uint16_t> h_pool(kElemsPerLayer);
    for (uint32_t i = 0; i < kElemsPerLayer; ++i) {
        h_pool[i] = static_cast<uint16_t>(0x3900u + ((i * 40503u) % 1700u));
    }
    (void)hipMemcpy(q_dev, h_q.data(), kQBytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(k_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
    for (uint32_t i = 0; i < kElemsPerLayer; ++i)
        h_pool[i] = static_cast<uint16_t>(0x3B00u + ((i * 2654435761u) % 1500u));
    (void)hipMemcpy(v_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
    (void)hipMemset(out_ref, 0xAB, kOutBytes + kGuardBytes);
    (void)hipMemset(out_aql, 0xAB, kOutBytes + kGuardBytes);
    (void)hipMemset(out_mcu, 0xAB, kOutBytes + kGuardBytes);

    const uint32_t h_slots[kRows] = {0u};
    const uint32_t h_pos[kRows] = {kPosition};
    uint32_t h_bt[kMaxSequences * kBlockTableStride];
    for (uint32_t i = 0; i < kMaxSequences * kBlockTableStride; ++i)
        h_bt[i] = i % kNumPages;
    (void)hipMemcpy(row_slots, h_slots, sizeof(h_slots), hipMemcpyHostToDevice);
    (void)hipMemcpy(row_pos, h_pos, sizeof(h_pos), hipMemcpyHostToDevice);
    (void)hipMemcpy(block_tables, h_bt, sizeof(h_bt), hipMemcpyHostToDevice);

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    kernel::PagedAttentionCommonArgs common{};
    fill_common(common, q_dev, out_ref, static_cast<const uint32_t*>(row_slots),
                static_cast<const uint32_t*>(row_pos),
                static_cast<const uint32_t*>(block_tables));

    ps::qwen35::runtime::PhysicalPagedAttentionPlan plan{};
    plan.optimized = true;
    plan.use_prefill = false;
    plan.splits = 1u;
    plan.physical_count = 1u;
    plan.kernels[0].variant =
        ps::qwen35::runtime::PhysicalPagedAttentionVariant::Bf16FullHead;
    plan.kernels[0].common = common;
    plan.kernels[0].k_pool = reinterpret_cast<uint64_t>(k_pool);
    plan.kernels[0].v_pool = reinterpret_cast<uint64_t>(v_pool);
    plan.kernels[0].grid = kRows * kQHeads;
    plan.kernels[0].workgroup = kernel::kPagedAttentionBlock;
    std::printf(
        "plan: physical_count=%u splits=%u use_prefill=%d optimized=%d "
        "grid=%u wg=%u rows=%u q_heads=%u kv_heads=%u head_dim=%u page_tokens=%u "
        "max_visible=%u position=%u\n",
        plan.physical_count, plan.splits, plan.use_prefill ? 1 : 0,
        plan.optimized ? 1 : 0, plan.kernels[0].grid, plan.kernels[0].workgroup,
        common.rows, common.q_heads, common.kv_heads, common.head_dim,
        common.page_tokens, kMaxVisible, kPosition);
    check(plan.splits == 1u, "split path not taken at this max_visible");
    check(ps::qwen35::runtime::launch_paged_attention_physical(plan, stream) ==
              hipSuccess,
          "host production paged attention launcher");
    check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();
    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO,
        kernel::kPagedAttentionBf16Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: paged attention code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    std::printf("paged attention bf16 hsaco: kernarg=%u private=%u group=%u "
                "sizeof(args)=%zu hidden_fits=%d\n",
                meta.kernarg_segment_size, meta.private_segment_size,
                meta.static_group_segment_size,
                sizeof(kernel::PagedAttentionBf16Args),
                static_cast<int>(mcu::aql_hidden_args_fits(
                    sizeof(kernel::PagedAttentionBf16Args),
                    meta.kernarg_segment_size)));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::PagedAttentionBf16Args),
          "kernarg segment covers the stable args");

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), 1u);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    kernel::PagedAttentionCommonArgs aql_common = common;
    aql_common.output = out_aql;
    {
        unsigned char* slot = region.base;
        std::memset(slot, 0, region.slot_stride);
        kernel::PagedAttentionBf16Args args{};
        args.common = aql_common;
        args.k_pool = static_cast<const ps::bf16_t*>(k_pool);
        args.v_pool = static_cast<const ps::bf16_t*>(v_pool);
        std::memcpy(slot, &args, sizeof(args));

        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = plan.kernels[0].grid * kernel::kPagedAttentionBlock;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(kernel::kPagedAttentionBlock);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::PagedAttentionBf16Args),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size,
                                       plan.kernels[0].grid, 1u, 1u,
                                       kernel::kPagedAttentionBlock);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
        check(wait_queue_drained(queue, queue.host_queued_index(), 15000),
              "aql queue drained");
        (void)hipDeviceSynchronize();
    }

    std::vector<uint32_t> ref_words(kOutBytes / 4u, 0u);
    std::vector<uint32_t> aql_words(kOutBytes / 4u, 0u);
    std::vector<uint32_t> ref_guard(kGuardBytes / 4u, 0u);
    std::vector<uint32_t> aql_guard(kGuardBytes / 4u, 0u);
    (void)hipMemcpy(ref_words.data(), out_ref, kOutBytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(aql_words.data(), out_aql, kOutBytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(ref_guard.data(),
                    static_cast<unsigned char*>(out_ref) + kOutBytes, kGuardBytes,
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(aql_guard.data(),
                    static_cast<unsigned char*>(out_aql) + kOutBytes, kGuardBytes,
                    hipMemcpyDeviceToHost);
    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(ref_words.data(), aql_words.data(), kOutBytes, &first,
                             &diffs),
          "host produced aql is bit exact with the host launch");
    if (diffs)
        std::printf("  aql attention diff count=%zu first=%zu\n", diffs, first);
    check(aql_guard == ref_guard, "aql output sentinel intact");

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");
    (void)hipStreamDestroy(stream);

    Rig rig;
    if (!rig.open(0, 4u, 3u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords, meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO,
                              kernel::kPagedAttentionBf16Symbol, mcu_meta)) {
        return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, mcu_meta,
                                 mcu::kMcuKernargRecipeAttentionPaged,
                                 kernel::kPagedAttentionBlock,
                                 plan.kernels[0].grid, mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};

    std::vector<mcu::McuPlanNode> nodes(4);
    nodes[0].variant_id = 0u;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeAttentionPaged;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0u;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1u;
    nodes[1].variant_id = 1u;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[1].completion_slot = 0;
    nodes[1].input_slot = mcu::kMcuNoInput;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2u;
    nodes[2].completion_slot = 0;
    nodes[2].flags = mcu::kMcuNodeWait;
    nodes[2].next = 3u;
    nodes[3].flags = mcu::kMcuNodeEnd;
    if (!rig.upload(nodes, variants)) return 1;

    mcu::McuPagedAttentionInvocation inv{};
    inv.q = reinterpret_cast<uint64_t>(q_dev);
    inv.output = reinterpret_cast<uint64_t>(out_mcu);
    inv.output_dtype =
        static_cast<uint32_t>(kernel::PagedAttentionOutputDType::F32);
    inv.q_row_stride = common.q_row_stride;
    inv.output_row_stride = common.output_row_stride;
    inv.row_sequence_slots = reinterpret_cast<uint64_t>(row_slots);
    inv.row_positions = reinterpret_cast<uint64_t>(row_pos);
    inv.block_tables = reinterpret_cast<uint64_t>(block_tables);
    inv.block_table_stride = common.block_table_stride;
    inv.max_sequences = common.max_sequences;
    inv.rows = common.rows;
    inv.layer = common.layer;
    inv.page_tokens = common.page_tokens;
    inv.num_pages = common.num_pages;
    inv.num_attention_layers = common.num_attention_layers;
    inv.q_heads = common.q_heads;
    inv.kv_heads = common.kv_heads;
    inv.head_dim = common.head_dim;
    inv.elems_per_token = common.elems_per_token;
    inv.elems_per_page = common.elems_per_page;
    inv.elems_per_layer = common.elems_per_layer;
    inv.scale = common.scale;
    inv.k_pool = reinterpret_cast<uint64_t>(k_pool);
    inv.v_pool = reinterpret_cast<uint64_t>(v_pool);
    if (!rig.set_attention_paged_invocations(&inv, 1u)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x50415454u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 2u;
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
    check(rig.fsm.dispatches() == 2u, "two dispatches (attention + marker)");

    std::vector<uint32_t> mcu_words(kOutBytes / 4u, 0u);
    std::vector<uint32_t> mcu_guard(kGuardBytes / 4u, 0u);
    rig.device_to_host(mcu_words.data(), out_mcu, kOutBytes);
    rig.device_to_host(mcu_guard.data(),
                       static_cast<unsigned char*>(out_mcu) + kOutBytes,
                       kGuardBytes);
    first = 0;
    diffs = 0;
    check(chain::bytes_equal(ref_words.data(), mcu_words.data(), kOutBytes, &first,
                             &diffs),
          "mcu paged attention matches the host launch");
    if (diffs)
        std::printf("  mcu attention diff count=%zu first=%zu\n", diffs, first);
    check(mcu_guard == ref_guard, "mcu output sentinel intact");
    std::printf("  attention R=%u q_heads=%u kv_heads=%u head_dim=%u ok=%d\n",
                kRows, kQHeads, kKvHeads, kHeadDim,
                static_cast<int>(diffs == 0u));

    check(rig.close(), "rig teardown");
    (void)hipFree(q_dev);
    (void)hipFree(k_pool);
    (void)hipFree(v_pool);
    (void)hipFree(out_ref);
    (void)hipFree(out_aql);
    (void)hipFree(out_mcu);
    (void)hipFree(row_slots);
    (void)hipFree(row_pos);
    (void)hipFree(block_tables);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_paged_attention: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_paged_attention: PASS\n");
    return 0;
}
