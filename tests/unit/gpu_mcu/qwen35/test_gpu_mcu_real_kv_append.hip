#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/kv_append.h>
#include <phaseshift/models/qwen35/runtime/physical_launch.h>
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

constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kNumPages = 4u;
constexpr uint32_t kNumLayers = 1u;
constexpr uint32_t kBlockTableStride = 4u;
constexpr uint32_t kMaxSequences = 2u;
constexpr uint32_t kRows = 1u;

constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;
constexpr uint32_t kElemsPerLayer = kNumPages * kElemsPerPage;
constexpr size_t kPoolBytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);
constexpr size_t kInputBytes = static_cast<size_t>(kRows) * kElemsPerToken * sizeof(uint16_t);

const uint32_t kPositions[] = {0u, kPageTokens - 1u, kPageTokens};

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

void fill_common(kernel::KvAppendCommonArgs& c, const void* k_in, const void* v_in,
                 const uint32_t* row_slots, const uint32_t* row_pos,
                 const uint32_t* block_tables) {
    c.k_input = static_cast<const ps::bf16_t*>(k_in);
    c.v_input = static_cast<const ps::bf16_t*>(v_in);
    c.k_row_stride = kElemsPerToken;
    c.v_row_stride = kElemsPerToken;
    c.row_sequence_slots = row_slots;
    c.row_positions = row_pos;
    c.block_tables = block_tables;
    c.block_table_stride = kBlockTableStride;
    c.max_sequences = kMaxSequences;
    c.rows = kRows;
    c.layer = 0u;
    c.page_tokens = kPageTokens;
    c.num_pages = kNumPages;
    c.num_attention_layers = kNumLayers;
    c.kv_heads = kKvHeads;
    c.head_dim = kHeadDim;
    c.elems_per_token = kElemsPerToken;
    c.elems_per_page = kElemsPerPage;
    c.elems_per_layer = kElemsPerLayer;
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

    void* k_in = nullptr;
    void* v_in = nullptr;
    void* k_ref = nullptr;
    void* v_ref = nullptr;
    void* k_aql = nullptr;
    void* v_aql = nullptr;
    void* k_mcu = nullptr;
    void* v_mcu = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    if (hipMalloc(&k_in, kInputBytes) != hipSuccess ||
        hipMalloc(&v_in, kInputBytes) != hipSuccess ||
        hipMalloc(&k_ref, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_ref, kPoolBytes) != hipSuccess ||
        hipMalloc(&k_aql, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_aql, kPoolBytes) != hipSuccess ||
        hipMalloc(&k_mcu, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_mcu, kPoolBytes) != hipSuccess ||
        hipMalloc(&row_slots, kRows * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, sizeof(kPositions)) != hipSuccess ||
        hipMalloc(&block_tables, kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }

    std::vector<uint16_t> h_k(kElemsPerToken), h_v(kElemsPerToken);
    for (uint32_t i = 0; i < kElemsPerToken; ++i) {
        h_k[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
        h_v[i] = static_cast<uint16_t>(0x3A00u + ((i * 40503u) % 1900u));
    }
    (void)hipMemcpy(k_in, h_k.data(), kInputBytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(v_in, h_v.data(), kInputBytes, hipMemcpyHostToDevice);

    const uint32_t h_slots[kRows] = {0u};
    uint32_t h_bt[kMaxSequences * kBlockTableStride];
    for (uint32_t i = 0; i < kMaxSequences * kBlockTableStride; ++i)
        h_bt[i] = i % kNumPages;
    (void)hipMemcpy(row_slots, h_slots, sizeof(h_slots), hipMemcpyHostToDevice);
    (void)hipMemcpy(block_tables, h_bt, sizeof(h_bt), hipMemcpyHostToDevice);

    const uint32_t case_count =
        static_cast<uint32_t>(sizeof(kPositions) / sizeof(kPositions[0]));

    (void)hipMemcpy(row_pos, kPositions, sizeof(kPositions), hipMemcpyHostToDevice);

    auto common_for = [&](uint32_t index) {
        kernel::KvAppendCommonArgs c{};
        fill_common(c, k_in, v_in, static_cast<const uint32_t*>(row_slots),
                    static_cast<const uint32_t*>(row_pos) + index,
                    static_cast<const uint32_t*>(block_tables));
        return c;
    };

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    (void)hipMemset(k_ref, 0, kPoolBytes);
    (void)hipMemset(v_ref, 0, kPoolBytes);
    for (uint32_t i = 0; i < case_count; ++i) {
        ps::qwen35::runtime::PhysicalKvAppendLaunch launch{};
        launch.variant = ps::qwen35::runtime::PhysicalKvAppendVariant::Bf16;
        launch.common = common_for(i);
        launch.k_pool = reinterpret_cast<uint64_t>(k_ref);
        launch.v_pool = reinterpret_cast<uint64_t>(v_ref);
        check(ps::qwen35::runtime::launch_kv_append_physical(launch, stream) ==
                  hipSuccess,
              "host production kv append launcher");
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
        queue, PHASESHIFT_GPU_MCU_KV_APPEND_HSACO, kernel::kKvAppendBf16Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: kv append code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
        new mcu::GpuMcuAqlCodeObject(code_res.release()));
    const mcu::GpuAqlKernelMetadata meta = code->metadata();
    const uint32_t chunks = (kHeadDim + 3u) / 4u;
    const uint32_t block =
        chunks <= 32u ? 32u : chunks <= 64u ? 64u : chunks <= 128u ? 128u : 256u;
    const uint32_t grid = kRows * kKvHeads;
    std::printf("kv append bf16 hsaco: kernarg=%u private=%u group=%u "
                "sizeof(args)=%zu grid=%u block=%u\n",
                meta.kernarg_segment_size, meta.private_segment_size,
                meta.static_group_segment_size, sizeof(kernel::KvAppendBf16Args),
                grid, block);
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::KvAppendBf16Args),
          "kernarg segment covers the stable args");
    std::printf("  hidden_args_fits=%d\n",
                static_cast<int>(mcu::aql_hidden_args_fits(
                    sizeof(kernel::KvAppendBf16Args), meta.kernarg_segment_size)));

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), case_count);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    auto submit_case = [&](unsigned char* slot, uint32_t index, const void* k_pool,
                           const void* v_pool) {
        std::memset(slot, 0, region.slot_stride);
        kernel::KvAppendBf16Args args{};
        args.common = common_for(index);
        args.k_pool = static_cast<ps::bf16_t*>(const_cast<void*>(k_pool));
        args.v_pool = static_cast<ps::bf16_t*>(const_cast<void*>(v_pool));
        std::memcpy(slot, &args, sizeof(args));
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
        mcu::build_aql_hidden_args(slot, sizeof(kernel::KvAppendBf16Args),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid, 1u, 1u,
                                       block);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        return queue.submit_host(&tmpl, 1).ok();
    };

    (void)hipMemset(k_aql, 0, kPoolBytes);
    (void)hipMemset(v_aql, 0, kPoolBytes);
    check(hipDeviceSynchronize() == hipSuccess, "kernarg pools zeroed");
    for (uint32_t i = 0; i < case_count; ++i) {
        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        check(submit_case(slot, i, k_aql, v_aql), "host aql submit");
        check(wait_queue_drained(queue, queue.host_queued_index(), 15000),
              "aql queue drained");
    }

    std::vector<uint32_t> k_ref_words(kPoolBytes / 4u, 0u);
    std::vector<uint32_t> v_ref_words(kPoolBytes / 4u, 0u);
    std::vector<uint32_t> k_aql_words(kPoolBytes / 4u, 0u);
    std::vector<uint32_t> v_aql_words(kPoolBytes / 4u, 0u);
    (void)hipMemcpy(k_ref_words.data(), k_ref, kPoolBytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(v_ref_words.data(), v_ref, kPoolBytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(k_aql_words.data(), k_aql, kPoolBytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(v_aql_words.data(), v_aql, kPoolBytes, hipMemcpyDeviceToHost);
    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(k_ref_words.data(), k_aql_words.data(), kPoolBytes, &first,
                             &diffs),
          "aql K state is bit exact with the host launch");
    if (diffs)
        std::printf("  aql K diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(v_ref_words.data(), v_aql_words.data(), kPoolBytes, &first,
                             &diffs),
          "aql V state is bit exact with the host launch");
    if (diffs)
        std::printf("  aql V diff count=%zu first=%zu\n", diffs, first);

    (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");
    (void)hipStreamDestroy(stream);

    Rig rig;
    if (!rig.open(0, case_count + 3u, 2u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords, meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return 1;
    }
    mcu::GpuAqlKernelMetadata mcu_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_KV_APPEND_HSACO,
                              kernel::kKvAppendBf16Symbol, mcu_meta)) {
        return 1;
    }

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, mcu_meta, mcu::kMcuKernargRecipeKvAppend, block,
                                 grid, mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(1u, rig.meta, 64u, 1u, mcu::kMcuSystemAqlPolicy)};

    std::vector<mcu::McuPlanNode> nodes(case_count + 3u);
    for (uint32_t i = 0; i < case_count; ++i) {
        nodes[i].variant_id = 0u;
        nodes[i].kernarg_recipe = mcu::kMcuKernargRecipeKvAppend;
        nodes[i].completion_slot = 0;
        nodes[i].invocation_index = i;
        nodes[i].flags = mcu::kMcuNodeDispatch;
        nodes[i].next = static_cast<uint16_t>(i + 1u);
    }
    nodes[case_count].variant_id = 1u;
    nodes[case_count].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[case_count].completion_slot = 0;
    nodes[case_count].input_slot = mcu::kMcuNoInput;
    nodes[case_count].flags = mcu::kMcuNodeDispatch;
    nodes[case_count].next = static_cast<uint16_t>(case_count + 1u);
    nodes[case_count + 1u].completion_slot = 0;
    nodes[case_count + 1u].flags = mcu::kMcuNodeWait;
    nodes[case_count + 1u].next = static_cast<uint16_t>(case_count + 2u);
    nodes[case_count + 2u].flags = mcu::kMcuNodeEnd;
    if (!rig.upload(nodes, variants)) return 1;

    (void)hipMemset(k_mcu, 0, kPoolBytes);
    (void)hipMemset(v_mcu, 0, kPoolBytes);
    std::vector<mcu::McuKvAppendInvocation> invocations(case_count);
    for (uint32_t i = 0; i < case_count; ++i) {
        const kernel::KvAppendCommonArgs c = common_for(i);
        mcu::McuKvAppendInvocation& inv = invocations[i];
        inv = mcu::McuKvAppendInvocation{};
        inv.k_input = reinterpret_cast<uint64_t>(c.k_input);
        inv.v_input = reinterpret_cast<uint64_t>(c.v_input);
        inv.k_row_stride = c.k_row_stride;
        inv.v_row_stride = c.v_row_stride;
        inv.row_sequence_slots = reinterpret_cast<uint64_t>(c.row_sequence_slots);
        inv.row_positions = reinterpret_cast<uint64_t>(c.row_positions);
        inv.block_tables = reinterpret_cast<uint64_t>(c.block_tables);
        inv.block_table_stride = c.block_table_stride;
        inv.max_sequences = c.max_sequences;
        inv.rows = c.rows;
        inv.layer = c.layer;
        inv.page_tokens = c.page_tokens;
        inv.num_pages = c.num_pages;
        inv.num_attention_layers = c.num_attention_layers;
        inv.kv_heads = c.kv_heads;
        inv.head_dim = c.head_dim;
        inv.elems_per_token = c.elems_per_token;
        inv.elems_per_page = c.elems_per_page;
        inv.elems_per_layer = c.elems_per_layer;
        inv.k_pool = reinterpret_cast<uint64_t>(k_mcu);
        inv.v_pool = reinterpret_cast<uint64_t>(v_mcu);
    }
    if (!rig.set_kv_append_invocations(invocations.data(), case_count)) return 1;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x4b564150u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = case_count + 1u;
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
    check(rig.fsm.dispatches() == static_cast<uint64_t>(case_count) + 1u,
          "dispatch count");

    std::vector<uint32_t> k_mcu_words(kPoolBytes / 4u, 0u);
    std::vector<uint32_t> v_mcu_words(kPoolBytes / 4u, 0u);
    rig.device_to_host(k_mcu_words.data(), k_mcu, kPoolBytes);
    rig.device_to_host(v_mcu_words.data(), v_mcu, kPoolBytes);
    first = 0;
    diffs = 0;
    check(chain::bytes_equal(k_ref_words.data(), k_mcu_words.data(), kPoolBytes, &first,
                             &diffs),
          "mcu K state matches the host launch");
    if (diffs)
        std::printf("  mcu K diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(v_ref_words.data(), v_mcu_words.data(), kPoolBytes, &first,
                             &diffs),
          "mcu V state matches the host launch");
    if (diffs)
        std::printf("  mcu V diff count=%zu first=%zu\n", diffs, first);
    std::printf("  positions=%u,%u,%u page_tokens=%u grid=%u block=%u kv_heads=%u "
                "head_dim=%u ok=%d\n",
                kPositions[0], kPositions[1], kPositions[2], kPageTokens, grid, block,
                kKvHeads, kHeadDim, static_cast<int>(diffs == 0u));

    check(rig.close(), "rig teardown");
    (void)hipFree(k_in);
    (void)hipFree(v_in);
    (void)hipFree(k_ref);
    (void)hipFree(v_ref);
    (void)hipFree(k_aql);
    (void)hipFree(v_aql);
    (void)hipFree(k_mcu);
    (void)hipFree(v_mcu);
    (void)hipFree(row_slots);
    (void)hipFree(row_pos);
    (void)hipFree(block_tables);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_kv_append: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_kv_append: PASS\n");
    return 0;
}
