#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"
#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

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
namespace mcu = ::ps::runtime::gpu_mcu;
namespace kernel = ::ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kRows = 1u;
constexpr uint32_t kKvHeads = 4u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kPageTokens = 16u;
constexpr uint32_t kBlockTableStride = 256u;
constexpr uint32_t kNumPages = 256u;
constexpr uint32_t kMaxSequences = 2u;
constexpr uint32_t kPosition = 100u;
constexpr uint32_t kSplits = 16u;

constexpr uint32_t kElemsPerToken = kKvHeads * kHeadDim;
constexpr uint32_t kElemsPerPage = kPageTokens * kElemsPerToken;
constexpr uint32_t kElemsPerLayer = kNumPages * kElemsPerPage;
constexpr size_t kPoolBytes = static_cast<size_t>(kElemsPerLayer) * sizeof(uint16_t);
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
                 const uint32_t* block_tables, uint32_t q_heads,
                 uint32_t q_features) {
    c.q = static_cast<const ps::bf16_t*>(q);
    c.output = out;
    c.output_dtype = kernel::PagedAttentionOutputDType::F32;
    c.q_row_stride = q_features;
    c.output_row_stride = q_features;
    c.row_sequence_slots = row_slots;
    c.row_positions = row_pos;
    c.block_tables = block_tables;
    c.block_table_stride = kBlockTableStride;
    c.max_sequences = kMaxSequences;
    c.rows = kRows;
    c.layer = 0u;
    c.page_tokens = kPageTokens;
    c.num_pages = kNumPages;
    c.num_attention_layers = 1u;
    c.q_heads = q_heads;
    c.kv_heads = kKvHeads;
    c.head_dim = kHeadDim;
    c.elems_per_token = kElemsPerToken;
    c.elems_per_page = kElemsPerPage;
    c.elems_per_layer = kElemsPerLayer;
    c.scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
}

void run_split_scenario(uint32_t q_heads, const char* label) {
    const uint32_t q_per_kv = q_heads / kKvHeads;
    const uint32_t q_features = q_heads * kHeadDim;
    const size_t q_bytes = static_cast<size_t>(q_features) * sizeof(uint16_t);
    const size_t out_bytes = static_cast<size_t>(q_features) * sizeof(float);
    const size_t partial_floats =
        static_cast<size_t>(kRows) * kKvHeads * kSplits * q_per_kv *
        (2u + kHeadDim);
    const size_t partial_bytes = partial_floats * sizeof(float);
    const size_t guarded_out = out_bytes + kGuardBytes;

    const uint32_t qh_per_wg = kernel::paged_attention_split_reduce_qh_per_wg();
    const bool qh_route = qh_per_wg < q_per_kv && q_per_kv % qh_per_wg == 0u;
    const char* split_symbol = nullptr;
    const char* reduce_symbol = nullptr;
    uint32_t reduce_grid = 0u;
    if (q_per_kv == 4u) {
        split_symbol = kernel::kPagedAttentionSplitQ4FullSymbol;
        if (qh_route && qh_per_wg == 1u) {
            reduce_symbol = kernel::kPagedAttentionReduceQ4FullQh1Symbol;
        } else if (qh_route && qh_per_wg == 2u) {
            reduce_symbol = kernel::kPagedAttentionReduceQ4FullQh2Symbol;
        } else {
            reduce_symbol = kernel::kPagedAttentionReduceQ4FullBaseSymbol;
        }
        reduce_grid =
            qh_route ? kRows * kKvHeads * (q_per_kv / qh_per_wg) : kRows * kKvHeads;
    } else {
        split_symbol = kernel::kPagedAttentionSplitOtherSymbol;
        reduce_symbol = kernel::kPagedAttentionReduceOtherSymbol;
        reduce_grid = kRows * kKvHeads;
    }
    const uint32_t split_grid = kRows * kKvHeads * kSplits;
    std::printf("[%s] paged attention split/reduce: q_heads=%u q_per_kv=%u "
                "full_head=%d splits=%u qh_per_wg=%u qh_route=%d split_grid=%u "
                "reduce_grid=%u reduce=%s\n",
                label, q_heads, q_per_kv,
                kHeadDim == kernel::kPagedAttentionHeadLimit ? 1 : 0, kSplits,
                qh_per_wg, qh_route ? 1 : 0, split_grid, reduce_grid,
                reduce_symbol);

    void* q_dev = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* partials_host_run = nullptr;
    void* partials_aql = nullptr;
    void* partials_mcu = nullptr;
    void* out_ref = nullptr;
    void* out_aql = nullptr;
    void* out_mcu = nullptr;
    void* row_slots = nullptr;
    void* row_pos = nullptr;
    void* block_tables = nullptr;
    if (hipMalloc(&q_dev, q_bytes) != hipSuccess ||
        hipMalloc(&k_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&v_pool, kPoolBytes) != hipSuccess ||
        hipMalloc(&partials_host_run, partial_bytes) != hipSuccess ||
        hipMalloc(&partials_aql, partial_bytes) != hipSuccess ||
        hipMalloc(&partials_mcu, partial_bytes) != hipSuccess ||
        hipMalloc(&out_ref, guarded_out) != hipSuccess ||
        hipMalloc(&out_aql, guarded_out) != hipSuccess ||
        hipMalloc(&out_mcu, guarded_out) != hipSuccess ||
        hipMalloc(&row_slots, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&row_pos, sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&block_tables,
                  kMaxSequences * kBlockTableStride * sizeof(uint32_t)) !=
            hipSuccess) {
        check(false, "buffer alloc");
        return;
    }

    std::vector<uint16_t> h_q(q_features);
    for (uint32_t i = 0; i < q_features; ++i)
        h_q[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1900u));
    std::vector<uint16_t> h_pool(kElemsPerLayer);
    for (uint32_t i = 0; i < kElemsPerLayer; ++i)
        h_pool[i] = static_cast<uint16_t>(0x3900u + ((i * 40503u) % 1700u));
    (void)hipMemcpy(q_dev, h_q.data(), q_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(k_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);
    for (uint32_t i = 0; i < kElemsPerLayer; ++i)
        h_pool[i] = static_cast<uint16_t>(0x3B00u + ((i * 2654435761u) % 1500u));
    (void)hipMemcpy(v_pool, h_pool.data(), kPoolBytes, hipMemcpyHostToDevice);

    const uint32_t h_slots[kRows] = {0u};
    const uint32_t h_pos[kRows] = {kPosition};
    std::vector<uint32_t> h_bt(kMaxSequences * kBlockTableStride);
    for (uint32_t i = 0; i < kMaxSequences * kBlockTableStride; ++i)
        h_bt[i] = i % kNumPages;
    (void)hipMemcpy(row_slots, h_slots, sizeof(h_slots), hipMemcpyHostToDevice);
    (void)hipMemcpy(row_pos, h_pos, sizeof(h_pos), hipMemcpyHostToDevice);
    (void)hipMemcpy(block_tables, h_bt.data(),
                    h_bt.size() * sizeof(uint32_t), hipMemcpyHostToDevice);

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    kernel::PagedAttentionCommonArgs common{};
    fill_common(common, q_dev, out_ref, static_cast<const uint32_t*>(row_slots),
                static_cast<const uint32_t*>(row_pos),
                static_cast<const uint32_t*>(block_tables), q_heads, q_features);

    check(kernel::launch_attention_paged_bf16_split(
              common, static_cast<const ps::bf16_t*>(k_pool),
              static_cast<const ps::bf16_t*>(v_pool),
              static_cast<float*>(partials_host_run), kSplits, stream) ==
              hipSuccess,
          "host production split launcher");
    check(hipStreamSynchronize(stream) == hipSuccess, "split sync");

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return;
    }
    auto queue = queue_res.release();
    auto split_code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO, split_symbol);
    if (!split_code_res.ok()) {
        std::printf("FAIL: split code object: %s\n",
                    split_code_res.status().message().c_str());
        return;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> split_code(
        new mcu::GpuMcuAqlCodeObject(split_code_res.release()));
    auto reduce_code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO, reduce_symbol);
    if (!reduce_code_res.ok()) {
        std::printf("FAIL: reduce code object: %s\n",
                    reduce_code_res.status().message().c_str());
        return;
    }
    std::unique_ptr<mcu::GpuMcuAqlCodeObject> reduce_code(
        new mcu::GpuMcuAqlCodeObject(reduce_code_res.release()));

    const mcu::GpuAqlKernelMetadata split_meta = split_code->metadata();
    const mcu::GpuAqlKernelMetadata reduce_meta = reduce_code->metadata();
    std::printf("split args=%zu kernarg=%u hidden_fits=%d | reduce args=%zu "
                "kernarg=%u hidden_fits=%d\n",
                sizeof(kernel::PagedAttentionSplitArgs),
                split_meta.kernarg_segment_size,
                static_cast<int>(mcu::aql_hidden_args_fits(
                    sizeof(kernel::PagedAttentionSplitArgs),
                    split_meta.kernarg_segment_size)),
                sizeof(kernel::PagedAttentionSplitReduceArgs),
                reduce_meta.kernarg_segment_size,
                static_cast<int>(mcu::aql_hidden_args_fits(
                    sizeof(kernel::PagedAttentionSplitReduceArgs),
                    reduce_meta.kernarg_segment_size)));
    check(split_meta.kernel_object != 0, "split kernel_object resolved");
    check(reduce_meta.kernel_object != 0, "reduce kernel_object resolved");
    check(split_meta.kernarg_segment_size >= sizeof(kernel::PagedAttentionSplitArgs),
          "split kernarg covers the stable args");
    check(reduce_meta.kernarg_segment_size >=
              sizeof(kernel::PagedAttentionSplitReduceArgs),
          "reduce kernarg covers the stable args");

    // ---- host AQL split, compare partials ----
    {
        auto region_res = queue.allocate_kernarg(
            mcu::aql_kernarg_slot_stride(split_meta.kernarg_segment_size), 1u);
        if (!region_res.ok()) {
            std::printf("FAIL: split kernarg: %s\n",
                        region_res.status().message().c_str());
            return;
        }
        auto region = region_res.release();
        unsigned char* slot = region.base;
        std::memset(slot, 0, region.slot_stride);
        kernel::PagedAttentionSplitArgs args{};
        args.common = common;
        args.k_pool = static_cast<const ps::bf16_t*>(k_pool);
        args.v_pool = static_cast<const ps::bf16_t*>(v_pool);
        args.partials = static_cast<float*>(partials_aql);
        args.splits = kSplits;
        std::memcpy(slot, &args, sizeof(args));
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = split_meta.kernel_object;
        desc.global_work_items_x = split_grid * kernel::kPagedAttentionSplitBlock;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x =
            static_cast<uint16_t>(kernel::kPagedAttentionSplitBlock);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = split_meta.private_segment_size;
        desc.group_segment_size = split_meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(args), split_meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, split_meta.kernarg_segment_size,
                                       split_grid, 1u, 1u,
                                       kernel::kPagedAttentionSplitBlock);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "split aql submit");
        check(wait_queue_drained(queue, queue.host_queued_index(), 15000),
              "split queue drained");
        (void)hipDeviceSynchronize();
    }
    std::vector<uint32_t> ref_p(partial_bytes / 4u, 0u);
    std::vector<uint32_t> aql_p(partial_bytes / 4u, 0u);
    (void)hipMemcpy(ref_p.data(), partials_host_run, partial_bytes,
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(aql_p.data(), partials_aql, partial_bytes,
                    hipMemcpyDeviceToHost);
    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(ref_p.data(), aql_p.data(), partial_bytes, &first,
                             &diffs),
          "split partials host vs host-aql byte exact");
    if (diffs)
        std::printf("  split partials diff=%zu first=%zu\n", diffs, first);

    // ---- host reduce from the same partials snapshot ----
    check(kernel::launch_attention_paged_bf16_split_reduce(
              common, static_cast<const float*>(partials_host_run), kSplits,
              stream) == hipSuccess,
          "host production reduce launcher");
    check(hipStreamSynchronize(stream) == hipSuccess, "reduce sync");
    {
        auto region_res = queue.allocate_kernarg(
            mcu::aql_kernarg_slot_stride(reduce_meta.kernarg_segment_size), 1u);
        if (!region_res.ok()) {
            std::printf("FAIL: reduce kernarg: %s\n",
                        region_res.status().message().c_str());
            return;
        }
        auto region = region_res.release();
        unsigned char* slot = region.base;
        std::memset(slot, 0, region.slot_stride);
        kernel::PagedAttentionCommonArgs rc = common;
        rc.output = out_aql;
        kernel::PagedAttentionSplitReduceArgs args{};
        args.common = rc;
        args.partials = static_cast<const float*>(partials_aql);
        args.splits = kSplits;
        std::memcpy(slot, &args, sizeof(args));
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = reduce_meta.kernel_object;
        desc.global_work_items_x =
            reduce_grid * kernel::kPagedAttentionSplitBlock;
        desc.global_work_items_y = 1u;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x =
            static_cast<uint16_t>(kernel::kPagedAttentionSplitBlock);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = reduce_meta.private_segment_size;
        desc.group_segment_size = reduce_meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(args),
                                   reduce_meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, reduce_meta.kernarg_segment_size,
                                       reduce_grid, 1u, 1u,
                                       kernel::kPagedAttentionSplitBlock);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "reduce aql submit");
        check(wait_queue_drained(queue, queue.host_queued_index(), 15000),
              "reduce queue drained");
        (void)hipDeviceSynchronize();
    }
    std::vector<uint32_t> ref_o(out_bytes / 4u, 0u);
    std::vector<uint32_t> aql_o(out_bytes / 4u, 0u);
    (void)hipMemcpy(ref_o.data(), out_ref, out_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(aql_o.data(), out_aql, out_bytes, hipMemcpyDeviceToHost);
    first = 0;
    diffs = 0;
    check(chain::bytes_equal(ref_o.data(), aql_o.data(), out_bytes, &first, &diffs),
          "reduce output host vs host-aql byte exact");
    if (diffs)
        std::printf("  reduce output diff=%zu first=%zu\n", diffs, first);

    (void)split_code->shutdown();
    (void)reduce_code->shutdown();
    check(queue.shutdown().ok(), "host queue shutdown");

    // ---- MCU two-physical chain ----
    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize,
                  mcu::kMcuMaxDebugRecords, split_meta.kernarg_segment_size)) {
        std::printf("FAIL: rig open\n");
        return;
    }
    mcu::GpuAqlKernelMetadata mcu_split_meta{};
    mcu::GpuAqlKernelMetadata mcu_reduce_meta{};
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO,
                             split_symbol, mcu_split_meta))
        return;
    if (!rig.load_extra_code(PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO,
                             reduce_symbol, mcu_reduce_meta))
        return;

    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_kernel_variant(0u, mcu_split_meta,
                                 mcu::kMcuKernargRecipeAttentionPagedSplit,
                                 kernel::kPagedAttentionSplitBlock, split_grid,
                                 mcu::kMcuAgentAqlPolicy),
        mcu::make_kernel_variant(1u, mcu_reduce_meta,
                                 mcu::kMcuKernargRecipeAttentionPagedReduce,
                                 kernel::kPagedAttentionSplitBlock, reduce_grid,
                                 mcu::kMcuAgentAqlPolicy),
        mcu::make_probe_variant(2u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy)};

    std::vector<mcu::McuPlanNode> nodes(5);
    nodes[0].variant_id = 0u;
    nodes[0].kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedSplit;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0u;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1u;
    nodes[1].variant_id = 1u;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeAttentionPagedReduce;
    nodes[1].completion_slot = 0;
    nodes[1].invocation_index = 0u;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2u;
    nodes[2].variant_id = 2u;
    nodes[2].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[2].completion_slot = 0;
    nodes[2].input_slot = mcu::kMcuNoInput;
    nodes[2].flags = mcu::kMcuNodeDispatch;
    nodes[2].next = 3u;
    nodes[3].completion_slot = 0;
    nodes[3].flags = mcu::kMcuNodeWait;
    nodes[3].next = 4u;
    nodes[4].flags = mcu::kMcuNodeEnd;
    if (!rig.upload(nodes, variants)) return;

    mcu::McuPagedAttentionSplitInvocation sinv{};
    sinv.q = reinterpret_cast<uint64_t>(common.q);
    sinv.output = reinterpret_cast<uint64_t>(common.output);
    sinv.output_dtype = static_cast<uint32_t>(common.output_dtype);
    sinv.q_row_stride = common.q_row_stride;
    sinv.output_row_stride = common.output_row_stride;
    sinv.row_sequence_slots = reinterpret_cast<uint64_t>(common.row_sequence_slots);
    sinv.row_positions = reinterpret_cast<uint64_t>(common.row_positions);
    sinv.block_tables = reinterpret_cast<uint64_t>(common.block_tables);
    sinv.block_table_stride = common.block_table_stride;
    sinv.max_sequences = common.max_sequences;
    sinv.rows = common.rows;
    sinv.layer = common.layer;
    sinv.page_tokens = common.page_tokens;
    sinv.num_pages = common.num_pages;
    sinv.num_attention_layers = common.num_attention_layers;
    sinv.q_heads = common.q_heads;
    sinv.kv_heads = common.kv_heads;
    sinv.head_dim = common.head_dim;
    sinv.elems_per_token = common.elems_per_token;
    sinv.elems_per_page = common.elems_per_page;
    sinv.elems_per_layer = common.elems_per_layer;
    sinv.scale = common.scale;
    sinv.k_pool = reinterpret_cast<uint64_t>(k_pool);
    sinv.v_pool = reinterpret_cast<uint64_t>(v_pool);
    sinv.partials = reinterpret_cast<uint64_t>(partials_mcu);
    sinv.splits = kSplits;
    mcu::McuPagedAttentionReduceInvocation rinv{};
    rinv.q = reinterpret_cast<uint64_t>(common.q);
    rinv.output = reinterpret_cast<uint64_t>(out_mcu);
    rinv.output_dtype = static_cast<uint32_t>(common.output_dtype);
    rinv.q_row_stride = common.q_row_stride;
    rinv.output_row_stride = common.output_row_stride;
    rinv.row_sequence_slots = reinterpret_cast<uint64_t>(common.row_sequence_slots);
    rinv.row_positions = reinterpret_cast<uint64_t>(common.row_positions);
    rinv.block_tables = reinterpret_cast<uint64_t>(common.block_tables);
    rinv.block_table_stride = common.block_table_stride;
    rinv.max_sequences = common.max_sequences;
    rinv.rows = common.rows;
    rinv.layer = common.layer;
    rinv.page_tokens = common.page_tokens;
    rinv.num_pages = common.num_pages;
    rinv.num_attention_layers = common.num_attention_layers;
    rinv.q_heads = common.q_heads;
    rinv.kv_heads = common.kv_heads;
    rinv.head_dim = common.head_dim;
    rinv.elems_per_token = common.elems_per_token;
    rinv.elems_per_page = common.elems_per_page;
    rinv.elems_per_layer = common.elems_per_layer;
    rinv.scale = common.scale;
    rinv.partials = reinterpret_cast<uint64_t>(partials_mcu);
    rinv.splits = kSplits;
    if (!rig.set_attention_paged_split_invocations(&sinv, 1u)) return;
    if (!rig.set_attention_paged_reduce_invocations(&rinv, 1u)) return;
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x53504c54u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = 4u;
    config.doorbell_mode = static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    config.queue_ahead_depth = 4u;
    if (!rig.start(config)) {
        std::printf("FAIL: fsm start\n");
        return;
    }
    check(rig.fsm.request_run().ok(), "mcu chain requested");
    check(rig.fsm.wait_plans(1u, 30000).ok(), "mcu chain completed");
    check(rig.drain_queue(30000), "mcu chain drained");
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    check(rig.fsm.dispatches() == 3u,
          "three dispatches (split + reduce + marker)");
    check(rig.fsm.request_stop().ok(), "mcu stop requested");
    check(rig.fsm.wait_stopped(5000).ok(), "mcu stopped before readback");

    std::vector<uint32_t> mcu_p(partial_bytes / 4u, 0u);
    std::vector<uint32_t> mcu_o(out_bytes / 4u, 0u);
    rig.device_to_host(mcu_p.data(), partials_mcu, partial_bytes);
    rig.device_to_host(mcu_o.data(), out_mcu, out_bytes);
    first = 0;
    diffs = 0;
    check(chain::bytes_equal(ref_p.data(), mcu_p.data(), partial_bytes, &first,
                             &diffs),
          "mcu split partials match the host launcher");
    if (diffs)
        std::printf("  mcu partials diff=%zu first=%zu\n", diffs, first);
    first = 0;
    diffs = 0;
    check(chain::bytes_equal(ref_o.data(), mcu_o.data(), out_bytes, &first, &diffs),
          "mcu reduce output matches the host launcher");
    if (diffs)
        std::printf("  mcu reduce diff=%zu first=%zu\n", diffs, first);
    std::printf("  chain: partials=%zu B output=%zu B ok=%d\n", partial_bytes,
                out_bytes, static_cast<int>(diffs == 0u));

    check(rig.close(), "rig teardown");
    (void)hipStreamDestroy(stream);
    (void)hipFree(q_dev);
    (void)hipFree(k_pool);
    (void)hipFree(v_pool);
    (void)hipFree(partials_host_run);
    (void)hipFree(partials_aql);
    (void)hipFree(partials_mcu);
    (void)hipFree(out_ref);
    (void)hipFree(out_aql);
    (void)hipFree(out_mcu);
    (void)hipFree(row_slots);
    (void)hipFree(row_pos);
    (void)hipFree(block_tables);
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

    run_split_scenario(16u, "q4_full");
    run_split_scenario(24u, "other_q6");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_paged_attention_split: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_paged_attention_split: PASS\n");
    return 0;
}
