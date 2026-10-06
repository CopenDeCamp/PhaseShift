#include <phaseshift/models/qwen35/runtime/mcu_decode_runtime.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

static_assert(sizeof(::ps::runtime::gpu_mcu::McuGdnRecurrenceInvocation) ==
              sizeof(::ps::kernel::GdnRecurrenceArgs));
static_assert(offsetof(::ps::runtime::gpu_mcu::McuGdnRecurrenceInvocation,
                       capture_verify_only) ==
              offsetof(::ps::kernel::GdnRecurrenceArgs, capture_verify_only));

namespace ps {
namespace qwen35 {
namespace runtime {

namespace mcu = ::ps::runtime::gpu_mcu;

namespace {

template <typename T>
Status upload_table(void** dst, uint32_t* dst_count, const std::vector<T>& src) {
    *dst = nullptr;
    *dst_count = 0u;
    if (src.empty()) return Status::make_ok();
    const std::size_t bytes = src.size() * sizeof(T);
    void* ptr = nullptr;
    if (hipMalloc(&ptr, bytes) != hipSuccess) {
        return Status::hip_error("mcu invocation table allocation",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    if (hipMemcpy(ptr, src.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        const hipError_t err = hipGetLastError();
        (void)hipFree(ptr);
        return Status::hip_error("mcu invocation table upload",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    *dst = ptr;
    *dst_count = static_cast<uint32_t>(src.size());
    return Status::make_ok();
}

Status free_table(void** ptr, uint32_t* count) noexcept {
    if (*ptr == nullptr) {
        *count = 0u;
        return Status::make_ok();
    }
    void* p = *ptr;
    *ptr = nullptr;
    *count = 0u;
    const hipError_t err = hipFree(p);
    if (err != hipSuccess) {
        return Status::hip_error("mcu invocation table free",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

bool mcu_debug_records_disabled() {
    const char* e = std::getenv("PHASESHIFT_MCU_DEBUG_RECORDS");
    return e != nullptr && e[0] == '0';
}

}  // namespace

McuDecodeRuntime::~McuDecodeRuntime() { (void)shutdown(); }

Result<McuDecodeRuntime> McuDecodeRuntime::create(
    const McuDecodeRuntimeOptions& options) {
    McuDecodeRuntime out;
    out.device_ = options.device;
    out.queue_ahead_depth_ = options.queue_ahead_depth;
    out.driver_mode_ = options.driver_mode;

    auto part = mcu::GpuMcuCuPartition::create(options.device,
                                               options.cu_reservation);
    if (!part.ok()) {
        (void)out.shutdown();
        return part.status();
    }
    out.partition_ = part.release();
    out.partition_active_ = true;

    auto q = mcu::GpuMcuAqlQueue::create_with_mask(
        options.device, mcu::AqlQueueOwner::GpuMcu, options.queue_size,
        out.partition_.hsa_worker_cu_mask_view());
    if (!q.ok()) {
        (void)out.shutdown();
        return q.status();
    }
    out.queue_ = q.release();
    out.queue_active_ = true;

    const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
    auto probe = mcu::GpuMcuAqlCodeObject::load_memory(
        out.queue_, image.data, image.bytes, mcu::kGpuMcuFsmWorkerSymbol);
    if (!probe.ok()) {
        (void)out.shutdown();
        return probe.status();
    }
    out.probe_code_ = probe.release();
    out.probe_meta_ = out.probe_code_.metadata();

    const std::size_t node_bytes =
        static_cast<std::size_t>(mcu::kMcuMaxNodes) *
        sizeof(mcu::McuPlanNode);
    const std::size_t variant_bytes =
        static_cast<std::size_t>(mcu::kMcuMaxVariants) *
        sizeof(mcu::McuKernelVariantDesc);
    const std::size_t retained_bytes =
        static_cast<std::size_t>(mcu::kMcuMaxVariants) *
        sizeof(mcu::GpuMcuRetainedPacket);
    const uint32_t completions = 8u;
    if (hipMalloc(reinterpret_cast<void**>(&out.plan_nodes_), node_bytes) !=
            hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&out.plan_variants_),
                  variant_bytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&out.plan_retained_),
                  retained_bytes) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&out.completions_),
                  static_cast<std::size_t>(completions) *
                      sizeof(mcu::GpuMcuDeviceCompletion)) != hipSuccess) {
        (void)out.shutdown();
        return Status::hip_error("mcu plan buffers allocation",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    if (hipMalloc(reinterpret_cast<void**>(&out.plan_output_),
                  static_cast<std::size_t>(mcu::kMcuMaxNodes) *
                      sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&out.plan_timestamps_),
                  static_cast<std::size_t>(mcu::kMcuMaxNodes) * 2u *
                      sizeof(uint64_t)) != hipSuccess) {
        (void)out.shutdown();
        return Status::hip_error("mcu output buffers allocation",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    (void)hipMemset(out.plan_output_, 0,
                    static_cast<std::size_t>(mcu::kMcuMaxNodes) *
                        sizeof(uint32_t));
    (void)hipMemset(out.plan_timestamps_, 0,
                    static_cast<std::size_t>(mcu::kMcuMaxNodes) * 2u *
                        sizeof(uint64_t));
    out.completion_count_ = completions;
    (void)hipMemset(out.completions_, 0,
                    static_cast<std::size_t>(completions) *
                        sizeof(mcu::GpuMcuDeviceCompletion));
    (void)hipMemset(out.plan_nodes_, 0, node_bytes);
    (void)hipMemset(out.plan_variants_, 0, variant_bytes);
    (void)hipMemset(out.plan_retained_, 0, retained_bytes);

    if (options.record_entries != 0u && !mcu_debug_records_disabled()) {
        if (hipMalloc(reinterpret_cast<void**>(&out.records_),
                      static_cast<std::size_t>(options.record_entries) *
                          sizeof(mcu::McuDispatchRecord)) != hipSuccess) {
            (void)out.shutdown();
            return Status::hip_error("mcu record buffer allocation",
                                     hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        out.record_count_ = options.record_entries;
        (void)hipMemset(out.records_, 0,
                        static_cast<std::size_t>(options.record_entries) *
                            sizeof(mcu::McuDispatchRecord));
    }

    auto start_mem = ::ps::runtime::stream_signal_alloc(64u);
    if (!start_mem.ok()) {
        (void)out.shutdown();
        return start_mem.status();
    }
    out.start_signal_ = start_mem.value();
    auto done_mem = ::ps::runtime::stream_signal_alloc(64u);
    if (!done_mem.ok()) {
        (void)out.shutdown();
        return done_mem.status();
    }
    out.done_signal_ = done_mem.value();
    auto result_mem = ::ps::runtime::stream_signal_alloc(64u);
    if (!result_mem.ok()) {
        (void)out.shutdown();
        return result_mem.status();
    }
    out.result_signal_ = result_mem.value();

    return out;
}

Status McuDecodeRuntime::release_invocations() noexcept {
    Status status = Status::make_ok();
    status = free_table(&rms_, &rms_count_);
    Status s = free_table(&quant_, &quant_count_);
    if (!s.ok()) status = s;
    s = free_table(&e4m3_, &e4m3_count_);
    if (!s.ok()) status = s;
    s = free_table(&psq4_, &psq4_count_);
    if (!s.ok()) status = s;
    s = free_table(&psq4_multi_, &psq4_multi_count_);
    if (!s.ok()) status = s;
    s = free_table(&verify_accept_, &verify_accept_count_);
    if (!s.ok()) status = s;
    s = free_table(&verify_accept_batch_, &verify_accept_batch_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_spec_restore_, &gdn_spec_restore_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_spec_restore_from_counts_,
                   &gdn_spec_restore_from_counts_count_);
    if (!s.ok()) status = s;
    s = free_table(&argmax_f32_, &argmax_f32_count_);
    if (!s.ok()) status = s;
    s = free_table(&elementwise_, &elementwise_count_);
    if (!s.ok()) status = s;
    s = free_table(&rope_, &rope_count_);
    if (!s.ok()) status = s;
    s = free_table(&kv_append_, &kv_append_count_);
    if (!s.ok()) status = s;
    s = free_table(&paged_, &paged_count_);
    if (!s.ok()) status = s;
    s = free_table(&paged_split_, &paged_split_count_);
    if (!s.ok()) status = s;
    s = free_table(&paged_reduce_, &paged_reduce_count_);
    if (!s.ok()) status = s;
    s = free_table(&bf16_, &bf16_count_);
    if (!s.ok()) status = s;
    s = free_table(&bf16_wmma_, &bf16_wmma_count_);
    if (!s.ok()) status = s;
    s = free_table(&l2_, &l2_count_);
    if (!s.ok()) status = s;
    s = free_table(&embedding_, &embedding_count_);
    if (!s.ok()) status = s;
    s = free_table(&embedding_psq8_, &embedding_psq8_count_);
    if (!s.ok()) status = s;
    s = free_table(&output_gather_, &output_gather_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_conv_, &gdn_conv_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_recur_, &gdn_recur_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_reset_, &gdn_reset_count_);
    if (!s.ok()) status = s;
    if (row_global_patches_ != nullptr) {
        if (hipFree(row_global_patches_) != hipSuccess) {
            status = Status::hip_error(
                "row global patch free", hipGetErrorString(hipGetLastError()),
                __FILE__, __LINE__);
        }
        row_global_patches_ = nullptr;
        row_global_patch_count_ = 0u;
    }
    if (dynamic_bindings_ != nullptr) {
        if (hipFree(dynamic_bindings_) != hipSuccess) {
            status = Status::hip_error(
                "dynamic binding free", hipGetErrorString(hipGetLastError()),
                __FILE__, __LINE__);
        }
        dynamic_bindings_ = nullptr;
        dynamic_binding_count_ = 0u;
    }
    return status;
}

Status McuDecodeRuntime::rebuild_row_global_patches() noexcept {
    if (row_global_patches_ != nullptr) {
        (void)hipFree(row_global_patches_);
        row_global_patches_ = nullptr;
        row_global_patch_count_ = 0u;
    }
    uint32_t geometry_patch_count = 0u;
    if (dynamic_bindings_ != nullptr) {
        for (const McuGeometryPatch& patch : geometry_patches_) {
            if (patch.node_index < dynamic_binding_count_) {
                ++geometry_patch_count;
            }
        }
    }
    const uint32_t count = 2u * gdn_conv_count_ + 3u * gdn_recur_count_ +
                           argmax_f32_count_ + output_gather_count_ +
                           verify_accept_count_ +
                           static_cast<uint32_t>(attention_regions_.size()) +
                           static_cast<uint32_t>(bf16_catalogs_.size()) +
                           geometry_patch_count +
                           static_cast<uint32_t>(
                               invocation_row_patches_.size());
    const auto row_table = [&](McuInvocationTable table, const void*& base,
                               std::size_t& stride, std::size_t& offset,
                               uint32_t& limit) -> bool {
        switch (table) {
            case McuInvocationTable::RmsNorm:
                base = rms_;
                limit = rms_count_;
                stride = sizeof(mcu::McuRmsNormInvocation);
                offset = offsetof(mcu::McuRmsNormInvocation, rows);
                return true;
            case McuInvocationTable::Elementwise:
                base = elementwise_;
                limit = elementwise_count_;
                stride = sizeof(mcu::McuElementwiseInvocation);
                offset = offsetof(mcu::McuElementwiseInvocation, rows);
                return true;
            case McuInvocationTable::KvAppend:
                base = kv_append_;
                limit = kv_append_count_;
                stride = sizeof(mcu::McuKvAppendInvocation);
                offset = offsetof(mcu::McuKvAppendInvocation, rows);
                return true;
            case McuInvocationTable::EmbeddingBf16:
                base = embedding_;
                limit = embedding_count_;
                stride = sizeof(mcu::McuEmbeddingBf16Invocation);
                offset = offsetof(mcu::McuEmbeddingBf16Invocation, rows);
                return true;
            case McuInvocationTable::EmbeddingPsq8:
                base = embedding_psq8_;
                limit = embedding_psq8_count_;
                stride = sizeof(mcu::McuEmbeddingPsq8Invocation);
                offset = offsetof(mcu::McuEmbeddingPsq8Invocation, rows);
                return true;
            case McuInvocationTable::Bf16Wmma:
                base = bf16_wmma_;
                limit = bf16_wmma_count_;
                stride = sizeof(mcu::McuBf16WmmaInvocation);
                offset = offsetof(mcu::McuBf16WmmaInvocation, rows);
                return true;
            case McuInvocationTable::Psq4MultiRow:
                base = psq4_multi_;
                limit = psq4_multi_count_;
                stride = sizeof(mcu::McuPsq4MultiRowInvocation);
                offset = offsetof(mcu::McuPsq4MultiRowInvocation, rows);
                return true;
        }
        return false;
    };
    if (count == 0u) return Status::make_ok();
    std::vector<mcu::McuInvocationPatch> patches(count);
    uint32_t index = 0u;
    auto* gdn_base = static_cast<unsigned char*>(gdn_conv_);
    for (uint32_t i = 0u; i < gdn_conv_count_; ++i) {
        patches[index].target = reinterpret_cast<uint64_t>(
            gdn_base +
            static_cast<size_t>(i) * sizeof(mcu::McuGdnConv1dInvocation) +
            offsetof(mcu::McuGdnConv1dInvocation, actual_rows));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::ActualRows);
        ++index;
    }
    for (uint32_t i = 0u; i < gdn_conv_count_; ++i) {
        patches[index].target = reinterpret_cast<uint64_t>(
            gdn_base +
            static_cast<size_t>(i) * sizeof(mcu::McuGdnConv1dInvocation) +
            offsetof(mcu::McuGdnConv1dInvocation, capture_verify_only));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::VerifyRequests);
        ++index;
    }
    auto* recur_base = static_cast<unsigned char*>(gdn_recur_);
    for (uint32_t i = 0u; i < gdn_recur_count_; ++i) {
        auto* invocation = reinterpret_cast<mcu::McuGdnRecurrenceInvocation*>(
            recur_base + static_cast<size_t>(i) *
                             sizeof(mcu::McuGdnRecurrenceInvocation));
        patches[index].target = reinterpret_cast<uint64_t>(
            reinterpret_cast<unsigned char*>(invocation) +
            offsetof(mcu::McuGdnRecurrenceInvocation, capture_verify_only));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::VerifyRequests);
        ++index;
        patches[index].target = reinterpret_cast<uint64_t>(
            reinterpret_cast<unsigned char*>(invocation) +
            offsetof(::ps::kernel::GdnRecurrenceArgs, actual_rows));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::ActualRows);
        ++index;
        patches[index].target = reinterpret_cast<uint64_t>(
            reinterpret_cast<unsigned char*>(invocation) +
            offsetof(::ps::kernel::GdnRecurrenceArgs, num_requests));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::NumRequests);
        ++index;
    }
    auto* argmax_base = static_cast<unsigned char*>(argmax_f32_);
    for (uint32_t i = 0u; i < argmax_f32_count_; ++i) {
        patches[index].target = reinterpret_cast<uint64_t>(
            argmax_base +
            static_cast<size_t>(i) * sizeof(mcu::McuArgmaxF32Invocation) +
            offsetof(mcu::McuArgmaxF32Invocation, outputs));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::NumOutputs);
        ++index;
    }
    auto* gather_base = static_cast<unsigned char*>(output_gather_);
    for (uint32_t i = 0u; i < output_gather_count_; ++i) {
        patches[index].target = reinterpret_cast<uint64_t>(
            gather_base +
            static_cast<size_t>(i) *
                sizeof(mcu::McuOutputGatherBf16Invocation) +
            offsetof(mcu::McuOutputGatherBf16Invocation, num_outputs));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::NumOutputs);
        ++index;
    }
    auto* verify_base = static_cast<unsigned char*>(verify_accept_);
    for (uint32_t i = 0u; i < verify_accept_count_; ++i) {
        patches[index].target = reinterpret_cast<uint64_t>(
            verify_base +
            static_cast<size_t>(i) *
                sizeof(mcu::McuVerifyAcceptPrefixInvocation) +
            offsetof(mcu::McuVerifyAcceptPrefixInvocation, request_count));
        patches[index].source =
            static_cast<uint8_t>(mcu::McuInvocationPatchSource::NumRequests);
        ++index;
    }
    auto* paged_base = static_cast<unsigned char*>(paged_);
    auto* split_base = static_cast<unsigned char*>(paged_split_);
    auto* reduce_base = static_cast<unsigned char*>(paged_reduce_);
    for (const McuAttentionRegion& region : attention_regions_) {
        uint64_t target = 0u;
        if (region.invocation_kind == 0u) {
            target = reinterpret_cast<uint64_t>(
                paged_base +
                static_cast<size_t>(region.invocation_index) *
                    sizeof(mcu::McuPagedAttentionInvocation) +
                offsetof(mcu::McuPagedAttentionInvocation, rows));
        } else if (region.invocation_kind == 1u) {
            target = reinterpret_cast<uint64_t>(
                split_base +
                static_cast<size_t>(region.invocation_index) *
                    sizeof(mcu::McuPagedAttentionSplitInvocation) +
                offsetof(mcu::McuPagedAttentionSplitInvocation, rows));
        } else {
            target = reinterpret_cast<uint64_t>(
                reduce_base +
                static_cast<size_t>(region.invocation_index) *
                    sizeof(mcu::McuPagedAttentionReduceInvocation) +
                offsetof(mcu::McuPagedAttentionReduceInvocation, rows));
        }
        patches[index].target = target;
        patches[index].source = static_cast<uint8_t>(
            mcu::McuInvocationPatchSource::AttentionRegionRows);
        patches[index].param =
            (region.row_begin & 0xFFFFu) | (region.rows << 16);
        ++index;
    }
    auto* binding_base = static_cast<unsigned char*>(dynamic_bindings_);
    for (const McuBf16VariantCatalog& catalog : bf16_catalogs_) {
        patches[index].target = reinterpret_cast<uint64_t>(
            binding_base +
            static_cast<size_t>(catalog.node_index) *
                sizeof(mcu::McuDynamicNodeBinding) +
            offsetof(mcu::McuDynamicNodeBinding, variant_override));
        patches[index].source = static_cast<uint8_t>(
            mcu::McuInvocationPatchSource::Bf16ExactRowsVariant);
        patches[index].param = catalog.variant_base;
        ++index;
    }
    for (const McuGeometryPatch& patch : geometry_patches_) {
        if (dynamic_bindings_ == nullptr ||
            patch.node_index >= dynamic_binding_count_) {
            continue;
        }
        std::size_t offset = offsetof(mcu::McuDynamicNodeBinding,
                                      workgroup_count_x);
        if (patch.axis == McuGeometryAxis::Y) {
            offset = offsetof(mcu::McuDynamicNodeBinding, workgroup_count_y);
        } else if (patch.axis == McuGeometryAxis::Z) {
            offset = offsetof(mcu::McuDynamicNodeBinding, workgroup_count_z);
        }
        patches[index].target = reinterpret_cast<uint64_t>(
            binding_base +
            static_cast<std::size_t>(patch.node_index) *
                sizeof(mcu::McuDynamicNodeBinding) +
            offset);
        patches[index].source = patch.source;
        patches[index].param = patch.param;
        ++index;
    }
    for (const McuInvocationRowPatch& patch : invocation_row_patches_) {
        const void* base = nullptr;
        std::size_t stride = 0u;
        std::size_t offset = 0u;
        uint32_t limit = 0u;
        if (!row_table(patch.table, base, stride, offset, limit)) continue;
        if (base == nullptr || patch.index >= limit) continue;
        patches[index].target = reinterpret_cast<uint64_t>(
            static_cast<const unsigned char*>(base) +
            static_cast<std::size_t>(patch.index) * stride + offset);
        patches[index].source = patch.source;
        ++index;
    }
    if (index != count) {
        return Status::invalid_state("row global patch count mismatch",
                                     __FILE__, __LINE__);
    }
    if (hipMalloc(reinterpret_cast<void**>(&row_global_patches_),
                  patches.size() * sizeof(mcu::McuInvocationPatch)) !=
        hipSuccess) {
        row_global_patches_ = nullptr;
        return Status::hip_error("row global patch allocation",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    if (hipMemcpy(row_global_patches_, patches.data(),
                  patches.size() * sizeof(mcu::McuInvocationPatch),
                  hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(row_global_patches_);
        row_global_patches_ = nullptr;
        return Status::hip_error("row global patch upload",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    row_global_patch_count_ = count;
    return Status::make_ok();
}

Status McuDecodeRuntime::clear_plan() noexcept {
    Status status = Status::make_ok();
    if (fsm_active_) {
            const Status stop = fsm_.request_stop();
        if (!stop.ok()) status = stop;
            const Status waited = fsm_.wait_stopped(5000u);
        if (!waited.ok()) status = waited;
            const Status down = fsm_.shutdown();
        if (!down.ok()) status = down;
            fsm_active_ = false;
    }
    const Status inv = release_invocations();
    if (!inv.ok()) status = inv;
    code_objects_.clear();
    plan_node_count_ = 0u;
    plan_variant_count_ = 0u;
    return status;
}

Status McuDecodeRuntime::prepare_plan(const McuCompiledPlan& plan) {
    if (!queue_active_ || !partition_active_) {
        return Status::invalid_state("mcu runtime not created", __FILE__,
                                     __LINE__);
    }
    if (plan.nodes.empty()) {
        return Status::invalid_argument("mcu plan has no nodes", __FILE__,
                                        __LINE__);
    }
    if (plan.nodes.size() > mcu::kMcuMaxNodes ||
        plan.variants.size() > mcu::kMcuMaxVariants) {
        return Status::invalid_argument("mcu plan exceeds table capacity",
                                        __FILE__, __LINE__);
    }

    const Status cleared = clear_plan();
    if (!cleared.ok()) return cleared;

    std::vector<mcu::McuKernelVariantDesc> variants(plan.variants.size());
    std::vector<mcu::GpuMcuRetainedPacket> retained(plan.variants.size());
    uint32_t needed_segment = sizeof(mcu::GpuMcuFsmWorkerArgs);
    for (std::size_t i = 0; i < plan.variants.size(); ++i) {
        const McuCompiledVariant& v = plan.variants[i];
        if (v.kind == McuCompiledVariantKind::CompletionMarker) {
            variants[i] = mcu::make_probe_variant(
                static_cast<uint16_t>(i), probe_meta_, v.workgroup, v.grid,
                mcu::kMcuSystemAqlPolicy);
            retained[i] = variants[i].packet;
            needed_segment =
                std::max(needed_segment, probe_meta_.kernarg_segment_size);
            continue;
        }
        auto co = mcu_kernel_code_object(v.kind);
        if (!co.ok()) return co.status();
        auto loaded = mcu::GpuMcuAqlCodeObject::load_memory(
            queue_, co.value().blob.blob, co.value().blob.size,
            co.value().symbol);
        if (!loaded.ok()) return loaded.status();
        auto object =
            std::make_unique<mcu::GpuMcuAqlCodeObject>(loaded.release());
        const mcu::GpuAqlKernelMetadata meta = object->metadata();
        variants[i] = mcu::make_kernel_variant(
            static_cast<uint16_t>(i), meta, co.value().kernarg_recipe,
            v.workgroup, v.grid, mcu::kMcuAgentAqlPolicy,
            co.value().hidden_args_policy, v.grid_y, v.grid_z);
        retained[i] = variants[i].packet;
        needed_segment = std::max(needed_segment, meta.kernarg_segment_size);
        code_objects_.push_back(std::move(object));
    }
    plan_variant_count_ = static_cast<uint32_t>(plan.variants.size());

    if (hipMemcpy(plan_nodes_, plan.nodes.data(),
                  plan.nodes.size() * sizeof(mcu::McuPlanNode),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(plan_variants_, variants.data(),
                  variants.size() * sizeof(mcu::McuKernelVariantDesc),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(plan_retained_, retained.data(),
                  retained.size() * sizeof(mcu::GpuMcuRetainedPacket),
                  hipMemcpyHostToDevice) != hipSuccess) {
        return Status::hip_error("mcu plan upload",
                                 hipGetErrorString(hipGetLastError()), __FILE__,
                                 __LINE__);
    }
    plan_node_count_ = static_cast<uint32_t>(plan.nodes.size());

    Status status = upload_table(&rms_, &rms_count_, plan.rms);
    if (status.ok()) status = upload_table(&e4m3_, &e4m3_count_, plan.quant_e4m3);
    if (status.ok()) status = upload_table(&psq4_, &psq4_count_, plan.psq4);
    if (status.ok())
        status = upload_table(&psq4_multi_, &psq4_multi_count_, plan.psq4_multi);
    if (status.ok())
        status = upload_table(&verify_accept_, &verify_accept_count_,
                              plan.verify_accept);
    if (status.ok())
        status = upload_table(&verify_accept_batch_,
                              &verify_accept_batch_count_,
                              plan.verify_accept_batch);
    if (status.ok())
        status = upload_table(&gdn_spec_restore_, &gdn_spec_restore_count_,
                              plan.gdn_spec_restore);
    if (status.ok())
        status = upload_table(&gdn_spec_restore_from_counts_,
                              &gdn_spec_restore_from_counts_count_,
                              plan.gdn_spec_restore_from_counts);
    if (status.ok())
        status = upload_table(&argmax_f32_, &argmax_f32_count_,
                              plan.argmax_f32);
    if (status.ok())
        status = upload_table(&elementwise_, &elementwise_count_,
                              plan.elementwise);
    if (status.ok()) status = upload_table(&rope_, &rope_count_, plan.rope);
    if (status.ok())
        status = upload_table(&kv_append_, &kv_append_count_, plan.kv_append);
    if (status.ok())
        status = upload_table(&paged_, &paged_count_, plan.attention_paged);
    if (status.ok())
        status = upload_table(&paged_split_, &paged_split_count_,
                              plan.attention_paged_split);
    if (status.ok())
        status = upload_table(&paged_reduce_, &paged_reduce_count_,
                              plan.attention_paged_reduce);
    if (status.ok()) status = upload_table(&bf16_, &bf16_count_, plan.bf16);
    if (status.ok())
        status = upload_table(&bf16_wmma_, &bf16_wmma_count_, plan.bf16_wmma);
    if (status.ok()) status = upload_table(&l2_, &l2_count_, plan.l2);
    if (status.ok())
        status = upload_table(&embedding_, &embedding_count_, plan.embedding);
    if (status.ok())
        status = upload_table(&embedding_psq8_, &embedding_psq8_count_,
                              plan.embedding_psq8);
    if (status.ok())
        status = upload_table(&output_gather_, &output_gather_count_,
                              plan.output_gather);
    if (status.ok())
        status = upload_table(&gdn_conv_, &gdn_conv_count_, plan.gdn_conv1d);
    if (status.ok())
        status = upload_table(&gdn_recur_, &gdn_recur_count_,
                              plan.gdn_recurrence);
    if (status.ok())
        status = upload_table(&gdn_reset_, &gdn_reset_count_, plan.gdn_reset);
    if (!status.ok()) {
        (void)release_invocations();
        return status;
    }
    attention_regions_ = plan.attention_regions;
    bf16_catalogs_ = plan.bf16_variant_catalogs;
    geometry_patches_ = plan.geometry_patches;
    invocation_row_patches_ = plan.invocation_row_patches;
    if (!plan.nodes.empty()) {
        std::vector<mcu::McuDynamicNodeBinding> bindings(plan.nodes.size());
        for (std::size_t i = 0; i < plan.nodes.size(); ++i) {
            bindings[i].enabled = 1u;
            bindings[i].variant_id = plan.nodes[i].variant_id;
            bindings[i].invocation_index = plan.nodes[i].invocation_index;
        }
        const std::size_t bytes =
            bindings.size() * sizeof(mcu::McuDynamicNodeBinding);
        if (hipMalloc(&dynamic_bindings_, bytes) != hipSuccess) {
            dynamic_bindings_ = nullptr;
            (void)release_invocations();
            return Status::hip_error(
                "dynamic binding allocation",
                hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
        }
        if (hipMemcpy(dynamic_bindings_, bindings.data(), bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            (void)hipFree(dynamic_bindings_);
            dynamic_bindings_ = nullptr;
            (void)release_invocations();
            return Status::hip_error(
                "dynamic binding upload",
                hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
        }
        dynamic_binding_count_ = static_cast<uint32_t>(bindings.size());
    }
    status = rebuild_row_global_patches();
    if (!status.ok()) {
        (void)release_invocations();
        return status;
    }

    const std::size_t slot_bytes =
        mcu::aql_kernarg_slot_stride(needed_segment);
    auto region = queue_.allocate_kernarg(slot_bytes, plan_node_count_ + 1u);
    if (!region.ok()) return region.status();
    kernarg_ = region.release();

    if (log_region_.base == nullptr) {
        const char* log_env = std::getenv("PHASESHIFT_MCU_LOG");
        if (log_env != nullptr && log_env[0] == '1') {
            auto log_res = queue_.allocate_log_region(mcu::kMcuLogBytes);
            if (log_res.ok()) log_region_ = log_res.release();
        }
    }

    mcu::GpuMcuFsmConfig config{};
    config.queue = queue_.device_queue_view();
    config.plan = plan_nodes_;
    config.node_count = plan_node_count_;
    config.plan_id = 0u;
    config.variants = plan_variants_;
    config.variant_count = plan_variant_count_;
    config.rmsnorm_invocations = static_cast<const mcu::McuRmsNormInvocation*>(rms_);
    config.rmsnorm_invocation_count = rms_count_;
    config.quantize_invocations =
        static_cast<const mcu::McuActivationQuantizeInvocation*>(quant_);
    config.quantize_invocation_count = quant_count_;
    config.e4m3_invocations =
        static_cast<const mcu::McuActivationQuantizeE4m3Invocation*>(e4m3_);
    config.e4m3_invocation_count = e4m3_count_;
    config.psq4_invocations =
        static_cast<const mcu::McuPsq4Decode1Invocation*>(psq4_);
    config.psq4_invocation_count = psq4_count_;
    config.psq4_multi_invocations =
        static_cast<const mcu::McuPsq4MultiRowInvocation*>(psq4_multi_);
    config.psq4_multi_invocation_count = psq4_multi_count_;
    config.verify_accept_invocations =
        static_cast<const mcu::McuVerifyAcceptPrefixInvocation*>(verify_accept_);
    config.verify_accept_invocation_count = verify_accept_count_;
    config.verify_accept_batch_invocations =
        static_cast<const mcu::McuVerifyAcceptBatchInvocation*>(
            verify_accept_batch_);
    config.verify_accept_batch_invocation_count = verify_accept_batch_count_;
    config.gdn_spec_restore_invocations =
        static_cast<const mcu::McuGdnSpecRestoreInvocation*>(gdn_spec_restore_);
    config.gdn_spec_restore_invocation_count = gdn_spec_restore_count_;
    config.gdn_spec_restore_from_counts_invocations =
        static_cast<const mcu::McuGdnSpecRestoreFromCountsInvocation*>(
            gdn_spec_restore_from_counts_);
    config.gdn_spec_restore_from_counts_invocation_count =
        gdn_spec_restore_from_counts_count_;
    config.argmax_f32_invocations =
        static_cast<const mcu::McuArgmaxF32Invocation*>(argmax_f32_);
    config.argmax_f32_invocation_count = argmax_f32_count_;
    config.elementwise_invocations =
        static_cast<const mcu::McuElementwiseInvocation*>(elementwise_);
    config.elementwise_invocation_count = elementwise_count_;
    config.rope_invocations = static_cast<const mcu::McuRopeInvocation*>(rope_);
    config.rope_invocation_count = rope_count_;
    config.kv_append_invocations =
        static_cast<const mcu::McuKvAppendInvocation*>(kv_append_);
    config.kv_append_invocation_count = kv_append_count_;
    config.attention_paged_invocations =
        static_cast<const mcu::McuPagedAttentionInvocation*>(paged_);
    config.attention_paged_invocation_count = paged_count_;
    config.attention_paged_split_invocations =
        static_cast<const mcu::McuPagedAttentionSplitInvocation*>(paged_split_);
    config.attention_paged_split_invocation_count = paged_split_count_;
    config.attention_paged_reduce_invocations =
        static_cast<const mcu::McuPagedAttentionReduceInvocation*>(paged_reduce_);
    config.attention_paged_reduce_invocation_count = paged_reduce_count_;
    config.bf16_invocations =
        static_cast<const mcu::McuBf16ExactRowsInvocation*>(bf16_);
    config.bf16_invocation_count = bf16_count_;
    config.bf16_wmma_invocations =
        static_cast<const mcu::McuBf16WmmaInvocation*>(bf16_wmma_);
    config.bf16_wmma_invocation_count = bf16_wmma_count_;
    config.l2_invocations = static_cast<const mcu::McuL2NormalizeInvocation*>(l2_);
    config.l2_invocation_count = l2_count_;
    config.embedding_invocations =
        static_cast<const mcu::McuEmbeddingBf16Invocation*>(embedding_);
    config.embedding_invocation_count = embedding_count_;
    config.embedding_psq8_invocations =
        static_cast<const mcu::McuEmbeddingPsq8Invocation*>(embedding_psq8_);
    config.embedding_psq8_invocation_count = embedding_psq8_count_;
    config.output_gather_invocations =
        static_cast<const mcu::McuOutputGatherBf16Invocation*>(output_gather_);
    config.output_gather_invocation_count = output_gather_count_;
    config.gdn_conv1d_invocations =
        static_cast<const mcu::McuGdnConv1dInvocation*>(gdn_conv_);
    config.gdn_conv1d_invocation_count = gdn_conv_count_;
    config.gdn_recurrence_invocations =
        static_cast<const mcu::McuGdnRecurrenceInvocation*>(gdn_recur_);
    config.gdn_recurrence_invocation_count = gdn_recur_count_;
    config.gdn_reset_invocations =
        static_cast<const mcu::GpuMcuGdnResetInvocation*>(gdn_reset_);
    config.gdn_reset_invocation_count = gdn_reset_count_;
    config.retained = plan_retained_;
    config.retained_count = plan_variant_count_;
    config.kernarg_base = reinterpret_cast<uint64_t>(kernarg_.base);
    config.kernarg_slot_stride = kernarg_.slot_stride;
    config.kernarg_slot_count = plan_node_count_ + 1u;
    config.dynamic_node_bindings =
        static_cast<const mcu::McuDynamicNodeBinding*>(dynamic_bindings_);
    config.dynamic_node_binding_count = dynamic_binding_count_;
    config.log_base = reinterpret_cast<uint64_t>(log_region_.base);
    config.output_base = reinterpret_cast<uint64_t>(plan_output_);
    config.timestamps_base = reinterpret_cast<uint64_t>(plan_timestamps_);
    config.completions = completions_;
    config.completion_count = completion_count_;
    config.timing = nullptr;
    config.timing_count = 0u;
    config.debug_records = records_;
    config.debug_record_count = record_count_;
    config.barrier = 1u;
    config.queue_ahead_depth = queue_ahead_depth_;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    if (const char* e = std::getenv("PHASESHIFT_MCU_QUEUE_AHEAD")) {
        const unsigned long v = std::strtoul(e, nullptr, 10);
        if (v >= 1ul && v <= 256ul)
            config.queue_ahead_depth = static_cast<uint32_t>(v);
    }
    if (const char* e = std::getenv("PHASESHIFT_MCU_DOORBELL")) {
        if (std::strcmp(e, "coalesce") == 0)
            config.doorbell_mode =
                static_cast<uint32_t>(mcu::McuDoorbellMode::coalesce);
        else if (std::strcmp(e, "first_only") == 0)
            config.doorbell_mode =
                static_cast<uint32_t>(mcu::McuDoorbellMode::first_only);
    }
    config.issue_wait_running = 0u;
    config.template_source = 0u;
    config.prepared_enabled = 0u;
    if (!external_driver()) {
        config.start_signal = start_signal_.device_ptr;
        config.done_signal = done_signal_.device_ptr;
        config.result_code = result_signal_.device_ptr;
    }

    auto created = mcu::GpuMcuFsm::create(device_, partition_.control_stream());
    if (!created.ok()) return created.status();
    fsm_ = created.release();
    status = fsm_.configure(config);
    if (!status.ok()) return status;
    if (!external_driver()) {
        status = fsm_.start();
        if (!status.ok()) return status;
    }
    fsm_active_ = true;
    return Status::make_ok();
}

Status McuDecodeRuntime::enqueue_run(hipStream_t stream, uint32_t epoch) const {
    if (external_driver()) {
        return Status::invalid_state(
            "mcu plan uses external persistent driver", __FILE__, __LINE__);
    }
    if (!fsm_active_) {
        return Status::invalid_state("mcu plan not prepared", __FILE__,
                                     __LINE__);
    }
    return ::ps::runtime::stream_write_value32(stream, start_signal_, epoch);
}

Status McuDecodeRuntime::enqueue_wait(hipStream_t stream, uint32_t epoch) const {
    if (external_driver()) {
        return Status::invalid_state(
            "mcu plan uses external persistent driver", __FILE__, __LINE__);
    }
    if (!fsm_active_) {
        return Status::invalid_state("mcu plan not prepared", __FILE__,
                                     __LINE__);
    }
    return ::ps::runtime::stream_wait_value32(stream, done_signal_, epoch);
}

void McuDecodeRuntime::dump_log(std::FILE* out) const noexcept {
    if (out == nullptr) return;
    static const char* const kEventNames[] = {
        "None",          "PlanBegin",          "DispatchPrepare",
        "DispatchPublish", "DispatchDoorbell", "WaitBegin",
        "WaitComplete",  "WaitSpinTimeout",    "CompletionSlotMismatch",
        "KernargBuildFailed", "Fault", "PlanEnd", "ExecutionSkip",
        "ExecutionEpoch"};
    const uint32_t used = fsm_.log_head();
    if (used == 0u) {
        std::fprintf(out, "[mcu-log] no records\n");
        return;
    }
    const uint64_t lo =
        used > mcu::kMcuLogLineCount
            ? static_cast<uint64_t>(used) - mcu::kMcuLogLineCount
            : 0ull;
    const uint32_t show = used < 64u ? used : 64u;
    std::fprintf(out, "[mcu-log] records=%u newest=%u\n", used, used - 1u);
    for (uint32_t i = 0u; i < show; ++i) {
        const uint32_t slot = (used - 1u - i) & (mcu::kMcuLogLineCount - 1u);
        const mcu::McuLogRecord* rec = fsm_.log_record(slot);
        if (rec == nullptr) continue;
        if (rec->sequence < lo ||
            rec->sequence >= static_cast<uint64_t>(used))
            continue;
        const uint32_t ev = rec->event;
        const char* name =
            ev < sizeof(kEventNames) / sizeof(kEventNames[0]) ? kEventNames[ev]
                                                              : "?";
        std::fprintf(out,
                     "  seq=%llu ev=%s pc=%u var=%u cslot=%u gen=%llu "
                     "v0=%llu v1=%llu\n",
                     static_cast<unsigned long long>(rec->sequence), name,
                     rec->pc, rec->variant, rec->completion_slot,
                     static_cast<unsigned long long>(rec->generation),
                     static_cast<unsigned long long>(rec->value0),
                     static_cast<unsigned long long>(rec->value1));
    }
}

uint32_t McuDecodeRuntime::result_code() const noexcept {
    return ::ps::runtime::stream_signal_load(result_signal_);
}

Status McuDecodeRuntime::shutdown() noexcept {
    Status status = clear_plan();

    Status s = free_table(&rms_, &rms_count_);
    if (!s.ok()) status = s;
    s = free_table(&quant_, &quant_count_);
    if (!s.ok()) status = s;
    s = free_table(&e4m3_, &e4m3_count_);
    if (!s.ok()) status = s;
    s = free_table(&psq4_, &psq4_count_);
    if (!s.ok()) status = s;
    s = free_table(&psq4_multi_, &psq4_multi_count_);
    if (!s.ok()) status = s;
    s = free_table(&verify_accept_, &verify_accept_count_);
    if (!s.ok()) status = s;
    s = free_table(&verify_accept_batch_, &verify_accept_batch_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_spec_restore_, &gdn_spec_restore_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_spec_restore_from_counts_,
                   &gdn_spec_restore_from_counts_count_);
    if (!s.ok()) status = s;
    s = free_table(&argmax_f32_, &argmax_f32_count_);
    if (!s.ok()) status = s;
    s = free_table(&elementwise_, &elementwise_count_);
    if (!s.ok()) status = s;
    s = free_table(&rope_, &rope_count_);
    if (!s.ok()) status = s;
    s = free_table(&kv_append_, &kv_append_count_);
    if (!s.ok()) status = s;
    s = free_table(&paged_, &paged_count_);
    if (!s.ok()) status = s;
    s = free_table(&paged_split_, &paged_split_count_);
    if (!s.ok()) status = s;
    s = free_table(&paged_reduce_, &paged_reduce_count_);
    if (!s.ok()) status = s;
    s = free_table(&bf16_, &bf16_count_);
    if (!s.ok()) status = s;
    s = free_table(&bf16_wmma_, &bf16_wmma_count_);
    if (!s.ok()) status = s;
    s = free_table(&l2_, &l2_count_);
    if (!s.ok()) status = s;
    s = free_table(&embedding_, &embedding_count_);
    if (!s.ok()) status = s;
    s = free_table(&embedding_psq8_, &embedding_psq8_count_);
    if (!s.ok()) status = s;
    s = free_table(&output_gather_, &output_gather_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_conv_, &gdn_conv_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_recur_, &gdn_recur_count_);
    if (!s.ok()) status = s;
    s = free_table(&gdn_reset_, &gdn_reset_count_);
    if (!s.ok()) status = s;
    if (row_global_patches_ != nullptr) {
        if (hipFree(row_global_patches_) != hipSuccess) {
            status = Status::hip_error(
                "row global patch free", hipGetErrorString(hipGetLastError()),
                __FILE__, __LINE__);
        }
        row_global_patches_ = nullptr;
        row_global_patch_count_ = 0u;
    }
    if (dynamic_bindings_ != nullptr) {
        if (hipFree(dynamic_bindings_) != hipSuccess) {
            status = Status::hip_error(
                "dynamic binding free", hipGetErrorString(hipGetLastError()),
                __FILE__, __LINE__);
        }
        dynamic_bindings_ = nullptr;
        dynamic_binding_count_ = 0u;
    }

    if (plan_nodes_ != nullptr) {
        s = hipFree(plan_nodes_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("plan node free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        plan_nodes_ = nullptr;
        if (!s.ok()) status = s;
    }
    if (plan_variants_ != nullptr) {
        s = hipFree(plan_variants_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("plan variant free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        plan_variants_ = nullptr;
        if (!s.ok()) status = s;
    }
    if (plan_retained_ != nullptr) {
        s = hipFree(plan_retained_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("plan retained free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        plan_retained_ = nullptr;
        if (!s.ok()) status = s;
    }
    if (completions_ != nullptr) {
        s = hipFree(completions_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("completion free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        completions_ = nullptr;
        if (!s.ok()) status = s;
    }
    if (plan_output_ != nullptr) {
        s = hipFree(plan_output_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("plan output free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        plan_output_ = nullptr;
        if (!s.ok()) status = s;
    }
    if (plan_timestamps_ != nullptr) {
        s = hipFree(plan_timestamps_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("plan timestamp free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        plan_timestamps_ = nullptr;
        if (!s.ok()) status = s;
    }
    if (records_ != nullptr) {
        s = hipFree(records_) == hipSuccess
                ? Status::make_ok()
                : Status::hip_error("record free",
                                    hipGetErrorString(hipGetLastError()),
                                    __FILE__, __LINE__);
        records_ = nullptr;
        if (!s.ok()) status = s;
    }

    s = ::ps::runtime::stream_signal_free(start_signal_);
    if (!s.ok()) status = s;
    start_signal_ = ::ps::runtime::StreamSignal{};
    s = ::ps::runtime::stream_signal_free(done_signal_);
    if (!s.ok()) status = s;
    done_signal_ = ::ps::runtime::StreamSignal{};
    s = ::ps::runtime::stream_signal_free(result_signal_);
    if (!s.ok()) status = s;
    result_signal_ = ::ps::runtime::StreamSignal{};

    if (queue_active_) {
        const Status down = probe_code_.shutdown();
        if (!down.ok()) status = down;
        const Status q = queue_.shutdown();
        if (!q.ok()) status = q;
        queue_active_ = false;
    }
    partition_active_ = false;
    plan_node_count_ = 0u;
    plan_variant_count_ = 0u;
    return status;
}

void McuDecodeRuntime::move_from(McuDecodeRuntime&& other) noexcept {
    partition_ = std::move(other.partition_);
    queue_ = std::move(other.queue_);
    fsm_ = std::move(other.fsm_);
    kernarg_ = std::move(other.kernarg_);
    log_region_ = std::move(other.log_region_);
    probe_code_ = std::move(other.probe_code_);
    probe_meta_ = other.probe_meta_;
    code_objects_ = std::move(other.code_objects_);
    plan_nodes_ = other.plan_nodes_;
    plan_variants_ = other.plan_variants_;
    plan_retained_ = other.plan_retained_;
    plan_output_ = other.plan_output_;
    plan_timestamps_ = other.plan_timestamps_;
    completions_ = other.completions_;
    records_ = other.records_;
    plan_node_count_ = other.plan_node_count_;
    plan_variant_count_ = other.plan_variant_count_;
    completion_count_ = other.completion_count_;
    record_count_ = other.record_count_;
    rms_ = other.rms_;
    rms_count_ = other.rms_count_;
    quant_ = other.quant_;
    quant_count_ = other.quant_count_;
    e4m3_ = other.e4m3_;
    e4m3_count_ = other.e4m3_count_;
    psq4_ = other.psq4_;
    psq4_count_ = other.psq4_count_;
    psq4_multi_ = other.psq4_multi_;
    psq4_multi_count_ = other.psq4_multi_count_;
    verify_accept_ = other.verify_accept_;
    verify_accept_count_ = other.verify_accept_count_;
    verify_accept_batch_ = other.verify_accept_batch_;
    verify_accept_batch_count_ = other.verify_accept_batch_count_;
    gdn_spec_restore_ = other.gdn_spec_restore_;
    gdn_spec_restore_count_ = other.gdn_spec_restore_count_;
    gdn_spec_restore_from_counts_ = other.gdn_spec_restore_from_counts_;
    gdn_spec_restore_from_counts_count_ =
        other.gdn_spec_restore_from_counts_count_;
    argmax_f32_ = other.argmax_f32_;
    argmax_f32_count_ = other.argmax_f32_count_;
    elementwise_ = other.elementwise_;
    elementwise_count_ = other.elementwise_count_;
    rope_ = other.rope_;
    rope_count_ = other.rope_count_;
    kv_append_ = other.kv_append_;
    kv_append_count_ = other.kv_append_count_;
    paged_ = other.paged_;
    paged_count_ = other.paged_count_;
    paged_split_ = other.paged_split_;
    paged_split_count_ = other.paged_split_count_;
    paged_reduce_ = other.paged_reduce_;
    paged_reduce_count_ = other.paged_reduce_count_;
    bf16_ = other.bf16_;
    bf16_count_ = other.bf16_count_;
    bf16_wmma_ = other.bf16_wmma_;
    bf16_wmma_count_ = other.bf16_wmma_count_;
    l2_ = other.l2_;
    l2_count_ = other.l2_count_;
    embedding_ = other.embedding_;
    embedding_count_ = other.embedding_count_;
    embedding_psq8_ = other.embedding_psq8_;
    embedding_psq8_count_ = other.embedding_psq8_count_;
    output_gather_ = other.output_gather_;
    output_gather_count_ = other.output_gather_count_;
    gdn_conv_ = other.gdn_conv_;
    gdn_conv_count_ = other.gdn_conv_count_;
    row_global_patches_ = other.row_global_patches_;
    row_global_patch_count_ = other.row_global_patch_count_;
    attention_regions_ = std::move(other.attention_regions_);
    bf16_catalogs_ = std::move(other.bf16_catalogs_);
    geometry_patches_ = std::move(other.geometry_patches_);
    invocation_row_patches_ = std::move(other.invocation_row_patches_);
    dynamic_bindings_ = other.dynamic_bindings_;
    dynamic_binding_count_ = other.dynamic_binding_count_;
    gdn_recur_ = other.gdn_recur_;
    gdn_recur_count_ = other.gdn_recur_count_;
    gdn_reset_ = other.gdn_reset_;
    gdn_reset_count_ = other.gdn_reset_count_;
    start_signal_ = other.start_signal_;
    done_signal_ = other.done_signal_;
    result_signal_ = other.result_signal_;
    device_ = other.device_;
    queue_ahead_depth_ = other.queue_ahead_depth_;
    driver_mode_ = other.driver_mode_;
    queue_active_ = other.queue_active_;
    fsm_active_ = other.fsm_active_;
    partition_active_ = other.partition_active_;

    other.code_objects_.clear();
    other.plan_nodes_ = nullptr;
    other.plan_variants_ = nullptr;
    other.plan_retained_ = nullptr;
    other.plan_output_ = nullptr;
    other.plan_timestamps_ = nullptr;
    other.completions_ = nullptr;
    other.records_ = nullptr;
    other.rms_ = nullptr;
    other.quant_ = nullptr;
    other.e4m3_ = nullptr;
    other.psq4_ = nullptr;
    other.psq4_multi_ = nullptr;
    other.verify_accept_ = nullptr;
    other.verify_accept_batch_ = nullptr;
    other.gdn_spec_restore_ = nullptr;
    other.gdn_spec_restore_from_counts_ = nullptr;
    other.argmax_f32_ = nullptr;
    other.elementwise_ = nullptr;
    other.rope_ = nullptr;
    other.kv_append_ = nullptr;
    other.paged_ = nullptr;
    other.paged_split_ = nullptr;
    other.paged_reduce_ = nullptr;
    other.bf16_ = nullptr;
    other.bf16_wmma_ = nullptr;
    other.bf16_wmma_count_ = 0u;
    other.l2_ = nullptr;
    other.embedding_ = nullptr;
    other.embedding_psq8_ = nullptr;
    other.embedding_psq8_count_ = 0u;
    other.output_gather_ = nullptr;
    other.gdn_conv_ = nullptr;
    other.gdn_recur_ = nullptr;
    other.gdn_reset_ = nullptr;
    other.row_global_patches_ = nullptr;
    other.dynamic_bindings_ = nullptr;
    other.dynamic_binding_count_ = 0u;
    other.rms_count_ = 0u;
    other.quant_count_ = 0u;
    other.e4m3_count_ = 0u;
    other.psq4_count_ = 0u;
    other.psq4_multi_count_ = 0u;
    other.verify_accept_count_ = 0u;
    other.verify_accept_batch_count_ = 0u;
    other.gdn_spec_restore_count_ = 0u;
    other.gdn_spec_restore_from_counts_count_ = 0u;
    other.argmax_f32_count_ = 0u;
    other.elementwise_count_ = 0u;
    other.rope_count_ = 0u;
    other.kv_append_count_ = 0u;
    other.paged_count_ = 0u;
    other.paged_split_count_ = 0u;
    other.paged_reduce_count_ = 0u;
    other.bf16_count_ = 0u;
    other.l2_count_ = 0u;
    other.gdn_conv_count_ = 0u;
    other.gdn_recur_count_ = 0u;
    other.gdn_reset_count_ = 0u;
    other.row_global_patch_count_ = 0u;
    other.start_signal_ = ::ps::runtime::StreamSignal{};
    other.done_signal_ = ::ps::runtime::StreamSignal{};
    other.result_signal_ = ::ps::runtime::StreamSignal{};
    other.plan_node_count_ = 0u;
    other.plan_variant_count_ = 0u;
    other.completion_count_ = 0u;
    other.record_count_ = 0u;
    other.queue_active_ = false;
    other.fsm_active_ = false;
    other.partition_active_ = false;
}

McuDecodeRuntime::McuDecodeRuntime(McuDecodeRuntime&& other) noexcept {
    move_from(std::move(other));
}

McuDecodeRuntime& McuDecodeRuntime::operator=(
    McuDecodeRuntime&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        move_from(std::move(other));
    }
    return *this;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
