__device__ __forceinline__ void mcu_publish_counter(uint64_t* field, uint64_t value) {
    __scoped_atomic_store_n(field, value, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void mcu_fault(GpuMcuFsmState* state,
                                          McuFaultCode code,
                                          uint32_t pc,
                                          uint32_t variant,
                                          uint32_t generation) {
    mcu_log_fault(state, code, pc, variant, generation);
    __scoped_atomic_store_n(&state->fault_pc, pc, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(&state->fault_variant, variant, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(&state->fault_generation, generation, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(&state->fault_code, static_cast<uint32_t>(code),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(&state->supervisor,
                            static_cast<uint32_t>(McuSupervisorState::fault),
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void mcu_publish_done(GpuMcuFsmState* state,
                                                uint64_t epoch,
                                                uint32_t code) {
    if (state->external_done_signal == nullptr) return;
    if (state->external_result_code != nullptr) {
        __scoped_atomic_store_n(state->external_result_code, code,
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    __scoped_atomic_store_n(
        state->external_done_signal, static_cast<uint32_t>(epoch),
        __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t mcu_read_index(const DeviceAqlQueueView& queue) {
    return __scoped_atomic_load_n(
        const_cast<const uint64_t*>(queue.read_index), __ATOMIC_ACQUIRE,
        __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ bool mcu_make_queue_space(
    const DeviceAqlQueueView& queue,
    uint64_t write_cursor,
    uint32_t limit,
    const volatile uint32_t* stop_requested) {
    if (queue.read_index == nullptr || queue.size == 0u) return false;
    uint64_t effective = limit == 0u
                             ? static_cast<uint64_t>(queue.size > 1u ? queue.size - 1u
                                                                     : 1u)
                             : static_cast<uint64_t>(limit);
    if (effective == 0u) effective = 1u;
    if (effective > queue.size) effective = queue.size;
    bool waited = false;
    while ((write_cursor - mcu_read_index(queue)) >= effective) {
        waited = true;
        __builtin_amdgcn_s_sleep(8);
        if (stop_requested != nullptr &&
            __scoped_atomic_load_n(const_cast<uint32_t*>(stop_requested),
                                   __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) != 0u) {
            break;
        }
    }
    return waited;
}

__device__ __forceinline__ bool mcu_reserve_kernarg_slot(
    const GpuMcuFsmState* state,
    uint32_t region_seq,
    uint32_t* slot) {
    if (region_seq >= state->kernarg_slot_count) return false;
    *slot = region_seq;
    return true;
}

__device__ __forceinline__ bool mcu_kernarg_recipe_supported(
    const GpuMcuFsmState*,
    const McuPlanNode& node) {
    return node.kernarg_recipe > kMcuKernargRecipeNone &&
           node.kernarg_recipe < kMcuKernargRecipeCount;
}

__device__ __forceinline__ unsigned char* mcu_kernarg_slot(
    const GpuMcuFsmState* state,
    uint32_t slot) {
    return reinterpret_cast<unsigned char*>(state->kernarg_base) +
           static_cast<uint64_t>(slot) * state->kernarg_slot_stride;
}

__device__ __forceinline__ bool mcu_apply_hidden_args(
    const McuKernelVariantDesc& variant,
    void* kernarg_slot,
    std::size_t explicit_args_bytes,
    const GpuAqlDispatchDesc& desc) {
    const auto policy =
        static_cast<AqlHiddenArgsPolicy>(variant.hidden_args_policy);
    if (policy == AqlHiddenArgsPolicy::None) return true;
    if (build_aql_hidden_args(kernarg_slot, explicit_args_bytes,
                              variant.kernarg_size, policy, desc)) {
        return true;
    }
    return policy != AqlHiddenArgsPolicy::Required;
}

constexpr std::size_t mcu_recipe_explicit_args_bytes(uint16_t recipe) {
    switch (recipe) {
        case kMcuKernargRecipeProbe:
            return sizeof(GpuMcuFsmWorkerArgs);
        case kMcuKernargRecipeRmsNormBf16PfOnePlus:
            return sizeof(McuRmsNormInvocation);
        case kMcuKernargRecipeActivationQuantizeA8:
            return sizeof(McuActivationQuantizeInvocation);
        case kMcuKernargRecipeActivationQuantizeE4m3K5120:
            return sizeof(McuActivationQuantizeE4m3Invocation);
        case kMcuKernargRecipePsq4Decode1Bf16U16:
        case kMcuKernargRecipePsq4Decode1Bf16U8:
        case kMcuKernargRecipePsq8Decode1Bf16U8:
            return sizeof(McuPsq4Decode1Invocation);
        case kMcuKernargRecipePsq4MultiRowBf16:
            return sizeof(McuPsq4MultiRowInvocation);
        case kMcuKernargRecipeVerifyAcceptPrefix:
            return sizeof(McuVerifyAcceptPrefixInvocation);
        case kMcuKernargRecipeVerifyAcceptBatch:
            return sizeof(McuVerifyAcceptBatchInvocation);
        case kMcuKernargRecipeGdnSpecRestoreFromCounts:
            return sizeof(McuGdnSpecRestoreFromCountsInvocation);
        case kMcuKernargRecipeGdnSpecRestore:
            return sizeof(McuGdnSpecRestoreInvocation);
        case kMcuKernargRecipeArgmaxF32:
            return sizeof(McuArgmaxF32Invocation);
        case kMcuKernargRecipeElementwise:
            return sizeof(McuElementwiseInvocation);
        case kMcuKernargRecipeRope:
            return sizeof(McuRopeInvocation);
        case kMcuKernargRecipeKvAppend:
            return sizeof(McuKvAppendInvocation);
        case kMcuKernargRecipeAttentionPaged:
            return sizeof(McuPagedAttentionInvocation);
        case kMcuKernargRecipeAttentionPagedSplit:
            return sizeof(McuPagedAttentionSplitInvocation);
        case kMcuKernargRecipeAttentionPagedReduce:
            return sizeof(McuPagedAttentionReduceInvocation);
        case kMcuKernargRecipeBf16ExactRows:
            return sizeof(McuBf16ExactRowsInvocation);
        case kMcuKernargRecipeBf16Wmma:
            return sizeof(McuBf16WmmaInvocation);
        case kMcuKernargRecipeL2Normalize:
            return sizeof(McuL2NormalizeInvocation);
        case kMcuKernargRecipeEmbeddingBf16:
            return sizeof(McuEmbeddingBf16Invocation);
        case kMcuKernargRecipeEmbeddingPsq8:
            return sizeof(McuEmbeddingPsq8Invocation);
        case kMcuKernargRecipeOutputGatherBf16:
            return sizeof(McuOutputGatherBf16Invocation);
        case kMcuKernargRecipeGdnConv1d:
            return sizeof(McuGdnConv1dInvocation);
        case kMcuKernargRecipeGdnRecurrence:
            return sizeof(McuGdnRecurrenceInvocation);
        case kMcuKernargRecipeGdnReset:
            return sizeof(GpuMcuGdnResetInvocation);
        default:
            return 0u;
    }
}

__device__ __forceinline__ bool mcu_write_probe_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t node_index,
    uint32_t slot,
    uint32_t generation) {
    auto* args = reinterpret_cast<GpuMcuFsmWorkerArgs*>(mcu_kernarg_slot(state, slot));
    GpuMcuFsmWorkerArgs out{};
    out.output = state->output_base +
                 static_cast<uint64_t>(node_index) * sizeof(uint32_t);
    out.device_completion = reinterpret_cast<uint64_t>(
        state->completions + node.completion_slot);
    out.input = state->batch_input;
    if (out.input == 0ull && node.input_slot != kMcuNoInput) {
        out.input = state->output_base +
                    static_cast<uint64_t>(node.input_slot) * sizeof(uint32_t);
    }
    out.timestamps = state->timestamps_base +
                     static_cast<uint64_t>(node_index) * 2ull * sizeof(uint64_t);
    out.value = node.value;
    out.element_count = node.element_count;
    out.generation = generation;
    out.status_detail = state->plan_id;
    out.work = node.work;
    out.flags = 0;
    *args = out;
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    GpuAqlDispatchDesc hidden{};
    hidden.global_work_items_x =
        variant.workgroup_count_x * variant.workgroup_x;
    hidden.global_work_items_y =
        variant.workgroup_count_y * variant.workgroup_y;
    hidden.global_work_items_z =
        variant.workgroup_count_z * variant.workgroup_z;
    hidden.workgroup_size_x = static_cast<uint16_t>(variant.workgroup_x);
    hidden.workgroup_size_y = static_cast<uint16_t>(variant.workgroup_y);
    hidden.workgroup_size_z = static_cast<uint16_t>(variant.workgroup_z);
    if (!mcu_apply_hidden_args(variant, args, sizeof(GpuMcuFsmWorkerArgs),
                               hidden)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ void mcu_variant_dispatch_geometry(
    const McuKernelVariantDesc& variant,
    GpuAqlDispatchDesc& desc) {
    desc.global_work_items_x = variant.workgroup_count_x * variant.workgroup_x;
    desc.global_work_items_y = variant.workgroup_count_y * variant.workgroup_y;
    desc.global_work_items_z = variant.workgroup_count_z * variant.workgroup_z;
    desc.workgroup_size_x = static_cast<uint16_t>(variant.workgroup_x);
    desc.workgroup_size_y = static_cast<uint16_t>(variant.workgroup_y);
    desc.workgroup_size_z = static_cast<uint16_t>(variant.workgroup_z);
}

__device__ __forceinline__ bool mcu_write_rmsnorm_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->rmsnorm_invocations == nullptr ||
        node.invocation_index >= state->rmsnorm_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuRmsNormInvocation*>(args) =
        state->rmsnorm_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuRmsNormInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_quantize_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->quantize_invocations == nullptr ||
        node.invocation_index >= state->quantize_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuActivationQuantizeInvocation*>(args) =
        state->quantize_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuActivationQuantizeInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_psq4_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->psq4_invocations == nullptr ||
        node.invocation_index >= state->psq4_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuPsq4Decode1Invocation*>(args) =
        state->psq4_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuPsq4Decode1Invocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_psq4_multi_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->psq4_multi_invocations == nullptr ||
        node.invocation_index >= state->psq4_multi_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuPsq4MultiRowInvocation*>(args) =
        state->psq4_multi_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuPsq4MultiRowInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_verify_accept_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->verify_accept_invocations == nullptr ||
        node.invocation_index >= state->verify_accept_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuVerifyAcceptPrefixInvocation*>(args) =
        state->verify_accept_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuVerifyAcceptPrefixInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_gdn_spec_restore_from_counts_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->gdn_spec_restore_from_counts_invocations == nullptr ||
        node.invocation_index >=
            state->gdn_spec_restore_from_counts_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuGdnSpecRestoreFromCountsInvocation*>(args) =
        state->gdn_spec_restore_from_counts_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuGdnSpecRestoreFromCountsInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_gdn_spec_restore_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->gdn_spec_restore_invocations == nullptr ||
        node.invocation_index >= state->gdn_spec_restore_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuGdnSpecRestoreInvocation*>(args) =
        state->gdn_spec_restore_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuGdnSpecRestoreInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_argmax_f32_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->argmax_f32_invocations == nullptr ||
        node.invocation_index >= state->argmax_f32_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuArgmaxF32Invocation*>(args) =
        state->argmax_f32_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuArgmaxF32Invocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_e4m3_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->e4m3_invocations == nullptr ||
        node.invocation_index >= state->e4m3_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuActivationQuantizeE4m3Invocation*>(args) =
        state->e4m3_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuActivationQuantizeE4m3Invocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_elementwise_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->elementwise_invocations == nullptr ||
        node.invocation_index >= state->elementwise_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuElementwiseInvocation*>(args) =
        state->elementwise_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuElementwiseInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_attention_paged_split_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->attention_paged_split_invocations == nullptr ||
        node.invocation_index >= state->attention_paged_split_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuPagedAttentionSplitInvocation*>(args) =
        state->attention_paged_split_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuPagedAttentionSplitInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_attention_paged_reduce_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->attention_paged_reduce_invocations == nullptr ||
        node.invocation_index >= state->attention_paged_reduce_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuPagedAttentionReduceInvocation*>(args) =
        state->attention_paged_reduce_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuPagedAttentionReduceInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_attention_paged_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->attention_paged_invocations == nullptr ||
        node.invocation_index >= state->attention_paged_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuPagedAttentionInvocation*>(args) =
        state->attention_paged_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuPagedAttentionInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_kv_append_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->kv_append_invocations == nullptr ||
        node.invocation_index >= state->kv_append_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuKvAppendInvocation*>(args) =
        state->kv_append_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuKvAppendInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_rope_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->rope_invocations == nullptr ||
        node.invocation_index >= state->rope_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuRopeInvocation*>(args) =
        state->rope_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuRopeInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_bf16_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->bf16_invocations == nullptr ||
        node.invocation_index >= state->bf16_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuBf16ExactRowsInvocation*>(args) =
        state->bf16_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuBf16ExactRowsInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_bf16_wmma_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->bf16_wmma_invocations == nullptr ||
        node.invocation_index >= state->bf16_wmma_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuBf16WmmaInvocation*>(args) =
        state->bf16_wmma_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuBf16WmmaInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_l2_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->l2_invocations == nullptr ||
        node.invocation_index >= state->l2_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuL2NormalizeInvocation*>(args) =
        state->l2_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuL2NormalizeInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_embedding_bf16_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->embedding_invocations == nullptr ||
        node.invocation_index >= state->embedding_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    McuEmbeddingBf16Invocation inv =
        state->embedding_invocations[node.invocation_index];
    if (state->batch_input != 0ull) {
        inv.token_ids = state->batch_input;
    }
    *reinterpret_cast<McuEmbeddingBf16Invocation*>(args) = inv;
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuEmbeddingBf16Invocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_embedding_psq8_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->embedding_psq8_invocations == nullptr ||
        node.invocation_index >= state->embedding_psq8_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    McuEmbeddingPsq8Invocation inv =
        state->embedding_psq8_invocations[node.invocation_index];
    if (state->batch_input != 0ull) {
        inv.token_ids = state->batch_input;
    }
    *reinterpret_cast<McuEmbeddingPsq8Invocation*>(args) = inv;
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuEmbeddingPsq8Invocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_output_gather_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->output_gather_invocations == nullptr ||
        node.invocation_index >= state->output_gather_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuOutputGatherBf16Invocation*>(args) =
        state->output_gather_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuOutputGatherBf16Invocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_verify_accept_batch_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->verify_accept_batch_invocations == nullptr ||
        node.invocation_index >= state->verify_accept_batch_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuVerifyAcceptBatchInvocation*>(args) =
        state->verify_accept_batch_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuVerifyAcceptBatchInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_gdn_conv1d_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->gdn_conv1d_invocations == nullptr ||
        node.invocation_index >= state->gdn_conv1d_invocation_count) {
        return false;
    }    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuGdnConv1dInvocation*>(args) =
        state->gdn_conv1d_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, sizeof(McuGdnConv1dInvocation),
                               desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_gdn_recurrence_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->gdn_recurrence_invocations == nullptr ||
        node.invocation_index >= state->gdn_recurrence_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<McuGdnRecurrenceInvocation*>(args) =
        state->gdn_recurrence_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(McuGdnRecurrenceInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool mcu_write_gdn_reset_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot) {
    if (state->gdn_reset_invocations == nullptr ||
        node.invocation_index >= state->gdn_reset_invocation_count) {
        return false;
    }
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    *reinterpret_cast<GpuMcuGdnResetInvocation*>(args) =
        state->gdn_reset_invocations[node.invocation_index];
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args,
                               sizeof(GpuMcuGdnResetInvocation), desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size,
                              variant.workgroup_count_x,
                              variant.workgroup_count_y,
                              variant.workgroup_count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

__device__ __noinline__ bool mcu_build_kernarg(const GpuMcuFsmState* state,
                                                  const McuPlanNode& node,
                                                  uint32_t node_index,
                                                  uint32_t slot,
                                                  uint32_t generation) {
    switch (node.kernarg_recipe) {
        case kMcuKernargRecipeProbe:
            return mcu_write_probe_kernarg(state, node, node_index, slot,
                                           generation);
        case kMcuKernargRecipeRmsNormBf16PfOnePlus:
            return mcu_write_rmsnorm_kernarg(state, node, slot);
        case kMcuKernargRecipeActivationQuantizeA8:
            return mcu_write_quantize_kernarg(state, node, slot);
        case kMcuKernargRecipeActivationQuantizeE4m3K5120:
            return mcu_write_e4m3_kernarg(state, node, slot);
        case kMcuKernargRecipePsq4Decode1Bf16U16:
        case kMcuKernargRecipePsq4Decode1Bf16U8:
        case kMcuKernargRecipePsq8Decode1Bf16U8:
            return mcu_write_psq4_kernarg(state, node, slot);
        case kMcuKernargRecipePsq4MultiRowBf16:
            return mcu_write_psq4_multi_kernarg(state, node, slot);
        case kMcuKernargRecipeVerifyAcceptPrefix:
            return mcu_write_verify_accept_kernarg(state, node, slot);
        case kMcuKernargRecipeGdnSpecRestore:
            return mcu_write_gdn_spec_restore_kernarg(state, node, slot);
        case kMcuKernargRecipeArgmaxF32:
            return mcu_write_argmax_f32_kernarg(state, node, slot);
        case kMcuKernargRecipeElementwise:
            return mcu_write_elementwise_kernarg(state, node, slot);
        case kMcuKernargRecipeRope:
            return mcu_write_rope_kernarg(state, node, slot);
        case kMcuKernargRecipeKvAppend:
            return mcu_write_kv_append_kernarg(state, node, slot);
        case kMcuKernargRecipeAttentionPaged:
            return mcu_write_attention_paged_kernarg(state, node, slot);
        case kMcuKernargRecipeAttentionPagedSplit:
            return mcu_write_attention_paged_split_kernarg(state, node, slot);
        case kMcuKernargRecipeAttentionPagedReduce:
            return mcu_write_attention_paged_reduce_kernarg(state, node, slot);
        case kMcuKernargRecipeBf16ExactRows:
            return mcu_write_bf16_kernarg(state, node, slot);
        case kMcuKernargRecipeBf16Wmma:
            return mcu_write_bf16_wmma_kernarg(state, node, slot);
        case kMcuKernargRecipeL2Normalize:
            return mcu_write_l2_kernarg(state, node, slot);
        case kMcuKernargRecipeEmbeddingBf16:
            return mcu_write_embedding_bf16_kernarg(state, node, slot);
        case kMcuKernargRecipeEmbeddingPsq8:
            return mcu_write_embedding_psq8_kernarg(state, node, slot);
        case kMcuKernargRecipeOutputGatherBf16:
            return mcu_write_output_gather_kernarg(state, node, slot);
        case kMcuKernargRecipeVerifyAcceptBatch:
            return mcu_write_verify_accept_batch_kernarg(state, node, slot);
        case kMcuKernargRecipeGdnSpecRestoreFromCounts:
            return mcu_write_gdn_spec_restore_from_counts_kernarg(state, node,
                                                                  slot);
        case kMcuKernargRecipeGdnConv1d:
            return mcu_write_gdn_conv1d_kernarg(state, node, slot);
        case kMcuKernargRecipeGdnRecurrence:
            return mcu_write_gdn_recurrence_kernarg(state, node, slot);
        case kMcuKernargRecipeGdnReset:
            return mcu_write_gdn_reset_kernarg(state, node, slot);
        default:
            return false;
    }
}

__device__ __noinline__ bool mcu_override_geometry(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t slot,
    uint32_t workgroup_count_x,
    uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
    const std::size_t explicit_bytes =
        mcu_recipe_explicit_args_bytes(node.kernarg_recipe);
    if (explicit_bytes == 0u) return false;
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    auto* args = mcu_kernarg_slot(state, slot);
    GpuAqlDispatchDesc desc{};
    desc.global_work_items_x = workgroup_count_x * variant.workgroup_x;
    desc.global_work_items_y = workgroup_count_y * variant.workgroup_y;
    desc.global_work_items_z = workgroup_count_z * variant.workgroup_z;
    desc.workgroup_size_x = static_cast<uint16_t>(variant.workgroup_x);
    desc.workgroup_size_y = static_cast<uint16_t>(variant.workgroup_y);
    desc.workgroup_size_z = static_cast<uint16_t>(variant.workgroup_z);
    if (!mcu_apply_hidden_args(variant, args, explicit_bytes, desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size, workgroup_count_x,
                              workgroup_count_y, workgroup_count_z,
                              variant.workgroup_x);
    __threadfence_system();
    return true;
}

