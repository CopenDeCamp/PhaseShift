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

__device__ __forceinline__ unsigned char* mcu_kernarg_slot(
    const GpuMcuFsmState* state,
    uint32_t slot) {
    return reinterpret_cast<unsigned char*>(state->kernarg_base) +
           static_cast<uint64_t>(slot) * state->kernarg_slot_stride;
}

__device__ __forceinline__ const McuKernargSourceDesc* mcu_kernarg_source(
    const GpuMcuFsmState* state,
    uint32_t node_index) {
    if (state->kernarg_sources == nullptr) return nullptr;
    if (node_index >= state->kernarg_source_count) return nullptr;
    return &state->kernarg_sources[node_index];
}

__device__ __forceinline__ void mcu_copy_explicit_args(void* dst_ptr,
                                                       const void* src_ptr,
                                                       uint32_t bytes) {
    auto* dst = static_cast<unsigned char*>(dst_ptr);
    const auto* src = static_cast<const unsigned char*>(src_ptr);
    uint32_t i = 0u;
    while (i + 16u <= bytes) {
        __builtin_memcpy(dst + i, src + i, 16);
        i += 16u;
    }
    while (i + 4u <= bytes) {
        __builtin_memcpy(dst + i, src + i, 4);
        i += 4u;
    }
    while (i < bytes) {
        dst[i] = src[i];
        ++i;
    }
}

__device__ __forceinline__ bool mcu_kernarg_source_supported(
    const GpuMcuFsmState* state,
    uint32_t node_index) {
    const McuKernargSourceDesc* source = mcu_kernarg_source(state, node_index);
    if (source == nullptr) return false;
    if ((source->flags & kMcuKernargSourcePrepared) != 0u) {
        return source->source != 0ull && source->explicit_args_bytes != 0u;
    }
    if ((source->flags & kMcuKernargSourceSupervisorProbe) != 0u) {
        return source->source == 0ull;
    }
    return false;
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

__device__ __forceinline__ std::size_t mcu_kernarg_explicit_bytes(
    const GpuMcuFsmState* state,
    uint32_t node_index) {
    const McuKernargSourceDesc* source = mcu_kernarg_source(state, node_index);
    return source != nullptr ? source->explicit_args_bytes : 0u;
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

__device__ __forceinline__ bool mcu_write_prepared_kernarg(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    const McuKernargSourceDesc* source,
    uint32_t slot) {
    if (source == nullptr) return false;
    if ((source->flags & kMcuKernargSourcePrepared) == 0u) return false;
    if (source->source == 0ull) return false;
    const std::size_t explicit_bytes = source->explicit_args_bytes;
    if (explicit_bytes == 0u) return false;
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    if (explicit_bytes > variant.kernarg_size) return false;
    auto* args = mcu_kernarg_slot(state, slot);
    mcu_copy_explicit_args(args, reinterpret_cast<const void*>(source->source),
                           static_cast<uint32_t>(explicit_bytes));
    GpuAqlDispatchDesc desc{};
    mcu_variant_dispatch_geometry(variant, desc);
    if (!mcu_apply_hidden_args(variant, args, explicit_bytes, desc)) {
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
    const McuKernargSourceDesc* source = mcu_kernarg_source(state, node_index);
    if (source == nullptr) return false;
    if ((source->flags & kMcuKernargSourcePrepared) != 0u) {
        return mcu_write_prepared_kernarg(state, node, source, slot);
    }
    if ((source->flags & kMcuKernargSourceSupervisorProbe) != 0u) {
        return mcu_write_probe_kernarg(state, node, node_index, slot,
                                       generation);
    }
    return false;
}

__device__ __noinline__ bool mcu_override_geometry(
    const GpuMcuFsmState* state,
    const McuPlanNode& node,
    uint32_t node_index,
    uint32_t slot,
    uint32_t workgroup_count_x,
    uint32_t workgroup_count_y,
    uint32_t workgroup_count_z) {
    const std::size_t explicit_bytes =
        mcu_kernarg_explicit_bytes(state, node_index);
    if (explicit_bytes == 0u) return false;
    const McuKernelVariantDesc& variant = state->variants[node.variant_id];
    const uint32_t count_x = workgroup_count_x != 0u
                                 ? workgroup_count_x
                                 : variant.workgroup_count_x;
    const uint32_t count_y = workgroup_count_y != 0u
                                 ? workgroup_count_y
                                 : variant.workgroup_count_y;
    const uint32_t count_z = workgroup_count_z != 0u
                                 ? workgroup_count_z
                                 : variant.workgroup_count_z;
    auto* args = mcu_kernarg_slot(state, slot);
    GpuAqlDispatchDesc desc{};
    desc.global_work_items_x = count_x * variant.workgroup_x;
    desc.global_work_items_y = count_y * variant.workgroup_y;
    desc.global_work_items_z = count_z * variant.workgroup_z;
    desc.workgroup_size_x = static_cast<uint16_t>(variant.workgroup_x);
    desc.workgroup_size_y = static_cast<uint16_t>(variant.workgroup_y);
    desc.workgroup_size_z = static_cast<uint16_t>(variant.workgroup_z);
    if (!mcu_apply_hidden_args(variant, args, explicit_bytes, desc)) {
        return false;
    }
    build_aql_launch_metadata(args, variant.kernarg_size, count_x, count_y,
                              count_z, variant.workgroup_x);
    __threadfence_system();
    return true;
}

