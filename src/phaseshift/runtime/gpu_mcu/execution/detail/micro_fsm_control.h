__device__ __forceinline__ void mcu_control_loop(GpuMcuFsmState* state,
                                                 GpuMcuRetainedPacket* lds) {
    state->run_ctx = GpuMcuFsmRunContext{};
    __scoped_atomic_store_n(&state->started, 1u, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    for (;;) {
        if (__scoped_atomic_load_n(&state->stop_requested, __ATOMIC_ACQUIRE,
                                   __MEMORY_SCOPE_SYSTEM) != 0u) {
            break;
        }
        ++state->run_ctx.iterations;
        if ((state->run_ctx.iterations & 0x3ffu) == 0u) {
            ++state->run_ctx.heartbeat;
            mcu_publish_counter(&state->heartbeat, state->run_ctx.heartbeat);
            mcu_publish_counter(&state->iterations, state->run_ctx.iterations);
        }
        if (mcu_run_once(state, lds)) continue;
        const uint32_t idle_sleep = state->idle_sleep;
        if (idle_sleep >= 64u) {
            __builtin_amdgcn_s_sleep(64);
        } else if (idle_sleep >= 32u) {
            __builtin_amdgcn_s_sleep(32);
        } else {
            __builtin_amdgcn_s_sleep(8);
        }
    }
    mcu_publish_counter(&state->dispatches_committed,
                        state->run_ctx.dispatches_committed);
    mcu_publish_counter(&state->completions_observed,
                        state->run_ctx.completions_observed);
    mcu_publish_counter(&state->plans_started, state->run_ctx.plans_started);
    __scoped_atomic_store_n(&state->supervisor, state->run_ctx.supervisor,
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(&state->started, 0u, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ bool gpu_mcu_execution_step(
    GpuMcuPersistentState* state) {
    state->stage_trace.stage = kMcuStageExecutionStepEnter;
    state->stage_trace.seq += 1u;
    if (state->execution_enabled == 0u || state->execution_fsm == nullptr) {
        return false;
    }
    DeviceBatchContext* context = state->batch_context;
    if (context == nullptr || state->batch_ready_epoch == nullptr) {
        return false;
    }
    if (state->execution_wrap_faulted != 0u) {
        return false;
    }
    const uint64_t ready = __scoped_atomic_load_n(
        state->batch_ready_epoch, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    if (ready == state->active_batch_ready_epoch) {
        return false;
    }
    if (context->actual_rows == 0u) {
        if (state->execution_fsm != nullptr) {
            mcu_log_event(state->execution_fsm, McuLogEvent::ExecutionSkip,
                          0u, 0u, 0u, ready,
                          static_cast<uint64_t>(
                              state->active_batch_ready_epoch),
                          /*reason actual_rows_zero*/ 1u);
        }
        state->active_batch_ready_epoch = ready;
        return false;
    }
    if (state->execution_epoch == UINT64_MAX) {
        state->execution_wrap_faulted = 1u;
        return false;
    }
    const uint64_t consumed = state->execution_fsm->run_ctx.run_consumed;
    uint64_t epoch = state->execution_epoch + 1u;
    if (epoch <= consumed) epoch = consumed + 1u;
    gpu_mcu_bind_attention_paths(context,
                                 state->execution_fsm->attention_paths,
                                 state->execution_fsm->attention_path_count,
                                 state->execution_fsm->runtime_node_bindings,
                                 state->execution_fsm
                                     ->runtime_node_binding_count);
    gpu_mcu_bind_execution_plan(context, state->execution_patches,
                                state->execution_patch_count,
                                &state->stage_trace);
    state->stage_trace.stage = kMcuStageExecutionStepEnter;
    state->stage_trace.seq += 1u;
    state->execution_fsm->batch_input =
        reinterpret_cast<uint64_t>(context->token_ids);
    if (state->execution_fsm != nullptr) {
        mcu_log_event(state->execution_fsm, McuLogEvent::ExecutionEpoch,
                      0u, 0u, 0u, epoch,
                      static_cast<uint64_t>(
                          state->execution_fsm->run_ctx.run_consumed),
                      static_cast<uint64_t>(state->execution_fsm->run_request));
    }
    __scoped_atomic_store_n(&state->execution_fsm->run_request, epoch,
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    state->stage_trace.stage = kMcuStageRunOnceEnter;
    state->stage_trace.seq += 1u;
    const bool ran = mcu_run_once(state->execution_fsm, nullptr);
    state->stage_trace.stage = kMcuStageRunOnceExit;
    state->stage_trace.seq += 1u;
    if (state->execution_fsm != nullptr) {
        mcu_log_event(state->execution_fsm, McuLogEvent::ExecutionEpoch,
                      1u, ran ? 1u : 0u, 0u, epoch,
                      static_cast<uint64_t>(
                          state->execution_fsm->run_ctx.run_consumed),
                      static_cast<uint64_t>(state->execution_epoch));
    }
    state->active_batch_ready_epoch = ready;
    if (!ran) {
        return false;
    }
    state->execution_epoch = epoch;
    state->execution_telemetry.batches_dispatched += 1u;
    if (state->execution_fsm->fault_code ==
        static_cast<uint32_t>(McuFaultCode::none)) {
        state->execution_telemetry.batches_completed += 1u;
        GpuMcuBatchCommitView commit{};
        commit.context = context;
        commit.slots = state->request_slots;
        commit.max_slots = state->request_max_slots;
        commit.bindings = state->slot_bindings;
        commit.binding_max_slots = state->binding_max_slots;
        commit.request_limits_valid =
            state->request_runtime_enabled != 0u ? 1u : 0u;
        commit.sampled_tokens = state->execution_sampled_tokens;
        commit.sampled_capacity = state->execution_sampled_capacity;
        commit.verify_committed_counts = state->execution_verify_counts;
        commit.verify_capacity = state->execution_verify_capacity;
        commit.resources = state->sequence_resources;
        commit.kv_pages = state->kv_pages;
        commit.kv_blocks = state->kv_blocks;
        commit.slot_runtime = state->slot_runtime;
        commit.resources_enabled = state->resources_enabled;
        commit.telemetry = &state->commit_telemetry;
        if (gpu_mcu_commit_active_batch(commit)) {
            state->execution_telemetry.batches_committed += 1u;
        }
        state->last_committed_execution_epoch = epoch;
        state->execution_telemetry.autonomous_loops += 1u;
        state->scheduler_dirty = 1u;
    } else {
        state->execution_telemetry.batches_failed += 1u;
        if (state->request_slots != nullptr) {
            for (uint32_t i = 0; i < context->num_requests; ++i) {
                const RequestHandle handle =
                    context->requests[i].request_handle;
                if (handle.slot >= state->request_max_slots) continue;
                (void)gpu_mcu_slot_mark_terminal(
                    state->request_slots[handle.slot], handle,
                    static_cast<uint32_t>(GpuMcuTerminalReason::error));
            }
        }
        state->last_committed_execution_epoch = epoch;
        state->scheduler_dirty = 1u;
    }
    return true;
}

__device__ void gpu_mcu_persistent_loop(GpuMcuPersistentState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    __scoped_atomic_store_n(&state->started, 1u, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    uint64_t count = 0;
    state->stage_trace.stage = kMcuStageLoopBegin;
    state->stage_trace.seq += 1u;
    while (true) {
        if (__scoped_atomic_load_n(&state->stop_requested, __ATOMIC_ACQUIRE,
                                   __MEMORY_SCOPE_SYSTEM) != 0u) {
            break;
        }
        ++count;
        state->stage_trace.stage = kMcuStageIteration;
        state->stage_trace.seq += 1u;
        __scoped_atomic_store_n(&state->heartbeat, count, __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(&state->iterations, count, __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(&state->last_loop_ts,
                                static_cast<uint64_t>(wall_clock64()),
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);

        bool progressed = false;
        if (state->request_runtime_enabled != 0u) {
            state->stage_trace.stage = kMcuStageSchedulerEnter;
            state->stage_trace.seq += 1u;
            progressed = gpu_mcu_scheduler_boundary(state);
            state->stage_trace.stage = kMcuStageSchedulerExit;
            state->stage_trace.seq += 1u;
        }
        if (gpu_mcu_execution_step(state)) {
            progressed = true;
        }
        state->stage_trace.stage = kMcuStageIteration;
        state->stage_trace.seq += 1u;

        bool emitted = false;
        if (state->emit_enabled != 0u && state->template_count != 0u) {
            state->stage_trace.stage = kMcuStageEmitEnter;
            state->stage_trace.seq += 1u;
            const uint64_t target = __scoped_atomic_load_n(
                &state->submit_request, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            const uint64_t published = state->published;
            if (published < target) {
                const uint32_t index =
                    static_cast<uint32_t>(published % state->template_count);
                if (state->publish_ts != nullptr) {
                    state->publish_ts[index] =
                        static_cast<uint64_t>(wall_clock64());
                }
                GpuMcuAqlStatus status = GpuMcuAqlStatus::INVALID_PARAMS;
                if (state->emit_streaming != 0u) {
                    if (state->write_base_valid == 0u) {
                        state->write_base = *state->queue.write_index;
                        __threadfence_system();
                        state->write_base_valid = 1u;
                    }
                    status = gpu_mcu_aql_publish_streaming(
                        state->queue, state->templates + index, 1u,
                        state->write_base + published, state->barrier != 0u);
                } else {
                    GpuMcuPublishBatchArgs args{};
                    args.queue = state->queue;
                    args.packet_templates = state->templates + index;
                    args.packet_count = 1u;
                    status = gpu_mcu_aql_publish_batch(args);
                }
                if (state->doorbell_ts != nullptr) {
                    state->doorbell_ts[index] =
                        static_cast<uint64_t>(wall_clock64());
                }
                if (status == GpuMcuAqlStatus::COMPLETE) {
                    __scoped_atomic_store_n(&state->published, published + 1u,
                                            __ATOMIC_RELEASE,
                                            __MEMORY_SCOPE_SYSTEM);
                    emitted = true;
                } else {
                    __scoped_atomic_store_n(
                        &state->publish_failures,
                        __scoped_atomic_load_n(&state->publish_failures,
                                               __ATOMIC_RELAXED,
                                               __MEMORY_SCOPE_SYSTEM) +
                            1u,
                        __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
                }
            }
        }
        if (!emitted && !progressed) {
            if (state->idle_sleep >= 64u) {
                __builtin_amdgcn_s_sleep(64);
            } else if (state->idle_sleep >= 32u) {
                __builtin_amdgcn_s_sleep(32);
            } else {
                __builtin_amdgcn_s_sleep(8);
            }
        }
    }
    state->stage_trace.stage = kMcuStageLoopEnd;
    state->stage_trace.seq += 1u;
    __scoped_atomic_store_n(&state->started, 0u, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
}

__global__ void gpu_mcu_fsm_kernel(GpuMcuFsmState* state) {
    extern __shared__ unsigned char shared_bytes[];
    auto* lds = reinterpret_cast<GpuMcuRetainedPacket*>(shared_bytes);

    if (state->template_source != 0u && state->retained_count != 0u) {
        const uint32_t lanes = blockDim.x;
        if (state->copy_lanes <= 1u) {
            if (threadIdx.x == 0u) {
                for (uint32_t i = 0; i < state->retained_count; ++i) {
                    lds[i] = state->retained_vram[i];
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < state->retained_count; i += lanes) {
                lds[i] = state->retained_vram[i];
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        mcu_control_loop(state, lds);
    }
}

uint32_t load_u32(const uint32_t* ptr) {
    return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(ptr))
        .load(std::memory_order_acquire);
}

uint64_t load_u64(const uint64_t* ptr) {
    return std::atomic_ref<uint64_t>(*const_cast<uint64_t*>(ptr))
        .load(std::memory_order_acquire);
}

void store_u64(uint64_t* ptr, uint64_t value) {
    std::atomic_ref<uint64_t>(*ptr).store(value, std::memory_order_release);
}

