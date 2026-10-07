__device__ __forceinline__ bool mcu_run_once(GpuMcuFsmState* state,
                                             GpuMcuRetainedPacket* lds) {
    GpuMcuFsmRunContext& ctx = state->run_ctx;
    const DeviceAqlQueueView queue = state->queue;
    const McuPlanNode* plan = state->plan;
    const uint32_t node_count = state->node_count;
    const uint32_t variant_count = state->variant_count;
    const uint64_t kernarg_base = state->kernarg_base;
    const uint64_t kernarg_stride = state->kernarg_slot_stride;
    const uint32_t kernarg_slot_count = state->kernarg_slot_count;
    GpuMcuDeviceCompletion* completions = state->completions;
    const uint32_t completion_count = state->completion_count;
    GpuMcuRetainedPacket* vram = state->retained_vram;
    const uint32_t retained_count = state->retained_count;
    const McuDynamicNodeBinding* dynamic_bindings =
        state->dynamic_node_bindings;
    const uint32_t dynamic_binding_count = state->dynamic_node_binding_count;
    const uint32_t template_source = state->template_source;
    const uint32_t barrier = state->barrier;
    const uint32_t prepared_enabled = state->prepared_enabled;
    McuDispatchTiming* timing = state->timing;
    const uint32_t timing_count = state->timing_count;
    const uint32_t queue_ahead_depth = state->queue_ahead_depth;
    const uint32_t doorbell_batch =
        state->doorbell_batch != 0u ? state->doorbell_batch : 1u;
    const McuDoorbellMode doorbell_mode =
        static_cast<McuDoorbellMode>(state->doorbell_mode);
    const uint32_t issue_wait_running = state->issue_wait_running;
    McuDispatchRecord* debug_records = state->debug_records;
    const uint32_t debug_record_count = state->debug_record_count;
    uint32_t* const start_signal = state->external_start_signal;
    uint32_t* const done_signal = state->external_done_signal;
    uint32_t* const result_code = state->external_result_code;

    McuSupervisorState supervisor =
        static_cast<McuSupervisorState>(ctx.supervisor);
    if (supervisor == McuSupervisorState::boot) {
        supervisor = McuSupervisorState::idle;
    }
    __scoped_atomic_store_n(&state->supervisor,
                            static_cast<uint32_t>(supervisor), __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    if (__scoped_atomic_load_n(&state->stop_requested, __ATOMIC_ACQUIRE,
                            __MEMORY_SCOPE_SYSTEM) != 0u) {
        ctx.supervisor = static_cast<uint32_t>(McuSupervisorState::stopping);
        return true;
    }

    if (supervisor == McuSupervisorState::idle) {
        uint64_t request = 0;
        if (start_signal != nullptr) {
            request = static_cast<uint64_t>(__scoped_atomic_load_n(
                start_signal, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM));
            __scoped_atomic_store_n(
                &state->external_epoch_seen,
                static_cast<uint32_t>(request), __ATOMIC_RELEASE,
                __MEMORY_SCOPE_SYSTEM);
        } else {
            request = __scoped_atomic_load_n(&state->run_request,
                                             __ATOMIC_ACQUIRE,
                                             __MEMORY_SCOPE_SYSTEM);
        }
        if (request <= ctx.run_consumed) {
            return false;
        }
        if (plan == nullptr || node_count == 0u || node_count > kMcuMaxNodes) {
            mcu_fault(state, McuFaultCode::invalid_plan, 0u, 0u, 0u);
            mcu_publish_done(state, request,
                             static_cast<uint32_t>(McuFaultCode::invalid_plan));
            supervisor = McuSupervisorState::fault;
            ctx.supervisor = static_cast<uint32_t>(supervisor);
            return false;
        }
        if (start_signal != nullptr) {
            ctx.run_consumed = request;
        } else {
            ctx.run_consumed += 1u;
        }
        ctx.region_seq = 0u;
        supervisor = McuSupervisorState::running;
        __scoped_atomic_store_n(&state->supervisor,
                                static_cast<uint32_t>(supervisor),
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        mcu_log_event(state, McuLogEvent::PlanBegin, 0u, 0u, 0u, state->plan_id,
                      state->node_count, 0u);
    }

    if (supervisor == McuSupervisorState::fault) {
        return false;
    }

    bool faulted = false;
    uint32_t pc = 0;
    uint32_t active_completion = 0xffffffffu;
    uint32_t active_generation = 0u;
    uint32_t active_timing = 0xffffffffu;
    bool have_active = false;
    bool have_prepared = false;
    uint32_t prepared_pc = 0;
    uint64_t prepared_queue_slot = 0;
    uint32_t prepared_generation = 0;
    uint32_t prepared_timing = 0;
    GpuMcuStagedPacket prepared_staged{};
    uint32_t pending_doorbell = 0;
    uint64_t last_published_slot = 0;
    bool issued_any = false;
    uint32_t run_issues = 0;
    uint32_t prev_completion = 0xffffffffu;
    uint32_t prev_generation = 0u;
    uint64_t run_append_ahead = 0;
    uint64_t run_queue_empty = 0;
    uint64_t run_queue_full = 0;
    uint64_t run_refill = 0;
    uint64_t run_doorbell = 0;
    uint64_t run_max_ahead = 0;
    uint64_t run_wrap = 0;

    while (supervisor == McuSupervisorState::running) {
        if (pc >= node_count) {
            mcu_fault(state, McuFaultCode::invalid_node, pc, 0u, 0u);
            faulted = true;
            break;
        }
        McuPlanNode node = plan[pc];
        if ((node.flags & kMcuNodeEnd) != 0u) {
            break;
        }
        if (dynamic_bindings != nullptr && pc < dynamic_binding_count) {
            const McuDynamicNodeBinding binding = dynamic_bindings[pc];
            if (binding.enabled == 0u) {
                uint32_t skip = node.next == kMcuNoNext ? pc + 1u : node.next;
                if (skip < node_count &&
                    (plan[skip].flags & kMcuNodeWait) != 0u &&
                    plan[skip].completion_slot == node.completion_slot) {
                    skip = plan[skip].next == kMcuNoNext ? skip + 1u
                                                         : plan[skip].next;
                }
                pc = skip;
                continue;
            }
            node.variant_id = binding.variant_id;
            node.invocation_index = binding.invocation_index;
            if (binding.variant_override != 0xFFFFFFFFu) {
                node.variant_id = static_cast<uint16_t>(binding.variant_override);
            }
        }

        if ((node.flags & kMcuNodeDispatch) != 0u) {
            if (node.variant_id >= variant_count) {
                mcu_fault(state, McuFaultCode::invalid_variant, pc,
                          node.variant_id, 0u);
                faulted = true;
                break;
            }
            if (node.completion_slot >= completion_count) {
                mcu_fault(state, McuFaultCode::invalid_node, pc,
                          node.variant_id, 0u);
                faulted = true;
                break;
            }
            if (kernarg_slot_count == 0u) {
                mcu_fault(state, McuFaultCode::kernarg_slot_unavailable, pc,
                          node.variant_id, 0u);
                faulted = true;
                break;
            }
            if (node.variant_id >= retained_count) {
                mcu_fault(state, McuFaultCode::invalid_variant, pc,
                          node.variant_id, 0u);
                faulted = true;
                break;
            }
            if (!mcu_kernarg_source_supported(state, pc)) {
                mcu_fault(state, McuFaultCode::invalid_node, pc,
                          node.variant_id, 0u);
                faulted = true;
                break;
            }

            if (issue_wait_running != 0u && run_issues == 1u &&
                prev_completion != 0xffffffffu && completions != nullptr) {
                uint64_t start_spins = 0;
                for (;;) {
                    const uint32_t observed =
                        gpu_mcu_device_completion_observe(
                            completions + prev_completion);
                    const uint32_t observed_generation =
                        __scoped_atomic_load_n(
                            &completions[prev_completion].generation,
                            __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
                    const bool started =
                        observed_generation == prev_generation &&
                        (observed == static_cast<uint32_t>(
                                         GpuMcuDeviceCompletionState::running) ||
                         observed == static_cast<uint32_t>(
                                         GpuMcuDeviceCompletionState::complete));
                    if (started) break;
                    if (__scoped_atomic_load_n(&state->stop_requested,
                                               __ATOMIC_ACQUIRE,
                                               __MEMORY_SCOPE_SYSTEM) != 0u) {
                        break;
                    }
                    if (++start_spins >= kMcuStartSpinLimit) break;
                    __builtin_amdgcn_s_sleep(4);
                }
            }

            const uint64_t stage_start = wall_clock64();
            const uint32_t timing_index =
                timing != nullptr && timing_count != 0u
                    ? static_cast<uint32_t>(ctx.dispatch_seq % timing_count)
                    : 0xffffffffu;
            const bool debug_on =
                debug_records != nullptr && debug_record_count != 0u;

            uint64_t queue_slot = 0;
            uint32_t generation = 0;
            GpuMcuStagedPacket staged{};
            uint32_t record_flags = 0u;
            bool prepared = have_prepared && prepared_pc == pc;
            if (prepared) {
                queue_slot = prepared_queue_slot;
                generation = prepared_generation;
                staged = prepared_staged;
                have_prepared = false;
            } else {
                if (run_issues == 0u) {
                    mcu_publish_counter(&state->region_start_ts,
                                        wall_clock64());
                }
                generation = static_cast<uint32_t>(ctx.dispatch_seq + 1u);
                uint32_t grid_override_x = 0u;
                uint32_t grid_override_y = 0u;
                uint32_t grid_override_z = 0u;
                if (dynamic_bindings != nullptr &&
                    pc < dynamic_binding_count) {
                    const McuDynamicNodeBinding binding = dynamic_bindings[pc];
                    grid_override_x = binding.workgroup_count_x;
                    grid_override_y = binding.workgroup_count_y;
                    grid_override_z = binding.workgroup_count_z;
                }
                uint32_t kernarg_slot = 0u;
                if (!mcu_reserve_kernarg_slot(state, ctx.region_seq,
                                              &kernarg_slot)) {
                    mcu_fault(state, McuFaultCode::kernarg_region_exhausted,
                              pc, node.variant_id, active_generation);
                    faulted = true;
                    break;
                }
                ++ctx.region_seq;
                if (!mcu_build_kernarg(state, node, pc, kernarg_slot,
                                        generation)) {
                    mcu_fault(state, McuFaultCode::kernarg_build_failed, pc,
                              node.variant_id, 0u);
                    faulted = true;
                    break;
                }
                mcu_log_event(state, McuLogEvent::DispatchPrepare, pc,
                              node.variant_id, node.completion_slot, generation,
                              kernarg_slot, 0u);
                if (grid_override_x != 0u || grid_override_y != 0u ||
                    grid_override_z != 0u) {
                    if (!mcu_override_geometry(state, node, pc, kernarg_slot,
                                               grid_override_x,
                                               grid_override_y,
                                               grid_override_z)) {
                        mcu_fault(state, McuFaultCode::kernarg_build_failed, pc,
                                  node.variant_id, 0u);
                        faulted = true;
                        break;
                    }
                }
                if (timing != nullptr && timing_index != 0xffffffffu) {
                    timing[timing_index].kernarg_build_end = wall_clock64();
                }
                if (!ctx.write_base_valid) {
                    ctx.write_base = *queue.write_index;
                    __threadfence_system();
                    ctx.write_base_valid = true;
                }
                queue_slot = ctx.write_base + ctx.dispatch_seq;

                if (debug_on) {
                    const uint64_t read_before = mcu_read_index(queue);
                    if (read_before == queue_slot) {
                        record_flags |= kMcuRecordQueueEmpty;
                        ++run_queue_empty;
                        if (run_issues != 0u) {
                            __scoped_atomic_store_n(
                                &state->mid_region_queue_empty_count,
                                __scoped_atomic_load_n(
                                    &state->mid_region_queue_empty_count,
                                    __ATOMIC_ACQUIRE,
                                    __MEMORY_SCOPE_SYSTEM) +
                                    1u,
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
                        }
                    }
                    if (read_before > ctx.write_base) {
                        record_flags |= kMcuRecordRefill;
                        ++run_refill;
                    }
                    if (prev_completion != 0xffffffffu &&
                        completions != nullptr &&
                        !gpu_mcu_device_completion_matches(
                            completions + prev_completion, prev_generation)) {
                        record_flags |= kMcuRecordAppendAhead;
                        ++run_append_ahead;
                    }
                }

                if (mcu_make_queue_space(queue, queue_slot, queue_ahead_depth,
                                         &state->stop_requested)) {
                    record_flags |= kMcuRecordBackpressure;
                    ++run_queue_full;
                }
                if (ctx.dispatch_seq != 0u && (queue_slot % queue.size) == 0u) {
                    record_flags |= kMcuRecordWrap;
                    ++run_wrap;
                }

                const GpuMcuRetainedPacket& templ =
                    template_source != 0u ? lds[node.variant_id]
                                          : vram[node.variant_id];
                staged = gpu_mcu_aql_stage_retained_packet(
                    queue, queue_slot, templ,
                    kernarg_base +
                        static_cast<uint64_t>(kernarg_slot) * kernarg_stride,
                    grid_override_x, grid_override_y, grid_override_z);
                mcu_log_event(
                    state, McuLogEvent::DispatchPublish, pc, node.variant_id,
                    node.completion_slot, generation, queue_slot,
                    kernarg_base +
                        static_cast<uint64_t>(kernarg_slot) * kernarg_stride);
                if (debug_on) {
                    const uint64_t read_after = mcu_read_index(queue);
                    const uint64_t occupancy =
                        queue_slot + 1u > read_after
                            ? queue_slot + 1u - read_after
                            : 0u;
                    if (occupancy > run_max_ahead) run_max_ahead = occupancy;
                }
            }
            if (!prepared && timing != nullptr &&
                timing_index != 0xffffffffu) {
                timing[timing_index].node_fetch_start = stage_start;
                timing[timing_index].packet_stage_end = wall_clock64();
            }

            bool ring = false;
            switch (doorbell_mode) {
                case McuDoorbellMode::per_packet:
                    ring = true;
                    break;
                case McuDoorbellMode::coalesce:
                    ++pending_doorbell;
                    if (pending_doorbell >= doorbell_batch) {
                        ring = true;
                        pending_doorbell = 0u;
                    }
                    break;
                case McuDoorbellMode::first_only:
                    ring = !issued_any;
                    break;
                default:
                    ring = true;
                    break;
            }

            const uint64_t commit_start = wall_clock64();
            (void)gpu_mcu_aql_commit_packet(queue, staged, queue_slot + 1u,
                                            true, barrier != 0u, ring);
            const uint64_t publish_ts = wall_clock64();
            if (timing != nullptr && timing_index != 0xffffffffu) {
                timing[timing_index].commit_start = commit_start;
                timing[timing_index].dw0_publish = publish_ts;
                timing[timing_index].doorbell = ring ? publish_ts : 0u;
            }
            if (ring) {
                ++run_doorbell;
                mcu_log_event(state, McuLogEvent::DispatchDoorbell, pc,
                              node.variant_id, node.completion_slot, generation,
                              queue_slot, 0u);
            }
            last_published_slot = queue_slot;
            issued_any = true;
            ++run_issues;

            if (debug_on) {
                McuDispatchRecord rec{};
                rec.dispatch_seq = ctx.dispatch_seq;
                rec.publish_ts = publish_ts;
                rec.doorbell_ts = ring ? publish_ts : 0u;
                rec.generation = generation;
                rec.ring_slot =
                    static_cast<uint32_t>(queue_slot % queue.size);
                rec.flags =
                    record_flags | (ring ? kMcuRecordDoorbell : 0u);
                debug_records[ctx.dispatch_seq % debug_record_count] = rec;
            }

            prev_completion = node.completion_slot;
            prev_generation = generation;
            active_completion = node.completion_slot;
            active_generation = generation;
            active_timing = timing_index;
            have_active = true;
            ++ctx.dispatches_committed;
            ++ctx.dispatch_seq;

            const uint32_t next_pc =
                node.next == kMcuNoNext ? pc + 1u : node.next;
            uint32_t prefetch_pc = kMcuNoNext;
            if (prepared_enabled != 0u) {
                for (uint32_t cursor = next_pc, guard = 0u;
                     cursor < node_count && guard < node_count; ++guard) {
                    const McuPlanNode cand = plan[cursor];
                    if ((cand.flags & kMcuNodeEnd) != 0u) break;
                    if ((cand.flags & kMcuNodeDispatch) != 0u) {
                        if (dynamic_bindings != nullptr &&
                            cursor < dynamic_binding_count &&
                            dynamic_bindings[cursor].enabled == 0u) {
                            cursor = cand.next == kMcuNoNext ? cursor + 1u
                                                             : cand.next;
                            continue;
                        }
                        prefetch_pc = cursor;
                        break;
                    }
                    cursor = cand.next == kMcuNoNext ? cursor + 1u : cand.next;
                }
            }
            if (prefetch_pc != kMcuNoNext) {
                const uint32_t next_pc_dispatch = prefetch_pc;
                McuPlanNode next_node = plan[next_pc_dispatch];
                if (dynamic_bindings != nullptr &&
                    next_pc_dispatch < dynamic_binding_count) {
                    const McuDynamicNodeBinding binding =
                        dynamic_bindings[next_pc_dispatch];
                    next_node.variant_id = binding.variant_id;
                    next_node.invocation_index = binding.invocation_index;
                    if (binding.variant_override != 0xFFFFFFFFu) {
                        next_node.variant_id =
                            static_cast<uint16_t>(binding.variant_override);
                    }
                }
                if (next_node.variant_id < variant_count &&
                    kernarg_slot_count != 0u &&
                    next_node.completion_slot < completion_count &&
                    next_node.variant_id < retained_count &&
                    mcu_kernarg_source_supported(state, next_pc_dispatch)) {
                    const uint32_t next_generation =
                        static_cast<uint32_t>(ctx.dispatch_seq + 1u);
                    uint32_t next_grid_x = 0u;
                    uint32_t next_grid_y = 0u;
                    uint32_t next_grid_z = 0u;
                    if (dynamic_bindings != nullptr &&
                        next_pc_dispatch < dynamic_binding_count) {
                        const McuDynamicNodeBinding binding =
                            dynamic_bindings[next_pc_dispatch];
                        next_grid_x = binding.workgroup_count_x;
                        next_grid_y = binding.workgroup_count_y;
                        next_grid_z = binding.workgroup_count_z;
                    }
                    uint32_t next_kernarg_slot = 0u;
                    if (!mcu_reserve_kernarg_slot(state, ctx.region_seq,
                                                  &next_kernarg_slot)) {
                        mcu_fault(state,
                                  McuFaultCode::kernarg_region_exhausted,
                                  next_pc_dispatch,
                                  next_node.variant_id, active_generation);
                        faulted = true;
                        break;
                    }
                    ++ctx.region_seq;
                    if (!mcu_build_kernarg(state, next_node,
                                            next_pc_dispatch,
                                            next_kernarg_slot,
                                            next_generation)) {
                        mcu_fault(state, McuFaultCode::kernarg_build_failed,
                                  next_pc_dispatch, next_node.variant_id,
                                  active_generation);
                        faulted = true;
                        break;
                    }
                    mcu_log_event(state, McuLogEvent::DispatchPrepare,
                                  next_pc_dispatch, next_node.variant_id,
                                  next_node.completion_slot, next_generation,
                                  next_kernarg_slot, 0u);
                    if (next_grid_x != 0u || next_grid_y != 0u ||
                        next_grid_z != 0u) {
                        if (!mcu_override_geometry(state, next_node,
                                                   next_pc_dispatch,
                                                   next_kernarg_slot,
                                                   next_grid_x, next_grid_y,
                                                   next_grid_z)) {
                            mcu_fault(state,
                                      McuFaultCode::kernarg_build_failed,
                                      next_pc_dispatch, next_node.variant_id,
                                      active_generation);
                            faulted = true;
                            break;
                        }
                    }
                    const uint64_t prefetch_kernarg_end = wall_clock64();
                    const uint64_t next_slot = ctx.write_base + ctx.dispatch_seq;
                    (void)mcu_make_queue_space(queue, next_slot,
                                               queue_ahead_depth,
                                               &state->stop_requested);
                    const GpuMcuRetainedPacket& next_templ =
                        template_source != 0u ? lds[next_node.variant_id]
                                              : vram[next_node.variant_id];
                    const uint64_t prep_start = wall_clock64();
                    prepared_staged = gpu_mcu_aql_stage_retained_packet(
                        queue, next_slot, next_templ,
                        kernarg_base +
                            static_cast<uint64_t>(next_kernarg_slot) *
                                kernarg_stride,
                        next_grid_x, next_grid_y, next_grid_z);
                    prepared_queue_slot = next_slot;
                    prepared_generation = next_generation;
                    prepared_pc = next_pc_dispatch;
                    prepared_timing =
                        timing != nullptr && timing_count != 0u
                            ? static_cast<uint32_t>(ctx.dispatch_seq % timing_count)
                            : 0xffffffffu;
                    if (timing != nullptr &&
                        prepared_timing != 0xffffffffu) {
                        timing[prepared_timing].node_fetch_start = prep_start;
                        timing[prepared_timing].kernarg_build_end =
                            prefetch_kernarg_end;
                        timing[prepared_timing].packet_stage_end =
                            wall_clock64();
                    }
                    mcu_log_event(
                        state, McuLogEvent::DispatchPublish, next_pc_dispatch,
                        next_node.variant_id, next_node.completion_slot,
                        next_generation, next_slot,
                        kernarg_base +
                            static_cast<uint64_t>(next_kernarg_slot) *
                                kernarg_stride);
                    have_prepared = true;
                }
            }
            pc = next_pc;
            continue;
        }

        if ((node.flags & kMcuNodeWait) != 0u) {
            const uint64_t observed_generation =
                have_active && active_completion < completion_count
                    ? static_cast<uint64_t>(__scoped_atomic_load_n(
                          &completions[active_completion].generation,
                          __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE))
                    : 0ull;
            mcu_log_event(state, McuLogEvent::WaitBegin, pc, node.variant_id,
                          node.completion_slot, active_generation,
                          have_active ? 1ull : 0ull, observed_generation);
            if (!have_active || node.completion_slot != active_completion) {
                mcu_log_event(state, McuLogEvent::CompletionSlotMismatch, pc,
                              node.variant_id, node.completion_slot,
                              active_generation, have_active ? 1ull : 0ull,
                              active_completion);
                mcu_fault(state,
                          McuFaultCode::completion_wait_state_mismatch, pc,
                          node.variant_id, active_generation);
                faulted = true;
                break;
            }
            if (pending_doorbell != 0u &&
                doorbell_mode == McuDoorbellMode::coalesce) {
                __threadfence_system();
                ring_aql_doorbell(queue, last_published_slot);
            mcu_log_event(state, McuLogEvent::DispatchDoorbell, pc, 0u, 0u, 0u,
                          last_published_slot, 1u);
                if (debug_records != nullptr && debug_record_count != 0u &&
                    ctx.dispatch_seq != 0u) {
                    McuDispatchRecord& rec =
                        debug_records[(ctx.dispatch_seq - 1u) %
                                      debug_record_count];
                    rec.doorbell_ts = wall_clock64();
                    rec.flags |= kMcuRecordDoorbell;
                }
                ++run_doorbell;
                pending_doorbell = 0u;
            }
            uint64_t spins = 0;
            while (!gpu_mcu_device_completion_matches(
                completions + active_completion, active_generation)) {
                if ((spins & kMcuCompletionStopCheckMask) == 0u) {
                    if (__scoped_atomic_load_n(&state->stop_requested,
                                               __ATOMIC_ACQUIRE,
                                               __MEMORY_SCOPE_SYSTEM) != 0u) {
                        supervisor = McuSupervisorState::stopping;
                        break;
                    }
                }
                if (++spins >= kMcuCompletionSpinLimit) {
                    mcu_log_event(state, McuLogEvent::WaitSpinTimeout, pc,
                                  node.variant_id, active_completion,
                                  active_generation, spins,
                                  static_cast<uint64_t>(active_completion));
                    mcu_fault(state,
                              McuFaultCode::completion_generation_mismatch,
                              pc, node.variant_id, active_generation);
                    faulted = true;
                    break;
                }
                __builtin_amdgcn_s_sleep(4);
            }
            if (faulted || supervisor == McuSupervisorState::stopping) {
                break;
            }
            mcu_log_event(state, McuLogEvent::WaitComplete, pc,
                          node.variant_id, active_completion, active_generation,
                          spins, ctx.completions_observed);
            if (timing != nullptr && active_timing != 0xffffffffu) {
                timing[active_timing].dependency_ready = wall_clock64();
            }
            ++ctx.completions_observed;
            have_active = false;
            active_completion = 0xffffffffu;
            pc = node.next == kMcuNoNext ? pc + 1u : node.next;
            continue;
        }

        mcu_fault(state, McuFaultCode::invalid_node, pc, node.variant_id, 0u);
        faulted = true;
        break;
    }

    if (faulted) {
        mcu_publish_done(state, ctx.run_consumed,
                         __scoped_atomic_load_n(&state->fault_code,
                                                __ATOMIC_ACQUIRE,
                                                __MEMORY_SCOPE_SYSTEM));
        supervisor = McuSupervisorState::fault;
        ctx.supervisor = static_cast<uint32_t>(supervisor);
        return true;
    } else if (supervisor == McuSupervisorState::stopping) {
        ctx.supervisor = static_cast<uint32_t>(supervisor);
        return true;
    } else {
        mcu_publish_counter(&state->region_end_ts, wall_clock64());
        mcu_publish_done(state, ctx.run_consumed, 0u);
        if (pending_doorbell != 0u &&
            doorbell_mode == McuDoorbellMode::coalesce) {
            __threadfence_system();
            ring_aql_doorbell(queue, last_published_slot);
            mcu_log_event(state, McuLogEvent::DispatchDoorbell, pc, 0u, 0u, 0u,
                          last_published_slot, 1u);
            if (debug_records != nullptr && debug_record_count != 0u &&
                ctx.dispatch_seq != 0u) {
                McuDispatchRecord& rec =
                    debug_records[(ctx.dispatch_seq - 1u) % debug_record_count];
                rec.doorbell_ts = wall_clock64();
                rec.flags |= kMcuRecordDoorbell;
            }
            ++run_doorbell;
            pending_doorbell = 0u;
        }
        ++ctx.plans_started;
        mcu_log_event(state, McuLogEvent::PlanEnd, 0u, 0u, 0u, 0u,
                      ctx.dispatches_committed, ctx.completions_observed);
        mcu_publish_counter(&state->dispatches_committed,
                            ctx.dispatches_committed);
        mcu_publish_counter(&state->completions_observed,
                            ctx.completions_observed);
        if (debug_records != nullptr && debug_record_count != 0u) {
            ctx.append_ahead_total += run_append_ahead;
            ctx.queue_empty_total += run_queue_empty;
            ctx.queue_full_total += run_queue_full;
            ctx.refill_total += run_refill;
            ctx.doorbell_total += run_doorbell;
            if (run_max_ahead > ctx.max_ahead_total) {
                ctx.max_ahead_total = run_max_ahead;
            }
            ctx.wrap_total += run_wrap;
            mcu_publish_counter(&state->append_ahead_count,
                                ctx.append_ahead_total);
            mcu_publish_counter(&state->queue_empty_count,
                                ctx.queue_empty_total);
            mcu_publish_counter(&state->backpressure_wait_count,
                                ctx.queue_full_total);
            mcu_publish_counter(&state->refill_count, ctx.refill_total);
            mcu_publish_counter(&state->doorbell_count, ctx.doorbell_total);
            mcu_publish_counter(&state->max_ahead, ctx.max_ahead_total);
            mcu_publish_counter(&state->wrap_count, ctx.wrap_total);
        }
        mcu_publish_counter(&state->plans_started, ctx.plans_started);
        supervisor = McuSupervisorState::idle;
        ctx.supervisor = static_cast<uint32_t>(supervisor);
        __scoped_atomic_store_n(&state->supervisor,
                                static_cast<uint32_t>(supervisor),
                                __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        return true;
    }
}
