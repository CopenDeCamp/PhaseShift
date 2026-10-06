__device__ __forceinline__ void mcu_log_event(GpuMcuFsmState* state,
                                              McuLogEvent event, uint32_t pc,
                                              uint32_t variant,
                                              uint32_t completion_slot,
                                              uint64_t generation,
                                              uint64_t value0,
                                              uint64_t value1) {
    if (state->log_enabled == 0u) return;
    const uint64_t seq = static_cast<uint64_t>(
        __atomic_fetch_add(&state->log_head, 1u, __ATOMIC_RELAXED));
    const uint32_t slot = static_cast<uint32_t>(seq) & (kMcuLogLineCount - 1u);
    auto* rec = reinterpret_cast<McuLogRecord*>(
        state->log_base + static_cast<uint64_t>(slot) * kMcuLogLineBytes);
    rec->sequence = seq;
    rec->clock = wall_clock64();
    rec->event = static_cast<uint32_t>(event);
    rec->pc = pc;
    rec->variant = variant;
    rec->completion_slot = completion_slot;
    rec->generation = generation;
    rec->value0 = value0;
    rec->value1 = value1;
    rec->message[0] = '\0';
}

__device__ __forceinline__ void mcu_log_fault(GpuMcuFsmState* state,
                                              McuFaultCode code, uint32_t pc,
                                              uint32_t variant,
                                              uint64_t generation) {
    const McuLogEvent event =
        code == McuFaultCode::kernarg_build_failed
            ? McuLogEvent::KernargBuildFailed
            : McuLogEvent::Fault;
    mcu_log_event(state, event, pc, variant, 0u, generation,
                  static_cast<uint64_t>(code), 0u);
}
