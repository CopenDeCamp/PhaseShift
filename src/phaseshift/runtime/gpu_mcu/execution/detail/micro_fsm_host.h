GpuMcuFsm::GpuMcuFsm(GpuMcuFsm&& other) noexcept { move_from(other); }

GpuMcuFsm& GpuMcuFsm::operator=(GpuMcuFsm&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        move_from(other);
    }
    return *this;
}

void GpuMcuFsm::move_from(GpuMcuFsm& other) noexcept {
    device_ = other.device_;
    host_allocation_ = other.host_allocation_;
    host_state_ = other.host_state_;
    device_state_ = other.device_state_;
    control_stream_ = other.control_stream_;
    lds_bytes_ = other.lds_bytes_;
    requested_runs_ = other.requested_runs_;
    launched_ = other.launched_;
    other.device_ = -1;
    other.host_allocation_ = nullptr;
    other.host_state_ = nullptr;
    other.device_state_ = nullptr;
    other.control_stream_ = nullptr;
    other.lds_bytes_ = 0;
    other.requested_runs_ = 0;
    other.launched_ = false;
}

Result<GpuMcuFsm> GpuMcuFsm::create(int device, hipStream_t control_stream) {
    if (device < 0) {
        return Status::invalid_argument("device must be >= 0", __FILE__, __LINE__);
    }
    if (control_stream == nullptr) {
        return Status::invalid_argument("control stream must not be null",
                                        __FILE__, __LINE__);
    }
    void* host = nullptr;
    if (hipHostMalloc(&host, sizeof(GpuMcuFsmState),
                      hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
        return Status::hip_error("hipHostMalloc(fsm state)", "allocation failed",
                                 __FILE__, __LINE__);
    }
    void* dev = nullptr;
    if (hipHostGetDevicePointer(&dev, host, 0) != hipSuccess) {
        (void)hipFreeHost(host);
        return Status::hip_error("hipHostGetDevicePointer(fsm state)",
                                 "mapping failed", __FILE__, __LINE__);
    }
    auto* host_state = static_cast<GpuMcuFsmState*>(host);
    *host_state = GpuMcuFsmState{};

    GpuMcuFsm out;
    out.device_ = device;
    out.host_allocation_ = host;
    out.host_state_ = host_state;
    out.device_state_ = static_cast<GpuMcuFsmState*>(dev);
    out.control_stream_ = control_stream;
    return out;
}

Status GpuMcuFsm::configure(const GpuMcuFsmConfig& config) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("fsm not created", __FILE__, __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("fsm already started", __FILE__, __LINE__);
    }
    if (config.node_count == 0u || config.node_count > kMcuMaxNodes) {
        return Status::invalid_argument("invalid node count", __FILE__, __LINE__);
    }
    if (config.variant_count == 0u || config.variant_count > kMcuMaxVariants) {
        return Status::invalid_argument("invalid variant count", __FILE__, __LINE__);
    }
    if (config.completion_count == 0u ||
        config.completion_count > kMcuMaxCompletionSlots) {
        return Status::invalid_argument("invalid completion count", __FILE__,
                                        __LINE__);
    }
    if (config.kernarg_slot_count == 0u ||
        config.kernarg_slot_count > kMcuMaxKernargSlots) {
        return Status::invalid_argument("invalid kernarg slot count", __FILE__,
                                        __LINE__);
    }
    if (config.timing_count > kMcuMaxTiming) {
        return Status::invalid_argument("invalid timing count", __FILE__, __LINE__);
    }
    if (config.debug_record_count > kMcuMaxDebugRecords) {
        return Status::invalid_argument("invalid debug record count", __FILE__,
                                        __LINE__);
    }
    if (config.doorbell_mode >
        static_cast<uint32_t>(McuDoorbellMode::first_only)) {
        return Status::invalid_argument("invalid doorbell mode", __FILE__,
                                        __LINE__);
    }
    if (config.plan == nullptr || config.variants == nullptr ||
        config.retained == nullptr || config.completions == nullptr) {
        return Status::invalid_argument("null fsm table", __FILE__, __LINE__);
    }
    if (config.kernarg_source_count > kMcuMaxNodes) {
        return Status::invalid_argument("invalid kernarg source count", __FILE__,
                                        __LINE__);
    }
    if (config.kernarg_source_count != 0u &&
        config.kernarg_sources == nullptr) {
        return Status::invalid_argument("null kernarg source table", __FILE__,
                                        __LINE__);
    }
    for (uint32_t i = 0; i < config.kernarg_source_count; ++i) {
        const McuKernargSourceDesc& source = config.kernarg_sources[i];
        if ((source.flags & ~(kMcuKernargSourcePrepared |
                              kMcuKernargSourceSupervisorProbe)) != 0u) {
            return Status::invalid_argument("unknown kernarg source flags",
                                            __FILE__, __LINE__);
        }
        if ((source.flags & kMcuKernargSourcePrepared) != 0u &&
            (source.source == 0ull || source.explicit_args_bytes == 0u)) {
            return Status::invalid_argument("incomplete prepared kernarg source",
                                            __FILE__, __LINE__);
        }
        if ((source.flags & kMcuKernargSourceSupervisorProbe) != 0u &&
            source.source != 0ull) {
            return Status::invalid_argument("supervisor probe source must be empty",
                                            __FILE__, __LINE__);
        }
    }

    for (uint32_t i = 0; i < config.variant_count; ++i) {
        const McuKernelVariantDesc& v = config.variants[i];
        if (v.hidden_args_policy >
            static_cast<uint32_t>(AqlHiddenArgsPolicy::Required)) {
            return Status::invalid_argument("invalid hidden args policy", __FILE__,
                                            __LINE__);
        }
        if (config.kernarg_slot_stride != 0u &&
            config.kernarg_slot_stride <
                aql_kernarg_slot_stride(v.kernarg_size)) {
            return Status::invalid_state(
                "kernarg slot stride does not cover a variant segment", __FILE__,
                __LINE__);
        }
    }
    for (uint32_t i = 0; i < config.node_count; ++i) {
        const McuPlanNode& node = config.plan[i];
        if (node.variant_id >= config.variant_count) continue;
        const McuKernelVariantDesc& v = config.variants[node.variant_id];
        if (static_cast<AqlHiddenArgsPolicy>(v.hidden_args_policy) !=
            AqlHiddenArgsPolicy::Required) {
            continue;
        }
        const McuKernargSourceDesc* source =
            config.kernarg_sources != nullptr && i < config.kernarg_source_count
                ? &config.kernarg_sources[i]
                : nullptr;
        const std::size_t explicit_bytes =
            source != nullptr ? source->explicit_args_bytes : 0u;
        if (explicit_bytes == 0u ||
            !aql_hidden_args_fits(explicit_bytes, v.kernarg_size)) {
            return Status::invalid_state(
                "kernarg source does not cover the required hidden args",
                __FILE__, __LINE__);
        }
    }
    if (config.retained_count < config.variant_count) {
        return Status::invalid_argument("retained table smaller than variants",
                                        __FILE__, __LINE__);
    }
    if (config.template_source != 0u) {
        const uint64_t needed =
            static_cast<uint64_t>(config.retained_count) *
            sizeof(GpuMcuRetainedPacket);
        if (needed > config.lds_budget_bytes) {
            return Status::invalid_state(
                "retained templates exceed the LDS budget", __FILE__, __LINE__);
        }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_) != hipSuccess) {
        return Status::hip_error("hipGetDeviceProperties", "query failed",
                                 __FILE__, __LINE__);
    }
    if (config.lds_budget_bytes > prop.sharedMemPerBlock) {
        return Status::invalid_argument("LDS budget exceeds device limit",
                                        __FILE__, __LINE__);
    }

    auto* s = host_state_;
    s->queue = config.queue;
    s->plan = config.plan;
    s->node_count = config.node_count;
    s->plan_id = config.plan_id;
    s->variants = config.variants;
    s->variant_count = config.variant_count;
    s->dynamic_node_bindings = config.dynamic_node_bindings;
    s->dynamic_node_binding_count = config.dynamic_node_binding_count;
    s->kernarg_sources = config.kernarg_sources;
    s->kernarg_source_count = config.kernarg_source_count;
    s->external_start_signal = config.start_signal;
    s->external_done_signal = config.done_signal;
    s->external_result_code = config.result_code;
    s->log_base = config.log_base;
    s->log_head = 0u;
    s->log_enabled = config.log_base != 0ull ? 1u : 0u;
    s->kernarg_base = config.kernarg_base;
    s->kernarg_slot_stride = config.kernarg_slot_stride;
    s->kernarg_slot_count = config.kernarg_slot_count != 0u
                                ? config.kernarg_slot_count
                                : config.node_count;
    s->completions = config.completions;
    s->completion_count = config.completion_count;
    s->output_base = config.output_base;
    s->timestamps_base = config.timestamps_base;
    s->retained_vram = const_cast<GpuMcuRetainedPacket*>(config.retained);
    s->retained_count = config.retained_count;
    s->template_source = config.template_source;
    s->copy_lanes = config.copy_lanes;
    s->lds_budget_bytes = config.lds_budget_bytes;
    s->barrier = config.barrier;
    s->prepared_enabled = config.prepared_enabled;
    s->idle_sleep = config.idle_sleep;
    s->timing = config.timing;
    s->timing_count = config.timing_count;
    s->queue_ahead_depth = config.queue_ahead_depth;
    s->doorbell_batch = config.doorbell_batch != 0u ? config.doorbell_batch : 1u;
    s->doorbell_mode = config.doorbell_mode;
    s->issue_wait_running = config.issue_wait_running;
    s->debug_records = config.debug_records;
    s->debug_record_count = config.debug_record_count;
    lds_bytes_ = config.template_source != 0u ? config.lds_budget_bytes : 0u;
    return Status::make_ok();
}

Status GpuMcuFsm::start() {
    if (host_state_ == nullptr) {
        return Status::invalid_state("fsm not created", __FILE__, __LINE__);
    }
    if (launched_) {
        return Status::invalid_state("fsm already started", __FILE__, __LINE__);
    }
    (void)hipGetLastError();
    gpu_mcu_fsm_kernel<<<1, 32, lds_bytes_, control_stream_>>>(device_state_);
    if (hipGetLastError() != hipSuccess) {
        return Status::hip_error("fsm launch", "launch failed", __FILE__, __LINE__);
    }
    launched_ = true;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (load_u32(&host_state_->started) == 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::invalid_state("fsm did not start", __FILE__, __LINE__);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return Status::make_ok();
}

Status GpuMcuFsm::request_run() {
    if (host_state_ == nullptr) {
        return Status::invalid_state("fsm not created", __FILE__, __LINE__);
    }
    requested_runs_ += 1;
    store_u64(&host_state_->run_request, requested_runs_);
    return Status::make_ok();
}

Status GpuMcuFsm::wait_plans(uint64_t count, uint32_t timeout_ms) const {
    if (host_state_ == nullptr) {
        return Status::invalid_state("fsm not created", __FILE__, __LINE__);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (load_u32(&host_state_->fault_code) != 0u) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "fsm fault code %u at pc %u",
                          load_u32(&host_state_->fault_code),
                          load_u32(&host_state_->fault_pc));
            return Status::invalid_state(buf, __FILE__, __LINE__);
        }
        if (load_u64(&host_state_->plans_started) >= count) {
            return Status::make_ok();
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::invalid_state("fsm plan wait timed out", __FILE__,
                                         __LINE__);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

bool GpuMcuFsm::running() const noexcept {
    return host_state_ != nullptr && load_u32(&host_state_->started) != 0u;
}

McuSupervisorState GpuMcuFsm::supervisor() const noexcept {
    if (host_state_ == nullptr) return McuSupervisorState::boot;
    return static_cast<McuSupervisorState>(load_u32(&host_state_->supervisor));
}

McuFaultCode GpuMcuFsm::fault_code() const noexcept {
    if (host_state_ == nullptr) return McuFaultCode::none;
    return static_cast<McuFaultCode>(load_u32(&host_state_->fault_code));
}

uint32_t GpuMcuFsm::fault_pc() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u32(&host_state_->fault_pc);
}

uint32_t GpuMcuFsm::log_head() const noexcept {
    if (host_state_ == nullptr) return 0u;
    return load_u32(&host_state_->log_head);
}

const McuLogRecord* GpuMcuFsm::log_record(uint32_t slot) const noexcept {
    if (host_state_ == nullptr || host_state_->log_base == 0ull) return nullptr;
    if (slot >= kMcuLogLineCount) return nullptr;
    return reinterpret_cast<const McuLogRecord*>(
        host_state_->log_base +
        static_cast<uint64_t>(slot) * kMcuLogLineBytes);
}

uint64_t GpuMcuFsm::plans_started() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->plans_started);
}

uint64_t GpuMcuFsm::dispatches() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->dispatches_committed);
}

uint64_t GpuMcuFsm::completions_observed() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->completions_observed);
}

uint64_t GpuMcuFsm::heartbeat() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->heartbeat);
}

uint64_t GpuMcuFsm::append_ahead_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->append_ahead_count);
}

uint64_t GpuMcuFsm::queue_empty_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->queue_empty_count);
}

uint64_t GpuMcuFsm::mid_region_queue_empty_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->mid_region_queue_empty_count);
}

uint64_t GpuMcuFsm::backpressure_wait_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->backpressure_wait_count);
}

uint64_t GpuMcuFsm::refill_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->refill_count);
}

uint64_t GpuMcuFsm::doorbell_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->doorbell_count);
}

uint64_t GpuMcuFsm::max_ahead() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->max_ahead);
}

uint32_t GpuMcuFsm::external_epoch_seen() const noexcept {
    if (host_state_ == nullptr) return 0;
    return std::atomic_ref<uint32_t>(host_state_->external_epoch_seen)
        .load(std::memory_order_acquire);
}

uint64_t GpuMcuFsm::wrap_count() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->wrap_count);
}

uint64_t GpuMcuFsm::region_start_ts() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->region_start_ts);
}

uint64_t GpuMcuFsm::region_end_ts() const noexcept {
    if (host_state_ == nullptr) return 0;
    return load_u64(&host_state_->region_end_ts);
}

Status GpuMcuFsm::request_stop() {
    if (host_state_ == nullptr) {
        return Status::invalid_state("fsm not created", __FILE__, __LINE__);
    }
    std::atomic_ref<uint32_t>(host_state_->stop_requested)
        .store(1u, std::memory_order_release);
    return Status::make_ok();
}

Status GpuMcuFsm::wait_stopped(uint32_t timeout_ms) {
    if (host_state_ == nullptr) {
        return Status::invalid_state("fsm not created", __FILE__, __LINE__);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (load_u32(&host_state_->started) != 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::invalid_state("fsm did not stop", __FILE__, __LINE__);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return Status::make_ok();
}

Status GpuMcuFsm::shutdown() noexcept {
    if (host_state_ == nullptr) return Status::make_ok();
    Status first = Status::make_ok();
    auto note = [&](Status s) {
        if (!s.ok() && first.ok()) first = s;
    };
    if (launched_) {
        note(request_stop());
        Status stopped = wait_stopped(5000);
        if (!stopped.ok()) {
            note(request_stop());
            stopped = wait_stopped(5000);
        }
        if (!stopped.ok()) {
            return Status::invalid_state(
                "mcu fsm did not stop; refusing to release resources while the "
                "device loop is alive",
                __FILE__, __LINE__);
        }
        if (hipStreamSynchronize(control_stream_) != hipSuccess) {
            note(Status::hip_error("hipStreamSynchronize(fsm)", "sync failed",
                                   __FILE__, __LINE__));
        }
        launched_ = false;
    }
    void* host = host_allocation_;
    host_allocation_ = nullptr;
    host_state_ = nullptr;
    device_state_ = nullptr;
    if (host != nullptr && hipFreeHost(host) != hipSuccess) {
        note(Status::hip_error("hipFreeHost(fsm state)", "free failed", __FILE__,
                               __LINE__));
    }
    return first;
}

