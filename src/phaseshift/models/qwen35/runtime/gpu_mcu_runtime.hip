#include <phaseshift/models/qwen35/runtime/gpu_mcu_runtime.h>

#include <cstddef>
#include <cstring>
#include <utility>

namespace ps {
namespace qwen35 {
namespace runtime {

GpuMcuRuntime::GpuMcuRuntime(
    Executor& executor,
    SequenceSlotPool& seq_pool,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    GpuMcuRuntimeConfig config,
    hipStream_t stream)
    : executor_(executor),
      seq_pool_(seq_pool),
      gdn_pool_(gdn_pool),
      kv_pool_(kv_pool),
      config_(std::move(config)),
      stream_(stream) {}

GpuMcuRuntime::~GpuMcuRuntime() noexcept {
    if (!shutdown_) {
        (void)shutdown();
    }
}

Result<uint64_t> GpuMcuRuntime::submit(
    std::vector<int32_t> input_tokens,
    uint32_t max_new_tokens,
    const SamplingConfig& sampling) {
    (void)input_tokens;
    (void)max_new_tokens;
    (void)sampling;
    if (shutdown_) {
        return Status::invalid_state("gpu-mcu runtime is shut down",
                                     __FILE__, __LINE__);
    }
    if (config_.max_scheduled_requests == 0u) {
        return Status::invalid_state(
            "gpu-mcu runtime is not configured", __FILE__, __LINE__);
    }
    return Status::unsupported(
        "gpu-mcu submit bridge is not connected", __FILE__,
        __LINE__);
}

Result<StepResult> GpuMcuRuntime::step() {
    if (shutdown_) {
        return Status::invalid_state("gpu-mcu runtime is shut down",
                                     __FILE__, __LINE__);
    }
    StepResult result{};
    for (auto it = requests_.begin(); it != requests_.end();) {
        if (it->state != RequestState::Finished) {
            ++it;
            continue;
        }
        request_index_.erase(it->id);
        it = requests_.erase(it);
        result.executed = true;
        ++result.num_requests;
    }
    return result;
}

Status GpuMcuRuntime::cancel(uint64_t id) {
    if (shutdown_) {
        return Status::invalid_state("gpu-mcu runtime is shut down",
                                     __FILE__, __LINE__);
    }
    if (request_index_.find(id) == request_index_.end()) {
        return Status::invalid_argument("unknown gpu-mcu request id", __FILE__,
                                        __LINE__);
    }
    return Status::unsupported(
        "gpu-mcu cancel bridge is not connected", __FILE__,
        __LINE__);
}

Status GpuMcuRuntime::retire(uint64_t id) {
    const auto found = request_index_.find(id);
    if (found == request_index_.end()) {
        return Status::invalid_argument("unknown gpu-mcu request id", __FILE__,
                                        __LINE__);
    }
    if (found->second->state != RequestState::Finished) {
        return Status::invalid_argument("gpu-mcu request is not terminal",
                                        __FILE__, __LINE__);
    }
    requests_.erase(found->second);
    request_index_.erase(found);
    return Status::make_ok();
}

bool GpuMcuRuntime::has_pending() const noexcept {
    return !requests_.empty();
}

const RuntimeRequest* GpuMcuRuntime::request(uint64_t id) const {
    const auto found = request_index_.find(id);
    if (found == request_index_.end()) return nullptr;
    return &*found->second;
}

Status GpuMcuRuntime::cancel_all() {
    if (requests_.empty()) return Status::make_ok();
    return Status::unsupported(
        "gpu-mcu cancel bridge is not connected", __FILE__,
        __LINE__);
}

Status GpuMcuRuntime::allocate_controller_resources(int device) {
    if (controller_resources_ready_) return Status::make_ok();
    if (config_.max_scheduled_requests == 0u) {
        return Status::invalid_state(
            "gpu-mcu runtime slot capacity is zero", __FILE__,
            __LINE__);
    }

    auto control = ::ps::runtime::gpu_mcu::GpuMcuControlRing::create(device);
    if (!control.ok()) return control.status();
    control_ring_ = control.release();

    auto ingress = ::ps::runtime::gpu_mcu::GpuMcuRequestIngressArena::create(
        device, config_.max_scheduled_requests);
    if (!ingress.ok()) {
        (void)control_ring_.shutdown();
        control_ring_ = ::ps::runtime::gpu_mcu::GpuMcuControlRing{};
        return ingress.status();
    }
    ingress_ = ingress.release();

    auto output = ::ps::runtime::gpu_mcu::GpuMcuOutputRing::create(device);
    if (!output.ok()) {
        (void)ingress_.shutdown();
        (void)control_ring_.shutdown();
        ingress_ = ::ps::runtime::gpu_mcu::GpuMcuRequestIngressArena{};
        control_ring_ = ::ps::runtime::gpu_mcu::GpuMcuControlRing{};
        return output.status();
    }
    output_ring_ = output.release();

    auto slots =
        ::ps::runtime::gpu_mcu::GpuMcuSlotTable::create(
            device, config_.max_scheduled_requests);
    if (!slots.ok()) {
        (void)output_ring_.shutdown();
        (void)ingress_.shutdown();
        (void)control_ring_.shutdown();
        output_ring_ = ::ps::runtime::gpu_mcu::GpuMcuOutputRing{};
        ingress_ = ::ps::runtime::gpu_mcu::GpuMcuRequestIngressArena{};
        control_ring_ = ::ps::runtime::gpu_mcu::GpuMcuControlRing{};
        return slots.status();
    }
    slot_table_ = slots.release();

    max_slots_ = slot_table_.max_slots();
    admission_capacity_ = config_.max_scheduled_requests;
    const std::size_t runnable_bytes =
        static_cast<std::size_t>(max_slots_) *
        sizeof(::ps::runtime::RequestHandle);
    const std::size_t runtime_bytes =
        static_cast<std::size_t>(max_slots_) *
        sizeof(::ps::runtime::gpu_mcu::GpuMcuSlotRuntimeState);
    const std::size_t admission_bytes =
        static_cast<std::size_t>(admission_capacity_) *
        sizeof(::ps::runtime::gpu_mcu::GpuMcuPendingAdmission);

    const hipError_t rerr =
        hipMalloc(reinterpret_cast<void**>(&runnable_), runnable_bytes);
    const hipError_t serr =
        rerr == hipSuccess
            ? hipMalloc(reinterpret_cast<void**>(&slot_runtime_), runtime_bytes)
            : rerr;
    const hipError_t aerr =
        serr == hipSuccess
            ? hipMalloc(reinterpret_cast<void**>(&admission_), admission_bytes)
            : serr;
    if (aerr != hipSuccess) {
        if (runnable_ != nullptr) (void)hipFree(runnable_);
        if (slot_runtime_ != nullptr) (void)hipFree(slot_runtime_);
        runnable_ = nullptr;
        slot_runtime_ = nullptr;
        (void)slot_table_.shutdown();
        (void)output_ring_.shutdown();
        (void)ingress_.shutdown();
        (void)control_ring_.shutdown();
        slot_table_ = ::ps::runtime::gpu_mcu::GpuMcuSlotTable{};
        output_ring_ = ::ps::runtime::gpu_mcu::GpuMcuOutputRing{};
        ingress_ = ::ps::runtime::gpu_mcu::GpuMcuRequestIngressArena{};
        control_ring_ = ::ps::runtime::gpu_mcu::GpuMcuControlRing{};
        max_slots_ = 0u;
        admission_capacity_ = 0u;
        return Status::hip_error("gpu-mcu controller allocation",
                                 hipGetErrorString(aerr), __FILE__, __LINE__);
    }

    std::memset(runnable_, 0, runnable_bytes);
    std::memset(slot_runtime_, 0, runtime_bytes);
    std::memset(admission_, 0, admission_bytes);
    controller_resources_ready_ = true;
    return Status::make_ok();
}

Status GpuMcuRuntime::shutdown() {
    if (shutdown_) return Status::make_ok();
    shutdown_ = true;
    request_index_.clear();
    requests_.clear();

    Status first_error = Status::make_ok();
    if (admission_ != nullptr) {
        const hipError_t err = hipFree(admission_);
        if (err != hipSuccess && first_error.ok()) {
            first_error = Status::hip_error("hipFree gpu-mcu admission queue",
                                            hipGetErrorString(err), __FILE__,
                                            __LINE__);
        }
        admission_ = nullptr;
    }
    if (slot_runtime_ != nullptr) {
        const hipError_t err = hipFree(slot_runtime_);
        if (err != hipSuccess && first_error.ok()) {
            first_error = Status::hip_error("hipFree gpu-mcu slot runtime",
                                            hipGetErrorString(err), __FILE__,
                                            __LINE__);
        }
        slot_runtime_ = nullptr;
    }
    if (runnable_ != nullptr) {
        const hipError_t err = hipFree(runnable_);
        if (err != hipSuccess && first_error.ok()) {
            first_error = Status::hip_error("hipFree gpu-mcu runnable handles",
                                            hipGetErrorString(err), __FILE__,
                                            __LINE__);
        }
        runnable_ = nullptr;
    }
    if (controller_resources_ready_) {
        const Status slot_status = slot_table_.shutdown();
        if (!slot_status.ok() && first_error.ok()) first_error = slot_status;
        const Status output_status = output_ring_.shutdown();
        if (!output_status.ok() && first_error.ok()) first_error = output_status;
        const Status ingress_status = ingress_.shutdown();
        if (!ingress_status.ok() && first_error.ok()) first_error = ingress_status;
        const Status control_status = control_ring_.shutdown();
        if (!control_status.ok() && first_error.ok()) {
            first_error = control_status;
        }
        controller_resources_ready_ = false;
    }
    max_slots_ = 0u;
    admission_capacity_ = 0u;
    return first_error;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
