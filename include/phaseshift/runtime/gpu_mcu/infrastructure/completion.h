#pragma once

#include <phaseshift/core/status.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

namespace ps::runtime::gpu_mcu {

enum class GpuMcuCompletionState : uint32_t {
    idle = 0,
    running = 1,
    complete = 2,
    error = 3,
};

struct alignas(64) GpuMcuCompletion {
    uint64_t sequence = 0;
    uint32_t state = 0;
    uint32_t detail = 0;
    uint64_t aux = 0;
    uint32_t reserved[10] = {};
};

static_assert(sizeof(GpuMcuCompletion) == 64);
static_assert(alignof(GpuMcuCompletion) == 64);
static_assert(offsetof(GpuMcuCompletion, sequence) == 0);
static_assert(offsetof(GpuMcuCompletion, state) == 8);

struct GpuMcuCompletionSnapshot {
    uint64_t sequence = 0;
    GpuMcuCompletionState state = GpuMcuCompletionState::idle;
    uint32_t detail = 0;
    uint64_t aux = 0;

    bool is_running() const noexcept { return state == GpuMcuCompletionState::running; }
    bool is_complete() const noexcept { return state == GpuMcuCompletionState::complete; }
    bool is_error() const noexcept { return state == GpuMcuCompletionState::error; }
};

__device__ __forceinline__ void gpu_mcu_completion_publish(
    GpuMcuCompletion* completion,
    uint32_t state,
    uint32_t detail,
    uint64_t sequence) {
    completion->detail = detail;
    __scoped_atomic_store_n(&completion->sequence, sequence, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    __scoped_atomic_store_n(&completion->state, state, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
}

class GpuMcuCompletionWord {
public:
    GpuMcuCompletionWord() = default;
    ~GpuMcuCompletionWord() noexcept { (void)shutdown(); }

    GpuMcuCompletionWord(const GpuMcuCompletionWord&) = delete;
    GpuMcuCompletionWord& operator=(const GpuMcuCompletionWord&) = delete;

    GpuMcuCompletionWord(GpuMcuCompletionWord&& other) noexcept { move_from(other); }
    GpuMcuCompletionWord& operator=(GpuMcuCompletionWord&& other) noexcept {
        if (this != &other) {
            (void)shutdown();
            move_from(other);
        }
        return *this;
    }

    static Result<GpuMcuCompletionWord> create() {
        void* host = nullptr;
        if (hipHostMalloc(&host, sizeof(GpuMcuCompletion),
                          hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
            return Status::hip_error("hipHostMalloc(completion)", "allocation failed",
                                     __FILE__, __LINE__);
        }
        void* device = nullptr;
        if (hipHostGetDevicePointer(&device, host, 0) != hipSuccess) {
            (void)hipFreeHost(host);
            return Status::hip_error("hipHostGetDevicePointer(completion)",
                                     "mapping failed", __FILE__, __LINE__);
        }
        GpuMcuCompletionWord out;
        out.host_ = static_cast<GpuMcuCompletion*>(host);
        out.device_ = static_cast<GpuMcuCompletion*>(device);
        out.host_->sequence = 0;
        out.host_->state = static_cast<uint32_t>(GpuMcuCompletionState::idle);
        out.host_->detail = 0;
        out.host_->aux = 0;
        return out;
    }

    bool valid() const noexcept { return host_ != nullptr; }

    GpuMcuCompletion* device_ptr() const noexcept { return device_; }
    GpuMcuCompletion* host_ptr() const noexcept { return host_; }

    void reset() noexcept {
        if (host_ == nullptr) return;
        host_->sequence = 0;
        host_->detail = 0;
        host_->aux = 0;
        std::atomic_ref<uint32_t>(host_->state)
            .store(static_cast<uint32_t>(GpuMcuCompletionState::idle),
                   std::memory_order_release);
    }

    GpuMcuCompletionSnapshot observe() const noexcept {
        GpuMcuCompletionSnapshot out{};
        if (host_ == nullptr) return out;
        out.state = static_cast<GpuMcuCompletionState>(
            std::atomic_ref<uint32_t>(host_->state).load(std::memory_order_acquire));
        out.sequence = host_->sequence;
        out.detail = host_->detail;
        out.aux = host_->aux;
        return out;
    }

    Result<GpuMcuCompletionSnapshot> wait_for(
        GpuMcuCompletionState wanted,
        uint32_t timeout_ms) const {
        if (host_ == nullptr) {
            return Status::invalid_state("completion word not created", __FILE__, __LINE__);
        }
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (true) {
            const GpuMcuCompletionSnapshot snap = observe();
            if (snap.state == wanted || snap.is_error()) return snap;
            if (std::chrono::steady_clock::now() >= deadline) {
                return Status::invalid_state("completion wait timed out", __FILE__, __LINE__);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    Status shutdown() noexcept {
        if (host_ == nullptr) return Status::make_ok();
        void* host = host_;
        host_ = nullptr;
        device_ = nullptr;
        const hipError_t err = hipFreeHost(host);
        if (err != hipSuccess) {
            return Status::hip_error("hipFreeHost(completion)", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        return Status::make_ok();
    }

private:
    void move_from(GpuMcuCompletionWord& other) noexcept {
        host_ = other.host_;
        device_ = other.device_;
        other.host_ = nullptr;
        other.device_ = nullptr;
    }

    GpuMcuCompletion* host_ = nullptr;
    GpuMcuCompletion* device_ = nullptr;
};

}  // namespace ps::runtime::gpu_mcu
