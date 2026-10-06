#pragma once

#include <phaseshift/models/qwen35/runtime/decode_backend.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>

#include <cstdint>

namespace ps {
namespace qwen35 {
namespace runtime {

enum class McuPlanShape : uint8_t {
    Direct = 0,
    Split = 1,
    Count = 2,
};

struct McuPlanFingerprint {
    uint64_t hash = 0;
    uint32_t dispatch_count = 0;
    uint32_t variant_count = 0;

    bool operator==(const McuPlanFingerprint& other) const noexcept {
        return hash == other.hash && dispatch_count == other.dispatch_count &&
               variant_count == other.variant_count;
    }
    bool operator!=(const McuPlanFingerprint& other) const noexcept {
        return !(*this == other);
    }
};

McuPlanFingerprint mcu_plan_fingerprint(const McuCompiledPlan& plan) noexcept;

struct McuPlanCacheStep {
    bool needs_switch = false;
    bool needs_upload = false;
    bool reused = false;
};

class McuPlanCache {
public:
    void reset() noexcept {
        active_ = false;
        active_shape_ = McuPlanShape::Direct;
        active_fingerprint_ = McuPlanFingerprint{};
        counters_ = DecodeRuntimeCounters{};
    }

    McuPlanCacheStep observe(McuPlanShape shape,
                             const McuPlanFingerprint& fingerprint) noexcept {
        McuPlanCacheStep step{};
        if (active_ && shape == active_shape_ &&
            fingerprint == active_fingerprint_) {
            step.reused = true;
            return step;
        }
        ++counters_.plan_switch_count;
        ++counters_.plan_upload_count;
        active_ = true;
        active_shape_ = shape;
        active_fingerprint_ = fingerprint;
        step.needs_switch = true;
        step.needs_upload = true;
        return step;
    }

    bool has_active() const noexcept { return active_; }
    McuPlanShape active_shape() const noexcept { return active_shape_; }
    const McuPlanFingerprint& active_fingerprint() const noexcept {
        return active_fingerprint_;
    }
    const DecodeRuntimeCounters& counters() const noexcept { return counters_; }
    void note_compile() noexcept { ++counters_.plan_compile_count; }
    void note_host_fallback() noexcept { ++counters_.host_fallback_count; }
    void note_mcu_run() noexcept { ++counters_.mcu_run_count; }

private:
    bool active_ = false;
    McuPlanShape active_shape_ = McuPlanShape::Direct;
    McuPlanFingerprint active_fingerprint_{};
    DecodeRuntimeCounters counters_{};
};

const char* mcu_plan_shape_name(McuPlanShape shape) noexcept;

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
