#include <phaseshift/runtime/gpu_mcu/infrastructure/wall_clock.h>

#include <hip/hip_runtime.h>

namespace ps {
namespace runtime {
namespace gpu_mcu {

Result<int> gpu_wall_clock_rate_khz(int device) {
    int rate = 0;
    const hipError_t err =
        hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, device);
    if (err != hipSuccess) {
        return Status::hip_error("gpu wall clock rate query", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (rate <= 0) {
        return Status::invalid_state(
            "gpu wall clock rate unavailable; timing cannot be converted",
            __FILE__, __LINE__);
    }
    return rate;
}

double gpu_wall_clock_ticks_to_ms(uint64_t ticks, int wall_clock_rate_khz) {
    if (wall_clock_rate_khz <= 0) return 0.0;
    return static_cast<double>(ticks) /
           static_cast<double>(wall_clock_rate_khz);
}

double gpu_wall_clock_ticks_to_us(uint64_t ticks, int wall_clock_rate_khz) {
    if (wall_clock_rate_khz <= 0) return 0.0;
    return static_cast<double>(ticks) * 1000.0 /
           static_cast<double>(wall_clock_rate_khz);
}

}  // namespace gpu_mcu
}  // namespace runtime
}  // namespace ps
