#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>

namespace ps {
namespace runtime {
namespace gpu_mcu {

Result<int> gpu_wall_clock_rate_khz(int device);

double gpu_wall_clock_ticks_to_ms(uint64_t ticks, int wall_clock_rate_khz);

double gpu_wall_clock_ticks_to_us(uint64_t ticks, int wall_clock_rate_khz);

}  // namespace gpu_mcu
}  // namespace runtime
}  // namespace ps
