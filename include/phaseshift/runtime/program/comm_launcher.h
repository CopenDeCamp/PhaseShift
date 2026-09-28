#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::runtime {

using CommLaunchFn = Status (*)(void* self, const CommDescriptor& descriptor,
                                const void* send, void* recv, uint64_t elements,
                                hipStream_t stream);

}
