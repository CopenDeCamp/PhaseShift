#pragma once

#include <cstddef>

namespace ps {
namespace runtime {
namespace gpu_mcu {

struct GpuMcuEmbeddedKernel {
    const unsigned char* blob = nullptr;
    std::size_t size = 0;
};

GpuMcuEmbeddedKernel gpu_mcu_embedded_kernel(const char* name) noexcept;

}  // namespace gpu_mcu
}  // namespace runtime
}  // namespace ps
