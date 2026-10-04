#include <phaseshift/runtime/gpu_mcu/worker_image.h>

#include <phaseshift_gpu_mcu_probe_worker_hsaco.inc>

namespace ps::runtime::gpu_mcu {

GpuMcuWorkerImage gpu_mcu_worker_image() {
    return GpuMcuWorkerImage{
        reinterpret_cast<const unsigned char*>(phaseshift_gpu_mcu_probe_worker_hsaco),
        phaseshift_gpu_mcu_probe_worker_hsaco_len,
    };
}

}  // namespace ps::runtime::gpu_mcu
