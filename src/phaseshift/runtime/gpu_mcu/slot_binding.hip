#include <phaseshift/runtime/gpu_mcu/slot_binding.h>

#include <cstddef>

namespace ps::runtime::gpu_mcu {
namespace {

__global__ void gpu_mcu_binding_init_kernel(GpuMcuSlotBinding* bindings,
                                            uint32_t max_slots) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= max_slots) return;
    gpu_mcu_binding_init(bindings[index], index);
}

}  // namespace

GpuMcuSlotBindingTable::~GpuMcuSlotBindingTable() noexcept {
    const Status status = shutdown();
    (void)status;
}

Result<GpuMcuSlotBindingTable> GpuMcuSlotBindingTable::create(
    int device,
    uint32_t max_slots) {
    if (max_slots == 0u || max_slots > kGpuMcuMaxSlots) {
        return Status::invalid_argument("binding slot count out of range",
                                        __FILE__, __LINE__);
    }
    GpuMcuSlotBindingTable table{};
    table.max_slots_ = max_slots;
    table.device_ = device;

    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return Status::hip_error("hipSetDevice", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    void* ptr = nullptr;
    err = hipMalloc(&ptr, static_cast<std::size_t>(max_slots) *
                              sizeof(GpuMcuSlotBinding));
    if (err != hipSuccess) {
        return Status::hip_error("hipMalloc binding table",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    table.bindings_ = static_cast<GpuMcuSlotBinding*>(ptr);

    hipStream_t stream = nullptr;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        (void)table.shutdown();
        return Status::hip_error("hipStreamCreate binding init",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    const uint32_t block = 128u;
    const uint32_t grid = (max_slots + block - 1u) / block;
    gpu_mcu_binding_init_kernel<<<grid, block, 0u, stream>>>(table.bindings_,
                                                            max_slots);
    err = hipStreamSynchronize(stream);
    const hipError_t destroy_err = hipStreamDestroy(stream);
    if (err != hipSuccess) {
        (void)table.shutdown();
        return Status::hip_error("binding table init", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (destroy_err != hipSuccess) {
        (void)table.shutdown();
        return Status::hip_error("binding init stream destroy",
                                 hipGetErrorString(destroy_err), __FILE__,
                                 __LINE__);
    }
    return table;
}

Status GpuMcuSlotBindingTable::shutdown() noexcept {
    if (bindings_ == nullptr) {
        max_slots_ = 0u;
        return Status::make_ok();
    }
    hipError_t err = hipSetDevice(device_);
    if (err != hipSuccess) {
        return Status::hip_error("hipSetDevice shutdown",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    GpuMcuSlotBinding* bindings = bindings_;
    bindings_ = nullptr;
    max_slots_ = 0u;
    err = hipFree(bindings);
    if (err != hipSuccess) {
        return Status::hip_error("hipFree binding table",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

GpuMcuSlotBindingTable::GpuMcuSlotBindingTable(
    GpuMcuSlotBindingTable&& other) noexcept
    : bindings_(other.bindings_),
      max_slots_(other.max_slots_),
      device_(other.device_) {
    other.bindings_ = nullptr;
    other.max_slots_ = 0u;
}

GpuMcuSlotBindingTable& GpuMcuSlotBindingTable::operator=(
    GpuMcuSlotBindingTable&& other) noexcept {
    if (this != &other) {
        const Status status = shutdown();
        (void)status;
        bindings_ = other.bindings_;
        max_slots_ = other.max_slots_;
        device_ = other.device_;
        other.bindings_ = nullptr;
        other.max_slots_ = 0u;
    }
    return *this;
}

}  // namespace ps::runtime::gpu_mcu
