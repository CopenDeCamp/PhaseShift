#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace ps::runtime::gpu_mcu {

inline constexpr uint32_t kGpuMcuCusPerWgp = 2u;

struct GpuMcuMaskSemantics {
    uint32_t reported_multi_processor_count = 0;
    uint32_t hip_mask_unit_count = 0;
    uint32_t hsa_compute_unit_count = 0;
    bool full_mask_readback_exact = false;
    bool single_bit_readback_exact = false;
    bool unpaired_bits_readback_exact = false;
};

Result<GpuMcuMaskSemantics> gpu_mcu_probe_mask_semantics(int device);

class GpuMcuCuPartition {
public:
    GpuMcuCuPartition() = default;
    ~GpuMcuCuPartition() noexcept { (void)shutdown(); }

    GpuMcuCuPartition(const GpuMcuCuPartition&) = delete;
    GpuMcuCuPartition& operator=(const GpuMcuCuPartition&) = delete;

    GpuMcuCuPartition(GpuMcuCuPartition&& other) noexcept { move_from(other); }
    GpuMcuCuPartition& operator=(GpuMcuCuPartition&& other) noexcept {
        if (this != &other) {
            (void)shutdown();
            move_from(other);
        }
        return *this;
    }

    static Result<GpuMcuCuPartition> create(int device, uint32_t control_wgps = 1u);

    bool valid() const noexcept { return control_stream_ != nullptr; }
    int device() const noexcept { return device_; }

    hipStream_t control_stream() const noexcept { return control_stream_; }
    hipStream_t worker_stream() const noexcept { return worker_stream_; }

    const std::vector<uint32_t>& hip_control_mask() const noexcept {
        return hip_control_mask_;
    }
    const std::vector<uint32_t>& hip_worker_mask() const noexcept {
        return hip_worker_mask_;
    }
    AqlCuMaskView hip_control_mask_view() const noexcept {
        return AqlCuMaskView{hip_control_mask_.data(), hip_mask_bit_count_};
    }
    AqlCuMaskView hip_worker_mask_view() const noexcept {
        return AqlCuMaskView{hip_worker_mask_.data(), hip_mask_bit_count_};
    }

    const std::vector<uint32_t>& hsa_control_cu_mask() const noexcept {
        return hsa_control_cu_mask_;
    }
    const std::vector<uint32_t>& hsa_worker_cu_mask() const noexcept {
        return hsa_worker_cu_mask_;
    }
    AqlCuMaskView hsa_control_cu_mask_view() const noexcept {
        return AqlCuMaskView{hsa_control_cu_mask_.data(), hsa_cu_bit_count_};
    }
    AqlCuMaskView hsa_worker_cu_mask_view() const noexcept {
        return AqlCuMaskView{hsa_worker_cu_mask_.data(), hsa_cu_bit_count_};
    }

    uint32_t hip_mask_unit_count() const noexcept { return hip_mask_unit_count_; }
    uint32_t hip_mask_bit_count() const noexcept { return hip_mask_bit_count_; }
    uint32_t hsa_cu_count() const noexcept { return hsa_cu_count_; }
    uint32_t hsa_cu_bit_count() const noexcept { return hsa_cu_bit_count_; }

    uint32_t control_wgp_count() const noexcept { return control_wgp_count_; }
    uint32_t worker_wgp_count() const noexcept { return worker_wgp_count_; }
    uint32_t control_cu_count() const noexcept { return control_cu_count_; }
    uint32_t worker_cu_count() const noexcept { return worker_cu_count_; }

    Status shutdown() noexcept;

private:
    void move_from(GpuMcuCuPartition& other) noexcept;

    int device_ = -1;
    uint32_t hip_mask_unit_count_ = 0;
    uint32_t hip_mask_bit_count_ = 0;
    uint32_t hsa_cu_count_ = 0;
    uint32_t hsa_cu_bit_count_ = 0;
    uint32_t control_wgp_count_ = 0;
    uint32_t worker_wgp_count_ = 0;
    uint32_t control_cu_count_ = 0;
    uint32_t worker_cu_count_ = 0;
    std::vector<uint32_t> hip_control_mask_;
    std::vector<uint32_t> hip_worker_mask_;
    std::vector<uint32_t> hsa_control_cu_mask_;
    std::vector<uint32_t> hsa_worker_cu_mask_;
    hipStream_t control_stream_ = nullptr;
    hipStream_t worker_stream_ = nullptr;
};

}  // namespace ps::runtime::gpu_mcu
