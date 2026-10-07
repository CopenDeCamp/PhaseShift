#include <phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace ps::runtime::gpu_mcu {

namespace {

void set_bit(std::vector<uint32_t>& words, uint32_t index) {
    words[index / 32u] |= (1u << (index % 32u));
}

uint32_t count_bits(const std::vector<uint32_t>& words) {
    uint32_t count = 0;
    for (uint32_t word : words) {
        count += static_cast<uint32_t>(__builtin_popcount(word));
    }
    return count;
}

Status readback_stream_mask(hipStream_t stream, std::vector<uint32_t>& out) {
    if (hipExtStreamGetCUMask(stream, static_cast<uint32_t>(out.size()),
                              out.data()) != hipSuccess) {
        return Status::hip_error("hipExtStreamGetCUMask", "readback failed",
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status readback_exact(uint32_t bit_count,
                      const std::vector<uint32_t>& mask,
                      bool& exact) {
    hipStream_t stream = nullptr;
    if (hipExtStreamCreateWithCUMask(&stream, bit_count, mask.data()) != hipSuccess) {
        return Status::hip_error("hipExtStreamCreateWithCUMask", "create failed",
                                 __FILE__, __LINE__);
    }
    std::vector<uint32_t> actual(mask.size(), 0u);
    const Status s = readback_stream_mask(stream, actual);
    (void)hipStreamDestroy(stream);
    if (!s.ok()) return s;
    exact = (actual == mask);
    return Status::make_ok();
}

}  // namespace

Result<GpuMcuMaskSemantics> gpu_mcu_probe_mask_semantics(int device) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        return Status::hip_error("hipGetDeviceProperties", "query failed",
                                 __FILE__, __LINE__);
    }
    GpuMcuMaskSemantics out{};
    out.reported_multi_processor_count = static_cast<uint32_t>(prop.multiProcessorCount);
    if (out.reported_multi_processor_count == 0u) {
        return Status::invalid_state("device reports zero scheduling units",
                                     __FILE__, __LINE__);
    }

    auto cu_count = gpu_mcu_hsa_compute_unit_count(device);
    if (!cu_count.ok()) return cu_count.status();
    out.hsa_compute_unit_count = cu_count.value();

    const uint32_t units = out.reported_multi_processor_count;
    const uint32_t words = (units + 31u) / 32u;
    out.hip_mask_unit_count = units;

    std::vector<uint32_t> full(words, 0u);
    for (uint32_t i = 0; i < units; ++i) set_bit(full, i);
    if (const Status s = readback_exact(words, full, out.full_mask_readback_exact);
        !s.ok()) {
        return s;
    }

    std::vector<uint32_t> single(words, 0u);
    set_bit(single, 0);
    if (const Status s = readback_exact(words, single, out.single_bit_readback_exact);
        !s.ok()) {
        return s;
    }

    std::vector<uint32_t> unpaired(words, 0u);
    set_bit(unpaired, 0);
    set_bit(unpaired, units - 1u);
    if (const Status s =
            readback_exact(words, unpaired, out.unpaired_bits_readback_exact);
        !s.ok()) {
        return s;
    }
    return out;
}

void GpuMcuCuPartition::move_from(GpuMcuCuPartition& other) noexcept {
    device_ = other.device_;
    hip_mask_unit_count_ = other.hip_mask_unit_count_;
    hip_mask_bit_count_ = other.hip_mask_bit_count_;
    hsa_cu_count_ = other.hsa_cu_count_;
    hsa_cu_bit_count_ = other.hsa_cu_bit_count_;
    control_wgp_count_ = other.control_wgp_count_;
    worker_wgp_count_ = other.worker_wgp_count_;
    control_cu_count_ = other.control_cu_count_;
    worker_cu_count_ = other.worker_cu_count_;
    hip_control_mask_ = std::move(other.hip_control_mask_);
    hip_worker_mask_ = std::move(other.hip_worker_mask_);
    hsa_control_cu_mask_ = std::move(other.hsa_control_cu_mask_);
    hsa_worker_cu_mask_ = std::move(other.hsa_worker_cu_mask_);
    control_stream_ = other.control_stream_;
    worker_stream_ = other.worker_stream_;
    other.device_ = -1;
    other.control_stream_ = nullptr;
    other.worker_stream_ = nullptr;
}

Status GpuMcuCuPartition::shutdown() noexcept {
    Status first = Status::make_ok();
    auto note = [&](Status s) {
        if (!s.ok() && first.ok()) first = s;
    };
    if (control_stream_ != nullptr) {
        if (hipStreamDestroy(control_stream_) != hipSuccess) {
            note(Status::hip_error("hipStreamDestroy(control)", "destroy failed",
                                   __FILE__, __LINE__));
        }
        control_stream_ = nullptr;
    }
    if (worker_stream_ != nullptr) {
        if (hipStreamDestroy(worker_stream_) != hipSuccess) {
            note(Status::hip_error("hipStreamDestroy(worker)", "destroy failed",
                                   __FILE__, __LINE__));
        }
        worker_stream_ = nullptr;
    }
    hip_control_mask_.clear();
    hip_worker_mask_.clear();
    hsa_control_cu_mask_.clear();
    hsa_worker_cu_mask_.clear();
    return first;
}

Result<GpuMcuCuPartition> GpuMcuCuPartition::create(int device, uint32_t control_wgps) {
    if (device < 0) {
        return Status::invalid_argument("device must be >= 0", __FILE__, __LINE__);
    }
    if (control_wgps == 0u) {
        return Status::invalid_argument("control WGP count must be nonzero",
                                        __FILE__, __LINE__);
    }

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        return Status::hip_error("hipGetDeviceProperties", "query failed",
                                 __FILE__, __LINE__);
    }
    const uint32_t units = static_cast<uint32_t>(prop.multiProcessorCount);
    if (units <= control_wgps) {
        return Status::invalid_argument("control WGP must leave worker WGPs",
                                        __FILE__, __LINE__);
    }

    auto cu_count_res = gpu_mcu_hsa_compute_unit_count(device);
    if (!cu_count_res.ok()) return cu_count_res.status();
    const uint32_t cu_count = cu_count_res.value();
    if (cu_count != units * kGpuMcuCusPerWgp) {
        return Status::invalid_state(
            "HSA CU count does not equal 2x the HIP scheduling units",
            __FILE__, __LINE__);
    }

    const uint32_t hip_words = (units + 31u) / 32u;
    std::vector<uint32_t> hip_control(hip_words, 0u);
    std::vector<uint32_t> hip_worker(hip_words, 0u);
    for (uint32_t unit = 0; unit < units; ++unit) {
        set_bit(unit < control_wgps ? hip_control : hip_worker, unit);
    }

    const uint32_t cu_words = (cu_count + 31u) / 32u;
    std::vector<uint32_t> hsa_control(cu_words, 0u);
    std::vector<uint32_t> hsa_worker(cu_words, 0u);
    for (uint32_t cu = 0; cu < cu_count; ++cu) {
        set_bit(cu < control_wgps * kGpuMcuCusPerWgp ? hsa_control : hsa_worker, cu);
    }

    if (count_bits(hip_control) != control_wgps ||
        count_bits(hip_worker) != units - control_wgps) {
        return Status::invalid_state("invalid HIP mask construction", __FILE__, __LINE__);
    }
    if (count_bits(hsa_control) != control_wgps * kGpuMcuCusPerWgp ||
        count_bits(hsa_worker) != cu_count - control_wgps * kGpuMcuCusPerWgp) {
        return Status::invalid_state("invalid HSA CU mask construction", __FILE__,
                                     __LINE__);
    }
    for (uint32_t i = 0; i < hip_words; ++i) {
        if ((hip_control[i] & hip_worker[i]) != 0u) {
            return Status::invalid_state("HIP control and worker masks overlap",
                                         __FILE__, __LINE__);
        }
    }
    for (uint32_t i = 0; i < cu_words; ++i) {
        if ((hsa_control[i] & hsa_worker[i]) != 0u) {
            return Status::invalid_state("HSA control and worker CU masks overlap",
                                         __FILE__, __LINE__);
        }
    }
    if (!aql_cu_mask_is_wgp_paired(hsa_control.data(), cu_count) ||
        !aql_cu_mask_is_wgp_paired(hsa_worker.data(), cu_count)) {
        return Status::invalid_state("HSA CU mask violates WGP pairwise enable",
                                     __FILE__, __LINE__);
    }

    GpuMcuCuPartition out;
    out.device_ = device;
    out.hip_mask_unit_count_ = units;
    out.hip_mask_bit_count_ = hip_words * 32u;
    out.hsa_cu_count_ = cu_count;
    out.hsa_cu_bit_count_ = cu_words * 32u;
    out.control_wgp_count_ = control_wgps;
    out.worker_wgp_count_ = units - control_wgps;
    out.control_cu_count_ = control_wgps * kGpuMcuCusPerWgp;
    out.worker_cu_count_ = cu_count - control_wgps * kGpuMcuCusPerWgp;
    out.hip_control_mask_ = std::move(hip_control);
    out.hip_worker_mask_ = std::move(hip_worker);
    out.hsa_control_cu_mask_ = std::move(hsa_control);
    out.hsa_worker_cu_mask_ = std::move(hsa_worker);

    auto fail = [&](Status s) -> Result<GpuMcuCuPartition> {
        (void)out.shutdown();
        return s;
    };

    if (hipExtStreamCreateWithCUMask(&out.control_stream_, out.hip_mask_bit_count_,
                                     out.hip_control_mask_.data()) != hipSuccess) {
        return fail(Status::hip_error("hipExtStreamCreateWithCUMask(control)",
                                      "create failed", __FILE__, __LINE__));
    }
    if (hipExtStreamCreateWithCUMask(&out.worker_stream_, out.hip_mask_bit_count_,
                                     out.hip_worker_mask_.data()) != hipSuccess) {
        return fail(Status::hip_error("hipExtStreamCreateWithCUMask(worker)",
                                      "create failed", __FILE__, __LINE__));
    }

    std::vector<uint32_t> actual(hip_words, 0u);
    if (const Status s = readback_stream_mask(out.control_stream_, actual); !s.ok()) {
        return fail(s);
    }
    if (actual != out.hip_control_mask_) {
        return fail(Status::invalid_state("control stream mask readback mismatch",
                                          __FILE__, __LINE__));
    }
    std::fill(actual.begin(), actual.end(), 0u);
    if (const Status s = readback_stream_mask(out.worker_stream_, actual); !s.ok()) {
        return fail(s);
    }
    if (actual != out.hip_worker_mask_) {
        return fail(Status::invalid_state("worker stream mask readback mismatch",
                                          __FILE__, __LINE__));
    }
    return out;
}

}  // namespace ps::runtime::gpu_mcu
