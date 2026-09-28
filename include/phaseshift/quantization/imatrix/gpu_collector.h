#pragma once
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <phaseshift/core/memory/arena.h>
#include <map>
#include <vector>

namespace ps::quantization::imatrix {

struct ImatrixAccumulator {
    uint64_t count = 0;
    std::vector<double> sum_sq;
};

struct DeviceImatrixDiagnostics {
    uint64_t nonfinite_count = 0;
    uint64_t first_bad_tag = UINT64_MAX;
};

class GpuImatrixCollector : public ImatrixCollector {
public:
    Status initialize(ps::gpu::GpuArena& arena,
                      const std::vector<uint32_t>& tags,
                      const std::vector<uint64_t>& k_vals);

    Status record(uint32_t tag, const void* data, uint32_t row_stride,
                  uint32_t features, uint32_t rows, ImatrixDType dtype,
                  hipStream_t stream) override;

    Status flush(hipStream_t stream);

    const std::map<uint32_t, ImatrixAccumulator>& accumulators() const { return host_; }
    const std::map<uint32_t, DeviceImatrixDiagnostics>& diagnostics() const { return diagnostics_; }

private:
    struct DeviceAcc {
        float* sum_sq = nullptr;
        DeviceImatrixDiagnostics* diag = nullptr;
        uint32_t k = 0;
    };

    std::map<uint32_t, DeviceAcc> device_;
    std::map<uint32_t, ImatrixAccumulator> host_;
    std::map<uint32_t, DeviceImatrixDiagnostics> diagnostics_;
};

}
