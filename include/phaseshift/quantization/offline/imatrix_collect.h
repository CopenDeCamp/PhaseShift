#pragma once
#include <phaseshift/core/status.h>
#include <cstdint>
#include <string>

namespace ps::quantization::imatrix {

struct ImatrixOptions {
    std::string input_dir;
    std::string model_dir;
    std::string tokens_path;
    std::string output_path;
    std::size_t window = 512;
    std::size_t stride = 512;
    std::size_t max_calibration_tokens = 0;
    std::size_t flush_windows = 16;
    double arena_gib = 24.0;
    int device = 0;
};

struct ImatrixSummary {
    uint64_t positions = 0;
    uint64_t entries = 0;
    uint64_t windows = 0;
};

Result<ImatrixSummary> collect_imatrix(const ImatrixOptions& options);

}
