#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/io/safetensors_reader.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::quantization::fpx {

struct GpuChunkMetrics {
    double sum_abs_error = 0.0;
    double sum_error_sq = 0.0;
    double sum_reference_sq = 0.0;
    double sum_reconstructed_sq = 0.0;
    double dot = 0.0;
    double max_abs_error = 0.0;
    uint64_t logical_elements = 0;
    uint64_t source_nonfinite = 0;
    uint64_t reconstructed_nonfinite = 0;
    uint64_t zero_code_count = 0;
    uint64_t terminal_code_count = 0;
    uint64_t negative_128_count = 0;
    uint64_t scale_zero_count = 0;
    uint64_t max_magnitude_code_count = 0;
    uint32_t scale_min_nonzero = 0;
    uint32_t scale_max = 0;
    uint32_t flags = 0;
};

constexpr uint32_t kGpuValidationFlagPaddingNonzero = 1u;
constexpr uint32_t kGpuValidationFlagIllegalScale = 2u;
constexpr uint32_t kGpuValidationFlagNeg128 = 4u;

struct GpuValidationWorkspace {
    void* h_pinned_source = nullptr;
    void* h_pinned_codes = nullptr;
    void* h_pinned_scales = nullptr;
    void* d_source = nullptr;
    void* d_codes = nullptr;
    void* d_scales = nullptr;
    GpuChunkMetrics* d_partial = nullptr;
    GpuChunkMetrics* d_final = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    std::size_t chunk_elements = 0;
    std::size_t num_blocks = 256;
    std::size_t threads = 256;
};

Status gpu_validation_workspace_create(GpuValidationWorkspace& ws, std::size_t chunk_elements);
Status gpu_validation_workspace_destroy(GpuValidationWorkspace& ws);

Status gpu_validate_fp4_chunk(
    const void* source,
    const uint8_t* codes,
    const uint8_t* scales,
    ps::io::SType source_dtype,
    uint64_t logical_k,
    uint64_t padded_k,
    GpuValidationWorkspace& ws,
    GpuChunkMetrics* metrics_out);

Status gpu_validate_fp8_chunk(
    const void* source,
    const uint8_t* codes,
    const uint8_t* scales,
    ps::io::SType source_dtype,
    uint64_t logical_k,
    uint64_t padded_k,
    GpuValidationWorkspace& ws,
    GpuChunkMetrics* metrics_out);

Status gpu_validate_bf16_chunk(
    const void* source,
    const void* bundle_bf16,
    ps::io::SType source_dtype,
    GpuValidationWorkspace& ws,
    GpuChunkMetrics* metrics_out);

}
