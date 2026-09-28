#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/io/safetensors_reader.h>
#include <phaseshift/quantization/fpx/crc32.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps::quantization::fpx {

uint16_t fp32_to_bf16(float f);
float bf16_to_fp32(uint16_t h);
float fp16_to_fp32(uint16_t h);

struct SourceView {
    const uint8_t* data = nullptr;
    std::size_t byte_stride = 0;
    ps::io::SType dtype = ps::io::SType::BF16;
    std::vector<int64_t> shape;
};

void read_row_f32(const SourceView& sv, uint64_t row, uint64_t logical_k, uint64_t padded_k, float* out);

Status write_bf16_row(const SourceView& sv, uint64_t row, uint64_t k, FILE* out, uint64_t offset, Crc32Accumulator* crc);

// Converts one source row to BF16 bytes in `out` (size k) and updates `crc`.
void bf16_row_bytes(const SourceView& sv, uint64_t row, uint64_t k, std::vector<uint16_t>& out, Crc32Accumulator* crc);

std::string dominant_dtype(std::map<std::string, int>& counts);

std::vector<float> build_quant_weights(const double* sum_sq, uint64_t count, uint64_t k, uint64_t padded_k);

std::string sha256_hex(const void* data, std::size_t len);

}
