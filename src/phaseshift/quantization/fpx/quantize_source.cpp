#include <phaseshift/quantization/fpx/quantize_source.h>
#include <openssl/sha.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ps::quantization::fpx {

uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t bias = 0x00007FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + bias) >> 16);
}

float bf16_to_fp32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (static_cast<uint32_t>(h) & 0x7C00u) >> 10;
    const uint32_t man = static_cast<uint32_t>(h) & 0x03FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) {
            u = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            do { ++e; m <<= 1; } while ((m & 0x0400u) == 0);
            u = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void read_row_f32(const SourceView& sv, uint64_t row, uint64_t logical_k, uint64_t padded_k, float* out) {
    const uint64_t k = logical_k;
    const uint8_t* base = sv.data + row * k * sv.byte_stride;
    for (uint64_t j = 0; j < padded_k; ++j) out[j] = 0.0f;
    switch (sv.dtype) {
        case ps::io::SType::BF16: {
            const auto* p = reinterpret_cast<const uint16_t*>(base);
            for (uint64_t j = 0; j < k; ++j) out[j] = bf16_to_fp32(p[j]);
            break;
        }
        case ps::io::SType::F32: {
            const auto* p = reinterpret_cast<const float*>(base);
            for (uint64_t j = 0; j < k; ++j) out[j] = p[j];
            break;
        }
        case ps::io::SType::F16: {
            const auto* p = reinterpret_cast<const uint16_t*>(base);
            for (uint64_t j = 0; j < k; ++j) out[j] = fp16_to_fp32(p[j]);
            break;
        }
        default: {
            for (uint64_t j = 0; j < k; ++j) out[j] = 0.0f;
            break;
        }
    }
}

Status write_bf16_row(const SourceView& sv, uint64_t row, uint64_t k, FILE* out, uint64_t offset, Crc32Accumulator* crc) {
    const uint8_t* base = sv.data + row * k * sv.byte_stride;
    if (sv.dtype == ps::io::SType::BF16) {
        std::fseek(out, static_cast<long>(offset), SEEK_SET);
        const std::size_t n = k * 2;
        if (std::fwrite(base, 1, n, out) != n) {
            return Status::invalid_argument("write failed", __FILE__, __LINE__);
        }
        crc->update(base, n);
        return Status::make_ok();
    }
    std::vector<uint16_t> b16(k);
    bf16_row_bytes(sv, row, k, b16, crc);
    std::fseek(out, static_cast<long>(offset), SEEK_SET);
    const std::size_t n = k * 2;
    if (std::fwrite(b16.data(), 1, n, out) != n) {
        return Status::invalid_argument("write failed", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

void bf16_row_bytes(const SourceView& sv, uint64_t row, uint64_t k, std::vector<uint16_t>& out, Crc32Accumulator* crc) {
    const uint8_t* base = sv.data + row * k * sv.byte_stride;
    if (sv.dtype == ps::io::SType::BF16) {
        std::memcpy(out.data(), base, k * 2);
    } else if (sv.dtype == ps::io::SType::F32) {
        const auto* p = reinterpret_cast<const float*>(base);
        for (uint64_t j = 0; j < k; ++j) out[j] = fp32_to_bf16(p[j]);
    } else if (sv.dtype == ps::io::SType::F16) {
        const auto* p = reinterpret_cast<const uint16_t*>(base);
        for (uint64_t j = 0; j < k; ++j) out[j] = fp32_to_bf16(fp16_to_fp32(p[j]));
    } else {
        const auto* p = reinterpret_cast<const uint16_t*>(base);
        for (uint64_t j = 0; j < k; ++j) out[j] = p[j];
    }
    crc->update(out.data(), k * 2);
}

std::string dominant_dtype(std::map<std::string, int>& counts) {
    std::string best = "bf16";
    int best_count = -1;
    for (const auto& kv : counts) {
        if (kv.second > best_count) {
            best = kv.first;
            best_count = kv.second;
        }
    }
    return best;
}

std::vector<float> build_quant_weights(const double* sum_sq, uint64_t count, uint64_t k, uint64_t padded_k) {
    std::vector<float> qw(padded_k, 0.0f);
    for (uint64_t j = 0; j < k; ++j) {
        const double v = sum_sq[j] / static_cast<double>(count);
        if (!std::isfinite(v) || v < 0.0) {
            return {};
        }
        qw[j] = static_cast<float>(v);
    }
    return qw;
}

std::string sha256_hex(const void* data, std::size_t len) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char*>(data), len, hash);
    char buf[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(buf + i * 2, 3, "%02x", hash[i]);
    }
    return std::string(buf, 64);
}

}
