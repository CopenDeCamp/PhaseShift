#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/correctness/standalone/gemm_bf16_correctness.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace ps::test {

inline ps::bf16_t to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return ps::bf16_t{static_cast<std::uint16_t>((bits + 0x8000u) >> 16)};
}

inline float to_f32(ps::bf16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

struct HostMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<ps::bf16_t> data;

    ps::bf16_t& at(std::uint32_t r, std::uint32_t c) {
        return data[static_cast<std::size_t>(r) * cols + c];
    }

    ps::bf16_t at(std::uint32_t r, std::uint32_t c) const {
        return data[static_cast<std::size_t>(r) * cols + c];
    }
};

inline HostMatrix random_matrix(std::uint32_t rows, std::uint32_t cols, std::uint32_t seed) {
    HostMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.data.resize(static_cast<std::size_t>(rows) * cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : m.data) v = to_bf16(dist(rng));
    return m;
}

inline HostMatrix zeros_matrix(std::uint32_t rows, std::uint32_t cols) {
    HostMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.data.assign(static_cast<std::size_t>(rows) * cols, ps::bf16_t{0});
    return m;
}

inline HostMatrix reference_gemm(const HostMatrix& x, const HostMatrix& w) {
    HostMatrix y = zeros_matrix(x.rows, w.rows);
    for (std::uint32_t r = 0; r < x.rows; ++r) {
        for (std::uint32_t n = 0; n < w.rows; ++n) {
            float acc = 0.0f;
            for (std::uint32_t k = 0; k < x.cols; ++k)
                acc += to_f32(x.at(r, k)) * to_f32(w.at(n, k));
            y.at(r, n) = to_bf16(acc);
        }
    }
    return y;
}

struct GpuMatrix {
    ps::bf16_t* ptr = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;

    std::size_t bytes() const {
        return static_cast<std::size_t>(rows) * cols * sizeof(ps::bf16_t);
    }
};

inline bool upload_matrix(const HostMatrix& host, GpuMatrix& out) {
    const std::size_t count = host.data.size();
    void* raw = nullptr;
    if (hipMalloc(&raw, count * sizeof(ps::bf16_t)) != hipSuccess) return false;
    out.ptr = static_cast<ps::bf16_t*>(raw);
    out.rows = host.rows;
    out.cols = host.cols;
    return hipMemcpy(out.ptr, host.data.data(), count * sizeof(ps::bf16_t),
                     hipMemcpyHostToDevice) == hipSuccess;
}

inline bool download_matrix(const GpuMatrix& gpu, HostMatrix& out) {
    out.rows = gpu.rows;
    out.cols = gpu.cols;
    out.data.resize(static_cast<std::size_t>(gpu.rows) * gpu.cols);
    return hipMemcpy(out.data.data(), gpu.ptr, gpu.bytes(),
                     hipMemcpyDeviceToHost) == hipSuccess;
}

inline bool free_matrix(GpuMatrix& m) {
    if (m.ptr == nullptr) return true;
    const hipError_t err = hipFree(m.ptr);
    m.ptr = nullptr;
    return err == hipSuccess;
}

struct MaxError {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    std::size_t mismatch = 0;
    std::size_t count = 0;
};

inline MaxError compare_matrices(const HostMatrix& expected, const HostMatrix& actual) {
    MaxError out;
    if (expected.rows != actual.rows || expected.cols != actual.cols) {
        out.mismatch = 1;
        out.count = 1;
        return out;
    }
    out.count = expected.data.size();
    for (std::size_t i = 0; i < expected.data.size(); ++i) {
        const float e = to_f32(expected.data[i]);
        const float a = to_f32(actual.data[i]);
        const float d = e > a ? e - a : a - e;
        if (d != 0.0f) ++out.mismatch;
        if (d > out.max_abs) out.max_abs = d;
        const float denom = e > 0.0f ? e : -e;
        if (denom > 0.0f) {
            const float rel = d / denom;
            if (rel > out.max_rel) out.max_rel = rel;
        }
    }
    return out;
}

}
