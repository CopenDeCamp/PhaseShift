#pragma once

#include <phaseshift/core/memory/types.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gpu_mcu_rmsnorm {

inline constexpr float kEps = 1e-6f;

struct Shape {
    uint32_t features;
    uint32_t group;
};

inline constexpr Shape kShapes[] = {{256u, 256u}, {5120u, 5120u}};

inline uint32_t h32(uint64_t x) {
    uint32_t h = static_cast<uint32_t>(x * 2654435761u + 12345u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

inline float f32_from_bf16(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

inline void build_input(uint32_t rows, uint32_t features, uint32_t stride,
                        std::vector<uint16_t>& out) {
    out.assign(static_cast<size_t>(rows) * stride, 0u);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t c = 0; c < features; ++c) {
            const double v =
                (static_cast<double>(
                     h32(static_cast<uint64_t>(m) * 7919u + c * 31u + 7u) % 2001u) -
                 1000.0) /
                1000.0;
            out[static_cast<size_t>(m) * stride + c] =
                ps::f32_to_bf16_rne(static_cast<float>(v));
        }
    }
}

inline void build_weight(uint32_t features, std::vector<uint16_t>& out) {
    out.assign(features, 0u);
    for (uint32_t c = 0; c < features; ++c) {
        const double v =
            (static_cast<double>(h32(static_cast<uint64_t>(c) * 104729u + 11u) % 2001u) -
             1000.0) /
            1000.0;
        out[c] = ps::f32_to_bf16_rne(static_cast<float>(v));
    }
}

inline void cpu_reference(const std::vector<uint16_t>& in,
                          const std::vector<uint16_t>& weight, uint32_t rows,
                          uint32_t features, uint32_t group, uint32_t stride,
                          std::vector<float>& out) {
    out.assign(static_cast<size_t>(rows) * stride, 0.0f);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t g0 = 0; g0 < features; g0 += group) {
            double sum = 0.0;
            for (uint32_t h = 0; h < group; ++h) {
                const double x = static_cast<double>(
                    f32_from_bf16(in[static_cast<size_t>(m) * stride + g0 + h]));
                sum += x * x;
            }
            const double inv = 1.0 / std::sqrt(sum / static_cast<double>(group) + kEps);
            for (uint32_t h = 0; h < group; ++h) {
                const double x = static_cast<double>(
                    f32_from_bf16(in[static_cast<size_t>(m) * stride + g0 + h]));
                const double w = static_cast<double>(f32_from_bf16(weight[g0 + h]));
                out[static_cast<size_t>(m) * stride + g0 + h] =
                    static_cast<float>(x * inv * (1.0 + w));
            }
        }
    }
}

inline bool compare_to_reference(const std::vector<float>& got,
                                 const std::vector<float>& ref,
                                 uint32_t features, uint32_t stride,
                                 const char* label) {
    const double bf16_eps = 0.00390625;
    bool ok = true;
    size_t arg = 0;
    for (size_t i = 0; i < got.size() && ok; ++i) {
        const double g = got[i];
        const double r = ref[i];
        const size_t c = i % stride;
        if (std::isnan(static_cast<float>(g)) || std::isinf(static_cast<float>(g))) {
            ok = false;
            arg = i;
            break;
        }
        if (c >= features) continue;
        const double tol = 2.0 * bf16_eps * std::max(1.0, std::fabs(r));
        if (!(std::fabs(g - r) <= tol)) {
            ok = false;
            arg = i;
            break;
        }
    }
    if (!ok) {
        std::printf("  mismatch [%s] at %zu: got=%.9g ref=%.9g\n", label, arg,
                    got[arg], ref[arg]);
    }
    return ok;
}

inline bool bit_exact(const std::vector<uint16_t>& host,
                      const std::vector<uint16_t>& other, size_t* diffs,
                      size_t* first) {
    *diffs = 0;
    *first = 0;
    if (host.size() != other.size()) return false;
    for (size_t i = 0; i < host.size(); ++i) {
        if (host[i] != other[i]) {
            if (*diffs == 0) *first = i;
            ++*diffs;
        }
    }
    return *diffs == 0;
}

}  // namespace gpu_mcu_rmsnorm
