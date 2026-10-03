#pragma once

#include <phaseshift/models/qwen35/kernels/dflash2/radix_topn.h>
#include <phaseshift/models/qwen35/kernels/optimized/psq8_candidate_rerank.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace dflash2_int2_test {

struct Rng {
    uint32_t state = 0x1234567u;

    uint32_t next_u32() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }

    float unit() {
        return static_cast<float>(next_u32()) / static_cast<float>(1u << 24);
    }

    float sym() {
        return unit() * 2.0f - 1.0f;
    }

    uint32_t below(uint32_t n) {
        return next_u32() % (n == 0u ? 1u : n);
    }
};

inline std::size_t psq8_code_index(uint32_t r, uint32_t k, uint32_t k_padded) {
    const uint32_t ib = k >> 5u;
    const uint32_t rk = k & 31u;
    const uint32_t half = rk >> 4u;
    const uint32_t kg = (rk >> 3u) & 1u;
    const uint32_t j = rk & 7u;
    return static_cast<std::size_t>(r >> 4u) * (static_cast<std::size_t>(k_padded) * 16u) +
           static_cast<std::size_t>(r & 15u) * 16u + static_cast<std::size_t>(ib) * 512u +
           half * 256u + kg * 8u + j;
}

inline std::size_t int2_code_index(uint32_t r, uint32_t k, uint32_t k_padded) {
    const uint32_t ib = k >> 5u;
    const uint32_t rk = k & 31u;
    const uint32_t half = rk >> 4u;
    const uint32_t kg = (rk >> 3u) & 1u;
    const uint32_t j = rk & 7u;
    return static_cast<std::size_t>(r >> 4u) * (static_cast<std::size_t>(k_padded) * 4u) +
           static_cast<std::size_t>(r & 15u) * 4u + static_cast<std::size_t>(ib) * 128u +
           half * 64u + kg * 2u + (j >> 2u);
}

inline uint32_t int2_shift(uint32_t k) {
    return (k & 3u) * 2u;
}

inline std::size_t psq8_scale_index(uint32_t r, uint32_t k, uint32_t scale_stride) {
    const uint32_t ib = k >> 5u;
    return static_cast<std::size_t>(r >> 4u) * (scale_stride / 2u) +
           static_cast<std::size_t>(r & 15u) + static_cast<std::size_t>(ib) * 16u;
}

inline std::size_t act_code_index(uint32_t r, uint32_t k, uint32_t code_stride) {
    const uint32_t ib = k >> 5u;
    const uint32_t rk = k & 31u;
    const uint32_t half = rk >> 4u;
    const uint32_t kg = (rk >> 3u) & 1u;
    const uint32_t j = rk & 7u;
    return static_cast<std::size_t>(r >> 4u) * (static_cast<std::size_t>(code_stride) * 16u) +
           static_cast<std::size_t>(r & 15u) * 16u + static_cast<std::size_t>(ib) * 512u +
           half * 256u + kg * 8u + j;
}

inline uint16_t bf16_bits(float v) {
    const uint32_t u = *reinterpret_cast<const uint32_t*>(&v);
    return static_cast<uint16_t>((u + 0x8000u + ((u >> 16) & 1u)) >> 16);
}

struct TopnOracle {
    std::vector<int32_t> ids;
    std::vector<float> values;
};

inline TopnOracle cpu_topn(const std::vector<float>& row, uint32_t pool) {
    std::vector<int32_t> order(row.size());
    for (std::size_t i = 0u; i < row.size(); ++i)
        order[i] = static_cast<int32_t>(i);
    std::stable_sort(order.begin(), order.end(), [&row](int32_t a, int32_t b) {
        if (row[static_cast<std::size_t>(a)] != row[static_cast<std::size_t>(b)])
            return row[static_cast<std::size_t>(a)] > row[static_cast<std::size_t>(b)];
        return a < b;
    });
    TopnOracle out;
    const uint32_t n = static_cast<uint32_t>(std::min<std::size_t>(pool, order.size()));
    out.ids.resize(n);
    out.values.resize(n);
    for (uint32_t i = 0u; i < n; ++i) {
        out.ids[i] = order[i];
        out.values[i] = row[static_cast<std::size_t>(order[i])];
    }
    return out;
}

}  // namespace dflash2_int2_test
