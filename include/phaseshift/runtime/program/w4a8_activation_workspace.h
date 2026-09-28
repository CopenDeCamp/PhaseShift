#pragma once
#include <cstdint>

namespace ps::runtime {

struct W4a8ActivationWorkspaceLayout {
    uint64_t codes_offset = 0;
    uint64_t scales_offset = 0;
    uint64_t total_bytes = 0;
    uint32_t k_padded = 0;
    uint32_t max_rows = 0;

    static constexpr uint32_t block_size() noexcept {
        return 32u;
    }

    static constexpr uint32_t scale_element_bytes() noexcept {
        return 4u;
    }

    static uint32_t k_padded_for(uint32_t k) noexcept {
        return (k + 31u) & ~31u;
    }

    static W4a8ActivationWorkspaceLayout make(uint32_t k, uint32_t rows) noexcept {
        W4a8ActivationWorkspaceLayout l;
        l.k_padded = k_padded_for(k);
        l.max_rows = rows;
        l.codes_offset = 0;
        const uint64_t codes_bytes =
            static_cast<uint64_t>(l.max_rows) * l.k_padded;
        l.scales_offset = (codes_bytes + 63u) & ~63ull;
        const uint64_t scales_bytes =
            static_cast<uint64_t>(l.max_rows) * (l.k_padded / block_size()) * scale_element_bytes();
        l.total_bytes = l.scales_offset + scales_bytes;
        return l;
    }

    uint64_t codes_bytes() const noexcept {
        return static_cast<uint64_t>(max_rows) * k_padded;
    }

    uint64_t scales_bytes() const noexcept {
        return total_bytes - scales_offset;
    }
};

}
