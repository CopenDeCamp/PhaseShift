#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <vector>

namespace ps::runtime {

constexpr uint32_t kParallelMaxWorldSize = 4;

struct ParallelConfig {
    uint32_t pp_size = 1;
    uint32_t tp_size = 1;
    std::vector<int> devices;

    uint32_t world_size() const noexcept { return pp_size * tp_size; }

    uint32_t global_rank(uint32_t pp_rank, uint32_t tp_rank) const noexcept {
        return pp_rank * tp_size + tp_rank;
    }

    uint32_t pp_rank_of(uint32_t rank) const noexcept { return rank / tp_size; }

    uint32_t tp_rank_of(uint32_t rank) const noexcept { return rank % tp_size; }

    std::vector<int> tensor_group_devices(uint32_t pp_rank) const {
        std::vector<int> out;
        out.reserve(tp_size);
        for (uint32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank)
            out.push_back(devices[pp_rank * tp_size + tp_rank]);
        return out;
    }

    std::vector<int> pipeline_group_devices(uint32_t tp_rank) const {
        std::vector<int> out;
        out.reserve(pp_size);
        for (uint32_t pp_rank = 0; pp_rank < pp_size; ++pp_rank)
            out.push_back(devices[pp_rank * tp_size + tp_rank]);
        return out;
    }

    bool same_tensor_group(uint32_t a, uint32_t b) const noexcept {
        return a < world_size() && b < world_size() && pp_rank_of(a) == pp_rank_of(b);
    }

    bool same_pipeline_group(uint32_t a, uint32_t b) const noexcept {
        return a < world_size() && b < world_size() && tp_rank_of(a) == tp_rank_of(b);
    }

    Status validate() const {
        if (pp_size < 1)
            return Status::invalid_argument("pp_size must be >= 1", __FILE__, __LINE__);
        if (tp_size < 1)
            return Status::invalid_argument("tp_size must be >= 1", __FILE__, __LINE__);
        if (pp_size > kParallelMaxWorldSize || tp_size > kParallelMaxWorldSize)
            return Status::unsupported("pp_size / tp_size exceed the supported range",
                                       __FILE__, __LINE__);
        const uint32_t world = world_size();
        if (world != 1 && world != 2 && world != 4)
            return Status::unsupported("only world_size 1, 2 and 4 are supported",
                                       __FILE__, __LINE__);
        if (devices.size() != static_cast<std::size_t>(world))
            return Status::invalid_argument("devices.size() must equal pp_size * tp_size",
                                            __FILE__, __LINE__);
        for (std::size_t i = 0; i < devices.size(); ++i) {
            if (devices[i] < 0)
                return Status::invalid_argument("device index must be >= 0", __FILE__, __LINE__);
            for (std::size_t j = i + 1; j < devices.size(); ++j) {
                if (devices[i] == devices[j])
                    return Status::invalid_argument("duplicate device index", __FILE__,
                                                    __LINE__);
            }
        }
        return Status::make_ok();
    }
};

}
