#pragma once
#include <cstdint>

namespace ps::qwen35::runtime {

enum class SamplingBatchMode : uint8_t {
    NoSampling = 0,
    Mixed = 1,
    AllGreedy = 2,
    HasStochastic = 3,
};

enum class SamplingImplementation : uint8_t {
    Correctness = 0,
    SingleBlock = 1,
    Partitioned = 2,
    StochasticSingleBlock = 3,
};

struct SamplingChoice {
    SamplingImplementation implementation = SamplingImplementation::Correctness;
    uint32_t partitions = 0;
};

struct SamplingSelectorInput {
    uint32_t vocab_size = 0;
    uint32_t outputs = 0;
    uint32_t sampled_outputs = 0;
    uint32_t stochastic_outputs = 0;
    uint32_t scratch_pair_capacity = 0;
};

struct SamplingRule {
    uint32_t vocab_size;
    SamplingBatchMode batch_mode;
    uint32_t min_outputs;
    uint32_t max_outputs;
    SamplingImplementation implementation;
    uint32_t partitions;
};

SamplingChoice select_sampling_implementation(const SamplingSelectorInput& in);

}  // namespace ps::qwen35::runtime
