#pragma once

#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/device_program.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kMaxSamplingAttempts = 4096u;
inline constexpr uint32_t kSamplingNoneMode = 0u;
inline constexpr uint32_t kSamplingGreedyMode = 1u;
inline constexpr uint32_t kSamplingStochasticMode = 2u;

__host__ __device__ __forceinline__ uint64_t sampling_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__host__ __device__ __forceinline__ uint64_t sampling_rng_bits(
    uint64_t seed,
    uint64_t sample_index,
    uint64_t attempt,
    uint64_t token_id) {
    uint64_t h = sampling_splitmix64(seed + 0x243F6A8885A308D3ull);
    h = sampling_splitmix64(h ^ (sample_index * 0x9E3779B97F4A7C15ull));
    h = sampling_splitmix64(h ^ (attempt * 0xC2B2AE3D27D4EB4Full));
    h = sampling_splitmix64(h ^ (token_id + 0x165667B19E3779F9ull));
    return h;
}

__host__ __device__ __forceinline__ float sampling_uniform(
    uint64_t seed,
    uint64_t sample_index,
    uint64_t attempt,
    uint64_t token_id) {
    const uint32_t bits =
        static_cast<uint32_t>(sampling_rng_bits(seed, sample_index, attempt, token_id));
    return (static_cast<float>(bits) + 0.5f) * (1.0f / 4294967296.0f);
}

__host__ __device__ __forceinline__ float sampling_gumbel(float u) {
    return -::logf(-::logf(u));
}

struct SamplingRowParams {
    uint32_t mode = kSamplingNoneMode;
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0u;
    uint64_t seed = 0u;
    uint64_t sample_index = 0u;
};

struct SamplingConstraint {
    const uint32_t* mask = nullptr;
    bool active = false;
};

__device__ __forceinline__ bool sampling_token_allowed(
    const SamplingConstraint& constraint, uint32_t token_id) {
    if (!constraint.active || constraint.mask == nullptr) return true;
    return ((constraint.mask[token_id >> 5u] >> (token_id & 31u)) & 1u) != 0u;
}

__host__ __device__ __forceinline__ SamplingRowParams sampling_row_params(
    const ::ps::runtime::DeviceSamplingParams& p) {
    SamplingRowParams r;
    r.mode = p.mode;
    r.temperature = p.temperature;
    r.top_p = p.top_p;
    r.top_k = p.top_k;
    r.seed = p.seed;
    r.sample_index = p.sample_index;
    return r;
}

__device__ __forceinline__ void sampling_report_error(uint32_t* error_word) {
    if (error_word == nullptr) return;
    atomicCAS(error_word, 0u,
              static_cast<uint32_t>(::ps::runtime::ProgramStatus::DISPATCH_FAILED));
}

namespace sampling_detail {

constexpr uint32_t kNoToken = 0xFFFFFFFFu;
constexpr float kNegInf = -3.4e38f;

__device__ __forceinline__ bool argmax_better(
    float cv, uint32_t ct, float bv, uint32_t bt) {
    return cv > bv || (cv == bv && ct < bt);
}

__device__ __forceinline__ void block_argmax(
    float& value, uint32_t& token, float* sv, uint32_t* st) {
    const uint32_t n = blockDim.x;
    sv[threadIdx.x] = value;
    st[threadIdx.x] = token;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off];
            const uint32_t ot = st[threadIdx.x + off];
            if (argmax_better(ov, ot, sv[threadIdx.x], st[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                st[threadIdx.x] = ot;
            }
        }
        __syncthreads();
    }
    value = sv[0];
    token = st[0];
}

__device__ __forceinline__ float block_max(float value, float* sv) {
    const uint32_t n = blockDim.x;
    sv[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off];
            if (ov > sv[threadIdx.x]) sv[threadIdx.x] = ov;
        }
        __syncthreads();
    }
    return sv[0];
}

__device__ __forceinline__ double block_sum_f64(double value, double* sd) {
    const uint32_t n = blockDim.x;
    sd[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) sd[threadIdx.x] += sd[threadIdx.x + off];
        __syncthreads();
    }
    return sd[0];
}

__device__ __forceinline__ uint32_t block_sum_u32(uint32_t value, uint32_t* su) {
    const uint32_t n = blockDim.x;
    su[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t off = n >> 1u; off > 0u; off >>= 1u) {
        if (threadIdx.x < off) su[threadIdx.x] += su[threadIdx.x + off];
        __syncthreads();
    }
    return su[0];
}

}  // namespace sampling_detail

__device__ __forceinline__ void sampling_sample_row(
    const float* __restrict__ row,
    uint32_t vocab_size,
    const SamplingRowParams& params,
    const SamplingConstraint& constraint,
    int32_t* __restrict__ out_token,
    uint32_t* error_word,
    uint32_t* attempt_count = nullptr) {
    using namespace sampling_detail;

    __shared__ float s_val[1024];
    __shared__ uint32_t s_tok[1024];
    __shared__ double s_acc[1024];

    if (attempt_count != nullptr && threadIdx.x == 0u) *attempt_count = 0u;

    if (params.mode == kSamplingNoneMode) {
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    if (params.mode == kSamplingGreedyMode) {
        float best_value = kNegInf;
        uint32_t best_token = kNoToken;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            if (!sampling_token_allowed(constraint, v)) continue;
            const float val = row[v];
            if (argmax_better(val, v, best_value, best_token)) {
                best_value = val;
                best_token = v;
            }
        }
        block_argmax(best_value, best_token, s_val, s_tok);
        if (threadIdx.x == 0u) {
            *out_token = best_token == kNoToken ? -1 : static_cast<int32_t>(best_token);
        }
        __syncthreads();
        if (best_token == kNoToken) sampling_report_error(error_word);
        return;
    }

    if (vocab_size == 0u || !(params.temperature > 0.0f)) {
        sampling_report_error(error_word);
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    const float temperature = params.temperature;
    const float* logits = row;

    float lmax = kNegInf;
    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        if (!sampling_token_allowed(constraint, v)) continue;
        const float val = logits[v];
        if (val > lmax) lmax = val;
    }
    lmax = block_max(lmax, s_val);

    bool usable = isfinite(lmax) != 0;
    double z = 0.0;
    if (usable) {
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            if (!sampling_token_allowed(constraint, v)) continue;
            z += static_cast<double>(::expf((logits[v] - lmax) / temperature));
        }
        z = block_sum_f64(z, s_acc);
        usable = (isfinite(z) != 0) && z > 0.0;
    }

    if (!usable) {
        sampling_report_error(error_word);
        if (threadIdx.x == 0u) *out_token = -1;
        __syncthreads();
        return;
    }

    const double top_p_mass = static_cast<double>(params.top_p) * z;

    int32_t result = -1;
    uint32_t attempt = 0u;
    while (attempt < kMaxSamplingAttempts) {
        float best_score = kNegInf;
        uint32_t best_token = kNoToken;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            if (!sampling_token_allowed(constraint, v)) continue;
            const float u =
                sampling_uniform(params.seed, params.sample_index, attempt, v);
            const float score =
                (logits[v] - lmax) / temperature + sampling_gumbel(u);
            if (argmax_better(score, v, best_score, best_token)) {
                best_score = score;
                best_token = v;
            }
        }
        block_argmax(best_score, best_token, s_val, s_tok);
        const uint32_t candidate = s_tok[0];
        if (candidate == kNoToken) {
            ++attempt;
            continue;
        }
        const float candidate_logit = logits[candidate];

        uint32_t better_count = 0u;
        double better_mass = 0.0;
        for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            if (!sampling_token_allowed(constraint, v)) continue;
            const float val = logits[v];
            const bool better =
                (val > candidate_logit) || (val == candidate_logit && v < candidate);
            if (better) {
                ++better_count;
                better_mass +=
                    static_cast<double>(::expf((val - lmax) / temperature));
            }
        }
        better_count = block_sum_u32(better_count, s_tok);
        better_mass = block_sum_f64(better_mass, s_acc);

        const bool top_k_ok = (params.top_k == 0u) || (better_count < params.top_k);
        const bool top_p_ok =
            (params.top_p >= 1.0f) || (better_mass < top_p_mass);
        if (top_k_ok && top_p_ok) {
            result = static_cast<int32_t>(candidate);
            break;
        }
        ++attempt;
    }

    if (result < 0) sampling_report_error(error_word);
    if (threadIdx.x == 0u) {
        *out_token = result;
        if (attempt_count != nullptr) {
            *attempt_count = (result < 0) ? kMaxSamplingAttempts : (attempt + 1u);
        }
    }
    __syncthreads();
}

}  // namespace ps::kernel
