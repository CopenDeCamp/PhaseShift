#include <phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.h>

#include "detail/coarse_topn_device.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = kDflash2Int2TopnThreads;
constexpr uint32_t kPerThread = kDflash2Int2TopnPerThread;
constexpr uint32_t kWarps = kThreads / 32u;
constexpr uint32_t kMergeThreads = 128u;
constexpr uint32_t kMergeWarps = kMergeThreads / 32u;

using detail::TopnKeyBest;
using detail::topn_block_best_key;
using detail::topn_broadcast;
using detail::topn_insert_key;
using detail::topn_key;
using detail::topn_warp_best_key;

__global__ void __launch_bounds__(kThreads) coarse_topn_partition_kernel(
    const float* __restrict__ logits,
    const uint32_t vocab,
    const uint32_t row_stride,
    const uint32_t partitions,
    const uint32_t partition_size,
    const uint32_t pool,
    int32_t* __restrict__ scratch_ids,
    float* __restrict__ scratch_logits,
    const uint32_t scratch_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t row = blockIdx.x / partitions;
    const uint32_t part = blockIdx.x - row * partitions;
    const float* rowp = logits + static_cast<std::size_t>(row) * row_stride;
    const uint32_t beg = part * partition_size;
    uint32_t end = beg + partition_size;
    if (end > vocab)
        end = vocab;

    uint64_t local_keys[kPerThread];
    float local_values[kPerThread];
    int32_t local_ids[kPerThread];
    #pragma unroll
    for (uint32_t i = 0u; i < kPerThread; ++i) {
        local_keys[i] = 0ull;
        local_values[i] = -INFINITY;
        local_ids[i] = 0x7FFFFFFF;
    }
    uint32_t count = 0u;
    for (uint32_t v = beg + tid; v < end; v += kThreads) {
        const float value = rowp[v];
        if (value == -INFINITY)
            continue;
        if (count < kPerThread) {
            local_keys[count] = topn_key(value, static_cast<int32_t>(v));
            local_values[count] = value;
            local_ids[count] = static_cast<int32_t>(v);
            ++count;
        }
    }

    static_assert((kPerThread & (kPerThread - 1u)) == 0u, "bitonic needs power of two");
    #pragma unroll
    for (uint32_t k = 2u; k <= kPerThread; k <<= 1u) {
        #pragma unroll
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            #pragma unroll
            for (uint32_t i = 0u; i < kPerThread; ++i) {
                const uint32_t l = i ^ j;
                if (l <= i)
                    continue;
                const bool up = (i & k) == 0u;
                const uint64_t ki = local_keys[i];
                const uint64_t kl = local_keys[l];
                const float vi = local_values[i];
                const float vl = local_values[l];
                const int32_t ii = local_ids[i];
                const int32_t il = local_ids[l];
                const bool sw = up ? (ki < kl) : (ki > kl);
                local_keys[i] = sw ? kl : ki;
                local_keys[l] = sw ? ki : kl;
                local_values[i] = sw ? vl : vi;
                local_values[l] = sw ? vi : vl;
                local_ids[i] = sw ? il : ii;
                local_ids[l] = sw ? ii : il;
            }
        }
    }

    __shared__ float warp_pool_value[kWarps * kDflash2Int2MaxPool];
    __shared__ int32_t warp_pool_id[kWarps * kDflash2Int2MaxPool];

    uint32_t cursor = 0u;
    for (uint32_t o = 0u; o < pool; ++o) {
        TopnKeyBest b{0ull, -INFINITY, 0x7FFFFFFF, -1};
        if (cursor < kPerThread && local_keys[cursor] != 0ull) {
            b.key = local_keys[cursor];
            b.value = local_values[cursor];
            b.id = local_ids[cursor];
            b.index = static_cast<int32_t>(lane);
        }
        b = topn_broadcast(topn_warp_best_key<32u>(b));
        if (lane == 0u) {
            warp_pool_value[warp * pool + o] = b.value;
            warp_pool_id[warp * pool + o] = b.id;
        }
        if (b.index == static_cast<int32_t>(lane))
            ++cursor;
    }

    __syncthreads();

    if (warp == 0u) {
        uint32_t wcur = 0u;
        for (uint32_t o = 0u; o < pool; ++o) {
            TopnKeyBest b{0ull, -INFINITY, 0x7FFFFFFF, -1};
            if (lane < kWarps && wcur < pool) {
                const float value = warp_pool_value[lane * pool + wcur];
                if (value != -INFINITY) {
                    const int32_t id = warp_pool_id[lane * pool + wcur];
                    b.key = topn_key(value, id);
                    b.value = value;
                    b.id = id;
                    b.index = static_cast<int32_t>(lane);
                }
            }
            b = topn_broadcast(topn_warp_best_key<32u>(b));
            if (lane == 0u) {
                const std::size_t off = static_cast<std::size_t>(row) * scratch_row_stride +
                                        static_cast<std::size_t>(part) * pool + o;
                scratch_logits[off] = b.value;
                scratch_ids[off] = b.id;
            }
            if (b.index == static_cast<int32_t>(lane))
                ++wcur;
        }
    }
}

__global__ void __launch_bounds__(kMergeThreads) coarse_topn_merge_kernel(
    const int32_t* __restrict__ scratch_ids,
    float* __restrict__ scratch_logits,
    const uint32_t partitions,
    const uint32_t pool,
    const uint32_t scratch_row_stride,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_logits) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const int32_t* row_ids =
        scratch_ids + static_cast<std::size_t>(row) * scratch_row_stride;
    float* row_values =
        scratch_logits + static_cast<std::size_t>(row) * scratch_row_stride;

    __shared__ uint64_t shared_key[kMergeWarps];
    __shared__ float shared_value[kMergeWarps];
    __shared__ int32_t shared_id[kMergeWarps];
    __shared__ int32_t shared_index[kMergeWarps];

    uint32_t cursor = 0u;
    for (uint32_t o = 0u; o < pool; ++o) {
        TopnKeyBest b{0ull, -INFINITY, 0x7FFFFFFF, -1};
        if (tid < partitions && cursor < pool) {
            const float value =
                row_values[static_cast<std::size_t>(tid) * pool + cursor];
            if (value != -INFINITY) {
                const int32_t id = row_ids[static_cast<std::size_t>(tid) * pool + cursor];
                b.key = topn_key(value, id);
                b.value = value;
                b.id = id;
                b.index = static_cast<int32_t>(tid);
            }
        }
        b = topn_block_best_key<kMergeWarps>(b, shared_key, shared_value, shared_id,
                                             shared_index);
        if (tid == 0u) {
            const bool found = b.index >= 0;
            out_ids[static_cast<std::size_t>(row) * pool + o] = found ? b.id : 0;
            out_logits[static_cast<std::size_t>(row) * pool + o] =
                found ? b.value : -INFINITY;
        }
        if (b.index == static_cast<int32_t>(tid))
            ++cursor;
    }
}

}  // namespace

hipError_t launch_dflash2_coarse_topn(
    const float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t pool,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t scratch_row_stride,
    uint32_t partitions,
    hipStream_t stream) {
    if (logits == nullptr || out_ids == nullptr || out_logits == nullptr ||
        scratch_ids == nullptr || scratch_logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (pool == 0u || pool > kDflash2Int2MaxPool)
        return hipErrorInvalidValue;
    if (vocab == 0u || row_stride < vocab)
        return hipErrorInvalidValue;
    if (partitions == 0u || partitions > kDflash2Int2MaxPartitions)
        return hipErrorInvalidValue;
    const uint32_t partition_size = (vocab + partitions - 1u) / partitions;
    if (partition_size > kThreads * kPerThread)
        return hipErrorInvalidValue;
    if (static_cast<uint64_t>(partitions) * pool > scratch_row_stride)
        return hipErrorInvalidValue;

    coarse_topn_partition_kernel<<<rows * partitions, kThreads, 0, stream>>>(
        logits, vocab, row_stride, partitions, partition_size, pool, scratch_ids,
        scratch_logits, scratch_row_stride);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return err;
    return launch_dflash2_coarse_topn_merge(scratch_ids, scratch_logits, rows, partitions,
                                            pool, scratch_row_stride, out_ids, out_logits,
                                            stream);
}

hipError_t launch_dflash2_coarse_topn_merge(
    const int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t rows,
    uint32_t partitions,
    uint32_t pool,
    uint32_t scratch_row_stride,
    int32_t* out_ids,
    float* out_logits,
    hipStream_t stream) {
    if (scratch_ids == nullptr || scratch_logits == nullptr || out_ids == nullptr ||
        out_logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u)
        return hipSuccess;
    if (pool == 0u || pool > kDflash2Int2MaxPool)
        return hipErrorInvalidValue;
    if (partitions == 0u)
        return hipErrorInvalidValue;
    if (static_cast<uint64_t>(partitions) * pool > scratch_row_stride)
        return hipErrorInvalidValue;
    coarse_topn_merge_kernel<<<rows, kMergeThreads, 0, stream>>>(
        scratch_ids, scratch_logits, partitions, pool, scratch_row_stride, out_ids,
        out_logits);
    return hipGetLastError();
}

}  // namespace ps::kernel
