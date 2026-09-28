#include <phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

__device__ __forceinline__ uint32_t spec_restore_total(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

}  // namespace

extern "C" __global__ __launch_bounds__(kGdnSpecRestoreThreads)
void phaseshift_gpu_mcu_gdn_spec_restore(GdnSpecRestoreArgs args) {
    const uint32_t step = blockDim.x * gridDim.x;
    const uint32_t total =
        spec_restore_total(args.conv_elems, args.recurrent_elems);

    auto* conv_dst = reinterpret_cast<uint16_t*>(
        static_cast<unsigned char*>(args.conv_pool) +
        static_cast<size_t>(args.sequence_slot) * args.conv_slot_stride * 2u);
    const auto* conv_src = reinterpret_cast<const uint16_t*>(
        static_cast<const unsigned char*>(args.conv_history) +
        static_cast<size_t>(args.history_row) * args.conv_history_stride * 2u);

    auto* rec_dst = reinterpret_cast<float*>(
        static_cast<unsigned char*>(args.recurrent_pool) +
        static_cast<size_t>(args.sequence_slot) * args.recurrent_slot_stride *
            4u);
    const auto* rec_src = reinterpret_cast<const float*>(
        static_cast<const unsigned char*>(args.recurrent_history) +
        static_cast<size_t>(args.history_row) * args.recurrent_history_stride *
            4u);

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += step) {
        if (i < args.conv_elems) {
            conv_dst[i] = conv_src[i];
        }
        if (i < args.recurrent_elems) {
            rec_dst[i] = rec_src[i];
        }
    }
}

hipError_t launch_gdn_spec_restore(
    const GdnSpecRestoreArgs& args, uint32_t total_elems, hipStream_t stream) {
    if (args.conv_pool == nullptr || args.recurrent_pool == nullptr) {
        return hipErrorInvalidValue;
    }
    if (total_elems == 0u) {
        return hipErrorInvalidValue;
    }
    const uint32_t blocks =
        (total_elems + kGdnSpecRestoreThreads - 1u) / kGdnSpecRestoreThreads;
    phaseshift_gpu_mcu_gdn_spec_restore<<<blocks, kGdnSpecRestoreThreads, 0,
                                          stream>>>(args);
    return hipGetLastError();
}

extern "C" __global__ __launch_bounds__(kGdnSpecRestoreThreads)
void phaseshift_gpu_mcu_gdn_spec_restore_from_counts(
    GdnSpecRestoreFromCountsArgs args) {
    if (args.context == nullptr || args.committed_counts == nullptr) return;
    const uint32_t total =
        spec_restore_total(args.conv_elems, args.recurrent_elems);
    const uint32_t step = blockDim.x * gridDim.x;
    for (uint32_t request = 0u; request < args.context->num_requests;
         ++request) {
        const ::ps::runtime::DeviceRequestDescriptor& descriptor =
            args.context->requests[request];
        if (descriptor.execution_class !=
            ::ps::runtime::ExecutionClass::SPEC_VERIFY) {
            continue;
        }
        const uint32_t accepted = args.committed_counts[request];
        if (accepted > args.max_candidates) continue;
        const uint32_t sequence_slot =
            args.context->row_sequence_slots[descriptor.row_begin];
        auto* conv_dst = reinterpret_cast<uint16_t*>(
            static_cast<unsigned char*>(args.conv_pool) +
            static_cast<size_t>(sequence_slot) * args.conv_slot_stride * 2u);
        const auto* conv_src = reinterpret_cast<const uint16_t*>(
            static_cast<const unsigned char*>(args.conv_history) +
            static_cast<size_t>(accepted) * args.conv_history_stride * 2u);
        auto* rec_dst = reinterpret_cast<float*>(
            static_cast<unsigned char*>(args.recurrent_pool) +
            static_cast<size_t>(sequence_slot) * args.recurrent_slot_stride *
                4u);
        const auto* rec_src = reinterpret_cast<const float*>(
            static_cast<const unsigned char*>(args.recurrent_history) +
            static_cast<size_t>(accepted) * args.recurrent_history_stride *
                4u);
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
             i += step) {
            if (i < args.conv_elems) conv_dst[i] = conv_src[i];
            if (i < args.recurrent_elems) rec_dst[i] = rec_src[i];
        }
        __syncthreads();
    }
}

hipError_t launch_gdn_spec_restore_from_counts(
    const GdnSpecRestoreFromCountsArgs& args, uint32_t total_elems,
    uint32_t request_count, hipStream_t stream) {
    if (args.context == nullptr || args.committed_counts == nullptr ||
        args.conv_pool == nullptr || args.recurrent_pool == nullptr) {
        return hipErrorInvalidValue;
    }
    if (total_elems == 0u || request_count == 0u) {
        return hipErrorInvalidValue;
    }
    const uint32_t blocks =
        (total_elems + kGdnSpecRestoreThreads - 1u) / kGdnSpecRestoreThreads;
    phaseshift_gpu_mcu_gdn_spec_restore_from_counts<<<blocks,
                                                      kGdnSpecRestoreThreads, 0,
                                                      stream>>>(args);
    return hipGetLastError();
}

}  // namespace ps::kernel
