#pragma once

#include <phaseshift/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kGdnSpecRestoreSymbol =
    "phaseshift_gpu_mcu_gdn_spec_restore";

inline constexpr uint32_t kGdnSpecRestoreThreads = 256u;

struct GdnSpecRestoreArgs {
    const void* conv_history = nullptr;
    void* conv_pool = nullptr;
    const void* recurrent_history = nullptr;
    void* recurrent_pool = nullptr;

    uint64_t conv_slot_stride = 0;
    uint64_t recurrent_slot_stride = 0;
    uint64_t conv_history_stride = 0;
    uint64_t recurrent_history_stride = 0;

    uint32_t sequence_slot = 0;
    uint32_t history_row = 0;
    uint32_t conv_elems = 0;
    uint32_t recurrent_elems = 0;
};

static_assert(sizeof(GdnSpecRestoreArgs) == 80);
static_assert(alignof(GdnSpecRestoreArgs) == 8);
static_assert(offsetof(GdnSpecRestoreArgs, conv_history) == 0);
static_assert(offsetof(GdnSpecRestoreArgs, conv_pool) == 8);
static_assert(offsetof(GdnSpecRestoreArgs, recurrent_history) == 16);
static_assert(offsetof(GdnSpecRestoreArgs, recurrent_pool) == 24);
static_assert(offsetof(GdnSpecRestoreArgs, conv_slot_stride) == 32);
static_assert(offsetof(GdnSpecRestoreArgs, recurrent_slot_stride) == 40);
static_assert(offsetof(GdnSpecRestoreArgs, conv_history_stride) == 48);
static_assert(offsetof(GdnSpecRestoreArgs, recurrent_history_stride) == 56);
static_assert(offsetof(GdnSpecRestoreArgs, sequence_slot) == 64);
static_assert(offsetof(GdnSpecRestoreArgs, history_row) == 68);
static_assert(offsetof(GdnSpecRestoreArgs, conv_elems) == 72);
static_assert(offsetof(GdnSpecRestoreArgs, recurrent_elems) == 76);

hipError_t launch_gdn_spec_restore(
    const GdnSpecRestoreArgs& args, uint32_t total_elems, hipStream_t stream);

inline constexpr const char* kGdnSpecRestoreFromCountsSymbol =
    "phaseshift_gpu_mcu_gdn_spec_restore_from_counts";

struct GdnSpecRestoreFromCountsArgs {
    const ::ps::runtime::DeviceBatchContext* context = nullptr;
    const uint32_t* committed_counts = nullptr;
    const void* conv_history = nullptr;
    void* conv_pool = nullptr;
    const void* recurrent_history = nullptr;
    void* recurrent_pool = nullptr;

    uint64_t conv_slot_stride = 0;
    uint64_t recurrent_slot_stride = 0;
    uint64_t conv_history_stride = 0;
    uint64_t recurrent_history_stride = 0;

    uint32_t conv_elems = 0;
    uint32_t recurrent_elems = 0;
    uint32_t max_candidates = 0;
    uint32_t reserved = 0;
};

static_assert(sizeof(GdnSpecRestoreFromCountsArgs) == 96);
static_assert(alignof(GdnSpecRestoreFromCountsArgs) == 8);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, context) == 0);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, committed_counts) == 8);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, conv_history) == 16);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, conv_pool) == 24);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, recurrent_history) == 32);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, recurrent_pool) == 40);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, conv_slot_stride) == 48);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, recurrent_slot_stride) ==
              56);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, conv_history_stride) == 64);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, recurrent_history_stride) ==
              72);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, conv_elems) == 80);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, recurrent_elems) == 84);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, max_candidates) == 88);
static_assert(offsetof(GdnSpecRestoreFromCountsArgs, reserved) == 92);

hipError_t launch_gdn_spec_restore_from_counts(
    const GdnSpecRestoreFromCountsArgs& args, uint32_t total_elems,
    uint32_t request_count, hipStream_t stream);

}  // namespace ps::kernel
