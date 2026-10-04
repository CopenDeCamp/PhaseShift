#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/gpu_mcu/device_completion.h>
#include <phaseshift/runtime/gpu_mcu/gdn_reset.h>
#include <phaseshift/runtime/gpu_mcu/retained_packet.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

constexpr uint32_t kMcuMaxNodes = 16384;
constexpr uint32_t kMcuMaxVariants = 64;
constexpr uint32_t kMcuMaxCompletionSlots = 32;
constexpr uint32_t kMcuMaxKernargSlots = kMcuMaxNodes;
constexpr uint32_t kMcuMaxRmsNormInvocations = 4096;
constexpr uint32_t kMcuMaxQuantizeInvocations = 4096;
constexpr uint32_t kMcuMaxQuantizeE4m3Invocations = 4096;
constexpr uint32_t kMcuMaxPsq4Invocations = 4096;
constexpr uint32_t kMcuMaxPsq4MultiInvocations = 4096;
constexpr uint32_t kMcuMaxVerifyAcceptInvocations = 4096;
constexpr uint32_t kMcuMaxGdnSpecRestoreInvocations = 4096;
constexpr uint32_t kMcuMaxArgmaxF32Invocations = 4096;
constexpr uint32_t kMcuMaxElementwiseInvocations = 4096;
constexpr uint32_t kMcuMaxRopeInvocations = 4096;
constexpr uint32_t kMcuMaxKvAppendInvocations = 4096;
constexpr uint32_t kMcuMaxPagedAttentionInvocations = 4096;
constexpr uint32_t kMcuMaxPagedAttentionSplitInvocations = 4096;
constexpr uint32_t kMcuMaxPagedAttentionReduceInvocations = 4096;
constexpr uint32_t kMcuMaxBf16Invocations = 4096;
constexpr uint32_t kMcuMaxL2Invocations = 4096;
constexpr uint32_t kMcuMaxEmbeddingInvocations = 4096;
constexpr uint32_t kMcuMaxOutputGatherInvocations = 4096;
constexpr uint32_t kMcuMaxGdnConv1dInvocations = 4096;
constexpr uint32_t kMcuMaxGdnRecurrenceInvocations = 4096;
constexpr uint32_t kMcuMaxTiming = 4096;
constexpr uint32_t kMcuMaxDebugRecords = 4096;

enum class McuSupervisorState : uint32_t {
    boot = 0,
    idle = 1,
    running = 2,
    fault = 3,
    stopping = 4,
};

enum class McuFaultCode : uint32_t {
    none = 0,
    invalid_plan = 1,
    invalid_node = 2,
    invalid_variant = 3,
    kernarg_slot_unavailable = 4,
    queue_stage_failure = 5,
    completion_generation_mismatch = 6,
    kernarg_region_exhausted = 7,
    kernarg_build_failed = 8,
};

enum : uint32_t {
    kMcuNodeDispatch = 1u << 0,
    kMcuNodeWait = 1u << 1,
    kMcuNodeEnd = 1u << 2,
};

enum class McuDoorbellMode : uint32_t {
    per_packet = 0,
    coalesce = 1,
    first_only = 2,
};

enum : uint32_t {
    kMcuRecordDoorbell = 1u << 0,
    kMcuRecordAppendAhead = 1u << 1,
    kMcuRecordQueueEmpty = 1u << 2,
    kMcuRecordRefill = 1u << 3,
    kMcuRecordBackpressure = 1u << 4,
    kMcuRecordWrap = 1u << 5,
};

struct alignas(16) McuDispatchRecord {
    uint64_t dispatch_seq = 0;
    uint64_t publish_ts = 0;
    uint64_t doorbell_ts = 0;
    uint64_t generation = 0;
    uint32_t ring_slot = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

static_assert(sizeof(McuDispatchRecord) == 48);
static_assert(alignof(McuDispatchRecord) == 16);

enum : uint16_t {
    kMcuNoNext = 0xffffu,
    kMcuNoInput = 0xffffu,
};

enum : uint16_t {
    kMcuKernargRecipeNone = 0,
    kMcuKernargRecipeProbe = 1,
    kMcuKernargRecipeRmsNormBf16PfOnePlus = 2,
    kMcuKernargRecipeActivationQuantizeA8 = 3,
    kMcuKernargRecipePsq4Decode1Bf16U16 = 4,
    kMcuKernargRecipePsq4Decode1Bf16U8 = 5,
    kMcuKernargRecipeActivationQuantizeE4m3K5120 = 6,
    kMcuKernargRecipeElementwise = 7,
    kMcuKernargRecipeBf16ExactRows = 8,
    kMcuKernargRecipeL2Normalize = 9,
    kMcuKernargRecipeGdnConv1d = 10,
    kMcuKernargRecipeGdnRecurrence = 11,
    kMcuKernargRecipeRope = 12,
    kMcuKernargRecipeKvAppend = 13,
    kMcuKernargRecipeAttentionPaged = 14,
    kMcuKernargRecipeAttentionPagedSplit = 15,
    kMcuKernargRecipeAttentionPagedReduce = 16,
    kMcuKernargRecipeGdnReset = 17,
    kMcuKernargRecipePsq8Decode1Bf16U8 = 18,
    kMcuKernargRecipePsq4MultiRowBf16 = 19,
    kMcuKernargRecipeVerifyAcceptPrefix = 20,
    kMcuKernargRecipeGdnSpecRestore = 21,
    kMcuKernargRecipeArgmaxF32 = 22,
    kMcuKernargRecipeEmbeddingBf16 = 23,
    kMcuKernargRecipeOutputGatherBf16 = 24,
    kMcuKernargRecipeVerifyAcceptBatch = 25,
    kMcuKernargRecipeGdnSpecRestoreFromCounts = 26,
    kMcuKernargRecipeCount = 27,
};

struct McuActivationQuantizeInvocation {
    uint64_t input = 0;
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
};

static_assert(sizeof(McuActivationQuantizeInvocation) == 48);
static_assert(alignof(McuActivationQuantizeInvocation) == 8);
static_assert(offsetof(McuActivationQuantizeInvocation, input) == 0);
static_assert(offsetof(McuActivationQuantizeInvocation, codes) == 8);
static_assert(offsetof(McuActivationQuantizeInvocation, scales) == 16);
static_assert(offsetof(McuActivationQuantizeInvocation, rows) == 24);
static_assert(offsetof(McuActivationQuantizeInvocation, k) == 28);
static_assert(offsetof(McuActivationQuantizeInvocation, k_padded) == 32);
static_assert(offsetof(McuActivationQuantizeInvocation, input_row_stride) == 36);
static_assert(offsetof(McuActivationQuantizeInvocation, code_row_stride) == 40);
static_assert(offsetof(McuActivationQuantizeInvocation, scale_row_stride) == 44);

struct McuActivationQuantizeE4m3Invocation {
    uint64_t input = 0;
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
};

static_assert(sizeof(McuActivationQuantizeE4m3Invocation) == 40);
static_assert(alignof(McuActivationQuantizeE4m3Invocation) == 8);
static_assert(offsetof(McuActivationQuantizeE4m3Invocation, input) == 0);
static_assert(offsetof(McuActivationQuantizeE4m3Invocation, codes) == 8);
static_assert(offsetof(McuActivationQuantizeE4m3Invocation, scales) == 16);
static_assert(offsetof(McuActivationQuantizeE4m3Invocation, input_row_stride) == 24);
static_assert(offsetof(McuActivationQuantizeE4m3Invocation, code_row_stride) == 28);
static_assert(offsetof(McuActivationQuantizeE4m3Invocation, scale_row_stride) == 32);

struct McuPsq4Decode1Invocation {
    uint64_t weight_codes = 0;
    uint64_t weight_scales = 0;
    uint64_t activation_codes = 0;
    uint64_t activation_scales = 0;
    uint64_t output = 0;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
};

static_assert(sizeof(McuPsq4Decode1Invocation) == 48);
static_assert(alignof(McuPsq4Decode1Invocation) == 8);
static_assert(offsetof(McuPsq4Decode1Invocation, weight_codes) == 0);
static_assert(offsetof(McuPsq4Decode1Invocation, weight_scales) == 8);
static_assert(offsetof(McuPsq4Decode1Invocation, activation_codes) == 16);
static_assert(offsetof(McuPsq4Decode1Invocation, activation_scales) == 24);
static_assert(offsetof(McuPsq4Decode1Invocation, output) == 32);
static_assert(offsetof(McuPsq4Decode1Invocation, k_padded) == 40);
static_assert(offsetof(McuPsq4Decode1Invocation, weight_scale_stride) == 44);

struct McuPsq4MultiRowInvocation {
    uint64_t weight_codes = 0;
    uint64_t weight_scales = 0;
    uint64_t activation_codes = 0;
    uint64_t activation_scales = 0;
    uint64_t output = 0;
    uint32_t output_dtype = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k_padded = 0;
    uint32_t weight_scale_stride = 0;
    uint32_t activation_code_stride = 0;
    uint32_t activation_scale_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(McuPsq4MultiRowInvocation) == 72);
static_assert(alignof(McuPsq4MultiRowInvocation) == 8);
static_assert(offsetof(McuPsq4MultiRowInvocation, weight_codes) == 0);
static_assert(offsetof(McuPsq4MultiRowInvocation, weight_scales) == 8);
static_assert(offsetof(McuPsq4MultiRowInvocation, activation_codes) == 16);
static_assert(offsetof(McuPsq4MultiRowInvocation, activation_scales) == 24);
static_assert(offsetof(McuPsq4MultiRowInvocation, output) == 32);
static_assert(offsetof(McuPsq4MultiRowInvocation, output_dtype) == 40);
static_assert(offsetof(McuPsq4MultiRowInvocation, rows) == 44);
static_assert(offsetof(McuPsq4MultiRowInvocation, out_features) == 48);
static_assert(offsetof(McuPsq4MultiRowInvocation, k_padded) == 52);
static_assert(offsetof(McuPsq4MultiRowInvocation, weight_scale_stride) == 56);
static_assert(offsetof(McuPsq4MultiRowInvocation, activation_code_stride) == 60);
static_assert(offsetof(McuPsq4MultiRowInvocation, activation_scale_stride) == 64);
static_assert(offsetof(McuPsq4MultiRowInvocation, output_row_stride) == 68);

struct McuVerifyAcceptPrefixInvocation {
    uint64_t candidates = 0;
    uint64_t sampled_tokens = 0;
    uint64_t candidate_counts = 0;
    uint64_t committed_counts = 0;
    uint32_t request_count = 0;
    uint32_t candidate_stride = 0;
    uint32_t max_candidates = 0;
    uint32_t reserved = 0;
    uint64_t requests = 0;
    uint64_t output_rows = 0;
};

static_assert(sizeof(McuVerifyAcceptPrefixInvocation) == 64);
static_assert(alignof(McuVerifyAcceptPrefixInvocation) == 8);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, candidates) == 0);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, sampled_tokens) == 8);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, candidate_counts) == 16);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, committed_counts) == 24);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, request_count) == 32);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, candidate_stride) == 36);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, max_candidates) == 40);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, reserved) == 44);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, requests) == 48);
static_assert(offsetof(McuVerifyAcceptPrefixInvocation, output_rows) == 56);

struct McuGdnSpecRestoreInvocation {
    uint64_t conv_history = 0;
    uint64_t conv_pool = 0;
    uint64_t recurrent_history = 0;
    uint64_t recurrent_pool = 0;
    uint64_t conv_slot_stride = 0;
    uint64_t recurrent_slot_stride = 0;
    uint64_t conv_history_stride = 0;
    uint64_t recurrent_history_stride = 0;
    uint32_t sequence_slot = 0;
    uint32_t history_row = 0;
    uint32_t conv_elems = 0;
    uint32_t recurrent_elems = 0;
};

static_assert(sizeof(McuGdnSpecRestoreInvocation) == 80);
static_assert(alignof(McuGdnSpecRestoreInvocation) == 8);
static_assert(offsetof(McuGdnSpecRestoreInvocation, conv_history) == 0);
static_assert(offsetof(McuGdnSpecRestoreInvocation, conv_pool) == 8);
static_assert(offsetof(McuGdnSpecRestoreInvocation, recurrent_history) == 16);
static_assert(offsetof(McuGdnSpecRestoreInvocation, recurrent_pool) == 24);
static_assert(offsetof(McuGdnSpecRestoreInvocation, conv_slot_stride) == 32);
static_assert(offsetof(McuGdnSpecRestoreInvocation, recurrent_slot_stride) == 40);
static_assert(offsetof(McuGdnSpecRestoreInvocation, conv_history_stride) == 48);
static_assert(offsetof(McuGdnSpecRestoreInvocation, recurrent_history_stride) ==
              56);
static_assert(offsetof(McuGdnSpecRestoreInvocation, sequence_slot) == 64);
static_assert(offsetof(McuGdnSpecRestoreInvocation, history_row) == 68);
static_assert(offsetof(McuGdnSpecRestoreInvocation, conv_elems) == 72);
static_assert(offsetof(McuGdnSpecRestoreInvocation, recurrent_elems) == 76);

struct McuArgmaxF32Invocation {
    uint64_t logits = 0;
    uint64_t sampling = 0;
    uint64_t sampled_tokens = 0;
    uint32_t outputs = 0;
    uint32_t vocab_size = 0;
    uint32_t logits_row_stride = 0;
    uint32_t reserved0 = 0;
    uint64_t constraint_masks = 0;
    uint32_t constraint_mask_words = 0;
    uint32_t reserved1 = 0;
    uint64_t output_rows = 0;
};

static_assert(sizeof(McuArgmaxF32Invocation) == 64);
static_assert(alignof(McuArgmaxF32Invocation) == 8);
static_assert(offsetof(McuArgmaxF32Invocation, logits) == 0);
static_assert(offsetof(McuArgmaxF32Invocation, sampling) == 8);
static_assert(offsetof(McuArgmaxF32Invocation, sampled_tokens) == 16);
static_assert(offsetof(McuArgmaxF32Invocation, outputs) == 24);
static_assert(offsetof(McuArgmaxF32Invocation, vocab_size) == 28);
static_assert(offsetof(McuArgmaxF32Invocation, logits_row_stride) == 32);
static_assert(offsetof(McuArgmaxF32Invocation, constraint_masks) == 40);
static_assert(offsetof(McuArgmaxF32Invocation, constraint_mask_words) == 48);
static_assert(offsetof(McuArgmaxF32Invocation, output_rows) == 56);

struct McuRmsNormInvocation {
    uint64_t input = 0;
    uint64_t weight = 0;
    uint64_t output = 0;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    float eps = 0.0f;
};

static_assert(sizeof(McuRmsNormInvocation) == 48);
static_assert(alignof(McuRmsNormInvocation) == 8);
static_assert(offsetof(McuRmsNormInvocation, input) == 0);
static_assert(offsetof(McuRmsNormInvocation, weight) == 8);
static_assert(offsetof(McuRmsNormInvocation, output) == 16);
static_assert(offsetof(McuRmsNormInvocation, rows) == 24);
static_assert(offsetof(McuRmsNormInvocation, features) == 28);
static_assert(offsetof(McuRmsNormInvocation, group_size) == 32);
static_assert(offsetof(McuRmsNormInvocation, input_row_stride) == 36);
static_assert(offsetof(McuRmsNormInvocation, output_row_stride) == 40);
static_assert(offsetof(McuRmsNormInvocation, eps) == 44);

struct McuElementwiseInvocation {
    uint64_t in0 = 0;
    uint32_t in0_row_stride = 0;
    uint64_t in1 = 0;
    uint32_t in1_row_stride = 0;
    uint64_t out0 = 0;
    uint32_t out0_row_stride = 0;
    uint64_t out1 = 0;
    uint32_t out1_row_stride = 0;
    uint32_t features = 0;
    uint32_t features_b = 0;
    uint32_t aux = 0;
    uint32_t reserved = 0;
    float scalar_a = 0.0f;
    uint32_t rows = 0;
};

static_assert(sizeof(McuElementwiseInvocation) == 88);
static_assert(alignof(McuElementwiseInvocation) == 8);
static_assert(offsetof(McuElementwiseInvocation, in0) == 0);
static_assert(offsetof(McuElementwiseInvocation, in0_row_stride) == 8);
static_assert(offsetof(McuElementwiseInvocation, in1) == 16);
static_assert(offsetof(McuElementwiseInvocation, in1_row_stride) == 24);
static_assert(offsetof(McuElementwiseInvocation, out0) == 32);
static_assert(offsetof(McuElementwiseInvocation, out0_row_stride) == 40);
static_assert(offsetof(McuElementwiseInvocation, out1) == 48);
static_assert(offsetof(McuElementwiseInvocation, out1_row_stride) == 56);
static_assert(offsetof(McuElementwiseInvocation, features) == 60);
static_assert(offsetof(McuElementwiseInvocation, features_b) == 64);
static_assert(offsetof(McuElementwiseInvocation, aux) == 68);
static_assert(offsetof(McuElementwiseInvocation, scalar_a) == 76);
static_assert(offsetof(McuElementwiseInvocation, rows) == 80);

struct McuRopeInvocation {
    uint64_t input = 0;
    uint32_t input_row_stride = 0;
    uint64_t output = 0;
    uint32_t output_row_stride = 0;
    uint64_t positions = 0;
    uint64_t inv_freq = 0;
    uint32_t features = 0;
};

static_assert(sizeof(McuRopeInvocation) == 56);
static_assert(alignof(McuRopeInvocation) == 8);
static_assert(offsetof(McuRopeInvocation, input) == 0);
static_assert(offsetof(McuRopeInvocation, input_row_stride) == 8);
static_assert(offsetof(McuRopeInvocation, output) == 16);
static_assert(offsetof(McuRopeInvocation, output_row_stride) == 24);
static_assert(offsetof(McuRopeInvocation, positions) == 32);
static_assert(offsetof(McuRopeInvocation, inv_freq) == 40);
static_assert(offsetof(McuRopeInvocation, features) == 48);

struct McuKvAppendInvocation {
    uint64_t k_input = 0;
    uint64_t v_input = 0;
    uint32_t k_row_stride = 0;
    uint32_t v_row_stride = 0;
    uint64_t row_sequence_slots = 0;
    uint64_t row_positions = 0;
    uint64_t block_tables = 0;
    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;
    uint32_t rows = 0;
    uint32_t layer = 0;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
};

static_assert(sizeof(McuKvAppendInvocation) == 112);
static_assert(alignof(McuKvAppendInvocation) == 8);
static_assert(offsetof(McuKvAppendInvocation, k_input) == 0);
static_assert(offsetof(McuKvAppendInvocation, v_input) == 8);
static_assert(offsetof(McuKvAppendInvocation, k_row_stride) == 16);
static_assert(offsetof(McuKvAppendInvocation, v_row_stride) == 20);
static_assert(offsetof(McuKvAppendInvocation, row_sequence_slots) == 24);
static_assert(offsetof(McuKvAppendInvocation, row_positions) == 32);
static_assert(offsetof(McuKvAppendInvocation, block_tables) == 40);
static_assert(offsetof(McuKvAppendInvocation, block_table_stride) == 48);
static_assert(offsetof(McuKvAppendInvocation, max_sequences) == 52);
static_assert(offsetof(McuKvAppendInvocation, rows) == 56);
static_assert(offsetof(McuKvAppendInvocation, layer) == 60);
static_assert(offsetof(McuKvAppendInvocation, page_tokens) == 64);
static_assert(offsetof(McuKvAppendInvocation, num_pages) == 68);
static_assert(offsetof(McuKvAppendInvocation, num_attention_layers) == 72);
static_assert(offsetof(McuKvAppendInvocation, kv_heads) == 76);
static_assert(offsetof(McuKvAppendInvocation, head_dim) == 80);
static_assert(offsetof(McuKvAppendInvocation, elems_per_token) == 84);
static_assert(offsetof(McuKvAppendInvocation, elems_per_page) == 88);
static_assert(offsetof(McuKvAppendInvocation, elems_per_layer) == 92);
static_assert(offsetof(McuKvAppendInvocation, k_pool) == 96);
static_assert(offsetof(McuKvAppendInvocation, v_pool) == 104);

struct McuPagedAttentionInvocation {
    uint64_t q = 0;
    uint64_t output = 0;
    uint32_t output_dtype = 0;
    uint32_t q_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint64_t row_sequence_slots = 0;
    uint64_t row_positions = 0;
    uint64_t block_tables = 0;
    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;
    uint32_t rows = 0;
    uint32_t layer = 0;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
    float scale = 0.0f;
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
};

static_assert(sizeof(McuPagedAttentionInvocation) == 128);
static_assert(alignof(McuPagedAttentionInvocation) == 8);
static_assert(offsetof(McuPagedAttentionInvocation, q) == 0);
static_assert(offsetof(McuPagedAttentionInvocation, output) == 8);
static_assert(offsetof(McuPagedAttentionInvocation, output_dtype) == 16);
static_assert(offsetof(McuPagedAttentionInvocation, q_row_stride) == 20);
static_assert(offsetof(McuPagedAttentionInvocation, output_row_stride) == 24);
static_assert(offsetof(McuPagedAttentionInvocation, row_sequence_slots) == 32);
static_assert(offsetof(McuPagedAttentionInvocation, row_positions) == 40);
static_assert(offsetof(McuPagedAttentionInvocation, block_tables) == 48);
static_assert(offsetof(McuPagedAttentionInvocation, block_table_stride) == 56);
static_assert(offsetof(McuPagedAttentionInvocation, max_sequences) == 60);
static_assert(offsetof(McuPagedAttentionInvocation, rows) == 64);
static_assert(offsetof(McuPagedAttentionInvocation, layer) == 68);
static_assert(offsetof(McuPagedAttentionInvocation, page_tokens) == 72);
static_assert(offsetof(McuPagedAttentionInvocation, num_pages) == 76);
static_assert(offsetof(McuPagedAttentionInvocation, num_attention_layers) == 80);
static_assert(offsetof(McuPagedAttentionInvocation, q_heads) == 84);
static_assert(offsetof(McuPagedAttentionInvocation, kv_heads) == 88);
static_assert(offsetof(McuPagedAttentionInvocation, head_dim) == 92);
static_assert(offsetof(McuPagedAttentionInvocation, elems_per_token) == 96);
static_assert(offsetof(McuPagedAttentionInvocation, elems_per_page) == 100);
static_assert(offsetof(McuPagedAttentionInvocation, elems_per_layer) == 104);
static_assert(offsetof(McuPagedAttentionInvocation, scale) == 108);
static_assert(offsetof(McuPagedAttentionInvocation, k_pool) == 112);
static_assert(offsetof(McuPagedAttentionInvocation, v_pool) == 120);

struct McuPagedAttentionSplitInvocation {
    uint64_t q = 0;
    uint64_t output = 0;
    uint32_t output_dtype = 0;
    uint32_t q_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint64_t row_sequence_slots = 0;
    uint64_t row_positions = 0;
    uint64_t block_tables = 0;
    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;
    uint32_t rows = 0;
    uint32_t layer = 0;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
    float scale = 0.0f;
    uint64_t k_pool = 0;
    uint64_t v_pool = 0;
    uint64_t partials = 0;
    uint32_t splits = 0;
};

static_assert(sizeof(McuPagedAttentionSplitInvocation) == 144);
static_assert(alignof(McuPagedAttentionSplitInvocation) == 8);
static_assert(offsetof(McuPagedAttentionSplitInvocation, scale) == 108);
static_assert(offsetof(McuPagedAttentionSplitInvocation, k_pool) == 112);
static_assert(offsetof(McuPagedAttentionSplitInvocation, v_pool) == 120);
static_assert(offsetof(McuPagedAttentionSplitInvocation, partials) == 128);
static_assert(offsetof(McuPagedAttentionSplitInvocation, splits) == 136);

struct McuPagedAttentionReduceInvocation {
    uint64_t q = 0;
    uint64_t output = 0;
    uint32_t output_dtype = 0;
    uint32_t q_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint64_t row_sequence_slots = 0;
    uint64_t row_positions = 0;
    uint64_t block_tables = 0;
    uint32_t block_table_stride = 0;
    uint32_t max_sequences = 0;
    uint32_t rows = 0;
    uint32_t layer = 0;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t q_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
    float scale = 0.0f;
    uint64_t partials = 0;
    uint32_t splits = 0;
};

static_assert(sizeof(McuPagedAttentionReduceInvocation) == 128);
static_assert(alignof(McuPagedAttentionReduceInvocation) == 8);
static_assert(offsetof(McuPagedAttentionReduceInvocation, scale) == 108);
static_assert(offsetof(McuPagedAttentionReduceInvocation, partials) == 112);
static_assert(offsetof(McuPagedAttentionReduceInvocation, splits) == 120);

struct McuBf16ExactRowsInvocation {
    uint64_t weight = 0;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t output_dtype = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(McuBf16ExactRowsInvocation) == 48);
static_assert(alignof(McuBf16ExactRowsInvocation) == 8);
static_assert(offsetof(McuBf16ExactRowsInvocation, weight) == 0);
static_assert(offsetof(McuBf16ExactRowsInvocation, input) == 8);
static_assert(offsetof(McuBf16ExactRowsInvocation, output) == 16);
static_assert(offsetof(McuBf16ExactRowsInvocation, output_dtype) == 24);
static_assert(offsetof(McuBf16ExactRowsInvocation, n) == 28);
static_assert(offsetof(McuBf16ExactRowsInvocation, k) == 32);
static_assert(offsetof(McuBf16ExactRowsInvocation, input_row_stride) == 36);
static_assert(offsetof(McuBf16ExactRowsInvocation, output_row_stride) == 40);

struct McuL2NormalizeInvocation {
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t group_size = 0;
    float eps = 0.0f;
};

static_assert(sizeof(McuL2NormalizeInvocation) == 32);
static_assert(alignof(McuL2NormalizeInvocation) == 8);
static_assert(offsetof(McuL2NormalizeInvocation, input) == 0);
static_assert(offsetof(McuL2NormalizeInvocation, output) == 8);
static_assert(offsetof(McuL2NormalizeInvocation, input_row_stride) == 16);
static_assert(offsetof(McuL2NormalizeInvocation, output_row_stride) == 20);
static_assert(offsetof(McuL2NormalizeInvocation, group_size) == 24);
static_assert(offsetof(McuL2NormalizeInvocation, eps) == 28);

struct McuEmbeddingBf16Invocation {
    uint64_t table = 0;
    uint64_t token_ids = 0;
    uint64_t output = 0;
    uint64_t error_word = 0;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(McuEmbeddingBf16Invocation) == 48);
static_assert(alignof(McuEmbeddingBf16Invocation) == 8);
static_assert(offsetof(McuEmbeddingBf16Invocation, table) == 0);
static_assert(offsetof(McuEmbeddingBf16Invocation, token_ids) == 8);
static_assert(offsetof(McuEmbeddingBf16Invocation, output) == 16);
static_assert(offsetof(McuEmbeddingBf16Invocation, error_word) == 24);
static_assert(offsetof(McuEmbeddingBf16Invocation, rows) == 32);
static_assert(offsetof(McuEmbeddingBf16Invocation, vocab_size) == 36);
static_assert(offsetof(McuEmbeddingBf16Invocation, hidden_size) == 40);
static_assert(offsetof(McuEmbeddingBf16Invocation, output_row_stride) == 44);

struct McuOutputGatherBf16Invocation {
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t output_rows = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t num_outputs = 0;
    uint32_t features = 0;
};

static_assert(sizeof(McuOutputGatherBf16Invocation) == 40);
static_assert(alignof(McuOutputGatherBf16Invocation) == 8);static_assert(offsetof(McuOutputGatherBf16Invocation, input) == 0);
static_assert(offsetof(McuOutputGatherBf16Invocation, output) == 8);
static_assert(offsetof(McuOutputGatherBf16Invocation, output_rows) == 16);
static_assert(offsetof(McuOutputGatherBf16Invocation, input_row_stride) == 24);
static_assert(offsetof(McuOutputGatherBf16Invocation, output_row_stride) == 28);
static_assert(offsetof(McuOutputGatherBf16Invocation, num_outputs) == 32);
static_assert(offsetof(McuOutputGatherBf16Invocation, features) == 36);

struct McuVerifyAcceptBatchInvocation {
    uint64_t context = 0;
    uint64_t sampled_tokens = 0;
    uint64_t committed_counts = 0;
    uint32_t max_candidates = 0;
    uint32_t reserved = 0;
};

static_assert(sizeof(McuVerifyAcceptBatchInvocation) == 32);
static_assert(alignof(McuVerifyAcceptBatchInvocation) == 8);
static_assert(offsetof(McuVerifyAcceptBatchInvocation, context) == 0);
static_assert(offsetof(McuVerifyAcceptBatchInvocation, sampled_tokens) == 8);
static_assert(offsetof(McuVerifyAcceptBatchInvocation, committed_counts) == 16);
static_assert(offsetof(McuVerifyAcceptBatchInvocation, max_candidates) == 24);

struct McuGdnSpecRestoreFromCountsInvocation {
    uint64_t context = 0;
    uint64_t committed_counts = 0;
    uint64_t conv_history = 0;
    uint64_t conv_pool = 0;
    uint64_t recurrent_history = 0;
    uint64_t recurrent_pool = 0;
    uint64_t conv_slot_stride = 0;
    uint64_t recurrent_slot_stride = 0;
    uint64_t conv_history_stride = 0;
    uint64_t recurrent_history_stride = 0;
    uint32_t conv_elems = 0;
    uint32_t recurrent_elems = 0;
    uint32_t max_candidates = 0;
    uint32_t reserved = 0;
};

static_assert(sizeof(McuGdnSpecRestoreFromCountsInvocation) == 96);
static_assert(alignof(McuGdnSpecRestoreFromCountsInvocation) == 8);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation, context) == 0);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       committed_counts) == 8);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation, conv_history) ==
              16);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation, conv_pool) == 24);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       recurrent_history) == 32);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       recurrent_pool) == 40);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       conv_slot_stride) == 48);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       recurrent_slot_stride) == 56);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       conv_history_stride) == 64);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       recurrent_history_stride) == 72);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation, conv_elems) == 80);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       recurrent_elems) == 84);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation,
                       max_candidates) == 88);
static_assert(offsetof(McuGdnSpecRestoreFromCountsInvocation, reserved) == 92);

struct McuGdnConv1dInvocation {
    uint64_t input = 0;
    uint64_t output = 0;
    uint64_t conv_weight = 0;
    uint64_t conv_state = 0;
    uint64_t requests = 0;
    uint32_t actual_rows = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t conv_dim = 0;
    uint32_t state_index = 0;
    uint32_t max_sequences = 0;
    uint64_t conv_slot_stride = 0;
    uint64_t conv_layer_stride = 0;
    uint64_t conv_history_stride = 0;
    uint32_t row_tile = 0;
    uint64_t conv_history_store = 0;
    uint64_t conv_history_row_stride = 0;
    uint32_t capture_rows = 0;
    uint32_t capture_verify_only = 0;
};

static_assert(sizeof(McuGdnConv1dInvocation) == 120);
static_assert(alignof(McuGdnConv1dInvocation) == 8);
static_assert(offsetof(McuGdnConv1dInvocation, input) == 0);
static_assert(offsetof(McuGdnConv1dInvocation, output) == 8);
static_assert(offsetof(McuGdnConv1dInvocation, conv_weight) == 16);
static_assert(offsetof(McuGdnConv1dInvocation, conv_state) == 24);
static_assert(offsetof(McuGdnConv1dInvocation, requests) == 32);
static_assert(offsetof(McuGdnConv1dInvocation, actual_rows) == 40);
static_assert(offsetof(McuGdnConv1dInvocation, input_row_stride) == 44);
static_assert(offsetof(McuGdnConv1dInvocation, output_row_stride) == 48);
static_assert(offsetof(McuGdnConv1dInvocation, conv_dim) == 52);
static_assert(offsetof(McuGdnConv1dInvocation, state_index) == 56);
static_assert(offsetof(McuGdnConv1dInvocation, max_sequences) == 60);
static_assert(offsetof(McuGdnConv1dInvocation, conv_slot_stride) == 64);
static_assert(offsetof(McuGdnConv1dInvocation, conv_layer_stride) == 72);
static_assert(offsetof(McuGdnConv1dInvocation, conv_history_stride) == 80);
static_assert(offsetof(McuGdnConv1dInvocation, row_tile) == 88);
static_assert(offsetof(McuGdnConv1dInvocation, conv_history_store) == 96);
static_assert(offsetof(McuGdnConv1dInvocation, conv_history_row_stride) == 104);
static_assert(offsetof(McuGdnConv1dInvocation, capture_rows) == 112);
static_assert(offsetof(McuGdnConv1dInvocation, capture_verify_only) == 116);

struct McuGdnRecurrenceInvocation {
    uint64_t words[23];
    uint32_t capture_verify_only = 0;
    uint32_t reserved = 0;
    uint64_t compact_delta = 0;
    uint64_t compact_k = 0;
    uint64_t compact_a = 0;
    uint64_t compact_delta_layer_stride = 0;
    uint64_t compact_k_layer_stride = 0;
    uint64_t compact_a_layer_stride = 0;
    uint32_t compact_rows = 0;
    uint32_t reserved_tail = 0;
};

static_assert(sizeof(McuGdnRecurrenceInvocation) == 248);
static_assert(alignof(McuGdnRecurrenceInvocation) == 8);
static_assert(offsetof(McuGdnRecurrenceInvocation, capture_verify_only) == 184);
static_assert(offsetof(McuGdnRecurrenceInvocation, compact_delta) == 192);
static_assert(offsetof(McuGdnRecurrenceInvocation, compact_rows) == 240);

struct McuPlanNode {
    uint16_t variant_id = 0;
    uint16_t next = kMcuNoNext;
    uint16_t kernarg_recipe = kMcuKernargRecipeProbe;
    uint16_t completion_slot = 0;
    uint32_t value = 0;
    uint16_t element_count = 0;
    uint16_t input_slot = kMcuNoInput;
    uint32_t work = 0;
    uint32_t flags = 0;
    uint32_t invocation_index = 0;
};

static_assert(sizeof(McuPlanNode) == 28);

struct alignas(32) McuDynamicNodeBinding {
    uint16_t enabled = 0;
    uint16_t variant_id = 0;
    uint32_t workgroup_count_x = 0;
    uint32_t workgroup_count_y = 0;
    uint32_t workgroup_count_z = 0;
    uint32_t invocation_index = 0;
    uint32_t variant_override = 0xFFFFFFFFu;
    uint32_t reserved1 = 0;
};

static_assert(sizeof(McuDynamicNodeBinding) == 32);
static_assert(alignof(McuDynamicNodeBinding) == 32);
static_assert(offsetof(McuDynamicNodeBinding, enabled) == 0);
static_assert(offsetof(McuDynamicNodeBinding, variant_id) == 2);
static_assert(offsetof(McuDynamicNodeBinding, workgroup_count_x) == 4);
static_assert(offsetof(McuDynamicNodeBinding, workgroup_count_y) == 8);
static_assert(offsetof(McuDynamicNodeBinding, workgroup_count_z) == 12);
static_assert(offsetof(McuDynamicNodeBinding, invocation_index) == 16);
static_assert(offsetof(McuDynamicNodeBinding, variant_override) == 20);

enum class McuInvocationPatchSource : uint8_t {
    ActualRows = 0,
    NumRequests = 1,
    NumOutputs = 2,
    AttentionRegionRows = 3,
    VerifyRequests = 4,
    Bf16ExactRowsVariant = 5,
};

struct alignas(16) McuInvocationPatch {
    uint64_t target = 0;
    uint8_t source = 0;
    uint8_t reserved[3] = {};
    uint32_t param = 0;
};

static_assert(sizeof(McuInvocationPatch) == 16);
static_assert(offsetof(McuInvocationPatch, param) == 12);
static_assert(alignof(McuInvocationPatch) == 16);
static_assert(offsetof(McuInvocationPatch, target) == 0);
static_assert(offsetof(McuInvocationPatch, source) == 8);

struct McuKernelVariantDesc {
    uint16_t variant_id = 0;
    uint16_t kernarg_recipe = kMcuKernargRecipeProbe;
    uint32_t kernarg_size = 0;
    uint32_t workgroup_count_x = 1;
    uint32_t workgroup_count_y = 1;
    uint32_t workgroup_count_z = 1;
    uint32_t workgroup_x = 1;
    uint32_t workgroup_y = 1;
    uint32_t workgroup_z = 1;
    uint32_t hidden_args_policy = 1;
    GpuMcuRetainedPacket packet{};
};

static_assert(alignof(McuKernelVariantDesc) == 16);

struct alignas(64) McuDispatchTiming {
    uint64_t node_fetch_start = 0;
    uint64_t kernarg_build_end = 0;
    uint64_t packet_stage_end = 0;
    uint64_t dependency_ready = 0;
    uint64_t commit_start = 0;
    uint64_t dw0_publish = 0;
    uint64_t doorbell = 0;
    uint64_t reserved = 0;
};

static_assert(sizeof(McuDispatchTiming) == 64);

struct GpuMcuFsmRunContext {
    uint64_t iterations = 0;
    uint64_t heartbeat = 0;
    uint64_t plans_started = 0;
    uint64_t dispatches_committed = 0;
    uint64_t completions_observed = 0;
    uint64_t run_consumed = 0;
    uint64_t write_base = 0;
    uint64_t dispatch_seq = 0;
    uint64_t append_ahead_total = 0;
    uint64_t queue_empty_total = 0;
    uint64_t queue_full_total = 0;
    uint64_t refill_total = 0;
    uint64_t doorbell_total = 0;
    uint64_t max_ahead_total = 0;
    uint64_t wrap_total = 0;
    uint32_t write_base_valid = 0;
    uint32_t region_seq = 0;
    uint32_t supervisor = static_cast<uint32_t>(McuSupervisorState::idle);
};

struct alignas(64) GpuMcuFsmState {
    uint32_t stop_requested = 0;
    uint32_t started = 0;
    uint32_t supervisor = 0;
    uint32_t reserved_ctrl = 0;
    uint64_t run_request = 0;
    uint64_t heartbeat = 0;
    uint64_t iterations = 0;
    uint64_t plans_started = 0;
    uint64_t dispatches_committed = 0;
    uint64_t completions_observed = 0;
    uint32_t fault_code = 0;
    uint32_t fault_pc = 0;
    uint32_t fault_variant = 0;
    uint32_t fault_generation = 0;

    DeviceAqlQueueView queue{};
    const McuPlanNode* plan = nullptr;
    uint32_t node_count = 0;
    uint32_t plan_id = 0;
    const McuKernelVariantDesc* variants = nullptr;
    uint32_t variant_count = 0;
    const McuDynamicNodeBinding* dynamic_node_bindings = nullptr;
    uint32_t dynamic_node_binding_count = 0;
    uint32_t rmsnorm_invocation_count = 0;
    const McuRmsNormInvocation* rmsnorm_invocations = nullptr;
    uint32_t quantize_invocation_count = 0;
    const McuActivationQuantizeInvocation* quantize_invocations = nullptr;
    uint32_t e4m3_invocation_count = 0;
    const McuActivationQuantizeE4m3Invocation* e4m3_invocations = nullptr;
    uint32_t psq4_invocation_count = 0;
    const McuPsq4Decode1Invocation* psq4_invocations = nullptr;
    uint32_t psq4_multi_invocation_count = 0;
    const McuPsq4MultiRowInvocation* psq4_multi_invocations = nullptr;
    uint32_t verify_accept_invocation_count = 0;
    const McuVerifyAcceptPrefixInvocation* verify_accept_invocations = nullptr;
    uint32_t verify_accept_batch_invocation_count = 0;
    const McuVerifyAcceptBatchInvocation* verify_accept_batch_invocations =
        nullptr;
    uint32_t gdn_spec_restore_invocation_count = 0;
    const McuGdnSpecRestoreInvocation* gdn_spec_restore_invocations = nullptr;
    uint32_t gdn_spec_restore_from_counts_invocation_count = 0;
    const McuGdnSpecRestoreFromCountsInvocation*
        gdn_spec_restore_from_counts_invocations = nullptr;
    uint32_t argmax_f32_invocation_count = 0;
    const McuArgmaxF32Invocation* argmax_f32_invocations = nullptr;
    uint32_t elementwise_invocation_count = 0;
    const McuElementwiseInvocation* elementwise_invocations = nullptr;
    uint32_t rope_invocation_count = 0;
    const McuRopeInvocation* rope_invocations = nullptr;
    uint32_t kv_append_invocation_count = 0;
    const McuKvAppendInvocation* kv_append_invocations = nullptr;
    uint32_t attention_paged_invocation_count = 0;
    const McuPagedAttentionInvocation* attention_paged_invocations = nullptr;
    uint32_t attention_paged_split_invocation_count = 0;
    const McuPagedAttentionSplitInvocation*
        attention_paged_split_invocations = nullptr;
    uint32_t attention_paged_reduce_invocation_count = 0;
    const McuPagedAttentionReduceInvocation*
        attention_paged_reduce_invocations = nullptr;
    uint32_t bf16_invocation_count = 0;
    const McuBf16ExactRowsInvocation* bf16_invocations = nullptr;
    uint32_t l2_invocation_count = 0;
    const McuL2NormalizeInvocation* l2_invocations = nullptr;
    uint32_t embedding_invocation_count = 0;
    const McuEmbeddingBf16Invocation* embedding_invocations = nullptr;
    uint32_t output_gather_invocation_count = 0;
    const McuOutputGatherBf16Invocation* output_gather_invocations = nullptr;
    uint32_t gdn_conv1d_invocation_count = 0;
    const McuGdnConv1dInvocation* gdn_conv1d_invocations = nullptr;
    uint32_t gdn_recurrence_invocation_count = 0;
    const McuGdnRecurrenceInvocation* gdn_recurrence_invocations = nullptr;
    uint32_t gdn_reset_invocation_count = 0;
    const GpuMcuGdnResetInvocation* gdn_reset_invocations = nullptr;
    uint64_t kernarg_base = 0;
    uint64_t kernarg_slot_stride = 0;
    uint32_t kernarg_slot_count = 0;
    uint32_t reserved1 = 0;
    GpuMcuDeviceCompletion* completions = nullptr;
    uint32_t completion_count = 0;
    uint32_t reserved2 = 0;
    uint64_t output_base = 0;
    uint64_t timestamps_base = 0;
    GpuMcuRetainedPacket* retained_vram = nullptr;
    uint32_t retained_count = 0;
    uint32_t template_source = 0;
    uint32_t copy_lanes = 1;
    uint32_t lds_budget_bytes = 0;
    uint32_t barrier = 1;
    uint32_t prepared_enabled = 0;
    uint32_t idle_sleep = 8;
    uint32_t reserved3 = 0;
    McuDispatchTiming* timing = nullptr;
    uint32_t timing_count = 0;
    uint32_t queue_ahead_depth = 0;
    uint32_t doorbell_batch = 1;
    uint32_t doorbell_mode = 0;
    uint32_t issue_wait_running = 0;
    McuDispatchRecord* debug_records = nullptr;
    uint32_t debug_record_count = 0;
    uint32_t reserved5 = 0;
    uint64_t region_start_ts = 0;
    uint64_t region_end_ts = 0;
    uint64_t append_ahead_count = 0;
    uint64_t queue_empty_count = 0;
    uint64_t mid_region_queue_empty_count = 0;
    uint64_t backpressure_wait_count = 0;
    uint64_t refill_count = 0;
    uint64_t doorbell_count = 0;
    uint64_t max_ahead = 0;
    uint64_t wrap_count = 0;
    uint32_t external_epoch_seen = 0;
    uint32_t* external_start_signal = nullptr;
    uint32_t* external_done_signal = nullptr;
    uint32_t* external_result_code = nullptr;

    uint64_t batch_input = 0;

    GpuMcuFsmRunContext run_ctx{};
};

struct GpuMcuFsmConfig {
    DeviceAqlQueueView queue{};
    const McuPlanNode* plan = nullptr;
    uint32_t node_count = 0;
    uint32_t plan_id = 0;
    const McuKernelVariantDesc* variants = nullptr;
    uint32_t variant_count = 0;
    const McuDynamicNodeBinding* dynamic_node_bindings = nullptr;
    uint32_t dynamic_node_binding_count = 0;
    uint32_t rmsnorm_invocation_count = 0;
    const McuRmsNormInvocation* rmsnorm_invocations = nullptr;
    uint32_t quantize_invocation_count = 0;
    const McuActivationQuantizeInvocation* quantize_invocations = nullptr;
    uint32_t e4m3_invocation_count = 0;
    const McuActivationQuantizeE4m3Invocation* e4m3_invocations = nullptr;
    uint32_t psq4_invocation_count = 0;
    const McuPsq4Decode1Invocation* psq4_invocations = nullptr;
    uint32_t psq4_multi_invocation_count = 0;
    const McuPsq4MultiRowInvocation* psq4_multi_invocations = nullptr;
    uint32_t verify_accept_invocation_count = 0;
    const McuVerifyAcceptPrefixInvocation* verify_accept_invocations = nullptr;
    uint32_t verify_accept_batch_invocation_count = 0;
    const McuVerifyAcceptBatchInvocation* verify_accept_batch_invocations =
        nullptr;
    uint32_t gdn_spec_restore_invocation_count = 0;
    const McuGdnSpecRestoreInvocation* gdn_spec_restore_invocations = nullptr;
    uint32_t gdn_spec_restore_from_counts_invocation_count = 0;
    const McuGdnSpecRestoreFromCountsInvocation*
        gdn_spec_restore_from_counts_invocations = nullptr;
    uint32_t argmax_f32_invocation_count = 0;
    const McuArgmaxF32Invocation* argmax_f32_invocations = nullptr;
    uint32_t elementwise_invocation_count = 0;
    const McuElementwiseInvocation* elementwise_invocations = nullptr;
    uint32_t rope_invocation_count = 0;
    const McuRopeInvocation* rope_invocations = nullptr;
    uint32_t kv_append_invocation_count = 0;
    const McuKvAppendInvocation* kv_append_invocations = nullptr;
    uint32_t attention_paged_invocation_count = 0;
    const McuPagedAttentionInvocation* attention_paged_invocations = nullptr;
    uint32_t attention_paged_split_invocation_count = 0;
    const McuPagedAttentionSplitInvocation*
        attention_paged_split_invocations = nullptr;
    uint32_t attention_paged_reduce_invocation_count = 0;
    const McuPagedAttentionReduceInvocation*
        attention_paged_reduce_invocations = nullptr;
    uint32_t bf16_invocation_count = 0;
    const McuBf16ExactRowsInvocation* bf16_invocations = nullptr;
    uint32_t l2_invocation_count = 0;
    const McuL2NormalizeInvocation* l2_invocations = nullptr;
    uint32_t embedding_invocation_count = 0;
    const McuEmbeddingBf16Invocation* embedding_invocations = nullptr;
    uint32_t output_gather_invocation_count = 0;
    const McuOutputGatherBf16Invocation* output_gather_invocations = nullptr;
    uint32_t gdn_conv1d_invocation_count = 0;
    const McuGdnConv1dInvocation* gdn_conv1d_invocations = nullptr;
    uint32_t gdn_recurrence_invocation_count = 0;
    const McuGdnRecurrenceInvocation* gdn_recurrence_invocations = nullptr;
    uint32_t gdn_reset_invocation_count = 0;
    const GpuMcuGdnResetInvocation* gdn_reset_invocations = nullptr;
    const GpuMcuRetainedPacket* retained = nullptr;
    uint32_t retained_count = 0;
    uint64_t kernarg_base = 0;
    uint64_t kernarg_slot_stride = 0;
    uint32_t kernarg_slot_count = 0;
    GpuMcuDeviceCompletion* completions = nullptr;
    uint32_t completion_count = 0;
    uint64_t output_base = 0;
    uint64_t timestamps_base = 0;
    McuDispatchTiming* timing = nullptr;
    uint32_t timing_count = 0;
    uint32_t lds_budget_bytes = 0;
    uint32_t template_source = 0;
    uint32_t copy_lanes = 1;
    uint32_t barrier = 1;
    uint32_t prepared_enabled = 0;
    uint32_t idle_sleep = 8;
    uint32_t queue_ahead_depth = 0;
    uint32_t doorbell_batch = 1;
    uint32_t doorbell_mode = 0;
    uint32_t issue_wait_running = 0;
    McuDispatchRecord* debug_records = nullptr;
    uint32_t debug_record_count = 0;
    uint32_t* start_signal = nullptr;
    uint32_t* done_signal = nullptr;
    uint32_t* result_code = nullptr;
};

constexpr AqlMemoryPolicy kMcuAgentAqlPolicy{
    .acquire_scope = AqlFenceScope::Agent,
    .release_scope = AqlFenceScope::Agent,
    .producer_visibility_fence = true,
};

constexpr AqlMemoryPolicy kMcuSystemAqlPolicy{
    .acquire_scope = AqlFenceScope::System,
    .release_scope = AqlFenceScope::System,
    .producer_visibility_fence = true,
};

McuKernelVariantDesc make_kernel_variant(
    uint16_t variant_id,
    const GpuAqlKernelMetadata& meta,
    uint16_t kernarg_recipe,
    uint32_t workgroup_x,
    uint32_t workgroup_count_x,
    const AqlMemoryPolicy& policy,
    AqlHiddenArgsPolicy hidden_policy = AqlHiddenArgsPolicy::IfFits,
    uint32_t workgroup_count_y = 1u,
    uint32_t workgroup_count_z = 1u);

McuKernelVariantDesc make_probe_variant(
    uint16_t variant_id,
    const GpuAqlKernelMetadata& meta,
    uint32_t workgroup_x,
    uint32_t workgroup_count_x,
    const AqlMemoryPolicy& policy = kMcuAgentAqlPolicy,
    AqlHiddenArgsPolicy hidden_policy = AqlHiddenArgsPolicy::IfFits);

class GpuMcuFsm {
public:
    GpuMcuFsm() = default;
    ~GpuMcuFsm() noexcept { (void)shutdown(); }

    GpuMcuFsm(const GpuMcuFsm&) = delete;
    GpuMcuFsm& operator=(const GpuMcuFsm&) = delete;

    GpuMcuFsm(GpuMcuFsm&& other) noexcept;
    GpuMcuFsm& operator=(GpuMcuFsm&& other) noexcept;

    static Result<GpuMcuFsm> create(int device, hipStream_t control_stream);

    bool valid() const noexcept { return host_state_ != nullptr; }

    Status configure(const GpuMcuFsmConfig& config);
    Status start();

    Status request_run();
    Status wait_plans(uint64_t count, uint32_t timeout_ms) const;
    bool running() const noexcept;
    McuSupervisorState supervisor() const noexcept;
    McuFaultCode fault_code() const noexcept;
    uint32_t fault_pc() const noexcept;
    uint64_t plans_started() const noexcept;
    uint64_t dispatches() const noexcept;
    uint64_t completions_observed() const noexcept;
    uint64_t heartbeat() const noexcept;
    uint64_t append_ahead_count() const noexcept;
    uint64_t queue_empty_count() const noexcept;
    uint64_t mid_region_queue_empty_count() const noexcept;
    uint64_t backpressure_wait_count() const noexcept;
    uint64_t refill_count() const noexcept;
    uint64_t doorbell_count() const noexcept;
    uint64_t max_ahead() const noexcept;
    uint64_t wrap_count() const noexcept;
    uint32_t external_epoch_seen() const noexcept;
    uint64_t region_start_ts() const noexcept;
    uint64_t region_end_ts() const noexcept;

    Status request_stop();
    Status wait_stopped(uint32_t timeout_ms);
    Status shutdown() noexcept;

    GpuMcuFsmState* device_state() const noexcept { return device_state_; }
    GpuMcuFsmState* host_state() const noexcept { return host_state_; }

private:
    void move_from(GpuMcuFsm& other) noexcept;

    int device_ = -1;
    void* host_allocation_ = nullptr;
    GpuMcuFsmState* host_state_ = nullptr;
    GpuMcuFsmState* device_state_ = nullptr;
    hipStream_t control_stream_ = nullptr;
    uint32_t lds_bytes_ = 0;
    uint64_t requested_runs_ = 0;
    bool launched_ = false;
};

}  // namespace ps::runtime::gpu_mcu
