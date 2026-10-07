#pragma once

#include <cstddef>
#include <cstdint>

namespace ps::runtime::gpu_mcu {

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
    uint64_t output_rows = 0;
};

static_assert(sizeof(McuArgmaxF32Invocation) == 48);
static_assert(alignof(McuArgmaxF32Invocation) == 8);
static_assert(offsetof(McuArgmaxF32Invocation, logits) == 0);
static_assert(offsetof(McuArgmaxF32Invocation, sampling) == 8);
static_assert(offsetof(McuArgmaxF32Invocation, sampled_tokens) == 16);
static_assert(offsetof(McuArgmaxF32Invocation, outputs) == 24);
static_assert(offsetof(McuArgmaxF32Invocation, vocab_size) == 28);
static_assert(offsetof(McuArgmaxF32Invocation, logits_row_stride) == 32);
static_assert(offsetof(McuArgmaxF32Invocation, output_rows) == 40);

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

struct McuBf16WmmaInvocation {
    uint64_t weight = 0;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t output_dtype = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(McuBf16WmmaInvocation) == 48);
static_assert(alignof(McuBf16WmmaInvocation) == 8);
static_assert(offsetof(McuBf16WmmaInvocation, weight) == 0);
static_assert(offsetof(McuBf16WmmaInvocation, input) == 8);
static_assert(offsetof(McuBf16WmmaInvocation, output) == 16);
static_assert(offsetof(McuBf16WmmaInvocation, output_dtype) == 24);
static_assert(offsetof(McuBf16WmmaInvocation, rows) == 28);
static_assert(offsetof(McuBf16WmmaInvocation, out_features) == 32);
static_assert(offsetof(McuBf16WmmaInvocation, k) == 36);
static_assert(offsetof(McuBf16WmmaInvocation, input_row_stride) == 40);
static_assert(offsetof(McuBf16WmmaInvocation, output_row_stride) == 44);

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

struct McuEmbeddingPsq8Invocation {
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint64_t token_ids = 0;
    uint64_t output = 0;
    uint64_t error_word = 0;
    uint32_t codes_row_stride_bytes = 0;
    uint32_t scale_row_stride_bytes = 0;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(McuEmbeddingPsq8Invocation) == 64);
static_assert(alignof(McuEmbeddingPsq8Invocation) == 8);
static_assert(offsetof(McuEmbeddingPsq8Invocation, codes) == 0);
static_assert(offsetof(McuEmbeddingPsq8Invocation, scales) == 8);
static_assert(offsetof(McuEmbeddingPsq8Invocation, token_ids) == 16);
static_assert(offsetof(McuEmbeddingPsq8Invocation, output) == 24);
static_assert(offsetof(McuEmbeddingPsq8Invocation, error_word) == 32);
static_assert(offsetof(McuEmbeddingPsq8Invocation, codes_row_stride_bytes) == 40);
static_assert(offsetof(McuEmbeddingPsq8Invocation, scale_row_stride_bytes) == 44);
static_assert(offsetof(McuEmbeddingPsq8Invocation, rows) == 48);
static_assert(offsetof(McuEmbeddingPsq8Invocation, vocab_size) == 52);
static_assert(offsetof(McuEmbeddingPsq8Invocation, hidden_size) == 56);
static_assert(offsetof(McuEmbeddingPsq8Invocation, output_row_stride) == 60);

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

}  // namespace ps::runtime::gpu_mcu
