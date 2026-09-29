#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_model.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduled_batch.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <phaseshift/models/qwen35/state/paged_sequence_state.h>
#include <phaseshift/models/qwen35/state/sequence_slot_pool.h>
#include <phaseshift/runtime/program/comm_launcher.h>
#include <hip/hip_runtime.h>

#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace ps::test {

struct SyntheticRunOptions {
    std::string model_dir;
    int device = 0;
    ::ps::qwen35::ModelPartition partition{};
    uint32_t pipeline_peer_rank = 0;
    void* comm_self = nullptr;
    ::ps::runtime::CommLaunchFn comm_launch = nullptr;
    uint32_t prefill_tokens = 16;
    uint32_t decode_steps = 4;
    bool sample = true;
    std::vector<uint32_t> hidden_taps;
    std::vector<int32_t>* shared_tokens = nullptr;
    std::barrier<>* step_barrier = nullptr;
    bool read_shared_tokens = false;
    bool publish_shared_tokens = false;
    uint32_t tensor_parallel_size = 1;
    uint32_t tensor_parallel_rank = 0;
    bool tp_full_attention = true;
    bool tp_linear_attention = false;
    bool tp_mlp = false;
    ::ps::qwen35::Qwen35TensorShard tensor_shard;
    uint64_t arena_bytes = 512ull * 1024ull * 1024ull;
};

struct SyntheticRunResult {
    bool ok = false;
    std::string error;
    std::vector<int32_t> tokens;
    std::vector<float> logits;
    std::vector<std::uint16_t> hidden;
    std::vector<std::vector<std::uint16_t>> tap_boundary;
    std::vector<uint32_t> positions;
    std::vector<uint32_t> block_counts;
    std::size_t comm_count = 0;
    std::vector<uint64_t> step_us;
    std::size_t weight_bytes = 0;
    bool unowned_layers_empty = false;
};

class SyntheticRunner {
public:
    static SyntheticRunResult run(const SyntheticRunOptions& options) {
        SyntheticRunResult result;
        if (hipSetDevice(options.device) != hipSuccess) {
            result.error = "hipSetDevice failed";
            return result;
        }
        SyntheticRunner runner;
        if (hipStreamCreate(&runner.stream_)) {
            result.error = "hipStreamCreate failed";
            return result;
        }
        const int total_steps = static_cast<int>(options.decode_steps) + 1;
        int completed = 0;
        if (runner.initialize(options, result)) {
            runner.execute(options, result, &completed);
            runner.record_comm_count(result);
        }
        runner.cleanup();
        for (int i = completed; i < total_steps; ++i) arrive(options);
        if (result.error.empty() && !runner.error_.empty()) result.error = runner.error_;
        result.ok = result.error.empty();
        return result;
    }

private:
    SyntheticRunner() = default;

    ~SyntheticRunner() { cleanup(); }

    static void arrive(const SyntheticRunOptions& options) {
        if (options.step_barrier != nullptr) options.step_barrier->arrive_and_wait();
    }

    static std::size_t matrix_weight_bytes(const ps::weights::MatrixWeight& w) {
        return w.bf16.allocation_bytes() + w.data.allocation_bytes() +
               w.codes.allocation_bytes() + w.scales.allocation_bytes() +
               w.compute_codes.allocation_bytes() + w.compute_scales_bf16.allocation_bytes();
    }

    static std::size_t layer_weight_bytes(const ::ps::qwen35::Qwen35LayerWeights& lw) {
        return lw.input_layernorm_weight.allocation_bytes() +
               lw.post_attention_layernorm_weight.allocation_bytes() +
               matrix_weight_bytes(lw.mlp_gate_proj) + matrix_weight_bytes(lw.mlp_up_proj) +
               matrix_weight_bytes(lw.mlp_down_proj) + matrix_weight_bytes(lw.attn_q_proj) +
               matrix_weight_bytes(lw.attn_k_proj) + matrix_weight_bytes(lw.attn_v_proj) +
               matrix_weight_bytes(lw.attn_o_proj) + lw.attn_q_norm_weight.allocation_bytes() +
               lw.attn_k_norm_weight.allocation_bytes() +
               matrix_weight_bytes(lw.attn_in_proj_a) + matrix_weight_bytes(lw.attn_in_proj_b) +
               matrix_weight_bytes(lw.attn_in_proj_qkv) + matrix_weight_bytes(lw.attn_in_proj_z) +
               lw.attn_conv1d_weight.allocation_bytes() +
               matrix_weight_bytes(lw.attn_out_proj) + lw.attn_norm_weight.allocation_bytes() +
               lw.attn_dt_bias.allocation_bytes() + lw.attn_A_log.allocation_bytes();
    }

    static std::size_t model_weight_bytes(const ::ps::qwen35::Qwen35ModelWeights& mw) {
        std::size_t bytes = matrix_weight_bytes(mw.embed_tokens) +
                            mw.final_norm_weight.allocation_bytes();
        if (!mw.lm_head_tied) bytes += matrix_weight_bytes(mw.lm_head);
        for (const auto& lw : mw.layers) bytes += layer_weight_bytes(lw);
        if (mw.mtp.present) {
            bytes += mw.mtp.pre_fc_norm_embedding_weight.allocation_bytes() +
                     mw.mtp.pre_fc_norm_hidden_weight.allocation_bytes() +
                     mw.mtp.final_norm_weight.allocation_bytes() +
                     matrix_weight_bytes(mw.mtp.fc) + layer_weight_bytes(mw.mtp.layer);
        }
        return bytes;
    }

    bool initialize(const SyntheticRunOptions& options, SyntheticRunResult& result) {
        auto arena_result = ::ps::gpu::GpuArena::create(options.device, options.arena_bytes);
        if (!arena_result.ok()) {
            result.error = message("GpuArena::create", arena_result.status());
            return false;
        }
        arena_ = std::make_unique<::ps::gpu::GpuArena>(arena_result.release());

        ::ps::qwen35::Qwen35LoadOptions load_options;
        load_options.tensor_shard = options.tensor_shard;
        load_options.partition = options.partition;
        auto model_result = ::ps::qwen35::Qwen35Model::load_from_safetensors(
            options.model_dir, *arena_, stream_, load_options);
        if (!model_result.ok()) {
            result.error = message("load model", model_result.status());
            return false;
        }
        model_ = std::make_unique<::ps::qwen35::Qwen35Model>(model_result.release());

        result.weight_bytes = model_weight_bytes(model_->weights());

        const auto& tc = model_->text_config();
        {
            const auto partition = ::ps::qwen35::resolve_model_partition(
                options.partition, static_cast<uint32_t>(tc.num_hidden_layers));
            result.unowned_layers_empty = true;
            const auto& layers = model_->weights().layers;
            for (std::size_t l = 0; l < layers.size(); ++l) {
                if (l >= partition.layer_begin && l < partition.layer_end) continue;
                const auto& lw = layers[l];
                if (lw.attn_q_proj.rows != 0 || lw.attn_k_proj.rows != 0 ||
                    lw.attn_v_proj.rows != 0 || lw.attn_o_proj.rows != 0 ||
                    lw.attn_in_proj_qkv.rows != 0 || lw.mlp_gate_proj.rows != 0 ||
                    lw.mlp_up_proj.rows != 0 || lw.mlp_down_proj.rows != 0) {
                    result.unowned_layers_empty = false;
                }
            }
        }
        max_blocks_ = (max_seq_len_ + page_tokens_ - 1) / page_tokens_;

        auto slot_result =
            ::ps::qwen35::SequenceSlotPool::create(*arena_, 1, max_blocks_);
        if (!slot_result.ok()) {
            result.error = message("SequenceSlotPool::create", slot_result.status());
            return false;
        }
        slot_pool_ = std::make_unique<::ps::qwen35::SequenceSlotPool>(slot_result.release());

        const uint32_t gdn_tp =
            options.tp_linear_attention ? options.tensor_parallel_size : 1u;
        const auto partition = ::ps::qwen35::resolve_model_partition(
            options.partition, static_cast<uint32_t>(tc.layer_types.size()));
        auto gdn_result = ::ps::qwen35::GdnStatePool::create(
            *arena_, 1,
            ::ps::qwen35::GdnStatePoolLayout::from_text_config(
                tc, gdn_tp, partition.layer_begin, partition.layer_end));
        if (!gdn_result.ok()) {
            result.error = message("GdnStatePool::create", gdn_result.status());
            return false;
        }
        gdn_pool_ = std::make_unique<::ps::qwen35::GdnStatePool>(gdn_result.release());

        uint32_t attn_layers = 0;
        for (uint32_t i = partition.layer_begin;
             i < partition.layer_end && i < tc.layer_types.size(); ++i)
            if (tc.layer_types[i] == 1) ++attn_layers;
        auto kv_result = ::ps::qwen35::PagedKVPool::create(
            *arena_, max_blocks_ + 4, page_tokens_, attn_layers, tc.num_key_value_heads,
            tc.attention_head_dim, ::ps::qwen35::KVCacheDType::BF16);
        if (!kv_result.ok()) {
            result.error = message("PagedKVPool::create", kv_result.status());
            return false;
        }
        kv_pool_ = std::make_unique<::ps::qwen35::PagedKVPool>(kv_result.release());

        ::ps::qwen35::ExecutorConfig config;
        config.max_scheduled_tokens = max_tokens_;
        config.max_scheduled_requests = 1;
        config.max_scheduled_output_rows = max_tokens_;
        config.backend = ::ps::qwen35::runtime::DecodeBackend::Host;
        config.partition = options.partition;
        config.pipeline_peer_rank = options.pipeline_peer_rank;
        config.tensor_parallel_size = options.tensor_parallel_size;
        config.tensor_parallel_rank = options.tensor_parallel_rank;
        config.tp_full_attention = options.tp_full_attention;
        config.tp_linear_attention = options.tp_linear_attention;
        config.tp_mlp = options.tp_mlp;
        for (uint32_t i = 0; i < options.hidden_taps.size() && i < max_taps; ++i)
            config.target_hidden_taps[i] = options.hidden_taps[i];
        config.target_hidden_tap_count =
            static_cast<uint32_t>(options.hidden_taps.size() < max_taps
                                      ? options.hidden_taps.size()
                                      : max_taps);
        auto executor_result = ::ps::qwen35::create_model_executor(
            *model_, *slot_pool_, *gdn_pool_, *kv_pool_, *arena_, config);
        if (!executor_result.ok()) {
            result.error = message("create_model_executor", executor_result.status());
            return false;
        }
        executor_ = std::make_unique<::ps::qwen35::Executor>(executor_result.release());
        executor_->comm_self = options.comm_self;
        executor_->comm_launch = options.comm_launch;

        auto sequence_result = ::ps::qwen35::create_paged_sequence_state(
            *slot_pool_, *gdn_pool_, *kv_pool_, max_seq_len_, stream_);
        if (!sequence_result.ok()) {
            result.error = message("create_paged_sequence_state", sequence_result.status());
            return false;
        }
        sequence_ = std::make_unique<::ps::qwen35::PagedSequenceState>(
            sequence_result.release());
        return true;
    }

    void record_comm_count(SyntheticRunResult& result) {
        if (executor_ == nullptr) return;
        result.comm_count = executor_->program_set.programs[0].comms.size();
    }

    void execute(const SyntheticRunOptions& options, SyntheticRunResult& result,
                 int* completed) {
        std::vector<int32_t> context(options.prefill_tokens);
        for (uint32_t i = 0; i < options.prefill_tokens; ++i)
            context[i] = static_cast<int32_t>((i * 7 + 3) % 64);

        int32_t sampled = 0;
        bool ok = run_step(context, options.prefill_tokens,
                           ::ps::runtime::ExecutionClass::PREFILL, options, result,
                           &sampled);
        finish_step(options, sampled, ok);
        ++*completed;
        if (!ok) return;

        for (uint32_t step = 0; step < options.decode_steps; ++step) {
            int32_t token = sampled;
            if (options.read_shared_tokens) {
                if (options.shared_tokens == nullptr ||
                    step >= options.shared_tokens->size()) {
                    result.error = "coordinator token was not produced";
                    return;
                }
                token = (*options.shared_tokens)[step];
            }
            std::vector<int32_t> token_ids = {token};
            ok = run_step(token_ids, 1, ::ps::runtime::ExecutionClass::DECODE, options,
                          result, &sampled);
            finish_step(options, sampled, ok);
            ++*completed;
            if (!ok) return;
        }
    }

    void finish_step(const SyntheticRunOptions& options, int32_t sampled, bool ok) {
        if (ok && options.publish_shared_tokens && options.shared_tokens != nullptr)
            options.shared_tokens->push_back(sampled);
        arrive(options);
    }

    bool run_step(const std::vector<int32_t>& token_ids, uint32_t num_tokens,
                  ::ps::runtime::ExecutionClass execution_class,
                  const SyntheticRunOptions& options, SyntheticRunResult& result,
                  int32_t* sampled) {
        const std::size_t step_index = result.positions.size();
        const auto step_begin = std::chrono::steady_clock::now();
        result.positions.push_back(sequence_->position);
        result.block_counts.push_back(
            static_cast<uint32_t>(sequence_->block_table.size()));
        std::printf("device=%d step=%zu class=%s begin position=%u\n", options.device,
                    step_index,
                    execution_class == ::ps::runtime::ExecutionClass::PREFILL
                        ? "PREFILL"
                        : "DECODE",
                    sequence_->position);
        ::ps::qwen35::ScheduledRequest req;
        req.sequence = sequence_.get();
        req.handle = sequence_->request_handle();
        req.execution_class = execution_class;
        req.token_begin = 0;
        req.num_tokens = num_tokens;
        req.prefix_tokens = sequence_->position;
        req.compute_logits = options.sample;
        req.sample = options.sample;
        req.num_output_rows = 1;
        std::vector<::ps::qwen35::ScheduledRequest> requests = {req};

        ::ps::qwen35::ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.requests = requests;
        batch.num_tokens = num_tokens;
        batch.num_requests = 1;
        if (execution_class == ::ps::runtime::ExecutionClass::PREFILL)
            batch.num_prefill_requests = 1;
        else
            batch.num_decode_requests = 1;

        auto executed = ::ps::qwen35::execute_batch(*executor_, batch, stream_);
        if (!executed.ok()) {
            result.error = message("execute_batch", executed.status());
            return false;
        }
        ::ps::qwen35::BatchExecutionOutput output = executed.release();
        if (!options.hidden_taps.empty())
            result.tap_boundary.push_back(capture_tap(num_tokens));
        if (output.num_outputs > 0) {
            const size_t vocab = model_->text_config().vocab_size;
            const size_t hidden = model_->text_config().hidden_size;
            std::vector<float> step_logits(output.num_outputs * vocab);
            if (hipMemcpy(step_logits.data(), output.logits.data<float>(),
                          step_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                result.error = "logits copy failed";
                return false;
            }
            result.logits.insert(result.logits.end(), step_logits.begin(),
                                 step_logits.end());
            std::vector<std::uint16_t> step_hidden(output.num_outputs * hidden);
            if (hipMemcpy(step_hidden.data(), output.final_hidden.data<void>(),
                          step_hidden.size() * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                result.error = "hidden copy failed";
                return false;
            }
            result.hidden.insert(result.hidden.end(), step_hidden.begin(),
                                 step_hidden.end());
            if (options.sample) {
                int32_t token = 0;
                if (hipMemcpy(&token, output.sampled_tokens.data<int32_t>(),
                              sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
                    result.error = "sample copy failed";
                    return false;
                }
                const size_t vocab = model_->text_config().vocab_size;
                if (token < 0 || static_cast<size_t>(token) >= vocab) {
                    result.error = "sampled token out of vocabulary range";
                    return false;
                }
                *sampled = token;
                result.tokens.push_back(token);
            }
        }
        result.step_us.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - step_begin)
                .count()));
        std::printf("device=%d step=%zu done\n", options.device, step_index);
        return true;
    }

    std::vector<std::uint16_t> capture_tap(uint32_t rows) const {
        const uint32_t hidden = static_cast<uint32_t>(model_->text_config().hidden_size);
        const uint32_t stride = static_cast<uint32_t>(
            ((static_cast<uint64_t>(hidden) * 2ull + 15ull) / 16ull) * 16ull / 2ull);
        const size_t count = static_cast<size_t>(rows) * stride;
        std::vector<std::uint16_t> out(count, 0u);
        const ::ps::runtime::Program& program = executor_->program_set.programs[0];
        const void* src = nullptr;
        if (program.external_output_count == 1) {
            src = executor_->output_hidden.data<void>();
        } else if (program.external_output_count > 4) {
            src = executor_->dflash_target_hidden[0].data<void>();
        }
        if (src == nullptr) return out;
        (void)hipMemcpy(out.data(), src, count * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost);
        return out;
    }

    void cleanup() {
        if (sequence_ != nullptr && slot_pool_ != nullptr && gdn_pool_ != nullptr &&
            kv_pool_ != nullptr) {
            ::ps::Status release =
                ::ps::qwen35::release_paged_sequence_state(*sequence_, stream_);
            if (!release.ok() && error_.empty()) error_ = message("release sequence", release);
            sequence_.reset();
        }
        if (executor_ != nullptr) {
            ::ps::Status shutdown = ::ps::qwen35::executor_shutdown(*executor_);
            if (!shutdown.ok() && error_.empty())
                error_ = message("executor shutdown", shutdown);
            executor_.reset();
        }
        kv_pool_.reset();
        gdn_pool_.reset();
        slot_pool_.reset();
        model_.reset();
        if (arena_ != nullptr) {
            arena_->shutdown();
            arena_.reset();
        }
        if (stream_ != nullptr) {
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    static std::string message(const char* what, const ::ps::Status& status) {
        return std::string(what) + ": " + status.message();
    }

    static constexpr uint32_t max_taps = ::ps::qwen35::kMaxTargetHiddenTaps;

    hipStream_t stream_ = nullptr;
    uint32_t max_seq_len_ = 64;
    uint32_t page_tokens_ = 16;
    uint32_t max_tokens_ = 64;
    uint32_t max_blocks_ = 0;
    std::unique_ptr<::ps::gpu::GpuArena> arena_;
    std::unique_ptr<::ps::qwen35::Qwen35Model> model_;
    std::unique_ptr<::ps::qwen35::SequenceSlotPool> slot_pool_;
    std::unique_ptr<::ps::qwen35::GdnStatePool> gdn_pool_;
    std::unique_ptr<::ps::qwen35::PagedKVPool> kv_pool_;
    std::unique_ptr<::ps::qwen35::Executor> executor_;
    std::unique_ptr<::ps::qwen35::PagedSequenceState> sequence_;
    std::string error_;
};

inline SyntheticRunResult run_synthetic_model(const SyntheticRunOptions& options) {
    return SyntheticRunner::run(options);
}

}
