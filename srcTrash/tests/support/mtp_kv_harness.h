#pragma once

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/runtime/staging_layout.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ps::mtp_test {

namespace rt = ::ps::runtime;
namespace gpu = ::ps::gpu;
namespace q35rt = ::ps::qwen35::runtime;

struct Geometry {
    uint32_t q_heads = 4;
    uint32_t kv_heads = 2;
    uint32_t head_dim = 4;
    uint32_t rotary_dim = 2;
    uint32_t page_tokens = 4;
    uint32_t num_pages = 3;
    uint32_t max_rows = 16;

    uint32_t q_features() const { return q_heads * head_dim; }
    uint32_t kv_features() const { return kv_heads * head_dim; }
    uint32_t capacity() const { return page_tokens * num_pages; }
};

inline std::uint16_t bf16_bits(float f) { return ps::f32_to_bf16_rne(f); }

inline float bits_to_f32(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline std::vector<float> to_f32(const std::vector<std::uint16_t>& bits) {
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bits_to_f32(bits[i]);
    return out;
}

inline std::vector<std::uint16_t> to_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = bf16_bits(v[i]);
    return out;
}

struct TokenPattern {
    std::vector<float> k;
    std::vector<float> v;
};

class KvHarness {
 public:
    bool init(gpu::GpuArena& arena, hipStream_t stream, const Geometry& geom, std::string& err) {
        arena_ = &arena;
        stream_ = stream;
        geom_ = geom;

        q35rt::MtpKvConfig cfg;
        cfg.num_pages = geom.num_pages;
        cfg.page_tokens = geom.page_tokens;
        cfg.num_attention_layers = 1u;
        cfg.kv_heads = geom.kv_heads;
        cfg.head_dim = geom.head_dim;
        cfg.dtype = ps::qwen35::KVCacheDType::BF16;
        auto st_res = q35rt::create_mtp_kv_state(arena, cfg, 0u, stream);
        if (!st_res.ok()) {
            err = "create_mtp_kv_state: " + st_res.status().message();
            return false;
        }
        state_ = st_res.release();

        if (!alloc_bytes(geom.max_rows * geom.kv_features() * 2, t_k_, err)) return false;
        if (!alloc_bytes(geom.max_rows * geom.kv_features() * 2, t_v_, err)) return false;
        if (!alloc_bytes(geom.max_rows * geom.q_features() * 2, t_q_, err)) return false;
        if (!alloc_bytes(geom.max_rows * geom.q_features() * 4, t_out_, err)) return false;
        if (!alloc_bytes(geom.max_rows * 4, t_rope_, err)) return false;

        rt::PrimitiveGraph g;
        const rt::AttentionShapeKey attn_key{
            geom.q_heads, geom.kv_heads, geom.head_dim, geom.rotary_dim};
        const rt::ValueId v_k =
            g.alloc_value(geom.kv_features(), rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        const rt::ValueId v_v =
            g.alloc_value(geom.kv_features(), rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        const rt::ValueId v_q =
            g.alloc_value(geom.q_features(), rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        g.external_inputs = {v_k, v_v, v_q};
        const rt::ValueId v_ctx =
            g.alloc_value(geom.q_features(), rt::ValueDType::F32, rt::ValueRowDomain::TOKEN_ROWS);
        g.external_outputs = {v_ctx};

        const rt::StateId kv_state = g.alloc_state(rt::PrimitiveStateKind::KV_CACHE, 0u);
        {
            rt::KvAppendNode n{attn_key, 0u};
            auto& node = g.add_node(rt::PrimitiveNode{n});
            node.inputs = {v_k, v_v};
            node.state_outputs = {kv_state};
        }
        {
            const float scale = geom.head_dim != 0u
                                    ? 1.0f / std::sqrt(static_cast<float>(geom.head_dim))
                                    : 1.0f;
            rt::PagedAttentionNode n{attn_key, scale, 0u};
            auto& node = g.add_node(rt::PrimitiveNode{n});
            node.inputs = {v_q};
            node.outputs = {v_ctx};
            node.state_inputs = {kv_state};
        }

        rt::WeightTableView wv{nullptr, 0};
        rt::StaticParameterTableView pv{nullptr, 0};
        rt::ProgramBuildOptions opts;
        opts.validate_workspace = true;
        opts.max_token_rows = geom.max_rows;
        opts.max_output_rows = 1u;
        auto set_res = rt::build_program_set(g, wv, pv, rt::ExecutionClass::DECODE, opts);
        if (!set_res.ok()) {
            err = "build_program_set: " + set_res.status().message();
            return false;
        }
        program_set_ = set_res.release();
        program_ = program_set_.resolve(rt::RowBucket::R16);
        if (program_ == nullptr) {
            err = "R16 program not resolved";
            return false;
        }

        auto bctx = rt::create_device_batch_context(arena, 1u, geom.max_rows);
        if (!bctx.ok()) {
            err = "create_device_batch_context: " + bctx.status().message();
            return false;
        }
        bstorage_ = bctx.release();

        const std::size_t ws_bytes =
            (static_cast<std::size_t>(program_->workspace.total_bytes) + 255ull) & ~255ull;
        const std::size_t scratch_bytes = kScratchFloats * sizeof(float);
        staging_bytes_ = rt::max_host_staging_bytes(*program_);
        if (hipMalloc(reinterpret_cast<void**>(&ws_), ws_bytes + scratch_bytes) != hipSuccess ||
            hipMemsetAsync(ws_, 0, ws_bytes + scratch_bytes, stream) != hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&error_word_), sizeof(std::uint32_t)) != hipSuccess ||
            hipMemsetAsync(error_word_, 0, sizeof(std::uint32_t), stream) != hipSuccess) {
            err = "workspace allocation";
            return false;
        }
        ws_bytes_ = ws_bytes + scratch_bytes;
        {
            if (hipMalloc(&staging_dev_, staging_bytes_) != hipSuccess ||
                hipHostMalloc(&staging_host_, staging_bytes_) != hipSuccess) {
                err = "staging allocation";
                return false;
            }
            staging_pool_.max_bindings = static_cast<std::uint32_t>(program_->dispatches.size());
            const std::size_t nslots = staging_pool_.max_bindings;
            staging_pool_.offsets.assign(nslots, 0);
            staging_pool_.bytes.assign(nslots, 0);
            staging_pool_.uploaded.assign(nslots, 0);
            std::uint64_t cur = 0;
            for (std::size_t i = 0; i < program_->dispatches.size(); ++i) {
                const std::uint64_t need = rt::staging_bytes_for(program_->dispatches[i]);
                staging_pool_.offsets[i] = cur;
                staging_pool_.bytes[i] = static_cast<std::uint32_t>(need);
                cur += (need + 15ull) & ~15ull;
            }
            if (cur > 0 && hipMalloc(&staging_pool_.device, cur) != hipSuccess) {
                err = "staging pool allocation";
                return false;
            }
            staging_pool_.pool_bytes = cur;
        }
        {
            const std::size_t sc = program_->states.size();
            if (sc > 0) {
                std::vector<rt::DeviceStateBinding> hs(sc);
                for (std::size_t i = 0; i < sc; ++i) {
                    hs[i].slot = program_->states[i].slot;
                    hs[i].kind = static_cast<std::uint8_t>(program_->states[i].kind);
                    hs[i].state_index = program_->states[i].state_index;
                }
                if (hipMalloc(&states_dev_, sc * sizeof(hs[0])) != hipSuccess ||
                    hipMemcpy(states_dev_, hs.data(), sc * sizeof(hs[0]),
                              hipMemcpyHostToDevice) != hipSuccess) {
                    err = "state binding upload";
                    return false;
                }
                ::ps::qwen35::ProgramStagingMeta meta{};
                meta.ranges = nullptr;
                meta.range_count = 0u;
                meta.states = states_dev_;
                meta.state_count = static_cast<std::uint32_t>(sc);
                meta_host_ = meta;
            }
        }
        ready_ = true;
        return true;
    }

    ~KvHarness() {
        if (ws_ != nullptr) ps::gpu::discard_cleanup_result(hipFree(ws_));
        if (error_word_ != nullptr) ps::gpu::discard_cleanup_result(hipFree(error_word_));
        if (staging_dev_ != nullptr) ps::gpu::discard_cleanup_result(hipFree(staging_dev_));
        if (staging_host_ != nullptr) ps::gpu::discard_cleanup_result(hipHostFree(staging_host_));
        if (staging_pool_.device != nullptr)
            ps::gpu::discard_cleanup_result(hipFree(staging_pool_.device));
        if (states_dev_ != nullptr) ps::gpu::discard_cleanup_result(hipFree(states_dev_));
    }

    KvHarness() = default;
    KvHarness(const KvHarness&) = delete;
    KvHarness& operator=(const KvHarness&) = delete;

    bool run(const std::vector<float>& k, const std::vector<float>& v,
             const std::vector<float>& q, uint32_t rows, uint32_t kv_index,
             uint32_t rope_position, std::vector<float>& out, std::string& err) {
        if (!ready_) {
            err = "harness not ready";
            return false;
        }
        if (kv_index != state_.logical_length) {
            err = "kv_index != state logical length (append semantics)";
            return false;
        }
        if (k.size() != rows * geom_.kv_features() || v.size() != rows * geom_.kv_features() ||
            q.size() != rows * geom_.q_features()) {
            err = "run input size mismatch";
            return false;
        }
        const auto kb = to_bits(k);
        const auto vb = to_bits(v);
        const auto qb = to_bits(q);
        if (hipMemcpyAsync(t_k_.data<void>(), kb.data(), kb.size() * 2, hipMemcpyHostToDevice,
                           stream_) != hipSuccess ||
            hipMemcpyAsync(t_v_.data<void>(), vb.data(), vb.size() * 2, hipMemcpyHostToDevice,
                           stream_) != hipSuccess ||
            hipMemcpyAsync(t_q_.data<void>(), qb.data(), qb.size() * 2, hipMemcpyHostToDevice,
                           stream_) != hipSuccess) {
            err = "input upload";
            return false;
        }

        std::vector<std::uint32_t> rope(rows);
        for (uint32_t t = 0; t < rows; ++t) rope[t] = rope_position + t;
        if (hipMemcpyAsync(t_rope_.data<void>(), rope.data(), rows * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream_) != hipSuccess) {
            err = "rope upload";
            return false;
        }

        {
            rt::DeviceRequestDescriptor desc{};
            desc.request_handle = state_.handle;
            desc.row_begin = 0u;
            desc.row_count = rows;
            desc.prefix_length = kv_index;
            desc.sequence_length = kv_index + rows;
            desc.output_index = 0u;
            desc.execution_class = rows == 1u ? rt::ExecutionClass::DECODE
                                              : rt::ExecutionClass::PREFILL;
            desc.compute_logits = 0u;
            desc.sampling.mode = 0u;
            if (hipMemcpyAsync(bstorage_.requests.data<rt::DeviceRequestDescriptor>(), &desc,
                               sizeof(desc), hipMemcpyHostToDevice, stream_) != hipSuccess) {
                err = "request descriptor";
                return false;
            }
            rt::DeviceBatchContext hc{};
            hc.actual_rows = rows;
            hc.num_requests = 1u;
            hc.num_outputs = 1u;
            hc.num_decode_requests = rows == 1u ? 1u : 0u;
            hc.num_prefill_requests = rows == 1u ? 0u : 1u;
            hc.token_ids = bstorage_.token_ids.data<int32_t>();
            hc.requests = bstorage_.requests.data<rt::DeviceRequestDescriptor>();
            hc.row_sequence_slots = bstorage_.row_sequence_slots.data<std::uint32_t>();
            hc.row_positions = bstorage_.row_positions.data<std::uint32_t>();
            hc.rope_positions = t_rope_.data<std::uint32_t>();
            hc.output_rows = bstorage_.output_rows.data<std::uint32_t>();
            hc.output_sampling_params =
                bstorage_.output_sampling_params
                    .data<::ps::runtime::DeviceSamplingParams>();
            hc.device_ptr = bstorage_.device_struct.data<rt::DeviceBatchContext>();
            if (hipMemcpyAsync(bstorage_.device_struct.data<rt::DeviceBatchContext>(), &hc,
                               sizeof(hc), hipMemcpyHostToDevice, stream_) != hipSuccess) {
                err = "batch context";
                return false;
            }
        }
        {
            auto pst = rt::launch_prepare_batch_descriptor(
                bstorage_.device_struct.data<rt::DeviceBatchContext>(), 1u, stream_);
            if (!pst.ok()) {
                err = "prepare descriptor: " + pst.message();
                return false;
            }
        }

        const void* inputs[3] = {t_k_.data<void>(), t_v_.data<void>(), t_q_.data<void>()};
        void* outputs[1] = {t_out_.data<void>()};

        q35rt::HostExecutionContext ctx{};
        ctx.batch_context = bstorage_.device_struct.data<rt::DeviceBatchContext>();
        ctx.model_state = q35rt::mtp_kv_state_view(state_);
        ctx.row_positions = bstorage_.row_positions.data<std::uint32_t>();
        ctx.rope_positions = t_rope_.data<std::uint32_t>();
        ctx.row_sequence_slots = bstorage_.row_sequence_slots.data<std::uint32_t>();
        ctx.external_inputs = inputs;
        ctx.external_input_count = 3u;
        ctx.external_outputs = outputs;
        ctx.external_output_count = 1u;
        ctx.workspace = static_cast<std::uint8_t*>(ws_);
        ctx.workspace_bytes = ws_bytes_;
        ctx.scratch = reinterpret_cast<float*>(static_cast<std::uint8_t*>(ws_) + ws_scratch_off());
        ctx.scratch_floats = kScratchFloats;
        ctx.error_word = error_word_;
        ctx.bucket_m = rt::row_bucket_limit(rt::RowBucket::R16);
        ctx.staging = staging_dev_;
        ctx.staging_bytes = staging_bytes_;
        ctx.host_staging = staging_host_;
        ctx.staging_pool = &staging_pool_;
        ctx.program_meta = (states_dev_ != nullptr) ? &meta_host_ : nullptr;
        ctx.actual_rows = rows;
        ctx.actual_outputs = 1u;
        ctx.actual_sampled_outputs = 0u;

        auto est = q35rt::execute_program(*program_, ctx, stream_);
        if (!est.ok()) {
            err = "execute_program: " + est.message();
            return false;
        }
        if (hipStreamSynchronize(stream_) != hipSuccess) {
            err = "stream sync";
            return false;
        }
        std::uint32_t ew = 1;
        ps::gpu::discard_cleanup_result(
            hipMemcpy(&ew, error_word_, sizeof(ew), hipMemcpyDeviceToHost));
        if (ew != 0u) {
            err = "device error word=" + std::to_string(ew);
            return false;
        }

        std::vector<float> ob(rows * geom_.q_features());
        if (hipMemcpy(ob.data(), t_out_.data<void>(), ob.size() * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            err = "output read";
            return false;
        }
        out = std::move(ob);
        auto reserve = q35rt::mtp_kv_reserve(state_, rows);
        if (!reserve.ok()) {
            err = "mtp_kv_reserve: " + reserve.message();
            return false;
        }
        return true;
    }

    void poison(float k_val, float v_val) {
        const std::uint16_t kb = bf16_bits(k_val);
        const std::uint16_t vb = bf16_bits(v_val);
        const std::size_t n = static_cast<std::size_t>(geom_.capacity()) * geom_.kv_features();
        std::vector<std::uint16_t> kbuf(n, kb);
        std::vector<std::uint16_t> vbuf(n, vb);
        if (hipMemcpy(state_.pool->k_pool().data<void>(), kbuf.data(), n * 2,
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(state_.pool->v_pool().data<void>(), vbuf.data(), n * 2,
                      hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL: poison upload\n");
        }
        if (hipDeviceSynchronize() != hipSuccess) {
            std::printf("FAIL: poison sync\n");
        }
    }

    std::vector<float> dump_k(uint32_t length) {
        std::vector<ps::bf16_t> buf(static_cast<std::size_t>(length) * geom_.kv_features());
        std::vector<ps::bf16_t> vbuf(buf.size());
        auto st = q35rt::mtp_kv_dump_canonical(state_, length, buf.data(), vbuf.data(), stream_);
        if (!st.ok()) return {};
        std::vector<std::uint16_t> bits(buf.size());
        std::memcpy(bits.data(), buf.data(), buf.size() * 2);
        return to_f32(bits);
    }

    std::vector<float> dump_v(uint32_t length) {
        std::vector<ps::bf16_t> buf(static_cast<std::size_t>(length) * geom_.kv_features());
        std::vector<ps::bf16_t> kbuf(buf.size());
        auto st = q35rt::mtp_kv_dump_canonical(state_, length, kbuf.data(), buf.data(), stream_);
        if (!st.ok()) return {};
        std::vector<std::uint16_t> bits(buf.size());
        std::memcpy(bits.data(), buf.data(), buf.size() * 2);
        return to_f32(bits);
    }

    q35rt::MtpKvState& state() { return state_; }
    const Geometry& geometry() const { return geom_; }

    static constexpr std::uint32_t kScratchFloats = 8192u;

 private:
    std::size_t ws_scratch_off() const {
        return (static_cast<std::size_t>(program_->workspace.total_bytes) + 255ull) & ~255ull;
    }

    bool alloc_bytes(std::size_t n_bytes, gpu::Tensor& out, std::string& err) {
        auto view = arena_->allocate_aligned(n_bytes == 0 ? 1 : n_bytes, 256);
        if (!view.ok()) {
            err = "allocate: " + view.status().message();
            return false;
        }
        auto t = gpu::Tensor::create<ps::bf16_t>(view.release(), {n_bytes / 2}, {1});
        if (!t.ok()) {
            err = "tensor: " + t.status().message();
            return false;
        }
        out = t.release();
        return true;
    }

    gpu::GpuArena* arena_ = nullptr;
    hipStream_t stream_ = nullptr;
    Geometry geom_{};
    q35rt::MtpKvState state_{};

    gpu::Tensor t_k_, t_v_, t_q_, t_out_, t_rope_;

    rt::ProgramSet program_set_;
    const rt::Program* program_ = nullptr;

    rt::DeviceBatchContextStorage bstorage_{};

    void* ws_ = nullptr;
    std::size_t ws_bytes_ = 0;
    std::uint32_t* error_word_ = nullptr;
    void* staging_dev_ = nullptr;
    void* staging_host_ = nullptr;
    void* states_dev_ = nullptr;
    ::ps::qwen35::ProgramStagingMeta meta_host_{};
    ::ps::qwen35::DispatchStagingPool staging_pool_{};
    std::uint64_t staging_bytes_ = 0;
    bool ready_ = false;
};

inline std::vector<float> reference_attention(
    const Geometry& geom, const std::vector<float>& q_row,
    const std::vector<std::vector<float>>& k_hist,
    const std::vector<std::vector<float>>& v_hist) {
    const uint32_t hd = geom.head_dim;
    const uint32_t qh = geom.q_heads;
    const uint32_t kh = geom.kv_heads;
    const uint32_t group = qh / kh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> out(geom.q_features(), 0.0f);
    const std::size_t n = k_hist.size();
    for (uint32_t h = 0; h < qh; ++h) {
        const uint32_t kv_head = h / group;
        double max_score = -1e30;
        for (std::size_t p = 0; p < n; ++p) {
            double sc = 0.0;
            for (uint32_t i = 0; i < hd; ++i) {
                sc += static_cast<double>(q_row[h * hd + i]) *
                      static_cast<double>(k_hist[p][kv_head * hd + i]);
            }
            sc *= static_cast<double>(scale);
            if (sc > max_score) max_score = sc;
        }
        double sum = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            double sc = 0.0;
            for (uint32_t i = 0; i < hd; ++i) {
                sc += static_cast<double>(q_row[h * hd + i]) *
                      static_cast<double>(k_hist[p][kv_head * hd + i]);
            }
            sc *= static_cast<double>(scale);
            sum += std::exp(sc - max_score);
        }
        for (uint32_t i = 0; i < hd; ++i) {
            double acc = 0.0;
            for (std::size_t p = 0; p < n; ++p) {
                double sc = 0.0;
                for (uint32_t j = 0; j < hd; ++j) {
                    sc += static_cast<double>(q_row[h * hd + j]) *
                          static_cast<double>(k_hist[p][kv_head * hd + j]);
                }
                sc *= static_cast<double>(scale);
                acc += std::exp(sc - max_score) * static_cast<double>(v_hist[p][kv_head * hd + i]);
            }
            out[h * hd + i] = static_cast<float>(acc / sum);
        }
    }
    return out;
}

}  // namespace ps::mtp_test
