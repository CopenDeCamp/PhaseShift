#include <phaseshift/models/qwen35/runtime/mtp_executor.h>
#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/rope.h>
#include <phaseshift/models/qwen35/kernels/optimized/sampling.h>
#include <phaseshift/runtime/staging_layout.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <algorithm>
#include <cstring>

namespace ps::qwen35::runtime {

namespace {

template <typename T>
Status alloc_tensor(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = gpu::Tensor::create<T>(backing, shape, strides);
    if (!t.ok()) return t.status();
    out = t.release();
    return Status::make_ok();
}

}  // namespace

Result<MtpExecutor> create_mtp_executor(
    const Qwen35Model& model,
    KVCacheDType kv_dtype,
    gpu::GpuArena& arena,
    const MtpExecutorConfig& config,
    hipStream_t stream) {

    const Qwen35TextConfig& tc = model.text_config();
    if (kv_dtype != KVCacheDType::BF16 && kv_dtype != KVCacheDType::FP8_E4M3) {
        return Status::unsupported("MTP executor supports BF16 or FP8_E4M3 KV only",
                                   __FILE__, __LINE__);
    }
    if (!model.weights().mtp.present) {
        return Status::unsupported("model has no MTP weights", __FILE__, __LINE__);
    }

    auto scoped_device = gpu::ScopedDevice::create(arena.device());
    if (!scoped_device.ok()) return scoped_device.status();

    MtpExecutor ex;
    ex.model = &model;
    ex.kv_dtype = kv_dtype;
    ex.max_rows = config.max_rows < 16u ? config.max_rows : 16u;

    const std::size_t H = tc.hidden_size;
    const std::size_t V = tc.vocab_size;

    {
        MtpKvConfig kv_cfg;
        kv_cfg.num_pages = config.num_pages;
        kv_cfg.page_tokens = config.page_tokens;
        kv_cfg.num_attention_layers = 1u;
        kv_cfg.kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
        kv_cfg.head_dim = static_cast<uint32_t>(tc.attention_head_dim);
        kv_cfg.dtype = kv_dtype;
        auto kv_res = create_mtp_kv_state(arena, kv_cfg, 0u, stream);
        if (!kv_res.ok()) return kv_res.status();
        ex.kv_state = kv_res.release();
    }

    auto st = alloc_tensor<bf16_t>(arena, {static_cast<std::size_t>(config.max_rows) * H},
                                   ex.hidden_in);
    if (!st.ok()) return st;
    st = alloc_tensor<bf16_t>(arena, {H}, ex.hidden_out);
    if (!st.ok()) return st;
    st = alloc_tensor<bf16_t>(arena, {H}, ex.hidden_out_normed);
    if (!st.ok()) return st;
    st = alloc_tensor<float>(arena, {V}, ex.logits);
    if (!st.ok()) return st;
    st = alloc_tensor<float>(arena, {4}, ex.top2);
    if (!st.ok()) return st;
    st = alloc_tensor<int32_t>(arena, {1}, ex.sampled);
    if (!st.ok()) return st;
    st = alloc_tensor<uint32_t>(arena, {1}, ex.execution_status);
    if (!st.ok()) return st;
    {
        uint32_t zero = 0;
        hipError_t ce = hipMemcpyAsync(
            ex.execution_status.data<uint32_t>(), &zero, sizeof(uint32_t),
            hipMemcpyHostToDevice, stream);
        if (ce != hipSuccess) {
            return Status::hip_error("mtp execution_status init", hipGetErrorString(ce),
                                     __FILE__, __LINE__);
        }
    }

    {
        auto bctx = ::ps::runtime::create_device_batch_context(arena, 1u, config.max_rows);
        if (!bctx.ok()) return bctx.status();
        ex.batch_storage = bctx.release();
        ex.batch_context = ex.batch_storage.device_struct
                               .data<::ps::runtime::DeviceBatchContext>();
        st = alloc_tensor<uint32_t>(arena, {config.max_rows}, ex.batch_storage.rope_positions);
        if (!st.ok()) return st;
    }

    ex.model_state = mtp_kv_state_view(ex.kv_state);

    {
        hipStream_t upload_stream = nullptr;
        hipError_t herr = hipStreamCreate(&upload_stream);
        if (herr != hipSuccess) {
            return Status::hip_error("mtp upload stream", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
        auto lower = lower_qwen35_mtp_to_primitives(tc, model.weights());
        if (!lower.ok()) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
            return lower.status();
        }
        auto lowered = lower.release();
        std::vector<::ps::runtime::WeightSlot> weight_slots = std::move(lowered.weights);
        std::vector<::ps::runtime::StaticParameterSlot> param_slots =
            std::move(lowered.parameters);
        ex.host_parameter_slots = param_slots;
        ::ps::runtime::PrimitiveGraph graph = std::move(lowered.graph);

        ::ps::runtime::WeightTableView wv{nullptr, 0};
        wv.slots = weight_slots.data();
        wv.count = weight_slots.size();
        ::ps::runtime::StaticParameterTableView pv;
        pv.slots = param_slots.data();
        pv.count = param_slots.size();

        ::ps::runtime::ProgramBuildOptions opts;
        opts.validate_workspace = true;
        opts.max_token_rows = ex.max_rows;
        opts.max_output_rows = 1u;
        auto set_res = ::ps::runtime::build_program_set(
            graph, wv, pv, ::ps::runtime::ExecutionClass::DECODE, opts);
        if (!set_res.ok()) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
            return set_res.status();
        }
        ::ps::runtime::ProgramSet built = set_res.release();

        std::size_t max_workspace = 0;
        uint64_t max_staging = 0;
        for (auto& prog : built.programs) {
            max_workspace = std::max(max_workspace, prog.workspace.total_bytes);
            max_staging = std::max(max_staging, ::ps::runtime::max_host_staging_bytes(prog));
        }
        constexpr std::size_t kScratchBytes = 8192u * sizeof(float);
        const std::size_t ws_bytes =
            max_workspace + kScratchBytes + kDecodeAttnPartialBytes;
        if (ws_bytes > 0) {
            st = alloc_tensor<uint8_t>(arena, {ws_bytes}, ex.workspace);
            if (!st.ok()) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return st;
            }
            hipError_t werr = hipMemsetAsync(
                ex.workspace.data<uint8_t>(), 0, ws_bytes, upload_stream);
            if (werr != hipSuccess) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return Status::hip_error("mtp workspace memset", hipGetErrorString(werr),
                                         __FILE__, __LINE__);
            }
        }

        st = alloc_tensor<double>(arena, {32}, ex.rope_inv_freq);
        if (!st.ok()) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
            return st;
        }
        {
            float rope_theta = tc.rope_theta;
            if (rope_theta <= 0.0f) rope_theta = 10000000.0f;
            hipError_t rerr = ::ps::kernel::launch_rope_inv_freq_init(
                ex.rope_inv_freq.data<double>(), 32, rope_theta, upload_stream);
            if (rerr != hipSuccess) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return Status::hip_error("mtp rope inv_freq init", hipGetErrorString(rerr),
                                         __FILE__, __LINE__);
            }
        }

        {
            auto wtable = upload_host_weight_table(wv, &ex.host_weight_table, upload_stream);
            if (!wtable.ok()) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return wtable;
            }
        }
        {
            auto ptable = upload_host_parameter_table(pv, &ex.host_parameter_table, upload_stream);
            if (!ptable.ok()) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return ptable;
            }
        }

        constexpr size_t kBuckets = ::ps::runtime::ProgramSet::kRowBucketCount;
        ex.program_staging_meta.resize(kBuckets);
        for (size_t b = 0; b < kBuckets; ++b) {
            auto& prog = built.programs[b];
            const uint64_t rc = prog.workspace.ranges.size();
            const uint64_t sc = prog.states.size();
            const uint64_t mbytes = ::ps::runtime::program_staging_meta_bytes(
                static_cast<uint32_t>(rc), static_cast<uint32_t>(sc));
            auto& meta = ex.program_staging_meta[b];
            meta.range_count = static_cast<uint32_t>(rc);
            meta.state_count = static_cast<uint32_t>(sc);
            if (mbytes == 0) continue;
            void* mb = nullptr;
            hipError_t merr = hipMalloc(&mb, mbytes);
            if (merr != hipSuccess) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return Status::hip_error("mtp meta malloc", hipGetErrorString(merr),
                                         __FILE__, __LINE__);
            }
            uint8_t* p = static_cast<uint8_t*>(mb);
            if (rc > 0) {
                std::vector<::ps::runtime::DeviceWorkspaceRange> hr(rc);
                for (uint64_t i = 0; i < rc; ++i) {
                    hr[i].offset = prog.workspace.ranges[i].offset;
                    hr[i].bytes = prog.workspace.ranges[i].bytes;
                }
                merr = hipMemcpy(p, hr.data(), rc * sizeof(hr[0]), hipMemcpyHostToDevice);
                if (merr != hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipFree(mb));
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("mtp meta ranges", hipGetErrorString(merr),
                                             __FILE__, __LINE__);
                }
                p += (rc * sizeof(hr[0]) + 15ull) & ~15ull;
            }
            if (sc > 0) {
                std::vector<::ps::runtime::DeviceStateBinding> hs(sc);
                for (uint64_t i = 0; i < sc; ++i) {
                    hs[i].slot = prog.states[i].slot;
                    hs[i].kind = static_cast<uint8_t>(prog.states[i].kind);
                    hs[i].state_index = prog.states[i].state_index;
                }
                merr = hipMemcpy(p, hs.data(), sc * sizeof(hs[0]), hipMemcpyHostToDevice);
                if (merr != hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipFree(mb));
                    ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                    return Status::hip_error("mtp meta states", hipGetErrorString(merr),
                                             __FILE__, __LINE__);
                }
            }
            const uint64_t ranges_bytes = (rc * sizeof(::ps::runtime::DeviceWorkspaceRange) + 15ull) & ~15ull;
            meta.ranges = (rc > 0) ? mb : nullptr;
            meta.states = (rc > 0) ? (static_cast<uint8_t*>(mb) + ranges_bytes)
                                   : ((sc > 0) ? mb : nullptr);
            meta.device_bytes = mbytes;
        }

        if (max_staging > 0) {
            hipError_t serr = hipMalloc(&ex.dispatch_staging, max_staging);
            if (serr != hipSuccess) {
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return Status::hip_error("mtp dispatch staging malloc", hipGetErrorString(serr),
                                         __FILE__, __LINE__);
            }
            ex.dispatch_staging_bytes = max_staging;
            void* host_stage = nullptr;
            hipError_t hserr = hipHostMalloc(&host_stage, max_staging);
            if (hserr != hipSuccess) {
                ps::gpu::discard_cleanup_result(hipFree(ex.dispatch_staging));
                ex.dispatch_staging = nullptr;
                ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                return Status::hip_error("mtp host staging malloc", hipGetErrorString(hserr),
                                         __FILE__, __LINE__);
            }
            ex.host_dispatch_staging = host_stage;
        }

        {
            uint32_t max_bindings = 0;
            for (size_t b = 0; b < kBuckets; ++b) {
                max_bindings = std::max<uint32_t>(
                    max_bindings, static_cast<uint32_t>(built.programs[b].dispatches.size()));
            }
            if (max_bindings > 0) {
                ex.staging_pool.max_bindings = max_bindings;
                const size_t nslots = kBuckets * max_bindings;
                ex.staging_pool.offsets.assign(nslots, 0);
                ex.staging_pool.bytes.assign(nslots, 0);
                ex.staging_pool.uploaded.assign(nslots, 0);
                ex.staging_pool.run_seen.assign(nslots, 0);
                ex.staging_pool.bucket_ran.assign(kBuckets, 0);
                uint64_t cur = 0;
                for (size_t b = 0; b < kBuckets; ++b) {
                    auto& prog = built.programs[b];
                    for (size_t i = 0; i < prog.dispatches.size(); ++i) {
                        const uint64_t need = ::ps::runtime::staging_bytes_for(prog.dispatches[i]);
                        const uint64_t slot = ex.staging_pool.slot(
                            static_cast<uint32_t>(b), static_cast<uint32_t>(i));
                        ex.staging_pool.offsets[slot] = cur;
                        ex.staging_pool.bytes[slot] = static_cast<uint32_t>(need);
                        cur += (need + 15ull) & ~15ull;
                    }
                }
                if (cur > 0) {
                    hipError_t perr = hipMalloc(&ex.staging_pool.device, cur);
                    if (perr != hipSuccess) {
                        ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
                        return Status::hip_error("mtp staging pool malloc",
                                                 hipGetErrorString(perr), __FILE__, __LINE__);
                    }
                    ex.staging_pool.pool_bytes = cur;
                }
            }
        }

        ex.program_set = std::move(built);
        ex.program_set_ready = true;
        hipError_t sync = hipStreamSynchronize(upload_stream);
        ps::gpu::discard_cleanup_result(hipStreamDestroy(upload_stream));
        if (sync != hipSuccess) {
            return Status::hip_error("mtp upload sync", hipGetErrorString(sync),
                                     __FILE__, __LINE__);
        }
    }

    return ex;
}

Status mtp_executor_shutdown(MtpExecutor& ex) noexcept {
    Status kv_status = mtp_kv_shutdown(ex.kv_state);
    (void)kv_status;
    if (ex.host_weight_table != nullptr) {
        ps::gpu::discard_cleanup_result(hipFree(ex.host_weight_table));
        ex.host_weight_table = nullptr;
    }
    if (ex.host_parameter_table != nullptr) {
        ps::gpu::discard_cleanup_result(hipFree(ex.host_parameter_table));
        ex.host_parameter_table = nullptr;
    }
    for (auto& meta : ex.program_staging_meta) {
        if (meta.device_bytes > 0) {
            void* p = (meta.ranges != nullptr) ? meta.ranges : meta.states;
            ps::gpu::discard_cleanup_result(hipFree(p));
            meta = ProgramStagingMeta{};
        }
    }
    ex.program_staging_meta.clear();
    if (ex.dispatch_staging != nullptr) {
        ps::gpu::discard_cleanup_result(hipFree(ex.dispatch_staging));
        ex.dispatch_staging = nullptr;
    }
    if (ex.host_dispatch_staging != nullptr) {
        ps::gpu::discard_cleanup_result(hipHostFree(ex.host_dispatch_staging));
        ex.host_dispatch_staging = nullptr;
    }
    if (ex.staging_pool.device != nullptr) {
        ps::gpu::discard_cleanup_result(hipFree(ex.staging_pool.device));
        ex.staging_pool.device = nullptr;
    }
    ex.program_set_ready = false;
    ex.model = nullptr;
    return Status::make_ok();
}

namespace {

Result<uint32_t> run_mtp_rows_on_state(
    MtpExecutor& ex,
    MtpKvState& state,
    const bf16_t* hidden,
    const int32_t* tokens,
    uint32_t rows,
    uint32_t position_start,
    uint32_t kv0,
    hipStream_t stream,
    ValueTraceSink* trace) {

    if (!ex.program_set_ready || ex.model == nullptr) {
        return Status::invalid_state("MTP executor not ready", __FILE__, __LINE__);
    }
    if (!state.initialized) {
        return Status::invalid_state("MTP KV state not initialized", __FILE__, __LINE__);
    }
    if (hidden == nullptr || tokens == nullptr) {
        return Status::invalid_argument("MTP inputs must be non-null", __FILE__, __LINE__);
    }
    if (rows == 0u || rows > ex.max_rows) {
        return Status::invalid_argument("MTP rows out of range", __FILE__, __LINE__);
    }
    const Qwen35TextConfig& tc = ex.model->text_config();
    const uint32_t H = static_cast<uint32_t>(tc.hidden_size);

    hipError_t herr = hipMemcpyAsync(
        ex.hidden_in.data<bf16_t>(), hidden,
        static_cast<size_t>(rows) * H * sizeof(bf16_t), hipMemcpyDeviceToDevice, stream);
    if (herr != hipSuccess) {
        return Status::hip_error("mtp hidden copy", hipGetErrorString(herr), __FILE__, __LINE__);
    }
    herr = hipMemcpyAsync(ex.batch_storage.token_ids.data<int32_t>(), tokens,
                          static_cast<size_t>(rows) * sizeof(int32_t),
                          hipMemcpyHostToDevice, stream);
    if (herr != hipSuccess) {
        return Status::hip_error("mtp token copy", hipGetErrorString(herr), __FILE__, __LINE__);
    }

    {
        auto* requests = ex.batch_storage.requests
                             .data<::ps::runtime::DeviceRequestDescriptor>();
        ::ps::runtime::DeviceRequestDescriptor desc{};
        desc.request_handle = state.handle;
        desc.row_begin = 0u;
        desc.row_count = rows;
        desc.prefix_length = kv0;
        desc.sequence_length = kv0 + rows;
        desc.output_index = 0u;
        desc.execution_class = rows == 1u
                                   ? ::ps::runtime::ExecutionClass::DECODE
                                   : ::ps::runtime::ExecutionClass::PREFILL;
        desc.compute_logits = 1u;
        desc.sampling.mode = static_cast<uint8_t>(::ps::qwen35::runtime::SamplingMode::Greedy);
        desc.sampling.temperature = 0.0f;
        desc.sampling.top_p = 1.0f;
        desc.sampling.top_k = 0u;
        desc.sampling.seed = 0u;
        desc.sampling.sample_index = 0u;
        herr = hipMemcpyAsync(requests, &desc, sizeof(desc), hipMemcpyHostToDevice, stream);
        if (herr != hipSuccess) {
            return Status::hip_error("mtp descriptor copy", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
        uint32_t rope_pos[16];
        for (uint32_t t = 0u; t < rows; ++t) rope_pos[t] = position_start + t;
        herr = hipMemcpyAsync(ex.batch_storage.rope_positions.data<uint32_t>(), rope_pos,
                              sizeof(uint32_t) * rows, hipMemcpyHostToDevice, stream);
        if (herr != hipSuccess) {
            return Status::hip_error("mtp rope position copy", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
        ::ps::runtime::DeviceBatchContext hc{};
        hc.actual_rows = rows;
        hc.num_requests = 1u;
        hc.num_outputs = 1u;
        hc.num_decode_requests = rows == 1u ? 1u : 0u;
        hc.num_prefill_requests = rows == 1u ? 0u : 1u;
        hc.token_ids = ex.batch_storage.token_ids.data<int32_t>();
        hc.requests = requests;
        hc.row_sequence_slots = ex.batch_storage.row_sequence_slots.data<uint32_t>();
        hc.row_positions = ex.batch_storage.row_positions.data<uint32_t>();
        hc.rope_positions = ex.batch_storage.rope_positions.data<uint32_t>();
        hc.output_rows = ex.batch_storage.output_rows.data<uint32_t>();
        hc.output_sampling_params =
            ex.batch_storage.output_sampling_params
                .data<::ps::runtime::DeviceSamplingParams>();
        hc.device_ptr = ex.batch_context;
        herr = hipMemcpyAsync(ex.batch_context, &hc, sizeof(hc), hipMemcpyHostToDevice, stream);
        if (herr != hipSuccess) {
            return Status::hip_error("mtp batch context copy", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
    }

    Status st = ::ps::runtime::launch_prepare_batch_descriptor(ex.batch_context, 1u, stream);
    if (!st.ok()) return st;

    {
        uint32_t zero = 0;
        herr = hipMemcpyAsync(ex.execution_status.data<uint32_t>(), &zero, sizeof(uint32_t),
                              hipMemcpyHostToDevice, stream);
        if (herr != hipSuccess) {
            return Status::hip_error("mtp status reset", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
    }

    auto* program = ex.program_set.resolve(::ps::runtime::RowBucket::R16);
    if (program == nullptr) {
        return Status::invalid_state("MTP program not resolved", __FILE__, __LINE__);
    }

    const void* inputs[8] = {};
    inputs[0] = ex.batch_storage.token_ids.data<void>();
    inputs[1] = ex.hidden_in.data<void>();
    void* outputs[8] = {};
    outputs[0] = ex.hidden_out.data<void>();
    outputs[1] = ex.logits.data<void>();
    outputs[2] = ex.sampled.data<void>();
    outputs[3] = ex.hidden_out_normed.data<void>();

    auto* ws_base = ex.workspace.data<uint8_t>();
    if (ws_base == nullptr) {
        return Status::invalid_state("MTP workspace not allocated", __FILE__, __LINE__);
    }
    constexpr uint64_t kScratchFloats = 8192u;
    const uint64_t scratch_offset =
        (static_cast<uint64_t>(program->workspace.total_bytes) + 255ull) & ~255ull;

    runtime::HostExecutionContext ctx{};
    ctx.batch_context = ex.batch_context;
    ctx.model_state = mtp_kv_state_view(state);
    ctx.weights = ex.host_weight_table;
    ctx.weight_count = program->weight_slot_count;
    ctx.parameters = ex.host_parameter_table;
    ctx.parameter_count = program->parameter_slot_count;
    ctx.host_parameters = ex.host_parameter_slots.data();
    ctx.host_parameter_count = static_cast<uint32_t>(ex.host_parameter_slots.size());
    ctx.row_positions = ex.batch_storage.row_positions.data<uint32_t>();
    ctx.rope_positions = ex.batch_storage.rope_positions.data<uint32_t>();
    ctx.rope_inv_freq = ex.rope_inv_freq.data<double>();
    ctx.row_sequence_slots = ex.batch_storage.row_sequence_slots.data<uint32_t>();
    ctx.external_inputs = inputs;
    ctx.external_input_count = 2u;
    ctx.external_outputs = outputs;
    ctx.external_output_count = program->external_output_count;
    ctx.workspace = ws_base;
    ctx.workspace_bytes = ex.workspace.allocation_bytes();
    ctx.scratch = reinterpret_cast<float*>(ws_base + scratch_offset);
    ctx.scratch_floats = static_cast<uint32_t>(kScratchFloats);
    const uint64_t attn_partial_offset =
        (scratch_offset + kScratchFloats * sizeof(float) + 255ull) & ~255ull;
    if (attn_partial_offset + kDecodeAttnPartialBytes > ex.workspace.allocation_bytes()) {
        return Status::insufficient_memory("MTP workspace too small for attention partials",
                                           __FILE__, __LINE__);
    }
    ctx.decode_attn_partials = ws_base + attn_partial_offset;
    ctx.decode_attn_partial_bytes = static_cast<uint32_t>(kDecodeAttnPartialBytes);
    ctx.error_word = ex.execution_status.data<uint32_t>();
    ctx.bucket_m = ::ps::runtime::row_bucket_limit(::ps::runtime::RowBucket::R16);
    ctx.min_visible_tokens = kv0 + 1u;
    ctx.max_visible_tokens = kv0 + rows;
    ctx.staging = ex.dispatch_staging;
    ctx.staging_bytes = ex.dispatch_staging_bytes;
    ctx.host_staging = ex.host_dispatch_staging;
    ctx.staging_pool = &ex.staging_pool;
    ctx.program_meta = ex.program_staging_meta.empty() ? nullptr : &ex.program_staging_meta[0];
    ctx.actual_rows = rows;
    ctx.actual_outputs = 1u;
    ctx.actual_sampled_outputs = 1u;
    ctx.actual_stochastic_outputs = 0u;
    ctx.output_sampling_params =
        ex.batch_storage.output_sampling_params
            .data<::ps::runtime::DeviceSamplingParams>();
    ctx.value_trace = trace;

    st = runtime::execute_program(*program, ctx, stream);
    if (!st.ok()) return st;

    {
        const uint32_t V = static_cast<uint32_t>(ex.logits.dim(0));
        auto* top2_out = reinterpret_cast<::ps::kernel::ArgMaxTop2*>(ex.top2.data<float>());
        if (ex.collect_top2 && V > 0u && top2_out != nullptr) {
            hipError_t t2 = ::ps::kernel::launch_argmax_f32_top2(
                ex.logits.data<float>(), nullptr, top2_out, 1u, V, V, stream);
            if (t2 != hipSuccess) {
                return Status::hip_error("mtp top2 launch", hipGetErrorString(t2),
                                         __FILE__, __LINE__);
            }
        }
    }

    herr = hipStreamSynchronize(stream);
    if (herr != hipSuccess) {
        return Status::hip_error("mtp sync", hipGetErrorString(herr), __FILE__, __LINE__);
    }
    uint32_t status = 0;
    herr = hipMemcpy(&status, ex.execution_status.data<uint32_t>(), sizeof(status),
                     hipMemcpyDeviceToHost);
    if (herr != hipSuccess) {
        return Status::hip_error("mtp status read", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    }
    if (status != 0u) {
        return Status::invalid_state(
            ("MTP forward failed: device status=" + std::to_string(status) +
             " (a dispatch returned non-COMPLETE)").c_str(),
            __FILE__, __LINE__);
    }
    int32_t out = -1;
    herr = hipMemcpy(&out, ex.sampled.data<int32_t>(), sizeof(out), hipMemcpyDeviceToHost);
    if (herr != hipSuccess) {
        return Status::hip_error("mtp sample read", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    }
    if (out < 0) {
        return Status::invalid_state("MTP head produced no token", __FILE__, __LINE__);
    }
    {
        ::ps::kernel::ArgMaxTop2 t2v{};
        herr = hipMemcpy(&t2v, ex.top2.data<float>(), sizeof(t2v), hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            return Status::hip_error("mtp top2 read", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
        ex.last_top1_logit = t2v.v1;
        ex.last_top2_logit = t2v.v2;
        ex.last_top2_token = t2v.t2;
    }
    return static_cast<uint32_t>(out);
}

}  // namespace

Result<uint32_t> run_mtp_rows(
    MtpExecutor& ex,
    const bf16_t* hidden,
    const int32_t* tokens,
    uint32_t rows,
    uint32_t position_start,
    hipStream_t stream,
    ValueTraceSink* trace,
    uint32_t kv_position_start) {
    const uint32_t kv0 =
        (kv_position_start == 0xFFFFFFFFu) ? position_start : kv_position_start;
    return run_mtp_rows_on_state(ex, ex.kv_state, hidden, tokens, rows, position_start, kv0,
                                 stream, trace);
}

Result<MtpStepOutcome> mtp_forward_step(
    MtpExecutor& ex,
    MtpKvState& state,
    const bf16_t* hidden,
    const int32_t* tokens,
    uint32_t rows,
    uint32_t absolute_position_start,
    hipStream_t stream,
    ValueTraceSink* trace,
    MtpStateTraceSink* state_trace) {
    if (!state.initialized) {
        return Status::invalid_state("MTP KV state not initialized", __FILE__, __LINE__);
    }
    if (rows == 0u || rows > ex.max_rows) {
        return Status::invalid_argument("MTP rows out of range", __FILE__, __LINE__);
    }
    const uint32_t before = state.logical_length;
    const uint64_t after = static_cast<uint64_t>(before) + rows;
    if (after > static_cast<uint64_t>(state.capacity_tokens)) {
        return Status::out_of_range("MTP KV capacity overflow", __FILE__, __LINE__);
    }

    auto run = run_mtp_rows_on_state(ex, state, hidden, tokens, rows,
                                     absolute_position_start, before, stream, trace);
    if (!run.ok()) return run.status();

    Status reserve = mtp_kv_reserve(state, rows);
    if (!reserve.ok()) return reserve;

    if (state_trace != nullptr && state_trace->on_step != nullptr) {
        MtpStateStep step{};
        step.sequence_id = state.sequence_id;
        step.step = state.step_index - 1u;
        step.input_token = tokens[rows - 1u];
        step.hidden_source_index = before + rows - 1u;
        step.absolute_position = absolute_position_start + rows - 1u;
        step.kv_length_before = before;
        step.kv_length_after = state.logical_length;
        step.block_id = before / state.page_tokens;
        step.offset_in_block = before % state.page_tokens;
        auto slot = mtp_kv_physical_slot(state, before);
        step.physical_slot = slot.ok() ? slot.value() : 0xFFFFFFFFu;
        step.draft_argmax = run.value();
        step.execution_status = 0u;
        state_trace->on_step(step);
    }

    MtpStepOutcome outcome{};
    outcome.draft_token = run.value();
    outcome.kv_length_before = before;
    outcome.kv_length_after = state.logical_length;
    outcome.absolute_position = absolute_position_start + rows - 1u;
    outcome.top1_logit = ex.last_top1_logit;
    outcome.top2_logit = ex.last_top2_logit;
    outcome.top2_token = ex.last_top2_token;
    return outcome;
}

Result<uint32_t> run_mtp_head(
    MtpExecutor& ex,
    const bf16_t* hidden,
    int32_t token,
    uint32_t position,
    hipStream_t stream) {
    return run_mtp_rows(ex, hidden, &token, 1u, position, stream);
}

}  // namespace ps::qwen35::runtime
