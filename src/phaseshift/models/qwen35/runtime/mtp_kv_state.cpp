#include <phaseshift/models/qwen35/runtime/mtp_kv_state.h>

#include <cstring>
#include <limits>

namespace ps::qwen35::runtime {

Result<MtpKvState> create_mtp_kv_state(
    gpu::GpuArena& arena,
    const MtpKvConfig& config,
    uint32_t sequence_id,
    hipStream_t stream) {
    if (config.num_pages == 0u || config.page_tokens == 0u ||
        config.num_attention_layers == 0u || config.kv_heads == 0u ||
        config.head_dim == 0u) {
        return Status::invalid_argument("MTP KV config has zero geometry", __FILE__, __LINE__);
    }
    if (config.dtype != KVCacheDType::BF16) {
        return Status::unsupported("MTP KV correctness path is BF16 only", __FILE__, __LINE__);
    }

    MtpKvState state;
    state.sequence_id = sequence_id;
    state.page_tokens = config.page_tokens;
    state.kv_heads = config.kv_heads;
    state.head_dim = config.head_dim;
    state.dtype = config.dtype;

    {
        auto pool_res = PagedKVPool::create(
            arena, config.num_pages, config.page_tokens, config.num_attention_layers,
            config.kv_heads, config.head_dim, config.dtype);
        if (!pool_res.ok()) return pool_res.status();
        state.pool.emplace(pool_res.release());
    }
    {
        auto slots_res = SequenceSlotPool::create(arena, 1u, config.num_pages);
        if (!slots_res.ok()) return slots_res.status();
        state.slots.emplace(slots_res.release());
        auto handle = state.slots->allocate_handle();
        if (!handle.ok()) return handle.status();
        state.handle = handle.value();
    }

    state.block_to_page.reserve(config.num_pages);
    for (uint32_t b = 0u; b < config.num_pages; ++b) {
        auto page = state.pool->allocate();
        if (!page.ok()) return page.status();
        Status st = state.slots->set_block(state.handle.slot, b, page.value(), stream);
        if (!st.ok()) return st;
        state.block_to_page.push_back(page.value());
    }

    const uint64_t capacity =
        static_cast<uint64_t>(config.num_pages) * static_cast<uint64_t>(config.page_tokens);
    if (capacity > std::numeric_limits<uint32_t>::max()) {
        return Status::out_of_range("MTP KV capacity exceeds uint32", __FILE__, __LINE__);
    }
    state.capacity_tokens = static_cast<uint32_t>(capacity);
    state.logical_length = 0u;
    state.step_index = 0u;
    state.initialized = true;
    return state;
}

Status mtp_kv_reset(MtpKvState& state, hipStream_t stream) {
    (void)stream;
    if (!state.initialized) {
        return Status::invalid_state("MTP KV state not initialized", __FILE__, __LINE__);
    }
    state.logical_length = 0u;
    state.step_index = 0u;
    return Status::make_ok();
}

Status mtp_kv_reserve(MtpKvState& state, uint32_t rows) {
    if (!state.initialized) {
        return Status::invalid_state("MTP KV state not initialized", __FILE__, __LINE__);
    }
    if (rows == 0u) {
        return Status::invalid_argument("MTP KV reserve rows must be > 0", __FILE__, __LINE__);
    }
    const uint64_t after = static_cast<uint64_t>(state.logical_length) + rows;
    if (after > static_cast<uint64_t>(state.capacity_tokens)) {
        return Status::out_of_range("MTP KV capacity overflow", __FILE__, __LINE__);
    }
    state.logical_length = static_cast<uint32_t>(after);
    ++state.step_index;
    return Status::make_ok();
}

Status mtp_kv_shutdown(MtpKvState& state) noexcept {
    if (state.slots.has_value() && ::ps::runtime::request_handle_valid(state.handle)) {
        Status released = state.slots->release_handle(state.handle);
        (void)released;
        state.handle = ::ps::runtime::RequestHandle{};
    }
    state.block_to_page.clear();
    state.initialized = false;
    state.logical_length = 0u;
    state.step_index = 0u;
    return Status::make_ok();
}

Result<uint32_t> mtp_kv_physical_slot(const MtpKvState& state, uint32_t logical_index) {
    if (!state.initialized) {
        return Status::invalid_state("MTP KV state not initialized", __FILE__, __LINE__);
    }
    if (logical_index >= state.logical_length) {
        return Status::out_of_range("MTP logical index beyond logical length", __FILE__, __LINE__);
    }
    const uint32_t block = logical_index / state.page_tokens;
    const uint32_t off = logical_index % state.page_tokens;
    if (block >= state.block_to_page.size()) {
        return Status::out_of_range("MTP block index out of range", __FILE__, __LINE__);
    }
    return state.block_to_page[block] * state.page_tokens + off;
}

::ps::kernel::ModelDispatchStateView mtp_kv_state_view(const MtpKvState& state) {
    ::ps::kernel::ModelDispatchStateView view{};
    if (!state.pool.has_value() || !state.slots.has_value()) return view;
    view.block_table = state.slots->device_view();
    view.kv_dtype = static_cast<uint32_t>(state.dtype);
    if (state.dtype == KVCacheDType::FP8_E4M3) {
        view.kv_fp8 = make_paged_kv_fp8_view(*state.pool);
    } else if (state.dtype == KVCacheDType::BF16) {
        view.kv_bf16 = make_paged_kv_bf16_view(*state.pool);
    }
    return view;
}

Status mtp_kv_dump_canonical(
    const MtpKvState& state,
    uint32_t length,
    bf16_t* k_out,
    bf16_t* v_out,
    hipStream_t stream) {
    if (!state.initialized || !state.pool.has_value() || !state.slots.has_value()) {
        return Status::invalid_state("MTP KV state not initialized", __FILE__, __LINE__);
    }
    if (k_out == nullptr || v_out == nullptr) {
        return Status::invalid_argument("MTP KV dump outputs must be non-null", __FILE__, __LINE__);
    }
    if (length > state.logical_length) {
        return Status::out_of_range("MTP KV dump length beyond logical length", __FILE__, __LINE__);
    }
    if (state.dtype != KVCacheDType::BF16) {
        return Status::unsupported("MTP KV canonical dump is BF16 only", __FILE__, __LINE__);
    }

    const uint32_t ept = state.kv_heads * state.head_dim;
    const uint32_t page_tokens = state.page_tokens;
    const uint32_t elems_per_page = page_tokens * ept;
    const size_t pool_elems = static_cast<size_t>(state.pool->num_pages()) * elems_per_page;

    std::vector<bf16_t> host_k(pool_elems);
    std::vector<bf16_t> host_v(pool_elems);
    hipError_t err = hipMemcpyAsync(
        host_k.data(), state.pool->k_pool().data<bf16_t>(), pool_elems * sizeof(bf16_t),
        hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        return Status::hip_error("MTP K pool copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(
        host_v.data(), state.pool->v_pool().data<bf16_t>(), pool_elems * sizeof(bf16_t),
        hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        return Status::hip_error("MTP V pool copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        return Status::hip_error("MTP KV dump sync", hipGetErrorString(err), __FILE__, __LINE__);
    }

    for (uint32_t t = 0u; t < length; ++t) {
        const uint32_t block = t / page_tokens;
        const uint32_t off = t % page_tokens;
        const uint32_t page = state.block_to_page[block];
        const size_t src = static_cast<size_t>(page) * elems_per_page +
                           static_cast<size_t>(off) * ept;
        std::memcpy(k_out + static_cast<size_t>(t) * ept, host_k.data() + src,
                    ept * sizeof(bf16_t));
        std::memcpy(v_out + static_cast<size_t>(t) * ept, host_v.data() + src,
                    ept * sizeof(bf16_t));
    }
    return Status::make_ok();
}

}  // namespace ps::qwen35::runtime
