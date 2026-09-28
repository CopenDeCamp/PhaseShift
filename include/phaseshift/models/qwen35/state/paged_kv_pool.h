#pragma once

#include <phaseshift/models/qwen35/state/paged_types.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace ps {
namespace qwen35 {

struct PagedKVBatchDeviceView {
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;

    const uint32_t* block_tables = nullptr;
    uint32_t block_table_stride = 0;

    const SequenceSlotId* request_slots = nullptr;
    const uint32_t* query_start_loc = nullptr;
    const uint32_t* prefix_lens = nullptr;
    const uint32_t* sequence_lens = nullptr;

    uint32_t num_requests = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t attention_layer = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
};

struct PagedKVBatchFP8DeviceView {
    fp8e4m3_storage_t* k_pool = nullptr;
    fp8e4m3_storage_t* v_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;

    const uint32_t* block_tables = nullptr;
    uint32_t block_table_stride = 0;

    const SequenceSlotId* request_slots = nullptr;
    const uint32_t* query_start_loc = nullptr;
    const uint32_t* prefix_lens = nullptr;
    const uint32_t* sequence_lens = nullptr;

    uint32_t num_requests = 0;

    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t attention_layer = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
};

struct PagedKVPoolBF16View {
    bf16_t* k_pool = nullptr;
    bf16_t* v_pool = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
};

struct PagedKVPoolFP8View {
    fp8e4m3_storage_t* k_pool = nullptr;
    fp8e4m3_storage_t* v_pool = nullptr;
    float* k_scale_pool = nullptr;
    float* v_scale_pool = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t elems_per_token = 0;
    uint32_t elems_per_page = 0;
    uint32_t elems_per_layer = 0;
};

struct PagedKVPoolPSQ4View {
    uint8_t* k_code_pool = nullptr;
    uint8_t* v_code_pool = nullptr;
    bf16_t* k_scale_pool = nullptr;
    bf16_t* v_scale_pool = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t blocks_per_head = 0;
    uint32_t code_bytes_per_head = 0;
    uint32_t scale_bytes_per_head = 0;
    uint32_t blocks_per_token = 0;
    uint32_t blocks_per_page = 0;
    uint32_t blocks_per_layer = 0;
};

struct PagedKVPoolPSQ8View {
    uint8_t* k_code_pool = nullptr;
    uint8_t* v_code_pool = nullptr;
    bf16_t* k_scale_pool = nullptr;
    bf16_t* v_scale_pool = nullptr;
    uint32_t page_tokens = 0;
    uint32_t num_pages = 0;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t blocks_per_head = 0;
    uint32_t code_bytes_per_head = 0;
    uint32_t scale_bytes_per_head = 0;
    uint32_t blocks_per_token = 0;
    uint32_t blocks_per_page = 0;
    uint32_t blocks_per_layer = 0;
};

class PagedKVPool {
 public:
    static Result<PagedKVPool> create(
        gpu::GpuArena& arena,
        uint32_t num_pages,
        uint32_t page_tokens,
        uint32_t kv_heads,
        uint32_t head_dim);

    static Result<PagedKVPool> create(
        gpu::GpuArena& arena,
        uint32_t num_pages,
        uint32_t page_tokens,
        uint32_t num_attention_layers,
        uint32_t kv_heads,
        uint32_t head_dim);

    static Result<PagedKVPool> create(
        gpu::GpuArena& arena,
        uint32_t num_pages,
        uint32_t page_tokens,
        uint32_t num_attention_layers,
        uint32_t kv_heads,
        uint32_t head_dim,
        KVCacheDType dtype);

    Result<PageId> allocate();

    Status release(PageId page);

    Status release_many(const std::vector<PageId>& pages);

    Status validate_release_many(const std::vector<PageId>& pages) const;

    gpu::Tensor k_view(uint32_t layer, PageId page) const;

    gpu::Tensor v_view(uint32_t layer, PageId page) const;

    Status copy_page_from(
        PageId dst_page,
        const PagedKVPool& src,
        PageId src_page,
        hipStream_t stream);

    const gpu::Tensor& k_pool() const noexcept { return k_pool_; }
    const gpu::Tensor& v_pool() const noexcept { return v_pool_; }
    const gpu::Tensor& k_scale_pool() const noexcept { return k_scale_pool_; }
    const gpu::Tensor& v_scale_pool() const noexcept { return v_scale_pool_; }
    const gpu::Tensor& k_psq4_code_pool() const noexcept { return k_psq4_code_pool_; }
    const gpu::Tensor& v_psq4_code_pool() const noexcept { return v_psq4_code_pool_; }
    const gpu::Tensor& k_psq4_scale_pool() const noexcept { return k_psq4_scale_pool_; }
    const gpu::Tensor& v_psq4_scale_pool() const noexcept { return v_psq4_scale_pool_; }
    const gpu::Tensor& k_psq8_code_pool() const noexcept { return k_psq8_code_pool_; }
    const gpu::Tensor& v_psq8_code_pool() const noexcept { return v_psq8_code_pool_; }
    const gpu::Tensor& k_psq8_scale_pool() const noexcept { return k_psq8_scale_pool_; }
    const gpu::Tensor& v_psq8_scale_pool() const noexcept { return v_psq8_scale_pool_; }
    uint32_t psq4_blocks_per_head() const noexcept { return psq4_blocks_per_head_; }
    uint32_t psq8_blocks_per_head() const noexcept { return psq8_blocks_per_head_; }
    KVCacheDType dtype() const noexcept { return dtype_; }

    uint32_t num_pages() const noexcept { return num_pages_; }
    uint32_t page_tokens() const noexcept { return page_tokens_; }
    uint32_t num_attention_layers() const noexcept { return num_attention_layers_; }
    uint32_t kv_heads() const noexcept { return kv_heads_; }
    uint32_t head_dim() const noexcept { return head_dim_; }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    std::size_t bytes_per_page() const noexcept {
        return reserved_bytes_ / static_cast<std::size_t>(num_pages_);
    }

    uint32_t num_used_pages() const noexcept { return used_count_; }

    uint32_t num_free_pages() const noexcept { return num_pages_ - used_count_; }

    std::size_t used_bytes() const noexcept {
        return static_cast<std::size_t>(used_count_) * bytes_per_page();
    }

    std::size_t free_bytes() const noexcept {
        return reserved_bytes_ - used_bytes();
    }

 private:
    gpu::Tensor k_pool_;
    gpu::Tensor v_pool_;
    gpu::Tensor k_scale_pool_;
    gpu::Tensor v_scale_pool_;
    gpu::Tensor k_psq4_code_pool_;
    gpu::Tensor v_psq4_code_pool_;
    gpu::Tensor k_psq4_scale_pool_;
    gpu::Tensor v_psq4_scale_pool_;
    gpu::Tensor k_psq8_code_pool_;
    gpu::Tensor v_psq8_code_pool_;
    gpu::Tensor k_psq8_scale_pool_;
    gpu::Tensor v_psq8_scale_pool_;
    KVCacheDType dtype_ = KVCacheDType::BF16;

    uint32_t num_pages_;
    uint32_t page_tokens_;
    uint32_t num_attention_layers_;
    uint32_t kv_heads_;
    uint32_t head_dim_;
    uint32_t psq4_blocks_per_head_ = 0;
    uint32_t psq8_blocks_per_head_ = 0;

    std::vector<bool> in_use_;
    std::vector<uint32_t> free_list_;

    std::size_t reserved_bytes_;
    uint32_t used_count_;

    PagedKVPool() = default;
};

inline PagedKVPoolBF16View make_paged_kv_bf16_view(const PagedKVPool& pool) {
    PagedKVPoolBF16View v{};
    v.k_pool = pool.k_pool().data<bf16_t>();
    v.v_pool = pool.v_pool().data<bf16_t>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.elems_per_token = pool.kv_heads() * pool.head_dim();
    v.elems_per_page = v.page_tokens * v.elems_per_token;
    v.elems_per_layer = v.num_pages * v.elems_per_page;
    return v;
}

inline PagedKVPoolFP8View make_paged_kv_fp8_view(const PagedKVPool& pool) {
    PagedKVPoolFP8View v{};
    v.k_pool = pool.k_pool().data<fp8e4m3_storage_t>();
    v.v_pool = pool.v_pool().data<fp8e4m3_storage_t>();
    v.k_scale_pool = pool.k_scale_pool().data<float>();
    v.v_scale_pool = pool.v_scale_pool().data<float>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.elems_per_token = pool.kv_heads() * pool.head_dim();
    v.elems_per_page = v.page_tokens * v.elems_per_token;
    v.elems_per_layer = v.num_pages * v.elems_per_page;
    return v;
}

inline PagedKVPoolPSQ4View make_paged_kv_psq4_view(const PagedKVPool& pool) {
    PagedKVPoolPSQ4View v{};
    v.k_code_pool = pool.k_psq4_code_pool().data<uint8_t>();
    v.v_code_pool = pool.v_psq4_code_pool().data<uint8_t>();
    v.k_scale_pool = pool.k_psq4_scale_pool().data<bf16_t>();
    v.v_scale_pool = pool.v_psq4_scale_pool().data<bf16_t>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.blocks_per_head = pool.psq4_blocks_per_head();
    v.code_bytes_per_head = v.blocks_per_head * kPsq4CodeBytesPerBlock;
    v.scale_bytes_per_head = v.blocks_per_head * kPsq4ScaleBytesPerBlock;
    v.blocks_per_token = v.kv_heads * v.blocks_per_head;
    v.blocks_per_page = v.page_tokens * v.blocks_per_token;
    v.blocks_per_layer = v.num_pages * v.blocks_per_page;
    return v;
}

inline PagedKVPoolPSQ8View make_paged_kv_psq8_view(const PagedKVPool& pool) {
    PagedKVPoolPSQ8View v{};
    v.k_code_pool = pool.k_psq8_code_pool().data<uint8_t>();
    v.v_code_pool = pool.v_psq8_code_pool().data<uint8_t>();
    v.k_scale_pool = pool.k_psq8_scale_pool().data<bf16_t>();
    v.v_scale_pool = pool.v_psq8_scale_pool().data<bf16_t>();
    v.page_tokens = pool.page_tokens();
    v.num_pages = pool.num_pages();
    v.num_attention_layers = pool.num_attention_layers();
    v.kv_heads = pool.kv_heads();
    v.head_dim = pool.head_dim();
    v.blocks_per_head = pool.psq8_blocks_per_head();
    v.code_bytes_per_head = v.blocks_per_head * kPsq8CodeBytesPerBlock;
    v.scale_bytes_per_head = v.blocks_per_head * kPsq8ScaleBytesPerBlock;
    v.blocks_per_token = v.kv_heads * v.blocks_per_head;
    v.blocks_per_page = v.page_tokens * v.blocks_per_token;
    v.blocks_per_layer = v.num_pages * v.blocks_per_page;
    return v;
}

}
}
