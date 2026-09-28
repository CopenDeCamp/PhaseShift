#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/state/kv_cache_types.h>
#include <phaseshift/models/qwen35/state/paged_kv_pool.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
        std::printf("PASS: %s\n", name);
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

constexpr uint32_t kPageTokens = 16;
constexpr uint32_t kKvHeads = 4;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kLayers = 1;
constexpr uint32_t kBlocksPerHead = 8;
constexpr uint32_t kCodeBytesPerBlock = 16;

uint8_t byte_at(uint32_t tag, std::size_t i, bool is_v) {
    return static_cast<uint8_t>(
        (tag * 131u + static_cast<uint32_t>(i) * (is_v ? 7u : 3u) + 17u) & 0xFFu);
}

struct Fixture {
    std::optional<ps::gpu::GpuArena> arena;
    std::optional<ps::qwen35::PagedKVPool> pool;
};

bool make_pool(Fixture& fx, ps::qwen35::KVCacheDType dtype, uint32_t pages) {
    auto arena_result = ps::gpu::GpuArena::create(0, 64u * 1024u * 1024u);
    if (!arena_result.ok()) return false;
    fx.arena.emplace(arena_result.release());
    auto kv = ps::qwen35::PagedKVPool::create(
        *fx.arena, pages, kPageTokens, kLayers, kKvHeads, kHeadDim, dtype);
    if (!kv.ok()) return false;
    fx.pool.emplace(kv.release());
    return true;
}

void verify_reserved_bytes() {
    Fixture bf16;
    check(make_pool(bf16, ps::qwen35::KVCacheDType::BF16, 1), "bf16 pool create");
    if (bf16.pool.has_value()) {
        check(bf16.pool->reserved_bytes() == 65536u, "bf16 reserved bytes 4096/token");
    }
    Fixture fp8;
    check(make_pool(fp8, ps::qwen35::KVCacheDType::FP8_E4M3, 1), "fp8 pool create");
    if (fp8.pool.has_value()) {
        check(fp8.pool->reserved_bytes() == 33280u, "fp8 reserved bytes 2080/token");
    }
    Fixture psq4;
    check(make_pool(psq4, ps::qwen35::KVCacheDType::PSQ4_W32, 1), "psq4 pool create");
    if (psq4.pool.has_value()) {
        check(psq4.pool->reserved_bytes() == 18432u, "psq4 reserved bytes 1152/token");
        check(psq4.pool->bytes_per_page() == 18432u, "psq4 bytes per page");
        check(psq4.pool->psq4_blocks_per_head() == kBlocksPerHead, "psq4 blocks per head");
        const auto view = ps::qwen35::make_paged_kv_psq4_view(*psq4.pool);
        check(view.code_bytes_per_head == 128u, "psq4 code bytes per head");
        check(view.scale_bytes_per_head == 16u, "psq4 scale bytes per head");
        check(view.blocks_per_page == kPageTokens * kKvHeads * kBlocksPerHead,
              "psq4 blocks per page");
        check(psq4.pool->k_psq4_code_pool().dim(0) ==
                  static_cast<std::size_t>(kPageTokens * kKvHeads * kBlocksPerHead) *
                      kCodeBytesPerBlock,
              "psq4 k code pool size");
        check(psq4.pool->k_psq4_scale_pool().dim(0) ==
                  static_cast<std::size_t>(kPageTokens * kKvHeads * kBlocksPerHead),
              "psq4 k scale pool size");
    }

    Fixture psq8;
    check(make_pool(psq8, ps::qwen35::KVCacheDType::PSQ8_W32, 1), "psq8 pool create");
    if (psq8.pool.has_value()) {
        check(psq8.pool->reserved_bytes() == 34816u, "psq8 reserved bytes 2176/token");
        check(psq8.pool->bytes_per_page() == 34816u, "psq8 bytes per page");
        check(psq8.pool->psq8_blocks_per_head() == kBlocksPerHead, "psq8 blocks per head");
        const auto view = ps::qwen35::make_paged_kv_psq8_view(*psq8.pool);
        check(view.code_bytes_per_head == 256u, "psq8 code bytes per head");
        check(view.scale_bytes_per_head == 16u, "psq8 scale bytes per head");
        check(view.blocks_per_page == kPageTokens * kKvHeads * kBlocksPerHead,
              "psq8 blocks per page");
        check(psq8.pool->k_psq8_code_pool().dim(0) ==
                  static_cast<std::size_t>(kPageTokens * kKvHeads * kBlocksPerHead) *
                      ps::qwen35::kPsq8CodeBytesPerBlock,
              "psq8 k code pool size");
        check(psq8.pool->k_psq8_scale_pool().dim(0) ==
                  static_cast<std::size_t>(kPageTokens * kKvHeads * kBlocksPerHead),
              "psq8 k scale pool size");
    }
}

void verify_dtype_mismatch() {
    Fixture psq4;
    Fixture bf16;
    check(make_pool(psq4, ps::qwen35::KVCacheDType::PSQ4_W32, 2), "mismatch psq4 create");
    check(make_pool(bf16, ps::qwen35::KVCacheDType::BF16, 2), "mismatch bf16 create");
    if (!psq4.pool.has_value() || !bf16.pool.has_value()) return;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        check(false, "mismatch stream create");
        return;
    }
    auto st = psq4.pool->copy_page_from(0u, *bf16.pool, 0u, stream);
    check(!st.ok(), "psq4 copy_page_from rejects dtype mismatch");
    hipStreamDestroy(stream);
    psq4.arena->shutdown();
    bf16.arena->shutdown();
}

using TensorGetter = const ps::gpu::Tensor& (ps::qwen35::PagedKVPool::*)() const noexcept;

void verify_copy_page(ps::qwen35::KVCacheDType dtype, TensorGetter k_code_pool,
                      TensorGetter v_code_pool, TensorGetter k_scale_pool,
                      TensorGetter v_scale_pool, uint32_t code_bytes_per_block,
                      const char* label) {
    Fixture src;
    Fixture dst;
    check(make_pool(src, dtype, 2), (std::string(label) + " copy src create").c_str());
    check(make_pool(dst, dtype, 2), (std::string(label) + " copy dst create").c_str());
    if (!src.pool.has_value() || !dst.pool.has_value()) return;

    const std::size_t blocks_per_page =
        static_cast<std::size_t>(kPageTokens) * kKvHeads * kBlocksPerHead;
    const std::size_t code_bytes = blocks_per_page * code_bytes_per_block;
    const std::size_t scale_elems = blocks_per_page;

    std::vector<uint8_t> kc(code_bytes);
    std::vector<uint8_t> vc(code_bytes);
    for (std::size_t i = 0; i < code_bytes; ++i) {
        kc[i] = byte_at(3u, i, false);
        vc[i] = byte_at(3u, i, true);
    }
    std::vector<uint16_t> ks(scale_elems);
    std::vector<uint16_t> vs(scale_elems);
    for (std::size_t i = 0; i < scale_elems; ++i) {
        ks[i] = static_cast<uint16_t>(0x3F00u + i);
        vs[i] = static_cast<uint16_t>(0x3E00u + i);
    }

    const uint32_t src_page = 1u;
    const std::size_t page_blocks = blocks_per_page * src_page;
    const std::size_t page_code_off = page_blocks * code_bytes_per_block;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        check(false, "stream create");
        return;
    }

    auto* src_pool = &*src.pool;
    auto* dst_pool = &*dst.pool;
    const auto& src_k_code = (src_pool->*k_code_pool)();
    const auto& src_v_code = (src_pool->*v_code_pool)();
    const auto& src_k_scale = (src_pool->*k_scale_pool)();
    const auto& src_v_scale = (src_pool->*v_scale_pool)();

    bool seeded = true;
    seeded &= hipMemcpy(src_k_code.data<uint8_t>() + page_code_off, kc.data(),
                        code_bytes, hipMemcpyHostToDevice) == hipSuccess;
    seeded &= hipMemcpy(src_v_code.data<uint8_t>() + page_code_off, vc.data(),
                        code_bytes, hipMemcpyHostToDevice) == hipSuccess;
    seeded &= hipMemcpy(src_k_scale.data<ps::bf16_t>() + page_blocks, ks.data(),
                        scale_elems * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess;
    seeded &= hipMemcpy(src_v_scale.data<ps::bf16_t>() + page_blocks, vs.data(),
                        scale_elems * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess;
    check(seeded, (std::string(label) + " copy seed").c_str());

    auto st = dst_pool->copy_page_from(0u, *src_pool, src_page, stream);
    check(st.ok(), (std::string(label) + " copy_page_from").c_str());
    hipStreamSynchronize(stream);

    std::vector<uint8_t> got_kc(code_bytes, 0);
    std::vector<uint8_t> got_vc(code_bytes, 0);
    std::vector<uint16_t> got_ks(scale_elems, 0);
    std::vector<uint16_t> got_vs(scale_elems, 0);
    const auto& dst_k_code = (dst_pool->*k_code_pool)();
    const auto& dst_v_code = (dst_pool->*v_code_pool)();
    const auto& dst_k_scale = (dst_pool->*k_scale_pool)();
    const auto& dst_v_scale = (dst_pool->*v_scale_pool)();
    hipMemcpy(got_kc.data(), dst_k_code.data<uint8_t>(), code_bytes,
              hipMemcpyDeviceToHost);
    hipMemcpy(got_vc.data(), dst_v_code.data<uint8_t>(), code_bytes,
              hipMemcpyDeviceToHost);
    hipMemcpy(got_ks.data(), dst_k_scale.data<ps::bf16_t>(),
              scale_elems * sizeof(uint16_t), hipMemcpyDeviceToHost);
    hipMemcpy(got_vs.data(), dst_v_scale.data<ps::bf16_t>(),
              scale_elems * sizeof(uint16_t), hipMemcpyDeviceToHost);

    check(got_kc == kc, (std::string(label) + " copy K code").c_str());
    check(got_vc == vc, (std::string(label) + " copy V code").c_str());
    check(got_ks == ks, (std::string(label) + " copy K scale").c_str());
    check(got_vs == vs, (std::string(label) + " copy V scale").c_str());

    const auto view = dst_pool->k_view(0u, 0u);
    check(view.ndim() != 0 && view.dim(0) == code_bytes,
          (std::string(label) + " k_view code slice").c_str());

    hipStreamDestroy(stream);
    src.arena->shutdown();
    dst.arena->shutdown();
}

void verify_copy_page_all() {
    verify_copy_page(ps::qwen35::KVCacheDType::PSQ4_W32,
                     &ps::qwen35::PagedKVPool::k_psq4_code_pool,
                     &ps::qwen35::PagedKVPool::v_psq4_code_pool,
                     &ps::qwen35::PagedKVPool::k_psq4_scale_pool,
                     &ps::qwen35::PagedKVPool::v_psq4_scale_pool,
                     ps::qwen35::kPsq4CodeBytesPerBlock, "psq4");
    verify_copy_page(ps::qwen35::KVCacheDType::PSQ8_W32,
                     &ps::qwen35::PagedKVPool::k_psq8_code_pool,
                     &ps::qwen35::PagedKVPool::v_psq8_code_pool,
                     &ps::qwen35::PagedKVPool::k_psq8_scale_pool,
                     &ps::qwen35::PagedKVPool::v_psq8_scale_pool,
                     ps::qwen35::kPsq8CodeBytesPerBlock, "psq8");
}

void verify_create_reject() {
    auto arena_result = ps::gpu::GpuArena::create(0, 16u * 1024u * 1024u);
    check(arena_result.ok(), "reject arena create");
    if (!arena_result.ok()) return;
    auto arena = arena_result.release();
    auto bad = ps::qwen35::PagedKVPool::create(
        arena, 2u, kPageTokens, kLayers, kKvHeads, 128u, ps::qwen35::KVCacheDType::PSQ4_W32);
    check(!bad.ok(), "psq4 requires head_dim 256");
    auto bad8 = ps::qwen35::PagedKVPool::create(
        arena, 2u, kPageTokens, kLayers, kKvHeads, 128u, ps::qwen35::KVCacheDType::PSQ8_W32);
    check(!bad8.ok(), "psq8 requires head_dim 256");
    arena.shutdown();
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess) {
        std::printf("SKIP: no HIP runtime\n");
        return 77;
    }
    if (device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    verify_reserved_bytes();
    verify_dtype_mismatch();
    verify_copy_page_all();
    verify_create_reject();

    std::printf("passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
