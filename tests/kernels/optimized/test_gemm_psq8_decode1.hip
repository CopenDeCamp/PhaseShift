#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        passed++;
        std::printf("PASS: %s\n", name);
    } else {
        failed++;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t xr(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

uint8_t code_byte(uint32_t& s) {
    uint8_t b = static_cast<uint8_t>(xr(s) & 0xFFu);
    if ((b & 0x7Fu) == 0x7Fu) b ^= 0x40u;
    return b;
}

uint16_t scale_bf16(uint32_t& s) {
    return static_cast<uint16_t>(0x3C00u | (xr(s) & 0x03FFu));
}

struct Buf {
    uint8_t* wc = nullptr;
    uint8_t* ws = nullptr;
    int8_t* ac = nullptr;
    float* as = nullptr;
    void* out = nullptr;
    size_t wc_bytes = 0;
    size_t ws_bytes = 0;
    size_t ac_bytes = 0;
    size_t as_bytes = 0;
};

struct Case {
    uint32_t k = 0;
    uint32_t n = 0;
    const char* name = "";
};

bool alloc_case(const Case& c, Buf& b) {
    const uint32_t kp = (c.k + 31u) & ~31u;
    const uint32_t nb = kp >> 5;
    const uint32_t tiles = (c.n + 15u) / 16u;
    b.wc_bytes = static_cast<size_t>(tiles) * kp * 16u;
    b.ws_bytes = static_cast<size_t>(tiles) * nb * 32u;
    b.ac_bytes = static_cast<size_t>(kp) * 16u;
    b.as_bytes = static_cast<size_t>(nb) * 4u * 16u;
    if (hipMalloc(&b.wc, b.wc_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.ws, b.ws_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.ac, b.ac_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.as, b.as_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.out, static_cast<size_t>(c.n) * 4u) != hipSuccess) return false;
    return true;
}

void free_case(Buf& b) {
    ps::gpu::discard_cleanup_result(hipFree(b.wc));
    ps::gpu::discard_cleanup_result(hipFree(b.ws));
    ps::gpu::discard_cleanup_result(hipFree(b.ac));
    ps::gpu::discard_cleanup_result(hipFree(b.as));
    ps::gpu::discard_cleanup_result(hipFree(b.out));
}

void fill_case(const Case& c, Buf& b) {
    uint32_t s = 0x1234567u + c.k * 131u + c.n;
    std::vector<uint8_t> hwc(b.wc_bytes);
    for (auto& v : hwc) v = code_byte(s);
    ps::gpu::discard_cleanup_result(hipMemcpy(b.wc, hwc.data(), hwc.size(), hipMemcpyHostToDevice));

    std::vector<uint8_t> hws(b.ws_bytes, 0);
    for (size_t i = 0; i + 1 < hws.size(); i += 2) {
        const uint16_t v = scale_bf16(s);
        hws[i] = static_cast<uint8_t>(v & 0xFFu);
        hws[i + 1] = static_cast<uint8_t>(v >> 8u);
    }
    ps::gpu::discard_cleanup_result(hipMemcpy(b.ws, hws.data(), hws.size(), hipMemcpyHostToDevice));

    std::vector<int8_t> hac(b.ac_bytes);
    for (auto& v : hac) v = static_cast<int8_t>(code_byte(s));
    ps::gpu::discard_cleanup_result(hipMemcpy(b.ac, hac.data(), hac.size(), hipMemcpyHostToDevice));

    std::vector<float> has(b.as_bytes / 4u, 1.0f);
    for (size_t i = 0; i < has.size(); ++i) {
        const uint32_t r = xr(s);
        has[i] = 0.5f + static_cast<float>(r % 1000u) / 1000.0f;
    }
    ps::gpu::discard_cleanup_result(hipMemcpy(b.as, has.data(), has.size() * 4u, hipMemcpyHostToDevice));
}

bool run_case_unroll(const Case& c, bool bf16, uint32_t unroll, ps::gpu::GpuArena& arena,
                     hipStream_t stream) {
    const uint32_t kp = (c.k + 31u) & ~31u;
    const uint32_t nb = kp >> 5;
    const ps::kernel::Psq8GemmOutputDType odt = bf16
        ? ps::kernel::Psq8GemmOutputDType::BF16
        : ps::kernel::Psq8GemmOutputDType::F32;
    const size_t out_bytes = static_cast<size_t>(c.n) * (bf16 ? 2u : 4u);

    Buf ref;
    Buf dec;
    auto ref_out = arena.allocate_aligned(out_bytes, 256);
    auto dec_out = arena.allocate_aligned(out_bytes, 256);
    if (!alloc_case(c, ref) || !alloc_case(c, dec) || !ref_out.ok() || !dec_out.ok()) {
        std::printf("FAIL: %s %s alloc\n", c.name, bf16 ? "bf16" : "f32");
        free_case(ref);
        free_case(dec);
        return false;
    }
    fill_case(c, ref);
    if (hipMemcpy(dec.wc, ref.wc, ref.wc_bytes, hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpy(dec.ws, ref.ws, ref.ws_bytes, hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpy(dec.ac, ref.ac, ref.ac_bytes, hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpy(dec.as, ref.as, ref.as_bytes, hipMemcpyDeviceToDevice) != hipSuccess) {
        std::printf("FAIL: %s %s d2d\n", c.name, bf16 ? "bf16" : "f32");
        free_case(ref);
        free_case(dec);
        return false;
    }
    void* ref_ptr = ref_out.release().data();
    void* dec_ptr = dec_out.release().data();
    if (hipMemset(ref_ptr, 0xAB, out_bytes) != hipSuccess ||
        hipMemset(dec_ptr, 0xAB, out_bytes) != hipSuccess) {
        std::printf("FAIL: %s %s memset\n", c.name, bf16 ? "bf16" : "f32");
        free_case(ref);
        free_case(dec);
        return false;
    }

    const ps::kernel::Psq8GemmConfig cfg{ps::kernel::Psq8GemmConfigId::RowBlock1};
    hipError_t hr = ps::kernel::launch_gemm_psq8_w8a8_wmma(
        cfg, ref.wc, ref.ws, ref.ac, ref.as, ref_ptr, odt,
        1u, c.n, kp, nb * 32u, kp, nb * 4u, c.n, stream);
    hipError_t hd = ps::kernel::launch_gemm_psq8_w8a8_wmma_decode1(
        dec.wc, dec.ws, reinterpret_cast<const uint8_t*>(dec.ac), dec.as, dec_ptr, odt,
        1u, c.n, kp, nb * 32u, kp, nb * 4u, c.n, unroll, stream);
    if (hr != hipSuccess || hd != hipSuccess) {
        std::printf("FAIL: %s %s launch ref=%s dec=%s\n", c.name, bf16 ? "bf16" : "f32",
                    hipGetErrorString(hr), hipGetErrorString(hd));
        free_case(ref);
        free_case(dec);
        return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: %s %s sync\n", c.name, bf16 ? "bf16" : "f32");
        free_case(ref);
        free_case(dec);
        return false;
    }

    std::vector<uint8_t> a(out_bytes);
    std::vector<uint8_t> b(out_bytes);
    if (hipMemcpy(a.data(), ref_ptr, out_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(b.data(), dec_ptr, out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: %s %s d2h\n", c.name, bf16 ? "bf16" : "f32");
        free_case(ref);
        free_case(dec);
        return false;
    }

    bool ok = true;
    size_t bad = 0;
    for (size_t i = 0; i < out_bytes; ++i) {
        if (a[i] != b[i]) { ok = false; bad = i; break; }
    }
    char label[192];
    std::snprintf(label, sizeof(label), "decode1 vs rowblock1 %-22s k=%-6u n=%-7u %-4s u=%u",
                  c.name, c.k, c.n, bf16 ? "bf16" : "f32", unroll);
    if (!ok) {
        std::printf("  [diag] first byte diff at %zu: ref=0x%02x dec=0x%02x\n", bad, a[bad], b[bad]);
    }
    check(ok, label);
    free_case(ref);
    free_case(dec);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== PSQ8 decode1 bit-exact vs RowBlock1 ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result = ps::gpu::GpuArena::create(0, 4096ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    const Case production[] = {
        {5120u, 6144u, "prod-k5120-n6144"},
        {5120u, 10240u, "prod-k5120-n10240"},
        {6144u, 5120u, "prod-k6144-n5120"},
        {17408u, 5120u, "prod-k17408-n5120"},
        {5120u, 248320u, "prod-lmhead"},
    };
    const Case small[] = {
        {32u, 16u, "small-k32-n16"},
        {64u, 32u, "small-k64-n32"},
        {128u, 64u, "small-k128-n64"},
        {512u, 256u, "small-k512-n256"},
        {1024u, 512u, "small-k1024-n512"},
    };

    const uint32_t unrolls[] = {0u, 2u, 4u, 8u, 16u};
    for (const auto& c : production) {
        for (uint32_t u : unrolls) run_case_unroll(c, false, u, arena, stream);
        for (uint32_t u : unrolls) run_case_unroll(c, true, u, arena, stream);
    }
    for (const auto& c : small) {
        run_case_unroll(c, false, 0u, arena, stream);
        run_case_unroll(c, true, 0u, arena, stream);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
