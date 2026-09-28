#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
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

uint32_t expand_nibble(uint32_t c) {
    static const uint8_t mags[8] = {0x00u, 0x38u, 0x40u, 0x44u,
                                    0x48u, 0x4Cu, 0x50u, 0x52u};
    return static_cast<uint32_t>(mags[c & 7u]) | ((c >> 3u) ? 0x80u : 0u);
}

uint32_t ref_even(uint32_t q4) {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i)
        out |= expand_nibble((q4 >> (8 * i)) & 0xFu) << (8 * i);
    return out;
}

uint32_t ref_odd(uint32_t q4) {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i)
        out |= expand_nibble((q4 >> (8 * i + 4)) & 0xFu) << (8 * i);
    return out;
}

__global__ void expand_kernel(const uint32_t* in, uint2* out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = ps::kernel::psq4_cb10_expand(in[i]);
}

bool run_batch(const std::vector<uint32_t>& cases, const char* label) {
    const uint32_t n = static_cast<uint32_t>(cases.size());
    uint32_t* d_in = nullptr;
    uint2* d_out = nullptr;
    if (hipMalloc(&d_in, n * 4u) != hipSuccess || hipMalloc(&d_out, n * 8u) != hipSuccess) {
        std::printf("FAIL: %s alloc\n", label);
        ps::gpu::discard_cleanup_result(hipFree(d_in));
        ps::gpu::discard_cleanup_result(hipFree(d_out));
        return false;
    }
    if (hipMemcpy(d_in, cases.data(), n * 4u, hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL: %s h2d\n", label);
        ps::gpu::discard_cleanup_result(hipFree(d_in));
        ps::gpu::discard_cleanup_result(hipFree(d_out));
        return false;
    }
    const uint32_t threads = 256u;
    const uint32_t blocks = (n + threads - 1u) / threads;
    expand_kernel<<<blocks, threads>>>(d_in, d_out, n);
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL: %s sync\n", label);
        ps::gpu::discard_cleanup_result(hipFree(d_in));
        ps::gpu::discard_cleanup_result(hipFree(d_out));
        return false;
    }
    std::vector<uint2> got(n);
    if (hipMemcpy(got.data(), d_out, n * 8u, hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: %s d2h\n", label);
        ps::gpu::discard_cleanup_result(hipFree(d_in));
        ps::gpu::discard_cleanup_result(hipFree(d_out));
        return false;
    }
    ps::gpu::discard_cleanup_result(hipFree(d_in));
    ps::gpu::discard_cleanup_result(hipFree(d_out));

    bool ok = true;
    size_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (got[i].x != ref_even(cases[i]) || got[i].y != ref_odd(cases[i])) {
            ok = false;
            bad = i;
            break;
        }
    }
    if (!ok) {
        std::printf("  [diag] case=%zu q4=0x%08x got=(0x%08x,0x%08x) ref=(0x%08x,0x%08x)\n",
                    bad, cases[bad], got[bad].x, got[bad].y,
                    ref_even(cases[bad]), ref_odd(cases[bad]));
    }
    check(ok, label);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== PSQ4 codebook expansion bit-exact tests ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    {
        std::vector<uint32_t> cases;
        for (int pos = 0; pos < 8; ++pos)
            for (uint32_t c = 0u; c < 16u; ++c)
                cases.push_back(c << (4 * pos));
        run_batch(cases, "all 16 nibble codes in all 8 positions");
    }
    {
        std::vector<uint32_t> cases;
        for (uint32_t c = 0u; c < 16u; ++c) {
            uint32_t q = 0u;
            for (int i = 0; i < 8; ++i) q |= c << (4 * i);
            cases.push_back(q);
        }
        run_batch(cases, "all 8 nibbles set to the same code");
    }
    {
        std::vector<uint32_t> cases;
        for (int b = 0; b < 4; ++b)
            for (uint32_t v = 0u; v < 256u; ++v)
                cases.push_back(v << (8 * b));
        run_batch(cases, "exhaustive 0x00..0xFF in every byte position");
    }
    {
        std::vector<uint32_t> cases;
        cases.reserve(1u << 18);
        uint32_t s = 0x2468ace0u;
        for (uint32_t i = 0; i < (1u << 18); ++i) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            cases.push_back(s);
        }
        run_batch(cases, "262144 random uint32_t");
    }
    {
        std::vector<uint32_t> cases;
        for (uint32_t v = 0u; v < 256u; ++v) {
            uint32_t q = 0u;
            for (int b = 0; b < 4; ++b) q |= v << (8 * b);
            cases.push_back(q);
        }
        run_batch(cases, "uniform byte 0x00..0xFF in all bytes");
    }

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
