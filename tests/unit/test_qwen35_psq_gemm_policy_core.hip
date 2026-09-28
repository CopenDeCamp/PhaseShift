#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
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

uint8_t e4_fixture(uint32_t h) {
    uint8_t b = static_cast<uint8_t>(h & 0xFFu);
    if ((b & 0x78u) == 0x78u) b = static_cast<uint8_t>(b & ~0x40u);
    return b;
}

uint16_t bf16_unit(uint32_t& s) {
    return static_cast<uint16_t>(0x3C00u | (xr(s) & 0x03FFu));
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint32_t align16(uint32_t v) { return (v + 15u) & ~15u; }

struct Layout {
    uint32_t k_padded;
    uint32_t nb;
    uint32_t tiles;
    uint32_t atiles;
    uint32_t wcode_bytes;
    uint32_t wscale_bytes;
    uint32_t acode_bytes;
    uint32_t ascale_floats;
    uint32_t ascale_bytes;
};

Layout make_layout(uint32_t n, uint32_t k, uint32_t rows, bool psq8) {
    Layout l{};
    l.k_padded = (k + 31u) & ~31u;
    l.nb = l.k_padded >> 5u;
    l.tiles = (n + 15u) / 16u;
    l.atiles = (rows + 15u) / 16u;
    l.wcode_bytes = l.tiles * l.k_padded * (psq8 ? 16u : 8u);
    l.wscale_bytes = l.tiles * l.nb * 32u;
    l.acode_bytes = l.atiles * l.k_padded * 16u;
    l.ascale_floats = ((rows + 15u) / 16u) * 16u * (align16(l.nb * 4u) / 4u);
    l.ascale_bytes = l.ascale_floats * 4u;
    return l;
}

struct Buffers {
    uint8_t* wc = nullptr;
    uint8_t* ws = nullptr;
    uint8_t* ac = nullptr;
    float* as = nullptr;
    void* out = nullptr;
    size_t out_bytes = 0;
};

bool alloc_buffers(const Layout& l, uint32_t n, uint32_t rows, bool bf16, Buffers& b) {
    b.out_bytes = static_cast<size_t>(rows) * n * (bf16 ? 2u : 4u);
    if (hipMalloc(&b.wc, l.wcode_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.ws, l.wscale_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.ac, l.acode_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.as, l.ascale_bytes) != hipSuccess) return false;
    if (hipMalloc(&b.out, b.out_bytes) != hipSuccess) return false;
    return true;
}

void free_buffers(Buffers& b) {
    ps::gpu::discard_cleanup_result(hipFree(b.wc));
    ps::gpu::discard_cleanup_result(hipFree(b.ws));
    ps::gpu::discard_cleanup_result(hipFree(b.ac));
    ps::gpu::discard_cleanup_result(hipFree(b.as));
    ps::gpu::discard_cleanup_result(hipFree(b.out));
}

void fill_buffers(const Layout& l, bool psq8, Buffers& b) {
    uint32_t s = 0x51ed270bu;
    std::vector<uint8_t> hwc(l.wcode_bytes);
    for (auto& v : hwc) v = psq8 ? e4_fixture(xr(s)) : static_cast<uint8_t>(xr(s) & 0xFFu);
    ps::gpu::discard_cleanup_result(hipMemcpy(b.wc, hwc.data(), hwc.size(), hipMemcpyHostToDevice));

    std::vector<uint8_t> hws(l.wscale_bytes, 0);
    for (size_t i = 0; i + 1 < hws.size(); i += 2) {
        const uint16_t v = bf16_unit(s);
        hws[i] = static_cast<uint8_t>(v & 0xFFu);
        hws[i + 1] = static_cast<uint8_t>(v >> 8u);
    }
    ps::gpu::discard_cleanup_result(hipMemcpy(b.ws, hws.data(), hws.size(), hipMemcpyHostToDevice));

    std::vector<uint8_t> hac(l.acode_bytes);
    for (auto& v : hac) v = e4_fixture(xr(s));
    ps::gpu::discard_cleanup_result(hipMemcpy(b.ac, hac.data(), hac.size(), hipMemcpyHostToDevice));

    std::vector<float> has(l.ascale_floats, 1.0f);
    for (size_t i = 0; i < has.size(); ++i)
        has[i] = 0.5f + static_cast<float>(xr(s) % 1000u) / 1000.0f;
    ps::gpu::discard_cleanup_result(hipMemcpy(b.as, has.data(), has.size() * 4u, hipMemcpyHostToDevice));
}

uint64_t hash_case(uint32_t n, uint32_t k, uint32_t rows, bool psq8, bool bf16,
                   bool capture, hipStream_t stream) {
    const Layout l = make_layout(n, k, rows, psq8);
    Buffers b;
    if (!alloc_buffers(l, n, rows, bf16, b)) {
        std::printf("FAIL: alloc n=%u k=%u rows=%u\n", n, k, rows);
        free_buffers(b);
        return 0u;
    }
    fill_buffers(l, psq8, b);
    const uint32_t wss = l.nb * 32u;
    const uint32_t ass = align16(l.nb * 4u);
    hipError_t herr;
    if (psq8) {
        herr = ps::kernel::launch_gemm_psq8_w8a8_wmma_auto(
            ps::kernel::Psq8GemmConfig{},
            b.wc, b.ws, reinterpret_cast<const int8_t*>(b.ac), b.as, b.out,
            bf16 ? ps::kernel::Psq8GemmOutputDType::BF16
                 : ps::kernel::Psq8GemmOutputDType::F32,
            rows, n, l.k_padded, wss, l.k_padded, ass, n, stream);
    } else {
        herr = ps::kernel::launch_gemm_psq4_w4a8_wmma_auto(
            ps::kernel::Psq4GemmConfig{},
            b.wc, b.ws, reinterpret_cast<const int8_t*>(b.ac), b.as, b.out,
            bf16 ? ps::kernel::Psq4GemmOutputDType::BF16
                 : ps::kernel::Psq4GemmOutputDType::F32,
            rows, n, l.k_padded, wss, l.k_padded, ass, n, stream);
    }
    if (herr != hipSuccess) {
        std::printf("FAIL: launch n=%u k=%u rows=%u psq8=%d: %s\n", n, k, rows,
                    psq8 ? 1 : 0, hipGetErrorString(herr));
        free_buffers(b);
        return 0u;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL: sync n=%u k=%u rows=%u\n", n, k, rows);
        free_buffers(b);
        return 0u;
    }
    if (capture) {
        free_buffers(b);
        return 0u;
    }
    std::vector<uint8_t> got(b.out_bytes);
    if (hipMemcpy(got.data(), b.out, b.out_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL: d2h n=%u k=%u rows=%u\n", n, k, rows);
        free_buffers(b);
        return 0u;
    }
    const uint64_t h = fnv1a(got.data(), got.size());
    free_buffers(b);
    return h;
}

struct Golden {
    bool psq8;
    uint32_t n;
    uint32_t k;
    uint32_t rows;
    bool bf16;
    uint64_t hash;
};

static constexpr Golden kGolden[] = {
    {0, 17408, 5120, 1, 1, 0x83f830ff76b555c8ull},
    {0, 17408, 5120, 1, 0, 0x3196a8e5a393eb8dull},
    {0, 17408, 5120, 2, 1, 0x0ec5eae3162e7b23ull},
    {0, 17408, 5120, 2, 0, 0x83b2af7fb03667edull},
    {0, 17408, 5120, 3, 1, 0x72f106b9d26c0ffcull},
    {0, 17408, 5120, 3, 0, 0x38c493f34f31b4dfull},
    {0, 17408, 5120, 7, 1, 0xab24fc841121926eull},
    {0, 17408, 5120, 7, 0, 0x65f6985bbf29c9cfull},
    {0, 17408, 5120, 8, 1, 0x068e28fc198b53d5ull},
    {0, 17408, 5120, 8, 0, 0x824a2d6355e77542ull},
    {0, 5120, 17408, 1, 1, 0xea45d3d9458f9654ull},
    {0, 5120, 17408, 1, 0, 0xf93e50e5cd6497a6ull},
    {0, 5120, 17408, 2, 1, 0x1ed3144fd69b8195ull},
    {0, 5120, 17408, 2, 0, 0x6e9c8b4d87faa31full},
    {0, 5120, 17408, 3, 1, 0xa0aeabf0c0386334ull},
    {0, 5120, 17408, 3, 0, 0x41cbd548c396e9e8ull},
    {0, 5120, 17408, 7, 1, 0xb92ee4e94bd2fbe4ull},
    {0, 5120, 17408, 7, 0, 0xcf752e40f3e4ac85ull},
    {0, 5120, 17408, 8, 1, 0xcb86a93e16868b4aull},
    {0, 5120, 17408, 8, 0, 0xa3c2a2f9e80ba884ull},
    {1, 17408, 5120, 1, 1, 0x2baba84501a4d7deull},
    {1, 17408, 5120, 1, 0, 0x32ff63620cebda0eull},
    {1, 17408, 5120, 2, 1, 0x470a148371df562cull},
    {1, 17408, 5120, 2, 0, 0x20b433cc0e9b5387ull},
    {1, 17408, 5120, 3, 1, 0xe4d13e6099ea48b9ull},
    {1, 17408, 5120, 3, 0, 0x6c3fced4c9feb910ull},
    {1, 17408, 5120, 7, 1, 0x5602b0d0fde4f7eaull},
    {1, 17408, 5120, 7, 0, 0xcd3868228c36ddadull},
    {1, 17408, 5120, 8, 1, 0x89992fdd329af6e4ull},
    {1, 17408, 5120, 8, 0, 0x1b1dd59a22ba6fe5ull},
    {1, 5120, 17408, 1, 1, 0x47e6597f7ebfa49full},
    {1, 5120, 17408, 1, 0, 0x223d51c6f6ce66ccull},
    {1, 5120, 17408, 2, 1, 0x4a3faeb2a1e99fc7ull},
    {1, 5120, 17408, 2, 0, 0x1f761e026a9987f5ull},
    {1, 5120, 17408, 3, 1, 0x5827da669ca7922aull},
    {1, 5120, 17408, 3, 0, 0xbe8831d9f1137deaull},
    {1, 5120, 17408, 7, 1, 0x5b647335bc203bc6ull},
    {1, 5120, 17408, 7, 0, 0x34228455be309aecull},
    {1, 5120, 17408, 8, 1, 0x74a317a256bd4430ull},
    {1, 5120, 17408, 8, 0, 0xb4ae12786a4dd12cull},
};

}  // namespace

int main(int argc, char** argv) {
    const bool capture = argc > 1 && std::strcmp(argv[1], "capture") == 0;
    std::printf("=== PSQ GEMM policy core exactness ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    const uint32_t shapes[][2] = {{17408u, 5120u}, {5120u, 17408u}};
    const uint32_t rows_list[] = {1u, 2u, 3u, 7u, 8u};
    const bool encodings[] = {false, true};
    const bool dtypes[] = {true, false};

    if (capture) {
        std::printf("static constexpr Golden kGolden[] = {\n");
        for (bool psq8 : encodings)
            for (const auto& sh : shapes)
                for (uint32_t rows : rows_list)
                    for (bool bf16 : dtypes) {
                        const uint64_t h = hash_case(sh[0], sh[1], rows, psq8, bf16, false, stream);
                        std::printf("    {%d, %u, %u, %u, %d, 0x%016llxull},\n",
                                    psq8 ? 1 : 0, sh[0], sh[1], rows, bf16 ? 1 : 0,
                                    static_cast<unsigned long long>(h));
                    }
        std::printf("};\n");
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return 0;
    }

    for (const Golden& g : kGolden) {
        const uint64_t h = hash_case(g.n, g.k, g.rows, g.psq8, g.bf16, false, stream);
        char label[192];
        std::snprintf(label, sizeof(label), "policy %-4s n=%-6u k=%-6u rows=%u %-4s",
                      g.psq8 ? "psq8" : "psq4", g.n, g.k, g.rows, g.bf16 ? "bf16" : "f32");
        if (h != g.hash) {
            std::printf("  [diag] got=0x%016llx want=0x%016llx\n",
                        static_cast<unsigned long long>(h),
                        static_cast<unsigned long long>(g.hash));
        }
        check(h == g.hash, label);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
