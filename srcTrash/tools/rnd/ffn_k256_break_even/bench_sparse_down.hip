#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/fusion_policy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kThreads = 32u;
constexpr uint32_t kGroups = 68u;
constexpr uint32_t kK32PerGroup = 8u;

enum Mode {
    kModeDense = 0,
    kModeIdeal = 1,
    kModeRuntime = 2,
    kModeMaskOnly = 3,
    kModeProdDense = 4,
};

__device__ __forceinline__ int32x2_t psq4_cb10_pair(uint32_t q4) {
    const uint2 e = ps::kernel::psq4_cb10_expand(q4);
    return int32x2_t{static_cast<int32_t>(e.x), static_cast<int32_t>(e.y)};
}

__device__ __forceinline__ void psq4_load_fragment(const uint8_t* wrow, const uint16_t* wsrow,
                                                   uint32_t ib, int32x2_t& w0, int32x2_t& w1,
                                                   float& ws) {
    const uint64_t packed =
        *reinterpret_cast<const uint64_t*>(wrow + static_cast<size_t>(ib) * 256u);
    const int32x2_t lo = psq4_cb10_pair(static_cast<uint32_t>(packed));
    const int32x2_t hi = psq4_cb10_pair(static_cast<uint32_t>(packed >> 32u));
    w0 = int32x2_t{lo[0], hi[0]};
    w1 = int32x2_t{lo[1], hi[1]};
    ws = __bfloat162float(__ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
}

__device__ __forceinline__ float8_t psq4_wmma(const int32x2_t& a0, const int32x2_t& a1,
                                              const int32x2_t& w0, const int32x2_t& w1) {
    float8_t c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
    return c;
}

template <ps::kernel::Psq4GemmOutputDType kOT, int kMode>
__global__ void __launch_bounds__(kThreads) psq4_sparse_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t* __restrict__ skip_mask,
    const uint32_t ideal_skip,
    uint32_t* __restrict__ sink) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;
    const uint32_t o_self = blockIdx.x * kTileOut + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 8u) +
        static_cast<size_t>(o_self & 15u) * 16u + static_cast<size_t>(k_group) * 8u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(
        weight_scales + static_cast<size_t>(o_self >> 4) * weight_scale_stride +
        static_cast<size_t>(o_self & 15u) * 2u);

    const ps::kernel::CanonicalA8ActivationPolicy act{
        reinterpret_cast<const uint8_t*>(activation_codes), 0u, activation_scales, 4u};
    const uint8_t* arow = act.row_base(0u, k_group);

    uint32_t m0 = 0u, m1 = 0u, m2 = 0u;
    if constexpr (kMode == kModeRuntime || kMode == kModeMaskOnly) {
        const uint32_t* m = skip_mask + static_cast<size_t>(blockIdx.x >> 2) * 3u;
        m0 = m[0];
        m1 = m[1];
        m2 = m[2];
    }
    uint32_t mtest = 0u;

    float acc0 = 0.0f;
    for (uint32_t g = 0; g < kGroups; ++g) {
        if constexpr (kMode == kModeIdeal) {
            if (g < ideal_skip)
                continue;
        } else if constexpr (kMode == kModeRuntime || kMode == kModeMaskOnly) {
            const uint32_t word = (g < 32u) ? m0 : (g < 64u) ? m1 : m2;
            const bool skip = ((word >> (g & 31u)) & 1u) != 0u;
            if constexpr (kMode == kModeMaskOnly) {
                mtest |= skip ? 1u : 0u;
            } else {
                if (skip)
                    continue;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < kK32PerGroup; ++j) {
            const uint32_t ib = g * kK32PerGroup + j;
            int32x2_t w0 = {};
            int32x2_t w1 = {};
            float ws = 0.0f;
            psq4_load_fragment(wrow, wsrow, ib, w0, w1, ws);
            int32x2_t a0 = {};
            int32x2_t a1 = {};
            act.load_fragment(arow, ib, a0, a1);
            const float8_t c = psq4_wmma(a0, a1, w0, w1);
            acc0 = __builtin_fmaf(c[0], ws, acc0);
        }
    }
    if constexpr (kMode == kModeMaskOnly) {
        if (tid == 0u)
            sink[blockIdx.x] = mtest;
    }
    if (k_group == 0u) {
        const float v = acc0 * act.row_scale(0u);
        if constexpr (kOT == ps::kernel::Psq4GemmOutputDType::BF16) {
            reinterpret_cast<::ps::bf16_t*>(output)[o_self] =
                ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            reinterpret_cast<float*>(output)[o_self] = v;
        }
    }
}

template <ps::kernel::Psq8GemmOutputDType kOT, int kMode>
__global__ void __launch_bounds__(kThreads) psq8_sparse_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t* __restrict__ skip_mask,
    const uint32_t ideal_skip,
    uint32_t* __restrict__ sink) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;
    const uint32_t o_self = blockIdx.x * kTileOut + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o_self & 15u) * 16u + static_cast<size_t>(k_group) * 8u;
    const uint16_t* wsrow = reinterpret_cast<const uint16_t*>(
        weight_scales + static_cast<size_t>(o_self >> 4) * weight_scale_stride +
        static_cast<size_t>(o_self & 15u) * 2u);

    const ps::kernel::CanonicalA8ActivationPolicy act{
        reinterpret_cast<const uint8_t*>(activation_codes), 0u, activation_scales, 4u};
    const uint8_t* arow = act.row_base(0u, k_group);

    uint32_t m0 = 0u, m1 = 0u, m2 = 0u;
    if constexpr (kMode == kModeRuntime || kMode == kModeMaskOnly) {
        const uint32_t* m = skip_mask + static_cast<size_t>(blockIdx.x >> 2) * 3u;
        m0 = m[0];
        m1 = m[1];
        m2 = m[2];
    }
    uint32_t mtest = 0u;

    float acc0 = 0.0f;
    for (uint32_t g = 0; g < kGroups; ++g) {
        if constexpr (kMode == kModeIdeal) {
            if (g < ideal_skip)
                continue;
        } else if constexpr (kMode == kModeRuntime || kMode == kModeMaskOnly) {
            const uint32_t word = (g < 32u) ? m0 : (g < 64u) ? m1 : m2;
            const bool skip = ((word >> (g & 31u)) & 1u) != 0u;
            if constexpr (kMode == kModeMaskOnly) {
                mtest |= skip ? 1u : 0u;
            } else {
                if (skip)
                    continue;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < kK32PerGroup; ++j) {
            const uint32_t ib = g * kK32PerGroup + j;
            const int32x2_t w0 = *reinterpret_cast<const int32x2_t*>(
                wrow + static_cast<size_t>(ib) * 512u);
            const int32x2_t w1 = *reinterpret_cast<const int32x2_t*>(
                wrow + static_cast<size_t>(ib) * 512u + 256u);
            int32x2_t a0 = {};
            int32x2_t a1 = {};
            act.load_fragment(arow, ib, a0, a1);
            const float ws = __bfloat162float(
                __ushort_as_bfloat16(wsrow[static_cast<size_t>(ib) * 16u]));
            float8_t c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
            acc0 = __builtin_fmaf(c[0], ws, acc0);
        }
    }
    if constexpr (kMode == kModeMaskOnly) {
        if (tid == 0u)
            sink[blockIdx.x] = mtest;
    }
    if (k_group == 0u) {
        const float v = acc0 * act.row_scale(0u);
        if constexpr (kOT == ps::kernel::Psq8GemmOutputDType::BF16) {
            reinterpret_cast<::ps::bf16_t*>(output)[o_self] =
                ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            reinterpret_cast<float*>(output)[o_self] = v;
        }
    }
}

uint16_t f32_to_bf16_bits(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t round = 0x7fffu + ((x >> 16) & 1u);
    x += round;
    return static_cast<uint16_t>(x >> 16);
}

float bf16_bits_to_f32(uint16_t b) {
    const uint32_t x = static_cast<uint32_t>(b) << 16u;
    float f = 0.0f;
    std::memcpy(&f, &x, 4);
    return f;
}

struct Stats {
    double p10 = 0.0, p50 = 0.0, p90 = 0.0, mean = 0.0, min = 0.0, max = 0.0;
};

Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    double sum = 0.0;
    for (double x : v)
        sum += x;
    return Stats{v[n / 10], v[n / 2], v[(n * 9) / 10], sum / static_cast<double>(n), v.front(),
                 v.back()};
}

struct Timing {
    std::vector<double> samples;
    std::vector<double> repeat_p50;
};

struct Entry {
    std::string pattern;
    uint32_t skip = 0;
    Stats prod_dense, dense, ideal, runtime, maskonly;
    std::vector<double> prod_dense_rep, runtime_rep;
    double dense_vs_prod = 0.0;
    double runtime_vs_ref = 0.0;
    double runtime_vs_dense = 0.0;
};

struct Args {
    std::string kind = "psq4";
    uint32_t n = 5120;
    uint32_t k = 17408;
    int device = 0;
    int global_warmup = 4000;
    int warmup = 200;
    int launches = 50;
    int samples = 100;
    int repeats = 3;
    int copies = 4;
    std::string out = "bench.json";
    std::string mask_dir;
    std::vector<uint32_t> skips{0, 7, 10, 14, 16, 17, 20, 27, 28, 30};
};

double maxabs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

class Runner {
public:
    explicit Runner(const Args& a) : args_(a) {}
    bool setup();
    void run();

private:
    Args args_;
    uint32_t n16_ = 0;
    uint32_t n64_ = 0;
    uint32_t cpb_ = 8;
    int copies_ = 4;
    size_t code_bytes_ = 0;
    size_t scale_bytes_ = 0;
    uint32_t scale_stride_ = 0;
    size_t act_bytes_ = 0;

    uint8_t* w_codes_ = nullptr;
    uint8_t* w_scales_ = nullptr;
    uint8_t* r_codes_ = nullptr;
    uint8_t* r_scales_ = nullptr;
    uint8_t* a_codes_ = nullptr;
    float* a_scales_ = nullptr;
    void* out_prod_ = nullptr;
    void* out_ref_ = nullptr;
    void* out_dense_ = nullptr;
    void* out_ideal_ = nullptr;
    void* out_runtime_ = nullptr;
    void* out_maskonly_ = nullptr;
    uint32_t* mask_ = nullptr;
    uint32_t* sink_ = nullptr;

    std::vector<uint8_t> h_codes_;
    std::vector<uint8_t> h_scales_;
    std::vector<std::string> patterns_;
    std::vector<Entry> entries_;

    std::vector<uint32_t> build_mask(uint32_t skip, const std::string& pattern);
    void upload_mask(const std::vector<uint32_t>& m);
    void zero_skipped_groups();
    const uint8_t* codes_at(int c) const {
        return w_codes_ + static_cast<size_t>(c % copies_) * code_bytes_;
    }
    const uint8_t* scales_at(int c) const {
        return w_scales_ + static_cast<size_t>(c % copies_) * scale_bytes_;
    }
    void launch_prod_ptr(const uint8_t* codes, const uint8_t* scales, void* out);
    void launch_dense_prod(void* out, int copy);
    void launch_mode(int mode, uint32_t skip, void* out, int copy);
    Timing time_mode(int mode, uint32_t skip, void* out);
    std::vector<float> read_output(void* buf);
    std::string json() const;
};

void Runner::launch_prod_ptr(const uint8_t* codes, const uint8_t* scales, void* out) {
    if (args_.kind == "psq8") {
        ps::kernel::launch_gemm_psq8_w8a8_wmma_decode1(
            codes, scales, a_codes_, a_scales_, out, ps::kernel::Psq8GemmOutputDType::BF16,
            1u, args_.n, args_.k, scale_stride_, 0u, 4u, args_.n, 0u, nullptr);
    } else {
        ps::kernel::launch_gemm_psq4_w4a8_wmma_decode1(
            codes, scales, a_codes_, a_scales_, out, ps::kernel::Psq4GemmOutputDType::BF16,
            1u, args_.n, args_.k, scale_stride_, 0u, 4u, args_.n, 0u, nullptr);
    }
}

void Runner::launch_dense_prod(void* out, int copy) {
    launch_prod_ptr(codes_at(copy), scales_at(copy), out);
}

void Runner::launch_mode(int mode, uint32_t skip, void* out, int copy) {
    if (mode == kModeProdDense) {
        launch_dense_prod(out, copy);
        return;
    }
    const dim3 grid(n16_);
    const uint8_t* codes = codes_at(copy);
    const uint8_t* scales = scales_at(copy);
    const uint32_t* m = mask_;
    uint32_t* sink = sink_;
    auto psq4 = [&](auto tag) {
        constexpr int kM = decltype(tag)::value;
        psq4_sparse_kernel<ps::kernel::Psq4GemmOutputDType::BF16, kM>
            <<<grid, kThreads>>>(codes, scales, a_codes_, a_scales_, out, args_.k,
                                 scale_stride_, m, skip, sink);
    };
    auto psq8 = [&](auto tag) {
        constexpr int kM = decltype(tag)::value;
        psq8_sparse_kernel<ps::kernel::Psq8GemmOutputDType::BF16, kM>
            <<<grid, kThreads>>>(codes, scales, a_codes_, a_scales_, out, args_.k,
                                 scale_stride_, m, skip, sink);
    };
    if (args_.kind == "psq8") {
        if (mode == kModeDense) psq8(std::integral_constant<int, kModeDense>{});
        else if (mode == kModeIdeal) psq8(std::integral_constant<int, kModeIdeal>{});
        else if (mode == kModeRuntime) psq8(std::integral_constant<int, kModeRuntime>{});
        else psq8(std::integral_constant<int, kModeMaskOnly>{});
    } else {
        if (mode == kModeDense) psq4(std::integral_constant<int, kModeDense>{});
        else if (mode == kModeIdeal) psq4(std::integral_constant<int, kModeIdeal>{});
        else if (mode == kModeRuntime) psq4(std::integral_constant<int, kModeRuntime>{});
        else psq4(std::integral_constant<int, kModeMaskOnly>{});
    }
}

bool Runner::setup() {
    if (args_.k % (kGroups * 32u) != 0u || args_.n % 64u != 0u)
        return false;
    cpb_ = (args_.kind == "psq8") ? 16u : 8u;
    copies_ = std::max(1, args_.copies);
    n16_ = args_.n / kTileOut;
    n64_ = args_.n / 64u;
    scale_stride_ = args_.k;
    code_bytes_ = static_cast<size_t>(n16_) * args_.k * cpb_;
    scale_bytes_ = static_cast<size_t>(n16_) * scale_stride_;
    act_bytes_ = static_cast<size_t>(args_.k) * 16u + 1024u;

    h_codes_.resize(code_bytes_);
    h_scales_.resize(scale_bytes_);
    std::mt19937 rng(20260924u);
    for (auto& b : h_codes_)
        b = static_cast<uint8_t>(rng() & 0xffu);
    for (size_t i = 0; i + 1 < h_scales_.size(); i += 2) {
        const float s = 0.005f + 0.05f * static_cast<float>(rng() % 1000u) / 1000.0f;
        const uint16_t bits = f32_to_bf16_bits(s);
        h_scales_[i] = static_cast<uint8_t>(bits & 0xffu);
        h_scales_[i + 1] = static_cast<uint8_t>(bits >> 8u);
    }
    std::vector<uint8_t> h_act(act_bytes_, 0u);
    for (size_t i = 0; i < act_bytes_; ++i)
        h_act[i] = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(rng() % 33u) - 16));
    std::vector<float> h_ascale(1, 0.02f);

    if (hipSetDevice(args_.device) != hipSuccess)
        return false;
    auto alloc = [](void** p, size_t bytes) { return hipMalloc(p, bytes) == hipSuccess; };
    if (!alloc(reinterpret_cast<void**>(&w_codes_), code_bytes_ * copies_)) return false;
    if (!alloc(reinterpret_cast<void**>(&w_scales_), scale_bytes_ * copies_)) return false;
    if (!alloc(reinterpret_cast<void**>(&r_codes_), code_bytes_)) return false;
    if (!alloc(reinterpret_cast<void**>(&r_scales_), scale_bytes_)) return false;
    if (!alloc(reinterpret_cast<void**>(&a_codes_), act_bytes_)) return false;
    if (!alloc(reinterpret_cast<void**>(&a_scales_), sizeof(float))) return false;
    if (!alloc(&out_prod_, static_cast<size_t>(args_.n) * 4u)) return false;
    if (!alloc(&out_ref_, static_cast<size_t>(args_.n) * 4u)) return false;
    if (!alloc(&out_dense_, static_cast<size_t>(args_.n) * 4u)) return false;
    if (!alloc(&out_ideal_, static_cast<size_t>(args_.n) * 4u)) return false;
    if (!alloc(&out_runtime_, static_cast<size_t>(args_.n) * 4u)) return false;
    if (!alloc(&out_maskonly_, static_cast<size_t>(args_.n) * 4u)) return false;
    if (!alloc(reinterpret_cast<void**>(&mask_), static_cast<size_t>(n64_) * 3u * 4u))
        return false;
    if (!alloc(reinterpret_cast<void**>(&sink_), static_cast<size_t>(n16_) * 4u))
        return false;

    if (hipMemcpy(w_codes_, h_codes_.data(), code_bytes_, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(w_scales_, h_scales_.data(), scale_bytes_, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    for (int c = 1; c < copies_; ++c) {
        hipMemcpy(w_codes_ + static_cast<size_t>(c) * code_bytes_, h_codes_.data(), code_bytes_,
                  hipMemcpyHostToDevice);
        hipMemcpy(w_scales_ + static_cast<size_t>(c) * scale_bytes_, h_scales_.data(),
                  scale_bytes_, hipMemcpyHostToDevice);
    }
    if (hipMemcpy(a_codes_, h_act.data(), act_bytes_, hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMemcpy(a_scales_, h_ascale.data(), sizeof(float), hipMemcpyHostToDevice) != hipSuccess)
        return false;
    return true;
}

void Runner::upload_mask(const std::vector<uint32_t>& m) {
    hipMemcpy(mask_, m.data(), static_cast<size_t>(n64_) * 3u * 4u, hipMemcpyHostToDevice);
}

std::vector<uint32_t> Runner::build_mask(uint32_t skip, const std::string& pattern) {
    std::vector<uint32_t> mask(n64_ * 3u, 0u);
    if (pattern == "oracle") {
        const std::string path =
            args_.mask_dir + "/oracle_mask_k" + std::to_string(skip) + ".txt";
        std::ifstream f(path);
        if (!f) {
            std::fprintf(stderr, "mask file missing: %s\n", path.c_str());
            std::exit(2);
        }
        for (uint32_t t = 0; t < n64_; ++t) {
            std::string line;
            std::getline(f, line);
            std::istringstream ss(line);
            uint32_t g = 0;
            while (ss >> g)
                mask[t * 3u + (g >> 5u)] |= (1u << (g & 31u));
        }
        return mask;
    }
    for (uint32_t t = 0; t < n64_; ++t) {
        if (pattern == "contiguous") {
            for (uint32_t g = 0; g < skip; ++g)
                mask[t * 3u + (g >> 5u)] |= (1u << (g & 31u));
        } else {
            std::vector<uint32_t> groups(kGroups);
            for (uint32_t g = 0; g < kGroups; ++g)
                groups[g] = g;
            std::mt19937 rng(1234u + t * 131u + skip * 7919u);
            std::shuffle(groups.begin(), groups.end(), rng);
            for (uint32_t i = 0; i < skip; ++i) {
                const uint32_t g = groups[i];
                mask[t * 3u + (g >> 5u)] |= (1u << (g & 31u));
            }
        }
    }
    return mask;
}

void Runner::zero_skipped_groups() {
    std::vector<uint8_t> h_rc = h_codes_;
    std::vector<uint8_t> h_rs = h_scales_;
    std::vector<uint32_t> host_mask(n64_ * 3u);
    hipMemcpy(host_mask.data(), mask_, static_cast<size_t>(n64_) * 3u * 4u,
              hipMemcpyDeviceToHost);
    const size_t group_code_bytes = static_cast<size_t>(kK32PerGroup) * cpb_ * 32u;
    const size_t group_scale_bytes = static_cast<size_t>(kK32PerGroup) * 32u;
    for (uint32_t t = 0; t < n64_; ++t) {
        for (uint32_t g = 0; g < kGroups; ++g) {
            const bool skipg = ((host_mask[t * 3u + (g >> 5u)] >> (g & 31u)) & 1u) != 0u;
            if (!skipg)
                continue;
            for (uint32_t u = t * 4u; u < t * 4u + 4u; ++u) {
                const size_t cbase = static_cast<size_t>(u) * args_.k * cpb_ +
                                     static_cast<size_t>(g) * group_code_bytes;
                std::memset(h_rc.data() + cbase, 0, group_code_bytes);
                const size_t sbase = static_cast<size_t>(u) * scale_stride_ +
                                     static_cast<size_t>(g) * group_scale_bytes;
                std::memset(h_rs.data() + sbase, 0, group_scale_bytes);
            }
        }
    }
    hipMemcpy(r_codes_, h_rc.data(), code_bytes_, hipMemcpyHostToDevice);
    hipMemcpy(r_scales_, h_rs.data(), scale_bytes_, hipMemcpyHostToDevice);
}

Timing Runner::time_mode(int mode, uint32_t skip, void* out) {
    Timing t;
    int it = 0;
    hipEvent_t e0 = nullptr, e1 = nullptr;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    for (int rep = 0; rep < args_.repeats; ++rep) {
        for (int i = 0; i < args_.warmup; ++i)
            launch_mode(mode, skip, out, it++);
        hipDeviceSynchronize();
        std::vector<double> rs;
        for (int s = 0; s < args_.samples; ++s) {
            hipEventRecord(e0);
            for (int l = 0; l < args_.launches; ++l)
                launch_mode(mode, skip, out, it++);
            hipEventRecord(e1);
            hipEventSynchronize(e1);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, e0, e1);
            const double us =
                static_cast<double>(ms) * 1000.0 / static_cast<double>(args_.launches);
            t.samples.push_back(us);
            rs.push_back(us);
        }
        std::sort(rs.begin(), rs.end());
        t.repeat_p50.push_back(rs[rs.size() / 2]);
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    return t;
}

std::vector<float> Runner::read_output(void* buf) {
    std::vector<uint16_t> tmp(args_.n);
    hipMemcpy(tmp.data(), buf, static_cast<size_t>(args_.n) * 2u, hipMemcpyDeviceToHost);
    std::vector<float> out(args_.n);
    for (uint32_t i = 0; i < args_.n; ++i)
        out[i] = bf16_bits_to_f32(tmp[i]);
    return out;
}

std::string Runner::json() const {
    auto st = [](const Stats& s) {
        std::ostringstream ss;
        ss << "{\"p10\": " << s.p10 << ", \"p50\": " << s.p50 << ", \"p90\": " << s.p90
           << ", \"mean\": " << s.mean << ", \"min\": " << s.min << ", \"max\": " << s.max << "}";
        return ss.str();
    };
    std::ostringstream o;
    o << "{\n";
    o << " \"meta\": {\"kind\": \"" << args_.kind << "\", \"n\": " << args_.n
      << ", \"k\": " << args_.k << ", \"device\": " << args_.device
      << ", \"warmup\": " << args_.warmup << ", \"launches\": " << args_.launches
      << ", \"samples\": " << args_.samples << ", \"repeats\": " << args_.repeats
      << ", \"copies\": " << copies_
      << ", \"working_set_mb\": "
      << (static_cast<double>((code_bytes_ + scale_bytes_) * static_cast<size_t>(copies_)) / 1e6)
      << ", \"mask_dir\": \"" << args_.mask_dir << "\"},\n";
    o << " \"results\": [\n";
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry& e = entries_[i];
        if (i)
            o << ",\n";
        o << "  {\"pattern\": \"" << e.pattern << "\", \"skip\": " << e.skip
          << ", \"actual_skip\": " << (static_cast<double>(e.skip) / kGroups)
          << ", \"prod_dense\": " << st(e.prod_dense) << ", \"dense\": " << st(e.dense)
          << ", \"ideal\": " << st(e.ideal) << ", \"runtime\": " << st(e.runtime)
          << ", \"maskonly\": " << st(e.maskonly) << ", \"prod_dense_repeat_p50\": [";
        for (size_t j = 0; j < e.prod_dense_rep.size(); ++j)
            o << (j ? ", " : "") << e.prod_dense_rep[j];
        o << "], \"runtime_repeat_p50\": [";
        for (size_t j = 0; j < e.runtime_rep.size(); ++j)
            o << (j ? ", " : "") << e.runtime_rep[j];
        o << "], \"correctness\": {\"dense_vs_prod_maxabs\": " << e.dense_vs_prod
          << ", \"runtime_vs_zeroed_ref_maxabs\": " << e.runtime_vs_ref
          << ", \"runtime_vs_dense_skip0\": " << e.runtime_vs_dense << "}}";
    }
    o << "\n ]\n}\n";
    return o.str();
}

void Runner::run() {
    for (int i = 0; i < args_.global_warmup; ++i)
        launch_mode(kModeDense, 0u, out_dense_, i);
    hipDeviceSynchronize();

    launch_dense_prod(out_prod_, 0);
    hipDeviceSynchronize();
    const auto prod_ref = read_output(out_prod_);

    launch_mode(kModeDense, 0u, out_dense_, 0);
    hipDeviceSynchronize();
    const auto mine_dense = read_output(out_dense_);
    const double dvp = maxabs_diff(prod_ref, mine_dense);

    Timing prod_t = time_mode(kModeProdDense, 0u, out_prod_);
    Timing dense_t = time_mode(kModeDense, 0u, out_dense_);

    patterns_.push_back("contiguous");
    patterns_.push_back("random");
    if (!args_.mask_dir.empty())
        patterns_.push_back("oracle");

    for (const auto& pat : patterns_) {
        for (uint32_t skip : args_.skips) {
            const std::vector<uint32_t> mask = build_mask(skip, pat);
            upload_mask(mask);
            zero_skipped_groups();
            launch_prod_ptr(r_codes_, r_scales_, out_ref_);
            hipDeviceSynchronize();
            const auto ref = read_output(out_ref_);

            launch_mode(kModeRuntime, skip, out_runtime_, 0);
            hipDeviceSynchronize();
            const auto rt = read_output(out_runtime_);
            const double rvr = maxabs_diff(ref, rt);

            double rvd = -1.0;
            if (skip == 0)
                rvd = maxabs_diff(rt, mine_dense);

            Timing ideal_t = time_mode(kModeIdeal, skip, out_ideal_);
            Timing runtime_t = time_mode(kModeRuntime, skip, out_runtime_);
            Timing maskonly_t = time_mode(kModeMaskOnly, skip, out_maskonly_);

            Entry e;
            e.pattern = pat;
            e.skip = skip;
            e.prod_dense = stats_of(prod_t.samples);
            e.dense = stats_of(dense_t.samples);
            e.ideal = stats_of(ideal_t.samples);
            e.runtime = stats_of(runtime_t.samples);
            e.maskonly = stats_of(maskonly_t.samples);
            e.prod_dense_rep = prod_t.repeat_p50;
            e.runtime_rep = runtime_t.repeat_p50;
            e.dense_vs_prod = dvp;
            e.runtime_vs_ref = rvr;
            e.runtime_vs_dense = rvd;
            entries_.push_back(e);
            std::fprintf(stderr, "[bench] %s skip=%u p50 dense=%.2f ideal=%.2f runtime=%.2f refdiff=%.3g\n",
                         pat.c_str(), skip, e.dense.p50, e.ideal.p50, e.runtime.p50, rvr);
        }
    }
    std::ofstream f(args_.out);
    f << json();
    std::fprintf(stderr, "wrote %s (%zu entries)\n", args_.out.c_str(), entries_.size());
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (k == "--kind") a.kind = next();
        else if (k == "--n") a.n = static_cast<uint32_t>(std::stoul(next()));
        else if (k == "--k") a.k = static_cast<uint32_t>(std::stoul(next()));
        else if (k == "--device") a.device = std::stoi(next());
        else if (k == "--global-warmup") a.global_warmup = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else if (k == "--launches") a.launches = std::stoi(next());
        else if (k == "--samples") a.samples = std::stoi(next());
        else if (k == "--repeats") a.repeats = std::stoi(next());
        else if (k == "--copies") a.copies = std::stoi(next());
        else if (k == "--out") a.out = next();
        else if (k == "--mask-dir") a.mask_dir = next();
        else if (k == "--skips") {
            a.skips.clear();
            std::istringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ','))
                a.skips.push_back(static_cast<uint32_t>(std::stoul(tok)));
        }
    }
    Runner r(a);
    if (!r.setup()) {
        std::fprintf(stderr, "setup failed\n");
        return 1;
    }
    r.run();
    return 0;
}
