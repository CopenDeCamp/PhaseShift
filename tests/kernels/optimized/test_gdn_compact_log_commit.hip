#include <phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

float rndf(uint32_t& s) {
    return ((float)(int32_t)(xr(s) % 2001u) - 1000.0f) / 1000.0f;
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

constexpr uint32_t kKeyHeads = 16u;
constexpr uint32_t kHeads = 48u;
constexpr uint32_t kHeadK = 128u;
constexpr uint32_t kHeadV = 128u;
constexpr uint32_t kRows = 16u;
constexpr uint32_t kStates = 2u;
constexpr uint32_t kQk = kKeyHeads * kHeadK;
constexpr uint32_t kVdim = kHeads * kHeadV;
constexpr uint64_t kSlotStride = (uint64_t)kHeads * kHeadK * kHeadV;

struct Buf {
    float* q = nullptr;
    float* k = nullptr;
    uint16_t* v = nullptr;
    uint16_t* a = nullptr;
    uint16_t* b = nullptr;
    uint16_t* dt_bias = nullptr;
    uint16_t* a_log = nullptr;
    float* state = nullptr;
    float* out = nullptr;
    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    void* requests = nullptr;
    void free_all() {
        void* bufs[] = {q, k, v, a, b, dt_bias, a_log, state, out,
                        compact_delta, compact_k, compact_a, requests};
        for (void* p : bufs)
            if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool setup(Buf& d, uint32_t seed) {
    const size_t qk_bytes = (size_t)kRows * kQk * 4u;
    const size_t vd_bytes = (size_t)kRows * kVdim * 2u;
    const size_t g_bytes = (size_t)kRows * kHeads * 2u;
    if (hipMalloc(&d.q, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.k, qk_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.v, vd_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.a, g_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.b, g_bytes) != hipSuccess) return false;
    if (hipMalloc(&d.dt_bias, kHeads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.a_log, kHeads * 2u) != hipSuccess) return false;
    if (hipMalloc(&d.state, kStates * kSlotStride * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.out, (size_t)kRows * kVdim * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.compact_delta, (size_t)kStates * kRows * kVdim * 4u) != hipSuccess)
        return false;
    if (hipMalloc(&d.compact_k, (size_t)kStates * kRows * kQk * 4u) != hipSuccess) return false;
    if (hipMalloc(&d.compact_a, (size_t)kStates * kRows * kHeads * 4u) != hipSuccess)
        return false;
    if (hipMalloc(&d.requests, sizeof(::ps::runtime::DeviceRequestDescriptor)) != hipSuccess)
        return false;

    uint32_t s = seed;
    std::vector<float> hq((size_t)kRows * kQk), hk((size_t)kRows * kQk);
    std::vector<uint16_t> hv((size_t)kRows * kVdim), ha((size_t)kRows * kHeads),
        hb((size_t)kRows * kHeads), hdt(kHeads), hal(kHeads);
    for (float& x : hq) x = rndf(s);
    for (float& x : hk) x = rndf(s);
    for (uint16_t& x : hv) x = bf16_of(rndf(s));
    for (uint16_t& x : ha) x = bf16_of(rndf(s) * 0.5f);
    for (uint16_t& x : hb) x = bf16_of(rndf(s) * 0.5f);
    for (uint16_t& x : hdt) x = bf16_of(rndf(s) * 0.5f);
    for (uint16_t& x : hal) x = bf16_of(rndf(s) * 0.5f);
    std::vector<float> hs((size_t)kStates * kSlotStride);
    for (float& x : hs) x = rndf(s) * 0.25f;

    if (hipMemcpy(d.q, hq.data(), hq.size() * 4u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.k, hk.data(), hk.size() * 4u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.v, hv.data(), hv.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.a, ha.data(), ha.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.b, hb.data(), hb.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.dt_bias, hdt.data(), hdt.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.a_log, hal.data(), hal.size() * 2u, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d.state, hs.data(), hs.size() * 4u, hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ::ps::runtime::DeviceRequestDescriptor req{};
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = kRows;
    return hipMemcpy(d.requests, &req, sizeof(req), hipMemcpyHostToDevice) == hipSuccess;
}

::ps::kernel::GdnRecurrenceArgs make_args(const Buf& d) {
    ::ps::kernel::GdnRecurrenceArgs a;
    a.q = d.q;
    a.k = d.k;
    a.v = reinterpret_cast<const ps::bf16_t*>(d.v);
    a.a = reinterpret_cast<const ps::bf16_t*>(d.a);
    a.b = reinterpret_cast<const ps::bf16_t*>(d.b);
    a.dt_bias = reinterpret_cast<const ps::bf16_t*>(d.dt_bias);
    a.a_log = reinterpret_cast<const ps::bf16_t*>(d.a_log);
    a.recurrent_state = d.state;
    a.output = d.out;
    a.requests = reinterpret_cast<const ::ps::runtime::DeviceRequestDescriptor*>(d.requests);
    a.num_requests = 1u;
    a.actual_rows = kRows;
    a.q_row_stride = kQk;
    a.k_row_stride = kQk;
    a.v_row_stride = kVdim;
    a.a_row_stride = kHeads;
    a.b_row_stride = kHeads;
    a.output_row_stride = kVdim;
    a.state_index = 0u;
    a.max_sequences = 1u;
    a.num_gdn_states = kStates;
    a.key_heads = kKeyHeads;
    a.num_v_heads = kHeads;
    a.head_k = kHeadK;
    a.head_v = kHeadV;
    a.recurrent_slot_stride = (uint64_t)kStates * kSlotStride;
    a.recurrent_layer_stride = kSlotStride;
    a.compact_delta = d.compact_delta;
    a.compact_k = d.compact_k;
    a.compact_a = d.compact_a;
    a.compact_delta_layer_stride = (uint64_t)kRows * kVdim;
    a.compact_k_layer_stride = (uint64_t)kRows * kQk;
    a.compact_a_layer_stride = (uint64_t)kRows * kHeads;
    return a;
}

bool all_bits_equal(const std::vector<float>& x, const std::vector<float>& y, size_t& first) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::memcmp(&x[i], &y[i], 4) != 0) {
            first = i;
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    std::printf("=== GDN compact-log commit bit-exactness ===\n");
    unsetenv("PHASESHIFT_GDN_RECURRENCE_EXACT");
    unsetenv("PHASESHIFT_GDN_RECURRENCE_LOSSY");

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!::ps::kernel::gdn_recurrence_lossy_enabled()) {
        std::printf("FAIL: lossy mode expected for decode1 commit path\n");
        return 1;
    }

    Buf d;
    if (!setup(d, 20260930u)) {
        std::printf("FAIL: setup\n");
        d.free_all();
        return 1;
    }

    const size_t state_n = (size_t)kStates * kSlotStride;
    std::vector<float> s0(state_n);
    ps::gpu::discard_cleanup_result(
        hipMemcpy(s0.data(), d.state, state_n * 4u, hipMemcpyDeviceToHost));

    std::vector<std::vector<float>> ref_traj(kRows + 1u);
    for (auto& v : ref_traj) v.resize(state_n);

    bool ok = true;
    for (uint32_t r = 0u; r < kRows && ok; ++r) {
        for (uint32_t li = 0u; li < kStates && ok; ++li) {
            ::ps::kernel::GdnRecurrenceArgs a = make_args(d);
            a.state_index = li;
            a.row_override = static_cast<int32_t>(r);
            if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1(a, stream) != hipSuccess)
                ok = false;
        }
        if (ok && hipStreamSynchronize(stream) != hipSuccess) ok = false;
        if (ok)
            ok = hipMemcpy(ref_traj[r + 1u].data(), d.state, state_n * 4u,
                           hipMemcpyDeviceToHost) == hipSuccess;
    }
    check(ok, "spill chain runs");

    bool all_hist_ok = true;
    bool all_commit_ok = true;
    for (uint32_t m = 1u; m <= kRows; ++m) {
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, s0.data(), state_n * 4u, hipMemcpyHostToDevice));
        ::ps::kernel::GdnRecurrenceArgs c = make_args(d);
        c.compact_rows = m;
        if (::ps::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) != hipSuccess) {
            all_commit_ok = false;
            break;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            all_commit_ok = false;
            break;
        }
        std::vector<float> got(state_n);
        ps::gpu::discard_cleanup_result(
            hipMemcpy(got.data(), d.state, state_n * 4u, hipMemcpyDeviceToHost));
        size_t first = 0;
        if (!all_bits_equal(ref_traj[m], got, first)) {
            all_commit_ok = false;
            std::printf("  [commit] m=%u first_bad=%zu ref=%.6g got=%.6g\n", m, first,
                        (double)ref_traj[m][first], (double)got[first]);
            break;
        }
        if (m == kRows) all_hist_ok = true;
    }
    check(all_commit_ok, "commit from compact log is bit-exact vs sequential decode1");
    (void)all_hist_ok;

    std::vector<float> serial16(state_n);
    {
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, s0.data(), state_n * 4u, hipMemcpyHostToDevice));
        bool sok = true;
        for (uint32_t li = 0u; li < kStates && sok; ++li) {
            ::ps::kernel::GdnRecurrenceArgs a = make_args(d);
            a.state_index = li;
            if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1_serial(a, stream) !=
                hipSuccess)
                sok = false;
        }
        if (sok && hipStreamSynchronize(stream) != hipSuccess) sok = false;
        if (sok)
            sok = hipMemcpy(serial16.data(), d.state, state_n * 4u, hipMemcpyDeviceToHost) ==
                  hipSuccess;
        check(sok, "decode1_serial chain runs (multi-layer)");
        size_t first = 0;
        check(all_bits_equal(ref_traj[kRows], serial16, first),
              "decode1_serial matches per-row decode1 chain (multi-layer)");
    }

    {
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, s0.data(), state_n * 4u, hipMemcpyHostToDevice));
        ::ps::kernel::GdnRecurrenceArgs c = make_args(d);
        c.compact_rows = kRows;
        bool cok =
            ::ps::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) == hipSuccess;
        if (cok && hipStreamSynchronize(stream) != hipSuccess) cok = false;
        std::vector<float> got(state_n);
        if (cok)
            cok = hipMemcpy(got.data(), d.state, state_n * 4u, hipMemcpyDeviceToHost) ==
                  hipSuccess;
        size_t first = 0;
        if (cok && !all_bits_equal(serial16, got, first)) {
            cok = false;
            std::printf("  [serial-commit] first_bad=%zu ref=%.6g got=%.6g\n", first,
                        (double)serial16[first], (double)got[first]);
        }
        check(cok, "commit reproduces decode1_serial state (multi-layer, rows=16)");
    }

    constexpr uint32_t kRowsExact = 8u;
    std::vector<float> rows8(state_n);
    {
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, s0.data(), state_n * 4u, hipMemcpyHostToDevice));
        bool sok = true;
        for (uint32_t li = 0u; li < kStates && sok; ++li) {
            ::ps::kernel::GdnRecurrenceArgs a = make_args(d);
            a.state_index = li;
            a.actual_rows = kRowsExact;
            if (::ps::kernel::launch_gdn_recurrence_f32_wmma_decode_rows_exact(a, stream) !=
                hipSuccess)
                sok = false;
        }
        if (sok && hipStreamSynchronize(stream) != hipSuccess) sok = false;
        if (sok)
            sok = hipMemcpy(rows8.data(), d.state, state_n * 4u, hipMemcpyDeviceToHost) ==
                  hipSuccess;
        check(sok, "decode_rows_exact chain runs (multi-layer)");
        size_t first = 0;
        check(all_bits_equal(ref_traj[kRowsExact], rows8, first),
              "decode_rows_exact matches per-row decode1 chain (multi-layer)");
    }

    {
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, s0.data(), state_n * 4u, hipMemcpyHostToDevice));
        ::ps::kernel::GdnRecurrenceArgs c = make_args(d);
        c.compact_rows = kRowsExact;
        bool cok =
            ::ps::kernel::launch_gdn_recurrence_f32_wmma_commit(c, stream) == hipSuccess;
        if (cok && hipStreamSynchronize(stream) != hipSuccess) cok = false;
        std::vector<float> got(state_n);
        if (cok)
            cok = hipMemcpy(got.data(), d.state, state_n * 4u, hipMemcpyDeviceToHost) ==
                  hipSuccess;
        size_t first = 0;
        if (cok && !all_bits_equal(rows8, got, first)) {
            cok = false;
            std::printf("  [rows-commit] first_bad=%zu ref=%.6g got=%.6g\n", first,
                        (double)rows8[first], (double)got[first]);
        }
        check(cok, "commit reproduces decode_rows_exact state (multi-layer, rows=8)");
    }

    d.free_all();
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
