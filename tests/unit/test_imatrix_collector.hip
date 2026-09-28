#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/quantization/imatrix/gpu_collector.h>
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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

constexpr uint32_t kRows = 4;
constexpr uint32_t kFeatures = 8;
constexpr uint32_t kRowStride = 12;

float value_at(uint32_t row, uint32_t col) {
    return static_cast<float>((row + 1u) * (col + 1u));
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no HIP device\n");
        return 0;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    auto arena_res = ps::gpu::GpuArena::create(0, 8u * 1024u * 1024u);
    if (!arena_res.ok()) {
        std::printf("FAIL: arena create\n");
        return 1;
    }
    auto arena = arena_res.release();

    const uint32_t gate_up_tag =
        ps::quantization::imatrix::imatrix_tag(3u, ps::quantization::imatrix::ImatrixSite::MlpGateUpInput);
    const uint32_t down_tag =
        ps::quantization::imatrix::imatrix_tag(3u, ps::quantization::imatrix::ImatrixSite::MlpDownInput);
    const std::vector<uint32_t> tags = {gate_up_tag, down_tag};
    const std::vector<uint64_t> k_vals = {kFeatures, kFeatures};

    ps::quantization::imatrix::GpuImatrixCollector collector;
    auto init = collector.initialize(arena, tags, k_vals);
    check(init.ok(), "collector initialize");

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "stream create");

    std::vector<::ps::bf16_t> bf16_rows(static_cast<std::size_t>(kRows) * kRowStride);
    std::vector<float> f32_rows(static_cast<std::size_t>(kRows) * kRowStride);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t c = 0; c < kRowStride; ++c) {
            const float v = c < kFeatures ? value_at(r, c) : 1.0e6f;
            bf16_rows[static_cast<std::size_t>(r) * kRowStride + c] = ::ps::bf16_t{::ps::f32_to_bf16_rne(v)};
            f32_rows[static_cast<std::size_t>(r) * kRowStride + c] = v;
        }
    }

    ::ps::bf16_t* dev_bf16 = nullptr;
    float* dev_f32 = nullptr;
    check(hipMalloc(&dev_bf16, bf16_rows.size() * sizeof(::ps::bf16_t)) == hipSuccess, "dev bf16 alloc");
    check(hipMalloc(&dev_f32, f32_rows.size() * sizeof(float)) == hipSuccess, "dev f32 alloc");
    check(hipMemcpy(dev_bf16, bf16_rows.data(), bf16_rows.size() * sizeof(::ps::bf16_t),
                    hipMemcpyHostToDevice) == hipSuccess, "bf16 upload");
    check(hipMemcpy(dev_f32, f32_rows.data(), f32_rows.size() * sizeof(float),
                    hipMemcpyHostToDevice) == hipSuccess, "f32 upload");

    auto rec_bf16 = collector.record(gate_up_tag, dev_bf16, kRowStride, kFeatures, kRows,
                                     ps::quantization::imatrix::ImatrixDType::BF16, stream);
    check(rec_bf16.ok(), "record bf16");
    auto rec_f32 = collector.record(down_tag, dev_f32, kRowStride, kFeatures, kRows,
                                    ps::quantization::imatrix::ImatrixDType::F32, stream);
    check(rec_f32.ok(), "record f32");
    auto bad = collector.record(gate_up_tag, dev_bf16, kRowStride, kFeatures + 1u, kRows,
                                ps::quantization::imatrix::ImatrixDType::BF16, stream);
    check(!bad.ok(), "k mismatch rejected");

    auto flush = collector.flush(stream);
    check(flush.ok(), "flush");

    const auto& accs = collector.accumulators();
    for (uint32_t tag : tags) {
        auto it = accs.find(tag);
        check(it != accs.end(), "accumulator present");
        if (it == accs.end()) continue;
        check(it->second.count == kRows, "accumulator count");
        bool sums_ok = true;
        for (uint32_t c = 0; c < kFeatures; ++c) {
            const double expect = 30.0 * (c + 1.0) * (c + 1.0);
            if (std::fabs(it->second.sum_sq[c] - expect) > 1e-3) sums_ok = false;
        }
        check(sums_ok, tag == gate_up_tag ? "bf16 sum_sq" : "f32 sum_sq");
    }

    for (const auto& kv : collector.diagnostics())
        check(kv.second.nonfinite_count == 0, "no nonfinite values");

    ps::gpu::discard_cleanup_result(hipFree(dev_bf16));
    ps::gpu::discard_cleanup_result(hipFree(dev_f32));
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));

    std::printf("%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
