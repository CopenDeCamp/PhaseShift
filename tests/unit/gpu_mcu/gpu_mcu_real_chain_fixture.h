#pragma once

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/rmsnorm.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace gpu_mcu_chain {

namespace mcu = ps::runtime::gpu_mcu;

namespace kernel = ps::kernel;

inline constexpr uint32_t kRows = 1u;
inline constexpr uint32_t kFeatures = 5120u;
inline constexpr uint32_t kKp = 5120u;
inline constexpr uint32_t kN = 1024u;
inline constexpr uint32_t kNb = kKp / 32u;
inline constexpr uint32_t kTiles = (kN + 15u) / 16u;
inline constexpr float kEps = 1e-6f;

inline constexpr uint32_t kWeightScaleStride = kNb * 32u;

inline uint32_t xr(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

inline uint8_t code_byte(uint32_t& s) {
    uint8_t b = static_cast<uint8_t>(xr(s) & 0xFFu);
    if ((b & 0x7Fu) == 0x7Fu) b ^= 0x40u;
    return b;
}

inline uint16_t scale_bf16(uint32_t& s) {
    return static_cast<uint16_t>(0x3C00u | (xr(s) & 0x03FFu));
}

inline ps::bf16_t bf16_from_float(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, 4);
    return ps::bf16_t{static_cast<uint16_t>((ui + 0x8000u) >> 16)};
}

inline float f32_from_bf16(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

inline void build_rms_input(std::vector<uint16_t>& out) {
    out.assign(static_cast<size_t>(kRows) * kFeatures, 0u);
    for (uint32_t c = 0; c < kFeatures; ++c) {
        const uint32_t h = c * 2654435761u + 12345u;
        const int base = static_cast<int>(h % 2001u) - 1000;
        out[c] = bf16_from_float(static_cast<float>(base) / 1000.0f).data;
    }
}

inline void build_rms_weight(std::vector<uint16_t>& out) {
    out.assign(kFeatures, 0u);
    for (uint32_t c = 0; c < kFeatures; ++c) {
        const uint32_t h = c * 104729u + 11u;
        const int base = static_cast<int>(h % 2001u) - 1000;
        out[c] = bf16_from_float(static_cast<float>(base) / 1000.0f).data;
    }
}

struct Workspace {
    void* input = nullptr;
    void* rms_weight = nullptr;
    void* rms_out = nullptr;
    void* a8 = nullptr;
    int8_t* codes = nullptr;
    float* scales = nullptr;
    void* w_codes = nullptr;
    void* w_scales = nullptr;
    void* final_out = nullptr;

    static constexpr size_t kInputBytes =
        static_cast<size_t>(kRows) * kFeatures * sizeof(uint16_t);
    static constexpr size_t kRmsWeightBytes = kFeatures * sizeof(uint16_t);
    static constexpr size_t kWCodesBytes = static_cast<size_t>(kTiles) * kKp * 8u;
    static constexpr size_t kWScalesBytes = static_cast<size_t>(kTiles) * kNb * 32u;
    static constexpr size_t kOutBytes = static_cast<size_t>(kN) * sizeof(uint16_t);

    bool allocate() {
        const auto layout =
            ::ps::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, kRows);
        if (hipMalloc(&input, kInputBytes) != hipSuccess ||
            hipMalloc(&rms_weight, kRmsWeightBytes) != hipSuccess ||
            hipMalloc(&rms_out, kInputBytes) != hipSuccess ||
            hipMalloc(&a8, layout.total_bytes) != hipSuccess ||
            hipMalloc(&w_codes, kWCodesBytes) != hipSuccess ||
            hipMalloc(&w_scales, kWScalesBytes) != hipSuccess ||
            hipMalloc(&final_out, kOutBytes) != hipSuccess) {
            return false;
        }
        codes = reinterpret_cast<int8_t*>(
            static_cast<uint8_t*>(a8) + layout.codes_offset);
        scales = reinterpret_cast<float*>(
            static_cast<uint8_t*>(a8) + layout.scales_offset);
        return true;
    }

    void fill_inputs() {
        std::vector<uint16_t> in;
        std::vector<uint16_t> wt;
        build_rms_input(in);
        build_rms_weight(wt);
        (void)hipMemcpy(input, in.data(), kInputBytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(rms_weight, wt.data(), kRmsWeightBytes,
                        hipMemcpyHostToDevice);

        uint32_t s = 0x1234567u + kKp * 131u + kN;
        std::vector<uint8_t> hwc(kWCodesBytes);
        for (auto& v : hwc) v = code_byte(s);
        (void)hipMemcpy(w_codes, hwc.data(), hwc.size(), hipMemcpyHostToDevice);

        std::vector<uint8_t> hws(kWScalesBytes, 0);
        for (size_t i = 0; i + 1 < hws.size(); i += 2) {
            const uint16_t v = scale_bf16(s);
            hws[i] = static_cast<uint8_t>(v & 0xFFu);
            hws[i + 1] = static_cast<uint8_t>(v >> 8u);
        }
        (void)hipMemcpy(w_scales, hws.data(), hws.size(), hipMemcpyHostToDevice);

        (void)hipMemset(rms_out, 0xAB, kInputBytes);
        (void)hipMemset(a8, 0xAB, layout_total());
        (void)hipMemset(final_out, 0xAB, kOutBytes);
    }

    static size_t layout_total() {
        return ::ps::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, kRows)
            .total_bytes;
    }

    static uint32_t code_row_stride() {
        return ::ps::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, kRows)
            .code_row_stride_bytes;
    }

    static uint32_t scale_row_stride() {
        return ::ps::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, kRows)
            .scale_row_stride_bytes;
    }

    void free_all() {
        (void)hipFree(input);
        (void)hipFree(rms_weight);
        (void)hipFree(rms_out);
        (void)hipFree(a8);
        (void)hipFree(w_codes);
        (void)hipFree(w_scales);
        (void)hipFree(final_out);
        input = nullptr;
        rms_weight = nullptr;
        rms_out = nullptr;
        a8 = nullptr;
        codes = nullptr;
        scales = nullptr;
        w_codes = nullptr;
        w_scales = nullptr;
        final_out = nullptr;
    }
};

inline hipError_t run_host_rmsnorm(Workspace& ws, hipStream_t stream) {
    return kernel::launch_rmsnorm(
        ws.input, kernel::RmsNormDataType::BF16, ws.rms_weight,
        kernel::RmsNormDataType::BF16, kernel::RmsNormWeightLayout::PER_FEATURE,
        kernel::RmsNormWeightMode::ONE_PLUS, ws.rms_out,
        kernel::RmsNormDataType::BF16, kRows, kFeatures, kFeatures, kFeatures,
        kFeatures, kEps, stream);
}

inline hipError_t run_host_quantize_from(const void* input_bf16, Workspace& ws,
                                         hipStream_t stream) {
    return kernel::launch_activation_quantize_a8(
        static_cast<const ps::bf16_t*>(input_bf16), ws.codes, ws.scales, kRows,
        kFeatures, kKp, kFeatures, Workspace::code_row_stride(),
        Workspace::scale_row_stride(), stream);
}

inline hipError_t run_host_quantize(Workspace& ws, hipStream_t stream) {
    return run_host_quantize_from(ws.rms_out, ws, stream);
}

inline hipError_t run_host_quantize_e4m3_from(const void* input_bf16,
                                              Workspace& ws,
                                              hipStream_t stream) {
    return kernel::launch_activation_quantize_e4m3(
        static_cast<const ps::bf16_t*>(input_bf16),
        reinterpret_cast<uint8_t*>(ws.codes),
        reinterpret_cast<float*>(ws.scales), kRows, kFeatures, kKp, kFeatures,
        Workspace::code_row_stride(), Workspace::scale_row_stride(), stream);
}

inline hipError_t run_host_psq4(Workspace& ws, hipStream_t stream) {
    return kernel::launch_gemm_psq4_w4a8_wmma_decode1(
        static_cast<const uint8_t*>(ws.w_codes),
        static_cast<const uint8_t*>(ws.w_scales),
        reinterpret_cast<const uint8_t*>(ws.codes), ws.scales, ws.final_out,
        kernel::Psq4GemmOutputDType::BF16, kRows, kN, kKp, kWeightScaleStride,
        kKp, Workspace::scale_row_stride(), kN, 16u, stream);
}

inline hipError_t run_host_chain(Workspace& ws, hipStream_t stream) {
    hipError_t e = run_host_rmsnorm(ws, stream);
    if (e != hipSuccess) return e;
    e = run_host_quantize(ws, stream);
    if (e != hipSuccess) return e;
    return run_host_psq4(ws, stream);
}

inline bool wait_queue_drained(const mcu::GpuMcuAqlQueue& queue,
                               uint64_t target, uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (queue.host_consumed_index() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
}

template <typename T>
inline void device_to_host_async(T* dst, const void* src, size_t bytes,
                                 hipStream_t stream) {
    (void)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
}

inline bool bytes_equal(const void* a, const void* b, size_t bytes,
                        size_t* first, size_t* diffs) {
    const auto* pa = static_cast<const unsigned char*>(a);
    const auto* pb = static_cast<const unsigned char*>(b);
    *first = 0;
    *diffs = 0;
    for (size_t i = 0; i < bytes; ++i) {
        if (pa[i] != pb[i]) {
            if (*diffs == 0) *first = i;
            ++*diffs;
        }
    }
    return *diffs == 0;
}

}  // namespace gpu_mcu_chain
