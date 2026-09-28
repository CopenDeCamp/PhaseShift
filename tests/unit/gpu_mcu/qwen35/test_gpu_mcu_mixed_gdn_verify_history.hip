#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/models/qwen35/runtime/gdn_spec_history.h>
#include <phaseshift/models/qwen35/state/gdn_state_pool.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/plan_binder.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ps::bf16_t;
using ps::qwen35::GdnStatePool;
using ps::qwen35::GdnStatePoolDeviceView;
using ps::qwen35::GdnStatePoolLayout;
using ps::qwen35::runtime::GdnSpecHistory;
using ps::qwen35::runtime::create_gdn_spec_history;
using ps::qwen35::runtime::shutdown_gdn_spec_history;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static bool alloc_bf16(ps::gpu::GpuArena& arena,
                       const std::vector<std::size_t>& shape,
                       ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(bf16_t), 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<bf16_t>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_f32(ps::gpu::GpuArena& arena,
                      const std::vector<std::size_t>& shape,
                      ps::gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) count *= d;
    auto view = arena.allocate_aligned(count * sizeof(float), 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto t = ps::gpu::Tensor::create<float>(backing, shape, strides);
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static bool alloc_u8(ps::gpu::GpuArena& arena, std::size_t bytes,
                     ps::gpu::Tensor& out) {
    auto view = arena.allocate_aligned(bytes, 256);
    if (!view.ok()) return false;
    auto backing = view.release();
    auto t = ps::gpu::Tensor::create<uint8_t>(backing, {bytes}, {1u});
    if (!t.ok()) return false;
    out = t.release();
    return true;
}

static float randf(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f -
           1.0f;
}

static ps::bf16_t randb(uint32_t& s) {
    return ps::bf16_t{ps::f32_to_bf16_rne(randf(s) * 0.5f)};
}

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    constexpr uint32_t kDecodeRows = 1u;
    constexpr uint32_t kVerifyRows = 8u;
    constexpr uint32_t kPrefillRows = 16u;
    constexpr uint32_t kActualRows = kDecodeRows + kVerifyRows + kPrefillRows;
    constexpr uint32_t kCapture = kVerifyRows + 1u;
    constexpr uint32_t kRequests = 3u;

    {
        ::ps::runtime::DeviceBatchContext ctx{};
        ctx.actual_rows = kActualRows;
        ctx.num_requests = kRequests;
        ctx.num_verify_requests = 1u;
        check(::ps::runtime::gpu_mcu::gpu_mcu_patch_source_value(
                  static_cast<uint8_t>(
                      ::ps::runtime::gpu_mcu::McuInvocationPatchSource::
                          VerifyRequests),
                  0u, &ctx) == 1u,
              "patch source reports verify batch");
        ctx.num_verify_requests = 0u;
        check(::ps::runtime::gpu_mcu::gpu_mcu_patch_source_value(
                  static_cast<uint8_t>(
                      ::ps::runtime::gpu_mcu::McuInvocationPatchSource::
                          VerifyRequests),
                  0u, &ctx) == 0u,
              "patch source reports plain batch");
        check(::ps::runtime::gpu_mcu::gpu_mcu_patch_source_value(
                  static_cast<uint8_t>(
                      ::ps::runtime::gpu_mcu::McuInvocationPatchSource::
                          VerifyRequests),
                  0u, nullptr) == 0u,
              "patch source tolerates null context");
    }

    auto arena_result = ps::gpu::GpuArena::create(
        0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n",
                    arena_result.status().message().c_str());
        return 77;
    }
    ps::gpu::GpuArena arena = arena_result.release();
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL hipStreamCreate\n");
        return 1;
    }

    const uint32_t kKeyHeads = 16u;
    const uint32_t kNumVHeads = 48u;
    const uint32_t kHeadK = 128u;
    const uint32_t kHeadV = 128u;
    const uint32_t q_feat = kKeyHeads * kHeadK;
    const uint32_t v_feat = kNumVHeads * kHeadV;
    const uint32_t conv_dim = 2u * q_feat + v_feat;

    const GdnStatePoolLayout layout =
        GdnStatePoolLayout::make(1u, conv_dim, 3u, kNumVHeads, kHeadK, kHeadV);
    auto pool_result = GdnStatePool::create(arena, 1u, layout);
    check(pool_result.ok(), "pool create");
    if (!pool_result.ok()) return 1;
    GdnStatePool pool = pool_result.release();
    check(pool.initialize(0u, stream).ok(), "pool initialize");
    check(pool.reset(0u, stream).ok(), "pool reset");
    (void)hipStreamSynchronize(stream);

    auto history_result = create_gdn_spec_history(arena, pool, kCapture);
    check(history_result.ok(), "history create");
    if (!history_result.ok()) return 1;
    GdnSpecHistory history = history_result.release();
    const GdnStatePoolDeviceView view = pool.device_view();

    ps::gpu::Tensor conv_in;
    ps::gpu::Tensor conv_w;
    ps::gpu::Tensor conv_out;
    ps::gpu::Tensor req_mem;
    const bool alloc_ok =
        alloc_bf16(arena, {kActualRows, conv_dim}, conv_in) &&
        alloc_bf16(arena, {conv_dim * 4u}, conv_w) &&
        alloc_f32(arena, {kActualRows, conv_dim}, conv_out) &&
        alloc_u8(arena, kRequests * sizeof(::ps::runtime::DeviceRequestDescriptor),
                 req_mem);
    check(alloc_ok, "buffer alloc");
    if (!alloc_ok) return 1;

    std::vector<bf16_t> hci(static_cast<std::size_t>(kActualRows) * conv_dim);
    std::vector<bf16_t> hcw(static_cast<std::size_t>(conv_dim) * 4u);
    uint32_t seed = 0x51f3d7a9u;
    for (bf16_t& x : hci) x = randb(seed);
    for (bf16_t& x : hcw) x = randb(seed);
    check(hipMemcpy(conv_in.data<bf16_t>(), hci.data(),
                    hci.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
              hipSuccess,
          "conv input upload");
    check(hipMemcpy(conv_w.data<bf16_t>(), hcw.data(),
                    hcw.size() * sizeof(bf16_t), hipMemcpyHostToDevice) ==
              hipSuccess,
          "conv weight upload");

    {
        ::ps::runtime::DeviceRequestDescriptor host[kRequests]{};
        host[0].request_handle.slot = 0u;
        host[0].request_handle.generation = 1u;
        host[0].execution_class = ::ps::runtime::ExecutionClass::DECODE;
        host[0].row_begin = 0u;
        host[0].row_count = kDecodeRows;
        host[1].request_handle.slot = 0u;
        host[1].request_handle.generation = 1u;
        host[1].execution_class = ::ps::runtime::ExecutionClass::SPEC_VERIFY;
        host[1].row_begin = kDecodeRows;
        host[1].row_count = kVerifyRows;
        host[2].request_handle.slot = 0u;
        host[2].request_handle.generation = 1u;
        host[2].execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        host[2].row_begin = kDecodeRows + kVerifyRows;
        host[2].row_count = kPrefillRows;
        check(hipMemcpy(req_mem.data<uint8_t>(), host, sizeof(host),
                        hipMemcpyHostToDevice) == hipSuccess,
              "request upload");
    }

    auto* req = reinterpret_cast<::ps::runtime::DeviceRequestDescriptor*>(
        req_mem.data<uint8_t>());
    const std::size_t conv_row_bytes = history.conv.dim(1) * sizeof(bf16_t);
    const std::size_t history_rows_bytes =
        static_cast<std::size_t>(kCapture) * conv_row_bytes;

    auto make_args = [&](uint32_t capture_verify_only) {
        ::ps::kernel::GdnConv1dArgs c{};
        c.input = conv_in.data<bf16_t>();
        c.output = conv_out.data<float>();
        c.conv_weight = conv_w.data<bf16_t>();
        c.conv_state = view.conv_base;
        c.requests = req;
        c.num_requests = kRequests;
        c.actual_rows = kActualRows;
        c.max_request_rows = kPrefillRows;
        c.input_row_stride = conv_dim;
        c.output_row_stride = conv_dim;
        c.conv_dim = conv_dim;
        c.conv_history = 3u;
        c.state_index = 0u;
        c.max_sequences = pool.max_sequences();
        c.conv_slot_stride = view.conv_slot_stride;
        c.conv_layer_stride = view.conv_layer_stride;
        c.conv_history_stride = view.conv_history_stride;
        c.conv_history_store = history.conv.data<bf16_t>();
        c.conv_history_row_stride = static_cast<uint64_t>(history.conv.dim(1));
        c.capture_rows = kCapture;
        c.capture_verify_only = capture_verify_only;
        return c;
    };

    auto run_capture = [&](uint32_t capture_verify_only,
                           std::vector<uint8_t>& out) {
        check(pool.reset(0u, stream).ok(), "capture reset");
        std::vector<uint8_t> pattern(history_rows_bytes, 0xABu);
        check(hipMemcpy(history.conv.data<bf16_t>(), pattern.data(),
                        history_rows_bytes, hipMemcpyHostToDevice) == hipSuccess,
              "history seed");
        auto ca = make_args(capture_verify_only);
        check(::ps::kernel::launch_gdn_conv1d_bf16_f32(ca, stream) ==
                  hipSuccess,
              "conv launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "capture sync");
        out.assign(history_rows_bytes, 0u);
        check(hipMemcpy(out.data(), history.conv.data<bf16_t>(),
                        history_rows_bytes, hipMemcpyDeviceToHost) == hipSuccess,
              "history read");
    };

    std::vector<uint8_t> legacy;
    std::vector<uint8_t> verify_only;
    run_capture(0u, legacy);
    run_capture(1u, verify_only);

    for (uint32_t i = 0u; i < kVerifyRows; ++i) {
        const uint8_t* want = legacy.data() + (i + 1u) * conv_row_bytes;
        const uint8_t* got = verify_only.data() + i * conv_row_bytes;
        check(std::memcmp(want, got, conv_row_bytes) == 0,
              "verify local row " + std::to_string(i) + " matches batch row " +
                  std::to_string(i + 1u));
    }
    {
        std::vector<uint8_t> untouched(conv_row_bytes, 0xABu);
        const uint8_t* tail = verify_only.data() + kVerifyRows * conv_row_bytes;
        check(std::memcmp(tail, untouched.data(), conv_row_bytes) == 0,
              "verify capture leaves the unused history row untouched");
    }
    {
        std::vector<uint8_t> legacy_row7(
            legacy.data() + kVerifyRows * conv_row_bytes,
            legacy.data() + (kVerifyRows + 1u) * conv_row_bytes);
        check(std::memcmp(legacy_row7.data(), verify_only.data() +
                                                 kVerifyRows * conv_row_bytes,
                          conv_row_bytes) != 0,
              "legacy capture writes the row that verify capture skips");
    }

    check(shutdown_gdn_spec_history(history).ok(), "history shutdown");
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_gpu_mcu_mixed_gdn_verify_history: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
