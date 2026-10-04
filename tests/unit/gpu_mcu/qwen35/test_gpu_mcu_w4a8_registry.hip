#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>

#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace q35 = ps::qwen35::runtime;
namespace rt = ps::runtime;

int g_failed = 0;

bool check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
    return condition;
}

uint32_t xr(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

ps::bf16_t bf16_from_float(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, 4);
    return ps::bf16_t{static_cast<uint16_t>((ui + 0x8000u) >> 16)};
}

void fill_bf16(std::vector<uint16_t>& out, uint32_t count, uint32_t seed) {
    out.assign(count, 0u);
    uint32_t s = seed;
    for (uint32_t i = 0; i < count; ++i) {
        const int base = static_cast<int>(xr(s) % 2001u) - 1000;
        out[i] = bf16_from_float(static_cast<float>(base) / 1000.0f).data;
    }
}

void fill_bytes(std::vector<uint8_t>& out, size_t count, uint32_t seed) {
    out.assign(count, 0u);
    uint32_t s = seed;
    for (size_t i = 0; i < count; ++i) out[i] = static_cast<uint8_t>(xr(s) & 0xFFu);
}

struct RegistryEntry {
    q35::McuCompiledVariantKind kind;
    const char* symbol;
    uint16_t recipe;
};

bool load_registry_code(Rig& rig, q35::McuCompiledVariantKind kind,
                        const char* expected_symbol, uint16_t expected_recipe,
                        mcu::GpuAqlKernelMetadata& out_meta) {
    auto co = q35::mcu_kernel_code_object(kind);
    if (!check(co.ok(), "registry code object resolved")) {
        std::printf("  kind=%s status=%s\n",
                    q35::mcu_kernel_variant_kind_name(kind),
                    co.ok() ? "" : co.status().message().c_str());
        return false;
    }
    const q35::McuKernelCodeObject& obj = co.value();
    if (!check(obj.blob.blob != nullptr && obj.blob.size != 0u,
               "registry blob is present")) {
        return false;
    }
    if (!check(obj.symbol != nullptr &&
                   std::strcmp(obj.symbol, expected_symbol) == 0,
               "registry symbol matches")) {
        std::printf("  kind=%s symbol=%s expected=%s\n",
                    q35::mcu_kernel_variant_kind_name(kind),
                    obj.symbol == nullptr ? "<null>" : obj.symbol,
                    expected_symbol);
        return false;
    }
    check(obj.kernarg_recipe == expected_recipe, "registry recipe matches");
    auto loaded = mcu::GpuMcuAqlCodeObject::load_memory(
        rig.queue, obj.blob.blob, obj.blob.size, obj.symbol);
    if (!check(loaded.ok(), "registry blob loads into an hsa code object")) {
        std::printf("  kind=%s load=%s\n",
                    q35::mcu_kernel_variant_kind_name(kind),
                    loaded.status().message().c_str());
        return false;
    }
    out_meta = loaded.value().metadata();
    rig.extra_codes.push_back(loaded.release());
    return true;
}

bool run_single_primitive(Rig& rig,
                          const mcu::McuKernelVariantDesc& kernel_variant,
                          uint32_t node_count, const char* label,
                          const std::function<void(Rig&)>& set_invocation) {
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u,
                                mcu::kMcuSystemAqlPolicy),
        kernel_variant};
    std::vector<mcu::McuPlanNode> nodes(node_count);
    nodes[0].variant_id = 1;
    nodes[0].kernarg_recipe = kernel_variant.kernarg_recipe;
    nodes[0].completion_slot = 0;
    nodes[0].invocation_index = 0;
    nodes[0].flags = mcu::kMcuNodeDispatch;
    nodes[0].next = 1;
    nodes[1].variant_id = 0;
    nodes[1].kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    nodes[1].completion_slot = 0;
    nodes[1].input_slot = mcu::kMcuNoInput;
    nodes[1].flags = mcu::kMcuNodeDispatch;
    nodes[1].next = 2;
    nodes[2].completion_slot = 0;
    nodes[2].flags = mcu::kMcuNodeWait;
    nodes[2].next = 3;
    nodes[3].flags = mcu::kMcuNodeEnd;

    if (!rig.upload(nodes, variants)) return false;
    set_invocation(rig);
    rig.poison_all_kernarg();

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 0x57345247u;
    config.template_source = 0u;
    config.barrier = 1u;
    config.timing_count = 0u;
    config.kernarg_slot_count = node_count;
    config.doorbell_mode =
        static_cast<uint32_t>(mcu::McuDoorbellMode::per_packet);
    if (!rig.start(config)) return false;
    if (!check(rig.fsm.request_run().ok(), "plan requested")) return false;
    if (!check(rig.fsm.wait_plans(1u, 30000).ok(), "plan completed")) return false;
    if (!check(rig.drain_queue(30000), "queue drained")) return false;
    const bool no_fault =
        check(rig.fsm.fault_code() == mcu::McuFaultCode::none,
              "plan completed without fault");
    if (!no_fault) {
        std::printf("  %s fault=%u pc=%u\n", label,
                    static_cast<unsigned>(rig.fsm.fault_code()),
                    rig.fsm.fault_pc());
    }
    if (!check(rig.fsm.request_stop().ok(), "fsm stop requested") ||
        !check(rig.fsm.wait_stopped(5000).ok(), "fsm stopped before readback")) {
        return false;
    }
    return no_fault;
}

void test_e4m3_shape(q35::McuCompiledVariantKind kind,
                     const char* symbol, uint16_t recipe, uint32_t features,
                     uint32_t seed) {
    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize, 0u)) {
        check(false, "e4m3 rig open");
        return;
    }
    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(features, 1u);

    void* input = nullptr;
    void* a8_host = nullptr;
    void* a8_mcu = nullptr;
    if (hipMalloc(&input, static_cast<size_t>(features) * 2u) != hipSuccess ||
        hipMalloc(&a8_host, layout.total_bytes) != hipSuccess ||
        hipMalloc(&a8_mcu, layout.total_bytes) != hipSuccess) {
        check(false, "e4m3 buffers allocated");
        check(rig.close(), "e4m3 rig teardown");
        return;
    }
    (void)hipMemset(a8_host, 0xAB, layout.total_bytes);
    (void)hipMemset(a8_mcu, 0xAB, layout.total_bytes);

    std::vector<uint16_t> h_in;
    fill_bf16(h_in, features, seed);
    (void)hipMemcpy(input, h_in.data(), h_in.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice);

    hipStream_t stream = rig.partition.worker_stream();
    const uint8_t* host_codes =
        static_cast<uint8_t*>(a8_host) + layout.codes_offset;
    const float* host_scales =
        reinterpret_cast<const float*>(static_cast<uint8_t*>(a8_host) +
                                       layout.scales_offset);
    check(kernel::activation_quantize_e4m3_vec_supported(
              static_cast<const ps::bf16_t*>(input), host_codes, features,
              layout.k_padded, features),
          "e4m3 vec path is supported for the test shape");
    std::printf("  [e4m3 k=%u] host launch\n", features);
    (void)kernel::launch_activation_quantize_e4m3(
        static_cast<const ps::bf16_t*>(input),
        const_cast<uint8_t*>(host_codes), const_cast<float*>(host_scales), 1u,
        features, layout.k_padded, features, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, stream);
    (void)hipStreamSynchronize(stream);
    std::printf("  [e4m3 k=%u] host done\n", features);

    mcu::GpuAqlKernelMetadata meta{};
    if (!load_registry_code(rig, kind, symbol, recipe, meta)) {
        (void)hipFree(input);
        (void)hipFree(a8_host);
        (void)hipFree(a8_mcu);
        check(rig.close(), "e4m3 rig teardown");
        return;
    }

    mcu::McuActivationQuantizeE4m3Invocation inv{};
    inv.input = reinterpret_cast<uint64_t>(input);
    inv.codes = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(a8_mcu) +
                                           layout.codes_offset);
    inv.scales = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(a8_mcu) +
                                            layout.scales_offset);
    inv.input_row_stride = features;
    inv.code_row_stride = layout.code_row_stride_bytes;
    inv.scale_row_stride = layout.scale_row_stride_bytes;

    const mcu::McuKernelVariantDesc variant = mcu::make_kernel_variant(
        1u, meta, recipe, 256u, 1u, mcu::kMcuAgentAqlPolicy,
        mcu::AqlHiddenArgsPolicy::None);
    const bool ran = run_single_primitive(
        rig, variant, 4u, "e4m3",
        [&](Rig& r) { (void)r.set_e4m3_invocations(&inv, 1u); });
    std::printf("  [e4m3 k=%u] mcu done ran=%d\n", features, static_cast<int>(ran));
    if (ran) {
        std::vector<uint8_t> h_codes(layout.codes_bytes());
        std::vector<uint8_t> m_codes(layout.codes_bytes());
        std::vector<uint8_t> h_scales(layout.scales_bytes());
        std::vector<uint8_t> m_scales(layout.scales_bytes());
        rig.device_to_host(h_codes.data(),
                           static_cast<uint8_t*>(a8_host) + layout.codes_offset,
                           h_codes.size());
        rig.device_to_host(m_codes.data(),
                           static_cast<uint8_t*>(a8_mcu) + layout.codes_offset,
                           m_codes.size());
        rig.device_to_host(h_scales.data(),
                           static_cast<uint8_t*>(a8_host) + layout.scales_offset,
                           h_scales.size());
        rig.device_to_host(m_scales.data(),
                           static_cast<uint8_t*>(a8_mcu) + layout.scales_offset,
                           m_scales.size());
        const bool codes_equal =
            std::memcmp(h_codes.data(), m_codes.data(), h_codes.size()) == 0;
        check(codes_equal, "mcu e4m3 codes match the host path");
        check(std::memcmp(h_scales.data(), m_scales.data(), h_scales.size()) == 0,
              "mcu e4m3 scales match the host path");
        std::printf("  e4m3 k=%u codes=%zu scales=%zu matched=%d\n", features,
                    h_codes.size(), h_scales.size(),
                    static_cast<int>(codes_equal));
    }

    (void)hipFree(input);
    (void)hipFree(a8_host);
    (void)hipFree(a8_mcu);
    check(rig.close(), "e4m3 rig teardown");
}

void test_psq4_u8(uint32_t out_features, uint32_t features, uint32_t seed) {
    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize, 0u)) {
        check(false, "psq4 rig open");
        return;
    }
    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(features, 1u);
    const uint32_t tiles = (out_features + 15u) / 16u;
    const uint32_t nb = layout.k_padded / 32u;
    const size_t w_codes_bytes =
        static_cast<size_t>(tiles) * layout.k_padded * 8u;
    const size_t w_scales_bytes = static_cast<size_t>(tiles) * nb * 32u;
    const size_t out_bytes = static_cast<size_t>(out_features) * 2u;
    const uint32_t weight_scale_stride = nb * 32u;

    void* input = nullptr;
    void* a8 = nullptr;
    void* w_codes = nullptr;
    void* w_scales = nullptr;
    void* out_host = nullptr;
    void* out_mcu = nullptr;
    if (hipMalloc(&input, static_cast<size_t>(features) * 2u) != hipSuccess ||
        hipMalloc(&a8, layout.total_bytes) != hipSuccess ||
        hipMalloc(&w_codes, w_codes_bytes) != hipSuccess ||
        hipMalloc(&w_scales, w_scales_bytes) != hipSuccess ||
        hipMalloc(&out_host, out_bytes) != hipSuccess ||
        hipMalloc(&out_mcu, out_bytes) != hipSuccess) {
        check(false, "psq4 buffers allocated");
        check(rig.close(), "psq4 rig teardown");
        return;
    }
    (void)hipMemset(a8, 0xAB, layout.total_bytes);
    (void)hipMemset(out_host, 0xAB, out_bytes);
    (void)hipMemset(out_mcu, 0xAB, out_bytes);

    std::vector<uint16_t> h_in;
    fill_bf16(h_in, features, seed);
    (void)hipMemcpy(input, h_in.data(), h_in.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice);

    std::vector<uint8_t> h_wc;
    std::vector<uint8_t> h_ws;
    fill_bytes(h_wc, w_codes_bytes, seed ^ 0x9E3779B9u);
    fill_bytes(h_ws, w_scales_bytes, seed ^ 0x7F4A7C15u);
    (void)hipMemcpy(w_codes, h_wc.data(), w_codes_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(w_scales, h_ws.data(), w_scales_bytes,
                    hipMemcpyHostToDevice);

    hipStream_t stream = rig.partition.worker_stream();
    uint8_t* act_codes = static_cast<uint8_t*>(a8) + layout.codes_offset;
    float* act_scales =
        reinterpret_cast<float*>(static_cast<uint8_t*>(a8) +
                                 layout.scales_offset);
    (void)kernel::launch_activation_quantize_e4m3(
        static_cast<const ps::bf16_t*>(input), act_codes, act_scales, 1u,
        features, layout.k_padded, features, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, stream);
    (void)hipStreamSynchronize(stream);

    const uint32_t unroll = kernel::psq4_decode1_unroll(out_features,
                                                        layout.k_padded);
    check(unroll == 8u, "large out_features selects the unroll 8 variant");
    (void)kernel::launch_gemm_psq4_w4a8_wmma_decode1(
        static_cast<const uint8_t*>(w_codes),
        static_cast<const uint8_t*>(w_scales), act_codes, act_scales, out_host,
        kernel::Psq4GemmOutputDType::BF16, 1u, out_features, layout.k_padded,
        weight_scale_stride, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, out_features, unroll, stream);
    (void)hipStreamSynchronize(stream);

    mcu::GpuAqlKernelMetadata meta{};
    if (!load_registry_code(rig, q35::McuCompiledVariantKind::Psq4Decode1Bf16U8,
                            kernel::kPsq4Decode1Bf16U8Symbol,
                            mcu::kMcuKernargRecipePsq4Decode1Bf16U8, meta)) {
        (void)hipFree(input);
        (void)hipFree(a8);
        (void)hipFree(w_codes);
        (void)hipFree(w_scales);
        (void)hipFree(out_host);
        (void)hipFree(out_mcu);
        check(rig.close(), "psq4 rig teardown");
        return;
    }

    mcu::McuPsq4Decode1Invocation inv{};
    inv.weight_codes = reinterpret_cast<uint64_t>(w_codes);
    inv.weight_scales = reinterpret_cast<uint64_t>(w_scales);
    inv.activation_codes = reinterpret_cast<uint64_t>(act_codes);
    inv.activation_scales = reinterpret_cast<uint64_t>(act_scales);
    inv.output = reinterpret_cast<uint64_t>(out_mcu);
    inv.k_padded = layout.k_padded;
    inv.weight_scale_stride = weight_scale_stride;

    const uint32_t grid = kernel::psq4_decode1_grid(out_features);
    const mcu::McuKernelVariantDesc variant = mcu::make_kernel_variant(
        1u, meta, mcu::kMcuKernargRecipePsq4Decode1Bf16U8,
        kernel::psq4_decode1_block_size(), grid, mcu::kMcuAgentAqlPolicy,
        mcu::AqlHiddenArgsPolicy::None);
    const bool ran = run_single_primitive(
        rig, variant, 4u, "psq4_u8",
        [&](Rig& r) { (void)r.set_psq4_invocations(&inv, 1u); });
    if (ran) {
        std::vector<uint8_t> h_out(out_bytes);
        std::vector<uint8_t> m_out(out_bytes);
        rig.device_to_host(h_out.data(), out_host, out_bytes);
        rig.device_to_host(m_out.data(), out_mcu, out_bytes);
        const bool equal = std::memcmp(h_out.data(), m_out.data(), out_bytes) == 0;
        check(equal, "mcu psq4 u8 output matches the host path");
        bool nontrivial = false;
        for (uint8_t b : m_out) {
            if (b != 0u && b != 0xABu) {
                nontrivial = true;
                break;
            }
        }
        check(nontrivial, "mcu psq4 u8 produced non-trivial output");
        std::printf("  psq4 u8 out=%u unroll=%u grid=%u matched=%d\n",
                    out_features, unroll, grid, static_cast<int>(equal));
    }

    (void)hipFree(input);
    (void)hipFree(a8);
    (void)hipFree(w_codes);
    (void)hipFree(w_scales);
    (void)hipFree(out_host);
    (void)hipFree(out_mcu);
    check(rig.close(), "psq4 rig teardown");
}

void test_psq8_u8(uint32_t out_features, uint32_t features, uint32_t seed) {
    Rig rig;
    if (!rig.open(0, 8u, 4u, 8u, 0u, false, mcu::kAqlDefaultQueueSize, 0u)) {
        check(false, "psq8 rig open");
        return;
    }
    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(features, 1u);
    const uint32_t tiles = (out_features + 15u) / 16u;
    const uint32_t nb = layout.k_padded / 32u;
    const size_t w_codes_bytes =
        static_cast<size_t>(tiles) * layout.k_padded * 16u;
    const size_t w_scales_bytes = static_cast<size_t>(tiles) * nb * 32u;
    const size_t out_bytes = static_cast<size_t>(out_features) * 2u;
    const uint32_t weight_scale_stride = nb * 32u;

    void* input = nullptr;
    void* a8 = nullptr;
    void* w_codes = nullptr;
    void* w_scales = nullptr;
    void* out_host = nullptr;
    void* out_mcu = nullptr;
    if (hipMalloc(&input, static_cast<size_t>(features) * 2u) != hipSuccess ||
        hipMalloc(&a8, layout.total_bytes) != hipSuccess ||
        hipMalloc(&w_codes, w_codes_bytes) != hipSuccess ||
        hipMalloc(&w_scales, w_scales_bytes) != hipSuccess ||
        hipMalloc(&out_host, out_bytes) != hipSuccess ||
        hipMalloc(&out_mcu, out_bytes) != hipSuccess) {
        check(false, "psq8 buffers allocated");
        check(rig.close(), "psq8 rig teardown");
        return;
    }
    (void)hipMemset(a8, 0xAB, layout.total_bytes);
    (void)hipMemset(out_host, 0xAB, out_bytes);
    (void)hipMemset(out_mcu, 0xAB, out_bytes);

    std::vector<uint16_t> h_in;
    fill_bf16(h_in, features, seed);
    (void)hipMemcpy(input, h_in.data(), h_in.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice);

    std::vector<uint8_t> h_wc;
    std::vector<uint8_t> h_ws;
    fill_bytes(h_wc, w_codes_bytes, seed ^ 0x9E3779B9u);
    fill_bytes(h_ws, w_scales_bytes, seed ^ 0x7F4A7C15u);
    (void)hipMemcpy(w_codes, h_wc.data(), w_codes_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(w_scales, h_ws.data(), w_scales_bytes,
                    hipMemcpyHostToDevice);

    hipStream_t stream = rig.partition.worker_stream();
    uint8_t* act_codes = static_cast<uint8_t*>(a8) + layout.codes_offset;
    float* act_scales =
        reinterpret_cast<float*>(static_cast<uint8_t*>(a8) +
                                 layout.scales_offset);
    (void)kernel::launch_activation_quantize_e4m3(
        static_cast<const ps::bf16_t*>(input), act_codes, act_scales, 1u,
        features, layout.k_padded, features, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, stream);
    (void)hipStreamSynchronize(stream);

    const uint32_t unroll = kernel::psq8_decode1_unroll(out_features,
                                                        layout.k_padded);
    check(unroll == 8u, "psq8 selects the unroll 8 variant");
    (void)kernel::launch_gemm_psq8_w8a8_wmma_decode1(
        static_cast<const uint8_t*>(w_codes),
        static_cast<const uint8_t*>(w_scales), act_codes, act_scales, out_host,
        kernel::Psq8GemmOutputDType::BF16, 1u, out_features, layout.k_padded,
        weight_scale_stride, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, out_features, unroll, stream);
    (void)hipStreamSynchronize(stream);

    mcu::GpuAqlKernelMetadata meta{};
    if (!load_registry_code(rig, q35::McuCompiledVariantKind::Psq8Decode1Bf16U8,
                            kernel::kPsq8Decode1Bf16U8Symbol,
                            mcu::kMcuKernargRecipePsq8Decode1Bf16U8, meta)) {
        (void)hipFree(input);
        (void)hipFree(a8);
        (void)hipFree(w_codes);
        (void)hipFree(w_scales);
        (void)hipFree(out_host);
        (void)hipFree(out_mcu);
        check(rig.close(), "psq8 rig teardown");
        return;
    }

    mcu::McuPsq4Decode1Invocation inv{};
    inv.weight_codes = reinterpret_cast<uint64_t>(w_codes);
    inv.weight_scales = reinterpret_cast<uint64_t>(w_scales);
    inv.activation_codes = reinterpret_cast<uint64_t>(act_codes);
    inv.activation_scales = reinterpret_cast<uint64_t>(act_scales);
    inv.output = reinterpret_cast<uint64_t>(out_mcu);
    inv.k_padded = layout.k_padded;
    inv.weight_scale_stride = weight_scale_stride;

    const uint32_t grid = kernel::psq8_decode1_grid(out_features);
    const mcu::McuKernelVariantDesc variant = mcu::make_kernel_variant(
        1u, meta, mcu::kMcuKernargRecipePsq8Decode1Bf16U8,
        kernel::psq8_decode1_block_size(), grid, mcu::kMcuAgentAqlPolicy,
        mcu::AqlHiddenArgsPolicy::None);
    const bool ran = run_single_primitive(
        rig, variant, 4u, "psq8_u8",
        [&](Rig& r) { (void)r.set_psq4_invocations(&inv, 1u); });
    if (ran) {
        std::vector<uint8_t> h_out(out_bytes);
        std::vector<uint8_t> m_out(out_bytes);
        rig.device_to_host(h_out.data(), out_host, out_bytes);
        rig.device_to_host(m_out.data(), out_mcu, out_bytes);
        const bool equal = std::memcmp(h_out.data(), m_out.data(), out_bytes) == 0;
        check(equal, "mcu psq8 u8 output matches the host path");
        bool nontrivial = false;
        for (uint8_t b : m_out) {
            if (b != 0u && b != 0xABu) {
                nontrivial = true;
                break;
            }
        }
        check(nontrivial, "mcu psq8 u8 produced non-trivial output");
        std::printf("  psq8 u8 out=%u k=%u unroll=%u grid=%u matched=%d\n",
                    out_features, layout.k_padded, unroll, grid,
                    static_cast<int>(equal));
    }

    (void)hipFree(input);
    (void)hipFree(a8);
    (void)hipFree(w_codes);
    (void)hipFree(w_scales);
    (void)hipFree(out_host);
    (void)hipFree(out_mcu);
    check(rig.close(), "psq8 rig teardown");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    std::printf("=== w4a8 registry embedding ===\n");

    {
        Rig registry_rig;
        if (!registry_rig.open(0, 8u, 4u, 8u, 0u, false,
                               mcu::kAqlDefaultQueueSize, 0u)) {
            std::printf("test_gpu_mcu_w4a8_registry: setup failed\n");
            return 1;
        }
        const RegistryEntry entries[] = {
            {q35::McuCompiledVariantKind::ActivationQuantizeE4m3K5120,
             kernel::kActivationQuantizeE4m3K5120Symbol,
             mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120},
            {q35::McuCompiledVariantKind::ActivationQuantizeE4m3K6144,
             kernel::kActivationQuantizeE4m3K6144Symbol,
             mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120},
            {q35::McuCompiledVariantKind::ActivationQuantizeE4m3K17408,
             kernel::kActivationQuantizeE4m3K17408Symbol,
             mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120},
            {q35::McuCompiledVariantKind::Psq4Decode1Bf16U16,
             kernel::kPsq4Decode1Bf16U16Symbol,
             mcu::kMcuKernargRecipePsq4Decode1Bf16U16},
            {q35::McuCompiledVariantKind::Psq4Decode1Bf16U8,
             kernel::kPsq4Decode1Bf16U8Symbol,
             mcu::kMcuKernargRecipePsq4Decode1Bf16U8},
            {q35::McuCompiledVariantKind::Psq8Decode1Bf16U8,
             kernel::kPsq8Decode1Bf16U8Symbol,
             mcu::kMcuKernargRecipePsq8Decode1Bf16U8},
            {q35::McuCompiledVariantKind::AttentionPagedSplitOther,
             kernel::kPagedAttentionSplitOtherSymbol,
             mcu::kMcuKernargRecipeAttentionPagedSplit},
            {q35::McuCompiledVariantKind::AttentionPagedReduceOther,
             kernel::kPagedAttentionReduceOtherSymbol,
             mcu::kMcuKernargRecipeAttentionPagedReduce},
        };
        for (const RegistryEntry& e : entries) {
            mcu::GpuAqlKernelMetadata meta{};
            const bool ok =
                load_registry_code(registry_rig, e.kind, e.symbol, e.recipe, meta);
            std::printf("  registry %s ok=%d kernarg=%u\n",
                        q35::mcu_kernel_variant_kind_name(e.kind),
                        static_cast<int>(ok), meta.kernarg_segment_size);
        }
        check(registry_rig.close(), "registry rig teardown");
    }

    test_e4m3_shape(q35::McuCompiledVariantKind::ActivationQuantizeE4m3K6144,
                    kernel::kActivationQuantizeE4m3K6144Symbol,
                    mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120, 6144u,
                    0x6144u);
    test_e4m3_shape(q35::McuCompiledVariantKind::ActivationQuantizeE4m3K17408,
                    kernel::kActivationQuantizeE4m3K17408Symbol,
                    mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120, 17408u,
                    0x17408u);
    test_psq4_u8(4096u, 5120u, 0x4096u);
    test_psq8_u8(6144u, 5120u, 0x6144u);
    test_psq8_u8(5120u, 17408u, 0x17408u);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_w4a8_registry: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_w4a8_registry: PASS\n");
    return 0;
}
