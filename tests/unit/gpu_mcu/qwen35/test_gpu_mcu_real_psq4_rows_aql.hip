#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>
#include <phaseshift/runtime/program/int8_activation_workspace.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;
namespace q35 = ps::qwen35::runtime;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kFeatures = chain::kFeatures;
constexpr uint32_t kKp = chain::kKp;
constexpr uint32_t kN = chain::kN;
constexpr uint32_t kNb = chain::kNb;
constexpr uint32_t kTiles = chain::kTiles;
constexpr uint32_t kWeightScaleStride = chain::kWeightScaleStride;

struct WeightBuffers {
    void* w_codes = nullptr;
    void* w_scales = nullptr;
    bool allocate() {
        const size_t codes = static_cast<size_t>(kTiles) * kKp * 8u;
        const size_t scales = static_cast<size_t>(kTiles) * kNb * 32u;
        return hipMalloc(&w_codes, codes) == hipSuccess &&
               hipMalloc(&w_scales, scales) == hipSuccess;
    }
    void fill() {
        const size_t codes = static_cast<size_t>(kTiles) * kKp * 8u;
        const size_t scales = static_cast<size_t>(kTiles) * kNb * 32u;
        uint32_t s = 0x51A7u + kKp * 131u + kN;
        std::vector<uint8_t> hc(codes);
        for (auto& v : hc) v = chain::code_byte(s);
        (void)hipMemcpy(w_codes, hc.data(), hc.size(), hipMemcpyHostToDevice);
        std::vector<uint8_t> hs(scales, 0u);
        for (size_t i = 0; i + 1 < hs.size(); i += 2) {
            const uint16_t v = chain::scale_bf16(s);
            hs[i] = static_cast<uint8_t>(v & 0xFFu);
            hs[i + 1] = static_cast<uint8_t>(v >> 8u);
        }
        (void)hipMemcpy(w_scales, hs.data(), hs.size(), hipMemcpyHostToDevice);
    }
    void free_all() {
        (void)hipFree(w_codes);
        (void)hipFree(w_scales);
        w_codes = nullptr;
        w_scales = nullptr;
    }
};

bool symbol_for_config(kernel::Psq4GemmConfigId id, const char** symbol,
                       uint32_t* mb_rows, uint32_t* workgroup,
                       uint32_t* out_block) {
    *out_block = 0u;
    switch (id) {
        case kernel::Psq4GemmConfigId::RowBlock1:
            *symbol = kernel::kPsq4RowBlock1Bf16Symbol;
            *mb_rows = 1u;
            break;
        case kernel::Psq4GemmConfigId::RowBlock2:
            *symbol = kernel::kPsq4RowBlock2Bf16Symbol;
            *mb_rows = 2u;
            break;
        case kernel::Psq4GemmConfigId::RowBlock4:
            *symbol = kernel::kPsq4RowBlock4Bf16Symbol;
            *mb_rows = 4u;
            break;
        case kernel::Psq4GemmConfigId::RowBlock8:
            *symbol = kernel::kPsq4RowBlock8Bf16Symbol;
            *mb_rows = 8u;
            break;
        case kernel::Psq4GemmConfigId::Prefill2D_K64N64:
            *symbol = kernel::kPsq4Prefill2DK64N64Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq4GemmPrefill2dBlock64;
            break;
        case kernel::Psq4GemmConfigId::Prefill2D_K128N64:
            *symbol = kernel::kPsq4Prefill2DK128N64Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq4GemmPrefill2dBlock64;
            break;
        case kernel::Psq4GemmConfigId::Prefill2D_K64N128:
            *symbol = kernel::kPsq4Prefill2DK64N128Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq4GemmPrefill2dBlock128;
            break;
        case kernel::Psq4GemmConfigId::Prefill2D_K128N128:
            *symbol = kernel::kPsq4Prefill2DK128N128Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq4GemmPrefill2dBlock128;
            break;
        default:
            return false;
    }
    *workgroup = *out_block != 0u ? kernel::kPsq4Prefill2DThreads : 32u;
    return true;
}

bool run_case(mcu::GpuMcuAqlQueue& queue, const WeightBuffers& w, uint32_t rows,
              const char* name) {
    const auto config =
        q35::select_psq4_gemm_config({rows, kN, kFeatures, kKp});
    if (!config) {
        std::printf("FAIL: %s selector rejected the shape\n", name);
        return false;
    }
    const char* symbol = nullptr;
    uint32_t mb_rows = 0u;
    uint32_t workgroup = 0u;
    uint32_t out_block = 0u;
    if (!symbol_for_config(config->id, &symbol, &mb_rows, &workgroup,
                           &out_block)) {
        std::printf("FAIL: %s selector chose an unsupported config\n", name);
        return false;
    }

    const auto layout =
        ps::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, rows);
    void* input = nullptr;
    void* a8 = nullptr;
    void* host_out = nullptr;
    void* aql_out = nullptr;
    if (hipMalloc(&input, static_cast<size_t>(rows) * kFeatures * sizeof(uint16_t)) !=
            hipSuccess ||
        hipMalloc(&a8, layout.total_bytes) != hipSuccess ||
        hipMalloc(&host_out, static_cast<size_t>(rows) * kN * sizeof(uint16_t)) !=
            hipSuccess ||
        hipMalloc(&aql_out, static_cast<size_t>(rows) * kN * sizeof(uint16_t)) !=
            hipSuccess) {
        check(false, "row-block buffers allocated");
        return false;
    }
    auto release = [&]() {
        (void)hipFree(aql_out);
        (void)hipFree(host_out);
        (void)hipFree(a8);
        (void)hipFree(input);
    };

    std::vector<uint16_t> h_input(static_cast<size_t>(rows) * kFeatures);
    for (size_t i = 0; i < h_input.size(); ++i) {
        const uint32_t h = static_cast<uint32_t>(i) * 2654435761u + 7u;
        const int base = static_cast<int>(h % 2001u) - 1000;
        h_input[i] = chain::bf16_from_float(static_cast<float>(base) / 1000.0f).data;
    }
    (void)hipMemcpy(input, h_input.data(), h_input.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice);

    auto* codes = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(a8) +
                                            layout.codes_offset);
    auto* scales = reinterpret_cast<float*>(static_cast<uint8_t*>(a8) +
                                            layout.scales_offset);

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    const hipError_t qe = kernel::launch_activation_quantize_a8(
        static_cast<const ps::bf16_t*>(input), codes, scales, rows, kFeatures,
        kKp, kFeatures, layout.code_row_stride_bytes,
        layout.scale_row_stride_bytes, stream);
    if (qe != hipSuccess ||
        kernel::launch_gemm_psq4_w4a8_wmma(
            *config, static_cast<const uint8_t*>(w.w_codes),
            static_cast<const uint8_t*>(w.w_scales),
            reinterpret_cast<const int8_t*>(codes), scales, host_out,
            kernel::Psq4GemmOutputDType::BF16, rows, kN, kKp, kWeightScaleStride,
            layout.code_row_stride_bytes, layout.scale_row_stride_bytes, kN,
            stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        check(false, "host row-block launch");
        (void)hipStreamDestroy(stream);
        release();
        return false;
    }

    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_PSQ4_HSACO, symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: %s load %s: %s\n", name, symbol,
                    code_res.status().message().c_str());
        (void)hipStreamDestroy(stream);
        release();
        return false;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();
    auto region_res =
        queue.allocate_kernarg(mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
                               1u);
    if (!region_res.ok()) {
        check(false, "row-block kernarg allocated");
        (void)code.shutdown();
        (void)hipStreamDestroy(stream);
        release();
        return false;
    }
    auto region = region_res.release();

    uint32_t grid_x = 0u;
    uint32_t grid_y = 0u;
    if (out_block != 0u) {
        grid_x = kernel::psq4_prefill2d_grid_x(kN, out_block);
        grid_y = kernel::psq4_prefill2d_grid_y(rows);
    } else {
        grid_x = (kN + 15u) / 16u;
        grid_y = (rows + mb_rows * 16u - 1u) / (mb_rows * 16u);
    }

    unsigned char* slot = region.base;
    std::memset(slot, 0, region.slot_stride);
    auto* args = reinterpret_cast<kernel::Psq4MultiRowBf16Args*>(slot);
    args->weight_codes = w.w_codes;
    args->weight_scales = w.w_scales;
    args->activation_codes = codes;
    args->activation_scales = scales;
    args->output = aql_out;
    args->output_dtype = static_cast<uint32_t>(kernel::Psq4GemmOutputDType::BF16);
    args->rows = rows;
    args->out_features = kN;
    args->k_padded = kKp;
    args->weight_scale_stride = kWeightScaleStride;
    args->activation_code_stride = layout.code_row_stride_bytes;
    args->activation_scale_stride = layout.scale_row_stride_bytes;
    args->output_row_stride = kN;

    mcu::GpuAqlDispatchDesc desc{};
    desc.kernel_object = meta.kernel_object;
    desc.global_work_items_x = grid_x * workgroup;
    desc.global_work_items_y = grid_y;
    desc.global_work_items_z = 1u;
    desc.workgroup_size_x = static_cast<uint16_t>(workgroup);
    desc.workgroup_size_y = 1u;
    desc.workgroup_size_z = 1u;
    desc.private_segment_size = meta.private_segment_size;
    desc.group_segment_size = meta.static_group_segment_size;
    desc.kernarg_address = slot;
    mcu::build_aql_hidden_args(slot, sizeof(kernel::Psq4MultiRowBf16Args),
                               meta.kernarg_segment_size,
                               mcu::AqlHiddenArgsPolicy::IfFits, desc);
    check(!mcu::aql_hidden_args_fits(sizeof(kernel::Psq4MultiRowBf16Args),
                                     meta.kernarg_segment_size),
          "row-block segment excludes hidden args");
    mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid_x, grid_y,
                                   1u, workgroup);
    const mcu::GpuAqlPacketTemplate tmpl =
        mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
    check(queue.submit_host(&tmpl, 1).ok(), "row-block aql submit");
    const uint64_t target = queue.host_queued_index();
    const bool drained = chain::wait_queue_drained(queue, target, 15000);
    check(drained, "row-block aql queue drained");

    const size_t out_bytes = static_cast<size_t>(rows) * kN * sizeof(uint16_t);
    std::vector<uint16_t> h_host(out_bytes / 2u, 0u);
    std::vector<uint16_t> h_aql(out_bytes / 2u, 0u);
    chain::device_to_host_async(h_host.data(), host_out, out_bytes, stream);
    chain::device_to_host_async(h_aql.data(), aql_out, out_bytes, stream);
    size_t first = 0u;
    size_t diffs = 0u;
    const bool same = chain::bytes_equal(h_host.data(), h_aql.data(), out_bytes,
                                         &first, &diffs);
    check(same, "row-block output byte exact: host launcher vs host aql");
    if (diffs != 0u) {
        std::printf("  %s diff count=%zu first=%zu\n", name, diffs, first);
    }
    bool nontrivial = false;
    for (uint16_t v : h_aql) {
        if (v != 0u && v != 0xABABu) {
            nontrivial = true;
            break;
        }
    }
    check(nontrivial, "row-block aql output non-trivial");
    std::printf("  %s rows=%u config=%d symbol=%s grid=(%u,%u) diffs=%zu\n", name,
                rows, static_cast<int>(config->id), symbol, grid_x, grid_y, diffs);

    (void)code.shutdown();
    (void)hipStreamDestroy(stream);
    release();
    return same && nontrivial;
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

    {
        const q35::McuCompiledVariantKind kinds[8] = {
            q35::McuCompiledVariantKind::Psq4RowBlock1Bf16,
            q35::McuCompiledVariantKind::Psq4RowBlock2Bf16,
            q35::McuCompiledVariantKind::Psq4RowBlock4Bf16,
            q35::McuCompiledVariantKind::Psq4RowBlock8Bf16,
            q35::McuCompiledVariantKind::Psq4Prefill2DK64N64,
            q35::McuCompiledVariantKind::Psq4Prefill2DK128N64,
            q35::McuCompiledVariantKind::Psq4Prefill2DK64N128,
            q35::McuCompiledVariantKind::Psq4Prefill2DK128N128};
        const char* symbols[8] = {
            kernel::kPsq4RowBlock1Bf16Symbol, kernel::kPsq4RowBlock2Bf16Symbol,
            kernel::kPsq4RowBlock4Bf16Symbol, kernel::kPsq4RowBlock8Bf16Symbol,
            kernel::kPsq4Prefill2DK64N64Symbol,
            kernel::kPsq4Prefill2DK128N64Symbol,
            kernel::kPsq4Prefill2DK64N128Symbol,
            kernel::kPsq4Prefill2DK128N128Symbol};
        for (int i = 0; i < 8; ++i) {
            auto entry = q35::mcu_kernel_code_object(kinds[i]);
            check(entry.ok(), "row-block registry entry resolves");
            if (!entry.ok()) continue;
            check(std::strcmp(entry.value().symbol, symbols[i]) == 0,
                  "row-block registry symbol matches");
            check(entry.value().kernarg_recipe ==
                      mcu::kMcuKernargRecipePsq4MultiRowBf16,
                  "row-block registry recipe matches");
            check(entry.value().blob.blob != nullptr && entry.value().blob.size != 0u,
                  "row-block embedded blob present");
        }
    }

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    WeightBuffers w;
    if (!w.allocate()) {
        std::printf("FAIL: weight buffers\n");
        return 1;
    }
    w.fill();

    (void)run_case(queue, w, 2u, "rowblock_rows2");
    (void)run_case(queue, w, 8u, "rowblock_rows8");
    (void)run_case(queue, w, 32u, "rowblock_rows32");
    (void)run_case(queue, w, 26u, "rowblock_mixed26");

    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk64bn64", 1);
    (void)run_case(queue, w, 512u, "prefill2d_k64n64_rows512");
    (void)run_case(queue, w, 768u, "prefill2d_k64n64_rows768");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk128", 1);
    (void)run_case(queue, w, 512u, "prefill2d_k128n64_rows512");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bn128", 1);
    (void)run_case(queue, w, 512u, "prefill2d_k64n128_rows512");
    (void)run_case(queue, w, 768u, "prefill2d_k64n128_rows768");
    setenv("PHASESHIFT_PSQ_PREFILL_2D", "bk128bn128", 1);
    (void)run_case(queue, w, 512u, "prefill2d_k128n128_rows512");
    (void)run_case(queue, w, 1024u, "prefill2d_k128n128_rows1024");
    unsetenv("PHASESHIFT_PSQ_PREFILL_2D");

    w.free_all();
    check(queue.shutdown().ok(), "queue shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_psq4_rows_aql: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_psq4_rows_aql: PASS\n");
    return 0;
}
