#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/psq8.h>
#include <phaseshift/models/qwen35/runtime/linear_selector.h>
#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
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
    static constexpr size_t kCodesBytes =
        static_cast<size_t>(kTiles) * kKp * 16u;
    static constexpr size_t kScalesBytes =
        static_cast<size_t>(kTiles) * kNb * 32u;
    bool allocate() {
        return hipMalloc(&w_codes, kCodesBytes) == hipSuccess &&
               hipMalloc(&w_scales, kScalesBytes) == hipSuccess;
    }
    void fill() {
        uint32_t s = 0x2B19u + kKp * 131u + kN;
        std::vector<uint8_t> hc(kCodesBytes);
        for (auto& v : hc) v = chain::code_byte(s);
        (void)hipMemcpy(w_codes, hc.data(), hc.size(), hipMemcpyHostToDevice);
        std::vector<uint8_t> hs(kScalesBytes, 0u);
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

bool symbol_for_config(kernel::Psq8GemmConfigId id, const char** symbol,
                       uint32_t* mb_rows, uint32_t* out_block) {
    *out_block = 0u;
    switch (id) {
        case kernel::Psq8GemmConfigId::RowBlock1:
            *symbol = kernel::kPsq8RowBlock1Bf16Symbol;
            *mb_rows = 1u;
            break;
        case kernel::Psq8GemmConfigId::RowBlock2:
            *symbol = kernel::kPsq8RowBlock2Bf16Symbol;
            *mb_rows = 2u;
            break;
        case kernel::Psq8GemmConfigId::RowBlock4:
            *symbol = kernel::kPsq8RowBlock4Bf16Symbol;
            *mb_rows = 4u;
            break;
        case kernel::Psq8GemmConfigId::RowBlock8:
            *symbol = kernel::kPsq8RowBlock8Bf16Symbol;
            *mb_rows = 8u;
            break;
        case kernel::Psq8GemmConfigId::Prefill2D:
            *symbol = kernel::kPsq8Prefill2DN64K64Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq8GemmPrefill2dBlock64;
            break;
        case kernel::Psq8GemmConfigId::Prefill2D_K64N128:
            *symbol = kernel::kPsq8Prefill2DN128K64Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq8GemmPrefill2dBlock128;
            break;
        case kernel::Psq8GemmConfigId::Prefill2D_K128N128:
            *symbol = kernel::kPsq8Prefill2DN128K128Symbol;
            *mb_rows = 0u;
            *out_block = kernel::kPsq8GemmPrefill2dBlock128;
            break;
        default:
            return false;
    }
    return true;
}

constexpr uint32_t kWorkgroup = 32u;

bool run_case(mcu::GpuMcuAqlQueue& queue, const WeightBuffers& w, uint32_t rows,
              kernel::Psq8GemmOutputDType output_dtype, const char* name) {
    const auto config =
        q35::select_psq8_gemm_config({rows, kN, kFeatures, kKp});
    if (!config) {
        std::printf("FAIL: %s selector rejected the shape\n", name);
        return false;
    }
    const char* symbol = nullptr;
    uint32_t mb_rows = 0u;
    uint32_t out_block = 0u;
    if (!symbol_for_config(config->id, &symbol, &mb_rows, &out_block)) {
        std::printf("FAIL: %s selector chose an unsupported config\n", name);
        return false;
    }

    const uint32_t elem_bytes =
        output_dtype == kernel::Psq8GemmOutputDType::BF16 ? 2u : 4u;
    const auto layout =
        ps::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, rows);
    void* input = nullptr;
    void* a8 = nullptr;
    void* host_out = nullptr;
    void* aql_out = nullptr;
    if (hipMalloc(&input, static_cast<size_t>(rows) * kFeatures * sizeof(uint16_t)) !=
            hipSuccess ||
        hipMalloc(&a8, layout.total_bytes) != hipSuccess ||
        hipMalloc(&host_out, static_cast<size_t>(rows) * kN * elem_bytes) !=
            hipSuccess ||
        hipMalloc(&aql_out, static_cast<size_t>(rows) * kN * elem_bytes) !=
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
        const uint32_t h = static_cast<uint32_t>(i) * 2654435761u + 11u;
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
        kernel::launch_gemm_psq8_w8a8_wmma(
            *config, static_cast<const uint8_t*>(w.w_codes),
            static_cast<const uint8_t*>(w.w_scales),
            reinterpret_cast<const int8_t*>(codes), scales, host_out,
            output_dtype, rows, kN, kKp, kWeightScaleStride,
            layout.code_row_stride_bytes, layout.scale_row_stride_bytes, kN,
            stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        check(false, "host row-block launch");
        (void)hipStreamDestroy(stream);
        release();
        return false;
    }

    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_PSQ8_HSACO, symbol);
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
    uint32_t workgroup = 0u;
    if (out_block != 0u) {
        grid_x = kernel::psq8_prefill2d_grid_x(kN, out_block);
        grid_y = kernel::psq8_prefill2d_grid_y(rows);
        workgroup = kernel::kPsq8Prefill2DThreads;
        check(grid_x != 0u && grid_y != 0u, "prefill2d grid computed");
    } else {
        grid_x = kernel::psq8_rowblock_grid_x(kN);
        grid_y = kernel::psq8_rowblock_grid_y(rows, mb_rows);
        workgroup = kWorkgroup;
    }

    unsigned char* slot = region.base;
    std::memset(slot, 0, region.slot_stride);
    auto* args = reinterpret_cast<kernel::Psq8MultiRowBf16Args*>(slot);
    args->weight_codes = w.w_codes;
    args->weight_scales = w.w_scales;
    args->activation_codes = codes;
    args->activation_scales = scales;
    args->output = aql_out;
    args->output_dtype = static_cast<uint32_t>(output_dtype);
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
    mcu::build_aql_hidden_args(slot, sizeof(kernel::Psq8MultiRowBf16Args),
                               meta.kernarg_segment_size,
                               mcu::AqlHiddenArgsPolicy::IfFits, desc);
    check(!mcu::aql_hidden_args_fits(sizeof(kernel::Psq8MultiRowBf16Args),
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

    const size_t out_bytes = static_cast<size_t>(rows) * kN * elem_bytes;
    std::vector<uint8_t> h_host(out_bytes, 0u);
    std::vector<uint8_t> h_aql(out_bytes, 0u);
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
    for (size_t i = 0; i < out_bytes; ++i) {
        if (h_aql[i] != 0u && h_aql[i] != 0xABu) {
            nontrivial = true;
            break;
        }
    }
    check(nontrivial, "row-block aql output non-trivial");
    std::printf("  %s rows=%u dtype=%s config=%d symbol=%s grid=(%u,%u) diffs=%zu\n",
                name, rows,
                output_dtype == kernel::Psq8GemmOutputDType::BF16 ? "bf16" : "f32",
                static_cast<int>(config->id), symbol, grid_x, grid_y, diffs);

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
        const q35::McuCompiledVariantKind kinds[7] = {
            q35::McuCompiledVariantKind::Psq8RowBlock1Bf16,
            q35::McuCompiledVariantKind::Psq8RowBlock2Bf16,
            q35::McuCompiledVariantKind::Psq8RowBlock4Bf16,
            q35::McuCompiledVariantKind::Psq8RowBlock8Bf16,
            q35::McuCompiledVariantKind::Psq8Prefill2DN64K64,
            q35::McuCompiledVariantKind::Psq8Prefill2DN128K64,
            q35::McuCompiledVariantKind::Psq8Prefill2DN128K128};
        const char* symbols[7] = {
            kernel::kPsq8RowBlock1Bf16Symbol, kernel::kPsq8RowBlock2Bf16Symbol,
            kernel::kPsq8RowBlock4Bf16Symbol, kernel::kPsq8RowBlock8Bf16Symbol,
            kernel::kPsq8Prefill2DN64K64Symbol,
            kernel::kPsq8Prefill2DN128K64Symbol,
            kernel::kPsq8Prefill2DN128K128Symbol};
        for (int i = 0; i < 7; ++i) {
            auto entry = q35::mcu_kernel_code_object(kinds[i]);
            check(entry.ok(), "psq8 registry entry resolves");
            if (!entry.ok()) continue;
            check(std::strcmp(entry.value().symbol, symbols[i]) == 0,
                  "psq8 registry symbol matches");
            check(entry.value().kernarg_recipe ==
                      mcu::kMcuKernargRecipePsq4MultiRowBf16,
                  "psq8 registry recipe matches");
            check(entry.value().blob.blob != nullptr && entry.value().blob.size != 0u,
                  "psq8 embedded blob present");
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

    (void)run_case(queue, w, 1u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows1");
    (void)run_case(queue, w, 2u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows2");
    (void)run_case(queue, w, 8u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows8");
    (void)run_case(queue, w, 32u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows32");
    (void)run_case(queue, w, 26u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows26");
    (void)run_case(queue, w, 64u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows64");
    (void)run_case(queue, w, 128u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows128");
    (void)run_case(queue, w, 65u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows65");
    (void)run_case(queue, w, 66u, kernel::Psq8GemmOutputDType::BF16, "bf16_rows66");
    (void)run_case(queue, w, 65u, kernel::Psq8GemmOutputDType::F32, "f32_rows65");
    (void)run_case(queue, w, 1u, kernel::Psq8GemmOutputDType::F32, "f32_rows1");
    (void)run_case(queue, w, 2u, kernel::Psq8GemmOutputDType::F32, "f32_rows2");
    (void)run_case(queue, w, 8u, kernel::Psq8GemmOutputDType::F32, "f32_rows8");
    (void)run_case(queue, w, 64u, kernel::Psq8GemmOutputDType::F32, "f32_rows64");

    setenv("PHASESHIFT_PSQ8_PREFILL_2D", "bk64bn64", 1);
    (void)run_case(queue, w, 512u, kernel::Psq8GemmOutputDType::BF16,
                   "prefill2d_n64k64_rows512");
    (void)run_case(queue, w, 768u, kernel::Psq8GemmOutputDType::BF16,
                   "prefill2d_n64k64_rows768");
    (void)run_case(queue, w, 512u, kernel::Psq8GemmOutputDType::F32,
                   "prefill2d_n64k64_f32_rows512");
    setenv("PHASESHIFT_PSQ8_PREFILL_2D", "bn128", 1);
    (void)run_case(queue, w, 512u, kernel::Psq8GemmOutputDType::BF16,
                   "prefill2d_n128k64_rows512");
    (void)run_case(queue, w, 768u, kernel::Psq8GemmOutputDType::BF16,
                   "prefill2d_n128k64_rows768");
    setenv("PHASESHIFT_PSQ8_PREFILL_2D", "bk128bn128", 1);
    (void)run_case(queue, w, 512u, kernel::Psq8GemmOutputDType::BF16,
                   "prefill2d_n128k128_rows512");
    (void)run_case(queue, w, 1024u, kernel::Psq8GemmOutputDType::BF16,
                   "prefill2d_n128k128_rows1024");
    (void)run_case(queue, w, 1024u, kernel::Psq8GemmOutputDType::F32,
                   "prefill2d_n128k128_f32_rows1024");
    unsetenv("PHASESHIFT_PSQ8_PREFILL_2D");

    w.free_all();
    check(queue.shutdown().ok(), "queue shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_psq8_rows_aql: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_psq8_rows_aql: PASS\n");
    return 0;
}
