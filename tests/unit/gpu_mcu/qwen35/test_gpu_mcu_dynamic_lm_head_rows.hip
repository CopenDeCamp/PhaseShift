#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/runtime/mcu_plan_compiler.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/gpu_mcu/binding/plan_binder.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint16_t f32_bf16(float f) {
    return __bfloat16_as_ushort(__float2bfloat16(f));
}

bool run_rows(uint32_t rows, uint32_t N, uint32_t K, uint32_t seed,
              ps::gpu::GpuArena& arena, hipStream_t stream, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.3f, 0.3f);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<uint16_t> w(static_cast<size_t>(N) * K);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * K);
    for (auto& v : w) v = f32_bf16(wd(rng));
    for (auto& v : x) v = f32_bf16(xd(rng));

    auto wr = arena.allocate_aligned(w.size() * 2u, 256);
    auto xr = arena.allocate_aligned(x.size() * 2u, 256);
    auto exact_r = arena.allocate_aligned(static_cast<size_t>(rows) * N * 2u, 256);
    auto ref_r = arena.allocate_aligned(static_cast<size_t>(rows) * N * 2u, 256);
    if (!wr.ok() || !xr.ok() || !exact_r.ok() || !ref_r.ok()) {
        check(false, label);
        return false;
    }
    uint16_t* d_w = static_cast<uint16_t*>(wr.release().data());
    uint16_t* d_x = static_cast<uint16_t*>(xr.release().data());
    void* d_exact = exact_r.release().data();
    void* d_ref = ref_r.release().data();
    if (hipMemcpy(d_w, w.data(), w.size() * 2u, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(d_x, x.data(), x.size() * 2u, hipMemcpyHostToDevice) !=
            hipSuccess) {
        check(false, label);
        return false;
    }
    const ::ps::bf16_t* dw = reinterpret_cast<const ::ps::bf16_t*>(d_w);
    const ::ps::bf16_t* dx = reinterpret_cast<const ::ps::bf16_t*>(d_x);
    const auto odt = ps::kernel::Bf16GemmOutputDType::BF16;

    const ps::kernel::Bf16GemmConfig exact_cfg{
        ps::kernel::Bf16GemmConfigId::ExactRows, static_cast<uint8_t>(rows)};
    if (::ps::kernel::launch_gemm_bf16(exact_cfg, dw, dx, d_exact, odt, rows, N,
                                       K, K, N, stream) != hipSuccess) {
        check(false, label);
        return false;
    }
    const ps::kernel::Bf16GemmConfig row_cfg{ps::kernel::Bf16GemmConfigId::ExactRows,
                                             1u};
    for (uint32_t m = 0; m < rows; ++m) {
        void* dst = static_cast<uint8_t*>(d_ref) + static_cast<std::size_t>(m) * N * 2u;
        if (::ps::kernel::launch_gemm_bf16(row_cfg, dw,
                                           dx + static_cast<std::size_t>(m) * K,
                                           dst, odt, 1u, N, K, K, N,
                                           stream) != hipSuccess) {
            check(false, label);
            return false;
        }
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        check(false, label);
        return false;
    }
    const std::size_t out_bytes = static_cast<std::size_t>(rows) * N * 2u;
    std::vector<uint8_t> exact(out_bytes);
    std::vector<uint8_t> ref(out_bytes);
    if (hipMemcpy(exact.data(), d_exact, out_bytes, hipMemcpyDeviceToHost) !=
            hipSuccess ||
        hipMemcpy(ref.data(), d_ref, out_bytes, hipMemcpyDeviceToHost) !=
            hipSuccess) {
        check(false, label);
        return false;
    }
    check(std::memcmp(exact.data(), ref.data(), out_bytes) == 0, label);
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    {
        const char* symbols[16] = {
            ps::kernel::kBf16ExactRowsSymbols[0],
            ps::kernel::kBf16ExactRowsSymbols[1],
            ps::kernel::kBf16ExactRowsSymbols[2],
            ps::kernel::kBf16ExactRowsSymbols[3],
            ps::kernel::kBf16ExactRowsSymbols[4],
            ps::kernel::kBf16ExactRowsSymbols[5],
            ps::kernel::kBf16ExactRowsSymbols[6],
            ps::kernel::kBf16ExactRowsSymbols[7],
            ps::kernel::kBf16ExactRowsSymbols[8],
            ps::kernel::kBf16ExactRowsSymbols[9],
            ps::kernel::kBf16ExactRowsSymbols[10],
            ps::kernel::kBf16ExactRowsSymbols[11],
            ps::kernel::kBf16ExactRowsSymbols[12],
            ps::kernel::kBf16ExactRowsSymbols[13],
            ps::kernel::kBf16ExactRowsSymbols[14],
            ps::kernel::kBf16ExactRowsSymbols[15],
        };
        bool distinct = true;
        for (uint32_t i = 0; i < 16u; ++i) {
            for (uint32_t j = i + 1u; j < 16u; ++j) {
                if (std::strcmp(symbols[i], symbols[j]) == 0) distinct = false;
            }
        }
        check(distinct, "bf16 exact rows symbols are distinct");
    }

    namespace mcu = ::ps::runtime::gpu_mcu;
    namespace rt = ::ps::runtime;
    {
        rt::DeviceBatchContext ctx{};
        const uint8_t source = static_cast<uint8_t>(
            mcu::McuInvocationPatchSource::Bf16ExactRowsVariant);
        constexpr uint32_t kBase = 7u;
        ctx.num_outputs = 11u;
        ctx.actual_rows = 26u;
        check(mcu::gpu_mcu_patch_source_value(source, kBase, &ctx) == kBase + 10u,
              "variant source maps 11 outputs");
        ctx.num_outputs = 1u;
        check(mcu::gpu_mcu_patch_source_value(source, kBase, &ctx) == kBase,
              "variant source maps a single output");
        ctx.num_outputs = 16u;
        check(mcu::gpu_mcu_patch_source_value(source, kBase, &ctx) == kBase + 15u,
              "variant source maps 16 outputs");
        ctx.num_outputs = 17u;
        check(mcu::gpu_mcu_patch_source_value(source, kBase, &ctx) == 0xFFFFFFFFu,
              "variant source rejects more than 16 outputs");
        ctx.num_outputs = 0u;
        ctx.actual_rows = 4u;
        check(mcu::gpu_mcu_patch_source_value(source, kBase, &ctx) == kBase + 3u,
              "variant source falls back to actual rows");
        check(mcu::gpu_mcu_patch_source_value(source, kBase, nullptr) == 0u,
              "variant source tolerates a null context");
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;
    auto arena_result =
        ps::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("FAIL: arena\n");
        return 1;
    }
    ps::gpu::GpuArena arena = arena_result.release();

    const uint32_t K = 256u;
    const uint32_t N = 256u;

    {
        namespace q35 = ::ps::qwen35::runtime;
        constexpr uint32_t kPlanN = 256u;
        constexpr uint32_t kPlanK = 5120u;
        rt::Program program{};
        program.row_bucket = rt::RowBucket::R16;
        program.execution_class = rt::ExecutionClass::DECODE;
        program.weight_slot_count = 1u;

        rt::ValueBinding in{};
        in.logical_value = rt::ValueId{0u};
        in.storage = rt::ValueStorage::EXTERNAL_INPUT;
        in.dtype = rt::ValueDType::BF16;
        in.row_domain = rt::ValueRowDomain::TOKEN_ROWS;
        in.slot = 0u;
        in.feature_count = kPlanK;
        in.row_stride = kPlanK;
        program.values.push_back(in);

        rt::ValueBinding out{};
        out.logical_value = rt::ValueId{1u};
        out.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
        out.dtype = rt::ValueDType::F32;
        out.row_domain = rt::ValueRowDomain::OUTPUT_ROWS;
        out.slot = 0u;
        out.feature_count = kPlanN;
        out.row_stride = kPlanN;
        program.values.push_back(out);

        rt::DispatchBinding lin{};
        lin.kernel_id = rt::KernelId::LINEAR_BF16;
        lin.input_count = 1u;
        lin.input_slots[0] = 0u;
        lin.output_count = 1u;
        lin.output_slots[0] = 1u;
        lin.weight_index = 0u;
        program.dispatches.push_back(lin);
        program.dispatch_source_node.push_back(0u);

        void* weight_dev = nullptr;
        void* input_dev = nullptr;
        void* output_dev = nullptr;
        check(hipMalloc(&weight_dev, static_cast<size_t>(kPlanN) * kPlanK * 2u) ==
                      hipSuccess &&
                  hipMalloc(&input_dev, 16u * kPlanK * 2u) == hipSuccess &&
                  hipMalloc(&output_dev, 16u * kPlanN * 4u) == hipSuccess,
              "lm-head plan buffers");

        rt::WeightSlot weights[1]{};
        weights[0].encoding = 0u;
        weights[0].device_ptr = weight_dev;
        weights[0].rows = kPlanN;
        weights[0].cols = kPlanK;

        const void* ext_in[1] = {input_dev};
        void* ext_out[1] = {output_dev};
        q35::HostExecutionContext ctx{};
        ctx.external_inputs = ext_in;
        ctx.external_input_count = 1u;
        ctx.external_outputs = ext_out;
        ctx.external_output_count = 1u;
        ctx.weights = weights;
        ctx.weight_count = 1u;
        ctx.actual_rows = 16u;
        ctx.actual_outputs = 11u;
        ctx.role = rt::ExecutionRole::Verify;
        ctx.numeric_mode = rt::VerifyNumericMode::Exact;

        q35::McuPlanCompileOptions opts{};
        opts.dispatch_begin = 0u;
        opts.dispatch_end = 1u;
        q35::McuCompiledPlan plan{};
        const q35::McuStaticPlanCompileContext ctx_static =
            q35::make_static_plan_compile_context(
                ctx, ctx.actual_rows, ctx.actual_outputs,
                ctx.staged_request_count,
                ctx.actual_stochastic_outputs);
        const ps::Status st = q35::compile_mcu_plan(program,ctx_static, opts, plan);
        if (!st.ok()) std::printf("compile error: %s\n", st.message().c_str());
        check(st.ok(), "bf16 lm-head plan compiles");
        check(plan.bf16_variant_catalogs.size() == 1u,
              "bf16 lm-head catalog is emitted");
        if (plan.bf16_variant_catalogs.size() == 1u) {
            const q35::McuBf16VariantCatalog& cat =
                plan.bf16_variant_catalogs[0];
            check(static_cast<uint32_t>(
                      plan.nodes[cat.node_index].variant_id) ==
                      cat.variant_base + 10u,
                  "lm-head node carries the 11-output variant");
            bool kinds_ok = true;
            for (uint32_t r = 1u; r <= 16u; ++r) {
                const uint16_t id =
                    static_cast<uint16_t>(cat.variant_base + (r - 1u));
                if (id >= plan.variants.size() ||
                    plan.variants[id].kind !=
                        q35::kMcuBf16ExactRowsKinds[r - 1u]) {
                    kinds_ok = false;
                }
            }
            check(kinds_ok, "catalog rows 1..16 map to the exact-rows kinds");
        }
        (void)hipFree(weight_dev);
        (void)hipFree(input_dev);
        (void)hipFree(output_dev);
    }

    for (uint32_t rows = 1u; rows <= 16u; ++rows) {
        char label[96];
        std::snprintf(label, sizeof(label),
                      "exact rows=%u matches the M=1 gemv reference", rows);
        run_rows(rows, N, K, 0x7777u + rows, arena, stream, label);
    }

    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    check(arena.shutdown().ok(), "arena shutdown");
    std::printf("test_gpu_mcu_dynamic_lm_head_rows: %s (%d passed, %d failed)\n",
                failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed > 0 ? 1 : 0;
}
