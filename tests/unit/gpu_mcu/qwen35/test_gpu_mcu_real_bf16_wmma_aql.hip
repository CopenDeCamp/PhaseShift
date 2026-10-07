#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/linear/bf16.h>
#include <phaseshift/models/qwen35/runtime/mcu_kernel_registry.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
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

constexpr uint32_t kGuardBytes = 64u;

struct WmmaCase {
    kernel::Bf16GemmConfigId config;
    const char* symbol;
    uint32_t workgroup;
    uint32_t row_tiles;
};

const WmmaCase kVariants[] = {
    {kernel::Bf16GemmConfigId::WmmaKPartition, kernel::kBf16SplitKWmmaSymbol,
     kernel::kBf16SplitKWmmaThreads, 1u},
    {kernel::Bf16GemmConfigId::Wmma, kernel::kBf16WmmaSymbol,
     kernel::kBf16WmmaThreads, 1u},
    {kernel::Bf16GemmConfigId::WmmaWide, kernel::kBf16WmmaWideSymbol,
     kernel::kBf16WmmaThreads, kernel::kBf16WmmaWideRowTiles},
};

constexpr uint32_t kVariantCount =
    static_cast<uint32_t>(sizeof(kVariants) / sizeof(kVariants[0]));

const uint32_t kRows[] = {1u, 2u, 8u, 17u, 64u, 65u, 129u};
constexpr uint32_t kRowCount =
    static_cast<uint32_t>(sizeof(kRows) / sizeof(kRows[0]));
constexpr uint32_t kCaseCount = kVariantCount * kRowCount;

constexpr uint32_t kN = 5120u;
constexpr uint32_t kK = 2560u;

hipError_t host_launch(const WmmaCase& v, uint32_t rows, const void* weight,
                       const void* input, void* output, hipStream_t stream) {
    const kernel::Bf16GemmConfig config{v.config};
    return kernel::launch_gemm_bf16(
        config, static_cast<const ps::bf16_t*>(weight),
        static_cast<const ps::bf16_t*>(input), output,
        kernel::Bf16GemmOutputDType::BF16, rows, kN, kK, kK, kN, stream);
}

void fill_args(const WmmaCase& v, uint32_t rows, const void* weight,
               const void* input, void* output, kernel::Bf16GemmWmmaArgs& a) {
    a = kernel::Bf16GemmWmmaArgs{};
    a.weight = weight;
    a.input = input;
    a.output = output;
    a.output_dtype = static_cast<uint32_t>(kernel::Bf16GemmOutputDType::BF16);
    a.rows = rows;
    a.out_features = kN;
    a.k = kK;
    a.input_row_stride = kK;
    a.output_row_stride = kN;
    (void)v;
}

bool wait_drained(const mcu::GpuMcuAqlQueue& queue, uint64_t target,
                  uint32_t timeout_ms) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (queue.host_consumed_index() < target) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
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
        const q35::McuCompiledVariantKind kinds[3] = {
            q35::McuCompiledVariantKind::Bf16Wmma,
            q35::McuCompiledVariantKind::Bf16WmmaWide,
            q35::McuCompiledVariantKind::Bf16SplitKWmma};
        const char* symbols[3] = {kernel::kBf16WmmaSymbol,
                                  kernel::kBf16WmmaWideSymbol,
                                  kernel::kBf16SplitKWmmaSymbol};
        for (int i = 0; i < 3; ++i) {
            auto entry = q35::mcu_kernel_code_object(kinds[i]);
            check(entry.ok(), "bf16 wmma registry entry resolves");
            if (!entry.ok()) continue;
            check(std::strcmp(entry.value().symbol, symbols[i]) == 0,
                  "bf16 wmma registry symbol matches");
            check(entry.value().kernarg_recipe ==
                      mcu::kMcuKernargRecipeBf16Wmma,
                  "bf16 wmma registry recipe matches");
            check(entry.value().blob.blob != nullptr &&
                      entry.value().blob.size != 0u,
                  "bf16 wmma embedded blob present");
        }
    }

    const size_t weight_bytes = static_cast<size_t>(kN) * kK * 2u;
    const size_t input_bytes = static_cast<size_t>(kK) * 2u * kRows[kRowCount - 1u];

    void* weight = nullptr;
    void* input = nullptr;
    if (hipMalloc(&weight, weight_bytes) != hipSuccess ||
        hipMalloc(&input, input_bytes) != hipSuccess) {
        std::printf("FAIL: buffer alloc\n");
        return 1;
    }
    std::vector<uint16_t> h_weight(static_cast<size_t>(kN) * kK);
    for (size_t i = 0; i < h_weight.size(); ++i)
        h_weight[i] = static_cast<uint16_t>(0x3A00u + ((i * 2654435761u) % 1500u));
    std::vector<uint16_t> h_input(input_bytes / 2u);
    for (size_t i = 0; i < h_input.size(); ++i)
        h_input[i] = static_cast<uint16_t>(0x3A00u + ((i * 40503u) % 1500u));
    (void)hipMemcpy(weight, h_weight.data(), weight_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(input, h_input.data(), input_bytes, hipMemcpyHostToDevice);

    std::vector<void*> out_ref(kCaseCount, nullptr);
    std::vector<void*> out_cand(kCaseCount, nullptr);
    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const uint32_t rows = kRows[i % kRowCount];
        const size_t bytes = static_cast<size_t>(rows) * kN * 2u + kGuardBytes;
        if (hipMalloc(&out_ref[i], bytes) != hipSuccess ||
            hipMalloc(&out_cand[i], bytes) != hipSuccess) {
            std::printf("FAIL: output alloc\n");
            return 1;
        }
    }

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    auto queue_res = mcu::GpuMcuAqlQueue::create(0, mcu::AqlQueueOwner::Host,
                                                 mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    std::vector<std::unique_ptr<mcu::GpuMcuAqlCodeObject>> codes;
    for (uint32_t v = 0; v < kVariantCount; ++v) {
        auto code_res = mcu::GpuMcuAqlCodeObject::load(
            queue, PHASESHIFT_GPU_MCU_BF16_HSACO, kVariants[v].symbol);
        if (!code_res.ok()) {
            std::printf("FAIL: code object %s: %s\n", kVariants[v].symbol,
                        code_res.status().message().c_str());
            return 1;
        }
        std::unique_ptr<mcu::GpuMcuAqlCodeObject> code(
            new mcu::GpuMcuAqlCodeObject(code_res.release()));
        check(code->metadata().kernel_object != 0, "kernel_object resolved");
        check(code->metadata().kernarg_segment_size >=
                  sizeof(kernel::Bf16GemmWmmaArgs),
              "kernarg segment covers Bf16GemmWmmaArgs");
        codes.push_back(std::move(code));
    }

    const uint32_t slot_stride = mcu::aql_kernarg_slot_stride(
        codes[0]->metadata().kernarg_segment_size);
    auto region_res = queue.allocate_kernarg(slot_stride, kCaseCount);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    for (uint32_t i = 0; i < kCaseCount; ++i) {
        const WmmaCase& v = kVariants[i / kRowCount];
        const uint32_t rows = kRows[i % kRowCount];
        const size_t bytes = static_cast<size_t>(rows) * kN * 2u;
        const mcu::GpuAqlKernelMetadata meta = codes[i / kRowCount]->metadata();

        (void)hipMemset(out_ref[i], 0xAB, bytes + kGuardBytes);
        (void)hipMemset(out_cand[i], 0xAB, bytes + kGuardBytes);
        check(host_launch(v, rows, weight, input, out_ref[i], stream) ==
                  hipSuccess,
              "host bf16 wmma launch");
        check(hipStreamSynchronize(stream) == hipSuccess, "host launch sync");

        unsigned char* slot =
            region.base + static_cast<size_t>(i) * region.slot_stride;
        std::memset(slot, 0, region.slot_stride);
        kernel::Bf16GemmWmmaArgs args{};
        fill_args(v, rows, weight, input, out_cand[i], args);
        std::memcpy(slot, &args, sizeof(args));

        const uint32_t grid_x = kernel::bf16_wmma_grid_x(kN);
        const uint32_t grid_y = kernel::bf16_wmma_grid_y(rows, v.row_tiles);
        mcu::GpuAqlDispatchDesc desc{};
        desc.kernel_object = meta.kernel_object;
        desc.global_work_items_x = grid_x * v.workgroup;
        desc.global_work_items_y = grid_y;
        desc.global_work_items_z = 1u;
        desc.workgroup_size_x = static_cast<uint16_t>(v.workgroup);
        desc.workgroup_size_y = 1u;
        desc.workgroup_size_z = 1u;
        desc.private_segment_size = meta.private_segment_size;
        desc.group_segment_size = meta.static_group_segment_size;
        desc.kernarg_address = slot;
        mcu::build_aql_hidden_args(slot, sizeof(kernel::Bf16GemmWmmaArgs),
                                   meta.kernarg_segment_size,
                                   mcu::AqlHiddenArgsPolicy::IfFits, desc);
        mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid_x,
                                       grid_y, 1u, v.workgroup);
        const mcu::GpuAqlPacketTemplate tmpl =
            mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
        check(queue.submit_host(&tmpl, 1).ok(), "bf16 wmma aql submit");
        check(wait_drained(queue, queue.host_queued_index(), 20000),
              "aql queue drained");

        std::vector<uint8_t> ref(bytes, 0u);
        std::vector<uint8_t> cand(bytes, 0u);
        std::vector<uint8_t> ref_guard(kGuardBytes, 0u);
        std::vector<uint8_t> cand_guard(kGuardBytes, 0u);
        (void)hipMemcpy(ref.data(), out_ref[i], bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand.data(), out_cand[i], bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(ref_guard.data(),
                        static_cast<unsigned char*>(out_ref[i]) + bytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(cand_guard.data(),
                        static_cast<unsigned char*>(out_cand[i]) + bytes,
                        kGuardBytes, hipMemcpyDeviceToHost);
        size_t first = 0;
        size_t diffs = 0;
        check(chain::bytes_equal(ref.data(), cand.data(), bytes, &first, &diffs),
              "host launched aql is bit exact");
        if (diffs)
            std::printf("  config=%d rows=%u grid=(%u,%u) diff count=%zu first=%zu\n",
                        static_cast<int>(v.config), rows, grid_x, grid_y, diffs,
                        first);
        check(ref_guard == cand_guard, "aql output sentinel intact");
        std::printf("  config=%d rows=%u symbol=%s grid=(%u,%u) ok=%d\n",
                    static_cast<int>(v.config), rows, v.symbol, grid_x, grid_y,
                    static_cast<int>(diffs == 0u));
    }

    for (auto& code : codes) (void)code->shutdown();
    check(queue.shutdown().ok(), "queue shutdown");
    (void)hipStreamDestroy(stream);
    (void)hipFree(out_ref[0]);
    (void)hipFree(input);
    (void)hipFree(weight);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_bf16_wmma_aql: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_bf16_wmma_aql: PASS\n");
    return 0;
}
