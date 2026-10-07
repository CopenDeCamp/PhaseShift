#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/activation_quantize.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace mcu = ps::runtime::gpu_mcu;
namespace kernel = ps::kernel;
namespace chain = gpu_mcu_chain;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
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

    auto queue_res = mcu::GpuMcuAqlQueue::create(
        0, mcu::AqlQueueOwner::Host, mcu::kAqlDefaultQueueSize);
    if (!queue_res.ok()) {
        std::printf("FAIL: queue: %s\n", queue_res.status().message().c_str());
        return 1;
    }
    auto queue = queue_res.release();

    auto code_res = mcu::GpuMcuAqlCodeObject::load(
        queue, PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO,
        kernel::kActivationQuantizeA8Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: activation_quantize code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();
    std::printf(
        "quantize hsaco: kernel_object=%p kernarg=%u private=%u group=%u "
        "sizeof(args)=%zu\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        meta.private_segment_size, meta.static_group_segment_size,
        sizeof(kernel::ActivationQuantizeA8Args));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::ActivationQuantizeA8Args),
          "kernarg segment covers the stable args");

    auto region_res = queue.allocate_kernarg(
        mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size), 1u);
    if (!region_res.ok()) {
        std::printf("FAIL: kernarg: %s\n", region_res.status().message().c_str());
        return 1;
    }
    auto region = region_res.release();

    chain::Workspace host_ws;
    chain::Workspace aql_ws;
    if (!host_ws.allocate() || !aql_ws.allocate()) {
        std::printf("FAIL: workspace alloc\n");
        return 1;
    }
    host_ws.fill_inputs();
    aql_ws.fill_inputs();

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);
    const hipError_t hr =
        chain::run_host_quantize_from(host_ws.input, host_ws, stream);
    check(hr == hipSuccess, "host launch_activation_quantize_a8");
    check(hipStreamSynchronize(stream) == hipSuccess, "host quantize sync");

    const uint32_t grid =
        kernel::activation_quantize_a8_grid(chain::kRows, chain::kKp);
    const uint32_t block = kernel::activation_quantize_a8_block_size();

    unsigned char* slot = region.base;
    std::memset(slot, 0, region.slot_stride);
    auto* args = reinterpret_cast<kernel::ActivationQuantizeA8Args*>(slot);
    args->input = aql_ws.input;
    args->codes = aql_ws.codes;
    args->scales = aql_ws.scales;
    args->rows = chain::kRows;
    args->k = chain::kFeatures;
    args->k_padded = chain::kKp;
    args->input_row_stride = chain::kFeatures;
    args->code_row_stride = chain::Workspace::code_row_stride();
    args->scale_row_stride = chain::Workspace::scale_row_stride();

    mcu::GpuAqlDispatchDesc desc{};
    desc.kernel_object = meta.kernel_object;
    desc.global_work_items_x = grid * block;
    desc.global_work_items_y = 1u;
    desc.global_work_items_z = 1u;
    desc.workgroup_size_x = static_cast<uint16_t>(block);
    desc.workgroup_size_y = 1u;
    desc.workgroup_size_z = 1u;
    desc.private_segment_size = meta.private_segment_size;
    desc.group_segment_size = meta.static_group_segment_size;
    desc.kernarg_address = slot;
    mcu::build_aql_hidden_args(slot, sizeof(kernel::ActivationQuantizeA8Args),
                               meta.kernarg_segment_size,
                               mcu::AqlHiddenArgsPolicy::IfFits, desc);
    mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size, grid, 1u, 1u,
                                   block);

    const mcu::GpuAqlPacketTemplate tmpl =
        mcu::build_aql_packet_template(desc, mcu::kDefaultAqlPolicy);
    check(queue.submit_host(&tmpl, 1).ok(), "host aql submit");
    const uint64_t target = queue.host_queued_index();
    const bool drained = chain::wait_queue_drained(queue, target, 15000);
    if (!drained) {
        std::printf("  drain stalled: write=%llu read=%llu\n",
                    static_cast<unsigned long long>(target),
                    static_cast<unsigned long long>(queue.host_consumed_index()));
    }
    check(drained, "aql queue drained");

    const size_t codes_bytes =
        static_cast<size_t>(chain::Workspace::code_row_stride()) * 16u;
    const size_t scale_bytes = chain::Workspace::scale_row_stride();
    std::vector<uint8_t> host_codes(codes_bytes);
    std::vector<uint8_t> aql_codes(codes_bytes);
    std::vector<uint8_t> host_scales(scale_bytes);
    std::vector<uint8_t> aql_scales(scale_bytes);
    chain::device_to_host_async(host_codes.data(), host_ws.codes, codes_bytes,
                                stream);
    chain::device_to_host_async(aql_codes.data(), aql_ws.codes, codes_bytes,
                                stream);
    chain::device_to_host_async(host_scales.data(), host_ws.scales, scale_bytes,
                                stream);
    chain::device_to_host_async(aql_scales.data(), aql_ws.scales, scale_bytes,
                                stream);

    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(host_codes.data(), aql_codes.data(), codes_bytes,
                             &first, &diffs),
          "a8 codes bit exact: host launcher vs host aql");
    if (diffs) std::printf("  codes diff count=%zu first=%zu\n", diffs, first);
    check(chain::bytes_equal(host_scales.data(), aql_scales.data(), scale_bytes,
                             &first, &diffs),
          "a8 scales bit exact: host launcher vs host aql");
    if (diffs) std::printf("  scales diff count=%zu first=%zu\n", diffs, first);

    bool nontrivial = false;
    for (size_t i = 0; i < codes_bytes; ++i) {
        if (aql_codes[i] != 0u && aql_codes[i] != 0xABu) {
            nontrivial = true;
            break;
        }
    }
    check(nontrivial, "aql codes non-trivial");

    std::printf("  quantize R=%u K=%u Kp=%u grid=%u block=%u codes=%zu scales=%zu\n",
                chain::kRows, chain::kFeatures, chain::kKp, grid, block,
                codes_bytes, scale_bytes);

    (void)hipStreamDestroy(stream);
    check(code.shutdown().ok(), "quantize code object shutdown");
    check(queue.shutdown().ok(), "aql queue shutdown");
    host_ws.free_all();
    aql_ws.free_all();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_activation_quantize_aql: %d failures\n",
                    g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_activation_quantize_aql: PASS\n");
    return 0;
}
