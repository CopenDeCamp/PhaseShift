#include "unit/gpu_mcu/gpu_mcu_real_chain_fixture.h"

#include <phaseshift/models/qwen35/kernels/optimized/linear/psq4.h>
#include <phaseshift/runtime/gpu_mcu/aql.h>

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
        queue, PHASESHIFT_GPU_MCU_PSQ4_HSACO, kernel::kPsq4Decode1Bf16U16Symbol);
    if (!code_res.ok()) {
        std::printf("FAIL: psq4 decode1 code object: %s\n",
                    code_res.status().message().c_str());
        return 1;
    }
    auto code = code_res.release();
    const mcu::GpuAqlKernelMetadata meta = code.metadata();
    std::printf(
        "psq4 decode1 hsaco: kernel_object=%p kernarg=%u private=%u group=%u "
        "sizeof(args)=%zu\n",
        reinterpret_cast<void*>(meta.kernel_object), meta.kernarg_segment_size,
        meta.private_segment_size, meta.static_group_segment_size,
        sizeof(kernel::Psq4Decode1Bf16Args));
    check(meta.kernel_object != 0, "kernel_object resolved");
    check(meta.kernarg_segment_size >= sizeof(kernel::Psq4Decode1Bf16Args),
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
    const hipError_t hq = chain::run_host_quantize_from(host_ws.input, host_ws, stream);
    check(hq == hipSuccess, "host quantize for activation codes");
    const hipError_t hr = chain::run_host_psq4(host_ws, stream);
    check(hr == hipSuccess, "host launch_gemm_psq4_w4a8_wmma_decode1");
    check(hipStreamSynchronize(stream) == hipSuccess, "host decode1 sync");

    const uint32_t grid = kernel::psq4_decode1_grid(chain::kN);
    const uint32_t block = kernel::psq4_decode1_block_size();

    unsigned char* slot = region.base;
    std::memset(slot, 0, region.slot_stride);
    auto* args = reinterpret_cast<kernel::Psq4Decode1Bf16Args*>(slot);
    args->weight_codes = host_ws.w_codes;
    args->weight_scales = host_ws.w_scales;
    args->activation_codes = host_ws.codes;
    args->activation_scales = host_ws.scales;
    args->output = aql_ws.final_out;
    args->k_padded = chain::kKp;
    args->weight_scale_stride = chain::kWeightScaleStride;

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
    mcu::build_aql_hidden_args(slot, sizeof(kernel::Psq4Decode1Bf16Args),
                               meta.kernarg_segment_size,
                               mcu::AqlHiddenArgsPolicy::IfFits, desc);
    check(!mcu::aql_hidden_args_fits(sizeof(kernel::Psq4Decode1Bf16Args),
                                     meta.kernarg_segment_size),
          "psq4 segment excludes hidden args");
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

    const size_t out_bytes = chain::Workspace::kOutBytes;
    std::vector<uint16_t> host_out(out_bytes / 2, 0u);
    std::vector<uint16_t> aql_out(out_bytes / 2, 0u);
    chain::device_to_host_async(host_out.data(), host_ws.final_out, out_bytes,
                                stream);
    chain::device_to_host_async(aql_out.data(), aql_ws.final_out, out_bytes,
                                stream);

    size_t first = 0;
    size_t diffs = 0;
    check(chain::bytes_equal(host_out.data(), aql_out.data(), out_bytes, &first,
                             &diffs),
          "psq4 decode1 bf16 output byte exact: host launcher vs host aql");
    if (diffs) {
        std::printf("  out diff count=%zu first=%zu host=%.9g aql=%.9g\n", diffs,
                    first, chain::f32_from_bf16(host_out[first]),
                    chain::f32_from_bf16(aql_out[first]));
    }

    bool nontrivial = false;
    for (uint16_t v : aql_out) {
        if (v != 0u && v != 0xABABu) {
            nontrivial = true;
            break;
        }
    }
    check(nontrivial, "aql decode1 output non-trivial");

    std::printf("  psq4 k=%u n=%u rows=%u unroll=16 grid=%u block=%u "
                "weight_scale_stride=%u\n",
                chain::kKp, chain::kN, chain::kRows, grid, block,
                chain::kWeightScaleStride);

    (void)hipStreamDestroy(stream);
    check(code.shutdown().ok(), "psq4 code object shutdown");
    check(queue.shutdown().ok(), "aql queue shutdown");
    host_ws.free_all();
    aql_ws.free_all();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_real_psq4_decode1_aql: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_real_psq4_decode1_aql: PASS\n");
    return 0;
}
