#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/runtime/gpu_mcu/binding/plan_binder.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using gpu_mcu_test::Rig;
using gpu_mcu_test::Chain;
using gpu_mcu_test::make_chain;
namespace mcu = ps::runtime::gpu_mcu;

constexpr uint32_t kSentinel = 0xffffffffu;
constexpr uint32_t kOutputSpan = 256u;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

__global__ void set_binding_kernel(mcu::McuRuntimeNodeBinding* dst,
                                   mcu::McuRuntimeNodeBinding value) {
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        dst[0] = value;
        __threadfence_system();
    }
}

__global__ void clear_output_kernel(uint32_t* out, uint32_t count,
                                    uint32_t value) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        out[gid] = value;
    }
}

mcu::McuRuntimeNodeBinding binding(uint16_t enabled, uint16_t variant_id,
                                   uint32_t grid_x, uint32_t grid_y,
                                   uint32_t grid_z) {
    mcu::McuRuntimeNodeBinding out{};
    out.enabled = enabled;
    out.variant_id = variant_id;
    out.workgroup_count_x = grid_x;
    out.workgroup_count_y = grid_y;
    out.workgroup_count_z = grid_z;
    return out;
}

bool output_written(const std::vector<uint32_t>& out, uint32_t index) {
    return index < out.size() && out[index] != kSentinel;
}

std::vector<uint32_t> read_span(const Rig& rig) {
    std::vector<uint32_t> out(kOutputSpan, 0u);
    rig.device_to_host(out.data(), rig.output,
                       kOutputSpan * sizeof(uint32_t));
    return out;
}

void expect_written_range(const std::vector<uint32_t>& out, uint32_t begin,
                          uint32_t end, const char* message) {
    bool ok = true;
    for (uint32_t i = begin; i < end; ++i) {
        if (!output_written(out, i)) ok = false;
    }
    check(ok, message);
}

void expect_sentinel_range(const std::vector<uint32_t>& out, uint32_t begin,
                           uint32_t end, const char* message) {
    bool ok = true;
    for (uint32_t i = begin; i < end && i < out.size(); ++i) {
        if (out[i] != kSentinel) ok = false;
    }
    check(ok, message);
}

void clear_and_patch(Rig& rig, const mcu::McuRuntimeNodeBinding& b) {
    hipStream_t stream = rig.partition.worker_stream();
    clear_output_kernel<<<(kOutputSpan + 63u) / 64u, 64u, 0, stream>>>(
        rig.output, kOutputSpan, kSentinel);
    set_binding_kernel<<<1, 32, 0, stream>>>(rig.runtime_bindings, b);
    (void)hipStreamSynchronize(stream);
}

bool run_runtime_case(Rig& rig, const mcu::McuRuntimeNodeBinding& b,
                      uint64_t plans, const char* name) {
    clear_and_patch(rig, b);
    if (!rig.fsm.request_run().ok()) {
        check(false, "request_run failed");
        return false;
    }
    if (!rig.fsm.wait_plans(plans, 10000).ok()) {
        std::printf("FAIL: %s plan wait\n", name);
        ++g_failed;
        return false;
    }
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    return true;
}

bool run_prepared_case(Rig& rig, uint64_t plans, const char* name) {
    hipStream_t stream = rig.partition.worker_stream();
    clear_output_kernel<<<(kOutputSpan + 63u) / 64u, 64u, 0, stream>>>(
        rig.output, kOutputSpan, kSentinel);
    (void)hipStreamSynchronize(stream);
    if (!rig.fsm.request_run().ok()) {
        check(false, "request_run failed");
        return false;
    }
    if (!rig.fsm.wait_plans(plans, 10000).ok()) {
        std::printf("FAIL: %s plan wait\n", name);
        ++g_failed;
        return false;
    }
    check(rig.fsm.fault_code() == mcu::McuFaultCode::none, "no fault");
    return true;
}

bool dynamic_geometry_case() {
    Rig rig;
    if (!rig.open(0, 512u, 2u, 4u, 0u)) {
        std::printf("dynamic binding rig setup failed\n");
        return false;
    }
    Chain chain = make_chain(1u);
    chain.nodes[0].element_count = static_cast<uint16_t>(kOutputSpan);
    chain.nodes[0].value = 1u;
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 32u, 1u)};
    if (!rig.upload(chain.nodes, variants)) return false;

    const mcu::McuRuntimeNodeBinding initial =
        binding(1u, 0u, 0u, 0u, 0u);
    if (!rig.set_runtime_bindings(&initial, 1u)) return false;

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 21u;
    config.template_source = 1u;
    config.copy_lanes = 1u;
    config.lds_budget_bytes = 8192u;
    config.barrier = 1u;
    config.prepared_enabled = 0u;
    if (!rig.start(config)) return false;

    uint64_t plans = 0u;

    if (!run_runtime_case(rig, binding(1u, 0u, 0u, 0u, 0u), ++plans, "static")) {
        return false;
    }
    std::vector<uint32_t> out = read_span(rig);
    expect_written_range(out, 0u, 64u, "static geometry writes 64 threads");
    expect_sentinel_range(out, 64u, kOutputSpan,
                          "static geometry leaves tail untouched");
    check(rig.fsm.dispatches() == 1u, "static geometry one dispatch");

    if (!run_runtime_case(rig, binding(1u, 1u, 0u, 0u, 0u), ++plans,
                          "variant switch")) {
        return false;
    }
    out = read_span(rig);
    expect_written_range(out, 0u, 32u, "variant switch writes 32 threads");
    expect_sentinel_range(out, 32u, kOutputSpan,
                          "variant switch leaves tail untouched");
    check(rig.fsm.dispatches() == 2u, "variant switch one dispatch");

    if (!run_runtime_case(rig, binding(1u, 0u, 2u, 1u, 1u), ++plans,
                          "grid override")) {
        return false;
    }
    out = read_span(rig);
    expect_written_range(out, 0u, 128u, "grid override writes 128 threads");
    expect_sentinel_range(out, 128u, kOutputSpan,
                          "grid override leaves tail untouched");
    check(rig.fsm.dispatches() == 3u, "grid override one dispatch");

    const uint64_t before = rig.fsm.dispatches();
    if (!run_runtime_case(rig, binding(0u, 0u, 0u, 0u, 0u), ++plans,
                          "disabled")) {
        return false;
    }
    out = read_span(rig);
    expect_sentinel_range(out, 0u, kOutputSpan, "disabled dispatch skipped");
    check(rig.fsm.dispatches() == before, "disabled dispatch not committed");
    check(rig.fsm.completions_observed() == 3u,
          "disabled dispatch observes no completion");
    check(rig.fsm.plans_started() == plans, "plans started match");

    check(rig.close(), "dynamic binding rig teardown");
    return true;
}

bool prepared_prefetch_case() {
    Rig rig;
    if (!rig.open(0, 512u, 2u, 4u, 0u)) {
        std::printf("prepared binding rig setup failed\n");
        return false;
    }
    Chain chain = make_chain(2u);
    chain.nodes[0].element_count = 1u;
    chain.nodes[0].value = 7u;
    chain.nodes[2].element_count = static_cast<uint16_t>(kOutputSpan);
    chain.nodes[2].value = 3u;
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u),
        mcu::make_probe_variant(1u, rig.meta, 32u, 1u)};
    if (!rig.upload(chain.nodes, variants)) return false;

    std::vector<mcu::McuRuntimeNodeBinding> bindings(5u);
    bindings[0] = binding(1u, 0u, 0u, 0u, 0u);
    bindings[1] = binding(1u, 0u, 0u, 0u, 0u);
    bindings[2] = binding(1u, 1u, 4u, 1u, 1u);
    bindings[3] = binding(1u, 0u, 0u, 0u, 0u);
    bindings[4] = binding(1u, 0u, 0u, 0u, 0u);
    if (!rig.set_runtime_bindings(bindings.data(),
                                  static_cast<uint32_t>(bindings.size()))) {
        return false;
    }

    mcu::GpuMcuFsmConfig config{};
    config.plan_id = 22u;
    config.template_source = 1u;
    config.copy_lanes = 1u;
    config.lds_budget_bytes = 8192u;
    config.barrier = 1u;
    config.prepared_enabled = 1u;
    if (!rig.start(config)) return false;

    if (!run_prepared_case(rig, 1u, "prepared prefetch")) return false;
    const std::vector<uint32_t> out = read_span(rig);
    check(out.size() > 1u && out[0] == 7u, "first dispatch value");
    check(out.size() > 2u && out[2] == 10u, "prefetched dispatch base");
    expect_written_range(out, 2u, 130u,
                         "prefetched grid override writes 128 threads");
    expect_sentinel_range(out, 130u, kOutputSpan,
                          "prefetched grid override leaves tail untouched");
    check(rig.fsm.dispatches() == 2u, "prepared two dispatches");
    check(rig.fsm.completions_observed() == 2u, "prepared two completions");
    check(rig.fsm.plans_started() == 1u, "prepared one plan");

    check(rig.close(), "prepared binding rig teardown");
    return true;
}

}  // namespace

void attention_path_binding_case() {
    mcu::McuRuntimeNodeBinding bindings[4] = {};
    for (auto& b : bindings) {
        b.enabled = 1u;
        b.variant_id = 7u;
    }

    mcu::McuAttentionPathSpan spans[1] = {};
    spans[0].node_begin = 1u;
    spans[0].direct_node_end = 2u;
    spans[0].split_node_end = 4u;
    spans[0].split_min_visible = 2048u;

    ::ps::runtime::DeviceRequestDescriptor requests[1] = {};
    ::ps::runtime::DeviceBatchContext context{};
    context.num_requests = 1u;
    context.requests = requests;

    requests[0].sequence_length = 100u;
    mcu::gpu_mcu_bind_attention_paths(&context, spans, 1u, bindings, 4u);
    check(bindings[0].enabled == 1u, "node outside the span stays enabled");
    check(bindings[1].enabled == 1u,
          "direct node enabled below the split threshold");
    check(bindings[2].enabled == 0u,
          "split node disabled below the split threshold");
    check(bindings[3].enabled == 0u,
          "reduce node disabled below the split threshold");
    check(bindings[1].variant_id == 7u, "binder leaves variant id alone");

    requests[0].sequence_length = 4096u;
    mcu::gpu_mcu_bind_attention_paths(&context, spans, 1u, bindings, 4u);
    check(bindings[1].enabled == 0u,
          "direct node disabled above the split threshold");
    check(bindings[2].enabled == 1u,
          "split node enabled above the split threshold");
    check(bindings[3].enabled == 1u,
          "reduce node enabled above the split threshold");

    spans[0].split_node_end = spans[0].direct_node_end;
    mcu::gpu_mcu_bind_attention_paths(&context, spans, 1u, bindings, 4u);
    check(bindings[1].enabled == 1u,
          "direct only plan keeps its node enabled");

    mcu::gpu_mcu_bind_attention_paths(&context, spans, 0u, bindings, 4u);
    check(bindings[1].enabled == 1u, "empty span table touches nothing");

    ::ps::runtime::DeviceBatchContext empty{};
    empty.num_requests = 1u;
    mcu::gpu_mcu_bind_attention_paths(&empty, spans, 1u, bindings, 4u);
    check(bindings[1].enabled == 1u, "missing request list falls back to direct");
}

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }

    (void)dynamic_geometry_case();
    (void)prepared_prefetch_case();
    attention_path_binding_case();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_dynamic_plan_binding: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_dynamic_plan_binding: PASS\n");
    return 0;
}
