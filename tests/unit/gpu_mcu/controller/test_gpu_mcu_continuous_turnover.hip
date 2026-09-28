#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/runtime/gpu_mcu/output_ring.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>
#include <phaseshift/runtime/gpu_mcu/slot_runtime.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

namespace {

using gpu_mcu_test::Chain;
using gpu_mcu_test::make_chain;
using gpu_mcu_test::Rig;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

constexpr uint32_t kSlots = 2u;
constexpr uint32_t kIngress = 8u;
constexpr uint32_t kAdmission = 8u;
constexpr uint32_t kRequests = 20u;
constexpr uint32_t kMaxRows = 4u;
constexpr uint32_t kMaxOutputs = 4u;
constexpr int32_t kPromptToken = 700;

void* g_prompt_dev = nullptr;
void* g_sampling_dev = nullptr;

uint32_t push_submit(mcu::GpuMcuControlRing& ring,
                     mcu::GpuMcuRequestIngressArena& arena, uint64_t request_id) {
    const uint32_t handle = arena.reserve();
    if (handle == 0u) return 0u;
    mcu::GpuMcuRequestDescriptor* descriptor = arena.descriptor(handle);
    if (descriptor == nullptr) return 0u;
    descriptor->request_id = request_id;
    descriptor->prompt_tokens_handle =
        mcu::gpu_mcu_encode_device_address(g_prompt_dev);
    descriptor->prompt_length = 1u;
    descriptor->stable_prefix_boundary = 0u;
    descriptor->sampling_params_handle =
        mcu::gpu_mcu_encode_device_address(g_sampling_dev);
    descriptor->max_new_tokens = 1u;
    descriptor->max_sequence_length = 64u;
    if (!arena.publish(handle).ok()) return 0u;
    mcu::GpuMcuControlCommand command{};
    command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::submit);
    command.version = mcu::kGpuMcuControlVersion;
    command.request_id = request_id;
    command.descriptor_handle = handle;
    if (!ring.try_push_input(mcu::encode_control_command(command))) return 0u;
    return handle;
}

constexpr uint32_t kResources = 4u;
constexpr uint32_t kPoolPages = 8u;

__global__ void init_pool_kernel(mcu::GpuMcuKvPagePoolView pool) {
    for (uint32_t i = 0; i < pool.capacity; ++i) {
        pool.free_stack[i] = i;
        pool.owner[i] = 0u;
    }
    *pool.free_count = pool.capacity;
}

__global__ void init_resources_kernel(
    mcu::DeviceSequenceResourceManagerView mgr) {
    for (uint32_t i = 0; i < mgr.resource_count; ++i) {
        mgr.resources[i] = mcu::DeviceSequenceResourceEntry{};
        mgr.free_stack[i] = i;
    }
    *mgr.free_count = mgr.resource_count;
}

bool pop_event(mcu::GpuMcuControlRing& ring, mcu::GpuMcuEvent& out) {
    mcu::ControlRingPayload payload{};
    if (!ring.try_pop_event(payload)) return false;
    mcu::decode_event(payload, out);
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

    Rig rig;
    if (!rig.open(0, 8u, 1u, 8u, 0u)) {
        std::printf("test_gpu_mcu_continuous_turnover: rig setup failed\n");
        return 1;
    }
    Chain chain = make_chain(1u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
    if (!rig.upload(chain.nodes, variants)) return 1;
    mcu::GpuMcuFsmConfig fsm_config{};
    fsm_config.plan_id = 0xcb90001u;
    fsm_config.template_source = 0u;
    fsm_config.copy_lanes = 1u;
    fsm_config.barrier = 1u;
    fsm_config.prepared_enabled = 0u;
    if (!rig.prepare(fsm_config)) {
        std::printf("test_gpu_mcu_continuous_turnover: fsm prepare failed\n");
        return 1;
    }

    hipStream_t work = rig.partition.worker_stream();

    auto ring_result = mcu::GpuMcuControlRing::create(0);
    check(ring_result.ok(), "control ring create");
    if (!ring_result.ok()) return 1;
    mcu::GpuMcuControlRing ring = ring_result.release();

    auto output_result = mcu::GpuMcuOutputRing::create(0);
    check(output_result.ok(), "output ring create");
    if (!output_result.ok()) return 1;
    mcu::GpuMcuOutputRing outputs = output_result.release();

    auto slots_result = mcu::GpuMcuSlotTable::create(0, kSlots);
    check(slots_result.ok(), "slot table create");
    if (!slots_result.ok()) return 1;
    mcu::GpuMcuSlotTable slots = slots_result.release();

    auto arena_result = mcu::GpuMcuRequestIngressArena::create(0, kIngress);
    check(arena_result.ok(), "ingress arena create");
    if (!arena_result.ok()) return 1;
    mcu::GpuMcuRequestIngressArena arena = arena_result.release();

    void* runnable_host = nullptr;
    void* runnable_dev = nullptr;
    void* plan_epoch_host = nullptr;
    void* plan_epoch_dev = nullptr;
    void* ctx_host = nullptr;
    void* ctx_dev = nullptr;
    void* requests_host = nullptr;
    void* requests_dev = nullptr;
    void* rows_host = nullptr;
    void* rows_dev = nullptr;
    void* positions_host = nullptr;
    void* positions_dev = nullptr;
    void* token_ids_host = nullptr;
    void* token_ids_dev = nullptr;
    void* out_rows_host = nullptr;
    void* out_rows_dev = nullptr;
    void* out_sampling_host = nullptr;
    void* out_sampling_dev = nullptr;
    void* bindings_host = nullptr;
    void* bindings_dev = nullptr;
    void* ready_host = nullptr;
    void* ready_dev = nullptr;
    void* prompt_host = nullptr;
    void* prompt_dev = nullptr;
    void* sampling_host = nullptr;
    void* sampling_dev = nullptr;
    void* runtime_host = nullptr;
    void* runtime_dev = nullptr;
    void* admission_host = nullptr;
    void* admission_dev = nullptr;
    void* resource_host = nullptr;
    void* resource_dev = nullptr;
    void* resource_stack_host = nullptr;
    void* resource_stack_dev = nullptr;
    void* resource_count_host = nullptr;
    void* resource_count_dev = nullptr;
    void* pool_stack_host = nullptr;
    void* pool_stack_dev = nullptr;
    void* pool_count_host = nullptr;
    void* pool_count_dev = nullptr;
    void* pool_owner_host = nullptr;
    void* pool_owner_dev = nullptr;
    check(map_alloc(kSlots * sizeof(rt::RequestHandle), &runnable_host,
                    &runnable_dev),
          "runnable alloc");
    check(map_alloc(sizeof(uint64_t), &plan_epoch_host, &plan_epoch_dev),
          "plan epoch alloc");
    check(map_alloc(sizeof(rt::DeviceBatchContext), &ctx_host, &ctx_dev),
          "context alloc");
    check(map_alloc(kSlots * sizeof(rt::DeviceRequestDescriptor), &requests_host,
                    &requests_dev),
          "requests alloc");
    check(map_alloc(kMaxRows * sizeof(uint32_t), &rows_host, &rows_dev),
          "row slots alloc");
    check(map_alloc(kMaxRows * sizeof(uint32_t), &positions_host,
                    &positions_dev),
          "row positions alloc");
    check(map_alloc(kMaxRows * sizeof(int32_t), &token_ids_host, &token_ids_dev),
          "token ids alloc");
    check(map_alloc(kMaxOutputs * sizeof(uint32_t), &out_rows_host,
                    &out_rows_dev),
          "output rows alloc");
    check(map_alloc(kMaxOutputs * sizeof(rt::DeviceSamplingParams),
                    &out_sampling_host, &out_sampling_dev),
          "output sampling alloc");
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding), &bindings_host,
                    &bindings_dev),
          "bindings alloc");
    check(map_alloc(sizeof(uint64_t), &ready_host, &ready_dev),
          "ready epoch alloc");
    check(map_alloc(8u * sizeof(int32_t), &prompt_host, &prompt_dev),
          "prompt alloc");
    check(map_alloc(sizeof(rt::DeviceSamplingParams), &sampling_host,
                    &sampling_dev),
          "sampling alloc");
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotRuntimeState), &runtime_host,
                    &runtime_dev),
          "slot runtime alloc");
    check(map_alloc(kAdmission * sizeof(mcu::GpuMcuPendingAdmission),
                    &admission_host, &admission_dev),
          "admission alloc");
    check(map_alloc(kResources * sizeof(mcu::DeviceSequenceResourceEntry),
                    &resource_host, &resource_dev),
          "resource alloc");
    check(map_alloc(kResources * sizeof(uint32_t), &resource_stack_host,
                    &resource_stack_dev),
          "resource stack alloc");
    check(map_alloc(sizeof(uint32_t), &resource_count_host,
                    &resource_count_dev),
          "resource count alloc");
    check(map_alloc(kPoolPages * sizeof(uint32_t), &pool_stack_host,
                    &pool_stack_dev),
          "pool stack alloc");
    check(map_alloc(sizeof(uint32_t), &pool_count_host, &pool_count_dev),
          "pool count alloc");
    check(map_alloc(kPoolPages * sizeof(uint64_t), &pool_owner_host,
                    &pool_owner_dev),
          "pool owner alloc");

    auto* context = static_cast<rt::DeviceBatchContext*>(ctx_host);
    std::memset(context, 0, sizeof(*context));
    context->token_ids = static_cast<const int32_t*>(token_ids_dev);
    context->requests =
        static_cast<const rt::DeviceRequestDescriptor*>(requests_dev);
    context->row_sequence_slots = static_cast<uint32_t*>(rows_dev);
    context->row_positions = static_cast<uint32_t*>(positions_dev);
    context->output_rows = static_cast<uint32_t*>(out_rows_dev);
    context->output_sampling_params =
        static_cast<rt::DeviceSamplingParams*>(out_sampling_dev);
    for (uint32_t i = 0; i < 8u; ++i) {
        static_cast<int32_t*>(prompt_host)[i] = kPromptToken;
    }
    g_prompt_dev = prompt_dev;
    g_sampling_dev = sampling_dev;
    static_cast<uint64_t*>(ready_host)[0] = 0u;

    auto persistent_result =
        mcu::GpuMcuPersistentMcu::create(0, rig.partition.control_stream());
    check(persistent_result.ok(), "persistent mcu create");
    if (!persistent_result.ok()) return 1;
    mcu::GpuMcuPersistentMcu persistent = persistent_result.release();

    check(persistent
              .configure_request_runtime(
                  ring.device_input(), ring.device_events(),
                  arena.device_entries(), kIngress, slots.device_slots(), kSlots,
                  static_cast<rt::RequestHandle*>(runnable_dev), kSlots)
              .ok(),
          "configure request runtime");
    check(persistent
              .configure_batch_plan(
                  static_cast<rt::DeviceBatchContext*>(ctx_dev),
                  static_cast<rt::DeviceRequestDescriptor*>(requests_dev),
                  static_cast<uint32_t*>(rows_dev),
                  static_cast<uint32_t*>(positions_dev), kSlots, kMaxRows, 0u,
                  static_cast<uint64_t*>(plan_epoch_dev))
              .ok(),
          "configure batch plan");
    check(persistent
              .configure_batch_binding(
                  static_cast<mcu::GpuMcuSlotBinding*>(bindings_dev), kSlots,
                  static_cast<int32_t*>(token_ids_dev),
                  static_cast<uint32_t*>(out_rows_dev),
                  static_cast<rt::DeviceSamplingParams*>(out_sampling_dev),
                  kMaxOutputs, static_cast<uint64_t*>(ready_dev))
              .ok(),
          "configure batch binding");
    check(persistent
              .configure_execution(rig.fsm.device_state(), 1u,
                                   static_cast<uint64_t*>(ready_dev),
                                   reinterpret_cast<const int32_t*>(rig.output),
                                   8u, nullptr, 0u)
              .ok(),
          "configure execution");
    check(persistent.configure_output(outputs.device_ring()).ok(),
          "configure output");
    check(persistent
              .configure_slot_runtime(
                  static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_dev))
              .ok(),
          "configure slot runtime");
    check(persistent
              .configure_admission(
                  static_cast<mcu::GpuMcuPendingAdmission*>(admission_dev),
                  kAdmission)
              .ok(),
          "configure admission");
    check(persistent.configure_control_drain(4u).ok(),
          "configure control drain");
    check(persistent.control_drain_budget() == 4u, "drain budget is set");
    mcu::DeviceSequenceResourceManagerView resources{};
    resources.resources =
        static_cast<mcu::DeviceSequenceResourceEntry*>(resource_dev);
    resources.free_stack = static_cast<uint32_t*>(resource_stack_dev);
    resources.free_count = static_cast<uint32_t*>(resource_count_dev);
    resources.resource_count = kResources;
    init_resources_kernel<<<1, 1, 0, work>>>(resources);
    check(hipStreamSynchronize(work) == hipSuccess, "resource init");
    mcu::GpuMcuKvPagePoolView kv_pages{};
    kv_pages.free_stack = static_cast<uint32_t*>(pool_stack_dev);
    kv_pages.free_count = static_cast<uint32_t*>(pool_count_dev);
    kv_pages.owner = static_cast<uint64_t*>(pool_owner_dev);
    kv_pages.capacity = kPoolPages;
    init_pool_kernel<<<1, 1, 0, work>>>(kv_pages);
    check(hipStreamSynchronize(work) == hipSuccess, "pool init");
    check(persistent.configure_resources(kv_pages, resources).ok(),
          "configure resources");
    check(persistent.resources_enabled(), "resources are enabled");
    check(persistent.start().ok(), "persistent mcu start");

    std::vector<mcu::GpuMcuOutputRecord> tokens;
    std::vector<mcu::GpuMcuOutputRecord> terminals;
    auto collect = [&]() {
        mcu::GpuMcuOutputRecord record{};
        while (outputs.try_pop(record)) {
            if ((record.flags & mcu::kOutputRecordFlagTerminal) != 0u) {
                terminals.push_back(record);
            } else {
                tokens.push_back(record);
            }
        }
    };

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(30);
    uint32_t next = 0u;
    uint32_t admitted = 0u;
    while (std::chrono::steady_clock::now() < deadline) {
        const uint32_t outstanding = next - admitted;
        if (next < kRequests && outstanding < kIngress) {
            const uint32_t handle =
                push_submit(ring, arena, 9200u + next);
            check(handle != 0u, "turnover submit");
            ++next;
        }
        mcu::GpuMcuEvent event{};
        while (pop_event(ring, event)) {
            if (event.opcode ==
                static_cast<uint32_t>(mcu::GpuMcuEventOpcode::claimed)) {
                check(arena.recycle(event.descriptor_handle).ok(),
                      "turnover recycle");
                ++admitted;
            }
        }
        collect();
        if (terminals.size() >= kRequests) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    check(persistent.request_stop().ok(), "stop requested");
    check(persistent.wait_stopped(5000).ok(), "persistent mcu stopped");

    check(admitted == kRequests, "every submitted request is admitted");
    check(terminals.size() >= kRequests, "every request reaches a terminal");
    check(persistent.batches_failed() == 0u, "no failed batch");

    const mcu::GpuMcuExecutionTelemetry telemetry =
        persistent.execution_telemetry();
    std::printf(
        "telemetry: alloc=%llu rel=%llu resv=%llu freed=%llu blocked=%llu "
        "unblocked=%llu\n",
        static_cast<unsigned long long>(telemetry.resources_allocated),
        static_cast<unsigned long long>(telemetry.resources_released),
        static_cast<unsigned long long>(telemetry.kv_pages_reserved),
        static_cast<unsigned long long>(telemetry.kv_pages_released),
        static_cast<unsigned long long>(telemetry.resource_blocked_events),
        static_cast<unsigned long long>(telemetry.resource_unblocked_events));
    check(telemetry.resources_allocated == kRequests,
          "every request allocates a sequence resource");
    check(telemetry.resources_released == kRequests,
          "every sequence resource is released");
    check(telemetry.kv_pages_reserved > 0u, "the loop reserves kv pages");
    check(telemetry.kv_pages_released == telemetry.kv_pages_reserved,
          "every reserved kv page is released");
    check(telemetry.resource_blocked_events == 0u,
          "an eight page pool never blocks");

    // Each request must deliver exactly one terminal record.
    uint32_t duplicates = 0u;
    for (uint32_t i = 0; i < kRequests; ++i) {
        uint32_t seen = 0u;
        for (const mcu::GpuMcuOutputRecord& record : terminals) {
            if (record.request_id == 9200u + i) ++seen;
        }
        if (seen != 1u) ++duplicates;
    }
    check(duplicates == 0u, "each request delivers exactly one terminal");

    // The slot generation advances on every reuse.
    uint32_t out_of_order = 0u;
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
        uint32_t previous = 0u;
        for (const mcu::GpuMcuOutputRecord& record : terminals) {
            const uint32_t record_slot =
                static_cast<uint32_t>(record.request_handle_bits & 0xffffffffu);
            const uint32_t record_generation =
                static_cast<uint32_t>(record.request_handle_bits >> 32u);
            if (record_slot != slot) continue;
            if (record_generation <= previous) ++out_of_order;
            previous = record_generation;
        }
    }
    check(out_of_order == 0u, "the generation advances on every reuse");

    // The host never launched a batch and never selected a slot itself.
    check(persistent.admission_count() == 0u, "the backlog drains completely");
    {
        const auto* count = static_cast<const uint32_t*>(resource_count_host);
        const auto* entries =
            static_cast<const mcu::DeviceSequenceResourceEntry*>(resource_host);
        check(*count == kResources, "every sequence resource is released");
        uint32_t active = 0u;
        for (uint32_t i = 0; i < kResources; ++i) {
            if (entries[i].state == mcu::kGpuMcuResourceActive) ++active;
        }
        check(active == 0u, "no sequence resource stays active");
        const auto* pool_free = static_cast<const uint32_t*>(pool_count_host);
        const auto* pool_owner =
            static_cast<const uint64_t*>(pool_owner_host);
        check(*pool_free == kPoolPages, "every kv page comes back");
        uint32_t owned = 0u;
        for (uint32_t i = 0; i < kPoolPages; ++i) {
            if (pool_owner[i] != 0u) ++owned;
        }
        check(owned == 0u, "no kv page keeps an owner");
    }
    check(persistent.commands_consumed() >= kRequests,
          "every submit reached the device");

    check(persistent.shutdown().ok(), "persistent mcu shutdown");
    check(outputs.shutdown().ok(), "output ring shutdown");
    check(arena.shutdown().ok(), "ingress arena shutdown");
    check(slots.shutdown().ok(), "slot table shutdown");
    check(ring.shutdown().ok(), "control ring shutdown");
    check(rig.close(), "rig teardown");
    check(hipHostFree(runnable_host) == hipSuccess, "runnable free");
    check(hipHostFree(plan_epoch_host) == hipSuccess, "plan epoch free");
    check(hipHostFree(ctx_host) == hipSuccess, "context free");
    check(hipHostFree(requests_host) == hipSuccess, "requests free");
    check(hipHostFree(rows_host) == hipSuccess, "rows free");
    check(hipHostFree(positions_host) == hipSuccess, "positions free");
    check(hipHostFree(token_ids_host) == hipSuccess, "token ids free");
    check(hipHostFree(out_rows_host) == hipSuccess, "out rows free");
    check(hipHostFree(out_sampling_host) == hipSuccess, "out sampling free");
    check(hipHostFree(bindings_host) == hipSuccess, "bindings free");
    check(hipHostFree(ready_host) == hipSuccess, "ready free");
    check(hipHostFree(prompt_host) == hipSuccess, "prompt free");
    check(hipHostFree(sampling_host) == hipSuccess, "sampling free");
    check(hipHostFree(runtime_host) == hipSuccess, "slot runtime free");
    check(hipHostFree(admission_host) == hipSuccess, "admission free");
    check(hipHostFree(resource_host) == hipSuccess, "resource free");
    check(hipHostFree(resource_stack_host) == hipSuccess,
          "resource stack free");
    check(hipHostFree(resource_count_host) == hipSuccess,
          "resource count free");
    check(hipHostFree(pool_stack_host) == hipSuccess, "pool stack free");
    check(hipHostFree(pool_count_host) == hipSuccess, "pool count free");
    check(hipHostFree(pool_owner_host) == hipSuccess, "pool owner free");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_continuous_turnover: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_continuous_turnover: PASS\n");
    return 0;
}
