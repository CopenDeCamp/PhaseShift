#include <phaseshift/runtime/gpu_mcu/execution/persistent_mcu.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <type_traits>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

static_assert(std::is_standard_layout_v<mcu::GpuMcuPersistentState>);
static_assert(sizeof(mcu::GpuMcuPersistentState) % 8u == 0u);

namespace {

using Status = ::ps::Status;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSlots = 4u;
constexpr uint32_t kIngressEntries = 8u;
constexpr uint32_t kAdmission = 8u;


__global__ void fill_events_kernel(mcu::GpuMcuPersistentState* state,
                                   uint32_t count) {
    mcu::ControlRingPayload payload{};
    for (uint32_t i = 0; i < count; ++i) {
        if (!mcu::control_ring_try_push(state->request_event_ring,
                                        state->request_event_position,
                                        payload)) {
            break;
        }
    }
}

__global__ void run_boundary_kernel(mcu::GpuMcuPersistentState* state,
                                    uint32_t passes) {
    for (uint32_t i = 0; i < passes; ++i) {
        if (!mcu::gpu_mcu_scheduler_boundary(state)) break;
    }
}

void* g_prompt_dev = nullptr;
void* g_sampling_dev = nullptr;

uint32_t push_submit(mcu::GpuMcuControlRing& ring,
                     mcu::GpuMcuRequestIngressArena& arena,
                     uint64_t request_id,
                     uint32_t prompt_length,
                     uint32_t stable_prefix_boundary,
                     uint32_t max_new_tokens,
                     uint32_t max_sequence_length) {
    const uint32_t handle = arena.reserve();
    check(handle != 0u, "ingress reserve");
    if (handle == 0u) return 0u;
    mcu::GpuMcuRequestDescriptor* descriptor = arena.descriptor(handle);
    check(descriptor != nullptr, "ingress descriptor");
    if (descriptor == nullptr) return 0u;
    descriptor->request_id = request_id;
    descriptor->prompt_tokens_handle =
        mcu::gpu_mcu_encode_device_address(g_prompt_dev);
    descriptor->prompt_length = prompt_length;
    descriptor->stable_prefix_boundary = stable_prefix_boundary;
    descriptor->sampling_params_handle =
        mcu::gpu_mcu_encode_device_address(g_sampling_dev);
    descriptor->stop_conditions_handle = 0x3000u;
    descriptor->max_new_tokens = max_new_tokens;
    descriptor->max_sequence_length = max_sequence_length;
    const Status published = arena.publish(handle);
    check(published.ok(), "ingress publish");
    mcu::GpuMcuControlCommand command{};
    command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::submit);
    command.version = mcu::kGpuMcuControlVersion;
    command.request_id = request_id;
    command.descriptor_handle = handle;
    check(ring.try_push_input(mcu::encode_control_command(command)),
          "push submit");
    return handle;
}


template <typename F>
bool wait_until(F predicate, uint32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

void push_cancel(mcu::GpuMcuControlRing& ring, uint64_t request_id,
                 uint64_t descriptor_handle, rt::RequestHandle handle) {
    mcu::GpuMcuControlCommand command{};
    command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::cancel);
    command.version = mcu::kGpuMcuControlVersion;
    command.request_id = request_id;
    command.descriptor_handle = descriptor_handle;
    command.target_handle_bits = mcu::encode_target_handle(handle);
    check(ring.try_push_input(mcu::encode_control_command(command)),
          "push cancel");
}

bool pop_event(mcu::GpuMcuControlRing& ring, mcu::GpuMcuEvent& out) {
    mcu::ControlRingPayload payload{};
    if (!ring.try_pop_event(payload)) return false;
    mcu::decode_event(payload, out);
    return true;
}


__global__ void set_output_blocked_kernel(mcu::GpuMcuPersistentState* state,
                                          uint32_t slot,
                                          uint32_t blocked) {
    state->request_slots[slot].output_blocked = blocked;
    state->scheduler_dirty = 1u;
}

uint32_t active_slots(mcu::GpuMcuSlotTable& slots, hipStream_t stream) {
    std::vector<mcu::GpuMcuSlotState> host(kSlots, mcu::GpuMcuSlotState{});
    (void)hipMemcpyAsync(host.data(), slots.device_slots(),
                         kSlots * sizeof(mcu::GpuMcuSlotState),
                         hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
    uint32_t count = 0u;
    for (uint32_t i = 0; i < kSlots; ++i) {
        if (mcu::gpu_mcu_slot_has_active_request(host[i])) ++count;
    }
    return count;
}

}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    hipStream_t control = nullptr;
    check(hipStreamCreate(&control) == hipSuccess, "control stream create");
    hipStream_t work = nullptr;
    check(hipStreamCreate(&work) == hipSuccess, "work stream create");

    auto ring_result = mcu::GpuMcuControlRing::create(0);
    check(ring_result.ok(), "control ring create");
    if (!ring_result.ok()) return 1;
    mcu::GpuMcuControlRing ring = ring_result.release();

    auto slots_result = mcu::GpuMcuSlotTable::create(0, kSlots);
    check(slots_result.ok(), "slot table create");
    if (!slots_result.ok()) return 1;
    mcu::GpuMcuSlotTable slots = slots_result.release();

    auto arena_result =
        mcu::GpuMcuRequestIngressArena::create(0, kIngressEntries);
    check(arena_result.ok(), "ingress arena create");
    if (!arena_result.ok()) return 1;
    mcu::GpuMcuRequestIngressArena arena = arena_result.release();

    const auto map_alloc = [](std::size_t bytes, void** host, void** dev) {
        if (hipHostMalloc(host, bytes,
                          hipHostMallocMapped | hipHostMallocCoherent) !=
            hipSuccess) {
            return false;
        }
        std::memset(*host, 0, bytes);
        return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
    };

    void* plan_context_host = nullptr;
    void* plan_context_dev = nullptr;
    void* plan_requests_host = nullptr;
    void* plan_requests_dev = nullptr;
    void* plan_rows_host = nullptr;
    void* plan_rows_dev = nullptr;
    void* plan_positions_host = nullptr;
    void* plan_positions_dev = nullptr;
    void* plan_epoch_host = nullptr;
    void* plan_epoch_dev = nullptr;
    void* admission_host = nullptr;
    void* admission_dev = nullptr;
    void* bind_table_host = nullptr;
    void* bind_table_dev = nullptr;
    void* prompt_host = nullptr;
    void* prompt_dev = nullptr;
    void* sampling_host = nullptr;
    void* sampling_dev = nullptr;
    void* tokens_host = nullptr;
    void* tokens_dev = nullptr;
    void* orows_host = nullptr;
    void* orows_dev = nullptr;
    void* osamp_host = nullptr;
    void* osamp_dev = nullptr;
    void* ready_host = nullptr;
    void* ready_dev = nullptr;
    check(map_alloc(sizeof(rt::DeviceBatchContext), &plan_context_host,
                    &plan_context_dev),
          "plan context alloc");
    check(map_alloc(kSlots * sizeof(rt::DeviceRequestDescriptor),
                    &plan_requests_host, &plan_requests_dev),
          "plan requests alloc");
    check(map_alloc(32u * sizeof(uint32_t), &plan_rows_host, &plan_rows_dev),
          "plan rows alloc");
    check(map_alloc(32u * sizeof(uint32_t), &plan_positions_host,
                    &plan_positions_dev),
          "plan positions alloc");
    check(map_alloc(sizeof(uint64_t), &plan_epoch_host, &plan_epoch_dev),
          "plan epoch alloc");
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding), &bind_table_host,
                    &bind_table_dev),
          "slot bindings alloc");
    check(map_alloc(256u * sizeof(int32_t), &prompt_host, &prompt_dev),
          "prompt resource alloc");
    check(map_alloc(sizeof(rt::DeviceSamplingParams), &sampling_host,
                    &sampling_dev),
          "sampling resource alloc");
    check(map_alloc(32u * sizeof(int32_t), &tokens_host, &tokens_dev),
          "token binding alloc");
    check(map_alloc(32u * sizeof(uint32_t), &orows_host, &orows_dev),
          "output rows alloc");
    check(map_alloc(32u * sizeof(rt::DeviceSamplingParams), &osamp_host,
                    &osamp_dev),
          "output sampling alloc");
    check(map_alloc(sizeof(uint64_t), &ready_host, &ready_dev),
          "ready epoch alloc");
    for (uint32_t i = 0; i < 256u; ++i) {
        static_cast<int32_t*>(prompt_host)[i] = 7000 + static_cast<int32_t>(i);
    }
    g_prompt_dev = prompt_dev;
    g_sampling_dev = sampling_dev;

    rt::RequestHandle* runnable = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&runnable),
                    kSlots * sizeof(rt::RequestHandle)) == hipSuccess,
          "runnable snapshot alloc");
    check(map_alloc(kAdmission * sizeof(mcu::GpuMcuPendingAdmission),
                    &admission_host, &admission_dev),
          "admission alloc");

    auto mcu_result = mcu::GpuMcuPersistentMcu::create(0, control);
    check(mcu_result.ok(), "persistent mcu create");
    if (!mcu_result.ok()) return 1;
    mcu::GpuMcuPersistentMcu persistent = mcu_result.release();

    check(!persistent.request_runtime_enabled(),
          "request runtime is disabled before configure");

    const Status configured = persistent.configure_request_runtime(
        ring.device_input(), ring.device_events(), arena.device_entries(),
        arena.capacity(), slots.device_slots(), slots.max_slots(), runnable,
        kSlots);
    check(configured.ok(), "configure_request_runtime");
    check(persistent.configure_admission(
                  static_cast<mcu::GpuMcuPendingAdmission*>(admission_dev),
                  kAdmission)
              .ok(),
          "configure admission");
    check(persistent.request_runtime_enabled(),
          "request runtime is enabled after configure");
    check(persistent.request_input_ring() == ring.device_input(),
          "input ring is stored");
    check(persistent.request_event_ring() == ring.device_events(),
          "event ring is stored");
    check(persistent.request_slots() == slots.device_slots(),
          "slot table is stored");
    check(persistent.request_max_slots() == kSlots, "max slots is stored");
    check(persistent.runnable_handles() == runnable,
          "runnable snapshot is stored");
    check(persistent.runnable_capacity() == kSlots,
          "runnable capacity is stored");
    check(persistent.runnable_count() == 0u,
          "runnable count starts empty");

    const Status plan_configured = persistent.configure_batch_plan(
        static_cast<rt::DeviceBatchContext*>(plan_context_dev),
        static_cast<rt::DeviceRequestDescriptor*>(plan_requests_dev),
        static_cast<uint32_t*>(plan_rows_dev),
        static_cast<uint32_t*>(plan_positions_dev), kSlots, 32u, 0u,
        static_cast<uint64_t*>(plan_epoch_dev));
    check(plan_configured.ok(), "configure_batch_plan");
    check(persistent.batch_plan_enabled(), "batch plan is enabled");

    const Status binding_configured = persistent.configure_batch_binding(
        static_cast<mcu::GpuMcuSlotBinding*>(bind_table_dev), kSlots,
        static_cast<int32_t*>(tokens_dev),
        static_cast<uint32_t*>(orows_dev),
        static_cast<rt::DeviceSamplingParams*>(osamp_dev), 32u,
        static_cast<uint64_t*>(ready_dev));
    check(binding_configured.ok(), "configure_batch_binding");
    check(persistent.batch_binding_enabled(), "batch binding is enabled");

    auto output_result = mcu::GpuMcuOutputRing::create(0);
    if (!output_result.ok()) {
        std::printf("FAIL: output ring create: %s\n",
                    output_result.status().message().c_str());
        return 1;
    }
    mcu::GpuMcuOutputRing outputs = output_result.release();
    check(persistent.configure_output(outputs.device_ring()).ok(),
          "configure_output");

    const Status started = persistent.start();
    check(started.ok(), "persistent mcu start");
    if (!started.ok()) {
        std::printf("test_gpu_mcu_backlog_cancel: start failed\n");
        return 1;
    }

    check(persistent.request_runtime_enabled(),
          "request runtime survives the start reset");
    check(persistent.request_input_ring() == ring.device_input(),
          "input ring survives the start reset");
    check(persistent.request_event_ring() == ring.device_events(),
          "event ring survives the start reset");
    check(persistent.request_slots() == slots.device_slots(),
          "slot table survives the start reset");
    check(persistent.request_max_slots() == kSlots,
          "max slots survives the start reset");
    check(persistent.runnable_handles() == runnable,
          "runnable snapshot survives the start reset");
    check(persistent.runnable_capacity() == kSlots,
          "runnable capacity survives the start reset");
    check(persistent.runnable_count() == 0u, "runnable count starts empty");
    check(persistent.scheduler_epoch() <= 1u,
          "the epoch only carries the initial snapshot at this point");
    check(persistent.commands_consumed() == 0u, "no command consumed yet");
    check(persistent.events_published() == 0u, "no event published yet");
    check(!persistent.pending_event_valid(), "no pending event");

    const Status reconfigure = persistent.configure_request_runtime(
        ring.device_input(), ring.device_events(), arena.device_entries(),
        arena.capacity(), slots.device_slots(), slots.max_slots(), runnable,
        kSlots);
    check(!reconfigure.ok(),
          "reconfigure while running is rejected");

    const Status bad = persistent.configure_request_runtime(
        nullptr, ring.device_events(), arena.device_entries(), arena.capacity(),
        slots.device_slots(), slots.max_slots(), runnable, kSlots);
    check(!bad.ok(), "a null pointer is rejected");

    {
        uint64_t backlog_descriptor = 0u;
        for (uint32_t i = 0u; i < 5u; ++i) {
            const uint32_t handle = push_submit(
                ring, arena, 12000u + i, 1u, 0u, 100000u, 200000u);
            if (i == 4u) backlog_descriptor = handle;
        }

        uint64_t claimed_seen = 0u;
        const bool claimed_four = wait_until(
            [&]() {
                mcu::GpuMcuEvent e{};
                while (pop_event(ring, e)) {
                    if (e.opcode ==
                        static_cast<uint32_t>(
                            mcu::GpuMcuEventOpcode::claimed)) {
                        ++claimed_seen;
                    }
                }
                return claimed_seen >= 4u;
            },
            5000u);
        check(claimed_four, "the four slots are claimed");

        rt::RequestHandle stale_handle{};
        stale_handle.slot = 0u;
        stale_handle.generation = 0u;
        push_cancel(ring, 12004u, backlog_descriptor, stale_handle);

        mcu::GpuMcuOutputRecord record{};
        const bool got_terminal = wait_until(
            [&]() { return outputs.try_pop(record); }, 5000u);
        check(got_terminal, "the terminal record reaches the output ring");
        check(record.request_id == 12004u,
              "the record belongs to the cancelled request");
        check((record.flags & mcu::kOutputRecordFlagTerminal) != 0u,
              "the record is a terminal record");
        check(record.terminal_reason ==
                  static_cast<uint32_t>(
                      mcu::GpuMcuTerminalReason::cancelled),
              "the terminal reason is cancelled");
        check(record.token_id == -1,
              "the terminal record carries no token");

        mcu::GpuMcuEvent event{};
        const bool got_event = wait_until(
            [&]() { return pop_event(ring, event); }, 5000u);
        check(got_event, "the cancel event reaches the host");
        check(event.opcode ==
                  static_cast<uint32_t>(
                      mcu::GpuMcuEventOpcode::cancel_accepted),
              "the backlog cancel is accepted");
        check(event.request_id == 12004u,
              "the cancel event carries the request id");
        check(!persistent.pending_event_valid(),
              "the cancel event does not stay pending");
    }

    const Status stop = persistent.request_stop();
    check(stop.ok(), "stop requested");
    check(persistent.wait_stopped(5000).ok(), "persistent mcu stopped");

    std::printf(
        "test_gpu_mcu_backlog_cancel: PASS consumed=%llu published=%llu\n",
        static_cast<unsigned long long>(persistent.commands_consumed()),
        static_cast<unsigned long long>(persistent.events_published()));


    const Status down = persistent.shutdown();
    check(down.ok(), "shutdown");

    if (runnable != nullptr) (void)hipFree(runnable);
    if (admission_host != nullptr) (void)hipFree(admission_host);
    const Status arena_shutdown = arena.shutdown();
    check(arena_shutdown.ok(), "arena shutdown");
    const Status slot_shutdown = slots.shutdown();
    check(slot_shutdown.ok(), "slot shutdown");
    const Status output_shutdown = outputs.shutdown();
    check(output_shutdown.ok(), "output ring shutdown");
    const Status ring_shutdown = ring.shutdown();
    check(ring_shutdown.ok(), "ring shutdown");
    if (plan_context_host != nullptr) (void)hipHostFree(plan_context_host);
    if (plan_requests_host != nullptr) (void)hipHostFree(plan_requests_host);
    if (plan_rows_host != nullptr) (void)hipHostFree(plan_rows_host);
    if (plan_positions_host != nullptr) (void)hipHostFree(plan_positions_host);
    if (plan_epoch_host != nullptr) (void)hipHostFree(plan_epoch_host);
    if (bind_table_host != nullptr) (void)hipHostFree(bind_table_host);
    if (prompt_host != nullptr) (void)hipHostFree(prompt_host);
    if (sampling_host != nullptr) (void)hipHostFree(sampling_host);
    if (tokens_host != nullptr) (void)hipHostFree(tokens_host);
    if (orows_host != nullptr) (void)hipHostFree(orows_host);
    if (osamp_host != nullptr) (void)hipHostFree(osamp_host);
    if (ready_host != nullptr) (void)hipHostFree(ready_host);
    (void)hipStreamDestroy(work);
    (void)hipStreamDestroy(control);

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_backlog_cancel: failed=%d\n", g_failed);
        return 1;
    }
    return 0;
}
