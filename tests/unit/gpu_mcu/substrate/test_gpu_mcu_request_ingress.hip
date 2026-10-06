#include <phaseshift/runtime/gpu_mcu/request_ingress.h>

#include <hip/hip_runtime.h>

#include <cstddef>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

static_assert(sizeof(mcu::GpuMcuControlCommand) <= 56u);
static_assert(sizeof(mcu::GpuMcuEvent) <= 56u);
static_assert(sizeof(mcu::GpuMcuRequestDescriptor) == 128u);
static_assert(alignof(mcu::GpuMcuRequestDescriptor) == 64u);
static_assert(offsetof(mcu::GpuMcuRequestDescriptor, request_id) == 0u);
static_assert(sizeof(mcu::GpuMcuIngressCursor) == 64u);

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
constexpr uint32_t kIngressEntries = 16u;

struct EventOut {
    bool ok = false;
    mcu::GpuMcuEvent event{};
};

EventOut pop_event(mcu::GpuMcuControlRing& ring) {
    mcu::ControlRingPayload payload{};
    EventOut out{};
    if (!ring.try_pop_event(payload)) return out;
    mcu::decode_event(payload, out.event);
    out.ok = true;
    return out;
}

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
    descriptor->prompt_tokens_handle = 0x1000u;
    descriptor->prompt_length = prompt_length;
    descriptor->stable_prefix_boundary = stable_prefix_boundary;
    descriptor->sampling_params_handle = 0x2000u;
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

void push_cancel(mcu::GpuMcuControlRing& ring,
                 uint64_t request_id,
                 rt::RequestHandle handle) {
    mcu::GpuMcuControlCommand command{};
    command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::cancel);
    command.version = mcu::kGpuMcuControlVersion;
    command.request_id = request_id;
    command.descriptor_handle = 0u;
    command.target_handle_bits = mcu::encode_target_handle(handle);
    check(ring.try_push_input(mcu::encode_control_command(command)),
          "push cancel");
}

struct Harness {
    mcu::GpuMcuControlRing ring;
    mcu::GpuMcuSlotTable slots;
    mcu::GpuMcuRequestIngressArena arena;
    mcu::GpuMcuIngressCursor* cursor = nullptr;
    uint32_t* applied = nullptr;
    hipStream_t stream = nullptr;
    std::vector<mcu::GpuMcuSlotState> host;

    bool open() {
        auto ring_result = mcu::GpuMcuControlRing::create(0);
        check(ring_result.ok(), "control ring create");
        if (!ring_result.ok()) return false;
        ring = ring_result.release();

        auto slots_result = mcu::GpuMcuSlotTable::create(0, kSlots);
        check(slots_result.ok(), "slot table create");
        if (!slots_result.ok()) return false;
        slots = slots_result.release();

        auto arena_result =
            mcu::GpuMcuRequestIngressArena::create(0, kIngressEntries);
        check(arena_result.ok(), "ingress arena create");
        if (!arena_result.ok()) return false;
        arena = arena_result.release();

        auto cursor_result = mcu::gpu_mcu_ingress_cursor_create(0);
        check(cursor_result.ok(), "ingress cursor create");
        if (!cursor_result.ok()) return false;
        cursor = cursor_result.release();

        check(hipMalloc(reinterpret_cast<void**>(&applied), sizeof(uint32_t)) ==
                  hipSuccess,
              "applied alloc");
        check(hipStreamCreate(&stream) == hipSuccess, "stream create");
        host.assign(kSlots, mcu::GpuMcuSlotState{});
        return true;
    }

    bool process(uint32_t max_commands = 1u) {
        const Status status = mcu::gpu_mcu_process_ingress_once(
            ring, cursor, arena, slots, nullptr, 0u, max_commands, stream);
        check(status.ok(), "process ingress once");
        check(hipStreamSynchronize(stream) == hipSuccess, "process sync");
        return true;
    }

    bool refresh() {
        const hipError_t err =
            hipMemcpy(host.data(), slots.device_slots(),
                      kSlots * sizeof(mcu::GpuMcuSlotState),
                      hipMemcpyDeviceToHost);
        check(err == hipSuccess, "slot readback");
        return err == hipSuccess;
    }

    uint32_t active_slots() {
        refresh();
        uint32_t count = 0u;
        for (uint32_t i = 0; i < kSlots; ++i) {
            if (mcu::gpu_mcu_slot_has_active_request(host[i])) ++count;
        }
        return count;
    }

    void close() {
        if (cursor != nullptr) {
            (void)hipFree(cursor);
            cursor = nullptr;
        }
        if (applied != nullptr) {
            (void)hipFree(applied);
            applied = nullptr;
        }
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
            stream = nullptr;
        }
        const Status arena_status = arena.shutdown();
        check(arena_status.ok(), "arena shutdown");
        const Status slot_status = slots.shutdown();
        check(slot_status.ok(), "slot shutdown");
        const Status ring_status = ring.shutdown();
        check(ring_status.ok(), "ring shutdown");
    }
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Harness h{};
    if (!h.open()) {
        std::printf("test_gpu_mcu_request_ingress: setup failed\n");
        return 1;
    }

    const uint32_t entry_a =
        push_submit(h.ring, h.arena, 1001u, 128u, 96u, 64u, 512u);
    check(h.arena.state_of(entry_a) == mcu::kGpuMcuIngressReady,
          "the entry is ready before the device consumes it");
    check(h.active_slots() == 0u, "no slot is claimed before processing");

    h.process();
    {
        const EventOut event = pop_event(h.ring);
        check(event.ok, "claimed event arrives");
        check(event.event.opcode ==
                  static_cast<uint32_t>(mcu::GpuMcuEventOpcode::claimed),
              "event opcode is claimed");
        check(event.event.request_id == 1001u, "event carries the request id");
        check(event.event.slot < kSlots, "event slot is in range");
        check(event.event.generation == 1u, "event generation is 1");
        check(event.event.descriptor_handle == entry_a,
              "event carries the descriptor handle");
        check(h.arena.state_of(entry_a) == mcu::kGpuMcuIngressClaimed,
              "the entry becomes claimed");
        check(h.arena.is_claimed(entry_a), "is_claimed helper");

        h.refresh();
        const mcu::GpuMcuSlotState& slot = h.host[event.event.slot];
        check(slot.phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill),
              "claimed slot is in prefill");
        check(slot.handle.generation == 1u, "claimed generation is 1");
        check(slot.request_id == 1001u, "slot carries the request id");
        check(slot.prompt_length == 128u, "slot carries the prompt length");
        check(slot.stable_prefix_boundary == 96u,
              "slot carries the stable prefix boundary");
        check(slot.max_new_tokens == 64u, "slot carries max new tokens");
        check(slot.max_sequence_length == 512u,
              "slot carries max sequence length");
        check(slot.terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none),
              "claimed slot has no terminal reason");
        check(slot.cancel_requested == 0u, "claimed slot is not cancelled");
        check(slot.output_blocked == 0u, "claimed slot is not blocked");
        check(slot.prefix_restored_length == 0u,
              "no prefix restore happened");
        check(mcu::gpu_mcu_slot_validate(slot), "claimed slot validates");
    }

    {
        const uint32_t handle =
            push_submit(h.ring, h.arena, 1002u, 32u, 0u, 8u, 256u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.request_id == 1002u,
              "second request claims");
        check(event.event.slot == 1u,
              "device first-fit picks the next idle slot");
        check(h.arena.state_of(handle) == mcu::kGpuMcuIngressClaimed,
              "second entry is claimed");
    }

    rt::RequestHandle handle_b{};
    {
        const uint32_t handle =
            push_submit(h.ring, h.arena, 1003u, 16u, 16u, 4u, 128u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.request_id == 1003u,
              "third request claims");
        check(event.event.slot == 2u, "third slot is 2");
        handle_b.slot = event.event.slot;
        handle_b.generation = event.event.generation;
    }
    {
        h.refresh();
        uint32_t distinct = 0u;
        for (uint32_t i = 0; i < kSlots; ++i) {
            if (mcu::gpu_mcu_slot_has_active_request(h.host[i])) ++distinct;
        }
        check(distinct == 3u, "three distinct slots are active");
        for (uint32_t i = 0; i < kSlots; ++i) {
            if (!mcu::gpu_mcu_slot_has_active_request(h.host[i])) continue;
            check(h.host[i].handle.generation == 1u,
                  "every first claim uses generation 1");
        }
    }

    {
        const uint32_t handle =
            push_submit(h.ring, h.arena, 1004u, 8u, 0u, 4u, 64u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.request_id == 1004u,
              "fourth request claims the last slot");
        check(h.arena.state_of(handle) == mcu::kGpuMcuIngressClaimed,
              "fourth entry is claimed");
        check(h.active_slots() == kSlots, "all slots are active");
    }

    {
        const uint32_t handle =
            push_submit(h.ring, h.arena, 2000u, 8u, 0u, 4u, 64u);
        h.refresh();
        const std::vector<mcu::GpuMcuSlotState> before = h.host;
        h.process();
        const EventOut event = pop_event(h.ring);
        check(!event.ok, "a full slot table defers instead of rejecting");
        check(h.arena.state_of(handle) == mcu::kGpuMcuIngressQueued,
              "the deferred descriptor is queued");
        h.refresh();
        for (uint32_t i = 0; i < kSlots; ++i) {
            check(std::memcmp(&before[i], &h.host[i],
                              sizeof(mcu::GpuMcuSlotState)) == 0,
                  "no slot changed when the table is full");
        }
        const Status recycled = h.arena.recycle(handle);
        check(recycled.ok(), "recycle the rejected entry");
    }

    rt::RequestHandle handle_c{};
    rt::RequestHandle handle_d{};
    {
        h.refresh();
        handle_c.slot = handle_b.slot;
        handle_c.generation = handle_b.generation;
        push_cancel(h.ring, 1003u, handle_c);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok, "cancel event arrives");
        check(event.event.opcode ==
                  static_cast<uint32_t>(mcu::GpuMcuEventOpcode::cancel_accepted),
              "cancel is accepted for the active handle");
        h.refresh();
        const mcu::GpuMcuSlotState& slot = h.host[handle_c.slot];
        check(slot.cancel_requested == 1u, "cancel flag is set");
        check(slot.phase == static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill),
              "cancel does not change the phase");
    }

    {
        push_cancel(h.ring, 1003u, rt::RequestHandle{handle_c.slot, 99u});
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.opcode ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuEventOpcode::cancel_stale),
              "a wrong generation is stale");
    }
    {
        push_cancel(h.ring, 4242u, handle_c);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.opcode ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuEventOpcode::cancel_stale),
              "a wrong request id is stale");
    }

    {
        push_submit(h.ring, h.arena, 3000u, 128u, 129u, 8u, 512u);
        h.refresh();
        const uint32_t before_active = h.active_slots();
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.opcode ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuEventOpcode::rejected),
              "an invalid boundary is rejected");
        check(event.event.reason ==
                  static_cast<uint32_t>(mcu::GpuMcuRejectReason::invalid_boundary),
              "reason is invalid_boundary");
        h.refresh();
        check(h.active_slots() == before_active,
              "no slot is claimed for an invalid descriptor");
    }
    {
        push_submit(h.ring, h.arena, 3001u, 8u, 0u, 4u, 0u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.reason ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuRejectReason::invalid_descriptor),
              "a zero max sequence length is rejected");
    }
    {
        push_submit(h.ring, h.arena, 3002u, 512u, 0u, 4u, 64u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.reason ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuRejectReason::invalid_descriptor),
              "prompt longer than max sequence is rejected");
    }
    {
        mcu::GpuMcuRequestDescriptor descriptor{};
        descriptor.prompt_length = 8u;
        descriptor.max_new_tokens = 4u;
        descriptor.max_sequence_length = 12u;
        check(mcu::gpu_mcu_descriptor_validate(descriptor),
              "prompt plus max new equals the sequence limit");
        descriptor.max_new_tokens = 5u;
        check(!mcu::gpu_mcu_descriptor_validate(descriptor),
              "prompt plus max new beyond the sequence limit");
    }
    {
        mcu::GpuMcuRequestDescriptor descriptor{};
        descriptor.prompt_length = 0xFFFFFF00u;
        descriptor.max_new_tokens = 0x200u;
        descriptor.max_sequence_length = 0xFFFFFFFFu;
        check(!mcu::gpu_mcu_descriptor_validate(descriptor),
              "a wrapped prompt plus max new sum is rejected");
    }
    {
        push_submit(h.ring, h.arena, 3004u, 8u, 0u, 5u, 12u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.reason ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuRejectReason::invalid_descriptor),
              "prompt plus max new beyond the limit is rejected on submit");
    }
    {
        mcu::GpuMcuControlCommand command{};
        command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::submit);
        command.version = mcu::kGpuMcuControlVersion + 7u;
        command.request_id = 3003u;
        command.descriptor_handle = 1u;
        check(h.ring.try_push_input(mcu::encode_control_command(command)),
              "push unsupported version");
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.reason ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuRejectReason::unsupported_version),
              "an unsupported version is rejected");
    }

    {
        h.refresh();
        const uint32_t slot_index = handle_c.slot;
        const Status released = mcu::gpu_mcu_slot_table_release(
            h.slots, slot_index, handle_c, h.applied, h.stream);
        check(released.ok(), "release for the stale cancel scenario");
        check(hipStreamSynchronize(h.stream) == hipSuccess, "release sync");

        const uint32_t entry_d =
            push_submit(h.ring, h.arena, 4000u, 16u, 0u, 4u, 128u);
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.request_id == 4000u,
              "request B claims after the release");
        check(event.event.slot == slot_index, "first-fit reuses the slot");
        check(event.event.generation == 2u,
              "the reused slot has generation 2");
        handle_d.slot = event.event.slot;
        handle_d.generation = event.event.generation;

        push_cancel(h.ring, 1003u, handle_c);
        h.process();
        const EventOut stale = pop_event(h.ring);
        check(stale.ok && stale.event.opcode ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuEventOpcode::cancel_stale),
              "the stale cancel is rejected after the reuse");
        h.refresh();
        check(h.host[slot_index].cancel_requested == 0u,
              "request B is not cancelled by the stale handle");
        check(h.host[slot_index].request_id == 4000u,
              "request B keeps its request id");
        check(mcu::gpu_mcu_slot_validate(h.host[slot_index]),
              "request B validates");
        check(h.arena.state_of(entry_d) == mcu::kGpuMcuIngressClaimed,
              "entry D is claimed");
    }

    {
        const Status released_d = mcu::gpu_mcu_slot_table_release(
            h.slots, handle_d.slot, handle_d, h.applied, h.stream);
        check(released_d.ok(), "release request D to free a slot");
        check(hipStreamSynchronize(h.stream) == hipSuccess, "release D sync");
        const uint32_t entry_e =
            push_submit(h.ring, h.arena, 5000u, 8u, 0u, 4u, 128u);
        const uint32_t blocked = h.arena.reserve();
        check(blocked != 0u && blocked != entry_e,
              "a ready entry is not reusable before the claim");
        const Status recycled = h.arena.recycle(blocked);
        check(recycled.ok(), "recycle the reserved probe entry");
        check(h.arena.state_of(entry_e) == mcu::kGpuMcuIngressReady,
              "the submitted entry stays ready until claimed");
        h.process();
        const EventOut event = pop_event(h.ring);
        check(event.ok && event.event.request_id == 5000u &&
                  event.event.opcode ==
                      static_cast<uint32_t>(mcu::GpuMcuEventOpcode::claimed),
              "request E is claimed");
        check(h.arena.state_of(entry_e) == mcu::kGpuMcuIngressClaimed,
              "request E entry is claimed");
        const Status recycled_e = h.arena.recycle(entry_e);
        check(recycled_e.ok(), "recycle after the claim");
        check(h.arena.state_of(entry_e) == mcu::kGpuMcuIngressFree,
              "the entry is free after the recycle");
        const uint32_t reused = h.arena.reserve();
        check(reused == entry_e, "the freed entry is reused");
        const Status recycled_again = h.arena.recycle(reused);
        check(recycled_again.ok(), "recycle the reused entry");
    }

    h.close();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_request_ingress: failed=%d\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_request_ingress: PASS\n");
    return 0;
}
