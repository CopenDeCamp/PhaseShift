#include <phaseshift/runtime/gpu_mcu/io/request_ingress.h>
#include <phaseshift/runtime/gpu_mcu/slot_binding.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

static_assert(sizeof(mcu::GpuMcuSlotBinding) == 128u);
static_assert(alignof(mcu::GpuMcuSlotBinding) == 64u);
static_assert(std::is_standard_layout_v<mcu::GpuMcuSlotBinding>);
static_assert(std::is_trivially_copyable_v<mcu::GpuMcuSlotBinding>);
static_assert(sizeof(mcu::GpuMcuRequestDescriptor) == 128u);
static_assert(offsetof(mcu::GpuMcuSlotBinding, prompt_tokens_handle) == 8u);

namespace {

using Status = ::ps::Status;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSlots = 8u;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto bindings_result = mcu::GpuMcuSlotBindingTable::create(0, kSlots);
    check(bindings_result.ok(), "binding table create");
    if (!bindings_result.ok()) {
        std::printf("test_gpu_mcu_slot_binding: setup failed\n");
        return 1;
    }
    mcu::GpuMcuSlotBindingTable bindings = bindings_result.release();
    check(bindings.max_slots() == kSlots, "binding table capacity");

    auto slots_result = mcu::GpuMcuSlotTable::create(0, kSlots);
    check(slots_result.ok(), "slot table create");
    if (!slots_result.ok()) return 1;
    mcu::GpuMcuSlotTable slots = slots_result.release();
    check(bindings.max_slots() == slots.max_slots(),
          "binding table matches the slot table capacity");

    std::vector<mcu::GpuMcuSlotBinding> host(kSlots, mcu::GpuMcuSlotBinding{});
    const hipError_t copied =
        hipMemcpy(host.data(), bindings.device_bindings(),
                  kSlots * sizeof(mcu::GpuMcuSlotBinding), hipMemcpyDeviceToHost);
    check(copied == hipSuccess, "binding readback");
    for (uint32_t i = 0; i < kSlots; ++i) {
        check(host[i].handle.slot == i, "binding slot index");
        check(host[i].handle.generation == rt::kInvalidRequestGeneration,
              "binding generation starts invalid");
        check(host[i].prompt_tokens_handle == 0u, "prompt handle starts zero");
        check(host[i].sampling_params_handle == 0u,
              "sampling handle starts zero");
        check(host[i].stop_conditions_handle == 0u,
              "stop handle starts zero");
        check(host[i].decode_input_token == -1, "decode token starts unset");
        check(host[i].decode_input_valid == 0u, "decode token starts invalid");
    }

    {
        const int probe = 7;
        const uint64_t handle = mcu::gpu_mcu_encode_device_address(&probe);
        check(handle != 0u, "device address encode");
        check(mcu::gpu_mcu_resolve_device_address<int>(handle) == &probe,
              "device address resolve round trip");
        check(mcu::gpu_mcu_resolve_device_address<int>(0u) == nullptr,
              "a zero handle resolves to null");
    }

    {
        mcu::GpuMcuSlotState slot{};
        slot.handle.slot = 3u;
        slot.handle.generation = 9u;
        slot.phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode);
        mcu::GpuMcuSlotBinding binding{};
        binding.handle = rt::RequestHandle{3u, 9u};
        check(mcu::gpu_mcu_binding_valid_for(slot, binding),
              "a matching binding is valid");
        binding.handle = rt::RequestHandle{3u, 8u};
        check(!mcu::gpu_mcu_binding_valid_for(slot, binding),
              "a stale generation is rejected");
        slot.phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::idle);
        binding.handle = rt::RequestHandle{3u, 9u};
        check(!mcu::gpu_mcu_binding_valid_for(slot, binding),
              "an idle slot has no valid binding");
    }

    {
        auto ring_result = mcu::GpuMcuControlRing::create(0);
        check(ring_result.ok(), "control ring create");
        if (!ring_result.ok()) return 1;
        mcu::GpuMcuControlRing ring = ring_result.release();
        auto arena_result = mcu::GpuMcuRequestIngressArena::create(0, 4u);
        check(arena_result.ok(), "ingress arena create");
        if (!arena_result.ok()) return 1;
        mcu::GpuMcuRequestIngressArena arena = arena_result.release();
        auto cursor_result = mcu::gpu_mcu_ingress_cursor_create(0);
        check(cursor_result.ok(), "ingress cursor create");
        if (!cursor_result.ok()) return 1;
        mcu::GpuMcuIngressCursor* cursor = cursor_result.release();
        hipStream_t stream = nullptr;
        check(hipStreamCreate(&stream) == hipSuccess, "stream create");
        uint32_t* applied = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&applied), sizeof(uint32_t)) ==
                  hipSuccess,
              "applied alloc");

        const uint32_t entry = arena.reserve();
        check(entry != 0u, "ingress reserve");
        mcu::GpuMcuRequestDescriptor* descriptor = arena.descriptor(entry);
        descriptor->request_id = 9001u;
        descriptor->prompt_tokens_handle = 0x1000u;
        descriptor->prompt_length = 32u;
        descriptor->stable_prefix_boundary = 0u;
        descriptor->sampling_params_handle = 0x2000u;
        descriptor->stop_conditions_handle = 0x3000u;
        descriptor->max_new_tokens = 8u;
        descriptor->max_sequence_length = 256u;
        check(arena.publish(entry).ok(), "ingress publish");

        mcu::GpuMcuControlCommand command{};
        command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::submit);
        command.version = mcu::kGpuMcuControlVersion;
        command.request_id = 9001u;
        command.descriptor_handle = entry;
        check(ring.try_push_input(mcu::encode_control_command(command)),
              "push submit");

        check(mcu::gpu_mcu_process_ingress_once(
                  ring, cursor, arena, slots, bindings.device_bindings(),
                  bindings.max_slots(), 1u, stream)
                  .ok(),
              "process ingress once");
        check(hipStreamSynchronize(stream) == hipSuccess, "process sync");

        mcu::ControlRingPayload payload{};
        mcu::GpuMcuEvent claimed{};
        check(ring.try_pop_event(payload), "claimed event arrives");
        mcu::decode_event(payload, claimed);
        check(claimed.opcode ==
                  static_cast<uint32_t>(mcu::GpuMcuEventOpcode::claimed),
              "the submit is claimed");
        const rt::RequestHandle handle{claimed.slot, claimed.generation};

        check(arena.recycle(entry).ok(),
              "the ingress descriptor is recycled after CLAIMED");

        std::vector<mcu::GpuMcuSlotBinding> bound(kSlots,
                                                  mcu::GpuMcuSlotBinding{});
        check(hipMemcpy(bound.data(), bindings.device_bindings(),
                        kSlots * sizeof(mcu::GpuMcuSlotBinding),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "binding readback");
        check(rt::request_handle_equal(bound[handle.slot].handle, handle),
              "the binding carries the claimed handle");
        check(bound[handle.slot].prompt_tokens_handle == 0x1000u,
              "the binding keeps the prompt resource after recycle");
        check(bound[handle.slot].sampling_params_handle == 0x2000u,
              "the binding keeps the sampling resource after recycle");
        check(bound[handle.slot].stop_conditions_handle == 0x3000u,
              "the binding keeps the stop resource after recycle");
        check(bound[handle.slot].decode_input_valid == 0u,
              "the decode token starts unset");

        const mcu::GpuMcuSlotBinding old_binding = bound[handle.slot];
        check(mcu::gpu_mcu_slot_table_release(slots, handle.slot, handle,
                                              applied, stream)
                  .ok(),
              "release the slot");
        check(hipStreamSynchronize(stream) == hipSuccess, "release sync");
        const uint32_t second = arena.reserve();
        check(second != 0u, "second reserve");
        mcu::GpuMcuRequestDescriptor* d2 = arena.descriptor(second);
        d2->request_id = 9002u;
        d2->prompt_length = 16u;
        d2->max_sequence_length = 256u;
        check(arena.publish(second).ok(), "second publish");
        mcu::GpuMcuControlCommand c2{};
        c2.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::submit);
        c2.version = mcu::kGpuMcuControlVersion;
        c2.request_id = 9002u;
        c2.descriptor_handle = second;
        check(ring.try_push_input(mcu::encode_control_command(c2)),
              "push second submit");
        (void)mcu::gpu_mcu_process_ingress_once(
            ring, cursor, arena, slots, bindings.device_bindings(),
            bindings.max_slots(), 1u, stream);
        check(hipStreamSynchronize(stream) == hipSuccess, "second process sync");
        mcu::ControlRingPayload payload2{};
        mcu::GpuMcuEvent claimed2{};
        check(ring.try_pop_event(payload2), "second claimed event");
        mcu::decode_event(payload2, claimed2);
        check(claimed2.generation == claimed.generation + 1u,
              "the reused slot advances the generation");

        std::vector<mcu::GpuMcuSlotState> slot_state(kSlots,
                                                     mcu::GpuMcuSlotState{});
        check(hipMemcpy(slot_state.data(), slots.device_slots(),
                        kSlots * sizeof(mcu::GpuMcuSlotState),
                        hipMemcpyDeviceToHost) == hipSuccess,
              "slot readback");
        check(!mcu::gpu_mcu_binding_valid_for(slot_state[claimed2.slot],
                                              old_binding),
              "the previous generation binding is rejected for the new request");

        if (applied != nullptr) (void)hipFree(applied);
        if (cursor != nullptr) (void)hipFree(cursor);
        (void)hipStreamDestroy(stream);
        check(arena.shutdown().ok(), "arena shutdown");
        check(ring.shutdown().ok(), "ring shutdown");
    }

    check(bindings.shutdown().ok(), "binding table shutdown");
    check(bindings.is_shutdown(), "shutdown clears the binding table");
    check(slots.shutdown().ok(), "slot table shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_slot_binding: failed=%d\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_slot_binding: R1 PASS\n");
    return 0;
}
