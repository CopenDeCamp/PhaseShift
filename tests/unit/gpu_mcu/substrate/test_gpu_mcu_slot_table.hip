#include <phaseshift/runtime/gpu_mcu/slot_table.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

static_assert(sizeof(mcu::GpuMcuSlotState) == 256u);
static_assert(alignof(mcu::GpuMcuSlotState) == 64u);
static_assert(std::is_standard_layout_v<mcu::GpuMcuSlotState>);
static_assert(std::is_trivially_copyable_v<mcu::GpuMcuSlotState>);
static_assert(static_cast<uint32_t>(mcu::GpuMcuSlotPhase::idle) == 0u);
static_assert(static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill) == 1u);
static_assert(static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode) == 2u);
static_assert(static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify) == 3u);
static_assert(static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none) == 0u);
static_assert(static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos) == 1u);
static_assert(static_cast<uint32_t>(mcu::GpuMcuTerminalReason::error) == 6u);

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

__global__ void set_pending_kernel(mcu::GpuMcuSlotState* slot,
                                   rt::RequestHandle handle,
                                   uint32_t count,
                                   int32_t base,
                                   uint32_t* applied) {
    if (count > mcu::kGpuMcuSlotPendingTokens ||
        !mcu::gpu_mcu_slot_matches_handle(*slot, handle)) {
        *applied = 0u;
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        slot->pending_tokens[i] = base + static_cast<int32_t>(i);
    }
    slot->pending_count = count;
    slot->pending_begin = 0u;
    *applied = 1u;
}

__global__ void drain_pending_kernel(mcu::GpuMcuSlotState* slot,
                                     rt::RequestHandle handle,
                                     uint32_t* applied) {
    if (!mcu::gpu_mcu_slot_matches_handle(*slot, handle)) {
        *applied = 0u;
        return;
    }
    slot->pending_count = 0u;
    slot->pending_begin = 0u;
    *applied = 1u;
}

struct Harness {
    mcu::GpuMcuSlotTable table;
    uint32_t* applied = nullptr;
    std::vector<mcu::GpuMcuSlotState> host;
    hipStream_t stream = nullptr;

    bool open(int device, uint32_t slots) {
        auto created = mcu::GpuMcuSlotTable::create(device, slots);
        check(created.ok(), "slot table create");
        if (!created.ok()) return false;
        table = created.release();
        check(hipStreamCreate(&stream) == hipSuccess, "stream create");
        check(hipMalloc(reinterpret_cast<void**>(&applied), sizeof(uint32_t)) ==
                  hipSuccess,
              "applied alloc");
        host.assign(slots, mcu::GpuMcuSlotState{});
        return true;
    }

    bool refresh() {
        const std::size_t bytes = host.size() * sizeof(mcu::GpuMcuSlotState);
        const hipError_t err = hipMemcpy(host.data(), table.device_slots(),
                                         bytes, hipMemcpyDeviceToHost);
        check(err == hipSuccess, "slot table readback");
        return err == hipSuccess;
    }

    bool applied_flag(uint32_t expected, const char* what) {
        uint32_t value = 0u;
        const hipError_t err =
            hipMemcpy(&value, applied, sizeof(value), hipMemcpyDeviceToHost);
        check(err == hipSuccess, "applied readback");
        if (err != hipSuccess) return false;
        if (value != expected) {
            std::printf("FAIL: %s applied=%u expected=%u\n", what, value,
                        expected);
            ++g_failed;
            return false;
        }
        return true;
    }

    void close() {
        if (applied != nullptr) {
            (void)hipFree(applied);
            applied = nullptr;
        }
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
            stream = nullptr;
        }
        const Status status = table.shutdown();
        check(status.ok(), "slot table shutdown");
        check(table.is_shutdown(), "shutdown clears the table");
    }
};

rt::RequestHandle claim(Harness& h,
                        uint32_t index,
                        uint64_t request_id,
                        uint32_t prompt_length,
                        uint32_t stable_prefix_boundary,
                        uint32_t max_sequence_length,
                        uint32_t max_new_tokens,
                        bool expect_applied) {
    const Status status = mcu::gpu_mcu_slot_table_claim(
        h.table, index, request_id, prompt_length, stable_prefix_boundary,
        max_sequence_length, max_new_tokens, h.applied, h.stream);
    check(status.ok(), "claim launches");
    if (hipStreamSynchronize(h.stream) != hipSuccess) {
        check(false, "claim sync");
    }
    h.applied_flag(expect_applied ? 1u : 0u, "claim");
    h.refresh();
    return h.host.size() > index ? h.host[index].handle : rt::RequestHandle{};
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Harness h{};
    if (!h.open(0, kSlots)) {
        std::printf("test_gpu_mcu_slot_table: setup failed\n");
        return 1;
    }

    {
        check(h.table.max_slots() == kSlots, "max_slots");
        const std::size_t bytes = kSlots * sizeof(mcu::GpuMcuSlotState);
        check(bytes == kSlots * 256u, "table bytes are 256 per slot");
    }

    check(h.refresh(), "initial readback");
    for (uint32_t i = 0; i < kSlots; ++i) {
        const mcu::GpuMcuSlotState& slot = h.host[i];
        check(slot.phase == static_cast<uint32_t>(mcu::GpuMcuSlotPhase::idle),
              "initial phase is idle");
        check(slot.terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none),
              "initial terminal reason is none");
        check(slot.handle.slot == i, "initial handle slot matches the index");
        check(slot.handle.generation == rt::kInvalidRequestGeneration,
              "initial generation is zero");
        check(slot.cancel_requested == 0u, "initial cancel flag is clear");
        check(slot.output_blocked == 0u, "initial output blocked is clear");
        check(slot.pending_count == 0u, "initial pending count is zero");
        check(!mcu::gpu_mcu_slot_has_active_request(slot),
              "initial slot is not active");
    }

    rt::RequestHandle handle_a = claim(h, 3u, 9001u, 64u, 16u, 4096u, 128u, true);
    {
        const mcu::GpuMcuSlotState& slot = h.host[3];
        check(slot.phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill),
              "claim moves the slot to prefill");
        check(slot.handle.slot == 3u, "claim keeps the slot index");
        check(slot.handle.generation == 1u, "first claim uses generation 1");
        check(slot.request_id == 9001u, "claim carries the request id");
        check(slot.prompt_length == 64u, "claim carries the prompt length");
        check(slot.stable_prefix_boundary == 16u,
              "claim carries the stable prefix boundary");
        check(slot.max_sequence_length == 4096u, "claim carries max seq len");
        check(slot.max_new_tokens == 128u, "claim carries max new tokens");
        check(slot.sequence_length == 0u && slot.committed_position == 0u,
              "claim resets the progress counters");
        check(slot.prefill_position == 0u, "claim resets the prefill position");
        check(mcu::gpu_mcu_slot_validate(slot), "claimed slot validates");
        check(rt::request_handle_equal(handle_a, slot.handle),
              "returned handle matches the device state");
    }

    {
        const Status status = mcu::gpu_mcu_slot_table_claim(
            h.table, 3u, 1u, 1u, 0u, 8u, 8u, h.applied, h.stream);
        check(status.ok(), "double claim launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(0u, "double claim is rejected");
    }

    {
        const Status status = mcu::gpu_mcu_slot_table_mark_terminal(
            h.table, 3u, rt::RequestHandle{3u, 1u},
            static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos), h.applied,
            h.stream);
        check(status.ok(), "terminal launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "terminal eos applies");
        h.refresh();
        check(h.host[3].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos),
              "terminal reason is eos");
        check(!mcu::gpu_mcu_slot_is_runnable(h.host[3]),
              "terminal slot is not runnable");
        check(mcu::gpu_mcu_slot_validate(h.host[3]), "terminal slot validates");
    }

    {
        const Status status = mcu::gpu_mcu_slot_table_release(
            h.table, 3u, rt::RequestHandle{3u, 1u}, h.applied, h.stream);
        check(status.ok(), "release launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "release applies");
        h.refresh();
        check(h.host[3].phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::idle),
              "release returns the slot to idle");
        check(h.host[3].handle.generation == 2u,
              "release advances the generation");
        check(!mcu::gpu_mcu_slot_matches_handle(h.host[3],
                                               rt::RequestHandle{3u, 1u}),
              "the released slot stops matching the old handle");
    }

    {
        const Status status = mcu::gpu_mcu_slot_table_request_cancel(
            h.table, 3u, rt::RequestHandle{3u, 1u}, h.applied, h.stream);
        check(status.ok(), "stale cancel launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(0u, "cancel with a stale handle is rejected");
    }

    rt::RequestHandle handle_b = claim(h, 3u, 9002u, 32u, 0u, 2048u, 64u, true);
    {
        check(handle_b.generation == 2u,
              "the second claim increments the generation");
        check(h.host[3].request_id == 9002u, "the second claim replaces state");
        check(h.host[3].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none),
              "the second claim clears the terminal reason");
    }

    {
        const Status status = mcu::gpu_mcu_slot_table_request_cancel(
            h.table, 3u, handle_a, h.applied, h.stream);
        check(status.ok(), "stale cancel of A launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(0u, "handle A is stale after reclaim");
        h.refresh();
        check(h.host[3].cancel_requested == 0u,
              "the stale cancel did not touch request B");
        check(h.host[3].request_id == 9002u,
              "request B is unchanged by the stale cancel");
    }

    {
        const Status status = mcu::gpu_mcu_slot_table_set_phase(
            h.table, 3u, handle_b,
            static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify), h.applied,
            h.stream);
        check(status.ok(), "set phase verify launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "set phase verify applies");
    }
    {
        const Status status = mcu::gpu_mcu_slot_table_request_cancel(
            h.table, 3u, handle_b, h.applied, h.stream);
        check(status.ok(), "cancel on verify launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "cancel applies on a verify slot");
        h.refresh();
        check(h.host[3].cancel_requested == 1u, "cancel flag is set");
        check(h.host[3].phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify),
              "cancel does not change the phase");
    }
    {
        const Status status = mcu::gpu_mcu_slot_table_mark_output_blocked(
            h.table, 3u, handle_b, 1u, h.applied, h.stream);
        check(status.ok(), "output blocked launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "output blocked applies");
        h.refresh();
        check(h.host[3].output_blocked == 1u, "output blocked flag is set");
        check(h.host[3].phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify),
              "output blocked does not change the phase");
        check(h.host[3].cancel_requested == 1u,
              "cancel stays independent from output blocked");
    }

    {
        set_pending_kernel<<<1, 1, 0u, h.stream>>>(
            h.table.device_slots() + 3u, handle_b, 32u, 1000, h.applied);
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "32 pending tokens apply");
        h.refresh();
        check(h.host[3].pending_count == 32u, "pending count is 32");
        check(h.host[3].pending_tokens[0] == 1000, "pending token 0");
        check(h.host[3].pending_tokens[31] == 1031, "pending token 31");
        check(mcu::gpu_mcu_slot_validate(h.host[3]),
              "a full pending window validates");
    }
    {
        set_pending_kernel<<<1, 1, 0u, h.stream>>>(
            h.table.device_slots() + 3u, handle_b, 33u, 0, h.applied);
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(0u, "33 pending tokens are rejected");
    }
    {
        const Status status = mcu::gpu_mcu_slot_table_release(
            h.table, 3u, handle_b, h.applied, h.stream);
        check(status.ok(), "release with pending launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(0u, "release is refused while output is pending");
        h.refresh();
        check(h.host[3].phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify),
              "the slot stays active while pending is non empty");
    }
    {
        drain_pending_kernel<<<1, 1, 0u, h.stream>>>(
            h.table.device_slots() + 3u, handle_b, h.applied);
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "drain applies");
        const Status status = mcu::gpu_mcu_slot_table_release(
            h.table, 3u, handle_b, h.applied, h.stream);
        check(status.ok(), "release after drain launches");
        (void)hipStreamSynchronize(h.stream);
        h.applied_flag(1u, "release after drain applies");
        h.refresh();
        check(h.host[3].phase ==
                  static_cast<uint32_t>(mcu::GpuMcuSlotPhase::idle),
              "the slot is idle after the drain and release");
    }

    rt::RequestHandle handle_c = claim(h, 5u, 9003u, 8u, 8u, 512u, 16u, true);
    {
        check(handle_c.generation == 1u,
              "an untouched slot claims with generation 1");
        check(mcu::gpu_mcu_slot_validate(h.host[5]), "slot 5 validates");
    }

    for (uint32_t i = 0; i < kSlots; ++i) {
        check(mcu::gpu_mcu_slot_validate(h.host[i]), "every slot validates");
    }

    h.close();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_slot_table: failed=%d\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_slot_table: PASS slots=%u\n", kSlots);
    return 0;
}
