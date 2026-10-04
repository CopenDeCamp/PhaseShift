#include <phaseshift/runtime/gpu_mcu/output_ring.h>
#include <phaseshift/runtime/gpu_mcu/slot_runtime.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace mcu = ::ps::runtime::gpu_mcu;

namespace {

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSlots = 4u;

__global__ void publish_terminals_kernel(
    mcu::OutputRing* ring, uint64_t* position, const mcu::GpuMcuSlotState* slots,
    mcu::GpuMcuSlotRuntimeState* runtime, uint32_t max_slots,
    mcu::GpuMcuTerminalPublishResult* result) {
    *result = mcu::gpu_mcu_publish_slot_terminals(ring, *position, slots,
                                                  runtime, max_slots);
}

__global__ void release_terminals_kernel(
    mcu::GpuMcuSlotState* slots, mcu::GpuMcuSlotRuntimeState* runtime,
    mcu::GpuMcuSlotBinding* bindings, uint32_t max_slots,
    uint32_t binding_max_slots, mcu::GpuMcuSlotReleaseResult* result) {
    mcu::DeviceSequenceResourceManagerView resources{};
    mcu::GpuMcuKvPagePoolView kv_pages{};
    mcu::GpuMcuBlockTableView kv_blocks{};
    *result = mcu::gpu_mcu_release_terminal_slots(
        slots, runtime, bindings, max_slots, binding_max_slots, resources, 0u,
        kv_pages, kv_blocks);
}

__global__ void flush_kernel(mcu::OutputRing* ring, uint64_t* position,
                             mcu::GpuMcuSlotState* slots, uint32_t max_slots,
                             mcu::GpuMcuOutputFlushResult* result) {
    *result = mcu::gpu_mcu_flush_slot_outputs(ring, *position, slots, max_slots);
}

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
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

    auto ring_result = mcu::GpuMcuOutputRing::create(0);
    check(ring_result.ok(), "output ring create");
    if (!ring_result.ok()) return 1;
    mcu::GpuMcuOutputRing ring = ring_result.release();

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "stream create");

    void* slots_host = nullptr;
    void* slots_dev = nullptr;
    void* position_host = nullptr;
    void* position_dev = nullptr;
    void* result_host = nullptr;
    void* result_dev = nullptr;
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotState), &slots_host,
                    &slots_dev),
          "slots alloc");
    check(map_alloc(sizeof(uint64_t), &position_host, &position_dev),
          "position alloc");
    check(map_alloc(sizeof(mcu::GpuMcuOutputFlushResult), &result_host,
                    &result_dev),
          "result alloc");
    void* runtime_host = nullptr;
    void* runtime_dev = nullptr;
    void* terminal_host = nullptr;
    void* terminal_dev = nullptr;
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotRuntimeState), &runtime_host,
                    &runtime_dev),
          "runtime state alloc");
    check(map_alloc(sizeof(mcu::GpuMcuTerminalPublishResult), &terminal_host,
                    &terminal_dev),
          "terminal result alloc");
    void* bindings_host = nullptr;
    void* bindings_dev = nullptr;
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding), &bindings_host,
                    &bindings_dev),
          "bindings alloc");
    void* release_host = nullptr;
    void* release_dev = nullptr;
    check(map_alloc(sizeof(mcu::GpuMcuSlotReleaseResult), &release_host,
                    &release_dev),
          "release result alloc");

    auto* slots = static_cast<mcu::GpuMcuSlotState*>(slots_host);
    auto* position = static_cast<uint64_t*>(position_host);
    auto* result = static_cast<mcu::GpuMcuOutputFlushResult*>(result_host);
    std::memset(slots, 0, kSlots * sizeof(mcu::GpuMcuSlotState));
    *position = 0u;

    slots[0].handle.slot = 0u;
    slots[0].handle.generation = 3u;
    slots[0].request_id = 77u;
    slots[0].generated_tokens = 3u;
    slots[0].pending_count = 3u;
    slots[0].pending_begin = 0u;
    slots[0].pending_tokens[0] = 10;
    slots[0].pending_tokens[1] = 11;
    slots[0].pending_tokens[2] = 12;

    auto flush = [&]() {
        flush_kernel<<<1, 1, 0, stream>>>(
            ring.device_ring(), position,
            static_cast<mcu::GpuMcuSlotState*>(slots_dev), kSlots,
            static_cast<mcu::GpuMcuOutputFlushResult*>(result_dev));
        check(hipStreamSynchronize(stream) == hipSuccess, "flush sync");
    };

    // An empty ring reports empty without blocking.
    mcu::GpuMcuOutputRecord drained{};
    check(!ring.try_pop(drained), "empty try_pop returns false");

    flush();
    check(result->flushed_tokens == 3u, "three tokens flushed");
    check(slots[0].pending_count == 0u, "pending drained");
    check(slots[0].output_blocked == 0u, "slot not blocked");

    // The host consumes the published tokens in order.
    for (int32_t i = 0; i < 3; ++i) {
        mcu::GpuMcuOutputRecord record{};
        check(ring.try_pop(record), "token published");
        check(record.token_id == 10 + i, "token order preserved");
        check(record.token_index == static_cast<uint32_t>(i),
              "token index preserved");
        check(record.request_id == 77u, "request id preserved");
        check(record.request_handle_bits ==
                  ((static_cast<uint64_t>(3u) << 32u) | 0u),
              "request handle bits preserved");
    }
    check(!ring.try_pop(drained), "ring empty after the drain");

    // Fill the ring to capacity; the next flush must block the slot only.
    for (uint32_t round = 0; round < 8u; ++round) {
        slots[0].pending_count = 32u;
        slots[0].pending_begin = 0u;
        slots[0].generated_tokens += 32u;
        for (uint32_t j = 0; j < 32u; ++j) {
            slots[0].pending_tokens[j] = 100 + static_cast<int32_t>(j);
        }
        flush();
        check(result->flushed_tokens == 32u, "round flushes fully");
    }
    check(*position - ring.position() == mcu::kOutputRingCapacity,
          "ring is full");

    slots[0].pending_count = 32u;
    slots[0].pending_begin = 0u;
    slots[0].generated_tokens += 32u;
    flush();
    check(result->flushed_tokens == 0u, "full ring publishes nothing");
    check(result->blocked_slots == 1u, "full ring blocks one slot");
    check(slots[0].output_blocked == 1u, "the slot is output blocked");
    check(slots[1].output_blocked == 0u, "other slots stay runnable");

    // The host drains the pending tokens; the next flush recovers the slot.
    for (uint32_t j = 0; j < 32u; ++j) {
        mcu::GpuMcuOutputRecord record{};
        check(ring.try_pop(record), "recovery token published");
        check(record.token_id == 100 + static_cast<int32_t>(j),
              "recovery order preserved");
    }
    flush();
    check(result->flushed_tokens == 32u, "recovery flushes the pending tokens");
    check(result->blocked_slots == 0u, "recovery blocks nothing");
    check(slots[0].output_blocked == 0u, "the slot is unblocked");
    check(slots[0].pending_count == 0u, "the slot pending is drained");

    // Terminal delivery: exactly one terminal record per request, even when no
    // token is pending.
    {
        mcu::GpuMcuOutputRecord drain{};
        while (ring.try_pop(drain)) {
        }
        auto* runtime = static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_host);
        auto* terminal =
            static_cast<mcu::GpuMcuTerminalPublishResult*>(terminal_host);
        std::memset(runtime_host, 0,
                    kSlots * sizeof(mcu::GpuMcuSlotRuntimeState));
        slots[1].handle.slot = 1u;
        slots[1].handle.generation = 9u;
        slots[1].request_id = 88u;
        slots[1].generated_tokens = 5u;
        slots[1].pending_count = 0u;
        slots[1].terminal_reason =
            static_cast<uint32_t>(mcu::GpuMcuTerminalReason::cancelled);
        publish_terminals_kernel<<<1, 1, 0, stream>>>(
            ring.device_ring(), position,
            static_cast<const mcu::GpuMcuSlotState*>(slots_dev),
            static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_dev), kSlots,
            static_cast<mcu::GpuMcuTerminalPublishResult*>(terminal_dev));
        check(hipStreamSynchronize(stream) == hipSuccess, "terminal sync");
        check(terminal->published == 1u, "one terminal record published");
        {
            mcu::GpuMcuOutputRecord record{};
            check(ring.try_pop(record), "the terminal record is available");
            check((record.flags & mcu::kOutputRecordFlagTerminal) != 0u,
                  "the record is a terminal record");
            check(record.token_id == -1, "a terminal record carries no token");
            check(record.request_id == 88u, "the terminal request id matches");
            check(record.terminal_reason ==
                      static_cast<uint32_t>(
                          mcu::GpuMcuTerminalReason::cancelled),
                  "the terminal reason matches");
        }
        slots[1].phase =
            mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::decode);
        check(runtime[1].terminal_published == 1u,
              "the slot is marked terminal");
        check(runtime[1].release_pending == 1u, "the slot is release pending");
        publish_terminals_kernel<<<1, 1, 0, stream>>>(
            ring.device_ring(), position,
            static_cast<const mcu::GpuMcuSlotState*>(slots_dev),
            static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_dev), kSlots,
            static_cast<mcu::GpuMcuTerminalPublishResult*>(terminal_dev));
        check(hipStreamSynchronize(stream) == hipSuccess, "terminal resync");
        check(terminal->published == 0u, "the terminal is published once");
        {
            mcu::GpuMcuOutputRecord record{};
            check(!ring.try_pop(record), "no second terminal record");
        }
        auto* bindings = static_cast<mcu::GpuMcuSlotBinding*>(bindings_host);
        auto* release = static_cast<mcu::GpuMcuSlotReleaseResult*>(release_host);
        bindings[1].handle = slots[1].handle;
        bindings[1].prompt_tokens_handle = 123u;
        bindings[1].sampling_params_handle = 456u;
        bindings[1].decode_input_token = 42;
        bindings[1].decode_input_valid = 1u;
        release_terminals_kernel<<<1, 1, 0, stream>>>(
            slots, static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_dev),
            static_cast<mcu::GpuMcuSlotBinding*>(bindings_dev), kSlots, kSlots,
            static_cast<mcu::GpuMcuSlotReleaseResult*>(release_dev));
        check(hipStreamSynchronize(stream) == hipSuccess, "release sync");
        check(release->released == 1u, "the terminal slot is released");
        check(slots[1].phase ==
                  mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::idle),
              "the released slot is idle");
        check(slots[1].handle.generation == 10u,
              "the released slot advances the generation");
        check(bindings[1].prompt_tokens_handle == 0u &&
                  bindings[1].sampling_params_handle == 0u &&
                  bindings[1].decode_input_token == -1 &&
                  bindings[1].decode_input_valid == 0u,
              "the binding is detached");
        check(!mcu::gpu_mcu_slot_matches_handle(slots[1],
                                               ::ps::runtime::RequestHandle{1u, 9u}),
              "the old handle is stale after release");
    }

    check(hipStreamDestroy(stream) == hipSuccess, "stream destroy");
    check(hipHostFree(slots_host) == hipSuccess, "slots free");
    check(hipHostFree(position_host) == hipSuccess, "position free");
    check(hipHostFree(result_host) == hipSuccess, "result free");
    check(hipHostFree(runtime_host) == hipSuccess, "runtime state free");
    check(hipHostFree(bindings_host) == hipSuccess, "bindings free");
    check(hipHostFree(release_host) == hipSuccess, "release result free");
    check(hipHostFree(terminal_host) == hipSuccess, "terminal result free");
    check(ring.shutdown().ok(), "output ring shutdown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_output_ring: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_output_ring: PASS\n");
    return 0;
}
