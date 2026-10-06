#include "unit/gpu_mcu/gpu_mcu_fsm_test_util.h"

#include <phaseshift/runtime/gpu_mcu/output_ring.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>

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

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

constexpr uint32_t kSlots = 4u;
constexpr uint32_t kIngress = 8u;
constexpr uint32_t kMaxRequests = 4u;
constexpr uint32_t kMaxRows = 8u;
constexpr uint32_t kMaxOutputs = 8u;
constexpr uint32_t kMaxNewTokens = 8u;
constexpr int32_t kPromptToken = 500;

void* g_prompt_dev = nullptr;
void* g_sampling_dev = nullptr;

__global__ void prepare_verify_slot_kernel(mcu::GpuMcuSlotState* slot,
                                          uint32_t candidates,
                                          uint64_t candidate_ref,
                                          uint32_t* scheduler_dirty) {
    slot->phase = mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::verify);
    slot->verify_candidate_count = candidates;
    slot->verify_candidate_ref = candidate_ref;
    *scheduler_dirty = 1u;
}

void push_cancel(mcu::GpuMcuControlRing& ring, uint64_t request_id,
                 rt::RequestHandle handle) {
    mcu::GpuMcuControlCommand command{};
    command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::cancel);
    command.version = mcu::kGpuMcuControlVersion;
    command.request_id = request_id;
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

void device_to_host(void* dst, const void* src, std::size_t bytes,
                    hipStream_t stream) {
    (void)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
}

uint32_t push_submit(mcu::GpuMcuControlRing& ring,
                     mcu::GpuMcuRequestIngressArena& arena, uint64_t request_id,
                     uint32_t prompt_length, uint32_t max_new_tokens) {
    const uint32_t handle = arena.reserve();
    check(handle != 0u, "ingress reserve");
    if (handle == 0u) return 0u;
    mcu::GpuMcuRequestDescriptor* descriptor = arena.descriptor(handle);
    check(descriptor != nullptr, "ingress descriptor");
    descriptor->request_id = request_id;
    descriptor->prompt_tokens_handle =
        mcu::gpu_mcu_encode_device_address(g_prompt_dev);
    descriptor->prompt_length = prompt_length;
    descriptor->stable_prefix_boundary = 0u;
    descriptor->sampling_params_handle =
        mcu::gpu_mcu_encode_device_address(g_sampling_dev);
    descriptor->stop_conditions_handle = 0u;
    descriptor->max_new_tokens = max_new_tokens;
    descriptor->max_sequence_length = 64u;
    check(arena.publish(handle).ok(), "ingress publish");
    mcu::GpuMcuControlCommand command{};
    command.opcode = static_cast<uint32_t>(mcu::GpuMcuControlOpcode::submit);
    command.version = mcu::kGpuMcuControlVersion;
    command.request_id = request_id;
    command.descriptor_handle = handle;
    check(ring.try_push_input(mcu::encode_control_command(command)),
          "push submit");
    return handle;
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
        std::printf("test_gpu_mcu_autonomous_loop: rig setup failed\n");
        return 1;
    }
    Chain chain = make_chain(1u);
    std::vector<mcu::McuKernelVariantDesc> variants{
        mcu::make_probe_variant(0u, rig.meta, 64u, 1u)};
    if (!rig.upload(chain.nodes, variants)) return 1;
    mcu::GpuMcuFsmConfig fsm_config{};
    fsm_config.plan_id = 0xcb80001u;
    fsm_config.template_source = 0u;
    fsm_config.copy_lanes = 1u;
    fsm_config.barrier = 1u;
    fsm_config.prepared_enabled = 0u;
    if (!rig.prepare(fsm_config)) {
        std::printf("test_gpu_mcu_autonomous_loop: fsm prepare failed\n");
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
    check(map_alloc(kSlots * sizeof(rt::RequestHandle), &runnable_host,
                    &runnable_dev),
          "runnable alloc");
    void* plan_epoch_host = nullptr;
    void* plan_epoch_dev = nullptr;
    check(map_alloc(sizeof(uint64_t), &plan_epoch_host, &plan_epoch_dev),
          "plan epoch alloc");

    void* ctx_host = nullptr;
    void* ctx_dev = nullptr;
    check(map_alloc(sizeof(rt::DeviceBatchContext), &ctx_host, &ctx_dev),
          "context alloc");
    void* requests_host = nullptr;
    void* requests_dev = nullptr;
    check(map_alloc(kMaxRequests * sizeof(rt::DeviceRequestDescriptor),
                    &requests_host, &requests_dev),
          "requests alloc");
    void* rows_host = nullptr;
    void* rows_dev = nullptr;
    check(map_alloc(kMaxRows * sizeof(uint32_t), &rows_host, &rows_dev),
          "row slots alloc");
    void* positions_host = nullptr;
    void* positions_dev = nullptr;
    check(map_alloc(kMaxRows * sizeof(uint32_t), &positions_host,
                    &positions_dev),
          "row positions alloc");
    void* token_ids_host = nullptr;
    void* token_ids_dev = nullptr;
    check(map_alloc(kMaxRows * sizeof(int32_t), &token_ids_host, &token_ids_dev),
          "token ids alloc");
    void* out_rows_host = nullptr;
    void* out_rows_dev = nullptr;
    check(map_alloc(kMaxOutputs * sizeof(uint32_t), &out_rows_host,
                    &out_rows_dev),
          "output rows alloc");
    void* out_sampling_host = nullptr;
    void* out_sampling_dev = nullptr;
    check(map_alloc(kMaxOutputs * sizeof(rt::DeviceSamplingParams),
                    &out_sampling_host, &out_sampling_dev),
          "output sampling alloc");
    void* bindings_host = nullptr;
    void* bindings_dev = nullptr;
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding), &bindings_host,
                    &bindings_dev),
          "bindings alloc");
    void* ready_host = nullptr;
    void* ready_dev = nullptr;
    check(map_alloc(sizeof(uint64_t), &ready_host, &ready_dev),
          "ready epoch alloc");
    void* verify_host = nullptr;
    void* verify_dev = nullptr;
    check(map_alloc(kSlots * sizeof(uint32_t), &verify_host, &verify_dev),
          "verify counts alloc");
    void* runtime_host = nullptr;
    void* runtime_dev = nullptr;
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotRuntimeState), &runtime_host,
                    &runtime_dev),
          "slot runtime alloc");
    void* candidates_host = nullptr;
    void* candidates_dev = nullptr;
    check(map_alloc(8u * sizeof(int32_t), &candidates_host, &candidates_dev),
          "candidates alloc");
    static_cast<uint32_t*>(verify_host)[0] = 8u;
    for (uint32_t i = 0; i < 8u; ++i) {
        static_cast<int32_t*>(candidates_host)[i] = 3000 + static_cast<int32_t>(i);
    }
    void* prompt_host = nullptr;
    void* prompt_dev = nullptr;
    check(map_alloc(8u * sizeof(int32_t), &prompt_host, &prompt_dev),
          "prompt alloc");
    void* sampling_host = nullptr;
    void* sampling_dev = nullptr;
    check(map_alloc(sizeof(rt::DeviceSamplingParams), &sampling_host,
                    &sampling_dev),
          "sampling alloc");

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
                  arena.device_entries(), kIngress, slots.device_slots(),
                  kSlots, static_cast<rt::RequestHandle*>(runnable_dev), kSlots)
              .ok(),
          "configure request runtime");
    check(persistent
              .configure_batch_plan(
                  static_cast<rt::DeviceBatchContext*>(ctx_dev),
                  static_cast<rt::DeviceRequestDescriptor*>(requests_dev),
                  static_cast<uint32_t*>(rows_dev),
                  static_cast<uint32_t*>(positions_dev), kMaxRequests, kMaxRows,
                  1u, static_cast<uint64_t*>(plan_epoch_dev))
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
                                   8u, static_cast<const uint32_t*>(verify_dev),
                                   kSlots)
              .ok(),
          "configure execution");
    check(persistent
              .configure_output(outputs.device_ring())
              .ok(),
          "configure output");
    check(persistent
              .configure_slot_runtime(
                  static_cast<mcu::GpuMcuSlotRuntimeState*>(runtime_dev))
              .ok(),
          "configure slot runtime");
    check(persistent.start().ok(), "persistent mcu start");
    check(persistent.running(), "persistent mcu is running");

    const uint32_t submitted =
        push_submit(ring, arena, 9001u, 1u, kMaxNewTokens);
    check(submitted != 0u, "submit published");

    std::vector<mcu::GpuMcuOutputRecord> published;
    std::vector<mcu::GpuMcuOutputRecord> terminals;
    auto collect = [&]() {
        mcu::GpuMcuOutputRecord record{};
        while (outputs.try_pop(record)) {
            if ((record.flags & mcu::kOutputRecordFlagTerminal) != 0u) {
                terminals.push_back(record);
            } else {
                published.push_back(record);
            }
        }
    };
    const bool drained = wait_until(
        [&] {
            collect();
            return published.size() >= kMaxNewTokens;
        },
        15000u);
    check(drained, "the autonomous loop publishes every token");
    check(persistent.batches_committed() == kMaxNewTokens,
          "one commit per committed token");
    const std::vector<mcu::GpuMcuOutputRecord> autonomous_tokens(
        published.begin(), published.end());

    const int32_t step = static_cast<int32_t>(chain.values[0]);

    // Scenario 1: chunked prefill. A three token prompt with a one row chunk
    // budget runs three prefill loops before the decode loops.

    const uint32_t dispatched_after_a = persistent.batches_dispatched();
    check(push_submit(ring, arena, 9002u, 3u, 3u) != 0u, "submit chunked");
    check(wait_until(
              [&] {
                  return persistent.batches_committed() >=
                         kMaxNewTokens + 3u;
              },
              15000u),
          "the chunked prefill reaches the decode loops");
    const std::size_t tokens_before_chunked = published.size();
    collect();
    const std::vector<mcu::GpuMcuOutputRecord> chunked(
        published.begin() + static_cast<std::ptrdiff_t>(tokens_before_chunked),
        published.end());
    check(chunked.size() == 3u, "chunked prefill publishes three tokens");
    for (std::size_t i = 0; i < chunked.size(); ++i) {
        check(chunked[i].token_id ==
                  kPromptToken + step * static_cast<int32_t>(i + 1),
              "chunked prefill token chain");
    }
    check(persistent.batches_dispatched() - dispatched_after_a == 5u,
          "three prefill chunks plus two decode loops");

    // Scenario 6: cancel at the loop boundary. The current transaction finishes
    // and the slot leaves the following snapshots.
    check(push_submit(ring, arena, 9003u, 1u, 40u) != 0u, "submit cancel");
    rt::RequestHandle cancel_handle{};
    bool claimed = false;
    check(wait_until(
              [&] {
                  mcu::GpuMcuEvent event{};
                  while (pop_event(ring, event)) {
                      if (event.request_id == 9003u &&
                          event.opcode ==
                              static_cast<uint32_t>(
                                  mcu::GpuMcuEventOpcode::claimed)) {
                          cancel_handle.slot = event.slot;
                          cancel_handle.generation = event.generation;
                          claimed = true;
                      }
                  }
                  return claimed;
              },
              5000u),
          "the cancelled request is claimed");
    const uint64_t committed_at_cancel = persistent.batches_committed();
    check(wait_until(
              [&] {
                  return persistent.batches_committed() >= committed_at_cancel + 2u;
              },
              5000u),
          "the cancelled slot commits before the cancel lands");
    push_cancel(ring, 9003u, cancel_handle);
    uint64_t seen = persistent.batches_committed();
    uint32_t stable_ms = 0u;
    while (stable_ms < 400u) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const uint64_t now = persistent.batches_committed();
        if (now == seen) {
            stable_ms += 50u;
        } else {
            seen = now;
            stable_ms = 0u;
        }
    }
    check(persistent.batches_failed() == 0u, "cancel does not fail a batch");
    {
        collect();
        bool terminal_seen = false;
        for (const mcu::GpuMcuOutputRecord& record : terminals) {
            if (record.request_id == 9003u) terminal_seen = true;
        }
        check(terminal_seen, "the cancelled request gets a terminal record");
    }

    // Scenario 3: verify. A slot placed in the verify phase with eight
    // candidates commits the accepted prefix reported by the execution result.
    uint32_t* applied = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&applied), sizeof(uint32_t)) ==
              hipSuccess,
          "applied alloc");
    check(mcu::gpu_mcu_slot_table_claim(slots, 3u, 9004u, 1u, 0u, 64u, 8u,
                                        applied, work)
              .ok(),
          "verify slot claim launches");
    check(hipStreamSynchronize(work) == hipSuccess, "verify slot claim sync");
    {
        std::vector<mcu::GpuMcuSlotState> claimed(kSlots);
        device_to_host(claimed.data(), slots.device_slots(),
                       kSlots * sizeof(mcu::GpuMcuSlotState), work);
        auto* table = static_cast<mcu::GpuMcuSlotBinding*>(bindings_host);
        table[3].handle = claimed[3].handle;
        table[3].prompt_tokens_handle =
            mcu::gpu_mcu_encode_device_address(prompt_dev);
        table[3].sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(sampling_dev);
    }
    const uint64_t committed_before_verify = persistent.batches_committed();
    prepare_verify_slot_kernel<<<1, 1, 0, work>>>(
        slots.device_slots() + 3u, 8u,
        mcu::gpu_mcu_encode_device_address(candidates_dev),
        &persistent.device_state()->scheduler_dirty);
    check(hipStreamSynchronize(work) == hipSuccess, "verify slot prepared");
    check(wait_until(
              [&] {
                  return persistent.batches_committed() >
                         committed_before_verify;
              },
              10000u),
          "the verify slot commits the accepted prefix");
    check(persistent.batches_failed() == 0u, "the verify batch does not fail");

    check(persistent.request_stop().ok(), "stop requested");
    check(persistent.wait_stopped(5000).ok(), "persistent mcu stopped");

    check(persistent.autonomous_loops() > 1u, "more than one autonomous loop");
    check(persistent.batches_dispatched() > 1u, "more than one batch dispatched");
    check(persistent.batches_completed() == persistent.batches_dispatched(),
          "every dispatched batch completed");
    check(persistent.batches_failed() == 0u, "no failed batch");

    check(autonomous_tokens.size() == kMaxNewTokens,
          "token count matches");
    for (std::size_t i = 0; i < autonomous_tokens.size(); ++i) {
        const int32_t expected =
            kPromptToken + step * static_cast<int32_t>(i + 1);
        check(autonomous_tokens[i].token_id == expected,
              "the committed token feeds the next loop");
        check(autonomous_tokens[i].request_id == 9001u,
              "request id preserved");
    }


    {
        collect();
        std::size_t verify_tokens = 0u;
        bool verify_terminal = false;
        for (const mcu::GpuMcuOutputRecord& record : published) {
            if (record.request_id == 9004u) ++verify_tokens;
        }
        for (const mcu::GpuMcuOutputRecord& record : terminals) {
            if (record.request_id == 9004u) verify_terminal = true;
        }
        check(verify_tokens == 8u,
              "the verify slot commits the accepted prefix");
        check(verify_terminal, "the verify slot delivers a terminal");
    }

    check(hipFree(applied) == hipSuccess, "applied free");
    check(hipHostFree(verify_host) == hipSuccess, "verify counts free");
    check(hipHostFree(runtime_host) == hipSuccess, "slot runtime free");
    check(hipHostFree(candidates_host) == hipSuccess, "candidates free");
    check(hipHostFree(prompt_host) == hipSuccess, "prompt free");
    check(hipHostFree(sampling_host) == hipSuccess, "sampling free");
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
    check(persistent.shutdown().ok(), "persistent mcu shutdown");
    check(outputs.shutdown().ok(), "output ring shutdown");
    check(arena.shutdown().ok(), "ingress arena shutdown");
    check(slots.shutdown().ok(), "slot table shutdown");
    check(ring.shutdown().ok(), "control ring shutdown");
    check(rig.close(), "rig teardown");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_autonomous_loop: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_autonomous_loop: PASS\n");
    return 0;
}
