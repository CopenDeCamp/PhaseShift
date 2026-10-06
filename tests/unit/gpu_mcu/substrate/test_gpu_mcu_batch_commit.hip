#include <phaseshift/runtime/gpu_mcu/batch_commit.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

namespace {

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSlots = 32u;
constexpr uint32_t kRows = 64u;

__global__ void commit_once_kernel(mcu::GpuMcuBatchCommitView view,
                                   uint32_t* ok_out) {
    ok_out[0] = mcu::gpu_mcu_commit_active_batch(view) ? 1u : 0u;
}

bool map_alloc(std::size_t bytes, void** host, void** dev) {
    if (hipHostMalloc(host, bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess) {
        return false;
    }
    return hipHostGetDevicePointer(dev, *host, 0) == hipSuccess;
}

void fill(rt::DeviceRequestDescriptor& d, rt::RequestHandle handle,
          uint32_t row_begin, uint32_t row_count, uint32_t prefix,
          uint32_t sequence, rt::ExecutionClass cls, uint32_t output_index,
          uint8_t output_count) {
    d = rt::DeviceRequestDescriptor{};
    d.request_handle = handle;
    d.row_begin = row_begin;
    d.row_count = row_count;
    d.prefix_length = prefix;
    d.sequence_length = sequence;
    d.execution_class = cls;
    d.output_index = output_index;
    d.output_count = output_count;
    d.compute_logits = output_count != 0u ? 1u : 0u;
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

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream) == hipSuccess, "stream create");

    auto slots_result = mcu::GpuMcuSlotTable::create(0, kSlots);
    check(slots_result.ok(), "slot table create");
    if (!slots_result.ok()) return 1;
    mcu::GpuMcuSlotTable slots = slots_result.release();

    uint32_t* applied = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&applied), sizeof(uint32_t)) ==
              hipSuccess,
          "applied alloc");

    void* ctx_host = nullptr;
    void* ctx_dev = nullptr;
    void* req_host = nullptr;
    void* req_dev = nullptr;
    void* rows_host = nullptr;
    void* rows_dev = nullptr;
    void* sampled_host = nullptr;
    void* sampled_dev = nullptr;
    void* verify_host = nullptr;
    void* verify_dev = nullptr;
    void* bind_host = nullptr;
    void* bind_dev = nullptr;
    void* telemetry_host = nullptr;
    void* telemetry_dev = nullptr;
    check(map_alloc(sizeof(rt::DeviceBatchContext), &ctx_host, &ctx_dev),
          "context alloc");
    check(map_alloc(kSlots * sizeof(rt::DeviceRequestDescriptor), &req_host,
                    &req_dev),
          "requests alloc");
    check(map_alloc(kRows * sizeof(uint32_t), &rows_host, &rows_dev),
          "output rows alloc");
    check(map_alloc(kRows * sizeof(int32_t), &sampled_host, &sampled_dev),
          "sampled alloc");
    check(map_alloc(kSlots * sizeof(uint32_t), &verify_host, &verify_dev),
          "verify alloc");
    check(map_alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding), &bind_host,
                    &bind_dev),
          "bindings alloc");
    check(map_alloc(sizeof(mcu::GpuMcuBatchCommitTelemetry), &telemetry_host,
                    &telemetry_dev),
          "telemetry alloc");

    auto* context = static_cast<rt::DeviceBatchContext*>(ctx_host);
    std::memset(context, 0, sizeof(*context));
    context->requests = static_cast<const rt::DeviceRequestDescriptor*>(req_dev);
    context->output_rows = static_cast<uint32_t*>(rows_dev);

    auto* reqs = static_cast<rt::DeviceRequestDescriptor*>(req_host);
    auto* rows = static_cast<uint32_t*>(rows_host);
    auto* sampled = static_cast<int32_t*>(sampled_host);
    auto* verify = static_cast<uint32_t*>(verify_host);
    auto* bindings = static_cast<mcu::GpuMcuSlotBinding*>(bind_host);
    for (uint32_t i = 0; i < kRows; ++i) {
        rows[i] = i;
        sampled[i] = 1000 + static_cast<int32_t>(i);
    }
    for (uint32_t i = 0; i < kSlots; ++i) {
        bindings[i] = mcu::GpuMcuSlotBinding{};
        bindings[i].handle.slot = i;
        bindings[i].handle.generation = rt::kInvalidRequestGeneration;
    }

    std::vector<mcu::GpuMcuSlotState> host_slots(kSlots);

    auto refresh = [&]() {
        (void)hipMemcpyAsync(host_slots.data(), slots.device_slots(),
                             kSlots * sizeof(mcu::GpuMcuSlotState),
                             hipMemcpyDeviceToHost, stream);
        (void)hipStreamSynchronize(stream);
    };

    auto claim_with = [&](uint32_t index, uint32_t prompt_length,
                          uint32_t max_sequence_length,
                          uint32_t max_new_tokens) {
        check(mcu::gpu_mcu_slot_table_claim(slots, index, 100u + index,
                                            prompt_length, 0u,
                                            max_sequence_length,
                                            max_new_tokens, applied, stream)
                  .ok(),
              "slot claim launches");
        check(hipStreamSynchronize(stream) == hipSuccess, "slot claim sync");
        refresh();
        bindings[index].handle = host_slots[index].handle;
    };

    auto claim = [&](uint32_t index, uint32_t prompt_length,
                     uint32_t max_new_tokens) {
        claim_with(index, prompt_length, 256u, max_new_tokens);
    };

    auto run = [&](uint32_t request_limits_valid = 1u) {
        uint32_t* ok = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&ok), sizeof(uint32_t)) ==
                  hipSuccess,
              "ok alloc");
        mcu::GpuMcuBatchCommitView view{};
        view.context = static_cast<rt::DeviceBatchContext*>(ctx_dev);
        view.slots = slots.device_slots();
        view.max_slots = kSlots;
        view.bindings = static_cast<mcu::GpuMcuSlotBinding*>(bind_dev);
        view.binding_max_slots = kSlots;
        view.request_limits_valid = request_limits_valid;
        view.sampled_tokens = static_cast<const int32_t*>(sampled_dev);
        view.sampled_capacity = kRows;
        view.verify_committed_counts = static_cast<const uint32_t*>(verify_dev);
        view.verify_capacity = kSlots;
        view.telemetry =
            static_cast<mcu::GpuMcuBatchCommitTelemetry*>(telemetry_dev);
        commit_once_kernel<<<1, 1, 0, stream>>>(view, ok);
        check(hipStreamSynchronize(stream) == hipSuccess, "commit sync");
        uint32_t host_ok = 0u;
        (void)hipMemcpyAsync(&host_ok, ok, sizeof(uint32_t),
                             hipMemcpyDeviceToHost, stream);
        (void)hipStreamSynchronize(stream);
        (void)hipFree(ok);
        refresh();
        return host_ok != 0u;
    };

    auto* telemetry =
        static_cast<mcu::GpuMcuBatchCommitTelemetry*>(telemetry_host);

    // A. PREFILL chunk, then the final chunk that produces one token.
    claim(0u, 16u, 8u);
    fill(reqs[0], host_slots[0].handle, 0u, 4u, 0u, 4u,
         rt::ExecutionClass::PREFILL, 0xFFFFFFFFu, 0u);
    context->num_requests = 1u;
    context->actual_rows = 4u;
    check(run(), "prefill chunk commits");
    check(host_slots[0].prefill_position == 4u, "prefill chunk advances");
    check(host_slots[0].sequence_length == 4u, "prefill chunk sequence");
    check(host_slots[0].phase ==
              mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::prefill),
          "prefill chunk keeps the phase");
    check(host_slots[0].generated_tokens == 0u, "prefill chunk emits nothing");

    rows[0] = 15u;
    fill(reqs[0], host_slots[0].handle, 4u, 12u, 4u, 16u,
         rt::ExecutionClass::PREFILL, 0u, 1u);
    context->actual_rows = 12u;
    check(run(), "final prefill commits");
    check(host_slots[0].prefill_position == 16u, "final prefill advances");
    check(host_slots[0].phase ==
              mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::decode),
          "final prefill moves to decode");
    check(host_slots[0].generated_tokens == 1u, "final prefill emits one");
    check(host_slots[0].pending_count == 1u, "final prefill stages one");
    check(host_slots[0].pending_tokens[0] == 1015,
          "final prefill stages the sampled token");
    check(bindings[0].decode_input_valid == 1u &&
              bindings[0].decode_input_token == 1015,
          "final prefill feeds the next decode input");

    // B. DECODE token.
    rows[0] = 17u;
    fill(reqs[0], host_slots[0].handle, 0u, 1u, 16u, 17u,
         rt::ExecutionClass::DECODE, 0u, 1u);
    context->actual_rows = 1u;
    check(run(), "decode commits");
    check(host_slots[0].sequence_length == 17u, "decode advances sequence");
    check(host_slots[0].generated_tokens == 2u, "decode counts the token");
    check(host_slots[0].pending_count == 2u, "decode stages the token");
    check(host_slots[0].pending_tokens[1] == 1017,
          "decode stages the sampled token");
    check(bindings[0].decode_input_token == 1017,
          "decode feeds the next decode input");

    // C. VERIFY accepted counts 0, 8 and 32, then the overflow guard.
    const uint32_t accepted[kSlots] = {0u, 8u, 32u, 33u};
    for (uint32_t c = 0; c < 4u; ++c) {
        const uint32_t index = 1u + c;
        claim(index, 8u, 64u);
        const rt::RequestHandle handle = host_slots[index].handle;
        fill(reqs[0], handle, 0u, 32u, 8u, 40u, rt::ExecutionClass::SPEC_VERIFY,
             0u, 32u);
        verify[0] = accepted[c];
        context->num_requests = 1u;
        context->actual_rows = 32u;
        const bool ok = run();
        if (c == 3u) {
            check(!ok, "verify overflow fails closed");
            check(host_slots[index].terminal_reason ==
                      static_cast<uint32_t>(mcu::GpuMcuTerminalReason::error),
                  "verify overflow is terminal");
            check(host_slots[index].sequence_length == 0u,
                  "verify overflow keeps the progress");
            continue;
        }
        check(ok, "verify commits");
        check(host_slots[index].verify_committed_count == accepted[c],
              "verify records the accepted count");
        check(host_slots[index].sequence_length == 8u + accepted[c],
              "verify advances by the accepted count");
        check(host_slots[index].generated_tokens == accepted[c],
              "verify counts the accepted tokens");
        check(host_slots[index].pending_count == accepted[c],
              "verify stages the accepted tokens");
    }

    // F. Mixed batch: two decodes, one verify and one prefill final chunk
    // commit in a single transaction.
    {
        const uint32_t b = 4u;
        claim(b + 0u, 16u, 8u);
        claim(b + 1u, 16u, 8u);
        claim(b + 2u, 8u, 64u);
        claim(b + 3u, 16u, 8u);
        for (uint32_t s = 0u; s < 4u; ++s) {
            bindings[b + s].handle = host_slots[b + s].handle;
            bindings[b + s].decode_input_valid = 0u;
        }
        for (uint32_t r = 0u; r < 12u; ++r) {
            rows[r] = r;
            sampled[r] = 2000 + static_cast<int32_t>(r);
        }
        fill(reqs[0], host_slots[b + 0u].handle, 0u, 1u, 16u, 17u,
             rt::ExecutionClass::DECODE, 0u, 1u);
        fill(reqs[1], host_slots[b + 1u].handle, 1u, 1u, 16u, 17u,
             rt::ExecutionClass::DECODE, 1u, 1u);
        fill(reqs[2], host_slots[b + 2u].handle, 2u, 8u, 8u, 16u,
             rt::ExecutionClass::SPEC_VERIFY, 2u, 8u);
        fill(reqs[3], host_slots[b + 3u].handle, 10u, 16u, 0u, 16u,
             rt::ExecutionClass::PREFILL, 10u, 1u);
        verify[2] = 8u;
        context->num_requests = 4u;
        context->actual_rows = 26u;
        check(run(), "mixed batch commits");
        check(telemetry->committed_tokens == 11u,
              "mixed batch commits every token");
        check(telemetry->committed_requests == 4u,
              "mixed batch commits every request");
        check(host_slots[b + 0u].sequence_length == 17u,
              "mixed decode0 advances");
        check(host_slots[b + 1u].sequence_length == 17u,
              "mixed decode1 advances");
        check(host_slots[b + 2u].sequence_length == 16u,
              "mixed verify advances");
        check(host_slots[b + 2u].verify_committed_count == 8u,
              "mixed verify records the accepted count");
        check(host_slots[b + 2u].phase ==
                  mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::decode),
              "mixed verify moves to decode");
        check(host_slots[b + 3u].prefill_position == 16u,
              "mixed prefill advances");
        check(host_slots[b + 3u].phase ==
                  mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::decode),
              "mixed prefill moves to decode");
        check(host_slots[b + 0u].pending_tokens[0] == 2000,
              "mixed decode0 stages its token");
        check(host_slots[b + 1u].pending_tokens[0] == 2001,
              "mixed decode1 stages its token");
        check(host_slots[b + 2u].pending_tokens[0] == 2002 &&
                  host_slots[b + 2u].pending_tokens[7] == 2009,
              "mixed verify stages the accepted prefix");
        check(host_slots[b + 3u].pending_tokens[0] == 2010,
              "mixed prefill stages its token");
        check(bindings[b + 0u].decode_input_token == 2000,
              "mixed decode0 feeds the next input");
        check(bindings[b + 1u].decode_input_token == 2001,
              "mixed decode1 feeds the next input");
        check(bindings[b + 3u].decode_input_token == 2010,
              "mixed prefill feeds the next input");
    }

    // E. Stale generation: an old descriptor must not commit to the slot.
    claim(6u, 8u, 8u);
    const std::vector<mcu::GpuMcuSlotState> before = host_slots;
    rt::RequestHandle stale = host_slots[6].handle;
    stale.generation = stale.generation + 1u;
    fill(reqs[0], stale, 0u, 4u, 0u, 4u, rt::ExecutionClass::PREFILL,
         0xFFFFFFFFu, 0u);
    context->num_requests = 1u;
    context->actual_rows = 4u;
    check(!run(), "stale commit fails closed");
    check(telemetry->stale_handles >= 1u, "stale handle is counted");
    check(std::memcmp(&host_slots[6], &before[6],
                      sizeof(mcu::GpuMcuSlotState)) == 0,
          "stale commit does not touch the slot");

    {
        void* ids_host = nullptr;
        void* ids_dev = nullptr;
        void* cond_host = nullptr;
        void* cond_dev = nullptr;
        check(map_alloc(sizeof(int32_t), &ids_host, &ids_dev),
              "stop ids alloc");
        check(map_alloc(sizeof(mcu::GpuMcuStopConditions), &cond_host,
                        &cond_dev),
              "stop conditions alloc");
        const int32_t kEos = 4242;
        static_cast<int32_t*>(ids_host)[0] = kEos;
        auto* stop_cond = static_cast<mcu::GpuMcuStopConditions*>(cond_host);
        stop_cond->token_ids = static_cast<const int32_t*>(ids_dev);
        stop_cond->token_count = 1u;
        stop_cond->flags = 0u;

        rows[0] = 0u; rows[1] = 1u; rows[2] = 2u; rows[3] = 3u; rows[4] = 4u;
        sampled[0] = 9001;
        sampled[1] = kEos;
        sampled[2] = 9003;
        sampled[3] = 9004;
        sampled[4] = 9005;

        claim(8u, 8u, 64u);
        bindings[8].stop_conditions_handle =
            reinterpret_cast<uint64_t>(cond_dev);
        fill(reqs[0], host_slots[8].handle, 0u, 5u, 8u, 13u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 5u);
        verify[0] = 5u;
        context->num_requests = 1u;
        context->actual_rows = 5u;
        check(run(), "verify with stop boundary commits");
        check(host_slots[8].verify_committed_count == 2u,
              "effective_commit_count stops at the stop token");
        check(host_slots[8].sequence_length == 10u,
              "sequence advances only by the effective count");
        check(host_slots[8].generated_tokens == 2u,
              "only the effective prefix is staged");
        check(host_slots[8].pending_tokens[0] == 9001 &&
                  host_slots[8].pending_tokens[1] == kEos,
              "staged tokens are the effective prefix only");
        check(host_slots[8].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos),
              "eos latches at the stop boundary");

        sampled[1] = 9002;
        sampled[2] = kEos;

        claim(9u, 8u, 3u);
        bindings[9].stop_conditions_handle =
            reinterpret_cast<uint64_t>(cond_dev);
        fill(reqs[0], host_slots[9].handle, 0u, 5u, 8u, 13u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 5u);
        verify[0] = 5u;
        check(run(), "verify with a tie commits");
        check(host_slots[9].verify_committed_count == 3u,
              "the tie keeps the stop boundary");
        check(host_slots[9].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos),
              "the stop wins the tie");

        claim(10u, 8u, 2u);
        bindings[10].stop_conditions_handle =
            reinterpret_cast<uint64_t>(cond_dev);
        fill(reqs[0], host_slots[10].handle, 0u, 5u, 8u, 13u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 5u);
        verify[0] = 5u;
        check(run(), "verify bounded by budget commits");
        check(host_slots[10].verify_committed_count == 2u,
              "the budget bounds the commit");
        check(host_slots[10].sequence_length == 10u,
              "the budget bounds the sequence");
        check(host_slots[10].terminal_reason ==
                  static_cast<uint32_t>(
                      mcu::GpuMcuTerminalReason::max_new_tokens),
              "max_new_tokens latches when the budget wins");

        fill(reqs[0], host_slots[8].handle, 0u, 1u, 10u, 11u,
             rt::ExecutionClass::DECODE, 0u, 1u);
        check(run(), "post-terminal commit runs");
        check(host_slots[8].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos),
              "the latched eos reason is not overwritten");

        claim(11u, 16u, 0u);
        fill(reqs[0], host_slots[11].handle, 0u, 4u, 0u, 4u,
             rt::ExecutionClass::PREFILL, 0xFFFFFFFFu, 0u);
        context->num_requests = 1u;
        context->actual_rows = 4u;
        check(run(), "max_new zero partial prefill commits");
        check(host_slots[11].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none),
              "partial prefill does not terminate max_new_tokens zero");
        check(host_slots[11].phase ==
                  mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::prefill),
              "partial prefill keeps the prefill phase");
        fill(reqs[0], host_slots[11].handle, 4u, 12u, 4u, 16u,
             rt::ExecutionClass::PREFILL, 0u, 1u);
        context->actual_rows = 12u;
        check(run(), "max_new zero final prefill commits");
        check(host_slots[11].terminal_reason ==
                  static_cast<uint32_t>(
                      mcu::GpuMcuTerminalReason::max_new_tokens),
              "final prefill terminates max_new_tokens zero");
        check(host_slots[11].phase ==
                  mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::prefill),
              "max_new_tokens zero does not enter decode");
        check(host_slots[11].generated_tokens == 0u &&
                  host_slots[11].pending_count == 0u,
              "max_new_tokens zero stages no token");

        rows[0] = 0u;
        rows[1] = 1u;
        rows[2] = 2u;
        rows[3] = 3u;
        context->num_requests = 1u;
        context->actual_rows = 4u;

        claim_with(12u, 8u, 10u, 64u);
        sampled[0] = 9100;
        sampled[1] = 9101;
        sampled[2] = 9102;
        sampled[3] = 9103;
        fill(reqs[0], host_slots[12].handle, 0u, 4u, 8u, 12u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 4u);
        verify[0] = 4u;
        check(run(), "capacity bounded verify commits");
        check(host_slots[12].verify_committed_count == 2u,
              "remaining sequence capacity bounds the commit");
        check(host_slots[12].sequence_length == 10u,
              "the sequence stops at the sequence limit");
        check(host_slots[12].generated_tokens == 2u &&
                  host_slots[12].pending_count == 2u,
              "only the capacity bounded prefix is staged");
        check(host_slots[12].terminal_reason ==
                  static_cast<uint32_t>(
                      mcu::GpuMcuTerminalReason::max_seq_len),
              "max_seq_len latches when the capacity wins");

        claim_with(13u, 8u, 11u, 64u);
        bindings[13].stop_conditions_handle =
            reinterpret_cast<uint64_t>(cond_dev);
        sampled[2] = kEos;
        fill(reqs[0], host_slots[13].handle, 0u, 4u, 8u, 12u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 4u);
        verify[0] = 4u;
        check(run(), "verify with a stop at the capacity boundary commits");
        check(host_slots[13].verify_committed_count == 3u,
              "the stop boundary equals the remaining capacity");
        check(host_slots[13].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos),
              "eos wins the tie against the sequence capacity");

        claim_with(14u, 8u, 10u, 64u);
        bindings[14].stop_conditions_handle =
            reinterpret_cast<uint64_t>(cond_dev);
        fill(reqs[0], host_slots[14].handle, 0u, 4u, 8u, 12u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 4u);
        verify[0] = 4u;
        check(run(), "verify with a stop beyond the capacity commits");
        check(host_slots[14].verify_committed_count == 2u,
              "the capacity bounds the commit before the stop");
        check(host_slots[14].terminal_reason ==
                  static_cast<uint32_t>(
                      mcu::GpuMcuTerminalReason::max_seq_len),
              "max_seq_len wins when the stop lies out of range");

        claim_with(15u, 16u, 256u, 0u);
        sampled[0] = 9200;
        fill(reqs[0], host_slots[15].handle, 0u, 4u, 0u, 4u,
             rt::ExecutionClass::PREFILL, 0xFFFFFFFFu, 0u);
        context->actual_rows = 4u;
        check(run(0u), "host fed partial prefill commits");
        check(host_slots[15].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none),
              "host fed partial prefill does not terminate");
        fill(reqs[0], host_slots[15].handle, 4u, 12u, 4u, 16u,
             rt::ExecutionClass::PREFILL, 0u, 1u);
        context->actual_rows = 12u;
        check(run(0u), "host fed final prefill commits");
        check(host_slots[15].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::none),
              "host fed final prefill does not treat a zero limit as a request");
        check(host_slots[15].phase ==
                  mcu::gpu_mcu_slot_phase_value(mcu::GpuMcuSlotPhase::decode),
              "host fed final prefill moves to decode");
        check(host_slots[15].generated_tokens == 1u &&
                  host_slots[15].pending_count == 1u,
              "host fed final prefill stages one token");
        check(host_slots[15].pending_tokens[0] == 9200,
              "host fed final prefill stages the sampled token");

        claim_with(16u, 8u, 256u, 0u);
        bindings[16].stop_conditions_handle =
            reinterpret_cast<uint64_t>(cond_dev);
        sampled[0] = 9300;
        sampled[1] = kEos;
        sampled[2] = 9302;
        sampled[3] = 9303;
        fill(reqs[0], host_slots[16].handle, 0u, 4u, 8u, 12u,
             rt::ExecutionClass::SPEC_VERIFY, 0u, 4u);
        verify[0] = 4u;
        check(run(0u), "host fed verify with a stop boundary commits");
        check(host_slots[16].verify_committed_count == 2u,
              "the stop boundary applies without request limits");
        check(host_slots[16].sequence_length == 10u,
              "the host fed sequence follows the stop boundary");
        check(host_slots[16].terminal_reason ==
                  static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos),
              "host fed verify latches eos");

        check(hipHostFree(ids_host) == hipSuccess, "stop ids free");
        check(hipHostFree(cond_host) == hipSuccess, "stop conditions free");
    }

    check(slots.shutdown().ok(), "slot table shutdown");
    check(hipFree(applied) == hipSuccess, "applied free");
    check(hipHostFree(ctx_host) == hipSuccess, "context free");
    check(hipHostFree(req_host) == hipSuccess, "requests free");
    check(hipHostFree(rows_host) == hipSuccess, "rows free");
    check(hipHostFree(sampled_host) == hipSuccess, "sampled free");
    check(hipHostFree(verify_host) == hipSuccess, "verify free");
    check(hipHostFree(bind_host) == hipSuccess, "bindings free");
    check(hipHostFree(telemetry_host) == hipSuccess, "telemetry free");
    check(hipStreamDestroy(stream) == hipSuccess, "stream destroy");

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_batch_commit: %d failures\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_batch_commit: PASS\n");
    return 0;
}
