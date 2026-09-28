#include <phaseshift/runtime/gpu_mcu/batch_planner.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <type_traits>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

static_assert(sizeof(rt::DeviceRequestDescriptor) == 64u);
static_assert(std::is_standard_layout_v<mcu::GpuMcuBatchPlanLimits>);

namespace {

using Status = ::ps::Status;

int g_failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++g_failed;
    }
}

constexpr uint32_t kSlots = 12u;
constexpr uint32_t kMaxRequests = 4u;
constexpr uint32_t kMaxRows = 32u;

struct Mapped {
    void* host = nullptr;
    void* device = nullptr;

    template <typename T>
    T* host_as() const {
        return static_cast<T*>(host);
    }
    template <typename T>
    T* device_as() const {
        return static_cast<T*>(device);
    }
    bool alloc(std::size_t bytes) {
        if (hipHostMalloc(&host, bytes,
                          hipHostMallocMapped | hipHostMallocCoherent) !=
            hipSuccess) {
            return false;
        }
        std::memset(host, 0, bytes);
        if (hipHostGetDevicePointer(&device, host, 0) != hipSuccess) {
            return false;
        }
        return true;
    }
    void free_mem() {
        if (host != nullptr) {
            (void)hipHostFree(host);
            host = nullptr;
            device = nullptr;
        }
    }
};

struct Harness {
    Mapped slots;
    Mapped runnable;
    Mapped requests;
    Mapped row_slots;
    Mapped row_positions;
    Mapped context;
    Mapped telemetry;
    Mapped epoch;
    Mapped row_slots_ref;
    Mapped row_positions_ref;
    Mapped context_ref;
    Mapped resources;
    Mapped resource_stack;
    Mapped resource_count;
    bool resolve_slots = false;
    hipStream_t stream = nullptr;

    bool open() {
        check(slots.alloc(kSlots * sizeof(mcu::GpuMcuSlotState)), "slots alloc");
        check(runnable.alloc(kSlots * sizeof(rt::RequestHandle)),
              "runnable alloc");
        check(requests.alloc(kMaxRequests * sizeof(rt::DeviceRequestDescriptor)),
              "requests alloc");
        check(row_slots.alloc(kMaxRows * sizeof(uint32_t)), "row slots alloc");
        check(row_positions.alloc(kMaxRows * sizeof(uint32_t)),
              "row positions alloc");
        check(context.alloc(sizeof(rt::DeviceBatchContext)), "context alloc");
        check(telemetry.alloc(sizeof(mcu::GpuMcuBatchPlanTelemetry)),
              "telemetry alloc");
        check(epoch.alloc(sizeof(uint64_t)), "epoch alloc");
        check(row_slots_ref.alloc(kMaxRows * sizeof(uint32_t)),
              "reference row slots alloc");
        check(row_positions_ref.alloc(kMaxRows * sizeof(uint32_t)),
              "reference row positions alloc");
        check(context_ref.alloc(sizeof(rt::DeviceBatchContext)),
              "reference context alloc");
        check(resources.alloc(kSlots * sizeof(mcu::DeviceSequenceResourceEntry)),
              "resources alloc");
        check(resource_stack.alloc(kSlots * sizeof(uint32_t)),
              "resource stack alloc");
        check(resource_count.alloc(sizeof(uint32_t)), "resource count alloc");
        check(hipStreamCreate(&stream) == hipSuccess, "stream create");
        return true;
    }

    mcu::GpuMcuSlotState* slot(uint32_t index) {
        return slots.host_as<mcu::GpuMcuSlotState>() + index;
    }
    rt::RequestHandle* handles() {
        return runnable.host_as<rt::RequestHandle>();
    }
    rt::DeviceRequestDescriptor* descriptors() {
        return requests.host_as<rt::DeviceRequestDescriptor>();
    }
    uint32_t* rows() {
        return row_slots.host_as<uint32_t>();
    }
    uint32_t* positions() {
        return row_positions.host_as<uint32_t>();
    }
    rt::DeviceBatchContext* ctx() {
        return context.host_as<rt::DeviceBatchContext>();
    }
    mcu::DeviceSequenceResourceManagerView manager() {
        mcu::DeviceSequenceResourceManagerView manager{};
        manager.resources =
            resources.device_as<mcu::DeviceSequenceResourceEntry>();
        manager.free_stack = resource_stack.device_as<uint32_t>();
        manager.free_count = resource_count.device_as<uint32_t>();
        manager.resource_count = kSlots;
        return manager;
    }
    void set_resource(uint32_t index, uint32_t generation, uint32_t sequence_slot) {
        mcu::DeviceSequenceResourceEntry* entries =
            resources.host_as<mcu::DeviceSequenceResourceEntry>();
        entries[index] = mcu::DeviceSequenceResourceEntry{};
        entries[index].generation = generation;
        entries[index].state = mcu::kGpuMcuResourceActive;
        entries[index].sequence_slot = sequence_slot;
        entries[index].gdn_slot = sequence_slot;
    }
    const mcu::GpuMcuBatchPlanTelemetry* telemetry_view() {
        return telemetry.host_as<mcu::GpuMcuBatchPlanTelemetry>();
    }

    void set_slot(uint32_t index, uint32_t generation, uint32_t phase,
                  uint32_t committed_position, uint32_t prompt_length,
                  uint32_t prefill_position, uint32_t candidates,
                  uint32_t sequence_length, uint32_t max_sequence_length) {
        mcu::GpuMcuSlotState* s = slot(index);
        *s = mcu::GpuMcuSlotState{};
        s->handle.slot = index;
        s->handle.generation = generation;
        s->phase = phase;
        s->committed_position = committed_position;
        s->prompt_length = prompt_length;
        s->prefill_position = prefill_position;
        s->verify_candidate_count = candidates;
        s->sequence_length = sequence_length;
        s->max_sequence_length = max_sequence_length;
    }

    bool build(uint32_t runnable_count, uint32_t max_rows,
               uint32_t max_prefill_rows_per_slot, const char* label) {
        mcu::GpuMcuBatchPlanLimits limits{};
        limits.max_requests = kMaxRequests;
        limits.max_rows = max_rows;
        limits.max_prefill_rows_per_slot = max_prefill_rows_per_slot;
        mcu::GpuMcuBatchPlanView view{};
        view.context = ctx();
        view.requests = descriptors();
        view.row_sequence_slots = rows();
        view.row_positions = positions();
        view.batch_plan_epoch = epoch.host_as<uint64_t>();
        view.resources = manager();
        view.resolve_sequence_slots = resolve_slots ? 1u : 0u;
        const Status status = mcu::gpu_mcu_build_batch_geometry_launch(
            runnable.device_as<rt::RequestHandle>(), runnable_count,
            slots.device_as<mcu::GpuMcuSlotState>(), kSlots, limits, view,
            telemetry.host_as<mcu::GpuMcuBatchPlanTelemetry>(), stream);
        check(status.ok(), label);
        if (!status.ok()) return false;
        check(hipStreamSynchronize(stream) == hipSuccess, "build sync");
        return true;
    }

    bool prepare_reference(uint32_t max_requests, const char* label) {
        rt::DeviceBatchContext* ref = context_ref.host_as<rt::DeviceBatchContext>();
        *ref = *ctx();
        ref->requests = descriptors();
        ref->row_sequence_slots = row_slots_ref.device_as<uint32_t>();
        ref->row_positions = row_positions_ref.device_as<uint32_t>();
        const Status status =
            rt::launch_prepare_batch_descriptor(ref, max_requests, stream);
        check(status.ok(), label);
        if (!status.ok()) return false;
        check(hipStreamSynchronize(stream) == hipSuccess, "reference sync");
        return true;
    }

    void close() {
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
            stream = nullptr;
        }
        slots.free_mem();
        runnable.free_mem();
        requests.free_mem();
        row_slots.free_mem();
        row_positions.free_mem();
        context.free_mem();
        telemetry.free_mem();
        epoch.free_mem();
        row_slots_ref.free_mem();
        row_positions_ref.free_mem();
        context_ref.free_mem();
        resources.free_mem();
        resource_stack.free_mem();
        resource_count.free_mem();
    }
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Harness h{};
    if (!h.open()) {
        std::printf("test_gpu_mcu_batch_planner: setup failed\n");
        return 1;
    }

    const uint32_t decode_phase =
        static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode);
    const uint32_t prefill_phase =
        static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill);
    const uint32_t verify_phase =
        static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify);

    {
        h.set_slot(2u, 4u, decode_phase, 100u, 0u, 0u, 0u, 100u, 4096u);
        h.handles()[0] = rt::RequestHandle{2u, 4u};
        check(h.build(1u, kMaxRows, 0u, "decode build"), "decode build");
        const rt::DeviceBatchContext* ctx = h.ctx();
        check(ctx->num_requests == 1u, "decode: one request");
        check(ctx->actual_rows == 1u, "decode: one row");
        check(ctx->num_decode_requests == 1u, "decode: class count");
        check(ctx->num_verify_requests == 0u, "decode: no verify");
        check(ctx->num_prefill_requests == 0u, "decode: no prefill");
        check(ctx->num_outputs == 0u, "decode: geometry only");
        const rt::DeviceRequestDescriptor& d = h.descriptors()[0];
        check(d.request_handle.slot == 2u && d.request_handle.generation == 4u,
              "decode: handle");
        check(d.row_begin == 0u, "decode: row begin");
        check(d.row_count == 1u, "decode: row count");
        check(d.prefix_length == 100u, "decode: prefix length");
        check(d.sequence_length == 101u, "decode: sequence length");
        check(d.execution_class == rt::ExecutionClass::DECODE,
              "decode: execution class");
        check(d.compute_logits == 0u && d.output_count == 0u,
              "decode: geometry only flags");
        check(d.output_index == 0xFFFFFFFFu, "decode: output index unset");
        check(h.rows()[0] == 2u, "decode: row slot");
        check(h.positions()[0] == 100u, "decode: row position");
        check(h.telemetry_view()->planned_requests == 1u,
              "decode: telemetry requests");
        check(*h.epoch.host_as<uint64_t>() == 1u, "decode: plan epoch");
    }

    {
        h.set_slot(7u, 1u, prefill_phase, 0u, 200u, 64u, 0u, 64u, 4096u);
        h.handles()[0] = rt::RequestHandle{7u, 1u};
        check(h.build(1u, 32u, 16u, "prefill build"), "prefill build");
        const rt::DeviceBatchContext* ctx = h.ctx();
        check(ctx->num_requests == 1u, "prefill: one request");
        check(ctx->actual_rows == 16u, "prefill: chunk rows");
        check(ctx->num_prefill_requests == 1u, "prefill: class count");
        const rt::DeviceRequestDescriptor& d = h.descriptors()[0];
        check(d.row_count == 16u, "prefill: row count is the chunk");
        check(d.prefix_length == 64u, "prefill: prefix length");
        check(d.sequence_length == 80u, "prefill: sequence length");
        check(d.execution_class == rt::ExecutionClass::PREFILL,
              "prefill: execution class");
        bool rows_ok = true;
        for (uint32_t t = 0; t < 16u; ++t) {
            if (h.rows()[t] != 7u) rows_ok = false;
            if (h.positions()[t] != 64u + t) rows_ok = false;
        }
        check(rows_ok, "prefill: row mapping covers the chunk");
        check(h.slot(7u)->prefill_position == 64u,
              "prefill: the plan does not commit progress");
        check(h.slot(7u)->generated_tokens == 0u,
              "prefill: generated tokens untouched");
    }

    {
        h.set_slot(5u, 2u, verify_phase, 500u, 0u, 0u, 8u, 500u, 4096u);
        h.handles()[0] = rt::RequestHandle{5u, 2u};
        check(h.build(1u, kMaxRows, 0u, "verify build"), "verify build");
        const rt::DeviceBatchContext* ctx = h.ctx();
        check(ctx->num_requests == 1u, "verify: one request");
        check(ctx->actual_rows == 8u, "verify: candidate rows");
        check(ctx->num_verify_requests == 1u, "verify: class count");
        const rt::DeviceRequestDescriptor& d = h.descriptors()[0];
        check(d.row_count == 8u, "verify: row count");
        check(d.prefix_length == 500u, "verify: prefix length");
        check(d.sequence_length == 508u, "verify: sequence length");
        check(d.execution_class == rt::ExecutionClass::SPEC_VERIFY,
              "verify: execution class");
        bool rows_ok = true;
        for (uint32_t t = 0; t < 8u; ++t) {
            if (h.rows()[t] != 5u) rows_ok = false;
            if (h.positions()[t] != 500u + t) rows_ok = false;
        }
        check(rows_ok, "verify: row mapping");
        check(h.slot(5u)->committed_position == 500u,
              "verify: the plan does not commit progress");
    }

    {
        h.set_slot(3u, 1u, verify_phase, 10u, 0u, 0u, 32u, 10u, 4096u);
        h.handles()[0] = rt::RequestHandle{3u, 1u};
        check(h.build(1u, 32u, 0u, "verify 32 build"), "verify 32 build");
        check(h.ctx()->num_requests == 1u && h.ctx()->actual_rows == 32u,
              "verify: 32 candidates fit");
        check(h.descriptors()[0].row_count == 32u, "verify: 32 row count");

        h.set_slot(3u, 2u, verify_phase, 10u, 0u, 0u, 33u, 10u, 4096u);
        h.handles()[0] = rt::RequestHandle{3u, 2u};
        check(h.build(1u, 32u, 0u, "verify 33 build"), "verify 33 build");
        check(h.ctx()->num_requests == 0u && h.ctx()->actual_rows == 0u,
              "verify: 33 candidates are rejected, not clamped");
        check(h.telemetry_view()->rejected_invalid == 1u,
              "verify: 33 counts as invalid");
    }

    {
        h.set_slot(7u, 1u, prefill_phase, 0u, 200u, 64u, 0u, 64u, 4096u);
        h.set_slot(2u, 4u, decode_phase, 100u, 0u, 0u, 0u, 100u, 4096u);
        h.set_slot(9u, 3u, verify_phase, 500u, 0u, 0u, 8u, 500u, 4096u);
        h.set_slot(4u, 5u, decode_phase, 700u, 0u, 0u, 0u, 700u, 4096u);
        h.handles()[0] = rt::RequestHandle{7u, 1u};
        h.handles()[1] = rt::RequestHandle{2u, 4u};
        h.handles()[2] = rt::RequestHandle{9u, 3u};
        h.handles()[3] = rt::RequestHandle{4u, 5u};
        check(h.build(4u, kMaxRows, 16u, "mixed build"), "mixed build");
        check(h.ctx()->num_requests == 4u, "mixed: four descriptors");
        check(h.ctx()->actual_rows == 26u, "mixed: 26 rows");
        check(h.ctx()->num_decode_requests == 2u, "mixed: two decodes");
        check(h.ctx()->num_verify_requests == 1u, "mixed: one verify");
        check(h.ctx()->num_prefill_requests == 1u, "mixed: one prefill");
        const rt::DeviceRequestDescriptor* d = h.descriptors();
        check(d[0].request_handle.slot == 2u && d[0].row_begin == 0u &&
                  d[0].row_count == 1u,
              "mixed: descriptor 0 is the slot 2 decode");
        check(d[1].request_handle.slot == 4u && d[1].row_begin == 1u &&
                  d[1].row_count == 1u,
              "mixed: descriptor 1 is the slot 4 decode");
        check(d[2].request_handle.slot == 9u && d[2].row_begin == 2u &&
                  d[2].row_count == 8u,
              "mixed: descriptor 2 is the slot 9 verify");
        check(d[3].request_handle.slot == 7u && d[3].row_begin == 10u &&
                  d[3].row_count == 16u,
              "mixed: descriptor 3 is the slot 7 prefill");
        check(h.rows()[0] == 2u && h.rows()[1] == 4u, "mixed: decode rows");
        bool verify_rows = true;
        for (uint32_t r = 2u; r < 10u; ++r) {
            if (h.rows()[r] != 9u) verify_rows = false;
            if (h.positions()[r] != 500u + (r - 2u)) verify_rows = false;
        }
        bool prefill_rows = true;
        for (uint32_t r = 10u; r < 26u; ++r) {
            if (h.rows()[r] != 7u) prefill_rows = false;
            if (h.positions()[r] != 64u + (r - 10u)) prefill_rows = false;
        }
        check(verify_rows, "mixed: verify rows");
        check(prefill_rows, "mixed: prefill rows");
    }

    {
        h.set_slot(2u, 1u, decode_phase, 10u, 0u, 0u, 0u, 10u, 4096u);
        h.set_slot(4u, 1u, decode_phase, 20u, 0u, 0u, 0u, 20u, 4096u);
        h.set_slot(7u, 1u, prefill_phase, 0u, 200u, 0u, 0u, 0u, 4096u);
        h.handles()[0] = rt::RequestHandle{2u, 1u};
        h.handles()[1] = rt::RequestHandle{4u, 1u};
        h.handles()[2] = rt::RequestHandle{7u, 1u};
        check(h.build(3u, 4u, 0u, "row capacity build"), "row capacity build");
        check(h.ctx()->actual_rows == 4u, "capacity: four rows");
        check(h.ctx()->num_requests == 3u, "capacity: three descriptors");
        check(h.descriptors()[2].row_count == 2u,
              "capacity: the prefill shrinks to the remaining rows");
        check(h.slot(7u)->prefill_position == 0u,
              "capacity: the shrunken plan does not commit");
    }

    {
        h.set_slot(9u, 1u, verify_phase, 500u, 0u, 0u, 16u, 500u, 4096u);
        h.handles()[0] = rt::RequestHandle{9u, 1u};
        check(h.build(1u, 8u, 0u, "verify partial build"),
              "verify partial build");
        check(h.ctx()->num_requests == 0u,
              "capacity: a verify window is not split");
        check(h.ctx()->actual_rows == 0u, "capacity: no partial verify rows");
        check(h.telemetry_view()->skipped_no_capacity == 1u,
              "capacity: the verify is counted as skipped");
    }

    {
        h.set_slot(7u, 1u, prefill_phase, 0u, 200u, 64u, 0u, 64u, 4096u);
        h.set_slot(2u, 4u, decode_phase, 100u, 0u, 0u, 0u, 100u, 4096u);
        h.set_slot(9u, 3u, verify_phase, 500u, 0u, 0u, 8u, 500u, 4096u);
        h.set_slot(4u, 5u, decode_phase, 700u, 0u, 0u, 0u, 700u, 4096u);
        h.handles()[0] = rt::RequestHandle{7u, 1u};
        h.handles()[1] = rt::RequestHandle{2u, 4u};
        h.handles()[2] = rt::RequestHandle{9u, 3u};
        h.handles()[3] = rt::RequestHandle{4u, 5u};
        check(h.build(4u, kMaxRows, 16u, "parity build"), "parity build");
        check(h.ctx()->actual_rows == 26u && h.ctx()->num_requests == 4u,
              "parity: geometry is the mixed batch");
        check(h.prepare_reference(kMaxRequests, "reference launch"),
              "parity: the existing kernel runs");
        const std::size_t bytes = 26u * sizeof(uint32_t);
        check(std::memcmp(h.rows(), h.row_slots_ref.host, bytes) == 0,
              "parity: row_sequence_slots byte equal");
        check(std::memcmp(h.positions(), h.row_positions_ref.host, bytes) == 0,
              "parity: row_positions byte equal");
    }

    {
        h.set_slot(3u, 6u, decode_phase, 10u, 0u, 0u, 0u, 10u, 4096u);
        h.handles()[0] = rt::RequestHandle{3u, 5u};
        check(h.build(1u, kMaxRows, 0u, "stale build"), "stale build");
        check(h.ctx()->num_requests == 0u,
              "stale: a reused slot is not planned");
        check(h.telemetry_view()->stale_handles == 1u,
              "stale: the count is reported");
        check(h.slot(3u)->handle.generation == 6u,
              "stale: the current slot is untouched");
        h.handles()[0] = rt::RequestHandle{3u, 6u};
        check(h.build(1u, kMaxRows, 0u, "fresh build"), "fresh build");
        check(h.ctx()->num_requests == 1u,
              "stale: the current handle is planned");
    }

    {
        h.set_slot(1u, 1u, decode_phase, 10u, 0u, 0u, 0u, 10u, 4096u);
        h.slot(1u)->terminal_reason =
            static_cast<uint32_t>(mcu::GpuMcuTerminalReason::eos);
        h.set_slot(2u, 1u, decode_phase, 10u, 0u, 0u, 0u, 10u, 4096u);
        h.slot(2u)->cancel_requested = 1u;
        h.set_slot(4u, 1u, decode_phase, 10u, 0u, 0u, 0u, 10u, 4096u);
        h.slot(4u)->output_blocked = 1u;
        h.set_slot(5u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 4096u);
        h.set_slot(6u, 1u, prefill_phase, 0u, 0u, 0u, 0u, 0u, 4096u);
        h.handles()[0] = rt::RequestHandle{1u, 1u};
        h.handles()[1] = rt::RequestHandle{2u, 1u};
        h.handles()[2] = rt::RequestHandle{4u, 1u};
        h.handles()[3] = rt::RequestHandle{5u, 1u};
        h.handles()[4] = rt::RequestHandle{6u, 1u};
        check(h.build(5u, kMaxRows, 0u, "filter build"), "filter build");
        check(h.ctx()->num_requests == 0u,
              "filter: terminal, cancelled, blocked, idle and empty prefill plan"
              " nothing");
        check(h.telemetry_view()->skipped_not_plannable == 3u,
              "filter: terminal, cancelled and blocked are counted");
        check(h.telemetry_view()->empty_prefill == 1u,
              "filter: an exhausted prefill is counted");
    }

    {
        h.set_slot(2u, 4u, decode_phase, 100u, 0u, 0u, 0u, 100u, 4096u);
        h.set_slot(9u, 3u, verify_phase, 500u, 0u, 0u, 8u, 500u, 4096u);
        h.set_slot(7u, 1u, prefill_phase, 0u, 200u, 64u, 0u, 64u, 4096u);
        h.handles()[0] = rt::RequestHandle{2u, 4u};
        h.handles()[1] = rt::RequestHandle{9u, 3u};
        h.handles()[2] = rt::RequestHandle{7u, 1u};
        std::vector<mcu::GpuMcuSlotState> before(kSlots, mcu::GpuMcuSlotState{});
        std::memcpy(before.data(), h.slots.host,
                    kSlots * sizeof(mcu::GpuMcuSlotState));
        check(h.build(3u, kMaxRows, 16u, "mutation build"), "mutation build");
        check(h.ctx()->num_requests == 3u, "mutation: three descriptors");
        check(std::memcmp(before.data(), h.slots.host,
                          kSlots * sizeof(mcu::GpuMcuSlotState)) == 0,
              "mutation: the plan does not mutate the slot table");
    }

    {
        h.resolve_slots = true;
        h.set_slot(2u, 4u, decode_phase, 100u, 0u, 0u, 0u, 100u, 4096u);
        h.slot(2u)->kv_sequence_handle = mcu::gpu_mcu_resource_handle(1u, 7u);
        h.set_resource(1u, 7u, 5u);
        h.handles()[0] = rt::RequestHandle{2u, 4u};
        check(h.build(1u, kMaxRows, 0u, "adapter build"), "adapter build");
        check(h.ctx()->num_requests == 1u, "adapter: one request");
        check(h.rows()[0] == 5u,
              "adapter: the row carries the physical sequence slot");
        check(h.telemetry_view()->unresolved_resources == 0u,
              "adapter: a resolvable handle is not counted");

        h.set_resource(1u, 8u, 5u);
        check(h.build(1u, kMaxRows, 0u, "stale adapter build"),
              "stale adapter build");
        check(h.ctx()->num_requests == 0u,
              "adapter: a stale resource handle is not planned");
        check(h.telemetry_view()->unresolved_resources == 1u,
              "adapter: the stale resource handle is counted");
        h.resolve_slots = false;
    }

    h.close();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_batch_planner: failed=%d\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_batch_planner: R1+R2+R3+R4-filter+adapter PASS\n");
    return 0;
}
