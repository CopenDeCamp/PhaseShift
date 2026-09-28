#include <phaseshift/runtime/gpu_mcu/batch_binding.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace mcu = ::ps::runtime::gpu_mcu;
namespace rt = ::ps::runtime;

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
constexpr uint32_t kRequests = 4u;
constexpr uint32_t kRows = 32u;
constexpr uint32_t kOutputs = 32u;
constexpr uint32_t kPrompt = 128u;

struct Mapped {
    void* host = nullptr;
    void* device = nullptr;
    bool alloc(std::size_t bytes) {
        if (hipHostMalloc(&host, bytes,
                          hipHostMallocMapped | hipHostMallocCoherent) !=
            hipSuccess) {
            return false;
        }
        std::memset(host, 0, bytes);
        return hipHostGetDevicePointer(&device, host, 0) == hipSuccess;
    }
    template <typename T>
    T* host_as() const {
        return static_cast<T*>(host);
    }
    template <typename T>
    T* device_as() const {
        return static_cast<T*>(device);
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
    Mapped bindings;
    Mapped requests;
    Mapped token_ids;
    Mapped output_rows;
    Mapped output_sampling;
    Mapped context;
    Mapped telemetry;
    Mapped epoch;
    Mapped prompt;
    Mapped verify_tokens;
    Mapped sampling;
    Mapped ref_rows;
    Mapped ref_sampling;
    Mapped ref_context;
    Mapped ref_row_slots;
    Mapped ref_row_positions;
    hipStream_t stream = nullptr;

    bool open() {
        check(slots.alloc(kSlots * sizeof(mcu::GpuMcuSlotState)), "slots");
        check(bindings.alloc(kSlots * sizeof(mcu::GpuMcuSlotBinding)),
              "bindings");
        check(requests.alloc(kRequests * sizeof(rt::DeviceRequestDescriptor)),
              "requests");
        check(token_ids.alloc(kRows * sizeof(int32_t)), "token ids");
        check(output_rows.alloc(kOutputs * sizeof(uint32_t)), "output rows");
        check(output_sampling.alloc(kOutputs * sizeof(rt::DeviceSamplingParams)),
              "output sampling");
        check(context.alloc(sizeof(rt::DeviceBatchContext)), "context");
        check(telemetry.alloc(sizeof(mcu::GpuMcuBatchBindingTelemetry)),
              "telemetry");
        check(epoch.alloc(sizeof(uint64_t)), "epoch");
        check(prompt.alloc(kPrompt * sizeof(int32_t)), "prompt");
        check(verify_tokens.alloc(kRows * sizeof(int32_t)), "verify tokens");
        check(sampling.alloc(sizeof(rt::DeviceSamplingParams)), "sampling");
        check(ref_rows.alloc(kOutputs * sizeof(uint32_t)), "ref rows");
        check(ref_sampling.alloc(kOutputs * sizeof(rt::DeviceSamplingParams)),
              "ref sampling");
        check(ref_context.alloc(sizeof(rt::DeviceBatchContext)), "ref context");
        check(ref_row_slots.alloc(kRows * sizeof(uint32_t)), "ref row slots");
        check(ref_row_positions.alloc(kRows * sizeof(uint32_t)),
              "ref row positions");
        check(hipStreamCreate(&stream) == hipSuccess, "stream");
        return true;
    }

    mcu::GpuMcuSlotState* slot(uint32_t i) {
        return slots.host_as<mcu::GpuMcuSlotState>() + i;
    }
    mcu::GpuMcuSlotBinding* binding(uint32_t i) {
        return bindings.host_as<mcu::GpuMcuSlotBinding>() + i;
    }
    rt::DeviceRequestDescriptor* descriptor(uint32_t i) {
        return requests.host_as<rt::DeviceRequestDescriptor>() + i;
    }
    int32_t* tokens() { return token_ids.host_as<int32_t>(); }
    uint32_t* rows() { return output_rows.host_as<uint32_t>(); }
    rt::DeviceSamplingParams* sampling_out() {
        return output_sampling.host_as<rt::DeviceSamplingParams>();
    }
    rt::DeviceBatchContext* ctx() {
        return context.host_as<rt::DeviceBatchContext>();
    }
    const mcu::GpuMcuBatchBindingTelemetry* telemetry_view() {
        return telemetry.host_as<mcu::GpuMcuBatchBindingTelemetry>();
    }

    void reset(uint32_t num_requests) {
        std::memset(context.host, 0, sizeof(rt::DeviceBatchContext));
        std::memset(telemetry.host, 0, sizeof(mcu::GpuMcuBatchBindingTelemetry));
        std::memset(requests.host, 0,
                    kRequests * sizeof(rt::DeviceRequestDescriptor));
        std::memset(token_ids.host, 0, kRows * sizeof(int32_t));
        std::memset(output_rows.host, 0, kOutputs * sizeof(uint32_t));
        std::memset(output_sampling.host, 0,
                    kOutputs * sizeof(rt::DeviceSamplingParams));
        for (uint32_t i = 0; i < kSlots; ++i) {
            *slot(i) = mcu::GpuMcuSlotState{};
            *binding(i) = mcu::GpuMcuSlotBinding{};
        }
        ctx()->num_requests = num_requests;
        ctx()->actual_rows = kRows;
    }

    bool bind(const char* label) {
        mcu::GpuMcuBatchBindingView view{};
        view.context = context.device_as<rt::DeviceBatchContext>();
        view.requests = requests.device_as<rt::DeviceRequestDescriptor>();
        view.token_ids = token_ids.device_as<int32_t>();
        view.output_rows = output_rows.device_as<uint32_t>();
        view.output_sampling_params =
            output_sampling.device_as<rt::DeviceSamplingParams>();
        view.slots = slots.device_as<mcu::GpuMcuSlotState>();
        view.bindings = bindings.device_as<mcu::GpuMcuSlotBinding>();
        view.max_slots = kSlots;
        view.max_outputs = kOutputs;
        view.batch_ready_epoch = epoch.device_as<uint64_t>();
        const Status status = mcu::gpu_mcu_bind_batch_io_launch(
            view, telemetry.device_as<mcu::GpuMcuBatchBindingTelemetry>(),
            stream);
        check(status.ok(), label);
        if (!status.ok()) return false;
        check(hipStreamSynchronize(stream) == hipSuccess, "bind sync");
        return true;
    }

    bool prepare_reference(uint32_t max_requests, const char* label) {
        rt::DeviceBatchContext* ref =
            ref_context.host_as<rt::DeviceBatchContext>();
        *ref = *ctx();
        ref->requests = requests.device_as<rt::DeviceRequestDescriptor>();
        ref->row_sequence_slots = ref_row_slots.device_as<uint32_t>();
        ref->row_positions = ref_row_positions.device_as<uint32_t>();
        ref->output_rows = ref_rows.device_as<uint32_t>();
        ref->output_sampling_params =
            ref_sampling.device_as<rt::DeviceSamplingParams>();
        const Status status =
            rt::launch_prepare_batch_descriptor(ref, max_requests, stream);
        check(status.ok(), label);
        if (!status.ok()) return false;
        check(hipStreamSynchronize(stream) == hipSuccess, "ref sync");
        return true;
    }

    void close() {
        if (stream != nullptr) (void)hipStreamDestroy(stream);
        slots.free_mem();
        bindings.free_mem();
        requests.free_mem();
        token_ids.free_mem();
        output_rows.free_mem();
        output_sampling.free_mem();
        context.free_mem();
        telemetry.free_mem();
        epoch.free_mem();
        prompt.free_mem();
        verify_tokens.free_mem();
        sampling.free_mem();
        ref_rows.free_mem();
        ref_sampling.free_mem();
        ref_context.free_mem();
        ref_row_slots.free_mem();
        ref_row_positions.free_mem();
    }
};

void set_prefill(Harness& h, uint32_t index, uint32_t generation,
                 uint32_t prompt_length, uint32_t prefill_position,
                 uint32_t max_new_tokens, uint32_t generated_tokens) {
    mcu::GpuMcuSlotState* s = h.slot(index);
    *s = mcu::GpuMcuSlotState{};
    s->handle.slot = index;
    s->handle.generation = generation;
    s->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill);
    s->prompt_length = prompt_length;
    s->prefill_position = prefill_position;
    s->max_new_tokens = max_new_tokens;
    s->generated_tokens = generated_tokens;
    s->sequence_length = prefill_position;
    s->max_sequence_length = 4096u;
    mcu::GpuMcuSlotBinding* b = h.binding(index);
    *b = mcu::GpuMcuSlotBinding{};
    b->handle = s->handle;
    b->prompt_tokens_handle =
        mcu::gpu_mcu_encode_device_address(h.prompt.host);
    b->sampling_params_handle =
        mcu::gpu_mcu_encode_device_address(h.sampling.host);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Harness h{};
    if (!h.open()) {
        std::printf("test_gpu_mcu_batch_binding: setup failed\n");
        return 1;
    }
    for (uint32_t i = 0; i < kPrompt; ++i) {
        h.prompt.host_as<int32_t>()[i] = 10 + static_cast<int32_t>(i);
    }

    {
        h.reset(1u);
        set_prefill(h, 0u, 3u, 100u, 20u, 8u, 0u);
        rt::DeviceRequestDescriptor* d = h.descriptor(0);
        d->request_handle = rt::RequestHandle{0u, 3u};
        d->row_begin = 0u;
        d->row_count = 8u;
        d->prefix_length = 20u;
        d->sequence_length = 36u;
        d->execution_class = rt::ExecutionClass::PREFILL;
        check(h.bind("prefill bind"), "prefill bind");
        bool tokens_ok = true;
        for (uint32_t t = 0; t < 8u; ++t) {
            if (h.tokens()[t] != 10 + 20 + static_cast<int32_t>(t)) {
                tokens_ok = false;
            }
        }
        check(tokens_ok, "PREFILL packs prompt[prefix + t]");
        check(d->compute_logits == 0u && d->output_count == 0u,
              "non-final PREFILL has no output");
        check(d->output_index == 0xFFFFFFFFu, "non-final PREFILL output index");
        check(h.ctx()->num_outputs == 0u, "non-final PREFILL num_outputs");
    }

    {
        h.reset(1u);
        set_prefill(h, 0u, 3u, 100u, 84u, 8u, 0u);
        rt::DeviceRequestDescriptor* d = h.descriptor(0);
        d->request_handle = rt::RequestHandle{0u, 3u};
        d->row_count = 16u;
        d->prefix_length = 84u;
        d->sequence_length = 100u;
        d->execution_class = rt::ExecutionClass::PREFILL;
        check(h.bind("final prefill bind"), "final prefill bind");
        check(d->compute_logits == 1u && d->output_count == 1u,
              "final PREFILL samples one token");
        check(d->output_index == 0u, "final PREFILL output index");
        check(h.ctx()->num_outputs == 1u, "final PREFILL num_outputs");
        check(h.rows()[0] == 15u, "final PREFILL outputs the last row");

        h.reset(1u);
        set_prefill(h, 0u, 3u, 100u, 84u, 0u, 0u);
        rt::DeviceRequestDescriptor* d2 = h.descriptor(0);
        d2->request_handle = rt::RequestHandle{0u, 3u};
        d2->row_count = 16u;
        d2->prefix_length = 84u;
        d2->sequence_length = 100u;
        d2->execution_class = rt::ExecutionClass::PREFILL;
        check(h.bind("prefill only bind"), "prefill only bind");
        check(d2->compute_logits == 0u && d2->output_count == 0u,
              "a PREFILL only request has no output");
        check(h.ctx()->num_outputs == 0u, "PREFILL only num_outputs");
    }

    {
        h.reset(1u);
        mcu::GpuMcuSlotState* s = h.slot(1);
        *s = mcu::GpuMcuSlotState{};
        s->handle.slot = 1u;
        s->handle.generation = 5u;
        s->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode);
        s->committed_position = 40u;
        s->generated_tokens = 5u;
        s->max_new_tokens = 8u;
        s->sequence_length = 40u;
        s->max_sequence_length = 4096u;
        mcu::GpuMcuSlotBinding* b = h.binding(1);
        *b = mcu::GpuMcuSlotBinding{};
        b->handle = s->handle;
        b->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);

        rt::DeviceRequestDescriptor* d = h.descriptor(0);
        d->request_handle = rt::RequestHandle{1u, 5u};
        d->row_begin = 0u;
        d->row_count = 1u;
        d->prefix_length = 40u;
        d->sequence_length = 41u;
        d->execution_class = rt::ExecutionClass::DECODE;
        b->decode_input_token = 12345;
        b->decode_input_valid = 1u;
        check(h.bind("decode bind"), "decode bind");
        check(h.tokens()[0] == 12345, "DECODE uses the bound input token");
        check(d->compute_logits == 1u && d->output_count == 1u,
              "DECODE samples one token");
        check(d->sampling.sample_index == 5u,
              "sampling index starts at generated_tokens");
        check(h.ctx()->num_outputs == 1u, "DECODE num_outputs");

        h.reset(1u);
        mcu::GpuMcuSlotState* s2 = h.slot(1);
        *s2 = mcu::GpuMcuSlotState{};
        s2->handle.slot = 1u;
        s2->handle.generation = 5u;
        s2->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode);
        s2->committed_position = 40u;
        s2->generated_tokens = 5u;
        s2->max_new_tokens = 8u;
        s2->sequence_length = 40u;
        s2->max_sequence_length = 4096u;
        mcu::GpuMcuSlotBinding* b2 = h.binding(1);
        *b2 = mcu::GpuMcuSlotBinding{};
        b2->handle = s2->handle;
        b2->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);
        b2->decode_input_valid = 0u;
        rt::DeviceRequestDescriptor* d2 = h.descriptor(0);
        *d2 = *d;
        const uint64_t before = *h.epoch.host_as<uint64_t>();
        check(h.bind("missing decode bind"), "missing decode bind");
        check(*h.epoch.host_as<uint64_t>() == before,
              "a missing decode token does not publish the ready epoch");
        check(d2->output_count == 0u && d2->compute_logits == 0u,
              "a missing decode token binds no output");
    }

    {
        h.reset(1u);
        mcu::GpuMcuSlotState* s = h.slot(2);
        *s = mcu::GpuMcuSlotState{};
        s->handle.slot = 2u;
        s->handle.generation = 7u;
        s->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify);
        s->committed_position = 50u;
        s->verify_candidate_count = 8u;
        s->verify_candidate_ref =
            mcu::gpu_mcu_encode_device_address(h.verify_tokens.host);
        s->generated_tokens = 17u;
        s->max_new_tokens = 32u;
        s->sequence_length = 50u;
        s->max_sequence_length = 4096u;
        mcu::GpuMcuSlotBinding* b = h.binding(2);
        *b = mcu::GpuMcuSlotBinding{};
        b->handle = s->handle;
        b->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);
        for (uint32_t t = 0; t < 8u; ++t) {
            h.verify_tokens.host_as<int32_t>()[t] = 100 + static_cast<int32_t>(t);
        }
        rt::DeviceRequestDescriptor* d = h.descriptor(0);
        d->request_handle = rt::RequestHandle{2u, 7u};
        d->row_begin = 0u;
        d->row_count = 8u;
        d->prefix_length = 50u;
        d->sequence_length = 58u;
        d->execution_class = rt::ExecutionClass::SPEC_VERIFY;
        check(h.bind("verify bind"), "verify bind");
        bool tokens_ok = true;
        for (uint32_t t = 0; t < 8u; ++t) {
            if (h.tokens()[t] != 100 + static_cast<int32_t>(t)) tokens_ok = false;
        }
        check(tokens_ok, "VERIFY packs the candidate tokens");
        check(d->compute_logits == 1u && d->output_count == 8u,
              "VERIFY takes one output per row");
        for (uint32_t j = 0; j < 8u; ++j) {
            if (h.sampling_out()[j].sample_index != 17u + j) {
                tokens_ok = false;
            }
            if (h.rows()[j] != j) tokens_ok = false;
        }
        check(tokens_ok, "VERIFY output rows and sampling indices");
        check(h.ctx()->num_outputs == 8u, "VERIFY num_outputs");
    }

    {
        h.reset(1u);
        mcu::GpuMcuSlotState* s = h.slot(3);
        *s = mcu::GpuMcuSlotState{};
        s->handle.slot = 3u;
        s->handle.generation = 1u;
        s->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode);
        s->generated_tokens = 0u;
        s->max_new_tokens = 8u;
        s->sequence_length = 0u;
        s->max_sequence_length = 4096u;
        mcu::GpuMcuSlotBinding* b = h.binding(3);
        *b = mcu::GpuMcuSlotBinding{};
        b->handle = s->handle;
        b->constraint_state_handle = 0xABCull;
        b->decode_input_token = 5;
        b->decode_input_valid = 1u;
        rt::DeviceRequestDescriptor* d = h.descriptor(0);
        d->request_handle = rt::RequestHandle{3u, 1u};
        d->row_count = 1u;
        d->execution_class = rt::ExecutionClass::DECODE;
        const uint64_t before = *h.epoch.host_as<uint64_t>();
        check(h.bind("constraint bind"), "constraint bind");
        check(*h.epoch.host_as<uint64_t>() == before,
              "a constrained request does not publish the ready epoch");
        check(d->output_count == 0u && d->compute_logits == 0u,
              "a constrained request binds no output");
    }

    {
        h.reset(4u);
        mcu::GpuMcuSlotBinding* ba = h.binding(0);
        mcu::GpuMcuSlotState* sa = h.slot(0);
        *sa = mcu::GpuMcuSlotState{};
        sa->handle.slot = 0u;
        sa->handle.generation = 1u;
        sa->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::decode);
        sa->generated_tokens = 5u;
        sa->max_new_tokens = 8u;
        sa->max_sequence_length = 4096u;
        *ba = mcu::GpuMcuSlotBinding{};
        ba->handle = sa->handle;
        ba->decode_input_token = 1000;
        ba->decode_input_valid = 1u;
        ba->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);

        mcu::GpuMcuSlotBinding* bb = h.binding(1);
        mcu::GpuMcuSlotState* sb = h.slot(1);
        *sb = *sa;
        sb->handle.slot = 1u;
        sb->generated_tokens = 6u;
        *bb = *ba;
        bb->handle = sb->handle;
        bb->decode_input_token = 1001;

        mcu::GpuMcuSlotBinding* bc = h.binding(2);
        mcu::GpuMcuSlotState* sc = h.slot(2);
        *sc = mcu::GpuMcuSlotState{};
        sc->handle.slot = 2u;
        sc->handle.generation = 1u;
        sc->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify);
        sc->verify_candidate_count = 8u;
        sc->verify_candidate_ref =
            mcu::gpu_mcu_encode_device_address(h.verify_tokens.host);
        sc->generated_tokens = 17u;
        sc->max_new_tokens = 32u;
        sc->max_sequence_length = 4096u;
        *bc = mcu::GpuMcuSlotBinding{};
        bc->handle = sc->handle;
        bc->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);
        for (uint32_t t = 0; t < 8u; ++t) {
            h.verify_tokens.host_as<int32_t>()[t] = 200 + static_cast<int32_t>(t);
        }

        mcu::GpuMcuSlotBinding* bd = h.binding(3);
        mcu::GpuMcuSlotState* sd = h.slot(3);
        *sd = mcu::GpuMcuSlotState{};
        sd->handle.slot = 3u;
        sd->handle.generation = 1u;
        sd->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::prefill);
        sd->prompt_length = 200u;
        sd->prefill_position = 64u;
        sd->max_new_tokens = 8u;
        sd->max_sequence_length = 4096u;
        *bd = mcu::GpuMcuSlotBinding{};
        bd->handle = sd->handle;
        bd->prompt_tokens_handle =
            mcu::gpu_mcu_encode_device_address(h.prompt.host);
        bd->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);

        rt::DeviceRequestDescriptor* d0 = h.descriptor(0);
        d0->request_handle = sa->handle;
        d0->row_begin = 0u;
        d0->row_count = 1u;
        d0->prefix_length = 0u;
        d0->sequence_length = 1u;
        d0->execution_class = rt::ExecutionClass::DECODE;
        rt::DeviceRequestDescriptor* d1 = h.descriptor(1);
        d1->request_handle = sb->handle;
        d1->row_begin = 1u;
        d1->row_count = 1u;
        d1->prefix_length = 0u;
        d1->sequence_length = 1u;
        d1->execution_class = rt::ExecutionClass::DECODE;
        rt::DeviceRequestDescriptor* d2 = h.descriptor(2);
        d2->request_handle = sc->handle;
        d2->row_begin = 2u;
        d2->row_count = 8u;
        d2->prefix_length = 0u;
        d2->sequence_length = 8u;
        d2->execution_class = rt::ExecutionClass::SPEC_VERIFY;
        rt::DeviceRequestDescriptor* d3 = h.descriptor(3);
        d3->request_handle = sd->handle;
        d3->row_begin = 10u;
        d3->row_count = 16u;
        d3->prefix_length = 64u;
        d3->sequence_length = 80u;
        d3->execution_class = rt::ExecutionClass::PREFILL;

        std::vector<mcu::GpuMcuSlotState> before(
            kSlots, mcu::GpuMcuSlotState{});
        std::memcpy(before.data(), h.slots.host,
                    kSlots * sizeof(mcu::GpuMcuSlotState));

        check(h.bind("mixed bind"), "mixed bind");
        check(h.tokens()[0] == 1000 && h.tokens()[1] == 1001,
              "mixed: decode tokens");
        bool verify_ok = true;
        for (uint32_t t = 0; t < 8u; ++t) {
            if (h.tokens()[2u + t] != 200 + static_cast<int32_t>(t)) {
                verify_ok = false;
            }
        }
        check(verify_ok, "mixed: verify tokens");
        bool prefill_ok = true;
        for (uint32_t t = 0; t < 16u; ++t) {
            if (h.tokens()[10u + t] != 10 + 64 + static_cast<int32_t>(t)) {
                prefill_ok = false;
            }
        }
        check(prefill_ok, "mixed: prefill tokens");
        check(h.ctx()->num_outputs == 10u, "mixed: ten outputs");
        check(d0->output_index == 0u && d1->output_index == 1u &&
                  d2->output_index == 2u && d3->output_index == 0xFFFFFFFFu,
              "mixed: output index allocation");
        check(d0->sampling.sample_index == 5u, "mixed: decode sampling index");
        bool sampling_ok = true;
        for (uint32_t j = 0; j < 8u; ++j) {
            if (h.sampling_out()[2u + j].sample_index != 17u + j) {
                sampling_ok = false;
            }
        }
        check(sampling_ok, "mixed: verify sampling indices");
        check(d3->compute_logits == 0u && d3->output_count == 0u,
              "mixed: a non-final prefill has no output");

        check(std::memcmp(before.data(), h.slots.host,
                          kSlots * sizeof(mcu::GpuMcuSlotState)) == 0,
              "binding does not mutate the slot table");

        check(h.prepare_reference(kRequests, "reference launch"),
              "reference mapper runs");
        const std::size_t output_bytes =
            h.ctx()->num_outputs * sizeof(uint32_t);
        check(std::memcmp(h.rows(), h.ref_rows.host, output_bytes) == 0,
              "output_rows byte equal to the existing mapper");
        check(std::memcmp(h.sampling_out(), h.ref_sampling.host,
                          h.ctx()->num_outputs *
                              sizeof(rt::DeviceSamplingParams)) == 0,
              "output_sampling_params byte equal to the existing mapper");
    }

    {
        h.reset(1u);
        mcu::GpuMcuSlotState* s = h.slot(0);
        *s = mcu::GpuMcuSlotState{};
        s->handle.slot = 0u;
        s->handle.generation = 2u;
        s->phase = static_cast<uint32_t>(mcu::GpuMcuSlotPhase::verify);
        s->verify_candidate_count = 32u;
        s->verify_candidate_ref =
            mcu::gpu_mcu_encode_device_address(h.verify_tokens.host);
        s->generated_tokens = 0u;
        s->max_new_tokens = 32u;
        s->max_sequence_length = 4096u;
        mcu::GpuMcuSlotBinding* b = h.binding(0);
        *b = mcu::GpuMcuSlotBinding{};
        b->handle = s->handle;
        b->sampling_params_handle =
            mcu::gpu_mcu_encode_device_address(h.sampling.host);
        for (uint32_t t = 0; t < 32u; ++t) {
            h.verify_tokens.host_as<int32_t>()[t] = 300 + static_cast<int32_t>(t);
        }
        rt::DeviceRequestDescriptor* d = h.descriptor(0);
        d->request_handle = s->handle;
        d->row_begin = 0u;
        d->row_count = 32u;
        d->prefix_length = 0u;
        d->sequence_length = 32u;
        d->execution_class = rt::ExecutionClass::SPEC_VERIFY;
        check(h.bind("verify 32 bind"), "verify 32 bind");
        bool tokens_ok = true;
        for (uint32_t t = 0; t < 32u; ++t) {
            if (h.tokens()[t] != 300 + static_cast<int32_t>(t)) tokens_ok = false;
        }
        check(tokens_ok, "VERIFY 32 packs every candidate");
        check(d->compute_logits == 1u && d->output_count == 32u,
              "VERIFY 32 asks for 32 outputs");
        check(h.ctx()->num_outputs == 32u, "VERIFY 32 num_outputs");
        bool sampling_ok = true;
        for (uint32_t j = 0; j < 32u; ++j) {
            if (h.sampling_out()[j].sample_index != j) sampling_ok = false;
            if (h.rows()[j] != j) sampling_ok = false;
        }
        check(sampling_ok, "VERIFY 32 output rows and sampling indices");
    }

    h.close();

    if (g_failed != 0) {
        std::printf("test_gpu_mcu_batch_binding: failed=%d\n", g_failed);
        return 1;
    }
    std::printf("test_gpu_mcu_batch_binding: R3 PASS\n");
    return 0;
}
