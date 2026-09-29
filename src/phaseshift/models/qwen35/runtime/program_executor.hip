#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/models/qwen35/runtime/program_executor.h>
#include <phaseshift/models/qwen35/runtime/optimized_dispatch.h>
#include <phaseshift/models/qwen35/runtime/lm_head_proxy.h>
#include <phaseshift/models/qwen35/runtime/decode_perf_stats.h>
#include <phaseshift/models/qwen35/runtime/kv_calib_dump.h>
#include <phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/quantization/imatrix/imatrix_collector.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <vector>
namespace ps::qwen35::runtime {

using ps::Result;
using ps::Status;
using ::ps::runtime::StagingProgramRefs;
using ::ps::runtime::StagingSectionOffsets;
using ::ps::runtime::staging_section_offsets;
using ::ps::runtime::staging_bytes_for;

Status fill_staging_image(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& binding,
    const HostExecutionContext& ctx,
    ::ps::runtime::DeviceProgramView* prog_out,
    ::ps::runtime::DispatchBinding* binding_out,
    ::ps::runtime::DeviceValueBinding* values_out,
    StagingProgramRefs* refs_out) {
    const auto& plan = program;
    uint32_t n = 0;
    for (uint32_t i = 0; i < binding.input_count; ++i) {
        const ::ps::runtime::ValueId vid{binding.input_slots[i]};
        const auto* found = plan.find_value(vid);
        if (found == nullptr)
            return Status::invalid_argument("dispatch input value unbound", __FILE__, __LINE__);
        auto& d = values_out[n++];
        d.logical_value = vid.id;
        d.storage = found->storage;
        d.dtype = found->dtype;
        d.row_domain = found->row_domain;
        d.slot = found->slot;
        d.feature_count = found->feature_count;
        d.row_stride = found->row_stride;
        d.bytes = found->bytes;
        if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_INPUT) {
            if (found->slot >= ctx.external_input_count || ctx.external_inputs == nullptr)
                return Status::invalid_argument("external input slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_inputs[found->slot]);
        } else if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
            if (found->slot >= ctx.external_output_count || ctx.external_outputs == nullptr)
                return Status::invalid_argument("external output slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_outputs[found->slot]);
        } else {
            d.offset = found->offset;
        }
    }
    for (uint32_t i = 0; i < binding.output_count; ++i) {
        const ::ps::runtime::ValueId vid{binding.output_slots[i]};
        const auto* found = plan.find_value(vid);
        if (found == nullptr)
            return Status::invalid_argument("dispatch output value unbound", __FILE__, __LINE__);
        auto& d = values_out[n++];
        d.logical_value = vid.id;
        d.storage = found->storage;
        d.dtype = found->dtype;
        d.row_domain = found->row_domain;
        d.slot = found->slot;
        d.feature_count = found->feature_count;
        d.row_stride = found->row_stride;
        d.bytes = found->bytes;
        if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_INPUT) {
            if (found->slot >= ctx.external_input_count || ctx.external_inputs == nullptr)
                return Status::invalid_argument("external input slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_inputs[found->slot]);
        } else if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
            if (found->slot >= ctx.external_output_count || ctx.external_outputs == nullptr)
                return Status::invalid_argument("external output slot out of range", __FILE__, __LINE__);
            d.offset = reinterpret_cast<uint64_t>(ctx.external_outputs[found->slot]);
        } else {
            d.offset = found->offset;
        }
    }
    if (refs_out != nullptr) {
        const ProgramStagingMeta* meta = ctx.program_meta;
        refs_out->ranges = (meta != nullptr) ? meta->ranges : nullptr;
        refs_out->range_count =
            (meta != nullptr) ? meta->range_count
                              : static_cast<uint32_t>(plan.workspace.ranges.size());
        refs_out->states = (meta != nullptr) ? meta->states : nullptr;
        refs_out->state_count =
            (meta != nullptr) ? meta->state_count
                              : static_cast<uint32_t>(plan.states.size());
    }
    *binding_out = binding;
    prog_out->values = static_cast<const ::ps::runtime::DeviceValueBinding*>(nullptr);
    prog_out->value_count = n;
    prog_out->weight_table = ctx.weights;
    prog_out->weight_slot_count = ctx.weight_count;
    prog_out->parameter_table = ctx.parameters;
    prog_out->parameter_slot_count = ctx.parameter_count;
    prog_out->workspace_ranges = static_cast<const ::ps::runtime::DeviceWorkspaceRange*>(nullptr);
    prog_out->workspace_range_count = static_cast<uint32_t>(plan.workspace.ranges.size());
    prog_out->states = static_cast<const ::ps::runtime::DeviceStateBinding*>(nullptr);
    prog_out->state_count = static_cast<uint32_t>(plan.states.size());
    prog_out->workspace_base = reinterpret_cast<uint64_t>(ctx.workspace);
    return Status::make_ok();
}

Status validate_host_weight_slot(const ::ps::runtime::WeightSlot& s, size_t index) {
    const uint8_t enc = s.encoding;
    if (s.rows == 0 || s.cols == 0) {
        return Status::invalid_argument(
            ("weight table slot has zero rows/cols: index " + std::to_string(index)).c_str(),
            __FILE__, __LINE__);
    }
    if (enc == 0u) {
        if (s.device_ptr == nullptr || s.codes != nullptr || s.scales != nullptr) {
            return Status::invalid_argument(
                ("bf16 weight table slot needs device_ptr and no codes/scales: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 2u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("rocm fp8 weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded < s.cols || (s.k_padded & 31u) != 0u) {
            return Status::invalid_argument(
                ("rocm fp8 weight table slot k_padded invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 1u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("rocm fp4 weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded < s.cols || (s.k_padded & 31u) != 0u) {
            return Status::invalid_argument(
                ("rocm fp4 weight table slot k_padded invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 3u || enc == 4u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("psq weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        if (s.k_padded < s.cols || (s.k_padded & 31u) != 0u) {
            return Status::invalid_argument(
                ("psq weight table slot k_padded invalid: index " + std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (enc == 5u || enc == 6u) {
        if (s.device_ptr != nullptr || s.codes == nullptr || s.scales == nullptr) {
            return Status::invalid_argument(
                ("block scaled weight table slot needs codes+scales and no device_ptr: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        const uint32_t k_align = (enc == 5u) ? 127u : 31u;
        if (s.k_padded < s.cols || (s.k_padded & k_align) != 0u) {
            return Status::invalid_argument(
                ("block scaled weight table slot k_padded invalid: index " +
                 std::to_string(index))
                    .c_str(),
                __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    return Status::invalid_argument(
        ("unsupported weight table encoding: index " + std::to_string(index)).c_str(),
        __FILE__, __LINE__);
}


using ::ps::Result;
using ::ps::Status;
using ::ps::runtime::WeightSlot;
using ::ps::runtime::StaticParameterSlot;

Result<HostResolvedValue> resolve_host_value(
    const ::ps::runtime::Program& program,
    ::ps::runtime::ValueId id,
    const HostExecutionContext& ctx) {
    const auto* found = program.find_value(id);
    if (found == nullptr)
        return Status::invalid_argument("value unbound", __FILE__, __LINE__);
    HostResolvedValue out{};
    out.row_stride = found->row_stride;
    out.feature_count = found->feature_count;
    out.dtype = found->dtype;
    out.row_domain = found->row_domain;
    if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_INPUT) {
        if (found->slot >= ctx.external_input_count || ctx.external_inputs == nullptr)
            return Status::invalid_argument("external input slot out of range", __FILE__, __LINE__);
        out.ptr = const_cast<void*>(ctx.external_inputs[found->slot]);
        return out;
    }
    if (found->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
        if (found->slot >= ctx.external_output_count || ctx.external_outputs == nullptr)
            return Status::invalid_argument("external output slot out of range", __FILE__, __LINE__);
        out.ptr = ctx.external_outputs[found->slot];
        return out;
    }
    if (ctx.workspace == nullptr)
        return Status::invalid_state("workspace not bound", __FILE__, __LINE__);
    out.ptr = ctx.workspace + found->offset;
    return out;
}

namespace {
const bool g_op_dump_enabled = [] {
    const char* e = std::getenv("PHASESHIFT_OP_DUMP");
    return e != nullptr && e[0] != '\0';
}();
const bool g_op_sync_enabled = [] {
    const char* e = std::getenv("PHASESHIFT_OP_SYNC");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}();
struct OpDumpTimer {
    std::chrono::steady_clock::time_point t0;
    const ::ps::runtime::KernelId id;
    explicit OpDumpTimer(::ps::runtime::KernelId k) : t0(std::chrono::steady_clock::now()), id(k) {}
    ~OpDumpTimer() {
        if (!g_op_dump_enabled) return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "OPDUMP %s %.4f\n", ::ps::runtime::kernel_to_name(id), ms);
    }
};
}

static int bucket_to_index(::ps::runtime::RowBucket bucket) {
    switch (bucket) {
        case ::ps::runtime::RowBucket::R16: return 0;
        case ::ps::runtime::RowBucket::R32: return 1;
        case ::ps::runtime::RowBucket::R64: return 2;
        case ::ps::runtime::RowBucket::R128: return 3;
        case ::ps::runtime::RowBucket::R256: return 4;
        case ::ps::runtime::RowBucket::R512: return 5;
        case ::ps::runtime::RowBucket::R1024: return 6;
        case ::ps::runtime::RowBucket::R2048: return 7;
    }
    return -1;
}

Status launch_host_binding(
    const ::ps::runtime::DispatchBinding& binding,
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    const bool sync_each_op = g_op_dump_enabled || g_op_sync_enabled;
    if (ctx.batch_context == nullptr)
        return Status::invalid_state("batch context not bound", __FILE__, __LINE__);
    if (ctx.scratch == nullptr || ctx.scratch_floats < 2048)
        return Status::invalid_state("dispatch scratch not bound", __FILE__, __LINE__);
    if (ctx.error_word == nullptr)
        return Status::invalid_state("error word not bound", __FILE__, __LINE__);
    const uint64_t nval = static_cast<uint64_t>(binding.input_count + binding.output_count);
    const auto seco = staging_section_offsets(nval);
#ifdef PHASESHIFT_HIP_GRAPH
    bool stream_capturing = false;
    {
        hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &cap_status) == hipSuccess)
            stream_capturing = (cap_status == hipStreamCaptureStatusActive);
    }
#endif
    uint8_t* dev_base = nullptr;
    bool via_pool = false;
    if (ctx.staging_pool != nullptr && ctx.staging_pool->device != nullptr) {
        const int bidx = bucket_to_index(program.row_bucket);
        const uint32_t bindx =
            static_cast<uint32_t>(&binding - program.dispatches.data());
        if (bidx >= 0 && bindx < program.dispatches.size()) {
            const uint64_t slot = ctx.staging_pool->slot(static_cast<uint32_t>(bidx), bindx);
            if (slot < ctx.staging_pool->bytes.size() && ctx.staging_pool->bytes[slot] > 0) {
                dev_base = static_cast<uint8_t*>(ctx.staging_pool->device)
                           + ctx.staging_pool->offsets[slot];
                via_pool = true;
                if (slot < ctx.staging_pool->run_seen.size())
                    ctx.staging_pool->run_seen[slot] = 1;
                if (!ctx.staging_pool->uploaded[slot]) {
#ifdef PHASESHIFT_HIP_GRAPH
                    if (stream_capturing)
                        return Status::invalid_state(
                            "staging pool upload during capture", __FILE__, __LINE__);
#endif
                    if (ctx.host_staging == nullptr || ctx.staging_bytes < ctx.staging_pool->bytes[slot])
                        return Status::invalid_state(
                            "dispatch staging not bound for pool upload", __FILE__, __LINE__);
                    auto* host_base = static_cast<uint8_t*>(ctx.host_staging);
                    auto* binding_host = reinterpret_cast<::ps::runtime::DispatchBinding*>(host_base);
                    auto* values_host = reinterpret_cast<::ps::runtime::DeviceValueBinding*>(host_base + seco.values);
                    auto* refs_host = reinterpret_cast<StagingProgramRefs*>(host_base + seco.refs);
                    ::ps::runtime::DeviceProgramView prog_host{};
                    Status fst = fill_staging_image(
                        program, binding, ctx, &prog_host, binding_host, values_host,
                        refs_host);
                    if (!fst.ok()) return fst;
                    hipError_t uherr = hipMemcpyAsync(dev_base, ctx.host_staging,
                                                      ctx.staging_pool->bytes[slot],
                                                      hipMemcpyHostToDevice, stream);
                    if (uherr != hipSuccess)
                        return Status::hip_error("hipMemcpyAsync staging pool",
                                                 hipGetErrorString(uherr), __FILE__, __LINE__);
                    uherr = hipStreamSynchronize(stream);
                    if (uherr != hipSuccess)
                        return Status::hip_error("staging pool upload sync",
                                                 hipGetErrorString(uherr), __FILE__, __LINE__);
                    ctx.staging_pool->uploaded[slot] = 1;
                }
            }
        }
    }
    if (dev_base == nullptr) {
#ifdef PHASESHIFT_HIP_GRAPH
        if (stream_capturing)
            return Status::invalid_state(
                "non-pool dispatch staging during capture", __FILE__, __LINE__);
#endif
        if (ctx.staging == nullptr || ctx.host_staging == nullptr || ctx.staging_bytes == 0)
            return Status::invalid_state("dispatch staging not bound", __FILE__, __LINE__);
        const uint64_t need = staging_bytes_for(binding);
        if (need > ctx.staging_bytes)
            return Status::insufficient_memory("dispatch staging too small", __FILE__, __LINE__);
        auto* host_base = static_cast<uint8_t*>(ctx.host_staging);
        auto* binding_host =
            reinterpret_cast<::ps::runtime::DispatchBinding*>(host_base);
        auto* values_host = reinterpret_cast<::ps::runtime::DeviceValueBinding*>(host_base + seco.values);
        auto* refs_host = reinterpret_cast<StagingProgramRefs*>(host_base + seco.refs);
        ::ps::runtime::DeviceProgramView prog_host{};
        Status fst = fill_staging_image(
            program, binding, ctx, &prog_host, binding_host, values_host, refs_host);
        if (!fst.ok()) return fst;
        hipError_t herr = hipMemcpyAsync(ctx.staging, ctx.host_staging, need,
                                         hipMemcpyHostToDevice, stream);
        if (herr != hipSuccess)
            return Status::hip_error("hipMemcpyAsync dispatch staging",
                                     hipGetErrorString(herr), __FILE__, __LINE__);
        dev_base = static_cast<uint8_t*>(ctx.staging);
    }
    const auto* refs_dev = reinterpret_cast<const StagingProgramRefs*>(dev_base + seco.refs);
    ::ps::runtime::DeviceProgramView prog_dev{};
    prog_dev.values = reinterpret_cast<const ::ps::runtime::DeviceValueBinding*>(dev_base + seco.values);
    prog_dev.value_count = static_cast<uint32_t>(nval);
    prog_dev.weight_table = ctx.weights;
    prog_dev.weight_slot_count = ctx.weight_count;
    prog_dev.parameter_table = ctx.parameters;
    prog_dev.parameter_slot_count = ctx.parameter_count;
    prog_dev.workspace_ranges =
        reinterpret_cast<const ::ps::runtime::DeviceWorkspaceRange*>(refs_dev->ranges);
    prog_dev.workspace_range_count = refs_dev->range_count;
    prog_dev.states =
        reinterpret_cast<const ::ps::runtime::DeviceStateBinding*>(refs_dev->states);
    prog_dev.state_count = refs_dev->state_count;
    prog_dev.workspace_base = reinterpret_cast<uint64_t>(ctx.workspace);
    Status lstat = ::ps::kernel::launch_model_dispatch_correctness(
        prog_dev, binding, ctx.batch_context, ctx.model_state, ctx.scratch,
        ctx.error_word, stream);
    if (!lstat.ok()) return lstat;
    hipError_t herr = hipSuccess;
    if (!via_pool || sync_each_op ||
        binding.kernel_id == ::ps::runtime::KernelId::EMBEDDING_LOOKUP) {
#ifdef PHASESHIFT_HIP_GRAPH
        if (stream_capturing && via_pool)
            return Status::make_ok();
#endif
        herr = hipStreamSynchronize(stream);
        if (herr != hipSuccess)
            return Status::hip_error("host_dispatch_kernel sync",
                                     hipGetErrorString(herr), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

namespace {

uint32_t probe_rows(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ::ps::runtime::ValueRowDomain::OUTPUT_ROWS) return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

::ps::quantization::imatrix::ImatrixDType probe_dtype(::ps::runtime::ValueDType dtype) {
    return dtype == ::ps::runtime::ValueDType::F32
               ? ::ps::quantization::imatrix::ImatrixDType::F32
               : ::ps::quantization::imatrix::ImatrixDType::BF16;
}

Status record_imatrix_probes(const ::ps::runtime::Program& program, std::size_t first,
                             uint32_t count, const HostExecutionContext& ctx,
                             hipStream_t stream) {
    auto* collector = ctx.imatrix_collector;
    if (collector == nullptr) return Status::make_ok();
    for (uint32_t j = 0; j < count; ++j) {
        const auto& binding = program.dispatches[first + j];
        for (uint32_t o = 0; o < binding.output_count; ++o) {
            const uint32_t vid = binding.output_slots[o];
            const auto* probe = program.find_imatrix_probe(vid);
            if (probe == nullptr || probe->tag == 0) continue;
            auto resolved = resolve_host_value(program, ::ps::runtime::ValueId{vid}, ctx);
            if (!resolved.ok()) return resolved.status();
            const HostResolvedValue v = resolved.value();
            if (v.ptr == nullptr || v.feature_count == 0) continue;
            Status st = collector->record(probe->tag, v.ptr, v.row_stride, v.feature_count,
                                          probe_rows(v, ctx), probe_dtype(v.dtype), stream);
            if (!st.ok()) return st;
        }
    }
    return Status::make_ok();
}

Status record_value_trace(const ::ps::runtime::Program& program, std::size_t first,
                          uint32_t count, const HostExecutionContext& ctx,
                          hipStream_t stream) {
    auto* sink = ctx.value_trace;
    if (sink == nullptr || sink->on_value == nullptr) return Status::make_ok();
    for (uint32_t j = 0; j < count; ++j) {
        const auto& binding = program.dispatches[first + j];
        for (uint32_t o = 0; o < binding.output_count; ++o) {
            const uint32_t vid = binding.output_slots[o];
            auto resolved = resolve_host_value(program, ::ps::runtime::ValueId{vid}, ctx);
            if (!resolved.ok()) return resolved.status();
            const HostResolvedValue v = resolved.value();
            if (v.ptr == nullptr || v.feature_count == 0) continue;
            sink->on_value(vid, v.ptr, v.row_stride, v.feature_count, v.dtype,
                           probe_rows(v, ctx), stream);
        }
    }
    return Status::make_ok();
}

Status launch_communication(const ::ps::runtime::Program& program, std::size_t index,
                            const HostExecutionContext& ctx, hipStream_t stream) {
    if (ctx.comm_launch == nullptr)
        return Status::invalid_state("communication node without a transport", __FILE__,
                                     __LINE__);
    if (index >= program.dispatch_comm.size())
        return Status::invalid_state("communication metadata missing", __FILE__, __LINE__);
    const uint32_t ci = program.dispatch_comm[index];
    if (ci == ::ps::runtime::kNoComm)
        return Status::invalid_state("communication dispatch lacks descriptor", __FILE__,
                                     __LINE__);
    const ::ps::runtime::CommDescriptor& descriptor = program.comms[ci];
    const ::ps::runtime::CommOperation operation = descriptor.operation;
    const void* send_ptr = nullptr;
    void* recv_ptr = nullptr;
    uint64_t elements = 0;

    auto resolve = [&](::ps::runtime::ValueId id, bool is_input) -> Status {
        auto resolved = resolve_host_value(program, id, ctx);
        if (!resolved.ok()) return resolved.status();
        const HostResolvedValue value = resolved.value();
        if (value.ptr == nullptr)
            return Status::invalid_state("communication buffer is null", __FILE__, __LINE__);
        if (value.dtype != descriptor.dtype)
            return Status::invalid_state("communication dtype mismatch", __FILE__, __LINE__);
        const uint32_t rows = probe_rows(value, ctx);
        const uint64_t count = static_cast<uint64_t>(rows) * value.row_stride;
        if (count == 0)
            return Status::invalid_argument("communication count is zero", __FILE__, __LINE__);
        if (elements == 0) {
            elements = count;
        } else if (elements != count) {
            return Status::invalid_state("communication count mismatch", __FILE__, __LINE__);
        }
        if (is_input)
            send_ptr = value.ptr;
        else
            recv_ptr = value.ptr;
        return Status::make_ok();
    };

    if (operation != ::ps::runtime::CommOperation::Recv) {
        Status st = resolve(descriptor.input, true);
        if (!st.ok()) return st;
    }
    if (operation != ::ps::runtime::CommOperation::Send) {
        Status st = resolve(descriptor.output, false);
        if (!st.ok()) return st;
    }
#ifdef PHASESHIFT_HIP_GRAPH
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    hipError_t capture_error = hipStreamIsCapturing(stream, &capture_status);
    if (capture_error != hipSuccess)
        return Status::hip_error("hipStreamIsCapturing", hipGetErrorString(capture_error),
                                 __FILE__, __LINE__);
    if (capture_status == hipStreamCaptureStatusActive)
        return Status::invalid_state("communication node during graph capture", __FILE__,
                                     __LINE__);
#endif
    return ctx.comm_launch(ctx.comm_self, descriptor, send_ptr, recv_ptr, elements, stream);
}

}

Status execute_program_range(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream,
    size_t begin,
    size_t end) {
    hipError_t herr = hipSuccess;
    auto mode_result = read_qwen35_kernel_mode();
    if (!mode_result.ok()) return mode_result.status();
    const Qwen35KernelMode mode = mode_result.value();
    const bool sync_each_op = g_op_dump_enabled || g_op_sync_enabled;
    size_t i = begin;
    while (i < end) {
        const auto& binding = program.dispatches[i];
        if (::ps::runtime::is_comm_kernel(binding.kernel_id)) {
            Status comm_status = launch_communication(program, i, ctx, stream);
            if (!comm_status.ok()) return comm_status;
            const uint32_t comm_kid = static_cast<uint32_t>(binding.kernel_id);
            record_logical_covered(&comm_kid, 1);
            Status trace_status = record_value_trace(program, i, 1, ctx, stream);
            if (!trace_status.ok()) return trace_status;
            ++i;
            continue;
        }
        kv_calib_dump_maybe(program, i, ctx, stream);
        OpDumpTimer op_timer(binding.kernel_id);
        const bool sem_on = semantic_timing_enabled();
        const uint32_t sem_seq = sem_on ? semantic_timing_start(stream) : 0xFFFFFFFFu;
        if (mode == Qwen35KernelMode::Auto) {
            OptimizedLaunchResult opt_result{};
            auto lm_proxy = try_launch_lm_head_proxy_fusion(program, i, ctx, stream);
            if (!lm_proxy.ok()) return lm_proxy.status();
            opt_result = lm_proxy.value();
            if (opt_result.kind != OptimizedLaunchKind::Launched) {
                auto optimized = try_launch_optimized(program, i, ctx, stream);
                if (!optimized.ok()) return optimized.status();
                opt_result = optimized.value();
            }
            const OptimizedLaunchKind opt_kind = opt_result.kind;
            if (opt_kind == OptimizedLaunchKind::Skipped) {
                const uint32_t consumed = opt_result.consumed_dispatches;
                if (consumed == 0 || i + consumed > end)
                    return Status::invalid_state(
                        "invalid skipped dispatch consumption", __FILE__, __LINE__);
                uint32_t covered_kids[4] = {0, 0, 0, 0};
                for (uint32_t j = 0; j < consumed && j < 4u; ++j)
                    covered_kids[j] = static_cast<uint32_t>(program.dispatches[i + j].kernel_id);
                record_logical_covered(covered_kids, consumed < 4u ? consumed : 4u);
                i += consumed;
                continue;
            }
            if (opt_kind == OptimizedLaunchKind::Launched) {
                const uint32_t consumed = opt_result.consumed_dispatches;
                if (consumed == 0 || i + consumed > end)
                    return Status::invalid_state(
                        "invalid optimized dispatch consumption", __FILE__, __LINE__);
                if (sync_each_op) {
                    herr = hipStreamSynchronize(stream);
                    if (herr != hipSuccess)
                        return Status::hip_error(
                            "optimized dispatch sync", hipGetErrorString(herr),
                            __FILE__, __LINE__);
                }
                uint32_t covered_kids[4] = {0, 0, 0, 0};
                for (uint32_t j = 0; j < consumed && j < 4u; ++j)
                    covered_kids[j] = static_cast<uint32_t>(program.dispatches[i + j].kernel_id);
                record_logical_covered(covered_kids, consumed < 4u ? consumed : 4u);
                if (sem_on)
                    semantic_timing_finish(stream, sem_seq, static_cast<uint32_t>(binding.kernel_id),
                                           static_cast<uint32_t>(SemPath::Optimized),
                                           consumed);
                {
                    Status pst = record_imatrix_probes(program, i, consumed, ctx, stream);
                    if (!pst.ok()) return pst;
                    Status tst = record_value_trace(program, i, consumed, ctx, stream);
                    if (!tst.ok()) return tst;
                }
                i += consumed;
                continue;
            }
        }
        Status st = launch_host_binding(binding, program, ctx, stream);
        if (!st.ok()) return st;
        record_correctness_fallback(binding.kernel_id);
        record_physical_launch(1);
        const uint32_t single_kid = static_cast<uint32_t>(binding.kernel_id);
        record_logical_covered(&single_kid, 1);
        if (sem_on)
            semantic_timing_finish(stream, sem_seq, single_kid,
                                   static_cast<uint32_t>(SemPath::Correctness), 1);
        {
            Status pst = record_imatrix_probes(program, i, 1, ctx, stream);
            if (!pst.ok()) return pst;
            Status tst = record_value_trace(program, i, 1, ctx, stream);
            if (!tst.ok()) return tst;
        }
        ++i;
    }
    if (ctx.lm_head_shadow_pending) {
        Status shadow_status = Status::make_ok();
        if (ctx.lm_head_proxy != nullptr) {
            shadow_status = ctx.lm_head_proxy->compare_shadow(
                ctx.lm_head_shadow_full, ctx.lm_head_shadow_rows, stream);
        }
        ctx.lm_head_shadow_pending = false;
        ctx.lm_head_shadow_full = nullptr;
        ctx.lm_head_shadow_rows = 0;
        if (!shadow_status.ok())
            return shadow_status;
    }
#ifdef PHASESHIFT_HIP_GRAPH
    if (ctx.staging_pool != nullptr) {
        const int bidx = bucket_to_index(program.row_bucket);
        if (bidx >= 0 &&
            static_cast<std::size_t>(bidx) < ctx.staging_pool->bucket_ran.size())
            ctx.staging_pool->bucket_ran[static_cast<std::size_t>(bidx)] = 1;
    }
#endif
    return Status::make_ok();
}

Status execute_program(
    const ::ps::runtime::Program& program,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (ctx.error_word == nullptr)
        return Status::invalid_state("error word not bound", __FILE__, __LINE__);
    const hipError_t herr =
        hipMemsetAsync(ctx.error_word, 0, sizeof(uint32_t), stream);
    if (herr != hipSuccess)
        return Status::hip_error("hipMemsetAsync error word",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    return execute_program_range(program, ctx, stream, 0u,
                                 program.dispatches.size());
}

Status upload_host_weight_table(
    const ::ps::runtime::WeightTableView& weights,
    WeightSlot** out,
    hipStream_t stream) {
    if (out == nullptr)
        return Status::invalid_argument("weight table out is null", __FILE__, __LINE__);
    *out = nullptr;
    std::vector<WeightSlot> host(weights.count);
    for (size_t i = 0; i < weights.count; ++i) {
        auto s = weights.at(i);
        Status vst = validate_host_weight_slot(s, i);
        if (!vst.ok()) return vst;
        host[i].device_ptr = s.device_ptr;
        host[i].rows = s.rows;
        host[i].cols = s.cols;
        host[i].k_padded = s.k_padded;
        host[i].encoding = s.encoding;
        host[i].codes = s.codes;
        host[i].scales = s.scales;
        host[i].storage_scale_stride_bytes = s.storage_scale_stride_bytes;
        host[i].compute_codes = s.compute_codes;
        host[i].compute_scales_bf16 = s.compute_scales_bf16;
        host[i].codes_row_stride_bytes = s.codes_row_stride_bytes;
        host[i].scale_row_stride_bytes = s.scale_row_stride_bytes;
        host[i].weight_scale_group = s.weight_scale_group;
        host[i].preshuffled = s.preshuffled;
    }
    WeightSlot* dev = nullptr;
    hipError_t err =
        hipMalloc(reinterpret_cast<void**>(&dev), sizeof(WeightSlot) * host.size());
    if (err != hipSuccess)
        return Status::hip_error("hipMalloc weight table", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    err = hipMemcpyWithStream(dev, host.data(), sizeof(WeightSlot) * host.size(),
                              hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipFree(dev));
        return Status::hip_error("hipMemcpy weight table", hipGetErrorString(err), __FILE__,
                                  __LINE__);
    }
    *out = dev;
    return Status::make_ok();
}

Status upload_host_parameter_table(
    const ::ps::runtime::StaticParameterTableView& params,
    ::ps::runtime::StaticParameterSlot** out,
    hipStream_t stream) {
    if (out == nullptr)
        return Status::invalid_argument("parameter table out is null", __FILE__, __LINE__);
    *out = nullptr;
    std::vector<::ps::runtime::StaticParameterSlot> host(params.count);
    for (size_t i = 0; i < params.count; ++i) {
        host[i] = params.slots[i];
    }
    ::ps::runtime::StaticParameterSlot* dev = nullptr;
    const size_t bytes =
        sizeof(::ps::runtime::StaticParameterSlot) * host.size();
    hipError_t err = hipMalloc(reinterpret_cast<void**>(&dev), bytes);
    if (err != hipSuccess)
        return Status::hip_error("hipMalloc parameter table", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    err = hipMemcpyWithStream(dev, host.data(), bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        ps::gpu::discard_cleanup_result(hipFree(dev));
        return Status::hip_error("hipMemcpy parameter table", hipGetErrorString(err), __FILE__,
                                  __LINE__);
    }
    *out = dev;
    return Status::make_ok();
}

}
