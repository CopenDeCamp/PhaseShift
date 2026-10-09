#pragma once

#include <phaseshift/runtime/gpu_mcu/infrastructure/aql.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/completion.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/device_completion.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/fsm_worker.h>
#include <phaseshift/models/qwen35/kernels/optimized/gdn/reset.h>
#include <phaseshift/models/qwen35/runtime/gpu_mcu/invocation_abi.h>
#include <phaseshift/models/qwen35/runtime/gpu_mcu/kernarg_recipe.h>
#include <phaseshift/runtime/gpu_mcu/execution/micro_fsm.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/retained_packet.h>
#include <phaseshift/runtime/gpu_mcu/infrastructure/worker_image.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

namespace gpu_mcu_test {

namespace mcu = ps::runtime::gpu_mcu;

inline std::size_t align_up(std::size_t v, std::size_t a) {
    return a <= 1 ? v : (v + a - 1) / a * a;
}

struct Chain {
    std::vector<mcu::McuPlanNode> nodes;
    std::vector<uint32_t> dispatch_pc;
    std::vector<uint32_t> values;
};

inline Chain make_chain(uint32_t dispatch_count,
                        uint32_t work = 0u,
                        uint32_t value_base = 100u) {
    Chain out;
    uint32_t previous_pc = mcu::kMcuNoInput;
    for (uint32_t i = 0; i < dispatch_count; ++i) {
        const uint32_t dispatch_pc = static_cast<uint32_t>(out.nodes.size());
        mcu::McuPlanNode dispatch{};
        dispatch.variant_id = 0;
        dispatch.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
        dispatch.completion_slot = static_cast<uint16_t>(i);
        dispatch.value = value_base + i;
        dispatch.element_count = 1;
        dispatch.input_slot = static_cast<uint16_t>(previous_pc);
        dispatch.work = work;
        dispatch.flags = mcu::kMcuNodeDispatch;
        dispatch.next = static_cast<uint16_t>(out.nodes.size() + 1u);
        out.nodes.push_back(dispatch);
        out.dispatch_pc.push_back(dispatch_pc);
        out.values.push_back(dispatch.value);

        mcu::McuPlanNode wait{};
        wait.completion_slot = static_cast<uint16_t>(i);
        wait.flags = mcu::kMcuNodeWait;
        wait.next = static_cast<uint16_t>(out.nodes.size() + 1u);
        out.nodes.push_back(wait);

        previous_pc = dispatch_pc;
    }
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    out.nodes.push_back(end);
    return out;
}

inline Chain make_feed_chain(uint32_t dispatch_count,
                             uint32_t work = 0u,
                             uint32_t value_base = 1u) {
    Chain out;
    for (uint32_t i = 0; i < dispatch_count; ++i) {
        mcu::McuPlanNode dispatch{};
        dispatch.variant_id = 0;
        dispatch.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
        dispatch.completion_slot = 0;
        dispatch.value = value_base;
        dispatch.element_count = 1;
        dispatch.input_slot =
            static_cast<uint16_t>(i == 0u ? mcu::kMcuNoInput : i - 1u);
        dispatch.work = work;
        dispatch.flags = mcu::kMcuNodeDispatch;
        dispatch.next = static_cast<uint16_t>(i + 1u);
        out.nodes.push_back(dispatch);
        out.dispatch_pc.push_back(i);
        out.values.push_back(value_base);
    }
    mcu::McuPlanNode wait{};
    wait.completion_slot = 0;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = static_cast<uint16_t>(dispatch_count + 1u);
    out.nodes.push_back(wait);
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    out.nodes.push_back(end);
    return out;
}

inline Chain make_feed_chain_with_marker(uint32_t data_count,
                                         uint32_t work = 0u,
                                         uint32_t value_base = 1u,
                                         uint16_t marker_variant = 1u) {
    Chain out;
    for (uint32_t i = 0; i < data_count; ++i) {
        mcu::McuPlanNode dispatch{};
        dispatch.variant_id = 0;
        dispatch.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
        dispatch.completion_slot = 0;
        dispatch.value = value_base;
        dispatch.element_count = 1;
        dispatch.input_slot =
            static_cast<uint16_t>(i == 0u ? mcu::kMcuNoInput : i - 1u);
        dispatch.work = work;
        dispatch.flags = mcu::kMcuNodeDispatch;
        dispatch.next = static_cast<uint16_t>(out.nodes.size() + 1u);
        out.nodes.push_back(dispatch);
        out.dispatch_pc.push_back(i);
        out.values.push_back(value_base);
    }
    mcu::McuPlanNode marker{};
    marker.variant_id = marker_variant;
    marker.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
    marker.completion_slot = 0;
    marker.input_slot = mcu::kMcuNoInput;
    marker.flags = mcu::kMcuNodeDispatch;
    marker.next = static_cast<uint16_t>(out.nodes.size() + 1u);
    out.nodes.push_back(marker);
    mcu::McuPlanNode wait{};
    wait.completion_slot = 0;
    wait.flags = mcu::kMcuNodeWait;
    wait.next = static_cast<uint16_t>(out.nodes.size() + 1u);
    out.nodes.push_back(wait);
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    out.nodes.push_back(end);
    return out;
}

inline Chain make_feed_chain_no_wait(uint32_t dispatch_count,
                                     uint32_t work = 0u,
                                     uint32_t value_base = 1u) {
    Chain out;
    for (uint32_t i = 0; i < dispatch_count; ++i) {
        mcu::McuPlanNode dispatch{};
        dispatch.variant_id = 0;
        dispatch.kernarg_recipe = mcu::kMcuKernargRecipeProbe;
        dispatch.completion_slot = 0;
        dispatch.value = value_base;
        dispatch.element_count = 1;
        dispatch.input_slot =
            static_cast<uint16_t>(i == 0u ? mcu::kMcuNoInput : i - 1u);
        dispatch.work = work;
        dispatch.flags = mcu::kMcuNodeDispatch;
        dispatch.next = static_cast<uint16_t>(i + 1u);
        out.nodes.push_back(dispatch);
        out.dispatch_pc.push_back(i);
        out.values.push_back(value_base);
    }
    mcu::McuPlanNode end{};
    end.flags = mcu::kMcuNodeEnd;
    out.nodes.push_back(end);
    return out;
}

inline std::vector<uint32_t> chain_expected(const Chain& chain) {
    std::vector<uint32_t> expected(chain.dispatch_pc.size(), 0u);
    uint32_t accumulated = 0u;
    for (std::size_t i = 0; i < chain.values.size(); ++i) {
        accumulated += chain.values[i];
        expected[i] = accumulated;
    }
    return expected;
}

struct Rig {
    mcu::GpuMcuCuPartition partition;
    mcu::GpuMcuAqlQueue queue;
    mcu::GpuMcuAqlCodeObject code;
    std::vector<mcu::GpuMcuAqlCodeObject> extra_codes;
    mcu::GpuMcuKernargRegion region{};
    mcu::GpuMcuKernargRegion log_region{};
    bool log_enabled = false;
    mcu::GpuAqlKernelMetadata meta{};

    uint32_t* output = nullptr;
    uint64_t* timestamps = nullptr;
    mcu::GpuMcuDeviceCompletion* completions = nullptr;
    mcu::McuPlanNode* plan = nullptr;
    mcu::McuKernelVariantDesc* variants = nullptr;
    mcu::GpuMcuRetainedPacket* retained = nullptr;
    mcu::McuRuntimeNodeBinding* runtime_bindings = nullptr;
    uint32_t runtime_binding_count = 0;
    mcu::McuKernargSourceDesc* kernarg_sources = nullptr;
    uint32_t kernarg_source_count = 0;
    mcu::McuDispatchTiming* timing = nullptr;
    mcu::McuDispatchRecord* records = nullptr;
    mcu::McuRmsNormInvocation* invocations = nullptr;
    uint32_t invocation_count = 0;
    mcu::McuActivationQuantizeInvocation* quantize_invocations = nullptr;
    uint32_t quantize_count = 0;
    mcu::McuActivationQuantizeE4m3Invocation* e4m3_invocations = nullptr;
    uint32_t e4m3_count = 0;
    mcu::McuPsq4Decode1Invocation* psq4_invocations = nullptr;
    uint32_t psq4_count = 0;
    mcu::McuPsq4MultiRowInvocation* psq4_multi_invocations = nullptr;
    uint32_t psq4_multi_count = 0;
    mcu::McuVerifyAcceptPrefixInvocation* verify_accept_invocations = nullptr;
    uint32_t verify_accept_count = 0;
    mcu::McuVerifyAcceptBatchInvocation* verify_accept_batch_invocations = nullptr;
    uint32_t verify_accept_batch_count = 0;
    mcu::McuGdnSpecRestoreInvocation* gdn_spec_restore_invocations = nullptr;
    uint32_t gdn_spec_restore_count = 0;
    mcu::McuGdnSpecRestoreFromCountsInvocation*
        gdn_spec_restore_from_counts_invocations = nullptr;
    uint32_t gdn_spec_restore_from_counts_count = 0;
    mcu::McuArgmaxF32Invocation* argmax_f32_invocations = nullptr;
    uint32_t argmax_f32_count = 0;
    mcu::McuOutputGatherBf16Invocation* output_gather_invocations = nullptr;
    uint32_t output_gather_count = 0;
    mcu::McuElementwiseInvocation* elementwise_invocations = nullptr;
    uint32_t elementwise_count = 0;
    mcu::McuRopeInvocation* rope_invocations = nullptr;
    uint32_t rope_count = 0;
    mcu::McuKvAppendInvocation* kv_append_invocations = nullptr;
    uint32_t kv_append_count = 0;
    mcu::McuPagedAttentionInvocation* attention_paged_invocations = nullptr;
    uint32_t attention_paged_count = 0;
    mcu::McuPagedAttentionSplitInvocation* attention_paged_split_invocations = nullptr;
    uint32_t attention_paged_split_count = 0;
    mcu::McuPagedAttentionReduceInvocation* attention_paged_reduce_invocations =
        nullptr;
    uint32_t attention_paged_reduce_count = 0;
    mcu::McuBf16ExactRowsInvocation* bf16_invocations = nullptr;
    uint32_t bf16_count = 0;
    mcu::McuBf16WmmaInvocation* bf16_wmma_invocations = nullptr;
    uint32_t bf16_wmma_count = 0;
    mcu::McuL2NormalizeInvocation* l2_invocations = nullptr;
    uint32_t l2_count = 0;
    mcu::McuEmbeddingBf16Invocation* embedding_invocations = nullptr;
    uint32_t embedding_count = 0;
    mcu::McuEmbeddingPsq8Invocation* embedding_psq8_invocations = nullptr;
    uint32_t embedding_psq8_count = 0;
    mcu::McuGdnConv1dInvocation* gdn_conv1d_invocations = nullptr;
    uint32_t gdn_conv1d_count = 0;
    mcu::McuGdnRecurrenceInvocation* gdn_recurrence_invocations = nullptr;
    uint32_t gdn_recurrence_count = 0;
    mcu::GpuMcuGdnResetInvocation* gdn_reset_invocations = nullptr;
    uint32_t gdn_reset_count = 0;

    mcu::GpuMcuFsm fsm;
    uint32_t node_count = 0;
    uint32_t variant_count = 0;
    uint32_t completion_cap = 0;
    uint32_t timing_cap = 0;
    uint32_t record_cap = 0;
    bool fsm_started = false;
    bool fsm_configured = false;
    void* coherent_kernarg_base_ = nullptr;
    std::vector<mcu::McuPlanNode> host_plan;
    std::vector<mcu::McuKernelVariantDesc> host_variants;

    static bool check(bool condition, const char* message) {
        if (!condition) std::printf("FAIL: %s\n", message);
        return condition;
    }

    bool open(int device,
              uint32_t node_cap,
              uint32_t variant_cap,
              uint32_t completion_slots,
              uint32_t timing_entries,
              bool coherent_kernarg = false,
              uint32_t queue_size = mcu::kAqlDefaultQueueSize,
              uint32_t record_entries = 0u,
              uint32_t min_kernarg_segment = 0u) {
        node_count = node_cap;
        variant_count = variant_cap;
        completion_cap = completion_slots;
        timing_cap = timing_entries;
        record_cap = record_entries;

        auto part = mcu::GpuMcuCuPartition::create(device, 1u);
        if (!check(part.ok(), "partition created")) return false;
        partition = part.release();

        auto q = mcu::GpuMcuAqlQueue::create_with_mask(
            device, mcu::AqlQueueOwner::GpuMcu, queue_size,
            partition.hsa_worker_cu_mask_view());
        if (!check(q.ok(), "worker queue created")) return false;
        queue = q.release();

        const mcu::GpuMcuWorkerImage image = mcu::gpu_mcu_worker_image();
        auto c = mcu::GpuMcuAqlCodeObject::load_memory(
            queue, image.data, image.bytes, mcu::kGpuMcuFsmWorkerSymbol);
        if (!check(c.ok(), "fsm worker code loaded")) return false;
        code = c.release();
        meta = code.metadata();

        if (hipMalloc(reinterpret_cast<void**>(&output),
                      static_cast<std::size_t>(node_cap) * sizeof(uint32_t)) !=
                hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&timestamps),
                      static_cast<std::size_t>(node_cap) * 2u * sizeof(uint64_t)) !=
                hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&this->completions),
                      static_cast<std::size_t>(completion_slots) *
                          sizeof(mcu::GpuMcuDeviceCompletion)) != hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&plan),
                      static_cast<std::size_t>(node_cap) *
                          sizeof(mcu::McuPlanNode)) != hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&this->variants),
                      static_cast<std::size_t>(variant_cap) *
                          sizeof(mcu::McuKernelVariantDesc)) != hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&retained),
                      static_cast<std::size_t>(variant_cap) *
                          sizeof(mcu::GpuMcuRetainedPacket)) != hipSuccess) {
            check(false, "device tables allocated");
            return false;
        }
        if (timing_entries != 0u) {
            if (hipMalloc(reinterpret_cast<void**>(&timing),
                          static_cast<std::size_t>(timing_entries) *
                              sizeof(mcu::McuDispatchTiming)) != hipSuccess) {
                check(false, "timing table allocated");
                return false;
            }
        }
        if (record_entries != 0u) {
            if (hipMalloc(reinterpret_cast<void**>(&records),
                          static_cast<std::size_t>(record_entries) *
                              sizeof(mcu::McuDispatchRecord)) != hipSuccess) {
                check(false, "debug records allocated");
                return false;
            }
            (void)hipMemset(records, 0,
                            static_cast<std::size_t>(record_entries) *
                                sizeof(mcu::McuDispatchRecord));
        }
        (void)hipMemset(output, 0,
                        static_cast<std::size_t>(node_cap) * sizeof(uint32_t));
        (void)hipMemset(timestamps, 0,
                        static_cast<std::size_t>(node_cap) * 2u * sizeof(uint64_t));
        (void)hipMemset(completions, 0,
                        static_cast<std::size_t>(completion_slots) *
                            sizeof(mcu::GpuMcuDeviceCompletion));

        std::size_t slot_bytes = std::max(
            mcu::aql_kernarg_slot_stride(meta.kernarg_segment_size),
            mcu::aql_kernarg_slot_stride(sizeof(mcu::GpuMcuFsmWorkerArgs)));
        if (min_kernarg_segment != 0u)
            slot_bytes = std::max(
                slot_bytes, mcu::aql_kernarg_slot_stride(min_kernarg_segment));
        if (coherent_kernarg) {
            const std::size_t stride = align_up(slot_bytes, 64);
            void* base = nullptr;
            if (hipMalloc(&base, stride * node_cap) != hipSuccess) {
                check(false, "coherent kernarg allocated");
                return false;
            }
            region.base = static_cast<unsigned char*>(base);
            region.requested_slot_bytes = slot_bytes;
            region.granule = 16;
            region.actual_size = stride * node_cap;
            region.slot_stride = stride;
            region.slot_count = node_cap;
            coherent_kernarg_base_ = base;
        } else {
            auto r = queue.allocate_kernarg(slot_bytes, node_cap);
            if (!check(r.ok(), "kernarg region allocated")) return false;
            region = r.release();
        }
        return true;
    }

    bool upload(const std::vector<mcu::McuPlanNode>& nodes,
                const std::vector<mcu::McuKernelVariantDesc>& vars) {
        host_plan = nodes;
        host_variants = vars;
        node_count = static_cast<uint32_t>(nodes.size());
        variant_count = static_cast<uint32_t>(vars.size());
        if (!check(node_count <= mcu::kMcuMaxNodes, "node count within limit") ||
            !check(variant_count <= mcu::kMcuMaxVariants,
                   "variant count within limit")) {
            return false;
        }
        for (const auto& node : nodes) {
            if ((node.flags & mcu::kMcuNodeEnd) == 0u &&
                node.completion_slot >= completion_cap) {
                check(false, "node completion slot within capacity");
                return false;
            }
        }

        std::vector<mcu::GpuMcuRetainedPacket> host_retained(vars.size());
        for (std::size_t i = 0; i < vars.size(); ++i) {
            host_retained[i] = vars[i].packet;
        }
        if (hipMemcpy(plan, nodes.data(),
                      nodes.size() * sizeof(mcu::McuPlanNode),
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(variants, vars.data(),
                      vars.size() * sizeof(mcu::McuKernelVariantDesc),
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(retained, host_retained.data(),
                      host_retained.size() * sizeof(mcu::GpuMcuRetainedPacket),
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "tables uploaded");
        }

        for (uint32_t i = 0; i < node_count; ++i) {
            const mcu::McuKernelVariantDesc& v = vars[nodes[i].variant_id];
            unsigned char* slot = region.slot(i);
            std::memset(slot, 0, region.slot_stride);
            mcu::build_aql_launch_metadata(slot, meta.kernarg_segment_size,
                                           v.workgroup_count_x, v.workgroup_count_y, v.workgroup_count_z,
                                           v.workgroup_x);
        }
        return true;
    }

    void poison_kernarg_args() {
        for (uint32_t i = 0; i < node_count; ++i) {
            std::memset(region.slot(i), 0xAB, sizeof(mcu::GpuMcuFsmWorkerArgs));
        }
    }

    void poison_all_kernarg() {
        for (uint32_t i = 0; i < node_count; ++i) {
            std::memset(region.slot(i), 0xAB, region.slot_stride);
        }
    }

    bool load_extra_code(const char* hsaco_path, const char* symbol,
                         mcu::GpuAqlKernelMetadata& out) {
        auto res = mcu::GpuMcuAqlCodeObject::load(queue, hsaco_path, symbol);
        if (!check(res.ok(), "extra code object loaded")) return false;
        auto loaded = res.release();
        out = loaded.metadata();
        extra_codes.push_back(std::move(loaded));
        return true;
    }

    bool load_extra_code_memory(const void* code_object, std::size_t code_size,
                                const char* symbol,
                                mcu::GpuAqlKernelMetadata& out) {
        auto res = mcu::GpuMcuAqlCodeObject::load_memory(
            queue, code_object, code_size, symbol);
        if (!check(res.ok(), "extra code object loaded")) return false;
        auto loaded = res.release();
        out = loaded.metadata();
        extra_codes.push_back(std::move(loaded));
        return true;
    }

    bool set_output_gather_invocations(
        const mcu::McuOutputGatherBf16Invocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuOutputGatherBf16Invocation);
        if (hipMalloc(reinterpret_cast<void**>(&output_gather_invocations),
                      bytes) != hipSuccess) {
            return check(false, "output gather invocation table allocated");
        }
        output_gather_count = count;
        if (hipMemcpy(output_gather_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "output gather invocation table uploaded");
        }
        return true;
    }

    bool set_invocations(const mcu::McuRmsNormInvocation* source,
                         uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(mcu::McuRmsNormInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&invocations), bytes) !=
            hipSuccess) {
            return check(false, "invocation table allocated");
        }
        invocation_count = count;
        if (hipMemcpy(invocations, source, bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
            return check(false, "invocation table uploaded");
        }
        return true;
    }

    bool set_quantize_invocations(
        const mcu::McuActivationQuantizeInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuActivationQuantizeInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&quantize_invocations), bytes) !=
            hipSuccess) {
            return check(false, "quantize invocation table allocated");
        }
        quantize_count = count;
        if (hipMemcpy(quantize_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "quantize invocation table uploaded");
        }
        return true;
    }

    bool set_e4m3_invocations(const mcu::McuActivationQuantizeE4m3Invocation* source,
                              uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuActivationQuantizeE4m3Invocation);
        if (hipMalloc(reinterpret_cast<void**>(&e4m3_invocations), bytes) !=
            hipSuccess) {
            return check(false, "e4m3 invocation table allocated");
        }
        e4m3_count = count;
        if (hipMemcpy(e4m3_invocations, source, bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
            return check(false, "e4m3 invocation table uploaded");
        }
        return true;
    }

    bool set_psq4_invocations(const mcu::McuPsq4Decode1Invocation* source,
                              uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(mcu::McuPsq4Decode1Invocation);
        if (hipMalloc(reinterpret_cast<void**>(&psq4_invocations), bytes) !=
            hipSuccess) {
            return check(false, "psq4 invocation table allocated");
        }
        psq4_count = count;
        if (hipMemcpy(psq4_invocations, source, bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
            return check(false, "psq4 invocation table uploaded");
        }
        return true;
    }

    bool set_psq4_multi_invocations(const mcu::McuPsq4MultiRowInvocation* source,
                                    uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuPsq4MultiRowInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&psq4_multi_invocations), bytes) !=
            hipSuccess) {
            return check(false, "psq4 multi invocation table allocated");
        }
        psq4_multi_count = count;
        if (hipMemcpy(psq4_multi_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "psq4 multi invocation table uploaded");
        }
        return true;
    }

    bool set_elementwise_invocations(
        const mcu::McuElementwiseInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuElementwiseInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&elementwise_invocations), bytes) !=
            hipSuccess) {
            return check(false, "elementwise invocation table allocated");
        }
        elementwise_count = count;
        if (hipMemcpy(elementwise_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "elementwise invocation table uploaded");
        }
        return true;
    }

    bool set_rope_invocations(const mcu::McuRopeInvocation* source,
                              uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuRopeInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&rope_invocations), bytes) !=
            hipSuccess) {
            return check(false, "rope invocation table allocated");
        }
        rope_count = count;
        if (hipMemcpy(rope_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "rope invocation table uploaded");
        }
        return true;
    }

    bool set_kv_append_invocations(const mcu::McuKvAppendInvocation* source,
                                   uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuKvAppendInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&kv_append_invocations), bytes) !=
            hipSuccess) {
            return check(false, "kv append invocation table allocated");
        }
        kv_append_count = count;
        if (hipMemcpy(kv_append_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "kv append invocation table uploaded");
        }
        return true;
    }

    bool set_attention_paged_invocations(
        const mcu::McuPagedAttentionInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuPagedAttentionInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&attention_paged_invocations),
                      bytes) != hipSuccess) {
            return check(false, "paged attention invocation table allocated");
        }
        attention_paged_count = count;
        if (hipMemcpy(attention_paged_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "paged attention invocation table uploaded");
        }
        return true;
    }

    bool set_attention_paged_split_invocations(
        const mcu::McuPagedAttentionSplitInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuPagedAttentionSplitInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&attention_paged_split_invocations),
                      bytes) != hipSuccess) {
            return check(false, "paged attention split table allocated");
        }
        attention_paged_split_count = count;
        if (hipMemcpy(attention_paged_split_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "paged attention split table uploaded");
        }
        return true;
    }

    bool set_attention_paged_reduce_invocations(
        const mcu::McuPagedAttentionReduceInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuPagedAttentionReduceInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&attention_paged_reduce_invocations),
                      bytes) != hipSuccess) {
            return check(false, "paged attention reduce table allocated");
        }
        attention_paged_reduce_count = count;
        if (hipMemcpy(attention_paged_reduce_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "paged attention reduce table uploaded");
        }
        return true;
    }

    bool set_bf16_invocations(const mcu::McuBf16ExactRowsInvocation* source,
                              uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuBf16ExactRowsInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&bf16_invocations), bytes) !=
            hipSuccess) {
            return check(false, "bf16 invocation table allocated");
        }
        bf16_count = count;
        if (hipMemcpy(bf16_invocations, source, bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
            return check(false, "bf16 invocation table uploaded");
        }
        return true;
    }

    bool set_bf16_wmma_invocations(const mcu::McuBf16WmmaInvocation* source,
                                   uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(mcu::McuBf16WmmaInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&bf16_wmma_invocations), bytes) !=
            hipSuccess) {
            return check(false, "bf16 wmma invocation table allocated");
        }
        bf16_wmma_count = count;
        if (hipMemcpy(bf16_wmma_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "bf16 wmma invocation table uploaded");
        }
        return true;
    }

    bool set_l2_invocations(const mcu::McuL2NormalizeInvocation* source,
                            uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuL2NormalizeInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&l2_invocations), bytes) !=
            hipSuccess) {
            return check(false, "l2 invocation table allocated");
        }
        l2_count = count;
        if (hipMemcpy(l2_invocations, source, bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
            return check(false, "l2 invocation table uploaded");
        }
        return true;
    }

    bool set_embedding_invocations(
        const mcu::McuEmbeddingBf16Invocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuEmbeddingBf16Invocation);
        if (hipMalloc(reinterpret_cast<void**>(&embedding_invocations), bytes) !=
            hipSuccess) {
            return check(false, "embedding invocation table allocated");
        }
        embedding_count = count;
        if (hipMemcpy(embedding_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "embedding invocation table uploaded");
        }
        return true;
    }

    bool set_embedding_psq8_invocations(
        const mcu::McuEmbeddingPsq8Invocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuEmbeddingPsq8Invocation);
        if (hipMalloc(reinterpret_cast<void**>(&embedding_psq8_invocations),
                      bytes) != hipSuccess) {
            return check(false, "psq8 embedding invocation table allocated");
        }
        embedding_psq8_count = count;
        if (hipMemcpy(embedding_psq8_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "psq8 embedding invocation table uploaded");
        }
        return true;
    }

    bool set_gdn_conv1d_invocations(
        const mcu::McuGdnConv1dInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuGdnConv1dInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&gdn_conv1d_invocations), bytes) !=
            hipSuccess) {
            return check(false, "gdn conv1d invocation table allocated");
        }
        gdn_conv1d_count = count;
        if (hipMemcpy(gdn_conv1d_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "gdn conv1d invocation table uploaded");
        }
        return true;
    }

    bool set_gdn_recurrence_invocations(
        const mcu::McuGdnRecurrenceInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuGdnRecurrenceInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&gdn_recurrence_invocations),
                      bytes) != hipSuccess) {
            return check(false, "gdn recurrence invocation table allocated");
        }
        gdn_recurrence_count = count;
        if (hipMemcpy(gdn_recurrence_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "gdn recurrence invocation table uploaded");
        }
        return true;
    }

    bool set_gdn_reset_invocations(
        const mcu::GpuMcuGdnResetInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::GpuMcuGdnResetInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&gdn_reset_invocations), bytes) !=
            hipSuccess) {
            return check(false, "gdn reset invocation table allocated");
        }
        gdn_reset_count = count;
        if (hipMemcpy(gdn_reset_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "gdn reset invocation table uploaded");
        }
        return true;
    }

    bool set_verify_accept_invocations(
        const mcu::McuVerifyAcceptPrefixInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuVerifyAcceptPrefixInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&verify_accept_invocations),
                      bytes) != hipSuccess) {
            return check(false, "verify accept invocation table allocated");
        }
        verify_accept_count = count;
        if (hipMemcpy(verify_accept_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "verify accept invocation table uploaded");
        }
        return true;
    }

    bool set_verify_accept_batch_invocations(
        const mcu::McuVerifyAcceptBatchInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuVerifyAcceptBatchInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&verify_accept_batch_invocations),
                      bytes) != hipSuccess) {
            return check(false, "verify accept batch invocation table allocated");
        }
        verify_accept_batch_count = count;
        if (hipMemcpy(verify_accept_batch_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "verify accept batch invocation table uploaded");
        }
        return true;
    }

    bool set_gdn_spec_restore_invocations(
        const mcu::McuGdnSpecRestoreInvocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuGdnSpecRestoreInvocation);
        if (hipMalloc(reinterpret_cast<void**>(&gdn_spec_restore_invocations),
                      bytes) != hipSuccess) {
            return check(false, "gdn spec restore invocation table allocated");
        }
        gdn_spec_restore_count = count;
        if (hipMemcpy(gdn_spec_restore_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "gdn spec restore invocation table uploaded");
        }
        return true;
    }

    bool set_gdn_spec_restore_from_counts_invocations(
        const mcu::McuGdnSpecRestoreFromCountsInvocation* source,
        uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuGdnSpecRestoreFromCountsInvocation);
        if (hipMalloc(
                reinterpret_cast<void**>(&gdn_spec_restore_from_counts_invocations),
                bytes) != hipSuccess) {
            return check(false,
                         "gdn spec restore from counts invocation table allocated");
        }
        gdn_spec_restore_from_counts_count = count;
        if (hipMemcpy(gdn_spec_restore_from_counts_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false,
                         "gdn spec restore from counts invocation table uploaded");
        }
        return true;
    }

    bool set_argmax_f32_invocations(
        const mcu::McuArgmaxF32Invocation* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(mcu::McuArgmaxF32Invocation);
        if (hipMalloc(reinterpret_cast<void**>(&argmax_f32_invocations), bytes) !=
            hipSuccess) {
            return check(false, "argmax f32 invocation table allocated");
        }
        argmax_f32_count = count;
        if (hipMemcpy(argmax_f32_invocations, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "argmax f32 invocation table uploaded");
        }
        return true;
    }

    bool set_runtime_bindings(
        const mcu::McuRuntimeNodeBinding* source, uint32_t count) {
        if (count == 0u) return true;
        const std::size_t bytes = static_cast<std::size_t>(count) *
                                  sizeof(mcu::McuRuntimeNodeBinding);
        if (hipMalloc(reinterpret_cast<void**>(&runtime_bindings), bytes) !=
            hipSuccess) {
            return check(false, "dynamic binding table allocated");
        }
        runtime_binding_count = count;
        if (hipMemcpy(runtime_bindings, source, bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "dynamic binding table uploaded");
        }
        return true;
    }

    bool upload_kernarg_sources(
        const std::vector<mcu::McuKernargSourceDesc>& sources) {
        release_kernarg_sources();
        if (sources.empty()) return true;
        const std::size_t bytes =
            sources.size() * sizeof(mcu::McuKernargSourceDesc);
        if (hipMalloc(reinterpret_cast<void**>(&kernarg_sources), bytes) !=
            hipSuccess) {
            return check(false, "kernarg source table allocated");
        }
        kernarg_source_count = static_cast<uint32_t>(sources.size());
        if (hipMemcpy(kernarg_sources, sources.data(), bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            return check(false, "kernarg source table uploaded");
        }
        return true;
    }

    void release_kernarg_sources() {
        if (kernarg_sources != nullptr) {
            (void)hipFree(kernarg_sources);
            kernarg_sources = nullptr;
        }
        kernarg_source_count = 0u;
    }

    bool recipe_invocation_table(uint16_t recipe, const void*& base,
                                 uint32_t& count, std::size_t& stride) const {
        base = nullptr;
        count = 0u;
        stride = 0u;
        switch (recipe) {
            case mcu::kMcuKernargRecipeRmsNormBf16PfOnePlus:
                base = invocations;
                count = invocation_count;
                stride = sizeof(mcu::McuRmsNormInvocation);
                return true;
            case mcu::kMcuKernargRecipeActivationQuantizeA8:
                base = quantize_invocations;
                count = quantize_count;
                stride = sizeof(mcu::McuActivationQuantizeInvocation);
                return true;
            case mcu::kMcuKernargRecipeActivationQuantizeE4m3K5120:
                base = e4m3_invocations;
                count = e4m3_count;
                stride = sizeof(mcu::McuActivationQuantizeE4m3Invocation);
                return true;
            case mcu::kMcuKernargRecipePsq4Decode1Bf16U16:
            case mcu::kMcuKernargRecipePsq4Decode1Bf16U8:
            case mcu::kMcuKernargRecipePsq8Decode1Bf16U8:
                base = psq4_invocations;
                count = psq4_count;
                stride = sizeof(mcu::McuPsq4Decode1Invocation);
                return true;
            case mcu::kMcuKernargRecipePsq4MultiRowBf16:
                base = psq4_multi_invocations;
                count = psq4_multi_count;
                stride = sizeof(mcu::McuPsq4MultiRowInvocation);
                return true;
            case mcu::kMcuKernargRecipeVerifyAcceptPrefix:
                base = verify_accept_invocations;
                count = verify_accept_count;
                stride = sizeof(mcu::McuVerifyAcceptPrefixInvocation);
                return true;
            case mcu::kMcuKernargRecipeVerifyAcceptBatch:
                base = verify_accept_batch_invocations;
                count = verify_accept_batch_count;
                stride = sizeof(mcu::McuVerifyAcceptBatchInvocation);
                return true;
            case mcu::kMcuKernargRecipeGdnSpecRestore:
                base = gdn_spec_restore_invocations;
                count = gdn_spec_restore_count;
                stride = sizeof(mcu::McuGdnSpecRestoreInvocation);
                return true;
            case mcu::kMcuKernargRecipeGdnSpecRestoreFromCounts:
                base = gdn_spec_restore_from_counts_invocations;
                count = gdn_spec_restore_from_counts_count;
                stride = sizeof(mcu::McuGdnSpecRestoreFromCountsInvocation);
                return true;
            case mcu::kMcuKernargRecipeArgmaxF32:
                base = argmax_f32_invocations;
                count = argmax_f32_count;
                stride = sizeof(mcu::McuArgmaxF32Invocation);
                return true;
            case mcu::kMcuKernargRecipeElementwise:
                base = elementwise_invocations;
                count = elementwise_count;
                stride = sizeof(mcu::McuElementwiseInvocation);
                return true;
            case mcu::kMcuKernargRecipeRope:
                base = rope_invocations;
                count = rope_count;
                stride = sizeof(mcu::McuRopeInvocation);
                return true;
            case mcu::kMcuKernargRecipeKvAppend:
                base = kv_append_invocations;
                count = kv_append_count;
                stride = sizeof(mcu::McuKvAppendInvocation);
                return true;
            case mcu::kMcuKernargRecipeAttentionPaged:
                base = attention_paged_invocations;
                count = attention_paged_count;
                stride = sizeof(mcu::McuPagedAttentionInvocation);
                return true;
            case mcu::kMcuKernargRecipeAttentionPagedSplit:
                base = attention_paged_split_invocations;
                count = attention_paged_split_count;
                stride = sizeof(mcu::McuPagedAttentionSplitInvocation);
                return true;
            case mcu::kMcuKernargRecipeAttentionPagedReduce:
                base = attention_paged_reduce_invocations;
                count = attention_paged_reduce_count;
                stride = sizeof(mcu::McuPagedAttentionReduceInvocation);
                return true;
            case mcu::kMcuKernargRecipeBf16ExactRows:
                base = bf16_invocations;
                count = bf16_count;
                stride = sizeof(mcu::McuBf16ExactRowsInvocation);
                return true;
            case mcu::kMcuKernargRecipeBf16Wmma:
                base = bf16_wmma_invocations;
                count = bf16_wmma_count;
                stride = sizeof(mcu::McuBf16WmmaInvocation);
                return true;
            case mcu::kMcuKernargRecipeL2Normalize:
                base = l2_invocations;
                count = l2_count;
                stride = sizeof(mcu::McuL2NormalizeInvocation);
                return true;
            case mcu::kMcuKernargRecipeEmbeddingBf16:
                base = embedding_invocations;
                count = embedding_count;
                stride = sizeof(mcu::McuEmbeddingBf16Invocation);
                return true;
            case mcu::kMcuKernargRecipeEmbeddingPsq8:
                base = embedding_psq8_invocations;
                count = embedding_psq8_count;
                stride = sizeof(mcu::McuEmbeddingPsq8Invocation);
                return true;
            case mcu::kMcuKernargRecipeOutputGatherBf16:
                base = output_gather_invocations;
                count = output_gather_count;
                stride = sizeof(mcu::McuOutputGatherBf16Invocation);
                return true;
            case mcu::kMcuKernargRecipeGdnConv1d:
                base = gdn_conv1d_invocations;
                count = gdn_conv1d_count;
                stride = sizeof(mcu::McuGdnConv1dInvocation);
                return true;
            case mcu::kMcuKernargRecipeGdnRecurrence:
                base = gdn_recurrence_invocations;
                count = gdn_recurrence_count;
                stride = sizeof(mcu::McuGdnRecurrenceInvocation);
                return true;
            case mcu::kMcuKernargRecipeGdnReset:
                base = gdn_reset_invocations;
                count = gdn_reset_count;
                stride = sizeof(mcu::GpuMcuGdnResetInvocation);
                return true;
            default:
                return false;
        }
    }

    bool build_kernarg_sources() {
        release_kernarg_sources();
        if (host_plan.empty()) return true;
        std::vector<mcu::McuKernargSourceDesc> sources(host_plan.size());
        for (std::size_t i = 0; i < host_plan.size(); ++i) {
            const mcu::McuPlanNode& node = host_plan[i];
            if (node.kernarg_recipe == mcu::kMcuKernargRecipeProbe) {
                sources[i].flags = mcu::kMcuKernargSourceSupervisorProbe;
                sources[i].explicit_args_bytes =
                    static_cast<uint32_t>(sizeof(mcu::GpuMcuFsmWorkerArgs));
                continue;
            }
            const void* base = nullptr;
            uint32_t count = 0u;
            std::size_t stride = 0u;
            if (!recipe_invocation_table(node.kernarg_recipe, base, count,
                                         stride)) {
                continue;
            }
            if (base == nullptr || node.invocation_index >= count) continue;
            const auto* byte_base = static_cast<const unsigned char*>(base);
            sources[i].source = reinterpret_cast<uint64_t>(
                byte_base +
                static_cast<std::size_t>(node.invocation_index) * stride);
            sources[i].explicit_args_bytes = static_cast<uint32_t>(stride);
            sources[i].flags = mcu::kMcuKernargSourcePrepared;
        }
        return upload_kernarg_sources(sources);
    }

    std::vector<uint32_t> read_output_at(
        const std::vector<uint32_t>& pcs) const {
        std::vector<uint32_t> all(node_count, 0);
        device_to_host(all.data(), output, node_count * sizeof(uint32_t));
        std::vector<uint32_t> out;
        out.reserve(pcs.size());
        for (uint32_t pc : pcs) {
            out.push_back(pc < all.size() ? all[pc] : 0u);
        }
        return out;
    }

    void build_fsm_config(const mcu::GpuMcuFsmConfig& in,
                          mcu::GpuMcuFsmConfig& c) {
        c = in;
        c.queue = queue.device_queue_view();
        c.plan = plan;
        c.node_count = node_count;
        c.variants = variants;
        c.variant_count = variant_count;
        c.runtime_node_bindings = runtime_bindings;
        c.runtime_node_binding_count = runtime_binding_count;
        c.kernarg_sources = kernarg_sources;
        c.kernarg_source_count = kernarg_source_count;
        c.retained = retained;
        c.retained_count = variant_count;
        c.completions = completions;
        c.completion_count = completion_cap;
        c.output_base = reinterpret_cast<uint64_t>(output);
        c.timestamps_base = reinterpret_cast<uint64_t>(timestamps);
        c.kernarg_base = reinterpret_cast<uint64_t>(region.base);
        c.kernarg_slot_stride = region.slot_stride;
        if (c.kernarg_slot_count == 0u) {
            c.kernarg_slot_count = node_count;
        }
        c.timing = timing;
        c.timing_count = timing_cap;
        if (timing_cap == 0u) {
            c.timing = nullptr;
        }
        c.debug_records = records;
        c.debug_record_count = record_cap;
        if (record_cap == 0u) {
            c.debug_records = nullptr;
        }
    }

    bool configure_fsm(const mcu::GpuMcuFsmConfig& in, bool launch) {
        if (!build_kernarg_sources()) return false;
        mcu::GpuMcuFsmConfig c{};
        build_fsm_config(in, c);
        if (log_enabled && log_region.base == nullptr) {
            auto log_res = queue.allocate_log_region(mcu::kMcuLogBytes);
            if (log_res.ok()) log_region = log_res.release();
        }
        c.log_base = reinterpret_cast<uint64_t>(log_region.base);
        if (!check(c.kernarg_slot_count <= mcu::kMcuMaxKernargSlots,
                   "kernarg slots within limit")) {
            return false;
        }
        auto f = mcu::GpuMcuFsm::create(0, partition.control_stream());
        if (!check(f.ok(), "fsm created")) return false;
        fsm = f.release();
        const ps::Status cfg_status = fsm.configure(c);
        if (!cfg_status.ok())
            std::printf("  configure: %s\n", cfg_status.message().c_str());
        if (!check(cfg_status.ok(), "fsm configured")) return false;
        fsm_configured = true;
        if (!launch) {
            return true;
        }
        const ps::Status start_status = fsm.start();
        if (!start_status.ok())
            std::printf("  fsm start: %s\n", start_status.message().c_str());
        if (!check(start_status.ok(), "fsm started")) return false;
        fsm_started = true;
        return true;
    }

    bool start(const mcu::GpuMcuFsmConfig& in) {
        return configure_fsm(in, true);
    }

    bool prepare(const mcu::GpuMcuFsmConfig& in) {
        return configure_fsm(in, false);
    }

    void device_to_host(void* dst, const void* src, std::size_t bytes) const {
        hipStream_t stream = partition.worker_stream();
        (void)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream);
        (void)hipStreamSynchronize(stream);
    }

    bool drain_queue(uint32_t timeout_ms) const {
        const mcu::DeviceAqlQueueView view = queue.device_queue_view();
        if (view.read_index == nullptr || view.write_index == nullptr) return false;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const uint64_t read = *view.read_index;
            const uint64_t write = *view.write_index;
            if (read >= write) return true;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    std::vector<uint32_t> read_output() const {
        std::vector<uint32_t> out(node_count, 0);
        device_to_host(out.data(), output, node_count * sizeof(uint32_t));
        return out;
    }

    std::vector<uint64_t> read_timestamps() const {
        std::vector<uint64_t> out(
            static_cast<std::size_t>(node_count) * 2u, 0);
        device_to_host(out.data(), timestamps, out.size() * sizeof(uint64_t));
        return out;
    }

    std::vector<mcu::McuDispatchRecord> read_records() const {
        std::vector<mcu::McuDispatchRecord> out(record_cap);
        if (record_cap == 0u) return out;
        device_to_host(out.data(), records,
                       out.size() * sizeof(mcu::McuDispatchRecord));
        return out;
    }

    bool close() {
        bool ok = true;
        if (fsm_started) {
            ok &= check(fsm.request_stop().ok(), "fsm stop requested");
            ok &= check(fsm.wait_stopped(5000).ok(), "fsm stopped");
            ok &= check(fsm.shutdown().ok(), "fsm shutdown");
            fsm_started = false;
            fsm_configured = false;
        }
        if (fsm_configured) {
            ok &= check(fsm.shutdown().ok(), "fsm shutdown");
            fsm_configured = false;
        }
        if (output) { ok &= hipFree(output) == hipSuccess; output = nullptr; }
        if (timestamps) { ok &= hipFree(timestamps) == hipSuccess; timestamps = nullptr; }
        if (completions) { ok &= hipFree(completions) == hipSuccess; completions = nullptr; }
        if (plan) { ok &= hipFree(plan) == hipSuccess; plan = nullptr; }
        if (variants) { ok &= hipFree(variants) == hipSuccess; variants = nullptr; }
        if (retained) { ok &= hipFree(retained) == hipSuccess; retained = nullptr; }
        if (runtime_bindings) {
            ok &= hipFree(runtime_bindings) == hipSuccess;
            runtime_bindings = nullptr;
            runtime_binding_count = 0;
        }
        if (kernarg_sources) {
            ok &= hipFree(kernarg_sources) == hipSuccess;
            kernarg_sources = nullptr;
            kernarg_source_count = 0;
        }
        if (timing) { ok &= hipFree(timing) == hipSuccess; timing = nullptr; }
        if (records) { ok &= hipFree(records) == hipSuccess; records = nullptr; }
        if (invocations) {
            ok &= hipFree(invocations) == hipSuccess;
            invocations = nullptr;
            invocation_count = 0;
        }
        if (quantize_invocations) {
            ok &= hipFree(quantize_invocations) == hipSuccess;
            quantize_invocations = nullptr;
            quantize_count = 0;
        }
        if (e4m3_invocations) {
            ok &= hipFree(e4m3_invocations) == hipSuccess;
            e4m3_invocations = nullptr;
            e4m3_count = 0;
        }
        if (psq4_invocations) {
            ok &= hipFree(psq4_invocations) == hipSuccess;
            psq4_invocations = nullptr;
            psq4_count = 0;
        }
        if (psq4_multi_invocations) {
            ok &= hipFree(psq4_multi_invocations) == hipSuccess;
            psq4_multi_invocations = nullptr;
            psq4_multi_count = 0;
        }
        if (verify_accept_invocations) {
            ok &= hipFree(verify_accept_invocations) == hipSuccess;
            verify_accept_invocations = nullptr;
            verify_accept_count = 0;
        }
        if (verify_accept_batch_invocations) {
            ok &= hipFree(verify_accept_batch_invocations) == hipSuccess;
            verify_accept_batch_invocations = nullptr;
            verify_accept_batch_count = 0;
        }
        if (gdn_spec_restore_invocations) {
            ok &= hipFree(gdn_spec_restore_invocations) == hipSuccess;
            gdn_spec_restore_invocations = nullptr;
            gdn_spec_restore_count = 0;
        }
        if (gdn_spec_restore_from_counts_invocations) {
            ok &= hipFree(gdn_spec_restore_from_counts_invocations) == hipSuccess;
            gdn_spec_restore_from_counts_invocations = nullptr;
            gdn_spec_restore_from_counts_count = 0;
        }
        if (output_gather_invocations) {
            ok &= hipFree(output_gather_invocations) == hipSuccess;
            output_gather_invocations = nullptr;
            output_gather_count = 0;
        }
        if (argmax_f32_invocations) {
            ok &= hipFree(argmax_f32_invocations) == hipSuccess;
            argmax_f32_invocations = nullptr;
            argmax_f32_count = 0;
        }
        if (elementwise_invocations) {
            ok &= hipFree(elementwise_invocations) == hipSuccess;
            elementwise_invocations = nullptr;
            elementwise_count = 0;
        }
        if (rope_invocations) {
            ok &= hipFree(rope_invocations) == hipSuccess;
            rope_invocations = nullptr;
            rope_count = 0;
        }
        if (kv_append_invocations) {
            ok &= hipFree(kv_append_invocations) == hipSuccess;
            kv_append_invocations = nullptr;
            kv_append_count = 0;
        }
        if (attention_paged_invocations) {
            ok &= hipFree(attention_paged_invocations) == hipSuccess;
            attention_paged_invocations = nullptr;
            attention_paged_count = 0;
        }
        if (attention_paged_split_invocations) {
            ok &= hipFree(attention_paged_split_invocations) == hipSuccess;
            attention_paged_split_invocations = nullptr;
            attention_paged_split_count = 0;
        }
        if (attention_paged_reduce_invocations) {
            ok &= hipFree(attention_paged_reduce_invocations) == hipSuccess;
            attention_paged_reduce_invocations = nullptr;
            attention_paged_reduce_count = 0;
        }
        if (bf16_invocations) {
            ok &= hipFree(bf16_invocations) == hipSuccess;
            bf16_invocations = nullptr;
            bf16_count = 0;
        }
        if (bf16_wmma_invocations) {
            ok &= hipFree(bf16_wmma_invocations) == hipSuccess;
            bf16_wmma_invocations = nullptr;
            bf16_wmma_count = 0;
        }
        if (l2_invocations) {
            ok &= hipFree(l2_invocations) == hipSuccess;
            l2_invocations = nullptr;
            l2_count = 0;
        }
        if (embedding_invocations) {
            ok &= hipFree(embedding_invocations) == hipSuccess;
            embedding_invocations = nullptr;
            embedding_count = 0;
        }
        if (embedding_psq8_invocations) {
            ok &= hipFree(embedding_psq8_invocations) == hipSuccess;
            embedding_psq8_invocations = nullptr;
            embedding_psq8_count = 0;
        }
        if (gdn_conv1d_invocations) {
            ok &= hipFree(gdn_conv1d_invocations) == hipSuccess;
            gdn_conv1d_invocations = nullptr;
            gdn_conv1d_count = 0;
        }
        if (gdn_recurrence_invocations) {
            ok &= hipFree(gdn_recurrence_invocations) == hipSuccess;
            gdn_recurrence_invocations = nullptr;
            gdn_recurrence_count = 0;
        }
        if (gdn_reset_invocations) {
            ok &= hipFree(gdn_reset_invocations) == hipSuccess;
            gdn_reset_invocations = nullptr;
            gdn_reset_count = 0;
        }
        if (coherent_kernarg_base_) {
            ok &= hipFree(coherent_kernarg_base_) == hipSuccess;
            coherent_kernarg_base_ = nullptr;
        }
        for (auto& extra : extra_codes) {
            ok &= check(extra.shutdown().ok(), "extra code shutdown");
        }
        extra_codes.clear();
        ok &= check(code.shutdown().ok(), "code shutdown");
        ok &= check(queue.shutdown().ok(), "queue shutdown");
        ok &= check(partition.shutdown().ok(), "partition shutdown");
        return ok;
    }
};

}  // namespace gpu_mcu_test
