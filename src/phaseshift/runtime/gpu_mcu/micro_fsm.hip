#include <phaseshift/runtime/gpu_mcu/micro_fsm.h>

#include <phaseshift/runtime/gpu_mcu/execution_bridge.h>
#include <phaseshift/runtime/gpu_mcu/fsm_worker.h>
#include <phaseshift/runtime/gpu_mcu/persistent_mcu.h>
#include <phaseshift/runtime/gpu_mcu/plan_binder.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>

namespace ps::runtime::gpu_mcu {

namespace {

constexpr uint32_t kMcuCompletionSpinLimit = 4000000000u;
constexpr uint32_t kMcuCompletionStopCheckMask = 0xffu;
constexpr uint32_t kMcuStartSpinLimit = 10000000u;

#include "detail/mcu_log.h"
#include "detail/micro_fsm_kernarg.h"
#include "detail/micro_fsm_run_once.h"
#include "detail/micro_fsm_control.h"
}  // namespace

__global__ void gpu_mcu_persistent_loop_kernel(GpuMcuPersistentState* state) {
    gpu_mcu_persistent_loop(state);
}


McuKernelVariantDesc make_kernel_variant(uint16_t variant_id,
                                         const GpuAqlKernelMetadata& meta,
                                         uint16_t kernarg_recipe,
                                         uint32_t workgroup_x,
                                         uint32_t workgroup_count_x,
                                         const AqlMemoryPolicy& policy,
                                         AqlHiddenArgsPolicy hidden_policy,
                                         uint32_t workgroup_count_y,
                                         uint32_t workgroup_count_z) {
    McuKernelVariantDesc out{};
    out.variant_id = variant_id;
    out.kernarg_recipe = kernarg_recipe;
    out.kernarg_size = meta.kernarg_segment_size;
    out.workgroup_count_x = workgroup_count_x;
    out.workgroup_count_y = workgroup_count_y;
    out.workgroup_count_z = workgroup_count_z;
    out.workgroup_x = workgroup_x;
    out.workgroup_y = 1;
    out.workgroup_z = 1;
    out.hidden_args_policy = static_cast<uint32_t>(hidden_policy);

    GpuAqlDispatchDesc desc{};
    desc.kernel_object = meta.kernel_object;
    desc.global_work_items_x = workgroup_count_x * workgroup_x;
    desc.global_work_items_y = workgroup_count_y;
    desc.global_work_items_z = workgroup_count_z;
    desc.workgroup_size_x = static_cast<uint16_t>(workgroup_x);
    desc.workgroup_size_y = 1;
    desc.workgroup_size_z = 1;
    desc.private_segment_size = meta.private_segment_size;
    desc.group_segment_size = meta.static_group_segment_size;
    desc.kernarg_address = nullptr;
    desc.completion_signal = 0;
    out.packet = make_retained_packet(build_aql_packet_template(desc, policy));
    return out;
}

McuKernelVariantDesc make_probe_variant(uint16_t variant_id,
                                        const GpuAqlKernelMetadata& meta,
                                        uint32_t workgroup_x,
                                        uint32_t workgroup_count_x,
                                        const AqlMemoryPolicy& policy,
                                        AqlHiddenArgsPolicy hidden_policy) {
    return make_kernel_variant(variant_id, meta, kMcuKernargRecipeProbe,
                               workgroup_x, workgroup_count_x, policy,
                               hidden_policy);
}

#include "detail/micro_fsm_host.h"
}  // namespace ps::runtime::gpu_mcu
