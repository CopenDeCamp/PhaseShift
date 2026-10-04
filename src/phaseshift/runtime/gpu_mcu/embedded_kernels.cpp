#include <phaseshift/runtime/gpu_mcu/embedded_kernels.h>

#include <cstring>

#include "phaseshift_gpu_mcu_attention_paged_hsaco.inc"
#include "phaseshift_gpu_mcu_attention_paged_prefill_hsaco.inc"
#include "phaseshift_gpu_mcu_activation_quantize_hsaco.inc"
#include "phaseshift_gpu_mcu_bf16_hsaco.inc"
#include "phaseshift_gpu_mcu_embedding_hsaco.inc"
#include "phaseshift_gpu_mcu_output_gather_hsaco.inc"
#include "phaseshift_gpu_mcu_elementwise_hsaco.inc"
#include "phaseshift_gpu_mcu_gdn_conv1d_hsaco.inc"
#include "phaseshift_gpu_mcu_gdn_recurrence_hsaco.inc"
#include "phaseshift_gpu_mcu_gdn_reset_hsaco.inc"
#include "phaseshift_gpu_mcu_kv_append_hsaco.inc"
#include "phaseshift_gpu_mcu_l2_normalize_hsaco.inc"
#include "phaseshift_gpu_mcu_psq4_hsaco.inc"
#include "phaseshift_gpu_mcu_psq8_hsaco.inc"
#include "phaseshift_gpu_mcu_rmsnorm_hsaco.inc"
#include "phaseshift_gpu_mcu_rope_hsaco.inc"
#include "phaseshift_gpu_mcu_verify_accept_hsaco.inc"
#include "phaseshift_gpu_mcu_gdn_spec_restore_hsaco.inc"
#include "phaseshift_gpu_mcu_sampling_hsaco.inc"

namespace ps {
namespace runtime {
namespace gpu_mcu {

namespace {

struct EmbeddedEntry {
    const char* name;
    const unsigned char* blob;
    std::size_t size;
};

#define PS_MCU_EMBED_ENTRY(name)                                      \
    {                                                                 \
        #name, phaseshift_gpu_mcu_##name##_hsaco,                     \
            sizeof(phaseshift_gpu_mcu_##name##_hsaco)                 \
    }

const EmbeddedEntry kEntries[] = {
    PS_MCU_EMBED_ENTRY(rmsnorm),
    PS_MCU_EMBED_ENTRY(elementwise),
    PS_MCU_EMBED_ENTRY(bf16),
    PS_MCU_EMBED_ENTRY(embedding),
    PS_MCU_EMBED_ENTRY(output_gather),
    PS_MCU_EMBED_ENTRY(l2_normalize),
    PS_MCU_EMBED_ENTRY(gdn_conv1d),
    PS_MCU_EMBED_ENTRY(gdn_recurrence),
    PS_MCU_EMBED_ENTRY(gdn_reset),
    PS_MCU_EMBED_ENTRY(rope),
    PS_MCU_EMBED_ENTRY(verify_accept),
    PS_MCU_EMBED_ENTRY(gdn_spec_restore),
    PS_MCU_EMBED_ENTRY(sampling),
    PS_MCU_EMBED_ENTRY(kv_append),
    PS_MCU_EMBED_ENTRY(attention_paged),
    PS_MCU_EMBED_ENTRY(attention_paged_prefill),
    PS_MCU_EMBED_ENTRY(activation_quantize),
    PS_MCU_EMBED_ENTRY(psq4),
    PS_MCU_EMBED_ENTRY(psq8),
};

#undef PS_MCU_EMBED_ENTRY

}  // namespace

GpuMcuEmbeddedKernel gpu_mcu_embedded_kernel(const char* name) noexcept {
    GpuMcuEmbeddedKernel out{};
    if (name == nullptr) return out;
    for (const EmbeddedEntry& entry : kEntries) {
        if (std::strcmp(entry.name, name) != 0) continue;
        out.blob = entry.blob;
        out.size = entry.size;
        return out;
    }
    return out;
}

}  // namespace gpu_mcu
}  // namespace runtime
}  // namespace ps
