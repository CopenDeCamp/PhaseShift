get_filename_component(_PS_HIPCC_HINT_DIR "${CMAKE_HIP_COMPILER}" DIRECTORY)
find_program(PS_HIPCC_EXECUTABLE hipcc
    HINTS "${_PS_HIPCC_HINT_DIR}/../../bin" "${_PS_HIPCC_HINT_DIR}"
    PATH_SUFFIXES bin
    REQUIRED)

# Real production primitives are dispatched by stable extern "C" symbol, so each
# source also has to exist as a standalone HSACO. The flags mirror the main HIP
# build so the host copy and the HSACO copy generate identical device code.
function(phaseshift_add_hsaco target source)
    set(_bundle "${CMAKE_CURRENT_BINARY_DIR}/${target}.hsaco")
    set(_elf "${CMAKE_CURRENT_BINARY_DIR}/${target}.elf.hsaco")
    add_custom_command(
        OUTPUT "${_bundle}"
        COMMAND "${PS_HIPCC_EXECUTABLE}"
            --genco
            -DNDEBUG
            -DUSE_PROF_API=1
            -D__HIP_PLATFORM_AMD__=1
            -D__HIP_ROCclr__=1
            -I${CMAKE_SOURCE_DIR}/include
            -I${CMAKE_SOURCE_DIR}/src
            -O3
            -std=gnu++20
            --offload-arch=${CMAKE_HIP_ARCHITECTURES}
            "${CMAKE_SOURCE_DIR}/${source}"
            -o "${_bundle}"
        DEPENDS "${CMAKE_SOURCE_DIR}/${source}" ${ARGN}
        VERBATIM)
    add_custom_command(
        OUTPUT "${_elf}"
        COMMAND dd "if=${_bundle}" "of=${_elf}" bs=4096 skip=1
        DEPENDS "${_bundle}"
        VERBATIM)
    add_custom_target("${target}" DEPENDS "${_elf}")
    set("${target}_HSACO" "${_elf}" PARENT_SCOPE)
endfunction()

# Real production HSACOs are embedded as blobs so the production runtime never
# depends on a build directory path. Each blob is registered by a short name.
function(phaseshift_embed_gpu_mcu_hsaco name target hsaco owner)
    set(_inc
        "${CMAKE_CURRENT_BINARY_DIR}/gen/phaseshift_gpu_mcu_${name}_hsaco.inc")
    add_custom_command(
        OUTPUT "${_inc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/gen"
        COMMAND sh -c
            "xxd -i -n phaseshift_gpu_mcu_${name}_hsaco '${hsaco}' > '${_inc}'"
        DEPENDS "${hsaco}" ${target}
        VERBATIM)
    set_source_files_properties("${_inc}"
        PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
    target_sources(${owner} PRIVATE "${_inc}")
endfunction()

phaseshift_add_hsaco(gpu_mcu_hsaco_rmsnorm
    src/phaseshift/models/qwen35/kernels/optimized/rmsnorm.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/rmsnorm.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/rmsnorm_device.h")
set(PS_GPU_MCU_RMSNORM_HSACO "${gpu_mcu_hsaco_rmsnorm_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_activation_quantize
    src/phaseshift/models/qwen35/kernels/optimized/activation_quantize.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/activation_quantize.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/activation_quantize_device.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/fpx/e4m3_detail.h")
set(PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO
    "${gpu_mcu_hsaco_activation_quantize_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_psq4
    src/phaseshift/models/qwen35/kernels/optimized/linear/psq4.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/psq4.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/gemm_policy.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/config.h")
set(PS_GPU_MCU_PSQ4_HSACO "${gpu_mcu_hsaco_psq4_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_psq8
    src/phaseshift/models/qwen35/kernels/optimized/linear/psq8.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/psq8.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/gemm_policy.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/config.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h")
set(PS_GPU_MCU_PSQ8_HSACO "${gpu_mcu_hsaco_psq8_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_elementwise
    src/phaseshift/models/qwen35/kernels/optimized/elementwise.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/elementwise.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/vector_io.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/residual_add_device.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/target_swiglu_device.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/sigmoid_device.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/gdn/detail/gdn_prepare_device.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/gdn/detail/gdn_post_device.h")
set(PS_GPU_MCU_ELEMENTWISE_HSACO "${gpu_mcu_hsaco_elementwise_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_rope
    src/phaseshift/models/qwen35/kernels/optimized/attention/rope.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/attention/rope.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h")
set(PS_GPU_MCU_ROPE_HSACO "${gpu_mcu_hsaco_rope_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_kv_append
    src/phaseshift/models/qwen35/kernels/optimized/kv_append.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/kv_append.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/state/kv_cache_types.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/psq/psq4_kv.h")
set(PS_GPU_MCU_KV_APPEND_HSACO "${gpu_mcu_hsaco_kv_append_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_attention_paged
    src/phaseshift/models/qwen35/kernels/optimized/attention/paged_decode.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/attention_paged_common.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/fpx/e4m3_detail.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/psq/psq_device.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/state/kv_cache_types.h")
set(PS_GPU_MCU_ATTENTION_PAGED_HSACO "${gpu_mcu_hsaco_attention_paged_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_attention_paged_prefill
    src/phaseshift/models/qwen35/kernels/optimized/attention/paged_prefill.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/attention/paged_attention.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/detail/attention_paged_common.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/fpx/e4m3_detail.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/psq/psq_device.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/state/kv_cache_types.h")
set(PS_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO "${gpu_mcu_hsaco_attention_paged_prefill_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_bf16
    src/phaseshift/models/qwen35/kernels/optimized/linear/bf16.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/bf16.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/linear/config.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h")
set(PS_GPU_MCU_BF16_HSACO "${gpu_mcu_hsaco_bf16_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_embedding
    src/phaseshift/models/qwen35/kernels/optimized/embedding.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/embedding.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/quantization/fpx/e4m3_detail.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/program/device_program.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h")
set(PS_GPU_MCU_EMBEDDING_HSACO "${gpu_mcu_hsaco_embedding_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_output_gather
    src/phaseshift/models/qwen35/kernels/optimized/output_gather.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/output_gather.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/batch/device_batch_context.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/graph/value_type.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h")
set(PS_GPU_MCU_OUTPUT_GATHER_HSACO "${gpu_mcu_hsaco_output_gather_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_l2_normalize
    src/phaseshift/models/qwen35/kernels/optimized/l2_normalize.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/l2_normalize.h"
    "${CMAKE_SOURCE_DIR}/src/phaseshift/models/qwen35/kernels/optimized/gdn/detail/gdn_prepare_device.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h")
set(PS_GPU_MCU_L2_NORMALIZE_HSACO "${gpu_mcu_hsaco_l2_normalize_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_verify_accept
    src/phaseshift/models/qwen35/kernels/optimized/verify_accept.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/verify_accept.h")
set(PS_GPU_MCU_VERIFY_ACCEPT_HSACO "${gpu_mcu_hsaco_verify_accept_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_gdn_spec_restore
    src/phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/gdn/spec_restore.h")
set(PS_GPU_MCU_GDN_SPEC_RESTORE_HSACO "${gpu_mcu_hsaco_gdn_spec_restore_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_sampling
    src/phaseshift/models/qwen35/kernels/optimized/sampling.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/sampling.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/batch/device_batch_context.h")
set(PS_GPU_MCU_SAMPLING_HSACO "${gpu_mcu_hsaco_sampling_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_gdn_conv1d
    src/phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/batch/device_batch_context.h")
set(PS_GPU_MCU_GDN_CONV1D_HSACO "${gpu_mcu_hsaco_gdn_conv1d_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_gdn_recurrence
    src/phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/core/memory/types.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/batch/device_batch_context.h")
set(PS_GPU_MCU_GDN_RECURRENCE_HSACO "${gpu_mcu_hsaco_gdn_recurrence_HSACO}")

phaseshift_add_hsaco(gpu_mcu_hsaco_gdn_reset
    src/phaseshift/runtime/gpu_mcu/gdn_reset.hip
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/gpu_mcu/gdn_reset.h"
    "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/batch/device_batch_context.h")
set(PS_GPU_MCU_GDN_RESET_HSACO "${gpu_mcu_hsaco_gdn_reset_HSACO}")
