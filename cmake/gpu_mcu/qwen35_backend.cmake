add_library(phaseshift_qwen35_gpu_mcu STATIC
    src/phaseshift/runtime/gpu_mcu/embedded_kernels.cpp
)
target_compile_features(phaseshift_qwen35_gpu_mcu PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_gpu_mcu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35_gpu_mcu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")
target_link_libraries(phaseshift_qwen35_gpu_mcu PUBLIC phaseshift_gpu_mcu)
target_compile_options(phaseshift_qwen35_gpu_mcu PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

phaseshift_embed_gpu_mcu_hsaco(rmsnorm gpu_mcu_hsaco_rmsnorm
    "${PS_GPU_MCU_RMSNORM_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(elementwise gpu_mcu_hsaco_elementwise
    "${PS_GPU_MCU_ELEMENTWISE_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(bf16 gpu_mcu_hsaco_bf16
    "${PS_GPU_MCU_BF16_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(embedding gpu_mcu_hsaco_embedding
    "${PS_GPU_MCU_EMBEDDING_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(output_gather gpu_mcu_hsaco_output_gather
    "${PS_GPU_MCU_OUTPUT_GATHER_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(l2_normalize gpu_mcu_hsaco_l2_normalize
    "${PS_GPU_MCU_L2_NORMALIZE_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(verify_accept gpu_mcu_hsaco_verify_accept
    "${PS_GPU_MCU_VERIFY_ACCEPT_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(gdn_spec_restore gpu_mcu_hsaco_gdn_spec_restore
    "${PS_GPU_MCU_GDN_SPEC_RESTORE_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(sampling gpu_mcu_hsaco_sampling
    "${PS_GPU_MCU_SAMPLING_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(gdn_conv1d gpu_mcu_hsaco_gdn_conv1d
    "${PS_GPU_MCU_GDN_CONV1D_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(gdn_recurrence gpu_mcu_hsaco_gdn_recurrence
    "${PS_GPU_MCU_GDN_RECURRENCE_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(gdn_reset gpu_mcu_hsaco_gdn_reset
    "${PS_GPU_MCU_GDN_RESET_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(rope gpu_mcu_hsaco_rope
    "${PS_GPU_MCU_ROPE_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(kv_append gpu_mcu_hsaco_kv_append
    "${PS_GPU_MCU_KV_APPEND_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(attention_paged gpu_mcu_hsaco_attention_paged
    "${PS_GPU_MCU_ATTENTION_PAGED_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(attention_paged_prefill gpu_mcu_hsaco_attention_paged_prefill
    "${PS_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(activation_quantize gpu_mcu_hsaco_activation_quantize
    "${PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(psq4 gpu_mcu_hsaco_psq4
    "${PS_GPU_MCU_PSQ4_HSACO}" phaseshift_qwen35_gpu_mcu)
phaseshift_embed_gpu_mcu_hsaco(psq8 gpu_mcu_hsaco_psq8
    "${PS_GPU_MCU_PSQ8_HSACO}" phaseshift_qwen35_gpu_mcu)
