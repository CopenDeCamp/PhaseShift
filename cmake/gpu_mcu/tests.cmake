phaseshift_add_test(NAME test_gpu_mcu_mixed_gdn_verify_history SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_mixed_gdn_verify_history.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_verify_epilogue SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_verify_epilogue.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_verify_epilogue PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_dynamic_lm_head_rows SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_dynamic_lm_head_rows.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_dynamic_lm_head_rows PRIVATE "${CMAKE_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# GPU-MCU low-level substrate (required).
# ---------------------------------------------------------------------------

phaseshift_add_test(NAME test_gpu_mcu_aql_packet SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_aql_packet.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_aql_queue SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_aql_queue.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_cu_partition SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_cu_partition.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_worker_code_object SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_worker_code_object.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu DEFS PHASESHIFT_GPU_MCU_PROBE_HSACO="${CMAKE_BINARY_DIR}/gpu_mcu_probe_worker.elf.hsaco")
phaseshift_add_test(NAME test_gpu_mcu_device_enqueue SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_device_enqueue.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_batch_binding SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_batch_binding.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_slot_binding SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_slot_binding.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_batch_planner SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_batch_planner.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_persistent_ingress SOURCE unit/gpu_mcu/controller/test_gpu_mcu_persistent_ingress.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_batch_dispatch SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_batch_dispatch.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_batch_commit SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_batch_commit.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_output_ring SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_output_ring.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_autonomous_loop SOURCE unit/gpu_mcu/controller/test_gpu_mcu_autonomous_loop.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_continuous_turnover SOURCE unit/gpu_mcu/controller/test_gpu_mcu_continuous_turnover.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_kv_page_allocator SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_kv_page_allocator.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_sequence_resource SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_sequence_resource.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_verify_kv_transaction SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_verify_kv_transaction.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_gdn_reset SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_gdn_reset.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_GDN_RESET_HSACO="${PS_GPU_MCU_GDN_RESET_HSACO}" DEPENDS gpu_mcu_hsaco_gdn_reset)
phaseshift_add_test(NAME test_gpu_mcu_kv_page_turnover SOURCE unit/gpu_mcu/controller/test_gpu_mcu_kv_page_turnover.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_kv_page_turnover PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_continuous_turnover PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_autonomous_loop PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_batch_dispatch PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_request_ingress SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_request_ingress.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_slot_table SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_slot_table.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_control_ring SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_control_ring.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_persistent_mcu SOURCE unit/gpu_mcu/controller/test_gpu_mcu_persistent_mcu.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_persistent_emit SOURCE unit/gpu_mcu/controller/test_gpu_mcu_persistent_emit.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_aql_stage_commit SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_aql_stage_commit.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_retained_packet SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_retained_packet.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_production_decode SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_production_decode.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 24 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_gpu_mcu phaseshift_weights phaseshift_runtime phaseshift_gpu DEFS PS_MODEL_DIR_4B="${PHASESHIFT_MODEL_DIR_4B}")
target_include_directories(test_gpu_mcu_production_decode PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_plan_cache SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_plan_cache.hip LABELS "cpu;required" LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_plan_cache PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_decode_backend_policy SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_decode_backend_policy.hip LABELS "cpu;required" LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_decode_backend_policy PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_runtime_lifecycle SOURCE unit/gpu_mcu/controller/test_gpu_mcu_runtime_lifecycle.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 1
set_tests_properties(test_gpu_mcu_runtime_lifecycle PROPERTIES RUN_SERIAL TRUE) LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_gpu_mcu phaseshift_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_runtime_lifecycle PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_decode_fault_bridge SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_decode_fault_bridge.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" DEPENDS gpu_mcu_hsaco_elementwise)
phaseshift_add_test(NAME test_gpu_mcu_hip_stream_bridge SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_hip_stream_bridge.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" DEPENDS gpu_mcu_hsaco_elementwise)
phaseshift_add_test(NAME test_gpu_mcu_wall_clock_conversion SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_wall_clock_conversion.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_gpu)
phaseshift_add_test(NAME test_gpu_mcu_kernarg_hidden_bounds SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_kernarg_hidden_bounds.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_lds_templates SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_lds_templates.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_device_kernarg SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_device_kernarg.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_device_completion SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_device_completion.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_micro_fsm SOURCE unit/gpu_mcu/controller/test_gpu_mcu_micro_fsm.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_dynamic_plan_binding SOURCE unit/gpu_mcu/controller/test_gpu_mcu_dynamic_plan_binding.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_external_persistent_driver SOURCE unit/gpu_mcu/controller/test_gpu_mcu_external_persistent_driver.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_external_persistent_driver PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_prepared_dispatch SOURCE unit/gpu_mcu/controller/test_gpu_mcu_prepared_dispatch.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_append_while_running SOURCE unit/gpu_mcu/controller/test_gpu_mcu_append_while_running.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_append_multiple_while_running SOURCE unit/gpu_mcu/controller/test_gpu_mcu_append_multiple_while_running.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_doorbell_coalescing SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_doorbell_coalescing.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_queue_fed_chain SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_queue_fed_chain.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_continuous_refill SOURCE unit/gpu_mcu/controller/test_gpu_mcu_continuous_refill.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_continuous_ring_wrap SOURCE unit/gpu_mcu/controller/test_gpu_mcu_continuous_ring_wrap.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_append_without_redoorbell SOURCE unit/gpu_mcu/controller/test_gpu_mcu_append_without_redoorbell.hip LABELS "gpu1;gpu_mcu" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_kernarg_region_lifetime SOURCE unit/gpu_mcu/substrate/test_gpu_mcu_kernarg_region_lifetime.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu)
phaseshift_add_test(NAME test_gpu_mcu_real_rmsnorm_aql SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_rmsnorm_aql.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm)
phaseshift_add_test(NAME test_gpu_mcu_real_rmsnorm_fsm SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_rmsnorm_fsm.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm)
phaseshift_add_test(NAME test_gpu_mcu_real_rmsnorm_feed SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_rmsnorm_feed.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm)
set_tests_properties(test_gpu_mcu_real_rmsnorm_feed PROPERTIES RUN_SERIAL TRUE)
phaseshift_add_test(NAME test_gpu_mcu_real_rmsnorm_f32_pg SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_rmsnorm_f32_pg.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm)
phaseshift_add_test(NAME test_gpu_mcu_real_elementwise SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_elementwise.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" DEPENDS gpu_mcu_hsaco_elementwise)
phaseshift_add_test(NAME test_gpu_mcu_real_bf16_exact_rows SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_bf16_exact_rows.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_BF16_HSACO="${PS_GPU_MCU_BF16_HSACO}" DEPENDS gpu_mcu_hsaco_bf16)
phaseshift_add_test(NAME test_gpu_mcu_real_paged_attention_split SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_paged_attention_split.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_HSACO}" DEPENDS gpu_mcu_hsaco_attention_paged)
target_include_directories(test_gpu_mcu_real_paged_attention_split PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_real_paged_attention SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_paged_attention.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_HSACO}" DEPENDS gpu_mcu_hsaco_attention_paged)
target_include_directories(test_gpu_mcu_real_paged_attention PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_mixed_attention_regions SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_mixed_attention_regions.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_HSACO}" PHASESHIFT_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO}" DEPENDS gpu_mcu_hsaco_attention_paged gpu_mcu_hsaco_attention_paged_prefill)
target_include_directories(test_gpu_mcu_mixed_attention_regions PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_real_kv_append SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_kv_append.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_KV_APPEND_HSACO="${PS_GPU_MCU_KV_APPEND_HSACO}" DEPENDS gpu_mcu_hsaco_kv_append)
target_include_directories(test_gpu_mcu_real_kv_append PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_real_rope SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_rope.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ROPE_HSACO="${PS_GPU_MCU_ROPE_HSACO}" DEPENDS gpu_mcu_hsaco_rope)
phaseshift_add_test(NAME test_gpu_mcu_real_attention_rmsnorm SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_attention_rmsnorm.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm)
phaseshift_add_test(NAME test_gpu_mcu_real_split SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_split.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" DEPENDS gpu_mcu_hsaco_elementwise)
phaseshift_add_test(NAME test_gpu_mcu_real_l2_normalize SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_l2_normalize.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO="${PS_GPU_MCU_L2_NORMALIZE_HSACO}" DEPENDS gpu_mcu_hsaco_l2_normalize)
phaseshift_add_test(NAME test_gpu_mcu_real_embedding SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_embedding.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_EMBEDDING_HSACO="${PS_GPU_MCU_EMBEDDING_HSACO}" DEPENDS gpu_mcu_hsaco_embedding)
phaseshift_add_test(NAME test_gpu_mcu_verify_accept SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_verify_accept.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_VERIFY_ACCEPT_HSACO="${PS_GPU_MCU_VERIFY_ACCEPT_HSACO}" DEPENDS gpu_mcu_hsaco_verify_accept)
phaseshift_add_test(NAME test_gpu_mcu_gdn_verify_restore SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_gdn_verify_restore.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_GDN_SPEC_RESTORE_HSACO="${PS_GPU_MCU_GDN_SPEC_RESTORE_HSACO}" DEPENDS gpu_mcu_hsaco_gdn_spec_restore)
phaseshift_add_test(NAME test_gpu_mcu_sampling_argmax SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_sampling_argmax.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_SAMPLING_HSACO="${PS_GPU_MCU_SAMPLING_HSACO}" DEPENDS gpu_mcu_hsaco_sampling)
phaseshift_add_test(NAME test_gpu_mcu_mixed_batch_execution SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_mixed_batch_execution.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu_mcu phaseshift_gpu)
target_include_directories(test_gpu_mcu_mixed_batch_execution PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_plan_binder SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_plan_binder.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_plan_binder PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_multirow_chain_bridge SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_multirow_chain_bridge.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu_mcu phaseshift_gpu)
target_include_directories(test_gpu_mcu_multirow_chain_bridge PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_mixed_linear_attention_bridge SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_mixed_linear_attention_bridge.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu_mcu phaseshift_gpu)
target_include_directories(test_gpu_mcu_mixed_linear_attention_bridge PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_gdn_chain_bridge SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_gdn_chain_bridge.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu_mcu phaseshift_gpu)
target_include_directories(test_gpu_mcu_gdn_chain_bridge PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_real_gdn_conv1d SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_gdn_conv1d.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_GDN_CONV1D_HSACO="${PS_GPU_MCU_GDN_CONV1D_HSACO}" DEPENDS gpu_mcu_hsaco_gdn_conv1d)
phaseshift_add_test(NAME test_gpu_mcu_real_gdn_recurrence SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_gdn_recurrence.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_GDN_RECURRENCE_HSACO="${PS_GPU_MCU_GDN_RECURRENCE_HSACO}" DEPENDS gpu_mcu_hsaco_gdn_recurrence)
phaseshift_add_test(NAME test_gpu_mcu_real_primitive_chain SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_primitive_chain.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO="${PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO}" PHASESHIFT_GPU_MCU_PSQ4_HSACO="${PS_GPU_MCU_PSQ4_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm gpu_mcu_hsaco_activation_quantize gpu_mcu_hsaco_psq4)
phaseshift_add_test(NAME test_gpu_mcu_real_activation_quantize_aql SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_activation_quantize_aql.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO="${PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO}" DEPENDS gpu_mcu_hsaco_activation_quantize)
phaseshift_add_test(NAME test_gpu_mcu_real_activation_quantize_e4m3_aql SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_activation_quantize_e4m3_aql.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO="${PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO}" DEPENDS gpu_mcu_hsaco_activation_quantize)
phaseshift_add_test(NAME test_gpu_mcu_real_psq4_decode1_aql SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_psq4_decode1_aql.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_PSQ4_HSACO="${PS_GPU_MCU_PSQ4_HSACO}" DEPENDS gpu_mcu_hsaco_psq4)
phaseshift_add_test(NAME test_gpu_mcu_real_psq4_rows_aql SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_real_psq4_rows_aql.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_runtime phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_PSQ4_HSACO="${PS_GPU_MCU_PSQ4_HSACO}" DEPENDS gpu_mcu_hsaco_psq4)
target_include_directories(test_gpu_mcu_real_psq4_rows_aql PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_production_w4a8_chain SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_production_w4a8_chain.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO="${PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO}" PHASESHIFT_GPU_MCU_PSQ4_HSACO="${PS_GPU_MCU_PSQ4_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm gpu_mcu_hsaco_activation_quantize gpu_mcu_hsaco_psq4)
target_include_directories(test_gpu_mcu_production_w4a8_chain PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_w4a8_registry SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_w4a8_registry.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu)
target_include_directories(test_gpu_mcu_w4a8_registry PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_program_plan_three_primitive SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_program_plan_three_primitive.hip LABELS "gpu1;gpu_mcu;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu_mcu phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_gpu DEFS PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" PHASESHIFT_GPU_MCU_ACTIVATION_QUANTIZE_HSACO="${PS_GPU_MCU_ACTIVATION_QUANTIZE_HSACO}" PHASESHIFT_GPU_MCU_PSQ4_HSACO="${PS_GPU_MCU_PSQ4_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm gpu_mcu_hsaco_activation_quantize gpu_mcu_hsaco_psq4)
target_include_directories(test_gpu_mcu_program_plan_three_primitive PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_layer_dispatch_range SOURCE unit/gpu_mcu/qwen35/test_gpu_mcu_layer_dispatch_range.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_gpu_mcu_layer_dispatch_range PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_gpu_mcu_qwen_one_layer_inventory SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_qwen_one_layer_inventory.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 16 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_weights phaseshift_runtime phaseshift_gpu DEFS PS_MODEL_DIR_4B="${PHASESHIFT_MODEL_DIR_4B}")
target_include_directories(test_gpu_mcu_qwen_one_layer_inventory PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_qwen_one_layer_inventory SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")

phaseshift_add_test(NAME test_gpu_mcu_full_transformer_body_plan SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_full_transformer_body_plan.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 16 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_gpu_mcu phaseshift_weights phaseshift_runtime phaseshift_gpu DEFS PS_MODEL_DIR_4B="${PHASESHIFT_MODEL_DIR_4B}" PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" PHASESHIFT_GPU_MCU_BF16_HSACO="${PS_GPU_MCU_BF16_HSACO}" PHASESHIFT_GPU_MCU_EMBEDDING_HSACO="${PS_GPU_MCU_EMBEDDING_HSACO}" PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO="${PS_GPU_MCU_L2_NORMALIZE_HSACO}" PHASESHIFT_GPU_MCU_GDN_CONV1D_HSACO="${PS_GPU_MCU_GDN_CONV1D_HSACO}" PHASESHIFT_GPU_MCU_GDN_RECURRENCE_HSACO="${PS_GPU_MCU_GDN_RECURRENCE_HSACO}" PHASESHIFT_GPU_MCU_GDN_RESET_HSACO="${PS_GPU_MCU_GDN_RESET_HSACO}" PHASESHIFT_GPU_MCU_ROPE_HSACO="${PS_GPU_MCU_ROPE_HSACO}" PHASESHIFT_GPU_MCU_KV_APPEND_HSACO="${PS_GPU_MCU_KV_APPEND_HSACO}" PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_HSACO}" PHASESHIFT_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm gpu_mcu_hsaco_elementwise gpu_mcu_hsaco_bf16 gpu_mcu_hsaco_embedding gpu_mcu_hsaco_l2_normalize gpu_mcu_hsaco_gdn_conv1d gpu_mcu_hsaco_gdn_recurrence gpu_mcu_hsaco_gdn_reset gpu_mcu_hsaco_rope gpu_mcu_hsaco_kv_append gpu_mcu_hsaco_attention_paged gpu_mcu_hsaco_attention_paged_prefill)
target_include_directories(test_gpu_mcu_full_transformer_body_plan PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_full_transformer_body_plan SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
phaseshift_add_test(NAME test_gpu_mcu_attention_one_layer_plan SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_attention_one_layer_plan.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 16 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_gpu_mcu phaseshift_weights phaseshift_runtime phaseshift_gpu DEFS PS_MODEL_DIR_4B="${PHASESHIFT_MODEL_DIR_4B}" PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" PHASESHIFT_GPU_MCU_BF16_HSACO="${PS_GPU_MCU_BF16_HSACO}" PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO="${PS_GPU_MCU_L2_NORMALIZE_HSACO}" PHASESHIFT_GPU_MCU_ROPE_HSACO="${PS_GPU_MCU_ROPE_HSACO}" PHASESHIFT_GPU_MCU_KV_APPEND_HSACO="${PS_GPU_MCU_KV_APPEND_HSACO}" PHASESHIFT_GPU_MCU_ATTENTION_PAGED_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_HSACO}" PHASESHIFT_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm gpu_mcu_hsaco_elementwise gpu_mcu_hsaco_bf16 gpu_mcu_hsaco_l2_normalize gpu_mcu_hsaco_rope gpu_mcu_hsaco_kv_append gpu_mcu_hsaco_attention_paged gpu_mcu_hsaco_attention_paged_prefill)
target_include_directories(test_gpu_mcu_attention_one_layer_plan PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_attention_one_layer_plan SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
phaseshift_add_test(NAME test_gpu_mcu_qwen_one_layer_plan SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_qwen_one_layer_plan.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 16 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_gpu_mcu phaseshift_weights phaseshift_runtime phaseshift_gpu DEFS PS_MODEL_DIR_4B="${PHASESHIFT_MODEL_DIR_4B}" PHASESHIFT_GPU_MCU_RMSNORM_HSACO="${PS_GPU_MCU_RMSNORM_HSACO}" PHASESHIFT_GPU_MCU_ELEMENTWISE_HSACO="${PS_GPU_MCU_ELEMENTWISE_HSACO}" PHASESHIFT_GPU_MCU_BF16_HSACO="${PS_GPU_MCU_BF16_HSACO}" PHASESHIFT_GPU_MCU_L2_NORMALIZE_HSACO="${PS_GPU_MCU_L2_NORMALIZE_HSACO}" PHASESHIFT_GPU_MCU_GDN_CONV1D_HSACO="${PS_GPU_MCU_GDN_CONV1D_HSACO}" PHASESHIFT_GPU_MCU_GDN_RECURRENCE_HSACO="${PS_GPU_MCU_GDN_RECURRENCE_HSACO}" PHASESHIFT_GPU_MCU_GDN_RESET_HSACO="${PS_GPU_MCU_GDN_RESET_HSACO}" PHASESHIFT_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO="${PS_GPU_MCU_ATTENTION_PAGED_PREFILL_HSACO}" DEPENDS gpu_mcu_hsaco_rmsnorm gpu_mcu_hsaco_elementwise gpu_mcu_hsaco_bf16 gpu_mcu_hsaco_l2_normalize gpu_mcu_hsaco_gdn_conv1d gpu_mcu_hsaco_gdn_recurrence gpu_mcu_hsaco_gdn_reset gpu_mcu_hsaco_attention_paged_prefill)
target_include_directories(test_gpu_mcu_qwen_one_layer_plan PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_qwen_one_layer_plan SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")

phaseshift_add_test(NAME test_gpu_mcu_attention_layer_inventory SOURCE unit/gpu_mcu/acceptance/test_gpu_mcu_attention_layer_inventory.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 16 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35 phaseshift_weights phaseshift_runtime phaseshift_gpu DEFS PS_MODEL_DIR_4B="${PHASESHIFT_MODEL_DIR_4B}")
target_include_directories(test_gpu_mcu_attention_layer_inventory PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_gpu_mcu_attention_layer_inventory SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")

set(PS_GPU_MCU_REQUIRED_TESTS
    test_gpu_mcu_mixed_gdn_verify_history
    test_gpu_mcu_dynamic_lm_head_rows
    test_gpu_mcu_verify_epilogue
    test_gpu_mcu_aql_packet
    test_gpu_mcu_aql_queue
    test_gpu_mcu_cu_partition
    test_gpu_mcu_worker_code_object
    test_gpu_mcu_device_enqueue
    test_gpu_mcu_control_ring
    test_gpu_mcu_persistent_mcu
    test_gpu_mcu_persistent_emit
    test_gpu_mcu_aql_stage_commit
    test_gpu_mcu_retained_packet
    test_gpu_mcu_kernarg_hidden_bounds
    test_gpu_mcu_lds_templates
    test_gpu_mcu_device_kernarg
    test_gpu_mcu_device_completion
    test_gpu_mcu_micro_fsm
    test_gpu_mcu_dynamic_plan_binding
    test_gpu_mcu_external_persistent_driver
    test_gpu_mcu_verify_accept
    test_gpu_mcu_gdn_verify_restore
    test_gpu_mcu_sampling_argmax
    test_gpu_mcu_mixed_batch_execution
    test_gpu_mcu_plan_binder
    test_gpu_mcu_multirow_chain_bridge
    test_gpu_mcu_mixed_linear_attention_bridge
    test_gpu_mcu_gdn_chain_bridge
    test_gpu_mcu_prepared_dispatch
    test_gpu_mcu_append_while_running
    test_gpu_mcu_append_multiple_while_running
    test_gpu_mcu_doorbell_coalescing
    test_gpu_mcu_queue_fed_chain
    test_gpu_mcu_continuous_refill
    test_gpu_mcu_continuous_ring_wrap
    test_gpu_mcu_kernarg_region_lifetime
    test_gpu_mcu_real_rmsnorm_aql
    test_gpu_mcu_real_rmsnorm_fsm
    test_gpu_mcu_real_rmsnorm_feed
    test_gpu_mcu_real_rmsnorm_f32_pg
    test_gpu_mcu_real_elementwise
    test_gpu_mcu_real_split
    test_gpu_mcu_real_attention_rmsnorm
    test_gpu_mcu_real_rope
    test_gpu_mcu_real_kv_append
    test_gpu_mcu_real_paged_attention
    test_gpu_mcu_real_paged_attention_split
    test_gpu_mcu_mixed_attention_regions
    test_gpu_mcu_real_bf16_exact_rows
    test_gpu_mcu_real_l2_normalize
    test_gpu_mcu_real_embedding
    test_gpu_mcu_real_gdn_conv1d
    test_gpu_mcu_real_gdn_recurrence
    test_gpu_mcu_real_primitive_chain
    test_gpu_mcu_real_activation_quantize_aql
    test_gpu_mcu_real_activation_quantize_e4m3_aql
    test_gpu_mcu_real_psq4_decode1_aql
    test_gpu_mcu_real_psq4_rows_aql
    test_gpu_mcu_verify_kv_transaction
    test_gpu_mcu_production_w4a8_chain
    test_gpu_mcu_w4a8_registry
    test_gpu_mcu_program_plan_three_primitive
    test_gpu_mcu_layer_dispatch_range
)
