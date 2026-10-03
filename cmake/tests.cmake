# PhaseShift test targets.
#
# Test policy:
#   - required: 14 self-contained contract / correctness tests.
#     Repository + ROCm + 1 GPU only. No external files, no skip machinery.
#   - optional: Qwen3.5-4B application E2E (external model at
#     PHASESHIFT_MODEL_DIR_4B + committed oracle fixture). Built only when
#     PHASESHIFT_BUILD_OPTIONAL_TESTS=ON.
#
# Included via tests/CMakeLists.txt (thin wrapper).

set(_PS_TESTS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(phaseshift-gpu-test-runner
    "${_PS_TESTS_ROOT}/support/gpu_test_runner/main.cpp"
)
target_compile_features(phaseshift-gpu-test-runner PRIVATE cxx_std_20)

# phaseshift_add_test(
#   NAME <test>
#   SOURCE <relpath from tests/>
#   [LABELS <label...>]
#   [LIBRARIES <libs...>]
#   [TIMEOUT <sec>]
#   [GPU_COUNT <n>]
#   [GPU_COST_GB <gb>]
# )
#
# Links exactly the listed libraries (never the phaseshift aggregate).
# GPU_COUNT > 0 wraps the test in phaseshift-gpu-test-runner:
# VRAM budget reservation (default budget 24GB/GPU, env
# PHASESHIFT_TEST_GPU_BUDGET_GB), GPU assignment via HIP_VISIBLE_DEVICES,
# hard deadline (SIGKILL on timeout), exit code normalization.
function(phaseshift_add_test)
    cmake_parse_arguments(
        PS_TEST
        ""
        "NAME;SOURCE;TIMEOUT;GPU_COUNT;GPU_COST_GB"
        "LIBRARIES;LABELS;DEFS;DEPENDS"
        ${ARGN}
    )
    add_executable(${PS_TEST_NAME} "${_PS_TESTS_ROOT}/${PS_TEST_SOURCE}")
    phaseshift_set_rocm_rpath(${PS_TEST_NAME})
    phaseshift_set_hip_archs(${PS_TEST_NAME})
    target_compile_features(${PS_TEST_NAME} PRIVATE cxx_std_20)
    target_include_directories(${PS_TEST_NAME} PRIVATE "${_PS_TESTS_ROOT}")
    target_link_libraries(${PS_TEST_NAME} PRIVATE ${PS_TEST_LIBRARIES})
    if(PS_TEST_DEFS)
        target_compile_definitions(${PS_TEST_NAME} PRIVATE ${PS_TEST_DEFS})
    endif()
    if(PS_TEST_DEPENDS)
        add_dependencies(${PS_TEST_NAME} ${PS_TEST_DEPENDS})
    endif()
    if(PS_TEST_GPU_COUNT GREATER 0)
        if(NOT PS_TEST_TIMEOUT)
            set(PS_TEST_TIMEOUT 120)
        endif()
        if(NOT PS_TEST_GPU_COST_GB)
            set(PS_TEST_GPU_COST_GB 1)
        endif()
        math(EXPR PS_TEST_BACKSTOP "${PS_TEST_TIMEOUT} + 960")
        add_test(NAME ${PS_TEST_NAME} COMMAND phaseshift-gpu-test-runner
            --gpu-count ${PS_TEST_GPU_COUNT}
            --cost-gb ${PS_TEST_GPU_COST_GB}
            --timeout ${PS_TEST_TIMEOUT}
            --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
            -- $<TARGET_FILE:${PS_TEST_NAME}>)
        add_dependencies(${PS_TEST_NAME} phaseshift-gpu-test-runner)
        set_tests_properties(${PS_TEST_NAME} PROPERTIES TIMEOUT ${PS_TEST_BACKSTOP})
    else()
        add_test(NAME ${PS_TEST_NAME} COMMAND ${PS_TEST_NAME})
        if(PS_TEST_TIMEOUT)
            set_tests_properties(${PS_TEST_NAME} PROPERTIES TIMEOUT ${PS_TEST_TIMEOUT})
        endif()
    endif()
    set_tests_properties(${PS_TEST_NAME} PROPERTIES LABELS "${PS_TEST_LABELS}")
endfunction()

# ---------------------------------------------------------------------------
# required — self-contained contract / correctness tests
# ---------------------------------------------------------------------------

phaseshift_add_test(NAME test_safetensors_writer SOURCE unit/test_safetensors_writer.cpp LABELS "cpu;required" LIBRARIES phaseshift_io)
phaseshift_add_test(NAME test_fpx_layout SOURCE unit/test_fpx_layout.cpp LABELS "cpu;required" LIBRARIES phaseshift_fpx_format)
phaseshift_add_test(NAME test_int8_compute_contract SOURCE unit/test_int8_compute_contract.cpp LABELS "cpu;required" LIBRARIES phaseshift_quant_reference)
phaseshift_add_test(NAME test_physical_alignment_contract SOURCE unit/test_physical_alignment_contract.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES phaseshift_core phaseshift_qwen35_state)
phaseshift_add_test(NAME test_ocp_e4m3_codec SOURCE unit/test_ocp_e4m3_codec.cpp LABELS "cpu;required" LIBRARIES phaseshift_quant_reference)
phaseshift_add_test(NAME test_psq_payload SOURCE unit/test_psq_payload.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_quant_reference)
phaseshift_add_test(NAME test_mxfp4_codec SOURCE unit/test_mxfp4_codec.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_quant_reference)
phaseshift_add_test(NAME test_mxfp4_payload SOURCE unit/test_mxfp4_payload.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_quant_reference phaseshift_fpx_format)
phaseshift_add_test(NAME test_fp8_block128_payload SOURCE unit/test_fp8_block128_payload.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_quant_reference phaseshift_fpx_format)
phaseshift_add_test(NAME test_tensor_partition SOURCE unit/test_tensor_partition.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_weights)
phaseshift_add_test(NAME test_canonical_partition SOURCE unit/test_canonical_partition.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES phaseshift_weights phaseshift_quant_reference)
phaseshift_add_test(NAME test_qwen35_tensor_parallel_plan SOURCE unit/test_qwen35_tensor_parallel_plan.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35)
phaseshift_add_test(NAME test_qwen35_tp_context SOURCE unit/test_qwen35_tp_context.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35)
phaseshift_add_test(NAME test_tp_reduction SOURCE unit/test_tp_reduction.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_runtime)
target_include_directories(test_tp_reduction PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_tp_transport SOURCE unit/test_tp_transport.cpp LABELS "gpu2;optional" TIMEOUT 120 GPU_COUNT 2 GPU_COST_GB 1 LIBRARIES phaseshift_runtime)
phaseshift_add_test(NAME test_qwen35_tp_execution SOURCE unit/test_qwen35_tp_execution.cpp LABELS "gpu2;optional" TIMEOUT 600 GPU_COUNT 2 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_tp_execution PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_tp_e2e SOURCE unit/test_qwen35_tp_e2e.cpp LABELS "gpu2;optional;external_files" TIMEOUT 2400 GPU_COUNT 2 GPU_COST_GB 20 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_qwen35_tp_weight_load SOURCE unit/test_qwen35_tp_weight_load.cpp LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35)
target_include_directories(test_qwen35_tp_e2e PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_include_directories(test_qwen35_tp_execution PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_weight_load SOURCE unit/test_weight_load.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_weights phaseshift_qwen35)
phaseshift_add_test(NAME test_qwen35_mtp_weight_load SOURCE unit/test_qwen35_mtp_weight_load.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_weights phaseshift_qwen35)
phaseshift_add_test(NAME test_dflash2_config SOURCE unit/test_dflash2_config.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
phaseshift_add_test(NAME test_qwen35_config_tie SOURCE unit/test_qwen35_config_tie.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35)
phaseshift_add_test(NAME test_dflash2_quantization_adapter SOURCE unit/test_dflash2_quantization_adapter.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_quantizer_core)
phaseshift_add_test(NAME test_dflash2_weight_contract SOURCE unit/test_dflash2_weight_contract.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_weights phaseshift_qwen35)
phaseshift_add_test(NAME test_dflash2_feature_concat SOURCE unit/test_dflash2_feature_concat.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_grouped_dynamic_conv SOURCE unit/test_dflash2_grouped_dynamic_conv.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_bf16_geometry SOURCE unit/test_dflash2_bf16_geometry.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_dflash2_bf16_geometry PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_dflash2_rope SOURCE unit/test_dflash2_rope.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_attention SOURCE unit/test_dflash2_attention.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_topk SOURCE unit/test_dflash2_topk.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_gdn_spec_history SOURCE unit/test_gdn_spec_history.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_executor_move_assignment SOURCE unit/test_executor_move_assignment.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_runtime phaseshift_gpu)
phaseshift_add_test(NAME test_dflash2_candidate_selector SOURCE unit/test_dflash2_candidate_selector.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_kv_ring SOURCE unit/test_dflash2_kv_ring.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_attention_ring SOURCE unit/test_dflash2_attention_ring.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_dflash2_psq4_shapes SOURCE unit/test_dflash2_psq4_shapes.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_qwen35_runtime phaseshift_quant_reference phaseshift_gpu)
target_include_directories(test_dflash2_psq4_shapes PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_dflash2_radix_topn SOURCE unit/test_dflash2_radix_topn.hip LABELS "gpu1;required" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_dflash2_radix_topn_perf SOURCE unit/test_dflash2_radix_topn_perf.hip LABELS "gpu1;perf" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 6 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_dflash2_psq8_rerank SOURCE unit/test_dflash2_psq8_rerank.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_quantizer_core phaseshift_gpu)
phaseshift_add_test(NAME test_constraint_candidates SOURCE unit/test_constraint_candidates.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_quantizer_core phaseshift_gpu)
phaseshift_add_test(NAME test_qwen35_mtp_lowering SOURCE unit/test_qwen35_mtp_lowering.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35)
phaseshift_add_test(NAME test_qwen35_lowering_contract SOURCE unit/test_qwen35_lowering_contract.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35)
phaseshift_add_test(NAME test_qwen35_spec_verify SOURCE unit/test_qwen35_spec_verify.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_decode_backend_contract SOURCE unit/test_decode_backend_contract.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_paged_types SOURCE unit/test_paged_types.cpp LABELS "cpu;required" LIBRARIES phaseshift_qwen35_state)
phaseshift_add_test(NAME test_constraint_mask SOURCE unit/test_constraint_mask.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_dflash2_constraint_mask SOURCE unit/test_dflash2_constraint_mask.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_structural_tool_constraint SOURCE unit/test_structural_tool_constraint.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_composite_structural_constraint SOURCE unit/test_composite_structural_constraint.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES phaseshift_qwen35_runtime)
phaseshift_add_test(NAME test_xgrammar_cxx20 SOURCE compile/test_xgrammar_cxx20.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_xgrammar)
phaseshift_add_test(NAME test_tensor SOURCE unit/test_tensor.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 LIBRARIES phaseshift_gpu)
phaseshift_add_test(NAME test_correctness_primitives SOURCE kernels/common/test_correctness_primitives.hip LABELS "gpu1;required" TIMEOUT 30 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels)
phaseshift_add_test(NAME test_architecture_boundaries SOURCE unit/test_architecture_boundaries.cpp LABELS "cpu;required" TIMEOUT 60 LIBRARIES phaseshift_core DEFS PHASESHIFT_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
phaseshift_add_test(NAME test_program_workspace_lifetime SOURCE unit/test_program_workspace_lifetime.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_runtime)
phaseshift_add_test(NAME test_program_block_scaled_lowering SOURCE unit/test_program_block_scaled_lowering.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_runtime)
phaseshift_add_test(NAME test_dispatch_staging_size SOURCE unit/test_dispatch_staging_size.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_runtime)
phaseshift_add_test(NAME test_gpu_arena_vmm SOURCE unit/test_gpu_arena_vmm.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu)
phaseshift_add_test(NAME test_qwen35_prefix_cache SOURCE unit/test_qwen35_prefix_cache.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_runtime phaseshift_qwen35_state phaseshift_gpu)
phaseshift_add_test(NAME test_qwen35_psq_kv_pool SOURCE unit/test_qwen35_psq_kv_pool.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_state phaseshift_gpu)
phaseshift_add_test(NAME test_imatrix_collector SOURCE unit/test_imatrix_collector.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_quantizer_core phaseshift_core)

# 12xx-only kernel tests (psq4/psq8): required on gfx1201.
phaseshift_add_test(NAME test_gemm_bf16_correctness SOURCE kernels/linear/test_gemm_bf16_correctness.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels)
phaseshift_add_test(NAME test_gemm_block_scaled_reference SOURCE kernels/linear/test_gemm_block_scaled_reference.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_quant_reference phaseshift_gpu)
phaseshift_add_test(NAME test_activation_quantize_a8 SOURCE kernels/optimized/test_activation_quantize_a8.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_activation_quantize_e4m3 SOURCE kernels/optimized/test_activation_quantize_e4m3.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_psq4_w4a8_wmma SOURCE kernels/optimized/test_gemm_psq4_w4a8_wmma.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35 phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_psq4_codebook SOURCE kernels/optimized/test_psq4_codebook.hip LABELS "gpu1;required" TIMEOUT 60 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_psq4_decode1 SOURCE kernels/optimized/test_gemm_psq4_decode1.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 6 LIBRARIES phaseshift_qwen35 phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_psq8_w8a8_wmma SOURCE kernels/optimized/test_gemm_psq8_w8a8_wmma.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35 phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_gemm_psq8_decode1 SOURCE kernels/optimized/test_gemm_psq8_decode1.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 6 LIBRARIES phaseshift_qwen35 phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_qwen35_psq_gemm_policy_core SOURCE unit/test_qwen35_psq_gemm_policy_core.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_fp8_block128_optimized SOURCE kernels/linear/test_gemm_fp8_block128_optimized.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_quant_reference phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_mxfp4_optimized SOURCE kernels/linear/test_gemm_mxfp4_optimized.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_quant_reference phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_block_scaled_dispatch SOURCE kernels/linear/test_gemm_block_scaled_dispatch.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_quant_reference phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_bf16_wmma SOURCE kernels/optimized/test_gemm_bf16_wmma.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gemm_bf16_verify_exact SOURCE kernels/optimized/test_gemm_bf16_verify_exact.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_argmax_f32_top2 SOURCE kernels/optimized/test_argmax_f32_top2.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_rmsnorm SOURCE kernels/optimized/test_rmsnorm.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_l2_normalize SOURCE kernels/optimized/test_l2_normalize.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_output_gather SOURCE kernels/optimized/test_output_gather.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_kv_append SOURCE kernels/optimized/test_kv_append.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_paged_attention SOURCE kernels/optimized/test_paged_attention.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_rope SOURCE kernels/optimized/test_rope.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gdn_recurrence SOURCE kernels/optimized/test_gdn_recurrence.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gdn_recurrence_decode1 SOURCE kernels/optimized/test_gdn_recurrence_decode1.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 4 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gdn_recurrence_exact_history SOURCE kernels/optimized/test_gdn_recurrence_exact_history.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 4 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gdn_compact_log_commit SOURCE kernels/optimized/test_gdn_compact_log_commit.hip LABELS "gpu1;optional" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 4 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_gdn_compact_commit_poc SOURCE unit/test_gdn_compact_commit_poc.hip LABELS "gpu1;optional" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_gpu)
phaseshift_add_test(NAME test_gdn_conv1d SOURCE kernels/optimized/test_gdn_conv1d.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_elementwise SOURCE kernels/optimized/test_elementwise.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_embedding SOURCE kernels/optimized/test_embedding.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_argmax SOURCE kernels/optimized/test_argmax.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_qwen35_kernel_mode SOURCE unit/test_qwen35_kernel_mode.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_kernel_mode PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_linear_selector SOURCE unit/test_qwen35_linear_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_linear_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_verify_exact_scope SOURCE unit/test_verify_exact_scope.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_verify_exact_scope PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_rmsnorm_selector SOURCE unit/test_qwen35_rmsnorm_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_rmsnorm_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_l2_normalize_selector SOURCE unit/test_qwen35_l2_normalize_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_l2_normalize_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_kv_append_selector SOURCE unit/test_qwen35_kv_append_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_kv_append_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_rope_selector SOURCE unit/test_qwen35_rope_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_rope_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_paged_attention_selector SOURCE unit/test_qwen35_paged_attention_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_paged_attention_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_paged_attention_split_reduce_exact SOURCE unit/test_paged_attention_split_reduce_exact.hip LABELS "gpu1;required" TIMEOUT 180 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized)
phaseshift_add_test(NAME test_qwen35_gdn_recurrence_selector SOURCE unit/test_qwen35_gdn_recurrence_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_gdn_recurrence_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_gdn_conv_selector SOURCE unit/test_qwen35_gdn_conv_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_gdn_conv_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_elementwise_selector SOURCE unit/test_qwen35_elementwise_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_elementwise_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_embedding_selector SOURCE unit/test_qwen35_embedding_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_embedding_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_sampling_selector SOURCE unit/test_qwen35_sampling_selector.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_sampling_selector PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_sampling_params SOURCE unit/test_qwen35_sampling_params.cpp LABELS "cpu;required" TIMEOUT 30 LIBRARIES phaseshift_qwen35_runtime)
target_include_directories(test_qwen35_sampling_params PRIVATE "${CMAKE_SOURCE_DIR}/src")
phaseshift_add_test(NAME test_qwen35_sampling_rng SOURCE unit/test_qwen35_sampling_rng.hip LABELS "gpu1;required" TIMEOUT 120 GPU_COUNT 1 GPU_COST_GB 1 LIBRARIES phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_stochastic_sampling SOURCE kernels/optimized/test_stochastic_sampling.hip LABELS "gpu1;required" TIMEOUT 300 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)
phaseshift_add_test(NAME test_stochastic_topk SOURCE kernels/optimized/test_stochastic_topk.hip LABELS "gpu1;required" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 3 LIBRARIES phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_gpu)


# Default required build unit. Required acceptance builds only this target.
add_custom_target(
    phaseshift-required-tests
    DEPENDS
        test_safetensors_writer
        test_fpx_layout
        test_int8_compute_contract
        test_physical_alignment_contract
        test_ocp_e4m3_codec
        test_psq_payload
        test_mxfp4_codec
        test_mxfp4_payload
        test_fp8_block128_payload
        test_tensor_partition
        test_canonical_partition
        test_qwen35_tensor_parallel_plan
        test_qwen35_tp_context
        test_tp_reduction
        test_weight_load
        test_qwen35_tp_weight_load
        test_qwen35_mtp_weight_load
        test_dflash2_config
        test_qwen35_config_tie
        test_dflash2_quantization_adapter
        test_dflash2_weight_contract
        test_dflash2_feature_concat
        test_dflash2_grouped_dynamic_conv
        test_dflash2_bf16_geometry
        test_dflash2_rope
        test_dflash2_attention
        test_dflash2_topk
        test_gdn_spec_history
        test_executor_move_assignment
        test_dflash2_candidate_selector
        test_dflash2_kv_ring
        test_dflash2_attention_ring
        test_dflash2_psq4_shapes
        test_dflash2_radix_topn
        test_dflash2_psq8_rerank
        test_constraint_candidates
        test_paged_types
        test_paged_attention_split_reduce_exact
        test_tensor
        test_correctness_primitives
        test_gemm_bf16_correctness
        test_gemm_block_scaled_reference
        test_activation_quantize_a8
        test_activation_quantize_e4m3
        test_gemm_psq4_w4a8_wmma
        test_psq4_codebook
        test_gemm_psq4_decode1
        test_gemm_psq8_w8a8_wmma
        test_gemm_psq8_decode1
        test_qwen35_psq_gemm_policy_core
        test_gemm_fp8_block128_optimized
        test_gemm_mxfp4_optimized
        test_gemm_block_scaled_dispatch
        test_gemm_bf16_wmma
        test_rmsnorm
        test_l2_normalize
        test_output_gather
        test_kv_append
        test_paged_attention
        test_rope
        test_gdn_recurrence
        test_gdn_recurrence_decode1
        test_gdn_recurrence_exact_history
        test_gdn_conv1d
        test_elementwise
        test_embedding
        test_argmax
        test_qwen35_kernel_mode
        test_qwen35_linear_selector
        test_qwen35_rmsnorm_selector
        test_qwen35_l2_normalize_selector
        test_qwen35_kv_append_selector
        test_qwen35_rope_selector
        test_qwen35_paged_attention_selector
        test_qwen35_gdn_recurrence_selector
        test_qwen35_gdn_conv_selector
        test_qwen35_elementwise_selector
        test_qwen35_embedding_selector
        test_qwen35_sampling_selector
        test_qwen35_sampling_params
        test_qwen35_sampling_rng
        test_stochastic_sampling
        test_stochastic_topk
        test_qwen35_mtp_lowering
        test_qwen35_lowering_contract
        test_qwen35_spec_verify
        test_decode_backend_contract
        test_architecture_boundaries
        test_qwen35_prefix_cache
        test_qwen35_psq_kv_pool
        test_imatrix_collector
        test_gemm_bf16_verify_exact
        test_argmax_f32_top2
        test_verify_exact_scope
        test_program_workspace_lifetime
        test_program_block_scaled_lowering
        test_dispatch_staging_size
        test_gpu_arena_vmm
        test_constraint_mask
        test_dflash2_constraint_mask
        test_structural_tool_constraint
        test_composite_structural_constraint
        test_xgrammar_cxx20
        phaseshift-compute
)

# ---------------------------------------------------------------------------
# phaseshift-bench CLI help smoke (CPU-only: no GPU, no model).
# --help must exit 0 before any hipSetDevice / model load / file load.
# ---------------------------------------------------------------------------
if(PHASESHIFT_BUILD_BENCHMARKS)
    add_test(NAME test_bench_help COMMAND $<TARGET_FILE:phaseshift-bench> --help)
    set_tests_properties(test_bench_help PROPERTIES LABELS "cpu;required")

    foreach(_cmd
            pp
            tg
            activation-quantize
            gemm
            rmsnorm
            l2-normalize
            kv-append
            paged-attention
            rope
            gdn-recurrence
            gdn-conv1d
            elementwise
            embedding
            sampling
            tp-reduce
            gpu-memory)
        string(REPLACE "-" "_" _bench_name ${_cmd})
        add_test(NAME test_bench_help_${_bench_name}
                 COMMAND $<TARGET_FILE:phaseshift-bench> ${_cmd} --help)
        set_tests_properties(test_bench_help_${_bench_name} PROPERTIES LABELS "cpu;required")
    endforeach()

    foreach(_bad wmma direct all)
        add_test(NAME test_bench_gemm_reject_${_bad}
                 COMMAND $<TARGET_FILE:phaseshift-bench> gemm --variant ${_bad})
        set_tests_properties(test_bench_gemm_reject_${_bad} PROPERTIES
                             LABELS "cpu;required" WILL_FAIL TRUE)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# phaseshift-compute decode backend contract (CPU-only: no GPU, no model).
# --decode-backend must be decided before any model load: host keeps reaching
# the ordinary startup path.
# ---------------------------------------------------------------------------
add_test(NAME test_compute_decode_backend_host
         COMMAND $<TARGET_FILE:phaseshift-compute> --decode-backend host)
set_tests_properties(test_compute_decode_backend_host PROPERTIES
                     LABELS "cpu;required"
                     PASS_REGULAR_EXPRESSION "model-dir required")

# ---------------------------------------------------------------------------
# Qwen3.5-4B full application E2E.
# OPTIONAL + EXTERNAL_FILES: requires the external 4B model at
# PHASESHIFT_MODEL_DIR_4B and the committed HF/PyTorch oracle fixture.
# ---------------------------------------------------------------------------

if(PHASESHIFT_BUILD_OPTIONAL_TESTS)
    if(NOT PHASESHIFT_BUILD_BENCHMARKS)
        message(
            FATAL_ERROR
            "Qwen3.5-4B E2E requires PHASESHIFT_BUILD_BENCHMARKS=ON"
        )
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_custom_target(
        phaseshift-e2e-apps
        DEPENDS
            phaseshift-compute
            phaseshift-cli-stage
            phaseshift-quantizer
            phaseshift-bench
            phaseshift-gpu-test-runner
    )

    add_test(
        NAME test_apps_qwen35_4b_inference_e2e
        COMMAND phaseshift-gpu-test-runner
            --gpu-count 1
            --cost-gb 24
            --timeout 1800
            --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
            --
            "${Python3_EXECUTABLE}"
            "${_PS_TESTS_ROOT}/e2e/test_apps_qwen35_4b.py"
            --suite inference
            --model-dir "${PHASESHIFT_MODEL_DIR_4B}"
            --oracle "${_PS_TESTS_ROOT}/e2e/fixtures/qwen35_4b_oracle.json"
            --compute "$<TARGET_FILE:phaseshift-compute>"
            --cli "${PHASESHIFT_CLI_TARGET}"
            --bench "$<TARGET_FILE:phaseshift-bench>"
            --quantizer "$<TARGET_FILE:phaseshift-quantizer>"
    )
    set_tests_properties(test_apps_qwen35_4b_inference_e2e PROPERTIES
        TIMEOUT 2760
        LABELS "acceptance;gpu1;optional;external_files;e2e")

    add_test(
        NAME test_apps_qwen35_4b_quantizer_e2e
        COMMAND phaseshift-gpu-test-runner
            --gpu-count 1
            --cost-gb 24
            --timeout 7200
            --state-dir "${CMAKE_BINARY_DIR}/gpu-test-state"
            --
            "${Python3_EXECUTABLE}"
            "${_PS_TESTS_ROOT}/e2e/test_apps_qwen35_4b.py"
            --suite quantizer
            --model-dir "${PHASESHIFT_MODEL_DIR_4B}"
            --oracle "${_PS_TESTS_ROOT}/e2e/fixtures/qwen35_4b_oracle.json"
            --compute "$<TARGET_FILE:phaseshift-compute>"
            --cli "${PHASESHIFT_CLI_TARGET}"
            --bench "$<TARGET_FILE:phaseshift-bench>"
            --quantizer "$<TARGET_FILE:phaseshift-quantizer>"
    )
    set_tests_properties(test_apps_qwen35_4b_quantizer_e2e PROPERTIES
        TIMEOUT 8160
        LABELS "acceptance;gpu1;optional;external_files;e2e;long")

    # MTP weight contract against a real model directory.
    # Model dir via PHASESHIFT_MODEL_DIR_MTP (preferred), PHASESHIFT_MODEL_DIR_4B,
    # or PHASESHIFT_MODEL_DIR_4B_PSQ. Skips (exit 77) when none is set.
    phaseshift_add_test(NAME test_qwen35_mtp_weight_real SOURCE unit/test_qwen35_mtp_weight_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 16 LIBRARIES phaseshift_weights phaseshift_qwen35)

    # DFlash2 drafter weight contract against a real model directory.
    # Model dir via PHASESHIFT_MODEL_DIR_DFLASH2. Skips (exit 77) when unset.
    phaseshift_add_test(NAME test_dflash2_weight_real SOURCE unit/test_dflash2_weight_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 600 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35)

    # DFlash2 Gate 3: target feature projection and grouped dynamic conv.
    # Fixture dir via PHASESHIFT_DFLASH2_GATE3_FIXTURE (default
    # build/dflash2-gate3-reference). Skips (exit 77) when unset or missing.
    phaseshift_add_test(NAME test_dflash2_gate3_real SOURCE unit/test_dflash2_gate3_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 3 per-op timing (not a correctness gate).
    phaseshift_add_test(NAME test_dflash2_gate3_perf SOURCE unit/test_dflash2_gate3_perf.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate3_perf PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 4: layer 0 standalone forward against the official reference.
    # Fixture dir via PHASESHIFT_DFLASH2_GATE4_FIXTURE (default
    # build/dflash2-gate4-reference). Skips (exit 77) when unset or missing.
    phaseshift_add_test(NAME test_dflash2_gate4_real SOURCE unit/test_dflash2_gate4_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 4 per-op timing (not a correctness gate).
    phaseshift_add_test(NAME test_dflash2_gate4_perf SOURCE unit/test_dflash2_gate4_perf.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate4_perf PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 5: full 5-layer stateless backbone + final norm.
    # Fixture dir via PHASESHIFT_DFLASH2_GATE5_FIXTURE (default
    # build/dflash2-gate5-reference). Skips (exit 77) when unset or missing.
    phaseshift_add_test(NAME test_dflash2_gate5_real SOURCE unit/test_dflash2_gate5_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 1200 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    phaseshift_add_test(NAME test_dflash2_gate6_perf SOURCE unit/test_dflash2_gate6_perf.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate6_perf PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 6: target lm_head + top16 + candidate selector.
    # Fixture dir via PHASESHIFT_DFLASH2_GATE6_FIXTURE (default
    # build/dflash2-gate6-reference). Needs the 27B-PSQ target for lm_head.
    phaseshift_add_test(NAME test_dflash2_gate6_real SOURCE unit/test_dflash2_gate6_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 7: persistent context KV ring (append / cached backbone / proposer).
    # Fixture dir via PHASESHIFT_DFLASH2_GATE7_FIXTURE (default
    # build/dflash2-gate7-reference). Needs the 27B-PSQ target for lm_head.
    phaseshift_add_test(NAME test_dflash2_gate7_real SOURCE unit/test_dflash2_gate7_real.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    phaseshift_add_test(NAME test_dflash2_gate7_perf SOURCE unit/test_dflash2_gate7_perf.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate7_perf PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 8: live target bridge (prompt prefill -> taps -> ring -> proposal).
    phaseshift_add_test(NAME test_dflash2_gate8_live SOURCE unit/test_dflash2_gate8_live.hip LABELS "gpu1;optional;external_files" TIMEOUT 2400 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 9: greedy speculative decode E2E vs target-only greedy.
    phaseshift_add_test(NAME test_dflash2_gate9_e2e SOURCE unit/test_dflash2_gate9_e2e.hip LABELS "gpu1;optional;external_files" TIMEOUT 3600 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 10: phaseshift-compute speculative decoding integration.
    phaseshift_add_test(NAME test_dflash2_gate10_cli SOURCE unit/test_dflash2_gate10_cli.hip LABELS "gpu1;optional;external_files" TIMEOUT 3600 GPU_COUNT 1 GPU_COST_GB 30)

    # DFlash2 Gate 11A: per-kernel profile baseline.
    phaseshift_add_test(NAME test_dflash2_gate11_profile SOURCE unit/test_dflash2_gate11_profile.hip LABELS "gpu1;optional;external_files" TIMEOUT 3600 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate11_profile PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 11B: standalone top16 optimization.
    phaseshift_add_test(NAME test_dflash2_gate11b_topk SOURCE unit/test_dflash2_gate11b_topk.hip LABELS "gpu1;optional" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 2 LIBRARIES phaseshift_qwen35_kernels_optimized)

    # DFlash2 Gate 11C: GDN history e2e parity and target verify capture overhead.
    phaseshift_add_test(NAME test_dflash2_gate11c_e2e SOURCE unit/test_dflash2_gate11c_e2e.hip LABELS "gpu1;optional;external_files" TIMEOUT 3600 GPU_COUNT 1 GPU_COST_GB 30)
    phaseshift_add_test(NAME test_dflash2_gate11c_perf SOURCE unit/test_dflash2_gate11c_perf.hip LABELS "gpu1;optional;external_files" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    phaseshift_add_test(NAME test_dflash2_gate11d_verify_profile SOURCE unit/test_dflash2_gate11d_verify_profile.hip LABELS "gpu1;optional;external_files" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 11H: target verify exact primitive ceiling profile.
    phaseshift_add_test(NAME test_dflash2_gate11h_verify_profile SOURCE unit/test_dflash2_gate11h_verify_profile.hip LABELS "gpu1;optional;external_files" TIMEOUT 3600 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate11h_verify_profile PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 5.1: dump real target taps for a real-input fixture.
    phaseshift_add_test(NAME test_dflash2_gate51_target_fixture SOURCE unit/test_dflash2_gate51_target_fixture.hip LABELS "gpu1;optional;external_files" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 30 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_gate51_target_fixture PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # DFlash2 Gate 5.1 diagnostic: dump every stage buffer for all 5 layers.
    phaseshift_add_test(NAME test_dflash2_gate51_trace SOURCE unit/test_dflash2_gate51_trace.hip LABELS "gpu1;optional;external_files" TIMEOUT 1200 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 Gate 5 per-layer / full-backbone timing (not a correctness gate).
    phaseshift_add_test(NAME test_dflash2_gate5_perf SOURCE unit/test_dflash2_gate5_perf.hip LABELS "gpu1;optional;external_files" TIMEOUT 900 GPU_COUNT 1 GPU_COST_GB 8 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)

    # DFlash2 target hidden taps: external tap == internal layer output.
    # Target model dir via PHASESHIFT_MODEL_DIR_DFLASH2_TARGET (skips when unset);
    # tap layer ids come from PHASESHIFT_MODEL_DIR_DFLASH2 when available.
    phaseshift_add_test(NAME test_dflash2_target_taps SOURCE unit/test_dflash2_target_taps.hip LABELS "gpu1;optional;external_files" TIMEOUT 1800 GPU_COUNT 1 GPU_COST_GB 24 LIBRARIES phaseshift_weights phaseshift_qwen35 phaseshift_qwen35_runtime)
    target_include_directories(test_dflash2_target_taps PRIVATE "${CMAKE_SOURCE_DIR}/src")

    # Target lm_head certified proxy: real hidden probe (num_output_rows large).

    # Target lm_head certified proxy: real hidden shadow (pool / margin sweep).

    # Target lm_head certified proxy: isolated real-head performance (Gate 6/7).



    # Model dir via PHASESHIFT_MODEL_DIR_MTP or PHASESHIFT_MODEL_DIR_4B.

endif()
