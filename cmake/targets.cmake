# PhaseShift production / library targets.
#
# Included from root CMakeLists.txt. Defines the library targets and their
# dependency graph.

# Vendored XGrammar v0.2.5 (pinned, see vendor/xgrammar/VERSION). Built as a
# native C++ static library without Python bindings or CUDA kernels.
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/xgrammar"
    "${CMAKE_CURRENT_BINARY_DIR}/xgrammar"
)

# Base INTERFACE target: public include root.
add_library(phaseshift_core INTERFACE)
target_compile_features(phaseshift_core INTERFACE cxx_std_20)
target_include_directories(
    phaseshift_core
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# GPU abstraction.
add_library(phaseshift_gpu STATIC
    src/phaseshift/core/memory/tensor.cpp
    src/phaseshift/core/gpu/scoped_device.cpp
    src/phaseshift/core/memory/arena.hip
)
target_compile_features(phaseshift_gpu PRIVATE cxx_std_20)
target_include_directories(phaseshift_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_gpu PUBLIC phaseshift_core hip::host)
target_compile_options(phaseshift_gpu PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_gpu)

# File I/O.
add_library(phaseshift_io STATIC
    src/phaseshift/io/safetensors_reader.cpp
    src/phaseshift/io/safetensors_writer.cpp
)

target_compile_features(phaseshift_io PRIVATE cxx_std_20)
target_include_directories(phaseshift_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_io SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_io PUBLIC phaseshift_core)
target_compile_options(phaseshift_io PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Generic weight loading (model-independent): safetensors / quantized
# safetensors -> GPU MatrixWeight / Tensor. Depends on format + quantization
# layers, never on models/.
add_library(phaseshift_weights STATIC
    src/phaseshift/weights/weight_loader.cpp
)
target_compile_features(phaseshift_weights PRIVATE cxx_std_20)
target_include_directories(phaseshift_weights PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_weights SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_weights PUBLIC
    phaseshift_core
    phaseshift_gpu
    phaseshift_io
    phaseshift_fpx_format
    phaseshift_quant_reference
)
target_compile_options(phaseshift_weights PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.5 correctness kernels. One model reference executor = one translation
# unit. model_dispatch_correctness.hip is the single __global__ dispatch and
# includes detail/*.inc (elementwise / normalization / linear / attention / gdn
# / embedding / output) as source fragments. standalone/ keeps the per-op
# reference kernels used by kernel-level tests.
add_library(phaseshift_qwen35_kernels STATIC
    src/phaseshift/models/qwen35/kernels/correctness/model_dispatch_correctness.hip
    src/phaseshift/models/qwen35/kernels/correctness/standalone/rmsnorm_correctness.hip
    src/phaseshift/models/qwen35/kernels/correctness/standalone/silu_correctness.hip
    src/phaseshift/models/qwen35/kernels/correctness/standalone/swiglu_correctness.hip
    src/phaseshift/models/qwen35/kernels/correctness/standalone/gemm_bf16_correctness.hip
    src/phaseshift/models/qwen35/kernels/correctness/standalone/gemm_block_scaled_correctness.hip
)
target_compile_features(phaseshift_qwen35_kernels PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_qwen35_kernels PUBLIC phaseshift_gpu phaseshift_runtime phaseshift_qwen35_state)
target_compile_options(phaseshift_qwen35_kernels PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_qwen35_kernels)

# Qwen3.5 optimized kernels. One source per implementation family (the same
# optimization problem), not per KernelId or dtype. Public launchers live in the
# matching optimized/*.h and select dtype/algorithm variants internally.
set(PHASESHIFT_QWEN35_OPTIMIZED_SOURCES
    src/phaseshift/models/qwen35/kernels/optimized/linear/bf16.hip
    src/phaseshift/models/qwen35/kernels/optimized/linear/psq4.hip
    src/phaseshift/models/qwen35/kernels/optimized/linear/psq8.hip
    src/phaseshift/models/qwen35/kernels/optimized/linear/fp8_block128.hip
    src/phaseshift/models/qwen35/kernels/optimized/linear/mxfp4.hip
    src/phaseshift/models/qwen35/kernels/optimized/attention/paged_decode.hip
    src/phaseshift/models/qwen35/kernels/optimized/attention/paged_prefill.hip
    src/phaseshift/models/qwen35/kernels/optimized/attention/rope.hip
    src/phaseshift/models/qwen35/kernels/optimized/embedding.hip
    src/phaseshift/models/qwen35/kernels/optimized/elementwise.hip
    src/phaseshift/models/qwen35/kernels/optimized/rmsnorm.hip
    src/phaseshift/models/qwen35/kernels/optimized/l2_normalize.hip
    src/phaseshift/models/qwen35/kernels/optimized/output_gather.hip
    src/phaseshift/models/qwen35/kernels/optimized/activation_quantize.hip
    src/phaseshift/models/qwen35/kernels/optimized/kv_append.hip
    src/phaseshift/models/qwen35/kernels/optimized/gdn/recurrence.hip
    src/phaseshift/models/qwen35/kernels/optimized/gdn/conv1d.hip
    src/phaseshift/models/qwen35/kernels/optimized/sampling.hip
    src/phaseshift/models/qwen35/kernels/optimized/stochastic_sampling.hip
    src/phaseshift/models/qwen35/kernels/dflash2/candidate_selector.hip
    src/phaseshift/models/qwen35/kernels/dflash2/feature_concat.hip
    src/phaseshift/models/qwen35/kernels/dflash2/grouped_dynamic_conv.hip
    src/phaseshift/models/qwen35/kernels/dflash2/rope.hip
    src/phaseshift/models/qwen35/kernels/dflash2/rmsnorm.hip
    src/phaseshift/models/qwen35/kernels/dflash2/topk.hip
    src/phaseshift/models/qwen35/kernels/dflash2/topk_optimized.hip
    src/phaseshift/models/qwen35/kernels/dflash2/draft_head_int2.hip
    src/phaseshift/models/qwen35/kernels/dflash2/coarse_topn.hip
    src/phaseshift/models/qwen35/kernels/dflash2/swiglu.hip
    src/phaseshift/models/qwen35/kernels/dflash2/attention.hip
    src/phaseshift/models/qwen35/kernels/dflash2/kv_ring.hip
    src/phaseshift/models/qwen35/kernels/dflash2/noise_input.hip
)

add_library(phaseshift_qwen35_kernels_optimized STATIC ${PHASESHIFT_QWEN35_OPTIMIZED_SOURCES})
target_compile_features(phaseshift_qwen35_kernels_optimized PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_kernels_optimized PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35_kernels_optimized PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_qwen35_kernels_optimized PUBLIC phaseshift_gpu phaseshift_runtime phaseshift_qwen35_state)
target_compile_options(phaseshift_qwen35_kernels_optimized PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_qwen35_kernels_optimized)

# Qwen3.5 paged sequence state management.
add_library(phaseshift_runtime STATIC
    src/phaseshift/runtime/batch/device_batch_context.hip
    src/phaseshift/runtime/graph/primitive_graph.cpp
    src/phaseshift/runtime/stream_bridge.cpp
    src/phaseshift/runtime/program/program.cpp
)
target_compile_features(phaseshift_runtime PRIVATE cxx_std_20)
target_include_directories(phaseshift_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_runtime PUBLIC phaseshift_core phaseshift_gpu)
target_compile_options(phaseshift_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_runtime)

add_library(phaseshift_qwen35_state STATIC
    src/phaseshift/models/qwen35/state/paged_sequence_state.cpp
    src/phaseshift/models/qwen35/state/sequence_slot_pool.cpp
    src/phaseshift/models/qwen35/state/gdn_state_pool.cpp
    src/phaseshift/models/qwen35/state/paged_kv_pool.cpp
)
target_compile_features(phaseshift_qwen35_state PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_state PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_qwen35_state PUBLIC phaseshift_core phaseshift_gpu)
target_compile_options(phaseshift_qwen35_state PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.5 model sources.
set(
    PHASESHIFT_QWEN35_SOURCES
    src/phaseshift/models/qwen35/weights/model_weights.cpp
    src/phaseshift/models/qwen35/model/qwen35_config.cpp
    src/phaseshift/models/qwen35/model/qwen35_model.cpp
    src/phaseshift/models/qwen35/model/lower_to_primitives.cpp
    src/phaseshift/models/qwen35/dflash2/config.cpp
    src/phaseshift/models/qwen35/dflash2/weights.cpp
    src/phaseshift/models/qwen35/dflash2/context_state.hip
)

add_library(phaseshift_qwen35 STATIC ${PHASESHIFT_QWEN35_SOURCES})
target_compile_features(phaseshift_qwen35 PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(phaseshift_qwen35 SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_qwen35 PUBLIC phaseshift_weights phaseshift_io phaseshift_qwen35_state phaseshift_quant_reference phaseshift_fpx_format phaseshift_runtime)
target_compile_options(phaseshift_qwen35 PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Qwen3.5 continuous batching runtime.
add_library(phaseshift_qwen35_runtime STATIC
    src/phaseshift/models/qwen35/runtime/runtime_request.cpp
    src/phaseshift/models/qwen35/runtime/token_constraint.cpp
    src/phaseshift/models/qwen35/runtime/sampling_params.cpp
    src/phaseshift/models/qwen35/runtime/token_budget_scheduler.cpp
    src/phaseshift/models/qwen35/runtime/kv_capacity_manager.cpp
    src/phaseshift/models/qwen35/runtime/kv_banker.cpp
    src/phaseshift/models/qwen35/runtime/continuous_batcher.cpp
    src/phaseshift/models/qwen35/runtime/prefix_cache.cpp
    src/phaseshift/models/qwen35/runtime/program_executor.hip
    src/phaseshift/models/qwen35/runtime/executor.hip
    src/phaseshift/models/qwen35/runtime/mtp_kv_state.cpp
    src/phaseshift/models/qwen35/runtime/mtp_executor.hip
    src/phaseshift/models/qwen35/runtime/spec_decode.cpp
    src/phaseshift/models/qwen35/runtime/spec_decoder.cpp
    src/phaseshift/models/qwen35/runtime/dflash2_spec_decoder.cpp
    src/phaseshift/models/qwen35/runtime/gdn_spec_history.hip
    src/phaseshift/models/qwen35/runtime/optimized_dispatch.hip
    src/phaseshift/models/qwen35/runtime/kv_append_dispatch.hip
    src/phaseshift/models/qwen35/runtime/kv_calib_dump.cpp
    src/phaseshift/models/qwen35/runtime/paged_attention_dispatch.hip
    src/phaseshift/models/qwen35/runtime/scheduled_batch.cpp
    src/phaseshift/models/qwen35/runtime/decode_backend.cpp
    src/phaseshift/models/qwen35/runtime/linear_selector.cpp
    src/phaseshift/models/qwen35/runtime/physical_launch.hip
    src/phaseshift/models/qwen35/runtime/rmsnorm_selector.cpp
    src/phaseshift/models/qwen35/runtime/l2_normalize_selector.cpp
    src/phaseshift/models/qwen35/runtime/l2_normalize_dispatch.hip
    src/phaseshift/models/qwen35/runtime/kv_append_selector.cpp
    src/phaseshift/models/qwen35/runtime/paged_attention_selector.cpp
    src/phaseshift/models/qwen35/runtime/gdn_recurrence_dispatch.hip
    src/phaseshift/models/qwen35/runtime/gdn_recurrence_selector.cpp
    src/phaseshift/models/qwen35/runtime/gdn_conv_selector.cpp
    src/phaseshift/models/qwen35/runtime/gdn_conv_dispatch.hip
    src/phaseshift/models/qwen35/runtime/rope_selector.cpp
    src/phaseshift/models/qwen35/runtime/rope_dispatch.hip
    src/phaseshift/models/qwen35/runtime/elementwise_selector.cpp
    src/phaseshift/models/qwen35/runtime/elementwise_dispatch.hip
    src/phaseshift/models/qwen35/runtime/embedding_selector.cpp
    src/phaseshift/models/qwen35/runtime/embedding_dispatch.hip
    src/phaseshift/models/qwen35/runtime/sampling_selector.cpp
    src/phaseshift/models/qwen35/runtime/sampling_dispatch.hip
    src/phaseshift/models/qwen35/runtime/decode_perf_stats.cpp
    src/phaseshift/models/qwen35/runtime/lm_head_proxy.hip
    src/phaseshift/models/qwen35/dflash2/executor.hip
)
target_compile_features(phaseshift_qwen35_runtime PRIVATE cxx_std_20)
target_include_directories(phaseshift_qwen35_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_qwen35_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phaseshift_qwen35_runtime PUBLIC phaseshift_qwen35 phaseshift_qwen35_state phaseshift_qwen35_kernels phaseshift_qwen35_kernels_optimized phaseshift_runtime phaseshift_gpu phaseshift_xgrammar)
target_compile_options(phaseshift_qwen35_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_qwen35_runtime)
if(PHASESHIFT_PA_PROBE)
    target_compile_definitions(phaseshift_qwen35_runtime PRIVATE PHASESHIFT_PA_PROBE=1)
endif()
if(PHASESHIFT_HIP_GRAPH)
    set_source_files_properties(
        src/phaseshift/models/qwen35/runtime/program_executor.hip
        src/phaseshift/models/qwen35/runtime/executor.hip
        PROPERTIES COMPILE_DEFINITIONS "PHASESHIFT_HIP_GRAPH")
endif()


# Aggregate INTERFACE target. Tests and benchmarks link against this.
add_library(phaseshift INTERFACE)
target_link_libraries(
    phaseshift
    INTERFACE
        phaseshift_core
        phaseshift_gpu
        phaseshift_io
        phaseshift_weights
        phaseshift_qwen35_kernels
        phaseshift_qwen35_kernels_optimized
        phaseshift_runtime
        phaseshift_qwen35_state
        phaseshift_qwen35
        phaseshift_qwen35_runtime
)

# FPX format definitions (enum/profile/layout/bundle). Linked into the Qwen3.5
# runtime via phaseshift_qwen35 (quantized safetensors load-time contract).
add_library(phaseshift_fpx_format STATIC
    src/phaseshift/quantization/fpx/profile.cpp
    src/phaseshift/quantization/fpx/crc32.cpp
    src/phaseshift/quantization/fpx/quantized_manifest.cpp
    src/phaseshift/quantization/fpx/quantized_model_reader.cpp
    src/phaseshift/quantization/fpx/quantize_source.cpp
)
target_compile_features(phaseshift_fpx_format PRIVATE cxx_std_20)
target_include_directories(phaseshift_fpx_format PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_fpx_format SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_fpx_format PUBLIC phaseshift_core phaseshift_io OpenSSL::Crypto)
target_compile_options(phaseshift_fpx_format PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# FPX reference quantizers (pure CPU math; independent of model_format).
add_library(phaseshift_quant_reference STATIC
    src/phaseshift/quantization/fpx/e4m3.cpp
    src/phaseshift/quantization/fpx/ue4m3.cpp
    src/phaseshift/quantization/psq/psq.cpp
    src/phaseshift/quantization/psq/quant_canonical.cpp
    src/phaseshift/quantization/psq/quant_preshuffle.cpp
    src/phaseshift/quantization/mxfp4/mxfp4.cpp
    src/phaseshift/quantization/fp8/block128.cpp
    src/phaseshift/quantization/fp8/block128_native.cpp
    src/phaseshift/quantization/mxfp4/mxfp4_native.cpp
)
set_source_files_properties(
    src/phaseshift/quantization/psq/psq.cpp
    src/phaseshift/quantization/mxfp4/mxfp4.cpp
    src/phaseshift/quantization/fp8/block128.cpp
    src/phaseshift/quantization/fp8/block128_native.cpp
    src/phaseshift/quantization/mxfp4/mxfp4_native.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_compile_features(phaseshift_quant_reference PRIVATE cxx_std_20)
target_include_directories(phaseshift_quant_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_quant_reference PUBLIC phaseshift_core)
target_compile_options(phaseshift_quant_reference PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

add_library(phaseshift_kld_primitives STATIC
    src/phaseshift/quantization/offline/kld_metrics.cpp
    src/phaseshift/quantization/offline/kld_logit_cache.cpp
)
target_compile_features(phaseshift_kld_primitives PRIVATE cxx_std_20)
target_include_directories(phaseshift_kld_primitives PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(phaseshift_kld_primitives PUBLIC phaseshift_core)
target_compile_options(phaseshift_kld_primitives PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Offline quantization core (quantizer + iMatrix format + model fingerprint).
# Shared by the phaseshift-quantizer app and optional quantizer tests.
# NOT linked into the phaseshift runtime aggregate.
add_library(phaseshift_quantizer_core STATIC
    src/phaseshift/quantization/offline/quantizer.cpp
    src/phaseshift/quantization/offline/qwen35_adapter.cpp
    src/phaseshift/quantization/offline/dflash2_adapter.cpp
    src/phaseshift/quantization/offline/adapter_dispatch.cpp
    src/phaseshift/quantization/offline/shard_resolver.cpp
    src/phaseshift/quantization/offline/gpu_quantizer.hip
    src/phaseshift/quantization/offline/imatrix_format.cpp
    src/phaseshift/quantization/offline/imatrix_collect.cpp
    src/phaseshift/quantization/offline/ppl.cpp
    src/phaseshift/quantization/offline/model_fingerprint.cpp
    src/phaseshift/quantization/offline/token_corpus.cpp
    src/phaseshift/quantization/imatrix/gpu_collector.hip
    src/phaseshift/quantization/imatrix/model_sites.cpp
    src/phaseshift/quantization/offline/gpu_kld.hip
    src/phaseshift/quantization/offline/kld_shadow_model.cpp
    src/phaseshift/quantization/offline/kld.cpp
)
target_compile_features(phaseshift_quantizer_core PRIVATE cxx_std_20)
target_include_directories(phaseshift_quantizer_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_quantizer_core SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_link_libraries(phaseshift_quantizer_core PUBLIC
    phaseshift_fpx_format
    phaseshift_quant_reference
    phaseshift_io
    phaseshift_kld_primitives
    phaseshift_qwen35
    phaseshift_qwen35_runtime
    phaseshift_core
    hip::host
)
target_link_libraries(phaseshift_quantizer_core PRIVATE OpenSSL::Crypto)
target_compile_options(phaseshift_quantizer_core PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_quantizer_core)
# FPX weighted quantization must reproduce the CPU reference's non-fused FP32
# arithmetic bit-for-bit. AMDGPU's default fp-contract fuses mul+add into FMA
# which can flip 1-ULP scale-selection ties; disable contraction for this TU.
set_source_files_properties(src/phaseshift/quantization/offline/gpu_quantizer.hip PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off")

find_package(OpenSSL REQUIRED)

