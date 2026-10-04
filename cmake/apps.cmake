# PhaseShift application entrypoints.
#
# The five user-facing executables:
#   phaseshift-compute   raw token ID in/out inference engine
#   phaseshift-cli       chat UI (Python, tokenizer + chat template)
#   phaseshift-server    OpenAI-compatible serving product (LocalAI frontend)
#   phaseshift-quantizer quantize / verify / convert / validate
#   phaseshift-bench     E2E benchmark (pp / tg) + GEMM kernel micro-benchmark

# Inference engine (raw token IDs in, GENERATED_IDS out).
add_executable(phaseshift-compute
    src/apps/compute/main.hip
    src/apps/compute/compute_runtime.hip
)
phaseshift_set_rocm_rpath(phaseshift-compute)
target_compile_features(phaseshift-compute PRIVATE cxx_std_20)
target_include_directories(phaseshift-compute SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
target_include_directories(phaseshift-compute PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(phaseshift-compute PRIVATE phaseshift)
phaseshift_set_hip_archs(phaseshift-compute)

# Quantizer: quantize / verify / convert / validate subcommands.
add_executable(phaseshift-quantizer
    src/apps/quantizer/main.cpp
    src/apps/quantizer/commands/quantize.cpp
    src/apps/quantizer/commands/kld.cpp
    src/apps/quantizer/commands/imatrix.cpp
    src/apps/quantizer/commands/ppl.cpp
)
phaseshift_set_rocm_rpath(phaseshift-quantizer)
target_compile_features(phaseshift-quantizer PRIVATE cxx_std_20)
target_link_libraries(phaseshift-quantizer PRIVATE phaseshift_quantizer_core)
phaseshift_set_hip_archs(phaseshift-quantizer)
target_compile_options(phaseshift-quantizer PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)

# Chat UI: Python script copied into the build directory.
set(PHASESHIFT_CLI_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/cli/phaseshift_cli.py")
set(PHASESHIFT_CLI_TARGET "${CMAKE_CURRENT_BINARY_DIR}/phaseshift-cli")
add_custom_command(
    OUTPUT "${PHASESHIFT_CLI_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PHASESHIFT_CLI_SCRIPT}" "${PHASESHIFT_CLI_TARGET}"
    COMMAND /bin/sh -c "chmod +x '${PHASESHIFT_CLI_TARGET}'"
    DEPENDS "${PHASESHIFT_CLI_SCRIPT}"
    COMMENT "Staging phaseshift-cli"
)
add_custom_target(phaseshift-cli-stage ALL DEPENDS "${PHASESHIFT_CLI_TARGET}")

# Server product: Python launcher plus the LocalAI backend, the vendored
# LocalAI protobuf stubs, and the shared chat codec. Everything the server
# needs lives under build/phaseshift-server-lib/; nothing is imported from the
# source tree at runtime.
set(PHASESHIFT_SERVER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/phaseshift_server.py")
set(PHASESHIFT_SERVER_TARGET "${CMAKE_CURRENT_BINARY_DIR}/phaseshift-server")
set(PHASESHIFT_SERVER_LIB "${CMAKE_CURRENT_BINARY_DIR}/phaseshift-server-lib")
set(PHASESHIFT_SERVER_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/backend")
set(PHASESHIFT_SERVER_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/server/localai_proto")
set(PHASESHIFT_CHAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/apps/common/phaseshift_chat")
set(PHASESHIFT_CHAT_STAGE_DIR "${PHASESHIFT_SERVER_LIB}/common/phaseshift_chat")
add_custom_command(
    OUTPUT "${PHASESHIFT_SERVER_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PHASESHIFT_SERVER_SCRIPT}" "${PHASESHIFT_SERVER_TARGET}"
    COMMAND /bin/sh -c "chmod +x '${PHASESHIFT_SERVER_TARGET}'"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${PHASESHIFT_SERVER_LIB}/backend"
        "${PHASESHIFT_SERVER_LIB}/localai_proto"
        "${PHASESHIFT_CHAT_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PHASESHIFT_SERVER_BACKEND_DIR}" "${PHASESHIFT_SERVER_LIB}/backend"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PHASESHIFT_SERVER_PROTO_DIR}" "${PHASESHIFT_SERVER_LIB}/localai_proto"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PHASESHIFT_CHAT_DIR}" "${PHASESHIFT_CHAT_STAGE_DIR}"
    COMMAND /bin/sh -c "chmod +x '${PHASESHIFT_SERVER_LIB}/backend/phaseshift_backend.py'"
    DEPENDS
        "${PHASESHIFT_SERVER_SCRIPT}"
        "${PHASESHIFT_SERVER_BACKEND_DIR}/phaseshift_backend.py"
        "${PHASESHIFT_SERVER_BACKEND_DIR}/compute_client.py"
        "${PHASESHIFT_SERVER_BACKEND_DIR}/message_codec.py"
        "${PHASESHIFT_CHAT_DIR}/codec.py"
        "${PHASESHIFT_CHAT_DIR}/tool_constraint.py"
    COMMENT "Staging phaseshift-server"
)
add_custom_target(phaseshift-server-stage ALL DEPENDS "${PHASESHIFT_SERVER_TARGET}")
add_dependencies(phaseshift-cli-stage phaseshift-server-stage)

# E2E benchmark (pp / tg)
# + Qwen35 micro-benchmarks
if(PHASESHIFT_BUILD_BENCHMARKS)
    add_executable(phaseshift-bench
        src/apps/bench/main.cpp
        src/apps/bench/pp.hip
        src/apps/bench/tg.hip
        src/apps/bench/batch.hip
        src/apps/bench/gemm.hip
        src/apps/bench/activation_quantize.hip
        src/apps/bench/rmsnorm.hip
        src/apps/bench/l2_normalize.hip
        src/apps/bench/kv_append.hip
        src/apps/bench/paged_attention.hip
        src/apps/bench/rope.hip
        src/apps/bench/gdn_recurrence.hip
        src/apps/bench/gdn_conv1d.hip
        src/apps/bench/elementwise.hip
        src/apps/bench/embedding.hip
        src/apps/bench/sampling.hip
        src/apps/bench/gpu_memory.hip
        src/apps/bench/tp_reduce.cpp
        src/apps/bench/gpu_dispatch.hip
        src/apps/bench/gpu_sync.hip
        src/apps/bench/gpu_ext_dispatch.hip
    )
    phaseshift_set_rocm_rpath(phaseshift-bench)
    target_compile_features(phaseshift-bench PRIVATE cxx_std_20)
    target_link_libraries(phaseshift-bench PRIVATE phaseshift phaseshift_gpu_mcu)
    phaseshift_set_hip_archs(phaseshift-bench)
    target_include_directories(phaseshift-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    execute_process(
        COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
        OUTPUT_VARIABLE PHASESHIFT_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PHASESHIFT_GIT_SHA)
        target_compile_definitions(
            phaseshift-bench PRIVATE PHASESHIFT_GIT_SHA="${PHASESHIFT_GIT_SHA}")
    endif()
    if(ROCTX_FOUND)
        target_compile_definitions(phaseshift-bench PRIVATE PHASESHIFT_HAS_ROCTX=1)
        target_include_directories(phaseshift-bench PRIVATE "${ROCTX_INCLUDE_DIR}")
        target_link_libraries(phaseshift-bench PRIVATE "${ROCTX_LIBRARY}")
    endif()
endif()
