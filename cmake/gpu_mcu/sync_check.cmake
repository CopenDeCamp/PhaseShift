# ---------------------------------------------------------------------------
# MCU sync contract check
#
# The persistent MCU wave keeps its streams busy forever, so a synchronous
# device->host copy or a device-wide sync never returns while it is alive.
# Enforce the contract before compiling the mcu-aware translation units.
# ---------------------------------------------------------------------------

find_program(PHASESHIFT_PYTHON_EXECUTABLE NAMES python3)
if(NOT PHASESHIFT_PYTHON_EXECUTABLE)
    message(FATAL_ERROR "python3 is required for the mcu sync contract check")
endif()

set(
    PHASESHIFT_MCU_SYNC_SCOPE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/runtime/gpu_mcu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/mcu_decode_runtime.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/mcu_decode_state.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/mcu_plan_cache.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/mcu_plan_cache.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/mcu_plan_compiler.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/mcu_kernel_registry.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/phaseshift/models/qwen35/runtime/executor.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/controller/test_gpu_mcu_persistent_ingress.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/substrate/test_gpu_mcu_batch_dispatch.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/controller/test_gpu_mcu_autonomous_loop.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/controller/test_gpu_mcu_continuous_turnover.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/controller/test_gpu_mcu_kv_page_turnover.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/qwen35/test_gpu_mcu_gdn_reset.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/controller/test_gpu_mcu_persistent_mcu.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/gpu_mcu/controller/test_gpu_mcu_runtime_lifecycle.hip"
)

add_custom_target(
    phaseshift_mcu_sync_check
    COMMAND
        "${PHASESHIFT_PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mcu_sync.py"
        ${PHASESHIFT_MCU_SYNC_SCOPE}
    COMMENT "phaseshift: checking the mcu sync contract"
    VERBATIM
)

add_dependencies(phaseshift_gpu_mcu phaseshift_mcu_sync_check)
add_dependencies(phaseshift_qwen35_runtime phaseshift_mcu_sync_check)
