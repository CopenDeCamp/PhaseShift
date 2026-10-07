# GPU-MCU low-level substrate: HSA AQL queue / packet / doorbell, CU partition,
# mapped control ring, persistent MCU, worker image. Model-independent and
# never linked into the phaseshift aggregate; only low-level tests and the
# GPU-MCU RnD bench subcommands link it directly.
find_package(hsa-runtime64 1.0 REQUIRED)

set(_PS_GPU_MCU_PROBE_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/gpu_mcu_probe_worker.hsaco")
set(_PS_GPU_MCU_PROBE_HSACO "${CMAKE_CURRENT_BINARY_DIR}/gpu_mcu_probe_worker.elf.hsaco")
set(_PS_GPU_MCU_PROBE_INC
    "${CMAKE_CURRENT_BINARY_DIR}/gen/phaseshift_gpu_mcu_probe_worker_hsaco.inc")

# hipcc --genco emits a fixed 4096-byte CLANG_OFFLOAD_BUNDLE header followed by
# the bare ELF HSACO; the trailing ELF is what the HSA reader consumes.
# Verified on the pinned ROCm toolchain (see docs/developer/gpu_mcu/low_level.md).
add_custom_command(
    OUTPUT "${_PS_GPU_MCU_PROBE_BUNDLE}"
    COMMAND "${PS_HIPCC_EXECUTABLE}"
        --genco
        -O3
        -DNDEBUG
        -std=c++20
        --offload-arch=${CMAKE_HIP_ARCHITECTURES}
        -I${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/phaseshift/runtime/gpu_mcu/infrastructure/worker_probe.hip
        -o ${_PS_GPU_MCU_PROBE_BUNDLE}
    DEPENDS
        "${CMAKE_SOURCE_DIR}/src/phaseshift/runtime/gpu_mcu/infrastructure/worker_probe.hip"
        "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/gpu_mcu/infrastructure/worker_image.h"
        "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/gpu_mcu/infrastructure/completion.h"
        "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/gpu_mcu/infrastructure/device_completion.h"
        "${CMAKE_SOURCE_DIR}/include/phaseshift/runtime/gpu_mcu/infrastructure/fsm_worker.h"
    VERBATIM)
add_custom_command(
    OUTPUT "${_PS_GPU_MCU_PROBE_HSACO}"
    COMMAND dd "if=${_PS_GPU_MCU_PROBE_BUNDLE}" "of=${_PS_GPU_MCU_PROBE_HSACO}"
        bs=4096 skip=1
    DEPENDS "${_PS_GPU_MCU_PROBE_BUNDLE}"
    VERBATIM)
add_custom_command(
    OUTPUT "${_PS_GPU_MCU_PROBE_INC}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/gen"
    COMMAND sh -c
        "xxd -i -n phaseshift_gpu_mcu_probe_worker_hsaco '${_PS_GPU_MCU_PROBE_HSACO}' > '${_PS_GPU_MCU_PROBE_INC}'"
    DEPENDS "${_PS_GPU_MCU_PROBE_HSACO}"
    VERBATIM)

add_library(phaseshift_gpu_mcu STATIC
    src/phaseshift/runtime/gpu_mcu/infrastructure/aql.hip
    src/phaseshift/runtime/gpu_mcu/infrastructure/cu_partition.hip
    src/phaseshift/runtime/gpu_mcu/io/control_ring.hip
    src/phaseshift/runtime/gpu_mcu/io/output_ring.hip
    src/phaseshift/runtime/gpu_mcu/scheduling/slot_table.hip
    src/phaseshift/runtime/gpu_mcu/io/request_ingress.hip
    src/phaseshift/runtime/gpu_mcu/scheduling/batch_planner.hip
    src/phaseshift/runtime/gpu_mcu/binding/slot_binding.hip
    src/phaseshift/runtime/gpu_mcu/binding/batch_binding.hip
    src/phaseshift/runtime/gpu_mcu/execution/persistent_mcu.hip
    src/phaseshift/runtime/gpu_mcu/execution/micro_fsm.hip
    src/phaseshift/runtime/gpu_mcu/infrastructure/wall_clock.cpp
    src/phaseshift/runtime/gpu_mcu/infrastructure/worker_image.cpp
)
set_source_files_properties("${_PS_GPU_MCU_PROBE_INC}"
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
target_sources(phaseshift_gpu_mcu PRIVATE "${_PS_GPU_MCU_PROBE_INC}")
target_compile_features(phaseshift_gpu_mcu PRIVATE cxx_std_20)
target_include_directories(phaseshift_gpu_mcu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_gpu_mcu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")
target_link_libraries(phaseshift_gpu_mcu PUBLIC
    phaseshift_core
    phaseshift_gpu
    hip::host
    hsa-runtime64::hsa-runtime64
)
target_compile_options(phaseshift_gpu_mcu PRIVATE -Wall -Wextra -Wpedantic -Werror=return-type)
phaseshift_set_hip_archs(phaseshift_gpu_mcu)

phaseshift_set_rocm_rpath(phaseshift_gpu_mcu)
