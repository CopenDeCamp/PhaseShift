# RCCL discovery.
#
# No RCCL CMake config package is assumed; nccl.h and librccl.so are located
# by explicit header/library search rooted at the resolved ROCm installation.

option(PHASESHIFT_ENABLE_RCCL "Build the RCCL multi-GPU parallel runtime" ON)

set(PHASESHIFT_HAVE_RCCL OFF)

if(PHASESHIFT_ENABLE_RCCL)
    find_path(
        PHASESHIFT_RCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS "${PHASESHIFT_ROCM_ROOT}/include" "$ENV{ROCM_PATH}/include"
        PATH_SUFFIXES rocm rocm/include
    )
    find_library(
        PHASESHIFT_RCCL_LIBRARY
        NAMES rccl
        HINTS "${PHASESHIFT_ROCM_ROOT}/lib" "$ENV{ROCM_PATH}/lib"
        PATH_SUFFIXES rocm rocm/lib
    )
    if(NOT PHASESHIFT_RCCL_INCLUDE_DIR OR NOT PHASESHIFT_RCCL_LIBRARY)
        message(
            FATAL_ERROR
            "PHASESHIFT_ENABLE_RCCL is ON but RCCL was not found "
            "(nccl.h: '${PHASESHIFT_RCCL_INCLUDE_DIR}', "
            "librccl: '${PHASESHIFT_RCCL_LIBRARY}'). "
            "Install RCCL, or configure with -DPHASESHIFT_ENABLE_RCCL=OFF."
        )
    endif()
    set(PHASESHIFT_HAVE_RCCL ON)
endif()

if(PHASESHIFT_HAVE_RCCL)
    message(STATUS "RCCL: ${PHASESHIFT_RCCL_LIBRARY}")
    add_library(phaseshift_rccl SHARED IMPORTED)
    set_target_properties(
        phaseshift_rccl
        PROPERTIES
            IMPORTED_LOCATION "${PHASESHIFT_RCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PHASESHIFT_RCCL_INCLUDE_DIR}"
    )
else()
    message(STATUS "RCCL: disabled")
endif()
