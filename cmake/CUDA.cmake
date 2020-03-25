option(XMRIG_LARGEGRID "Support large CUDA block count > 128" ON)
if (XMRIG_LARGEGRID)
    add_definitions("-DXMRIG_LARGEGRID=${XMRIG_LARGEGRID}")
endif()

set(DEVICE_COMPILER "nvcc")
set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND DEVICE_COMPILER "clang")
endif()

set_property(CACHE CUDA_COMPILER PROPERTY STRINGS "${DEVICE_COMPILER}")

list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

set(CUDA_STATIC ON)
find_package(CUDA 8.0 REQUIRED)

find_library(CUDA_LIB libcuda cuda HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBCUDA_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_LIB} ${CUDA_NVRTC_LIB})

set(DEFAULT_CUDA_ARCH "30;50")

# Fermi GPUs are only supported with CUDA < 9.0
if (CUDA_VERSION VERSION_LESS 9.0)
    list(APPEND DEFAULT_CUDA_ARCH "20;21")
endif()

# add Pascal support for CUDA >= 8.0
if (NOT CUDA_VERSION VERSION_LESS 8.0)
    list(APPEND DEFAULT_CUDA_ARCH "60")
endif()

# add Volta support for CUDA >= 9.0
if (NOT CUDA_VERSION VERSION_LESS 9.0)
    list(APPEND DEFAULT_CUDA_ARCH "70")
endif()

set(CUDA_ARCH "${DEFAULT_CUDA_ARCH}" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")

# validate architectures (only numbers are allowed)
foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
    string(REGEX MATCH "^[0-9]+$" IS_NUMBER ${CUDA_ARCH})
    if(NOT IS_NUMBER)
        message(FATAL_ERROR "Defined compute architecture '${CUDA_ARCH_ELEM}' in "
                            "'${CUDA_ARCH}' is not an integral number, use e.g. '30' (for compute architecture 3.0).")
    endif()
    unset(IS_NUMBER)

    if(${CUDA_ARCH_ELEM} LESS 20)
        message(FATAL_ERROR "Unsupported CUDA architecture '${CUDA_ARCH_ELEM}' specified. "
                            "Use '20' (for compute architecture 2.0) or higher.")
    endif()
endforeach()
list(SORT CUDA_ARCH)

option(CUDA_SHOW_REGISTER "Show registers used for each kernel and compute architecture" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps" OFF)

if("${CUDA_COMPILER}" STREQUAL "clang")
    set(LIBS ${LIBS} cudart_static)
    set(CLANG_BUILD_FLAGS "-O3 -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    # activation usage of FMA
    set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -ffp-contract=fast")

    if (CUDA_SHOW_REGISTER)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -Xcuda-ptxas -v")
    endif(CUDA_SHOW_REGISTER)

    if (CUDA_KEEP_FILES)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -save-temps=${PROJECT_BINARY_DIR}")
    endif(CUDA_KEEP_FILES)

    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architectures
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} --cuda-gpu-arch=sm_${CUDA_ARCH_ELEM}")
    endforeach()

elseif("${CUDA_COMPILER}" STREQUAL "nvcc")
    # avoid that nvcc in CUDA < 8 tries to use libc `memcpy` within the kernel
    if (CUDA_VERSION VERSION_LESS 8.0)
        add_definitions(-D_FORCE_INLINES)
        add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
    elseif(CUDA_VERSION VERSION_LESS 9.0)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Wno-deprecated-gpu-targets")
    endif()
    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architecture
        if("${CUDA_ARCH_ELEM}" STREQUAL "21")
            # "2.1" actually does run faster when compiled as itself, versus in "2.0" compatible mode
            # strange virtual code type on top of compute_20, with no compute_21 (so the normal rule fails)
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
                    "--generate-code arch=compute_20,code=sm_21")
        else()
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
                    "--generate-code arch=compute_${CUDA_ARCH_ELEM},code=sm_${CUDA_ARCH_ELEM} --generate-code arch=compute_${CUDA_ARCH_ELEM},code=compute_${CUDA_ARCH_ELEM}")
        endif()
    endforeach()

    # give each thread an independent default stream
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} static")

    option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

    if (CUDA_SHOW_CODELINES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -lineinfo)
        set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif(CUDA_SHOW_CODELINES)

    if (CUDA_SHOW_REGISTER)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
    endif(CUDA_SHOW_REGISTER)

    if (CUDA_KEEP_FILES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}")
    endif(CUDA_KEEP_FILES)

else()
    message(FATAL_ERROR "selected CUDA compiler '${CUDA_COMPILER}' is not supported")
endif()

set(CUDA_RANDOMX_SOURCES
    src/RandomX/aes_cuda.hpp
    src/RandomX/arqma/configuration.h
    src/RandomX/arqma/randomx_arqma.cu
    src/RandomX/blake2b_cuda.hpp
    src/RandomX/common.hpp
    src/RandomX/hash.hpp
    src/RandomX/loki/configuration.h
    src/RandomX/loki/randomx_loki.cu
    src/RandomX/monero/configuration.h
    src/RandomX/monero/randomx_monero.cu
    src/RandomX/randomx_cuda.hpp
    src/RandomX/randomx.cu
    src/RandomX/wownero/configuration.h
    src/RandomX/wownero/randomx_wownero.cu
)

set(CUDA_SOURCES
    src/cryptonight.h
    src/cuda_aes.hpp
    src/cuda_blake.hpp
    src/cuda_core.cu
    src/cuda_device.hpp
    src/cuda_extra.cu
    src/cuda_extra.h
    src/cuda_fast_int_math_v2.hpp
    src/cuda_groestl.hpp
    src/cuda_jh.hpp
    src/cuda_keccak.hpp
    src/cuda_skein.hpp
)

if("${CUDA_COMPILER}" STREQUAL "clang")
    add_library(xmrig-cu STATIC ${CUDA_SOURCES} ${CUDA_RANDOMX_SOURCES})

    set_target_properties(xmrig-cu PROPERTIES COMPILE_FLAGS ${CLANG_BUILD_FLAGS})
    set_target_properties(xmrig-cu PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${CUDA_SOURCES} ${CUDA_RANDOMX_SOURCES} PROPERTIES LANGUAGE CXX)
else()
    cuda_add_library(xmrig-cu STATIC ${CUDA_SOURCES} ${CUDA_RANDOMX_SOURCES})
endif()
