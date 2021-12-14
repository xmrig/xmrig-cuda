set(MSG_CUDA_MAP "\n\n"
    "  Valid CUDA Toolkit Map:\n"
    "   8.x for Fermi/Kepler       /Maxwell/Pascal,\n"
    "   9.0 for       Kepler       /Maxwell/Pascal/Volta(70),\n"
    "   9.1 for       Kepler       /Maxwell/Pascal/Volta(72),\n"
    "  10.x for       Kepler       /Maxwell/Pascal/Volta    /Turing,\n"
    "  11.0 for       Kepler(35/37)/Maxwell/Pascal/Volta    /Turing/Ampere(80)\n"
    "  11.1 for       Kepler(35/37)/Maxwell/Pascal/Volta    /Turing/Ampere(86)\n"
    "  11.5 for       Kepler(35/37)/Maxwell/Pascal/Volta    /Turing/Ampere(87)\n\n"
    "  11.8 for       Kepler(35/37)/Maxwell/Pascal/Volta    /Turing/Ampere(87)/Hopper\n\n"
    "Reference https://developer.nvidia.com/cuda-gpus#compute for arch and family name\n\n"
)

add_definitions(-DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)

option(XMRIG_LARGEGRID "Support large CUDA block count > 128" ON)
if (XMRIG_LARGEGRID)
    add_definitions("-DXMRIG_LARGEGRID=${XMRIG_LARGEGRID}")
endif()

# Obtain actual list of supported arch from nvcc
execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} --help OUTPUT_VARIABLE NVCC_HELP OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" NVCC_HELP "${NVCC_HELP}")
foreach(LINE ${NVCC_HELP})
    string(STRIP "${LINE}" LINE)
    if(FOUND_VALUES)
        string(FIND "${LINE}" "--" POS)
        if(POS EQUAL 0)
            break()
        endif()
        string(APPEND SUPPORT_CUDA_ARCH "${LINE}")
        continue()
    endif()
    if(FOUND_OPT)
        string(FIND "${LINE}" "Allowed values for this option:" POS)
        if(POS EQUAL -1)
            continue()
        endif()
        math(EXPR chop "${POS} + 31")
        string(SUBSTRING "${LINE}" ${chop} -1 LINE)
        string(STRIP "${LINE}" SUPPORT_CUDA_ARCH)
        set(FOUND_VALUES TRUE)
        continue()
    endif()
    string(FIND "${LINE}" "--gpu-architecture" POS)
    if(POS EQUAL 0)
        set(FOUND_OPT TRUE)
    endif()
endforeach()
unset(NVCC_HELP)
unset(FOUND_VALUES)
unset(FOUND_OPT)
unset(LINE)
string(REPLACE "compute_" "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "sm_" "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "lto_" "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "all-major" "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "all" "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "." "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "'" "" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
string(REPLACE "," ";" SUPPORT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")
list(SORT SUPPORT_CUDA_ARCH)
list(REMOVE_DUPLICATES SUPPORT_CUDA_ARCH)
list(REMOVE_ITEM SUPPORT_CUDA_ARCH "")
message(STATUS "CUDA Architectures supported by [${CUDA_NVCC_EXECUTABLE}]: ${SUPPORT_CUDA_ARCH}")
set(DEFAULT_CUDA_ARCH "${SUPPORT_CUDA_ARCH}")

set(CUDA_ARCH "${DEFAULT_CUDA_ARCH}" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")

# validate architectures (only numbers are allowed)
foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
    string(REGEX MATCH "^[0-9]+$" IS_NUMBER ${CUDA_ARCH_ELEM})
    if(NOT IS_NUMBER)
        message(FATAL_ERROR "Defined compute architecture '${CUDA_ARCH_ELEM}' in "
                            "'${CUDA_ARCH}' is not an integral number, use e.g. '30' (for compute architecture 3.0).")
    endif()
    unset(IS_NUMBER)

    list(FIND SUPPORT_CUDA_ARCH "${CUDA_ARCH_ELEM}" POS)
    if(POS EQUAL -1)
        message("${MSG_CUDA_MAP}")
        set(CUDA_UNSUP "Currently selected nvcc does not support CUDA architecture '${CUDA_ARCH_ELEM}'. Use CUDA Toolkit")
        if("${CUDA_ARCH_ELEM}" LESS 30)
            message(FATAL_ERROR "${CUDA_UNSUP} v8.x maximum, Fermi (20/21) was dropped at v9.0")
        endif()
        if("${CUDA_ARCH_ELEM}" GREATER_EQUAL 30 AND "${CUDA_ARCH_ELEM}" LESS 40)
            if("${CUDA_ARCH_ELEM}" LESS_EQUAL 32)
                message(FATAL_ERROR "${CUDA_UNSUP} v10.x maximum, Kepler (30/32) was dropped at v11.0")
            else()
                message(FATAL_ERROR "${CUDA_UNSUP} v11.x maximum, Kepler (35/37) was dropped at v12.0")
            endif()
        endif()
        if("${CUDA_ARCH_ELEM}" GREATER_EQUAL 50 AND "${CUDA_ARCH_ELEM}" LESS 60)
            message(FATAL_ERROR "${CUDA_UNSUP} v11.x maximum, Maxwell (50/52/53) was dropped at v12.0")
        endif()
        if("${CUDA_ARCH_ELEM}" GREATER_EQUAL 70 AND "${CUDA_ARCH_ELEM}" LESS 80)
            if("${CUDA_ARCH_ELEM}" LESS 72)
                message(FATAL_ERROR "${CUDA_UNSUP} v9.0 minimum, Volta (70) was added at v9.0")
            elseif("${CUDA_ARCH_ELEM}" LESS 75)
                message(FATAL_ERROR "${CUDA_UNSUP} v9.1 minimum, Volta (72) was added at v9.1")
            else()
                message(FATAL_ERROR "${CUDA_UNSUP} v10.0 minimum, Turing (75) was added at v10.0")
            endif()
        endif()
        if("${CUDA_ARCH_ELEM}" GREATER_EQUAL 80 AND "${CUDA_ARCH_ELEM}" LESS 90)
            if("${CUDA_ARCH_ELEM}" LESS 86)
                message(FATAL_ERROR "${CUDA_UNSUP} v11.0 minimum, Ampere (80) was added at v11.0")
            elseif("${CUDA_ARCH_ELEM}" EQUAL 86)
                message(FATAL_ERROR "${CUDA_UNSUP} v11.1 minimum, Ampere (86) was added at v11.1")
            else()
                message(FATAL_ERROR "${CUDA_UNSUP} v11.4 minimum, Ampere (87) was added at v11.4")
            endif()
        endif()
    endif()
endforeach()

unset(POS)
unset(CUDA_ARCH_ELEM)
unset(SUPPORT_CUDA_ARCH)
unset(MSG_CUDA_MAP)
list(SORT CUDA_ARCH)
list(REMOVE_DUPLICATES CUDA_ARCH)
list(REMOVE_ITEM CUDA_ARCH "")
message(STATUS "CUDA Architectures being built: ${CUDA_ARCH}")

add_definitions(-DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT)

option(XMRIG_LARGEGRID "Support large CUDA block count > 128" ON)
if (XMRIG_LARGEGRID)
    add_definitions("-DXMRIG_LARGEGRID=${XMRIG_LARGEGRID}")
endif()
option(CUDA_SHOW_REGISTER "Show registers used for each kernel and compute architecture" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps" OFF)

if (WITH_DRIVER_API)
    set(CUDA_LIB_HINTS "${LIBCUDA_LIBRARY_DIR}")
    set(CUDA_NVRTC_LIB_HINTS "${LIBNVRTC_LIBRARY_DIR}")
    if (XMRIG_OS_APPLE)
        list(APPEND CUDA_LIB_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib")
        list(APPEND CUDA_NVRTC_LIB_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib")
    else()
        set(LIB_HINTS
            "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
            "/usr/lib64"
            "/usr/local/cuda/lib64"
            )
        list(APPEND CUDA_LIB_HINTS ${LIB_HINTS})
        list(APPEND CUDA_NVRTC_LIB_HINTS ${LIB_HINTS})
        unset(LIB_HINTS)
    endif()
    find_library(CUDA_LIB libcuda cuda HINTS ${CUDA_LIB_HINTS})
    find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS ${CUDA_NVRTC_LIB_HINTS})
    unset(CUDA_LIB_HINTS)
    unset(CUDA_NVRTC_LIB_HINTS)

    list(APPEND LIBS ${CUDA_LIB} ${CUDA_NVRTC_LIB})
endif()

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
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Wno-deprecated-gpu-targets")

    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architecture
        if("${CUDA_ARCH_ELEM}" STREQUAL "21")
            # "2.1" actually does run faster when compiled as itself, versus in "2.0" compatible mode
            # strange virtual code type on top of compute_20, with no compute_21 (so the normal rule fails)
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--generate-code arch=compute_20,code=sm_21")
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
    endif()

    if (CUDA_SHOW_REGISTER)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
    endif()

    if (CUDA_KEEP_FILES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}")
    endif()

else()
    message(FATAL_ERROR "selected CUDA compiler '${CUDA_COMPILER}' is not supported")
endif()

if (WITH_RANDOMX)
    set(CUDA_RANDOMX_SOURCES
        src/RandomX/aes_cuda.hpp
        src/RandomX/arqma/configuration.h
        src/RandomX/arqma/randomx_arqma.cu
        src/RandomX/blake2b_cuda.hpp
        src/RandomX/common.hpp
        src/RandomX/graft/configuration.h
        src/RandomX/graft/randomx_graft.cu
        src/RandomX/hash.hpp
        src/RandomX/keva/configuration.h
        src/RandomX/keva/randomx_keva.cu
        src/RandomX/monero/configuration.h
        src/RandomX/monero/randomx_monero.cu
        src/RandomX/randomx_cuda.hpp
        src/RandomX/randomx.cu
        src/RandomX/wownero/configuration.h
        src/RandomX/wownero/randomx_wownero.cu
    )
else()
    set(CUDA_RANDOMX_SOURCES "")
endif()

if (WITH_KAWPOW AND WITH_DRIVER_API)
    set(CUDA_KAWPOW_SOURCES
        src/KawPow/raven/CudaKawPow_gen.cpp
        src/KawPow/raven/CudaKawPow_gen.h
        src/KawPow/raven/KawPow.cu
    )
else()
    set(CUDA_KAWPOW_SOURCES "")
endif()

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
    ${CUDA_RANDOMX_SOURCES}
    ${CUDA_KAWPOW_SOURCES}
)

if("${CUDA_COMPILER}" STREQUAL "clang")
    add_library(xmrig-cu STATIC ${CUDA_SOURCES})

    set_target_properties(xmrig-cu PROPERTIES COMPILE_FLAGS ${CLANG_BUILD_FLAGS})
    set_target_properties(xmrig-cu PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
else()
    cuda_add_library(xmrig-cu STATIC ${CUDA_SOURCES})
endif()
