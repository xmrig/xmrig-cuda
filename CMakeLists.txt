cmake_minimum_required(VERSION 2.8)
project(xmrig-cuda)


option(WITH_DRIVER_API "Enable CUDA Driver API and NVRTC, required for cn/r and kawpow algorithms" ON)


include_directories(src)
add_definitions(/DXMRIG_ALGO_CN_LITE /DXMRIG_ALGO_CN_HEAVY /DXMRIG_ALGO_CN_PICO)


include(cmake/cpu.cmake)
include(cmake/flags.cmake)
include(cmake/CUDA.cmake)


set(SOURCES
    src/crypto/cn/c_blake256.c
    src/crypto/common/Algorithm.cpp
    src/crypto/common/Algorithm.h
    src/version.h
    src/xmrig-cuda.cpp
    src/xmrig-cuda.h
    )


if (WITH_DRIVER_API)
    list(APPEND SOURCES src/CudaCryptonightR_gen.cpp)
endif()


add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} xmrig-cu ${LIBS})

if (WITH_DRIVER_API AND WIN32)
    file(GLOB NVRTCDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc64*.dll")
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVRTCDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)

    file(GLOB NVRTCBUILTINDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc-builtins64*.dll")
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVRTCBUILTINDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()
