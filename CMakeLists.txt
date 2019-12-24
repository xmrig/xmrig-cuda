cmake_minimum_required(VERSION 2.8)
project(xmrig-cuda)


option(WITH_RANDOMX "Enable RandomX algorithms family" ON)


include_directories(src)
add_definitions(/DXMRIG_ALGO_CN_LITE)
add_definitions(/DXMRIG_ALGO_CN_HEAVY)
add_definitions(/DXMRIG_ALGO_CN_GPU)
add_definitions(/DXMRIG_ALGO_CN_PICO)


include(cmake/cpu.cmake)
include(cmake/flags.cmake)
include(cmake/CUDA.cmake)


set(SOURCES
    src/crypto/cn/c_blake256.c
    src/crypto/common/Algorithm.h
    src/crypto/common/Algorithm.cpp
    src/CudaCryptonightR_gen.cpp
    src/version.h
    src/xmrig-cuda.cpp
    src/xmrig-cuda.h
    )


if (WIN32)
    set(SOURCES_OS
        res/app.rc
        )
else()
    set(SOURCES_OS "")
endif()


add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES} ${SOURCES_OS})
target_link_libraries(${CMAKE_PROJECT_NAME} xmrig-cu ${LIBS})

if (WIN32)
    file(GLOB NVRTCDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc64*.dll")
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVRTCDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)

    file(GLOB NVRTCBUILTINDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc-builtins64*.dll")
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVRTCBUILTINDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()
