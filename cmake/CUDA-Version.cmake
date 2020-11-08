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

if (WITH_DRIVER_API)
    find_library(CUDA_LIB libcuda cuda HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBCUDA_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
    find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

    set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_LIB} ${CUDA_NVRTC_LIB})
else()
    set(LIBS ${LIBS} ${CUDA_LIBRARIES})
endif()
