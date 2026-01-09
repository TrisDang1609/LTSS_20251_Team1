# ============================================================================
# ORB-SLAM3 CUDA Extension CMake Configuration
# 
# This file configures CUDA compilation for the GPU-accelerated modules.
# Include this file from the main CMakeLists.txt to enable CUDA support.
#
# Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
# CUDA Version: 13.0+
# C++ Standard: C++17
# ============================================================================

cmake_minimum_required(VERSION 3.22)

# Enable CUDA language support
enable_language(CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

message(STATUS "=====================================")
message(STATUS "  ORB-SLAM3 CUDA Configuration")
message(STATUS "=====================================")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Library Dir: ${CUDAToolkit_LIBRARY_DIR}")

# ============================================================================
# CUDA Compiler Configuration
# ============================================================================

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures - RTX 4060 is Ada Lovelace (SM 8.9)
# Include 8.6 for RTX 30 series compatibility
set(CMAKE_CUDA_ARCHITECTURES "86;89")

# Separate compilation for relocatable device code
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA compile flags as space-separated string (not list!)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -lineinfo")

# ============================================================================
# Include Directories
# ============================================================================

set(CUDA_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ============================================================================
# Source Files
# ============================================================================

set(CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/CudaMemoryManager.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/ORBExtractorCuda.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/ORBMatcherCuda.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/ImagePreprocessCuda.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/FeatureGridCuda.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/GpuPipeline.cu
)

set(CUDA_HEADERS
    ${CMAKE_SOURCE_DIR}/include/cuda/CudaUtils.h
    ${CMAKE_SOURCE_DIR}/include/cuda/GpuTypes.h
    ${CMAKE_SOURCE_DIR}/include/cuda/CudaMemoryManager.h
    ${CMAKE_SOURCE_DIR}/include/cuda/ORBExtractorCuda.h
    ${CMAKE_SOURCE_DIR}/include/cuda/ORBMatcherCuda.h
    ${CMAKE_SOURCE_DIR}/include/cuda/ImagePreprocessCuda.h
    ${CMAKE_SOURCE_DIR}/include/cuda/FeatureGridCuda.h
    ${CMAKE_SOURCE_DIR}/include/cuda/GpuPipeline.h
)

# ============================================================================
# CUDA Library Target
# ============================================================================

add_library(${PROJECT_NAME}_cuda STATIC
    ${CUDA_SOURCES}
    ${CUDA_HEADERS}
)

# Set CUDA-specific properties
set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
)

# Include directories
target_include_directories(${PROJECT_NAME}_cuda 
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    PRIVATE
        ${OpenCV_LIBS}
)

# ============================================================================
# OpenCV CUDA Module Check
# ============================================================================

# Check if OpenCV was built with CUDA support
if(OpenCV_CUDA_VERSION)
    message(STATUS "OpenCV CUDA Version: ${OpenCV_CUDA_VERSION}")
    target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE OPENCV_CUDA_AVAILABLE)
else()
    message(WARNING "OpenCV was not built with CUDA support. "
                    "Some GPU features will fall back to CPU implementations.")
endif()

# ============================================================================
# Thrust Configuration
# ============================================================================

# Thrust is header-only and included with CUDA Toolkit
target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
)

# ============================================================================
# Performance Tuning Definitions
# ============================================================================

# Note: Most GPU constants are defined as constexpr in CudaUtils.h
# Only define macros needed for preprocessing conditionals here
target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE
    COMPILEDWITHC17
)

# ============================================================================
# Integration with Main Library
# ============================================================================

# Function to link CUDA library to the main ORB-SLAM3 library
# Using plain signature to match existing CMakeLists.txt style
function(link_cuda_to_orbslam TARGET_NAME)
    target_link_libraries(${TARGET_NAME} ${PROJECT_NAME}_cuda)
    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/cuda)
    target_compile_definitions(${TARGET_NAME} PRIVATE CUDA_ENABLED)
endfunction()

# ============================================================================
# Debug Utilities
# ============================================================================

if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    # Add CUDA memory check tools
    find_program(CUDA_MEMCHECK cuda-memcheck)
    if(CUDA_MEMCHECK)
        message(STATUS "Found cuda-memcheck: ${CUDA_MEMCHECK}")
    endif()
    
    # Add CUDA profiler tools
    find_program(NSIGHT_COMPUTE ncu)
    find_program(NSIGHT_SYSTEMS nsys)
    if(NSIGHT_COMPUTE)
        message(STATUS "Found Nsight Compute: ${NSIGHT_COMPUTE}")
    endif()
    if(NSIGHT_SYSTEMS)
        message(STATUS "Found Nsight Systems: ${NSIGHT_SYSTEMS}")
    endif()
endif()

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "CUDA Configuration Summary:")
message(STATUS "  CUDA Standard: C++${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Sources: ${CUDA_SOURCES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=====================================")
