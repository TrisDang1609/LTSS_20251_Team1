# =============================================================================
# OpenCL Wrapper Library
# =============================================================================
add_library(libopencl STATIC opencl.cpp opencl_global.cpp)
target_include_directories(libopencl PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(libopencl PUBLIC OpenCL)

# Define OpenCL 3.0 target version
target_compile_definitions(libopencl PRIVATE CL_TARGET_OPENCL_VERSION=300)

# =============================================================================
# Image Processing Library (CPU + GPU operations)
# =============================================================================
add_library(libimage STATIC image.cpp)
target_include_directories(libimage PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(libimage PUBLIC libopencl OpenCL)
target_compile_definitions(libimage PRIVATE CL_TARGET_OPENCL_VERSION=300)

# =============================================================================
# SIFT Library (CPU + GPU implementations)
# =============================================================================
add_library(libsift STATIC sift.cpp)
target_include_directories(libsift PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(libsift PUBLIC 
    libimage 
    libopencl 
    OpenCL
)

# Ensure SIFT has access to all OpenCL kernels at compile time
target_compile_definitions(libsift PRIVATE 
    KERNEL_DIR="${CMAKE_SOURCE_DIR}/src/opencl_kernels"
    CL_TARGET_OPENCL_VERSION=300
)

# =============================================================================
# Library Export Configuration
# =============================================================================
# Make libraries available to parent scope
set_target_properties(libopencl PROPERTIES
    OUTPUT_NAME "opencl_api"
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(libimage PROPERTIES
    OUTPUT_NAME "image_processing"
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(libsift PROPERTIES
    OUTPUT_NAME "sift_gpu"
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Installation Rules (Optional)
# =============================================================================
install(TARGETS libopencl libimage libsift
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/libs/
    DESTINATION include/gpu_sift
    FILES_MATCHING PATTERN "*.hpp"
)
