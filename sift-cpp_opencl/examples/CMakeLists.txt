# =============================================================================
# GPU SIFT Examples
# =============================================================================

# Common include directories for all examples
set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/libs
    ${CMAKE_SOURCE_DIR}/src
)

# Common libraries for all examples
set(COMMON_LIBRARIES
    libsift
    libimage
    libopencl
    OpenCL
)

# =============================================================================
# Example 1: Find Keypoints (Single Image SIFT)
# =============================================================================
add_executable(find_keypoints find_keypoints.cpp)
target_include_directories(find_keypoints PRIVATE ${COMMON_INCLUDES})
target_link_libraries(find_keypoints PRIVATE ${COMMON_LIBRARIES})
target_compile_definitions(find_keypoints PRIVATE CL_TARGET_OPENCL_VERSION=300)
set_target_properties(find_keypoints PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# =============================================================================
# Example 2: Match Features (Two Image SIFT + Matching)
# =============================================================================
add_executable(match_features match_features.cpp)
target_include_directories(match_features PRIVATE ${COMMON_INCLUDES})
target_link_libraries(match_features PRIVATE ${COMMON_LIBRARIES})
target_compile_definitions(match_features PRIVATE CL_TARGET_OPENCL_VERSION=300)
set_target_properties(match_features PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# =============================================================================
# Custom Targets for Convenience
# =============================================================================
add_custom_target(examples_all
    DEPENDS find_keypoints match_features
    COMMENT "Building all GPU SIFT examples"
)

# Clean output images
add_custom_target(clean_outputs
    COMMAND ${CMAKE_COMMAND} -E remove -f 
        ${CMAKE_SOURCE_DIR}/gpu_keypoints.jpg
        ${CMAKE_SOURCE_DIR}/gpu_matches.jpg
    COMMENT "Cleaning output images"
)