#include <iostream>
#include <string>
#include <chrono>

#include "image.hpp"
#include "sift.hpp"
#include "opencl.hpp"

// CRITICAL: Use global OpenCL API instance (unified lifecycle management)
opencl::OPENCL_API opencl_api;

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3)
    {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png)\n";
        return 1;
    }

    // ========================================================================
    // STAGE 1: Initialize OpenCL (ONE-TIME SETUP)
    // ========================================================================
    std::cout << "[GPU SIFT] Initializing OpenCL for feature matching...\n";

    cl_int err = opencl_api.init("NVIDIA", CL_DEVICE_TYPE_GPU, "-cl-fast-relaxed-math");
    if (err != CL_SUCCESS)
    {
        std::cerr << "[ERROR] Failed to initialize OpenCL (err=" << err << ")\n";
        return 1;
    }

    // Load all required kernels (SIFT + matching)
    std::vector<std::string> kernel_files = {
        "src/opencl_kernels/common_kernel.cl",
        "src/opencl_kernels/image_gaussian_blur.cl",
        "src/opencl_kernels/image_resize.cl",
        "src/opencl_kernels/image_tograyscale.cl",
        "src/opencl_kernels/sift_dog_pyramid.cl",
        "src/opencl_kernels/sift_find_keypoints.cl",
        "src/opencl_kernels/sift_refine_keypoints.cl",
        "src/opencl_kernels/sift_gradient_pyramid.cl",
        "src/opencl_kernels/sift_orientation.cl",
        "src/opencl_kernels/sift_descriptor.cl"};

    err = opencl_api.load_kernel_source(kernel_files, "");
    if (err != CL_SUCCESS)
    {
        std::cerr << "[ERROR] Failed to load kernels (err=" << err << ")\n";
        return 1;
    }

    std::cout << "[GPU SIFT] OpenCL initialized successfully\n";

    // ========================================================================
    // STAGE 2: Load Input Images
    // ========================================================================
    std::cout << "\n[GPU SIFT] Loading images...\n";
    std::cout << "  Image A: " << argv[1] << "\n";
    std::cout << "  Image B: " << argv[2] << "\n";

    Image a(argv[1]), b(argv[2]);

    // Convert to grayscale if needed
    if (a.channels == 3)
    {
        std::cout << "[GPU SIFT] Converting image A to grayscale...\n";
        a = rgb_to_grayscale(a);
    }
    if (b.channels == 3)
    {
        std::cout << "[GPU SIFT] Converting image B to grayscale...\n";
        b = rgb_to_grayscale(b);
    }

    std::cout << "[GPU SIFT] Image A: " << a.width << "×" << a.height << "\n";
    std::cout << "[GPU SIFT] Image B: " << b.width << "×" << b.height << "\n";

    // ========================================================================
    // STAGE 3: GPU SIFT Pipeline for Image A
    // ========================================================================
    std::cout << "\n[GPU SIFT] Processing Image A...\n";
    std::cout << "========================================\n";

    auto start_a = std::chrono::high_resolution_clock::now();

    std::vector<sift::Keypoint> kps_a = sift::gpu_find_keypoints_and_descriptors(
        a,
        sift::SIGMA_MIN,
        sift::N_OCT,
        sift::N_SPO,
        sift::C_DOG,
        sift::C_EDGE,
        sift::LAMBDA_ORI,
        sift::LAMBDA_DESC);

    auto end_a = std::chrono::high_resolution_clock::now();
    auto duration_a = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_a - start_a)
                          .count();

    std::cout << "========================================\n";
    std::cout << "[GPU SIFT] Image A: " << kps_a.size()
              << " keypoints in " << duration_a << "ms\n";

    // ========================================================================
    // STAGE 4: GPU SIFT Pipeline for Image B
    // ========================================================================
    std::cout << "\n[GPU SIFT] Processing Image B...\n";
    std::cout << "========================================\n";

    auto start_b = std::chrono::high_resolution_clock::now();

    std::vector<sift::Keypoint> kps_b = sift::gpu_find_keypoints_and_descriptors(
        b,
        sift::SIGMA_MIN,
        sift::N_OCT,
        sift::N_SPO,
        sift::C_DOG,
        sift::C_EDGE,
        sift::LAMBDA_ORI,
        sift::LAMBDA_DESC);

    auto end_b = std::chrono::high_resolution_clock::now();
    auto duration_b = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_b - start_b)
                          .count();

    std::cout << "========================================\n";
    std::cout << "[GPU SIFT] Image B: " << kps_b.size()
              << " keypoints in " << duration_b << "ms\n";

    // ========================================================================
    // STAGE 5: Feature Matching (CPU Implementation)
    // ========================================================================
    // NOTE: Matching uses CPU implementation from sift::find_keypoint_matches()
    // For GPU matching, implement sift::gpu_find_keypoint_matches() using
    // the specification in GPU_SIFT_DESIGN.md (Section: Feature Matching)

    std::cout << "\n[GPU SIFT] Matching features (CPU)...\n";

    auto start_match = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(
        kps_a,
        kps_b,
        sift::THRESH_RELATIVE, // Lowe's ratio test = 0.7
        sift::THRESH_ABSOLUTE  // Absolute distance = 350
    );

    auto end_match = std::chrono::high_resolution_clock::now();
    auto duration_match = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_match - start_match)
                              .count();

    std::cout << "[GPU SIFT] Found " << matches.size()
              << " matches in " << duration_match << "ms\n";

    // ========================================================================
    // STAGE 6: Visualize Matches
    // ========================================================================
    std::cout << "\n[GPU SIFT] Visualizing matches...\n";

    // Convert to RGB for colored visualization
    Image rgb_a = grayscale_to_rgb(a);
    Image rgb_b = grayscale_to_rgb(b);

    Image result = sift::draw_matches(rgb_a, rgb_b, kps_a, kps_b, matches);

    std::string output_path = "gpu_matches.jpg";
    result.save(output_path);

    std::cout << "[GPU SIFT] Output saved to: " << output_path << "\n";

    // ========================================================================
    // STAGE 7: Performance Summary
    // ========================================================================
    int total_time = duration_a + duration_b + duration_match;

    std::cout << "\n========================================\n";
    std::cout << "PERFORMANCE SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Image A processing:  " << duration_a << "ms\n";
    std::cout << "Image B processing:  " << duration_b << "ms\n";
    std::cout << "Feature matching:    " << duration_match << "ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Total time:          " << total_time << "ms\n";
    std::cout << "Keypoints (A):       " << kps_a.size() << "\n";
    std::cout << "Keypoints (B):       " << kps_b.size() << "\n";
    std::cout << "Valid matches:       " << matches.size() << "\n";
    std::cout << "Match ratio:         "
              << (100.0 * matches.size() / std::min(kps_a.size(), kps_b.size()))
              << "%\n";
    std::cout << "========================================\n";

    // ========================================================================
    // STAGE 8: Cleanup
    // ========================================================================
    std::cout << "\n[GPU SIFT] Cleaning up OpenCL resources...\n";
    opencl_api.release();

    std::cout << "[GPU SIFT] Done!\n";

    return 0;
}