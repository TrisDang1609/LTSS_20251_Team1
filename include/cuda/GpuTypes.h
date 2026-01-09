/**
 * GPU-Compatible Data Types for ORB-SLAM3 CUDA Pipeline
 *
 * This header defines GPU-friendly data structures that mirror the CPU types
 * but are optimized for GPU memory coalescing and warp-level operations.
 *
 * Design Principles:
 * 1. Structure of Arrays (SoA) layout for memory coalescing
 * 2. 128-byte alignment for optimal cache line usage
 * 3. POD types for direct GPU memory transfers
 */

#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <opencv2/core/cuda.hpp>

// Undefine macros from sys/sysmacros.h that conflict with CUDA identifiers
// These must be undefined AFTER all system/OpenCV includes
#ifdef major
#undef major
#endif
#ifdef minor
#undef minor
#endif

namespace ORB_SLAM3
{
    namespace cuda
    {

        // Note: GpuArray template is defined in CudaMemoryManager.h

        // ============================================================================
        // GPU Keypoint Structure (Aligned for coalescing)
        // ============================================================================

        // Single keypoint data (32 bytes aligned for coalescing)
        struct alignas(32) GpuKeyPoint
        {
            float x;        // 4 bytes - Keypoint x coordinate
            float y;        // 4 bytes - Keypoint y coordinate
            float size;     // 4 bytes - Keypoint size
            float angle;    // 4 bytes - Orientation angle in degrees
            float response; // 4 bytes - Keypoint response/score
            int octave;     // 4 bytes - Pyramid level
            int classId;    // 4 bytes - Classification ID
            int padding;    // 4 bytes - Padding for alignment
        };

        // Structure of Arrays layout for keypoints (optimal memory coalescing)
        struct GpuKeyPointSoA
        {
            float *x;        // X coordinates
            float *y;        // Y coordinates
            float *size;     // Keypoint sizes
            float *angle;    // Orientation angles
            float *response; // Response/scores
            int *octave;     // Pyramid levels
            int count;       // Number of keypoints
            int capacity;    // Allocated capacity

            __host__ __device__ GpuKeyPoint get(int idx) const
            {
                GpuKeyPoint kp;
                kp.x = x[idx];
                kp.y = y[idx];
                kp.size = size[idx];
                kp.angle = angle[idx];
                kp.response = response[idx];
                kp.octave = octave[idx];
                kp.classId = 0;
                kp.padding = 0;
                return kp;
            }

            __host__ __device__ void set(int idx, const GpuKeyPoint &kp)
            {
                x[idx] = kp.x;
                y[idx] = kp.y;
                size[idx] = kp.size;
                angle[idx] = kp.angle;
                response[idx] = kp.response;
                octave[idx] = kp.octave;
            }
        };

        // ============================================================================
        // GPU Descriptor Structure
        // ============================================================================

        // ORB descriptor is 256 bits = 32 bytes = 8 uint32_t
        struct alignas(32) GpuDescriptor
        {
            uint32_t data[8]; // 256-bit descriptor as 8 x 32-bit words

            __host__ __device__ uint32_t &operator[](int idx) { return data[idx]; }
            __host__ __device__ const uint32_t &operator[](int idx) const { return data[idx]; }
        };

        // Descriptor array with metadata
        struct GpuDescriptorArray
        {
            GpuDescriptor *descriptors; // Array of descriptors
            int count;                  // Number of descriptors
            int capacity;               // Allocated capacity
        };

        // ============================================================================
        // GPU Image Pyramid Level
        // ============================================================================

        struct GpuPyramidLevel
        {
            cv::cuda::GpuMat image;   // Image data at this level
            cv::cuda::GpuMat blurred; // Gaussian-blurred for descriptor computation
            float scale;              // Scale factor relative to base
            float invScale;           // Inverse scale factor
            int width;                // Image width at this level
            int height;               // Image height at this level
        };

        // ============================================================================
        // GPU Image Pyramid
        // ============================================================================

        struct GpuImagePyramid
        {
            static constexpr int MAX_LEVELS = 8;

            GpuPyramidLevel levels[MAX_LEVELS];
            int numLevels;
            float scaleFactor;

            __host__ GpuPyramidLevel &operator[](int level) { return levels[level]; }
            __host__ const GpuPyramidLevel &operator[](int level) const { return levels[level]; }
        };

        // ============================================================================
        // GPU Feature Grid for Spatial Indexing
        // ============================================================================

        constexpr int GRID_ROWS = 48;
        constexpr int GRID_COLS = 64;
        constexpr int GRID_CELLS = GRID_ROWS * GRID_COLS;
        constexpr int MAX_FEATURES_PER_CELL = 64;

        struct GpuFeatureGrid
        {
            int *cellStart;      // Start index for each cell (GRID_CELLS)
            int *cellCount;      // Feature count per cell (GRID_CELLS)
            int *featureIndices; // Sorted feature indices by cell
            float cellWidth;
            float cellHeight;
            float invCellWidth;
            float invCellHeight;
            int imageWidth;
            int imageHeight;
        };

        // ============================================================================
        // GPU Match Result
        // ============================================================================

        struct alignas(8) GpuMatch
        {
            int queryIdx; // Index in query set
            int trainIdx; // Index in train set
            int distance; // Hamming distance
            int padding;  // Alignment padding
        };

        // Match result array
        struct GpuMatchArray
        {
            GpuMatch *matches;
            int *matchCount; // Number of valid matches
            int capacity;
        };

        // ============================================================================
        // GPU FAST Corner Response
        // ============================================================================

        struct GpuFastResponse
        {
            int16_t x;
            int16_t y;
            int16_t response; // Corner response score
            int16_t level;    // Pyramid level
        };

        // ============================================================================
        // GPU Extractor Node for Octree Distribution
        // ============================================================================

        struct GpuExtractorNode
        {
            int ulX, ulY; // Upper-left corner
            int brX, brY; // Bottom-right corner
            int startIdx; // Start index in keypoint array
            int count;    // Number of keypoints
            bool noMore;  // Flag: only one keypoint
            int padding;  // Alignment
        };

        // ============================================================================
        // GPU Frame Data Structure
        // ============================================================================

        struct GpuFrameData
        {
            // Image data
            cv::cuda::GpuMat image;  // Original grayscale image
            GpuImagePyramid pyramid; // Image pyramid

            // Features
            GpuKeyPointSoA keypoints;       // Keypoints in SoA format
            GpuDescriptorArray descriptors; // ORB descriptors
            GpuFeatureGrid grid;            // Spatial indexing grid

            // Metadata
            int frameId;
            int numKeypoints;
            float timestamp;

            // Camera parameters (on device)
            float fx, fy, cx, cy;
            float invfx, invfy;
            float k1, k2, p1, p2, k3; // Distortion coefficients
        };

        // ============================================================================
        // GPU Projection Data for Feature Matching
        // ============================================================================

        struct GpuProjection
        {
            float projX;        // Projected x coordinate
            float projY;        // Projected y coordinate
            float viewCos;      // Viewing angle cosine
            int predictedLevel; // Predicted scale level
            int mapPointIdx;    // Index of source map point
            bool valid;         // Projection validity flag
        };

        struct GpuProjectionArray
        {
            GpuProjection *projections;
            int count;
            int capacity;
        };

        // ============================================================================
        // ORB Pattern for Descriptor Computation (Device Constant Memory)
        // ============================================================================

        // Will be stored in constant memory for fast access
        struct OrbPattern
        {
            int2 points[512]; // 512 comparison point pairs
        };

        // ============================================================================
        // Hamming Distance LUT (Look-Up Table)
        // ============================================================================

        // Popcount lookup table for 8-bit values (stored in constant memory)
        struct PopcountLUT
        {
            uint8_t table[256];
        };

        // ============================================================================
        // Configuration for GPU Pipeline
        // ============================================================================

        struct GpuPipelineConfig
        {
            // ORB extractor parameters
            int nFeatures;
            float scaleFactor;
            int nLevels;
            int iniThFAST;
            int minThFAST;

            // Matching parameters
            float nnRatio;
            int thHigh;
            int thLow;
            bool checkOrientation;

            // Image parameters
            int imageWidth;
            int imageHeight;

            // Performance tuning
            bool useAsyncMemcpy;
            bool useGraphCapture;
            int numStreams;
        };

        // ============================================================================
        // GPU Statistics for Profiling
        // ============================================================================

        struct GpuPipelineStats
        {
            // Pipeline timing
            float preprocessTimeMs = 0.0f;
            float pyramidTimeMs = 0.0f;
            float detectionTimeMs = 0.0f;
            float descriptionTimeMs = 0.0f;
            float gridTimeMs = 0.0f;
            float matchingTimeMs = 0.0f;
            float totalTimeMs = 0.0f;

            // Legacy names (for compatibility)
            float fastTimeMs = 0.0f;
            float orientTimeMs = 0.0f;
            float descriptorTimeMs = 0.0f;

            // Counts
            int numKeypoints = 0;
            int numMatches = 0;
            size_t memoryUsedBytes = 0;
        };

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // GPU_TYPES_H
