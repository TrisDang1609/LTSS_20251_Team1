/**
 * ORB Extractor CUDA - GPU-Accelerated ORB Feature Extraction
 *
 * This is a non-destructive add-on to the original ORBextractor.cc
 * providing a fully GPU-resident ORB feature extraction pipeline.
 *
 * Pipeline Stages (all on GPU):
 * 1. Image pyramid construction (cv::cuda::pyrDown + border handling)
 * 2. FAST corner detection (custom CUDA kernel)
 * 3. Orientation computation (IC_Angle on GPU)
 * 4. Octree-based keypoint distribution (GPU parallel)
 * 5. ORB descriptor computation (warp-parallel)
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 */

#ifndef ORB_EXTRACTOR_CUDA_H
#define ORB_EXTRACTOR_CUDA_H

#include "CudaUtils.h"
#include "GpuTypes.h"
#include "CudaMemoryManager.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>

#include <vector>
#include <memory>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // ORB Extractor CUDA Class
        // ============================================================================

        class ORBExtractorCuda
        {
        public:
            /**
             * Constructor matching original ORBextractor signature
             * @param nfeatures Total number of features to extract
             * @param scaleFactor Scale factor between pyramid levels
             * @param nlevels Number of pyramid levels
             * @param iniThFAST Initial FAST threshold
             * @param minThFAST Minimum FAST threshold (fallback)
             */
            ORBExtractorCuda(int nfeatures, float scaleFactor, int nlevels,
                             int iniThFAST, int minThFAST);

            ~ORBExtractorCuda();

            /**
             * Main extraction operator - GPU version
             * All processing happens on GPU, only final keypoints/descriptors
             * are transferred back to CPU if needed.
             *
             * @param gpuImage Input image already on GPU (GpuMat)
             * @param gpuKeypoints Output keypoints (GPU-resident)
             * @param gpuDescriptors Output descriptors (GPU-resident)
             * @return Number of extracted keypoints
             */
            int operator()(const cv::cuda::GpuMat &gpuImage,
                           GpuKeyPointSoA &gpuKeypoints,
                           GpuDescriptorArray &gpuDescriptors);

            /**
             * CPU-compatible interface (uploads to GPU internally)
             */
            int operator()(cv::InputArray image, cv::InputArray mask,
                           std::vector<cv::KeyPoint> &keypoints,
                           cv::OutputArray descriptors,
                           std::vector<int> &vLappingArea);

            /**
             * Extract features and keep everything on GPU
             * This is the preferred method for zero-copy pipeline
             */
            int extractGpuResident(const cv::cuda::GpuMat &gpuImage,
                                   GpuFrameData &frameData);

            // ========================================================================
            // Accessors (matching original ORBextractor)
            // ========================================================================

            int GetLevels() const { return nlevels_; }
            float GetScaleFactor() const { return scaleFactor_; }
            std::vector<float> GetScaleFactors() const { return mvScaleFactor_; }
            std::vector<float> GetInverseScaleFactors() const { return mvInvScaleFactor_; }
            std::vector<float> GetScaleSigmaSquares() const { return mvLevelSigma2_; }
            std::vector<float> GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2_; }

            // Get GPU image pyramid for external use
            GpuImagePyramid &getGpuPyramid() { return gpuPyramid_; }

            // Performance statistics
            const GpuPipelineStats &getStats() const { return stats_; }

        private:
            // ========================================================================
            // Core GPU Pipeline Methods
            // ========================================================================

            /**
             * Compute image pyramid on GPU
             * Uses cv::cuda::pyrDown and custom border replication
             */
            void computePyramidGpu(const cv::cuda::GpuMat &image);

            /**
             * Detect FAST corners at all pyramid levels
             * Custom CUDA kernel for FAST-9 detection
             */
            void detectFastCornersGpu();

            /**
             * Select top N keypoints per level based on response
             * Simpler alternative to octree distribution
             */
            void selectTopKeypointsGpu();

            /**
             * Distribute keypoints using octree (GPU-parallel)
             * Uses Thrust for sorting and selection
             */
            void distributeKeypointsGpu();

            /**
             * Compute keypoint orientations (IC_Angle on GPU)
             */
            void computeOrientationsGpu();

            /**
             * Compute ORB descriptors (warp-parallel)
             */
            void computeDescriptorsGpu();

            /**
             * Apply Gaussian blur for descriptor computation
             */
            void applyGaussianBlurGpu();

            // ========================================================================
            // Helper Methods
            // ========================================================================

            void initializePatternGpu();
            void initializeUmaxGpu();
            void uploadConstantMemory();

            // Copy final results to CPU (if needed)
            void downloadResults(std::vector<cv::KeyPoint> &keypoints,
                                 cv::Mat &descriptors);

            // ========================================================================
            // Parameters
            // ========================================================================

            int nfeatures_;
            float scaleFactor_;
            int nlevels_;
            int iniThFAST_;
            int minThFAST_;

            // Scale factors
            std::vector<float> mvScaleFactor_;
            std::vector<float> mvInvScaleFactor_;
            std::vector<float> mvLevelSigma2_;
            std::vector<float> mvInvLevelSigma2_;
            std::vector<int> mnFeaturesPerLevel_;

            // ========================================================================
            // GPU Resources
            // ========================================================================

            // Image pyramid (device-resident)
            GpuImagePyramid gpuPyramid_;

            // Keypoints per level (device-resident)
            std::vector<GpuKeyPointSoA> gpuKeypointsPerLevel_;

            // Final merged keypoints
            GpuKeyPointSoA gpuKeypoints_;

            // Descriptors
            GpuDescriptorArray gpuDescriptors_;

            // CUDA filters
            cv::Ptr<cv::cuda::Filter> gaussianFilter_;

            // Device memory for ORB pattern and umax
            int2 *d_pattern_; // ORB sampling pattern (constant memory)
            int *d_umax_;     // Umax array for orientation

            // Temporary buffers
            GpuArray<GpuFastResponse> fastResponseBuffer_;
            GpuArray<int> keypointCountBuffer_;

            // CUDA streams for async operations
            cudaStream_t pyramidStream_;
            cudaStream_t fastStream_;
            cudaStream_t descriptorStream_;

            // Performance stats
            GpuPipelineStats stats_;

            // Events for timing
            cudaEvent_t startEvent_, stopEvent_;
        };

        // ============================================================================
        // CUDA Kernel Declarations (implemented in .cu file)
        // ============================================================================

        /**
         * FAST-9 corner detection kernel
         * Each thread processes one pixel, uses shared memory for neighborhood
         */
        void launchFastDetectionKernel(
            const cv::cuda::GpuMat &image,
            GpuFastResponse *responses,
            int *responseCount,
            int threshold,
            int maxKeypoints,
            cudaStream_t stream);

        /**
         * IC_Angle orientation computation kernel
         * One thread per keypoint, uses circular pattern
         */
        void launchOrientationKernel(
            const cv::cuda::GpuMat &image,
            GpuKeyPointSoA &keypoints,
            const int *umax,
            cudaStream_t stream);

        /**
         * ORB descriptor computation kernel
         * One warp (32 threads) per descriptor for parallel bit operations
         */
        void launchDescriptorKernel(
            const cv::cuda::GpuMat &blurredImage,
            const GpuKeyPointSoA &keypoints,
            GpuDescriptor *descriptors,
            const int2 *pattern,
            cudaStream_t stream);

        /**
         * Keypoint distribution kernel using grid cells
         * Parallel octree-like distribution
         */
        void launchDistributionKernel(
            GpuKeyPointSoA &inputKeypoints,
            GpuKeyPointSoA &outputKeypoints,
            int targetCount,
            int imageWidth,
            int imageHeight,
            cudaStream_t stream);

        /**
         * Scale keypoint coordinates kernel
         * Adjusts coordinates from pyramid level to full resolution
         */
        void launchScaleCoordinatesKernel(
            GpuKeyPointSoA &keypoints,
            float scale,
            cudaStream_t stream);

        /**
         * Merge keypoints from all levels
         */
        void launchMergeKeypointsKernel(
            const std::vector<GpuKeyPointSoA> &perLevelKeypoints,
            GpuKeyPointSoA &mergedKeypoints,
            cudaStream_t stream);

        /**
         * Build feature grid for spatial indexing
         */
        void launchBuildGridKernel(
            const GpuKeyPointSoA &keypoints,
            GpuFeatureGrid &grid,
            cudaStream_t stream);

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // ORB_EXTRACTOR_CUDA_H
