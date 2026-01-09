/**
 * ORB Matcher CUDA - GPU-Accelerated Feature Matching
 *
 * This is a non-destructive add-on to the original ORBmatcher.cc
 * providing GPU-accelerated feature matching with Hamming distance.
 *
 * Features:
 * 1. Warp-parallel Hamming distance computation
 * 2. Thrust-based sorting for nearest neighbor search
 * 3. GPU-resident matching for zero host-device copies
 * 4. Batch matching for multiple map points
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 */

#ifndef ORB_MATCHER_CUDA_H
#define ORB_MATCHER_CUDA_H

#include "CudaUtils.h"
#include "GpuTypes.h"
#include "CudaMemoryManager.h"

#include <opencv2/core/cuda.hpp>
#include <vector>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Matching Configuration
        // ============================================================================

        struct MatchingConfig
        {
            float nnRatio = 0.6f;         // Nearest neighbor ratio threshold
            int thHigh = 100;             // High distance threshold
            int thLow = 50;               // Low distance threshold
            bool checkOrientation = true; // Check keypoint orientation consistency
            int histoLength = 30;         // Orientation histogram bins
            int maxMatches = 5000;        // Maximum matches per operation
            bool useBruteForce = true;    // Use brute-force vs approximate
        };

        // ============================================================================
        // ORB Matcher CUDA Class
        // ============================================================================

        class ORBMatcherCuda
        {
        public:
            /**
             * Constructor matching original ORBmatcher signature
             */
            ORBMatcherCuda(float nnratio = 0.6f, bool checkOri = true);

            /**
             * Constructor with full configuration
             */
            explicit ORBMatcherCuda(const MatchingConfig &config);

            ~ORBMatcherCuda();

            // ========================================================================
            // GPU-Resident Matching (Preferred Methods)
            // ========================================================================

            /**
             * Match descriptors between two GPU-resident descriptor sets
             * All operations on GPU, results stay on GPU
             *
             * @param queryDescriptors Query descriptors (from current frame)
             * @param trainDescriptors Train descriptors (from keyframe/map)
             * @param matches Output matches (GPU-resident)
             * @return Number of valid matches
             */
            int matchGpu(const GpuDescriptorArray &queryDescriptors,
                         const GpuDescriptorArray &trainDescriptors,
                         GpuMatchArray &matches);

            /**
             * Match with spatial constraints using feature grid
             * More efficient for projection-based matching
             */
            int matchWithGridGpu(const GpuDescriptorArray &queryDescriptors,
                                 const GpuDescriptorArray &trainDescriptors,
                                 const GpuFeatureGrid &trainGrid,
                                 const GpuProjectionArray &projections,
                                 GpuMatchArray &matches);

            /**
             * Search by projection - matches map points to frame features
             * GPU version of SearchByProjection
             */
            int searchByProjectionGpu(const GpuFrameData &frame,
                                      const GpuDescriptorArray &mapPointDescriptors,
                                      const GpuProjectionArray &projections,
                                      float *radii,
                                      GpuMatchArray &matches);

            /**
             * Brute-force KNN matching on GPU
             * Returns k best matches for each query descriptor
             */
            int matchKnnGpu(const GpuDescriptorArray &queryDescriptors,
                            const GpuDescriptorArray &trainDescriptors,
                            int k,
                            GpuMatch *matches,
                            int *matchCounts);

            // ========================================================================
            // CPU-Compatible Interface
            // ========================================================================

            /**
             * Compute Hamming distance on CPU (for compatibility)
             */
            static int descriptorDistance(const cv::Mat &a, const cv::Mat &b);

            /**
             * Match with CPU Mat interface, uses GPU internally
             */
            int match(const cv::Mat &queryDescriptors,
                      const cv::Mat &trainDescriptors,
                      std::vector<cv::DMatch> &matches);

            /**
             * Download matches to CPU
             */
            void downloadMatches(const GpuMatchArray &gpuMatches,
                                 std::vector<cv::DMatch> &cpuMatches);

            // ========================================================================
            // Utility Functions
            // ========================================================================

            /**
             * Apply ratio test on GPU
             */
            void applyRatioTestGpu(GpuMatch *knnMatches,
                                   int numQueries,
                                   int k,
                                   float ratio,
                                   GpuMatch *goodMatches,
                                   int *numGoodMatches);

            /**
             * Check orientation consistency using histogram
             */
            void checkOrientationGpu(const GpuKeyPointSoA &queryKeypoints,
                                     const GpuKeyPointSoA &trainKeypoints,
                                     GpuMatchArray &matches);

            // ========================================================================
            // Configuration
            // ========================================================================

            void setConfig(const MatchingConfig &config) { config_ = config; }
            const MatchingConfig &getConfig() const { return config_; }

            // Statistics
            struct MatchingStats
            {
                float distanceComputeMs;
                float sortingMs;
                float ratioTestMs;
                float totalMs;
                int numMatches;
            };

            const MatchingStats &getStats() const { return stats_; }

        private:
            // Configuration
            MatchingConfig config_;

            // CUDA streams
            cudaStream_t matchStream_;
            cudaStream_t sortStream_;

            // Temporary buffers
            GpuArray<int> distanceBuffer_;
            GpuArray<int> indexBuffer_;
            GpuArray<GpuMatch> tempMatchBuffer_;

            // Orientation histogram (on GPU)
            GpuArray<int> orientationHist_;

            // Events for timing
            cudaEvent_t startEvent_, stopEvent_;

            // Statistics
            MatchingStats stats_;

            // Helper methods
            void initializeBuffers(int maxDescriptors);
            void computeDistanceMatrix(const GpuDescriptorArray &query,
                                       const GpuDescriptorArray &train,
                                       int *distances);
        };

        // ============================================================================
        // CUDA Kernel Declarations
        // ============================================================================

        /**
         * Hamming distance kernel - computes distance between all descriptor pairs
         * Uses popcount intrinsics for bit counting
         */
        void launchHammingDistanceKernel(
            const GpuDescriptor *queryDescriptors,
            const GpuDescriptor *trainDescriptors,
            int numQuery,
            int numTrain,
            int *distances,
            cudaStream_t stream);

        /**
         * Warp-parallel Hamming distance for single pair
         * One warp computes one distance using parallel popcount
         */
        void launchWarpHammingKernel(
            const GpuDescriptor *queryDescriptors,
            const GpuDescriptor *trainDescriptors,
            int numQuery,
            int numTrain,
            int *distances,
            cudaStream_t stream);

        /**
         * Find K-nearest neighbors using distance matrix
         */
        void launchKnnSearchKernel(
            const int *distances,
            int numQuery,
            int numTrain,
            int k,
            GpuMatch *matches,
            cudaStream_t stream);

        /**
         * Apply ratio test to KNN results
         */
        void launchRatioTestKernel(
            const GpuMatch *knnMatches,
            int numQuery,
            int k,
            float ratio,
            GpuMatch *goodMatches,
            int *numGoodMatches,
            cudaStream_t stream);

        /**
         * Spatial matching kernel - match within projection radius
         */
        void launchSpatialMatchKernel(
            const GpuDescriptor *queryDescriptors,
            const GpuDescriptor *trainDescriptors,
            const GpuKeyPointSoA &trainKeypoints,
            const GpuProjection *projections,
            const float *radii,
            int numProjections,
            int numTrainDescriptors,
            GpuMatch *matches,
            int *matchCount,
            int thHigh,
            float nnRatio,
            cudaStream_t stream);

        /**
         * Build orientation histogram
         */
        void launchOrientationHistKernel(
            const float *queryAngles,
            const float *trainAngles,
            const GpuMatch *matches,
            int numMatches,
            int *histogram,
            int histLength,
            cudaStream_t stream);

        /**
         * Filter matches based on orientation histogram
         */
        void launchOrientationFilterKernel(
            const float *queryAngles,
            const float *trainAngles,
            GpuMatch *matches,
            int *numMatches,
            const int *histogram,
            int histLength,
            int topK,
            cudaStream_t stream);

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // ORB_MATCHER_CUDA_H
