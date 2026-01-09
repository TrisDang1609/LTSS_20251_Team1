/**
 * GPU Pipeline Header
 *
 * Unified GPU pipeline for ORB-SLAM3 that achieves zero host-device round-trips.
 * This is the main integration point for all CUDA modules.
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 */

#ifndef ORB_SLAM3_GPU_PIPELINE_H
#define ORB_SLAM3_GPU_PIPELINE_H

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cuda/CudaUtils.h"
#include "cuda/GpuTypes.h"
#include "cuda/CudaMemoryManager.h"
#include "cuda/ORBExtractorCuda.h"
#include "cuda/ORBMatcherCuda.h"
#include "cuda/ImagePreprocessCuda.h"
#include "cuda/FeatureGridCuda.h"

namespace ORB_SLAM3
{
    namespace cuda
    {

        /**
         * Pipeline Mode
         */
        enum class PipelineMode
        {
            MONOCULAR,
            STEREO,
            RGBD,
            MONOCULAR_INERTIAL,
            STEREO_INERTIAL,
            RGBD_INERTIAL
        };

        /**
         * Frame Processing Stage
         */
        enum class ProcessingStage
        {
            IDLE,
            PREPROCESS,
            PYRAMID,
            DETECTION,
            DESCRIPTION,
            GRID_BUILD,
            MATCHING,
            COMPLETE,
            ERROR
        };

        /**
         * Stereo Matching Configuration
         */
        struct StereoConfig
        {
            float baseline = 0.0f; // Stereo baseline in meters
            float fx = 0.0f;       // Focal length x
            float bf = 0.0f;       // Baseline * focal length
            float minDisparity = 0.0f;
            float maxDisparity = 96.0f;
            int blockSize = 21;
            int numDisparities = 96;
            bool useSGBM = true;
        };

        /**
         * RGB-D Configuration
         */
        struct RGBDConfig
        {
            float depthFactor = 5000.0f; // Depth scale factor
            float minDepth = 0.1f;       // Minimum valid depth
            float maxDepth = 10.0f;      // Maximum valid depth
            bool filterDepth = true;
        };

        /**
         * GPU Frame Result
         * Contains processed frame data on GPU
         */
        struct GpuFrameResult
        {
            // Keypoints (SoA format) - shallow copy, pointers to GPU memory
            GpuKeyPointSoA keypoints;

            // Descriptors - stored as GpuDescriptorArray for compatibility
            GpuDescriptorArray descriptorArray;
            int numDescriptors = 0;

            // Feature grid
            GpuFeatureGrid grid;

            // For stereo: right image keypoints
            GpuKeyPointSoA keypointsRight;
            GpuDescriptorArray descriptorArrayRight;
            int numDescriptorsRight = 0;

            // Stereo matches (left to right)
            GpuArray<int> stereoMatches;
            GpuArray<float> stereoDepths;
            int numStereoMatches = 0;

            // Depth (for RGBD)
            cv::cuda::GpuMat depthMap;

            // Image pyramid (for visualization/debug)
            GpuImagePyramid pyramid;

            // Processing stats
            GpuPipelineStats stats;

            // Processing stage
            ProcessingStage stage = ProcessingStage::IDLE;
            bool success = false;
        };

        /**
         * GPU Pipeline
         *
         * Main class for end-to-end GPU processing of frames.
         * Designed to keep all data GPU-resident throughout processing.
         */
        class GpuPipeline
        {
        public:
            /**
             * Constructor
             */
            GpuPipeline(PipelineMode mode,
                        int imageWidth, int imageHeight,
                        int numFeatures = 1250,
                        float scaleFactor = 1.2f,
                        int numLevels = 8);

            /**
             * Destructor
             */
            ~GpuPipeline();

            /**
             * Initialize the pipeline
             */
            bool initialize();

            /**
             * Shutdown the pipeline
             */
            void shutdown();

            /**
             * Configure camera intrinsics
             */
            void setCameraIntrinsics(float fx, float fy, float cx, float cy);

            /**
             * Configure distortion parameters (for undistortion)
             */
            void setDistortionParams(const std::vector<float> &distCoeffs, bool isFisheye = false);

            /**
             * Configure stereo parameters
             */
            void setStereoConfig(const StereoConfig &config);

            /**
             * Configure RGBD parameters
             */
            void setRGBDConfig(const RGBDConfig &config);

            // ========================================================================
            // Frame Processing Methods
            // ========================================================================

            /**
             * Process a monocular frame (full pipeline)
             * @param image Input image (CPU cv::Mat or GPU cv::cuda::GpuMat)
             * @param timestamp Frame timestamp
             * @return Processing result with GPU-resident data
             */
            GpuFrameResult processMonocular(const cv::Mat &image, double timestamp);
            GpuFrameResult processMonocular(const cv::cuda::GpuMat &image, double timestamp);

            /**
             * Process a stereo frame pair
             */
            GpuFrameResult processStereo(const cv::Mat &imageLeft,
                                         const cv::Mat &imageRight,
                                         double timestamp);
            GpuFrameResult processStereo(const cv::cuda::GpuMat &imageLeft,
                                         const cv::cuda::GpuMat &imageRight,
                                         double timestamp);

            /**
             * Process an RGB-D frame
             */
            GpuFrameResult processRGBD(const cv::Mat &imageRGB,
                                       const cv::Mat &depth,
                                       double timestamp);
            GpuFrameResult processRGBD(const cv::cuda::GpuMat &imageRGB,
                                       const cv::cuda::GpuMat &depth,
                                       double timestamp);

            // ========================================================================
            // Map Point Processing
            // ========================================================================

            /**
             * Project map points to image plane (batch operation on GPU)
             */
            void projectMapPoints(const float *mapPointsWorld, // Nx3 world coordinates
                                  const float *Rcw,            // 3x3 rotation matrix
                                  const float *tcw,            // 3x1 translation
                                  int numPoints,
                                  GpuProjection *projections); // Output projections

            /**
             * Search for matches between projected map points and frame features
             */
            int searchByProjection(const GpuFrameResult &frame,
                                   const GpuProjection *projections,
                                   const uint32_t *mapPointDescriptors, // 8 x numPoints
                                   int numProjections,
                                   int *matchedMapPointIdx,
                                   int *matchedKeypointIdx,
                                   float radiusMultiplier = 1.0f);

            // ========================================================================
            // Utility Methods
            // ========================================================================

            /**
             * Download keypoints to CPU
             */
            void downloadKeypoints(const GpuFrameResult &result,
                                   std::vector<cv::KeyPoint> &keypoints);

            /**
             * Download descriptors to CPU
             */
            void downloadDescriptors(const GpuFrameResult &result,
                                     cv::Mat &descriptors);

            /**
             * Get processing statistics
             */
            GpuPipelineStats getStats() const;

            /**
             * Reset statistics
             */
            void resetStats();

            /**
             * Synchronize all streams
             */
            void synchronize();

            /**
             * Check if pipeline is ready
             */
            bool isReady() const { return initialized_; }

            /**
             * Get pipeline mode
             */
            PipelineMode getMode() const { return mode_; }

        private:
            // Mode and configuration
            PipelineMode mode_;
            int imageWidth_;
            int imageHeight_;
            int numFeatures_;
            float scaleFactor_;
            int numLevels_;
            bool initialized_ = false;

            // Camera parameters
            CameraIntrinsics intrinsics_;
            DistortionParams distortion_;
            bool hasDistortion_ = false;

            // Stereo/RGBD config
            StereoConfig stereoConfig_;
            RGBDConfig rgbdConfig_;

            // CUDA modules
            std::unique_ptr<ORBExtractorCuda> extractor_;
            std::unique_ptr<ORBExtractorCuda> extractorRight_; // For stereo
            std::unique_ptr<ORBMatcherCuda> matcher_;
            std::unique_ptr<ImagePreprocessorCuda> preprocessor_;
            std::unique_ptr<FeatureGridCuda> featureGrid_;

            // Memory manager
            GpuMemoryPool *memPool_;
            GpuFrameManager *frameManager_;

            // CUDA streams
            cudaStream_t streamMain_;
            cudaStream_t streamRight_; // For stereo right image
            cudaStream_t streamAsync_;

            // Timing events
            cudaEvent_t eventStart_;
            cudaEvent_t eventPreprocess_;
            cudaEvent_t eventExtract_;
            cudaEvent_t eventGrid_;
            cudaEvent_t eventEnd_;

            // Statistics
            GpuPipelineStats stats_;

            // Pre-allocated buffers
            cv::cuda::GpuMat gpuImageGray_;
            cv::cuda::GpuMat gpuImageUndist_;
            cv::cuda::GpuMat gpuImageRight_;
            cv::cuda::GpuMat gpuDepth_;

            // Internal methods
            void allocateBuffers();
            void createStreams();
            void destroyStreams();

            GpuFrameResult processFrameInternal(const cv::cuda::GpuMat &image,
                                                double timestamp,
                                                cudaStream_t stream);

            void computeStereoMatches(GpuFrameResult &result);
            void computeDepthFromRGBD(const cv::cuda::GpuMat &depth,
                                      GpuFrameResult &result);
        };

        /**
         * Create GPU pipeline for specified mode
         */
        std::unique_ptr<GpuPipeline> createGpuPipeline(
            PipelineMode mode,
            int imageWidth,
            int imageHeight,
            int numFeatures = 1250,
            float scaleFactor = 1.2f,
            int numLevels = 8);

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_GPU_PIPELINE_H
