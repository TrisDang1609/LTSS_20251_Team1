/**
 * GPU Pipeline Implementation
 *
 * End-to-end GPU processing pipeline for ORB-SLAM3.
 */

#include "cuda/GpuPipeline.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Constructor / Destructor
        // ============================================================================

        GpuPipeline::GpuPipeline(PipelineMode mode,
                                 int imageWidth, int imageHeight,
                                 int numFeatures,
                                 float scaleFactor,
                                 int numLevels)
            : mode_(mode),
              imageWidth_(imageWidth),
              imageHeight_(imageHeight),
              numFeatures_(numFeatures),
              scaleFactor_(scaleFactor),
              numLevels_(numLevels),
              memPool_(nullptr),
              frameManager_(nullptr)
        {

            // Default intrinsics
            intrinsics_ = {0.0f, 0.0f, 0.0f, 0.0f};
            distortion_ = {};
        }

        GpuPipeline::~GpuPipeline()
        {
            shutdown();
        }

        // ============================================================================
        // Initialization
        // ============================================================================

        bool GpuPipeline::initialize()
        {
            if (initialized_)
                return true;

            try
            {
                // Get memory pool singleton
                memPool_ = &GpuMemoryPool::getInstance();
                frameManager_ = &GpuFrameManager::getInstance();

                // Create CUDA streams
                createStreams();

                // Create timing events
                CUDA_CHECK(cudaEventCreate(&eventStart_));
                CUDA_CHECK(cudaEventCreate(&eventPreprocess_));
                CUDA_CHECK(cudaEventCreate(&eventExtract_));
                CUDA_CHECK(cudaEventCreate(&eventGrid_));
                CUDA_CHECK(cudaEventCreate(&eventEnd_));

                // Create ORB extractor (5 params: nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST)
                constexpr int INI_TH_FAST = 20;
                constexpr int MIN_TH_FAST = 7;
                extractor_ = std::make_unique<ORBExtractorCuda>(
                    numFeatures_, scaleFactor_, numLevels_, INI_TH_FAST, MIN_TH_FAST);

                // For stereo mode, create second extractor
                if (mode_ == PipelineMode::STEREO || mode_ == PipelineMode::STEREO_INERTIAL)
                {
                    extractorRight_ = std::make_unique<ORBExtractorCuda>(
                        numFeatures_, scaleFactor_, numLevels_, INI_TH_FAST, MIN_TH_FAST);
                }

                // Create matcher (constructor takes nnratio and checkOri)
                matcher_ = std::make_unique<ORBMatcherCuda>(0.75f, true);

                // Create preprocessor (default constructor, initialize later with camera params)
                preprocessor_ = std::make_unique<ImagePreprocessorCuda>();

                // Create feature grid
                featureGrid_ = std::make_unique<FeatureGridCuda>(imageWidth_, imageHeight_);

                // Allocate buffers
                allocateBuffers();

                initialized_ = true;
                return true;
            }
            catch (const std::exception &e)
            {
                fprintf(stderr, "GpuPipeline initialization failed: %s\n", e.what());
                return false;
            }
        }

        void GpuPipeline::shutdown()
        {
            if (!initialized_)
                return;

            synchronize();

            // Destroy events
            cudaEventDestroy(eventStart_);
            cudaEventDestroy(eventPreprocess_);
            cudaEventDestroy(eventExtract_);
            cudaEventDestroy(eventGrid_);
            cudaEventDestroy(eventEnd_);

            destroyStreams();

            // Reset unique_ptrs
            extractor_.reset();
            extractorRight_.reset();
            matcher_.reset();
            preprocessor_.reset();
            featureGrid_.reset();

            // Release GPU mats
            gpuImageGray_.release();
            gpuImageUndist_.release();
            gpuImageRight_.release();
            gpuDepth_.release();

            initialized_ = false;
        }

        void GpuPipeline::createStreams()
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streamMain_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&streamRight_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&streamAsync_, cudaStreamNonBlocking));
        }

        void GpuPipeline::destroyStreams()
        {
            cudaStreamDestroy(streamMain_);
            cudaStreamDestroy(streamRight_);
            cudaStreamDestroy(streamAsync_);
        }

        void GpuPipeline::allocateBuffers()
        {
            // Pre-allocate GPU image buffers
            gpuImageGray_.create(imageHeight_, imageWidth_, CV_8UC1);
            gpuImageUndist_.create(imageHeight_, imageWidth_, CV_8UC1);

            if (mode_ == PipelineMode::STEREO || mode_ == PipelineMode::STEREO_INERTIAL)
            {
                gpuImageRight_.create(imageHeight_, imageWidth_, CV_8UC1);
            }

            if (mode_ == PipelineMode::RGBD || mode_ == PipelineMode::RGBD_INERTIAL)
            {
                gpuDepth_.create(imageHeight_, imageWidth_, CV_32FC1);
            }
        }

        // ============================================================================
        // Configuration
        // ============================================================================

        void GpuPipeline::setCameraIntrinsics(float fx, float fy, float cx, float cy)
        {
            intrinsics_.fx = fx;
            intrinsics_.fy = fy;
            intrinsics_.cx = cx;
            intrinsics_.cy = cy;
            intrinsics_.compute();

            // Re-initialize preprocessor with updated intrinsics
            if (preprocessor_ && hasDistortion_)
            {
                preprocessor_->initialize(intrinsics_, distortion_, imageWidth_, imageHeight_);
            }
        }

        void GpuPipeline::setDistortionParams(const std::vector<float> &distCoeffs, bool isFisheye)
        {
            if (distCoeffs.size() >= 4)
            {
                distortion_.k1 = distCoeffs[0];
                distortion_.k2 = distCoeffs[1];
                distortion_.p1 = distCoeffs[2];
                distortion_.p2 = distCoeffs[3];
                if (distCoeffs.size() >= 5)
                    distortion_.k3 = distCoeffs[4];
                distortion_.isFisheye = isFisheye;
                hasDistortion_ = true;

                // Initialize preprocessor with camera parameters
                if (preprocessor_)
                {
                    preprocessor_->initialize(intrinsics_, distortion_, imageWidth_, imageHeight_);
                }
            }
        }

        void GpuPipeline::setStereoConfig(const StereoConfig &config)
        {
            stereoConfig_ = config;
        }

        void GpuPipeline::setRGBDConfig(const RGBDConfig &config)
        {
            rgbdConfig_ = config;
        }

        // ============================================================================
        // Frame Processing - Monocular
        // ============================================================================

        GpuFrameResult GpuPipeline::processMonocular(const cv::Mat &image, double timestamp)
        {
            // Upload to GPU
            cv::cuda::GpuMat gpuImage;
            gpuImage.upload(image, cv::cuda::Stream::Null());
            return processMonocular(gpuImage, timestamp);
        }

        GpuFrameResult GpuPipeline::processMonocular(const cv::cuda::GpuMat &image, double timestamp)
        {
            GpuFrameResult result;

            if (!initialized_)
            {
                result.stage = ProcessingStage::ERROR;
                return result;
            }

            // Record start
            CUDA_CHECK(cudaEventRecord(eventStart_, streamMain_));
            result.stage = ProcessingStage::PREPROCESS;

            // Step 1: Preprocess (convert to grayscale, undistort)
            cv::cuda::GpuMat processedImage;
            cv::cuda::Stream cvStreamMain = cv::cuda::StreamAccessor::wrapStream(streamMain_);

            if (image.channels() == 3)
            {
                cv::cuda::cvtColor(image, gpuImageGray_, cv::COLOR_BGR2GRAY, 0, cvStreamMain);
                processedImage = gpuImageGray_;
            }
            else
            {
                processedImage = image;
            }

            if (hasDistortion_)
            {
                preprocessor_->undistort(processedImage, gpuImageUndist_, streamMain_);
                processedImage = gpuImageUndist_;
            }

            CUDA_CHECK(cudaEventRecord(eventPreprocess_, streamMain_));
            result.stage = ProcessingStage::DETECTION;

            // Step 2: Extract ORB features
            GpuKeyPointSoA keypoints;
            GpuDescriptorArray descriptorArray;

            int numKeypoints = (*extractor_)(processedImage, keypoints, descriptorArray);

            result.keypoints = keypoints;
            result.descriptorArray = descriptorArray; // Store descriptor array
            // Store descriptor count - actual descriptors are in descriptorArray
            result.numDescriptors = numKeypoints;

            CUDA_CHECK(cudaEventRecord(eventExtract_, streamMain_));
            result.stage = ProcessingStage::GRID_BUILD;

            // Step 3: Build feature grid
            if (numKeypoints > 0)
            {
                featureGrid_->buildGrid(result.keypoints, streamMain_);

                // Copy grid structure (pointers remain valid on GPU)
                result.grid = featureGrid_->getGrid();
            }

            CUDA_CHECK(cudaEventRecord(eventGrid_, streamMain_));

            // Step 4: Copy pyramid for visualization if needed
            result.pyramid = extractor_->getGpuPyramid();

            CUDA_CHECK(cudaEventRecord(eventEnd_, streamMain_));

            // Compute timing
            CUDA_CHECK(cudaStreamSynchronize(streamMain_));

            float preprocTime, extractTime, gridTime, totalTime;
            cudaEventElapsedTime(&preprocTime, eventStart_, eventPreprocess_);
            cudaEventElapsedTime(&extractTime, eventPreprocess_, eventExtract_);
            cudaEventElapsedTime(&gridTime, eventExtract_, eventGrid_);
            cudaEventElapsedTime(&totalTime, eventStart_, eventEnd_);

            result.stats.preprocessTimeMs = preprocTime;
            result.stats.pyramidTimeMs = 0;                    // Included in extract
            result.stats.detectionTimeMs = extractTime * 0.6f; // Approximate split
            result.stats.descriptionTimeMs = extractTime * 0.4f;
            result.stats.gridTimeMs = gridTime;
            result.stats.totalTimeMs = totalTime;
            result.stats.numKeypoints = numKeypoints;

            stats_ = result.stats;

            result.stage = ProcessingStage::COMPLETE;
            result.success = true;

            return result;
        }

        // ============================================================================
        // Frame Processing - Stereo
        // ============================================================================

        GpuFrameResult GpuPipeline::processStereo(const cv::Mat &imageLeft,
                                                  const cv::Mat &imageRight,
                                                  double timestamp)
        {
            cv::cuda::GpuMat gpuLeft, gpuRight;
            gpuLeft.upload(imageLeft, cv::cuda::Stream::Null());
            gpuRight.upload(imageRight, cv::cuda::Stream::Null());
            return processStereo(gpuLeft, gpuRight, timestamp);
        }

        GpuFrameResult GpuPipeline::processStereo(const cv::cuda::GpuMat &imageLeft,
                                                  const cv::cuda::GpuMat &imageRight,
                                                  double timestamp)
        {
            GpuFrameResult result;

            if (!initialized_ || !extractorRight_)
            {
                result.stage = ProcessingStage::ERROR;
                return result;
            }

            CUDA_CHECK(cudaEventRecord(eventStart_, streamMain_));

            // Process left image on main stream
            cv::cuda::GpuMat leftGray;
            cv::cuda::Stream cvStreamMainStereo = cv::cuda::StreamAccessor::wrapStream(streamMain_);
            cv::cuda::Stream cvStreamRightStereo = cv::cuda::StreamAccessor::wrapStream(streamRight_);

            if (imageLeft.channels() == 3)
            {
                cv::cuda::cvtColor(imageLeft, leftGray, cv::COLOR_BGR2GRAY, 0, cvStreamMainStereo);
            }
            else
            {
                leftGray = imageLeft;
            }

            // Process right image on separate stream (parallel)
            cv::cuda::GpuMat rightGray;
            if (imageRight.channels() == 3)
            {
                cv::cuda::cvtColor(imageRight, rightGray, cv::COLOR_BGR2GRAY, 0, cvStreamRightStereo);
            }
            else
            {
                rightGray = imageRight;
            }

            // Undistort both images (parallel)
            if (hasDistortion_)
            {
                preprocessor_->undistort(leftGray, gpuImageUndist_, streamMain_);
                preprocessor_->undistort(rightGray, gpuImageRight_, streamRight_);
                leftGray = gpuImageUndist_;
                rightGray = gpuImageRight_;
            }

            CUDA_CHECK(cudaEventRecord(eventPreprocess_, streamMain_));

            // Extract features from left image
            GpuDescriptorArray descLeft;
            int numLeft = (*extractor_)(leftGray, result.keypoints, descLeft);
            result.numDescriptors = numLeft;

            // Extract features from right image (parallel)
            GpuDescriptorArray descRight;
            int numRight = (*extractorRight_)(rightGray, result.keypointsRight, descRight);
            result.numDescriptorsRight = numRight;

            // Synchronize right stream before matching
            CUDA_CHECK(cudaStreamSynchronize(streamRight_));

            CUDA_CHECK(cudaEventRecord(eventExtract_, streamMain_));

            // Build feature grid for left image
            if (numLeft > 0)
            {
                featureGrid_->buildGrid(result.keypoints, streamMain_);
                result.grid = featureGrid_->getGrid();
            }

            CUDA_CHECK(cudaEventRecord(eventGrid_, streamMain_));

            // Compute stereo matches
            computeStereoMatches(result);

            CUDA_CHECK(cudaEventRecord(eventEnd_, streamMain_));
            CUDA_CHECK(cudaStreamSynchronize(streamMain_));

            // Timing
            float totalTime;
            cudaEventElapsedTime(&totalTime, eventStart_, eventEnd_);
            result.stats.totalTimeMs = totalTime;
            result.stats.numKeypoints = numLeft;

            stats_ = result.stats;
            result.stage = ProcessingStage::COMPLETE;
            result.success = true;

            return result;
        }

        void GpuPipeline::computeStereoMatches(GpuFrameResult &result)
        {
            if (result.numDescriptors == 0 || result.numDescriptorsRight == 0)
            {
                result.numStereoMatches = 0;
                return;
            }

            // Allocate match buffers if needed
            if (result.stereoMatches.size() < static_cast<size_t>(result.numDescriptors))
            {
                result.stereoMatches = GpuArray<int>(result.numDescriptors);
                result.stereoDepths = GpuArray<float>(result.numDescriptors);
            }

            // Use spatial matching constrained by epipolar geometry
            // For rectified stereo, features should be on same row (within tolerance)
            const float epipolarTolerance = 2.0f; // pixels

            // Match using ORB matcher
            // Note: For stereo matching, we'd use matchGpu with GpuDescriptorArrays
            // This is a placeholder - full implementation needs descriptor arrays stored
            int numMatches = 0;
            // TODO: Implement proper stereo matching with stored descriptor arrays

            // TODO: Filter matches by epipolar constraint and compute depth
            // For now, store raw matches
            result.numStereoMatches = numMatches;
        }

        // ============================================================================
        // Frame Processing - RGBD
        // ============================================================================

        GpuFrameResult GpuPipeline::processRGBD(const cv::Mat &imageRGB,
                                                const cv::Mat &depth,
                                                double timestamp)
        {
            cv::cuda::GpuMat gpuRGB, gpuDepth;
            gpuRGB.upload(imageRGB, cv::cuda::Stream::Null());
            gpuDepth.upload(depth, cv::cuda::Stream::Null());
            return processRGBD(gpuRGB, gpuDepth, timestamp);
        }

        GpuFrameResult GpuPipeline::processRGBD(const cv::cuda::GpuMat &imageRGB,
                                                const cv::cuda::GpuMat &depth,
                                                double timestamp)
        {
            // First, process as monocular to get features
            GpuFrameResult result = processMonocular(imageRGB, timestamp);

            if (!result.success)
                return result;

            // Store depth map
            result.depthMap = depth;

            // Compute depth for each keypoint
            computeDepthFromRGBD(depth, result);

            return result;
        }

        void GpuPipeline::computeDepthFromRGBD(const cv::cuda::GpuMat &depth,
                                               GpuFrameResult &result)
        {
            // TODO: Implement GPU kernel to sample depth at keypoint locations
            // For now, this is a placeholder

            if (result.stereoDepths.size() < static_cast<size_t>(result.numDescriptors))
            {
                result.stereoDepths = GpuArray<float>(result.numDescriptors);
            }
        }

        // ============================================================================
        // Map Point Operations
        // ============================================================================

        void GpuPipeline::projectMapPoints(const float *mapPointsWorld,
                                           const float *Rcw,
                                           const float *tcw,
                                           int numPoints,
                                           GpuProjection *projections)
        {
            // TODO: Implement projection kernel
            // This will project world points to image coordinates using:
            // p_cam = Rcw * p_world + tcw
            // p_img = K * p_cam (normalized)
        }

        int GpuPipeline::searchByProjection(const GpuFrameResult &frame,
                                            const GpuProjection *projections,
                                            const uint32_t *mapPointDescriptors,
                                            int numProjections,
                                            int *matchedMapPointIdx,
                                            int *matchedKeypointIdx,
                                            float radiusMultiplier)
        {
            if (numProjections == 0 || frame.numDescriptors == 0)
                return 0;

            // Step 1: Find candidate keypoints for each projection using grid
            const int maxCandidatesPerProj = 10;
            GpuArray<int> candidates(numProjections * maxCandidatesPerProj);
            GpuArray<int> candidateCounts(numProjections);

            // Compute search radii based on scale
            GpuArray<float> radii(numProjections);
            // TODO: Fill radii based on predicted scale level

            // Batch radius search
            featureGrid_->getFeaturesInAreaBatch(
                projections, radii.data(), numProjections,
                candidates.data(), candidateCounts.data(),
                maxCandidatesPerProj, streamMain_);

            // Step 2: Match descriptors for candidates
            // TODO: Implement candidate matching kernel

            return 0; // Return number of matches
        }

        // ============================================================================
        // Utility Methods
        // ============================================================================

        void GpuPipeline::downloadKeypoints(const GpuFrameResult &result,
                                            std::vector<cv::KeyPoint> &keypoints)
        {
            if (result.numDescriptors == 0 || result.keypoints.x == nullptr)
            {
                keypoints.clear();
                return;
            }

            keypoints.resize(result.numDescriptors);

            // Download SoA data
            std::vector<float> x(result.numDescriptors);
            std::vector<float> y(result.numDescriptors);
            std::vector<float> size(result.numDescriptors);
            std::vector<float> angle(result.numDescriptors);
            std::vector<float> response(result.numDescriptors);
            std::vector<int> octave(result.numDescriptors);

            CUDA_CHECK(cudaMemcpy(x.data(), result.keypoints.x,
                                  result.numDescriptors * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y.data(), result.keypoints.y,
                                  result.numDescriptors * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(size.data(), result.keypoints.size,
                                  result.numDescriptors * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(angle.data(), result.keypoints.angle,
                                  result.numDescriptors * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(response.data(), result.keypoints.response,
                                  result.numDescriptors * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(octave.data(), result.keypoints.octave,
                                  result.numDescriptors * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < result.numDescriptors; ++i)
            {
                keypoints[i].pt.x = x[i];
                keypoints[i].pt.y = y[i];
                keypoints[i].size = size[i];
                keypoints[i].angle = angle[i];
                keypoints[i].response = response[i];
                keypoints[i].octave = octave[i];
            }
        }

        void GpuPipeline::downloadDescriptors(const GpuFrameResult &result,
                                              cv::Mat &descriptors)
        {
            if (result.numDescriptors == 0 || result.descriptorArray.descriptors == nullptr)
            {
                descriptors = cv::Mat();
                return;
            }

            descriptors.create(result.numDescriptors, 32, CV_8UC1);

            // Download from GpuDescriptorArray
            CUDA_CHECK(cudaMemcpy(descriptors.data, result.descriptorArray.descriptors,
                                  result.numDescriptors * sizeof(GpuDescriptor), cudaMemcpyDeviceToHost));
        }

        GpuPipelineStats GpuPipeline::getStats() const
        {
            return stats_;
        }

        void GpuPipeline::resetStats()
        {
            stats_ = GpuPipelineStats();
        }

        void GpuPipeline::synchronize()
        {
            CUDA_CHECK(cudaStreamSynchronize(streamMain_));
            CUDA_CHECK(cudaStreamSynchronize(streamRight_));
            CUDA_CHECK(cudaStreamSynchronize(streamAsync_));
        }

        // ============================================================================
        // Factory Function
        // ============================================================================

        std::unique_ptr<GpuPipeline> createGpuPipeline(
            PipelineMode mode,
            int imageWidth,
            int imageHeight,
            int numFeatures,
            float scaleFactor,
            int numLevels)
        {

            auto pipeline = std::make_unique<GpuPipeline>(
                mode, imageWidth, imageHeight, numFeatures, scaleFactor, numLevels);

            if (!pipeline->initialize())
            {
                return nullptr;
            }

            return pipeline;
        }

    } // namespace cuda
} // namespace ORB_SLAM3
