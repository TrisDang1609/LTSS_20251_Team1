/**
 * Image Preprocessing CUDA - GPU-Accelerated Image Operations
 *
 * Provides GPU-resident image preprocessing for the ORB-SLAM3 pipeline:
 * 1. Color conversion (BGR/RGB to Grayscale)
 * 2. Lens undistortion (Pinhole, Fisheye)
 * 3. Image resizing and border handling
 * 4. Histogram equalization (CLAHE)
 *
 * All operations keep data on GPU to avoid host-device transfers.
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 */

#ifndef IMAGE_PREPROCESS_CUDA_H
#define IMAGE_PREPROCESS_CUDA_H

#include "CudaUtils.h"
#include "GpuTypes.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Camera Distortion Parameters
        // ============================================================================

        struct DistortionParams
        {
            // Pinhole model (5 params)
            float k1 = 0.0f;
            float k2 = 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            float k3 = 0.0f;

            // Fisheye (Kannala-Brandt) model (4 params)
            float kb1 = 0.0f;
            float kb2 = 0.0f;
            float kb3 = 0.0f;
            float kb4 = 0.0f;

            bool isFisheye = false;
        };

        struct CameraIntrinsics
        {
            float fx, fy;       // Focal lengths
            float cx, cy;       // Principal point
            float invfx, invfy; // Inverse focal lengths

            void compute()
            {
                invfx = 1.0f / fx;
                invfy = 1.0f / fy;
            }
        };

        // ============================================================================
        // Image Preprocessor CUDA Class
        // ============================================================================

        class ImagePreprocessorCuda
        {
        public:
            ImagePreprocessorCuda();
            ~ImagePreprocessorCuda();

            /**
             * Initialize with camera parameters
             */
            void initialize(const CameraIntrinsics &intrinsics,
                            const DistortionParams &distortion,
                            int imageWidth, int imageHeight);

            /**
             * Full preprocessing pipeline (GPU-resident)
             * Converts color, undistorts, and prepares for feature extraction
             *
             * @param inputImage Input image (can be on CPU or GPU)
             * @param outputImage Output grayscale undistorted image (GPU)
             * @param stream CUDA stream for async operation
             */
            void preprocess(const cv::cuda::GpuMat &inputImage,
                            cv::cuda::GpuMat &outputImage,
                            cudaStream_t stream = 0);

            /**
             * Upload CPU image to GPU and preprocess
             */
            void preprocessFromCpu(const cv::Mat &cpuImage,
                                   cv::cuda::GpuMat &outputImage,
                                   cudaStream_t stream = 0);

            // ========================================================================
            // Individual Operations
            // ========================================================================

            /**
             * Convert to grayscale on GPU
             */
            void convertToGray(const cv::cuda::GpuMat &input,
                               cv::cuda::GpuMat &output,
                               cudaStream_t stream = 0);

            /**
             * Undistort image on GPU
             */
            void undistort(const cv::cuda::GpuMat &input,
                           cv::cuda::GpuMat &output,
                           cudaStream_t stream = 0);

            /**
             * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
             */
            void applyClahe(const cv::cuda::GpuMat &input,
                            cv::cuda::GpuMat &output,
                            double clipLimit = 3.0,
                            cv::Size tileSize = cv::Size(8, 8),
                            cudaStream_t stream = 0);

            /**
             * Resize with interpolation
             */
            void resize(const cv::cuda::GpuMat &input,
                        cv::cuda::GpuMat &output,
                        cv::Size newSize,
                        int interpolation = cv::INTER_LINEAR,
                        cudaStream_t stream = 0);

            /**
             * Add border (for pyramid edge handling)
             */
            void addBorder(const cv::cuda::GpuMat &input,
                           cv::cuda::GpuMat &output,
                           int borderSize,
                           int borderType = cv::BORDER_REFLECT_101,
                           cudaStream_t stream = 0);

            // ========================================================================
            // Stereo Operations
            // ========================================================================

            /**
             * Rectify stereo pair
             */
            void rectifyStereo(const cv::cuda::GpuMat &leftInput,
                               const cv::cuda::GpuMat &rightInput,
                               cv::cuda::GpuMat &leftOutput,
                               cv::cuda::GpuMat &rightOutput,
                               cudaStream_t stream = 0);

            /**
             * Compute disparity map (using OpenCV CUDA stereo matchers)
             */
            void computeDisparity(const cv::cuda::GpuMat &left,
                                  const cv::cuda::GpuMat &right,
                                  cv::cuda::GpuMat &disparity,
                                  cudaStream_t stream = 0);

            // ========================================================================
            // Utility Methods
            // ========================================================================

            /**
             * Get undistortion maps for remap (pre-computed)
             */
            void getUndistortMaps(cv::cuda::GpuMat &mapX, cv::cuda::GpuMat &mapY) const;

            /**
             * Check if initialized
             */
            bool isInitialized() const { return initialized_; }

            /**
             * Get image dimensions
             */
            int getWidth() const { return imageWidth_; }
            int getHeight() const { return imageHeight_; }

        private:
            // Parameters
            CameraIntrinsics intrinsics_;
            DistortionParams distortion_;
            int imageWidth_;
            int imageHeight_;
            bool initialized_;

            // Pre-computed undistortion maps (GPU-resident)
            cv::cuda::GpuMat undistortMapX_;
            cv::cuda::GpuMat undistortMapY_;

            // Stereo rectification maps
            cv::cuda::GpuMat leftRectMapX_, leftRectMapY_;
            cv::cuda::GpuMat rightRectMapX_, rightRectMapY_;

            // CLAHE filter
            cv::Ptr<cv::cuda::CLAHE> clahe_;

            // Temporary buffers
            cv::cuda::GpuMat tempGray_;
            cv::cuda::GpuMat tempUndistort_;

            // Internal methods
            void computeUndistortionMaps();
            void computeFisheyeUndistortionMaps();
        };

        // ============================================================================
        // CUDA Kernel Declarations
        // ============================================================================

        /**
         * Custom undistortion kernel for Pinhole model
         * More efficient than OpenCV remap for this specific case
         */
        void launchUndistortPinholeKernel(
            const unsigned char *input,
            unsigned char *output,
            int width, int height, int step,
            float fx, float fy, float cx, float cy,
            float k1, float k2, float p1, float p2, float k3,
            cudaStream_t stream);

        /**
         * Custom undistortion kernel for Fisheye (Kannala-Brandt) model
         */
        void launchUndistortFisheyeKernel(
            const unsigned char *input,
            unsigned char *output,
            int width, int height, int step,
            float fx, float fy, float cx, float cy,
            float kb1, float kb2, float kb3, float kb4,
            cudaStream_t stream);

        /**
         * Fast grayscale conversion kernel
         * Optimized for memory coalescing
         */
        void launchRgbToGrayKernel(
            const unsigned char *input,
            unsigned char *output,
            int width, int height,
            int inputStep, int outputStep,
            bool isBgr,
            cudaStream_t stream);

        /**
         * Image normalization kernel
         * For neural network preprocessing
         */
        void launchNormalizeKernel(
            const unsigned char *input,
            float *output,
            int width, int height, int step,
            float mean, float stddev,
            cudaStream_t stream);

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // IMAGE_PREPROCESS_CUDA_H
