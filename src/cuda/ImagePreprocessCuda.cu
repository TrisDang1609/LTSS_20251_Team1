/**
 * Image Preprocessing CUDA Implementation
 *
 * GPU-accelerated image preprocessing for ORB-SLAM3.
 */

#include "cuda/ImagePreprocessCuda.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // CUDA Kernels
        // ============================================================================

        __global__ void rgbToGrayKernel(
            const unsigned char *__restrict__ input,
            unsigned char *__restrict__ output,
            int width, int height,
            int inputStep, int outputStep,
            bool isBgr)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height)
                return;

            int inputIdx = y * inputStep + x * 3;
            int outputIdx = y * outputStep + x;

            unsigned char r, g, b;
            if (isBgr)
            {
                b = input[inputIdx];
                g = input[inputIdx + 1];
                r = input[inputIdx + 2];
            }
            else
            {
                r = input[inputIdx];
                g = input[inputIdx + 1];
                b = input[inputIdx + 2];
            }

            // ITU-R BT.601 conversion
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            output[outputIdx] = static_cast<unsigned char>(gray);
        }

        __global__ void undistortPinholeKernel(
            const unsigned char *__restrict__ input,
            unsigned char *__restrict__ output,
            int width, int height, int step,
            float fx, float fy, float cx, float cy,
            float invfx, float invfy,
            float k1, float k2, float p1, float p2, float k3)
        {
            int u = blockIdx.x * blockDim.x + threadIdx.x;
            int v = blockIdx.y * blockDim.y + threadIdx.y;

            if (u >= width || v >= height)
                return;

            // Convert to normalized camera coordinates
            float x = (u - cx) * invfx;
            float y = (v - cy) * invfy;

            float r2 = x * x + y * y;
            float r4 = r2 * r2;
            float r6 = r4 * r2;

            // Radial distortion
            float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

            // Tangential distortion
            float x_dist = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
            float y_dist = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;

            // Convert back to pixel coordinates
            float u_dist = x_dist * fx + cx;
            float v_dist = y_dist * fy + cy;

            // Bilinear interpolation
            int u0 = __float2int_rd(u_dist);
            int v0 = __float2int_rd(v_dist);
            int u1 = u0 + 1;
            int v1 = v0 + 1;

            if (u0 < 0 || v0 < 0 || u1 >= width || v1 >= height)
            {
                output[v * step + u] = 0;
                return;
            }

            float fu = u_dist - u0;
            float fv = v_dist - v0;

            float p00 = input[v0 * step + u0];
            float p01 = input[v0 * step + u1];
            float p10 = input[v1 * step + u0];
            float p11 = input[v1 * step + u1];

            float value = (1.0f - fu) * (1.0f - fv) * p00 +
                          fu * (1.0f - fv) * p01 +
                          (1.0f - fu) * fv * p10 +
                          fu * fv * p11;

            output[v * step + u] = static_cast<unsigned char>(value);
        }

        __global__ void undistortFisheyeKernel(
            const unsigned char *__restrict__ input,
            unsigned char *__restrict__ output,
            int width, int height, int step,
            float fx, float fy, float cx, float cy,
            float invfx, float invfy,
            float kb1, float kb2, float kb3, float kb4)
        {
            int u = blockIdx.x * blockDim.x + threadIdx.x;
            int v = blockIdx.y * blockDim.y + threadIdx.y;

            if (u >= width || v >= height)
                return;

            // Convert to normalized camera coordinates
            float x = (u - cx) * invfx;
            float y = (v - cy) * invfy;

            float r = sqrtf(x * x + y * y);

            if (r < 1e-8f)
            {
                output[v * step + u] = input[v * step + u];
                return;
            }

            // Inverse fisheye model (Kannala-Brandt)
            float theta = r;

            // Newton-Raphson iteration for inverse
            for (int i = 0; i < 5; ++i)
            {
                float theta2 = theta * theta;
                float theta3 = theta2 * theta;
                float theta5 = theta3 * theta2;
                float theta7 = theta5 * theta2;
                float theta9 = theta7 * theta2;

                float f = theta + kb1 * theta3 + kb2 * theta5 + kb3 * theta7 + kb4 * theta9 - r;
                float df = 1.0f + 3.0f * kb1 * theta2 + 5.0f * kb2 * theta2 * theta2 +
                           7.0f * kb3 * theta2 * theta2 * theta2 + 9.0f * kb4 * theta2 * theta2 * theta2 * theta2;

                theta -= f / df;
            }

            // Distorted coordinates
            float scale = (r < 1e-8f) ? 1.0f : tanf(theta) / r;
            float x_dist = x * scale;
            float y_dist = y * scale;

            float u_dist = x_dist * fx + cx;
            float v_dist = y_dist * fy + cy;

            // Bilinear interpolation
            int u0 = __float2int_rd(u_dist);
            int v0 = __float2int_rd(v_dist);

            if (u0 < 0 || v0 < 0 || u0 >= width - 1 || v0 >= height - 1)
            {
                output[v * step + u] = 0;
                return;
            }

            float fu = u_dist - u0;
            float fv = v_dist - v0;

            float p00 = input[v0 * step + u0];
            float p01 = input[v0 * step + u0 + 1];
            float p10 = input[(v0 + 1) * step + u0];
            float p11 = input[(v0 + 1) * step + u0 + 1];

            float value = (1.0f - fu) * (1.0f - fv) * p00 +
                          fu * (1.0f - fv) * p01 +
                          (1.0f - fu) * fv * p10 +
                          fu * fv * p11;

            output[v * step + u] = static_cast<unsigned char>(value);
        }

        __global__ void normalizeKernel(
            const unsigned char *__restrict__ input,
            float *__restrict__ output,
            int width, int height, int step,
            float mean, float invStddev)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height)
                return;

            int idx = y * step + x;
            output[idx] = (static_cast<float>(input[idx]) - mean) * invStddev;
        }

        // ============================================================================
        // ImagePreprocessorCuda Implementation
        // ============================================================================

        ImagePreprocessorCuda::ImagePreprocessorCuda()
            : imageWidth_(0), imageHeight_(0), initialized_(false)
        {
        }

        ImagePreprocessorCuda::~ImagePreprocessorCuda()
        {
        }

        void ImagePreprocessorCuda::initialize(const CameraIntrinsics &intrinsics,
                                               const DistortionParams &distortion,
                                               int imageWidth, int imageHeight)
        {
            intrinsics_ = intrinsics;
            intrinsics_.compute(); // Compute inverse focal lengths
            distortion_ = distortion;
            imageWidth_ = imageWidth;
            imageHeight_ = imageHeight;

            // Pre-compute undistortion maps
            if (distortion_.isFisheye)
            {
                computeFisheyeUndistortionMaps();
            }
            else
            {
                computeUndistortionMaps();
            }

            // Create CLAHE filter
            clahe_ = cv::cuda::createCLAHE(3.0, cv::Size(8, 8));

            // Allocate temporary buffers
            tempGray_.create(imageHeight_, imageWidth_, CV_8UC1);
            tempUndistort_.create(imageHeight_, imageWidth_, CV_8UC1);

            initialized_ = true;
        }

        void ImagePreprocessorCuda::computeUndistortionMaps()
        {
            // Use OpenCV to compute maps on CPU, then upload
            cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << intrinsics_.fx, 0, intrinsics_.cx,
                                    0, intrinsics_.fy, intrinsics_.cy,
                                    0, 0, 1);

            cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << distortion_.k1, distortion_.k2,
                                  distortion_.p1, distortion_.p2,
                                  distortion_.k3);

            cv::Mat mapX, mapY;
            cv::initUndistortRectifyMap(cameraMatrix, distCoeffs,
                                        cv::Mat(), cameraMatrix,
                                        cv::Size(imageWidth_, imageHeight_),
                                        CV_32FC1, mapX, mapY);

            undistortMapX_.upload(mapX);
            undistortMapY_.upload(mapY);
        }

        void ImagePreprocessorCuda::computeFisheyeUndistortionMaps()
        {
            // For fisheye, we'll use our custom kernel directly
            // Maps are not pre-computed for fisheye
            undistortMapX_.create(imageHeight_, imageWidth_, CV_32FC1);
            undistortMapY_.create(imageHeight_, imageWidth_, CV_32FC1);
        }

        void ImagePreprocessorCuda::preprocess(const cv::cuda::GpuMat &inputImage,
                                               cv::cuda::GpuMat &outputImage,
                                               cudaStream_t stream)
        {
            if (!initialized_)
            {
                throw std::runtime_error("ImagePreprocessorCuda not initialized");
            }

            cv::cuda::GpuMat grayImage;

            // Convert to grayscale if needed
            if (inputImage.channels() == 3)
            {
                convertToGray(inputImage, grayImage, stream);
            }
            else
            {
                grayImage = inputImage;
            }

            // Undistort
            undistort(grayImage, outputImage, stream);
        }

        void ImagePreprocessorCuda::preprocessFromCpu(const cv::Mat &cpuImage,
                                                      cv::cuda::GpuMat &outputImage,
                                                      cudaStream_t stream)
        {
            cv::cuda::GpuMat gpuImage;

            // Async upload using stream
            cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
            gpuImage.upload(cpuImage, cvStream);

            preprocess(gpuImage, outputImage, stream);
        }

        void ImagePreprocessorCuda::convertToGray(const cv::cuda::GpuMat &input,
                                                  cv::cuda::GpuMat &output,
                                                  cudaStream_t stream)
        {
            if (input.channels() == 1)
            {
                output = input;
                return;
            }

            output.create(input.rows, input.cols, CV_8UC1);

            dim3 block(16, 16);
            dim3 grid((input.cols + block.x - 1) / block.x,
                      (input.rows + block.y - 1) / block.y);

            rgbToGrayKernel<<<grid, block, 0, stream>>>(
                input.ptr<unsigned char>(),
                output.ptr<unsigned char>(),
                input.cols, input.rows,
                static_cast<int>(input.step),
                static_cast<int>(output.step),
                true); // Assume BGR (OpenCV default)

            CUDA_CHECK_LAST();
        }

        void ImagePreprocessorCuda::undistort(const cv::cuda::GpuMat &input,
                                              cv::cuda::GpuMat &output,
                                              cudaStream_t stream)
        {
            // Check if distortion is negligible
            if (std::abs(distortion_.k1) < 1e-6f &&
                std::abs(distortion_.k2) < 1e-6f &&
                std::abs(distortion_.p1) < 1e-6f &&
                std::abs(distortion_.p2) < 1e-6f)
            {
                output = input;
                return;
            }

            output.create(input.rows, input.cols, input.type());

            if (distortion_.isFisheye)
            {
                // Custom fisheye undistortion
                dim3 block(16, 16);
                dim3 grid((input.cols + block.x - 1) / block.x,
                          (input.rows + block.y - 1) / block.y);

                undistortFisheyeKernel<<<grid, block, 0, stream>>>(
                    input.ptr<unsigned char>(),
                    output.ptr<unsigned char>(),
                    input.cols, input.rows,
                    static_cast<int>(input.step),
                    intrinsics_.fx, intrinsics_.fy,
                    intrinsics_.cx, intrinsics_.cy,
                    intrinsics_.invfx, intrinsics_.invfy,
                    distortion_.kb1, distortion_.kb2,
                    distortion_.kb3, distortion_.kb4);

                CUDA_CHECK_LAST();
            }
            else
            {
                // Use pre-computed maps with OpenCV remap
                cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
                cv::cuda::remap(input, output, undistortMapX_, undistortMapY_,
                                cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                cv::Scalar(), cvStream);
            }
        }

        void ImagePreprocessorCuda::applyClahe(const cv::cuda::GpuMat &input,
                                               cv::cuda::GpuMat &output,
                                               double clipLimit,
                                               cv::Size tileSize,
                                               cudaStream_t stream)
        {
            clahe_->setClipLimit(clipLimit);
            clahe_->setTilesGridSize(tileSize);

            cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
            clahe_->apply(input, output, cvStream);
        }

        void ImagePreprocessorCuda::resize(const cv::cuda::GpuMat &input,
                                           cv::cuda::GpuMat &output,
                                           cv::Size newSize,
                                           int interpolation,
                                           cudaStream_t stream)
        {
            cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
            cv::cuda::resize(input, output, newSize, 0, 0, interpolation, cvStream);
        }

        void ImagePreprocessorCuda::addBorder(const cv::cuda::GpuMat &input,
                                              cv::cuda::GpuMat &output,
                                              int borderSize,
                                              int borderType,
                                              cudaStream_t stream)
        {
            cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
            cv::cuda::copyMakeBorder(input, output,
                                     borderSize, borderSize,
                                     borderSize, borderSize,
                                     borderType, cv::Scalar(), cvStream);
        }

        void ImagePreprocessorCuda::getUndistortMaps(cv::cuda::GpuMat &mapX,
                                                     cv::cuda::GpuMat &mapY) const
        {
            mapX = undistortMapX_;
            mapY = undistortMapY_;
        }

        // ============================================================================
        // Kernel Launch Wrappers
        // ============================================================================

        void launchRgbToGrayKernel(
            const unsigned char *input,
            unsigned char *output,
            int width, int height,
            int inputStep, int outputStep,
            bool isBgr,
            cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);

            rgbToGrayKernel<<<grid, block, 0, stream>>>(
                input, output, width, height, inputStep, outputStep, isBgr);
        }

        void launchUndistortPinholeKernel(
            const unsigned char *input,
            unsigned char *output,
            int width, int height, int step,
            float fx, float fy, float cx, float cy,
            float k1, float k2, float p1, float p2, float k3,
            cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);

            undistortPinholeKernel<<<grid, block, 0, stream>>>(
                input, output, width, height, step,
                fx, fy, cx, cy, 1.0f / fx, 1.0f / fy,
                k1, k2, p1, p2, k3);
        }

        void launchUndistortFisheyeKernel(
            const unsigned char *input,
            unsigned char *output,
            int width, int height, int step,
            float fx, float fy, float cx, float cy,
            float kb1, float kb2, float kb3, float kb4,
            cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);

            undistortFisheyeKernel<<<grid, block, 0, stream>>>(
                input, output, width, height, step,
                fx, fy, cx, cy, 1.0f / fx, 1.0f / fy,
                kb1, kb2, kb3, kb4);
        }

        void launchNormalizeKernel(
            const unsigned char *input,
            float *output,
            int width, int height, int step,
            float mean, float stddev,
            cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);

            normalizeKernel<<<grid, block, 0, stream>>>(
                input, output, width, height, step, mean, 1.0f / stddev);
        }

    } // namespace cuda
} // namespace ORB_SLAM3
