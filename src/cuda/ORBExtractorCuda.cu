/**
 * ORB Extractor CUDA - GPU-Accelerated ORB Feature Extraction Implementation
 *
 * Full CUDA implementation of ORB feature extraction with:
 * - Custom FAST-9 detection kernel
 * - Warp-parallel descriptor computation
 * - Thrust-based keypoint distribution
 * - Zero-copy pipeline design
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 * CUDA: 13.0, C++17
 */

#include "cuda/ORBExtractorCuda.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <set>

namespace cg = cooperative_groups;

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Constant Memory Declarations
        // ============================================================================

        // ORB descriptor pattern (512 point pairs)
        __constant__ int2 c_pattern[512];

        // Umax array for IC_Angle (HALF_PATCH_SIZE + 1 = 16 elements)
        __constant__ int c_umax[16];

        // FAST circle pattern offsets (16 pixels)
        __constant__ int2 c_fastCircle[16];

        // Bit pattern for FAST score computation
        static const int FAST_CIRCLE[16][2] = {
            {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

        // ORB bit pattern (matching original ORB-SLAM3)
        static const int BIT_PATTERN_31[256 * 4] = {
            8, -3, 9, 5, 4, 2, 7, -12, -11, 9, -8, 2, 7, -12, 12, -13,
            2, -13, 2, 12, 1, -7, 1, 6, -2, -10, -2, -4, -13, -13, -11, -8,
            -13, -3, -12, -9, 10, 4, 11, 9, -13, -8, -8, -9, -11, 7, -9, 12,
            7, 7, 12, 6, -4, -5, -3, 0, -13, 2, -12, -3, -9, 0, -7, 5,
            12, -6, 12, -1, -3, 6, -2, 12, -6, -13, -4, -8, 11, -13, 12, -8,
            4, 7, 5, 1, 5, -3, 10, -3, 3, -7, 6, 12, -8, -7, -6, -2,
            -2, 11, -1, -10, -13, 12, -8, 10, -7, 3, -5, -3, -4, 2, -3, 7,
            -10, -12, -6, 11, 5, -12, 6, -7, 5, -6, 7, -1, 1, 0, 4, -5,
            9, 11, 11, -13, 4, 7, 4, 12, 2, -1, 4, 4, -4, -12, -2, 7,
            -8, -5, -7, -10, 4, 11, 9, 12, 0, -8, 1, -13, -13, -2, -8, 2,
            -3, -2, -2, 3, -6, 9, -4, -9, 8, 12, 10, 7, 0, 9, 1, 3,
            7, -5, 11, -10, -13, -6, -11, 0, 10, 7, 12, 1, -6, -3, -6, 12,
            10, -9, 12, -4, -13, 8, -8, -12, -13, 0, -8, -4, 3, 3, 7, 8,
            5, 7, 10, -7, -1, 7, 1, -12, 3, -10, 5, 6, 2, -4, 3, -10,
            -13, 0, -13, 5, -13, -7, -12, 12, -13, 3, -11, 8, -7, 12, -4, 7,
            6, -10, 12, 8, -9, -1, -7, -6, -2, -5, 0, 12, -12, 5, -7, 5,
            3, -10, 8, -13, -7, -7, -4, 5, -3, -2, -1, -7, 2, 9, 5, -11,
            -11, -13, -5, -13, -1, 6, 0, -1, 5, -3, 5, 2, -4, -13, -4, 12,
            -9, -6, -9, 6, -12, -10, -8, -4, 10, 2, 12, -3, 7, 12, 12, 12,
            -7, -13, -6, 5, -4, 9, -3, 4, 7, -1, 12, 2, -7, 6, -5, 1,
            -13, 11, -12, 5, -3, 7, -2, -6, 7, -8, 12, -7, -13, -7, -11, -12,
            1, -3, 12, 12, 2, -6, 3, 0, -4, 3, -2, -13, -1, -13, 1, 9,
            7, 1, 8, -6, 1, -1, 3, 12, 9, 1, 12, 6, -1, -9, -1, 3,
            -13, -13, -10, 5, 7, 7, 10, 12, 12, -5, 12, 9, 6, 3, 7, 11,
            5, -13, 6, 10, 2, -12, 2, 3, 3, 8, 4, -6, 2, 6, 12, -13,
            9, -12, 10, 3, -8, 4, -7, 9, -11, 12, -4, -6, 1, 12, 2, -8,
            6, -9, 7, -4, 2, 3, 3, -2, 6, 3, 11, 0, 3, -3, 8, -8,
            7, 8, 9, 3, -11, -5, -6, -4, -10, 11, -5, 10, -5, -8, -3, 12,
            -10, 5, -9, 0, 8, -1, 12, -6, 4, -6, 6, -11, -10, 12, -8, 7,
            4, -2, 6, 7, -2, 0, -2, 12, -5, -8, -5, 2, 7, -6, 10, 12,
            -9, -13, -8, -8, -5, -13, -5, -2, 8, -8, 9, -13, -9, -11, -9, 0,
            1, -8, 1, -2, 7, -4, 9, 1, -2, 1, -1, -4, 11, -6, 12, -11};

        // ============================================================================
        // FAST Detection Kernel
        // ============================================================================

        __device__ __forceinline__ bool isFastCorner9(
            const unsigned char *__restrict__ image,
            int x, int y, int step, int threshold,
            const int2 *circle)
        {
            int center = image[y * step + x];
            int th_high = center + threshold;
            int th_low = center - threshold;

            // Check pixels 0, 4, 8, 12 first (quick reject)
            int p0 = image[(y + circle[0].y) * step + x + circle[0].x];
            int p4 = image[(y + circle[4].y) * step + x + circle[4].x];
            int p8 = image[(y + circle[8].y) * step + x + circle[8].x];
            int p12 = image[(y + circle[12].y) * step + x + circle[12].x];

            int countBright = (p0 > th_high) + (p4 > th_high) + (p8 > th_high) + (p12 > th_high);
            int countDark = (p0 < th_low) + (p4 < th_low) + (p8 < th_low) + (p12 < th_low);

            if (countBright < 3 && countDark < 3)
                return false;

            // Full check - need 9 contiguous pixels
            unsigned int brightMask = 0, darkMask = 0;

#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                int pixel = image[(y + circle[i].y) * step + x + circle[i].x];
                if (pixel > th_high)
                    brightMask |= (1u << i);
                if (pixel < th_low)
                    darkMask |= (1u << i);
            }

            // Check for 9 contiguous bits (wrapping around)
            unsigned int mask9 = 0x1FF; // 9 bits set

            for (int i = 0; i < 16; ++i)
            {
                unsigned int rotBright = ((brightMask >> i) | (brightMask << (16 - i))) & 0xFFFF;
                unsigned int rotDark = ((darkMask >> i) | (darkMask << (16 - i))) & 0xFFFF;
                if ((rotBright & mask9) == mask9 || (rotDark & mask9) == mask9)
                {
                    return true;
                }
            }

            return false;
        }

        __device__ __forceinline__ int computeFastScore(
            const unsigned char *__restrict__ image,
            int x, int y, int step, int threshold,
            const int2 *circle)
        {
            int center = image[y * step + x];
            int score = 0;

#pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                int diff = abs(image[(y + circle[i].y) * step + x + circle[i].x] - center);
                if (diff > threshold)
                {
                    score += diff - threshold;
                }
            }

            return score;
        }

        __global__ void fastDetectionKernel(
            const unsigned char *__restrict__ image,
            int width, int height, int step,
            GpuFastResponse *__restrict__ responses,
            int *__restrict__ responseCount,
            int threshold,
            int maxKeypoints,
            int level)
        {
            // Border must account for:
            // - FAST circle radius (3)
            // - Orientation patch (HALF_PATCH=15)
            // - Descriptor pattern radius (~16)
            // Total border needed from image edge = max(3, 15, 16) + some margin = 19
            // But since the image already has EDGE_THRESHOLD=19 border,
            // keypoints detected at x must allow orientation to access [x-15, x+15]
            // So the detection border within the padded image should be at least 19
            // (matching EDGE_THRESHOLD) to ensure orientation kernel can access safely.
            const int BORDER = 19;

            int x = blockIdx.x * blockDim.x + threadIdx.x + BORDER;
            int y = blockIdx.y * blockDim.y + threadIdx.y + BORDER;

            if (x >= width - BORDER || y >= height - BORDER)
                return;

            if (isFastCorner9(image, x, y, step, threshold, c_fastCircle))
            {
                int score = computeFastScore(image, x, y, step, threshold, c_fastCircle);

                // Non-maximum suppression in 3x3 window
                bool isMax = true;
                for (int dy = -1; dy <= 1 && isMax; ++dy)
                {
                    for (int dx = -1; dx <= 1 && isMax; ++dx)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int nx = x + dx, ny = y + dy;
                        if (isFastCorner9(image, nx, ny, step, threshold, c_fastCircle))
                        {
                            int neighborScore = computeFastScore(image, nx, ny, step, threshold, c_fastCircle);
                            if (neighborScore > score)
                                isMax = false;
                        }
                    }
                }

                if (isMax)
                {
                    int idx = atomicAdd(responseCount, 1);
                    if (idx < maxKeypoints)
                    {
                        responses[idx].x = static_cast<int16_t>(x);
                        responses[idx].y = static_cast<int16_t>(y);
                        responses[idx].response = static_cast<int16_t>(score);
                        responses[idx].level = static_cast<int16_t>(level);
                    }
                }
            }
        }

        // ============================================================================
        // AoS to SoA Conversion Kernel
        // ============================================================================

        __global__ void convertAoSToSoAKernel(
            const GpuFastResponse *__restrict__ responses,
            float *__restrict__ outX,
            float *__restrict__ outY,
            float *__restrict__ outSize,
            float *__restrict__ outAngle,
            float *__restrict__ outResponse,
            int *__restrict__ outOctave,
            int numKeypoints,
            int level,
            float scaleFactor)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            // Account for EDGE_THRESHOLD offset in coordinates
            outX[idx] = static_cast<float>(responses[idx].x);
            outY[idx] = static_cast<float>(responses[idx].y);
            outSize[idx] = 31.0f * scaleFactor; // PATCH_SIZE = 31 (matches CPU ORBextractor)
            outAngle[idx] = 0.0f;               // Will be computed by orientation kernel
            outResponse[idx] = static_cast<float>(responses[idx].response);
            outOctave[idx] = level;
        }

        // ============================================================================
        // IC_Angle Orientation Kernel
        // ============================================================================

        __global__ void orientationKernel(
            const unsigned char *__restrict__ image,
            int step,
            float *__restrict__ kpX,
            float *__restrict__ kpY,
            float *__restrict__ kpAngle,
            int numKeypoints,
            int imgWidth,
            int imgHeight)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            const int HALF_PATCH = 15;
            int cx = __float2int_rn(kpX[idx]);
            int cy = __float2int_rn(kpY[idx]);

            // Bounds check - skip keypoints too close to edge
            if (cx < HALF_PATCH || cx >= imgWidth - HALF_PATCH ||
                cy < HALF_PATCH || cy >= imgHeight - HALF_PATCH)
            {
                kpAngle[idx] = 0.0f;
                return;
            }

            int m_01 = 0, m_10 = 0;

            // Center line (v=0)
            const unsigned char *center = image + cy * step + cx;

#pragma unroll
            for (int u = -HALF_PATCH; u <= HALF_PATCH; ++u)
            {
                m_10 += u * center[u];
            }

            // Other lines
            for (int v = 1; v <= HALF_PATCH; ++v)
            {
                int v_sum = 0;
                int d = c_umax[v];

                for (int u = -d; u <= d; ++u)
                {
                    int val_plus = center[u + v * step];
                    int val_minus = center[u - v * step];
                    v_sum += (val_plus - val_minus);
                    m_10 += u * (val_plus + val_minus);
                }
                m_01 += v * v_sum;
            }

            // Match OpenCV's fastAtan2 which returns [0, 360) range
            float angle = atan2f(static_cast<float>(m_01), static_cast<float>(m_10)) * (180.0f / 3.14159265358979323846f);
            if (angle < 0.0f)
                angle += 360.0f;
            kpAngle[idx] = angle;
        }

        // ============================================================================
        // ORB Descriptor Kernel (Warp-Parallel)
        // ============================================================================

        __global__ void descriptorKernel(
            const unsigned char *__restrict__ blurredImage,
            int step,
            const float *__restrict__ kpX,
            const float *__restrict__ kpY,
            const float *__restrict__ kpAngle,
            GpuDescriptor *__restrict__ descriptors,
            int numKeypoints)
        {
            // One warp per descriptor
            int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
            int laneId = threadIdx.x % 32;

            if (warpId >= numKeypoints)
                return;

            float x = kpX[warpId];
            float y = kpY[warpId];
            float angle = kpAngle[warpId] * (3.14159265358979323846f / 180.0f);
            float cosA = cosf(angle);
            float sinA = sinf(angle);

            const unsigned char *center = blurredImage + __float2int_rn(y) * step + __float2int_rn(x);

            // Each lane computes 8 descriptor bytes (256 bits / 32 lanes = 8 bits per lane)
            uint32_t desc = 0;

            // Each lane handles pattern pairs [laneId*16, laneId*16+15]
            int patternStart = laneId * 16;

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int pairIdx = patternStart + i * 2;

                int2 p0 = c_pattern[pairIdx];
                int2 p1 = c_pattern[pairIdx + 1];

                // Rotate pattern
                int x0 = __float2int_rn(p0.x * cosA - p0.y * sinA);
                int y0 = __float2int_rn(p0.x * sinA + p0.y * cosA);
                int x1 = __float2int_rn(p1.x * cosA - p1.y * sinA);
                int y1 = __float2int_rn(p1.x * sinA + p1.y * cosA);

                int v0 = center[y0 * step + x0];
                int v1 = center[y1 * step + x1];

                if (v0 < v1)
                {
                    desc |= (1u << i);
                }
            }

            // Combine across warp using shuffle
            // Gather all 32 bytes into descriptor
            uint32_t fullDesc[8];

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                uint32_t byte = (desc >> i) & 0x01;
                uint32_t combined = 0;

                for (int lane = 0; lane < 32; ++lane)
                {
                    uint32_t bit = __shfl_sync(0xFFFFFFFF, byte, lane);
                    combined |= (bit << lane);
                }

                if (laneId == 0)
                {
                    fullDesc[i] = combined;
                }
            }

            // Lane 0 writes the descriptor
            if (laneId == 0)
            {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    descriptors[warpId].data[i] = fullDesc[i];
                }
            }
        }

        // ============================================================================
        // Grid Building Kernel
        // ============================================================================

        __global__ void buildGridKernel(
            const float *__restrict__ kpX,
            const float *__restrict__ kpY,
            int numKeypoints,
            int *__restrict__ cellCount,
            int *__restrict__ cellStart,
            int *__restrict__ featureIndices,
            float invCellWidth,
            float invCellHeight,
            int gridCols,
            int gridRows)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            int cellX = min(static_cast<int>(kpX[idx] * invCellWidth), gridCols - 1);
            int cellY = min(static_cast<int>(kpY[idx] * invCellHeight), gridRows - 1);
            int cellIdx = cellY * gridCols + cellX;

            // Atomic increment to count features per cell
            int posInCell = atomicAdd(&cellCount[cellIdx], 1);

            // Store index (will be sorted later)
            int globalPos = cellStart[cellIdx] + posInCell;
            featureIndices[globalPos] = idx;
        }

        // ============================================================================
        // Keypoint Scaling Kernel
        // ============================================================================

        __global__ void scaleCoordinatesKernel(
            float *__restrict__ x,
            float *__restrict__ y,
            float *__restrict__ size,
            float scale,
            int patchSize,
            int numKeypoints)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            x[idx] *= scale;
            y[idx] *= scale;
            size[idx] = static_cast<float>(patchSize) * scale;
        }

        // ============================================================================
        // ORBExtractorCuda Implementation
        // ============================================================================

        ORBExtractorCuda::ORBExtractorCuda(int nfeatures, float scaleFactor, int nlevels,
                                           int iniThFAST, int minThFAST)
            : nfeatures_(nfeatures), scaleFactor_(scaleFactor), nlevels_(nlevels),
              iniThFAST_(iniThFAST), minThFAST_(minThFAST),
              d_pattern_(nullptr), d_umax_(nullptr)
        {
            // Initialize scale factors (matching original ORB-SLAM3)
            mvScaleFactor_.resize(nlevels_);
            mvLevelSigma2_.resize(nlevels_);
            mvScaleFactor_[0] = 1.0f;
            mvLevelSigma2_[0] = 1.0f;

            for (int i = 1; i < nlevels_; ++i)
            {
                mvScaleFactor_[i] = mvScaleFactor_[i - 1] * scaleFactor_;
                mvLevelSigma2_[i] = mvScaleFactor_[i] * mvScaleFactor_[i];
            }

            mvInvScaleFactor_.resize(nlevels_);
            mvInvLevelSigma2_.resize(nlevels_);
            for (int i = 0; i < nlevels_; ++i)
            {
                mvInvScaleFactor_[i] = 1.0f / mvScaleFactor_[i];
                mvInvLevelSigma2_[i] = 1.0f / mvLevelSigma2_[i];
            }

            // Compute features per level
            mnFeaturesPerLevel_.resize(nlevels_);
            float factor = 1.0f / scaleFactor_;
            float nDesiredFeaturesPerScale = nfeatures_ * (1 - factor) /
                                             (1 - powf(factor, static_cast<float>(nlevels_)));

            int sumFeatures = 0;
            for (int level = 0; level < nlevels_ - 1; ++level)
            {
                mnFeaturesPerLevel_[level] = static_cast<int>(nDesiredFeaturesPerScale + 0.5f);
                sumFeatures += mnFeaturesPerLevel_[level];
                nDesiredFeaturesPerScale *= factor;
            }
            mnFeaturesPerLevel_[nlevels_ - 1] = std::max(nfeatures_ - sumFeatures, 0);

            // Initialize GPU pyramid
            gpuPyramid_.numLevels = nlevels_;
            gpuPyramid_.scaleFactor = scaleFactor_;

            // Initialize keypoint buffers per level - allocate GPU memory for each
            gpuKeypointsPerLevel_.resize(nlevels_);
            int maxKeypointsPerLevel = MAX_KEYPOINTS_PER_FRAME / nlevels_ + 1000;
            for (int level = 0; level < nlevels_; ++level)
            {
                GpuKeyPointSoA &kp = gpuKeypointsPerLevel_[level];
                kp.capacity = maxKeypointsPerLevel;
                kp.count = 0;
                CUDA_CHECK(cudaMalloc(&kp.x, maxKeypointsPerLevel * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&kp.y, maxKeypointsPerLevel * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&kp.size, maxKeypointsPerLevel * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&kp.angle, maxKeypointsPerLevel * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&kp.response, maxKeypointsPerLevel * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&kp.octave, maxKeypointsPerLevel * sizeof(int)));
            }

            // Allocate combined keypoint buffer (gpuKeypoints_)
            gpuKeypoints_.capacity = MAX_KEYPOINTS_PER_FRAME;
            gpuKeypoints_.count = 0;
            CUDA_CHECK(cudaMalloc(&gpuKeypoints_.x, MAX_KEYPOINTS_PER_FRAME * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuKeypoints_.y, MAX_KEYPOINTS_PER_FRAME * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuKeypoints_.size, MAX_KEYPOINTS_PER_FRAME * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuKeypoints_.angle, MAX_KEYPOINTS_PER_FRAME * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuKeypoints_.response, MAX_KEYPOINTS_PER_FRAME * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuKeypoints_.octave, MAX_KEYPOINTS_PER_FRAME * sizeof(int)));

            // Allocate temporary buffers
            fastResponseBuffer_.resize(MAX_KEYPOINTS_PER_FRAME);
            keypointCountBuffer_.resize(nlevels_);

            // Create CUDA streams
            CUDA_CHECK(cudaStreamCreateWithFlags(&pyramidStream_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&fastStream_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&descriptorStream_, cudaStreamNonBlocking));

            // Create events for timing
            CUDA_CHECK(cudaEventCreate(&startEvent_));
            CUDA_CHECK(cudaEventCreate(&stopEvent_));

            // Create Gaussian filter
            gaussianFilter_ = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1,
                                                             cv::Size(7, 7), 2.0, 2.0,
                                                             cv::BORDER_REFLECT_101);

            // Upload constant memory
            uploadConstantMemory();
            initializePatternGpu();
            initializeUmaxGpu();
        }

        ORBExtractorCuda::~ORBExtractorCuda()
        {
            // Free keypoint buffers per level
            for (auto &kp : gpuKeypointsPerLevel_)
            {
                if (kp.x)
                    cudaFree(kp.x);
                if (kp.y)
                    cudaFree(kp.y);
                if (kp.size)
                    cudaFree(kp.size);
                if (kp.angle)
                    cudaFree(kp.angle);
                if (kp.response)
                    cudaFree(kp.response);
                if (kp.octave)
                    cudaFree(kp.octave);
            }

            // Free combined keypoint buffer (gpuKeypoints_)
            if (gpuKeypoints_.x)
                cudaFree(gpuKeypoints_.x);
            if (gpuKeypoints_.y)
                cudaFree(gpuKeypoints_.y);
            if (gpuKeypoints_.size)
                cudaFree(gpuKeypoints_.size);
            if (gpuKeypoints_.angle)
                cudaFree(gpuKeypoints_.angle);
            if (gpuKeypoints_.response)
                cudaFree(gpuKeypoints_.response);
            if (gpuKeypoints_.octave)
                cudaFree(gpuKeypoints_.octave);

            // Free descriptor buffer
            if (gpuDescriptors_.descriptors)
                cudaFree(gpuDescriptors_.descriptors);

            if (d_pattern_)
                cudaFree(d_pattern_);
            if (d_umax_)
                cudaFree(d_umax_);

            cudaStreamDestroy(pyramidStream_);
            cudaStreamDestroy(fastStream_);
            cudaStreamDestroy(descriptorStream_);

            cudaEventDestroy(startEvent_);
            cudaEventDestroy(stopEvent_);
        }

        void ORBExtractorCuda::uploadConstantMemory()
        {
            // Upload FAST circle pattern
            int2 fastCircle[16];
            for (int i = 0; i < 16; ++i)
            {
                fastCircle[i].x = FAST_CIRCLE[i][0];
                fastCircle[i].y = FAST_CIRCLE[i][1];
            }
            CUDA_CHECK(cudaMemcpyToSymbol(c_fastCircle, fastCircle, 16 * sizeof(int2)));
        }

        void ORBExtractorCuda::initializePatternGpu()
        {
            // Convert pattern to int2 pairs
            int2 pattern[512];
            for (int i = 0; i < 256; ++i)
            {
                pattern[i * 2].x = BIT_PATTERN_31[i * 4];
                pattern[i * 2].y = BIT_PATTERN_31[i * 4 + 1];
                pattern[i * 2 + 1].x = BIT_PATTERN_31[i * 4 + 2];
                pattern[i * 2 + 1].y = BIT_PATTERN_31[i * 4 + 3];
            }

            CUDA_CHECK(cudaMemcpyToSymbol(c_pattern, pattern, 512 * sizeof(int2)));
        }

        void ORBExtractorCuda::initializeUmaxGpu()
        {
            // Compute umax for IC_Angle (matching original)
            // umax[v] gives the maximum u coordinate for row v in the circular patch
            int umax[16] = {0}; // Initialize all to 0 first
            int v, v0, vmax = static_cast<int>(HALF_PATCH_SIZE * sqrtf(2.0f) / 2 + 1);
            int vmin = static_cast<int>(ceilf(HALF_PATCH_SIZE * sqrtf(2.0f) / 2));
            const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;

            // Fill the first part of the circular patch (v = 0 to vmax)
            for (v = 0; v <= vmax; ++v)
            {
                umax[v] = static_cast<int>(sqrtf(hp2 - v * v) + 0.5f);
            }

            // Make sure we are symmetric - fill v = vmin to HALF_PATCH_SIZE
            for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
            {
                while (umax[v0] == umax[v0 + 1])
                    ++v0;
                umax[v] = v0;
                ++v0;
            }

            CUDA_CHECK(cudaMemcpyToSymbol(c_umax, umax, 16 * sizeof(int)));
        }

        void ORBExtractorCuda::computePyramidGpu(const cv::cuda::GpuMat &image)
        {
            CUDA_CHECK(cudaEventRecord(startEvent_, pyramidStream_));

            // Temporary buffer for border operation
            cv::cuda::GpuMat tempBuffer;

            for (int level = 0; level < nlevels_; ++level)
            {
                float scale = mvInvScaleFactor_[level];
                cv::Size sz(static_cast<int>(image.cols * scale),
                            static_cast<int>(image.rows * scale));

                GpuPyramidLevel &lvl = gpuPyramid_.levels[level];
                lvl.scale = mvScaleFactor_[level];
                lvl.invScale = mvInvScaleFactor_[level];
                lvl.width = sz.width;
                lvl.height = sz.height;

                int fullWidth = sz.width + 2 * EDGE_THRESHOLD;
                int fullHeight = sz.height + 2 * EDGE_THRESHOLD;

                // Ensure image buffer is allocated
                if (lvl.image.empty() || lvl.image.cols != fullWidth ||
                    lvl.image.rows != fullHeight)
                {
                    lvl.image.create(fullHeight, fullWidth, CV_8UC1);
                    lvl.blurred.create(fullHeight, fullWidth, CV_8UC1);
                }

                // Resize or copy to temporary buffer first
                if (level == 0)
                {
                    // Use copyMakeBorder directly from source image to destination
                    cv::cuda::copyMakeBorder(image, lvl.image,
                                             EDGE_THRESHOLD, EDGE_THRESHOLD,
                                             EDGE_THRESHOLD, EDGE_THRESHOLD,
                                             cv::BORDER_REFLECT_101, cv::Scalar(),
                                             cv::cuda::Stream::Null());
                }
                else
                {
                    // Resize from previous level's inner region
                    cv::cuda::GpuMat prevRoi = gpuPyramid_.levels[level - 1].image(
                        cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD,
                                 gpuPyramid_.levels[level - 1].width,
                                 gpuPyramid_.levels[level - 1].height));

                    // Resize to temporary buffer
                    cv::cuda::resize(prevRoi, tempBuffer, sz, 0, 0, cv::INTER_LINEAR,
                                     cv::cuda::Stream::Null());

                    // Add border from temp buffer to final image
                    cv::cuda::copyMakeBorder(tempBuffer, lvl.image,
                                             EDGE_THRESHOLD, EDGE_THRESHOLD,
                                             EDGE_THRESHOLD, EDGE_THRESHOLD,
                                             cv::BORDER_REFLECT_101, cv::Scalar(),
                                             cv::cuda::Stream::Null());
                }
            }

            CUDA_CHECK(cudaEventRecord(stopEvent_, pyramidStream_));
            CUDA_CHECK(cudaStreamSynchronize(pyramidStream_));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.pyramidTimeMs, startEvent_, stopEvent_));
        }

        void ORBExtractorCuda::detectFastCornersGpu()
        {
            CUDA_CHECK(cudaEventRecord(startEvent_, fastStream_));

            int *d_responseCount;
            CUDA_CHECK(cudaMalloc(&d_responseCount, nlevels_ * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_responseCount, 0, nlevels_ * sizeof(int)));

            for (int level = 0; level < nlevels_; ++level)
            {
                const GpuPyramidLevel &lvl = gpuPyramid_.levels[level];

                // Full image dimensions (with border)
                int fullWidth = lvl.width + 2 * EDGE_THRESHOLD;
                int fullHeight = lvl.height + 2 * EDGE_THRESHOLD;

                dim3 block(16, 16);
                dim3 grid((fullWidth + block.x - 1) / block.x,
                          (fullHeight + block.y - 1) / block.y);

                int offset = level * (MAX_KEYPOINTS_PER_FRAME / nlevels_);

                // First pass with high threshold
                fastDetectionKernel<<<grid, block, 0, fastStream_>>>(
                    lvl.image.ptr<unsigned char>(),
                    fullWidth,
                    fullHeight,
                    static_cast<int>(lvl.image.step),
                    fastResponseBuffer_.data() + offset,
                    d_responseCount + level,
                    iniThFAST_,
                    MAX_KEYPOINTS_PER_FRAME / nlevels_,
                    level);

                CUDA_CHECK_LAST();
            }

            CUDA_CHECK(cudaStreamSynchronize(fastStream_));

            // Copy counts back
            std::vector<int> counts(nlevels_);
            CUDA_CHECK(cudaMemcpy(counts.data(), d_responseCount,
                                  nlevels_ * sizeof(int), cudaMemcpyDeviceToHost));

            // Store keypoint counts
            for (int level = 0; level < nlevels_; ++level)
            {
                gpuKeypointsPerLevel_[level].count = counts[level];
            }

            // Convert from AoS (GpuFastResponse) to SoA (GpuKeyPointSoA) for each level
            for (int level = 0; level < nlevels_; ++level)
            {
                int count = counts[level];
                if (count == 0)
                    continue;

                // Clamp to capacity
                count = std::min(count, gpuKeypointsPerLevel_[level].capacity);
                gpuKeypointsPerLevel_[level].count = count;

                int offset = level * (MAX_KEYPOINTS_PER_FRAME / nlevels_);
                float scaleFactor = mvScaleFactor_[level];

                dim3 block(256);
                dim3 grid((count + block.x - 1) / block.x);

                convertAoSToSoAKernel<<<grid, block, 0, fastStream_>>>(
                    fastResponseBuffer_.data() + offset,
                    gpuKeypointsPerLevel_[level].x,
                    gpuKeypointsPerLevel_[level].y,
                    gpuKeypointsPerLevel_[level].size,
                    gpuKeypointsPerLevel_[level].angle,
                    gpuKeypointsPerLevel_[level].response,
                    gpuKeypointsPerLevel_[level].octave,
                    count,
                    level,
                    scaleFactor);

                CUDA_CHECK_LAST();
            }

            CUDA_CHECK(cudaStreamSynchronize(fastStream_));

            cudaFree(d_responseCount);

            CUDA_CHECK(cudaEventRecord(stopEvent_, fastStream_));
            CUDA_CHECK(cudaEventSynchronize(stopEvent_));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.fastTimeMs, startEvent_, stopEvent_));
        }

        // Kernel to rearrange keypoints based on sorted indices
        __global__ void rearrangeKeypointsKernel(
            const float *__restrict__ srcX,
            const float *__restrict__ srcY,
            const float *__restrict__ srcSize,
            const float *__restrict__ srcAngle,
            const float *__restrict__ srcResponse,
            const int *__restrict__ srcOctave,
            const int *__restrict__ indices,
            float *__restrict__ dstX,
            float *__restrict__ dstY,
            float *__restrict__ dstSize,
            float *__restrict__ dstAngle,
            float *__restrict__ dstResponse,
            int *__restrict__ dstOctave,
            int count)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count)
                return;

            int srcIdx = indices[idx];
            dstX[idx] = srcX[srcIdx];
            dstY[idx] = srcY[srcIdx];
            dstSize[idx] = srcSize[srcIdx];
            dstAngle[idx] = srcAngle[srcIdx];
            dstResponse[idx] = srcResponse[srcIdx];
            dstOctave[idx] = srcOctave[srcIdx];
        }

        // Kernel to assign keypoints to grid cells
        __global__ void assignGridCellsKernel(
            const float *__restrict__ x,
            const float *__restrict__ y,
            int *__restrict__ cellIndices,
            int count,
            int gridCols,
            int gridRows,
            float cellWidth,
            float cellHeight)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count)
                return;

            int cellX = min(static_cast<int>(x[idx] / cellWidth), gridCols - 1);
            int cellY = min(static_cast<int>(y[idx] / cellHeight), gridRows - 1);
            cellIndices[idx] = cellY * gridCols + cellX;
        }

        void ORBExtractorCuda::selectTopKeypointsGpu()
        {
            // Grid-based spatial distribution similar to CPU's octree approach
            // This ensures keypoints are spread evenly across the image

            for (int level = 0; level < nlevels_; ++level)
            {
                GpuKeyPointSoA &kp = gpuKeypointsPerLevel_[level];
                int maxFeatures = mnFeaturesPerLevel_[level];

                if (kp.count == 0 || kp.count <= maxFeatures)
                    continue; // No need to select, keep all

                const GpuPyramidLevel &lvl = gpuPyramid_.levels[level];
                int imgWidth = lvl.width;
                int imgHeight = lvl.height;

                // Use a smaller grid size (8x8 = 64 cells) to avoid too many empty cells
                // This better matches the octree behavior which adapts to feature density
                int gridCols = 8;
                int gridRows = 8;
                int numCells = gridCols * gridRows;
                float cellWidth = static_cast<float>(imgWidth) / gridCols;
                float cellHeight = static_cast<float>(imgHeight) / gridRows;

                // Download to CPU for grid-based selection
                std::vector<float> h_x(kp.count), h_y(kp.count), h_response(kp.count);
                std::vector<float> h_size(kp.count), h_angle(kp.count);
                std::vector<int> h_octave(kp.count);

                CUDA_CHECK(cudaMemcpy(h_x.data(), kp.x, kp.count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_y.data(), kp.y, kp.count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_response.data(), kp.response, kp.count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_size.data(), kp.size, kp.count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_angle.data(), kp.angle, kp.count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_octave.data(), kp.octave, kp.count * sizeof(int), cudaMemcpyDeviceToHost));

                // Assign keypoints to grid cells
                std::vector<std::vector<int>> cellKeypoints(numCells);
                for (int i = 0; i < kp.count; ++i)
                {
                    int cellX = std::min(static_cast<int>(h_x[i] / cellWidth), gridCols - 1);
                    int cellY = std::min(static_cast<int>(h_y[i] / cellHeight), gridRows - 1);
                    cellX = std::max(0, cellX);
                    cellY = std::max(0, cellY);
                    int cellIdx = cellY * gridCols + cellX;
                    cellKeypoints[cellIdx].push_back(i);
                }

                // Sort keypoints within each cell by response (descending)
                for (auto &cell : cellKeypoints)
                {
                    std::sort(cell.begin(), cell.end(), [&h_response](int a, int b)
                              { return h_response[a] > h_response[b]; });
                }

                // Distribute features across cells using round-robin selection
                // This spreads keypoints spatially while respecting the feature budget
                std::vector<int> selectedIndices;
                selectedIndices.reserve(maxFeatures);
                std::vector<int> cellPointers(numCells, 0); // Track position in each cell

                // Keep selecting until we have enough features or run out
                bool hasMore = true;
                while (selectedIndices.size() < static_cast<size_t>(maxFeatures) && hasMore)
                {
                    hasMore = false;
                    for (int cell = 0; cell < numCells && selectedIndices.size() < static_cast<size_t>(maxFeatures); ++cell)
                    {
                        if (cellPointers[cell] < static_cast<int>(cellKeypoints[cell].size()))
                        {
                            selectedIndices.push_back(cellKeypoints[cell][cellPointers[cell]]);
                            cellPointers[cell]++;
                            hasMore = true;
                        }
                    }
                }

                // If still need more keypoints (cells exhausted), fill with remaining best responses
                if (selectedIndices.size() < static_cast<size_t>(maxFeatures))
                {
                    // Get all remaining keypoints sorted by response
                    std::vector<std::pair<float, int>> remainingKps;
                    std::set<int> selectedSet(selectedIndices.begin(), selectedIndices.end());
                    
                    for (int i = 0; i < kp.count; ++i)
                    {
                        if (selectedSet.find(i) == selectedSet.end())
                        {
                            remainingKps.emplace_back(h_response[i], i);
                        }
                    }
                    
                    std::sort(remainingKps.begin(), remainingKps.end(), 
                              [](const auto& a, const auto& b) { return a.first > b.first; });
                    
                    for (const auto& [resp, idx] : remainingKps)
                    {
                        if (selectedIndices.size() >= static_cast<size_t>(maxFeatures))
                            break;
                        selectedIndices.push_back(idx);
                    }
                }

                // Prepare output arrays
                int selectedCount = static_cast<int>(selectedIndices.size());
                std::vector<float> out_x(selectedCount), out_y(selectedCount), out_size(selectedCount);
                std::vector<float> out_angle(selectedCount), out_response(selectedCount);
                std::vector<int> out_octave(selectedCount);

                for (int i = 0; i < selectedCount; ++i)
                {
                    int idx = selectedIndices[i];
                    out_x[i] = h_x[idx];
                    out_y[i] = h_y[idx];
                    out_size[i] = h_size[idx];
                    out_angle[i] = h_angle[idx];
                    out_response[i] = h_response[idx];
                    out_octave[i] = h_octave[idx];
                }

                // Upload back to GPU
                CUDA_CHECK(cudaMemcpy(kp.x, out_x.data(), selectedCount * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(kp.y, out_y.data(), selectedCount * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(kp.size, out_size.data(), selectedCount * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(kp.angle, out_angle.data(), selectedCount * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(kp.response, out_response.data(), selectedCount * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(kp.octave, out_octave.data(), selectedCount * sizeof(int), cudaMemcpyHostToDevice));

                // Update count
                kp.count = selectedCount;
            }
        }

        void ORBExtractorCuda::computeOrientationsGpu()
        {
            CUDA_CHECK(cudaEventRecord(startEvent_, descriptorStream_));

            for (int level = 0; level < nlevels_; ++level)
            {
                GpuKeyPointSoA &kp = gpuKeypointsPerLevel_[level];
                if (kp.count == 0)
                    continue;

                const GpuPyramidLevel &lvl = gpuPyramid_.levels[level];

                if (kp.x == nullptr || kp.y == nullptr || kp.angle == nullptr || lvl.image.empty())
                    continue;

                dim3 block(256);
                dim3 grid((kp.count + block.x - 1) / block.x);

                orientationKernel<<<grid, block, 0, descriptorStream_>>>(
                    lvl.image.ptr<unsigned char>(),
                    static_cast<int>(lvl.image.step),
                    kp.x, kp.y, kp.angle,
                    kp.count,
                    lvl.image.cols, lvl.image.rows);

                CUDA_CHECK_LAST();
            }

            CUDA_CHECK(cudaStreamSynchronize(descriptorStream_));

            CUDA_CHECK(cudaEventRecord(stopEvent_, descriptorStream_));
            CUDA_CHECK(cudaEventSynchronize(stopEvent_));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.orientTimeMs, startEvent_, stopEvent_));
        }

        void ORBExtractorCuda::applyGaussianBlurGpu()
        {
            for (int level = 0; level < nlevels_; ++level)
            {
                GpuPyramidLevel &lvl = gpuPyramid_.levels[level];
                gaussianFilter_->apply(lvl.image, lvl.blurred, cv::cuda::Stream::Null());
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        void ORBExtractorCuda::computeDescriptorsGpu()
        {
            CUDA_CHECK(cudaEventRecord(startEvent_, descriptorStream_));

            // Apply Gaussian blur first
            applyGaussianBlurGpu();

            int totalKeypoints = 0;
            for (int level = 0; level < nlevels_; ++level)
            {
                totalKeypoints += gpuKeypointsPerLevel_[level].count;
            }

            if (totalKeypoints == 0)
            {
                stats_.descriptorTimeMs = 0;
                return;
            }

            // Ensure descriptor buffer is large enough
            if (gpuDescriptors_.capacity < totalKeypoints)
            {
                if (gpuDescriptors_.descriptors)
                    cudaFree(gpuDescriptors_.descriptors);
                CUDA_CHECK(cudaMalloc(&gpuDescriptors_.descriptors,
                                      totalKeypoints * sizeof(GpuDescriptor)));
                gpuDescriptors_.capacity = totalKeypoints;
            }

            int offset = 0;
            for (int level = 0; level < nlevels_; ++level)
            {
                GpuKeyPointSoA &kp = gpuKeypointsPerLevel_[level];
                if (kp.count == 0)
                    continue;

                const GpuPyramidLevel &lvl = gpuPyramid_.levels[level];

                // One warp per keypoint
                dim3 block(256);
                dim3 grid((kp.count * 32 + block.x - 1) / block.x);

                descriptorKernel<<<grid, block, 0, descriptorStream_>>>(
                    lvl.blurred.ptr<unsigned char>(),
                    static_cast<int>(lvl.blurred.step),
                    kp.x, kp.y, kp.angle,
                    gpuDescriptors_.descriptors + offset,
                    kp.count);

                CUDA_CHECK_LAST();

                offset += kp.count;
            }

            gpuDescriptors_.count = totalKeypoints;

            // Merge per-level keypoints into gpuKeypoints_
            // Use cudaMemcpyAsync to copy each level's keypoints to combined buffer
            int kpOffset = 0;
            for (int level = 0; level < nlevels_; ++level)
            {
                GpuKeyPointSoA &kp = gpuKeypointsPerLevel_[level];
                if (kp.count == 0)
                    continue;

                // Check for buffer overflow
                if (kpOffset + kp.count > gpuKeypoints_.capacity)
                    break;

                float scale = mvScaleFactor_[level];

                // Copy keypoints to merged buffer (need to scale coordinates back to original image)
                CUDA_CHECK(cudaMemcpyAsync(gpuKeypoints_.x + kpOffset, kp.x, kp.count * sizeof(float),
                                           cudaMemcpyDeviceToDevice, descriptorStream_));
                CUDA_CHECK(cudaMemcpyAsync(gpuKeypoints_.y + kpOffset, kp.y, kp.count * sizeof(float),
                                           cudaMemcpyDeviceToDevice, descriptorStream_));
                CUDA_CHECK(cudaMemcpyAsync(gpuKeypoints_.size + kpOffset, kp.size, kp.count * sizeof(float),
                                           cudaMemcpyDeviceToDevice, descriptorStream_));
                CUDA_CHECK(cudaMemcpyAsync(gpuKeypoints_.angle + kpOffset, kp.angle, kp.count * sizeof(float),
                                           cudaMemcpyDeviceToDevice, descriptorStream_));
                CUDA_CHECK(cudaMemcpyAsync(gpuKeypoints_.response + kpOffset, kp.response, kp.count * sizeof(float),
                                           cudaMemcpyDeviceToDevice, descriptorStream_));
                CUDA_CHECK(cudaMemcpyAsync(gpuKeypoints_.octave + kpOffset, kp.octave, kp.count * sizeof(int),
                                           cudaMemcpyDeviceToDevice, descriptorStream_));

                // Scale coordinates from pyramid level back to original image (like CPU does)
                // Note: Level 0 has scale=1.0, so we still run the kernel to set correct size
                {
                    dim3 block(256);
                    dim3 grid((kp.count + block.x - 1) / block.x);
                    scaleCoordinatesKernel<<<grid, block, 0, descriptorStream_>>>(
                        gpuKeypoints_.x + kpOffset,
                        gpuKeypoints_.y + kpOffset,
                        gpuKeypoints_.size + kpOffset,
                        scale,
                        PATCH_SIZE,
                        kp.count);
                    CUDA_CHECK_LAST();
                }

                kpOffset += kp.count;
            }

            // Update count to actual keypoints copied (may be less than totalKeypoints if capped)
            gpuKeypoints_.count = kpOffset;

            // Also update descriptors count to match (important for download)
            gpuDescriptors_.count = kpOffset;

            CUDA_CHECK(cudaStreamSynchronize(descriptorStream_));

            CUDA_CHECK(cudaEventRecord(stopEvent_, descriptorStream_));
            CUDA_CHECK(cudaEventSynchronize(stopEvent_));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.descriptorTimeMs, startEvent_, stopEvent_));
        }

        int ORBExtractorCuda::operator()(const cv::cuda::GpuMat &gpuImage,
                                         GpuKeyPointSoA &gpuKeypoints,
                                         GpuDescriptorArray &gpuDescriptors)
        {
            if (gpuImage.empty())
                return 0;

            // 1. Compute pyramid
            computePyramidGpu(gpuImage);

            // 2. Detect FAST corners
            detectFastCornersGpu();

            // 3. Select top N keypoints per level (response-based selection)
            selectTopKeypointsGpu();

            // 4. Compute orientations
            computeOrientationsGpu();

            // 5. Compute descriptors
            computeDescriptorsGpu();

            // Copy results to output
            gpuKeypoints = gpuKeypoints_;
            gpuDescriptors = gpuDescriptors_;

            return gpuDescriptors_.count;
        }

        int ORBExtractorCuda::extractGpuResident(const cv::cuda::GpuMat &gpuImage,
                                                 GpuFrameData &frameData)
        {
            if (gpuImage.empty())
                return 0;

            cudaEvent_t totalStart, totalStop;
            CUDA_CHECK(cudaEventCreate(&totalStart));
            CUDA_CHECK(cudaEventCreate(&totalStop));
            CUDA_CHECK(cudaEventRecord(totalStart));

            // 1. Store image in frame
            frameData.image = gpuImage;

            // 2. Compute pyramid
            computePyramidGpu(gpuImage);
            frameData.pyramid = gpuPyramid_;

            // 3. Detect FAST corners
            detectFastCornersGpu();

            // 4. Select top N keypoints per level (response-based selection)
            selectTopKeypointsGpu();

            // 5. Compute orientations
            computeOrientationsGpu();

            // 6. Compute descriptors
            computeDescriptorsGpu();

            // 7. Copy to frame data
            frameData.keypoints = gpuKeypoints_;
            frameData.descriptors = gpuDescriptors_;
            frameData.numKeypoints = gpuDescriptors_.count;

            CUDA_CHECK(cudaEventRecord(totalStop));
            CUDA_CHECK(cudaEventSynchronize(totalStop));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.totalTimeMs, totalStart, totalStop));

            stats_.numKeypoints = frameData.numKeypoints;

            cudaEventDestroy(totalStart);
            cudaEventDestroy(totalStop);

            return frameData.numKeypoints;
        }

        // ============================================================================
        // Kernel Launch Wrappers
        // ============================================================================

        void launchFastDetectionKernel(
            const cv::cuda::GpuMat &image,
            GpuFastResponse *responses,
            int *responseCount,
            int threshold,
            int maxKeypoints,
            cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid((image.cols + block.x - 1) / block.x,
                      (image.rows + block.y - 1) / block.y);

            fastDetectionKernel<<<grid, block, 0, stream>>>(
                image.ptr<unsigned char>(),
                image.cols, image.rows,
                static_cast<int>(image.step),
                responses,
                responseCount,
                threshold,
                maxKeypoints,
                0);
        }

        void launchOrientationKernel(
            const cv::cuda::GpuMat &image,
            GpuKeyPointSoA &keypoints,
            const int *umax,
            cudaStream_t stream)
        {
            if (keypoints.count == 0)
                return;

            dim3 block(256);
            dim3 grid((keypoints.count + block.x - 1) / block.x);

            orientationKernel<<<grid, block, 0, stream>>>(
                image.ptr<unsigned char>(),
                static_cast<int>(image.step),
                keypoints.x,
                keypoints.y,
                keypoints.angle,
                keypoints.count,
                image.cols, image.rows);
        }

        void launchDescriptorKernel(
            const cv::cuda::GpuMat &blurredImage,
            const GpuKeyPointSoA &keypoints,
            GpuDescriptor *descriptors,
            const int2 *pattern,
            cudaStream_t stream)
        {
            if (keypoints.count == 0)
                return;

            dim3 block(256);
            dim3 grid((keypoints.count * 32 + block.x - 1) / block.x);

            descriptorKernel<<<grid, block, 0, stream>>>(
                blurredImage.ptr<unsigned char>(),
                static_cast<int>(blurredImage.step),
                keypoints.x,
                keypoints.y,
                keypoints.angle,
                descriptors,
                keypoints.count);
        }

        void launchScaleCoordinatesKernel(
            GpuKeyPointSoA &keypoints,
            float scale,
            cudaStream_t stream)
        {
            if (keypoints.count == 0)
                return;

            dim3 block(256);
            dim3 grid((keypoints.count + block.x - 1) / block.x);

            scaleCoordinatesKernel<<<grid, block, 0, stream>>>(
                keypoints.x,
                keypoints.y,
                keypoints.size,
                scale,
                PATCH_SIZE,
                keypoints.count);
        }

        void launchBuildGridKernel(
            const GpuKeyPointSoA &keypoints,
            GpuFeatureGrid &grid,
            cudaStream_t stream)
        {
            if (keypoints.count == 0)
                return;

            // First reset counts
            CUDA_CHECK(cudaMemsetAsync(grid.cellCount, 0, GRID_CELLS * sizeof(int), stream));

            // Compute prefix sum for cell starts
            // This would require Thrust in full implementation

            dim3 block(256);
            dim3 gridDim((keypoints.count + block.x - 1) / block.x);

            buildGridKernel<<<gridDim, block, 0, stream>>>(
                keypoints.x,
                keypoints.y,
                keypoints.count,
                grid.cellCount,
                grid.cellStart,
                grid.featureIndices,
                grid.invCellWidth,
                grid.invCellHeight,
                GRID_COLS,
                GRID_ROWS);
        }

    } // namespace cuda
} // namespace ORB_SLAM3
