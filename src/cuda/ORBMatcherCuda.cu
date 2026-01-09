/**
 * ORB Matcher CUDA Implementation
 *
 * GPU-accelerated feature matching using:
 * - Warp-parallel Hamming distance (popcount intrinsics)
 * - Thrust sorting for KNN
 * - Parallel ratio test
 * - Orientation consistency check
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 */

#include "cuda/ORBMatcherCuda.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scan.h>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Device Functions
        // ============================================================================

        /**
         * Compute Hamming distance between two 256-bit descriptors
         * Uses hardware popcount instruction
         */
        __device__ __forceinline__ int hammingDistance(
            const GpuDescriptor &a,
            const GpuDescriptor &b)
        {
            int distance = 0;

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                distance += __popc(a.data[i] ^ b.data[i]);
            }

            return distance;
        }

        /**
         * Warp-cooperative Hamming distance
         * Each thread handles one uint32, result reduced across warp
         */
        __device__ __forceinline__ int warpHammingDistance(
            const GpuDescriptor &a,
            const GpuDescriptor &b,
            int laneId)
        {
            int partialDist = 0;

            if (laneId < 8)
            {
                partialDist = __popc(a.data[laneId] ^ b.data[laneId]);
            }

// Warp reduction
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                partialDist += __shfl_down_sync(0xFFFFFFFF, partialDist, offset);
            }

            return partialDist;
        }

        // ============================================================================
        // Hamming Distance Matrix Kernel
        // ============================================================================

        __global__ void hammingDistanceMatrixKernel(
            const GpuDescriptor *__restrict__ queryDesc,
            const GpuDescriptor *__restrict__ trainDesc,
            int numQuery,
            int numTrain,
            int *__restrict__ distances)
        {
            int queryIdx = blockIdx.y * blockDim.y + threadIdx.y;
            int trainIdx = blockIdx.x * blockDim.x + threadIdx.x;

            if (queryIdx >= numQuery || trainIdx >= numTrain)
                return;

            int dist = hammingDistance(queryDesc[queryIdx], trainDesc[trainIdx]);
            distances[queryIdx * numTrain + trainIdx] = dist;
        }

        // Optimized version using shared memory
        __global__ void hammingDistanceMatrixSharedKernel(
            const GpuDescriptor *__restrict__ queryDesc,
            const GpuDescriptor *__restrict__ trainDesc,
            int numQuery,
            int numTrain,
            int *__restrict__ distances)
        {
            __shared__ GpuDescriptor sharedQuery[16];
            __shared__ GpuDescriptor sharedTrain[16];

            int queryBlockStart = blockIdx.y * 16;
            int trainBlockStart = blockIdx.x * 16;

            int localQueryIdx = threadIdx.y;
            int localTrainIdx = threadIdx.x;

            // Load descriptors to shared memory
            if (threadIdx.x == 0 && queryBlockStart + localQueryIdx < numQuery)
            {
                sharedQuery[localQueryIdx] = queryDesc[queryBlockStart + localQueryIdx];
            }
            if (threadIdx.y == 0 && trainBlockStart + localTrainIdx < numTrain)
            {
                sharedTrain[localTrainIdx] = trainDesc[trainBlockStart + localTrainIdx];
            }

            __syncthreads();

            int queryIdx = queryBlockStart + localQueryIdx;
            int trainIdx = trainBlockStart + localTrainIdx;

            if (queryIdx < numQuery && trainIdx < numTrain)
            {
                int dist = hammingDistance(sharedQuery[localQueryIdx], sharedTrain[localTrainIdx]);
                distances[queryIdx * numTrain + trainIdx] = dist;
            }
        }

        // ============================================================================
        // KNN Search Kernel
        // ============================================================================

        __global__ void knnSearchKernel(
            const int *__restrict__ distances,
            int numQuery,
            int numTrain,
            int k,
            GpuMatch *__restrict__ matches)
        {
            int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (queryIdx >= numQuery)
                return;

            // Find k smallest distances for this query
            // Simple selection for k=2 (most common case)
            if (k == 2)
            {
                int bestDist1 = 256, bestDist2 = 256;
                int bestIdx1 = -1, bestIdx2 = -1;

                for (int i = 0; i < numTrain; ++i)
                {
                    int dist = distances[queryIdx * numTrain + i];

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestIdx2 = bestIdx1;
                        bestDist1 = dist;
                        bestIdx1 = i;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                        bestIdx2 = i;
                    }
                }

                matches[queryIdx * 2].queryIdx = queryIdx;
                matches[queryIdx * 2].trainIdx = bestIdx1;
                matches[queryIdx * 2].distance = bestDist1;

                matches[queryIdx * 2 + 1].queryIdx = queryIdx;
                matches[queryIdx * 2 + 1].trainIdx = bestIdx2;
                matches[queryIdx * 2 + 1].distance = bestDist2;
            }
        }

        // ============================================================================
        // Ratio Test Kernel
        // ============================================================================

        __global__ void ratioTestKernel(
            const GpuMatch *__restrict__ knnMatches,
            int numQuery,
            int k,
            float ratio,
            int thHigh,
            GpuMatch *__restrict__ goodMatches,
            int *__restrict__ numGoodMatches)
        {
            int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (queryIdx >= numQuery)
                return;

            GpuMatch best = knnMatches[queryIdx * k];
            GpuMatch secondBest = knnMatches[queryIdx * k + 1];

            // Apply ratio test and threshold
            if (best.distance <= thHigh &&
                best.distance < ratio * secondBest.distance)
            {
                int idx = atomicAdd(numGoodMatches, 1);
                goodMatches[idx] = best;
            }
        }

        // ============================================================================
        // Spatial Match Kernel
        // ============================================================================

        __global__ void spatialMatchKernel(
            const GpuDescriptor *__restrict__ queryDesc,
            const GpuDescriptor *__restrict__ trainDesc,
            const float *__restrict__ trainX,
            const float *__restrict__ trainY,
            const int *__restrict__ trainOctave,
            const GpuProjection *__restrict__ projections,
            const float *__restrict__ radii,
            int numProjections,
            int numTrain,
            GpuMatch *__restrict__ matches,
            int *__restrict__ matchCount,
            int thHigh,
            float nnRatio)
        {
            int projIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (projIdx >= numProjections)
                return;

            GpuProjection proj = projections[projIdx];
            if (!proj.valid)
                return;

            float radius = radii[projIdx];
            float radiusSq = radius * radius;

            int bestDist1 = 256, bestDist2 = 256;
            int bestIdx = -1;

            // Search for matches within radius
            for (int i = 0; i < numTrain; ++i)
            {
                // Check scale level
                if (trainOctave[i] < proj.predictedLevel - 1 ||
                    trainOctave[i] > proj.predictedLevel)
                    continue;

                // Check spatial distance
                float dx = trainX[i] - proj.projX;
                float dy = trainY[i] - proj.projY;
                float distSq = dx * dx + dy * dy;

                if (distSq > radiusSq)
                    continue;

                // Compute descriptor distance
                int descDist = hammingDistance(queryDesc[projIdx], trainDesc[i]);

                if (descDist < bestDist1)
                {
                    bestDist2 = bestDist1;
                    bestDist1 = descDist;
                    bestIdx = i;
                }
                else if (descDist < bestDist2)
                {
                    bestDist2 = descDist;
                }
            }

            // Apply ratio test
            if (bestIdx >= 0 && bestDist1 <= thHigh &&
                bestDist1 < nnRatio * bestDist2)
            {
                int idx = atomicAdd(matchCount, 1);
                matches[idx].queryIdx = projIdx;
                matches[idx].trainIdx = bestIdx;
                matches[idx].distance = bestDist1;
            }
        }

        // ============================================================================
        // Orientation Histogram Kernels
        // ============================================================================

        __global__ void orientationHistKernel(
            const float *__restrict__ queryAngles,
            const float *__restrict__ trainAngles,
            const GpuMatch *__restrict__ matches,
            int numMatches,
            int *__restrict__ histogram,
            int histLength)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numMatches)
                return;

            GpuMatch m = matches[idx];
            float rot = queryAngles[m.queryIdx] - trainAngles[m.trainIdx];

            if (rot < 0.0f)
                rot += 360.0f;

            int bin = __float2int_rn(rot * histLength / 360.0f);
            if (bin >= histLength)
                bin = 0;

            atomicAdd(&histogram[bin], 1);
        }

        __global__ void orientationFilterKernel(
            const float *__restrict__ queryAngles,
            const float *__restrict__ trainAngles,
            const GpuMatch *__restrict__ inMatches,
            GpuMatch *__restrict__ outMatches,
            int numMatches,
            int *__restrict__ numOutMatches,
            int topBin1, int topBin2, int topBin3,
            int histLength)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numMatches)
                return;

            GpuMatch m = inMatches[idx];
            float rot = queryAngles[m.queryIdx] - trainAngles[m.trainIdx];

            if (rot < 0.0f)
                rot += 360.0f;

            int bin = __float2int_rn(rot * histLength / 360.0f);
            if (bin >= histLength)
                bin = 0;

            // Keep match if it's in one of the top 3 bins
            if (bin == topBin1 || bin == topBin2 || bin == topBin3)
            {
                int outIdx = atomicAdd(numOutMatches, 1);
                outMatches[outIdx] = m;
            }
        }

        // ============================================================================
        // ORBMatcherCuda Implementation
        // ============================================================================

        ORBMatcherCuda::ORBMatcherCuda(float nnratio, bool checkOri)
        {
            config_.nnRatio = nnratio;
            config_.checkOrientation = checkOri;

            CUDA_CHECK(cudaStreamCreateWithFlags(&matchStream_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&sortStream_, cudaStreamNonBlocking));

            CUDA_CHECK(cudaEventCreate(&startEvent_));
            CUDA_CHECK(cudaEventCreate(&stopEvent_));

            // Initialize buffers
            initializeBuffers(5000);
        }

        ORBMatcherCuda::ORBMatcherCuda(const MatchingConfig &config) : config_(config)
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&matchStream_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&sortStream_, cudaStreamNonBlocking));

            CUDA_CHECK(cudaEventCreate(&startEvent_));
            CUDA_CHECK(cudaEventCreate(&stopEvent_));

            // Initialize buffers
            initializeBuffers(config_.maxMatches);
        }

        ORBMatcherCuda::~ORBMatcherCuda()
        {
            cudaStreamDestroy(matchStream_);
            cudaStreamDestroy(sortStream_);
            cudaEventDestroy(startEvent_);
            cudaEventDestroy(stopEvent_);
        }

        void ORBMatcherCuda::initializeBuffers(int maxDescriptors)
        {
            size_t maxDistances = maxDescriptors * maxDescriptors;
            distanceBuffer_.resize(maxDistances);
            indexBuffer_.resize(maxDescriptors);
            tempMatchBuffer_.resize(maxDescriptors * 2); // For KNN k=2
            orientationHist_.resize(config_.histoLength);
        }

        int ORBMatcherCuda::matchGpu(const GpuDescriptorArray &queryDescriptors,
                                     const GpuDescriptorArray &trainDescriptors,
                                     GpuMatchArray &matches)
        {
            if (queryDescriptors.count == 0 || trainDescriptors.count == 0)
                return 0;

            CUDA_CHECK(cudaEventRecord(startEvent_, matchStream_));

            int numQuery = queryDescriptors.count;
            int numTrain = trainDescriptors.count;

            // Ensure buffers are large enough
            if (distanceBuffer_.size() < static_cast<size_t>(numQuery * numTrain))
            {
                distanceBuffer_.resize(numQuery * numTrain);
            }
            if (tempMatchBuffer_.size() < static_cast<size_t>(numQuery * 2))
            {
                tempMatchBuffer_.resize(numQuery * 2);
            }

            // Compute distance matrix
            CUDA_CHECK(cudaEventRecord(startEvent_, matchStream_));

            dim3 block(16, 16);
            dim3 grid((numTrain + block.x - 1) / block.x,
                      (numQuery + block.y - 1) / block.y);

            hammingDistanceMatrixSharedKernel<<<grid, block, 0, matchStream_>>>(
                queryDescriptors.descriptors,
                trainDescriptors.descriptors,
                numQuery, numTrain,
                distanceBuffer_.data());

            CUDA_CHECK_LAST();

            cudaEvent_t distEvent;
            CUDA_CHECK(cudaEventCreate(&distEvent));
            CUDA_CHECK(cudaEventRecord(distEvent, matchStream_));
            CUDA_CHECK(cudaEventSynchronize(distEvent));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.distanceComputeMs, startEvent_, distEvent));

            // KNN search (k=2 for ratio test)
            CUDA_CHECK(cudaEventRecord(startEvent_, matchStream_));

            dim3 knnBlock(256);
            dim3 knnGrid((numQuery + knnBlock.x - 1) / knnBlock.x);

            knnSearchKernel<<<knnGrid, knnBlock, 0, matchStream_>>>(
                distanceBuffer_.data(),
                numQuery, numTrain, 2,
                tempMatchBuffer_.data());

            CUDA_CHECK_LAST();

            cudaEvent_t knnEvent;
            CUDA_CHECK(cudaEventCreate(&knnEvent));
            CUDA_CHECK(cudaEventRecord(knnEvent, matchStream_));
            CUDA_CHECK(cudaEventSynchronize(knnEvent));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.sortingMs, distEvent, knnEvent));

            // Apply ratio test
            CUDA_CHECK(cudaEventRecord(startEvent_, matchStream_));

            // Reset match count
            CUDA_CHECK(cudaMemsetAsync(matches.matchCount, 0, sizeof(int), matchStream_));

            ratioTestKernel<<<knnGrid, knnBlock, 0, matchStream_>>>(
                tempMatchBuffer_.data(),
                numQuery, 2,
                config_.nnRatio,
                config_.thHigh,
                matches.matches,
                matches.matchCount);

            CUDA_CHECK_LAST();

            CUDA_CHECK(cudaEventRecord(stopEvent_, matchStream_));
            CUDA_CHECK(cudaStreamSynchronize(matchStream_));
            CUDA_CHECK(cudaEventElapsedTime(&stats_.ratioTestMs, knnEvent, stopEvent_));

            // Get number of matches
            int numMatches;
            CUDA_CHECK(cudaMemcpy(&numMatches, matches.matchCount, sizeof(int), cudaMemcpyDeviceToHost));

            stats_.numMatches = numMatches;
            stats_.totalMs = stats_.distanceComputeMs + stats_.sortingMs + stats_.ratioTestMs;

            cudaEventDestroy(distEvent);
            cudaEventDestroy(knnEvent);

            return numMatches;
        }

        int ORBMatcherCuda::searchByProjectionGpu(const GpuFrameData &frame,
                                                  const GpuDescriptorArray &mapPointDescriptors,
                                                  const GpuProjectionArray &projections,
                                                  float *radii,
                                                  GpuMatchArray &matches)
        {
            if (projections.count == 0 || frame.numKeypoints == 0)
                return 0;

            // Reset match count
            CUDA_CHECK(cudaMemsetAsync(matches.matchCount, 0, sizeof(int), matchStream_));

            dim3 block(256);
            dim3 grid((projections.count + block.x - 1) / block.x);

            spatialMatchKernel<<<grid, block, 0, matchStream_>>>(
                mapPointDescriptors.descriptors,
                frame.descriptors.descriptors,
                frame.keypoints.x,
                frame.keypoints.y,
                frame.keypoints.octave,
                projections.projections,
                radii,
                projections.count,
                frame.numKeypoints,
                matches.matches,
                matches.matchCount,
                config_.thHigh,
                config_.nnRatio);

            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaStreamSynchronize(matchStream_));

            int numMatches;
            CUDA_CHECK(cudaMemcpy(&numMatches, matches.matchCount, sizeof(int), cudaMemcpyDeviceToHost));

            return numMatches;
        }

        void ORBMatcherCuda::checkOrientationGpu(const GpuKeyPointSoA &queryKeypoints,
                                                 const GpuKeyPointSoA &trainKeypoints,
                                                 GpuMatchArray &matches)
        {
            int numMatches;
            CUDA_CHECK(cudaMemcpy(&numMatches, matches.matchCount, sizeof(int), cudaMemcpyDeviceToHost));

            if (numMatches == 0)
                return;

            // Build histogram
            CUDA_CHECK(cudaMemset(orientationHist_.data(), 0, config_.histoLength * sizeof(int)));

            dim3 block(256);
            dim3 grid((numMatches + block.x - 1) / block.x);

            orientationHistKernel<<<grid, block, 0, matchStream_>>>(
                queryKeypoints.angle,
                trainKeypoints.angle,
                matches.matches,
                numMatches,
                orientationHist_.data(),
                config_.histoLength);

            CUDA_CHECK(cudaStreamSynchronize(matchStream_));

            // Find top 3 bins on CPU (small array)
            std::vector<int> hist(config_.histoLength);
            orientationHist_.copyToHost(hist.data());

            int topBin1 = -1, topBin2 = -1, topBin3 = -1;
            int topVal1 = 0, topVal2 = 0, topVal3 = 0;

            for (int i = 0; i < config_.histoLength; ++i)
            {
                if (hist[i] > topVal1)
                {
                    topVal3 = topVal2;
                    topBin3 = topBin2;
                    topVal2 = topVal1;
                    topBin2 = topBin1;
                    topVal1 = hist[i];
                    topBin1 = i;
                }
                else if (hist[i] > topVal2)
                {
                    topVal3 = topVal2;
                    topBin3 = topBin2;
                    topVal2 = hist[i];
                    topBin2 = i;
                }
                else if (hist[i] > topVal3)
                {
                    topVal3 = hist[i];
                    topBin3 = i;
                }
            }

            // Filter matches
            GpuArray<GpuMatch> filteredMatches(numMatches);
            int *d_numFiltered;
            CUDA_CHECK(cudaMalloc(&d_numFiltered, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_numFiltered, 0, sizeof(int)));

            orientationFilterKernel<<<grid, block, 0, matchStream_>>>(
                queryKeypoints.angle,
                trainKeypoints.angle,
                matches.matches,
                filteredMatches.data(),
                numMatches,
                d_numFiltered,
                topBin1, topBin2, topBin3,
                config_.histoLength);

            CUDA_CHECK(cudaStreamSynchronize(matchStream_));

            // Copy filtered matches back
            int numFiltered;
            CUDA_CHECK(cudaMemcpy(&numFiltered, d_numFiltered, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(matches.matches, filteredMatches.data(),
                                  numFiltered * sizeof(GpuMatch), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(matches.matchCount, &numFiltered, sizeof(int), cudaMemcpyHostToDevice));

            cudaFree(d_numFiltered);
        }

        void ORBMatcherCuda::downloadMatches(const GpuMatchArray &gpuMatches,
                                             std::vector<cv::DMatch> &cpuMatches)
        {
            int numMatches;
            CUDA_CHECK(cudaMemcpy(&numMatches, gpuMatches.matchCount, sizeof(int), cudaMemcpyDeviceToHost));

            if (numMatches == 0)
            {
                cpuMatches.clear();
                return;
            }

            std::vector<GpuMatch> tempMatches(numMatches);
            CUDA_CHECK(cudaMemcpy(tempMatches.data(), gpuMatches.matches,
                                  numMatches * sizeof(GpuMatch), cudaMemcpyDeviceToHost));

            cpuMatches.resize(numMatches);
            for (int i = 0; i < numMatches; ++i)
            {
                cpuMatches[i].queryIdx = tempMatches[i].queryIdx;
                cpuMatches[i].trainIdx = tempMatches[i].trainIdx;
                cpuMatches[i].distance = static_cast<float>(tempMatches[i].distance);
            }
        }

        int ORBMatcherCuda::descriptorDistance(const cv::Mat &a, const cv::Mat &b)
        {
            const int *pa = a.ptr<int32_t>();
            const int *pb = b.ptr<int32_t>();

            int dist = 0;
            for (int i = 0; i < 8; ++i)
            {
                unsigned int v = pa[i] ^ pb[i];
#if defined(__GNUC__) || defined(__clang__)
                dist += __builtin_popcount(v);
#else
                v = v - ((v >> 1) & 0x55555555);
                v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
                dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
            }

            return dist;
        }

        // ============================================================================
        // Kernel Launch Wrappers
        // ============================================================================

        void launchHammingDistanceKernel(
            const GpuDescriptor *queryDescriptors,
            const GpuDescriptor *trainDescriptors,
            int numQuery,
            int numTrain,
            int *distances,
            cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid((numTrain + block.x - 1) / block.x,
                      (numQuery + block.y - 1) / block.y);

            hammingDistanceMatrixSharedKernel<<<grid, block, 0, stream>>>(
                queryDescriptors, trainDescriptors,
                numQuery, numTrain, distances);
        }

        void launchKnnSearchKernel(
            const int *distances,
            int numQuery,
            int numTrain,
            int k,
            GpuMatch *matches,
            cudaStream_t stream)
        {
            dim3 block(256);
            dim3 grid((numQuery + block.x - 1) / block.x);

            knnSearchKernel<<<grid, block, 0, stream>>>(
                distances, numQuery, numTrain, k, matches);
        }

        void launchRatioTestKernel(
            const GpuMatch *knnMatches,
            int numQuery,
            int k,
            float ratio,
            GpuMatch *goodMatches,
            int *numGoodMatches,
            cudaStream_t stream)
        {
            dim3 block(256);
            dim3 grid((numQuery + block.x - 1) / block.x);

            ratioTestKernel<<<grid, block, 0, stream>>>(
                knnMatches, numQuery, k, ratio, 100, goodMatches, numGoodMatches);
        }

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
            cudaStream_t stream)
        {
            dim3 block(256);
            dim3 grid((numProjections + block.x - 1) / block.x);

            spatialMatchKernel<<<grid, block, 0, stream>>>(
                queryDescriptors, trainDescriptors,
                trainKeypoints.x, trainKeypoints.y, trainKeypoints.octave,
                projections, radii,
                numProjections, numTrainDescriptors,
                matches, matchCount,
                thHigh, nnRatio);
        }

    } // namespace cuda
} // namespace ORB_SLAM3
