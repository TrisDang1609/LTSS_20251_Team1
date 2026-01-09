/**
 * Feature Grid CUDA Implementation
 *
 * GPU-accelerated spatial indexing for feature queries.
 */

#include "cuda/FeatureGridCuda.h"
#include "cuda/CudaMemoryManager.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // CUDA Kernels
        // ============================================================================

        __global__ void assignCellsKernel(
            const float *__restrict__ kpX,
            const float *__restrict__ kpY,
            int numKeypoints,
            int *__restrict__ cellAssignments,
            float invCellWidth,
            float invCellHeight,
            int cellsX,
            int cellsY)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            int cellX = min(static_cast<int>(kpX[idx] * invCellWidth), cellsX - 1);
            int cellY = min(static_cast<int>(kpY[idx] * invCellHeight), cellsY - 1);
            cellX = max(0, cellX);
            cellY = max(0, cellY);

            cellAssignments[idx] = cellY * cellsX + cellX;
        }

        __global__ void countCellsKernel(
            const int *__restrict__ cellAssignments,
            int numKeypoints,
            int *__restrict__ cellCounts,
            int numCells)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            int cell = cellAssignments[idx];
            if (cell >= 0 && cell < numCells)
            {
                atomicAdd(&cellCounts[cell], 1);
            }
        }

        __global__ void scatterToGridKernel(
            const int *__restrict__ cellAssignments,
            int numKeypoints,
            int *__restrict__ featureIndices,
            int *__restrict__ cellStarts,
            int *__restrict__ cellPositions) // Temporary: current position in each cell
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numKeypoints)
                return;

            int cell = cellAssignments[idx];
            int pos = atomicAdd(&cellPositions[cell], 1);
            featureIndices[cellStarts[cell] + pos] = idx;
        }

        __global__ void radiusSearchKernel(
            const float *__restrict__ kpX,
            const float *__restrict__ kpY,
            const int *__restrict__ kpOctave,
            const int *__restrict__ cellStarts,
            const int *__restrict__ cellCounts,
            const int *__restrict__ featureIndices,
            float invCellWidth,
            float invCellHeight,
            int cellsX,
            int cellsY,
            float queryX,
            float queryY,
            float radius,
            int minLevel,
            int maxLevel,
            int *__restrict__ resultIndices,
            int *__restrict__ resultCount,
            int maxResults)
        {
            // Single-threaded for single query
            // For batch queries, use batchRadiusSearchKernel

            if (threadIdx.x != 0 || blockIdx.x != 0)
                return;

            float radiusSq = radius * radius;
            int count = 0;

            // Calculate cell range to search
            int minCellX = max(0, static_cast<int>((queryX - radius) * invCellWidth));
            int maxCellX = min(cellsX - 1, static_cast<int>((queryX + radius) * invCellWidth));
            int minCellY = max(0, static_cast<int>((queryY - radius) * invCellHeight));
            int maxCellY = min(cellsY - 1, static_cast<int>((queryY + radius) * invCellHeight));

            // Search cells in range
            for (int cy = minCellY; cy <= maxCellY && count < maxResults; ++cy)
            {
                for (int cx = minCellX; cx <= maxCellX && count < maxResults; ++cx)
                {
                    int cellIdx = cy * cellsX + cx;
                    int start = cellStarts[cellIdx];
                    int cellCount = cellCounts[cellIdx];

                    for (int i = 0; i < cellCount && count < maxResults; ++i)
                    {
                        int kpIdx = featureIndices[start + i];

                        // Check level
                        if (minLevel >= 0 && kpOctave[kpIdx] < minLevel)
                            continue;
                        if (maxLevel >= 0 && kpOctave[kpIdx] > maxLevel)
                            continue;

                        // Check distance
                        float dx = kpX[kpIdx] - queryX;
                        float dy = kpY[kpIdx] - queryY;
                        float distSq = dx * dx + dy * dy;

                        if (distSq <= radiusSq)
                        {
                            resultIndices[count++] = kpIdx;
                        }
                    }
                }
            }

            *resultCount = count;
        }

        __global__ void batchRadiusSearchKernel(
            const float *__restrict__ kpX,
            const float *__restrict__ kpY,
            const int *__restrict__ kpOctave,
            const int *__restrict__ cellStarts,
            const int *__restrict__ cellCounts,
            const int *__restrict__ featureIndices,
            float invCellWidth,
            float invCellHeight,
            int cellsX,
            int cellsY,
            const GpuProjection *__restrict__ projections,
            const float *__restrict__ radii,
            int numQueries,
            int *__restrict__ resultIndices,
            int *__restrict__ resultCounts,
            int maxResultsPerQuery)
        {
            int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (queryIdx >= numQueries)
                return;

            GpuProjection proj = projections[queryIdx];
            if (!proj.valid)
            {
                resultCounts[queryIdx] = 0;
                return;
            }

            float queryX = proj.projX;
            float queryY = proj.projY;
            float radius = radii[queryIdx];
            float radiusSq = radius * radius;
            int predictedLevel = proj.predictedLevel;

            int *results = resultIndices + queryIdx * maxResultsPerQuery;
            int count = 0;

            // Calculate cell range
            int minCellX = max(0, static_cast<int>((queryX - radius) * invCellWidth));
            int maxCellX = min(cellsX - 1, static_cast<int>((queryX + radius) * invCellWidth));
            int minCellY = max(0, static_cast<int>((queryY - radius) * invCellHeight));
            int maxCellY = min(cellsY - 1, static_cast<int>((queryY + radius) * invCellHeight));

            // Search cells
            for (int cy = minCellY; cy <= maxCellY && count < maxResultsPerQuery; ++cy)
            {
                for (int cx = minCellX; cx <= maxCellX && count < maxResultsPerQuery; ++cx)
                {
                    int cellIdx = cy * cellsX + cx;
                    int start = cellStarts[cellIdx];
                    int cellCount = cellCounts[cellIdx];

                    for (int i = 0; i < cellCount && count < maxResultsPerQuery; ++i)
                    {
                        int kpIdx = featureIndices[start + i];

                        // Check level (predicted level +/- 1)
                        int level = kpOctave[kpIdx];
                        if (level < predictedLevel - 1 || level > predictedLevel)
                            continue;

                        // Check distance
                        float dx = kpX[kpIdx] - queryX;
                        float dy = kpY[kpIdx] - queryY;
                        float distSq = dx * dx + dy * dy;

                        if (distSq <= radiusSq)
                        {
                            results[count++] = kpIdx;
                        }
                    }
                }
            }

            resultCounts[queryIdx] = count;
        }

        // ============================================================================
        // FeatureGridCuda Implementation
        // ============================================================================

        FeatureGridCuda::FeatureGridCuda(int imageWidth, int imageHeight,
                                         int cellsX, int cellsY)
            : imageWidth_(imageWidth), imageHeight_(imageHeight),
              cellsX_(cellsX), cellsY_(cellsY)
        {
            computeCellParameters();
            allocateGrid();
        }

        FeatureGridCuda::~FeatureGridCuda()
        {
            if (grid_.cellStart)
                cudaFree(grid_.cellStart);
            if (grid_.cellCount)
                cudaFree(grid_.cellCount);
            if (grid_.featureIndices)
                cudaFree(grid_.featureIndices);
        }

        void FeatureGridCuda::initialize(int imageWidth, int imageHeight)
        {
            imageWidth_ = imageWidth;
            imageHeight_ = imageHeight;
            computeCellParameters();
        }

        void FeatureGridCuda::computeCellParameters()
        {
            cellWidth_ = static_cast<float>(imageWidth_) / cellsX_;
            cellHeight_ = static_cast<float>(imageHeight_) / cellsY_;

            grid_.cellWidth = cellWidth_;
            grid_.cellHeight = cellHeight_;
            grid_.invCellWidth = 1.0f / cellWidth_;
            grid_.invCellHeight = 1.0f / cellHeight_;
            grid_.imageWidth = imageWidth_;
            grid_.imageHeight = imageHeight_;
        }

        void FeatureGridCuda::allocateGrid()
        {
            int numCells = cellsX_ * cellsY_;

            CUDA_CHECK(cudaMalloc(&grid_.cellStart, numCells * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&grid_.cellCount, numCells * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&grid_.featureIndices, MAX_KEYPOINTS_PER_FRAME * sizeof(int)));

            cellAssignments_.resize(MAX_KEYPOINTS_PER_FRAME);
            sortedIndices_.resize(MAX_KEYPOINTS_PER_FRAME);
        }

        void FeatureGridCuda::reset(cudaStream_t stream)
        {
            int numCells = cellsX_ * cellsY_;
            CUDA_CHECK(cudaMemsetAsync(grid_.cellCount, 0, numCells * sizeof(int), stream));
        }

        void FeatureGridCuda::buildGrid(const GpuKeyPointSoA &keypoints, cudaStream_t stream)
        {
            if (keypoints.count == 0)
            {
                reset(stream);
                return;
            }

            int numCells = cellsX_ * cellsY_;

            // Step 1: Reset cell counts
            CUDA_CHECK(cudaMemsetAsync(grid_.cellCount, 0, numCells * sizeof(int), stream));

            // Ensure buffers are large enough
            if (cellAssignments_.size() < static_cast<size_t>(keypoints.count))
            {
                cellAssignments_.resize(keypoints.count);
            }

            dim3 block(256);
            dim3 grid((keypoints.count + block.x - 1) / block.x);

            // Step 2: Assign keypoints to cells
            assignCellsKernel<<<grid, block, 0, stream>>>(
                keypoints.x,
                keypoints.y,
                keypoints.count,
                cellAssignments_.data(),
                grid_.invCellWidth,
                grid_.invCellHeight,
                cellsX_,
                cellsY_);

            CUDA_CHECK_LAST();

            // Step 3: Count keypoints per cell
            countCellsKernel<<<grid, block, 0, stream>>>(
                cellAssignments_.data(),
                keypoints.count,
                grid_.cellCount,
                numCells);

            CUDA_CHECK_LAST();

            // Step 4: Compute cell start indices (exclusive prefix sum)
            thrust::device_ptr<int> countPtr(grid_.cellCount);
            thrust::device_ptr<int> startPtr(grid_.cellStart);

            thrust::exclusive_scan(thrust::cuda::par.on(stream),
                                   countPtr, countPtr + numCells,
                                   startPtr);

            // Step 5: Scatter keypoints to grid (need temp array for positions)
            GpuArray<int> cellPositions(numCells);
            CUDA_CHECK(cudaMemsetAsync(cellPositions.data(), 0, numCells * sizeof(int), stream));

            scatterToGridKernel<<<grid, block, 0, stream>>>(
                cellAssignments_.data(),
                keypoints.count,
                grid_.featureIndices,
                grid_.cellStart,
                cellPositions.data());

            CUDA_CHECK_LAST();
        }

        void FeatureGridCuda::getFeaturesInArea(float x, float y, float radius,
                                                int minLevel, int maxLevel,
                                                int *indices, int *count,
                                                cudaStream_t stream)
        {
            // For single queries, use simple kernel
            // TODO: Store keypoint pointers in class for this to work
            // For now, this needs to be called with external keypoint data

            CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int), stream));

            // Note: This is a placeholder - in practice, you'd pass keypoint data
        }

        void FeatureGridCuda::getFeaturesInAreaBatch(const GpuProjection *projections,
                                                     const float *radii,
                                                     int numProjections,
                                                     int *indicesPerQuery,
                                                     int *countsPerQuery,
                                                     int maxIndicesPerQuery,
                                                     cudaStream_t stream)
        {
            if (numProjections == 0)
                return;

            dim3 block(256);
            dim3 gridDim((numProjections + block.x - 1) / block.x);

            // Note: This needs keypoint data passed in
            // The batch kernel expects keypoint arrays
        }

        // ============================================================================
        // Kernel Launch Wrappers
        // ============================================================================

        void launchAssignCellsKernel(
            const float *kpX,
            const float *kpY,
            int numKeypoints,
            int *cellAssignments,
            float invCellWidth,
            float invCellHeight,
            int cellsX,
            int cellsY,
            cudaStream_t stream)
        {
            dim3 block(256);
            dim3 grid((numKeypoints + block.x - 1) / block.x);

            assignCellsKernel<<<grid, block, 0, stream>>>(
                kpX, kpY, numKeypoints, cellAssignments,
                invCellWidth, invCellHeight, cellsX, cellsY);
        }

        void launchCountCellsKernel(
            const int *cellAssignments,
            int numKeypoints,
            int *cellCounts,
            int numCells,
            cudaStream_t stream)
        {
            dim3 block(256);
            dim3 grid((numKeypoints + block.x - 1) / block.x);

            countCellsKernel<<<grid, block, 0, stream>>>(
                cellAssignments, numKeypoints, cellCounts, numCells);
        }

        void launchPrefixSumKernel(
            int *cellCounts,
            int *cellStarts,
            int numCells,
            cudaStream_t stream)
        {
            thrust::device_ptr<int> countPtr(cellCounts);
            thrust::device_ptr<int> startPtr(cellStarts);

            thrust::exclusive_scan(thrust::cuda::par.on(stream),
                                   countPtr, countPtr + numCells,
                                   startPtr);
        }

        void launchRadiusSearchKernel(
            const float *kpX,
            const float *kpY,
            const int *kpOctave,
            const GpuFeatureGrid &grid,
            float queryX,
            float queryY,
            float radius,
            int minLevel,
            int maxLevel,
            int *resultIndices,
            int *resultCount,
            int maxResults,
            cudaStream_t stream)
        {
            radiusSearchKernel<<<1, 1, 0, stream>>>(
                kpX, kpY, kpOctave,
                grid.cellStart, grid.cellCount, grid.featureIndices,
                grid.invCellWidth, grid.invCellHeight,
                GRID_COLS, GRID_ROWS,
                queryX, queryY, radius,
                minLevel, maxLevel,
                resultIndices, resultCount, maxResults);
        }

        void launchBatchRadiusSearchKernel(
            const float *kpX,
            const float *kpY,
            const int *kpOctave,
            const GpuFeatureGrid &grid,
            const GpuProjection *projections,
            const float *radii,
            int numQueries,
            int *resultIndices,
            int *resultCounts,
            int maxResultsPerQuery,
            cudaStream_t stream)
        {
            dim3 block(256);
            dim3 gridDim((numQueries + block.x - 1) / block.x);

            batchRadiusSearchKernel<<<gridDim, block, 0, stream>>>(
                kpX, kpY, kpOctave,
                grid.cellStart, grid.cellCount, grid.featureIndices,
                grid.invCellWidth, grid.invCellHeight,
                GRID_COLS, GRID_ROWS,
                projections, radii, numQueries,
                resultIndices, resultCounts, maxResultsPerQuery);
        }

    } // namespace cuda
} // namespace ORB_SLAM3
