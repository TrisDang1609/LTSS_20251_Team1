/**
 * Feature Grid CUDA - GPU-Accelerated Spatial Feature Indexing
 *
 * Provides GPU-resident spatial indexing for efficient feature lookup.
 * Used by SearchByProjection and other spatial queries.
 *
 * Features:
 * 1. Parallel grid cell assignment
 * 2. Thrust-based sorting by cell
 * 3. GPU-resident radius search
 * 4. Cell-based feature retrieval
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 */

#ifndef FEATURE_GRID_CUDA_H
#define FEATURE_GRID_CUDA_H

#include "CudaUtils.h"
#include "GpuTypes.h"
#include "CudaMemoryManager.h"

#include <vector>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Feature Grid CUDA Class
        // ============================================================================

        class FeatureGridCuda
        {
        public:
            /**
             * Constructor
             * @param imageWidth Image width in pixels
             * @param imageHeight Image height in pixels
             * @param cellsX Number of grid cells horizontally
             * @param cellsY Number of grid cells vertically
             */
            FeatureGridCuda(int imageWidth = 640, int imageHeight = 480,
                            int cellsX = GRID_COLS, int cellsY = GRID_ROWS);

            ~FeatureGridCuda();

            /**
             * Initialize grid for given image dimensions
             */
            void initialize(int imageWidth, int imageHeight);

            /**
             * Build grid from keypoints (GPU-resident)
             * Assigns keypoints to cells and prepares for queries
             */
            void buildGrid(const GpuKeyPointSoA &keypoints, cudaStream_t stream = 0);

            /**
             * Get features in area (GPU-resident query)
             * Returns indices of features within radius of (x, y)
             *
             * @param x Query x coordinate
             * @param y Query y coordinate
             * @param radius Search radius in pixels
             * @param minLevel Minimum pyramid level (-1 for all)
             * @param maxLevel Maximum pyramid level (-1 for all)
             * @param indices Output array of feature indices
             * @param count Output count of features found
             * @param stream CUDA stream
             */
            void getFeaturesInArea(float x, float y, float radius,
                                   int minLevel, int maxLevel,
                                   int *indices, int *count,
                                   cudaStream_t stream = 0);

            /**
             * Batch query - get features for multiple projections
             * More efficient for SearchByProjection
             */
            void getFeaturesInAreaBatch(const GpuProjection *projections,
                                        const float *radii,
                                        int numProjections,
                                        int *indicesPerQuery,
                                        int *countsPerQuery,
                                        int maxIndicesPerQuery,
                                        cudaStream_t stream = 0);

            /**
             * Get raw grid data (for advanced queries)
             */
            GpuFeatureGrid &getGrid() { return grid_; }
            const GpuFeatureGrid &getGrid() const { return grid_; }

            /**
             * Reset grid (clear all cells)
             */
            void reset(cudaStream_t stream = 0);

            // Grid dimensions
            int getCellsX() const { return cellsX_; }
            int getCellsY() const { return cellsY_; }
            int getNumCells() const { return cellsX_ * cellsY_; }

        private:
            // Grid parameters
            int imageWidth_;
            int imageHeight_;
            int cellsX_;
            int cellsY_;
            float cellWidth_;
            float cellHeight_;

            // GPU grid structure
            GpuFeatureGrid grid_;

            // Temporary buffers
            GpuArray<int> cellAssignments_; // Cell index for each keypoint
            GpuArray<int> sortedIndices_;   // Keypoint indices sorted by cell

            // Internal methods
            void allocateGrid();
            void computeCellParameters();
        };

        // ============================================================================
        // CUDA Kernel Declarations
        // ============================================================================

        /**
         * Assign keypoints to grid cells
         */
        void launchAssignCellsKernel(
            const float *kpX,
            const float *kpY,
            int numKeypoints,
            int *cellAssignments,
            float invCellWidth,
            float invCellHeight,
            int cellsX,
            int cellsY,
            cudaStream_t stream);

        /**
         * Count keypoints per cell (histogram)
         */
        void launchCountCellsKernel(
            const int *cellAssignments,
            int numKeypoints,
            int *cellCounts,
            int numCells,
            cudaStream_t stream);

        /**
         * Compute cell start indices (prefix sum)
         */
        void launchPrefixSumKernel(
            int *cellCounts,
            int *cellStarts,
            int numCells,
            cudaStream_t stream);

        /**
         * Sort keypoint indices by cell
         */
        void launchSortByCellKernel(
            const int *cellAssignments,
            int numKeypoints,
            int *sortedIndices,
            int *cellStarts,
            int *cellCounts,
            cudaStream_t stream);

        /**
         * Radius search within grid
         */
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
            cudaStream_t stream);

        /**
         * Batch radius search for multiple queries
         */
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
            cudaStream_t stream);

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // FEATURE_GRID_CUDA_H
