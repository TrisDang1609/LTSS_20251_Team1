/**
 * GPU Memory Manager Implementation
 *
 * Implementation of the unified GPU memory pool with stream-ordered allocations.
 */

#include "cuda/CudaMemoryManager.h"
#include <iostream>
#include <algorithm>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // GpuMemoryPool Implementation
        // ============================================================================

        void GpuMemoryPool::initialize(const MemoryPoolConfig &config)
        {
            if (initialized_)
            {
                std::cerr << "GpuMemoryPool already initialized" << std::endl;
                return;
            }

            config_ = config;

            // Allocate all pools
            allocateKeypointBuffers();
            allocateDescriptorBuffer();
            allocatePyramidBuffer();
            allocateFeatureGrid();
            allocateMatchBuffer();

            initialized_ = true;

            std::cout << "GpuMemoryPool initialized with "
                      << (totalAllocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
        }

        void GpuMemoryPool::release()
        {
            if (!initialized_)
                return;

            // Free keypoint buffers
            for (auto &kpBuf : keypointBuffers_)
            {
                if (kpBuf.x)
                    cudaFree(kpBuf.x);
                if (kpBuf.y)
                    cudaFree(kpBuf.y);
                if (kpBuf.size)
                    cudaFree(kpBuf.size);
                if (kpBuf.angle)
                    cudaFree(kpBuf.angle);
                if (kpBuf.response)
                    cudaFree(kpBuf.response);
                if (kpBuf.octave)
                    cudaFree(kpBuf.octave);
            }
            keypointBuffers_.clear();

            // Free descriptor buffer
            if (descriptorBuffer_.descriptors)
            {
                cudaFree(descriptorBuffer_.descriptors);
            }

            // Free feature grid
            if (featureGrid_.cellStart)
                cudaFree(featureGrid_.cellStart);
            if (featureGrid_.cellCount)
                cudaFree(featureGrid_.cellCount);
            if (featureGrid_.featureIndices)
                cudaFree(featureGrid_.featureIndices);

            // Free match buffer
            if (matchBuffer_.matches)
                cudaFree(matchBuffer_.matches);
            if (matchBuffer_.matchCount)
                cudaFree(matchBuffer_.matchCount);

            // Free temp buffers
            for (auto &pair : tempBufferPool_)
            {
                for (void *ptr : pair.second)
                {
                    cudaFree(ptr);
                }
            }
            tempBufferPool_.clear();

            totalAllocated_ = 0;
            initialized_ = false;
        }

        void GpuMemoryPool::allocateKeypointBuffers()
        {
            keypointBuffers_.resize(config_.maxPyramidLevels);

            size_t keypointsPerLevel = static_cast<size_t>(
                config_.maxKeypoints * config_.bufferMultiplier / config_.maxPyramidLevels);

            for (size_t i = 0; i < config_.maxPyramidLevels; ++i)
            {
                GpuKeyPointSoA &kp = keypointBuffers_[i];
                kp.capacity = static_cast<int>(keypointsPerLevel);
                kp.count = 0;

                size_t bytes = keypointsPerLevel * sizeof(float);
                CUDA_CHECK(cudaMalloc(&kp.x, bytes));
                CUDA_CHECK(cudaMalloc(&kp.y, bytes));
                CUDA_CHECK(cudaMalloc(&kp.size, bytes));
                CUDA_CHECK(cudaMalloc(&kp.angle, bytes));
                CUDA_CHECK(cudaMalloc(&kp.response, bytes));
                CUDA_CHECK(cudaMalloc(&kp.octave, keypointsPerLevel * sizeof(int)));

                totalAllocated_ += 5 * bytes + keypointsPerLevel * sizeof(int);
            }

            peakAllocated_ = std::max(peakAllocated_, totalAllocated_);
        }

        void GpuMemoryPool::allocateDescriptorBuffer()
        {
            size_t numDescriptors = static_cast<size_t>(
                config_.maxKeypoints * config_.bufferMultiplier);

            descriptorBuffer_.capacity = static_cast<int>(numDescriptors);
            descriptorBuffer_.count = 0;

            size_t bytes = numDescriptors * sizeof(GpuDescriptor);
            CUDA_CHECK(cudaMalloc(&descriptorBuffer_.descriptors, bytes));

            totalAllocated_ += bytes;
            peakAllocated_ = std::max(peakAllocated_, totalAllocated_);
        }

        void GpuMemoryPool::allocatePyramidBuffer()
        {
            pyramidBuffer_.numLevels = static_cast<int>(config_.maxPyramidLevels);
            pyramidBuffer_.scaleFactor = 1.2f; // Default, can be changed

            float scale = 1.0f;
            for (size_t i = 0; i < config_.maxPyramidLevels; ++i)
            {
                int width = static_cast<int>(config_.maxImageWidth / scale);
                int height = static_cast<int>(config_.maxImageHeight / scale);

                // Pre-allocate GpuMat with border for edge handling
                int paddedWidth = width + 2 * EDGE_THRESHOLD;
                int paddedHeight = height + 2 * EDGE_THRESHOLD;

                pyramidBuffer_.levels[i].image.create(paddedHeight, paddedWidth, CV_8UC1);
                pyramidBuffer_.levels[i].blurred.create(paddedHeight, paddedWidth, CV_8UC1);
                pyramidBuffer_.levels[i].scale = scale;
                pyramidBuffer_.levels[i].invScale = 1.0f / scale;
                pyramidBuffer_.levels[i].width = width;
                pyramidBuffer_.levels[i].height = height;

                // Count memory
                totalAllocated_ += 2 * paddedWidth * paddedHeight;

                scale *= 1.2f; // Default scale factor
            }

            peakAllocated_ = std::max(peakAllocated_, totalAllocated_);
        }

        void GpuMemoryPool::allocateFeatureGrid()
        {
            // Grid cell arrays
            size_t gridBytes = GRID_CELLS * sizeof(int);
            CUDA_CHECK(cudaMalloc(&featureGrid_.cellStart, gridBytes));
            CUDA_CHECK(cudaMalloc(&featureGrid_.cellCount, gridBytes));

            // Feature indices (max keypoints)
            size_t indicesBytes = config_.maxKeypoints * sizeof(int);
            CUDA_CHECK(cudaMalloc(&featureGrid_.featureIndices, indicesBytes));

            // Initialize grid dimensions
            featureGrid_.imageWidth = config_.maxImageWidth;
            featureGrid_.imageHeight = config_.maxImageHeight;
            featureGrid_.cellWidth = static_cast<float>(config_.maxImageWidth) / GRID_COLS;
            featureGrid_.cellHeight = static_cast<float>(config_.maxImageHeight) / GRID_ROWS;
            featureGrid_.invCellWidth = 1.0f / featureGrid_.cellWidth;
            featureGrid_.invCellHeight = 1.0f / featureGrid_.cellHeight;

            totalAllocated_ += 2 * gridBytes + indicesBytes;
            peakAllocated_ = std::max(peakAllocated_, totalAllocated_);
        }

        void GpuMemoryPool::allocateMatchBuffer()
        {
            size_t numMatches = static_cast<size_t>(
                config_.maxMatches * config_.bufferMultiplier);

            matchBuffer_.capacity = static_cast<int>(numMatches);

            CUDA_CHECK(cudaMalloc(&matchBuffer_.matches, numMatches * sizeof(GpuMatch)));
            CUDA_CHECK(cudaMalloc(&matchBuffer_.matchCount, sizeof(int)));

            totalAllocated_ += numMatches * sizeof(GpuMatch) + sizeof(int);
            peakAllocated_ = std::max(peakAllocated_, totalAllocated_);
        }

        GpuKeyPointSoA &GpuMemoryPool::getKeypointBuffer(int level)
        {
            if (level < 0 || level >= static_cast<int>(keypointBuffers_.size()))
            {
                throw std::out_of_range("Keypoint buffer level out of range");
            }
            return keypointBuffers_[level];
        }

        void GpuMemoryPool::resetKeypointBuffer(int level)
        {
            if (level >= 0 && level < static_cast<int>(keypointBuffers_.size()))
            {
                keypointBuffers_[level].count = 0;
            }
        }

        GpuDescriptorArray &GpuMemoryPool::getDescriptorBuffer()
        {
            return descriptorBuffer_;
        }

        void GpuMemoryPool::resetDescriptorBuffer()
        {
            descriptorBuffer_.count = 0;
        }

        GpuImagePyramid &GpuMemoryPool::getPyramidBuffer()
        {
            return pyramidBuffer_;
        }

        void GpuMemoryPool::ensurePyramidLevelSize(int level, int width, int height)
        {
            if (level < 0 || level >= pyramidBuffer_.numLevels)
                return;

            GpuPyramidLevel &lvl = pyramidBuffer_.levels[level];
            int paddedWidth = width + 2 * EDGE_THRESHOLD;
            int paddedHeight = height + 2 * EDGE_THRESHOLD;

            if (lvl.image.cols < paddedWidth || lvl.image.rows < paddedHeight)
            {
                // Need to reallocate
                size_t oldSize = lvl.image.cols * lvl.image.rows * 2;
                lvl.image.create(paddedHeight, paddedWidth, CV_8UC1);
                lvl.blurred.create(paddedHeight, paddedWidth, CV_8UC1);
                size_t newSize = paddedWidth * paddedHeight * 2;

                totalAllocated_ = totalAllocated_ - oldSize + newSize;
                peakAllocated_ = std::max(peakAllocated_, totalAllocated_);
            }

            lvl.width = width;
            lvl.height = height;
        }

        GpuFeatureGrid &GpuMemoryPool::getFeatureGrid()
        {
            return featureGrid_;
        }

        void GpuMemoryPool::resetFeatureGrid()
        {
            CUDA_CHECK(cudaMemset(featureGrid_.cellCount, 0, GRID_CELLS * sizeof(int)));
        }

        GpuMatchArray &GpuMemoryPool::getMatchBuffer()
        {
            return matchBuffer_;
        }

        void GpuMemoryPool::resetMatchBuffer()
        {
            CUDA_CHECK(cudaMemset(matchBuffer_.matchCount, 0, sizeof(int)));
        }

        void *GpuMemoryPool::getTempBuffer(size_t sizeBytes, cudaStream_t stream)
        {
            std::lock_guard<std::mutex> lock(tempBufferMutex_);

            // Round up to 256 byte alignment
            sizeBytes = (sizeBytes + 255) & ~255;

            auto it = tempBufferPool_.find(sizeBytes);
            if (it != tempBufferPool_.end() && !it->second.empty())
            {
                void *ptr = it->second.back();
                it->second.pop_back();
                return ptr;
            }

            // Allocate new buffer
            void *ptr;
            CUDA_CHECK(cudaMallocAsync(&ptr, sizeBytes, stream));
            totalAllocated_ += sizeBytes;
            peakAllocated_ = std::max(peakAllocated_, totalAllocated_);

            return ptr;
        }

        void GpuMemoryPool::releaseTempBuffer(void *ptr)
        {
            // Note: In production, you might want to return this to the pool
            // For now, we just free it
            cudaFree(ptr);
        }

        void GpuMemoryPool::printMemoryStats() const
        {
            std::cout << "=== GPU Memory Pool Stats ===" << std::endl;
            std::cout << "Total Allocated: " << (totalAllocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "Peak Allocated: " << (peakAllocated_ / (1024.0 * 1024.0)) << " MB" << std::endl;
            std::cout << "Keypoint Buffers: " << keypointBuffers_.size() << std::endl;
            std::cout << "=============================" << std::endl;
        }

        // ============================================================================
        // GpuFrameManager Implementation
        // ============================================================================

        void GpuFrameManager::initialize(int numFrameBuffers)
        {
            numBuffers_ = numFrameBuffers;
            frameBuffers_.resize(numFrameBuffers);
            currentFrameIdx_ = 0;
        }

        GpuFrameData &GpuFrameManager::getCurrentFrame()
        {
            return frameBuffers_[currentFrameIdx_];
        }

        const GpuFrameData &GpuFrameManager::getPreviousFrame() const
        {
            int prevIdx = (currentFrameIdx_ - 1 + numBuffers_) % numBuffers_;
            return frameBuffers_[prevIdx];
        }

        void GpuFrameManager::swapFrames()
        {
            currentFrameIdx_ = (currentFrameIdx_ + 1) % numBuffers_;
        }

        void GpuFrameManager::resetCurrentFrame()
        {
            GpuFrameData &frame = getCurrentFrame();
            frame.numKeypoints = 0;
            frame.keypoints.count = 0;
            frame.descriptors.count = 0;
        }

    } // namespace cuda
} // namespace ORB_SLAM3
