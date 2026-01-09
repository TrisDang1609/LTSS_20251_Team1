/**
 * GPU Memory Manager for ORB-SLAM3 CUDA Pipeline
 *
 * This header provides a unified GPU memory pool with stream-ordered allocations
 * to minimize allocation overhead and achieve zero host-device round-trips.
 *
 * Features:
 * 1. Pre-allocated memory pools for each data type
 * 2. Stream-ordered memory operations (CUDA 11.2+)
 * 3. Memory reuse across frames
 * 4. Device-resident data lifecycle management
 */

#ifndef CUDA_MEMORY_MANAGER_H
#define CUDA_MEMORY_MANAGER_H

#include "CudaUtils.h"
#include "GpuTypes.h"
#include <opencv2/core/cuda.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // Memory Pool Configuration
        // ============================================================================

        struct MemoryPoolConfig
        {
            // Maximum allocations per type
            size_t maxKeypoints = 10000;
            size_t maxDescriptors = 10000;
            size_t maxMatches = 50000;
            size_t maxPyramidLevels = 8;

            // Image dimensions (for pre-allocation)
            int maxImageWidth = 1920;
            int maxImageHeight = 1080;

            // Buffer multiplier for safety margin
            float bufferMultiplier = 1.5f;
        };

        // ============================================================================
        // GPU Memory Pool
        // ============================================================================

        class GpuMemoryPool
        {
        public:
            static GpuMemoryPool &getInstance()
            {
                static GpuMemoryPool instance;
                return instance;
            }

            // Initialize with configuration
            void initialize(const MemoryPoolConfig &config);

            // Release all GPU memory
            void release();

            // Check if initialized
            bool isInitialized() const { return initialized_; }

            // ========================================================================
            // Keypoint Memory Management
            // ========================================================================

            // Get pre-allocated keypoint SoA
            GpuKeyPointSoA &getKeypointBuffer(int level = 0);

            // Reset keypoint count for reuse
            void resetKeypointBuffer(int level = 0);

            // ========================================================================
            // Descriptor Memory Management
            // ========================================================================

            // Get pre-allocated descriptor array
            GpuDescriptorArray &getDescriptorBuffer();

            // Reset descriptor count
            void resetDescriptorBuffer();

            // ========================================================================
            // Image Pyramid Memory Management
            // ========================================================================

            // Get pre-allocated pyramid
            GpuImagePyramid &getPyramidBuffer();

            // Resize pyramid level if needed (uses internal reallocation)
            void ensurePyramidLevelSize(int level, int width, int height);

            // ========================================================================
            // Feature Grid Memory Management
            // ========================================================================

            // Get pre-allocated feature grid
            GpuFeatureGrid &getFeatureGrid();

            // Reset grid for new frame
            void resetFeatureGrid();

            // ========================================================================
            // Match Buffer Management
            // ========================================================================

            // Get pre-allocated match buffer
            GpuMatchArray &getMatchBuffer();

            // Reset match count
            void resetMatchBuffer();

            // ========================================================================
            // Temporary Buffer Management
            // ========================================================================

            // Get temporary device buffer of specified size
            void *getTempBuffer(size_t sizeBytes, cudaStream_t stream = 0);

            // Release temporary buffer
            void releaseTempBuffer(void *ptr);

            // ========================================================================
            // Memory Statistics
            // ========================================================================

            size_t getTotalAllocatedBytes() const { return totalAllocated_; }
            size_t getPeakAllocatedBytes() const { return peakAllocated_; }

            void printMemoryStats() const;

        private:
            GpuMemoryPool() = default;
            ~GpuMemoryPool() { release(); }

            GpuMemoryPool(const GpuMemoryPool &) = delete;
            GpuMemoryPool &operator=(const GpuMemoryPool &) = delete;

            // Internal allocation helpers
            void allocateKeypointBuffers();
            void allocateDescriptorBuffer();
            void allocatePyramidBuffer();
            void allocateFeatureGrid();
            void allocateMatchBuffer();

            // Configuration
            MemoryPoolConfig config_;
            bool initialized_ = false;

            // Memory pools
            std::vector<GpuKeyPointSoA> keypointBuffers_; // One per pyramid level
            GpuDescriptorArray descriptorBuffer_;
            GpuImagePyramid pyramidBuffer_;
            GpuFeatureGrid featureGrid_;
            GpuMatchArray matchBuffer_;

            // Temporary buffer pool
            std::unordered_map<size_t, std::vector<void *>> tempBufferPool_;
            std::mutex tempBufferMutex_;

            // Statistics
            size_t totalAllocated_ = 0;
            size_t peakAllocated_ = 0;
        };

        // ============================================================================
        // GPU Frame Manager - Manages complete frame data lifecycle
        // ============================================================================

        class GpuFrameManager
        {
        public:
            static GpuFrameManager &getInstance()
            {
                static GpuFrameManager instance;
                return instance;
            }

            // Initialize frame buffers
            void initialize(int numFrameBuffers = 2);

            // Get current frame buffer (for writing)
            GpuFrameData &getCurrentFrame();

            // Get previous frame buffer (for reading)
            const GpuFrameData &getPreviousFrame() const;

            // Swap frame buffers
            void swapFrames();

            // Reset current frame for new processing
            void resetCurrentFrame();

        private:
            GpuFrameManager() = default;
            ~GpuFrameManager() = default;

            std::vector<GpuFrameData> frameBuffers_;
            int currentFrameIdx_ = 0;
            int numBuffers_ = 0;
        };

        // ============================================================================
        // RAII Wrapper for GPU Memory
        // ============================================================================

        template <typename T>
        class GpuArray
        {
        public:
            GpuArray() : data_(nullptr), size_(0), capacity_(0) {}

            explicit GpuArray(size_t size) : size_(size), capacity_(size)
            {
                CUDA_CHECK(cudaMalloc(&data_, size * sizeof(T)));
            }

            ~GpuArray()
            {
                if (data_)
                {
                    cudaFree(data_);
                }
            }

            // Move semantics
            GpuArray(GpuArray &&other) noexcept
                : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
            {
                other.data_ = nullptr;
                other.size_ = 0;
                other.capacity_ = 0;
            }

            GpuArray &operator=(GpuArray &&other) noexcept
            {
                if (this != &other)
                {
                    if (data_)
                        cudaFree(data_);
                    data_ = other.data_;
                    size_ = other.size_;
                    capacity_ = other.capacity_;
                    other.data_ = nullptr;
                    other.size_ = 0;
                    other.capacity_ = 0;
                }
                return *this;
            }

            // No copying
            GpuArray(const GpuArray &) = delete;
            GpuArray &operator=(const GpuArray &) = delete;

            // Resize (reallocate if needed)
            void resize(size_t newSize)
            {
                if (newSize > capacity_)
                {
                    T *newData;
                    CUDA_CHECK(cudaMalloc(&newData, newSize * sizeof(T)));
                    if (data_ && size_ > 0)
                    {
                        CUDA_CHECK(cudaMemcpy(newData, data_, size_ * sizeof(T),
                                              cudaMemcpyDeviceToDevice));
                    }
                    if (data_)
                        cudaFree(data_);
                    data_ = newData;
                    capacity_ = newSize;
                }
                size_ = newSize;
            }

            // Reserve capacity without changing size
            void reserve(size_t newCapacity)
            {
                if (newCapacity > capacity_)
                {
                    T *newData;
                    CUDA_CHECK(cudaMalloc(&newData, newCapacity * sizeof(T)));
                    if (data_ && size_ > 0)
                    {
                        CUDA_CHECK(cudaMemcpy(newData, data_, size_ * sizeof(T),
                                              cudaMemcpyDeviceToDevice));
                    }
                    if (data_)
                        cudaFree(data_);
                    data_ = newData;
                    capacity_ = newCapacity;
                }
            }

            // Clear without deallocation
            void clear() { size_ = 0; }

            // Accessors
            T *data() { return data_; }
            const T *data() const { return data_; }
            size_t size() const { return size_; }
            size_t capacity() const { return capacity_; }
            bool empty() const { return size_ == 0; }

            // Async copy from host
            void copyFromHostAsync(const T *hostData, size_t count, cudaStream_t stream)
            {
                if (count > capacity_)
                    resize(count);
                size_ = count;
                CUDA_CHECK(cudaMemcpyAsync(data_, hostData, count * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));
            }

            // Async copy to host
            void copyToHostAsync(T *hostData, cudaStream_t stream) const
            {
                CUDA_CHECK(cudaMemcpyAsync(hostData, data_, size_ * sizeof(T),
                                           cudaMemcpyDeviceToHost, stream));
            }

            // Synchronous copy from host
            void copyFromHost(const T *hostData, size_t count)
            {
                if (count > capacity_)
                    resize(count);
                size_ = count;
                CUDA_CHECK(cudaMemcpy(data_, hostData, count * sizeof(T),
                                      cudaMemcpyHostToDevice));
            }

            // Synchronous copy to host
            void copyToHost(T *hostData) const
            {
                CUDA_CHECK(cudaMemcpy(hostData, data_, size_ * sizeof(T),
                                      cudaMemcpyDeviceToHost));
            }

        private:
            T *data_;
            size_t size_;
            size_t capacity_;
        };

        // ============================================================================
        // Pinned Host Memory for Fast Transfers
        // ============================================================================

        template <typename T>
        class PinnedHostArray
        {
        public:
            PinnedHostArray() : data_(nullptr), size_(0), capacity_(0) {}

            explicit PinnedHostArray(size_t size) : size_(size), capacity_(size)
            {
                CUDA_CHECK(cudaMallocHost(&data_, size * sizeof(T)));
            }

            ~PinnedHostArray()
            {
                if (data_)
                {
                    cudaFreeHost(data_);
                }
            }

            // Move semantics
            PinnedHostArray(PinnedHostArray &&other) noexcept
                : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
            {
                other.data_ = nullptr;
                other.size_ = 0;
                other.capacity_ = 0;
            }

            PinnedHostArray &operator=(PinnedHostArray &&other) noexcept
            {
                if (this != &other)
                {
                    if (data_)
                        cudaFreeHost(data_);
                    data_ = other.data_;
                    size_ = other.size_;
                    capacity_ = other.capacity_;
                    other.data_ = nullptr;
                    other.size_ = 0;
                    other.capacity_ = 0;
                }
                return *this;
            }

            // No copying
            PinnedHostArray(const PinnedHostArray &) = delete;
            PinnedHostArray &operator=(const PinnedHostArray &) = delete;

            void resize(size_t newSize)
            {
                if (newSize > capacity_)
                {
                    T *newData;
                    CUDA_CHECK(cudaMallocHost(&newData, newSize * sizeof(T)));
                    if (data_ && size_ > 0)
                    {
                        memcpy(newData, data_, size_ * sizeof(T));
                    }
                    if (data_)
                        cudaFreeHost(data_);
                    data_ = newData;
                    capacity_ = newSize;
                }
                size_ = newSize;
            }

            T *data() { return data_; }
            const T *data() const { return data_; }
            size_t size() const { return size_; }
            T &operator[](size_t idx) { return data_[idx]; }
            const T &operator[](size_t idx) const { return data_[idx]; }

        private:
            T *data_;
            size_t size_;
            size_t capacity_;
        };

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // CUDA_MEMORY_MANAGER_H
