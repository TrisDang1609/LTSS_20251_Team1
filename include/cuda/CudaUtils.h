/**
 * CUDA Utilities for ORB-SLAM3 GPU Pipeline
 *
 * This file provides common CUDA utilities, error checking macros,
 * stream management, and device query functions for the GPU-accelerated
 * ORB-SLAM3 pipeline.
 *
 * Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
 * CUDA: 13.0, C++17
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

// Undefine macros from sys/sysmacros.h that conflict with cudaDeviceProp members
#ifdef major
#undef major
#endif
#ifdef minor
#undef minor
#endif

namespace ORB_SLAM3
{
    namespace cuda
    {

        // ============================================================================
        // CUDA Error Checking Macros
        // ============================================================================

#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "             \
                      << cudaGetErrorString(err) << std::endl;                               \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        }                                                                                    \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define CUDA_SYNC_CHECK()                    \
    do                                       \
    {                                        \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaGetLastError());      \
    } while (0)

// Debug mode - synchronize after every kernel
#ifdef CUDA_DEBUG
#define CUDA_DEBUG_SYNC() CUDA_SYNC_CHECK()
#else
#define CUDA_DEBUG_SYNC()
#endif

        // ============================================================================
        // GPU Constants and Tuning Parameters (RTX 4060 Ada Lovelace)
        // ============================================================================

        // RTX 4060 Specifications
        constexpr int GPU_SM_COUNT = 24;      // Ada Lovelace SM count
        constexpr int GPU_CORES_PER_SM = 128; // CUDA cores per SM
        constexpr int GPU_WARP_SIZE = 32;
        constexpr int GPU_MAX_THREADS_PER_BLOCK = 1024;
        constexpr int GPU_MAX_BLOCKS_PER_SM = 16;
        constexpr int GPU_MAX_SHARED_MEMORY_PER_BLOCK = 49152; // 48 KB
        constexpr int GPU_L2_CACHE_SIZE = 32 * 1024 * 1024;    // 32 MB L2 cache

        // Optimal block sizes for different kernels
        constexpr int BLOCK_SIZE_1D = 256;
        constexpr int BLOCK_SIZE_2D_X = 16;
        constexpr int BLOCK_SIZE_2D_Y = 16;
        constexpr int BLOCK_SIZE_DESCRIPTOR = 32; // One warp for descriptor operations

        // ORB-SLAM3 specific constants
        constexpr int ORB_DESCRIPTOR_SIZE = 32; // 256 bits = 32 bytes
        constexpr int ORB_DESCRIPTOR_BITS = 256;
        constexpr int PATCH_SIZE = 31;
        constexpr int HALF_PATCH_SIZE = 15;
        constexpr int EDGE_THRESHOLD = 19;
        constexpr int MAX_KEYPOINTS_PER_FRAME = 30000; // Large buffer for all pyramid levels
        constexpr int MAX_PYRAMID_LEVELS = 8;

        // ============================================================================
        // CUDA Stream Manager
        // ============================================================================

        class CudaStreamManager
        {
        public:
            static CudaStreamManager &getInstance()
            {
                static CudaStreamManager instance;
                return instance;
            }

            // Get streams for different pipeline stages
            cudaStream_t getPreprocessStream() const { return preprocessStream_; }
            cudaStream_t getPyramidStream() const { return pyramidStream_; }
            cudaStream_t getFastStream() const { return fastStream_; }
            cudaStream_t getDescriptorStream() const { return descriptorStream_; }
            cudaStream_t getMatchingStream() const { return matchingStream_; }
            cudaStream_t getDefaultStream() const { return defaultStream_; }

            // Synchronize all streams
            void synchronizeAll()
            {
                CUDA_CHECK(cudaStreamSynchronize(preprocessStream_));
                CUDA_CHECK(cudaStreamSynchronize(pyramidStream_));
                CUDA_CHECK(cudaStreamSynchronize(fastStream_));
                CUDA_CHECK(cudaStreamSynchronize(descriptorStream_));
                CUDA_CHECK(cudaStreamSynchronize(matchingStream_));
                CUDA_CHECK(cudaStreamSynchronize(defaultStream_));
            }

            // Record event on stream
            void recordEvent(cudaEvent_t event, cudaStream_t stream)
            {
                CUDA_CHECK(cudaEventRecord(event, stream));
            }

            // Wait for event on another stream
            void waitEvent(cudaStream_t stream, cudaEvent_t event)
            {
                CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
            }

        private:
            CudaStreamManager()
            {
                CUDA_CHECK(cudaStreamCreateWithFlags(&preprocessStream_, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&pyramidStream_, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&fastStream_, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&descriptorStream_, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&matchingStream_, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&defaultStream_, cudaStreamNonBlocking));
            }

            ~CudaStreamManager()
            {
                cudaStreamDestroy(preprocessStream_);
                cudaStreamDestroy(pyramidStream_);
                cudaStreamDestroy(fastStream_);
                cudaStreamDestroy(descriptorStream_);
                cudaStreamDestroy(matchingStream_);
                cudaStreamDestroy(defaultStream_);
            }

            CudaStreamManager(const CudaStreamManager &) = delete;
            CudaStreamManager &operator=(const CudaStreamManager &) = delete;

            cudaStream_t preprocessStream_;
            cudaStream_t pyramidStream_;
            cudaStream_t fastStream_;
            cudaStream_t descriptorStream_;
            cudaStream_t matchingStream_;
            cudaStream_t defaultStream_;
        };

        // ============================================================================
        // CUDA Event Pool for Performance Profiling
        // ============================================================================

        class CudaEventPool
        {
        public:
            static CudaEventPool &getInstance()
            {
                static CudaEventPool instance;
                return instance;
            }

            cudaEvent_t getEvent()
            {
                cudaEvent_t event;
                CUDA_CHECK(cudaEventCreate(&event));
                events_.push_back(event);
                return event;
            }

            void recordAndGetElapsed(cudaEvent_t start, cudaEvent_t stop,
                                     cudaStream_t stream, float &elapsed_ms)
            {
                CUDA_CHECK(cudaEventRecord(stop, stream));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            }

        private:
            CudaEventPool() = default;

            ~CudaEventPool()
            {
                for (auto &event : events_)
                {
                    cudaEventDestroy(event);
                }
            }

            std::vector<cudaEvent_t> events_;
        };

        // ============================================================================
        // Device Info Query
        // ============================================================================

        struct DeviceInfo
        {
            std::string name;
            int computeMajor;
            int computeMinor;
            size_t totalGlobalMem;
            size_t sharedMemPerBlock;
            int maxThreadsPerBlock;
            int maxThreadsDimX;
            int maxThreadsDimY;
            int maxThreadsDimZ;
            int maxGridSizeX;
            int maxGridSizeY;
            int maxGridSizeZ;
            int multiProcessorCount;
            int warpSize;
            int l2CacheSize;
            bool unifiedAddressing;
            bool concurrentKernels;
        };

        inline DeviceInfo getDeviceInfo(int deviceId = 0)
        {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

            DeviceInfo info;
            info.name = prop.name;
            info.computeMajor = prop.major;
            info.computeMinor = prop.minor;
            info.totalGlobalMem = prop.totalGlobalMem;
            info.sharedMemPerBlock = prop.sharedMemPerBlock;
            info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
            info.maxThreadsDimX = prop.maxThreadsDim[0];
            info.maxThreadsDimY = prop.maxThreadsDim[1];
            info.maxThreadsDimZ = prop.maxThreadsDim[2];
            info.maxGridSizeX = prop.maxGridSize[0];
            info.maxGridSizeY = prop.maxGridSize[1];
            info.maxGridSizeZ = prop.maxGridSize[2];
            info.multiProcessorCount = prop.multiProcessorCount;
            info.warpSize = prop.warpSize;
            info.l2CacheSize = prop.l2CacheSize;
            info.unifiedAddressing = prop.unifiedAddressing;
            info.concurrentKernels = prop.concurrentKernels;

            return info;
        }

        inline void printDeviceInfo(int deviceId = 0)
        {
            DeviceInfo info = getDeviceInfo(deviceId);
            std::cout << "=== CUDA Device Info ===" << std::endl;
            std::cout << "Name: " << info.name << std::endl;
            std::cout << "Compute Capability: " << info.computeMajor << "." << info.computeMinor << std::endl;
            std::cout << "Total Global Memory: " << info.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "Shared Memory per Block: " << info.sharedMemPerBlock / 1024 << " KB" << std::endl;
            std::cout << "Max Threads per Block: " << info.maxThreadsPerBlock << std::endl;
            std::cout << "Multiprocessors: " << info.multiProcessorCount << std::endl;
            std::cout << "L2 Cache Size: " << info.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
            std::cout << "=========================" << std::endl;
        }

        // ============================================================================
        // Utility Functions
        // ============================================================================

        // Calculate grid size for 1D kernel launch
        inline dim3 calcGrid1D(int totalThreads, int blockSize = BLOCK_SIZE_1D)
        {
            return dim3((totalThreads + blockSize - 1) / blockSize);
        }

        // Calculate grid size for 2D kernel launch
        inline dim3 calcGrid2D(int width, int height,
                               int blockX = BLOCK_SIZE_2D_X,
                               int blockY = BLOCK_SIZE_2D_Y)
        {
            return dim3((width + blockX - 1) / blockX,
                        (height + blockY - 1) / blockY);
        }

        // Round up to nearest multiple
        template <typename T>
        __host__ __device__ inline T roundUp(T value, T multiple)
        {
            return ((value + multiple - 1) / multiple) * multiple;
        }

        // Ceiling division
        template <typename T>
        __host__ __device__ inline T ceilDiv(T a, T b)
        {
            return (a + b - 1) / b;
        }

    } // namespace cuda
} // namespace ORB_SLAM3

#endif // CUDA_UTILS_H
