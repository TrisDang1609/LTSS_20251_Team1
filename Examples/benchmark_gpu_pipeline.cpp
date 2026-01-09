/**
 * ORB-SLAM3 GPU Pipeline Benchmark
 *
 * Benchmark application to validate and measure GPU pipeline performance.
 * Processes video_benchmark/flycam.mp4 and reports timing statistics.
 *
 * Usage: ./benchmark_gpu_pipeline [video_path] [num_frames]
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "cuda/GpuPipeline.h"
#include "cuda/CudaUtils.h"

using namespace ORB_SLAM3::cuda;

/**
 * Print GPU device information
 */
void printDeviceInfo()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  GPU Device Information" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "========================================\n"
              << std::endl;
}

/**
 * Benchmark statistics
 */
struct BenchmarkStats
{
    std::vector<float> preprocessTimes;
    std::vector<float> extractTimes;
    std::vector<float> gridTimes;
    std::vector<float> totalTimes;
    std::vector<int> keypointCounts;

    int totalFrames = 0;

    void addSample(const GpuPipelineStats &stats)
    {
        preprocessTimes.push_back(stats.preprocessTimeMs);
        extractTimes.push_back(stats.detectionTimeMs + stats.descriptionTimeMs);
        gridTimes.push_back(stats.gridTimeMs);
        totalTimes.push_back(stats.totalTimeMs);
        keypointCounts.push_back(stats.numKeypoints);
        totalFrames++;
    }

    float mean(const std::vector<float> &v) const
    {
        if (v.empty())
            return 0;
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    }

    float stddev(const std::vector<float> &v) const
    {
        if (v.size() < 2)
            return 0;
        float m = mean(v);
        float sq_sum = 0;
        for (float x : v)
            sq_sum += (x - m) * (x - m);
        return std::sqrt(sq_sum / (v.size() - 1));
    }

    float percentile(std::vector<float> v, float p) const
    {
        if (v.empty())
            return 0;
        std::sort(v.begin(), v.end());
        int idx = static_cast<int>(p * (v.size() - 1));
        return v[idx];
    }

    void print() const
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Benchmark Results (" << totalFrames << " frames)" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "\n  Preprocessing:" << std::endl;
        std::cout << "    Mean: " << mean(preprocessTimes) << " ms" << std::endl;
        std::cout << "    Std:  " << stddev(preprocessTimes) << " ms" << std::endl;

        std::cout << "\n  Feature Extraction (FAST + ORB):" << std::endl;
        std::cout << "    Mean: " << mean(extractTimes) << " ms" << std::endl;
        std::cout << "    Std:  " << stddev(extractTimes) << " ms" << std::endl;

        std::cout << "\n  Grid Building:" << std::endl;
        std::cout << "    Mean: " << mean(gridTimes) << " ms" << std::endl;
        std::cout << "    Std:  " << stddev(gridTimes) << " ms" << std::endl;

        std::cout << "\n  Total Pipeline:" << std::endl;
        std::cout << "    Mean: " << mean(totalTimes) << " ms" << std::endl;
        std::cout << "    Std:  " << stddev(totalTimes) << " ms" << std::endl;
        std::cout << "    P50:  " << percentile(totalTimes, 0.50f) << " ms" << std::endl;
        std::cout << "    P95:  " << percentile(totalTimes, 0.95f) << " ms" << std::endl;
        std::cout << "    P99:  " << percentile(totalTimes, 0.99f) << " ms" << std::endl;

        float avgFps = 1000.0f / mean(totalTimes);
        std::cout << "\n  Throughput: " << avgFps << " FPS" << std::endl;

        std::cout << "\n  Keypoints:" << std::endl;
        float avgKp = std::accumulate(keypointCounts.begin(), keypointCounts.end(), 0.0f) / keypointCounts.size();
        std::cout << "    Average per frame: " << static_cast<int>(avgKp) << std::endl;

        std::cout << "========================================\n"
                  << std::endl;
    }
};

/**
 * Run benchmark on video file
 */
int runBenchmark(const std::string &videoPath, int maxFrames = -1)
{
    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video: " << videoPath << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "\nVideo Information:" << std::endl;
    std::cout << "  Path: " << videoPath << std::endl;
    std::cout << "  Resolution: " << width << " x " << height << std::endl;
    std::cout << "  Total Frames: " << totalFrames << std::endl;
    std::cout << "  FPS: " << fps << std::endl;

    if (maxFrames > 0 && maxFrames < totalFrames)
    {
        totalFrames = maxFrames;
        std::cout << "  Processing: " << totalFrames << " frames" << std::endl;
    }

    // Create GPU pipeline
    std::cout << "\nInitializing GPU pipeline..." << std::endl;

    auto pipeline = createGpuPipeline(
        PipelineMode::MONOCULAR,
        width,
        height,
        1250, // numFeatures
        1.2f, // scaleFactor
        8     // numLevels
    );

    if (!pipeline)
    {
        std::cerr << "Error: Failed to create GPU pipeline" << std::endl;
        return -1;
    }

    // Set camera intrinsics (default, adjust for your camera)
    float fx = 500.0f;
    float fy = 500.0f;
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    pipeline->setCameraIntrinsics(fx, fy, cx, cy);

    std::cout << "GPU pipeline initialized successfully!" << std::endl;

    // Warmup
    std::cout << "\nWarming up GPU..." << std::endl;
    cv::Mat warmupFrame;
    for (int i = 0; i < 10; ++i)
    {
        cap >> warmupFrame;
        if (warmupFrame.empty())
            break;
        pipeline->processMonocular(warmupFrame, i * 0.033);
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    // Benchmark
    std::cout << "\nRunning benchmark..." << std::endl;

    BenchmarkStats stats;
    cv::Mat frame;
    int frameIdx = 0;

    auto startTime = std::chrono::high_resolution_clock::now();

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        if (maxFrames > 0 && frameIdx >= maxFrames)
            break;

        double timestamp = frameIdx / fps;

        // Process frame on GPU
        GpuFrameResult result = pipeline->processMonocular(frame, timestamp);

        if (result.success)
        {
            stats.addSample(result.stats);
        }

        frameIdx++;

        // Progress
        if (frameIdx % 100 == 0)
        {
            std::cout << "  Processed " << frameIdx << " / " << totalFrames << " frames" << std::endl;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\nBenchmark completed in " << duration.count() << " ms" << std::endl;

    // Print results
    stats.print();

    // Cleanup
    pipeline->shutdown();
    cap.release();

    return 0;
}

/**
 * Test keypoint download functionality
 */
void testKeypointDownload(GpuPipeline &pipeline)
{
    std::cout << "\nTesting keypoint download..." << std::endl;

    // Create test image
    cv::Mat testImage(480, 640, CV_8UC1);
    cv::randu(testImage, 0, 255);

    // Add some features
    cv::circle(testImage, cv::Point(320, 240), 10, cv::Scalar(255), -1);
    cv::rectangle(testImage, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(0), -1);

    // Process
    GpuFrameResult result = pipeline.processMonocular(testImage, 0.0);

    if (!result.success)
    {
        std::cerr << "  Processing failed!" << std::endl;
        return;
    }

    std::cout << "  Detected " << result.numDescriptors << " keypoints on GPU" << std::endl;

    // Download to CPU
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    pipeline.downloadKeypoints(result, keypoints);
    pipeline.downloadDescriptors(result, descriptors);

    std::cout << "  Downloaded " << keypoints.size() << " keypoints to CPU" << std::endl;
    std::cout << "  Descriptors: " << descriptors.rows << " x " << descriptors.cols << std::endl;

    if (!keypoints.empty())
    {
        std::cout << "  First keypoint: (" << keypoints[0].pt.x << ", " << keypoints[0].pt.y
                  << ") size=" << keypoints[0].size << " angle=" << keypoints[0].angle << std::endl;
    }
}

/**
 * Main entry point
 */
int main(int argc, char **argv)
{
    std::cout << "\n======================================" << std::endl;
    std::cout << "  ORB-SLAM3 GPU Pipeline Benchmark" << std::endl;
    std::cout << "======================================" << std::endl;

    // Print device info
    ORB_SLAM3::cuda::printDeviceInfo();

    // Parse arguments
    std::string videoPath = "../video_benchmark/flycam.mp4";
    int maxFrames = -1;

    if (argc >= 2)
    {
        videoPath = argv[1];
    }
    if (argc >= 3)
    {
        maxFrames = std::atoi(argv[2]);
    }

    // Run benchmark
    int result = runBenchmark(videoPath, maxFrames);

    return result;
}
