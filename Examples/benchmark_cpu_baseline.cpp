/**
 * CPU Baseline Benchmark for ORB-SLAM3
 *
 * This executable runs the original CPU-only ORB feature extraction pipeline
 * on the same video input to establish baseline performance metrics for
 * comparison with the GPU-accelerated version.
 *
 * Metrics logged:
 * - Frame preprocessing time
 * - Feature extraction time (FAST + ORB descriptors)
 * - Total pipeline time
 * - Throughput (FPS)
 * - Keypoint statistics
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// Include the original CPU ORB extractor
#include "ORBextractor.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

// Timing utilities
class Timer
{
public:
    void start()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stopMs()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Statistics calculator
struct BenchmarkStats
{
    double mean;
    double stddev;
    double min;
    double max;
    double p50;
    double p95;
    double p99;

    static BenchmarkStats compute(std::vector<double> &data)
    {
        BenchmarkStats stats;
        if (data.empty())
        {
            stats.mean = stats.stddev = stats.min = stats.max = 0;
            stats.p50 = stats.p95 = stats.p99 = 0;
            return stats;
        }

        // Sort for percentiles
        std::sort(data.begin(), data.end());

        // Mean
        stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

        // Stddev
        double sq_sum = 0;
        for (double v : data)
        {
            sq_sum += (v - stats.mean) * (v - stats.mean);
        }
        stats.stddev = std::sqrt(sq_sum / data.size());

        // Min/Max
        stats.min = data.front();
        stats.max = data.back();

        // Percentiles
        stats.p50 = data[data.size() * 50 / 100];
        stats.p95 = data[data.size() * 95 / 100];
        stats.p99 = data[data.size() * 99 / 100];

        return stats;
    }
};

// Result structure for JSON export
struct BenchmarkResult
{
    std::string version; // "CPU" or "GPU"
    int totalFrames;
    double totalTimeMs;
    BenchmarkStats preprocessStats;
    BenchmarkStats extractionStats;
    BenchmarkStats totalPipelineStats;
    double avgKeypoints;
    double throughputFps;
};

void printResults(const BenchmarkResult &result)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "  " << result.version << " Benchmark Results (" << result.totalFrames << " frames)" << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "  Preprocessing:" << std::endl;
    std::cout << "    Mean: " << result.preprocessStats.mean << " ms" << std::endl;
    std::cout << "    Std:  " << result.preprocessStats.stddev << " ms" << std::endl;

    std::cout << "\n  Feature Extraction (FAST + ORB):" << std::endl;
    std::cout << "    Mean: " << result.extractionStats.mean << " ms" << std::endl;
    std::cout << "    Std:  " << result.extractionStats.stddev << " ms" << std::endl;

    std::cout << "\n  Total Pipeline:" << std::endl;
    std::cout << "    Mean: " << result.totalPipelineStats.mean << " ms" << std::endl;
    std::cout << "    Std:  " << result.totalPipelineStats.stddev << " ms" << std::endl;
    std::cout << "    P50:  " << result.totalPipelineStats.p50 << " ms" << std::endl;
    std::cout << "    P95:  " << result.totalPipelineStats.p95 << " ms" << std::endl;
    std::cout << "    P99:  " << result.totalPipelineStats.p99 << " ms" << std::endl;

    std::cout << "\n  Throughput: " << result.throughputFps << " FPS" << std::endl;

    std::cout << "\n  Keypoints:" << std::endl;
    std::cout << "    Average per frame: " << static_cast<int>(result.avgKeypoints) << std::endl;
    std::cout << "========================================\n"
              << std::endl;
}

void saveResultsJson(const BenchmarkResult &result, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not open " << filename << " for writing" << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(4);
    file << "{\n";
    file << "  \"version\": \"" << result.version << "\",\n";
    file << "  \"totalFrames\": " << result.totalFrames << ",\n";
    file << "  \"totalTimeMs\": " << result.totalTimeMs << ",\n";
    file << "  \"preprocessing\": {\n";
    file << "    \"meanMs\": " << result.preprocessStats.mean << ",\n";
    file << "    \"stdMs\": " << result.preprocessStats.stddev << ",\n";
    file << "    \"minMs\": " << result.preprocessStats.min << ",\n";
    file << "    \"maxMs\": " << result.preprocessStats.max << "\n";
    file << "  },\n";
    file << "  \"featureExtraction\": {\n";
    file << "    \"meanMs\": " << result.extractionStats.mean << ",\n";
    file << "    \"stdMs\": " << result.extractionStats.stddev << ",\n";
    file << "    \"minMs\": " << result.extractionStats.min << ",\n";
    file << "    \"maxMs\": " << result.extractionStats.max << "\n";
    file << "  },\n";
    file << "  \"totalPipeline\": {\n";
    file << "    \"meanMs\": " << result.totalPipelineStats.mean << ",\n";
    file << "    \"stdMs\": " << result.totalPipelineStats.stddev << ",\n";
    file << "    \"p50Ms\": " << result.totalPipelineStats.p50 << ",\n";
    file << "    \"p95Ms\": " << result.totalPipelineStats.p95 << ",\n";
    file << "    \"p99Ms\": " << result.totalPipelineStats.p99 << "\n";
    file << "  },\n";
    file << "  \"avgKeypoints\": " << result.avgKeypoints << ",\n";
    file << "  \"throughputFps\": " << result.throughputFps << "\n";
    file << "}\n";

    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "\n======================================" << std::endl;
    std::cout << "  ORB-SLAM3 CPU Baseline Benchmark" << std::endl;
    std::cout << "======================================" << std::endl;

    // Parse arguments
    std::string videoPath = "../video_benchmark/flycam.mp4";
    std::string outputJson = "benchmark_cpu_result.json";
    int maxFrames = -1; // -1 means all frames

    if (argc >= 2)
    {
        videoPath = argv[1];
    }
    if (argc >= 3)
    {
        maxFrames = std::stoi(argv[2]);
    }
    if (argc >= 4)
    {
        outputJson = argv[3];
    }

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

    if (maxFrames > 0 && maxFrames < totalFrames)
    {
        totalFrames = maxFrames;
    }

    std::cout << "\nVideo Information:" << std::endl;
    std::cout << "  Path: " << videoPath << std::endl;
    std::cout << "  Resolution: " << width << " x " << height << std::endl;
    std::cout << "  Total Frames: " << totalFrames << std::endl;
    std::cout << "  FPS: " << fps << std::endl;

    // ORB Extractor parameters (matching ORB-SLAM3 defaults)
    int nFeatures = 1000;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;

    std::cout << "\nORB Extractor Parameters:" << std::endl;
    std::cout << "  Features: " << nFeatures << std::endl;
    std::cout << "  Scale Factor: " << scaleFactor << std::endl;
    std::cout << "  Levels: " << nLevels << std::endl;
    std::cout << "  Initial FAST Threshold: " << iniThFAST << std::endl;
    std::cout << "  Minimum FAST Threshold: " << minThFAST << std::endl;

    // Initialize CPU ORB extractor
    std::cout << "\nInitializing CPU ORB extractor..." << std::endl;
    ORBextractor orbExtractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
    std::cout << "CPU ORB extractor initialized!" << std::endl;

    // Warmup (process a few frames to warm up caches)
    std::cout << "\nWarming up CPU..." << std::endl;
    cv::Mat warmupFrame, warmupGray;
    std::vector<cv::KeyPoint> warmupKps;
    cv::Mat warmupDesc;
    std::vector<int> warmupLapping = {0, 0};

    for (int i = 0; i < 5 && cap.read(warmupFrame); ++i)
    {
        if (warmupFrame.channels() == 3)
        {
            cv::cvtColor(warmupFrame, warmupGray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            warmupGray = warmupFrame;
        }
        orbExtractor(warmupGray, cv::Mat(), warmupKps, warmupDesc, warmupLapping);
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset to beginning

    // Timing vectors
    std::vector<double> preprocessTimes;
    std::vector<double> extractionTimes;
    std::vector<double> totalPipelineTimes;
    std::vector<int> keypointCounts;

    preprocessTimes.reserve(totalFrames);
    extractionTimes.reserve(totalFrames);
    totalPipelineTimes.reserve(totalFrames);
    keypointCounts.reserve(totalFrames);

    // Main benchmark loop
    std::cout << "\nRunning benchmark..." << std::endl;
    Timer timer;
    cv::Mat frame, grayFrame;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> lapping = {0, 0};

    auto benchStart = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    while (cap.read(frame) && (maxFrames < 0 || frameCount < maxFrames))
    {
        double pipelineStart = 0, preprocessEnd = 0, extractionEnd = 0;

        // Total pipeline timer
        timer.start();
        pipelineStart = 0;

        // Preprocessing (color conversion)
        if (frame.channels() == 3)
        {
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        }
        else
        {
            grayFrame = frame.clone();
        }
        preprocessEnd = timer.stopMs();
        preprocessTimes.push_back(preprocessEnd);

        // Feature extraction
        timer.start();
        keypoints.clear();
        orbExtractor(grayFrame, cv::Mat(), keypoints, descriptors, lapping);
        extractionEnd = timer.stopMs();
        extractionTimes.push_back(extractionEnd);

        // Total pipeline time
        double totalTime = preprocessEnd + extractionEnd;
        totalPipelineTimes.push_back(totalTime);

        keypointCounts.push_back(static_cast<int>(keypoints.size()));

        ++frameCount;

        if (frameCount % 100 == 0)
        {
            std::cout << "  Processed " << frameCount << " / " << totalFrames << " frames" << std::endl;
        }
    }

    auto benchEnd = std::chrono::high_resolution_clock::now();
    double totalBenchTimeMs = std::chrono::duration<double, std::milli>(benchEnd - benchStart).count();

    std::cout << "\nBenchmark completed in " << static_cast<int>(totalBenchTimeMs) << " ms" << std::endl;

    // Compute statistics
    BenchmarkResult result;
    result.version = "CPU";
    result.totalFrames = frameCount;
    result.totalTimeMs = totalBenchTimeMs;
    result.preprocessStats = BenchmarkStats::compute(preprocessTimes);
    result.extractionStats = BenchmarkStats::compute(extractionTimes);
    result.totalPipelineStats = BenchmarkStats::compute(totalPipelineTimes);

    // Average keypoints
    double avgKp = std::accumulate(keypointCounts.begin(), keypointCounts.end(), 0.0) / keypointCounts.size();
    result.avgKeypoints = avgKp;

    // Throughput
    result.throughputFps = 1000.0 / result.totalPipelineStats.mean;

    // Print results
    printResults(result);

    // Save to JSON
    saveResultsJson(result, outputJson);

    cap.release();

    return 0;
}
