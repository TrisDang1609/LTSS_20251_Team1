/**
 * validate_gpu_accuracy.cpp
 *
 * GPU accuracy validation benchmark.
 * Exports keypoint and descriptor data to log files for comparison
 * with CPU implementation.
 *
 * Usage: ./validate_gpu_accuracy <video_path> [num_frames] [output_dir]
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "cuda/GpuPipeline.h"
#include "cuda/CudaUtils.h"
#include "cuda/AccuracyLogger.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM3::cuda;

void printUsage(const char *progName)
{
    cout << "Usage: " << progName << " <video_path> [num_frames] [output_dir]" << endl;
    cout << "  video_path  - Path to input video file" << endl;
    cout << "  num_frames  - Number of frames to process (-1 for all, default: 100)" << endl;
    cout << "  output_dir  - Output directory for logs (default: ./validation_results)" << endl;
}

void printGpuDeviceInfo()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "GPU Device: " << prop.name << endl;
    cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "  SMs: " << prop.multiProcessorCount << endl;
    cout << "  Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
}

int main(int argc, char **argv)
{
    cout << "\n======================================" << endl;
    cout << "  GPU Accuracy Validation" << endl;
    cout << "======================================\n"
         << endl;

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    string videoPath = argv[1];
    int maxFrames = (argc > 2) ? atoi(argv[2]) : 100;
    string outputDir = (argc > 3) ? argv[3] : "./validation_results";

    // Create output directory
    std::filesystem::create_directories(outputDir);

    // Print device info
    printGpuDeviceInfo();
    cout << endl;

    // Open video
    VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cerr << "Error: Cannot open video: " << videoPath << endl;
        return 1;
    }

    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    if (maxFrames <= 0 || maxFrames > totalFrames)
    {
        maxFrames = totalFrames;
    }

    cout << "Video Information:" << endl;
    cout << "  Path: " << videoPath << endl;
    cout << "  Resolution: " << width << " x " << height << endl;
    cout << "  Total Frames: " << totalFrames << endl;
    cout << "  FPS: " << fps << endl;
    cout << "  Frames to process: " << maxFrames << endl;
    cout << "  Output: " << outputDir << endl;
    cout << endl;

    // GPU Pipeline parameters (should match CPU for fair comparison)
    const int nFeatures = 1000; // Match CPU for comparison
    const float scaleFactor = 1.2f;
    const int nLevels = 8;
    const int iniThFAST = 20;
    const int minThFAST = 7;

    cout << "ORB Extractor Parameters:" << endl;
    cout << "  Features: " << nFeatures << endl;
    cout << "  Scale Factor: " << scaleFactor << endl;
    cout << "  Levels: " << nLevels << endl;
    cout << "  Initial FAST Threshold: " << iniThFAST << endl;
    cout << "  Minimum FAST Threshold: " << minThFAST << endl;
    cout << endl;

    // Initialize GPU pipeline using factory function
    cout << "Initializing GPU pipeline..." << endl;

    auto pipeline = createGpuPipeline(
        PipelineMode::MONOCULAR,
        width, height,
        nFeatures,
        scaleFactor,
        nLevels);

    if (!pipeline)
    {
        cerr << "Error: Failed to create GPU pipeline" << endl;
        return 1;
    }

    // Set camera intrinsics
    pipeline->setCameraIntrinsics(500.0f, 500.0f, width / 2.0f, height / 2.0f);

    cout << "GPU pipeline initialized!" << endl;

    // Initialize accuracy logger
    ORB_SLAM3::validation::AccuracyLogger logger(outputDir,
                                                 ORB_SLAM3::validation::AccuracyLogger::Mode::GPU);

    cout << "\nProcessing frames..." << endl;

    Mat frame;
    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

    while (frameCount < maxFrames)
    {
        cap >> frame;
        if (frame.empty())
            break;

        double timestamp = frameCount / fps;

        // Process frame through GPU pipeline
        GpuFrameResult result = pipeline->processMonocular(frame, timestamp);

        if (result.success)
        {
            // Download results for logging
            vector<KeyPoint> keypoints;
            Mat descriptors;

            pipeline->downloadKeypoints(result, keypoints);
            pipeline->downloadDescriptors(result, descriptors);

            // Log to accuracy file
            logger.logFrame(frameCount, keypoints, descriptors, frame.size());
        }

        frameCount++;

        if (frameCount % 10 == 0 || frameCount == maxFrames)
        {
            cout << "  Processed " << frameCount << " / " << maxFrames << " frames" << endl;
        }
    }

    auto endTime = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();

    // Shutdown pipeline
    pipeline->shutdown();

    cout << "\n======================================" << endl;
    cout << "  GPU Validation Complete" << endl;
    cout << "======================================" << endl;
    cout << "  Frames processed: " << frameCount << endl;
    cout << "  Total keypoints: " << logger.getTotalKeypoints() << endl;
    cout << "  Avg keypoints/frame: " << logger.getTotalKeypoints() / frameCount << endl;
    cout << "  Processing time: " << fixed << setprecision(2) << totalTime << " s" << endl;
    cout << "  Output file: " << outputDir << "/gpu_accuracy.log" << endl;
    cout << "======================================\n"
         << endl;

    // Write summary JSON
    string jsonPath = outputDir + "/gpu_summary.json";
    ofstream jsonFile(jsonPath);
    jsonFile << "{\n";
    jsonFile << "  \"mode\": \"GPU\",\n";
    jsonFile << "  \"video\": \"" << videoPath << "\",\n";
    jsonFile << "  \"frames_processed\": " << frameCount << ",\n";
    jsonFile << "  \"total_keypoints\": " << logger.getTotalKeypoints() << ",\n";
    jsonFile << "  \"avg_keypoints_per_frame\": " << logger.getTotalKeypoints() / frameCount << ",\n";
    jsonFile << "  \"nFeatures\": " << nFeatures << ",\n";
    jsonFile << "  \"scaleFactor\": " << scaleFactor << ",\n";
    jsonFile << "  \"nLevels\": " << nLevels << ",\n";
    jsonFile << "  \"iniThFAST\": " << iniThFAST << ",\n";
    jsonFile << "  \"minThFAST\": " << minThFAST << "\n";
    jsonFile << "}\n";
    jsonFile.close();

    return 0;
}
