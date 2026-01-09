/**
 * validate_cpu_accuracy.cpp
 *
 * CPU accuracy validation benchmark.
 * Exports keypoint and descriptor data to log files for comparison
 * with GPU implementation.
 *
 * Usage: ./validate_cpu_accuracy <video_path> [num_frames] [output_dir]
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "ORBextractor.h"
#include "cuda/AccuracyLogger.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

void printUsage(const char *progName)
{
    cout << "Usage: " << progName << " <video_path> [num_frames] [output_dir]" << endl;
    cout << "  video_path  - Path to input video file" << endl;
    cout << "  num_frames  - Number of frames to process (-1 for all, default: 100)" << endl;
    cout << "  output_dir  - Output directory for logs (default: ./validation_results)" << endl;
}

int main(int argc, char **argv)
{
    cout << "\n======================================" << endl;
    cout << "  CPU Accuracy Validation" << endl;
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

    // ORB extractor parameters (must match GPU for fair comparison)
    const int nFeatures = 1000;
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

    // Initialize ORB extractor
    ORBextractor orbExtractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    // Initialize accuracy logger
    validation::AccuracyLogger logger(outputDir, validation::AccuracyLogger::Mode::CPU);

    cout << "Processing frames..." << endl;

    Mat frame, gray;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

    while (frameCount < maxFrames)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Convert to grayscale
        if (frame.channels() == 3)
        {
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        }
        else
        {
            gray = frame.clone();
        }

        // Extract ORB features
        keypoints.clear();
        vector<int> lapping = {0, 0}; // For stereo lapping area, must be initialized
        orbExtractor(gray, cv::Mat(), keypoints, descriptors, lapping);

        // Log to accuracy file
        logger.logFrame(frameCount, keypoints, descriptors, gray.size());

        frameCount++;

        if (frameCount % 10 == 0 || frameCount == maxFrames)
        {
            cout << "  Processed " << frameCount << " / " << maxFrames << " frames" << endl;
        }
    }

    auto endTime = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();

    cout << "\n======================================" << endl;
    cout << "  CPU Validation Complete" << endl;
    cout << "======================================" << endl;
    cout << "  Frames processed: " << frameCount << endl;
    cout << "  Total keypoints: " << logger.getTotalKeypoints() << endl;
    cout << "  Avg keypoints/frame: " << logger.getTotalKeypoints() / frameCount << endl;
    cout << "  Processing time: " << fixed << setprecision(2) << totalTime << " s" << endl;
    cout << "  Output file: " << outputDir << "/cpu_accuracy.log" << endl;
    cout << "======================================\n"
         << endl;

    // Write summary JSON
    string jsonPath = outputDir + "/cpu_summary.json";
    ofstream jsonFile(jsonPath);
    jsonFile << "{\n";
    jsonFile << "  \"mode\": \"CPU\",\n";
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
