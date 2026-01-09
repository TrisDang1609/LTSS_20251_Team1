/**
 * CUDA-Accelerated Video File SLAM
 *
 * This file is part of ORB-SLAM3 CUDA Extension
 *
 * Process video files (MP4, AVI, etc.) through the CUDA-accelerated SLAM pipeline.
 * Optimized for NVIDIA RTX 4060 (Ada Lovelace, SM 8.9).
 *
 * Features:
 * - GPU-accelerated ORB feature extraction and matching
 * - Video playback speed control (real-time, fast, slow)
 * - Comprehensive performance profiling and statistics
 * - Support for various video codecs via OpenCV
 * - Frame-by-frame or continuous processing modes
 * - Trajectory saving and visualization
 *
 * Usage: ./mono_video_cuda path_to_vocabulary path_to_settings path_to_video [options]
 *
 * Author: CUDA Optimization Team
 * Target: NVIDIA RTX 4060, CUDA 13.0, Ubuntu 24.04
 */

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <cuda_runtime.h>

#include <System.h>
#include "cuda/GpuPipeline.h"
#include "cuda/CudaUtils.h"
#include "cuda/CudaMemoryManager.h"

using namespace std;
using namespace ORB_SLAM3;

// ============================================================================
// Configuration Constants
// ============================================================================
constexpr int PROGRESS_UPDATE_INTERVAL = 50; // Update progress every N frames
constexpr bool ENABLE_GPU_WARMUP = true;     // Warmup GPU before processing

// ============================================================================
// Playback Mode Enumeration
// ============================================================================
enum class PlaybackMode
{
    REALTIME, // Process at video's original FPS
    FAST,     // Process as fast as possible
    STEP      // Step through frame by frame
};

// ============================================================================
// GPU Device Information
// ============================================================================
void printGpuInfo()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        cerr << "ERROR: No CUDA devices found!" << endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << endl;
    cout << "╔══════════════════════════════════════════════════════╗" << endl;
    cout << "║         CUDA GPU Device Information                  ║" << endl;
    cout << "╠══════════════════════════════════════════════════════╣" << endl;
    cout << "║  Device: " << left << setw(43) << prop.name << "║" << endl;
    cout << "║  Compute Capability: " << prop.major << "." << prop.minor;
    cout << setw(32) << " " << "║" << endl;
    cout << "║  Streaming Multiprocessors: " << setw(24) << prop.multiProcessorCount << "║" << endl;
    cout << "║  Global Memory: " << setw(29) << to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB" << "║" << endl;
    cout << "║  Memory Bandwidth: " << setw(26) << to_string(prop.memoryBusWidth) + "-bit" << "║" << endl;
    cout << "╚══════════════════════════════════════════════════════╝" << endl;
    cout << endl;
}

// ============================================================================
// Video Information
// ============================================================================
struct VideoInfo
{
    string path;
    int width;
    int height;
    int totalFrames;
    double fps;
    double duration;
    string codec;

    void print() const
    {
        cout << "╔══════════════════════════════════════════════════════╗" << endl;
        cout << "║              Video Information                       ║" << endl;
        cout << "╠══════════════════════════════════════════════════════╣" << endl;
        cout << "║  Path: " << left << setw(45) << path.substr(path.find_last_of("/\\") + 1) << "║" << endl;
        cout << "║  Resolution: " << setw(39) << (to_string(width) + " x " + to_string(height)) << "║" << endl;
        cout << "║  Total Frames: " << setw(37) << totalFrames << "║" << endl;
        cout << "║  FPS: " << setw(46) << fixed << setprecision(2) << fps << "║" << endl;
        cout << "║  Duration: " << setw(36) << (to_string(static_cast<int>(duration)) + " seconds") << "║" << endl;
        cout << "║  Codec: " << setw(44) << codec << "║" << endl;
        cout << "╚══════════════════════════════════════════════════════╝" << endl;
    }
};

VideoInfo getVideoInfo(cv::VideoCapture &cap, const string &path)
{
    VideoInfo info;
    info.path = path;
    info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    info.fps = cap.get(cv::CAP_PROP_FPS);
    info.duration = (info.fps > 0) ? info.totalFrames / info.fps : 0;

    // Decode fourcc
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    info.codec = string(1, fourcc & 0xFF) +
                 string(1, (fourcc >> 8) & 0xFF) +
                 string(1, (fourcc >> 16) & 0xFF) +
                 string(1, (fourcc >> 24) & 0xFF);

    return info;
}

// ============================================================================
// Performance Statistics
// ============================================================================
class PerformanceStats
{
public:
    void addFrame(double trackingTimeMs, int numKeypoints, int trackingState)
    {
        trackingTimes_.push_back(trackingTimeMs);
        keypointCounts_.push_back(numKeypoints);
        trackingStates_.push_back(trackingState);
        totalFrames_++;
    }

    void reset()
    {
        trackingTimes_.clear();
        keypointCounts_.clear();
        trackingStates_.clear();
        totalFrames_ = 0;
    }

    int getTotalFrames() const { return totalFrames_; }

    double getMeanTrackingTime() const
    {
        if (trackingTimes_.empty())
            return 0;
        return accumulate(trackingTimes_.begin(), trackingTimes_.end(), 0.0) / trackingTimes_.size();
    }

    double getStdDevTrackingTime() const
    {
        if (trackingTimes_.size() < 2)
            return 0;
        double mean = getMeanTrackingTime();
        double sq_sum = 0;
        for (double t : trackingTimes_)
            sq_sum += (t - mean) * (t - mean);
        return sqrt(sq_sum / (trackingTimes_.size() - 1));
    }

    double getMinTrackingTime() const
    {
        if (trackingTimes_.empty())
            return 0;
        return *min_element(trackingTimes_.begin(), trackingTimes_.end());
    }

    double getMaxTrackingTime() const
    {
        if (trackingTimes_.empty())
            return 0;
        return *max_element(trackingTimes_.begin(), trackingTimes_.end());
    }

    double getPercentile(double p) const
    {
        if (trackingTimes_.empty())
            return 0;
        vector<double> sorted = trackingTimes_;
        sort(sorted.begin(), sorted.end());
        int idx = static_cast<int>(p * (sorted.size() - 1));
        return sorted[idx];
    }

    double getAverageFps() const
    {
        double avgTime = getMeanTrackingTime();
        return avgTime > 0 ? 1000.0 / avgTime : 0;
    }

    double getAverageKeypoints() const
    {
        if (keypointCounts_.empty())
            return 0;
        return static_cast<double>(accumulate(keypointCounts_.begin(), keypointCounts_.end(), 0)) / keypointCounts_.size();
    }

    int getTrackingLostCount() const
    {
        int count = 0;
        for (int state : trackingStates_)
        {
            if (state == 3) // Lost state
                count++;
        }
        return count;
    }

    double getTrackingSuccessRate() const
    {
        if (trackingStates_.empty())
            return 0;
        int successCount = 0;
        for (int state : trackingStates_)
        {
            if (state == 2) // OK state
                successCount++;
        }
        return 100.0 * successCount / trackingStates_.size();
    }

    void printSummary() const
    {
        cout << endl;
        cout << "╔══════════════════════════════════════════════════════╗" << endl;
        cout << "║             Performance Statistics                   ║" << endl;
        cout << "╠══════════════════════════════════════════════════════╣" << endl;
        cout << fixed << setprecision(2);
        cout << "║  Frames Processed: " << setw(32) << totalFrames_ << "║" << endl;
        cout << "║                                                      ║" << endl;
        cout << "║  Tracking Time (ms):                                 ║" << endl;
        cout << "║    Mean:    " << setw(40) << getMeanTrackingTime() << "║" << endl;
        cout << "║    Std Dev: " << setw(40) << getStdDevTrackingTime() << "║" << endl;
        cout << "║    Min:     " << setw(40) << getMinTrackingTime() << "║" << endl;
        cout << "║    Max:     " << setw(40) << getMaxTrackingTime() << "║" << endl;
        cout << "║    P50:     " << setw(40) << getPercentile(0.50) << "║" << endl;
        cout << "║    P95:     " << setw(40) << getPercentile(0.95) << "║" << endl;
        cout << "║    P99:     " << setw(40) << getPercentile(0.99) << "║" << endl;
        cout << "║                                                      ║" << endl;
        cout << "║  Throughput:                                         ║" << endl;
        cout << "║    Average FPS: " << setw(36) << getAverageFps() << "║" << endl;
        cout << "║                                                      ║" << endl;
        cout << "║  Tracking Quality:                                   ║" << endl;
        cout << "║    Avg Keypoints: " << setw(33) << getAverageKeypoints() << "║" << endl;
        cout << "║    Success Rate:  " << setw(31) << (to_string(getTrackingSuccessRate()) + " %") << "║" << endl;
        cout << "║    Tracking Lost: " << setw(30) << (to_string(getTrackingLostCount()) + " times") << "║" << endl;
        cout << "╚══════════════════════════════════════════════════════╝" << endl;
        cout << endl;
    }

private:
    vector<double> trackingTimes_;
    vector<int> keypointCounts_;
    vector<int> trackingStates_;
    int totalFrames_ = 0;
};

// ============================================================================
// Progress Bar
// ============================================================================
void printProgressBar(int current, int total, double fps, const string &status)
{
    const int barWidth = 40;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    cout << "\r[";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            cout << "█";
        else if (i == pos)
            cout << "▓";
        else
            cout << "░";
    }
    cout << "] " << fixed << setprecision(1);
    cout << setw(5) << (progress * 100.0) << "% | ";
    cout << "Frame: " << setw(6) << current << "/" << total << " | ";
    cout << "FPS: " << setw(6) << fps << " | ";
    cout << status << "     " << flush;
}

// ============================================================================
// Command Line Parser
// ============================================================================
struct CommandLineArgs
{
    string vocabPath;
    string settingsPath;
    string videoPath;
    PlaybackMode mode = PlaybackMode::FAST;
    bool saveTrajectory = true;
    int startFrame = 0;
    int endFrame = -1; // -1 means process all
    bool enableViewer = true;

    static void printUsage()
    {
        cout << "Usage: ./mono_video_cuda vocabulary settings video [options]" << endl;
        cout << endl;
        cout << "Required arguments:" << endl;
        cout << "  vocabulary          Path to ORB vocabulary file" << endl;
        cout << "  settings            Path to camera settings YAML file" << endl;
        cout << "  video               Path to input video file (.mp4, .avi, etc.)" << endl;
        cout << endl;
        cout << "Optional arguments:" << endl;
        cout << "  --mode MODE         Playback mode: realtime, fast, step (default: fast)" << endl;
        cout << "  --start N           Start processing from frame N (default: 0)" << endl;
        cout << "  --end N             Stop processing at frame N (default: all frames)" << endl;
        cout << "  --no-viewer         Disable the 3D viewer" << endl;
        cout << "  --no-save           Don't save trajectory at the end" << endl;
        cout << endl;
        cout << "Example:" << endl;
        cout << "  ./mono_video_cuda Vocabulary/ORBvoc.txt Examples/Monocular/webcam_cuda.yaml myvideo.mp4 --mode fast" << endl;
    }
};

bool parseArgs(int argc, char **argv, CommandLineArgs &args)
{
    if (argc < 4)
    {
        CommandLineArgs::printUsage();
        return false;
    }

    args.vocabPath = argv[1];
    args.settingsPath = argv[2];
    args.videoPath = argv[3];

    for (int i = 4; i < argc; ++i)
    {
        string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc)
        {
            string mode = argv[++i];
            if (mode == "realtime")
                args.mode = PlaybackMode::REALTIME;
            else if (mode == "fast")
                args.mode = PlaybackMode::FAST;
            else if (mode == "step")
                args.mode = PlaybackMode::STEP;
            else
            {
                cerr << "Unknown mode: " << mode << endl;
                return false;
            }
        }
        else if (arg == "--start" && i + 1 < argc)
        {
            args.startFrame = atoi(argv[++i]);
        }
        else if (arg == "--end" && i + 1 < argc)
        {
            args.endFrame = atoi(argv[++i]);
        }
        else if (arg == "--no-viewer")
        {
            args.enableViewer = false;
        }
        else if (arg == "--no-save")
        {
            args.saveTrajectory = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            CommandLineArgs::printUsage();
            return false;
        }
    }

    return true;
}

// ============================================================================
// GPU Warmup
// ============================================================================
void warmupGpu(cv::VideoCapture &cap, ORB_SLAM3::System &SLAM, int numWarmupFrames = 10)
{
    cout << "Warming up GPU with " << numWarmupFrames << " frames..." << endl;

    cv::Mat frame;
    double savedPos = cap.get(cv::CAP_PROP_POS_FRAMES);

    for (int i = 0; i < numWarmupFrames; ++i)
    {
        cap >> frame;
        if (frame.empty())
            break;

        double timestamp = i * 0.033;
        SLAM.TrackMonocular(frame, timestamp);
    }

    // Reset to beginning
    cap.set(cv::CAP_PROP_POS_FRAMES, savedPos);

    // Reset SLAM for fresh start
    SLAM.Reset();

    // Ensure GPU is synced
    cudaDeviceSynchronize();

    cout << "GPU warmup complete!" << endl;
}

// ============================================================================
// Main Application
// ============================================================================
int main(int argc, char **argv)
{
    cout << endl;
    cout << "╔══════════════════════════════════════════════════════╗" << endl;
    cout << "║   ORB-SLAM3 CUDA Video Processing Demo               ║" << endl;
    cout << "║   Optimized for NVIDIA RTX 4060                      ║" << endl;
    cout << "╚══════════════════════════════════════════════════════╝" << endl;
    cout << endl;

    // Parse command line arguments
    CommandLineArgs args;
    if (!parseArgs(argc, argv, args))
    {
        return 1;
    }

    // Print GPU information
    printGpuInfo();

    // Initialize CUDA
    cout << "Initializing CUDA..." << endl;
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        cerr << "ERROR: CUDA initialization failed: " << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }
    cout << "CUDA initialized successfully!" << endl;

    // Open video file
    cout << "\nOpening video file..." << endl;
    cv::VideoCapture cap(args.videoPath);

    if (!cap.isOpened())
    {
        cerr << "ERROR: Cannot open video file: " << args.videoPath << endl;
        cerr << "Make sure the file exists and is a valid video format." << endl;
        return 1;
    }

    // Get video information
    VideoInfo videoInfo = getVideoInfo(cap, args.videoPath);
    videoInfo.print();

    // Validate frame range
    if (args.startFrame < 0)
        args.startFrame = 0;
    if (args.startFrame >= videoInfo.totalFrames)
    {
        cerr << "ERROR: Start frame exceeds total frames" << endl;
        return 1;
    }
    if (args.endFrame < 0 || args.endFrame > videoInfo.totalFrames)
        args.endFrame = videoInfo.totalFrames;
    if (args.endFrame <= args.startFrame)
    {
        cerr << "ERROR: End frame must be greater than start frame" << endl;
        return 1;
    }

    int framesToProcess = args.endFrame - args.startFrame;

    cout << "\nProcessing Configuration:" << endl;
    cout << "  Start Frame: " << args.startFrame << endl;
    cout << "  End Frame: " << args.endFrame << endl;
    cout << "  Frames to Process: " << framesToProcess << endl;
    cout << "  Mode: ";
    switch (args.mode)
    {
    case PlaybackMode::REALTIME:
        cout << "Real-time" << endl;
        break;
    case PlaybackMode::FAST:
        cout << "Fast (maximum speed)" << endl;
        break;
    case PlaybackMode::STEP:
        cout << "Step-by-step" << endl;
        break;
    }

    // Create SLAM system
    cout << "\nInitializing ORB-SLAM3 system..." << endl;
    cout << "  Vocabulary: " << args.vocabPath << endl;
    cout << "  Settings: " << args.settingsPath << endl;

    ORB_SLAM3::System SLAM(args.vocabPath, args.settingsPath,
                           ORB_SLAM3::System::MONOCULAR, args.enableViewer);
    float imageScale = SLAM.GetImageScale();

    // Seek to start frame if needed
    if (args.startFrame > 0)
    {
        cout << "Seeking to frame " << args.startFrame << "..." << endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, args.startFrame);
    }

    // GPU Warmup
    if (ENABLE_GPU_WARMUP)
    {
        warmupGpu(cap, SLAM, 10);
        // Re-seek after warmup if needed
        if (args.startFrame > 0)
        {
            cap.set(cv::CAP_PROP_POS_FRAMES, args.startFrame);
        }
    }

    cout << endl;
    cout << "╔══════════════════════════════════════════════════════╗" << endl;
    cout << "║   Processing Video - CUDA Accelerated SLAM          ║" << endl;
    cout << "╠══════════════════════════════════════════════════════╣" << endl;
    cout << "║   Controls (in step mode):                          ║" << endl;
    cout << "║     [SPACE] - Next frame                            ║" << endl;
    cout << "║     [Q/ESC] - Quit                                  ║" << endl;
    cout << "║     [S]     - Save trajectory                       ║" << endl;
    cout << "╚══════════════════════════════════════════════════════╝" << endl;
    cout << endl;

    // Performance tracking
    PerformanceStats stats;
    auto startTime = chrono::steady_clock::now();

    // Frame timing for real-time playback
    double frameIntervalMs = 1000.0 / videoInfo.fps;

    // Main processing loop
    cv::Mat frame;
    int currentFrame = args.startFrame;
    int processedFrames = 0;

    while (currentFrame < args.endFrame)
    {
        auto frameStartTime = chrono::steady_clock::now();

        // Read frame
        cap >> frame;
        if (frame.empty())
        {
            cout << "\nEnd of video reached." << endl;
            break;
        }

        // Calculate timestamp
        double timestamp = currentFrame / videoInfo.fps;

        // Apply image scaling if needed
        if (imageScale != 1.f)
        {
            int width = static_cast<int>(frame.cols * imageScale);
            int height = static_cast<int>(frame.rows * imageScale);
            cv::resize(frame, frame, cv::Size(width, height));
        }

        // Track time
        auto t1 = chrono::steady_clock::now();

        // Process frame through SLAM
        Sophus::SE3f pose = SLAM.TrackMonocular(frame, timestamp);

        auto t2 = chrono::steady_clock::now();
        double trackingTimeMs = chrono::duration<double, milli>(t2 - t1).count();

        // Update statistics
        int numTrackedPoints = SLAM.GetTrackedMapPoints().size();
        int trackingState = SLAM.GetTrackingState();
        stats.addFrame(trackingTimeMs, numTrackedPoints, trackingState);
        processedFrames++;
        currentFrame++;

        // Get status string based on tracking state
        string status;
        switch (trackingState)
        {
        case 0:
            status = "INIT";
            break;
        case 1:
            status = "NOT_READY";
            break;
        case 2:
            status = "OK";
            break;
        case 3:
            status = "LOST";
            break;
        default:
            status = "UNKNOWN";
        }

        // Update progress
        if (processedFrames % PROGRESS_UPDATE_INTERVAL == 0 || processedFrames == framesToProcess)
        {
            printProgressBar(processedFrames, framesToProcess, stats.getAverageFps(), status);
        }

        // Handle playback mode
        if (args.mode == PlaybackMode::REALTIME)
        {
            // Wait to maintain real-time playback
            auto frameEndTime = chrono::steady_clock::now();
            double elapsedMs = chrono::duration<double, milli>(frameEndTime - frameStartTime).count();
            double waitMs = frameIntervalMs - elapsedMs;
            if (waitMs > 0)
            {
                this_thread::sleep_for(chrono::milliseconds(static_cast<int>(waitMs)));
            }
        }
        else if (args.mode == PlaybackMode::STEP)
        {
            // Wait for user input
            char key = static_cast<char>(cv::waitKey(0));
            if (key == 'q' || key == 'Q' || key == 27)
            {
                cout << "\n\nQuit requested by user." << endl;
                break;
            }
            else if (key == 's' || key == 'S')
            {
                cout << "\n\nSaving trajectory..." << endl;
                SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_video_cuda_intermediate.txt");
                cout << "Trajectory saved!" << endl;
            }
        }

        // Check for quit in non-step modes
        if (args.mode != PlaybackMode::STEP)
        {
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 'Q' || key == 27)
            {
                cout << "\n\nQuit requested by user." << endl;
                break;
            }
        }
    }

    // Calculate total runtime
    auto endTime = chrono::steady_clock::now();
    double totalTimeSeconds = chrono::duration<double>(endTime - startTime).count();

    cout << endl
         << endl;
    cout << "Processing complete!" << endl;
    cout << "Total runtime: " << fixed << setprecision(2) << totalTimeSeconds << " seconds" << endl;
    cout << "Effective throughput: " << (processedFrames / totalTimeSeconds) << " FPS" << endl;

    // Print performance statistics
    stats.printSummary();

    // Shutdown SLAM
    cout << "Shutting down SLAM system..." << endl;
    SLAM.Shutdown();

    // Save trajectory
    if (args.saveTrajectory)
    {
        string trajectoryFile = "KeyFrameTrajectory_video_cuda.txt";
        cout << "Saving trajectory to " << trajectoryFile << "..." << endl;
        SLAM.SaveKeyFrameTrajectoryTUM(trajectoryFile);
        cout << "Trajectory saved!" << endl;
    }

    // Cleanup
    cap.release();
    cudaDeviceReset();

    cout << endl;
    cout << "╔══════════════════════════════════════════════════════╗" << endl;
    cout << "║   Video Processing Session Complete!                 ║" << endl;
    cout << "╚══════════════════════════════════════════════════════╝" << endl;
    cout << endl;

    return 0;
}
